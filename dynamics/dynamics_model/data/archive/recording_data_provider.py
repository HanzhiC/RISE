# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import os
import numpy as np
from loguru import logger

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import (
    ClosedLoopTrajectoryPose,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)
from projectaria_tools.core.stream_id import StreamId
from torchcodec.decoders import VideoDecoder
from typing import List
from easydict import EasyDict as edict


class AriaStream(Enum):
    camera_slam_left = "1201-1"
    camera_slam_right = "1201-2"
    camera_rgb = "214-1"
    imu_right = "1202-1"
    imu_left = "1202-2"


class RecordingPathProvider:
    """
    \brief This class will not check of input recording path is valid
    """

    def __init__(self, vrs_path: str, mps_dir: str = None):
        self.vrs_path: str = vrs_path
        self.mps_dir: str = mps_dir

    @property
    def data_vrsfile(self):
        return self.vrs_path

    @property
    def mps_path(self):
        if os.path.isdir(self.mps_dir):
            return MpsDataPathsProvider(self.mps_dir)
        else:
            return None

    @property
    def points_npz_cache(self):
        return os.path.join(self.mps_dir, "semidense_points_cached.npz")


class RecordingDataProvider(RecordingPathProvider):
    def __init__(self, vrs_path: str, mps_dir: str = None, vid_path: str = None):
        super().__init__(vrs_path, mps_dir)

        self._vrs_dp = None
        self._mps_dp = None
        self._video_decoder = None

        # load vrs
        self._vrs_dp = data_provider.create_vrs_data_provider(self.vrs_path)

        # load mps
        if self.mps_dir is not None:
            self._mps_dp = MpsDataProvider(self.mps_path.get_data_paths())

        # load video
        if vid_path is not None:
            self._video_decoder = VideoDecoder(vid_path)

    @property
    def vrs_dp(self):
        return self._vrs_dp

    @property
    def mps_dp(self):
        return self._mps_dp

    @property
    def video_decoder(self):
        return self._video_decoder

    def get_global_timespan_ns(self):
        if self.vrs_dp is None:
            raise RuntimeError(
                f"require {self.data_vrsfile=} "
            )

        t_start = self.vrs_dp.get_first_time_ns_all_streams(
            TimeDomain.TIME_CODE)
        t_end = self.vrs_dp.get_last_time_ns_all_streams(TimeDomain.TIME_CODE)
        return t_start, t_end

    @property
    def has_pointcloud(self):
        if self.mps_dp is None or not self.mps_dp.has_semidense_point_cloud():
            return False
        else:
            return True

    def get_pointcloud(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int = 50_000,
        cache_to_npz: bool = False,
    ):
        assert self.has_pointcloud, "recording has no point cloud"
        points = self.mps_dp.get_semidense_point_cloud()

        points = mps.utils.filter_points_from_confidence(
            raw_points=points, threshold_dep=th_dep, threshold_invdep=th_invdep
        )
        points = mps.utils.filter_points_from_count(
            raw_points=points, max_point_count=max_point_count
        )

        points = np.array([x.position_world for x in points])

        if cache_to_npz:
            np.savez(
                self.points_npz_cache,
                points=points,
                threshold_dep=th_dep,
                threshold_invdep=th_invdep,
                max_point_count=max_point_count,
            )
        return points

    def get_pointcloud_cached(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int = 50_000,
    ):
        assert self.has_pointcloud, "recording has no point cloud"
        if os.path.isfile(self.points_npz_cache):
            logger.info(
                f"load cached point cloud from {self.points_npz_cache}")
            return np.load(self.points_npz_cache)["points"]

        return self.get_pointcloud(cache_to_npz=True)

    @property
    def has_vrs(self):
        return self.vrs_dp is not None

    @property
    def has_video(self):
        return self.video_decoder is not None

    @property
    def has_rgb(self):
        return (self.has_vrs and self.vrs_dp.check_stream_is_active(StreamId("214-1"))) or self.has_video

    def get_rgb_image(
        self, t_ns: int,
        time_domain: TimeDomain = TimeDomain.DEVICE_TIME,
        use_video: bool = False,
        as_array: bool = True,
    ):
        assert self.has_rgb, "recording has no rgb video"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if use_video:
            assert self.has_video, "recording has no rgb video"
            t_s = t_ns / 10 ** 9
            frame = self.video_decoder.get_frame_played_at(seconds=t_s)
            image_data = frame.data.numpy().transpose(1, 2, 0)
            return image_data, frame, 0

        else:
            # Use VRS
            if time_domain == TimeDomain.TIME_CODE:
                t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                    timecode_time_ns=t_ns
                )
            else:
                t_ns_device = t_ns

            image_data, image_meta = self.vrs_dp.get_image_data_by_time_ns(
                StreamId("214-1"),
                time_ns=t_ns_device,
                time_domain=TimeDomain.DEVICE_TIME,
                time_query_options=TimeQueryOptions.CLOSEST,
            )
            t_diff = t_ns_device - image_meta.capture_timestamp_ns

            if as_array:
                return image_data.to_numpy_array(), image_meta, t_diff
            else:
                return image_data, image_meta, t_diff

    def get_rgb_images(
        self, t_ns_list: List[int],
        time_domain: TimeDomain = TimeDomain.DEVICE_TIME,
        use_video: bool = False,
        as_array: bool = True,
    ):
        assert self.has_rgb, "recording has no rgb video"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if use_video:
            assert self.has_video, "recording has no rgb video"
            t_s_list = [t_ns / 10 ** 9 for t_ns in t_ns_list]
            frames = self.video_decoder.get_frames_played_at(seconds=t_s_list)
            image_data_list = frames.data.numpy().transpose(
                0, 2, 3, 1)  # [N, H, W, 3]
            t_diffs = np.zeros(len(t_ns_list))
            return image_data_list, frames, t_diffs

        else:
            image_data_list = []
            image_meta_list = []
            t_diff_list = []
            for t_ns in t_ns_list:
                image_data, image_meta, t_diff = self.get_rgb_image(
                    t_ns, time_domain, as_array, use_video)
                image_data_list.append(image_data)
                image_meta_list.append(image_meta)
                t_diff_list.append(t_diff)
            if as_array:
                image_data_list = np.stack(image_data_list, axis=0)
            t_diff_list = np.array(t_diff_list)
            return image_data_list, image_meta_list, t_diff_list

    @property
    def has_slam_left_image(self):
        return self.has_vrs and self.vrs_dp.check_stream_is_active(StreamId("1201-1"))

    def get_slam_left_image(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.DEVICE_TIME, as_array: bool = False
    ):
        assert self.has_slam_left_image, "recording has no slam left image"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        else:
            t_ns_device = t_ns

        image_data, image_meta = self.vrs_dp.get_image_data_by_time_ns(
            StreamId("1201-1"),
            time_ns=t_ns_device,
            time_domain=TimeDomain.DEVICE_TIME,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        t_diff = t_ns_device - image_meta.capture_timestamp_ns

        if as_array:
            return image_data.to_numpy_array(), image_meta, t_diff
        else:
            return image_data, image_meta, t_diff

    @property
    def has_slam_right_image(self):
        return self.has_vrs and self.vrs_dp.check_stream_is_active(StreamId("1201-2"))

    def get_slam_right_image(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.DEVICE_TIME, as_array: bool = False
    ):
        assert self.has_slam_right_image, "recording has no slam right image"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        else:
            t_ns_device = t_ns

        image_data, image_meta = self.vrs_dp.get_image_data_by_time_ns(
            StreamId("1201-2"),
            time_ns=t_ns_device,
            time_domain=TimeDomain.DEVICE_TIME,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        t_diff = t_ns_device - image_meta.capture_timestamp_ns

        if as_array:
            return image_data.to_numpy_array(), image_meta, t_diff
        else:
            return image_data, image_meta, t_diff

    @property
    def has_pose(self):
        if self.mps_dp is None or not self.mps_dp.has_closed_loop_poses():
            return False
        else:
            return True

    @property
    def has_wrist_and_palm_pose(self):
        if self.mps_dp is None or not self.mps_dp.has_wrist_and_palm_poses():
            return False
        else:
            return True

    @property
    def has_eye_gaze(self):
        if self.mps_dp is None or not self.mps_dp.has_general_eyegaze():
            return False
        else:
            return True

    def get_wrist_and_palm_pose(self, t_ns: int, time_domain: TimeDomain = TimeDomain.DEVICE_TIME, dict_format: bool = True):
        assert self.has_wrist_and_palm_pose, "recording has no wrist and palm pose"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        else:
            t_ns_device = t_ns
        wrist_and_palm_pose = self.mps_dp.get_wrist_and_palm_pose(
            t_ns_device, TimeQueryOptions.CLOSEST)

        t_diff = wrist_and_palm_pose.tracking_timestamp.total_seconds() * 1e9 - \
            t_ns_device
        if dict_format:
            results = {}
            if wrist_and_palm_pose.left_hand is not None:
                results["left"] = {}
                results["left"]["wrist"] = wrist_and_palm_pose.left_hand.wrist_position_device
                results["left"]["palm"] = wrist_and_palm_pose.left_hand.palm_position_device
                results["left"]["confidence"] = wrist_and_palm_pose.left_hand.confidence
            else:
                results["left"] = None
            if wrist_and_palm_pose.right_hand is not None:
                results["right"] = {}
                results["right"]["wrist"] = wrist_and_palm_pose.right_hand.wrist_position_device
                results["right"]["palm"] = wrist_and_palm_pose.right_hand.palm_position_device
                results["right"]["confidence"] = wrist_and_palm_pose.right_hand.confidence
            else:
                results["right"] = None
            results = edict(results)
            return results, t_diff
        else:
            return wrist_and_palm_pose, t_diff

    def get_eye_gaze(self, t_ns: int, time_domain: TimeDomain = TimeDomain.DEVICE_TIME):
        assert self.has_eye_gaze, "recording has no eye gaze"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        else:
            t_ns_device = t_ns

        eye_gaze = self.mps_dp.get_general_eyegaze(
            t_ns_device, TimeQueryOptions.CLOSEST)
        t_diff = eye_gaze.tracking_timestamp.total_seconds() * 1e9 - t_ns_device
        return eye_gaze, t_diff

    def get_pose(
        self, t_ns: int, time_domain: TimeDomain, martrix_format: bool = True
    ):
        t_ns = int(t_ns)
        assert self.has_pose, "recording has no closed loop trajectory"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            assert self.vrs_dp, "require vrs for time domain mapping"
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )

        else:
            t_ns_device = t_ns
        pose = self.mps_dp.get_closed_loop_pose(
            t_ns_device, TimeQueryOptions.CLOSEST)
        t_diff = pose.tracking_timestamp.total_seconds() * 1e9 - t_ns_device
        return pose.transform_world_device.to_matrix(), t_diff

    def sample_trajectory_world_device(self, sample_fps: float = 1):
        assert self.has_pose, "recording has no closed loop trajectory"
        assert self.has_vrs, "current implementation assume vrs is loaded."
        t_start, t_end = self.get_global_timespan_ns()
        t_start = self.vrs_dp.convert_from_timecode_to_device_time_ns(t_start)
        t_end = self.vrs_dp.convert_from_timecode_to_device_time_ns(t_end)

        dt = int(1e9 / sample_fps)
        traj_world_device = []
        for t_ns in range(t_start, t_end, dt):
            pose = self.mps_dp.get_closed_loop_pose(
                t_ns, TimeQueryOptions.CLOSEST)
            traj_world_device.append(
                pose.transform_world_device.to_matrix().astype(np.float32)
            )

        traj_world_device = np.stack(traj_world_device, axis=0)
        return traj_world_device

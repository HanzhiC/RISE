# import os
# import numpy as np
# import cv2
# import pandas as pd
# import pickle
# import click
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import cv2
# from projectaria_tools.core import calibration
# import open3d as o3d
# from scipy.spatial.transform import Rotation as R
# import os
# from transformations import rotation_matrix
# import utils.dataset_utils as DataUtils
# import utils.aria_utils as AriaUtils
# from projectaria_tools.core import mps
# from projectaria_tools.core.data_provider import (
#     VrsDataProvider,
#     create_vrs_data_provider,
# )
# from projectaria_tools.core.mps.utils import (
#     filter_points_from_confidence,
#     filter_points_from_count,
# )
# from projectaria_tools.core.sophus import SE3
# from projectaria_tools.core.sensor_data import TimeDomain

# from dataclasses import dataclass
# from typing import List
# import pandas as pd
# from torch.utils.data import Dataset

# NARRATION_DIR = "./data/annotations/hd-epic-annotations/narrations-and-action-segments"
# DATA_DIR = "/home/wiss/chenh/storage/group/dataset_mirrors/01_incoming/hd-epickitchens/HD-EPIC/"


# class HD_EPIC_Dataset(Dataset):
#     def __init__(
#         self,
#         data_dir=DATA_DIR,
#         video_id="P07-20240529-102652",
#         hand_threshold=0.0,
#         fps=30,
#         focal=280,
#         resolution=(512, 512),
#         calibrate_slam=False,
#         rectify_images=True,
#     ):
#         # Set the pre-defined resolution
#         self.fps = fps
#         self.camera_label = "camera-rgb"
#         self.height, self.width, self.focal = resolution[0], resolution[1], focal
#         self.rectify_images = rectify_images
#         # Set the data directory
#         self.data_dir = data_dir
#         self.narration_dir = NARRATION_DIR

#         # Set the video id
#         self.video_id = video_id
#         self.participant_id = video_id.split("-")[0]

#         # Load the annotations
#         narrations_fpath = os.path.join(self.narration_dir, "HD_EPIC_Narrations.pkl")
#         noun_classes_fpath = os.path.join(self.narration_dir, "HD_EPIC_noun_class.csv")
#         verb_classes_fpath = os.path.join(self.narration_dir, "HD_EPIC_verb_class.csv")
#         with open(narrations_fpath, "rb") as f:
#             annotations = pickle.load(f)
#         self.annotations = pd.DataFrame(annotations)
#         self.annotations = self.annotations[
#             self.annotations["video_id"] == self.video_id
#         ]

#         # Set the other directories
#         self.mps_dir = os.path.join(
#             self.data_dir, "SLAM-and-Gaze", self.participant_id, "SLAM/multi/0"
#         )
#         self.video_timestamps_fname = os.path.join(
#             self.data_dir,
#             "Videos",
#             self.participant_id,
#             f"{self.video_id}_mp4_to_vrs_time_ns.csv",
#         )
#         self.video_fname = os.path.join(
#             self.data_dir, "Videos", self.participant_id, f"{self.video_id}.mp4"
#         )
#         self.vrs_fname = os.path.join(
#             self.data_dir, "VRS", self.participant_id, f"{self.video_id}_anonymized.vrs"
#         )
#         self.recording_data_provider = RecordingDataProvider(
#             self.vrs_fname, self.mps_dir, self.video_fname
#         )
#         # Initialize the timestamp mapping and camera calibration
#         self._get_timestamp_mapping()
#         self._get_slam_metainfos()
#         self._filter_annotations()

#         # Some hyperparameters
#         self.hand_threshold = hand_threshold
#         self.calibrate_slam = calibrate_slam
#         self.time_stamp_aligned_keys = [
#             "rgb_images",
#             "timestamps_mp4",
#             "timestamps_vrs",
#             "slam_poses",
#             "sparse_depths",
#             "wrist_traj_left",
#             "wrist_traj_right",
#             "palm_traj_left",
#             "palm_traj_right",
#             "hand_traj_left_valid",
#             "hand_traj_right_valid",
#             "hand_traj_left_visible",
#             "hand_traj_right_visible",
#             "hand_traj_vrs_timestamps_left",
#             "hand_traj_vrs_timestamps_right",
#         ]

#     def _get_timestamp_mapping(self):
#         # Get the timesstamps with slam poses
#         video_timestamps = pd.read_csv(self.video_timestamps_fname)
#         self.video_timestamps_vrs = np.array(
#             video_timestamps["vrs_device_time_ns"].values
#         )
#         self.video_timestamps_mp4 = np.array(video_timestamps["mp4_time_ns"].values)
#         self.timestamps_mp4_to_vrs = {
#             mp4_time_ns: vrs_time_ns
#             for mp4_time_ns, vrs_time_ns in zip(
#                 self.video_timestamps_mp4, self.video_timestamps_vrs
#             )
#         }
#         self.timestamps_vrs_to_mp4 = {
#             vrs_time_ns: mp4_time_ns
#             for mp4_time_ns, vrs_time_ns in zip(
#                 self.video_timestamps_mp4, self.video_timestamps_vrs
#             )
#         }

#     def _get_slam_metainfos(self):
#         # Read the data provider
#         self.calib = self.recording_data_provider.vrs_dp.get_device_calibration().get_camera_calib(
#             self.camera_label
#         )
#         self.pinhole = calibration.get_linear_camera_calibration(
#             self.height,
#             self.width,
#             self.focal,
#             self.camera_label,
#             self.calib.get_transform_device_camera(),
#         )
#         self.T_head_cam = self.calib.get_transform_device_camera().to_matrix()
#         self.slam_points = self.recording_data_provider.get_pointcloud_cached()
#         self.hand_points_approx = np.asarray(
#             o3d.geometry.TriangleMesh.create_sphere(radius=0.05).vertices
#         )
#         self.hand_points_approx = (
#             self.hand_points_approx - self.hand_points_approx.mean(axis=0)
#         )  # Center the points

#     def _filter_annotations(self):
#         # Only select the annotation with one hand
#         self.annotations = self.annotations[self.annotations["hands"].apply(len) == 1]
#         self.annotations = self.annotations[
#             self.annotations["hands"].apply(lambda x: "both hands" not in x)
#         ]

#     def _get_rgb_images(self, idx, data):
#         # Get the start and end timestamp of the annotation
#         start_timestamp_i = self.annotations["start_timestamp"].iloc[idx]
#         end_timestamp_i = self.annotations["end_timestamp"].iloc[idx]
#         print(f"Start timestamp: {start_timestamp_i}, End timestamp: {end_timestamp_i}")
#         duration = end_timestamp_i - start_timestamp_i
#         end_timestamp_i = end_timestamp_i - duration * 0.2
#         # Acquire the nearest mp4 timestamp and corresponding vrs timestamp
#         # Backward 2 seconds to avoid the initial motion
#         start_timestamp_i = start_timestamp_i
#         start_timestamp_i_after = self.video_timestamps_mp4[
#             self.video_timestamps_mp4 >= (start_timestamp_i) * 1e9
#         ]
#         end_timestamp_i_before = self.video_timestamps_mp4[
#             self.video_timestamps_mp4 <= (end_timestamp_i) * 1e9
#         ]
#         start_mp4_timestamp_i = start_timestamp_i_after[0]
#         end_mp4_timestamp_i = end_timestamp_i_before[-1]
#         start_vrs_timestamp_i = self.timestamps_mp4_to_vrs[start_mp4_timestamp_i]
#         end_vrs_timestamp_i = self.timestamps_mp4_to_vrs[end_mp4_timestamp_i]

#         # Do the offset to find the handless frame => DEPRECATED
#         pre_sample_timestamp_offset = 0  # duration * 0.2
#         start_mp4_timestamp_i_prev = (
#             start_mp4_timestamp_i - pre_sample_timestamp_offset * 1e9
#         )
#         start_vrs_timestamp_i_prev = (
#             start_vrs_timestamp_i - pre_sample_timestamp_offset * 1e9
#         )

#         # Decide for the frame range
#         sampled_mp4_timestamps = np.arange(
#             start_mp4_timestamp_i_prev, end_mp4_timestamp_i, 1e9 / self.fps
#         ).astype(np.int64)
#         sampled_vrs_timestamps = []
#         for mp4_timestamp in sampled_mp4_timestamps:
#             # Finding nearest vrs timestamp
#             time_diff = np.abs(self.video_timestamps_mp4 - mp4_timestamp)
#             nearest_mp4_timestamp = self.video_timestamps_mp4[
#                 np.argmin(np.abs(self.video_timestamps_mp4 - mp4_timestamp))
#             ]
#             nearest_vrs_timestamp = self.timestamps_mp4_to_vrs[nearest_mp4_timestamp]
#             sampled_vrs_timestamps.append(nearest_vrs_timestamp)
#         sampled_vrs_timestamps = np.array(sampled_vrs_timestamps)

#         # sampled_vrs_timestamps = np.arange(
#         #     start_vrs_timestamp_i_prev, end_vrs_timestamp_i, 1e9/self.fps).astype(np.int64)
#         # video_len = min(len(sampled_mp4_timestamps),
#         #                 len(sampled_vrs_timestamps))
#         # sampled_mp4_timestamps = sampled_mp4_timestamps[:video_len]
#         # sampled_vrs_timestamps = sampled_vrs_timestamps[:video_len]

#         # Get the handless frame
#         start_frame_id = np.where(sampled_mp4_timestamps >= start_mp4_timestamp_i)[0][0]
#         end_frame_id = np.where(sampled_mp4_timestamps <= end_mp4_timestamp_i)[0][-1]

#         # Sample frames
#         _rgb_images = self.recording_data_provider.get_rgb_images(
#             sampled_mp4_timestamps,
#             TimeDomain.DEVICE_TIME,
#             use_video=True,
#             as_array=True,
#         )[0]
#         if self.rectify_images:
#             rgb_images = []
#             for rgb_image in _rgb_images:
#                 rgb_image = calibration.distort_by_calibration(
#                     rgb_image, self.pinhole, self.calib
#                 )
#                 rgb_images.append(rgb_image)
#         else:
#             rgb_images = _rgb_images
#         rgb_images = np.stack(rgb_images, axis=0)
#         data["rgb_images"] = rgb_images
#         data["timestamps_mp4"] = sampled_mp4_timestamps
#         data["timestamps_vrs"] = sampled_vrs_timestamps
#         data["start_frame_id"] = start_frame_id
#         data["end_frame_id"] = end_frame_id
#         return data

#     def _get_slam_poses_and_hand_trajs(self, idx, data):
#         # Get the slam poses and hand trajectory
#         sampled_vrs_timestamps = data["timestamps_vrs"]
#         intr = np.array(
#             [
#                 [self.focal, 0, self.width / 2],
#                 [0, self.focal, self.height / 2],
#                 [0, 0, 1],
#             ]
#         )
#         sparse_depths, slam_poses = [], []

#         # Get the hand trajectory
#         wrist_traj_left, wrist_traj_right = (
#             np.ones((len(sampled_vrs_timestamps), 3)) * np.nan,
#             np.ones((len(sampled_vrs_timestamps), 3)) * np.nan,
#         )
#         palm_traj_left, palm_traj_right = (
#             np.ones((len(sampled_vrs_timestamps), 3)) * np.nan,
#             np.ones((len(sampled_vrs_timestamps), 3)) * np.nan,
#         )

#         # Get the validity of the hand trajectory
#         hand_traj_timestamps_left, hand_traj_timestamps_right = (
#             np.ones((len(sampled_vrs_timestamps),)) * np.nan,
#             np.ones((len(sampled_vrs_timestamps),)) * np.nan,
#         )
#         hand_traj_left_valid, hand_traj_right_valid = np.zeros(
#             (len(sampled_vrs_timestamps),)
#         ), np.zeros((len(sampled_vrs_timestamps),))
#         hand_traj_left_visible, hand_traj_right_visible = np.zeros(
#             (len(sampled_vrs_timestamps),)
#         ), np.zeros((len(sampled_vrs_timestamps),))

#         for ti, sampled_vrs_timestamp in enumerate(sampled_vrs_timestamps):
#             # Acquire the slam pose
#             T_world_head, t_diff = self.recording_data_provider.get_pose(
#                 sampled_vrs_timestamp, TimeDomain.DEVICE_TIME, martrix_format=True
#             )
#             T_world_cam = T_world_head @ self.T_head_cam
#             if self.calibrate_slam:
#                 T_world_cam = T_world_cam @ AriaUtils.T_z_m90
#             slam_poses.append(T_world_cam)

#             # Acquire the sparse depth
#             depth_sparse = AriaUtils.get_sparse_depth(
#                 self.slam_points, T_world_cam, intr, self.height, self.width
#             )
#             sparse_depths.append(depth_sparse)

#             # Acquire the wrist trajectory
#             wrist_and_palm_results, t_diff = (
#                 self.recording_data_provider.get_wrist_and_palm_pose(
#                     sampled_vrs_timestamp, dict_format=True
#                 )
#             )

#             # Get the hand trajectory
#             # query_hands = data["hands"]
#             query_hands = ["left", "right"]
#             for qh in query_hands:
#                 if wrist_and_palm_results[qh] is None:
#                     continue
#                 if wrist_and_palm_results[qh].confidence > self.hand_threshold:
#                     wrist_wp = wrist_and_palm_results[qh].wrist
#                     palm_wp = wrist_and_palm_results[qh].palm
#                     wrist_wp = DataUtils.transform_points(wrist_wp, T_world_head)
#                     palm_wp = DataUtils.transform_points(palm_wp, T_world_head)
#                     palm_points_approx = self.hand_points_approx + palm_wp
#                     wrist_points_approx = self.hand_points_approx + wrist_wp
#                     hand_points_approx = np.concatenate(
#                         [palm_points_approx, wrist_points_approx], axis=0
#                     )
#                     if not self.calibrate_slam:
#                         _T_cam_world = np.linalg.inv(T_world_cam @ AriaUtils.T_z_m90)
#                     else:
#                         _T_cam_world = np.linalg.inv(T_world_cam)
#                     hand_uvs_in_image_space = DataUtils.project_points_to_image(
#                         hand_points_approx, intr, _T_cam_world
#                     )
#                     hand_uvs_in_image_space = hand_uvs_in_image_space.astype(np.int32)

#                     # Compute the visibility score
#                     hand_uvs_in_image_space_valid = (
#                         (hand_uvs_in_image_space[:, 0] >= 0)
#                         & (hand_uvs_in_image_space[:, 0] < self.width)
#                         & (hand_uvs_in_image_space[:, 1] >= 0)
#                         & (hand_uvs_in_image_space[:, 1] < self.height)
#                     )
#                     hand_uvs_in_image_space_valid = hand_uvs_in_image_space_valid.sum()
#                     hand_uvs_in_image_space_valid = hand_uvs_in_image_space_valid / len(
#                         hand_uvs_in_image_space
#                     )

#                     if qh == "left":
#                         wrist_traj_left[ti] = wrist_wp
#                         palm_traj_left[ti] = palm_wp
#                         hand_traj_left_valid[ti] = wrist_and_palm_results[qh].confidence
#                         hand_traj_left_visible[ti] = hand_uvs_in_image_space_valid
#                         hand_traj_timestamps_left[ti] = sampled_vrs_timestamp
#                     elif qh == "right":
#                         wrist_traj_right[ti] = wrist_wp
#                         palm_traj_right[ti] = palm_wp
#                         hand_traj_right_valid[ti] = wrist_and_palm_results[
#                             qh
#                         ].confidence
#                         hand_traj_left_visible[ti] = hand_uvs_in_image_space_valid
#                         hand_traj_timestamps_right[ti] = sampled_vrs_timestamp
#                     else:
#                         raise ValueError(f"Invalid hand: {qh}")

#         slam_poses = np.stack(slam_poses, axis=0)
#         sparse_depths = np.stack(sparse_depths, axis=0)
#         data["intr"] = intr
#         data["sparse_depths"] = sparse_depths
#         data["slam_poses"] = slam_poses
#         data["wrist_traj_left"] = wrist_traj_left
#         data["wrist_traj_right"] = wrist_traj_right
#         data["palm_traj_left"] = palm_traj_left
#         data["palm_traj_right"] = palm_traj_right

#         # The timestamps of the hand trajectory
#         data["hand_traj_vrs_timestamps_left"] = hand_traj_timestamps_left
#         data["hand_traj_vrs_timestamps_right"] = hand_traj_timestamps_right

#         # The predicted score of the hand
#         data["hand_traj_left_valid"] = hand_traj_left_valid
#         data["hand_traj_right_valid"] = hand_traj_right_valid

#         # The visibility score of the hand
#         data["hand_traj_left_visible"] = hand_traj_left_visible
#         data["hand_traj_right_visible"] = hand_traj_right_visible

#         # Resample the hand timestamps; make sure the initial timestamp has a valid hand trajectory
#         if hand_traj_left_valid.sum() > 0:
#             left_hand_idx = np.where(hand_traj_left_valid > 0)[0][0]
#         else:
#             left_hand_idx = 0

#         if hand_traj_right_valid.sum() > 0:
#             right_hand_idx = np.where(hand_traj_right_valid > 0)[0][0]
#         else:
#             right_hand_idx = 0

#         first_valid_hand_idx = min(left_hand_idx, right_hand_idx)

#         # Update every timestamp to the first valid hand trajectory
#         data["start_frame_id"] = first_valid_hand_idx
#         for k in self.time_stamp_aligned_keys:
#             if k in data:
#                 data[k] = data[k][first_valid_hand_idx:]
#             else:
#                 raise ValueError(f"Key {k} not found in data")
#         # assert data["hand_traj_left_valid"][0] > 0 or data["hand_traj_right_valid"][0] > 0
#         return data

#     def _get_annotations(self, idx, data):
#         unique_narration_id = self.annotations["unique_narration_id"].iloc[idx]
#         narration_i = self.annotations["narration"].iloc[idx]
#         main_actions_i = self.annotations["main_actions"].iloc[idx]
#         hand_i = self.annotations["hands"].iloc[idx][0]
#         nouns_i = self.annotations["nouns"].iloc[idx]
#         verbs_i = self.annotations["verbs"].iloc[idx]
#         query_hands_i = {
#             "both hands": ["left", "right"],
#             "left hand": ["left"],
#             "right hand": ["right"],
#         }[hand_i]

#         data.update(
#             {
#                 # Annotation from the dataset
#                 "hands": query_hands_i,
#                 "nouns": nouns_i,
#                 "verbs": verbs_i,
#                 "narration": narration_i,
#                 "main_actions": main_actions_i,
#                 "unique_narration_id": unique_narration_id,
#             }
#         )

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         data = {}

#         # Get the rgb images
#         self._get_annotations(idx, data)
#         self._get_rgb_images(idx, data)
#         self._get_slam_poses_and_hand_trajs(idx, data)
#         return data


# if __name__ == "__main__":
#     dataset = HD_EPIC_Dataset()
#     data = dataset[20]
#     rgb_images = data["rgb_images"]
#     sparse_depths = data["sparse_depths"]
#     for rgb_image, sparse_depth in zip(rgb_images, sparse_depths):
#         rgb_image = np.rot90(rgb_image, axes=(0, 1)).copy()

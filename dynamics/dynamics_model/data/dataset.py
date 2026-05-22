import os
import numpy as np
from pandas.core import frame
import torch
from torch.utils.data import Dataset
import cv2
import pickle
import pandas as pd
from utils.aria_utils import parse_hoi4d_mask, read_o3d_poses
import utils.dataset_utils as DatasetUtils
from scipy.signal import savgol_filter
import open3d as o3d
from utils.viewer_utils import SceneViewer
import utils.aria_utils as AriaUtils
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
import time

DATA_DIR = "/storage/group/dataset_mirrors/01_incoming/egoasis4d/"
DEFAULT_FPS = 15
SAMPLE_WEIGHTS = {
    "hoi4d": 1.0,
    "arti4d": 1.0,
    "egodex": 1.0,
}


class Egoasis4DDataset(Dataset):
    ACTION_DIM = 48
    WRIST_ACTION_DIM = 20
    STATE_DIM = 3
    DOWNSAMPLING_FACTOR_HOI4D = 4
    HEAD_IMAGE_SIZE = (224, 224)

    def __init__(
        self,
        split_fpath,
        data_dir=DATA_DIR,
        dataset_sample_weights=SAMPLE_WEIGHTS,
        clip_length=1,
        fps=15,
        horizon=15,
        flow_horizon=15,
        action_chunk_size=15,
        transform=None,
        language_max_length=30,
        track_patch_size=32,
        load_hands=True,
        load_tracks=False,
        load_rgbd_frames=True,
        build_track_online=True,
        load_distance_to_goal=False,
        feature_extractor="dinov3",  # or "cut3r"
        distance_to_goal_format="distance",  # "progress" or "distance"
        track_by_clip=False,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        load_finger_tips=False,
        action_include_progress=False,
        action_in_relative=False,
        flow_in_relative=True,
        training=False,
        reward_type="dense", # "sparse" or "dense"
        dinov3_visual_dim=768,
        **kwargs,
    ):
        super().__init__()
        df = pd.read_csv(split_fpath)
        # Add the dataset name if not exists
        if "dataset" not in df.columns:
            if "hoi4d_" in split_fpath:
                df["dataset"] = "hoi4d"
            elif "arti4d_" in split_fpath:
                df["dataset"] = "arti4d"
            elif os.path.basename(split_fpath).startswith("arti4d_releaseclip"):
                # e.g. arti4d_releaseclip_valid_frame_ranges.csv (no "arti4d_" in path)
                df["dataset"] = "arti4d"
            elif "egodex_" in split_fpath:
                df["dataset"] = "egodex"
        assert (
            self.HEAD_IMAGE_SIZE[0] == self.HEAD_IMAGE_SIZE[1]
        ), "Head image size should be square"
        # # Repeat the samples according to the sample weights
        # if dataset_sample_weights:
        #     weight_series = df["dataset"].map(dataset_sample_weights)
        #     if weight_series.isna().any():
        #         missing = sorted(df.loc[weight_series.isna(), "dataset"].unique())
        #         raise ValueError(f"Missing SAMPLE_WEIGHTS for: {missing}")
        #     repeat_counts = np.rint(weight_series.values).astype(np.int64)
        #     repeat_counts = np.clip(repeat_counts, 1, None)
        #     df = df.loc[df.index.repeat(repeat_counts)].reset_index(drop=True)

        basename = split_fpath.split("/")[-1].strip(".csv")
        basename = basename.split("_")[:2]
        basename = "_".join(basename)

        self.data_dir = data_dir
        self.fps = fps
        self.track_patch_size = track_patch_size
        self.df = df
        if "advantage_label" not in self.df.columns:
            self.df["advantage_label"] = -1

        self.advantage_labels = self.df["advantage_label"].values

        # self.downsample_factor = downsample_factor
        self.transform = transform
        self.horizon = horizon
        self.flow_horizon = flow_horizon
        self.clip_length = clip_length
        self.track_horizon_of_interest = track_horizon_of_interest
        self.language_max_length = language_max_length
        self.load_hands = load_hands
        self.load_tracks = load_tracks
        self.build_track_online = build_track_online
        self.load_distance_to_goal = load_distance_to_goal
        self.distance_to_goal_format = distance_to_goal_format
        self.load_finger_tips = load_finger_tips
        self.track_by_clip = track_by_clip
        self.action_include_progress = action_include_progress
        self.action_in_relative = action_in_relative
        self.flow_in_relative = flow_in_relative
        self.training = training
        self.load_rgbd_frames = load_rgbd_frames
        self.action_chunk_size = action_chunk_size
        self.feature_extractor = feature_extractor
        self.reward_type = reward_type
        self.dinov3_visual_dim = dinov3_visual_dim
        # if not self.track_by_clip:
        #     assert (
        #         self.build_track_online
        #     ), "Build track online must be True when track_by_clip is False"
        #     self.build_track_online = True

        # Determine the action and state statistics file paths
        action_prefix = "rel" if self.action_in_relative else ""
        state_prefix = "clip" if self.track_by_clip else "seq"
        state_suffix = "InAbs" if not self.flow_in_relative else ""
        stat_action_fpath = os.path.join(
            "data/splits",
            f"egoasis4d_releaseclip_{action_prefix}action_FPS15_HORIZON15.json",
        )
        stat_action_for_dynamics_fpath = os.path.join(
            "data/splits",
            f"egoasis4d_releaseclip_relaction_FPS15_HORIZON15.json",
        )
        if (
            "hoiarti4d_" in split_fpath
            or "hoi4d_" in split_fpath
            or "arti4d_" in split_fpath
            or "mrl4d_" in split_fpath
        ):
            stat_state_fpath = os.path.join(
                "data/splits",
                f"hoiarti4d_releaseclipAll-{state_prefix}_state{state_suffix}_FPS15_HORIZON15.json",
            )
        elif "egoasis4d_" in split_fpath:
            stat_state_fpath = os.path.join(
                "data/splits",
                f"egoasis4d_releaseclip-{state_prefix}_state{state_suffix}_FPS15_HORIZON15.json",
            )

        if self.load_hands:
            print(f"Loading the action statistics from {stat_action_fpath}...")
            stat_action = DatasetUtils.load_json(stat_action_fpath)
            self.action_mean = np.array(stat_action["action_mean"])
            self.action_std = np.array(stat_action["action_std"])
            self.action_norm_max = np.array(stat_action["action_norm_max99.5"])
            self.action_norm_min = np.array(stat_action["action_norm_min99.5"])

            # Load the action statistics for the dynamics
            stat_action_for_dynamics = DatasetUtils.load_json(
                stat_action_for_dynamics_fpath
            )
            self.action_for_dynamics_mean = np.array(
                stat_action_for_dynamics["action_mean"]
            )
            self.action_for_dynamics_std = np.array(
                stat_action_for_dynamics["action_std"]
            )
            self.action_for_dynamics_norm_max = np.array(
                stat_action_for_dynamics["action_norm_max"]
            )
            self.action_for_dynamics_norm_min = np.array(
                stat_action_for_dynamics["action_norm_min"]
            )

            if not self.load_finger_tips:
                (
                    self.action_mean,
                    self.action_std,
                    self.action_norm_max,
                    self.action_norm_min,
                ) = self._postprocess_action_statistics(
                    self.action_mean,
                    self.action_std,
                    self.action_norm_max,
                    self.action_norm_min,
                )
                (
                    self.action_for_dynamics_mean,
                    self.action_for_dynamics_std,
                    self.action_for_dynamics_norm_max,
                    self.action_for_dynamics_norm_min,
                ) = self._postprocess_action_statistics(
                    self.action_for_dynamics_mean,
                    self.action_for_dynamics_std,
                    self.action_for_dynamics_norm_max,
                    self.action_for_dynamics_norm_min,
                )

            for k, v in zip(
                ["action_mean", "action_std", "action_norm_max", "action_norm_min"],
                [
                    self.action_mean,
                    self.action_std,
                    self.action_norm_max,
                    self.action_norm_min,
                ],
            ):
                print(f"Value of {k}: {v}")
        # Load the state statistics
        if self.load_tracks:
            # input_string = input(
            #     f"Are we sure we get the right state statistics? (y/n): {stat_state_fpath}"
            # )
            # if input_string != "y" or input_string != "Y":
            #     raise ValueError("We are not sure we get the right state statistics!")
            print(f"Loading the state statistics from {stat_state_fpath}...")
            stat_state = DatasetUtils.load_json(stat_state_fpath)
            (
                self.state_mean,
                self.state_std,
                self.state_norm_max,
                self.state_norm_min,
            ) = (
                [],
                [],
                [],
                [],
            )
            for k in range(self.flow_horizon):
                _k = k  # self.flow_horizon - 1
                self.state_mean.append(
                    np.array(stat_state["state_residual_" + str(_k)]["mean"])
                )
                self.state_std.append(
                    np.array(stat_state["state_residual_" + str(_k)]["std"])
                )
                self.state_norm_max.append(
                    np.array(stat_state["state_residual_" + str(_k)]["norm_max"])
                )
                self.state_norm_min.append(
                    np.array(stat_state["state_residual_" + str(_k)]["norm_min"])
                )

            # Stack the state statistics
            self.state_mean = np.stack(self.state_mean, axis=0)  # (H_state, 3)
            self.state_std = np.stack(self.state_std, axis=0)  # (H_state, 3)
            self.state_norm_max = np.stack(self.state_norm_max, axis=0)  # (H_state, 3)
            self.state_norm_min = np.stack(self.state_norm_min, axis=0)  # (H_state, 3)
            self.state_mean = self.state_mean[-self.horizon :]
            self.state_std = self.state_std[-self.horizon :]
            self.state_norm_max = self.state_norm_max[-self.horizon :]
            self.state_norm_min = self.state_norm_min[-self.horizon :]

        self.split_fpath = split_fpath
        self.samples = df["sample"].tolist()
        self.valid_frame_ranges = df["valid_frame_range"].tolist()
        self.valid_frame_ranges = [eval(v) for v in self.valid_frame_ranges]
        self.fine_actions = df["fine_action"].tolist()
        self.dataset_categories = df["dataset"].tolist()
        # Load class and task labels
        self._build_sample_clip_pairs()

    def __len__(self):
        return len(self.sample_clip_pairs)

    def _postprocess_action_statistics(
        self,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        action_norm_max: np.ndarray,
        action_norm_min: np.ndarray,
    ):
        # 拆左右手
        left_action_mean, right_action_mean = (
            action_mean[: self.ACTION_DIM // 2],
            action_mean[self.ACTION_DIM // 2 :],
        )
        left_action_std, right_action_std = (
            action_std[: self.ACTION_DIM // 2],
            action_std[self.ACTION_DIM // 2 :],
        )
        left_action_norm_max, right_action_norm_max = (
            action_norm_max[: self.ACTION_DIM // 2],
            action_norm_max[self.ACTION_DIM // 2 :],
        )
        left_action_norm_min, right_action_norm_min = (
            action_norm_min[: self.ACTION_DIM // 2],
            action_norm_min[self.ACTION_DIM // 2 :],
        )

        # 每手 3 trans + 1 closure + 6 r6d
        def build_hand_stats(mean, std, nmax, nmin):
            hand_mean = np.concatenate([mean[:3], np.array([0.0]), mean[-6:]], axis=0)
            hand_std = np.concatenate([std[:3], np.array([1.0]), std[-6:]], axis=0)
            hand_norm_max = np.concatenate(
                [nmax[:3], np.array([1.0]), nmax[-6:]], axis=0
            )
            hand_norm_min = np.concatenate(
                [nmin[:3], np.array([0.0]), nmin[-6:]], axis=0
            )
            return hand_mean, hand_std, hand_norm_max, hand_norm_min

        (
            left_action_mean,
            left_action_std,
            left_action_norm_max,
            left_action_norm_min,
        ) = build_hand_stats(
            left_action_mean,
            left_action_std,
            left_action_norm_max,
            left_action_norm_min,
        )
        (
            right_action_mean,
            right_action_std,
            right_action_norm_max,
            right_action_norm_min,
        ) = build_hand_stats(
            right_action_mean,
            right_action_std,
            right_action_norm_max,
            right_action_norm_min,
        )

        action_mean = np.concatenate([left_action_mean, right_action_mean], axis=0)
        action_std = np.concatenate([left_action_std, right_action_std], axis=0)
        action_norm_max = np.concatenate(
            [left_action_norm_max, right_action_norm_max], axis=0
        )
        action_norm_min = np.concatenate(
            [left_action_norm_min, right_action_norm_min], axis=0
        )

        # 可选：检查维度
        assert action_mean.shape[0] == self.WRIST_ACTION_DIM
        return action_mean, action_std, action_norm_max, action_norm_min

    def _build_sample_clip_pairs(self):
        self.sample_clip_pairs = []
        for sample, valid_frame_range in tqdm(
            zip(self.samples, self.valid_frame_ranges),
            total=len(self.samples),
            desc="Building sample clip pairs",
        ):
            for frame in range(
                valid_frame_range[0], valid_frame_range[1], self.clip_length
            ):
                frame_start, frame_end = frame, frame + self.clip_length - 1
                self.sample_clip_pairs.append((sample, [frame_start, frame_end]))

        # Build sample name to index mapping
        self.sample_name_to_index = {
            sample: idx for idx, sample in enumerate(self.samples)
        }

    def _get_clip(self, idx):
        sample, clip_range = self.sample_clip_pairs[idx]
        valid_frame_range = self.valid_frame_ranges[self.sample_name_to_index[sample]]
        clip_data = []
        # valid_frames = []
        for frame in np.arange(clip_range[0], clip_range[0] + self.clip_length):
            frame = min(frame, valid_frame_range[1] - 1)
            clip_data.append(self._get_frame(sample, frame))
        # print(valid_frames)
        return clip_data

    def _get_frame(self, sample_name, frame_idx):
        sample_idx = self.sample_name_to_index[sample_name]
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        fine_action = self.fine_actions[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        frame = frame_idx

        if dataset_name == "hoi4d":
            sample, clip_idx = sample_name.split(".")[0], sample_name.split(".")[1]
        else:
            _sample_name = sample_name.split("/")
            sample = "/".join(_sample_name[:-1])
            clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"

        # Acquire the file paths
        # traj_suffix = "" if self.fps == 30 else f"_{self.fps}fps"
        dataset_path = os.path.join(
            self.data_dir, dataset_name, f"{dataset_name.upper()}_release"
        )
        color_video_fpath = os.path.join(dataset_path, sample, "rgb")
        depth_video_fpath = os.path.join(dataset_path, sample, "depth")
        traj_video_fpath = os.path.join(dataset_path, sample, f"dex_traj")
        mask_video_fpath = os.path.join(dataset_path, sample, "2Dseg/mask")
        # visual_feature_fpath = os.path.join(
        #     dataset_path, sample, "cut3r_visual_feature_by_clip_224"
        # )
        # track_fpath = os.path.join(dataset_path, sample, "tapip3d_tracks_224_64")

        # action_anno_fpath = os.path.join(self.data_dir, sample, "action/color.json")
        # pose_anno_fpath = os.path.join(self.data_dir, sample, "3Dseg/output.log")
        language_embedding_fpath = os.path.join(
            self.data_dir,
            dataset_name,
            "language_embedding",
            f"{fine_action.replace(' ', '_')}.npz",
        )
        language_embedding_raw = np.load(language_embedding_fpath)["embedding"][
            : self.language_max_length
        ]
        idx = np.arange(self.language_max_length) % language_embedding_raw.shape[0]
        language_embedding = language_embedding_raw[idx]

        if dataset_name == "hoi4d":
            camera_intr_fpath = os.path.join(
                dataset_path, sample.split("/")[0], "intrinsics.npy"
            )
            camera_extr_fpath = os.path.join(
                dataset_path,
                sample,
                "extr_cam0cam",
                f"{sample.replace('/', '_')}-{clip_idx}.npz",
            )
            camera_intr = np.load(camera_intr_fpath)
            camera_extr = np.load(camera_extr_fpath)
            camera_intr[:2] /= self.DOWNSAMPLING_FACTOR_HOI4D
            T_wc_list, T_c0c_list = (
                camera_extr["extrinsics_world_cam"],
                camera_extr["extrinsics"],
            )
        else:
            camera_intr_fpath = os.path.join(
                dataset_path, sample, "intr", f"intrinsics_{clip_idx}.npz"
            )
            camera_extr_fpath = os.path.join(
                dataset_path, sample, "extr_cam0cam", f"extrinsics_{clip_idx}.npz"
            )
            camera_intr = np.load(camera_intr_fpath)["intrinsics"]
            camera_extr = np.load(camera_extr_fpath)
            T_wc_list, T_c0c_list = (
                camera_extr["extrinsics"],
                camera_extr["extrinsics"],
            )

        # Parse the extrinsics
        T_wc0 = T_wc_list[0]
        T_wc = T_wc_list[frame - valid_frame_range[0]]
        T_c0c = np.linalg.inv(T_wc0) @ T_wc

        # Parse the file paths
        num_digit = 5 if dataset_name == "hoi4d" else 6
        depth_frame_fpath_curr = os.path.join(
            depth_video_fpath, f"{frame:0{num_digit}d}.png"
        )
        color_frame_fpath_curr = os.path.join(
            color_video_fpath, f"{frame:0{num_digit}d}.jpg"
        )

        traj_npz_fpath_curr = os.path.join(
            traj_video_fpath, f"{frame:0{num_digit}d}.npz"
        )

        # track_fpath_curr = os.path.join(track_fpath, f"{frame:05d}.npz")
        if self.track_by_clip:
            mask_frame_fpath_curr = os.path.join(
                mask_video_fpath, f"{frame:0{num_digit}d}.png"
            )
            color_frame_fpath_init = os.path.join(
                color_video_fpath, f"{frame:0{num_digit}d}.jpg"
            )
        else:
            mask_frame_fpath_curr = os.path.join(
                mask_video_fpath, f"{valid_frame_range[0]:0{num_digit}d}.png"
            )
            color_frame_fpath_init = os.path.join(
                color_video_fpath, f"{valid_frame_range[0]:0{num_digit}d}.jpg"
            )

        color_frame_init = cv2.imread(color_frame_fpath_init)[..., [2, 1, 0]]
        has_dynamic_mask = False
        if os.path.exists(mask_frame_fpath_curr):
            mask_frame_curr = cv2.imread(mask_frame_fpath_curr)[..., [2, 1, 0]]
            mask_frame_curr = cv2.resize(
                mask_frame_curr,
                dsize=None,
                fx=1 / self.DOWNSAMPLING_FACTOR_HOI4D,
                fy=1 / self.DOWNSAMPLING_FACTOR_HOI4D,
                interpolation=cv2.INTER_NEAREST,
            )
            # Select the non-black regions
            mask_dynamic_curr = (mask_frame_curr > 0).astype(np.float32)
            masks = parse_hoi4d_mask(mask_frame_curr)
            if len(masks) == 0:
                mask_dynamic_curr = np.ones(
                    (color_frame_init.shape[0], color_frame_init.shape[1])
                )
            else:
                mask_dynamic_curr = (
                    np.sum(np.stack(masks, axis=0), axis=0) > 0
                ).astype(np.float32)
                has_dynamic_mask = True
        else:
            mask_dynamic_curr = np.ones(
                (color_frame_init.shape[0], color_frame_init.shape[1])
            )

        if dataset_name == "hoi4d":
            height, width = color_frame_init.shape[:2]
            assert height < width, "Height should be less than width"
            crop_size = height
            crop_x = (width - crop_size) // 2
            crop_y = (height - crop_size) // 2
            color_frame_init = color_frame_init[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            mask_dynamic_curr = mask_dynamic_curr[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            color_frame_init = cv2.resize(
                color_frame_init,
                (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_dynamic_curr = cv2.resize(
                mask_dynamic_curr,
                (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            camera_intr[0, 0] *= self.HEAD_IMAGE_SIZE[1] / crop_size
            camera_intr[1, 1] *= self.HEAD_IMAGE_SIZE[0] / crop_size
            camera_intr[0, 2] = self.HEAD_IMAGE_SIZE[1] / 2
            camera_intr[1, 2] = self.HEAD_IMAGE_SIZE[0] / 2

        # Get the rgbd frames if needed
        if self.load_rgbd_frames:
            depth_frame = (
                cv2.imread(depth_frame_fpath_curr, cv2.IMREAD_UNCHANGED) / 1000.0
            )
            color_frame = cv2.imread(color_frame_fpath_curr)[..., [2, 1, 0]]
            if dataset_name == "hoi4d":
                color_frame = color_frame[
                    crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
                ]
                depth_frame = depth_frame[
                    crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
                ]

                # Resize the image to the target size
                color_frame = cv2.resize(
                    color_frame,
                    (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                depth_frame = cv2.resize(
                    depth_frame,
                    (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            color_frame = np.zeros(
                (self.HEAD_IMAGE_SIZE[0], self.HEAD_IMAGE_SIZE[1], 3)
            )
            depth_frame = np.zeros((self.HEAD_IMAGE_SIZE[0], self.HEAD_IMAGE_SIZE[1]))

        # Crop the image and resize to the target size

        data = {
            "color": color_frame,
            "depth": depth_frame,
            "color_init": color_frame_init,
            "mask_dynamic_curr": mask_dynamic_curr,
            # "mask_is_valid": mask_is_valid,
            # "mask_hand": mask_hand,
            # "mask_object": mask_object,
            "intrinsics": camera_intr,
            "T_cam0_cam": T_c0c,
            "T_world_cam": T_wc,
            "slam_pose": T_wc,
            "frame_range": valid_frame_range,
            "frame_idx": frame,
            "language_feature": language_embedding,
            "relative_time": np.array([frame - valid_frame_range[0]]).squeeze()
            / DEFAULT_FPS,
        }

        # Load the feature data
        visual_feature_data = self._build_visual_feature_history(
            sample_idx, sample_name, frame
        )
        visual_history_patch, visual_goal_patch = self._process_visual_feature(
            visual_feature_data
        )

        # Read the extrinsics data
        history_T_ch, history_raymap = self._build_raymap_history(
            sample_idx,
            sample_name,
            frame,
            T_wc_list,
            camera_intr,
            valid_frame_range,
            input_size=self.HEAD_IMAGE_SIZE,
            patch_size=(
                self.HEAD_IMAGE_SIZE[0] // 16,
                self.HEAD_IMAGE_SIZE[1] // 16,
            ),
        )
        # Compute the state value
        T = valid_frame_range[1] - valid_frame_range[0]
        idx = frame - valid_frame_range[0]
        if self.reward_type == "sparse":
            gamma = 0.995
            gt_state_value = gamma ** (T - idx)
        elif self.reward_type == "dense":
            gt_state_value = idx / T
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        advantage_label = self.advantage_labels[sample_idx]

        # Update the data
        data.update(
            {
                "history_visual_feature_patch": visual_history_patch,
                "goal_visual_feature_patch": visual_goal_patch,
                "history_T_cam_history": history_T_ch,
                "history_raymap": history_raymap,
                "gt_state_value": gt_state_value,
                "advantage_label": advantage_label,
            }
        )

        if self.load_hands:
            # Get the hand trajectory
            if os.path.exists(traj_npz_fpath_curr):
                hand_data = np.load(traj_npz_fpath_curr, allow_pickle=True)
            else:
                hand_data = None
            (
                hand_traj,
                hand_valid,
                hand_traj_timestamps,
                hand_traj_history,
                hand_valid_history,
                hand_traj_timestamps_history,
                start_pos,
            ) = self._process_hand_data(hand_data)
            hand_traj_rel = (
                DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                    hand_traj, start_pos, has_finger_tips=self.load_finger_tips
                )
            )
            hand_traj_history_rel = (
                DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                    hand_traj_history, start_pos, has_finger_tips=self.load_finger_tips
                )
            )

            if self.action_in_relative:
                hand_traj = (
                    DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                        hand_traj_rel, start_pos, has_finger_tips=self.load_finger_tips
                    )
                )
                hand_traj_history = (
                    DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                        hand_traj_history_rel,
                        start_pos,
                        has_finger_tips=self.load_finger_tips,
                    )
                )

            if self.action_include_progress:

                progress_value = np.arange(
                    frame_idx, frame_idx + self.action_chunk_size
                )[
                    :, None
                ]  # [H, 1]
                progress_value = (progress_value - valid_frame_range[0]) / (
                    valid_frame_range[1] - valid_frame_range[0] + 1
                )  # [H, 1]
                progress_value = np.clip(progress_value, 0, 1)
                value_valid = np.ones((self.action_chunk_size, 1))
                hand_traj = np.concatenate([hand_traj, progress_value], axis=1)
                hand_traj_rel = np.concatenate([hand_traj_rel, progress_value], axis=1)
                hand_valid = np.concatenate([hand_valid, value_valid], axis=1)

            data.update(
                {
                    # Add the action data
                    "start_pos": start_pos,
                    "gt_action": hand_traj,
                    "gt_action_rel": hand_traj_rel,
                    "action_valid": hand_valid,
                    "action_timestamps": hand_traj_timestamps,
                    "action_mean": self.action_mean,
                    "action_std": self.action_std,
                    "action_norm_max_bound": self.action_norm_max,
                    "action_norm_min_bound": self.action_norm_min,
                    "action_for_dynamics_mean": self.action_for_dynamics_mean,
                    "action_for_dynamics_std": self.action_for_dynamics_std,
                    "action_for_dynamics_norm_max_bound": self.action_for_dynamics_norm_max,
                    "action_for_dynamics_norm_min_bound": self.action_for_dynamics_norm_min,
                    # Add the history data
                    "history_action": hand_traj_history,
                    "history_action_rel": hand_traj_history_rel,
                    "history_action_valid": hand_valid_history,
                    "history_action_timestamps": hand_traj_timestamps_history,
                }
            )

        if self.load_tracks:
            # if os.path.exists(track_fpath_curr):
            #     track_data = np.load(track_fpath_curr)
            # else:
            #     track_data = None
            track_data = self._build_track_data(
                sample_idx, sample_name, frame, T_c0c, track_by_clip=self.track_by_clip
            )
            (
                gt_track_future,  # [T, 3, H, W]
                gt_track_future_valid,  # [T, 3, H, W]
                gt_track_future_visib,  # [T, 1, H, W]
                gt_track_history,  # [T, 3, H, W]
                gt_track_history_valid,  # [T, 3, H, W]
                gt_track_history_visib,  # [T, 1, H, W]
                gt_track_init,  # [3, H, W]
                gt_track_color,  # [3, H, W]
                gt_track_distance_to_goal,  # [1, 3, H, W]
            ) = self._process_track_data(track_data)

            # gt_track_residual = gt_track_future - gt_track_init[None]  # [T, 3, H, W]
            gt_track_residual = (
                gt_track_future - gt_track_history[-1][None]
            )  # gt_track_init[None]  # [T, 3, H, W]

            gt_track_mask_valid = cv2.resize(
                mask_dynamic_curr,
                (self.track_patch_size, self.track_patch_size),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_track_mask_valid = cv2.erode(
                gt_track_mask_valid,
                kernel=np.ones((3, 3), np.uint8),
                iterations=1,
            )[
                None, None
            ]  # [1, 1, H, W]

            if has_dynamic_mask:
                gt_track_mask_valid = np.where(
                    gt_track_mask_valid > 0, 2.0, 1.0
                )  # Emphasize the dynamic regions!
            else:
                gt_track_mask_valid = np.ones_like(gt_track_mask_valid)
            gt_track_future_valid *= gt_track_mask_valid
            gt_track_history_valid *= gt_track_mask_valid
            ## DEBUG: mask the track data

            # Acquire the state statistics
            state_mean = self.state_mean[self.track_horizon_of_interest]
            state_std = self.state_std[self.track_horizon_of_interest]
            state_norm_max = self.state_norm_max[self.track_horizon_of_interest]
            state_norm_min = self.state_norm_min[self.track_horizon_of_interest]
            if self.load_distance_to_goal:
                if self.distance_to_goal_format == "progress":
                    distance_to_goal_mean = np.zeros_like(state_mean[-1])
                    distance_to_goal_std = np.ones_like(state_std[-1])
                    distance_to_goal_norm_max = np.ones_like(state_norm_max[-1])
                    distance_to_goal_norm_min = np.zeros_like(state_norm_min[-1])
                elif self.distance_to_goal_format == "distance":
                    distance_to_goal_mean = state_mean[-1]  # [3]
                    distance_to_goal_std = state_std[-1] * 2  # [3]
                    distance_to_goal_norm_max = state_norm_max[-1] * 2  # [3]
                    distance_to_goal_norm_min = state_norm_min[-1] * 2  # [3]
                else:
                    raise ValueError(
                        f"Invalid distance to goal format: {self.distance_to_goal_format}"
                    )

            # Squeeze the data from size [T, 3, H, W] to [T*3, H, W]
            T, C, H, W = gt_track_residual.shape
            gt_track_future = gt_track_future.reshape(T * C, H, W)
            gt_track_future_visib = gt_track_future_visib.reshape(T * 1, H, W)
            gt_track_history = gt_track_history.reshape(T * C, H, W)
            gt_track_history_visib = gt_track_history_visib.reshape(T * 1, H, W)
            gt_track_history_valid = gt_track_history_valid.reshape(T * C, H, W)

            # Change the residual shape
            # if self.load_distance_to_goal:
            #     gt_track_residual = np.concatenate(
            #         [gt_track_residual, gt_track_distance_to_goal], axis=0
            #     )  # [T+1, 3, H, W]
            #     gt_track_future_valid = np.concatenate(
            #         [gt_track_future_valid, gt_track_future_valid[-1:]], axis=0
            #     )  # [T+1, 1, H, W]
            #     gt_track_future_valid = gt_track_future_valid.reshape((T + 1) * C, H, W)
            #     gt_track_residual = gt_track_residual.reshape((T + 1) * C, H, W)
            #     state_mean = state_mean.reshape((T + 1) * C)
            #     state_std = state_std.reshape((T + 1) * C)
            #     state_norm_max = state_norm_max.reshape((T + 1) * C)
            #     state_norm_min = state_norm_min.reshape((T + 1) * C)
            # else:
            gt_track_residual = gt_track_residual.reshape(T * C, H, W)
            gt_track_future_valid = gt_track_future_valid.reshape(T * C, H, W)
            state_mean = state_mean.reshape(T * C)
            state_std = state_std.reshape(T * C)
            state_norm_max = state_norm_max.reshape(T * C)
            state_norm_min = state_norm_min.reshape(T * C)

            # Load the distance to goal data
            data.update(
                {
                    # Add the state data
                    "state_mean": state_mean,  # [T*C]
                    "state_std": state_std,  # [T*C]
                    "state_norm_max_bound": state_norm_max,  # [T*C]
                    "state_norm_min_bound": state_norm_min,  # [T*C]
                    # Add the track data
                    "start_state": gt_track_init,  # [C, H, W]
                    "state_color": gt_track_color,  # [C, H, W]
                    # Add the history data; all state are meassured in the first frame of the clip!
                    "history_state": gt_track_history,  # [T*C, H, W]
                    "history_state_valid": gt_track_history_valid,  # [T*C, H, W]
                    "history_state_visib": gt_track_history_visib,  # [T*C, 1, H, W]
                    # Add the residual data; all state are meassured in the first frame of the clip!
                    "gt_state": gt_track_future,  # [T*C, H, W]
                    "gt_state_residual": gt_track_residual,  # [T*C, H, W]
                    "state_visib": gt_track_future_visib,  # [T*C, H, W]
                    "state_valid": gt_track_future_valid,  # [T*C, H, W]
                    "state_timestamp": np.array([1.0]),
                }
            )
            if self.load_distance_to_goal:
                data.update(
                    {
                        "distance_to_goal_mean": distance_to_goal_mean,
                        "distance_to_goal_std": distance_to_goal_std,
                        "distance_to_goal_norm_max_bound": distance_to_goal_norm_max,
                        "distance_to_goal_norm_min_bound": distance_to_goal_norm_min,
                        "gt_distance_to_goal": gt_track_distance_to_goal[
                            0
                        ],  # [C, H, W]
                    }
                )

        for k, v in data.items():
            data[k] = np.array(v).astype(np.float32)
        return data

    def _build_raymap_history(
        self,
        sample_idx,
        sample_name,
        frame_idx,
        T_wc_list,
        camera_intr,
        valid_frame_range,
        input_size=(224, 224),
        patch_size=(14, 14),
    ):
        dataset_name = self.dataset_categories[sample_idx]
        history_T_ch = []
        history_raymap = []
        T_wc = T_wc_list[frame_idx - valid_frame_range[0]]
        patch_size_h, patch_size_w = patch_size
        input_size_h, input_size_w = input_size
        factor_h = patch_size_h / input_size_h
        factor_w = patch_size_w / input_size_w
        for i in range(
            frame_idx - (self.horizon - 1) * (self.fps // DEFAULT_FPS),
            frame_idx + (self.fps // DEFAULT_FPS),
            self.fps // DEFAULT_FPS,
        ):
            _i = max(i, valid_frame_range[0])
            _i = min(_i, valid_frame_range[1] - 1)
            T_wh = T_wc_list[_i - valid_frame_range[0]]
            T_ch = np.linalg.inv(T_wc) @ T_wh
            camera_intr_patch = camera_intr.copy()
            # Scale the camera intrinsics to the patch size
            camera_intr_patch[0, 0] *= factor_w
            camera_intr_patch[0, 2] *= factor_w
            camera_intr_patch[1, 1] *= factor_h
            camera_intr_patch[1, 2] *= factor_h
            raymap_h = AriaUtils.get_ray_map(
                T_ch,
                camera_intr_patch,
                patch_size_h,
                patch_size_w,
                in_pluecker=True,
            )
            raymap_h = np.transpose(raymap_h, (2, 0, 1))  # (6, H, W)
            history_T_ch.append(T_ch)
            history_raymap.append(raymap_h)
        history_T_ch = np.stack(history_T_ch)  # (T, 4, 4)
        history_raymap = np.stack(history_raymap)  # (T, 6, H, W)
        if dataset_name == "egodex":
            horizon = history_T_ch.shape[0]
            half_horizon = horizon // 2
            history_info = {
                "history_T_ch": history_T_ch,
                "history_raymap": history_raymap,
            }
            for k in ["history_T_ch", "history_raymap"]:
                history_val = history_info[k][-half_horizon:]
                history_val_first = history_val[:-1]
                history_val_last = history_val[1:]
                history_val_interp = [history_val[0], history_val[0]]
                for h_f, h_l in zip(history_val_first, history_val_last):
                    history_val_interp.extend([h_f, h_l])
                history_val_interp.append(history_val[-1])
                history_val_interp = np.stack(history_val_interp, axis=0)
                history_info[k] = history_val_interp
                assert (
                    history_val_interp.shape[0] == horizon
                ), f"History value interpolation shape {history_val_interp.shape[0]} is not equal to horizon {horizon}"
            history_T_ch = history_info["history_T_ch"]
            history_raymap = history_info["history_raymap"]
        return history_T_ch, history_raymap

    def _build_visual_feature_history(self, sample_idx, sample_name, frame_idx):
        frame = frame_idx
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(
            self.data_dir, dataset_name, f"{dataset_name.upper()}_release"
        )
        suffix = (
            ".npz" if self.feature_extractor == "dinov3" else ""
        )  # Hack to fix pre-save bug ...

        if dataset_name == "hoi4d":
            sample, clip_idx = sample_name.split(".")[0], sample_name.split(".")[1]
            visual_feature_fpath = os.path.join(
                dataset_dir,
                sample,
                f"{self.feature_extractor}_visual_feature_by_clip_224",
                f"{sample.replace('/', '_')}-{clip_idx}{suffix}.npy",
            )
        else:
            _sample_name = sample_name.split("/")
            sample = "/".join(_sample_name[:-1])
            clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
            visual_feature_fpath = os.path.join(
                dataset_dir,
                sample,
                f"{self.feature_extractor}_visual_feature_by_clip_224",
                f"{clip_idx}{suffix}.npy",
            )
        if not os.path.exists(visual_feature_fpath):
            print(
                f"Visual feature data not found for {sample_name} at frame {frame_idx}"
            )
            return None

        visual_observation = np.load(visual_feature_fpath)

        assert (
            visual_observation.shape[-1] == self.dinov3_visual_dim
        ), (
            f"Visual feature last dim {visual_observation.shape[-1]} != "
            f"dinov3_visual_dim={self.dinov3_visual_dim} ({visual_feature_fpath})"
        )

        assert (
            visual_observation.shape[0] == valid_frame_range[1] - valid_frame_range[0]
        ), f"Visual observation shape {visual_observation.shape} is not equal to horizon {self.horizon}"

        # Build the history visual feature
        frame_indices = np.arange(
            frame - (self.horizon - 1) * (self.fps // DEFAULT_FPS),
            frame + (self.fps // DEFAULT_FPS),
            self.fps // DEFAULT_FPS,
        )

        # Clip indices to valid range and convert to array indices
        clipped_indices = np.clip(
            frame_indices, valid_frame_range[0], valid_frame_range[1] - 1
        )
        array_indices = clipped_indices - valid_frame_range[0]
        history_visual_observation = visual_observation[array_indices]  # [H, 196, 768]

        # For history: keep the full shape [H, 196, 768]
        visual_feature_data = {
            "history_patch": history_visual_observation,
        }

        # For goal configuration
        goal_horizon = self.horizon - 1
        if dataset_name == "egodex":
            # Match the last frame of future_tracks in _build_track_data
            # future_tracks[:half_horizon] has last frame at frame + (half_horizon-1)*step
            goal_horizon = (self.horizon // 2) - 1
        frame_index_goal = np.clip(
            frame + (goal_horizon * (self.fps // DEFAULT_FPS)),
            valid_frame_range[0],
            valid_frame_range[1] - 1,
        )
        array_index_goal = frame_index_goal - valid_frame_range[0]
        visual_observation_goal = visual_observation[array_index_goal]  # [196, 768]
        visual_feature_data["goal_patch"] = visual_observation_goal
        visual_feature_data["goal_patch"] = visual_feature_data["goal_patch"] / 5.0

        if dataset_name == "egodex":
            horizon = history_visual_observation.shape[0]
            half_horizon = horizon // 2
            # Note: The interpolation formula (2*n + 1) only works correctly for odd horizon values.
            # For even horizon, the output will be horizon + 1 elements, which will fail the assertion.
            # Currently, the project uses horizon=15 (odd), so this is not an issue.
            for k in ["history_patch"]:
                # Only use the last half_horizon elements, matching track data processing
                feature_val = visual_feature_data[k][-half_horizon:]
                feature_val_first = feature_val[:-1]  # [0, 1, 2, ..., half_horizon-2]
                feature_val_last = feature_val[1:]  # [1, 2, 3, ..., half_horizon-1]
                feature_val_interp = [feature_val[0], feature_val[0]]
                for f_f, f_l in zip(feature_val_first, feature_val_last):
                    feature_val_interp.extend([f_f, f_l])
                feature_val_interp.append(feature_val[-1])
                feature_val_interp = np.stack(feature_val_interp, axis=0)
                visual_feature_data[k] = feature_val_interp
                assert (
                    feature_val_interp.shape[0] == horizon
                ), f"Visual feature value interpolation shape {feature_val_interp.shape[0]} is not equal to horizon {horizon}"
        return visual_feature_data

    def _build_track_data(
        self, sample_idx, sample_name, frame_idx, T_cam0_cam, track_by_clip=False
    ):
        # Acquire the saved coords data
        frame = frame_idx
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(
            self.data_dir, dataset_name, f"{dataset_name.upper()}_release"
        )
        track_suffix = "_by_clip" if track_by_clip else ""
        if not self.build_track_online:
            assert track_by_clip, "Track by clip is required for offline building!"
            if dataset_name == "hoi4d":
                sample_dir, clip_idx = (
                    sample_name.split(".")[0],
                    sample_name.split(".")[1],
                )
            else:
                _sample_name = sample_name.split("/")
                sample_dir = "/".join(_sample_name[:-1])
            track_data_fpath = os.path.join(
                dataset_dir,
                sample_dir,
                f"tapip3d_tracks{track_suffix}_224_{self.track_patch_size}",
                f"{frame_idx:05d}.npz",
            )
            track_data = np.load(track_data_fpath)
            # npz to dict
            track_data = dict(track_data)
            if track_by_clip:
                assert np.allclose(
                    track_data["T_cam0_cam"], np.eye(4), atol=1e-3
                ), "T_cam0_cam is not the identity matrix!"
        else:
            assert (
                not track_by_clip
            ), "Track by clip is not supported for online building!"
            if dataset_name == "hoi4d":
                sample_dir, clip_idx = (
                    sample_name.split(".")[0],
                    sample_name.split(".")[1],
                )
                track_fpath = os.path.join(
                    dataset_dir,
                    sample_dir,
                    f"tapip3d_tracks_224_{self.track_patch_size}",
                    f"{sample_dir.replace('/', '_')}-{clip_idx}.npz",
                )
            else:
                _sample_name = sample_name.split("/")
                sample = "/".join(_sample_name[:-1])
                clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
                track_fpath = os.path.join(
                    dataset_dir,
                    sample,
                    f"tapip3d_tracks_224_{self.track_patch_size}",
                    f"{clip_idx}.npz",
                )

            T_cam_cam0 = np.linalg.inv(T_cam0_cam)

            if not os.path.exists(track_fpath):
                print(f"Track data not found for {sample_name} at frame {frame_idx}")
                return None
            full_track_data = np.load(track_fpath)
            coords = full_track_data["tracks"]
            visibs = full_track_data["tracks_visib"]
            query_point = full_track_data["query_point"]
            valid = full_track_data["tracks_valid"][None].repeat(
                coords.shape[0], axis=0
            )
            tracks_color = full_track_data["tracks_color"][None].repeat(
                coords.shape[0], axis=0
            )

            # Start building the window-based track data (vectorized)
            # Generate all frame indices at once
            frame_indices = np.arange(
                frame - (self.horizon - 1) * (self.fps // DEFAULT_FPS),
                frame + (self.horizon * (self.fps // DEFAULT_FPS)),
                self.fps // DEFAULT_FPS,
            )
            # Clip indices to valid range and convert to array indices
            clipped_indices = np.clip(
                frame_indices, valid_frame_range[0], valid_frame_range[1] - 1
            )
            array_indices = clipped_indices - valid_frame_range[0]
            # Extract all data at once using advanced indexing
            all_tracks = coords[array_indices]  # [num_frames, num_points, 3]
            all_tracks_visib = visibs[array_indices]  # [num_frames, num_points]
            all_tracks_valid = valid[array_indices]  # [num_frames, num_points]
            all_tracks_color = tracks_color[
                array_indices
            ]  # [num_frames, num_points, 3]

            # Apply transformation to all tracks at once
            # Reshape to [num_frames * num_points, 3] for batch transformation
            num_frames, num_points = all_tracks.shape[:2]
            all_tracks_flat = all_tracks.reshape(-1, 3)
            all_tracks_transformed = DatasetUtils.transform_points(
                all_tracks_flat, T_cam_cam0
            )  # From initial frame to current frame, historical reason ...
            all_tracks = all_tracks_transformed.reshape(num_frames, num_points, 3)

            # Separate the history and future tracks
            # Note: all_tracks, all_tracks_visib, etc. are already in the correct shape [num_frames, ...]
            history_tracks = all_tracks[: self.horizon]
            history_tracks_visib = all_tracks_visib[: self.horizon]
            history_tracks_valid = all_tracks_valid[: self.horizon]

            future_tracks = all_tracks[self.horizon - 1 :]
            future_tracks_visib = all_tracks_visib[self.horizon - 1 :]
            future_tracks_valid = all_tracks_valid[self.horizon - 1 :]
            future_tracks_color = all_tracks_color[self.horizon - 1 :]

            # Build the track data
            track_data = {
                "valid": future_tracks_valid,
                "tracks": future_tracks,
                "visib": future_tracks_visib,
                "valid": future_tracks_valid,
                "history_tracks": history_tracks,
                "history_visib": history_tracks_visib,
                "history_valid": history_tracks_valid,
                "state": query_point,
                "color": future_tracks_color[0],
                "T_cam0_cam": T_cam0_cam,
            }
        if dataset_name == "egodex":

            horizon = track_data["tracks"].shape[0]
            # Only select the first half of the horizon
            # Note: The interpolation formula (2*n + 1) only works correctly for odd horizon values.
            # For even horizon, the output will be horizon + 1 elements, which will fail the assertion.
            # Currently, the project uses horizon=15 (odd), so this is not an issue.
            half_horizon = horizon // 2
            for k in [
                "valid",
                "tracks",
                "visib",
                "history_tracks",
                "history_visib",
                "history_valid",
            ]:
                if "history" in k:
                    track_val = track_data[k][-half_horizon:]
                else:
                    track_val = track_data[k][:half_horizon]
                track_val_first = track_val[:-1]  # [0, 1, 2, 3, 4, 5]
                track_val_last = track_val[1:]  # [1, 2, 3, 4, 5, 6]

                track_val_interp = [track_val[0], track_val[0]]
                for t_f, t_l in zip(track_val_first, track_val_last):
                    track_val_interp.extend([t_f, t_l])
                track_val_interp.append(track_val[-1])
                track_val_interp = np.stack(track_val_interp, axis=0)  # [H, N]
                track_data[k] = track_val_interp
                assert (
                    track_val_interp.shape[0] == horizon
                ), "Track value interpolation shape is not equal to horizon"

            history_tracks_last = track_data["history_tracks"][-1]
            future_tracks_first = track_data["tracks"][0]

            # Check if they are all close
            assert np.allclose(
                history_tracks_last, future_tracks_first, atol=1e-3
            ), "History tracks and future tracks are not all close"

        # Compute the distance to goal
        track_data.update(
            self._build_track_distance_to_goal_data(
                sample_idx, sample_name, frame_idx, T_cam0_cam
            )
        )
        return track_data

    def _build_track_distance_to_goal_data(
        self, sample_idx, sample_name, frame_idx, T_cam0_cam
    ):
        # Acquire the saved coords data
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(
            self.data_dir, dataset_name, f"{dataset_name.upper()}_release"
        )
        if self.distance_to_goal_format == "distance":
            if dataset_name == "hoi4d":
                sample_dir, clip_idx = (
                    sample_name.split(".")[0],
                    sample_name.split(".")[1],
                )
                track_fpath = os.path.join(
                    dataset_dir,
                    sample_dir,
                    f"tapip3d_tracks_224_{self.track_patch_size}",
                    f"{sample_dir.replace('/', '_')}-{clip_idx}.npz",
                )
            else:
                _sample_name = sample_name.split("/")
                sample = "/".join(_sample_name[:-1])
                clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
                track_fpath = os.path.join(
                    dataset_dir,
                    sample,
                    f"tapip3d_tracks_224_{self.track_patch_size}",
                    f"{clip_idx}.npz",
                )
            full_track_data = np.load(track_fpath)
            tracks = full_track_data["tracks"]  # [T, N, 3]
            future_horizon = self.horizon - 1
            if dataset_name == "egodex":
                # Match the last frame of future_tracks in _build_track_data
                # future_tracks[:half_horizon] has last frame at frame + (half_horizon-1)*step
                future_horizon = (self.horizon // 2) - 1

            # Current frame index and goal frame index (goal = current + future_horizon)
            curr_idx = frame_idx + future_horizon - valid_frame_range[0]
            goal_idx = tracks.shape[0] - 1
            curr_idx = np.clip(curr_idx, 0, tracks.shape[0] - 1)
            goal_idx = np.clip(goal_idx, 0, tracks.shape[0] - 1)

            curr_track, goal_track = tracks[curr_idx], tracks[goal_idx]
            distance_to_goal = goal_track - curr_track  # [N, 3]

        elif self.distance_to_goal_format == "progress":
            # Progress in [0, 1]: 0 = first frame of clip, 1 = last frame of clip
            num_frames = valid_frame_range[1] - valid_frame_range[0] + 1
            full_progress = np.arange(num_frames) / max(1, num_frames)

            future_horizon = self.horizon - 1
            if dataset_name == "egodex":
                # Match the last frame of future_tracks in _build_track_data
                # future_tracks[:half_horizon] has last frame at frame + (half_horizon-1)*step
                future_horizon = (self.horizon // 2) - 1
            # Index of the goal frame (current + future_horizon) in the clip
            i = frame_idx + future_horizon - valid_frame_range[0]
            i = np.clip(i, 0, full_progress.shape[0] - 1)
            distance_to_goal = 1 - (
                np.ones((self.track_patch_size**2, 3)) * full_progress[i, None]
            )
        else:
            raise ValueError(
                f"Invalid distance to goal format: {self.distance_to_goal_format}"
            )

        track_data = {
            "distance_to_goal": distance_to_goal,
        }  # Smaller is better
        return track_data

    def _process_visual_feature(self, visual_feature_data):
        visual_history_patch = visual_feature_data["history_patch"]
        visual_goal_patch = visual_feature_data["goal_patch"]
        return visual_history_patch, visual_goal_patch

    def _process_hand_data(self, hand_data):
        if hand_data is None:
            # print(f"Hand data is None, setting all to -1e3")
            hand_traj = np.ones((self.horizon, self.ACTION_DIM)) * -1e3
            hand_valid = np.zeros((self.horizon, self.ACTION_DIM))
            hand_traj_timestamps = np.zeros((self.horizon,))
            hand_traj_history = np.ones((self.horizon, self.ACTION_DIM)) * -1e3
            hand_valid_history = np.zeros((self.horizon, self.ACTION_DIM))
            hand_traj_timestamps_history = np.zeros((self.horizon,))
            start_pos = np.ones((self.ACTION_DIM)) * -1e3
        else:
            hand_traj = hand_data["trajectory"]
            hand_valid = hand_data["valid"]
            start_timestamp = hand_data["timestamps"][self.horizon]
            hand_traj_timestamps = (
                hand_data["timestamps"][self.horizon :] - start_timestamp
            )
            hand_traj_history = hand_data["history_trajectory"]
            hand_valid_history = hand_data["history_valid"]
            hand_traj_timestamps_history = (
                hand_data["history_timestamps"][self.horizon :] - start_timestamp
            )
            # start_pos = hand_data["state"]
            hand_traj_history = np.nan_to_num(hand_traj_history, nan=0.0)
            hand_traj = np.nan_to_num(hand_traj, nan=0.0)

            # Make sure the starting position is the same for both hands
            start_pos = hand_traj[0].copy()
            hand_traj_history[-1] = hand_traj[0].copy()

            # Futher process the GT hand trajectory for stable action loss
            left_hand_traj, right_hand_traj = (
                hand_traj[:, : self.ACTION_DIM // 2],
                hand_traj[:, self.ACTION_DIM // 2 :],
            )
            left_hand_traj_valid, right_hand_traj_valid = (
                hand_valid[:, : self.ACTION_DIM // 2],
                hand_valid[:, self.ACTION_DIM // 2 :],
            )
            left_hand_traj_history, right_hand_traj_history = (
                hand_traj_history[:, : self.ACTION_DIM // 2],
                hand_traj_history[:, self.ACTION_DIM // 2 :],
            )

            if right_hand_traj_valid[:, 0].sum() < 2:
                right_hand_traj = left_hand_traj.copy()
                right_hand_traj_valid = left_hand_traj_valid.copy()
                right_hand_traj_history = left_hand_traj_history.copy()

            if left_hand_traj_valid[:, 0].sum() < 2:
                left_hand_traj = right_hand_traj.copy()
                left_hand_traj_valid = right_hand_traj_valid.copy()
                left_hand_traj_history = right_hand_traj_history.copy()

            hand_traj = np.concatenate([left_hand_traj, right_hand_traj], axis=1)
            hand_valid = np.concatenate(
                [left_hand_traj_valid, right_hand_traj_valid], axis=1
            )
            hand_traj_history = np.concatenate(
                [left_hand_traj_history, right_hand_traj_history], axis=1
            )
            start_pos = hand_traj_history[-1]

            # if right_hand_traj_valid.sum() < 2:
            #     right_hand_traj = left_hand_traj[0][None, :].repeat(
            #         right_hand_traj.shape[0], axis=0
            #     )

            # if left_hand_traj_valid.sum() < 2:
            #     left_hand_traj = right_hand_traj[0][None, :].repeat(
            #         left_hand_traj.shape[0], axis=0
            #     )

            # hand_traj = np.concatenate([left_hand_traj, right_hand_traj], axis=1)

        # If no finger tips, remove the intermediate points
        if not self.load_finger_tips:
            closure = 0.5 if hand_data is not None else -1e3
            start_pos_orig = start_pos
            hand_traj, hand_valid, start_pos = self._postprocess_hand_trajectory(
                hand_traj, hand_valid, start_pos_orig, closure=closure
            )
            hand_traj_history, hand_valid_history, _ = (
                self._postprocess_hand_trajectory(
                    hand_traj_history,
                    hand_valid_history,
                    start_pos_orig,
                    closure=closure,
                )
            )

        # Do horizon truncation
        hand_traj = hand_traj[: self.horizon]
        hand_valid = hand_valid[: self.horizon]
        hand_traj_timestamps = hand_traj_timestamps[: self.horizon]

        hand_traj_history = hand_traj_history[-self.horizon :]
        hand_valid_history = hand_valid_history[-self.horizon :]
        hand_traj_timestamps_history = hand_traj_timestamps_history[-self.horizon :]

        # Upsample the hand trajectory to the action chunk size
        hand_traj, hand_valid, hand_traj_timestamps = self._upsample_hand_trajectory(
            hand_traj, hand_valid, hand_traj_timestamps, self.action_chunk_size
        )
        return (
            hand_traj,  # [T, ACTION_DIM]
            hand_valid,  # [T, ACTION_DIM]
            hand_traj_timestamps,  # [T, 1]
            hand_traj_history,  # [T, ACTION_DIM]
            hand_valid_history,  # [T, ACTION_DIM]
            hand_traj_timestamps_history,  # [T, 1]
            start_pos,  # [ACTION_DIM]
        )

    def _upsample_hand_trajectory(
        self, hand_traj, hand_valid, hand_traj_timestamps, action_chunk_size
    ):

        assert (
            hand_traj.shape[0] == hand_valid.shape[0] == hand_traj_timestamps.shape[0]
        )
        assert hand_traj.shape[1] == self.WRIST_ACTION_DIM
        assert self.load_finger_tips == False

        def _upsample_one_hand(one_hand_traj, one_hand_valid, one_hand_traj_timestamps):
            one_hand_traj_timestamps = np.arange(one_hand_traj.shape[0])
            translation, closure, r6d = (
                one_hand_traj[:, :3],
                one_hand_traj[:, 3],
                one_hand_traj[:, -6:],
            )  # [T, 3], [T, 1], [T, 6]
            upsampled_translation, upsampled_timestamps, _ = (
                DatasetUtils.interpolate_trajectory(
                    one_hand_traj_timestamps, translation, num_points=action_chunk_size
                )
            )
            upsampled_closure = DatasetUtils.interpolate_closure(
                one_hand_traj_timestamps, closure, num_points=action_chunk_size
            )[0]
            upsampled_r6d = DatasetUtils.interpolate_rotation(
                one_hand_traj_timestamps,
                r6d,
                num_points=action_chunk_size,
                rotation_format="r6d",
            )[0]
            upsampled_hand_traj = np.concatenate(
                [upsampled_translation, upsampled_closure[:, None], upsampled_r6d],
                axis=-1,
            )
            updsampled_hand_valid = DatasetUtils.interpolate_closure(
                one_hand_traj_timestamps,
                one_hand_valid[:, 0],
                num_points=action_chunk_size,
            )[0]

            updsampled_hand_valid = updsampled_hand_valid[:, None].repeat(
                one_hand_traj.shape[1], axis=1
            )  # [T, action_chunk_size]

            return upsampled_hand_traj, updsampled_hand_valid, upsampled_timestamps

        if action_chunk_size > self.horizon:
            assert action_chunk_size % self.horizon == 0
            left_hand_traj, left_hand_valid, left_hand_traj_timestamps = (
                _upsample_one_hand(
                    hand_traj[:, : self.WRIST_ACTION_DIM // 2],
                    hand_valid[:, : self.WRIST_ACTION_DIM // 2],
                    hand_traj_timestamps,
                )
            )
            right_hand_traj, right_hand_valid, right_hand_traj_timestamps = (
                _upsample_one_hand(
                    hand_traj[:, self.WRIST_ACTION_DIM // 2 :],
                    hand_valid[:, self.WRIST_ACTION_DIM // 2 :],
                    hand_traj_timestamps,
                )
            )
            hand_traj = np.concatenate([left_hand_traj, right_hand_traj], axis=1)
            hand_valid = np.concatenate([left_hand_valid, right_hand_valid], axis=1)
            hand_traj_timestamps = (
                left_hand_traj_timestamps + right_hand_traj_timestamps
            ) / 2
        else:
            hand_traj, hand_valid, hand_traj_timestamps = (
                hand_traj,
                hand_valid,
                hand_traj_timestamps,
            )
        return hand_traj, hand_valid, hand_traj_timestamps

    def _postprocess_hand_trajectory(
        self, hand_traj, hand_valid, start_pos, closure=0.5
    ):
        """
        hand_traj: [T, ACTION_DIM]
        hand_valid: [T, ACTION_DIM]
        start_pos: [ACTION_DIM]
        closure: float
        return:
            hand_traj: [T, WRIST_ACTION_DIM]
            hand_valid: [T, WRIST_ACTION_DIM]
            start_pos: [WRIST_ACTION_DIM]
        """
        left_hand_traj, right_hand_traj = (
            hand_traj[:, : self.ACTION_DIM // 2],
            hand_traj[:, self.ACTION_DIM // 2 :],
        )
        left_hand_valid, right_hand_valid = (
            hand_valid[:, : self.ACTION_DIM // 2],
            hand_valid[:, self.ACTION_DIM // 2 :],
        )

        left_trans, left_r6d = left_hand_traj[:, :3], left_hand_traj[:, -6:]
        right_trans, right_r6d = right_hand_traj[:, :3], right_hand_traj[:, -6:]

        left_closure = np.ones((hand_traj.shape[0], 1)) * closure
        right_closure = np.ones((hand_traj.shape[0], 1)) * closure

        left_hand_traj = np.concatenate([left_trans, left_closure, left_r6d], axis=-1)
        right_hand_traj = np.concatenate(
            [right_trans, right_closure, right_r6d], axis=-1
        )

        # valid：trans 3 + closure 1 + r6d 6
        left_trans_valid = left_hand_valid[:, :3]
        left_r6d_valid = left_hand_valid[:, -6:]
        left_closure_valid = left_trans_valid[:, :1]

        right_trans_valid = right_hand_valid[:, :3]
        right_r6d_valid = right_hand_valid[:, -6:]
        right_closure_valid = right_trans_valid[:, :1]

        left_valid = np.concatenate(
            [left_trans_valid, left_closure_valid, left_r6d_valid], axis=-1
        )
        right_valid = np.concatenate(
            [right_trans_valid, right_closure_valid, right_r6d_valid], axis=-1
        )

        hand_traj = np.concatenate([left_hand_traj, right_hand_traj], axis=1)
        hand_valid = np.concatenate([left_valid, right_valid], axis=1)
        assert hand_traj.shape[1] == self.WRIST_ACTION_DIM
        assert hand_valid.shape[1] == self.WRIST_ACTION_DIM

        left_start_pos, right_start_pos = (
            start_pos[: self.ACTION_DIM // 2],
            start_pos[self.ACTION_DIM // 2 :],
        )
        left_start_pos_trans, left_start_pos_r6d = (
            left_start_pos[:3],
            left_start_pos[-6:],
        )
        right_start_pos_trans, right_start_pos_r6d = (
            right_start_pos[:3],
            right_start_pos[-6:],
        )
        left_start_pos = np.concatenate(
            [left_start_pos_trans, np.array([closure]), left_start_pos_r6d], axis=0
        )
        right_start_pos = np.concatenate(
            [right_start_pos_trans, np.array([closure]), right_start_pos_r6d], axis=0
        )
        start_pos = np.concatenate([left_start_pos, right_start_pos], axis=0)
        assert start_pos.shape[0] == self.WRIST_ACTION_DIM

        return hand_traj, hand_valid, start_pos

    def _process_track_data(self, track_data):
        # gt_track_3d = track_data["pred_track_3d_in_camcurr"]
        # gt_track_color = track_data["pred_track_color"]
        # gt_track_timestamp = track_data["pred_track_timestamp"]
        # gt_track_visibility = track_data["pred_visibility"]
        # gt_track_visibility[0].fill(True)
        if track_data is None:
            gt_track_future = (
                np.ones(
                    (
                        self.flow_horizon,
                        self.STATE_DIM,
                        self.track_patch_size,
                        self.track_patch_size,
                    )
                )
                * 1e-3
            )
            gt_track_future_valid = np.zeros(
                (self.flow_horizon, 3, self.track_patch_size, self.track_patch_size)
            )
            gt_track_future_visib = np.zeros(
                (self.flow_horizon, 1, self.track_patch_size, self.track_patch_size)
            )
            gt_track_history = (
                np.ones(
                    (
                        self.flow_horizon,
                        self.STATE_DIM,
                        self.track_patch_size,
                        self.track_patch_size,
                    )
                )
                * -1e3
            )
            gt_track_history_valid = np.zeros(
                (self.flow_horizon, 3, self.track_patch_size, self.track_patch_size)
            )
            gt_track_history_visib = np.zeros(
                (self.flow_horizon, 1, self.track_patch_size, self.track_patch_size)
            )
            gt_track_init = (
                np.ones((self.STATE_DIM, self.track_patch_size, self.track_patch_size))
                * -1e3
            )
            gt_track_color = np.zeros(
                (self.STATE_DIM, self.track_patch_size, self.track_patch_size)
            )
            gt_track_distance_to_goal = np.zeros(
                (1, self.track_patch_size, self.track_patch_size, 3)
            )
        else:
            track_data = {k: v for k, v in track_data.items()}
            tracks = rearrange(track_data["tracks"], "t n c -> (t n) c")
            T_cam0_cam = track_data["T_cam0_cam"]
            tracks = DatasetUtils.transform_points(tracks, T_cam0_cam)
            track_data["tracks"] = rearrange(
                tracks, "(t n) c -> t n c", t=self.flow_horizon
            )  # Transform to the first frame; everything is meassured in the first frame!

            history_tracks = rearrange(track_data["history_tracks"], "t n c -> (t n) c")
            history_tracks = DatasetUtils.transform_points(history_tracks, T_cam0_cam)
            track_data["history_tracks"] = rearrange(
                history_tracks, "(t n) c -> t n c", t=self.flow_horizon
            )  # Transform to the first frame; everything is meassured in the first frame!

            # State data
            gt_track_color = track_data["color"][None]  # [1, N, 3]
            gt_track_3d_init = track_data["state"][None]  # [1, N, 3]
            gt_track_valid_init = track_data["valid"][:1]  # [1, N]

            # Future data
            gt_track_future = track_data["tracks"]  # [T, N, 3]
            gt_track_future_valid = track_data["valid"].astype(np.float32)  # [T, N]
            gt_track_future_visib = track_data["visib"].astype(np.float32)  # [T, N]
            gt_track_future_valid = gt_track_future_valid * gt_track_valid_init

            # History data
            gt_track_history = track_data["history_tracks"]  # [T, N, 3]
            gt_track_history_visib = track_data["history_visib"].astype(
                np.float32
            )  # [T, N]
            gt_track_history_valid = track_data["history_valid"].astype(
                np.float32
            )  # [T, N]
            gt_track_history_valid = gt_track_history_valid * gt_track_valid_init

            # Distance to goal
            gt_track_distance_to_goal = track_data["distance_to_goal"][
                None
            ]  # [1, N, 3]
            track_patch_size, track_horizon = (
                int(gt_track_history.shape[1] ** 0.5),
                gt_track_history.shape[0],
            )

            # Reshape the track data
            gt_track_future = gt_track_future.reshape(
                track_horizon, track_patch_size, track_patch_size, 3
            )
            gt_track_future_valid = gt_track_future_valid.reshape(
                track_horizon, track_patch_size, track_patch_size, 1
            )
            gt_track_future_visib = gt_track_future_visib.reshape(
                track_horizon, track_patch_size, track_patch_size, 1
            )

            gt_track_history = gt_track_history.reshape(
                track_horizon, track_patch_size, track_patch_size, 3
            )
            gt_track_history_visib = gt_track_history_visib.reshape(
                track_horizon, track_patch_size, track_patch_size, 1
            )
            gt_track_history_valid = gt_track_history_valid.reshape(
                track_horizon, track_patch_size, track_patch_size, 1
            )
            gt_track_color = gt_track_color.reshape(
                1, track_patch_size, track_patch_size, 3
            )
            gt_track_3d_init = gt_track_3d_init.reshape(
                1, track_patch_size, track_patch_size, 3
            )
            gt_track_distance_to_goal = gt_track_distance_to_goal.reshape(
                1, track_patch_size, track_patch_size, 3
            )
            # [T, P, P, 1]
            gt_track_future_valid = np.repeat(gt_track_future_valid, 3, axis=-1)
            gt_track_history_valid = np.repeat(gt_track_history_valid, 3, axis=-1)

            # Do the permutatation in numpy
            gt_track_future = np.transpose(gt_track_future, (0, 3, 1, 2))
            gt_track_future_valid = np.transpose(gt_track_future_valid, (0, 3, 1, 2))
            gt_track_future_visib = np.transpose(gt_track_future_visib, (0, 3, 1, 2))
            gt_track_history = np.transpose(gt_track_history, (0, 3, 1, 2))
            gt_track_history_visib = np.transpose(gt_track_history_visib, (0, 3, 1, 2))
            gt_track_history_valid = np.transpose(gt_track_history_valid, (0, 3, 1, 2))
            gt_track_init = np.transpose(gt_track_3d_init, (0, 3, 1, 2))
            gt_track_color = np.transpose(gt_track_color, (0, 3, 1, 2))
            gt_track_distance_to_goal = np.transpose(
                gt_track_distance_to_goal, (0, 3, 1, 2)
            )

            # Interpolate the track data if the patch size is different
            if self.track_patch_size != track_patch_size:
                gt_track_future = torch.from_numpy(gt_track_future)
                gt_track_future_valid = torch.from_numpy(gt_track_future_valid)
                gt_track_future_visib = torch.from_numpy(gt_track_future_visib)
                gt_track_history = torch.from_numpy(gt_track_history)
                gt_track_history_visib = torch.from_numpy(gt_track_history_visib)
                gt_track_history_valid = torch.from_numpy(gt_track_history_valid)
                gt_track_init = torch.from_numpy(gt_track_init)
                gt_track_color = torch.from_numpy(gt_track_color)
                gt_track_distance_to_goal = torch.from_numpy(gt_track_distance_to_goal)

                # Do the interpolation
                gt_track_future = F.interpolate(
                    gt_track_future,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )
                gt_track_future_valid = F.interpolate(
                    gt_track_future_valid,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )
                gt_track_future_visib = F.interpolate(
                    gt_track_future_visib,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )

                # Do the interpolation
                gt_track_history = F.interpolate(
                    gt_track_history,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )  # [T, D, H, W]
                gt_track_history_valid = F.interpolate(
                    gt_track_history_valid,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )
                gt_track_history_visib = F.interpolate(
                    gt_track_history_visib,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )

                gt_track_init = F.interpolate(
                    gt_track_init,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="bilinear",
                )
                gt_track_color = F.interpolate(
                    gt_track_color,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="bilinear",
                )
                gt_track_distance_to_goal = F.interpolate(
                    gt_track_distance_to_goal,
                    size=(self.track_patch_size, self.track_patch_size),
                    mode="nearest",
                )  # [1, 3, H, W]

                # To numpy
                gt_track_future = gt_track_future.cpu().numpy()
                gt_track_future_valid = gt_track_future_valid.cpu().numpy()
                gt_track_future_visib = gt_track_future_visib.cpu().numpy()
                gt_track_history = gt_track_history.cpu().numpy()
                gt_track_history_valid = gt_track_history_valid.cpu().numpy()
                gt_track_history_visib = gt_track_history_visib.cpu().numpy()
                gt_track_init = gt_track_init.cpu().numpy()
                gt_track_color = gt_track_color.cpu().numpy()
                gt_track_distance_to_goal = gt_track_distance_to_goal.cpu().numpy()

        gt_track_future = gt_track_future[self.track_horizon_of_interest]
        gt_track_future_valid = gt_track_future_valid[self.track_horizon_of_interest]
        gt_track_future_visib = gt_track_future_visib[self.track_horizon_of_interest]
        gt_track_history = gt_track_history[self.track_horizon_of_interest]
        gt_track_history_valid = gt_track_history_valid[self.track_horizon_of_interest]
        gt_track_history_visib = gt_track_history_visib[self.track_horizon_of_interest]
        gt_track_init = gt_track_init
        gt_track_color = gt_track_color

        # Do horizon truncation
        gt_track_future = gt_track_future[: self.horizon]
        gt_track_future_valid = gt_track_future_valid[: self.horizon]
        gt_track_future_visib = gt_track_future_visib[: self.horizon]
        gt_track_history = gt_track_history[-self.horizon :]
        gt_track_history_valid = gt_track_history_valid[-self.horizon :]
        gt_track_history_visib = gt_track_history_visib[-self.horizon :]

        return (
            gt_track_future,  # [T, 3, H, W]
            gt_track_future_valid,  # [T, 1, H, W]
            gt_track_future_visib,  # [T, 1, H, W]
            gt_track_history,  # [T, 3, H, W]
            gt_track_history_valid,  # [T, 3, H, W]
            gt_track_history_visib,  # [T, 1, H, W]
            gt_track_init[0],  # [3, H, W]
            gt_track_color[0],  # [3, H, W]
            gt_track_distance_to_goal,  # [1, 3, H, W]
        )

    def _array_to_tensor(self, data):
        for k, v in data.items():
            if (
                k == "color"
                or k == "history_color_frames"
                or k == "color_init"
                or k == "color_gripper"
            ):
                if v.ndim == 5:
                    data[k] = (
                        torch.from_numpy(v).permute(0, 1, 4, 2, 3) / 255.0
                    )  # [T, H, h, w, 3] => [T, H, 3, h, w]
                elif v.ndim == 4:
                    data[k] = torch.from_numpy(v).permute(0, 3, 1, 2) / 255.0
                else:
                    data[k] = torch.from_numpy(v).permute(2, 0, 1) / 255.0
            else:
                data[k] = torch.from_numpy(v)
        return data

    def __getitem__(self, idx):
        # data = self._get_frame(idx)
        # data = self._array_to_tensor(data)
        clip_data = self._get_clip(idx)
        sample = {}
        for frame_data in clip_data:
            for k, v in frame_data.items():
                if k not in sample:
                    sample[k] = []
                sample[k].append(v)
        for k, v in sample.items():
            sample[k] = np.stack(v, axis=0)
        sample = self._array_to_tensor(sample)
        return sample

    def _get_video(
        self, sample, to_tensor=False, downsample_factor=1, frame_range_ratio=(0.0, 1.0)
    ):

        sample_idx = self.sample_name_to_index[sample]
        _frame_range = self.valid_frame_ranges[sample_idx]
        n_frames = int(_frame_range[1] - _frame_range[0])
        frame_range = (
            _frame_range[0] + int(n_frames * frame_range_ratio[0]),
            _frame_range[0] + int(n_frames * frame_range_ratio[1]),
        )

        video_data = []
        for frame_idx in tqdm(
            range(frame_range[0], frame_range[1], downsample_factor),
            desc="Getting video data",
            total=(frame_range[1] - frame_range[0]) // downsample_factor,
        ):
            data = self._get_frame(sample, frame_idx)
            if to_tensor:
                data = self._array_to_tensor(data)
            video_data.append(data)
        return video_data


if __name__ == "__main__":
    patch_size = 64
    dataset = Egoasis4DDataset(
        # split_fpath="data/splits/hoi4d_release_valid_frame_ranges.csv",
        # split_fpath="data/splits/hoi4d_releaseclip_valid_frame_ranges_w_trajectory.csv",
        # The code is a Python comment that includes a file path
        # `data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv` which seems to be related to
        # splitting data or defining frame ranges for validation in a project or script.
        split_fpath="data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv",
        horizon=15,
        load_tracks=False,
        load_hands=True,
        clip_length=1,
        track_patch_size=patch_size,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    for ii, data in enumerate(dataset):
        if ii == 0:
            history_feat0 = data["history_visual_feature_patch"]
        diff = (ii, data["history_visual_feature_patch"] - history_feat0).abs().mean()
        # breakpoint()
        # print(ii % 15)
        print(diff)

    data_sample = dataset[30]
    # vis_img = data_sample["color"]
    # vis_img_init = data_sample["color_init"]
    # for k, v in data_sample.items():
    #     print(k, v.squeeze(0).shape)
    # vis_imgs = []
    # for i in tqdm(range(len(dataset))):
    #     data_sample = dataset[i]
    #     colors = data_sample["color"]
    #     print(colors.shape)
    #     for color in colors:
    #         color_np = color.permute(1, 2, 0).numpy()
    #         color_np = (color_np * 255).astype(np.uint8)[..., ::-1].copy()
    #         vis_imgs.append(color_np)

    # for i in range(len(vis_imgs)):
    #     color_np = vis_imgs[i]
    #     cv2.imshow("color", color_np)
    #     cv2.waitKey(0)
    # cv2.imwrite(f".tmp/color_{i}.png", color_np)
    # i += 1
    data_sample = {k: v.squeeze(0) for k, v in data_sample.items()}
    vis_img = data_sample["color_init"]
    vis_img_init = data_sample["color_init"]
    intr = data_sample["intrinsics"]

    start_state = data_sample["start_state"]  # [3, H, W]
    gt_state_residual = data_sample["gt_state_residual"]
    gt_state_residual = rearrange(
        gt_state_residual, "(t c) h w -> t c h w", c=3, h=patch_size, w=patch_size
    )  # [T*3, H, W] => [T, 3, H, W]
    history_state = data_sample["history_state"]
    history_state = rearrange(
        history_state, "(t c) h w -> t c h w", c=3, h=patch_size, w=patch_size
    )  # [T*3, H, W] => [T, 3, H, W]
    history_state_valid = data_sample["history_state_valid"]
    history_state_valid = rearrange(
        history_state_valid, "(t c) h w -> t c h w", c=3, h=patch_size, w=patch_size
    )  # [T*3, H, W] => [T, 3, H, W]
    start_state = start_state[None]  # [1, 3, H, W]
    gt_state = start_state + gt_state_residual  # [T, 3, H, W]
    state_valid = data_sample["state_valid"]
    state_color = data_sample["state_color"]  # [3, H, W]
    state_valid = rearrange(
        state_valid, "(t c) h w -> t c h w", c=3, h=patch_size, w=patch_size
    )  # [T*3, H, W] => [T, 3, H, W]
    vis = []
    # for i in range(gt_state.shape[0]):
    #     track_i = gt_state[i].reshape(3, -1).T
    #     color  = state_color.reshape(3, -1).T
    #     track_i_valid = state_valid[i].reshape(3, -1).T
    #     pcd = DatasetUtils.visualize_points(track_i, color, as_spheres=False)
    #     vis.append(pcd)
    # o3d.visualization.draw(vis)
    # for i in range(history_state.shape[0]):
    #     track_i = history_state[i].reshape(3, -1).T
    #     color = state_color.reshape(3, -1).T
    #     track_i_valid = history_state_valid[i].reshape(3, -1).T
    #     pcd = DatasetUtils.visualize_points(track_i, color, as_spheres=False)
    #     vis.append(pcd)
    # o3d.visualization.draw(vis)
    gt_state = rearrange(gt_state[1], "h p1 p2 -> (p1 p2) h")
    # gt_state = rearrange(history_state[-1], "h p1 p2 -> (p1 p2) h")

    start_state = rearrange(start_state[-1], "h p1 p2 -> (p1 p2) h")
    state_valid = rearrange(state_valid[-1], "h p1 p2 -> (p1 p2) h")[:, 0]
    start_state = start_state.numpy()
    gt_state = gt_state.numpy()
    state_valid = state_valid.numpy()
    intr = intr.numpy()

    track_colors = DatasetUtils.random_colors(gt_state.shape[0])
    track_colors = np.array(track_colors)

    # Draw the tracks
    vis_img_init = (vis_img_init * 255).permute(1, 2, 0).cpu().numpy()
    vis_img_init = vis_img_init[:, :, ::-1].copy().astype(np.uint8)

    vis_img = (vis_img * 255).permute(1, 2, 0).cpu().numpy()
    vis_img = vis_img[:, :, ::-1].copy().astype(np.uint8)

    # Project the tracks to the image
    start_state_proj = DatasetUtils.project_points_to_image(
        start_state, intr, np.eye(4)
    )
    start_state_proj = start_state_proj.astype(np.int32)
    for i in range(start_state_proj.shape[0]):
        uv = start_state_proj[i]
        track_color = (
            int(track_colors[i, 2] * 255),
            int(track_colors[i, 1] * 255),
            int(track_colors[i, 0] * 255),
        )
        if (
            uv[0] > 0
            and uv[0] < vis_img.shape[1]
            and uv[1] > 0
            and uv[1] < vis_img.shape[0]
            and state_valid[i] > 0
        ):
            cv2.circle(vis_img_init, (uv[0], uv[1]), 3, track_color, -1)

    gt_state_proj = DatasetUtils.project_points_to_image(gt_state, intr, np.eye(4))
    gt_state_proj = gt_state_proj.astype(np.int32)

    for i in range(gt_state_proj.shape[0]):
        uv = gt_state_proj[i]
        track_color = (
            int(track_colors[i, 2] * 255),
            int(track_colors[i, 1] * 255),
            int(track_colors[i, 0] * 255),
        )
        if (
            uv[0] > 0
            and uv[0] < vis_img.shape[1]
            and uv[1] > 0
            and uv[1] < vis_img.shape[0]
            and state_valid[i] > 1
        ):
            # filled = -1 if state_valid[i] else 0
            cv2.circle(vis_img, (uv[0], uv[1]), 3, track_color, -1)

    vis_img = np.concatenate([vis_img_init, vis_img], axis=1)
    # cv2.imshow("vis_img", vis_img)
    # cv2.waitKey(0)
    cv2.imwrite(".tmp/vis_img_5.png", vis_img)

    # for k, v in data_sample.items():
    #     print(k, v.shape)
    # color = data_sample["color"].permute(1, 2, 0).numpy()
    # depth = data_sample["depth"].numpy()
    # intr = data_sample["intrinsics"].numpy()
    # gt_action = data_sample["gt_action"].numpy()
    # history_action = data_sample["history_action"].numpy()
    # points, scene_ids = DatasetUtils.backproject(depth, intr, depth > 0)
    # point_colors = color[scene_ids[0], scene_ids[1]]
    # pcd = DatasetUtils.visualize_points(points, point_colors)
    # gt_traj_visual = DatasetUtils.visualize_3d_trajectory(
    #     gt_action, size=0.01, cmap_name="turbo", to_mesh=True
    # )
    # history_traj_visual = DatasetUtils.visualize_3d_trajectory(
    #     history_action, size=0.01, cmap_name="magma", to_mesh=True
    # )
    # vis_scenes = [pcd, gt_traj_visual, history_traj_visual]
    # o3d.visualization.draw(vis_scenes)

    # for k, v in data_sample.items():
    #     print(k, v.shape)
    # video_data = dataset._get_video("ZY20210800003/H3/C3/N22/S277/s05/T2.003")
    # vis_scenes, vis_trajs = [], []
    # slam_poses = []
    # for ii, data in enumerate(video_data):

    #     # Visualize the data
    #     color = data["color"]
    #     depth = data["depth"]
    #     # mask_hand = data["mask_hand"]
    #     # mask_object = data["mask_object"]
    #     intrinsics = data["intrinsics"]
    #     T_cam0_cam = data["T_cam0_cam"]
    #     slam_pose = data["slam_pose"]
    #     start_pos = data["start_pos"]
    #     gt_action = data["gt_action"][:, 3:]
    #     action_valid = data["action_valid"]

    #     # Visualize the point cloud
    #     points, scene_ids = DatasetUtils.backproject(depth, intrinsics, depth > 0)
    #     point_colors = color[scene_ids[0], scene_ids[1]] / 255
    #     points_world = DatasetUtils.transform_points(points, T_cam0_cam)
    #     hand_traj_world = DatasetUtils.transform_points(gt_action, T_cam0_cam)

    #     pcd = DatasetUtils.visualize_points(points_world, point_colors)
    #     traj = DatasetUtils.visualize_3d_trajectory(
    #         hand_traj_world, size=0.03, cmap_name="turbo", to_mesh=True
    #     )
    #     vis_scenes.append(pcd)
    #     vis_trajs.append(traj)
    #     slam_poses.append(T_cam0_cam @ AriaUtils.T_z_m90.T)

    # viewer = SceneFlowViewer(
    #     vis_scenes=vis_scenes,
    #     vis_trajs=vis_trajs,
    #     slam_poses=slam_poses,
    #     viewer_name="Policy Closed Loop",
    # )
    # viewer.run()

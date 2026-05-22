import os
import numpy as np
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
from data.dataset import Egoasis4DDataset
from scipy.spatial.transform import Rotation as R

# DATA_DIR = "/storage/group/srl/stretch/"
DATA_DIR = "/storage/group/srl/stretch/"
DEFAULT_FPS = 15
# DOWNSAMPLING_FACTOR_DYNAMICS = 3
DOWNSAMPLING_FACTOR_DYNAMICS = 2


class Robot4DDataset(Egoasis4DDataset):
    ACTION_DIM = 20
    WRIST_ACTION_DIM = 20
    GRIPPER_IMAGE_SIZE = (240, 320)
    HEAD_IMAGE_SIZE = (224, 224)

    def __init__(
        self,
        split_fpath,
        data_dir=DATA_DIR,
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
        load_distance_to_goal=False,
        distance_to_goal_format="progress",  # "progress" or "ditance"
        feature_extractor="dinov3",  # or "cut3r"
        build_track_online=False,
        track_by_clip=True,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        load_finger_tips=False,
        load_rgbd_frames=True,
        load_value_prediction=False,
        action_include_progress=False,
        action_in_relative=False,
        flow_in_relative=True,
        action_in_world_frame=True,
        training=False,
        rl_mode=False,
        rl_round=0,
        reward_type="dense",  # "sparse" or "dense"
        dinov3_visual_dim=768,
        **kwargs,
    ):
        super().__init__(
            split_fpath=split_fpath,
            data_dir=data_dir,
            clip_length=clip_length,
            fps=fps,
            horizon=horizon,
            flow_horizon=flow_horizon,
            action_include_progress=action_include_progress,
            action_in_relative=action_in_relative,
            flow_in_relative=flow_in_relative,
            track_by_clip=track_by_clip,
            track_horizon_of_interest=track_horizon_of_interest,
            language_max_length=language_max_length,
            track_patch_size=track_patch_size,
            transform=transform,
            distance_to_goal_format=distance_to_goal_format,
            feature_extractor=feature_extractor,
            load_distance_to_goal=load_distance_to_goal,
            training=training,
            build_track_online=build_track_online,
            dataset_sample_weights=False,
            load_finger_tips=False,
            load_tracks=False,
            load_hands=False,
            reward_type=reward_type,
            dinov3_visual_dim=dinov3_visual_dim,
        )
        assert (
            self.HEAD_IMAGE_SIZE[0] == self.HEAD_IMAGE_SIZE[1]
        ), "Target size should equal head image size"
        self.load_tracks = load_tracks
        self.load_hands = load_hands
        self.is_robocasa = "robocasa" in split_fpath
        self.action_in_world_frame = action_in_world_frame
        self.load_rgbd_frames = load_rgbd_frames
        self.action_chunk_size = action_chunk_size
        self.rl_mode = rl_mode
        self.rl_round = rl_round
        self.reward_type = reward_type
        # Reload the action statistics
        if self.load_hands:
            prefix = "rel" if self.action_in_relative else ""
            suffix = "InWorld" if self.action_in_world_frame else ""
            if self.is_robocasa:
                robocasa_task = self.split_fpath.split("/")[-1].split("_")[1]
                stat_action_fpath = os.path.join(
                    "data/splits_robocasa",
                    f"robocasa-downsampled_{robocasa_task}_{prefix}action{suffix}_FPS15_HORIZON15.json",
                )
                stat_action_for_dynamics_fpath = os.path.join(
                    "data/splits_robocasa",
                    f"robocasa-downsampled_{robocasa_task}_{prefix}action_FPS15_HORIZON15.json",
                )
                print(
                    f"Loading the action statistics from {stat_action_fpath} for robocasa {robocasa_task} dataset..."
                )

            else:
                basename = self.split_fpath.split("/")[-1]
                basename = basename.split("_")[:2]
                basename = "_".join(basename)
                stat_action_fpath = os.path.join(
                    "data/splits_robot",
                    f"{basename}_{prefix}action{suffix}_FPS15_HORIZON15.json",
                )
                stat_action_for_dynamics_fpath = os.path.join(
                    "data/splits_robot",
                    f"{basename}_relaction{suffix}_FPS15_HORIZON15.json",
                )
                print(
                    f"Loading the action statistics from {stat_action_fpath} for stretch robot dataset..."
                )

            # Load the action statistics
            stat_action = DatasetUtils.load_json(stat_action_fpath)
            self.action_mean = np.array(stat_action["action_mean"])
            self.action_std = np.array(stat_action["action_std"])
            self.action_norm_max = np.array(stat_action["action_norm_max"])
            self.action_norm_min = np.array(stat_action["action_norm_min"])

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

        # Reload the state statistics
        if self.load_tracks:
            print("Loading the state statistics for stretch robot dataset...")
            assert (
                not self.is_robocasa
            ), "State statistics are not available for robocasa dataset yet ..."
            basename = self.split_fpath.split("/")[-1]
            basename = basename.split("_")[:2]
            basename = "_".join(basename)
            prefix = "clip" if self.track_by_clip else "seq"
            suffix = "InAbs" if not self.flow_in_relative else ""
            stat_state_fpath = os.path.join(
                "data/splits_robot",
                f"{basename}-{prefix}_state{suffix}_FPS15_HORIZON15.json",
            )
            print(
                f"Loading the state statistics from {stat_state_fpath} for stretch robot dataset..."
            )
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
            self.state_mean = self.state_mean[: self.horizon]
            self.state_std = self.state_std[: self.horizon]
            self.state_norm_max = self.state_norm_max[: self.horizon]
            self.state_norm_min = self.state_norm_min[: self.horizon]

        # Adjust the dataset for different modes
        if self.rl_mode:
            print(
                "====> VLA-RL mode: Training on all samples, parsing advantage prediction ..."
            )
            if "round" in self.df.columns:
                valid_mask = self.df["round"].fillna(-1).astype(int).values <= int(
                    self.rl_round
                )
                self.df = self.df.loc[valid_mask].reset_index(drop=True)
                print(
                    f"====> VLA-RL mode: Keeping samples with rl_round <= {self.rl_round}, {len(self.df)} samples left."
                )
                self.advantage_labels = self.df["advantage_label"].values
                self.samples = self.df["sample"].tolist()
                self.valid_frame_ranges = [
                    eval(v) for v in self.df["valid_frame_range"].tolist()
                ]
                self.fine_actions = self.df["fine_action"].tolist()
                self.dataset_categories = self.df["dataset"].tolist()
                self._build_sample_clip_pairs()
            else:
                raise ValueError(
                    "====> VLA-RL mode: No 'round' column in split, skipping rl_round filtering."
                )
            self._parse_advantage_prediction()
        else:
            self.load_value_prediction = False
            if not self.load_tracks:
                print(
                    "====> VLA-BC mode: Removing non-teleop samples from the dataset ..."
                )
                valid_mask = self.df["is_teleop"].values == 1
                self.df = self.df.loc[valid_mask].reset_index(drop=True)
                self.advantage_labels = self.df["advantage_label"].values
                self.samples = self.df["sample"].tolist()
                self.valid_frame_ranges = [
                    eval(v) for v in self.df["valid_frame_range"].tolist()
                ]
                self.fine_actions = self.df["fine_action"].tolist()
                self.dataset_categories = self.df["dataset"].tolist()
                # Load class and task labels
                self._build_sample_clip_pairs()
            else:
                if "round" in self.df.columns:
                    valid_mask = self.df["round"].fillna(-1).astype(int).values <= int(
                        self.rl_round
                    )
                    self.df = self.df.loc[valid_mask].reset_index(drop=True)
                    print(
                        f"====> WM+VM mode: Keeping samples with round <= {self.rl_round}, {len(self.df)} samples left."
                    )
                    self.advantage_labels = self.df["advantage_label"].values
                    self.samples = self.df["sample"].tolist()
                    self.valid_frame_ranges = [
                        eval(v) for v in self.df["valid_frame_range"].tolist()
                    ]
                    self.fine_actions = self.df["fine_action"].tolist()
                    self.dataset_categories = self.df["dataset"].tolist()
                    self._build_sample_clip_pairs()
                else:
                    print("====> WM+VM mode: Training on all samples ...")

        self.success_labels = self.df["success"].values
        self.is_teleop_labels = self.df["is_teleop"].values
        self._parse_epsisodes()

    def _parse_advantage_prediction(self):
        self.load_value_prediction = True
        advantage_pred_list = []
        for sample_name in tqdm(
            self.samples, desc="Parsing advantage prediction", total=len(self.samples)
        ):
            sample_idx = self.sample_name_to_index[sample_name]
            dataset_name = self.dataset_categories[sample_idx]
            dataset_path = os.path.join(self.data_dir, dataset_name)

            _sample_name = sample_name.split("/")
            sample = "/".join(_sample_name[:-1])
            clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
            clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"

            value_pred_dir = os.path.join(
                dataset_path, sample, f"value_prediction"
            )
            value_pred_save_path = os.path.join(value_pred_dir, f"{clip_idx}.npz")

            if not os.path.exists(value_pred_save_path):
                continue

            value_pred_data = dict(np.load(value_pred_save_path))
            advantage_pred = value_pred_data["advantage"]

            # if "advantage_guided" in value_pred_data and self.training:
            #     advantage_pred_guided = value_pred_data["advantage_guided"]
            #     advantage_pred = np.where(
            #         advantage_pred_guided > advantage_pred,
            #         advantage_pred_guided,
            #         advantage_pred,
            #     )
            advantage_pred_list.append(advantage_pred)

        if len(advantage_pred_list) == 0:
            self.advantage_threshold = (
                -1e3
            )  # No advantage prediction found, set a very small threshold
            self.load_value_prediction = False

        else:
            # assert len(advantage_pred_list) == len(self.samples) or len(
            #     advantage_pred_list
            # ) == len(self.samples) - sum(self.is_teleop_labels)
            # Set the advantage to be top 30% of the advantage predictions
            advantage_pred_list = np.concatenate(advantage_pred_list)
            self.advantage_threshold = np.percentile(advantage_pred_list, 70)
        print(
            f"=============> RL mode: Advantage threshold: {self.advantage_threshold}"
        )
        return self.advantage_threshold

    def _parse_epsisodes(self):
        self.max_episode_len = 0
        for sample_name in tqdm(
            self.samples, desc="Parsing episodes", total=len(self.samples)
        ):
            sample_idx = self.sample_name_to_index[sample_name]
            valid_frame_range = self.valid_frame_ranges[sample_idx]
            self.max_episode_len = max(
                self.max_episode_len, valid_frame_range[1] - valid_frame_range[0] + 1
            )

        print(f"=============> Max episode length: {self.max_episode_len}")

    def __len__(self):
        return len(self.sample_clip_pairs)

    def _get_frame(self, sample_name, frame_idx):
        sample_idx = self.sample_name_to_index[sample_name]
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        fine_action = self.fine_actions[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        frame = frame_idx

        _sample_name = sample_name.split("/")
        sample = "/".join(_sample_name[:-1])
        clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
        clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"

        # Acquire the file paths
        # traj_suffix = "" if self.fps == 30 else f"_{self.fps}fps"
        dataset_path = os.path.join(self.data_dir, dataset_name)
        gripper_color_video_fpath = os.path.join(dataset_path, sample, "gripper_rgb")
        gripper_depth_video_fpath = os.path.join(dataset_path, sample, "gripper_depth")
        head_color_video_fpath = os.path.join(dataset_path, sample, "head_rgb")
        head_depth_video_fpath = os.path.join(dataset_path, sample, "head_depth")
        dex_traj_video_fpath = os.path.join(dataset_path, sample, f"dex_traj")
        dex_traj_guided_video_fpath = os.path.join(
            dataset_path, sample, f"dex_traj_guided_latest_rlround{self.rl_round}"
        )

        language_embedding_fpath = os.path.join(
            self.data_dir,
            "language_embedding",
            f"{fine_action.replace(' ', '_')}.npz",
        )
        language_embedding_raw = np.load(language_embedding_fpath)["embedding"][
            : self.language_max_length
        ]
        idx = np.arange(self.language_max_length) % language_embedding_raw.shape[0]
        language_embedding = language_embedding_raw[idx]

        # Load the intrinsics and extrinsics
        camera_intr_fpath = os.path.join(
            dataset_path, sample, "intr", f"intrinsics.npz"
        )
        camera_extr_fpath = os.path.join(
            dataset_path, sample, "extr_cam0cam", f"{clip_idx}.npz"
        )
        camera_intr = np.load(camera_intr_fpath)
        camera_extr = np.load(camera_extr_fpath)
        T_wc_list = camera_extr["extrinsics_world_headcam"]
        T_wg_list = camera_extr["extrinsics_world_eecam"]
        intr, intr_gripper = camera_intr["HEAD_CAM_K"], camera_intr["EE_CAM_K"]

        # Parse the extrinsics
        T_wc0 = T_wc_list[0]
        T_wg0 = T_wg_list[0]
        T_wc = T_wc_list[frame - valid_frame_range[0]]
        T_wg = T_wg_list[frame - valid_frame_range[0]]
        T_c0c = np.linalg.inv(T_wc0) @ T_wc
        T_g0g = np.linalg.inv(T_wg0) @ T_wg

        # Parse the file paths
        depth_frame_fpath_curr = os.path.join(
            head_depth_video_fpath, f"{frame:06d}.png"
        )
        color_frame_fpath_curr = os.path.join(
            head_color_video_fpath, f"{frame:06d}.png"
        )
        gripper_color_frame_fpath_curr = os.path.join(
            gripper_color_video_fpath, f"{frame:06d}.png"
        )
        gripper_depth_frame_fpath_curr = os.path.join(
            gripper_depth_video_fpath, f"{frame:06d}.png"
        )
        traj_npz_fpath_curr = os.path.join(dex_traj_video_fpath, f"{frame:06d}.npz")

        traj_guided_npz_fpath_curr = os.path.join(
            dex_traj_guided_video_fpath, f"{frame:06d}.npz"
        )

        if self.track_by_clip:
            color_frame_fpath_init = os.path.join(
                head_color_video_fpath, f"{frame:06d}.png"
            )
        else:
            color_frame_fpath_init = os.path.join(
                head_color_video_fpath, f"{valid_frame_range[0]:06d}.png"
            )

        # Load the initial frame, and compute the crop size
        color_frame_init = cv2.imread(color_frame_fpath_init)[..., [2, 1, 0]]
        mask_dynamic_init = np.ones((self.HEAD_IMAGE_SIZE[0], self.HEAD_IMAGE_SIZE[1]))
        height, width = color_frame_init.shape[:2]
        crop_size = min(height, width)
        crop_x = (width - crop_size) // 2
        crop_y = (height - crop_size) // 2
        color_frame_init = color_frame_init[
            crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
        ]
        color_frame_init = cv2.resize(
            color_frame_init,
            (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Get the rgbd frames
        if self.load_rgbd_frames:
            depth_frame = (
                cv2.imread(depth_frame_fpath_curr, cv2.IMREAD_UNCHANGED) / 1000.0
            )
            color_frame = cv2.imread(color_frame_fpath_curr)[..., [2, 1, 0]]
            gripper_color_frame = cv2.imread(gripper_color_frame_fpath_curr)[
                ..., [2, 1, 0]
            ]
            gripper_depth_frame = (
                cv2.imread(gripper_depth_frame_fpath_curr, cv2.IMREAD_UNCHANGED)
                / 1000.0
            )
            color_frame = color_frame[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            depth_frame = depth_frame[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]

            # Do the resize operation
            gripper_color_frame = cv2.resize(
                gripper_color_frame,
                (self.GRIPPER_IMAGE_SIZE[1], self.GRIPPER_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            gripper_depth_frame = cv2.resize(
                gripper_depth_frame,
                (self.GRIPPER_IMAGE_SIZE[1], self.GRIPPER_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_NEAREST,
            )
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
            gripper_color_frame = np.zeros(
                (self.GRIPPER_IMAGE_SIZE[0], self.GRIPPER_IMAGE_SIZE[1], 3)
            )
            color_frame = np.zeros(
                (self.HEAD_IMAGE_SIZE[0], self.HEAD_IMAGE_SIZE[1], 3)
            )
            depth_frame = np.zeros((self.HEAD_IMAGE_SIZE[0], self.HEAD_IMAGE_SIZE[1]))
            gripper_depth_frame = np.zeros(
                (self.GRIPPER_IMAGE_SIZE[0], self.GRIPPER_IMAGE_SIZE[1])
            )

        height_gripper, width_gripper = gripper_color_frame.shape[:2]

        # Intrinsics for head camera
        intr[0, 0] *= self.HEAD_IMAGE_SIZE[1] / crop_size
        intr[1, 1] *= self.HEAD_IMAGE_SIZE[0] / crop_size
        intr[0, 2] = intr[1, 2] = self.HEAD_IMAGE_SIZE[1] / 2

        # Intrinsics for gripper camera
        intr_gripper[0, 0] *= self.GRIPPER_IMAGE_SIZE[1] / width_gripper
        intr_gripper[0, 2] *= self.GRIPPER_IMAGE_SIZE[1] / width_gripper
        intr_gripper[1, 1] *= self.GRIPPER_IMAGE_SIZE[0] / height_gripper
        intr_gripper[1, 2] *= self.GRIPPER_IMAGE_SIZE[0] / height_gripper

        data = {
            "color": color_frame,
            "color_gripper": gripper_color_frame,
            "depth_gripper": gripper_depth_frame,
            "depth": depth_frame,
            "color_init": color_frame_init,
            "mask_dynamic_init": mask_dynamic_init,
            "intrinsics": intr,
            "intrinsics_gripper": intr_gripper,
            "T_cam0_cam": T_c0c,
            "T_world_cam": T_wc,
            "T_world_grippercam": T_wg,
            "T_grippercam0_grippercam": T_g0g,
            "slam_pose": T_wc,
            "frame_range": valid_frame_range,
            "frame_idx": frame,
            "language_feature": language_embedding,
            "relative_time": np.array([frame - valid_frame_range[0]]).squeeze()
            / DEFAULT_FPS,
        }

        # Load the feature data
        visual_feature_data = self._build_visual_feature_history(
            sample_idx, sample_name, frame, cam_stream="head_rgb"
        )
        visual_history_patch, visual_history_patch_next, visual_goal_patch = (
            self._process_visual_feature(visual_feature_data)
        )

        # Load the gripper visual feature data
        visual_feature_data_gripper = self._build_visual_feature_history(
            sample_idx, sample_name, frame, cam_stream="gripper_rgb"
        )
        (
            visual_history_patch_gripper,
            visual_history_patch_next_gripper,
            visual_goal_patch_gripper,
        ) = self._process_visual_feature(visual_feature_data_gripper)

        # Read the extrinsics data
        # Calculate patch_size from visual_history_patch shape: [H, 196, ...] -> 196 = 14*14
        # Use shape[1] because it's always 196 regardless of whether shape is [H, 196, 768] or [H, 196, flow_horizon, 768]
        history_T_ch, history_raymap = self._build_raymap_history(
            sample_idx,
            sample_name,
            frame,
            T_wc_list,
            intr,
            valid_frame_range,
            # int(visual_history_patch.shape[1] ** 0.5),
            input_size=self.HEAD_IMAGE_SIZE,
            patch_size=(
                self.HEAD_IMAGE_SIZE[0] // 16,
                self.HEAD_IMAGE_SIZE[1] // 16,
            ),
        )
        _, history_raymap_next = self._build_raymap_history(
            sample_idx,
            sample_name,
            frame + 1,
            T_wc_list,
            intr,
            valid_frame_range,
            input_size=self.HEAD_IMAGE_SIZE,
            patch_size=(
                self.HEAD_IMAGE_SIZE[0] // 16,
                self.HEAD_IMAGE_SIZE[1] // 16,
            ),
        )

        history_T_gh, history_raymap_gripper = self._build_raymap_history(
            sample_idx,
            sample_name,
            frame,
            T_wg_list,
            intr_gripper,
            valid_frame_range,
            input_size=self.GRIPPER_IMAGE_SIZE,
            patch_size=(
                self.GRIPPER_IMAGE_SIZE[0] // 16,
                self.GRIPPER_IMAGE_SIZE[1] // 16,
            ),
        )
        _, history_raymap_next_gripper = self._build_raymap_history(
            sample_idx,
            sample_name,
            frame + 1,
            T_wg_list,
            intr_gripper,
            valid_frame_range,
            input_size=self.GRIPPER_IMAGE_SIZE,
            patch_size=(
                self.GRIPPER_IMAGE_SIZE[0] // 16,
                self.GRIPPER_IMAGE_SIZE[1] // 16,
            ),
        )

        # Compute RISE-style labels
        gt_state_value, is_terminal, gt_td_reward = self._compute_value_info(
            sample_idx, sample_name, frame_idx
        )
        gt_state_value_future, is_terminal_future, gt_td_reward_future = (
            self._compute_value_info(
                sample_idx, sample_name, frame_idx + self.action_chunk_size - 1
            )
        )

        is_teleop = self.is_teleop_labels[sample_idx]
        advantage_label = int(self.success_labels[sample_idx])
        data.update(
            {
                "advantage_label": advantage_label,
                # From head camera
                "history_visual_feature_patch": visual_history_patch,
                "history_visual_feature_patch_next": visual_history_patch_next,
                "goal_visual_feature_patch": visual_goal_patch,
                "history_T_cam_history": history_T_ch,
                "history_raymap": history_raymap,
                "history_raymap_next": history_raymap_next,
                # From gripper camera
                "history_visual_feature_patch_gripper": visual_history_patch_gripper,
                "history_visual_feature_patch_next_gripper": visual_history_patch_next_gripper,
                "goal_visual_feature_patch_gripper": visual_goal_patch_gripper,
                "history_T_cam_history_gripper": history_T_gh,
                "history_raymap_gripper": history_raymap_gripper,
                "history_raymap_next_gripper": history_raymap_next_gripper,
                # Gt value
                "gt_state_value": gt_state_value,
                "gt_td_reward": gt_td_reward,
                "is_terminal": is_terminal,
                # GT value future
                "gt_state_value_future": gt_state_value_future,
                "gt_td_reward_future": gt_td_reward_future,
                "is_terminal_future": is_terminal_future,
            }
        )

        # Load the value prediction
        if self.load_value_prediction:
            # Acquire the advantage label
            is_teleop = self.is_teleop_labels[sample_idx]
            is_success = self.success_labels[sample_idx]
            is_improved = False
            value_pred_fpath = os.path.join(
                dataset_path,
                sample,
                f"value_prediction",
                f"{clip_idx}.npz",
            )

            if is_teleop:
                assert is_success, f"Teleop sample is not successful: {sample_name}"
                advantage_pred = self.advantage_threshold + 0.01
                value_pred = gt_state_value
                value_future_pred = gt_state_value_future
                advantage_label = int(is_success)
            else:
                value_pred_data = dict(np.load(value_pred_fpath))
                value_pred = value_pred_data["value"][frame - valid_frame_range[0]]
                advantage_pred = value_pred_data[f"advantage"][
                    frame - valid_frame_range[0]
                ]
                value_future_pred = value_pred_data[f"value_future"][
                    frame - valid_frame_range[0]
                ]
                if "advantage_guided" in value_pred_data and self.training:
                    advantage_pred_guided = value_pred_data["advantage_guided"][
                        frame - valid_frame_range[0]
                    ]
                    value_future_pred_guided = value_pred_data["value_future_guided"][
                        frame - valid_frame_range[0]
                    ]
                    is_improved = advantage_pred_guided > max(advantage_pred, 0) + 0.01
                    advantage_pred = (
                        advantage_pred_guided if is_improved else advantage_pred
                    )
                    value_future_pred = (
                        value_future_pred_guided if is_improved else value_future_pred
                    )

                advantage_label = int(advantage_pred > self.advantage_threshold)

            # Update the advantage label
            data.update(
                {
                    "advantage": advantage_pred,
                    "advantage_label": advantage_label,
                    "advantage_threshold": self.advantage_threshold,
                    "value_expected": value_pred,
                    "value_future_expected": value_future_pred,
                    "is_improved": int(is_improved),
                }
            )

        # Load the hand trajectory
        if self.load_hands:
            # Get the hand trajectory
            if os.path.exists(traj_npz_fpath_curr):
                hand_data = np.load(traj_npz_fpath_curr, allow_pickle=True)

                # See if we need to use the guided hand trajectory
                if self.rl_mode and self.training:
                    assert (
                        self.load_value_prediction
                    ), "Value prediction must be loaded for guided action"
                    is_improved = data["is_improved"]
                    if os.path.exists(traj_guided_npz_fpath_curr):
                        hand_data_guided = np.load(
                            traj_guided_npz_fpath_curr, allow_pickle=True
                        )
                        if is_improved:
                            hand_data = hand_data_guided
                hand_data = dict(hand_data)
                hand_data["T_world_cam"] = T_wc

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
                    hand_traj, start_pos, has_finger_tips=False
                )
            )
            hand_traj_history_rel = (
                DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                    hand_traj_history, start_pos, has_finger_tips=False
                )
            )

            if self.action_in_relative:
                hand_traj = (
                    DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                        hand_traj_rel, start_pos, has_finger_tips=False
                    )
                )
                hand_traj_history = (
                    DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                        hand_traj_history_rel, start_pos, has_finger_tips=False
                    )
                )

            # if self.training:
            #     hand_traj_history = self._augment_hand_trajectory(
            #         hand_traj_history, 0.3
            #     )
            #     hand_traj_history_rel = self._augment_hand_trajectory(
            #         hand_traj_history_rel, 0.3
            #     )

            # Concatenate the progress value to the action data
            if self.action_include_progress:
                progress_value = np.arange(
                    frame_idx, frame_idx + self.action_chunk_size
                )[
                    :, None
                ]  # [H, 1]
                progress_value = (progress_value - valid_frame_range[0]) / (
                    valid_frame_range[1] - valid_frame_range[0] + 1
                )  # [H, 1]
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
                    "gt_state_value": gt_state_value,
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

    def _process_visual_feature(self, visual_feature_data):
        visual_history_patch = visual_feature_data["history_patch"]
        visual_history_patch_next = visual_feature_data["history_patch_next"]
        visual_goal_patch = visual_feature_data["goal_patch"]
        return visual_history_patch, visual_history_patch_next, visual_goal_patch

    def _compute_value_info(self, sample_idx, sample_name, frame_idx):
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        is_success = self.success_labels[sample_idx]
        T = valid_frame_range[1] - valid_frame_range[0]
        T_max = self.max_episode_len
        idx = frame_idx - valid_frame_range[0]
        # progress supervision
        if self.reward_type == "sparse":
            gamma = 0.995
            gt_state_value = gamma ** (T_max - idx)
        elif self.reward_type == "dense":
            gt_state_value = idx / T_max
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        gt_state_value = np.clip(gt_state_value, 0, 1)

        # terminal flag (based on real episode length)
        is_terminal = 1 if idx == (T - 1) else 0

        terminal_reward = 1.0 if is_success else -1.0
        gt_td_reward = terminal_reward if is_terminal else 0.0

        return gt_state_value, is_terminal, gt_td_reward

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
        history_T_ch = []
        history_raymap = []
        frame_idx = min(frame_idx, valid_frame_range[1] - 1)
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
        return history_T_ch, history_raymap

    def _build_visual_feature_history(
        self, sample_idx, sample_name, frame_idx, cam_stream="head_rgb"
    ):
        frame = frame_idx
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(self.data_dir, dataset_name)

        _sample_name = sample_name.split("/")
        sample = "/".join(_sample_name[:-1])
        # clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
        clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
        clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"

        visual_feature_fpath = os.path.join(
            dataset_dir,
            sample,
            f"{self.feature_extractor}_visual_feature_by_clip_224_{cam_stream}",
            f"{clip_idx}.npy",
        )

        if not os.path.exists(visual_feature_fpath):
            print(
                f"Visual feature data not found for {sample_name} at frame {frame_idx}"
            )
            return None
        visual_observation = np.load(visual_feature_fpath, mmap_mode="r")

        assert (
            visual_observation.shape[-1] == self.dinov3_visual_dim
        ), (
            f"Visual feature last dim {visual_observation.shape[-1]} != "
            f"dinov3_visual_dim={self.dinov3_visual_dim} ({visual_feature_fpath})"
        )

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
        clipped_indices_next = np.clip(
            frame_indices + 1,
            valid_frame_range[0],
            valid_frame_range[1] - 1,
        )
        array_indices = clipped_indices - valid_frame_range[0]
        array_indices_next = clipped_indices_next - valid_frame_range[0]
        history_visual_observation = visual_observation[array_indices]  # [H, 196, 768]
        history_visual_observation_next = visual_observation[
            array_indices_next
        ]  # [H, 196, 768]
        # For history: keep the full shape [H, 196, 768]
        visual_feature_data = {
            "history_patch": history_visual_observation,
            "history_patch_next": history_visual_observation_next,
        }

        # For goal configuration
        frame_index_goal = np.clip(
            frame + self.action_chunk_size - 1,
            valid_frame_range[0],
            valid_frame_range[1] - 1,
        )
        array_index_goal = frame_index_goal - valid_frame_range[0]
        visual_observation_goal = visual_observation[array_index_goal]  # [196, 768]
        visual_feature_data["goal_patch"] = visual_observation_goal
        visual_feature_data["goal_patch"] = visual_feature_data["goal_patch"] / 5.0
        return visual_feature_data

    def _build_track_data(
        self, sample_idx, sample_name, frame_idx, T_cam0_cam, track_by_clip=False
    ):
        # Acquire the saved coords data
        frame = frame_idx
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        track_suffix = "_by_clip" if track_by_clip else ""

        if not self.build_track_online:
            # assert track_by_clip, "Track by clip is required for offline building!"
            _sample_name = sample_name.split("/")
            sample_dir = "/".join(_sample_name[:-1])
            fi = (
                frame_idx - valid_frame_range[0]
            ) // DOWNSAMPLING_FACTOR_DYNAMICS  # 0,1,2 => 0; 3,4,5 => 1; 6,7,8 => 2; ...
            frame_idx_load = (
                fi * DOWNSAMPLING_FACTOR_DYNAMICS + valid_frame_range[0]
            )  # 0,1,2 => 0; 3,4,5 => 3; 6,7,8 => 6; ...
            track_data_meta_fpath = os.path.join(
                dataset_dir,
                sample_dir,
                f"tapip3d_tracks{track_suffix}_224_{self.track_patch_size}",
                f"{frame_idx_load:06d}_meta.npz",
            )
            track_data_tracks_fpath = os.path.join(
                dataset_dir,
                sample_dir,
                f"tapip3d_tracks{track_suffix}_224_{self.track_patch_size}",
                f"{frame_idx_load:06d}_tracks.npy",
            )
            track_data_history_tracks_fpath = os.path.join(
                dataset_dir,
                sample_dir,
                f"tapip3d_tracks{track_suffix}_224_{self.track_patch_size}",
                f"{frame_idx_load:06d}_history_tracks.npy",
            )
            track_data = np.load(track_data_meta_fpath)
            track_data_tracks = np.load(track_data_tracks_fpath)
            track_data_history_tracks = np.load(track_data_history_tracks_fpath)
            track_data = dict(track_data)
            valid_init = track_data["valid"]
            valid = np.repeat(valid_init[None, :], len(track_data_tracks), axis=0)
            if (frame_idx - valid_frame_range[0]) % DOWNSAMPLING_FACTOR_DYNAMICS != 0:
                valid = np.zeros((self.horizon, self.track_patch_size**2))

            track_data.update(
                {
                    "tracks": track_data_tracks,
                    "history_tracks": track_data_history_tracks,
                    "valid": valid,
                    "visib": valid,
                    "history_valid": valid,
                    "history_visib": valid,
                }
            )
        else:
            assert (
                not track_by_clip
            ), "Track by clip is not supported for online building!"
            _sample_name = sample_name.split("/")
            sample = "/".join(_sample_name[:-1])
            # clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
            clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
            clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"
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
            query_point = full_track_data["query_point"]
            valid = full_track_data["tracks_valid"][None].repeat(
                coords.shape[0], axis=0
            )
            tracks_color = full_track_data["tracks_color"][None].repeat(
                coords.shape[0], axis=0
            )
            visibs = valid

            # Start building the window-based track data (vectorized)
            # Generate all frame indices at once
            fi = (
                frame_idx - valid_frame_range[0]
            ) // DOWNSAMPLING_FACTOR_DYNAMICS  # downsampled frame index
            array_indices = np.arange(
                fi - (self.horizon - 1) * (self.fps // DEFAULT_FPS),
                fi + (self.horizon * (self.fps // DEFAULT_FPS)),
                self.fps // DEFAULT_FPS,
            )
            # Clip indices to valid range and convert to array indices
            array_indices = np.clip(array_indices, 0, coords.shape[0] - 1)

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
            if (
                frame_idx - valid_frame_range[0]
            ) % DOWNSAMPLING_FACTOR_DYNAMICS != 0 and self.training:
                future_tracks_valid.fill(0)
                history_tracks_valid.fill(0)
            # Build the track data
            track_data = {
                "valid": future_tracks_valid,
                "tracks": future_tracks,
                "visib": future_tracks_visib,
                "history_tracks": history_tracks,
                "history_visib": history_tracks_visib,
                "history_valid": history_tracks_valid,
                "state": query_point,
                "color": future_tracks_color[0],
                "T_cam0_cam": T_cam0_cam,
            }

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
        # frame = frame_idx
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        dataset_name = self.dataset_categories[sample_idx]
        dataset_dir = os.path.join(
            self.data_dir,
            dataset_name,
        )

        fi = (
            frame_idx - valid_frame_range[0]
        ) // DOWNSAMPLING_FACTOR_DYNAMICS  # downsampled frame index
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
                # clip_idx = f"{valid_frame_range[0]:06d}_{valid_frame_range[1]:06d}"
                clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
                clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"
                track_fpath = os.path.join(
                    dataset_dir,
                    sample,
                    f"tapip3d_tracks_224_{self.track_patch_size}",
                    f"{clip_idx}.npz",
                )
            full_track_data = np.load(track_fpath)
            tracks = full_track_data["tracks"]  # [T, N, 3]
            future_horizon = self.horizon - 1

            # Current frame index and goal frame index (goal = current + future_horizon)
            curr_idx = fi + future_horizon
            goal_idx = valid_frame_range[1] - valid_frame_range[0]
            curr_idx = np.clip(curr_idx, 0, tracks.shape[0] - 1)
            goal_idx = np.clip(goal_idx, 0, tracks.shape[0] - 1)

            curr_track, goal_track = tracks[curr_idx], tracks[goal_idx]
            distance_to_goal = goal_track - curr_track  # [N, 3]

        elif self.distance_to_goal_format == "progress":
            # Progress in [0, 1]: 0 = first frame of clip, 1 = last frame of clip
            num_frames = valid_frame_range[1] - valid_frame_range[0] + 1
            full_progress = np.arange(num_frames) / max(1, num_frames)

            future_horizon = self.horizon - 1
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

    def _process_hand_data(self, hand_data):
        if hand_data is None:
            print(f"Hand data is None, setting all to -1e3")
            hand_traj = np.ones((self.action_chunk_size, self.ACTION_DIM)) * -1e3
            hand_valid = np.zeros((self.action_chunk_size, self.ACTION_DIM))
            hand_traj_timestamps = np.zeros((self.action_chunk_size,))
            hand_traj_history = (
                np.ones((self.action_chunk_size, self.ACTION_DIM)) * -1e3
            )
            hand_valid_history = np.zeros((self.action_chunk_size, self.ACTION_DIM))
            hand_traj_timestamps_history = np.zeros((self.action_chunk_size,))
            start_pos = np.ones((self.ACTION_DIM)) * -1e3
        else:
            hand_traj = hand_data["trajectory"]
            hand_traj_history = hand_data["history_trajectory"]
            T_cam_world = np.linalg.inv(hand_data["T_world_cam"])

            # To Egoasis4D format
            hand_traj = self._postprocess_hand_trajectory(hand_traj)
            hand_traj_history = self._postprocess_hand_trajectory(hand_traj_history)

            # Transform action from the base frame to the current frame
            if not self.action_in_world_frame:
                hand_traj = DatasetUtils.transform_hand_trajectory(
                    hand_traj, T_cam_world, has_finger_tips=False
                )
                hand_traj_history = DatasetUtils.transform_hand_trajectory(
                    hand_traj_history, T_cam_world, has_finger_tips=False
                )
            hand_traj = np.concatenate([hand_traj, hand_traj], axis=1)
            hand_traj_history = np.concatenate(
                [hand_traj_history, hand_traj_history], axis=1
            )
            hand_valid = np.ones_like(hand_traj)
            hand_valid_history = hand_valid
            hand_traj_timestamps = np.arange(self.action_chunk_size)[:, None]
            hand_traj_timestamps_history = np.arange(self.horizon)[:, None]
            start_pos = hand_traj_history[-1]

        # Do horizon truncation
        hand_traj = hand_traj[: self.action_chunk_size]
        hand_valid = hand_valid[: self.action_chunk_size]
        hand_traj_timestamps = hand_traj_timestamps[: self.action_chunk_size]

        hand_traj_history = hand_traj_history[-self.horizon :]
        hand_valid_history = hand_valid_history[-self.horizon :]
        hand_traj_timestamps_history = hand_traj_timestamps_history[-self.horizon :]

        return (
            hand_traj,  # [T, ACTION_DIM]
            hand_valid,  # [T, ACTION_DIM]
            hand_traj_timestamps,  # [T, 1]
            hand_traj_history,  # [T, ACTION_DIM]
            hand_valid_history,  # [T, ACTION_DIM]
            hand_traj_timestamps_history,  # [T, 1]
            start_pos,  # [ACTION_DIM]
        )

    def _augment_hand_trajectory(
        self, hand_trajectory, probability=0.5, std=0.01, max_deg=5
    ):
        """
        hand_trajectory: [T, ACTION_DIM] = [T, (left 10 + right 10)]
        layout per hand: [x, y, z, closure, r6d(6)]
        """

        T = hand_trajectory.shape[0]
        assert self.ACTION_DIM % 2 == 0
        per_hand_dim = self.ACTION_DIM // 2

        traj_left = hand_trajectory[:, :per_hand_dim]  # [T, 10]
        traj_right = hand_trajectory[:, per_hand_dim:]  # [T, 10]

        def split(traj):
            xyz = traj[:, :3]  # [T, 3]
            closure = traj[:, 3:4]  # [T, 1]
            r6d = traj[:, 4:]  # [T, 6]
            rot = AriaUtils.rotation_6d_to_matrix(
                torch.from_numpy(r6d).float()
            ).numpy()  # [T, 3, 3]
            return xyz, closure, r6d, rot

        l_xyz, l_closure, l_r6d, l_rot = split(traj_left)
        r_xyz, r_closure, r_r6d, r_rot = split(traj_right)

        if np.random.uniform(0, 1) < probability:
            global_shift = np.random.normal(scale=std, size=(1, 3))
            rpys_global = np.random.uniform(-max_deg, max_deg, size=(3,))
            R_global = R.from_euler(
                "xyz", rpys_global, degrees=True
            ).as_matrix()  # [3, 3]

            def apply_global(xyz, rot):
                # 点：x' = R_global @ x + shift
                # xyz_aug = (R_global @ xyz.T).T + global_shift  # [T, 3]
                xyz_aug = xyz + global_shift

                # 姿态：根据你的定义选择左乘还是右乘
                rot_aug = R_global[None, :, :] @ rot  # [T, 3, 3]
                return xyz_aug, rot_aug

            l_xyz, l_rot = apply_global(l_xyz, l_rot)
            r_xyz, r_rot = apply_global(r_xyz, r_rot)

        if np.random.uniform(0, 1) < probability:
            l_xyz += np.random.normal(scale=std * 0.2, size=l_xyz.shape)
            r_xyz += np.random.normal(scale=std * 0.2, size=r_xyz.shape)

        if np.random.uniform(0, 1) < probability:
            l_closure += np.random.normal(scale=std * 0.2, size=l_closure.shape)
            r_closure += np.random.normal(scale=std * 0.2, size=r_closure.shape)
            # l_closure = np.clip(l_closure, 0.0, 1.0)
            # r_closure = np.clip(r_closure, 0.0, 1.0)

        # === 4) rot -> r6d，重组左右手 ===
        l_r6d = AriaUtils.matrix_to_rotation_6d(torch.from_numpy(l_rot).float()).numpy()
        r_r6d = AriaUtils.matrix_to_rotation_6d(torch.from_numpy(r_rot).float()).numpy()

        traj_left_aug = np.concatenate([l_xyz, l_closure, l_r6d], axis=-1)  # [T, 10]
        traj_right_aug = np.concatenate([r_xyz, r_closure, r_r6d], axis=-1)  # [T, 10]

        hand_trajectory_aug = np.concatenate(
            [traj_left_aug, traj_right_aug], axis=-1
        )  # [T, ACTION_DIM]
        return hand_trajectory_aug

    def _postprocess_hand_trajectory(self, hand_traj):
        hand_traj, hand_closure = (
            hand_traj[:, :16],
            hand_traj[:, -1:],
        )  # [T, 16], [T, 1]
        hand_traj = hand_traj.reshape(-1, 4, 4)  # [T, 4, 4]
        hand_traj_translation = hand_traj[:, :3, 3]  # [T, 3]
        hand_traj_rotation = hand_traj[:, :3, :3]  # [T, 3, 3]
        hand_traj_r6d = AriaUtils.matrix_to_rotation_6d(
            torch.from_numpy(hand_traj_rotation).float()
        ).numpy()  # [T, 6]
        horizon = hand_traj.shape[0]
        # if horizon > self.horizon:
        #     assert horizon % self.horizon == 0
        #     step = horizon // self.horizon
        #     hand_traj_translation = hand_traj_translation[::step]
        #     hand_traj_rotation = hand_traj_rotation[::step]
        #     hand_traj_r6d = hand_traj_r6d[::step]
        #     hand_closure = hand_closure[::step]
        hand_traj = np.concatenate(
            [hand_traj_translation, hand_closure, hand_traj_r6d], axis=-1
        )  # [T, 10]
        return hand_traj


if __name__ == "__main__":
    patch_size = 64
    dataset = Egoasis4DDataset(
        # split_fpath="data/splits/hoi4d_release_valid_frame_ranges.csv",
        # split_fpath="data/splits/hoi4d_releaseclip_valid_frame_ranges_w_trajectory.csv",
        split_fpath="data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv",
        horizon=15,
        load_tracks=True,
        load_hands=True,
        clip_length=1,
        track_patch_size=patch_size,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    data = dataset[0]

    ## Visualize the data
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

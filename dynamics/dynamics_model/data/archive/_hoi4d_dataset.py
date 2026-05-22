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

DATA_DIR = (
    "/home/wiss/chenh/storage/group/dataset_mirrors/01_incoming/hoi4d/HOI4D_release"
)
CLASS_LABLE_PATH = (
    "./data/annotations/HOI4D-Instructions/definitions/motion_segmentation/label.csv"
)
TASK_LABEL_PATH = (
    "./data/annotations/HOI4D-Instructions/definitions/task/task_definitions.csv"
)
WRIST_POSITION = (
    np.array([90.7384, 6.0633, 5.3507]) / 1000.0
)  # Hardcoded, might be fine ...
DEFAULT_FPS = 15


class HOI4DDataset(Dataset):
    ACTION_DIM = 48
    STATE_DIM = 3

    def __init__(
        self,
        split_fpath,
        data_dir=DATA_DIR,
        class_label_path=CLASS_LABLE_PATH,
        task_label_path=TASK_LABEL_PATH,
        fps=DEFAULT_FPS,
        clip_length=1,
        horizon=15,
        flow_horizon=15,
        downsample_factor=4,
        target_size=224,
        transform=None,
        language_max_length=30,
        track_patch_size=32,
        do_post_process=True,
        load_hands=True,
        load_tracks=False,
        load_history_frames=False,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
        super().__init__()
        df = pd.read_csv(split_fpath)
        basename = split_fpath.split("/")[-1].strip(".csv")
        basename = basename.split("_")[:2]
        basename = "_".join(basename)

        stat_action_fpath = os.path.join(
            os.path.dirname(split_fpath),
            "hoi4d_releaseclip_dexaction_FPS{}_HORIZON{}.json".format(fps, horizon),
        )
        stat_state_fpath = os.path.join(
            os.path.dirname(split_fpath),
            "hoi4d_releaseclip_state_FPS30_HORIZON15.json",
        )
        assert fps in [15, 30], "FPS must be 15 or 30"
        self.data_dir = data_dir
        self.fps = fps
        self.track_patch_size = track_patch_size

        self.downsample_factor = downsample_factor
        self.transform = transform
        self.horizon = horizon
        self.flow_horizon = flow_horizon
        self.clip_length = clip_length
        self.track_horizon_of_interest = track_horizon_of_interest
        self.target_size = target_size
        self.language_max_length = language_max_length
        self.do_post_process = do_post_process
        self.load_hands = load_hands
        self.load_tracks = load_tracks
        self.load_history_frames = (
            load_history_frames  # Whether to load the history frames
        )

        # Load the norm statistics
        if load_hands:
            stat_action = DatasetUtils.load_json(stat_action_fpath)
            self.action_mean = np.array(stat_action["action_mean"])
            self.action_std = np.array(stat_action["action_std"])
            self.action_norm_max = np.array(stat_action["action_norm_max99.5"])
            self.action_norm_min = np.array(stat_action["action_norm_min99.5"])

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
        if load_tracks:
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

        self.split_fpath = split_fpath
        self.samples = df["sample"].tolist()
        self.valid_frame_ranges = df["valid_frame_range"].tolist()
        self.valid_frame_ranges = [eval(v) for v in self.valid_frame_ranges]
        self.fine_actions = df["fine_action"].tolist()
        # Load class and task labels
        self.class_labels = pd.read_csv(class_label_path)
        self.task_labels = pd.read_csv(task_label_path)
        self._build_sample_clip_pairs()

    def __len__(self):
        return len(self.sample_clip_pairs)

    def _build_sample_clip_pairs(self):
        self.sample_clip_pairs = []
        for sample, valid_frame_range in zip(self.samples, self.valid_frame_ranges):
            for frame in range(
                valid_frame_range[0], valid_frame_range[1], self.clip_length
            ):
                frame_start, frame_end = frame, frame + self.clip_length - 1
                self.sample_clip_pairs.append((sample, [frame_start, frame_end]))

    def _get_clip(self, idx):
        sample, clip_range = self.sample_clip_pairs[idx]
        valid_frame_range = self.valid_frame_ranges[self.samples.index(sample)]
        clip_data = []
        # valid_frames = []
        for frame in np.arange(clip_range[0], clip_range[0] + self.clip_length):
            frame = min(frame, valid_frame_range[1] - 1)
            clip_data.append(self._get_frame(sample, frame))
        # print(valid_frames)
        return clip_data

    def _get_frame(self, sample_name, frame_idx):
        sample, _ = sample_name.split(".")[0], sample_name.split(".")[1]
        sample_idx = self.samples.index(sample_name)
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        fine_action = self.fine_actions[sample_idx]
        frame = frame_idx
        category_id, task_id = sample.split("/")[2], sample.split("/")[-1]
        task_label = self.task_labels.loc[
            self.task_labels["Category ID"] == category_id, task_id
        ].values[0]
        # category_name = self.class_labels.loc[
        #     self.class_labels["Category ID"] == category_id, "Category"
        # ].values[0]
        class_labels_curr = self.class_labels.loc[
            self.class_labels["Category ID"] == category_id,
            self.class_labels.columns[2:],
        ].values.flatten()
        class2id_curr = {
            label: i + 1 for i, label in enumerate(class_labels_curr) if pd.notna(label)
        }
        id2class_curr = {v: k for k, v in class2id_curr.items()}

        # Acquire the file paths
        traj_suffix = "" if self.fps == 30 else f"_{self.fps}fps"
        color_video_fpath = os.path.join(self.data_dir, sample, "align_rgb")
        depth_video_fpath = os.path.join(self.data_dir, sample, "align_depth")
        traj_video_fpath = os.path.join(self.data_dir, sample, f"dex_traj{traj_suffix}")
        mask_video_fpath = os.path.join(self.data_dir, sample, "2Dseg/shift_mask")
        visual_feature_fpath = os.path.join(
            self.data_dir, sample, "cut3r_visual_feature_by_clip_224"
        )
        track_fpath = os.path.join(
            self.data_dir,
            sample,
            "tapip3d_tracks_224_64",
        )

        # action_anno_fpath = os.path.join(self.data_dir, sample, "action/color.json")
        pose_anno_fpath = os.path.join(self.data_dir, sample, "3Dseg/output.log")
        language_embedding_fpath = os.path.join(
            self.data_dir.replace("HOI4D_release", "language_embedding"),
            f"{fine_action.replace(' ', '_')}.npz",
        )
        camera_intr_fpath = os.path.join(
            self.data_dir, sample.split("/")[0], "intrin.npy"
        )

        # Repeat the language embedding (circularly)
        language_embedding_raw = np.load(language_embedding_fpath)["embedding"][
            : self.language_max_length
        ]
        idx = np.arange(self.language_max_length) % language_embedding_raw.shape[0]
        language_embedding = language_embedding_raw[idx]

        # Read the data
        # action_anno = DatasetUtils.load_json(action_anno_fpath)
        camera_intr = np.load(camera_intr_fpath)
        poses_anno = read_o3d_poses(pose_anno_fpath)
        camera_intr[:2] /= self.downsample_factor

        depth_frame_fpath_curr = os.path.join(depth_video_fpath, f"{frame:05d}.png")
        color_frame_fpath_curr = os.path.join(color_video_fpath, f"{frame:05d}.jpg")

        history_color_frame_fpaths = [
            os.path.join(
                color_video_fpath,
                f"{max(frame - i, valid_frame_range[0]):05d}.jpg",
            )
            for i in range(self.horizon)
        ]
        history_depth_frame_fpaths = [
            os.path.join(
                depth_video_fpath,
                f"{max(frame - i, valid_frame_range[0]):05d}.png",
            )
            for i in range(self.horizon)
        ]
        traj_npz_fpath_curr = os.path.join(traj_video_fpath, f"{frame:05d}.npz")

        # track_fpath_curr = os.path.join(track_fpath, f"{frame:05d}.npz")
        color_frame_fpath_init = os.path.join(
            color_video_fpath, f"{valid_frame_range[0]:05d}.jpg"
        )
        mask_frame_fpath_init = os.path.join(
            mask_video_fpath, f"{valid_frame_range[0]:05d}.png"
        )
        if not os.path.exists(mask_frame_fpath_init):
            mask_frame_fpath_init = mask_frame_fpath_init.replace("shift_mask", "mask")
        T_wc0 = poses_anno[valid_frame_range[0]]
        T_wc = poses_anno[frame]
        T_c0c = np.linalg.inv(T_wc0) @ T_wc

        # Get the rgbd frames
        depth_frame = cv2.imread(depth_frame_fpath_curr, cv2.IMREAD_UNCHANGED) / 1000.0
        color_frame = cv2.imread(color_frame_fpath_curr)[..., [2, 1, 0]]
        color_frame_init = cv2.imread(color_frame_fpath_init)[..., [2, 1, 0]]

        if os.path.exists(mask_frame_fpath_init):
            mask_frame_init = cv2.imread(mask_frame_fpath_init)[..., [2, 1, 0]]
            mask_frame_init = cv2.resize(
                mask_frame_init,
                dsize=None,
                fx=1 / self.downsample_factor,
                fy=1 / self.downsample_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            masks, instances, classes = parse_hoi4d_mask(mask_frame_init, category_id)
            if len(masks) == 0:
                mask_dynamic_init = np.ones(
                    (color_frame.shape[0], color_frame.shape[1])
                )
            else:
                mask_dynamic_init = (
                    np.sum(np.stack(masks, axis=0), axis=0) > 0
                ).astype(np.float32)
        else:
            mask_dynamic_init = np.ones((color_frame.shape[0], color_frame.shape[1]))

        if self.load_history_frames:
            history_color_frames = [
                cv2.imread(fpath)[..., [2, 1, 0]]
                for fpath in history_color_frame_fpaths
            ]
            history_depth_frames = [
                cv2.imread(fpath, cv2.IMREAD_UNCHANGED) / 1000.0
                for fpath in history_depth_frame_fpaths
            ]
        else:
            history_color_frames = None
            history_depth_frames = None

        # Crop the image and resize to the target size
        if self.do_post_process:
            height, width = color_frame.shape[:2]
            assert height < width, "Height should be less than width"
            crop_size = height
            crop_x = (width - crop_size) // 2
            crop_y = (height - crop_size) // 2
            color_frame = color_frame[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            depth_frame = depth_frame[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            color_frame_init = color_frame_init[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            mask_dynamic_init = mask_dynamic_init[
                crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            ]
            # mask_hand = mask_hand[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
            # mask_object = mask_object[
            #     crop_y : crop_y + crop_size, crop_x : crop_x + crop_size
            # ]
            camera_intr[0, 2] = (height / width) * camera_intr[0, 2]

            # Resize the image to the target size
            color_frame = cv2.resize(
                color_frame,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            depth_frame = cv2.resize(
                depth_frame,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST,
            )
            color_frame_init = cv2.resize(
                color_frame_init,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_dynamic_init = cv2.resize(
                mask_dynamic_init,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST,
            )
            camera_intr[:2] = (self.target_size / crop_size) * camera_intr[:2]

            if self.load_history_frames:
                history_color_frames = [
                    frame[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
                    for frame in history_color_frames
                ]

                history_depth_frames = [
                    frame[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
                    for frame in history_depth_frames
                ]
                history_color_frames = [
                    cv2.resize(
                        frame,
                        (self.target_size, self.target_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in history_color_frames
                ]
                history_depth_frames = [
                    cv2.resize(
                        frame,
                        (self.target_size, self.target_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for frame in history_depth_frames
                ]

        data = {
            "color": color_frame,
            "depth": depth_frame,
            "color_init": color_frame_init,
            "mask_dynamic_init": mask_dynamic_init,
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

        # print("relative_time", data["relative_time"] * DEFAULT_FPS)
        if self.load_history_frames:
            history_color_frames = np.stack(
                history_color_frames, axis=0
            )  # [H, 224, 224, 3]
            history_depth_frames = np.stack(
                history_depth_frames, axis=0
            )  # [H, 224, 224,]
            data.update(
                {
                    "history_color_frames": history_color_frames,  # [H, 224, 224, 3]
                    "history_depth_frames": history_depth_frames,  # [H, 224, 224,]
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

            # if os.path.exists(visual_feature_fpath_curr):
            #     visual_feature_data = np.load(visual_feature_fpath_curr)
            # else:
            #     visual_feature_data = None
            visual_feature_data = self._build_visual_feature_history(sample_name, frame)
            visual_history, visual_history_patch, visual_goal, visual_goal_patch = self._process_visual_feature(
                visual_feature_data
            )
            # Read the extrinsics data
            history_T_c0h, history_raymap = [], []
            for i in range(
                frame - (self.horizon - 1) * (self.fps // DEFAULT_FPS),
                frame + (self.fps // DEFAULT_FPS),
                self.fps // DEFAULT_FPS,
            ):
                _i = max(i, valid_frame_range[0])
                T_wh = poses_anno[_i]
                T_c0h = np.linalg.inv(T_wc0) @ T_wh
                visual_patch_size = int(visual_history_patch.shape[-2] ** 0.5)
                camera_intr_patch = camera_intr.copy()
                camera_intr_patch[:2] = (
                    visual_patch_size / self.target_size
                ) * camera_intr_patch[:2]
                raymap_h = AriaUtils.get_ray_map(
                    T_c0h,
                    camera_intr_patch,
                    visual_patch_size,
                    visual_patch_size,
                    in_pluecker=True,
                )
                raymap_h = np.transpose(raymap_h, (2, 0, 1))  # (6, H, W)
                history_T_c0h.append(T_c0h)
                history_raymap.append(raymap_h)
            history_T_c0h = np.stack(history_T_c0h)  # (T, 4, 4)
            history_raymap = np.stack(history_raymap)  # (T, 6, H, W)

            # Load the state value
            values_fpath = os.path.join(
                self.data_dir.replace("HOI4D_release", "HOI4D_release_state_value"),
                sample,
                f"state_value_64",
                f"{sample.replace('/', '_')}-{sample_name.split('.')[1]}.npz",
            )

            # Load the state value
            # gamma = 0.9
            # state_deltas = np.load(values_fpath)["deltas"]  # [T,]
            # gt_state_delta = state_deltas[
            #     min(frame - valid_frame_range[0], len(state_deltas) - 1)
            # ]
            # gt_state_value = 1.0 * gamma**gt_state_delta
            # value_valid = hand_valid.max(axis=-1) > 0  # [H, 48] => [H, ]
            # value_valid = value_valid.sum() > 1

            # Compute the state value
            rewards = np.ones(valid_frame_range[1] - valid_frame_range[0]) * -1.0
            rewards[-1] = 0.0
            gt_state_value_all = rewards[::-1].cumsum()[::-1]
            gt_state_value = gt_state_value_all[frame - valid_frame_range[0]]

            # Normalize the state value to [0, 1]
            num_bins = 200
            gt_state_value = (gt_state_value - gt_state_value_all.min()) / (
                gt_state_value_all.max() - gt_state_value_all.min()
            )   # [0, 1]
            gt_state_value = int(np.clip(gt_state_value * num_bins, 0, num_bins - 1))

            data.update(
                {
                    # Add the action data
                    "start_pos": start_pos,
                    "gt_action": hand_traj,
                    "action_valid": hand_valid,
                    "action_timestamps": hand_traj_timestamps,
                    "action_mean": self.action_mean,
                    "action_std": self.action_std,
                    "action_norm_max_bound": self.action_norm_max,
                    "action_norm_min_bound": self.action_norm_min,
                    # Add the history data
                    "history_visual_feature": visual_history,
                    "history_visual_feature_patch": visual_history_patch,
                    "goal_visual_feature": visual_goal,
                    "goal_visual_feature_patch": visual_goal_patch,
                    "history_action": hand_traj_history,
                    "history_action_valid": hand_valid_history,
                    "history_action_timestamps": hand_traj_timestamps_history,
                    "history_T_cam0_cam": history_T_c0h,
                    "history_raymap": history_raymap,
                    "gt_state_value": gt_state_value,
                }
            )

        if self.load_tracks:
            # if os.path.exists(track_fpath_curr):
            #     track_data = np.load(track_fpath_curr)
            # else:
            #     track_data = None
            track_data = self._build_track_data(sample_name, frame, T_c0c)
            (
                gt_track_future,  # [T, 3, H, W]
                gt_track_future_valid,  # [T, 1, H, W]
                gt_track_future_visib,  # [T, 1, H, W]
                gt_track_history,  # [T, 3, H, W]
                gt_track_history_valid,  # [T, 3, H, W]
                gt_track_history_visib,  # [T, 1, H, W]
                gt_track_init,  # [3, H, W]
                gt_track_color,  # [3, H, W]
            ) = self._process_track_data(track_data)

            gt_track_residual = gt_track_future - gt_track_init[None]  # [T, 3, H, W]

            ## DEBUG: mask the track data
            gt_track_mask_valid = cv2.resize(
                mask_dynamic_init,
                (self.track_patch_size, self.track_patch_size),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_track_mask_valid = cv2.erode(
                gt_track_mask_valid,
                kernel=np.ones((3, 3), np.uint8),
                iterations=1,
            )[None, None]

            gt_track_mask_valid = np.where(
                gt_track_mask_valid > 0, 2.0, 1.0
            )  # Emphasize the dynamic regions!
            gt_track_future_valid *= gt_track_mask_valid
            gt_track_history_valid *= gt_track_mask_valid
            ## DEBUG: mask the track data

            # Acquire the state statistics
            state_mean = self.state_mean[self.track_horizon_of_interest]
            state_std = self.state_std[self.track_horizon_of_interest]
            state_norm_max = self.state_norm_max[self.track_horizon_of_interest]
            state_norm_min = self.state_norm_min[self.track_horizon_of_interest]

            # Squeeze the data from size [T, 3, H, W] to [T*3, H, W]
            T, C, H, W = gt_track_residual.shape
            gt_track_residual = gt_track_residual.reshape(T * C, H, W)
            gt_track_future = gt_track_future.reshape(T * C, H, W)
            gt_track_future_valid = gt_track_future_valid.reshape(T * C, H, W)
            gt_track_future_visib = gt_track_future_visib.reshape(T * 1, H, W)
            gt_track_history = gt_track_history.reshape(T * C, H, W)
            gt_track_history_visib = gt_track_history_visib.reshape(T * 1, H, W)
            gt_track_history_valid = gt_track_history_valid.reshape(T * C, H, W)
            state_mean = state_mean.reshape(T * C)
            state_std = state_std.reshape(T * C)
            state_norm_max = state_norm_max.reshape(T * C)
            state_norm_min = state_norm_min.reshape(T * C)

            data.update(
                {
                    # Add the state data
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "state_norm_max_bound": state_norm_max,
                    "state_norm_min_bound": state_norm_min,
                    # Add the track data
                    "start_state": gt_track_init,
                    "state_color": gt_track_color,
                    # Add the history data; all state are meassured in the first frame of the clip!
                    "history_state": gt_track_history,  # [T, 3, H, W]
                    "history_state_valid": gt_track_history_valid,
                    "history_state_visib": gt_track_history_visib,
                    # Add the residual data; all state are meassured in the first frame of the clip!
                    "gt_state": gt_track_future,
                    "gt_state_residual": gt_track_residual,
                    "state_visib": gt_track_future_visib,
                    "state_valid": gt_track_future_valid,
                    "state_timestamp": np.array([1.0]),
                }
            )

        for k, v in data.items():
            data[k] = np.array(v).astype(np.float32)
        return data

    def _build_visual_feature_history(self, sample_name, frame_idx):
        sample_dir, clip_idx = sample_name.split(".")[0], sample_name.split(".")[1]
        sample_idx = self.samples.index(sample_name)
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        frame = frame_idx
        visual_feature_save_dir = self.data_dir.replace(
            "HOI4D_release", "HOI4D_release_cut3r_visual_feature"
        )
        visual_feature_fpath = os.path.join(
            visual_feature_save_dir,
            sample_dir,
            f"cut3r_visual_feature_by_clip_224",
            f"{sample_dir.replace('/', '_')}-{clip_idx}.npz",
        )
        if not os.path.exists(visual_feature_fpath):
            print(
                f"Visual feature data not found for {sample_name} at frame {frame_idx}"
            )
            return None
        visual_feature_data = np.load(visual_feature_fpath)
        visual_observation = ((visual_feature_data["visual_observation"] + 1) / 5.0)  # [T, 196, 768]

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
            "history": history_visual_observation.max(axis=1),
            "history_patch": history_visual_observation,
        }

        # For goal configuration
        frame_index_goal = np.clip(
            frame + (self.horizon * (self.fps // DEFAULT_FPS)) - 1,
            valid_frame_range[0],
            valid_frame_range[1] - 1,
        )         
        array_index_goal = frame_index_goal - valid_frame_range[0]
        visual_observation_goal = visual_observation[array_index_goal]  # [196, 768]
        visual_feature_data["goal"] = visual_observation_goal.max(axis=1)
        visual_feature_data["goal_patch"] = visual_observation_goal
        return visual_feature_data

    def _build_track_data(self, sample_name, frame_idx, T_cam0_cam):
        # Acquire the saved coords data
        sample_dir, clip_idx = sample_name.split(".")[0], sample_name.split(".")[1]
        sample_idx = self.samples.index(sample_name)
        valid_frame_range = self.valid_frame_ranges[sample_idx]
        frame = frame_idx
        T_cam_cam0 = np.linalg.inv(T_cam0_cam)

        # Load the full track data
        track_save_dir = self.data_dir.replace(
            "HOI4D_release", "HOI4D_release_tapip3d_tracks"
        )
        track_fpath = os.path.join(
            track_save_dir,
            sample_dir,
            f"tapip3d_tracks_224_{self.track_patch_size}",
            f"{sample_dir.replace('/', '_')}-{clip_idx}.npz",
        )
        if not os.path.exists(track_fpath):
            return None
        full_track_data = np.load(track_fpath)
        coords = full_track_data["tracks"]
        visibs = full_track_data["tracks_visib"]
        query_point = full_track_data["query_point"]
        valid = full_track_data["tracks_valid"][None].repeat(coords.shape[0], axis=0)
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
        all_tracks_color = tracks_color[array_indices]  # [num_frames, num_points, 3]

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
        return track_data

    def _process_visual_feature(self, visual_feature_data):
        if visual_feature_data is None:
            visual_history = np.zeros((self.horizon, 196, 768))
            visual_history_patch = np.zeros((self.horizon, 196, self.flow_horizon, 768))
            visual_goal = np.zeros((196, 768))
            visual_goal_patch = np.zeros((196, 768))
        else:
            visual_history = visual_feature_data["history"]
            visual_history_patch = visual_feature_data["history_patch"]
            visual_goal = visual_feature_data["goal"]
            visual_goal_patch = visual_feature_data["goal_patch"]
        return visual_history, visual_history_patch, visual_goal, visual_goal_patch

    def _process_hand_data(self, hand_data):
        if hand_data is None:
            print(f"Hand data is None, setting all to -1e3")
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
            start_pos = hand_data["state"]
            hand_traj_history = np.nan_to_num(hand_traj_history, nan=0.0)
            hand_traj = np.nan_to_num(hand_traj, nan=0.0)

            # Futher process the hand trajectory
            left_hand_traj, right_hand_traj = (
                hand_traj[:, : self.ACTION_DIM // 2],
                hand_traj[:, self.ACTION_DIM // 2 :],
            )
            left_hand_traj_valid, right_hand_traj_valid = (
                hand_valid[:, : self.ACTION_DIM // 2][:, 0],
                hand_valid[:, self.ACTION_DIM // 2 :][:, 0],
            )
            if right_hand_traj_valid.sum() < 2:
                right_hand_traj = left_hand_traj[0][None, :].repeat(
                    right_hand_traj.shape[0], axis=0
                )

            if left_hand_traj_valid.sum() < 2:
                left_hand_traj = right_hand_traj[0][None, :].repeat(
                    left_hand_traj.shape[0], axis=0
                )
            hand_traj = np.concatenate([left_hand_traj, right_hand_traj], axis=1)
        return (
            hand_traj,  # [T, ACTION_DIM]
            hand_valid,  # [T, ACTION_DIM]
            hand_traj_timestamps,  # [T, 1]
            hand_traj_history,  # [T, ACTION_DIM]
            hand_valid_history,  # [T, ACTION_DIM]
            hand_traj_timestamps_history,  # [T, 1]
            start_pos,  # [ACTION_DIM]
        )

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

            # Transform the track data to torch
            gt_track_future = torch.from_numpy(gt_track_future).permute(0, 3, 1, 2)
            gt_track_future_valid = torch.from_numpy(gt_track_future_valid).permute(
                0, 3, 1, 2
            )
            gt_track_future_visib = torch.from_numpy(gt_track_future_visib).permute(
                0, 3, 1, 2
            )

            gt_track_history = torch.from_numpy(gt_track_history).permute(0, 3, 1, 2)
            gt_track_history_visib = torch.from_numpy(gt_track_history_visib).permute(
                0, 3, 1, 2
            )
            gt_track_history_valid = torch.from_numpy(gt_track_history_valid).permute(
                0, 3, 1, 2
            )

            gt_track_init = torch.from_numpy(gt_track_3d_init).permute(0, 3, 1, 2)
            gt_track_color = torch.from_numpy(gt_track_color).permute(0, 3, 1, 2)

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
            )
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
            )[
                0
            ]  # [N, 3]
            gt_track_color = F.interpolate(
                gt_track_color,
                size=(self.track_patch_size, self.track_patch_size),
                mode="bilinear",
            )[
                0
            ]  # [N, 3]
            gt_track_future_valid = gt_track_future_valid.repeat(1, 3, 1, 1)
            gt_track_history_valid = gt_track_history_valid.repeat(1, 3, 1, 1)

            # To numpy
            gt_track_future = gt_track_future.cpu().numpy()
            gt_track_future_valid = gt_track_future_valid.cpu().numpy()
            gt_track_future_visib = gt_track_future_visib.cpu().numpy()
            gt_track_history = gt_track_history.cpu().numpy()
            gt_track_history_valid = gt_track_history_valid.cpu().numpy()
            gt_track_history_visib = gt_track_history_visib.cpu().numpy()
            gt_track_init = gt_track_init.cpu().numpy()
            gt_track_color = gt_track_color.cpu().numpy()
            # gt_track_future[0] += (
            #     np.random.randn(*gt_track_future[0].shape) * 1e-3
            # )  # breakpoint
        return (
            gt_track_future[self.track_horizon_of_interest],  # [T, 3, H, W]
            gt_track_future_valid[self.track_horizon_of_interest],  # [T, 3, H, W]
            gt_track_future_visib[self.track_horizon_of_interest],  # [T, 1, H, W]
            gt_track_history[self.track_horizon_of_interest],  # [T, 3, H, W]
            gt_track_history_valid[self.track_horizon_of_interest],  # [T, 3, H, W]
            gt_track_history_visib[self.track_horizon_of_interest],  # [T, 1, H, W]
            gt_track_init,  # [3, H, W]
            gt_track_color,  # [3, H, W]
        )

    def _array_to_tensor(self, data):
        for k, v in data.items():
            if k == "color" or k == "history_color_frames" or k == "color_init":
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

    def _get_video(self, sample, to_tensor=False):
        sample_idx = self.samples.index(sample)
        frame_range = self.valid_frame_ranges[sample_idx]
        video_data = []
        # for frame_idx in tqdm(
        #     range(frame_range[0], frame_range[1]), desc="Getting video data"
        # ):
        for frame_idx in range(frame_range[0], frame_range[1]):
            data = self._get_frame(sample, frame_idx)
            if to_tensor:
                data = self._array_to_tensor(data)
            video_data.append(data)
        return video_data


if __name__ == "__main__":
    patch_size = 64
    dataset = HOI4DDataset(
        # split_fpath="data/splits/hoi4d_release_valid_frame_ranges.csv",
        split_fpath="data/splits/hoi4d_releaseclip_valid_frame_ranges_w_trajectory.csv",
        horizon=15,
        fps=15,
        load_tracks=True,
        load_hands=True,
        clip_length=1,
        track_patch_size=patch_size,
        do_post_process=True,
        track_horizon_of_interest=[2, 5, 8, 11, 14],
    )
    video_data = dataset._get_video("ZY20210800004/H4/C3/N67/S380/s05/T2.000")
    for data in video_data:
        # print(data["gt_state_value"])
        # breakpoint()
        vis_img = data["color"]
        # vis_img = vis_img.transpose(1, 2, 0)
        vis_img = vis_img[:, :, ::-1].copy().astype(np.uint8)
        gt_state_value = float(data["gt_state_value"])
        gt_state_value = gt_state_value
        gt_state_value -= 1
        # gt_state_value = gt_state_value.item()
        cv2.putText(vis_img, f"v={gt_state_value:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("vis_img", vis_img)
        cv2.waitKey(0)
    breakpoint()
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
    cv2.imwrite(".tmp/vis_img2.png", vis_img)

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

from data.robot_dataset import Robot4DDataset
from data.dataset import Egoasis4DDataset
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

load_finger_tips = True
VIEWER = "viser"
ACTION_DIM = 20 if not load_finger_tips else 48


if __name__ == "__main__":
    # load_finger_tips = True
    dataset = Egoasis4DDataset(
        # split_fpath="data/splits_robot/stretchrobot_valid_frame_ranges.csv",
        split_fpath="data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv",
        horizon=15,
        load_hands=True,
        load_tracks=False,
        build_track_online=False,
        track_by_clip=True,
        track_patch_size=64,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        load_finger_tips=load_finger_tips,
        action_in_relative=True,
    )
    # dataset = Robot4DDataset(
    #     split_fpath="data/splits_robot/stretchrobot_valid_frame_ranges.csv",
    #     horizon=15,
    #     load_hands=True,
    #     load_tracks=False,
    #     build_track_online=False,
    #     track_by_clip=True,
    #     track_patch_size=64,
    #     track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    #     load_finger_tips=False,
    # )
    for i in range(len(dataset)):
        sample = dataset[i]
        gt_action = sample["gt_action"][None, None].numpy()
        # start_pos = sample["start_pos"][None, None].numpy()
        start_pos = gt_action[..., 0, :]
        action_valid = sample["action_valid"][None, None].numpy() > 0
        print(f"GT action: {gt_action.shape}, Start pos: {start_pos.shape}")
        gt_action_relative = (
            DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                gt_action, start_pos, has_finger_tips=load_finger_tips
            )
        )
        gt_action_absolute = (
            DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                gt_action_relative, start_pos, has_finger_tips=load_finger_tips
            )
        )
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
            gt_action_absolute = gt_action_absolute.cpu().numpy()

        gt_action_left_traj, gt_action_right_traj = (
            gt_action[..., 0:3],
            gt_action[..., ACTION_DIM // 2 : ACTION_DIM // 2 + 3],
        )
        gt_action_absolute_left_traj, gt_action_absolute_right_traj = (
            gt_action_absolute[..., 0:3],
            gt_action_absolute[..., ACTION_DIM // 2 : ACTION_DIM // 2 + 3],
        )
        gt_action = np.concatenate([gt_action_left_traj, gt_action_right_traj], axis=-1)
        gt_action_absolute = np.concatenate(
            [gt_action_absolute_left_traj, gt_action_absolute_right_traj], axis=-1
        )
        diff = (gt_action - gt_action_absolute).mean()
        close = np.allclose(gt_action, gt_action_absolute, atol=1e-3)
        # print(f"Test passed for sample {i}")
        print(f"Diff: {diff:.10f}, Close: {close}")

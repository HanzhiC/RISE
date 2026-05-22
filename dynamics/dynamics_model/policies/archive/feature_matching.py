from data.robot_dataset import Robot4DDataset
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

load_finger_tips = False
concatenate_distance_to_goal = False
ACTION_DIM = 20

dataset = Robot4DDataset(
    split_fpath="data/splits_robot/stretchrobot_pnp-sponge_all_valid_frame_ranges.csv",
    horizon=15,
    action_chunk_size=30,
    load_hands=True,
    load_tracks=False,
    build_track_online=False,
    track_by_clip=False,
    track_patch_size=64,
    load_finger_tips=False,
    action_in_world_frame=True,  # Action in world frame
    action_in_relative=False,
    flow_in_relative=True,
    action_include_progress=False,
    training=False,
    load_rgbd_frames=True,
    load_distance_to_goal=False,
    track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
)


video_data_reference = dataset._get_video(
    "2026-03-10--11-15-23/0-171",
    downsample_factor=1,
    to_tensor=True,
    frame_range_ratio=(0.0, 0.5),
)

video_data_to_be_optimized = dataset._get_video(
    "2026-03-10--11-09-28/0-123",
    downsample_factor=1,
    to_tensor=True,
    frame_range_ratio=(0.0, 0.5),
)

for idx, frame_data in enumerate(video_data_to_be_optimized):
    visual_feature_curr = frame_data["history_visual_feature_patch"][-1]

    idx_in_reference = np.arange(idx - 10, idx + 10).clip(
        0, len(video_data_reference) - 1
    )
    frame_corr_data_reference = [video_data_reference[i] for i in idx_in_reference]

    visual_feature_reference_bank = [
        frame_data["history_visual_feature_patch"][-1]
        for frame_data in frame_corr_data_reference
    ]
    visual_feature_reference_bank = torch.stack(visual_feature_reference_bank, axis=0)

    # Measure the similarity between the current visual feature and the reference bank
    # L is the number of visual feature tokens, and B is the number of reference frames
    # visual_feature_curr: [L, 768] - current visual feature
    # visual_feature_reference_bank: [B, L, 768] - reference visual feature bank
    similarity = F.cosine_similarity(
        visual_feature_curr.unsqueeze(0), visual_feature_reference_bank, dim=2
    )  # [B, L] - similarity between the current visual feature and the reference bank
    similarity = similarity.mean(
        dim=1
    )  # [B] - average similarity over the reference frames
    most_similar_idx = similarity.argmax().item()
    most_similar_frame_data = frame_corr_data_reference[most_similar_idx]
    visual_feature_most_similar = most_similar_frame_data[
        "history_visual_feature_patch"
    ][-1]

    # Acquire the goal feature of the most similar frame
    goal_feature_most_similar = most_similar_frame_data["goal_visual_feature_patch"]

    # TODO: use this visual feature to optimize the action of the current frame with the dynamics model

    # Visualization
    color = frame_data["color"].cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
    depth = frame_data["depth"].cpu().numpy()
    intr = frame_data["intrinsics"].cpu().numpy()
    T_world_cam = frame_data["T_world_cam"].cpu().numpy()
    points, scene_ids = DatasetUtils.backproject(
        depth, intr, depth < 1.5, False
    )
    points_rgb = color[scene_ids[0], scene_ids[1]]
    points_world = DatasetUtils.transform_points(points, T_world_cam)
    pcd_scene = DatasetUtils.visualize_points(points_world, colors=points_rgb)

    # Visualize the action
    gt_action = frame_data["gt_action"].cpu().numpy()
    gt_right_action = gt_action[:, ACTION_DIM // 2 :]
    gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)
    vis_action = DatasetUtils.visualize_6d_trajectory(
        gt_right_root_action, size=0.01, cmap_name="plasma", to_mesh=True
    )

    # Visualize the most similar frame's gt action
    color_most_similar = most_similar_frame_data["color"].cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
    gt_action_most_similar = most_similar_frame_data["gt_action"].cpu().numpy()
    gt_right_action_most_similar = gt_action_most_similar[:, ACTION_DIM // 2 :]
    gt_right_root_action_most_similar = DatasetUtils.get_root_transformation(gt_right_action_most_similar)
    vis_action_most_similar = DatasetUtils.visualize_6d_trajectory(
        gt_right_root_action_most_similar, size=0.01, cmap_name="turbo", to_mesh=True
    )
    o3d.visualization.draw([pcd_scene, vis_action, vis_action_most_similar])

    color_vis = np.concatenate([color, color_most_similar], axis=1)
    color_vis = (color_vis * 255).astype(np.uint8)[..., ::-1].copy()
    cv2.imshow("color_vis", color_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

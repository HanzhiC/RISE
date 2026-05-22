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
concatenate_distance_to_goal = True

horizon = 15

VIEWER = "viser"
ACTION_DIM = 20
if __name__ == "__main__":
    dataset = Robot4DDataset(
        split_fpath="data/splits_robot/stretchrobot_wipe-table_all_valid_frame_ranges.csv",
        horizon=15,
        action_chunk_size=30,
        load_hands=True,
        load_tracks=True,
        build_track_online=True,
        track_by_clip=False,
        track_patch_size=64,
        load_finger_tips=False,
        action_in_world_frame=True,  # Action in world frame
        action_in_relative=True,
        flow_in_relative=True,
        action_include_progress=True,
        training=True,
        rl_mode=True,
        # load_rgbd_frames=True,
        load_rgbd_frames=True,
        load_distance_to_goal=True,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    print("========> Size of the dataset: ", len(dataset))

    video_list = dataset.samples
    video_list = [
        # "2026-03-26--19-56-55/0-364",
        # "2026-03-29--16-14-13/0-144",
        # "2026-03-29--14-50-07/0-168",
        "2026-03-17--13-00-49/0-30",
        "2026-03-17--11-12-53/0-341",
        "2026-03-15--14-14-31/0-277",
    ]
    # success_labels = dataset.success_labels
    # video_list = [
    #     video_name
    #     for video_name in video_list
    #     if not success_labels[dataset.sample_name_to_index[video_name]]
    # ]

    for video_name in video_list:
        video_data = dataset._get_video(video_name, downsample_factor=1)
        frame_range = dataset.valid_frame_ranges[
            dataset.sample_name_to_index[video_name]
        ]
        vis_scenes, vis_trajs, slam_poses = [], [], []
        track_color = DatasetUtils.random_colors(64 * 64)
        track_color = np.array(track_color)
        for ii, data in enumerate(video_data):
            # Visualize the data
            color = data["color"]
            depth = data["depth"]
            intrinsics = data["intrinsics"]
            T_cam0_cam = data["T_cam0_cam"]
            T_world_cam = data["T_world_cam"]
            T_world_cam0 = T_world_cam @ np.linalg.inv(T_cam0_cam)
            action_valid = data["action_valid"]
            gt_action = data["gt_action"][:, :ACTION_DIM]
            gt_progress = data["gt_action"][:, -1]
            gt_state_value = data["gt_state_value"]
            advantage_label = data["advantage_label"]
            advantage = data["advantage"]
            advantage_threshold = data["advantage_threshold"]
            is_improved = data["is_improved"]
            print(
                "advantage_label: ",
                advantage_label,
                "advantage: ",
                advantage,
                "advantage_threshold: ",
                advantage_threshold,
                "is_improved: ",
                is_improved,
            )

            # Visualize the point cloud
            depth[np.logical_or(depth > 1.7, depth < 0.0)] = 0
            points, scene_ids = DatasetUtils.backproject(depth, intrinsics, depth > 0)
            point_colors = color[scene_ids[0], scene_ids[1]] / 255
            points_world = DatasetUtils.transform_points(points, T_world_cam)
            pcd = DatasetUtils.visualize_points(points_world, point_colors)

            vis_scenes.append(pcd)
            slam_poses.append(T_world_cam @ AriaUtils.T_z_m90.T)

            # Visualize the action
            vis_action = o3d.geometry.TriangleMesh() if VIEWER == "o3d" else []
            gt_right_action = gt_action[:, ACTION_DIM // 2 :]
            if not dataset.action_in_world_frame:
                gt_right_action = DatasetUtils.transform_hand_trajectory(
                    gt_right_action, T_world_cam, has_finger_tips=load_finger_tips
                )
            gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)

            gt_right_action_valid = action_valid[:, ACTION_DIM // 2]  # [H]

            closure = gt_right_action[0, 3]
            cmap_name = "turbo" if closure < 0.5 else "cool"
            if gt_right_action_valid.sum() > 0:
                if VIEWER == "viser":
                    vis_action_right = DatasetUtils.visualize_3d_trajectory(
                        gt_right_root_action[:, :3, 3],
                        size=0.01,
                        cmap_name=cmap_name,
                        to_mesh=False,
                    )
                else:
                    vis_action_right = DatasetUtils.visualize_6d_trajectory(
                        gt_right_root_action,
                        size=0.01,
                        cmap_name=cmap_name,
                        to_mesh=True,
                    )
                # vis_action_right += DatasetUtils.visualize_fingertips_trajectory(
                #     gt_right_action, size=0.01, cmap_name="turbo"
                # )
                vis_action += vis_action_right
                # vis_action += DatasetUtils.visualize_axis_o3d(T_world_cam)
                # vis_action += DatasetUtils.visualize_axis_o3d(np.eye(4), size=0.5)
            vis_trajs.append(vis_action)

        viewer = SceneViewer[VIEWER](
            vis_scenes=vis_scenes,
            vis_trajs=vis_trajs,
            slam_poses=slam_poses,
            viewer_name="Policy Closed Loop",
            front_distance=2.0,
        )
        viewer.run()

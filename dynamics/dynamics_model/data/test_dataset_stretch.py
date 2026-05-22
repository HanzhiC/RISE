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
        training=False,
        # load_rgbd_frames=True,
        load_rgbd_frames=True,
        load_distance_to_goal=True,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    print("========> Size of the dataset: ", len(dataset))

    video_list = dataset.samples
    video_list = ["2026-03-17--13-21-20/0-257"]

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

            print(
                "frame_range: ",
                frame_range,
                "frame_idx: ",
                ii,
                "is_terminal: ",
                data["is_terminal"],
                "gt_td_reward: ",
                data["gt_td_reward"],
                "gt_state_value: ",
                gt_state_value,
            )

            # Visualize the point cloud
            depth[np.logical_or(depth > 1.7, depth < 0.0)] = 0
            points, scene_ids = DatasetUtils.backproject(depth, intrinsics, depth > 0)
            point_colors = color[scene_ids[0], scene_ids[1]] / 255
            points_world = DatasetUtils.transform_points(points, T_world_cam)
            pcd = DatasetUtils.visualize_points(points_world, point_colors)

            # Load the track data
            if dataset.load_tracks:
                start_state = data["start_state"]  # [3, 64, 64]
                history_state = data["history_state"]
                gt_state = data["gt_state"]
                state_color = data["state_color"]  # [3, 64, 64]
                gt_state_res = data["gt_state_residual"]
                gt_state_valid = data["state_valid"]
                gt_distance_to_goal = data["gt_distance_to_goal"]
                gt_state = gt_state.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
                gt_state_valid = gt_state_valid.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
                gt_state_res = gt_state_res.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
                history_state = history_state.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
                vis_dynamics = o3d.geometry.PointCloud()

                track_i_valid = start_state[:, 2] > 0.3
                track_i_last = gt_state[-1].reshape(3, -1).T  # [N, 3]
                track_i_dist2goal = gt_distance_to_goal.reshape(3, -1).T  # [N, 3]
                track_i_init2subgoal = gt_state_res[-1].reshape(3, -1).T  # [N, 3]
                track_i_init = history_state[-1].reshape(3, -1).T  # [N, 3]
                track_i_short_goal = track_i_init + track_i_init2subgoal
                track_i_long_goal = track_i_short_goal + track_i_dist2goal

                if track_i_valid[:, 0].sum() > 0:
                    track_i_long_goal = DatasetUtils.transform_points(
                        track_i_long_goal, T_world_cam0
                    )
                    track_i_short_goal = DatasetUtils.transform_points(
                        track_i_short_goal, T_world_cam0
                    )
                    track_i_vis = DatasetUtils.visualize_points(
                        track_i_short_goal, track_color
                    )
                    # track_i_vis += DatasetUtils.visualize_points(
                    #     track_i_long_goal, track_color
                    # )
                    vis_dynamics += track_i_vis

                    pcd += vis_dynamics
                    # o3d.visualization.draw([pcd, vis_dynamics])

                else:
                    pcd = o3d.geometry.PointCloud()
            vis_scenes.append(pcd)
            slam_poses.append(T_world_cam @ AriaUtils.T_z_m90.T)

            # Visualize the action
            vis_action = o3d.geometry.TriangleMesh() if VIEWER == "o3d" else []
            gt_right_action = gt_action[:, ACTION_DIM // 2 :]
            print("gt_right_action: ", gt_right_action.shape)
            if not dataset.action_in_world_frame:
                gt_right_action = DatasetUtils.transform_hand_trajectory(
                    gt_right_action, T_world_cam, has_finger_tips=load_finger_tips
                )
            gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)

            gt_right_action_valid = action_valid[:, ACTION_DIM // 2]  # [H]
            # if gt_left_action_valid.sum() > 0:
            #     if VIEWER == "viser":
            #         vis_action_left = DatasetUtils.visualize_3d_trajectory(
            #             gt_left_root_action[:, :3, 3],
            #             size=0.01,
            #             cmap_name="plasma",
            #             to_mesh=False,
            #         )
            #     else:
            #         vis_action_left = DatasetUtils.visualize_6d_trajectory(
            #             gt_left_root_action, size=0.01, cmap_name="plasma", to_mesh=True
            #         )
            #     # vis_action_left += DatasetUtils.visualize_fingertips_trajectory(
            #     #     gt_left_action, size=0.01, cmap_name="turbo"
            #     # )
            #     vis_action += vis_action_left
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

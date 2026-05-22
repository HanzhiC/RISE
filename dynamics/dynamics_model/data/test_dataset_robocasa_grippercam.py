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
from random import shuffle
load_finger_tips = False
concatenate_distance_to_goal = True
VIEWER = "o3d"
ACTION_DIM = 20


if __name__ == "__main__":
    dataset = Robot4DDataset(
        # data_dir="/storage/group/srl/robocasa/",
        # split_fpath="data/splits_robocasa/robocasa_PnPCabToCounter_releaseDummy_valid_frame_ranges.csv",
        data_dir="/storage/group/srl/robocasa_downsampled/",
        split_fpath="data/splits_robocasa/robocasa-downsampled_PnPCabToCounter_release_valid_frame_ranges.csv",
        horizon=15,
        load_hands=True,
        load_tracks=False,
        build_track_online=False,
        track_by_clip=True,
        track_patch_size=64,
        concatenate_distance_to_goal=concatenate_distance_to_goal,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        load_finger_tips=False,
    )
    print("========> Size of the dataset: ", len(dataset))

    video_list = dataset.samples
    # shuffle(video_list)
    for video_name in video_list:
        video_data = dataset._get_video(video_name)
        vis_scenes, vis_trajs, slam_poses = [], [], []
        track_color = DatasetUtils.random_colors(64 * 64)
        track_color = np.array(track_color)
        # video_data = video_data[:200]
        for ii, data in enumerate(video_data):

            # Visualize the data
            color = data["color_gripper"]
            depth = data["depth_gripper"]
            intrinsics = data["intrinsics_gripper"]
            T_world_cam = data["T_world_cam"]
            T_world_grippercam = data["T_world_grippercam"]
            action_valid = data["action_valid"]
            gt_action = data["gt_action"]  # in head cam frame
            # depth[depth > 0.8] = 0

            # # Load the track data
            # start_state = data["start_state"]  # [3, 64, 64]
            # gt_state = data["gt_state"]
            # gt_state_res = data["gt_state_residual"]
            # gt_state_valid = data["state_valid"]
            # gt_state = gt_state.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
            # gt_state_valid = gt_state_valid.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
            # gt_state_res = gt_state_res.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]

            # dist_to_goal = gt_state_res[-1]
            # print("Distance to goal: ", (dist_to_goal**2).mean() ** 0.5)

            # # Todo: interpolate the 7 frames before and after the current frame to visualize the dynamics
            # vis_dynamics = o3d.geometry.PointCloud()
            # # Visualize the last frame (index horizon-1) of the state trajectory
            # last_frame_idx = gt_state_res.shape[0] - 1
            # if last_frame_idx >= 0:
            #     track_i_res = gt_state_res[last_frame_idx].reshape(3, -1).T  # [N, 3]
            #     track_i_start = start_state.reshape(3, -1).T  # [N, 3]
            #     track_i = track_i_res + track_i_start
            #     if dataset.track_by_clip and not concatenate_distance_to_goal:
            #         track_i = DatasetUtils.transform_points(track_i, T_cam0_cam)
            #     track_i_vis = DatasetUtils.visualize_points(track_i, track_color)
            #     vis_dynamics += track_i_vis
            # # pcd += vis_dynamics

            # Visualize the point cloud
            points, scene_ids = DatasetUtils.backproject(depth, intrinsics, depth > 0)
            point_colors = color[scene_ids[0], scene_ids[1]] / 255
            points_world = DatasetUtils.transform_points(points, T_world_grippercam)
            pcd = DatasetUtils.visualize_points(points_world, point_colors)
            vis_scenes.append(pcd)
            slam_poses.append(T_world_grippercam @ AriaUtils.T_z_m90.T)

            # Visualize the action
            vis_action = o3d.geometry.TriangleMesh() if VIEWER == "o3d" else []
            gt_right_action = gt_action[:, ACTION_DIM // 2 :]
            gt_right_action = DatasetUtils.transform_hand_trajectory(
                gt_right_action, T_world_cam, has_finger_tips=load_finger_tips
            )
            gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)

            gt_right_action_valid = action_valid[:, ACTION_DIM // 2]  # [H]
            closure = gt_right_action[:, 3:4]
            if closure[0] < 0: # Open
                cmap_name = "cool"
            else: # Close
                cmap_name = "turbo" 
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
            vis_trajs.append(vis_action)

        viewer = SceneViewer[VIEWER](
            vis_scenes=vis_scenes,
            vis_trajs=vis_trajs,
            slam_poses=slam_poses,
            viewer_name="Policy Closed Loop",
            front_distance=2.0,
        )
        viewer.run()

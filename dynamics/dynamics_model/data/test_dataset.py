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

load_finger_tips = False
load_distance_to_goal = False
ACTION_DIM = 48 if load_finger_tips else 20
VIEWER_TYPE = "o3d"

if __name__ == "__main__":
    dataset = Egoasis4DDataset(
        # split_fpath="data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv",
        # split_fpath="data/splits/hoiarti4d_releaseclipTest_valid_frame_ranges_hoi4dlegacy.csv",
        # split_fpath="data/splits/hoiarti4d_releaseclipDummy_valid_frame_ranges.csv",
        # split_fpath="data/splits/mrl4d_releaseclip_valid_frame_ranges.csv",
        split_fpath="data/splits/egoasis4d_releaseclip_valid_frame_ranges.csv",
        horizon=15,
        action_chunk_size=45,
        load_tracks=True,
        load_hands=True,
        clip_length=1,
        track_patch_size=64,
        track_horizon_of_interest=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        build_track_online=True,
        track_by_clip=False,
        load_distance_to_goal=load_distance_to_goal,
        load_finger_tips=load_finger_tips,
        action_in_relative=False,
    )
    print("========> Size of the dataset: ", len(dataset))
    video_list = [
        "basic_pick_place/9978/0-42"
        # "rh201/scene_2025-04-24-19-21-50/545-567",
        # "ZY20210800004/H4/C3/N67/S380/s05/T2.000",
        # "basic_pick_place/1/0-46",
        # "basic_pick_place/0/0-43",
        # "basic_pick_place/15/0-82"
        # "ZY20210800004/H4/C3/N08/S97/s04/T2.000",  # Open laptop
    ]
    # video_list = dataset.samples

    # video_list = [sample for sample in video_list if sample.count("/") == 2]
    # video_data = dataset._get_video("rh201/scene_2025-04-24-19-21-50/545-567")
    for video_name in video_list:
        video_data = dataset._get_video(video_name)
        vis_scenes, vis_trajs, slam_poses = [], [], []
        track_color = DatasetUtils.random_colors(64 * 64)
        track_color = np.array(track_color)
        for ii, data in enumerate(video_data):
            # Visualize the data
            color = data["color"]
            depth = data["depth"]
            intrinsics = data["intrinsics"]
            T_cam0_cam = data["T_cam0_cam"]
            slam_pose = data["slam_pose"]
            start_pos = data["start_pos"]
            action_valid = data["action_valid"]
            gt_action = data["history_action"]
            # mask_dynamic_init = data["mask_dynamic_init"]

            # Load the track data
            start_state = data["start_state"]  # [3, 64, 64]
            gt_state = data["gt_state"]
            gt_state_res = data["gt_state_residual"]
            gt_state_valid = data["state_valid"]
            gt_state = gt_state.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
            gt_state_valid = gt_state_valid.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]
            gt_state_res = gt_state_res.reshape(-1, 3, 64, 64)  # [H, 3, 64, 64]

            # Todo: interpolate the 7 frames before and after the current frame to visualize the dynamics
            vis_dynamics = o3d.geometry.PointCloud()
            # Visualize the last frame (index horizon-1) of the state trajectory

            # Very important: we now know how to parse the state ...
            track_i_last = gt_state[-1].reshape(3, -1).T  # [N, 3]
            track_i_vis = DatasetUtils.visualize_points(track_i_last, track_color)
            vis_dynamics += track_i_vis

            # Visualize the point cloud
            points, scene_ids = DatasetUtils.backproject(depth, intrinsics, depth > 0)
            point_colors = color[scene_ids[0], scene_ids[1]] / 255
            points_world = DatasetUtils.transform_points(points, T_cam0_cam)
            pcd = DatasetUtils.visualize_points(points_world, point_colors)
            pcd += vis_dynamics
            vis_scenes.append(pcd)
            slam_poses.append(T_cam0_cam @ AriaUtils.T_z_m90.T)

            # Visualize the action
            vis_action = o3d.geometry.TriangleMesh() if VIEWER_TYPE == "o3d" else []
            gt_action = DatasetUtils.transform_two_hands_trajectory(
                gt_action, T_cam0_cam, ACTION_DIM, has_finger_tips=load_finger_tips
            )
            gt_left_action = gt_action[:, : ACTION_DIM // 2]
            gt_right_action = gt_action[:, ACTION_DIM // 2 :]
            # gt_left_action = DatasetUtils.transform_hand_trajectory(
            #     gt_left_action, T_cam0_cam, has_finger_tips=load_finger_tips
            # )
            # gt_right_action = DatasetUtils.transform_hand_trajectory(
            #     gt_right_action, T_cam0_cam, has_finger_tips=load_finger_tips
            # )
            gt_left_root_action = DatasetUtils.get_root_transformation(gt_left_action)
            gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)

            gt_left_action_valid = action_valid[:, 0]  # [H]
            gt_right_action_valid = action_valid[:, ACTION_DIM // 2]  # [H]
            if gt_left_action_valid.sum() > 0:
                if VIEWER_TYPE == "o3d":
                    vis_action_left = DatasetUtils.visualize_6d_trajectory(
                        gt_left_root_action, size=0.01, cmap_name="plasma", to_mesh=True
                    )
                else:
                    vis_action_left = DatasetUtils.visualize_3d_trajectory(
                        gt_left_root_action[:, :3, 3],
                        size=0.01,
                        cmap_name="plasma",
                        to_mesh=False,
                    )
                vis_action += vis_action_left

            if gt_right_action_valid.sum() > 0:
                if VIEWER_TYPE == "o3d":
                    vis_action_right = DatasetUtils.visualize_6d_trajectory(
                        gt_right_root_action,
                        size=0.01,
                        cmap_name="plasma",
                        to_mesh=True,
                    )
                else:
                    vis_action_right = DatasetUtils.visualize_3d_trajectory(
                        gt_right_root_action[:, :3, 3],
                        size=0.01,
                        cmap_name="plasma",
                        to_mesh=False,
                    )
                vis_action += vis_action_right
            vis_trajs.append(vis_action)

        viewer = SceneViewer[VIEWER_TYPE](
            vis_scenes=vis_scenes,
            vis_trajs=vis_trajs,
            slam_poses=slam_poses,
            viewer_name="Policy Closed Loop",
            front_distance=2.0,
        )
        viewer.run()

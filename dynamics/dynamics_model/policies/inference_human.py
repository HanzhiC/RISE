import argparse
import sys
import os
import socket

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import utils.dataset_utils as DatasetUtils
from utils.inference_utils import get_ray_map_in_batch, prepare_cut3r_input
from omegaconf import OmegaConf

from utils.viewer_utils import SceneViewer
import utils.aria_utils as AriaUtils
from policies.policy_wrapper import PolicyVLAWorldModelWrapper
import torch
import time
import options
import open3d as o3d
import numpy as np
from data.data_factory import datamodule_factory
from tqdm import tqdm
import cv2
from collections import deque
import torch.nn.functional as F
import time
import einops
DEFAULT_FPS = 15
QUERY_HORIZON = 15
EXECUTE_HORIZON = 15
SMOOTH_WEIGHT = 0.1  # Favor recent actions
TRAJ_SCALE = 1.0
VIEWER_TYPE = "o3d"  # "viser" or "o3d"
KEYS_FOR_STREAMING = [
    # Action Meta
    "action_valid",
    "action_mean",
    "action_std",
    "action_norm_max_bound",
    "action_norm_min_bound",
    "gt_action",
    # Dynamics Meta
    "state_valid",
    "state_mean",
    "state_std",
    "state_norm_max_bound",
    "state_norm_min_bound",
    "gt_state",
    "gt_state_residual",
    "state_color",
    # Sensor Data
    "language_feature",
    "color",  # [1, 3, H, W], (0,1); RGB <= [1, H, W, 3]; (0,255); RGB
    "depth",  # [1, H, W], m <= [1, H, W]. m,
    "intrinsics",  # [1, 3, 3] <= [3, 3]
    "T_world_cam",  # [1, 4, 4] <= [4, 4]
    "start_pos",  # [1, 17] <= [1, 20]
    "T_cam0_cam",
    "start_state",  # "start_state_dinov3_feature",
    # History Data
    "history_state",
    "history_action",
    "history_action_rel",
    "history_raymap",
    "history_visual_feature_patch",
]

# INSTRUCTION = "carry laptop"
# INSTRUCTION = "pick up laptop"
# INSTRUCTION = "open laptop"  # "open laptop"
INSTRUCTION = "carry laptop"
INSTRUCTION = None


@torch.no_grad()
def main(cfg):
    ACTION_DIM = cfg.ALGORITHM.model.action_dim 
    cfg.DATA.load_tracks = True
    datamodule = datamodule_factory(cfg)
    datamodule.setup()

    weight_fpath = os.path.join(cfg.log_dir, cfg.name, "last.ckpt")

    val_dataset = datamodule.val_dataset
    sample_list = [
        "ZY20210800004/H4/C3/N67/S380/s05/T2.003",
        # "ZY20210800002/H2/C1/N01/S336/s02/T1.004",
        # "ZY20210800004/H4/C3/N67/S380/s05/T2.001",  # Move to the right
        # "ZY20210800004/H4/C3/N67/S380/s05/T2.000",  # Pick up laptop
        # "rh201/scene_2025-04-24-19-21-50/545-567",
    ]

    # Or simply run on all validation samples:
    sample_list = val_dataset.samples
    sample_list = [sample for sample in sample_list if sample.count("/") == 2]
    policy = PolicyVLAWorldModelWrapper(
        cfg,
        weight_ckpt=weight_fpath,
        action_chunk_size=1,
        online_update_robot_state=False,
        online_update_visual_state=False,
        online_update_extrinsics_state=False,
        online_update_envrionment_state=False,
        policy_only=False,
    )

    # Visualization containers
    dynamics_color = DatasetUtils.random_colors(4096)
    dynamics_color = np.array(dynamics_color)

    for sidx, sample in enumerate(sample_list):
        vis_actions = []
        vis_scenes = []
        vis_slam_poses = []
        print(f"[INFO] Running sample: {sample}")
        sample_idx = val_dataset.sample_name_to_index[sample]
        video_data = val_dataset._get_video(sample, to_tensor=True)
        dataset_name = val_dataset.dataset_categories[sample_idx]
        last_frame_idx = len(video_data) - 1
        policy.reset()
        for i, _data_batch in tqdm(
            enumerate(video_data),
            total=last_frame_idx + 1,
            desc="Online joint policy rollout ...",
        ):
            if i > last_frame_idx:
                break

            # Build a batch of size 1
            _data_batch = {k: v[None].to(policy.device) for k, v in _data_batch.items()}
            data_batch = {k: _data_batch[k] for k in KEYS_FOR_STREAMING}
            del data_batch["T_world_cam"]
            data_batch["T_world_cam"] = data_batch["T_cam0_cam"].clone()
            data_batch["start_pos_world"] = DatasetUtils.transform_two_hands_trajectory(
                data_batch["start_pos"],  # [B, D]
                data_batch["T_world_cam"][0],  # [4, 4]
                has_finger_tips=False,
            )

            if INSTRUCTION is not None:
                data_batch["language_instruction"] = INSTRUCTION

            # Do the inference
            outputs = policy.inference(data_batch, action_only=True, align_to_current_state=True)
            pred_action_chunk = outputs["latest_predicted_action"].cpu().numpy()
            pred_progress = pred_action_chunk[..., -1].mean().item()
            pred_action_chunk = pred_action_chunk[..., :ACTION_DIM]
            print(f" ===> Current progress: {pred_progress:.3f}")

            # ------------------------------------------------------------------
            #  Build point cloud for current frame scene
            # ------------------------------------------------------------------
            color = _data_batch["color"][0].cpu().numpy().transpose(1, 2, 0)
            depth = _data_batch["depth"][0].cpu().numpy()
            intr = _data_batch["intrinsics"][0].cpu().numpy()
            T_world_cam = data_batch["T_world_cam"][0].cpu().numpy()

            points, scene_ids = DatasetUtils.backproject(
                depth, intr, depth < 1.5, False
            )
            points_rgb = color[scene_ids[0], scene_ids[1]]
            points_world = DatasetUtils.transform_points(points, T_world_cam)
            pcd_scene = DatasetUtils.visualize_points(points_world, colors=points_rgb)
            vis_scenes.append(pcd_scene)
            vis_slam_poses.append(T_world_cam @ AriaUtils.T_z_m90.T)
            if outputs["latest_dynamics"] is not None:
                valid_dynamics = data_batch["state_valid"][0].view(-1, 3, 64, 64)
                valid_dynamics_np = valid_dynamics.cpu().numpy()
                dynamics_id = 1 if dataset_name == "hoi4d" else 0
                dynamics_mask = (valid_dynamics_np[0] > dynamics_id).transpose(
                    1, 2, 0
                )  # [64, 64, 3]

                dynamics_mask = cv2.dilate(
                    dynamics_mask.astype(np.uint8), np.ones((3, 3)), iterations=1
                )
                dynamics_mask = (dynamics_mask * 255).astype(np.uint8)

                dynamics_mask = dynamics_mask.reshape(-1, 3)

                # Get the latest dynamics
                pred_state = outputs["latest_dynamics"].cpu().numpy()
                pred_state_last = pred_state[-1]
                pred_state_last = pred_state_last.reshape(3, -1).T
                pred_state_valid = pred_state_last[:, 2] > 0.3
                dynamics_mask = dynamics_mask.sum(axis=1) > 0
                pred_state_valid = pred_state_valid & dynamics_mask
                pred_state_color = dynamics_color[pred_state_valid]
                state_vis = DatasetUtils.visualize_points(
                    pred_state_last[pred_state_valid],
                    colors=pred_state_color,
                    as_spheres=False,
                    size=0.01,
                )
                vis_scenes[-1] += state_vis

            # Acquire the action trajectory
            action_valid = data_batch["action_valid"][0].cpu().numpy() # [H, ACTION_DIM]
            gt_action = data_batch["gt_action"][0].cpu().numpy()[:, :ACTION_DIM] # [H, ACTION_DIM]
            gt_action = DatasetUtils.transform_two_hands_trajectory(
                gt_action,
                T_world_cam,
                has_finger_tips=False,
            )
            vis_action = o3d.geometry.TriangleMesh() if VIEWER_TYPE == "o3d" else []
            vis_action_pred = DatasetUtils.draw_two_hands_trajectory(
                pred_action_chunk,
                action_valid,
                viewer_type=VIEWER_TYPE,
                cmap_name="turbo",
            )
            # vis_action_gt = DatasetUtils.draw_two_hands_trajectory(
            #     gt_action,
            #     action_valid,
            #     viewer_type=VIEWER_TYPE,
            #     color=np.array([0, 1, 0]),
            # )
            # vis_action += vis_action_gt

            vis_action += vis_action_pred
            vis_actions.append(vis_action)

        viewer = SceneViewer[VIEWER_TYPE](
            vis_scenes=vis_scenes,
            vis_trajs=vis_actions,
            slam_poses=vis_slam_poses,
            pcd_slam=None,
            viewer_name=f"Joint Policy",
            port=int(8089 + sidx),
        )
        viewer.run()


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.print_options(opt)
    options.save_options_file(opt)
    main(opt)
    # find_lastest_checkpoint(opt)

# python train_scripts/stream_action_and_dynamics_w_chunking_online.py  --config=/home/wiss/chenh/storage/logs/egoasis4d-action/vanilla_vla_wm_cditAbsPos_ActTransformer_trainonfullMixed/config.yaml  --DATA.split_fpath_test=data/splits/hoi4d_dummy_valid_frame_ranges_w_trajectory.csv

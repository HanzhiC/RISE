import argparse
import sys
import os
import socket

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import utils.dataset_utils as DatasetUtils
from utils.inference_utils import (
    get_ray_map_in_batch,
    prepare_cut3r_input,
    populate_queues,
)
from omegaconf import OmegaConf
from utils.viewer_utils import SceneViewer
import utils.aria_utils as AriaUtils
from algos.vla_wm_predictor import VLAWorldModelModule
from algos.cut3r_predictor import Cut3RGeometryPredictor
from policies.policy_wrapper import (
    PolicyVLAWorldModelWrapper,
    OBS_ENV_STATE,
    OBS_ROBOT_STATE,
    OBS_VISUAL_STATE,
    OBS_HEAD_CAM_STATE,
    HISTORY_OBS_ENV_STATE,
    HISTORY_OBS_ROBOT_STATE,
    HISTORY_OBS_VISUAL_STATE,
    HISTORY_OBS_HEAD_CAM_STATE,
)
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
import einops
from scipy.spatial.transform import Rotation as SciR
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot

# Prediction
ACTION = "action"

# Observation key
OBS_GRIPPER_VISUAL_STATE = "visual_feature_patch_gripper"
OBS_GRIPPER_CAM_STATE = "T_world_grippercam"
HISTORY_OBS_GRIPPER_VISUAL_STATE = "history_visual_feature_patch_gripper"
HISTORY_OBS_GRIPPER_CAM_STATE = "history_raymap_gripper"


class PolicyVLAWorldModelWrapperRoboCasa(PolicyVLAWorldModelWrapperStretchRobot):
    TARGET_RESOLUTION = 224

    def __init__(
        self,
        cfg,
        action_chunk_size=8,
        online_update_robot_state=True,
        online_update_visual_state=True,
        online_update_extrinsics_state=True,
        online_update_environment_state=True,
        weight_ckpt=None,
        policy_only=False,
        device="cuda",
        run_on_robot=False,
        action_meta_fpath=None,
        state_meta_fpath=None,
        **kwargs,
    ):
        super(PolicyVLAWorldModelWrapperRoboCasa, self).__init__(
            cfg=cfg,
            action_chunk_size=action_chunk_size,
            online_update_visual_state=online_update_visual_state,
            online_update_extrinsics_state=online_update_extrinsics_state,
            online_update_robot_state=online_update_robot_state,
            online_update_environment_state=online_update_environment_state,
            run_on_robot=run_on_robot,
            action_meta_fpath=action_meta_fpath,
            state_meta_fpath=state_meta_fpath,
            policy_only=policy_only,
            weight_ckpt=weight_ckpt,
            device=device,
        )

    def _postprocess_action(self, outputs, data_batch):
        pred_actions, pred_progress = PolicyVLAWorldModelWrapper._postprocess_action(
            self, outputs, data_batch
        )
        if not self.run_on_robot:
            action_dim = self.cfg.ALGORITHM.model.action_dim
            pred_actions = pred_actions[:, action_dim // 2 :]
            pred_actions_xyz = pred_actions[:, :3]  # [H, 3]
            pred_actions_aux = pred_actions[:, 3:-6]  # [H, 1]
            pred_actions_r6d = pred_actions[:, -6:]  # [H, 6]
            pred_actions_rot = AriaUtils.rotation_6d_to_matrix(
                pred_actions_r6d
            )  # [H, 3, 3];

            # Convert rotation matrix to quaternion (batch)
            pred_actions_qua = SciR.from_matrix(
                pred_actions_rot.cpu().numpy()
            ).as_quat()  # [H, 4]; xyzw
            pred_actions_qua = torch.from_numpy(pred_actions_qua).float()  # [H, 4]
            pred_actions = torch.cat(
                [
                    pred_actions_xyz,
                    pred_actions_qua,
                    pred_actions_aux,
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError("Running on robot is not implemented yet")

        return pred_actions, pred_progress

    def _preprocess_observation(self, data_batch):
        PolicyVLAWorldModelWrapper._preprocess_observation(self, data_batch)

        # Add the action statistics
        if self.action_meta_fpath is not None:
            data_batch["action_mean"] = self.action_mean[None].to(self.device)
            data_batch["action_std"] = self.action_std[None].to(self.device)
            data_batch["action_norm_max_bound"] = self.action_norm_max_bound[None].to(
                self.device
            )
            data_batch["action_norm_min_bound"] = self.action_norm_min_bound[None].to(
                self.device
            )

        if self.state_meta_fpath is not None:
            data_batch["state_mean"] = self.state_mean[None].to(self.device)
            data_batch["state_std"] = self.state_std[None].to(self.device)
            data_batch["state_norm_max_bound"] = self.state_norm_max_bound[None].to(
                self.device
            )
            data_batch["state_norm_min_bound"] = self.state_norm_min_bound[None].to(
                self.device
            )

        if self.run_on_robot:
            raise NotImplementedError("Running on robot is not implemented yet")

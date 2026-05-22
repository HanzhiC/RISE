import os
import utils.dataset_utils as DatasetUtils
from utils.inference_utils import (
    get_ray_map_in_batch,
    prepare_cut3r_input,
    populate_queues,
)
import utils.aria_utils as AriaUtils
from algos.vla_wm_predictor import VLAWorldModelModule
from algos.cut3r_predictor import Cut3RGeometryPredictor
from policies.policy_wrapper import PolicyVLAWorldModelWrapper
from policies.conventions import *
import torch
import time
import numpy as np
import cv2
from collections import deque
from einops import rearrange
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot
from policies.policy_wrapper import PolicyVLAWorldModelWrapper

class PolicyVLAWorldModelWrapperExploration(PolicyVLAWorldModelWrapperStretchRobot):
    GRIPPER_IMAGE_SIZE = (240, 320)
    HEAD_IMAGE_SIZE = (224, 224)  # H, W

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
        action_smoothing_alpha=0.3,
        action_temporal_smoothing=True,
        head_image_crop_mode="bottom",
        action_xyz_offset=[0.0, 0.0, 0.0],
        action_xyz_scale=1.0,
        **kwargs,
    ):
        super(PolicyVLAWorldModelWrapperExploration, self).__init__(
            cfg=cfg,
            action_chunk_size=action_chunk_size,
            online_update_robot_state=online_update_robot_state,
            online_update_visual_state=online_update_visual_state,
            online_update_extrinsics_state=online_update_extrinsics_state,
            online_update_environment_state=online_update_environment_state,
            weight_ckpt=weight_ckpt,
            policy_only=policy_only,
            device=device,
            run_on_robot=run_on_robot,
            action_meta_fpath=action_meta_fpath,
            state_meta_fpath=state_meta_fpath,
            action_smoothing_alpha=action_smoothing_alpha,
            action_temporal_smoothing=action_temporal_smoothing,
        )
        self.head_image_crop_mode = head_image_crop_mode
        self.action_xyz_offset = torch.tensor(action_xyz_offset, device=device)
        self.action_xyz_scale = action_xyz_scale

    def _extract_visual_features(self, data_batch):
        # Extract the visual features for the head camera, no need for gripper camera
        PolicyVLAWorldModelWrapper._extract_visual_features(self, data_batch)

    def _prepare_history_conditioning(self, data_batch):
        # Acquire the current state
        has_finger_tips = self.cfg.ALGORITHM.model.action_dim == 48
        T_world_cam_curr = data_batch[OBS_HEAD_CAM_STATE].clone()  # [B, 4, 4]
        start_pos_world = data_batch[OBS_ROBOT_STATE].clone()  # [B,D]
        intrinsics = data_batch["intrinsics"].clone()  # [B, 3, 3]
        T_cam_world = torch.linalg.pinv(T_world_cam_curr)[0]  # [4, 4], compute once

        # Extract the history of the environment state
        history_env_state = self._queues.get(OBS_ENV_STATE, None)
        history_robot_state = self._queues.get(OBS_ROBOT_STATE, None)  # in world frame
        history_visual_state = self._queues.get(OBS_VISUAL_STATE, None)
        history_head_cam_state = self._queues.get(OBS_HEAD_CAM_STATE, None)
        # Get the history environment states
        if history_env_state is not None and self.online_update_environment_state:
            history_env_state = torch.stack(list(history_env_state), dim=0)
            history_env_state = history_env_state.flatten(
                start_dim=0, end_dim=2
            )  # [T*3, 64, 64]
            history_env_state = history_env_state[None].contiguous()
            data_batch[HISTORY_OBS_ENV_STATE] = history_env_state

        # Get the history tesnors
        if history_visual_state is not None and self.online_update_visual_state:
            history_visual_state = torch.stack(list(history_visual_state), dim=0)
            history_visual_state = history_visual_state[None].contiguous()
            # history_visual_state_gt = data_batch["history_visual_feature_patch"]
            # diff = (history_visual_state - history_visual_state_gt).abs()
            # diff_val = (diff**2).mean() ** 0.5
            # print(f"Diff of history_visual_feature_patch: {diff_val:.10f}")
            # print(f"mean absolute difference: {diff.mean():.10f}")
            # print(f"max absolute difference: {diff.max():.10f}")
            data_batch[HISTORY_OBS_VISUAL_STATE] = history_visual_state

        # Stack all history transforms first, then do batched matmul
        if history_head_cam_state is not None and self.online_update_extrinsics_state:
            # history_T_world = torch.stack(
            #     [T_world_history[0] for T_world_history in history_head_cam_state]
            # )  # [T, 4, 4]

            # history_T_cam_history = (
            #     T_cam_world @ history_T_world
            # )  # [4, 4] @ [T, 4, 4] -> [T, 4, 4]
            history_T_cam_history = torch.eye(4)[None].repeat(len(history_head_cam_state), 1, 1).to(self.device) #  Hard code to identity!
            history_raymap = get_ray_map_in_batch(
                history_T_cam_history,
                intrinsics,
                original_size=self.HEAD_IMAGE_SIZE,
                target_size=(
                    self.HEAD_IMAGE_SIZE[0] // 16,
                    self.HEAD_IMAGE_SIZE[1] // 16,
                ),
            )
            history_raymap = history_raymap.permute(0, 3, 1, 2)  # [T, 6, H, W]
            history_raymap = history_raymap[None].contiguous()
            data_batch[HISTORY_OBS_HEAD_CAM_STATE] = history_raymap
            # history_raymap_gt = data_batch["history_raymap"]
            # diff = (history_raymap - history_raymap_gt).abs()
            # diff_val = (diff**2).mean() ** 0.5
            # print(f"Diff of history_raymap: {diff_val:.10f}")

        if history_robot_state is not None and self.online_update_robot_state:
            history_robot_abs_state = torch.stack(list(history_robot_state), dim=0)[
                :, 0
            ]  # [T, D], start_pos_world, in the world frame

            # Change the history robot state to the camera frame if
            # the action is predicted in the camera frame
            if self.cfg.ALGORITHM.model.am_predict_action_frame == "camera":
                history_robot_abs_state = DatasetUtils.transform_two_hands_trajectory(
                    history_robot_abs_state,
                    T_cam_world,
                    has_finger_tips=has_finger_tips,
                )  # start_pos, from world frame to camera frame
                assert not has_finger_tips, "Finger tips are not supported for exploration policy"
                # Match dataset-side history processing:
                # 1) right-multiply local x translation offset by -0.25m
                # 2) right-multiply rotation by Rz(180)
                # 3) set closure to 0.5
                dt_local = torch.tensor(
                    [-0.25, 0.0, 0.0],
                    device=history_robot_abs_state.device,
                    dtype=history_robot_abs_state.dtype,
                )
                rz_180 = torch.tensor(
                    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                    device=history_robot_abs_state.device,
                    dtype=torch.float32,
                )
                hand_dim = history_robot_abs_state.shape[-1] // 2
                for hand_start in (0, hand_dim):
                    hand_state = history_robot_abs_state[
                        :, hand_start : hand_start + hand_dim
                    ]
                    rot = AriaUtils.rotation_6d_to_matrix(hand_state[:, -6:].float())
                    hand_state[:, :3] = hand_state[:, :3] + torch.matmul(
                        rot, dt_local[:, None].float()
                    )[:, :, 0].to(hand_state.dtype)
                    rot = torch.matmul(rot, rz_180[None])
                    hand_state[:, -6:] = AriaUtils.matrix_to_rotation_6d(rot).to(
                        hand_state.dtype
                    )
                    hand_state[:, 3] = 0.5

            # # Build the history robot state in relative or absolute format;
            # # depending on the action frame and whether to use relative action
            # if self.cfg.ALGORITHM.model.am_use_relative_action:
            #     if self.cfg.ALGORITHM.model.am_predict_action_frame == "world":
            #         # Start pos is in the world frame
            #         start_pos = start_pos_world[0]
            #     else:
            #         # Start pos is in the camera frame
            #         start_pos = DatasetUtils.transform_two_hands_trajectory(
            #             start_pos_world,  # [B, D]
            #             T_cam_world,  # [4, 4]
            #             has_finger_tips=has_finger_tips,
            #         )[0]

            #     # Change the history robot state to relative to the start pos
            #     history_robot_state = (
            #         DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
            #             history_robot_abs_state[None],  # [1, H, D]
            #             start_pos[None],  # [1, D]
            #             has_finger_tips=has_finger_tips,
            #         )
            #     )[0]
            #     history_robot_state = history_robot_state[None].contiguous()

            #     # ################ DEBUG CODE ################
            #     # history_robot_state_rel_gt = data_batch["history_action_rel"]
            #     # offset = (history_robot_state_rel_gt - history_robot_state).abs()
            #     # offset_val = (offset**2).mean() ** 0.5
            #     # all_close = torch.allclose(
            #     #     history_robot_state, history_robot_state_rel_gt, atol=1e-3
            #     # )
            #     # print(
            #     #     f"Offset mean: {offset_val:.10f}, max: {offset.max():.10f}, min: {offset.min():.10f}, all close: {all_close} \n"
            #     # )
            #     # if not all_close:
            #     #     print(f"!!!!!!!!!!!!!!!!! Offset is too large: {offset_val:.10f}")
            #     # # # ################ DEBUG CODE ################
            # else:

            history_robot_state = history_robot_abs_state[None].contiguous()
            # history_robot_state_gt = data_batch["history_action"]
            # diff = (history_robot_state - history_robot_state_gt).abs()
            # diff_val = (diff**2).mean() ** 0.5
            # print(f"Diff of history_action: {diff_val:.10f}")

            data_batch[f"{HISTORY_OBS_ROBOT_STATE}"] = history_robot_state

    def _postprocess_action(self, outputs, data_batch):
        # Use StretchRobot path: world/cam transform + right-hand slice + r6d -> quaternion.
        pred_actions, pred_progress = super()._postprocess_action(outputs, data_batch)
        # Exploration-only: rescale translation magnitude about first waypoint, then bias.
        pred_actions_xyz = pred_actions[:, :3]
        pred_actions_rest = pred_actions[:, 3:]
        pred_actions_xyz = DatasetUtils.descale_trajectory_length(
            pred_actions_xyz[None], self.action_xyz_scale
        )[0]
        off = self.action_xyz_offset.to(
            device=pred_actions_xyz.device, dtype=pred_actions_xyz.dtype
        )
        pred_actions_xyz = pred_actions_xyz + off
        pred_actions = torch.cat([pred_actions_xyz, pred_actions_rest], dim=-1)
        return pred_actions, pred_progress

    def reset(self):
        PolicyVLAWorldModelWrapper.reset(self)

    def _crop_resize_head_and_update_intrinsics(
        self, color, depth, intrinsics, crop_mode="center"
    ):
        height, width = color.shape[:2]
        crop_size = min(height, width)

        crop_x = (width - crop_size) // 2
        if crop_mode == "upper":
            crop_y = 0
        elif crop_mode == "center":
            crop_y = (height - crop_size) // 2
        elif crop_mode == "bottom":
            crop_y = height - crop_size
        else:
            raise ValueError(
                f"Invalid head_image_crop_mode: {crop_mode}. "
                "Expected one of ['upper', 'center', 'bottom']."
            )

        color = color[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        depth = depth[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]

        target_h, target_w = self.HEAD_IMAGE_SIZE
        color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        intrinsics = intrinsics.copy()
        sx = target_w / crop_size
        sy = target_h / crop_size
        intrinsics[0, 0] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[0, 2] = (intrinsics[0, 2] - crop_x) * sx
        intrinsics[1, 2] = (intrinsics[1, 2] - crop_y) * sy

        return color, depth, intrinsics

    def _preprocess_observation(self, data_batch):
        super()._preprocess_observation(data_batch)

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

        if not self.run_on_robot:
            return

        # Accessing the sensor data
        color = data_batch["observation.images.head"] / 255.0
        color_gripper = data_batch["observation.images.gripper"] / 255.0
        depth = data_batch["observation.depths.head"]
        state = data_batch["observation.state"]  # D
        intrinsics = data_batch["HEAD_CAM_K"].copy()
        intrinsics_gripper = data_batch["EE_CAM_K"].copy()
        T_world_cam = data_batch["head_cam_pose"]
        T_world_grippercam = data_batch["ee_cam_pose"]

        # Crop/resize head image and keep intrinsics in sync.
        color, depth, intrinsics = self._crop_resize_head_and_update_intrinsics(
            color=color,
            depth=depth,
            intrinsics=intrinsics,
            crop_mode=self.head_image_crop_mode,
        )

        # do resize to the images
        color_gripper = cv2.resize(
            color_gripper,
            (self.GRIPPER_IMAGE_SIZE[1], self.GRIPPER_IMAGE_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        height_gripper, width_gripper = color_gripper.shape[:2]

        # do resize to the intrinsics for the gripper camera
        intrinsics_gripper[0, 0] *= self.GRIPPER_IMAGE_SIZE[1] / width_gripper
        intrinsics_gripper[0, 2] *= self.GRIPPER_IMAGE_SIZE[1] / width_gripper
        intrinsics_gripper[1, 1] *= self.GRIPPER_IMAGE_SIZE[0] / height_gripper
        intrinsics_gripper[1, 2] *= self.GRIPPER_IMAGE_SIZE[0] / height_gripper

        # transform the state to our convention, double check with Anran
        T_world_gripper, gripper_closure = state[:16].reshape(4, 4), state[16:]
        tra_world_gripper, rot_world_gripper = (
            T_world_gripper[:3, 3],
            T_world_gripper[:3, :3],
        )
        r6d_world_gripper = AriaUtils.matrix_to_rotation_6d(
            torch.from_numpy(rot_world_gripper).float()[None]
        ).numpy()[
            0
        ]  # [6]
        gripper_state = np.concatenate(
            [tra_world_gripper, gripper_closure, r6d_world_gripper], axis=-1
        )  # [10]
        gripper_state = np.concatenate([gripper_state, gripper_state], axis=0)  # [20]
        gripper_state_cam = DatasetUtils.transform_two_hands_trajectory(
            gripper_state[None],  # [B, D]
            np.linalg.inv(T_world_cam),  # [4, 4]
            has_finger_tips=False,
        )[
            0
        ]  # Start pos is correct
        color = color.transpose(2, 0, 1)
        color_gripper = color_gripper.transpose(2, 0, 1)

        # To tensor
        data_batch["color"] = torch.from_numpy(color).float()[None]
        data_batch["color_init"] = torch.from_numpy(color).float()[None]
        data_batch["depth"] = torch.from_numpy(depth).float()[None]
        data_batch["intrinsics"] = torch.from_numpy(intrinsics).float()[None]
        data_batch["intrinsics_gripper"] = torch.from_numpy(intrinsics_gripper).float()[
            None
        ]
        data_batch["color_gripper"] = torch.from_numpy(color_gripper).float()[None]
        data_batch["T_world_cam"] = torch.from_numpy(T_world_cam).float()[None]
        data_batch["T_world_grippercam"] = torch.from_numpy(T_world_grippercam).float()[
            None
        ]
        data_batch["start_pos_world"] = torch.from_numpy(gripper_state).float()[None]
        data_batch["start_pos"] = torch.from_numpy(gripper_state_cam).float()[None]
        if self.cfg.ALGORITHM.model.am_predict_action_frame == "world":
            data_batch["start_pos"] = data_batch["start_pos_world"]
        else:
            data_batch["start_pos"] = DatasetUtils.transform_two_hands_trajectory(
                data_batch["start_pos_world"],
                torch.linalg.inv(data_batch["T_world_cam"][0]),
                has_finger_tips=False,
            )

        # To device
        for key in data_batch:
            if isinstance(data_batch[key], torch.Tensor):
                data_batch[key] = data_batch[key].to(self.device)

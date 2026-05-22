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
from scipy.spatial.transform import Rotation as SciR
from einops import rearrange


class PolicyVLAWorldModelWrapperStretchRobot(PolicyVLAWorldModelWrapper):
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
        **kwargs,
    ):
        super(PolicyVLAWorldModelWrapperStretchRobot, self).__init__(
            cfg=cfg,
            action_chunk_size=action_chunk_size,
            online_update_visual_state=online_update_visual_state,
            online_update_extrinsics_state=online_update_extrinsics_state,
            online_update_robot_state=online_update_robot_state,
            online_update_environment_state=online_update_environment_state,
            policy_only=policy_only,
            weight_ckpt=weight_ckpt,
            device=device,
        )
        assert (
            self.HEAD_IMAGE_SIZE[0] == self.HEAD_IMAGE_SIZE[1]
        ), "Head image size should be square"
        self.run_on_robot = run_on_robot
        self.action_meta_fpath = action_meta_fpath
        self.state_meta_fpath = state_meta_fpath
        self._load_action_statistics()

    def _load_action_statistics(self):
        if self.action_meta_fpath is not None:
            action_meta = np.load(self.action_meta_fpath)
            self.action_mean = torch.from_numpy(action_meta["action_mean"]).float()
            self.action_std = torch.from_numpy(action_meta["action_std"]).float()
            self.action_norm_max_bound = torch.from_numpy(
                action_meta["action_norm_max_bound"]
            ).float()
            self.action_norm_min_bound = torch.from_numpy(
                action_meta["action_norm_min_bound"]
            ).float()
        else:
            self.action_mean = None
            self.action_std = None
            self.action_norm_max_bound = None
            self.action_norm_min_bound = None

        if self.state_meta_fpath is not None:
            state_meta = np.load(self.state_meta_fpath)
            self.state_mean = torch.from_numpy(state_meta["state_mean"]).float()
            self.state_std = torch.from_numpy(state_meta["state_std"]).float()
            self.state_norm_max_bound = torch.from_numpy(
                state_meta["state_norm_max_bound"]
            ).float()
            self.state_norm_min_bound = torch.from_numpy(
                state_meta["state_norm_min_bound"]
            ).float()
        else:
            self.state_mean = None
            self.state_std = None
            self.state_norm_max_bound = None
            self.state_norm_min_bound = None

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        super().reset()
        self._queues.update(
            {
                # Track the gripper camera state
                OBS_GRIPPER_CAM_STATE: deque(maxlen=self.cfg.ALGORITHM.model.horizon),
                # Track the gripper visual state
                OBS_GRIPPER_VISUAL_STATE: deque(
                    maxlen=self.cfg.ALGORITHM.model.horizon
                ),
            }
        )
        # Gripper visual state
        self.visual_state_feats_gripper = None
        self.visual_observation_feats_gripper = None
        self.visual_shapes_gripper = None
        self.visual_poss_gripper = None
        if not self.online_update_visual_state:
            del self._queues[OBS_GRIPPER_VISUAL_STATE]
        if not self.online_update_extrinsics_state:
            del self._queues[OBS_GRIPPER_CAM_STATE]


    def _extract_visual_features(self, data_batch):

        # Extract the visual features for the head camera
        super()._extract_visual_features(data_batch)

        # Extract the visual features for the gripper camera
        if (
            self.visual_feature_extractor is not None
            and self.online_update_visual_state
        ):
            assert (
                OBS_GRIPPER_VISUAL_STATE in self._queues
            ), f"{OBS_GRIPPER_VISUAL_STATE} is required for streaming ..."
            if self.cfg.ALGORITHM.model.visual_feature_type == "cut3r":
                view_inp_gripper = prepare_cut3r_input(data_batch["color_gripper"])
                (
                    self.visual_state_feats_gripper,
                    self.visual_observation_feats_gripper,
                    self.visual_shapes_gripper,
                    self.visual_poss_gripper,
                ) = self.visual_feature_extractor.model.extract_state_observation_feature_streaming(
                    view_inp_gripper,
                    self.visual_state_feats_gripper,
                    self.visual_observation_feats_gripper,
                    self.visual_shapes_gripper,
                    self.visual_poss_gripper,
                    max_seq_length=1,
                )
                visual_feature_patch_gripper = self.visual_observation_feats_gripper[
                    -1
                ][-1][0, 1:]
            else:
                visual_feature_patch_gripper = (
                    self.visual_feature_extractor.extract_features(
                        data_batch["color_gripper"]
                    )[-1]
                )
                visual_feature_patch_gripper = rearrange(
                    visual_feature_patch_gripper, "b c h w -> b (h w) c"
                )[0]
            data_batch[OBS_GRIPPER_VISUAL_STATE] = visual_feature_patch_gripper

    def _prepare_history_conditioning(self, data_batch):
        super()._prepare_history_conditioning(data_batch)
        intrinsics_gripper = data_batch["intrinsics_gripper"].clone()  # [B, 3, 3]
        T_world_grippercam_curr = data_batch[OBS_GRIPPER_CAM_STATE]

        # prepare the history conditioning for the gripper visual state
        history_gripper_visual_state = self._queues.get(OBS_GRIPPER_VISUAL_STATE, None)
        history_gripper_cam_state = self._queues.get(OBS_GRIPPER_CAM_STATE, None)
        # Get the history tesnors
        if history_gripper_visual_state is not None and self.online_update_visual_state:
            history_gripper_visual_state = torch.stack(
                list(history_gripper_visual_state), dim=0
            )
            history_gripper_visual_state = history_gripper_visual_state[
                None
            ].contiguous()
            # history_gripper_visual_state_gt = data_batch[
            #     "history_visual_feature_patch_gripper"
            # ]
            # diff = (
            #     history_gripper_visual_state - history_gripper_visual_state_gt
            # ).abs()
            # diff_val = (diff**2).mean() ** 0.5
            # print(f"Diff of history_visual_feature_patch_gripper: {diff_val:.10f}")
            data_batch[HISTORY_OBS_GRIPPER_VISUAL_STATE] = history_gripper_visual_state

        # get the history gripper camera state
        if (
            history_gripper_cam_state is not None
            and self.online_update_extrinsics_state
        ):
            history_T_world = torch.stack(
                [T_world_history[0] for T_world_history in history_gripper_cam_state]
            )
            T_grippercam_world = torch.linalg.pinv(T_world_grippercam_curr)[
                0
            ]  # [4, 4], compute once
            history_T_grippercam_history = (
                T_grippercam_world @ history_T_world
            )  # [4, 4] @ [T, 4, 4] -> [T, 4, 4]
            history_raymap_gripper = get_ray_map_in_batch(
                history_T_grippercam_history,
                intrinsics_gripper,
                original_size=self.GRIPPER_IMAGE_SIZE,
                target_size=(
                    self.GRIPPER_IMAGE_SIZE[0] // 16,
                    self.GRIPPER_IMAGE_SIZE[1] // 16,
                ),
            )
            history_raymap_gripper = history_raymap_gripper.permute(
                0, 3, 1, 2
            )  # [T, 6, H, W]
            history_raymap_gripper = history_raymap_gripper[None].contiguous()

            # history_raymap_gripper_gt = data_batch["history_raymap_gripper"]
            # diff = (history_raymap_gripper - history_raymap_gripper_gt).abs()
            # diff_val = (diff**2).mean() ** 0.5
            # print(f"Diff of history_raymap_gripper: {diff_val:.10f}")

            data_batch[HISTORY_OBS_GRIPPER_CAM_STATE] = history_raymap_gripper

    def _postprocess_action(self, outputs, data_batch):
        pred_actions, pred_progress = super()._postprocess_action(outputs, data_batch)
        # pred_actions = data_batch["history_action"][0].cpu()
        # pred_actions = DatasetUtils.transform_two_hands_trajectory(
        #     pred_actions,
        #     data_batch["T_world_cam"][0].cpu(), # [4, 4]
        #     has_finger_tips=False,
        # )
        action_dim = self.cfg.ALGORITHM.model.action_dim
        pred_actions = pred_actions[:, :action_dim]
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

        # # Important: Downsample by 2 to match the action chunk size
        # if self.action_chunk_size <= 1 + pred_actions.shape[0] // 2:
        #     downsampled_actions = pred_actions[::2]
        #     downsampled_progress = pred_progress[::2]

        #     # Only add last element if sequence length is even (odd length already includes it)
        #     if pred_actions.shape[0] % 2 == 0:
        #         downsampled_actions = torch.cat([downsampled_actions, pred_actions[-1:]], dim=0)
        #         downsampled_progress = torch.cat([downsampled_progress, pred_progress[-1:]], dim=0)

        #     pred_actions = downsampled_actions
        #     pred_progress = downsampled_progress
        return pred_actions, pred_progress

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

        # crop the gripper image to get the square image
        height, width = color.shape[:2]
        crop_size = min(height, width)
        crop_x = (width - crop_size) // 2
        crop_y = (height - crop_size) // 2
        color = color[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        depth = depth[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]

        # do resize to the images
        color_gripper = cv2.resize(
            color_gripper,
            (self.GRIPPER_IMAGE_SIZE[1], self.GRIPPER_IMAGE_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        color = cv2.resize(
            color,
            (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        depth = cv2.resize(
            depth,
            (self.HEAD_IMAGE_SIZE[1], self.HEAD_IMAGE_SIZE[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        height_gripper, width_gripper = color_gripper.shape[:2]

        # do resize to the intrinsics
        intrinsics[0, 0] *= self.HEAD_IMAGE_SIZE[1] / crop_size
        intrinsics[1, 1] *= self.HEAD_IMAGE_SIZE[0] / crop_size
        intrinsics[0, 2] = intrinsics[1, 2] = self.HEAD_IMAGE_SIZE[1] / 2

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

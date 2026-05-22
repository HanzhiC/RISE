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
from algos.feature_extractor import DINOv3FeatureExtractor

import torch
import time
from tqdm import tqdm
from collections import deque
import einops
from policies.conventions import *
from einops import rearrange
import torch.nn.functional as F
import copy
import cv2
import numpy as np
import utils.tensor_utils as TensorUtils
import open3d as o3d
import os


class PolicyVLAWorldModelWrapper:
    HEAD_IMAGE_SIZE = (224, 224)

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
        **kwargs,
    ):
        # Init config
        self.cfg = cfg
        self.device = device
        # Init model
        if weight_ckpt is None:
            weight_ckpt = os.path.join(cfg.log_dir, cfg.name, "last.ckpt")
        print(f"===================================================")
        print(f"====> Loading VLA-WM model from {weight_ckpt} <=====")
        print(f"===================================================")
        self.model = VLAWorldModelModule.load_from_checkpoint(
            weight_ckpt,
            algo_config=cfg.ALGORITHM,
            train_config=cfg.TRAIN,
        )
        self.model.eval()
        self.model.to(self.device)
        self.action_chunk_size = action_chunk_size
        self.online_update_visual_state = online_update_visual_state
        self.online_update_extrinsics_state = online_update_extrinsics_state
        self.online_update_robot_state = online_update_robot_state
        self.online_update_environment_state = online_update_environment_state
        self.policy_only = policy_only
        if self.cfg.ALGORITHM.model.visual_feature_type == "cut3r":
            self.visual_feature_extractor = Cut3RGeometryPredictor(
                model_path="./weights/cut3r_224_linear_4.pth",
                size=224,
                device=self.device,
                square_ok=False,
                verbose=False,
            )
        elif self.cfg.ALGORITHM.model.visual_feature_type == "dinov3":
            self.visual_feature_extractor = DINOv3FeatureExtractor(
                model_name="dinov3_vitb16",
                device=self.device,
            )

        # Init queues
        self._queues = None
        self.reset()
        # Print the keys in the queues
        print(f"===================================================")
        print(f"====> Keys in the queues: {list(self._queues.keys())} <=====")
        print(f"===================================================")

    def set_guidance(self, guidance_config):
        self.model.set_guidance(guidance_config)

    def clear_guidance(self):
        self.model.clear_guidance()

    def get_flow_matching_policy(self):
        """Same ``VLAFlowMatching`` used in ``VLAWorldModelModule.forward`` (EMA when enabled)."""
        m = self.model
        if getattr(m, "use_ema", False) and m.use_ema:
            ep = getattr(m, "ema_policy", None)
            if ep is not None:
                return ep
        return m.nets["policy"]

    def compress_visual_tokens_for_wm_vm_infer(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply WM/VM DINO token compression when ``wm_vm_visual_compress_dim`` < 768."""
        return self.get_flow_matching_policy().compress_visual_tokens_for_wm_vm(tokens)

    def reset(self):
        print(f"===> Resetting the policy wrapper ... {self.action_chunk_size=}")
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            # Predict the future
            ACTION: deque(maxlen=self.action_chunk_size),
            # Track the history
            OBS_ROBOT_STATE: deque(maxlen=self.cfg.ALGORITHM.model.horizon),
            OBS_VISUAL_STATE: deque(maxlen=self.cfg.ALGORITHM.model.horizon),
            OBS_ENV_STATE: deque(maxlen=self.cfg.ALGORITHM.model.horizon),
            OBS_HEAD_CAM_STATE: deque(
                maxlen=self.cfg.ALGORITHM.model.horizon
            ),  # T_world_cam
        }
        self.visual_state_feats = None
        self.visual_observation_feats = None
        self.visual_shapes = None
        self.visual_poss = None
        self.predicted_action_latest = None
        self.predicted_state_latest = None
        self.predicted_state_meta_latest = None
        self.language_feature_global = None
        self.instruction_global = ""

        if not self.online_update_visual_state:
            del self._queues[OBS_VISUAL_STATE]
        if not self.online_update_environment_state:
            del self._queues[OBS_ENV_STATE]
        if not self.online_update_extrinsics_state:
            del self._queues[OBS_HEAD_CAM_STATE]
        if not self.online_update_robot_state:
            del self._queues[OBS_ROBOT_STATE]
        if self.policy_only and OBS_ENV_STATE in self._queues:
            del self._queues[OBS_ENV_STATE]

        self.clear_guidance()

    def _preprocess_observation(self, data_batch):
        # do the dict remapping here ...
        if "language_instruction" in data_batch:
            if data_batch["language_instruction"] != self.instruction_global:
                print(
                    f"===> Extracting language features... changing from '{self.instruction_global}' to '{data_batch['language_instruction']}'"
                )
                self._extract_language_features(data_batch)
            else:
                data_batch["language_feature"] = self.language_feature_global
        else:
            assert "language_feature" in data_batch, "language_feature is required"

    def _extract_language_features(self, data_batch):
        self.instruction_global = data_batch["language_instruction"]
        self.language_feature_global = self.model.extract_language_features(
            data_batch["language_instruction"]
        )
        data_batch["language_feature"] = self.language_feature_global

    def _extract_visual_features(self, data_batch):
        if (
            self.visual_feature_extractor is not None
            and self.online_update_visual_state
        ):
            assert (
                OBS_VISUAL_STATE in self._queues
            ), f"{OBS_VISUAL_STATE} is required for streaming ..."
            if self.cfg.ALGORITHM.model.visual_feature_type == "cut3r":
                view_inp = prepare_cut3r_input(data_batch["color"])
                (
                    self.visual_state_feats,
                    self.visual_observation_feats,
                    self.visual_shapes,
                    self.visual_poss,
                ) = self.visual_feature_extractor.model.extract_state_observation_feature_streaming(
                    view_inp,
                    self.visual_state_feats,
                    self.visual_observation_feats,
                    self.visual_shapes,
                    self.visual_poss,
                    max_seq_length=1,
                )
                visual_feature_patch = self.visual_observation_feats[-1][-1][0, 1:]
            else:
                assert (
                    data_batch["color"].ndim == 4
                ), f"Color must be 4D tensor, but got {data_batch['color'].ndim}"
                visual_feature_patch = self.visual_feature_extractor.extract_features(
                    data_batch["color"]
                )[-1]
                visual_feature_patch = rearrange(
                    visual_feature_patch, "b c h w -> b (h w) c"
                )[0]
            data_batch[OBS_VISUAL_STATE] = visual_feature_patch

    def _extract_dinov3_features(self, data_batch):
        return self.model.extract_dinov3_features(data_batch)

    def _extract_query_points(self, data_batch):
        if self.online_update_environment_state:
            grid_size = self.cfg.ALGORITHM.model.dynamics_input_size
            depths = data_batch["depth"].clone()  # [B, H, W]\
            depths[depths == 0] = 1e-3
            intrs = data_batch["intrinsics"]  # [B, 3, 3]
            extrs = torch.eye(4, device=depths.device).repeat(
                depths.shape[0], 1, 1
            )  # [B, 4, 4]
            query_points = AriaUtils.get_grid_queries(
                grid_size=grid_size,
                depths=depths,
                intrinsics=intrs,
                extrinsics=extrs,
            )
            query_points = query_points[..., 1:]
            query_points = query_points.view(grid_size, grid_size, 3)
            query_points = query_points.permute(2, 0, 1)[None].contiguous()
            # # check the difference
            # original_query_points = data_batch[OBS_ENV_STATE]
            # # Check the difference
            # diff = query_points - original_query_points
            # diff = (diff**2).mean() ** 0.5
            data_batch[OBS_ENV_STATE] = query_points

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
            history_T_world = torch.stack(
                [T_world_history[0] for T_world_history in history_head_cam_state]
            )  # [T, 4, 4]

            history_T_cam_history = (
                T_cam_world @ history_T_world
            )  # [4, 4] @ [T, 4, 4] -> [T, 4, 4]
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
            ]  # [T, D], in the world frame

            # Change the history robot state to the camera frame if
            # the action is predicted in the camera frame
            if self.cfg.ALGORITHM.model.am_predict_action_frame == "camera":
                history_robot_abs_state = DatasetUtils.transform_two_hands_trajectory(
                    history_robot_abs_state,
                    T_cam_world,
                    has_finger_tips=has_finger_tips,
                )  # from world frame to camera frame

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
        load_finger_tips = self.cfg.ALGORITHM.model.action_dim == 48
        pred_actions = outputs["action_predictions"][0, 0]  # [H, ACTION_DIM]
        pred_actions_np = pred_actions.cpu().numpy()
        T_world_cam = data_batch[OBS_HEAD_CAM_STATE][0].cpu().numpy()  # [4, 4]
        if self.cfg.ALGORITHM.model.am_predict_action_frame == "world":
            pred_actions_transformed = pred_actions_np
        else:
            pred_actions_transformed = DatasetUtils.transform_two_hands_trajectory(
                pred_actions_np,
                T_world_cam,
                self.cfg.ALGORITHM.model.action_dim,
                has_finger_tips=load_finger_tips,
            )
        if self.cfg.ALGORITHM.model.am_predict_progress:
            pred_progress = outputs["progress_predictions"][0, 0]  # [H]
            pred_progress_np = pred_progress.cpu().numpy()
        else:
            pred_progress_np = -1 * torch.ones(pred_actions.shape[0])  # [H]
        pred_progress = torch.tensor(pred_progress_np)
        pred_actions_transformed = torch.from_numpy(
            pred_actions_transformed
        )  # [H, ACTION_DIM]
        return pred_actions_transformed, pred_progress

    def _postprocess_state(self, outputs, data_batch):
        pred_dynamics = outputs["dynamics_predictions"][0, 0]  # [Dg, Hg, Wg]
        Dg, Hg, Wg = pred_dynamics.shape
        pred_dynamics = pred_dynamics.view(Dg // 3, 3, Hg, Wg)  # [H, 3, 64, 64]
        pred_dynamics_np = pred_dynamics.cpu().numpy()
        T_world_cam = data_batch[OBS_HEAD_CAM_STATE][0].cpu().numpy()  # [4, 4]

        # Transform the dynamics to the world frame
        pred_dynamics_np = einops.rearrange(pred_dynamics_np, "t c h w -> (t h w) c")
        pred_dynamics_np = DatasetUtils.transform_points(pred_dynamics_np, T_world_cam)
        pred_dynamics_np = einops.rearrange(
            pred_dynamics_np, "(t h w) c -> t c h w", h=Hg, w=Wg
        )

        if self.cfg.ALGORITHM.model.wm_predict_distance_to_goal:
            pred_distance_to_goal = outputs["dynamics_distance_to_goal_predictions"][
                0, 0
            ]
            pred_distance_to_goal = pred_distance_to_goal.view(-1, 3, Hg, Wg)[-1]
            pred_distance_to_goal_np = pred_distance_to_goal.cpu().numpy()
            pred_distance_to_goal_np = (pred_distance_to_goal_np**2).mean() ** 0.5
        else:
            pred_distance_to_goal_np = -1.0
        pred_distance_to_goal = torch.tensor(pred_distance_to_goal_np)[None]
        pred_dynamics = torch.from_numpy(pred_dynamics_np)
        return pred_dynamics, pred_distance_to_goal

    @torch.no_grad()
    def inference(
        self,
        data_batch,
        action_only=False,
        value_only=False,
        enable_guidance=False,
        w_advantage=1.0,
        w_conditional=1.0,
        align_to_current_state=False,
    ):
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # Preprocess the observation
        self._preprocess_observation(data_batch)
        self._extract_visual_features(data_batch)
        self._extract_query_points(data_batch)

        if not action_only:
            assert (
                not self.policy_only
            ), "Policy only mode does not support world dynamics prediction ..."
            self._extract_dinov3_features(data_batch)

        # Initialize the queues
        self._queues = populate_queues(self._queues, data_batch, exclude_keys=[ACTION])

        # Prepare the history conditioning
        self._prepare_history_conditioning(data_batch)

        # Do inference
        if len(self._queues[ACTION]) == 0:
            outputs = self.model(
                data_batch,
                action_only=action_only,
                value_only=value_only,
                enable_guidance=enable_guidance,
                w_advantage=w_advantage,
                w_conditional=w_conditional,
                align_to_current_state=align_to_current_state,
            )

            pred_actions, pred_progress = self._postprocess_action(outputs, data_batch)
            pred_actions = torch.cat(
                [pred_actions, pred_progress[..., None]], dim=-1
            )  # [H, ACTION_DIM + 1]
            # self.predicted_action_latest = pred_actions
            self.predicted_action_latest = pred_actions
            pred_actions = pred_actions[
                : min(len(pred_actions), self.action_chunk_size)
            ]

            self._queues[ACTION].extend(pred_actions)  # [H, ACTION_DIM]
            if not action_only:
                self.predicted_state_latest, self.predicted_state_meta_latest = (
                    self._postprocess_state(outputs, data_batch)
                )

        # Get the next action
        latest_action_chunk = self.get_current_action_chunk(data_batch)
        # latest_action_chunk = self.predicted_action_latest
        action = self._queues[ACTION].popleft()

        # Acquire the output dictionary
        predictions = {
            "selected_action": action,  # [ACTION_DIM]
            "latest_action_chunk": latest_action_chunk,  # [H, ACTION_DIM]
            "latest_predicted_action": self.predicted_action_latest,  # [H, ACTION_DIM]
            "latest_dynamics": self.predicted_state_latest,  # [Hg, 3, 64, 64]
            "latest_dynamics_meta": self.predicted_state_meta_latest,  # [1]
        }
        return predictions

    def get_current_action_chunk(self, data_batch):
        has_finger_tips = self.cfg.ALGORITHM.model.action_dim == 48
        action_dim = self.cfg.ALGORITHM.model.action_dim

        rest_of_actions = self._queues[ACTION]
        if len(rest_of_actions) == 0:
            # print(f"!!! No actions in the queue ... returning None")
            return None

        rest_of_actions = torch.stack(list(rest_of_actions), dim=0)  # [H, ACTION_DIM]
        return rest_of_actions

    @torch.no_grad()
    def inference_value(self, data_batch):
        # value_only path: flat batch B*N here; model uses default num_samples=1, caller does .view(-1, N).
        action_suffix = (
            "_rel" if self.cfg.ALGORITHM.model.am_use_relative_action else ""
        )
        input_actions = data_batch[
            f"gt_action{action_suffix}"
        ]  # Just to get the first tracking target as gripper state
        history_state = data_batch[f"history_state"]
        outputs = self.model(
            data_batch,
            action_only=False,
            value_only=True,
            input_actions=input_actions,
            input_dynamics=history_state,
        )
        return outputs["value_predictions"].squeeze(-1)

    @torch.no_grad()
    def inference_advantage(self, data_batch_curr, data_batch_future):
        value_curr = self.inference_value(data_batch_curr)
        value_future = self.inference_value(data_batch_future)
        advantage = value_future - value_curr
        results = {
            "advantage_predictions": advantage,
            "value_predictions": value_curr,
            "future_value_predictions": value_future,
        }
        return results

    def inference_action_with_reference_guidance(
        self,
        data_batch,
        data_batch_reference,
        num_samples=1,
        w_conditional=1.0,
        enable_guidance=True,
        eval_guidance=True,
        align_to_current_state=False,
        add_history_state_actions_null=False,
    ):
        assert (
            self.cfg.ALGORITHM.model.wm_predict_visual
        ), "Visual prediction is not enabled ..."
        assert (
            self.cfg.ALGORITHM.model.history_visual_horizon == 1
        ), "History visual horizon must be 1 ..."
        assert (
            self.cfg.ALGORITHM.model.history_sample_mode == "latest"
        ), "History visual sample mode must be latest ..."

        # For composition ...
        data_batch["history_visual_feature_patch_null"] = (
            data_batch_reference["visual_feature_patch"]
            .clone()[:, None]
            .expand(-1, 15, -1, -1)
        )
        data_batch["history_visual_feature_patch_gripper_null"] = (
            data_batch_reference["visual_feature_patch_gripper"]
            .clone()[:, None]
            .expand(-1, 15, -1, -1)
        )
        if add_history_state_actions_null:
            data_batch["history_state_actions_null"] = (
                data_batch_reference["start_pos"].clone()[:, None].expand(-1, 15, -1)
            )  # [B, 15, D]

        data_batch["goal_visual_feature_patch"] = data_batch_reference[
            "goal_visual_feature_patch"
        ].clone()

        data_batch["goal_state_residual"] = data_batch_reference[
            "gt_state_residual"
        ].clone()
        # data_batch["state_valid"] = data_batch_reference["state_valid"].clone()
        # data_batch["start_pos"] = data_batch_reference["start_pos"].clone()
        # data_batch["start_state"] = data_batch_reference["start_state"].clone()

        # Forward the model
        if "start_state_dinov3_feature" not in data_batch:
            self.model.extract_dinov3_features(data_batch)

        # Infer actions and values
        outputs = self.model(
            data_batch,
            num_samples=num_samples,
            action_only=False,
            w_conditional=w_conditional,
            enable_guidance=enable_guidance,
            eval_guidance=eval_guidance,
            align_to_current_state=align_to_current_state,
        )

        # Acquire the final action
        pred_actions = outputs["action_predictions"]  # [B, N, H, D]
        pred_dynamics = outputs["dynamics_predictions"]  # [B, N, C, H, W]
        pred_visual = outputs["visual_predictions"]  # [B, N, C, H, W]

        ## Hack for debugging ...
        # pred_visual = data_batch["goal_visual_feature_patch"][:, None].expand(
        #     -1, num_samples, -1, -1
        # )  # [B, N, L, C]
        # pred_visual = rearrange(pred_visual, "b n (h w) c -> (b n) c h w", h=14, w=14)
        # pred_visual = F.interpolate(
        #     pred_visual, size=(64, 64), mode="bilinear", align_corners=False
        # )
        # pred_visual = rearrange(pred_visual, "(b n) c h w -> b n c h w", n=num_samples)
        # pred_visual = pred_visual * 5.0
        # pred_actions = data_batch["gt_action"][:, None].expand(-1, num_samples, -1, -1)
        ## Hack for debugging ...

        # Infer the future state's value
        data_batch_future = copy.deepcopy(data_batch)
        data_batch_current = copy.deepcopy(data_batch)
        data_batch_future = TensorUtils.repeat_by_expand_at(
            data_batch_future, repeats=num_samples, dim=0
        )
        data_batch_current = TensorUtils.repeat_by_expand_at(
            data_batch_current, repeats=num_samples, dim=0
        )
        data_batch_future["history_visual_feature_patch"] = (
            self.build_history_visual_feature_patch_conditioning(pred_visual)
        ).flatten(
            0, 1
        )  # [B * N, 15, 196, C_vis] — C_vis is 768 or wm_vm_visual_in_dim (WM pred)
        data_batch_future["gt_action"] = (
            self.build_action_conditioning(
                pred_actions,
                action_chunk_size=self.cfg.ALGORITHM.model.action_chunk_size,
            )
        ).flatten(
            0, 1
        )  # [B * N, H, D] — aligns with repeat_by_expand_at row order
        data_batch_future["history_state"] = pred_dynamics.flatten(0, 1)

        value_info = self.inference_advantage(data_batch_current, data_batch_future)
        value_info["advantage_predictions"] = value_info["advantage_predictions"].view(
            -1, num_samples
        )
        value_info["value_predictions"] = value_info["value_predictions"].view(
            -1, num_samples
        )
        value_info["future_value_predictions"] = value_info[
            "future_value_predictions"
        ].view(-1, num_samples)

        # Acquire all results
        results = {
            "action_predictions": pred_actions,
            "dynamics_predictions": pred_dynamics,
            "visual_predictions": pred_visual,
            "value_info": value_info,
        }
        if eval_guidance:
            results["guidance_cost"] = outputs["guidance_losses"][
                "dynamics_regression_guidance_loss"
            ]
        return results

    def select_best_sample(self, results, criteria="value"):
        value_info_selected = copy.deepcopy(results["value_info"])
        pred_actions = results["action_predictions"]
        pred_dynamics = results["dynamics_predictions"]
        pred_visual = results["visual_predictions"]
        if criteria == "value":
            selected_action_idx = value_info_selected[
                "future_value_predictions"
            ].argmax(
                dim=1
            )  # [B]
        elif criteria == "cost":
            guidance_cost = results["guidance_cost"]
            selected_action_idx = guidance_cost.argmin(dim=1)  # [B]
        else:
            raise ValueError(f"Invalid criteria: {criteria}")

        batch_idx = torch.arange(
            value_info_selected["advantage_predictions"].shape[0],
            device=selected_action_idx.device,
        )
        # Per batch element b, keep sample n* = argmax_n advantage[b, n].
        pred_actions_selected = pred_actions[
            batch_idx, selected_action_idx
        ]  # [B, H, D]
        pred_dynamics_selected = pred_dynamics[batch_idx, selected_action_idx]
        pred_visual_selected = pred_visual[batch_idx, selected_action_idx]
        value_info_selected["value_predictions"] = value_info_selected[
            "value_predictions"
        ][batch_idx, selected_action_idx]
        value_info_selected["future_value_predictions"] = value_info_selected[
            "future_value_predictions"
        ][batch_idx, selected_action_idx]
        value_info_selected["advantage_predictions"] = value_info_selected[
            "advantage_predictions"
        ][batch_idx, selected_action_idx]

        results.update(
            {
                "selected_action_prediction": pred_actions_selected,
                "selected_dynamics_prediction": pred_dynamics_selected,
                "selected_visual_prediction": pred_visual_selected,
                "selected_value_info": value_info_selected,
            }
        )
        if criteria == "cost":
            guidance_cost_selected = guidance_cost[batch_idx, selected_action_idx]
            results["selected_guidance_cost"] = guidance_cost_selected
        return results

    def visualize_visual_predictions(
        self, data_batch, outputs, ranking_criteria="value", viser_server=None
    ):
        assert data_batch["color"].shape[0] == 1, "Only support batch size 1 ..."
        B, N, C, H, W = outputs["visual_predictions"].shape
        if C != 768:
            print(
                f"[WARN] visualize_visual_predictions: WM visual has C={C} (not DINO 768); "
                "skipping DINO feature visualization."
            )
            return []
        scale_factor = 4
        visual_predictions = outputs["visual_predictions"]  # [B, N, D, H, W]
        visual_predictions = rearrange(visual_predictions, "b n c h w -> (b n) c h w")
        visual_predictions_vis = F.interpolate(
            visual_predictions,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )  # [B * N, C, 64, 64]
        visual_predictions_vis = self.visual_feature_extractor.visualize_feature(
            visual_predictions_vis
        )  # [B * N, C, H, W]
        visual_predictions_vis = visual_predictions_vis.view(
            B, N, 3, H * scale_factor, W * scale_factor
        )  # [B, N, C, H, W]
        visual_predictions_vis = rearrange(
            visual_predictions_vis, "b n c h w -> b n h w c"
        )
        visual_predictions_vis = (visual_predictions_vis.cpu().numpy() * 255.0).astype(
            np.uint8
        )  # [B, N, H, W, C] #
        visual_predictions_vis = visual_predictions_vis[0][
            ..., [2, 1, 0]
        ].copy()  # [N, H, W, 3]

        guided_reward_pred = (
            -outputs["guidance_cost"][0].cpu().numpy()
            if "guidance_cost" in outputs
            else None
        )
        value_future_pred = (
            outputs["value_info"]["future_value_predictions"][0].cpu().numpy()
        )
        ranking_scores = (
            value_future_pred if ranking_criteria == "value" else guided_reward_pred
        )

        vis = []
        for i in range(N):
            visual_predictions_vis_i = visual_predictions_vis[i]  # [H, W, 3]
            cv2.putText(
                visual_predictions_vis_i,
                f"{ranking_criteria}: {ranking_scores[i]:.5f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )
            vis.append(visual_predictions_vis_i)
        if viser_server is None:
            cv2.imshow("visual_predictions_vis", np.concatenate(vis, axis=1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            os.makedirs(".tmp", exist_ok=True)
            cv2.imwrite(".tmp/visual_predictions_vis.png", np.concatenate(vis, axis=1))
            print(f"Saved visual predictions to .tmp/visual_predictions_vis.png")
        return vis

    def visualize_dynamics_predictions(
        self, data_batch, outputs, ranking_criteria="value", viser_server=None
    ):
        """Visualize the dynamics prediction in the 3D scene.

        Parameters
        ----------
        data_batch : dict
            The data batch containing the color, depth, intrinsics, and T_world_cam.
        outputs : dict
            The outputs containing the dynamics prediction.
        """
        assert data_batch["color"].shape[0] == 1, "Only support batch size 1 ..."
        color = data_batch["color"][0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        depth = data_batch["depth"][0].cpu().numpy()
        intr = data_batch["intrinsics"][0].cpu().numpy()
        T_world_cam = data_batch["T_world_cam"][0].cpu().numpy()
        T_cam0_cam = data_batch["T_cam0_cam"][0].cpu().numpy()
        T_word_cam0 = T_world_cam @ np.linalg.inv(T_cam0_cam)  # [4, 4]
        points, scene_ids = DatasetUtils.backproject(depth, intr, depth < 1.5, False)
        points_rgb = color[scene_ids[0], scene_ids[1]]
        points_world = DatasetUtils.transform_points(points, T_world_cam)

        pred_dynamics = (
            outputs["dynamics_predictions"][0].cpu().numpy()
        )  # [N, H*3, 64, 64]
        pred_dynamics = pred_dynamics.reshape(
            pred_dynamics.shape[0],
            -1,
            3,
            pred_dynamics.shape[2],
            pred_dynamics.shape[3],
        )
        init_dynamics = data_batch["start_state"][0].cpu().numpy()  # [3, 64, 64]
        valid_dynamics_mask = init_dynamics[2] > 0.3  # [64, 64]

        # Acquire the cost and value information
        guided_reward_pred = (
            -outputs["guidance_cost"][0].cpu().numpy()
            if "guidance_cost" in outputs
            else None
        )
        value_future_pred = (
            outputs["value_info"]["future_value_predictions"][0].cpu().numpy()
        )
        ranking_scores = (
            value_future_pred if ranking_criteria == "value" else guided_reward_pred
        )
        dynamics_colors = DatasetUtils.get_heatmap(
            ranking_scores, cmap_name="rainbow_r"
        )
        vis = viser_server if viser_server is not None else []

        if viser_server is None:  # visualize in the 3D scene
            pcd_scene = DatasetUtils.visualize_points(points_world, colors=points_rgb)
            vis.append(pcd_scene)
            for i in range(pred_dynamics.shape[0]):
                pred_dynamics_i = (
                    pred_dynamics[i, -1].reshape(3, -1).T
                )  # [N, 3], last time step
                pred_dynamics_i = DatasetUtils.transform_points(
                    pred_dynamics_i, T_word_cam0
                )
                valid_dynamics_mask_i = valid_dynamics_mask.reshape(-1)
                pred_dynamics_i = pred_dynamics_i[valid_dynamics_mask_i]
                pred_dynamics_color_i = dynamics_colors[i][None].repeat(
                    len(pred_dynamics_i), axis=0
                )
                vis_pred_dynamics = DatasetUtils.visualize_points(
                    pred_dynamics_i,
                    colors=pred_dynamics_color_i,
                    as_spheres=True,
                    size=0.001,
                )
                vis.append(vis_pred_dynamics)
            o3d.visualization.draw(vis)

        else:  # visualize in the viser server
            vis.scene.add_point_cloud(
                name="/rl_debug/scene",
                points=points_world.astype(np.float32),
                colors=points_rgb.astype(np.float32),
                point_size=0.003,
                point_shape="circle",
            )
            for i in range(pred_dynamics.shape[0]):
                pred_dynamics_i = (
                    pred_dynamics[i, -1].reshape(3, -1).T
                )  # [N, 3], last time step
                valid_dynamics_mask_i = valid_dynamics_mask.reshape(-1)
                pred_dynamics_i = pred_dynamics_i[valid_dynamics_mask_i]
                pred_dynamics_color_i = dynamics_colors[i][None].repeat(
                    len(pred_dynamics_i), axis=0
                )
                vis.scene.add_point_cloud(
                    name=f"/rl_debug/pred_dynamics_{i}",
                    points=pred_dynamics_i.astype(np.float32),
                    colors=pred_dynamics_color_i.astype(np.float32),
                    point_size=0.012,
                    point_shape="circle",
                )
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nEmpty the viser server...")
                vis.scene.remove_by_name("/rl_debug")
        return vis

    def visualize_action_predictions(
        self,
        data_batch,
        outputs,
        ranking_criteria="value",
        viser_server=None,
        view="head",
    ):
        """Visualize the action prediction in the 3D scene.

        Parameters
        ----------
        data_batch : dict
            The data batch containing the color, depth, intrinsics, and T_world_cam.
        outputs : dict
            The outputs containing the action prediction.
        """
        assert data_batch["color"].shape[0] == 1, "Only support batch size 1 ..."

        if view == "head":
            color = data_batch["color"][0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            depth = data_batch["depth"][0].cpu().numpy()
            intr = data_batch["intrinsics"][0].cpu().numpy()
            T_world_cam = data_batch["T_world_cam"][0].cpu().numpy()
        elif view == "gripper":
            color = (
                data_batch["color_gripper"][0].cpu().numpy().transpose(1, 2, 0)
            )  # [H, W, 3]
            depth = data_batch["depth_gripper"][0].cpu().numpy()
            intr = data_batch["intrinsics_gripper"][0].cpu().numpy()
            T_world_cam = data_batch["T_world_grippercam"][0].cpu().numpy()
        else:
            raise ValueError(f"Invalid view: {view}")

        points, scene_ids = DatasetUtils.backproject(depth, intr, depth < 1.5, False)
        points_rgb = color[scene_ids[0], scene_ids[1]]
        points_world = DatasetUtils.transform_points(points, T_world_cam)
        gt_action = data_batch["gt_action"][0].cpu().numpy()
        gt_right_action = gt_action[:, self.cfg.ALGORITHM.model.action_dim // 2 :]
        gt_right_root_action = DatasetUtils.get_root_transformation(gt_right_action)

        pred_actions = outputs["action_predictions"][0].cpu().numpy()
        pred_actions = pred_actions[..., self.cfg.ALGORITHM.model.action_dim // 2 :]

        # Acquire the cost and value information
        guided_reward_pred = (
            -outputs["guidance_cost"][0].cpu().numpy()
            if "guidance_cost" in outputs
            else None
        )
        value_future_pred = (
            outputs["value_info"]["future_value_predictions"][0].cpu().numpy()
        )
        ranking_scores = (
            value_future_pred if ranking_criteria == "value" else guided_reward_pred
        )
        traj_colors = DatasetUtils.get_heatmap(ranking_scores, cmap_name="rainbow_r")
        vis = viser_server if viser_server is not None else []
        if viser_server is None:  # visualize in the 3D scene
            pcd_scene = DatasetUtils.visualize_points(points_world, colors=points_rgb)
            vis_action = DatasetUtils.visualize_6d_trajectory(
                gt_right_root_action,
                size=0.005,
                cmap_name="Greens_r",
                to_mesh=True,
            )
            vis.append(pcd_scene)
            vis.append(vis_action)
            for i in range(pred_actions.shape[0]):
                pred_right_root_action = DatasetUtils.get_root_transformation(
                    pred_actions[i]
                )
                if len(pred_actions) > 1:
                    traj_color = traj_colors[i]
                else:
                    traj_color = None
                vis_pred_action = DatasetUtils.visualize_6d_trajectory(
                    pred_right_root_action,
                    size=0.005,
                    cmap_name="turbo",
                    color=traj_color,
                    to_mesh=True,
                )
                vis.append(vis_pred_action)

            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                o3d.visualization.draw(vis)
            else:
                print(
                    "[WARN] visualize_action_predictions: no display detected, "
                    "skipping o3d.visualization.draw_geometries to avoid segfault. "
                    "Pass --use_viser for headless visualization."
                )

        else:  # visualize in the viser server
            vis.scene.add_point_cloud(
                name="/rl_debug/scene",
                points=points_world.astype(np.float32),
                colors=points_rgb.astype(np.float32),
                point_size=0.003,
                point_shape="circle",
            )
            vis.scene.add_point_cloud(
                name="/rl_debug/gt_traj",
                points=gt_right_root_action[:, :3, 3].astype(np.float32),
                colors=DatasetUtils.get_heatmap(
                    np.arange(len(gt_right_root_action[:, :3, 3])), cmap_name="Greens_r"
                ),
                point_size=0.012,
                point_shape="circle",
            )
            for i in range(pred_actions.shape[0]):
                pred_right_root_action = DatasetUtils.get_root_transformation(
                    pred_actions[i]
                )
                if len(pred_actions) > 1:
                    traj_color = traj_colors[i][None].repeat(
                        len(pred_right_root_action[:, :3, 3]), axis=0
                    )
                else:
                    traj_color = DatasetUtils.get_heatmap(
                        np.arange(len(pred_right_root_action[:, :3, 3])),
                        cmap_name="turbo",
                    )
                vis.scene.add_point_cloud(
                    name=f"/rl_debug/pred_traj_{i}",
                    points=pred_right_root_action[:, :3, 3].astype(np.float32),
                    # colors=DatasetUtils.get_heatmap(np.arange(len(pred_right_root_action[:, :3, 3])), cmap_name="turbo"),
                    colors=traj_color,
                    point_size=0.012,
                    point_shape="circle",
                )

            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nEmpty the viser server...")
                vis.scene.remove_by_name("/rl_debug")
        return vis

    def build_history_visual_feature_patch_conditioning(
        self, pred_visual, target_size=(14, 14), history_length=15
    ):
        """Resample WM-predicted spatial maps to patch tokens for VM ``history_visual_feature_patch``.

        ``pred_visual`` is ``[B, N, C, H, W]`` where ``C`` is DINO 768 or ``wm_vm_visual_compress_dim``
        when WM predicts compressed dynamics visual.
        """
        assert (
            self.cfg.ALGORITHM.model.wm_predict_visual
        ), "Visual prediction is not enabled ..."
        assert (
            self.cfg.ALGORITHM.model.history_visual_horizon == 1
        ), "History visual horizon must be 1 ..."
        assert (
            self.cfg.ALGORITHM.model.history_sample_mode == "latest"
        ), "History visual sample mode must be latest ..."
        B, N, C, H, W = pred_visual.shape
        pred_visual = pred_visual.view(B * N, C, H, W)
        pred_visual = F.interpolate(
            pred_visual,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        pred_visual = rearrange(pred_visual, "(b n) c h w -> b n (h w) c", n=N)
        pred_visual = pred_visual[:, :, None].expand(-1, -1, history_length, -1, -1)
        return pred_visual

    def build_action_conditioning(self, pred_action, action_chunk_size=30):
        assert (
            self.cfg.ALGORITHM.model.history_action_horizon == 1
        ), "History action horizon must be 1 ..."
        assert (
            self.cfg.ALGORITHM.model.history_sample_mode == "latest"
        ), "History action sample mode must be latest ..."
        B, N, H, D = pred_action.shape  # [B, N, H, D]
        pred_action = pred_action.view(B * N, H, D)
        pred_action_last = pred_action[:, -1, :][:, None, :]  # [B*N, 1, D]
        pred_action_last_repeat = pred_action_last.repeat(
            1, action_chunk_size, 1
        )  # [B*N, action_chunk_size, D]
        pred_action_last_repeat = rearrange(
            pred_action_last_repeat, "(b n) h d -> b n h d", b=B, n=N
        )
        return pred_action_last_repeat

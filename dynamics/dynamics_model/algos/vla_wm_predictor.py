import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import utils.dataset_utils as DatasetUtils
import utils.tensor_utils as TensorUtils
import utils.aria_utils as AriaUtils
from models.vla.flow_matching_model import VLAFlowMatching
import open3d as o3d
import time
import cv2
import torchvision
from einops import rearrange
from algos.feature_extractor import DINOv3FeatureExtractor
from models.multimodal.t5_encoder import T5Embedder
import utils.dataset_utils as DatasetUtils
from algos.helpers import EMA

EXTRACTOR = DINOv3FeatureExtractor(
    model_name="dinov3_vitb16",
    device="cuda",
)

T5_EMBEDDER = T5Embedder(
    from_pretrained="google-t5/t5-base",
    model_max_length=1024,
    use_offload_folder=None,
    device="cuda",
)


class VLAWorldModelModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(VLAWorldModelModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()
        self.ema_policy = None

        # Conditioning parsing
        self.cond_drop_language_p = (
            algo_config.training.conditioning_drop_language
            if "conditioning_drop_language" in algo_config.training
            else 0.0
        )
        self.cond_drop_state_p = (
            algo_config.training.conditioning_drop_state
            if "conditioning_drop_state" in algo_config.training
            else 0.0
        )
        self.cond_drop_action_p = (
            algo_config.training.conditioning_drop_action
            if "conditioning_drop_action" in algo_config.training
            else 0.0
        )
        self.cond_drop_advantage_p = (
            algo_config.training.conditioning_drop_advantage
            if "conditioning_drop_advantage" in algo_config.training
            else 0.0
        )
        self.cond_drop_visual_p = (
            algo_config.training.conditioning_drop_visual
            if "conditioning_drop_visual" in algo_config.training
            else 0.0
        )
        self.dinov3_upsampling_factor = (
            algo_config.model.dinov3_upsampling_factor
            if "dinov3_upsampling_factor" in algo_config.model
            else 1
        )
        self.value_start_td_loss_at_step = (
            algo_config.training.value_start_td_loss_at_step
            if "value_start_td_loss_at_step" in algo_config.training
            else 0
        )
        self.value_end_loss_at_step = (
            algo_config.training.value_end_loss_at_step
            if "value_end_loss_at_step" in algo_config.training
            else float("inf")
        )
        self._value_model_frozen = False
        self.use_ema = (
            algo_config.training.use_ema if "use_ema" in algo_config.training else False
        )

        # Initialize the diffuser
        self.nets["policy"] = VLAFlowMatching(algo_config.model)
        self.langugae_tokens_null = self.extract_language_features("")  # [1, 30, 768]
        self.langugae_tokens_null = self.langugae_tokens_null.to(self.device)
        self.curr_train_step = 0  # step within an epoch
        self.visualize_batch_idx = (
            None  # Will be set to a random batch index for visualization
        )
        self.action_dim = algo_config.model.action_dim
        self.dynamics_dim = algo_config.model.dynamics_dim  # 45

        if self.use_ema:
            print(f"===> Using EMA with decay: {algo_config.training.ema.decay}")
            self.ema = EMA(algo_config.training.ema.decay)
            self.ema_update_every = algo_config.training.ema.update_every
            self.ema_start_step = algo_config.training.ema.start_step
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            # Track the last optimizer step that triggered EMA update.
            self.ema_last_global_step = -1
            self.reset_ema_parameters()

    @torch.no_grad()
    def extract_language_features(
        self, language_text: str, data_batch: dict = None, max_length: int = 30
    ):
        print(f"===> Extracting language features for '{language_text}'")
        t5_device = T5_EMBEDDER.device
        tokens = T5_EMBEDDER.tokenizer(
            language_text, return_tensors="pt", padding="longest", truncation=True
        )["input_ids"].to(t5_device)
        tokens = tokens.view(1, -1)
        lang_raw = T5_EMBEDDER.model(tokens).last_hidden_state.detach().float()
        lang_raw = lang_raw[:, :max_length]  # [1, max_length, 768]

        # Repeat the language features to match the desired max length
        idx = torch.arange(max_length, device=lang_raw.device) % lang_raw.shape[1]
        lang = lang_raw[:, idx]  # [1, max_length, 768]
        if data_batch is not None:
            data_batch["language_feature"] = lang
        assert lang.shape[1] == max_length
        return lang

    @torch.no_grad()
    def extract_dinov3_features(self, data_batch, color_key="color_init"):
        Hg, Wg = data_batch["start_state"].shape[-2:]
        color = F.interpolate(
            data_batch[color_key],
            scale_factor=self.dinov3_upsampling_factor,
            mode="bilinear",
            align_corners=True,
        )
        # Global EXTRACTOR is created at import with device="cuda" (often cuda:0).
        # Under DDP, batches live on each rank's GPU — move backbone once to match.
        target_dev = color.device
        if next(EXTRACTOR.model.parameters()).device != target_dev:
            EXTRACTOR.model.to(target_dev)
            EXTRACTOR.device = target_dev
        feature = EXTRACTOR.extract_features(color)[-1]
        feature = F.interpolate(
            feature, size=(Hg, Wg), mode="bilinear", align_corners=True
        )
        data_batch["start_state_dinov3_feature"] = feature

    @torch.no_grad()
    def check_loss_validity(self, data_batch):
        include_actions_loss = True
        include_dynamics_loss = True
        include_values_loss = "+vm" in self.algo_config.model.mode

        # Sanity check for actions
        action_valid = data_batch["action_valid"]  # [B, H, D]
        B, H, _ = action_valid.shape
        action_valid_left, action_valid_right = (
            action_valid[..., 0],
            action_valid[..., action_valid.shape[-1] // 2],
        )  # [B, H]; [B, H]
        action_valid_left = action_valid_left.mean(dim=-1) > 0.2  # [B]
        action_valid_right = action_valid_right.mean(dim=-1) > 0.2  # [B]
        action_valid_left_score = action_valid_left.sum() / B
        action_valid_right_score = action_valid_right.sum() / B
        action_valid_score = max(action_valid_left_score, action_valid_right_score)
        if action_valid_score <= 0.1:
            include_actions_loss = False

        # Sanity check for states
        if "state_valid" in data_batch:
            state_valid = data_batch["state_valid"]  # [B, D, H, W]
            state_valid = rearrange(state_valid, "b (t c) h w -> b t c h w", c=3)[
                :, 0, 0
            ].view(
                B, -1
            )  # [B, HW]
            state_valid = state_valid.mean(dim=-1) > 0.5  # [B]
            state_valid_score = state_valid.sum() / B
            if state_valid_score <= 0.1:
                include_dynamics_loss = False
        else:
            include_dynamics_loss = False
        return include_actions_loss, include_dynamics_loss, include_values_loss

    @property
    def checkpoint_monitor_keys(self):
        prefix = "ema_" if self.use_ema else ""
        return {f"actionRMSE": f"val/{prefix}action_rmse"}

    def forward(
        self,
        data_batch,
        action_only: bool = False,
        value_only: bool = False,
        input_actions: torch.Tensor = None,
        input_dynamics: torch.Tensor = None,
        enable_guidance: bool = False,
        num_samples: int = 1,
        w_advantage: float = 1.0,
        w_conditional: float = 1.0,
        align_to_current_state: bool = False,
        eval_guidance: bool = False,
        random_noise_scale: float = 0.0,
    ):
        """
        data_batch: dict
        action_only: bool
        input_actions: torch.Tensor
        enable_guidance: bool
        """
        data_batch["language_feature_null"] = self.langugae_tokens_null.expand(
            data_batch["language_feature"].shape[0], -1, -1
        ).to(data_batch["language_feature"].device)
        policy = self.ema_policy if self.use_ema else self.nets["policy"]
        outputs = policy(
            data_batch,
            action_only=action_only,
            value_only=value_only,
            input_actions=input_actions,
            input_dynamics=input_dynamics,
            enable_guidance=enable_guidance,
            num_samples=num_samples,
            w_advantage=w_advantage,
            w_conditional=w_conditional,
            eval_guidance=eval_guidance,
            random_noise_scale=random_noise_scale,
        )
        if align_to_current_state:
            assert "start_pos" in data_batch, "start_pos is required for alignment"
            pred_actions = outputs["action_predictions"]  # [B, N, H, D]
            num_samples = pred_actions.shape[1]
            start_pos = data_batch["start_pos"][:, None].repeat(
                1, num_samples, 1
            )  # [B, N, D]

            # First compute the offset
            pred_actions_rel = (
                DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                    pred_actions,
                    pred_actions[..., 0, :].clone().detach(),
                    has_finger_tips=self.action_dim == 48,
                )
            )  # [B, N, H, D]

            # Then align the predictions to the start position
            pred_actions_aligned = (
                DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
                    pred_actions_rel,
                    start_pos,
                    has_finger_tips=self.action_dim == 48,
                )
            )  # [B, N, H, D]
            outputs["action_predictions"] = pred_actions_aligned
        return outputs

    def set_guidance(self, guidance_config):
        policy = self.nets["policy"] if not self.use_ema else self.ema_policy
        policy.set_guidance(guidance_config)

    def clear_guidance(self):
        policy = self.nets["policy"] if not self.use_ema else self.ema_policy
        policy.current_guidance = None

    def configure_optimizers(self):
        optim_params = self.algo_config.optimzation
        optimizer = torch.optim.AdamW(
            params=self.nets["policy"].parameters(),
            lr=optim_params.learning_rate,
            betas=optim_params.betas,
            eps=optim_params.eps,
            weight_decay=optim_params.weight_decay,
        )
        return {
            "optimizer": optimizer,
        }

    def reset_ema_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step: int):
        if step < self.ema_start_step:
            self.reset_ema_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after optimizer step (global_step advances on optimizer step).
        if not self.use_ema:
            return
        global_step = int(self.trainer.global_step)
        if global_step <= self.ema_last_global_step:
            return
        self.ema_last_global_step = global_step
        if global_step % self.ema_update_every == 0:
            self.step_ema(global_step)

    def on_validation_start(self):
        self.visualize_batch_idx = None

    def training_step(self, data_batch, batch_idx):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]
        include_actions_loss, include_dynamics_loss, include_values_loss = (
            self.check_loss_validity(data_batch)
        )
        device = data_batch["gt_action"].device
        if (
            include_values_loss
            and self.value_end_loss_at_step != float("inf")
            and int(self.trainer.global_step) >= self.value_end_loss_at_step
            and not self._value_model_frozen
        ):
            print(
                f"=====> Freezing value model at step {self.trainer.global_step} because value_end_loss_at_step is reached!"
            )
            self.nets["policy"].set_models_frozen(["value_model"])
            self._value_model_frozen = True
        drop_mask_lang = (
            torch.rand(len(data_batch["language_feature"]), device=device)
            < self.cond_drop_language_p
        )  # [B]
        drop_mask_history_action = (
            torch.rand(len(data_batch["history_action"]), device=device)
            < self.cond_drop_action_p
        )  # [B]
        drop_mask_visual = (
            torch.rand(len(data_batch["history_visual_feature_patch"]), device=device)
            < self.cond_drop_visual_p
        )  # [B]
        drop_mask_advantage = (
            torch.rand(len(data_batch["advantage_label"]), device=device)
            < self.cond_drop_advantage_p
        )  # [B]
        # print(f"drop_mask_visual: {drop_mask_visual}")
        data_batch["history_action"][drop_mask_history_action] = -1e3
        data_batch["history_visual_feature_patch"][drop_mask_visual] = 1e-3
        data_batch["advantage_label"][drop_mask_advantage] = -1
        lang_slice = data_batch["language_feature"][drop_mask_lang]
        lang_slice.copy_(self.langugae_tokens_null.to(device).expand_as(lang_slice))

        if "history_action_rel" in data_batch:
            data_batch["history_action_rel"][drop_mask_history_action] = -1e3
        if "history_visual_feature_patch_gripper" in data_batch:
            data_batch["history_visual_feature_patch_gripper"][drop_mask_visual] = 1e-3

        # Random drop: with prob cond_drop_advantage_p set label to -1 (uncond)
        if include_dynamics_loss:
            drop_mask_history_dynamics = (
                torch.rand(len(data_batch["history_state"]), device=device)
                < self.cond_drop_state_p
            )  # [B]
            data_batch["start_state"][drop_mask_history_dynamics] = 0.0
            data_batch["history_state"][drop_mask_history_dynamics] = 0.0

        if (
            "+wm" in self.algo_config.model.mode
            and self.algo_config.model.wm_concat_dinov3_feature
        ):
            self.extract_dinov3_features(data_batch)

        # self.visualize_trajectory(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch)

        # Summarize loss
        total_loss = 0.0
        for lk in list(losses.keys()):
            losses[lk] = losses[lk] * self.algo_config.training.loss_weights[lk]
            if include_actions_loss and "actions" in lk:
                total_loss += losses[lk]
            if include_dynamics_loss and "dynamics" in lk:
                total_loss += losses[lk]
            if include_values_loss and "values" in lk:
                # Stop accumulating losses for the value model after a certain number of steps
                if int(self.trainer.global_step) >= self.value_end_loss_at_step:
                    continue
                # TD error for value model is only computed after a certain number of steps
                if (
                    lk == "absolute_loss_values_td"
                    # and self.curr_train_step < self.value_start_td_loss_at_step
                    and int(self.trainer.global_step) < self.value_start_td_loss_at_step
                ):
                    continue
                total_loss += losses[lk]

        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        elif not total_loss.requires_grad:
            # Frozen-only subgraphs yield no grad; PL still needs a scalar backward can run on.
            anchor = next((p for p in self.parameters() if p.requires_grad), None)
            if anchor is not None:
                total_loss = total_loss + 0.0 * anchor.sum()
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for lk_weight in self.algo_config.training.loss_weights.keys():
            if lk_weight in losses:
                if include_actions_loss and "actions" in lk_weight:
                    self.log(f"train/losses_" + lk_weight, losses[lk_weight])
                if include_dynamics_loss and "dynamics" in lk_weight:
                    self.log(f"train/losses_" + lk_weight, losses[lk_weight])
                # Stop training the value model after a certain number of steps
                if int(self.trainer.global_step) >= self.value_end_loss_at_step:
                    continue
                if include_values_loss and "values" in lk_weight:
                    # TD error for value model is only computed after a certain number of steps
                    if (
                        lk_weight == "absolute_loss_values_td"
                        # and self.curr_train_step < self.value_start_td_loss_at_step
                        and int(self.trainer.global_step)
                        < self.value_start_td_loss_at_step
                    ):
                        continue
                    self.log(f"train/losses_" + lk_weight, losses[lk_weight])
        self.log("train/losses_total_loss", total_loss)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def compute_progress_accuracy(self, data_batch, outputs):
        pred_progress = outputs["progress_predictions"][:, 0, 0]  # [B, 1, H] => [B]
        # Progress is the same across horizon, so just take the first timestep
        gt_progress = data_batch["gt_action"][:, 0, -1]  # [B, H, D] => [B]
        progress_rmse = (pred_progress - gt_progress).pow(2)
        progress_rmse = progress_rmse.mean().pow(0.5).float()
        data_batch.update({"pred_progress": pred_progress})
        return progress_rmse

    def compute_distance_to_goal_accuracy(self, data_batch, outputs):
        pred_distance_to_goal = outputs["dynamics_distance_to_goal_predictions"][
            :, 0
        ]  # [B, 3, 32, 32]
        gt_distance_to_goal = data_batch["gt_distance_to_goal"]
        state_valid = data_batch["state_valid"][:, :3]  # [B, D, 32, 32]
        distance_to_goal_rmse = (
            pred_distance_to_goal[state_valid > 0]
            - gt_distance_to_goal[state_valid > 0]
        ).pow(2)
        distance_to_goal_rmse = distance_to_goal_rmse.mean().pow(0.5).float()
        data_batch.update({"pred_distance_to_goal": pred_distance_to_goal})
        return distance_to_goal_rmse

    def compute_state_accuracy(self, data_batch, outputs):
        pred_state = outputs["dynamics_predictions"][:, 0]  # [B, D, 32, 32]
        gt_state = data_batch["gt_state"]  # [B, D, 32, 32]

        gt_state_valid = data_batch["state_valid"]  # [B, D, 32, 32]
        gt_state_valid = gt_state_valid[:, : gt_state.shape[1]]
        state_rmse = pred_state[gt_state_valid > 0] - gt_state[gt_state_valid > 0]
        state_rmse = state_rmse.pow(2)
        state_rmse = state_rmse.mean().pow(0.5).float()
        if self.algo_config.model.wm_predict_visual:
            pred_visual = outputs["visual_predictions"][:, 0]  # [B, D, 32, 32]
            pred_visual = (pred_visual) / 5.0
            gt_visual = data_batch["goal_visual_feature_patch"]
            feature_res = int(gt_visual.shape[-2] ** 0.5)
            gt_visual = rearrange(
                gt_visual, "b (h w) c -> b c h w", h=feature_res, w=feature_res
            )
            gt_visual = F.interpolate(
                gt_visual,
                size=(
                    pred_visual.shape[-2],
                    pred_visual.shape[-1],
                ),
                mode="bilinear",
                align_corners=False,
            )
            gt_state_valid = gt_state_valid[:, :1].expand(
                -1, gt_visual.shape[1], -1, -1
            )
            visual_rmse = (
                pred_visual[gt_state_valid > 0] - gt_visual[gt_state_valid > 0]
            ).pow(2)
            visual_rmse = visual_rmse.mean().pow(0.5).float()
            data_batch.update({"pred_visual": pred_visual})
        else:
            visual_rmse = torch.tensor(-1.0, dtype=torch.float32, device=self.device)
        data_batch.update({"pred_state": pred_state})
        return state_rmse, visual_rmse

    def compute_action_accuracy(self, data_batch, outputs):
        # Offset the predicted trajectories to the start position
        dim_action = self.algo_config.model.action_dim
        pred_actions = outputs["action_predictions"]  # [B, N, H, D]
        gt_actions = data_batch["gt_action"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 3]
        gt_actions_valid = data_batch["action_valid"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 3]
        if self.algo_config.model.am_predict_progress:
            gt_actions = gt_actions[:, :, :, :-1]
            gt_actions_valid = gt_actions_valid[:, :, :, :-1]
            assert gt_actions.shape[-1] == dim_action
            assert gt_actions_valid.shape[-1] == dim_action

        assert dim_action == pred_actions.shape[-1] == gt_actions.shape[-1]

        # Align the predictions to the initial waypoint of the GT trajectory
        pred_actions_rel = (
            DatasetUtils.transform_two_hands_trajectory_absolute_to_relative(
                pred_actions,
                pred_actions[..., 0, :],
                has_finger_tips=self.algo_config.model.action_dim == 48,
            )
        )
        pred_actions = DatasetUtils.transform_two_hands_trajectory_relative_to_absolute(
            pred_actions_rel,
            gt_actions[..., 0, :],
            has_finger_tips=self.algo_config.model.action_dim == 48,
        )

        pred_actions = torch.nan_to_num(pred_actions, nan=0.0)
        gt_actions = torch.nan_to_num(gt_actions, nan=0.0)

        pred_actions = torch.nan_to_num(pred_actions, nan=0.0)
        gt_actions = torch.nan_to_num(gt_actions, nan=0.0)
        # Measure the distance between the predicted and gt trajectories
        left_action_rmse = (
            pred_actions[:, :, :, : dim_action // 2][
                gt_actions_valid[:, :, :, : dim_action // 2] > 0
            ]
            - gt_actions[:, :, :, : dim_action // 2][
                gt_actions_valid[:, :, :, : dim_action // 2] > 0
            ]
        ).pow(2)
        right_action_rmse = (
            pred_actions[:, :, :, dim_action // 2 :][
                gt_actions_valid[:, :, :, dim_action // 2 :] > 0
            ]
            - gt_actions[:, :, :, dim_action // 2 :][
                gt_actions_valid[:, :, :, dim_action // 2 :] > 0
            ]
        ).pow(2)

        action_rmse = 0.0
        if len(left_action_rmse) > 0:
            left_action_rmse = (
                (left_action_rmse.sum() / len(left_action_rmse)).pow(0.5).float()
            )
            action_rmse += left_action_rmse
        else:
            left_action_rmse = torch.tensor(
                -1.0, dtype=torch.float32, device=self.device
            )

        if len(right_action_rmse) > 0:
            right_action_rmse = (
                (right_action_rmse.sum() / len(right_action_rmse)).pow(0.5).float()
            )
            action_rmse += right_action_rmse
        else:
            right_action_rmse = torch.tensor(
                -1.0, dtype=torch.float32, device=self.device
            )

        if action_rmse == 0.0:
            action_rmse = -1.0

        data_batch.update({"pred_actions": pred_actions})
        return left_action_rmse, right_action_rmse, action_rmse

    def compute_value_accuracy(self, data_batch, outputs):
        pred_value = outputs["value_predictions"][:, 0].squeeze(-1)  # [B]
        gt_value = data_batch["gt_state_value"].float()
        gt_value = gt_value.view(-1)
        value_rmse = (pred_value - gt_value).pow(2).mean().pow(0.5).float()
        data_batch.update({"pred_value": pred_value})
        return value_rmse

    def validation_step(self, data_batch, batch_idx):
        self._validation_step(data_batch, batch_idx, use_ema=False)
        if self.use_ema:
            self._validation_step(data_batch, batch_idx, use_ema=True)

    def _validation_step(self, data_batch, batch_idx, use_ema: bool = False):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]
        data_batch["advantage_label"] = -torch.ones(
            len(data_batch["gt_action"]),
            device=data_batch["gt_action"].device,
            dtype=torch.long,
        )  # unconditional advantage label
        curr_policy = self.nets["policy"] if not use_ema else self.ema_policy
        include_actions_loss, include_dynamics_loss, include_values_loss = (
            self.check_loss_validity(data_batch)
        )
        if self.algo_config.model.wm_concat_dinov3_feature and include_dynamics_loss:
            self.extract_dinov3_features(data_batch)
        losses = TensorUtils.detach(curr_policy.compute_losses(data_batch))
        vis_mode = self.train_config.validation.visualize_mode if not use_ema else ""
        action_only = not ("+wm" in self.algo_config.model.mode) and not (
            "+vm" in self.algo_config.model.mode
        )

        # Teacher forcing: feed GT actions/dynamics so dynamics and VM see GT-conditioned path.
        # Action/dynamics outputs are therefore not open-loop; for open-loop val use input_*=None.
        action_suffix = "_rel" if self.algo_config.model.am_use_relative_action else ""
        input_actions = (
            data_batch[f"gt_action{action_suffix}"] if not action_only else None
        )
        input_dynamics = (
            data_batch[f"history_state"]
            if "+vm" in self.algo_config.model.mode
            else None
        )
        with torch.no_grad():
            out = curr_policy(
                data_batch,
                action_only=action_only,
                input_actions=input_actions,
                input_dynamics=input_dynamics,
            )
        val_prefix = "ema_" if use_ema else ""
        if "+wm" in self.algo_config.model.mode:  # and include_dynamics_loss:
            state_rmse, visual_rmse = self.compute_state_accuracy(data_batch, out)
            self.log(
                f"val/{val_prefix}state_rmse", state_rmse, prog_bar=True, sync_dist=True
            )
            if vis_mode == "draw":
                self.visualize_state(data_batch)
                pred_state_vis = torchvision.utils.make_grid(
                    data_batch["pred_state_vis"], nrow=4
                )
                gt_state_vis = torchvision.utils.make_grid(
                    data_batch["gt_state_vis"], nrow=4
                )
                self.logger.log_image(
                    f"val/{val_prefix}pred_state_vis",
                    [pred_state_vis],
                    caption=[f"Batch {batch_idx}"],
                )
                self.logger.log_image(
                    "val/gt_state_vis",
                    [gt_state_vis],
                    caption=[f"Batch {batch_idx}"],
                )

            if self.algo_config.model.wm_predict_visual:
                self.log(
                    f"val/{val_prefix}visual_rmse",
                    visual_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )
                if vis_mode == "draw":
                    self.visualize_visual_prediction(data_batch)
                    goal_feature_vis = torchvision.utils.make_grid(
                        data_batch["goal_feature_vis"], nrow=4
                    )
                    pred_feature_vis = torchvision.utils.make_grid(
                        data_batch["pred_feature_vis"], nrow=4
                    )

                    self.logger.log_image(
                        f"val/goal_feature_vis",
                        [goal_feature_vis],
                        caption=[f"Batch {batch_idx}"],
                    )
                    self.logger.log_image(
                        f"val/{val_prefix}pred_feature_vis",
                        [pred_feature_vis],
                        caption=[f"Batch {batch_idx}"],
                    )

            # if self.algo_config.model.wm_concat_dinov3_feature:
            #     self.visualize_dinov3_feature(data_batch)
            #     dinov3_feature_vis = torchvision.utils.make_grid(
            #         data_batch["dinov3_feature_vis"], nrow=4
            #     )
            #     self.logger.log_image(
            #         "val/dinov3_feature_vis",
            #         [dinov3_feature_vis],
            #         caption=[f"Batch {batch_idx}"],
            #     )

        if "+vm" in self.algo_config.model.mode:
            value_rmse = self.compute_value_accuracy(data_batch, out)
            self.log(
                f"val/{val_prefix}value_rmse", value_rmse, prog_bar=True, sync_dist=True
            )

        # if include_actions_loss:
        if self.algo_config.model.am_predict_progress:
            progress_rmse = self.compute_progress_accuracy(data_batch, out)
            self.log(
                f"val/{val_prefix}progress_rmse",
                progress_rmse,
                prog_bar=True,
                sync_dist=True,
            )

        left_action_rmse, right_action_rmse, action_rmse = self.compute_action_accuracy(
            data_batch, out
        )
        self.log(
            f"val/{val_prefix}left_action_rmse",
            left_action_rmse,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"val/{val_prefix}right_action_rmse",
            right_action_rmse,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"val/{val_prefix}action_rmse",
            action_rmse,
            prog_bar=True,
            sync_dist=True,
        )

        # Visualize the trajectories only for the first batch
        if vis_mode == "draw":
            self.visualize_action(data_batch)
            pred_action_vis = torchvision.utils.make_grid(
                data_batch["pred_action_vis"], nrow=4
            )
            self.logger.log_image(
                f"val/{val_prefix}pred_action_vis",
                [pred_action_vis],
                caption=[f"Batch {batch_idx}"],
            )

            gt_action_vis = torchvision.utils.make_grid(
                data_batch["gt_action_vis"], nrow=4
            )
            self.logger.log_image(
                "val/gt_action_vis", [gt_action_vis], caption=[f"Batch {batch_idx}"]
            )

            # ## FIXME: Visualize the history visual feature, just for debugging
            # self.visualize_history_visual_features(data_batch)
            # history_visual_feature_patch_vis = torchvision.utils.make_grid(
            #     data_batch["history_visual_feature_patch_vis"], nrow=4
            # )
            # history_visual_feature_patch_gripper_vis = torchvision.utils.make_grid(
            #     data_batch["history_visual_feature_patch_gripper_vis"], nrow=4
            # )
            # self.logger.log_image(
            #     "val/history_visual_feature_patch_vis",
            #     [history_visual_feature_patch_vis],
            #     caption=[f"Batch {batch_idx}"],
            # )
            # self.logger.log_image(
            #     "val/history_visual_feature_patch_gripper_vis",
            #     [history_visual_feature_patch_gripper_vis],
            #     caption=[f"Batch {batch_idx}"],
            # )

        return_dict = {f"{val_prefix}losses": losses}
        return return_dict

    def visualize_visual_prediction(self, data_batch, **kwargs):
        target_size = 64
        goal_feature = data_batch["goal_visual_feature_patch"]
        pred_feature = data_batch["pred_visual"]
        goal_feature = goal_feature * 5.0
        feature_res = int(goal_feature.shape[-2] ** 0.5)
        goal_feature = rearrange(
            goal_feature, "b (h w) c -> b c h w", h=feature_res, w=feature_res
        )
        goal_feature = F.interpolate(
            goal_feature,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        pred_feature = F.interpolate(
            pred_feature,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        goal_feature_vis = EXTRACTOR.visualize_feature(goal_feature)
        pred_feature_vis = EXTRACTOR.visualize_feature(pred_feature)
        data_batch.update(
            {"goal_feature_vis": goal_feature_vis, "pred_feature_vis": pred_feature_vis}
        )
        return goal_feature_vis, pred_feature_vis

    def visualize_state(self, data_batch, **kwargs):
        num_tracks = (
            data_batch["start_state"].shape[-1] * data_batch["start_state"].shape[-2]
        )
        track_colors = DatasetUtils.random_colors(num_tracks)
        track_colors = np.array(track_colors)

        def _draw_state(start_state, state, state_valid, color, intr, text):
            # vis_img_init = (color * 255).astype(np.uint8)[:, :, ::-1].copy()
            color_vis = (color * 255).astype(np.uint8)[:, :, ::-1].copy()
            vis_img = np.zeros_like(color_vis)

            # start_state_proj = DatasetUtils.project_points_to_image(
            #     start_state, intr, np.eye(4)
            # )
            # start_state_proj = start_state_proj.astype(np.int32)
            # for i in range(start_state_proj.shape[0]):
            #     uv = start_state_proj[i]
            #     track_color = (
            #         int(track_colors[i, 2] * 255),
            #         int(track_colors[i, 1] * 255),
            #         int(track_colors[i, 0] * 255),
            #     )

            #     cv2.circle(vis_img_init, (uv[0], uv[1]), 3, track_color, -1)

            state_proj = DatasetUtils.project_points_to_image(state, intr, np.eye(4))
            state_proj = np.floor(state_proj).astype(np.int32)
            for i in range(state_proj.shape[0]):
                uv = state_proj[i]
                track_color = (
                    int(track_colors[i, 2] * 255),
                    int(track_colors[i, 1] * 255),
                    int(track_colors[i, 0] * 255),
                )
                if state_valid[i] > 0:
                    cv2.circle(vis_img, (uv[0], uv[1]), 3, track_color, -1)

            # vis_img = np.concatenate([vis_img_init, vis_img], axis=1)
            vis_img = cv2.addWeighted(color_vis, 0.2, vis_img, 0.8, 0)
            cv2.putText(
                vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            return vis_img

        batch_size = len(data_batch["color_init"])
        results_gt, results_pred = [], []
        for i in range(batch_size):
            color = data_batch["color_init"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            rel_time = data_batch["relative_time"][i].cpu().numpy()
            start_state = data_batch["start_state"][i].cpu().numpy()  # [24, 32, 32]
            gt_state = data_batch["gt_state"][i].cpu().numpy()  # [24, 32, 32]
            pred_state = data_batch["pred_state"][i].cpu().numpy()  # [24, 32, 32]
            gt_state_valid = data_batch["state_valid"][i].cpu().numpy()  # [24, 32, 32]
            gt_state_visib = data_batch["state_visib"][i].cpu().numpy()  # [8, 32, 32]
            gt_state_valid = gt_state_valid[: gt_state.shape[0]]

            D, Hg, Wg = gt_state.shape
            vis_img_gt_all_horizon = []
            vis_img_pred_all_horizon = []
            # Downsample the state for visualization
            N_state = D // 3
            vis_range = range(N_state)[:: N_state // 5]
            for ti in [-1] + list(vis_range) + [N_state - 1]:
                start_state_ti = start_state.reshape(1, 3, Hg, Wg)[0]
                gt_state_ti = gt_state.reshape(N_state, 3, Hg, Wg)[ti]
                pred_state_ti = pred_state.reshape(N_state, 3, Hg, Wg)[ti]
                gt_state_visib_ti = gt_state_visib.reshape(N_state, 1, Hg, Wg)[ti]
                gt_state_valid_ti = gt_state_valid.reshape(N_state, 3, Hg, Wg)[ti]

                if "pred_state_visib" in data_batch:
                    pred_state_visib_ti = (
                        data_batch["pred_state_visib"][i].cpu().numpy()
                    )
                    pred_state_visib_ti = pred_state_visib_ti.reshape(
                        N_state, 1, Hg, Wg
                    )[ti]
                else:
                    pred_state_visib_ti = gt_state_visib_ti

                start_state_ti = rearrange(start_state_ti, "h p1 p2 -> (p1 p2) h")
                gt_state_ti = rearrange(gt_state_ti, "h p1 p2 -> (p1 p2) h")
                pred_state_ti = rearrange(pred_state_ti, "h p1 p2 -> (p1 p2) h")
                gt_state_visib_ti = rearrange(
                    gt_state_visib_ti, "h p1 p2 -> (p1 p2) h"
                )[:, 0]
                pred_state_visib_ti = rearrange(
                    pred_state_visib_ti, "h p1 p2 -> (p1 p2) h"
                )[:, 0]
                gt_state_valid_ti = rearrange(
                    gt_state_valid_ti, "h p1 p2 -> (p1 p2) h"
                )[:, 0]
                vis_img_gt_ti = _draw_state(
                    start_state_ti,
                    gt_state_ti if ti != -1 else start_state_ti,
                    gt_state_valid_ti,
                    color,
                    intr,
                    f"{rel_time + ti * 1 / 15:.3f}s" if ti != -1 else "Init",
                )
                vis_img_pred_ti = _draw_state(
                    start_state_ti,
                    pred_state_ti if ti != -1 else start_state_ti,
                    gt_state_valid_ti,
                    color,
                    intr,
                    f"{rel_time + ti * 1 / 15:.3f}s" if ti != -1 else "Init",
                )
                vis_img_gt_all_horizon.append(vis_img_gt_ti)
                vis_img_pred_all_horizon.append(vis_img_pred_ti)
            vis_img_gt = np.concatenate(vis_img_gt_all_horizon, axis=1)
            vis_img_pred = np.concatenate(vis_img_pred_all_horizon, axis=1)
            vis_img_gt = torchvision.transforms.ToTensor()(vis_img_gt[..., [2, 1, 0]])
            vis_img_pred = torchvision.transforms.ToTensor()(
                vis_img_pred[..., [2, 1, 0]]
            )

            results_gt.append(vis_img_gt)
            results_pred.append(vis_img_pred)

        results_gt = torch.stack(results_gt, dim=0)  # [B, C, H, W]
        results_pred = torch.stack(results_pred, dim=0)
        data_batch.update({"pred_state_vis": results_pred, "gt_state_vis": results_gt})

    def visualize_action(self, data_batch, **kwargs):
        batch_size = len(data_batch["color"])
        results_gt, results_pred = [], []
        dim_action = self.algo_config.model.action_dim
        for i in range(batch_size):
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            gt_traj = data_batch["gt_action"][i].cpu().numpy()
            T_world_cam = data_batch["T_world_cam"][i].cpu().numpy()  # [4, 4]
            if self.algo_config.model.am_predict_action_frame == "world":
                gt_traj = DatasetUtils.transform_two_hands_trajectory(
                    gt_traj,
                    np.linalg.inv(T_world_cam),
                    action_dim=dim_action,
                    has_finger_tips=False,
                )  # [H, ACTION_DIM], [4, 4]
            gt_traj_valid = data_batch["action_valid"][i].cpu().numpy()
            gt_progress = None
            gt_state_value = None
            gt_distance_to_goal = None
            if self.algo_config.model.am_predict_progress:
                # Progress is the same across horizon, so just take the first timestep
                gt_progress = (
                    data_batch["gt_action"][i, 0, -1].cpu().numpy()
                )  # [B, H, D] => [B]

            if "+vm" in self.algo_config.model.mode:
                gt_state_value = data_batch["gt_state_value"][i].cpu().numpy()

            if (
                "+wm" in self.algo_config.model.mode
                and self.algo_config.model.wm_predict_distance_to_goal
            ):
                gt_distance_to_goal = (
                    data_batch["gt_state_residual"][i, -3:, :, :].cpu().numpy()
                )
                gt_distance_to_goal = np.abs(gt_distance_to_goal).mean()

            vis_gt = np.ascontiguousarray(color * 255, dtype=np.uint8)[
                :, :, ::-1
            ].copy()
            gt_traj_left = gt_traj[:, : dim_action // 2][:, :3]  # [H, 3]
            gt_traj_right = gt_traj[:, dim_action // 2 :][:, :3]  # [H, 3]
            gt_traj_left_valid = gt_traj_valid[:, : dim_action // 2][:, 0]  # [H,]
            gt_traj_right_valid = gt_traj_valid[:, dim_action // 2 :][:, 0]  # [H,]
            for traj_name, traj, traj_valid in zip(
                ["left", "right"],
                [gt_traj_left, gt_traj_right],
                [gt_traj_left_valid, gt_traj_right_valid],
            ):
                if traj_valid.sum() < 3:
                    continue
                cmap_name = "viridis" if traj_name == "left" else "jet"
                vis_gt = DatasetUtils.visualize_2d_trajectory(
                    vis_gt, traj, intr, cmap_name=cmap_name, **kwargs
                )

            if gt_progress is not None:
                cv2.putText(
                    vis_gt,
                    (
                        f"p*={gt_progress.item():.3f}"
                        if gt_progress is not None
                        else "N/A"
                    ),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            if gt_state_value is not None:
                cv2.putText(
                    vis_gt,
                    (
                        f"v*={gt_state_value.item():.3f}"
                        if gt_state_value is not None
                        else "N/A"
                    ),
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            if gt_distance_to_goal is not None:
                cv2.putText(
                    vis_gt,
                    (
                        f"d*={gt_distance_to_goal.item() * 1000:.2f}mm"
                        if gt_distance_to_goal is not None
                        else "N/A"
                    ),
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
            vis_gt = torchvision.transforms.ToTensor()(vis_gt[..., [2, 1, 0]])
            results_gt.append(vis_gt)

            if "pred_actions" in data_batch or "pred_distance_to_goal" in data_batch:
                # print("===> Visualizing pred actions")
                pred_trajs = data_batch["pred_actions"][i].cpu().numpy()
                T_world_cam = data_batch["T_world_cam"][i].cpu().numpy()  # [4, 4]
                if self.algo_config.model.am_predict_action_frame == "world":
                    pred_trajs = DatasetUtils.transform_two_hands_trajectory(
                        pred_trajs,
                        np.linalg.inv(T_world_cam),
                        action_dim=dim_action,
                        has_finger_tips=False,
                    )  # [H, ACTION_DIM], [4, 4]
                pred_progress = None
                pred_distance_to_goal = None
                pred_state_value = None

                if "pred_progress" in data_batch:
                    pred_progress = data_batch["pred_progress"][i].cpu().numpy()

                if "pred_value" in data_batch:
                    pred_state_value = data_batch["pred_value"][i].cpu().numpy()

                if "pred_distance_to_goal" in data_batch:
                    pred_distance_to_goal = (
                        data_batch["pred_distance_to_goal"][i].cpu().numpy()
                    )  # [B, 3, 32, 32] => [1]
                    pred_distance_to_goal = np.abs(pred_distance_to_goal).mean()

                pred_traj_colors = DatasetUtils.random_colors(len(pred_trajs))
                vis_pred = np.ascontiguousarray(color * 255, dtype=np.uint8)[
                    :, :, ::-1
                ].copy()
                traj_color = None
                for pi, pred_traj in enumerate(pred_trajs):
                    if len(pred_trajs) > 1:
                        traj_color = (pred_traj_colors[pi] * 255).astype(np.uint8)
                    pred_traj_left = pred_traj[:, : dim_action // 2][:, :3]
                    pred_traj_right = pred_traj[:, dim_action // 2 :][:, :3]
                    for traj_name, traj, traj_valid in zip(
                        ["left", "right"],
                        [pred_traj_left, pred_traj_right],
                        [gt_traj_left_valid, gt_traj_right_valid],
                    ):
                        if traj_valid.sum() < 3:
                            continue
                        cmap_name = "plasma" if traj_name == "left" else "turbo"
                        vis_pred = DatasetUtils.visualize_2d_trajectory(
                            vis_pred,
                            traj,
                            intr,
                            traj_color,
                            cmap_name=cmap_name,
                            **kwargs,
                        )

                if pred_progress is not None:
                    cv2.putText(
                        vis_pred,
                        (
                            f"p={pred_progress.item():.3f}"
                            if pred_progress is not None
                            else "N/A"
                        ),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                if pred_state_value is not None:
                    cv2.putText(
                        vis_pred,
                        (
                            f"v={pred_state_value.item():.3f}"
                            if pred_state_value is not None
                            else "N/A"
                        ),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                if pred_distance_to_goal is not None:
                    cv2.putText(
                        vis_pred,
                        (
                            f"d={pred_distance_to_goal.item() * 1000:.2f}mm"
                            if pred_distance_to_goal is not None
                            else "N/A"
                        ),
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                vis_pred = torchvision.transforms.ToTensor()(vis_pred[..., [2, 1, 0]])
                results_pred.append(vis_pred)

        results_gt = torch.stack(results_gt, dim=0)  # [B, C, H, W]
        results_pred = torch.stack(results_pred, dim=0)
        data_batch.update(
            {"pred_action_vis": results_pred, "gt_action_vis": results_gt}
        )

    def visualize_dinov3_feature(self, data_batch):
        # feature = data_batch["start_state_dinov3_feature"]
        feature = data_batch["start_state_dinov3_feature"]
        visualized_features = EXTRACTOR.visualize_feature(feature)
        data_batch.update({"dinov3_feature_vis": visualized_features})

    def visualize_trajectory_o3d(
        self,
        data_batch,
        window=False,
        return_vis=False,
        **kwargs,
    ):
        batch_size = len(data_batch["color"])
        results = []
        for i in range(batch_size):
            vis_o3d = []
            depth = data_batch["depth"][i].cpu().numpy()
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            # gt_traj = data_batch["gt_trajectory"][i].cpu().numpy()

            # backproject
            points_scene, scene_ids = DatasetUtils.backproject(
                depth,
                intr,
                depth > 0,
                # np.logical_and(hand_mask == 0, depth > 0),
                NOCS_convention=False,
            )

            colors_scene = color.copy()[scene_ids[0], scene_ids[1]]
            pcd_scene = DatasetUtils.visualize_points(points_scene, colors_scene)
            # gt_traj_vis = DatasetUtils.visualize_3d_trajectory(
            #     gt_traj, size=0.02, cmap_name="viridis"
            # )

            vis_o3d = [pcd_scene]  # + gt_traj_vis
            if "pred_actions" in data_batch:
                print("===> Visualizing pred trajectories")
                pred_trajs = data_batch["pred_actions"][i].cpu().numpy()
                pred_traj_colors = DatasetUtils.random_colors(len(pred_trajs))
                for pi, pred_traj in enumerate(pred_trajs):
                    _pred_traj_vis = DatasetUtils.visualize_3d_trajectory(
                        pred_traj,
                        size=0.03,
                        cmap_name="plasma",
                    )
                    if len(pred_trajs) > 1:
                        _pred_traj_vis = [
                            s.paint_uniform_color(pred_traj_colors[pi])
                            for s in _pred_traj_vis
                        ]

                    pred_traj_vis = _pred_traj_vis[0]
                    for ii in range(1, len(_pred_traj_vis)):
                        pred_traj_vis += _pred_traj_vis[ii]

                    vis_o3d += [pred_traj_vis]

            if window:
                o3d.visualization.draw(vis_o3d)

            if return_vis:
                return vis_o3d

        #     render_dist = np.median(np.linalg.norm(points_scene, axis=1))
        #     render_img = DatasetUtils.render_offscreen(
        #         vis_o3d,
        #         config_path,
        #         # dist=render_dist,
        #         resize_factor=0.5,
        #     )
        #     render_img = torchvision.transforms.ToTensor()(render_img)
        #     results.append(render_img)
        # results = torch.stack(results, dim=0)  # [B, C, H, W]
        # data_batch.update({"pred_vis": results})

    # Hack to visualize the history visual features
    def visualize_history_visual_features(self, data_batch):
        history_visual_feature_patch = data_batch["history_visual_feature_patch"]
        history_visual_feature_patch_gripper = data_batch[
            "history_visual_feature_patch_gripper"
        ]
        history_visual_feature_patch = rearrange(
            history_visual_feature_patch, "b t (h w) c -> b t c h w", h=14, w=14
        )[:, -1]
        history_visual_feature_patch_gripper = rearrange(
            history_visual_feature_patch_gripper, "b t (h w) c -> b t c h w", h=15, w=20
        )[:, -1]
        history_visual_feature_patch_vis = EXTRACTOR.visualize_feature(
            history_visual_feature_patch
        )
        history_visual_feature_patch_gripper_vis = EXTRACTOR.visualize_feature(
            history_visual_feature_patch_gripper
        )
        data_batch.update(
            {
                "history_visual_feature_patch_vis": history_visual_feature_patch_vis,
                "history_visual_feature_patch_gripper_vis": history_visual_feature_patch_gripper_vis,
            }
        )
        return (
            history_visual_feature_patch_vis,
            history_visual_feature_patch_gripper_vis,
        )

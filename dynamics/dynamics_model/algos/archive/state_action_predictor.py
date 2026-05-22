import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import utils.dataset_utils as DatasetUtils
import utils.tensor_utils as TensorUtils
from models.helpers import EMA
import open3d as o3d
import time
import cv2
import torchvision
from models.clip import clip, tokenize
import pandas as pd
from einops import rearrange
from models.stateact.diffuser_joint import StateActionDiffusionModel
from algos.feature_extractor import DINOv3FeatureExtractor

EXTRACTOR = DINOv3FeatureExtractor(
    model_name="dinov3_vitb16",
    device="cuda",
)


class StateActionDiffusionModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(StateActionDiffusionModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Conditioning parsing
        self.cond_drop_color_p = algo_config.training.conditioning_drop_color
        self.cond_drop_language_p = algo_config.training.conditioning_drop_language
        self.cond_drop_state_p = algo_config.training.conditioning_drop_state
        self.cond_drop_action_p = algo_config.training.conditioning_drop_action
        self.mode = algo_config.mode

        # Reasoning mode parsing
        self.reasoning_modes = algo_config.model.reasoning_modes
        self.reasoning_sampling_probability = np.array(
            [
                algo_config.training.reasoning_sampling_probability[r]
                for r in self.reasoning_modes
            ]
        )  # paired with the reasoning modes in the same order
        self.reasoning_sampling_probability = (
            self.reasoning_sampling_probability
            / self.reasoning_sampling_probability.sum()
        )
        self.reasoning_freq = {r: 0 for r in self.reasoning_modes}
        print(
            f"Reasoning modes: {self.reasoning_modes} has sampling probability: {self.reasoning_sampling_probability}"
        )

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        self.nets["policy"] = StateActionDiffusionModel(**policy_kwargs)

        self.curr_train_step = 0  # step within an epoch
        self.visualize_batch_idx = (
            None  # Will be set to a random batch index for visualization
        )
        assert "masked" in self.reasoning_modes, "Masked reasoning mode is required "

    @property
    def checkpoint_monitor_keys(self):
        return {"jointRMSE": "val-masked/joint_rmse"}

    @torch.no_grad()
    def extract_features(self, data_batch):
        Hg, Wg = data_batch["start_state"].shape[-2:]
        scale_factor = 4  # 4x downsampling, 1 for early-stage model...
        color_in = F.interpolate(
            data_batch["color_init"],
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=True,
        )
        feature = EXTRACTOR.extract_features(color_in)[-1]
        feature = F.interpolate(
            feature, size=(Hg, Wg), mode="bilinear", align_corners=True
        )
        data_batch["start_state_dinov3_feature"] = feature
        if "history_color_frames" in data_batch:
            B, T = data_batch["history_color_frames"].shape[:2]
            history_color_frames = rearrange(
                data_batch["history_color_frames"], "b t c h w -> (b t) c h w"
            )
            history_visual_features = EXTRACTOR.extract_features(history_color_frames)
            history_visual_feature = history_visual_features[-1]  # (b t) c h w
            history_visual_feature = rearrange(
                history_visual_feature, "(b t) c h w -> b t (h w) c", b=B, t=T
            )
            data_batch["history_visual_feature_dinov3"] = history_visual_feature.max(
                dim=-2
            )
            data_batch["history_visual_feature_patch_dinov3"] = history_visual_feature

    def forward(
        self,
        data_batch,
        mode="joint",
        reasoning_mode="masked",
        forward_value=True,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,
        apply_guidance=False,
        class_free_guide_w=0.0,
        guide_clean=False,
    ):
        curr_policy = self.nets["policy"]
        return curr_policy(
            data_batch,
            mode=mode,
            reasoning_mode=reasoning_mode,
            num_samp=num_samp,
            forward_value=forward_value,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            apply_guidance=apply_guidance,
            class_free_guide_w=class_free_guide_w,
            guide_clean=guide_clean,
        )

    def configure_optimizers(self):
        if self.mode in ["action", "joint"]:
            action_decoder_type = self.algo_config.model.action_decoder_type
            if action_decoder_type == "cdit":
                optim_params = self.algo_config.optimzation
                optimizer = optim.AdamW(
                    params=self.nets["policy"].parameters(),
                    lr=optim_params.learning_rate,
                )
            elif action_decoder_type == "transformer":
                optim_params = self.algo_config.optimzation
                optimizer = self.nets[
                    "policy"
                ].action_diffusion_model.action_decoder.configure_optimizers(
                    learning_rate=optim_params.learning_rate,
                )
                # Automatically add all other network components to optimizer
                for attr_name in dir(self.nets["policy"]):
                    if (
                        attr_name.startswith("_")
                        or attr_name
                        == "action_diffusion_model"  # Skip the action diffusion model as it is already added
                    ):
                        continue
                    attr_value = getattr(self.nets["policy"], attr_name)
                    if (
                        isinstance(attr_value, nn.Module)
                        and len(list(attr_value.parameters())) > 0
                    ):
                        optimizer.add_param_group({"params": attr_value.parameters()})
            else:
                raise ValueError(f"Invalid action decoder type: {action_decoder_type}")
        else:
            optimizer = optim.AdamW(
                params=self.nets["policy"].state_diffusion_model.parameters(),
                lr=self.algo_config.optimzation.learning_rate,
            )

        # Compute the model size of trainable parameters
        size_model = 0
        for param in optimizer.param_groups:
            for p in param["params"]:
                if p.data.is_floating_point():
                    size_model += p.numel() * torch.finfo(p.data.dtype).bits
                else:
                    size_model += p.numel() * torch.iinfo(p.data.dtype).bits

        print(
            f"================== Trainable parameters size: {size_model} / bit | {size_model / 8e6:.2f} / MB =================="
        )

        return optimizer

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def on_validation_start(self):
        self.visualize_batch_idx = None

    def sample_reasoning_mode(self, data_batch, reasoning_mode=None):
        # Draw a random reasoning mode
        # _reasoning_mode = np.random.choice(["masked", "non-masked"])
        # if ("forward" in self.reasoning_modes or "inverse" in self.reasoning_modes) and _reasoning_mode == "non-masked":
        #     reasoning_mode = np.random.choice(list(set(self.reasoning_modes) - set(["masked"])))
        # else:
        #     reasoning_mode = _reasoning_mode # "masked" or "non-masked"

        if reasoning_mode is None:
            # reasoning_mode = np.random.choice(self.reasoning_modes)
            reasoning_mode = np.random.choice(
                self.reasoning_modes,
                p=self.reasoning_sampling_probability,
            )

        # Sanity check for actions
        action_valid = data_batch["action_valid"]  # [B, H, D]
        B, H, _ = action_valid.shape
        action_valid_left, action_valid_right = (
            action_valid[..., 0],
            action_valid[..., action_valid.shape[-1] // 2],
        )  # [B, H]; [B, H]
        action_valid_left = action_valid_left.mean(dim=-1) > 0.2
        action_valid_right = action_valid_right.mean(dim=-1) > 0.2
        action_valid_left_score = action_valid_left.sum() / B
        action_valid_right_score = action_valid_right.sum() / B
        action_valid_score = max(action_valid_left_score, action_valid_right_score)

        if action_valid_score <= 0.3 and reasoning_mode != "forward":
            # if self.reasoning_freq["forward"] < 50:
            #     print(
            #         f"DEBUG ONLY: Action valid score is {action_valid_score}, setting reasoning mode to forward from {reasoning_mode}"
            #     )
            reasoning_mode = "forward"

        # Sanity check for states
        state_valid = data_batch["state_valid"]  # [B, D, H, W]
        state_valid = rearrange(state_valid, "b (t c) h w -> b t c h w", c=3)[
            :, 0, 0
        ].view(
            B, -1
        )  # [B, H, W]
        state_valid = state_valid.mean(dim=-1) > 0.2  # [B]
        state_valid_score = state_valid.sum() / B
        if state_valid_score <= 0.3 and reasoning_mode != "inverse":
            reasoning_mode = "inverse"

        return reasoning_mode

    def training_step(self, data_batch, batch_idx):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]
        drop_mask_color = (
            torch.rand(len(data_batch["history_visual_feature_patch"]))
            < self.cond_drop_color_p
        )
        drop_mask_lang = (
            torch.rand(len(data_batch["language_feature"])) < self.cond_drop_language_p
        )
        drop_mask_history_action = (
            torch.rand(len(data_batch["history_action"])) < self.cond_drop_action_p
        )
        drop_mask_history_state = (
            torch.rand(len(data_batch["history_state"])) < self.cond_drop_state_p
        )
        data_batch["language_feature"][drop_mask_lang].fill_(0)
        data_batch["history_visual_feature"][drop_mask_color].fill_(0)
        data_batch["history_visual_feature_patch"][drop_mask_color].fill_(0)
        data_batch["history_action"][drop_mask_history_action].fill_(-1e3)
        data_batch["history_state"][drop_mask_history_state].fill_(-1e3)
        data_batch["start_pos"][drop_mask_history_action].fill_(-1e3)

        self.extract_features(data_batch)
        # self.visualize_trajectory(data_batch)
        reasoning_mode = self.sample_reasoning_mode(data_batch)
        self.reasoning_freq[reasoning_mode] += 1

        # Train the model under current reasoning mode
        losses = self.nets["policy"].compute_losses(
            data_batch, mode=self.mode, reasoning_mode=reasoning_mode
        )

        # Summarize loss
        total_loss = 0.0
        for lk in list(losses.keys()):
            # Delete the reasoning mode from the loss key
            for r in self.reasoning_modes:
                if f"_{r}" in lk:
                    lk_weight = lk.replace(
                        f"_{r}", ""
                    )  # Delete the reasoning mode from the loss key
                    break
            losses[lk] = losses[lk] * self.algo_config.training.loss_weights[lk_weight]
            losses[lk_weight] = losses[lk]
            total_loss += losses[lk]

        for r in self.reasoning_modes:
            for lk, l in losses.items():
                if f"_{r}" in lk:
                    self.log(f"train-{r}/losses_" + lk, l)

        for lk_weight in self.algo_config.training.loss_weights.keys():
            if lk_weight in losses:
                self.log(f"train/losses_" + lk_weight, losses[lk_weight])

        self.log("train/losses_total_loss", total_loss)
        for r, f in self.reasoning_freq.items():
            self.log(f"train/training_frequency_{r}", f)
        self.training_step_end(data_batch)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def compute_state_accuracy(self, data_batch, outputs):
        pred_state = outputs["state_predictions"][:, 0]  # [B, D, 32, 32]
        if self.algo_config.model.state_include_visibility:
            pred_state, pred_state_visibility = (
                pred_state[..., : self.algo_config.model.output_state_dim, :, :],
                pred_state[..., self.algo_config.model.output_state_dim :, :, :],
            )
        else:
            pred_state_visibility = None
        gt_state = data_batch["gt_state"]  # [B, D, 32, 32]
        gt_state_valid = data_batch["state_valid"]  # [B, D, 32, 32]
        state_rmse = pred_state[gt_state_valid > 0] - gt_state[gt_state_valid > 0]
        state_rmse = state_rmse.pow(2)
        state_rmse = state_rmse.mean().pow(0.5).float()
        data_batch.update({"pred_state": pred_state})
        if pred_state_visibility is not None:
            data_batch.update({"pred_state_visib": pred_state_visibility > 0.5})
        return state_rmse

    def compute_action_accuracy(self, data_batch, outputs):
        # Offset the predicted trajectories to the start position
        dim_action = self.algo_config.model.output_action_dim
        pred_actions = outputs["action_predictions"]  # [B, N, H, 3]
        gt_actions = data_batch["gt_action"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 3]
        gt_actions_valid = data_batch["action_valid"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 3]
        assert dim_action == pred_actions.shape[-1] == gt_actions.shape[-1]

        gt_start_pos = data_batch["start_pos"]  # [B, 3]
        pred_start_poses = pred_actions[:, :, 0]  # [B, N, 3]
        offsets = gt_start_pos[:, None, :] - pred_start_poses  # [B, N, 3]
        pred_actions += offsets[:, :, None, :]  # [B, N, H, 3]
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
            action_rmse += right_action_rmse
        if action_rmse == 0.0:
            action_rmse = -1.0

        data_batch.update({"pred_actions": pred_actions})
        return left_action_rmse, right_action_rmse, action_rmse

    def compute_value_accuracy(self, data_batch, outputs):
        pred_value = outputs["value_predictions"]  # [B, 1]
        gt_value = data_batch["gt_state_value"]  # [B, 1]
        value_valid = data_batch["value_valid"]  # [B, 1]
        value_rmse = (pred_value[value_valid > 0] - gt_value[value_valid > 0]).pow(2)
        value_rmse = value_rmse.mean().pow(0.5).float()
        data_batch.update({"pred_values": pred_value})
        return value_rmse

    def validation_step(self, data_batch, batch_idx):
        for ri, _reasoning_mode in enumerate(self.reasoning_modes):
            data_batch_current = TensorUtils.join_dimensions(
                data_batch, begin_axis=0, end_axis=2
            )  # [B, T, ...] -> [B*T, ...]
            self.extract_features(data_batch_current)
            reasoning_mode = self.sample_reasoning_mode(
                data_batch_current, reasoning_mode=_reasoning_mode
            )  # If full zero, then set to forward or inverse

            curr_policy = self.nets["policy"]
            losses = TensorUtils.detach(
                curr_policy.compute_losses(
                    data_batch_current, mode=self.mode, reasoning_mode=reasoning_mode
                )
            )

            out = curr_policy(
                data_batch_current,
                mode=self.mode,
                num_samp=1,
                return_diffusion=False,
                return_guidance_losses=False,
                apply_guidance=False,
            )

            # Check the state predictions accuracy and update the output
            # to the data batch
            value_rmse = -1.0
            if self.mode in ["state", "joint"] and reasoning_mode in [
                "forward",
                "masked",
            ]:
                state_rmse = self.compute_state_accuracy(data_batch_current, out)
            else:
                state_rmse = -1.0

            if self.mode in ["action", "joint"] and reasoning_mode in [
                "inverse",
                "masked",
            ]:
                left_action_rmse, right_action_rmse, action_rmse = (
                    self.compute_action_accuracy(data_batch_current, out)
                )
                if self.algo_config.model.predict_state_value:
                    value_rmse = self.compute_value_accuracy(data_batch_current, out)
            else:
                left_action_rmse, right_action_rmse, action_rmse = -1.0, -1.0, -1.0

            if reasoning_mode in ["masked"]:
                joint_rmse = 0.0
                if state_rmse >= 0.0:
                    joint_rmse += state_rmse / 2
                if action_rmse >= 0.0:
                    joint_rmse += action_rmse / 2

            # Set random batch index for visualization on first validation step
            if self.visualize_batch_idx is None:
                num_batches = len(self.trainer.datamodule.val_dataloader())
                if self.train_config.validation.num_val_batches is not None:
                    num_batches = min(
                        num_batches, self.train_config.validation.num_val_batches
                    )
                self.visualize_batch_idx = min(
                    np.random.randint(0, num_batches), num_batches - 1
                )

            if batch_idx == self.visualize_batch_idx:
                if self.mode in ["state", "joint"] and reasoning_mode in [
                    "forward",
                    "masked",
                ]:
                    self.visualize_state(data_batch_current)

                    pred_state_vis = torchvision.utils.make_grid(
                        data_batch_current["pred_state_vis"], nrow=4
                    )
                    self.logger.log_image(
                        f"val-{reasoning_mode}/pred_state_vis",
                        [pred_state_vis],
                        caption=[f"Batch {batch_idx}"],
                    )
                    gt_state_vis = torchvision.utils.make_grid(
                        data_batch_current["gt_state_vis"], nrow=4
                    )
                    self.logger.log_image(
                        f"val-{reasoning_mode}/gt_state_vis",
                        [gt_state_vis],
                        caption=[f"Batch {batch_idx}"],
                    )

                if self.mode in ["action", "joint"] and reasoning_mode in [
                    "inverse",
                    "masked",
                ]:
                    self.visualize_action(data_batch_current)
                    pred_action_vis = torchvision.utils.make_grid(
                        data_batch_current["pred_action_vis"], nrow=4
                    )
                    self.logger.log_image(
                        f"val-{reasoning_mode}/pred_action_vis",
                        [pred_action_vis],
                        caption=[f"Batch {batch_idx}"],
                    )
                    gt_action_vis = torchvision.utils.make_grid(
                        data_batch_current["gt_action_vis"], nrow=4
                    )
                    self.logger.log_image(
                        f"val-{reasoning_mode}/gt_action_vis",
                        [gt_action_vis],
                        caption=[f"Batch {batch_idx}"],
                    )

            if self.mode in ["action", "joint"] and reasoning_mode in [
                "masked",
                "inverse",
            ]:
                self.log(
                    f"val-{reasoning_mode}/action_rmse",
                    action_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"val-{reasoning_mode}/left_action_rmse",
                    left_action_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"val-{reasoning_mode}/right_action_rmse",
                    right_action_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"val-{reasoning_mode}/value_rmse",
                    value_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )

            if self.mode in ["state", "joint"] and reasoning_mode in [
                "masked",
                "forward",
            ]:
                self.log(
                    f"val-{reasoning_mode}/state_rmse",
                    state_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )

            if reasoning_mode in ["masked"]:
                self.log(
                    f"val-{reasoning_mode}/joint_rmse",
                    joint_rmse,
                    prog_bar=True,
                    sync_dist=True,
                )

            # Log the diffusion loss for checkpoint monitoring
            for lk, l in losses.items():
                self.log(f"val-{reasoning_mode}/losses_" + lk, l, sync_dist=True)
            # Visualize the trajectories only for the first batch
            return_dict = {
                "losses": losses,
            }

        return return_dict

    def visualize_dinov3_feature(self, data_batch):
        # feature = data_batch["start_state_dinov3_feature"]
        if "history_visual_feature_patch_dinov3" in data_batch:
            index = np.random.randint(
                0, data_batch["history_visual_feature_patch_dinov3"].shape[1]
            )
            feature = data_batch["history_visual_feature_patch_dinov3"][
                :, index
            ]  # [B, 196, 768]
            feature = rearrange(feature, "b (h w) c -> b c h w", h=14, w=14)
        else:
            feature = data_batch["start_state_dinov3_feature"]
        visualized_features = EXTRACTOR.visualize_feature(feature)
        data_batch.update({"dinov3_feature_vis": visualized_features})

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
            state_proj = state_proj.astype(np.int32)
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
        dim_action = self.algo_config.model.output_action_dim
        for i in range(batch_size):
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            gt_traj = data_batch["gt_action"][i].cpu().numpy()
            gt_traj_valid = data_batch["action_valid"][i].cpu().numpy()
            gt_value = None
            if "gt_state_value" in data_batch and "value_valid" in data_batch:
                value_valid = data_batch["value_valid"][i].cpu().numpy()
                if value_valid > 0:
                    gt_value = data_batch["gt_state_value"][i].cpu().numpy()

            pred_value = None
            if "pred_values" in data_batch:
                pred_value = data_batch["pred_values"][i].cpu().numpy()

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

            cv2.putText(
                vis_gt,
                (f"v={gt_value:.3f}" if gt_value is not None else "N/A"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            vis_gt = torchvision.transforms.ToTensor()(vis_gt[..., [2, 1, 0]])
            results_gt.append(vis_gt)

            if "pred_actions" in data_batch:
                # print("===> Visualizing pred actions")
                pred_trajs = data_batch["pred_actions"][i].cpu().numpy()
                pred_value = None
                if "pred_values" in data_batch:
                    pred_value = data_batch["pred_values"][i].cpu().numpy()

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
                        cmap_name = "plasma" if traj_name == "left" else "turbo"
                        vis_pred = DatasetUtils.visualize_2d_trajectory(
                            vis_pred,
                            traj,
                            intr,
                            traj_color,
                            cmap_name=cmap_name,
                            **kwargs,
                        )

                cv2.putText(
                    vis_pred,
                    (f"v={pred_value:.3f}" if pred_value is not None else "N/A"),
                    (10, 30),
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

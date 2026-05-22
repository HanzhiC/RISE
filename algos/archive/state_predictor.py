import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import utils.dataset_utils as DatasetUtils
import utils.tensor_utils as TensorUtils
from models.stateact.diffuser_state import StateDiffusionModel
from models.helpers import EMA
import open3d as o3d
import time
import cv2
import torchvision
from models.clip import clip, tokenize
import pandas as pd
from einops import rearrange


class StateDiffusionModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(StateDiffusionModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Conditioning parsing
        self.cond_drop_color_p = algo_config.training.conditioning_drop_color
        self.cond_drop_language_p = algo_config.training.conditioning_drop_language
        self.cond_drop_state_p = algo_config.training.conditioning_drop_state

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        self.nets["policy"] = StateDiffusionModel(**policy_kwargs)

        self.curr_train_step = 0  # step within an epoch
        self.visualize_batch_idx = (
            None  # Will be set to a random batch index for visualization
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"stateRMSE": "val/state_rmse"}

    def forward(
        self,
        data_batch,
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
            num_samp,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            apply_guidance=apply_guidance,
            class_free_guide_w=class_free_guide_w,
            guide_clean=guide_clean,
        )

    def configure_optimizers(self):
        optim_params = self.algo_config.optimzation
        optimizer = optim.AdamW(
            params=self.nets["policy"].parameters(),
            lr=optim_params.learning_rate,
        )
        return optimizer

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def on_validation_start(self):
        self.visualize_batch_idx = None

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
        drop_mask_state = (
            torch.rand(len(data_batch["history_action"])) < self.cond_drop_state_p
        )

        data_batch["history_visual_feature"][drop_mask_color].fill_(0)
        data_batch["history_visual_feature_patch"][drop_mask_color].fill_(0)
        data_batch["history_action"][drop_mask_state].fill_(1e3)
        data_batch["language_feature"][drop_mask_lang].fill_(0)
        data_batch["start_state"][drop_mask_state].fill_(0)

        # self.visualize_trajectory(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch)

        # Summarize loss
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.training.loss_weights[lk]
            total_loss += losses[lk]

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        self.training_step_end(data_batch)

        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def validation_step(self, data_batch, batch_idx):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]
        curr_policy = self.nets["policy"]
        losses = TensorUtils.detach(curr_policy.compute_losses(data_batch))

        # Log the diffusion loss for checkpoint monitoring
        if "diffusion_loss" in losses:
            self.log(
                "val/losses_diffusion_loss", losses["diffusion_loss"], sync_dist=True
            )

        out = curr_policy(
            data_batch,
            num_samp=1,  # self.algo_config.training.num_eval_samples,
            return_diffusion=False,
            return_guidance_losses=False,  # FIXME: this is a hack to avoid guidance
            apply_guidance=False,  # FIXME: this is a hack to avoid guidance
        )

        # Offset the predicted trajectories to the start position
        pred_state = out["predictions"][:, 0]  # [B, D, 32, 32]

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
        self.log("val/state_rmse", state_rmse, prog_bar=True, sync_dist=True)

        # Log the images
        data_batch.update({"pred_state": pred_state})
        if pred_state_visibility is not None:
            data_batch.update({"pred_state_visib": pred_state_visibility > 0.5})

        # Visualize the trajectories only for the first batch
        return_dict = {"losses": losses}

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
            print("===> Visualizing state from batch {}".format(batch_idx))
            self.visualize_state(data_batch)

            pred_state_vis = torchvision.utils.make_grid(
                data_batch["pred_state_vis"], nrow=4
            )
            self.logger.log_image(
                "val/pred_state_vis", [pred_state_vis], caption=[f"Batch {batch_idx}"]
            )

            gt_state_vis = torchvision.utils.make_grid(
                data_batch["gt_state_vis"], nrow=4
            )
            self.logger.log_image(
                "val/gt_state_vis", [gt_state_vis], caption=[f"Batch {batch_idx}"]
            )

            return_dict["vis"] = data_batch["pred_state_vis"]
        return return_dict

    def visualize_state(self, data_batch, **kwargs):
        num_tracks = (
            data_batch["start_state"].shape[-1] * data_batch["start_state"].shape[-2]
        )
        track_colors = DatasetUtils.random_colors(num_tracks)
        track_colors = np.array(track_colors)

        def _draw_state(start_state, state, state_valid, color, intr, text):
            # vis_img_init = (color * 255).astype(np.uint8)[:, :, ::-1].copy()
            vis_img = (color * 255).astype(np.uint8)[:, :, ::-1].copy()

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
            cv2.putText(
                vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            return vis_img

        batch_size = len(data_batch["color"])
        results_gt, results_pred = [], []
        for i in range(batch_size):
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()

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
            vis_range = range(N_state)[::N_state // 5]
            for ti in list(vis_range) + [N_state - 1]:
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
                    start_state_ti, gt_state_ti, gt_state_valid_ti, color, intr, str(ti)
                )
                vis_img_pred_ti = _draw_state(
                    start_state_ti,
                    pred_state_ti,
                    gt_state_valid_ti,
                    color,
                    intr,
                    str(ti),
                )
                vis_img_gt_all_horizon.append(vis_img_gt_ti)
                vis_img_pred_all_horizon.append(vis_img_pred_ti)
            vis_img_gt = np.concatenate(vis_img_gt_all_horizon, axis=1)
            vis_img_pred = np.concatenate(vis_img_pred_all_horizon, axis=1)
            vis_img_gt = torchvision.transforms.ToTensor()(vis_img_gt[..., [2, 1, 0]])
            vis_img_pred = torchvision.transforms.ToTensor()(vis_img_pred[..., [2, 1, 0]])

            results_gt.append(vis_img_gt)
            results_pred.append(vis_img_pred)

        results_gt = torch.stack(results_gt, dim=0)  # [B, C, H, W]
        results_pred = torch.stack(results_pred, dim=0)
        data_batch.update({"pred_state_vis": results_pred, "gt_state_vis": results_gt})

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import utils.dataset_utils as DatasetUtils
import utils.tensor_utils as TensorUtils
from models.stateact.diffuser_action import ActionDiffusionModel
from models.helpers import EMA
import open3d as o3d
import time
import cv2
import torchvision
from models.clip import clip, tokenize
import pandas as pd


class ActionDiffusionModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(ActionDiffusionModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Conditioning parsing
        self.cond_drop_color_p = algo_config.training.conditioning_drop_color
        self.cond_drop_language_p = algo_config.training.conditioning_drop_language
        self.cond_drop_state_p = algo_config.training.conditioning_drop_state

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        self.nets["policy"] = ActionDiffusionModel(**policy_kwargs)

        self.curr_train_step = 0  # step within an epoch
        self.visualize_batch_idx = (
            None  # Will be set to a random batch index for visualization
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            # "valLoss": "val/losses_diffusion_loss",
            # "leftRMSE": "val/left_action_rmse",
            # "rightRMSE": "val/right_action_rmse",
            "actionRMSE": "val/action_rmse"
        }

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
        model_type = self.algo_config.model.decoder_type

        if model_type == "cdit":
            optim_params = self.algo_config.optimzation
            optimizer = optim.AdamW(
                params=self.nets["policy"].parameters(),
                lr=optim_params.learning_rate,
            )
            return optimizer
        elif model_type == "transformer":
            optim_params = self.algo_config.optimzation
            optimizer = self.nets["policy"].action_decoder.configure_optimizers(
                learning_rate=optim_params.learning_rate,
            )

            optimizer.add_param_group(
                {"params": self.nets["policy"].action_proj.parameters()}
            )
            optimizer.add_param_group(
                {"params": self.nets["policy"].visual_proj.parameters()}
            )
            optimizer.add_param_group(
                {"params": self.nets["policy"].language_proj.parameters()}
            )
            # Add optimizer groups
            if self.algo_config.model.use_feature_fuser:
                optimizer.add_param_group(
                    {"params": self.nets["policy"].fuser.parameters()}
                )
            return optimizer
        else:
            raise ValueError(f"Invalid model type: {model_type}")

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

        # self.visualize_trajectory(data_batch)
        losses = self.nets["policy"].compute_losses(data_batch)

        # Summarize loss
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.training.loss_weights[lk]
            total_loss += losses[lk]

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        # Monitor the trajectory length
        gt_traj = data_batch["gt_action"]
        traj_len = torch.linalg.norm(gt_traj[:, 0] - gt_traj[:, -1], dim=-1)
        traj_len = traj_len.mean()
        self.log("train/traj_len", traj_len)
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
        pred_actions = out["predictions"]  # [B, N, H, 3]
        gt_actions = data_batch["gt_action"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 3]
        gt_actions_valid = data_batch["action_valid"][:, None].repeat(
            1, pred_actions.shape[1], 1, 1
        )  # [B, N, H, 2]
        gt_start_pos = data_batch["start_pos"]  # [B, 3]
        pred_start_poses = pred_actions[:, :, 0]  # [B, N, 3]
        offsets = gt_start_pos[:, None, :] - pred_start_poses  # [B, N, 3]
        pred_actions += offsets[:, :, None, :]  # [B, N, H, 3]
        pred_actions = torch.nan_to_num(pred_actions, nan=0.0)
        gt_actions = torch.nan_to_num(gt_actions, nan=0.0)
        # Measure the distance between the predicted and gt trajectories
        left_action_rmse = (
            pred_actions[:, :, :, :3][gt_actions_valid[:, :, :, :3] > 0]
            - gt_actions[:, :, :, :3][gt_actions_valid[:, :, :, :3] > 0]
        ).pow(2)
        right_action_rmse = (
            pred_actions[:, :, :, 3:][gt_actions_valid[:, :, :, 3:] > 0]
            - gt_actions[:, :, :, 3:][gt_actions_valid[:, :, :, 3:] > 0]
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

        self.log(
            "val/left_action_rmse", left_action_rmse, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/right_action_rmse", right_action_rmse, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/action_rmse",
            action_rmse,
            prog_bar=True,
            sync_dist=True,
        )

        # Log the images
        data_batch.update({"pred_actions": pred_actions})

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
            print("===> Visualizing trajectories from batch {}".format(batch_idx))
            self.visualize_trajectory(data_batch)

            pred_action_vis = torchvision.utils.make_grid(
                data_batch["pred_action_vis"], nrow=4
            )
            self.logger.log_image(
                "val/pred_action_vis", [pred_action_vis], caption=[f"Batch {batch_idx}"]
            )

            gt_action_vis = torchvision.utils.make_grid(
                data_batch["gt_action_vis"], nrow=4
            )
            self.logger.log_image(
                "val/gt_action_vis", [gt_action_vis], caption=[f"Batch {batch_idx}"]
            )

            return_dict["vis"] = data_batch["pred_vis"]
        return return_dict

    def visualize_action(self, data_batch, **kwargs):
        batch_size = len(data_batch["color"])
        results_gt, results_pred = [], []
        for i in range(batch_size):
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            gt_traj = data_batch["gt_action"][i].cpu().numpy()
            gt_traj_valid = data_batch["action_valid"][i].cpu().numpy()

            vis_gt = np.ascontiguousarray(color * 255, dtype=np.uint8)[
                :, :, ::-1
            ].copy()
            gt_traj_left = gt_traj[:, :3]  # [H, 3]
            gt_traj_right = gt_traj[:, 3:]  # [H, 3]
            gt_traj_left_valid = gt_traj_valid[:, :3][:, 0]  # [H,]
            gt_traj_right_valid = gt_traj_valid[:, 3:][:, 0]  # [H,]
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
                    "GT",
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
                pred_traj_colors = DatasetUtils.random_colors(len(pred_trajs))
                vis_pred = np.ascontiguousarray(color * 255, dtype=np.uint8)[
                    :, :, ::-1
                ].copy()
                traj_color = None
                for pi, pred_traj in enumerate(pred_trajs):
                    if len(pred_trajs) > 1:
                        traj_color = (pred_traj_colors[pi] * 255).astype(np.uint8)
                    pred_traj_left = pred_traj[:, :3]
                    pred_traj_right = pred_traj[:, 3:]
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
                        "Pred",
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

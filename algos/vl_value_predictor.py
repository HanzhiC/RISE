import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import utils.tensor_utils as TensorUtils
from models.vla.value_model import VLValueModel

# from models.vla.flow_matching_model_action import VLAFlowMatching
import cv2
import torchvision

class VLValuePredictorModule(pl.LightningModule):
    def __init__(self, algo_config, train_config):
        super(VLValuePredictorModule, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.nets = nn.ModuleDict()

        # Conditioning parsing
        self.cond_drop_color_p = (
            algo_config.training.conditioning_drop_color
            if "conditioning_drop_color" in algo_config.training
            else 0.0
        )
        self.cond_drop_language_p = (
            algo_config.training.conditioning_drop_language
            if "conditioning_drop_language" in algo_config.training
            else 0.0
        )
        # Initialize the diffuser
        self.nets["value"] = VLValueModel(algo_config.model)
        # self.lang_tokens_null = self.get_language_tokens("")[0]  # [1, L]
        self.lang_tokens_null = torch.zeros(1, 30, 768)
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
            "valueRMSE": "val/value_rmse"
        }

    def forward(
        self,
        data_batch,
    ):
        """
        data_batch: dict
        """
        curr_value = self.nets["value"]
        outputs = curr_value(
            data_batch,
        )
        return outputs

    def configure_optimizers(self):
        optim_params = self.algo_config.optimzation
        optimizer = torch.optim.AdamW(
            params=self.nets["value"].parameters(),
            lr=optim_params.learning_rate,
            betas=optim_params.betas,
            eps=optim_params.eps,
            weight_decay=optim_params.weight_decay,
        )
        return {
            "optimizer": optimizer,
        }

    def training_step(self, data_batch, batch_idx):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]

        device = data_batch["gt_state_value"].device
        drop_mask_color = (
            torch.rand(len(data_batch["history_visual_feature_patch"]), device=device)
            < self.cond_drop_color_p
        )
        drop_mask_lang = (
            torch.rand(len(data_batch["language_feature"]), device=device)
            < self.cond_drop_language_p
        )  # [B]

        data_batch["history_visual_feature_patch"][drop_mask_color] = 0.0
        data_batch["language_feature"][drop_mask_lang] = 0.0

        losses = self.nets["value"].compute_losses(data_batch)
        total_loss = 0.0
        for lk in list(losses.keys()):
            losses[lk] = losses[lk] * self.algo_config.training.loss_weights[lk]
            total_loss += losses[lk]
        self.log("train/losses_total_loss", total_loss)
        return {
            "loss": total_loss,
            "all_losses": losses,
        }

    def training_step_end(self, data_batch):
        self.curr_train_step += 1

    def on_validation_start(self):
        self.visualize_batch_idx = None

    def validation_step(self, data_batch, batch_idx):
        data_batch = TensorUtils.join_dimensions(
            data_batch, begin_axis=0, end_axis=2
        )  # [B, T, ...] -> [B*T, ...]
        losses = self.nets["value"].compute_losses(data_batch)
        outputs = self.nets["value"](data_batch)
        value_rmse = self.compute_value_accuracy(data_batch, outputs)
        self.log("val/value_rmse", value_rmse, prog_bar=True, sync_dist=True)
        self.visualize_value(data_batch)
        pred_value_vis = torchvision.utils.make_grid(
            data_batch["pred_value_vis"], nrow=4
        )
        self.logger.log_image(
            "val/pred_value_vis", [pred_value_vis], caption=[f"Batch {batch_idx}"]
        )
        for k, v in losses.items():
            self.log(f"val/losses_{k}", v, prog_bar=True, sync_dist=True)
        return_dict = {"losses": losses}
        return return_dict

    def compute_value_accuracy(self, data_batch, outputs):
        pred_value = outputs["value_predictions"][:, 0]  # [B]
        gt_value = data_batch["gt_state_value"]  # [B,]
        gt_value = gt_value
        gt_value -= 1
        value_rmse = (pred_value - gt_value).pow(2)
        value_rmse = value_rmse.mean().pow(0.5).float()
        data_batch.update({"pred_value": pred_value})
        return value_rmse

    def visualize_value(self, data_batch, **kwargs):
        batch_size = len(data_batch["color"])
        results = []
        for i in range(batch_size):
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            pred_value = data_batch["pred_value"][i].cpu().numpy()
            gt_value = data_batch["gt_state_value"][i].cpu().numpy()
            gt_value = gt_value
            gt_value -= 1

            vis = np.ascontiguousarray(color * 255, dtype=np.uint8)[
                :, :, ::-1
            ].copy()

            cv2.putText(
                vis,
                (f"GT={gt_value:.3f}"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                (f"PRED={pred_value:.3f}"),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            vis = torchvision.transforms.ToTensor()(vis[..., [2, 1, 0]])
            results.append(vis)

        results = torch.stack(results, dim=0)  # [B, C, H, W]   
        data_batch.update({"pred_value_vis": results})

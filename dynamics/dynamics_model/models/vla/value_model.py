import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import pytorch_lightning as pl

import easydict as edict
import utils.tensor_utils as TensorUtils
from collections import OrderedDict
import time
from models.vla.cdit import (
    FinalLayerAction,
    FinalLayerDynamics,
    CDiTDynamics,
    TransformerAction,
)
from models.vla.cdit_rope import (
    RoPECDiTDynamics,
    RoPECDiTAction,
)
from models.vla.guidance import DynamicsGuidance
from typing import Callable
from contextlib import AbstractContextManager
from einops import rearrange

class VLValueModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = 768
        self.value_num_bins = 200

        self.config = config
        self.mode = config.mode  # "vla", "vla+wm"
        # self.dtype = torch.float16 if config.dtype == "torch.float16" else torch.float32
        if self.config.dtype == "torch.float16":
            self.dtype = torch.float16
        elif self.config.dtype == "torch.bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Visoon and language projection
        self.visual_proj = nn.Linear(768, self.feature_dim)
        self.language_proj = nn.Linear(768, self.feature_dim)

        self.value_query = nn.Parameter(
            torch.randn(1, 1, self.feature_dim), requires_grad=True
        )
        vm_model_kwargs = dict(
            transition_dim=self.feature_dim,
            context_dim=self.feature_dim,
            history_dim=self.feature_dim,
            action_dim=self.feature_dim,
            output_dim=self.value_num_bins,
            dim=int(self.config.vm_width_multiplier * self.feature_dim),
            n_layer=self.config.num_vm_layers,
            n_head=self.config.num_vm_heads,
            dtype=self.dtype,
            n_cond_layers=2,
            language_token_size=30,
            causal_attn=False,
        )
        if "vm_type" in self.config:
            self.value_model_type = self.config.vm_type
        else:
            self.value_model_type = "abs_pos_transformer"

        if self.value_model_type == "abs_pos_transformer":
            self.value_model = TransformerAction(**vm_model_kwargs)
        elif self.value_model_type == "rope_transformer":
            self.value_model = RoPECDiTAction(**vm_model_kwargs)
        else:
            raise ValueError(f"Invalid vm model type: {self.value_model_type}")

        self.value_query.to(dtype=self.dtype)
        self.value_model.to(dtype=self.dtype)
        self.vm_out_proj = FinalLayerAction(
            self.value_model.dim, self.value_num_bins
        )  # 200 bins

        print(f"=====> VM model type: {self.value_model_type}")

    def embed_vision_language_features(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
    ):
        """
        Embed the vision and language features
        Args:
            image_features: [B, H, 196, 768] - batch of H image features per sample
            language_features: [B, L, 768] - batch of L language features per sample
            history_raymaps: [B, H, 6, 14, 14] - batch of H raymaps per sample
        Returns:
            image_features: [B, 5*196, D] - batch of image features
            language_features: [B, L, D] - batch of language features
        """
        # Sample 0, 3, 6, 9, 12, 15 from image_featuressa
        sample_indices = [2, 5, 8, 11, 14]
        image_features = image_features[:, sample_indices, :, :]  # [B, 5, 196, 768]
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )  # [B, 5*196, 768]
        image_features = self.visual_proj(image_features)
        language_features = self.language_proj(language_features)
        # vl_feature = torch.cat([image_features, language_features], dim=1)
        return image_features, language_features

    def value_loss(self, image_features: torch.Tensor, language_features: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute the value loss
        Args:
            image_features: [B, 5*196, 768] - batch of 5*196 image features per sample
            language_features: [B, L, 768] - batch of L language features per sample
            values: [B, ] - batch of ground truth values per sample
        Returns:
            value_loss: [1] - value loss
        """
        prefix_image, prefix_language = self.embed_vision_language_features(
            image_features=image_features,
            language_features=language_features,
        )
        prefix = torch.cat([prefix_image, prefix_language], dim=1)  # [B, L+H, D]
        v = self.value_query.expand(prefix.shape[0], -1, -1)
        v, c = self.value_model(
            x=v,
            x_history=prefix,
            context=prefix_image,
            language=prefix_language,
            timestamp=torch.zeros(prefix.shape[0], device=prefix.device, dtype=torch.float32),
        )
        v = v.to(dtype=torch.float32)
        v = self.vm_out_proj(v, c).squeeze(1)  # [B, 200]
        value_loss = F.cross_entropy(v, values.long())
        losses = {"cross_entropy_value_loss": value_loss}
        return losses

    def compute_losses(self, data_batch: dict) -> torch.Tensor:
        image_features = data_batch["history_visual_feature_patch"]  # [B, H, 196, 768]
        language_features = data_batch["language_feature"]  # [B, L, 768]
        values = data_batch["gt_state_value"]  # [B, 1]
        losses = self.value_loss(image_features, language_features, values)
        return losses

    def forward(self, data_batch: dict) -> torch.Tensor:
        image_features = data_batch["history_visual_feature_patch"]  # [B, H, 196, 768]
        language_features = data_batch["language_feature"]  # [B, L, 768]
        prefix_image, prefix_language = self.embed_vision_language_features(
            image_features=image_features,
            language_features=language_features,
        )
        prefix = torch.cat([prefix_image, prefix_language], dim=1)  # [B, L+H, D]
        v = self.value_query.expand(prefix.shape[0], -1, -1)
        v, c = self.value_model(
            x=v,
            x_history=prefix,
            context=prefix_image,
            language=prefix_language,
            timestamp=torch.zeros(
                prefix.shape[0], device=prefix.device, dtype=torch.float32
            ),
        )
        v = v.to(dtype=torch.float32)
        v = self.vm_out_proj(v, c).squeeze(1)  # [B, 200]
        v = v.argmax(dim=-1)  # [B]
        v = (v.float() + 0.5) / self.value_num_bins  # [0, 1]
        v = v - 1.0  # [0, 1] => [-1, 0]
        outputs = {"value_predictions": v[:, None]}
        return outputs

if __name__ == "__main__":
    config = edict.EasyDict(
        mode="vm",
        dtype="torch.float16",
        vm_type="abs_pos_transformer",
        vm_width_multiplier=1,
        num_vm_layers=1,
        num_vm_heads=1,
    )
    value_model = VLValueModel(config)
    value_model.cuda()
    print(value_model)
    data_batch = {
        "history_visual_feature_patch": torch.randn(1, 15, 196, 768).cuda(),
        "language_feature": torch.randn(1, 30, 768).cuda(),
        "gt_state_value": torch.randint(0, 200, (1,)).cuda(),  # 0-199 bin index
    }
    losses = value_model.compute_losses(data_batch)
    print(losses)
    with torch.no_grad():
        outputs = value_model(data_batch)
        print(outputs)

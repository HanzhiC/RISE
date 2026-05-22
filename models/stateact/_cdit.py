# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.stateact.blocks import TimestepEmbedder, CDiTBlock, FinalLayer, modulate


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        output_dim,
        dim=384,
        n_layer=6,
        n_head=4,
        mlp_ratio=4.0,
        patch_size=2,
        input_size=32,
        cond_size=14,
        cond_horizon=1,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio

        self.x_embedder = PatchEmbed(
            input_size, patch_size, transition_dim, dim, bias=True
        )
        self.t_embedder = TimestepEmbedder(dim)

        self.cond_embedder = PatchEmbed(cond_size, patch_size, cond_dim, dim, bias=True)
        self.x_pos_embed = nn.Parameter(
            torch.zeros(self.x_embedder.num_patches, dim),
            requires_grad=True,
        )
        # for context and for predicted frame
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(cond_horizon, self.cond_embedder.num_patches, dim),
            requires_grad=True,
        )  # for context and for predicted frame

        self.blocks = nn.ModuleList(
            [CDiTBlock(dim, n_head, mlp_ratio=mlp_ratio) for _ in range(n_layer)]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.x_pos_embed, std=0.02)
        nn.init.normal_(self.cond_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize cond_embedder like nn.Linear (instead of nn.Conv2d):
        w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.output_dim
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, cond, time):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        cond: (B, T, C, H, W) tensor of context
        t: (N,) tensor of diffusion timesteps
        """

        x = self.x_embedder(x) + self.x_pos_embed
        cond = (
            self.cond_embedder(cond.flatten(0, 1)).unflatten(
                0, (cond.shape[0], cond.shape[1])
            )  # [B, T, L, D]
            + self.cond_pos_embed  # [T, L, D]
        )
        cond = cond.flatten(1, 2)

        t = self.t_embedder(time[..., None])
        for block in self.blocks:
            x = block(x, t, cond)
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


if __name__ == "__main__":
    model = CDiT(
        horizon=15,
        transition_dim=3,
        cond_dim=384,
        dim=384,
        n_layer=6,
        n_head=4,
        mlp_ratio=4.0,
        patch_size=2,
        input_size=32,
        cond_size=14,
        cond_horizon=1,
        output_dim=3,
    )
    x = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 1, 384, 14, 14)
    t = torch.randint(0, 10, (2,))
    print(model(x, cond, t).shape)

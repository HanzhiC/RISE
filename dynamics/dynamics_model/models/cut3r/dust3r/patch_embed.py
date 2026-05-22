# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import dust3r.utils.path_to_croco  # noqa: F401
import torch.nn as nn
import collections
from itertools import repeat
from models.cut3r.dust3r.blocks import PositionGetter

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        torch._assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):
    assert patch_embed_cls in ["PatchEmbedDust3R", "ManyAR_PatchEmbed"]
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, in_chans, enc_embed_dim)
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed(PatchEmbed):
    """Handle images with non-square aspect ratio.
    All images in the same batch have the same aspect ratio.
    true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape

        assert (
            H % self.patch_size[0] == 0
        ), f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert (
            W % self.patch_size[1] == 0
        ), f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (
            B,
            2,
        ), f"true_shape has the wrong shape={true_shape.shape}"

        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T

        is_landscape = torch.ones_like(width, dtype=torch.bool)
        is_portrait = ~is_landscape

        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        x[is_landscape] = (
            self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        )
        x[is_portrait] = (
            self.proj(img[is_portrait].swapaxes(-1, -2))
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
            .float()
        )

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos

import torch
import torch.nn as nn
from models.stateact.attention import CrossAttnBlock, Attention, AttnBlock
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Tuple

import math
import numpy as np
import time
from timm.models.vision_transformer import PatchEmbed
from models.stateact.blocks import (
    TimestepEmbedder,
    CDiTBlock,
    FinalLayer,
    PointNetEncoder,
)
import time


class ActionTransformer(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        output_dim,
        dim=384,
        n_layer=6,
        n_head=4,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        n_cond_layers=1,
        n_cond_tokens=30,
        **kwargs,
    ):
        super().__init__()
        # intrerediate parameters
        dim_feedforward = 4 * dim

        # input embedding stem
        self.input_emb = nn.Linear(transition_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, dim))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = TimestepEmbedder(dim)
        self.cond_obs_emb = nn.Linear(cond_dim, dim)
        self.n_cond_tokens = n_cond_tokens

        # Positional encoding
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_cond_tokens, dim))

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(dim, dim_feedforward),
                nn.Mish(),
                nn.Linear(dim_feedforward, dim),
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )
        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = horizon
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

        # constants
        self.horizon = horizon  # T
        self.transition_dim = transition_dim  # T_cond
        self.cond_dim = cond_dim
        self.output_dim = output_dim

        # init
        self.apply(self._init_weights)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, x, cond, time):
        """
        x : [ batch x horizon x transition ]
        cond: [ batch x n_cond_tokens x cond_dim ]
        time: [ batch ]
        """

        input_emb = self.input_emb(x)  # [B, 1, dim]
        time_emb = self.time_emb(time.unsqueeze(1)).unsqueeze(1)  # [batch, 1, dim]

        # encoder
        cond_embeddings = time_emb  # [B, 1, dim]
        cond_obs_emb = self.cond_obs_emb(cond)  # [B, L, dim]

        cond_embeddings = torch.cat(
            [time_emb, cond_obs_emb], dim=1
        )  # (B, n_cond + 1, dim)
        position_embeddings = torch.cat([self.time_pos_emb, self.cond_pos_emb], dim=1)
        memory = self.drop(cond_embeddings + position_embeddings)  # (B, 1, dim)
        memory = self.encoder(memory)  # (B, 1, dim)
        memory = memory
        # (B, 1, dim)

        # decoder
        token_embeddings = input_emb
        token_length = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :token_length
        ]  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
        )
        # (B,T,n_emb)

        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, TimestepEmbedder):
            torch.nn.init.normal_(module.mlp[0].weight, std=0.02)
            torch.nn.init.normal_(module.mlp[2].weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ActionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.time_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("time_pos_emb")
        no_decay.add("cond_pos_emb")
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


class ActionCDiT(nn.Module):
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
        n_query_tokens=1024,
        n_cond_tokens=30,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_query_tokens = n_query_tokens
        self.n_cond_tokens = n_cond_tokens
        self.mlp_ratio = mlp_ratio
        self.x_embedder = nn.Linear(transition_dim, dim)
        self.cond_embedder = nn.Linear(cond_dim, dim)
        self.t_embedder = TimestepEmbedder(dim)
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(self.n_cond_tokens + self.n_query_tokens, n_query_tokens, dim),
        #     requires_grad=True,
        # )  # for context and for predicted frame
        self.query_pos_embed = nn.Parameter(
            torch.zeros(self.n_query_tokens, dim),
            requires_grad=True,
        )
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(self.n_cond_tokens, dim),
            requires_grad=True,
        )
        self.blocks = nn.ModuleList(
            [CDiTBlock(dim, n_head, mlp_ratio=mlp_ratio) for _ in range(n_layer)]
        )
        self.final_layer = FinalLayer(dim, 1, self.output_dim)
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
        nn.init.normal_(self.query_pos_embed, std=0.02)
        nn.init.normal_(self.cond_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize cond embedding:
        nn.init.normal_(self.cond_embedder.weight, std=0.02)
        nn.init.constant_(self.cond_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize final layer:
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond, time):
        """
        Forward pass of DiT.
        x: (B, N, C) tensor of spatial inputs (images or latent representations of images)
        cond: (B, N, T, C) tensor of context inputs
        t: (B,) tensor of diffusion timesteps
        """
        x = self.x_embedder(x) + self.query_pos_embed[None]  # [B, N, D]
        cond = self.cond_embedder(cond) + self.cond_pos_embed[None]  # [B, N, D]
        time = self.t_embedder(time[..., None])

        for block in self.blocks:
            x = block(x, time, cond)
        x = self.final_layer(x, time)
        return x


class StateCDiTV1(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        action_dim,
        output_dim,
        dim=768,
        n_layer=12,
        n_head=12,
        n_state_layer=1,
        mlp_ratio=4.0,
        patch_factor=2,
        input_size=32,
        cond_size=14,
        cond_horizon=1,
        state_size=226,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_state_layer = n_state_layer
        self.mlp_ratio = mlp_ratio
        self.cond_horizon = cond_horizon
        self.cond_size = cond_size
        self.state_size = state_size
        self.xs_cross_attn_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    self.dim, self.dim, self.n_head, mlp_ratio=self.mlp_ratio
                )
                for _ in range(self.n_state_layer)
            ]
        )
        self.x_embedder = PatchEmbed(
            input_size, 
            patch_factor, 
            transition_dim, 
            dim, 
            bias=True
        )
        self.x_history_embedder = PatchEmbed(
            input_size, 
            patch_factor,  # history frames are at lower resolution
            output_dim, 
            dim, 
            bias=True, 
        )
        self.s_embedder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(action_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.cond_embedder = PatchEmbed(
            cond_size,
            patch_factor,
            cond_dim,
            dim,
            norm_layer=nn.LayerNorm,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(dim)
        self.rel_t_embedder = TimestepEmbedder(dim)
        self.a_embedder = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        self.x_pos_embed = nn.Parameter(
            torch.zeros(self.x_embedder.num_patches, dim),
            requires_grad=True,
        )

        self.x_history_pos_embed = nn.Parameter(    
            torch.zeros(self.x_history_embedder.num_patches, dim),
            requires_grad=True,
        )
        # for context and for predicted frame
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(cond_horizon, self.cond_embedder.num_patches, dim),
            requires_grad=True,
        )  # for context and for predicted frame
        self.state_pos_embed = nn.Parameter(
            torch.zeros(state_size, dim),
            requires_grad=True,
        )  # for state

        self.blocks = nn.ModuleList(
            [CDiTBlock(dim, n_head, mlp_ratio=mlp_ratio) for _ in range(n_layer)]
        )
        self.final_layer_patch_size = patch_factor
        self.final_layer = FinalLayer(dim, self.final_layer_patch_size, self.output_dim)
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
        nn.init.normal_(self.x_history_pos_embed, std=0.02)
        nn.init.normal_(self.cond_pos_embed, std=0.02)
        nn.init.normal_(self.state_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize history patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_history_embedder.proj.weight.data    
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_history_embedder.proj.bias, 0)

        # Initialize cond_embedder like nn.Linear (instead of nn.Conv2d):
        w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        # Initialize s_embedder:
        nn.init.normal_(self.s_embedder[1].weight, std=0.02)
        nn.init.constant_(self.s_embedder[1].bias, 0)
        nn.init.normal_(self.s_embedder[3].weight, std=0.02)
        nn.init.constant_(self.s_embedder[3].bias, 0)

        # Initialize a_embedder:
        nn.init.normal_(self.a_embedder[1].weight, std=0.02)
        nn.init.constant_(self.a_embedder[1].bias, 0)
        nn.init.normal_(self.a_embedder[3].weight, std=0.02)
        nn.init.constant_(self.a_embedder[3].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize rel_t_embedder:
        nn.init.normal_(self.rel_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.rel_t_embedder.mlp[2].weight, std=0.02)

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
        p = self.final_layer_patch_size  # self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, x_history, cond, timestamp, state=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        x: (B, C, H, W) tensor of history spatial inputs (images or latent representations of images)
        cond: (B, T, C, H, W) tensor of context
        action: (B, D) tensor of action labels
        state: (B, N, D) tensor of state features
        t: (B,) tensor of diffusion timesteps
        rel_time: (B,) tensor of relative diffusion timesteps
        """
        x = self.x_embedder(x) + self.x_pos_embed[None]  # [B, N, D]
        x_history = self.x_history_embedder(x_history) + self.x_history_pos_embed[None]  # [B, N, D]
        cond = (
            self.cond_embedder(cond.flatten(0, 1)).unflatten(
                0, (cond.shape[0], cond.shape[1])
            )  # [B, T, L, D]
            + self.cond_pos_embed[None]  # [T, L, D]
        )
        cond = cond.flatten(1, 2)  # [B, TL, D]
        cond = torch.cat([cond, x_history], dim=1)  # [B, TL+N, D]  

        if state is not None:
            # assert state.shape[1] == self.state_pos_embed.shape[0]
            state = (
                self.s_embedder(state) + self.state_pos_embed[None, : state.shape[1]]
            )
            cond = torch.cat([cond, state], dim=1)
            for cross_attn_block in self.xs_cross_attn_blocks:
                x = cross_attn_block(x, state)

        c = self.t_embedder(timestamp[..., None])

        for block_idx, block in enumerate(self.blocks):
            start_time = time.time()    
            x = block(x, c, cond)
            end_time = time.time()
            print(
                f"Time taken for block {block_idx} in layer {block_idx}: {end_time - start_time} seconds"
            )
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


# class StateCDiTV2(nn.Module):
#     """
#     Diffusion model with a Transformer backbone.
#     """

#     def __init__(
#         self,
#         horizon,
#         transition_dim,
#         cond_dim,
#         action_dim,
#         output_dim,
#         dim=768,
#         n_layer=12,
#         n_head=12,
#         mlp_ratio=4.0,
#         patch_factor=1,
#         input_size=32,
#         cond_size=14,
#         cond_horizon=1,
#     ):
#         super().__init__()

#         self.horizon = horizon
#         self.transition_dim = transition_dim
#         self.cond_dim = cond_dim
#         self.output_dim = output_dim
#         self.dim = dim
#         self.n_layer = n_layer
#         self.n_head = n_head
#         self.mlp_ratio = mlp_ratio
#         self.input_size = input_size
#         self.x_embedder = PointNetEncoder(
#             global_feat=False,
#             feature_transform=False,
#             feature_dim=dim,
#             channel=transition_dim,
#         )
#         self.cond_embedder = PatchEmbed(
#             cond_size,
#             patch_factor,
#             cond_dim,
#             dim,
#             norm_layer=nn.LayerNorm,
#             bias=True,
#         )

#         self.t_embedder = TimestepEmbedder(dim)
#         self.rel_t_embedder = TimestepEmbedder(dim)
#         self.a_embedder = nn.Sequential(
#             nn.Linear(action_dim, dim, bias=True),
#             nn.SiLU(),
#             nn.Linear(dim, dim, bias=True),
#         )

#         # self.x_pos_embed = nn.Parameter(
#         #     torch.zeros(input_size * input_size, dim),
#         #     requires_grad=True,
#         # )
#         # for context and for predicted frame
#         self.cond_pos_embed = nn.Parameter(
#             torch.zeros(cond_horizon, self.cond_embedder.num_patches, dim),
#             requires_grad=True,
#         )  # for context and for predicted frame

#         self.blocks = nn.ModuleList(
#             [CDiTBlock(dim, n_head, mlp_ratio=mlp_ratio) for _ in range(n_layer)]
#         )
#         self.final_layer = FinalLayer(dim, 1, self.output_dim)
#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize transformer layers:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#         self.apply(_basic_init)

#         # Initialize (and freeze) pos_embed by sin-cos embedding:
#         # nn.init.normal_(self.x_pos_embed, std=0.02)
#         nn.init.normal_(self.cond_pos_embed, std=0.02)
#         nn.init.normal_(self.x_embedder.conv1.weight, std=0.02)
#         nn.init.constant_(self.x_embedder.conv1.bias, 0)
#         nn.init.normal_(self.x_embedder.conv2.weight, std=0.02)
#         nn.init.constant_(self.x_embedder.conv2.bias, 0)
#         nn.init.normal_(self.x_embedder.conv3.weight, std=0.02)
#         nn.init.constant_(self.x_embedder.conv3.bias, 0)

#         # Initialize cond_embedder like nn.Linear (instead of nn.Conv2d):
#         w = self.cond_embedder.proj.weight.data
#         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         nn.init.constant_(self.cond_embedder.proj.bias, 0)

#         # Initialize timestep embedding MLP:
#         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

#         # Initialize rel_t_embedder:
#         nn.init.normal_(self.rel_t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.rel_t_embedder.mlp[2].weight, std=0.02)

#         # Zero-out adaLN modulation layers in DiT blocks:
#         for block in self.blocks:
#             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers:
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)

#     def unpatchify(self, x):
#         """
#         x: (N, T, patch_size**2 * C)
#         imgs: (N, C, H, W)
#         """
#         # c = self.output_dim
#         # p = self.x_embedder.patch_size[0]
#         # h = w = int(x.shape[1] ** 0.5)
#         # assert h * w == x.shape[1]

#         # x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
#         # x = torch.einsum("nhwpqc->nchpwq", x)
#         # imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
#         x = rearrange(x, "b (h w) c -> b c h w", h=self.input_size, w=self.input_size)
#         return x

#     def forward(self, x, cond, time, rel_time=None, action=None):
#         """
#         Forward pass of DiT.
#         x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
#         cond: (B, T, C, H, W) tensor of context
#         action: (B, D) tensor of action features

#         t: (B,) tensor of diffusion timesteps
#         rel_time: (B,) tensor of relative diffusion timesteps
#         """
#         x = x.flatten(2).transpose(
#             1, 2
#         )  # [B, C, H, W] => [B, C, H * W] => [B, H * W, C]
#         x = self.x_embedder(x, transposed_input=True)[0]
#         cond = (
#             self.cond_embedder(cond.flatten(0, 1)).unflatten(
#                 0, (cond.shape[0], cond.shape[1])
#             )  # [B, T, L, D]
#             + self.cond_pos_embed  # [T, L, D]
#         )
#         cond = cond.flatten(1, 2)  # [B, T, L, D]
#         if action is not None:
#             a = self.a_embedder(action)
#         else:
#             a = None
#         c = self.t_embedder(time[..., None])

#         if rel_time is not None:
#             rel_t = self.rel_t_embedder(rel_time[..., None])
#             c = c + rel_t

#         if a is not None:
#             c = c + a

#         for block in self.blocks:
#             x = block(x, c, cond)
#         x = self.final_layer(x, c)
#         x = self.unpatchify(x)
#         return x


class ValueTransformer(nn.Module):

    def __init__(
        self,
        transition_dim,
        cond_dim,
        output_dim=1,
        dim=384,
        n_layer=6,
        n_head=4,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        n_cond_layers=1,
        n_cond_tokens=30,
        **kwargs,
    ):
        super().__init__()
        # intrerediate parameters
        dim_feedforward = 4 * dim

        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.cond_obs_emb = nn.Linear(cond_dim, dim)
        self.n_cond_tokens = n_cond_tokens

        # Positional encoding
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_cond_tokens, dim))

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(dim, dim_feedforward),
                nn.Mish(),
                nn.Linear(dim_feedforward, dim),
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )

        # A leanarble token
        self.value_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.value_pos_emb = nn.Parameter(torch.zeros(1, 1, dim))

        # decoder head
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, output_dim), nn.Sigmoid())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        self.apply(self._init_weights)
        nn.init.normal_(self.value_token, std=0.02)
        nn.init.normal_(self.value_pos_emb, std=0.02)
        nn.init.normal_(self.cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.cond_obs_emb.bias, 0)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

    def forward(self, cond):
        """
        cond: [B, N, D]
        """
        cond_obs_emb = self.cond_obs_emb(cond)
        memory = self.drop(cond_obs_emb + self.cond_pos_emb)
        memory = self.encoder(memory)
        query = self.value_token.repeat(cond.shape[0], 1, 1)  # [B, 1, D]
        query = self.drop(query + self.value_pos_emb)  # [B, 1, D]
        query = self.decoder(query, memory)
        query = self.ln_f(query)
        query = self.head(query).squeeze(-1).squeeze(-1) # [B, ]
        return query  # [B,]


if __name__ == "__main__":
    # model = ActionCDiT(
    #     horizon=15,
    #     transition_dim=6,
    #     cond_dim=384,
    #     dim=384,
    #     n_layer=6,
    #     n_head=4,
    #     n_query_tokens=15,
    #     n_cond_tokens=30,
    #     output_dim=6,
    # ).cuda()
    # x = torch.randn(2, 15, 6).cuda()
    # cond = torch.randn(2, 30, 384).cuda()
    # t = torch.randint(0, 10, (2,)).cuda()
    # with torch.no_grad():
    #     for _ in range(10):
    #         start_time = time.time()
    #         out = model(x, cond, t)
    #         end_time = time.time()
    #         print(f"ActionCDiT Time taken: {end_time - start_time} seconds")
    # print("ActionCDiT output shape: ", out.shape)

    # # Action Transformer
    # model = ActionTransformer(
    #     horizon=15,
    #     transition_dim=6,
    #     cond_dim=384,
    #     dim=384,
    #     n_layer=6,
    #     n_head=4,
    #     n_cond_tokens=30,
    #     output_dim=6,
    # ).cuda()
    # x = torch.randn(2, 15, 6).cuda()
    # cond = torch.randn(2, 30, 384).cuda()
    # t = torch.randint(0, 10, (2,)).cuda()
    # with torch.no_grad():
    #     for _ in range(10):
    #         start_time = time.time()
    #         out = model(x, cond, t)
    #         end_time = time.time()
    #         print(f"ActionTransformer Time taken: {end_time - start_time} seconds")
    # print("ActionTransformer output shape: ", out.shape)

    # State CDiT
    model = StateCDiTV1(
        horizon=15,
        transition_dim=3,
        cond_dim=768,
        action_dim=45,
        n_layer=12,
        n_head=12,
        output_dim=3,
        input_size=32,
        patch_factor=2,
    ).cuda()
    x = torch.randn(2, 3, 32, 32).cuda()
    cond = torch.randn(2, 1, 768, 14, 14).cuda()
    t = torch.randint(0, 10, (2,), dtype=torch.long).cuda()
    rel_time = torch.randint(0, 10, (2,), dtype=torch.long).cuda()
    action = torch.randn(2, 45).cuda()

    # Force CPU usage to avoid CUDA kernel issues
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA temporarily

    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            out = model(x, cond, t, rel_time, action)
            end_time = time.time()
            print(f"StateCDiT Time taken: {end_time - start_time} seconds")
    print("StateCDiT output shape: ", out.shape)

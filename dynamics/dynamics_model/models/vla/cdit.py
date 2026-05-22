import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed

# from models.stateact.blocks import TimestepEmbedder, CDiTBlock, FinalLayer
import time
from typing import Tuple

from models.vla.blocks import (
    modulate,
    TimestepEmbedder,
    AttentionPooling,
    CDiTBlockAbsPos,
)


class FinalLayerDynamics(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, int(patch_size * patch_size * out_channels), bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayerAction(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def initialize_weights(self):
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        x = self.linear(x)
        return x


class CDiTDynamics(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        transition_dim,
        history_dim,
        context_dim,
        action_dim,
        output_dim,
        dim=768,
        n_layer=12,
        n_head=12,
        mlp_ratio=4.0,
        patch_factor=2,
        input_size=32,
        context_length=4096,
        action_chunk_size=15,
        languge_token_size=30,
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.context_length = context_length
        self.patch_factor = patch_factor
        self.action_chunk_size = action_chunk_size
        self.languge_token_size = languge_token_size
        self.x_embedder = PatchEmbed(
            input_size, patch_factor, transition_dim, dim, bias=True
        )
        self.x_history_embedder = PatchEmbed(
            input_size,
            patch_factor,
            history_dim,
            dim,
            norm_layer=nn.LayerNorm,
            bias=True,
        )
        self.mask_action_emb = nn.Parameter(
            torch.zeros(1, self.action_chunk_size, dim), requires_grad=True
        )
        self.t_embedder = TimestepEmbedder(dim)
        self.ctx_embedder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.a_embedder = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.ta_embedder = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        self.a_attn_pool = nn.Sequential(
            AttentionPooling(dim, n_head),
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
        )

        self.layers = nn.ModuleList(
            [
                CDiTBlockAbsPos(
                    dim,
                    n_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_layer)
            ]
        )

        # Initialize positional encodings
        self.x_pos_embed = nn.Parameter(
            torch.zeros(self.x_embedder.num_patches, dim),
            requires_grad=True,
        )
        self.x_history_pos_embed = nn.Parameter(
            torch.zeros(self.x_history_embedder.num_patches, dim),
            requires_grad=True,
        )
        self.ctx_pos_embed = nn.Parameter(
            torch.zeros(context_length, dim),
            requires_grad=True,
        )
        self.action_pos_embed = nn.Parameter(
            torch.zeros(action_chunk_size, dim),
            requires_grad=True,
        )

        # self.final_layer_patch_size = patch_factor
        # self.final_layer = FinalLayer(dim, self.final_layer_patch_size, self.output_dim)

        # Calculate patch grid dimensions for 2D positional encoding
        self.patch_grid_size = input_size // patch_factor
        self.initialize_weights()
        self.dtype = dtype

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional encodings
        nn.init.normal_(self.x_pos_embed, std=0.02)
        nn.init.normal_(self.x_history_pos_embed, std=0.02)
        nn.init.normal_(self.ctx_pos_embed, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize history patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_history_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_history_embedder.proj.bias, 0)

        # Initialize ctx_embedder like nn.Linear (instead of nn.Conv2d):
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

        # Initialize a_embedder:
        nn.init.normal_(self.a_embedder[1].weight, std=0.02)
        nn.init.constant_(self.a_embedder[1].bias, 0)
        nn.init.normal_(self.a_embedder[3].weight, std=0.02)
        nn.init.constant_(self.a_embedder[3].bias, 0)

        # Initialize ta_embedder:
        nn.init.normal_(self.ta_embedder[1].weight, std=0.02)
        nn.init.constant_(self.ta_embedder[1].bias, 0)
        nn.init.normal_(self.ta_embedder[3].weight, std=0.02)
        nn.init.constant_(self.ta_embedder[3].bias, 0)

        # Initialize ctx_embedder:
        nn.init.normal_(self.ctx_embedder[1].weight, std=0.02)
        nn.init.constant_(self.ctx_embedder[1].bias, 0)
        nn.init.normal_(self.ctx_embedder[3].weight, std=0.02)
        nn.init.constant_(self.ctx_embedder[3].bias, 0)

        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

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

    def embed_prefix_and_suffix(
        self, x, x_history, context, timestamp, action, drop_action_mask
    ):
        """
        Embed prefix and suffix of the inputs.
        Args:
            x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
            x_history: (B, C, H, W) tensor of history spatial inputs (images or latent representations of images)
            context: (B, L, D) tensor of context features
            timestamp: (B,) tensor of diffusion timesteps
            action: (B, L_action, D) tensor of action
            drop_action_mask: (B,) tensor of mask for dropping action tokens; True means drop
        Returns:
            x: (B, N, D) tensor of spatial inputs (images or latent representations of images)
            x_cond: (B, L + L_action, D) tensor of context features (concatenated with action)
            c: (B, D) tensor of diffusion timesteps and global pooling of action
        """
        batch_size = x.shape[0]
        x = x.to(dtype=self.dtype)
        x_history = x_history.to(dtype=self.dtype)
        timestamp = timestamp.to(dtype=self.dtype)

        # Convert context and action to dtype only if they're not None
        context = context.to(dtype=self.dtype)
        action = action.to(dtype=self.dtype)
        if drop_action_mask is not None:
            drop_action_mask = drop_action_mask.to(dtype=self.dtype)

        # Embed inputs (without adding positional encodings - RoPE handles positions separately)
        x = (
            self.x_embedder(x) + self.x_pos_embed[None]
        )  # [B, N, D] where N = x_grid_h * x_grid_w
        x_history = (
            self.x_history_embedder(x_history) + self.x_history_pos_embed[None]
        )  # [B, N, D]
        context = (
            self.ctx_embedder(context) + self.ctx_pos_embed[None, : context.shape[1]]
        )  # [B, L, D]

        # Process action
        action = self.a_embedder(action)  # [B, L_action, D]
        if drop_action_mask is not None:
            # Use masked action embedding - expand to batch size
            mask_action_emb = self.mask_action_emb.expand(
                batch_size, -1, -1
            )  # [B, L_action, D]
            action = (
                action * (1 - drop_action_mask[:, None, None])
                + mask_action_emb * drop_action_mask[:, None, None]
            )
        action = action + self.action_pos_embed[None]  # [B, L_action, D]

        # Build the final context
        x_cond = torch.cat([context, x_history, action], dim=1)  # [B, L + L_action, D]

        # Combine timestep and action embeddings
        t = self.t_embedder(timestamp[..., None])  # [B, D]
        c = t
        # a = self.a_attn_pool(action)  # [B, D]
        # c = torch.cat([t, a], dim=-1)  # [B, 2*D]
        # c = self.ta_embedder(c)  # [B, D]

        return x, x_cond, c

    def forward(self, x, x_history, context, timestamp, action, drop_action_mask=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        x_history: (B, C, H, W) tensor of history spatial inputs (images or latent representations of images)
        context: (B, L, D) tensor of context features
        timestamp: (B,) tensor of diffusion timesteps
        action: (B, L, D) tensor of action
        """
        x, x_cond, c = self.embed_prefix_and_suffix(
            x, x_history, context, timestamp, action, drop_action_mask
        )
        # Pass positions to blocks for RoPE
        for _, layer in enumerate(self.layers):
            x = layer(
                x=x,
                c=c,
                x_cond=x_cond,
            )
        return x, c


class TransformerAction(nn.Module):

    def __init__(
        self,
        transition_dim,
        history_dim,
        context_dim,
        action_dim,
        output_dim,
        dim=768,
        n_layer=12,
        n_head=12,
        mlp_ratio=4.0,
        patch_factor=2,
        input_size=32,
        context_length=4096,
        action_chunk_size=15,
        languge_token_size=30,
        dtype=torch.float32,
        n_cond_layers=4,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        **kwargs,
    ):
        super().__init__()
        # intrerediate parameters
        dim_feedforward = 4 * dim
        # constants
        self.action_chunk_size = action_chunk_size  # T
        self.transition_dim = transition_dim  # T_transition
        self.history_dim = history_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.dtype = dtype

        # input embedding stem
        self.input_emb = nn.Linear(transition_dim, dim)
        self.history_emb = nn.Linear(history_dim, dim)
        self.context_emb = nn.Linear(context_dim, dim)
        self.language_emb = nn.Linear(context_dim, dim)
        self.drop = nn.Dropout(p_drop_emb)

        # Advantage embedding
        self.advantage_emb = nn.Parameter(
            torch.zeros(3, action_chunk_size, dim), requires_grad=True
        )  # [uncond, negative, positive]

        # Action embedding
        self.pos_emb = nn.Parameter(
            torch.zeros(1, action_chunk_size, dim), requires_grad=True
        )

        # cond encoder
        self.time_emb = TimestepEmbedder(dim)
        self.cond_obs_emb = nn.Linear(dim, dim)

        # Positional encoding
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.cond_pos_emb = nn.Parameter(
            torch.zeros(1, context_length, dim), requires_grad=True
        )
        self.adv_pos_emb = nn.Parameter(
            torch.zeros(1, action_chunk_size, dim), requires_grad=True
        )

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
            sz = self.action_chunk_size
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
        # self.final_layer = FinalLayerAction(dim, 1, output_dim)

        # init
        self.apply(self._init_weights)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        x,
        x_history,
        context,
        timestamp,
        language,
        advantage=None,
        drop_language_mask=None,
    ):
        """
        x : [ batch x horizon x transition ]
        cond: [ batch x n_cond_tokens x cond_dim ]
        time: [ batch ]
        """
        x = x.to(dtype=self.dtype)
        # cond = cond.to(dtype=self.dtype)
        x_history = x_history.to(dtype=self.dtype)
        context = context.to(dtype=self.dtype)
        timestamp = timestamp.to(dtype=self.dtype)
        language = language.to(dtype=self.dtype)
        if drop_language_mask is not None:
            drop_language_mask = drop_language_mask.to(dtype=self.dtype)

        # Project the history
        language = self.language_emb(language)
        x_history = self.history_emb(x_history)
        context = self.context_emb(context)

        # Build the condition
        cond = torch.cat([language, x_history, context], dim=1)
        obs_emb = self.cond_obs_emb(cond)  # [B, L, dim]
        input_emb = self.input_emb(x)  # [B, 1, dim]
        time_emb = self.time_emb(timestamp.unsqueeze(1)).unsqueeze(1)  # [batch, 1, dim]

        # Retrieve advantage embedding: 1=positive, 0=negative, -1=uncond
        # Map -1->0, 0->1, 1->2 and index into [uncond, negative, positive]
        if advantage is not None:
            adv_index = (advantage + 1).long().clamp(0, 2)  # [B]
            adv_emb = self.advantage_emb[adv_index]  # [B, action_chunk_size, dim]
        else:
            adv_emb = self.advantage_emb[:1].expand(x.shape[0], -1, -1)  # unconditional

        # encoder
        cond_embeddings = torch.cat(
            [time_emb, adv_emb, obs_emb], dim=1
        )  # (B, n_cond + 1, dim)
        position_embeddings = torch.cat(
            [
                self.time_pos_emb,  # [1, 1, dim]
                self.adv_pos_emb,  # [1, action_chunk_size, dim]
                self.cond_pos_emb[:, : obs_emb.shape[1]],  # [1, L, dim]
            ],
            dim=1,
        )
        memory = self.drop(cond_embeddings + position_embeddings)  # (B, 1, dim)
        memory = self.encoder(memory)  # (B, 1, dim)
        # (B, 1, dim)

        # decoder
        token_embeddings = input_emb
        position_embeddings = self.pos_emb[
            :, : token_embeddings.shape[1]
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
        # x = self.final_layer(x, time_emb.squeeze(1))
        return x, time_emb.squeeze(1)

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
        elif isinstance(module, TransformerAction):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.time_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    # def get_optim_groups(self, weight_decay: float = 1e-3):
    #     """
    #     This long function is unfortunately doing something very simple and is being very defensive:
    #     We are separating out all parameters of the model into two buckets: those that will experience
    #     weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    #     We are then returning the PyTorch optimizer object.
    #     """

    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

    #             if pn.endswith("bias"):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.startswith("bias"):
    #                 # MultiheadAttention bias starts with "bias"
    #                 no_decay.add(fpn)
    #             elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add("time_pos_emb")
    #     no_decay.add("cond_pos_emb")
    #     no_decay.add("pos_emb")

    #     # validate that we considered every parameter
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert (
    #         len(inter_params) == 0
    #     ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    #     assert (
    #         len(param_dict.keys() - union_params) == 0
    #     ), "parameters %s were not separated into either decay/no_decay set!" % (
    #         str(param_dict.keys() - union_params),
    #     )

    #     # create the pytorch optimizer object
    #     optim_groups = [
    #         {
    #             "params": [param_dict[pn] for pn in sorted(list(decay))],
    #             "weight_decay": weight_decay,
    #         },
    #         {
    #             "params": [param_dict[pn] for pn in sorted(list(no_decay))],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     return optim_groups

    # def configure_optimizers(
    #     self,
    #     learning_rate: float = 1e-4,
    #     weight_decay: float = 1e-3,
    #     betas: Tuple[float, float] = (0.9, 0.95),
    # ):
    #     optim_groups = self.get_optim_groups(weight_decay=weight_decay)
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    #     return optimizer


# if __name__ == "__main__":
#     x = torch.randn(1, 3, 32, 32).cuda()
#     x_history = torch.randn(1, 3, 32, 32).cuda()
#     context = torch.randn(1, 1028, 768).cuda()
#     timestamp = torch.randn(1).cuda()
#     action = torch.randn(1, 15, 10).cuda()
#     model = RoPECDiT(
#         transition_dim=3, cond_dim=768, action_dim=10, output_dim=3, context_length=1024
#     ).cuda()
#     output = model(x, x_history, context, timestamp, action)
#     print(output.shape)
#     # print(output)

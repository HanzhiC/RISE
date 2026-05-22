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
    CDiTBlock,
    AttentionPooling,
    SkipEmbedder,
)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
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


class RoPECDiTDynamics(nn.Module):
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
        context_length=2048,
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
        self.a_embedder = SkipEmbedder(action_dim, dim, dim)
        self.ta_embedder = SkipEmbedder(dim * 2, dim, dim)
        if self.context_dim > 0:
            self.ctx_embedder = SkipEmbedder(context_dim, dim, dim)
        else:
            self.ctx_embedder = None

        self.a_attn_pool = AttentionPooling(dim, n_head)
        self.layers = nn.ModuleList(
            [
                CDiTBlock(
                    dim,
                    n_head,
                    mlp_ratio=mlp_ratio,
                    use_rope=True,
                    rope_max_wavelength=10000,
                )
                for _ in range(n_layer)
            ]
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

        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_2d_positions(self, batch_size, device, grid_h=None, grid_w=None):
        """
        Generate 2D positional encodings for RoPE.

        Args:
            batch_size: Batch size B
            device: Device to create tensors on
            grid_h: Optional height of patch grid (defaults to self.patch_grid_size)
            grid_w: Optional width of patch grid (defaults to self.patch_grid_size)

        Returns:
            positions: [B, 2, H*W] tensor where:
                - positions[:, 0, :] are y-coordinates (row indices)
                - positions[:, 1, :] are x-coordinates (column indices)
        """
        if grid_h is None:
            grid_h = self.patch_grid_size
        if grid_w is None:
            grid_w = self.patch_grid_size

        # Create coordinate grids
        y_coords = torch.arange(grid_h, device=device, dtype=torch.long)  # [H]
        x_coords = torch.arange(grid_w, device=device, dtype=torch.long)  # [W]

        # Create meshgrid: y_grid [H, W], x_grid [H, W]
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Flatten to [H*W]
        y_flat = y_grid.flatten()  # [H*W]
        x_flat = x_grid.flatten()  # [H*W]

        # Expand to batch: [B, H*W]
        y_positions = y_flat.unsqueeze(0).expand(batch_size, -1)  # [B, H*W]
        x_positions = x_flat.unsqueeze(0).expand(batch_size, -1)  # [B, H*W]

        # Stack to [B, 2, H*W]
        positions = torch.stack([y_positions, x_positions], dim=1)  # [B, 2, H*W]

        return positions

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
            x_history: (B, N, D) tensor of history spatial inputs (images or latent representations of images)
            context: (B, L, D) tensor of context features (concatenated with action)
            c: (B, D) tensor of diffusion timesteps and global pooling of action
            pos_x: (B, 2, H*W) tensor of positions for spatial inputs (images)
            pos_x_history: (B, 2, H*W) tensor of positions for history spatial inputs (images)
            pos_context: (B, L) tensor of positions for context features
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

        # Calculate patch grid dimensions (after patch embedding, spatial dims are divided by patch_factor)
        x_grid_h = x.shape[2] // self.patch_factor
        x_grid_w = x.shape[3] // self.patch_factor
        x_history_grid_h = x_history.shape[2] // self.patch_factor
        x_history_grid_w = x_history.shape[3] // self.patch_factor
        # x_history_max_pos = max(x_history_grid_h, x_history_grid_w)

        # Generate 2D positions for spatial inputs (images)
        pos_x = self.get_2d_positions(
            x.shape[0], x.device, grid_h=x_grid_h, grid_w=x_grid_w
        )  # [B, 2, H*W]
        pos_x_history = self.get_2d_positions(
            x_history.shape[0],
            x_history.device,
            grid_h=x_history_grid_h,
            grid_w=x_history_grid_w,
        )  # [B, 2, H*W]

        # Embed inputs (without adding positional encodings - RoPE handles positions separately)
        x = self.x_embedder(x)  # [B, N, D] where N = x_grid_h * x_grid_w
        x_history = self.x_history_embedder(x_history)  # [B, N, D]
        t = self.t_embedder(timestamp[..., None])  # [B, D]

        # Process context (language + visual features)
        if self.ctx_embedder is not None:
            context = self.ctx_embedder(context)  # [B, L, D]
            pos_context = torch.arange(
                context.shape[1], device=context.device, dtype=torch.long
            )[None, :].expand(
                context.shape[0], -1
            )  # [B, L] - sequence indices starting from 0
            L_context = context.shape[1]
        else:
            context = None
            pos_context = None
            L_context = 0

        # Process action
        action_embedded = self.a_embedder(action)  # [B, L_action, D]
        if drop_action_mask is not None:
            # Use masked action embedding - expand to batch size
            mask_action_emb = self.mask_action_emb.expand(
                batch_size, -1, -1
            )  # [B, L_action, D]
            action_embedded = (
                action_embedded * (1 - drop_action_mask[:, None, None])
                + mask_action_emb * drop_action_mask[:, None, None]
            )

        L_action = action_embedded.shape[1]

        # Create positions for action tokens
        # Offset action positions to continue from where context ends (continuous indexing)
        pos_action = torch.arange(L_action, device=x.device, dtype=torch.long)[
            None, :
        ].expand(
            batch_size, -1
        )  # [B, L_action] - sequence indices starting from 0
        # Offset by context_end_pos to continue the position sequence
        pos_action = pos_action + L_context  # [B, L_action]

        # Pool the action BEFORE concatenating to get global action representation
        a = self.a_attn_pool(action_embedded)  # [B, D]

        # Concatenate context and action for the prefix
        if context is not None:
            context = torch.cat(
                [context, action_embedded], dim=1
            )  # [B, L + L_action, D]
            pos_context = torch.cat(
                [pos_context, pos_action], dim=-1
            )  # [B, L + L_action]
        else:
            context = action_embedded  # [B, L_action, D]
            pos_context = pos_action  # [B, L_action]

        # Combine timestep and action embeddings
        c = torch.cat([t, a], dim=-1)  # [B, 2*D]
        c = self.ta_embedder(c)  # [B, D]

        return x, x_history, context, c, pos_x, pos_x_history, pos_context

    def forward(self, x, x_history, context, timestamp, action, drop_action_mask=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        x_history: (B, C, H, W) tensor of history spatial inputs (images or latent representations of images)
        context: (B, L, D) tensor of context features
        timestamp: (B,) tensor of diffusion timesteps
        action: (B, L, D) tensor of action
        """
        x, x_history, context, c, pos_x, pos_x_history, pos_context = (
            self.embed_prefix_and_suffix(
                x, x_history, context, timestamp, action, drop_action_mask
            )
        )
        # Pass positions to blocks for RoPE
        for _, layer in enumerate(self.layers):
            x = layer(
                x=x,
                c=c,
                x_cond=x_history,
                context=context,
                x_positions=pos_x,
                x_cond_positions=pos_x_history,
                context_positions=pos_context,
            )

        # x = self.forward_final_layer(x, c)
        return x, c

    # def forward_final_layer(self, x, c):
    #     x = self.final_layer(x, c)
    #     x = self.unpatchify(x)
    #     return x

    # def forward_layer(
    #     self,
    #     layer_idx,
    #     x,
    #     x_history,
    #     context,
    #     timestamp,
    #     action,
    # ):

    #     x, x_history, context, c, pos_x, pos_x_history, pos_context = (
    #         self.embed_prefix_and_suffix(x, x_history, context, timestamp, action)
    #     )
    #     x = self.layers[layer_idx](
    #         x=x,
    #         c=c,
    #         x_cond=x_cond,
    #         context=context,
    #         x_positions=x_positions,
    #         x_cond_positions=x_cond_positions,
    #         context_positions=context_positions,
    #     )
    #     return x


class RoPECDiTAction(RoPECDiTDynamics):
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
        context_length=2048,
        action_chunk_size=15,
        languge_token_size=30,
        dtype=torch.float32,
        **kwargs,
    ):
        super(RoPECDiTAction, self).__init__(
            transition_dim=transition_dim,
            history_dim=history_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            output_dim=output_dim,
            dim=dim,
            n_layer=n_layer,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            patch_factor=patch_factor,
            input_size=input_size,
            context_length=context_length,
            action_chunk_size=action_chunk_size,
            languge_token_size=languge_token_size,
            dtype=dtype,
            **kwargs,
        )
        self.x_embedder = nn.Linear(transition_dim, dim)
        self.x_history_embedder = SkipEmbedder(history_dim, dim, dim)
        self.l_embedder = SkipEmbedder(context_dim, dim, dim)
        self.tl_embedder = SkipEmbedder(dim * 2, dim, dim)
        self.l_attn_pool = AttentionPooling(dim, n_head)
        self.mask_language_emb = nn.Parameter(
            torch.zeros(1, self.languge_token_size, dim), requires_grad=True
        )
        del self.a_embedder, self.ta_embedder, self.a_attn_pool, self.mask_action_emb
        del self.patch_grid_size, self.patch_factor

    def embed_prefix_and_suffix(
        self, x, x_history, context, timestamp, language, drop_language_mask
    ):
        """
        Embed prefix and suffix of the inputs.
        Args:
            x: (B, H,) tensor of spatial inputs (images or latent representations of images)
            x_history: (B, C, D) tensor of history action
            context: (B, L, D) tensor of context features
            timestamp: (B,) tensor of diffusion timesteps
            language: (B, L, D) tensor of language features
            drop_language_mask: (B,) tensor of mask for dropping language tokens; True means drop

        Returns:
            x: (B, N, D) tensor of spatial inputs (images or latent representations of images)
            x_history: (B, N, D) tensor of history spatial inputs (images or latent representations of images)
            context: (B, L, D) tensor of context features (concatenated with language)
            c: (B, D) tensor of diffusion timesteps and global pooling of language
            pos_x: (B, H*W) tensor of positions for spatial inputs (images)
            pos_x_history: (B, H*W) tensor of positions for history spatial inputs (images)
            pos_context: (B, L) tensor of positions for context features
        """
        batch_size = x.shape[0]
        x = x.to(dtype=self.dtype)
        x_history = x_history.to(dtype=self.dtype)
        timestamp = timestamp.to(dtype=self.dtype)

        # Convert context and action to dtype only if they're not None
        context = context.to(dtype=self.dtype)
        language = language.to(dtype=self.dtype)
        if drop_language_mask is not None:
            drop_language_mask = drop_language_mask.to(dtype=self.dtype)

        # Generate 1D positions for spatial inputs (images)
        pos_x_history = torch.arange(
            x_history.shape[1], device=x_history.device, dtype=torch.long
        )[None, :].expand(
            x_history.shape[0], -1
        )  # [B, N]
        x_history_max_pos = x_history.shape[1]

        # Embed actions
        x = self.x_embedder(x)  # [B, N, D]
        x_history = self.x_history_embedder(x_history)  # [B, N, D]
        t = self.t_embedder(timestamp[..., None])  # [B, D]

        # Process context (visual features)
        if self.ctx_embedder is not None:
            context = self.ctx_embedder(context)  # [B, L, D]
            pos_context = torch.arange(
                context.shape[1], device=context.device, dtype=torch.long
            )[None, :].expand(
                context.shape[0], -1
            )  # [B, L] - sequence indices starting from 0
            pos_context = x_history_max_pos + pos_context  # [B, L]
            L_context = context.shape[1]
        else:
            context = None
            pos_context = None
            L_context = 0

        # Process language
        language_embedded = self.l_embedder(language)  # [B, L, D]
        if drop_language_mask is not None:
            # Use masked language embedding - expand to batch size
            mask_language_emb = self.mask_language_emb.expand(
                batch_size, -1, -1
            )  # [B, L, D]
            language_embedded = (
                language_embedded * (1 - drop_language_mask[:, None, None])
                + mask_language_emb * drop_language_mask[:, None, None]
            )

        L_language = language_embedded.shape[1]

        # Create positions for language tokens
        pos_language = torch.arange(L_language, device=x.device, dtype=torch.long)[
            None, :
        ].expand(
            batch_size, -1
        )  # [B, L_language] - sequence indices starting from 0
        # Offset by context_end_pos to continue the position sequence
        pos_language = x_history_max_pos + L_context + pos_language  # [B, L_language]

        pos_x = torch.arange(x.shape[1], device=x.device, dtype=torch.long)[
            None, :
        ].expand(
            x.shape[0], -1
        )  # [B, N]

        pos_x = x_history_max_pos + L_context + L_language + pos_x  # [B, N]

        # Pool the language BEFORE concatenating to get global language representation
        l = self.l_attn_pool(language_embedded)  # [B, D]

        # Concatenate context and language for the prefix
        if context is not None:
            context = torch.cat(
                [context, language_embedded], dim=1
            )  # [B, L + L_language, D]
            pos_context = torch.cat(
                [pos_context, pos_language], dim=-1
            )  # [B, L + L_language]
        else:
            context = language_embedded  # [B, L_language, D]
            pos_context = pos_language  # [B, L_language]

        # Combine timestep and language embeddings
        c = torch.cat([t, l], dim=-1)  # [B, 2*D]
        c = self.tl_embedder(c)  # [B, D]
        return x, x_history, context, c, pos_x, pos_x_history, pos_context

    def forward(
        self, x, x_history, context, timestamp, language, drop_language_mask=None
    ):
        x, x_history, context, c, pos_x, pos_x_history, pos_context = (
            self.embed_prefix_and_suffix(
                x, x_history, context, timestamp, language, drop_language_mask
            )
        )
        # Pass positions to blocks for RoPE
        for _, layer in enumerate(self.layers):
            x = layer(
                x=x,
                c=c,
                x_cond=x_history,
                context=context,
                x_positions=pos_x,
                x_cond_positions=pos_x_history,
                context_positions=pos_context,
            )
        return x, c


class ActionTransformer(nn.Module):

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
        context_length=2048,
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
        self.pos_emb = nn.Parameter(torch.zeros(1, action_chunk_size, dim))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = TimestepEmbedder(dim)
        self.cond_obs_emb = nn.Linear(dim, dim)

        # Positional encoding
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, context_length, dim))

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
        # self.final_layer = FinalLayer(dim, 1, output_dim)

        # init
        self.apply(self._init_weights)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(
        self, x, x_history, context, timestamp, language, drop_language_mask=None
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
        x_history = self.history_emb(x_history)
        context = self.context_emb(context)
        language = self.language_emb(language)

        # Build the condition
        cond = torch.cat([context, language, x_history], dim=1)

        L_cond = cond.shape[1]
        input_emb = self.input_emb(x)  # [B, 1, dim]
        time_emb = self.time_emb(timestamp.unsqueeze(1)).unsqueeze(1)  # [batch, 1, dim]

        # encoder
        cond_obs_emb = self.cond_obs_emb(cond)  # [B, L, dim]

        cond_embeddings = torch.cat(
            [time_emb, cond_obs_emb], dim=1
        )  # (B, n_cond + 1, dim)
        position_embeddings = torch.cat(
            [self.time_pos_emb, self.cond_pos_emb[:, :L_cond]], dim=1
        )
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

import torch
import torch.nn as nn
from models.stateact.attention import (
    CrossAttnBlock,
    Attention,
    AttnBlock,
    AttentionPooling,
)
from models.stateact.decoder import (
    ActionTransformer,
    StateCDiTV2 as StateCDiT,
    ActionCDiT,
)
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Tuple

from models.layers_2d import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from models.helpers import (
    get_1d_sincos_pos_embed_from_grid,
    get_nd_sincos_pos_embed_from_grid,
)
import math
import numpy as np
import time
from torch.profiler import profile, ProfilerActivity


class MultiModalFusionTransformer(nn.Module):
    """
    Transformer model that fuses spatial and temporal features.
    """

    def __init__(
        self,
        horizon=15,
        transition_dim=320,
        output_dim=3,
        dim=768,
        n_time_blocks=3,
        n_space_blocks=3,
        n_head=4,
        n_virtual_register_states=64,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.horizon_past = horizon
        self.horizon_future = horizon
        self.n_head = n_head
        self.n_virtual_register_states = n_virtual_register_states
        # self.n_virtual_future = self.horizon_future
        self.dim = dim

        self.state_emb = torch.nn.Linear(transition_dim, dim, bias=True)
        self.action_emb = torch.nn.Linear(transition_dim, dim, bias=True)
        self.language_emb = torch.nn.Linear(transition_dim, dim, bias=True)

        self.virtual_register_state_tokens = nn.Parameter(
            torch.randn(1, self.n_virtual_register_states, 1, dim)
        )
        self.virtual_future_state_tokens = nn.Parameter(torch.randn(1, 196, self.horizon_future, dim))
        self.virtual_future_action_tokens = nn.Parameter(
            torch.randn(1, self.horizon_future, dim)
        )

        # Blocks do temporal attention
        self.temporal_state_blocks = nn.ModuleList(
            [
                AttnBlock(
                    self.dim,
                    self.n_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_time_blocks)
            ]
        )
        self.temporal_action_blocks = nn.ModuleList(
            [
                AttnBlock(
                    self.dim,
                    self.n_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_time_blocks)
            ]
        )

        self.temporal_state2action_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )
        self.temporal_action2state_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )

        self.temporal_action2lang_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )

        # Blocks do spatial attention for virtual tracks
        self.spatial_virtual_blocks = nn.ModuleList(
            [
                AttnBlock(
                    self.dim,
                    self.n_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_space_blocks)
            ]
        )

        self.spatial_virtual2lang_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )

        # Blocks do spatial attention from point tokens to virtual track tokens
        self.spatial_obs2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )

        # Blocks do spatial attention from virtual track tokens to point tokens
        self.spatial_virtual2obs_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )
        # assert len(self.temporal_blocks) >= len(self.spatial_virtual2obs_blocks)

        # Some other layers
        self.last_layer_state = nn.Linear(dim, output_dim)
        self.last_layer_action = nn.Linear(dim, output_dim)
        # Build the causal mask
        # S => H, P => N
        N, H = 196 + self.n_virtual_register_states, (self.horizon_past + self.horizon_future)
        sz = H * N
        # causal_mask = torch.zeros((sz, sz), dtype=torch.bool)
        # for hi in range(H):
        #     curr_view_start = hi * N
        #     curr_view_end = (hi + 1) * N
        #     causal_mask[curr_view_start:curr_view_end, curr_view_end:] = float("-inf")
        causal_mask = torch.zeros((sz, sz), dtype=torch.float)
        for hi in range(H):
            for hj in range(H):
                if hj > hi:  # Future time steps
                    # Block all spatial positions for future time steps
                    start_i = hi * N
                    end_i = (hi + 1) * N
                    start_j = hj * N
                    end_j = (hj + 1) * N
                    causal_mask[start_i:end_i, start_j:end_j] = float("-inf")
        self.register_buffer("causal_mask", causal_mask)
        # self.causal_mask = None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, x_obs, x_action, x_language):

        # x_obs: (B, N, H_past, D)
        # x_action: (B, H_past, D)
        # x_language: (B, L, D)
        # time: (B,)

        assert (
            x_obs.shape[2] == self.horizon_past
        ), "x should be of shape (B, N, H_past, D)"
        B, N = x_obs.shape[:2]

        # Prpare the virtual state tokens and future tokens
        virtual_register_state_tokens = self.virtual_register_state_tokens.repeat(
            B, 1, self.horizon_past + self.horizon_future, 1
        )  # [B, K, H, hidden_size]
        virtual_future_state_tokens = self.virtual_future_state_tokens.repeat(
            B, 1, 1, 1
        )  # [B, N, H_future, hidden_size]
        virtual_future_action_tokens = self.virtual_future_action_tokens.repeat(
            B, 1, 1
        )  # [B, H_future, hidden_size]

        # Map the state, action, and language tokens to the hidden size
        state_tokens = self.state_emb(x_obs)  # [B, N, H, D] -> [B, N, H, hidden_size]
        action_tokens = self.action_emb(x_action)  # [B, H, D] -> [B, H, hidden_size]
        language_tokens = self.language_emb(
            x_language
        )  # [B, L, D] -> [B, L, hidden_size]

        # Concatenate the state, action, and language tokens
        state_tokens = torch.cat(
            [state_tokens, virtual_future_state_tokens], dim=-2
        )  # [B, N, H, hidden_size]
        state_tokens = torch.cat(
            [state_tokens, virtual_register_state_tokens], dim=1
        )  # [B, N, H, hidden_size] -> [B, N + num_virtual_tracks, H, hidden_size]
        action_tokens = torch.cat(
            [action_tokens, virtual_future_action_tokens], dim=1
        )  # [B, H, hidden_size]

        language_tokens_action = language_tokens  # [B, L, hidden_size]
        language_tokens_state = language_tokens  # [B, L, hidden_size]
        language_tokens_state = language_tokens_state[:, None].repeat(
            self.horizon_past + self.horizon_future, 1, 1, 1
        )  # [B, H, L, hidden_size]

        # action_tokens = action_tokens.view(
        #     B * (N + self.n_virtual_register_states), self.horizon, self.dim
        # )
        language_tokens_state = language_tokens_state.view(
            B * (self.horizon_past + self.horizon_future),
            language_tokens.shape[1],
            self.dim,
        )

        N, H = state_tokens.shape[1], action_tokens.shape[1]
        assert (
            H == self.horizon_past + self.horizon_future
        ), "H should be equal to horizon"
        j = 0

        for i in range(len(self.temporal_state2action_blocks)):
            # Self-attention of state and action tokens
            # temporal_state_tokens = state_tokens.contiguous().view(
            #     B * N, self.horizon_past + self.horizon_future, -1
            # )  # B N H C -> (B N) H C
            temporal_state_tokens = rearrange(state_tokens, 'b n h c -> b (h n) c')

            start_time = time.time()
            temporal_state_tokens = self.temporal_state_blocks[i](
                temporal_state_tokens, mask=self.causal_mask
            )
            print(f"Time taken for temporal_state_blocks: {(time.time() - start_time)*1000:.3f}ms")

            spatial_state_tokens = (
                state_tokens.permute(0, 2, 1, 3)
                .contiguous()
                .view(B * (self.horizon_past + self.horizon_future), N, -1)
            )  # B N T C -> (B T) N C

            state_tokens = spatial_state_tokens.view(
                B, self.horizon_past + self.horizon_future, N, -1
            ).permute(
                0, 2, 1, 3
            )  # (B T) N C -> B N T C
            j += 1
        state_tokens = state_tokens[
            :, : N - self.n_virtual_register_states
        ]  # [B, N, H, D]
        # state_tokens = state_tokens[:, :, self.horizon_past:]
        # action_tokens = action_tokens[:, self.horizon_past:]
        state_tokens = self.last_layer_state(state_tokens)
        action_tokens = self.last_layer_action(action_tokens)
        return state_tokens, action_tokens

if __name__ == "__main__":
    state_dim, action_dim, dim, horizon = 4, 6, 384, 40
    B, N, H, K, D, L = 1, 196, horizon, 40, dim, 53
    N_query_points = 1024
    N_query_states = 4
    visual_size = 14
    attn_pooling = AttentionPooling(dim=dim, num_heads=8, head_dim=48)
    feature_extractor = MultiModalFusionTransformer(
        horizon=horizon,
        transition_dim=dim,
        output_dim=dim,
        dim=384,
        n_time_blocks=1,
        n_space_blocks=1,
        n_head=4,
        n_virtual_register_states=64,
        mlp_ratio=4.0,
    )

    feature_extractor.cuda()
    x_obs = torch.randn(B, N, H, D).cuda()
    x_action = torch.randn(B, H, D).cuda()
    x_language = torch.randn(B, 30, D).cuda()
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = feature_extractor(x_obs, x_action, x_language)
    # torch.cuda.synchronize()
    print("=========================================================")
    print("Warm-up done")
    print("=========================================================")

    with torch.inference_mode():
        for _ in range(30):
            state_tokens, action_tokens = feature_extractor(x_obs, x_action, x_language)
    print(state_tokens.shape, action_tokens.shape)

    # with profile(activities=[ProfilerActivity.CUDA]) as prof:
    #     # Run with and without causal masking
    #     _ = feature_extractor(x_obs, x_action, x_language)

    # print(prof.key_averages().table(sort_by="cuda_time_total"))


# # Timed runs with inference mode
# for i in range(10):
#     torch.cuda.synchronize()
#     start_time = time.time()
#     with torch.inference_mode():
#         state_tokens, action_tokens = feature_extractor(x_obs, x_action, x_language)
#     torch.cuda.synchronize()
#     # print(
#         # f"===> Time taken for feature_extractor: {(time.time() - start_time)*1000:.3f}ms"
#     # )
# print(state_tokens.shape, action_tokens.shape)

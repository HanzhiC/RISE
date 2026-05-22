from turtle import st
import torch
import torch.nn as nn
from models.stateact.attention import (
    CrossAttnBlock,
    Attention,
    AttnBlock,
    AttentionPooling,
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
from timm.models.vision_transformer import PatchEmbed

class MultiModalFusionTransformer(nn.Module):
    """
    Transformer model that fuses spatial and temporal features.
    """

    def __init__(
        self,
        horizon=15,
        horizon_future=5,
        transition_dim=320,
        output_dim=3,
        dim=768,
        n_time_blocks=3,
        n_space_blocks=3,
        n_head=4,
        n_visual_tokens=196,
        n_virtual_register_states=64,
        mlp_ratio=4.0,
        factor_attn=True,
        causal_attn=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.horizon_past = horizon
        self.horizon_future = horizon_future    
        self.n_head = n_head
        self.n_virtual_register_states = n_virtual_register_states
        self.n_visual_tokens = n_visual_tokens
        # self.n_virtual_future = self.horizon_future
        self.dim = dim

        self.state_emb = torch.nn.Linear(transition_dim, dim, bias=True)
        self.action_emb = torch.nn.Linear(transition_dim, dim, bias=True)
        self.language_emb = torch.nn.Linear(transition_dim, dim, bias=True)

        self.virtual_register_state_tokens = nn.Parameter(
            torch.randn(1, self.n_virtual_register_states, 1, dim)  # [B, K, 1, D]
        )
        self.virtual_future_state_tokens = nn.Parameter(
            torch.randn(1, self.n_visual_tokens, self.horizon_future, dim)
        )  # [B, N, H_future, D]
        self.virtual_future_action_tokens = nn.Parameter(
            torch.randn(1, self.horizon_future, dim)  # [B, H_future, D]
        )

        self.temporal_state_action_blocks = nn.ModuleList(
            [
                AttnBlock(
                    self.dim,
                    self.n_head,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(n_time_blocks)
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

        self.action2lang_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )
        self.virtual2lang_blocks = nn.ModuleList(
            [
                CrossAttnBlock(self.dim, self.dim, self.n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_space_blocks)
            ]
        )

        # Some other layers
        self.last_layer_state = nn.Linear(dim, output_dim)
        self.last_layer_action = nn.Linear(dim, output_dim)
        self.factor_attn = factor_attn
        self.causal_attn = causal_attn
        self.initialize_attn_mask()
        # Build the causal mask
        # St = f(St, At)
        # At = f(St+1, St)

    def initialize_attn_mask(self):
        # Build the state-action mask

        if not self.factor_attn and not self.causal_attn:
            self.attn_mask = None
            print(
                f"No attention masking is used in the fuser!!! self.attn_mask == {self.attn_mask}"
            )
            return

        H = self.horizon_past + self.horizon_future
        attn_mask = torch.zeros((H * 2, H * 2), dtype=torch.float)

        if self.factor_attn:
            # First build the attention mask for the state tokens; only attend to the others and its corrsponding action tokens;
            # For example, It attend to all I and At
            for hi in range(H):
                for hj in range(H, 2 * H):
                    if hj != hi + H - 1:
                        attn_mask[hi, hj] = float("-inf")

            # Then build the attention mask for action tokens; only attend to its corresponding state tokens and the next state token
            for hi in range(H, 2 * H):
                for hj in range(H):
                    if not (hj == hi - H + 1 or hj == hi - H):
                        attn_mask[hi, hj] = float("-inf")

        if self.causal_attn:
            # Establish the causality between state and action tokens
            for hi in range(H):
                for hj in range(H):
                    if hj > hi:
                        attn_mask[hi, hj] = float("-inf")

            for hi in range(H, 2 * H):
                for hj in range(H, 2 * H):
                    if hj > hi:
                        attn_mask[hi, hj] = float("-inf")
        self.register_buffer("attn_mask", attn_mask)

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

    def forward(self, x_obs, x_action, x_language, future_state_tokens=None, future_action_tokens=None):

        # x_obs: (B, N, H_past, D)
        # x_action: (B, H_past, D)
        # x_language: (B, L, D)
        # future_state_tokens: (B, N, H_future, D)
        # future_action_tokens: (B, H_future, D)

        assert (
            x_obs.shape[2] == self.horizon_past
        ), "x should be of shape (B, N, H_past, D)"
        B, N = x_obs.shape[:2]

        # Prpare the virtual state tokens and future tokens
        virtual_register_state_tokens = self.virtual_register_state_tokens.repeat(
            B, 1, self.horizon_past + self.horizon_future, 1
        )  # [B, K, H, hidden_size]

        if future_state_tokens is not None:
            virtual_future_state_tokens = future_state_tokens
        else:
            virtual_future_state_tokens = self.virtual_future_state_tokens.repeat(
                B, 1, 1, 1
            )  # [B, N, H_future, hidden_size]

        if future_action_tokens is not None:
            virtual_future_action_tokens = future_action_tokens
        else:
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
        for i in range(len(self.temporal_state_action_blocks)):

            # Self-attention of state and action tokens
            temporal_state_tokens = state_tokens.contiguous().view(
                B * N, self.horizon_past + self.horizon_future, -1
            )  # B N H C -> (B N) H C
            temporal_action_tokens = action_tokens  # [B, H, C]
            # Cross-attention of state and action tokens
            temporal_action_tokens = temporal_action_tokens[:, None].repeat(
                1, N, 1, 1
            )  # [B, N, H, C]
            temporal_action_tokens = temporal_action_tokens.view(
                B * N, H, -1
            )  # (B N) H C -> (B N) H C

            temporal_state_action_tokens = torch.cat(
                [temporal_state_tokens, temporal_action_tokens], dim=1
            )  # (B N) (H + H) C
            temporal_state_action_tokens = self.temporal_state_action_blocks[i](
                temporal_state_action_tokens, mask=self.attn_mask
            )

            temporal_state_tokens, temporal_action_tokens = (
                temporal_state_action_tokens.chunk(2, dim=1)
            )
            # Let the action tokens attend to the language tokens
            action_tokens = temporal_action_tokens.view(B, N, H, -1).mean(
                dim=1
            )  # (B N) H C -> B N H C -> B H C

            # Cross-attention of action and language tokens
            action_tokens = self.action2lang_blocks[i](
                action_tokens, language_tokens_action
            )

            # Spatial-wise self-attention of state tokens
            state_tokens = temporal_state_tokens.view(
                B, N, self.horizon_past + self.horizon_future, -1
            )  # (B N) H C -> B N H C
            spatial_state_tokens = (
                state_tokens.permute(0, 2, 1, 3)
                .contiguous()
                .view(B * (self.horizon_past + self.horizon_future), N, -1)
            )  # B N T C -> (B T) N C
            observed_tokens = spatial_state_tokens[
                :, : N - self.n_virtual_register_states
            ]  # [B T, N - n_virtual_register_states, hidden_size]
            virtual_register_state_tokens = spatial_state_tokens[
                :, N - self.n_virtual_register_states :
            ]  # [B T, n_virtual_register_states, hidden_size]

            virtual_register_state_tokens = self.spatial_virtual2obs_blocks[j](
                virtual_register_state_tokens, observed_tokens
            )  # [(B H), K, C]

            virtual_register_state_tokens = self.spatial_virtual_blocks[j](
                virtual_register_state_tokens
            )  # [(B H), K, C]

            # Cross-attention of virtual register state tokens and language tokens
            virtual_register_state_tokens = self.virtual2lang_blocks[j](
                virtual_register_state_tokens, language_tokens_state
            )  # [(B H), K, C]

            observed_tokens = self.spatial_obs2virtual_blocks[j](
                observed_tokens, virtual_register_state_tokens
            )  # [(B H), N, C]

            spatial_state_tokens = torch.cat(
                [observed_tokens, virtual_register_state_tokens], dim=1
            )
            state_tokens = spatial_state_tokens.view(
                B, self.horizon_past + self.horizon_future, N, -1
            ).permute(
                0, 2, 1, 3
            )  # (B T) N C -> B N T C
            j += 1

        state_tokens = state_tokens[
            :, : N - self.n_virtual_register_states
        ]  # [B, N, H, D]
        state_tokens = self.last_layer_state(state_tokens) # [B, N, H, D]
        action_tokens = self.last_layer_action(action_tokens) # [B, H, D]
        return state_tokens, action_tokens


class Action2StateTokenProjector(nn.Module):
    def __init__(
        self,
        horizon=15,
        horizon_future=5,
        action_dim=48,
        dim=384,
        n_head=4,
        mlp_ratio=4.0,
        n_visual_tokens=196,
        n_blocks=3,
        n_space_blocks=3,
    ):
        super().__init__()
        self.horizon_future = horizon_future
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.n_space_blocks = n_space_blocks
        self.dim = dim
        self.action_dim = action_dim
        self.mlp_ratio = mlp_ratio
        self.n_visual_tokens = n_visual_tokens
        self.action_emb = nn.Linear(action_dim, dim, bias=True)
        self.action_pos_emb = nn.Parameter(
            torch.zeros(1, horizon, dim),
            requires_grad=True,
        )
        self.virtual_state_tokens = nn.Parameter(
            torch.randn(1, n_visual_tokens, horizon_future, dim)  # [B, 1, H_future, D]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttnBlock(dim, dim, n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_blocks)
            ]
        )
        self.self_attn_blocks = nn.ModuleList(
            [
                AttnBlock(dim, n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_blocks)
            ]
        )
        self.last_layer = nn.Linear(dim, dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.action_pos_emb, std=0.02)

    def forward(self, x):
        """_summary_

        Parameters
        ----------
        x: (B, H, D')

        Returns:
        -------
        state_tokens (B, N, H_future, D)
        -------

        """
        B, H, _ = x.shape
        _, N, H_f, _ = self.virtual_state_tokens.shape
        x = self.action_emb(x) + self.action_pos_emb # [B, H, D]
        state_tokens = self.virtual_state_tokens.flatten(start_dim=1, end_dim=2) # [B, H_f, D]
        state_tokens = state_tokens.repeat(B, 1, 1) # [B, H_f, D]
        for i in range(self.n_blocks):
            state_tokens = self.cross_attn_blocks[i](state_tokens, x)
            state_tokens = self.self_attn_blocks[i](state_tokens)
        state_tokens = self.last_layer(state_tokens)
        state_tokens = state_tokens.view(B, N, H_f, -1)
        return state_tokens

class State2ActionTokenProjector(nn.Module):

    def __init__(
        self,
        horizon=15,
        horizon_future=5,
        state_dim=6,  # Concate start state and end state
        dim=384,
        n_head=4,
        mlp_ratio=4.0,
        patch_factor=2,
        input_size=64,
        n_blocks=3,
        n_space_blocks=3,
    ):
        super().__init__()
        self.horizon_future = horizon_future
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.n_space_blocks = n_space_blocks
        self.dim = dim
        self.state_dim = state_dim
        self.mlp_ratio = mlp_ratio
        self.patch_factor = patch_factor
        self.state_emb = PatchEmbed(
            input_size, 
            patch_factor, 
            state_dim, 
            dim, 
            bias=True
        )
        self.state_pos_emb = nn.Parameter(
            torch.zeros(1, self.state_emb.num_patches, dim),
            requires_grad=True,
        )
        self.virual_action_tokens = nn.Parameter(
            torch.randn(1, horizon_future, dim)  # [B, H_future, D]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttnBlock(dim, dim, n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_blocks)
            ]
        )
        self.self_attn_blocks = nn.ModuleList(
            [
                AttnBlock(dim, n_head, mlp_ratio=mlp_ratio)
                for _ in range(n_blocks)
            ]
        )
        self.last_layer = nn.Linear(dim, dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.state_pos_emb, std=0.02)

    def forward(self, x):
        """
        x: (B, D, P, P)
        return: (B, H, D)
        """
        B = x.shape[0]
        x = self.state_emb(x) + self.state_pos_emb # [B, N, D]
        action_tokens = self.virual_action_tokens.repeat(B, 1, 1) # [B, H_future, D]
        for i in range(self.n_blocks):
            action_tokens = self.cross_attn_blocks[i](action_tokens, x)
            action_tokens = self.self_attn_blocks[i](action_tokens)
        action_tokens = self.last_layer(action_tokens) # [B, H_future, D]
        return action_tokens

if __name__ == "__main__":
    # state_dim, action_dim, dim, horizon = 4, 48, 384, 15
    horizon = 15
    B, N, H, K, D, L = 1, 196, horizon, 15, 384, 53
    horizon_future = 5
    N_query_points = 1024   
    N_query_states = 4
    visual_size = 14
    attn_pooling = AttentionPooling(dim=384, num_heads=8, head_dim=48)
    feature_extractor = MultiModalFusionTransformer(
        horizon=horizon,
        horizon_future=horizon_future,
        transition_dim=384,
        output_dim=384,
        dim=384,
        n_time_blocks=1,
        n_space_blocks=1,
        n_head=4,
        n_virtual_register_states=64,
        mlp_ratio=4.0,
        factor_attn=True,
        causal_attn=True,
    )
    state2action_projector = State2ActionTokenProjector(
        horizon=horizon,
        horizon_future=horizon_future,
        state_dim=6,
        dim=384,
        n_head=4,
        n_blocks=1,
        n_space_blocks=1,
    )

    action2state_projector = Action2StateTokenProjector(
        horizon=horizon,
        horizon_future=horizon_future,
        action_dim=48,
        dim=384,
        n_head=4,
    )
    state2action_projector.cuda()
    action2state_projector.cuda()
    feature_extractor.cuda()

    x_obs = torch.randn(B, N, H, D).cuda()
    x_action = torch.randn(B, H, D).cuda()
    x_language = torch.randn(B, 30, D).cuda()
    action = torch.randn(B, H, 48).cuda()
    delta_state = torch.randn(B, 6, 64, 64).cuda()
    future_action_tokens = state2action_projector(delta_state)
    future_state_tokens = action2state_projector(action)
    print(future_action_tokens.shape, future_state_tokens.shape)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = feature_extractor(x_obs, x_action, x_language)

    print("=========================================================")
    print("Warm-up done")
    print("=========================================================")

    # with profile(activities=[ProfilerActivity.CUDA]) as prof:
    #     # Run with and without causal masking
    #     _ = feature_extractor(x_obs, x_action, x_language)

    # print(prof.key_averages().table(sort_by="cuda_time_total"))

    # Timed runs with inference mode
    for i in range(10):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.inference_mode():
            state_tokens, action_tokens = feature_extractor(x_obs, x_action, x_language, future_state_tokens, future_action_tokens)
        torch.cuda.synchronize()
        # print(
        # f"===> Time taken for feature_extractor: {(time.time() - start_time)*1000:.3f}ms"
        # )
    print(state_tokens.shape, action_tokens.shape)

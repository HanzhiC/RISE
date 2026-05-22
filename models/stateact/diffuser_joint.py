from re import S
import torch
import torch.nn as nn
import numpy as np
import utils.tensor_utils as TensorUtils
from collections import OrderedDict
from models.helpers import (
    cosine_beta_schedule,
    extract,
    default,
    fourier_positional_encoding,
)
from models.stateact.decoder import StateCDiTV1 as StateCDiT
from models.stateact.fuser import (
    MultiModalFusionTransformer,
    State2ActionTokenProjector,
    Action2StateTokenProjector,
)
import torch.nn.functional as F
from models.layers_2d import MLP
from einops import rearrange

# from utils.guidance_loss import DiffuserGuidance, verify_guidance_config_list
import math
from models.stateact.diffuser_state import StateDiffusionModel
from models.stateact.diffuser_action import ActionDiffusionModel
from models.stateact.attention import AttentionPooling
from models.stateact.decoder import ValueTransformer


class StateActionDiffusionModel(nn.Module):
    """
    State diffusion model that predicts the state of the next step.
    """

    def __init__(
        self,
        horizon=15,
        patch_size=64,
        action_loss_type="l2",
        state_loss_type="l2",
        state_n_timesteps=100,
        action_n_timesteps=10,
        observation_state_dim=3,
        output_state_dim=3,
        observation_action_dim=6,
        output_action_dim=6,
        base_dim=384,
        cond_fill_value=-1.0,
        supervise_epsilons=False,
        state_include_visibility=False,
        action_force_start=False,
        action_concat_language_feature=False,
        state_concat_language_feature=False,
        state_concat_dinov3_feature=False,
        state_concat_history_state=False,
        num_language_tokens=30,
        state_decoder_type="cdit",
        action_decoder_type="transformer",
        use_feature_fuser=True,
        predict_state_value=False,
        reasoning_modes=["masked", "forward", "inverse"],
        fuser_kwargs={
            "n_time_blocks": 3,
            "n_space_blocks": 3,
            "n_head": 4,
            "n_virtual_register_states": 64,
            "mlp_ratio": 4.0,
            "horizon_future": 5,
        },
        state_decoder_kwargs={
            "n_layer": 6,
            "n_head": 4,
            "input_size": 64,
            "cond_size": 64,
            "cond_horizon": 1,
            "patch_factor": 2,
            "action_dim": 384,
            "cond_horizon": 5,
        },
        action_decoder_kwargs={
            "n_layer": 6,
            "n_head": 4,
            "p_drop_emb": 0.1,
            "p_drop_attn": 0.1,
            "causal_attn": True,
            "n_cond_layers": 2,
            "n_cond_tokens": 5,
        },
    ):
        super(StateActionDiffusionModel, self).__init__()

        # Initialize all the parameters above
        self.action_loss_type = action_loss_type
        self.state_loss_type = state_loss_type
        self.horizon = horizon
        self.patch_size = patch_size
        self.state_n_timesteps = state_n_timesteps
        self.action_n_timesteps = action_n_timesteps
        self.observation_state_dim = observation_state_dim
        self.output_state_dim = output_state_dim
        self.observation_action_dim = observation_action_dim
        self.output_action_dim = output_action_dim
        self.base_dim = base_dim
        self.cond_fill_value = cond_fill_value
        self.supervise_epsilons = supervise_epsilons
        self.state_include_visibility = state_include_visibility
        self.action_force_start = action_force_start
        self.action_concat_language_feature = action_concat_language_feature
        self.state_concat_language_feature = state_concat_language_feature
        self.state_concat_dinov3_feature = state_concat_dinov3_feature
        self.state_concat_history_state = state_concat_history_state
        self.num_language_tokens = num_language_tokens
        self.state_decoder_type = state_decoder_type
        self.action_decoder_type = action_decoder_type
        self.state_decoder_kwargs = state_decoder_kwargs
        self.action_decoder_kwargs = action_decoder_kwargs
        self.use_feature_fuser = use_feature_fuser
        self.predict_state_value = predict_state_value
        self.reasoning_modes = reasoning_modes
        # Initialize the feature fuser
        if self.use_feature_fuser:
            self.fuser_kwargs = dict(fuser_kwargs)
            self.fuser_kwargs.update(
                dict(
                    horizon=self.horizon,
                    transition_dim=self.base_dim,
                    output_dim=self.base_dim,
                    dim=self.base_dim,
                )
            )
            self.fuser = MultiModalFusionTransformer(
                **self.fuser_kwargs,
            )
        else:
            self.fuser = None

        # Initialize the state diffusion models
        if state_decoder_kwargs is not None:
            self.state_diffusion_model = StateDiffusionModel(
                n_timesteps=self.state_n_timesteps,
                loss_type=self.state_loss_type,
                horizon=self.horizon,
                patch_size=self.patch_size,
                observation_state_dim=self.observation_state_dim,
                output_state_dim=self.output_state_dim,
                observation_action_dim=self.observation_action_dim,
                output_action_dim=self.output_action_dim,
                base_dim=self.base_dim,
                cond_fill_value=self.cond_fill_value,
                supervise_epsilons=self.supervise_epsilons,
                state_include_visibility=self.state_include_visibility,
                concat_language_feature=self.state_concat_language_feature,
                concat_dinov3_feature=self.state_concat_dinov3_feature,
                conact_history_state=self.state_concat_history_state,
                num_language_tokens=self.num_language_tokens,
                decoder_type=self.state_decoder_type,
                state_decoder_kwargs=self.state_decoder_kwargs,
                use_feature_fuser=False,
            )
        else:
            self.state_diffusion_model = None

        # Initialize the action diffusion model
        if action_decoder_kwargs is not None:
            self.action_diffusion_model = ActionDiffusionModel(
                n_timesteps=self.action_n_timesteps,
                loss_type=self.action_loss_type,
                horizon=self.horizon,
                observation_state_dim=self.observation_state_dim,
                output_state_dim=self.output_state_dim,
                observation_action_dim=self.observation_action_dim,
                output_action_dim=self.output_action_dim,
                base_dim=self.base_dim,
                cond_fill_value=self.cond_fill_value,
                supervise_epsilons=self.supervise_epsilons,
                force_start=self.action_force_start,
                concat_language_feature=self.action_concat_language_feature,
                num_language_tokens=self.num_language_tokens,
                decoder_type=self.action_decoder_type,
                action_decoder_kwargs=self.action_decoder_kwargs,
                use_feature_fuser=False,
            )
        else:
            self.action_diffusion_model = None

        # Delete the projection models within the state and action diffusion models
        if self.state_diffusion_model is not None:
            self.state_diffusion_model.action_proj = None
            self.state_diffusion_model.visual_proj = None
            self.state_diffusion_model.language_proj = None
            self.state_diffusion_model.fuser = None

        if self.action_diffusion_model is not None:
            self.action_diffusion_model.visual_proj = None
            self.action_diffusion_model.language_proj = None
            self.action_diffusion_model.action_proj = None
            self.action_diffusion_model.fuser = None

        # Initialize the projection models
        self.action_proj = nn.Linear(
            self.observation_action_dim, self.base_dim, bias=True
        )
        self.visual_proj = nn.Linear(768 + 6, self.base_dim, bias=True)
        self.language_proj = nn.Linear(768, self.base_dim, bias=True)

        if self.predict_state_value:
            self.value_predictor = ValueTransformer(
                transition_dim=self.base_dim,
                cond_dim=self.base_dim,
                output_dim=1,
                n_cond_tokens=self.fuser_kwargs["horizon_future"],
            )
        else:
            self.value_predictor = None

        # Initialize the reasoning models
        if "inverse" in self.reasoning_modes:
            self.state2action_projector = State2ActionTokenProjector(
                horizon=self.horizon,
                horizon_future=self.fuser_kwargs["horizon_future"],
                state_dim=self.observation_state_dim * 2,
                dim=self.base_dim,
                n_head=self.fuser_kwargs["n_head"],
            )
        else:
            self.state2action_projector = None

        if "forward" in self.reasoning_modes:
            self.action2state_projector = Action2StateTokenProjector(
                horizon=self.horizon,
                horizon_future=self.fuser_kwargs["horizon_future"],
                action_dim=self.observation_action_dim,
                dim=self.base_dim,
                n_head=self.fuser_kwargs["n_head"],
            )
        else:
            self.action2state_projector = None

    def scale_action(self, action, data_batch):
        action = self.action_diffusion_model.scale_action(
            action, data_batch
        )  # [B, H, 3]
        return action

    def descale_action(self, action, data_batch):
        action = self.action_diffusion_model.descale_action(
            action, data_batch
        )  # [B, H, 3]
        return action

    def scale_state(self, state, data_batch):
        state = self.state_diffusion_model.scale_state(state, data_batch)
        return state

    def descale_state(self, state, data_batch):
        state = self.state_diffusion_model.descale_state(state, data_batch)
        return state

    def get_aux_info(
        self,
        data_batch,
        include_class_free_cond=False,
        training=False,
        reasoning_mode="masked",
    ):
        def _transform_state(state, pose):
            H, W = state.shape[-2:]
            state = rearrange(state, "b c h w -> b (h w) c")
            state = torch.bmm(state, pose[..., :3, :3].transpose(-1, -2)) + pose[..., None, :3, 3]
            state = rearrange(state, "b (h w) c -> b c h w", h=H, w=W)
            return state

        aux_info = {}
        history_action = data_batch["history_action"]  # [B, H, 3]
        language_feature = data_batch["language_feature"]  # [B, L, 768]
        history_visual_feature_patch = data_batch[
            "history_visual_feature_patch"
        ]  # [B, H, 196, 768]
        history_raymap = data_batch["history_raymap"]  # [B, H, 6, 14, 14]
        history_raymap = rearrange(
            history_raymap, "b t c h w -> b t (h w) c"
        )  # [B, H, 196, 6]
        history_visual_feature_patch = torch.cat(
            [history_visual_feature_patch, history_raymap], dim=-1
        )  # [B, H, 196, 768 + 6]

        history_visual_feature_patch = history_visual_feature_patch.permute(
            0, 2, 1, 3
        )  # [B, 196, H, 768]

        # Project history action, language feature, and history visual feature patch to base dimension
        history_action = self.action_proj(history_action)  # [B, H, 768]
        language_feature = self.language_proj(language_feature)  # [B, L, 768]
        history_visual_feature_patch = self.visual_proj(
            history_visual_feature_patch
        )  # [B, 196, H, 768]

        future_action_tokens, future_state_tokens = None, None
        if reasoning_mode == "inverse":
            gt_state = data_batch["gt_state"]
            T_cam_cam0 = torch.inverse(data_batch["T_cam0_cam"])
            gt_state = rearrange(gt_state, "b (t c) h w -> b t c h w", c=3)
            curr_state = gt_state[:, 0]
            goal_state = gt_state[:, -1]
            curr_state = _transform_state(curr_state, T_cam_cam0) # State is in current camera frame
            goal_state = _transform_state(goal_state, T_cam_cam0) # State is in current camera frame
            curr_goal_state = torch.cat([curr_state, goal_state], dim=1)  # [B, 6, H, W]
            future_action_tokens = self.state2action_projector(curr_goal_state)
            aux_info["gt_future_action_tokens"] = future_action_tokens

        if reasoning_mode == "forward":
            gt_action = data_batch["gt_action"]
            gt_action_scaled = self.scale_action(gt_action, data_batch)
            future_state_tokens = self.action2state_projector(gt_action_scaled)
            aux_info["gt_future_state_tokens"] = future_state_tokens

        if self.fuser is not None:
            state_tokens, action_tokens = self.fuser(
                history_visual_feature_patch,
                history_action,
                language_feature,
                future_state_tokens,
                future_action_tokens,
            )  # [B, 196, 768]
            action_tokens = action_tokens[:, self.horizon :]  # [B, H, 768]
            state_tokens = state_tokens[:, :, self.horizon :]  # [B, 196, H, 768]
        else:
            state_tokens = history_visual_feature_patch  # [B, 196, H, 768]
            state_tokens = torch.cat(
                [state_tokens, state_tokens], dim=-2
            )  # [B, 196, H, 768 + 6]
            action_tokens = history_action  # [B, H, 768]
            history_visual_feature, _ = history_visual_feature_patch.max(
                dim=1
            )  # [B, H, 768]
            action_tokens = torch.cat(
                [history_action, history_visual_feature], dim=1
            )  # [B, H + H, 768]

        # action_tokens_for_state = self.action_attn_pooling(action_tokens)

        if self.action_concat_language_feature:
            action_tokens = torch.cat(
                [action_tokens, language_feature], dim=1
            )  # [B, H + L, 768]

        if self.state_concat_language_feature:
            language_feature_state = language_feature.unsqueeze(-2).repeat(
                1, 1, state_tokens.shape[-2], 1
            )  # [B, L, H, 768]
            state_tokens = torch.cat(
                [state_tokens, language_feature_state], dim=1
            )  # [B, 196+L, H, 768]

        aux_info["state_tokens"] = state_tokens
        aux_info["action_tokens"] = action_tokens
        aux_info["language_tokens"] = language_feature
        aux_info["action_tokens_for_state"] = None  # action_tokens_for_state

        # Make sure no same keys in aux_info and data_batch, no loop
        for key in data_batch.keys():
            if key in aux_info:
                raise ValueError(f"Key {key} already in aux_info")
        aux_info.update(data_batch)

        if include_class_free_cond:
            history_action_non_cond = (
                torch.ones_like(data_batch["history_action"]) * -1e3
            )
            language_feature_non_cond = (
                torch.ones_like(data_batch["language_feature_null"]) * -1e3
            )
            history_visual_feature_patch_non_cond = (
                torch.ones_like(data_batch["history_visual_feature_patch_null"]) * -1e3
            )
            history_visual_feature_patch_non_cond = (
                history_visual_feature_patch_non_cond.permute(0, 2, 1, 3)
            )  # [B, 196, H, 768]
            state_tokens_non_cond, action_tokens_non_cond = self.fuser(
                history_visual_feature_patch_non_cond,
                history_action_non_cond,
                language_feature_non_cond,
            )  # [B, 196, 768]
            state_tokens_non_cond = state_tokens_non_cond[
                :, :, self.horizon :
            ]  # [B, 196, 768]
            action_tokens_non_cond = action_tokens_non_cond[
                :, self.horizon :
            ]  # [B, H, 768]
            if self.action_concat_language_feature:
                action_tokens_non_cond = torch.cat(
                    [action_tokens_non_cond, language_feature_non_cond], dim=1
                )  # [B, H + L, 768]
            if self.state_concat_language_feature:
                language_feature_state_non_cond = language_feature_non_cond.unsqueeze(
                    -2
                ).repeat(
                    1, 1, state_tokens_non_cond.shape[-2], 1
                )  # [B, L, H, 768]
                state_tokens_non_cond = torch.cat(
                    [state_tokens_non_cond, language_feature_state_non_cond], dim=1
                )  # [B, 196 + L, H, 768]
            # action_tokens_for_state_non_cond = self.action_attn_pooling(
            #     action_tokens_non_cond
            # )

            aux_info["state_tokens_non_cond"] = state_tokens_non_cond
            aux_info["action_tokens_non_cond"] = action_tokens_non_cond
            aux_info["language_tokens_non_cond"] = language_feature_non_cond
            aux_info["start_state_non_cond"] = data_batch["start_state"].fill(0)
            aux_info["history_state_non_cond"] = data_batch["history_state"].fill(0)
            # aux_info["action_tokens_for_state_non_cond"] = (
            #     action_tokens_for_state_non_cond
            # )
        return aux_info

    def forward_state(
        self,
        data_batch,
        aux_info,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        cond_samp_out = self.state_diffusion_model.conditional_sample(
            data_batch,
            aux_info=aux_info,
            horizon=None,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance,
            guide_clean=guide_clean,
        )
        state_residual_scaled = cond_samp_out[
            "pred_state_residual"
        ]  # [B, N, 3H, 32, 32]
        state_init = data_batch["start_state"][:, None].repeat(
            1, num_samp, state_residual_scaled.shape[2] // 3, 1, 1
        )  # [B, 3, H, W] => [B, N, 3H, H, W]
        if self.state_include_visibility:
            state_visbility_init = torch.zeros_like(
                state_init[:, :, : self.output_state_dim // 3]
            )
            state_init = torch.cat([state_init, state_visbility_init], dim=2)
        state = self.descale_state(state_residual_scaled, data_batch) + state_init

        outputs = {"state_predictions": state}
        if "guide_losses" in cond_samp_out:
            outputs["state_guide_losses"] = cond_samp_out["guide_losses"]
        return outputs

    def forward_action(
        self,
        data_batch,
        aux_info,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        cond_samp_out = self.action_diffusion_model.conditional_sample(
            data_batch,
            horizon=None,
            aux_info=aux_info,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance,
            guide_clean=guide_clean,
        )
        action_scaled = cond_samp_out["pred_action"]

        action = self.descale_action(action_scaled, data_batch)
        outputs = {"action_predictions": action}
        if "guide_losses" in cond_samp_out:
            outputs["action_guide_losses"] = cond_samp_out["guide_losses"]
        return outputs

    def forward_value(self, data_batch, aux_info):
        assert self.value_predictor is not None
        action_tokens = aux_info["action_tokens"]  # [B, H, 768]
        if self.action_concat_language_feature:
            action_tokens, _ = action_tokens.split(
                [
                    action_tokens.shape[1] - self.num_language_tokens,
                    self.num_language_tokens,
                ],
                dim=1,
            )
        pred_value = self.value_predictor(action_tokens)
        outputs = {"value_predictions": pred_value}
        return outputs

    def forward(
        self,
        data_batch,
        mode="joint",
        reasoning_mode="masked",
        num_samp=1,
        forward_value=True,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        use_class_free_guide = class_free_guide_w != 0.0
        outputs = {}
        # Aux info
        aux_info = self.get_aux_info(
            data_batch,
            include_class_free_cond=use_class_free_guide,
            reasoning_mode=reasoning_mode,
        )
        if mode in ["action", "joint"]:
            outputs.update(
                self.forward_action(
                    data_batch,
                    aux_info,
                    num_samp,
                    return_diffusion,
                    return_guidance_losses,
                    class_free_guide_w,
                    apply_guidance,
                    guide_clean,
                )
            )
            if self.predict_state_value and forward_value:
                outputs.update(self.forward_value(data_batch, aux_info))

        if mode in ["state", "joint"]:
            outputs.update(
                self.forward_state(
                    data_batch,
                    aux_info,
                    num_samp,
                    return_diffusion,
                    return_guidance_losses,
                    class_free_guide_w,
                    apply_guidance,
                    guide_clean,
                )
            )

        return outputs

    def compute_action_losses(self, data_batch, aux_info, reasoning_mode="masked"):
        action = data_batch["gt_action"]
        x = self.scale_action(action, data_batch)
        diffusion_loss = self.action_diffusion_model.loss(x, aux_info=aux_info)
        losses = {"action_diffusion_loss_" + reasoning_mode: diffusion_loss}
        return losses

    def compute_state_losses(self, data_batch, aux_info, reasoning_mode="masked"):
        state_residual = data_batch["gt_state_residual"]
        state_visib = data_batch["state_visib"]
        if self.state_include_visibility:
            state_inp = torch.cat([state_residual, state_visib], dim=1)
        else:
            state_inp = state_residual
        x = self.scale_state(state_inp, data_batch)
        diffusion_loss = self.state_diffusion_model.loss(x, aux_info=aux_info)
        losses = {"state_diffusion_loss_" + reasoning_mode: diffusion_loss}
        return losses

    def compute_value_losses(self, data_batch, aux_info, reasoning_mode="masked"):
        assert self.value_predictor is not None
        gt_value = data_batch["gt_state_value"]
        value_valid = data_batch["value_valid"]
        action_tokens = aux_info["action_tokens"]
        if self.action_concat_language_feature:
            action_tokens, _ = action_tokens.split(
                [
                    action_tokens.shape[1] - self.num_language_tokens,
                    self.num_language_tokens,
                ],
                dim=1,
            )
        pred_value = self.value_predictor(action_tokens)
        value_loss = F.mse_loss(pred_value, gt_value, reduction="none")
        value_loss = (value_loss * value_valid).sum() / value_valid.sum()
        losses = {"value_loss_" + reasoning_mode: value_loss}
        return losses

    def compute_losses(self, data_batch, mode="state", reasoning_mode="masked"):
        aux_info = self.get_aux_info(
            data_batch, training=True, reasoning_mode=reasoning_mode
        )
        losses = {}
        if mode in ["state", "joint"]:
            if reasoning_mode in ["forward", "masked"]:
                losses.update(
                    self.compute_state_losses(data_batch, aux_info, reasoning_mode)
                )
        if mode in ["action", "joint"]:
            if reasoning_mode in ["inverse", "masked"]:
                losses.update(
                    self.compute_action_losses(data_batch, aux_info, reasoning_mode)
                )
            if self.predict_state_value:
                losses.update(
                    self.compute_value_losses(data_batch, aux_info, reasoning_mode)
                )
        # total_loss = 0.0
        # for lk, l in losses.items():
        #     total_loss += l
        # losses["total_diffusion_loss"] = total_loss
        return losses


if __name__ == "__main__":
    n_state_tokens = 15
    patch_size = 64
    state_dim = 45
    T = 1
    data_batch_action = {
        # Required for trajectory scaling/descaling
        "action_norm_min_bound": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])[
            :, None, :
        ].repeat(
            1, T, 1
        ),  # [B, 3] - minimum bounds for trajectory
        "action_norm_max_bound": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])[
            :, None, :
        ].repeat(
            1, T, 1
        ),  # [B, 3] - maximum bounds for trajectory
        "action_mean": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])[
            :, None, :
        ].repeat(
            1, T, 1
        ),  # [B, 3] - mean for trajectory
        "action_std": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])[:, None, :].repeat(
            1, T, 1
        ),  # [B, 3] - std for trajectory
        # Required for training (ground truth trajectory)
        "gt_action": torch.randn(
            1, T, 15, 6
        ),  # [B, horizon, observation_dim] - ground truth trajectory
        # Required for start position (when force_start=True)
        "start_pos": torch.tensor([[0.1, 0.2, 0.3]])[:, None, :].repeat(
            1, T, 1
        ),  # [B, 3] - descaled start position
        # Required conditional features (these get concatenated)
        "history_action": torch.randn(
            1, T, n_state_tokens, 6
        ),  # [B, history_action_dim] - history action features
        "language_feature": torch.randn(
            1, T, 30, 768
        ),  # [B, language_feature_dim] - action features
        "history_visual_feature_patch": torch.randn(
            1, T, n_state_tokens, 196, 768
        ),  # [B, 196, H, 768] - color features
        "history_visual_feature_patch_dinov3": torch.randn(
            1, T, n_state_tokens, 196, 768
        ),  # [B, 196, H, 768] - dinov3 history visual features
        # Optional: for class-free guidance (when class_free_guide_w != 0.0)
        "history_visual_feature_null": torch.randn(
            1, T, n_state_tokens, 196, 768
        ),  # [B, 196, H, 768] - null history visual features
        "language_feature_null": torch.randn(
            1, T, 30, 768
        ),  # [B, language_feature_dim] - null action features
        # Optional: for batch size detection in conditional_sample
        "color": torch.randn(1, T, 3, 224, 224),  # [B, C, H, W] - color images
        # OR "color_aug": torch.randn(1, 3, 256, 256),  # [B, C, H, W] - augmented color images
        "start_pos": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])[:, None, :].repeat(
            1, T, 1
        ),  # [B, 6] - descaled start position
        "action_valid": torch.ones(1, T, 15, 6),  # [B, H, 6] - action valid
    }
    data_batch_action = {k: v.cuda() for k, v in data_batch_action.items()}
    data_batch_action = TensorUtils.join_dimensions(
        data_batch_action, begin_axis=0, end_axis=2
    )

    # State
    data_batch_state = {
        "language_feature": torch.randn(
            1, 30, 768
        ),  # [B, language_feature_dim] - action features
        "history_visual_feature_patch": torch.randn(
            1, n_state_tokens, 196, 768
        ),  # [B, 196, H, 768] - color features
        # Optional: for class-free guidance (when class_free_guide_w != 0.0)
        "history_visual_feature_null": torch.randn(
            1, n_state_tokens, 196, 768
        ),  # [B, 196, H, 768] - null history visual features
        "language_feature_null": torch.randn(
            1, 30, 768
        ),  # [B, language_feature_dim] - null action features
        # Optional: for batch size detection in conditional_sample
        "color": torch.randn(1, 3, 224, 224),  # [B, C, H, W] - color images
        # OR "color_aug": torch.randn(1, 3, 256, 256),  # [B, C, H, W] - augmented color images
        "state_mean": torch.tensor(
            torch.zeros(1, state_dim)
        ),  # [B, 3] - mean for state
        "state_std": torch.tensor(torch.ones(1, state_dim)),  # [B, 3] - std for state
        "state_norm_max_bound": torch.tensor(
            torch.ones(1, state_dim)
        ),  # [B, 3] - max bound for state
        "state_norm_min_bound": torch.tensor(
            torch.zeros(1, state_dim)
        ),  # [B, 3] - min bound for state
        "state_valid": torch.randn(1, state_dim, patch_size, patch_size) > 0.5,
        "start_state": torch.randn(1, 3, patch_size, patch_size),
        "state_timestamp": torch.randn(1),
        "gt_state_residual": torch.randn(1, state_dim, patch_size, patch_size),
        "state_visib": torch.randn(1, state_dim // 3, patch_size, patch_size),
        "history_state": torch.randn(1, state_dim, patch_size, patch_size),
        # "history_state_visib": torch.randn(1, state_dim // 3, patch_size, patch_size),
        # "history_state_valid": torch.randn(1, state_dim, patch_size, patch_size),
        "start_state_dinov3_feature": torch.randn(1, 768, patch_size, patch_size),
        "relative_time": torch.randn(1),
        "history_raymap": torch.randn(1, n_state_tokens, 6, 14, 14),
        "gt_state_value": torch.randn(1),
    }
    data_batch = data_batch_action
    data_batch.update(data_batch_state)
    data_batch = {k: v.cuda() for k, v in data_batch.items()}

    model = StateActionDiffusionModel(
        action_loss_type="l2",
        state_loss_type="l2",
        horizon=15,
        patch_size=patch_size,
        state_n_timesteps=10,
        action_n_timesteps=10,
        observation_state_dim=3,
        output_state_dim=state_dim,
        observation_action_dim=6,
        output_action_dim=6,
        base_dim=384,
        cond_fill_value=-1.0,
        supervise_epsilons=False,
        state_include_visibility=False,
        action_force_start=False,
        action_concat_language_feature=True,
        state_concat_language_feature=True,
        state_concat_dinov3_feature=True,
        state_concat_history_state=False,
        num_language_tokens=30,
        state_decoder_type="cdit",
        action_decoder_type="transformer",
        use_feature_fuser=True,
        fuser_kwargs={
            "n_time_blocks": 3,
            "n_space_blocks": 3,
            "n_head": 4,
            "n_virtual_register_states": 64,
            "mlp_ratio": 4.0,
            "horizon_future": 15,
        },
        state_decoder_kwargs={
            "n_layer": 6,
            "n_head": 4,
            "input_size": 64,
            "cond_size": 14,
            "cond_horizon": 5,
            "cond_dim": 384,  # 768 + 6
            "patch_factor": 2,
            "action_dim": 384,
        },
        action_decoder_kwargs={
            "n_layer": 8,
            "n_head": 4,
            "p_drop_emb": 0.1,
            "p_drop_attn": 0.1,
            "causal_attn": True,
            "n_cond_layers": 4,
            "n_cond_tokens": 5,
        },
        predict_state_value=False,
    )
    model.cuda()

    # # Test action
    # print("Test action losses")
    # losses = model.compute_losses(data_batch, mode="action")
    # print(losses)

    # # Test state
    # print("Test state losses")
    # losses = model.compute_losses(data_batch, mode="state")
    # print(losses)

    # # Test joint
    print("Test joint losses")
    losses = model.compute_losses(data_batch, mode="joint")
    print(losses)
    import time
    # Test action forward
    with torch.no_grad():
        print("Test action forward")
        outputs = model.forward(data_batch, mode="action")
        for k, v in outputs.items():
            print(k, v.shape)

        # Test state forward
        print("Test state forward")
        for _ in range(10):
            start_time = time.time()
            outputs = model.forward(data_batch, mode="state")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            # for k, v in outputs.items():
            #     print(k, v.shape)

        # Test joint forward
        print("Test joint forward")
        outputs = model.forward(data_batch, mode="joint")
        for k, v in outputs.items():
            print(k, v.shape)

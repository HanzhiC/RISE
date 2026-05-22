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
from typing import Callable, Optional
from contextlib import AbstractContextManager
from einops import rearrange
from utils.dataset_utils import (
    transform_two_hands_trajectory,
    transform_two_hands_trajectory_absolute_to_relative,
    transform_two_hands_trajectory_relative_to_absolute,
)
import numpy as np
import random
from timm.models.vision_transformer import PatchEmbed


class VLAFlowMatching(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = 768
        self.visual_horizon = 5
        self.config = config
        self.mode = config.mode  # "vla", "vla+wm"
        self.predict_x0 = self.config.predict_x0
        self.horizon = self.config.horizon
        self.history_action_horizon = self.config.history_action_horizon
        self.history_visual_horizon = self.config.history_visual_horizon
        self.history_sample_mode = self.config.history_sample_mode
        self.visual_feature_type = config.get("visual_feature_type", "dinov3")

        # self.dtype = torch.float16 if config.dtype == "torch.float16" else torch.float32
        if self.config.dtype == "torch.float16":
            self.dtype = torch.float16
        elif self.config.dtype == "torch.bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # See if the model use gripper image
        self.am_use_gripper_image = self.config.get("am_use_gripper_image", False)
        self.am_cat_gripper_image = self.config.get("am_concat_gripper_image", False)
        self.action_chunk_size = self.config.get("action_chunk_size", self.horizon)

        # Visoon and language projection
        self.visual_proj = nn.Linear(768 + 6, self.feature_dim)
        self.language_proj = nn.Linear(768, self.feature_dim)
        if self.am_use_gripper_image:
            self.visual_proj_gripper = nn.Linear(768 + 6, self.feature_dim)

        # Action model related config
        self.action_predict_progress = self.config.get("am_predict_progress", False)
        self.action_in_relative = self.config.get("am_use_relative_action", False)
        self.action_frame = self.config.get("am_predict_action_frame", "camera")
        self.action_dim = self.config.action_dim
        self.action_output_dim = self.config.action_dim
        if self.action_predict_progress:
            self.action_output_dim += 1
        action_model_kwargs = dict(
            context_dim=self.feature_dim,
            history_dim=self.config.action_dim,
            action_dim=self.action_output_dim,
            output_dim=self.action_output_dim,
            transition_dim=self.action_output_dim,
            dim=int(self.config.am_width_multiplier * self.feature_dim),
            n_layer=self.config.num_am_layers,
            n_head=self.config.num_am_heads,
            dtype=self.dtype,
            action_chunk_size=self.action_chunk_size,
            language_token_size=30,
        )
        self.action_model_type = self.config.get("am_type", "abs_pos_transformer")

        if self.action_model_type == "abs_pos_transformer":
            self.action_model = TransformerAction(**action_model_kwargs)
        elif self.action_model_type == "rope_transformer":
            self.action_model = RoPECDiTAction(**action_model_kwargs)
        else:
            raise ValueError(f"Invalid action model type: {self.action_model_type}")
        self.action_model.to(dtype=self.dtype)
        self.action_out_proj = FinalLayerAction(
            int(self.config.am_width_multiplier * self.feature_dim),
            self.action_output_dim,
        )

        # Print am related config
        print("============================================================")
        print("================= Policy Model related config: ======================")
        for key, value in self.config.items():
            if "am_" in key:
                print(f"=====> {key}: {value} \n")
        print("============================================================")

        # Parse the dynamics model related config
        self.use_wm = "+wm" in self.mode
        if self.use_wm:
            self.visual_proj_for_wm = nn.Linear(768 + 6, self.feature_dim)
            self.language_proj_for_wm = nn.Linear(768, self.feature_dim)

            context_dim = self.feature_dim
            self.dynamics_in_relative = self.config.get("wm_use_relative_flow", True)
            self.dynamics_predict_visual = self.config.get("wm_predict_visual", False)
            self.dynamics_transition_dim = self.config.dynamics_dim + 3

            # geometric, visual, history dynamics dimensions
            self.dynamics_geometric_dim = self.config.dynamics_dim
            self.dynamics_visual_dim = 768 if self.dynamics_predict_visual else 0
            self.dynamics_history_dim = self.config.dynamics_dim

            # add additional dimensions when concatenating dinov3 feature
            if self.config.wm_concat_dinov3_feature:
                self.dynamics_transition_dim += 768

            # add additional dimensions when predicting visual
            if self.dynamics_predict_visual:
                self.dynamics_transition_dim += 768

            # # add additional dimensions when predicting distance to goal
            # if self.dynamics_predict_distance_to_goal:
            #     # self.dynamics_geometric_dim += 3
            #     self.dynamics_transition_dim += 3

            # finalize the final dynamics dimension
            self.dynamics_dim = self.dynamics_geometric_dim + self.dynamics_visual_dim
            # if self.dynamics_predict_distance_to_goal:
            #     self.dynamics_dim += 3

            # dynamics model kwargs
            dynamics_model_kwargs = dict(
                context_dim=context_dim,
                transition_dim=self.dynamics_transition_dim,
                history_dim=self.dynamics_history_dim,
                action_dim=self.action_output_dim,
                output_dim=self.dynamics_geometric_dim,  # only predict geometric dynamics
                dim=int(
                    self.config.wm_width_multiplier * self.feature_dim
                ),  # int() is used to ensure the dimension is an integer
                n_layer=self.config.num_wm_layers,
                n_head=self.config.num_wm_heads,
                patch_factor=self.config.dynamics_patch_factor,
                input_size=self.config.dynamics_input_size,
                dtype=self.dtype,
                action_chunk_size=self.action_chunk_size,
                language_token_size=30,
            )

            self.dynamics_model_type = self.config.get("wm_type", "abs_pos_transformer")

            if self.dynamics_model_type == "abs_pos_transformer":
                self.dynamics_model = CDiTDynamics(**dynamics_model_kwargs)
            elif self.dynamics_model_type == "rope_transformer":
                self.dynamics_model = RoPECDiTDynamics(**dynamics_model_kwargs)
            else:
                raise ValueError(
                    f"Invalid dynamics model type: {self.dynamics_model_type}"
                )

            self.dynamics_model.to(dtype=self.dtype)

            # Initialize the dynamics out projection
            self.dynamics_out_proj = FinalLayerDynamics(
                self.dynamics_model.dim,
                self.config.dynamics_patch_factor,
                self.dynamics_geometric_dim,
            )
            # if self.dynamics_predict_distance_to_goal:
            #     self.dynamics_dist2goal_proj = FinalLayerDynamics(
            #         self.dynamics_model.dim,
            #         self.config.dynamics_patch_factor,
            #         3,
            #     )
            # else:
            #     self.dynamics_dist2goal_proj = None

            if self.dynamics_predict_visual:
                self.dynamics_visual_proj = FinalLayerDynamics(
                    self.dynamics_model.dim,
                    1,
                    self.dynamics_visual_dim,
                )
            else:
                self.dynamics_visual_proj = None
            # Print am related config
            print("============================================================")
            print(
                "================= World Model related config: ======================"
            )
            for key, value in self.config.items():
                if "wm_" in key:
                    print(f"=====> {key}: {value} \n")
            print("============================================================")

            self.wm_concat_dinov3_feature = self.config.wm_concat_dinov3_feature
        else:
            self.dynamics_model = None
        # Parse the value model related config
        self.use_vm = "+vm" in self.mode
        if self.use_vm:
            self.visual_proj_for_vm = nn.Linear(768 + 6, self.feature_dim)
            self.language_proj_for_vm = nn.Linear(768, self.feature_dim)
            self.dynamics_proj_for_vm = PatchEmbed(
                self.config.dynamics_input_size,
                self.config.vm_dynamics_patch_factor,
                self.config.dynamics_dim,
                self.feature_dim,
                bias=True,
            )  # [B, N, D]
            # if self.am_use_gripper_image:
            #     self.visual_proj_gripper_for_vm = nn.Linear(768 + 6, self.feature_dim)

            self.value_query = nn.Parameter(
                torch.randn(1, 1, self.feature_dim), requires_grad=True
            )
            vm_model_kwargs = dict(
                transition_dim=self.feature_dim,
                context_dim=self.feature_dim,
                history_dim=self.action_output_dim,
                action_dim=self.config.action_dim,
                output_dim=self.config.action_dim,
                dim=int(self.config.vm_width_multiplier * self.feature_dim),
                n_layer=self.config.num_vm_layers,
                n_head=self.config.num_vm_heads,
                dtype=self.dtype,
                n_cond_layers=1,
                language_token_size=30,
                causal_attn=False,
            )
            self.value_model_type = self.config.get("vm_type", "abs_pos_transformer")
            if self.value_model_type == "abs_pos_transformer":
                self.value_model = TransformerAction(**vm_model_kwargs)
            elif self.value_model_type == "rope_transformer":
                self.value_model = RoPECDiTAction(**vm_model_kwargs)
            else:
                raise ValueError(f"Invalid vm model type: {self.value_model_type}")
            self.vm_use_predict_visual_prob = self.config.get(
                "vm_use_predict_visual_prob", 0.0
            )
            self.value_query.to(dtype=self.dtype)
            self.value_model.to(dtype=self.dtype)
            self.vm_out_proj = FinalLayerAction(self.value_model.dim, 1)
            print(f"=====> VM model type: {self.value_model_type}")

        else:
            self.value_model = None

        # Parse if we need to freeze the model components
        self.model_components_to_freeze = []
        self.frozen_models = self.config.get("frozen_models", "")
        if "vla" in self.frozen_models:
            self.model_components_to_freeze.append("action_model")
        if "+wm" in self.frozen_models and self.use_wm:
            self.model_components_to_freeze.append("dynamics_model")
        if "+vm" in self.frozen_models and self.use_vm:
            self.model_components_to_freeze.append("value_model")
        self.set_models_frozen(self.model_components_to_freeze)

        # Initialize the guidance
        self.current_guidance = None
        if self.use_wm:
            self.guidance_config = dict[str, dict[str, int]](
                DynamicsRegressionGuidance=dict(
                    weight=50,
                    dynamics_geometric_dim=self.dynamics_geometric_dim,
                    dynamics_visual_dim=self.dynamics_visual_dim,
                    predict_visual=self.dynamics_predict_visual,
                )
            )
            self.set_guidance(self.guidance_config)

    def set_models_frozen(self, model_components: list[str]):
        model_name_to_component = {
            "action_model": self.action_model,
            "dynamics_model": self.dynamics_model,
            "value_model": self.value_model,
        }
        # Freeze the vla model and the projector model
        for component in model_components:
            model = model_name_to_component[component]

            for _, param in model.named_parameters():
                param.requires_grad = False

            if component == "action_model":
                print(f"=====> Freezing action model ...")
                for _, param in self.action_out_proj.named_parameters():
                    param.requires_grad = False
                for _, param in self.visual_proj.named_parameters():
                    param.requires_grad = False
                for _, param in self.language_proj.named_parameters():
                    param.requires_grad = False
                if self.am_use_gripper_image:
                    for _, param in self.visual_proj_gripper.named_parameters():
                        param.requires_grad = False

            elif component == "dynamics_model":
                print(f"=====> Freezing dynamics model ...")
                for _, param in self.dynamics_out_proj.named_parameters():
                    param.requires_grad = False
                for _, param in self.visual_proj_for_wm.named_parameters():
                    param.requires_grad = False
                for _, param in self.language_proj_for_wm.named_parameters():
                    param.requires_grad = False
                if self.dynamics_predict_visual:
                    for _, param in self.dynamics_visual_proj.named_parameters():
                        param.requires_grad = False

            elif component == "value_model":
                print(f"=====> Freezing value model ...")
                self.value_query.requires_grad = False
                for _, param in self.vm_out_proj.named_parameters():
                    param.requires_grad = False
                for _, param in self.visual_proj_for_vm.named_parameters():
                    param.requires_grad = False
                for _, param in self.language_proj_for_vm.named_parameters():
                    param.requires_grad = False
                for _, param in self.dynamics_proj_for_vm.named_parameters():
                    param.requires_grad = False
                # if self.am_use_gripper_image:
                #     for _, param in self.visual_proj_gripper_for_vm.named_parameters():
                #         param.requires_grad = False

            else:
                raise ValueError(f"Invalid component: {component}")

    def set_dynamics_model(self, fm_model):
        # Copy config first so conditional assignment uses fm_model's config
        self.dynamics_in_relative = fm_model.dynamics_in_relative
        self.dynamics_predict_visual = fm_model.dynamics_predict_visual
        self.dynamics_geometric_dim = fm_model.dynamics_geometric_dim
        self.dynamics_visual_dim = fm_model.dynamics_visual_dim
        self.dynamics_transition_dim = fm_model.dynamics_transition_dim
        self.dynamics_history_dim = fm_model.dynamics_history_dim
        self.dynamics_dim = fm_model.dynamics_dim
        self.wm_concat_dinov3_feature = fm_model.wm_concat_dinov3_feature

        # Set the dynamics model and the related modules
        self.dynamics_model = fm_model.dynamics_model
        self.dynamics_out_proj = fm_model.dynamics_out_proj
        self.visual_proj_for_wm = fm_model.visual_proj_for_wm
        self.language_proj_for_wm = fm_model.language_proj_for_wm
        if self.dynamics_predict_visual:
            self.dynamics_visual_proj = fm_model.dynamics_visual_proj
        else:
            self.dynamics_visual_proj = None

    def set_value_model(self, fm_model):
        # copy the related config
        self.value_model_type = fm_model.value_model_type
        self.vm_use_predict_visual_prob = fm_model.vm_use_predict_visual_prob

        # Set the value model and the related modules
        self.value_model = fm_model.value_model
        self.value_query = fm_model.value_query
        self.visual_proj_for_vm = fm_model.visual_proj_for_vm
        self.language_proj_for_vm = fm_model.language_proj_for_vm
        # if self.am_use_gripper_image:
        #     self.visual_proj_gripper_for_vm = fm_model.visual_proj_gripper_for_vm
        self.vm_out_proj = fm_model.vm_out_proj

    def set_guidance(self, guidance_config):
        """
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        """
        print(f"=====> Setting guidance config: {guidance_config}")
        self.current_guidance = DynamicsGuidance(guidance_config)

    def scale_action(
        self,
        action: torch.Tensor,
        action_norm_min_bound: torch.Tensor,
        action_norm_max_bound: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ):
        """
        - traj: B x H x 3
        """
        if len(action.shape) == 3:
            min_bound_batch = action_norm_min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = action_norm_max_bound.unsqueeze(1)  # [B, 1, 3]
            mean_batch = action_mean.unsqueeze(1)  # [B, 1, 3]
            std_batch = action_std.unsqueeze(1)  # [B, 1, 3]
        elif len(action.shape) == 4:
            min_bound_batch = action_norm_min_bound.unsqueeze(1).unsqueeze(
                1
            )  # [B, 1, 1, 3]
            max_bound_batch = action_norm_max_bound.unsqueeze(1).unsqueeze(
                1
            )  # [B, 1, 1, 3]
            mean_batch = action_mean.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            std_batch = action_std.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        else:
            raise ValueError("Invalid shape of the input trajectory")

        # First normalize the trajectory
        action = (action - mean_batch) / (std_batch + 1e-5)
        # Then scale the trajectory
        scale = max_bound_batch - min_bound_batch
        action = (action - min_bound_batch) / (scale + 1e-5)

        # Finally, clamp the trajectory
        action = action * 2 - 1
        action = action.clamp(-6, 6)
        return action

    def descale_action(
        self,
        action: torch.Tensor,
        action_norm_min_bound: torch.Tensor,
        action_norm_max_bound: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ):
        """
        - traj: B x N x H x 3
        """
        if len(action.shape) == 3:
            min_bound_batch = action_norm_min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = action_norm_max_bound.unsqueeze(1)  # [B, 1, 3]
            mean_batch = action_mean.unsqueeze(1)  # [B, 1, 3]
            std_batch = action_std.unsqueeze(1)  # [B, 1, 3]
        elif len(action.shape) == 4:
            min_bound_batch = action_norm_min_bound.unsqueeze(1).unsqueeze(
                1
            )  # [B, 1, 1, 3]
            max_bound_batch = action_norm_max_bound.unsqueeze(1).unsqueeze(
                1
            )  # [B, 1, 1, 3]
            mean_batch = action_mean.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            std_batch = action_std.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        else:
            raise ValueError("Invalid shape of the input trajectory")

        # First unscale the trajectory
        scale = max_bound_batch - min_bound_batch
        action = (action + 1) / 2
        action = action * scale + min_bound_batch

        # Then unnormalize the trajectory
        action = action * (std_batch + 1e-5) + mean_batch
        return action

    def scale_state(self, state, state_norm_min, state_norm_max, state_mean, state_std):
        """
        - state: B x D x P x P
        """
        if len(state.shape) == 4:
            state_mean_batch = state_mean[..., None, None]  # [B, D, 1, 1]
            state_std_batch = state_std[..., None, None]  # [B, D, 1, 1]
            state_norm_max_batch = state_norm_max[..., None, None]  # [B, D, 1, 1]
            state_norm_min_batch = state_norm_min[..., None, None]  # [B, D, 1, 1]
            cat_dim = 1
        elif len(state.shape) == 5:
            state_mean_batch = state_mean[:, None, :, None, None]  # [B, 1, D, 1, 1]
            state_std_batch = state_std[:, None, :, None, None]  # [B, 1, D, 1, 1]
            state_norm_max_batch = state_norm_max[
                :, None, :, None, None
            ]  # [B, 1, D, 1, 1]
            state_norm_min_batch = state_norm_min[
                :, None, :, None, None
            ]  # [B, 1, D, 1, 1]
            cat_dim = 2
        else:
            raise ValueError(f"Invalid shape of the input state: {state.shape}")
        # First normalize the state
        state = (state - state_mean_batch) / (state_std_batch + 1e-5)

        # Then scale the state
        scale = state_norm_max_batch - state_norm_min_batch
        state = (state - state_norm_min_batch) / (scale + 1e-5)
        # Finally, clamp the state
        state = state * 2 - 1
        state = state.clamp(-6, 6)
        return state

    def descale_state(
        self, state, state_norm_min, state_norm_max, state_mean, state_std
    ):
        """
        - state: B x D x P x P
        """
        if len(state.shape) == 4:
            state_mean_batch = state_mean[..., None, None]  # [B, D, 1, 1]
            state_std_batch = state_std[..., None, None]  # [B, D, 1, 1]
            state_norm_max_batch = state_norm_max[..., None, None]  # [B, D, 1, 1]
            state_norm_min_batch = state_norm_min[..., None, None]  # [B, D, 1, 1]
            cat_dim = 1
        elif len(state.shape) == 5:
            state_mean_batch = state_mean[:, None, :, None, None]  # [B, 1, D, 1, 1]
            state_std_batch = state_std[:, None, :, None, None]  # [B, 1, D, 1, 1]
            state_norm_max_batch = state_norm_max[
                :, None, :, None, None
            ]  # [B, 1, D, 1, 1]
            state_norm_min_batch = state_norm_min[
                :, None, :, None, None
            ]  # [B, 1, D, 1, 1]
            cat_dim = 2
        else:
            raise ValueError(f"Invalid shape of the input state: {state.shape}")

        # First unscale the state
        scale = state_norm_max_batch - state_norm_min_batch
        state = (state + 1) / 2
        state = state * scale + state_norm_min_batch

        # Then unnormalize the state
        state = state * (state_std_batch + 1e-5) + state_mean_batch
        return state

    def scale_image(self, image: torch.Tensor):
        return image * 2 - 1

    def descale_image(self, image_scaled: torch.Tensor):
        return (image_scaled + 1) / 2

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def unpatchify(self, x, c=None, p=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        if c is None:
            c = self.dynamics_model.output_dim
        if p is None:
            p = self.dynamics_model.patch_factor
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def velocity_loss(
        self,
        image_features,
        language_features,
        actions,
        dynamics,
        dynamics_absolute,
        values=None,
        values_future=None,
        values_meta=None,
        image_features_next=None,
        image_features_gripper=None,
        image_features_next_gripper=None,
        initial_dynamics=None,
        initial_dynamics_feature=None,
        history_state_actions=None,
        history_state_dynamics=None,
        history_raymaps=None,
        history_raymaps_gripper=None,
        history_raymaps_next=None,
        history_raymaps_next_gripper=None,
        actions_valid=None,
        dynamics_valid=None,
        noise_actions=None,
        noise_dynamics=None,
        time=None,
        advantage_label=None,
        drop_action_mask=None,
        aux_data=None,
    ):
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)
        Args:
            image_features: [B, H, 196, 768] - batch of H image features per sample
            language_features: [B, L, 768] - batch of L language features per sample
            actions: [B, T, D] - actions
            dynamics: [B, D, H, W] - dynamics
            values: [B, 1] - values
            image_features_gripper: [B, H, 196, 768] - batch of H gripper image features per sample
            history_state_actions: [B, H, D] - history action configuration
            history_state_dynamics: [B, T*3, H, W] - history state configuration
            history_raymaps_gripper: [B, H, 6, 14, 14] - batch of H raymaps per sample
            actions_valid: [B, T, D] - validity mask for actions
            dynamics_valid: [B, T*3, H, W] - validity mask for dynamics
            noise_actions: [B, T, D] - noise for actions
            noise_dynamics: [B, D, H, W] - noise for dynamics
            time: [B] - time
            drop_action_mask: [B, T] - mask for dropping actions
            advantage_label: [B] - advantage label (1=positive, 0=negative, -1=uncond)
            aux_data: dict - auxiliary data
        Returns:
            losses: [B, T] - loss for each action
        """

        if time is None:
            time = self.sample_time(image_features.shape[0], image_features.device)

        if noise_actions is None:
            noise_actions = self.sample_noise(actions.shape, actions.device)

        time_expanded_actions = time[:, None, None]

        # Sample the actions
        x_t_actions = (
            time_expanded_actions * noise_actions
            + (1 - time_expanded_actions) * actions
        )

        if self.predict_x0:
            target_t_actions = actions  # position target
        else:
            target_t_actions = noise_actions - actions  # velocity target

        prefix_image_action, prefix_language_action = (
            self.prepare_vision_language_features(
                model="vla",
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps,
                image_features_gripper=image_features_gripper,
                history_raymaps_gripper=history_raymaps_gripper,
            )
        )

        # Ensure advantage is a tensor (1=positive, 0=negative, -1=uncond); None -> all uncond
        if advantage_label is None:
            advantage_label = torch.full(
                (image_features.shape[0],),
                -1,
                device=image_features.device,
                dtype=torch.long,
            )

        x_t_actions, c_actions = self.action_model(
            x=x_t_actions,
            x_history=history_state_actions,
            context=prefix_image_action,
            timestamp=time,
            language=prefix_language_action,
            advantage=advantage_label,
        )
        x_t_actions = x_t_actions.to(dtype=torch.float32)
        c_actions = c_actions.to(dtype=torch.float32)
        pred_t_actions = self.action_out_proj(x_t_actions, c_actions)

        losses = OrderedDict()
        # Compute the loss for the actions
        loss_actions = F.mse_loss(
            target_t_actions, pred_t_actions, reduction="none"
        )  # [B, T, D]
        if actions_valid is not None:
            loss_actions = (loss_actions * actions_valid).sum() / (
                actions_valid.sum() + 1e-5
            )
        else:
            loss_actions = loss_actions.mean()
        losses["velocity_loss_actions"] = loss_actions

        if self.use_wm:
            prefix_image_dynamics, prefix_language_dynamics = (
                self.prepare_vision_language_features(
                    model="wm",
                    image_features=image_features,
                    language_features=language_features,
                    history_raymaps=history_raymaps,
                    image_features_gripper=image_features_gripper,
                    history_raymaps_gripper=history_raymaps_gripper,
                )
            )
            prefix_dynamics = torch.cat(
                [prefix_image_dynamics, prefix_language_dynamics], dim=1
            )
            if noise_dynamics is None:
                noise_dynamics = self.sample_noise(dynamics.shape, dynamics.device)
            time_expanded_dynamics = time[:, None, None, None]

            # Sample the dynamics
            x_t_dynamics = (
                time_expanded_dynamics * noise_dynamics
                + (1 - time_expanded_dynamics) * dynamics
            )
            x_t_dynamics = torch.cat(
                [initial_dynamics, x_t_dynamics], dim=1
            )  # [B, D+3, H, W]
            if self.wm_concat_dinov3_feature:
                assert initial_dynamics_feature is not None
                x_t_dynamics = torch.cat(
                    [initial_dynamics_feature, x_t_dynamics], dim=1
                )  # [B, D+3+768, H, W]

            if self.predict_x0:
                target_t_dynamics = dynamics  # position target
            else:
                target_t_dynamics = noise_dynamics - dynamics  # velocity target

            # Build the action condition for the dynamics model
            action_cond = self.prepare_action_for_dynamics_model(
                actions, aux_data=aux_data, grad_withctx=torch.no_grad
            )

            x_t_dynamics, c_dynamics = self.dynamics_model(
                x=x_t_dynamics,
                x_history=history_state_dynamics,
                context=prefix_dynamics,
                timestamp=time,
                action=action_cond,
                # drop_action_mask=drop_action_mask,
            )
            x_t_dynamics = x_t_dynamics.to(dtype=torch.float32)
            c_dynamics = c_dynamics.to(dtype=torch.float32)
            pred_t_dynamics_geometric = self.dynamics_out_proj(x_t_dynamics, c_dynamics)
            pred_t_dynamics_geometric = self.unpatchify(
                pred_t_dynamics_geometric,
                c=self.dynamics_geometric_dim,
                p=self.config.dynamics_patch_factor,
            )
            pred_t_dynamics = pred_t_dynamics_geometric

            # Predict the visual feature
            if self.dynamics_predict_visual:
                pred_t_dynamics_visual = self.dynamics_visual_proj(
                    x_t_dynamics, c_dynamics
                )
                pred_t_dynamics_visual = self.unpatchify(
                    pred_t_dynamics_visual, c=self.feature_dim, p=1
                )
                pred_t_dynamics_visual = F.interpolate(
                    pred_t_dynamics_visual,
                    size=(
                        self.config.dynamics_input_size,
                        self.config.dynamics_input_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_t_dynamics = torch.cat(
                    [pred_t_dynamics, pred_t_dynamics_visual], dim=1
                )

            loss_dynamics = F.mse_loss(
                target_t_dynamics, pred_t_dynamics, reduction="none"
            )

            # Parse loss for geometric dynamics
            loss_dynamics_geometric = loss_dynamics[:, : self.dynamics_geometric_dim]
            if dynamics_valid is not None:
                loss_dynamics_geometric = (
                    loss_dynamics_geometric * dynamics_valid
                ).sum() / (dynamics_valid.sum() + 1e-5)
            else:
                loss_dynamics_geometric = loss_dynamics_geometric.mean()
            losses["velocity_loss_dynamics"] = loss_dynamics_geometric

            # Parse loss for visual dynamics
            if self.dynamics_predict_visual:
                loss_dynamics_visual = loss_dynamics[:, -self.dynamics_visual_dim :]  #
                loss_dynamics_visual = loss_dynamics_visual.mean()
                losses["velocity_loss_dynamics_visual"] = loss_dynamics_visual

        if self.use_vm:
            assert values is not None
            image_features_vm = image_features
            action_features_vm = actions[:, 0, :][:, None, :].repeat(
                1, self.action_chunk_size, 1
            )  # Current state action

            # Current state value
            prefix_image_value, prefix_language_value = (
                self.prepare_vision_language_features(
                    model="vm",
                    image_features=image_features_vm,
                    language_features=language_features,
                    history_raymaps=history_raymaps,
                )
            )
            dynamics_features_value = self.dynamics_proj_for_vm(
                history_state_dynamics
            )  # [B, N, D]
            context_vm = torch.cat([prefix_image_value, dynamics_features_value], dim=1)
            v = self.value_query.expand(prefix_image_value.shape[0], -1, -1)
            v, c_values = self.value_model(
                x=v,
                x_history=action_features_vm,
                context=context_vm,
                language=prefix_language_value,
                timestamp=torch.zeros_like(time),
            )
            v = self.vm_out_proj(v, c_values).squeeze((-1, -2))
            loss_value = F.mse_loss(v, values.float().view_as(v))
            losses["absolute_loss_values"] = loss_value

            if (
                self.use_wm
                and self.dynamics_predict_visual
                and self.history_sample_mode == "latest"
                and self.history_visual_horizon == 1
                and random.random() < self.vm_use_predict_visual_prob
            ):
                assert aux_data is not None
                assert dynamics_absolute is not None
                assert self.predict_x0

                res_visual = int(image_features.shape[-2] ** 0.5)

                with torch.no_grad():
                    gt_t_dynamics_visual = dynamics[:, -self.dynamics_visual_dim :]
                    future_image_features_vm_gt = (
                        self.dynamics_visual_to_vm_image_features(
                            gt_t_dynamics_visual, res_visual
                        )
                    )
                    future_absolute_geometric_dynamics_gt = dynamics_absolute
                    future_actions_features_gt = actions[:, -1, :][:, None, :].repeat(
                        1, self.action_chunk_size, 1
                    )
                    # Do inference for the gt future state as target
                    future_prefix_image_value_gt, future_prefix_language_value_gt = (
                        self.prepare_vision_language_features(
                            model="vm",
                            image_features=future_image_features_vm_gt,
                            language_features=language_features,
                            history_raymaps=history_raymaps,
                        )
                    )
                    future_dynamics_features_value_gt = self.dynamics_proj_for_vm(
                        future_absolute_geometric_dynamics_gt
                    )
                    future_context_vm_gt = torch.cat(
                        [
                            future_prefix_image_value_gt,
                            future_dynamics_features_value_gt,
                        ],
                        dim=1,
                    )
                    future_v = self.value_query.expand(
                        future_prefix_image_value_gt.shape[0], -1, -1
                    )
                    future_v, future_c_values = self.value_model(
                        x=future_v,
                        x_history=future_actions_features_gt,
                        context=future_context_vm_gt,
                        language=future_prefix_language_value_gt,
                        timestamp=torch.zeros_like(time),
                    )
                    future_v = self.vm_out_proj(future_v, future_c_values).squeeze(
                        (-1, -2)
                    )

                future_image_features_vm_pred = (
                    self.dynamics_visual_to_vm_image_features(
                        pred_t_dynamics_visual, res_visual
                    )
                )
                future_absolute_geometric_dynamics_pred = (
                    self.scaled_dynamics_output_to_absolute_geometric(
                        pred_t_dynamics, aux_data, history_state_dynamics
                    )
                )
                future_actions_features_pred = pred_t_actions[:, -1, :][
                    :, None, :
                ].repeat(1, self.action_chunk_size, 1)
                future_prefix_image_value_pred, future_prefix_language_value_pred = (
                    self.prepare_vision_language_features(
                        model="vm",
                        image_features=future_image_features_vm_pred,
                        language_features=language_features,
                        history_raymaps=history_raymaps,
                    )
                )
                future_dynamics_features_value_pred = self.dynamics_proj_for_vm(
                    future_absolute_geometric_dynamics_pred
                )
                future_context_vm_pred = torch.cat(
                    [
                        future_prefix_image_value_pred,
                        future_dynamics_features_value_pred,
                    ],
                    dim=1,
                )
                future_v_pred = self.value_query.expand(
                    future_prefix_image_value_pred.shape[0], -1, -1
                )
                future_v_pred, future_c_values_pred = self.value_model(
                    x=future_v_pred,
                    x_history=future_actions_features_pred,
                    context=future_context_vm_pred,
                    language=future_prefix_language_value_pred,
                    timestamp=torch.zeros_like(time),
                )
                future_v_pred = self.vm_out_proj(
                    future_v_pred, future_c_values_pred
                ).squeeze((-1, -2))

                loss_value_future_pred = F.mse_loss(future_v_pred, future_v.detach())
                losses["absolute_loss_future_values"] = loss_value_future_pred
            else:
                losses["absolute_loss_future_values"] = torch.zeros_like(loss_value)

            if values_meta[0] is not None and values_meta[1] is not None:
                assert (
                    image_features_next is not None and history_raymaps_next is not None
                ), "image_features_next and history_raymaps_next are required for VM"
                # Compute the value for the next state
                next_a_idx = 1  # if actions.shape[1] > 1 else 0
                action_features_vm_next = actions[:, next_a_idx, :][:, None, :].repeat(
                    1, self.action_chunk_size, 1
                )  # Next state action (a_1 if T>1, else a_0)
                prefix_image_value_next, prefix_language_value_next = (
                    self.prepare_vision_language_features(
                        model="vm",
                        image_features=image_features_next,
                        language_features=language_features,
                        history_raymaps=history_raymaps_next,
                        image_features_gripper=image_features_next_gripper,
                        history_raymaps_gripper=history_raymaps_next_gripper,
                    )
                )

                _history_dynamics = rearrange(
                    history_state_dynamics,
                    "b (t c) h w -> b t c h w",
                    t=self.horizon,
                    c=3,
                )[
                    :, 1:
                ]  # [B, T-1, 3, H, W]
                _current_dynamics = rearrange(
                    dynamics_absolute, "b (t c) h w -> b t c h w", t=self.horizon, c=3
                )[:, 1:2]
                dynamics_features_value_next = torch.cat(
                    [_history_dynamics, _current_dynamics], dim=1
                )
                dynamics_features_value_next = rearrange(
                    dynamics_features_value_next,
                    "b t c h w -> b (t c) h w",
                )
                dynamics_features_value_next = self.dynamics_proj_for_vm(
                    dynamics_features_value_next
                )

                context_vm_next = torch.cat(
                    [prefix_image_value_next, dynamics_features_value_next], dim=1
                )
                v_next = self.value_query.expand(
                    prefix_image_value_next.shape[0], -1, -1
                )
                v_next, c_values_next = self.value_model(
                    x=v_next,
                    x_history=action_features_vm_next,
                    context=context_vm_next,
                    language=prefix_language_value_next,
                    timestamp=torch.zeros_like(time),
                )
                v_next = self.vm_out_proj(v_next, c_values_next).squeeze((-1, -2))

                # TD Targets
                is_terminal, gt_td_reward = values_meta
                is_terminal = is_terminal.float().view_as(v)
                gt_td_reward = gt_td_reward.float().view_as(v)

                gamma = 0.995
                td_target = gt_td_reward + gamma * (1.0 - is_terminal) * v_next.detach()
                loss_td = F.mse_loss(v, td_target)
                losses["absolute_loss_values_td"] = loss_td

        return losses

    def dynamics_visual_to_vm_image_features(
        self, dynamics_visual: Tensor, res_visual: int
    ):
        """
        Map predicted dynamics visual features [B, C, H, W] to the value-model
        image feature layout [B, T, L, C] with T = horizon and L the token count
        (H' * W' after bilinear resampling to (res_visual, res_visual)).
        """
        # x = dynamics_visual
        x = dynamics_visual.clone().detach()
        x = F.interpolate(
            x,
            size=(res_visual, res_visual),
            mode="bilinear",
            align_corners=False,
        )
        x = rearrange(x, "b c h w -> b (h w) c")
        return x[:, None].expand(-1, self.horizon, -1, -1) * 5.0

    def scaled_dynamics_output_to_absolute_geometric(
        self,
        pred_t_dynamics: Tensor,
        aux_data: dict,
        history_state: Optional[Tensor] = None,
    ):
        """
        From WM output in scaled space (optional [geometric | visual] on dim=1) to
        absolute geometric dynamics, same path as `outputs['dynamics_predictions']` in
        `forward` (descale + add last history frame when `dynamics_in_relative`).
        """
        dynamics_geometric = pred_t_dynamics[:, : self.dynamics_geometric_dim]
        dynamics_geometric = dynamics_geometric.clone().detach()
        dynamics_geometric = self.descale_state(
            dynamics_geometric,
            aux_data["state_norm_min_bound"],
            aux_data["state_norm_max_bound"],
            aux_data["state_mean"],
            aux_data["state_std"],
        )
        if self.dynamics_in_relative:
            assert history_state is not None
            history_state_last = rearrange(
                history_state, "b (t c) h w -> b t c h w", c=3
            )[:, -1]
            history_state_last = history_state_last.repeat(
                1, dynamics_geometric.shape[1] // 3, 1, 1
            )
            dynamics_geometric = dynamics_geometric + history_state_last
        return dynamics_geometric

    def prepare_vision_language_features(
        self,
        model,
        image_features,
        language_features,
        history_raymaps,
        image_features_gripper=None,
        history_raymaps_gripper=None,
    ):
        if model == "vla":
            prefix_image, prefix_language = self.embed_vision_language_features(
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps,
                visual_projector=self.visual_proj,
                language_projector=self.language_proj,
            )
            if self.am_use_gripper_image:
                prefix_gripper_image = self.embed_vision_features_gripper(
                    image_features_gripper=image_features_gripper,
                    history_raymaps_gripper=history_raymaps_gripper,
                    visual_projector=self.visual_proj_gripper,
                )
                if self.am_cat_gripper_image:
                    prefix_image = torch.cat(
                        [prefix_image, prefix_gripper_image], dim=1
                    )
                else:
                    prefix_image = prefix_gripper_image

        elif model == "wm":
            assert self.visual_proj_for_wm is not None
            assert self.language_proj_for_wm is not None
            prefix_image, prefix_language = self.embed_vision_language_features(
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps,
                visual_projector=self.visual_proj_for_wm,
                language_projector=self.language_proj_for_wm,
            )

        elif model == "vm":
            # Raymaps are not used for VM, so we set them to 0.0 with a hack
            assert self.visual_proj_for_vm is not None
            assert self.language_proj_for_vm is not None
            prefix_image, prefix_language = self.embed_vision_language_features(
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps * 0.0,
                visual_projector=self.visual_proj_for_vm,
                language_projector=self.language_proj_for_vm,
            )
            # if self.am_use_gripper_image:
            #     prefix_gripper_image = self.embed_vision_features_gripper(
            #         image_features_gripper=image_features_gripper,
            #         history_raymaps_gripper=history_raymaps_gripper * 0.0,
            #         visual_projector=self.visual_proj_gripper_for_vm,
            #     )
            #     if self.am_cat_gripper_image:
            #         prefix_image = torch.cat(
            #             [prefix_image, prefix_gripper_image], dim=1
            #         )
            #     else:
            #         prefix_image = prefix_gripper_image

        else:
            raise ValueError(f"Invalid model: {model}")

        return prefix_image, prefix_language

    def prepare_action_for_dynamics_model(
        self,
        action_scaled: torch.Tensor,
        aux_data: dict,
        grad_withctx: Callable[[], AbstractContextManager] = torch.no_grad,
    ):
        # Descale the action
        with grad_withctx():
            if self.action_in_relative:
                action_scaled_for_dynamics = action_scaled
            else:
                start_pos = aux_data["start_pos"]  # [B, ACTION_DIM]
                action_descaled = self.descale_action(
                    action_scaled,
                    aux_data["action_norm_min_bound"],
                    aux_data["action_norm_max_bound"],
                    aux_data["action_mean"],
                    aux_data["action_std"],
                )  # [B, H, ACTION_DIM]

                # Transform the action to relative
                action_relative_descaled = (
                    transform_two_hands_trajectory_absolute_to_relative(
                        action_descaled,
                        start_pos,
                        has_finger_tips=self.action_dim == 48,
                    )
                )

                # Scale the action back to scale used for dynamics model
                action_scaled_for_dynamics = self.scale_action(
                    action_relative_descaled,
                    aux_data["action_for_dynamics_norm_min_bound"],
                    aux_data["action_for_dynamics_norm_max_bound"],
                    aux_data["action_for_dynamics_mean"],
                    aux_data["action_for_dynamics_std"],
                )  # [B, H, ACTION_DIM]

        return action_scaled_for_dynamics

    def preprocess_history_actions(
        self, history: torch.Tensor, sample_mode: str = "uniform"
    ):
        """
        Downsample the history, then repeat along T so output length is self.horizon.
        E.g. Bx15xD, history_action_horizon=5 -> sample [2,5,8,11,14] -> [2,2,2,5,5,5,8,8,8,11,11,11,14,14,14]
        Args:
            history: [B, T, D] - history with T=horizon
        Returns:
            history: [B, horizon, D] - downsampled then repeated to horizon
        """
        # Downsample: e.g. Bx15xD, history_action_horizon=5 -> indices [2, 5, 8, 11, 14]
        if sample_mode == "uniform":
            if self.history_action_horizon <= 1:
                sample_indices = [self.horizon - 1]
            else:
                step = (self.horizon - 1) // (self.history_action_horizon - 1)
                start = (self.horizon - 1) - step * (self.history_action_horizon - 1)
                sample_indices = [
                    start + i * step for i in range(self.history_action_horizon)
                ]
        elif sample_mode == "latest":
            sample_indices = list(
                np.arange(
                    self.horizon - 1, self.horizon - self.history_action_horizon - 1, -1
                )
            )[::-1]
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")

        history = history[:, sample_indices, :]  # [B, history_action_horizon, D]

        # Repeat each sampled step so T dimension is self.horizon again
        repeat_count = self.horizon // self.history_action_horizon
        history = history.repeat_interleave(
            repeat_count, dim=1
        )  # [B, horizon or less, D]
        if history.size(1) < self.horizon:
            pad_len = self.horizon - history.size(1)
            history = torch.cat(
                [history, history[:, -1:, :].expand(-1, pad_len, -1)], dim=1
            )
        return history

    def preprocess_history_visual_features(
        self, history: torch.Tensor, sample_mode: str = "uniform"
    ):
        """
        Downsample the history, then repeat along T so output length is self.horizon.
        E.g. Bx15x..., history_visual_horizon=5 -> sample [2,5,8,11,14]
        Args:
            history: [B, T, ...] - history with T=horizon
        Returns:
            history: [B, history_visual_horizon, ...] - downsampled then repeated to horizon
        """

        # Downsample: e.g. Bx15x..., history_visual_horizon=5 -> indices [2, 5, 8, 11, 14]
        if sample_mode == "uniform":
            if self.history_visual_horizon <= 1:
                sample_indices = [self.horizon - 1]
            else:
                step = (self.horizon - 1) // (self.history_visual_horizon - 1)
                start = (self.horizon - 1) - step * (self.history_visual_horizon - 1)
                sample_indices = [
                    start + i * step for i in range(self.history_visual_horizon)
                ]
        elif sample_mode == "latest":
            sample_indices = list(
                np.arange(
                    self.horizon - 1, self.horizon - self.history_visual_horizon - 1, -1
                )
            )[::-1]
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")
        history = history[:, sample_indices]  # [B, history_visual_horizon, ...]

        # Repeat each sampled step so T dimension is 5
        repeat_count = self.visual_horizon // self.history_visual_horizon
        history = history.repeat_interleave(repeat_count, dim=1)  # [B, 5, ...]
        if history.size(1) < self.visual_horizon:
            pad_len = self.visual_horizon - history.size(1)
            rest_of_the_shape = tuple(history.shape[2:])
            history = torch.cat(
                [history, history[:, -1:, ...].expand(-1, pad_len, *rest_of_the_shape)],
                dim=1,
            )
        return history

    def compute_losses(self, data_batch: dict):
        action_rel = "_rel" if self.action_in_relative else ""
        image_features = data_batch["history_visual_feature_patch"]  # [B, H, 196, 768]
        image_features_next = data_batch.get("history_visual_feature_patch_next", None)
        language_features = data_batch["language_feature"]  # [B, L, 768]
        history_raymaps = data_batch["history_raymap"]
        actions_scaled = self.scale_action(
            data_batch[f"gt_action{action_rel}"][:, :, : self.action_dim],
            data_batch["action_norm_min_bound"],
            data_batch["action_norm_max_bound"],
            data_batch["action_mean"],
            data_batch["action_std"],
        )  # [B, T, D], in [-1, 1]
        if self.action_predict_progress:
            # Do normalization for progress
            # Progress is in [0, 1] range, need to scale to [-1, 1]
            progress = data_batch[f"gt_action{action_rel}"][
                :, :, self.action_dim :
            ]  # [B, T, 1]
            progress_scaled = progress * 2.0 - 1.0  # [0, 1] => [-1, 1]
            actions_scaled = torch.cat([actions_scaled, progress_scaled], dim=-1)

        if not self.action_in_relative:
            history_state_actions = self.scale_action(
                data_batch[f"history_action"],
                data_batch["action_norm_min_bound"],
                data_batch["action_norm_max_bound"],
                data_batch["action_mean"],
                data_batch["action_std"],
            )  # [B, T, D]
        else:
            history_state_actions = data_batch[f"history_action"]

        history_state_actions = self.preprocess_history_actions(
            history_state_actions, sample_mode=self.history_sample_mode
        )

        action_valid = data_batch["action_valid"]
        if "optimality_label" in data_batch:
            action_valid = action_valid * data_batch["optimality_label"][:, None, None]
        advantage_label = data_batch.get(
            "advantage_label",
            torch.full(
                (len(data_batch["gt_action"]),),
                -1,
                device=data_batch["gt_action"].device,
                dtype=torch.long,
            ),
        )  # Unconditional
        if self.use_wm:
            dynamics_suffix = "_residual" if self.dynamics_in_relative else ""
            dynamics_scaled = self.scale_state(
                data_batch[f"gt_state{dynamics_suffix}"],
                data_batch["state_norm_min_bound"],
                data_batch["state_norm_max_bound"],
                data_batch["state_mean"],
                data_batch["state_std"],
            )  # [B, D, H, W]

            # if self.dynamics_predict_distance_to_goal:
            #     dist2goal_dynamics_scaled = self.scale_state(
            #         data_batch["gt_distance_to_goal"],  # [B, 3, H, W]
            #         data_batch["distance_to_goal_norm_min_bound"],  # [B, 3]
            #         data_batch["distance_to_goal_norm_max_bound"],  # [B, 3]
            #         data_batch["distance_to_goal_mean"],  # [B, 3]
            #         data_batch["distance_to_goal_std"],  # [B, 3]
            #     )  # [B, 3, H, W]
            #     dynamics_scaled = torch.cat(
            #         [dynamics_scaled, dist2goal_dynamics_scaled], dim=1
            #     )

            if self.dynamics_predict_visual:
                visual_res = int(
                    data_batch["goal_visual_feature_patch"].shape[-2] ** 0.5
                )
                visual_dynamics_scaled = rearrange(
                    data_batch["goal_visual_feature_patch"],
                    "b (h w) c -> b c h w",
                    h=visual_res,
                    w=visual_res,
                )
                visual_dynamics_scaled = F.interpolate(
                    visual_dynamics_scaled,
                    size=(
                        self.config.dynamics_input_size,
                        self.config.dynamics_input_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                dynamics_scaled = torch.cat(
                    [dynamics_scaled, visual_dynamics_scaled], dim=1
                )
            initial_dynamics = data_batch["start_state"]
            if self.wm_concat_dinov3_feature:
                initial_dynamics_feature = data_batch["start_state_dinov3_feature"]
            else:
                initial_dynamics_feature = None
            history_state_dynamics = data_batch["history_state"]
            state_valid = data_batch["state_valid"]
            dynamics_absolute = data_batch["gt_state"]

        else:
            dynamics_scaled = None
            initial_dynamics = None
            initial_dynamics_feature = None
            if self.use_vm:
                history_state_dynamics = data_batch["history_state"]
                dynamics_absolute = data_batch["gt_state"]
                state_valid = data_batch["state_valid"]

            else:
                history_state_dynamics = None
                dynamics_absolute = None
                state_valid = None

        if "drop_action_mask" in data_batch:
            drop_action_mask = data_batch["drop_action_mask"]
        else:
            drop_action_mask = None

        if self.am_use_gripper_image:
            assert (
                "history_visual_feature_patch_gripper" in data_batch
            ), "history_visual_feature_patch_gripper is required for gripper image mode"
            image_features_gripper = data_batch["history_visual_feature_patch_gripper"]
            history_raymaps_gripper = data_batch["history_raymap_gripper"]
        else:
            image_features_gripper = None
            history_raymaps_gripper = None

        if self.use_vm:
            values = data_batch["gt_state_value"]
        else:
            values = None
        values_future = data_batch.get("gt_state_value_future", None)
        image_features_next_gripper = data_batch.get(
            "history_visual_feature_patch_next_gripper", None
        )
        history_raymaps_next_gripper = data_batch.get(
            "history_raymap_next_gripper", None
        )
        image_features_next = data_batch.get("history_visual_feature_patch_next", None)
        history_raymaps_next = data_batch.get("history_raymap_next", None)
        gt_td_reward = data_batch.get("gt_td_reward", None)
        is_terminal = data_batch.get("is_terminal", None)
        values_meta = (is_terminal, gt_td_reward)

        aux_data = self.get_aux_info(data_batch)
        losses = self.velocity_loss(
            image_features=image_features,
            image_features_gripper=image_features_gripper,
            image_features_next=image_features_next,
            image_features_next_gripper=image_features_next_gripper,
            language_features=language_features,
            actions=actions_scaled,
            dynamics=dynamics_scaled,
            dynamics_absolute=dynamics_absolute,
            values=values,
            values_future=values_future,
            values_meta=values_meta,
            # values_future=values_future,
            initial_dynamics=initial_dynamics,
            initial_dynamics_feature=initial_dynamics_feature,
            history_state_actions=history_state_actions,
            history_state_dynamics=history_state_dynamics,
            history_raymaps=history_raymaps,
            history_raymaps_gripper=history_raymaps_gripper,
            history_raymaps_next=history_raymaps_next,
            history_raymaps_next_gripper=history_raymaps_next_gripper,
            actions_valid=action_valid,
            dynamics_valid=state_valid,
            noise_actions=None,
            noise_dynamics=None,
            time=None,
            advantage_label=advantage_label,
            drop_action_mask=drop_action_mask,
            aux_data=aux_data,
        )

        return losses

    def get_aux_info(self, data_batch: dict):
        """
        Get the auxiliary information for the model
        Args:
            data_batch: dict - data batch
        Returns:
            aux_data: dict - auxiliary information
        """
        aux_info = {
            "T_world_cam": data_batch["T_world_cam"],
            "action_norm_min_bound": data_batch["action_norm_min_bound"],
            "action_norm_max_bound": data_batch["action_norm_max_bound"],
            "action_mean": data_batch["action_mean"],
            "action_std": data_batch["action_std"],
        }
        if "action_for_dynamics_norm_min_bound" in data_batch:
            aux_info["action_for_dynamics_norm_min_bound"] = data_batch[
                "action_for_dynamics_norm_min_bound"
            ]
            aux_info["action_for_dynamics_norm_max_bound"] = data_batch[
                "action_for_dynamics_norm_max_bound"
            ]
            aux_info["action_for_dynamics_mean"] = data_batch[
                "action_for_dynamics_mean"
            ]
            aux_info["action_for_dynamics_std"] = data_batch["action_for_dynamics_std"]
        if (
            "history_visual_feature_patch" in data_batch
            and "history_visual_feature_patch_null" not in data_batch
        ):
            aux_info["history_visual_feature_patch_null"] = (
                data_batch["history_visual_feature_patch"].clone().fill_(1e-3)
            )
        elif "history_visual_feature_patch_null" in data_batch:
            aux_info["history_visual_feature_patch_null"] = data_batch[
                "history_visual_feature_patch_null"
            ]
        if (
            "history_visual_feature_patch_gripper" in data_batch
            and "history_visual_feature_patch_gripper_null" not in data_batch
        ):
            aux_info["history_visual_feature_patch_gripper_null"] = (
                data_batch["history_visual_feature_patch_gripper"].clone().fill_(1e-3)
            )
        elif "history_visual_feature_patch_gripper_null" in data_batch:
            aux_info["history_visual_feature_patch_gripper_null"] = data_batch[
                "history_visual_feature_patch_gripper_null"
            ]

        if "history_state_actions_null" in data_batch:
            aux_info["history_state_actions_null"] = data_batch[
                "history_state_actions_null"
            ]
        if "history_raymap" in data_batch:
            aux_info["history_raymap"] = data_batch["history_raymap"]
        if "history_raymap_gripper" in data_batch:
            aux_info["history_raymap_gripper"] = data_batch["history_raymap_gripper"]
        if "state_norm_min_bound" in data_batch:
            aux_info["state_norm_min_bound"] = data_batch["state_norm_min_bound"]
            aux_info["state_norm_max_bound"] = data_batch["state_norm_max_bound"]
            aux_info["state_mean"] = data_batch["state_mean"]
            aux_info["state_std"] = data_batch["state_std"]
        if "distance_to_goal_norm_min_bound" in data_batch:
            aux_info["distance_to_goal_norm_min_bound"] = data_batch[
                "distance_to_goal_norm_min_bound"
            ]
            aux_info["distance_to_goal_norm_max_bound"] = data_batch[
                "distance_to_goal_norm_max_bound"
            ]
            aux_info["distance_to_goal_mean"] = data_batch["distance_to_goal_mean"]
            aux_info["distance_to_goal_std"] = data_batch["distance_to_goal_std"]
        if "start_pos" in data_batch:
            aux_info["start_pos"] = data_batch["start_pos"]
        if "language_feature_null" in data_batch:
            aux_info["language_feature_null"] = data_batch["language_feature_null"]
        if "goal_visual_feature_patch" in data_batch:
            aux_info["goal_visual_feature_patch"] = data_batch[
                "goal_visual_feature_patch"
            ]
        if "goal_distance_to_goal" in data_batch:
            aux_info["goal_distance_to_goal"] = data_batch["goal_distance_to_goal"]
        if "goal_state_residual" in data_batch:
            aux_info["goal_state_residual"] = data_batch["goal_state_residual"]
        if "goal_state" in data_batch:
            aux_info["goal_state"] = data_batch["goal_state"]
        if "state_color" in data_batch:
            aux_info["state_color"] = data_batch["state_color"]
        if "start_state" in data_batch:
            aux_info["start_state"] = data_batch["start_state"]
        if "gt_action" in data_batch:
            aux_info["gt_action"] = data_batch["gt_action"]
        if "gt_state_residual" in data_batch:
            aux_info["gt_state_residual"] = data_batch["gt_state_residual"]
        if "gt_state" in data_batch:
            aux_info["gt_state"] = data_batch["gt_state"]
        if "state_valid" in data_batch:
            aux_info["state_valid"] = data_batch["state_valid"]
        return aux_info

    def embed_vision_language_features(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
        history_raymaps: torch.Tensor,
        visual_projector: nn.Module,
        language_projector: nn.Module,
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
        image_features = self.preprocess_history_visual_features(
            image_features, sample_mode=self.history_sample_mode
        )
        history_raymaps = self.preprocess_history_visual_features(
            history_raymaps, sample_mode=self.history_sample_mode
        )
        history_raymaps = rearrange(
            history_raymaps, "b t c h w -> b t (h w) c"
        )  # [B, 5, 196, 6]
        image_features = torch.cat(
            [image_features, history_raymaps], dim=-1
        )  # [B, 5, 196, 768 + 6]
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )  # [B, 5*196, 768 + 6]
        image_features = visual_projector(image_features)
        language_features = language_projector(language_features)
        return image_features, language_features

    def embed_vision_features_gripper(
        self,
        image_features_gripper: torch.Tensor,
        history_raymaps_gripper: torch.Tensor,
        visual_projector: nn.Module,
    ):
        """
        Embed the gripper image features
        Args:
            image_features_gripper: [B, H, 196, 768] - batch of H gripper image features per sample
            history_raymaps_gripper: [B, H, 6, 14, 14] - batch of H raymaps per sample
        Returns:
            image_features_gripper: [B, 5 * 196, D] - batch of gripper image features
        """
        image_features_gripper = self.preprocess_history_visual_features(
            image_features_gripper,
            sample_mode=self.history_sample_mode,
        )
        history_raymaps_gripper = self.preprocess_history_visual_features(
            history_raymaps_gripper,
            sample_mode=self.history_sample_mode,
        )
        history_raymaps_gripper = rearrange(
            history_raymaps_gripper, "b t c h w -> b t (h w) c"
        )  # [B, 5, 196, 6]
        image_features_gripper = torch.cat(
            [image_features_gripper, history_raymaps_gripper], dim=-1
        )  # [B, 5, 196, 768 + 6]
        image_features_gripper = image_features_gripper.reshape(
            image_features_gripper.shape[0], -1, image_features_gripper.shape[-1]
        )  # [B, 5*196, 768 + 6]
        image_features_gripper = visual_projector(image_features_gripper)
        return image_features_gripper

    def forward(
        self,
        data_batch: dict,
        action_only: bool = False,
        value_only: bool = False,
        input_actions: torch.Tensor = None,
        input_dynamics: torch.Tensor = None,
        w_advantage: float = 1.0,
        w_conditional: float = 1.0,
        num_samples: int = 1,
        enable_guidance: bool = False,
        eval_guidance: bool = False,
        random_noise_scale: float = 0.0,
    ):
        """
        Do a full inference forward and compute the action (batch_size x num_steps x num_motors)
        Args:
            data_batch: dict - data batch
            action_only: bool - whether to predict the actions
            value_only: bool - whether to predict the values only. When use_vm is True,
                input_actions must be provided (same conditioning as training).
            input_actions: torch.Tensor - action chunk for conditioning (required if value_only
                with use_vm); also used as flow noise / guidance when provided.
        Returns:
            outputs: dict - output dictionary
        """
        # Based on the number of samples, we need to repeat the data batch
        outputs = OrderedDict()
        image_features = data_batch["history_visual_feature_patch"]  # [B, H, 196, 768]
        language_features = data_batch["language_feature"]  # [B, L, 768]
        history_raymaps = data_batch["history_raymap"]
        image_features_gripper = data_batch.get(
            "history_visual_feature_patch_gripper", None
        )
        history_raymaps_gripper = data_batch.get("history_raymap_gripper", None)
        original_bsize = image_features.shape[0]
        # action_rel = "_rel" if self.action_in_relative else ""

        if input_actions is not None:
            input_actions_scaled = self.scale_action(
                input_actions[..., : self.action_dim],
                data_batch["action_norm_min_bound"],
                data_batch["action_norm_max_bound"],
                data_batch["action_mean"],
                data_batch["action_std"],
            )
            if self.action_predict_progress:
                # Do normalization for progress
                # Progress is in [0, 1] range, need to scale to [-1, 1]
                progress = input_actions[..., self.action_dim :]  # [B, T, 1]
                progress_scaled = progress * 2.0 - 1.0  # [0, 1] => [-1, 1]
                input_actions_scaled = torch.cat(
                    [input_actions_scaled, progress_scaled], dim=-1
                )
            vm_action_features = input_actions_scaled[:, :1, :].repeat(
                1, self.action_chunk_size, 1
            )

        else:
            input_actions_scaled = None
            vm_action_features = None

        # Action prediction and dynamics prediction when not value-only
        if not value_only:
            if not self.action_in_relative:
                history_state_actions = self.scale_action(
                    data_batch[f"history_action"],
                    data_batch["action_norm_min_bound"],
                    data_batch["action_norm_max_bound"],
                    data_batch["action_mean"],
                    data_batch["action_std"],
                )  # [B, T, D]
            else:
                history_state_actions = data_batch[f"history_action"]

            history_state_actions = self.preprocess_history_actions(
                history_state_actions, sample_mode=self.history_sample_mode
            )
            initial_action = data_batch["start_pos"]
            initial_action_scaled = self.scale_action(
                initial_action[:, None],
                data_batch["action_norm_min_bound"],
                data_batch["action_norm_max_bound"],
                data_batch["action_mean"],
                data_batch["action_std"],
            )[:, 0]

            if self.use_wm and not action_only:
                history_state_dynamics = data_batch["history_state"]  # [B, T*3, H, W]
                initial_dynamics = data_batch["start_state"]  # [B, 3, H, W]

                if self.wm_concat_dinov3_feature:
                    initial_dynamics_feature = data_batch["start_state_dinov3_feature"]
                else:
                    initial_dynamics_feature = None

            else:
                history_state_dynamics = None
                initial_dynamics = None
                initial_dynamics_feature = None

            aux_data = self.get_aux_info(data_batch)

            # Repeat the data batch for sampling
            image_features = TensorUtils.repeat_by_expand_at(
                image_features, repeats=num_samples, dim=0
            )
            language_features = TensorUtils.repeat_by_expand_at(
                language_features, repeats=num_samples, dim=0
            )
            history_raymaps = TensorUtils.repeat_by_expand_at(
                history_raymaps, repeats=num_samples, dim=0
            )
            initial_action_scaled = TensorUtils.repeat_by_expand_at(
                initial_action_scaled, repeats=num_samples, dim=0
            )
            history_state_actions = TensorUtils.repeat_by_expand_at(
                history_state_actions, repeats=num_samples, dim=0
            )
            if image_features_gripper is not None:
                image_features_gripper = TensorUtils.repeat_by_expand_at(
                    image_features_gripper, repeats=num_samples, dim=0
                )
            if history_raymaps_gripper is not None:
                history_raymaps_gripper = TensorUtils.repeat_by_expand_at(
                    history_raymaps_gripper, repeats=num_samples, dim=0
                )
            if initial_dynamics is not None:
                initial_dynamics = TensorUtils.repeat_by_expand_at(
                    initial_dynamics, repeats=num_samples, dim=0
                )
            if initial_dynamics_feature is not None:
                initial_dynamics_feature = TensorUtils.repeat_by_expand_at(
                    initial_dynamics_feature, repeats=num_samples, dim=0
                )
            if history_state_dynamics is not None:
                history_state_dynamics = TensorUtils.repeat_by_expand_at(
                    history_state_dynamics, repeats=num_samples, dim=0
                )
            if input_actions_scaled is not None:
                input_actions_scaled = TensorUtils.repeat_by_expand_at(
                    input_actions_scaled, repeats=num_samples, dim=0
                )
            if len(aux_data) > 0:
                aux_data = TensorUtils.repeat_by_expand_at(
                    aux_data, repeats=num_samples, dim=0
                )

            # Do sampling
            if w_conditional < 1:
                w_conditional_samples = torch.linspace(
                    w_conditional, 1, num_samples, device=image_features.device
                )  # [N]
                w_conditional_samples = w_conditional_samples[None].repeat(
                    original_bsize, 1
                )  # [B, N]
                w_conditional_samples = w_conditional_samples.view(-1)
            else:
                w_conditional_samples = None

            # ---- Phase 1: sample actions ----
            # Guidance (when enabled) calls sample_dynamics internally at every
            # action denoising step to get a fully-denoised state prediction and
            # backpropagate ∂L_guidance / ∂x_t_actions through it.
            actions_scaled, guidance_losses_actions = self.sample_actions(
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps,
                history_state_actions=history_state_actions,
                w_advantage=w_advantage,
                w_conditional=w_conditional_samples,
                aux_data=aux_data,
                image_features_gripper=image_features_gripper,
                history_raymaps_gripper=history_raymaps_gripper,
                random_noise_scale=random_noise_scale,
                # Used for guidance
                enable_guidance=enable_guidance,
                initial_dynamics=initial_dynamics,
                initial_dynamics_feature=initial_dynamics_feature,
                history_state_dynamics=history_state_dynamics,
                drop_action_mask=None,
            )

            # ---- Phase 2: sample dynamics conditioned on the final actions ----
            # Always run when the world model is active and full inference is
            # requested.  When guidance was enabled in Phase 1, this gives the
            # clean dynamics output for the guided actions.
            if self.use_wm and not action_only:
                dynamics_scaled = self.sample_dynamics(
                    image_features=image_features,
                    language_features=language_features,
                    history_raymaps=history_raymaps,
                    input_actions=(
                        input_actions_scaled
                        if input_actions_scaled is not None
                        else actions_scaled
                    ),
                    initial_dynamics=initial_dynamics,
                    initial_dynamics_feature=initial_dynamics_feature,
                    history_state_dynamics=history_state_dynamics,
                    aux_data=aux_data,
                    image_features_gripper=image_features_gripper,
                    history_raymaps_gripper=history_raymaps_gripper,
                )
            else:
                dynamics_scaled = None

            # ---- eval_guidance: score the final dynamics prediction ----
            if eval_guidance and dynamics_scaled is not None:
                assert self.current_guidance is not None
                bsize_expanded = actions_scaled.shape[0]
                _, guidance_losses_actions = (
                    self.current_guidance.compute_guidance_loss(
                        dynamics_scaled,
                        torch.zeros(bsize_expanded, device=actions_scaled.device),
                        aux_data,
                    )
                )

            if guidance_losses_actions is not None:
                for k, v in guidance_losses_actions.items():
                    if k != "total_guidance_loss":
                        guidance_losses_actions[k] = v.reshape(
                            original_bsize, num_samples
                        )
                outputs["guidance_losses"] = guidance_losses_actions

            if self.action_predict_progress:
                # Extract progress before slicing (progress is at index self.action_dim)
                progress_scaled = actions_scaled[:, :, -1]  # [B*N, T]
                actions_scaled = actions_scaled[:, :, :-1]  # [B*N, T, action_dim]
                progress = (progress_scaled + 1.0) / 2.0  # [-1, 1] => [0, 1]
                progress = progress.clamp(0.0, 1.0)
                progress = progress.reshape(
                    original_bsize, num_samples, -1
                )  # [B, N, T]
                outputs["progress_predictions"] = progress

            actions = self.descale_action(
                actions_scaled,
                aux_data["action_norm_min_bound"],
                aux_data["action_norm_max_bound"],
                aux_data["action_mean"],
                aux_data["action_std"],
            )  # [B, T, D]
            if self.action_in_relative:
                actions = transform_two_hands_trajectory_relative_to_absolute(
                    actions, aux_data["start_pos"], has_finger_tips=False
                )  # [B, T, ACTION_DIM], [B, ACTION_DIM]
            actions = actions.reshape(
                original_bsize, num_samples, actions.shape[-2], actions.shape[-1]
            )  # [B, N, T, D]
            outputs["action_predictions"] = actions

            if self.use_wm and not action_only:
                assert dynamics_scaled is not None
                assert initial_dynamics is not None
                dynamics_geometric = dynamics_scaled[:, : self.dynamics_geometric_dim]
                dynamics_geometric = self.descale_state(
                    dynamics_geometric,
                    aux_data["state_norm_min_bound"],
                    aux_data["state_norm_max_bound"],
                    aux_data["state_mean"],
                    aux_data["state_std"],
                )  # [B, D, H, W]
                if self.dynamics_in_relative:
                    # initial_dynamics = history_state_dynamics.repeat(
                    #     1, dynamics_geometric.shape[1] // 3, 1, 1
                    # )  # [B, D, H, W]
                    history_state = data_batch["history_state"]  # [B, T*3, H, W]
                    history_state_last = rearrange(
                        history_state, "b (t c) h w -> b t c h w", c=3
                    )[
                        :, -1
                    ]  # [B, 3, H, W]
                    history_state_last = history_state_last.repeat(
                        1, dynamics_geometric.shape[1] // 3, 1, 1
                    )  # [B, D, H, W]
                    dynamics_geometric = dynamics_geometric + history_state_last
                _, _, Hg, Wg = dynamics_geometric.shape
                dynamics_geometric = dynamics_geometric.reshape(
                    original_bsize, num_samples, -1, Hg, Wg
                )
                outputs["dynamics_predictions"] = dynamics_geometric

                if self.dynamics_predict_visual:
                    dynamics_visual = dynamics_scaled[:, -self.dynamics_visual_dim :]
                    _, _, Hv, Wv = dynamics_visual.shape
                    dynamics_visual = dynamics_visual * 5.0
                    dynamics_visual = dynamics_visual.reshape(
                        original_bsize, num_samples, -1, Hv, Wv
                    )
                    outputs["visual_predictions"] = dynamics_visual

        # Compute values when value_only OR full inference (not action-only)
        if (
            self.use_vm
            and (value_only or not action_only)
            and input_actions is not None
            and input_dynamics is not None
        ):
            assert vm_action_features is not None

            prefix_image_value, prefix_language_value = (
                self.prepare_vision_language_features(
                    model="vm",
                    image_features=image_features,
                    language_features=language_features,
                    history_raymaps=history_raymaps,
                    image_features_gripper=image_features_gripper,
                    history_raymaps_gripper=history_raymaps_gripper,
                )
            )
            values = self.predict_values_step(
                action_state=vm_action_features,
                dynamics_state=input_dynamics,
                prefix=prefix_image_value,
                language=prefix_language_value,
            )
            values = values.reshape(
                original_bsize, num_samples, -1
            )  # [B, N, 1] scalar value per sample
            outputs["value_predictions"] = values
        return outputs

    # ------------------------------------------------------------------ #
    #  Decoupled sampling: sample_dynamics  +  sample_actions            #
    # ------------------------------------------------------------------ #

    def sample_dynamics(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
        history_raymaps: torch.Tensor,
        input_actions: torch.Tensor,
        initial_dynamics: torch.Tensor,
        initial_dynamics_feature: torch.Tensor,
        history_state_dynamics: torch.Tensor,
        aux_data: dict,
        noise_dynamics: torch.Tensor = None,
        drop_action_mask: torch.Tensor = None,
        image_features_gripper: torch.Tensor = None,
        history_raymaps_gripper: torch.Tensor = None,
        grad_withctx: Callable[[], AbstractContextManager] = torch.no_grad,
    ):
        """Run the full dynamics denoising loop conditioned on fixed input actions.

        Runs all ``config.num_steps`` denoising steps and returns the clean
        dynamics prediction.  When called with ``grad_withctx=torch.enable_grad``
        the output retains a computation graph w.r.t. ``input_actions``, which
        is required for guidance gradient computation.

        Args:
            input_actions: fixed action trajectory [B, H, D] used as condition.
            grad_withctx: gradient context for both the action preprocessing and
                the dynamics denoising steps.  Pass ``torch.enable_grad`` when
                the caller needs ``∂output / ∂input_actions``.

        Returns:
            x_t_dynamics: clean dynamics prediction [B, Dg, Hg, Wg].
        """
        assert self.use_wm, "sample_dynamics requires a world model (use_wm=True)"
        assert self.predict_x0, "sample_dynamics requires predict_x0=True"

        bsize = image_features.shape[0]
        device = image_features.device

        # Build WM vision-language prefix
        prefix_image, prefix_language = self.prepare_vision_language_features(
            model="wm",
            image_features=image_features,
            language_features=language_features,
            history_raymaps=history_raymaps,
            image_features_gripper=image_features_gripper,
            history_raymaps_gripper=history_raymaps_gripper,
        )
        prefix = torch.cat([prefix_image, prefix_language], dim=1)  # [B, L, D]

        # Initialise noise for dynamics (done outside grad context — no grad needed)
        if noise_dynamics is None:
            dynamics_shape = (
                bsize,
                self.dynamics_dim,
                self.config.dynamics_input_size,
                self.config.dynamics_input_size,
            )
            noise_dynamics = self.sample_noise(dynamics_shape, device)

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Everything from action preprocessing through the denoising Euler steps
        # must run inside the same grad context so that the full computation graph
        # from x_t_dynamics_clean → action_cond → input_actions is preserved
        # when grad_withctx=torch.enable_grad (required for guidance gradients).
        with grad_withctx():
            # Convert actions into the dynamics model's coordinate frame/scale.
            action_cond = self.prepare_action_for_dynamics_model(
                input_actions, aux_data=aux_data, grad_withctx=grad_withctx
            )

            x_t_dynamics = noise_dynamics
            time_dyn = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time_dyn >= -dt / 2:
                expanded_time = time_dyn.expand(bsize)
                pred_t_dynamics = self.denoise_dynamics_step(
                    prefix=prefix,
                    action=action_cond,
                    x_t=x_t_dynamics,
                    timestep=expanded_time,
                    initial_dynamics=initial_dynamics,
                    initial_dynamics_feature=initial_dynamics_feature,
                    history_state_dynamics=history_state_dynamics,
                    drop_action_mask=drop_action_mask,
                    grad_withctx=grad_withctx,
                )
                v_t_dynamics = (x_t_dynamics - pred_t_dynamics) / expanded_time[
                    :, None, None, None
                ]
                x_t_dynamics = x_t_dynamics + dt * v_t_dynamics
                # Non-inplace update: avoid invalidating the computation graph
                # that flows through expanded_time during backward.
                time_dyn = time_dyn + dt

        return x_t_dynamics

    def sample_actions(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
        history_raymaps: torch.Tensor,
        history_state_actions: torch.Tensor,
        noise_actions: torch.Tensor = None,
        w_advantage: float = 1.0,
        w_conditional: float = None,
        aux_data: dict = None,
        image_features_gripper: torch.Tensor = None,
        history_raymaps_gripper: torch.Tensor = None,
        random_noise_scale: float = 0.0,
        # Guidance via full dynamics sampling — only used when enable_guidance=True
        enable_guidance: bool = False,
        initial_dynamics: torch.Tensor = None,
        initial_dynamics_feature: torch.Tensor = None,
        history_state_dynamics: torch.Tensor = None,
        drop_action_mask: torch.Tensor = None,
    ):
        """Sample actions with optional guidance through full dynamics denoising.

        When ``enable_guidance=True``, at **each** action denoising step the
        complete dynamics denoising loop (``sample_dynamics``) is called with the
        current ``x_t_actions`` as a differentiable condition.  The resulting
        clean dynamics prediction is used to compute the guidance gradient
        ``∂L_guidance(dynamics_clean) / ∂x_t_actions``, which is then subtracted
        from ``x_t_actions`` before the next action step.

        This differs from the joint mode in ``sample_scaled_actions_and_dynamics``
        where a *single* noisy dynamics denoising step is used for guidance.  Here
        the dynamics are **fully denoised** at every action step, yielding higher-
        quality guidance at the cost of ``num_steps`` × more dynamics forward passes.

        Returns:
            x_t_actions: clean action prediction [B, H, D]
            guidance_losses: dict of guidance loss scalars (empty when not guided)
        """
        bsize = image_features.shape[0]
        device = image_features.device

        # Sample or use caller-provided initial noise
        if noise_actions is None:
            actions_shape = (bsize, self.action_chunk_size, self.action_output_dim)
            noise_actions = self.sample_noise(actions_shape, device)

        # Build VLA vision-language prefix (conditional)
        prefix_image_action, prefix_language_action = (
            self.prepare_vision_language_features(
                model="vla",
                image_features=image_features,
                language_features=language_features,
                history_raymaps=history_raymaps,
                image_features_gripper=image_features_gripper,
                history_raymaps_gripper=history_raymaps_gripper,
            )
        )

        # Build unconditional prefix (for CFG / w_conditional)
        if w_conditional is not None:
            prefix_image_action_uncond, prefix_language_action_uncond = (
                self.prepare_vision_language_features(
                    model="vla",
                    image_features=aux_data["history_visual_feature_patch_null"],
                    language_features=aux_data["language_feature_null"],
                    history_raymaps=aux_data["history_raymap"],
                    image_features_gripper=aux_data.get(
                        "history_visual_feature_patch_gripper_null", None
                    ),
                    history_raymaps_gripper=aux_data.get(
                        "history_raymap_gripper", None
                    ),
                )
            )
            if "history_state_actions_null" in aux_data:
                history_state_actions_null = aux_data["history_state_actions_null"]
            else:
                history_state_actions_null = history_state_actions
        else:
            prefix_image_action_uncond = prefix_image_action
            prefix_language_action_uncond = prefix_language_action
            history_state_actions_null = history_state_actions

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t_actions = noise_actions
        guidance_losses_actions: dict = {}
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        curr_step = 0

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # ----- Action denoising step (same branching as the joint sampler) -----
            if w_advantage < 1.0:
                advantage = (
                    torch.cat([torch.ones(bsize), -torch.ones(bsize)]).long().to(device)
                )
                pred_t_actions = self.denoise_actions_step(
                    prefix=torch.cat([prefix_image_action, prefix_image_action], dim=0),
                    language=torch.cat(
                        [prefix_language_action, prefix_language_action], dim=0
                    ),
                    x_t=torch.cat([x_t_actions, x_t_actions], dim=0),
                    timestep=torch.cat([expanded_time, expanded_time], dim=0),
                    history_state_actions=torch.cat(
                        [history_state_actions, history_state_actions], dim=0
                    ),
                    advantage=advantage,
                )
                pred_t_actions_pos, pred_t_actions_uncond = (
                    pred_t_actions[:bsize],
                    pred_t_actions[bsize:],
                )
                if self.predict_x0:
                    v_pos = (x_t_actions - pred_t_actions_pos) / expanded_time[
                        :, None, None
                    ]
                    v_unc = (x_t_actions - pred_t_actions_uncond) / expanded_time[
                        :, None, None
                    ]
                else:
                    v_pos, v_unc = pred_t_actions_pos, pred_t_actions_uncond
                v_t_actions = v_pos * w_advantage + v_unc * (1 - w_advantage)

            else:
                advantage = torch.ones(bsize).long().to(device)
                if w_conditional is not None:
                    pred_t_actions_full = self.denoise_actions_step(
                        prefix=torch.cat(
                            [prefix_image_action, prefix_image_action_uncond], dim=0
                        ),
                        language=torch.cat(
                            [prefix_language_action, prefix_language_action_uncond],
                            dim=0,
                        ),
                        x_t=torch.cat([x_t_actions, x_t_actions], dim=0),
                        timestep=torch.cat([expanded_time, expanded_time], dim=0),
                        history_state_actions=torch.cat(
                            [history_state_actions, history_state_actions_null], dim=0
                        ),
                        advantage=torch.cat([advantage, advantage], dim=0),
                    )
                    pred_t_actions_cond, pred_t_actions_uncond = (
                        pred_t_actions_full[:bsize],
                        pred_t_actions_full[bsize:],
                    )
                    if self.predict_x0:
                        v_cond = (x_t_actions - pred_t_actions_cond) / expanded_time[
                            :, None, None
                        ]
                        v_unc = (x_t_actions - pred_t_actions_uncond) / expanded_time[
                            :, None, None
                        ]
                    else:
                        v_cond, v_unc = pred_t_actions_cond, pred_t_actions_uncond
                    delta = v_cond - v_unc
                    v_t_actions = v_unc + w_conditional[:, None, None] * delta
                else:
                    pred_t_actions = self.denoise_actions_step(
                        prefix=prefix_image_action,
                        language=prefix_language_action,
                        x_t=x_t_actions,
                        timestep=expanded_time,
                        history_state_actions=history_state_actions,
                        advantage=advantage,
                    )
                    if self.predict_x0:
                        v_t_actions = (x_t_actions - pred_t_actions) / expanded_time[
                            :, None, None
                        ]
                    else:
                        v_t_actions = pred_t_actions

            # Euler step
            _rns = (
                random_noise_scale if curr_step < int(self.config.num_steps) - 1 else 0
            )
            random_noise = torch.randn_like(v_t_actions) * torch.sqrt(-dt) * _rns
            x_t_actions = x_t_actions + dt * v_t_actions + random_noise

            # ----- Guidance via full dynamics denoising -----
            # At this action denoising step, run the complete dynamics denoising
            # loop (all num_steps) conditioned on x_t_actions to obtain a clean
            # dynamics prediction, then compute ∂guidance_loss / ∂x_t_actions.
            if enable_guidance:
                assert self.current_guidance is not None
                assert (
                    initial_dynamics is not None and history_state_dynamics is not None
                )

                # Create a leaf tensor so autograd can compute the gradient
                x_t_actions_leaf = x_t_actions.detach().clone().requires_grad_(True)

                # Full dynamics denoising — gradients flow through action_cond
                x_t_dynamics_clean = self.sample_dynamics(
                    image_features=image_features,
                    language_features=language_features,
                    history_raymaps=history_raymaps,
                    input_actions=x_t_actions_leaf,
                    initial_dynamics=initial_dynamics,
                    initial_dynamics_feature=initial_dynamics_feature,
                    history_state_dynamics=history_state_dynamics,
                    aux_data=aux_data,
                    drop_action_mask=drop_action_mask,
                    image_features_gripper=image_features_gripper,
                    history_raymaps_gripper=history_raymaps_gripper,
                    grad_withctx=torch.enable_grad,  # keep graph for guidance grad
                )

                # ∂L_guidance(dynamics_clean) / ∂x_t_actions_leaf
                guide_grad, guidance_losses_actions = self.guidance(
                    x_t_dynamics_clean,
                    expanded_time,
                    aux_data,
                    return_grad_of=x_t_actions_leaf,
                )
                x_t_actions = x_t_actions - guide_grad

            time += dt
            curr_step += 1

        return x_t_actions, guidance_losses_actions

    def guidance(self, x, t, aux_data_batch, return_grad_of=None):
        """Compute the guidance gradient.

        The gradient is ∂L(x) / ∂target, flowing through:
            tot_loss → compute_guidance_loss(x) → x → (computation graph) → target

        IMPORTANT: the caller is responsible for ensuring that `x` was computed
        with gradients enabled (e.g. via torch.enable_grad) so that a computation
        graph connecting `x` to `target` exists before this function is called.
        The `with torch.enable_grad()` block here only covers the guidance loss
        computation on top of `x`; it cannot retroactively build a graph for `x`
        itself.

        Args:
            x: tensor on which the guidance loss is evaluated (e.g. pred_t_dynamics)
            t: diffusion timestep
            aux_data_batch: auxiliary data passed to the guidance module
            return_grad_of: if not None, differentiate w.r.t. this tensor instead
                of x.  Must be a leaf tensor with requires_grad=True that x depends
                on through the computation graph.
        """
        assert self.current_guidance is not None

        target = x if return_grad_of is None else return_grad_of
        assert target.requires_grad, (
            "guidance() target must have requires_grad=True. "
            "Make sure x (pred_t_dynamics) was computed under torch.enable_grad "
            "and target was created with .requires_grad_()"
        )

        with torch.enable_grad():
            tot_loss, guidance_losses = self.current_guidance.compute_guidance_loss(
                x, t, aux_data_batch
            )

            # ∂tot_loss / ∂target flows through:
            #   tot_loss → compute_guidance_loss(x) → x → dynamics_model(action_cond)
            #            → prepare_action_for_dynamics_model → target (x_t_actions)
            (guide_grad,) = torch.autograd.grad(
                tot_loss,
                target,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )

        guidance_losses["total_guidance_loss"] = tot_loss
        return guide_grad.detach(), guidance_losses

    def denoise_actions_step(
        self,
        prefix: torch.Tensor,
        language: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        history_state_actions: torch.Tensor,
        advantage: torch.Tensor = None,
        drop_language_mask: torch.Tensor = None,
        grad_withctx: Callable[[], AbstractContextManager] = torch.no_grad,
    ):
        """Apply one denoising step of the noise `x_t` for actions at a given timestep."""
        with grad_withctx():
            x_t, c = self.action_model(
                x=x_t,
                x_history=history_state_actions,
                context=prefix,
                language=language,
                timestamp=timestep,
                advantage=advantage,
                drop_language_mask=drop_language_mask,
            )
            x_t = x_t.to(dtype=torch.float32)
            c = c.to(dtype=torch.float32)
            pred_t_actions = self.action_out_proj(x_t, c)
        return pred_t_actions

    def denoise_dynamics_step(
        self,
        prefix,
        action,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        initial_dynamics: torch.Tensor,
        initial_dynamics_feature: torch.Tensor,
        history_state_dynamics: torch.Tensor,
        drop_action_mask: torch.Tensor = None,
        grad_withctx: Callable[[], AbstractContextManager] = torch.no_grad,
    ):
        """Apply one denoising step of the noise `x_t` for dynamics at a given timestep."""
        with grad_withctx():
            x_t = torch.cat([initial_dynamics, x_t], dim=1)  # [B, D+3, H, W]
            if self.wm_concat_dinov3_feature:
                x_t = torch.cat(
                    [initial_dynamics_feature, x_t], dim=1
                )  # [B, D+3+768, H, W]
            x_t, c = self.dynamics_model(
                x=x_t,
                x_history=history_state_dynamics,
                context=prefix,
                timestamp=timestep,
                action=action,
                # drop_action_mask=drop_action_mask,
            )
            x_t = x_t.to(dtype=torch.float32)
            c = c.to(dtype=torch.float32)
            pred_t_dynamics = self.dynamics_out_proj(x_t, c)
            pred_t_dynamics = self.unpatchify(
                pred_t_dynamics,
                c=self.dynamics_geometric_dim,
                p=self.config.dynamics_patch_factor,
            )

            # if self.dynamics_predict_distance_to_goal:
            #     pred_t_dynamics_dist2goal = self.dynamics_dist2goal_proj(x_t, c)
            #     pred_t_dynamics_dist2goal = self.unpatchify(
            #         pred_t_dynamics_dist2goal, c=3, p=self.config.dynamics_patch_factor
            #     )
            #     pred_t_dynamics = torch.cat(
            #         [pred_t_dynamics, pred_t_dynamics_dist2goal], dim=1
            #     )

            if self.dynamics_predict_visual:
                pred_t_dynamics_visual = self.dynamics_visual_proj(x_t, c)
                pred_t_dynamics_visual = self.unpatchify(
                    pred_t_dynamics_visual, c=self.feature_dim, p=1
                )
                pred_t_dynamics_visual = F.interpolate(
                    pred_t_dynamics_visual,
                    size=(
                        self.config.dynamics_input_size,
                        self.config.dynamics_input_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_t_dynamics = torch.cat(
                    [pred_t_dynamics, pred_t_dynamics_visual], dim=1
                )
        return pred_t_dynamics

    def predict_values_step(
        self,
        action_state: torch.Tensor,
        dynamics_state: torch.Tensor,
        prefix: torch.Tensor,
        language: torch.Tensor,
        grad_withctx: Callable[[], AbstractContextManager] = torch.no_grad,
    ):
        """Apply one denoising step of the noise `x_t` for actions at a given timestep."""
        with grad_withctx():
            v = self.value_query.expand(prefix.shape[0], -1, -1)
            dynamics_features_value = self.dynamics_proj_for_vm(dynamics_state)
            context_vm = torch.cat([prefix, dynamics_features_value], dim=1)
            v, c = self.value_model(
                x=v,
                x_history=action_state,
                context=context_vm,
                language=language,
                timestamp=torch.zeros(
                    prefix.shape[0], device=prefix.device, dtype=torch.float32
                ),
            )
            v = v.to(dtype=torch.float32)
            v = self.vm_out_proj(v, c).squeeze((-1, -2))
        return v

    # def sample_scaled_actions_and_dynamics(
    #     self,
    #     image_features: torch.Tensor,
    #     language_features: torch.Tensor,
    #     history_raymaps: torch.Tensor,
    #     image_features_gripper: torch.Tensor = None,
    #     history_raymaps_gripper: torch.Tensor = None,
    #     initial_action: torch.Tensor = None,
    #     initial_dynamics: torch.Tensor = None,
    #     initial_dynamics_feature: torch.Tensor = None,
    #     history_state_actions: torch.Tensor = None,
    #     history_state_dynamics: torch.Tensor = None,
    #     noise_actions: torch.Tensor = None,
    #     noise_dynamics: torch.Tensor = None,
    #     drop_action_mask: torch.Tensor = None,
    #     w_advantage: float = 1.0,
    #     w_conditional: float = None,
    #     aux_data: dict = None,
    #     enable_guidance: bool = False,
    #     eval_guidance: bool = False,
    #     random_noise_scale: float = 0.0,
    #     decouple_action_dynamics: bool = False,
    # ):
    #     """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

    #     Args:
    #         decouple_action_dynamics: if True, run two separate denoising loops —
    #             Phase 1 samples actions only (no dynamics), Phase 2 samples dynamics
    #             conditioned on the final clean actions from Phase 1.  Guidance is
    #             disabled in this mode.
    #     """
    #     bsize = image_features.shape[0]
    #     device = image_features.device

    #     # Decoupled mode: actions first, then dynamics — guidance not applicable.
    #     if decouple_action_dynamics:
    #         assert not enable_guidance, "guidance not supported in decouple_action_dynamics mode"

    #     grad_withctx = torch.enable_grad if enable_guidance else torch.no_grad
    #     dynamics_cond_on_input_actions = False
    #     # In decoupled mode the first loop is action-only regardless of other flags.
    #     inference_action_only = (
    #         True
    #         if (decouple_action_dynamics or
    #             (initial_dynamics is None and not enable_guidance))
    #         else False
    #     )
    #     if noise_actions is None:
    #         actions_shape = (bsize, self.action_chunk_size, self.action_output_dim)
    #         noise_actions = self.sample_noise(actions_shape, device)
    #     else:
    #         dynamics_cond_on_input_actions = True

    #     if noise_dynamics is None and self.use_wm:
    #         dynamics_shape = (
    #             bsize,
    #             self.dynamics_dim,
    #             self.config.dynamics_input_size,
    #             self.config.dynamics_input_size,
    #         )
    #         noise_dynamics = self.sample_noise(dynamics_shape, device)

    #     # Acquire the prefix for the actions
    #     prefix_image_action, prefix_language_action = (
    #         self.prepare_vision_language_features(
    #             model="vla",
    #             image_features=image_features,
    #             language_features=language_features,
    #             history_raymaps=history_raymaps,
    #             image_features_gripper=image_features_gripper,
    #             history_raymaps_gripper=history_raymaps_gripper,
    #         )
    #     )
    #     if self.use_wm and not decouple_action_dynamics:
    #         prefix_image_dynamics, prefix_language_dynamics = (
    #             self.prepare_vision_language_features(
    #                 model="wm",
    #                 image_features=image_features,
    #                 language_features=language_features,
    #                 history_raymaps=history_raymaps,
    #                 image_features_gripper=image_features_gripper,
    #                 history_raymaps_gripper=history_raymaps_gripper,
    #             )
    #         )
    #         prefix_dynamics = torch.cat(
    #             [prefix_image_dynamics, prefix_language_dynamics], dim=1
    #         )

    #     if w_conditional is not None:
    #         prefix_image_action_uncond, prefix_language_action_uncond = (
    #             self.prepare_vision_language_features(
    #                 model="vla",
    #                 image_features=aux_data["history_visual_feature_patch_null"],
    #                 language_features=aux_data["language_feature_null"],
    #                 history_raymaps=aux_data["history_raymap"],
    #                 image_features_gripper=aux_data.get(
    #                     "history_visual_feature_patch_gripper_null", None
    #                 ),
    #                 history_raymaps_gripper=aux_data.get(
    #                     "history_raymap_gripper_null", None
    #                 ),
    #             )
    #         )
    #     else:
    #         prefix_language_action_uncond = prefix_language_action
    #         prefix_image_action_uncond = prefix_image_action

    #     # Prepare th time scheduler for denosing
    #     dt = -1.0 / self.config.num_steps
    #     dt = torch.tensor(dt, dtype=torch.float32, device=device)

    #     # Denoise the actions
    #     x_t_actions = noise_actions
    #     x_t_dynamics = noise_dynamics

    #     time = torch.tensor(1.0, dtype=torch.float32, device=device)
    #     curr_step = 0
    #     while time >= -dt / 2:
    #         expanded_time = time.expand(bsize)

    #         # Detach the gradient as we need to compute the gradient of the guidance
    #         # loss w.r.t. the input trajectory
    #         # if enable_guidance:
    #         #     x_t_actions = x_t_actions.detach().clone().requires_grad_()

    #         # Denoise the actions, we enable gradient if we do guidance
    #         if w_advantage < 1.0:
    #             advantage = torch.cat(
    #                 [torch.ones(bsize), -torch.ones(bsize)]
    #             ).long()  # positive conditional + unconditional
    #             advantage = advantage.to(device)
    #             pred_t_actions = self.denoise_actions_step(
    #                 prefix=torch.cat([prefix_image_action, prefix_image_action], dim=0),
    #                 language=torch.cat(
    #                     [prefix_language_action, prefix_language_action], dim=0
    #                 ),
    #                 x_t=torch.cat([x_t_actions, x_t_actions], dim=0),
    #                 timestep=torch.cat([expanded_time, expanded_time], dim=0),
    #                 history_state_actions=torch.cat(
    #                     [history_state_actions, history_state_actions], dim=0
    #                 ),
    #                 advantage=advantage,
    #                 grad_withctx=grad_withctx,
    #             )
    #             # Split: pred_t_actions[:bsize]=positive cond, [bsize:]=uncond
    #             (
    #                 pred_t_actions,
    #                 pred_t_actions_advantage_uncond,
    #             ) = (
    #                 pred_t_actions[:bsize],
    #                 pred_t_actions[bsize:],
    #             )
    #             assert pred_t_actions.shape == pred_t_actions_advantage_uncond.shape
    #             if self.predict_x0:
    #                 v_t_actions_advantage_positive_cond = (
    #                     x_t_actions - pred_t_actions
    #                 ) / expanded_time[:, None, None]
    #                 v_t_actions_advantage_uncond = (
    #                     x_t_actions - pred_t_actions_advantage_uncond
    #                 ) / expanded_time[:, None, None]
    #             else:
    #                 v_t_actions_advantage_positive_cond = pred_t_actions
    #                 v_t_actions_advantage_uncond = pred_t_actions_advantage_uncond

    #             # Weighted sum of velocities
    #             v_t_actions = (
    #                 v_t_actions_advantage_positive_cond * w_advantage
    #                 + v_t_actions_advantage_uncond * (1 - w_advantage)
    #             )

    #         else:
    #             advantage = torch.ones(bsize).long().to(device)  # Positive conditional
    #             if w_conditional is not None:

    #                 # history_state_actions_uncond = (
    #                 #     torch.ones_like(history_state_actions) * -1
    #                 # )

    #                 if enable_guidance:
    #                     pred_t_actions = self.denoise_actions_step(
    #                         prefix=prefix_image_action,
    #                         language=prefix_language_action,
    #                         x_t=x_t_actions,
    #                         timestep=expanded_time,
    #                         history_state_actions=history_state_actions,
    #                         advantage=advantage,
    #                         grad_withctx=grad_withctx,
    #                     )

    #                     pred_t_actions_uncond = self.denoise_actions_step(
    #                         prefix=prefix_image_action_uncond,
    #                         language=prefix_language_action_uncond,
    #                         x_t=x_t_actions.clone().detach(),
    #                         timestep=expanded_time,
    #                         # history_state_actions=history_state_actions_uncond,
    #                         history_state_actions=history_state_actions,
    #                         advantage=advantage,
    #                         grad_withctx=grad_withctx,
    #                     )
    #                 else:
    #                     pred_t_actions_full = self.denoise_actions_step(
    #                         prefix=torch.cat(
    #                             [prefix_image_action, prefix_image_action_uncond],
    #                             dim=0,
    #                         ),
    #                         language=torch.cat(
    #                             [prefix_language_action, prefix_language_action_uncond],
    #                             dim=0,
    #                         ),
    #                         x_t=torch.cat([x_t_actions, x_t_actions], dim=0),
    #                         timestep=torch.cat([expanded_time, expanded_time], dim=0),
    #                         history_state_actions=torch.cat(
    #                             [
    #                                 history_state_actions,
    #                                 # history_state_actions_uncond,
    #                                 history_state_actions,
    #                             ],
    #                             dim=0,
    #                         ),
    #                         advantage=torch.cat([advantage, advantage], dim=0),
    #                         grad_withctx=grad_withctx,
    #                     )
    #                     pred_t_actions, pred_t_actions_uncond = (
    #                         pred_t_actions_full[:bsize],
    #                         pred_t_actions_full[bsize:],
    #                     )
    #                 assert pred_t_actions.shape == pred_t_actions_uncond.shape
    #                 if self.predict_x0:
    #                     v_t_actions_cond = (
    #                         x_t_actions - pred_t_actions
    #                     ) / expanded_time[:, None, None]
    #                     v_t_actions_uncond = (
    #                         x_t_actions - pred_t_actions_uncond
    #                     ) / expanded_time[:, None, None]
    #                 else:
    #                     v_t_actions_cond = pred_t_actions
    #                     v_t_actions_uncond = pred_t_actions_uncond

    #                 # v_t_actions = (
    #                 #     v_t_actions_cond * w_conditional[:, None, None]
    #                 #     + v_t_actions_uncond * (1 - w_conditional[:, None, None])
    #                 # )
    #                 delta = v_t_actions_cond - v_t_actions_uncond  # [B, H, D]
    #                 dot = (delta * v_t_actions_uncond).sum(dim=-1, keepdim=True)
    #                 norm_sq = (v_t_actions_uncond * v_t_actions_uncond).sum(
    #                     dim=-1, keepdim=True
    #                 ) + 1e-6
    #                 proj = (dot / norm_sq) * v_t_actions_uncond
    #                 delta_perp = delta - proj
    #                 v_t_actions = (
    #                     v_t_actions_uncond + w_conditional[:, None, None] * delta
    #                 )
    #                 # v_t_actions = v_t_actions_uncond + w_conditional[:, None, None] * delta_perp

    #             else:
    #                 pred_t_actions = self.denoise_actions_step(
    #                     prefix=prefix_image_action,
    #                     language=prefix_language_action,
    #                     x_t=x_t_actions,
    #                     timestep=expanded_time,
    #                     history_state_actions=history_state_actions,
    #                     advantage=advantage,
    #                     grad_withctx=grad_withctx,
    #                 )

    #                 if self.predict_x0:
    #                     v_t_actions = (x_t_actions - pred_t_actions) / expanded_time[
    #                         :, None, None
    #                     ]
    #                 else:
    #                     v_t_actions = pred_t_actions

    #         # Euler step
    #         random_noise_scale = (
    #             random_noise_scale if curr_step < int(self.config.num_steps) - 1 else 0
    #         )
    #         random_noise = (
    #             torch.randn_like(v_t_actions) * torch.sqrt(-dt) * random_noise_scale
    #         )
    #         x_t_actions = x_t_actions + dt * v_t_actions + random_noise

    #         # Denoise the dynamics, we enable gradient if we do guidance
    #         if not inference_action_only:
    #             assert history_state_dynamics is not None
    #             assert self.use_wm and self.predict_x0

    #             if enable_guidance:
    #                 x_t_actions = x_t_actions.detach().clone().requires_grad_()

    #             action_cond = (
    #                 x_t_actions if not dynamics_cond_on_input_actions else noise_actions
    #             )
    #             action_cond = self.prepare_action_for_dynamics_model(
    #                 action_cond, aux_data=aux_data, grad_withctx=grad_withctx
    #             )
    #             pred_t_dynamics = self.denoise_dynamics_step(
    #                 prefix=prefix_dynamics,
    #                 action=action_cond,
    #                 x_t=x_t_dynamics,
    #                 timestep=expanded_time,
    #                 initial_dynamics=initial_dynamics,
    #                 initial_dynamics_feature=initial_dynamics_feature,
    #                 history_state_dynamics=history_state_dynamics,
    #                 drop_action_mask=drop_action_mask,
    #                 grad_withctx=grad_withctx,
    #             )

    #             if self.predict_x0:
    #                 v_t_dynamics = (x_t_dynamics - pred_t_dynamics) / expanded_time[
    #                     :, None, None, None
    #                 ]
    #             else:
    #                 v_t_dynamics = pred_t_dynamics
    #         else:
    #             v_t_dynamics = None  # action-only or decoupled mode: skip dynamics

    #         # Apply action guidance here using the dynamics model
    #         # Here we are doing clean guidance on the dynamics:
    #         # Action model: pred_t_actions = f_act(x_t_actions)
    #         # Dynamics model: pred_t_dynamics = f_dyn(x_t_dynamics, pred_t_actions)
    #         # Thus the loss term is evaluated on the pred_t_dynamics (clean prediction), and we
    #         # acquire the gradient of the guidance loss w.r.t. the input actions x_t_actions

    #         if enable_guidance:
    #             guide_grad_actions, guidance_losses_actions = self.guidance(
    #                 pred_t_dynamics,
    #                 expanded_time,
    #                 aux_data,
    #                 return_grad_of=x_t_actions,
    #             )
    #             x_t_actions = x_t_actions - guide_grad_actions

    #             # Check the norm of the guide_grad
    #             # print(f"Norm of the guide_grad: {guide_grad_actions.norm()}")
    #             # print(f"Norm of the v_t_actions: {(v_t_actions * dt).norm()}")
    #             # print(f"================================================")

    #         else:
    #             guidance_losses_actions = dict()

    #         # # Euler step
    #         # random_noise_scale = (
    #         #     random_noise_scale if curr_step < int(self.config.num_steps) - 1 else 0
    #         # )
    #         # random_noise = (
    #         #     torch.randn_like(v_t_actions) * torch.sqrt(-dt) * random_noise_scale
    #         # )
    #         # x_t_actions = x_t_actions + dt * v_t_actions + random_noise

    #         if v_t_dynamics is not None:  # joint mode only (not action-only / decoupled)
    #             x_t_dynamics = x_t_dynamics + dt * v_t_dynamics

    #         # Time update
    #         time += dt
    #         curr_step += 1

    #     # ------------------------------------------------------------------ #
    #     # Phase 2 (decoupled mode): sample dynamics conditioned on the final  #
    #     # clean actions obtained from Phase 1 above.                          #
    #     # ------------------------------------------------------------------ #
    #     if decouple_action_dynamics and self.use_wm:
    #         assert history_state_dynamics is not None
    #         assert initial_dynamics is not None
    #         assert self.predict_x0, "decouple_action_dynamics requires predict_x0=True"

    #         # Build the WM prefix (vision + language)
    #         prefix_image_dynamics, prefix_language_dynamics = (
    #             self.prepare_vision_language_features(
    #                 model="wm",
    #                 image_features=image_features,
    #                 language_features=language_features,
    #                 history_raymaps=history_raymaps,
    #                 image_features_gripper=image_features_gripper,
    #                 history_raymaps_gripper=history_raymaps_gripper,
    #             )
    #         )
    #         prefix_dynamics = torch.cat(
    #             [prefix_image_dynamics, prefix_language_dynamics], dim=1
    #         )

    #         # Convert the final sampled actions into the dynamics model's scale
    #         action_cond_for_dynamics = self.prepare_action_for_dynamics_model(
    #             x_t_actions, aux_data=aux_data, grad_withctx=torch.no_grad
    #         )

    #         # Fresh noise for dynamics
    #         if noise_dynamics is None:
    #             dynamics_shape = (
    #                 bsize,
    #                 self.dynamics_dim,
    #                 self.config.dynamics_input_size,
    #                 self.config.dynamics_input_size,
    #             )
    #             noise_dynamics = self.sample_noise(dynamics_shape, device)

    #         x_t_dynamics = noise_dynamics
    #         time_dyn = torch.tensor(1.0, dtype=torch.float32, device=device)
    #         while time_dyn >= -dt / 2:
    #             expanded_time_dyn = time_dyn.expand(bsize)
    #             pred_t_dynamics = self.denoise_dynamics_step(
    #                 prefix=prefix_dynamics,
    #                 action=action_cond_for_dynamics,
    #                 x_t=x_t_dynamics,
    #                 timestep=expanded_time_dyn,
    #                 initial_dynamics=initial_dynamics,
    #                 initial_dynamics_feature=initial_dynamics_feature,
    #                 history_state_dynamics=history_state_dynamics,
    #                 drop_action_mask=drop_action_mask,
    #                 grad_withctx=torch.no_grad,
    #             )
    #             v_t_dynamics = (
    #                 (x_t_dynamics - pred_t_dynamics) / expanded_time_dyn[:, None, None, None]
    #             )
    #             x_t_dynamics = x_t_dynamics + dt * v_t_dynamics
    #             time_dyn += dt

    #     if eval_guidance:
    #         tot_loss, guidance_losses_actions = (
    #             self.current_guidance.compute_guidance_loss(
    #                 x_t_dynamics,
    #                 expanded_time,
    #                 aux_data,
    #             )
    #         )
    #     else:
    #         guidance_losses_actions = dict()

    #     return (
    #         x_t_actions,
    #         x_t_dynamics,
    #         guidance_losses_actions,
    #     )

    # Hack to compute the guidance loss w.r.t. the input actions
    # x_t_dynamics_descaled = self.descale_state(
    #     x_t_dynamics,
    #     aux_data["state_norm_min_bound"],
    #     aux_data["state_norm_max_bound"],
    #     aux_data["state_mean"],
    #     aux_data["state_std"],
    # )  # [B, D, H, W]
    # gt_state_residual = aux_data["gt_state_residual"]  # [B, D, H, W]
    # x_t_dynamics_descaled = rearrange(
    #     x_t_dynamics_descaled, "b (t c) h w -> b t c h w", c=3
    # )
    # gt_state_residual = rearrange(
    #     gt_state_residual, "b (t c) h w -> b t c h w", c=3
    # )
    # error_dynamics = (
    #     (x_t_dynamics_descaled - gt_state_residual).pow(2).sum(dim=(-3))
    # )  # [B, T, H, W]
    # error_dynamics = error_dynamics[:, :, -1:]  # [B, T, H, W]
    # error_dynamics = error_dynamics.mean(dim=(-1, -2, -3)).pow(0.5).float()  # [B, T]
    # guidance_losses_actions["dummy_dynamics_guidance_loss"] = error_dynamics


if __name__ == "__main__":
    config = edict.EasyDict(
        {
            "mode": "vla+wm+vm",
            "dtype": "torch.float16",
            "horizon": 15,
            "num_steps": 10,
            "action_dim": 6,
            "device": "cuda",
            "dynamics_dim": 45,
            "dynamics_input_size": 64,
            "dynamics_patch_factor": 2,
            "max_period": 4.0,
            "min_period": 0.004,
            "num_am_layers": 12,
            "num_am_heads": 12,
            "num_am_cond_layers": 4,
            "num_wm_heads": 12,
            "num_wm_layers": 12,
            "wm_width_multiplier": 0.5,
            "am_width_multiplier": 0.5,
            "predict_x0": True,
            "wm_concat_dinov3_feature": True,
            "dinov3_upsampling_factor": 1.5,
            "vm_type": "abs_pos_transformer",
            "vm_width_multiplier": 0.5,
            "num_vm_layers": 12,
            "num_vm_heads": 12,
            "wm_predict_visual": True,
        }
    )

    model = VLAFlowMatching(config)
    # Example data batch for DiffuserModel
    n_state_tokens = 15
    T = 8
    patch_size = 64
    state_dim = 45
    data_batch = {
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
        # Required conditional features (these get concatenated)
        "history_action": torch.randn(
            1, T, n_state_tokens, 6
        ),  # [B, history_action_dim] - history action features
        "language_tokens": torch.randint(0, 10000, (1, T, 30)),
        "language_mask": torch.ones(1, T, 30, dtype=torch.bool),
        "color": torch.randn(1, T, 3, 512, 512),  # [B, C, H, W] - color images
        "history_color_frames": torch.randn(
            1, T, 5, 3, 512, 512
        ),  # [B, H, C, H, W] - history color images
        "action_valid": torch.ones(1, T, 15, 6),  # [B, H, 6] - action valid
        "state_mean": torch.tensor(
            torch.zeros(1, T, state_dim)
        ),  # [B, 3] - mean for state
        "state_std": torch.tensor(
            torch.ones(1, T, state_dim)
        ),  # [B, 3] - std for state
        "state_norm_max_bound": torch.tensor(
            torch.ones(1, T, state_dim)
        ),  # [B, 3] - max bound for state
        "state_norm_min_bound": torch.tensor(
            torch.zeros(1, state_dim)
        ),  # [B, 3] - min bound for state
        "state_valid": torch.randn(1, T, state_dim, patch_size, patch_size) > 0.5,
        "start_state": torch.randn(1, T, 3, patch_size, patch_size),
        "gt_state_residual": torch.randn(1, T, state_dim, patch_size, patch_size),
        "state_visib": torch.randn(1, T, state_dim // 3, patch_size, patch_size),
        "history_state": torch.randn(1, T, state_dim, patch_size, patch_size),
        "history_raymap": torch.randn(1, T, 15, 6, 14, 14),
        "start_state_dinov3_feature": torch.randn(1, T, 768, patch_size, patch_size),
        "history_visual_feature_patch": torch.randn(1, T, 15, 196, 768),
        "language_feature": torch.randn(1, T, 30, 768),
        "goal_visual_feature_patch": torch.randn(1, T, 196, 768),
        "goal_visual_feature": torch.randn(1, 196, 768),
        "advantage_label": torch.full((1, T), -1, dtype=torch.long),
    }
    model.cuda()
    data_batch = {k: v.cuda() for k, v in data_batch.items()}
    data_batch = TensorUtils.join_dimensions(data_batch, begin_axis=0, end_axis=2)
    # outputs = model(data_batch)
    losses = model.compute_losses(data_batch)
    # print(losses)
    # print(model.dynamics_model)
    for _ in range(10):
        with torch.no_grad():
            start_time = time.time()
            outputs = model(data_batch)
            for k, v in outputs.items():
                print(k, v.shape)
            end_time = time.time()
            for k, v in outputs.items():
                print(k, v.shape)
            print(f"Time taken: {end_time - start_time} seconds!")

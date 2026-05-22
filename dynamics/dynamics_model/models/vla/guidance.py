import torch
from torch import nn
import torch.nn.functional as F
from models.layers_2d import Project3D, BackprojectDepth
import utils.dataset_utils as DatasetUtils
import open3d as o3d
from einops import rearrange
import viser
import numpy as np


class Guidance:
    def __init__(self, scale=1.0):
        self.scale = scale

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
        action = action.clamp(-1, 1)
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

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        return loss_tot, guide_losses


class ActionRegressionGuidance(Guidance):
    def __init__(self, scale=1.0, **kwargs):
        super(ActionRegressionGuidance, self).__init__(scale)
        self.kwargs = kwargs
        self.action_dim = kwargs.get("action_dim", 20)

    def compute_guidance_loss(self, x, t, aux_data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, H, D) the trajectory to use to compute losses
        - aux_data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0
        gt_action = aux_data_batch["gt_action"]
        gt_action_traj_scaled = self.scale_action(
            gt_action[..., : self.action_dim],
            aux_data_batch["action_norm_min_bound"],
            aux_data_batch["action_norm_max_bound"],
            aux_data_batch["action_mean"],
            aux_data_batch["action_std"],
        )
        action_regression_loss = F.mse_loss(
            x[..., : self.action_dim],
            gt_action_traj_scaled,
            reduction="none",
        )
        action_regression_loss = action_regression_loss.mean(dim=(-1, -2))
        print(
            f"action_regression_loss: {action_regression_loss.mean(dim=0).item() : .6f}"
        )
        guide_losses["dummy_action_regression_guidance_loss"] = action_regression_loss
        loss_tot += action_regression_loss.mean()

        return loss_tot, guide_losses


class DynamicsRegressionGuidance(Guidance):
    def __init__(self, scale=1.0, **kwargs):
        super(DynamicsRegressionGuidance, self).__init__(scale)
        self.kwargs = kwargs
        self.dynamics_geometric_dim = kwargs.get("dynamics_geometric_dim", 45)
        self.dynamics_visual_dim = kwargs.get("dynamics_visual_dim", 768)
        self.predict_distance_to_goal = kwargs.get("predict_distance_to_goal", True)
        self.predict_visual = kwargs.get("predict_visual", True)
        self.geometric_weight = kwargs.get("geometric_weight", 1.0)
        self.visual_weight = kwargs.get("visual_weight", 1.0)
        self.spatial_visual_compress = kwargs.get("spatial_visual_compress", None)

    def compute_guidance_loss(self, x, t, aux_data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, D, H, W) the trajectory to use to compute losses
        - aux_data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        B, _, H, W = x.shape
        guide_losses = dict()
        loss_tot = 0.0
        loss_dynamics = 0.0
        gt_dynamics = aux_data_batch["gt_state_residual"]
        gt_dynamics_scaled = self.scale_state(
            gt_dynamics,
            aux_data_batch["state_norm_min_bound"],
            aux_data_batch["state_norm_max_bound"],
            aux_data_batch["state_mean"],
            aux_data_batch["state_std"],
        )
        # Build the state valid mask
        _state_valid = aux_data_batch["state_valid"]  # [B, T, H, W]
        _state_valid = rearrange(_state_valid, "b (t c) h w -> b t c h w", c=3)[
            :, 0, 0
        ]  # [B, H, W]

        # Compute the geometric loss
        loss_geometric = F.mse_loss(
            x[:, : self.dynamics_geometric_dim],
            gt_dynamics_scaled[:, : self.dynamics_geometric_dim],
            reduction="none",
        )
        state_valid_geometric = (
            _state_valid[:, None]
            .clone()
            .expand(-1, self.dynamics_geometric_dim, -1, -1)
        ).clone()
        state_valid_geometric = rearrange(state_valid_geometric, "b (t c) h w -> b t c h w", c=3)
        state_valid_geometric[:, :-5] = 0 # only use the last 5 timesteps for geometric loss
        state_valid_geometric = rearrange(state_valid_geometric, "b t c h w -> b (t c) h w")
        loss_geometric = (loss_geometric * state_valid_geometric).sum(
            dim=(-1, -2, -3)
        ) / (state_valid_geometric.sum(dim=(-1, -2, -3)) + 1e-5)
        loss_geometric *= self.geometric_weight
        loss_dynamics += loss_geometric
        print_msg = f"geometric loss: {loss_geometric.mean(dim=0).item() : .6f}"
        if self.predict_visual:
            visual_res = int(
                aux_data_batch["goal_visual_feature_patch"].shape[-2] ** 0.5
            )  # 14
            goal_visual_scaled = rearrange(
                aux_data_batch["goal_visual_feature_patch"],
                "b (h w) c -> b c h w",
                h=visual_res,
                w=visual_res,
            )
            goal_visual_scaled = F.interpolate(
                goal_visual_scaled,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            if self.spatial_visual_compress is not None:
                goal_visual_scaled = self.spatial_visual_compress(
                    rearrange(goal_visual_scaled, "b c h w -> b h w c")
                ).permute(0, 3, 1, 2)
            state_valid_visual = (
                _state_valid[:, None]
                .clone()
                .expand(-1, self.dynamics_visual_dim, -1, -1)
            ).fill_(1)
            # state_valid_visual[:, :512] = 1.0

            loss_visual = F.mse_loss(
                x[:, -self.dynamics_visual_dim :],
                goal_visual_scaled,
                reduction="none",
            )
            loss_visual = (loss_visual * state_valid_visual).sum(dim=(-1, -2, -3)) / (
                state_valid_visual.sum(dim=(-1, -2, -3)) + 1e-5
            )
            loss_visual *= self.visual_weight
            loss_dynamics += loss_visual
            print_msg += f", visual loss: {loss_visual.mean(dim=0).item() : .6f}"
        print(print_msg)
        guide_losses["dynamics_regression_guidance_loss"] = loss_dynamics
        loss_tot += loss_dynamics.mean()
        return loss_tot, guide_losses


class CorrespondenceGoalConditionedDynamicsGuidance(Guidance):
    def __init__(self, scale=1.0, **kwargs):
        super(CorrespondenceGoalConditionedDynamicsGuidance, self).__init__(scale)
        self.kwargs = kwargs
        self.dynamics_geometric_dim = kwargs.get("dynamics_geometric_dim", 45)
        self.use_chamfer_geometric_loss = kwargs.get(
            "use_chamfer_geometric_loss", True
        )  # True: Chamfer distance; False: MSE with point correspondence
        self.predict_distance_to_goal = kwargs.get("predict_distance_to_goal", True)
        self.predict_visual = kwargs.get("predict_visual", True)
        self.dynamics_visual_dim = kwargs.get("dynamics_visual_dim", 768)
        self.spatial_visual_compress = kwargs.get("spatial_visual_compress", None)

    def compute_guidance_loss(self, x, t, aux_data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, D, H, W) the trajectory to use to compute losses
        - aux_data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - aux_data_batch["goal_state_residual"] : (B, 3, H, W) the goal state residual to use for the guidance
        """
        B, _, H, W = x.shape
        guide_losses = dict()
        loss_tot = 0.0

        # Pre-process the geometric state residual
        goal_state_residual = aux_data_batch["goal_state_residual"]
        if goal_state_residual.shape[1] < self.dynamics_geometric_dim:
            assert goal_state_residual.shape[1] == 3
            goal_state_residual = goal_state_residual.repeat(
                1, self.dynamics_geometric_dim // 3, 1, 1
            )
        goal_dynamics_scaled = self.scale_state(
            goal_state_residual,
            aux_data_batch["state_norm_min_bound"],
            aux_data_batch["state_norm_max_bound"],
            aux_data_batch["state_mean"],
            aux_data_batch["state_std"],
        )  # [B, D, H, W]

        # Pre-process the distance to goal
        if self.predict_distance_to_goal:
            if "goal_distance_to_goal" in aux_data_batch:
                goal_distance_to_goal_scaled = self.scale_state(
                    aux_data_batch["goal_distance_to_goal"],
                    aux_data_batch["distance_to_goal_norm_min_bound"],
                    aux_data_batch["distance_to_goal_norm_max_bound"],
                    aux_data_batch["distance_to_goal_mean"],
                    aux_data_batch["distance_to_goal_std"],
                )  # [B, 3, H, W]
            else:
                goal_distance_to_goal_scaled = torch.zeros_like(x[:, :3])
            goal_dynamics_scaled = torch.cat(
                [goal_dynamics_scaled, goal_distance_to_goal_scaled], dim=1
            )

        # Pre-process the goal visual feature
        if self.predict_visual:
            if "goal_visual_feature_patch" in aux_data_batch:
                goal_visual_feature_scaled = rearrange(
                    aux_data_batch["goal_visual_feature_patch"],
                    "b (h w) c -> b c h w",
                    h=14,
                    w=14,
                )  # [B, 768, 14, 14]
                goal_visual_feature_scaled = F.interpolate(
                    goal_visual_feature_scaled,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )  # [B, 768, H, W]
                if self.spatial_visual_compress is not None:
                    goal_visual_feature_scaled = self.spatial_visual_compress(
                        rearrange(
                            goal_visual_feature_scaled, "b c h w -> b h w c"
                        )
                    ).permute(0, 3, 1, 2)
            else:
                goal_visual_feature_scaled = torch.zeros_like(
                    x[:, -self.dynamics_visual_dim :]
                )

            goal_dynamics_scaled = torch.cat(
                [goal_dynamics_scaled, goal_visual_feature_scaled], dim=1
            )
        # Establish the dynamics for loss computation
        x_dynamics_scaled = x
        dynamics_loss = F.mse_loss(
            x_dynamics_scaled, goal_dynamics_scaled, reduction="none"
        )

        # Geometric loss: Chamfer or MSE with point correspondence
        if self.use_chamfer_geometric_loss:
            # Chamfer distance on last timestep
            x_last_dynamics_scaled = x[:, : self.dynamics_geometric_dim].view(
                B, -1, 3, H, W
            )[:, -1]
            goal_last_dynamics_scaled = goal_dynamics_scaled[
                :, : self.dynamics_geometric_dim
            ].view(B, -1, 3, H, W)[:, -1]
            x_last_dynamics_scaled = rearrange(
                x_last_dynamics_scaled, "b c h w -> b (h w) c"
            )
            goal_last_dynamics_scaled = rearrange(
                goal_last_dynamics_scaled, "b c h w -> b (h w) c"
            )
            N1, N2 = (
                x_last_dynamics_scaled.shape[1],
                goal_last_dynamics_scaled.shape[1],
            )
            x_last_dynamics_scaled = x_last_dynamics_scaled[:, :, None].expand(
                -1, N1, N2, -1
            )
            goal_last_dynamics_scaled = goal_last_dynamics_scaled[:, None, :, :].expand(
                -1, N1, N2, -1
            )
            dynamics_geometric_loss = (
                F.mse_loss(
                    x_last_dynamics_scaled,
                    goal_last_dynamics_scaled,
                    reduction="none",
                )
                .sum(dim=-1)
                .min(dim=-1)[0]
                .mean(dim=-1)
            )
        else:
            # MSE with point correspondence (same spatial position)
            dynamics_geometric_loss = (
                dynamics_loss[:, : self.dynamics_geometric_dim]
                .view(B, -1, 3, H, W)
                .mean(dim=1)
                .mean(dim=(-1, -2, -3))
            )

        # Acquire the loss for the distance to goal
        if self.predict_distance_to_goal and "goal_distance_to_goal" in aux_data_batch:
            dynamics_distance_to_goal_loss = dynamics_loss[
                :, self.dynamics_geometric_dim : self.dynamics_geometric_dim + 3
            ].mean(dim=(-1, -2, -3))
        else:
            dynamics_distance_to_goal_loss = 0.0

        # Acquire the loss for the visual feature
        if self.predict_visual and "goal_visual_feature_patch" in aux_data_batch:
            dynamics_visual_loss = dynamics_loss[:, -self.dynamics_visual_dim :].mean(
                dim=(-1, -2, -3)
            )
        else:
            dynamics_visual_loss = 0.0

        ### MSE loss ###
        dynamics_loss = (
            dynamics_geometric_loss
            + dynamics_distance_to_goal_loss
            + dynamics_visual_loss
        )
        print(f"dynamics_loss: {dynamics_loss.mean(dim=0).item() : .6f}")

        guide_losses["goal_conditioned_dynamics_guidance_loss"] = dynamics_loss
        dynamics_loss = dynamics_loss.mean()
        loss_tot += dynamics_loss

        # ##### Some debug #####
        # state_color = aux_data_batch["state_color"][0]
        # init_state = aux_data_batch["start_state"][0]  # [3, H, W]
        # x_dynamics_descaled = self.descale_state(
        #     x_dynamics_scaled,
        #     aux_data_batch["state_norm_min_bound"],
        #     aux_data_batch["state_norm_max_bound"],
        #     aux_data_batch["state_mean"],
        #     aux_data_batch["state_std"],
        # )
        # x_dynamics_descaled_vis = (
        #     x_dynamics_descaled.view(B, -1, 3, H, W)[0, -1] + init_state
        # )
        # goal_dynamics_descaled_vis = (
        #     aux_data_batch["goal_state_residual"].view(B, -1, 3, H, W)[0, -1]
        #     + init_state
        # )
        # x_dynamics_descaled_vis = x_dynamics_descaled_vis.view(3, -1).T  # [N, 3]
        # goal_dynamics_descaled_vis = goal_dynamics_descaled_vis.view(3, -1).T  # [N, 3]
        # state_color = state_color.view(3, -1).T  # [N, 3]
        # x_dynamics_descaled_vis = x_dynamics_descaled_vis.detach().cpu().numpy()
        # goal_dynamics_descaled_vis = goal_dynamics_descaled_vis.detach().cpu().numpy()
        # state_color = state_color.detach().cpu().numpy()

        # server = viser.ViserServer()
        # server.scene.add_point_cloud(
        #     name=f"flow_state",
        #     points=x_dynamics_descaled_vis,
        #     colors=state_color,
        #     point_size=0.01,
        #     point_shape="circle",
        # )
        # server.scene.add_point_cloud(
        #     name=f"goal",
        #     points=goal_dynamics_descaled_vis,
        #     colors=state_color,
        #     point_size=0.01,
        #     point_shape="circle",
        # )
        # # If ctrl-c, close the server
        # try:
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     print("Server closed")

        # ##### Some debug #####
        return loss_tot, guide_losses


class GoalConditionedDynamicsGuidance(Guidance):
    def __init__(self, scale=1.0, **kwargs):
        super(GoalConditionedDynamicsGuidance, self).__init__(scale)
        self.kwargs = kwargs
        self.dynamics_geometric_dim = kwargs.get("dynamics_geometric_dim", 45)

    def compute_guidance_loss(self, x, t, aux_data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, D, H, W) the trajectory to use to compute losses
        - aux_data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        - aux_data_batch["goal_state_residual"] : (B, 3, H, W) the goal state residual to use for the guidance
        """
        B, _, H, W = x.shape
        guide_losses = dict()
        loss_tot = 0.0
        x_dynamics_scaled = x[:, : self.dynamics_geometric_dim]
        goal_dynamics = aux_data_batch["goal_state_residual"]  # [B, D, H, W]
        goal_dynamics = goal_dynamics.repeat(1, self.dynamics_geometric_dim // 3, 1, 1)
        goal_dynamics_scaled = self.scale_state(
            goal_dynamics,
            aux_data_batch["state_norm_min_bound"],
            aux_data_batch["state_norm_max_bound"],
            aux_data_batch["state_mean"],
            aux_data_batch["state_std"],
        )  # [B, D, H, W]
        # ##### Some debug #####
        # state_color = aux_data_batch["state_color"][0]
        # init_state = aux_data_batch["start_state"][0]  # [3, H, W]
        # x_dynamics_descaled = self.descale_state(
        #     x_dynamics_scaled,
        #     aux_data_batch["state_norm_min_bound"],
        #     aux_data_batch["state_norm_max_bound"],
        #     aux_data_batch["state_mean"],
        #     aux_data_batch["state_std"],
        # )
        # x_dynamics_descaled_vis = (
        #     x_dynamics_descaled.view(B, -1, 3, H, W)[0, dynamics_idx] + init_state
        # )
        # goal_dynamics_descaled_vis = (
        #     goal_dynamics.view(B, -1, 3, H, W)[0, dynamics_idx] + init_state
        # )
        # x_dynamics_descaled_vis = x_dynamics_descaled_vis.view(3, -1).T  # [N, 3]
        # goal_dynamics_descaled_vis = goal_dynamics_descaled_vis.view(3, -1).T  # [N, 3]
        # state_color = state_color.view(3, -1).T  # [N, 3]
        # x_dynamics_descaled_vis = x_dynamics_descaled_vis.detach().cpu().numpy()
        # goal_dynamics_descaled_vis = goal_dynamics_descaled_vis.detach().cpu().numpy()
        # state_color = state_color.detach().cpu().numpy()

        # server = viser.ViserServer()
        # server.scene.add_point_cloud(
        #     name=f"flow_state",
        #     points=x_dynamics_descaled_vis,
        #     colors=state_color,
        #     point_size=0.01,
        #     point_shape="circle",
        # )
        # server.scene.add_point_cloud(
        #     name=f"goal",
        #     points=goal_dynamics_descaled_vis,
        #     colors=state_color,
        #     point_size=0.01,
        #     point_shape="circle",
        # )
        # # If ctrl-c, close the server
        # try:
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     print("Server closed")

        # ##### Some debug #####

        x_last_dynamics_scaled = x_dynamics_scaled.view(B, -1, 3, H, W)[:, -1]
        goal_last_dynamics_scaled = goal_dynamics_scaled.view(B, -1, 3, H, W)[:, -1]

        # Compute the chamfer distance loss
        x_last_dynamics_scaled = rearrange(
            x_last_dynamics_scaled, "b c h w -> b (h w) c"
        )  # [B, N1, 3]
        goal_last_dynamics_scaled = rearrange(
            goal_last_dynamics_scaled, "b c h w -> b (h w) c"
        )  # [B, N2, 3]
        N1, N2 = x_last_dynamics_scaled.shape[1], goal_last_dynamics_scaled.shape[1]
        x_last_dynamics_scaled = x_last_dynamics_scaled[:, :, None].expand(
            -1, N1, N2, -1
        )  # [B, N1, N2, 3]
        goal_last_dynamics_scaled = goal_last_dynamics_scaled[:, None, :, :].expand(
            -1, N1, N2, -1
        )  # [B, N1, N2, 3]

        dynamics_goal_loss = F.mse_loss(
            x_last_dynamics_scaled, goal_last_dynamics_scaled, reduction="none"
        )
        dynamics_goal_loss = dynamics_goal_loss.sum(dim=-1)  # [B, N1, N2]
        dynamics_goal_loss = dynamics_goal_loss.min(dim=-1)[0]  # [B, N1]
        dynamics_goal_loss = dynamics_goal_loss.mean(dim=-1)  # [B, ]

        ### MSE loss with point correspondence ###
        # dynamics_goal_loss = F.mse_loss(
        #     x_last_dynamics_scaled, goal_dynamics_scaled, reduction="none"
        # )
        # dynamics_goal_loss = dynamics_goal_loss.mean(dim=-1) # [B, N1]
        # dynamics_goal_loss = dynamics_goal_loss.mean(dim=-1)  # [B, ]
        ### MSE loss ###
        print(f"dynamics_goal_loss: {dynamics_goal_loss.item() : .6f}")

        guide_losses["goal_conditioned_dynamics_guidance_loss"] = dynamics_goal_loss
        dynamics_goal_loss = dynamics_goal_loss.mean()
        loss_tot += dynamics_goal_loss

        return loss_tot, guide_losses


GUIDANCE_REGISTRY = {
    "CorrespondenceGoalConditionedDynamicsGuidance": CorrespondenceGoalConditionedDynamicsGuidance,
    "GoalConditionedDynamicsGuidance": GoalConditionedDynamicsGuidance,
    "ActionRegressionGuidance": ActionRegressionGuidance,
    "DynamicsRegressionGuidance": DynamicsRegressionGuidance,
}


class DynamicsGuidance(Guidance):
    def __init__(self, guidance_config, scale=1.0):
        super().__init__(scale)
        self.guidance_config = guidance_config
        self.guidance_funcs, self.guidance_weights = [], []
        for guidance_class_name, guidance_class_config in guidance_config.items():
            if guidance_class_name not in GUIDANCE_REGISTRY:
                raise ValueError(f"Unknown guidance: {guidance_class_name}")
            guidance_class = GUIDANCE_REGISTRY[guidance_class_name]
            guidance_class_config.update(scale=self.scale)
            self.guidance_funcs.append(guidance_class(**guidance_class_config))
            self.guidance_weights.append(guidance_class_config["weight"])

    def compute_guidance_loss(self, x, t, aux_data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, D, H, W) the trajectory to use to compute losses
        - aux_data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0
        for func, weight in zip(self.guidance_funcs, self.guidance_weights):
            loss, losses_dict = func.compute_guidance_loss(x, t, aux_data_batch)
            loss_tot += loss * weight
            guide_losses.update(losses_dict)

        return loss_tot, guide_losses


# class GoalConditionedGuidanceDummy(Guidance):
#     def __init__(self, scale=1.0, valid_horizon=-1):
#         super(GoalConditionedGuidanceDummy, self).__init__(scale, valid_horizon)

#     def compute_guidance_loss(self, x, t, aux_data_batch):
#         """
#         Evaluates all guidance losses and total and individual values.
#         - x: (B, H, 3) the trajectory to use to compute losses
#         - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
#         """
#         ACTION_DIM = 48
#         guide_losses = dict()
#         loss_tot = 0.0
#         factor = 200
#         # bsize, num_samp, horizon, _ = x.size()

#         # Select the number of waypoints to use for the goal
#         x_goal = x
#         # goal_scaled = aux_data_batch["goal_action_scaled"]  # [B, D]
#         goal_scaled = self.scale_action(
#             aux_data_batch["goal_action"][:, None],
#             aux_data_batch["action_norm_min_bound"],
#             aux_data_batch["action_norm_max_bound"],
#             aux_data_batch["action_mean"],
#             aux_data_batch["action_std"],
#         )[:, 0]

#         goal_loss_mask = torch.zeros_like(x_goal, dtype=torch.bool)  # [B, D]
#         goal_loss_mask[:, :, ACTION_DIM // 2 :] = True
#         # goal_loss_mask[:, -1 : ACTION_DIM // 2 : ACTION_DIM // 2 + 3] = True
#         goal_loss = F.mse_loss(x_goal, goal_scaled, reduction="none")  # [B, D]
#         goal_loss = goal_loss * goal_loss_mask.float()
#         guide_losses["action_goal_loss"] = goal_loss
#         goal_loss = goal_loss.sum() / (goal_loss_mask.sum() + 1e-5)  # [B, D]
#         goal_loss *= factor
#         loss_tot += goal_loss
#         # print(f"Goal loss: {goal_loss.item() : .6f}")
#         # # # breakpoint()
#         # x_goal_traj = x_goal[goal_loss_mask]
#         # goal_scaled_tra = goal_scaled[goal_loss_mask]
#         return goal_loss, guide_losses

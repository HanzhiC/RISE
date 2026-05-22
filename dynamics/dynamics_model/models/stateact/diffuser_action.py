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
from models.stateact.decoder import ActionTransformer, ActionCDiT
from models.stateact.fuser import (
    MultiModalFusionTransformer,
)
import torch.nn.functional as F
from models.layers_2d import MLP

# from utils.guidance_loss import DiffuserGuidance, verify_guidance_config_list
import math


class ActionDiffusionModel(nn.Module):
    """
    Action diffusion model that predicts the action of the next step.
    """

    def __init__(
        self,
        n_timesteps=100,
        loss_type="l2",
        horizon=40,
        observation_state_dim=3,
        output_state_dim=3,
        observation_action_dim=6,
        output_action_dim=6,
        base_dim=768,
        dim_mults=[2, 4, 8],
        cond_fill_value=-1.0,
        supervise_epsilons=False,
        force_start=False,
        num_language_tokens=30,
        concat_language_feature=False,
        use_feature_fuser=True,
        decoder_type="transformer",
        fuser_kwargs={
            "n_time_blocks": 3,
            "n_space_blocks": 3,
            "n_head": 4,
            "n_virtual_register_states": 64,
            "mlp_ratio": 4.0,
        },
        action_decoder_kwargs={
            "n_layer": 6,
            "n_head": 4,
            "p_drop_emb": 0.1,
            "p_drop_attn": 0.1,
            "causal_attn": True,
            "n_cond_layers": 2,
            "n_cond_tokens": 15,
        },
        **kwargs,
    ):
        super(ActionDiffusionModel, self).__init__()
        self.n_timesteps = int(n_timesteps)
        self.horizon = horizon
        self.observation_action_dim = observation_action_dim
        self.output_action_dim = output_action_dim
        self.base_dim = base_dim
        self.dim_mults = dim_mults
        self.cond_fill_value = cond_fill_value
        self.predict_epsilons = supervise_epsilons
        self.supervise_epsilons = supervise_epsilons
        self.force_start = force_start
        # self.use_map_feat_grid = use_map_feat_grid
        self.use_feature_fuser = use_feature_fuser

        ## diffuser architecture
        self.register_diffusion_params()
        self.transition_in_dim = self.observation_action_dim + self.output_action_dim
        self.concat_language_feature = concat_language_feature
        self.num_language_tokens = num_language_tokens

        ## Multi-modal fusion transformer
        self.action_proj = nn.Linear(
            self.observation_action_dim, self.base_dim, bias=True
        )
        self.visual_proj = nn.Linear(768, self.base_dim, bias=True)
        self.language_proj = nn.Linear(768, self.base_dim, bias=True)

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

        ## Action transformer
        # self.n_cond_tokens = self.horizon
        # if not self.use_feature_fuser:
        #     self.n_cond_tokens += self.horizon
        self.n_cond_tokens = action_decoder_kwargs["n_cond_tokens"]
        if self.concat_language_feature:
            self.n_cond_tokens += self.num_language_tokens

        self.decoder_type = decoder_type
        self.action_decoder_kwargs = dict(action_decoder_kwargs)
        self.action_decoder_kwargs.update(
            dict(
                horizon=self.horizon,
                transition_dim=self.transition_in_dim,
                cond_dim=self.base_dim,
                output_dim=self.output_action_dim,
                dim=self.base_dim,
                n_cond_tokens=self.n_cond_tokens,
                causal_attn=True,
            )
        )
        if self.decoder_type == "transformer":
            self.action_decoder = ActionTransformer(
                **self.action_decoder_kwargs,
            )
        elif self.decoder_type == "cdit":
            self.action_decoder_kwargs.update(
                dict(
                    n_query_tokens=self.horizon,
                )
            )
            del self.action_decoder_kwargs["p_drop_emb"]
            del self.action_decoder_kwargs["p_drop_attn"]
            del self.action_decoder_kwargs["causal_attn"]
            del self.action_decoder_kwargs["n_cond_layers"]
            self.action_decoder = ActionCDiT(
                **self.action_decoder_kwargs,
            )
        else:
            raise ValueError(f"Invalid decoder type {self.decoder_type}")

        self.loss_type = loss_type
        self.current_guidance = None

    def register_diffusion_params(self):
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(
            alphas_cumprod / (1.0 - alphas_cumprod)
        )
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(
            1.0 - alphas_cumprod
        )

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def set_guidance(self, guidance):
        """
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        """
        self.current_guidance = guidance

    def scale_action(self, action, data_batch):
        """
        - traj: B x H x 3
        """

        min_bound = data_batch["action_norm_min_bound"]
        max_bound = data_batch["action_norm_max_bound"]
        mean = data_batch["action_mean"]
        std = data_batch["action_std"]

        if len(action.shape) == 3:
            min_bound_batch = min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1)  # [B, 1, 3]
            mean_batch = mean.unsqueeze(1)  # [B, 1, 3]
            std_batch = std.unsqueeze(1)  # [B, 1, 3]
        elif len(action.shape) == 4:
            min_bound_batch = min_bound.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            mean_batch = mean.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            std_batch = std.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
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

    def descale_action(self, action, data_batch):
        """
        - traj: B x N x H x 3
        """
        min_bound = data_batch["action_norm_min_bound"]
        max_bound = data_batch["action_norm_max_bound"]
        mean = data_batch["action_mean"]
        std = data_batch["action_std"]

        if len(action.shape) == 3:
            min_bound_batch = min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1)  # [B, 1, 3]
            mean_batch = mean.unsqueeze(1)  # [B, 1, 3]
            std_batch = std.unsqueeze(1)  # [B, 1, 3]
        elif len(action.shape) == 4:
            min_bound_batch = min_bound.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            mean_batch = mean.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
            std_batch = std.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        else:
            raise ValueError("Invalid shape of the input trajectory")

        # First unscale the trajectory
        scale = max_bound_batch - min_bound_batch
        action = (action + 1) / 2
        action = action * scale + min_bound_batch

        # Then unnormalize the trajectory
        action = action * (std_batch + 1e-5) + mean_batch
        return action

    def forward(
        self,
        data_batch,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        use_class_free_guide = class_free_guide_w != 0.0
        aux_info = self.get_aux_info(
            data_batch, include_class_free_cond=use_class_free_guide
        )
        cond_samp_out = self.conditional_sample(
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
        # map_feats = aux_info["color_features_pyramid"][-1]
        outputs = {"predictions": action}  # , "map_feature": map_feats}
        if "guide_losses" in cond_samp_out:
            outputs["guide_losses"] = cond_samp_out["guide_losses"]
        return outputs

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch, training=True)
        action = data_batch["gt_action"]
        x = self.scale_action(action, data_batch)
        diffusion_loss = self.loss(x, aux_info=aux_info)
        losses = OrderedDict(
            diffusion_loss=diffusion_loss,
        )
        return losses

    def get_aux_info(self, data_batch, include_class_free_cond=False, training=False):
        aux_info = {}
        history_action = data_batch["history_action"]  # [B, H, 3]
        language_feature = data_batch["language_feature"]  # [B, L, 768]
        history_visual_feature_patch = data_batch[
            "history_visual_feature_patch"
        ]  # [B, H, 196, 768]
        history_visual_feature_patch = history_visual_feature_patch.permute(
            0, 2, 1, 3
        )  # [B, 196, H, 768]

        history_action = self.action_proj(history_action)  # [B, H, 768]
        language_feature = self.language_proj(language_feature)  # [B, L, 768]
        history_visual_feature_patch = self.visual_proj(
            history_visual_feature_patch
        )  # [B, 196, H, 768]
        if self.fuser is not None:
            state_tokens, action_tokens = self.fuser(
                history_visual_feature_patch,
                history_action,
                language_feature,
            )  # [B, 196, 768]
            state_tokens = state_tokens[:, :, self.horizon :]  # [B, 196, H, 768]
            action_tokens = action_tokens[:, self.horizon :]  # [B, H, 768]
        else:
            state_tokens = history_visual_feature_patch
            action_tokens = history_action
            history_visual_feature, _ = history_visual_feature_patch.max(
                dim=1
            )  # [B, H, 768]
            action_tokens = torch.cat(
                [history_action, history_visual_feature], dim=1
            )  # [B, H + H + L, 768]

        if self.concat_language_feature:
            language_feature_state = language_feature.unsqueeze(1).repeat(
                1, state_tokens.shape[1], 1, 1
            )  # [B, 196, L, 768]
            state_tokens = torch.cat(
                [state_tokens, language_feature_state], dim=2
            )  # [B, 196, H + L, 768]
            action_tokens = torch.cat(
                [action_tokens, language_feature], dim=1
            )  # [B, H + L, 768]
        aux_info["state_tokens"] = state_tokens
        aux_info["action_tokens"] = action_tokens
        aux_info["language_tokens"] = language_feature
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
            if self.fuser is not None:
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
            else:
                state_tokens_non_cond = history_visual_feature_patch_non_cond
                action_tokens_non_cond = history_action_non_cond
                history_visual_feature_non_cond, _ = (
                    history_visual_feature_patch_non_cond.max(dim=1)
                )  # [B, H, 768]
                action_tokens_non_cond = torch.cat(
                    [history_action_non_cond, history_visual_feature_non_cond], dim=1
                )  # [B, H + H + L, 768]
            if self.concat_language_feature:
                language_feature_state_non_cond = language_feature_non_cond.unsqueeze(
                    1
                ).repeat(
                    1, state_tokens_non_cond.shape[1], 1, 1
                )  # [B, 196, L, 768]
                state_tokens_non_cond = torch.cat(
                    [state_tokens_non_cond, language_feature_state_non_cond], dim=2
                )  # [B, 196, H + L, 768]
                action_tokens_non_cond = torch.cat(
                    [action_tokens_non_cond, language_feature_non_cond], dim=1
                )  # [B, H + L, 768]
            aux_info["state_tokens_non_cond"] = state_tokens_non_cond
            aux_info["action_tokens_non_cond"] = action_tokens_non_cond
            aux_info["language_tokens_non_cond"] = language_feature_non_cond
        return aux_info

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):
        if force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            extract(
                self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape
            )
            * x_t
            - extract(
                self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device),
                t,
                x_t.shape,
            )
            * x_start
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def guidance(self, x, t, data_batch, aux_info, num_samp=1, return_grad_of=None):
        """
        estimate the gradient of rule reward w.r.t. the input trajectory
        Input:
            x: [batch_size*num_samp, time_steps, feature_dim].  scaled input trajectory.
            data_batch: additional info.
            aux_info: additional info.
            return_grad_of: which variable to take gradient of guidance loss wrt, if not given,
                            takes wrt the input x.
        """
        assert (
            self.current_guidance is not None
        ), "Must instantiate guidance object before calling"
        bsize = int(x.size(0) / num_samp)
        horizon = x.size(1)
        with torch.enable_grad():
            # compute losses and gradient
            x_loss = x.reshape((bsize, num_samp, horizon, -1))
            tot_loss, per_losses = self.current_guidance.compute_guidance_loss(
                x_loss, t, data_batch
            )
            # print(tot_loss)
            tot_loss.backward()
            guide_grad = x.grad if return_grad_of is None else return_grad_of.grad

            return guide_grad, per_losses

    def forward_action_decoder(self, x, t, aux_info={}, class_free=False):
        suffix = "_non_cond" if class_free else ""
        start_pos = aux_info["start_pos"][:, None].repeat(1, x.shape[1], 1)  # [B, H, 6]
        action_tokens = aux_info["action_tokens" + suffix]
        if self.concat_language_feature:
            action_tokens, language_tokens = action_tokens.split(
                [action_tokens.shape[1] - self.num_language_tokens, self.num_language_tokens],
                dim=1,
            )
            cond_indices = torch.arange(
                0,
                action_tokens.shape[1],
                action_tokens.shape[1] // (self.action_decoder.n_cond_tokens - self.num_language_tokens),
            )
        else:
            language_tokens = None
            cond_indices = torch.arange(
                0,
                action_tokens.shape[1],
                action_tokens.shape[1] // self.action_decoder.n_cond_tokens,
            )
        action_tokens = action_tokens[:, cond_indices]
        if language_tokens is not None:
            cond = torch.cat([action_tokens, language_tokens], dim=1)
        else:
            cond = action_tokens
        start_pos = self.scale_action(start_pos, aux_info)
        x_noisy_inp = torch.cat([start_pos, x], dim=-1)
        model_prediction = self.action_decoder(
            x_noisy_inp, cond, t
        )
        return model_prediction

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0):
        t_inp = t
        # if self.use_map_feat_grid:
        #     # time_start = time_perf.time()
        #     map_feat_traj = self.query_map_feat_grid(x.detach(), aux_info)
        #     # print("Time taken to query map features: ", time_perf.time() - time_start)
        #     x_inp = torch.cat([x, map_feat_traj], dim=-1)
        # else:
        x_inp = x
        model_prediction = self.forward_action_decoder(
            x_inp, t_inp, aux_info, class_free=False
        )

        if class_free_guide_w != 0.0:
            x_non_cond_inp = x.clone()
            # if self.use_map_feat_grid:
            #     map_feat_traj = self.query_map_feat_grid(
            #         x_non_cond_inp.detach(), aux_info
            #     )
            #     x_non_cond_inp = torch.cat([x_non_cond_inp, map_feat_traj], dim=-1)

            # model predicts noise from that brings t to t-1
            model_non_cond_prediction = self.forward_action_decoder(
                x_non_cond_inp, t_inp, aux_info, class_free=True
            )

            if not self.predict_epsilons:
                # ... and combine to get actual model prediction (in noise space as in original paper)
                model_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_prediction
                )  # noise 1
                model_non_cond_pred_noise = self.predict_noise_from_start(
                    x_t=x, t=t, x_start=model_non_cond_prediction
                )  # noise 2
                class_free_guide_noise = (
                    (1 + class_free_guide_w) * model_pred_noise
                    - class_free_guide_w * model_non_cond_pred_noise
                )  # compose noise
                model_prediction = self.predict_start_from_noise(
                    x_t=x, t=t, noise=class_free_guide_noise, force_noise=True
                )  # get actual model prediction by sampling back
                # x_recon = self.predict_start_from_noise(
                #     x, t=t, noise=model_prediction, force_noise=False
                # )  # x_recon = model_prediction
            else:
                model_pred_noise = model_prediction
                model_non_cond_pred_noise = model_non_cond_prediction
                class_free_guide_noise = (
                    (1 + class_free_guide_w) * model_pred_noise
                    - class_free_guide_w * model_non_cond_pred_noise
                )
                model_prediction = class_free_guide_noise
                # x_recon = self.predict_start_from_noise(
                #     x, t=t, noise=model_prediction, force_noise=True
                # )
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=model_prediction, force_noise=self.predict_epsilons
        )  # x_recon = model_prediction
        # if self.predict_epsilons:
        x_recon.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )  # q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x, t)

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        data_batch,
        aux_info={},
        num_samp=1,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=True,
        eval_final_guide_loss=False,
    ):
        # NOTE: guide_clean is usually True
        b, *_, _device = *x.shape, x.device
        with_func = torch.no_grad
        # print("===> Time {} X_in {}".format(t, x.mean()))
        if self.current_guidance is not None and apply_guidance and guide_clean:
            # will need to take grad wrt noisy input
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad

        with with_func():
            # get prior mean and variance for next step => q(x_{t-1} | x_t, x_0)
            model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(
                x=x, t=t, aux_info=aux_info, class_free_guide_w=class_free_guide_w
            )

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        noise = torch.randn_like(model_mean)
        sigma = (0.5 * model_log_variance).exp()

        # compute guidance
        guide_losses = None
        guide_grad = torch.zeros_like(model_mean)
        if self.current_guidance is not None and apply_guidance:
            assert (
                not self.predict_epsilons
            ), "Guidance not implemented for epsilon prediction"
            if guide_clean:  # Return gradients of x_{t-1}
                # We want to guide the predicted clean traj from model, not the noisy one
                model_clean_pred = q_posterior_in[0]
                x_guidance = model_clean_pred
                return_grad_of = x
            else:  # Returerequires_grad gradients of x_0
                x_guidance = model_mean.clone().detach()
                return_grad_of = x_guidance
                x_guidance.requires_grad_()
            guide_grad, guide_losses = self.guidance(
                x_guidance,
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
                return_grad_of=return_grad_of,
            )

            # NOTE: empirally, scaling by the variance (sigma) seems to degrade results
            guide_grad = nonzero_mask * guide_grad  # * sigma

        noise = nonzero_mask * sigma * noise

        if self.current_guidance is not None and guide_clean:
            assert (
                not self.predict_epsilons
            ), "Guidance not implemented for epsilon prediction"
            # perturb clean trajectory
            guided_clean = (
                q_posterior_in[0] - guide_grad
            )  # x_0' = x_0 - grad (The use of guidance)
            # use the same noisy input again
            guided_x_t = q_posterior_in[1]  # x_{t}
            # re-compute next step distribution with guided clean & noisy trajectories => q(x_{t-1}|x_{t}, x_0')
            # And remember in the training process, we want to make the output of every diffusion step to be x_0
            model_mean, _, _ = self.q_posterior(
                x_start=guided_clean, x_t=guided_x_t, t=q_posterior_in[2]
            )
            # NOTE: variance is not dependent on x_start, so it won't change. Therefore, fine to use same noise.
            x_out = model_mean + noise
        else:
            x_out = model_mean - guide_grad + noise

        if self.force_start:
            start_pos = data_batch["start_pos"]  # descaled start position
            start_pos = self.scale_trajectory(start_pos[:, None], data_batch)[:, 0]
            x_out[:, 0, :] = start_pos

        if self.current_guidance is not None and eval_final_guide_loss:
            assert (
                not self.predict_epsilons
            ), "Guidance not implemented for epsilon prediction"
            # eval guidance loss one last time for filtering if desired
            #       (even if not applied during sampling)
            _, guide_losses = self.guidance(
                x_out.clone().detach().requires_grad_(),
                t,
                data_batch,
                aux_info,
                num_samp=num_samp,
            )
        return x_out, guide_losses

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        data_batch,
        num_samp,
        aux_info={},
        return_diffusion=False,
        return_guidance_losses=False,
        class_free_guide_w=0.0,
        apply_guidance=True,
        guide_clean=False,
    ):
        device = self.betas.device

        batch_size = shape[0]
        # sample from base distribution
        x = torch.randn(shape, device=device)  # (B, num_samp, horizon, transition)

        x = TensorUtils.join_dimensions(
            x, begin_axis=0, end_axis=2
        )  # B*num_samp, horizon, transition
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

        if self.current_guidance is not None and not apply_guidance:
            print(
                "DIFFUSER: Note, not using guidance during sampling, only evaluating guidance loss at very end..."
            )

        if return_diffusion:
            diffusion = [x]

        stride = 1  # NOTE: different from training time if > 1
        steps = [i for i in reversed(range(0, self.n_timesteps, stride))]

        if self.force_start:
            start_pos = data_batch["start_pos"]  # descaled start position
            start_pos = self.scale_trajectory(start_pos[:, None], data_batch)[:, 0]

            x[:, 0, :] = start_pos

        for i in steps:
            timesteps = torch.full(
                (batch_size * num_samp,), i, device=device, dtype=torch.long
            )
            x, guide_losses = self.p_sample(
                x,
                timesteps,
                data_batch,
                aux_info=aux_info,
                num_samp=num_samp,
                class_free_guide_w=class_free_guide_w,
                apply_guidance=apply_guidance,
                guide_clean=guide_clean,
                eval_final_guide_loss=(i == steps[-1]),
            )
            if return_diffusion:
                diffusion.append(x)

        x = TensorUtils.reshape_dimensions(
            x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp)
        )

        out_dict = {"pred_action": x}
        if return_guidance_losses:
            out_dict["guide_losses"] = guide_losses

        if return_diffusion:
            diffusion = [
                TensorUtils.reshape_dimensions(
                    cur_diff,
                    begin_axis=0,
                    end_axis=1,
                    target_dims=(batch_size, num_samp),
                )
                for cur_diff in diffusion
            ]
            out_dict["diffusion"] = torch.stack(diffusion, dim=3)
        return out_dict

    @torch.no_grad()
    def conditional_sample(
        self, data_batch, horizon=None, num_samp=1, class_free_guide_w=0.0, **kwargs
    ):
        try:
            batch_size = data_batch["color"].shape[0]
        except Exception:
            batch_size = data_batch["color_aug"].shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, num_samp, horizon, self.observation_action_dim)
        return self.p_sample_loop(
            shape,
            data_batch,
            num_samp=num_samp,
            class_free_guide_w=class_free_guide_w,
            **kwargs,
        )

    # def query_map_feat_grid(self, x, aux_info):
    #     gt_traj_min_bound = aux_info["gt_traj_min_bound"]
    #     gt_traj_max_bound = aux_info["gt_traj_max_bound"]

    #     query_points = self.descale_trajectory(x, gt_traj_min_bound, gt_traj_max_bound)
    #     points_features = self.map_feature_extractor(
    #         query_points=query_points, **aux_info
    #     )  # [B, N, D]
    #     return points_features

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start_init, t, aux_info={}, noise=None):
        # noise_init = torch.randn_like(x_start_init)
        noise = default(noise, lambda: torch.randn_like(x_start_init))
        # Forward process to get x_t
        x_start = x_start_init
        start_pos = x_start[:, 0, :]  # scaled start position
        # noise = noise_init
        x_noisy = self.q_sample(x_start, t, noise)
        t_inp = t

        # Reverse process to predict x_start from x_t
        # The input to the noise model includes the map features
        # if self.use_map_feat_grid:
        #     map_feat_traj = self.query_map_feat_grid(x_noisy.detach(), aux_info)
        #     x_noisy_inp = torch.cat([x_noisy, map_feat_traj], dim=-1)
        # else:
        x_noisy_inp = x_noisy
        model_prediction = self.forward_action_decoder(
            x_noisy_inp, t_inp, aux_info, class_free=False
        )
        x_recon = self.predict_start_from_noise(
            x_noisy, t_inp, model_prediction, force_noise=self.predict_epsilons
        )  # x_recon = noise
        x_mask = aux_info["action_valid"]

        if not self.predict_epsilons:
            noise_pred = self.predict_noise_from_start(
                x_noisy, t_inp, x_recon
            )  # noise_pred = x_recon
            if self.force_start:
                x_recon[:, 0, :] = start_pos
        else:
            x_recon = self.predict_start_from_noise(
                x_noisy, t_inp, model_prediction, force_noise=True
            )
            noise_pred = model_prediction
            # loss = self.loss_fn(noise_pred, noise)
            if self.force_start:
                noise_pred[:, 0, :] = start_pos

        if self.supervise_epsilons:
            assert self.predict_epsilons
            loss = self.loss_fn(noise_pred, noise, reduction="none")
            loss = (loss * x_mask).sum() * 3 / x_mask.sum()
        else:
            assert not self.predict_epsilons
            loss = self.loss_fn(x_recon, x_start, reduction="none")
            loss = (loss * x_mask).sum() * 3 / x_mask.sum()
        return loss

    def loss(self, x, aux_info={}):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, aux_info=aux_info)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


if __name__ == "__main__":
    import time

    model = ActionDiffusionModel(
        n_timesteps=10,
        loss_type="l2",
        horizon=15,
        observation_dim=6,
        output_dim=6,
        base_dim=384,
        dim_mults=[2, 4, 8],
        cond_fill_value=-1.0,
        supervise_epsilons=False,
        force_start=False,
        decoder_type="transformer",
        concat_language_feature=True,
        use_feature_fuser=False,
        num_language_tokens=30,
        fuser_kwargs={
            "n_time_blocks": 3,
            "n_space_blocks": 3,
            "n_head": 4,
            "n_virtual_register_states": 64,
            "mlp_ratio": 4.0,
        },
        action_decoder_kwargs={
            "n_layer": 8,
            "n_head": 4,
            "p_drop_emb": 0.1,
            "p_drop_attn": 0.1,
            "n_cond_layers": 4,
            "n_cond_tokens": 5,
        },
    )
    # Example data batch for DiffuserModel
    n_state_tokens = 15
    T = 8
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
    model.cuda()
    data_batch = {k: v.cuda() for k, v in data_batch.items()}
    data_batch = TensorUtils.join_dimensions(data_batch, begin_axis=0, end_axis=2)
    aux_info = model.get_aux_info(data_batch)
    # for _ in range(100):
    #     time_start = time.time()
    #     out_info = model.conditional_sample(
    #         data_batch=data_batch,
    #         aux_info=aux_info,
    #         apply_guidance=False,
    #         return_guidance_losses=False,
    #         guide_clean=False,
    #         num_samp=1,
    #     )
    #     print(time.time() - time_start)
    losses = model.compute_losses(data_batch)

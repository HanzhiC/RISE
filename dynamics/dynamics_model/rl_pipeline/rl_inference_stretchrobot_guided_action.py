import sys
import os
from policies import policy_wrapper
import utils.dataset_utils as DatasetUtils
from utils.viewer_utils import SceneViewer
import utils.aria_utils as AriaUtils
import torch
import time
import open3d as o3d
import numpy as np
from data.data_factory import datamodule_factory
from tqdm import tqdm
import cv2
import time
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot
from scipy.spatial.transform import Rotation as SciR
import utils.dataset_utils as DatasetUtils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import copy
import json
from easydict import EasyDict as edict
import yaml
import utils.tensor_utils as TensorUtils
import viser
from tqdm import tqdm
from einops import rearrange
from rl_pipeline.rl_params import TASK_PARAMS, EPISODE_CORRESPONDENCE
from infer import prepare_model, normalize_act_tokens
from utils import save_video
from pathlib import Path


def _get_video_window_length(valid_frame_range, frame_range_ratio):
    """Match ``Egoasis4DDataset._get_video``: len(range(frame_start, frame_end))."""
    n = int(valid_frame_range[1] - valid_frame_range[0])
    if n <= 0:
        return 0
    r0, r1 = frame_range_ratio
    return int(n * r1) - int(n * r0)


def load_cached_states(dataset, video_seqs, device=None):
    seqs2meta = {}
    for video_seq in tqdm(
        video_seqs, desc="Loading cached states...", total=len(video_seqs)
    ):
        # Save as the npy file
        sample_idx = dataset.sample_name_to_index[video_seq]
        valid_frame_range = dataset.valid_frame_ranges[sample_idx]
        _sample_name = video_seq.split("/")
        sample = "/".join(_sample_name[:-1])
        dataset_name = dataset.dataset_categories[sample_idx]
        dataset_path = os.path.join(dataset.data_dir, dataset_name)
        gripper_color_video_fpath = os.path.join(dataset_path, sample, "gripper_rgb")
        head_color_video_fpath = os.path.join(dataset_path, sample, "head_rgb")
        head_color_filenames, gripper_color_filenames = [], []
        for frame in range(valid_frame_range[0], valid_frame_range[1]):
            gripper_color_filenames.append(
                os.path.join(gripper_color_video_fpath, f"{frame:06d}.png")
            )
            head_color_filenames.append(
                os.path.join(head_color_video_fpath, f"{frame:06d}.png")
            )

        cache_state_dir = os.path.join(dataset_path, sample, f"state_cache")
        visual_feature_gripper_curr_fpath = os.path.join(
            cache_state_dir, "visual_feature_gripper.npy"
        )
        visual_feature_head_curr_fpath = os.path.join(
            cache_state_dir, "visual_feature.npy"
        )
        goal_visual_feature_curr_fpath = os.path.join(
            cache_state_dir, "goal_visual_feature.npy"
        )
        goal_state_residual_curr_fpath = os.path.join(
            cache_state_dir, "goal_state_residual.npy"
        )
        state_valid_curr_fpath = os.path.join(cache_state_dir, "state_valid.npy")
        start_pos_curr_fpath = os.path.join(cache_state_dir, "start_pos.npy")
        start_state_curr_fpath = os.path.join(cache_state_dir, "start_state.npy")
        gt_state_curr_fpath = os.path.join(cache_state_dir, "gt_state.npy")
        value_pred_fpath = os.path.join(cache_state_dir, "value_expected.npy")
        frame_idx_fpath = os.path.join(cache_state_dir, "frame_idx.npy")
        visual_feature_gripper_curr = np.load(
            visual_feature_gripper_curr_fpath, mmap_mode="r"
        )
        visual_feature_head_curr = np.load(
            visual_feature_head_curr_fpath, mmap_mode="r"
        )
        goal_visual_feature_curr = np.load(
            goal_visual_feature_curr_fpath, mmap_mode="r"
        )
        goal_state_residual_curr = np.load(
            goal_state_residual_curr_fpath, mmap_mode="r"
        )
        state_valid_curr = np.load(state_valid_curr_fpath, mmap_mode="r")
        start_pos_curr = np.load(start_pos_curr_fpath, mmap_mode="r")
        start_state_curr = np.load(start_state_curr_fpath, mmap_mode="r")
        gt_state_curr = np.load(gt_state_curr_fpath, mmap_mode="r")
        value_pred_curr = np.load(value_pred_fpath, mmap_mode="r")
        frame_idx_curr = (
            np.load(frame_idx_fpath, mmap_mode="r")
            if os.path.isfile(frame_idx_fpath)
            else None
        )

        def _t(x):
            t = torch.from_numpy(x)
            return t.to(device, non_blocking=True) if device is not None else t

        state_meta = {
            "visual_feature_patch_gripper": _t(visual_feature_gripper_curr),
            "visual_feature_patch": _t(visual_feature_head_curr),
            "goal_visual_feature_patch": _t(goal_visual_feature_curr),
            "gt_state_residual": _t(goal_state_residual_curr),
            # Cached as [T, 1, H, W]; we expand to [T, 3, H, W] later.
            "state_valid": _t(state_valid_curr),
            "start_pos": _t(start_pos_curr),
            "start_state": _t(start_state_curr),
            "gt_state": _t(gt_state_curr),
            "value_expected": _t(value_pred_curr),
            "head_color_filenames": head_color_filenames,
            "gripper_color_filenames": gripper_color_filenames,
        }
        if frame_idx_curr is not None:
            state_meta["frame_idx"] = _t(frame_idx_curr)
        seqs2meta[video_seq] = state_meta
    return seqs2meta


def gather_frame_data_bank(
    frame_data, cached_states_success_rollout, top_k=50, return_filenames=False
):
    # Make sure the query tensor lives on the same device as cached tensors.
    cached_device = next(iter(cached_states_success_rollout.values()))[
        "start_pos"
    ].device
    start_pos_curr = frame_data["start_pos"].to(cached_device)
    value_curr = frame_data["value_expected"].to(cached_device)
    # NOTE: This function is performance-critical. We return tensor banks instead of
    # creating a Python list of dicts for every retrieved frame.
    banks_visual_feature_patch_gripper = []
    banks_visual_feature_patch = []
    banks_goal_visual_feature_patch = []
    banks_gt_state_residual = []
    banks_state_valid = []
    banks_start_pos = []
    banks_start_state = []
    banks_gt_state = []
    banks_value_expected = []
    gripper_color_filenames = []
    head_color_filenames = []

    # cached_states_success_rollout is a dict: {video_seq: state_meta}
    for _, state_meta in cached_states_success_rollout.items():
        # Two-stage retrieval:
        # 1) coarse by value_expected difference, 2) fine by start_pos (state) distance.
        value_diff_all = torch.abs(state_meta["value_expected"] - value_curr)
        num_candidates = value_diff_all.shape[0]
        coarse_k = min(max(top_k * 4, top_k), num_candidates)
        coarse_idx = torch.topk(
            value_diff_all, k=coarse_k, largest=False, sorted=False
        ).indices

        # Support start_pos shapes:
        # - bank [N, D] with query [D]
        # - bank [N, H, D] with query [H, D]
        # and reduce to one distance score per candidate: [N].
        coarse_start_pos_bank = state_meta["start_pos"][coarse_idx]
        start_pos_l2 = torch.norm(
            coarse_start_pos_bank - start_pos_curr.unsqueeze(0), dim=-1
        )
        if start_pos_l2.ndim == 1:
            start_pos_diff = start_pos_l2
        else:
            start_pos_diff = start_pos_l2.reshape(start_pos_l2.shape[0], -1).mean(
                dim=-1
            )
        k = min(top_k, coarse_k)
        # Keep k entries with smallest start_pos distance from value-retrieved set.
        fine_local_idx = torch.topk(
            start_pos_diff, k=k, largest=False, sorted=False
        ).indices
        top_k_idx = coarse_idx[fine_local_idx]

        # Selected tensor slices. Shapes:
        # - visual_feature_patch_gripper: [k, 196, 768]
        # - visual_feature_patch: [k, 196, 768]
        # - goal_visual_feature_patch: [k, 196, 768]
        # - gt_state_residual: [k, 3, H, W]
        # - state_valid: [k, 1, H, W] (cached as state_valid[:1])
        visual_feature_patch_gripper = state_meta["visual_feature_patch_gripper"][
            top_k_idx
        ]
        visual_feature_patch = state_meta["visual_feature_patch"][top_k_idx]
        goal_visual_feature_patch = state_meta["goal_visual_feature_patch"][top_k_idx]
        gt_state_residual = state_meta["gt_state_residual"][top_k_idx]
        state_valid = state_meta["state_valid"][top_k_idx]
        start_pos = state_meta["start_pos"][top_k_idx]
        start_state = state_meta["start_state"][top_k_idx]
        gt_state = state_meta["gt_state"][top_k_idx]
        value_expected = state_meta["value_expected"][top_k_idx]
        # Expand state_valid from [k, 1, H, W] -> [k, 3, H, W] once for the batch.
        state_valid = state_valid.expand(
            -1, gt_state_residual.shape[1], -1, -1
        )  # [k, 3, H, W]

        banks_visual_feature_patch_gripper.append(visual_feature_patch_gripper)
        banks_visual_feature_patch.append(visual_feature_patch)
        banks_goal_visual_feature_patch.append(goal_visual_feature_patch)
        banks_gt_state_residual.append(gt_state_residual)
        banks_state_valid.append(state_valid)
        banks_start_pos.append(start_pos)
        banks_start_state.append(start_state)
        banks_gt_state.append(gt_state)
        banks_value_expected.append(value_expected)
        if return_filenames:
            # Only used for visualization; converting GPU indices to Python
            # values will synchronize, so we do it only when needed.
            top_k_idx_list = top_k_idx.tolist()
            gripper_color_filenames.extend(
                [state_meta["gripper_color_filenames"][i] for i in top_k_idx_list]
            )
            head_color_filenames.extend(
                [state_meta["head_color_filenames"][i] for i in top_k_idx_list]
            )

    # Concatenate across all cached sequences.
    return (
        torch.cat(banks_visual_feature_patch_gripper, dim=0),
        torch.cat(banks_visual_feature_patch, dim=0),
        torch.cat(banks_goal_visual_feature_patch, dim=0),
        torch.cat(banks_gt_state_residual, dim=0),
        torch.cat(banks_state_valid, dim=0),
        torch.cat(banks_start_pos, dim=0),
        torch.cat(banks_start_state, dim=0),
        torch.cat(banks_gt_state, dim=0),
        torch.cat(banks_value_expected, dim=0),
        gripper_color_filenames,
        head_color_filenames,
    )


def mean_token_cosine_similarity(query_feature, feature_bank):
    """Average token-wise cosine similarity.

    Args:
        query_feature: [L, C] or [1, L, C].
        feature_bank: [N, L, C].
    Returns:
        [N] similarity score.
    """
    if query_feature.ndim == 2:
        query_feature = query_feature.unsqueeze(0)
    return F.cosine_similarity(query_feature, feature_bank, dim=2).mean(dim=1)


def pooled_token_cosine_similarity(query_feature, feature_bank):
    """Cosine similarity between mean-pooled token descriptors.

    Args:
        query_feature: [L, C] or [1, L, C]
        feature_bank: [N, L, C]

    Returns:
        sims: [N]
    """
    if query_feature.ndim == 2:
        query_feature = query_feature.unsqueeze(0)

    assert query_feature.ndim == 3, query_feature.shape
    assert feature_bank.ndim == 3, feature_bank.shape
    assert query_feature.shape[0] == 1, "query_feature should contain a single query"
    assert (
        query_feature.shape[1:] == feature_bank.shape[1:]
    ), f"Shape mismatch: query {query_feature.shape}, bank {feature_bank.shape}"

    query_desc = query_feature.mean(dim=1)  # [1, C]
    bank_desc = feature_bank.mean(dim=1)  # [N, C]

    query_desc = F.normalize(query_desc, dim=-1)
    bank_desc = F.normalize(bank_desc, dim=-1)

    sims = query_desc @ bank_desc.T  # [1, N]
    return sims.squeeze(0)  # [N]


def geometry_measurement_similarity(
    query_state,
    state_bank,
    state_valid_bank=None,
    eps=1e-6,
    verbose=False,
):
    """Geometry similarity using gt_state only.

    Args:
        query_state: [H, W], [1, H, W], [C, H, W], or [B, C, H, W].
        state_bank: [N, H, W] or [N, C, H, W].
        state_valid_bank: optional [N, H, W] or [N, C, H, W] validity mask.
    Returns:
        [N] normalized similarity scores (higher is better).
    """
    if query_state.ndim == 2:
        query_state = query_state.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif query_state.ndim == 3:
        query_state = query_state.unsqueeze(0)  # [1, C, H, W]
    elif query_state.ndim != 4:
        raise ValueError(f"Unsupported query_state shape: {query_state.shape}")
    # If query is stacked as [B, T*3, H, W], keep only the first state [B, 3, H, W].
    if (
        query_state.ndim == 4
        and query_state.shape[1] % 3 == 0
        and query_state.shape[1] != 3
    ):
        query_state = rearrange(query_state, "b (t c) h w -> b t c h w", c=3)[:, 0]

    if state_bank.ndim == 3:
        state_bank = state_bank.unsqueeze(1)  # [N, 1, H, W]
    elif state_bank.ndim != 4:
        raise ValueError(f"Unsupported state_bank shape: {state_bank.shape}")
    # If bank is stacked as [N, T*3, H, W], keep the first state [N, 3, H, W].
    if (
        state_bank.ndim == 4
        and state_bank.shape[1] % 3 == 0
        and state_bank.shape[1] != 3
    ):
        state_bank = rearrange(state_bank, "n (t c) h w -> n t c h w", c=3)[:, 0]

    compute_device = state_bank.device
    query_state = query_state.to(device=compute_device, dtype=state_bank.dtype)
    query_state = query_state.expand(state_bank.shape[0], -1, -1, -1)
    state_bank = state_bank.to(device=compute_device, dtype=query_state.dtype)

    if state_valid_bank is not None:
        if state_valid_bank.ndim == 3:
            state_valid_bank = state_valid_bank.unsqueeze(1)  # [N, 1, H, W]
        elif state_valid_bank.ndim != 4:
            raise ValueError(
                f"Unsupported state_valid_bank shape: {state_valid_bank.shape}"
            )
        if (
            state_valid_bank.ndim == 4
            and state_valid_bank.shape[1] % 3 == 0
            and state_valid_bank.shape[1] not in (1, 3)
        ):
            state_valid_bank = rearrange(
                state_valid_bank, "n (t c) h w -> n t c h w", c=3
            )[:, 0]
        valid_mask = (state_valid_bank > 0).to(
            device=compute_device, dtype=query_state.dtype
        )
        if valid_mask.shape[1] == 1 and query_state.shape[1] > 1:
            valid_mask = valid_mask.expand(-1, query_state.shape[1], -1, -1)
        query_state = query_state * valid_mask
        state_bank = state_bank * valid_mask

    query_flat = query_state.flatten(start_dim=1)
    bank_flat = state_bank.flatten(start_dim=1)
    neg_rmse = -torch.sqrt(((query_flat - bank_flat) ** 2).mean(dim=1) + eps)
    score = (neg_rmse - neg_rmse.mean()) / (neg_rmse.std() + eps)
    score = torch.tanh(score)
    if verbose:
        top_idx = int(torch.argmax(score).item())

        def _state_chw_to_points(state_chw):
            # state_chw: [C, H, W] -> [H*W, 3]
            if state_chw.shape[0] > 3:
                state_chw = state_chw[:3]
            elif state_chw.shape[0] == 1:
                state_chw = state_chw.repeat(3, 1, 1)
            return rearrange(state_chw, "c h w -> (h w) c")

        query_points = _state_chw_to_points(query_state[0]).detach().cpu().numpy()
        closest_points = (
            _state_chw_to_points(state_bank[top_idx]).detach().cpu().numpy()
        )
        query_colors = np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (query_points.shape[0], 1)
        )
        closest_colors = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (closest_points.shape[0], 1)
        )
        query_pcd = DatasetUtils.visualize_points(query_points, colors=query_colors)
        closest_pcd = DatasetUtils.visualize_points(
            closest_points, colors=closest_colors
        )
        o3d.visualization.draw([query_pcd, closest_pcd])
    return score


def hierarchical_retrieval(
    coarse_scores,
    fine_scores,
    coarse_k=25,
    final_k=1,
    coarse_largest=True,
    fine_largest=True,
):
    """Select candidates by a coarse score, then rerank them by a fine score.

    ``fine_scores`` can be either a [N] tensor or a callable that receives the
    coarse candidate indices and returns scores for those candidates.
    """
    coarse_k = min(int(coarse_k), coarse_scores.shape[0])
    final_k = min(int(final_k), coarse_k)
    _, coarse_idx = torch.topk(
        coarse_scores,
        k=coarse_k,
        largest=coarse_largest,
    )
    fine_scores_on_coarse = (
        fine_scores(coarse_idx) if callable(fine_scores) else fine_scores[coarse_idx]
    )
    final_local_idx = torch.topk(
        fine_scores_on_coarse,
        k=final_k,
        largest=fine_largest,
    ).indices
    final_idx = coarse_idx[final_local_idx]
    return final_idx, coarse_idx, fine_scores_on_coarse, final_local_idx


def infer_with_top_k_references(
    data_batch,
    ref_indices,
    visual_feature_head_reference_bank,
    visual_feature_gripper_reference_bank,
    goal_visual_feature_patch_bank,
    gt_state_residual_bank,
    state_valid_bank,
    start_pos_bank,
    start_state_bank,
    policy_wrapper,
    num_samples_per_reference=5,
    w_conditional=0.3,
    enable_guidance=False,
    eval_guidance=False,
    align_to_current_state=False,
    action_only=True,
):
    """Run inference once per reference index and concatenate results along the sample dim (dim=1).

    Args:
        data_batch: current frame data batch (already moved to device).
        ref_indices: 1-D tensor of bank indices to use as references.
        *_bank: pre-built tensor banks from gather_frame_data_bank.
        policy_wrapper: PolicyVLAWorldModelWrapperStretchRobot instance.
        num_samples: number of diffusion samples per reference.
        ...
    Returns:
        Merged outputs dict with action/dynamics/visual predictions concatenated along dim=1
        and value_info concatenated along dim=1.
    """
    all_outputs = []
    for ref_idx in ref_indices:
        data_batch_reference = {
            "visual_feature_patch": visual_feature_head_reference_bank[ref_idx][
                None
            ].to(policy_wrapper.device),
            "visual_feature_patch_gripper": visual_feature_gripper_reference_bank[
                ref_idx
            ][None].to(policy_wrapper.device),
            "goal_visual_feature_patch": goal_visual_feature_patch_bank[ref_idx][
                None
            ].to(policy_wrapper.device),
            "gt_state_residual": gt_state_residual_bank[ref_idx][None].to(
                policy_wrapper.device
            ),
            "state_valid": state_valid_bank[ref_idx][None].to(policy_wrapper.device),
            "start_pos": start_pos_bank[ref_idx][None].to(policy_wrapper.device),
            "start_state": start_state_bank[ref_idx][None].to(policy_wrapper.device),
        }
        ref_outputs = policy_wrapper.inference_action_with_reference_guidance(
            data_batch,
            data_batch_reference,
            num_samples=num_samples_per_reference,
            w_conditional=w_conditional,
            enable_guidance=enable_guidance,
            eval_guidance=eval_guidance,
            align_to_current_state=align_to_current_state,
            add_history_state_actions_null=args.task == "wipe-table",
            action_only=action_only,
        )
        all_outputs.append(ref_outputs)

    all_outputs_dict = {
        "action_predictions": torch.cat(
            [o["action_predictions"] for o in all_outputs], dim=1
        ),
    }
    if not action_only and "dynamics_predictions" in all_outputs[0]:
        all_outputs_dict["dynamics_predictions"] = torch.cat(
            [o["dynamics_predictions"] for o in all_outputs], dim=1
        )
    if not action_only and "visual_predictions" in all_outputs[0]:
        all_outputs_dict["visual_predictions"] = torch.cat(
            [o["visual_predictions"] for o in all_outputs], dim=1
        )
    if eval_guidance:
        all_outputs_dict["guidance_cost"] = torch.cat(
            [o["guidance_cost"] for o in all_outputs], dim=1
        )
    return all_outputs_dict


def infer_advantage_with_video_visual_conditioning(
    policy_wrapper,
    data_batch,
    outputs,
    history_length=15,
):
    """Value/advantage per sample using DINO features from predicted video last frames."""

    future_frames = outputs["image_predictions"] # [B, N, 3, 224, 224]
    pred_actions = outputs["action_predictions"] # [B, N, H, D]
    B, N, _, _, _ = future_frames.shape
    future_frames = future_frames.view(B * N, 3, 224, 224)
    future_visual_feature_patch = (
        policy_wrapper.visual_feature_extractor.extract_features(future_frames)[-1]
    )
    future_visual_feature_patch = rearrange(future_visual_feature_patch, "b c h w -> b (h w) c", h=14, w=14)
    future_visual_feature_patch = rearrange(
        future_visual_feature_patch, "(b n) l c -> b n l c", b=B, n=N
    ) # [B, N, 196, 768]
    future_visual_feature_patch = future_visual_feature_patch
    future_visual_feature_patch = future_visual_feature_patch[:, :, None].expand(-1, -1, history_length, -1, -1) # [B, N, H, 196, 768]

    pred_action_cond = policy_wrapper.build_action_conditioning(
        pred_actions,
        action_chunk_size=policy_wrapper.cfg.ALGORITHM.model.action_chunk_size,
    ) # [B, N, H, D]
    
    data_batch_future = copy.deepcopy(data_batch)
    data_batch_current = copy.deepcopy(data_batch)
    data_batch_future = TensorUtils.repeat_by_expand_at(
        data_batch_future, repeats=N, dim=0
    )
    data_batch_current = TensorUtils.repeat_by_expand_at(
        data_batch_current, repeats=N, dim=0
    )
    data_batch_future["history_visual_feature_patch"] = future_visual_feature_patch.flatten(0, 1) # [B*N, history_length, 196, 768]
    data_batch_future[f"gt_action"] = pred_action_cond.flatten(0, 1)
    
    value_info = policy_wrapper.inference_advantage(data_batch_current, data_batch_future)
    value_info["advantage_predictions"] = value_info["advantage_predictions"].view(
        -1, N
    )
    value_info["value_predictions"] = value_info["value_predictions"].view(
        -1, N
    )
    value_info["future_value_predictions"] = value_info[
        "future_value_predictions"
    ].view(-1, N)
    outputs["value_info"] = value_info
    return outputs


def pred_actions_to_rise(pred_actions, target_horizon=50):
    """Convert model action predictions to RISE format.

    Args:
        pred_actions: torch.Tensor of shape [N, H, D]
        target_horizon: int target H (default 50)

    Returns:
        torch.Tensor of shape [N, target_horizon, 7] (xyz, rotvec, closure)
    """
    # Pad or truncate to target horizon, padding by repeating last action
    if pred_actions.shape[1] < target_horizon:
        pad_h = target_horizon - pred_actions.shape[1]
        pred_actions = torch.cat(
            [pred_actions, pred_actions[:, -1:, :].expand(-1, pad_h, -1)], dim=1
        )
    elif pred_actions.shape[1] > target_horizon:
        pred_actions = pred_actions[:, :target_horizon]

    # Keep only first 10 dims (model predicts extras)
    pred = pred_actions[..., :10]
    xyz = pred[..., :3]  # [N, H, 3]
    closure = pred[..., 3:4] * 2.0 - 1.0  # map [0,1]->[-1,1]
    r6d = pred[..., 4:10]  # [N, H, 6]

    # Convert 6D rotation to rotation matrix then to rotvec
    rotm = AriaUtils.rotation_6d_to_matrix(r6d)  # [N, H, 3, 3] (torch)
    rotm_np = rotm.detach().cpu().numpy()
    rotvec_np = SciR.from_matrix(rotm_np).as_rotvec()  # [N, H, 3]
    rotvec = torch.from_numpy(rotvec_np).to(
        device=pred_actions.device, dtype=pred_actions.dtype
    )

    out = torch.cat([xyz, rotvec, closure], dim=-1)
    assert out.shape[-1] == 7
    return out


def _dynamics_model_root():
    return Path(__file__).resolve().parents[1]


def _resolve_infer_paths(
    infer_cfg_path,
    model_root=None,
    diffusion_ckpt=None,
):
    dynamics_root = _dynamics_model_root()
    if model_root is None:
        model_root = str(dynamics_root / "checkpoints")
    if diffusion_ckpt is None:
        finetuned_ckpt = (
            dynamics_root
            / "results/2026_05_19_18_25_28/step_7000/diffusion_pytorch_model.safetensors"
        )
        pretrained_ckpt = (
            dynamics_root
            / "checkpoints/dynamics_model/pretrained/diffusion_pytorch_model.safetensors"
        )
        diffusion_ckpt = str(
            finetuned_ckpt if finetuned_ckpt.is_file() else pretrained_ckpt
        )
    base_cfg_path = infer_cfg_path
    if not os.path.isabs(base_cfg_path):
        base_cfg_path = str(dynamics_root / base_cfg_path)
    if not os.path.isfile(base_cfg_path):
        raise FileNotFoundError(f"infer base config not found: {base_cfg_path}")
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"model root not found: {model_root}")
    if not os.path.isfile(diffusion_ckpt):
        raise FileNotFoundError(f"diffusion checkpoint not found: {diffusion_ckpt}")
    return base_cfg_path, model_root, diffusion_ckpt


def _load_patched_infer_args(base_cfg_path, model_root, diffusion_ckpt):
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)
    infer_cfg["pretrained_model_name_or_path"] = model_root
    infer_cfg["diffusion_model"]["model_path"] = diffusion_ckpt
    return argparse.Namespace(**infer_cfg)


def _obs_from_head_image_path(head_image_path, size=(256, 192)):
    """Build obs tensor exactly like ``infer.load_images`` (resize only, no center crop)."""
    img = cv2.imread(head_image_path)
    if img is None:
        raise ValueError(f"Failed to read head image: {head_image_path}")
    img = img[:, :, ::-1]
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
    frame = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    frames = torch.stack([frame] * 4, dim=1)
    return torch.stack([frames], dim=0)


# Match PolicyVLAWorldModelWrapper.HEAD_IMAGE_SIZE / DINO input grid (H, W).
VIDEO_OUTPUT_SIZE = (224, 224)


def _resize_video_spatial(video, size=VIDEO_OUTPUT_SIZE):
    """Resize ``[C, T, H, W]`` (in [-1, 1]) to target ``(H, W)``."""
    target_h, target_w = size
    h, w = video.shape[-2:]
    if h == target_h and w == target_w:
        return video
    frames = video.permute(1, 0, 2, 3).float()  # [T, C, H, W]
    frames = F.interpolate(
        frames, size=size, mode="bilinear", align_corners=False
    )
    return frames.permute(1, 0, 2, 3).to(dtype=video.dtype)


def _video_tensor_to_01(video_tensor):
    """Map [C,T,H,W] from [-1,1] to [C,T,H,W] in [0,1]."""
    return ((video_tensor.detach().float() + 1) / 2).clamp(0.0, 1.0)


def _save_cached_action_videos_mp4(videos_01, save_path, fps=10):
    """Concat cached ``[C,T,H,W]`` videos (0–1) side-by-side and write one mp4."""
    if not videos_01:
        return
    t_len = min(v.shape[1] for v in videos_01)
    aligned = [v[:, :t_len].cpu() for v in videos_01]
    grid = torch.cat(aligned, dim=-1)  # [C, T, H, N*W]
    grid = grid * 2.0 - 1.0  # save_video expects [-1, 1]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save_video(grid, save_path, fps=fps)
    print(
        f"[INFO] Saved {len(videos_01)} cached videos to {save_path} "
        f"(grid shape C,T,H,W = {tuple(grid.shape)})"
    )


class VideoDynamicsInferencer:
    """Loads the LTX video dynamics model once and runs pipe.infer in-process."""

    _instances = {}

    def __init__(self, infer_args, device="cuda", num_denois_steps=50, seed=42):
        self.infer_args = infer_args
        self.device = device
        self.num_denois_steps = num_denois_steps
        self.seed = seed
        self.video_fps = 10
        print(f"[INFO] Loading video dynamics model on {device}...")
        _, _, _, _, _, self.pipe = prepare_model(infer_args, device=device)
        print("[INFO] Video dynamics model ready.")

    @classmethod
    def get_or_create(
        cls,
        infer_cfg_path="configs/ltx_model/infer.yaml",
        model_root=None,
        diffusion_ckpt=None,
        device="cuda",
        num_denois_steps=50,
        seed=42,
    ):
        base_cfg_path, model_root, diffusion_ckpt = _resolve_infer_paths(
            infer_cfg_path, model_root, diffusion_ckpt
        )
        cache_key = (base_cfg_path, model_root, diffusion_ckpt, device)
        if cache_key not in cls._instances:
            infer_args = _load_patched_infer_args(
                base_cfg_path, model_root, diffusion_ckpt
            )
            print(
                "[INFO] Video dynamics paths:",
                f"config={base_cfg_path}",
                f"model_root={model_root}",
                f"diffusion_ckpt={diffusion_ckpt}",
            )
            cls._instances[cache_key] = cls(
                infer_args,
                device=device,
                num_denois_steps=num_denois_steps,
                seed=seed,
            )
        return cls._instances[cache_key]

    def infer(
        self,
        head_image_path,
        act_tokens,
        n_chunk=1,
        norm_constant="FINETUNE_TASK",
        norm_config_path=None,
        domain_name=None,
        action_chunk=50,
        return_action=False,
        video_save_path=None,
    ):
        """Returns ``(last_frame, video)`` in [0, 1]; optionally writes ``video_save_path`` mp4."""
        obs = _obs_from_head_image_path(head_image_path)
        _, _, _, h, w = obs.shape

        if not isinstance(act_tokens, torch.Tensor):
            act_tokens = torch.tensor(act_tokens, dtype=torch.float32)
        act_tokens = act_tokens.to(self.device)
        action_dim = act_tokens.shape[-1]
        act_tokens = normalize_act_tokens(
            act_tokens,
            norm_constant,
            norm_config_path=norm_config_path,
            domain_name=domain_name,
        )
        pad_size = 30 - act_tokens.shape[2]
        act_tokens = F.pad(act_tokens, (0, pad_size), mode="constant", value=0.0)

        preds = self.pipe.infer(
            image=obs.to(self.device),
            prompt=[""],
            negative_prompt="",
            num_inference_steps=self.num_denois_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            height=h,
            width=w,
            n_view=1,
            guidance_scale=1.0,
            return_action=return_action,
            n_prev=4,
            chunk=4,
            return_video=True,
            noise_seed=self.seed,
            action_chunk=action_chunk,
            action_dim=action_dim,
            history_action_state=None,
            pixel_wise_timestep=True,
            n_chunk=n_chunk,
            act_tokens=act_tokens,
        )[0]

        if not self.infer_args.return_video:
            raise RuntimeError("infer config has return_video=false")
        video = preds["video"].data.cpu()[0]  # [C, T, H, W] in [-1, 1], native ~192x256
        video = _resize_video_spatial(video)  # [C, T, 224, 224] for WM/DINO parity
        if video_save_path is not None:
            os.makedirs(os.path.dirname(video_save_path) or ".", exist_ok=True)
            save_video(video, video_save_path, fps=self.video_fps)
            print(f"[INFO] Saved inferred video to {video_save_path}")
        video_01 = _video_tensor_to_01(video)
        return video_01[:, video_01.shape[1] // 2], video_01


def infer_video_from_pred_actions(
    pred_actions_rise,
    head_image_path,
    infer_cfg_path="configs/ltx_model/infer.yaml",
    device="cuda",
    n_chunk=1,
    norm_constant="FINETUNE_TASK",
    norm_config_path=None,
    domain_name=None,
    action_chunk=50,
    model_root=None,
    diffusion_ckpt=None,
    video_save_path=None,
    act_tokens_path=None,
):
    """Run video dynamics inference; optionally save mp4 to ``video_save_path``.

    Returns ``(last_frame, video)`` with tensors in [0, 1].

    Use ``act_tokens_path`` (same ``.pt`` as ``infer.sh``) for parity with
    ``infer_example_from_dataset.sh``. ``pred_actions_rise`` from the VLA-WM policy
    is a different action space and usually must not be fed directly to the video model.
    """
    if act_tokens_path is not None:
        act_tokens = torch.load(act_tokens_path).float()
        if act_tokens.ndim == 2:
            act_tokens = act_tokens.unsqueeze(0)
    else:
        if isinstance(pred_actions_rise, torch.Tensor):
            pa = pred_actions_rise.detach().cpu()
        else:
            pa = torch.tensor(pred_actions_rise)
        if pa.ndim != 3:
            raise ValueError("pred_actions_rise must be [N,H,7]")
        sample_actions = pa[0]
        if sample_actions.shape[0] < 50:
            raise ValueError(
                "pred_actions_rise horizon must be >=50 before sampling tokens"
            )
        act_tokens = sample_actions[1:50:2, :].unsqueeze(0).float()  # [1,25,7]
        # print(
        #     "[WARN] infer_video_from_pred_actions: using WM-predicted actions for "
        #     "video dynamics (not dataset parquet). For debugging parity with "
        #     "infer_example_from_dataset.sh, pass --infer_act_tokens_path."
        # )
    video_inferencer = VideoDynamicsInferencer.get_or_create(
        infer_cfg_path=infer_cfg_path,
        model_root=model_root,
        diffusion_ckpt=diffusion_ckpt,
        device=device,
    )
    return video_inferencer.infer(
        head_image_path,
        act_tokens,
        n_chunk=n_chunk,
        norm_constant=norm_constant,
        norm_config_path=norm_config_path,
        domain_name=domain_name,
        action_chunk=action_chunk,
        return_action=False,
        video_save_path=video_save_path,
    )


def build_action_to_save(outputs, action_dim=20):
    action = outputs["selected_action_prediction"][0]  # [H, D]
    horizon = action.shape[0]
    action = action[:, action_dim // 2 :]
    pred_action_xyz = action[:, :3].cpu().numpy()
    pred_action_aux = action[:, 3:-6].cpu().numpy()
    pred_action_rot = AriaUtils.rotation_6d_to_matrix(action[:, -6:]).cpu().numpy()
    pred_action_mat = np.eye(4)[None].repeat(horizon, axis=0)
    pred_action_mat[:, :3, 3] = pred_action_xyz
    pred_action_mat[:, :3, :3] = pred_action_rot
    pred_action_mat = pred_action_mat.reshape(horizon, -1)
    pred_action_mat = np.concatenate([pred_action_mat, pred_action_aux], axis=-1)
    assert pred_action_mat.shape[-1] == 17
    return pred_action_mat


def check_existence_of_action_save_video(action_dir, frame_range):
    for frame in range(frame_range[0], frame_range[1]):
        action_fpath = os.path.join(action_dir, f"{frame:06d}.npz")
        if not os.path.exists(action_fpath):
            return False
    return True


def remove_saved_action_if_exists(action_fpath, reason):
    if os.path.isfile(action_fpath):
        os.remove(action_fpath)
        print(f"[INFO] Removed stale guided action ({reason}): {action_fpath}")


@torch.no_grad()
def main(args):
    cfg = edict(yaml.load(open(args.cfg, "r"), Loader=yaml.FullLoader))
    # cfg.ALGORITHM.model.num_steps = 5
    cfg.DATA.rl_mode = True
    cfg.DATA.rl_round = args.rl_round
    task_params = edict(TASK_PARAMS[args.task])

    datamodule = datamodule_factory(cfg)
    datamodule.setup()
    weight_fpath = os.path.join(os.path.dirname(args.cfg), args.ckpt)
    val_dataset = datamodule.val_dataset
    val_dataset.load_rgbd_frames = True if args.visualize_action else False

    policy_wrapper = PolicyVLAWorldModelWrapperStretchRobot(
        cfg,
        weight_ckpt=weight_fpath,
        action_chunk_size=1,
        online_update_robot_state=False,
        online_update_visual_state=False,
        online_update_extrinsics_state=False,
        online_update_environment_state=True,
        policy_only=False,
    )

    # Set the guidance (visual channel dim must match WM ``wm_vm_visual_compress_dim``)
    mdl = cfg.ALGORITHM.model
    dynamics_visual_dim = int(getattr(mdl, "wm_vm_visual_compress_dim", 768))
    guidance_config = dict(
        DynamicsRegressionGuidance=dict(
            weight=10,
            dynamics_geometric_dim=mdl.dynamics_dim,
            predict_visual=mdl.wm_predict_visual,
            dynamics_visual_dim=dynamics_visual_dim,
            geometric_weight=1,
            visual_weight=10,
        )
    )
    policy_wrapper.set_guidance(guidance_config)

    # Acquire the video data
    success_labels = val_dataset.success_labels
    isteleop_labels = val_dataset.is_teleop_labels
    video_seqs_failed = []
    video_seqs_success = []

    for si, sample_name in enumerate(val_dataset.samples):
        if not success_labels[si]:
            video_seqs_failed.append(sample_name)

    for si, sample_name in enumerate(val_dataset.samples):
        if args.rollout_only:
            assert (
                not args.teleop_only
            ), "Cannot specify both rollout_only and teleop_only"
            # print(f"[INFO] Loading successful rollout sequences only")
            if success_labels[si] and not isteleop_labels[si]:
                video_seqs_success.append(sample_name)
        elif args.teleop_only:
            assert (
                not args.rollout_only
            ), "Cannot specify both rollout_only and teleop_only"
            # print(f"[INFO] Loading successful teleop sequences only")
            if success_labels[si] and isteleop_labels[si]:
                video_seqs_success.append(sample_name)
        else:
            # print(f"[INFO] Loading all successful sequences")
            if success_labels[si]:
                video_seqs_success.append(sample_name)

    print(f"[INFO] Failed sequences: {len(video_seqs_failed)}")
    print(f"[INFO] Success sequences: {len(video_seqs_success)}")
    # Quick hack
    # video_seqs_success = ["2026-03-17--11-07-42/0-340"]  # wipe
    # video_seqs_success = ["2026-03-11--17-30-21/0-227"] # basketball
    # video_seqs_failed = ["2026-03-15--12-20-14/0-333"]  # microwave
    # video_seqs_failed = ["2026-03-27--18-26-13/0-66"]  # socks
    # video_seqs_failed = ["2026-03-29--16-14-13/0-144"]  # ricecooker; didn't correct well, so value model rejects
    # video_seqs_failed = [
    #     "2026-03-17--13-00-49/0-30",
    #     "2026-03-17--13-01-37/0-29",
    # ]  # wipe

    # video_seqs_failed = [
    #     "2026-05-08--17-26-08/0-298",
    #     "2026-05-08--17-23-14/0-298",
    #     "2026-05-08--16-51-50/0-389",
    #     "2026-05-08--17-27-08/0-315",
    # ]  # microwave

    # video_seqs_failed = [
    #     "2026-03-29--14-50-51/0-169",
    # ]  # wipe

    # video_seqs_failed = [
    #     "2026-03-27--18-25-19/0-219",
    #     "2026-03-27--17-07-02/0-196",
    #     "2026-03-27--17-54-41/0-209",
    #     "2026-03-27--17-59-12/0-170",
    # ]  # socks

    # video_seqs_failed = [
    #     # "2026-04-09--17-39-43/0-42",
    #     # "2026-04-09--17-35-49/0-49",
    #     # "2026-04-09--19-05-12/0-149",
    #     # "2026-04-09--17-41-45/0-63",
    # ]  # basketball
    # video_seqs_success = ["2026-04-09--17-43-31/0-320"]
    # video_seqs_failed = ["2026-03-27--18-19-03/0-69"]
    # video_seqs_failed = [
    #     "2026-04-09--17-35-49/0-49",
    # ]
    # video_seqs_failed = ["2026-03-29--15-20-08/0-409"]
    # Load the cached states
    # Keep cached tensors on CPU: avoids a large one-time CPU->GPU transfer.
    # In episode correspondence mode we load per-episode references in the loop.
    if not args.use_episode_correspondence:
        cached_states_success_rollout = load_cached_states(
            val_dataset, video_seqs_success
        )
    else:
        cached_states_success_rollout = None
    episode_correspondence = EPISODE_CORRESPONDENCE.get(args.task, {})
    episode_cache_bank = {}
    if args.visualize:
        plt.ion()
        fig, ax = plt.subplots()
        (line_rollout,) = ax.plot([], [], label="value_future_rollout", color="orange")
        (line_actual,) = ax.plot([], [], label="value_future", color="blue")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.set_title("Value Future: Rollout vs Actual")
        ax.legend()
        # Force interactive window creation for some backends.
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

    viser_server = None
    # use_viser = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_viser = True
    if use_viser and args.visualize_action:
        viser_server = viser.ViserServer(host="0.0.0.0", port=args.viser_port)

    for video_seq in video_seqs_failed:
        if args.use_episode_correspondence:
            mapped_video_seqs = episode_correspondence.get(video_seq, [])
            if len(mapped_video_seqs) == 0:
                print(
                    f"[WARN] No episode correspondence for {video_seq}; fallback to all successful sequences."
                )
                mapped_video_seqs = video_seqs_success
            mapped_video_seqs = [
                s for s in mapped_video_seqs if s in val_dataset.sample_name_to_index
            ]
            if len(mapped_video_seqs) == 0:
                print(
                    f"[WARN] No valid mapped episodes found for {video_seq}; fallback to all successful sequences."
                )
                mapped_video_seqs = video_seqs_success
            cache_key = tuple(mapped_video_seqs)
            if cache_key not in episode_cache_bank:
                print(
                    f"[INFO] Loading cache states for {video_seq} from {len(mapped_video_seqs)} mapped episode(s)."
                )
                episode_cache_bank[cache_key] = load_cached_states(
                    val_dataset, mapped_video_seqs
                )
            cached_states_success_rollout = episode_cache_bank[cache_key]

        # Acquire the action path
        sample_idx = val_dataset.sample_name_to_index[video_seq]
        valid_frame_range = val_dataset.valid_frame_ranges[sample_idx]
        dataset_name = val_dataset.dataset_categories[sample_idx]
        _sample_name = video_seq.split("/")
        sample = "/".join(_sample_name[:-1])
        clip_idx_start, clip_idx_end = _sample_name[-1].split("-")
        clip_idx = f"{int(clip_idx_start):06d}_{int(clip_idx_end):06d}"
        dataset_path = os.path.join(val_dataset.data_dir, dataset_name)
        action_save_video_fpath = os.path.join(
            dataset_path, sample, f"dex_traj_guided_risevideogen_rlround{args.rl_round}"
        )
        action_original_video_fpath = os.path.join(dataset_path, sample, "dex_traj")
        value_pred_save_path = os.path.join(
            dataset_path,
            sample,
            f"value_prediction",
            f"{clip_idx}.npz",
        )

        # Check existence of the action save video path
        all_filenames_exist = check_existence_of_action_save_video(
            action_save_video_fpath, valid_frame_range
        )
        if all_filenames_exist and not args.overwrite:
            print(
                f"[INFO] Action save video path already exists for sequence: {video_seq}"
            )
            continue

        print(f"[INFO] Running sample: {video_seq}")
        value_future_rollout_seq, value_future_expected_seq = [], []
        span = valid_frame_range[1] - valid_frame_range[0]
        frame_range_ratio = (
            max(span - 50, 0) / span,
            1,
        )
        video_data_to_be_optimized = val_dataset._get_video(
            video_seq,
            downsample_factor=1,
            to_tensor=True,
            frame_range_ratio=frame_range_ratio,
        )
        if len(video_data_to_be_optimized) == 0:
            print(f"[WARN] No frames found for sequence: {video_seq}")
            continue

        value_seq, value_seq_future, advantage_seq = [], [], []
        guidance_meta_entries = []
        need_reference_filenames = args.visualize_action or args.save_meta
        for _, frame_data in tqdm(
            enumerate(video_data_to_be_optimized),
            desc="Processing frames...",
            total=len(video_data_to_be_optimized),
        ):

            # Check if we need to do the optimization ...
            frame_idx = int(frame_data["frame_idx"].item())
            if frame_data["advantage_label"].item() == 1:
                if not args.no_save:
                    action_save_fpath = os.path.join(
                        action_save_video_fpath, f"{frame_idx:06d}.npz"
                    )
                    remove_saved_action_if_exists(
                        action_save_fpath,
                        "advantage label",
                    )
                value_seq.append(frame_data["value_expected"].item())
                value_seq_future.append(frame_data["value_future_expected"].item())
                advantage_seq.append(frame_data["advantage"].item())
                continue

            frame_data_bank = gather_frame_data_bank(
                frame_data,
                cached_states_success_rollout,
                return_filenames=need_reference_filenames,
            )
            visual_feature_gripper_curr = frame_data[
                "history_visual_feature_patch_gripper"
            ][-1]
            visual_feature_head_curr = frame_data["history_visual_feature_patch"][-1]
            gt_state_curr = frame_data["gt_state"]  # [45, 64, 64]

            # Similarity retrieval runs on GPU for speed.
            visual_feature_gripper_curr = visual_feature_gripper_curr.to(
                policy_wrapper.device
            )
            visual_feature_head_curr = visual_feature_head_curr.to(
                policy_wrapper.device
            )
            gt_state_curr = gt_state_curr.to(policy_wrapper.device)
            (
                visual_feature_gripper_reference_bank,
                visual_feature_head_reference_bank,
                goal_visual_feature_patch_bank,
                gt_state_residual_bank,
                state_valid_bank,
                start_pos_bank,
                start_state_bank,
                gt_state_bank,
                value_expected_bank,
                gripper_color_filenames,
                head_color_filenames,
            ) = frame_data_bank
            # Move only the retrieved reference banks to GPU for cosine similarity.
            # (Keep the full cached bank on CPU to avoid large transfer overhead.)
            visual_feature_gripper_reference_bank = (
                visual_feature_gripper_reference_bank.to(policy_wrapper.device)
            )
            visual_feature_head_reference_bank = visual_feature_head_reference_bank.to(
                policy_wrapper.device
            )
            gt_state_residual_bank = gt_state_residual_bank.to(policy_wrapper.device)
            gt_state_bank = gt_state_bank.to(policy_wrapper.device)
            state_valid_bank = state_valid_bank.to(policy_wrapper.device)
            value_expected_bank = value_expected_bank.to(policy_wrapper.device)

            # Measure the similarity between the current visual feature and the reference bank
            # L is the number of visual feature tokens, and B is the number of reference frames
            # visual_feature_curr: [L, 768] - current visual feature
            # visual_feature_reference_bank: [N, L, 768] - reference visual feature bank
            similarity_gripper = pooled_token_cosine_similarity(
                visual_feature_gripper_curr,
                visual_feature_gripper_reference_bank,
            )  # [N] - similarity between the current visual feature and the reference bank
            similarity_head = pooled_token_cosine_similarity(
                visual_feature_head_curr,
                visual_feature_head_reference_bank,
            )  # [N] - similarity between the current visual feature and the reference bank
            similarity_geometric = geometry_measurement_similarity(
                gt_state_curr,
                gt_state_bank,
                state_valid_bank=state_valid_bank[:, 0],
            )  # [N] - similarity between the current visual feature and the reference bank
            similarity = (
                task_params.similarity_head_weight * similarity_head
                + task_params.similarity_gripper_weight * similarity_gripper
                + task_params.similarity_geometric_weight * similarity_geometric
            )
            if args.visualize_action:
                print(
                    f"similarity_head: {similarity_head.mean().item()}, similarity_gripper: {similarity_gripper.mean().item()}, similarity_geometric: {similarity_geometric.mean().item()}"
                )
            # Hierarchical retrieval:
            # 1) select top-k by head similarity; 2) select top-3 by joint score within that set.
            top_k_of_interest = args.num_references
            top_k = min(25, similarity_head.shape[0])
            (
                top_k_of_interest_idx,
                _,
                _,
                _,
            ) = hierarchical_retrieval(
                coarse_scores=similarity_head,
                fine_scores=similarity,
                coarse_k=top_k,
                final_k=top_k_of_interest,
                coarse_largest=True,
                fine_largest=True,
            )
            selected_vals_joint = similarity[top_k_of_interest_idx]
            local_best = int(torch.argmax(selected_vals_joint).item())
            best_bank_idx_joint = top_k_of_interest_idx[local_best]
            value_curr_for_threshold = frame_data["value_expected"].to(
                policy_wrapper.device
            )
            retrieval_selected_value_diff = float(
                torch.abs(
                    value_expected_bank[best_bank_idx_joint] - value_curr_for_threshold
                )
                .reshape(-1)
                .mean()
                .item()
            )
            retrieval_selected_similarity = float(
                selected_vals_joint[local_best].item()
            )
            value_diff_threshold_cfg = float(task_params.value_diff_threshold)
            similarity_threshold_cfg = float(task_params.similarity_threshold)
            if retrieval_selected_value_diff >= value_diff_threshold_cfg:
                print(
                    "[INFO] Skip optimization (retrieval value_diff): "
                    f"episode={video_seq} frame={frame_idx} "
                    f"selected_value_diff={retrieval_selected_value_diff:.6f} >= "
                    f"value_diff_threshold={value_diff_threshold_cfg:.6f}"
                )
                if not args.no_save:
                    action_save_fpath = os.path.join(
                        action_save_video_fpath, f"{frame_idx:06d}.npz"
                    )
                    remove_saved_action_if_exists(
                        action_save_fpath,
                        "retrieval value_diff",
                    )
                value_seq.append(frame_data["value_expected"].item())
                value_seq_future.append(frame_data["value_future_expected"].item())
                advantage_seq.append(frame_data["advantage"].item())
                continue

            if retrieval_selected_similarity <= similarity_threshold_cfg:
                print(
                    "[INFO] Skip optimization (retrieval similarity): "
                    f"episode={video_seq} frame={frame_idx} "
                    f"selected_similarity={retrieval_selected_similarity:.6f} <= "
                    f"similarity_threshold={similarity_threshold_cfg:.6f}"
                )
                if not args.no_save:
                    action_save_fpath = os.path.join(
                        action_save_video_fpath, f"{frame_idx:06d}.npz"
                    )
                    remove_saved_action_if_exists(
                        action_save_fpath,
                        "retrieval similarity",
                    )
                value_seq.append(frame_data["value_expected"].item())
                value_seq_future.append(frame_data["value_future_expected"].item())
                advantage_seq.append(frame_data["advantage"].item())
                continue

            # Use this visual feature to optimize the action of the current frame with the dynamics model
            data_batch = {
                k: v[None].to(policy_wrapper.device) for k, v in frame_data.items()
            }

            # Run inference for top-3 references and merge samples along dim=1.
            infer_video_value = getattr(args, "infer_future_video", False)
            outputs = infer_with_top_k_references(
                data_batch=data_batch,
                ref_indices=top_k_of_interest_idx,
                visual_feature_head_reference_bank=visual_feature_head_reference_bank,
                visual_feature_gripper_reference_bank=visual_feature_gripper_reference_bank,
                goal_visual_feature_patch_bank=goal_visual_feature_patch_bank,
                gt_state_residual_bank=gt_state_residual_bank,
                state_valid_bank=state_valid_bank,
                start_pos_bank=start_pos_bank,
                start_state_bank=start_state_bank,
                policy_wrapper=policy_wrapper,
                num_samples_per_reference=args.num_actions,
                w_conditional=0.1,
                enable_guidance=False,
                eval_guidance=False,
                align_to_current_state=False,
                action_only=not infer_video_value,
            )

            pred_actions_rise = pred_actions_to_rise(
                outputs["action_predictions"][0], target_horizon=50
            )
            # Current frame head/gripper image paths (can be used for visualization)
            head_image_path = os.path.join(
                dataset_path, sample, "head_rgb", f"{frame_idx:06d}.png"
            )

            # Infer future video
            infer_save_dir = getattr(
                args, "infer_save_dir", "tmp_infer_case/outputs"
            )
            os.makedirs(infer_save_dir, exist_ok=True)
            dynamics_predictions = []
            cached_videos = []
            for sample_idx, pred_action_rise in enumerate(pred_actions_rise):
                try:
                    last_frame, pred_video = infer_video_from_pred_actions(
                        pred_action_rise[None],
                        head_image_path,
                        infer_cfg_path=getattr(
                            args, "infer_cfg", "configs/ltx_model/infer.yaml"
                        ),
                        device=getattr(args, "infer_device", "cuda"),
                        n_chunk=1,
                        norm_constant="FINETUNE_TASK",
                        norm_config_path="data/utils/action_norm.json",  # change this
                        domain_name="open_the_ricecooker_pi_abs",  # change this
                        action_chunk=50,
                        model_root=getattr(args, "infer_model_root", None),
                        diffusion_ckpt=getattr(args, "infer_diffusion_ckpt", None),
                        act_tokens_path=getattr(
                            args, "infer_act_tokens_path", None
                        ),
                    )
                    dynamics_predictions.append(last_frame)
                    cached_videos.append(pred_video)
                except Exception as e:
                    print(
                        f"[ERROR] Future video inference failed for frame {frame_idx} "
                        f"sample {sample_idx}: {e}"
                    )

            if cached_videos:
                combined_mp4 = os.path.join(
                    infer_save_dir, f"{frame_idx:06d}_all_samples.mp4"
                )
                _save_cached_action_videos_mp4(cached_videos, combined_mp4, fps=10)

            if len(dynamics_predictions) == 0:
                print(
                    f"[WARN] No video frames for frame {frame_idx}, skip value infer"
                )
                continue

            video_last_frames = torch.stack(
                dynamics_predictions, dim=0
            )  # [N, 3, H, W]
            outputs["image_predictions"] = video_last_frames[None].to(policy_wrapper.device)
            outputs = infer_advantage_with_video_visual_conditioning(
                policy_wrapper,
                data_batch,
                outputs,
            )
            outputs = policy_wrapper.select_best_sample(outputs, criteria="value")

            # Select the best sample

            value_info_rollout = outputs["selected_value_info"]

            # Print the value information
            advantage_rollout, value_future_rollout = (
                value_info_rollout["advantage_predictions"],
                value_info_rollout["future_value_predictions"],
            )

            # Acquire the expected value information
            advantage_expected, value_future_expected = (
                frame_data["advantage"],
                frame_data["value_future_expected"],
            )
            advantage_threshold = frame_data["advantage_threshold"]

            # TODO: set up a task-dependent filtering strategy for the final guided cost
            print(
                f"====> Rollout Value Future: {value_future_rollout.item():.5f}, Expected Value Future: {value_future_expected.item():.5f} <==="
            )
            print(
                f"====> Rollout Advantage: {advantage_rollout.item():.5f}, Expected Advantage: {advantage_expected.item():.5f}, Threshold: {advantage_threshold:.5f} <==="
            )

            is_improved = (
                advantage_rollout.item() > max(advantage_expected.item(), 0) + 0.01
            )
            if not is_improved:
                value_future_rollout = value_future_expected
                advantage_rollout = advantage_expected

            # Add the value predictions to the video
            value_seq.append(frame_data["value_expected"].item())
            value_seq_future.append(value_future_rollout.item())
            advantage_seq.append(advantage_rollout.item())

            # Save the action
            if not args.no_save:
                action_to_save = build_action_to_save(outputs)
                action_original_fpath = os.path.join(
                    action_original_video_fpath, f"{frame_idx:06d}.npz"
                )
                history_action_original = np.load(action_original_fpath)[
                    "history_trajectory"
                ]
                action_save_fpath = os.path.join(
                    action_save_video_fpath, f"{frame_idx:06d}.npz"
                )
                action_to_save_final = {
                    "history_trajectory": history_action_original,
                    "trajectory": action_to_save,
                    "is_improved": np.array(is_improved, dtype=np.bool_),
                }
                os.makedirs(os.path.dirname(action_save_fpath), exist_ok=True)
                np.savez(action_save_fpath, **action_to_save_final)
                if is_improved:
                    print(
                        f"=============> Action improved: Saved guided action to {action_save_fpath}"
                    )

            if args.visualize_action:  # and is_improved:

                policy_wrapper.visualize_action_predictions(
                    data_batch,
                    outputs,
                    ranking_criteria="value",
                    viser_server=viser_server,
                    view="head",
                )



    if args.visualize:
        plt.ioff()

        os.makedirs(".tmp", exist_ok=True)
        fig.savefig(".tmp/stretchrobot_value_future_rollout_vs_actual.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL inference stretchrobot action priviledged."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/stretchrobot_action_priviledged.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="last.ckpt",
        help="Name of the checkpoint file.",
    )  # Seems 55k is good enough ...
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing value predictions.",
    )
    parser.add_argument(
        "--no_save",
        "-n",
        action="store_true",
        help="Do not save the value predictions.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize the value predictions.",
    )
    parser.add_argument(
        "--visualize_action",
        "-va",
        action="store_true",
        help="Visualize the action.",
    )
    parser.add_argument(
        "--rollout_only",
        "-ro",
        action="store_true",
        help="Only rollout the action.",
    )
    parser.add_argument(
        "--teleop_only",
        "-to",
        action="store_true",
        help="Only rollout the teleop sequences.",
    )
    parser.add_argument(
        "--viser_port",
        type=int,
        default=8080,
        help="Port for viser visualization server.",
    )
    parser.add_argument(
        "--task",
        "-t",
        required=True,
        type=str,
        default="pnp-basketball",
        choices=list(TASK_PARAMS.keys()),
        help="Task name.",
    )
    parser.add_argument(
        "--num_references",
        "-k",
        type=int,
        default=1,
        help="Number of references to use.",
    )
    parser.add_argument(
        "--num_actions",
        "-a",
        type=int,
        default=5,
        help="Number of actions to sample.",
    )
    parser.add_argument(
        "--infer_future_video",
        action="store_true",
        help="Run diffusion-based future video inference for the predicted actions.",
    )
    parser.add_argument(
        "--infer_cfg",
        type=str,
        default="configs/ltx_model/infer.yaml",
        help="Base infer config (patched with model/diffusion paths, like infer.sh)",
    )
    parser.add_argument(
        "--infer_model_root",
        type=str,
        default=None,
        help="Tokenizer/VAE root for infer (default: dynamics_model/checkpoints)",
    )
    parser.add_argument(
        "--infer_diffusion_ckpt",
        type=str,
        default=None,
        help="Diffusion checkpoint for infer (default: checkpoints/.../diffusion_pytorch_model.safetensors)",
    )
    parser.add_argument(
        "--infer_save_dir",
        type=str,
        default="tmp_infer_case/outputs",
        help="Directory to save inferred future videos as mp4",
    )
    parser.add_argument(
        "--infer_act_tokens_path",
        type=str,
        default=None,
        help="Optional act_tokens.pt (dataset format, same as infer.sh). Overrides WM actions.",
    )
    parser.add_argument(
        "--save_meta",
        "-sm",
        action="store_true",
        help="Save the meta data.",
    )
    parser.add_argument(
        "--use_episode_correspondence",
        "-uec",
        action="store_true",
        help="Load cache states per failed episode using rl_pipeline/rl_params.py EPISODE_CORRESPONDENCE.",
    )
    parser.add_argument(
        "--rl_round",
        "-rr",
        type=int,
        default=2,
        help="RL round.",
    )
    args = parser.parse_args()
    main(args)

#  python rl_pipeline/rl_inference_stretchrobot_guided_action.py --cfg /home/wiss/chenh/storage/logs/egoasis4d-stretchrobot-vlawmvm-repre-ablation/ablation_repre_wo_geometric_frozenvla+wm+vm_stretchrobot_pnp-ricecooker/config.yaml  -t pnp-ricecooker -n -o --use_episode_correspondence --infer_future_video --infer_save_dir tmp_infer_case/outputs     --infer_diffusion_ckpt results/2026_05_19_18_25_28/step_7000/diffusion_pytorch_model.safetensors -a 2 -n -va
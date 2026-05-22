import sys
import os
import utils.dataset_utils as DatasetUtils
from utils.viewer_utils import SceneViewer
import utils.aria_utils as AriaUtils
import torch
import time
import options
import open3d as o3d
import numpy as np
from data.data_factory import datamodule_factory
from tqdm import tqdm
import cv2
import time
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot
from scipy.spatial.transform import Rotation as SciR
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F 
DEFAULT_FPS = 15
QUERY_HORIZON = 15
EXECUTE_HORIZON = 15
SMOOTH_WEIGHT = 0.1  # Favor recent actions
TRAJ_SCALE = 1.0

INSTRUCTION = None
ACTION_DIM = 20
VIEWER_TYPE = "viser"  # "viser" or "o3d"


@torch.no_grad()
def main(cfg):
    # cfg.ALGORITHM.model.num_steps = 2
    # cfg.DATA.load_tracks = True

    datamodule = datamodule_factory(cfg)
    datamodule.setup()
    weight_fpath = os.path.join(cfg.log_dir, cfg.name, "last.ckpt")
    val_dataset = datamodule.val_dataset
    val_dataset.load_rgbd_frames = True
    val_dataset.load_tracks = True
    # Or simply run on all validation samples:
    sample_list = val_dataset.samples

    policy = PolicyVLAWorldModelWrapperStretchRobot(
        cfg,
        weight_ckpt=weight_fpath,
        action_chunk_size=1,
        online_update_robot_state=False,
        online_update_visual_state=False,
        online_update_extrinsics_state=False,
        online_update_environment_state=False,
        policy_only=True,
        # action_meta_fpath="assets/stretchrobot_pickupbottle_relaction_meta.npz",
        # state_meta_fpath="assets/stretchrobot_pickupbottle_state_meta.npz",
    )
    model = policy.model
    advtange_list = []
    sample_list = [
        "2026-03-10--11-09-28/0-123",  # Failure
        "2026-03-10--11-17-52/0-169",  # Success
    ]
    for sidx, sample in enumerate(sample_list):
        print(f"[INFO] Running sample: {sample}")
        sample_idx = val_dataset.sample_name_to_index[sample]
        video_data = val_dataset._get_video(
            sample, to_tensor=True, downsample_factor=1, frame_range_ratio=(0.0, 1.0)
        )
        dataset_name = val_dataset.dataset_categories[sample_idx]
        last_frame_idx = len(video_data) - 1
        value_seq_current = []
        value_seq_future = []
        for i, data_batch in tqdm(
            enumerate(video_data),
            total=last_frame_idx + 1,
            desc="Online stretch robot policy rollout ...",
        ):

            data_batch = {k: v[None].to(model.device) for k, v in data_batch.items()}
            model.extract_dinov3_features(data_batch)
            outputs = model(
                data_batch,
                action_only=False,
                enable_guidance=True,
                eval_guidance=True,
                align_to_current_state=False,
            )
            value_pred = outputs["value_predictions"][:, 0]
            gt_value = data_batch["gt_state_value"].float().item()
            print(
                f"Current predicted value: {value_pred.item()}, GT value: {gt_value}"
            )
            value_seq_current.append(value_pred.item())

            # Infer the value using the predicted visual
            res_visual = int(data_batch["history_visual_feature_patch"].shape[-2] ** 0.5)
            visual_predictions = outputs["visual_predictions"][:, 0]
            visual_predictions = F.interpolate(
                visual_predictions,
                size=(res_visual, res_visual),
                mode="bilinear",
                align_corners=False,
            )
            visual_predictions = rearrange(visual_predictions, "b c h w -> b (h w) c")
            visual_predictions = visual_predictions[:, None].expand(-1, 15, -1, -1)
            data_batch["history_visual_feature_patch"] = visual_predictions
            outputs = model(
                data_batch,
                action_only=False,
                value_only=True,
            )
            value_pred_future = outputs["value_predictions"][:, 0]
            gt_value_future = data_batch["gt_state_value_future"].float().item()
            print(
                f"Future predicted value: {value_pred_future.item()}, GT value: {gt_value_future}"
            )
            value_seq_future.append(value_pred_future.item())

            # # Visualize the result
            # color = data_batch["color"].cpu().numpy().transpose(1, 2, 0)
            # color_vis = (color * 255).astype(np.uint8)[..., ::-1].copy()
            # cv2.putText(color_vis, f"Pr-V: {value_pred:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(color_vis, f"GT-V: {gt_value:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imshow("color_vis", color_vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Draw the value sequence
        if sidx == 0:
            value_seq_current_failure = np.array(value_seq_current)
            value_seq_future_failure = np.array(value_seq_future)
        elif sidx == 1:
            value_seq_current_success = np.array(value_seq_current)
            value_seq_future_success = np.array(value_seq_future)
    plt.plot(
        value_seq_current_failure,
        label="Current Value Failure Traj.",
        color="red",
        linestyle="solid",
    )
    plt.plot(
        value_seq_future_failure,
        label="Future Value Failure Traj.",
        color="orange",
        linestyle="dashdot",
    )
    plt.plot(
        value_seq_current_success,
        label="Current Value Success Traj.",
        color="green",
        linestyle="solid",
    )
    plt.plot(
        value_seq_future_success,
        label="Future Value Success Traj.",
        color="blue",
        linestyle="dashdot",
    )
    plt.legend()
    plt.show()
    # os.makedirs(".tmp", exist_ok=True)
    # plt.savefig(".tmp/stretchrobot_value_comparison.png")

    advantage_list = np.array(advtange_list)
    advantage_threshold = np.percentile(
        advantage_list, 0.3
    )  # top 30% advantage samples are considered as successful
    print(f"=============> Advantage threshold: {advantage_threshold}")


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.print_options(opt)
    options.save_options_file(opt)
    main(opt)
    # find_lastest_checkpoint(opt)

# python train_scripts/stream_action_and_dynamics_w_chunking_online.py  --config=/home/wiss/chenh/storage/logs/egoasis4d-action/vanilla_vla_wm_cditAbsPos_ActTransformer_trainonfullMixed/config.yaml  --DATA.split_fpath_test=data/splits/hoi4d_dummy_valid_frame_ranges_w_trajectory.csv
# # Save the statistics

# action_meta_to_save = {
#     "action_mean": data_batch["action_mean"][0].cpu().numpy(),
#     "action_std": data_batch["action_std"][0].cpu().numpy(),
#     "action_norm_max_bound": data_batch["action_norm_max_bound"][0]
#     .cpu()
#     .numpy(),
#     "action_norm_min_bound": data_batch["action_norm_min_bound"][0]
#     .cpu()
#     .numpy(),
# }
# np.savez(
#     "assets/stretchrobot_pickupbottle_relaction_meta.npz",
#     **action_meta_to_save,
# )
# dynamics_meta_to_save = {
#     "state_mean": data_batch["state_mean"][0].cpu().numpy(),
#     "state_std": data_batch["state_std"][0].cpu().numpy(),
#     "state_norm_max_bound": data_batch["state_norm_max_bound"][0]
#     .cpu()
#     .numpy(),
#     "state_norm_min_bound": data_batch["state_norm_min_bound"][0]
#     .cpu()
#     .numpy(),
# }
# np.savez(
#     "assets/stretchrobot_pickupbottle_state_meta.npz",
#     **dynamics_meta_to_save,
# )

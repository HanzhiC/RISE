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
from policies.exploration_policy_wrapper import PolicyVLAWorldModelWrapperExploration
from scipy.spatial.transform import Rotation as SciR

DEFAULT_FPS = 15
QUERY_HORIZON = 15
EXECUTE_HORIZON = 15
SMOOTH_WEIGHT = 0.1  # Favor recent actions
TRAJ_SCALE = 1.0
KEYS_FOR_STREAMING = [
    # Action Meta
    "action_valid",
    "action_mean",
    "action_std",
    "action_norm_max_bound",
    "action_norm_min_bound",
    "gt_action",
    # Dynamics Meta
    # "state_valid",
    # "state_mean",
    # "state_std",
    # "state_norm_max_bound",
    # "state_norm_min_bound",
    # "gt_state",
    # "state_color",
    # Sensor Data
    "language_feature",
    "color",
    "depth",
    "intrinsics",
    "T_cam0_cam",
    "T_world_cam",
    "start_pos",
    # "start_state",
    # History Data
    # "history_state",
    "history_action",
    "history_raymap",
    "history_visual_feature_patch",
]

# INSTRUCTION = "carry laptop"
# INSTRUCTION = "pick up laptop"
# INSTRUCTION = "open laptop"  # "open laptop"
INSTRUCTION = None
ACTION_DIM = 20
VIEWER_TYPE = "o3d"  # "viser" or "o3d"


@torch.no_grad()
def main(cfg):
    # cfg.ALGORITHM.model.num_steps = 2
    # cfg.DATA.load_tracks = True
    cfg.DATA.split_fpath_test = (
        "data/splits/mrl4d_releaserobotclip_valid_frame_ranges.csv"
    )

    datamodule = datamodule_factory(cfg)
    datamodule.setup()
    weight_fpath = os.path.join(cfg.log_dir, cfg.name, "last.ckpt")
    val_dataset = datamodule.val_dataset
    val_dataset.load_rgbd_frames = True

    # Or simply run on all validation samples:
    sample_list = val_dataset.samples
    # sample_list = [
    #     "2026-03-15--14-13-20/0-308",
    # ]

    policy = PolicyVLAWorldModelWrapperExploration(
        cfg,
        weight_ckpt=weight_fpath,
        action_chunk_size=1,
        online_update_robot_state=True,
        online_update_visual_state=False,
        online_update_extrinsics_state=True,
        online_update_environment_state=False,
        policy_only=True,
        action_xyz_offset=[0.0, 0.0, 0.0],
        action_xyz_scale=2.0,
        # action_meta_fpath="assets/stretchrobot_pickupbottle_relaction_meta.npz",
        # state_meta_fpath="assets/stretchrobot_pickupbottle_state_meta.npz",
    )
    # Visualization containers
    dynamics_color = DatasetUtils.random_colors(4096)
    dynamics_color = np.array(dynamics_color)

    for sidx, sample in enumerate(sample_list):
        vis_actions = []
        vis_scenes = []
        vis_slam_poses = []
        print(f"[INFO] Running sample: {sample}")
        sample_idx = val_dataset.sample_name_to_index[sample]
        video_data = val_dataset._get_video(
            sample, to_tensor=True, downsample_factor=1, frame_range_ratio=(0.0, 1)
        )
        dataset_name = val_dataset.dataset_categories[sample_idx]
        last_frame_idx = len(video_data) - 1
        policy.reset()
        for i, _data_batch in tqdm(
            enumerate(video_data),
            total=last_frame_idx + 1,
            desc="Online exploration robot policy rollout ...",
        ):
            if i > last_frame_idx:
                break

            # Build a batch of size 1
            _data_batch = {k: v[None].to(policy.device) for k, v in _data_batch.items()}
            data_batch = {k: _data_batch[k] for k in KEYS_FOR_STREAMING}
            data_batch = _data_batch

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
            # os.makedirs("assets", exist_ok=True)
            # np.savez(
            #     "assets/egoasis4d_ExploreRobot_actionInworld_statistics.npz",
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
            #     "assets/stretchrobot_pick-and-place_state_meta.npz",
            #     **dynamics_meta_to_save,
            # )
            if policy.cfg.ALGORITHM.model.am_predict_action_frame == "world":
                data_batch["start_pos_world"] = data_batch["start_pos"]
            else:
                data_batch["start_pos_world"] = (
                    DatasetUtils.transform_two_hands_trajectory(
                        data_batch["start_pos"],  # [1, D]
                        data_batch["T_world_cam"][0],  # [4, 4]
                        has_finger_tips=False,
                    )
                )

            # Do the inference
            outputs = policy.inference(data_batch, action_only=True, align_to_current_state=True)
            if "gt_action" in data_batch:
                gt_action = data_batch["gt_action"][0][
                    :, :ACTION_DIM
                ]  # [H, ACTION_DIM]
                gt_action = gt_action[:, ACTION_DIM // 2 :]
                gt_action_xyz = gt_action[:, :3]  # [H, 3]
                gt_action_aux = gt_action[:, 3:-6]  # [H, 1]
                gt_action_r6d = gt_action[:, -6:]  # [H, 6]
                gt_action_rot = AriaUtils.rotation_6d_to_matrix(
                    gt_action_r6d
                )  # [H, 3, 3]
            else:
                gt_action = None

            pred_action_chunk = outputs["latest_predicted_action"].cpu().numpy()
            pred_progress = pred_action_chunk[..., -1].mean().item()
            pred_action_chunk = pred_action_chunk[..., :ACTION_DIM]
            # print(f" ===> Current progress: {pred_progress:.3f}")

            # ------------------------------------------------------------------
            #  Build point cloud for current frame scene
            # ------------------------------------------------------------------
            color = _data_batch["color"][0].cpu().numpy().transpose(1, 2, 0)
            # color_gripper = (
            #     _data_batch["color_gripper"][0].cpu().numpy().transpose(1, 2, 0)
            # )
            # cv2.imshow("color_gripper", (color_gripper * 255).astype(np.uint8)[..., ::-1].copy())
            # cv2.waitKey(1)
            depth = _data_batch["depth"][0].cpu().numpy()
            intr = _data_batch["intrinsics"][0].cpu().numpy()
            T_world_cam = data_batch["T_world_cam"][0].cpu().numpy()

            points, scene_ids = DatasetUtils.backproject(
                depth, intr, depth < 1.5, False
            )
            points_rgb = color[scene_ids[0], scene_ids[1]]
            points_world = DatasetUtils.transform_points(points, T_world_cam)
            pcd_scene = DatasetUtils.visualize_points(points_world, colors=points_rgb)
            vis_scenes.append(pcd_scene)
            vis_slam_poses.append(T_world_cam @ AriaUtils.T_z_m90.T)

            # Acquire the action trajectory
            vis_action = o3d.geometry.TriangleMesh() if VIEWER_TYPE == "o3d" else []
            xyz, qua, closure, progress = (
                pred_action_chunk[:, :3],
                pred_action_chunk[:, 3:7],
                pred_action_chunk[:, 7:8],
                pred_action_chunk[:, 8:],
            )
            closure = closure.mean().item()
            progress = progress[:, 0] * (len(video_data) - 1)
            rot = SciR.from_quat(qua).as_matrix()

            # Visualize the predicted action
            hand_action = np.eye(4)[None, :, :].repeat(xyz.shape[0], axis=0)
            hand_action[:, :3, 3] = xyz
            hand_action[:, :3, :3] = rot

            # Visualize the ground-truth action
            if gt_action is not None:
                hand_action_gt = np.eye(4)[None, :, :].repeat(
                    gt_action.shape[0], axis=0
                )
                hand_action_gt[:, :3, 3] = gt_action_xyz.cpu().numpy()
                hand_action_gt[:, :3, :3] = gt_action_rot.cpu().numpy()

            cmap_name = "turbo"

            if VIEWER_TYPE == "o3d":
                vis_action = DatasetUtils.visualize_6d_trajectory(
                    hand_action, size=0.01, cmap_name=cmap_name, to_mesh=True
                )

            else:
                vis_action = DatasetUtils.visualize_3d_trajectory(
                    hand_action[:, :3, 3],
                    size=0.01,
                    cmap_name=cmap_name,
                    to_mesh=False,
                )

            vis_actions.append(vis_action)

        viewer = SceneViewer[VIEWER_TYPE](
            vis_scenes=vis_scenes,
            vis_trajs=vis_actions,
            slam_poses=vis_slam_poses,
            pcd_slam=None,
            viewer_name=f"Joint Policy",
            port=int(8089 + sidx),
        )
        viewer.run()


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

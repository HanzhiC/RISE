import torch
import numpy as np

# from data.hoi4d_dataset import HOI4DDataset
from data.dataset import Egoasis4DDataset
from einops import rearrange
import utils.dataset_utils as DatasetUtils
import open3d as o3d
import utils.aria_utils as AriaUtils


class FlowMotionPlanner:
    "borrowed from GeneralFlow repo https://github.com/michaelyuancb/general_flow/blob/main/aff_exec.py"

    def __init__(self, device):
        self.device = device
        self.default_rel_finger_tips = np.array(
            [
                [-0.00254485, -0.07379916, 0.09254396],
                [0.01526573, -0.03315647, 0.12669331],
                [0.00314775, -0.02333091, 0.11296159],
                [-0.01055881, -0.01680289, 0.12792534],
                [-0.02136223, 0.02216195, 0.11833942],
            ]
        )

    def rigid_transform_3d(self, A, B, weights=None, weight_threshold=0):
        """
        CodeBase: https://github.com/zhongcl-thu/3D-Implicit-Transporter
        Input:
            - A:       [bs, num_kps, 3], source point cloud
            - B:       [bs, num_kps, 3], target point cloud
            - weights: [bs, num_kps]     weight for each correspondence
            - weight_threshold: float,    clips points with weight below threshold
            all is Tensor
        Output:
            - R, t [bs, 3, 3], [bs, 3, 1]  rotation and translation
            - success: bool
        """
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0
        # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
        )
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
        )

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm

        try:
            U, S, Vt = torch.svd(H.cpu())
            U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
            delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
            eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).double().to(A.device)
            eye[:, -1, -1] = delta_UV
            R = Vt @ eye @ U.permute(0, 2, 1)
            t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
            # warp_A = transform(A, integrate_trans(R,t))
            # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
            return R, t, True
        except:
            print("Fail to Generation.")
            return (
                torch.eye(3).unsqueeze(0).repeat(A.shape[0]).to(self.device),
                torch.zeros(A.shape[0], 3, 1).to(self.device),
                False,
            )

    def get_motion_planning(self, kpst, weights, plan_step=1, commit=""):
        """
        Input:  kpst: [bs, num_kps, num_steps, 3], keypoints at each time step
        """
        if plan_step > kpst.shape[2]:
            raise ValueError(
                f"plan_step={plan_step} should be smaller than KPST.Length={kpst.shape[2]}"
            )
        motion_plan = []
        if weights is not None:
            if weights.ndim == 1:
                weights = weights[None, :].repeat(kpst.shape[0], 1)  # (bs, num_kps)
        for i in range(plan_step):  # from timestep i to i + 1
            pcd_A = kpst[:, :, i]  # (bs, num_kps, 3)
            pcd_B = kpst[:, :, i + 1]  # (bs, num_kps, 3) # next time step
            R, t, success = self.rigid_transform_3d(
                pcd_A, pcd_B, weights=weights
            )  # [bs, 3, 3]; [bs, 3, 1], bs
            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            R_save = R.squeeze(0)  # R: [3, 3]
            t_save = t.squeeze(0).squeeze(-1)  # t: [3]
            motion_plan.append((R_save, t_save, success))
        return motion_plan

    def get_kps_3d_and_weights(
        self, kps, gripper_3d_pos, radius=0.08, weight_beta=0.1, kps_max=256
    ):
        """
        get kps and weights from flow3d
        Args:
            kps: N, T, 3
            gripper_3d_pos: (3,)
            radius: float, radius around gripper pose to sample kps
            weight_beta: float, beta for weights
            kps_max: int, max number of kps to sample

        Returns:
            kps: (kps_max, 3)
            weights: (kps_max,)
        """
        device = (
            kps.device
            if isinstance(kps, torch.Tensor)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Convert inputs to torch tensors if they aren't already
        assert gripper_3d_pos.shape == (3,), "gripper_3d_pos should be (3,)"
        if not isinstance(kps, torch.Tensor):
            kps = torch.tensor(kps, device=device)
        if not isinstance(gripper_3d_pos, torch.Tensor):
            gripper_3d_pos = torch.tensor(gripper_3d_pos, device=device)
        kps_t0 = kps[:, 0, :]  # (M, 3)
        dist = torch.norm(kps_t0 - gripper_3d_pos, dim=1)  # dist from kps to gripper
        idx = dist < radius
        kps, dist = kps[idx], dist[idx]
        assert (
            kps.shape[0] > 0
        ), "No keypoints found within the specified radius of the gripper position."

        if kps.shape[0] > kps_max:
            idx = torch.argsort(dist)[:kps_max]
            kps, dist = kps[idx], dist[idx]

        weights = 1 / (
            dist + weight_beta
        )  #! points sampled around the EE -> closer to the gripper, higher weight

        if kps.shape[0] < kps_max:
            repeat_idx = torch.randint(
                0, kps.shape[0], (kps_max - kps.shape[0],), device=device
            )
            kps = torch.cat([kps, kps[repeat_idx]], dim=0)
            weights = torch.cat([weights, weights[repeat_idx]])

        return kps, weights

    def compute_motion_plan(self, flow3d, contact_pt, num_steps=4, radius=1.4):
        """compute motion plan from flow3d using SVD
        flow3d: [N, T, 3], N: num points,
        num steps: int, number of steps to plan
        radius: float, radius to sample kps
        return:
            motion_plan: [(R, t, success), ...], R, t is relative motion and translation in flow3d frame
            flows: [bs, Q, num_steps, 3]
        """
        if isinstance(flow3d, np.ndarray):
            flow3d = torch.tensor(flow3d)

        N, T, _ = flow3d.shape
        bs = 1
        Q = N // bs
        flows = flow3d[:, :, :3]  # (N, T, 5) -> (N, T, 3)
        flows = flows.view(bs, Q, T, 3).to(
            torch.float64
        )  # (bs, Q, T, 3) # ! why bs*Q = N???? -> into batch size, we set bs=1 for now

        # get sparse flow for computing
        plan_idx = np.linspace(0, T - 1, num_steps, dtype=int)
        flows = flows[:, :, plan_idx, :]  # bs, Q, num_steps, 3
        flows, weights = self.get_kps_3d_and_weights(
            flows.squeeze(), contact_pt, radius=radius, kps_max=256
        )  # flows: [bs, kps_max, num_steps, 3], weights[bs, kps_max]

        flows = flows.unsqueeze(0)

        motion_plan = self.get_motion_planning(
            flows, weights=weights, plan_step=num_steps - 1
        )
        return motion_plan, flows

    def apply_motion_plan(
        self, pose_init, motion_plan, filter_threshold=0.01, trajectory_length=15
    ):
        """apply motion plan to the initial pose, from relative R, t to absolute pose
        Args:
        - pose_init: [4, 4],
        - motion_plan: List[(R, t, success), ...], R: [3, 3], t: [3, ]
        Returns:
        - poses: [4, 4], list of *absolute* poses
        """
        poses = [pose_init.copy()]
        current_pose = poses[0]

        for R, t, success in motion_plan:
            new_pose = np.eye(4)
            new_pose[:3, :3] = R @ current_pose[:3, :3]
            pos = current_pose[:3, 3].copy()
            new_pose[:3, 3] = np.matmul(R, pos[..., None]).squeeze() + t
            poses.append(new_pose)
            current_pose = new_pose
        poses = np.array(poses)

        # Filter out the invalid poses
        poses_filtered = [poses[0]]
        for pose_curr in poses[1:]:
            pose_prev = poses_filtered[-1]
            t_prev, t_curr = pose_prev[:3, 3], pose_curr[:3, 3]
            dt = np.linalg.norm(t_curr - t_prev)
            if dt > filter_threshold:
                poses_filtered.append(pose_curr)
        poses_filtered = np.array(poses_filtered)
        poses_filtered = self.interpolate_trajectory(poses_filtered, trajectory_length)
        return poses_filtered

    def interpolate_trajectory(self, traj, num_points=15):
        """
        Interpolate the trajectory to the given number of points
        traj: (T, 4, 4)
        num_points: int
        """
        traj_rot, traj_tra = traj[:, :3, :3], traj[:, :3, 3]
        fill_indices = np.arange(traj_rot.shape[0])
        traj_rot_interp, _, _ = DatasetUtils.interpolate_rotation(
            fill_indices, traj_rot, num_points=num_points, rotation_format="matrix"
        )
        traj_tra_interp, _, _ = DatasetUtils.interpolate_trajectory(
            fill_indices, traj_tra, num_points=num_points
        )
        traj_interp = np.eye(4)[None, :, :].repeat(traj_rot_interp.shape[0], axis=0)
        traj_interp[:, :3, :3] = traj_rot_interp
        traj_interp[:, :3, 3] = traj_tra_interp
        return traj_interp

    def build_hand_trajectory(self, root_traj, finger_tips_traj=None):
        """
        Build the hand trajectory from the root trajectory and finger tips trajectory
        root_traj: (T, 4, 4)
        """
        horizon = root_traj.shape[0]
        if finger_tips_traj is None:
            finger_tips_traj = self.default_rel_finger_tips.copy()[None, :, :].repeat(
                horizon, axis=0
            )
            finger_tips_traj = finger_tips_traj + root_traj[:, :3, 3][:, None]
            finger_tips_traj = finger_tips_traj.reshape(horizon, -1)

        traj_tra = root_traj[:, :3, 3]  # [T, 3]
        traj_rot = root_traj[:, :3, :3]  # [T, 3, 3]
        traj_r6d = AriaUtils.matrix_to_rotation_6d(
            torch.from_numpy(traj_rot).float()
        ).numpy()  # [T, 6]
        traj_r6d = traj_r6d.reshape(horizon, -1)

        # Build the hand trajectory
        traj = np.concatenate([traj_tra, finger_tips_traj, traj_r6d], axis=-1)
        return traj


def transform_state(state, pose):
    """
    Transform state to the pose frame
    state: (T * 3, H, W)
    pose: (4, 4)
    """
    state = rearrange(state, "(t c) h w -> t h w c", c=3)
    T, H, W, C = state.shape
    state = state.reshape(-1, 3)
    state = state @ pose[:3, :3].T + pose[:3, 3]
    state = state.reshape(T, H, W, C)
    state = rearrange(state, "t h w c -> (t c) h w")
    return state


if __name__ == "__main__":
    dataset = Egoasis4DDataset(
        # split_fpath="data/splits/hoi4d_release_valid_frame_ranges.csv",
        split_fpath="data/splits/hoi4d_releaseclipTest_valid_frame_ranges_w_trajectory.csv",
        horizon=15,
        fps=30,
        load_tracks=True,
        load_hands=True,
        clip_length=1,
        track_patch_size=64,
        do_post_process=True,
        track_horizon_of_interest=[2, 5, 8, 11, 14],
    )
    flow_planner = FlowMotionPlanner(device="cpu")
    # video_data = dataset._get_video("ZY20210800004/H4/C3/N67/S380/s05/T2.000")
    # for data in dataset:
    data = dataset[8]
    data = {k: v[0] for k, v in data.items()}
    gt_state = data["gt_state"]  # T * 3, H, W
    state_valid = data["state_valid"]  # T * 3, H, W
    state_color = data["state_color"]  # 3, H, W
    gt_action = data["gt_action"]  # T, D
    T_cam0_cam = data["T_cam0_cam"]  # 4, 4
    T_cam_cam0 = np.linalg.inv(T_cam0_cam)
    gt_state = transform_state(gt_state, T_cam_cam0)

    gt_state = rearrange(gt_state, "(t c) h w -> (h w) t c", c=3)  # N, T, 3
    state_valid = rearrange(state_valid, "(t c) h w -> (h w) t c", c=3)[:, 0, 0]  # N'
    state_color = rearrange(state_color, "c h w -> (h w) c", c=3)  # N, 3

    gt_state = gt_state[state_valid > 0]  # N', T, 3
    state_color = state_color[state_valid > 0]  # N', 3

    # compute motion plan
    gt_state = gt_state.numpy()
    state_color = state_color.numpy()
    gt_action = gt_action[:, 24:].numpy()
    gt_action_root = DatasetUtils.get_root_transformation(gt_action)
    gt_action_finger_tips = DatasetUtils.get_finger_tips_trajectory(gt_action)
    gt_action_finger_tips_init = gt_action_finger_tips[0]  # [5, 3]
    pose_init = gt_action_root[0]
    init_root_rel_finger_tips = (
        gt_action_finger_tips_init - pose_init[:3, 3][None]
    )  # [5, 3]
    root_rel_finger_tips = init_root_rel_finger_tips[None, :, :].repeat(
        gt_action_root.shape[0], axis=0
    )  # [T, 5, 3]

    # Ways to acquire the trajectory
    motion_plan, flows = flow_planner.compute_motion_plan(
        gt_state, pose_init[:3, 3], num_steps=15, radius=0.3
    )
    traj = flow_planner.apply_motion_plan(
        pose_init, motion_plan, filter_threshold=0.01, trajectory_length=15
    )
    traj_hand = flow_planner.build_hand_trajectory(
        traj,
    )

    # Visualize the motion plan
    start_state = gt_state[:, 0, :]  # N', 3
    goal_state = gt_state[:, -1, :]  # N', 3
    traj_root = DatasetUtils.get_root_transformation(traj_hand)
    pcd_start = DatasetUtils.visualize_points(
        start_state, state_color, size=0.01, as_spheres=True
    )
    pcd_goal = DatasetUtils.visualize_points(
        goal_state, state_color, size=0.01, as_spheres=True
    )
    vis_traj = DatasetUtils.visualize_6d_trajectory(
        traj_root, to_mesh=True, cmap_name="turbo", size=0.01
    )
    # vis_traj = DatasetUtils.visualize_6d_trajectory(
    #     gt_action_root, to_mesh=True, cmap_name="plasma", size=0.005
    # )
    vis = [pcd_start, pcd_goal, vis_traj]
    vis += [DatasetUtils.visualize_fingertips_trajectory(traj_hand)]
    o3d.visualization.draw(vis)

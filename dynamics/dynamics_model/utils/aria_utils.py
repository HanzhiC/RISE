import numpy as np
import cv2
import os
import sys
import utils.dataset_utils as DatasetUtils
import open3d as o3d
from compas.geometry import oriented_bounding_box_numpy
from transformations import rotation_matrix
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from typing import List
import pandas as pd
from torch.utils.data import Dataset
# from torchcodec.decoders import VideoDecoder
import av
import torch
import numpy as np
from typing import Optional
import torch.nn.functional as F     

T_z_m90 = np.array(
    [
        [0, -1, 0, 0],  # cos(90), -sin(90), 0, 0
        [1, 0, 0, 0],  # sin(90),  cos(90), 0, 0
        [0, 0, 1, 0],  # 0,        0,       1, 0
        [0, 0, 0, 1],  # 0,        0,       0, 1
    ]
).T


def rotate_uv_coordinates_90_degrees(
    uv_coords: np.ndarray,
    direction: str = "clockwise",
    image_size: Tuple[int, int] = (1.0, 1.0),
):
    """
    Rotate UV coordinates by 90 degrees.

    Args:
        uv_coords: Array of UV coordinates with shape (N, 2) where each row is [u, v]
        direction: "clockwise" or "counterclockwise"
        image_size: Size of the image (width, height) in UV space (default: 1.0, 1.0)

    Returns:
        np.ndarray: Rotated UV coordinates with same shape as input
    """

    if uv_coords.ndim != 2 or uv_coords.shape[1] != 2:
        raise ValueError("uv_coords must be a 2D array with shape (N, 2)")

    # Normalize UV coordinates to [0, 1] range if they're not already
    width, height = image_size
    u_coords = uv_coords[:, 0] / width
    v_coords = uv_coords[:, 1] / height

    if direction.lower() == "clockwise":
        # Clockwise 90 degree rotation:
        # u_new = v_old
        # v_new = 1 - u_old
        u_new = v_coords
        v_new = 1.0 - u_coords
    elif direction.lower() == "counterclockwise":
        # Counterclockwise 90 degree rotation:
        # u_new = 1 - v_old
        # v_new = u_old
        u_new = 1.0 - v_coords
        v_new = u_coords
    else:
        raise ValueError("direction must be 'clockwise' or 'counterclockwise'")

    # Scale back to original image size
    u_new = u_new * width
    v_new = v_new * height

    return np.column_stack([u_new, v_new])


def compute_bbox3d_orientation_pca(corners):
    """Compute orientation using PCA on corner points"""
    # Center the points
    center = np.mean(corners, axis=0)
    centered_corners = corners - center

    # Compute covariance matrix
    cov_matrix = np.cov(centered_corners.T)

    # Get eigenvectors (principal components)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending order) - THIS IS CRITICAL!
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    return eigenvectors, center


def compute_object_bbox3d_and_pose(points, filtering_percentile=3):
    # Compute the 3D bbox for the object
    center = np.mean(points, axis=0)
    points = points - center

    if filtering_percentile > 0:
        lower_percentile = filtering_percentile
        upper_percentile = 100 - filtering_percentile

        # Vectorized percentile filtering (much faster than loop)
        lower_bounds = np.percentile(points, lower_percentile, axis=0)
        upper_bounds = np.percentile(points, upper_percentile, axis=0)

        # Create mask for all dimensions at once
        mask = np.all((points >= lower_bounds) & (points <= upper_bounds), axis=1)
        if mask.sum() < 10:
            return None, None
        points = points[mask]

    bbox_3d_corners = oriented_bounding_box_numpy(points)
    bbox_3d_corners = np.array(bbox_3d_corners)
    bbox_3d_corners = bbox_3d_corners + center
    # Compute the orientation of the bbox
    rot_pca, center_pca = compute_bbox3d_orientation_pca(bbox_3d_corners)
    T_world_object = np.eye(4)
    T_world_object[:3, :3] = rot_pca
    T_world_object[:3, 3] = center_pca
    return bbox_3d_corners, T_world_object


def align_bbox3d_with_gravity(
    bbox_3d_corners, T_world_object, gravity_vector=np.array([0, 0, 1])
):
    """
    Align the z-axis of the object with the gravity vector (upwards)
    This ensures the object's z-axis is always pointing up
    """
    # Extract rotation and translation from the transformation matrix
    R_world_object = T_world_object[:3, :3]
    t_world_object = T_world_object[:3, 3]
    T_object_world = np.linalg.inv(T_world_object)

    # Get the current axes of the object
    x_axis = R_world_object[:, 0]  # Current x-axis
    y_axis = R_world_object[:, 1]  # Current y-axis
    z_axis = R_world_object[:, 2]  # Current z-axis

    corners_in_object_frame = DatasetUtils.transform_points(
        bbox_3d_corners, T_object_world
    )

    # We want to align the z-axis with gravity (upwards)
    # First, find which current axis is most aligned with gravity
    dot_products = [
        abs(np.dot(x_axis, gravity_vector)),
        abs(np.dot(y_axis, gravity_vector)),
        abs(np.dot(z_axis, gravity_vector)),
    ]

    # Find the axis most aligned with gravity
    most_aligned_idx = np.argmax(dot_products)
    most_aligned_axis = R_world_object[most_aligned_idx]
    # print(f"Most aligned axis: {most_aligned_idx}")

    # If the most aligned axis is pointing in the wrong direction, flip it
    if np.dot(most_aligned_axis, gravity_vector) < 0:
        R_world_object[:, most_aligned_idx] = -R_world_object[:, most_aligned_idx]

    x_axis = R_world_object[:, 0]
    y_axis = R_world_object[:, 1]
    z_axis = R_world_object[:, 2]

    if most_aligned_idx == 0:
        new_x_axis = gravity_vector
        new_y_axis = np.cross(z_axis, new_x_axis)
        new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
        new_z_axis = np.cross(new_x_axis, new_y_axis)
        new_z_axis = new_z_axis / np.linalg.norm(new_z_axis)
        new_R_world_object = np.stack([-new_z_axis, new_y_axis, new_x_axis], axis=1)
        corners_in_object_frame = (
            corners_in_object_frame @ rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3].T
        )

    elif most_aligned_idx == 1:
        new_y_axis = gravity_vector
        new_x_axis = -np.cross(z_axis, new_y_axis)
        new_x_axis = new_x_axis / np.linalg.norm(new_x_axis)
        new_z_axis = -np.cross(new_y_axis, new_x_axis)
        new_z_axis = new_z_axis / np.linalg.norm(new_z_axis)
        new_R_world_object = np.stack([new_x_axis, -new_z_axis, new_y_axis], axis=1)
        corners_in_object_frame = (
            corners_in_object_frame @ rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3].T
        )
    elif most_aligned_idx == 2:
        new_z_axis = gravity_vector
        new_x_axis = np.cross(y_axis, new_z_axis)
        new_x_axis = new_x_axis / np.linalg.norm(new_x_axis)
        new_y_axis = np.cross(new_z_axis, new_x_axis)
        new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
        new_R_world_object = np.stack([new_x_axis, new_y_axis, new_z_axis], axis=1)

    new_T_world_object = np.eye(4)
    new_T_world_object[:3, :3] = new_R_world_object
    new_T_world_object[:3, 3] = t_world_object  # Keep the same center

    bbox_3d_corners_aligned = DatasetUtils.transform_points(
        corners_in_object_frame, new_T_world_object
    )

    return bbox_3d_corners_aligned, new_T_world_object


def get_sparse_depth(world_points, T_world_cam, intr, height, width):
    world_points_cam = DatasetUtils.transform_points(
        world_points, np.linalg.inv(T_world_cam)
    )
    world_points_cam = world_points_cam[world_points_cam[:, 2] > 0]
    world_points_cam_norm = world_points_cam / (
        world_points_cam[:, 2:3] + 1e-6
    )  # [X/Z, Y/Z, 1], (N, 3)
    d_points = world_points_cam[:, 2]
    uv_points = (world_points_cam_norm @ intr.T)[:, :2].astype(np.int32)  # (N, 2)
    valid_mask = (
        (uv_points[:, 0] > 0)
        & (uv_points[:, 0] < width)
        & (uv_points[:, 1] > 0)
        & (uv_points[:, 1] < height)
    )
    uv_points = uv_points[valid_mask]
    d_points = d_points[valid_mask]

    # Vectorized sparse depth map creation (much faster than loop)
    depth_sparse = np.ones((height, width)) * 1e6
    if len(uv_points) > 0:
        # Create linear indices for the uv_points
        linear_indices = uv_points[:, 1] * width + uv_points[:, 0]

        # Sort by depth (ascending) to handle overlapping points
        sort_indices = np.argsort(d_points)  # Sort by depth (ascending)
        sorted_linear_indices = linear_indices[sort_indices]
        sorted_depths = d_points[sort_indices]

        # Use advanced indexing to fill the depth map
        depth_sparse.flat[sorted_linear_indices] = sorted_depths
    depth_sparse[depth_sparse == 1e6] = 0
    return depth_sparse


def synchronize_hands_trajectories(
    left_hand_traj,
    right_hand_traj,
    left_hand_timestamps,
    right_hand_timestamps,
    left_hand_valid,
    right_hand_valid,
    minimal_valid_points=5,
    num_points=80,
):
    """
    left_hand_traj: [H, 3]
    right_hand_traj: [H, 3]
    left_hand_timestamps: [H]
    right_hand_timestamps: [H]
    left_hand_valid: [H]
    right_hand_valid: [H]
    """

    INVALID_WAYPOINT_VALUE = -np.ones(3) * 1000

    # Step 1: fit a curve to the two trajectories
    if left_hand_valid.sum() > minimal_valid_points:
        left_hand_traj_interp, left_hand_timestamps_interp, left_hand_curves = (
            DatasetUtils.interpolate_trajectory(
                left_hand_timestamps[left_hand_valid > 0],
                left_hand_traj[left_hand_valid > 0],
            )
        )
        left_hand_min_timestamp = left_hand_timestamps_interp[0]
        left_hand_max_timestamp = left_hand_timestamps_interp[-1]
    else:
        left_hand_traj = None
        left_hand_timestamps = None
        left_hand_curves = None
        left_hand_min_timestamp = np.inf
        left_hand_max_timestamp = -np.inf

    if right_hand_valid.sum() > minimal_valid_points:
        right_hand_traj_interp, right_hand_timestamps_interp, right_hand_curves = (
            DatasetUtils.interpolate_trajectory(
                right_hand_timestamps[right_hand_valid > 0],
                right_hand_traj[right_hand_valid > 0],
            )
        )
        right_hand_min_timestamp = right_hand_timestamps_interp[0]
        right_hand_max_timestamp = right_hand_timestamps_interp[-1]
    else:
        right_hand_traj = None
        right_hand_timestamps = None
        right_hand_curves = None
        right_hand_min_timestamp = np.inf
        right_hand_max_timestamp = -np.inf

    # Step 2: Find the minimal time between the two trajectories
    hand_traj_min_timestamp = min(left_hand_min_timestamp, right_hand_min_timestamp)
    hand_traj_max_timestamp = max(left_hand_max_timestamp, right_hand_max_timestamp)
    hand_traj_timestamps = np.linspace(
        hand_traj_min_timestamp, hand_traj_max_timestamp, num_points
    )

    # Step 3: Leverage new time stamps to query the curves of the two trajectories
    if left_hand_curves is not None:
        left_hand_traj_x, left_hand_traj_y, left_hand_traj_z = (
            left_hand_curves[0](hand_traj_timestamps),
            left_hand_curves[1](hand_traj_timestamps),
            left_hand_curves[2](hand_traj_timestamps),
        )
        left_hand_traj = np.stack(
            [left_hand_traj_x, left_hand_traj_y, left_hand_traj_z], axis=1
        )
        left_hand_traj_mask = np.logical_and(
            hand_traj_timestamps >= left_hand_min_timestamp,
            hand_traj_timestamps <= left_hand_max_timestamp,
        )
    else:
        left_hand_traj = np.ones((num_points, 3)) * INVALID_WAYPOINT_VALUE
        left_hand_traj_mask = np.zeros(num_points, dtype=bool)

    if right_hand_curves is not None:
        right_hand_traj_x, right_hand_traj_y, right_hand_traj_z = (
            right_hand_curves[0](hand_traj_timestamps),
            right_hand_curves[1](hand_traj_timestamps),
            right_hand_curves[2](hand_traj_timestamps),
        )
        right_hand_traj = np.stack(
            [right_hand_traj_x, right_hand_traj_y, right_hand_traj_z], axis=1
        )
        right_hand_traj_mask = np.logical_and(
            hand_traj_timestamps >= right_hand_min_timestamp,
            hand_traj_timestamps <= right_hand_max_timestamp,
        )
    else:
        right_hand_traj = np.ones((num_points, 3)) * INVALID_WAYPOINT_VALUE
        right_hand_traj_mask = np.zeros(num_points, dtype=bool)

    # step 4: return allthe results
    assert left_hand_traj.shape == right_hand_traj.shape == (num_points, 3)
    assert left_hand_traj_mask.shape == right_hand_traj_mask.shape == (num_points,)
    assert hand_traj_timestamps.shape == (num_points,)
    assert left_hand_traj_mask[0] > 0 or right_hand_traj_mask[0] > 0
    return (
        left_hand_traj,
        right_hand_traj,
        left_hand_traj_mask,
        right_hand_traj_mask,
        hand_traj_timestamps,
    )


def interpolate_camera_trajectory(camera_trajectory, timestamps, num_points=100):
    """
    T_cam0_cam_traj: [N, 4, 4]
    timestamps_mp4: [N]
    """
    target_timestamps = np.linspace(
        timestamps[0], timestamps[-1], num_points, dtype=np.float64
    )
    T_cam0_cam_traj = []
    for i in range(len(target_timestamps)):
        if i == 0:
            T_i = camera_trajectory[0]
        elif i == len(target_timestamps) - 1:
            T_i = camera_trajectory[-1]
        else:
            # Find the timestamp in the camera trajectory that is smaller and greater than the target timestamp
            t = target_timestamps[i]
            # Find the nearest timestamp in the camera trajectory that is smaller and greater than the target timestamp
            idx_smaller = np.where(timestamps <= t)[0][-1]
            idx_greater = np.where(timestamps > t)[0][0]
            t_s, t_g = timestamps[idx_smaller], timestamps[idx_greater]
            factor = (t - t_s) / (t_g - t_s)
            T_s, T_g = camera_trajectory[idx_smaller], camera_trajectory[idx_greater]
            quat_s, quat_g = (
                R.from_matrix(T_s[:3, :3]).as_quat(),
                R.from_matrix(T_g[:3, :3]).as_quat(),
            )
            tra_s, tra_g = T_s[:3, 3], T_g[:3, 3]
            quat_i = quat_s * factor + quat_g * (1 - factor)
            tra_i = tra_s * factor + tra_g * (1 - factor)
            quat_i = quat_i / np.linalg.norm(quat_i)
            T_i = np.eye(4)
            T_i[:3, :3] = R.from_quat(quat_i).as_matrix()
            T_i[:3, 3] = tra_i
        T_cam0_cam_traj.append(T_i)
    T_cam0_cam_traj = np.stack(T_cam0_cam_traj, axis=0).astype(np.float32)  # [N, 4, 4]
    cam_traj_timestamps = target_timestamps.astype(np.float32)
    assert T_cam0_cam_traj.shape == (num_points, 4, 4)
    assert cam_traj_timestamps.shape == (num_points,)
    return T_cam0_cam_traj, cam_traj_timestamps


# The i-th number of each category represents the semantic label number corresponding to the i-th color
HOI4D_CATEGORY2LABEL_MAP = {
    "C1": [24, 47],
    "C2": [17, 47, 17],
    "C3": [25, 47, 25],
    "C4": [19, 47, 19, 19, 16],
    "C5": [16, 47, 17],
    "C6": [27, 47, 27, 16],
    "C7": [15, 47, 46],
    "C8": [9, 47, 9, 9, 9, 47],
    "C9": [48, 47, 48],
    "C10": [22, 47],
    "C11": [23, 47, 23],
    "C12": [21, 47, 17],
    "C13": [31, 47, 46],
    "C14": [10, 47, 10, 16],
    "C15": [20, 47, 20, 20, 20, 20],
    "C16": [30, 47, 30, 30],
    "C17": [29, 47, 29, 29, 29, 47],
    "C18": [28, 47, 28, 28, 46],
    "C19": [26, 47, 26, 26, 26, 26],
    "C20": [2, 47, 47],
}

# The i-th number of each category represents the instance label number corresponding to the ith color (extended backward based on the original instance number)
HOI4D_CATEGORY2LABEL_MAP_INSTANCESEG = {
    "C1": [1, 2],
    "C2": [1, 2, 3],
    "C3": [1, 2, 1],
    "C4": [1, 2, 1, 1, 3],
    "C5": [1, 2, 3],
    "C6": [1, 2, 1, 3],
    "C7": [1, 2, 3],
    "C8": [1, 2, 1, 3, 3, 4],
    "C9": [1, 2, 1],
    "C10": [1, 2],
    "C11": [1, 2, 1],
    "C12": [1, 2, 3],
    "C13": [1, 2, 3],
    "C14": [1, 2, 1, 3],
    "C15": [1, 2, 1, 1, 1, 1],
    "C16": [1, 2, 1, 1],
    "C17": [1, 2, 1, 1, 1, 3],
    "C18": [1, 2, 1, 1, 3],
    "C19": [1, 2, 1, 1, 1, 1],
    "C20": [1, 2, 3],
}


def read_depth_video(filepath, target_fps=15):
    container = av.open(filepath)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # Determine frame skipping based on source fps
    input_fps = float(stream.average_rate)
    frame_interval = round(input_fps / target_fps)

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % frame_interval != 0:
            continue

        # frame.to_ndarray(format='gray16') is often needed for depth
        arr = frame.to_ndarray(
            format="gray16le"
        )  # or try 'gray16' depending on encoding
        frames.append(arr)

    return np.stack(frames) / 1000.0  # Shape: [T, H, W] with uint16 values


# def read_color_video(filepath: str, target_fps: float = None):
#     """
#     Read an RGB color video into a torch.Tensor.

#     Args:
#         filepath (str): Path to video file.
#         target_fps (float, optional): If given, subsample to this FPS.

#     Returns:
#         video_tensor (torch.Tensor): Tensor of shape [T, C, H, W], float32 in [0, 1].
#     """
#     # Load video using torchcodec (uses torchvision backend)
#     decoder = VideoDecoder(filepath)
#     orig_fps = int(decoder.metadata.average_fps)

#     if target_fps is not None and target_fps < orig_fps:
#         print(f"Subsampling color video from {orig_fps} to {target_fps} FPS")
#         num_frames = video.shape[0]
#         target_num_frames = int(np.ceil(num_frames * target_fps / orig_fps))
#         indices = np.linspace(0, num_frames - 1, target_num_frames).astype(int)
#         video = video[indices]

#     frames = decoder[:].data.numpy().transpose(0, 2, 3, 1)
#     return frames


def second_to_frame_index(time_s: float, fps: float):
    return int(time_s * fps)


def read_o3d_poses(fpath: str):
    pose_anno = o3d.io.read_pinhole_camera_trajectory(fpath)
    poses = []
    for param in pose_anno.parameters:
        pose = np.linalg.inv(np.array(param.extrinsic))
        # pose[:3, 3] = pose[:3, 3] / 1000.0
        # print(pose[:3, 3])
        poses.append(pose)
    poses = np.stack(poses, axis=0)
    return poses


def get_color_map(N=256):
    """
    Return the color (R, G, B) of each label index.
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def parse_hoi4d_mask_backup(mask, category):
    # image = cv2.imread(img_path)[:, :, ::-1]
    # assert image.shape == (1080, 1920, 3)
    # # image = shift_mask(image, img_path)
    color_map = get_color_map(256)

    arrs = []
    labels = []  # semantic segmentation
    labels_instanceseg = []  # instance segmentation
    for i in range(10):
        color = color_map[i + 1]
        valid = (
            (mask[..., 0] == color[0])
            & (mask[..., 1] == color[1])
            & (mask[..., 2] == color[2])
        )
        if np.sum(valid) == 0:
            continue
        if i >= len(HOI4D_CATEGORY2LABEL_MAP[category]):
            continue
        arrs.append(valid)
        labels.append(HOI4D_CATEGORY2LABEL_MAP[category][i])
        labels_instanceseg.append(HOI4D_CATEGORY2LABEL_MAP_INSTANCESEG[category][i])
    return arrs, np.array(labels_instanceseg), np.array(labels)


def parse_hoi4d_mask(mask):
    # image = cv2.imread(img_path)[:, :, ::-1]
    # assert image.shape == (1080, 1920, 3)
    # # image = shift_mask(image, img_path)
    color_map = get_color_map(256)

    arrs = []
    for i in range(10):
        color = color_map[i + 1]
        valid = (
            (mask[..., 0] == color[0])
            & (mask[..., 1] == color[1])
            & (mask[..., 2] == color[2])
        )
        if np.sum(valid) == 0:
            continue
        arrs.append(valid)
    return arrs


def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def get_grid_queries(
    grid_size: int,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
):
    if len(depths.shape) == 3:
        return get_grid_queries(
            grid_size=grid_size,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
        ).squeeze(0)

    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size).to(intrinsics.device)  # type: ignore
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, 0][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, 0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, 0])
    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum("ij,bnj->bni", inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat(
        [local_coords, torch.ones_like(local_coords[..., :1])], dim=-1
    )
    world_coords = torch.einsum("ij,bnj->bni", inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.zeros_like(xy[:, :, :1]), world_coords], dim=-1).to(depths.device)  # type: ignore
    return queries


def get_ray_map(T_wc, intrinsics, h, w, in_pluecker=True):
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = T_wc[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (T_wc @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    if in_pluecker:
        rord = np.cross(ro, rd, axis=-1)
        ray_map = np.concatenate([rord, rd], axis=-1)
    else:
        ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map


def rotation_6d_to_matrix(d6: torch.Tensor):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) :
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

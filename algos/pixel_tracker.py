import sys
import os
from typing import Tuple

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, "third_party", "co-tracker"))
sys.path.append(os.path.join(current_dir, "third_party", "unimatch"))
sys.path.append(os.path.join(current_dir, "third_party", "TAPIP3D"))

# import utils.dataset_utils as DatasetUtils
# from unimatch.unimatch import UniMatch  # type: ignore
# from cotracker.utils.visualizer import Visualizer  # type: ignore
# from cotracker.predictor import CoTrackerPredictor  # type: ignore
# from cotracker.utils.visualizer import Visualizer, read_video_from_path  # type: ignore
from tapip3d_utils.inference_utils import (  # type: ignore
    load_model,
    inference,
    get_grid_queries,
)  # type: ignore
from tapip3d_utils.moge_utils3d import (  # type: ignore
    depth_edge,
    normals_edge,
    points_to_normals,
)  # type: ignore

import torch.nn.functional as F
import numpy as np
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
import time
from models.layers_2d import Project3D

def resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized


def _filter_one_depth(
    depth: np.ndarray, depth_rtol: float, normal_tol: float, intrinsics: np.ndarray
)  :
    inv_intrinsics = np.linalg.inv(intrinsics)
    uv_grid = np.meshgrid(
        np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing="xy"
    )
    uv_homo = np.stack([uv_grid[0], uv_grid[1], np.ones_like(uv_grid[0])], axis=-1)
    xyz_homo = np.einsum("ij,uvj->uvi", inv_intrinsics, uv_homo)
    xyz_homo = xyz_homo * depth[..., None]
    valid_mask = depth > 0.0
    normals, normals_mask = points_to_normals(xyz_homo, mask=valid_mask)
    depth_in = depth.copy().astype(np.float32)
    assert np.all(depth_in < 1e9)
    depth_in[~valid_mask] = 1e9
    edge_mask = depth_edge(depth_in, rtol=depth_rtol, mask=valid_mask) & normals_edge(
        normals, tol=normal_tol, mask=normals_mask
    )
    distance, indices = ndimage.distance_transform_edt(edge_mask | (~valid_mask), return_indices=True)  # type: ignore
    filled_depth = depth.copy()
    filled_depth[edge_mask] = depth[tuple(indices)][edge_mask]
    return filled_depth


class CoTrackerPixelPredictor:
    def __init__(
        self,
        offline=True,
        checkpoint_path="third_party/co-tracker/checkpoints/scaled_offline.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        if offline:
            window_len = 60
        else:
            window_len = 16
        if checkpoint_path is not None:
            self.predictor = CoTrackerPredictor(
                checkpoint_path, offline=offline, window_len=window_len
            )
        else:
            self.predictor = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline"
            )
        self.predictor.to(device)
        self.device = device

    def predict(
        self,
        images,
        query_frame_mask=None,
        grid_query_frame=0,
        grid_size=16,
        backward_tracking=False,
        verbose_filename="video",
        verbose=False,
    ):
        """
        images: list of images, each image is a numpy array of shape (H, W, 3)
        query_frame_mask: numpy array of shape (H, W)
        grid_query_frame: int, the frame index of the query frame
        grid_size: int, the size of the grid
        backward_tracking: bool, whether to use backward tracking; need to provide query_frame_mask
        verbose: bool, whether to print verbose output
        """
        # images = np.stack(images, axis=0) # (T, H, W, 3)
        video = (
            torch.from_numpy(images).permute(0, 3, 1, 2)[None].float().to(self.device)
        )
        if query_frame_mask is not None:
            query_frame_mask = torch.from_numpy(query_frame_mask)[None, None]
            query_frame_mask = query_frame_mask.to(self.device)
        pred_tracks, pred_visibility = self.predictor(
            video,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=backward_tracking,
            segm_mask=query_frame_mask,
        )
        if verbose:
            vis = Visualizer(save_dir="./.tmp", pad_value=120, linewidth=3)
            vis.visualize(
                video,
                pred_tracks,
                pred_visibility,
                query_frame=0 if backward_tracking else grid_query_frame,
                filename=verbose_filename,
            )
            print("... Pixel tracks saved to ./tmp")
        pred_tracks = pred_tracks[0].cpu().numpy()  # [T, N, 2]
        pred_visibility = pred_visibility[0].cpu().numpy()  # [T, N]

        # Compute the tracking quality
        tracking_quality = pred_visibility.mean(axis=1)  # [T]

        return pred_tracks, pred_visibility, tracking_quality


class UniMatchFlowPredictor:
    def __init__(
        self,
        pretrained_path="third_party/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.task = "flow"
        self.model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task=self.task,
        )

        self.model.eval()
        self.model.load_state_dict(torch.load(pretrained_path)["model"], strict=True)

        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        rgbs,
        infer_batch_size=4,
    ):

        # Run the inference in a sliding window manner
        image1, image2 = rgbs[:-1], rgbs[1:]
        batch_size, height, width, _ = image1.shape

        image1 = (
            torch.from_numpy(image1).permute(0, 3, 1, 2).float().to(self.device)
        )  # [B, C, H, W]
        image2 = (
            torch.from_numpy(image2).permute(0, 3, 1, 2).float().to(self.device)
        )  # [B, C, H, W]

        padding_factor = 32
        nearest_size = [
            int(np.ceil(height / padding_factor)) * padding_factor,
            int(np.ceil(width / padding_factor)) * padding_factor,
        ]

        # resize to nearest size or specified size
        max_inference_size = [384, 768]
        inference_size = [
            min(max_inference_size[0], nearest_size[0]),
            min(max_inference_size[1], nearest_size[1]),
        ]

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(
                image1, size=inference_size, mode="bilinear", align_corners=True
            )
            image2 = F.interpolate(
                image2, size=inference_size, mode="bilinear", align_corners=True
            )

        # Run the inference but with a maximum batch size
        if batch_size > infer_batch_size:
            flow_preds = []
            for i in range(0, batch_size, infer_batch_size):
                step_size = min(infer_batch_size, batch_size - i)
                image1_batch = image1[i : i + step_size]
                image2_batch = image2[i : i + step_size]
                results_dict = self.model(
                    image1_batch,
                    image2_batch,
                    attn_type="swin",
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1],
                    num_reg_refine=6,
                    task="flow",
                )

                flow_pr = results_dict["flow_preds"][-1]  # [B', 2, H, W]
                flow_preds.append(flow_pr)
            flow_preds = torch.cat(flow_preds, dim=0)  # [B, 2, H, W]
        else:
            results_dict = self.model(
                image1,
                image2,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task="flow",
            )
            flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]
            flow_preds = flow_pr

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_preds = F.interpolate(
                flow_preds, size=ori_size, mode="bilinear", align_corners=True
            )
            flow_preds[:, 0] = flow_preds[:, 0] * ori_size[-1] / inference_size[-1]
            flow_preds[:, 1] = flow_preds[:, 1] * ori_size[-2] / inference_size[-2]
        flow_preds = flow_preds.cpu().numpy()  # [B, 2, H, W]
        assert flow_preds.shape == (batch_size, 2, height, width)
        assert flow_preds.shape[0] == rgbs.shape[0] - 1
        return flow_preds


class TAPIP3DPixelPredictor:
    # DEFAULT_QUERY_GRID_SIZE = 64

    def __init__(
        self, resolution_factor=2, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = load_model(
            "weights/tapip3d_final.pth",
        )
        self.model.to(device)
        self.inference_res = (
            int(self.model.image_size[0] * np.sqrt(resolution_factor)),
            int(self.model.image_size[1] * np.sqrt(resolution_factor)),
        )
        self.model.set_image_size(self.inference_res)
        self.num_threads = 8
        self.num_iters = 6
        self.project_3d = Project3D()
        self.project_3d.to(device)

    def _prepare_inputs(self, rgbs, depths, extrinsics, intrinsics):
        """
        rgbs: [T, H, W, 3]
        depths: [T, H, W]
        extrinsics: [T, 4, 4]
        intrinsics: [T, 3, 3]
        """
        # Convert to tensor
        _original_res = depths.shape[1:3]
        intrinsics[:, 0, :] *= (self.inference_res[1] - 1) / (_original_res[1] - 1)
        intrinsics[:, 1, :] *= (self.inference_res[0] - 1) / (_original_res[0] - 1)
        with ThreadPoolExecutor(self.num_threads) as executor:
            video_futures = [
                executor.submit(
                    cv2.resize,
                    rgb,
                    (self.inference_res[1], self.inference_res[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for rgb in rgbs
            ]
            depths_futures = [
                executor.submit(
                    resize_depth_bilinear,
                    depth,
                    (self.inference_res[1], self.inference_res[0]),
                )
                for depth in depths
            ]

            video = np.stack([future.result() for future in video_futures])
            depths = np.stack([future.result() for future in depths_futures])

            depths_futures = [
                executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
                for depth, intrinsic in zip(depths, intrinsics)
            ]
            depths = np.stack([future.result() for future in depths_futures])
        video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(
            self.device
        )
        depths = torch.from_numpy(depths).float().to(self.device)
        intrinsics = torch.from_numpy(intrinsics).float().to(self.device)
        extrinsics = torch.from_numpy(extrinsics).float().to(self.device)
        return video, depths, intrinsics, extrinsics

    def predict(
        self,
        rgbs,
        depths,
        extrinsics,
        intrinsics,
        grid_size=64,
        support_grid_size=16,
        query_point=None,
        verbose=False,
    ):
        """
        rgbs: [T, H, W, 3]
        depths: [T, H, W]
        extrinsics: [T, 4, 4]
        intrinsics: [T, 3, 3]
        """
        start_time = time.time()
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics[None].repeat(rgbs.shape[0], 1, 1)  # [T, 3, 3]
        if query_point is not None:
            query_point = torch.from_numpy(query_point).float().to(self.device)
        else:
            support_grid_size = 0
            _depths = torch.from_numpy(depths).float().to(self.device)
            _depths[_depths == 0] = 1e-3
            query_point = get_grid_queries(
                grid_size=grid_size,
                depths=_depths,
                intrinsics=torch.from_numpy(intrinsics).float().to(self.device),
                extrinsics=torch.from_numpy(extrinsics).float().to(self.device),
            )
        # Sample the colors of the query points
        color0 = torch.from_numpy(rgbs[:1] / 255.0).clone().float().to(self.device).permute(0, 3, 1, 2) # [B, 3, H, W]
        height, width = color0.shape[-2], color0.shape[-1]
        intr0 = torch.from_numpy(intrinsics[:1]).float().to(self.device)  # [B, 3, 3]
        pix_coords = query_point.clone().float().to(self.device)[None, :, 1:]
        pix_coords = self.project_3d(pix_coords, intr0[0])  # [B, 2, N] # [B, 2, N]
        pix_coords[:, 0] = (pix_coords[:, 0] / width ) * 2 - 1
        pix_coords[:, 1] = (pix_coords[:, 1] / height) * 2 - 1
        pix_coords = pix_coords.view(-1, 2, grid_size, grid_size).permute(0, 2, 3, 1) # [B, G, G, 2]
        track_colors = F.grid_sample(color0, pix_coords, align_corners=True, padding_mode="border") # [B, 3, G, G]
        track_colors = track_colors.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, 3)
        track_colors = track_colors.repeat(rgbs.shape[0], 1, 1)

        # Infer the 3D coordinates
        video, depths, intrinsics, extrinsics = self._prepare_inputs(
            rgbs, depths, extrinsics, intrinsics
        )

        # Do the inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            coords, visibs = inference(
                model=self.model,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_point,
                num_iters=self.num_iters,
                grid_size=support_grid_size,
            )
        coords = coords.cpu().numpy()
        visibs = visibs.cpu().numpy()
        track_colors = track_colors.cpu().numpy()
        query_point = query_point.cpu().numpy()[:, 1:]

        if verbose:
            print(
                f"TAPIP3D inference time: {time.time() - start_time} seconds for {rgbs.shape[0]} frames with grid size {grid_size}"
            )

            # Save results
            video = video.cpu().numpy()
            depths = depths.cpu().numpy()
            intrinsics = intrinsics.cpu().numpy()
            extrinsics = extrinsics.cpu().numpy()
            np.savez(
                ".tmp/tapip3d_result.npz",
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                coords=coords,
                visibs=visibs,
                query_points=query_point,
            )
            print(
                f"Results saved to .tmp/tapip3d_result.npz.\n To visualize them, run: python third_party/TAPIP3D/visualize.py .tmp/tapip3d_result.npz "
            )
            input("Press Enter to continue...")

        return coords, visibs, track_colors, query_point


if __name__ == "__main__":
    tracker = TAPIP3DPixelPredictor()
    npz_file = "third_party/TAPIP3D/demo_inputs/dexycb.npz"
    data = np.load(npz_file)
    rgbs = data["video"]
    depths = data["depths"]
    # breakpoint()
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    tracker.predict(rgbs, depths, extrinsics, intrinsics, verbose=True)


# if __name__ == "__main__":
# tracker = PixelTracker()
# image_names = sorted(os.listdir("demo_dataset/rgb"))[:20]
# images = [cv2.imread(
#     f"demo_dataset/rgb/{image_name}")[:, :, ::-1] for image_name in image_names]
# tracker.predict(images, backward_tracking=False, verbose=True)

# tracker = UniMatchFlowPredictor()
# image_names = sorted(os.listdir("demo_dataset/rgb"))[:8]
# images = [cv2.imread(
#     f"demo_dataset/rgb/{image_name}")[:, :, ::-1] for image_name in image_names]
# # image_names = sorted(os.listdir("./third_party/unimatch/demo/flow-davis"))
# # images = [cv2.imread(
# #     f"./third_party/unimatch/demo/flow-davis/{image_name}")[:, :, ::-1] for image_name in image_names]
# # images = np.stack(images, axis=0)
# images = np.load(".tmp/dataset_idx_30_sceneflow.npz")["rgb_images"]
# flow_preds = tracker.predict(images, infer_batch_size=4)
# image1 = images[:-1]
# image2 = images[1:]
# for ii, flow_pred in enumerate(flow_preds):
#     flow_image = DatasetUtils.visualize_vector_field(flow_pred)
#     image1_vis = image1[ii]
#     image2_vis = image2[ii]
#     vis = np.concatenate([image1_vis, image2_vis, flow_image], axis=1)
#     cv2.imshow("flow", vis[:, :, ::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

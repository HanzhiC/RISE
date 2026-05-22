import numpy as np
import torch
import torch.nn as nn
import os, sys
from models.cut3r.dust3r.model import ARCroco3DStereo
from models.cut3r.dust3r.model import ARCroco3DStereoOutput
import cv2
import torchvision.transforms as tvf
from models.cut3r.dust3r.utils.device import to_cpu
from models.cut3r.dust3r.utils.camera import pose_encoding_to_camera
from models.cut3r.dust3r.post_process import estimate_focal_knowing_depth
from models.cut3r.dust3r.utils.geometry import geotrf
import viser
import time


def add_path_to_dust3r(ckpt):
    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)


class Cut3RGeometryPredictor:

    def __init__(
        self,
        model_path="./weights/cut3r_224_linear_4.pth",
        size=224,
        device="cuda",
        square_ok=False,
        verbose=True,
    ):
        add_path_to_dust3r(model_path)
        self.model_path = model_path
        self.size = size
        self.device = device
        self.model = ARCroco3DStereo.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.square_ok = square_ok
        self.verbose = verbose

    def _resize_numpy_image(self, img, long_edge_size):
        S = max(img.shape[:2])
        if S > long_edge_size:
            interp = cv2.INTER_LANCZOS4
        elif S <= long_edge_size:
            interp = cv2.INTER_CUBIC
        new_size = tuple(int(round(x * long_edge_size / S)) for x in img.shape[:2])
        return cv2.resize(img, new_size, interpolation=interp)

    def _prepare_input(self, rgbs):
        views = []
        for i in range(len(rgbs)):
            rgb = rgbs[i]
            H1, W1 = rgb.shape[:2]
            assert H1 == W1, "Image must be square"
            if self.size == 224:
                rgb = self._resize_numpy_image(
                    rgb, round(self.size * max(W1 / H1, H1 / W1))
                )
            else:
                rgb = self._resize_numpy_image(rgb, self.size)

            H, W = rgb.shape[:2]
            cx, cy = W // 2, H // 2
            if self.size == 224:
                half = min(cx, cy)
                rgb = rgb[cy - half : cy + half, cx - half : cx + half]
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not (self.square_ok) and W == H:
                    halfh = 3 * halfw / 4
                rgb = rgb[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
            if self.verbose:
                H2, W2 = rgb.shape[:2]
                # print(f" - adding image with resolution {W1}x{H1} --> {W2}x{H2}")
            rgb_processed = self.transform(rgb)[None]
            view = {
                "img": rgb_processed.to(self.device),
                "ray_map": torch.full(
                    (
                        rgb_processed.shape[0],
                        6,
                        rgb_processed.shape[-2],
                        rgb_processed.shape[-1],
                    ),
                    torch.nan,
                ).to(self.device),
                "img_mask": torch.tensor(True).unsqueeze(0).to(self.device),
                "ray_mask": torch.tensor(False).unsqueeze(0).to(self.device),
                "update": torch.tensor(True).unsqueeze(0).to(self.device),
                "reset": torch.tensor(False).unsqueeze(0).to(self.device),
            }
            views.append(view)
        return views

    def _prepare_output(self, outputs, use_pose=True):
        valid_length = len(outputs["pred"])
        # Only keep the outputs corresponding to one full pass.
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self_ls = [
            output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
        ]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self_ls, 0)

        # Recover camera poses.
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
        t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)
        T_c2w = torch.eye(4, device=pts3ds_self.device).repeat(
            pts3ds_self.shape[0], 1, 1
        )
        T_c2w[:, :3, :3] = R_c2w
        T_c2w[:, :3, 3] = t_c2w

        if use_pose:
            transformed_pts3ds_other = []
            for pose, pself in zip(pr_poses, pts3ds_self):
                transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
            pts3ds_other = transformed_pts3ds_other
            conf_other = conf_self

        # Estimate focal length based on depth.
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        colors = [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]

        focal = focal.cpu().numpy()
        pp = pp.cpu().numpy()
        T_c2w = T_c2w.cpu().numpy()
        R_c2w = R_c2w.cpu().numpy()
        t_c2w = t_c2w.cpu().numpy()
        intrs, extrs = [], []
        for i in range(len(T_c2w)):
            intr = np.eye(3)
            intr[0, 0] = focal[i]
            intr[1, 1] = focal[i]
            intr[0, 2] = pp[i, 0]
            intr[1, 2] = pp[i, 1]
            extr = T_c2w[i]
            intrs.append(intr)
            extrs.append(extr)

        intrs = np.stack(intrs, 0)
        extrs = np.stack(extrs, 0)
        pts3ds_other = torch.cat(pts3ds_other, 0).numpy()  # (B, H, W, 3)
        colors = torch.cat(colors, 0).numpy()  # (B, H, W, 3)
        conf_other = torch.cat(conf_other, 0).numpy()  # (B, H, W)
        return pts3ds_other, colors, conf_other, intrs, extrs

    @torch.no_grad()
    def extract_state_observation_features_offline(self, rgbs, wrap_features=False):
        views = self._prepare_input(rgbs)
        state_feats, observation_feats, shapes, poss = (
            self.model.extract_state_observation_feature(views)
        )
        assert (
            len(state_feats) - 1 == len(observation_feats) == len(shapes) == len(poss)
        )
        if wrap_features:
            state_feats, observation_feats = self.wrap_state_observation_features(
                state_feats, observation_feats
            )
            return state_feats, observation_feats
        return state_feats, observation_feats, shapes, poss

    @torch.no_grad()
    def extract_state_observation_feature_online(
        self, rgbs, wrap_features=False, max_seq_length=500
    ):
        state_feats, observation_feats, shapes, poss = (None, None, None, None)
        for rgb in rgbs:
            start_time = time.time()
            view = self._prepare_input([rgb])[0]

            state_feats, observation_feats, shapes, poss = (
                self.model.extract_state_observation_feature_streaming(
                    view,
                    state_feats,
                    observation_feats,
                    shapes,
                    poss,
                    max_seq_length=max_seq_length,
                )
            )
            end_time = time.time()
            # print(f"Feature extraction time: {end_time - start_time} seconds")
        # assert (
        #     len(state_feats) - 1 == len(observation_feats) == len(shapes) == len(poss)
        # )
        if wrap_features:
            state_feats, observation_feats = self.wrap_state_observation_features(
                state_feats, observation_feats
            )
            return state_feats, observation_feats
        return state_feats, observation_feats, shapes, poss

    @torch.no_grad()
    def wrap_state_observation_features(self, state_feats, observation_feats):
        state_feats_final = []
        observation_feats_final = []
        for i in range(len(state_feats)):
            state_feats_final.append(state_feats[i][0])  # (B, 197, 768)
            if i < len(observation_feats):
                observation_feats_final.append(
                    observation_feats[i][-1]
                )  # (B, 196, 768)
        state_feats_final = torch.cat(state_feats_final, 0).cpu().numpy()
        observation_feats_final = torch.cat(observation_feats_final, 0).cpu().numpy()
        return state_feats_final, observation_feats_final

    @torch.no_grad()
    def predict_geometry_with_features(
        self, rgbs, state_feats, observation_feats, shapes, poss, use_pose=True
    ):
        # assert (
        #     len(observation_feats) == len(state_feats) - 1
        # ), f"{len(observation_feats)}, {len(state_feats)}"
        views = self._prepare_input(rgbs)
        outputs = self.model.predict_geometry(
            views, shapes, poss, observation_feats, state_feats
        )
        preds, batch = outputs.ress, outputs.views
        results = to_cpu(dict(views=batch, pred=preds))
        pts3ds_other, colors, conf_other, intrs, extrs = self._prepare_output(
            results, use_pose=use_pose
        )
        return pts3ds_other, colors, conf_other, intrs, extrs


# Test the models
if __name__ == "__main__":
    import open3d as o3d

    extractor = Cut3RGeometryPredictor(
        model_path="./weights/cut3r_224_linear_4.pth",
        size=224,
        device="cuda",
        square_ok=False,
        verbose=True,
    )
    rgb_dir = "../cut3r-customized/examples/hdepic1/"
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
    rgb_files.sort()
    rgbs = [cv2.imread(os.path.join(rgb_dir, f))[..., [2, 1, 0]] for f in rgb_files]

    state_feats, observation_feats, shapes, poss = (
        extractor.extract_state_observation_feature_online(rgbs)
    )
    # state_feats, observation_feats, shapes, poss = (
    #     extractor.extract_state_observation_features_offline(rgbs)
    # )

    # # Meassure the difference between the stream and the offline features
    # for ii in range(len(state_feats_online)):
    #     state_feats_online_ii = state_feats_online[ii][0]
    #     state_feats_offline_ii = state_feats_offline[ii][0]
    #     diff = torch.mean(torch.abs(state_feats_online_ii - state_feats_offline_ii))
    #     print(
    #         f"state_feats between the stream and the offline features for view {ii}: {diff}"
    #     )

    # # Meassure the difference between the stream and the offline features
    # for ii in range(len(observation_feats_online)):
    #     observation_feats_online_ii = observation_feats_online[ii][-1]
    #     observation_feats_offline_ii = observation_feats_offline[ii][-1]
    #     diff = torch.mean(
    #         torch.abs(observation_feats_online_ii - observation_feats_offline_ii)
    #     )
    #     print(
    #         f"observation_feats between the stream and the offline features for view {ii}: {diff}"
    #     )

    pts3ds_other, colors, conf_other, intrs, extrs = (
        extractor.predict_geometry_with_features(
            rgbs,
            state_feats,
            observation_feats,
            shapes,
            poss,
        )
    )
    vis = []
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    for i in range(len(pts3ds_other)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3ds_other[i].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors[i].reshape(-1, 3))
        vis.append(pcd)

        server.scene.add_point_cloud(
            name=f"/frame_{i}",
            points=np.array(pcd.points),
            colors=np.array(pcd.colors),
            point_size=0.01,
            point_shape="circle",
        )

    while True:
        pass
    # o3d.visualization.draw(vis)
    # state_feats_final, observation_feats_final = (
    #     extractor.wrap_state_observation_features(
    #         state_feats_online, observation_feats_online
    #     )
    # )

    # breakpoint()

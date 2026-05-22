import numpy as np
import cv2
import torch
import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(
    current_dir, "third_party", "vggt-pytorch-inference"))
sys.path.append(os.path.join(
    current_dir, "third_party", "UniDepth"))
sys.path.append(os.path.join(
    current_dir, "third_party", "UniK3D"))
sys.path.append(os.path.join(
    current_dir, "third_party", "Depth-Anything-3/src"))
# Import third-party libraries
from vggt_inference import VGGTInference  # type: ignore
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old  # type: ignore
from unidepth.utils.camera import Pinhole  # type: ignore
from unik3d.utils.camera import MEI, OPENCV, BatchCamera, Fisheye624, Pinhole, Spherical  # type: ignore
from unik3d.models import UniK3D  # type: ignore
from depth_anything_3.api import DepthAnything3 # type: ignore


class Metric3DDepthPredictor:
    def __init__(self, model_name="metric3d_vit_large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.input_size = (616, 1064)  # for vit model
        self.depth_model = torch.hub.load(
            'yvanyin/metric3d', model_name, pretrain=True)
        self.depth_model.to(device).eval()  # type: ignore
        self.device = device

    def predict(self, rgbs, intr):
        """
        Predict the depth of the image using the depth model.
        Args:
            rgb_origin: The original RGB image. np.ndarray, [H, W, 3], RGB, UINT8
            intr: The intrinsic matrix of the camera. np.ndarray, [3, 3]
        """
        h_orig, w_orig = rgbs[0].shape[:2]
        image_tensors = []

        for rgb_origin in rgbs:

            intrinsic = [intr[0, 0], intr[1, 1],
                         intr[0, 2], intr[1, 2]]  # fx, fy, cx, cy
            # ajust input size to fit pretrained model
            # keep ratio resize
            input_size = (616, 1064)  # for vit model
            # input_size = (544, 1216) # for convnext model
            h, w = rgb_origin.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)
            # remember to scale intrinsic, hold depth
            intrinsic = [intrinsic[0] * scale, intrinsic[1] *
                         scale, intrinsic[2] * scale, intrinsic[3] * scale]
            # padding to input_size
            padding = [123.675, 116.28, 103.53]
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                                     pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
            pad_info = [pad_h_half, pad_h - pad_h_half,
                        pad_w_half, pad_w - pad_w_half]

            # normalize
            mean = torch.tensor([123.675, 116.28, 103.53]
                                ).float()[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
            rgb = torch.div((rgb - mean), std)
            rgb = rgb[None, :, :, :].cuda()
            image_tensors.append(rgb)

        image_tensors = torch.cat(image_tensors, dim=0).to(self.device)
        ###################### canonical camera space ######################
        # inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.depth_model.inference(  # type: ignore
                {'input': image_tensors})

        # un pad
        pred_depth = pred_depth[:, :, pad_info[0]: pred_depth.shape[-2] -
                                pad_info[1], pad_info[2]: pred_depth.shape[-1] - pad_info[3]]
        confidence = confidence[:, :, pad_info[0]: confidence.shape[-2] -
                                pad_info[1], pad_info[2]: confidence.shape[-1] - pad_info[3]]

        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, (h_orig, w_orig), mode='bilinear', align_corners=True)
        confidence = torch.nn.functional.interpolate(
            confidence, (h_orig, w_orig), mode='bilinear', align_corners=True)
        ###################### canonical camera space ######################

        canonical_to_real_scale = intrinsic[0] / 1000.0
        pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 5)

        pred_depth = pred_depth.squeeze(1).cpu().numpy()  # [B, H, W]
        confidence = confidence.squeeze(1).cpu().numpy()  # [B, H, W]
        return pred_depth, confidence

    # def predict(self, rgb_origin, intr):
    #     """
    #     Predict the depth of the image using the depth model.
    #     Args:
    #         rgb_origin: The original RGB image. np.ndarray, [H, W, 3], RGB, UINT8
    #         intr: The intrinsic matrix of the camera. np.ndarray, [3, 3]
    #     """
    #     intrinsic = [intr[0, 0], intr[1, 1],
    #                  intr[0, 2], intr[1, 2]]  # fx, fy, cx, cy
    #     # ajust input size to fit pretrained model
    #     # keep ratio resize
    #     h, w = rgb_origin.shape[:2]
    #     scale = min(self.input_size[0] / h, self.input_size[1] / w)
    #     rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)),
    #                      interpolation=cv2.INTER_LINEAR)
    #     # remember to scale intrinsic, hold depth
    #     intrinsic = [intrinsic[0] * scale, intrinsic[1] *
    #                  scale, intrinsic[2] * scale, intrinsic[3] * scale]
    #     # padding to input_size
    #     padding = [123.675, 116.28, 103.53]
    #     h, w = rgb.shape[:2]
    #     pad_h = self.input_size[0] - h
    #     pad_w = self.input_size[1] - w
    #     pad_h_half = pad_h // 2
    #     pad_w_half = pad_w // 2
    #     rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
    #                              pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    #     pad_info = [pad_h_half, pad_h - pad_h_half,
    #                 pad_w_half, pad_w - pad_w_half]

    #     # normalize
    #     mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    #     std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    #     rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    #     rgb = torch.div((rgb - mean), std)
    #     rgb = rgb[None, :, :, :].to(self.device)

    #     ###################### canonical camera space ######################
    #     # inference
    #     with torch.no_grad():
    #         pred_depth, confidence, output_dict = self.depth_model.inference({
    #             'input': rgb})

    #     # un pad
    #     pred_depth = pred_depth.squeeze()
    #     pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] -
    #                             pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    #     confidence = confidence.squeeze()
    #     confidence = confidence[pad_info[0]: confidence.shape[0] -
    #                             pad_info[1], pad_info[2]: confidence.shape[1] - pad_info[3]]

    #     # upsample to original size
    #     pred_depth = torch.nn.functional.interpolate(
    #         pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
    #     confidence = torch.nn.functional.interpolate(
    #         confidence[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()

    #     ###################### canonical camera space ######################

    #     # de-canonical transform
    #     # 1000.0 is the focal length of canonical camera
    #     canonical_to_real_scale = intrinsic[0] / 1000.0
    #     pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    #     pred_depth = torch.clamp(pred_depth, 0, 300)
    #     pred_depth = pred_depth.cpu().numpy()
    #     confidence = confidence.cpu().numpy()
    #     return pred_depth, confidence


class VGGDepthPredictor:
    def __init__(self, weight_path="third_party/vggt-pytorch-inference/weights/model.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = VGGTInference(
            weight_path=weight_path)
        self.device = device
        
    def predict(self, rgbs, intr=None):
        height, width = rgbs[0].shape[:2]
        results = self.model(rgbs)
        depth_maps = [cv2.resize(result.depth_map, (width, height),
                                 interpolation=cv2.INTER_LINEAR) for result in results]
        confidences = [cv2.resize(result.depth_conf, (width, height),
                                  interpolation=cv2.INTER_LINEAR) for result in results]
        depth_maps = np.stack(depth_maps, axis=0)
        confidences = np.stack(confidences, axis=0)
        return depth_maps, confidences

class UniDepthPredictor:
    def __init__(self, model_name="unidepth-v2-vitl14", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
        self.model.interpolation_mode = "bilinear"
        self.model.to(device).eval()
        self.device = device

    def predict(self, rgbs, intr, infer_batch_size=4):
        batch_size, height, width = rgbs.shape[:3]
        if batch_size > infer_batch_size:
            depth_preds = []
            confidence_preds = []
            for i in range(0, batch_size, infer_batch_size):
                step_size = min(infer_batch_size, batch_size - i)
                rgb_torch = torch.from_numpy(rgbs[i:i+step_size]).permute(0, 3, 1, 2) # [B, 3, H, W] => [B, H, W, 3]
                intrinsics_torch = torch.from_numpy(intr).unsqueeze(0).repeat(step_size, 1, 1) # [3, 3] => [B, 3, 3]
                predictions = self.model.infer(rgb_torch, intrinsics_torch)
                depth_preds.append(predictions["depth"].squeeze(1).cpu().numpy())
                confidence_preds.append(predictions["confidence"].squeeze(1).cpu().numpy())
            depth_preds = np.concatenate(depth_preds, axis=0)
            confidence_preds = np.concatenate(confidence_preds, axis=0)
        else:
            rgb_torch = torch.from_numpy(rgbs).permute(0, 3, 1, 2) # [B, 3, H, W] => [B, H, W, 3]
            intrinsics_torch = torch.from_numpy(intr).unsqueeze(0) # [3, 3] => [B, 3, 3]
            predictions = self.model.infer(rgb_torch, intrinsics_torch)
            depth_preds = predictions["depth"].squeeze(1).cpu().numpy() # [B, H, W]
            confidence_preds = predictions["confidence"].squeeze(1).cpu().numpy() # [B, H, W]
        assert depth_preds.shape == (batch_size, height, width)
        assert confidence_preds.shape == (batch_size, height, width)
        return depth_preds, confidence_preds

class UniK3DepthPredictor(UniDepthPredictor):
    def __init__(self, model_name="unik3d-vitl", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = UniK3D.from_pretrained(f"lpiccinelli/{model_name}")
        self.model.interpolation_mode = "bilinear"
        self.model.resolution_level = 9
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, rgbs, intr, infer_batch_size=4):
        batch_size, height, width = rgbs.shape[:3]
        if batch_size > infer_batch_size:
            depth_preds = []
            confidence_preds = []
            for i in range(0, batch_size, infer_batch_size):
                step_size = min(infer_batch_size, batch_size - i)
                rgb_torch = torch.from_numpy(rgbs[i:i+step_size]).permute(0, 3, 1, 2) # [B, 3, H, W] => [B, H, W, 3]
                intrinsics_torch = torch.from_numpy(intr).unsqueeze(0).repeat(step_size, 1, 1) # [3, 3] => [B, 3, 3]
                camera = Pinhole(K=intrinsics_torch)
                predictions = self.model.infer(rgb_torch, camera)
                depth_preds.append(predictions["depth"].squeeze(1).cpu().numpy())
                confidence_preds.append(predictions["confidence"].squeeze(1).cpu().numpy())
            depth_preds = np.concatenate(depth_preds, axis=0)
            confidence_preds = np.concatenate(confidence_preds, axis=0)
        else:
            rgb_torch = torch.from_numpy(rgbs).permute(0, 3, 1, 2) # [B, 3, H, W] => [B, H, W, 3]
            intrinsics_torch = torch.from_numpy(intr).unsqueeze(0).repeat(batch_size, 1, 1) # [3, 3] => [B, 3, 3]
            camera = Pinhole(K=intrinsics_torch)
            predictions = self.model.infer(rgb_torch, camera)
            depth_preds = predictions["depth"].squeeze(1).cpu().numpy() # [B, H, W]
            confidence_preds = predictions["confidence"].squeeze(1).cpu().numpy() # [B, H, W]
        assert depth_preds.shape == (batch_size, height, width)
        assert confidence_preds.shape == (batch_size, height, width)
        return depth_preds, confidence_preds

class DepthAnything3Predictor:

    def __init__(
        self,
        model_name="depth-anything/DA3METRIC-LARGE",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device

    def predict(self, rgbs, intr, extr=None, infer_batch_size=8):
        batch_size, height, width = rgbs.shape[:3]
        if batch_size > infer_batch_size:
            depth_preds = []
            confidence_preds = []
            for i in range(0, batch_size, infer_batch_size):
                step_size = min(infer_batch_size, batch_size - i)
                # rgb_torch = torch.from_numpy(rgbs[i:i+step_size]).permute(0, 3, 1, 2) # [B, 3, H, W] => [B, H, W, 3]
                # intrinsics_torch = torch.from_numpy(intr).unsqueeze(0).repeat(step_size, 1, 1) # [3, 3] => [B, 3, 3]
                rgb_in = [rgbs[ii] for ii in range(i, i + step_size)]
                intr_in = [intr[ii] for ii in range(i, i + step_size)]
                if extr is not None:
                    extr_in = [extr[ii] for ii in range(i, i + step_size)]
                else:
                    extr_in = None

                predictions = self.model.inference(
                    rgb_in,
                    extrinsics=extr_in,
                    intrinsics=intr_in,
                )
                depth_preds.append(predictions.depth)
                confidence_preds.append(np.ones_like(predictions.depth))
            depth_preds = np.concatenate(depth_preds, axis=0)
            confidence_preds = np.concatenate(confidence_preds, axis=0)
        else:
            rgb_in = [rgbs[ii] for ii in range(batch_size)]
            intr_in = [intr[ii] for ii in range(batch_size)]
            if extr is not None:
                extr_in = [extr[ii] for ii in range(batch_size)]
            else:
                extr_in = None
            predictions = self.model.inference(
                rgb_in,
                extrinsics=extr_in,
                intrinsics=intr_in,
            )

            depth_preds = predictions.depth # [B, H, W]
            confidence_preds = np.ones_like(predictions.depth)
        assert depth_preds.shape == (batch_size, height, width)
        assert confidence_preds.shape == (batch_size, height, width)
        return depth_preds, confidence_preds

if __name__ == "__main__":
    predictor = DepthAnything3Predictor()
    rgb_paths = ["demo_dataset/rgb/0000000000000000.jpg",
                 "demo_dataset/rgb/0000000066666666.jpg"]
    rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths]
    rgbs = np.stack(rgbs, axis=0)
    depth_maps, confidences = predictor.predict(rgbs)

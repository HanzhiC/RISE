
# Add the Grounded-SAM-2 directory to the Python path
import os
import sys
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(current_dir, "third_party", "Grounded-SAM-2")) 

from PIL import Image
from algos.depth_predictor import Metric3DDepthPredictor

from torchvision.ops import box_convert
from pathlib import Path
import pycocotools.mask as mask_util
import supervision as sv
import numpy as np
import torch
import json
import cv2

import utils.dataset_utils as DatasetUtils
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
from sam2.build_sam import build_sam2  # type: ignore
import groundingdino.datasets.transforms as T  # type: ignore

SAM2_CHECKPOINT = os.path.join(
    current_dir, "third_party", "Grounded-SAM-2", "checkpoints", "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")
GROUNDING_DINO_CONFIG = os.path.join(
    current_dir, "third_party", "Grounded-SAM-2", "grounding_dino", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(
    current_dir, "third_party", "Grounded-SAM-2", "gdino_checkpoints", "groundingdino_swint_ogc.pth")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class OpenVocabularyObjectDetector:
    def __init__(self, use_sam=True, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.detector = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=device
        )
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        if use_sam:
            sam2_model = build_sam2(
                SAM2_MODEL_CONFIG,
                SAM2_CHECKPOINT,
                device=device
            )
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        else:
            self.sam2_predictor = None

    def preprocess_image(self, image):
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def predict(self, image, text_prompt, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, use_sam=True, visualize=True):
        """
        Predict the objects in the image using the Grounding DINO model.
        If use_sam is True, use the SAM2 model to predict the objects.
        Args:
            image: The image to predict the objects in. np.ndarray,[H, W, 3], RGB, UINT8
            text_prompt: The text prompt to use for the prediction.
            box_threshold: The threshold for the bounding box.
            text_threshold: The threshold for the text.
        """
        image_pil = Image.fromarray(image)
        image_inp, _ = self.transform(image_pil, None)

        boxes, confidences, labels = predict(
            model=self.detector,
            image=image_inp,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        h, w, _ = image.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(
            boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        if len(boxes) == 0:
            return None, None, None, None, None
        
        if use_sam:
            self.sam2_predictor.set_image(image)

            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        else:
            masks = np.zeros((len(boxes), image.shape[0], image.shape[1]))
            scores = np.zeros(len(boxes))
            logits = np.zeros(len(boxes))

        if visualize:
            detections = sv.Detections(
                xyxy=boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=np.arange(len(labels))
            )

            annotated_frame = self.box_annotator.annotate(
                scene=image.copy(), detections=detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels)
            if use_sam:
                annotated_frame = self.mask_annotator.annotate(
                    scene=annotated_frame, detections=detections)
        else:
            annotated_frame = None
        return boxes, masks, confidences, labels, annotated_frame


if __name__ == "__main__":
    object_detector = OpenVocabularyObjectDetector()
    depth_predictor = Metric3DDepthPredictor()
    camera_params_path = "/home/wiss/chenh/mobile_manip/egoasis3D/demo_dataset/camera_params.json"

    # image_path = "/home/wiss/chenh/mobile_manip/egoasis3D/demo_dataset/rgb/0000627933333333.jpg"
    # image_name = image_path.split("/")[-1].split(".")[0]

    # camera_params = DatasetUtils.load_json(camera_params_path)
    # intr = np.array(camera_params["intrinsics"]).reshape(3, 3)
    # T_world_cam = np.load(".tmp/T_world_cam.npz")["T_world_cam"]
    # image = cv2.imread(image_path)[..., ::-1].copy()

    npz_path = ".tmp/rgbd.npz"
    npz = np.load(npz_path)
    image = npz["rgb_image"]
    # cv2.imshow("image", image[..., ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    intr = npz["intr"]
    T_world_cam = npz["T_world_cam"]

    depth, confidence = depth_predictor.predict(image, intr)
    # confidence_threshold = np.percentile(confidence, 90)
    # mask_confidence = np.ones_like(confidence)
    # mask_confidence[confidence < confidence_threshold] = 0
    # mask_confidence = cv2.dilate(mask_confidence, np.ones((3, 3)))
    # conf_vis = DatasetUtils.get_heatmap(confidence)[..., ::-1]
    # cv2.imshow("mask_confidence", (conf_vis * 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # depth[mask_confidence == 0] = 0

    boxes, masks, confidences, labels, annotated_frame = object_detector.predict(
        image, "drawer . cabinet . dishwasher", visualize=True)

    slam_points_path = "/home/wiss/chenh/storage/group/dataset_mirrors/01_incoming/hd-epickitchens/HD-EPIC/SLAM-and-Gaze/P07/SLAM/multi/0/semidense_points_cached.npz"
    slam_points = np.load(slam_points_path)["points"]
    # Save everything to a npz file
    save_path = ".tmp/rgbd_detection6.npz"
    np.savez(save_path, boxes=boxes, masks=masks, confidences=confidences, labels=labels,
             depth=depth, confidence=confidence, intr=intr, T_world_cam=T_world_cam, image=image, slam_points=slam_points)

    cv2.imshow("annotated_frame.jpg", annotated_frame[..., ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

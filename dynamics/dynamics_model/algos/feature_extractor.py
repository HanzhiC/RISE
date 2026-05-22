import torch


class DINOv3FeatureExtractor:
    MODEL_TO_NUM_LAYERS = {
        # "dinov3_vits16": 12,
        # "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        # "dinov3_vith16plus": 32,
        # "dinov3_vit7b16": 40,
    }
    MODEL_TO_WEIGHTS = {
        "dinov3_vitb16": "weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    }
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        num_layers: int = 12,
    ):
        self.model_name = model_name
        self.model = torch.hub.load(
            repo_or_dir="third_party/dinov3",
            model=self.model_name,
            source="local",
            weights=self.MODEL_TO_WEIGHTS[self.model_name],
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the image using the DINOv3 model.

        Args:
            image: A tensor of shape (B, C, H, W) representing the image.

        Returns:
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            features = self.model.get_intermediate_layers(
                image,
                n=range(self.MODEL_TO_NUM_LAYERS[self.model_name]),
                reshape=True,
                norm=True,
            )
        return features

    def extract_features_in_sequence(self, images: torch.Tensor, batch_size: int = 8):
        """
        Extract DINOv3 features in mini-batches to avoid OOM.

        Args:
            images: Tensor of shape (B, C, H, W).
            batch_size: Mini-batch size for inference.

        Returns:
            Tensor of shape (B, C, H, W)
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images of shape (B, C, H, W), got {tuple(images.shape)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        total_batch = images.shape[0]
        if total_batch <= batch_size:
            return self.extract_features(images)[-1] # (B, C, H, W)

        features_all = None
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            chunk = images[start:end]
            chunk_features = self.extract_features(chunk)[-1]

            if features_all is None:
                features_all = [chunk_features]
            else:
                features_all.append(chunk_features)

        return torch.cat(features_all, dim=0)

    @staticmethod
    def visualize_feature(feature, niter=5):
        """
        Visualize the features using PCA.

        Args:
            feature: A tensor of shape (B, C, H, W) representing the features.
        """
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1)
        feature_flat = feature.reshape(-1, feature.shape[-1])
        mean = feature_flat.mean(0)
        with torch.no_grad():
            U, S, V = torch.pca_lowrank(feature_flat - mean, niter=niter)
        proj_V = V[:, :3]  # (B, 3, 768, )
        low_rank = feature_flat @ proj_V
        low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
        low_rank_max = torch.quantile(low_rank, 0.99, dim=0)

        low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
        low_rank = torch.clamp(low_rank, 0, 1)

        colored_image = low_rank.reshape(feature.shape[:-1] + (3,))
        colored_image = colored_image.permute(0, 3, 1, 2) 
        return colored_image # (B, 3, H, W)

    @staticmethod
    def visualize_feature_in_sequence(
        features: torch.Tensor, niter: int = 5, batch_size: int = 8
    ):
        """
        Visualize features in mini-batches using PCA.

        Args:
            feature: A tensor of shape (B, C, H, W) representing the features.
            niter: Iterations for `torch.pca_lowrank`.
            batch_size: Mini-batch size for visualization.
        """
        if features.ndim != 4:
            raise ValueError(
                f"Expected features of shape (B, C, H, W), got {tuple(features.shape)}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        total_batch = features.shape[0]
        if total_batch <= batch_size:
            return DINOv3FeatureExtractor.visualize_feature(features, niter=niter)

        vis_all = []
        for start in range(0, total_batch, batch_size):
            end = min(start + batch_size, total_batch)
            chunk = features[start:end]
            vis_all.append(
                DINOv3FeatureExtractor.visualize_feature(chunk, niter=niter)
            )

        return torch.cat(vis_all, dim=0)

    # vis = []
    # for bi in range(B):
    #     feature_bi = feature[bi].cpu().numpy()
    #     feature_bi = feature_bi.reshape(C, H * W).T  # (HW, C)
    #     pca = PCA(n_components=3, whiten=True)
    #     pca.fit(feature_bi)
    #     projected_feature_bi = pca.transform(feature_bi)
    #     projected_feature_bi = torch.from_numpy(projected_feature_bi).view(
    #         H, W, 3
    #     )  # (HW, 3) => (H, W, 3)
    #     projected_feature_bi = torch.nn.functional.sigmoid(
    #         projected_feature_bi
    #     ).permute(
    #         2, 0, 1
    #     )  # (H, W, 3) => (3, H, W)
    #     vis.append(projected_feature_bi)
    # return torch.stack(vis)

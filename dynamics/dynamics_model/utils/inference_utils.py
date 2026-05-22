import torch
import torch.nn.functional as F
from collections import deque

def get_ray_map_in_batch(
    T_wc, intrinsics, original_size=(224, 224), target_size=(14, 14), in_pluecker=True
):
    """
    T_wc torch.Tensor  Bx4x4
    intrinsics torch.Tensor Bx3x3
    h int
    w int
    in_pluecker bool

    Returns
    -------
    ray_map torch.Tensor Bxhxwx(3 or 6)

    Returns
    -------
    ray_map torch.Tensor Bxhxwx(3 or 6)
    """
    batch_size = T_wc.shape[0]
    h, w = target_size
    factor = original_size[0] / target_size[0]
    assert original_size[1] / factor == target_size[1]
    intrinsics_patch = intrinsics.clone()

    intrinsics_patch[:, :2] /= factor
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")  # [h, w]
    grid = torch.stack([i, j, torch.ones_like(i)], dim=-1).float()  # [h, w, 3]
    grid = grid[None].to(T_wc.device).repeat(T_wc.shape[0], 1, 1, 1)  # [B, h, w, 3]
    ro = T_wc[:, :3, 3]  # [B, 3]
    rd = torch.linalg.inv(intrinsics_patch) @ grid.reshape(batch_size, -1, 3).transpose(
        1, 2
    )  # [B, 3, h*w]
    rd_homo = torch.cat([rd, torch.ones_like(rd[..., :1, :])], dim=-2)  # [B, 4, h*w]
    rd_homo = T_wc @ rd_homo  # [B, 4, h*w]
    rd = rd_homo.transpose(1, 2)[:, :, :3].view(batch_size, h, w, 3)  # [B, h, w, 3]
    rd = rd / torch.linalg.norm(rd, dim=-1, keepdims=True)  # [B, h, w, 3]
    ro = ro[:, None, None, :].repeat(1, h, w, 1)  # [B, h, w, 3]
    if in_pluecker:
        rord = torch.cross(ro, rd, dim=-1)
        ray_map = torch.concatenate([rord, rd], dim=-1)
    else:
        ray_map = torch.concatenate([ro, rd], dim=-1)
    return ray_map.contiguous()


def prepare_cut3r_input(rgb, size=224, square_ok=False):
    """
    Prepare input for Cut3R model, matching the logic from _prepare_input.

    Args:
        rgb: torch.Tensor, [B, 3, H, W], values in [0, 1]
        size: int, target size (default 224)
        square_ok: bool, whether square images are acceptable for non-224 sizes

    Returns:
        view: dict with keys ['img', 'ray_map', 'img_mask', 'ray_mask', 'update', 'reset']
    """
    assert rgb.max() <= 1.0 and rgb.min() >= 0.0
    B, C, H1, W1 = rgb.shape
    assert H1 == W1, "Image must be square"

    # Resize logic matching _prepare_input
    if size == 224:
        # For size 224, resize first (though redundant for square input)
        target_size = round(size * max(W1 / H1, H1 / W1))
        if H1 != target_size:
            rgb = F.interpolate(
                rgb, (target_size, target_size), mode="bilinear", align_corners=False
            )
    else:
        # For other sizes, resize to target size
        if H1 != size:
            rgb = F.interpolate(rgb, (size, size), mode="bilinear", align_corners=False)

    # Crop logic matching _prepare_input
    _, _, H, W = rgb.shape
    cx, cy = W // 2, H // 2

    if size == 224:
        # For size 224, crop to square centered
        half = min(cx, cy)
        rgb = rgb[:, :, cy - half : cy + half, cx - half : cx + half]
    else:
        # For other sizes, crop with specific logic
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = int(3 * halfw / 4)
        rgb = rgb[:, :, cy - halfh : cy + halfh, cx - halfw : cx + halfw]

    # Normalize: (rgb - 0.5) / 0.5, matching transform in _prepare_input
    view = {
        "img": (rgb - 0.5) / 0.5,
        "ray_map": torch.full(
            (rgb.shape[0], 6, rgb.shape[2], rgb.shape[3]),
            torch.nan,
        ).to(rgb.device),
        "img_mask": torch.tensor(True).unsqueeze(0).to(rgb.device),
        "ray_mask": torch.tensor(False).unsqueeze(0).to(rgb.device),
        "update": torch.tensor(True).unsqueeze(0).to(rgb.device),
        "reset": torch.tensor(False).unsqueeze(0).to(rgb.device),
    }
    return view


def populate_queues(
    queues: dict[str, deque],
    batch: dict[str, torch.Tensor],
    exclude_keys: list[str] | None = None,
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues

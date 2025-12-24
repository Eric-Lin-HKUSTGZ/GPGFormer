import torch
import torch.nn.functional as F
from typing import Tuple


def scale_bbox_xyxy(bbox: torch.Tensor, scale: float, img_w: int, img_h: int) -> torch.Tensor:
    x1, y1, x2, y2 = bbox.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = (cx - w * 0.5).clamp(0, img_w - 1)
    ny1 = (cy - h * 0.5).clamp(0, img_h - 1)
    nx2 = (cx + w * 0.5).clamp(0, img_w - 1)
    ny2 = (cy + h * 0.5).clamp(0, img_h - 1)
    return torch.stack([nx1, ny1, nx2, ny2], dim=-1)


def crop_and_resize(img: torch.Tensor, bbox: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    _, _, H, W = img.shape
    x1, y1, x2, y2 = bbox.unbind(-1)
    x1 = x1.round().long().clamp(0, W - 1)
    x2 = x2.round().long().clamp(0, W - 1)
    y1 = y1.round().long().clamp(0, H - 1)
    y2 = y2.round().long().clamp(0, H - 1)

    crops = []
    for i in range(img.shape[0]):
        if x2[i] <= x1[i] or y2[i] <= y1[i]:
            crop = torch.zeros((img.shape[1], out_h, out_w), device=img.device, dtype=img.dtype)
        else:
            crop = img[i:i + 1, :, y1[i]:y2[i], x1[i]:x2[i]]
            crop = F.interpolate(crop, size=(out_h, out_w), mode="bilinear", align_corners=False)
            crop = crop[0]
        crops.append(crop)
    return torch.stack(crops, dim=0)


def crop_feature_and_resize(feat: torch.Tensor, bbox: torch.Tensor, in_h: int, in_w: int,
                            out_h: int, out_w: int) -> torch.Tensor:
    x1, y1, x2, y2 = bbox.unbind(-1)
    sx = feat.shape[-1] / float(in_w)
    sy = feat.shape[-2] / float(in_h)
    fbbox = torch.stack([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dim=-1)
    return crop_and_resize(feat, fbbox, out_h, out_w)

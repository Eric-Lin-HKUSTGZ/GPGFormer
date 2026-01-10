from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from third_party.moge_min.moge.model.v2 import MoGeModel


@dataclass(frozen=True)
class MoGe2Config:
    weights_path: str
    num_tokens: int = 1600


class MoGe2Prior(nn.Module):
    """
    Geometry prior using MoGe2 (Ruicheng/moge-2-vitl-normal).

    Requirements (framework.md):
    - use MoGe2 pretrained weights
    - use Conv neck output as target features
    - freeze MoGe2 parameters (no learning)
    """

    def __init__(self, cfg: MoGe2Config):
        super().__init__()
        self.cfg = cfg
        weights_path = Path(cfg.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"MoGe2 weights not found: {weights_path}")

        self.model = MoGeModel.from_pretrained(str(weights_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_01: (B,3,H,W) in [0,1], RGB
        Returns:
            target_features: (B,C,Hg,Wg) from MoGe2 conv neck last output ([-1])
        """
        # We need the Conv neck output (not the heads).
        # The vendored MoGe2 doesn't expose it directly, so we recompute the same neck-forward here.
        b, _, h, w = img_01.shape
        device, dtype = img_01.device, img_01.dtype

        # Copied from MoGeModel.forward (v2), up to the neck output.
        aspect_ratio = w / h
        base_h, base_w = (self.cfg.num_tokens / aspect_ratio) ** 0.5, (self.cfg.num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        features, cls_token = self.model.encoder(img_01, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        from third_party.moge_min.moge.utils.geometry_torch import normalized_view_plane_uv

        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        neck_out = self.model.neck(features)
        target = neck_out[-1]
        return target




from __future__ import annotations

import math

import torch
import torch.nn as nn


class GeoSideTuning(nn.Module):
    """
    Lightweight side-tuning adapter for geometry tokens.
    Input/Output: (B, N_geo, C_geo)
    """

    def __init__(
        self,
        geo_channels: int = 1280,
        side_channels: int = 256,
        dropout: float = 0.1,
        max_res_scale: float = 0.1,
        init_res_scale: float = 1e-3,
    ):
        super().__init__()
        self.side_net = nn.Sequential(
            nn.Linear(geo_channels, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, side_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_proj = nn.Linear(geo_channels + side_channels, geo_channels)
        # Bounded residual gate keeps geometric injection stable in late training.
        if max_res_scale <= 0:
            raise ValueError(f"max_res_scale must be > 0, got {max_res_scale}")
        self.max_res_scale = float(max_res_scale)
        init_ratio = float(init_res_scale) / self.max_res_scale
        init_ratio = min(max(init_ratio, 1e-6), 1.0 - 1e-6)
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.res_scale_raw = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.side_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Near-identity start with non-zero gradients for side branch.
        nn.init.normal_(self.fusion_proj.weight, std=1e-3)
        nn.init.zeros_(self.fusion_proj.bias)

    def forward(self, geo_tokens: torch.Tensor) -> torch.Tensor:
        side_feat = self.side_net(geo_tokens)
        fused = torch.cat([geo_tokens, side_feat], dim=-1)
        delta = self.fusion_proj(fused)
        res_scale = torch.sigmoid(self.res_scale_raw).to(dtype=geo_tokens.dtype) * self.max_res_scale
        return geo_tokens + res_scale * delta

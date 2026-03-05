from __future__ import annotations

import torch
import torch.nn as nn


class GeoSideTuning(nn.Module):
    """
    Lightweight side-tuning adapter for geometry tokens.
    Input/Output: (B, N_geo, C_geo)
    """

    def __init__(self, geo_channels: int = 1280, side_channels: int = 256, dropout: float = 0.1):
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
        # Learnable residual scale keeps start close to identity while allowing gradients.
        self.res_scale = nn.Parameter(torch.tensor(1e-3))
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
        return geo_tokens + self.res_scale * delta

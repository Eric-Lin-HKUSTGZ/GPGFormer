from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_group_count(channels: int, max_groups: int) -> int:
    groups = min(max(1, int(max_groups)), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class GeoSideAdapter(nn.Module):
    """
    HandOS-style map-level side adapter for MoGe2 neck features.
    Input:  (B, C, H, W)
    Output: (B, C + C_side, H, W) via channel concatenation.
    """

    def __init__(
        self,
        side_channels: int = 256,
        depth: int = 3,
        dropout: float = 0.05,
        norm_groups: int = 32,
    ):
        super().__init__()
        if side_channels <= 0:
            raise ValueError(f"side_channels must be > 0, got {side_channels}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.side_channels = int(side_channels)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.norm_groups = int(norm_groups)
        self.side_net: nn.Sequential | None = None

    def _make_block(self, in_channels: int, out_channels: int, kernel_size: int) -> list[nn.Module]:
        padding = kernel_size // 2
        groups = _resolve_group_count(out_channels, self.norm_groups)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        ]
        if self.dropout > 0.0:
            layers.append(nn.Dropout2d(self.dropout))
        return layers

    def _init_weights(self) -> None:
        if self.side_net is None:
            return
        for module in self.side_net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_if_needed(self, feat: torch.Tensor) -> None:
        if self.side_net is not None:
            return

        in_channels = int(feat.shape[1])
        layers: list[nn.Module] = []
        layers.extend(self._make_block(in_channels, self.side_channels, kernel_size=1))
        for _ in range(self.depth - 1):
            layers.extend(self._make_block(self.side_channels, self.side_channels, kernel_size=3))

        self.side_net = nn.Sequential(*layers).to(device=feat.device, dtype=feat.dtype)
        self._init_weights()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        self._build_if_needed(feat)
        assert self.side_net is not None
        side_feat = self.side_net(feat)
        return torch.cat([feat, side_feat], dim=1)

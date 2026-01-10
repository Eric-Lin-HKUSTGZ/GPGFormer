from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoTokenizer(nn.Module):
    """
    Tokenizer2: target feature map (B,C,H,W) -> tokens (B, N, D=1280)
    """

    def __init__(self, in_channels: int, embed_dim: int = 1280, out_hw: tuple[int, int] = (16, 12)):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_hw = out_hw
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.proj.bias)

        # Type embedding is applied in the encoder; keep tokenizer purely geometric.

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B,C,H,W)
        Returns:
            tokens: (B, N, D)
            coords: (H,W,2) normalized in [-1,1] for positional embedding (optional)
        """
        b, _, _, _ = feat.shape

        # IMPORTANT: pool FIRST to avoid massive intermediate activations.
        # MoGe2 neck outputs can be high-res; projecting to 1280 channels before pooling can OOM.
        oh, ow = self.out_hw
        feat_small = F.adaptive_avg_pool2d(feat, output_size=(oh, ow))  # (B,C,oh,ow)
        x = self.proj(feat_small)  # (B,D,oh,ow)

        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, oh*ow, D)

        # Normalized grid for optional positional encoding
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, oh, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, ow, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)  # (H,W,2)
        return tokens, coords


class CoordPosEmbed(nn.Module):
    """
    Lightweight positional embedding from normalized coords (x,y) -> (D).
    Works with variable token count.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Start near-zero so it doesn't disturb pretrained weights initially.
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, coords_hw2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_hw2: (H,W,2) in [-1,1]
        Returns:
            pos: (H*W, D)
        """
        h, w, _ = coords_hw2.shape
        pos = self.mlp(coords_hw2.reshape(h * w, 2))
        return pos



from __future__ import annotations

import torch


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Minimal MoGe2 dependency: create normalized UV grid for a view plane.

    Returns:
      uv: (H, W, 2) where top-left is negative and bottom-right is positive.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1.0 + aspect_ratio**2) ** 0.5
    span_y = 1.0 / (1.0 + aspect_ratio**2) ** 0.5

    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv



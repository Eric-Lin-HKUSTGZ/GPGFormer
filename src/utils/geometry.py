import torch


def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: torch.Tensor) -> torch.Tensor:
    points = points + translation[:, None, :]
    z = points[..., 2:3].clamp(min=1e-6)
    xy = points[..., :2] / z
    proj = xy * focal_length[:, None, :] + camera_center[:, None, :]
    return proj

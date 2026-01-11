from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from gpgformer.models.components.pose_transformer import TransformerDecoder
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat, rot6d_to_rotmat


def _rotmat_to_aa(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle.
    Args:
        R: (...,3,3)
    Returns:
        (...,3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)
    sin = torch.sin(angle)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)  # (...,3)

    denom = (2.0 * sin).unsqueeze(-1)
    axis = r / denom
    aa = axis * angle.unsqueeze(-1)

    # Small-angle approximation: aa ~= 0.5 * vee(R - R^T)
    small = (sin.abs() < 1e-4).unsqueeze(-1)
    aa_small = 0.5 * r
    return torch.where(small, aa_small, aa)


@dataclass(frozen=True)
class MANOHeadConfig:
    mean_params_path: str
    num_hand_joints: int = 15
    joint_rep: str = "aa"  # only 'aa' supported in this project

    ief_iters: int = 3
    transformer_input: str = "mean_shape"  # 'mean_shape' or 'zero'

    # Decoder transformer
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.0
    init_decoder_xavier: bool = False


class MANOTransformerDecoderHead(nn.Module):
    """
    HaMeR-style MANO head:
    - context: flattened spatial tokens from a backbone feature map
    - query: 1 token (zero or concat of current pose/shape/cam)
    - IEF: iteratively refine pose/shape/cam by predicting residuals.
    """

    def __init__(self, cfg: MANOHeadConfig, context_dim: int = 1280):
        super().__init__()
        self.cfg = cfg

        if cfg.joint_rep != "aa":
            raise ValueError(f"Only joint_rep='aa' is supported, got {cfg.joint_rep}")

        self.joint_rep_dim = 3
        self.npose = self.joint_rep_dim * (cfg.num_hand_joints + 1)  # 16 * 3 = 48
        self.input_is_mean_shape = cfg.transformer_input == "mean_shape"

        token_dim = (self.npose + 10 + 3) if self.input_is_mean_shape else 1
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=token_dim,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            context_dim=context_dim,
        )

        self.decpose = nn.Linear(cfg.dim, self.npose)
        self.decshape = nn.Linear(cfg.dim, 10)
        self.deccam = nn.Linear(cfg.dim, 3)

        if cfg.init_decoder_xavier:
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_path = Path(cfg.mean_params_path)
        if not mean_path.exists():
            raise FileNotFoundError(f"MANO mean params not found: {mean_path}")
        mean_params = np.load(str(mean_path))

        init_pose_np = mean_params["pose"].astype(np.float32).reshape(1, -1)
        init_betas_np = mean_params["shape"].astype(np.float32).reshape(1, -1)
        init_cam_np = mean_params["cam"].astype(np.float32).reshape(1, -1)

        # Make mean pose compatible with axis-angle if stored as 6D.
        expected_aa = 3 * (cfg.num_hand_joints + 1)
        expected_6d = 6 * (cfg.num_hand_joints + 1)
        if init_pose_np.size == expected_6d:
            with torch.no_grad():
                pose6d = torch.from_numpy(init_pose_np).view(-1, 6)
                R = rot6d_to_rotmat(pose6d)
                aa = _rotmat_to_aa(R).view(1, -1).cpu().numpy().astype(np.float32)
            init_pose_np = aa
        elif init_pose_np.size != expected_aa:
            raise ValueError(
                f"Unexpected mean pose size: {init_pose_np.size}. Expected {expected_aa} (aa) or {expected_6d} (6d)."
            )

        self.register_buffer("init_hand_pose", torch.from_numpy(init_pose_np))  # (1,48)
        self.register_buffer("init_betas", torch.from_numpy(init_betas_np))  # (1,10)
        self.register_buffer("init_cam", torch.from_numpy(init_cam_np))  # (1,3)

    def forward(self, feat: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) conditioning features (typically C=1280)
        Returns:
            pred_mano_params: dict with rotmats {global_orient(B,1,3,3), hand_pose(B,15,3,3), betas(B,10)}
            pred_cam: (B,3) weak-perspective camera params
        """
        if feat.ndim != 4:
            raise ValueError(f"feat must be (B,C,H,W), got {tuple(feat.shape)}")

        B, C, H, W = feat.shape
        # context tokens: (B, H*W, C)
        context = feat.flatten(2).transpose(1, 2).contiguous()

        pred_hand_pose = self.init_hand_pose.expand(B, -1)
        pred_betas = self.init_betas.expand(B, -1)
        pred_cam = self.init_cam.expand(B, -1)

        for _ in range(int(self.cfg.ief_iters)):
            if self.input_is_mean_shape:
                token = torch.cat([pred_hand_pose, pred_betas, pred_cam], dim=1).unsqueeze(1)  # (B,1,D)
            else:
                token = torch.zeros((B, 1, 1), device=feat.device, dtype=feat.dtype)

            token_out = self.transformer(token, context=context).squeeze(1)  # (B, dim)
            pred_hand_pose = pred_hand_pose + self.decpose(token_out)
            pred_betas = pred_betas + self.decshape(token_out)
            pred_cam = pred_cam + self.deccam(token_out)

        # axis-angle -> rotmats
        pose_rm = aa_to_rotmat(pred_hand_pose.view(-1, 3).contiguous()).view(B, self.cfg.num_hand_joints + 1, 3, 3)
        pred_mano_params = {
            "global_orient": pose_rm[:, [0]],
            "hand_pose": pose_rm[:, 1:],
            "betas": pred_betas,
        }
        return pred_mano_params, pred_cam







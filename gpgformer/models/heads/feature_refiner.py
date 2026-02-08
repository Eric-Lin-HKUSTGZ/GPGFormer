"""
Lightweight Feature Refinement Modules for GPGFormer.

Three refinement strategies designed to avoid OOM while improving performance:
1. SJTA (Sparse Joint Token Adapter) - Cross-attention based joint query readout
2. COEAR (Convolutional Edge/High-frequency Augmentation Refiner) - Depthwise conv enhancement
3. KCR (Keypoint Consistency Refinement) - 2D keypoint consistency driven self-correction

Reference: WiLoR特征精细化模块_核心理念抽象_轻量创新方案_20260208.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat


@dataclass(frozen=True)
class FeatureRefinerConfig:
    """Configuration for lightweight feature refinement modules."""
    # Refinement method: 'sjta', 'coear', 'kcr', 'none'
    method: str = "sjta"

    # Common settings
    feat_dim: int = 1280  # Input feature dimension from backbone
    num_joints: int = 16  # Number of MANO joints (1 global + 15 hand)
    joint_rep: str = "aa"  # 'aa' (axis-angle) or '6d'

    # SJTA specific settings
    sjta_bottleneck_dim: int = 256  # Bottleneck dimension for cross-attention
    sjta_num_heads: int = 4  # Number of attention heads
    sjta_use_2d_prior: bool = True  # Use projected 2D coordinates as prior

    # COEAR specific settings
    coear_dilation1: int = 1  # Dilation for first depthwise conv
    coear_dilation2: int = 2  # Dilation for second depthwise conv
    coear_gate_reduction: int = 8  # Channel reduction ratio for gate
    coear_init_alpha: float = 0.1  # Initial value for learnable alpha

    # KCR specific settings
    kcr_num_keypoints: int = 21  # Number of 2D keypoints to predict
    kcr_hidden_dim: int = 128  # Hidden dimension for correction MLP


class SJTA(nn.Module):
    """
    Sparse Joint Token Adapter (SJTA).

    Core idea: Treat patch tokens as a memory bank and joints as structured queries.
    Each joint reads relevant evidence from patch tokens via cross-attention.
    """

    def __init__(self, cfg: FeatureRefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.num_joints = cfg.num_joints
        self.bottleneck_dim = cfg.sjta_bottleneck_dim
        self.num_heads = cfg.sjta_num_heads
        self.head_dim = self.bottleneck_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[cfg.joint_rep]

        # Project patch tokens to bottleneck dimension (K, V)
        self.token_proj_k = nn.Linear(cfg.feat_dim, self.bottleneck_dim)
        self.token_proj_v = nn.Linear(cfg.feat_dim, self.bottleneck_dim)

        # Joint query construction
        joint_input_dim = 3 + (2 if cfg.sjta_use_2d_prior else 0)
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
        )

        # Learnable joint ID embeddings
        self.joint_id_embed = nn.Parameter(
            torch.randn(1, cfg.num_joints, self.bottleneck_dim) * 0.02
        )

        # Query projection and output
        self.q_proj = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.norm = nn.LayerNorm(self.bottleneck_dim)

        # Delta prediction heads
        npose = self.joint_rep_dim * cfg.num_joints
        self.dec_pose = nn.Linear(self.bottleneck_dim * cfg.num_joints, npose)
        self.dec_shape = nn.Linear(self.bottleneck_dim, 10)
        self.dec_cam = nn.Linear(self.bottleneck_dim, 3)
        self._init_output_heads()

    def _init_output_heads(self):
        for head in [self.dec_pose, self.dec_shape, self.dec_cam]:
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=0.01)

    def forward(
        self,
        img_feat: torch.Tensor,
        joints_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_mano_feats: Dict[str, torch.Tensor],
        focal_length: torch.Tensor,
        img_size: float = 256.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        B, C, H, W = img_feat.shape
        patch_tokens = img_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        N = patch_tokens.shape[1]

        K = self.token_proj_k(patch_tokens)
        V = self.token_proj_v(patch_tokens)

        # Construct joint queries
        if self.cfg.sjta_use_2d_prior:
            joints_2d = self._project_joints(joints_3d, pred_cam, focal_length, img_size)
            joints_2d_norm = joints_2d / (img_size / 2) - 1
            joint_input = torch.cat([joints_3d, joints_2d_norm], dim=-1)
        else:
            joint_input = joints_3d

        joint_feat = self.joint_encoder(joint_input) + self.joint_id_embed
        Q = self.q_proj(joint_feat)

        # Multi-head attention
        Q = Q.view(B, self.num_joints, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, self.num_joints, self.bottleneck_dim)
        out = self.norm(self.out_proj(out) + joint_feat)

        # Predict deltas
        delta_pose = self.dec_pose(out.flatten(1))
        delta_betas = self.dec_shape(out.mean(dim=1))
        delta_cam = self.dec_cam(out[:, 0])

        # Apply residual updates
        pred_hand_pose = pred_mano_feats['hand_pose'] + delta_pose
        pred_betas = pred_mano_feats['betas'] + delta_betas
        pred_cam_out = pred_mano_feats['cam'] + delta_cam

        return self._convert_to_rotmat(pred_hand_pose, pred_betas, B), pred_cam_out

    def _project_joints(self, joints_3d, pred_cam, focal_length, img_size):
        scale, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        tz = 2.0 * focal_length[:, 0] / (img_size * scale + 1e-9)
        cam_t = torch.stack([tx, ty, tz], dim=-1)
        joints_cam = joints_3d + cam_t.unsqueeze(1)
        z = joints_cam[..., 2:3].clamp(min=1e-6)
        xy = joints_cam[..., :2] / z
        return xy * focal_length.unsqueeze(1) + img_size / 2

    def _convert_to_rotmat(self, pred_hand_pose, pred_betas, B):
        if self.cfg.joint_rep == '6d':
            from third_party.wilor_min.wilor.utils.geometry import rot6d_to_rotmat
            rm = rot6d_to_rotmat(pred_hand_pose.view(-1, 6)).view(B, self.num_joints, 3, 3)
        else:
            rm = aa_to_rotmat(pred_hand_pose.view(-1, 3)).view(B, self.num_joints, 3, 3)
        return {'global_orient': rm[:, [0]], 'hand_pose': rm[:, 1:], 'betas': pred_betas}


class COEAR(nn.Module):
    """
    Convolutional Edge/High-frequency Augmentation Refiner (COEAR).
    Enhances ViT features with CNN-style local inductive bias.
    """

    def __init__(self, cfg: FeatureRefinerConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.feat_dim

        # Depthwise convolutions with different dilations
        self.dw_conv1 = nn.Conv2d(C, C, 3, padding=cfg.coear_dilation1,
                                   dilation=cfg.coear_dilation1, groups=C, bias=False)
        self.dw_conv2 = nn.Conv2d(C, C, 3, padding=cfg.coear_dilation2,
                                   dilation=cfg.coear_dilation2, groups=C, bias=False)

        # Channel gate
        r = cfg.coear_gate_reduction
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, C // r), nn.ReLU(inplace=True),
            nn.Linear(C // r, C), nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(cfg.coear_init_alpha))

        nn.init.kaiming_normal_(self.dw_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        dw1 = self.dw_conv1(img_feat)
        dw2 = self.dw_conv2(img_feat)
        dw_combined = dw1 + dw2
        gate = self.gate(img_feat).unsqueeze(-1).unsqueeze(-1)
        return img_feat + self.alpha * gate * dw_combined


class KCR(nn.Module):
    """
    Keypoint Consistency Refinement (KCR).
    Uses 2D keypoint evidence to provide correction signals.
    """

    def __init__(self, cfg: FeatureRefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.num_keypoints = cfg.kcr_num_keypoints
        self.num_joints = cfg.num_joints
        self.hidden_dim = cfg.kcr_hidden_dim
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[cfg.joint_rep]

        self.heatmap_conv = nn.Conv2d(cfg.feat_dim, self.num_keypoints, kernel_size=1)

        correction_input_dim = self.num_keypoints * 2
        npose = self.joint_rep_dim * cfg.num_joints

        self.correction_mlp = nn.Sequential(
            nn.Linear(correction_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.dec_pose = nn.Linear(self.hidden_dim, npose)
        self.dec_shape = nn.Linear(self.hidden_dim, 10)
        self.dec_cam = nn.Linear(self.hidden_dim, 3)
        self._init_output_heads()

    def _init_output_heads(self):
        for head in [self.dec_pose, self.dec_shape, self.dec_cam]:
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=0.01)

    def _soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, K, -1)
        heatmaps_softmax = F.softmax(heatmaps_flat, dim=-1).view(B, K, H, W)
        device, dtype = heatmaps.device, heatmaps.dtype
        y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
        x = (heatmaps_softmax * x_coords).sum(dim=(2, 3))
        y = (heatmaps_softmax * y_coords).sum(dim=(2, 3))
        return torch.stack([x, y], dim=-1)

    def _project_joints_to_kp(self, joints_3d, pred_cam, focal_length, img_size, feat_size):
        B = joints_3d.shape[0]
        H, W = feat_size
        scale, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        tz = 2.0 * focal_length[:, 0] / (img_size * scale + 1e-9)
        cam_t = torch.stack([tx, ty, tz], dim=-1)
        joints_cam = joints_3d + cam_t.unsqueeze(1)
        z = joints_cam[..., 2:3].clamp(min=1e-6)
        xy = joints_cam[..., :2] / z
        joints_2d = xy * focal_length.unsqueeze(1) + img_size / 2
        joints_2d_feat = joints_2d.clone()
        joints_2d_feat[..., 0] = joints_2d[..., 0] * W / img_size
        joints_2d_feat[..., 1] = joints_2d[..., 1] * H / img_size

        # Pad or truncate to match num_keypoints
        if joints_2d_feat.shape[1] < self.num_keypoints:
            pad = torch.zeros(B, self.num_keypoints - joints_2d_feat.shape[1], 2,
                            device=joints_2d_feat.device, dtype=joints_2d_feat.dtype)
            joints_2d_feat = torch.cat([joints_2d_feat, pad], dim=1)
        elif joints_2d_feat.shape[1] > self.num_keypoints:
            joints_2d_feat = joints_2d_feat[:, :self.num_keypoints]
        return joints_2d_feat

    def forward(
        self,
        img_feat: torch.Tensor,
        joints_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_mano_feats: Dict[str, torch.Tensor],
        focal_length: torch.Tensor,
        img_size: float = 256.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        B, C, H, W = img_feat.shape

        heatmaps = self.heatmap_conv(img_feat)
        kp2d_feat = self._soft_argmax(heatmaps)
        kp2d_proj = self._project_joints_to_kp(joints_3d, pred_cam, focal_length, img_size, (H, W))

        discrepancy = (kp2d_feat - kp2d_proj) / max(H, W)
        correction_feat = self.correction_mlp(discrepancy.flatten(1))

        delta_pose = self.dec_pose(correction_feat)
        delta_betas = self.dec_shape(correction_feat)
        delta_cam = self.dec_cam(correction_feat)

        pred_hand_pose = pred_mano_feats['hand_pose'] + delta_pose
        pred_betas = pred_mano_feats['betas'] + delta_betas
        pred_cam_out = pred_mano_feats['cam'] + delta_cam

        return self._convert_to_rotmat(pred_hand_pose, pred_betas, B), pred_cam_out

    def _convert_to_rotmat(self, pred_hand_pose, pred_betas, B):
        if self.cfg.joint_rep == '6d':
            from third_party.wilor_min.wilor.utils.geometry import rot6d_to_rotmat
            rm = rot6d_to_rotmat(pred_hand_pose.view(-1, 6)).view(B, self.num_joints, 3, 3)
        else:
            rm = aa_to_rotmat(pred_hand_pose.view(-1, 3)).view(B, self.num_joints, 3, 3)
        return {'global_orient': rm[:, [0]], 'hand_pose': rm[:, 1:], 'betas': pred_betas}


# =============================================================================
# Wrapper and Factory
# =============================================================================

class FeatureRefinerWrapper(nn.Module):
    """Unified wrapper for all feature refinement methods."""

    def __init__(self, cfg: FeatureRefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.method = cfg.method.lower()

        if self.method == 'sjta':
            self.refiner = SJTA(cfg)
            self.is_feature_enhancer = False
        elif self.method == 'coear':
            self.refiner = COEAR(cfg)
            self.is_feature_enhancer = True
        elif self.method == 'kcr':
            self.refiner = KCR(cfg)
            self.is_feature_enhancer = False
        elif self.method == 'none':
            self.refiner = None
            self.is_feature_enhancer = False
        else:
            raise ValueError(f"Unknown method: {cfg.method}")

    def enhance_features(self, img_feat: torch.Tensor) -> torch.Tensor:
        """Enhance features (only for COEAR method)."""
        if self.method == 'coear' and self.refiner is not None:
            return self.refiner(img_feat)
        return img_feat

    def refine_params(
        self,
        img_feat: torch.Tensor,
        joints_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_mano_feats: Dict[str, torch.Tensor],
        focal_length: torch.Tensor,
        img_size: float = 256.0,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Refine MANO parameters (for SJTA and KCR methods)."""
        if self.method in ['sjta', 'kcr'] and self.refiner is not None:
            return self.refiner(
                img_feat, joints_3d, pred_cam,
                pred_mano_feats, focal_length, img_size
            )
        return None, pred_cam


def build_feature_refiner(cfg: FeatureRefinerConfig) -> Optional[FeatureRefinerWrapper]:
    """Factory function to build feature refiner."""
    if cfg.method.lower() == 'none':
        return None
    return FeatureRefinerWrapper(cfg)

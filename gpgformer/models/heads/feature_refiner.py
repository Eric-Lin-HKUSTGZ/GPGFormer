"""
Lightweight Feature Refinement Modules for GPGFormer.

Four refinement strategies designed to avoid OOM while improving performance:
1. SJTA (Sparse Joint Token Adapter) - Cross-attention based joint query readout
2. COEAR (Convolutional Edge/High-frequency Augmentation Refiner) - Depthwise conv enhancement
3. WiLoR-MSF (WiLoR-style Multi-Scale Feature Refiner) - Multi-branch local/global fusion
4. KCR (Keypoint Consistency Refinement) - 2D keypoint consistency driven self-correction

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
    # Refinement method: 'sjta', 'coear', 'wilor_msf', 'kcr', 'none'
    method: str = "sjta"

    # Common settings
    feat_dim: int = 1280  # Input feature dimension from backbone
    num_joints: int = 16  # Number of MANO joints (1 global + 15 hand)
    joint_rep: str = "aa"  # 'aa' (axis-angle) or '6d'

    # SJTA specific settings
    sjta_bottleneck_dim: int = 256  # Bottleneck dimension for cross-attention
    sjta_num_heads: int = 4  # Number of attention heads
    sjta_use_2d_prior: bool = True  # Use projected 2D coordinates as prior
    # SJTA-v3 settings
    # - Use 21 keypoint queries (16 MANO joints + 5 fingertips) to strengthen fingertip evidence.
    sjta_num_queries: int = 21
    # - Geometric proximity bias for cross-attn (soft routing towards nearby tokens).
    sjta_geo_bias: bool = True
    sjta_geo_bias_beta: float = 1.0
    sjta_geo_bias_sigma: float = 0.6
    # - IEF-style iterative refinement steps (shared SJTA weights).
    #   Implemented in the caller since it requires MANO to update joints/priors.
    sjta_num_steps: int = 2

    # COEAR specific settings
    coear_dilation1: int = 1  # Dilation for first depthwise conv
    coear_dilation2: int = 2  # Dilation for second depthwise conv
    coear_gate_reduction: int = 8  # Channel reduction ratio for gate
    coear_init_alpha: float = 0.1  # Initial value for learnable alpha
    # WiLoR-MSF settings
    wilor_msf_bottleneck_ratio: int = 4  # Channel bottleneck ratio before multi-scale branches
    wilor_msf_dilation1: int = 1  # Dilation for branch-1 depthwise conv
    wilor_msf_dilation2: int = 2  # Dilation for branch-2 depthwise conv
    wilor_msf_dilation3: int = 3  # Dilation for branch-3 depthwise conv
    wilor_msf_gate_reduction: int = 8  # Channel reduction ratio for gate
    wilor_msf_init_alpha: float = 0.1  # Initial value for learnable alpha

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
        self.num_mano_joints = int(cfg.num_joints)
        self.num_queries = int(getattr(cfg, "sjta_num_queries", cfg.num_joints))
        self.bottleneck_dim = cfg.sjta_bottleneck_dim
        self.num_heads = cfg.sjta_num_heads
        self.head_dim = self.bottleneck_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[cfg.joint_rep]
        self.geo_bias_enabled = bool(getattr(cfg, "sjta_geo_bias", False)) and bool(cfg.sjta_use_2d_prior)
        self.geo_bias_beta = float(getattr(cfg, "sjta_geo_bias_beta", 1.0))
        self.geo_bias_sigma = float(getattr(cfg, "sjta_geo_bias_sigma", 0.6))
        self._token_xy_cache: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor] = {}

        # Normalize backbone token features to stabilize attention/readout.
        self.token_ln = nn.LayerNorm(cfg.feat_dim)

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
            torch.randn(1, self.num_queries, self.bottleneck_dim) * 0.02
        )

        # Query projection and output
        self.q_proj = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.norm = nn.LayerNorm(self.bottleneck_dim)

        # Delta prediction heads
        # IMPORTANT:
        # For stability, we predict a SMALL axis-angle delta per joint (3 dims each) and apply it
        # in rotation-matrix space: R_new = Exp(delta) @ R_base.
        # This avoids the numerically fragile rotmat->axis-angle inverse in the training graph.
        self.dec_pose = nn.Linear(self.bottleneck_dim * self.num_queries, 3 * self.num_mano_joints)
        self.dec_shape = nn.Linear(self.bottleneck_dim, 10)
        self.dec_cam = nn.Linear(self.bottleneck_dim, 3)
        # Learnable (log) scales to keep residual updates tiny at init, but allow growth later.
        self.pose_log_scale = nn.Parameter(torch.tensor(-4.0))
        self.betas_log_scale = nn.Parameter(torch.tensor(-3.0))
        self.cam_log_scale = nn.Parameter(torch.tensor(-4.0))
        self._init_output_heads()

    def _init_output_heads(self):
        # Tiny init to ensure the refiner starts as (almost) identity and does not destroy the
        # pretrained/learned MANO head predictions in early epochs.
        for head in [self.dec_pose, self.dec_shape, self.dec_cam]:
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=1e-4)

    def _token_xy_norm(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Token-grid coordinates in [-1, 1], aligned with `joints_2d_norm`."""
        key = (int(H), int(W), str(device), dtype)
        cached = self._token_xy_cache.get(key, None)
        if cached is not None:
            return cached

        y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / float(H) * 2.0 - 1.0
        x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / float(W) * 2.0 - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (N,2), N=H*W
        self._token_xy_cache[key] = xy
        return xy

    def forward(
        self,
        img_feat: torch.Tensor,
        joints_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_mano_feats: Dict[str, torch.Tensor],
        focal_length: torch.Tensor,
        img_size: float = 256.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if joints_3d.ndim != 3 or joints_3d.shape[-1] != 3:
            raise ValueError(f"joints_3d must be (B,Q,3), got {tuple(joints_3d.shape)}")
        if joints_3d.shape[1] != self.num_queries:
            raise ValueError(
                f"SJTA expects joints_3d to have Q={self.num_queries} query points, got {joints_3d.shape[1]}. "
                f"With sjta_num_queries=21, pass 16 MANO joints + 5 fingertips as queries."
            )

        B, C, H, W = img_feat.shape
        patch_tokens = img_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        N = patch_tokens.shape[1]

        patch_tokens = self.token_ln(patch_tokens)
        K = self.token_proj_k(patch_tokens)
        V = self.token_proj_v(patch_tokens)

        # Construct joint queries
        joints_2d_norm = None
        if self.cfg.sjta_use_2d_prior:
            # NOTE: backbone feature map is token-grid (Hp, Wp) = (16, 12) for (256, 192) patch.
            # We infer patch width from aspect ratio to properly normalize x/y separately.
            patch_h = float(img_size)
            patch_w = float(img_size) * (float(W) / float(H)) if H > 0 else float(img_size)
            joints_2d = self._project_joints(joints_3d, pred_cam, focal_length, patch_h, patch_w)
            joints_2d_norm = torch.stack(
                [
                    joints_2d[..., 0] / (patch_w * 0.5) - 1.0,
                    joints_2d[..., 1] / (patch_h * 0.5) - 1.0,
                ],
                dim=-1,
            )
            joint_input = torch.cat([joints_3d, joints_2d_norm], dim=-1)
        else:
            joint_input = joints_3d

        joint_feat = self.joint_encoder(joint_input) + self.joint_id_embed
        Q = self.q_proj(joint_feat)

        # Multi-head attention
        Q = Q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,h,Q,N)
        if self.geo_bias_enabled:
            if joints_2d_norm is None:
                raise RuntimeError("sjta_geo_bias requires sjta_use_2d_prior=True.")
            token_xy = self._token_xy_norm(H, W, device=img_feat.device, dtype=img_feat.dtype)  # (N,2)
            diff = joints_2d_norm.unsqueeze(2) - token_xy.view(1, 1, N, 2)  # (B,Q,N,2)
            dist2 = (diff * diff).sum(dim=-1)  # (B,Q,N)
            sigma2 = float(max(self.geo_bias_sigma, 1e-6)) ** 2
            # Add a log-space Gaussian proximity bias so softmax acts like soft routing around the
            # projected query point (instead of hard grid_sample).
            geo_bias = -dist2 / (2.0 * sigma2)  # (B,Q,N)
            attn_logits = attn_logits + (self.geo_bias_beta * geo_bias).unsqueeze(1)

        attn = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, self.num_queries, self.bottleneck_dim)
        out = self.norm(self.out_proj(out) + joint_feat)

        # Predict deltas
        delta_pose = self.dec_pose(out.flatten(1)) * torch.exp(self.pose_log_scale)
        delta_betas = self.dec_shape(out.mean(dim=1)) * torch.exp(self.betas_log_scale)
        delta_cam = self.dec_cam(out[:, 0]) * torch.exp(self.cam_log_scale)

        # Apply residual updates
        if "pose_rotmat" not in pred_mano_feats:
            raise KeyError("SJTA expects pred_mano_feats['pose_rotmat'] with shape (B,16,3,3).")
        pose_rm = pred_mano_feats["pose_rotmat"]
        if pose_rm.ndim != 4 or pose_rm.shape[-2:] != (3, 3):
            raise ValueError(f"pose_rotmat must be (B,J,3,3), got {tuple(pose_rm.shape)}")
        if pose_rm.shape[1] != self.num_mano_joints:
            raise ValueError(f"pose_rotmat J mismatch: {pose_rm.shape[1]} != {self.num_mano_joints}")

        delta_aa = delta_pose.view(B, self.num_mano_joints, 3)
        delta_R = aa_to_rotmat(delta_aa.reshape(-1, 3)).view(B, self.num_mano_joints, 3, 3)
        pose_new = torch.matmul(delta_R, pose_rm)

        pred_betas = pred_mano_feats["betas"] + delta_betas
        pred_cam_out = pred_mano_feats["cam"] + delta_cam

        refined = {
            "global_orient": pose_new[:, [0]],
            "hand_pose": pose_new[:, 1:],
            "betas": pred_betas,
        }
        return refined, pred_cam_out

    def _project_joints(self, joints_3d, pred_cam, focal_length, patch_h: float, patch_w: float):
        scale, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        # Keep consistent with the main model's weak-perspective -> translation conversion,
        # which uses `image_size` (height, 256) in the denominator.
        tz = 2.0 * focal_length[:, 0] / (patch_h * scale + 1e-9)
        cam_t = torch.stack([tx, ty, tz], dim=-1)
        joints_cam = joints_3d + cam_t.unsqueeze(1)
        z = joints_cam[..., 2:3].clamp(min=1e-6)
        xy = joints_cam[..., :2] / z
        center = joints_cam.new_tensor([patch_w * 0.5, patch_h * 0.5]).view(1, 1, 2)
        return xy * focal_length.unsqueeze(1) + center


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


class WiLoRMSF(nn.Module):
    """
    WiLoR-style Multi-Scale Feature Refiner (MSF).

    Lightweight multi-branch feature enhancement:
    - Channel bottleneck projection
    - Three depthwise branches with different receptive fields (dilation 1/2/3)
    - One pooled context branch
    - Channel gate + learnable residual scale
    """

    def __init__(self, cfg: FeatureRefinerConfig):
        super().__init__()
        self.cfg = cfg
        C = int(cfg.feat_dim)

        bottleneck_ratio = max(1, int(cfg.wilor_msf_bottleneck_ratio))
        C_mid = max(32, C // bottleneck_ratio)

        self.pre = nn.Sequential(
            nn.Conv2d(C, C_mid, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.dw_conv1 = nn.Conv2d(
            C_mid,
            C_mid,
            kernel_size=3,
            padding=int(cfg.wilor_msf_dilation1),
            dilation=int(cfg.wilor_msf_dilation1),
            groups=C_mid,
            bias=False,
        )
        self.dw_conv2 = nn.Conv2d(
            C_mid,
            C_mid,
            kernel_size=3,
            padding=int(cfg.wilor_msf_dilation2),
            dilation=int(cfg.wilor_msf_dilation2),
            groups=C_mid,
            bias=False,
        )
        self.dw_conv3 = nn.Conv2d(
            C_mid,
            C_mid,
            kernel_size=3,
            padding=int(cfg.wilor_msf_dilation3),
            dilation=int(cfg.wilor_msf_dilation3),
            groups=C_mid,
            bias=False,
        )
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(C_mid, C_mid, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(C_mid * 4, C, kernel_size=1, bias=False),
            nn.GELU(),
        )

        gate_reduction = max(1, int(cfg.wilor_msf_gate_reduction))
        gate_hidden = max(1, C // gate_reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, C),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(cfg.wilor_msf_init_alpha)))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        x = self.pre(img_feat)
        b1 = self.dw_conv1(x)
        b2 = self.dw_conv2(x)
        b3 = self.dw_conv3(x)
        b4 = self.context_branch(x)
        ms = torch.cat([b1, b2, b3, b4], dim=1)
        delta = self.fuse(ms)
        gate = self.gate(img_feat).unsqueeze(-1).unsqueeze(-1)
        return img_feat + self.alpha * gate * delta


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
        # Keep residual updates tiny at init; prevents camera/pose blow-ups early in training.
        self.pose_log_scale = nn.Parameter(torch.tensor(-4.0))
        self.betas_log_scale = nn.Parameter(torch.tensor(-3.0))
        self.cam_log_scale = nn.Parameter(torch.tensor(-4.0))
        self._init_output_heads()

    def _init_output_heads(self):
        for head in [self.dec_pose, self.dec_shape, self.dec_cam]:
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=1e-4)

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
        # Feature maps typically follow the input aspect ratio (e.g., 16x12 for 256x192).
        # Use that ratio to avoid a square-image assumption in projection.
        patch_h = float(img_size)
        patch_w = float(img_size) * (float(W) / float(H)) if H > 0 else float(img_size)
        scale, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        tz = 2.0 * focal_length[:, 0] / (patch_h * scale + 1e-9)
        cam_t = torch.stack([tx, ty, tz], dim=-1)
        joints_cam = joints_3d + cam_t.unsqueeze(1)
        z = joints_cam[..., 2:3].clamp(min=1e-6)
        xy = joints_cam[..., :2] / z
        center = joints_cam.new_tensor([patch_w * 0.5, patch_h * 0.5]).view(1, 1, 2)
        joints_2d = xy * focal_length.unsqueeze(1) + center
        joints_2d_feat = torch.stack(
            [
                joints_2d[..., 0] * float(W) / float(patch_w),
                joints_2d[..., 1] * float(H) / float(patch_h),
            ],
            dim=-1,
        )

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
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = img_feat.shape

        heatmaps = self.heatmap_conv(img_feat)
        kp2d_feat = self._soft_argmax(heatmaps)
        kp2d_proj = self._project_joints_to_kp(joints_3d, pred_cam, focal_length, img_size, (H, W))

        discrepancy = (kp2d_feat - kp2d_proj) / max(H, W)
        correction_feat = self.correction_mlp(discrepancy.flatten(1))

        # Bound residual updates for stability.
        delta_pose = torch.tanh(self.dec_pose(correction_feat)) * torch.exp(self.pose_log_scale)
        delta_betas = torch.tanh(self.dec_shape(correction_feat)) * torch.exp(self.betas_log_scale)
        delta_cam = torch.tanh(self.dec_cam(correction_feat)) * torch.exp(self.cam_log_scale)

        if self.cfg.joint_rep == "aa":
            # Prevent occasional outliers from producing huge axis-angle steps.
            delta_aa = delta_pose.view(B, self.num_joints, 3)
            max_angle = 0.20  # rad (strong constraint; KCR should be a small-step refiner)
            ang = delta_aa.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            delta_aa = delta_aa * (max_angle / ang).clamp(max=1.0)
            delta_pose = delta_aa.view(B, -1)

        pred_hand_pose = pred_mano_feats['hand_pose'] + delta_pose
        pred_betas = pred_mano_feats['betas'] + delta_betas
        pred_cam_out = pred_mano_feats['cam'] + delta_cam
        # Avoid division blow-ups in weak-perspective -> translation conversion downstream.
        pred_cam_out = torch.cat(
            [pred_cam_out[:, 0:1].clamp(min=1e-4), pred_cam_out[:, 1:]],
            dim=1,
        )

        # Extra outputs for lightweight supervision/debug:
        # Normalize to [-0.5, 0.5] to match the dataloader / train.py convention.
        kp2d_feat_norm = torch.stack(
            [
                kp2d_feat[..., 0] / float(W) - 0.5,
                kp2d_feat[..., 1] / float(H) - 0.5,
            ],
            dim=-1,
        )

        aux = {
            "kcr_keypoints_2d": kp2d_feat_norm,
        }
        return self._convert_to_rotmat(pred_hand_pose, pred_betas, B), pred_cam_out, aux

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
        self.last_aux: Optional[Dict[str, torch.Tensor]] = None

        if self.method == 'sjta':
            self.refiner = SJTA(cfg)
            self.is_feature_enhancer = False
        elif self.method == 'coear':
            self.refiner = COEAR(cfg)
            self.is_feature_enhancer = True
        elif self.method == 'wilor_msf':
            self.refiner = WiLoRMSF(cfg)
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
        """Enhance features (for COEAR and WiLoR-MSF methods)."""
        if self.method in ['coear', 'wilor_msf'] and self.refiner is not None:
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
            if self.method == "kcr":
                refined, pred_cam_out, aux = self.refiner(
                    img_feat, joints_3d, pred_cam, pred_mano_feats, focal_length, img_size
                )
                self.last_aux = aux
                return refined, pred_cam_out
            self.last_aux = None
            return self.refiner(img_feat, joints_3d, pred_cam, pred_mano_feats, focal_length, img_size)
        self.last_aux = None
        return None, pred_cam


def build_feature_refiner(cfg: FeatureRefinerConfig) -> Optional[FeatureRefinerWrapper]:
    """Factory function to build feature refiner."""
    if cfg.method.lower() == 'none':
        return None
    return FeatureRefinerWrapper(cfg)

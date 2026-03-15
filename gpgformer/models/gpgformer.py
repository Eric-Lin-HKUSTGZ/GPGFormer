from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx.lbs import vertices2joints

from gpgformer.models.encoders.wilor_vit import WiLoRViTConfig, WiLoRViTWithGeo
from gpgformer.models.encoders.vitpose_vit import ViTPoseViTConfig, ViTPoseViTWithGeo
from gpgformer.models.heads.mano_head import MANOHeadConfig, MANOTransformerDecoderHead
from gpgformer.models.heads.feature_refiner import FeatureRefinerConfig, build_feature_refiner
from gpgformer.models.mano.mano_layer import MANOConfig, MANOLayer
from gpgformer.models.priors.moge2 import MoGe2Config, MoGe2Prior
from gpgformer.models.tokenizers.geo_tokenizer import GeoTokenizer, CoordPosEmbed
from gpgformer.models.tokenizers.geo_side_tuning import GeoSideTuning


@dataclass(frozen=True)
class GPGFormerConfig:
    # Weights
    backbone_type: str = "wilor"  # "wilor" | "vitpose"
    wilor_ckpt_path: str = ""
    vitpose_ckpt_path: str = ""
    mano_model_path: str = ""
    mano_mean_params: str = ""
    # Geometry prior (MoGe2)
    # - When use_geo_prior=False, moge2_weights_path may be None/empty and MoGe2 will not be loaded/run.
    moge2_weights_path: Optional[str] = None
    use_geo_prior: bool = True
    # MANO decode behavior
    mano_decoder: str = "wilor"  # "wilor" | "freihand_legacy"
    freihand_mano_root: Optional[str] = None

    # Camera / image settings
    image_size: int = 256
    image_hw: tuple[int, int] = (256, 192)
    focal_length: float = 5000.0 # 没使用

    # MoGe2
    moge2_num_tokens: int = 1600
    # MoGe2 output mode for geometry tokens
    # - "neck": use conv neck last feature map (original)
    # - "points": use final point map (xyz*mask + mask) as geometric prior tokens
    moge2_output: str = "neck"

    # Token fusion mode
    token_fusion_mode: str = "concat"  # "concat" | "sum" | "cross_attn"
    sum_fusion_strategy: str = "basic"  # "basic" | "weighted" | "normalized" | "weighted_normalized" | "channel_concat"
    # Learnable scalar gate for geometry injection in sum mode; effective gate is sigmoid(param).
    sum_geo_gate_init: float = 4.0
    fusion_proj_zero_init: bool = True
    # Cross-attention fusion settings (only used when token_fusion_mode="cross_attn")
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.0
    cross_attn_gate_init: float = 0.0
    # GeoTokenizer pooling: set to False to keep full resolution for cross_attn mode
    geo_tokenizer_use_pooling: bool = True
    # Optional side-tuning adapter on geo tokens (applied after GeoTokenizer).
    use_geo_side_tuning: bool = False
    geo_side_tuning_side_channels: int = 256
    geo_side_tuning_dropout: float = 0.1
    geo_side_tuning_max_res_scale: float = 0.1
    geo_side_tuning_init_res_scale: float = 1e-3
    # Modality dropout for geometry tokens during training (0 disables).
    geo_branch_dropout_prob: float = 0.0

    # HaMeR-style MANO head (defaults are reasonable if not provided by YAML)
    mano_head_ief_iters: int = 3
    mano_head_transformer_input: str = "mean_shape"  # 'mean_shape' or 'zero'
    mano_head_dim: int = 1024
    mano_head_depth: int = 6
    mano_head_heads: int = 8
    mano_head_dim_head: int = 64
    mano_head_mlp_dim: int = 2048
    mano_head_dropout: float = 0.0

    # Feature Refiner configuration (lightweight refinement module)
    # Methods: 'sjta', 'coear', 'wilor_msf', 'kcr', 'none'
    feature_refiner_method: str = "none"
    feature_refiner_feat_dim: int = 1280
    # SJTA settings
    feature_refiner_sjta_bottleneck_dim: int = 256
    feature_refiner_sjta_num_heads: int = 4
    feature_refiner_sjta_use_2d_prior: bool = True
    feature_refiner_sjta_num_steps: int = 2
    # COEAR settings
    feature_refiner_coear_dilation1: int = 1
    feature_refiner_coear_dilation2: int = 2
    feature_refiner_coear_gate_reduction: int = 8
    feature_refiner_coear_init_alpha: float = 0.1
    # WiLoR-MSF settings
    feature_refiner_wilor_msf_bottleneck_ratio: int = 4
    feature_refiner_wilor_msf_dilation1: int = 1
    feature_refiner_wilor_msf_dilation2: int = 2
    feature_refiner_wilor_msf_dilation3: int = 3
    feature_refiner_wilor_msf_gate_reduction: int = 8
    feature_refiner_wilor_msf_init_alpha: float = 0.1
    # KCR settings
    feature_refiner_kcr_num_keypoints: int = 21
    feature_refiner_kcr_hidden_dim: int = 128


class GPGFormer(nn.Module):
    """
    GPGFormer: Geometry-Prior Guided Transformer for Hand Reconstruction
    - tokenizer1 + transformer: WiLoR ViT-L
    - geometry prior: MoGe2 (frozen), using conv neck output
    - tokenizer2: conv neck features -> geo tokens
    - outputs: MANO params + camera translation, plus vertices/joints via MANO layer
    """

    def __init__(self, cfg: GPGFormerConfig):
        super().__init__()
        self.cfg = cfg

        self.use_geo_prior = bool(getattr(cfg, "use_geo_prior", True))

        self.moge2: MoGe2Prior | None = None
        if self.use_geo_prior:
            if not cfg.moge2_weights_path:
                raise ValueError("use_geo_prior=True requires moge2_weights_path to be set.")
            self.moge2 = MoGe2Prior(
                MoGe2Config(
                    weights_path=str(cfg.moge2_weights_path),
                    num_tokens=cfg.moge2_num_tokens,
                    output=str(getattr(cfg, "moge2_output", "neck")),
                )
            )

        # Geo tokenizer: in_channels depends on MoGe2 neck config; infer lazily at first forward.
        self.geo_tokenizer: GeoTokenizer | None = None
        self.geo_pos: CoordPosEmbed | None = CoordPosEmbed(embed_dim=1280) if self.use_geo_prior else None
        self.geo_side_tuning: GeoSideTuning | None = None
        if self.use_geo_prior and bool(getattr(cfg, "use_geo_side_tuning", False)):
            self.geo_side_tuning = GeoSideTuning(
                geo_channels=1280,
                side_channels=int(getattr(cfg, "geo_side_tuning_side_channels", 256)),
                dropout=float(getattr(cfg, "geo_side_tuning_dropout", 0.1)),
                max_res_scale=float(getattr(cfg, "geo_side_tuning_max_res_scale", 0.1)),
                init_res_scale=float(getattr(cfg, "geo_side_tuning_init_res_scale", 1e-3)),
            )

        backbone_type = getattr(cfg, 'backbone_type', 'wilor').lower()
        if backbone_type == "vitpose":
            self.encoder = ViTPoseViTWithGeo(
                ViTPoseViTConfig(
                    vitpose_ckpt_path=cfg.vitpose_ckpt_path,
                    mano_mean_params=cfg.mano_mean_params,
                    image_size=cfg.image_size,
                    joint_rep="aa",
                    token_fusion_mode=cfg.token_fusion_mode,
                    sum_fusion_strategy=getattr(cfg, 'sum_fusion_strategy', 'basic'),
                    sum_geo_gate_init=float(getattr(cfg, "sum_geo_gate_init", 4.0)),
                    fusion_proj_zero_init=bool(getattr(cfg, "fusion_proj_zero_init", True)),
                    cross_attn_num_heads=getattr(cfg, 'cross_attn_num_heads', 8),
                    cross_attn_dropout=getattr(cfg, 'cross_attn_dropout', 0.0),
                    cross_attn_gate_init=getattr(cfg, 'cross_attn_gate_init', 0.0),
                )
            )
        else:
            self.encoder = WiLoRViTWithGeo(
                WiLoRViTConfig(
                    wilor_ckpt_path=cfg.wilor_ckpt_path,
                    mano_mean_params=cfg.mano_mean_params,
                    image_size=cfg.image_size,
                    joint_rep="aa",
                    token_fusion_mode=cfg.token_fusion_mode,
                    sum_fusion_strategy=getattr(cfg, 'sum_fusion_strategy', 'basic'),
                    sum_geo_gate_init=float(getattr(cfg, "sum_geo_gate_init", 4.0)),
                    fusion_proj_zero_init=bool(getattr(cfg, "fusion_proj_zero_init", True)),
                    cross_attn_num_heads=getattr(cfg, 'cross_attn_num_heads', 8),
                    cross_attn_dropout=getattr(cfg, 'cross_attn_dropout', 0.0),
                    cross_attn_gate_init=getattr(cfg, 'cross_attn_gate_init', 0.0),
                )
            )

        # HaMeR-style MANO regression head (cross-attn decoder + IEF)
        self.mano_head = MANOTransformerDecoderHead(
            MANOHeadConfig(
                mean_params_path=cfg.mano_mean_params,
                num_hand_joints=15,
                joint_rep="aa",
                ief_iters=int(cfg.mano_head_ief_iters),
                transformer_input=str(cfg.mano_head_transformer_input),
                dim=int(cfg.mano_head_dim),
                depth=int(cfg.mano_head_depth),
                heads=int(cfg.mano_head_heads),
                dim_head=int(cfg.mano_head_dim_head),
                mlp_dim=int(cfg.mano_head_mlp_dim),
                dropout=float(cfg.mano_head_dropout),
            ),
            context_dim=1280,
        )

        self.mano = MANOLayer(MANOConfig(model_path=cfg.mano_model_path, mean_params=cfg.mano_mean_params))
        self.mano_decoder = str(getattr(cfg, "mano_decoder", "wilor")).lower()
        self.freihand_mano_root = getattr(cfg, "freihand_mano_root", None)
        self._freihand_mano_layer = None

        # Feature Refiner (lightweight refinement module)
        refiner_method = str(getattr(cfg, "feature_refiner_method", "none")).lower()
        self.feature_refiner = None
        if refiner_method != "none":
            refiner_cfg = FeatureRefinerConfig(
                method=refiner_method,
                feat_dim=int(getattr(cfg, "feature_refiner_feat_dim", 1280)),
                num_joints=16,
                joint_rep="aa",
                sjta_bottleneck_dim=int(getattr(cfg, "feature_refiner_sjta_bottleneck_dim", 256)),
                sjta_num_heads=int(getattr(cfg, "feature_refiner_sjta_num_heads", 4)),
                sjta_use_2d_prior=bool(getattr(cfg, "feature_refiner_sjta_use_2d_prior", True)),
                sjta_num_steps=int(getattr(cfg, "feature_refiner_sjta_num_steps", 2)),
                coear_dilation1=int(getattr(cfg, "feature_refiner_coear_dilation1", 1)),
                coear_dilation2=int(getattr(cfg, "feature_refiner_coear_dilation2", 2)),
                coear_gate_reduction=int(getattr(cfg, "feature_refiner_coear_gate_reduction", 8)),
                coear_init_alpha=float(getattr(cfg, "feature_refiner_coear_init_alpha", 0.1)),
                wilor_msf_bottleneck_ratio=int(getattr(cfg, "feature_refiner_wilor_msf_bottleneck_ratio", 4)),
                wilor_msf_dilation1=int(getattr(cfg, "feature_refiner_wilor_msf_dilation1", 1)),
                wilor_msf_dilation2=int(getattr(cfg, "feature_refiner_wilor_msf_dilation2", 2)),
                wilor_msf_dilation3=int(getattr(cfg, "feature_refiner_wilor_msf_dilation3", 3)),
                wilor_msf_gate_reduction=int(getattr(cfg, "feature_refiner_wilor_msf_gate_reduction", 8)),
                wilor_msf_init_alpha=float(getattr(cfg, "feature_refiner_wilor_msf_init_alpha", 0.1)),
                kcr_num_keypoints=int(getattr(cfg, "feature_refiner_kcr_num_keypoints", 21)),
                kcr_hidden_dim=int(getattr(cfg, "feature_refiner_kcr_hidden_dim", 128)),
            )
            self.feature_refiner = build_feature_refiner(refiner_cfg)

    def _init_geo_tokenizer_if_needed(self, feat: torch.Tensor) -> None:
        if self.geo_tokenizer is not None:
            return
        in_channels = feat.shape[1]
        out_hw = (self.cfg.image_hw[0] // 16, self.cfg.image_hw[1] // 16)  # (16, 12) for (256,192)
        use_pooling = getattr(self.cfg, 'geo_tokenizer_use_pooling', True)
        self.geo_tokenizer = GeoTokenizer(
            in_channels=in_channels, embed_dim=1280, out_hw=out_hw, use_pooling=use_pooling
        ).to(device=feat.device, dtype=feat.dtype)

    @staticmethod
    def _rotmat_to_aa(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos = (trace - 1.0) * 0.5
        cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos)
        sin = torch.sin(angle)

        rx = R[..., 2, 1] - R[..., 1, 2]
        ry = R[..., 0, 2] - R[..., 2, 0]
        rz = R[..., 1, 0] - R[..., 0, 1]
        r = torch.stack([rx, ry, rz], dim=-1)

        denom = (2.0 * sin).unsqueeze(-1)
        axis = r / denom
        aa = axis * angle.unsqueeze(-1)

        small = (sin.abs() < 1e-4).unsqueeze(-1)
        aa_small = 0.5 * r
        return torch.where(small, aa_small, aa)

    @staticmethod
    def _freihand_kp21_from_verts_mm(verts_mm: torch.Tensor, J_reg: torch.Tensor) -> torch.Tensor:
        j16_mm = torch.einsum("ji,bik->bjk", J_reg, verts_mm)
        B = verts_mm.shape[0]
        kp21_mm = torch.zeros((B, 21, 3), device=verts_mm.device, dtype=verts_mm.dtype)

        mano_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], device=verts_mm.device, dtype=torch.long)
        dst_ids = torch.tensor([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3], device=verts_mm.device, dtype=torch.long)
        kp21_mm.index_copy_(1, dst_ids, j16_mm.index_select(1, mano_ids))

        tip_vids = torch.tensor([744, 320, 443, 555, 672], device=verts_mm.device, dtype=torch.long)
        tip_dst = torch.tensor([4, 8, 12, 16, 20], device=verts_mm.device, dtype=torch.long)
        kp21_mm.index_copy_(1, tip_dst, verts_mm.index_select(1, tip_vids))
        return kp21_mm

    @staticmethod
    def _kp21_from_verts(verts: torch.Tensor, J_reg: torch.Tensor) -> torch.Tensor:
        """Build FreiHAND-style 21 keypoints from MANO joints + fingertip vertices."""
        j16 = vertices2joints(J_reg, verts)  # (B,16,3)
        B = verts.shape[0]
        kp21 = torch.zeros((B, 21, 3), device=verts.device, dtype=verts.dtype)

        mano_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            device=verts.device,
            dtype=torch.long,
        )
        dst_ids = torch.tensor(
            [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3],
            device=verts.device,
            dtype=torch.long,
        )
        kp21.index_copy_(1, dst_ids, j16.index_select(1, mano_ids))

        tip_vids = torch.tensor([744, 320, 443, 555, 672], device=verts.device, dtype=torch.long)
        tip_dst_ids = torch.tensor([4, 8, 12, 16, 20], device=verts.device, dtype=torch.long)
        kp21.index_copy_(1, tip_dst_ids, verts.index_select(1, tip_vids))
        return kp21

    def _init_freihand_mano_layer(self, device: torch.device, dtype: torch.dtype):
        if self._freihand_mano_layer is not None:
            return
        try:
            from manopth.manolayer import ManoLayer
        except Exception:
            # Fallback to local manopth copies in this repo
            fallback_roots = [
                Path(__file__).resolve().parents[3] / "HandGCAT" / "common" / "utils" / "manopth",
                Path(__file__).resolve().parents[3] / "HandOccNet" / "common" / "utils" / "manopth",
            ]
            for root in fallback_roots:
                if root.exists() and str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                    break
            from manopth.manolayer import ManoLayer  # type: ignore

        if self.freihand_mano_root is None:
            freihand_mano_root = Path(__file__).resolve().parents[3] / "freihand" / "data"
        else:
            freihand_mano_root = Path(self.freihand_mano_root)
        if not (freihand_mano_root / "MANO_RIGHT.pkl").exists():
            raise FileNotFoundError(
                f"FreiHAND MANO_RIGHT.pkl not found: {freihand_mano_root}/MANO_RIGHT.pkl"
            )

        self._freihand_mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side="right",
            mano_root=str(freihand_mano_root),
            use_pca=False,
        ).to(device=device, dtype=dtype)
        self._freihand_mano_layer.eval()

    def forward(
        self,
        img_01: torch.Tensor,
        cam_param: Optional[torch.Tensor] = None,
        apply_rgb_token_mask: bool = False,
        rgb_mask_ratio: float = 0.0,
        rgb_mask_mode: str = "block",
        rgb_mask_fill: str = "zero",
    ) -> Dict[str, Any]:
        """
        Args:
            img_01: (B,3,H,W) cropped RGB.
                    - Some dataloaders may output [0,1] RGB.
                    - Some may output ImageNet-normalized RGB (mean/std).
                    This forward auto-detects and handles both to keep MoGe2 and WiLoR paths consistent.
        Returns:
            dict including:
              - pred_mano_params
              - pred_cam_t
              - pred_vertices
              - pred_keypoints_3d
        """
        # Input is expected to already match WiLoR ViT aspect ratio (e.g., 256x192).
        # For safety, enforce channel-first and resize to the configured geometry size.
        img_crop = img_01
        if img_crop.ndim == 4 and img_crop.shape[1] != 3 and img_crop.shape[-1] == 3:
            img_crop = img_crop.permute(0, 3, 1, 2).contiguous()
        if img_crop.shape[-2:] != self.cfg.image_hw:
            img_crop = F.interpolate(img_crop, size=self.cfg.image_hw, mode="bilinear", align_corners=False)

        # Detect whether the input is ImageNet-normalized or [0,1].
        # Heuristic: normalized tensors typically contain negatives and/or values > 1.
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        is_imagenet_norm = bool((img_crop.min() < -0.2) or (img_crop.max() > 1.2))

        geo_tokens = None
        geo_pos = None
        if self.use_geo_prior:
            assert self.moge2 is not None
            assert self.geo_pos is not None

            # Geometry prior (frozen). MoGe2 expects [0,1] RGB.
            if is_imagenet_norm:
                img_moge = (img_crop * std + mean).clamp(0.0, 1.0)
            else:
                img_moge = img_crop.clamp(0.0, 1.0)
            geo_feat = self.moge2(img_moge)  # (B,Cg,Hg,Wg)
            # MoGe2Prior runs under inference_mode; clone to make it a normal tensor usable by autograd modules.
            geo_feat = geo_feat.clone()
            self._init_geo_tokenizer_if_needed(geo_feat)
            assert self.geo_tokenizer is not None

            geo_tokens, coords = self.geo_tokenizer(geo_feat)  # (B,N2,1280), (Hg,Wg,2)
            if self.geo_side_tuning is not None:
                geo_tokens = self.geo_side_tuning(geo_tokens)
            geo_drop_prob = float(getattr(self.cfg, "geo_branch_dropout_prob", 0.0))
            if self.training and geo_drop_prob > 0.0:
                geo_drop_prob = min(max(geo_drop_prob, 0.0), 1.0)
                if geo_drop_prob >= 1.0:
                    geo_tokens = torch.zeros_like(geo_tokens)
                else:
                    keep = (torch.rand((geo_tokens.shape[0], 1, 1), device=geo_tokens.device) >= geo_drop_prob).to(
                        dtype=geo_tokens.dtype
                    )
                    geo_tokens = geo_tokens * keep / (1.0 - geo_drop_prob)
            geo_pos = self.geo_pos(coords)  # (N2,1280)

        # WiLoR expects ImageNet normalization.
        img_wilor = img_crop if is_imagenet_norm else (img_crop - mean) / std

        # Per-sample focal (preferred): cam_param=(fx,fy,cx,cy) in pixels after crop/affine.
        focal_length_px = None
        if cam_param is not None:
            if cam_param.ndim != 2 or cam_param.shape[-1] < 2:
                raise ValueError(f"cam_param must be (B,4) or (B,>=2), got {tuple(cam_param.shape)}")
            focal_length_px = cam_param[:, :2]

        enc_out = self.encoder(
            img_wilor,
            geo_tokens=geo_tokens,
            geo_pos=geo_pos,
            focal_length_px=focal_length_px,
            apply_rgb_token_mask=apply_rgb_token_mask,
            rgb_mask_ratio=rgb_mask_ratio,
            rgb_mask_mode=rgb_mask_mode,
            rgb_mask_fill=rgb_mask_fill,
        )
        img_feat = enc_out["img_feat"]
        if self.feature_refiner is not None and self.feature_refiner.is_feature_enhancer:
            img_feat = self.feature_refiner.enhance_features(img_feat)

        # HaMeR-style MANO prediction from conditioning features (already geo-guided via token fusion)
        mano_params, pred_cam = self.mano_head(img_feat)

        B = img_feat.shape[0]
        if focal_length_px is None:
            focal = torch.full((B, 2), float(self.cfg.focal_length), device=img_feat.device, dtype=img_feat.dtype)
        else:
            fl = focal_length_px.to(device=img_feat.device, dtype=img_feat.dtype)
            if fl.ndim == 0:
                focal = fl.view(1, 1).expand(B, 2)
            elif fl.ndim == 1:
                focal = fl.view(-1, 1).expand(B, 2)
            elif fl.ndim == 2 and fl.shape[1] == 2:
                focal = fl
            else:
                raise ValueError(f"Unsupported focal_length_px shape: {tuple(fl.shape)} (expected (B,2) or broadcastable)")

        # Feature Refiner: refine MANO parameters or enhance features
        kcr_aux: Optional[Dict[str, torch.Tensor]] = None
        if self.feature_refiner is not None and not self.feature_refiner.is_feature_enhancer:
            # SJTA/KCR: refine MANO parameters after initial prediction.
            # SJTA can be applied iteratively (IEF-style), reusing the same refiner weights.
            num_steps = 1
            if self.feature_refiner.method == "sjta":
                num_steps = int(getattr(self.feature_refiner.cfg, "sjta_num_steps", 1))
                num_steps = max(1, num_steps)

            for _ in range(num_steps):
                with torch.no_grad():
                    init_mano_out = self.mano(mano_params, pose2rot=False)
                    if self.feature_refiner.method in ("sjta", "kcr"):
                        # Use 21 keypoints (16 MANO joints + 5 fingertips) for both:
                        # - SJTA queries (better fingertip evidence)
                        # - KCR 2D supervision (avoid dummy padding points)
                        J_reg = getattr(self.mano.mano, "J_regressor", None)
                        if J_reg is None:
                            raise AttributeError("MANO layer is missing J_regressor; cannot build 21 keypoints.")
                        init_joints = self._kp21_from_verts(init_mano_out.vertices, J_reg)
                    else:
                        init_joints = vertices2joints(self.mano.mano.J_regressor, init_mano_out.vertices)

                pred_mano_feats = {
                    "betas": mano_params["betas"],
                    "cam": pred_cam,
                }
                # Provide refiner-specific pose representation.
                if self.feature_refiner.method == "sjta":
                    # SJTA expects rotation matrices and applies Exp(delta) @ R.
                    pred_mano_feats["pose_rotmat"] = torch.cat(
                        [mano_params["global_orient"], mano_params["hand_pose"]], dim=1
                    )
                elif self.feature_refiner.method == "kcr":
                    # KCR expects axis-angle (or 6d) for all 16 joints (global + 15).
                    pose_rm = torch.cat([mano_params["global_orient"], mano_params["hand_pose"]], dim=1)
                    pose_aa = self._rotmat_to_aa(pose_rm.view(-1, 3, 3)).view(B, -1)
                    pred_mano_feats["hand_pose"] = pose_aa
                else:
                    # Fallback: default to rotmats to keep behavior stable.
                    pred_mano_feats["pose_rotmat"] = torch.cat(
                        [mano_params["global_orient"], mano_params["hand_pose"]], dim=1
                    )

                refined_params, pred_cam = self.feature_refiner.refine_params(
                    img_feat=img_feat,
                    joints_3d=init_joints,
                    pred_cam=pred_cam,
                    pred_mano_feats=pred_mano_feats,
                    focal_length=focal,
                    img_size=float(self.cfg.image_size),
                )
                if self.feature_refiner.method == "kcr":
                    kcr_aux = getattr(self.feature_refiner, "last_aux", None)
                if refined_params is not None:
                    mano_params = refined_params

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2.0 * focal[:, 0] / (self.cfg.image_size * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        if self.mano_decoder == "freihand_legacy":
            # Decode with legacy MANO assets (FreiHAND toolbox) using manopth.
            self._init_freihand_mano_layer(device=img_feat.device, dtype=img_feat.dtype)
            pose_rm = torch.cat([mano_params["global_orient"], mano_params["hand_pose"]], dim=1)  # (B,16,3,3)
            pose_aa = self._rotmat_to_aa(pose_rm.view(-1, 3, 3)).view(B, 16, 3).reshape(B, 48)
            betas = mano_params["betas"]
            trans = torch.zeros((B, 3), device=img_feat.device, dtype=img_feat.dtype)
            verts_mm, _ = self._freihand_mano_layer(pose_aa, betas, trans)  # mm
            J_reg = getattr(self._freihand_mano_layer, "th_J_regressor", None)
            if J_reg is None:
                raise AttributeError("freihand MANO layer missing th_J_regressor")
            kp21_mm = self._freihand_kp21_from_verts_mm(verts_mm, J_reg)
            v_cam = verts_mm / 1000.0
            j_cam = kp21_mm / 1000.0
        else:
            # MANO already applies global_orient internally. Keep outputs in MANO camera frame
            # and use pred_cam_t only for perspective projection (HaMeR convention).
            mano_out = self.mano(mano_params, pose2rot=False)
            v_cam = mano_out.vertices
            # Build FreiHAND 21 keypoints from MANO regressor + fingertip vertices.
            B = v_cam.shape[0]
            J_reg = getattr(self.mano.mano, "J_regressor", None)
            if J_reg is None:
                raise AttributeError("MANO layer is missing J_regressor; cannot build FreiHAND 21 keypoints.")
            j_cam = self._kp21_from_verts(v_cam, J_reg)

        out = {
            "img_feat": img_feat,
            "pred_mano_params": mano_params,
            "pred_cam": pred_cam,
            "pred_cam_t": pred_cam_t,
            # MANO output in camera frame (meters); translation is used in projection.
            "pred_vertices": v_cam,
            "pred_keypoints_3d": j_cam,
            # Debug: the raw 21 joints from the MANO wrapper (may use a different definition)
            "pred_keypoints_3d_raw": None if self.mano_decoder == "freihand_legacy" else mano_out.joints,
        }
        if kcr_aux is not None:
            out.update(kcr_aux)
        return out

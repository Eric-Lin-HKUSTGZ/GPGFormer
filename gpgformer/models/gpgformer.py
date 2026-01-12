from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from gpgformer.models.encoders.wilor_vit import WiLoRViTConfig, WiLoRViTWithGeo
from gpgformer.models.heads.mano_head import MANOHeadConfig, MANOTransformerDecoderHead
from gpgformer.models.mano.mano_layer import MANOConfig, MANOLayer
from gpgformer.models.priors.moge2 import MoGe2Config, MoGe2Prior
from gpgformer.models.tokenizers.geo_tokenizer import GeoTokenizer, CoordPosEmbed


@dataclass(frozen=True)
class GPGFormerConfig:
    # Weights
    wilor_ckpt_path: str
    moge2_weights_path: str
    mano_model_path: str
    mano_mean_params: str

    # Camera / image settings
    image_size: int = 256
    image_hw: tuple[int, int] = (256, 192)
    focal_length: float = 5000.0

    # MoGe2
    moge2_num_tokens: int = 1600

    # HaMeR-style MANO head (defaults are reasonable if not provided by YAML)
    mano_head_ief_iters: int = 3
    mano_head_transformer_input: str = "mean_shape"  # 'mean_shape' or 'zero'
    mano_head_dim: int = 1024
    mano_head_depth: int = 6
    mano_head_heads: int = 8
    mano_head_dim_head: int = 64
    mano_head_mlp_dim: int = 2048
    mano_head_dropout: float = 0.0


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

        self.moge2 = MoGe2Prior(MoGe2Config(weights_path=cfg.moge2_weights_path, num_tokens=cfg.moge2_num_tokens))

        # Geo tokenizer: in_channels depends on MoGe2 neck config; infer lazily at first forward.
        self.geo_tokenizer: GeoTokenizer | None = None
        self.geo_pos = CoordPosEmbed(embed_dim=1280)

        self.encoder = WiLoRViTWithGeo(
            WiLoRViTConfig(
                wilor_ckpt_path=cfg.wilor_ckpt_path,
                mano_mean_params=cfg.mano_mean_params,
                image_size=cfg.image_size,
                focal_length=cfg.focal_length,  # fallback when cam_param is not provided
                joint_rep="aa",
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

    def _init_geo_tokenizer_if_needed(self, feat: torch.Tensor) -> None:
        if self.geo_tokenizer is not None:
            return
        in_channels = feat.shape[1]
        # Pool geometry features to a small grid to avoid exploding token counts.
        out_hw = (self.cfg.image_hw[0] // 16, self.cfg.image_hw[1] // 16)  # (16, 12) for (256,192)
        self.geo_tokenizer = GeoTokenizer(in_channels=in_channels, embed_dim=1280, out_hw=out_hw).to(
            device=feat.device, dtype=feat.dtype
        )

    def forward(self, img_01: torch.Tensor, cam_param: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Args:
            img_01: (B,3,H,W) cropped RGB in [0,1]. Dataloaders commonly output 256x256;
                    we will center-crop to 256x192 for WiLoR ViT compatibility.
        Returns:
            dict including:
              - pred_mano_params
              - pred_cam_t
              - pred_vertices
              - pred_keypoints_3d
        """
        # Match WiLoR ViT expected aspect ratio (192x256) by slicing width.
        img_crop = img_01
        if img_crop.shape[-1] == 256:
            img_crop = img_crop[:, :, :, 32:-32]  # (B,3,256,192)

        # Geometry prior (frozen)
        geo_feat = self.moge2(img_crop)  # (B,Cg,Hg,Wg)
        # MoGe2Prior runs under inference_mode; clone to make it a normal tensor usable by autograd modules.
        geo_feat = geo_feat.clone()
        self._init_geo_tokenizer_if_needed(geo_feat)
        assert self.geo_tokenizer is not None

        geo_tokens, coords = self.geo_tokenizer(geo_feat)  # (B,N2,1280), (Hg,Wg,2)
        geo_pos = self.geo_pos(coords)  # (N2,1280)

        # WiLoR expects ImageNet normalization, while MoGe2 expects [0,1] (and normalizes internally).
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        img_wilor = (img_crop - mean) / std

        # Per-sample focal (preferred): cam_param=(fx,fy,cx,cy) in pixels after crop/affine.
        focal_length_px = None
        if cam_param is not None:
            if cam_param.ndim != 2 or cam_param.shape[-1] < 2:
                raise ValueError(f"cam_param must be (B,4) or (B,>=2), got {tuple(cam_param.shape)}")
            focal_length_px = cam_param[:, :2]

        enc_out = self.encoder(img_wilor, geo_tokens=geo_tokens, geo_pos=geo_pos, focal_length_px=focal_length_px)
        img_feat = enc_out["img_feat"]

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

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2.0 * focal[:, 0] / (self.cfg.image_size * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        mano_out = self.mano(mano_params, pose2rot=False)
        pred_vertices = mano_out.vertices
        pred_joints = mano_out.joints

        return {
            "img_feat": img_feat,
            "pred_mano_params": mano_params,
            "pred_cam": pred_cam,
            "pred_cam_t": pred_cam_t,
            "pred_vertices": pred_vertices,
            "pred_keypoints_3d": pred_joints,
        }



        geo_feat = self.moge2(img_crop)  # (B,Cg,Hg,Wg)
        # MoGe2Prior runs under inference_mode; clone to make it a normal tensor usable by autograd modules.
        geo_feat = geo_feat.clone()
        self._init_geo_tokenizer_if_needed(geo_feat)
        assert self.geo_tokenizer is not None

        geo_tokens, coords = self.geo_tokenizer(geo_feat)  # (B,N2,1280), (Hg,Wg,2)
        geo_pos = self.geo_pos(coords)  # (N2,1280)

        # WiLoR expects ImageNet normalization, while MoGe2 expects [0,1] (and normalizes internally).
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_crop.device, dtype=img_crop.dtype).view(1, 3, 1, 1)
        img_wilor = (img_crop - mean) / std

        # Per-sample focal (preferred): cam_param=(fx,fy,cx,cy) in pixels after crop/affine.
        focal_length_px = None
        if cam_param is not None:
            if cam_param.ndim != 2 or cam_param.shape[-1] < 2:
                raise ValueError(f"cam_param must be (B,4) or (B,>=2), got {tuple(cam_param.shape)}")
            focal_length_px = cam_param[:, :2]

        enc_out = self.encoder(img_wilor, geo_tokens=geo_tokens, geo_pos=geo_pos, focal_length_px=focal_length_px)
        img_feat = enc_out["img_feat"]

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

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2.0 * focal[:, 0] / (self.cfg.image_size * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        mano_out = self.mano(mano_params, pose2rot=False)
        pred_vertices = mano_out.vertices
        pred_joints = mano_out.joints

        return {
            "img_feat": img_feat,
            "pred_mano_params": mano_params,
            "pred_cam": pred_cam,
            "pred_cam_t": pred_cam_t,
            "pred_vertices": pred_vertices,
            "pred_keypoints_3d": pred_joints,
        }



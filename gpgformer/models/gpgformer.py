from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from gpgformer.models.encoders.wilor_vit import WiLoRViTConfig, WiLoRViTWithGeo
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

        out = self.encoder(img_wilor, geo_tokens=geo_tokens, geo_pos=geo_pos, focal_length_px=focal_length_px)
        mano_params = out["pred_mano_params"]

        mano_out = self.mano(mano_params, pose2rot=False)
        pred_vertices = mano_out.vertices
        pred_joints = mano_out.joints

        return {
            **out,
            "pred_vertices": pred_vertices,
            "pred_keypoints_3d": pred_joints,
        }



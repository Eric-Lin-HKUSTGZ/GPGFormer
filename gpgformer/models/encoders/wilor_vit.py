from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gpgformer.utils.attr_dict import AttrDict, to_attr_dict
from third_party.wilor_min.wilor.models.backbones.vit import vit as wilor_vit_factory


@dataclass(frozen=True)
class WiLoRViTConfig:
    wilor_ckpt_path: str
    mano_mean_params: str
    image_size: int = 256
    focal_length: float = 5000.0
    joint_rep: str = "aa"  # "6d" or "aa"


class WiLoRViTWithGeo(nn.Module):
    """
    Tokenizer1 + Transformer (WiLoR ViT-L backbone) with injected geometry tokens.

    IMPORTANT (GPGFormer variant):
    - We do NOT feed any MANO-related special tokens (pose/shape/cam) into the token sequence.
    - WiLoR ViT is used purely as a (geo-guided) feature extractor: tokens = [geo_tokens, patch_tokens].
    """

    def __init__(self, cfg: WiLoRViTConfig, geo_embed_dim: int = 1280):
        super().__init__()
        self.cfg = cfg

        # Build a minimal cfg object expected by WiLoR ViT implementation.
        cfg_obj = AttrDict()
        cfg_obj.MANO = AttrDict()
        cfg_obj.MANO.NUM_HAND_JOINTS = 15
        cfg_obj.MANO.MEAN_PARAMS = cfg.mano_mean_params

        cfg_obj.MODEL = AttrDict()
        cfg_obj.MODEL.MANO_HEAD = AttrDict()
        cfg_obj.MODEL.MANO_HEAD.JOINT_REP = cfg.joint_rep
        cfg_obj.MODEL.MANO_HEAD.INIT_DECODER_XAVIER = False

        # Instantiate ViT-Large used in WiLoR
        self.backbone = wilor_vit_factory(cfg_obj)

        # Learnable type embeddings (special/img/geo). Init to zero to keep pretrained behavior.
        # NOTE: special token type (0) remains in the table for compatibility, but is unused.
        self.type_embed = nn.Embedding(3, self.backbone.embed_dim)
        nn.init.zeros_(self.type_embed.weight)

        # Load WiLoR pretrained backbone weights
        self._load_wilor_backbone(cfg.wilor_ckpt_path)

        self.image_size = cfg.image_size
        self.focal_length = cfg.focal_length

        # We no longer use WiLoR's MANO special-token regressors; freeze them to avoid DDP unused-param pitfalls.
        for name in ["pose_emb", "shape_emb", "cam_emb", "decpose", "decshape", "deccam"]:
            m = getattr(self.backbone, name, None)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad_(False)

    def _load_wilor_backbone(self, ckpt_path: str) -> None:
        ckpt_path = str(Path(ckpt_path))
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"WiLoR ckpt not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Extract backbone.* keys if present
        if any(k.startswith("backbone.") for k in state.keys()):
            state = {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
        # Filter out mismatched shapes (e.g., when switching joint_rep 6d <-> aa).
        model_sd = self.backbone.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k not in model_sd:
                continue
            if model_sd[k].shape == v.shape:
                filtered[k] = v
                continue
            # Allow a simple warm-start when 6d->aa (or vice versa) for pose-related layers
            try:
                if k == "pose_emb.weight" and v.ndim == 2 and model_sd[k].ndim == 2:
                    # (embed_dim, in_dim)
                    in_dim = model_sd[k].shape[1]
                    if v.shape[1] >= in_dim:
                        filtered[k] = v[:, :in_dim].contiguous()
                        continue
                if k == "decpose.weight" and v.ndim == 2 and model_sd[k].ndim == 2:
                    # (out_dim, embed_dim)
                    out_dim = model_sd[k].shape[0]
                    if v.shape[0] >= out_dim:
                        filtered[k] = v[:out_dim, :].contiguous()
                        continue
                if k == "decpose.bias" and v.ndim == 1 and model_sd[k].ndim == 1:
                    out_dim = model_sd[k].shape[0]
                    if v.shape[0] >= out_dim:
                        filtered[k] = v[:out_dim].contiguous()
                        continue
            except Exception:
                pass
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        # Keep a tiny record for debugging
        self._wilor_missing = missing
        self._wilor_unexpected = unexpected
        self._wilor_skipped_mismatch = skipped

    def forward(
        self,
        img: torch.Tensor,
        geo_tokens: Optional[torch.Tensor] = None,
        geo_pos: Optional[torch.Tensor] = None,
        focal_length_px: Optional[torch.Tensor] = None,  # kept for API compatibility; unused here
    ) -> dict:
        """
        Args:
            img: (B,3,256,192) cropped RGB in WiLoR normalization space (handled outside)
            geo_tokens: (B,N2,D=1280) optional
            geo_pos: (N2,D) optional positional enc for geo tokens (broadcasted over batch)
        Returns:
            dict with:
              - img_feat: (B,1280,Hp,Wp) feature map built from patch tokens AFTER attending to geo tokens.
        """
        B = img.shape[0]

        # Patchify
        x, (Hp, Wp) = self.backbone.patch_embed(img)  # (B, N1, D)

        # WiLoR pos embedding logic
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed[:, 1:] + self.backbone.pos_embed[:, :1]

        # Assemble token sequence: [geo; img]
        toks = []
        n_geo = 0

        if geo_tokens is not None:
            if geo_tokens.shape[-1] != self.backbone.embed_dim:
                raise ValueError(f"geo_tokens dim {geo_tokens.shape[-1]} != {self.backbone.embed_dim}")
            if geo_pos is not None:
                geo_tokens = geo_tokens + geo_pos.unsqueeze(0).to(dtype=geo_tokens.dtype, device=geo_tokens.device)
            toks.append(geo_tokens)
            n_geo = geo_tokens.shape[1]

        toks.append(x)
        x = torch.cat(toks, dim=1)

        # Apply type embeddings: 1=img, 2=geo (0=special unused)
        if n_geo == 0:
            type_ids = torch.ones((x.shape[1],), device=img.device, dtype=torch.long)
        else:
            n_img = x.shape[1] - n_geo
            type_ids = torch.cat(
                [
                    torch.full((n_geo,), 2, device=img.device, dtype=torch.long),
                    torch.ones((n_img,), device=img.device, dtype=torch.long),
                ],
                dim=0,
            )
        x = x + self.type_embed(type_ids).unsqueeze(0)

        # Transformer blocks
        for blk in self.backbone.blocks:
            if self.backbone.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.backbone.last_norm(x)

        # Patch tokens are always at the tail; reshape them into a feature map.
        patch_out = x if n_geo == 0 else x[:, n_geo:]
        img_feat = patch_out.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        return {
            "img_feat": img_feat,
        }



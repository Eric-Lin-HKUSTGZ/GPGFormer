from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gpgformer.utils.attr_dict import AttrDict
from third_party.wilor_min.wilor.models.backbones.vit import vit as wilor_vit_factory


@dataclass(frozen=True)
class ViTPoseViTConfig:
    vitpose_ckpt_path: str
    mano_mean_params: str
    image_size: int = 256
    focal_length: float = 5000.0
    joint_rep: str = "aa"
    token_fusion_mode: str = "concat"
    sum_fusion_strategy: str = "basic"
    # Learnable scalar gate for geometry injection in sum mode; effective gate is sigmoid(param).
    sum_geo_gate_init: float = 4.0
    fusion_proj_zero_init: bool = True
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.0
    cross_attn_gate_init: float = 0.0


class ViTPoseViTWithGeo(nn.Module):
    """ViTPose ViT-L backbone with geometry token injection (same architecture as WiLoR)."""

    def __init__(self, cfg: ViTPoseViTConfig, geo_embed_dim: int = 1280):
        super().__init__()
        self.cfg = cfg
        self.token_fusion_mode = cfg.token_fusion_mode
        self.sum_fusion_strategy = getattr(cfg, 'sum_fusion_strategy', 'basic')

        cfg_obj = AttrDict()
        cfg_obj.MANO = AttrDict()
        cfg_obj.MANO.NUM_HAND_JOINTS = 15
        cfg_obj.MANO.MEAN_PARAMS = cfg.mano_mean_params

        cfg_obj.MODEL = AttrDict()
        cfg_obj.MODEL.MANO_HEAD = AttrDict()
        cfg_obj.MODEL.MANO_HEAD.JOINT_REP = cfg.joint_rep
        cfg_obj.MODEL.MANO_HEAD.INIT_DECODER_XAVIER = False

        self.backbone = wilor_vit_factory(cfg_obj)
        self.type_embed = nn.Embedding(3, self.backbone.embed_dim)
        nn.init.zeros_(self.type_embed.weight)

        if self.token_fusion_mode == "sum":
            if self.sum_fusion_strategy in ["weighted", "weighted_normalized"]:
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            self.sum_geo_gate = nn.Parameter(torch.tensor(float(getattr(cfg, "sum_geo_gate_init", 4.0))))
            if self.sum_fusion_strategy in ["normalized", "weighted_normalized"]:
                self.patch_norm = nn.LayerNorm(self.backbone.embed_dim)
                self.geo_norm = nn.LayerNorm(self.backbone.embed_dim)
            if self.sum_fusion_strategy == "channel_concat":
                embed_dim = self.backbone.embed_dim
                self.patch_norm = nn.LayerNorm(embed_dim)
                self.geo_norm = nn.LayerNorm(embed_dim)
                self.fusion_proj = nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim)
                )
                if bool(getattr(cfg, "fusion_proj_zero_init", True)):
                    nn.init.zeros_(self.fusion_proj[-1].weight)
                    if self.fusion_proj[-1].bias is not None:
                        nn.init.zeros_(self.fusion_proj[-1].bias)

        if self.token_fusion_mode == "cross_attn":
            embed_dim = self.backbone.embed_dim
            num_heads = getattr(cfg, 'cross_attn_num_heads', 8)
            dropout = getattr(cfg, 'cross_attn_dropout', 0.0)
            gate_init = float(getattr(cfg, "cross_attn_gate_init", 0.0))
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_norm1 = nn.LayerNorm(embed_dim)
            self.cross_attn_norm2 = nn.LayerNorm(embed_dim)
            self.cross_attn_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

        self._load_vitpose_backbone(cfg.vitpose_ckpt_path)
        self.image_size = cfg.image_size

        for name in ["pose_emb", "shape_emb", "cam_emb", "decpose", "decshape", "deccam"]:
            m = getattr(self.backbone, name, None)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad_(False)

    def _load_vitpose_backbone(self, ckpt_path: str) -> None:
        ckpt_path = str(Path(ckpt_path))
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"ViTPose ckpt not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # Extract backbone.* keys if present
        if any(k.startswith("backbone.") for k in state.keys()):
            state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

        model_sd = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v

        self.backbone.load_state_dict(filtered, strict=False)

    @staticmethod
    def _apply_rgb_token_mask(
        x: torch.Tensor,
        hp: int,
        wp: int,
        mask_ratio: float,
        mask_mode: str = "block",
        mask_fill: str = "zero",
    ) -> torch.Tensor:
        ratio = float(max(0.0, min(1.0, mask_ratio)))
        if ratio <= 0.0:
            return x

        b, n, d = x.shape
        if n != hp * wp:
            return x

        mode = str(mask_mode).lower()
        fill = str(mask_fill).lower()

        if mode == "random":
            keep = (torch.rand((b, n), device=x.device) >= ratio).to(dtype=x.dtype)
            if fill == "mean":
                mean_tok = x.mean(dim=1, keepdim=True)
                return x * keep.unsqueeze(-1) + (1.0 - keep.unsqueeze(-1)) * mean_tok
            return x * keep.unsqueeze(-1)

        grid = x.view(b, hp, wp, d)
        mask = torch.zeros((b, hp, wp), device=x.device, dtype=torch.bool)
        target_area = max(1, int(round(float(hp * wp) * ratio)))
        aspect = float(wp) / float(max(hp, 1))
        for bi in range(b):
            jitter = float(torch.empty((), device=x.device).uniform_(0.8, 1.2).item())
            area = max(1, min(hp * wp, int(round(target_area * jitter))))
            h_blk = max(1, int(round((area / max(aspect, 1e-6)) ** 0.5)))
            w_blk = max(1, int(round(float(area) / float(h_blk))))
            h_blk = min(h_blk, hp)
            w_blk = min(w_blk, wp)
            top = int(torch.randint(0, hp - h_blk + 1, (1,), device=x.device).item())
            left = int(torch.randint(0, wp - w_blk + 1, (1,), device=x.device).item())
            mask[bi, top:top + h_blk, left:left + w_blk] = True

        if fill == "mean":
            mean_tok = grid.mean(dim=(1, 2), keepdim=True)
            grid = torch.where(mask.unsqueeze(-1), mean_tok.expand_as(grid), grid)
        else:
            grid = grid.masked_fill(mask.unsqueeze(-1), 0.0)
        return grid.view(b, n, d)

    def forward(
        self,
        img: torch.Tensor,
        geo_tokens: Optional[torch.Tensor] = None,
        geo_pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        B = img.shape[0]
        x, (Hp, Wp) = self.backbone.patch_embed(img)

        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed[:, 1:] + self.backbone.pos_embed[:, :1]

        # Optional RGB token masking (training-only) to simulate occlusion.
        apply_rgb_mask = bool(kwargs.get("apply_rgb_token_mask", False)) and self.training
        if apply_rgb_mask:
            x = self._apply_rgb_token_mask(
                x=x,
                hp=Hp,
                wp=Wp,
                mask_ratio=float(kwargs.get("rgb_mask_ratio", 0.0)),
                mask_mode=str(kwargs.get("rgb_mask_mode", "block")),
                mask_fill=str(kwargs.get("rgb_mask_fill", "zero")),
            )

        if self.token_fusion_mode == "sum":
            if geo_tokens is not None:
                geo_gate = torch.sigmoid(self.sum_geo_gate).to(dtype=x.dtype, device=x.device)
                if self.sum_fusion_strategy == "basic":
                    x = x + geo_gate * geo_tokens
                elif self.sum_fusion_strategy == "weighted":
                    alpha = torch.sigmoid(self.fusion_weight)
                    x = alpha * x + (1 - alpha) * (geo_gate * geo_tokens)
                elif self.sum_fusion_strategy == "normalized":
                    x = self.patch_norm(x) + geo_gate * self.geo_norm(geo_tokens)
                elif self.sum_fusion_strategy == "weighted_normalized":
                    alpha = torch.sigmoid(self.fusion_weight)
                    x = alpha * self.patch_norm(x) + (1 - alpha) * (geo_gate * self.geo_norm(geo_tokens))
                elif self.sum_fusion_strategy == "channel_concat":
                    z = torch.cat([self.patch_norm(x), self.geo_norm(geo_tokens)], dim=-1)
                    x = x + geo_gate * self.fusion_proj(z)
                if geo_pos is not None:
                    x = x + geo_pos.unsqueeze(0).to(dtype=x.dtype, device=x.device)

            for blk in self.backbone.blocks:
                x = blk(x) if not self.backbone.use_checkpoint else torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            x = self.backbone.last_norm(x)
            img_feat = x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        elif self.token_fusion_mode == "cross_attn":
            if geo_tokens is not None:
                if geo_pos is not None:
                    geo_tokens = geo_tokens + geo_pos.unsqueeze(0).to(dtype=geo_tokens.dtype, device=geo_tokens.device)
                attn_out, _ = self.cross_attn(query=self.cross_attn_norm1(x), key=self.cross_attn_norm2(geo_tokens), value=self.cross_attn_norm2(geo_tokens))
                gate = torch.tanh(self.cross_attn_gate).to(dtype=x.dtype, device=x.device)
                x = x + gate * attn_out

            for blk in self.backbone.blocks:
                x = blk(x) if not self.backbone.use_checkpoint else torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            x = self.backbone.last_norm(x)
            img_feat = x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        else:
            toks = []
            n_geo = 0
            if geo_tokens is not None:
                if geo_pos is not None:
                    geo_tokens = geo_tokens + geo_pos.unsqueeze(0).to(dtype=geo_tokens.dtype, device=geo_tokens.device)
                toks.append(geo_tokens)
                n_geo = geo_tokens.shape[1]
            toks.append(x)
            x = torch.cat(toks, dim=1)

            if n_geo == 0:
                type_ids = torch.ones((x.shape[1],), device=img.device, dtype=torch.long)
            else:
                n_img = x.shape[1] - n_geo
                type_ids = torch.cat([
                    torch.full((n_geo,), 2, device=img.device, dtype=torch.long),
                    torch.ones((n_img,), device=img.device, dtype=torch.long),
                ], dim=0)
            x = x + self.type_embed(type_ids).unsqueeze(0)

            for blk in self.backbone.blocks:
                x = blk(x) if not self.backbone.use_checkpoint else torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            x = self.backbone.last_norm(x)
            patch_out = x if n_geo == 0 else x[:, n_geo:]
            img_feat = patch_out.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        return {"img_feat": img_feat}

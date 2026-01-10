from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gpgformer.utils.attr_dict import AttrDict, to_attr_dict
from third_party.wilor_min.wilor.utils.geometry import rot6d_to_rotmat, aa_to_rotmat
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

    We reuse WiLoR's ViT which already includes pose/shape/cam special tokens and decoders.
    We inject geo tokens between the special tokens and the image patch tokens.
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
        self.type_embed = nn.Embedding(3, self.backbone.embed_dim)
        nn.init.zeros_(self.type_embed.weight)

        # Load WiLoR pretrained backbone weights
        self._load_wilor_backbone(cfg.wilor_ckpt_path)

        self.image_size = cfg.image_size
        self.focal_length = cfg.focal_length

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
        focal_length_px: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            img: (B,3,256,192) cropped RGB in WiLoR normalization space (handled outside)
            geo_tokens: (B,N2,D=1280) optional
            geo_pos: (N2,D) optional positional enc for geo tokens (broadcasted over batch)
        Returns:
            dict with pred_mano_params (rotmats), pred_cam (weak-persp), pred_cam_t (persp trans),
            and img_feat (B,1280,Hp,Wp) like WiLoR.
        """
        B = img.shape[0]

        # Patchify
        x, (Hp, Wp) = self.backbone.patch_embed(img)  # (B, N1, D)

        # WiLoR pos embedding logic
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed[:, 1:] + self.backbone.pos_embed[:, :1]

        # Special tokens (pose/shape/cam) created exactly like WiLoR
        pose_tokens = self.backbone.pose_emb(
            self.backbone.init_hand_pose.reshape(1, 16, self.backbone.joint_rep_dim)
        ).repeat(B, 1, 1)
        shape_tokens = self.backbone.shape_emb(self.backbone.init_betas).unsqueeze(1).repeat(B, 1, 1)
        cam_tokens = self.backbone.cam_emb(self.backbone.init_cam).unsqueeze(1).repeat(B, 1, 1)

        # Assemble token sequence: [special; geo; img]
        toks = [pose_tokens, shape_tokens, cam_tokens]

        if geo_tokens is not None:
            if geo_tokens.shape[-1] != self.backbone.embed_dim:
                raise ValueError(f"geo_tokens dim {geo_tokens.shape[-1]} != {self.backbone.embed_dim}")
            if geo_pos is not None:
                geo_tokens = geo_tokens + geo_pos.unsqueeze(0).to(dtype=geo_tokens.dtype, device=geo_tokens.device)
            toks.append(geo_tokens)

        toks.append(x)
        x = torch.cat(toks, dim=1)

        # Apply type embeddings: 0=special, 1=img, 2=geo
        n_special = pose_tokens.shape[1] + shape_tokens.shape[1] + cam_tokens.shape[1]
        if geo_tokens is None:
            type_ids = torch.cat(
                [
                    torch.zeros(n_special, device=img.device, dtype=torch.long),
                    torch.ones(x.shape[1] - n_special, device=img.device, dtype=torch.long),
                ],
                dim=0,
            )
        else:
            n_geo = geo_tokens.shape[1]
            type_ids = torch.cat(
                [
                    torch.zeros(n_special, device=img.device, dtype=torch.long),
                    torch.full((n_geo,), 2, device=img.device, dtype=torch.long),
                    torch.ones(x.shape[1] - n_special - n_geo, device=img.device, dtype=torch.long),
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

        # Decode pose/shape/cam (same slicing as WiLoR)
        pose_feat = x[:, :16]
        shape_feat = x[:, 16:17]
        cam_feat = x[:, 17:18]

        pred_hand_pose = self.backbone.decpose(pose_feat).reshape(B, -1) + self.backbone.init_hand_pose
        pred_betas = self.backbone.decshape(shape_feat).reshape(B, -1) + self.backbone.init_betas
        pred_cam = self.backbone.deccam(cam_feat).reshape(B, -1) + self.backbone.init_cam

        # Convert pose to rotmats according to joint_rep
        if self.backbone.joint_rep_type == "6d":
            pred_hand_pose_rm = rot6d_to_rotmat(pred_hand_pose.reshape(-1, 6)).view(B, 16, 3, 3)
        elif self.backbone.joint_rep_type == "aa":
            pred_hand_pose_rm = aa_to_rotmat(pred_hand_pose.reshape(-1, 3).contiguous()).view(B, 16, 3, 3)
        else:
            raise ValueError(f"Unsupported joint_rep_type: {self.backbone.joint_rep_type}")
        pred_mano_params = {
            "global_orient": pred_hand_pose_rm[:, [0]],
            "hand_pose": pred_hand_pose_rm[:, 1:],
            "betas": pred_betas,
        }

        # Image feature map tokens are after: special + (optional geo)
        offset = n_special + (0 if geo_tokens is None else geo_tokens.shape[1])
        img_feat = x[:, offset:].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()

        # Compute camera translation (same as WiLoR; focal scaled outside for full-image later)
        if focal_length_px is None:
            focal = torch.full((B, 2), float(self.focal_length), device=img.device, dtype=img.dtype)
        else:
            fl = focal_length_px.to(device=img.device, dtype=img.dtype)
            # Accept (B,2) or (B,) / (,) and broadcast to (B,2)
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
                2.0 * focal[:, 0] / (self.image_size * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        return {
            "pred_mano_params": pred_mano_params,
            "pred_cam": pred_cam,
            "pred_cam_t": pred_cam_t,
            "img_feat": img_feat,
        }



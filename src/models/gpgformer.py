import os
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from .backbone.vit import PatchEmbed, Block, trunc_normal_
from ..utils.box import scale_bbox_xyxy, crop_and_resize, crop_feature_and_resize
from ..utils.geometry import perspective_projection
from ..utils.moge import MoGeFeatureExtractor
from ..utils.mano import MANO


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = nn.functional.normalize(a1)
    b2 = nn.functional.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def aa_to_rotmat(theta: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    angle = norm * 0.5
    normalized = theta / norm
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rot_mat = torch.stack(
        [
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(-1, 3, 3)
    return rot_mat


class GPGFormer(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int],
                 patch_size: int,
                 embed_dim: int,
                 vit_depth: int,
                 vit_num_heads: int,
                 mlp_ratio: float,
                 drop_rate: float,
                 attn_drop_rate: float,
                 drop_path_rate: float,
                 joint_rep_type: str,
                 num_hand_joints: int,
                 focal_length: float,
                 moge_checkpoint: str,
                 moge_num_tokens: int,
                 moge_use_fp16: bool,
                 mano_path: Optional[str] = None,
                 joint_regressor_extra: Optional[str] = None,
                 bbox_scale: float = 1.2):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.joint_rep_type = joint_rep_type
        self.num_hand_joints = num_hand_joints
        self.focal_length = focal_length
        self.bbox_scale = bbox_scale

        self.patch_embed_rgb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_patches = self.patch_embed_rgb.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches * 2, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, vit_depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=vit_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[i]
            ) for i in range(vit_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        pose_dim = (num_hand_joints + 1) * (6 if joint_rep_type == "6d" else 3)
        self.pose_head = nn.Linear(embed_dim, pose_dim)
        self.shape_head = nn.Linear(embed_dim, 10)
        self.cam_head = nn.Linear(embed_dim, 3)

        self.moge = MoGeFeatureExtractor(
            checkpoint=moge_checkpoint,
            num_tokens=moge_num_tokens,
            use_fp16=moge_use_fp16
        )
        self.geom_embed = nn.Conv2d(self.moge.out_channels, embed_dim, kernel_size=1)

        self.mano = None
        if mano_path is not None and os.path.exists(mano_path):
            mano_dir = mano_path if os.path.isdir(mano_path) else os.path.dirname(mano_path)
            self.mano = MANO(
                model_path=mano_dir,
                gender="neutral",
                num_hand_joints=num_hand_joints,
                use_pca=False,
                flat_hand_mean=True,
                joint_regressor_extra=joint_regressor_extra
            )

    def forward(
        self,
        rgb: torch.Tensor,
        bboxes: torch.Tensor,
        cam_param: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = rgb.shape
        device = rgb.device

        scaled_boxes = scale_bbox_xyxy(bboxes, self.bbox_scale, W, H)
        hand_patch = crop_and_resize(rgb, scaled_boxes, self.img_h, self.img_w)
        hand_tokens, (Hp, Wp) = self.patch_embed_rgb(hand_patch)

        geom_feat = self.moge(rgb)
        if geom_feat.shape[1] != self.geom_embed.in_channels:
            raise ValueError(
                f"MoGe feature channels ({geom_feat.shape[1]}) != geom_embed.in_channels ({self.geom_embed.in_channels})"
            )
        geom_crop = crop_feature_and_resize(geom_feat, scaled_boxes, H, W, Hp, Wp)
        if geom_crop.dtype != self.geom_embed.weight.dtype:
            geom_crop = geom_crop.to(dtype=self.geom_embed.weight.dtype)
        geom_tokens = self.geom_embed(geom_crop).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, hand_tokens, geom_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        pred_pose = self.pose_head(cls_out)
        pred_shape = self.shape_head(cls_out)
        pred_cam_t = self.cam_head(cls_out)

        output = {
            "pred_pose": pred_pose,
            "pred_shape": pred_shape,
            "pred_cam_t": pred_cam_t,
            "hand_patch": hand_patch
        }

        if self.mano is not None:
            if self.joint_rep_type == "aa":
                pose_aa = pred_pose.view(B, self.num_hand_joints + 1, 3)
                pose_aa_flat = pose_aa.reshape(-1, 3)
                rotmats = aa_to_rotmat(pose_aa_flat).reshape(B, self.num_hand_joints + 1, 3, 3)
                global_orient = rotmats[:, 0:1]
                hand_pose = rotmats[:, 1:]
            else:
                pose_6d = pred_pose.view(B, self.num_hand_joints + 1, 6)
                pose_6d_flat = pose_6d.reshape(-1, 6)
                rotmats = rot6d_to_rotmat(pose_6d_flat).reshape(B, self.num_hand_joints + 1, 3, 3)
                global_orient = rotmats[:, 0:1]
                hand_pose = rotmats[:, 1:]
            mano_out = self.mano(global_orient=global_orient, hand_pose=hand_pose, betas=pred_shape)
            output["pred_vertices"] = mano_out.vertices
            output["pred_joints"] = mano_out.joints

            if cam_param is not None:
                fx = cam_param[:, 0]
                fy = cam_param[:, 1]
                cx = cam_param[:, 2]
                cy = cam_param[:, 3]
                focal_length = torch.stack([fx, fy], dim=1)
                camera_center = torch.stack([cx, cy], dim=1)
            else:
                focal_length = torch.full(
                    (B, 2),
                    float(self.focal_length),
                    device=device,
                    dtype=pred_cam_t.dtype,
                )
                camera_center = torch.tensor(
                    [W / 2.0, H / 2.0],
                    device=device,
                    dtype=pred_cam_t.dtype,
                ).unsqueeze(0).repeat(B, 1)

            pred_joints_2d = perspective_projection(
                output["pred_joints"],
                translation=pred_cam_t,
                focal_length=focal_length,
                camera_center=camera_center,
            )
            x1, y1, x2, y2 = scaled_boxes.unbind(dim=1)
            widths = (x2 - x1).clamp(min=1.0)
            heights = (y2 - y1).clamp(min=1.0)
            x = (pred_joints_2d[..., 0] - x1[:, None]) / widths[:, None] * self.img_w
            y = (pred_joints_2d[..., 1] - y1[:, None]) / heights[:, None] * self.img_h
            x = x / self.img_w - 0.5
            y = y / self.img_h - 0.5
            output["pred_joints_2d"] = torch.stack([x, y], dim=-1)

        return output

"""
Losses for UTNet/GPGFormer (minimal & strict).

Requested changes:
- Remove RotationGeodesicLoss / MANOParameterPrior / AuxiliaryLoss.
- No fallback: if required keys are missing in pred/gt, raise immediately.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred_keypoints_2d: torch.Tensor,  # (B,N,2)
        gt_keypoints_2d: torch.Tensor,  # (B,N,2) or (B,N,3) with conf
        uv_valid: Optional[torch.Tensor] = None,
        bbox_expand_factor: Optional[torch.Tensor] = None,
        box_center: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gt_keypoints_2d.shape[-1] >= 3:
            conf = gt_keypoints_2d[:, :, -1:].clone()
            gt_xy = gt_keypoints_2d[:, :, :2]
        else:
            conf = torch.ones_like(pred_keypoints_2d[:, :, :1])
            gt_xy = gt_keypoints_2d

        if uv_valid is not None:
            uv_valid = uv_valid.to(device=pred_keypoints_2d.device, dtype=pred_keypoints_2d.dtype)
            if uv_valid.dim() == 1:
                conf = conf * uv_valid.view(-1, 1, 1)
            elif uv_valid.dim() == 2:
                conf = conf * uv_valid.unsqueeze(-1)
            else:
                conf = conf * uv_valid

        if bbox_expand_factor is not None:
            bef = bbox_expand_factor.to(device=pred_keypoints_2d.device, dtype=pred_keypoints_2d.dtype)
            if bef.dim() == 1:
                bef = bef.view(-1, 1, 1)
            pred_keypoints_2d = pred_keypoints_2d * bef
            gt_xy = gt_xy * bef

        if box_size is None and scale is not None:
            scale_t = scale.to(device=pred_keypoints_2d.device, dtype=pred_keypoints_2d.dtype)
            if scale_t.dim() == 1:
                scale_t = scale_t.view(-1, 1)
            box_size = (scale_t * 200.0).max(dim=1).values

        if box_size is not None:
            box_size_t = box_size.to(device=pred_keypoints_2d.device, dtype=pred_keypoints_2d.dtype)
            if box_size_t.dim() == 1:
                box_size_t = box_size_t.view(-1, 1, 1)
            if box_center is not None:
                box_center_t = box_center.to(device=pred_keypoints_2d.device, dtype=pred_keypoints_2d.dtype)
                if box_center_t.dim() == 2:
                    box_center_t = box_center_t.view(-1, 1, 2)
            else:
                box_center_t = 0.0
            pred_keypoints_2d = pred_keypoints_2d * box_size_t + box_center_t
            gt_xy = gt_xy * box_size_t + box_center_t

        per_elem = self.loss_fn(pred_keypoints_2d, gt_xy)  # (B,N,2)
        weighted = conf * per_elem

        if self.reduction == "sum":
            return weighted.sum()

        denom = conf.sum(dim=(1, 2)).clamp(min=1.0)
        return (weighted.sum(dim=(1, 2)) / denom).mean()


class Keypoint3DLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred_keypoints_3d: torch.Tensor,  # (B,N,3)
        gt_keypoints_3d: torch.Tensor,  # (B,N,3) or (B,N,4) with conf
        root_index: int,
        xyz_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ri = int(root_index)
        pred = pred_keypoints_3d - pred_keypoints_3d[:, [ri]]

        if gt_keypoints_3d.shape[-1] >= 4:
            conf = gt_keypoints_3d[:, :, -1:].clone()
            gt = gt_keypoints_3d[:, :, :3]
        else:
            conf = torch.ones_like(pred_keypoints_3d[:, :, :1])
            gt = gt_keypoints_3d
        gt = gt - gt[:, [ri]]

        if xyz_valid is not None:
            xyz_valid = xyz_valid.to(device=pred.device, dtype=pred.dtype)
            if xyz_valid.dim() == 1:
                conf = conf * xyz_valid.view(-1, 1, 1)
            elif xyz_valid.dim() == 2:
                conf = conf * xyz_valid.unsqueeze(-1)
            else:
                conf = conf * xyz_valid

        per_elem = self.loss_fn(pred, gt)  # (B,N,3)
        weighted = conf * per_elem

        if self.reduction == "sum":
            return weighted.sum()

        denom = conf.sum(dim=(1, 2)).clamp(min=1.0)
        return (weighted.sum(dim=(1, 2)) / denom).mean()


class Vertex3DLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        if root.dim() == 2:
            root = root.unsqueeze(1)
        pred_vertices = pred_vertices - root
        gt_vertices = gt_vertices - root
        per_elem = self.loss_fn(pred_vertices, gt_vertices)
        return per_elem.sum() if self.reduction == "sum" else per_elem.mean()


class ParameterLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError(f"ParameterLoss shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
        mask = mask.to(device=pred.device).bool().view(-1)
        if mask.numel() != pred.shape[0]:
            raise ValueError(f"ParameterLoss mask mismatch: mask={tuple(mask.shape)} pred_B={pred.shape[0]}")
        per = self.mse(pred, gt).mean(dim=-1)  # (B,)
        per = per[mask]
        if per.numel() == 0:
            return pred.sum() * 0.0
        return per.sum() if self.reduction == "sum" else per.mean()


class UTNetLoss(nn.Module):
    def __init__(
        self,
        w_2d: float = 1.0,
        w_3d_joint: float = 1.0,
        w_3d_vert: float = 0.0,
        w_global_orient: float = 1.0,
        w_hand_pose: float = 1.0,
        w_betas: float = 0.0,
        root_index: int = 9,
        # keep signature compatible with train.py
        w_root_abs: float = 0.0,
        w_mano_trans: float = 0.0,
        w_scale: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if float(w_root_abs) != 0.0 or float(w_mano_trans) != 0.0 or float(w_scale) != 0.0:
            raise NotImplementedError("w_root_abs / w_mano_trans / w_scale are removed in the minimal loss; set them to 0.")

        self.w_2d = float(w_2d)
        self.w_3d_joint = float(w_3d_joint)
        self.w_3d_vert = float(w_3d_vert)
        self.w_global_orient = float(w_global_orient)
        self.w_hand_pose = float(w_hand_pose)
        self.w_betas = float(w_betas)
        self.root_index = int(root_index)
        self.reduction = str(reduction)

        self.loss_2d = Keypoint2DLoss(reduction=self.reduction)
        self.loss_3d = Keypoint3DLoss(reduction=self.reduction)
        self.loss_vert = Vertex3DLoss(reduction=self.reduction)
        self.param_loss = ParameterLoss(reduction=self.reduction)

    @staticmethod
    def _require(d: Dict, key: str, where: str):
        if key not in d:
            raise KeyError(f"Missing `{key}` in {where}. Available keys: {list(d.keys())}")
        if d[key] is None:
            raise KeyError(f"`{key}` in {where} is None (strict mode).")
        return d[key]

    @staticmethod
    def _mask_from_has_param(hp, key: str, B: int, device: torch.device) -> torch.Tensor:
        if hp is None:
            raise KeyError("Missing `has_mano_params` in targets (strict mode).")
        if isinstance(hp, dict):
            v = hp.get(key, None)
            if v is None:
                raise KeyError(f"Missing `has_mano_params['{key}']` in targets (strict mode).")
            if isinstance(v, torch.Tensor):
                return v.to(device=device).bool().view(B)
            return torch.as_tensor(v, device=device).bool().view(B)
        if isinstance(hp, torch.Tensor):
            return hp.to(device=device).bool().view(B)
        return torch.as_tensor(hp, device=device).bool().view(B)

    @staticmethod
    def _is_axis_angle_flag(flags, key: str) -> bool:
        if flags is None:
            raise KeyError("Missing `mano_params_is_axis_angle` in targets (strict mode).")
        if isinstance(flags, dict):
            v = flags.get(key, None)
            if v is None:
                raise KeyError(f"Missing `mano_params_is_axis_angle['{key}']` in targets (strict mode).")
        else:
            v = flags
        if isinstance(v, torch.Tensor):
            return bool(v.bool().all().item())
        return bool(v)

    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        # strict keys (as requested)
        pred_kp2d = self._require(predictions, "keypoints_2d", "predictions")
        pred_kp3d = self._require(predictions, "keypoints_3d", "predictions")
        pred_mano = self._require(predictions, "mano_params", "predictions")
        gt_kp2d = self._require(targets, "keypoints_2d", "targets")
        gt_kp3d = self._require(targets, "keypoints_3d", "targets")

        loss_2d = self.loss_2d(
            pred_kp2d,
            gt_kp2d,
            uv_valid=targets.get("uv_valid", None),
            bbox_expand_factor=targets.get("bbox_expand_factor", None),
            box_center=targets.get("box_center", None),
            box_size=targets.get("box_size", None),
            scale=targets.get("_scale", None),
        )
        loss_3d = self.loss_3d(
            pred_kp3d,
            gt_kp3d,
            root_index=self.root_index,
            xyz_valid=targets.get("xyz_valid", None),
        )

        total = self.w_2d * loss_2d + self.w_3d_joint * loss_3d
        out: Dict[str, torch.Tensor] = {"loss_2d": loss_2d, "loss_3d_joint": loss_3d}

        if self.w_3d_vert > 0:
            pred_v = self._require(predictions, "vertices", "predictions")
            gt_v = self._require(targets, "vertices", "targets")
            root = pred_kp3d[:, self.root_index, :]
            loss_v = self.loss_vert(pred_v, gt_v, root=root)
            out["loss_3d_vert"] = loss_v
            total = total + self.w_3d_vert * loss_v

        if (self.w_global_orient > 0) or (self.w_hand_pose > 0) or (self.w_betas > 0):
            if not isinstance(pred_mano, dict):
                raise TypeError(f"`predictions['mano_params']` must be dict, got {type(pred_mano)}")
            for k in ("global_orient", "hand_pose", "betas"):
                if k not in pred_mano:
                    raise KeyError(f"Missing `mano_params['{k}']` in predictions (strict mode).")

            gt_pose = self._require(targets, "mano_pose", "targets")
            iaa = self._require(targets, "mano_params_is_axis_angle", "targets")
            has_param = self._require(targets, "has_mano_params", "targets")

            B = gt_pose.shape[0]
            dev = gt_pose.device
            has_go = self._mask_from_has_param(has_param, "global_orient", B, dev)
            has_hp = self._mask_from_has_param(has_param, "hand_pose", B, dev)
            has_b = self._mask_from_has_param(has_param, "betas", B, dev)

            if not self._is_axis_angle_flag(iaa, "global_orient"):
                raise ValueError("Non-axis-angle GT global_orient is not supported in minimal loss.")
            if not self._is_axis_angle_flag(iaa, "hand_pose"):
                raise ValueError("Non-axis-angle GT hand_pose is not supported in minimal loss.")

            gt_go_src = gt_pose[:, 0:3]
            gt_hp_src = gt_pose[:, 3:48]

            from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat

            gt_go = aa_to_rotmat(gt_go_src.reshape(-1, 3)).view(B, 3, 3).unsqueeze(1)
            gt_hp = aa_to_rotmat(gt_hp_src.reshape(-1, 3)).view(B, 15, 3, 3)

            if self.w_global_orient > 0:
                loss_go = self.param_loss(
                    pred_mano["global_orient"].reshape(B, -1),
                    gt_go.reshape(B, -1),
                    has_go,
                )
                out["loss_global_orient"] = loss_go
                total = total + self.w_global_orient * loss_go

            if self.w_hand_pose > 0:
                loss_hp = self.param_loss(
                    pred_mano["hand_pose"].reshape(B, -1),
                    gt_hp.reshape(B, -1),
                    has_hp,
                )
                out["loss_hand_pose"] = loss_hp
                total = total + self.w_hand_pose * loss_hp

            if self.w_betas > 0:
                gt_betas = self._require(targets, "mano_shape", "targets")
                loss_b = self.param_loss(
                    pred_mano["betas"].reshape(B, -1),
                    gt_betas.to(device=dev, dtype=pred_mano["betas"].dtype).reshape(B, -1),
                    has_b,
                )
                out["loss_betas"] = loss_b
                total = total + self.w_betas * loss_b

        out["total_loss"] = total
        return out


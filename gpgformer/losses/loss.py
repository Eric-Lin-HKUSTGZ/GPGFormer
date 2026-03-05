"""
Losses for UTNet/GPGFormer (minimal & strict).

Requested changes:
- Remove RotationGeodesicLoss / MANOParameterPrior / AuxiliaryLoss.
- No fallback: if required keys are missing in pred/gt, raise immediately.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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
        joint_weights: Optional[torch.Tensor] = None,  # (N,)
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

        # Normalization denominator uses validity mask ONLY (not joint_weights),
        # so that tip weighting adds extra gradient without diluting non-tip joints.
        denom = conf.sum(dim=(1, 2)).clamp(min=1.0)

        if joint_weights is not None:
            jw = joint_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1)
            if jw.shape[1] != pred.shape[1]:
                raise ValueError(
                    f"joint_weights length mismatch: got {jw.shape[1]}, expected {pred.shape[1]}"
                )
            conf = conf * jw

        per_elem = self.loss_fn(pred, gt)  # (B,N,3)
        weighted = conf * per_elem

        if self.reduction == "sum":
            return weighted.sum()

        return (weighted.sum(dim=(1, 2)) / denom).mean()


class Vertex3DLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        pred_root: torch.Tensor,
        gt_root: torch.Tensor,
    ) -> torch.Tensor:
        if pred_root.dim() == 2:
            pred_root = pred_root.unsqueeze(1)
        if gt_root.dim() == 2:
            gt_root = gt_root.unsqueeze(1)
        pred_vertices = pred_vertices - pred_root
        gt_vertices = gt_vertices - gt_root
        per_elem = self.loss_fn(pred_vertices, gt_vertices)
        return per_elem.sum() if self.reduction == "sum" else per_elem.mean()


class BoneLengthLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = str(reduction)
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred_keypoints_3d: torch.Tensor,  # (B,N,3)
        gt_keypoints_3d: torch.Tensor,  # (B,N,3) or (B,N,4) with conf
        bone_pairs: torch.Tensor,  # (E,2), long
        xyz_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bone_pairs.numel() == 0:
            return pred_keypoints_3d.sum() * 0.0

        if gt_keypoints_3d.shape[-1] >= 4:
            conf = gt_keypoints_3d[:, :, -1:].clone()
            gt = gt_keypoints_3d[:, :, :3]
        else:
            conf = torch.ones_like(pred_keypoints_3d[:, :, :1])
            gt = gt_keypoints_3d

        if xyz_valid is not None:
            xyz_valid = xyz_valid.to(device=pred_keypoints_3d.device, dtype=pred_keypoints_3d.dtype)
            if xyz_valid.dim() == 1:
                conf = conf * xyz_valid.view(-1, 1, 1)
            elif xyz_valid.dim() == 2:
                conf = conf * xyz_valid.unsqueeze(-1)
            else:
                conf = conf * xyz_valid

        idx_i = bone_pairs[:, 0]
        idx_j = bone_pairs[:, 1]

        pred_len = torch.norm(
            pred_keypoints_3d[:, idx_i, :] - pred_keypoints_3d[:, idx_j, :],
            dim=-1,
        )  # (B,E)
        gt_len = torch.norm(
            gt[:, idx_i, :] - gt[:, idx_j, :],
            dim=-1,
        )  # (B,E)

        per_bone = self.loss_fn(pred_len, gt_len)  # (B,E)
        bone_conf = conf[:, idx_i, 0] * conf[:, idx_j, 0]  # (B,E)
        weighted = bone_conf * per_bone

        if self.reduction == "sum":
            return weighted.sum()

        denom = bone_conf.sum(dim=1).clamp(min=1.0)
        return (weighted.sum(dim=1) / denom).mean()


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
    DEFAULT_TIP_JOINT_INDICES = (4, 8, 12, 16, 20)
    DEFAULT_BONE_PAIRS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    )

    def __init__(
        self,
        w_2d: float = 1.0,
        w_kcr_2d: float = 1.0,
        w_3d_joint: float = 1.0,
        w_bone_length: float = 0.0,
        w_3d_vert: float = 0.0,
        w_global_orient: float = 1.0,
        w_hand_pose: float = 1.0,
        w_betas: float = 0.0,
        joint_3d_tip_weight: float = 1.0,
        tip_joint_indices: Optional[Sequence[int]] = None,
        bone_pairs: Optional[Sequence[Sequence[int]]] = None,
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
        self.w_kcr_2d = float(w_kcr_2d)
        self.w_3d_joint = float(w_3d_joint)
        self.w_bone_length = float(w_bone_length)
        self.w_3d_vert = float(w_3d_vert)
        self.w_global_orient = float(w_global_orient)
        self.w_hand_pose = float(w_hand_pose)
        self.w_betas = float(w_betas)
        self.joint_3d_tip_weight = float(joint_3d_tip_weight)
        if tip_joint_indices is None:
            tip_joint_indices = self.DEFAULT_TIP_JOINT_INDICES
        self.tip_joint_indices = tuple(int(i) for i in tip_joint_indices)
        if bone_pairs is None:
            bone_pairs = self.DEFAULT_BONE_PAIRS
        parsed_pairs = []
        for pair in bone_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each bone pair must be length-2 sequence, got: {pair}")
            parsed_pairs.append((int(pair[0]), int(pair[1])))
        self.bone_pairs = tuple(parsed_pairs)
        self.root_index = int(root_index)
        self.reduction = str(reduction)

        self.loss_2d = Keypoint2DLoss(reduction=self.reduction)
        self.loss_3d = Keypoint3DLoss(reduction=self.reduction)
        self.loss_bone = BoneLengthLoss(reduction=self.reduction)
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

    def _build_joint_weights(self, num_joints: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.joint_3d_tip_weight == 1.0:
            return None
        weights = torch.ones((num_joints,), device=device, dtype=dtype)
        for idx in self.tip_joint_indices:
            if 0 <= idx < num_joints:
                weights[idx] = self.joint_3d_tip_weight
        return weights

    def _build_bone_pairs(self, num_joints: int, device: torch.device) -> torch.Tensor:
        valid_pairs = []
        for i, j in self.bone_pairs:
            if 0 <= i < num_joints and 0 <= j < num_joints and i != j:
                valid_pairs.append((i, j))
        if len(valid_pairs) == 0:
            return torch.empty((0, 2), device=device, dtype=torch.long)
        return torch.as_tensor(valid_pairs, device=device, dtype=torch.long)

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
        joint_weights = self._build_joint_weights(
            num_joints=pred_kp3d.shape[1],
            device=pred_kp3d.device,
            dtype=pred_kp3d.dtype,
        )
        loss_3d = self.loss_3d(
            pred_kp3d,
            gt_kp3d,
            root_index=self.root_index,
            xyz_valid=targets.get("xyz_valid", None),
            joint_weights=joint_weights,
        )

        total = self.w_2d * loss_2d + self.w_3d_joint * loss_3d
        out: Dict[str, torch.Tensor] = {"loss_2d": loss_2d, "loss_3d_joint": loss_3d}

        if self.w_bone_length > 0:
            bone_pairs = self._build_bone_pairs(num_joints=pred_kp3d.shape[1], device=pred_kp3d.device)
            loss_bone = self.loss_bone(
                pred_kp3d,
                gt_kp3d,
                bone_pairs=bone_pairs,
                xyz_valid=targets.get("xyz_valid", None),
            )
            out["loss_bone_length"] = loss_bone
            total = total + self.w_bone_length * loss_bone

        # Optional: supervise the KCR heatmap/soft-argmax branch with a lightweight 2D loss.
        # This only activates when predictions include `kcr_keypoints_2d`.
        kcr_kp2d = predictions.get("kcr_keypoints_2d", None)
        if self.w_kcr_2d > 0 and kcr_kp2d is not None:
            loss_kcr_2d = self.loss_2d(
                kcr_kp2d,
                gt_kp2d,
                uv_valid=targets.get("uv_valid", None),
                bbox_expand_factor=targets.get("bbox_expand_factor", None),
                box_center=targets.get("box_center", None),
                box_size=targets.get("box_size", None),
                scale=targets.get("_scale", None),
            )
            out["loss_kcr_2d"] = loss_kcr_2d
            total = total + self.w_kcr_2d * loss_kcr_2d

        if self.w_3d_vert > 0:
            pred_v = self._require(predictions, "vertices", "predictions")
            gt_v = self._require(targets, "vertices", "targets")
            pred_root = pred_kp3d[:, self.root_index, :]
            gt_root = targets.get("vertices_root", None)
            if gt_root is None:
                gt_root_src = gt_kp3d[:, self.root_index, :]
                gt_root = gt_root_src[:, :3] if gt_root_src.shape[-1] >= 4 else gt_root_src
            loss_v = self.loss_vert(pred_v, gt_v, pred_root=pred_root, gt_root=gt_root)
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

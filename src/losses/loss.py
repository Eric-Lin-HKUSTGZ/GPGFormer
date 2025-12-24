import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        conf = gt_keypoints_2d[..., -1].unsqueeze(-1).clone()
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[..., :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        gt_keypoints_3d = gt_keypoints_3d.clone()
        if pred_keypoints_3d.dim() == 4:
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :, pelvis_id, :].unsqueeze(dim=2)
            gt_keypoints_3d[..., :-1] = gt_keypoints_3d[..., :-1] - gt_keypoints_3d[:, :, pelvis_id, :-1].unsqueeze(dim=2)
        else:
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
            gt_keypoints_3d[..., :-1] = gt_keypoints_3d[..., :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[..., -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[..., :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = has_param * self.loss_fn(pred_param, gt_param)
        return loss_param.sum()


class GPGFormerLoss(nn.Module):
    def __init__(
        self,
        w_pose: float,
        w_shape: float,
        w_cam: float,
        w_joints_2d: float = 0.0,
        w_joints_3d: float = 0.0,
        loss_type_2d: str = "l1",
        loss_type_3d: str = "l1",
    ):
        super().__init__()
        self.w_pose = w_pose
        self.w_shape = w_shape
        self.w_cam = w_cam
        self.w_joints_2d = w_joints_2d
        self.w_joints_3d = w_joints_3d
        self.param_loss = ParameterLoss()
        self.kp2d_loss = Keypoint2DLoss(loss_type_2d)
        self.kp3d_loss = Keypoint3DLoss(loss_type_3d)

    def _has_param(self, target: torch.Tensor) -> torch.Tensor:
        return torch.ones(target.shape[0], device=target.device, dtype=target.dtype)

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        loss = 0.0
        if targets.get("mano_pose") is not None:
            has_pose = targets.get("has_mano_pose")
            if has_pose is None:
                has_pose = self._has_param(targets["mano_pose"])
            loss = loss + self.w_pose * self.param_loss(preds["pred_pose"], targets["mano_pose"], has_pose)
        if targets.get("mano_shape") is not None:
            has_shape = targets.get("has_mano_shape")
            if has_shape is None:
                has_shape = self._has_param(targets["mano_shape"])
            loss = loss + self.w_shape * self.param_loss(preds["pred_shape"], targets["mano_shape"], has_shape)
        if targets.get("cam_t") is not None:
            has_cam = targets.get("has_cam_t")
            if has_cam is None:
                has_cam = self._has_param(targets["cam_t"])
            loss = loss + self.w_cam * self.param_loss(preds["pred_cam_t"], targets["cam_t"], has_cam)

        if self.w_joints_3d > 0 and preds.get("pred_joints") is not None and targets.get("joints_3d_gt") is not None:
            pred_joints = preds["pred_joints"]
            gt_joints = targets["joints_3d_gt"]
            if gt_joints.dim() == 3 and gt_joints.shape[-1] == 3:
                conf = torch.ones_like(gt_joints[..., :1])
                gt_joints = torch.cat([gt_joints, conf], dim=-1)
            loss = loss + self.w_joints_3d * self.kp3d_loss(pred_joints, gt_joints)

        if self.w_joints_2d > 0 and preds.get("pred_joints_2d") is not None and targets.get("joints_2d") is not None:
            pred_2d = preds["pred_joints_2d"]
            gt_2d = targets["joints_2d"]
            if gt_2d.dim() == 3 and gt_2d.shape[-1] == 2:
                conf = torch.ones_like(gt_2d[..., :1])
                gt_2d = torch.cat([gt_2d, conf], dim=-1)
            loss = loss + self.w_joints_2d * self.kp2d_loss(pred_2d, gt_2d)

        return loss

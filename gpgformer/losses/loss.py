"""
Loss Functions for UTNet
包括2D/3D关键点损失、顶点损失、MANO参数正则、辅助损失
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


def _reduce_batch(x: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce a (B,) (or broadcastable) tensor across the batch dimension.
    - 'mean': average over batch
    - 'sum' : sum over batch
    """
    if reduction == 'mean':
        return x.mean()
    if reduction == 'sum':
        return x.sum()
    raise ValueError(f"Unsupported reduction: {reduction}. Expected 'mean' or 'sum'.")


class Keypoint2DLoss(nn.Module):
    """
    2D Keypoint Reprojection Loss (WiLoR style)
    L_2D = ||J_2D_pred - J_2D_gt||_1 with confidence weighting
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'sum'):
        """
        Args:
            loss_type: 'l1' or 'l2' loss
        """
        super(Keypoint2DLoss, self).__init__()
        self.reduction = reduction
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 2] or [B, S, N, 2] containing projected 2D keypoints
                                               (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 3] or [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        # Extract confidence from last dimension
        # Expected shapes:
        #   pred_keypoints_2d: (B, N, 2)
        #   gt_keypoints_2d:   (B, N, 3)  with last channel = conf
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()  # (B, N, 1)
        per_elem = self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])  # (B, N, 2)
        weighted = conf * per_elem

        if self.reduction == 'sum':
            # Match WiLoR-style scale: sum over all valid elements (no normalization).
            return weighted.sum()

        # 'mean': normalize per-sample by total confidence, then average over batch (stable across #kps)
        denom = conf.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
        loss_per_sample = weighted.sum(dim=(1, 2)) / denom  # (B,)
        return loss_per_sample.mean()


class Keypoint3DLoss(nn.Module):
    """
    3D Keypoint Loss (WiLoR style: relative to root joint)
    L_3D_joint = ||(J_3D_pred - J_root_pred) - (J_3D_gt - J_root_gt)||_1 with confidence weighting
    This removes global translation and focuses on relative joint positions.
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'sum'):
        """
        Args:
            loss_type: 'l1' or 'l2' loss
        """
        super(Keypoint3DLoss, self).__init__()
        self.reduction = reduction
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 3] or [B, S, N, 3] containing the predicted 3D keypoints
                                               (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 4] or [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
            pelvis_id (int): Index of root joint (wrist for hand, pelvis for body). Default is 0.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        
        # Debug: print statistics (controlled by environment variable)
        import os
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'[Debug 3D Loss]')
            print(f'  Before centering - Pred mean: {pred_keypoints_3d.mean():.3f}, std: {pred_keypoints_3d.std():.3f}')
            print(f'  Before centering - GT mean: {gt_keypoints_3d[:, :, :-1].mean():.3f}, std: {gt_keypoints_3d[:, :, :-1].std():.3f}')
        
        # Normalize by subtracting root joint (removes global translation)
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        
        # Extract confidence
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()  # (B, N, 1)
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]            # (B, N, 3)
        
        # Debug: print after centering
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'  After centering - Pred mean: {pred_keypoints_3d.mean():.3f}, std: {pred_keypoints_3d.std():.3f}')
            print(f'  After centering - GT mean: {gt_keypoints_3d.mean():.3f}, std: {gt_keypoints_3d.std():.3f}')
            per_element_loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)
            print(f'  Per-element loss mean: {per_element_loss.mean():.3f}, max: {per_element_loss.max():.3f}')
        
        # Compute loss with confidence weighting
        per_elem = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)  # (B, N, 3)
        weighted = conf * per_elem

        if self.reduction == 'sum':
            # Match WiLoR-style scale: sum over all valid elements (no normalization).
            return weighted.sum()

        # 'mean': normalize per-sample by total confidence, then average over batch
        denom = conf.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
        loss_per_sample = weighted.sum(dim=(1, 2)) / denom  # (B,)
        
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'  Loss value (mean): {loss_per_sample.mean().item():.3f}')
        
        return loss_per_sample.mean()


class Vertex3DLoss(nn.Module):
    """
    3D Vertex Loss (Root-relative)
    L_3D_vert = ||(M_pred - J_root_pred) - (M_gt - J_root_gt)||_1
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'sum'):
        super().__init__()
        self.reduction = reduction
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor,
                pred_root: Optional[torch.Tensor] = None, gt_root: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_vertices: (B, V, 3) predicted mesh vertices
            gt_vertices: (B, V, 3) ground truth mesh vertices
            pred_root: (B, 1, 3) or (B, 3) predicted root joint (optional)
            gt_root: (B, 1, 3) or (B, 3) ground truth root joint (optional)
        Returns:
            loss: scalar loss value
        """
        if pred_root is not None:
            if pred_root.dim() == 2:
                pred_root = pred_root.unsqueeze(1)
            pred_vertices = pred_vertices - pred_root
        if gt_root is not None:
            if gt_root.dim() == 2:
                gt_root = gt_root.unsqueeze(1)
            gt_vertices = gt_vertices - gt_root
            
        per_elem = self.loss_fn(pred_vertices, gt_vertices)
        
        if self.reduction == 'sum':
            return per_elem.sum()
        
        return per_elem.mean()


class MANOParameterPrior(nn.Module):
    """
    MANO Parameter Prior Loss
    Shape prior: ||β||_2^2 (encourage shape close to zero)
    Pose prior: optional penalty for unreasonable joint angles
    """
    def __init__(self, shape_weight: float = 1.0, pose_weight: float = 0.0, reduction: str = 'sum'):
        """
        Args:
            shape_weight: weight for shape prior
            pose_weight: weight for pose prior (0 to disable)
        """
        super().__init__()
        self.shape_weight = shape_weight
        self.pose_weight = pose_weight
        self.reduction = reduction

    def forward(self, betas: torch.Tensor, 
                hand_pose: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            betas: (B, 10) shape parameters
            hand_pose: (B, 15, 3, 3) hand pose rotation matrices (optional)
        Returns:
            dict with 'shape_prior' and optionally 'pose_prior'
        """
        losses = {}
        
        # Shape prior: L2 norm
        if self.reduction == 'mean':
            shape_prior = (betas ** 2).mean()
        elif self.reduction == 'sum':
            shape_prior = (betas ** 2).sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}. Expected 'mean' or 'sum'.")
        losses['shape_prior'] = self.shape_weight * shape_prior
        
        # Pose prior (optional): penalize large rotations
        if self.pose_weight > 0 and hand_pose is not None:
            # Convert rotation matrices to axis-angle and penalize large angles
            # Simplified: use Frobenius norm deviation from identity
            identity = torch.eye(3, device=hand_pose.device).unsqueeze(0).unsqueeze(0)
            pose_elem = ((hand_pose - identity) ** 2).sum(dim=(-2, -1))  # (B, J)
            if self.reduction == 'mean':
                pose_deviation = pose_elem.mean()
            elif self.reduction == 'sum':
                pose_deviation = pose_elem.sum()
            else:
                raise ValueError(f"Unsupported reduction: {self.reduction}. Expected 'mean' or 'sum'.")
            losses['pose_prior'] = self.pose_weight * pose_deviation
        
        return losses


class ParameterLoss(nn.Module):
    """
    MANO Parameter Loss (WiLoR style)
    Computes MSE loss on MANO parameters with mask support
    """
    def __init__(self, reduction: str = 'sum'):
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, 
                has_param: torch.Tensor) -> torch.Tensor:
        """
        Compute MANO parameter loss.
        Args:
            pred_param: (B, ...) predicted parameters (rotation matrices)
            gt_param: (B, ...) ground truth parameters (rotation matrices)
            has_param: (B,) mask indicating which samples have GT
        Returns:
            loss: scalar loss value
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        if self.reduction == 'mean':
            denom = has_param.sum().clamp(min=1.0) * pred_param[0].numel()
            return loss_param.sum() / denom
        if self.reduction == 'sum':
            return loss_param.sum()
        raise ValueError(f"Unsupported reduction: {self.reduction}. Expected 'mean' or 'sum'.")


class RotationGeodesicLoss(nn.Module):
    """
    Geodesic distance between rotation matrices (in degrees).
    Produces values in a human-interpretable range (~0..180) that are easier to balance
    against mm-scale joint losses than raw MSE on matrix elements.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'sum'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_R: torch.Tensor, gt_R: torch.Tensor, has_param: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_R: (B, ..., 3, 3)
            gt_R:   (B, ..., 3, 3)
            has_param: (B,) boolean mask
        Returns:
            scalar loss (mean geodesic angle in degrees over valid samples and joints)
        """
        B = pred_R.shape[0]
        # Relative rotation
        rel = torch.matmul(pred_R.transpose(-1, -2), gt_R)  # (B, ..., 3, 3)
        trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]  # (B, ...)
        cos = (trace - 1.0) * 0.5
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos)  # radians, (B, ...)
        angle_deg = angle * (180.0 / 3.141592653589793)

        # Mask invalid samples
        has_f = has_param.float().view(B, *([1] * (angle_deg.dim() - 1)))
        angle_deg = angle_deg * has_f
        denom = has_param.float().sum().clamp(min=1.0)
        # Mean over all non-batch dims, then reduce over valid samples
        per_sample = angle_deg.reshape(B, -1).mean(dim=1)  # (B,)
        summed = (per_sample * has_param.float()).sum()
        if self.reduction == 'mean':
            return summed / denom
        if self.reduction == 'sum':
            return summed
        raise ValueError(f"Unsupported reduction: {self.reduction}. Expected 'mean' or 'sum'.")


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary Loss for Coarse Predictions
    L_aux = λ_aux(||θ^c - θ_gt||_2^2 + ||β^c - β_gt||_2^2)
    Supports both axis-angle (48-dim) and 6D rotation (96-dim) representations.
    """
    def __init__(self, weight: float = 0.1, joint_rep_type: str = 'aa', reduction: str = 'sum'):
        """
        Args:
            weight: weight for auxiliary loss
            joint_rep_type: 'aa' for axis-angle (48-dim) or '6d' for 6D rotation (96-dim)
        """
        super().__init__()
        self.weight = weight
        self.joint_rep_type = joint_rep_type
        self.reduction = reduction
        self.pose_loss_fn = nn.MSELoss(reduction=reduction)
        self.shape_loss_fn = nn.MSELoss(reduction=reduction)

    def _aa_to_6d(self, pose_aa: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle pose to 6D rotation representation.
        Args:
            pose_aa: (B, 48) axis-angle pose (16 joints × 3)
        Returns:
            pose_6d: (B, 96) 6D rotation pose (16 joints × 6)
        """
        from ..models.backbone.vit import aa_to_rotmat
        B = pose_aa.shape[0]
        # Reshape to (B, 16, 3)
        pose_aa = pose_aa.reshape(B, 16, 3)
        # Convert to rotation matrices: (B, 16, 3, 3)
        rotmats = aa_to_rotmat(pose_aa.reshape(B * 16, 3)).reshape(B, 16, 3, 3)
        # Extract first two columns as 6D representation: (B, 16, 3, 2) -> (B, 16, 6)
        pose_6d = rotmats[:, :, :, :2].reshape(B, 16, 6).reshape(B, 96)
        return pose_6d

    def forward(self, coarse_pose: torch.Tensor, coarse_shape: torch.Tensor,
                gt_pose: torch.Tensor, gt_shape: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_pose: (B, 48) or (B, 96) coarse pose prediction
            coarse_shape: (B, 10) coarse shape prediction
            gt_pose: (B, 48) ground truth pose (always axis-angle)
            gt_shape: (B, 10) ground truth shape
        Returns:
            loss: scalar auxiliary loss
        """
        # Handle different joint representation types
        if coarse_pose.shape[1] == 96:
            # coarse_pose is 6D, convert gt_pose (axis-angle) to 6D
            gt_pose_6d = self._aa_to_6d(gt_pose)
            pose_loss = self.pose_loss_fn(coarse_pose, gt_pose_6d)
        elif coarse_pose.shape[1] == 48:
            # Both are axis-angle
            pose_loss = self.pose_loss_fn(coarse_pose, gt_pose)
        else:
            raise ValueError(f"Unexpected coarse_pose dimension: {coarse_pose.shape[1]}, expected 48 or 96")
        
        shape_loss = self.shape_loss_fn(coarse_shape, gt_shape)
        return self.weight * (pose_loss + shape_loss)


class UTNetLoss(nn.Module):
    """
    Combined Loss for UTNet
    L = w_2D*L_2D + w_3D_j*L_3D_joint + w_3D_v*L_3D_vert + w_prior*L_prior + w_aux*L_aux
        + w_scale*L_scale (optional)
        + w_betas*L_betas (optional)
    """
    def __init__(self, 
                 w_2d: float = 1.0,
                 w_3d_joint: float = 1.0,
                 w_3d_vert: float = 0.5,
                 w_prior: float = 0.01,
                 w_aux: float = 0.1,
                 w_global_orient: float = 0.001,
                 w_hand_pose: float = 0.001,
                 w_root_abs: float = 0.0,
                 w_mano_trans: float = 0.0,
                 w_scale: float = 0.0,
                 w_betas: float = 0.0,
                 use_vertex_loss: bool = True,
                 use_aux_loss: bool = True,
                 joint_rep_type: str = 'aa',
                 reduction: str = 'sum'):
        """
        Args:
            w_2d: weight for 2D keypoint loss
            w_3d_joint: weight for 3D joint loss
            w_3d_vert: weight for 3D vertex loss
            w_prior: weight for MANO parameter prior
            w_aux: weight for auxiliary loss
            w_global_orient: weight for global_orient parameter loss (WiLoR style)
            w_hand_pose: weight for hand_pose parameter loss (WiLoR style)
            w_scale: weight for scale consistency loss (recommended to reduce MPJPE vs PA-MPJPE gap)
            w_betas: weight for betas(shape) supervision loss (recommended when mano_shape GT exists)
            use_vertex_loss: whether to use vertex loss
            use_aux_loss: whether to use auxiliary loss
            joint_rep_type: 'aa' for axis-angle or '6d' for 6D rotation
        """
        super().__init__()
        self.w_2d = w_2d
        self.w_3d_joint = w_3d_joint
        self.w_3d_vert = w_3d_vert
        self.w_prior = w_prior
        self.w_aux = w_aux
        self.w_global_orient = w_global_orient
        self.w_hand_pose = w_hand_pose
        self.w_root_abs = w_root_abs
        self.w_mano_trans = w_mano_trans
        self.w_scale = w_scale
        self.w_betas = w_betas
        self.use_vertex_loss = use_vertex_loss
        self.use_aux_loss = use_aux_loss
        self.joint_rep_type = joint_rep_type
        self.reduction = reduction
        
        # Initialize loss modules (WiLoR style)
        self.loss_2d = Keypoint2DLoss(loss_type='l1', reduction=reduction)  # WiLoR uses L1 for 2D
        self.loss_3d_joint = Keypoint3DLoss(loss_type='l1', reduction=reduction)  # WiLoR uses L1 for 3D
        if use_vertex_loss:
            self.loss_3d_vert = Vertex3DLoss(reduction=reduction)
        self.loss_prior = MANOParameterPrior(reduction=reduction)
        self.parameter_loss = ParameterLoss(reduction=reduction)  # WiLoR style parameter loss
        self.rot_loss = RotationGeodesicLoss(reduction=reduction)
        self.betas_loss_fn = nn.MSELoss(reduction='none')
        if use_aux_loss:
            self.loss_aux = AuxiliaryLoss(weight=w_aux, joint_rep_type=joint_rep_type, reduction=reduction)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            predictions: dict containing:
                - keypoints_2d: (B, N, 2) predicted 2D keypoints
                - keypoints_3d: (B, N, 3) predicted 3D keypoints
                - vertices: (B, V, 3) predicted vertices (optional)
                - mano_params: dict with 'betas', 'hand_pose'
                - coarse_pose: (B, 48) coarse pose (optional)
                - coarse_shape: (B, 10) coarse shape (optional)
            targets: dict containing:
                - keypoints_2d: (B, N, 2) ground truth 2D keypoints
                - keypoints_3d: (B, N, 3) ground truth 3D keypoints
                - vertices: (B, V, 3) ground truth vertices (optional)
                - mano_pose: (B, 48) ground truth pose (optional)
                - mano_shape: (B, 10) ground truth shape (optional)
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        
        # 2D keypoint loss
        if 'keypoints_2d' in predictions and 'keypoints_2d' in targets:
            if targets['keypoints_2d'] is not None:
                loss_2d = self.loss_2d(predictions['keypoints_2d'], targets['keypoints_2d'])
                losses['loss_2d'] = loss_2d
                total_loss = self.w_2d * loss_2d
            else:
                total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        else:
            total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # 3D joint loss
        if 'keypoints_3d' in predictions and 'keypoints_3d' in targets:
            if targets['keypoints_3d'] is not None:
                loss_3d_joint = self.loss_3d_joint(predictions['keypoints_3d'], targets['keypoints_3d'])
                losses['loss_3d_joint'] = loss_3d_joint
                total_loss = total_loss + self.w_3d_joint * loss_3d_joint

                # Optional: absolute root loss (camera coordinates, mm)
                # This directly penalizes global translation drift that is invisible to root-centered 3D joint loss.
                if self.w_root_abs > 0:
                    B = predictions['keypoints_3d'].shape[0]
                    dev = predictions['keypoints_3d'].device
                    has_param = targets.get('has_mano_params', None)
                    # Use global_orient availability as proxy for "has MANO GT"
                    if has_param is None:
                        has_root = torch.zeros(B, device=dev, dtype=torch.bool)
                    elif isinstance(has_param, dict):
                        v = has_param.get('global_orient', None)
                        has_root = v.to(device=dev).bool().view(B) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=dev).bool().view(B)
                    else:
                        has_root = torch.as_tensor(has_param, device=dev).bool().view(B)

                    if has_root.any():
                        pred_root_mm = predictions['keypoints_3d'][:, 0, :].reshape(B, -1)  # (B, 3)
                        gt_root_mm = targets['keypoints_3d'][:, 0, :3].reshape(B, -1)       # (B, 3)
                        loss_root_abs = self.parameter_loss(pred_root_mm, gt_root_mm, has_root)
                        losses['loss_root_abs'] = loss_root_abs
                        total_loss = total_loss + self.w_root_abs * loss_root_abs

                # Optional: MANO translation / camera translation loss (meters)
                # Requires model to output predictions['cam_translation'] in meters and dataloader to provide targets['mano_trans'] in meters.
                if self.w_mano_trans > 0 and ('cam_translation' in predictions) and ('mano_trans' in targets) and (targets['mano_trans'] is not None):
                    B = predictions['cam_translation'].shape[0]
                    dev = predictions['cam_translation'].device
                    has_param = targets.get('has_mano_params', None)
                    if has_param is None:
                        has_t = torch.zeros(B, device=dev, dtype=torch.bool)
                    elif isinstance(has_param, dict):
                        v = has_param.get('global_orient', None)
                        has_t = v.to(device=dev).bool().view(B) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=dev).bool().view(B)
                    else:
                        has_t = torch.as_tensor(has_param, device=dev).bool().view(B)

                    if has_t.any():
                        pred_t = predictions['cam_translation'].reshape(B, -1)  # (B, 3) meters
                        gt_t = targets['mano_trans'].to(device=dev, dtype=pred_t.dtype).reshape(B, -1)  # (B, 3) meters
                        loss_mano_trans = self.parameter_loss(pred_t, gt_t, has_t)
                        losses['loss_mano_trans'] = loss_mano_trans
                        total_loss = total_loss + self.w_mano_trans * loss_mano_trans

                # Optional: scale consistency loss (root-centered)
                # This directly targets the common failure mode: MPJPE >> PA-MPJPE due to global scale mismatch.
                if self.w_scale > 0:
                    pred_j = predictions['keypoints_3d']  # (B, J, 3) in mm
                    gt_j_conf = targets['keypoints_3d']   # (B, J, 4) in mm + conf
                    gt_j = gt_j_conf[:, :, :3]
                    conf = gt_j_conf[:, :, 3:4]

                    pred_c = pred_j - pred_j[:, [0], :]
                    gt_c = gt_j - gt_j[:, [0], :]

                    # Weighted L2 norm per sample
                    pred_sq = (pred_c ** 2).sum(dim=-1, keepdim=True)  # (B, J, 1)
                    gt_sq = (gt_c ** 2).sum(dim=-1, keepdim=True)
                    pred_norm = torch.sqrt(torch.clamp((pred_sq * conf).sum(dim=1), min=1e-12))  # (B, 1)
                    gt_norm = torch.sqrt(torch.clamp((gt_sq * conf).sum(dim=1), min=1e-12))      # (B, 1)

                    # Use a numerically-stable, dimensionless log-ratio loss to avoid dominating the total loss
                    # and to reduce late-training "blow-up" caused by large mm-scale norms.
                    #
                    # loss = |log( clamp(pred_norm / gt_norm, r_min, r_max) )|
                    # - scale mismatch of 2x => |log(2)| ~ 0.693
                    # - scale mismatch of 1.1x => |log(1.1)| ~ 0.095
                    eps = 1e-6
                    ratio = pred_norm / (gt_norm + eps)  # (B, 1)
                    ratio = torch.clamp(ratio, 0.25, 4.0)
                    loss_scale = _reduce_batch(torch.abs(torch.log(ratio)).view(-1), self.reduction)
                    losses['loss_scale'] = loss_scale
                    total_loss = total_loss + self.w_scale * loss_scale
        
        # 3D vertex loss
        if self.use_vertex_loss and 'vertices' in predictions and 'vertices' in targets:
            if targets['vertices'] is not None:
                # Use root joints for centering if available
                # pred_vertices is in camera absolute coordinates, use predicted camera root for centering
                pred_root = predictions['keypoints_3d'][:, 0, :] if 'keypoints_3d' in predictions else None
                
                # gt_vertices is in MANO local space, use gt_vertices_root (local wrist) for centering
                # If gt_vertices_root is not provided, fallback to 0 (already local)
                gt_root = targets.get('vertices_root', None)
                
                loss_3d_vert = self.loss_3d_vert(predictions['vertices'], targets['vertices'],
                                                pred_root=pred_root, gt_root=gt_root)
                losses['loss_3d_vert'] = loss_3d_vert
                total_loss = total_loss + self.w_3d_vert * loss_3d_vert
        
        # MANO parameter prior
        if 'mano_params' in predictions:
            mano_params = predictions['mano_params']
            prior_losses = self.loss_prior(
                mano_params.get('betas'),
                mano_params.get('hand_pose')
            )
            for key, val in prior_losses.items():
                losses[key] = val
                total_loss = total_loss + self.w_prior * val
        
        # MANO parameter direct supervision loss (WiLoR style)
        if 'mano_params' in predictions and 'mano_pose' in targets:
            pred_mano_params = predictions['mano_params']
            has_param = targets.get('has_mano_params', None)
            is_axis_angle = targets.get('mano_params_is_axis_angle', None)

            def _mask_from_has_param(hp, key: str, B: int, device: torch.device) -> torch.Tensor:
                """
                HaMeR-style: hp can be dict {global_orient, hand_pose, betas} or a (B,) bool tensor.
                Returns a (B,) bool tensor mask.
                """
                if hp is None:
                    return torch.zeros(B, device=device, dtype=torch.bool)
                if isinstance(hp, dict):
                    v = hp.get(key, None)
                    if v is None:
                        return torch.zeros(B, device=device, dtype=torch.bool)
                    if isinstance(v, torch.Tensor):
                        return v.to(device=device).bool().view(B)
                    # allow python bool list
                    return torch.as_tensor(v, device=device).bool().view(B)
                if isinstance(hp, torch.Tensor):
                    return hp.to(device=device).bool().view(B)
                return torch.as_tensor(hp, device=device).bool().view(B)

            def _is_axis_angle_flag(iaa, key: str) -> bool:
                """
                HaMeR-style: iaa is dict {global_orient, hand_pose, betas} with bool-like values (often (B,) bool tensors).
                We follow HaMeR's logic: if iaa[key].all() is True => GT is axis-angle and must be converted to rotmat.
                """
                if iaa is None:
                    return True
                if isinstance(iaa, dict):
                    v = iaa.get(key, True)
                else:
                    v = iaa
                if isinstance(v, torch.Tensor):
                    return bool(v.bool().all().item())
                try:
                    return bool(v)
                except Exception:
                    return True

            def _ensure_rotmat(x: torch.Tensor, shape: str) -> torch.Tensor:
                """
                Convert common rotation-matrix layouts to canonical:
                  - global_orient: (B,1,3,3)
                  - hand_pose:     (B,15,3,3)
                """
                if shape == 'global':
                    if x.dim() == 4 and x.shape[-2:] == (3, 3):
                        return x if x.shape[1] == 1 else x[:, :1]
                    if x.dim() == 3 and x.shape[-2:] == (3, 3):
                        return x.unsqueeze(1)
                    if x.dim() == 2 and x.shape[1] == 9:
                        return x.view(-1, 3, 3).unsqueeze(1)
                    raise ValueError(f"Unsupported global_orient rotmat shape: {tuple(x.shape)}")
                if shape == 'hand':
                    if x.dim() == 4 and x.shape[-2:] == (3, 3):
                        return x
                    if x.dim() == 3 and x.shape[-2:] == (3, 3):
                        # (B*15,3,3) -> (B,15,3,3) is ambiguous without B; disallow
                        raise ValueError(f"Ambiguous hand_pose rotmat shape (missing joint dim): {tuple(x.shape)}")
                    if x.dim() == 2 and x.shape[1] == 15 * 9:
                        B_ = x.shape[0]
                        return x.view(B_, 15, 3, 3)
                    if x.dim() == 3 and x.shape[1:] == (15, 9):
                        return x.view(x.shape[0], 15, 3, 3)
                    raise ValueError(f"Unsupported hand_pose rotmat shape: {tuple(x.shape)}")
                raise ValueError(f"Unknown rotmat shape tag: {shape}")

            B = targets['mano_pose'].shape[0]
            dev = targets['mano_pose'].device
            has_go = _mask_from_has_param(has_param, 'global_orient', B, dev)
            has_hp = _mask_from_has_param(has_param, 'hand_pose', B, dev)
            has_betas = _mask_from_has_param(has_param, 'betas', B, dev)

            # HaMeR-style MANO parameter supervision:
            # - Convert GT axis-angle -> rotmat ONLY if mano_params_is_axis_angle[k].all() is True
            # - Use ParameterLoss (masked MSE) on flattened parameter vectors
            # - Weight each term by corresponding weight (w_global_orient/w_hand_pose/w_betas)
            if (has_go.any() or has_hp.any() or has_betas.any()) and (self.w_global_orient > 0 or self.w_hand_pose > 0 or self.w_betas > 0):
                from ..models.backbone.vit import aa_to_rotmat
                
                gt_pose = targets['mano_pose']  # can be AA or rotmat depending on flags
                gt_go_src = gt_pose[:, 0:3]
                gt_hp_src = gt_pose[:, 3:48]

                if _is_axis_angle_flag(is_axis_angle, 'global_orient'):
                    gt_global_orient = aa_to_rotmat(gt_go_src.reshape(-1, 3)).view(B, 3, 3).unsqueeze(1)
                else:
                    gt_global_orient = _ensure_rotmat(gt_go_src, 'global')

                if _is_axis_angle_flag(is_axis_angle, 'hand_pose'):
                    gt_hand_pose = aa_to_rotmat(gt_hp_src.reshape(-1, 3)).view(B, 15, 3, 3)
                else:
                    gt_hand_pose = _ensure_rotmat(gt_hp_src, 'hand')

                if self.w_global_orient > 0 and has_go.any():
                    pred = pred_mano_params['global_orient'].reshape(B, -1)
                    gt = gt_global_orient.reshape(B, -1)
                    loss_go = self.parameter_loss(pred, gt, has_go)
                    losses['loss_global_orient'] = loss_go
                    total_loss = total_loss + self.w_global_orient * loss_go

                if self.w_hand_pose > 0 and has_hp.any():
                    pred = pred_mano_params['hand_pose'].reshape(B, -1)
                    gt = gt_hand_pose.reshape(B, -1)
                    loss_hp = self.parameter_loss(pred, gt, has_hp)
                    losses['loss_hand_pose'] = loss_hp
                    total_loss = total_loss + self.w_hand_pose * loss_hp

                if self.w_betas > 0 and ('mano_shape' in targets) and (targets['mano_shape'] is not None) and has_betas.any():
                    pred_betas = pred_mano_params.get('betas', None)
                    if pred_betas is not None:
                        pred = pred_betas.reshape(B, -1)
                        gt = targets['mano_shape'].reshape(B, -1)
                        loss_b = self.parameter_loss(pred, gt, has_betas)
                        losses['loss_betas'] = loss_b
                        total_loss = total_loss + self.w_betas * loss_b
        
        # Auxiliary loss
        if self.use_aux_loss and 'coarse_pose' in predictions and 'coarse_shape' in predictions:
            if 'mano_pose' in targets and 'mano_shape' in targets:
                if targets['mano_pose'] is not None and targets['mano_shape'] is not None:
                    loss_aux = self.loss_aux(
                        predictions['coarse_pose'],
                        predictions['coarse_shape'],
                        targets['mano_pose'],
                        targets['mano_shape']
                    )
                    losses['loss_aux'] = loss_aux
                    total_loss = total_loss + loss_aux
        
        losses['total_loss'] = total_loss
        return losses

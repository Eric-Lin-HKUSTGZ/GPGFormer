"""
Evaluator class for batch evaluation
Based on WiLoR's Evaluator class
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .pose_metrics import compute_mpjpe, compute_pa_mpjpe


class Evaluator:
    """
    Class used for evaluating trained models on hand pose datasets.
    """
    def __init__(self,
                 dataset_length: int,
                 keypoint_list: Optional[List[int]] = None,
                 root_joint_idx: int = 0,
                 metrics: List[str] = ['mpjpe', 'pa_mpjpe'],
                 save_predictions: bool = False):
        """
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list (List[int]): List of keypoint indices to use for evaluation.
                                      If None, uses all keypoints.
            root_joint_idx (int): Index of root keypoint (wrist); used for centering.
            metrics (List[str]): List of evaluation metrics to record.
            save_predictions (bool): Whether to save predictions for later analysis.
        """
        self.dataset_length = dataset_length
        self.keypoint_list = keypoint_list
        self.root_joint_idx = root_joint_idx
        self.metrics = metrics
        self.save_predictions = save_predictions
        
        # Initialize metric arrays
        if self.metrics is not None:
            for metric in self.metrics:
                setattr(self, metric, np.zeros((dataset_length,)))
        
        # Initialize prediction storage
        if self.save_predictions:
            self.pred_keypoints_3d = []
            self.gt_keypoints_3d = []
            self.pred_vertices = []
            self.gt_vertices = []
        
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        
        print(f'{self.counter} / {self.dataset_length} samples')
        if self.metrics is not None:
            for metric in self.metrics:
                if metric in ['mpjpe', 'pa_mpjpe']:
                    unit = 'mm'
                else:
                    unit = ''
                mean_value = getattr(self, metric)[:self.counter].mean()
                print(f'{metric}: {mean_value:.3f} {unit}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        if self.counter == 0:
            return {}
        
        metrics_dict = {}
        if self.metrics is not None:
            for metric in self.metrics:
                mean_value = getattr(self, metric)[:self.counter].mean()
                metrics_dict[metric] = float(mean_value)
        
        return metrics_dict

    def get_predictions_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of saved predictions.
        """
        if not self.save_predictions:
            return {}
        
        return {
            'pred_keypoints_3d': np.array(self.pred_keypoints_3d),
            'gt_keypoints_3d': np.array(self.gt_keypoints_3d),
            'pred_vertices': np.array(self.pred_vertices) if len(self.pred_vertices) > 0 else None,
            'gt_vertices': np.array(self.gt_vertices) if len(self.gt_vertices) > 0 else None,
        }

    def __call__(self, 
                 pred_keypoints_3d: torch.Tensor,
                 gt_keypoints_3d: torch.Tensor,
                 pred_vertices: Optional[torch.Tensor] = None,
                 gt_vertices: Optional[torch.Tensor] = None) -> Dict:
        """
        Evaluate a batch of predictions.
        
        Args:
            pred_keypoints_3d (torch.Tensor): Predicted 3D keypoints of shape (B, N, 3).
            gt_keypoints_3d (torch.Tensor): Ground truth 3D keypoints of shape (B, N, 3).
            pred_vertices (torch.Tensor, optional): Predicted vertices of shape (B, V, 3).
            gt_vertices (torch.Tensor, optional): Ground truth vertices of shape (B, V, 3).
        Returns:
            Dict: Dictionary containing computed metrics for this batch.
        """
        batch_size = pred_keypoints_3d.shape[0]
        device = pred_keypoints_3d.device
        
        # Initialize batch metrics dictionary
        batch_metrics = {}
        
        # Center predictions and ground truth at root joint (wrist)
        pred_keypoints_3d_centered = pred_keypoints_3d - pred_keypoints_3d[:, [self.root_joint_idx], :]
        gt_keypoints_3d_centered = gt_keypoints_3d - gt_keypoints_3d[:, [self.root_joint_idx], :]

        # Filter invalid samples: zero variance or non-finite GT
        gt_var = gt_keypoints_3d_centered.var(dim=(1, 2))
        finite_mask = torch.isfinite(gt_keypoints_3d_centered).all(dim=(1, 2))
        valid_mask = (gt_var > 1e-8) & finite_mask

        if not valid_mask.any():
            # No valid samples in this batch; skip counting
            return batch_metrics

        pred_keypoints_3d_centered = pred_keypoints_3d_centered[valid_mask]
        gt_keypoints_3d_centered = gt_keypoints_3d_centered[valid_mask]
        batch_size = pred_keypoints_3d_centered.shape[0]
        
        # Select keypoints if keypoint_list is specified
        if self.keypoint_list is not None:
            pred_keypoints_3d_centered = pred_keypoints_3d_centered[:, self.keypoint_list, :]
            gt_keypoints_3d_centered = gt_keypoints_3d_centered[:, self.keypoint_list, :]
        
        # Compute metrics
        if 'mpjpe' in self.metrics:
            mpjpe = compute_mpjpe(pred_keypoints_3d_centered, gt_keypoints_3d_centered)
            # Ensure we don't exceed array bounds
            # mpjpe is a numpy array of shape (batch_size,)
            end_idx = min(self.counter + batch_size, self.dataset_length)
            actual_size = end_idx - self.counter
            if actual_size > 0:
                # Only store what fits in the array
                self.mpjpe[self.counter:end_idx] = mpjpe[:actual_size]
            batch_metrics['mpjpe'] = mpjpe
        
        if 'pa_mpjpe' in self.metrics:
            pa_mpjpe = compute_pa_mpjpe(pred_keypoints_3d_centered, gt_keypoints_3d_centered)
            # Ensure we don't exceed array bounds
            end_idx = min(self.counter + batch_size, self.dataset_length)
            actual_size = end_idx - self.counter
            if actual_size > 0:
                # Only store what fits in the array
                self.pa_mpjpe[self.counter:end_idx] = pa_mpjpe[:actual_size]
            batch_metrics['pa_mpjpe'] = pa_mpjpe
        
        # Save predictions if requested
        if self.save_predictions:
            self.pred_keypoints_3d.append(pred_keypoints_3d_centered.cpu().numpy())
            self.gt_keypoints_3d.append(gt_keypoints_3d_centered.cpu().numpy())
            if pred_vertices is not None:
                self.pred_vertices.append(pred_vertices.cpu().numpy())
            if gt_vertices is not None:
                self.gt_vertices.append(gt_vertices.cpu().numpy())
        
        # Only increment counter by the amount we actually stored
        self.counter += min(batch_size, actual_size if 'mpjpe' in self.metrics else batch_size)
        
        return batch_metrics



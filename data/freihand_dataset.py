# -*- coding: utf-8 -*-
"""
FreiHAND Dataset Loader (RGB-only)
Minimal loader to support UTNet training on FreiHAND_pub_v2.
"""
import json
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .wilor_utils import WILOR_JOINT_MAP


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _project_points(xyz, K):
    xyz = np.asarray(xyz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    uv = (K @ xyz.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


class FreiHANDDataset(Dataset):
    """
    FreiHAND dataset (RGB-only).
    Uses FreiHAND_pub_v2 for training and FreiHAND_pub_v2_eval for evaluation.
    Provides 2D/3D joints and MANO parameters for UTNet training.
    """
    def __init__(self, root_dir, img_size=256, cube_size=280,
                 train=True, color_factor=0.2, color_aug_prob=0.6,
                 align_wilor_aug=False, wilor_aug_config=None,
                 eval_root=None,
                 bbox_source: str = "gt",
                 detector_weights_path: str | None = None):
        """
        Args:
            root_dir: FreiHAND_pub_v2 root directory
            img_size: output image size (square)
            cube_size: cube size in mm for 3D normalization
            train: whether in training mode
            color_factor: color augmentation factor
            color_aug_prob: probability to apply color augmentation
        """
        self.root_dir = root_dir
        self.img_size = int(img_size)
        if isinstance(cube_size, (list, tuple, np.ndarray)):
            self.cube_size = list(cube_size)
        else:
            self.cube_size = [cube_size, cube_size, cube_size]
        self.train = train
        self.split = 'train' if train else 'evaluation'
        self.color_factor = color_factor
        self.color_aug_prob = color_aug_prob
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}
        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        if self.bbox_source == "detector" and (not self.train):
            if detector_weights_path is None:
                raise ValueError("bbox_source='detector' requires detector_weights_path")
            from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector
            self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=detector_weights_path))
        # WiLoR normalization (ImageNet stats * 255)
        self.mean = 255. * np.array([0.485, 0.456, 0.406])
        self.std = 255. * np.array([0.229, 0.224, 0.225])
        self.transform = transforms.ToTensor()

        if eval_root is None:
            default_eval_root = f'{root_dir}_eval'
            eval_root = default_eval_root if osp.isdir(default_eval_root) else None
        self.eval_root = eval_root

        self._load_annotations()
        print(f'Loaded {len(self.xyz_list)} samples from FreiHAND {self.split} split')

    def _load_annotations(self):
        if self.train:
            base_root = self.root_dir
            prefix = 'training'
            self.img_dir = osp.join(base_root, 'training', 'rgb')
        else:
            if self.eval_root is None or not osp.isdir(self.eval_root):
                raise FileNotFoundError(
                    f'FreiHAND eval root not found: {self.eval_root}. '
                    f'Please set eval_root to FreiHAND_pub_v2_eval.'
                )
            base_root = self.eval_root
            prefix = 'evaluation'
            self.img_dir = osp.join(base_root, 'evaluation', 'rgb')

        k_path = osp.join(base_root, f'{prefix}_K.json')
        mano_path = osp.join(base_root, f'{prefix}_mano.json')
        xyz_path = osp.join(base_root, f'{prefix}_xyz.json')

        self.K_list = _load_json(k_path)
        self.mano_list = _load_json(mano_path)
        self.xyz_list = _load_json(xyz_path)

        if not (len(self.K_list) == len(self.mano_list) == len(self.xyz_list)):
            raise ValueError('FreiHAND annotation sizes do not match.')

    def __len__(self):
        return len(self.xyz_list)

    def __getitem__(self, idx):
        # Load RGB (original resolution)
        img_path = osp.join(self.img_dir, f'{idx:08d}.jpg')
        rgb = cv2.imread(img_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f'Fail to read {img_path}')

        H0, W0 = rgb.shape[:2]

        # Camera intrinsics
        K = np.array(self.K_list[idx], dtype=np.float32)
        cam_para = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        # 3D joints in meters -> mm
        joints_xyz = np.array(self.xyz_list[idx], dtype=np.float32) * 1000.0  # (21, 3)
        center_xyz = joints_xyz.mean(0)
        gt3Dcrop = joints_xyz - center_xyz

        # MANO parameters
        mano_params = np.array(self.mano_list[idx][0], dtype=np.float32)
        mano_pose = mano_params[:48]
        mano_shape = mano_params[48:58]
        # FreiHAND mano_trans is in millimeters; convert to meters to match MANO output space.
        mano_trans = mano_params[58:61] / 1000.0

        # 2D projection (original image coordinates)
        joint_uv = _project_points(joints_xyz / 1000.0, K)  # use meters for projection

        # Reorder to WiLoR joint order
        joints_xyz = joints_xyz[WILOR_JOINT_MAP]
        joint_uv = joint_uv[WILOR_JOINT_MAP]

        if self.align_wilor_aug:
            from .wilor_utils import get_bbox, get_example
            # Build keypoints with confidence
            keypoints_2d = np.concatenate([joint_uv, np.ones((21, 1), dtype=np.float32)], axis=1)
            keypoints_3d = np.concatenate([joints_xyz, np.ones((21, 1), dtype=np.float32)], axis=1)
            if self.bbox_source == "detector" and self.detector is not None:
                bbox = self.detector(rgb)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                    scale = np.array([(x2 - x1) * 1.2, (y2 - y1) * 1.2], dtype=np.float32)
                else:
                    center, scale = get_bbox(keypoints_2d, rescale=1.2)
            else:
                center, scale = get_bbox(keypoints_2d, rescale=1.2)
            if scale[0] < 1 or scale[1] < 1:
                center = np.array([W0 / 2.0, H0 / 2.0], dtype=np.float32)
                scale = np.array([W0, H0], dtype=np.float32)
            mano_params = {
                'global_orient': mano_pose[:3].copy(),
                'hand_pose': mano_pose[3:].copy(),
                'betas': mano_shape.copy()
            }
            has_mano_params = {
                'global_orient': np.array([1.0], dtype=np.float32),
                'hand_pose': np.array([1.0], dtype=np.float32),
                'betas': np.array([1.0], dtype=np.float32)
            }
            flip_perm = list(range(21))
            img_patch, keypoints_2d, keypoints_3d, mano_params, _has_params, _, trans = get_example(
                rgb, center[0], center[1], scale[0], scale[1],
                keypoints_2d, keypoints_3d,
                mano_params, has_mano_params,
                flip_perm, self.img_size, self.img_size,
                None, None,
                do_augment=self.train, is_right=True,
                augm_config=self.wilor_aug_config,
                is_bgr=True,
                return_trans=True
            )
            imgRGB = torch.from_numpy(img_patch).float() / 255.0
            joint_img = keypoints_2d.astype(np.float32)
            joints_xyz = keypoints_3d[:, :3].astype(np.float32)
            mano_pose = np.concatenate(
                [mano_params['global_orient'], mano_params['hand_pose']],
                axis=0
            ).astype(np.float32)
            mano_shape = mano_params['betas'].astype(np.float32)
            trans_3x3 = np.eye(3, dtype=np.float32)
            trans_3x3[:2, :] = trans
            K_patch = trans_3x3 @ K
            cam_para = (K_patch[0, 0], K_patch[1, 1], K_patch[0, 2], K_patch[1, 2])
        else:
            # Resize to model input and apply simple color augmentation
            if H0 != self.img_size or W0 != self.img_size:
                rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                sx = self.img_size / float(W0)
                sy = self.img_size / float(H0)
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy
            # Color augmentation
            if self.train and self.color_factor > 0 and random.random() < self.color_aug_prob:
                c_up = 1.0 + self.color_factor
                c_low = 1.0 - self.color_factor
                color_scale = np.array([
                    random.uniform(c_low, c_up),
                    random.uniform(c_low, c_up),
                    random.uniform(c_low, c_up)
                ])
                rgb = np.clip(rgb * color_scale[None, None, :], 0, 255)
            # BGR -> RGB, output [0,1]
            rgb = rgb[:, :, ::-1].copy()
            imgRGB = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            joint_img = np.zeros((21, 3), dtype=np.float32)
            joint_img[:, 0:2] = joint_uv / self.img_size - 0.5
            joint_img[:, 2] = (joints_xyz[:, 2] - center_xyz[2]) / (self.cube_size[2] / 2.0)

        # MANO parameters already extracted above

        data_depth = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            'rgb': imgRGB,  # (3, H, W)
            'depth': data_depth,  # dummy depth
            'n_i': 0,  # no depth
            'has_depth': False,
            'joint_img': torch.from_numpy(joint_img).float(),  # (21, 3)
            'joint_3d': torch.from_numpy((joints_xyz - joints_xyz.mean(0)) / (self.cube_size[2] / 2.0)).float(),  # (21, 3)
            'joints_3d_gt': torch.from_numpy(joints_xyz).float(),  # (21, 3)
            'mano_pose': torch.from_numpy(mano_pose).float(),  # (48,)
            'mano_shape': torch.from_numpy(mano_shape).float(),  # (10,)
            'mano_trans': torch.from_numpy(mano_trans).float(),  # (3,)
            'cam_param': torch.tensor(cam_para, dtype=torch.float32),  # (4,)
            'center': torch.from_numpy(center_xyz).float(),  # (3,)
            'M': torch.eye(3, dtype=torch.float32),  # placeholder
            'cube': torch.tensor(self.cube_size, dtype=torch.float32),  # (3,)
            'hand_type': 'right',
            'is_right': 1.0
        }

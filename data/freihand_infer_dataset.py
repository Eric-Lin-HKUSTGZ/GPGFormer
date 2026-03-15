# -*- coding: utf-8 -*-
"""
FreiHAND Dataset Loader V2 (RGB-only) with Enhanced Data Augmentation
Extended version with additional augmentations:
1. Random center jitter

NOTE: Brightness and contrast adjustments have been REMOVED because get_example()
already applies color augmentation via COLOR_SCALE parameter. Adding additional
brightness/contrast adjustments causes double augmentation and degrades performance.
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

try:
    from .utils import get_example, WILOR_JOINT_MAP
except ImportError:  # pragma: no cover
    from utils import get_example, WILOR_JOINT_MAP


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _project_points(xyz, K):
    xyz = np.asarray(xyz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    uv = (K @ xyz.T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-7)
    return uv


def _sanitize_bbox_xyxy(xyxy, w, h):
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1)
    if xyxy.size != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w - 1)))
    y2 = max(0.0, min(y2, float(h - 1)))
    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_from_detector(detector, rgb, w, h, img_path, expand_factor=1.2):
    bb = detector(rgb)
    bb = _sanitize_bbox_xyxy(bb, w, h) if bb is not None else None
    if bb is None:
        raise RuntimeError(f"Detector failed to produce a valid hand bbox for image: {img_path}")
    x1, y1, x2, y2 = bb.tolist()
    bw = max(x2 - x1, 2.0)
    bh = max(y2 - y1, 2.0)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale_px = np.array([bw * float(expand_factor), bh * float(expand_factor)], dtype=np.float32)
    scale_px = np.maximum(scale_px, 2.0)
    return center, scale_px


class FreiHANDDatasetV2(Dataset):
    """
    FreiHAND dataset V2 (RGB-only) with enhanced data augmentation.

    New augmentations compared to FreiHANDDataset:
    1. Random center jitter during bbox calculation

    NOTE: Brightness/contrast adjustments removed due to double augmentation issue.
    get_example() already applies color augmentation via COLOR_SCALE parameter.
    """
    def __init__(self, root_dir, img_size=256, img_width=192, cube_size=280,
                 train=True, color_factor=0.2, color_aug_prob=0.6,
                 align_wilor_aug=False, wilor_aug_config=None,
                 eval_root=None,
                 bbox_source: str = "gt",
                 detector_weights_path: str | None = None,
                 root_index: int = 0,
                 trainval_ratio: float = 0.9,
                 trainval_seed: int = 42,
                 use_trainval_split: bool = True,
                 # New augmentation parameters
                 center_jitter_factor: float = 0.05,
                 brightness_limit: tuple = (-0.2, 0.1),
                 contrast_limit: tuple = (0.8, 1.2),
                 brightness_prob: float = 0.5,
                 contrast_prob: float = 0.5):
        """
        Args:
            root_dir: FreiHAND_pub_v2 root directory
            img_size: output image size (square)
            cube_size: cube size in mm for 3D normalization
            train: whether in training mode
            color_factor: color augmentation factor
            color_aug_prob: probability to apply color augmentation
            center_jitter_factor: random jitter factor for bbox center (default: 0.05 = 5%)
            brightness_limit: brightness adjustment range (default: (-0.2, 0.1))
            contrast_limit: contrast adjustment range (default: (0.8, 1.2))
            brightness_prob: probability to apply brightness adjustment
            contrast_prob: probability to apply contrast adjustment
        """
        self.root_dir = root_dir
        self.img_size = int(img_size)
        self.patch_height = self.img_size
        self.patch_width = int(img_width)
        self.root_index = int(root_index)
        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.use_trainval_split = bool(use_trainval_split)

        if isinstance(cube_size, (list, tuple, np.ndarray)):
            self.cube_size = list(cube_size)
        else:
            self.cube_size = [cube_size, cube_size, cube_size]

        self.train = train
        self.split = 'train' if train else 'val'
        self.color_factor = color_factor
        self.color_aug_prob = color_aug_prob
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}
        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        self.detector_bbox_expand = 1.2
        if self.bbox_source == "detector":
            if detector_weights_path is None or len(str(detector_weights_path).strip()) == 0:
                raise ValueError("FreiHANDDatasetV2 bbox_source='detector' requires detector_weights_path.")
            from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector

            self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=str(detector_weights_path)))
            if str(os.environ.get("RANK", "0")) == "0":
                print("[info] FreiHANDDatasetV2 uses detector-only hand crop (no label bbox fallback).")
        elif self.bbox_source != "gt":
            raise ValueError(f"Unsupported bbox_source for FreiHANDDatasetV2: {self.bbox_source}")

        # New augmentation parameters
        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        if eval_root is None:
            default_eval_root = f'{root_dir}_eval'
            eval_root = default_eval_root if osp.isdir(default_eval_root) else None
        self.eval_root = eval_root

        self._load_annotations()
        print(f'Loaded {len(self.indices)} images from FreiHAND V2 {self.split} split (total annotations: {len(self.xyz_list)})')

    def _load_annotations(self):
        # When use_trainval_split=False and in evaluation mode (train=False),
        # use eval_root if provided; otherwise use training data
        if not self.use_trainval_split and not self.train and self.eval_root is not None:
            # Load from evaluation set
            base_root = self.eval_root
            prefix = 'evaluation'
            self.img_dir = osp.join(base_root, 'evaluation', 'rgb')
        else:
            # Load from training set (default behavior)
            base_root = self.root_dir
            prefix = 'training'
            self.img_dir = osp.join(base_root, 'training', 'rgb')

        k_path = osp.join(base_root, f'{prefix}_K.json')
        mano_path = osp.join(base_root, f'{prefix}_mano.json')
        xyz_path = osp.join(base_root, f'{prefix}_xyz.json')

        self.K_list = _load_json(k_path)
        self.mano_list = _load_json(mano_path)
        self.xyz_list = _load_json(xyz_path)
        verts_path = osp.join(base_root, f'{prefix}_verts.json')
        self.verts_list = _load_json(verts_path) if osp.isfile(verts_path) else None

        if not (len(self.K_list) == len(self.mano_list) == len(self.xyz_list)):
            raise ValueError('FreiHAND annotation sizes do not match.')
        if self.verts_list is not None and len(self.verts_list) != len(self.xyz_list):
            raise ValueError('FreiHAND vertices size does not match xyz size.')

        # Count actual images in the directory
        num_images = len([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

        # Train/val split based on actual image count
        indices = list(range(num_images))
        if self.use_trainval_split:
            rng = random.Random(self.trainval_seed)
            rng.shuffle(indices)
            n_train = int(num_images * self.trainval_ratio)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            if self.train:
                indices = train_idx
            else:
                indices = val_idx
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Load RGB (original resolution)
        real_idx = self.indices[idx]
        img_path = osp.join(self.img_dir, f'{real_idx:08d}.jpg')
        rgb = cv2.imread(img_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f'Fail to read {img_path}')

        h, w = rgb.shape[:2]

        # Map image index to annotation index
        anno_idx = real_idx % len(self.K_list)

        # Camera intrinsics
        K = np.array(self.K_list[anno_idx], dtype=np.float32)
        cam_para = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        keypoints_3d = np.array(self.xyz_list[anno_idx], dtype=np.float32)

        # MANO parameters
        mano_params = np.array(self.mano_list[anno_idx][0], dtype=np.float32)
        global_orient = mano_params[:3]
        hand_pose = mano_params[3:48]
        hand_shape = mano_params[48:58]

        # 2D projection
        keypoints_2d = _project_points(keypoints_3d, K)

        # Calculate valid keypoints for bbox computation
        uv_norm = keypoints_2d.copy()
        uv_norm[:, 0] /= w
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32")
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        if self.bbox_source == "detector":
            try:
                center, scale_px = _bbox_from_detector(
                    self.detector, rgb, w, h, img_path, expand_factor=self.detector_bbox_expand
                )
            except RuntimeError:
                # Fallback to ground truth bbox when detector fails
                valid_points = [keypoints_2d[i] for i in range(len(keypoints_2d)) if coord_valid[i] == 1]
                if len(valid_points) >= 2:
                    points = np.asarray(valid_points, dtype=np.float32)
                    min_coord = points.min(axis=0)
                    max_coord = points.max(axis=0)
                    center = (max_coord + min_coord) / 2.0
                    scale_px = 2.0 * (max_coord - min_coord)
                else:
                    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
                    scale_px = np.array([float(max(w, h)), float(max(w, h))], dtype=np.float32)
                scale_px = np.maximum(scale_px, 2.0)
        else:
            valid_points = [keypoints_2d[i] for i in range(len(keypoints_2d)) if coord_valid[i] == 1]
            if len(valid_points) >= 2:
                points = np.asarray(valid_points, dtype=np.float32)
                min_coord = points.min(axis=0)
                max_coord = points.max(axis=0)
                center = (max_coord + min_coord) / 2.0
                scale_px = 2.0 * (max_coord - min_coord)
            else:
                center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
                scale_px = np.array([float(max(w, h)), float(max(w, h))], dtype=np.float32)
            scale_px = np.maximum(scale_px, 2.0)

        scale = (scale_px / 200.0).astype(np.float32)
        bbox_size = float(scale_px.max())
        denom = float(max(scale_px.max(), 1e-6))
        bbox_expand_factor = float(bbox_size / denom)

        # NEW AUGMENTATION 1: Random center jitter (only during training)
        # if self.train and self.center_jitter_factor > 0:
        #     jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
        #     jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
        #     center[0] += jitter_x
        #     center[1] += jitter_y

        # Prepare MANO parameters
        mano_params = {
            'global_orient': global_orient.copy(),
            'hand_pose': hand_pose.copy(),
            'betas': hand_shape.copy()
        }
        mano_params_is_axis_angle = {
            'global_orient': True,
            'hand_pose': True,
            'betas': False
        }
        has_mano_params = {
            'global_orient': np.array([1.0], dtype=np.float32),
            'hand_pose': np.array([1.0], dtype=np.float32),
            'betas': np.array([1.0], dtype=np.float32)
        }
        flip_perm = list(range(21))

        # Apply WiLoR-style augmentation and cropping
        img_patch, keypoints_2d, keypoints_3d, mano_params, _has_params, _, trans = get_example(
            rgb, center[0], center[1], bbox_size, bbox_size,
            keypoints_2d, keypoints_3d,
            mano_params, has_mano_params,
            flip_perm, self.patch_width, self.patch_height,
            None, None,
            do_augment=self.train, is_right=True,
            augm_config=self.wilor_aug_config,
            is_bgr=True,
            return_trans=True
        )

        # Convert to float [0, 1]
        imgRGB_01 = torch.from_numpy(img_patch).float() / 255.0

        # NEW AUGMENTATION 2 & 3: Brightness and Contrast adjustment (only during training)
        if self.train:
            # Apply brightness adjustment
            if np.random.rand() < self.brightness_prob:
                brightness_delta = np.random.uniform(self.brightness_limit[0], self.brightness_limit[1])
                imgRGB_01 = torch.clamp(imgRGB_01 + brightness_delta, 0.0, 1.0)

            # Apply contrast adjustment
            if np.random.rand() < self.contrast_prob:
                contrast_factor = np.random.uniform(self.contrast_limit[0], self.contrast_limit[1])
                mean_val = imgRGB_01.mean(dim=[1, 2], keepdim=True)
                imgRGB_01 = torch.clamp((imgRGB_01 - mean_val) * contrast_factor + mean_val, 0.0, 1.0)

        # Apply ImageNet normalization
        imgRGB = (imgRGB_01 - self.imagenet_mean) / self.imagenet_std

        # Update camera intrinsics after affine transformation
        trans_3x3 = np.eye(3, dtype=np.float32)
        trans_3x3[:2, :] = trans
        K_patch = trans_3x3 @ K
        cam_para = (K_patch[0, 0], K_patch[1, 1], K_patch[0, 2], K_patch[1, 2])

        # Calculate validity flags
        trans_uv01 = keypoints_2d.copy()
        trans_uv01[:, 0] = trans_uv01[:, 0] + 0.5
        trans_uv01[:, 1] = trans_uv01[:, 1] + 0.5

        trans_coord_valid = (trans_uv01 > 0).astype("float32") * (trans_uv01 < 1).astype("float32")
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid

        xyz = keypoints_3d.copy()
        xyz_valid = 1

        if trans_coord_valid[self.root_index] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0

        vertices_gt = None
        if self.verts_list is not None:
            vertices_gt = np.array(self.verts_list[anno_idx], dtype=np.float32)

        out = {
            'rgb': imgRGB,
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
            'keypoints_3d': torch.from_numpy(keypoints_3d.astype(np.float32)).float(),
            'mano_params': mano_params,
            'cam_param': torch.tensor(cam_para, dtype=torch.float32),
            'box_center': torch.from_numpy(center.astype(np.float32)),
            'box_size': torch.tensor(bbox_size, dtype=torch.float32),
            'bbox_expand_factor': torch.tensor(bbox_expand_factor, dtype=torch.float32),
            '_scale': torch.from_numpy(scale.astype(np.float32)),
            'mano_params_is_axis_angle': mano_params_is_axis_angle,
            'xyz_valid': xyz_valid,
            'uv_valid': trans_coord_valid,
            'hand_type': 'right',
            'is_right': 1.0
        }
        if vertices_gt is not None:
            out['vertices_gt'] = torch.from_numpy(vertices_gt.astype(np.float32)).float()
        return out


def main():
    """Debug/inspect FreiHANDDatasetV2 by printing a few samples."""
    import argparse
    from pprint import pformat

    def _describe_value(x, max_list_items: int = 8, max_str_len: int = 200) -> str:
        if isinstance(x, torch.Tensor):
            x_det = x.detach()
            desc = f"torch.Tensor(shape={tuple(x_det.shape)}, dtype={x_det.dtype}, device={x_det.device})"
            if x_det.numel() > 0 and x_det.is_floating_point():
                desc += f", min={x_det.min().item():.4g}, max={x_det.max().item():.4g}"
            return desc
        if isinstance(x, np.ndarray):
            desc = f"np.ndarray(shape={x.shape}, dtype={x.dtype})"
            if x.size > 0 and np.issubdtype(x.dtype, np.floating):
                desc += f", min={np.nanmin(x):.4g}, max={np.nanmax(x):.4g}"
            return desc
        if isinstance(x, (int, float, bool, np.number)):
            return f"{type(x).__name__}({x})"
        if isinstance(x, str):
            s = x if len(x) <= max_str_len else x[:max_str_len] + "..."
            return f"str(len={len(x)}): {s!r}"
        if isinstance(x, (list, tuple)):
            head = list(x[:max_list_items])
            more = "" if len(x) <= max_list_items else f", ... (+{len(x)-max_list_items} more)"
            return f"{type(x).__name__}(len={len(x)}): {head!r}{more}"
        if isinstance(x, dict):
            return f"dict(keys={list(x.keys())})"
        return f"{type(x).__name__}"

    parser = argparse.ArgumentParser(description="Print/inspect FreiHANDDatasetV2 contents.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to FreiHAND_pub_v2 root.")
    parser.add_argument("--eval-root", type=str, default=None, help="Path to FreiHAND_pub_v2_eval.")
    parser.add_argument("--train", action="store_true", help="Use training split.")
    parser.add_argument("--eval", dest="train", action="store_false", help="Use evaluation split.")
    parser.set_defaults(train=True)
    parser.add_argument("--img-size", type=int, default=256, help="Output image size.")
    parser.add_argument("--center-jitter", type=float, default=0.05, help="Center jitter factor.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    ds = FreiHANDDatasetV2(
        root_dir=args.root_dir,
        img_size=args.img_size,
        train=args.train,
        eval_root=args.eval_root,
        center_jitter_factor=args.center_jitter,
    )

    print("\n=== Dataset V2 summary ===")
    print(f"split={ds.split}, len={len(ds)}")
    print(f"img_dir={ds.img_dir}")
    print(f"center_jitter_factor={ds.center_jitter_factor}")
    print("=======================\n")

    random.seed(args.seed)
    n = min(args.num_samples, len(ds))
    indices = random.sample(range(len(ds)), k=n) if n > 0 else []

    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1}/{len(indices)} | idx={idx} ---")
        sample = ds[idx]
        for k in sorted(sample.keys()):
            v = sample[k]
            print(f"{k}: {_describe_value(v)}")


if __name__ == "__main__":
    main()

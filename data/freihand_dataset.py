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
# try:
#     # When imported as a package module (recommended): `python -m data.freihand_dataset ...`
#     from .wilor_utils import WILOR_JOINT_MAP
# except ImportError:  # pragma: no cover
#     # When executed as a script: `python data/freihand_dataset.py ...`
#     from wilor_utils import WILOR_JOINT_MAP
try:
    from .utils import  get_example, WILOR_JOINT_MAP
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


class FreiHANDDataset(Dataset):
    """
    FreiHAND dataset (RGB-only).
    Uses FreiHAND_pub_v2 for both training and validation (train/val split).
    Provides 2D/3D joints and MANO parameters for UTNet training.
    """
    def __init__(self, root_dir, img_size=256, img_width=192, cube_size=280,
                 train=True, color_factor=0.2, color_aug_prob=0.6,
                 align_wilor_aug=False, wilor_aug_config=None,
                 eval_root=None,
                 bbox_source: str = "gt",
                 detector_weights_path: str | None = None,
                 root_index: int = 9,
                 trainval_ratio: float = 0.9,
                 trainval_seed: int = 42,
                 use_trainval_split: bool = True):
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
        # Backbone(ViT-Large) expects 256x192
        self.patch_height = self.img_size
        self.patch_width = int(img_width) # 192
        self.root_index = int(root_index)
        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.use_trainval_split = bool(use_trainval_split)
        if isinstance(cube_size, (list, tuple, np.ndarray)):
            self.cube_size = list(cube_size)
        else:
            self.cube_size = [cube_size, cube_size, cube_size]
        self.train = train
        # default: use training split for both train/val by slicing
        self.split = 'train' if train else 'val'
        self.color_factor = color_factor
        self.color_aug_prob = color_aug_prob
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}
        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        # if self.bbox_source == "detector" and (not self.train):
        #     if detector_weights_path is None:
        #         raise ValueError("bbox_source='detector' requires detector_weights_path")
        #     from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector
        #     self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=detector_weights_path))
        # standard ImageNet normalization (ImageNet stats * 255)
        # self.mean = 255. * np.array([0.485, 0.456, 0.406])
        # self.std = 255. * np.array([0.229, 0.224, 0.225])
        self.transform = transforms.ToTensor()
        # ImageNet normalization (for ViT backbones / WiLoR-style inputs)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        if eval_root is None:
            default_eval_root = f'{root_dir}_eval'
            eval_root = default_eval_root if osp.isdir(default_eval_root) else None
        self.eval_root = eval_root

        self._load_annotations()
        print(f'Loaded {len(self.indices)} images from FreiHAND {self.split} split (total annotations: {len(self.xyz_list)})')

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

        if not (len(self.K_list) == len(self.mano_list) == len(self.xyz_list)):
            raise ValueError('FreiHAND annotation sizes do not match.')

        # Count actual images in the directory
        # FreiHAND has 4 background variants per annotation, so total images = 4 * num_annotations
        
        num_images = len([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

        # Train/val split based on actual image count (not annotation count)
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

        # Map image index to annotation index (annotations cycle every 32560 images)
        anno_idx = real_idx % len(self.K_list)

        # Camera intrinsics
        K = np.array(self.K_list[anno_idx], dtype=np.float32)
        cam_para = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        keypoints_3d = np.array(self.xyz_list[anno_idx], dtype=np.float32)  # (21, 3)
        # center_xyz = keypoints_3d[9] # 取中指根关节作为中心点

        # MANO parameters
        mano_params = np.array(self.mano_list[anno_idx][0], dtype=np.float32)
        global_orient = mano_params[:3]
        hand_pose = mano_params[3:48]
        hand_shape = mano_params[48:58]
       

        # 2D projection (original image coordinates)
        keypoints_2d = _project_points(keypoints_3d, K) 
        
        # NOTE:
        # FreiHAND `*_xyz.json` already uses the OpenPose-hand / WiLoR 21-joint order:
        #   wrist, thumb(4), index(4), middle(4), ring(4), pinky(4).
        # Our MANO wrapper (`third_party/wilor_min/.../mano_wrapper.py`) also outputs joints in this order.
        # Therefore we must NOT reorder joints here; otherwise GT/pred semantics mismatch and metrics stall.
        
        # Build keypoints with confidence，为啥要有这东西？我看其它方法都没有
        # keypoints_2d = np.concatenate([keypoints_2d, np.ones((21, 1), dtype=np.float32)], axis=1)
        # keypoints_3d = np.concatenate([keypoints_3d, np.ones((21, 1), dtype=np.float32)], axis=1)
        # 通过keypoint2d中的有效点计算hand crop所需的center和scale

        uv_norm = keypoints_2d.copy()
        uv_norm[:, 0] /= w   
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        valid_points = [keypoints_2d[i] for i in range(len(keypoints_2d)) if coord_valid[i]==1]
        points = np.array(valid_points)
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord)/2.0
        scale = 2*(max_coord - min_coord)/200.0

        bbox_size = float((scale * 200.0).max())
        bbox_expand_factor = float(bbox_size / (scale * 200.0).max())

        # print("bbox_size: ", bbox_size)
        # print("bbox_expand_factor: ", bbox_expand_factor)
        
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
        # is_right=True说明默认都是右手，这对吗？
        # 应该加个判断，如果标签中没有right的关键字，则默认是右手
        # 但FreiHand数据集没有right标签，所以可以不用管
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
        # Convert to float and apply ImageNet normalization (WiLoR/HaMeR convention).
        # NOTE: img_patch is CHW in [0,255] (uint8/float). Normalize to [0,1] then standardize.
        imgRGB_01 = torch.from_numpy(img_patch).float() / 255.0
        imgRGB = (imgRGB_01 - self.imagenet_mean) / self.imagenet_std
        
        # # 对3D关键点进行root-relative转换（由于3D Loss已经做了root-relative，这里不需要了）
        # root_point = keypoints_3d[self.root_index].copy()
        # keypoints_3d = keypoints_3d - root_point[None, :]

        # mano_pose = np.concatenate(
        #     [mano_params['global_orient'], mano_params['hand_pose']],
        #     axis=0
        # ).astype(np.float32)
        # mano_shape = mano_params['betas'].astype(np.float32)
        # 应用仿射变换时，K也要对应变换
        trans_3x3 = np.eye(3, dtype=np.float32)
        trans_3x3[:2, :] = trans
        K_patch = trans_3x3 @ K
        # 这里没有构建标准的3x3相机内参矩阵K
        cam_para = (K_patch[0, 0], K_patch[1, 1], K_patch[0, 2], K_patch[1, 2]) 


        # NOTE:
        # `get_example()` already normalizes keypoints_2d to [-0.5, 0.5] using patch size:
        #   u_norm = u_px / patch_width  - 0.5
        #   v_norm = v_px / patch_height - 0.5
        # So for validity in patch frame, convert back to [0,1] first.
        trans_uv01 = keypoints_2d.copy()
        trans_uv01[:, 0] = trans_uv01[:, 0] + 0.5
        trans_uv01[:, 1] = trans_uv01[:, 1] + 0.5

        trans_coord_valid = (trans_uv01 > 0).astype("float32") * (trans_uv01 < 1).astype("float32") # Nx2x21x2
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid

        xyz = keypoints_3d.copy()
        xyz_valid = 1

        if trans_coord_valid[self.root_index] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0

        return {
            'rgb': imgRGB,  # (3, H, W)
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),  # (21, 2)
            'keypoints_3d': torch.from_numpy(keypoints_3d.astype(np.float32)).float(),  # (21, 3) meters
            # 'joints_3d_gt': torch.from_numpy(keypoints_3d.astype(np.float32)).float(),  # alias (meters)
            # 'mano_pose': torch.from_numpy(mano_pose).float(),  # (3+48=51,)
            # 'mano_shape': torch.from_numpy(mano_shape).float(),  # (10,)
            'mano_params': mano_params,  # (3+48+10=61,)
            'cam_param': torch.tensor(cam_para, dtype=torch.float32),  # (4,)
            # 'cube': torch.tensor(self.cube_size, dtype=torch.float32),  # (3,)
            'box_center': torch.from_numpy(center.astype(np.float32)),
            'box_size': torch.tensor(bbox_size, dtype=torch.float32),
            'bbox_expand_factor': torch.tensor(bbox_expand_factor, dtype=torch.float32),
            '_scale': torch.from_numpy(scale.astype(np.float32)),
            'mano_params_is_axis_angle': mano_params_is_axis_angle,
            'xyz_valid': xyz_valid, # 最终的3D关键点有效性标志
            'uv_valid': trans_coord_valid, # 最终的2D关键点有效性标志
            'hand_type': 'right',
            'is_right': 1.0
        }


def main():
    """Debug/inspect FreiHANDDataset by printing a few samples."""
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

    parser = argparse.ArgumentParser(description="Print/inspect FreiHANDDataset contents.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to FreiHAND_pub_v2 root.")
    parser.add_argument("--eval-root", type=str, default=None, help="Path to FreiHAND_pub_v2_eval (for evaluation split).")
    parser.add_argument("--train", action="store_true", help="Use training split.")
    parser.add_argument("--eval", dest="train", action="store_false", help="Use evaluation split.")
    parser.set_defaults(train=True)
    parser.add_argument("--img-size", type=int, default=256, help="Output image size (square).")
    parser.add_argument("--cube-size", type=float, default=280, help="Cube size in mm for 3D normalization.")
    parser.add_argument("--align-wilor-aug", action="store_true", help="Use WiLoR-style crop/aug pipeline.")
    parser.add_argument("--bbox-source", type=str, default="gt", choices=["gt", "detector"], help="BBox source when align_wilor_aug is on.")
    parser.add_argument("--detector-weights-path", type=str, default=None, help="Detector weights (eval + bbox_source=detector).")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    parser.add_argument("--index", type=int, default=None, help="Inspect a specific index (overrides random sampling).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling indices.")
    parser.add_argument("--print-values", action="store_true", help="Also print small value previews for tensors/arrays.")
    args = parser.parse_args()

    ds = FreiHANDDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        cube_size=args.cube_size,
        train=args.train,
        align_wilor_aug=args.align_wilor_aug,
        wilor_aug_config={},
        eval_root=args.eval_root,
        bbox_source=args.bbox_source,
        detector_weights_path=args.detector_weights_path,
    )

    print("\n=== Dataset summary ===")
    print(f"split={ds.split}, len={len(ds)}")
    print(f"img_dir={ds.img_dir}")
    print(f"img_size={ds.img_size}, cube_size={ds.cube_size}")
    print(f"align_wilor_aug={ds.align_wilor_aug}, bbox_source={ds.bbox_source}")
    print("=======================\n")

    random.seed(args.seed)
    if args.index is not None:
        indices = [int(args.index)]
    else:
        n = min(args.num_samples, len(ds))
        indices = random.sample(range(len(ds)), k=n) if n > 0 else []

    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1}/{len(indices)} | idx={idx} ---")
        sample = ds[idx]
        if not isinstance(sample, dict):
            print(f"Unexpected sample type: {type(sample)}")
            print(sample)
            continue

        for k in sorted(sample.keys()):
            v = sample[k]
            print(f"{k}: {_describe_value(v)}")
            if args.print_values:
                if isinstance(v, torch.Tensor):
                    flat = v.detach().flatten()
                    head = flat[:16].cpu().numpy()
                    print(f"  values(head): {head}")
                elif isinstance(v, np.ndarray):
                    head = v.reshape(-1)[:16]
                    print(f"  values(head): {head}")
                else:
                    print(f"  value: {pformat(v)[:400]}")


if __name__ == "__main__":
    main()
    """
    python data/freihand_dataset.py --root-dir /path/to/FreiHAND_pub_v2 --num-samples 3
    python data/freihand_dataset.py --root-dir /path/to/FreiHAND_pub_v2 --eval --eval-root /path/to/FreiHAND_pub_v2_eval --num-samples 3
    python data/freihand_dataset.py --root-dir /path/to/FreiHAND_pub_v2 --num-samples 2 --print-values
    python data/freihand_dataset.py --root-dir /path/to/FreiHAND_pub_v2 --index 127 --print-values
    """


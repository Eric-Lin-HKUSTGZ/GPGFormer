# -*- coding: utf-8 -*-
"""
HO3D Dataset Loader with Random Modality Sampling
参考 KeypointFusion/dataloader/loader.py HO3D class
支持HO3D_v3格式：直接从pickle文件加载标注
"""
import os
import os.path as osp
import copy
import random
import math
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy import ndimage
from .utils import get_example

# HO3D to MANO joint mapping
# HO3D2MANO = [0,
#              1, 2, 3,
#              4, 5, 6,
#              7, 8, 9,
#              10, 11, 12,
#              13, 14, 15,
#              17,
#              18,
#              20,
#              19,
#              16]


# def transformPoint2D(pt, M):
#     """Transform point in 2D coordinates"""
#     pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
#     return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


# def transformPoints2D(pts, M):
#     """Transform points in 2D coordinates"""
#     ret = pts.copy()
#     for i in range(pts.shape[0]):
#         ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
#     return ret


# def rotatePoint2D(p1, center, angle):
#     """Rotate a point in 2D around center"""
#     alpha = angle * np.pi / 180.
#     pp = p1.copy()
#     pp[0:2] -= center[0:2]
#     pr = np.zeros_like(pp)
#     pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
#     pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
#     pr[2] = pp[2]
#     ps = pr
#     ps[0:2] += center[0:2]
#     return ps


class HO3DDataset(Dataset):
    """
    HO3D Dataset with random modality sampling
    Supports HO3D_v3 format
    """
    def __init__(
                 self,
                 data_split,
                 root_dir,
                 dataset_version='v3',
                 img_size=256,
                 aug_para=[10, 0.2, 180], cube_size=[280, 280, 280],
                 input_modal='RGBD', color_factor=0.2, p_drop=0.4, train=True,
                 aug_prob=0.8, color_aug_prob=0.6, align_wilor_aug=False,
                 wilor_aug_config=None,
                 bbox_source: str = "gt",
                 detector_weights_path: str | None = None,
                 # ---- train/val split carved from train.txt (for HO3D v3 where evaluation split lacks GT) ----
                 trainval_ratio: float = 0.9,
                 trainval_seed: int = 42,
                 trainval_split_by: str = "sequence",  # "sequence" | "frame"
                 root_index: int = 9,
                 # New augmentation parameters (align with FreiHANDDatasetV2)
                 center_jitter_factor: float = 0.05,
                 brightness_limit: tuple = (-0.2, 0.1),
                 contrast_limit: tuple = (0.8, 1.2),
                 brightness_prob: float = 0.5,
                 contrast_prob: float = 0.5):
        """
        Args:
            data_split: 'train', 'val', 'test', 'evaluation', or 'train_all'
            root_dir: root directory containing HO3D_v3 folder
            dataset_version: 'v3' for HO3D_v3
            img_size: output image size
            aug_para: [sigma_com, sigma_sc, rot_range] for augmentation
            cube_size: [x, y, z] cube size for cropping
            input_modal: 'RGBD' or 'RGB'
            color_factor: color augmentation factor
            p_drop: probability to drop depth during training (OmniVGGT style)
            train: whether in training mode
            aug_prob: probability to apply geometric augmentation (rotation, translation, scale)
            color_aug_prob: probability to apply color augmentation
        """
        self.data_split = data_split
        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.trainval_split_by = str(trainval_split_by).lower()
        self.dataset_version = dataset_version
        # Allow passing either the dataset parent dir or the HO3D_v3 dir itself
        candidate_root = osp.join(root_dir, 'HO3D_%s' % (dataset_version))
        if osp.exists(candidate_root):
            self.root_dir = candidate_root
        else:
            self.root_dir = root_dir
        self.root_joint_idx = int(root_index)
        self.color_factor = color_factor
        self.input_modal = "RGB"
        self.img_size = img_size
        # WiLoR ViT expects 256x192; keep height=img_size, width=0.75*img_size
        self.patch_height = self.img_size
        self.patch_width = int(round(self.img_size * 0.75))
        self.aug_para = aug_para
        self.cube_size = cube_size
        self.aug_modes = ['rot', 'com', 'sc', 'none']
        self.flip = 1
        self.p_drop = p_drop
        self.train = train
        self.aug_prob = aug_prob  # Probability to apply geometric augmentation
        self.color_aug_prob = color_aug_prob  # Probability to apply color augmentation
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}
        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        if self.bbox_source == "detector" and (not self.train):
            if detector_weights_path is None:
                raise ValueError("bbox_source='detector' requires detector_weights_path")
            from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector
            self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=detector_weights_path))

        # New augmentation parameters (align with FreiHANDDatasetV2)
        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        # WiLoR uses ImageNet normalization (align with FreiHANDDatasetV2)
        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        
        self.dataset_len = 0
        self.datalist = self.load_data()
        print(f'HO3D Dataset loaded: {self.dataset_len} samples from {data_split} split')

    def _read_split_lines(self, split_file: str):
        lines = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
        return lines

    def _split_train_lines(self, train_lines: list[str]):
        """
        Deterministically split train.txt into train/val subsets.
        - split_by="sequence": group by sequence to avoid leakage, then pack sequences to hit ratio by #frames.
        - split_by="frame": shuffle individual frames and take first ratio as train.
        """
        ratio = float(self.trainval_ratio)
        ratio = max(0.0, min(1.0, ratio))
        seed = int(self.trainval_seed)

        # Edge cases
        if len(train_lines) == 0:
            return [], []
        if ratio <= 0.0:
            return [], list(train_lines)
        if ratio >= 1.0:
            return list(train_lines), []

        split_by = str(self.trainval_split_by).lower()
        if split_by not in ("sequence", "frame"):
            split_by = "sequence"

        if split_by == "frame":
            rng = random.Random(seed)
            lines = list(train_lines)
            rng.shuffle(lines)
            cut = int(len(lines) * ratio)
            return lines[:cut], lines[cut:]

        # split_by == "sequence"
        seq_to_lines: dict[str, list[str]] = {}
        for ln in train_lines:
            parts = ln.split('/')
            if len(parts) != 2:
                continue
            seq = parts[0]
            seq_to_lines.setdefault(seq, []).append(ln)

        seqs = list(seq_to_lines.keys())
        rng = random.Random(seed)
        rng.shuffle(seqs)

        total = sum(len(v) for v in seq_to_lines.values())
        target_train = int(total * ratio)

        train_out: list[str] = []
        val_out: list[str] = []
        count = 0
        for seq in seqs:
            bucket = seq_to_lines[seq]
            if count < target_train:
                train_out.extend(bucket)
                count += len(bucket)
            else:
                val_out.extend(bucket)

        # Ensure both splits non-empty if possible
        if len(train_out) == 0 and len(val_out) > 0:
            train_out.append(val_out.pop(0))
        if len(val_out) == 0 and len(train_out) > 1:
            val_out.append(train_out.pop())
        return train_out, val_out

    def load_data(self):
        """Load dataset annotations from pickle files (HO3D_v3 format)"""
        # HO3D_v3 has train.txt and evaluation.txt (no test.txt).
        # We additionally support a deterministic "val" split carved out of train.txt (90/10 by default).
        split_name = str(self.data_split).lower()
        if split_name == 'test':
            # Keep backward compatibility: "test" maps to the official evaluation list.
            split_name = 'evaluation'

        train_txt = osp.join(self.root_dir, 'train.txt')
        eval_txt = osp.join(self.root_dir, 'evaluation.txt')

        selected_lines: list[str] = []
        if split_name in ('train', 'val', 'train_all'):
            if not osp.exists(train_txt):
                raise FileNotFoundError(f"Split file not found: {train_txt}")
            all_train_lines = self._read_split_lines(train_txt)
            train_lines, val_lines = self._split_train_lines(all_train_lines)
            if split_name == 'train':
                selected_lines = train_lines
            elif split_name == 'val':
                selected_lines = val_lines
            else:
                selected_lines = all_train_lines
            split_folder = 'train'
        elif split_name == 'evaluation':
            if not osp.exists(eval_txt):
                raise FileNotFoundError(f"Split file not found: {eval_txt}")
            selected_lines = self._read_split_lines(eval_txt)
            split_folder = 'evaluation'
        else:
            raise ValueError(f"Unsupported HO3D split: {self.data_split}")
        
        datalist = []
        for line in selected_lines:
            # Format: <sequence_name>/<file_id>
            # e.g., "MC1/0000"
            parts = line.split('/')
            if len(parts) != 2:
                continue
            seq_name, file_id = parts[0], parts[1]
                
            # Build paths
            seq_dir = osp.join(self.root_dir, split_folder, seq_name)
            rgb_path = osp.join(seq_dir, 'rgb', f'{file_id}.jpg')
            # Try png if jpg doesn't exist
            if not osp.exists(rgb_path):
                rgb_path = osp.join(seq_dir, 'rgb', f'{file_id}.png')
            
            meta_path = osp.join(seq_dir, 'meta', f'{file_id}.pkl')
            
            # Check if files exist
            if not osp.exists(rgb_path):
                continue
            if not osp.exists(meta_path):
                continue
            
            # Load pickle file
            try:
                with open(meta_path, 'rb') as pkl_file:
                    meta_data = pickle.load(pkl_file)
            except Exception as e:
                print(f'Warning: Failed to load {meta_path}: {e}')
                continue
                
            # Check if annotation exists
            # For evaluation split: handPose, handBeta, handTrans are not available
            # Only handJoints3D (root joint, 3x1) and handBoundingBox are available
            is_evaluation = (split_name == 'evaluation')
            
            if is_evaluation:
                # Evaluation data: check for handJoints3D (root joint) or handBoundingBox
                root_joint_3d = meta_data.get('handJoints3D')
                hand_bbox = meta_data.get('handBoundingBox')
                if root_joint_3d is None and hand_bbox is None:
                    continue
                # Ensure root_joint_3d is a valid array
                if root_joint_3d is not None:
                    root_joint_3d = np.array(root_joint_3d, dtype=np.float32)
                    if root_joint_3d.shape != (3,):
                        # If it's not (3,), try to reshape or skip
                        if root_joint_3d.size == 3:
                            root_joint_3d = root_joint_3d.reshape(3)
                        else:
                            continue
            else:
                # Train/val data: require full annotations
                if meta_data.get('handPose') is None or meta_data.get('handJoints3D') is None:
                    continue
                
                cam_mat = np.array(meta_data['camMat'], dtype=np.float32)  # (3, 3) intrinsic matrix
                
                # Extract data based on split type
                if is_evaluation:
                    # Evaluation: use dummy MANO parameters (will not be used for training)
                    hand_pose = np.zeros(48, dtype=np.float32)  # (48,)
                    hand_beta = np.zeros(10, dtype=np.float32)  # (10,)
                    hand_trans = np.zeros(3, dtype=np.float32)  # (3,)
                    
                    # handJoints3D in evaluation is root joint (3,), not full joints (21, 3)
                    # root_joint_3d was already extracted and validated above
                    if root_joint_3d is None:
                        root_joint_3d = np.array([0, 0, 0], dtype=np.float32)
                    # Ensure it's shape (3,)
                    if root_joint_3d.shape != (3,):
                        if root_joint_3d.size == 3:
                            root_joint_3d = root_joint_3d.reshape(3)
                        else:
                            root_joint_3d = np.array([0, 0, 0], dtype=np.float32)
                    
                    # Create dummy 21 joints using root joint (for compatibility)
                    # In evaluation, we don't have full joint annotations, so use root joint for all
                    hand_joints_3d = np.tile(root_joint_3d.reshape(1, 3), (21, 1)).astype(np.float32)  # (21, 3)
                else:
                    # Training: use full annotations
                    hand_pose = np.array(meta_data['handPose'], dtype=np.float32)  # (48,)
                    hand_beta = np.array(meta_data['handBeta'], dtype=np.float32)  # (10,)
                    hand_trans = np.array(meta_data['handTrans'], dtype=np.float32)  # (3,)
                    hand_joints_3d = np.array(meta_data['handJoints3D'], dtype=np.float32)  # (21, 3) in meters
                
                # Read image to get shape
                try:
                    img = cv2.imread(rgb_path)
                    if img is None:
                        continue
                    img_shape = (img.shape[0], img.shape[1])  # (height, width)
                except Exception as e:
                    print(f'Warning: Failed to read {rgb_path}: {e}')
                    continue
                
                # Convert camera matrix to focal and principal point format
                # camMat is 3x3 intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                fx = cam_mat[0, 0]
                fy = cam_mat[1, 1]
                fu = cam_mat[0, 2]  # principal point x
                fv = cam_mat[1, 2]  # principal point y
                
                cam_param = {
                    'focal': np.array([fx, fy], dtype=np.float32),
                    'princpt': np.array([fu, fv], dtype=np.float32)
                }
                
                # Convert joints from meters to meters (already in meters)
                joints_coord_cam = hand_joints_3d  # (21, 3) in meters
                
                if is_evaluation:
                    # For evaluation: use handBoundingBox if available, otherwise project root joint
                    hand_bbox = meta_data.get('handBoundingBox')
                    if hand_bbox is not None:
                        # handBoundingBox: [topLeftX, topLeftY, bottomRightX, bottomRightY]
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = hand_bbox
                        center_2d = [(bbox_x1 + bbox_x2) / 2.0, (bbox_y1 + bbox_y2) / 2.0]
                        # Project root joint to get depth
                        root_joint_2d = self.joint3DToImg(root_joint_3d.reshape(1, 3), (fx, fy, fu, fv))
                        center_2d = [center_2d[0], center_2d[1], root_joint_2d[0, 2]]  # Use bbox center + root depth
                    else:
                        # Fallback: project root joint
                        root_joint_2d = self.joint3DToImg(root_joint_3d.reshape(1, 3), (fx, fy, fu, fv))
                        center_2d = root_joint_2d[0].tolist()
                    
                    # Create dummy joints_coord_img for compatibility (only root joint is valid)
                    joints_coord_img = np.zeros((21, 3), dtype=np.float32)
                    joints_coord_img[0] = self.joint3DToImg(root_joint_3d.reshape(1, 3), (fx, fy, fu, fv))[0]
                    
                    # Use bounding box if available
                    if hand_bbox is not None:
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = hand_bbox
                        bbox = np.array([bbox_x1, bbox_y1, bbox_x2 - bbox_x1, bbox_y2 - bbox_y1], dtype=np.float32)
                    else:
                        # Create bbox from root joint (default size)
                        bbox = np.array([center_2d[0] - 50, center_2d[1] - 50, 100, 100], dtype=np.float32)
                    
                    bbox = self.process_bbox(bbox, img_shape[1], img_shape[0], expansion_factor=1.0)
                    if bbox is None:
                        continue
                else:
                    # Training: use full joint annotations
                    # Project joints to image coordinates
                    joints_coord_img = self.joint3DToImg(joints_coord_cam, (fx, fy, fu, fv))
                    center_2d = self.get_center(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]))
                    
                    # Get bounding box
                    bbox = self.get_bbox(joints_coord_img[:, :2], expansion_factor=1.5)
                    bbox = self.process_bbox(bbox, img_shape[1], img_shape[0], expansion_factor=1.0)
                    if bbox is None:
                        continue
                
                self.dataset_len += 1
                
                data = {
                    "img_path": rgb_path,
                    "img_shape": img_shape,
                    "joints_coord_cam": joints_coord_cam,
                    "joints_coord_img": joints_coord_img,
                    "center_2d": center_2d,
                    "cam_param": cam_param,
                    "mano_pose": hand_pose,
                    "mano_shape": hand_beta,
                    "mano_trans": hand_trans
                }
                
                datalist.append(data)
        
        return datalist

    def __len__(self):
        return self.dataset_len

    def jointImgTo3D(self, uvd, paras):
        """Convert joint from image coordinates to 3D"""
        fx, fy, fu, fv = paras
        ret = np.zeros_like(uvd, np.float32)
        
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = self.flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = self.flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = self.flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]
        return ret

    def joint3DToImg(self, xyz, paras):
        """Convert joint from 3D to image coordinates"""
        fx, fy, fu, fv = paras
        ret = np.zeros_like(xyz, np.float32)
        
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (self.flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (self.flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (self.flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    def get_center(self, joint_img, joint_valid):
        """Get center from joint image coordinates"""
        x_img, y_img = joint_img[:, 0], joint_img[:, 1]
        x_img = x_img[joint_valid == 1]
        y_img = y_img[joint_valid == 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)
        
        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        
        return [x_center, y_center]

    def get_bbox(self, joint_img, expansion_factor=1.0):
        """Get bounding box from joint image coordinates"""
        x_img, y_img = joint_img[:, 0], joint_img[:, 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)
        
        x_center = (xmin + xmax) / 2.
        width = (xmax - xmin) * expansion_factor
        xmin = x_center - 0.5 * width
        xmax = x_center + 0.5 * width
        
        y_center = (ymin + ymax) / 2.
        height = (ymax - ymin) * expansion_factor
        ymin = y_center - 0.5 * height
        ymax = y_center + 0.5 * height
        
        bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
        return bbox

    def process_bbox(self, bbox, img_width, img_height, expansion_factor=1.25):
        """Process and sanitize bounding box"""
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w * h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
        else:
            return None
        
        # Aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2.
        c_y = bbox[1] + h / 2.
        aspect_ratio = 1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w * expansion_factor
        bbox[3] = h * expansion_factor
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.
        
        return bbox

    def comToBounds(self, com, size, paras):
        """Calculate boundaries from center of mass"""
        fx, fy, fu, fv = paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize, paras):
        """Calculate affine transform from crop"""
        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size, paras)
        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = np.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = np.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1
        xstart = int(np.floor(dsize[0] / 2. - sz[0] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[1] / 2.))
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart
        return np.dot(off, np.dot(scale, trans))

    def getCrop(self, depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
        """Crop patch from image"""
        if len(depth.shape) == 2:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1]))), mode='constant',
                             constant_values=background)
        elif len(depth.shape) == 3:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]), :].copy()
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1])),
                                       (0, 0)), mode='constant', constant_values=background)
        else:
            raise NotImplementedError()
        if thresh_z is True:
            msk1 = np.logical_and(cropped < zstart, cropped != 0)
            msk2 = np.logical_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.
        return cropped

    def Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        """Crop area of hand in 3D volumina"""
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size, paras)
        cropped = self.getCrop(depth, xstart, xend, ystart, yend, zstart, zend)
        
        # Check if cropped is empty or invalid
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            # Return zero-filled image and identity transform
            ret = np.zeros(dsize, dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        wb = (xend - xstart)
        hb = (yend - ystart)
        
        # Ensure valid dimensions
        if wb <= 0 or hb <= 0:
            ret = np.zeros(dsize, dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])
        
        # Ensure sz is valid (both dimensions > 0)
        if sz[0] <= 0 or sz[1] <= 0:
            ret = np.zeros(dsize, dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])
        scale[2, 2] = 1
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        ret = np.ones(dsize, np.float32) * 0
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart
        return ret, np.dot(off, np.dot(scale, trans))

    def Crop_Image_deep_pp_RGB(self, rgb, com, size, dsize, paras):
        """Crop RGB image"""
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size, paras)
        cropped = self.getCrop(rgb, xstart, xend, ystart, yend, zstart, zend, thresh_z=False)
        
        # Check if cropped is empty or invalid
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            # Return zero-filled image and identity transform
            ret = np.zeros((dsize[0], dsize[1], 3), dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        wb = (xend - xstart)
        hb = (yend - ystart)
        
        # Ensure valid dimensions
        if wb <= 0 or hb <= 0:
            ret = np.zeros((dsize[0], dsize[1], 3), dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])
        
        # Ensure sz is valid (both dimensions > 0)
        if sz[0] <= 0 or sz[1] <= 0:
            ret = np.zeros((dsize[0], dsize[1], 3), dtype=np.float32)
            trans = np.eye(3)
            return ret, trans
        
        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])
        scale[2, 2] = 1
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_LINEAR)
        rgb_size = (dsize[0], dsize[1], 3)
        ret = np.ones(rgb_size, np.float32) * 0
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart
        return ret, np.dot(off, np.dot(scale, trans))

    def normalize_img(self, premax, imgD, com, cube):
        """Normalize depth image"""
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        """Random augmentation parameters"""
        if sigma_com is None:
            sigma_com = self.aug_para[0]
        if sigma_sc is None:
            sigma_sc = self.aug_para[1]
        if rot_range is None:
            rot_range = self.aug_para[2]
        mode = random.randint(0, len(self.aug_modes) - 1)
        off = np.array([random.uniform(-1, 1) for a in range(3)]) * sigma_com
        rot = random.uniform(-rot_range, rot_range)
        sc = abs(1. + random.uniform(-1, 1) * sigma_sc)
        return mode, off, rot, sc

    def rotateHand(self, dpt, cube, com, rot, joints3D, paras=None, pad_value=0, thresh_z=True):
        """Rotate hand virtually in the image plane"""
        if np.allclose(rot, 0.):
            return dpt, joints3D, rot
        rot = np.mod(rot, 360)
        M = cv2.getRotationMatrix2D((dpt.shape[1] // 2, dpt.shape[0] // 2), -rot, 1)
        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)
        if thresh_z and len(dpt[dpt > 0]) > 0:
            new_dpt[new_dpt < (np.min(dpt[dpt > 0]) - 1)] = 0
        com3D = self.jointImgTo3D(com, paras)
        joint_2D = self.joint3DToImg(joints3D + com3D, paras)
        data_2D = np.zeros_like(joint_2D)
        for k in range(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D, paras) - com3D)
        return new_dpt, new_joints3D, rot

    def moveCoM(self, dpt, cube, com, off, joints3D, M, paras=None, pad_value=0, thresh_z=True):
        """Adjust already cropped image such that a moving CoM normalization is simulated"""
        if np.allclose(off, 0.):
            return dpt, joints3D, com, M
        new_com = self.jointImgTo3D(self.jointImgTo3D(com, paras) + off, paras)
        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            Mnew = self.comToTransform(new_com, cube, dpt.shape, paras)
            if len(dpt[dpt > 0]) > 0:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=np.min(dpt[dpt > 0]) - 1, thresh_z=thresh_z, com=new_com, size=cube)
            else:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=-1, thresh_z=thresh_z, com=new_com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt
        new_joints3D = joints3D + self.jointImgTo3D(com, paras) - self.jointImgTo3D(new_com, paras)
        return new_dpt, new_joints3D, new_com, Mnew

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, paras, pad_value=0, thresh_z=True):
        """Virtually scale the hand by applying different cube"""
        if np.allclose(sc, 1.):
            return dpt, joints3D, cube, M
        new_cube = [s * sc for s in cube]
        if not np.allclose(com[2], 0.):
            Mnew = self.comToTransform(com, new_cube, dpt.shape, paras)
            if len(dpt[dpt > 0]) > 0:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=np.min(dpt[dpt > 0]) - 1, thresh_z=thresh_z, com=com, size=cube)
            else:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=-1, thresh_z=thresh_z, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt
        new_joints3D = joints3D
        return new_dpt, new_joints3D, new_cube, Mnew

    def recropHand(self, crop, M, Mnew, target_size, paras, background_value=0., nv_val=0., thresh_z=True, com=None, size=(280, 280, 280)):
        """Recrop hand with new transform"""
        flags = cv2.INTER_NEAREST
        if len(target_size) > 2:
            target_size = target_size[0:2]
        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        if thresh_z:
            warped[warped < nv_val] = background_value
        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size, paras)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.
        return warped

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras=None):
        """Commonly used function to augment hand poses"""
        assert len(img.shape) == 2
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
            new_joints3D = gt3Dcrop
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, paras, pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, paras, pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras, pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgD = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        imgD = self.normalize_img(premax, imgD, com, cube)
        return imgD, None, new_joints3D, np.asarray(cube), com, M, rot

    def augmentCrop_RGB(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras=None):
        """Augment RGB image"""
        if self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgRGB, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, paras, pad_value=0, thresh_z=False)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            M_rot = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -rot, 1)
            imgRGB = cv2.warpAffine(img, M_rot, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            com3D = self.jointImgTo3D(com, paras)
            joint_2D = self.joint3DToImg(gt3Dcrop + com3D, paras)
            data_2D = np.zeros_like(joint_2D)
            for k in range(data_2D.shape[0]):
                data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
            new_joints3D = (self.jointImgTo3D(data_2D, paras) - com3D)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgRGB, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras, pad_value=0, thresh_z=False)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgRGB = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        return imgRGB, None, new_joints3D, np.asarray(cube), com, M, rot

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']
        
        # Load RGB
        if 'RGB' in self.input_modal:
            rgb_path = data['img_path']
            rgb = cv2.imread(rgb_path)
            if not isinstance(rgb, np.ndarray):
                raise IOError(f"Fail to read {rgb_path}")
        else:
            rgb = None
        
        # RGB-only pipeline: ignore depth modality
        
        intrinsics = data['cam_param']
        cam_para = (intrinsics['focal'][0], intrinsics['focal'][1], 
                   intrinsics['princpt'][0], intrinsics['princpt'][1])
        
        # Check if this is evaluation data (no full annotations)
        # NOTE: 'val' is carved from 'train.txt' and DOES have full GT, so it is NOT evaluation.
        is_evaluation = (self.data_split == 'test' or self.data_split == 'evaluation')
        
        # IMPORTANT:
        # 'val' split is a subset of train.txt, so it should follow the same GT path as 'train'.
        if self.data_split in ('train', 'val', 'test', 'train_all', 'evaluation'):
            joint_xyz = data['joints_coord_cam'].reshape([21, 3])[HO3D2MANO, :] * 1000  # Convert to mm
            
            if is_evaluation:
                # For evaluation: joints_coord_cam is dummy (all same as root joint)
                # Use root joint position as center
                center_xyz = joint_xyz[0].copy()  # Use root joint as center
                # Create dummy gt3Dcrop (all zeros since we don't have real joint positions)
                gt3Dcrop = np.zeros_like(joint_xyz)
            else:
                # Training: use joint mean as center (WiLoR style, no refined center)
                center_xyz = joint_xyz.mean(0)
                gt3Dcrop = joint_xyz - center_xyz
            
            joint_uvd = self.joint3DToImg(joint_xyz, cam_para)
            joint_xyz = joint_xyz[WILOR_JOINT_MAP]
            joint_uvd = joint_uvd[WILOR_JOINT_MAP]
            gt3Dcrop = gt3Dcrop[WILOR_JOINT_MAP]
        else:
            # For other splits
            joint_xyz = np.ones([21, 3])
            joint_uvd = np.ones([21, 3])
            gt3Dcrop = np.ones([21, 3])
            center_xyz = joint_xyz.mean(0)
        
        center_uvd = self.joint3DToImg(center_xyz, cam_para)  # Convert to image coordinates

        if self.align_wilor_aug and rgb is not None and not is_evaluation:
            from .wilor_utils import get_bbox, get_example
            keypoints_2d = np.concatenate([joint_uvd[:, :2], np.ones((21, 1), dtype=np.float32)], axis=1)
            keypoints_3d = np.concatenate([joint_xyz, np.ones((21, 1), dtype=np.float32)], axis=1)
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
                center = np.array([img_shape[1] / 2.0, img_shape[0] / 2.0], dtype=np.float32)
                scale = np.array([img_shape[1], img_shape[0]], dtype=np.float32)

            # NEW AUGMENTATION 1: Random center jitter (align with FreiHANDDatasetV2)
            bbox_size = float(scale.max())
            if self.train and self.center_jitter_factor > 0:
                jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                center[0] += jitter_x
                center[1] += jitter_y

            mano_pose = data['mano_pose']
            mano_shape = data['mano_shape']
            mano_trans = data['mano_trans']
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
            img_patch, keypoints_2d, keypoints_3d, aug_mano_params, _, _, trans = get_example(
                rgb, center[0], center[1], scale[0], scale[1],
                keypoints_2d, keypoints_3d, mano_params, has_mano_params,
                flip_perm, self.patch_width, self.patch_height,
                None, None,
                do_augment=self.train, is_right=True,
                augm_config=self.wilor_aug_config, is_bgr=True,
                return_trans=True
            )

            # Convert to float [0, 1]
            imgRGB_01 = torch.from_numpy(img_patch).float() / 255.0

            # NEW AUGMENTATION 2 & 3: Brightness and Contrast adjustment (align with FreiHANDDatasetV2)
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

            joints_3d_np = keypoints_3d[:, :3].astype(np.float32)
            mano_pose = np.concatenate(
                [aug_mano_params['global_orient'], aug_mano_params['hand_pose']],
                axis=0
            ).astype(np.float32)
            mano_shape = aug_mano_params['betas'].astype(np.float32)
            trans_3x3 = np.eye(3, dtype=np.float32)
            trans_3x3[:2, :] = trans
            K = np.array(
                [[cam_para[0], 0.0, cam_para[2]], [0.0, cam_para[1], cam_para[3]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            K_patch = trans_3x3 @ K
            cam_para = (K_patch[0, 0], K_patch[1, 1], K_patch[0, 2], K_patch[1, 2])
            scale_norm = scale / 200.0
            # bbox_size already calculated earlier for center jitter
            bbox_expand_factor = float(bbox_size / max((scale_norm * 200.0).max(), 1e-6))
            uv_xy = keypoints_2d[:, :2]
            uv_valid = ((uv_xy > -0.5) & (uv_xy < 0.5)).astype(np.float32)
            uv_valid = (uv_valid[:, 0] * uv_valid[:, 1]).astype(np.float32)
            xyz_valid = 1.0
            if uv_valid[self.root_joint_idx] == 0 and uv_valid[0] == 0:
                xyz_valid = 0.0
            return {
                'rgb': imgRGB,
                'keypoints_2d': torch.from_numpy(keypoints_2d.astype(np.float32)).float(),
                'keypoints_3d': torch.from_numpy((joints_3d_np - joints_3d_np.mean(0)) / (self.cube_size[2] / 2.0)).float(),
                'joints_3d_gt': torch.from_numpy(joints_3d_np).float(),
                'mano_params': aug_mano_params,
                'mano_pose': torch.from_numpy(mano_pose).float(),
                'mano_shape': torch.from_numpy(mano_shape).float(),
                'mano_trans': torch.from_numpy(mano_trans).float(),
                'cam_param': torch.from_numpy(np.array(cam_para)).float(),
                'box_center': torch.from_numpy(center.astype(np.float32)),
                'box_size': torch.tensor(bbox_size, dtype=torch.float32),
                'bbox_expand_factor': torch.tensor(bbox_expand_factor, dtype=torch.float32),
                '_scale': torch.from_numpy(scale_norm.astype(np.float32)),
                'mano_params_is_axis_angle': {
                    'global_orient': True,
                    'hand_pose': True,
                    'betas': False
                },
                'uv_valid': torch.from_numpy(uv_valid.astype(np.float32)),
                'xyz_valid': torch.tensor(xyz_valid, dtype=torch.float32),
                'hand_type': 'right',
                'is_right': 1.0
            }
        
        raise NotImplementedError("HO3D RGB-only path requires align_wilor_aug=True.")


# HO3D to MANO joint mapping (moved from commented section at top)
HO3D2MANO = [0,
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
             10, 11, 12,
             13, 14, 15,
             17,
             18,
             20,
             19,
             16]

# Import WILOR_JOINT_MAP from utils
from .utils import WILOR_JOINT_MAP


def main():
    """Debug/inspect HO3DDataset by printing a few samples."""
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

    parser = argparse.ArgumentParser(description="Print/inspect HO3DDataset contents.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to HO3D root directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "evaluation", "train_all"], help="Dataset split.")
    parser.add_argument("--img-size", type=int, default=256, help="Output image size.")
    parser.add_argument("--center-jitter", type=float, default=0.05, help="Center jitter factor.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    ds = HO3DDataset(
        data_split=args.split,
        root_dir=args.root_dir,
        img_size=args.img_size,
        train=(args.split == 'train'),
        align_wilor_aug=True,
        center_jitter_factor=args.center_jitter,
    )

    print("\n=== HO3DDataset summary ===")
    print(f"split={ds.data_split}, len={len(ds)}")
    print(f"root_dir={ds.root_dir}")
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


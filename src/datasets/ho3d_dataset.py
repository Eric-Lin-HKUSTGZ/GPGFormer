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
from .wilor_utils import WILOR_JOINT_MAP

# HO3D to MANO joint mapping
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


def transformPoint2D(pt, M):
    """Transform point in 2D coordinates"""
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """Transform points in 2D coordinates"""
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


def rotatePoint2D(p1, center, angle):
    """Rotate a point in 2D around center"""
    alpha = angle * np.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = np.zeros_like(pp)
    pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
    pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def sample_modality(has_depth, p_drop=0.4):
    """
    Random modality sampling (OmniVGGT style)
    Args:
        has_depth: whether the sample has depth
        p_drop: probability to drop depth (0.3-0.5)
    Returns:
        n_i: 0 if depth is dropped, 1 if depth is kept
    """
    if has_depth:
        n_i = 0 if random.random() < p_drop else 1
    else:
        n_i = 0
    return n_i


class HO3DDataset(Dataset):
    """
    HO3D Dataset with random modality sampling
    Supports HO3D_v3 format
    """
    def __init__(self, data_split, root_dir, dataset_version='v3', img_size=256, 
                 aug_para=[10, 0.2, 180], cube_size=[280, 280, 280],
                 input_modal='RGBD', color_factor=0.2, p_drop=0.4, train=True,
                 aug_prob=0.8, color_aug_prob=0.6, align_wilor_aug=False,
                 wilor_aug_config=None):
        """
        Args:
            data_split: 'train', 'test', or 'train_all'
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
        self.dataset_version = dataset_version
        # Allow passing either the dataset parent dir or the HO3D_v3 dir itself
        candidate_root = osp.join(root_dir, 'HO3D_%s' % (dataset_version))
        if osp.exists(candidate_root):
            self.root_dir = candidate_root
        else:
            self.root_dir = root_dir
        self.root_joint_idx = 0
        self.color_factor = color_factor
        self.input_modal = input_modal
        self.img_size = img_size
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
        # WiLoR uses ImageNet normalization: (img - mean) / std
        # mean = 255 * [0.485, 0.456, 0.406], std = 255 * [0.229, 0.224, 0.225]
        self.mean = 255. * np.array([0.485, 0.456, 0.406])  # ImageNet mean
        self.std = 255. * np.array([0.229, 0.224, 0.225])  # ImageNet std
        self.transform = transforms.ToTensor()  # Converts to CHW and float32
        
        self.dataset_len = 0
        self.datalist = self.load_data()
        print(f'HO3D Dataset loaded: {self.dataset_len} samples from {data_split} split')

    def load_data(self):
        """Load dataset annotations from pickle files (HO3D_v3 format)"""
        # HO3D_v3 has train.txt and evaluation.txt (no test.txt)
        # Map 'test' to 'evaluation'
        split_name = self.data_split
        if split_name == 'test':
            split_name = 'evaluation'
        
        # Read image list from train.txt or evaluation.txt
        split_file = osp.join(self.root_dir, f'{split_name}.txt')
        if not osp.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        datalist = []
        # Map split to folder name
        if split_name == 'train':
            split_folder = 'train'
        else:
            split_folder = 'evaluation'  # evaluation or test -> evaluation folder
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
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
                
                depth_path = osp.join(seq_dir, 'depth', f'{file_id}.png')
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
                # For evaluation/test split: handPose, handBeta, handTrans are not available
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
                    # Training data: check for full annotations
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
                    "depth_path": depth_path,
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

    def read_depth_img(self, depth_filename):
        """Read the depth image in dataset and decode it"""
        if not osp.exists(depth_filename):
            return None
        depth_scale = 0.00012498664727900177
        depth_img = cv2.imread(depth_filename)
        if depth_img is None:
            return None
        # Convert to uint16 to avoid overflow when multiplying by 256
        # depth_img is uint8, so we need larger dtype for the calculation
        dpt = depth_img[:, :, 2].astype(np.uint16) + depth_img[:, :, 1].astype(np.uint16) * 256
        dpt = dpt.astype(np.float32) * depth_scale * 1000
        return dpt

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
        
        # Load Depth
        depth_path = data.get('depth_path', data['img_path'].replace('rgb', 'depth').replace('.jpg', '.png').replace('.png', '.png'))
        depth = self.read_depth_img(depth_path)
        has_depth = depth is not None and depth.size > 0
        
        # Random modality sampling (OmniVGGT style)
        if self.train:
            n_i = sample_modality(has_depth, self.p_drop)
        else:
            # During test, always use depth if available
            n_i = 1 if has_depth else 0
        
        intrinsics = data['cam_param']
        cam_para = (intrinsics['focal'][0], intrinsics['focal'][1], 
                   intrinsics['princpt'][0], intrinsics['princpt'][1])
        
        # Check if this is evaluation data (no full annotations)
        is_evaluation = (self.data_split == 'test' or self.data_split == 'evaluation')
        
        if self.data_split == 'train' or self.data_split == 'test' or self.data_split == 'train_all' or self.data_split == 'evaluation':
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
            center, scale = get_bbox(keypoints_2d, rescale=1.2)
            if scale[0] < 1 or scale[1] < 1:
                center = np.array([img_shape[1] / 2.0, img_shape[0] / 2.0], dtype=np.float32)
                scale = np.array([img_shape[1], img_shape[0]], dtype=np.float32)
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
            img_patch, keypoints_2d, keypoints_3d, aug_mano_params, _, _ = get_example(
                rgb, center[0], center[1], scale[0], scale[1],
                keypoints_2d, keypoints_3d, mano_params, has_mano_params,
                flip_perm, self.img_size, self.img_size,
                self.mean, self.std,
                do_augment=self.train, is_right=True,
                augm_config=self.wilor_aug_config, is_bgr=True
            )
            joints_3d_np = keypoints_3d[:, :3].astype(np.float32)
            mano_pose = np.concatenate(
                [aug_mano_params['global_orient'], aug_mano_params['hand_pose']],
                axis=0
            ).astype(np.float32)
            mano_shape = aug_mano_params['betas'].astype(np.float32)
            data_depth = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return {
                'rgb': torch.from_numpy(img_patch).float(),
                'depth': data_depth,
                'n_i': 0,
                'has_depth': False,
                'joint_img': torch.from_numpy(keypoints_2d.astype(np.float32)).float(),
                'joint_3d': torch.from_numpy((joints_3d_np - joints_3d_np.mean(0)) / (self.cube_size[2] / 2.0)).float(),
                'joints_3d_gt': torch.from_numpy(joints_3d_np).float(),
                'mano_pose': torch.from_numpy(mano_pose).float(),
                'mano_shape': torch.from_numpy(mano_shape).float(),
                'mano_trans': torch.from_numpy(mano_trans).float(),
                'cam_param': torch.from_numpy(np.array(cam_para)).float(),
                'center': torch.from_numpy(center_xyz).float(),
                'M': torch.eye(3).float(),
                'cube': torch.from_numpy(np.array(self.cube_size)).float(),
                'hand_type': 'right',
                'is_right': 1.0
            }
        
        # Crop depth
        if has_depth:
            depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size, 
                                                        (self.img_size, self.img_size), cam_para)
        else:
            depth_crop = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            trans = np.eye(3)
        
        # Crop RGB
        if rgb is not None:
            rgb_crop, trans_rgb = self.Crop_Image_deep_pp_RGB(copy.deepcopy(rgb), center_uvd, self.cube_size,
                                                              (self.img_size, self.img_size), cam_para)
        else:
            rgb_crop = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            trans_rgb = np.eye(3)
        
        # Augmentation
        if self.data_split == 'train' and self.train:
            # Apply geometric augmentation with probability aug_prob
            if random.random() < self.aug_prob:
                mode, off, rot, sc = self.rand_augment()
            else:
                # No geometric augmentation
                # mode = 3 corresponds to 'none' in self.aug_modes = ['rot', 'com', 'sc', 'none']
                mode = 3
                off = np.array([0, 0, 0])  # Must be numpy array, same shape as rand_augment() returns
                rot = 0
                sc = 1.0
            
            if has_depth:
                imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(
                    depth_crop, gt3Dcrop, center_uvd, self.cube_size, trans, mode, off, rot, sc, cam_para)
            else:
                imgD = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
                cube = np.array(self.cube_size)
                com2D = center_uvd
                M = trans
            
            if rgb is not None:
                imgRGB, _, curLabel_rgb, cube_rgb, com2D_rgb, M_rgb, _ = self.augmentCrop_RGB(
                    rgb_crop, gt3Dcrop, center_uvd, self.cube_size, trans_rgb, mode, off, rot, sc, cam_para)
                
                # Apply color augmentation with probability color_aug_prob
                if self.color_factor != 0 and random.random() < self.color_aug_prob:
                    # RGB augment (WiLoR style)
                    c_up = 1.0 + self.color_factor
                    c_low = 1.0 - self.color_factor
                    color_scale = np.array(
                        [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
                    imgRGB = np.clip(imgRGB * color_scale[None, None, :], 0, 255)
                
                # Convert to tensor (HWC -> CHW, float32)
                imgRGB = self.transform(imgRGB.astype(np.float32))
                # Apply WiLoR normalization: (img - mean) / std
                for n_c in range(3):
                    imgRGB[n_c, :, :] = (imgRGB[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
            else:
                imgRGB = torch.zeros((3, self.img_size, self.img_size))
            
            curLabel = curLabel / (cube[2] / 2.0)
        else:
            # Test mode
            if has_depth:
                imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            else:
                imgD = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            if rgb is not None:
                # Convert to tensor (HWC -> CHW, float32)
                imgRGB = self.transform(rgb_crop.astype(np.float32))
                # Apply WiLoR normalization: (img - mean) / std
                for n_c in range(3):
                    imgRGB[n_c, :, :] = (imgRGB[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
            else:
                imgRGB = torch.zeros((3, self.img_size, self.img_size))
            
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans
        
        com3D = self.jointImgTo3D(com2D, cam_para)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D, cam_para), M)
        # WiLoR normalization: keypoints_2d = keypoints_2d / patch_width - 0.5
        # This normalizes to [-0.5, 0.5] range instead of [-1, 1]
        joint_img[:, 0:2] = joint_img[:, 0:2] / self.img_size - 0.5
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)
        
        # Convert to tensors
        data_depth = torch.from_numpy(imgD).float().unsqueeze(0)  # (1, H, W)
        data_rgb = imgRGB  # (3, H, W)
        joint_img = torch.from_numpy(joint_img).float()  # (21, 3)
        joint = torch.from_numpy(curLabel).float()  # (21, 3)
        center = torch.from_numpy(com3D).float()  # (3,)
        M_tensor = torch.from_numpy(M).float()  # (3, 3)
        cube_tensor = torch.from_numpy(cube).float()  # (3,)
        cam_para_tensor = torch.from_numpy(np.array(cam_para)).float()  # (4,)
        
        # MANO parameters
        if self.data_split == 'train' or self.data_split == 'test' or self.data_split == 'train_all':
            mano_pose_tensor = torch.from_numpy(data['mano_pose']).float()  # (48,)
            mano_shape_tensor = torch.from_numpy(data['mano_shape']).float()  # (10,)
            mano_trans_tensor = torch.from_numpy(data['mano_trans']).float()  # (3,)
        else:
            # Dummy values for other splits
            mano_pose_tensor = torch.zeros(48).float()
            mano_shape_tensor = torch.zeros(10).float()
            mano_trans_tensor = torch.zeros(3).float()
        
        # Ground truth joints
        joints_3d_gt = torch.from_numpy(joint_xyz).float()  # (21, 3)
        
        return {
            'rgb': data_rgb,  # (3, H, W)
            'depth': data_depth,  # (1, H, W)
            'n_i': n_i,  # modality flag: 0 or 1
            'has_depth': has_depth,  # whether depth exists
            'joint_img': joint_img,  # (21, 3) normalized joint image coordinates
            'joint_3d': joint,  # (21, 3) normalized joint 3D coordinates
            'joints_3d_gt': joints_3d_gt,  # (21, 3) absolute joint 3D coordinates
            'mano_pose': mano_pose_tensor,  # (48,)
            'mano_shape': mano_shape_tensor,  # (10,)
            'mano_trans': mano_trans_tensor,  # (3,)
            'cam_param': cam_para_tensor,  # (4,)
            'center': center,  # (3,)
            'M': M_tensor,  # (3, 3)
            'cube': cube_tensor,  # (3,)
            'hand_type': 'right',  # HO3D is right hand only
            'is_right': 1.0
        }

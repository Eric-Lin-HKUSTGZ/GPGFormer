# -*- coding: utf-8 -*-
"""
Dex-YCB Dataset Loader with Random Modality Sampling
参考 KeypointFusion/dataloader/loader.py
"""
import os
import os.path as osp
import copy
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from scipy import ndimage
from .wilor_utils import WILOR_JOINT_MAP

# DexYCB to MANO joint mapping
DexYCB2MANO = [
    0,
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
    8, 12, 20, 16, 4
]


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


class DexYCBDataset(Dataset):
    """
    Dex-YCB Dataset with random modality sampling
    """
    def __init__(self, setup, split, root_dir, img_size=256, aug_para=[10, 0.2, 30], 
                 input_modal='RGBD', p_drop=0.4, train=True, color_factor=0.2, 
                 aug_prob=0.8, color_aug_prob=0.6, align_wilor_aug=False,
                 wilor_aug_config=None):
        """
        Args:
            setup: dataset setup (e.g., 's0')
            split: 'train' or 'test'
            root_dir: root directory containing dex-ycb folder
            img_size: output image size
            aug_para: [sigma_com, sigma_sc, rot_range] for augmentation
            input_modal: 'RGBD' or 'RGB'
            p_drop: probability to drop depth during training (OmniVGGT style)
            train: whether in training mode
            color_factor: color jitter factor for RGB augmentation
            aug_prob: probability to apply geometric augmentation
            color_aug_prob: probability to apply color augmentation
        """
        self.setup = setup
        if split == 'val':
            split = 'test'
        self.split = split
        self.root_dir = root_dir.rstrip('/')
        # Allow passing either the dataset parent dir or the dex-ycb dir itself
        candidate_root = os.path.join(self.root_dir, 'dex-ycb')
        if osp.exists(osp.join(candidate_root, 'annotations')):
            self.dex_ycb_root = candidate_root
        else:
            self.dex_ycb_root = self.root_dir
        self.annot_path = osp.join(self.dex_ycb_root, 'annotations')
        self.input_modal = input_modal
        self.img_size = img_size
        self.aug_para = aug_para
        self.cube_size = [250, 250, 250]
        self.aug_modes = ['rot', 'com', 'sc', 'none']
        self.flip = 1
        self.p_drop = p_drop
        self.train = train
        self.color_factor = color_factor
        self.aug_prob = aug_prob
        self.color_aug_prob = color_aug_prob
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}
        # WiLoR uses ImageNet normalization: (img - mean) / std
        # mean = 255 * [0.485, 0.456, 0.406], std = 255 * [0.229, 0.224, 0.225]
        self.mean = 255. * np.array([0.485, 0.456, 0.406])  # ImageNet mean
        self.std = 255. * np.array([0.229, 0.224, 0.225])  # ImageNet std
        self.transform = transforms.ToTensor()  # Converts to CHW and float32
        
        self.datalist = self.load_data()
        print(f'Loaded {len(self.datalist)} samples from Dex-YCB {setup} {split}')

    def load_data(self):
        """Load dataset annotations"""
        db = COCO(osp.join(self.annot_path, f"DEX_YCB_{self.setup}_{self.split}_data.json"))
        datalist = []

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            color_file_name = img['color_file_name']
            
            # Extract relative path
            def extract_subject_path(path):
                parts = path.split('/')
                for i, part in enumerate(parts):
                    if part and len(part) > 8 and part[0].isdigit() and 'subject' in part:
                        return '/'.join(parts[i:])
                return parts[-1] if parts else path
            
            if color_file_name.startswith('/home/pfren/dataset/') or color_file_name.startswith('/home/cyc/pycharm/data/hand/'):
                rel_path = color_file_name.replace('/home/pfren/dataset/', '')
                if rel_path.startswith('hand/'):
                    rel_path = rel_path.split('/', 1)[1] if '/' in rel_path else rel_path
                if rel_path.startswith('DexYCB/') or rel_path.startswith('dex-ycb/'):
                    rel_path = rel_path.split('/', 1)[1] if '/' in rel_path else rel_path
                rel_path = extract_subject_path(rel_path)
                img_path = osp.join(self.dex_ycb_root, rel_path)
            elif osp.isabs(color_file_name):
                if '/dex-ycb/' in color_file_name.lower() or '/dexycb/' in color_file_name.lower():
                    parts = color_file_name.split('/')
                    try:
                        dataset_idx = next(i for i, p in enumerate(parts) if p.lower() in ['dex-ycb', 'dexycb'])
                        rel_path = '/'.join(parts[dataset_idx+1:])
                        if rel_path.startswith('hand/'):
                            rel_path = rel_path[5:]
                        rel_path = extract_subject_path(rel_path)
                        img_path = osp.join(self.dex_ycb_root, rel_path)
                    except StopIteration:
                        rel_path = extract_subject_path(color_file_name)
                        img_path = osp.join(self.dex_ycb_root, rel_path)
                else:
                    rel_path = extract_subject_path(color_file_name)
                    img_path = osp.join(self.dex_ycb_root, rel_path)
            else:
                if color_file_name.startswith('hand/'):
                    color_file_name = color_file_name[5:]
                if color_file_name.startswith('DexYCB/') or color_file_name.startswith('dex-ycb/'):
                    color_file_name = color_file_name.split('/', 1)[1] if '/' in color_file_name else color_file_name
                rel_path = extract_subject_path(color_file_name)
                img_path = osp.join(self.dex_ycb_root, rel_path)
            
            img_path = osp.normpath(img_path)
            img_path = img_path.replace('/DexYCB/', '/dex-ycb/')
            img_shape = (img['height'], img['width'])

            joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) / 1000  # meter
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
            hand_type = ann['hand_type']

            if joints_coord_cam.sum() == -63:
                continue

            mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
            mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)
            mano_trans = np.array(ann['mano_param']['trans'], dtype=np.float32)

            data = {
                "img_path": img_path,
                "img_shape": img_shape,
                "joints_coord_cam": joints_coord_cam,
                "cam_param": cam_param,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                'mano_trans': mano_trans,
                "hand_type": hand_type
            }
            datalist.append(data)
        return datalist

    def _label_path_from_image(self, img_path):
        label_path = img_path.replace('color_', 'labels_')
        label_path = osp.splitext(label_path)[0] + '.npz'
        return label_path

    def _load_label_npz(self, img_path):
        label_path = self._label_path_from_image(img_path)
        if not osp.exists(label_path):
            return None
        try:
            with np.load(label_path) as data:
                return {k: data[k] for k in data.files}
        except Exception:
            return None

    def _compute_crop_center_from_labels(self, label_data, depth, img_shape, cam_para):
        if label_data is None:
            return None, None

        seg = label_data.get('seg')
        mask = seg == 255 if seg is not None else None

        joints_3d = label_data.get('joint_3d')
        if joints_3d is not None and joints_3d.size > 0:
            if joints_3d.ndim == 3:
                joints_3d = joints_3d[0]
            valid = np.isfinite(joints_3d).all(axis=1)
            if np.any(valid):
                # labels_*.npz joint_3d is in meters; convert to mm to match cube_size/depth units.
                center_xyz = (joints_3d[valid].mean(axis=0) * 1000.0).astype(np.float32)
                center_uvd = self.joint3DToImg(center_xyz, cam_para)
                return center_uvd.astype(np.float32), center_xyz

        joints_2d = label_data.get('joint_2d')
        if joints_2d is not None and joints_2d.size > 0:
            if joints_2d.ndim == 3:
                joints_2d = joints_2d[0]
            valid = np.isfinite(joints_2d).all(axis=1)
            if np.any(valid):
                cx = float(joints_2d[valid, 0].mean())
                cy = float(joints_2d[valid, 1].mean())
            else:
                cx = cy = None
        else:
            cx = cy = None

        if mask is not None and np.any(mask):
            ys, xs = np.where(mask)
            cx = float((xs.min() + xs.max()) / 2.0)
            cy = float((ys.min() + ys.max()) / 2.0)

        if cx is None or cy is None:
            return None, None

        z = 0.0
        if depth is not None and depth.size > 0:
            if mask is not None and np.any(mask):
                depth_vals = depth[mask]
                if depth_vals.size > 0:
                    z = float(np.median(depth_vals))
            else:
                ix = int(np.clip(round(cx), 0, img_shape[1] - 1))
                iy = int(np.clip(round(cy), 0, img_shape[0] - 1))
                z = float(depth[iy, ix])

        center_uvd = np.array([cx, cy, z], dtype=np.float32)
        center_xyz = self.jointImgTo3D(center_uvd, cam_para)
        return center_uvd, center_xyz

    def __len__(self):
        return len(self.datalist)

    def jointImgTo3D(self, uvd, paras):
        """Convert joint from image coordinates to 3D"""
        fx, fy, fu, fv = paras
        ret = np.zeros_like(uvd, np.float32)
        
        # Handle different input shapes: 1D (single point), 2D (batch of points), or 3D
        if len(ret.shape) == 1:
            # Single point: shape (3,)
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = self.flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            # Batch of points: shape (N, 3)
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = self.flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            # 3D array: shape (H, W, 3) or similar
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = self.flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]
        return ret

    def joint3DToImg(self, xyz, paras):
        """Convert joint from 3D to image coordinates"""
        fx, fy, fu, fv = paras
        ret = np.zeros_like(xyz, np.float32)
        
        # Handle different input shapes: 1D (single point), 2D (batch of points), or 3D
        if len(ret.shape) == 1:
            # Single point: shape (3,)
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (self.flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            # Batch of points: shape (N, 3)
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (self.flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            # 3D array: shape (H, W, 3) or similar
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (self.flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

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
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])
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
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])
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
        new_com = self.joint3DToImg(self.jointImgTo3D(com, paras) + off, paras)
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

    def recropHand(self, crop, M, Mnew, target_size, paras, background_value=0., nv_val=0., thresh_z=True, com=None, size=(250, 250, 250)):
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
        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')
        
        # Load RGB
        if 'RGB' in self.input_modal:
            rgb = cv2.imread(img_path)
            if not isinstance(rgb, np.ndarray):
                raise IOError(f"Fail to read {img_path}")
        else:
            rgb = None

        # Load Depth
        depth_path = img_path.replace('color_', 'aligned_depth_to_color_').replace('jpg', 'png')
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
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
        label_data = self._load_label_npz(img_path)
        label_center_uvd, label_center_xyz = self._compute_crop_center_from_labels(
            label_data, depth, img_shape, cam_para
        )

        joint_xyz = data['joints_coord_cam'].reshape([21, 3])[DexYCB2MANO, :] * 1000
        joint_uvd = self.joint3DToImg(joint_xyz, cam_para)
        mano_pose = data['mano_pose']
        mano_shape = data['mano_shape']
        mano_trans = data['mano_trans']

        if do_flip:
            if rgb is not None:
                rgb = rgb[:, ::-1].copy()
            if has_depth:
                depth = depth[:, ::-1].copy()
            joint_uvd[:, 0] = img_shape[1] - joint_uvd[:, 0] - 1
            if label_center_uvd is not None:
                label_center_uvd[0] = img_shape[1] - label_center_uvd[0] - 1
                label_center_xyz = self.jointImgTo3D(label_center_uvd, cam_para)

        joint_xyz = self.jointImgTo3D(joint_uvd, cam_para)
        joint_xyz = joint_xyz[WILOR_JOINT_MAP]
        joint_uvd = joint_uvd[WILOR_JOINT_MAP]
        center_xyz = joint_xyz.mean(0)
        center_uvd = self.joint3DToImg(center_xyz, cam_para)
        if label_center_uvd is not None and label_center_xyz is not None:
            center_uvd = label_center_uvd
            center_xyz = label_center_xyz
        gt3Dcrop = joint_xyz - center_xyz

        if self.align_wilor_aug and rgb is not None:
            from .wilor_utils import get_bbox, get_example
            keypoints_2d = np.concatenate([joint_uvd[:, :2], np.ones((21, 1), dtype=np.float32)], axis=1)
            keypoints_3d = np.concatenate([joint_xyz, np.ones((21, 1), dtype=np.float32)], axis=1)
            center, scale = get_bbox(keypoints_2d, rescale=1.2)
            if scale[0] < 1 or scale[1] < 1:
                center = np.array([img_shape[1] / 2.0, img_shape[0] / 2.0], dtype=np.float32)
                scale = np.array([img_shape[1], img_shape[0]], dtype=np.float32)
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
                flip_perm, self.img_size, self.img_size,
                self.mean, self.std,
                do_augment=self.train, is_right=(hand_type == 'right'),
                augm_config=self.wilor_aug_config, is_bgr=True,
                return_trans=True
            )
            imgRGB = torch.from_numpy(img_patch).float()
            joint_img = keypoints_2d.astype(np.float32)
            joints_3d_np = keypoints_3d[:, :3].astype(np.float32)
            joints_3d_gt = torch.from_numpy(joints_3d_np).float()
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
            data_depth = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return {
                'rgb': imgRGB,
                'depth': data_depth,
                'n_i': 0,
                'has_depth': False,
                'joint_img': torch.from_numpy(joint_img).float(),
                'joint_3d': torch.from_numpy((joints_3d_np - joints_3d_np.mean(0)) / (self.cube_size[2] / 2.0)).float(),
                'joints_3d_gt': joints_3d_gt,
                'mano_pose': torch.from_numpy(mano_pose).float(),
                'mano_shape': torch.from_numpy(mano_shape).float(),
                'mano_trans': torch.from_numpy(mano_trans).float(),
                'cam_param': torch.from_numpy(np.array(cam_para)).float(),
                'center': torch.from_numpy(center_xyz).float(),
                'M': torch.eye(3).float(),
                'cube': torch.from_numpy(np.array(self.cube_size)).float(),
                'hand_type': hand_type,
                'is_right': 1.0 if hand_type == 'right' else 0.0
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
        if self.split == 'train' and self.train:
            mode, off, rot, sc = self.rand_augment()
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
                
                # Apply color augmentation (KeypointFusion-inspired)
                if random.random() < self.color_aug_prob and self.color_factor > 0:
                    # Per-channel independent color jitter (±20%)
                    c_up = 1.0 + self.color_factor
                    c_low = 1.0 - self.color_factor
                    color_scale = np.array([
                        random.uniform(c_low, c_up),  # R channel
                        random.uniform(c_low, c_up),  # G channel
                        random.uniform(c_low, c_up)   # B channel
                    ])
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
        mano_pose_tensor = torch.from_numpy(mano_pose).float()  # (48,)
        mano_shape_tensor = torch.from_numpy(mano_shape).float()  # (10,)
        mano_trans_tensor = torch.from_numpy(mano_trans).float()  # (3,)

        # Ground truth joints and vertices (will be computed from MANO if needed)
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
            'hand_type': hand_type,  # 'left' or 'right'
            'is_right': 1.0 if hand_type == 'right' else 0.0
        }

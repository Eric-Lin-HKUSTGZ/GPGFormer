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
import torchvision.transforms as transforms
from scipy import ndimage
from .utils import WILOR_JOINT_MAP, get_bbox, get_example
# # DexYCB to MANO joint mapping
# DexYCB2MANO = [
#     0,
#     5, 6, 7,
#     9, 10, 11,
#     17, 18, 19,
#     13, 14, 15,
#     1, 2, 3,
#     8, 12, 20, 16, 4
# ]


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


class DexYCBDataset(Dataset):
    """
    Dex-YCB Dataset with random modality sampling
    """
    def __init__(self, setup, split, root_dir, img_size=256, aug_para=[10, 0.2, 30],
                 input_modal='RGBD', p_drop=0.4, train=True, color_factor=0.2,
                 aug_prob=0.8, color_aug_prob=0.6, align_wilor_aug=False,
                 wilor_aug_config=None,
                 bbox_source: str = "gt",
                 detector_weights_path: str | None = None,
                 root_index: int = 9,
                 # New augmentation parameters (align with FreiHANDDatasetV2)
                 center_jitter_factor: float = 0.05,
                 brightness_limit: tuple = (-0.2, 0.1),
                 contrast_limit: tuple = (0.8, 1.2),
                 brightness_prob: float = 0.5,
                 contrast_prob: float = 0.5):
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
        self.root_index = int(root_index)
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
        self.input_modal = "RGB"
        self.img_size = img_size
        # WiLoR ViT expects 256x192; keep height=img_size, width=0.75*img_size
        self.patch_height = self.img_size
        self.patch_width = int(round(self.img_size * 0.75))
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

        # New augmentation parameters (align with FreiHANDDatasetV2)
        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        # if self.bbox_source == "detector" and (not self.train):
        #     if detector_weights_path is None:
        #         raise ValueError("bbox_source='detector' requires detector_weights_path")
        #     from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector
        #     self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=detector_weights_path))
        # WiLoR uses ImageNet normalization (align with FreiHANDDatasetV2)
        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        
        self.datalist = self.load_data()
        print(f'Loaded {len(self.datalist)} samples from Dex-YCB {setup} {split}')

    def load_data(self):
        """Load dataset annotations"""
        import json

        json_path = osp.join(self.annot_path, f"DEX_YCB_{self.setup}_{self.split}_data.json")
        with open(json_path, "r") as f:
            db = json.load(f)

        images_by_id = {img["id"]: img for img in db.get("images", [])}
        annotations = db.get("annotations", [])
        datalist = []

        for ann in annotations:
            image_id = ann['image_id']
            img = images_by_id[image_id]
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
        
        # Load RGB only
        rgb = cv2.imread(img_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f"Fail to read {img_path}")

        intrinsics = data['cam_param']
        cam_para = (intrinsics['focal'][0], intrinsics['focal'][1], 
                   intrinsics['princpt'][0], intrinsics['princpt'][1])
        label_center_uvd, label_center_xyz = None, None

        joint_xyz = data['joints_coord_cam'].reshape([21, 3])[DexYCB2MANO, :] * 1000
        joint_uvd = self.joint3DToImg(joint_xyz, cam_para)
        mano_pose = data['mano_pose']
        mano_shape = data['mano_shape']
        mano_trans = data['mano_trans']

        if do_flip:
            rgb = rgb[:, ::-1].copy()
            joint_uvd[:, 0] = img_shape[1] - joint_uvd[:, 0] - 1
            if label_center_uvd is not None:
                label_center_uvd[0] = img_shape[1] - label_center_uvd[0] - 1
                label_center_xyz = self.jointImgTo3D(label_center_uvd, cam_para)

        joint_xyz = self.jointImgTo3D(joint_uvd, cam_para)
        joint_xyz = joint_xyz[WILOR_JOINT_MAP]
        joint_uvd = joint_uvd[WILOR_JOINT_MAP]
        # center_xyz = joint_xyz.mean(0)
        # center_uvd = self.joint3DToImg(center_xyz, cam_para)
        # if label_center_uvd is not None and label_center_xyz is not None:
        #     center_uvd = label_center_uvd
        #     center_xyz = label_center_xyz
        
        keypoints_2d = np.concatenate([joint_uvd[:, :2], np.ones((21, 1), dtype=np.float32)], axis=1)
        keypoints_3d = np.concatenate([joint_xyz, np.ones((21, 1), dtype=np.float32)], axis=1)
        # if self.bbox_source == "detector" and self.detector is not None:
        #     bbox = self.detector(rgb)
        #     if bbox is not None:
        #         x1, y1, x2, y2 = bbox
        #         center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        #         scale = np.array([(x2 - x1) * 1.2, (y2 - y1) * 1.2], dtype=np.float32)
        #     else:
        #         center, scale = get_bbox(keypoints_2d, rescale=1.2)
        # else:
        center, scale = get_bbox(keypoints_2d, rescale=1.2)
        if scale[0] < 1 or scale[1] < 1:
            center = np.array([img_shape[1] / 2.0, img_shape[0] / 2.0], dtype=np.float32)
            scale = np.array([img_shape[1], img_shape[0]], dtype=np.float32)

        
        bbox_size = float(scale.max())
        

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
            do_augment=self.train, is_right=(hand_type == 'right'),
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
        if uv_valid[self.root_index] == 0 and uv_valid[0] == 0:
            xyz_valid = 0.0
        mano_params_is_axis_angle = {
            'global_orient': True,
            'hand_pose': True,
            'betas': False
        }
        return {
            'rgb': imgRGB,
            'keypoints_2d': torch.from_numpy(keypoints_2d.astype(np.float32)).float(),
            'keypoints_3d': torch.from_numpy((joints_3d_np - joints_3d_np.mean(0)) / (self.cube_size[2] / 2.0)).float(),
            # 'joints_3d_gt': torch.from_numpy(joints_3d_np).float(),
            'mano_params': aug_mano_params,
            'cam_param': torch.from_numpy(np.array(cam_para)).float(),
            'box_center': torch.from_numpy(center.astype(np.float32)),
            'box_size': torch.tensor(bbox_size, dtype=torch.float32),
            'bbox_expand_factor': torch.tensor(bbox_expand_factor, dtype=torch.float32),
            '_scale': torch.from_numpy(scale_norm.astype(np.float32)),
            'mano_params_is_axis_angle': mano_params_is_axis_angle,
            'uv_valid': torch.from_numpy(uv_valid.astype(np.float32)),
            'xyz_valid': torch.tensor(xyz_valid, dtype=torch.float32),
            'hand_type': hand_type,
            'is_right': 1.0 if hand_type == 'right' else 0.0
        }

       


# DexYCB to MANO joint mapping (moved from commented section at top)
DexYCB2MANO = [
    0,
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
    8, 12, 20, 16, 4
]


def main():
    """Debug/inspect DexYCBDataset by printing a few samples."""
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

    parser = argparse.ArgumentParser(description="Print/inspect DexYCBDataset contents.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to DexYCB root directory.")
    parser.add_argument("--setup", type=str, default="s0", help="Dataset setup (e.g., s0, s1, s2, s3).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Dataset split.")
    parser.add_argument("--img-size", type=int, default=256, help="Output image size.")
    parser.add_argument("--center-jitter", type=float, default=0.05, help="Center jitter factor.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Default augmentation config (same as config_dexycb.yaml)
    default_wilor_aug_config = {
        'SCALE_FACTOR': 0.3,
        'ROT_FACTOR': 30,
        'TRANS_FACTOR': 0.02,
        'COLOR_SCALE': 0.2,
        'ROT_AUG_RATE': 0.6,
        'TRANS_AUG_RATE': 0.5,
        'DO_FLIP': False,
        'FLIP_AUG_RATE': 0.0,
        'EXTREME_CROP_AUG_RATE': 0.0,
        'EXTREME_CROP_AUG_LEVEL': 1,
    }

    ds = DexYCBDataset(
        setup=args.setup,
        split=args.split,
        root_dir=args.root_dir,
        img_size=args.img_size,
        train=(args.split == 'train'),
        align_wilor_aug=True,
        wilor_aug_config=default_wilor_aug_config,
        center_jitter_factor=args.center_jitter,
    )

    print("\n=== DexYCBDataset summary ===")
    print(f"setup={ds.setup}, split={ds.split}, len={len(ds)}")
    print(f"dex_ycb_root={ds.dex_ycb_root}")
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


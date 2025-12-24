# -*- coding: utf-8 -*-
"""
WiLoR-style dataset utilities (minimal subset).
These utilities align crop/augment/keypoint/MANO processing with WiLoR conventions.
"""
from typing import Dict, List, Tuple
import cv2
import numpy as np

WILOR_JOINT_MAP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def do_augmentation(aug_config: Dict) -> Tuple[float, float, bool, bool, int, List[float], float, float]:
    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config['TRANS_FACTOR']
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config['TRANS_FACTOR']
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config['SCALE_FACTOR'] + 1.0
    if np.random.rand() <= aug_config['ROT_AUG_RATE']:
        rot = np.clip(np.random.randn(), -2.0, 2.0) * aug_config['ROT_FACTOR']
    else:
        rot = 0.0
    do_flip = aug_config['DO_FLIP'] and (np.random.rand() <= aug_config['FLIP_AUG_RATE'])
    do_extreme_crop = np.random.rand() <= aug_config['EXTREME_CROP_AUG_RATE']
    extreme_crop_lvl = int(aug_config.get('EXTREME_CROP_AUG_LEVEL', 0))
    c_up = 1.0 + aug_config['COLOR_SCALE']
    c_low = 1.0 - aug_config['COLOR_SCALE']
    color_scale = [np.random.uniform(c_low, c_up),
                   np.random.uniform(c_low, c_up),
                   np.random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty


def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def trans_point2d(pt_2d: np.array, trans: np.array):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    img_height, img_width, _ = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height,
                                    patch_width, patch_height, scale, rot)
    img_patch = cv2.warpAffine(
        img, trans, (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value
    )
    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.array) -> np.array:
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)


def fliplr_params(mano_params: Dict, has_mano_params: Dict) -> Tuple[Dict, Dict]:
    global_orient = mano_params['global_orient'].copy()
    hand_pose = mano_params['hand_pose'].copy()
    betas = mano_params['betas'].copy()
    has_global_orient = has_mano_params['global_orient'].copy()
    has_hand_pose = has_mano_params['hand_pose'].copy()
    has_betas = has_mano_params['betas'].copy()

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    hand_pose[1::3] *= -1
    hand_pose[2::3] *= -1

    mano_params = {
        'global_orient': global_orient.astype(np.float32),
        'hand_pose': hand_pose.astype(np.float32),
        'betas': betas.astype(np.float32)
    }
    has_mano_params = {
        'global_orient': has_global_orient,
        'hand_pose': has_hand_pose,
        'betas': has_betas
    }
    return mano_params, has_mano_params


def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    joints = joints.copy()
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]
    return joints


def keypoint_3d_processing(keypoints_3d: np.array, flip_permutation: List[int],
                           rot: float, do_flip: bool) -> np.array:
    if do_flip:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    rot_mat = np.eye(3)
    if rot != 0:
        rot_rad = -rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    keypoints_3d[:, :-1] = np.einsum('ij,kj->ki', rot_mat, keypoints_3d[:, :-1])
    return keypoints_3d.astype(np.float32)


def rot_aa(aa: np.array, rot: float) -> np.array:
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    per_rdg, _ = cv2.Rodrigues(aa)
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)


def mano_param_processing(mano_params: Dict, has_mano_params: Dict,
                          rot: float, do_flip: bool) -> Tuple[Dict, Dict]:
    if do_flip:
        mano_params, has_mano_params = fliplr_params(mano_params, has_mano_params)
    mano_params['global_orient'] = rot_aa(mano_params['global_orient'], rot)
    return mano_params, has_mano_params


def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple[np.array, np.array]:
    valid = keypoints_2d[:, -1] > 0
    valid_keypoints = keypoints_2d[valid][:, :-1]
    if valid_keypoints.size == 0:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    scale = bbox_size * rescale
    return center, scale


def get_example(img: np.array, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, keypoints_3d: np.array,
                mano_params: Dict, has_mano_params: Dict,
                flip_kp_permutation: List[int],
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, is_right: bool, augm_config: Dict,
                is_bgr: bool = True,
                return_trans: bool = False) -> Tuple:
    cvimg = img.copy()
    img_height, img_width, img_channels = cvimg.shape
    img_size = np.array([img_height, img_width])

    if do_augment:
        scale, rot, do_flip, _, _, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, _, _, color_scale, tx, ty = 1.0, 0.0, False, False, 0, [1.0, 1.0, 1.0], 0.0, 0.0

    if not is_right:
        do_flip = True

    center_x += width * tx
    center_y += height * ty

    keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

    img_patch_cv, trans = generate_image_patch_cv2(
        cvimg, center_x, center_y, width, height,
        patch_width, patch_height, do_flip, scale, rot
    )

    image = img_patch_cv.copy()
    if is_bgr:
        image = image[:, :, ::-1]
    img_patch = convert_cvimg_to_tensor(image)

    mano_params, has_mano_params = mano_param_processing(mano_params, has_mano_params, rot, do_flip)

    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    if do_flip:
        keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)

    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

    if not return_trans:
        return img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size
    return img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size, trans

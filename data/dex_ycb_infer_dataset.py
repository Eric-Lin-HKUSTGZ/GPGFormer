# -*- coding: utf-8 -*-
"""
Dex-YCB Dataset Loader aligned with FreiHANDDatasetV2.

Changes vs. legacy Dex-YCB loader:
1) Keep Dex-YCB data loading logic (image/annotation paths) unchanged.
2) Align augmentation & output format with freihand_dataset_v2.py (get_example, ImageNet norm).
3) Remove unused complex augmentation utilities.
4) Convert MANO hand_pose_pca (45) to axis-angle (45) before augmentation.
"""
import json
import os
import os.path as osp
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import WILOR_JOINT_MAP, get_example

_MANO_PCA_CACHE = {}


def _resolve_mano_model_path(mano_root: str | None, side: str) -> Path | None:
    side = "right" if str(side).lower().startswith("r") else "left"
    filename = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"

    candidates = []
    if mano_root:
        p = Path(mano_root)
        if p.is_file():
            return p
        candidates.extend([p, p / "models", p / "MANO", p / "mano" / "models"])

    for var in ("MANO_ROOT", "MANO_DIR", "MANO_MODEL_DIR", "SMPLX_MANO_DIR"):
        env = os.environ.get(var)
        if env:
            candidates.append(Path(env))

    repo_root = Path(__file__).resolve().parents[2]  # hand_reconstruction
    candidates.extend(
        [
            repo_root / "freihand" / "mano_v1_2" / "models",
            repo_root / "freihand" / "data",
            repo_root / "HandGCAT" / "common" / "utils" / "manopth" / "mano" / "models",
            Path("/root/code/vepfs/GPGFormer/weights/mano"),
        ]
    )

    for root in candidates:
        cand = root / filename
        if cand.exists():
            return cand
    return None


def _load_mano_pca(mano_root: str | None, side: str) -> tuple[np.ndarray, np.ndarray]:
    key = (str(mano_root) if mano_root else "", side)
    if key in _MANO_PCA_CACHE:
        return _MANO_PCA_CACHE[key]

    path = _resolve_mano_model_path(mano_root, side)
    if path is None:
        raise FileNotFoundError(
            "MANO model not found for PCA decoding. "
            "Provide `mano_root` or set MANO_ROOT/MANO_DIR/SMPLX_MANO_DIR."
        )
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    hands_components = np.asarray(data["hands_components"], dtype=np.float32)
    hands_mean = np.asarray(data.get("hands_mean", np.zeros((45,), dtype=np.float32)), dtype=np.float32)
    _MANO_PCA_CACHE[key] = (hands_components, hands_mean)
    return hands_components, hands_mean


def _pca_to_axis_angle(hand_pose_pca: np.ndarray, mano_root: str | None, side: str) -> np.ndarray:
    pca = np.asarray(hand_pose_pca, dtype=np.float32).reshape(-1)
    comps, mean = _load_mano_pca(mano_root, side)
    ncomps = pca.shape[0]

    if comps.shape[0] >= ncomps:
        full = pca @ comps[:ncomps]
    elif comps.shape[1] >= ncomps:
        full = comps[:, :ncomps] @ pca
    else:
        raise ValueError(f"Unexpected MANO PCA components shape: {comps.shape} for ncomps={ncomps}")

    full = np.asarray(full, dtype=np.float32).reshape(-1)
    if mean.shape[0] != full.shape[0]:
        raise ValueError(f"MANO hands_mean shape {mean.shape} does not match hand pose {full.shape}")
    return mean + full


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


class DexYCBDataset(Dataset):
    """
    Dex-YCB dataset aligned with FreiHANDDatasetV2 (RGB-only).
    """

    def __init__(
        self,
        setup,
        split,
        root_dir,
        img_size=256,
        train=True,
        color_factor=0.2,
        color_aug_prob=0.6,
        align_wilor_aug=False,
        wilor_aug_config=None,
        bbox_source: str = "gt",
        detector_weights_path: str | None = None,
        root_index: int = 9,
        center_jitter_factor: float = 0.05,
        brightness_limit: tuple = (-0.2, 0.1),
        contrast_limit: tuple = (0.8, 1.2),
        brightness_prob: float = 0.5,
        contrast_prob: float = 0.5,
        img_width: int | None = None,
        mano_root: str | None = None,
        mano_pose_is_pca: bool = True,
    ):
        self.setup = setup
        self.root_index = int(root_index)
        if split == "val":
            split = "test"
        self.split = split
        self.root_dir = root_dir.rstrip("/")
        candidate_root = os.path.join(self.root_dir, "dex-ycb")
        if osp.exists(osp.join(candidate_root, "annotations")):
            self.dex_ycb_root = candidate_root
        else:
            self.dex_ycb_root = self.root_dir
        self.annot_path = osp.join(self.dex_ycb_root, "annotations")
        self.input_modal = "RGB"
        self.img_size = int(img_size)
        self.patch_height = self.img_size
        if img_width is None:
            img_width = int(round(self.img_size * 0.75))
        self.patch_width = int(img_width)
        self.cube_size = [250, 250, 250]
        self.flip = 1
        self.train = train
        self.color_factor = color_factor
        self.color_aug_prob = color_aug_prob
        self.align_wilor_aug = align_wilor_aug
        self.wilor_aug_config = wilor_aug_config or {}

        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        self.bbox_source = str(bbox_source).lower()
        self.detector = None
        self.detector_bbox_expand = 1.2
        if self.bbox_source == "detector":
            if detector_weights_path is None or len(str(detector_weights_path).strip()) == 0:
                raise ValueError("DexYCBDataset bbox_source='detector' requires detector_weights_path.")
            from gpgformer.models.detector.wilor_yolo import WiLoRDetectorConfig, WiLoRYOLODetector

            self.detector = WiLoRYOLODetector(WiLoRDetectorConfig(weights_path=str(detector_weights_path)))
            if str(os.environ.get("RANK", "0")) == "0":
                print("[info] DexYCBDataset uses detector-only hand crop (no label bbox fallback).")
        elif self.bbox_source != "gt":
            raise ValueError(f"Unsupported bbox_source for DexYCBDataset: {self.bbox_source}")
        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        self.mano_root = mano_root
        self.mano_pose_is_pca = bool(mano_pose_is_pca)

        self.datalist = self.load_data()
        print(f"Loaded {len(self.datalist)} samples from Dex-YCB {setup} {split}")

    def load_data(self):
        """Load dataset annotations"""
        json_path = osp.join(self.annot_path, f"DEX_YCB_{self.setup}_{self.split}_data.json")
        with open(json_path, "r") as f:
            db = json.load(f)

        images_by_id = {img["id"]: img for img in db.get("images", [])}
        annotations = db.get("annotations", [])
        datalist = []

        for ann in annotations:
            image_id = ann["image_id"]
            img = images_by_id[image_id]
            color_file_name = img["color_file_name"]

            def extract_subject_path(path):
                parts = path.split("/")
                for i, part in enumerate(parts):
                    if part and len(part) > 8 and part[0].isdigit() and "subject" in part:
                        return "/".join(parts[i:])
                return parts[-1] if parts else path

            if color_file_name.startswith("/home/pfren/dataset/") or color_file_name.startswith(
                "/home/cyc/pycharm/data/hand/"
            ):
                rel_path = color_file_name.replace("/home/pfren/dataset/", "")
                if rel_path.startswith("hand/"):
                    rel_path = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path
                if rel_path.startswith("DexYCB/") or rel_path.startswith("dex-ycb/"):
                    rel_path = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path
                rel_path = extract_subject_path(rel_path)
                img_path = osp.join(self.dex_ycb_root, rel_path)
            elif osp.isabs(color_file_name):
                if "/dex-ycb/" in color_file_name.lower() or "/dexycb/" in color_file_name.lower():
                    parts = color_file_name.split("/")
                    try:
                        dataset_idx = next(i for i, p in enumerate(parts) if p.lower() in ["dex-ycb", "dexycb"])
                        rel_path = "/".join(parts[dataset_idx + 1 :])
                        if rel_path.startswith("hand/"):
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
                if color_file_name.startswith("hand/"):
                    color_file_name = color_file_name[5:]
                if color_file_name.startswith("DexYCB/") or color_file_name.startswith("dex-ycb/"):
                    color_file_name = color_file_name.split("/", 1)[1] if "/" in color_file_name else color_file_name
                rel_path = extract_subject_path(color_file_name)
                img_path = osp.join(self.dex_ycb_root, rel_path)

            img_path = osp.normpath(img_path)
            img_path = img_path.replace("/DexYCB/", "/dex-ycb/")
            img_shape = (img["height"], img["width"])

            joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32)
            # Dex-YCB `joints_coord_cam` is commonly stored in millimeters (values ~[0, 900]).
            # The model / training loop uses meters end-to-end, so convert here to avoid a 1000x
            # scale mismatch that would blow up the 3D loss on Dex-YCB but not on FreiHAND
            # (whose `*_xyz.json` is already in meters).
            if np.isfinite(joints_coord_cam).all():
                # Heuristic: meters-scale joints have mean abs value <~ 2, mm-scale is >> 10.
                if float(np.abs(joints_coord_cam).mean()) > 10.0:
                    joints_coord_cam = joints_coord_cam / 1000.0
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()}
            hand_type = ann["hand_type"]

            if joints_coord_cam.sum() == -63:
                continue

            mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
            mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)
            mano_trans = np.array(ann["mano_param"]["trans"], dtype=np.float32)

            data = {
                "img_path": img_path,
                "img_shape": img_shape,
                "joints_coord_cam": joints_coord_cam,
                "cam_param": cam_param,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "mano_trans": mano_trans,
                "hand_type": hand_type,
            }
            datalist.append(data)
        return datalist

    def __len__(self):
        return len(self.datalist)

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

    def _decode_mano_pose(self, mano_pose: np.ndarray, hand_type: str) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(mano_pose, dtype=np.float32).reshape(-1)
        if not self.mano_pose_is_pca:
            if pose.shape[0] == 45:
                return np.zeros((3,), dtype=np.float32), pose
            return pose[:3].copy(), pose[3:].copy()

        if pose.shape[0] == 45:
            global_orient = np.zeros((3,), dtype=np.float32)
            hand_pose_pca = pose
        elif pose.shape[0] >= 48:
            global_orient = pose[:3].copy()
            hand_pose_pca = pose[3:].copy()
        else:
            raise ValueError(f"Unexpected Dex-YCB MANO pose length: {pose.shape[0]}")

        side = "right" if str(hand_type).lower().startswith("r") else "left"
        hand_pose = _pca_to_axis_angle(hand_pose_pca, self.mano_root, side)
        return global_orient, hand_pose

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        data = self.datalist[idx]
        img_path, img_shape = data["img_path"], data["img_shape"]
        hand_type = data["hand_type"]

        rgb = cv2.imread(img_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f"Fail to read {img_path}")

        h, w = rgb.shape[:2]

        intrinsics = data["cam_param"]
        cam_para = (intrinsics["focal"][0], intrinsics["focal"][1], intrinsics["princpt"][0], intrinsics["princpt"][1])

        # Dex-YCB joint order in annotations is already the canonical 21-joint hand skeleton:
        # wrist -> (thumb/index/middle/ring/little) mcp/pip/dip/tip.
        # Keep this order to match the model output `pred_keypoints_3d` (FreiHAND/OpenPose-style 21).
        joint_xyz = data["joints_coord_cam"].reshape([21, 3])
        joint_uvd = self.joint3DToImg(joint_xyz, cam_para)

        joint_xyz = self.jointImgTo3D(joint_uvd, cam_para)
        # NOTE: Do NOT apply WILOR_JOINT_MAP here; that mapping is for MANO-wrapper internal joints
        # to OpenPose order. Our Dex-YCB joints are already in the OpenPose-style order.

        keypoints_2d = joint_uvd[:, :2].astype(np.float32)
        keypoints_3d = joint_xyz.astype(np.float32)

        uv_norm = keypoints_2d.copy()
        uv_norm[:, 0] /= w
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32")
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        if self.bbox_source == "detector":
            center, scale_px = _bbox_from_detector(
                self.detector, rgb, w, h, img_path, expand_factor=self.detector_bbox_expand
            )
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
        if (not np.isfinite(bbox_size)) or bbox_size < 1.0:
            center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            scale_px = np.array([w, h], dtype=np.float32)
            scale = (scale_px / 200.0).astype(np.float32)
            bbox_size = float(scale_px.max())
        denom = float(max(scale_px.max(), 1e-6))
        bbox_expand_factor = float(bbox_size / denom)

        # NEW AUGMENTATION 1: Random center jitter (align with FreiHANDDatasetV2)
        # if self.train and self.center_jitter_factor > 0:
        #     jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
        #     jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
        #     center[0] += jitter_x
        #     center[1] += jitter_y

        global_orient, hand_pose = self._decode_mano_pose(data["mano_pose"], hand_type)
        mano_params = {
            "global_orient": global_orient.copy(),
            "hand_pose": hand_pose.copy(),
            "betas": data["mano_shape"].copy(),
        }
        has_mano_params = {
            "global_orient": np.array([1.0], dtype=np.float32),
            "hand_pose": np.array([1.0], dtype=np.float32),
            "betas": np.array([1.0], dtype=np.float32),
        }
        flip_perm = list(range(21))

        img_patch, keypoints_2d, keypoints_3d, mano_params, _has_params, _, trans = get_example(
            rgb,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            keypoints_2d,
            keypoints_3d,
            mano_params,
            has_mano_params,
            flip_perm,
            self.patch_width,
            self.patch_height,
            None,
            None,
            do_augment=self.train,
            is_right=(hand_type == "right"),
            augm_config=self.wilor_aug_config,
            is_bgr=True,
            return_trans=True,
        )

        imgRGB_01 = torch.from_numpy(img_patch).float() / 255.0

        # Brightness / contrast (align with FreiHANDDatasetV2)
        if self.train:
            if np.random.rand() < self.brightness_prob:
                brightness_delta = np.random.uniform(self.brightness_limit[0], self.brightness_limit[1])
                imgRGB_01 = torch.clamp(imgRGB_01 + brightness_delta, 0.0, 1.0)

            if np.random.rand() < self.contrast_prob:
                contrast_factor = np.random.uniform(self.contrast_limit[0], self.contrast_limit[1])
                mean_val = imgRGB_01.mean(dim=[1, 2], keepdim=True)
                imgRGB_01 = torch.clamp((imgRGB_01 - mean_val) * contrast_factor + mean_val, 0.0, 1.0)

        imgRGB = (imgRGB_01 - self.imagenet_mean) / self.imagenet_std

        trans_3x3 = np.eye(3, dtype=np.float32)
        trans_3x3[:2, :] = trans
        K = np.array(
            [[cam_para[0], 0.0, cam_para[2]], [0.0, cam_para[1], cam_para[3]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        K_patch = trans_3x3 @ K
        cam_para = (K_patch[0, 0], K_patch[1, 1], K_patch[0, 2], K_patch[1, 2])

        trans_uv01 = keypoints_2d.copy()
        trans_uv01[:, 0] = trans_uv01[:, 0] + 0.5
        trans_uv01[:, 1] = trans_uv01[:, 1] + 0.5

        trans_coord_valid = (trans_uv01 > 0).astype("float32") * (trans_uv01 < 1).astype("float32")
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid

        xyz_valid = 1
        if trans_coord_valid[self.root_index] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0

        mano_params_is_axis_angle = {
            "global_orient": True,
            "hand_pose": True,
            "betas": False,
        }

        return {
            "rgb": imgRGB,
            "keypoints_2d": torch.from_numpy(keypoints_2d.astype(np.float32)).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d.astype(np.float32)).float(),
            "mano_params": mano_params,
            "cam_param": torch.tensor(cam_para, dtype=torch.float32),
            "box_center": torch.from_numpy(center.astype(np.float32)),
            "box_size": torch.tensor(bbox_size, dtype=torch.float32),
            "bbox_expand_factor": torch.tensor(bbox_expand_factor, dtype=torch.float32),
            "_scale": torch.from_numpy(scale.astype(np.float32)),
            "mano_params_is_axis_angle": mano_params_is_axis_angle,
            "xyz_valid": xyz_valid,
            "uv_valid": trans_coord_valid,
            "hand_type": hand_type,
            "is_right": 1.0 if hand_type == "right" else 0.0,
        }


def main():
    """Debug/inspect DexYCBDataset by printing a few samples."""
    import argparse

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
    parser.add_argument("--mano-root", type=str, default=None, help="MANO model directory for PCA decoding.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to print.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    default_wilor_aug_config = {
        "SCALE_FACTOR": 0.3,
        "ROT_FACTOR": 30,
        "TRANS_FACTOR": 0.02,
        "COLOR_SCALE": 0.2,
        "ROT_AUG_RATE": 0.6,
        "TRANS_AUG_RATE": 0.5,
        "DO_FLIP": False,
        "FLIP_AUG_RATE": 0.0,
        "EXTREME_CROP_AUG_RATE": 0.0,
        "EXTREME_CROP_AUG_LEVEL": 1,
    }

    ds = DexYCBDataset(
        setup=args.setup,
        split=args.split,
        root_dir=args.root_dir,
        img_size=args.img_size,
        train=(args.split == "train"),
        align_wilor_aug=True,
        wilor_aug_config=default_wilor_aug_config,
        center_jitter_factor=args.center_jitter,
        mano_root=args.mano_root,
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

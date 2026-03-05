# -*- coding: utf-8 -*-
"""
HO3D_v3 dataset loader (RGB-only), aligned with this repo's FreiHAND/Dex-YCB loaders.

Alignment goals:
- Use WiLoR-style `get_example()` for crop + geometric augmentation.
- Return ImageNet-normalized patch `rgb` (CHW), normalized 2D joints in [-0.5, 0.5],
  3D joints in meters, and MANO parameters.

HO3D specifics handled here:
- Coordinate convention: HO3D_v3 annotations use a camera convention where z is typically negative
  for points in front of the camera (and y is flipped relative to the repo's other datasets).
  We rotate 180 degrees around the x-axis to match the common convention used by Dex-YCB/FreiHAND:
    (x, y, z) -> (x, -y, -z)
- Joint order: In this repo's HO3D_v3 dump, `handJoints3D` is already in the same 21-joint order
  as SMPL-X MANO's internal joints (16) + 5 fingertip vertices (5):
    [wrist, index(3), middle(3), pinky(3), ring(3), thumb(3), tips(thumb,index,middle,ring,pinky)]
  Therefore we only need to map to the model's 21-joint output order using `WILOR_JOINT_MAP`.
- Evaluation split: `evaluation.txt` meta files do not include full GT (often only root joint + bbox).
  We still return a sample with dummy joints/MANO and masks (`has_mano_params`, `uv_valid`, `xyz_valid`)
  that prevent losses from using missing GT.
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
import pickle
import random
import tempfile
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    from .utils import WILOR_JOINT_MAP, get_example
except ImportError:  # pragma: no cover
    from utils import WILOR_JOINT_MAP, get_example


# HO3D meta joint order mapping (length 21).
# Kept as a separate constant because different HO3D dumps/codebases sometimes permute the 5 fingertips.
# For /root/code/vepfs/dataset/HO3D_v3 used by this repo, the fingertips are already ordered as:
#   (thumb, index, middle, ring, pinky) at indices 16..20.
HO3D_META_JOINT_MAP: List[int] = list(range(21))


def _project_points(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points in camera coords to image coordinates."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    uvw = (K @ xyz.T).T  # (N,3)
    uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-7)
    return uv.astype(np.float32)


def _sanitize_bbox_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray | None:
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1)
    if xyxy.size != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    # Clamp to image bounds (xyxy in pixel coords)
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w - 1)))
    y2 = max(0.0, min(y2, float(h - 1)))
    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_from_keypoints(keypoints_2d: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bbox center/scale from keypoints and also return a per-joint valid mask (in image frame).
    Returns:
      center: (2,)
      scale_px: (2,) width/height in pixels
      coord_valid: (21,) 0/1 in the original image frame
    """
    kp = np.asarray(keypoints_2d, dtype=np.float32).reshape(-1, 2)
    uv_norm = kp.copy()
    uv_norm[:, 0] /= float(max(w, 1))
    uv_norm[:, 1] /= float(max(h, 1))
    valid = ((uv_norm > 0).astype(np.float32) * (uv_norm < 1).astype(np.float32))
    coord_valid = (valid[:, 0] * valid[:, 1]).astype(np.float32)

    pts = kp[coord_valid > 0.5]
    if pts.shape[0] >= 2:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = (mn + mx) / 2.0
        # Match FreiHAND/Dex-YCB loaders in this repo:
        #   scale = 2 * (max - min) / 200
        #   bbox_size = max(scale * 200) = 2 * (max - min)
        scale_px = 2.0 * (mx - mn)
    else:
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale_px = np.array([w, h], dtype=np.float32)
    scale_px = np.maximum(scale_px, 2.0)
    return center.astype(np.float32), scale_px.astype(np.float32), coord_valid


def _ho3d_cam_to_std_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Convert HO3D camera coords to the repo's standard camera coords.
    This is a 180-degree rotation around x-axis: diag(1, -1, -1).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    out = xyz.copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _ho3d_cam_to_std_global_orient(aa: np.ndarray) -> np.ndarray:
    """
    Convert global_orient axis-angle to match `_ho3d_cam_to_std_xyz`.
    For a basis rotation Rconv, we use: R' = Rconv @ R.
    """
    aa = np.asarray(aa, dtype=np.float32).reshape(3)
    Rconv = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    R, _ = cv2.Rodrigues(aa.astype(np.float32))
    Rn = (Rconv @ R).astype(np.float32)
    aa_n, _ = cv2.Rodrigues(Rn)
    return aa_n.reshape(3).astype(np.float32)


class HO3DDataset(Dataset):
    """
    HO3D_v3 dataset (RGB-only), aligned with FreiHANDDatasetV2 / DexYCBDataset.
    """

    def __init__(
        self,
        data_split: str,
        root_dir: str,
        dataset_version: str = "v3",
        img_size: int = 256,
        img_width: int | None = None,
        input_modal: str = "RGB",
        train: bool = True,
        align_wilor_aug: bool = True,
        wilor_aug_config: Dict[str, Any] | None = None,
        bbox_source: str = "gt",
        detector_weights_path: str | None = None,
        # train/val split carved from train.txt (HO3D v3 evaluation lacks GT)
        trainval_ratio: float = 0.9,
        trainval_seed: int = 42,
        trainval_split_by: str = "sequence",  # "sequence" | "frame"
        root_index: int = 9,
        # extra aug knobs (kept for config compatibility)
        center_jitter_factor: float = 0.05,
        brightness_limit: tuple = (-0.2, 0.1),
        contrast_limit: tuple = (0.8, 1.2),
        brightness_prob: float = 0.5,
        contrast_prob: float = 0.5,
    ):
        self.data_split = str(data_split).lower()
        self.dataset_version = str(dataset_version)
        self.train = bool(train)

        # Allow passing either the dataset parent dir or the HO3D_v3 dir itself.
        candidate_root = osp.join(root_dir, f"HO3D_{self.dataset_version}")
        self.root_dir = candidate_root if osp.exists(candidate_root) else root_dir

        self.img_size = int(img_size)
        self.patch_height = self.img_size
        if img_width is None:
            img_width = int(round(self.img_size * 0.75))
        self.patch_width = int(img_width)

        self.input_modal = "RGB" if "rgb" in str(input_modal).lower() else "RGB"
        self.align_wilor_aug = bool(align_wilor_aug)
        self.wilor_aug_config = wilor_aug_config or {}

        self.bbox_source = str(bbox_source).lower()
        # IMPORTANT:
        # HO3D provides labels that are sufficient for hand crop:
        # - train/val: full 3D joints + K -> project to 2D -> bbox from keypoints
        # - evaluation/test: `handBoundingBox` in the meta pickle
        # Do NOT use a detector model here (avoid Ultralytics/YOLO dependency and keep behavior
        # consistent with label-based cropping used in this repo).
        if self.bbox_source == "detector":
            if str(os.environ.get("RANK", "0")) == "0":
                print("[warn] HO3D bbox_source='detector' is disabled; using label-based crop instead.")
            self.bbox_source = "gt"
        _ = detector_weights_path  # kept for config compatibility; intentionally unused for HO3D

        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.trainval_split_by = str(trainval_split_by).lower()
        self.root_index = int(root_index)

        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        self.datalist = self.load_data()
        print(f"HO3D Dataset loaded: {len(self.datalist)} samples from {self.data_split} split")

    def __len__(self) -> int:
        return len(self.datalist)

    @staticmethod
    def _read_split_lines(split_file: str) -> List[str]:
        out: List[str] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
        return out

    def _split_train_lines(self, train_lines: List[str]) -> Tuple[List[str], List[str]]:
        """
        Deterministically split train.txt into train/val subsets.
        - split_by=\"sequence\": group by sequence to avoid leakage, then pack sequences by #frames.
        - split_by=\"frame\": shuffle individual frames and take first ratio as train.
        """
        ratio = float(max(0.0, min(1.0, self.trainval_ratio)))
        seed = int(self.trainval_seed)
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

        seq_to_lines: Dict[str, List[str]] = {}
        for ln in train_lines:
            parts = ln.split("/")
            if len(parts) != 2:
                continue
            seq_to_lines.setdefault(parts[0], []).append(ln)

        seqs = list(seq_to_lines.keys())
        rng = random.Random(seed)
        rng.shuffle(seqs)

        total = sum(len(v) for v in seq_to_lines.values())
        target = int(total * ratio)

        train_out: List[str] = []
        val_out: List[str] = []
        count = 0
        for seq in seqs:
            bucket = seq_to_lines[seq]
            if count < target:
                train_out.extend(bucket)
                count += len(bucket)
            else:
                val_out.extend(bucket)

        if len(train_out) == 0 and len(val_out) > 0:
            train_out.append(val_out.pop(0))
        if len(val_out) == 0 and len(train_out) > 1:
            val_out.append(train_out.pop())
        return train_out, val_out

    def _cache_path(self, split_name: str) -> str:
        """Return a deterministic cache file path based on split parameters."""
        key = f"{self.root_dir}|{split_name}|{self.trainval_ratio}|{self.trainval_seed}|{self.trainval_split_by}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.root_dir, f".cache_ho3d_{split_name}_{h}.pkl")

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset annotations from HO3D_v3 meta pickles, with on-disk caching
        to avoid re-reading tens of thousands of individual pickle files on NFS.
        """
        split_name = str(self.data_split).lower()
        if split_name == "test":
            split_name = "evaluation"

        # Try loading from cache first.
        cache_file = self._cache_path(split_name)
        if osp.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    datalist = pickle.load(f)
                print(f"[ho3d] Loaded {len(datalist)} samples from cache: {cache_file}", flush=True)
                return datalist
            except Exception:
                pass  # cache corrupt, rebuild

        print(f"[ho3d] Cache miss, reading meta pickles (this may be slow on NFS) ...", flush=True)
        datalist = self._load_data_from_pickles(split_name)

        # Save cache (atomic write).
        try:
            fd, tmp = tempfile.mkstemp(dir=self.root_dir, suffix=".tmp")
            os.close(fd)
            with open(tmp, "wb") as f:
                pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_file)
        except Exception:
            pass  # non-fatal; next run will just rebuild

        return datalist

    def _load_data_from_pickles(self, split_name: str) -> List[Dict[str, Any]]:
        train_txt = osp.join(self.root_dir, "train.txt")
        eval_txt = osp.join(self.root_dir, "evaluation.txt")

        selected: List[str]
        split_folder: str
        if split_name in ("train", "val", "train_all"):
            if not osp.exists(train_txt):
                raise FileNotFoundError(f"Split file not found: {train_txt}")
            all_train = self._read_split_lines(train_txt)
            train_lines, val_lines = self._split_train_lines(all_train)
            if split_name == "train":
                selected = train_lines
            elif split_name == "val":
                selected = val_lines
            else:
                selected = all_train
            split_folder = "train"
        elif split_name == "evaluation":
            if not osp.exists(eval_txt):
                raise FileNotFoundError(f"Split file not found: {eval_txt}")
            selected = self._read_split_lines(eval_txt)
            split_folder = "evaluation"
        else:
            raise ValueError(f"Unsupported HO3D split: {self.data_split}")

        total = len(selected)
        log_interval = max(total // 10, 1)
        datalist: List[Dict[str, Any]] = []
        for i, line in enumerate(selected):
            if i % log_interval == 0:
                print(f"[ho3d] loading meta {i}/{total} ...", flush=True)

            parts = line.split("/")
            if len(parts) != 2:
                continue
            seq_name, frame_id = parts[0], parts[1]

            seq_dir = osp.join(self.root_dir, split_folder, seq_name)
            rgb_path = osp.join(seq_dir, "rgb", f"{frame_id}.jpg")
            if not osp.exists(rgb_path):
                rgb_path = osp.join(seq_dir, "rgb", f"{frame_id}.png")
            meta_path = osp.join(seq_dir, "meta", f"{frame_id}.pkl")
            if (not osp.exists(rgb_path)) or (not osp.exists(meta_path)):
                continue

            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
            except Exception as e:
                print(f"[warn] Failed to load meta: {meta_path}: {e}")
                continue

            K = np.array(meta.get("camMat", None), dtype=np.float32) if meta.get("camMat", None) is not None else None
            if K is None or K.shape != (3, 3):
                continue

            is_eval = split_folder == "evaluation"
            if is_eval:
                bbox = meta.get("handBoundingBox", None)
                root_joint = meta.get("handJoints3D", None)
                datalist.append(
                    {
                        "img_path": rgb_path,
                        "seq_name": seq_name,
                        "frame_id": frame_id,
                        "K": K,
                        "is_eval": True,
                        "handBoundingBox": bbox,
                        "handRoot3D": root_joint,
                    }
                )
                continue

            joints = meta.get("handJoints3D", None)
            pose = meta.get("handPose", None)
            beta = meta.get("handBeta", None)
            trans = meta.get("handTrans", None)
            if joints is None or pose is None or beta is None or trans is None:
                continue

            joints = np.array(joints, dtype=np.float32).reshape(21, 3)
            pose = np.array(pose, dtype=np.float32).reshape(48)
            beta = np.array(beta, dtype=np.float32).reshape(10)
            trans = np.array(trans, dtype=np.float32).reshape(3)

            datalist.append(
                {
                    "img_path": rgb_path,
                    "seq_name": seq_name,
                    "frame_id": frame_id,
                    "K": K,
                    "is_eval": False,
                    "joints_coord_cam": joints,
                    "mano_pose": pose,
                    "mano_shape": beta,
                    "mano_trans": trans,
                }
            )

        return datalist

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.datalist[idx]
        rgb_path = data["img_path"]
        rgb = cv2.imread(rgb_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f"Fail to read {rgb_path}")
        h, w = rgb.shape[:2]

        K = np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        cam_para = (fx, fy, cx, cy)

        is_eval = bool(data.get("is_eval", False))
        hand_type = "right"

        # Prepare GT (or dummy placeholders for evaluation split).
        if not is_eval:
            joints_ho3d = np.asarray(data["joints_coord_cam"], dtype=np.float32).reshape(21, 3)
            joints_std = _ho3d_cam_to_std_xyz(joints_ho3d)
            joints_mano = joints_std[HO3D_META_JOINT_MAP, :]
            joints_wilor = joints_mano[WILOR_JOINT_MAP, :]

            keypoints_3d = joints_wilor.astype(np.float32)
            keypoints_2d = _project_points(keypoints_3d, K).astype(np.float32)  # (21,2) in pixels

            center, scale_px, coord_valid = _bbox_from_keypoints(keypoints_2d, w, h)

            bbox_size = float(scale_px.max())
            denom = float(max(float(scale_px.max()), 1e-6))
            bbox_expand_factor = float(bbox_size / denom)
            scale = (scale_px / 200.0).astype(np.float32)

            # Optional center jitter (only during training).
            if self.train and self.center_jitter_factor > 0:
                jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                center = center.copy()
                center[0] += jitter_x
                center[1] += jitter_y

            pose = np.asarray(data["mano_pose"], dtype=np.float32).reshape(48)
            beta = np.asarray(data["mano_shape"], dtype=np.float32).reshape(10)
            trans = np.asarray(data["mano_trans"], dtype=np.float32).reshape(3)

            global_orient = _ho3d_cam_to_std_global_orient(pose[:3])
            hand_pose = pose[3:].copy()
            mano_shape = beta.copy()
            mano_trans = _ho3d_cam_to_std_xyz(trans).reshape(3)

            mano_params = {"global_orient": global_orient, "hand_pose": hand_pose, "betas": mano_shape}
            has_mano_params = {
                "global_orient": np.array([1.0], dtype=np.float32),
                "hand_pose": np.array([1.0], dtype=np.float32),
                "betas": np.array([1.0], dtype=np.float32),
            }
        else:
            # Evaluation meta: bbox is in image coords, and joints are incomplete (root only).
            keypoints_2d = np.zeros((21, 2), dtype=np.float32)
            keypoints_3d = np.zeros((21, 3), dtype=np.float32)
            coord_valid = np.zeros((21,), dtype=np.float32)

            bbox = data.get("handBoundingBox", None)
            bb = _sanitize_bbox_xyxy(np.array(bbox, dtype=np.float32), w, h) if bbox is not None else None
            if bb is not None:
                x1, y1, x2, y2 = bb.tolist()
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                scale_px = np.array([(x2 - x1) * 1.2, (y2 - y1) * 1.2], dtype=np.float32)
            else:
                center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
                scale_px = np.array([float(max(w, h)), float(max(w, h))], dtype=np.float32)

            bbox_size = float(scale_px.max())
            denom = float(max(float(scale_px.max()), 1e-6))
            bbox_expand_factor = float(bbox_size / denom)
            scale = (scale_px / 200.0).astype(np.float32)

            mano_params = {
                "global_orient": np.zeros((3,), dtype=np.float32),
                "hand_pose": np.zeros((45,), dtype=np.float32),
                "betas": np.zeros((10,), dtype=np.float32),
            }
            has_mano_params = {
                "global_orient": np.array([0.0], dtype=np.float32),
                "hand_pose": np.array([0.0], dtype=np.float32),
                "betas": np.array([0.0], dtype=np.float32),
            }
            mano_shape = mano_params["betas"].copy()
            mano_trans = np.zeros((3,), dtype=np.float32)

        flip_perm = list(range(21))
        img_patch, kp2d_norm, kp3d_aug, aug_mano_params, _has_params, _, trans = get_example(
            rgb,
            float(center[0]),
            float(center[1]),
            float(bbox_size),
            float(bbox_size),
            keypoints_2d.astype(np.float32),
            keypoints_3d.astype(np.float32),
            mano_params,
            has_mano_params,
            flip_perm,
            int(self.patch_width),
            int(self.patch_height),
            None,
            None,
            do_augment=self.train,
            is_right=True,
            augm_config=self.wilor_aug_config,
            is_bgr=True,
            return_trans=True,
        )

        # Normalize image patch to ImageNet stats (WiLoR/HaMeR convention).
        imgRGB_01 = torch.from_numpy(img_patch).float() / 255.0
        if self.train:
            if np.random.rand() < self.brightness_prob:
                delta = float(np.random.uniform(self.brightness_limit[0], self.brightness_limit[1]))
                imgRGB_01 = torch.clamp(imgRGB_01 + delta, 0.0, 1.0)
            if np.random.rand() < self.contrast_prob:
                factor = float(np.random.uniform(self.contrast_limit[0], self.contrast_limit[1]))
                mean_val = imgRGB_01.mean(dim=[1, 2], keepdim=True)
                imgRGB_01 = torch.clamp((imgRGB_01 - mean_val) * factor + mean_val, 0.0, 1.0)
        imgRGB = (imgRGB_01 - self.imagenet_mean) / self.imagenet_std

        # Update intrinsics after affine warp into the patch.
        trans_3x3 = np.eye(3, dtype=np.float32)
        trans_3x3[:2, :] = trans
        K_patch = trans_3x3 @ K
        cam_para = (float(K_patch[0, 0]), float(K_patch[1, 1]), float(K_patch[0, 2]), float(K_patch[1, 2]))

        # Validity masks in patch frame.
        trans_uv01 = kp2d_norm.copy()
        trans_uv01[:, 0] = trans_uv01[:, 0] + 0.5
        trans_uv01[:, 1] = trans_uv01[:, 1] + 0.5
        trans_coord_valid = ((trans_uv01 > 0).astype(np.float32) * (trans_uv01 < 1).astype(np.float32))
        trans_coord_valid = (trans_coord_valid[:, 0] * trans_coord_valid[:, 1]).astype(np.float32)
        trans_coord_valid *= coord_valid.astype(np.float32)

        xyz_valid = 1.0
        ri = int(self.root_index)
        if (trans_coord_valid[ri] == 0) and (trans_coord_valid[0] == 0):
            xyz_valid = 0.0

        mano_pose = np.concatenate([aug_mano_params["global_orient"], aug_mano_params["hand_pose"]], axis=0).astype(
            np.float32
        )

        mano_params_is_axis_angle = {"global_orient": True, "hand_pose": True, "betas": False}

        return {
            "rgb": imgRGB,
            "keypoints_2d": torch.from_numpy(kp2d_norm.astype(np.float32)).float(),
            "keypoints_3d": torch.from_numpy(kp3d_aug.astype(np.float32)).float(),
            "mano_params": aug_mano_params,
            "mano_pose": torch.from_numpy(mano_pose).float(),
            "mano_shape": torch.from_numpy(aug_mano_params["betas"].astype(np.float32)).float(),
            "mano_trans": torch.from_numpy(mano_trans.astype(np.float32)).float(),
            "cam_param": torch.tensor(cam_para, dtype=torch.float32),
            "box_center": torch.from_numpy(center.astype(np.float32)),
            "box_size": torch.tensor(float(bbox_size), dtype=torch.float32),
            "bbox_expand_factor": torch.tensor(float(bbox_expand_factor), dtype=torch.float32),
            "_scale": torch.from_numpy(scale.astype(np.float32)),
            "mano_params_is_axis_angle": mano_params_is_axis_angle,
            "has_mano_params": has_mano_params,
            "uv_valid": trans_coord_valid.astype(np.float32),
            "xyz_valid": int(xyz_valid),
            "hand_type": hand_type,
            "is_right": 1.0,
            # Debug/meta (safe for DataLoader collation; training ignores these keys).
            "img_path": rgb_path,
            "seq_name": str(data.get("seq_name", "")),
            "frame_id": str(data.get("frame_id", "")),
        }


def main():  # pragma: no cover
    """Debug/inspect HO3DDataset by printing a few samples."""
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

    parser = argparse.ArgumentParser(description="Print/inspect HO3DDataset contents.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to HO3D root directory.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "evaluation", "train_all"],
        help="Dataset split.",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Output image size.")
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

    ds = HO3DDataset(
        data_split=args.split,
        root_dir=args.root_dir,
        img_size=args.img_size,
        train=(args.split == "train"),
        wilor_aug_config=default_wilor_aug_config,
    )
    random.seed(args.seed)
    n = min(args.num_samples, len(ds))
    indices = random.sample(range(len(ds)), k=n) if n > 0 else []
    print(f"\n=== HO3DDataset summary === split={ds.data_split} len={len(ds)} root_dir={ds.root_dir}\n")
    for i, j in enumerate(indices):
        print(f"\n--- Sample {i+1}/{len(indices)} | idx={j} ---")
        sample = ds[j]
        for k in sorted(sample.keys()):
            print(f"{k}: {_describe_value(sample[k])}")


if __name__ == "__main__":  # pragma: no cover
    main()

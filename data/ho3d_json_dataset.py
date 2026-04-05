# -*- coding: utf-8 -*-
"""
HO3D JSON dataset loader.

This loader reads train/test splits directly from COCO-style JSON files
(e.g. HO3D-train-normalized.json / HO3D-evaluation-*.json) and returns the
same training keys used by this repo's other dataset loaders.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
import random
import tempfile
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    from .utils import WILOR_JOINT_MAP, get_example
except ImportError:  # pragma: no cover
    from utils import WILOR_JOINT_MAP, get_example


# HO3D meta joint order mapping (same as ho3d_dataset.py)
HO3D_META_JOINT_MAP: List[int] = list(range(21))


def _project_points(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    uvw = (K @ xyz.T).T
    uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-7)
    return uv.astype(np.float32)


def _ho3d_cam_to_std_xyz(xyz: np.ndarray) -> np.ndarray:
    """Convert HO3D camera coords to repo standard camera coords: (x,y,z)->(x,-y,-z)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    out = xyz.copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _ho3d_cam_to_std_global_orient(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float32).reshape(3)
    rconv = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    r, _ = cv2.Rodrigues(aa.astype(np.float32))
    rn = (rconv @ r).astype(np.float32)
    aa_n, _ = cv2.Rodrigues(rn)
    return aa_n.reshape(3).astype(np.float32)


def _sanitize_bbox_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray | None:
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


def _bbox_from_keypoints(keypoints_2d: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        scale_px = 2.0 * (mx - mn)
    else:
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale_px = np.array([w, h], dtype=np.float32)
    scale_px = np.maximum(scale_px, 2.0)
    return center.astype(np.float32), scale_px.astype(np.float32), coord_valid


def _seq_from_file_name(file_name: str) -> str:
    parts = str(file_name).strip("/").split("/")
    if len(parts) >= 4 and parts[-2].lower() == "rgb":
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return str(file_name)


def _iter_json_array_objects(json_path: str, key: str, chunk_size: int = 2 * 1024 * 1024) -> Iterable[Dict[str, Any]]:
    token = f"\"{key}\""
    decoder = json.JSONDecoder()

    with open(json_path, "r", encoding="utf-8") as f:
        # Seek to key's array start.
        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise KeyError(f"Key '{key}' not found in JSON: {json_path}")
            buf += chunk
            i = buf.find(token)
            if i < 0:
                buf = buf[-(len(token) + 64):]
                continue
            j = buf.find("[", i + len(token))
            if j < 0:
                buf = buf[i:]
                continue
            buf = buf[j + 1:]
            break

        # Parse objects one by one from the array.
        done = False
        while not done:
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    done = True
                    pos += 1
                    break
                try:
                    obj, idx = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = idx

            buf = buf[pos:]
            if done:
                break

            chunk = f.read(chunk_size)
            if not chunk:
                if buf.strip():
                    raise ValueError(f"Unexpected EOF while parsing key '{key}' in JSON: {json_path}")
                break
            buf += chunk


class HO3DJsonDataset(Dataset):
    """
    HO3D loader that reads train/test split from JSON files.
    """

    def __init__(
        self,
        data_split: str,
        root_dir: str,
        train_json_path: str,
        test_json_path: str,
        img_size: int = 256,
        img_width: int | None = None,
        train: bool = True,
        align_wilor_aug: bool = True,
        wilor_aug_config: Dict[str, Any] | None = None,
        bbox_source: str = "gt",
        detector_weights_path: str | None = None,
        trainval_ratio: float = 0.9,
        trainval_seed: int = 42,
        trainval_split_by: str = "sequence",
        root_index: int = 9,
        json_kp3d_unit: str = "auto",  # auto | m | mm
        json_kp3d_scale: float = 1.0,
        json_convert_xyz: bool = False,  # (x,y,z) -> (x,-y,-z)
        center_jitter_factor: float = 0.05,
        brightness_limit: tuple = (-0.2, 0.1),
        contrast_limit: tuple = (0.8, 1.2),
        brightness_prob: float = 0.5,
        contrast_prob: float = 0.5,
    ):
        split_name = str(data_split).lower()
        # JSON mode is intentionally strict: only the configured train JSON is used for
        # training, and every non-train split uses the configured test/evaluation JSON.
        self.data_split = "train" if split_name == "train" else "evaluation"

        self.root_dir = root_dir
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path

        self.train = bool(train)
        self.align_wilor_aug = bool(align_wilor_aug)
        self.wilor_aug_config = wilor_aug_config or {}
        self.root_index = int(root_index)

        self.bbox_source = str(bbox_source).lower()
        if self.bbox_source == "detector":
            if str(os.environ.get("RANK", "0")) == "0":
                print("[warn] HO3D JSON loader uses label-based crop; ignoring bbox_source='detector'.")
            self.bbox_source = "gt"
        _ = detector_weights_path

        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.trainval_split_by = str(trainval_split_by).lower()

        self.json_kp3d_unit = str(json_kp3d_unit).lower()
        self.json_kp3d_scale = float(json_kp3d_scale)
        self.json_convert_xyz = bool(json_convert_xyz)

        self.img_size = int(img_size)
        self.patch_height = self.img_size
        if img_width is None:
            img_width = int(round(self.img_size * 0.75))
        self.patch_width = int(img_width)

        self.center_jitter_factor = float(center_jitter_factor)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = float(brightness_prob)
        self.contrast_prob = float(contrast_prob)

        self.transform = transforms.ToTensor()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        self.datalist = self.load_data()
        print(f"HO3D JSON Dataset loaded: {len(self.datalist)} samples from {self.data_split} split")

    def __len__(self) -> int:
        return len(self.datalist)

    def _cache_path(self) -> str:
        src = self.train_json_path if self.data_split == "train" else self.test_json_path
        st = os.stat(src)
        # Bump this when datalist semantics change (prevents stale, incompatible cache reuse).
        cache_version = "v4_meta_mano_mesh"
        key = (
            f"{cache_version}|{self.data_split}|{self.root_dir}|{self.train_json_path}|{self.test_json_path}|"
            f"{self.trainval_ratio}|{self.trainval_seed}|{self.trainval_split_by}|"
            f"{self.json_kp3d_unit}|{self.json_kp3d_scale}|{self.json_convert_xyz}|"
            f"{st.st_size}|{st.st_mtime}"
        )
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return osp.join(self.root_dir, f".cache_ho3d_json_{self.data_split}_{h}.pkl")

    def _split_train_ids(self, image_items: Sequence[Tuple[int, str]]) -> Tuple[List[int], List[int]]:
        ratio = float(max(0.0, min(1.0, self.trainval_ratio)))
        if len(image_items) == 0:
            return [], []
        if ratio <= 0.0:
            return [], [int(i) for i, _ in image_items]
        if ratio >= 1.0:
            return [int(i) for i, _ in image_items], []

        split_by = self.trainval_split_by if self.trainval_split_by in ("sequence", "frame") else "sequence"
        rng = random.Random(self.trainval_seed)

        if split_by == "frame":
            ids = [int(i) for i, _ in image_items]
            rng.shuffle(ids)
            cut = int(len(ids) * ratio)
            return ids[:cut], ids[cut:]

        seq_to_ids: Dict[str, List[int]] = {}
        for img_id, file_name in image_items:
            seq = _seq_from_file_name(file_name)
            seq_to_ids.setdefault(seq, []).append(int(img_id))

        seqs = list(seq_to_ids.keys())
        rng.shuffle(seqs)
        total = sum(len(v) for v in seq_to_ids.values())
        target = int(total * ratio)

        train_ids: List[int] = []
        val_ids: List[int] = []
        count = 0
        for seq in seqs:
            bucket = seq_to_ids[seq]
            if count < target:
                train_ids.extend(bucket)
                count += len(bucket)
            else:
                val_ids.extend(bucket)

        if len(train_ids) == 0 and len(val_ids) > 0:
            train_ids.append(val_ids.pop(0))
        if len(val_ids) == 0 and len(train_ids) > 1:
            val_ids.append(train_ids.pop())
        return train_ids, val_ids

    def _xyz3d_postprocess(self, xyz3d: np.ndarray) -> np.ndarray:
        xyz3d = np.asarray(xyz3d, dtype=np.float32).reshape(-1, 3)

        unit = self.json_kp3d_unit
        if unit == "mm":
            xyz3d = xyz3d / 1000.0
        elif unit == "auto":
            # Heuristic: values with very large depth are likely millimeters.
            med = float(np.median(np.abs(xyz3d[:, 2]))) if xyz3d.shape[0] > 0 else 0.0
            if med > 10.0:
                xyz3d = xyz3d / 1000.0
        elif unit in ("m", "meter", "meters"):
            pass
        else:
            raise ValueError(f"Unsupported json_kp3d_unit: {self.json_kp3d_unit}")

        xyz3d = xyz3d * self.json_kp3d_scale
        if self.json_convert_xyz:
            xyz3d = xyz3d.copy()
            xyz3d[:, 1] *= -1.0
            xyz3d[:, 2] *= -1.0
        return xyz3d.astype(np.float32)

    def _kp3d_postprocess(self, kp3d: np.ndarray) -> np.ndarray:
        return self._xyz3d_postprocess(kp3d)

    def _build_datalist_from_json(self, json_path: str, split_mode: str) -> List[Dict[str, Any]]:
        images: Dict[int, Dict[str, Any]] = {}
        image_items: List[Tuple[int, str]] = []

        for i, img in enumerate(_iter_json_array_objects(json_path, "images")):
            img_id = int(img["id"])
            file_name = str(img["file_name"])
            images[img_id] = {
                "file_name": file_name,
                "width": int(img.get("width", 0)),
                "height": int(img.get("height", 0)),
            }
            image_items.append((img_id, file_name))
            if i > 0 and i % 50000 == 0 and str(os.environ.get("RANK", "0")) == "0":
                print(f"[ho3d_json] parsed images: {i}", flush=True)

        if split_mode == "train":
            selected_ids = {int(i) for i, _ in image_items}
        elif split_mode in ("evaluation", "test"):
            selected_ids = {int(i) for i, _ in image_items}
        else:
            raise ValueError(f"Unsupported split mode: {split_mode}")

        # For train, JSON is used for indexing; metric supervision comes from HO3D meta/*.pkl.
        # Avoid scanning huge `annotations` here to reduce startup time and perceived "hang".
        if split_mode == "train":
            ordered_ids = [int(i) for i, _ in image_items if int(i) in selected_ids]
            total = len(ordered_ids)
            datalist: List[Dict[str, Any]] = []
            n_missing_meta = 0
            n_invalid_meta = 0

            for j, img_id in enumerate(ordered_ids):
                img_info = images.get(img_id, None)
                if img_info is None:
                    continue

                file_name = img_info["file_name"]
                img_path = file_name if osp.isabs(file_name) else osp.join(self.root_dir, file_name)
                path_parts = str(file_name).strip("/").split("/")
                frame_stem = osp.splitext(path_parts[-1])[0] if len(path_parts) >= 1 else ""
                split_folder = ""
                seq_name = ""
                if len(path_parts) >= 4 and path_parts[-2].lower() == "rgb":
                    split_folder = str(path_parts[-4])
                    seq_name = str(path_parts[-3])
                meta_path = (
                    osp.join(self.root_dir, split_folder, seq_name, "meta", f"{frame_stem}.pkl")
                    if split_folder and seq_name and frame_stem
                    else ""
                )
                if (not meta_path) or (not osp.exists(meta_path)):
                    n_missing_meta += 1
                    continue

                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                except Exception:
                    n_invalid_meta += 1
                    continue

                joints = meta.get("handJoints3D", None)
                pose = meta.get("handPose", None)
                beta = meta.get("handBeta", None)
                trans = meta.get("handTrans", None)
                k_meta = meta.get("camMat", None)
                if (
                    joints is None
                    or pose is None
                    or beta is None
                    or trans is None
                    or k_meta is None
                ):
                    n_invalid_meta += 1
                    continue

                datalist.append(
                    {
                        "img_path": img_path,
                        "file_name": file_name,
                        "image_id": img_id,
                        "width": int(img_info.get("width", 0)),
                        "height": int(img_info.get("height", 0)),
                        "K": np.asarray(k_meta, dtype=np.float32).reshape(3, 3),
                        "is_eval": False,
                        "joints_coord_cam": np.asarray(joints, dtype=np.float32).reshape(21, 3),
                        "mano_pose": np.asarray(pose, dtype=np.float32).reshape(48),
                        "mano_shape": np.asarray(beta, dtype=np.float32).reshape(10),
                        "mano_trans": np.asarray(trans, dtype=np.float32).reshape(3),
                        "is_right": 1.0,
                    }
                )

                if j > 0 and j % 5000 == 0 and str(os.environ.get("RANK", "0")) == "0":
                    print(
                        f"[ho3d_json] loading train meta: {j}/{total}, kept={len(datalist)}, "
                        f"missing_meta={n_missing_meta}, invalid_meta={n_invalid_meta}",
                        flush=True,
                    )

            if str(os.environ.get("RANK", "0")) == "0":
                print(
                    f"[ho3d_json] datalist source summary: "
                    f"meta_train={len(datalist)} missing_meta={n_missing_meta} invalid_meta={n_invalid_meta} "
                    f"json_fallback=0"
                )
            return datalist

        datalist: List[Dict[str, Any]] = []
        n_meta_eval = 0
        n_json_fallback = 0
        for i, ann in enumerate(_iter_json_array_objects(json_path, "annotations")):
            img_id = int(ann.get("image_id", ann.get("id", -1)))
            if img_id not in selected_ids:
                continue
            img_info = images.get(img_id, None)
            if img_info is None:
                continue

            k_raw = ann.get("camera_matrix", None)
            if k_raw is None:
                continue
            K = np.asarray(k_raw, dtype=np.float32).reshape(3, 3)

            kp3_raw = ann.get("keypoints_3d", None)
            if kp3_raw is None:
                continue
            kp3d = self._kp3d_postprocess(np.asarray(kp3_raw, dtype=np.float32).reshape(-1, 3))
            kp2_pack = ann.get("keypoints", None)
            if kp2_pack is not None:
                kp2_pack = np.asarray(kp2_pack, dtype=np.float32).reshape(-1, 3)
                kp2d = kp2_pack[:, :2]
                kp2d_valid = (kp2_pack[:, 2] > 0).astype(np.float32)
            else:
                # Fallback: project from 3D if keypoints are missing.
                uvw = (K @ kp3d.T).T
                kp2d = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
                kp2d_valid = np.ones((kp2d.shape[0],), dtype=np.float32)

            bbox_xyxy = None
            bbox_xywh = ann.get("bbox", None)
            if bbox_xywh is not None and len(bbox_xywh) == 4:
                x, y, bw, bh = [float(v) for v in bbox_xywh]
                bbox_xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)

            file_name = img_info["file_name"]
            img_path = file_name if osp.isabs(file_name) else osp.join(self.root_dir, file_name)
            path_parts = str(file_name).strip("/").split("/")
            frame_stem = osp.splitext(path_parts[-1])[0] if len(path_parts) >= 1 else ""
            split_folder = ""
            seq_name = ""
            if len(path_parts) >= 4 and path_parts[-2].lower() == "rgb":
                split_folder = str(path_parts[-4])
                seq_name = str(path_parts[-3])
            meta_path = (
                osp.join(self.root_dir, split_folder, seq_name, "meta", f"{frame_stem}.pkl")
                if split_folder and seq_name and frame_stem
                else ""
            )
            meta = None
            if meta_path and osp.exists(meta_path):
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                except Exception:
                    meta = None

            # Evaluation split may only provide bbox/root from meta.
            if (split_mode in ("evaluation", "test")) and isinstance(meta, dict):
                bbox = meta.get("handBoundingBox", None)
                if bbox is not None:
                    datalist.append(
                        {
                            "img_path": img_path,
                            "file_name": file_name,
                            "image_id": img_id,
                            "width": int(img_info.get("width", 0)),
                            "height": int(img_info.get("height", 0)),
                            "K": K,
                            "is_eval": True,
                            "handBoundingBox": bbox,
                            "is_right": float(ann.get("is_right", 1)),
                        }
                    )
                    n_meta_eval += 1
                    continue

            datalist.append(
                {
                    "img_path": img_path,
                    "file_name": file_name,
                    "image_id": img_id,
                    "width": int(img_info.get("width", 0)),
                    "height": int(img_info.get("height", 0)),
                    "K": K,
                    "bbox_xyxy": bbox_xyxy,
                    "keypoints_2d": kp2d.astype(np.float32),
                    "keypoints_2d_valid": kp2d_valid.astype(np.float32),
                    "keypoints_3d": kp3d.astype(np.float32),
                    "is_right": float(ann.get("is_right", 1)),
                }
            )
            n_json_fallback += 1
            if i > 0 and i % 50000 == 0 and str(os.environ.get("RANK", "0")) == "0":
                print(f"[ho3d_json] parsed annotations: {i}, kept={len(datalist)}", flush=True)

        if str(os.environ.get("RANK", "0")) == "0":
            print(
                f"[ho3d_json] datalist source summary: meta_eval={n_meta_eval} json_fallback={n_json_fallback}"
            )
        return datalist

    def load_data(self) -> List[Dict[str, Any]]:
        if self.data_split == "train":
            json_path = self.train_json_path
            split_mode = "train"
        else:
            json_path = self.test_json_path
            split_mode = "evaluation"

        if not json_path or (not osp.exists(json_path)):
            raise FileNotFoundError(
                f"HO3D JSON file not found for split '{self.data_split}': {json_path}. "
                "Set paths.ho3d_train_json / paths.ho3d_test_json in config."
            )

        cache_file = self._cache_path()
        if osp.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    datalist = pickle.load(f)
                print(f"[ho3d_json] loaded {len(datalist)} samples from cache: {cache_file}", flush=True)
                return datalist
            except Exception as e:
                # Common after NumPy major-version switches (e.g., numpy._core.* module path changes).
                # Fallback: invalidate stale cache and rebuild from JSON.
                try:
                    os.remove(cache_file)
                except Exception:
                    pass
                print(
                    f"[ho3d_json] cache load failed, invalidating cache and rebuilding: {cache_file} | "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        # In DDP, let rank0 build cache first to avoid all ranks parsing huge JSON at once.
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            wait_s = 1200.0
            t0 = time.time()
            while time.time() - t0 < wait_s:
                if osp.exists(cache_file):
                    try:
                        with open(cache_file, "rb") as f:
                            datalist = pickle.load(f)
                        print(f"[ho3d_json] rank{rank} loaded cache: {cache_file}", flush=True)
                        return datalist
                    except Exception as e:
                        # If rank0 wrote a stale/corrupted cache, force regeneration.
                        try:
                            os.remove(cache_file)
                        except Exception:
                            pass
                        print(
                            f"[ho3d_json] rank{rank} cache load failed, waiting for rebuilt cache: {cache_file} | "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                time.sleep(1.0)

        print(f"[ho3d_json] cache miss; parsing JSON: {json_path}", flush=True)
        datalist = self._build_datalist_from_json(json_path=json_path, split_mode=split_mode)

        try:
            fd, tmp = tempfile.mkstemp(dir=self.root_dir, suffix=".tmp")
            os.close(fd)
            with open(tmp, "wb") as f:
                pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_file)
        except Exception:
            pass

        return datalist

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.datalist[idx]
        rgb_path = data["img_path"]
        rgb = cv2.imread(rgb_path)
        if not isinstance(rgb, np.ndarray):
            raise IOError(f"Fail to read {rgb_path}")
        h, w = rgb.shape[:2]

        K = np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
        cam_para = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        is_eval = bool(data.get("is_eval", False))
        has_meta_train_gt = ("joints_coord_cam" in data) and (not is_eval)

        if has_meta_train_gt:
            # Match HO3DDataset: metric 3D supervision + MANO params from original meta.
            joints_ho3d = np.asarray(data["joints_coord_cam"], dtype=np.float32).reshape(21, 3)
            joints_std = _ho3d_cam_to_std_xyz(joints_ho3d)
            joints_mano = joints_std[HO3D_META_JOINT_MAP, :]
            keypoints_3d = joints_mano[WILOR_JOINT_MAP, :].astype(np.float32)
            keypoints_2d = _project_points(keypoints_3d, K).astype(np.float32)

            center, scale_px, coord_valid = _bbox_from_keypoints(keypoints_2d, w, h)

            bbox_size = float(scale_px.max())
            denom = float(max(float(scale_px.max()), 1e-6))
            bbox_expand_factor = float(bbox_size / denom)
            scale = (scale_px / 200.0).astype(np.float32)

            if self.train and self.center_jitter_factor > 0:
                jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                center = center.copy()
                center[0] += jitter_x
                center[1] += jitter_y

            pose = np.asarray(data["mano_pose"], dtype=np.float32).reshape(48)
            beta = np.asarray(data["mano_shape"], dtype=np.float32).reshape(10)
            trans = np.asarray(data["mano_trans"], dtype=np.float32).reshape(3)

            mano_params = {
                "global_orient": _ho3d_cam_to_std_global_orient(pose[:3]),
                "hand_pose": pose[3:].copy(),
                "betas": beta.copy(),
            }
            has_mano_params = {
                "global_orient": np.array([1.0], dtype=np.float32),
                "hand_pose": np.array([1.0], dtype=np.float32),
                "betas": np.array([1.0], dtype=np.float32),
            }
            mano_trans = _ho3d_cam_to_std_xyz(trans).reshape(3)
        elif is_eval:
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
            mano_trans = np.zeros((3,), dtype=np.float32)
        else:
            keypoints_2d = np.asarray(data["keypoints_2d"], dtype=np.float32).reshape(-1, 2)
            keypoints_3d = np.asarray(data["keypoints_3d"], dtype=np.float32).reshape(-1, 3)
            kp_valid = np.asarray(
                data.get("keypoints_2d_valid", np.ones((21,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)

            center, scale_px, coord_valid_img = _bbox_from_keypoints(keypoints_2d, w, h)
            coord_valid = (coord_valid_img * kp_valid).astype(np.float32)

            bbox_xyxy = data.get("bbox_xyxy", None)
            if np.sum(coord_valid > 0.5) < 2 and bbox_xyxy is not None:
                bb = _sanitize_bbox_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), w, h)
                if bb is not None:
                    x1, y1, x2, y2 = bb.tolist()
                    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                    bw = max(x2 - x1, 2.0)
                    bh = max(y2 - y1, 2.0)
                    scale_px = np.array([2.0 * bw, 2.0 * bh], dtype=np.float32)

            bbox_size = float(scale_px.max())
            denom = float(max(float(scale_px.max()), 1e-6))
            bbox_expand_factor = float(bbox_size / denom)
            scale = (scale_px / 200.0).astype(np.float32)

            if self.train and self.center_jitter_factor > 0:
                jitter_x = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                jitter_y = np.random.uniform(-self.center_jitter_factor, self.center_jitter_factor) * bbox_size
                center = center.copy()
                center[0] += jitter_x
                center[1] += jitter_y

            # JSON fallback has no reliable MANO params; mask out MANO losses.
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
            mano_trans = np.zeros((3,), dtype=np.float32)

        flip_perm = list(range(21))
        img_patch, kp2d_norm, kp3d_aug, aug_mano_params, _has_params, _, trans, rot, do_flip = get_example(
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
            is_right=bool(float(data.get("is_right", 1.0)) > 0.5),
            augm_config=self.wilor_aug_config,
            is_bgr=True,
            return_trans=True,
            return_aug_params=True,
        )

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

        trans_3x3 = np.eye(3, dtype=np.float32)
        trans_3x3[:2, :] = trans
        K_patch = trans_3x3 @ K
        cam_para = (float(K_patch[0, 0]), float(K_patch[1, 1]), float(K_patch[0, 2]), float(K_patch[1, 2]))

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

        if has_meta_train_gt:
            mano_pose = np.concatenate([aug_mano_params["global_orient"], aug_mano_params["hand_pose"]], axis=0).astype(
                np.float32
            )
            mano_shape = aug_mano_params["betas"].astype(np.float32)
        else:
            mano_pose = np.concatenate([mano_params["global_orient"], mano_params["hand_pose"]], axis=0).astype(np.float32)
            mano_shape = mano_params["betas"].astype(np.float32)

        mano_params_is_axis_angle = {"global_orient": True, "hand_pose": True, "betas": False}

        out = {
            "rgb": imgRGB,
            "keypoints_2d": torch.from_numpy(kp2d_norm.astype(np.float32)).float(),
            "keypoints_3d": torch.from_numpy(kp3d_aug.astype(np.float32)).float(),
            "mano_params": aug_mano_params,
            "mano_pose": torch.from_numpy(mano_pose).float(),
            "mano_shape": torch.from_numpy(mano_shape).float(),
            "mano_trans": torch.from_numpy(mano_trans).float(),
            "cam_param": torch.tensor(cam_para, dtype=torch.float32),
            "box_center": torch.from_numpy(center.astype(np.float32)),
            "box_size": torch.tensor(float(bbox_size), dtype=torch.float32),
            "bbox_expand_factor": torch.tensor(float(bbox_expand_factor), dtype=torch.float32),
            "_scale": torch.from_numpy(scale.astype(np.float32)),
            "mano_params_is_axis_angle": mano_params_is_axis_angle,
            "has_mano_params": has_mano_params,
            "uv_valid": trans_coord_valid.astype(np.float32),
            "xyz_valid": int(xyz_valid),
            "hand_type": "right" if float(data.get("is_right", 1.0)) > 0.5 else "left",
            "is_right": float(data.get("is_right", 1.0)),
            "img_path": rgb_path,
            "image_id": int(data.get("image_id", idx)),
            "file_name": str(data.get("file_name", "")),
        }
        return out

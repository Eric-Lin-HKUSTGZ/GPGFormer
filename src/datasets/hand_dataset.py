import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HandDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotation_path: Optional[str] = None,
        resize_to: Optional[Sequence[int]] = None,
        train: bool = True,
        color_factor: float = 0.0,
        color_aug_prob: float = 0.0,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ):
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.resize_to = resize_to
        self.train = train
        self.color_factor = color_factor
        self.color_aug_prob = color_aug_prob
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32) if normalize_mean is not None else None
        self.normalize_std = np.array(normalize_std, dtype=np.float32) if normalize_std is not None else None
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        if self.annotation_path is None:
            images = []
            for fn in os.listdir(self.root_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append({"image_path": os.path.join(self.root_dir, fn)})
            return images
        with open(self.annotation_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_augment_color(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self.train or self.color_factor <= 0.0:
            return img_bgr
        if np.random.rand() >= self.color_aug_prob:
            return img_bgr
        c_up = 1.0 + self.color_factor
        c_low = 1.0 - self.color_factor
        color_scale = np.array(
            [np.random.uniform(c_low, c_up),
             np.random.uniform(c_low, c_up),
             np.random.uniform(c_low, c_up)],
            dtype=np.float32
        )
        img_bgr = np.clip(img_bgr.astype(np.float32) * color_scale[None, None, :], 0, 255)
        return img_bgr.astype(np.uint8)

    def _normalize_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        if self.normalize_mean is None or self.normalize_std is None:
            return img_rgb
        mean = self.normalize_mean.reshape(1, 1, 3)
        std = self.normalize_std.reshape(1, 1, 3)
        return (img_rgb - mean) / std

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise IOError(f"Failed to read image: {img_path}")
        if self.resize_to is not None:
            img_bgr = cv2.resize(
                img_bgr,
                (int(self.resize_to[1]), int(self.resize_to[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        img_bgr = self._maybe_augment_color(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = self._normalize_rgb(img_rgb)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        out = {
            "rgb": img_tensor,
            "image_bgr": img_bgr,
            "image_path": img_path
        }

        if "bbox" in sample:
            out["bbox"] = torch.tensor(sample["bbox"], dtype=torch.float32)
        if "mano_pose" in sample:
            out["mano_pose"] = torch.tensor(sample["mano_pose"], dtype=torch.float32)
        if "mano_shape" in sample:
            out["mano_shape"] = torch.tensor(sample["mano_shape"], dtype=torch.float32)
        if "cam_t" in sample:
            out["cam_t"] = torch.tensor(sample["cam_t"], dtype=torch.float32)

        return out

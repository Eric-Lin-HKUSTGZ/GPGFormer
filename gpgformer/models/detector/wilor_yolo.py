from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class WiLoRDetectorConfig:
    weights_path: str
    conf: float = 0.3
    iou: float = 0.6


class WiLoRYOLODetector:
    """
    WiLoR detector wrapper (Ultralytics YOLO).
    Returns bbox in xyxy pixel coordinates on the original image.
    """

    def __init__(self, cfg: WiLoRDetectorConfig, device: Optional[str | int | torch.device] = None):
        from ultralytics import YOLO

        weights = Path(cfg.weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {weights}")
        self.cfg = cfg
        self.detector = YOLO(str(weights))
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: Optional[str | int | torch.device]) -> str | int:
        """
        Ultralytics `device` accepts:
        - 'cpu'
        - int GPU index (relative to visible devices)
        - strings like '0', '0,1', 'cuda:0'
        """
        if device is not None:
            if isinstance(device, torch.device):
                return str(device)
            return device

        # Optional override via env var
        env = os.environ.get("GPGFORMER_DETECTOR_DEVICE", None)
        if env is not None and len(env) > 0:
            return env

        if not torch.cuda.is_available():
            return "cpu"

        # In DDP training, running the detector on GPU can permanently occupy VRAM after the first
        # validation call, causing the *next epoch*'s training forward to OOM. Default to CPU in
        # distributed mode unless the user explicitly overrides via GPGFORMER_DETECTOR_DEVICE.
        if os.environ.get("RANK", None) is not None and os.environ.get("WORLD_SIZE", None) is not None:
            return "cpu"

        # Prefer LOCAL_RANK so each DDP process uses its own GPU
        lr = os.environ.get("LOCAL_RANK", None)
        if lr is not None:
            try:
                return int(lr)
            except Exception:
                pass
        return int(torch.cuda.current_device())

    def __call__(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            img_bgr: HxWx3 uint8 BGR
        Returns:
            bbox_xyxy float32: (4,) or None
        """
        try:
            det = self.detector(
                img_bgr,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                device=self.device,
                verbose=False,
            )[0]
        except torch.OutOfMemoryError:
            # Fallback: run detector on CPU if GPU is out of memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = "cpu"
            det = self.detector(
                img_bgr,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                device="cpu",
                verbose=False,
            )[0]
        except RuntimeError as e:
            # Some Ultralytics/PyTorch OOMs surface as RuntimeError with "out of memory"
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                det = self.detector(
                    img_bgr,
                    conf=float(self.cfg.conf),
                    iou=float(self.cfg.iou),
                    device="cpu",
                    verbose=False,
                )[0]
            else:
                raise
        if det is None or det.boxes is None or len(det.boxes) == 0:
            return None
        boxes = det.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = det.boxes.conf.detach().cpu().numpy().astype(np.float32)
        idx = int(np.argmax(scores))
        return boxes[idx]





        if not torch.cuda.is_available():
            return "cpu"

        # In DDP training, running the detector on GPU can permanently occupy VRAM after the first
        # validation call, causing the *next epoch*'s training forward to OOM. Default to CPU in
        # distributed mode unless the user explicitly overrides via GPGFORMER_DETECTOR_DEVICE.
        if os.environ.get("RANK", None) is not None and os.environ.get("WORLD_SIZE", None) is not None:
            return "cpu"

        # Prefer LOCAL_RANK so each DDP process uses its own GPU
        lr = os.environ.get("LOCAL_RANK", None)
        if lr is not None:
            try:
                return int(lr)
            except Exception:
                pass
        return int(torch.cuda.current_device())

    def __call__(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            img_bgr: HxWx3 uint8 BGR
        Returns:
            bbox_xyxy float32: (4,) or None
        """
        try:
            det = self.detector(
                img_bgr,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                device=self.device,
                verbose=False,
            )[0]
        except torch.OutOfMemoryError:
            # Fallback: run detector on CPU if GPU is out of memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = "cpu"
            det = self.detector(
                img_bgr,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                device="cpu",
                verbose=False,
            )[0]
        except RuntimeError as e:
            # Some Ultralytics/PyTorch OOMs surface as RuntimeError with "out of memory"
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                det = self.detector(
                    img_bgr,
                    conf=float(self.cfg.conf),
                    iou=float(self.cfg.iou),
                    device="cpu",
                    verbose=False,
                )[0]
            else:
                raise
        if det is None or det.boxes is None or len(det.boxes) == 0:
            return None
        boxes = det.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = det.boxes.conf.detach().cpu().numpy().astype(np.float32)
        idx = int(np.argmax(scores))
        return boxes[idx]




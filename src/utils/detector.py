from typing import List, Tuple
import numpy as np
import torch
from ultralytics import YOLO


class HandDetector:
    def __init__(self, model_path: str, conf: float = 0.3, iou: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[float, float, float, float]]:
        detections = self.model(image_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        bboxes = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            if bbox.ndim == 0:
                continue
            bboxes.append(bbox[:4].tolist())
        return bboxes

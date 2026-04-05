from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RTMLIB_REPO = Path("/root/code/hand_reconstruction/rtmlib")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if RTMLIB_REPO.exists() and str(RTMLIB_REPO) not in sys.path:
    sys.path.insert(0, str(RTMLIB_REPO))

from rtmlib import RTMPose
from rtmlib.visualization.skeleton.hand21 import hand21

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/hand_reconstruction/GPGFormer/outputs/wilor_rtmlib_hand_2d")
DEFAULT_DET_MODEL = Path("/root/code/hand_reconstruction/WiLoR/pretrained_models/detector.pt")
DEFAULT_POSE_MODEL = Path(
    "/root/code/hand_reconstruction/rtmlib/rtmpose_onnx/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx"
)
DEFAULT_POSE_INPUT_SIZE = (256, 256)
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RIGHT_POINT_COLOR = (90, 220, 120)
RIGHT_LINK_COLOR = (70, 170, 95)
LEFT_POINT_COLOR = (70, 150, 255)
LEFT_LINK_COLOR = (45, 105, 220)
TEXT_COLOR = (245, 245, 245)
LABEL_TEXT_COLOR = (20, 20, 20)
NO_DET_BG_COLOR = (32, 32, 180)
HAND_KEYPOINT_INFO = hand21["keypoint_info"]
HAND_SKELETON_INFO = hand21["skeleton_info"]
HAND_NAME_TO_ID = {info["name"]: info["id"] for info in HAND_KEYPOINT_INFO.values()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use WiLoR detector for left/right hand detection and RTMPose for 21-keypoint estimation."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--det-model", type=str, default=str(DEFAULT_DET_MODEL), help="WiLoR detector.pt path.")
    parser.add_argument("--pose-model", type=str, default=str(DEFAULT_POSE_MODEL), help="RTMPose hand ONNX path.")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=("onnxruntime", "opencv", "openvino"))
    parser.add_argument("--device", type=str, default="cpu", help="RTMPose device.")
    parser.add_argument("--detector-device", type=str, default="cpu", help="WiLoR detector device, e.g. cpu or 0.")
    parser.add_argument("--detector-conf", type=float, default=0.3)
    parser.add_argument("--detector-iou", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-dets-per-image", type=int, default=8)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--show-kpt-idx", action="store_true")
    return parser.parse_args()


def collect_image_paths(img_dir: Path) -> list[Path]:
    return sorted(path for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_wilor_detector(weights_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing dependency 'ultralytics' in the current environment.") from exc
    if not weights_path.exists():
        raise FileNotFoundError(f"WiLoR detector weights do not exist: {weights_path}")
    return YOLO(str(weights_path))


def detect_wilor_hands(
    detector: Any,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    device: str,
    max_dets: int,
) -> list[dict[str, Any]]:
    results = detector(image_bgr, conf=float(conf), iou=float(iou), device=device, verbose=False)[0]
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return []

    boxes = results.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = results.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = results.boxes.cls.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(-scores)

    detections: list[dict[str, Any]] = []
    keep_n = len(order) if int(max_dets) <= 0 else min(len(order), int(max_dets))
    for idx in order[:keep_n]:
        class_id = int(round(float(classes[idx])))
        is_right = bool(class_id > 0)
        detections.append(
            {
                "bbox_xyxy": boxes[idx].astype(np.float32),
                "score": float(scores[idx]),
                "class_id": class_id,
                "is_right": is_right,
                "handedness": "right" if is_right else "left",
            }
        )
    return detections


def add_banner(image_bgr: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x1, y1 = 14, 14
    x2 = min(out.shape[1] - 14, x1 + tw + 12)
    y2 = min(out.shape[0] - 14, y1 + th + baseline + 10)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
    cv2.putText(out, text, (x1 + 6, y2 - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, lineType=cv2.LINE_AA)
    return out


def _base_colors(is_right: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if is_right:
        return RIGHT_POINT_COLOR, RIGHT_LINK_COLOR
    return LEFT_POINT_COLOR, LEFT_LINK_COLOR


def _mix_color(base: tuple[int, int, int], score: float) -> tuple[int, int, int]:
    weight = float(np.clip(score, 0.0, 1.0))
    return tuple(int(round((1.0 - weight) * 255 + weight * c)) for c in base)


def draw_hand_pose(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    is_right: bool,
    radius: int,
    line_width: int,
    show_kpt_idx: bool,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    point_base, link_base = _base_colors(bool(is_right))

    for ske_info in HAND_SKELETON_INFO.values():
        src_name, dst_name = ske_info["link"]
        src_id = HAND_NAME_TO_ID[src_name]
        dst_id = HAND_NAME_TO_ID[dst_name]
        p0 = pts[src_id]
        p1 = pts[dst_id]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        blend_score = max(float(conf[src_id]), float(conf[dst_id]))
        color = _mix_color(link_base, 0.2 + 0.8 * blend_score)
        cv2.line(
            out,
            (int(round(float(p0[0]))), int(round(float(p0[1])))),
            (int(round(float(p1[0]))), int(round(float(p1[1])))),
            color,
            max(1, int(line_width)),
            lineType=cv2.LINE_AA,
        )

    for idx, point in enumerate(pts):
        if not np.isfinite(point).all():
            continue
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        point_color = _mix_color(point_base, 0.2 + 0.8 * float(conf[idx]))
        cv2.circle(out, center, max(2, int(radius) + 1), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, center, max(1, int(radius)), point_color, -1, lineType=cv2.LINE_AA)
        if show_kpt_idx:
            cv2.putText(out, str(idx), (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1, lineType=cv2.LINE_AA)

    return out


def draw_detection_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    handedness: str,
    det_score: float,
    pose_mean_score: float,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    is_right = str(handedness).lower() == "right"
    point_color, _ = _base_colors(is_right)
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    cv2.rectangle(out, (x1, y1), (x2, y2), point_color, 2, lineType=cv2.LINE_AA)

    label = f"{handedness} det={float(det_score):.2f} pose={float(pose_mean_score):.2f}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    text_y2 = max(y1, th + baseline + 6)
    text_y1 = max(0, text_y2 - th - baseline - 6)
    text_x2 = min(out.shape[1] - 1, x1 + tw + 8)
    cv2.rectangle(out, (x1, text_y1), (text_x2, text_y2), point_color, -1)
    cv2.putText(out, label, (x1 + 4, text_y2 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, LABEL_TEXT_COLOR, 1, lineType=cv2.LINE_AA)
    return out


def to_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype=np.float32).tolist()


def save_result(
    image_path: Path,
    out_dir: Path,
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    hands: list[dict[str, Any]],
) -> dict[str, Any]:
    image_out_dir = out_dir / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_out_dir / "input.png"), image_bgr)
    cv2.imwrite(str(image_out_dir / "wilor_rtmlib_overlay.png"), overlay_bgr)

    summary = {
        "image": str(image_path),
        "num_hands": int(len(hands)),
        "hands": hands,
    }
    (image_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    det_model_path = Path(args.det_model)
    pose_model_path = Path(args.pose_model)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    if not det_model_path.exists():
        raise FileNotFoundError(f"Detector model does not exist: {det_model_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Pose model does not exist: {pose_model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    detector = load_wilor_detector(det_model_path)
    pose_model = RTMPose(
        str(pose_model_path),
        model_input_size=DEFAULT_POSE_INPUT_SIZE,
        backend=args.backend,
        device=args.device,
    )

    print(f"Loaded WiLoR detector: {det_model_path}")
    print(f"Loaded RTMPose model: {pose_model_path}")

    all_summary: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WiLoR+RTM] skip unreadable image: {image_path}")
            continue

        detections = detect_wilor_hands(
            detector=detector,
            image_bgr=image_bgr,
            conf=float(args.detector_conf),
            iou=float(args.detector_iou),
            device=str(args.detector_device),
            max_dets=int(args.max_dets_per_image),
        )

        overlay = image_bgr.copy()
        hands_summary: list[dict[str, Any]] = []
        if not detections:
            overlay = add_banner(overlay, "No hands detected by WiLoR detector", NO_DET_BG_COLOR)
        else:
            bboxes = np.stack([np.asarray(det["bbox_xyxy"], dtype=np.float32) for det in detections], axis=0)
            keypoints, scores = pose_model(image_bgr, bboxes=bboxes)
            keypoints = np.asarray(keypoints, dtype=np.float32).reshape(len(detections), 21, 2)
            scores = np.asarray(scores, dtype=np.float32).reshape(len(detections), 21)

            for idx, det in enumerate(detections):
                pose_mean_score = float(np.mean(scores[idx]))
                overlay = draw_hand_pose(
                    overlay,
                    keypoints[idx],
                    scores[idx],
                    is_right=bool(det["is_right"]),
                    radius=int(args.radius),
                    line_width=int(args.line_width),
                    show_kpt_idx=bool(args.show_kpt_idx),
                )
                overlay = draw_detection_bbox(
                    overlay,
                    det["bbox_xyxy"],
                    det["handedness"],
                    float(det["score"]),
                    pose_mean_score,
                )
                hands_summary.append(
                    {
                        "hand_index": idx,
                        "handedness": str(det["handedness"]),
                        "is_right": bool(det["is_right"]),
                        "detector_score": float(det["score"]),
                        "detector_class_id": int(det["class_id"]),
                        "bbox_xyxy": to_list(det["bbox_xyxy"]),
                        "pose_mean_score": pose_mean_score,
                        "kpt_scores": to_list(scores[idx]),
                        "kpts_2d": to_list(keypoints[idx]),
                    }
                )

        summary = save_result(
            image_path=image_path,
            out_dir=out_dir,
            image_bgr=image_bgr,
            overlay_bgr=overlay,
            hands=hands_summary,
        )
        all_summary.append(summary)
        label_counts = ", ".join(hand["handedness"] for hand in hands_summary) if hands_summary else "none"
        print(f"[WiLoR+RTM] {image_path.name}: detected {summary['num_hands']} hand(s) [{label_counts}]")

    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "img_dir": str(img_dir),
                "det_model": str(det_model_path),
                "pose_model": str(pose_model_path),
                "num_images": len(all_summary),
                "results": all_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved WiLoR detector + RTMPose visualization to: {out_dir}")


if __name__ == "__main__":
    main()

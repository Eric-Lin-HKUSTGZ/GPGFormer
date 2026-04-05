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

from rtmlib import RTMDet, RTMPose, draw_bbox
from rtmlib.visualization.skeleton.hand21 import hand21

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/hand_reconstruction/GPGFormer/outputs/rtmlib_hand_2d")
DEFAULT_POSE_MODEL = Path(
    "/root/code/hand_reconstruction/rtmlib/rtmpose_onnx/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx"
)
DEFAULT_DET_MODEL = Path(
    "/root/code/hand_reconstruction/rtmlib/rtmdet_onnx/"
    "rtmdet_nano_8xb32-300e_hand-267f9c8f/end2end.onnx"
)
DEFAULT_MEDIAPIPE_MODEL = Path("/root/code/vepfs/GPGFormer/weights/hand_landmarker/hand_landmarker.task")
DEFAULT_DET_INPUT_SIZE = (320, 320)
DEFAULT_POSE_INPUT_SIZE = (256, 256)
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TEXT_COLOR = (245, 245, 245)
LABEL_BG_COLOR = (40, 40, 40)
NO_DET_BG_COLOR = (32, 32, 180)
RIGHT_BOX_COLOR = (90, 220, 120)
LEFT_BOX_COLOR = (70, 150, 255)
UNKNOWN_BOX_COLOR = (90, 220, 220)
HAND_KEYPOINT_INFO = hand21["keypoint_info"]
HAND_SKELETON_INFO = hand21["skeleton_info"]
HAND_NAME_TO_ID = {info["name"]: info["id"] for info in HAND_KEYPOINT_INFO.values()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTMLib hand detection + 21-keypoint pose estimation and save visualizations."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pose-model", type=str, default=str(DEFAULT_POSE_MODEL))
    parser.add_argument(
        "--det-model",
        type=str,
        default=str(DEFAULT_DET_MODEL),
        help="Local RTMDet hand ONNX path.",
    )
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=("onnxruntime", "opencv", "openvino"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--handedness-source",
        type=str,
        default="mediapipe",
        choices=("mediapipe", "none"),
        help="Optional left/right hand classification branch.",
    )
    parser.add_argument("--mediapipe-model", type=str, default=str(DEFAULT_MEDIAPIPE_MODEL))
    parser.add_argument("--max-num-hands", type=int, default=8)
    parser.add_argument("--min-det-conf", type=float, default=0.3)
    parser.add_argument("--min-track-conf", type=float, default=0.3)
    parser.add_argument("--handedness-bbox-expand", type=float, default=1.25)
    parser.add_argument("--kpt-thr", type=float, default=0.35)
    parser.add_argument("--skeleton-line-width", type=int, default=3)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--show-kpt-idx", action="store_true")
    return parser.parse_args()


def collect_image_paths(img_dir: Path) -> list[Path]:
    image_paths = sorted(
        path for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return image_paths


def add_banner(image_bgr: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x1, y1 = 14, 14
    x2 = min(out.shape[1] - 14, x1 + tw + 12)
    y2 = min(out.shape[0] - 14, y1 + th + baseline + 10)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        out,
        text,
        (x1 + 6, y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore

        return mp
    except Exception:
        return None


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float32).reshape(4).tolist()
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float32).reshape(4).tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def detect_mediapipe_hands_21(
    image_bgr: np.ndarray,
    max_num_hands: int = 2,
    min_det_conf: float = 0.3,
    min_track_conf: float = 0.3,
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    mp = _try_import_mediapipe()
    if mp is None:
        return []

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    model_path_use = str(model_path or "")
    if model_path_use and Path(model_path_use).is_file():
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path_use),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=int(max_num_hands),
                min_hand_detection_confidence=float(min_det_conf),
                min_hand_presence_confidence=float(min_det_conf),
                min_tracking_confidence=float(min_track_conf),
            )
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                results = landmarker.detect(mp_image)
        except Exception:
            results = None
        else:
            out: list[dict[str, Any]] = []
            hand_landmarks = getattr(results, "hand_landmarks", None)
            handedness_list = getattr(results, "handedness", None)
            if hand_landmarks:
                for idx, lm in enumerate(hand_landmarks):
                    pts = np.asarray([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                    bbox = np.array(
                        [
                            float(np.min(pts[:, 0])),
                            float(np.min(pts[:, 1])),
                            float(np.max(pts[:, 0])),
                            float(np.max(pts[:, 1])),
                        ],
                        dtype=np.float32,
                    )

                    handedness = "Unknown"
                    handedness_score = 0.0
                    if handedness_list and idx < len(handedness_list) and handedness_list[idx]:
                        handedness = str(handedness_list[idx][0].category_name)
                        handedness_score = float(handedness_list[idx][0].score)

                    out.append(
                        {
                            "kpts_2d": pts,
                            "bbox_xyxy": bbox,
                            "handedness": handedness,
                            "score": handedness_score,
                        }
                    )
                return out

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=int(max_num_hands),
            model_complexity=1,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf),
        )
        results = hands.process(img_rgb)
        hands.close()

        out: list[dict[str, Any]] = []
        if not results.multi_hand_landmarks:
            return out

        for idx, lm in enumerate(results.multi_hand_landmarks):
            pts = np.asarray([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
            bbox = np.array(
                [
                    float(np.min(pts[:, 0])),
                    float(np.min(pts[:, 1])),
                    float(np.max(pts[:, 0])),
                    float(np.max(pts[:, 1])),
                ],
                dtype=np.float32,
            )

            handedness = "Unknown"
            handedness_score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = str(results.multi_handedness[idx].classification[0].label)
                handedness_score = float(results.multi_handedness[idx].classification[0].score)

            out.append(
                {
                    "kpts_2d": pts,
                    "bbox_xyxy": bbox,
                    "handedness": handedness,
                    "score": handedness_score,
                }
            )
        return out

    return []


def assign_mediapipe_handedness(
    mp_hands: list[dict[str, Any]],
    bboxes: np.ndarray,
    min_match_iou: float = 0.05,
) -> list[dict[str, Any]]:
    assigned = [
        {
            "handedness": "unknown",
            "handedness_score": 0.0,
            "handedness_source": "none",
            "mediapipe_match_iou": 0.0,
            "mediapipe_bbox_xyxy": None,
        }
        for _ in range(len(bboxes))
    ]
    if len(bboxes) == 0 or not mp_hands:
        return assigned

    candidates: list[tuple[float, int, int]] = []
    for det_idx, bbox in enumerate(np.asarray(bboxes, dtype=np.float32)):
        for mp_idx, mp_hand in enumerate(mp_hands):
            iou = _bbox_iou_xyxy(bbox, np.asarray(mp_hand["bbox_xyxy"], dtype=np.float32))
            candidates.append((iou, det_idx, mp_idx))

    used_det: set[int] = set()
    used_mp: set[int] = set()
    for iou, det_idx, mp_idx in sorted(candidates, key=lambda x: x[0], reverse=True):
        if iou < float(min_match_iou) or det_idx in used_det or mp_idx in used_mp:
            continue
        mp_hand = mp_hands[mp_idx]
        used_det.add(det_idx)
        used_mp.add(mp_idx)
        assigned[det_idx] = {
            "handedness": str(mp_hand.get("handedness", "Unknown")).strip().lower(),
            "handedness_score": float(mp_hand.get("score", 0.0)),
            "handedness_source": "mediapipe",
            "mediapipe_match_iou": float(iou),
            "mediapipe_bbox_xyxy": to_list(np.asarray(mp_hand["bbox_xyxy"], dtype=np.float32)),
        }
    return assigned


def _handedness_color(label: str) -> tuple[int, int, int]:
    label_use = str(label).strip().lower()
    if label_use == "right":
        return RIGHT_BOX_COLOR
    if label_use == "left":
        return LEFT_BOX_COLOR
    return UNKNOWN_BOX_COLOR


def _expand_bbox_xyxy(bbox_xyxy: np.ndarray, image_shape: tuple[int, int], scale: float = 1.25) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    h, w = [int(v) for v in image_shape]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    half_w = 0.5 * bw * float(scale)
    half_h = 0.5 * bh * float(scale)
    out = np.array(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(w - 1), cx + half_w),
            min(float(h - 1), cy + half_h),
        ],
        dtype=np.float32,
    )
    return out


def infer_mediapipe_handedness_for_bboxes(
    image_bgr: np.ndarray,
    bboxes: np.ndarray,
    max_num_hands: int = 2,
    min_det_conf: float = 0.3,
    min_track_conf: float = 0.3,
    model_path: str | None = None,
    bbox_expand: float = 1.25,
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    if len(bboxes) == 0:
        return assigned

    image_h, image_w = image_bgr.shape[:2]
    for bbox in np.asarray(bboxes, dtype=np.float32):
        crop_bbox = _expand_bbox_xyxy(bbox, (image_h, image_w), scale=float(bbox_expand))
        x1, y1, x2, y2 = [int(round(v)) for v in crop_bbox.tolist()]
        if x2 <= x1 or y2 <= y1:
            assigned.append(
                {
                    "handedness": "unknown",
                    "handedness_score": 0.0,
                    "handedness_source": "none",
                    "mediapipe_match_iou": 0.0,
                    "mediapipe_bbox_xyxy": None,
                }
            )
            continue

        crop = image_bgr[y1:y2, x1:x2]
        mp_hands = detect_mediapipe_hands_21(
            image_bgr=crop,
            max_num_hands=max(1, int(max_num_hands)),
            min_det_conf=float(min_det_conf),
            min_track_conf=float(min_track_conf),
            model_path=model_path,
        )
        if not mp_hands:
            assigned.append(
                {
                    "handedness": "unknown",
                    "handedness_score": 0.0,
                    "handedness_source": "none",
                    "mediapipe_match_iou": 0.0,
                    "mediapipe_bbox_xyxy": None,
                }
            )
            continue

        best_hand = None
        best_score = -1e9
        for mp_hand in mp_hands:
            mp_bbox_crop = np.asarray(mp_hand["bbox_xyxy"], dtype=np.float32)
            mp_bbox_full = mp_bbox_crop.copy()
            mp_bbox_full[[0, 2]] += float(x1)
            mp_bbox_full[[1, 3]] += float(y1)
            iou = _bbox_iou_xyxy(bbox, mp_bbox_full)
            score = iou + 0.05 * float(mp_hand.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_hand = (mp_hand, mp_bbox_full, iou)

        if best_hand is None:
            assigned.append(
                {
                    "handedness": "unknown",
                    "handedness_score": 0.0,
                    "handedness_source": "none",
                    "mediapipe_match_iou": 0.0,
                    "mediapipe_bbox_xyxy": None,
                }
            )
            continue

        mp_hand, mp_bbox_full, iou = best_hand
        assigned.append(
            {
                "handedness": str(mp_hand.get("handedness", "Unknown")).strip().lower(),
                "handedness_score": float(mp_hand.get("score", 0.0)),
                "handedness_source": "mediapipe_crop",
                "mediapipe_match_iou": float(iou),
                "mediapipe_bbox_xyxy": to_list(mp_bbox_full),
            }
        )
    return assigned


def draw_pose_labels(
    image_bgr: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    handedness_infos: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    if len(bboxes) == 0:
        return out

    pose_scores = np.asarray(scores, dtype=np.float32)
    pose_scores = pose_scores.reshape(pose_scores.shape[0], -1)
    mean_scores = pose_scores.mean(axis=1)

    for idx, (bbox, mean_score) in enumerate(zip(np.asarray(bboxes, dtype=np.float32), mean_scores)):
        x1, y1, _, _ = [int(round(v)) for v in bbox.tolist()]
        handedness_info = handedness_infos[idx] if handedness_infos is not None and idx < len(handedness_infos) else None
        handedness = str(handedness_info.get("handedness", "unknown")) if handedness_info is not None else "unknown"
        handedness_score = float(handedness_info.get("handedness_score", 0.0)) if handedness_info is not None else 0.0
        label = f"hand_{idx:02d} {handedness} mp={handedness_score:.2f} pose={float(mean_score):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y2 = max(y1, th + baseline + 6)
        y1_box = max(0, y2 - th - baseline - 6)
        x2 = min(out.shape[1] - 1, x1 + tw + 8)
        box_color = _handedness_color(handedness)
        cv2.rectangle(out, (x1, y1_box), (x2, y2), box_color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def draw_handedness_bboxes(
    image_bgr: np.ndarray,
    bboxes: np.ndarray,
    handedness_infos: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    if len(bboxes) == 0:
        return out
    for idx, bbox in enumerate(np.asarray(bboxes, dtype=np.float32)):
        handedness = "unknown"
        if handedness_infos is not None and idx < len(handedness_infos):
            handedness = str(handedness_infos[idx].get("handedness", "unknown"))
        color = _handedness_color(handedness)
        out = draw_bbox(out, np.asarray([bbox], dtype=np.float32), color=color)
    return out


def draw_keypoint_indices(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    if len(keypoints) == 0:
        return out

    for hand_kpts, hand_scores in zip(np.asarray(keypoints), np.asarray(scores)):
        hand_scores = np.asarray(hand_scores, dtype=np.float32).reshape(-1)
        for idx, (point, score) in enumerate(zip(np.asarray(hand_kpts, dtype=np.float32), hand_scores)):
            if not np.isfinite(point).all() or float(score) < float(kpt_thr):
                continue
            px, py = int(round(float(point[0]))), int(round(float(point[1])))
            cv2.putText(
                out,
                str(idx),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT_COLOR,
                1,
                lineType=cv2.LINE_AA,
            )
    return out


def _blend_color(color: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
    w = float(np.clip(weight, 0.0, 1.0))
    return tuple(int(round((1.0 - w) * 255 + w * c)) for c in color)


def draw_hand_pose_all(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    line_width: int = 3,
    radius: int = 4,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    if len(keypoints) == 0:
        return out

    for hand_kpts, hand_scores in zip(np.asarray(keypoints, dtype=np.float32), np.asarray(scores, dtype=np.float32)):
        pts = np.asarray(hand_kpts, dtype=np.float32).reshape(-1, 2)
        conf = np.asarray(hand_scores, dtype=np.float32).reshape(-1)

        for ske_info in HAND_SKELETON_INFO.values():
            src_name, dst_name = ske_info["link"]
            src_id = HAND_NAME_TO_ID[src_name]
            dst_id = HAND_NAME_TO_ID[dst_name]
            p0 = pts[src_id]
            p1 = pts[dst_id]
            if not np.isfinite(p0).all() or not np.isfinite(p1).all():
                continue
            score_pair = max(float(conf[src_id]), float(conf[dst_id]))
            color = _blend_color(tuple(ske_info["color"]), 0.25 + 0.75 * score_pair)
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
            base_color = tuple(HAND_KEYPOINT_INFO[idx]["color"])
            score = float(conf[idx])
            point_color = _blend_color(base_color, 0.2 + 0.8 * score)
            cv2.circle(out, center, max(2, int(radius) + 1), (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, center, max(1, int(radius)), point_color, -1, lineType=cv2.LINE_AA)

    return out


def to_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype=np.float32).tolist()


def build_models(args: argparse.Namespace) -> tuple[RTMDet, RTMPose]:
    detector = RTMDet(
        args.det_model,
        model_input_size=DEFAULT_DET_INPUT_SIZE,
        backend=args.backend,
        device=args.device,
    )
    pose_model = RTMPose(
        args.pose_model,
        model_input_size=DEFAULT_POSE_INPUT_SIZE,
        backend=args.backend,
        device=args.device,
    )
    return detector, pose_model


def run_inference(
    detector: RTMDet,
    pose_model: RTMPose,
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = np.asarray(detector(image_bgr), dtype=np.float32)
    if bboxes.size == 0:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        empty_kpts = np.zeros((0, 21, 2), dtype=np.float32)
        empty_scores = np.zeros((0, 21), dtype=np.float32)
        return empty_boxes, empty_kpts, empty_scores

    bboxes = bboxes.reshape(-1, 4)
    keypoints, scores = pose_model(image_bgr, bboxes=bboxes)
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(len(bboxes), -1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(len(bboxes), -1)
    return bboxes, keypoints, scores


def save_result(
    image_path: Path,
    out_dir: Path,
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    bboxes: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    handedness_infos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image_out_dir = out_dir / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_out_dir / "input.png"), image_bgr)
    cv2.imwrite(str(image_out_dir / "rtmlib_hand_overlay.png"), overlay_bgr)

    hands_summary = []
    mean_scores = np.asarray(scores, dtype=np.float32).reshape(len(scores), -1).mean(axis=1) if len(scores) else []
    for idx, bbox in enumerate(np.asarray(bboxes, dtype=np.float32)):
        handedness_info = handedness_infos[idx] if handedness_infos is not None and idx < len(handedness_infos) else {}
        hands_summary.append(
            {
                "hand_index": idx,
                "bbox_xyxy": to_list(bbox),
                "handedness": str(handedness_info.get("handedness", "unknown")),
                "handedness_score": float(handedness_info.get("handedness_score", 0.0)),
                "handedness_source": str(handedness_info.get("handedness_source", "none")),
                "mediapipe_match_iou": float(handedness_info.get("mediapipe_match_iou", 0.0)),
                "mediapipe_bbox_xyxy": handedness_info.get("mediapipe_bbox_xyxy"),
                "pose_mean_score": float(mean_scores[idx]),
                "kpt_scores": to_list(scores[idx]),
                "kpts_2d": to_list(keypoints[idx]),
            }
        )

    summary = {
        "image": str(image_path),
        "num_hands": int(len(hands_summary)),
        "hands": hands_summary,
    }
    (image_out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    pose_model_path = Path(args.pose_model)
    mediapipe_model_path = Path(args.mediapipe_model)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Pose model does not exist: {pose_model_path}")
    if args.handedness_source == "mediapipe" and not mediapipe_model_path.exists():
        print(
            "[RTMLib] MediaPipe Tasks model not found; falling back to the legacy Solutions API for handedness. "
            "This may return unknown on harder in-the-wild images."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    detector, pose_model = build_models(args)
    all_summary: list[dict[str, Any]] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[RTMLib] skip unreadable image: {image_path}")
            continue

        bboxes, keypoints, scores = run_inference(detector, pose_model, image_bgr)
        handedness_infos = [
            {
                "handedness": "unknown",
                "handedness_score": 0.0,
                "handedness_source": "none",
                "mediapipe_match_iou": 0.0,
                "mediapipe_bbox_xyxy": None,
            }
            for _ in range(len(bboxes))
        ]
        if args.handedness_source == "mediapipe" and len(bboxes) > 0:
            handedness_infos = infer_mediapipe_handedness_for_bboxes(
                image_bgr=image_bgr,
                bboxes=bboxes,
                max_num_hands=int(args.max_num_hands),
                min_det_conf=float(args.min_det_conf),
                min_track_conf=float(args.min_track_conf),
                model_path=str(mediapipe_model_path),
                bbox_expand=float(args.handedness_bbox_expand),
            )

        overlay = image_bgr.copy()
        if len(bboxes) == 0:
            overlay = add_banner(overlay, "No hands detected by RTMLib", NO_DET_BG_COLOR)
        else:
            overlay = draw_handedness_bboxes(overlay, bboxes, handedness_infos)
            overlay = draw_hand_pose_all(
                overlay,
                keypoints,
                scores,
                line_width=int(args.skeleton_line_width),
                radius=int(args.radius),
            )
            overlay = draw_pose_labels(overlay, bboxes, scores, handedness_infos=handedness_infos)
            if args.show_kpt_idx:
                overlay = draw_keypoint_indices(overlay, keypoints, scores, float(args.kpt_thr))

        summary = save_result(
            image_path=image_path,
            out_dir=out_dir,
            image_bgr=image_bgr,
            overlay_bgr=overlay,
            bboxes=bboxes,
            keypoints=keypoints,
            scores=scores,
            handedness_infos=handedness_infos,
        )
        all_summary.append(summary)
        print(f"[RTMLib] {image_path.name}: detected {summary['num_hands']} hand(s)")

    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "img_dir": str(img_dir),
                "pose_model": str(pose_model_path),
                "det_model": str(args.det_model),
                "num_images": len(all_summary),
                "results": all_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved RTMLib hand visualization to: {out_dir}")


if __name__ == "__main__":
    main()

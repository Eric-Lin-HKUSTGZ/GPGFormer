from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from visualization.demo_gpgformer_tvec_origin import detect_mediapipe_hands_21

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/mediapipe_2d_raw")
DEFAULT_MEDIAPIPE_MODEL = Path("/root/code/vepfs/GPGFormer/weights/hand_landmarker.task")

HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]

RIGHT_POINT_COLOR = (90, 220, 120)
RIGHT_LINK_COLOR = (70, 170, 95)
LEFT_POINT_COLOR = (70, 150, 255)
LEFT_LINK_COLOR = (45, 105, 220)
TEXT_COLOR = (245, 245, 245)
NO_DET_BG_COLOR = (32, 32, 180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw-image 2D hand joints predicted by MediaPipe without any crop or data augmentation."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-num-hands", type=int, default=4)
    parser.add_argument("--min-det-conf", type=float, default=0.3)
    parser.add_argument("--min-track-conf", type=float, default=0.3)
    parser.add_argument("--show-kpt-idx", action="store_true")
    parser.add_argument(
        "--mediapipe-model",
        type=str,
        default=str(DEFAULT_MEDIAPIPE_MODEL),
        help="Optional MediaPipe Tasks .task model. Needed if your environment only has the Tasks API.",
    )
    return parser.parse_args()


def collect_image_paths(img_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(img_dir.glob(pattern)))
    uniq = sorted({path.resolve() for path in image_paths})
    return [Path(p) for p in uniq]


def _hand_colors(handedness: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if str(handedness).lower() == "left":
        return LEFT_POINT_COLOR, LEFT_LINK_COLOR
    return RIGHT_POINT_COLOR, RIGHT_LINK_COLOR


def draw_hand_joints(
    image_bgr: np.ndarray,
    joints_2d: np.ndarray,
    handedness: str,
    score: float,
    bbox_xyxy: np.ndarray,
    show_kpt_idx: bool = False,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    pts = np.asarray(joints_2d, dtype=np.float32).reshape(-1, 2)
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    point_color, link_color = _hand_colors(handedness)

    for child, parent in enumerate(HAND_PARENTS):
        if parent < 0:
            continue
        p0 = pts[parent]
        p1 = pts[child]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        cv2.line(
            out,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            link_color,
            2,
            lineType=cv2.LINE_AA,
        )

    for idx, p in enumerate(pts):
        if not np.isfinite(p).all():
            continue
        center = (int(round(p[0])), int(round(p[1])))
        cv2.circle(out, center, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, center, 3, point_color, -1, lineType=cv2.LINE_AA)
        if show_kpt_idx:
            cv2.putText(
                out,
                str(idx),
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT_COLOR,
                1,
                lineType=cv2.LINE_AA,
            )

    x1, y1, x2, y2 = [int(round(v)) for v in bbox.tolist()]
    cv2.rectangle(out, (x1, y1), (x2, y2), point_color, 2, lineType=cv2.LINE_AA)
    label = f"{handedness} {float(score):.2f}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_y2 = max(y1, th + baseline + 6)
    text_y1 = text_y2 - th - baseline - 6
    text_x2 = min(x1 + tw + 8, out.shape[1] - 1)
    text_x1 = max(0, text_x2 - (tw + 8))
    cv2.rectangle(out, (text_x1, text_y1), (text_x2, text_y2), point_color, -1)
    cv2.putText(
        out,
        label,
        (text_x1 + 4, text_y2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def draw_legend(image_bgr: np.ndarray) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    items = [
        (RIGHT_POINT_COLOR, "Right hand"),
        (LEFT_POINT_COLOR, "Left hand"),
    ]
    x = 14
    y = 18
    for color, text in items:
        cv2.circle(out, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            text,
            (x + 12, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )
        y += 20
    cv2.putText(
        out,
        "Input: original image only, no crop / no augmentation",
        (14, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def draw_no_detection_notice(image_bgr: np.ndarray) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    text = "No hands detected by MediaPipe on the original image"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x1 = 14
    y1 = 52
    x2 = min(out.shape[1] - 14, x1 + tw + 12)
    y2 = min(out.shape[0] - 14, y1 + th + baseline + 10)
    cv2.rectangle(out, (x1, y1), (x2, y2), NO_DET_BG_COLOR, -1)
    cv2.putText(
        out,
        text,
        (x1 + 6, y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]

    all_summary: list[dict[str, Any]] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        hands = detect_mediapipe_hands_21(
            image_bgr=image_bgr,
            max_num_hands=int(args.max_num_hands),
            min_det_conf=float(args.min_det_conf),
            min_track_conf=float(args.min_track_conf),
            model_path=args.mediapipe_model,
        )

        vis = image_bgr.copy()
        vis = draw_legend(vis)
        for hand in hands:
            vis = draw_hand_joints(
                image_bgr=vis,
                joints_2d=np.asarray(hand["kpts_2d"], dtype=np.float32),
                handedness=str(hand.get("handedness", "Unknown")),
                score=float(hand.get("score", 0.0)),
                bbox_xyxy=np.asarray(hand["bbox_xyxy"], dtype=np.float32),
                show_kpt_idx=bool(args.show_kpt_idx),
            )
        if not hands:
            vis = draw_no_detection_notice(vis)

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out_dir / "input.png"), image_bgr)
        cv2.imwrite(str(image_out_dir / "mediapipe_2d_overlay.png"), vis)

        summary = {
            "image": str(image_path),
            "num_hands": len(hands),
            "hands": [
                {
                    "handedness": str(hand.get("handedness", "Unknown")),
                    "score": float(hand.get("score", 0.0)),
                    "bbox_xyxy": np.asarray(hand["bbox_xyxy"], dtype=np.float32).tolist(),
                    "kpts_2d": np.asarray(hand["kpts_2d"], dtype=np.float32).tolist(),
                }
                for hand in hands
            ],
        }
        (image_out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_summary.append(summary)
        print(f"[MediaPipe] {image_path.name}: detected {len(hands)} hand(s)")

    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "img_dir": str(img_dir),
                "num_images": len(all_summary),
                "results": all_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved MediaPipe raw-image 2D visualization to: {out_dir}")


if __name__ == "__main__":
    main()

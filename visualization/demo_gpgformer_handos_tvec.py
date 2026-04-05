from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import torch

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency
    minimize = None

from visualization.demo_gpgformer_no_cam import (
    bbox_from_points,
    collect_image_paths,
    detect_hands,
    draw_detection_bbox,
    draw_detection_bboxes,
    extract_mano_faces,
    load_gpgformer,
    load_yolo_detector,
    project_points_with_translation,
    refine_translation_to_bbox,
    render_mesh_only,
    render_overlay_image_space_shaded,
    render_overlay_multiple_image_space_shaded,
    solve_translation_from_2d_correspondences,
    weakcam_crop_to_full,
)
from visualization.demo_gpgformer_tvec import (
    enhance_visualization_mesh,
    prepare_crop_weakcam,
    project_crop_points_to_full_image,
    restore_points_to_original_handedness,
    should_refine_to_detector,
)
from visualization.demo_gpgformer_tvec_origin import (
    detect_mediapipe_hands_21,
    match_mediapipe_to_yolo,
)

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_demo_handos_tvec_ho3d")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/mixed_ho3d_20260320/ho3d/gpgformer_best.pt"
    # "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_MEDIAPIPE_MODEL = Path("/root/code/vepfs/GPGFormer/weights/hand_landmarker.task")
DEFAULT_MESH_COLOR = "#7677D8"

HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]
TARGET_COLOR = (90, 230, 120)
REPROJ_COLOR = (70, 120, 255)
TARGET_LINK_COLOR = (70, 180, 95)
REPROJ_LINK_COLOR = (45, 70, 215)
DEFAULT_ROOT_INDEX = 9
MEDIAPIPE_SEMANTIC_WEIGHTS = np.array(
    [
        0.20,
        0.70, 0.80, 0.85, 0.35,
        0.95, 1.00, 1.00, 0.45,
        1.05, 1.10, 1.10, 0.45,
        0.95, 1.00, 1.00, 0.45,
        0.85, 0.90, 0.90, 0.40,
    ],
    dtype=np.float32,
)
MEDIAPIPE_STABLE_WEIGHTS = np.array(
    [
        0.0,
        0.60, 0.80, 0.90, 0.0,
        1.00, 1.00, 1.00, 0.0,
        1.10, 1.10, 1.10, 0.0,
        1.00, 1.00, 1.00, 0.0,
        0.85, 0.85, 0.85, 0.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPGFormer hand mesh reconstruction without camera intrinsics, following a HandOS-like default-camera+tvec pipeline."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--detector-ckpt", type=str, default=str(DEFAULT_DETECTOR_CKPT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--detector-conf", type=float, default=0.3)
    parser.add_argument("--detector-iou", type=float, default=0.6)
    parser.add_argument("--bbox-expand", type=float, default=1.25)
    parser.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-dets-per-image", type=int, default=8)
    parser.add_argument("--fov", type=float, default=60.0, help="Default camera FOV used to build K, similar to HandOS.")
    parser.add_argument(
        "--keypoints-source",
        type=str,
        default="auto",
        choices=("auto", "mediapipe", "predcam"),
        help="2D joints used to solve tvec. auto=MediaPipe if available else pred_cam projection.",
    )
    parser.add_argument(
        "--mediapipe-model",
        type=str,
        default=str(DEFAULT_MEDIAPIPE_MODEL),
        help="Optional MediaPipe Tasks hand_landmarker .task path.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.05,
        help="Minimum mesh depth after solving tvec; pushes the mesh forward to avoid clipped rendering.",
    )
    parser.add_argument(
        "--save-joint-debug",
        action="store_true",
        help="Also save a joints-only debug image for each detection.",
    )
    parser.add_argument(
        "--root-index",
        type=int,
        default=DEFAULT_ROOT_INDEX,
        help="Root joint index used to convert MANO outputs to HandOS-style root-relative 3D before solving tvec.",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def build_default_camera(image_w: int, image_h: int, fov: float) -> np.ndarray:
    theta = math.radians(float(fov) * 0.5)
    fx = float(image_w) * 0.5 / math.tan(theta)
    fy = float(image_h) * 0.5 / math.tan(theta)
    cx = float(image_w) * 0.5
    cy = float(image_h) * 0.5
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def ensure_points_in_front(points_3d: np.ndarray, cam_t: np.ndarray, z_min: float) -> np.ndarray:
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(3).copy()
    min_z = float(np.min(np.asarray(points_3d, dtype=np.float32)[:, 2] + cam_t[2]))
    if min_z < float(z_min):
        cam_t[2] += float(z_min) - min_z
    return cam_t


def compute_reproj_rmse(
    points_3d: np.ndarray,
    target_2d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
    joint_weights: np.ndarray | None = None,
) -> float:
    proj_2d = project_points_with_translation(points_3d, cam_param, cam_t)
    diff = proj_2d - np.asarray(target_2d, dtype=np.float32)
    sq = np.sum(diff * diff, axis=1)
    if joint_weights is None:
        return float(np.sqrt(np.mean(sq)))
    weights = np.asarray(joint_weights, dtype=np.float32).reshape(-1)
    weights = weights[: sq.shape[0]]
    weights = np.where(np.isfinite(weights), np.clip(weights, 0.0, None), 0.0)
    denom = float(np.sum(weights))
    if denom <= 1e-8:
        return float(np.sqrt(np.mean(sq)))
    return float(np.sqrt(np.sum(weights * sq) / denom))


def get_joint_weights(source_name: str, profile: str = "default") -> np.ndarray:
    if str(source_name) != "mediapipe":
        return np.ones((21,), dtype=np.float32)
    if str(profile) == "stable":
        return MEDIAPIPE_STABLE_WEIGHTS.copy()
    return MEDIAPIPE_SEMANTIC_WEIGHTS.copy()


def solve_translation_weighted(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    cam_param: np.ndarray,
    joint_weights: np.ndarray | None = None,
) -> np.ndarray | None:
    pts3 = np.asarray(points_3d, dtype=np.float32)
    pts2 = np.asarray(points_2d, dtype=np.float32)
    cam = np.asarray(cam_param, dtype=np.float32).reshape(-1)
    if pts3.ndim != 2 or pts3.shape[1] != 3 or pts2.ndim != 2 or pts2.shape[1] != 2 or cam.shape[0] < 4:
        return None

    finite = np.isfinite(pts3).all(axis=1) & np.isfinite(pts2).all(axis=1)
    weights = np.ones((pts3.shape[0],), dtype=np.float32) if joint_weights is None else np.asarray(joint_weights, dtype=np.float32).reshape(-1)
    weights = np.pad(weights, (0, max(0, pts3.shape[0] - weights.shape[0])), constant_values=1.0)[: pts3.shape[0]]
    finite &= np.isfinite(weights) & (weights > 1e-8)
    if int(finite.sum()) < 4:
        return None

    fx, fy, cx, cy = [float(v) for v in cam[:4]]
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None

    sel3 = pts3[finite]
    sel2 = pts2[finite]
    selw = weights[finite]
    xn = (sel2[:, 0] - cx) / fx
    yn = (sel2[:, 1] - cy) / fy

    a = np.zeros((sel3.shape[0] * 2, 3), dtype=np.float32)
    b = np.zeros((sel3.shape[0] * 2,), dtype=np.float32)
    a[0::2, 0] = 1.0
    a[0::2, 2] = -xn
    a[1::2, 1] = 1.0
    a[1::2, 2] = -yn
    b[0::2] = xn * sel3[:, 2] - sel3[:, 0]
    b[1::2] = yn * sel3[:, 2] - sel3[:, 1]

    row_w = np.repeat(np.sqrt(selw), 2).astype(np.float32)
    a *= row_w[:, None]
    b *= row_w

    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    sol = np.asarray(sol, dtype=np.float32).reshape(3)
    if not np.isfinite(sol).all():
        return None
    return sol


def convert_to_root_relative(
    vertices_3d: np.ndarray,
    joints_3d: np.ndarray,
    root_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(vertices_3d, dtype=np.float32).reshape(-1, 3).copy()
    joints = np.asarray(joints_3d, dtype=np.float32).reshape(-1, 3).copy()
    root_idx = int(root_index)
    if root_idx < 0 or root_idx >= joints.shape[0]:
        raise IndexError(f"root_index={root_idx} is out of range for joints with shape {joints.shape}")
    root_xyz = joints[root_idx].copy()
    verts -= root_xyz[None, :]
    joints -= root_xyz[None, :]
    return verts.astype(np.float32), joints.astype(np.float32), root_xyz.astype(np.float32)


def adjust_tvec_for_root_relative(cam_t: np.ndarray, root_xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(cam_t, dtype=np.float32).reshape(3) + np.asarray(root_xyz, dtype=np.float32).reshape(3)).astype(np.float32)


def compute_bbox_fit_terms(
    vertices_3d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
    target_bbox_xyxy: np.ndarray,
) -> tuple[float, float, float] | None:
    proj = project_points_with_translation(vertices_3d, cam_param, cam_t)
    proj_bbox = bbox_from_points(proj)
    if proj_bbox is None:
        return None

    proj_bbox = np.asarray(proj_bbox, dtype=np.float32).reshape(4)
    target = np.asarray(target_bbox_xyxy, dtype=np.float32).reshape(4)
    proj_w = max(float(proj_bbox[2] - proj_bbox[0]), 1e-6)
    proj_h = max(float(proj_bbox[3] - proj_bbox[1]), 1e-6)
    target_w = max(float(target[2] - target[0]), 1e-6)
    target_h = max(float(target[3] - target[1]), 1e-6)
    proj_c = np.array([(proj_bbox[0] + proj_bbox[2]) * 0.5, (proj_bbox[1] + proj_bbox[3]) * 0.5], dtype=np.float32)
    target_c = np.array([(target[0] + target[2]) * 0.5, (target[1] + target[3]) * 0.5], dtype=np.float32)
    target_diag = max(float(np.hypot(target_w, target_h)), 1e-6)
    center_term = float(np.linalg.norm(proj_c - target_c) / target_diag)
    size_term = abs(np.log(proj_w / target_w)) + abs(np.log(proj_h / target_h))
    area_ratio = float((proj_w * proj_h) / (target_w * target_h))
    return center_term, size_term, area_ratio


def score_tvec_candidate(
    vertices_3d: np.ndarray,
    joints_3d: np.ndarray,
    target_2d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
    target_bbox_xyxy: np.ndarray,
    joint_weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    rmse = compute_reproj_rmse(joints_3d, target_2d, cam_param, cam_t, joint_weights=joint_weights)
    bbox_terms = compute_bbox_fit_terms(vertices_3d, cam_param, cam_t, target_bbox_xyxy)
    if bbox_terms is None:
        return float("inf"), rmse, float("inf"), 0.0

    center_term, size_term, area_ratio = bbox_terms
    score = float(rmse + 80.0 * center_term + 45.0 * size_term)
    if area_ratio < 0.18 or area_ratio > 3.2:
        score += 250.0
    cam_t_z = float(np.asarray(cam_t, dtype=np.float32).reshape(3)[2])
    if cam_t_z < 0.05 or cam_t_z > 8.0:
        score += 200.0
    return score, rmse, center_term + size_term, area_ratio


def solve_tvec_handos_style(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    cam_param: np.ndarray,
    init_tvec: np.ndarray | None = None,
    max_attempts: int = 5,
    joint_weights: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, float, np.ndarray]:
    pts3 = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    pts2 = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    cam = np.asarray(cam_param, dtype=np.float32).reshape(4)

    valid = np.isfinite(pts3).all(axis=1) & np.isfinite(pts2).all(axis=1)
    if int(valid.sum()) < 4:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32) if init_tvec is None else np.asarray(init_tvec, dtype=np.float32).reshape(3)
        rmse = compute_reproj_rmse(pts3[valid], pts2[valid], cam, fallback, joint_weights=joint_weights[valid] if joint_weights is not None else None) if int(valid.sum()) > 0 else 1e9
        return False, fallback, rmse, valid

    sel3 = pts3[valid]
    sel2 = pts2[valid]

    sel_weights = None
    if joint_weights is not None:
        joint_weights = np.asarray(joint_weights, dtype=np.float32).reshape(-1)
        sel_weights = joint_weights[valid]
    ls_t = solve_translation_weighted(sel3, sel2, cam, joint_weights=sel_weights)
    if ls_t is None:
        ls_t = solve_translation_from_2d_correspondences(sel3, sel2, cam)
    weak_t = None if init_tvec is None else np.asarray(init_tvec, dtype=np.float32).reshape(3)

    candidate_ts: list[np.ndarray] = []
    if ls_t is not None and np.isfinite(ls_t).all():
        candidate_ts.append(np.asarray(ls_t, dtype=np.float32).reshape(3))
    if weak_t is not None and np.isfinite(weak_t).all():
        candidate_ts.append(weak_t.astype(np.float32))
    if not candidate_ts:
        candidate_ts.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    cur_t = min(
        candidate_ts,
        key=lambda t: compute_reproj_rmse(sel3, sel2, cam, np.asarray(t, dtype=np.float32).reshape(3), joint_weights=sel_weights),
    ).astype(np.float32)
    cur_t[2] = max(float(cur_t[2]), 1e-4)

    active = np.ones(sel3.shape[0], dtype=bool)
    used = active.copy()

    def per_point_error(tvec: np.ndarray, use_active: np.ndarray) -> np.ndarray:
        proj = project_points_with_translation(sel3[use_active], cam, tvec)
        diff = proj - sel2[use_active]
        return np.sqrt(np.sum(diff * diff, axis=1))

    for _ in range(int(max_attempts)):
        active_weights = sel_weights[active] if sel_weights is not None else None
        ls_t_active = solve_translation_weighted(sel3[active], sel2[active], cam, joint_weights=active_weights)
        if ls_t_active is None:
            ls_t_active = solve_translation_from_2d_correspondences(sel3[active], sel2[active], cam)
        if ls_t_active is not None and np.isfinite(ls_t_active).all():
            cur_t = np.asarray(ls_t_active, dtype=np.float32).reshape(3)

        err = per_point_error(cur_t, np.ones(sel3.shape[0], dtype=bool))
        if err.size == 0:
            break
        thr = max(float(err.mean() + err.std()), 6.0)
        new_active = err <= thr
        used = new_active
        if int(new_active.sum()) < 4 or np.array_equal(new_active, active):
            break
        active = new_active

    rmse = compute_reproj_rmse(
        sel3[used],
        sel2[used],
        cam,
        cur_t,
        joint_weights=sel_weights[used] if sel_weights is not None and int(used.sum()) > 0 else None,
    ) if int(used.sum()) > 0 else 1e9
    inlier_mask = np.zeros(pts3.shape[0], dtype=bool)
    inlier_mask[np.where(valid)[0][used]] = True
    return True, cur_t.astype(np.float32), rmse, inlier_mask


def draw_hand_joints(
    image_bgr: np.ndarray,
    joints_2d: np.ndarray,
    point_color: tuple[int, int, int],
    link_color: tuple[int, int, int],
    radius: int = 3,
    thickness: int = 2,
    hollow: bool = False,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    pts = np.asarray(joints_2d, dtype=np.float32).reshape(-1, 2)

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
            thickness,
            lineType=cv2.LINE_AA,
        )

    for p in pts:
        if not np.isfinite(p).all():
            continue
        center = (int(round(p[0])), int(round(p[1])))
        if hollow:
            cv2.circle(out, center, radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, center, radius, point_color, 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(out, center, radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, center, radius, point_color, -1, lineType=cv2.LINE_AA)
    return out


def draw_joint_legend(image_bgr: np.ndarray, source_name: str) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    items = [
        (TARGET_COLOR, f"target 2D: {source_name}"),
        (REPROJ_COLOR, "reproj 2D: solved tvec"),
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
            (245, 245, 245),
            1,
            lineType=cv2.LINE_AA,
        )
        y += 20
    return out


@torch.no_grad()
def infer_model(
    model: Any,
    crop_pack: dict[str, Any],
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    crop_rgb = crop_pack["crop_rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
    cam_param_patch = torch.from_numpy(np.asarray(crop_pack["cam_param_patch"], dtype=np.float32)).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    out = model(crop_rgb, cam_param=cam_param_patch)
    return {
        "v3d_crop": out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32),
        "k3d_crop": out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32),
        "pred_cam_crop": out["pred_cam"][0].detach().cpu().numpy().astype(np.float32),
    }


def collect_target_joints_2d(
    args: argparse.Namespace,
    det_bbox_xyxy: np.ndarray,
    is_right: bool,
    crop_pack: dict[str, Any],
    infer_out: dict[str, np.ndarray],
    mp_hands: list[dict[str, Any]],
) -> list[tuple[str, np.ndarray]]:
    targets: list[tuple[str, np.ndarray]] = []

    if args.keypoints_source in ("auto", "mediapipe"):
        mp_match = match_mediapipe_to_yolo(mp_hands, det_bbox_xyxy, is_right)
        if mp_match is not None:
            targets.append(("mediapipe", np.asarray(mp_match["kpts_2d"], dtype=np.float32)))
        elif args.keypoints_source == "mediapipe":
            raise RuntimeError("MediaPipe joints requested but no matched hand was found.")

    if args.keypoints_source in ("auto", "predcam"):
        predcam_k2d = project_crop_points_to_full_image(
            infer_out["k3d_crop"],
            infer_out["pred_cam_crop"],
            crop_pack,
            is_right,
        )
        targets.append(("predcam", predcam_k2d.astype(np.float32)))

    return targets


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    detector_device = "cpu" if device.type == "cpu" else str(device)

    model, cfg = load_gpgformer(Path(args.config), Path(args.ckpt), device)
    detector = load_yolo_detector(Path(args.detector_ckpt))
    faces = extract_mano_faces(model)

    paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        paths = paths[: int(args.max_images)]

    for image_path in paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        h, w = image_bgr.shape[:2]
        default_cam = build_default_camera(w, h, float(args.fov))
        detections = detect_hands(
            detector=detector,
            image_bgr=image_bgr,
            conf=args.detector_conf,
            iou=args.detector_iou,
            device=detector_device,
            max_dets=args.max_dets_per_image,
        )

        mp_hands: list[dict[str, Any]] = []
        if args.keypoints_source in ("auto", "mediapipe"):
            mp_hands = detect_mediapipe_hands_21(
                image_bgr,
                max_num_hands=max(4, len(detections)),
                model_path=args.mediapipe_model,
            )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        vis_vertices_list: list[np.ndarray] = []
        vis_depth_list: list[np.ndarray] = []
        vis_v2d_list: list[np.ndarray] = []
        target_k2d_list: list[np.ndarray] = []
        reproj_k2d_list: list[np.ndarray] = []
        meta_list: list[dict[str, Any]] = []
        vis_faces = faces

        for det_idx, det in enumerate(detections):
            det_bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            is_right = bool(det["is_right"])

            crop_pack = prepare_crop_weakcam(image_bgr, det_bbox, is_right, cfg, args.bbox_expand)
            infer_out = infer_model(model, crop_pack)

            v3d_raw = restore_points_to_original_handedness(infer_out["v3d_crop"], is_right)
            k3d_raw = restore_points_to_original_handedness(infer_out["k3d_crop"], is_right)
            v3d, k3d, root_xyz = convert_to_root_relative(v3d_raw, k3d_raw, int(args.root_index))

            try:
                target_sources = collect_target_joints_2d(
                    args=args,
                    det_bbox_xyxy=det_bbox,
                    is_right=is_right,
                    crop_pack=crop_pack,
                    infer_out=infer_out,
                    mp_hands=mp_hands,
                )
            except RuntimeError as exc:
                print(f"[WARN] {image_path.name} det{det_idx:02d}: {exc}")
                continue
            if not target_sources:
                print(f"[WARN] {image_path.name} det{det_idx:02d}: no valid 2D joint target source.")
                continue

            pred_cam = np.asarray(infer_out["pred_cam_crop"], dtype=np.float32).reshape(3).copy()
            if not is_right:
                pred_cam[1] *= -1.0
            weakcam_init_t = weakcam_crop_to_full(
                pred_cam=pred_cam,
                box_center=crop_pack["box_center"],
                box_size=crop_pack["box_size"],
                img_size_wh=crop_pack["img_size_wh"],
                focal_length=float(crop_pack["focal_full"]),
            )
            weakcam_init_t = adjust_tvec_for_root_relative(weakcam_init_t, root_xyz)

            candidate_records: list[dict[str, Any]] = []
            for target_name, target_k2d_cur in target_sources:
                profiles = ["default"] if target_name != "mediapipe" else ["default", "stable"]
                for profile_name in profiles:
                    joint_weights = get_joint_weights(target_name, profile=profile_name)
                    ls_t = solve_translation_weighted(k3d, target_k2d_cur, default_cam, joint_weights=joint_weights)
                    if ls_t is None:
                        ls_t = solve_translation_from_2d_correspondences(k3d, target_k2d_cur, default_cam)
                    if ls_t is not None and np.isfinite(ls_t).all():
                        candidate_records.append(
                            {
                                "candidate_name": f"{target_name}:{profile_name}:linear",
                                "source_name": target_name,
                                "target_k2d": target_k2d_cur,
                                "cam_t": np.asarray(ls_t, dtype=np.float32).reshape(3),
                                "inliers": np.isfinite(k3d).all(axis=1) & np.isfinite(target_k2d_cur).all(axis=1),
                                "joint_weights": joint_weights,
                                "weight_profile": profile_name,
                            }
                        )

                    ok, solved_t, _, inliers = solve_tvec_handos_style(
                        points_3d=k3d,
                        points_2d=target_k2d_cur,
                        cam_param=default_cam,
                        init_tvec=weakcam_init_t,
                        joint_weights=joint_weights,
                    )
                    if ok and np.isfinite(solved_t).all():
                        candidate_records.append(
                            {
                                "candidate_name": f"{target_name}:{profile_name}:robust",
                                "source_name": target_name,
                                "target_k2d": target_k2d_cur,
                                "cam_t": np.asarray(solved_t, dtype=np.float32).reshape(3),
                                "inliers": inliers,
                                "joint_weights": joint_weights,
                                "weight_profile": profile_name,
                            }
                        )

            candidate_records.append(
                {
                    "candidate_name": "predcam:weakcam_init",
                    "source_name": "predcam",
                    "target_k2d": dict(target_sources).get("predcam", target_sources[0][1]),
                    "cam_t": weakcam_init_t.astype(np.float32),
                    "inliers": np.isfinite(k3d).all(axis=1),
                    "joint_weights": get_joint_weights("predcam"),
                    "weight_profile": "default",
                }
            )

            expanded_candidates: list[dict[str, Any]] = []
            for cand in candidate_records:
                base_cam_t = ensure_points_in_front(v3d, cand["cam_t"], z_min=float(args.z_min))
                expanded_candidates.append({**cand, "cam_t": base_cam_t})
                refined_cam_t = refine_translation_to_bbox(
                    vertices_3d=v3d,
                    cam_param=default_cam,
                    init_cam_t=base_cam_t,
                    target_bbox_xyxy=det_bbox,
                    num_iters=4,
                ).astype(np.float32)
                refined_cam_t = ensure_points_in_front(v3d, refined_cam_t, z_min=float(args.z_min))
                expanded_candidates.append(
                    {
                        **cand,
                        "candidate_name": cand["candidate_name"] + ":bbox",
                        "cam_t": refined_cam_t,
                    }
                )

            best_candidate = None
            best_score = float("inf")
            for cand in expanded_candidates:
                score, rmse_cur, bbox_penalty, area_ratio = score_tvec_candidate(
                    vertices_3d=v3d,
                    joints_3d=k3d,
                    target_2d=cand["target_k2d"],
                    cam_param=default_cam,
                    cam_t=cand["cam_t"],
                    target_bbox_xyxy=det_bbox,
                    joint_weights=cand.get("joint_weights"),
                )
                cand["score"] = float(score)
                cand["rmse"] = float(rmse_cur)
                cand["bbox_penalty"] = float(bbox_penalty)
                cand["area_ratio"] = float(area_ratio)
                if score < best_score:
                    best_score = score
                    best_candidate = cand

            if best_candidate is None:
                print(f"[WARN] {image_path.name} det{det_idx:02d}: failed to construct any valid tvec candidate.")
                continue

            cam_t = np.asarray(best_candidate["cam_t"], dtype=np.float32)
            target_k2d = np.asarray(best_candidate["target_k2d"], dtype=np.float32)
            source_name = str(best_candidate["source_name"])
            reproj_rmse = float(best_candidate["rmse"])
            inliers = np.asarray(best_candidate["inliers"], dtype=bool)

            if source_name != "mediapipe":
                print(
                    f"[WARN] {image_path.name} det{det_idx:02d}: "
                    f"selected fallback 2D source '{source_name}' via {best_candidate['candidate_name']}."
                )

            vis_vertices, vis_faces = enhance_visualization_mesh(v3d, k3d, faces)
            vis_v2d = project_points_with_translation(vis_vertices, default_cam, cam_t)
            vis_depth = (vis_vertices[:, 2] + cam_t[2]).astype(np.float32)
            reproj_k2d = project_points_with_translation(k3d, default_cam, cam_t)
            reproj_rmse = compute_reproj_rmse(k3d, target_k2d, default_cam, cam_t)

            det_overlay = render_overlay_image_space_shaded(
                full_image_bgr=image_bgr.copy(),
                vertices_2d=vis_v2d,
                vertices_3d=vis_vertices,
                vertices_depth=vis_depth,
                faces=vis_faces,
                color_hex=args.mesh_color,
            )
            det_overlay = draw_hand_joints(det_overlay, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
            det_overlay = draw_hand_joints(det_overlay, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
            det_overlay = draw_detection_bbox(det_overlay, det_bbox, is_right, det.get("score"))
            det_overlay = draw_joint_legend(det_overlay, source_name)
            cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_overlay.png"), det_overlay)

            if args.save_joint_debug:
                joint_debug = image_bgr.copy()
                joint_debug = draw_hand_joints(joint_debug, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
                joint_debug = draw_hand_joints(joint_debug, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
                joint_debug = draw_detection_bbox(joint_debug, det_bbox, is_right, det.get("score"))
                joint_debug = draw_joint_legend(joint_debug, source_name)
                cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_joints.png"), joint_debug)

            mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
            cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_mesh.png"), mesh_only)

            vis_vertices_list.append(vis_vertices)
            vis_depth_list.append(vis_depth)
            vis_v2d_list.append(vis_v2d)
            target_k2d_list.append(target_k2d)
            reproj_k2d_list.append(reproj_k2d)
            meta_list.append(
                {
                    "det_idx": det_idx,
                    "bbox_xyxy": det_bbox.tolist(),
                    "is_right": is_right,
                    "keypoints_source": source_name,
                    "tvec_candidate": str(best_candidate["candidate_name"]),
                    "joint_weight_profile": str(best_candidate.get("weight_profile", "default")),
                    "tvec_score": float(best_candidate["score"]),
                    "reproj_rmse_px": float(reproj_rmse),
                    "num_inliers": int(np.count_nonzero(inliers)),
                    "root_index": int(args.root_index),
                    "root_joint_xyz_m": root_xyz.tolist(),
                    "cam_param": default_cam.tolist(),
                    "cam_t": cam_t.tolist(),
                    "target_kpts_2d": target_k2d.tolist(),
                    "reproj_kpts_2d": reproj_k2d.tolist(),
                }
            )

        if vis_v2d_list:
            all_overlay = render_overlay_multiple_image_space_shaded(
                full_image_bgr=image_bgr.copy(),
                vertices_2d_list=vis_v2d_list,
                vertices_3d_list=vis_vertices_list,
                vertices_depth_list=vis_depth_list,
                faces=vis_faces,
                color_hex=args.mesh_color,
            )
            for target_k2d, reproj_k2d, meta in zip(target_k2d_list, reproj_k2d_list, meta_list):
                all_overlay = draw_hand_joints(all_overlay, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
                all_overlay = draw_hand_joints(all_overlay, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
                all_overlay = draw_joint_legend(all_overlay, str(meta["keypoints_source"]))
            all_overlay = draw_detection_bboxes(all_overlay, detections)
            cv2.imwrite(str(image_out_dir / "all_hands_overlay.png"), all_overlay)

        (image_out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "fov": float(args.fov),
                    "num_detections": len(meta_list),
                    "detections": meta_list,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"Saved HandOS-style tvec visualization to: {out_dir}")


if __name__ == "__main__":
    main()

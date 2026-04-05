from __future__ import annotations

import copy
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
import trimesh
import yaml

from data.utils import get_example
from infer_to_json import build_model_from_cfg, warmup_lazy_modules

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

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

DEFAULT_MESH_VIEW_AZIM_DEG = 35.0
DEFAULT_MESH_VIEW_ELEV_DEG = -20.0

FINGERTIP_VERTEX_IDS = np.array([744, 320, 443, 555, 672], dtype=np.int64)
FINGERTIP_PREV_JOINT_IDS = np.array([3, 7, 11, 15, 19], dtype=np.int64)
FINGERTIP_JOINT_IDS = np.array([4, 8, 12, 16, 20], dtype=np.int64)
VIS_TIP_EXTENSION_M = 0.0045
VIS_SURFACE_INFLATE_M = 0.0015


def collect_image_paths(img_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(img_dir.glob(pattern)))
    uniq = sorted({path.resolve() for path in image_paths})
    return [Path(p) for p in uniq]


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


def _det_color_bgr(is_right: bool) -> tuple[int, int, int]:
    return (80, 200, 120) if is_right else (60, 170, 255)


def draw_detection_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    score: float | None = None,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    color = _det_color_bgr(bool(is_right))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    label = "right" if is_right else "left"
    if score is not None:
        label = f"{label} {float(score):.2f}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_y2 = max(y1, th + baseline + 6)
    text_y1 = text_y2 - th - baseline - 6
    text_x2 = min(x1 + tw + 8, out.shape[1] - 1)
    text_x1 = max(0, text_x2 - (tw + 8))
    cv2.rectangle(out, (text_x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(out, label, (text_x1 + 4, text_y2 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, lineType=cv2.LINE_AA)
    return out


def load_gpgformer(cfg_path: Path, ckpt_path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_file = yaml.safe_load(cfg_path.read_text())
    ckpt_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None

    if ckpt_cfg is not None:
        cfg_model = copy.deepcopy(ckpt_cfg)
        cfg_model.setdefault("paths", {})
        cfg_model["paths"].update(cfg_file.get("paths", {}))
    else:
        cfg_model = cfg_file

    model = build_model_from_cfg(cfg_model).to(device)
    warmup_lazy_modules(model, cfg_model, device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg_model


def extract_mano_faces(model: Any) -> np.ndarray:
    candidates: list[Any] = []
    mano_layer = getattr(model, "mano", None)
    if mano_layer is not None:
        candidates.append(mano_layer)
        inner_mano = getattr(mano_layer, "mano", None)
        if inner_mano is not None:
            candidates.append(inner_mano)

    for obj in candidates:
        for attr_name in ("faces", "th_faces", "faces_tensor"):
            if not hasattr(obj, attr_name):
                continue
            faces = getattr(obj, attr_name)
            if faces is None:
                continue
            if isinstance(faces, torch.Tensor):
                faces_np = faces.detach().cpu().numpy()
            else:
                faces_np = np.asarray(faces)
            if faces_np.size == 0:
                continue
            return np.asarray(faces_np, dtype=np.int64)

    inspected = [type(obj).__name__ for obj in candidates]
    raise AttributeError(
        "Could not find MANO face topology on model.mano. "
        f"Inspected objects: {inspected}."
    )


def prepare_crop_weakcam(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    cfg: dict[str, Any],
    bbox_expand: float,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    box_size = max(max(x2 - x1, 2.0), max(y2 - y1, 2.0)) * float(bbox_expand)
    box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    half = box_size * 0.5
    if box_size <= float(w):
        box_center[0] = float(np.clip(box_center[0], half, float(w) - half))
    if box_size <= float(h):
        box_center[1] = float(np.clip(box_center[1], half, float(h) - half))

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    img_size = int(model_cfg.get("image_size", dataset_cfg.get("img_size", 256)))
    img_width = int(model_cfg.get("image_width", int(img_size * 0.75)))
    focal_full = float(model_cfg.get("focal_length", 5000.0))

    img_patch, _, _, _, _, _, trans = get_example(
        image_bgr,
        box_center[0],
        box_center[1],
        box_size,
        box_size,
        np.zeros((21, 3), np.float32),
        np.zeros((21, 3), np.float32),
        {
            "global_orient": np.zeros((3,), np.float32),
            "hand_pose": np.zeros((45,), np.float32),
            "betas": np.zeros((10,), np.float32),
        },
        {
            "global_orient": np.zeros((1,), np.float32),
            "hand_pose": np.zeros((1,), np.float32),
            "betas": np.zeros((1,), np.float32),
        },
        list(range(21)),
        img_width,
        img_size,
        None,
        None,
        do_augment=False,
        is_right=is_right,
        augm_config=dataset_cfg.get("wilor_aug_config", {}),
        is_bgr=True,
        return_trans=True,
    )

    crop_rgb01 = torch.from_numpy(img_patch).float() / 255.0
    crop_rgb = (crop_rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    trans_inv = cv2.invertAffineTransform(np.asarray(trans, dtype=np.float32))

    cam_param_patch = np.array([focal_full, focal_full, img_width / 2.0, img_size / 2.0], dtype=np.float32)
    cam_param_full = np.array([focal_full, focal_full, w / 2.0, h / 2.0], dtype=np.float32)

    return {
        "crop_rgb": crop_rgb,
        "box_center": box_center,
        "box_size": float(box_size),
        "cam_param_patch": cam_param_patch,
        "cam_param_full": cam_param_full,
        "patch_w": float(img_width),
        "patch_h": float(img_size),
        "trans_inv": trans_inv,
        "img_w": float(w),
        "img_h": float(h),
        "img_size_wh": np.array([w, h], dtype=np.float32),
        "focal_full": focal_full,
    }


@torch.no_grad()
def infer_model(model: Any, crop_pack: dict[str, Any]) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    crop_rgb = crop_pack["crop_rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
    cam_param_patch = torch.from_numpy(np.asarray(crop_pack["cam_param_patch"], dtype=np.float32)).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    out = model(crop_rgb, cam_param=cam_param_patch)
    return {
        "v3d_crop": out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32),
        "k3d_crop": out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32),
        "pred_cam_crop": out["pred_cam"][0].detach().cpu().numpy().astype(np.float32),
    }


def restore_points_to_original_handedness(points_3d: np.ndarray, is_right: bool) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3).copy()
    if not is_right:
        points[:, 0] *= -1.0
    return points


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
    return (
        np.asarray(cam_t, dtype=np.float32).reshape(3) + np.asarray(root_xyz, dtype=np.float32).reshape(3)
    ).astype(np.float32)


def weakcam_crop_to_full(
    pred_cam: np.ndarray,
    box_center: np.ndarray,
    box_size: float,
    img_size_wh: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    pred_cam = np.asarray(pred_cam, dtype=np.float32).reshape(3)
    cx, cy = [float(v) for v in np.asarray(box_center, dtype=np.float32).reshape(2)]
    img_w, img_h = [float(v) for v in np.asarray(img_size_wh, dtype=np.float32).reshape(2)]
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = float(box_size) * float(pred_cam[0]) + 1e-9
    tz = 2.0 * float(focal_length) / bs
    tx = (2.0 * (cx - w_2) / bs) + float(pred_cam[1])
    ty = (2.0 * (cy - h_2) / bs) + float(pred_cam[2])
    return np.array([tx, ty, tz], dtype=np.float32)


def project_points_with_translation(
    points_3d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    cam = np.asarray(cam_param, dtype=np.float32).reshape(-1)
    trans = np.asarray(cam_t, dtype=np.float32).reshape(3)
    pts = pts + trans[None, :]
    z = pts[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = cam[0] * (pts[:, 0:1] / z) + cam[2]
    v = cam[1] * (pts[:, 1:2] / z) + cam[3]
    return np.concatenate([u, v], axis=1).astype(np.float32)


def solve_translation_from_2d_correspondences(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    cam_param: np.ndarray,
) -> np.ndarray | None:
    pts3 = np.asarray(points_3d, dtype=np.float32)
    pts2 = np.asarray(points_2d, dtype=np.float32)
    cam = np.asarray(cam_param, dtype=np.float32).reshape(-1)
    if pts3.ndim != 2 or pts3.shape[1] != 3 or pts2.ndim != 2 or pts2.shape[1] != 2 or cam.shape[0] < 4:
        return None

    finite = np.isfinite(pts3).all(axis=1) & np.isfinite(pts2).all(axis=1)
    if int(finite.sum()) < 4:
        return None

    fx, fy, cx, cy = [float(v) for v in cam[:4]]
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None

    sel3 = pts3[finite]
    sel2 = pts2[finite]
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

    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    sol = np.asarray(sol, dtype=np.float32).reshape(3)
    if not np.isfinite(sol).all():
        return None
    return sol


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
        rmse = compute_reproj_rmse(
            pts3[valid],
            pts2[valid],
            cam,
            fallback,
            joint_weights=joint_weights[valid] if joint_weights is not None else None,
        ) if int(valid.sum()) > 0 else 1e9
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
        key=lambda t: compute_reproj_rmse(
            sel3,
            sel2,
            cam,
            np.asarray(t, dtype=np.float32).reshape(3),
            joint_weights=sel_weights,
        ),
    ).astype(np.float32)
    cur_t[2] = max(float(cur_t[2]), 1e-4)

    active = np.ones(sel3.shape[0], dtype=bool)
    used = active.copy()

    def per_point_error(tvec: np.ndarray) -> np.ndarray:
        proj = project_points_with_translation(sel3, cam, tvec)
        diff = proj - sel2
        return np.sqrt(np.sum(diff * diff, axis=1))

    for _ in range(int(max_attempts)):
        active_weights = sel_weights[active] if sel_weights is not None else None
        ls_t_active = solve_translation_weighted(sel3[active], sel2[active], cam, joint_weights=active_weights)
        if ls_t_active is None:
            ls_t_active = solve_translation_from_2d_correspondences(sel3[active], sel2[active], cam)
        if ls_t_active is not None and np.isfinite(ls_t_active).all():
            cur_t = np.asarray(ls_t_active, dtype=np.float32).reshape(3)

        err = per_point_error(cur_t)
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


def bbox_from_points(points_2d: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 4:
        return None
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    if np.any(max_xy <= min_xy):
        return None
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)


def refine_translation_to_bbox(
    vertices_3d: np.ndarray,
    cam_param: np.ndarray,
    init_cam_t: np.ndarray,
    target_bbox_xyxy: np.ndarray,
    num_iters: int = 4,
) -> np.ndarray:
    cam = np.asarray(cam_param, dtype=np.float32).reshape(-1)
    target = np.asarray(target_bbox_xyxy, dtype=np.float32).reshape(4)
    cam_t = np.asarray(init_cam_t, dtype=np.float32).reshape(3).copy()
    fx, fy = float(cam[0]), float(cam[1])

    target_w = max(float(target[2] - target[0]), 1e-6)
    target_h = max(float(target[3] - target[1]), 1e-6)
    target_cx = (float(target[0]) + float(target[2])) * 0.5
    target_cy = (float(target[1]) + float(target[3])) * 0.5

    for _ in range(max(1, int(num_iters))):
        proj = project_points_with_translation(vertices_3d, cam, cam_t)
        bbox = bbox_from_points(proj)
        if bbox is None:
            break

        proj_w = max(float(bbox[2] - bbox[0]), 1e-6)
        proj_h = max(float(bbox[3] - bbox[1]), 1e-6)
        proj_cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        proj_cy = (float(bbox[1]) + float(bbox[3])) * 0.5

        scale_ratio = 0.5 * ((proj_w / target_w) + (proj_h / target_h))
        scale_ratio = float(np.clip(scale_ratio, 0.25, 4.0))
        cam_t[2] *= scale_ratio

        proj = project_points_with_translation(vertices_3d, cam, cam_t)
        bbox = bbox_from_points(proj)
        if bbox is None:
            break
        proj_cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        proj_cy = (float(bbox[1]) + float(bbox[3])) * 0.5

        dx_px = target_cx - proj_cx
        dy_px = target_cy - proj_cy
        cam_t[0] += dx_px * cam_t[2] / max(fx, 1e-6)
        cam_t[1] += dy_px * cam_t[2] / max(fy, 1e-6)

    return cam_t.astype(np.float32)


def enhance_visualization_mesh(
    vertices: np.ndarray,
    keypoints_3d: np.ndarray,
    faces: np.ndarray,
    is_right: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3).copy()
    joints = np.asarray(keypoints_3d, dtype=np.float32).reshape(-1, 3)
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if verts.shape[0] == 0 or faces_arr.shape[0] == 0 or joints.shape[0] < 21:
        return verts, faces_arr

    # Mirroring a right-hand MANO mesh into a left hand flips the triangle winding.
    # If we keep the original winding here, trimesh computes inward normals and the
    # later normal-based inflation step shrinks the left hand, making fingers look thin.
    geom_faces = faces_arr if bool(is_right) else faces_arr[:, [0, 2, 1]]

    neighbors = [set() for _ in range(verts.shape[0])]
    for a, b, c in geom_faces.tolist():
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    def _ring(seed: int, max_steps: int = 2) -> dict[int, int]:
        visited = {int(seed)}
        frontier = {int(seed)}
        result = {int(seed): 0}
        for dist in range(1, int(max_steps) + 1):
            nxt: set[int] = set()
            for vid in frontier:
                nxt.update(neighbors[vid])
            nxt -= visited
            for vid in nxt:
                result[int(vid)] = dist
            visited |= nxt
            frontier = nxt
        return result

    vis_verts = verts.copy()
    ring_weights = {0: 1.0, 1: 0.60, 2: 0.28}
    for tip_vid, prev_jid, tip_jid in zip(
        FINGERTIP_VERTEX_IDS.tolist(),
        FINGERTIP_PREV_JOINT_IDS.tolist(),
        FINGERTIP_JOINT_IDS.tolist(),
    ):
        direction = joints[int(tip_jid)] - joints[int(prev_jid)]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            continue
        direction = direction / direction_norm
        for vid, dist in _ring(int(tip_vid), max_steps=2).items():
            vis_verts[vid] += direction * (VIS_TIP_EXTENSION_M * ring_weights.get(int(dist), 0.0))

    try:
        vis_verts, vis_faces = trimesh.remesh.subdivide_loop(vis_verts, geom_faces, iterations=1)
        vis_verts = np.asarray(vis_verts, dtype=np.float32)
        vis_faces = np.asarray(vis_faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vis_verts, faces=vis_faces, process=False)
        vis_verts = vis_verts + np.asarray(mesh.vertex_normals, dtype=np.float32) * VIS_SURFACE_INFLATE_M

        # Keep the external contract unchanged: callers still pass canonical MANO face
        # winding and the renderer handles left-hand face flipping on its own.
        if not bool(is_right):
            vis_faces = vis_faces[:, [0, 2, 1]]
        return vis_verts.astype(np.float32), vis_faces.astype(np.int64)
    except Exception:
        return vis_verts.astype(np.float32), faces_arr.astype(np.int64)


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
        cv2.circle(out, center, radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        if hollow:
            cv2.circle(out, center, radius, point_color, 1, lineType=cv2.LINE_AA)
        else:
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


def _as_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rotate_vertices_for_display(
    vertices: np.ndarray,
    elev_deg: float,
    azim_deg: float,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    rot_x = trimesh.transformations.rotation_matrix(np.radians(float(elev_deg)), [1.0, 0.0, 0.0])[:3, :3]
    rot_y = trimesh.transformations.rotation_matrix(np.radians(float(azim_deg)), [0.0, 1.0, 0.0])[:3, :3]
    rot = rot_y @ rot_x
    return (verts @ rot.T).astype(np.float32)


def _create_raymond_lights() -> list[Any]:
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes: list[Any] = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp], dtype=np.float32)
        z /= np.linalg.norm(z) + 1e-8
        x = np.array([-z[1], z[0], 0.0], dtype=np.float32)
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


def _render_mesh_rgba(
    vertices: np.ndarray,
    faces: np.ndarray,
    color_rgb: tuple[float, float, float],
    is_right: bool,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_translation: np.ndarray,
    center_mesh: bool = False,
    scene_bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    import pyrender

    mesh_vertices = np.asarray(vertices, dtype=np.float32).copy()
    if center_mesh:
        mesh_vertices -= mesh_vertices.mean(axis=0, keepdims=True)
    faces_use = np.asarray(faces, dtype=np.int64).copy()
    if not is_right:
        faces_use = faces_use[:, [0, 2, 1]]

    vertex_colors = np.array([(*color_rgb, 1.0)] * mesh_vertices.shape[0], dtype=np.float32)
    mesh = trimesh.Trimesh(mesh_vertices, faces_use, vertex_colors=vertex_colors, process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180.0), [1.0, 0.0, 0.0]))
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.35, 0.35, 0.35))
    scene.add(pyr_mesh, name="mesh")

    camera_pose = np.eye(4, dtype=np.float32)
    cam_t = np.asarray(camera_translation, dtype=np.float32).reshape(3).copy()
    cam_t[0] *= -1.0
    camera_pose[:3, 3] = cam_t
    camera = pyrender.IntrinsicsCamera(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), zfar=1e12)
    scene.add(camera, pose=camera_pose)

    for node in _create_raymond_lights():
        scene.add_node(node)

    renderer = pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height), point_size=1.0)
    try:
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return rgba.astype(np.uint8)


def _composite_rgba(rgba_bottom: np.ndarray, rgba_top: np.ndarray) -> np.ndarray:
    base = np.asarray(rgba_bottom, dtype=np.uint8)
    over = np.asarray(rgba_top, dtype=np.uint8)
    base_rgb = base[:, :, :3].astype(np.float32)
    base_a = base[:, :, 3:4].astype(np.float32) / 255.0
    over_rgb = over[:, :, :3].astype(np.float32)
    over_a = over[:, :, 3:4].astype(np.float32) / 255.0
    out_a = over_a + base_a * (1.0 - over_a)
    denom = np.clip(out_a, 1e-6, 1.0)
    out_rgb = (over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)) / denom
    out = np.zeros_like(base, dtype=np.uint8)
    out[:, :, :3] = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)
    out[:, :, 3:4] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def composite_rgba_over_bgr(image_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    img_bgr = np.asarray(image_bgr, dtype=np.uint8)
    fg_rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    comp_rgb = bg_rgb * (1.0 - alpha) + fg_rgb * alpha
    return cv2.cvtColor(np.clip(comp_rgb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_overlay_paper_style(
    full_image_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    cam_param: np.ndarray,
    color_hex: str,
    is_right: bool,
) -> np.ndarray:
    img_bgr = np.asarray(full_image_bgr, dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    fx, fy, cx, cy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:4]]
    rgba = _render_mesh_rgba(
        vertices=vertices,
        faces=faces,
        color_rgb=_as_rgb01(color_hex),
        is_right=is_right,
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_translation=cam_t,
        center_mesh=False,
        scene_bg_color=(1.0, 1.0, 1.0),
    )
    return composite_rgba_over_bgr(img_bgr, rgba)


def render_overlay_multiple_paper_style(
    full_image_bgr: np.ndarray,
    vertices_list: list[np.ndarray],
    faces_list: list[np.ndarray],
    cam_t_list: list[np.ndarray],
    cam_param: np.ndarray,
    color_hex: str,
    is_right_list: list[bool],
) -> np.ndarray:
    img_bgr = np.asarray(full_image_bgr, dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    fx, fy, cx, cy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:4]]
    rgba_all = np.zeros((h, w, 4), dtype=np.uint8)
    color_rgb = _as_rgb01(color_hex)

    for vertices, faces, cam_t, is_right in zip(vertices_list, faces_list, cam_t_list, is_right_list):
        rgba = _render_mesh_rgba(
            vertices=vertices,
            faces=faces,
            color_rgb=color_rgb,
            is_right=bool(is_right),
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_translation=cam_t,
            center_mesh=False,
            scene_bg_color=(1.0, 1.0, 1.0),
        )
        rgba_all = _composite_rgba(rgba_all, rgba)
    return composite_rgba_over_bgr(img_bgr, rgba_all)


def render_mesh_only(
    vertices: np.ndarray,
    faces: np.ndarray,
    color_hex: str,
    is_right: bool,
    render_size: int,
    view_azim_deg: float = DEFAULT_MESH_VIEW_AZIM_DEG,
    view_elev_deg: float = DEFAULT_MESH_VIEW_ELEV_DEG,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    centered = verts - verts.mean(axis=0, keepdims=True)
    display_verts = _rotate_vertices_for_display(centered, elev_deg=view_elev_deg, azim_deg=view_azim_deg)
    radius = float(np.linalg.norm(display_verts, axis=1).max()) if display_verts.size else 0.1
    cam_z = max(0.7, radius * 6.0)
    focal = float(render_size) * 2.4
    rgba = _render_mesh_rgba(
        vertices=display_verts,
        faces=faces,
        color_rgb=_as_rgb01(color_hex),
        is_right=is_right,
        width=render_size,
        height=render_size,
        fx=focal,
        fy=focal,
        cx=render_size / 2.0,
        cy=render_size / 2.0,
        camera_translation=np.array([0.0, 0.0, cam_z], dtype=np.float32),
        center_mesh=False,
        scene_bg_color=(1.0, 1.0, 1.0),
    )
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, 255, dtype=np.uint8).astype(np.float32)
    comp = bg * (1.0 - alpha) + rgb.astype(np.float32) * alpha
    return cv2.cvtColor(np.clip(comp, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
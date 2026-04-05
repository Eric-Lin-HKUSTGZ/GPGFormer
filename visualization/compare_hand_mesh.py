from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
HAND_RECON_ROOT = WORKSPACE_ROOT / "hand_reconstruction"
WILOR_ROOT = HAND_RECON_ROOT / "WiLoR"
HAMER_ROOT = HAND_RECON_ROOT / "hamer"
SIMPLEHAND_ROOT = HAND_RECON_ROOT / "simpleHand"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import pyrender
import torch
import trimesh
import yaml
from torch.utils.data import DataLoader

from data.dex_ycb_dataset import DexYCBDataset
from data.freihand_dataset_v3 import FreiHANDDatasetV3
from data.ho3d_dataset import HO3DDataset, HO3D_META_JOINT_MAP, _ho3d_cam_to_std_global_orient, _ho3d_cam_to_std_xyz
from data.utils import WILOR_JOINT_MAP
from gpgformer.models.mano.mano_layer import MANOConfig, MANOLayer
from infer_to_json import build_model_from_cfg, warmup_lazy_modules
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat

DEFAULT_GPGFORMER_CONFIGS = {
    "freihand": REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml",
    "dexycb": REPO_ROOT / "configs" / "ablations_v2" / "datasets" / "config_dexycb.yaml",
    "ho3d": REPO_ROOT / "configs" / "ablations_v2" / "datasets" / "config_ho3d.yaml",
}

DEFAULT_GPGFORMER_CKPTS = {
    "freihand": Path(
        "/root/code/vepfs/GPGFormer/checkpoints/"
        "freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt"
    ),
    "dexycb": Path(
        "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260321/dexycb/gpgformer_best.pt"
    ),
    "ho3d": Path(
        "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/mixed_ho3d_target67_20260322/ho3d/gpgformer_best.pt"
    ),
}
DEFAULT_WILOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/wilor_final.ckpt")
DEFAULT_HAMER_CKPT = Path("/root/code/vepfs/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
DEFAULT_SIMPLEHAND_CKPT = Path("/root/code/vepfs/simpleHand/train_log3/models_fastvit_ma36/latest")

DEFAULT_MANO_DIR = Path("/root/code/vepfs/GPGFormer/weights/mano")
DEFAULT_MANO_MEAN_PARAMS = DEFAULT_MANO_DIR / "mano_mean_params.npz"

DEFAULT_FREIHAND_ROOT = Path("/root/code/vepfs/dataset/FreiHAND_pub_v2")
DEFAULT_FREIHAND_EVAL_ROOT = Path("/root/code/vepfs/dataset/FreiHAND_pub_v2_eval")
DEFAULT_DEXYCB_ROOT = Path("/root/code/vepfs/dataset/dex-ycb")
DEFAULT_HO3D_ROOT = Path("/root/code/vepfs/dataset/HO3D_v3")

DEFAULT_SIMPLEHAND_SCALE_ENLARGE = 1.25
DEFAULT_PANEL_TILE = 512
DEFAULT_ROOT_INDEX = 9
DEFAULT_MESH_VIEW_AZIM_DEG = 35.0
DEFAULT_MESH_VIEW_ELEV_DEG = -20.0
DEFAULT_MIN_VISIBLE_HAND_JOINTS = 4
DEFAULT_MIN_VISIBLE_HAND_SPAN_PX = 16.0

COLOR_HEX = {
    "gt": "#5C5C5C",
    "gt_mesh": "#5C5C5C",
    "gpgformer": "#7677D8",
    # "wilor": "#E8B07A",
    "wilor": "#2A9D8F",
    "hamer": "#D7E7FA",
    "simplehand": "#9FC9A6",
}


@dataclass
class SampleBundle:
    dataset_name: str
    sample_idx: int
    full_image_bgr: np.ndarray
    img_path: str
    box_center: np.ndarray
    box_size: float
    img_size_wh: np.ndarray
    cam_param_full: np.ndarray
    crop_rgb_gpgformer: torch.Tensor
    crop_cam_param: np.ndarray
    is_right: bool
    gt_joints_3d: np.ndarray
    gt_joints_2d: np.ndarray
    gt_vertices: np.ndarray | None
    gt_faces: np.ndarray | None


@dataclass
class PredictionBundle:
    method_name: str
    pred_vertices: np.ndarray | None
    pred_keypoints_3d: np.ndarray | None
    pred_cam_t: np.ndarray | None
    status: str = "ok"
    message: str = ""
    metrics: dict[str, float | None] | None = None


@contextlib.contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _to_numpy(x: Any, dtype: np.dtype | type[np.floating[Any]] = np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _as_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _as_bgr255(hex_color: str) -> tuple[int, int, int]:
    r, g, b = _as_rgb01(hex_color)
    return int(round(b * 255.0)), int(round(g * 255.0)), int(round(r * 255.0))


def _make_camera_matrix(cam_param: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:4]]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _project_points(points_3d: np.ndarray, cam_param: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    fx, fy, cx, cy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:4]]

    z_raw = pts[:, 2]
    finite = np.isfinite(z_raw)
    nonzero = finite & (np.abs(z_raw) > 1e-8)
    if np.any(nonzero):
        depth_sign = 1.0 if float(np.median(z_raw[nonzero])) >= 0.0 else -1.0
    else:
        depth_sign = 1.0
    pts = pts * depth_sign

    z = pts[:, 2]
    z = np.where(np.abs(z) < 1e-6, np.where(z < 0.0, -1e-6, 1e-6), z)
    u = fx * (pts[:, 0] / z) + cx
    v = fy * (pts[:, 1] / z) + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def solve_translation_from_2d_correspondences(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    cam_param: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray | None:
    pts3 = np.asarray(points_3d, dtype=np.float32)
    pts2 = np.asarray(points_2d, dtype=np.float32)
    cam = np.asarray(cam_param, dtype=np.float32).reshape(-1)
    if pts3.ndim != 2 or pts3.shape[1] != 3 or pts2.ndim != 2 or pts2.shape[1] != 2 or cam.shape[0] < 4:
        return None

    n = int(pts3.shape[0])
    mask = np.ones((n,), dtype=bool) if valid is None else (np.asarray(valid).reshape(-1)[:n] > 0.5)
    finite = np.isfinite(pts3).all(axis=1) & np.isfinite(pts2).all(axis=1)
    mask &= finite
    if int(mask.sum()) < 2:
        return None

    fx, fy, cx, cy = [float(v) for v in cam[:4]]
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None

    sel3 = pts3[mask]
    sel2 = pts2[mask]
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


def _root_relative(points_3d: np.ndarray | None, root_index: int = DEFAULT_ROOT_INDEX) -> np.ndarray | None:
    if points_3d is None:
        return None
    pts = np.asarray(points_3d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] <= int(root_index):
        return None
    return pts - pts[int(root_index) : int(root_index) + 1]


def _mean_l2_mm(pred_points: np.ndarray | None, gt_points: np.ndarray | None) -> float | None:
    if pred_points is None or gt_points is None:
        return None
    pred = np.asarray(pred_points, dtype=np.float32)
    gt = np.asarray(gt_points, dtype=np.float32)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 3:
        return None
    if pred.size == 0:
        return None
    errs_m = np.linalg.norm(pred - gt, axis=1)
    return float(np.mean(errs_m) * 1000.0)


def _compute_prediction_metrics(
    sample: SampleBundle,
    pred: PredictionBundle,
    root_index: int = DEFAULT_ROOT_INDEX,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "joint_rr_mm": None,
        "vertex_rr_mm": None,
        "score_mm": None,
    }
    if pred.status != "ok" or pred.pred_keypoints_3d is None:
        return metrics

    pred_joints_rr = _root_relative(pred.pred_keypoints_3d, root_index)
    gt_joints_rr = _root_relative(sample.gt_joints_3d, root_index)
    joint_rr_mm = _mean_l2_mm(pred_joints_rr, gt_joints_rr)
    metrics["joint_rr_mm"] = joint_rr_mm

    vertex_rr_mm = None
    if sample.gt_vertices is not None and pred.pred_vertices is not None and pred_joints_rr is not None and gt_joints_rr is not None:
        pred_root = np.asarray(pred.pred_keypoints_3d, dtype=np.float32)[int(root_index)]
        gt_root = np.asarray(sample.gt_joints_3d, dtype=np.float32)[int(root_index)]
        pred_vertices_rr = np.asarray(pred.pred_vertices, dtype=np.float32) - pred_root[None, :]
        gt_vertices_rr = np.asarray(sample.gt_vertices, dtype=np.float32) - gt_root[None, :]
        vertex_rr_mm = _mean_l2_mm(pred_vertices_rr, gt_vertices_rr)
    metrics["vertex_rr_mm"] = vertex_rr_mm

    score_terms = [v for v in (joint_rr_mm, vertex_rr_mm) if v is not None and np.isfinite(v)]
    metrics["score_mm"] = float(np.mean(score_terms)) if score_terms else None
    return metrics


def _is_gpgformer_clearly_better(
    preds_by_method: dict[str, PredictionBundle],
    margin_mm: float,
) -> bool:
    gpgformer = preds_by_method.get("gpgformer")
    if gpgformer is None or not gpgformer.metrics:
        return False
    gpg_joint = gpgformer.metrics.get("joint_rr_mm")
    gpg_score = gpgformer.metrics.get("score_mm")
    if gpg_joint is None or gpg_score is None:
        return False

    for method_name in ("wilor", "hamer", "simplehand"):
        competitor = preds_by_method.get(method_name)
        if competitor is None or not competitor.metrics:
            return False
        comp_joint = competitor.metrics.get("joint_rr_mm")
        comp_score = competitor.metrics.get("score_mm")
        if comp_joint is None or comp_score is None:
            return False
        if (comp_joint - gpg_joint) < float(margin_mm):
            return False
        if (comp_score - gpg_score) < float(margin_mm):
            return False
    return True


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


def _bbox_xyxy_from_center_size(box_center: np.ndarray, box_size: float, img_size_wh: np.ndarray) -> np.ndarray:
    cx, cy = [float(v) for v in np.asarray(box_center, dtype=np.float32).reshape(2)]
    half = float(box_size) / 2.0
    w, h = [float(v) for v in np.asarray(img_size_wh, dtype=np.float32).reshape(2)]
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(w - 1.0, cx + half)
    y2 = min(h - 1.0, cy + half)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _points_inside_image(
    points_2d: np.ndarray,
    img_size_wh: np.ndarray,
    depths: np.ndarray | None = None,
) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    w, h = [float(v) for v in np.asarray(img_size_wh, dtype=np.float32).reshape(2)]
    finite = np.isfinite(pts).all(axis=1)
    if depths is not None:
        z = np.asarray(depths, dtype=np.float32).reshape(-1)[: pts.shape[0]]
        finite &= np.isfinite(z) & (z > 1e-6)
    return finite & (pts[:, 0] >= 0.0) & (pts[:, 0] < w) & (pts[:, 1] >= 0.0) & (pts[:, 1] < h)


def _hand_region_visible_in_image(
    sample: SampleBundle,
    min_visible_joints: int = DEFAULT_MIN_VISIBLE_HAND_JOINTS,
    min_visible_span_px: float = DEFAULT_MIN_VISIBLE_HAND_SPAN_PX,
) -> bool:
    visible = _points_inside_image(sample.gt_joints_2d, sample.img_size_wh, sample.gt_joints_3d[:, 2])
    if int(visible.sum()) < int(min_visible_joints):
        return False

    visible_pts = np.asarray(sample.gt_joints_2d, dtype=np.float32)[visible]
    if visible_pts.shape[0] < 2:
        return False

    span_xy = visible_pts.max(axis=0) - visible_pts.min(axis=0)
    return bool(float(np.max(span_xy)) >= float(min_visible_span_px))


def _fit_tile(image_bgr: np.ndarray, size: int) -> np.ndarray:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (245, 245, 245), thickness=-1)
    cv2.putText(out, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return out


def _render_text_panel(lines: Iterable[str], size: int) -> np.ndarray:
    panel = np.full((size, size, 3), 255, dtype=np.uint8)
    y = 48
    for line in lines:
        cv2.putText(panel, line[:56], (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 2, cv2.LINE_AA)
        y += 34
        if y > size - 24:
            break
    return panel


def _render_placeholder(title: str, message: str, size: int) -> np.ndarray:
    return _render_text_panel([title, "", message], size)


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


def _create_raymond_lights() -> list[pyrender.Node]:
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes: list[pyrender.Node] = []
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


def render_overlay(
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
    fg_rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    comp_rgb = bg_rgb * (1.0 - alpha) + fg_rgb * alpha
    return cv2.cvtColor(np.clip(comp_rgb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


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


class MANOHelper:
    def __init__(self, device: torch.device):
        self.device = device
        self.layer_right = MANOLayer(
            MANOConfig(
                model_path=str(DEFAULT_MANO_DIR),
                mean_params=str(DEFAULT_MANO_MEAN_PARAMS),
                is_rhand=True,
            )
        ).to(device)
        self.layer_left = MANOLayer(
            MANOConfig(
                model_path=str(DEFAULT_MANO_DIR),
                mean_params=str(DEFAULT_MANO_MEAN_PARAMS),
                is_rhand=False,
            )
        ).to(device)
        self.layer_right.eval()
        self.layer_left.eval()
        self.faces = np.asarray(self.layer_right.mano.faces, dtype=np.int64)

    @torch.no_grad()
    def decode(
        self,
        global_orient_aa: np.ndarray,
        hand_pose_aa: np.ndarray,
        betas: np.ndarray,
        transl_m: np.ndarray | None = None,
        is_right: bool = True,
    ) -> np.ndarray:
        global_orient_aa = np.asarray(global_orient_aa, dtype=np.float32).reshape(3)
        hand_pose_aa = np.asarray(hand_pose_aa, dtype=np.float32).reshape(45)
        betas = np.asarray(betas, dtype=np.float32).reshape(10)

        pose_aa = np.concatenate([global_orient_aa, hand_pose_aa], axis=0).reshape(16, 3)
        pose_rm = aa_to_rotmat(torch.from_numpy(pose_aa).to(self.device)).view(1, 16, 3, 3)
        mano_params = {
            "global_orient": pose_rm[:, [0]],
            "hand_pose": pose_rm[:, 1:],
            "betas": torch.from_numpy(betas).to(self.device).view(1, 10),
        }
        layer = self.layer_right if bool(is_right) else self.layer_left
        out = layer(mano_params, pose2rot=False)
        verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        if transl_m is not None:
            verts = verts + np.asarray(transl_m, dtype=np.float32).reshape(1, 3)
        return verts


class RunnerStore:
    def __init__(self, args: argparse.Namespace, device: torch.device, mano_helper: MANOHelper):
        self.args = args
        self.device = device
        self.mano_helper = mano_helper

        self._gpgformer_cache: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        self._wilor: tuple[Any, Any, Any, Any] | None = None
        self._hamer: tuple[Any, Any, Any, Any] | None = None
        self._simplehand: tuple[Any, dict[str, Any], Any] | None = None

    @staticmethod
    def _override_mano_paths(model_cfg: Any) -> Any:
        if "MANO" not in model_cfg:
            return model_cfg
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = str(DEFAULT_MANO_DIR)
        model_cfg.MANO.MODEL_PATH = str(DEFAULT_MANO_DIR)
        model_cfg.MANO.MEAN_PARAMS = str(DEFAULT_MANO_MEAN_PARAMS)
        model_cfg.freeze()
        return model_cfg

    def _resolve_gpgformer_paths(self, dataset_name: str) -> tuple[Path, Path]:
        cfg_path = Path(self.args.gpgformer_config) if self.args.gpgformer_config else DEFAULT_GPGFORMER_CONFIGS[dataset_name]
        if self.args.gpgformer_ckpt:
            ckpt_path = Path(self.args.gpgformer_ckpt)
        else:
            ckpt_path = DEFAULT_GPGFORMER_CKPTS.get(dataset_name, None)
            if ckpt_path is None:
                raise ValueError(f"dataset={dataset_name} requires --gpgformer-ckpt.")
        return cfg_path, ckpt_path

    def load_gpgformer(self, dataset_name: str) -> tuple[Any, dict[str, Any]]:
        cfg_path, ckpt_path = self._resolve_gpgformer_paths(dataset_name)
        key = (str(cfg_path), str(ckpt_path))
        if key in self._gpgformer_cache:
            return self._gpgformer_cache[key]

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg_file = yaml.safe_load(cfg_path.read_text())
        ckpt_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None

        if ckpt_cfg is not None:
            cfg_model = copy.deepcopy(ckpt_cfg)
            cfg_model.setdefault("paths", {})
            cfg_model["paths"].update(cfg_file.get("paths", {}))
        else:
            cfg_model = cfg_file

        model = build_model_from_cfg(cfg_model).to(self.device)
        warmup_lazy_modules(model, cfg_model, self.device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        self._gpgformer_cache[key] = (model, cfg_model)
        return model, cfg_model

    def load_wilor(self) -> tuple[Any, Any, Any, Any]:
        if self._wilor is not None:
            return self._wilor
        _ensure_sys_path(WILOR_ROOT)
        with pushd(WILOR_ROOT):
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.configs import get_config
            from wilor.models import WiLoR
            from wilor.utils import recursive_to
            from wilor.utils.renderer import cam_crop_to_full

            cfg_path = str(WILOR_ROOT / "pretrained_models" / "model_config.yaml")
            model_cfg = get_config(cfg_path, update_cachedir=True)

            if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in model_cfg.MODEL):
                model_cfg.defrost()
                assert model_cfg.MODEL.IMAGE_SIZE == 256, (
                    f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
                )
                model_cfg.MODEL.BBOX_SHAPE = [192, 256]
                model_cfg.freeze()

            if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
                model_cfg.defrost()
                model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
                model_cfg.freeze()

            model_cfg = self._override_mano_paths(model_cfg)
            print("Loading ", self.args.wilor_ckpt)
            model = WiLoR.load_from_checkpoint(str(self.args.wilor_ckpt), strict=False, cfg=model_cfg)
        model = model.to(self.device)
        model.eval()
        self._wilor = (model, model_cfg, ViTDetDataset, (recursive_to, cam_crop_to_full))
        return self._wilor

    def load_hamer(self) -> tuple[Any, Any, Any, Any]:
        if self._hamer is not None:
            return self._hamer
        _ensure_sys_path(HAMER_ROOT)
        with pushd(HAMER_ROOT):
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.configs import get_config
            from hamer.models import HAMER
            from hamer.utils import recursive_to
            from hamer.utils.renderer import cam_crop_to_full

            cfg_path = str(Path(self.args.hamer_ckpt).parent.parent / "model_config.yaml")
            model_cfg = get_config(cfg_path, update_cachedir=True)

            if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
                model_cfg.defrost()
                assert model_cfg.MODEL.IMAGE_SIZE == 256, (
                    f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
                )
                model_cfg.MODEL.BBOX_SHAPE = [192, 256]
                model_cfg.freeze()

            if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
                model_cfg.defrost()
                model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
                model_cfg.freeze()

            model_cfg = self._override_mano_paths(model_cfg)
            model = HAMER.load_from_checkpoint(str(self.args.hamer_ckpt), strict=False, cfg=model_cfg)
        model = model.to(self.device)
        model.eval()
        self._hamer = (model, model_cfg, ViTDetDataset, (recursive_to, cam_crop_to_full))
        return self._hamer

    def load_simplehand(self) -> tuple[Any, dict[str, Any], Any]:
        if self._simplehand is not None:
            return self._simplehand

        _ensure_sys_path(SIMPLEHAND_ROOT)
        with pushd(SIMPLEHAND_ROOT):
            cfg_mod = importlib.import_module("cfg")
            hand_net_mod = importlib.import_module("hand_net")
            kp_mod = importlib.import_module("kp_preprocess")
            model = hand_net_mod.HandNet(cfg_mod._CONFIG, pretrained=False)
            checkpoint = torch.load(str(self.args.simplehand_ckpt), map_location="cpu")
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        self._simplehand = (model, cfg_mod._CONFIG, kp_mod.get_2d3d_perspective_transform)
        return self._simplehand

    @torch.no_grad()
    def run_gpgformer(self, sample: SampleBundle) -> PredictionBundle:
        try:
            model, _ = self.load_gpgformer(sample.dataset_name)
            rgb = sample.crop_rgb_gpgformer.unsqueeze(0).to(self.device, dtype=torch.float32)
            cam_param = torch.from_numpy(sample.crop_cam_param).unsqueeze(0).to(self.device, dtype=torch.float32)
            out = model(rgb, cam_param=cam_param)

            pred_vertices = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
            pred_joints = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
            pred_cam = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)

            if not sample.is_right:
                pred_vertices[:, 0] *= -1.0
                pred_joints[:, 0] *= -1.0
                pred_cam = pred_cam.copy()
                pred_cam[1] *= -1.0

            pred_cam_t = solve_translation_from_2d_correspondences(
                pred_joints,
                sample.gt_joints_2d,
                sample.cam_param_full,
            )
            if pred_cam_t is None:
                focal_full = float(np.mean(sample.cam_param_full[:2]))
                pred_cam_t = weakcam_crop_to_full(pred_cam, sample.box_center, sample.box_size, sample.img_size_wh, focal_full)

            return PredictionBundle(
                method_name="gpgformer",
                pred_vertices=pred_vertices,
                pred_keypoints_3d=pred_joints,
                pred_cam_t=pred_cam_t,
            )
        except Exception as exc:
            return PredictionBundle("gpgformer", None, None, None, status="failed", message=str(exc))

    @torch.no_grad()
    def run_wilor(self, sample: SampleBundle) -> PredictionBundle:
        try:
            model, model_cfg, dataset_cls, helpers = self.load_wilor()
            recursive_to, cam_crop_to_full = helpers
            boxes = _bbox_xyxy_from_center_size(sample.box_center, sample.box_size, sample.img_size_wh)[None, :]
            right = np.array([1.0 if sample.is_right else 0.0], dtype=np.float32)
            dataset = dataset_cls(model_cfg, sample.full_image_bgr, boxes, right, rescale_factor=1.0)
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
            batch = recursive_to(batch, self.device)
            out = model(batch)

            multiplier = 2.0 * batch["right"] - 1.0
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            pred_vertices = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
            pred_joints = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
            if not sample.is_right:
                pred_vertices[:, 0] *= -1.0
                pred_joints[:, 0] *= -1.0

            pred_cam_t = solve_translation_from_2d_correspondences(
                pred_joints,
                sample.gt_joints_2d,
                sample.cam_param_full,
            )
            if pred_cam_t is None:
                focal_full = float(np.mean(sample.cam_param_full[:2]))
                cam_t_full = cam_crop_to_full(
                    pred_cam,
                    box_center,
                    box_size,
                    img_size,
                    torch.tensor(focal_full, device=img_size.device, dtype=img_size.dtype),
                )[0].detach().cpu().numpy()
                pred_cam_t = cam_t_full.astype(np.float32)

            return PredictionBundle("wilor", pred_vertices, pred_joints, pred_cam_t.astype(np.float32))
        except Exception as exc:
            return PredictionBundle("wilor", None, None, None, status="failed", message=str(exc))

    @torch.no_grad()
    def run_hamer(self, sample: SampleBundle) -> PredictionBundle:
        try:
            model, model_cfg, dataset_cls, helpers = self.load_hamer()
            recursive_to, cam_crop_to_full = helpers
            boxes = _bbox_xyxy_from_center_size(sample.box_center, sample.box_size, sample.img_size_wh)[None, :]
            right = np.array([1.0 if sample.is_right else 0.0], dtype=np.float32)
            dataset = dataset_cls(model_cfg, sample.full_image_bgr, boxes, right, rescale_factor=1.0)
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
            batch = recursive_to(batch, self.device)
            out = model(batch)

            multiplier = 2.0 * batch["right"] - 1.0
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            pred_vertices = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
            pred_joints = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
            if not sample.is_right:
                pred_vertices[:, 0] *= -1.0
                pred_joints[:, 0] *= -1.0

            pred_cam_t = solve_translation_from_2d_correspondences(
                pred_joints,
                sample.gt_joints_2d,
                sample.cam_param_full,
            )
            if pred_cam_t is None:
                focal_full = float(np.mean(sample.cam_param_full[:2]))
                cam_t_full = cam_crop_to_full(
                    pred_cam,
                    box_center,
                    box_size,
                    img_size,
                    torch.tensor(focal_full, device=img_size.device, dtype=img_size.dtype),
                )[0].detach().cpu().numpy()
                pred_cam_t = cam_t_full.astype(np.float32)

            return PredictionBundle("hamer", pred_vertices, pred_joints, pred_cam_t.astype(np.float32))
        except Exception as exc:
            return PredictionBundle("hamer", None, None, None, status="failed", message=str(exc))

    @torch.no_grad()
    def run_simplehand(self, sample: SampleBundle) -> PredictionBundle:
        try:
            model, cfg, perspective_fn = self.load_simplehand()
            img_size = int(cfg["DATA"]["IMAGE_SHAPE"][0])
            root_index = int(cfg["DATA"].get("ROOT_INDEX", 9))

            K = _make_camera_matrix(sample.cam_param_full)
            scale = np.array([sample.box_size, sample.box_size], dtype=np.float32) * float(DEFAULT_SIMPLEHAND_SCALE_ENLARGE)
            new_K, trans_matrix_2d, trans_matrix_3d = perspective_fn(
                K,
                sample.box_center.astype(np.float32),
                scale.astype(np.float32),
                0,
                img_size,
            )
            img_processed = cv2.warpPerspective(sample.full_image_bgr, trans_matrix_2d, (img_size, img_size))
            if img_processed.ndim == 2:
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
            image = torch.from_numpy(np.transpose(img_processed, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(self.device)

            res = model(image)
            vertices = res["vertices"].reshape(-1, 778, 3)
            joints = res["joints"].reshape(-1, 21, 3)
            uv = res["uv"].reshape(-1, 21, 2) * float(img_size)

            joints_root = joints[:, root_index : root_index + 1, :]
            joints = joints - joints_root
            vertices = vertices - joints_root

            trans_matrix_3d_inv = torch.linalg.inv(torch.from_numpy(np.asarray(trans_matrix_3d, dtype=np.float32)).to(self.device))
            trans_matrix_2d_inv = torch.linalg.inv(torch.from_numpy(np.asarray(trans_matrix_2d, dtype=np.float32)).to(self.device))

            joints = (trans_matrix_3d_inv @ torch.transpose(joints, 1, 2)).transpose(1, 2)
            vertices = (trans_matrix_3d_inv @ torch.transpose(vertices, 1, 2)).transpose(1, 2)

            pad = torch.ones((uv.shape[0], uv.shape[1], 1), device=uv.device, dtype=uv.dtype)
            uv_h = torch.cat([uv, pad], dim=-1)
            uv_h = (trans_matrix_2d_inv @ torch.transpose(uv_h, 1, 2)).transpose(1, 2)
            uv_full = uv_h[:, :, :2] / (uv_h[:, :, 2:] + 1e-7)

            pred_joints = joints[0].detach().cpu().numpy().astype(np.float32)
            pred_vertices = vertices[0].detach().cpu().numpy().astype(np.float32)
            pred_cam_t = solve_translation_from_2d_correspondences(pred_joints, sample.gt_joints_2d, sample.cam_param_full)
            if pred_cam_t is None:
                pred_cam_t = solve_translation_from_2d_correspondences(pred_joints, uv_full[0].detach().cpu().numpy(), sample.cam_param_full)

            return PredictionBundle("simplehand", pred_vertices, pred_joints, pred_cam_t)
        except Exception as exc:
            return PredictionBundle("simplehand", None, None, None, status="failed", message=str(exc))


def build_dataset(dataset_name: str, split: str | None):
    dataset_name = dataset_name.lower()
    if dataset_name == "freihand":
        return FreiHANDDatasetV3(
            root_dir=str(DEFAULT_FREIHAND_ROOT),
            eval_root=str(DEFAULT_FREIHAND_EVAL_ROOT),
            img_size=256,
            train=False,
            align_wilor_aug=True,
            bbox_source="gt",
            use_trainval_split=False,
        )

    if dataset_name == "dexycb":
        ds_split = split or "test"
        return DexYCBDataset(
            setup="s0",
            split=ds_split,
            root_dir=str(DEFAULT_DEXYCB_ROOT),
            img_size=256,
            train=False,
            align_wilor_aug=True,
            bbox_source="gt",
            root_index=9,
            mano_root=str(DEFAULT_MANO_DIR),
        )

    if dataset_name == "ho3d":
        ds_split = split or "evaluation"
        return HO3DDataset(
            data_split=ds_split,
            root_dir=str(DEFAULT_HO3D_ROOT),
            dataset_version="v3",
            img_size=256,
            train=False,
            align_wilor_aug=True,
            bbox_source="gt",
            root_index=9,
            train_split_file=str(DEFAULT_HO3D_ROOT / "train.txt"),
            eval_split_file=str(DEFAULT_HO3D_ROOT / "evaluation.txt"),
            eval_xyz_json=str(DEFAULT_HO3D_ROOT / "evaluation_xyz.json"),
            eval_verts_json=str(DEFAULT_HO3D_ROOT / "evaluation_verts.json"),
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _bundle_from_freihand(dataset: FreiHANDDatasetV3, idx: int, sample: dict[str, Any], mano_helper: MANOHelper) -> SampleBundle:
    real_idx = int(dataset.indices[idx])
    anno_idx = real_idx % len(dataset.K_list)
    img_path = str(Path(dataset.img_dir) / f"{real_idx:08d}.jpg")
    full_image_bgr = cv2.imread(img_path)
    if full_image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    K = np.asarray(dataset.K_list[anno_idx], dtype=np.float32).reshape(3, 3)
    gt_joints_3d = np.asarray(dataset.xyz_list[anno_idx], dtype=np.float32).reshape(21, 3)
    gt_vertices = sample.get("vertices_gt", None)
    if gt_vertices is not None:
        gt_vertices = _to_numpy(gt_vertices)
    if gt_vertices is None:
        mano_raw = np.asarray(dataset.mano_list[anno_idx][0], dtype=np.float32).reshape(-1)
        transl = mano_raw[58:61] / 1000.0 if mano_raw.shape[0] >= 61 else None
        gt_vertices = mano_helper.decode(mano_raw[:3], mano_raw[3:48], mano_raw[48:58], transl)

    cam_param_full = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    return SampleBundle(
        dataset_name="freihand",
        sample_idx=idx,
        full_image_bgr=full_image_bgr,
        img_path=img_path,
        box_center=_to_numpy(sample["box_center"]),
        box_size=float(_to_numpy(sample["box_size"]).reshape(-1)[0]),
        img_size_wh=np.array([full_image_bgr.shape[1], full_image_bgr.shape[0]], dtype=np.float32),
        cam_param_full=cam_param_full,
        crop_rgb_gpgformer=sample["rgb"].detach().cpu(),
        crop_cam_param=_to_numpy(sample["cam_param"]),
        is_right=True,
        gt_joints_3d=gt_joints_3d,
        gt_joints_2d=_project_points(gt_joints_3d, cam_param_full),
        gt_vertices=gt_vertices,
        gt_faces=mano_helper.faces,
    )


def _bundle_from_dexycb(dataset: DexYCBDataset, idx: int, sample: dict[str, Any], mano_helper: MANOHelper) -> SampleBundle:
    data = dataset.datalist[idx]
    img_path = str(data["img_path"])
    full_image_bgr = cv2.imread(img_path)
    if full_image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    cam_info = data["cam_param"]
    cam_param_full = np.array(
        [cam_info["focal"][0], cam_info["focal"][1], cam_info["princpt"][0], cam_info["princpt"][1]],
        dtype=np.float32,
    )
    gt_joints_3d = np.asarray(data["joints_coord_cam"], dtype=np.float32).reshape(21, 3)

    gt_vertices = sample.get("vertices_gt", None)
    if gt_vertices is not None:
        gt_vertices = _to_numpy(gt_vertices)
    if gt_vertices is None:
        hand_type = str(data["hand_type"])
        is_right = hand_type.lower().startswith("r")
        global_orient, hand_pose = dataset._decode_mano_pose(np.asarray(data["mano_pose"], dtype=np.float32), hand_type)
        betas = np.asarray(data["mano_shape"], dtype=np.float32).reshape(10)
        transl = data.get("mano_trans", None)
        transl_m = np.asarray(transl, dtype=np.float32).reshape(3) if transl is not None else None
        if transl_m is None:
            temp_vertices = mano_helper.decode(global_orient, hand_pose, betas, None, is_right=is_right)
            temp_joints = _project_points(gt_joints_3d, cam_param_full)
            transl_m = solve_translation_from_2d_correspondences(gt_joints_3d, temp_joints, cam_param_full)
        gt_vertices = mano_helper.decode(global_orient, hand_pose, betas, transl_m, is_right=is_right)

    return SampleBundle(
        dataset_name="dexycb",
        sample_idx=idx,
        full_image_bgr=full_image_bgr,
        img_path=img_path,
        box_center=_to_numpy(sample["box_center"]),
        box_size=float(_to_numpy(sample["box_size"]).reshape(-1)[0]),
        img_size_wh=np.array([full_image_bgr.shape[1], full_image_bgr.shape[0]], dtype=np.float32),
        cam_param_full=cam_param_full,
        crop_rgb_gpgformer=sample["rgb"].detach().cpu(),
        crop_cam_param=_to_numpy(sample["cam_param"]),
        is_right=bool(float(sample.get("is_right", 1.0)) > 0.5),
        gt_joints_3d=gt_joints_3d,
        gt_joints_2d=_project_points(gt_joints_3d, cam_param_full),
        gt_vertices=gt_vertices,
        gt_faces=mano_helper.faces,
    )


def _bundle_from_ho3d(dataset: HO3DDataset, idx: int, sample: dict[str, Any], mano_helper: MANOHelper) -> SampleBundle:
    data = dataset.datalist[idx]
    img_path = str(data["img_path"])
    full_image_bgr = cv2.imread(img_path)
    if full_image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    K = np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
    cam_param_full = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    gt_joints_raw = _ho3d_cam_to_std_xyz(np.asarray(data["joints_coord_cam"], dtype=np.float32).reshape(21, 3))
    gt_joints_3d = gt_joints_raw[np.asarray(HO3D_META_JOINT_MAP, dtype=np.int64)][np.asarray(WILOR_JOINT_MAP, dtype=np.int64)]

    gt_vertices = sample.get("vertices_gt", None)
    if gt_vertices is not None:
        gt_vertices = _to_numpy(gt_vertices)
    if gt_vertices is None and data.get("vertices_cam", None) is not None:
        gt_vertices = _ho3d_cam_to_std_xyz(np.asarray(data["vertices_cam"], dtype=np.float32).reshape(-1, 3))
    if gt_vertices is None and not bool(data.get("is_eval", False)):
        pose = np.asarray(data["mano_pose"], dtype=np.float32).reshape(48)
        betas = np.asarray(data["mano_shape"], dtype=np.float32).reshape(10)
        transl = _ho3d_cam_to_std_xyz(np.asarray(data["mano_trans"], dtype=np.float32).reshape(3))
        gt_vertices = mano_helper.decode(_ho3d_cam_to_std_global_orient(pose[:3]), pose[3:], betas, transl)

    return SampleBundle(
        dataset_name="ho3d",
        sample_idx=idx,
        full_image_bgr=full_image_bgr,
        img_path=img_path,
        box_center=_to_numpy(sample["box_center"]),
        box_size=float(_to_numpy(sample["box_size"]).reshape(-1)[0]),
        img_size_wh=np.array([full_image_bgr.shape[1], full_image_bgr.shape[0]], dtype=np.float32),
        cam_param_full=cam_param_full,
        crop_rgb_gpgformer=sample["rgb"].detach().cpu(),
        crop_cam_param=_to_numpy(sample["cam_param"]),
        is_right=True,
        gt_joints_3d=gt_joints_3d,
        gt_joints_2d=_project_points(gt_joints_3d, cam_param_full),
        gt_vertices=gt_vertices,
        gt_faces=mano_helper.faces,
    )


def make_sample_bundle(dataset, idx: int, dataset_name: str, mano_helper: MANOHelper) -> SampleBundle:
    sample = dataset[idx]
    if dataset_name == "freihand":
        return _bundle_from_freihand(dataset, idx, sample, mano_helper)
    if dataset_name == "dexycb":
        return _bundle_from_dexycb(dataset, idx, sample, mano_helper)
    if dataset_name == "ho3d":
        return _bundle_from_ho3d(dataset, idx, sample, mano_helper)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_runners(device: torch.device, cli_args: argparse.Namespace) -> RunnerStore:
    return RunnerStore(cli_args, device, MANOHelper(device))


def _format_method_line(method_name: str, pred: PredictionBundle) -> str:
    if pred.status != "ok":
        return f"{method_name}: {pred.status}: {pred.message[:20]}"
    metrics = pred.metrics or {}
    parts: list[str] = []
    joint_rr = metrics.get("joint_rr_mm")
    vertex_rr = metrics.get("vertex_rr_mm")
    score_mm = metrics.get("score_mm")
    if joint_rr is not None:
        parts.append(f"J {joint_rr:.1f}")
    if vertex_rr is not None:
        parts.append(f"V {vertex_rr:.1f}")
    if score_mm is not None:
        parts.append(f"S {score_mm:.1f}")
    return f"{method_name}: {' '.join(parts) if parts else 'ok'}"


def compose_panel(
    sample: SampleBundle,
    gt_cam_t: np.ndarray | None,
    gt_overlay: np.ndarray | None,
    gt_mesh: np.ndarray | None,
    preds_by_method: dict[str, PredictionBundle],
    render_size: int,
) -> np.ndarray:
    info_lines = [
        f"dataset: {sample.dataset_name}",
        f"index: {sample.sample_idx}",
        f"hand: {'right' if sample.is_right else 'left'}",
        f"image: {Path(sample.img_path).name}",
    ]
    for method_name in ("gpgformer", "wilor", "hamer", "simplehand"):
        pred = preds_by_method[method_name]
        info_lines.append(_format_method_line(method_name, pred))

    top_tiles = [
        _add_title(_fit_tile(sample.full_image_bgr, render_size), "Image"),
        _add_title(
            _fit_tile(gt_overlay if gt_overlay is not None else _render_placeholder("GT", "overlay unavailable", render_size), render_size),
            "GT Overlay",
        ),
    ]
    bottom_tiles = [
        _add_title(
            _fit_tile(gt_mesh if gt_mesh is not None else _render_placeholder("GT", "mesh unavailable", render_size), render_size),
            "GT Mesh",
        ),
        _add_title(_fit_tile(_render_text_panel(info_lines, render_size), render_size), "GT Info"),
    ]

    for method_name, label in (
        ("gpgformer", "GPGFormer Overlay"),
        ("wilor", "WiLoR Overlay"),
        ("hamer", "HaMeR Overlay"),
        ("simplehand", "SimpleHand Overlay"),
    ):
        pred = preds_by_method[method_name]
        overlay = getattr(pred, "overlay_bgr", None)
        if overlay is None:
            overlay = _render_placeholder(method_name, pred.message or "failed", render_size)
        top_tiles.append(_add_title(_fit_tile(overlay, render_size), label))

    for method_name, label in (
        ("gpgformer", "GPGFormer Mesh"),
        ("wilor", "WiLoR Mesh"),
        ("hamer", "HaMeR Mesh"),
        ("simplehand", "SimpleHand Mesh"),
    ):
        pred = preds_by_method[method_name]
        mesh_only = getattr(pred, "mesh_bgr", None)
        if mesh_only is None:
            mesh_only = _render_placeholder(method_name, pred.message or "failed", render_size)
        bottom_tiles.append(_add_title(_fit_tile(mesh_only, render_size), label))

    top_row = np.concatenate(top_tiles, axis=1)
    bottom_row = np.concatenate(bottom_tiles, axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


def write_outputs(
    out_dir: Path,
    sample: SampleBundle,
    panel_bgr: np.ndarray,
    gt_overlay: np.ndarray | None,
    gt_mesh: np.ndarray | None,
    preds_by_method: dict[str, PredictionBundle],
    save_individual: bool,
) -> dict[str, Any]:
    sample_dir = out_dir / f"index_{sample.sample_idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = f"index_{sample.sample_idx:06d}"
    overview_name = f"{stem}_overview.png"
    cv2.imwrite(str(sample_dir / overview_name), panel_bgr)

    raw_image_name = f"{stem}_image.png"
    cv2.imwrite(str(sample_dir / raw_image_name), sample.full_image_bgr)

    entry: dict[str, Any] = {
        "dataset": sample.dataset_name,
        "sample_idx": sample.sample_idx,
        "img_path": sample.img_path,
        "is_right": bool(sample.is_right),
        "sample_dir": sample_dir.name,
        "raw_image_file": str(Path(sample_dir.name) / raw_image_name),
        "overview_file": str(Path(sample_dir.name) / overview_name),
        "methods": {},
    }

    if save_individual:
        if gt_overlay is not None:
            gt_overlay_name = f"{stem}_gt_overlay.png"
            cv2.imwrite(str(sample_dir / gt_overlay_name), gt_overlay)
            entry["gt_overlay_file"] = str(Path(sample_dir.name) / gt_overlay_name)
        if gt_mesh is not None:
            gt_mesh_name = f"{stem}_gt_mesh.png"
            cv2.imwrite(str(sample_dir / gt_mesh_name), gt_mesh)
            entry["gt_mesh_file"] = str(Path(sample_dir.name) / gt_mesh_name)

    for method_name, pred in preds_by_method.items():
        method_entry: dict[str, Any] = {
            "status": pred.status,
            "message": pred.message,
            "cam_t": pred.pred_cam_t.tolist() if pred.pred_cam_t is not None else None,
            "metrics": pred.metrics,
        }
        overlay = getattr(pred, "overlay_bgr", None)
        mesh_only = getattr(pred, "mesh_bgr", None)
        if save_individual and overlay is not None:
            overlay_name = f"{stem}_{method_name}_overlay.png"
            cv2.imwrite(str(sample_dir / overlay_name), overlay)
            method_entry["overlay_file"] = str(Path(sample_dir.name) / overlay_name)
        if save_individual and mesh_only is not None:
            mesh_name = f"{stem}_{method_name}_mesh.png"
            cv2.imwrite(str(sample_dir / mesh_name), mesh_only)
            method_entry["mesh_file"] = str(Path(sample_dir.name) / mesh_name)
        entry["methods"][method_name] = method_entry
    return entry


def _prepare_prediction_visuals(
    sample: SampleBundle,
    pred: PredictionBundle,
    render_size: int,
    mesh_view_azim_deg: float,
    mesh_view_elev_deg: float,
) -> PredictionBundle:
    if pred.status != "ok" or pred.pred_vertices is None or pred.pred_cam_t is None or sample.gt_faces is None:
        return pred
    try:
        pred.overlay_bgr = render_overlay(
            sample.full_image_bgr,
            pred.pred_vertices,
            sample.gt_faces,
            pred.pred_cam_t,
            sample.cam_param_full,
            COLOR_HEX[pred.method_name],
            sample.is_right,
        )
        pred.mesh_bgr = render_mesh_only(
            pred.pred_vertices,
            sample.gt_faces,
            COLOR_HEX[pred.method_name],
            sample.is_right,
            int(render_size),
            view_azim_deg=float(mesh_view_azim_deg),
            view_elev_deg=float(mesh_view_elev_deg),
        )
    except Exception as exc:
        pred.status = "failed"
        pred.message = f"render failed: {exc}"
    return pred


def _select_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.index is not None:
        indices = [int(args.index)]
    elif args.indices:
        indices = [int(v) for v in args.indices]
    elif args.index_range is not None:
        start, end = [int(v) for v in args.index_range]
        if end <= start:
            raise ValueError(f"--index-range expects end > start, got start={start}, end={end}")
        indices = list(range(start, end))
    else:
        count = min(int(args.num_samples), dataset_len)
        indices = list(range(count))

    invalid = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if invalid:
        preview = ", ".join(str(v) for v in invalid[:8])
        suffix = "..." if len(invalid) > 8 else ""
        raise IndexError(f"Sample indices out of range for dataset of size {dataset_len}: {preview}{suffix}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hand mesh reconstructions across four methods.")
    parser.add_argument("--dataset", required=True, choices=["freihand", "dexycb", "ho3d"])
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--index-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Visualize a half-open index range [START, END). Example: --index-range 50 100 generates 50 samples.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--render-size", type=int, default=DEFAULT_PANEL_TILE)
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--show-gpgformer-better", action="store_true")
    parser.add_argument("--gpgformer-better-margin-mm", type=float, default=10.0)
    parser.add_argument("--mesh-view-azim", type=float, default=DEFAULT_MESH_VIEW_AZIM_DEG)
    parser.add_argument("--mesh-view-elev", type=float, default=DEFAULT_MESH_VIEW_ELEV_DEG)
    parser.add_argument("--gpgformer-config", type=str, default=None)
    parser.add_argument("--gpgformer-ckpt", type=str, default=None)
    parser.add_argument("--wilor-ckpt", type=str, default=str(DEFAULT_WILOR_CKPT))
    parser.add_argument("--hamer-ckpt", type=str, default=str(DEFAULT_HAMER_CKPT))
    parser.add_argument("--simplehand-ckpt", type=str, default=str(DEFAULT_SIMPLEHAND_CKPT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.indices = args.indices or []
    args.wilor_ckpt = Path(args.wilor_ckpt)
    args.hamer_ckpt = Path(args.hamer_ckpt)
    args.simplehand_ckpt = Path(args.simplehand_ckpt)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args.dataset, args.split)
    runners = load_runners(device, args)
    indices = _select_indices(len(dataset), args)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "outputs" / "compare_hand_meshes" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    selected_count = 0
    enforce_visible_hand_region = bool(args.dataset == "dexycb" and args.show_gpgformer_better)
    for idx in indices:
        sample = make_sample_bundle(dataset, idx, args.dataset, runners.mano_helper)

        if enforce_visible_hand_region and not _hand_region_visible_in_image(sample):
            continue

        preds_by_method = {
            "gpgformer": runners.run_gpgformer(sample),
            "wilor": runners.run_wilor(sample),
            "hamer": runners.run_hamer(sample),
            "simplehand": runners.run_simplehand(sample),
        }
        for pred in preds_by_method.values():
            pred.metrics = _compute_prediction_metrics(sample, pred, DEFAULT_ROOT_INDEX)

        if bool(args.show_gpgformer_better) and not _is_gpgformer_clearly_better(
            preds_by_method,
            float(args.gpgformer_better_margin_mm),
        ):
            continue

        gt_overlay = None
        gt_mesh = None
        gt_cam_t = None
        if sample.gt_vertices is not None and sample.gt_faces is not None:
            gt_cam_t = solve_translation_from_2d_correspondences(sample.gt_joints_3d, sample.gt_joints_2d, sample.cam_param_full)
            if gt_cam_t is None:
                gt_cam_t = np.zeros((3,), dtype=np.float32)
            gt_overlay = render_overlay(
                sample.full_image_bgr,
                sample.gt_vertices,
                sample.gt_faces,
                gt_cam_t,
                sample.cam_param_full,
                COLOR_HEX["gt"],
                sample.is_right,
            )
            gt_mesh = render_mesh_only(
                sample.gt_vertices,
                sample.gt_faces,
                COLOR_HEX["gt_mesh"],
                sample.is_right,
                int(args.render_size),
                view_azim_deg=float(args.mesh_view_azim),
                view_elev_deg=float(args.mesh_view_elev),
            )

        for method_name in list(preds_by_method.keys()):
            preds_by_method[method_name] = _prepare_prediction_visuals(
                sample,
                preds_by_method[method_name],
                int(args.render_size),
                float(args.mesh_view_azim),
                float(args.mesh_view_elev),
            )

        panel_bgr = compose_panel(sample, gt_cam_t, gt_overlay, gt_mesh, preds_by_method, int(args.render_size))
        entry = write_outputs(out_dir, sample, panel_bgr, gt_overlay, gt_mesh, preds_by_method, bool(args.save_individual))
        summary.append(entry)
        selected_count += 1

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if bool(args.show_gpgformer_better):
        message = (
            f"Selected {selected_count} / {len(indices)} samples where GPGFormer beats WiLoR/HaMeR/SimpleHand "
            f"by >= {float(args.gpgformer_better_margin_mm):.1f} mm on both joint and combined scores."
        )
        if enforce_visible_hand_region:
            message += " DexYCB samples without a visible hand region in the source image were also filtered out."
        print(message)


if __name__ == "__main__":
    main()

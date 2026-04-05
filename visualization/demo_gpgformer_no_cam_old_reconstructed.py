from __future__ import annotations

import argparse
import copy
import json
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

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/WiLoR/demo_img")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_demo_no_cam")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    "/root/code/vepfs/GPGFormer/checkpoints/"
    "freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_MESH_COLOR = "#7677D8"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPGFormer hand mesh visualization without ground-truth camera intrinsics."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR), help="Input image directory.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG), help="GPGFormer YAML config.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="GPGFormer checkpoint.")
    parser.add_argument("--detector-ckpt", type=str, default=str(DEFAULT_DETECTOR_CKPT), help="YOLO hand detector.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, e.g. cuda or cpu.")
    parser.add_argument("--detector-conf", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--detector-iou", type=float, default=0.6, help="YOLO NMS IoU threshold.")
    parser.add_argument("--bbox-expand", type=float, default=2.0, help="Square crop expansion factor.")
    parser.add_argument("--default-focal-full", type=float, default=5000.0, help="Default full-image focal length used to solve tvec.")
    parser.add_argument(
        "--overlay-mode",
        type=str,
        default="image_space",
        choices=("image_space", "perspective"),
        help="How to render overlays: model-native weak-perspective image-space mapping, or perspective+tvec recovery.",
    )
    parser.add_argument("--tvec-mode", type=str, default="detector_refine", choices=("detector_refine", "solve", "weakcam"), help="How to recover full-image camera translation.")
    parser.add_argument("--max-images", type=int, default=0, help="Process at most N images, 0 for all.")
    parser.add_argument("--max-dets-per-image", type=int, default=8, help="Keep top-N detections per image.")
    parser.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR, help="Mesh color in #RRGGBB.")
    parser.add_argument("--save-crop", action="store_true", help="Also save the normalized crop input.")
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _as_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _det_color_bgr(is_right: bool) -> tuple[int, int, int]:
    return (80, 200, 120) if is_right else (60, 170, 255)


def _crop_color_bgr() -> tuple[int, int, int]:
    return (255, 255, 0)


def square_bbox_from_center_size(center_xy: np.ndarray, box_size: float) -> np.ndarray:
    cx, cy = [float(v) for v in np.asarray(center_xy, dtype=np.float32).reshape(2)]
    half = float(box_size) * 0.5
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def draw_labeled_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_y2 = max(y1, th + baseline + 6)
    text_y1 = text_y2 - th - baseline - 6
    text_x2 = min(x1 + tw + 8, out.shape[1] - 1)
    text_x1 = max(0, text_x2 - (tw + 8))
    cv2.rectangle(out, (text_x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(out, label, (text_x1 + 4, text_y2 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, lineType=cv2.LINE_AA)
    return out


def draw_detection_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    score: float | None = None,
) -> np.ndarray:
    label = "right" if is_right else "left"
    if score is not None:
        label = f"{label} {float(score):.2f}"
    return draw_labeled_bbox(image_bgr, bbox_xyxy, _det_color_bgr(is_right), label, thickness=2)


def draw_crop_bbox(image_bgr: np.ndarray, crop_bbox_xyxy: np.ndarray) -> np.ndarray:
    return draw_labeled_bbox(image_bgr, crop_bbox_xyxy, _crop_color_bgr(), "crop", thickness=1)


def draw_detection_bboxes(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    for det in detections:
        out = draw_detection_bbox(out, det["bbox_xyxy"], bool(det["is_right"]), det.get("score"))
    return out


def load_yolo_detector(weights_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'ultralytics'. Install it in the runtime environment "
            "used for this demo before running the script."
        ) from exc
    return YOLO(str(weights_path))


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
    scene_bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    import pyrender

    mesh_vertices = np.asarray(vertices, dtype=np.float32).copy()
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
    )
    fg_rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    comp_rgb = bg_rgb * (1.0 - alpha) + fg_rgb * alpha
    return cv2.cvtColor(np.clip(comp_rgb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_overlay_multiple(
    full_image_bgr: np.ndarray,
    vertices_list: list[np.ndarray],
    faces: np.ndarray,
    cam_t_list: list[np.ndarray],
    cam_param: np.ndarray,
    color_hex: str,
    is_right_list: list[bool],
) -> np.ndarray:
    img_bgr = np.asarray(full_image_bgr, dtype=np.uint8)
    bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    comp_rgb = bg_rgb.copy()
    for vertices, cam_t, is_right in zip(vertices_list, cam_t_list, is_right_list):
        rgba = _render_mesh_rgba(
            vertices=vertices,
            faces=faces,
            color_rgb=_as_rgb01(color_hex),
            is_right=is_right,
            width=img_bgr.shape[1],
            height=img_bgr.shape[0],
            fx=float(cam_param[0]),
            fy=float(cam_param[1]),
            cx=float(cam_param[2]),
            cy=float(cam_param[3]),
            camera_translation=cam_t,
        )
        fg_rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        comp_rgb = comp_rgb * (1.0 - alpha) + fg_rgb * alpha
    return cv2.cvtColor(np.clip(comp_rgb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_mesh_rgba_image_space(
    vertices_2d: np.ndarray,
    vertices_depth: np.ndarray,
    faces: np.ndarray,
    color_hex: str,
    width: int,
    height: int,
) -> np.ndarray:
    verts2d = np.asarray(vertices_2d, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(vertices_depth, dtype=np.float32).reshape(-1)
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    color_rgb = np.array(_as_rgb01(color_hex), dtype=np.float32)
    color_bgr = tuple(int(round(c * 255.0)) for c in color_rgb[::-1])
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)

    tri_depth = depths[faces_arr].mean(axis=1)
    order = np.argsort(-tri_depth)
    for face_idx in order.tolist():
        tri = faces_arr[face_idx]
        pts = verts2d[tri]
        if not np.isfinite(pts).all():
            continue
        poly = np.round(pts).astype(np.int32)
        if poly.shape != (3, 2):
            continue
        if ((poly[:, 0] < -width).all() or (poly[:, 0] > 2 * width).all() or (poly[:, 1] < -height).all() or (poly[:, 1] > 2 * height).all()):
            continue
        cv2.fillConvexPoly(rgb, poly, color_bgr, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(alpha, poly, 180, lineType=cv2.LINE_AA)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha
    return rgba


def composite_rgba_over_bgr(image_bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    img_bgr = np.asarray(image_bgr, dtype=np.uint8)
    fg_rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    comp_rgb = bg_rgb * (1.0 - alpha) + fg_rgb * alpha
    return cv2.cvtColor(np.clip(comp_rgb, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_overlay_image_space(
    full_image_bgr: np.ndarray,
    vertices_2d: np.ndarray,
    vertices_depth: np.ndarray,
    faces: np.ndarray,
    color_hex: str,
) -> np.ndarray:
    h, w = full_image_bgr.shape[:2]
    rgba = render_mesh_rgba_image_space(vertices_2d, vertices_depth, faces, color_hex, width=w, height=h)
    return composite_rgba_over_bgr(full_image_bgr, rgba)


def render_overlay_multiple_image_space(
    full_image_bgr: np.ndarray,
    vertices_2d_list: list[np.ndarray],
    vertices_depth_list: list[np.ndarray],
    faces: np.ndarray,
    color_hex: str,
) -> np.ndarray:
    h, w = full_image_bgr.shape[:2]
    rgba_all = np.zeros((h, w, 4), dtype=np.uint8)
    for vertices_2d, vertices_depth in zip(vertices_2d_list, vertices_depth_list):
        rgba = render_mesh_rgba_image_space(vertices_2d, vertices_depth, faces, color_hex, width=w, height=h)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        prev_rgb = rgba_all[:, :, :3].astype(np.float32)
        prev_a = rgba_all[:, :, 3:4].astype(np.float32) / 255.0
        new_rgb = rgba[:, :, :3].astype(np.float32)
        out_a = alpha + prev_a * (1.0 - alpha)
        denom = np.clip(out_a, 1e-6, 1.0)
        out_rgb = (new_rgb * alpha + prev_rgb * prev_a * (1.0 - alpha)) / denom
        rgba_all[:, :, :3] = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)
        rgba_all[:, :, 3:4] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return composite_rgba_over_bgr(full_image_bgr, rgba_all)


def render_mesh_only(
    vertices: np.ndarray,
    faces: np.ndarray,
    color_hex: str,
    is_right: bool,
    render_size: int = 512,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    centered = verts - verts.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=1).max()) if centered.size else 0.1
    cam_z = max(0.7, radius * 6.0)
    focal = float(render_size) * 2.4
    rgba = _render_mesh_rgba(
        vertices=centered,
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
    )
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, 255, dtype=np.uint8).astype(np.float32)
    comp = bg * (1.0 - alpha) + rgb.astype(np.float32) * alpha
    return cv2.cvtColor(np.clip(comp, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


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


def affine_transform_points(points_2d: np.ndarray, trans_2x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    trans = np.asarray(trans_2x3, dtype=np.float32).reshape(2, 3)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = (trans @ homo.T).T
    return out.astype(np.float32)


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

    inspected = []
    for obj in candidates:
        inspected.append(type(obj).__name__)
    raise AttributeError(
        "Could not find MANO face topology on model.mano. "
        f"Inspected objects: {inspected}."
    )


def collect_image_paths(img_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(img_dir.glob(pattern)))
    uniq = sorted({path.resolve() for path in image_paths})
    return [Path(p) for p in uniq]


def detect_hands(
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
    for idx in order[: max(0, int(max_dets))]:
        x1, y1, x2, y2 = boxes[idx].tolist()
        detections.append(
            {
                "bbox_xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "score": float(scores[idx]),
                "is_right": bool(float(classes[idx]) > 0.5),
            }
        )
    return detections


def prepare_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    cfg: dict[str, Any],
    bbox_expand: float,
    default_focal_full: float,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    bw = max(x2 - x1, 2.0)
    bh = max(y2 - y1, 2.0)
    box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    box_size = max(bw, bh) * float(bbox_expand)

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    image_size = int(model_cfg.get("image_size", dataset_cfg.get("img_size", 256)))
    image_width = int(model_cfg.get("image_width", int(round(image_size * 0.75))))

    keypoints_2d = np.zeros((21, 3), dtype=np.float32)
    keypoints_3d = np.zeros((21, 3), dtype=np.float32)
    mano_params = {
        "global_orient": np.zeros((3,), dtype=np.float32),
        "hand_pose": np.zeros((45,), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
    }
    has_mano_params = {
        "global_orient": np.zeros((1,), dtype=np.float32),
        "hand_pose": np.zeros((1,), dtype=np.float32),
        "betas": np.zeros((1,), dtype=np.float32),
    }

    img_patch, _, _, _, _, _, trans = get_example(
        image_bgr,
        box_center[0],
        box_center[1],
        box_size,
        box_size,
        keypoints_2d,
        keypoints_3d,
        mano_params,
        has_mano_params,
        list(range(21)),
        image_width,
        image_size,
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

    focal = float(model_cfg.get("focal_length", 5000.0))
    focal_full = float(default_focal_full)
    cam_param_full = np.array([focal_full, focal_full, w / 2.0, h / 2.0], dtype=np.float32)
    cam_param_patch = np.array([focal, focal, image_width / 2.0, image_size / 2.0], dtype=np.float32)

    return {
        "crop_rgb": crop_rgb,
        "crop_rgb01": crop_rgb01,
        "box_center": box_center,
        "box_size": float(box_size),
        "crop_bbox_xyxy": square_bbox_from_center_size(box_center, box_size),
        "img_size_wh": np.array([w, h], dtype=np.float32),
        "cam_param_full": cam_param_full,
        "cam_param_patch": cam_param_patch,
        "focal_full": float(focal_full),
        "crop_to_full_trans": cv2.invertAffineTransform(np.asarray(trans, dtype=np.float32)),
        "patch_height": int(image_size),
    }


@torch.no_grad()
def infer_one_hand(
    model: Any,
    crop_rgb: torch.Tensor,
    cam_param_full: np.ndarray,
    cam_param_patch: np.ndarray,
    crop_to_full_trans: np.ndarray,
    det_bbox_xyxy: np.ndarray,
    box_center: np.ndarray,
    box_size: float,
    img_size_wh: np.ndarray,
    patch_height: int,
    focal_length: float,
    tvec_mode: str,
    is_right: bool,
) -> dict[str, Any]:
    rgb = crop_rgb.unsqueeze(0).to(next(model.parameters()).device, dtype=torch.float32)
    cam_param = torch.from_numpy(cam_param_patch).unsqueeze(0).to(rgb.device, dtype=torch.float32)
    out = model(rgb, cam_param=cam_param)

    pred_vertices_crop = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
    pred_keypoints_3d_crop = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
    pred_cam_crop = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)

    pred_vertices = pred_vertices_crop.copy()
    pred_keypoints_3d = pred_keypoints_3d_crop.copy()
    pred_cam = pred_cam_crop.copy()

    if not is_right:
        pred_vertices[:, 0] *= -1.0
        pred_keypoints_3d[:, 0] *= -1.0
        pred_cam[1] *= -1.0

    pred_cam_t_weakcam = weakcam_crop_to_full(
        pred_cam=pred_cam,
        box_center=box_center,
        box_size=box_size,
        img_size_wh=img_size_wh,
        focal_length=focal_length,
    )
    pred_cam_t_full = pred_cam_t_weakcam.copy()

    crop_cam_t = np.array(
        [
            pred_cam_crop[1],
            pred_cam_crop[2],
            2.0 * float(cam_param_patch[0]) / (float(patch_height) * float(pred_cam_crop[0]) + 1e-9),
        ],
        dtype=np.float32,
    )
    pred_keypoints_2d_crop = project_points_with_translation(pred_keypoints_3d_crop, cam_param_patch, crop_cam_t)
    pred_keypoints_2d_full = affine_transform_points(pred_keypoints_2d_crop, crop_to_full_trans)
    pred_vertices_2d_crop = project_points_with_translation(pred_vertices_crop, cam_param_patch, crop_cam_t)
    pred_vertices_2d_full = affine_transform_points(pred_vertices_2d_crop, crop_to_full_trans)
    if not is_right:
        pred_keypoints_2d_full[:, 0] = float(img_size_wh[0]) - pred_keypoints_2d_full[:, 0] - 1.0
        pred_vertices_2d_full[:, 0] = float(img_size_wh[0]) - pred_vertices_2d_full[:, 0] - 1.0

    solved_cam_t = solve_translation_from_2d_correspondences(pred_keypoints_3d, pred_keypoints_2d_full, cam_param_full)
    if tvec_mode == "solve" and solved_cam_t is not None:
        pred_cam_t_full = solved_cam_t
    elif tvec_mode == "detector_refine":
        pred_cam_t_full = refine_translation_to_bbox(
            vertices_3d=pred_vertices,
            cam_param=cam_param_full,
            init_cam_t=pred_cam_t_weakcam,
            target_bbox_xyxy=det_bbox_xyxy,
        )

    return {
        "pred_vertices": pred_vertices,
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_cam": pred_cam,
        "pred_cam_t_full": pred_cam_t_full,
        "pred_cam_t_weakcam": pred_cam_t_weakcam,
        "pred_cam_t_solved": solved_cam_t,
        "pred_keypoints_2d_full": pred_keypoints_2d_full,
        "pred_vertices_2d_full": pred_vertices_2d_full,
        "pred_vertices_depth": (pred_vertices_crop[:, 2] + crop_cam_t[2]).astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    detector_ckpt = Path(args.detector_ckpt)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overlay_mode == "image_space":
        print("[demo_gpgformer_no_cam] overlay_mode=image_space, so tvec_mode/default_focal_full do not affect overlay rendering.")
        print("[demo_gpgformer_no_cam] Use --overlay-mode perspective if you want to compare weakcam/solve/detector_refine visually.")

    device = _resolve_device(args.device)
    detector_device = "cpu" if device.type == "cpu" else str(device)

    model, cfg = load_gpgformer(cfg_path, ckpt_path, device)
    detector = load_yolo_detector(detector_ckpt)
    faces = extract_mano_faces(model)
    focal_length = float(cfg.get("model", {}).get("focal_length", 5000.0))

    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]

    summary: dict[str, Any] = {
        "img_dir": str(img_dir),
        "out_dir": str(out_dir),
        "config": str(cfg_path),
        "ckpt": str(ckpt_path),
        "detector_ckpt": str(detector_ckpt),
        "device": str(device),
        "overlay_mode": args.overlay_mode,
        "tvec_mode": args.tvec_mode,
        "num_images": len(image_paths),
        "results": [],
    }

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            summary["results"].append({"image": str(image_path), "status": "failed_to_read"})
            continue

        detections = detect_hands(
            detector=detector,
            image_bgr=image_bgr,
            conf=args.detector_conf,
            iou=args.detector_iou,
            device=detector_device,
            max_dets=args.max_dets_per_image,
        )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        image_result: dict[str, Any] = {
            "image": str(image_path),
            "status": "ok" if detections else "no_detection",
            "overlay_mode": args.overlay_mode,
            "detections": [],
        }
        all_vertices_2d: list[np.ndarray] = []
        all_vertices_depth: list[np.ndarray] = []
        all_vertices: list[np.ndarray] = []
        all_cam_t: list[np.ndarray] = []
        all_is_right: list[bool] = []
        all_crop_bboxes: list[np.ndarray] = []
        image_cam_param_full: np.ndarray | None = None

        for det_idx, det in enumerate(detections):
            crop_pack = prepare_crop(
                image_bgr=image_bgr,
                bbox_xyxy=det["bbox_xyxy"],
                is_right=det["is_right"],
                cfg=cfg,
                bbox_expand=args.bbox_expand,
                default_focal_full=args.default_focal_full,
            )
            pred = infer_one_hand(
                model=model,
                crop_rgb=crop_pack["crop_rgb"],
                cam_param_full=crop_pack["cam_param_full"],
                cam_param_patch=crop_pack["cam_param_patch"],
                crop_to_full_trans=crop_pack["crop_to_full_trans"],
                det_bbox_xyxy=det["bbox_xyxy"],
                box_center=crop_pack["box_center"],
                box_size=crop_pack["box_size"],
                img_size_wh=crop_pack["img_size_wh"],
                patch_height=crop_pack["patch_height"],
                focal_length=crop_pack["focal_full"],
                tvec_mode=args.tvec_mode,
                is_right=det["is_right"],
            )

            all_vertices_2d.append(pred["pred_vertices_2d_full"])
            all_vertices_depth.append(pred["pred_vertices_depth"])
            all_vertices.append(pred["pred_vertices"])
            all_cam_t.append(pred["pred_cam_t_full"])
            all_is_right.append(bool(det["is_right"]))
            all_crop_bboxes.append(crop_pack["crop_bbox_xyxy"])
            image_cam_param_full = crop_pack["cam_param_full"]

            if args.overlay_mode == "image_space":
                overlay = render_overlay_image_space(
                    full_image_bgr=image_bgr,
                    vertices_2d=pred["pred_vertices_2d_full"],
                    vertices_depth=pred["pred_vertices_depth"],
                    faces=faces,
                    color_hex=args.mesh_color,
                )
            else:
                overlay = render_overlay(
                    full_image_bgr=image_bgr,
                    vertices=pred["pred_vertices"],
                    faces=faces,
                    cam_t=pred["pred_cam_t_full"],
                    cam_param=crop_pack["cam_param_full"],
                    color_hex=args.mesh_color,
                    is_right=det["is_right"],
                )
            overlay = draw_crop_bbox(overlay, crop_pack["crop_bbox_xyxy"])
            overlay = draw_detection_bbox(overlay, det["bbox_xyxy"], bool(det["is_right"]), det.get("score"))
            mesh_only = render_mesh_only(
                vertices=pred["pred_vertices"],
                faces=faces,
                color_hex=args.mesh_color,
                is_right=det["is_right"],
                render_size=512,
            )

            prefix = f"det_{det_idx:02d}"
            overlay_path = image_out_dir / f"{prefix}_overlay.png"
            mesh_path = image_out_dir / f"{prefix}_mesh.png"
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(mesh_path), mesh_only)

            crop_path = None
            if args.save_crop:
                crop_vis = (crop_pack["crop_rgb01"].permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0).clip(0.0, 255.0).astype(np.uint8)
                crop_path = image_out_dir / f"{prefix}_crop.png"
                cv2.imwrite(str(crop_path), crop_vis)

            meta = {
                "bbox_xyxy": det["bbox_xyxy"].tolist(),
                "score": det["score"],
                "is_right": det["is_right"],
                "cam_param_full": crop_pack["cam_param_full"].tolist(),
                "cam_param_patch": crop_pack["cam_param_patch"].tolist(),
                "focal_full": crop_pack["focal_full"],
                "box_center": crop_pack["box_center"].tolist(),
                "box_size": crop_pack["box_size"],
                "crop_bbox_xyxy": crop_pack["crop_bbox_xyxy"].tolist(),
                "pred_cam": pred["pred_cam"].tolist(),
                "pred_cam_t_full": pred["pred_cam_t_full"].tolist(),
                "pred_cam_t_weakcam": pred["pred_cam_t_weakcam"].tolist(),
                "pred_cam_t_solved": None if pred["pred_cam_t_solved"] is None else pred["pred_cam_t_solved"].tolist(),
                "overlay_mode": args.overlay_mode,
                "tvec_mode": args.tvec_mode,
                "pred_keypoints_2d_full": pred["pred_keypoints_2d_full"].tolist(),
                "pred_vertices_2d_full": pred["pred_vertices_2d_full"].tolist(),
                "overlay_path": str(overlay_path),
                "mesh_path": str(mesh_path),
                "crop_path": str(crop_path) if crop_path is not None else None,
            }
            meta_path = image_out_dir / f"{prefix}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            meta["meta_path"] = str(meta_path)
            image_result["detections"].append(meta)

        if args.overlay_mode == "image_space" and all_vertices_2d:
            overlay_all = render_overlay_multiple_image_space(
                full_image_bgr=image_bgr,
                vertices_2d_list=all_vertices_2d,
                vertices_depth_list=all_vertices_depth,
                faces=faces,
                color_hex=args.mesh_color,
            )
            for crop_bbox in all_crop_bboxes:
                overlay_all = draw_crop_bbox(overlay_all, crop_bbox)
            overlay_all = draw_detection_bboxes(overlay_all, detections)
            overlay_all_path = image_out_dir / "all_hands_overlay.png"
            cv2.imwrite(str(overlay_all_path), overlay_all)
            image_result["all_hands_overlay_path"] = str(overlay_all_path)
        elif all_vertices and image_cam_param_full is not None:
            overlay_all = render_overlay_multiple(
                full_image_bgr=image_bgr,
                vertices_list=all_vertices,
                faces=faces,
                cam_t_list=all_cam_t,
                cam_param=image_cam_param_full,
                color_hex=args.mesh_color,
                is_right_list=all_is_right,
            )
            for crop_bbox in all_crop_bboxes:
                overlay_all = draw_crop_bbox(overlay_all, crop_bbox)
            overlay_all = draw_detection_bboxes(overlay_all, detections)
            overlay_all_path = image_out_dir / "all_hands_overlay.png"
            cv2.imwrite(str(overlay_all_path), overlay_all)
            image_result["all_hands_overlay_path"] = str(overlay_all_path)

        summary["results"].append(image_result)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to: {out_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

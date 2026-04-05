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

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import torch
import trimesh

from data.utils import get_example
from visualization.demo_gpgformer_no_cam import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    affine_transform_points,
    bbox_from_points,
    collect_image_paths,
    detect_hands,
    draw_crop_bbox,
    draw_detection_bbox,
    draw_detection_bboxes,
    extract_mano_faces,
    load_gpgformer,
    load_yolo_detector,
    project_points_with_translation,
    refine_translation_to_bbox,
    render_mesh_only,
    render_overlay_multiple_image_space_shaded,
    solve_translation_from_2d_correspondences,
    weakcam_crop_to_full,
)

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_demo_tvec")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path("/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt")
DEFAULT_DETECTOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_MESH_COLOR = "#7677D8"
DEFAULT_VIS_SMOOTH_ITERS = 2
DEFAULT_VIS_SMOOTH_LAMBDA = 0.18
DEFAULT_VIS_SMOOTH_MU = -0.19
FINGERTIP_VERTEX_IDS = np.array([744, 320, 443, 555, 672], dtype=np.int64)
FINGERTIP_PREV_JOINT_IDS = np.array([3, 7, 11, 15, 19], dtype=np.int64)
FINGERTIP_JOINT_IDS = np.array([4, 8, 12, 16, 20], dtype=np.int64)
VIS_TIP_EXTENSION_M = 0.0045
VIS_SURFACE_INFLATE_M = 0.0015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust Hand Reconstruction via Default Camera PnP Optimization.")
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
    return parser.parse_args()


def prepare_crop_weakcam(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    cfg: dict[str, Any],
    bbox_expand: float,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    box_size = max(max(x2 - x1, 2.0), max(y2 - y1, 2.0)) * bbox_expand
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
        image_bgr, box_center[0], box_center[1], box_size, box_size,
        np.zeros((21,3), np.float32), np.zeros((21,3), np.float32),
        {"global_orient": np.zeros((3,), np.float32), "hand_pose": np.zeros((45,), np.float32), "betas": np.zeros((10,), np.float32)},
        {"global_orient": np.zeros((1,), np.float32), "hand_pose": np.zeros((1,), np.float32), "betas": np.zeros((1,), np.float32)},
        list(range(21)), img_width, img_size, None, None, do_augment=False, is_right=is_right,
        augm_config=dataset_cfg.get("wilor_aug_config", {}), is_bgr=True, return_trans=True,
    )

    crop_rgb01 = torch.from_numpy(img_patch).float() / 255.0
    crop_rgb = (crop_rgb01 - IMAGENET_MEAN) / IMAGENET_STD

    trans_inv = cv2.invertAffineTransform(np.asarray(trans, dtype=np.float32))
    cam_param_patch = np.array(
        [
            focal_full,
            focal_full,
            img_width / 2.0,
            img_size / 2.0,
        ],
        dtype=np.float32,
    )
    cam_param_full = np.array(
        [
            focal_full,
            focal_full,
            w / 2.0,
            h / 2.0,
        ],
        dtype=np.float32,
    )

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

def project_crop_points_to_full_image(
    points_3d: np.ndarray,
    pred_cam: np.ndarray,
    crop_pack: dict[str, Any],
    is_right: bool
) -> np.ndarray:
    """Map crop-space projections back to original-image pixels with the same semantics as weakcam_image_space."""
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    cam = np.asarray(pred_cam, dtype=np.float32).reshape(3)
    cam_patch = np.asarray(crop_pack["cam_param_patch"], dtype=np.float32).reshape(-1)
    patch_h = float(crop_pack["patch_h"])
    crop_cam_t = np.array(
        [
            float(cam[1]),
            float(cam[2]),
            2.0 * float(cam_patch[0]) / (patch_h * float(cam[0]) + 1e-9),
        ],
        dtype=np.float32,
    )

    patch_xy = project_points_with_translation(points, cam_patch, crop_cam_t)
    wild_xy = affine_transform_points(patch_xy, crop_pack["trans_inv"])
    if not is_right:
        wild_xy[:, 0] = float(crop_pack["img_w"]) - wild_xy[:, 0] - 1.0
    return wild_xy.astype(np.float32)

def restore_points_to_original_handedness(points_3d: np.ndarray, is_right: bool) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3).copy()
    if not is_right:
        points[:, 0] *= -1.0
    return points


def bbox_area_xyxy(bbox_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def should_refine_to_detector(det_bbox_xyxy: np.ndarray, weak_bbox_xyxy: np.ndarray | None) -> bool:
    if weak_bbox_xyxy is None:
        return False
    det_box = np.asarray(det_bbox_xyxy, dtype=np.float32).reshape(4)
    weak_box = np.asarray(weak_bbox_xyxy, dtype=np.float32).reshape(4)
    det_area = bbox_area_xyxy(det_box)
    weak_area = bbox_area_xyxy(weak_box)
    if det_area <= 1e-6 or weak_area <= 1e-6:
        return False

    area_ratio = det_area / weak_area
    det_w = max(float(det_box[2] - det_box[0]), 1e-6)
    det_h = max(float(det_box[3] - det_box[1]), 1e-6)
    weak_w = max(float(weak_box[2] - weak_box[0]), 1e-6)
    weak_h = max(float(weak_box[3] - weak_box[1]), 1e-6)
    aspect_ratio = (det_w / det_h) / (weak_w / weak_h)
    return 0.55 <= area_ratio <= 2.25 and 0.75 <= aspect_ratio <= 1.33


def smooth_visualization_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_iters: int = DEFAULT_VIS_SMOOTH_ITERS,
    lamb: float = DEFAULT_VIS_SMOOTH_LAMBDA,
    mu: float = DEFAULT_VIS_SMOOTH_MU,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3).copy()
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if verts.shape[0] == 0 or faces_arr.shape[0] == 0 or int(num_iters) <= 0:
        return verts

    neighbors = [set() for _ in range(verts.shape[0])]
    for a, b, c in faces_arr.tolist():
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    def _laplacian_step(cur_verts: np.ndarray, weight: float) -> np.ndarray:
        updated = cur_verts.copy()
        for vidx, nbrs in enumerate(neighbors):
            if not nbrs:
                continue
            nbr_idx = np.fromiter(nbrs, dtype=np.int64)
            nbr_mean = cur_verts[nbr_idx].mean(axis=0)
            updated[vidx] = cur_verts[vidx] + float(weight) * (nbr_mean - cur_verts[vidx])
        return updated

    for _ in range(int(num_iters)):
        verts = _laplacian_step(verts, lamb)
        verts = _laplacian_step(verts, mu)
    return verts.astype(np.float32)


def enhance_visualization_mesh(
    vertices: np.ndarray,
    keypoints_3d: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Round fingertip caps and slightly thicken the mesh for more natural image overlays.

    compare_hand_mesh.py does not contain a separate hand-shape fix on the same predictions; its
    apparent visual advantage mainly comes from cleaner dataset crops and camera metadata. For the
    in-the-wild demo we therefore improve the visualization mesh directly, without touching the
    alignment logic that already places the hand reasonably well.
    """
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3).copy()
    joints = np.asarray(keypoints_3d, dtype=np.float32).reshape(-1, 3)
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if verts.shape[0] == 0 or faces_arr.shape[0] == 0 or joints.shape[0] < 21:
        return verts, faces_arr

    neighbors = [set() for _ in range(verts.shape[0])]
    for a, b, c in faces_arr.tolist():
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
        vis_verts, vis_faces = trimesh.remesh.subdivide_loop(vis_verts, faces_arr, iterations=1)
        vis_verts = np.asarray(vis_verts, dtype=np.float32)
        vis_faces = np.asarray(vis_faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vis_verts, faces=vis_faces, process=False)
        vis_verts = vis_verts + np.asarray(mesh.vertex_normals, dtype=np.float32) * VIS_SURFACE_INFLATE_M
        return vis_verts.astype(np.float32), vis_faces.astype(np.int64)
    except Exception:
        return vis_verts.astype(np.float32), faces_arr.astype(np.int64)

def solve_tvec_and_project_mesh(
    v3d_crop: np.ndarray,
    k3d_crop: np.ndarray,
    pred_cam_crop: np.ndarray,
    direct_v2d: np.ndarray,
    wild_k2d: np.ndarray,
    det_bbox_xyxy: np.ndarray,
    crop_pack: dict[str, Any],
    is_right: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover only the full-image translation; model outputs are already in the camera-aligned MANO frame."""
    v3d = restore_points_to_original_handedness(v3d_crop, is_right)
    k3d = restore_points_to_original_handedness(k3d_crop, is_right)

    pred_cam = np.asarray(pred_cam_crop, dtype=np.float32).reshape(3).copy()
    if not is_right:
        pred_cam[1] *= -1.0

    cam_param_full = np.asarray(crop_pack["cam_param_full"], dtype=np.float32).reshape(-1)
    weakcam_tvec = weakcam_crop_to_full(
        pred_cam=pred_cam,
        box_center=crop_pack["box_center"],
        box_size=crop_pack["box_size"],
        img_size_wh=crop_pack["img_size_wh"],
        focal_length=float(crop_pack["focal_full"]),
    )
    solved_tvec = solve_translation_from_2d_correspondences(k3d, wild_k2d, cam_param_full)
    vertex_tvec = solve_translation_from_2d_correspondences(v3d, direct_v2d, cam_param_full)

    tvec = weakcam_tvec
    if is_right and vertex_tvec is not None and np.isfinite(vertex_tvec).all() and float(vertex_tvec[2]) > 1e-4:
        tvec = vertex_tvec.astype(np.float32)
    elif is_right and solved_tvec is not None and np.isfinite(solved_tvec).all() and float(solved_tvec[2]) > 1e-4:
        tvec = solved_tvec.astype(np.float32)

    weak_bbox = bbox_from_points(project_points_with_translation(v3d, cam_param_full, tvec))

    # Detector refinement helps right-hand localization, but only when the detector box is on the same scale
    # as the weakcam projection. Large container boxes (for example, hand + held object) would otherwise pull
    # the mesh far away from the true hand.
    if should_refine_to_detector(det_bbox_xyxy, weak_bbox):
        tvec = refine_translation_to_bbox(
            vertices_3d=v3d,
            cam_param=cam_param_full,
            init_cam_t=tvec,
            target_bbox_xyxy=det_bbox_xyxy,
            num_iters=4,
        )

    v2d_wild = project_points_with_translation(v3d, cam_param_full, tvec)
    depth_z = (v3d[:, 2] + tvec[2]).astype(np.float32)
    return v2d_wild.astype(np.float32), v3d.astype(np.float32), depth_z, tvec.astype(np.float32)


@torch.no_grad()
def infer_and_align_tvec(
    model: Any,
    crop_rgb: torch.Tensor,
    crop_pack: dict[str, Any],
    det_bbox_xyxy: np.ndarray,
    is_right: bool,
) -> dict[str, Any]:
    rgb = crop_rgb.unsqueeze(0).to(next(model.parameters()).device, dtype=torch.float32)

    cam_param = torch.from_numpy(np.asarray(crop_pack["cam_param_patch"], dtype=np.float32)).unsqueeze(0).to(
        rgb.device, dtype=torch.float32
    )
    out = model(rgb, cam_param=cam_param)
    v3d_crop = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
    k3d_crop = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
    pred_cam_crop = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)

    direct_v2d = project_crop_points_to_full_image(v3d_crop, pred_cam_crop, crop_pack, is_right)
    wild_k2d = project_crop_points_to_full_image(k3d_crop, pred_cam_crop, crop_pack, is_right)
    wild_v2d, rigid_v3d, depth_z, cam_t_full = solve_tvec_and_project_mesh(
        v3d_crop=v3d_crop,
        k3d_crop=k3d_crop,
        pred_cam_crop=pred_cam_crop,
        direct_v2d=direct_v2d,
        wild_k2d=wild_k2d,
        det_bbox_xyxy=det_bbox_xyxy,
        crop_pack=crop_pack,
        is_right=is_right,
    )

    return {
        "pred_vertices_2d_full": wild_v2d,
        "pred_vertices_depth": depth_z,
        "raw_vertices_for_mesh_render": rigid_v3d,
        "raw_keypoints_for_mesh_render": restore_points_to_original_handedness(k3d_crop, is_right),
        "pred_cam_t_full": cam_t_full,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, cfg = load_gpgformer(Path(args.config), Path(args.ckpt), device)
    detector = load_yolo_detector(Path(args.detector_ckpt))
    faces = extract_mano_faces(model)
    paths = collect_image_paths(Path(args.img_dir))

    for p in paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: continue
        
        detections = detect_hands(detector, img_bgr, args.detector_conf, args.detector_iou, "cpu", 8)
        img_out_dir = out_dir / p.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        v2d_list, v3d_list, depth_list = [], [], []
        vis_faces = faces
        cam_param_full = None
        
        for idx, det in enumerate(detections):
            is_right = bool(det["is_right"])
            crop_pack = prepare_crop_weakcam(img_bgr, det["bbox_xyxy"], is_right, cfg, args.bbox_expand)
            cam_param_full = crop_pack["cam_param_full"]

            pred = infer_and_align_tvec(
                model=model,
                crop_rgb=crop_pack["crop_rgb"],
                crop_pack=crop_pack,
                det_bbox_xyxy=det["bbox_xyxy"],
                is_right=is_right,
            )
            raw_vertices = np.asarray(pred["raw_vertices_for_mesh_render"], dtype=np.float32)
            raw_keypoints = np.asarray(pred["raw_keypoints_for_mesh_render"], dtype=np.float32)
            vis_vertices, vis_faces = enhance_visualization_mesh(raw_vertices, raw_keypoints, faces)
            vis_cam_t = np.asarray(pred["pred_cam_t_full"], dtype=np.float32)
            vis_v2d = project_points_with_translation(vis_vertices, cam_param_full, vis_cam_t)
            vis_depth = (vis_vertices[:, 2] + vis_cam_t[2]).astype(np.float32)

            v2d_list.append(vis_v2d.astype(np.float32))
            v3d_list.append(vis_vertices)
            depth_list.append(vis_depth)
            
            # Isolated rendering
            mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
            cv2.imwrite(str(img_out_dir / f"det_{idx:02d}_mesh.png"), mesh_only)

        if v2d_list and cam_param_full is not None:
            # Use image-space shaded compositing for the final preview. This preserves the
            # translation-aligned 2D silhouette while avoiding the opaque perspective render
            # that made curled fingers look artificially clipped and smaller than the real hand.
            res_img = render_overlay_multiple_image_space_shaded(
                img_bgr.copy(),
                v2d_list,
                v3d_list,
                depth_list,
                vis_faces,
                args.mesh_color,
            )
            res_img = draw_detection_bboxes(res_img, detections)
            cv2.imwrite(str(img_out_dir / "all_hands_overlay.png"), res_img)
            
    print(f"Alignment successfully resolved using translation-only recovery with image-space shaded overlay at: {out_dir}")

if __name__ == "__main__":
    main()

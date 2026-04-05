from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import torch

from data.utils import get_example
from visualization.demo_gpgformer_tvec import (
    enhance_visualization_mesh,
    infer_and_align_tvec,
    prepare_crop_weakcam,
)
from visualization.demo_gpgformer_no_cam import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    collect_image_paths,
    detect_hands,
    draw_detection_bbox,
    draw_detection_bboxes,
    extract_mano_faces,
    load_gpgformer,
    load_yolo_detector,
    project_points_with_translation,
    render_mesh_only,
    render_overlay_image_space_shaded,
)

# --------------------------
# Defaults
# --------------------------
DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_demo_defaultcam_pnp_mediapipe")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    "/root/code/vepfs/GPGFormer/checkpoints/"
    "freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_MESH_COLOR = "#7677D8"
DEFAULT_MEDIAPIPE_MODEL = Path("/root/code/vepfs/GPGFormer/weights/hand_landmarker.task")


# --------------------------
# Mediapipe (optional)
# --------------------------
def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except Exception:
        return None


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float).tolist()
    bx1, by1, bx2, by2 = b.astype(float).tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
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
    """
    Returns list of hands:
      - kpts_2d: (21,2) in pixel coordinates
      - handedness: 'Right'/'Left' (mediapipe's view)
      - score: handedness score
      - bbox_xyxy: from kpts min/max
    """
    mp = _try_import_mediapipe()
    if mp is None:
        return []

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    # Legacy MediaPipe Solutions API.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

        results = hands.process(img_rgb)
        hands.close()

        out: list[dict[str, Any]] = []
        if not results.multi_hand_landmarks:
            return out

        for idx, lm in enumerate(results.multi_hand_landmarks):
            pts = []
            for p in lm.landmark:
                pts.append([p.x * w, p.y * h])
            kpts = np.asarray(pts, dtype=np.float32)
            x1, y1 = float(np.min(kpts[:, 0])), float(np.min(kpts[:, 1]))
            x2, y2 = float(np.max(kpts[:, 0])), float(np.max(kpts[:, 1]))
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            handed = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                handed = results.multi_handedness[idx].classification[0].label
                score = float(results.multi_handedness[idx].classification[0].score)

            out.append(
                {
                    "kpts_2d": kpts,
                    "handedness": handed,
                    "score": score,
                    "bbox_xyxy": bbox,
                }
            )
        return out

    # New MediaPipe Tasks API. It requires an explicit .task model file.
    model_path_use = str(model_path or "")
    if not model_path_use or not Path(model_path_use).is_file():
        return []

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
        return []

    out: list[dict[str, Any]] = []
    hand_landmarks = getattr(results, "hand_landmarks", None)
    handedness_list = getattr(results, "handedness", None)
    if not hand_landmarks:
        return out

    for idx, lm in enumerate(hand_landmarks):
        pts = []
        for p in lm:
            pts.append([p.x * w, p.y * h])
        kpts = np.asarray(pts, dtype=np.float32)
        x1, y1 = float(np.min(kpts[:, 0])), float(np.min(kpts[:, 1]))
        x2, y2 = float(np.max(kpts[:, 0])), float(np.max(kpts[:, 1]))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        handed = "Unknown"
        score = 0.0
        if handedness_list and idx < len(handedness_list) and handedness_list[idx]:
            handed = str(handedness_list[idx][0].category_name)
            score = float(handedness_list[idx][0].score)

        out.append(
            {
                "kpts_2d": kpts,
                "handedness": handed,  # 'Right'/'Left'
                "score": score,
                "bbox_xyxy": bbox,
            }
        )
    return out


def match_mediapipe_to_yolo(
    mp_hands: list[dict[str, Any]],
    yolo_bbox_xyxy: np.ndarray,
    yolo_is_right: bool,
) -> Optional[dict[str, Any]]:
    """
    Match mediapipe hand instance to YOLO bbox by IoU and handedness.
    Note: mediapipe handedness is from the image view (usually corresponds to actual left/right reasonably well),
    but can be wrong. We treat it as a soft constraint.
    """
    if not mp_hands:
        return None

    target_label = "Right" if yolo_is_right else "Left"
    best = None
    best_score = -1e9
    for h in mp_hands:
        iou = _bbox_iou_xyxy(np.asarray(h["bbox_xyxy"]), np.asarray(yolo_bbox_xyxy))
        handed_bonus = 0.15 if h.get("handedness") == target_label else 0.0
        score = iou + handed_bonus
        if score > best_score:
            best_score = score
            best = h

    # require a minimum IoU to avoid mismatching
    if best is not None and _bbox_iou_xyxy(np.asarray(best["bbox_xyxy"]), np.asarray(yolo_bbox_xyxy)) < 0.05:
        return None
    return best


# --------------------------
# Default camera + robust PnP
# --------------------------
def build_default_K(img_w: int, img_h: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, img_w / 2.0],
         [0.0, focal, img_h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def solve_pnp_robust(
    k3d: np.ndarray,   # (N,3)
    k2d: np.ndarray,   # (N,2)
    K: np.ndarray,
    reproj_thresh_px: float = 8.0,
) -> tuple[bool, np.ndarray, np.ndarray, float]:
    """
    Returns:
      ok, rvec(3,1), tvec(3,1), reproj_rmse
    """
    dist = np.zeros((4, 1), dtype=np.float32)
    obj = k3d.astype(np.float32).reshape(-1, 1, 3)
    img = k2d.astype(np.float32).reshape(-1, 1, 2)

    # RANSAC for robustness (occlusion / bad kpts)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_SQPNP,
        reprojectionError=float(reproj_thresh_px),
        iterationsCount=200,
        confidence=0.999,
    )
    if not ok:
        return False, np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32), 1e9

    # refine with LM using inliers if present
    if inliers is not None and len(inliers) >= 6:
        inl = inliers.reshape(-1)
        obj_inl = obj[inl]
        img_inl = img[inl]
        rvec, tvec = cv2.solvePnPRefineLM(obj_inl, img_inl, K, dist, rvec, tvec)

    # compute rmse
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = proj - k2d.astype(np.float32)
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    return True, rvec.astype(np.float32), tvec.astype(np.float32), rmse


def ensure_points_in_front(
    v3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, z_min: float = 0.05
) -> np.ndarray:
    """
    If any vertex ends up behind/too close to camera (Z<=z_min), push tvec.z forward.
    This fixes the “finger looks clipped / missing triangles” artifact in many renderers.
    """
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    v_cam = (R @ v3d.T).T + tvec.reshape(1, 3)
    min_z = float(np.min(v_cam[:, 2]))
    if min_z < z_min:
        tvec = tvec.copy()
        tvec[2, 0] += (z_min - min_z)
    return tvec


def search_focal_and_pnp(
    k3d: np.ndarray,
    k2d: np.ndarray,
    img_w: int,
    img_h: int,
    focal_candidates: list[float],
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Search focal among candidates to minimize reprojection rmse.
    Returns ok, K, rvec, tvec, rmse
    """
    best = (False, None, None, None, 1e18)
    for f in focal_candidates:
        K = build_default_K(img_w, img_h, float(f))
        ok, rvec, tvec, rmse = solve_pnp_robust(k3d, k2d, K)
        if ok and rmse < best[4]:
            best = (ok, K, rvec, tvec, rmse)

    if not best[0]:
        return False, np.zeros((3, 3), np.float32), np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32), 1e18
    return best  # type: ignore


# --------------------------
# Model crop + inference
# --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DefaultCam+PnP+MediaPipe for no-intrinsics hand mesh overlay")
    p.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--detector-ckpt", type=str, default=str(DEFAULT_DETECTOR_CKPT))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--detector-conf", type=float, default=0.3)
    p.add_argument("--detector-iou", type=float, default=0.6)
    p.add_argument("--bbox-expand", type=float, default=1.25)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--max-dets-per-image", type=int, default=8)
    p.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR)

    # PnP tuning
    p.add_argument("--pnp-reproj-thresh", type=float, default=8.0)
    p.add_argument("--focal-search", action="store_true", help="Grid-search focal for better alignment.")
    p.add_argument("--focal-mult-min", type=float, default=0.8)
    p.add_argument("--focal-mult-max", type=float, default=2.2)
    p.add_argument("--focal-steps", type=int, default=10)

    # mediapipe
    p.add_argument("--no-mediapipe", action="store_true", help="Disable mediapipe and fallback to self-projected 2D.")
    p.add_argument("--mediapipe-model", type=str, default=str(DEFAULT_MEDIAPIPE_MODEL), help="Optional MediaPipe Tasks hand_landmarker .task file.")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def prepare_crop_for_model(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    cfg: dict[str, Any],
    bbox_expand: float,
) -> dict[str, Any]:
    """
    Only for feeding the model (crop/resize/normalize).
    PnP will be done in full image coordinates using MediaPipe 2D kpts.
    """
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    bw = max(x2 - x1, 2.0)
    bh = max(y2 - y1, 2.0)
    box_size = max(bw, bh) * float(bbox_expand)
    box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    image_size = int(model_cfg.get("image_size", dataset_cfg.get("img_size", 256)))
    image_width = int(model_cfg.get("image_width", int(round(image_size * 0.75))))

    img_patch, _, _, _, _, _, trans = get_example(
        image_bgr,
        float(box_center[0]),
        float(box_center[1]),
        float(box_size),
        float(box_size),
        np.zeros((21, 3), dtype=np.float32),
        np.zeros((21, 3), dtype=np.float32),
        {
            "global_orient": np.zeros((3,), dtype=np.float32),
            "hand_pose": np.zeros((45,), dtype=np.float32),
            "betas": np.zeros((10,), dtype=np.float32),
        },
        {"global_orient": np.zeros((1,), dtype=np.float32), "hand_pose": np.zeros((1,), dtype=np.float32), "betas": np.zeros((1,), dtype=np.float32)},
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

    focal_patch = float(model_cfg.get("focal_length", 5000.0))
    cam_param_patch = np.array([focal_patch, focal_patch, image_width / 2.0, image_size / 2.0], dtype=np.float32)

    return {"crop_rgb": crop_rgb, "cam_param_patch": cam_param_patch}


@torch.no_grad()
def infer_model(model: Any, crop_rgb: torch.Tensor, cam_param_patch: np.ndarray) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    rgb = crop_rgb.unsqueeze(0).to(device, dtype=torch.float32)
    cam_param = torch.from_numpy(cam_param_patch).unsqueeze(0).to(device, dtype=torch.float32)

    out = model(rgb, cam_param=cam_param)
    v3d = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)        # (778,3)
    k3d = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)    # (21,3)
    pred_cam = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)
    return {"v3d": v3d, "k3d": k3d, "pred_cam": pred_cam}


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

    for p in paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        mp_hands = []
        if not args.no_mediapipe:
            mp_hands = detect_mediapipe_hands_21(
                img_bgr,
                max_num_hands=4,
                model_path=args.mediapipe_model,
            )

        dets = detect_hands(
            detector=detector,
            image_bgr=img_bgr,
            conf=args.detector_conf,
            iou=args.detector_iou,
            device=detector_device,
            max_dets=args.max_dets_per_image,
        )

        img_out_dir = out_dir / p.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        overlay_all = img_bgr.copy()

        for det_idx, det in enumerate(dets):
            is_right = bool(det["is_right"])
            bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)

            # 1) get 2D keypoints from MediaPipe matched to this bbox
            mp_match = match_mediapipe_to_yolo(mp_hands, bbox, is_right)
            if mp_match is None:
                # Fall back to the translation-only full-image recovery path so the script still
                # produces usable overlays when MediaPipe is unavailable or the new Tasks API has
                # no model file configured in the current environment.
                print(f"[WARN] {p.name} det{det_idx:02d}: no mediapipe match, fallback to translation-only overlay.")
                crop_pack = prepare_crop_weakcam(img_bgr, bbox, is_right, cfg, args.bbox_expand)
                pred = infer_and_align_tvec(
                    model=model,
                    crop_rgb=crop_pack["crop_rgb"],
                    crop_pack=crop_pack,
                    det_bbox_xyxy=bbox,
                    is_right=is_right,
                )
                raw_vertices = np.asarray(pred["raw_vertices_for_mesh_render"], dtype=np.float32)
                raw_keypoints = np.asarray(pred["raw_keypoints_for_mesh_render"], dtype=np.float32)
                vis_vertices, vis_faces = enhance_visualization_mesh(raw_vertices, raw_keypoints, faces)
                vis_cam_t = np.asarray(pred["pred_cam_t_full"], dtype=np.float32)
                vis_v2d = project_points_with_translation(vis_vertices, crop_pack["cam_param_full"], vis_cam_t)
                vis_depth = (vis_vertices[:, 2] + vis_cam_t[2]).astype(np.float32)

                overlay_all = render_overlay_image_space_shaded(
                    full_image_bgr=overlay_all,
                    vertices_2d=vis_v2d,
                    vertices_3d=vis_vertices,
                    vertices_depth=vis_depth,
                    faces=vis_faces,
                    color_hex=args.mesh_color,
                )
                mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
                cv2.imwrite(str(img_out_dir / f"det_{det_idx:02d}_mesh.png"), mesh_only)
                overlay_all = draw_detection_bbox(overlay_all, bbox, is_right, det.get("score"))
                continue

            k2d = np.asarray(mp_match["kpts_2d"], dtype=np.float32)  # (21,2)

            # 2) run model to get 3D joints/mesh
            crop_pack = prepare_crop_for_model(img_bgr, bbox, is_right, cfg, args.bbox_expand)
            pred = infer_model(model, crop_pack["crop_rgb"], crop_pack["cam_param_patch"])
            v3d = pred["v3d"]
            k3d = pred["k3d"]

            # 3) left-hand: mirror 3D to match image hand
            if not is_right:
                v3d = v3d.copy()
                k3d = k3d.copy()
                v3d[:, 0] *= -1.0
                k3d[:, 0] *= -1.0

            # 4) Default camera + (optional) focal search + PnP
            if args.focal_search:
                f_min = args.focal_mult_min * max(W, H)
                f_max = args.focal_mult_max * max(W, H)
                focal_candidates = np.linspace(f_min, f_max, int(args.focal_steps)).tolist()
                ok, K, rvec, tvec, rmse = search_focal_and_pnp(k3d, k2d, W, H, focal_candidates)
            else:
                K = build_default_K(W, H, float(max(W, H) * 1.4))  # a stable default
                ok, rvec, tvec, rmse = solve_pnp_robust(k3d, k2d, K, reproj_thresh_px=float(args.pnp_reproj_thresh))

            if not ok:
                print(f"[WARN] {p.name} det{det_idx:02d}: solvePnP failed.")
                overlay_all = draw_detection_bbox(overlay_all, bbox, is_right, det.get("score"))
                continue

            # 5) Critical fix: avoid Z<=0 causing “clipped finger / holes”
            tvec = ensure_points_in_front(v3d, rvec, tvec, z_min=0.05)

            # The predicted 21-joint order already matches the FreiHAND / MANO fingertip layout used
            # during training, so the fixed index-finger artifact is not caused by a 2D/3D joint-order
            # mismatch. We only improve the visualization mesh after pose recovery.
            vis_vertices, vis_faces = enhance_visualization_mesh(v3d, k3d, faces)

            # 6) project vertices with this camera
            dist = np.zeros((4, 1), dtype=np.float32)
            v2d, _ = cv2.projectPoints(vis_vertices.astype(np.float32), rvec, tvec, K, dist)
            v2d = v2d.reshape(-1, 2).astype(np.float32)

            # 7) compute camera-space depth for stable rendering order
            R, _ = cv2.Rodrigues(rvec.astype(np.float64))
            v_cam = (R @ vis_vertices.T).T + tvec.reshape(1, 3)
            depth_z = v_cam[:, 2].astype(np.float32)
            depth_z = np.maximum(depth_z, 1e-4)

            # 8) render overlay
            overlay_all = render_overlay_image_space_shaded(
                full_image_bgr=overlay_all,
                vertices_2d=v2d,
                vertices_3d=vis_vertices,
                vertices_depth=depth_z,
                faces=vis_faces,
                color_hex=args.mesh_color,
            )
            overlay_all = draw_detection_bbox(overlay_all, bbox, is_right, det.get("score"))

            # 9) save mesh-only for debugging
            mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
            cv2.imwrite(str(img_out_dir / f"det_{det_idx:02d}_mesh.png"), mesh_only)

            meta = {
                "image": str(p),
                "det_idx": det_idx,
                "bbox_xyxy": bbox.tolist(),
                "is_right": is_right,
                "mp_handedness": mp_match.get("handedness"),
                "mp_score": float(mp_match.get("score", 0.0)),
                "pnp_rmse_px": float(rmse),
                "K": K.tolist(),
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": tvec.reshape(-1).tolist(),
            }
            (img_out_dir / f"det_{det_idx:02d}_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        overlay_path = img_out_dir / "all_hands_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_all)

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

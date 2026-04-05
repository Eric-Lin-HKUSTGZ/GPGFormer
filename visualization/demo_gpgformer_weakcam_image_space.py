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

from data.utils import get_example
from visualization.demo_gpgformer_no_cam import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    affine_transform_points,
    collect_image_paths,
    detect_hands,
    draw_crop_bbox,
    draw_detection_bbox,
    draw_detection_bboxes,
    extract_mano_faces,
    load_gpgformer,
    load_yolo_detector,
    project_points_with_translation,
    render_mesh_only,
    render_overlay_image_space_shaded,
    render_overlay_multiple_image_space_shaded,
    square_bbox_from_center_size,
)

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_demo_weakcam_image_space")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    "/root/code/vepfs/GPGFormer/checkpoints/"
    "freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/vepfs/GPGFormer/weights/detector.pt")
DEFAULT_MESH_COLOR = "#7677D8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strictly mathematically aligned in-the-wild hand reconstruction.")
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR), help="Input image directory.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG), help="GPGFormer YAML config.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="GPGFormer checkpoint.")
    parser.add_argument("--detector-ckpt", type=str, default=str(DEFAULT_DETECTOR_CKPT), help="YOLO hand detector.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--detector-conf", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--detector-iou", type=float, default=0.6, help="YOLO NMS IoU threshold.")
    parser.add_argument("--bbox-expand", type=float, default=1.4, help="Expand bbox by this factor.")
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all.")
    parser.add_argument("--max-dets-per-image", type=int, default=8, help="Max hands per image.")
    parser.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR, help="Hex color for meshes.")
    parser.add_argument("--save-crop", action="store_true", help="Save the backbone input patch.")
    parser.add_argument("--show-crop-box", action="store_true", help="Draw expanded bounding box.")
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def prepare_crop_weakcam(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    is_right: bool,
    cfg: dict[str, Any],
    bbox_expand: float,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    bw = max(x2 - x1, 2.0)
    bh = max(y2 - y1, 2.0)
    box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    box_size = max(bw, bh) * float(bbox_expand)

    half = box_size * 0.5
    if box_size <= float(w):
        box_center[0] = float(np.clip(box_center[0], half, float(w) - half))
    if box_size <= float(h):
        box_center[1] = float(np.clip(box_center[1], half, float(h) - half))

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
        {
            "global_orient": np.zeros((1,), dtype=np.float32),
            "hand_pose": np.zeros((1,), dtype=np.float32),
            "betas": np.zeros((1,), dtype=np.float32),
        },
        list(range(21)),
        image_width,
        image_size,
        None, None,
        do_augment=False,
        is_right=is_right,
        augm_config=dataset_cfg.get("wilor_aug_config", {}),
        is_bgr=True,
        return_trans=True,
    )

    crop_rgb01 = torch.from_numpy(img_patch).float() / 255.0
    crop_rgb = (crop_rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    cam_param_patch = np.array(
        [
            float(model_cfg.get("focal_length", 5000.0)),
            float(model_cfg.get("focal_length", 5000.0)),
            image_width / 2.0,
            image_size / 2.0,
        ],
        dtype=np.float32,
    )

    return {
        "crop_rgb": crop_rgb,
        "crop_rgb01": crop_rgb01,
        "box_center": box_center,
        "box_size": float(box_size),
        "crop_bbox_xyxy": square_bbox_from_center_size(box_center, box_size),
        "img_size_wh": np.array([w, h], dtype=np.float32),
        "cam_param_patch": cam_param_patch,
        "crop_to_full_trans": cv2.invertAffineTransform(np.asarray(trans, dtype=np.float32)),
        "patch_size_wh": np.array([image_width, image_size], dtype=np.float32),
    }


def weakcam_project_points_to_image(
    points_3d: np.ndarray,
    pred_cam: np.ndarray,
    cam_param_patch: np.ndarray,
    crop_to_full_trans: np.ndarray,
    img_size_wh: np.ndarray,
    is_right: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match the exact patch-space projection semantics used by GPGFormer training:
    pred_cam=[s, tx, ty] is converted to the crop-space perspective translation,
    projected onto patch pixels, and then mapped back through the inverse affine.

    For left hands, get_example() first flips the full image before cropping, so the
    horizontal undo must happen after inverse-affine mapping back to the full image.
    """
    points_canonical = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3).copy()
    cam = np.asarray(pred_cam, dtype=np.float32).reshape(3).copy()
    cam_patch = np.asarray(cam_param_patch, dtype=np.float32).reshape(-1)
    patch_h = float(cam_patch[3] * 2.0)
    crop_cam_t = np.array(
        [
            float(cam[1]),
            float(cam[2]),
            2.0 * float(cam_patch[0]) / (patch_h * float(cam[0]) + 1e-9),
        ],
        dtype=np.float32,
    )

    patch_xy = project_points_with_translation(points_canonical, cam_patch, crop_cam_t)
    image_xy = affine_transform_points(patch_xy, crop_to_full_trans)
    if not is_right:
        image_xy[:, 0] = float(img_size_wh[0]) - image_xy[:, 0] - 1.0

    points_original = points_canonical.copy()
    if not is_right:
        points_original[:, 0] *= -1.0

    return points_original, cam, crop_cam_t, image_xy.astype(np.float32)


@torch.no_grad()
def infer_one_hand_weakcam(
    model: Any,
    crop_rgb: torch.Tensor,
    cam_param_patch: np.ndarray,
    crop_to_full_trans: np.ndarray,
    img_size_wh: np.ndarray,
    is_right: bool,
) -> dict[str, Any]:
    rgb = crop_rgb.unsqueeze(0).to(next(model.parameters()).device, dtype=torch.float32)
    cam_param = torch.from_numpy(np.asarray(cam_param_patch, dtype=np.float32)).unsqueeze(0).to(rgb.device)
    out = model(rgb, cam_param=cam_param)

    pred_vertices_crop = out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32)
    pred_keypoints_3d_crop = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32)
    pred_cam_crop = out["pred_cam"][0].detach().cpu().numpy().astype(np.float32)

    pred_vertices, pred_cam, crop_cam_t, pred_vertices_2d_full = weakcam_project_points_to_image(
        points_3d=pred_vertices_crop,
        pred_cam=pred_cam_crop,
        cam_param_patch=cam_param_patch,
        crop_to_full_trans=crop_to_full_trans,
        img_size_wh=img_size_wh,
        is_right=is_right
    )
    pred_keypoints_3d, _, _, pred_keypoints_2d_full = weakcam_project_points_to_image(
        points_3d=pred_keypoints_3d_crop,
        pred_cam=pred_cam_crop,
        cam_param_patch=cam_param_patch,
        crop_to_full_trans=crop_to_full_trans,
        img_size_wh=img_size_wh,
        is_right=is_right
    )

    return {
        "pred_vertices": pred_vertices,
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_cam": pred_cam,
        "pred_cam_t_patch": crop_cam_t,
        "pred_vertices_2d_full": pred_vertices_2d_full,
        "pred_keypoints_2d_full": pred_keypoints_2d_full,
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
    device = _resolve_device(args.device)
    detector_device = "cpu" if device.type == "cpu" else str(device)

    model, cfg = load_gpgformer(cfg_path, ckpt_path, device)
    detector = load_yolo_detector(detector_ckpt)
    faces = extract_mano_faces(model)

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
        "pipeline": "weakcam_image_space_alignment",
        "bbox_expand": float(args.bbox_expand),
        "num_images": len(image_paths),
        "results": [],
    }

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            summary["results"].append({"image": str(image_path), "status": "failed_to_read"})
            continue

        detections = detect_hands(
            detector=detector, image_bgr=image_bgr, conf=args.detector_conf, 
            iou=args.detector_iou, device=detector_device, max_dets=args.max_dets_per_image,
        )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        image_result: dict[str, Any] = {
            "image": str(image_path),
            "status": "ok" if detections else "no_detection",
            "pipeline": "weakcam_image_space_alignment",
            "detections": [],
        }

        all_vertices_2d, all_vertices_3d, all_vertices_depth, all_crop_bboxes = [], [], [], []

        for det_idx, det in enumerate(detections):
            is_right = bool(det["is_right"])
            crop_pack = prepare_crop_weakcam(
                image_bgr=image_bgr, bbox_xyxy=det["bbox_xyxy"],
                is_right=is_right, cfg=cfg, bbox_expand=args.bbox_expand,
            )
            
            pred = infer_one_hand_weakcam(
                model=model,
                crop_rgb=crop_pack["crop_rgb"],
                cam_param_patch=crop_pack["cam_param_patch"],
                crop_to_full_trans=crop_pack["crop_to_full_trans"],
                img_size_wh=crop_pack["img_size_wh"],
                is_right=is_right,
            )

            all_vertices_2d.append(pred["pred_vertices_2d_full"])
            all_vertices_3d.append(pred["pred_vertices"])
            all_vertices_depth.append(pred["pred_vertices_depth"])
            all_crop_bboxes.append(crop_pack["crop_bbox_xyxy"])
            overlay = render_overlay_image_space_shaded(
                full_image_bgr=image_bgr,
                vertices_2d=pred["pred_vertices_2d_full"],
                vertices_3d=pred["pred_vertices"],
                vertices_depth=pred["pred_vertices_depth"],
                faces=faces,
                color_hex=args.mesh_color,
            )
            if args.show_crop_box:
                overlay = draw_crop_bbox(overlay, crop_pack["crop_bbox_xyxy"])
            overlay = draw_detection_bbox(overlay, det["bbox_xyxy"], is_right, det.get("score"))

            mesh_only = render_mesh_only(
                vertices=pred["pred_vertices"],
                faces=faces,
                color_hex=args.mesh_color,
                is_right=is_right, render_size=512,
            )

            prefix = f"det_{det_idx:02d}"
            overlay_path = image_out_dir / f"{prefix}_overlay.png"
            mesh_path = image_out_dir / f"{prefix}_mesh.png"
            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(mesh_path), mesh_only)

            crop_path = None
            if args.save_crop:
                crop_vis = (
                    crop_pack["crop_rgb01"].permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0
                ).clip(0, 255).astype(np.uint8)
                crop_path = image_out_dir / f"{prefix}_crop.png"
                cv2.imwrite(str(crop_path), crop_vis)

            meta = {
                "bbox_xyxy": np.asarray(det["bbox_xyxy"], dtype=np.float32).tolist(),
                "score": float(det["score"]),
                "is_right": bool(det["is_right"]),
                "crop_bbox_xyxy": crop_pack["crop_bbox_xyxy"].tolist(),
                "c_x": float(crop_pack["box_center"][0]),
                "c_y": float(crop_pack["box_center"][1]),
                "b_scale": float(crop_pack["box_size"]),
                "crop_center_xy": crop_pack["box_center"].tolist(),
                "crop_square_scale": float(crop_pack["box_size"]),
                "pred_cam_semantics": (
                    "pred_cam=[s,tx,ty] is converted to patch-space perspective translation "
                    "before projection, consistent with GPGFormer training"
                ),
                "pred_cam": pred["pred_cam"].tolist(),
                "pred_cam_t_patch": pred["pred_cam_t_patch"].tolist(),
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

        if all_vertices_2d:
            overlay_all = render_overlay_multiple_image_space_shaded(
                full_image_bgr=image_bgr,
                vertices_2d_list=all_vertices_2d,
                vertices_3d_list=all_vertices_3d,
                vertices_depth_list=all_vertices_depth,
                faces=faces,
                color_hex=args.mesh_color,
            )
            if args.show_crop_box:
                for b in all_crop_bboxes:
                    overlay_all = draw_crop_bbox(overlay_all, b)
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

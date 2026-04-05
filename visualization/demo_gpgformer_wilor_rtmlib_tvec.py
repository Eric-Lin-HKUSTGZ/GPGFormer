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

from rtmlib import RTMPose

from visualization.gpgformer_wilor_rtmlib_vis_utils import (
    REPROJ_COLOR,
    REPROJ_LINK_COLOR,
    TARGET_COLOR,
    TARGET_LINK_COLOR,
    adjust_tvec_for_root_relative,
    bbox_from_points,
    collect_image_paths,
    compute_reproj_rmse,
    convert_to_root_relative,
    detect_wilor_hands,
    draw_detection_bbox,
    draw_hand_joints,
    draw_joint_legend,
    enhance_visualization_mesh,
    ensure_points_in_front,
    extract_mano_faces,
    infer_model,
    load_gpgformer,
    load_wilor_detector,
    prepare_crop_weakcam,
    project_points_with_translation,
    refine_translation_to_bbox,
    render_mesh_only,
    render_overlay_multiple_paper_style,
    render_overlay_paper_style,
    restore_points_to_original_handedness,
    solve_tvec_handos_style,
    weakcam_crop_to_full,
)

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
# DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/WiLoR/demo_img")
# DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/hamer/example_data")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_dexycb_wilor_rtmlib_tvec")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    # "/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt"
    # "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/mixed_ho3d_20260320/ho3d/gpgformer_best.pt"
    "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/hand_reconstruction/WiLoR/pretrained_models/detector.pt")
DEFAULT_POSE_MODEL = Path(
    "/root/code/hand_reconstruction/rtmlib/rtmpose_onnx/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx"
)
DEFAULT_MESH_COLOR = "#7677D8"
DEFAULT_ROOT_INDEX = 9
DEFAULT_POSE_INPUT_SIZE = (256, 256)

RTM_SEMANTIC_WEIGHTS = np.array(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPGFormer hand mesh reconstruction using WiLoR detection and RTMPose 2D joints under unknown camera parameters."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--det-model", type=str, default=str(DEFAULT_DETECTOR_CKPT), help="WiLoR detector.pt path.")
    parser.add_argument("--pose-model", type=str, default=str(DEFAULT_POSE_MODEL), help="RTMPose hand ONNX path.")
    parser.add_argument("--device", type=str, default="cuda", help="GPGFormer device.")
    parser.add_argument("--pose-device", type=str, default="auto", help="RTMPose device, e.g. auto/cpu/cuda.")
    parser.add_argument("--detector-device", type=str, default="auto", help="WiLoR detector device, e.g. auto/cpu/0.")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=("onnxruntime", "opencv", "openvino"))
    parser.add_argument("--detector-conf", type=float, default=0.3)
    parser.add_argument("--detector-iou", type=float, default=0.6)
    parser.add_argument("--bbox-expand", type=float, default=1.25)
    parser.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-dets-per-image", type=int, default=0, help="Maximum detections per image; 0 keeps all detections.")
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--root-index", type=int, default=DEFAULT_ROOT_INDEX)
    parser.add_argument("--save-joint-debug", action="store_true")
    parser.add_argument("--save-per-det", action="store_true")
    return parser.parse_args()


def _resolve_torch_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_aux_device(device_str: str, main_device: torch.device) -> str:
    if device_str != "auto":
        return device_str
    if main_device.type == "cuda":
        return "cuda"
    return "cpu"


def build_rtm_joint_weights(scores: np.ndarray) -> np.ndarray:
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    conf = np.clip(conf, 0.05, 1.5) / 1.5
    return (RTM_SEMANTIC_WEIGHTS * (0.35 + 0.65 * conf)).astype(np.float32)


def refine_render_translation_to_bbox(
    render_vertices_3d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
    det_bbox_xyxy: np.ndarray,
) -> np.ndarray:
    refined_t = refine_translation_to_bbox(
        vertices_3d=render_vertices_3d,
        cam_param=cam_param,
        init_cam_t=cam_t,
        target_bbox_xyxy=det_bbox_xyxy,
        num_iters=2,
    )
    refined_t = np.asarray(refined_t, dtype=np.float32).reshape(3)

    proj_bbox = bbox_from_points(project_points_with_translation(render_vertices_3d, cam_param, refined_t))
    if proj_bbox is None:
        return refined_t.astype(np.float32)

    target = np.asarray(det_bbox_xyxy, dtype=np.float32).reshape(4)
    proj = np.asarray(proj_bbox, dtype=np.float32).reshape(4)
    target_w = max(float(target[2] - target[0]), 1e-6)
    target_h = max(float(target[3] - target[1]), 1e-6)
    proj_w = max(float(proj[2] - proj[0]), 1e-6)
    proj_h = max(float(proj[3] - proj[1]), 1e-6)

    contain_scale = max(proj_w / target_w, proj_h / target_h, 1.0)
    if contain_scale > 1.0:
        refined_t[2] *= float(contain_scale)
        proj_bbox = bbox_from_points(project_points_with_translation(render_vertices_3d, cam_param, refined_t))
        if proj_bbox is None:
            return refined_t.astype(np.float32)
        proj = np.asarray(proj_bbox, dtype=np.float32).reshape(4)

    proj_cx = float(proj[0] + proj[2]) * 0.5
    proj_cy = float(proj[1] + proj[3]) * 0.5
    target_cx = float(target[0] + target[2]) * 0.5
    target_cy = float(target[1] + target[3]) * 0.5
    fx, fy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:2]]
    refined_t[0] += (target_cx - proj_cx) * refined_t[2] / max(fx, 1e-6)
    refined_t[1] += (target_cy - proj_cy) * refined_t[2] / max(fy, 1e-6)
    return refined_t.astype(np.float32)


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_torch_device(args.device)
    detector_device = _resolve_aux_device(args.detector_device, device)
    pose_device = _resolve_aux_device(args.pose_device, device)

    model, cfg = load_gpgformer(Path(args.config), Path(args.ckpt), device)
    detector = load_wilor_detector(Path(args.det_model))
    pose_model = RTMPose(
        str(Path(args.pose_model)),
        model_input_size=DEFAULT_POSE_INPUT_SIZE,
        backend=args.backend,
        device=pose_device,
    )
    faces = extract_mano_faces(model)

    print(f"Loaded GPGFormer: {args.ckpt}")
    print(f"Loaded WiLoR detector: {args.det_model}")
    print(f"Loaded RTMPose model: {args.pose_model}")

    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]

    all_summary: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        detections = detect_wilor_hands(
            detector=detector,
            image_bgr=image_bgr,
            conf=float(args.detector_conf),
            iou=float(args.detector_iou),
            device=str(detector_device),
            max_dets=int(args.max_dets_per_image),
        )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out_dir / "input.png"), image_bgr)

        if not detections:
            cv2.imwrite(str(image_out_dir / "overlay.png"), image_bgr)
            summary = {"image": str(image_path), "num_hands": 0, "hands": []}
            (image_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            all_summary.append(summary)
            print(f"[WiLoR+RTM+Tvec] {image_path.name}: no detections")
            continue

        bboxes = np.stack([np.asarray(det["bbox_xyxy"], dtype=np.float32) for det in detections], axis=0)
        keypoints, scores = pose_model(image_bgr, bboxes=bboxes)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(len(detections), 21, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(len(detections), 21)

        vis_vertices_list: list[np.ndarray] = []
        vis_faces_list: list[np.ndarray] = []
        cam_t_list: list[np.ndarray] = []
        is_right_list: list[bool] = []
        hands_summary: list[dict[str, Any]] = []
        overlay_cam_param: np.ndarray | None = None

        for det_idx, det in enumerate(detections):
            det_bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            is_right = bool(det["is_right"])
            target_k2d = np.asarray(keypoints[det_idx], dtype=np.float32)
            target_scores = np.asarray(scores[det_idx], dtype=np.float32)
            pose_mean_score = float(np.mean(target_scores))

            crop_pack = prepare_crop_weakcam(image_bgr, det_bbox, is_right, cfg, float(args.bbox_expand))
            full_cam = np.asarray(crop_pack["cam_param_full"], dtype=np.float32)
            overlay_cam_param = full_cam
            infer_out = infer_model(model, crop_pack)

            v3d_raw = restore_points_to_original_handedness(infer_out["v3d_crop"], is_right)
            k3d_raw = restore_points_to_original_handedness(infer_out["k3d_crop"], is_right)
            v3d, k3d, root_xyz = convert_to_root_relative(v3d_raw, k3d_raw, int(args.root_index))

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

            joint_weights = build_rtm_joint_weights(target_scores)
            ok, cam_t, reproj_rmse_weighted, inliers = solve_tvec_handos_style(
                points_3d=k3d,
                points_2d=target_k2d,
                cam_param=full_cam,
                init_tvec=weakcam_init_t,
                joint_weights=joint_weights,
            )
            if not ok or not np.isfinite(cam_t).all():
                print(f"[WARN] {image_path.name} det{det_idx:02d}: HandOS-style tvec solve failed")
                continue

            cam_t = ensure_points_in_front(v3d, cam_t, z_min=float(args.z_min))
            vis_vertices, vis_faces = enhance_visualization_mesh(v3d, k3d, faces, is_right=is_right)
            cam_t = refine_render_translation_to_bbox(
                render_vertices_3d=vis_vertices,
                cam_param=full_cam,
                cam_t=cam_t,
                det_bbox_xyxy=det_bbox,
            )
            cam_t = ensure_points_in_front(v3d, cam_t, z_min=float(args.z_min))
            reproj_k2d = project_points_with_translation(k3d, full_cam, cam_t)
            reproj_rmse = compute_reproj_rmse(k3d, target_k2d, full_cam, cam_t, joint_weights=None)

            if args.save_per_det:
                det_overlay = render_overlay_paper_style(
                    full_image_bgr=image_bgr.copy(),
                    vertices=vis_vertices,
                    faces=vis_faces,
                    cam_t=cam_t,
                    cam_param=full_cam,
                    color_hex=args.mesh_color,
                    is_right=is_right,
                )
                cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_overlay.png"), det_overlay)

                if args.save_joint_debug:
                    joint_debug = image_bgr.copy()
                    joint_debug = draw_hand_joints(joint_debug, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
                    joint_debug = draw_hand_joints(joint_debug, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
                    joint_debug = draw_detection_bbox(joint_debug, det_bbox, is_right, det.get("score"))
                    joint_debug = draw_joint_legend(joint_debug, "rtm")
                    cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_joints.png"), joint_debug)

                mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
                cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_mesh.png"), mesh_only)

            vis_vertices_list.append(vis_vertices)
            vis_faces_list.append(vis_faces)
            cam_t_list.append(np.asarray(cam_t, dtype=np.float32))
            is_right_list.append(is_right)
            hands_summary.append(
                {
                    "hand_index": det_idx,
                    "handedness": str(det.get("handedness", "right" if is_right else "left")),
                    "is_right": is_right,
                    "detector_score": float(det.get("score", 0.0)),
                    "bbox_xyxy": det_bbox.astype(np.float32).tolist(),
                    "pose_mean_score": pose_mean_score,
                    "kpt_scores": target_scores.astype(np.float32).tolist(),
                    "target_kpts_2d": target_k2d.astype(np.float32).tolist(),
                    "reproj_kpts_2d": reproj_k2d.astype(np.float32).tolist(),
                    "solver": "handos_tvec",
                    "weight_profile": "semantic",
                    "num_inliers": int(np.sum(np.asarray(inliers, dtype=bool))),
                    "reproj_rmse": float(reproj_rmse),
                    "reproj_rmse_weighted": float(reproj_rmse_weighted),
                    "cam_param": full_cam.astype(np.float32).tolist(),
                    "cam_t": cam_t.astype(np.float32).tolist(),
                }
            )

        overlay = image_bgr.copy()
        if vis_vertices_list and overlay_cam_param is not None:
            overlay = render_overlay_multiple_paper_style(
                full_image_bgr=overlay,
                vertices_list=vis_vertices_list,
                faces_list=vis_faces_list,
                cam_t_list=cam_t_list,
                cam_param=overlay_cam_param,
                color_hex=args.mesh_color,
                is_right_list=is_right_list,
            )

        cv2.imwrite(str(image_out_dir / "overlay.png"), overlay)

        summary = {
            "image": str(image_path),
            "num_hands": len(hands_summary),
            "hands": hands_summary,
        }
        (image_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        all_summary.append(summary)
        print(f"[WiLoR+RTM+Tvec] {image_path.name}: reconstructed {len(hands_summary)} hand(s)")

    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "img_dir": str(img_dir),
                "config": str(args.config),
                "ckpt": str(args.ckpt),
                "det_model": str(args.det_model),
                "pose_model": str(args.pose_model),
                "num_images": len(all_summary),
                "results": all_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved WiLoR + RTM + GPGFormer tvec visualization to: {out_dir}")


if __name__ == "__main__":
    main()

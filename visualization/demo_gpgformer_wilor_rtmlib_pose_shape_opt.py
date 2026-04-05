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
import torch.nn.functional as F

from rtmlib import RTMPose
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat

from visualization.demo_gpgformer_handos_tvec import (
    REPROJ_COLOR,
    REPROJ_LINK_COLOR,
    TARGET_COLOR,
    TARGET_LINK_COLOR,
    compute_reproj_rmse,
    draw_hand_joints,
    draw_joint_legend,
    ensure_points_in_front,
    solve_tvec_handos_style,
)
from visualization.demo_gpgformer_no_cam import (
    bbox_from_points,
    collect_image_paths,
    draw_detection_bbox,
    extract_mano_faces,
    load_gpgformer,
    project_points_with_translation,
    render_mesh_only,
    render_overlay_image_space_shaded,
    render_overlay_multiple_image_space_shaded,
    weakcam_crop_to_full,
)
from visualization.demo_gpgformer_tvec import (
    enhance_visualization_mesh,
    prepare_crop_weakcam,
)
from visualization.demo_wilor_rtmlib_hand_2d import detect_wilor_hands, load_wilor_detector

DEFAULT_IMG_DIR = Path("/root/code/hand_reconstruction/GPGFormer/in-the-wild")
DEFAULT_OUT_DIR = Path("/root/code/vepfs/GPGFormer/outputs/gpgformer_wilor_rtmlib_pose_shape_opt")
DEFAULT_CFG = REPO_ROOT / "configs" / "config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml"
DEFAULT_CKPT = Path(
    "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/mixed_ho3d_20260320/ho3d/gpgformer_best.pt"
)
DEFAULT_DETECTOR_CKPT = Path("/root/code/hand_reconstruction/WiLoR/pretrained_models/detector.pt")
DEFAULT_POSE_MODEL = Path(
    "/root/code/hand_reconstruction/rtmlib/rtmpose_onnx/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx"
)
DEFAULT_MESH_COLOR = "#7677D8"
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
        description="Optimize GPGFormer MANO pose/shape+tvec using WiLoR detection and RTMPose 21-keypoint supervision."
    )
    parser.add_argument("--img-dir", type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--det-model", type=str, default=str(DEFAULT_DETECTOR_CKPT))
    parser.add_argument("--pose-model", type=str, default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pose-device", type=str, default="auto")
    parser.add_argument("--detector-device", type=str, default="auto")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=("onnxruntime", "opencv", "openvino"))
    parser.add_argument("--detector-conf", type=float, default=0.3)
    parser.add_argument("--detector-iou", type=float, default=0.6)
    parser.add_argument("--bbox-expand", type=float, default=1.25)
    parser.add_argument("--mesh-color", type=str, default=DEFAULT_MESH_COLOR)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-dets-per-image", type=int, default=8)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--opt-iters", type=int, default=120)
    parser.add_argument("--opt-lr", type=float, default=0.02)
    parser.add_argument("--pose-reg", type=float, default=0.002)
    parser.add_argument("--shape-reg", type=float, default=0.0005)
    parser.add_argument("--tvec-reg", type=float, default=0.01)
    parser.add_argument("--refine-to-bbox", action="store_true")
    parser.add_argument("--save-per-det", action="store_true")
    return parser.parse_args()


def _resolve_torch_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_aux_device(device_str: str, main_device: torch.device) -> str:
    if device_str != "auto":
        return device_str
    return "cuda" if main_device.type == "cuda" else "cpu"


def _rotmat_to_aa(rotmat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    trace = rotmat[..., 0, 0] + rotmat[..., 1, 1] + rotmat[..., 2, 2]
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)
    sin = torch.sin(angle)

    rx = rotmat[..., 2, 1] - rotmat[..., 1, 2]
    ry = rotmat[..., 0, 2] - rotmat[..., 2, 0]
    rz = rotmat[..., 1, 0] - rotmat[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)
    axis = r / (2.0 * sin).unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)
    aa_small = 0.5 * r
    small = (sin.abs() < 1e-4).unsqueeze(-1)
    return torch.where(small, aa_small, aa)


def _put_text_block(image_bgr: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    x = 14
    y = 18
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x - 2, y - th - 4), (x + tw + 6, y + baseline + 2), color, -1)
        cv2.putText(out, line, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, lineType=cv2.LINE_AA)
        y += th + baseline + 10
    return out


def build_rtm_joint_weights(scores: np.ndarray) -> np.ndarray:
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    conf = np.clip(conf, 0.05, 1.5) / 1.5
    weights = RTM_SEMANTIC_WEIGHTS * (0.35 + 0.65 * conf)
    weights[9] *= 1.15
    return weights.astype(np.float32)


@torch.no_grad()
def infer_model_full(model: Any, crop_pack: dict[str, Any]) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    crop_rgb = crop_pack["crop_rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
    cam_param_patch = torch.from_numpy(np.asarray(crop_pack["cam_param_patch"], dtype=np.float32)).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    out = model(crop_rgb, cam_param=cam_param_patch)
    pred_mano = out["pred_mano_params"]
    return {
        "pred_cam_crop": out["pred_cam"][0].detach().cpu().numpy().astype(np.float32),
        "pred_vertices": out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32),
        "pred_keypoints_3d": out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float32),
        "global_orient_rm": pred_mano["global_orient"][0].detach().cpu().numpy().astype(np.float32),
        "hand_pose_rm": pred_mano["hand_pose"][0].detach().cpu().numpy().astype(np.float32),
        "betas": pred_mano["betas"][0].detach().cpu().numpy().astype(np.float32),
    }


def _restore_points_to_original_handedness(points_3d: np.ndarray, is_right: bool) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3).copy()
    if not is_right:
        points[:, 0] *= -1.0
    return points


def _project_points_torch(points_3d: torch.Tensor, cam_param: torch.Tensor, cam_t: torch.Tensor) -> torch.Tensor:
    pts = points_3d + cam_t.unsqueeze(0)
    z = pts[:, 2:3]
    z = torch.where(torch.abs(z) < 1e-6, torch.full_like(z, 1e-6), z)
    u = cam_param[0] * (pts[:, 0:1] / z) + cam_param[2]
    v = cam_param[1] * (pts[:, 1:2] / z) + cam_param[3]
    return torch.cat([u, v], dim=1)


def _ensure_tensor_j_regressor(model: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    j_reg = getattr(model.mano.mano, "J_regressor", None)
    if j_reg is None:
        raise AttributeError("MANO layer is missing J_regressor.")
    if not isinstance(j_reg, torch.Tensor):
        j_reg = torch.as_tensor(j_reg, device=device, dtype=dtype)
    return j_reg.to(device=device, dtype=dtype)


def _decode_mano_right_hand(
    model: Any,
    global_orient_aa: torch.Tensor,
    hand_pose_aa: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_rm = aa_to_rotmat(global_orient_aa.view(-1, 3)).view(1, 1, 3, 3)
    hand_rm = aa_to_rotmat(hand_pose_aa.view(-1, 3)).view(1, 15, 3, 3)
    mano_params = {
        "global_orient": global_rm,
        "hand_pose": hand_rm,
        "betas": betas.view(1, 10),
    }
    mano_out = model.mano(mano_params, pose2rot=False)
    j_reg = _ensure_tensor_j_regressor(model, device=betas.device, dtype=betas.dtype)
    joints = model._kp21_from_verts(mano_out.vertices, j_reg)
    return mano_out.vertices[0], joints[0]


def _maybe_flip_left_hand(points_3d: torch.Tensor, is_right: bool) -> torch.Tensor:
    if is_right:
        return points_3d
    flipped = points_3d.clone()
    flipped[:, 0] *= -1.0
    return flipped


def refine_render_translation_to_bbox(
    render_vertices_3d: np.ndarray,
    cam_param: np.ndarray,
    cam_t: np.ndarray,
    det_bbox_xyxy: np.ndarray,
) -> np.ndarray:
    refined_t = np.asarray(cam_t, dtype=np.float32).reshape(3).copy()
    target = np.asarray(det_bbox_xyxy, dtype=np.float32).reshape(4)
    proj_bbox = bbox_from_points(project_points_with_translation(render_vertices_3d, cam_param, refined_t))
    if proj_bbox is None:
        return refined_t
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
            return refined_t
        proj = np.asarray(proj_bbox, dtype=np.float32).reshape(4)
    proj_cx = float(proj[0] + proj[2]) * 0.5
    proj_cy = float(proj[1] + proj[3]) * 0.5
    target_cx = float(target[0] + target[2]) * 0.5
    target_cy = float(target[1] + target[3]) * 0.5
    fx, fy = [float(v) for v in np.asarray(cam_param, dtype=np.float32).reshape(-1)[:2]]
    refined_t[0] += (target_cx - proj_cx) * refined_t[2] / max(fx, 1e-6)
    refined_t[1] += (target_cy - proj_cy) * refined_t[2] / max(fy, 1e-6)
    return refined_t.astype(np.float32)


def optimize_pose_shape_and_tvec(
    model: Any,
    init_global_orient_rm: np.ndarray,
    init_hand_pose_rm: np.ndarray,
    init_betas: np.ndarray,
    init_cam_t: np.ndarray,
    cam_param: np.ndarray,
    target_k2d: np.ndarray,
    joint_weights: np.ndarray,
    is_right: bool,
    num_iters: int,
    lr: float,
    pose_reg_weight: float,
    shape_reg_weight: float,
    tvec_reg_weight: float,
) -> dict[str, np.ndarray | float]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    init_global_orient_aa = _rotmat_to_aa(torch.from_numpy(init_global_orient_rm).to(device=device, dtype=dtype).view(-1, 3, 3)).view(1, 3)
    init_hand_pose_aa = _rotmat_to_aa(torch.from_numpy(init_hand_pose_rm).to(device=device, dtype=dtype).view(-1, 3, 3)).view(15, 3)
    init_betas_t = torch.from_numpy(np.asarray(init_betas, dtype=np.float32)).to(device=device, dtype=dtype).view(10)
    init_cam_t_t = torch.from_numpy(np.asarray(init_cam_t, dtype=np.float32)).to(device=device, dtype=dtype).view(3)
    cam_param_t = torch.from_numpy(np.asarray(cam_param, dtype=np.float32)).to(device=device, dtype=dtype).view(4)
    target_k2d_t = torch.from_numpy(np.asarray(target_k2d, dtype=np.float32)).to(device=device, dtype=dtype).view(21, 2)
    joint_weights_t = torch.from_numpy(np.asarray(joint_weights, dtype=np.float32)).to(device=device, dtype=dtype).view(21)
    joint_weights_t = joint_weights_t / torch.clamp(joint_weights_t.sum(), min=1e-6)

    global_orient_aa = init_global_orient_aa.clone().detach().requires_grad_(True)
    hand_pose_aa = init_hand_pose_aa.clone().detach().requires_grad_(True)
    betas = init_betas_t.clone().detach().requires_grad_(True)
    cam_t = init_cam_t_t.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([global_orient_aa, hand_pose_aa, betas, cam_t], lr=float(lr))

    best: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(max(1, int(num_iters))):
        optimizer.zero_grad(set_to_none=True)

        verts_right, joints_right = _decode_mano_right_hand(model, global_orient_aa, hand_pose_aa, betas)
        verts = _maybe_flip_left_hand(verts_right, is_right)
        joints = _maybe_flip_left_hand(joints_right, is_right)
        proj_k2d = _project_points_torch(joints, cam_param_t, cam_t)

        diff = proj_k2d - target_k2d_t
        per_joint = F.smooth_l1_loss(proj_k2d, target_k2d_t, reduction="none").sum(dim=1)
        reproj_loss = torch.sum(joint_weights_t * per_joint)

        pose_reg = torch.mean((global_orient_aa - init_global_orient_aa) ** 2) + torch.mean((hand_pose_aa - init_hand_pose_aa) ** 2)
        shape_reg = torch.mean((betas - init_betas_t) ** 2)
        tvec_reg = torch.mean((cam_t - init_cam_t_t) ** 2)
        depth_penalty = F.relu(0.02 - (verts[:, 2] + cam_t[2])).mean()
        loss = (
            reproj_loss
            + float(pose_reg_weight) * pose_reg
            + float(shape_reg_weight) * shape_reg
            + float(tvec_reg_weight) * tvec_reg
            + 2.0 * depth_penalty
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cam_t[2].clamp_(min=0.02)
            betas.clamp_(min=-3.0, max=3.0)
            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best = {
                    "global_orient_aa": global_orient_aa.detach().clone(),
                    "hand_pose_aa": hand_pose_aa.detach().clone(),
                    "betas": betas.detach().clone(),
                    "cam_t": cam_t.detach().clone(),
                    "verts": verts.detach().clone(),
                    "joints": joints.detach().clone(),
                    "proj_k2d": proj_k2d.detach().clone(),
                    "pixel_residual": torch.sqrt(torch.sum(diff.detach() * diff.detach(), dim=1)).mean(),
                }

    if best is None:
        raise RuntimeError("Optimization did not produce a valid solution.")

    return {
        "global_orient_aa": best["global_orient_aa"].cpu().numpy().astype(np.float32),
        "hand_pose_aa": best["hand_pose_aa"].cpu().numpy().astype(np.float32),
        "betas": best["betas"].cpu().numpy().astype(np.float32),
        "cam_t": best["cam_t"].cpu().numpy().astype(np.float32),
        "verts": best["verts"].cpu().numpy().astype(np.float32),
        "joints": best["joints"].cpu().numpy().astype(np.float32),
        "proj_k2d": best["proj_k2d"].cpu().numpy().astype(np.float32),
        "loss": float(best_loss),
        "pixel_residual": float(best["pixel_residual"].cpu()),
    }


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
            no_det = _put_text_block(image_bgr, ["No hands detected by WiLoR detector"], (32, 32, 180))
            cv2.imwrite(str(image_out_dir / "overlay.png"), no_det)
            summary = {"image": str(image_path), "num_hands": 0, "hands": []}
            (image_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            all_summary.append(summary)
            continue

        bboxes = np.stack([np.asarray(det["bbox_xyxy"], dtype=np.float32) for det in detections], axis=0)
        keypoints, scores = pose_model(image_bgr, bboxes=bboxes)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(len(detections), 21, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(len(detections), 21)

        vis_vertices_list: list[np.ndarray] = []
        vis_vertices_3d_list: list[np.ndarray] = []
        vis_depth_list: list[np.ndarray] = []
        vis_v2d_list: list[np.ndarray] = []
        target_k2d_list: list[np.ndarray] = []
        reproj_k2d_list: list[np.ndarray] = []
        meta_list: list[dict[str, Any]] = []
        hands_summary: list[dict[str, Any]] = []
        vis_faces = faces

        for det_idx, det in enumerate(detections):
            det_bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            is_right = bool(det["is_right"])
            target_k2d = np.asarray(keypoints[det_idx], dtype=np.float32)
            target_scores = np.asarray(scores[det_idx], dtype=np.float32)
            pose_mean_score = float(np.mean(target_scores))

            crop_pack = prepare_crop_weakcam(image_bgr, det_bbox, is_right, cfg, float(args.bbox_expand))
            full_cam = np.asarray(crop_pack["cam_param_full"], dtype=np.float32)
            infer_out = infer_model_full(model, crop_pack)

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

            joints_3d_init = _restore_points_to_original_handedness(infer_out["pred_keypoints_3d"], is_right)
            vertices_3d_init = _restore_points_to_original_handedness(infer_out["pred_vertices"], is_right)
            joint_weights = build_rtm_joint_weights(target_scores)
            ok, cam_t_init, reproj_rmse_weighted_init, inliers = solve_tvec_handos_style(
                points_3d=joints_3d_init,
                points_2d=target_k2d,
                cam_param=full_cam,
                init_tvec=weakcam_init_t,
                joint_weights=joint_weights,
            )
            if not ok or not np.isfinite(cam_t_init).all():
                cam_t_init = weakcam_init_t.copy()
            cam_t_init = ensure_points_in_front(vertices_3d_init, cam_t_init, z_min=float(args.z_min))

            reproj_k2d_init = project_points_with_translation(joints_3d_init, full_cam, cam_t_init)
            reproj_rmse_init = compute_reproj_rmse(joints_3d_init, target_k2d, full_cam, cam_t_init, joint_weights=None)

            opt_out = optimize_pose_shape_and_tvec(
                model=model,
                init_global_orient_rm=infer_out["global_orient_rm"],
                init_hand_pose_rm=infer_out["hand_pose_rm"],
                init_betas=infer_out["betas"],
                init_cam_t=cam_t_init,
                cam_param=full_cam,
                target_k2d=target_k2d,
                joint_weights=joint_weights,
                is_right=is_right,
                num_iters=int(args.opt_iters),
                lr=float(args.opt_lr),
                pose_reg_weight=float(args.pose_reg),
                shape_reg_weight=float(args.shape_reg),
                tvec_reg_weight=float(args.tvec_reg),
            )

            opt_cam_t = ensure_points_in_front(opt_out["verts"], np.asarray(opt_out["cam_t"], dtype=np.float32), z_min=float(args.z_min))
            vis_vertices, vis_faces = enhance_visualization_mesh(opt_out["verts"], opt_out["joints"], faces)
            if args.refine_to_bbox:
                opt_cam_t = refine_render_translation_to_bbox(vis_vertices, full_cam, opt_cam_t, det_bbox)
            opt_cam_t = ensure_points_in_front(opt_out["verts"], opt_cam_t, z_min=float(args.z_min))
            reproj_k2d = project_points_with_translation(opt_out["joints"], full_cam, opt_cam_t)
            reproj_rmse = compute_reproj_rmse(opt_out["joints"], target_k2d, full_cam, opt_cam_t, joint_weights=None)
            vis_v2d = project_points_with_translation(vis_vertices, full_cam, opt_cam_t)
            vis_depth = (vis_vertices[:, 2] + opt_cam_t[2]).astype(np.float32)

            if args.save_per_det:
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
                det_overlay = draw_joint_legend(det_overlay, "rtm")
                det_overlay = _put_text_block(
                    det_overlay,
                    [
                        "solver: pose_shape_tvec_opt",
                        f"rmse(init): {reproj_rmse_init:.2f}",
                        f"rmse(opt): {reproj_rmse:.2f}",
                    ],
                    (255, 235, 120),
                )
                cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_overlay.png"), det_overlay)
                mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
                cv2.imwrite(str(image_out_dir / f"det_{det_idx:02d}_mesh.png"), mesh_only)

            vis_vertices_list.append(vis_vertices)
            vis_vertices_3d_list.append(vis_vertices)
            vis_depth_list.append(vis_depth)
            vis_v2d_list.append(vis_v2d)
            target_k2d_list.append(target_k2d)
            reproj_k2d_list.append(reproj_k2d)
            meta_list.append(
                {
                    "bbox_xyxy": det_bbox,
                    "is_right": is_right,
                    "score": float(det.get("score", 0.0)),
                }
            )
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
                    "reproj_kpts_2d_init": reproj_k2d_init.astype(np.float32).tolist(),
                    "reproj_kpts_2d_opt": reproj_k2d.astype(np.float32).tolist(),
                    "solver": "pose_shape_tvec_opt",
                    "num_inliers_init": int(np.sum(np.asarray(inliers, dtype=bool))),
                    "reproj_rmse_init": float(reproj_rmse_init),
                    "reproj_rmse_init_weighted": float(reproj_rmse_weighted_init),
                    "reproj_rmse_opt": float(reproj_rmse),
                    "opt_loss": float(opt_out["loss"]),
                    "opt_pixel_residual": float(opt_out["pixel_residual"]),
                    "cam_param": full_cam.astype(np.float32).tolist(),
                    "cam_t_init": np.asarray(cam_t_init, dtype=np.float32).tolist(),
                    "cam_t_opt": np.asarray(opt_cam_t, dtype=np.float32).tolist(),
                    "betas_opt": np.asarray(opt_out["betas"], dtype=np.float32).tolist(),
                }
            )

        overlay = image_bgr.copy()
        if vis_vertices_list:
            overlay = render_overlay_multiple_image_space_shaded(
                full_image_bgr=overlay,
                vertices_2d_list=vis_v2d_list,
                vertices_3d_list=vis_vertices_3d_list,
                vertices_depth_list=vis_depth_list,
                faces=vis_faces,
                color_hex=args.mesh_color,
            )
            for target_k2d, reproj_k2d, meta in zip(target_k2d_list, reproj_k2d_list, meta_list):
                overlay = draw_hand_joints(overlay, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
                overlay = draw_hand_joints(overlay, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
                overlay = draw_detection_bbox(overlay, meta["bbox_xyxy"], bool(meta["is_right"]), meta.get("score"))
            overlay = draw_joint_legend(overlay, "rtm")
        else:
            overlay = _put_text_block(overlay, ["No valid reconstruction results"], (32, 32, 180))

        cv2.imwrite(str(image_out_dir / "overlay.png"), overlay)
        summary = {"image": str(image_path), "num_hands": len(hands_summary), "hands": hands_summary}
        (image_out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        all_summary.append(summary)
        print(f"[WiLoR+RTM+PoseShapeOpt] {image_path.name}: reconstructed {len(hands_summary)} hand(s)")

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
    print(f"Saved WiLoR + RTM + GPGFormer pose/shape optimization visualization to: {out_dir}")


if __name__ == "__main__":
    main()

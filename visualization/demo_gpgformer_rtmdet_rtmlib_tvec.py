from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RTMLIB_REPO = Path('/root/code/hand_reconstruction/rtmlib')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if RTMLIB_REPO.exists() and str(RTMLIB_REPO) not in sys.path:
    sys.path.insert(0, str(RTMLIB_REPO))

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import numpy as np
import torch

from rtmlib import RTMDet, RTMPose
from rtmlib.tools.object_detection.post_processings import multiclass_nms

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

DEFAULT_IMG_DIR = Path('/root/code/hand_reconstruction/GPGFormer/in-the-wild')
# DEFAULT_IMG_DIR = Path('/root/code/hand_reconstruction/WiLoR/demo_img')
DEFAULT_OUT_DIR = Path('/root/code/vepfs/GPGFormer/outputs/gpgformer_dexycb_rtmdet_rtmlib_tvec')
DEFAULT_CFG = REPO_ROOT / 'configs' / 'config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml'
DEFAULT_CKPT = Path(
    '/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt'
)
DEFAULT_DET_MODEL = Path(
    '/root/code/hand_reconstruction/rtmlib/rtmdet_onnx/'
    'rtmdet_nano_8xb32-300e_hand-267f9c8f'
)
DEFAULT_WILOR_HANDEDNESS_MODEL = Path('/root/code/hand_reconstruction/WiLoR/pretrained_models/detector.pt')
DEFAULT_POSE_MODEL = Path(
    '/root/code/hand_reconstruction/rtmlib/rtmpose_onnx/'
    'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx'
)
DEFAULT_DET_INPUT_SIZES = '320,640'
DEFAULT_POSE_INPUT_SIZE = (256, 256)
DEFAULT_MESH_COLOR = '#7677D8'
DEFAULT_ROOT_INDEX = 9

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
        description='GPGFormer hand mesh reconstruction using RTMDet hand detection and RTMPose 2D joints.'
    )
    parser.add_argument('--img-dir', type=str, default=str(DEFAULT_IMG_DIR))
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument('--config', type=str, default=str(DEFAULT_CFG))
    parser.add_argument('--ckpt', type=str, default=str(DEFAULT_CKPT))
    parser.add_argument('--det-model', type=str, default=str(DEFAULT_DET_MODEL), help='RTMDet ONNX file or directory.')
    parser.add_argument(
        '--wilor-handedness-model',
        type=str,
        default=str(DEFAULT_WILOR_HANDEDNESS_MODEL),
        help='Optional WiLoR detector.pt used only as a handedness oracle for RTMDet boxes.',
    )
    parser.add_argument('--pose-model', type=str, default=str(DEFAULT_POSE_MODEL), help='RTMPose hand ONNX path.')
    parser.add_argument('--device', type=str, default='cuda', help='GPGFormer device.')
    parser.add_argument('--pose-device', type=str, default='auto', help='RTMPose device, e.g. auto/cpu/cuda.')
    parser.add_argument('--detector-device', type=str, default='auto', help='RTMDet device, e.g. auto/cpu/cuda.')
    parser.add_argument('--wilor-device', type=str, default='auto', help='WiLoR handedness oracle device, e.g. auto/cpu/0.')
    parser.add_argument('--backend', type=str, default='onnxruntime', choices=('onnxruntime', 'opencv', 'openvino'))
    parser.add_argument('--det-input-sizes', type=str, default=DEFAULT_DET_INPUT_SIZES, help='Comma-separated RTMDet input sizes, e.g. "320,640".')
    parser.add_argument('--det-score-thr', type=float, default=0.18)
    parser.add_argument('--det-nms-iou', type=float, default=0.5)
    parser.add_argument('--det-flip-test', dest='det_flip_test', action='store_true')
    parser.add_argument('--no-det-flip-test', dest='det_flip_test', action='store_false')
    parser.add_argument('--bbox-expand', type=float, default=1.25)
    parser.add_argument('--mesh-color', type=str, default=DEFAULT_MESH_COLOR)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--max-dets-per-image', type=int, default=0)
    parser.add_argument('--pose-min-mean-score', type=float, default=0.15)
    parser.add_argument('--wilor-handedness-conf', type=float, default=0.25)
    parser.add_argument('--wilor-handedness-iou', type=float, default=0.6)
    parser.add_argument('--wilor-match-iou', type=float, default=0.2)
    parser.add_argument('--z-min', type=float, default=0.05)
    parser.add_argument('--root-index', type=int, default=DEFAULT_ROOT_INDEX)
    parser.add_argument('--save-joint-debug', action='store_true')
    parser.add_argument('--save-per-det', action='store_true')
    parser.add_argument('--save-hypothesis-debug', action='store_true')
    parser.set_defaults(det_flip_test=True)
    return parser.parse_args()


def _resolve_torch_device(device_str: str) -> torch.device:
    if device_str == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_str)


def _resolve_aux_device(device_str: str, main_device: torch.device) -> str:
    if device_str != 'auto':
        return device_str
    if main_device.type == 'cuda':
        return 'cuda'
    return 'cpu'


def _resolve_wilor_device(device_str: str, main_device: torch.device) -> str:
    if device_str != 'auto':
        return device_str
    if main_device.type == 'cuda':
        return '0'
    return 'cpu'


def _resolve_onnx_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_dir():
        onnx_path = path / 'end2end.onnx'
        if onnx_path.exists():
            return onnx_path
    return path


def _parse_det_input_sizes(spec: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for chunk in str(spec).split(','):
        token = chunk.strip().lower()
        if not token:
            continue
        if 'x' in token:
            h_str, w_str = token.split('x', 1)
            size = (max(32, int(h_str)), max(32, int(w_str)))
        else:
            side = max(32, int(token))
            size = (side, side)
        if size not in sizes:
            sizes.append(size)
    return sizes or [(320, 320)]


def build_rtm_joint_weights(scores: np.ndarray) -> np.ndarray:
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    conf = np.clip(conf, 0.05, 1.5) / 1.5
    return (RTM_SEMANTIC_WEIGHTS * (0.35 + 0.65 * conf)).astype(np.float32)


def _bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(box_a, dtype=np.float32).reshape(4)]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(box_b, dtype=np.float32).reshape(4)]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def _clip_bbox_xyxy(bbox_xyxy: np.ndarray, image_bgr: np.ndarray) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    x1 = float(np.clip(x1, 0.0, max(0.0, w - 1.0)))
    y1 = float(np.clip(y1, 0.0, max(0.0, h - 1.0)))
    x2 = float(np.clip(x2, x1 + 1.0, float(w)))
    y2 = float(np.clip(y2, y1 + 1.0, float(h)))
    if x2 - x1 <= 2.0 or y2 - y1 <= 2.0:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _decode_rtmdet_outputs(
    detector: RTMDet,
    outputs: np.ndarray,
    ratio: float,
    score_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(outputs)
    if raw.ndim < 3:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if raw.shape[-1] == 5:
        boxes = raw[0, :, :4].astype(np.float32) / max(float(ratio), 1e-9)
        scores = raw[0, :, 4].astype(np.float32)
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= float(score_thr))
        return boxes[valid], scores[valid]

    if raw.shape[-1] != 4:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hsizes = [detector.model_input_size[0] // stride for stride in strides]
    wsizes = [detector.model_input_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride, dtype=np.float32))

    grids = np.concatenate(grids, 1).astype(np.float32)
    expanded_strides = np.concatenate(expanded_strides, 1).astype(np.float32)
    preds = raw.copy()
    preds[..., :2] = (preds[..., :2] + grids) * expanded_strides
    preds[..., 2:4] = np.exp(preds[..., 2:4]) * expanded_strides

    predictions = preds[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= max(float(ratio), 1e-9)
    dets, _ = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=float(score_thr))
    if dets is None or len(dets) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    dets = np.asarray(dets, dtype=np.float32)
    class_ids = dets[:, 5].astype(np.int32)
    valid = class_ids == 0
    return dets[valid, :4].astype(np.float32), dets[valid, 4].astype(np.float32)


def _collect_rtmdet_candidates(
    detector: RTMDet,
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
    score_thr: float,
    flip_test: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    h, w = image_bgr.shape[:2]
    base_h, base_w = [int(v) for v in detector.model_input_size]
    scale_y = float(input_size[0]) / max(float(base_h), 1.0)
    scale_x = float(input_size[1]) / max(float(base_w), 1.0)

    if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
        scaled_w = max(8, int(round(w * scale_x)))
        scaled_h = max(8, int(round(h * scale_y)))
        image_work = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_work = image_bgr

    prep_img, ratio = detector.preprocess(image_work)
    outputs = detector.inference(prep_img)[0]
    boxes, scores = _decode_rtmdet_outputs(detector, outputs, ratio, score_thr)
    if boxes.size:
        boxes[:, [0, 2]] /= max(scale_x, 1e-9)
        boxes[:, [1, 3]] /= max(scale_y, 1e-9)
    for bbox_xyxy, score in zip(boxes, scores):
        clipped = _clip_bbox_xyxy(bbox_xyxy, image_bgr)
        if clipped is None:
            continue
        candidates.append(
            {
                'bbox_xyxy': clipped,
                'score': float(score),
                'class_id': 0,
                'source': f'rtmdet_scale_{input_size[0]}x{input_size[1]}',
            }
        )

    if not bool(flip_test):
        return candidates

    image_flip = cv2.flip(image_work, 1)
    prep_flip, ratio_flip = detector.preprocess(image_flip)
    outputs_flip = detector.inference(prep_flip)[0]
    boxes_flip, scores_flip = _decode_rtmdet_outputs(detector, outputs_flip, ratio_flip, score_thr)
    if boxes_flip.size:
        boxes_flip[:, [0, 2]] /= max(scale_x, 1e-9)
        boxes_flip[:, [1, 3]] /= max(scale_y, 1e-9)
    for bbox_xyxy, score in zip(boxes_flip, scores_flip):
        x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
        bbox_orig = np.array([w - x2, y1, w - x1, y2], dtype=np.float32)
        clipped = _clip_bbox_xyxy(bbox_orig, image_bgr)
        if clipped is None:
            continue
        candidates.append(
            {
                'bbox_xyxy': clipped,
                'score': float(score),
                'class_id': 0,
                'source': f'rtmdet_flip_scale_{input_size[0]}x{input_size[1]}',
            }
        )
    return candidates


def detect_rtm_hands(
    detector: RTMDet,
    image_bgr: np.ndarray,
    max_dets: int,
    score_thr: float,
    nms_iou: float,
    input_sizes: list[tuple[int, int]],
    flip_test: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for input_size in input_sizes:
        candidates.extend(
            _collect_rtmdet_candidates(
                detector=detector,
                image_bgr=image_bgr,
                input_size=input_size,
                score_thr=float(score_thr),
                flip_test=bool(flip_test),
            )
        )

    if not candidates:
        return []

    candidates.sort(
        key=lambda det: (
            -float(det.get('score', 0.0)),
            -float((det['bbox_xyxy'][2] - det['bbox_xyxy'][0]) * (det['bbox_xyxy'][3] - det['bbox_xyxy'][1])),
        )
    )

    detections: list[dict[str, Any]] = []
    keep_n = len(candidates) if int(max_dets) <= 0 else int(max_dets)
    for det in candidates:
        bbox_xyxy = np.asarray(det['bbox_xyxy'], dtype=np.float32)
        area = float((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))
        if not np.isfinite(bbox_xyxy).all() or area <= 4.0:
            continue
        if any(_bbox_iou_xyxy(bbox_xyxy, kept['bbox_xyxy']) >= float(nms_iou) for kept in detections):
            continue
        detections.append(
            {
                'bbox_xyxy': bbox_xyxy.astype(np.float32),
                'score': float(det.get('score', 0.0)),
                'class_id': 0,
                'source': str(det.get('source', 'rtmdet')),
            }
        )
        if len(detections) >= keep_n:
            break
    return detections


def _match_wilor_handedness(
    det_bbox_xyxy: np.ndarray,
    wilor_detections: list[dict[str, Any]],
    min_iou: float,
) -> dict[str, Any] | None:
    best = None
    best_iou = -1.0
    det_bbox = np.asarray(det_bbox_xyxy, dtype=np.float32).reshape(4)
    for cand in wilor_detections:
        iou = _bbox_iou_xyxy(det_bbox, np.asarray(cand['bbox_xyxy'], dtype=np.float32))
        if iou > best_iou:
            best_iou = iou
            best = cand
    if best is None or best_iou < float(min_iou):
        return None
    matched = dict(best)
    matched['match_iou'] = float(best_iou)
    return matched


def _bbox_fit_score(projected_bbox: np.ndarray | None, target_bbox: np.ndarray) -> float:
    if projected_bbox is None:
        return 1e6
    proj = np.asarray(projected_bbox, dtype=np.float32).reshape(4)
    target = np.asarray(target_bbox, dtype=np.float32).reshape(4)

    proj_w = max(float(proj[2] - proj[0]), 1e-6)
    proj_h = max(float(proj[3] - proj[1]), 1e-6)
    tgt_w = max(float(target[2] - target[0]), 1e-6)
    tgt_h = max(float(target[3] - target[1]), 1e-6)

    proj_cx = 0.5 * float(proj[0] + proj[2])
    proj_cy = 0.5 * float(proj[1] + proj[3])
    tgt_cx = 0.5 * float(target[0] + target[2])
    tgt_cy = 0.5 * float(target[1] + target[3])
    tgt_diag = max(float(np.hypot(tgt_w, tgt_h)), 1e-6)

    center_term = float(np.hypot(proj_cx - tgt_cx, proj_cy - tgt_cy) / tgt_diag)
    size_term = abs(np.log(proj_w / tgt_w)) + abs(np.log(proj_h / tgt_h))
    return float(30.0 * center_term + 20.0 * size_term)


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


def evaluate_handedness_hypothesis(
    model: Any,
    cfg: dict[str, Any],
    faces: np.ndarray,
    image_bgr: np.ndarray,
    det_bbox: np.ndarray,
    target_k2d: np.ndarray,
    target_scores: np.ndarray,
    is_right: bool,
    bbox_expand: float,
    root_index: int,
    z_min: float,
) -> dict[str, Any] | None:
    crop_pack = prepare_crop_weakcam(image_bgr, det_bbox, is_right, cfg, float(bbox_expand))
    full_cam = np.asarray(crop_pack['cam_param_full'], dtype=np.float32)
    infer_out = infer_model(model, crop_pack)

    v3d_raw = restore_points_to_original_handedness(infer_out['v3d_crop'], is_right)
    k3d_raw = restore_points_to_original_handedness(infer_out['k3d_crop'], is_right)
    v3d, k3d, root_xyz = convert_to_root_relative(v3d_raw, k3d_raw, int(root_index))

    pred_cam = np.asarray(infer_out['pred_cam_crop'], dtype=np.float32).reshape(3).copy()
    if not is_right:
        pred_cam[1] *= -1.0
    weakcam_init_t = weakcam_crop_to_full(
        pred_cam=pred_cam,
        box_center=crop_pack['box_center'],
        box_size=crop_pack['box_size'],
        img_size_wh=crop_pack['img_size_wh'],
        focal_length=float(crop_pack['focal_full']),
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
        return None

    cam_t = ensure_points_in_front(v3d, cam_t, z_min=float(z_min))
    vis_vertices, vis_faces = enhance_visualization_mesh(v3d, k3d, faces, is_right=is_right)
    cam_t = refine_render_translation_to_bbox(
        render_vertices_3d=vis_vertices,
        cam_param=full_cam,
        cam_t=cam_t,
        det_bbox_xyxy=det_bbox,
    )
    cam_t = ensure_points_in_front(v3d, cam_t, z_min=float(z_min))

    reproj_k2d = project_points_with_translation(k3d, full_cam, cam_t)
    reproj_rmse = compute_reproj_rmse(k3d, target_k2d, full_cam, cam_t, joint_weights=None)
    vis_v2d = project_points_with_translation(vis_vertices, full_cam, cam_t)
    proj_bbox = bbox_from_points(vis_v2d)
    bbox_fit = _bbox_fit_score(proj_bbox, det_bbox)
    score = float(reproj_rmse_weighted + 0.35 * reproj_rmse + bbox_fit)

    return {
        'is_right': bool(is_right),
        'label': 'right' if is_right else 'left',
        'crop_pack': crop_pack,
        'infer_out': infer_out,
        'full_cam': full_cam,
        'v3d': v3d,
        'k3d': k3d,
        'vis_vertices': vis_vertices,
        'vis_faces': vis_faces,
        'cam_t': np.asarray(cam_t, dtype=np.float32),
        'reproj_k2d': reproj_k2d.astype(np.float32),
        'reproj_rmse': float(reproj_rmse),
        'reproj_rmse_weighted': float(reproj_rmse_weighted),
        'bbox_fit_score': float(bbox_fit),
        'selection_score': score,
        'joint_weights': joint_weights.astype(np.float32),
        'num_inliers': int(np.sum(np.asarray(inliers, dtype=bool))),
    }


def choose_best_handedness(
    model: Any,
    cfg: dict[str, Any],
    faces: np.ndarray,
    image_bgr: np.ndarray,
    det_bbox: np.ndarray,
    target_k2d: np.ndarray,
    target_scores: np.ndarray,
    bbox_expand: float,
    root_index: int,
    z_min: float,
    preferred_is_right: bool | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    left = evaluate_handedness_hypothesis(
        model=model,
        cfg=cfg,
        faces=faces,
        image_bgr=image_bgr,
        det_bbox=det_bbox,
        target_k2d=target_k2d,
        target_scores=target_scores,
        is_right=False,
        bbox_expand=bbox_expand,
        root_index=root_index,
        z_min=z_min,
    )
    right = evaluate_handedness_hypothesis(
        model=model,
        cfg=cfg,
        faces=faces,
        image_bgr=image_bgr,
        det_bbox=det_bbox,
        target_k2d=target_k2d,
        target_scores=target_scores,
        is_right=True,
        bbox_expand=bbox_expand,
        root_index=root_index,
        z_min=z_min,
    )

    candidates = [c for c in (left, right) if c is not None]
    if not candidates:
        return None, left, right

    if preferred_is_right is not None:
        preferred = right if bool(preferred_is_right) else left
        if preferred is not None:
            return preferred, left, right

    best = min(candidates, key=lambda x: float(x['selection_score']))
    return best, left, right


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_torch_device(args.device)
    detector_device = _resolve_aux_device(args.detector_device, device)
    pose_device = _resolve_aux_device(args.pose_device, device)
    wilor_device = _resolve_wilor_device(args.wilor_device, device)
    det_input_sizes = _parse_det_input_sizes(args.det_input_sizes)

    model, cfg = load_gpgformer(Path(args.config), Path(args.ckpt), device)
    det_model_path = _resolve_onnx_path(args.det_model)
    detector = RTMDet(
        str(det_model_path),
        model_input_size=det_input_sizes[0],
        backend=args.backend,
        device=detector_device,
    )
    pose_model = RTMPose(
        str(Path(args.pose_model)),
        model_input_size=DEFAULT_POSE_INPUT_SIZE,
        backend=args.backend,
        device=pose_device,
    )
    faces = extract_mano_faces(model)

    wilor_handedness_detector = None
    wilor_handedness_path = Path(args.wilor_handedness_model) if str(args.wilor_handedness_model).strip() else None
    if wilor_handedness_path is not None and wilor_handedness_path.is_file():
        try:
            wilor_handedness_detector = load_wilor_detector(wilor_handedness_path)
        except Exception as exc:
            print(f'[WARN] Failed to load WiLoR handedness oracle from {wilor_handedness_path}: {exc}')

    print(f'Loaded GPGFormer: {args.ckpt}')
    print(f'Loaded RTMDet hand detector: {det_model_path}')
    print(f'Loaded RTMPose model: {args.pose_model}')
    print(f'RTMDet input sizes: {det_input_sizes}')
    if wilor_handedness_detector is not None:
        print(f'Loaded WiLoR handedness oracle: {wilor_handedness_path}')

    image_paths = collect_image_paths(img_dir)
    if args.max_images > 0:
        image_paths = image_paths[: int(args.max_images)]

    all_summary: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        detections = detect_rtm_hands(
            detector=detector,
            image_bgr=image_bgr,
            max_dets=int(args.max_dets_per_image),
            score_thr=float(args.det_score_thr),
            nms_iou=float(args.det_nms_iou),
            input_sizes=det_input_sizes,
            flip_test=bool(args.det_flip_test),
        )

        wilor_handedness_dets: list[dict[str, Any]] = []
        if wilor_handedness_detector is not None:
            wilor_handedness_dets = detect_wilor_hands(
                detector=wilor_handedness_detector,
                image_bgr=image_bgr,
                conf=float(args.wilor_handedness_conf),
                iou=float(args.wilor_handedness_iou),
                device=wilor_device,
                max_dets=max(int(args.max_dets_per_image), 16),
            )

        image_out_dir = out_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out_dir / 'input.png'), image_bgr)

        if not detections:
            cv2.imwrite(str(image_out_dir / 'overlay.png'), image_bgr)
            summary = {'image': str(image_path), 'num_hands': 0, 'hands': []}
            (image_out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
            all_summary.append(summary)
            print(f'[RTMDet+RTMPose+Tvec] {image_path.name}: no detections')
            continue

        bboxes = np.stack([np.asarray(det['bbox_xyxy'], dtype=np.float32) for det in detections], axis=0)
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
            det_bbox = np.asarray(det['bbox_xyxy'], dtype=np.float32)
            target_k2d = np.asarray(keypoints[det_idx], dtype=np.float32)
            target_scores = np.asarray(scores[det_idx], dtype=np.float32)
            pose_mean_score = float(np.mean(target_scores))
            if pose_mean_score < float(args.pose_min_mean_score):
                print(f'[WARN] {image_path.name} det{det_idx:02d}: skip low RTMPose mean score {pose_mean_score:.3f}')
                continue

            matched_wilor = _match_wilor_handedness(
                det_bbox_xyxy=det_bbox,
                wilor_detections=wilor_handedness_dets,
                min_iou=float(args.wilor_match_iou),
            )
            preferred_is_right = None if matched_wilor is None else bool(matched_wilor['is_right'])

            best, left_hyp, right_hyp = choose_best_handedness(
                model=model,
                cfg=cfg,
                faces=faces,
                image_bgr=image_bgr,
                det_bbox=det_bbox,
                target_k2d=target_k2d,
                target_scores=target_scores,
                bbox_expand=float(args.bbox_expand),
                root_index=int(args.root_index),
                z_min=float(args.z_min),
                preferred_is_right=preferred_is_right,
            )
            if best is None:
                print(f'[WARN] {image_path.name} det{det_idx:02d}: handedness selection failed')
                continue

            is_right = bool(best['is_right'])
            full_cam = np.asarray(best['full_cam'], dtype=np.float32)
            overlay_cam_param = full_cam
            vis_vertices = np.asarray(best['vis_vertices'], dtype=np.float32)
            vis_faces = np.asarray(best['vis_faces'], dtype=np.int64)
            cam_t = np.asarray(best['cam_t'], dtype=np.float32)
            reproj_k2d = np.asarray(best['reproj_k2d'], dtype=np.float32)
            reproj_rmse = float(best['reproj_rmse'])
            reproj_rmse_weighted = float(best['reproj_rmse_weighted'])
            handedness_source = 'wilor_match' if preferred_is_right is not None else 'score_only'

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
                cv2.imwrite(str(image_out_dir / f'det_{det_idx:02d}_overlay.png'), det_overlay)

                if args.save_joint_debug:
                    joint_debug = image_bgr.copy()
                    joint_debug = draw_hand_joints(joint_debug, target_k2d, TARGET_COLOR, TARGET_LINK_COLOR, hollow=False)
                    joint_debug = draw_hand_joints(joint_debug, reproj_k2d, REPROJ_COLOR, REPROJ_LINK_COLOR, hollow=True)
                    joint_debug = draw_detection_bbox(joint_debug, det_bbox, is_right, float(det.get('score', 0.0)))
                    joint_debug = draw_joint_legend(joint_debug, 'rtm')
                    cv2.imwrite(str(image_out_dir / f'det_{det_idx:02d}_joints.png'), joint_debug)

                if args.save_hypothesis_debug:
                    for hyp in (left_hyp, right_hyp):
                        if hyp is None:
                            continue
                        hyp_overlay = render_overlay_paper_style(
                            full_image_bgr=image_bgr.copy(),
                            vertices=np.asarray(hyp['vis_vertices'], dtype=np.float32),
                            faces=np.asarray(hyp['vis_faces'], dtype=np.int64),
                            cam_t=np.asarray(hyp['cam_t'], dtype=np.float32),
                            cam_param=np.asarray(hyp['full_cam'], dtype=np.float32),
                            color_hex=args.mesh_color,
                            is_right=bool(hyp['is_right']),
                        )
                        tag = 'right' if bool(hyp['is_right']) else 'left'
                        cv2.imwrite(str(image_out_dir / f'det_{det_idx:02d}_hyp_{tag}.png'), hyp_overlay)

                mesh_only = render_mesh_only(vis_vertices, vis_faces, args.mesh_color, is_right, 512)
                cv2.imwrite(str(image_out_dir / f'det_{det_idx:02d}_mesh.png'), mesh_only)

            vis_vertices_list.append(vis_vertices)
            vis_faces_list.append(vis_faces)
            cam_t_list.append(cam_t)
            is_right_list.append(is_right)

            hands_summary.append(
                {
                    'hand_index': det_idx,
                    'handedness': 'right' if is_right else 'left',
                    'is_right': is_right,
                    'detector_score': float(det.get('score', 0.0)),
                    'detector_source': str(det.get('source', 'rtmdet')),
                    'bbox_xyxy': det_bbox.astype(np.float32).tolist(),
                    'pose_mean_score': pose_mean_score,
                    'kpt_scores': target_scores.astype(np.float32).tolist(),
                    'target_kpts_2d': target_k2d.astype(np.float32).tolist(),
                    'reproj_kpts_2d': reproj_k2d.astype(np.float32).tolist(),
                    'solver': 'handos_tvec',
                    'weight_profile': 'semantic',
                    'num_inliers': int(best['num_inliers']),
                    'reproj_rmse': reproj_rmse,
                    'reproj_rmse_weighted': reproj_rmse_weighted,
                    'cam_param': full_cam.astype(np.float32).tolist(),
                    'cam_t': cam_t.astype(np.float32).tolist(),
                    'handedness_selection': {
                        'source': handedness_source,
                        'selected': str(best['label']),
                        'selected_score': float(best['selection_score']),
                        'selected_bbox_fit_score': float(best['bbox_fit_score']),
                        'wilor_match': None if matched_wilor is None else {
                            'handedness': str(matched_wilor.get('handedness', 'right' if matched_wilor.get('is_right') else 'left')),
                            'score': float(matched_wilor.get('score', 0.0)),
                            'match_iou': float(matched_wilor.get('match_iou', 0.0)),
                            'bbox_xyxy': np.asarray(matched_wilor['bbox_xyxy'], dtype=np.float32).tolist(),
                        },
                        'left': None if left_hyp is None else {
                            'score': float(left_hyp['selection_score']),
                            'reproj_rmse': float(left_hyp['reproj_rmse']),
                            'reproj_rmse_weighted': float(left_hyp['reproj_rmse_weighted']),
                            'bbox_fit_score': float(left_hyp['bbox_fit_score']),
                        },
                        'right': None if right_hyp is None else {
                            'score': float(right_hyp['selection_score']),
                            'reproj_rmse': float(right_hyp['reproj_rmse']),
                            'reproj_rmse_weighted': float(right_hyp['reproj_rmse_weighted']),
                            'bbox_fit_score': float(right_hyp['bbox_fit_score']),
                        },
                    },
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

        cv2.imwrite(str(image_out_dir / 'overlay.png'), overlay)

        summary = {
            'image': str(image_path),
            'num_hands': len(hands_summary),
            'hands': hands_summary,
        }
        (image_out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        all_summary.append(summary)
        print(f'[RTMDet+RTMPose+Tvec] {image_path.name}: reconstructed {len(hands_summary)} hand(s)')

    (out_dir / 'summary_all.json').write_text(
        json.dumps(
            {
                'img_dir': str(img_dir),
                'config': str(args.config),
                'ckpt': str(args.ckpt),
                'det_model': str(args.det_model),
                'pose_model': str(args.pose_model),
                'num_images': len(all_summary),
                'results': all_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    print(f'Saved RTMDet + RTMPose + GPGFormer tvec visualization to: {out_dir}')


if __name__ == '__main__':
    main()

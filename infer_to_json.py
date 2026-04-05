"""
GPGFormer inference + evaluation.

Outputs requested metrics:
1) MPJPE / PA-MPJPE
2) MPVPE / PA-MPVPE
3) F-score@5mm / F-score@15mm
4) AUC-J / AUC-V
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

from gpgformer.metrics.pose_metrics import compute_similarity_transform
from gpgformer.models import GPGFormer, GPGFormerConfig
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat


def calculate_fscore(gt: np.ndarray, pred: np.ndarray, th_m: float) -> Tuple[float, float, float]:
    """Compute F-score at threshold (meters) for two point clouds."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    if gt.size == 0 or pred.size == 0:
        return 0.0, 0.0, 0.0

    if o3d is not None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        pcd_pr = o3d.geometry.PointCloud()
        pcd_pr.points = o3d.utility.Vector3dVector(pred)
        d_gt2pr = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pr), dtype=np.float64)
        d_pr2gt = np.asarray(pcd_pr.compute_point_cloud_distance(pcd_gt), dtype=np.float64)
    else:
        # Fallback without open3d.
        tree_pr = cKDTree(pred)
        d_gt2pr, _ = tree_pr.query(gt, k=1)
        tree_gt = cKDTree(gt)
        d_pr2gt, _ = tree_gt.query(pred, k=1)

    precision = float(np.mean(d_gt2pr < th_m))
    recall = float(np.mean(d_pr2gt < th_m))
    if precision + recall <= 0.0:
        return 0.0, precision, recall
    fscore = 2.0 * precision * recall / (precision + recall)
    return float(fscore), precision, recall


def compute_auc(errors_mm: np.ndarray, val_min_mm: float = 0.0, val_max_mm: float = 50.0, steps: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute normalized AUC of PCK curve.
    errors_mm: flattened point-wise euclidean errors in mm.
    """
    errors_mm = np.asarray(errors_mm, dtype=np.float64).reshape(-1)
    errors_mm = errors_mm[np.isfinite(errors_mm)]
    if errors_mm.size == 0:
        thresholds = np.linspace(val_min_mm, val_max_mm, steps, dtype=np.float64)
        return float("nan"), np.zeros_like(thresholds), thresholds

    thresholds = np.linspace(val_min_mm, val_max_mm, steps, dtype=np.float64)
    pck = np.array([(errors_mm <= t).mean() for t in thresholds], dtype=np.float64)
    denom = max(float(val_max_mm - val_min_mm), 1e-9)
    auc = float(np.trapezoid(pck, thresholds) / denom)
    return auc, pck, thresholds


def procrustes_align_batch(pred_m: np.ndarray, gt_m: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """PA align predicted points to GT, both in meters, shape (B,N,3)."""
    pred_m = np.asarray(pred_m, dtype=np.float32)
    gt_m = np.asarray(gt_m, dtype=np.float32)
    if pred_m.shape != gt_m.shape:
        raise ValueError(f"Shape mismatch for PA alignment: pred={pred_m.shape}, gt={gt_m.shape}")

    out = np.empty_like(pred_m, dtype=np.float32)
    for i in range(0, pred_m.shape[0], chunk_size):
        p = torch.from_numpy(pred_m[i : i + chunk_size])
        g = torch.from_numpy(gt_m[i : i + chunk_size])
        pa = compute_similarity_transform(p, g).cpu().numpy().astype(np.float32)
        out[i : i + chunk_size] = pa
    return out


def build_dataset(cfg: dict):
    """Build eval dataset for FreiHAND / HO3D / DexYCB."""
    name = str(cfg["dataset"]["name"]).lower()
    cfg_bbox_eval = str(cfg["dataset"].get("bbox_source_eval", "detector")).lower()
    if cfg_bbox_eval != "detector":
        print(
            f"[warn] infer_to_json enforces detector-only hand crop for evaluation. "
            f"Overriding dataset.bbox_source_eval='{cfg_bbox_eval}' -> 'detector'."
        )
    bbox_source = "detector"
    detector_path = str(cfg["paths"].get("detector_ckpt", "")).strip()
    if len(detector_path) == 0:
        raise ValueError(
            "infer_to_json requires paths.detector_ckpt because evaluation is detector-only (no GT bbox crop)."
        )
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))

    if name in ("dexycb", "dex-ycb"):
        from data.dex_ycb_dataset import DexYCBDataset

        return DexYCBDataset(
            setup=cfg["dataset"]["dexycb_setup"],
            split="test",
            root_dir=cfg["paths"]["dexycb_root"],
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
        )

    if name in ("ho3d",):
        use_ho3d_json = bool(cfg["dataset"].get("ho3d_use_json_split", False))
        ho3d_train_json = str(cfg["paths"].get("ho3d_train_json", "")).strip()
        ho3d_test_json = str(cfg["paths"].get("ho3d_test_json", "")).strip()
        ho3d_eval_split = str(cfg["dataset"].get("ho3d_eval_split", "val")).lower()
        if use_ho3d_json:
            json_split = "evaluation"
            required_json = ho3d_test_json
            required_json_key = "paths.ho3d_test_json"

            config_hint = str(cfg.get("__config_path__", "<unknown>"))
            if not required_json:
                raise FileNotFoundError(
                    f"dataset.name=ho3d with dataset.ho3d_use_json_split=true, "
                    f"but {required_json_key} is empty for split '{json_split}'. "
                    f"config={config_hint}"
                )
            if not osp.exists(required_json):
                raise FileNotFoundError(
                    f"dataset.name=ho3d with dataset.ho3d_use_json_split=true, "
                    f"but {required_json_key} does not exist: {required_json}. "
                    f"split='{json_split}', config={config_hint}"
                )

            from data.ho3d_json_dataset import HO3DJsonDataset

            return HO3DJsonDataset(
                data_split=json_split,
                root_dir=cfg["paths"]["ho3d_root"],
                train_json_path=ho3d_train_json,
                test_json_path=ho3d_test_json,
                img_size=int(cfg["dataset"].get("img_size", 256)),
                train=False,
                align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
                wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
                bbox_source=bbox_source,
                detector_weights_path=detector_path,
                trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
                trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
                trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
                root_index=root_index,
                json_kp3d_unit=str(cfg["dataset"].get("ho3d_json_kp3d_unit", "auto")),
                json_kp3d_scale=float(cfg["dataset"].get("ho3d_json_kp3d_scale", 1.0)),
                json_convert_xyz=bool(cfg["dataset"].get("ho3d_json_convert_xyz", False)),
            )

        from data.ho3d_dataset import HO3DDataset

        return HO3DDataset(
            data_split=ho3d_eval_split,
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            train_split_file=str(cfg["paths"].get("ho3d_train_split_file", osp.join(cfg["paths"]["ho3d_root"], "train.txt"))),
            eval_split_file=str(cfg["paths"].get("ho3d_eval_split_file", osp.join(cfg["paths"]["ho3d_root"], "evaluation.txt"))),
            eval_xyz_json=str(cfg["paths"].get("ho3d_eval_xyz_json", osp.join(cfg["paths"]["ho3d_root"], "evaluation_xyz.json"))),
            eval_verts_json=str(cfg["paths"].get("ho3d_eval_verts_json", osp.join(cfg["paths"]["ho3d_root"], "evaluation_verts.json"))),
            root_index=root_index,
        )

    if name in ("freihand",):
        from data.freihand_dataset_v3 import FreiHANDDatasetV3

        return FreiHANDDatasetV3(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
            use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
            trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
        )

    raise ValueError(f"Unknown dataset.name: {cfg['dataset']['name']}")


def build_model_from_cfg(cfg: dict) -> GPGFormer:
    model_cfg = cfg.get("model", {})
    refiner_cfg = model_cfg.get("feature_refiner", {})
    moge2_num_tokens = int(model_cfg.get("moge2_num_tokens", 400))
    side_tuning_cfg = model_cfg.get("side_tuning", {})
    geo_side_adapter_cfg = model_cfg.get("geo_side_adapter", {})
    if moge2_num_tokens <= 0:
        raise ValueError(f"model.moge2_num_tokens must be a positive int, got {moge2_num_tokens}")

    return GPGFormer(
        GPGFormerConfig(
            backbone_type=str(model_cfg.get("backbone_type", "wilor")),
            wilor_ckpt_path=cfg["paths"].get("wilor_ckpt", ""),
            vitpose_ckpt_path=cfg["paths"].get("vitpose_ckpt", ""),
            moge2_weights_path=cfg["paths"].get("moge2_ckpt", None),
            use_geo_prior=bool(model_cfg.get("use_geo_prior", True)),
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            mano_decoder=str(model_cfg.get("mano_decoder", "wilor")),
            freihand_mano_root=model_cfg.get("freihand_mano_root", None),
            focal_length=float(model_cfg.get("focal_length", 5000.0)),
            mano_head_ief_iters=int(model_cfg.get("mano_head", {}).get("ief_iters", 3)),
            mano_head_transformer_input=str(model_cfg.get("mano_head", {}).get("transformer_input", "mean_shape")),
            mano_head_dim=int(model_cfg.get("mano_head", {}).get("dim", 1024)),
            mano_head_depth=int(model_cfg.get("mano_head", {}).get("depth", 6)),
            mano_head_heads=int(model_cfg.get("mano_head", {}).get("heads", 8)),
            mano_head_dim_head=int(model_cfg.get("mano_head", {}).get("dim_head", 64)),
            mano_head_mlp_dim=int(model_cfg.get("mano_head", {}).get("mlp_dim", 2048)),
            mano_head_dropout=float(model_cfg.get("mano_head", {}).get("dropout", 0.0)),
            moge2_num_tokens=moge2_num_tokens,
            moge2_output=str(model_cfg.get("moge2_output", "neck")),
            token_fusion_mode=str(model_cfg.get("token_fusion_mode", "concat")),
            sum_fusion_strategy=str(model_cfg.get("sum_fusion_strategy", "basic")),
            sum_geo_gate_init=float(model_cfg.get("sum_geo_gate_init", 4.0)),
            fusion_proj_zero_init=bool(model_cfg.get("fusion_proj_zero_init", True)),
            cross_attn_num_heads=int(model_cfg.get("cross_attn_num_heads", 8)),
            cross_attn_dropout=float(model_cfg.get("cross_attn_dropout", 0.0)),
            cross_attn_gate_init=float(model_cfg.get("cross_attn_gate_init", 0.0)),
            geo_tokenizer_use_pooling=bool(model_cfg.get("geo_tokenizer_use_pooling", True)),
            use_geo_side_tuning=bool(side_tuning_cfg.get("enabled", False)),
            geo_side_tuning_side_channels=int(side_tuning_cfg.get("side_channels", 256)),
            geo_side_tuning_dropout=float(side_tuning_cfg.get("dropout", 0.1)),
            geo_side_tuning_max_res_scale=float(side_tuning_cfg.get("max_res_scale", 0.1)),
            geo_side_tuning_init_res_scale=float(side_tuning_cfg.get("init_res_scale", 1e-3)),
            use_geo_side_adapter=bool(geo_side_adapter_cfg.get("enabled", False)),
            geo_side_adapter_side_channels=int(geo_side_adapter_cfg.get("side_channels", 256)),
            geo_side_adapter_depth=int(geo_side_adapter_cfg.get("depth", 3)),
            geo_side_adapter_dropout=float(geo_side_adapter_cfg.get("dropout", 0.05)),
            geo_side_adapter_norm_groups=int(geo_side_adapter_cfg.get("norm_groups", 32)),
            geo_branch_dropout_prob=float(model_cfg.get("geo_branch_dropout_prob", 0.0)),
            feature_refiner_method=str(refiner_cfg.get("method", "none")),
            feature_refiner_feat_dim=int(refiner_cfg.get("feat_dim", 1280)),
            feature_refiner_sjta_bottleneck_dim=int(refiner_cfg.get("sjta_bottleneck_dim", 256)),
            feature_refiner_sjta_num_heads=int(refiner_cfg.get("sjta_num_heads", 4)),
            feature_refiner_sjta_use_2d_prior=bool(refiner_cfg.get("sjta_use_2d_prior", True)),
            feature_refiner_sjta_num_steps=int(refiner_cfg.get("sjta_num_steps", 2)),
            feature_refiner_coear_dilation1=int(refiner_cfg.get("coear_dilation1", 1)),
            feature_refiner_coear_dilation2=int(refiner_cfg.get("coear_dilation2", 2)),
            feature_refiner_coear_gate_reduction=int(refiner_cfg.get("coear_gate_reduction", 8)),
            feature_refiner_coear_init_alpha=float(refiner_cfg.get("coear_init_alpha", 0.1)),
            feature_refiner_wilor_msf_bottleneck_ratio=int(refiner_cfg.get("wilor_msf_bottleneck_ratio", 4)),
            feature_refiner_wilor_msf_dilation1=int(refiner_cfg.get("wilor_msf_dilation1", 1)),
            feature_refiner_wilor_msf_dilation2=int(refiner_cfg.get("wilor_msf_dilation2", 2)),
            feature_refiner_wilor_msf_dilation3=int(refiner_cfg.get("wilor_msf_dilation3", 3)),
            feature_refiner_wilor_msf_gate_reduction=int(refiner_cfg.get("wilor_msf_gate_reduction", 8)),
            feature_refiner_wilor_msf_init_alpha=float(refiner_cfg.get("wilor_msf_init_alpha", 0.1)),
            feature_refiner_kcr_num_keypoints=int(refiner_cfg.get("kcr_num_keypoints", 21)),
            feature_refiner_kcr_hidden_dim=int(refiner_cfg.get("kcr_hidden_dim", 128)),
        )
    )


def warmup_lazy_modules(model: GPGFormer, cfg: dict, device: torch.device) -> None:
    model_cfg = cfg.get("model", {})
    h = int(model_cfg.get("image_size", 256))
    w = int(model_cfg.get("image_width", int(h * 0.75)))
    with torch.no_grad():
        img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
        cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
        _ = model(img_dummy, cam_param=cam_dummy)


def _as_t(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _extract_has_mano_mask(batch_data: Dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
    """Per-sample mask indicating whether MANO supervision is available."""
    hm = batch_data.get("has_mano_params", None)
    if not isinstance(hm, dict):
        return torch.ones((batch_size,), device=device, dtype=torch.bool)

    masks = []
    for k in ("global_orient", "hand_pose", "betas"):
        v = hm.get(k, None)
        if v is None:
            continue
        t = _as_t(v, device).reshape(batch_size, -1)
        masks.append(t[:, 0] > 0.5)
    if len(masks) == 0:
        return torch.ones((batch_size,), device=device, dtype=torch.bool)
    m = masks[0]
    for x in masks[1:]:
        m = m & x
    return m


def infer_gt_vertices_from_batch(
    model: GPGFormer,
    batch_data: Dict[str, Any],
    gt_joints_m: torch.Tensor,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Build GT vertices (meters) and FreiHAND-style kp21 (meters) from GT MANO parameters.
    If MANO global translation is available in batch_data, add it (converted to meters).
    """
    mano_params = batch_data.get("mano_params", None)
    if not isinstance(mano_params, dict):
        return None, None

    if not all(k in mano_params for k in ("global_orient", "hand_pose", "betas")):
        return None, None

    go_aa = _as_t(mano_params["global_orient"], gt_joints_m.device)
    hp_aa = _as_t(mano_params["hand_pose"], gt_joints_m.device)
    betas = _as_t(mano_params["betas"], gt_joints_m.device)

    if go_aa.dim() == 1:
        go_aa = go_aa.unsqueeze(0)
    if hp_aa.dim() == 1:
        hp_aa = hp_aa.unsqueeze(0)
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)

    if go_aa.shape[-1] != 3 or hp_aa.shape[-1] != 45:
        return None, None

    B = go_aa.shape[0]
    gt_trans_m = None
    trans_src = batch_data.get("mano_trans", None)
    if trans_src is None and isinstance(mano_params, dict):
        for k in ("transl", "translation", "trans"):
            if k in mano_params:
                trans_src = mano_params[k]
                break
    if trans_src is not None:
        gt_trans_m = _as_t(trans_src, gt_joints_m.device)
        if gt_trans_m.dim() == 1:
            gt_trans_m = gt_trans_m.unsqueeze(0)
        gt_trans_m = gt_trans_m.reshape(B, -1)[:, :3]
        # FreiHAND MANO translation is commonly stored in mm.
        # Heuristic: convert to meters when translation magnitude is too large for meters.
        trans_mag = torch.linalg.norm(gt_trans_m, dim=1)
        trans_med = float(torch.median(trans_mag).item()) if trans_mag.numel() > 0 else 0.0
        if np.isfinite(trans_med) and trans_med > 10.0:
            gt_trans_m = gt_trans_m / 1000.0

    pose_aa = torch.cat([go_aa, hp_aa], dim=-1).reshape(B, 16, 3)
    pose_rm = aa_to_rotmat(pose_aa.reshape(-1, 3)).view(B, 16, 3, 3)
    gt_mano_params = {
        "global_orient": pose_rm[:, [0]],
        "hand_pose": pose_rm[:, 1:],
        "betas": betas,
    }

    model_core = model.module if hasattr(model, "module") else model
    if str(getattr(model_core, "mano_decoder", "wilor")).lower() == "freihand_legacy":
        model_core._init_freihand_mano_layer(device=gt_joints_m.device, dtype=gt_joints_m.dtype)
        pose_aa_flat = torch.cat([go_aa, hp_aa], dim=-1).reshape(B, 48)
        trans_mm = torch.zeros((B, 3), device=gt_joints_m.device, dtype=gt_joints_m.dtype)
        if gt_trans_m is not None:
            trans_mm = gt_trans_m.to(dtype=gt_joints_m.dtype) * 1000.0
        verts_mm, _ = model_core._freihand_mano_layer(pose_aa_flat, betas, trans_mm)
        gt_verts_m = verts_mm / 1000.0
        J_reg = getattr(model_core._freihand_mano_layer, "th_J_regressor", None)
        if J_reg is None:
            return None, None
        gt_kp21_m = model_core._freihand_kp21_from_verts_mm(verts_mm, J_reg) / 1000.0
    else:
        mano_out = model_core.mano(gt_mano_params, pose2rot=False)
        gt_verts_m = mano_out.vertices
        J_reg = getattr(model_core.mano.mano, "J_regressor", None)
        if J_reg is None:
            return None, None
        gt_kp21_m = model_core._kp21_from_verts(gt_verts_m, J_reg)
        if gt_trans_m is not None:
            t = gt_trans_m.to(dtype=gt_verts_m.dtype)
            gt_verts_m = gt_verts_m + t[:, None, :]
            gt_kp21_m = gt_kp21_m + t[:, None, :]
    return gt_verts_m, gt_kp21_m


def infer_kp21_from_vertices(model: GPGFormer, verts_m: torch.Tensor) -> torch.Tensor | None:
    """Regress FreiHAND-style 21 joints (meters) from mesh vertices (meters)."""
    if not isinstance(verts_m, torch.Tensor) or verts_m.ndim != 3 or verts_m.shape[-1] != 3:
        return None
    model_core = model.module if hasattr(model, "module") else model
    if str(getattr(model_core, "mano_decoder", "wilor")).lower() == "freihand_legacy":
        model_core._init_freihand_mano_layer(device=verts_m.device, dtype=verts_m.dtype)
        J_reg = getattr(model_core._freihand_mano_layer, "th_J_regressor", None)
        if J_reg is None:
            return None
        return model_core._freihand_kp21_from_verts_mm(verts_m * 1000.0, J_reg) / 1000.0
    J_reg = getattr(model_core.mano.mano, "J_regressor", None)
    if J_reg is None:
        return None
    return model_core._kp21_from_verts(verts_m, J_reg)


def _is_detector_bbox_failure(exc: Exception) -> bool:
    msg = str(exc)
    return "Detector failed to produce a valid hand bbox for image:" in msg


def _extract_img_path_from_bbox_error(msg: str) -> str:
    token = "Detector failed to produce a valid hand bbox for image:"
    if token not in msg:
        return ""
    return msg.split(token, 1)[1].strip()


class _SafeEvalDataset(torch.utils.data.Dataset):
    """Wrap eval dataset so detector-bbox failures become skippable samples."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        try:
            sample = self.base[idx]
            if isinstance(sample, dict):
                sample = dict(sample)
                sample["_dataset_index"] = int(idx)
            return {"_skip": False, "_data": sample, "_meta": None}
        except Exception as e:
            if not _is_detector_bbox_failure(e):
                raise
            msg = str(e)
            return {
                "_skip": True,
                "_data": None,
                "_meta": {
                    "dataset_index": int(idx),
                    "img_path": _extract_img_path_from_bbox_error(msg),
                    "reason": msg,
                },
            }


def _safe_eval_collate(batch):
    valid_items = []
    skipped_items = []
    for x in batch:
        if bool(x.get("_skip", False)):
            meta = x.get("_meta", None)
            if isinstance(meta, dict):
                skipped_items.append(meta)
        else:
            valid_items.append(x["_data"])
    collated = default_collate(valid_items) if len(valid_items) > 0 else None
    return {"batch_data": collated, "skipped": skipped_items}


def infer_single_json(cfg: dict, model: GPGFormer, device: torch.device):
    dataset = build_dataset(cfg)
    safe_dataset = _SafeEvalDataset(dataset)
    req_workers = int(cfg["train"].get("num_workers", 4))
    num_workers = req_workers
    if str(getattr(dataset, "bbox_source", "")).lower() == "detector" and req_workers != 0:
        print(
            "[info] detector-only crop enabled; forcing DataLoader num_workers=0 "
            f"(requested={req_workers}) to avoid duplicated detector workers."
        )
        num_workers = 0
    dataloader = torch.utils.data.DataLoader(
        safe_dataset,
        sampler=torch.utils.data.SequentialSampler(safe_dataset),
        batch_size=int(cfg["train"].get("val_batch_size", 16)),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_safe_eval_collate,
    )

    pred_joints_m_list = []
    pred_verts_m_list = []
    gt_joints_m_list = []
    gt_verts_m_list = []
    pred_root_m_list = []
    gt_joint_root_m_list = []
    gt_vert_root_m_list = []
    valid_joint_mask_list = []
    valid_vert_mask_list = []
    skipped_samples = []
    skipped_mesh_samples = []
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))
    mesh_fallback_kp_consistency_thr_mm = float(cfg.get("metrics", {}).get("mesh_fallback_kp_consistency_thr_mm", 10.0))

    print(f"Running inference on {len(dataset)} samples...")
    for packed in tqdm(dataloader, desc="Inference"):
        skipped = packed.get("skipped", [])
        if len(skipped) > 0:
            skipped_samples.extend(skipped)
            for s in skipped:
                print(
                    f"[skip] idx={s.get('dataset_index', -1)} "
                    f"img={s.get('img_path', '<unknown>')} "
                    "reason=detector_invalid_bbox"
                )
        batch_data = packed.get("batch_data", None)
        if batch_data is None:
            continue

        img = batch_data["rgb"].to(device)
        cam_param = batch_data.get("cam_param", None)
        cam_param = cam_param.to(device) if cam_param is not None else None

        with torch.no_grad():
            out = model(img, cam_param=cam_param)

        # Keep MANO local frame for 3D metrics; do NOT add pred_cam_t.
        pred_keypoints_m = out["pred_keypoints_3d"]  # (B,21,3), meters
        pred_vertices_m = out["pred_vertices"]  # (B,778,3), meters
        pred_root_m = pred_keypoints_m[:, int(root_index), :]

        gt_joints_m = batch_data["keypoints_3d"].to(device)  # meters
        gt_joint_root_m = gt_joints_m[:, int(root_index), :]

        B = gt_joints_m.shape[0]
        # Joint-valid mask (HO3D evaluation split has degenerate/invalid GT).
        xyz_valid = batch_data.get("xyz_valid", None)
        if xyz_valid is not None:
            valid_joint_mask = _as_t(xyz_valid, device).reshape(B, -1)[:, 0] > 0.5
        else:
            gt_var = gt_joints_m.var(dim=(1, 2))
            finite_mask = torch.isfinite(gt_joints_m).all(dim=(1, 2))
            valid_joint_mask = (gt_var > 1e-8) & finite_mask

        gt_verts_m = batch_data.get("vertices_gt", None)
        used_mano_fallback = gt_verts_m is None
        gt_kp21_from_verts_m = None
        if gt_verts_m is not None:
            gt_verts_m = _as_t(gt_verts_m, device)
            gt_kp21_from_verts_m = infer_kp21_from_vertices(model, gt_verts_m)
        else:
            gt_verts_m, gt_kp21_from_verts_m = infer_gt_vertices_from_batch(model, batch_data, gt_joints_m)

        if gt_kp21_from_verts_m is not None:
            gt_vert_root_m = gt_kp21_from_verts_m[:, int(root_index), :]
        else:
            gt_vert_root_m = gt_joint_root_m

        has_mano_mask = _extract_has_mano_mask(batch_data, B, device)
        if gt_verts_m is None:
            gt_verts_m = torch.zeros_like(pred_vertices_m)
            valid_vert_mask = torch.zeros((B,), device=device, dtype=torch.bool)
        else:
            # Explicit mesh GT (e.g. HO3D evaluation) should stay valid even when
            # the dataset intentionally masks out MANO parameters.
            valid_vert_mask = valid_joint_mask if not used_mano_fallback else (valid_joint_mask & has_mano_mask)
            finite_vert = torch.isfinite(gt_verts_m).all(dim=(1, 2))
            valid_vert_mask = valid_vert_mask & finite_vert

        # If mesh GT is reconstructed from MANO params, verify it is consistent with GT joints.
        # FreiHAND MANO params can be inconsistent with xyz/verts in some dumps; drop such samples.
        if used_mano_fallback and gt_kp21_from_verts_m is not None and gt_kp21_from_verts_m.shape[:2] == gt_joints_m.shape[:2]:
            gt_kp21_rr = gt_kp21_from_verts_m - gt_kp21_from_verts_m[:, int(root_index) : int(root_index) + 1, :]
            gt_joints_rr = gt_joints_m - gt_joint_root_m[:, None, :]
            kp_consistency_mm = torch.norm(gt_kp21_rr - gt_joints_rr, dim=-1).mean(dim=1) * 1000.0
            consistent_mask = kp_consistency_mm <= mesh_fallback_kp_consistency_thr_mm

            invalid_mesh_mask = valid_joint_mask & (~consistent_mask)
            if bool(invalid_mesh_mask.any()):
                ds_idx = batch_data.get("_dataset_index", None)
                if ds_idx is not None:
                    ds_idx = _as_t(ds_idx, device).reshape(-1).long()
                    bad_ids = torch.nonzero(invalid_mesh_mask, as_tuple=False).reshape(-1)
                    for bi in bad_ids.tolist():
                        skipped_mesh_samples.append(
                            {
                                "dataset_index": int(ds_idx[bi].item()),
                                "img_path": "",
                                "reason": (
                                    "mesh_gt_from_mano_inconsistent_with_joints: "
                                    f"{float(kp_consistency_mm[bi].item()):.3f}mm > "
                                    f"{mesh_fallback_kp_consistency_thr_mm:.3f}mm"
                                ),
                            }
                        )
            filtered_mask = valid_vert_mask & consistent_mask
            # Avoid producing all-NaN mesh metrics when threshold is too strict for this config.
            # In that case keep finite MANO fallback meshes and continue evaluation.
            if bool(valid_vert_mask.any()) and (not bool(filtered_mask.any())):
                print(
                    "[warn] mesh_fallback_kp_consistency_thr_mm filtered all mesh samples; "
                    "disabling this filter for current batch to avoid NaN mesh metrics."
                )
            else:
                valid_vert_mask = filtered_mask

        finite_roots = (
            torch.isfinite(pred_root_m).all(dim=1)
            & torch.isfinite(gt_joint_root_m).all(dim=1)
            & torch.isfinite(gt_vert_root_m).all(dim=1)
        )
        valid_joint_mask = valid_joint_mask & finite_roots
        valid_vert_mask = valid_vert_mask & finite_roots

        # Keep joint metrics aligned with training-time validation (train.py):
        # always use dataset keypoints_3d as GT for MPJPE/PA-MPJPE.
        gt_joints_for_metrics_m = gt_joints_m
        gt_joint_root_for_metrics_m = gt_joint_root_m
        valid_joint_mask_for_metrics = valid_joint_mask

        pred_joints_m_list.append(pred_keypoints_m.cpu().numpy())
        pred_verts_m_list.append(pred_vertices_m.cpu().numpy())
        gt_joints_m_list.append(gt_joints_for_metrics_m.cpu().numpy())
        gt_verts_m_list.append(gt_verts_m.cpu().numpy())
        pred_root_m_list.append(pred_root_m.cpu().numpy())
        gt_joint_root_m_list.append(gt_joint_root_for_metrics_m.cpu().numpy())
        gt_vert_root_m_list.append(gt_vert_root_m.cpu().numpy())
        valid_joint_mask_list.append(valid_joint_mask_for_metrics.cpu().numpy())
        valid_vert_mask_list.append(valid_vert_mask.cpu().numpy())

    if len(pred_joints_m_list) > 0:
        pred_joints_m = np.concatenate(pred_joints_m_list, axis=0)
        pred_verts_m = np.concatenate(pred_verts_m_list, axis=0)
        gt_joints_m = np.concatenate(gt_joints_m_list, axis=0)
        gt_verts_m = np.concatenate(gt_verts_m_list, axis=0)
        pred_root_m = np.concatenate(pred_root_m_list, axis=0)
        gt_joint_root_m = np.concatenate(gt_joint_root_m_list, axis=0)
        gt_vert_root_m = np.concatenate(gt_vert_root_m_list, axis=0)
        valid_joint_mask = np.concatenate(valid_joint_mask_list, axis=0).astype(bool)
        valid_vert_mask = np.concatenate(valid_vert_mask_list, axis=0).astype(bool)
    else:
        pred_joints_m = np.zeros((0, 21, 3), dtype=np.float32)
        pred_verts_m = np.zeros((0, 778, 3), dtype=np.float32)
        gt_joints_m = np.zeros((0, 21, 3), dtype=np.float32)
        gt_verts_m = np.zeros((0, 778, 3), dtype=np.float32)
        pred_root_m = np.zeros((0, 3), dtype=np.float32)
        gt_joint_root_m = np.zeros((0, 3), dtype=np.float32)
        gt_vert_root_m = np.zeros((0, 3), dtype=np.float32)
        valid_joint_mask = np.zeros((0,), dtype=bool)
        valid_vert_mask = np.zeros((0,), dtype=bool)

    return (
        pred_joints_m,
        pred_verts_m,
        gt_joints_m,
        gt_verts_m,
        pred_root_m,
        gt_joint_root_m,
        gt_vert_root_m,
        valid_joint_mask,
        valid_vert_mask,
        skipped_samples,
        skipped_mesh_samples,
        len(dataset),
    )


def evaluate_metrics(
    pred_joints_m: np.ndarray,
    pred_verts_m: np.ndarray,
    gt_joints_m: np.ndarray,
    gt_verts_m: np.ndarray,
    pred_root_m: np.ndarray,
    gt_joint_root_m: np.ndarray,
    gt_vert_root_m: np.ndarray,
    valid_joint_mask: np.ndarray,
    valid_vert_mask: np.ndarray,
    auc_max_mm: float = 50.0,
    auc_steps: int = 100,
) -> Dict[str, float]:
    valid_joint_mask = np.asarray(valid_joint_mask, dtype=bool).reshape(-1)
    valid_vert_mask = np.asarray(valid_vert_mask, dtype=bool).reshape(-1)

    # Joint metrics
    if valid_joint_mask.any():
        pred_joints_sel = pred_joints_m[valid_joint_mask]
        gt_joints_sel = gt_joints_m[valid_joint_mask]
        pred_joint_root_sel = pred_root_m[valid_joint_mask]
        gt_joint_root_sel = gt_joint_root_m[valid_joint_mask]

        pred_joints_rr = pred_joints_sel - pred_joint_root_sel[:, None, :]
        gt_joints_rr = gt_joints_sel - gt_joint_root_sel[:, None, :]

        joint_err_mm = np.linalg.norm(pred_joints_rr - gt_joints_rr, axis=-1) * 1000.0
        mpjpe_mm = float(joint_err_mm.mean())

        pred_joints_pa_m = procrustes_align_batch(pred_joints_rr, gt_joints_rr)
        joint_err_pa_mm = np.linalg.norm(pred_joints_pa_m - gt_joints_rr, axis=-1) * 1000.0
        pa_mpjpe_mm = float(joint_err_pa_mm.mean())

        auc_j, _, _ = compute_auc(joint_err_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
        auc_j_pa, _, _ = compute_auc(joint_err_pa_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
    else:
        mpjpe_mm = float("nan")
        pa_mpjpe_mm = float("nan")
        auc_j = float("nan")
        auc_j_pa = float("nan")

    # Mesh metrics
    mpvpe_mm = float("nan")
    pa_mpvpe_mm = float("nan")
    f5 = float("nan")
    f15 = float("nan")
    f5_pa = float("nan")
    f15_pa = float("nan")
    auc_v = float("nan")
    auc_v_pa = float("nan")

    if valid_vert_mask.any():
        pred_verts_sel = pred_verts_m[valid_vert_mask]
        gt_verts_sel = gt_verts_m[valid_vert_mask]
        pred_vert_root_sel = pred_root_m[valid_vert_mask]
        gt_vert_root_sel = gt_vert_root_m[valid_vert_mask]

        pred_verts_rr = pred_verts_sel - pred_vert_root_sel[:, None, :]
        gt_verts_rr = gt_verts_sel - gt_vert_root_sel[:, None, :]

        vert_err_mm = np.linalg.norm(pred_verts_rr - gt_verts_rr, axis=-1) * 1000.0
        mpvpe_mm = float(vert_err_mm.mean())

        pred_verts_pa_m = procrustes_align_batch(pred_verts_rr, gt_verts_rr)
        vert_err_pa_mm = np.linalg.norm(pred_verts_pa_m - gt_verts_rr, axis=-1) * 1000.0
        pa_mpvpe_mm = float(vert_err_pa_mm.mean())

        auc_v, _, _ = compute_auc(vert_err_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
        auc_v_pa, _, _ = compute_auc(vert_err_pa_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)

        f5_list, f15_list, f5_pa_list, f15_pa_list = [], [], [], []
        for i in range(gt_verts_rr.shape[0]):
            f, _, _ = calculate_fscore(gt_verts_rr[i], pred_verts_rr[i], 0.005)
            f5_list.append(f)
            f, _, _ = calculate_fscore(gt_verts_rr[i], pred_verts_rr[i], 0.015)
            f15_list.append(f)

            f, _, _ = calculate_fscore(gt_verts_rr[i], pred_verts_pa_m[i], 0.005)
            f5_pa_list.append(f)
            f, _, _ = calculate_fscore(gt_verts_rr[i], pred_verts_pa_m[i], 0.015)
            f15_pa_list.append(f)

        f5 = float(np.mean(f5_list))
        f15 = float(np.mean(f15_list))
        f5_pa = float(np.mean(f5_pa_list))
        f15_pa = float(np.mean(f15_pa_list))

    return {
        "MPJPE_mm": mpjpe_mm,
        "PA-MPJPE_mm": pa_mpjpe_mm,
        "MPVPE_mm": mpvpe_mm,
        "PA-MPVPE_mm": pa_mpvpe_mm,
        "F-score@5mm": f5,
        "F-score@15mm": f15,
        "AUC-J": auc_j,
        "AUC-V": auc_v,
        # Extra aligned references (kept for analysis/debug).
        "PA-F-score@5mm": f5_pa,
        "PA-F-score@15mm": f15_pa,
        "PA-AUC-J": auc_j_pa,
        "PA-AUC-V": auc_v_pa,
    }


def main():
    parser = argparse.ArgumentParser(description="GPGFormer inference and metric evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON for metrics")
    parser.add_argument("--auc-max-mm", type=float, default=50.0, help="AUC threshold upper bound in mm")
    parser.add_argument("--auc-steps", type=int, default=100, help="AUC threshold steps")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["__config_path__"] = str(args.config)
    print(f"Loaded config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model_from_cfg(cfg).to(device)
    warmup_lazy_modules(model, cfg, device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    (
        pred_joints_m,
        pred_verts_m,
        gt_joints_m,
        gt_verts_m,
        pred_root_m,
        gt_joint_root_m,
        gt_vert_root_m,
        valid_joint_mask,
        valid_vert_mask,
        skipped_samples,
        skipped_mesh_samples,
        dataset_num_samples,
    ) = infer_single_json(
        cfg, model, device
    )

    print(f"\nCollected {pred_joints_m.shape[0]} samples")
    print(f"dataset samples: {int(dataset_num_samples)}")
    print(f"skipped samples: {len(skipped_samples)}")
    print(f"mesh-invalid samples: {len(skipped_mesh_samples)}")
    print(f"pred_joints: {pred_joints_m.shape}, gt_joints: {gt_joints_m.shape}")
    print(f"pred_verts:  {pred_verts_m.shape}, gt_verts:  {gt_verts_m.shape}")
    print(f"valid joints: {int(valid_joint_mask.sum())}/{valid_joint_mask.shape[0]}")
    print(f"valid verts:  {int(valid_vert_mask.sum())}/{valid_vert_mask.shape[0]}")

    metrics = evaluate_metrics(
        pred_joints_m=pred_joints_m,
        pred_verts_m=pred_verts_m,
        gt_joints_m=gt_joints_m,
        gt_verts_m=gt_verts_m,
        pred_root_m=pred_root_m,
        gt_joint_root_m=gt_joint_root_m,
        gt_vert_root_m=gt_vert_root_m,
        valid_joint_mask=valid_joint_mask,
        valid_vert_mask=valid_vert_mask,
        auc_max_mm=float(args.auc_max_mm),
        auc_steps=int(args.auc_steps),
    )

    print("\n" + "=" * 72)
    print("Requested Metrics")
    print("=" * 72)
    print(f"MPJPE:      {metrics['MPJPE_mm']:.3f} mm")
    print(f"PA-MPJPE:   {metrics['PA-MPJPE_mm']:.3f} mm")
    print(f"MPVPE:      {metrics['MPVPE_mm']:.3f} mm")
    print(f"PA-MPVPE:   {metrics['PA-MPVPE_mm']:.3f} mm")
    print(f"F-score@5:  {metrics['F-score@5mm']:.6f}")
    print(f"F-score@15: {metrics['F-score@15mm']:.6f}")
    print(f"AUC-J:      {metrics['AUC-J']:.6f}")
    print(f"AUC-V:      {metrics['AUC-V']:.6f}")

    print("\nExtra (PA-aligned reference)")
    print(f"PA-F@5:     {metrics['PA-F-score@5mm']:.6f}")
    print(f"PA-F@15:    {metrics['PA-F-score@15mm']:.6f}")
    print(f"PA-AUC-J:   {metrics['PA-AUC-J']:.6f}")
    print(f"PA-AUC-V:   {metrics['PA-AUC-V']:.6f}")
    print("=" * 72)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {
            "config": str(args.config),
            "ckpt": str(args.ckpt),
            "num_dataset_samples": int(dataset_num_samples),
            "num_samples": int(pred_joints_m.shape[0]),
            "num_skipped_samples": int(len(skipped_samples)),
            "num_mesh_invalid_samples": int(len(skipped_mesh_samples)),
            "num_valid_joint_samples": int(valid_joint_mask.sum()),
            "num_valid_vertex_samples": int(valid_vert_mask.sum()),
            "auc_max_mm": float(args.auc_max_mm),
            "auc_steps": int(args.auc_steps),
            "mesh_fallback_kp_consistency_thr_mm": float(
                cfg.get("metrics", {}).get("mesh_fallback_kp_consistency_thr_mm", 10.0)
            ),
            "metrics": metrics,
            "skipped_samples": skipped_samples,
            "mesh_invalid_samples": skipped_mesh_samples,
        }
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved metrics JSON to: {out_path}")


if __name__ == "__main__":
    main()

"""
GPGFormer inference + evaluation (HandOS-style metrics).

This script matches `infer_to_json.py` CLI and output format, but computes
metrics using HandOS conventions:
- MPJPE / MPVPE: mean L2 error without explicit root-relative subtraction.
- PA-MPJPE / PA-MPVPE: similarity Procrustes alignment (scale+R+t) then mean L2.

Notes:
- We reuse the same dataset/model/inference pipeline from `infer_to_json.py`.
- This is useful for apples-to-apples comparisons with HandOS' `SimpleMPJPE`
  (`p-mpjpe`, `p-mpvpe`) and `XyzFScore(align_mode='procrustes')`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

# Reuse the full inference pipeline and helpers from the canonical script.
import infer_to_json as _base


def evaluate_metrics_handos_style(
    pred_joints_m: np.ndarray,
    pred_verts_m: np.ndarray,
    gt_joints_m: np.ndarray,
    gt_verts_m: np.ndarray,
    valid_joint_mask: np.ndarray,
    valid_vert_mask: np.ndarray,
    root_index: int = 9,
    auc_max_mm: float = 50.0,
    auc_steps: int = 100,
) -> Dict[str, float]:
    """Compute metrics in HandOS style, plus root-relative subtraction.

    HandOS' metric implementations typically do not explicitly apply a dataset
    root subtraction (they can rely on datasets where root=0 / wrist is used as
    the origin). For FreiHAND under GPGFormer, we want to match the training/eval
    convention where root_index=9 (middle MCP) is the canonical root for all
    root-relative operations.
    """
    valid_joint_mask = np.asarray(valid_joint_mask, dtype=bool).reshape(-1)
    valid_vert_mask = np.asarray(valid_vert_mask, dtype=bool).reshape(-1)

    root_index = int(root_index)

    # Joint metrics
    if valid_joint_mask.any():
        pred_j_sel = np.asarray(pred_joints_m[valid_joint_mask], dtype=np.float32)
        gt_j_sel = np.asarray(gt_joints_m[valid_joint_mask], dtype=np.float32)

        # Root-relative (root_index=9)
        pred_root = pred_j_sel[:, root_index : root_index + 1, :]
        gt_root = gt_j_sel[:, root_index : root_index + 1, :]
        pred_j_rr = pred_j_sel - pred_root
        gt_j_rr = gt_j_sel - gt_root

        j_err_mm = np.linalg.norm(pred_j_rr - gt_j_rr, axis=-1) * 1000.0
        mpjpe_mm = float(j_err_mm.mean())

        pred_j_pa = _base.procrustes_align_batch(pred_j_rr, gt_j_rr)
        j_err_pa_mm = np.linalg.norm(pred_j_pa - gt_j_rr, axis=-1) * 1000.0
        pa_mpjpe_mm = float(j_err_pa_mm.mean())

        auc_j, _, _ = _base.compute_auc(j_err_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
        auc_j_pa, _, _ = _base.compute_auc(j_err_pa_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
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
        pred_v_sel = np.asarray(pred_verts_m[valid_vert_mask], dtype=np.float32)
        gt_v_sel = np.asarray(gt_verts_m[valid_vert_mask], dtype=np.float32)

        # Root-relative for verts: subtract the same root joint (index=9) from joints,
        # consistent with GPGFormer convention and avoids relying on a specific joint regressor.
        # Use the joint roots from the corresponding samples.
        pred_j_sel_for_verts = np.asarray(pred_joints_m[valid_vert_mask], dtype=np.float32)
        gt_j_sel_for_verts = np.asarray(gt_joints_m[valid_vert_mask], dtype=np.float32)
        pred_root_v = pred_j_sel_for_verts[:, root_index : root_index + 1, :]
        gt_root_v = gt_j_sel_for_verts[:, root_index : root_index + 1, :]

        pred_v_rr = pred_v_sel - pred_root_v
        gt_v_rr = gt_v_sel - gt_root_v

        v_err_mm = np.linalg.norm(pred_v_rr - gt_v_rr, axis=-1) * 1000.0
        mpvpe_mm = float(v_err_mm.mean())

        pred_v_pa = _base.procrustes_align_batch(pred_v_rr, gt_v_rr)
        v_err_pa_mm = np.linalg.norm(pred_v_pa - gt_v_rr, axis=-1) * 1000.0
        pa_mpvpe_mm = float(v_err_pa_mm.mean())

        auc_v, _, _ = _base.compute_auc(v_err_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)
        auc_v_pa, _, _ = _base.compute_auc(v_err_pa_mm.reshape(-1), val_min_mm=0.0, val_max_mm=auc_max_mm, steps=auc_steps)

        f5_list, f15_list, f5_pa_list, f15_pa_list = [], [], [], []
        for i in range(gt_v_rr.shape[0]):
            f, _, _ = _base.calculate_fscore(gt_v_rr[i], pred_v_rr[i], 0.005)
            f5_list.append(f)
            f, _, _ = _base.calculate_fscore(gt_v_rr[i], pred_v_rr[i], 0.015)
            f15_list.append(f)

            f, _, _ = _base.calculate_fscore(gt_v_rr[i], pred_v_pa[i], 0.005)
            f5_pa_list.append(f)
            f, _, _ = _base.calculate_fscore(gt_v_rr[i], pred_v_pa[i], 0.015)
            f15_pa_list.append(f)

        f5 = float(np.mean(f5_list)) if len(f5_list) else float("nan")
        f15 = float(np.mean(f15_list)) if len(f15_list) else float("nan")
        f5_pa = float(np.mean(f5_pa_list)) if len(f5_pa_list) else float("nan")
        f15_pa = float(np.mean(f15_pa_list)) if len(f15_pa_list) else float("nan")

    return {
        "MPJPE_mm": mpjpe_mm,
        "PA-MPJPE_mm": pa_mpjpe_mm,
        "MPVPE_mm": mpvpe_mm,
        "PA-MPVPE_mm": pa_mpvpe_mm,
        "F-score@5mm": f5,
        "F-score@15mm": f15,
        "AUC-J": auc_j,
        "AUC-V": auc_v,
        "PA-F-score@5mm": f5_pa,
        "PA-F-score@15mm": f15_pa,
        "PA-AUC-J": auc_j_pa,
        "PA-AUC-V": auc_v_pa,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPGFormer inference and metric evaluation (HandOS-style)")
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

    model = _base.build_model_from_cfg(cfg).to(device)
    _base.warmup_lazy_modules(model, cfg, device)

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
        _pred_root_m,
        _gt_joint_root_m,
        _gt_vert_root_m,
        valid_joint_mask,
        valid_vert_mask,
        skipped_samples,
        skipped_mesh_samples,
        dataset_num_samples,
    ) = _base.infer_single_json(cfg, model, device)

    print(f"\nCollected {pred_joints_m.shape[0]} samples")
    print(f"dataset samples: {int(dataset_num_samples)}")
    print(f"skipped samples: {len(skipped_samples)}")
    print(f"mesh-invalid samples: {len(skipped_mesh_samples)}")
    print(f"pred_joints: {pred_joints_m.shape}, gt_joints: {gt_joints_m.shape}")
    print(f"pred_verts:  {pred_verts_m.shape}, gt_verts:  {gt_verts_m.shape}")
    print(f"valid joints: {int(np.asarray(valid_joint_mask, dtype=bool).sum())}/{np.asarray(valid_joint_mask).shape[0]}")
    print(f"valid verts:  {int(np.asarray(valid_vert_mask, dtype=bool).sum())}/{np.asarray(valid_vert_mask).shape[0]}")

    metrics = evaluate_metrics_handos_style(
        pred_joints_m=pred_joints_m,
        pred_verts_m=pred_verts_m,
        gt_joints_m=gt_joints_m,
        gt_verts_m=gt_verts_m,
        valid_joint_mask=valid_joint_mask,
        valid_vert_mask=valid_vert_mask,
        root_index=9,
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
            "num_valid_joint_samples": int(np.asarray(valid_joint_mask, dtype=bool).sum()),
            "num_valid_vertex_samples": int(np.asarray(valid_vert_mask, dtype=bool).sum()),
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

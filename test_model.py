"""
Minimal test script for GPGFormer evaluation.
Uses same mesh calculation as training for MPVPE/PA-MPVPE consistency.
"""

import argparse
import torch
import yaml
from pathlib import Path

from infer_to_json import (
    build_dataset,
    build_model_from_cfg,
    warmup_lazy_modules,
    infer_single_json,
    evaluate_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Test GPGFormer model")
    parser.add_argument(
        "--config",
        type=str,
        default="/root/code/hand_reconstruction/GPGFormer/configs/config_freihand_multimodal_mask_consistency_recommended.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/root/code/vepfs/GPGFormer/checkpoints/freihand_multimodal_mask_consistency_recommended_20260311/freihand/gpgformer_best.pt",
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["__config_path__"] = str(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")

    model = build_model_from_cfg(cfg).to(device)
    warmup_lazy_modules(model, cfg, device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

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
    ) = infer_single_json(cfg, model, device)

    print(f"\nSamples: {pred_joints_m.shape[0]}/{dataset_num_samples}")
    print(f"Valid joints: {valid_joint_mask.sum()}")
    print(f"Valid verts: {valid_vert_mask.sum()}")

    metrics = evaluate_metrics(
        pred_joints_m,
        pred_verts_m,
        gt_joints_m,
        gt_verts_m,
        pred_root_m,
        gt_joint_root_m,
        gt_vert_root_m,
        valid_joint_mask,
        valid_vert_mask,
    )

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"MPJPE:      {metrics['MPJPE_mm']:.3f} mm")
    print(f"PA-MPJPE:   {metrics['PA-MPJPE_mm']:.3f} mm")
    print(f"MPVPE:      {metrics['MPVPE_mm']:.3f} mm")
    print(f"PA-MPVPE:   {metrics['PA-MPVPE_mm']:.3f} mm")
    print(f"F@5mm:      {metrics['F-score@5mm']:.6f}")
    print(f"F@15mm:     {metrics['F-score@15mm']:.6f}")
    print(f"AUC-J:      {metrics['AUC-J']:.6f}")
    print(f"AUC-V:      {metrics['AUC-V']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

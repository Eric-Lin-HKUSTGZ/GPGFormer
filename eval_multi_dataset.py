from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from infer_to_json import (
    build_model_from_cfg,
    evaluate_metrics,
    infer_single_json,
    warmup_lazy_modules,
)


DEFAULT_EVAL_DATASETS: dict[str, dict[str, Any]] = {
    "ho3d": {
        "name": "ho3d",
        "ho3d_version": "v3",
        "bbox_source_eval": "gt",
    },
    "dexycb": {
        "name": "dexycb",
        "dexycb_setup": "s0",
        "bbox_source_eval": "detector",
    },
    "freihand": {
        "name": "freihand",
        "bbox_source_eval": "detector",
        "use_trainval_split": False,
    },
}

SUMMARY_METRIC_KEYS = [
    "MPJPE_mm",
    "PA-MPJPE_mm",
    "MPVPE_mm",
    "PA-MPVPE_mm",
    "F-score@5mm",
    "F-score@15mm",
    "AUC-J",
    "AUC-V",
]


def _normalize_dataset_name(name: str) -> str:
    norm = str(name).strip().lower()
    if norm in ("dex-ycb",):
        return "dexycb"
    return norm


def _default_output_dir(ckpt_path: Path) -> Path:
    if len(ckpt_path.parents) >= 2:
        return ckpt_path.parent.parent / "multi_eval"
    return ckpt_path.parent / "multi_eval"


def _resolve_eval_entries(cfg: dict, datasets_arg: str | None) -> list[dict[str, Any]]:
    cfg_entries = cfg.get("dataset", {}).get("eval_datasets", [])
    by_name = {}
    for entry in cfg_entries:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        by_name[_normalize_dataset_name(entry["name"])] = entry

    if datasets_arg:
        requested = [_normalize_dataset_name(x) for x in datasets_arg.split(",") if x.strip()]
    elif by_name:
        requested = list(by_name.keys())
    else:
        requested = ["ho3d", "dexycb", "freihand"]

    resolved = []
    for name in requested:
        if name not in DEFAULT_EVAL_DATASETS:
            raise ValueError(f"Unsupported dataset for multi-eval: {name}")
        entry = copy.deepcopy(DEFAULT_EVAL_DATASETS[name])
        if name in by_name:
            entry.update(copy.deepcopy(by_name[name]))
        resolved.append(entry)
    return resolved


def _build_eval_cfg(base_cfg: dict, entry: dict[str, Any], config_path: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_cfg["name"] = entry["name"]
    for key, value in entry.items():
        if key == "name":
            continue
        dataset_cfg[key] = value
    cfg["__config_path__"] = f"{config_path}#eval:{entry['name']}"
    return cfg


def _safe_domain_average(results: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in SUMMARY_METRIC_KEYS:
        values = [float(item["metrics"].get(key, float("nan"))) for item in results]
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary[key] = float(finite.mean()) if finite.size > 0 else float("nan")
    return summary


def _to_serializable_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | None]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            out[key] = None if not np.isfinite(value) else float(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate one GPGFormer checkpoint on multiple datasets.")
    parser.add_argument("--config", type=str, required=True, help="Base mixed-training config")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets to evaluate. Default: config.dataset.eval_datasets or ho3d,dexycb,freihand",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save per-dataset JSON summaries")
    parser.add_argument("--auc-max-mm", type=float, default=50.0, help="AUC upper bound in mm")
    parser.add_argument("--auc-steps", type=int, default=100, help="AUC threshold steps")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(config_path.read_text())
    base_cfg["__config_path__"] = str(config_path)
    eval_entries = _resolve_eval_entries(base_cfg, args.datasets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded config: {config_path}")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Using device: {device}")
    print(f"Evaluating datasets: {[entry['name'] for entry in eval_entries]}")

    model = build_model_from_cfg(base_cfg).to(device)
    warmup_lazy_modules(model, base_cfg, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    results: list[dict[str, Any]] = []
    for entry in eval_entries:
        dataset_name = _normalize_dataset_name(entry["name"])
        eval_cfg = _build_eval_cfg(base_cfg, entry, config_path)

        print("\n" + "=" * 72)
        print(f"Evaluating dataset: {dataset_name}")
        print("=" * 72)

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
        ) = infer_single_json(eval_cfg, model, device)

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

        result = {
            "dataset": dataset_name,
            "config": str(config_path),
            "ckpt": str(ckpt_path),
            "num_dataset_samples": int(dataset_num_samples),
            "num_samples": int(pred_joints_m.shape[0]),
            "num_skipped_samples": int(len(skipped_samples)),
            "num_mesh_invalid_samples": int(len(skipped_mesh_samples)),
            "num_valid_joint_samples": int(valid_joint_mask.sum()),
            "num_valid_vertex_samples": int(valid_vert_mask.sum()),
            "metrics": _to_serializable_metrics(metrics),
            "skipped_samples": skipped_samples,
            "mesh_invalid_samples": skipped_mesh_samples,
        }
        results.append(result)

        print(
            f"[{dataset_name}] PA-MPJPE={metrics['PA-MPJPE_mm']:.3f} mm  "
            f"PA-MPVPE={metrics['PA-MPVPE_mm']:.3f} mm  "
            f"AUC-J={metrics['AUC-J']:.6f}  AUC-V={metrics['AUC-V']:.6f}"
        )

        dataset_out = output_dir / f"{dataset_name}_metrics.json"
        dataset_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {dataset_out}")

    domain_average = _safe_domain_average(results)
    summary = {
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "datasets": [item["dataset"] for item in results],
        "domain_average": _to_serializable_metrics(domain_average),
        "per_dataset": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Cross-dataset summary (domain average)")
    print("=" * 72)
    for key in SUMMARY_METRIC_KEYS:
        value = domain_average[key]
        if np.isfinite(value):
            print(f"{key}: {value:.6f}" if abs(value) < 10 else f"{key}: {value:.3f}")
        else:
            print(f"{key}: nan")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

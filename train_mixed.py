"""Mixed-dataset training script for GPGFormer.

Builds a ConcatDataset from multiple hand datasets with weighted sampling,
then delegates to the shared train_loop() from train.py.

Usage:
    python train_mixed.py --config configs/config_mixed_freihand.yaml
    torchrun --nproc_per_node=4 train_mixed.py --config configs/config_mixed_freihand.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from data.mixed_sampler import DistributedWeightedSampler
from train import build_dataset, train_loop
from gpgformer.utils.distributed import (
    DistInfo,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    seed_everything,
)


def mixed_collate_fn(batch):
    """Collate that handles dicts with inconsistent keys across datasets.

    ConcatDataset may mix samples from datasets returning different dict keys
    (e.g. HO3D has mano_pose/mano_shape/mano_trans/has_mano_params that others lack).
    Only collate keys present in ALL samples; the training loop handles missing keys.
    """
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys &= set(sample.keys())
    filtered = [{k: s[k] for k in common_keys} for s in batch]
    return default_collate(filtered)


def build_mixed_train_dataset(cfg):
    """Build a ConcatDataset from cfg['dataset']['mixed_datasets'].

    Returns (ConcatDataset, list[float]) — the concat dataset and per-sub-dataset weights.
    """
    mixed_cfgs = cfg["dataset"]["mixed_datasets"]
    datasets = []
    weights = []
    for entry in mixed_cfgs:
        # Build a temporary cfg that looks like a single-dataset config
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["dataset"]["name"] = entry["name"]
        # Copy dataset-specific keys from the entry
        for k, v in entry.items():
            if k not in ("name", "weight"):
                sub_cfg["dataset"][k] = v
        ds = build_dataset(sub_cfg, "train")
        datasets.append(ds)
        weights.append(float(entry.get("weight", 1.0)))
        if is_main_process():
            print(f"  [mixed] {entry['name']}: {len(ds)} samples, weight={weights[-1]}")
    concat_ds = ConcatDataset(datasets)
    if is_main_process():
        print(f"  [mixed] total: {len(concat_ds)} samples")
    return concat_ds, weights


def build_eval_cfg_from_entry(cfg: dict, entry: dict) -> dict:
    sub_cfg = copy.deepcopy(cfg)
    sub_cfg["dataset"]["name"] = entry["name"]
    for k, v in entry.items():
        if k != "name":
            sub_cfg["dataset"][k] = v
    return sub_cfg


def build_mixed_eval_loaders(cfg: dict, distributed: bool, dist_info: DistInfo) -> Dict[str, DataLoader]:
    eval_entries = cfg.get("dataset", {}).get("eval_datasets", [])
    if not eval_entries:
        if is_main_process():
            print("[progress] Building val dataset ...", flush=True)
        val_ds = build_dataset(cfg, "val")
        val_sampler = None
        if distributed:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=dist_info.world_size,
                rank=dist_info.rank,
                shuffle=False,
                drop_last=False,
            )
        return {
            str(cfg.get("dataset", {}).get("name", "val")): DataLoader(
                val_ds,
                batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=True,
            )
        }

    loaders: Dict[str, DataLoader] = {}
    for idx, entry in enumerate(eval_entries):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"dataset.eval_datasets[{idx}] must be a dict with a 'name' field, got: {entry!r}")
        sub_cfg = build_eval_cfg_from_entry(cfg, entry)
        ds_name = str(entry["name"]).lower()
        if is_main_process():
            print(f"[progress] Building val dataset [{ds_name}] ...", flush=True)
        val_ds = build_dataset(sub_cfg, "val")
        val_sampler = None
        if distributed:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=dist_info.world_size,
                rank=dist_info.rank,
                shuffle=False,
                drop_last=False,
            )
        loaders[ds_name] = DataLoader(
            val_ds,
            batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
        )
    return loaders


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # ---- distributed setup ----
    dist_info: DistInfo = setup_distributed(backend=args.backend)
    distributed = dist_info.distributed

    if (not distributed) and (not bool(cfg["train"].get("use_cuda", True))):
        device = torch.device("cpu")
    else:
        device = dist_info.device if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(int(cfg["train"].get("seed", 42)), rank=(dist_info.rank if distributed else 0))

    try:
        # ---- Mixed training dataset ----
        if is_main_process():
            print("[progress] Building mixed train dataset ...", flush=True)
        train_ds, ds_weights = build_mixed_train_dataset(cfg)

        # Weighted sampler (supports DDP)
        train_sampler = DistributedWeightedSampler(
            train_ds,
            weights=ds_weights,
            num_replicas=dist_info.world_size if distributed else 1,
            rank=dist_info.rank if distributed else 0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,  # sampler handles shuffling
            sampler=train_sampler,
            num_workers=int(cfg["train"].get("num_workers", 4)),
            pin_memory=True,
            drop_last=True,
            collate_fn=mixed_collate_fn,
        )

        # ---- Validation / test protocol ----
        # If dataset.eval_datasets is configured, evaluate on each listed domain and let train_loop
        # aggregate the metrics sample-wise for checkpoint selection.
        val_loader = build_mixed_eval_loaders(cfg, distributed=distributed, dist_info=dist_info)
        if len(val_loader) == 1:
            val_loader = next(iter(val_loader.values()))

        # Delegate to shared training loop
        train_loop(cfg, train_loader, val_loader, train_sampler, dist_info, distributed, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

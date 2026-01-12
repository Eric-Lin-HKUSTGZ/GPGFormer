from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

from gpgformer.models import GPGFormer, GPGFormerConfig
from gpgformer.losses import UTNetLoss
from gpgformer.metrics.pose_metrics import compute_pa_mpjpe
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat
from gpgformer.utils.distributed import (
    DistInfo,
    all_reduce_sum,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    seed_everything,
)


def build_dataset(cfg: dict, split: str):
    name = cfg["dataset"]["name"].lower()
    align_wilor_aug = bool(cfg["dataset"].get("align_wilor_aug", True))
    bbox_source = "gt" if split == "train" else cfg["dataset"].get("bbox_source_eval", "detector")
    detector_path = cfg["paths"]["detector_ckpt"] if bbox_source == "detector" else None

    if name in ("dexycb", "dex-ycb"):
        from data.dex_ycb_dataset import DexYCBDataset

        return DexYCBDataset(
            setup=cfg["dataset"]["dexycb_setup"],
            split=("train" if split == "train" else "test"),
            root_dir=cfg["paths"]["dexycb_root"],
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
        )

    if name in ("ho3d",):
        from data.ho3d_dataset import HO3DDataset

        # NOTE:
        # HO3D v3 "evaluation/test" split does NOT include full 21-joint GT (often only root joint),
        # which makes PA-MPJPE ill-defined (GT variance ~ 0). For training-time validation we default
        # to a split with GT available (train/train_all) unless overridden.
        ho3d_val_split = str(cfg["dataset"].get("ho3d_val_split", "val")).lower()
        if split != "train" and ho3d_val_split == "train_all":
            # HO3D_v3 in this repo expects a list file named "<split>.txt".
            # By default only train.txt / evaluation.txt exist, so treat train_all as train.
            ho3d_val_split = "train"
        if split != "train" and ho3d_val_split in ("evaluation", "test"):
            if is_main_process():
                print(
                    f"[warn] HO3D val split is '{ho3d_val_split}', which may not contain full GT joints; "
                    f"PA-MPJPE can be 0. Consider setting dataset.ho3d_val_split: train"
                )

        return HO3DDataset(
            data_split=("train" if split == "train" else ho3d_val_split),
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
            trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
        )

    if name in ("freihand",):
        from data.freihand_dataset import FreiHANDDataset

        return FreiHANDDataset(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
        )

    raise ValueError(f"Unknown dataset.name: {cfg['dataset']['name']}")


def mano_from_gt(batch: Dict[str, torch.Tensor], device: torch.device):
    """
    Compute GT joints/verts in camera coordinates (mm) from GT MANO parameters + GT translation.
    - mano_pose: axis-angle (48,) in radians
    - mano_shape: (10,)
    - mano_trans: (3,) in meters (DexYCB/HO3D) or already meters (FreiHAND loader converts)
    """
    mano_pose = batch["mano_pose"].to(device)  # (B,48)
    betas = batch["mano_shape"].to(device)  # (B,10)
    mano_trans_m = batch["mano_trans"].to(device)  # (B,3) meters

    B = mano_pose.shape[0]
    pose_aa = mano_pose.reshape(B, 16, 3)
    pose_rm = aa_to_rotmat(pose_aa.reshape(-1, 3)).view(B, 16, 3, 3)
    mano_params = {"global_orient": pose_rm[:, [0]], "hand_pose": pose_rm[:, 1:], "betas": betas}
    return mano_params, mano_trans_m


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    distributed: bool,
    dist_info: DistInfo,
):
    model.eval()

    mpjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    pampjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    n = torch.zeros((), device=device, dtype=torch.float64)

    it = tqdm(loader, desc="val", disable=(not is_main_process()))
    warned_degenerate_gt = False
    for batch in it:
        img = batch["rgb"].to(device)
        cam_param = batch.get("cam_param", None)
        cam_param = cam_param.to(device) if cam_param is not None else None

        out = model(img, cam_param=cam_param)
        pred_t_m = out["pred_cam_t"]
        pred_j_mm = (out["pred_keypoints_3d"] + pred_t_m.unsqueeze(1)) * 1000.0

        gt_j_mm = batch["joints_3d_gt"].to(device)

        # Filter invalid/degenerate GT (e.g. HO3D evaluation split may only have root joint tiled to 21 joints).
        gt_var = gt_j_mm.var(dim=(1, 2))
        finite_mask = torch.isfinite(gt_j_mm).all(dim=(1, 2))
        valid_mask = (gt_var > 1e-8) & finite_mask
        if not bool(valid_mask.any()):
            if (not warned_degenerate_gt) and is_main_process():
                print("[warn] Skipping val batch with degenerate/invalid GT joints; metrics are not meaningful for this split.")
                warned_degenerate_gt = True
            continue

        pred_j_mm = pred_j_mm[valid_mask]
        gt_j_mm = gt_j_mm[valid_mask]

        mpjpe = (pred_j_mm - gt_j_mm).norm(dim=-1).mean(dim=-1)  # (B_valid,)
        pamp = torch.from_numpy(compute_pa_mpjpe(pred_j_mm, gt_j_mm)).to(device=device)  # (B_valid,)

        mpjpe_sum += mpjpe.double().sum()
        pampjpe_sum += pamp.double().sum()
        n += float(pred_j_mm.shape[0])

        if is_main_process():
            denom = float(max(float(n.item()), 1.0))
            it.set_postfix(mpjpe=f"{float(mpjpe_sum.item())/denom:.2f}", pamp=f"{float(pampjpe_sum.item())/denom:.2f}")

    if distributed:
        all_reduce_sum(mpjpe_sum)
        all_reduce_sum(pampjpe_sum)
        all_reduce_sum(n)

    n_val = float(n.item())
    if n_val <= 0:
        return float("nan"), float("nan")
    return float(mpjpe_sum.item()) / n_val, float(pampjpe_sum.item()) / n_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="torch.distributed backend (default: nccl if cuda else gloo)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # ---- distributed setup (torchrun) ----
    dist_info: DistInfo = setup_distributed(backend=args.backend)
    distributed = dist_info.distributed

    # Respect config's use_cuda in single-process mode; in distributed we assume CUDA per-rank.
    if (not distributed) and (not bool(cfg["train"].get("use_cuda", True))):
        device = torch.device("cpu")
    else:
        device = dist_info.device if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed per-rank for determinism-ish
    seed_everything(int(cfg["train"].get("seed", 42)), rank=(dist_info.rank if distributed else 0))

    try:
        # Build datasets
        train_ds = build_dataset(cfg, "train")
        val_ds = build_dataset(cfg, "val")

        train_sampler = None
        val_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=dist_info.world_size,
                rank=dist_info.rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=dist_info.world_size,
                rank=dist_info.rank,
                shuffle=False,
                drop_last=False,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=int(cfg["train"].get("num_workers", 4)),
            pin_memory=True,
            drop_last=distributed,  # keep all ranks aligned
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,  # detector in dataset is heavy; keep deterministic
            pin_memory=True,
        )

        # Model
        model: torch.nn.Module = GPGFormer(
            GPGFormerConfig(
                wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
                moge2_weights_path=cfg["paths"]["moge2_ckpt"],
                mano_model_path=cfg["paths"]["mano_dir"],
                mano_mean_params=cfg["paths"]["mano_mean_params"],
                # Fallback focal; real per-sample focal comes from batch["cam_param"] (fx,fy,cx,cy).
                focal_length=float(cfg["model"].get("focal_length", 5000.0)),
                mano_head_ief_iters=int(cfg["model"].get("mano_head", {}).get("ief_iters", 3)),
                mano_head_transformer_input=str(cfg["model"].get("mano_head", {}).get("transformer_input", "mean_shape")),
                mano_head_dim=int(cfg["model"].get("mano_head", {}).get("dim", 1024)),
                mano_head_depth=int(cfg["model"].get("mano_head", {}).get("depth", 6)),
                mano_head_heads=int(cfg["model"].get("mano_head", {}).get("heads", 8)),
                mano_head_dim_head=int(cfg["model"].get("mano_head", {}).get("dim_head", 64)),
                mano_head_mlp_dim=int(cfg["model"].get("mano_head", {}).get("mlp_dim", 2048)),
                mano_head_dropout=float(cfg["model"].get("mano_head", {}).get("dropout", 0.0)),
            )
        ).to(device)

        # IMPORTANT for DDP:
        # GPGFormer lazily creates some submodules (e.g., geo_tokenizer) on first forward.
        # If we wrap with DDP before those parameters exist, they won't be tracked/optimized consistently.
        # Warm up a single forward pass to materialize all lazy parameters before DDP wrapping.
        with torch.no_grad():
            img_dummy = torch.zeros((1, 3, 256, 256), device=device, dtype=torch.float32)
            cam_dummy = torch.tensor([[600.0, 600.0, 128.0, 128.0]], device=device, dtype=torch.float32)
            _ = model(img_dummy, cam_param=cam_dummy)

        if distributed:
            # Default to True for safety (some params can be conditionally unused depending on loss/config).
            find_unused = bool(cfg["train"].get("ddp_find_unused_parameters", True))
            model = DDP(
                model,
                device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
                output_device=dist_info.local_rank if device.type == "cuda" else None,
                find_unused_parameters=find_unused,
                broadcast_buffers=False,
            )

        # Loss
        loss_cfg = cfg["loss"]
        criterion = UTNetLoss(
            w_2d=float(loss_cfg.get("w_2d", 5.0)),
            w_3d_joint=float(loss_cfg.get("w_3d_joint", 1.0)),
            w_3d_vert=float(loss_cfg.get("w_3d_vert", 0.5)),
            w_global_orient=float(loss_cfg.get("w_global_orient", 10.0)),
            w_hand_pose=float(loss_cfg.get("w_hand_pose", 10.0)),
            # Optional stabilizers (off by default in existing configs)
            w_root_abs=float(loss_cfg.get("w_root_abs", 0.0)),
            w_mano_trans=float(loss_cfg.get("w_mano_trans", 0.0)),
            w_scale=float(loss_cfg.get("w_scale", 0.0)),
            w_betas=float(loss_cfg.get("w_betas", 0.0)),
            reduction=str(loss_cfg.get("reduction", "mean")),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )

        out_dir = Path(cfg["train"].get("out_dir", "checkpoints")) / cfg["dataset"]["name"]
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)

        epochs = int(cfg["train"]["epochs"])
        image_size = int(cfg.get("model", {}).get("image_size", cfg["dataset"].get("img_size", 256)))

        # ---- LR schedule: warmup + cosine (UniHandFormer-style) ----
        # Step per iteration so it behaves consistently under DDP / different batch sizes.
        steps_per_epoch = max(int(len(train_loader)), 1)
        total_steps = max(epochs * steps_per_epoch, 1)

        base_lr = float(cfg["train"]["lr"])
        min_lr = float(cfg["train"].get("min_lr", 0.0))
        min_lr_ratio = 0.0 if base_lr <= 0 else max(min_lr / base_lr, 0.0)

        warmup_steps = int(cfg["train"].get("warmup_steps", 0))
        warmup_epochs = int(cfg["train"].get("warmup_epochs", 1))
        if warmup_steps <= 0:
            warmup_steps = max(warmup_epochs * steps_per_epoch, 0)
        warmup_steps = min(max(warmup_steps, 0), total_steps)

        def lr_lambda(step: int) -> float:
            # step is 0-based global step
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if step >= total_steps:
                return min_lr_ratio
            denom = max(total_steps - warmup_steps, 1)
            progress = float(step - warmup_steps) / float(denom)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_pampjpe_mm = float("inf")

        for epoch in range(1, epochs + 1):
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            pbar = tqdm(train_loader, desc=f"train e{epoch}/{epochs}", disable=(not is_main_process()))
            running = 0.0

            for batch in pbar:
                img = batch["rgb"].to(device)  # (B,3,256,256) in [0,1]
                cam_param = batch.get("cam_param", None)
                cam_param = cam_param.to(device) if cam_param is not None else None

                out = model(img, cam_param=cam_param)

                # Predictions in camera coordinates (mm)
                pred_t_m = out["pred_cam_t"]  # (B,3) meters
                pred_j_mm = (out["pred_keypoints_3d"] + pred_t_m.unsqueeze(1)) * 1000.0
                pred_v_mm = (out["pred_vertices"] + pred_t_m.unsqueeze(1)) * 1000.0

                # 2D prediction in normalized crop space (WiLoR style)
                from third_party.wilor_min.wilor.utils.geometry import perspective_projection

                B = img.shape[0]
                if cam_param is not None:
                    focal = cam_param[:, :2].to(device=device, dtype=img.dtype)
                else:
                    focal = torch.full(
                        (B, 2),
                        float(cfg["model"].get("focal_length", 5000.0)),
                        device=device,
                        dtype=img.dtype,
                    )
                # IMPORTANT:
                # `out["pred_keypoints_3d"]` is in meters without camera translation.
                # `perspective_projection()` adds `translation` internally, so DO NOT pre-add `pred_t_m`.
                pred_kp2d = perspective_projection(
                    out["pred_keypoints_3d"],  # meters
                    translation=pred_t_m,
                    focal_length=focal / float(image_size),
                )
                # Guard against numerical blow-ups when depth goes near 0.
                if not torch.isfinite(pred_kp2d).all():
                    if is_main_process():
                        print("[warn] Non-finite pred_kp2d encountered; skipping batch to avoid divergence.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                preds = {
                    "keypoints_2d": pred_kp2d,
                    "keypoints_3d": pred_j_mm,
                    "vertices": pred_v_mm,
                    # Optional loss can supervise translation directly (meters).
                    "cam_translation": pred_t_m,
                }

                # Targets
                gt_j_mm = batch["joints_3d_gt"].to(device)  # (B,21,3) mm
                gt_kp3d = torch.cat([gt_j_mm, torch.ones_like(gt_j_mm[:, :, :1])], dim=-1)  # (B,21,4)
                gt_kp2d = batch["joint_img"].to(device)  # (B,21,3) normalized xy + conf

                # GT vertices via MANO (meters -> mm), translated to camera
                gt_mano_params, gt_t_m = mano_from_gt(batch, device)
                model_core = model.module if isinstance(model, DDP) else model
                gt_mano_out = model_core.mano(gt_mano_params, pose2rot=False)
                gt_v_mm = (gt_mano_out.vertices + gt_t_m.unsqueeze(1)) * 1000.0
                gt_root_mm = (gt_mano_out.joints[:, 0, :] + gt_t_m) * 1000.0

                targets = {
                    "keypoints_2d": gt_kp2d,
                    "keypoints_3d": gt_kp3d,
                    "vertices": gt_v_mm,
                    "vertices_root": gt_root_mm,
                    # Optional translation supervision (meters)
                    "mano_trans": gt_t_m,
                    # HO3D path provides MANO params for all samples in this split.
                    "has_mano_params": {
                        "global_orient": torch.ones(B, device=device, dtype=torch.bool),
                        "hand_pose": torch.ones(B, device=device, dtype=torch.bool),
                        "betas": torch.ones(B, device=device, dtype=torch.bool),
                    },
                }

                loss_dict = criterion(preds, targets)
                # UTNetLoss returns 'total_loss' (not 'loss')
                loss = loss_dict.get("loss", loss_dict.get("total_loss", None))
                if loss is None:
                    raise KeyError(f"UTNetLoss did not return 'loss' or 'total_loss'. Keys={list(loss_dict.keys())}")
                if not torch.isfinite(loss):
                    if is_main_process():
                        print("[warn] Non-finite loss encountered; skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Optional grad clipping for stability (configure train.grad_clip_norm)
                clip = float(cfg.get("train", {}).get("grad_clip_norm", 0.0) or 0.0)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()
                scheduler.step()

                # rank-avg loss for logging
                loss_val = torch.tensor([float(loss.item())], device=device)
                if distributed:
                    all_reduce_sum(loss_val)
                    loss_val = loss_val / float(dist_info.world_size)
                loss_scalar = float(loss_val.item())

                if is_main_process():
                    running = 0.95 * running + 0.05 * loss_scalar if running else loss_scalar
                    pbar.set_postfix(
                        loss=f"{running:.4f}",
                        v=float(loss_dict.get("loss_3d_vert", torch.tensor(0.0)).item()),
                        lr=float(optimizer.param_groups[0]["lr"]),
                    )

            # Eval after each epoch (like UniHandFormer)
            if device.type == "cuda":
                # Help release cached memory before running the detector during validation.
                torch.cuda.empty_cache()
            mpjpe_mm, pampjpe_mm = evaluate_epoch(model, val_loader, device, image_size, distributed, dist_info)
            if is_main_process():
                print(f"[epoch {epoch}] val MPJPE(mm)={mpjpe_mm:.3f}  PA-MPJPE(mm)={pampjpe_mm:.3f}")

                # Save only when test/val PA-MPJPE improves (decreases)
                if pampjpe_mm < best_pampjpe_mm:
                    best_pampjpe_mm = pampjpe_mm
                    ckpt_path = out_dir / "gpgformer_best.pt"
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "model": model_to_save.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "cfg": cfg,
                            "val_mpjpe_mm": mpjpe_mm,
                            "val_pampjpe_mm": pampjpe_mm,
                            "best_val_pampjpe_mm": best_pampjpe_mm,
                        },
                        ckpt_path,
                    )
                    print(f"[epoch {epoch}] saved best checkpoint: {ckpt_path} (best PA-MPJPE={best_pampjpe_mm:.3f}mm)")
                else:
                    print(f"[epoch {epoch}] not saving (best PA-MPJPE={best_pampjpe_mm:.3f}mm)")

        if is_main_process():
            print(f"Done. Checkpoints in: {out_dir}")
    finally:
        # Ensure NCCL process group is always cleaned up (even on exceptions)
        cleanup_distributed()


if __name__ == "__main__":
    main()



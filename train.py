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
from gpgformer.utils.ema import ModelEMA


def build_dataset(cfg: dict, split: str):
    name = cfg["dataset"]["name"].lower()
    align_wilor_aug = bool(cfg["dataset"].get("align_wilor_aug", True))
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))
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
            root_index=root_index,
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
            root_index=root_index,
        )

    if name in ("freihand",):
        # from data.freihand_dataset import FreiHANDDataset

        # return FreiHANDDataset(
        #     root_dir=cfg["paths"]["freihand_root"],
        #     eval_root=cfg["paths"].get("freihand_eval_root", None),
        #     img_size=int(cfg["dataset"].get("img_size", 256)),
        #     train=(split == "train"),
        #     align_wilor_aug=align_wilor_aug,
        #     wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
        #     bbox_source=bbox_source,
        #     detector_weights_path=detector_path,
        #     root_index=root_index,
        #     trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
        #     trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
        #     use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
        # )
        from data.freihand_dataset_v2 import FreiHANDDatasetV2
        return FreiHANDDatasetV2(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),    
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
            trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
            use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
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
    root_index: int = 9,
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
        pred_t_m = out["pred_cam_t"]  # meters
        # Use meters end-to-end; convert to mm only for metric outputs.
        pred_j_m = out["pred_keypoints_3d"]

        
        gt_j_m = batch.get("keypoints_3d", None)
        gt_j_m = gt_j_m.to(device)

        # Filter invalid/degenerate GT (e.g. HO3D evaluation split may only have root joint tiled to 21 joints).
        gt_var = gt_j_m.var(dim=(1, 2))
        finite_mask = torch.isfinite(gt_j_m).all(dim=(1, 2))
        valid_mask = (gt_var > 1e-8) & finite_mask
        if not bool(valid_mask.any()):
            if (not warned_degenerate_gt) and is_main_process():
                print("[warn] Skipping val batch with degenerate/invalid GT joints; metrics are not meaningful for this split.")
                warned_degenerate_gt = True
            continue

        pred_j_m = pred_j_m[valid_mask]
        gt_j_m = gt_j_m[valid_mask]

        # Root-center both pred/gt with the same root index before MPJPE/PA-MPJPE.
        ri = int(root_index)
        pred_j_m = pred_j_m - pred_j_m[:, [ri]]
        gt_j_m = gt_j_m - gt_j_m[:, [ri]]

        mpjpe = (pred_j_m - gt_j_m).norm(dim=-1).mean(dim=-1)  # (B_valid,) meters
        pamp = torch.from_numpy(compute_pa_mpjpe(pred_j_m, gt_j_m)).to(device=device)  # (B_valid,) meters

        mpjpe_sum += mpjpe.double().sum()
        pampjpe_sum += pamp.double().sum()
        n += float(pred_j_m.shape[0])

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
    mpjpe_m = float(mpjpe_sum.item()) / n_val
    pampjpe_m = float(pampjpe_sum.item()) / n_val
    # Convert metrics to mm for logging/output
    return mpjpe_m * 1000.0, pampjpe_m * 1000.0


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
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))

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
                mano_decoder=str(cfg["model"].get("mano_decoder", "wilor")),
                freihand_mano_root=cfg["model"].get("freihand_mano_root", None),
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
                # Reduce MoGe2 token grid to avoid 32-bit index overflow in neck resamplers.
                moge2_num_tokens=int(cfg["model"].get("moge2_num_tokens", 400)),
                # Token fusion mode: concat or sum
                token_fusion_mode=str(cfg["model"].get("token_fusion_mode", "concat")),
                sum_fusion_strategy=str(cfg["model"].get("sum_fusion_strategy", "basic")),
                # Feature Refiner configuration
                feature_refiner_method=str(cfg["model"].get("feature_refiner", {}).get("method", "none")),
                feature_refiner_feat_dim=int(cfg["model"].get("feature_refiner", {}).get("feat_dim", 1280)),
                feature_refiner_sjta_bottleneck_dim=int(cfg["model"].get("feature_refiner", {}).get("sjta_bottleneck_dim", 256)),
                feature_refiner_sjta_num_heads=int(cfg["model"].get("feature_refiner", {}).get("sjta_num_heads", 4)),
                feature_refiner_sjta_use_2d_prior=bool(cfg["model"].get("feature_refiner", {}).get("sjta_use_2d_prior", True)),
                feature_refiner_coear_dilation1=int(cfg["model"].get("feature_refiner", {}).get("coear_dilation1", 1)),
                feature_refiner_coear_dilation2=int(cfg["model"].get("feature_refiner", {}).get("coear_dilation2", 2)),
                feature_refiner_coear_gate_reduction=int(cfg["model"].get("feature_refiner", {}).get("coear_gate_reduction", 8)),
                feature_refiner_coear_init_alpha=float(cfg["model"].get("feature_refiner", {}).get("coear_init_alpha", 0.1)),
                feature_refiner_kcr_num_keypoints=int(cfg["model"].get("feature_refiner", {}).get("kcr_num_keypoints", 21)),
                feature_refiner_kcr_hidden_dim=int(cfg["model"].get("feature_refiner", {}).get("kcr_hidden_dim", 128)),
            )
        ).to(device)

        # IMPORTANT for DDP:
        # GPGFormer lazily creates some submodules (e.g., geo_tokenizer) on first forward.
        # If we wrap with DDP before those parameters exist, they won't be tracked/optimized consistently.
        # Warm up a single forward pass to materialize all lazy parameters before DDP wrapping.
        with torch.no_grad():
            h = int(cfg.get("model", {}).get("image_size", 256))
            w = int(cfg.get("model", {}).get("image_width", int(h * 0.75)))
            img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
            cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
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
            # Strict loss: only enable vertex loss when explicitly configured AND GT vertices exist.
            w_3d_vert=float(loss_cfg.get("w_3d_vert", 0.0)),
            w_global_orient=float(loss_cfg.get("w_global_orient", 10.0)),
            w_hand_pose=float(loss_cfg.get("w_hand_pose", 10.0)),
            root_index=root_index,
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

        # ---- EMA (Exponential Moving Average) ----
        use_ema = bool(cfg["train"].get("use_ema", False))
        ema_decay = float(cfg["train"].get("ema_decay", 0.9999))
        model_ema = None
        if use_ema:
            # Get the base model (unwrap DDP if needed)
            base_model = model.module if isinstance(model, DDP) else model
            model_ema = ModelEMA(base_model, decay=ema_decay, device=device)
            if is_main_process():
                print(f"[info] EMA enabled with decay={ema_decay}")

        best_pampjpe_mm = float("inf")

        for epoch in range(1, epochs + 1):
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            pbar = tqdm(train_loader, desc=f"train e{epoch}/{epochs}", disable=(not is_main_process()))
            running = 0.0
            # Track per-loss running/epoch averages
            loss_sums = {}
            loss_count = 0.0
            ema_losses = {}

            for batch in pbar:
                img = batch["rgb"].to(device)  # (B,3,H,W) in [0,1]
                cam_param = batch.get("cam_param", None)
                cam_param = cam_param.to(device) if cam_param is not None else None

                out = model(img, cam_param=cam_param)

                # Predictions in camera coordinates (meters)
                pred_t_m = out["pred_cam_t"]  # (B,3) meters
                pred_j_m = out["pred_keypoints_3d"]
                pred_v_m = out["pred_vertices"]

                # 2D prediction:
                # - Project to patch pixels using cam_param=(fx,fy,cx,cy) in pixel units
                # - Then normalize to [-0.5,0.5] to match GT produced by `get_example()`
                from third_party.wilor_min.wilor.utils.geometry import perspective_projection

                B = img.shape[0]
                if cam_param is not None:
                    cam_param_t = cam_param.to(device=device, dtype=img.dtype)
                    focal = cam_param_t[:, :2]
                    center = cam_param_t[:, 2:4]
                else:
                    focal = torch.full(
                        (B, 2),
                        float(cfg["model"].get("focal_length", 5000.0)),
                        device=device,
                        dtype=img.dtype,
                    )
                    center = torch.zeros((B, 2), device=device, dtype=img.dtype)
                # IMPORTANT:
                # Follow HaMeR convention: MANO outputs are in camera frame (no translation applied).
                # Use pred_cam_t only inside perspective_projection.
                pred_kp2d_px = perspective_projection(
                    out["pred_keypoints_3d"],  # meters
                    translation=pred_t_m,
                    focal_length=focal,        # pixels
                    camera_center=center,      # pixels
                )
                # Guard against numerical blow-ups when depth goes near 0.
                if not torch.isfinite(pred_kp2d_px).all():
                    if is_main_process():
                        print("[warn] Non-finite pred_kp2d encountered; skipping batch to avoid divergence.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Normalize predicted 2D to [-0.5, 0.5] using actual patch size (supports non-square patches)
                H = float(img.shape[-2])
                W = float(img.shape[-1])
                pred_kp2d = pred_kp2d_px.clone()
                pred_kp2d[..., 0] = pred_kp2d[..., 0] / W - 0.5
                pred_kp2d[..., 1] = pred_kp2d[..., 1] / H - 0.5

                preds = {
                    "keypoints_2d": pred_kp2d,
                    "keypoints_3d": pred_j_m,
                    "vertices": pred_v_m,
                    # Optional loss can supervise translation directly (meters).
                    "cam_translation": pred_t_m,
                }
                # IMPORTANT:
                # UTNetLoss expects `predictions["mano_params"]` for MANO prior + direct MANO supervision losses.
                # The model outputs this under the key `pred_mano_params`.
                if "pred_mano_params" in out:
                    preds["mano_params"] = out["pred_mano_params"]

                # Targets (adapt to dataset outputs)
                gt_kp2d = batch.get("keypoints_2d", batch.get("joint_img"))
                gt_kp2d = gt_kp2d.to(device) if gt_kp2d is not None else None

                gt_kp3d = batch.get("keypoints_3d", None)
                if gt_kp3d is None:
                    gt_kp3d = batch.get("joints_3d_gt", None)
                gt_kp3d = gt_kp3d.to(device) if gt_kp3d is not None else None
                # Use meters end-to-end in training

                # Optional GT vertices via MANO (meters)
                gt_v_mm = None
                gt_root_mm = None
                gt_t_m = None
                if ("mano_pose" in batch) and ("mano_shape" in batch) and ("mano_trans" in batch):
                    gt_mano_params, gt_t_m = mano_from_gt(batch, device)
                    model_core = model.module if isinstance(model, DDP) else model
                    gt_mano_out = model_core.mano(gt_mano_params, pose2rot=False)
                    gt_v_mm = (gt_mano_out.vertices + gt_t_m.unsqueeze(1))
                    gt_root_mm = (gt_mano_out.joints[:, root_index, :] + gt_t_m)

                targets = {
                    "keypoints_2d": gt_kp2d,
                    "keypoints_3d": gt_kp3d,
                    "vertices": gt_v_mm,
                    "vertices_root": gt_root_mm,
                    "mano_trans": gt_t_m,
                }

                # MANO GT (axis-angle by default)
                mano_pose = batch.get("mano_pose", None)
                mano_shape = batch.get("mano_shape", None)
                mano_params = batch.get("mano_params", None)
                if mano_pose is None and isinstance(mano_params, dict):
                    go = torch.as_tensor(mano_params["global_orient"])
                    hp = torch.as_tensor(mano_params["hand_pose"])
                    if go.dim() == 1:
                        go = go.unsqueeze(0)
                    if hp.dim() == 1:
                        hp = hp.unsqueeze(0)
                    mano_pose = torch.cat([go, hp], dim=-1)
                if mano_shape is None and isinstance(mano_params, dict):
                    mano_shape = torch.as_tensor(mano_params["betas"])
                    if mano_shape.dim() == 1:
                        mano_shape = mano_shape.unsqueeze(0)
                if mano_pose is not None:
                    targets["mano_pose"] = mano_pose.to(device)
                if mano_shape is not None:
                    targets["mano_shape"] = mano_shape.to(device)

                has_mano_params = batch.get("has_mano_params", None)
                if has_mano_params is None:
                    has_mano_params = {
                        "global_orient": torch.ones(B, device=device, dtype=torch.bool),
                        "hand_pose": torch.ones(B, device=device, dtype=torch.bool),
                        "betas": torch.ones(B, device=device, dtype=torch.bool),
                    }
                targets["has_mano_params"] = has_mano_params

                def _to_device(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.to(device)
                    return torch.as_tensor(x, device=device)

                targets["mano_params_is_axis_angle"] = batch.get("mano_params_is_axis_angle", None)
                targets["uv_valid"] = _to_device(batch.get("uv_valid", None))
                targets["xyz_valid"] = _to_device(batch.get("xyz_valid", None))
                # GT keypoints_2d is already normalized by the dataloader, so do NOT re-map with bbox fields.
                targets["box_center"] = None
                targets["box_size"] = None
                targets["bbox_expand_factor"] = None
                targets["_scale"] = None

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

                # Update EMA model after optimizer step
                if model_ema is not None:
                    # Get the base model (unwrap DDP if needed)
                    base_model = model.module if isinstance(model, DDP) else model
                    model_ema.update(base_model)

                # rank-avg loss for logging
                loss_val = torch.tensor([float(loss.item())], device=device)
                if distributed:
                    all_reduce_sum(loss_val)
                    loss_val = loss_val / float(dist_info.world_size)
                loss_scalar = float(loss_val.item())

                # Collect sub-loss scalars for logging.
                # IMPORTANT: all ranks must participate in all_reduce to avoid deadlocks.
                base_loss_keys = [
                    "loss_2d",
                    "loss_3d_joint",
                    "loss_3d_vert",
                    "loss_global_orient",
                    "loss_hand_pose",
                    "loss_betas",
                ]
                subloss_vals = {}
                for k in base_loss_keys:
                    v = loss_dict.get(k, None)
                    if v is None or (not torch.is_tensor(v)):
                        val_t = torch.tensor([0.0], device=device)
                    else:
                        val_t = torch.tensor([float(v.item())], device=device)
                    if distributed:
                        all_reduce_sum(val_t)
                        val_t = val_t / float(dist_info.world_size)
                    subloss_vals[k] = float(val_t.item())

                if is_main_process():
                    running = 0.95 * running + 0.05 * loss_scalar if running else loss_scalar

                    # Update EMA and epoch sums only on rank0
                    batch_w = float(img.shape[0])
                    loss_count += batch_w
                    for k, v in subloss_vals.items():
                        if k in loss_dict:  # only track actually-used losses
                            ema_losses[k] = 0.95 * ema_losses.get(k, v) + 0.05 * v
                            loss_sums[k] = loss_sums.get(k, 0.0) + v * batch_w

                    postfix = {
                        "loss": f"{running:.4f}",
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    for k in sorted(ema_losses.keys()):
                        postfix[k] = f"{ema_losses[k]:.4f}"
                    pbar.set_postfix(**postfix)

            # ---- Optional train-set metrics after each epoch ----
            def _cfg_bool(val, default: bool = False) -> bool:
                if val is None:
                    return default
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in ("1", "true", "yes", "y", "on"):
                        return True
                    if v in ("0", "false", "no", "n", "off", ""):
                        return False
                return bool(val)

            # Allow both train.eval_train_metrics and top-level eval_train_metrics for compatibility.
            eval_train_metrics = _cfg_bool(
                cfg.get("train", {}).get("eval_train_metrics", cfg.get("eval_train_metrics", True)),
                default=True,
            )
            if eval_train_metrics:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # Use EMA model for evaluation if available
                eval_model = model_ema.module() if model_ema is not None else model
                train_mpjpe_mm, train_pampjpe_mm = evaluate_epoch(
                    eval_model,
                    train_loader,
                    device,
                    image_size,
                    distributed,
                    dist_info,
                    root_index=root_index,
                )
                if is_main_process():
                    ema_tag = " (EMA)" if model_ema is not None else ""
                    print(
                        f"[epoch {epoch}] train{ema_tag} MPJPE(mm)={train_mpjpe_mm:.3f}  "
                        f"PA-MPJPE(mm)={train_pampjpe_mm:.3f}"
                    )

            # Print epoch-mean sublosses
            if is_main_process() and loss_count > 0:
                parts = []
                for k in sorted(loss_sums.keys()):
                    parts.append(f"{k}={loss_sums[k]/loss_count:.4f}")
                if parts:
                    print(f"[epoch {epoch}] train losses: " + "  ".join(parts))

            # Eval after each epoch (like UniHandFormer)
            if device.type == "cuda":
                # Help release cached memory before running the detector during validation.
                torch.cuda.empty_cache()
            # Use EMA model for evaluation if available
            eval_model = model_ema.module() if model_ema is not None else model
            mpjpe_mm, pampjpe_mm = evaluate_epoch(
                eval_model,
                val_loader,
                device,
                image_size,
                distributed,
                dist_info,
                root_index=root_index,
            )
            if is_main_process():
                ema_tag = " (EMA)" if model_ema is not None else ""
                print(f"[epoch {epoch}] val{ema_tag} MPJPE(mm)={mpjpe_mm:.3f}  PA-MPJPE(mm)={pampjpe_mm:.3f}")

                # Save only when test/val PA-MPJPE improves (decreases)
                if pampjpe_mm < best_pampjpe_mm:
                    best_pampjpe_mm = pampjpe_mm
                    ckpt_path = out_dir / "gpgformer_best.pt"
                    model_to_save = model.module if isinstance(model, DDP) else model

                    # Prepare checkpoint dict
                    checkpoint = {
                        "model": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg,
                        "val_mpjpe_mm": mpjpe_mm,
                        "val_pampjpe_mm": pampjpe_mm,
                        "best_val_pampjpe_mm": best_pampjpe_mm,
                    }

                    # Save EMA model state if available
                    if model_ema is not None:
                        checkpoint["model_ema"] = model_ema.state_dict()

                    torch.save(checkpoint, ckpt_path)
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



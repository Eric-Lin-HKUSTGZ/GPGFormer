import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.hand_dataset import HandDataset
from src.datasets.ho3d_dataset import HO3DDataset
from src.datasets.freihand_dataset import FreiHANDDataset
from src.datasets.dex_ycb_dataset import DexYCBDataset
from src.models.gpgformer import GPGFormer
from src.losses.loss import GPGFormerLoss
from src.utils.detector import HandDetector

def _compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    mu1 = S1.mean(dim=-1, keepdim=True)
    mu2 = S2.mean(dim=-1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.matmul(X1, X2.transpose(1, 2))
    U, _, V = torch.svd(K)
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(S1.shape[0], 1, 1)
    det = torch.det(torch.matmul(V, U.transpose(1, 2)))
    Z[:, 2, 2] = torch.sign(det)
    R = torch.matmul(torch.matmul(V, Z), U.transpose(1, 2))
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    t = mu2 - scale * torch.matmul(R, mu1)
    S1_hat = scale * torch.matmul(R, S1) + t
    return S1_hat.permute(0, 2, 1)


def compute_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> np.ndarray:
    if pred_joints.dim() == 2:
        pred_joints = pred_joints.unsqueeze(0)
        gt_joints = gt_joints.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    pred_joints = torch.where(torch.isfinite(pred_joints), pred_joints, torch.zeros_like(pred_joints))
    gt_joints = torch.where(torch.isfinite(gt_joints), gt_joints, torch.zeros_like(gt_joints))
    var_gt = gt_joints.var(dim=(1, 2))
    valid_mask = var_gt > 1e-8
    if not valid_mask.any():
        zeros = torch.zeros((pred_joints.shape[0],), device=pred_joints.device)
        out = zeros.cpu().numpy()
        return out[0] if squeeze_output else out
    joint_errors = torch.sqrt(torch.clamp(((pred_joints - gt_joints) ** 2).sum(dim=-1), min=1e-12))
    mpjpe = joint_errors.mean(dim=-1)
    mpjpe_mm = mpjpe.cpu().numpy()
    return mpjpe_mm[0] if squeeze_output else mpjpe_mm


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> np.ndarray:
    if pred_joints.dim() == 2:
        pred_joints = pred_joints.unsqueeze(0)
        gt_joints = gt_joints.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    pred_joints = torch.where(torch.isfinite(pred_joints), pred_joints, torch.zeros_like(pred_joints))
    gt_joints = torch.where(torch.isfinite(gt_joints), gt_joints, torch.zeros_like(gt_joints))
    var_gt = gt_joints.var(dim=(1, 2))
    valid_mask = var_gt > 1e-8
    if not valid_mask.any():
        zeros = torch.zeros((pred_joints.shape[0],), device=pred_joints.device)
        out = zeros.cpu().numpy()
        return out[0] if squeeze_output else out
    pred_joints_aligned = _compute_similarity_transform(pred_joints, gt_joints)
    joint_errors = torch.sqrt(torch.clamp(((pred_joints_aligned - gt_joints) ** 2).sum(dim=-1), min=1e-12))
    pa_mpjpe = joint_errors.mean(dim=-1)
    pa_mpjpe_mm = pa_mpjpe.cpu().numpy()
    return pa_mpjpe_mm[0] if squeeze_output else pa_mpjpe_mm


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_normalization(dataset_cfg: dict) -> Tuple[Optional[list], Optional[list]]:
    if dataset_cfg.get("normalize", "none") == "imagenet":
        return dataset_cfg.get("normalize_mean"), dataset_cfg.get("normalize_std")
    return None, None


def _split_dataset_if_needed(dataset, split: str, val_split_ratio: float) -> Tuple[Subset, Optional[Subset]]:
    if split not in ["train", "val"] or val_split_ratio <= 0:
        return dataset, None
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(total_size * (1 - val_split_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    if split == "train":
        return Subset(dataset, train_indices), Subset(dataset, val_indices)
    return Subset(dataset, val_indices), Subset(dataset, train_indices)


def _resolve_dataset_img_size(dataset_cfg: dict):
    dataset_img_size = dataset_cfg.get("img_size")
    if isinstance(dataset_img_size, (list, tuple)):
        return dataset_img_size[0]
    return dataset_img_size


def _build_wilor_aug_config(config: dict) -> dict:
    aug_cfg = config.get("augmentation", {})
    return {
        "TRANS_FACTOR": aug_cfg.get("wilor_trans_factor", 0.02),
        "SCALE_FACTOR": aug_cfg.get("wilor_scale_factor", 0.25),
        "ROT_FACTOR": aug_cfg.get("wilor_rot_factor", 30.0),
        "ROT_AUG_RATE": aug_cfg.get("wilor_rot_aug_rate", 0.6),
        "DO_FLIP": aug_cfg.get("wilor_do_flip", False),
        "FLIP_AUG_RATE": aug_cfg.get("wilor_flip_aug_rate", 0.0),
        "EXTREME_CROP_AUG_RATE": aug_cfg.get("wilor_extreme_crop_aug_rate", 0.0),
        "COLOR_SCALE": aug_cfg.get("wilor_color_scale", 0.2),
    }


def _compute_loss_breakdown(criterion: GPGFormerLoss, preds: dict, targets: dict) -> dict:
    breakdown = {
        "loss_pose": 0.0,
        "loss_shape": 0.0,
        "loss_cam": 0.0,
        "loss_joints_2d": 0.0,
        "loss_joints_3d": 0.0,
        "total_loss": 0.0,
    }

    if targets.get("mano_pose") is not None:
        has_pose = targets.get("has_mano_pose")
        if has_pose is None:
            has_pose = criterion._has_param(targets["mano_pose"])
        loss_pose = criterion.param_loss(preds["pred_pose"], targets["mano_pose"], has_pose)
        breakdown["loss_pose"] = (criterion.w_pose * loss_pose).item()

    if targets.get("mano_shape") is not None:
        has_shape = targets.get("has_mano_shape")
        if has_shape is None:
            has_shape = criterion._has_param(targets["mano_shape"])
        loss_shape = criterion.param_loss(preds["pred_shape"], targets["mano_shape"], has_shape)
        breakdown["loss_shape"] = (criterion.w_shape * loss_shape).item()

    if targets.get("cam_t") is not None:
        has_cam = targets.get("has_cam_t")
        if has_cam is None:
            has_cam = criterion._has_param(targets["cam_t"])
        loss_cam = criterion.param_loss(preds["pred_cam_t"], targets["cam_t"], has_cam)
        breakdown["loss_cam"] = (criterion.w_cam * loss_cam).item()

    if criterion.w_joints_3d > 0 and preds.get("pred_joints") is not None and targets.get("joints_3d_gt") is not None:
        gt_joints = targets["joints_3d_gt"]
        if gt_joints.dim() == 3 and gt_joints.shape[-1] == 3:
            conf = torch.ones_like(gt_joints[..., :1])
            gt_joints = torch.cat([gt_joints, conf], dim=-1)
        loss_3d = criterion.kp3d_loss(preds["pred_joints"], gt_joints)
        breakdown["loss_joints_3d"] = (criterion.w_joints_3d * loss_3d).item()

    if criterion.w_joints_2d > 0 and preds.get("pred_joints_2d") is not None and targets.get("joints_2d") is not None:
        gt_2d = targets["joints_2d"]
        if gt_2d.dim() == 3 and gt_2d.shape[-1] == 2:
            conf = torch.ones_like(gt_2d[..., :1])
            gt_2d = torch.cat([gt_2d, conf], dim=-1)
        loss_2d = criterion.kp2d_loss(preds["pred_joints_2d"], gt_2d)
        breakdown["loss_joints_2d"] = (criterion.w_joints_2d * loss_2d).item()

    breakdown["total_loss"] = (
        breakdown["loss_pose"]
        + breakdown["loss_shape"]
        + breakdown["loss_cam"]
        + breakdown["loss_joints_2d"]
        + breakdown["loss_joints_3d"]
    )
    return breakdown


def _strip_moge_from_state_dict(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.startswith("moge.")}


def create_dataloader(config: dict, split: str = "train", distributed: bool = False,
                      rank: int = 0, world_size: int = 1):
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg.get("name", "image_folder").lower()
    val_split_ratio = config.get("training", {}).get("val_split_ratio", 0.1)
    wilor_aug_config = _build_wilor_aug_config(config)

    if dataset_name in ["ho3d", "freihand"] and split in ["train", "val"]:
        actual_split = "train"
    else:
        actual_split = split

    if dataset_name == "ho3d":
        dataset_img_size = _resolve_dataset_img_size(dataset_cfg)
        dataset = HO3DDataset(
            data_split=actual_split,
            root_dir=dataset_cfg["root_dir"],
            dataset_version=dataset_cfg.get("version", "v3"),
            img_size=dataset_img_size,
            aug_para=[
                config["augmentation"]["sigma_com"],
                config["augmentation"]["sigma_sc"],
                config["augmentation"]["rot_range"],
            ],
            cube_size=dataset_cfg.get("cube_size", [280, 280, 280]),
            input_modal="RGB",
            color_factor=config["augmentation"].get("color_factor", 0.2),
            p_drop=dataset_cfg.get("p_drop", 1.0),
            train=(split == "train"),
            aug_prob=config["augmentation"].get("aug_prob", 0.8),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=wilor_aug_config,
        )
    elif dataset_name == "freihand":
        dataset_img_size = _resolve_dataset_img_size(dataset_cfg)
        dataset = FreiHANDDataset(
            root_dir=dataset_cfg["root_dir"],
            img_size=dataset_img_size,
            cube_size=dataset_cfg.get("cube_size", 280),
            train=(split == "train"),
            color_factor=config["augmentation"].get("color_factor", 0.2),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=wilor_aug_config,
        )
    elif dataset_name in ["dexycb", "dex-ycb"]:
        dataset_img_size = _resolve_dataset_img_size(dataset_cfg)
        dataset = DexYCBDataset(
            setup=dataset_cfg.get("setup", "s0"),
            split=split,
            root_dir=dataset_cfg["root_dir"],
            img_size=dataset_img_size,
            aug_para=[
                config["augmentation"]["sigma_com"],
                config["augmentation"]["sigma_sc"],
                config["augmentation"]["rot_range"],
            ],
            input_modal="RGB",
            p_drop=dataset_cfg.get("p_drop", 1.0),
            train=(split == "train"),
            color_factor=config["augmentation"].get("color_factor", 0.2),
            aug_prob=config["augmentation"].get("aug_prob", 0.8),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=wilor_aug_config,
        )
    else:
        normalize_mean, normalize_std = _resolve_normalization(dataset_cfg)
        resize_to = dataset_cfg.get("input_size", dataset_cfg.get("img_size"))
        dataset = HandDataset(
            root_dir=dataset_cfg["root_dir"],
            annotation_path=dataset_cfg.get("annotation_path"),
            resize_to=resize_to,
            train=(split == "train"),
            color_factor=dataset_cfg.get("color_factor", 0.0),
            color_aug_prob=dataset_cfg.get("color_aug_prob", 0.0),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    dataset, val_subset = _split_dataset_if_needed(dataset, split, val_split_ratio)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train" and sampler is None),
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader, val_subset


def init_distributed() -> Tuple[bool, int, int, int]:
    if "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def create_optimizer(model: GPGFormer, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    optimizer_type = optimizer_config["type"]
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
            betas=optimizer_config.get("betas", (0.9, 0.999)),
        )
    if optimizer_type == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
            betas=optimizer_config.get("betas", (0.9, 0.999)),
        )
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "none")
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 30),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 1.0e-6),
        )
    return None


def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device,
         epoch: int, writer: Optional[SummaryWriter], rank: int, distributed: bool,
         detector: Optional[HandDetector]) -> dict:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_pred_keypoints = []
    all_gt_keypoints = []

    if distributed and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val {epoch}", disable=(rank != 0))
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            bboxes = batch.get("bbox")
            if bboxes is None and detector is not None and "image_bgr" in batch:
                box_list = []
                for img in batch["image_bgr"]:
                    img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
                    dets = detector.detect(img_np)
                    h, w = img_np.shape[:2]
                    box_list.append(dets[0] if dets else [0, 0, w - 1, h - 1])
                bboxes = torch.tensor(box_list, dtype=torch.float32, device=device)
            if bboxes is None:
                bboxes = torch.tensor(
                    [[0, 0, rgb.shape[-1] - 1, rgb.shape[-2] - 1]] * rgb.shape[0],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                bboxes = bboxes.to(device)

            cam_param = batch.get("cam_param")
            if cam_param is not None:
                cam_param = cam_param.to(device)
            preds = model(rgb, bboxes, cam_param=cam_param)

            cam_t = batch.get("cam_t")
            if cam_t is None and "mano_trans" in batch:
                cam_t = batch.get("mano_trans")
            joints_2d = None
            if "joint_img" in batch:
                joint_img = batch["joint_img"].to(device)
                if joint_img.dim() == 3:
                    conf = torch.ones_like(joint_img[..., :1])
                    joints_2d = torch.cat([joint_img[..., :2], conf], dim=-1)

            targets = {
                "mano_pose": batch.get("mano_pose").to(device) if "mano_pose" in batch else None,
                "mano_shape": batch.get("mano_shape").to(device) if "mano_shape" in batch else None,
                "cam_t": cam_t.to(device) if cam_t is not None else None,
                "joints_3d_gt": batch.get("joints_3d_gt").to(device) if "joints_3d_gt" in batch else None,
                "joints_2d": joints_2d,
            }

            loss = criterion(preds, targets)
            total_loss += float(loss)
            num_batches += 1

            if preds.get("pred_joints") is not None and targets.get("joints_3d_gt") is not None:
                all_pred_keypoints.append(preds["pred_joints"].detach().cpu())
                all_gt_keypoints.append(targets["joints_3d_gt"].detach().cpu())

    if distributed:
        loss_tensor = torch.tensor(total_loss / max(num_batches, 1), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = total_loss / max(num_batches, 1)

    metrics_dict = {
        "test_loss": avg_loss,
        "mpjpe": float("inf"),
        "pa_mpjpe": float("inf"),
        "avg_metric": float("inf"),
    }

    if not all_pred_keypoints or not all_gt_keypoints:
        return metrics_dict

    if distributed:
        local_pred = torch.cat(all_pred_keypoints, dim=0)
        local_gt = torch.cat(all_gt_keypoints, dim=0)
        world_size = dist.get_world_size()
        local_size = torch.tensor([local_pred.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        if rank == 0:
            all_preds_list = []
            all_gts_list = []
            for i, size in enumerate(size_list):
                num_samples = size.item()
                if i == rank:
                    all_preds_list.append(local_pred.cpu())
                    all_gts_list.append(local_gt.cpu())
                else:
                    recv_pred = torch.zeros(
                        num_samples,
                        local_pred.shape[1],
                        local_pred.shape[2],
                        dtype=local_pred.dtype,
                        device=device,
                    )
                    recv_gt = torch.zeros(
                        num_samples,
                        local_gt.shape[1],
                        local_gt.shape[2],
                        dtype=local_gt.dtype,
                        device=device,
                    )
                    dist.recv(recv_pred, src=i)
                    dist.recv(recv_gt, src=i)
                    all_preds_list.append(recv_pred.cpu())
                    all_gts_list.append(recv_gt.cpu())
            all_pred = torch.cat(all_preds_list, dim=0)
            all_gt = torch.cat(all_gts_list, dim=0)
        else:
            dist.send(local_pred.to(device), dst=0)
            dist.send(local_gt.to(device), dst=0)
            return metrics_dict
    else:
        all_pred = torch.cat(all_pred_keypoints, dim=0)
        all_gt = torch.cat(all_gt_keypoints, dim=0)

    all_pred_centered = all_pred - all_pred[:, [0], :]
    all_gt_centered = all_gt - all_gt[:, [0], :]
    gt_var = all_gt_centered.var(dim=(1, 2))
    finite_mask = torch.isfinite(all_gt_centered).all(dim=(1, 2))
    valid_mask = (gt_var > 1e-8) & finite_mask
    if valid_mask.any():
        all_pred_valid = all_pred_centered[valid_mask].to(device)
        all_gt_valid = all_gt_centered[valid_mask].to(device)
        mpjpe_array = compute_mpjpe(all_pred_valid, all_gt_valid)
        pa_mpjpe_array = compute_pa_mpjpe(all_pred_valid, all_gt_valid)
        mpjpe = float(np.mean(mpjpe_array))
        pa_mpjpe = float(np.mean(pa_mpjpe_array))
        avg_metric = (mpjpe + pa_mpjpe) / 2.0
    else:
        mpjpe = float("inf")
        pa_mpjpe = float("inf")
        avg_metric = float("inf")

    metrics_dict = {
        "test_loss": avg_loss,
        "mpjpe": mpjpe,
        "pa_mpjpe": pa_mpjpe,
        "avg_metric": avg_metric,
    }

    if rank == 0 and writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/mpjpe", mpjpe, epoch)
        writer.add_scalar("val/pa_mpjpe", pa_mpjpe, epoch)
        writer.add_scalar("val/avg_metric", avg_metric, epoch)

    return metrics_dict


def _initialize_geom_embed(model: GPGFormer, config: dict, device: torch.device) -> None:
    dataset_cfg = config.get("dataset", {})
    img_size = dataset_cfg.get("img_size", 256)
    if isinstance(img_size, (list, tuple)):
        img_h = int(img_size[0])
        img_w = int(img_size[1]) if len(img_size) > 1 else int(img_size[0])
    else:
        img_h = img_w = int(img_size)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_h, img_w, device=device)
        geom_feat = model.moge(dummy)
    if model.geom_embed.in_channels != geom_feat.shape[1]:
        model.geom_embed = nn.Conv2d(geom_feat.shape[1], model.embed_dim, kernel_size=1).to(device)


def load_wilor_pretrained(model: GPGFormer, ckpt_path: str) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Warning: pretrained weight not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()

    for key, value in state.items():
        if "patch_embed.proj.weight" in key:
            if model_state["patch_embed_rgb.proj.weight"].shape == value.shape:
                model.patch_embed_rgb.proj.weight.data.copy_(value)
        if "patch_embed.proj.bias" in key and model.patch_embed_rgb.proj.bias is not None:
            if model_state["patch_embed_rgb.proj.bias"].shape == value.shape:
                model.patch_embed_rgb.proj.bias.data.copy_(value)

        if "blocks." in key:
            suffix = key.split("blocks.", 1)[1]
            target_key = f"blocks.{suffix}"
            if target_key in model_state and model_state[target_key].shape == value.shape:
                model_state[target_key].copy_(value)
        if key.endswith("norm.weight") and "norm.weight" in model_state:
            if model_state["norm.weight"].shape == value.shape:
                model_state["norm.weight"].copy_(value)
        if key.endswith("norm.bias") and "norm.bias" in model_state:
            if model_state["norm.bias"].shape == value.shape:
                model_state["norm.bias"].copy_(value)

        if key.endswith("pos_embed"):
            pos_embed = value
            if pos_embed.ndim == 3 and pos_embed.shape[1] == model.patch_embed_rgb.num_patches + 1:
                patch_pos = pos_embed[:, 1:, :]
                model.pos_embed.data[:, 1:1 + patch_pos.shape[1], :].copy_(patch_pos)
                model.pos_embed.data[:, 1 + patch_pos.shape[1]:, :].copy_(patch_pos)

    print("Loaded WiLoR pretrained weights for ViT and patch embedding.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    distributed, rank, world_size, local_rank = init_distributed()
    if torch.cuda.is_available():
        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(config["runtime"]["device"])
    else:
        device = torch.device("cpu")

    train_loader, val_subset = create_dataloader(
        config,
        split="train",
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
        )

    model = GPGFormer(
        img_size=tuple(config["model"]["img_size"]),
        patch_size=config["model"]["patch_size"],
        embed_dim=config["model"]["embed_dim"],
        vit_depth=config["model"]["vit_depth"],
        vit_num_heads=config["model"]["vit_num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        attn_drop_rate=config["model"]["attn_drop_rate"],
        drop_path_rate=config["model"]["drop_path_rate"],
        joint_rep_type=config["model"]["joint_rep_type"],
        num_hand_joints=config["model"]["num_hand_joints"],
        focal_length=config["model"]["focal_length"],
        moge_checkpoint=config["moge"]["checkpoint"],
        moge_num_tokens=config["moge"]["num_tokens"],
        moge_use_fp16=config["moge"]["use_fp16"],
        mano_path=config["mano"]["model_path"],
        joint_regressor_extra=config["mano"].get("joint_regressor_extra"),
        bbox_scale=config["dataset"]["bbox_scale"]
    ).to(device)

    load_wilor_pretrained(model, config["model"]["pretrained_wilor"])
    _initialize_geom_embed(model, config, device)
    model.moge.eval()
    for param in model.moge.parameters():
        param.requires_grad = False

    detector = None
    if config["dataset"].get("use_detector", False):
        detector = HandDetector(
            config["dataset"]["detector_path"],
            conf=config["dataset"]["detector_conf"],
            iou=config["dataset"]["detector_iou"]
        ).to(device)
        detector.eval()
        for param in detector.parameters():
            param.requires_grad = False

    criterion = GPGFormerLoss(
        w_pose=config["loss"]["w_pose"],
        w_shape=config["loss"]["w_shape"],
        w_cam=config["loss"]["w_cam"],
        w_joints_2d=config["loss"].get("w_2d", 0.0),
        w_joints_3d=config["loss"].get("w_3d_joint", 0.0),
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    model.train()
    writer = None
    if rank == 0:
        os.makedirs(config["training"]["log_dir"], exist_ok=True)
        writer = SummaryWriter(config["training"]["log_dir"])

    best_avg_metric = None
    total_steps = 0
    debug_printed = False

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            bboxes = batch.get("bbox")
            if bboxes is None and detector is not None and "image_bgr" in batch:
                box_list = []
                for img in batch["image_bgr"]:
                    if isinstance(img, torch.Tensor):
                        img_np = img.detach().cpu().numpy()
                    else:
                        img_np = img
                    dets = detector.detect(img_np)
                    h, w = img_np.shape[:2]
                    box_list.append(dets[0] if dets else [0, 0, w - 1, h - 1])
                bboxes = torch.tensor(box_list, dtype=torch.float32, device=device)
            if bboxes is None:
                bboxes = torch.tensor(
                    [[0, 0, rgb.shape[-1] - 1, rgb.shape[-2] - 1]] * rgb.shape[0],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                bboxes = bboxes.to(device)

            cam_param = batch.get("cam_param")
            if cam_param is not None:
                cam_param = cam_param.to(device)
            preds = model(rgb, bboxes, cam_param=cam_param)

            if not debug_printed and rank == 0 and epoch == 0 and preds.get("pred_joints") is not None and "joints_3d_gt" in batch:
                pred_joints = preds["pred_joints"].detach()
                gt_joints = batch["joints_3d_gt"].detach()
                print(f"\n[Debug Batch 0]")
                print(f"  GT 3D joints mean: {gt_joints.mean():.3f}, std: {gt_joints.std():.3f}")
                print(f"  GT 3D joints range: [{gt_joints.min():.3f}, {gt_joints.max():.3f}]")
                print(f"  Pred 3D joints mean: {pred_joints.mean():.3f}, std: {pred_joints.std():.3f}")
                print(f"  Pred 3D joints range: [{pred_joints.min():.3f}, {pred_joints.max():.3f}]")
                print(f"  GT root joint (batch 0): {gt_joints[0, 0, :]}")
                print(f"  Pred root joint (batch 0): {pred_joints[0, 0, :]}")
                if preds.get("pred_cam_t") is not None:
                    print(f"  Pred cam_t mean: {preds['pred_cam_t'].mean():.3f}, std: {preds['pred_cam_t'].std():.3f}")
                if "mano_trans" in batch:
                    cam_t_debug = batch.get("cam_t") if batch.get("cam_t") is not None else batch.get("mano_trans")
                    if cam_t_debug is not None:
                        cam_t_debug = cam_t_debug.detach()
                        print(f"  GT cam_t mean: {cam_t_debug.mean():.3f}, std: {cam_t_debug.std():.3f}")
                debug_printed = True

            cam_t = batch.get("cam_t")
            if cam_t is None and "mano_trans" in batch:
                cam_t = batch.get("mano_trans")
            joints_2d = None
            if "joint_img" in batch:
                joint_img = batch["joint_img"].to(device)
                if joint_img.dim() == 3:
                    conf = torch.ones_like(joint_img[..., :1])
                    joints_2d = torch.cat([joint_img[..., :2], conf], dim=-1)

            targets = {
                "mano_pose": batch.get("mano_pose").to(device) if "mano_pose" in batch else None,
                "mano_shape": batch.get("mano_shape").to(device) if "mano_shape" in batch else None,
                "cam_t": cam_t.to(device) if cam_t is not None else None,
                "joints_3d_gt": batch.get("joints_3d_gt").to(device) if "joints_3d_gt" in batch else None,
                "joints_2d": joints_2d,
            }

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_steps += 1
            if rank == 0:
                pbar.set_postfix(loss=float(loss))
                if writer is not None and total_steps % config["training"]["log_freq"] == 0:
                    writer.add_scalar("train/loss", float(loss), total_steps)
                if total_steps % 100 == 0:
                    with torch.no_grad():
                        breakdown = _compute_loss_breakdown(criterion, preds, targets)
                    print(
                        f"[Step {total_steps}] "
                        f"pose={breakdown['loss_pose']:.3f}, "
                        f"shape={breakdown['loss_shape']:.3f}, "
                        f"cam={breakdown['loss_cam']:.3f}, "
                        f"j2d={breakdown['loss_joints_2d']:.3f}, "
                        f"j3d={breakdown['loss_joints_3d']:.3f}, "
                        f"total={breakdown['total_loss']:.3f}"
                    )

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_metrics = test(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                writer,
                rank,
                distributed,
                detector,
            )
            if rank == 0:
                mpjpe = val_metrics.get("mpjpe", float("inf"))
                pa_mpjpe = val_metrics.get("pa_mpjpe", float("inf"))
                avg_metric = val_metrics.get("avg_metric", float("inf"))
                print(
                    f"Val {epoch}: MPJPE: {mpjpe:.3f} mm, PA-MPJPE: {pa_mpjpe:.3f} mm, "
                    f"Avg Metric: {avg_metric:.3f} mm"
                )
                if best_avg_metric is None or avg_metric < best_avg_metric:
                    best_avg_metric = avg_metric
                    ckpt_path = os.path.join(config["training"]["save_dir"], "gpgformer_best.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": _strip_moge_from_state_dict(model.state_dict()),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                            "mpjpe": mpjpe,
                            "pa_mpjpe": pa_mpjpe,
                            "avg_metric": avg_metric,
                        },
                        ckpt_path,
                    )
        model.train()

        if rank == 0 and (epoch + 1) % config["training"]["save_freq"] == 0:
            ckpt_path = os.path.join(config["training"]["save_dir"], f"gpgformer_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": _strip_moge_from_state_dict(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()

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

    best_val = None
    total_steps = 0

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

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None and (epoch + 1) % config["training"]["test_freq"] == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch}", disable=(rank != 0)):
                    rgb = batch["rgb"].to(device)
                    bboxes = batch.get("bbox")
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
                    val_loss += float(loss)
                    val_steps += 1
            val_loss = val_loss / max(val_steps, 1)
            if rank == 0 and writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
        model.train()

        if rank == 0 and (epoch + 1) % config["training"]["save_freq"] == 0:
            ckpt_path = os.path.join(config["training"]["save_dir"], f"gpgformer_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

from gpgformer.models import GPGFormer, GPGFormerConfig
from gpgformer.metrics.pose_metrics import compute_pa_mpjpe
from gpgformer.utils.distributed import all_reduce_sum, cleanup_distributed, is_main_process, setup_distributed


def build_dataset(cfg: dict):
    name = cfg["dataset"]["name"].lower()
    bbox_source = cfg["dataset"].get("bbox_source_eval", "detector")
    detector_path = cfg["paths"]["detector_ckpt"] if bbox_source == "detector" else None

    if name in ("dexycb", "dex-ycb"):
        from data.dex_ycb_dataset import DexYCBDataset

        return DexYCBDataset(
            setup=cfg["dataset"]["dexycb_setup"],
            split="test",
            root_dir=cfg["paths"]["dexycb_root"],
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
        )

    if name in ("ho3d",):
        from data.ho3d_dataset import HO3DDataset

        return HO3DDataset(
            data_split="test",
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
        )

    if name in ("freihand",):
        from data.freihand_dataset import FreiHANDDataset

        return FreiHANDDataset(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
        )

    raise ValueError(f"Unknown dataset.name: {cfg['dataset']['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None, help="torch.distributed backend (default: nccl if cuda else gloo)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dist_info = setup_distributed(backend=args.backend)
    distributed = dist_info.distributed
    device = dist_info.device if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(cfg)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("val_batch_size", 16)),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    model = GPGFormer(
        GPGFormerConfig(
            wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
            moge2_weights_path=cfg["paths"]["moge2_ckpt"],
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            focal_length=float(cfg["model"].get("focal_length", 5000.0)),
        )
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    mpjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    pampjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    n = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        it = tqdm(loader, desc="eval", disable=(not is_main_process()))
        for batch in it:
            img = batch["rgb"].to(device)
            cam_param = batch.get("cam_param", None)
            cam_param = cam_param.to(device) if cam_param is not None else None
            out = model(img, cam_param=cam_param)

            pred_t_m = out["pred_cam_t"]
            pred_j_mm = (out["pred_keypoints_3d"] + pred_t_m.unsqueeze(1)) * 1000.0

            gt_j_mm = batch["joints_3d_gt"].to(device)

            # MPJPE (mm)
            mpjpe = (pred_j_mm - gt_j_mm).norm(dim=-1).mean(dim=-1)  # (B,)
            mpjpe_sum += mpjpe.double().sum()

            # PA-MPJPE (mm)
            pamp = torch.from_numpy(compute_pa_mpjpe(pred_j_mm, gt_j_mm)).to(device=device)  # (B,)
            pampjpe_sum += pamp.double().sum()

            n += float(pred_j_mm.shape[0])

    if distributed:
        all_reduce_sum(mpjpe_sum)
        all_reduce_sum(pampjpe_sum)
        all_reduce_sum(n)

    if is_main_process():
        denom = float(max(float(n.item()), 1.0))
        print(f"MPJPE(mm): {float(mpjpe_sum.item()) / denom:.3f}")
        print(f"PA-MPJPE(mm): {float(pampjpe_sum.item()) / denom:.3f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()



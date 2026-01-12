from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
import yaml

# Ensure project root is on sys.path when running from scripts/ under torchrun.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpgformer.models import GPGFormer, GPGFormerConfig
from gpgformer.utils.distributed import cleanup_distributed, is_main_process, setup_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP smoke test: run 1 step forward/backward to catch OOM early.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    dist_info = setup_distributed(backend=args.backend)
    device = dist_info.device

    cfg = yaml.safe_load(Path(args.config).read_text())

    model = GPGFormer(
        GPGFormerConfig(
            wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
            moge2_weights_path=cfg["paths"]["moge2_ckpt"],
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            focal_length=float(cfg.get("model", {}).get("focal_length", 5000.0)),
            image_size=int(cfg.get("model", {}).get("image_size", cfg["dataset"].get("img_size", 256))),
        )
    ).to(device)

    # A tiny learnable head so backward has something to do (and catches graph issues)
    head = torch.nn.Linear(3, 1, bias=False).to(device)

    # Synthetic input (cropped RGB in [0,1])
    B = int(args.batch_size)
    img = torch.rand((B, 3, 256, 256), device=device, dtype=torch.float32)
    # Synthetic intrinsics after crop: (fx,fy,cx,cy) in pixels
    cam_param = torch.tensor([[600.0, 600.0, 128.0, 128.0]], device=device, dtype=torch.float32).repeat(B, 1)

    # Forward
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    out = model(img, cam_param=cam_param)
    loss = head(out["pred_cam_t"]).mean()

    # Backward
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        mem = torch.cuda.max_memory_allocated(device=device) / (1024**3)
    else:
        mem = 0.0

    if is_main_process():
        print(f"[smoke] ok: forward+backward 1 step. max_cuda_mem_allocated={mem:.2f} GiB")
        print(f"[smoke] rank0 pid={os.getpid()} world_size={dist_info.world_size} local_rank={dist_info.local_rank}")

    cleanup_distributed()


if __name__ == "__main__":
    main()



import argparse
import os
from pathlib import Path
import sys

import torch
import yaml

# Ensure project root is on sys.path when running from scripts/ under torchrun.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpgformer.models import GPGFormer, GPGFormerConfig
from gpgformer.utils.distributed import cleanup_distributed, is_main_process, setup_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP smoke test: run 1 step forward/backward to catch OOM early.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    dist_info = setup_distributed(backend=args.backend)
    device = dist_info.device

    cfg = yaml.safe_load(Path(args.config).read_text())

    model = GPGFormer(
        GPGFormerConfig(
            wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
            moge2_weights_path=cfg["paths"]["moge2_ckpt"],
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            focal_length=float(cfg.get("model", {}).get("focal_length", 5000.0)),
            image_size=int(cfg.get("model", {}).get("image_size", cfg["dataset"].get("img_size", 256))),
        )
    ).to(device)

    # A tiny learnable head so backward has something to do (and catches graph issues)
    head = torch.nn.Linear(3, 1, bias=False).to(device)

    # Synthetic input (cropped RGB in [0,1])
    B = int(args.batch_size)
    img = torch.rand((B, 3, 256, 256), device=device, dtype=torch.float32)
    # Synthetic intrinsics after crop: (fx,fy,cx,cy) in pixels
    cam_param = torch.tensor([[600.0, 600.0, 128.0, 128.0]], device=device, dtype=torch.float32).repeat(B, 1)

    # Forward
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    out = model(img, cam_param=cam_param)
    loss = head(out["pred_cam_t"]).mean()

    # Backward
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        mem = torch.cuda.max_memory_allocated(device=device) / (1024**3)
    else:
        mem = 0.0

    if is_main_process():
        print(f"[smoke] ok: forward+backward 1 step. max_cuda_mem_allocated={mem:.2f} GiB")
        print(f"[smoke] rank0 pid={os.getpid()} world_size={dist_info.world_size} local_rank={dist_info.local_rank}")

    cleanup_distributed()


if __name__ == "__main__":
    main()



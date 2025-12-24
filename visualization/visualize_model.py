import argparse
import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.hand_dataset import HandDataset
from src.datasets.ho3d_dataset import HO3DDataset
from src.datasets.freihand_dataset import FreiHANDDataset
from src.datasets.dex_ycb_dataset import DexYCBDataset
from src.models.gpgformer import GPGFormer
from src.utils.geometry import perspective_projection


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_dataset(config: dict, split: str):
    dataset_cfg = config["dataset"]
    name = dataset_cfg.get("name", "image_folder").lower()
    if name in ["ho3d", "freihand"] and split in ["train", "val"]:
        actual_split = "train"
    else:
        actual_split = split

    if name == "ho3d":
        dataset_img_size = dataset_cfg.get("img_size", 256)
        if isinstance(dataset_img_size, (list, tuple)):
            dataset_img_size = dataset_img_size[0]
        return HO3DDataset(
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
            train=False,
            aug_prob=config["augmentation"].get("aug_prob", 0.8),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=None,
        )
    if name == "freihand":
        dataset_img_size = dataset_cfg.get("img_size", 256)
        if isinstance(dataset_img_size, (list, tuple)):
            dataset_img_size = dataset_img_size[0]
        return FreiHANDDataset(
            root_dir=dataset_cfg["root_dir"],
            img_size=dataset_img_size,
            cube_size=dataset_cfg.get("cube_size", 280),
            train=False,
            color_factor=config["augmentation"].get("color_factor", 0.2),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=None,
        )
    if name in ["dexycb", "dex-ycb"]:
        dataset_img_size = dataset_cfg.get("img_size", 256)
        if isinstance(dataset_img_size, (list, tuple)):
            dataset_img_size = dataset_img_size[0]
        return DexYCBDataset(
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
            train=False,
            color_factor=config["augmentation"].get("color_factor", 0.2),
            aug_prob=config["augmentation"].get("aug_prob", 0.8),
            color_aug_prob=config["augmentation"].get("color_aug_prob", 0.6),
            align_wilor_aug=dataset_cfg.get("align_wilor_aug", False),
            wilor_aug_config=None,
        )

    normalize_mean = None
    normalize_std = None
    if dataset_cfg.get("normalize", "none") == "imagenet":
        normalize_mean = dataset_cfg.get("normalize_mean")
        normalize_std = dataset_cfg.get("normalize_std")
    resize_to = dataset_cfg.get("input_size", dataset_cfg.get("img_size"))
    return HandDataset(
        root_dir=dataset_cfg["root_dir"],
        annotation_path=dataset_cfg.get("annotation_path"),
        resize_to=resize_to,
        train=False,
        color_factor=0.0,
        color_aug_prob=0.0,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )


def draw_joints(image: np.ndarray, joints_2d: np.ndarray) -> np.ndarray:
    out = image.copy()
    for x, y in joints_2d:
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="vis_out")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    split = config["dataset"].get("split", "test")
    dataset = _build_dataset(config, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            bboxes = batch.get("bbox")
            if bboxes is None:
                bboxes = torch.tensor(
                    [[0, 0, rgb.shape[-1] - 1, rgb.shape[-2] - 1]],
                    dtype=torch.float32,
                    device=device
                )
            else:
                bboxes = bboxes.to(device)

            preds = model(rgb, bboxes)
            img_bgr = batch["image_bgr"][0]
            if isinstance(img_bgr, torch.Tensor):
                img_bgr = img_bgr.detach().cpu().numpy()
            img_h, img_w = img_bgr.shape[:2]

            if "pred_joints" in preds:
                joints = preds["pred_joints"][0].detach().cpu().numpy()
                cam_t = preds["pred_cam_t"][0].detach().cpu().numpy()
                focal = config["model"]["focal_length"]
                camera_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
                joints_t = joints + cam_t[None, :]
                joints_2d = joints_t[:, :2] / np.clip(joints_t[:, 2:3], 1e-6, None)
                joints_2d = joints_2d * focal + camera_center[None, :]
                vis = draw_joints(img_bgr, joints_2d)
            else:
                vis = img_bgr

            out_path = os.path.join(args.out_dir, os.path.basename(batch["image_path"][0]))
            cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    main()

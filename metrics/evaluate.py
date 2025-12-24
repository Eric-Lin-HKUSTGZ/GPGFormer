import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.hand_dataset import HandDataset
from src.datasets.ho3d_dataset import HO3DDataset
from src.datasets.freihand_dataset import FreiHANDDataset
from src.datasets.dex_ycb_dataset import DexYCBDataset
from src.models.gpgformer import GPGFormer
from src.losses.loss import GPGFormerLoss


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    split = config["dataset"].get("split", "test")
    dataset = _build_dataset(config, split)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    criterion = GPGFormerLoss(
        w_pose=config["loss"]["w_pose"],
        w_shape=config["loss"]["w_shape"],
        w_cam=config["loss"]["w_cam"]
    )

    total_loss = 0.0
    num = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            rgb = batch["rgb"].to(device)
            bboxes = batch.get("bbox")
            if bboxes is None:
                bboxes = torch.tensor(
                    [[0, 0, rgb.shape[-1] - 1, rgb.shape[-2] - 1]] * rgb.shape[0],
                    dtype=torch.float32,
                    device=device
                )
            else:
                bboxes = bboxes.to(device)

            preds = model(rgb, bboxes)
            cam_t = batch.get("cam_t")
            if cam_t is None and "mano_trans" in batch:
                cam_t = batch.get("mano_trans")
            targets = {
                "mano_pose": batch.get("mano_pose").to(device) if "mano_pose" in batch else None,
                "mano_shape": batch.get("mano_shape").to(device) if "mano_shape" in batch else None,
                "cam_t": cam_t.to(device) if cam_t is not None else None,
            }
            loss = criterion(preds, targets)
            total_loss += float(loss)
            num += 1

    print(f"Mean loss: {total_loss / max(num, 1):.6f}")


if __name__ == "__main__":
    main()

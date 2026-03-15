from __future__ import annotations

import argparse
import os.path as osp
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
        use_ho3d_json = bool(cfg["dataset"].get("ho3d_use_json_split", False))
        ho3d_train_json = str(cfg["paths"].get("ho3d_train_json", "")).strip()
        ho3d_test_json = str(cfg["paths"].get("ho3d_test_json", "")).strip()

        # For HO3D, the public "evaluation/test" split may not contain full 21-joint GT.
        # Allow overriding for meaningful metric computation.
        ho3d_eval_split = str(cfg["dataset"].get("ho3d_eval_split", "val")).lower()
        if ho3d_eval_split in ("evaluation", "test"):
            # Keep as-is (useful for generating predictions), but metrics like PA-MPJPE can be meaningless.
            pass

        if use_ho3d_json:
            if ho3d_eval_split in ("train", "val", "train_all"):
                required_json = ho3d_train_json
                required_json_key = "paths.ho3d_train_json"
            else:
                required_json = ho3d_test_json
                required_json_key = "paths.ho3d_test_json"

            config_hint = str(cfg.get("__config_path__", "<unknown>"))
            if not required_json:
                raise FileNotFoundError(
                    f"dataset.name=ho3d with dataset.ho3d_use_json_split=true, "
                    f"but {required_json_key} is empty for split '{ho3d_eval_split}'. "
                    f"config={config_hint}"
                )
            if not osp.exists(required_json):
                raise FileNotFoundError(
                    f"dataset.name=ho3d with dataset.ho3d_use_json_split=true, "
                    f"but {required_json_key} does not exist: {required_json}. "
                    f"split='{ho3d_eval_split}', config={config_hint}"
                )

            from data.ho3d_json_dataset import HO3DJsonDataset
            return HO3DJsonDataset(
                data_split=ho3d_eval_split,
                root_dir=cfg["paths"]["ho3d_root"],
                train_json_path=ho3d_train_json,
                test_json_path=ho3d_test_json,
                img_size=int(cfg["dataset"].get("img_size", 256)),
                train=False,
                align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
                wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
                bbox_source=bbox_source,
                detector_weights_path=detector_path,
                trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
                trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
                trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
                root_index=int(cfg["dataset"].get("root_index", 9)),
                json_kp3d_unit=str(cfg["dataset"].get("ho3d_json_kp3d_unit", "auto")),
                json_kp3d_scale=float(cfg["dataset"].get("ho3d_json_kp3d_scale", 1.0)),
                json_convert_xyz=bool(cfg["dataset"].get("ho3d_json_convert_xyz", False)),
            )
        from data.ho3d_dataset import HO3DDataset
        return HO3DDataset(
            data_split=ho3d_eval_split,
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
            trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
        )

    if name in ("freihand",):
        from data.freihand_dataset_v3 import FreiHANDDatasetV3

        return FreiHANDDatasetV3(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=int(cfg.get("dataset", {}).get("root_index", 9)),
            use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
            trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
        )

    raise ValueError(f"Unknown dataset.name: {cfg['dataset']['name']}")


def build_model_from_cfg(cfg: dict) -> GPGFormer:
    model_cfg = cfg.get("model", {})
    refiner_cfg = model_cfg.get("feature_refiner", {})
    moge2_num_tokens = int(model_cfg.get("moge2_num_tokens", 400))
    if moge2_num_tokens <= 0:
        raise ValueError(f"model.moge2_num_tokens must be a positive int, got {moge2_num_tokens}")

    return GPGFormer(
        GPGFormerConfig(
            wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
            moge2_weights_path=cfg["paths"].get("moge2_ckpt", None),
            use_geo_prior=bool(model_cfg.get("use_geo_prior", True)),
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            mano_decoder=str(model_cfg.get("mano_decoder", "wilor")),
            freihand_mano_root=model_cfg.get("freihand_mano_root", None),
            focal_length=float(model_cfg.get("focal_length", 5000.0)),
            mano_head_ief_iters=int(model_cfg.get("mano_head", {}).get("ief_iters", 3)),
            mano_head_transformer_input=str(model_cfg.get("mano_head", {}).get("transformer_input", "mean_shape")),
            mano_head_dim=int(model_cfg.get("mano_head", {}).get("dim", 1024)),
            mano_head_depth=int(model_cfg.get("mano_head", {}).get("depth", 6)),
            mano_head_heads=int(model_cfg.get("mano_head", {}).get("heads", 8)),
            mano_head_dim_head=int(model_cfg.get("mano_head", {}).get("dim_head", 64)),
            mano_head_mlp_dim=int(model_cfg.get("mano_head", {}).get("mlp_dim", 2048)),
            mano_head_dropout=float(model_cfg.get("mano_head", {}).get("dropout", 0.0)),
            moge2_num_tokens=moge2_num_tokens,
            moge2_output=str(model_cfg.get("moge2_output", "neck")),
            token_fusion_mode=str(model_cfg.get("token_fusion_mode", "concat")),
            sum_fusion_strategy=str(model_cfg.get("sum_fusion_strategy", "basic")),
            fusion_proj_zero_init=bool(model_cfg.get("fusion_proj_zero_init", True)),
            cross_attn_num_heads=int(model_cfg.get("cross_attn_num_heads", 8)),
            cross_attn_dropout=float(model_cfg.get("cross_attn_dropout", 0.0)),
            cross_attn_gate_init=float(model_cfg.get("cross_attn_gate_init", 0.0)),
            geo_tokenizer_use_pooling=bool(model_cfg.get("geo_tokenizer_use_pooling", True)),
            feature_refiner_method=str(refiner_cfg.get("method", "none")),
            feature_refiner_feat_dim=int(refiner_cfg.get("feat_dim", 1280)),
            feature_refiner_sjta_bottleneck_dim=int(refiner_cfg.get("sjta_bottleneck_dim", 256)),
            feature_refiner_sjta_num_heads=int(refiner_cfg.get("sjta_num_heads", 4)),
            feature_refiner_sjta_use_2d_prior=bool(refiner_cfg.get("sjta_use_2d_prior", True)),
            feature_refiner_sjta_num_steps=int(refiner_cfg.get("sjta_num_steps", 2)),
            feature_refiner_coear_dilation1=int(refiner_cfg.get("coear_dilation1", 1)),
            feature_refiner_coear_dilation2=int(refiner_cfg.get("coear_dilation2", 2)),
            feature_refiner_coear_gate_reduction=int(refiner_cfg.get("coear_gate_reduction", 8)),
            feature_refiner_coear_init_alpha=float(refiner_cfg.get("coear_init_alpha", 0.1)),
            feature_refiner_wilor_msf_bottleneck_ratio=int(refiner_cfg.get("wilor_msf_bottleneck_ratio", 4)),
            feature_refiner_wilor_msf_dilation1=int(refiner_cfg.get("wilor_msf_dilation1", 1)),
            feature_refiner_wilor_msf_dilation2=int(refiner_cfg.get("wilor_msf_dilation2", 2)),
            feature_refiner_wilor_msf_dilation3=int(refiner_cfg.get("wilor_msf_dilation3", 3)),
            feature_refiner_wilor_msf_gate_reduction=int(refiner_cfg.get("wilor_msf_gate_reduction", 8)),
            feature_refiner_wilor_msf_init_alpha=float(refiner_cfg.get("wilor_msf_init_alpha", 0.1)),
            feature_refiner_kcr_num_keypoints=int(refiner_cfg.get("kcr_num_keypoints", 21)),
            feature_refiner_kcr_hidden_dim=int(refiner_cfg.get("kcr_hidden_dim", 128)),
        )
    )


def warmup_lazy_modules(model: GPGFormer, cfg: dict, device: torch.device) -> None:
    model_cfg = cfg.get("model", {})
    h = int(model_cfg.get("image_size", 256))
    w = int(model_cfg.get("image_width", int(h * 0.75)))
    # Avoid `torch.inference_mode()` here: lazily-created parameters become "inference tensors"
    # and `load_state_dict()` (which does in-place copies) will error outside InferenceMode.
    with torch.no_grad():
        img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
        cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
        _ = model(img_dummy, cam_param=cam_dummy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None, help="torch.distributed backend (default: nccl if cuda else gloo)")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    cfg["__config_path__"] = str(config_path)
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))
    dist_info = setup_distributed(backend=args.backend)
    distributed = dist_info.distributed
    device = dist_info.device if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    moge2_num_tokens = int(model_cfg.get("moge2_num_tokens", 400))
    if moge2_num_tokens <= 0:
        raise ValueError(f"model.moge2_num_tokens must be a positive int, got {moge2_num_tokens}")
    if is_main_process():
        dataset_name = str(cfg.get("dataset", {}).get("name", ""))
        ho3d_json_flag = bool(cfg.get("dataset", {}).get("ho3d_use_json_split", False))
        print(f"[info] config={config_path} dataset.name={dataset_name} ho3d_use_json_split={ho3d_json_flag}")
        moge2_output = str(model_cfg.get("moge2_output", "neck"))
        use_geo_prior = bool(model_cfg.get("use_geo_prior", True))
        print(f"[info] use_geo_prior={use_geo_prior} moge2_output={moge2_output} moge2_num_tokens={moge2_num_tokens}")

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

    model = build_model_from_cfg(cfg).to(device)
    # Ensure lazily created modules (e.g., geo_tokenizer) exist before loading checkpoint.
    warmup_lazy_modules(model, cfg, device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main_process():
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    mpjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    pampjpe_sum = torch.zeros((), device=device, dtype=torch.float64)
    n = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        it = tqdm(loader, desc="eval", disable=(not is_main_process()))
        warned_degenerate_gt = False
        for batch in it:
            img = batch["rgb"].to(device)
            cam_param = batch.get("cam_param", None)
            cam_param = cam_param.to(device) if cam_param is not None else None
            out = model(img, cam_param=cam_param)

            # Use raw MANO-frame joints (meters), consistent with train.py evaluate_epoch.
            # Do NOT add pred_cam_t — it is only for perspective projection (HaMeR convention).
            pred_j_m = out["pred_keypoints_3d"]

            gt_j_m = batch.get("keypoints_3d", None)
            if gt_j_m is None:
                gt_j_m = batch.get("joints_3d_gt", None)
            if gt_j_m is None:
                raise KeyError("Batch missing both 'keypoints_3d' and 'joints_3d_gt'.")
            gt_j_m = gt_j_m.to(device)

            # Filter invalid/degenerate GT
            gt_var = gt_j_m.var(dim=(1, 2))
            finite_mask = torch.isfinite(gt_j_m).all(dim=(1, 2))
            valid_mask = (gt_var > 1e-8) & finite_mask
            if not bool(valid_mask.any()):
                if (not warned_degenerate_gt) and is_main_process():
                    print("[warn] Skipping eval batch with degenerate/invalid GT joints; metrics are not meaningful for this split.")
                    warned_degenerate_gt = True
                continue

            pred_j_m = pred_j_m[valid_mask]
            gt_j_m = gt_j_m[valid_mask]

            # Root-center both pred and GT (standard root-relative MPJPE)
            ri = int(root_index)
            pred_j_m = pred_j_m - pred_j_m[:, [ri]]
            gt_j_m = gt_j_m - gt_j_m[:, [ri]]

            # MPJPE (meters)
            mpjpe = (pred_j_m - gt_j_m).norm(dim=-1).mean(dim=-1)  # (B_valid,)
            mpjpe_sum += mpjpe.double().sum()

            # PA-MPJPE (meters)
            pamp = torch.from_numpy(compute_pa_mpjpe(pred_j_m, gt_j_m)).to(device=device)  # (B_valid,)
            pampjpe_sum += pamp.double().sum()

            n += float(pred_j_m.shape[0])

    if distributed:
        all_reduce_sum(mpjpe_sum)
        all_reduce_sum(pampjpe_sum)
        all_reduce_sum(n)

    if is_main_process():
        denom = float(n.item())
        if denom <= 0:
            print("MPJPE(mm): nan")
            print("PA-MPJPE(mm): nan")
        else:
            # Convert metrics from meters to mm for display
            mpjpe_mm = float(mpjpe_sum.item()) / denom * 1000.0
            pampjpe_mm = float(pampjpe_sum.item()) / denom * 1000.0
            print(f"MPJPE(mm): {mpjpe_mm:.3f}")
            print(f"PA-MPJPE(mm): {pampjpe_mm:.3f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

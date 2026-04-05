#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dex_ycb_dataset import DexYCBDataset
from data.freihand_dataset_v3 import FreiHANDDatasetV3
from data.ho3d_dataset import HO3DDataset
from gpgformer.models import GPGFormer, GPGFormerConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

DEFAULT_MULTIMODAL_CKPTS = {
    "freihand": "/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt",
    "ho3d": "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/ho3d_20260319/ho3d/gpgformer_best.pt",
    "dexycb": "/root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt",
}
DEFAULT_RGB_ONLY_CKPTS = {
    "freihand": "/root/code/vepfs/GPGFormer/checkpoints/freihand_20260304_rgb_only/freihand/gpgformer_best.pt",
}


@dataclass
class HeatmapView:
    key: str
    title: str
    token_hw: Tuple[int, int]
    heatmap: np.ndarray
    metrics: Dict[str, Any]


@dataclass
class TokenVisualizationBundle:
    dataset_name: str
    model_label: str
    rgb_only_label: str
    input_image: np.ndarray
    dark_image: np.ndarray
    hand_bbox_xyxy: np.ndarray
    hand_mask: np.ndarray
    sample_identifier: str
    source_image_path: Optional[str]
    energy_views: List[HeatmapView]
    analysis_views: List[HeatmapView]


class TokenFlowRecorder:
    """
    Capture intermediate tensors around the encoder token flow.

    The five requested visualizations correspond to these stages:
    1. `rgb_patch_tokens_pre_fusion`: output of `patch_embed(img)` before multimodal fusion.
    2. `geo_tokens_encoder_input`: geometry tokens actually sent into the encoder.
       This is after MoGe2 -> optional GeoSideAdapter -> GeoTokenizer -> optional GeoSideTuning.
    3. `pre_backbone_tokens`: token sequence right before Transformer block 0.
       For the current sum+channel_concat checkpoints, this is the fused RGB+geometry token map.
       For concat-style checkpoints, we later slice out the patch-token tail for a comparable heatmap.
    4. `encoder_img_feat`: final encoder output `img_feat` before any optional feature-refiner module.
    """

    def __init__(self, model: GPGFormer):
        self.model = model
        self.cache: Dict[str, Any] = {}
        self.handles = []

        self._orig_encoder_forward = model.encoder.forward
        self._orig_geo_forward = model.geo_tokenizer.forward if getattr(model, "geo_tokenizer", None) is not None else None

        if self._orig_geo_forward is not None:
            def geo_forward_wrapper(feat: torch.Tensor):
                tokens, coords = self._orig_geo_forward(feat)
                self.cache["geo_tokens_raw"] = tokens.detach().cpu()
                self.cache["geo_coords"] = coords.detach().cpu()
                return tokens, coords

            model.geo_tokenizer.forward = geo_forward_wrapper

        def encoder_forward_wrapper(img: torch.Tensor, geo_tokens=None, geo_pos=None, **kwargs):
            if geo_tokens is not None:
                self.cache["geo_tokens_encoder_input"] = geo_tokens.detach().cpu()
            if geo_pos is not None:
                self.cache["geo_pos"] = geo_pos.detach().cpu()
            out = self._orig_encoder_forward(img, geo_tokens=geo_tokens, geo_pos=geo_pos, **kwargs)
            if isinstance(out, dict) and "img_feat" in out:
                self.cache["encoder_img_feat"] = out["img_feat"].detach().cpu()
            return out

        model.encoder.forward = encoder_forward_wrapper

        patch_embed = model.encoder.backbone.patch_embed
        self.handles.append(patch_embed.register_forward_hook(self._patch_embed_hook))

        if hasattr(model.encoder, "patch_norm"):
            self.handles.append(model.encoder.patch_norm.register_forward_hook(self._patch_norm_hook))
        if hasattr(model.encoder, "geo_norm"):
            self.handles.append(model.encoder.geo_norm.register_forward_hook(self._geo_norm_hook))

        blocks = getattr(model.encoder.backbone, "blocks", None)
        if blocks and len(blocks) > 0:
            self.handles.append(blocks[0].register_forward_pre_hook(self._first_block_pre_hook))

    def _patch_embed_hook(self, _module, _args, output) -> None:
        patch_tokens, hw = output
        self.cache["rgb_patch_tokens_pre_fusion"] = patch_tokens.detach().cpu()
        self.cache["rgb_token_hw"] = (int(hw[0]), int(hw[1]))

    def _first_block_pre_hook(self, _module, inputs) -> None:
        if inputs:
            self.cache["pre_backbone_tokens"] = inputs[0].detach().cpu()

    def _patch_norm_hook(self, _module, _args, output) -> None:
        self.cache["patch_norm_out"] = output.detach().cpu()

    def _geo_norm_hook(self, _module, _args, output) -> None:
        self.cache["geo_norm_out"] = output.detach().cpu()

    def remove(self) -> None:
        if self._orig_geo_forward is not None and getattr(self.model, "geo_tokenizer", None) is not None:
            self.model.geo_tokenizer.forward = self._orig_geo_forward
        self.model.encoder.forward = self._orig_encoder_forward
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GPGFormer token heatmaps across RGB, geometry, fusion, and final-backbone stages."
    )
    parser.add_argument("--dataset", choices=["freihand", "ho3d", "dexycb"], default="freihand")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--rgb-checkpoint",
        type=str,
        default=None,
        help="Optional true RGB-only checkpoint. If omitted, the script reuses the multimodal checkpoint and disables geometry at inference time.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["train", "eval", "test", "val", "evaluation"], default="eval")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--heatmap-reducer",
        "--magnitude-reducer",
        dest="heatmap_reducer",
        choices=["l2", "l1", "var"],
        default="l2",
    )
    parser.add_argument("--heatmap-cmap", choices=["jet", "viridis", "magma", "turbo"], default="jet")
    parser.add_argument("--overlay-alpha", type=float, default=0.58)
    parser.add_argument("--darken-factor", type=float, default=0.38)
    parser.add_argument("--show-bbox", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "sample"


def normalize_heatmap(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if denom > 1e-8:
        x = x / denom
    return x


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    image_uint8 = np.uint8(np.clip(image_rgb, 0.0, 1.0) * 255.0)
    cv2.imwrite(str(path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))


def draw_bbox_on_image(image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    image_u8 = np.uint8(np.clip(image_rgb, 0.0, 1.0) * 255.0)
    x1, y1, x2, y2 = bbox_xyxy.astype(np.int64).tolist()
    cv2.rectangle(image_u8, (x1, y1), (max(x1 + 1, x2 - 1), max(y1 + 1, y2 - 1)), (255, 255, 255), 2)
    return image_u8.astype(np.float32) / 255.0


def maybe_draw_bbox(image_rgb: np.ndarray, bbox_xyxy: np.ndarray, show_bbox: bool) -> np.ndarray:
    if not show_bbox:
        return image_rgb
    return draw_bbox_on_image(image_rgb, bbox_xyxy)


def denormalize_rgb(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float()
    if img.ndim == 4:
        img = img[0]
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def darken_image(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    factor = float(np.clip(factor, 0.0, 1.0))
    return np.clip(image_rgb.astype(np.float32) * factor, 0.0, 1.0)


def resolve_dataset_split(dataset_name: str, user_split: str) -> str:
    if dataset_name == "freihand":
        return "train" if user_split == "train" else "eval"
    if dataset_name == "ho3d":
        return "train" if user_split == "train" else "evaluation"
    if dataset_name == "dexycb":
        return "train" if user_split == "train" else "test"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_freihand_dataset(cfg: dict, split: str) -> FreiHANDDatasetV3:
    ds_cfg = dict(cfg["dataset"])
    paths = cfg["paths"]
    train = split == "train"
    use_trainval_split = bool(ds_cfg.get("use_trainval_split", True)) if train else False
    return FreiHANDDatasetV3(
        root_dir=paths["freihand_root"],
        eval_root=paths.get("freihand_eval_root", None),
        img_size=int(ds_cfg.get("img_size", 256)),
        train=train,
        align_wilor_aug=bool(ds_cfg.get("align_wilor_aug", True)),
        wilor_aug_config=ds_cfg.get("wilor_aug_config", {}),
        root_index=int(ds_cfg.get("root_index", 9)),
        trainval_ratio=float(ds_cfg.get("trainval_ratio", 0.9)),
        trainval_seed=int(ds_cfg.get("trainval_seed", 42)),
        use_trainval_split=use_trainval_split,
        load_vertices_gt=True,
    )


def build_ho3d_dataset(cfg: dict, split: str) -> HO3DDataset:
    ds_cfg = dict(cfg["dataset"])
    paths = cfg["paths"]
    return HO3DDataset(
        data_split=split,
        root_dir=paths["ho3d_root"],
        dataset_version=str(ds_cfg.get("ho3d_version", "v3")),
        img_size=int(ds_cfg.get("img_size", 256)),
        train=split == "train",
        align_wilor_aug=bool(ds_cfg.get("align_wilor_aug", True)),
        wilor_aug_config=ds_cfg.get("wilor_aug_config", {}),
        bbox_source="gt" if split == "train" else str(ds_cfg.get("bbox_source_eval", "gt")),
        train_split_file=paths.get("ho3d_train_split_file", None),
        eval_split_file=paths.get("ho3d_eval_split_file", None),
        eval_xyz_json=paths.get("ho3d_eval_xyz_json", None),
        eval_verts_json=paths.get("ho3d_eval_verts_json", None),
        root_index=int(ds_cfg.get("root_index", 9)),
    )


def build_dexycb_dataset(cfg: dict, split: str) -> DexYCBDataset:
    ds_cfg = dict(cfg["dataset"])
    paths = cfg["paths"]
    return DexYCBDataset(
        setup=str(ds_cfg.get("dexycb_setup", "s0")),
        split=split,
        root_dir=paths["dexycb_root"],
        img_size=int(ds_cfg.get("img_size", 256)),
        train=split == "train",
        align_wilor_aug=bool(ds_cfg.get("align_wilor_aug", True)),
        wilor_aug_config=ds_cfg.get("wilor_aug_config", {}),
        bbox_source="gt" if split == "train" else str(ds_cfg.get("bbox_source_eval", "gt")),
        root_index=int(ds_cfg.get("root_index", 9)),
        mano_root=paths.get("dexycb_mano_root", paths.get("mano_dir", None)),
        mano_pose_is_pca=bool(ds_cfg.get("mano_pose_is_pca", True)),
    )


def build_dataset(cfg: dict, dataset_name: str, split: str):
    if dataset_name == "freihand":
        return build_freihand_dataset(cfg, split)
    if dataset_name == "ho3d":
        return build_ho3d_dataset(cfg, split)
    if dataset_name == "dexycb":
        return build_dexycb_dataset(cfg, split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def select_sample_indices(dataset, split: str, args: argparse.Namespace) -> List[int]:
    if args.sample_indices:
        return list(args.sample_indices)
    total = len(dataset)
    count = min(max(int(args.num_samples), 0), total)
    if split != "train":
        return list(range(count))
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(total, size=count, replace=False).tolist())


def build_model_from_cfg(cfg: dict, device: torch.device) -> GPGFormer:
    model_cfg = cfg["model"]
    paths = cfg["paths"]
    side_tuning_cfg = model_cfg.get("side_tuning", {})
    geo_side_adapter_cfg = model_cfg.get("geo_side_adapter", {})
    feature_refiner_cfg = model_cfg.get("feature_refiner", {})
    image_size = int(model_cfg.get("image_size", cfg.get("dataset", {}).get("img_size", 256)))
    model = GPGFormer(
        GPGFormerConfig(
            backbone_type=str(model_cfg.get("backbone_type", "wilor")),
            wilor_ckpt_path=paths.get("wilor_ckpt", ""),
            vitpose_ckpt_path=paths.get("vitpose_ckpt", ""),
            mano_model_path=paths["mano_dir"],
            mano_mean_params=paths["mano_mean_params"],
            moge2_weights_path=paths.get("moge2_ckpt", None),
            use_geo_prior=bool(model_cfg.get("use_geo_prior", True)),
            image_size=image_size,
            image_hw=(image_size, int(round(image_size * 0.75))),
            focal_length=float(model_cfg.get("focal_length", 5000.0)),
            moge2_num_tokens=int(model_cfg.get("moge2_num_tokens", 400)),
            moge2_output=str(model_cfg.get("moge2_output", "neck")),
            token_fusion_mode=str(model_cfg.get("token_fusion_mode", "concat")),
            sum_fusion_strategy=str(model_cfg.get("sum_fusion_strategy", "basic")),
            sum_geo_gate_init=float(model_cfg.get("sum_geo_gate_init", 4.0)),
            fusion_proj_zero_init=bool(model_cfg.get("fusion_proj_zero_init", True)),
            fusion_proj_init_mode=str(model_cfg.get("fusion_proj_init_mode", "tiny")),
            cross_attn_num_heads=int(model_cfg.get("cross_attn_num_heads", 8)),
            cross_attn_dropout=float(model_cfg.get("cross_attn_dropout", 0.0)),
            cross_attn_gate_init=float(model_cfg.get("cross_attn_gate_init", 0.0)),
            geo_tokenizer_use_pooling=bool(model_cfg.get("geo_tokenizer_use_pooling", True)),
            use_geo_side_tuning=bool(side_tuning_cfg.get("enabled", False)),
            geo_side_tuning_side_channels=int(side_tuning_cfg.get("side_channels", 256)),
            geo_side_tuning_dropout=float(side_tuning_cfg.get("dropout", 0.1)),
            geo_side_tuning_max_res_scale=float(side_tuning_cfg.get("max_res_scale", 0.1)),
            geo_side_tuning_init_res_scale=float(side_tuning_cfg.get("init_res_scale", 1e-3)),
            use_geo_side_adapter=bool(geo_side_adapter_cfg.get("enabled", False)),
            geo_side_adapter_side_channels=int(geo_side_adapter_cfg.get("side_channels", 256)),
            geo_side_adapter_depth=int(geo_side_adapter_cfg.get("depth", 3)),
            geo_side_adapter_dropout=float(geo_side_adapter_cfg.get("dropout", 0.05)),
            geo_side_adapter_norm_groups=int(geo_side_adapter_cfg.get("norm_groups", 32)),
            geo_branch_dropout_prob=float(model_cfg.get("geo_branch_dropout_prob", 0.0)),
            mano_decoder=str(model_cfg.get("mano_decoder", "wilor")),
            freihand_mano_root=model_cfg.get("freihand_mano_root", None),
            mano_head_ief_iters=int(model_cfg.get("mano_head", {}).get("ief_iters", 3)),
            mano_head_transformer_input=str(model_cfg.get("mano_head", {}).get("transformer_input", "mean_shape")),
            mano_head_dim=int(model_cfg.get("mano_head", {}).get("dim", 1024)),
            mano_head_depth=int(model_cfg.get("mano_head", {}).get("depth", 6)),
            mano_head_heads=int(model_cfg.get("mano_head", {}).get("heads", 8)),
            mano_head_dim_head=int(model_cfg.get("mano_head", {}).get("dim_head", 64)),
            mano_head_mlp_dim=int(model_cfg.get("mano_head", {}).get("mlp_dim", 2048)),
            mano_head_dropout=float(model_cfg.get("mano_head", {}).get("dropout", 0.0)),
            feature_refiner_method=str(feature_refiner_cfg.get("method", "none")),
            feature_refiner_feat_dim=int(feature_refiner_cfg.get("feat_dim", 1280)),
            feature_refiner_sjta_bottleneck_dim=int(feature_refiner_cfg.get("sjta_bottleneck_dim", 256)),
            feature_refiner_sjta_num_heads=int(feature_refiner_cfg.get("sjta_num_heads", 4)),
            feature_refiner_sjta_use_2d_prior=bool(feature_refiner_cfg.get("sjta_use_2d_prior", True)),
            feature_refiner_sjta_num_steps=int(feature_refiner_cfg.get("sjta_num_steps", 2)),
            feature_refiner_coear_dilation1=int(feature_refiner_cfg.get("coear_dilation1", 1)),
            feature_refiner_coear_dilation2=int(feature_refiner_cfg.get("coear_dilation2", 2)),
            feature_refiner_coear_gate_reduction=int(feature_refiner_cfg.get("coear_gate_reduction", 8)),
            feature_refiner_coear_init_alpha=float(feature_refiner_cfg.get("coear_init_alpha", 0.1)),
            feature_refiner_wilor_msf_bottleneck_ratio=int(feature_refiner_cfg.get("wilor_msf_bottleneck_ratio", 4)),
            feature_refiner_wilor_msf_dilation1=int(feature_refiner_cfg.get("wilor_msf_dilation1", 1)),
            feature_refiner_wilor_msf_dilation2=int(feature_refiner_cfg.get("wilor_msf_dilation2", 2)),
            feature_refiner_wilor_msf_dilation3=int(feature_refiner_cfg.get("wilor_msf_dilation3", 3)),
            feature_refiner_wilor_msf_gate_reduction=int(feature_refiner_cfg.get("wilor_msf_gate_reduction", 8)),
            feature_refiner_wilor_msf_init_alpha=float(feature_refiner_cfg.get("wilor_msf_init_alpha", 0.1)),
            feature_refiner_kcr_num_keypoints=int(feature_refiner_cfg.get("kcr_num_keypoints", 21)),
            feature_refiner_kcr_hidden_dim=int(feature_refiner_cfg.get("kcr_hidden_dim", 128)),
        )
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def warmup_model(model: GPGFormer, device: torch.device) -> None:
    h = int(getattr(model.cfg, "image_size", 256))
    w = int(round(h * 0.75))
    img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
    cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
    _ = model(img_dummy, cam_param=cam_dummy)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model_bundle(ckpt_path: str, device: torch.device) -> tuple[GPGFormer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg") or ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint cfg is missing or invalid: {ckpt_path}")
    model = build_model_from_cfg(cfg, device)
    warmup_model(model, device)
    state_dict = strip_module_prefix(ckpt.get("model", ckpt.get("state_dict", {})))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading {ckpt_path}: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading {ckpt_path}: {len(unexpected)}")
    model.eval()
    return model, cfg


def hand_bbox_from_sample(sample: Dict[str, Any], img_h: int, img_w: int, margin: float = 8.0) -> np.ndarray:
    keypoints = as_numpy(sample["keypoints_2d"])
    uv_valid = as_numpy(sample["uv_valid"])
    pts = keypoints[uv_valid > 0.5]
    if pts.shape[0] == 0:
        pts = keypoints
    px = np.stack([(pts[:, 0] + 0.5) * float(img_w), (pts[:, 1] + 0.5) * float(img_h)], axis=-1)
    min_xy = px.min(axis=0) - margin
    max_xy = px.max(axis=0) + margin
    x1 = float(np.clip(min_xy[0], 0.0, img_w - 1.0))
    y1 = float(np.clip(min_xy[1], 0.0, img_h - 1.0))
    x2 = float(np.clip(max_xy[0], x1 + 1.0, img_w))
    y2 = float(np.clip(max_xy[1], y1 + 1.0, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def keypoints_px_from_sample(sample: Dict[str, Any], img_h: int, img_w: int) -> tuple[np.ndarray, np.ndarray]:
    keypoints = as_numpy(sample["keypoints_2d"]).astype(np.float32)
    uv_valid = as_numpy(sample["uv_valid"]).astype(np.float32).reshape(-1)
    points_px = np.stack(
        [
            (keypoints[:, 0] + 0.5) * float(img_w),
            (keypoints[:, 1] + 0.5) * float(img_h),
        ],
        axis=-1,
    )
    return points_px, uv_valid


def build_hand_region_mask(sample: Dict[str, Any], img_h: int, img_w: int, bbox_xyxy: np.ndarray) -> np.ndarray:
    """
    Build an approximate hand region mask for quantitative analysis.

    We intentionally do not use the full image bbox as the supervision region because bbox-based
    metrics are too loose and let large background areas leak in. Instead we approximate the hand
    silhouette from visible 2D joints using a convex hull plus dilation. This yields a much tighter
    region for judging whether a heatmap is actually concentrating on the hand.
    """

    points_px, uv_valid = keypoints_px_from_sample(sample, img_h, img_w)
    valid_points = points_px[uv_valid > 0.5]
    if valid_points.shape[0] < 3:
        return bbox_mask(img_h, img_w, bbox_xyxy)

    hull = cv2.convexHull(valid_points.astype(np.float32)).astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)

    bbox_w = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
    bbox_h = max(1.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
    dilation_radius = max(3, int(round(max(bbox_w, bbox_h) * 0.06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def bbox_mask(img_h: int, img_w: int, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.int64).tolist()
    x1 = int(np.clip(x1, 0, img_w - 1))
    y1 = int(np.clip(y1, 0, img_h - 1))
    x2 = int(np.clip(x2, x1 + 1, img_w))
    y2 = int(np.clip(y2, y1 + 1, img_h))
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def heatmap_mass_in_bbox(heatmap: np.ndarray, bbox_xyxy: np.ndarray) -> float:
    positive = np.maximum(np.asarray(heatmap, dtype=np.float32), 0.0)
    denom = float(positive.sum())
    if denom <= 1e-8:
        return float("nan")
    mask = bbox_mask(positive.shape[0], positive.shape[1], bbox_xyxy)
    return float(positive[mask].sum() / denom)


def heatmap_mass_in_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    positive = np.maximum(np.asarray(heatmap, dtype=np.float32), 0.0)
    mask = np.asarray(mask, dtype=bool)
    denom = float(positive.sum())
    if denom <= 1e-8:
        return float("nan")
    return float(positive[mask].sum() / denom)


def summarize_focus_values(values: np.ndarray, focus_mask: np.ndarray, topk_ratio: float = 0.1) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    focus_mask = np.asarray(focus_mask, dtype=bool)
    inside = values[focus_mask]
    outside = values[~focus_mask]
    flat = values.reshape(-1)
    focus_flat = focus_mask.reshape(-1)
    if flat.size == 0:
        topk_inside_ratio = float("nan")
    else:
        k = max(1, int(round(flat.size * float(topk_ratio))))
        top_idx = np.argpartition(flat, -k)[-k:]
        topk_inside_ratio = float(np.mean(focus_flat[top_idx])) if top_idx.size else float("nan")
    return {
        "mean_inside_hand": float(np.mean(inside)) if inside.size else float("nan"),
        "mean_outside_hand": float(np.mean(outside)) if outside.size else float("nan"),
        "inside_outside_gap": float(np.mean(inside) - np.mean(outside)) if inside.size and outside.size else float("nan"),
        # `top10_inside_ratio` measures how many of the strongest 10% responses fall inside the hand region.
        # Larger is better because it means the most salient activations are hand-centered rather than background-driven.
        "top10_inside_ratio": topk_inside_ratio,
    }


def upsample_map(heatmap: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(img_h, img_w), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def tokens_to_spatial_map(tokens: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2:
        raise ValueError(f"Expected token matrix (N, C), got shape {tokens.shape}")
    if tokens.shape[0] != h * w:
        raise ValueError(f"Token length {tokens.shape[0]} does not match spatial size {h}x{w}")
    return tokens.reshape(h, w, tokens.shape[1]).transpose(2, 0, 1).astype(np.float32)


def feature_map_to_tokens(feature_map_chw: np.ndarray) -> np.ndarray:
    feature_map_chw = np.asarray(feature_map_chw, dtype=np.float32)
    return feature_map_chw.transpose(1, 2, 0).reshape(-1, feature_map_chw.shape[0]).astype(np.float32)


def resize_mask(mask: np.ndarray, hw: Tuple[int, int], threshold: float = 0.15) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()
    return resized > float(threshold)


def reduce_channel_energy(feature_map_chw: np.ndarray, reducer: str) -> np.ndarray:
    feature_map_chw = np.asarray(feature_map_chw, dtype=np.float32)
    if reducer == "l1":
        reduced = np.abs(feature_map_chw).sum(axis=0)
    elif reducer == "var":
        reduced = np.var(feature_map_chw, axis=0)
    else:
        reduced = np.sqrt(np.square(feature_map_chw).sum(axis=0))
    return reduced.astype(np.float32)


def compute_token_cosine_map(tokens: np.ndarray, prototype: np.ndarray, hw: Tuple[int, int], eps: float = 1e-6) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    prototype = np.asarray(prototype, dtype=np.float32).reshape(1, -1)
    token_norm = np.linalg.norm(tokens, axis=1, keepdims=True)
    proto_norm = max(float(np.linalg.norm(prototype)), eps)
    cosine = np.sum(tokens * prototype, axis=1, keepdims=False) / np.maximum(token_norm[:, 0] * proto_norm, eps)
    return cosine.reshape(hw[0], hw[1]).astype(np.float32)


def compute_cross_modal_cosine_distance(rgb_tokens: np.ndarray, geo_tokens: np.ndarray, hw: Tuple[int, int], eps: float = 1e-6) -> np.ndarray:
    rgb_tokens = np.asarray(rgb_tokens, dtype=np.float32)
    geo_tokens = np.asarray(geo_tokens, dtype=np.float32)
    if rgb_tokens.shape != geo_tokens.shape:
        raise ValueError(f"RGB tokens {rgb_tokens.shape} and geometry tokens {geo_tokens.shape} must match for cosine distance.")
    rgb_norm = np.linalg.norm(rgb_tokens, axis=1, keepdims=True)
    geo_norm = np.linalg.norm(geo_tokens, axis=1, keepdims=True)
    cosine = np.sum(rgb_tokens * geo_tokens, axis=1, keepdims=False) / np.maximum(rgb_norm[:, 0] * geo_norm[:, 0], eps)
    return (1.0 - np.clip(cosine, -1.0, 1.0)).reshape(hw[0], hw[1]).astype(np.float32)


def colorize_heatmap(heatmap: np.ndarray, cmap_name: str) -> np.ndarray:
    heatmap_u8 = np.uint8(np.clip(normalize_heatmap(heatmap), 0.0, 1.0) * 255.0)
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "turbo": cv2.COLORMAP_TURBO,
    }
    colored = cv2.applyColorMap(heatmap_u8, cmap_map[cmap_name])
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def overlay_color_image(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base_rgb = np.clip(base_rgb.astype(np.float32), 0.0, 1.0)
    overlay_rgb = np.clip(overlay_rgb.astype(np.float32), 0.0, 1.0)
    return np.clip((1.0 - alpha) * base_rgb + alpha * overlay_rgb, 0.0, 1.0)


def format_metric(value: float, digits: int = 3) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def make_view(
    key: str,
    title: str,
    heatmap: np.ndarray,
    token_hw: Tuple[int, int],
    hand_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> HeatmapView:
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    metrics = {
        "token_hw": [int(token_hw[0]), int(token_hw[1])],
        # `inside_hand_mass` is the fraction of positive activation lying inside the approximated hand mask.
        # Larger is better because it means the response is concentrated on hand pixels rather than background.
        "inside_hand_mass": heatmap_mass_in_mask(heatmap, hand_mask),
        "inside_bbox_mass": heatmap_mass_in_bbox(heatmap, bbox_xyxy),
        "peak_xy": [float(peak_x), float(peak_y)],
        "peak_value": float(np.max(heatmap)),
        **summarize_focus_values(heatmap, hand_mask),
    }
    return HeatmapView(key=key, title=title, token_hw=token_hw, heatmap=heatmap.astype(np.float32), metrics=metrics)


@torch.no_grad()
def prepare_encoder_inputs(
    model: GPGFormer,
    img_01: torch.Tensor,
    cam_param: Optional[torch.Tensor] = None,
    use_geometry: bool = True,
) -> Dict[str, Any]:
    """
    Reproduce the exact preprocessing done inside `GPGFormer.forward`, but stop before MANO decoding.

    This lets us visualize backbone-stage tensors directly while keeping RGB normalization,
    MoGe2 inference, GeoSideAdapter, GeoTokenizer, GeoSideTuning, and camera handling identical
    to the real model path.
    """

    img_crop = img_01
    if img_crop.ndim == 4 and img_crop.shape[1] != 3 and img_crop.shape[-1] == 3:
        img_crop = img_crop.permute(0, 3, 1, 2).contiguous()
    if img_crop.shape[-2:] != self_or_cfg_image_hw(model):
        img_crop = F.interpolate(img_crop, size=self_or_cfg_image_hw(model), mode="bilinear", align_corners=False)

    mean = IMAGENET_MEAN.to(device=img_crop.device, dtype=img_crop.dtype).unsqueeze(0)
    std = IMAGENET_STD.to(device=img_crop.device, dtype=img_crop.dtype).unsqueeze(0)
    is_imagenet_norm = bool((img_crop.min() < -0.2) or (img_crop.max() > 1.2))

    geo_tokens = None
    geo_pos = None
    if use_geometry and bool(getattr(model, "use_geo_prior", False)):
        if model.moge2 is None or model.geo_pos is None:
            raise RuntimeError("Geometry prior is enabled in cfg but MoGe2 / geo_pos is not initialized.")

        img_moge = (img_crop * std + mean).clamp(0.0, 1.0) if is_imagenet_norm else img_crop.clamp(0.0, 1.0)
        geo_feat = model.moge2(img_moge).clone()
        if model.geo_side_adapter is not None:
            geo_feat = model.geo_side_adapter(geo_feat)
        model._init_geo_tokenizer_if_needed(geo_feat)
        if model.geo_tokenizer is None:
            raise RuntimeError("GeoTokenizer failed to initialize.")
        geo_tokens, coords = model.geo_tokenizer(geo_feat)
        if model.geo_side_tuning is not None:
            geo_tokens = model.geo_side_tuning(geo_tokens)
        geo_pos = model.geo_pos(coords)

    img_wilor = img_crop if is_imagenet_norm else (img_crop - mean) / std

    focal_length_px = None
    if cam_param is not None:
        if cam_param.ndim != 2 or cam_param.shape[-1] < 2:
            raise ValueError(f"cam_param must be (B,4) or (B,>=2), got {tuple(cam_param.shape)}")
        focal_length_px = cam_param[:, :2]

    return {
        "img_wilor": img_wilor,
        "geo_tokens": geo_tokens,
        "geo_pos": geo_pos,
        "focal_length_px": focal_length_px,
    }


def self_or_cfg_image_hw(model: GPGFormer) -> Tuple[int, int]:
    image_hw = getattr(model.cfg, "image_hw", None)
    if image_hw is None:
        size = int(getattr(model.cfg, "image_size", 256))
        return (size, int(round(size * 0.75)))
    return int(image_hw[0]), int(image_hw[1])


@torch.no_grad()
def run_encoder(
    model: GPGFormer,
    img: torch.Tensor,
    cam_param: torch.Tensor,
    use_geometry: bool,
) -> Dict[str, Any]:
    enc_inputs = prepare_encoder_inputs(model, img, cam_param=cam_param, use_geometry=use_geometry)
    enc_out = model.encoder(
        enc_inputs["img_wilor"],
        geo_tokens=enc_inputs["geo_tokens"],
        geo_pos=enc_inputs["geo_pos"],
        focal_length_px=enc_inputs["focal_length_px"],
    )
    return {
        **enc_inputs,
        "img_feat": enc_out["img_feat"],
    }


def extract_patch_tokens_before_block(
    token_sequence: np.ndarray,
    rgb_hw: Tuple[int, int],
    geo_token_count: Optional[int],
) -> np.ndarray:
    rgb_token_count = int(rgb_hw[0] * rgb_hw[1])
    if token_sequence.shape[0] == rgb_token_count:
        return token_sequence
    if token_sequence.shape[0] > rgb_token_count:
        # In concat-mode encoders, block-0 input contains `[geo_tokens, rgb_tokens]`.
        # We keep the trailing patch-token slice so the resulting heatmap still lives on the RGB patch grid.
        return token_sequence[-rgb_token_count:]
    if geo_token_count is not None and token_sequence.shape[0] == geo_token_count:
        raise ValueError("Pre-backbone sequence contains only geometry tokens; expected patch tokens as well.")
    raise ValueError(
        f"Unable to extract patch tokens from pre-backbone sequence of shape {token_sequence.shape} with rgb_hw={rgb_hw}"
    )


def resolve_sample_identifier(dataset, sample_idx: int, sample: Dict[str, Any]) -> tuple[str, Optional[str], str]:
    source_path = None
    if isinstance(sample.get("img_path", None), str):
        source_path = sample["img_path"]
    elif hasattr(dataset, "datalist") and sample_idx < len(dataset.datalist):
        data_item = dataset.datalist[sample_idx]
        if isinstance(data_item, dict) and isinstance(data_item.get("img_path", None), str):
            source_path = data_item["img_path"]

    if hasattr(dataset, "indices"):
        real_idx = int(dataset.indices[sample_idx])
        identifier = f"img_{real_idx:08d}"
        slug = f"sample_{sample_idx:05d}_{identifier}"
        return identifier, source_path, slug

    stem = Path(source_path).stem if source_path else f"idx_{sample_idx:05d}"
    identifier = sanitize_name(stem)
    slug = f"sample_{sample_idx:05d}_{identifier}"
    return identifier, source_path, slug


def extract_token_bundle(
    model: GPGFormer,
    rgb_reference_model: Optional[GPGFormer],
    dataset_name: str,
    dataset,
    sample_idx: int,
    sample: Dict[str, Any],
    device: torch.device,
    heatmap_reducer: str,
    darken_factor: float,
) -> TokenVisualizationBundle:
    img = sample["rgb"].unsqueeze(0).to(device)
    cam_param = sample["cam_param"].unsqueeze(0).to(device)
    image_rgb = denormalize_rgb(sample["rgb"])
    img_h, img_w = image_rgb.shape[:2]
    hand_bbox = hand_bbox_from_sample(sample, img_h=img_h, img_w=img_w)
    hand_mask = build_hand_region_mask(sample, img_h=img_h, img_w=img_w, bbox_xyxy=hand_bbox)
    dark_image = darken_image(image_rgb, factor=darken_factor)
    sample_identifier, source_image_path, _sample_slug = resolve_sample_identifier(dataset, sample_idx, sample)

    recorder = TokenFlowRecorder(model)
    try:
        with torch.no_grad():
            _ = run_encoder(model, img=img, cam_param=cam_param, use_geometry=True)
        cache = dict(recorder.cache)
    finally:
        recorder.remove()

    rgb_hw = cache.get("rgb_token_hw")
    if rgb_hw is None:
        raise RuntimeError("Failed to capture RGB patch token spatial size.")
    rgb_hw = (int(rgb_hw[0]), int(rgb_hw[1]))

    rgb_patch_tokens = cache.get("rgb_patch_tokens_pre_fusion")
    if rgb_patch_tokens is None:
        raise RuntimeError("Failed to capture patch-embedding RGB tokens.")
    rgb_patch_map = tokens_to_spatial_map(rgb_patch_tokens[0].numpy(), rgb_hw)
    rgb_patch_heatmap = upsample_map(reduce_channel_energy(rgb_patch_map, heatmap_reducer), img_h, img_w)

    geo_tokens_tensor = cache.get("geo_tokens_encoder_input", cache.get("geo_tokens_raw"))
    geo_coords = cache.get("geo_coords")
    if geo_tokens_tensor is None or geo_coords is None:
        raise RuntimeError("Failed to capture geometry tokens from the multimodal path.")
    geo_hw = (int(geo_coords.shape[0]), int(geo_coords.shape[1]))
    geo_token_map = tokens_to_spatial_map(geo_tokens_tensor[0].numpy(), geo_hw)
    geo_heatmap = upsample_map(reduce_channel_energy(geo_token_map, heatmap_reducer), img_h, img_w)

    pre_backbone_tokens = cache.get("pre_backbone_tokens")
    if pre_backbone_tokens is None:
        raise RuntimeError("Failed to capture the token sequence before backbone block 0.")
    geo_token_count = int(geo_tokens_tensor.shape[1]) if geo_tokens_tensor is not None else None
    fused_patch_tokens = extract_patch_tokens_before_block(pre_backbone_tokens[0].numpy(), rgb_hw, geo_token_count)
    fused_patch_map = tokens_to_spatial_map(fused_patch_tokens, rgb_hw)
    fused_heatmap = upsample_map(reduce_channel_energy(fused_patch_map, heatmap_reducer), img_h, img_w)

    encoder_img_feat = cache.get("encoder_img_feat")
    if encoder_img_feat is None:
        raise RuntimeError("Failed to capture multimodal backbone final features.")
    encoder_img_feat_np = encoder_img_feat[0].numpy()
    multimodal_heatmap = upsample_map(
        reduce_channel_energy(encoder_img_feat_np, heatmap_reducer),
        img_h,
        img_w,
    )
    final_mm_hw = (int(encoder_img_feat_np.shape[1]), int(encoder_img_feat_np.shape[2]))
    final_mm_tokens = feature_map_to_tokens(encoder_img_feat_np)

    hand_token_mask = resize_mask(hand_mask, final_mm_hw)
    if not np.any(hand_token_mask):
        hand_token_mask = resize_mask(bbox_mask(img_h, img_w, hand_bbox), final_mm_hw, threshold=0.01)
    if np.any(hand_token_mask):
        prototype = final_mm_tokens[hand_token_mask.reshape(-1)].mean(axis=0)
    else:
        prototype = final_mm_tokens.mean(axis=0)

    rgb_model = rgb_reference_model if rgb_reference_model is not None else model
    rgb_only_label = Path(getattr(rgb_model, "_visualization_checkpoint", "same_multimodal_checkpoint")).stem
    with torch.no_grad():
        rgb_only_out = run_encoder(rgb_model, img=img, cam_param=cam_param, use_geometry=False)
    rgb_only_img_feat = rgb_only_out["img_feat"][0].detach().cpu().numpy()
    rgb_only_hw = (int(rgb_only_img_feat.shape[1]), int(rgb_only_img_feat.shape[2]))
    rgb_only_heatmap = upsample_map(reduce_channel_energy(rgb_only_img_feat, heatmap_reducer), img_h, img_w)
    rgb_only_tokens = feature_map_to_tokens(rgb_only_img_feat)

    rgb_similarity = upsample_map(compute_token_cosine_map(rgb_patch_tokens[0].numpy(), prototype, rgb_hw), img_h, img_w)
    geo_similarity = upsample_map(compute_token_cosine_map(geo_tokens_tensor[0].numpy(), prototype, geo_hw), img_h, img_w)
    fused_similarity = upsample_map(compute_token_cosine_map(fused_patch_tokens, prototype, rgb_hw), img_h, img_w)
    final_mm_similarity = upsample_map(compute_token_cosine_map(final_mm_tokens, prototype, final_mm_hw), img_h, img_w)
    final_rgb_similarity = upsample_map(compute_token_cosine_map(rgb_only_tokens, prototype, rgb_only_hw), img_h, img_w)

    fusion_gain = fused_similarity - np.maximum(rgb_similarity, geo_similarity)
    final_multimodal_gain = final_mm_similarity - final_rgb_similarity

    analysis_views: List[HeatmapView] = []
    patch_norm_out = cache.get("patch_norm_out")
    geo_norm_out = cache.get("geo_norm_out")
    try:
        if patch_norm_out is not None and geo_norm_out is not None and rgb_hw == geo_hw:
            cross_modal_distance = upsample_map(
                compute_cross_modal_cosine_distance(patch_norm_out[0].numpy(), geo_norm_out[0].numpy(), rgb_hw),
                img_h,
                img_w,
            )
        elif rgb_patch_tokens[0].shape == geo_tokens_tensor[0].shape and rgb_hw == geo_hw:
            cross_modal_distance = upsample_map(
                compute_cross_modal_cosine_distance(rgb_patch_tokens[0].numpy(), geo_tokens_tensor[0].numpy(), rgb_hw),
                img_h,
                img_w,
            )
        else:
            cross_modal_distance = None
    except Exception:
        cross_modal_distance = None

    if cross_modal_distance is not None:
        analysis_views.append(
            make_view(
                "cross_modal_cosine_distance",
                "Cross-modal cosine distance",
                cross_modal_distance,
                rgb_hw,
                hand_mask,
                hand_bbox,
            )
        )

    analysis_views.extend(
        [
            make_view("rgb_hand_similarity", "RGB-hand similarity", rgb_similarity, rgb_hw, hand_mask, hand_bbox),
            make_view("geo_hand_similarity", "Geometry-hand similarity", geo_similarity, geo_hw, hand_mask, hand_bbox),
            make_view("fused_hand_similarity", "Fusion-hand similarity", fused_similarity, rgb_hw, hand_mask, hand_bbox),
            make_view("fusion_gain_over_best", "Fusion gain over best single", fusion_gain, rgb_hw, hand_mask, hand_bbox),
            make_view(
                "backbone_last_multimodal_similarity",
                "Final multimodal-hand similarity",
                final_mm_similarity,
                final_mm_hw,
                hand_mask,
                hand_bbox,
            ),
            make_view(
                "backbone_last_rgb_only_similarity",
                "Final RGB-only-hand similarity",
                final_rgb_similarity,
                rgb_only_hw,
                hand_mask,
                hand_bbox,
            ),
            make_view(
                "backbone_last_multimodal_gain",
                "Final multimodal gain",
                final_multimodal_gain,
                final_mm_hw,
                hand_mask,
                hand_bbox,
            ),
        ]
    )

    energy_views = [
        make_view("rgb_patch", "RGB patch tokens", rgb_patch_heatmap, rgb_hw, hand_mask, hand_bbox),
        make_view("geo_tokens", "Geometry tokens", geo_heatmap, geo_hw, hand_mask, hand_bbox),
        make_view("fused_pre_backbone", "Fusion before backbone", fused_heatmap, rgb_hw, hand_mask, hand_bbox),
        make_view(
            "backbone_last_multimodal",
            "Backbone last layer (multimodal)",
            multimodal_heatmap,
            final_mm_hw,
            hand_mask,
            hand_bbox,
        ),
        make_view("backbone_last_rgb_only", "Backbone last layer (RGB-only)", rgb_only_heatmap, rgb_only_hw, hand_mask, hand_bbox),
    ]

    return TokenVisualizationBundle(
        dataset_name=dataset_name,
        model_label=Path(getattr(model, "_visualization_checkpoint", "multimodal_checkpoint")).stem,
        rgb_only_label=rgb_only_label,
        input_image=image_rgb,
        dark_image=dark_image,
        hand_bbox_xyxy=hand_bbox,
        hand_mask=hand_mask,
        sample_identifier=sample_identifier,
        source_image_path=source_image_path,
        energy_views=energy_views,
        analysis_views=analysis_views,
    )


def hand_mask_preview(image_rgb: np.ndarray, hand_mask: np.ndarray, show_bbox: bool, bbox_xyxy: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[hand_mask] = 0.65 * overlay[hand_mask] + 0.35 * np.array([1.0, 0.2, 0.2], dtype=np.float32)
    return maybe_draw_bbox(overlay, bbox_xyxy, show_bbox)


def render_view_overlay(bundle: TokenVisualizationBundle, view: HeatmapView, cmap_name: str, overlay_alpha: float, show_bbox: bool) -> np.ndarray:
    overlay = overlay_color_image(bundle.dark_image, colorize_heatmap(view.heatmap, cmap_name), overlay_alpha)
    return maybe_draw_bbox(overlay, bundle.hand_bbox_xyxy, show_bbox)


def save_energy_overview(
    path: Path,
    bundle: TokenVisualizationBundle,
    cmap_name: str,
    overlay_alpha: float,
    show_bbox: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5))
    axes = axes.reshape(2, 3)

    axes[0, 0].imshow(maybe_draw_bbox(bundle.input_image, bundle.hand_bbox_xyxy, show_bbox))
    axes[0, 0].set_title(f"Input crop\n{bundle.sample_identifier}")
    axes[0, 0].axis("off")

    slot_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, view in zip(slot_axes, bundle.energy_views):
        ax.imshow(render_view_overlay(bundle, view, cmap_name, overlay_alpha, show_bbox))
        ax.set_title(f"{view.title}\ninside-hand={format_metric(view.metrics['inside_hand_mass'])}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_semantic_overview(
    path: Path,
    bundle: TokenVisualizationBundle,
    cmap_name: str,
    overlay_alpha: float,
    show_bbox: bool,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.5))
    axes = axes.reshape(3, 3)

    axes[0, 0].imshow(maybe_draw_bbox(bundle.input_image, bundle.hand_bbox_xyxy, show_bbox))
    axes[0, 0].set_title(f"Input crop\n{bundle.sample_identifier}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hand_mask_preview(bundle.input_image, bundle.hand_mask, show_bbox, bundle.hand_bbox_xyxy))
    axes[0, 1].set_title("Approx. hand region")
    axes[0, 1].axis("off")

    semantic_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]
    for ax, view in zip(semantic_axes, bundle.analysis_views):
        ax.imshow(render_view_overlay(bundle, view, cmap_name, overlay_alpha, show_bbox))
        ax.set_title(
            f"{view.title}\ninside-hand={format_metric(view.metrics['inside_hand_mass'])}, top10={format_metric(view.metrics['top10_inside_ratio'])}"
        )
        ax.axis("off")

    for ax in semantic_axes[len(bundle.analysis_views):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sample_outputs(
    sample_dir: Path,
    sample_idx: int,
    bundle: TokenVisualizationBundle,
    args: argparse.Namespace,
    checkpoint_path: str,
    rgb_checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_rgb_image(sample_dir / "input_crop.png", bundle.input_image)
    save_rgb_image(sample_dir / "input_dark.png", bundle.dark_image)
    if args.show_bbox:
        save_rgb_image(
            sample_dir / "input_crop_with_bbox.png",
            draw_bbox_on_image(bundle.input_image, bundle.hand_bbox_xyxy),
        )
    save_rgb_image(sample_dir / "hand_region_mask_preview.png", hand_mask_preview(bundle.input_image, bundle.hand_mask, args.show_bbox, bundle.hand_bbox_xyxy))

    view_metadata: Dict[str, Any] = {}
    for view in bundle.energy_views + bundle.analysis_views:
        color = colorize_heatmap(view.heatmap, args.heatmap_cmap)
        overlay = overlay_color_image(bundle.dark_image, color, args.overlay_alpha)
        save_rgb_image(sample_dir / f"{view.key}_heatmap.png", color)
        save_rgb_image(sample_dir / f"{view.key}_overlay.png", maybe_draw_bbox(overlay, bundle.hand_bbox_xyxy, args.show_bbox))
        view_metadata[view.key] = {
            "title": view.title,
            "metrics": view.metrics,
        }

    save_energy_overview(
        path=sample_dir / "overview_energy.png",
        bundle=bundle,
        cmap_name=args.heatmap_cmap,
        overlay_alpha=float(args.overlay_alpha),
        show_bbox=bool(args.show_bbox),
    )
    save_semantic_overview(
        path=sample_dir / "overview_semantic.png",
        bundle=bundle,
        cmap_name=args.heatmap_cmap,
        overlay_alpha=float(args.overlay_alpha),
        show_bbox=bool(args.show_bbox),
    )
    save_semantic_overview(
        path=sample_dir / "overview.png",
        bundle=bundle,
        cmap_name=args.heatmap_cmap,
        overlay_alpha=float(args.overlay_alpha),
        show_bbox=bool(args.show_bbox),
    )

    metadata = {
        "sample_index": int(sample_idx),
        "sample_identifier": bundle.sample_identifier,
        "dataset": bundle.dataset_name,
        "model_label": bundle.model_label,
        "rgb_only_label": bundle.rgb_only_label,
        "source_image_path": bundle.source_image_path,
        "multimodal_checkpoint": checkpoint_path,
        "rgb_checkpoint": rgb_checkpoint_path,
        "heatmap_reducer": args.heatmap_reducer,
        "heatmap_cmap": args.heatmap_cmap,
        "overlay_alpha": float(args.overlay_alpha),
        "darken_factor": float(args.darken_factor),
        "show_bbox": bool(args.show_bbox),
        "hand_bbox_xyxy": [float(x) for x in bundle.hand_bbox_xyxy.tolist()],
        "hand_mask_area_pixels": int(np.sum(bundle.hand_mask)),
        "views": view_metadata,
    }
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def resolve_checkpoint_arg(dataset_name: str, ckpt: Optional[str], is_rgb: bool = False) -> Optional[str]:
    if ckpt:
        return ckpt
    default_map = DEFAULT_RGB_ONLY_CKPTS if is_rgb else DEFAULT_MULTIMODAL_CKPTS
    return default_map.get(dataset_name, None)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = resolve_checkpoint_arg(args.dataset, args.checkpoint, is_rgb=False)
    if checkpoint_path is None:
        raise ValueError(f"No default multimodal checkpoint is configured for dataset={args.dataset!r}; please pass --checkpoint.")
    rgb_checkpoint_path = resolve_checkpoint_arg(args.dataset, args.rgb_checkpoint, is_rgb=True)

    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "outputs" / f"{args.dataset}_token_heatmap_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[info] loading multimodal checkpoint: {checkpoint_path}")
    model, cfg = load_model_bundle(checkpoint_path, device=device)
    model._visualization_checkpoint = checkpoint_path
    if not bool(getattr(model, "use_geo_prior", False)):
        raise ValueError("The selected multimodal checkpoint has use_geo_prior=False, so multimodal token visualization is unsupported.")

    rgb_reference_model: Optional[GPGFormer] = None
    if rgb_checkpoint_path is not None:
        print(f"[info] loading RGB reference checkpoint: {rgb_checkpoint_path}")
        rgb_reference_model, _ = load_model_bundle(rgb_checkpoint_path, device=device)
        rgb_reference_model._visualization_checkpoint = rgb_checkpoint_path
    else:
        print("[info] no RGB-only checkpoint provided; RGB-only heatmap will reuse the multimodal checkpoint with geometry disabled.")

    dataset_split = resolve_dataset_split(args.dataset, args.split)
    dataset = build_dataset(cfg, args.dataset, dataset_split)
    sample_indices = select_sample_indices(dataset, dataset_split, args)

    summary = {
        "dataset": args.dataset,
        "requested_split": args.split,
        "resolved_split": dataset_split,
        "multimodal_checkpoint": checkpoint_path,
        "rgb_checkpoint": rgb_checkpoint_path,
        "heatmap_reducer": args.heatmap_reducer,
        "heatmap_cmap": args.heatmap_cmap,
        "overlay_alpha": float(args.overlay_alpha),
        "darken_factor": float(args.darken_factor),
        "show_bbox": bool(args.show_bbox),
        "token_fusion_mode": str(getattr(model.encoder, "token_fusion_mode", "")),
        "sum_fusion_strategy": str(getattr(model.encoder, "sum_fusion_strategy", "")),
        "sample_indices": sample_indices,
        "samples": [],
    }

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        _, _, sample_slug = resolve_sample_identifier(dataset, sample_idx, sample)
        sample_dir = output_dir / sample_slug
        print(f"[progress] sample_idx={sample_idx} -> {sample_dir}")

        bundle = extract_token_bundle(
            model=model,
            rgb_reference_model=rgb_reference_model,
            dataset_name=args.dataset,
            dataset=dataset,
            sample_idx=sample_idx,
            sample=sample,
            device=device,
            heatmap_reducer=args.heatmap_reducer,
            darken_factor=float(args.darken_factor),
        )
        summary["samples"].append(
            save_sample_outputs(
                sample_dir=sample_dir,
                sample_idx=sample_idx,
                bundle=bundle,
                args=args,
                checkpoint_path=checkpoint_path,
                rgb_checkpoint_path=rgb_checkpoint_path,
            )
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

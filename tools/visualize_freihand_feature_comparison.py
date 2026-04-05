#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

from data.freihand_dataset_v3 import FreiHANDDatasetV3
from gpgformer.models import GPGFormer, GPGFormerConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_RGB_CKPT = (
    "/root/code/vepfs/GPGFormer/checkpoints/freihand_20260304_rgb_only/freihand/gpgformer_best.pt"
)
DEFAULT_MULTIMODAL_CKPT = (
    "/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/"
    "freihand/gpgformer_best.pt"
)
JOINT_NAME_TO_INDEX = {
    "wrist": 0,
    "thumb_mcp": 1,
    "thumb_pip": 2,
    "thumb_dip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}
DEFAULT_JOINT_NAMES = ["thumb_tip", "middle_mcp"]


@dataclass
class FeatureDebugBundle:
    model_name: str
    input_image: np.ndarray
    img_feat_map: np.ndarray
    rollout_maps: Dict[int, np.ndarray]
    sjta_maps: Dict[str, np.ndarray]
    gt_joint_pixels: Dict[str, np.ndarray]
    pred_joint_pixels: Dict[str, np.ndarray]
    hand_bbox_xyxy: np.ndarray
    metrics: Dict[str, Any]


class BackboneAttentionRecorder:
    def __init__(self, model: GPGFormer):
        self.model = model
        self.inputs: Dict[int, torch.Tensor] = {}
        self.handles = []
        for idx, blk in enumerate(self.model.encoder.backbone.blocks):
            handle = blk.attn.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(handle)

    def _make_hook(self, idx: int):
        def _hook(_module, args):
            self.inputs[idx] = args[0].detach()

        return _hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def get_attentions(self) -> List[torch.Tensor]:
        attns: List[torch.Tensor] = []
        for idx, blk in enumerate(self.model.encoder.backbone.blocks):
            x = self.inputs[idx]
            attn_mod = blk.attn
            qkv = attn_mod.qkv(x)
            qkv = qkv.reshape(x.shape[0], x.shape[1], 3, attn_mod.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q = q * attn_mod.scale
            attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
            attns.append(attn.detach().cpu())
        return attns


class SJTADebugRecorder:
    def __init__(self, model: GPGFormer):
        self.model = model
        self.cache: Dict[str, Any] = {}
        self.enabled = bool(
            getattr(model, "feature_refiner", None) is not None
            and getattr(model.feature_refiner, "method", "").lower() == "sjta"
        )
        self._orig = None
        if self.enabled:
            self._orig = model.feature_refiner.refine_params

            def wrapped(img_feat, joints_3d, pred_cam, pred_mano_feats, focal_length, img_size=256.0):
                self.cache = {
                    "img_feat": img_feat.detach().cpu(),
                    "joints_3d": joints_3d.detach().cpu(),
                    "pred_cam": pred_cam.detach().cpu(),
                    "focal_length": focal_length.detach().cpu(),
                    "img_size": float(img_size),
                }
                return self._orig(img_feat, joints_3d, pred_cam, pred_mano_feats, focal_length, img_size)

            model.feature_refiner.refine_params = wrapped

    def remove(self) -> None:
        if self.enabled and self._orig is not None:
            self.model.feature_refiner.refine_params = self._orig
            self._orig = None

    @torch.no_grad()
    def compute(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled or not self.cache:
            return None
        sjta = self.model.feature_refiner.refiner
        device = next(sjta.parameters()).device
        img_feat = self.cache["img_feat"].to(device=device, dtype=torch.float32)
        joints_3d = self.cache["joints_3d"].to(device=device, dtype=torch.float32)
        pred_cam = self.cache["pred_cam"].to(device=device, dtype=torch.float32)
        focal_length = self.cache["focal_length"].to(device=device, dtype=torch.float32)
        img_size = float(self.cache["img_size"])

        B, _, H, W = img_feat.shape
        patch_tokens = img_feat.flatten(2).permute(0, 2, 1)
        N = patch_tokens.shape[1]
        patch_tokens = sjta.token_ln(patch_tokens)
        K = sjta.token_proj_k(patch_tokens)

        if sjta.cfg.sjta_use_2d_prior:
            patch_h = float(img_size)
            patch_w = float(img_size) * (float(W) / float(H)) if H > 0 else float(img_size)
            joints_2d = sjta._project_joints(joints_3d, pred_cam, focal_length, patch_h, patch_w)
            joints_2d_norm = torch.stack(
                [
                    joints_2d[..., 0] / (patch_w * 0.5) - 1.0,
                    joints_2d[..., 1] / (patch_h * 0.5) - 1.0,
                ],
                dim=-1,
            )
            joint_input = torch.cat([joints_3d, joints_2d_norm], dim=-1)
        else:
            joints_2d = None
            joints_2d_norm = None
            joint_input = joints_3d

        joint_feat = sjta.joint_encoder(joint_input) + sjta.joint_id_embed.to(device=device)
        Q = sjta.q_proj(joint_feat)
        Q = Q.view(B, sjta.num_queries, sjta.num_heads, sjta.head_dim).transpose(1, 2)
        K = K.view(B, N, sjta.num_heads, sjta.head_dim).transpose(1, 2)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * sjta.scale

        if sjta.geo_bias_enabled:
            if joints_2d_norm is None:
                raise RuntimeError("sjta_geo_bias requires joints_2d_norm.")
            token_xy = sjta._token_xy_norm(H, W, device=device, dtype=torch.float32)
            diff = joints_2d_norm.unsqueeze(2) - token_xy.view(1, 1, N, 2)
            dist2 = (diff * diff).sum(dim=-1)
            sigma2 = float(max(sjta.geo_bias_sigma, 1e-6)) ** 2
            geo_bias = -dist2 / (2.0 * sigma2)
            attn_logits = attn_logits + (sjta.geo_bias_beta * geo_bias).unsqueeze(1)

        attn = torch.softmax(attn_logits, dim=-1)
        return {
            "mean_attn": attn.mean(dim=1).cpu(),
            "query_xy_px": joints_2d.cpu() if joints_2d is not None else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FreiHAND feature maps for GPGFormer RGB-only vs multimodal checkpoints."
    )
    parser.add_argument("--rgb-checkpoint", type=str, default=DEFAULT_RGB_CKPT)
    parser.add_argument("--multimodal-checkpoint", type=str, default=DEFAULT_MULTIMODAL_CKPT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "freihand_feature_compare_20260401"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["train",  "eval"], default="eval")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument("--layers", nargs="*", type=int, default=[4, 8, 12])
    parser.add_argument("--joint-names", nargs="*", type=str, default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--img-feat-reducer", choices=["l2", "var"], default="l2")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Draw the GT hand bbox on exported overlays. Disabled by default for cleaner figures.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_heatmap(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    denom = x.max()
    if denom > 1e-8:
        x = x / denom
    return x


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def heatmap_mass_in_bbox(heatmap: np.ndarray, bbox_xyxy: np.ndarray) -> float:
    # inside-bbox is designed as an intuitive focus score: a larger value means more activation
    # mass falls inside the GT hand region, so larger is better.
    # 表示热图正激活落在 GT 手部框内的质量占比，越大越好
    x1, y1, x2, y2 = bbox_xyxy.astype(np.int64).tolist()
    h, w = heatmap.shape
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    positive = np.maximum(heatmap.astype(np.float32), 0.0)
    denom = float(positive.sum())
    if denom <= 1e-8:
        return float("nan")
    return float(positive[y1:y2, x1:x2].sum() / denom)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, cmap_name: str, alpha: float) -> np.ndarray:
    image_rgb = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
    heatmap = normalize_heatmap(heatmap)
    color = matplotlib.colormaps[cmap_name](heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_rgb + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def denormalize_rgb(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float()
    if img.ndim == 4:
        img = img[0]
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def select_sample_indices(dataset: FreiHANDDatasetV3, args: argparse.Namespace) -> List[int]:
    if args.sample_indices:
        return list(args.sample_indices)
    total = len(dataset)
    count = min(max(int(args.num_samples), 0), total)
    if args.split == "val":
        return list(range(count))
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(total, size=count, replace=False).tolist())


def build_freihand_dataset(cfg: dict, split: str) -> FreiHANDDatasetV3:
    ds_cfg = dict(cfg["dataset"])
    paths = cfg["paths"]
    if split == "train":
        train = True
        use_trainval_split = bool(ds_cfg.get("use_trainval_split", True))
    elif split == "eval":
        train = False
        use_trainval_split = False
    else:
        train = False
        use_trainval_split = True
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


def build_model_from_cfg(cfg: dict, device: torch.device) -> GPGFormer:
    model_cfg = cfg["model"]
    paths = cfg["paths"]
    side_tuning_cfg = model_cfg.get("side_tuning", {})
    geo_side_adapter_cfg = model_cfg.get("geo_side_adapter", {})
    feature_refiner_cfg = model_cfg.get("feature_refiner", {})
    model = GPGFormer(
        GPGFormerConfig(
            backbone_type=str(model_cfg.get("backbone_type", "wilor")),
            wilor_ckpt_path=paths.get("wilor_ckpt", ""),
            vitpose_ckpt_path=paths.get("vitpose_ckpt", ""),
            mano_model_path=paths["mano_dir"],
            mano_mean_params=paths["mano_mean_params"],
            moge2_weights_path=paths.get("moge2_ckpt", None),
            use_geo_prior=bool(model_cfg.get("use_geo_prior", True)),
            image_size=int(model_cfg.get("image_size", 256)),
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
    w = int(h * 0.75)
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


def compute_rollout_maps(attns: Sequence[torch.Tensor], layer_numbers: Sequence[int], hp: int, wp: int) -> Dict[int, np.ndarray]:
    rollout = None
    layer_to_map: Dict[int, np.ndarray] = {}
    max_layer = max(layer_numbers)
    for layer_idx in range(max_layer):
        attn = attns[layer_idx].mean(dim=1)
        identity = torch.eye(attn.shape[-1], dtype=attn.dtype).unsqueeze(0)
        attn = attn + identity
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else attn @ rollout
        layer_num = layer_idx + 1
        if layer_num in layer_numbers:
            token_scores = rollout.mean(dim=1)[0].reshape(hp, wp).cpu().numpy()
            layer_to_map[layer_num] = normalize_heatmap(token_scores)
    return layer_to_map


def reduce_img_feat(img_feat: torch.Tensor, reducer: str) -> np.ndarray:
    if reducer == "var":
        reduced = img_feat.var(dim=1)
    else:
        reduced = torch.linalg.vector_norm(img_feat, ord=2, dim=1)
    return normalize_heatmap(reduced[0].detach().cpu().numpy())


def keypoints_to_pixels(sample: Dict[str, Any], joint_names: Sequence[str], img_h: int, img_w: int) -> Dict[str, np.ndarray]:
    keypoints = as_numpy(sample["keypoints_2d"])
    pixels = np.stack(
        [
            (keypoints[:, 0] + 0.5) * float(img_w),
            (keypoints[:, 1] + 0.5) * float(img_h),
        ],
        axis=-1,
    )
    return {name: pixels[JOINT_NAME_TO_INDEX[name]] for name in joint_names}


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


def project_predicted_joints(joints_3d: torch.Tensor, pred_cam: torch.Tensor, focal_length: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
    scale, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    tz = 2.0 * focal_length[:, 0] / (float(img_h) * scale + 1e-9)
    cam_t = torch.stack([tx, ty, tz], dim=-1)
    joints_cam = joints_3d + cam_t.unsqueeze(1)
    z = joints_cam[..., 2:3].clamp(min=1e-6)
    xy = joints_cam[..., :2] / z
    center = joints_cam.new_tensor([float(img_w) * 0.5, float(img_h) * 0.5]).view(1, 1, 2)
    joints_2d = xy * focal_length.unsqueeze(1) + center
    return joints_2d[0].detach().cpu().numpy()


def upsample_to_image(heatmap: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(img_h, img_w), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def make_overlay(image: np.ndarray, low_res_heatmap: np.ndarray, cmap_name: str, alpha: float) -> np.ndarray:
    hi_res = upsample_to_image(low_res_heatmap, image.shape[0], image.shape[1])
    return overlay_heatmap(image, hi_res, cmap_name=cmap_name, alpha=alpha)


def format_metric_text(metric_name: str, value: float, unit: str = "") -> str:
    # Metric semantics are kept in code comments instead of being painted on the exported figures.
    # - inside-bbox: fraction of positive heatmap mass that lies inside the GT hand bbox, so higher is better.
    # - peak-to-gt: pixel distance from the heatmap peak to the GT joint location, so lower is better.
    if metric_name == "inside-bbox":
        return f"inside-bbox={value:.3f} (higher better)"
    if metric_name == "peak-to-gt":
        suffix = unit or "px"
        return f"peak-to-gt={value:.2f}{suffix} (lower better)"
    return f"{metric_name}={value:.3f}{unit}"


def save_overlay_image(
    path: Path,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    cmap_name: str,
    alpha: float,
    bbox_xyxy: np.ndarray,
    show_bbox: bool,
    marker_xy: Optional[np.ndarray] = None,
    gt_xy: Optional[np.ndarray] = None,
) -> None:
    overlay = make_overlay(image_rgb, heatmap, cmap_name=cmap_name, alpha=alpha)
    fig, ax = plt.subplots(figsize=(4.8, 6.0))
    ax.imshow(overlay)
    # Visual marker semantics:
    # - white bbox: GT hand bbox on the input crop; useful for checking hand-region focus, but optional.
    # - red dot: projected model-predicted joint location on the SJTA map.
    # - cyan x: GT joint location used as a reference target.
    if show_bbox:
        x1, y1, x2, y2 = bbox_xyxy
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
    if marker_xy is not None:
        ax.scatter(marker_xy[0], marker_xy[1], s=28, c="red", marker="o", edgecolors="white", linewidths=0.8)
    if gt_xy is not None:
        ax.scatter(gt_xy[0], gt_xy[1], s=28, c="cyan", marker="x", linewidths=1.4)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    path: Path,
    image_rgb: np.ndarray,
    bundles: Sequence[FeatureDebugBundle],
    layer_numbers: Sequence[int],
    joint_names: Sequence[str],
    cmap_name: str,
    alpha: float,
    reducer_name: str,
    show_bbox: bool,
) -> None:
    rows = 2 + len(layer_numbers) + len(joint_names)
    cols = len(bundles)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, bundle in enumerate(bundles):
        axes[0, col].imshow(image_rgb)
        axes[0, col].set_title(f"{bundle.model_name} | input crop")
        axes[0, col].axis("off")

        bbox = bundle.hand_bbox_xyxy
        overlay = make_overlay(image_rgb, bundle.img_feat_map, cmap_name=cmap_name, alpha=alpha)
        axes[1, col].imshow(overlay)
        if show_bbox:
            axes[1, col].add_patch(
                plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor="white", facecolor="none")
            )
        axes[1, col].set_title(
            f"{bundle.model_name} | img_feat {reducer_name}\n"
            f"{format_metric_text('inside-bbox', bundle.metrics['img_feat']['mass_in_hand_bbox'])}"
        )
        axes[1, col].axis("off")

        row = 2
        for layer in layer_numbers:
            overlay = make_overlay(image_rgb, bundle.rollout_maps[layer], cmap_name=cmap_name, alpha=alpha)
            axes[row, col].imshow(overlay)
            if show_bbox:
                axes[row, col].add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor="white", facecolor="none")
                )
            score = bundle.metrics["rollout"][str(layer)]["mass_in_hand_bbox"]
            axes[row, col].set_title(
                f"{bundle.model_name} | rollout L{layer}\n{format_metric_text('inside-bbox', score)}"
            )
            axes[row, col].axis("off")
            row += 1

        for joint_name in joint_names:
            overlay = make_overlay(image_rgb, bundle.sjta_maps[joint_name], cmap_name=cmap_name, alpha=alpha)
            axes[row, col].imshow(overlay)
            if show_bbox:
                axes[row, col].add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor="white", facecolor="none")
                )
            pred_xy = bundle.pred_joint_pixels[joint_name]
            gt_xy = bundle.gt_joint_pixels[joint_name]
            axes[row, col].scatter(pred_xy[0], pred_xy[1], s=28, c="red", edgecolors="white", linewidths=0.8)
            axes[row, col].scatter(gt_xy[0], gt_xy[1], s=28, c="cyan", marker="x", linewidths=1.3)
            dist = bundle.metrics["sjta"][joint_name]["peak_to_gt_px"]
            axes[row, col].set_title(
                f"{bundle.model_name} | SJTA {joint_name}\n{format_metric_text('peak-to-gt', dist, 'px')}"
            )
            axes[row, col].axis("off")
            row += 1

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_bundle(
    model_name: str,
    model: GPGFormer,
    sample: Dict[str, Any],
    layer_numbers: Sequence[int],
    joint_names: Sequence[str],
    reducer_name: str,
    device: torch.device,
) -> FeatureDebugBundle:
    img = sample["rgb"].unsqueeze(0).to(device)
    cam_param = sample["cam_param"].unsqueeze(0).to(device)
    image_rgb = denormalize_rgb(sample["rgb"])
    img_h, img_w = image_rgb.shape[:2]

    attn_recorder = BackboneAttentionRecorder(model)
    sjta_recorder = SJTADebugRecorder(model)
    with torch.no_grad():
        out = model(img, cam_param=cam_param)
    attns = attn_recorder.get_attentions()
    sjta_debug = sjta_recorder.compute()
    attn_recorder.remove()
    sjta_recorder.remove()

    if sjta_debug is None:
        raise RuntimeError(f"{model_name} does not expose SJTA debug data.")

    img_feat = out["img_feat"]
    hp, wp = img_feat.shape[-2:]
    rollout_maps = compute_rollout_maps(attns, layer_numbers, hp, wp)
    img_feat_map = reduce_img_feat(img_feat, reducer=reducer_name)
    hand_bbox = hand_bbox_from_sample(sample, img_h=img_h, img_w=img_w)
    gt_joint_pixels = keypoints_to_pixels(sample, joint_names=joint_names, img_h=img_h, img_w=img_w)

    metrics: Dict[str, Any] = {
        # inside-bbox is the fraction of normalized heatmap mass inside the GT hand bbox.
        # A larger value means the feature response is more concentrated on the hand region.
        "img_feat": {"mass_in_hand_bbox": heatmap_mass_in_bbox(upsample_to_image(img_feat_map, img_h, img_w), hand_bbox)},
        "rollout": {},
        "sjta": {},
    }
    for layer in layer_numbers:
        hi = upsample_to_image(rollout_maps[layer], img_h, img_w)
        metrics["rollout"][str(layer)] = {"mass_in_hand_bbox": heatmap_mass_in_bbox(hi, hand_bbox)}

    pred_joints_px_all = project_predicted_joints(
        joints_3d=sjta_recorder.cache["joints_3d"].to(device),
        pred_cam=sjta_recorder.cache["pred_cam"].to(device),
        focal_length=sjta_recorder.cache["focal_length"].to(device),
        img_h=img_h,
        img_w=img_w,
    )

    mean_attn = sjta_debug["mean_attn"][0].numpy()
    sjta_maps: Dict[str, np.ndarray] = {}
    pred_joint_pixels: Dict[str, np.ndarray] = {}
    for joint_name in joint_names:
        joint_idx = JOINT_NAME_TO_INDEX[joint_name]
        sjta_map = normalize_heatmap(mean_attn[joint_idx].reshape(hp, wp))
        sjta_maps[joint_name] = sjta_map
        pred_joint_pixels[joint_name] = pred_joints_px_all[joint_idx]
        hi = upsample_to_image(sjta_map, img_h, img_w)
        peak_y, peak_x = np.unravel_index(np.argmax(hi), hi.shape)
        gt_xy = gt_joint_pixels[joint_name]
        metrics["sjta"][joint_name] = {
            "mass_in_hand_bbox": heatmap_mass_in_bbox(hi, hand_bbox),
            "peak_xy": [float(peak_x), float(peak_y)],
            # peak-to-gt_px measures the distance from the attention peak to the GT joint in pixels.
            # A smaller value indicates better spatial alignment.
            "peak_to_gt_px": float(np.linalg.norm(np.array([peak_x, peak_y], dtype=np.float32) - gt_xy.astype(np.float32))),
            # pred-to-gt_px measures the distance from the projected predicted joint to the GT joint in pixels.
            # A smaller value indicates the regressed joint itself is closer to the target.
            "pred_to_gt_px": float(np.linalg.norm(pred_joint_pixels[joint_name].astype(np.float32) - gt_xy.astype(np.float32))),
        }

    return FeatureDebugBundle(
        model_name=model_name,
        input_image=image_rgb,
        img_feat_map=img_feat_map,
        rollout_maps=rollout_maps,
        sjta_maps=sjta_maps,
        gt_joint_pixels=gt_joint_pixels,
        pred_joint_pixels=pred_joint_pixels,
        hand_bbox_xyxy=hand_bbox,
        metrics=metrics,
    )


def save_sample_outputs(
    sample_dir: Path,
    sample_idx: int,
    real_idx: int,
    bundles: Sequence[FeatureDebugBundle],
    layer_numbers: Sequence[int],
    joint_names: Sequence[str],
    cmap_name: str,
    alpha: float,
    reducer_name: str,
    show_bbox: bool,
) -> None:
    image_rgb = bundles[0].input_image
    cv2.imwrite(str(sample_dir / "input_crop.png"), cv2.cvtColor(np.uint8(image_rgb * 255.0), cv2.COLOR_RGB2BGR))

    metadata = {
        "sample_index": int(sample_idx),
        "real_index": int(real_idx),
        "layers": [int(x) for x in layer_numbers],
        "joint_names": list(joint_names),
        "img_feat_reducer": reducer_name,
        "show_bbox": bool(show_bbox),
        "models": {},
    }

    for bundle in bundles:
        model_dir = sample_dir / bundle.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        save_overlay_image(
            model_dir / f"img_feat_{reducer_name}.png",
            image_rgb=image_rgb,
            heatmap=bundle.img_feat_map,
            title=f"{bundle.model_name} | img_feat {reducer_name}",
            cmap_name=cmap_name,
            alpha=alpha,
            bbox_xyxy=bundle.hand_bbox_xyxy,
            show_bbox=show_bbox,
        )
        for layer in layer_numbers:
            save_overlay_image(
                model_dir / f"rollout_layer_{layer:02d}.png",
                image_rgb=image_rgb,
                heatmap=bundle.rollout_maps[layer],
                title=f"{bundle.model_name} | attention rollout L{layer}",
                cmap_name=cmap_name,
                alpha=alpha,
                bbox_xyxy=bundle.hand_bbox_xyxy,
                show_bbox=show_bbox,
            )
        for joint_name in joint_names:
            save_overlay_image(
                model_dir / f"sjta_{joint_name}.png",
                image_rgb=image_rgb,
                heatmap=bundle.sjta_maps[joint_name],
                title=f"{bundle.model_name} | SJTA {joint_name}",
                cmap_name=cmap_name,
                alpha=alpha,
                bbox_xyxy=bundle.hand_bbox_xyxy,
                show_bbox=show_bbox,
                marker_xy=bundle.pred_joint_pixels[joint_name],
                gt_xy=bundle.gt_joint_pixels[joint_name],
            )
        metadata["models"][bundle.model_name] = bundle.metrics

    save_overview(
        path=sample_dir / "overview.png",
        image_rgb=image_rgb,
        bundles=bundles,
        layer_numbers=layer_numbers,
        joint_names=joint_names,
        cmap_name=cmap_name,
        alpha=alpha,
        reducer_name=reducer_name,
        show_bbox=show_bbox,
    )

    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    joint_names = [name.lower() for name in args.joint_names]
    invalid_joint_names = [name for name in joint_names if name not in JOINT_NAME_TO_INDEX]
    if invalid_joint_names:
        raise ValueError(f"Unsupported joint names: {invalid_joint_names}")

    layer_numbers = sorted({int(layer) for layer in args.layers})
    if not layer_numbers or min(layer_numbers) <= 0:
        raise ValueError("--layers must be positive 1-based integers.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[info] loading RGB-only checkpoint: {args.rgb_checkpoint}")
    rgb_model, rgb_cfg = load_model_bundle(args.rgb_checkpoint, device=device)
    print(f"[info] loading multimodal checkpoint: {args.multimodal_checkpoint}")
    mm_model, _ = load_model_bundle(args.multimodal_checkpoint, device=device)

    dataset = build_freihand_dataset(rgb_cfg, split=args.split)
    sample_indices = select_sample_indices(dataset, args)
    summary = {
        "split": args.split,
        "sample_indices": sample_indices,
        "layers": layer_numbers,
        "joint_names": joint_names,
        "img_feat_reducer": args.img_feat_reducer,
        "show_bbox": bool(args.show_bbox),
        "rgb_checkpoint": args.rgb_checkpoint,
        "multimodal_checkpoint": args.multimodal_checkpoint,
        "samples": [],
    }

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        real_idx = dataset.indices[sample_idx]
        sample_dir = output_dir / f"sample_{sample_idx:05d}_img_{real_idx:08d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[progress] sample_idx={sample_idx} real_idx={real_idx} -> {sample_dir}")

        bundles = [
            build_bundle("rgb_only", rgb_model, sample, layer_numbers, joint_names, args.img_feat_reducer, device),
            build_bundle("multimodal", mm_model, sample, layer_numbers, joint_names, args.img_feat_reducer, device),
        ]
        save_sample_outputs(
            sample_dir=sample_dir,
            sample_idx=sample_idx,
            real_idx=real_idx,
            bundles=bundles,
            layer_numbers=layer_numbers,
            joint_names=joint_names,
            cmap_name=args.cmap,
            alpha=float(args.alpha),
            reducer_name=args.img_feat_reducer,
            show_bbox=bool(args.show_bbox),
        )
        with open(sample_dir / "metadata.json", "r", encoding="utf-8") as f:
            summary["samples"].append(json.load(f))

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from gpgformer.models import GPGFormer, GPGFormerConfig
from gpgformer.metrics.pose_metrics import compute_pa_mpjpe


DEFAULT_OUT_DIR = "outputs/visualization"

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=int(cfg["dataset"].get("root_index", 9)),
            img_width=cfg["dataset"].get("img_width", None),
            mano_root=cfg["paths"].get("mano_dir", None),
            mano_pose_is_pca=bool(cfg["dataset"].get("mano_pose_is_pca", True)),
        )

    if name in ("ho3d", "ho-3d", "ho3d-v3", "ho3d_v3"):
        from data.ho3d_dataset import HO3DDataset

        split = cfg["dataset"].get("ho3d_eval_split", None) or cfg["dataset"].get("ho3d_val_split", None) or "val"
        return HO3DDataset(
            data_split=str(split),
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=str(cfg["dataset"].get("ho3d_version", "v3")),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            img_width=cfg["dataset"].get("img_width", None),
            input_modal=str(cfg["dataset"].get("input_modal", "RGB")),
            train=False,
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
            trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
            root_index=int(cfg["dataset"].get("root_index", 9)),
        )

    if name not in ("freihand",):
        raise ValueError(
            "Only FreiHAND, Dex-YCB and HO3D visualization are supported in this script, "
            f"got: {cfg['dataset']['name']}"
        )

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
        use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
        trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
        trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
    )


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
    # Do NOT use `torch.inference_mode()` here. Some submodules/parameters are created
    # lazily on the first forward pass (e.g., geo_tokenizer). Creating parameters under
    # InferenceMode turns them into "inference tensors", and later `load_state_dict()`
    # will fail because it performs in-place copies outside InferenceMode.
    with torch.no_grad():
        img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
        cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
        _ = model(img_dummy, cam_param=cam_dummy)


def project_points(points_3d: np.ndarray, cam_param: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = cam_param
    z = np.maximum(points_3d[:, 2], 1e-6)
    u = fx * (points_3d[:, 0] / z) + cx
    v = fy * (points_3d[:, 1] / z) + cy
    return np.stack([u, v], axis=-1)


def denorm_image(img_chw: torch.Tensor) -> np.ndarray:
    img = (img_chw.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def _as_valid_mask(valid: np.ndarray | None, n: int) -> np.ndarray:
    if valid is None:
        return np.ones((n,), dtype=bool)
    v = np.asarray(valid).reshape(-1)
    if v.size != n:
        return np.ones((n,), dtype=bool)
    return v.astype(np.float32) > 0.5


def draw_skeleton_2d(ax, joints_2d: np.ndarray, color: str, alpha: float, valid: np.ndarray | None = None) -> None:
    n = int(joints_2d.shape[0])
    m = _as_valid_mask(valid, n)
    pts = joints_2d.copy()
    pts[~m] = np.nan
    ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, alpha=alpha)
    for s, e in SKELETON_EDGES:
        if not (m[s] and m[e]):
            continue
        ax.plot(
            [joints_2d[s, 0], joints_2d[e, 0]],
            [joints_2d[s, 1], joints_2d[e, 1]],
            color=color,
            alpha=alpha,
            linewidth=1.2,
        )


def draw_skeleton_3d(
    ax, joints_3d: np.ndarray, color: str, alpha: float, label: str | None, valid: np.ndarray | None = None
) -> None:
    n = int(joints_3d.shape[0])
    m = _as_valid_mask(valid, n)
    pts = joints_3d.copy()
    pts[~m] = np.nan
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c=color, alpha=alpha, label=label)
    for s, e in SKELETON_EDGES:
        if not (m[s] and m[e]):
            continue
        ax.plot(
            [joints_3d[s, 0], joints_3d[e, 0]],
            [joints_3d[s, 1], joints_3d[e, 1]],
            [joints_3d[s, 2], joints_3d[e, 2]],
            color=color,
            alpha=alpha,
            linewidth=1.0,
        )


def set_3d_equal_axes(ax, points_a: np.ndarray, points_b: np.ndarray) -> None:
    pts = np.concatenate([points_a, points_b], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_mano_faces_np(model: GPGFormer) -> np.ndarray:
    faces = None
    # Default path: smplx MANO layer inside WiLoR wrapper.
    try:
        faces = getattr(getattr(getattr(model, "mano", None), "mano", None), "faces", None)
    except Exception:
        faces = None
    if faces is None:
        # Fallback: legacy manopth ManoLayer (FreiHAND toolbox), if present.
        try:
            faces = getattr(getattr(model, "_freihand_mano_layer", None), "th_faces", None)
        except Exception:
            faces = None
    if faces is None:
        raise AttributeError(
            "Cannot locate MANO faces on the model. Expected model.mano.mano.faces (smplx) or model._freihand_mano_layer.th_faces."
        )
    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"MANO faces must be (F,3), got {faces_np.shape}")
    return faces_np


def draw_mesh_3d(
    ax,
    verts_3d: np.ndarray,
    faces: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    if verts_3d.ndim != 2 or verts_3d.shape[1] != 3:
        raise ValueError(f"verts_3d must be (V,3), got {verts_3d.shape}")
    tris = verts_3d[faces]  # (F,3,3)
    rgb = to_rgb(color)
    coll = Poly3DCollection(tris, facecolors=[rgb], edgecolors="none", linewidths=0.0, alpha=float(alpha))
    ax.add_collection3d(coll)


def overlay_mesh_on_image(
    img_hwc_01: np.ndarray,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    cam_param: np.ndarray,
    color: str,
    alpha: float,
) -> np.ndarray:
    if img_hwc_01.ndim != 3 or img_hwc_01.shape[2] != 3:
        raise ValueError(f"img_hwc_01 must be (H,W,3), got {img_hwc_01.shape}")
    if verts_cam.ndim != 2 or verts_cam.shape[1] != 3:
        raise ValueError(f"verts_cam must be (V,3), got {verts_cam.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")
    if cam_param.shape[-1] < 4:
        raise ValueError(f"cam_param must be (4,), got {cam_param.shape}")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    h, w, _ = img_hwc_01.shape
    uv = project_points(verts_cam, cam_param[:4])  # (V,2)
    z = verts_cam[:, 2].astype(np.float32)  # (V,)

    rgb = np.asarray(to_rgb(color), dtype=np.float32)  # (3,)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    colbuf = np.zeros((h, w, 3), dtype=np.float32)
    abuf = np.zeros((h, w), dtype=np.float32)

    eps = 1e-7
    for f0, f1, f2 in faces:
        z0 = float(z[f0])
        z1 = float(z[f1])
        z2 = float(z[f2])
        if z0 <= eps or z1 <= eps or z2 <= eps:
            continue

        x0, y0 = float(uv[f0, 0]), float(uv[f0, 1])
        x1, y1 = float(uv[f1, 0]), float(uv[f1, 1])
        x2, y2 = float(uv[f2, 0]), float(uv[f2, 1])

        xmin = int(np.floor(min(x0, x1, x2)))
        xmax = int(np.ceil(max(x0, x1, x2)))
        ymin = int(np.floor(min(y0, y1, y2)))
        ymax = int(np.ceil(max(y0, y1, y2)))
        if xmax < 0 or ymax < 0 or xmin >= w or ymin >= h:
            continue

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, w - 1)
        ymax = min(ymax, h - 1)
        if xmin > xmax or ymin > ymax:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)  # (hh,ww)

        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        zpix = w0 * z0 + w1 * z1 + w2 * z2
        zcur = zbuf[ymin : ymax + 1, xmin : xmax + 1]
        upd = inside & (zpix < zcur)
        if not upd.any():
            continue

        # Simple shading for depth cues (robust to winding by using abs(n_z)).
        v0 = verts_cam[f0]
        v1 = verts_cam[f1]
        v2 = verts_cam[f2]
        n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        n_norm = float(np.linalg.norm(n) + 1e-12)
        n = n / n_norm
        intensity = 0.35 + 0.65 * float(abs(n[2]))
        tri_rgb = np.clip(rgb * intensity, 0.0, 1.0)

        zcur[upd] = zpix[upd]
        zbuf[ymin : ymax + 1, xmin : xmax + 1] = zcur
        col_slice = colbuf[ymin : ymax + 1, xmin : xmax + 1]
        a_slice = abuf[ymin : ymax + 1, xmin : xmax + 1]
        col_slice[upd] = tri_rgb
        a_slice[upd] = float(alpha)
        colbuf[ymin : ymax + 1, xmin : xmax + 1] = col_slice
        abuf[ymin : ymax + 1, xmin : xmax + 1] = a_slice

    out = img_hwc_01.astype(np.float32).copy()
    m = abuf > 0.0
    if m.any():
        a = abuf[..., None]
        out = out * (1.0 - a) + colbuf * a
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GPGFormer hand reconstruction results.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML. If omitted, uses cfg saved in checkpoint.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory for visualization images. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--fallback-out-dir",
        type=str,
        default=None,
        help=(
            "Optional fallback output directory used when the primary out-dir cannot be written (e.g., disk quota). "
            "If omitted and you explicitly set --out-dir, the script will error instead of writing elsewhere."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of samples to visualize.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset sampling.")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for saved figures.")
    parser.add_argument("--render-mesh", action="store_true", help="Also render predicted MANO mesh in the 3D subplot.")
    parser.add_argument("--mesh-color", type=str, default="#4b5563", help="Mesh surface color (matplotlib color string).")
    parser.add_argument("--mesh-alpha", type=float, default=0.35, help="Mesh surface alpha in [0,1].")
    parser.add_argument("--overlay-mesh", action="store_true", help="Render predicted MANO mesh onto the 2D image plane and overlay on RGB.")
    parser.add_argument("--overlay-mesh-color", type=str, default="#4b5563", help="Overlay mesh color (matplotlib color string).")
    parser.add_argument("--overlay-mesh-alpha", type=float, default=0.45, help="Overlay mesh alpha in [0,1].")
    parser.add_argument(
        "--camera-center",
        type=str,
        default="cam_param",
        choices=("cam_param", "patch"),
        help=(
            "Which camera center (cx,cy) to use for projection/overlay. "
            "'cam_param' uses (cx,cy) from dataloader cam_param. "
            "'patch' forces (cx,cy) to patch center (W/2,H/2) for debugging translation/center mismatch."
        ),
    )
    parser.add_argument("--save-mesh-obj", action="store_true", help="Save predicted mesh as .obj alongside images.")
    parser.add_argument(
        "--root-index",
        type=int,
        default=None,
        help="Root joint index for MPJPE/PA-MPJPE (default: cfg.dataset.root_index or 9 for FreiHAND).",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=("png", "jpg", "jpeg"),
        help="Image format for saved figures.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (only used when --image-format is jpg/jpeg).")
    parser.add_argument("--no-save-images", action="store_true", help="Compute metrics but do not save visualization images.")
    args = parser.parse_args()

    def is_write_space_error(exc: BaseException) -> bool:
        return isinstance(exc, OSError) and getattr(exc, "errno", None) in (28, 122)  # ENOSPC, EDQUOT

    out_dir_arg = DEFAULT_OUT_DIR if args.out_dir is None else args.out_dir

    def get_fallback_dir() -> Path | None:
        if args.fallback_out_dir:
            return Path(args.fallback_out_dir).expanduser()
        # Backwards-compatible behavior: only auto-fallback when --out-dir is NOT explicitly provided.
        if args.out_dir is None:
            return Path(__file__).resolve().parent / "outputs" / "visualization_fallback"
        return None

    def ensure_out_dir(primary: Path) -> Path:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except OSError as e:
            if not is_write_space_error(e):
                raise
            fallback = get_fallback_dir()
            if fallback is None:
                raise OSError(
                    e.errno,
                    f"Cannot write to out-dir: {primary} (errno={e.errno}). "
                    "Set --fallback-out-dir to enable fallback output.",
                ) from e
            print(f"[warn] Cannot write to out-dir: {primary} (errno={e.errno}). Falling back to: {fallback}")
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    out_dir = ensure_out_dir(Path(out_dir_arg).expanduser())

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    cfg_file = None
    if args.config is not None:
        cfg_file = yaml.safe_load(Path(args.config).read_text())
        print(f"Using config file: {args.config}")

    ckpt_cfg = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None
    if cfg_file is None and ckpt_cfg is None:
        raise ValueError("Checkpoint does not include cfg; please provide --config.")

    # Use cfg_file for dataset/paths by default, but prefer checkpoint cfg for model architecture to avoid
    # state_dict shape mismatches when configs drift (e.g. moge2_output=neck vs points affects geo_tokenizer).
    cfg_data = cfg_file if cfg_file is not None else ckpt_cfg
    assert cfg_data is not None

    cfg_model = cfg_data
    if ckpt_cfg is not None:
        cfg_model = copy.deepcopy(ckpt_cfg)
        if cfg_file is not None:
            cfg_model.setdefault("paths", {})
            cfg_model["paths"].update(cfg_file.get("paths", {}))
            print("[info] Using model cfg embedded in checkpoint for architecture compatibility (paths from --config).")
    assert cfg_model is not None

    root_index = int(args.root_index) if args.root_index is not None else int(cfg_data.get("dataset", {}).get("root_index", 9))
    if root_index < 0:
        raise ValueError(f"root_index must be >= 0, got {root_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model_from_cfg(cfg_model).to(device)
    warmup_lazy_modules(model, cfg_model, device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    faces_np = get_mano_faces_np(model)

    dataset = build_dataset(cfg_data)
    n_total = len(dataset)
    n_vis = min(int(args.num_samples), n_total)
    if n_vis <= 0:
        raise ValueError("num_samples must be > 0")

    rng = np.random.default_rng(int(args.seed))
    subset_indices = np.sort(rng.choice(n_total, size=n_vis, replace=False)).tolist()
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )

    metrics = []
    print(f"Visualizing {n_vis} / {n_total} samples...")
    global_i = 0

    def _is_finite_number(x) -> bool:
        if x is None:
            return False
        try:
            return bool(np.isfinite(float(x)))
        except Exception:
            return False

    def _fmt_or_na(x: float, fmt: str = "{:.2f}") -> str:
        return fmt.format(float(x)) if _is_finite_number(x) else "N/A"

    def _json_safe_number(x: float | int | None) -> float | int | None:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        xf = float(x)
        return xf if np.isfinite(xf) else None

    def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
        with path.open("w", encoding="utf-8") as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in faces:
                a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
                f.write(f"f {a} {b} {c}\n")

    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            cam_param = batch["cam_param"].to(device)
            out = model(rgb, cam_param=cam_param)

            pred_t_m = out["pred_cam_t"]  # meters
            pred_j_m = out["pred_keypoints_3d"]  # meters
            pred_v_m = out["pred_vertices"]  # meters
            # For 2D projection / overlays we need camera-space translation.
            pred_j_cam_m = pred_j_m + pred_t_m.unsqueeze(1)
            pred_v_cam_m = pred_v_m + pred_t_m.unsqueeze(1)
            gt_j = batch["keypoints_3d"].to(device)  # meters

            bsz, _, h, w = rgb.shape
            pred_j_np = pred_j_cam_m.detach().cpu().numpy()
            pred_v_np = pred_v_cam_m.detach().cpu().numpy()
            pred_j_m_np = pred_j_m.detach().cpu().numpy()
            gt_j_np = gt_j.detach().cpu().numpy()
            cam_np = cam_param.detach().cpu().numpy()
            gt_uv_norm = batch["keypoints_2d"].detach().cpu().numpy()
            uv_valid_t = batch.get("uv_valid", None)
            xyz_valid_t = batch.get("xyz_valid", None)
            img_paths = batch.get("img_path", None)
            seq_names = batch.get("seq_name", None)
            frame_ids = batch.get("frame_id", None)

            for bi in range(bsz):
                img_np = denorm_image(rgb[bi])
                cam_for_proj = cam_np[bi].copy()
                if args.camera_center == "patch":
                    cam_for_proj[2] = float(w) * 0.5
                    cam_for_proj[3] = float(h) * 0.5
                pred_uv = project_points(pred_j_np[bi], cam_for_proj)
                gt_uv = np.stack(
                    [
                        (gt_uv_norm[bi, :, 0] + 0.5) * float(w),
                        (gt_uv_norm[bi, :, 1] + 0.5) * float(h),
                    ],
                    axis=-1,
                )

                ri = int(root_index)
                if ri >= gt_j_np.shape[1]:
                    raise ValueError(f"root_index={ri} out of range for keypoints_3d with N={gt_j_np.shape[1]}")

                uv_valid_np = None
                if uv_valid_t is not None:
                    uv_valid_np = uv_valid_t[bi].detach().cpu().numpy().reshape(-1)
                m2d = _as_valid_mask(uv_valid_np, int(gt_uv.shape[0]))
                has_gt_2d = bool(m2d.any())

                has_gt_3d = True
                if xyz_valid_t is not None:
                    has_gt_3d = bool(float(xyz_valid_t[bi].item()) > 0.5)

                pred_j_root = pred_j_m_np[bi] - pred_j_m_np[bi, ri : ri + 1]
                pred_v_root = pred_v_np[bi] - pred_j_np[bi, ri : ri + 1]
                gt_j_root = gt_j_np[bi] - gt_j_np[bi, ri : ri + 1] if has_gt_3d else None

                err_2d_px = float("nan")
                if has_gt_2d:
                    err_2d_px = float(np.linalg.norm(pred_uv[m2d] - gt_uv[m2d], axis=-1).mean())

                # Dataset/camera consistency check: GT 3D joints should reproject to GT 2D joints.
                # This uses the original cam_param from the dataloader (regardless of --camera-center).
                gt_reproj_err_2d_px = float("nan")
                if has_gt_2d and has_gt_3d:
                    gt_uv_reproj = project_points(gt_j_np[bi], cam_np[bi])
                    gt_reproj_err_2d_px = float(np.linalg.norm(gt_uv_reproj[m2d] - gt_uv[m2d], axis=-1).mean())

                err_3d_mm = float("nan")
                err_pa_3d_mm = float("nan")
                err_abs_3d_mm = float("nan")
                if has_gt_3d and gt_j_root is not None:
                    # Match training/eval definition:
                    # - use out["pred_keypoints_3d"] (no pred_cam_t)
                    # - root-center both with cfg.dataset.root_index (default 9)
                    err_3d_mm = float(np.linalg.norm(pred_j_root - gt_j_root, axis=-1).mean() * 1000.0)
                    err_pa_3d_mm = float(
                        compute_pa_mpjpe(torch.from_numpy(pred_j_root), torch.from_numpy(gt_j_root)) * 1000.0
                    )
                    # Absolute MPJPE in camera space (no root-centering): includes the effect of translation.
                    err_abs_3d_mm = float(np.linalg.norm(pred_j_np[bi] - gt_j_np[bi], axis=-1).mean() * 1000.0)

                sample_idx = subset_indices[global_i]
                img_path = None
                if isinstance(img_paths, (list, tuple)) and bi < len(img_paths):
                    img_path = str(img_paths[bi])
                seq_name = None
                if isinstance(seq_names, (list, tuple)) and bi < len(seq_names):
                    seq_name = str(seq_names[bi])
                frame_id = None
                if isinstance(frame_ids, (list, tuple)) and bi < len(frame_ids):
                    frame_id = str(frame_ids[bi])
                metrics.append(
                    {
                        "sample_index": int(sample_idx),
                        "img_path": img_path,
                        "seq_name": seq_name,
                        "frame_id": frame_id,
                        "has_gt_2d": bool(has_gt_2d),
                        "has_gt_3d": bool(has_gt_3d),
                        "mean_2d_error_px": _json_safe_number(err_2d_px),
                        "gt_2d_reproj_error_px": _json_safe_number(gt_reproj_err_2d_px),
                        "cam_center_offset_px": [
                            _json_safe_number(float(cam_np[bi][2] - float(w) * 0.5)),
                            _json_safe_number(float(cam_np[bi][3] - float(h) * 0.5)),
                        ],
                        "num_uv_valid": int(m2d.sum()),
                        "mpjpe_mm": _json_safe_number(err_3d_mm),
                        "pa_mpjpe_mm": _json_safe_number(err_pa_3d_mm),
                        "abs_mpjpe_mm": _json_safe_number(err_abs_3d_mm),
                        "root_index": int(root_index),
                    }
                )

                if args.save_mesh_obj:
                    obj_name = f"sample_{global_i:03d}_idx_{sample_idx}.obj"
                    try:
                        write_obj(out_dir / obj_name, pred_v_np[bi], faces_np)
                    except OSError as e:
                        if not is_write_space_error(e):
                            raise
                        fallback_dir = get_fallback_dir()
                        if fallback_dir is None:
                            raise OSError(
                                e.errno,
                                f"Failed to save obj to out-dir: {out_dir} (errno={e.errno}). "
                                "Set --fallback-out-dir to enable fallback output.",
                            ) from e
                        fallback = ensure_out_dir(fallback_dir)
                        if fallback != out_dir:
                            print(
                                f"[warn] Failed to save obj to {out_dir} (errno={e.errno}). "
                                f"Saving remaining outputs to fallback dir: {fallback}"
                            )
                            out_dir = fallback
                        write_obj(out_dir / obj_name, pred_v_np[bi], faces_np)

                if not args.no_save_images:
                    fig = plt.figure(figsize=(12, 4))
                    try:
                        ax1 = fig.add_subplot(1, 2, 1)
                        img_vis = img_np
                        if args.overlay_mesh:
                            img_vis = overlay_mesh_on_image(
                                img_hwc_01=img_np,
                                verts_cam=pred_v_np[bi],
                                faces=faces_np,
                                cam_param=cam_for_proj,
                                color=str(args.overlay_mesh_color),
                                alpha=float(args.overlay_mesh_alpha),
                            )
                        ax1.imshow(img_vis)
                        if has_gt_2d:
                            draw_skeleton_2d(ax1, gt_uv, color="#22c55e", alpha=0.9, valid=m2d.astype(np.float32))
                        draw_skeleton_2d(ax1, pred_uv, color="#ef4444", alpha=0.8)
                        ax1.set_title(f"idx={sample_idx} | 2D err={_fmt_or_na(err_2d_px)}px")
                        ax1.set_xlim([0, w])
                        ax1.set_ylim([h, 0])
                        ax1.set_aspect("equal")

                        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
                        if args.render_mesh:
                            draw_mesh_3d(
                                ax2,
                                pred_v_root,
                                faces_np,
                                color=str(args.mesh_color),
                                alpha=float(args.mesh_alpha),
                            )
                        if has_gt_3d and gt_j_root is not None:
                            draw_skeleton_3d(ax2, gt_j_root, color="#22c55e", alpha=0.9, label="GT")
                        draw_skeleton_3d(ax2, pred_j_root, color="#ef4444", alpha=0.8, label="Pred")
                        ref_a = gt_j_root if (has_gt_3d and gt_j_root is not None) else pred_j_root
                        ref_b = pred_v_root if args.render_mesh else pred_j_root
                        set_3d_equal_axes(ax2, ref_a, ref_b)
                        ax2.set_title(
                            f"3D root-relative (root={root_index}) | MPJPE={_fmt_or_na(err_3d_mm)}mm | PA={_fmt_or_na(err_pa_3d_mm)}mm"
                        )
                        ax2.legend(loc="upper right")
                        ax2.set_xlabel("X")
                        ax2.set_ylabel("Y")
                        ax2.set_zlabel("Z")
                        ax2.view_init(elev=20, azim=-60)

                        fig.tight_layout()

                        ext = "jpg" if args.image_format in ("jpg", "jpeg") else "png"
                        save_name = f"sample_{global_i:03d}_idx_{sample_idx}.{ext}"

                        def save_to(dir_path: Path) -> None:
                            save_path = dir_path / save_name
                            if ext == "jpg":
                                fig.savefig(
                                    save_path,
                                    dpi=int(args.dpi),
                                    format="jpg",
                                    pil_kwargs={"quality": int(args.jpeg_quality), "optimize": True},
                                )
                            else:
                                fig.savefig(save_path, dpi=int(args.dpi))

                        try:
                            save_to(out_dir)
                        except OSError as e:
                            if not is_write_space_error(e):
                                raise
                            fallback_dir = get_fallback_dir()
                            if fallback_dir is None:
                                raise OSError(
                                    e.errno,
                                    f"Failed to save to out-dir: {out_dir} (errno={e.errno}). "
                                    "Set --fallback-out-dir to enable fallback output.",
                                ) from e
                            fallback = ensure_out_dir(fallback_dir)
                            if fallback != out_dir:
                                print(
                                    f"[warn] Failed to save to {out_dir} (errno={e.errno}). "
                                    f"Saving remaining images to fallback dir: {fallback}"
                                )
                                out_dir = fallback
                            save_to(out_dir)
                    finally:
                        plt.close(fig)
                global_i += 1

    mean_2d_vals = [float(m["mean_2d_error_px"]) for m in metrics if _is_finite_number(m["mean_2d_error_px"])]
    mean_gt_reproj_2d_vals = [
        float(m["gt_2d_reproj_error_px"]) for m in metrics if _is_finite_number(m.get("gt_2d_reproj_error_px"))
    ]
    mean_3d_vals = [float(m["mpjpe_mm"]) for m in metrics if _is_finite_number(m["mpjpe_mm"])]
    mean_pa_3d_vals = [float(m["pa_mpjpe_mm"]) for m in metrics if _is_finite_number(m["pa_mpjpe_mm"])]
    mean_abs_3d_vals = [float(m["abs_mpjpe_mm"]) for m in metrics if _is_finite_number(m["abs_mpjpe_mm"])]
    mean_2d = float(np.mean(mean_2d_vals)) if mean_2d_vals else float("nan")
    mean_gt_reproj_2d = float(np.mean(mean_gt_reproj_2d_vals)) if mean_gt_reproj_2d_vals else float("nan")
    mean_3d = float(np.mean(mean_3d_vals)) if mean_3d_vals else float("nan")
    mean_pa_3d = float(np.mean(mean_pa_3d_vals)) if mean_pa_3d_vals else float("nan")
    mean_abs_3d = float(np.mean(mean_abs_3d_vals)) if mean_abs_3d_vals else float("nan")
    summary = {
        "checkpoint": args.ckpt,
        "num_samples": len(metrics),
        "mean_2d_error_px": _json_safe_number(mean_2d),
        "mean_2d_reproj_error_px": _json_safe_number(mean_gt_reproj_2d),
        "mean_mpjpe_mm": _json_safe_number(mean_3d),
        "mean_pa_mpjpe_mm": _json_safe_number(mean_pa_3d),
        "mean_abs_mpjpe_mm": _json_safe_number(mean_abs_3d),
        "root_index": int(root_index),
        "camera_center_mode": str(args.camera_center),
        "samples": metrics,
    }
    summary_path = out_dir / "summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError as e:
        if not is_write_space_error(e):
            raise
        fallback_dir = get_fallback_dir()
        if fallback_dir is None:
            raise OSError(
                e.errno,
                f"Failed to write summary to out-dir: {out_dir} (errno={e.errno}). "
                "Set --fallback-out-dir to enable fallback output.",
            ) from e
        fallback = ensure_out_dir(fallback_dir)
        if fallback != out_dir:
            print(f"[warn] Failed to write summary to {out_dir} (errno={e.errno}). Writing summary to: {fallback}")
            out_dir = fallback
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.no_save_images:
        print("Visualization images were not saved (--no-save-images).")
    else:
        print(f"Saved {len(metrics)} visualization images to: {out_dir}")
    # print(f"Mean 2D error: {mean_2d:.3f}px")
    print(f"Mean pred 2D error (px): {mean_2d:.3f}px")
    print(f"Mean GT 2D reprojection error (px): {mean_gt_reproj_2d:.3f}px")
    print(f"Mean MPJPE: {mean_3d:.3f}mm (root_index={root_index})")
    print(f"Mean PA-MPJPE: {mean_pa_3d:.3f}mm (root_index={root_index})")
    print("Mean abs MPJPE: {:.3f}mm (no root-center; uses pred_cam_t for camera-space joints)".format(mean_abs_3d))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

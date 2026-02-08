"""
Compute dataset-level consistency between:
  - GT 3D joints (xyz)
  - MANO(GT params) decoded joints

Supported datasets (via GPGFormer configs):
  - FreiHAND
  - HO3D v3
  - Dex-YCB

Outputs:
  - MPJPE (root-centered) mean in mm
  - PA-MPJPE (root-centered) mean in mm

Notes
-----
1) This script is meant as a sanity-check / lower-bound diagnostic:
   If MANO(GT params) decoded joints already disagree with GT xyz by a large margin,
   then supervising both MANO params and xyz can be fundamentally conflicting.

2) Root-centering removes global translation; therefore MANO translation (if present)
   is not required for this check.

Usage
-----
PYTHONPATH=/root/code/hand_reconstruction/GPGFormer \
  python scripts/gt_xyz_vs_mano_gtparams_metrics.py --config configs/config_freihand.yaml --split val --num-samples 512
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running this script directly from any working directory (no need to export PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise ImportError("This script requires PyYAML (pip install pyyaml).") from e

from smplx.lbs import vertices2joints

from gpgformer.metrics.pose_metrics import compute_pa_mpjpe
from gpgformer.models.mano.mano_layer import MANOLayer, MANOConfig
from third_party.wilor_min.wilor.utils.geometry import aa_to_rotmat


class FreiHANDRaw:
    def __init__(self, root_dir: str, eval_root: Optional[str], split: str):
        split = "train" if split == "train" else "evaluation"
        if split == "evaluation":
            if eval_root is None:
                eval_root = f"{root_dir}_eval"
            if not os.path.isdir(eval_root):
                raise FileNotFoundError(f"FreiHAND eval root not found: {eval_root}")
            base_root = eval_root
            prefix = "evaluation"
        else:
            base_root = root_dir
            prefix = "training"

        xyz_path = os.path.join(base_root, f"{prefix}_xyz.json")
        mano_path = os.path.join(base_root, f"{prefix}_mano.json")
        self.xyz_list = json.load(open(xyz_path, "r"))
        self.mano_list = json.load(open(mano_path, "r"))
        if len(self.xyz_list) != len(self.mano_list):
            raise ValueError("FreiHAND xyz/mano lengths mismatch.")

    def __len__(self):
        return len(self.xyz_list)

    def __getitem__(self, idx):
        xyz = np.array(self.xyz_list[idx], dtype=np.float32)  # meters
        mp = np.array(self.mano_list[idx][0], dtype=np.float32).reshape(-1)
        if mp.shape[0] < 58:
            raise ValueError("FreiHAND MANO params malformed.")
        mano_pose = mp[:48].copy()
        mano_shape = mp[48:58].copy()
        return {
            "keypoints_3d": torch.from_numpy(xyz),
            "mano_pose": torch.from_numpy(mano_pose),
            "mano_shape": torch.from_numpy(mano_shape),
        }


class HO3DRaw:
    def __init__(
        self,
        root_dir: str,
        split: str,
        dataset_version: str = "v3",
        trainval_ratio: float = 0.9,
        trainval_seed: int = 42,
        trainval_split_by: str = "sequence",
        val_split: str = "val",
    ):
        candidate_root = os.path.join(root_dir, f"HO3D_{dataset_version}")
        self.root_dir = candidate_root if os.path.exists(candidate_root) else root_dir
        self.trainval_ratio = float(trainval_ratio)
        self.trainval_seed = int(trainval_seed)
        self.trainval_split_by = str(trainval_split_by).lower()

        split = split.lower()
        if split == "val":
            split = str(val_split).lower()
        if split == "test":
            split = "evaluation"

        train_txt = os.path.join(self.root_dir, "train.txt")
        eval_txt = os.path.join(self.root_dir, "evaluation.txt")
        if split in ("train", "val", "train_all"):
            lines = _read_split_lines(train_txt)
            if split == "train_all":
                selected = lines
            else:
                train_lines, val_lines = _split_train_lines(lines, self.trainval_ratio, self.trainval_seed, self.trainval_split_by)
                selected = train_lines if split == "train" else val_lines
            split_folder = "train"
        elif split == "evaluation":
            selected = _read_split_lines(eval_txt)
            split_folder = "evaluation"
        else:
            raise ValueError(f"Unsupported HO3D split: {split}")

        self.records: List[Dict] = []
        for line in selected:
            parts = line.split("/")
            if len(parts) != 2:
                continue
            seq_name, file_id = parts
            meta_path = os.path.join(self.root_dir, split_folder, seq_name, "meta", f"{file_id}.pkl")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            # Require full annotations
            if meta.get("handPose") is None or meta.get("handJoints3D") is None:
                continue
            hand_pose = np.array(meta["handPose"], dtype=np.float32)  # (48,)
            hand_beta = np.array(meta["handBeta"], dtype=np.float32)  # (10,)
            hand_trans = np.array(meta["handTrans"], dtype=np.float32)  # (3,)
            hand_joints_3d = np.array(meta["handJoints3D"], dtype=np.float32)  # (21,3) meters

            self.records.append(
                {
                    "keypoints_3d": hand_joints_3d,
                    "mano_pose": hand_pose,
                    "mano_shape": hand_beta,
                    "mano_trans": hand_trans,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "keypoints_3d": torch.from_numpy(r["keypoints_3d"]),
            "mano_pose": torch.from_numpy(r["mano_pose"]),
            "mano_shape": torch.from_numpy(r["mano_shape"]),
            "mano_trans": torch.from_numpy(r["mano_trans"]),
        }


class DexYCBRaw:
    def __init__(self, root_dir: str, setup: str, split: str):
        root_dir = root_dir.rstrip("/")
        candidate_root = os.path.join(root_dir, "dex-ycb")
        dex_root = candidate_root if os.path.exists(os.path.join(candidate_root, "annotations")) else root_dir
        annot_path = os.path.join(dex_root, "annotations")
        if split == "val":
            split = "test"
        json_path = os.path.join(annot_path, f"DEX_YCB_{setup}_{split}_data.json")
        with open(json_path, "r") as f:
            db = json.load(f)
        self.records: List[Dict] = []
        for ann in db.get("annotations", []):
            joints_coord_cam = np.array(ann["joints_coord_cam"], dtype=np.float32).reshape(21, 3) / 1000.0  # meters
            mano_pose = np.array(ann["mano_param"]["pose"], dtype=np.float32)
            mano_shape = np.array(ann["mano_param"]["shape"], dtype=np.float32)
            mano_trans = np.array(ann["mano_param"]["trans"], dtype=np.float32)
            hand_type = ann.get("mano_param", {}).get("hand_type", None)
            self.records.append(
                {
                    "keypoints_3d": joints_coord_cam,
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "mano_trans": mano_trans,
                    "hand_type": hand_type,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "keypoints_3d": torch.from_numpy(r["keypoints_3d"]),
            "mano_pose": torch.from_numpy(r["mano_pose"]),
            "mano_shape": torch.from_numpy(r["mano_shape"]),
            "mano_trans": torch.from_numpy(r["mano_trans"]),
            "hand_type": r.get("hand_type", None),
        }


def _read_split_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def _split_train_lines(lines: List[str], ratio: float, seed: int, split_by: str) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    split_by = split_by.lower()
    if split_by == "frame":
        lines = list(lines)
        rng.shuffle(lines)
        n_train = int(len(lines) * ratio)
        return lines[:n_train], lines[n_train:]

    # default: split by sequence
    seq_to_lines: Dict[str, List[str]] = {}
    for ln in lines:
        seq = ln.split("/")[0]
        seq_to_lines.setdefault(seq, []).append(ln)
    seqs = list(seq_to_lines.keys())
    rng.shuffle(seqs)
    n_train = int(len(lines) * ratio)
    train_out: List[str] = []
    val_out: List[str] = []
    count = 0
    for seq in seqs:
        bucket = seq_to_lines[seq]
        if count < n_train:
            train_out.extend(bucket)
            count += len(bucket)
        else:
            val_out.extend(bucket)
    if len(train_out) == 0 and len(val_out) > 0:
        train_out.append(val_out.pop(0))
    if len(val_out) == 0 and len(train_out) > 1:
        val_out.append(train_out.pop())
    return train_out, val_out


def _build_dataset(cfg: dict, split: str, loader: str):
    name = str(cfg["dataset"]["name"]).lower()
    align_wilor_aug = bool(cfg["dataset"].get("align_wilor_aug", True))
    bbox_source = "gt" if split == "train" else cfg["dataset"].get("bbox_source_eval", "detector")
    detector_path = cfg["paths"].get("detector_ckpt", None) if bbox_source == "detector" else None
    root_index = int(cfg.get("dataset", {}).get("root_index", 9))

    if loader == "raw":
        if name in ("freihand",):
            return FreiHANDRaw(
                root_dir=cfg["paths"]["freihand_root"],
                eval_root=cfg["paths"].get("freihand_eval_root", None),
                split=split,
            )
        if name in ("ho3d",):
            return HO3DRaw(
                root_dir=cfg["paths"]["ho3d_root"],
                split=split,
                dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
                trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
                trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
                trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
                val_split=str(cfg["dataset"].get("ho3d_val_split", "val")),
            )
        if name in ("dexycb", "dex-ycb"):
            return DexYCBRaw(
                root_dir=cfg["paths"]["dexycb_root"],
                setup=cfg["dataset"]["dexycb_setup"],
                split=split,
            )
        raise ValueError(f"Unsupported dataset for raw loader: {cfg['dataset']['name']}")

    if name in ("freihand",):
        from data.freihand_dataset import FreiHANDDataset

        return FreiHANDDataset(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
        )

    if name in ("ho3d",):
        from data.ho3d_dataset import HO3DDataset

        ho3d_val_split = str(cfg["dataset"].get("ho3d_val_split", "val")).lower()
        if split == "train":
            data_split = "train"
        else:
            data_split = ho3d_val_split

        return HO3DDataset(
            data_split=data_split,
            root_dir=cfg["paths"]["ho3d_root"],
            dataset_version=cfg["dataset"].get("ho3d_version", "v3"),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
            trainval_ratio=float(cfg["dataset"].get("ho3d_trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("ho3d_trainval_seed", 42)),
            trainval_split_by=str(cfg["dataset"].get("ho3d_trainval_split_by", "sequence")),
        )

    if name in ("dexycb", "dex-ycb"):
        from data.dex_ycb_dataset import DexYCBDataset

        return DexYCBDataset(
            setup=cfg["dataset"]["dexycb_setup"],
            split=("train" if split == "train" else "test"),
            root_dir=cfg["paths"]["dexycb_root"],
            img_size=int(cfg["dataset"].get("img_size", 256)),
            input_modal="RGB",
            train=(split == "train"),
            align_wilor_aug=align_wilor_aug,
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=bbox_source,
            detector_weights_path=detector_path,
            root_index=root_index,
        )

    raise ValueError(f"Unsupported dataset.name: {cfg['dataset']['name']}")


def _infer_unit_to_mm(xyz: torch.Tensor) -> torch.Tensor:
    """
    Heuristic: if coordinates look like meters (|x| < ~10), convert to mm.
    If they already look like mm (|x| ~ 100+), keep.
    """
    m = float(xyz.abs().max().item()) if xyz.numel() else 0.0
    if m <= 10.0:
        return xyz * 1000.0
    return xyz


def _mano_from_batch(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (global_orient_R, hand_pose_R, betas) as rotmats/betas tensors on device.
    """
    if "mano_pose" in batch and batch["mano_pose"] is not None and "mano_shape" in batch and batch["mano_shape"] is not None:
        pose_aa = batch["mano_pose"].to(device=device, dtype=torch.float32)  # (B,48)
        betas = batch["mano_shape"].to(device=device, dtype=torch.float32)  # (B,10)
        go_aa = pose_aa[:, :3]
        hp_aa = pose_aa[:, 3:].reshape(-1, 3)
        go_R = aa_to_rotmat(go_aa).view(-1, 1, 3, 3)  # (B,1,3,3)
        hp_R = aa_to_rotmat(hp_aa).view(-1, 15, 3, 3)  # (B,15,3,3)
        return go_R, hp_R, betas

    if "mano_params" in batch and isinstance(batch["mano_params"], dict):
        mp = batch["mano_params"]
        go_aa = torch.as_tensor(mp["global_orient"], device=device, dtype=torch.float32)  # (B,3) or (3,)
        hp_aa = torch.as_tensor(mp["hand_pose"], device=device, dtype=torch.float32)  # (B,45) or (45,)
        betas = torch.as_tensor(mp["betas"], device=device, dtype=torch.float32)  # (B,10) or (10,)
        if go_aa.dim() == 1:
            go_aa = go_aa.unsqueeze(0)
        if hp_aa.dim() == 1:
            hp_aa = hp_aa.unsqueeze(0)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        go_R = aa_to_rotmat(go_aa).view(-1, 1, 3, 3)
        hp_R = aa_to_rotmat(hp_aa.view(-1, 3)).view(-1, 15, 3, 3)
        return go_R, hp_R, betas

    raise KeyError("Batch does not contain GT MANO params: expected (mano_pose+mano_shape) or mano_params dict.")


def _freihand_kp21_from_mano(vertices_m: torch.Tensor, mano_layer: MANOLayer) -> torch.Tensor:
    """
    Build FreiHAND official 21 keypoints (meters) from:
      - MANO 16 joints (J_regressor)
      - fingertip vertices (official IDs: 744, 320, 443, 555, 672)
    Returns: (B,21,3) meters
    """
    J_reg = getattr(mano_layer.mano, "J_regressor", None)
    if J_reg is None:
        raise AttributeError("Underlying MANO layer is missing J_regressor.")

    j16 = vertices2joints(J_reg, vertices_m)  # (B,16,3)
    B = vertices_m.shape[0]
    kp21 = torch.zeros((B, 21, 3), device=vertices_m.device, dtype=vertices_m.dtype)

    mano_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], device=vertices_m.device, dtype=torch.long)
    dst_ids = torch.tensor([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3], device=vertices_m.device, dtype=torch.long)
    kp21.index_copy_(1, dst_ids, j16.index_select(1, mano_ids))

    tip_vids = torch.tensor([744, 320, 443, 555, 672], device=vertices_m.device, dtype=torch.long)
    tip_dst = torch.tensor([4, 8, 12, 16, 20], device=vertices_m.device, dtype=torch.long)
    kp21.index_copy_(1, tip_dst, vertices_m.index_select(1, tip_vids))
    return kp21


_FREIHAND_MANOPTH_LAYER_CACHE: Dict[str, object] = {}


def _freihand_kp21_from_verts_mm(verts_mm: torch.Tensor, J_reg: torch.Tensor) -> torch.Tensor:
    """
    FreiHAND official 21 keypoints from:
      - MANO 16 joints (J_regressor)
      - fingertip vertices (IDs: 744, 320, 443, 555, 672)
    Inputs:
      verts_mm: (B,778,3) in mm
      J_reg:   (16,778) regressor
    Returns:
      kp21_mm: (B,21,3) in mm, OpenPose-hand order
    """
    # (B,16,3)
    j16_mm = torch.einsum("ji,bik->bjk", J_reg, verts_mm)
    B = verts_mm.shape[0]
    kp21_mm = torch.zeros((B, 21, 3), device=verts_mm.device, dtype=verts_mm.dtype)

    mano_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], device=verts_mm.device, dtype=torch.long)
    dst_ids = torch.tensor([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3], device=verts_mm.device, dtype=torch.long)
    kp21_mm.index_copy_(1, dst_ids, j16_mm.index_select(1, mano_ids))

    tip_vids = torch.tensor([744, 320, 443, 555, 672], device=verts_mm.device, dtype=torch.long)
    tip_dst = torch.tensor([4, 8, 12, 16, 20], device=verts_mm.device, dtype=torch.long)
    kp21_mm.index_copy_(1, tip_dst, verts_mm.index_select(1, tip_vids))
    return kp21_mm


def _decode_mano_freihand_manopth(
    mano_pose: torch.Tensor,
    mano_shape: torch.Tensor,
    device: torch.device,
    mano_root: Optional[Path] = None,
) -> torch.Tensor:
    """
    Decode MANO(GT params) for FreiHAND using the same legacy MANO assets
    as the official FreiHAND toolbox (`freihand/data/MANO_RIGHT.pkl`).

    We use a torch MANO layer (manopth) to load the legacy .pkl and then
    assemble FreiHAND's official 21 keypoints definition.
    Returns: (B,21,3) in meters.
    """
    global _FREIHAND_MANOPTH_LAYER_CACHE
    try:
        from manopth.manolayer import ManoLayer
    except Exception:
        # Fallback to a local manopth copy if available in this repo.
        fallback_roots = [
            Path(__file__).resolve().parents[2] / "HandGCAT" / "common" / "utils" / "manopth",
            Path(__file__).resolve().parents[2] / "HandOccNet" / "common" / "utils" / "manopth",
        ]
        for root in fallback_roots:
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
                break
        from manopth.manolayer import ManoLayer  # type: ignore

    if mano_root is None:
        mano_root = Path(__file__).resolve().parents[2] / "freihand" / "data"
    mano_root = Path(mano_root)
    if not (mano_root / "MANO_RIGHT.pkl").exists():
        raise FileNotFoundError(
            f"MANO_RIGHT.pkl not found at {mano_root}/MANO_RIGHT.pkl. "
            "For FreiHAND toolbox assets, run: python freihand/setup_mano.py <MANO_PATH>"
        )

    cache_key = str(mano_root.resolve())
    if cache_key in _FREIHAND_MANOPTH_LAYER_CACHE:
        mano_layer = _FREIHAND_MANOPTH_LAYER_CACHE[cache_key]
    else:
        mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side="right",
            mano_root=str(mano_root),
            use_pca=False,
        ).to(device)
        mano_layer.eval()
        _FREIHAND_MANOPTH_LAYER_CACHE[cache_key] = mano_layer

    pose = mano_pose.to(device=device, dtype=torch.float32)
    betas = mano_shape.to(device=device, dtype=torch.float32)
    trans = torch.zeros((pose.shape[0], 3), device=device, dtype=torch.float32)

    verts_mm, _ = mano_layer(pose, betas, trans)  # (B,778,3) mm
    # Use regressor from the layer to build the FreiHAND kp21 definition
    J_reg = getattr(mano_layer, "th_J_regressor", None)
    if J_reg is None:
        raise AttributeError("manopth ManoLayer missing th_J_regressor")
    kp21_mm = _freihand_kp21_from_verts_mm(verts_mm, J_reg)
    return kp21_mm / 1000.0


_HO3D_MANO_MODEL: Optional[object] = None


def _decode_mano_ho3d(
    mano_pose: torch.Tensor,
    mano_shape: torch.Tensor,
    mano_trans: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    HO3D visualization uses legacy MANO (chumpy) forwardKinematics:
      - m.J_transformed.r -> joints in MANO order (mm)
      - jointsMapManoToSimple reorders to the simple hand layout
    We follow the same and convert to meters.
    """
    global _HO3D_MANO_MODEL
    try:
        import chumpy as ch  # noqa: F401
        from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
    except Exception as e:  # pragma: no cover
        raise ImportError("HO3D MANO decoder requires ho3d/mano (chumpy) setup. Run ho3d/setup_mano.py first.") from e

    mano_root = Path(__file__).resolve().parents[2] / "ho3d" / "mano" / "models" / "MANO_RIGHT.pkl"
    if not mano_root.exists():
        raise FileNotFoundError(f"HO3D MANO model not found: {mano_root}")

    if _HO3D_MANO_MODEL is None:
        _HO3D_MANO_MODEL = load_model(str(mano_root), ncomps=6, flat_hand_mean=True)

    # jointsMapManoToSimple from ho3d/vis_HO3D.py
    joints_map = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

    B = mano_pose.shape[0]
    out = []
    for i in range(B):
        fullpose = mano_pose[i].detach().cpu().numpy().astype(np.float32)  # (48,)
        beta = mano_shape[i].detach().cpu().numpy().astype(np.float32)     # (10,)
        if mano_trans is None:
            trans = np.zeros((3,), dtype=np.float32)
        else:
            trans = mano_trans[i].detach().cpu().numpy().astype(np.float32)

        m = _HO3D_MANO_MODEL
        m.fullpose[:] = fullpose
        m.trans[:] = trans
        m.betas[:] = beta
        j = np.array(m.J_transformed.r, dtype=np.float32)  # (21,3) in MANO order (mm)
        j = j[joints_map]  # reorder to simple order
        j = j / 1000.0  # mm -> meters
        out.append(j)

    return torch.from_numpy(np.stack(out, axis=0)).to(device=device)


_DEXYCB_MANO_LAYER_CACHE: Dict[str, object] = {}


def _decode_mano_dexycb(
    mano_pose: torch.Tensor,
    mano_shape: torch.Tensor,
    mano_trans: Optional[torch.Tensor],
    hand_type: Optional[torch.Tensor],
    device: torch.device,
    mano_root: Optional[Path] = None,
) -> torch.Tensor:
    """
    Dex-YCB visualization uses manopth ManoLayer:
      ManoLayer(flat_hand_mean=False, ncomps=45, use_pca=True, side=right/left)
      vert, j = mano_layer(pose, betas, trans)
      vert/j are in mm -> convert to meters
    """
    try:
        from manopth.manolayer import ManoLayer
    except Exception:
        # Fallback to a local manopth copy if available in this repo.
        fallback_roots = [
            Path(__file__).resolve().parents[2] / "HandGCAT" / "common" / "utils" / "manopth",
            Path(__file__).resolve().parents[2] / "HandOccNet" / "common" / "utils" / "manopth",
        ]
        for root in fallback_roots:
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
                break
        try:
            from manopth.manolayer import ManoLayer
        except Exception as e:  # pragma: no cover
            raise ImportError("Dex-YCB MANO decoder requires manopth. Please install manopth in your env.") from e

    B = mano_pose.shape[0]
    out = []
    for i in range(B):
        side = "right"
        if hand_type is not None:
            ht = hand_type[i]
            if isinstance(ht, torch.Tensor):
                ht = ht.item()
            if isinstance(ht, (bytes, str)):
                side = "left" if str(ht).lower().startswith("l") else "right"
            else:
                side = "right" if float(ht) >= 0.5 else "left"

        cache_key = f"{side}:{mano_root}" if mano_root is not None else side
        if cache_key in _DEXYCB_MANO_LAYER_CACHE:
            mano_layer = _DEXYCB_MANO_LAYER_CACHE[cache_key]
        else:
            mano_layer = ManoLayer(
                flat_hand_mean=False,
                ncomps=45,
                side=side,
                mano_root=str(mano_root) if mano_root is not None else "mano/models",
                use_pca=True,
            )
            _DEXYCB_MANO_LAYER_CACHE[cache_key] = mano_layer

        pose = mano_pose[i].view(1, 48).to(device=device)
        betas = mano_shape[i].view(1, 10).to(device=device)
        if mano_trans is None:
            trans = torch.zeros((1, 3), device=device, dtype=pose.dtype)
        else:
            trans = mano_trans[i].view(1, 3).to(device=device)
        _, j = mano_layer(pose, betas, trans)  # (1,21,3) in mm
        j = (j / 1000.0).detach().cpu().numpy()
        out.append(j[0])

    return torch.from_numpy(np.stack(out, axis=0)).to(device=device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="GPGFormer YAML config path.")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Which split to sample from.")
    ap.add_argument("--num-samples", type=int, default=512, help="How many samples to evaluate (max).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--root-index", type=int, default=None, help="Override dataset.root_index for centering.")
    ap.add_argument(
        "--freihand-mano-root",
        type=str,
        default=None,
        help=(
            "FreiHAND only. Directory containing MANO_RIGHT.pkl used for decoding. "
            "Default: <repo>/freihand/data . "
            "Example to test GPGFormer weights MANO: /root/code/hand_reconstruction/GPGFormer/weights/mano"
        ),
    )
    ap.add_argument(
        "--loader",
        type=str,
        default="raw",
        choices=["raw", "orig"],
        help="Which dataloader to use. 'raw' = clean loader (no augmentation). 'orig' = GPGFormer dataset loader.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = _build_dataset(cfg, split=args.split, loader=args.loader)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    # Default MANO (from config)
    mano_cfg = MANOConfig(model_path=cfg["paths"]["mano_dir"], mean_params=cfg["paths"]["mano_mean_params"])
    mano_layer = MANOLayer(mano_cfg).to(device)
    mano_layer.eval()

    root_index = int(args.root_index) if args.root_index is not None else int(cfg.get("dataset", {}).get("root_index", 9))
    dataset_name = str(cfg["dataset"]["name"]).lower()
    mano_decoder = str(cfg.get("model", {}).get("mano_decoder", "wilor")).lower()
    freihand_mano_root_cfg = cfg.get("model", {}).get("freihand_mano_root", None)

    # # For FreiHAND, prefer the dataset's own MANO assets if available
    # if dataset_name == "freihand":
    #     freihand_mano_dir = Path(__file__).resolve().parents[2] / "freihand" / "data"
    #     if (freihand_mano_dir / "MANO_RIGHT.pkl").exists():
    #         mano_cfg = MANOConfig(model_path=str(freihand_mano_dir), mean_params=cfg["paths"]["mano_mean_params"])
    #         mano_layer = MANOLayer(mano_cfg).to(device)
    #         mano_layer.eval()

    n = min(int(args.num_samples), len(ds))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))

    mpjpe_sum = 0.0
    pamp_sum = 0.0
    count = 0

    for batch in loader:
        if count >= n:
            break

        # GT xyz
        gt = batch.get("joints_3d_gt", None)
        if gt is None:
            gt = batch.get("keypoints_3d", None)
        if gt is None:
            raise KeyError("Missing GT joints in batch: expected joints_3d_gt or keypoints_3d")
        gt = gt.to(device=device, dtype=torch.float32)
        gt_mm = _infer_unit_to_mm(gt)

        # MANO decode (dataset-specific)
        if dataset_name == "freihand":
            if mano_decoder == "freihand_legacy":
                # IMPORTANT: use the same legacy MANO assets as FreiHAND toolbox.
                root_override = args.freihand_mano_root or freihand_mano_root_cfg
                pred_m = _decode_mano_freihand_manopth(
                    batch["mano_pose"],
                    batch["mano_shape"],
                    device=device,
                    mano_root=Path(root_override) if root_override else None,
                )
            else:
                go_R, hp_R, betas = _mano_from_batch(batch, device=device)
                mano_out = mano_layer({"global_orient": go_R, "hand_pose": hp_R, "betas": betas}, pose2rot=False)
                verts_m = mano_out.vertices.to(device=device, dtype=torch.float32)
                pred_m = _freihand_kp21_from_mano(verts_m, mano_layer)  # meters
        elif dataset_name == "ho3d":
            pred_m = _decode_mano_ho3d(
                batch["mano_pose"].to(device=device, dtype=torch.float32),
                batch["mano_shape"].to(device=device, dtype=torch.float32),
                batch.get("mano_trans", None).to(device=device, dtype=torch.float32) if batch.get("mano_trans", None) is not None else None,
                device=device,
            )
        elif dataset_name in ("dexycb", "dex-ycb"):
            pred_m = _decode_mano_dexycb(
                batch["mano_pose"].to(device=device, dtype=torch.float32),
                batch["mano_shape"].to(device=device, dtype=torch.float32),
                batch.get("mano_trans", None).to(device=device, dtype=torch.float32) if batch.get("mano_trans", None) is not None else None,
                batch.get("hand_type", None),
                device=device,
                mano_root=Path(cfg["paths"]["mano_dir"]) if "paths" in cfg and "mano_dir" in cfg["paths"] else None,
            )
        else:
            raise ValueError(f"Unsupported dataset for MANO decode: {dataset_name}")

        pred_mm = pred_m * 1000.0

        # Root-center both
        ri = int(root_index)
        pred_c = pred_mm - pred_mm[:, [ri]]
        gt_c = gt_mm - gt_mm[:, [ri]]

        # Metrics
        # MPJPE per sample (mm)
        mpjpe_b = (pred_c - gt_c).norm(dim=-1).mean(dim=-1)  # (B,)
        # compute_pa_mpjpe expects mm inputs and returns mm already.
        pamp_b = torch.from_numpy(compute_pa_mpjpe(pred_c, gt_c)).to(device=device)  # (B,)

        # Reduce to the requested number of samples
        remain = n - count
        take = min(int(mpjpe_b.shape[0]), remain)
        mpjpe_sum += float(mpjpe_b[:take].sum().item())
        pamp_sum += float(pamp_b[:take].sum().item())
        count += take

    if count <= 0:
        raise RuntimeError("No samples were evaluated.")

    print(f"[dataset={dataset_name} split={args.split} loader={args.loader} root_index={root_index}]")
    print(f"  N={count}")
    print(f"  MPJPE(root-centered)   = {mpjpe_sum / count:.3f} mm")
    print(f"  PA-MPJPE(root-centered)= {pamp_sum / count:.3f} mm")


if __name__ == "__main__":
    main()
    """
    
    """

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dex_ycb_dataset import DexYCBDataset
from data.ho3d_dataset import HO3DDataset

DEFAULT_DEXYCB_ROOT = Path('/root/code/vepfs/dataset/dex-ycb')
DEFAULT_HO3D_ROOT = Path('/root/code/vepfs/dataset/HO3D_v3')
DEFAULT_OUTPUT_ROOT = Path('/root/code/vepfs/GPGFormer/outputs')
HO3D_DEPTH_SCALE = 0.00012498664727900177


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export raw RGB and depth images for a DexYCB or HO3D sample by dataset index.'
    )
    parser.add_argument('--dataset', choices=['dexycb', 'ho3d'], required=True, help='Dataset name.')
    parser.add_argument('--index', type=int, required=True, help='Dataset sample index.')
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Dataset split. Defaults: DexYCB=test, HO3D=evaluation.',
    )
    parser.add_argument(
        '--setup',
        type=str,
        default='s0',
        help='DexYCB setup name. Only used when --dataset dexycb. Default: s0.',
    )
    parser.add_argument(
        '--dexycb-root',
        type=Path,
        default=DEFAULT_DEXYCB_ROOT,
        help=f'DexYCB root directory. Default: {DEFAULT_DEXYCB_ROOT}',
    )
    parser.add_argument(
        '--ho3d-root',
        type=Path,
        default=DEFAULT_HO3D_ROOT,
        help=f'HO3D root directory. Default: {DEFAULT_HO3D_ROOT}',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f'Root directory for saved outputs. Default: {DEFAULT_OUTPUT_ROOT}',
    )
    return parser.parse_args()


def _resolve_dexycb_depth_path(rgb_path: Path) -> Path:
    name = rgb_path.name
    if not name.startswith('color_'):
        raise ValueError(f'Unexpected DexYCB RGB filename format: {rgb_path}')
    frame_token = rgb_path.stem[len('color_') :]
    depth_name = f'aligned_depth_to_color_{frame_token}.png'
    depth_path = rgb_path.with_name(depth_name)
    if not depth_path.exists():
        raise FileNotFoundError(f'DexYCB depth image not found: {depth_path}')
    return depth_path


def _resolve_ho3d_depth_path(rgb_path: Path) -> Path:
    if rgb_path.parent.name != 'rgb':
        raise ValueError(f'Unexpected HO3D RGB parent directory: {rgb_path}')
    depth_path = rgb_path.parent.parent / 'depth' / f'{rgb_path.stem}.png'
    if not depth_path.exists():
        raise FileNotFoundError(f'HO3D depth image not found: {depth_path}')
    return depth_path


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth_scalar = depth[..., 0]
    else:
        depth_scalar = depth

    finite = np.isfinite(depth_scalar)
    positive = finite & (depth_scalar > 0)
    if not np.any(positive):
        return np.zeros((depth_scalar.shape[0], depth_scalar.shape[1], 3), dtype=np.uint8)

    values = depth_scalar[positive].astype(np.float32)
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth_scalar.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    norm_u8 = np.round(norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    vis[~positive] = 0
    return vis


def _to_uint8_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x_min) / (x_max - x_min)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _decode_depth_for_vis(dataset_name: str, depth: np.ndarray) -> np.ndarray:
    if dataset_name == 'ho3d':
        if depth.ndim != 3 or depth.shape[2] < 3:
            raise ValueError(f'Unexpected HO3D depth format: shape={depth.shape}, dtype={depth.dtype}')
        decoded = depth[:, :, 2].astype(np.float32) + depth[:, :, 1].astype(np.float32) * 256.0
        decoded *= HO3D_DEPTH_SCALE
        return decoded

    if depth.ndim == 3:
        return depth[:, :, 0].astype(np.float32)
    return depth.astype(np.float32)


def _copy_or_write_rgb(rgb_path: Path, out_path: Path) -> None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if not isinstance(rgb, np.ndarray):
        raise IOError(f'Failed to read RGB image: {rgb_path}')
    if not cv2.imwrite(str(out_path), rgb):
        raise IOError(f'Failed to save RGB image to: {out_path}')


def _copy_or_write_depth(
    dataset_name: str,
    depth_path: Path,
    raw_out_path: Path,
    decoded_vis_out_path: Path,
) -> dict[str, Any]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise IOError(f'Failed to read depth image: {depth_path}')

    if depth_path.suffix.lower() == '.png':
        shutil.copy2(depth_path, raw_out_path)
    else:
        if not cv2.imwrite(str(raw_out_path), depth):
            raise IOError(f'Failed to save raw depth image to: {raw_out_path}')

    decoded_depth = _decode_depth_for_vis(dataset_name, depth)
    decoded_vis = _to_uint8_minmax(decoded_depth)
    if not cv2.imwrite(str(decoded_vis_out_path), decoded_vis):
        raise IOError(f'Failed to save decoded depth visualization to: {decoded_vis_out_path}')

    return {
        'dtype': str(depth.dtype),
        'shape': list(depth.shape),
        'min': int(np.min(depth)) if depth.size else None,
        'max': int(np.max(depth)) if depth.size else None,
        'decoded_min': float(np.min(decoded_depth)) if decoded_depth.size else None,
        'decoded_max': float(np.max(decoded_depth)) if decoded_depth.size else None,
    }


def _build_dataset(args: argparse.Namespace) -> tuple[Any, str]:
    if args.dataset == 'dexycb':
        split = (args.split or 'test').lower()
        dataset = DexYCBDataset(
            setup=args.setup,
            split=split,
            root_dir=str(args.dexycb_root),
            train=(split == 'train'),
            align_wilor_aug=False,
            bbox_source='gt',
        )
        return dataset, split

    split = (args.split or 'evaluation').lower()
    dataset = HO3DDataset(
        data_split=split,
        root_dir=str(args.ho3d_root),
        train=(split == 'train'),
        align_wilor_aug=False,
        bbox_source='gt',
    )
    return dataset, split


def _resolve_paths(dataset_name: str, sample: dict[str, Any]) -> tuple[Path, Path]:
    rgb_path = Path(sample['img_path']).resolve()
    if not rgb_path.exists():
        raise FileNotFoundError(f'RGB image not found: {rgb_path}')
    if dataset_name == 'dexycb':
        depth_path = _resolve_dexycb_depth_path(rgb_path)
    else:
        depth_path = _resolve_ho3d_depth_path(rgb_path)
    return rgb_path, depth_path


def main() -> None:
    args = _parse_args()
    dataset, split = _build_dataset(args)

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f'Index {args.index} out of range for {args.dataset} {split}: [0, {len(dataset) - 1}]')

    sample = dataset.datalist[args.index]
    rgb_path, depth_path = _resolve_paths(args.dataset, sample)

    export_dir = args.output_root / 'rgb_depth_by_index' / f'{args.dataset}_{split}_idx_{args.index:06d}'
    export_dir.mkdir(parents=True, exist_ok=True)

    rgb_out_path = export_dir / 'rgb.png'
    depth_raw_out_path = export_dir / 'depth_raw.png'
    depth_decoded_vis_out_path = export_dir / 'depth_decoded_vis.png'
    meta_out_path = export_dir / 'meta.json'

    _copy_or_write_rgb(rgb_path, rgb_out_path)
    depth_stats = _copy_or_write_depth(args.dataset, depth_path, depth_raw_out_path, depth_decoded_vis_out_path)

    meta = {
        'dataset': args.dataset,
        'split': split,
        'setup': args.setup if args.dataset == 'dexycb' else None,
        'index': args.index,
        'num_samples': len(dataset),
        'rgb_path': str(rgb_path),
        'depth_path': str(depth_path),
        'saved_rgb_path': str(rgb_out_path),
        'saved_depth_raw_path': str(depth_raw_out_path),
        'saved_depth_decoded_vis_path': str(depth_decoded_vis_out_path),
        'depth_stats': depth_stats,
    }
    if 'seq_name' in sample:
        meta['seq_name'] = sample['seq_name']
    if 'frame_id' in sample:
        meta['frame_id'] = sample['frame_id']
    if 'hand_type' in sample:
        meta['hand_type'] = sample['hand_type']

    meta_out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集标签检查脚本
用于读取、解析和打印 DexYCB 和 HO3D 数据集的标签文件
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def print_dict_structure(data: Dict[str, Any], indent: int = 0, max_items: int = 3):
    """
    递归打印字典结构

    Args:
        data: 要打印的字典
        indent: 缩进级别
        max_items: 每个列表最多显示的项目数
    """
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}: dict (keys={list(value.keys())[:5]})")
            if indent < 2:  # 限制递归深度
                print_dict_structure(value, indent + 1, max_items)
        elif isinstance(value, list):
            print(f"{prefix}{key}: list (len={len(value)})")
            if len(value) > 0 and indent < 2:
                print(f"{prefix}  [0]: {type(value[0]).__name__}")
                if isinstance(value[0], dict):
                    print(f"{prefix}  Sample keys: {list(value[0].keys())[:5]}")
        elif isinstance(value, np.ndarray):
            print(f"{prefix}{key}: np.ndarray (shape={value.shape}, dtype={value.dtype})")
        else:
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"{prefix}{key}: {type(value).__name__} = {val_str}")


def inspect_dexycb_labels(root_dir: str, setup: str = "s0", split: str = "train",
                          num_samples: int = 3):
    """
    检查 DexYCB 数据集的标签文件

    Args:
        root_dir: DexYCB 数据集根目录
        setup: 数据集设置 (s0, s1, s2, s3)
        split: 数据集划分 (train, test)
        num_samples: 显示的样本数量
    """
    print("\n" + "="*80)
    print(f"DexYCB Dataset - Setup: {setup}, Split: {split}")
    print("="*80)

    # 查找annotations目录
    root_path = Path(root_dir)
    if (root_path / "dex-ycb" / "annotations").exists():
        annot_dir = root_path / "dex-ycb" / "annotations"
    elif (root_path / "annotations").exists():
        annot_dir = root_path / "annotations"
    else:
        print(f"❌ Error: Cannot find annotations directory in {root_dir}")
        return

    # 读取JSON文件
    json_file = annot_dir / f"DEX_YCB_{setup}_{split}_data.json"
    if not json_file.exists():
        print(f"❌ Error: JSON file not found: {json_file}")
        return

    print(f"\n📁 Reading: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    # 打印整体结构
    print(f"\n📊 Dataset Structure:")
    print(f"  Top-level keys: {list(data.keys())}")

    if "images" in data:
        print(f"\n  Images: {len(data['images'])} entries")
        if len(data['images']) > 0:
            print(f"    Sample image keys: {list(data['images'][0].keys())}")

    if "annotations" in data:
        print(f"\n  Annotations: {len(data['annotations'])} entries")
        if len(data['annotations']) > 0:
            print(f"    Sample annotation keys: {list(data['annotations'][0].keys())}")

    # 打印样本
    print(f"\n📝 Sample Annotations (showing {num_samples}):")
    annotations = data.get("annotations", [])
    images_by_id = {img["id"]: img for img in data.get("images", [])}

    for i, ann in enumerate(annotations[:num_samples]):
        print(f"\n  --- Sample {i+1} ---")
        print(f"  image_id: {ann.get('image_id')}")

        # 获取对应的图像信息
        img_id = ann.get('image_id')
        if img_id in images_by_id:
            img = images_by_id[img_id]
            print(f"  image_path: {img.get('color_file_name', 'N/A')}")
            print(f"  image_size: ({img.get('height')}, {img.get('width')})")

        # 打印关键标注信息
        if 'joints_coord_cam' in ann:
            joints = np.array(ann['joints_coord_cam'])
            print(f"  joints_coord_cam: shape={joints.shape}, range=[{joints.min():.2f}, {joints.max():.2f}]")

        if 'cam_param' in ann:
            cam = ann['cam_param']
            print(f"  cam_param: focal={cam.get('focal')}, princpt={cam.get('princpt')}")

        if 'mano_param' in ann:
            mano = ann['mano_param']
            print(f"  mano_param keys: {list(mano.keys())}")
            if 'pose' in mano:
                pose = np.array(mano['pose'])
                print(f"    pose: shape={pose.shape}")
            if 'shape' in mano:
                shape = np.array(mano['shape'])
                print(f"    shape: shape={shape.shape}")

        print(f"  hand_type: {ann.get('hand_type', 'N/A')}")


def inspect_ho3d_labels(root_dir: str, split: str = "train", num_samples: int = 3):
    """
    检查 HO3D 数据集的标签文件

    Args:
        root_dir: HO3D 数据集根目录
        split: 数据集划分 (train, evaluation)
        num_samples: 显示的样本数量
    """
    print("\n" + "="*80)
    print(f"HO3D Dataset - Split: {split}")
    print("="*80)

    root_path = Path(root_dir)

    # 查找HO3D_v3目录
    if (root_path / "HO3D_v3").exists():
        ho3d_root = root_path / "HO3D_v3"
    elif root_path.name == "HO3D_v3":
        ho3d_root = root_path
    else:
        print(f"❌ Error: Cannot find HO3D_v3 directory in {root_dir}")
        return

    # 读取split文件
    split_file = ho3d_root / f"{split}.txt"
    if not split_file.exists():
        print(f"❌ Error: Split file not found: {split_file}")
        return

    print(f"\n📁 Reading split file: {split_file}")

    with open(split_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"  Total entries: {len(lines)}")

    # 打印样本
    print(f"\n📝 Sample Entries (showing {num_samples}):")

    for i, line in enumerate(lines[:num_samples]):
        print(f"\n  --- Sample {i+1} ---")
        print(f"  Entry: {line}")

        # 解析路径
        parts = line.split("/")
        if len(parts) >= 2:
            sequence = parts[0]
            frame = parts[1] if len(parts) > 1 else "unknown"
            print(f"  Sequence: {sequence}")
            print(f"  Frame: {frame}")

            # 尝试读取对应的标注文件
            meta_file = ho3d_root / split / sequence / "meta" / f"{frame}.pkl"
            if meta_file.exists():
                print(f"  📄 Meta file: {meta_file}")

                with open(meta_file, "rb") as f:
                    meta = pickle.load(f)

                print(f"  Meta keys: {list(meta.keys())}")

                # 打印关键信息
                if 'camMat' in meta:
                    cam_mat = meta['camMat']
                    print(f"    camMat: shape={cam_mat.shape if hasattr(cam_mat, 'shape') else 'N/A'}")

                if 'handJoints3D' in meta:
                    joints = meta['handJoints3D']
                    print(f"    handJoints3D: shape={joints.shape if hasattr(joints, 'shape') else 'N/A'}")

                if 'handPose' in meta:
                    pose = meta['handPose']
                    print(f"    handPose: shape={pose.shape if hasattr(pose, 'shape') else 'N/A'}")

                if 'handBeta' in meta:
                    beta = meta['handBeta']
                    print(f"    handBeta: shape={beta.shape if hasattr(beta, 'shape') else 'N/A'}")

                if 'handTrans' in meta:
                    trans = meta['handTrans']
                    print(f"    handTrans: shape={trans.shape if hasattr(trans, 'shape') else 'N/A'}")
            else:
                print(f"  ⚠️  Meta file not found: {meta_file}")

            # 检查图像文件
            rgb_file = ho3d_root / split / sequence / "rgb" / f"{frame}.jpg"
            if rgb_file.exists():
                print(f"  🖼️  RGB image: {rgb_file}")
            else:
                print(f"  ⚠️  RGB image not found: {rgb_file}")


def main():
    parser = argparse.ArgumentParser(
        description="检查 DexYCB 和 HO3D 数据集的标签文件"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dexycb", "ho3d", "both"],
        default="both",
        help="要检查的数据集"
    )
    parser.add_argument(
        "--dexycb-root",
        type=str,
        default="/root/code/vepfs/dataset/dex-ycb",
        help="DexYCB 数据集根目录"
    )
    parser.add_argument(
        "--ho3d-root",
        type=str,
        default="/root/code/vepfs/dataset/HO3D_v3",
        help="HO3D 数据集根目录"
    )
    parser.add_argument(
        "--dexycb-setup",
        type=str,
        default="s0",
        help="DexYCB 数据集设置 (s0, s1, s2, s3)"
    )
    parser.add_argument(
        "--dexycb-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="DexYCB 数据集划分"
    )
    parser.add_argument(
        "--ho3d-split",
        type=str,
        default="train",
        choices=["train", "evaluation"],
        help="HO3D 数据集划分"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="显示的样本数量"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("数据集标签检查工具")
    print("="*80)

    if args.dataset in ["dexycb", "both"]:
        inspect_dexycb_labels(
            args.dexycb_root,
            args.dexycb_setup,
            args.dexycb_split,
            args.num_samples
        )

    if args.dataset in ["ho3d", "both"]:
        inspect_ho3d_labels(
            args.ho3d_root,
            args.ho3d_split,
            args.num_samples
        )

    print("\n" + "="*80)
    print("检查完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

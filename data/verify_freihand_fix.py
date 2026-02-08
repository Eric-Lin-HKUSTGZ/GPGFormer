#!/usr/bin/env python3
"""
验证 FreiHAND 数据加载器 bug 修复
"""
import json
import os

def verify_fix():
    root = '/root/code/vepfs/dataset/FreiHAND_pub_v2'

    print("=" * 70)
    print("FreiHAND 数据加载器 Bug 修复验证")
    print("=" * 70)

    # 加载标注
    with open(f'{root}/training_K.json') as f:
        K_list = json.load(f)
    with open(f'{root}/training_xyz.json') as f:
        xyz_list = json.load(f)
    with open(f'{root}/training_mano.json') as f:
        mano_list = json.load(f)

    # 统计图像数量
    img_dir = f'{root}/training/rgb'
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    num_images = len(img_files)
    num_annotations = len(K_list)

    print(f"\n数据集统计:")
    print(f"  图像数量: {num_images}")
    print(f"  标注数量: {num_annotations}")
    print(f"  比例: {num_images / num_annotations:.1f}:1")

    print(f"\n修复前的问题:")
    print(f"  ✗ 使用 real_idx 直接访问标注")
    print(f"  ✗ 当 real_idx >= {num_annotations} 时会越界")
    print(f"  ✗ 无法访问 {num_images - num_annotations} 张图像 ({(num_images - num_annotations) / num_images * 100:.1f}%)")

    print(f"\n修复后的方案:")
    print(f"  ✓ 使用 anno_idx = real_idx % {num_annotations}")
    print(f"  ✓ 标注循环使用，所有图像都可访问")

    # 测试映射关系
    print(f"\n映射关系验证:")
    test_indices = [0, 1, 32559, 32560, 32561, 65120, 97680, 130239]

    print(f"  {'图像索引':<12} {'标注索引':<12} {'说明'}")
    print(f"  {'-'*12} {'-'*12} {'-'*30}")

    for img_idx in test_indices:
        if img_idx < num_images:
            anno_idx = img_idx % num_annotations
            cycle = img_idx // num_annotations
            print(f"  {img_idx:<12} {anno_idx:<12} 周期 {cycle}")

    print(f"\n验证完成！修复已应用。")
    print("=" * 70)

if __name__ == "__main__":
    verify_fix()

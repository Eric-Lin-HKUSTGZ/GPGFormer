#!/usr/bin/env python3
"""
Test script to verify FreiHAND dataset loading logic with use_trainval_split=False
"""
import os
import json

def test_dataset_logic():
    """Test the dataset loading logic without actually loading the dataset"""

    # Simulate the configuration
    configs = [
        {
            "name": "Scenario 1: use_trainval_split=True (default)",
            "use_trainval_split": True,
            "train": False,
            "eval_root": "/root/code/vepfs/dataset/FreiHAND_pub_v2_eval",
            "expected_dir": "/root/code/vepfs/dataset/FreiHAND_pub_v2/training/rgb",
            "expected_prefix": "training"
        },
        {
            "name": "Scenario 2: use_trainval_split=False, train=False, with eval_root",
            "use_trainval_split": False,
            "train": False,
            "eval_root": "/root/code/vepfs/dataset/FreiHAND_pub_v2_eval",
            "expected_dir": "/root/code/vepfs/dataset/FreiHAND_pub_v2_eval/evaluation/rgb",
            "expected_prefix": "evaluation"
        },
        {
            "name": "Scenario 3: use_trainval_split=False, train=True, with eval_root",
            "use_trainval_split": False,
            "train": True,
            "eval_root": "/root/code/vepfs/dataset/FreiHAND_pub_v2_eval",
            "expected_dir": "/root/code/vepfs/dataset/FreiHAND_pub_v2/training/rgb",
            "expected_prefix": "training"
        },
        {
            "name": "Scenario 4: use_trainval_split=False, train=False, no eval_root",
            "use_trainval_split": False,
            "train": False,
            "eval_root": None,
            "expected_dir": "/root/code/vepfs/dataset/FreiHAND_pub_v2/training/rgb",
            "expected_prefix": "training"
        }
    ]

    print("=" * 80)
    print("Testing FreiHAND Dataset Loading Logic")
    print("=" * 80)

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 80)

        # Simulate the logic from _load_annotations()
        use_trainval_split = config["use_trainval_split"]
        train = config["train"]
        eval_root = config["eval_root"]
        root_dir = "/root/code/vepfs/dataset/FreiHAND_pub_v2"

        if not use_trainval_split and not train and eval_root is not None:
            # Load from evaluation set
            base_root = eval_root
            prefix = 'evaluation'
            img_dir = os.path.join(base_root, 'evaluation', 'rgb')
        else:
            # Load from training set (default behavior)
            base_root = root_dir
            prefix = 'training'
            img_dir = os.path.join(base_root, 'training', 'rgb')

        print(f"  use_trainval_split: {use_trainval_split}")
        print(f"  train: {train}")
        print(f"  eval_root: {eval_root}")
        print(f"  → base_root: {base_root}")
        print(f"  → prefix: {prefix}")
        print(f"  → img_dir: {img_dir}")

        # Check if the directory exists
        exists = os.path.isdir(img_dir)
        print(f"  → Directory exists: {exists}")

        if exists:
            num_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            print(f"  → Number of images: {num_images}")

        # Verify against expected
        expected_match = (img_dir == config["expected_dir"] and prefix == config["expected_prefix"])
        status = "✓ PASS" if expected_match else "✗ FAIL"
        print(f"  → Status: {status}")

        if not expected_match:
            print(f"  → Expected img_dir: {config['expected_dir']}")
            print(f"  → Expected prefix: {config['expected_prefix']}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_dataset_logic()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check HO3D_v3 `handJoints3D` joint order in the cached annotation file.

Why this exists:
HO3D dumps in the wild sometimes permute the 5 fingertip joints (indices 16..20).
This repo's HO3D dataloader expects the SMPL-X MANO "internal" joint order:
  - 16 non-tip joints: [wrist, index(3), middle(3), pinky(3), ring(3), thumb(3)]
  - 5 tips at indices 16..20 in order: [thumb, index, middle, ring, pinky]

We infer which fingertip is which by nearest DIP joint in 3D (meters), aggregated over many samples.
"""

from __future__ import annotations

import argparse
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


FINGER_DIP_INDEX = {
    "index": 3,
    "middle": 6,
    "pinky": 9,
    "ring": 12,
    "thumb": 15,
}

TIP_INDICES = (16, 17, 18, 19, 20)
EXPECTED_TIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def _infer_tip_to_finger(joints21: np.ndarray) -> Dict[int, str]:
    j = np.asarray(joints21, dtype=np.float32).reshape(21, 3)
    dip_names = list(FINGER_DIP_INDEX.keys())
    dip_indices = [FINGER_DIP_INDEX[n] for n in dip_names]
    tips = j[list(TIP_INDICES), :]  # (5,3)
    dips = j[dip_indices, :]  # (5,3)
    dist = np.linalg.norm(tips[:, None, :] - dips[None, :, :], axis=-1)  # (5,5)
    nn = dist.argmin(axis=1).tolist()
    return {t: dip_names[i] for t, i in zip(TIP_INDICES, nn)}


def _majority_mapping(samples: List[Dict]) -> Tuple[Dict[int, str], Dict[str, int]]:
    counts: Dict[int, Dict[str, int]] = {t: defaultdict(int) for t in TIP_INDICES}
    for d in samples:
        m = _infer_tip_to_finger(d["joints_coord_cam"])
        for t, finger in m.items():
            counts[t][finger] += 1

    tip_to_finger: Dict[int, str] = {}
    for t in TIP_INDICES:
        if not counts[t]:
            continue
        tip_to_finger[t] = max(counts[t].items(), key=lambda kv: kv[1])[0]

    finger_to_tip: Dict[str, int] = {}
    for finger in EXPECTED_TIP_ORDER:
        best_t, best_c = None, -1
        for t in TIP_INDICES:
            c = counts[t].get(finger, 0)
            if c > best_c:
                best_t, best_c = t, c
        if best_t is not None:
            finger_to_tip[finger] = int(best_t)

    return tip_to_finger, finger_to_tip


def _recommended_meta_map(finger_to_tip: Dict[str, int]) -> List[int] | None:
    if any(f not in finger_to_tip for f in EXPECTED_TIP_ORDER):
        return None
    meta_map = list(range(21))
    desired = [finger_to_tip[f] for f in EXPECTED_TIP_ORDER]  # original indices in the desired order
    meta_map[16:21] = desired
    return meta_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=str,
        default="/root/code/vepfs/dataset/HO3D_v3/.cache_ho3d_train_da7d2677113f.pkl",
        help="Path to cached HO3D train/val annotations (.pkl).",
    )
    parser.add_argument("--num-samples", type=int, default=2000, help="How many random samples to use.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        datalist = pickle.load(f)

    if not isinstance(datalist, list) or not datalist:
        raise RuntimeError(f"Unexpected cache content: {type(datalist).__name__}")

    rng = random.Random(int(args.seed))
    k = int(min(max(args.num_samples, 1), len(datalist)))
    samples = [datalist[rng.randrange(len(datalist))] for _ in range(k)]

    tip_to_finger, finger_to_tip = _majority_mapping(samples)
    inferred_order = tuple(tip_to_finger.get(t, "unknown") for t in TIP_INDICES)

    print(f"[info] cache={args.cache} total={len(datalist)} sampled={k}")
    print(f"[info] inferred tip order @ indices 16..20: {inferred_order}")
    print(f"[info] expected tip order            : {EXPECTED_TIP_ORDER}")

    rec = _recommended_meta_map(finger_to_tip)
    if rec is None:
        print("[warn] Could not infer a consistent fingertip mapping; check dataset dump.")
        return
    print(f"[info] recommended HO3D_META_JOINT_MAP: {rec}")

    if inferred_order == EXPECTED_TIP_ORDER:
        print("[ok] Fingertip order matches expected. HO3D_META_JOINT_MAP should be identity.")
    else:
        print("[warn] Fingertip order differs; consider using the recommended HO3D_META_JOINT_MAP.")


if __name__ == "__main__":
    main()


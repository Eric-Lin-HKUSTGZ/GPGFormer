#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate HO3D train/val split txt files from a COCO-style JSON file.

Output format (one line per sample):
    SEQ/FRAME

Examples:
    python scripts/generate_ho3d_split_from_json.py \
      --json /c20250503/lwq/dataset/handos_data/HO3Dv3/HO3D-train-normalized.json \
      --out-train /root/code/vepfs/dataset/HO3D_v3/train_json_split.txt \
      --out-val /root/code/vepfs/dataset/HO3D_v3/val_json_split.txt \
      --ratio 0.9 \
      --seed 42 \
      --split-by sequence
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def iter_json_array_objects(json_path: Path, key: str, chunk_size: int = 2 * 1024 * 1024) -> Iterable[Dict]:
    token = f"\"{key}\""
    decoder = json.JSONDecoder()

    with json_path.open("r", encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise KeyError(f"Key '{key}' not found in JSON: {json_path}")
            buf += chunk
            i = buf.find(token)
            if i < 0:
                buf = buf[-(len(token) + 64):]
                continue
            j = buf.find("[", i + len(token))
            if j < 0:
                buf = buf[i:]
                continue
            buf = buf[j + 1 :]
            break

        done = False
        while not done:
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    done = True
                    pos += 1
                    break
                try:
                    obj, idx = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = idx

            buf = buf[pos:]
            if done:
                break

            chunk = f.read(chunk_size)
            if not chunk:
                if buf.strip():
                    raise ValueError(f"Unexpected EOF while parsing key '{key}' in JSON: {json_path}")
                break
            buf += chunk


def parse_seq_frame(file_name: str, expected_split: str = "train") -> Tuple[str, str] | None:
    s = str(file_name).replace("\\", "/").strip("/")
    parts = [p for p in s.split("/") if p]
    if not parts:
        return None

    if expected_split in parts:
        i = parts.index(expected_split)
        if i + 2 < len(parts):
            seq = parts[i + 1]
            if parts[i + 2].lower() == "rgb" and i + 3 < len(parts):
                frame = Path(parts[i + 3]).stem
            else:
                frame = Path(parts[i + 2]).stem
            if seq and frame:
                return seq, frame

    if len(parts) >= 3 and parts[-2].lower() == "rgb":
        return parts[-3], Path(parts[-1]).stem
    if len(parts) >= 2:
        return parts[-2], Path(parts[-1]).stem
    return None


def split_records(
    records: Sequence[Tuple[int, str, str]],
    ratio: float,
    seed: int,
    split_by: str,
) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    ratio = max(0.0, min(1.0, float(ratio)))
    if len(records) == 0:
        return [], []
    if ratio <= 0.0:
        return [], list(records)
    if ratio >= 1.0:
        return list(records), []

    rng = random.Random(int(seed))
    split_by = str(split_by).lower()
    if split_by not in ("sequence", "frame"):
        split_by = "sequence"

    if split_by == "frame":
        idxs = list(range(len(records)))
        rng.shuffle(idxs)
        cut = int(len(idxs) * ratio)
        train_idx = set(idxs[:cut])
        train = [records[i] for i in range(len(records)) if i in train_idx]
        val = [records[i] for i in range(len(records)) if i not in train_idx]
        return train, val

    seq_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for i, (_, seq, _) in enumerate(records):
        seq_to_idxs[seq].append(i)

    seqs = list(seq_to_idxs.keys())
    rng.shuffle(seqs)
    target = int(len(records) * ratio)

    train_idx = set()
    count = 0
    for seq in seqs:
        idx_bucket = seq_to_idxs[seq]
        if count < target:
            train_idx.update(idx_bucket)
            count += len(idx_bucket)

    if len(train_idx) == 0:
        train_idx.add(0)
    if len(train_idx) == len(records) and len(records) > 1:
        train_idx.remove(max(train_idx))

    train = [records[i] for i in range(len(records)) if i in train_idx]
    val = [records[i] for i in range(len(records)) if i not in train_idx]
    return train, val


def write_split(lines: Sequence[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HO3D train/val split txt from JSON images list.")
    parser.add_argument("--json", type=str, required=True, help="Path to HO3D JSON file (usually train JSON).")
    parser.add_argument("--out-train", type=str, required=True, help="Output train split txt path.")
    parser.add_argument("--out-val", type=str, required=True, help="Output val split txt path.")
    parser.add_argument("--ratio", type=float, default=0.9, help="Train ratio in [0,1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--split-by",
        type=str,
        default="sequence",
        choices=["sequence", "frame"],
        help="Split granularity.",
    )
    parser.add_argument(
        "--expected-split",
        type=str,
        default="train",
        help="Expected split token in file_name path (default: train).",
    )
    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    out_train = Path(args.out_train).expanduser().resolve()
    out_val = Path(args.out_val).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    records: List[Tuple[int, str, str]] = []
    dropped = 0
    seen = set()

    for i, img in enumerate(iter_json_array_objects(json_path, "images")):
        img_id = int(img["id"])
        file_name = str(img.get("file_name", ""))
        sf = parse_seq_frame(file_name, expected_split=str(args.expected_split))
        if sf is None:
            dropped += 1
            continue
        seq, frame = sf
        line = f"{seq}/{frame}"
        key = (img_id, line)
        if key in seen:
            continue
        seen.add(key)
        records.append((img_id, seq, frame))

        if i > 0 and i % 50000 == 0:
            print(f"[progress] parsed images: {i}, kept={len(records)}, dropped={dropped}", flush=True)

    if len(records) == 0:
        raise RuntimeError("No valid records parsed from JSON images.")

    train_records, val_records = split_records(
        records=records,
        ratio=float(args.ratio),
        seed=int(args.seed),
        split_by=str(args.split_by),
    )

    train_lines = [f"{seq}/{frame}" for _, seq, frame in train_records]
    val_lines = [f"{seq}/{frame}" for _, seq, frame in val_records]
    write_split(train_lines, out_train)
    write_split(val_lines, out_val)

    print(f"[done] json={json_path}")
    print(f"[done] total={len(records)} train={len(train_lines)} val={len(val_lines)} dropped={dropped}")
    print(f"[done] wrote train split: {out_train}")
    print(f"[done] wrote val split: {out_val}")


if __name__ == "__main__":
    main()


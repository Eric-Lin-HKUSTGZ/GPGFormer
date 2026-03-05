# GPGFormer (self-contained)

GPGFormer is a self-contained implementation of the **GPGFormer** framework (see `framework.md`) that vendors the minimal required code/weights from **WiLoR**, **MoGe2**, and **UniHandFormer/UTNet**.

## Quick start

### 1) Environment

- Python: 3.10+ (3.12 is supported with the compatibility patches already included in vendored MANO code)
- CUDA: recommended for training/eval

Install deps:

```bash
pip install -r requirements.txt
```

### 2) Weights & assets

Place/check the following (paths are referenced from YAML config files):

- **WiLoR**:
  - `weights/wilor/wilor_final.ckpt`
  - `weights/wilor/detector.pt`
- **MoGe2**:
  - `weights/moge2/model.pt`
- **MANO**:
  - `weights/mano/MANO_RIGHT.pkl`
  - `weights/mano/mano_mean_params.npz`

### 3) Dataset paths

Edit your dataset config under `configs/` (examples: `configs/config_ho3d.yaml`, `configs/config_freihand.yaml`, `configs/config_dexycb.yaml`) and set `paths.*` to your local paths.

### HO3D data loading modes

For HO3D, you can switch between two loaders in `configs/config_ho3d.yaml`:

1) Meta/pickle mode (default)

- Loader: `data/ho3d_dataset.py`
- Data source: `train.txt` / `evaluation.txt` + per-frame `meta/*.pkl`
- Config:

```yaml
dataset:
  ho3d_use_json_split: false
paths:
  ho3d_root: /path/to/HO3D_v3
```

2) JSON split mode

- Loader: `data/ho3d_json_dataset.py`
- Data source: COCO-style JSON files (`images` / `annotations`)
- Config:

```yaml
dataset:
  ho3d_use_json_split: true
  ho3d_json_kp3d_unit: auto     # auto | m | mm
  ho3d_json_kp3d_scale: 1.0
  ho3d_json_convert_xyz: false
paths:
  ho3d_root: /path/to/HO3D_v3
  ho3d_train_json: /c20250503/lwq/dataset/handos_data/HO3Dv3/HO3D-train-normalized.json
  ho3d_test_json: /c20250503/lwq/dataset/handos_data/HO3Dv3/HO3D-evaluation-meter.json
```

Notes:
- `ho3d_trainval_ratio / ho3d_trainval_seed / ho3d_trainval_split_by` still control train/val split in both modes.
- In JSON mode, large files are parsed in a streaming way and cached to `.cache_ho3d_json_*.pkl`.

### HO3D JSON -> split txt preprocessing

If you want to keep HO3D labels fully aligned with the original `meta/*.pkl` pipeline and only use JSON for split definition, you can preprocess JSON once and export split txt files.

Script:
- `scripts/generate_ho3d_split_from_json.py`

Example:

```bash
python scripts/generate_ho3d_split_from_json.py \
  --json /c20250503/lwq/dataset/handos_data/HO3Dv3/HO3D-train-normalized.json \
  --out-train /root/code/vepfs/dataset/HO3D_v3/train_json_split.txt \
  --out-val /root/code/vepfs/dataset/HO3D_v3/val_json_split.txt \
  --ratio 0.9 \
  --seed 42 \
  --split-by sequence
```

Output format:
- One sample per line: `SEQ/FRAME`
- Example: `MC1/0005`

Useful options:
- `--split-by sequence|frame`
- `--ratio` (train ratio)
- `--seed` (deterministic split)
- `--expected-split` (default: `train`)

Why `ho3d_json_convert_xyz: false`:
- The provided HO3Dv3 JSON files you referenced are already converted to this repo's coordinate convention (generator script applies `[..., 1:] *= -1` before export).
- Setting it to `true` would flip `(y, z)` a second time and introduce a coordinate mismatch.
- Only set `ho3d_json_convert_xyz: true` when your JSON is still in raw HO3D camera convention and has not been converted yet.

## Train / Eval (single GPU)

Train:

```bash
python train.py --config configs/config_ho3d.yaml
```

Eval:

```bash
python eval.py --config configs/config_ho3d.yaml --ckpt checkpoints/ho3d/gpgformer_epoch_1.pt
```

### Hand reconstruction visualization

Run visualization with a checkpoint (if checkpoint contains `cfg`, you can omit `--config`):

```bash
python visualize_reconstruction.py \
  --ckpt /path/to/gpgformer_best.pt \
  --config configs/config_freihand.yaml \
  --num-samples 24 \
  --batch-size 8
```

Dex-YCB visualization uses `dataset.name: dexycb` (or `dex-ycb`) in your config/ckpt and will visualize the **test** split.

For your provided checkpoint:

```bash
python visualize_reconstruction.py \
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/ho3d_20260224_neck_moge_400/ho3d/gpgformer_best.pt \
  --config configs/config_ho3d.yaml \
  --out-dir outputs/ho3d_20260224 \
  --overlay-mesh \
  --render-mesh \
  --num-samples 24 \
  --batch-size 8
```

If your target output path is on a quota-limited filesystem (e.g. `/root/code/vepfs/...`) and you see `OSError: [Errno 122] Disk quota exceeded`, either choose a different `--out-dir` or set `--fallback-out-dir`.

Note: if you explicitly set `--out-dir`, the script will not silently write outputs to a different location unless you also provide `--fallback-out-dir`.

```bash
python visualize_reconstruction.py \
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/freihand_20260213_sjta_v3/freihand/gpgformer_best.pt \
  --out-dir /root/code/vepfs/GPGFormer/outputs/freihand_20260213_sjta_v3_vis \
  --overlay-mesh \
  --render-mesh \
  --fallback-out-dir outputs/freihand_20260213_sjta_v3_vis_fallback \
  --num-samples 24 \
  --batch-size 8
```

Outputs:
- Visualization images: `sample_XXX_idx_YYY.png`
- (Optional) Predicted mesh: `sample_XXX_idx_YYY.obj` (enable with `--save-mesh-obj`)
- Metrics summary JSON: `summary.json` (contains per-sample 2D error, root-relative MPJPE and PA-MPJPE)

## Multi-GPU distributed training (torchrun)

GPGFormer supports multi-GPU distributed training via **PyTorch DDP** (same `torchrun` env-var workflow as UniHandFormer).

### Train with N GPUs

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/config_freihand.yaml
```

后台运行指令
```bash
nohup torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/config_freihand.yaml > /root/code/vepfs/GPGFormer/logs/training.log 2>&1 &
```

TMUX运行指令（可不用nohup）
```bash
stdbuf -oL -eL torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/config_freihand.yaml 2>&1 | tee -a /root/code/vepfs/GPGFormer/logs/training_20260209_kcr.log
```
或者
```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/config_freihand.yaml 2>&1 | tee -a /root/code/vepfs/GPGFormer/logs/training_20260209_kcr.log
```
如果想指定GPU卡号，在指令前加上CUDA_VISIBLE_DEVICES=4,5,6,7即可

验证数据加载器：
验证Dex-YCB数据集：
```bash
python -m data.dex_ycb_dataset --root-dir /root/code/vepfs/dataset/dex-ycb --setup s0 --split train --num-samples 3
```
验证HO3D_v3数据集：
```bash
python -m data.ho3d_dataset --root-dir /path/to/HO3D_v3 --split train --num-samples 3
```

### 检查数据集标签

使用数据集标签检查脚本来查看数据集的结构和标注信息：

检查两个数据集（默认）：
```bash
python scripts/inspect_dataset_labels.py
```

只检查 DexYCB 数据集：
```bash
python scripts/inspect_dataset_labels.py --dataset dexycb
```

只检查 HO3D 数据集：
```bash
python scripts/inspect_dataset_labels.py --dataset ho3d
```

自定义参数示例：
```bash
# 检查 DexYCB 的 s1 设置，test 划分，显示 5 个样本
python scripts/inspect_dataset_labels.py \
    --dataset dexycb \
    --dexycb-setup s1 \
    --dexycb-split test \
    --num-samples 5

# 检查 HO3D 的 evaluation 划分
python scripts/inspect_dataset_labels.py \
    --dataset ho3d \
    --ho3d-split evaluation \
    --num-samples 5
```

Notes:
- Only **rank 0** writes checkpoints and prints progress bars.
- `train.py` uses `DistributedSampler` and calls `sampler.set_epoch(epoch)` each epoch.
- Optional config knobs under `train`:
  - `seed` (default: 42)
  - `ddp_find_unused_parameters` (default: false)

### Distributed eval (optional)

```bash
torchrun --nproc_per_node=4 --master_port=29501 eval.py --config configs/config_ho3d.yaml --ckpt checkpoints/ho3d/gpgformer_epoch_1.pt
```

## Repository structure

- `gpgformer/`: main python package
  - `models/`: GPGFormer model + encoder/prior/tokenizers/MANO
  - `losses/`: vendored UniHandFormer/UTNet losses
  - `metrics/`: MPJPE / PA-MPJPE
  - `utils/`: helpers (including DDP utilities)
- `data/`: dataset loaders (Dex-YCB / HO3D / FreiHAND) + WiLoR-style augmentation
- `third_party/`: vendored minimal WiLoR / MoGe2 code
- `weights/`: all weights/assets

## Common pitfalls

- **Eval bbox source is dataset-dependent**:
  - FreiHAND/DexYCB commonly use detector bbox for eval.
  - HO3D config in this repo uses label-based crop (`dataset.bbox_source_eval=gt`) by default.
- **Detector + DataLoader workers**: eval uses `num_workers=0` because detector initialization is heavy and should be deterministic.
- **Geo token length / OOM**: geometric tokens are pooled to a fixed size in `gpgformer/models/tokenizers/geo_tokenizer.py` to keep transformer sequence length bounded.

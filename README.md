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

## Train / Eval (single GPU)

Train:

```bash
python train.py --config configs/config_ho3d.yaml
```

Eval:

```bash
python eval.py --config configs/config_ho3d.yaml --ckpt checkpoints/ho3d/gpgformer_epoch_1.pt
```

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

- **FreiHAND/HO3D eval bbox**: by default configs use a detector bbox for eval (`dataset.bbox_source_eval=detector`). Training uses GT bboxes.
- **Detector + DataLoader workers**: eval uses `num_workers=0` because detector initialization is heavy and should be deterministic.
- **Geo token length / OOM**: geometric tokens are pooled to a fixed size in `gpgformer/models/tokenizers/geo_tokenizer.py` to keep transformer sequence length bounded.











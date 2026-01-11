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
torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/config_ho3d.yaml
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











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

### Infer + full metrics (指定配置与权重路径)

运行 `infer_to_json.py`（现已支持 `freihand` / `ho3d` / `dexycb`，输出 MPJPE / PA-MPJPE / MPVPE / PA-MPVPE / F-score@5mm,15mm / AUC-J / AUC-V）：

说明：
- `infer_to_json.py` 已强制使用 **detector-only hand crop**（不会使用标签框裁剪；会忽略配置里的 `dataset.bbox_source_eval`）。
- 需要可用的 `paths.detector_ckpt`（如 `/root/code/vepfs/GPGFormer/weights/detector.pt`）和 `ultralytics` 依赖。
- 若某张图 detector 未检出有效手框，该样本会被自动跳过（不会回退到 GT bbox），并记录到输出 JSON 的 `skipped_samples` 中（含索引、图片路径、原因）。
- MPJPE/PA-MPJPE/MPVPE/PA-MPVPE 使用与 `train.py`/`eval.py` 一致的 **root-relative** 评测口径（`root_index` 默认 9）。
- FreiHAND 的 mesh 指标优先使用 `training_verts.json` 的真实顶点；只有在缺少 `vertices_gt` 时才会回退到 MANO 参数重建。
- 对 MANO 回退得到的 mesh 会做与关节 GT 的一致性校验（默认阈值 `metrics.mesh_fallback_kp_consistency_thr_mm=10`），不一致样本会从 mesh 指标剔除并记录到 JSON 的 `mesh_invalid_samples`。

```bash
# FreiHAND

python /root/code/hand_reconstruction/GPGFormer/infer_to_json.py   --config configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml --ckpt /root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt   --output /root/code/vepfs/GPGFormer/outputs/freihand_20260317/metrics_infer_to_json.json

python /root/code/hand_reconstruction/GPGFormer/infer_to_json.py   --config configs/ablations_v2/datasets/config_dexycb.yaml   --ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt   --output /root/code/vepfs/GPGFormer/outputs/dexycb_20260318/metrics_infer_to_json.json

# Dex-YCB
python /root/code/hand_reconstruction/GPGFormer/infer_to_json.py \
  --config configs/ablations_v2/datasets/config_dexycb.yaml \
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt \
  --output /root/code/vepfs/GPGFormer/outputs/dexycb_20260318/metrics_infer_to_json.json
```

### Hand reconstruction visualization

Cross-method comparison visualization (`visualization/compare_hand_mesh.py`) was verified in the Conda environment `/root/code/vepfs/miniconda3/envs/hamba`.

If you want to enable the `simplehand` branch in this script, make sure `hiera-transformer` is installed in that environment:

```bash
/root/code/vepfs/miniconda3/envs/hamba/bin/pip install hiera-transformer
```

The script can be run directly from the GPGFormer repo root:

```bash
cd /root/code/hand_reconstruction/GPGFormer
```

Tested commands:
使用hamba的conda环境
```bash
CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset freihand \
  --index 0 \
  --save-individual \
  --device cuda:0 \
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/freihand_fixed

CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset dexycb \
  --index 0 \
  --save-individual \
  --device cuda:0 \
  --gpgformer-ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt \
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/dexycb_fixed

CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset ho3d \
  --index 0 \
  --save-individual \
  --device cuda:0 \
  --gpgformer-ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/mixed_ho3d_20260320/ho3d/gpgformer_best.pt \
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/ho3d_fixed

# 通过index-range实现批量可视化
CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset freihand \
  --index-range 50 100 \
  --save-individual \
  --device cuda:0 \
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/freihand_range_50_100

# 只保留GPGFormer明显优于另外三个方法的样本，并把mesh-only图切到更容易看出差异的斜视角
CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset freihand \
  --index-range 0 500 \
  --show-gpgformer-better \
  --gpgformer-better-margin-mm 0.3 \
  --mesh-view-azim 45 \
  --mesh-view-elev -25 \
  --save-individual \
  --device cuda:0 \
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/freihand_gpgformer_better

  CUDA_VISIBLE_DEVICES=0 python visualization/compare_hand_mesh.py \
  --dataset dexycb \
  --index-range 4000 5000 \
  --show-gpgformer-better \
  --gpgformer-better-margin-mm 1.0 \
  --mesh-view-azim 45 \
  --mesh-view-elev -25 \
  --save-individual \
  --device cuda:0 \
  --gpgformer-ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt 
  --out-dir /root/code/vepfs/GPGFormer/outputs/compare_hand_mesh_tests/dexycb_gpgformer_better

```

Outputs from the tested runs were written to:

- `/tmp/compare_hand_mesh_tests/freihand_fixed`
- `/tmp/compare_hand_mesh_tests/dexycb_fixed`
- `/tmp/compare_hand_mesh_tests/ho3d_fixed`

If you omit `--out-dir`, the script writes to `outputs/compare_hand_meshes/<dataset>/` under the GPGFormer repo.

Each run writes:

- one subdirectory per sample index: `index_000050/`, `index_000051/`, ...
- one raw full image per sample: `index_000050/index_000050_image.png`
- one overview panel per sample: `index_000050/index_000050_overview.png`
- per-method overlay images such as `index_000050/index_000050_wilor_overlay.png`
- per-method mesh-only images such as `index_000050/index_000050_hamer_mesh.png`
- a root-level `summary.json` with per-sample status, recovered camera translations, and root-relative metrics (`joint_rr_mm`, `vertex_rr_mm`, `score_mm`)

Index selection options:

- `--index 7`: only visualize index 7
- `--indices 7 11 25`: visualize a manual index list
- `--index-range 50 100`: visualize the half-open interval `[50, 100)`, i.e. indices `50..99` for a total of 50 samples
- `--num-samples N`: default fallback when no explicit index selector is given

Quality-focused selection options:

- `--show-gpgformer-better`: only keep samples where GPGFormer is better than WiLoR / HaMeR / SimpleHand
- `--gpgformer-better-margin-mm 10`: require GPGFormer to beat all three competitors by at least 10 mm on both root-relative joint error and combined score
- Current selection metric uses `root_index=9`
- `joint_rr_mm`: mean joint error in mm after subtracting the root joint
- `vertex_rr_mm`: mean mesh vertex error in mm after subtracting the root joint
- `score_mm`: mean of available root-relative errors, currently averaging `joint_rr_mm` and `vertex_rr_mm`

Mesh-only view options:

- `--mesh-view-azim 45`
- `--mesh-view-elev -25`

These options only affect mesh-only renders, not overlay renders, and are useful when the original camera view hides shape differences.

### FreiHAND feature map comparison visualization

`/root/code/vepfs/GPGFormer/tools/visualize_freihand_feature_comparison.py` 用于在 FreiHAND 上对比单 RGB 与双模态 GPGFormer 的特征响应，帮助分析加入几何分支后，模型是否更聚焦于手部区域。

当前脚本会同时导出三类可视化：

- WiLoR backbone 中间层 self-attention rollout，默认第 `4 / 8 / 12` 层
- `img_feat` 空间激活图，默认对通道做 `L2 norm`
- `SJTA` 关节 query cross-attention，默认可视化 `thumb_tip` 和 `middle_mcp`

默认对比的两个 checkpoint：

- RGB-only:
  - `/root/code/vepfs/GPGFormer/checkpoints/freihand_20260304_rgb_only/freihand/gpgformer_best.pt`
- Multimodal:
  - `/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt`

运行环境：

- 推荐使用 `/root/code/vepfs/miniconda3/envs/moge`

示例命令：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_feature_comparison.py \
  --device cuda:0 \
  --split eval \
  --num-samples 8 \
  --layers 4 8 12 \
  --joint-names thumb_tip middle_mcp \
  --output-dir /root/code/vepfs/GPGFormer/outputs/freihand_feature_compare_run
```

如需额外把 GT hand bbox 画到叠加图上，可显式加上 `--show-bbox`：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_feature_comparison.py \
  --device cuda:0 \
  --split eval \
  --num-samples 8 \
  --layers 4 8 12 \
  --joint-names thumb_tip middle_mcp \
  --show-bbox \
  --output-dir /root/code/vepfs/GPGFormer/outputs/freihand_feature_compare_run_bbox
```

常用参数：

- `--sample-indices 0 5 12`：只跑指定样本
- `--num-samples 8`：当未指定样本索引时，抽取若干样本
- `--layers 4 8 12`：设置 rollout 层号
- `--joint-names thumb_tip middle_mcp`：设置 SJTA 可视化关节
- `--img-feat-reducer l2|var`：设置 `img_feat` 压缩方式
- `--cmap viridis`
- `--alpha 0.55`
- `--show-bbox`：可选地在导出的 overlay 和 overview 图中绘制白色 GT hand bbox，默认关闭

输出目录结构：

- 根目录下会写出 `summary.json`
- 每个样本一个子目录：`sample_<dataset_idx>_img_<real_idx>/`
- 每个样本目录下包含：
  - `input_crop.png`
  - `overview.png`
  - `metadata.json`
  - `rgb_only/`
  - `multimodal/`
- 模型子目录下会写出：
  - `img_feat_<reducer>.png`
  - `rollout_layer_04.png` / `rollout_layer_08.png` / `rollout_layer_12.png`
  - `sjta_thumb_tip.png`
  - `sjta_middle_mcp.png`

说明：

- 叠加图是画在模型输入 crop 上，不是回贴到原始整图；这样与 token/grid 特征是一一对应的。
- `summary.json` 与每个样本的 `metadata.json` 会记录一些简单的定量指标，例如热图在手部 GT bbox 内的质量占比、SJTA 峰值到 GT 关节的像素距离等。
- 白色 bbox、红色点、蓝绿色叉号以及 `inside-bbox` / `peak-to-gt` / `pred-to-gt` 的含义与优劣方向，现在只保留在脚本代码注释中说明，不再额外绘制在图面上。
- RGB-only checkpoint 加载时可能提示缺少 `encoder.sum_geo_gate`；这是因为该模型 `use_geo_prior=False`，该参数不参与实际前向，不影响可视化结果。

### Multi-dataset token analysis visualization

`/root/code/vepfs/GPGFormer/tools/visualize_freihand_token_analysis.py` 现在支持 `freihand`、`ho3d` 和 `dexycb` 三个数据集，用来直接分析 GPGFormer 在 token 流中的五个关键阶段。

当前脚本会保留以下五类原始 energy heatmap：

- `RGB patch tokens`：RGB 图像经过 `patch_embed` 后、进入融合前的 token 热图
- `Geometry tokens`：几何分支变成 token 后的热图
- `Fusion before backbone`：RGB token 与几何 token 融合后、进入 backbone block 0 之前的热图
- `Backbone last layer (multimodal)`：双模态条件下 backbone 最后一层输出热图
- `Backbone last layer (RGB-only)`：单 RGB 条件下 backbone 最后一层输出热图

同时还会额外导出一组更适合论文分析的 semantic / complementarity heatmap：

- `Cross-modal cosine distance`：看 Geometry token 与 RGB token 在哪里最不一致、最互补
- `RGB-hand similarity`：看 RGB patch token 与最终手部语义原型的相似度
- `Geometry-hand similarity`：看 Geometry token 与最终手部语义原型的相似度
- `Fusion-hand similarity`：看融合后 token 与最终手部语义原型的相似度
- `Fusion gain over best single`：看融合后 token 相比 `max(RGB, Geometry)` 的额外增益
- `Final multimodal-hand similarity` / `Final RGB-only-hand similarity`：看深层语义表征与手部原型的对齐程度
- `Final multimodal gain`：看双模态最终表征相比 RGB-only 的提升区域

其中第二类 `Geometry tokens` 可视化的是“真正送进 encoder 的几何 token”，也就是：

- `MoGe2 -> GeoSideAdapter(若启用) -> GeoTokenizer -> GeoSideTuning(若启用)` 之后的结果

运行环境：

- 推荐使用 `/root/code/vepfs/miniconda3/envs/moge`

FreiHAND 示例：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_token_analysis.py \
  --dataset freihand \
  --checkpoint /root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand/gpgformer_best.pt \
  --rgb-checkpoint /root/code/vepfs/GPGFormer/checkpoints/freihand_20260304_rgb_only/freihand/gpgformer_best.pt \
  --device cuda:0 \
  --split eval \
  --num-samples 8 \
  --heatmap-reducer l2 \
  --heatmap-cmap jet \
  --output-dir /root/code/vepfs/GPGFormer/outputs/freihand_token_heatmap_run
```

HO3D 示例：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_token_analysis.py \
  --dataset ho3d \
  --checkpoint /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/ho3d_20260319/ho3d/gpgformer_best.pt \
  --device cuda:0 \
  --split eval \
  --num-samples 8 \
  --output-dir /root/code/vepfs/GPGFormer/outputs/ho3d_token_heatmap_run
```

Dex-YCB 示例：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_token_analysis.py \
  --dataset dexycb \
  --checkpoint /root/code/vepfs/GPGFormer/checkpoints/ablations_v2/dexycb_20260318/dexycb/gpgformer_best.pt \
  --device cuda:0 \
  --split eval \
  --num-samples 8 \
  --output-dir /root/code/vepfs/GPGFormer/outputs/dexycb_token_heatmap_run
```

如需额外把 GT hand bbox 画在 overview 和 overlay 图上：

```bash
/root/code/vepfs/miniconda3/envs/moge/bin/python \
  /root/code/vepfs/GPGFormer/tools/visualize_freihand_token_analysis.py \
  --dataset freihand \
  --split eval \
  --sample-indices 0 12 58 \
  --show-bbox \
  --output-dir /root/code/vepfs/GPGFormer/outputs/freihand_token_heatmap_bbox
```

常用参数：

- `--dataset freihand|ho3d|dexycb`：选择数据集
- `--checkpoint <path>`：指定双模态 checkpoint；若不传，脚本会使用内置默认路径
- `--rgb-checkpoint <path>`：可选的单 RGB checkpoint；若不传，则脚本会复用双模态 checkpoint，并在前向时关闭几何分支得到 RGB-only heatmap
- `--sample-indices 0 5 12`：只跑指定样本
- `--num-samples 8`：当未指定样本索引时，非训练 split 默认取前 `N` 个样本，`train` 按随机种子采样
- `--heatmap-reducer l2|l1|var`：设置通道压缩方式；`l2` 最常用
- `--heatmap-cmap jet|viridis|magma|turbo`：设置伪彩色风格
- `--overlay-alpha 0.58`：控制热图与暗化输入图的叠加强度
- `--darken-factor 0.38`：控制输入底图变暗程度
- `--show-bbox`：可选地绘制白色 GT hand bbox，默认关闭

输出目录结构：

- 根目录下会写出 `summary.json`
- 每个样本一个子目录：`sample_<dataset_idx>_<sample_id>/`
- 每个样本目录下包含：
  - `input_crop.png`
  - `input_dark.png`
  - `input_crop_with_bbox.png`（仅当传入 `--show-bbox` 时导出）
  - `hand_region_mask_preview.png`
  - `overview_energy.png`
  - `overview_semantic.png`
  - `overview.png`（默认与 `overview_semantic.png` 相同，便于快速查看）
  - `metadata.json`
  - `rgb_patch_heatmap.png` / `rgb_patch_overlay.png`
  - `geo_tokens_heatmap.png` / `geo_tokens_overlay.png`
  - `fused_pre_backbone_heatmap.png` / `fused_pre_backbone_overlay.png`
  - `backbone_last_multimodal_heatmap.png` / `backbone_last_multimodal_overlay.png`
  - `backbone_last_rgb_only_heatmap.png` / `backbone_last_rgb_only_overlay.png`
  - `cross_modal_cosine_distance_heatmap.png` / `cross_modal_cosine_distance_overlay.png`
  - `rgb_hand_similarity_heatmap.png` / `rgb_hand_similarity_overlay.png`
  - `geo_hand_similarity_heatmap.png` / `geo_hand_similarity_overlay.png`
  - `fused_hand_similarity_heatmap.png` / `fused_hand_similarity_overlay.png`
  - `fusion_gain_over_best_heatmap.png` / `fusion_gain_over_best_overlay.png`
  - `backbone_last_multimodal_similarity_heatmap.png` / `backbone_last_multimodal_similarity_overlay.png`
  - `backbone_last_rgb_only_similarity_heatmap.png` / `backbone_last_rgb_only_similarity_overlay.png`
  - `backbone_last_multimodal_gain_heatmap.png` / `backbone_last_multimodal_gain_overlay.png`

说明：

- 所有 overlay 都画在模型输入 crop 上，不回贴原始整图，这样与 token 网格是一一对应的。
- 脚本会先根据 2D 关节构建一个 `convex hull + dilation` 的近似手部区域，而不是只用粗 bbox。这样可以更稳定地判断热图是否真的落在手上。
- `metadata.json` 和 `summary.json` 中会记录每类热图的 `inside_hand_mass`、`inside_outside_gap`、`top10_inside_ratio`、`peak_xy` 和 `peak_value`。
- `inside_hand_mass` 表示热图正响应中落在近似手部区域内的比例，越大越好。
- `top10_inside_ratio` 表示热图最强的前 10% 响应中有多少落在手部区域内，越大越好。
- `overview_energy.png` 用来诊断原始 token 能量分布。早期 energy map 经常会对背景纹理、强边缘和光照变化敏感，所以不要直接把它们当作“模型是否关注手”的证据。
- `overview_semantic.png` 才是更推荐的分析视图：它把“token 是否和最终手部语义一致”“Geometry 和 RGB 是否互补”“Fusion 是否带来额外收益”直接可视化出来。
- `Cross-modal cosine distance` 最适合找阴影、自遮挡和边界这些 RGB 歧义区域。
- `Fusion gain over best single` 最适合证明融合后 token 不只是折中，而是在困难区域超越了单独的 RGB / Geometry token。
- `Backbone last layer (multimodal)` 与 `Backbone last layer (RGB-only)` 以及对应的 similarity / gain 图，最适合做最终结论，判断双模态是否在深层表征中更稳定地对齐手部语义。
- 对于 HO3D 和 Dex-YCB，如果没有单独的 RGB-only checkpoint，脚本会自动退化为“同一个双模态模型、但关闭几何输入”的 RGB-only 视图，这样三套数据集都能统一跑通。

### Export RGB + depth by dataset index

`visualization/export_rgb_depth_by_index.py` 用于按数据集索引导出某个样本对应的原始 RGB 图和深度图，便于快速核对原始输入。

当前支持：

- `dexycb`
- `ho3d`

脚本会自动：

- 根据 `--dataset` 和 `--index` 定位样本
- 导出 `rgb.png`
- 导出原始深度图 `depth_raw.png`
- 导出解码后的深度可视化 `depth_decoded_vis.png`
- 导出样本元信息 `meta.json`

默认输出目录格式：

- `/root/code/vepfs/GPGFormer/outputs/rgb_depth_by_index/<dataset>_<split>_idx_<index>/`

说明：

- DexYCB 默认 `--split test`，并支持 `--setup`，默认是 `s0`
- HO3D 默认 `--split evaluation`
- 脚本内部使用 `bbox_source='gt'` 和 `align_wilor_aug=False`，只做原始样本定位与导出，不做训练时的数据增强
- HO3D 会额外把官方深度 PNG 解码成可视化灰度图保存到 `depth_decoded_vis.png`

运行前先进入仓库根目录：

```bash
cd /root/code/hand_reconstruction/GPGFormer
```

示例命令：

```bash
# DexYCB: 导出 test split 的第 0 个样本
python visualization/export_rgb_depth_by_index.py \
  --dataset dexycb \
  --index 0

# DexYCB: 指定 setup、split 和输出目录
python visualization/export_rgb_depth_by_index.py \
  --dataset dexycb \
  --setup s0 \
  --split test \
  --index 42 \
  --output-root /root/code/vepfs/GPGFormer/outputs

# HO3D: 导出 evaluation split 的第 0 个样本
python visualization/export_rgb_depth_by_index.py \
  --dataset ho3d \
  --index 0

# HO3D: 指定数据根目录和输出目录
python visualization/export_rgb_depth_by_index.py \
  --dataset ho3d \
  --split evaluation \
  --index 15 \
  --ho3d-root /root/code/vepfs/dataset/HO3D_v3 \
  --output-root /root/code/vepfs/GPGFormer/outputs
```

常用参数：

- `--dataset {dexycb,ho3d}`
- `--index <int>`
- `--split <name>`
- `--setup <name>`：仅 DexYCB 使用
- `--dexycb-root <path>`
- `--ho3d-root <path>`
- `--output-root <path>`

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
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/dexycb_20260305/dexycb/gpgformer_best.pt \
  --config configs/config_dexycb2.yaml \
  --out-dir outputs/dexycb_20260305 \
  --overlay-mesh \
  --render-mesh \
  --num-samples 24 \
  --batch-size 8

python visualize_reconstruction.py \
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/ablations/ho3d_multimodal_mask_consistency_20260310/ho3d/gpgformer_best.pt \
  --config configs/ablations/config_ho3d_multimodal_mask_consistency.yaml \
  --out-dir outputs/ho3d_20260310 \
  --overlay-mesh \
  --render-mesh \
  --num-samples 40 \
  --batch-size 8

python visualize_reconstruction.py \
  --ckpt /root/code/vepfs/GPGFormer/checkpoints/freihand_multimodal_mask_consistency_meshlight_20260311/freihand/gpgformer_best.pt \
  --config configs/config_freihand_multimodal_mask_consistency_meshlight.yaml \
  --out-dir outputs/freihand_20260311 \
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

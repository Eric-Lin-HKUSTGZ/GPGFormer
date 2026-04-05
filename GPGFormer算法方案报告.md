# GPGFormer 算法方案报告

## 1. 报告范围

本文档对应如下当前推荐配置：

`/root/code/hand_reconstruction/GPGFormer/configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`

这份配置的核心特点不是旧版的 `rgb_token_mask + consistency` 方案，而是：

1. WiLoR RGB 主干 + 冻结的 MoGe2 几何先验
2. `sum + channel_concat` 的保守多模态融合
3. 新增并启用的 `geo_side_adapter`（map-level 几何侧分支）
4. 关闭 token-level `side_tuning`
5. 使用 `SJTA` 做后验 MANO 参数细化
6. 使用 joint/mesh 联合指标选择最佳 checkpoint

文档中的结构说明、训练策略和推荐参数均以这份配置及当前代码实现为准。

## 2. 算法概述

GPGFormer 是一个面向单目 RGB 3D 手部重建的双模态框架。模型以 WiLoR 提供稳定的视觉表征，以冻结的 MoGe2 提供几何先验，再通过轻量融合与 MANO 解码头恢复手部姿态、形状和网格。

当前推荐方案相较旧版报告，最重要的更新是将几何增强从“只在 token 级做微调”推进到“先在 feature map 级补充分支，再 token 化再融合”。也就是新增的 `GeoSideAdapter`。它直接作用在 MoGe2 neck feature map 上，通过附加侧分支产生新的几何通道，然后与原始 MoGe2 特征按通道拼接，再交给 `GeoTokenizer` 压到 1280 维 token 空间。这样做的好处是：

1. 不覆盖原始 MoGe2 表达，而是保留原特征并追加可学习补充
2. 几何增强发生在 token 化之前，能先在空间维度上整理局部模式
3. 风险比直接强残差改写主特征更低，训练更稳

## 3. 整体流程

```text
输入 RGB crop
  (实际送入 backbone 前统一到 256x192)
    ↓
RGB 路径:
  WiLoR ViT-L backbone
    ↓
  RGB patch tokens / image features

几何路径:
  冻结 MoGe2
    ↓
  neck feature map
    ↓
  GeoSideAdapter
    ↓
  GeoTokenizer + CoordPosEmbed
    ↓
  geometry tokens

双模态融合:
  sum fusion + channel_concat
    ↓
融合后的 WiLoR token features
    ↓
MANO Transformer Decoder Head
  6-layer decoder + 3-step IEF
    ↓
初始 MANO 参数与相机
    ↓
SJTA refinement
  2 steps
    ↓
最终 MANO 参数
    ↓
MANO layer
    ↓
3D joints / mesh vertices
```

## 4. 核心模块

### 4.1 RGB 主干编码器

当前配置使用 `WiLoR` 作为 backbone：

1. `backbone_type: wilor`
2. `image_size: 256`
3. 实际 patch feature map 采用 `(256, 192)` 的输入高宽，对应 `16 x 12` patch grid
4. 输出上下文特征维度为 `1280`

WiLoR 路径仍然是主表征来源，因此训练上对 backbone 使用较小学习率倍率 `0.5`，以保护预训练知识。

### 4.2 几何先验模块 MoGe2

当前配置：

1. `use_geo_prior: true`
2. `moge2_output: neck`
3. `moge2_num_tokens: 400`

MoGe2 在当前实现中是冻结的，只负责给出几何先验，不参与主干反向更新。其 neck feature map 先被复制成普通 tensor，再交给后续可训练模块处理。

### 4.3 GeoSideAdapter

这是当前推荐方案最需要强调的新模块。

#### 4.3.1 模块位置

`GeoSideAdapter` 位于：

`MoGe2 neck feature map -> GeoTokenizer`

即它先处理二维几何特征图，再做 token 化，而不是处理已经展平后的 token。

代码路径：

1. `gpgformer/models/gpgformer.py`
2. `gpgformer/models/tokenizers/geo_side_adapter.py`

前向顺序可概括为：

```python
geo_feat = self.moge2(img_moge)
geo_feat = geo_feat.clone()

if self.geo_side_adapter is not None:
    geo_feat = self.geo_side_adapter(geo_feat)

geo_tokens, coords = self.geo_tokenizer(geo_feat)
```

#### 4.3.2 输入输出

`GeoSideAdapter` 的输入是 MoGe2 的 neck feature map：

```text
Input:  (B, C, H, W)
Output: (B, C + C_side, H, W)
```

当前推荐配置：

1. `enabled: true`
2. `side_channels: 128`
3. `depth: 2`
4. `dropout: 0.05`
5. `norm_groups: 32`

这意味着它会在原始几何通道之外，再生成 128 个侧分支通道，然后与原特征做 channel concat。

#### 4.3.3 结构形式

实现是 HandOS 风格的 map-level side adapter。结构可以写成：

```text
原始 MoGe2 neck feat (B,C,H,W)
  ↓
1x1 Conv(C -> 128) + GroupNorm + GELU + Dropout2d
  ↓
3x3 Conv(128 -> 128) + GroupNorm + GELU + Dropout2d
  ↓
side feat (B,128,H,W)
  ↓
channel concat with original feat
  ↓
augmented geo feat (B,C+128,H,W)
```

其中：

1. 第一层用 `1x1 conv` 做通道压缩和对齐
2. 后续 `depth - 1` 层使用 `3x3 conv` 建模局部空间上下文
3. `GroupNorm` 的组数会自动调整为能整除通道数的值，不会强行固定死
4. 输出不是残差相加，而是与原始特征做拼接

#### 4.3.4 为什么要用 channel concat

这一步的设计意图非常明确：不直接改写原始几何特征，而是“保留原始 MoGe2 表达 + 追加补充几何上下文”。和纯残差相比，这种方式有三个好处：

1. 原始 MoGe2 先验不会被侧分支覆盖
2. 新增信息由后续 `GeoTokenizer` 的 `1x1 proj` 自主学习如何压缩到 1280 维
3. 在训练早期更稳，适合与冻结先验一起工作

#### 4.3.5 与 token-level side tuning 的区别

当前代码里同时存在两个“侧分支”概念：

1. `geo_side_adapter`
   位置：feature map 级，发生在 `GeoTokenizer` 之前
2. `geo_side_tuning`
   位置：token 级，发生在 `GeoTokenizer` 之后

当前推荐配置中：

1. `geo_side_adapter.enabled: true`
2. `side_tuning.enabled: false`

也就是说，当前最终推荐方案明确选择的是 map-level 几何增强，而不是 token-level 侧调优。

### 4.4 GeoTokenizer 与位置编码

`GeoTokenizer` 将增强后的几何 feature map 压成 1280 维 geometry tokens。

当前配置：

1. `geo_tokenizer_use_pooling: true`
2. 输出 patch 尺度与 RGB 路径对齐，即 `16 x 12`

处理流程：

```text
augmented geo feat (B,C',H,W)
  ↓
AdaptiveAvgPool2d -> (16,12)
  ↓
1x1 Conv(C' -> 1280)
  ↓
flatten
  ↓
geo tokens (B,192,1280)
```

位置编码使用 `CoordPosEmbed`，根据归一化坐标 `[-1,1]` 的二维网格生成 1280 维位置向量。其参数近零初始化，用来降低对预训练 backbone 的扰动。

### 4.5 多模态融合

当前推荐配置使用：

1. `token_fusion_mode: sum`
2. `sum_fusion_strategy: channel_concat`
3. `sum_geo_gate_init: -1.2`
4. `fusion_proj_zero_init: true`

其实际融合逻辑不是简单相加，而是：

```python
x_norm = LN(x_rgb)
geo_norm = LN(x_geo)
z = cat([x_norm, geo_norm], dim=-1)
x = x_rgb + sigmoid(gate) * fusion_proj(z)
```

这里有三层稳定性设计：

1. 几何门控初值为 `-1.2`，即 `sigmoid(-1.2) ≈ 0.23`
2. `fusion_proj` 零初始化，训练初期近似纯 RGB
3. 几何分支还有独立的学习率 ramp，不会一开始就强推几何信息

### 4.6 MANO 解码头

当前使用 `wilor` 风格 MANO decoder，配置如下：

1. `ief_iters: 3`
2. `transformer_input: mean_shape`
3. `dim: 1024`
4. `depth: 6`
5. `heads: 8`
6. `dim_head: 64`
7. `mlp_dim: 2048`
8. `dropout: 0.0`

解码流程：

1. 从融合后的 image feature map 中提取上下文
2. 通过 6 层 Transformer decoder 回归 MANO latent
3. 用 3 次 IEF 迭代更新姿态、形状和相机参数
4. 最终交给 MANO layer 得到 mesh 与 joints

模型内部回归的是 MANO 参数，包括：

1. `global orient`
2. `hand pose`
3. `betas`
4. `weak-perspective camera`

### 4.7 SJTA 特征细化

当前配置：

1. `feature_refiner.method: sjta`
2. `sjta_bottleneck_dim: 256`
3. `sjta_num_heads: 4`
4. `sjta_use_2d_prior: true`
5. `sjta_num_steps: 2`

SJTA 的位置在 MANO head 之后。它不是特征图增强器，而是基于初始 MANO 结果做后验 refinement。流程可以概括为：

1. 先用 MANO head 得到初始 pose/shape/cam
2. 从当前 3D joints 和投影 2D 先验构造 joint tokens
3. 让 joint tokens 与 image feature 做 cross-attention
4. 预测参数增量
5. 重复 2 次 refinement

因此，当前推荐配置的整体结构并不是“融合后直接出最终结果”，而是：

`Fusion -> MANO Head -> SJTA -> Final MANO`

## 5. 训练策略

### 5.1 优化器与参数分组

优化器为 `AdamW`：

1. `lr: 2e-5`
2. `weight_decay: 1e-4`
3. `epochs: 60`
4. `batch_size: 64`
5. `val_batch_size: 64`
6. `grad_clip_norm: 1.0`

学习率倍率配置：

```yaml
lr_multiplier:
  backbone: 0.5
  head: 2.0
  geo_fusion: 0.25
  side_tuning: 0.8
```

其中参数组含义是：

1. `backbone`
   WiLoR 视觉主干
2. `head`
   `mano_head` 与 `feature_refiner`
3. `geo_fusion`
   `geo_tokenizer`、`geo_pos`、encoder 内部 fusion 模块
4. `side_tuning`
   名字虽然叫 side_tuning，但代码里同时覆盖 `geo_side_tuning` 和 `geo_side_adapter`

因此在当前配置下，`side_tuning: 0.8` 实际上是在控制 `GeoSideAdapter` 的学习率倍率。

### 5.2 学习率调度

主调度为 `warmup + cosine decay`：

1. `warmup_epochs: 8`
2. `min_lr: 5e-7`

除此之外还有两个额外 ramp：

#### 5.2.1 几何融合 ramp

1. `geo_fusion_start_factor: 0.05`
2. `geo_fusion_ramp_epochs: 20`

即几何融合相关模块前期只用 5% 的组内学习率，再逐步提升到完整倍率。

#### 5.2.2 侧分支 ramp

1. `side_tuning_start_factor: 0.1`
2. `side_tuning_ramp_epochs: 12`

虽然配置名仍叫 `side_tuning`，但在当前模型里它主要作用于 `GeoSideAdapter`。这意味着新加入的 map-level 侧分支在前 12 个 epoch 会以更保守的有效学习率启动。

### 5.3 数据增强与输入设置

数据集为 FreiHAND，当前配置：

1. `dataset.name: freihand`
2. `img_size: 256`
3. `root_index: 9`
4. `align_wilor_aug: true`
5. `bbox_source_eval: detector`
6. `use_trainval_split: false`

WiLoR 风格增强参数：

```yaml
wilor_aug_config:
  SCALE_FACTOR: 0.3
  ROT_FACTOR: 30
  TRANS_FACTOR: 0.02
  COLOR_SCALE: 0.2
  ROT_AUG_RATE: 0.6
  TRANS_AUG_RATE: 0.5
  DO_FLIP: false
  FLIP_AUG_RATE: 0.0
  EXTREME_CROP_AUG_RATE: 0.0
  EXTREME_CROP_AUG_LEVEL: 1
```

这套配置是比较稳健的 FreiHAND 训练设定，强调：

1. 中等尺度与旋转扰动
2. 不做左右翻转
3. 不启用极端裁剪

### 5.4 当前推荐配置中已关闭的训练项

和旧版报告不同，以下模块在当前推荐配置中全部关闭：

#### 5.4.1 RGB token mask

```yaml
rgb_token_mask:
  enabled: false
  apply_to_main_forward: false
  apply_prob: 0.0
  ratio_start: 0.0
  ratio_end: 0.0
  ramp_epochs: 0
```

即当前推荐方案不再依赖遮挡辅助分支。

#### 5.4.2 consistency loss

```yaml
consistency:
  enabled: false
  w_3d: 0.0
  w_mesh: 0.0
```

因此总损失中不包含额外的一致性正则。

### 5.5 几何分支 dropout

当前配置：

`geo_branch_dropout_prob: 0.04`

这一步发生在 `geo_tokens` 形成之后。训练时会以 4% 概率对几何 token 分支做随机丢弃，用于避免模型过度依赖几何先验。

## 6. 损失函数设计

当前推荐配置中的主要损失项如下：

### 6.1 关节与网格监督

```yaml
w_2d: 1.0
w_kcr_2d: 1.0
w_3d_joint: 5.0
w_bone_length: 1.0
joint_3d_tip_weight: 2.5
w_3d_vert: 0.10
```

说明：

1. `w_3d_joint: 5.0` 仍是主监督
2. 指尖关节 `tip_joint_indices=[4,8,12,16,20]` 额外加权 `2.5`
3. 网格顶点监督保留但较轻，仅 `0.10`
4. `bone_length` 继续用于约束结构合理性

### 6.2 MANO 参数监督

```yaml
w_global_orient: 0.4
w_hand_pose: 0.4
w_scale: 0.0
w_betas: 0.01
w_shape: 0.05
mano_param_loss_type: smooth_l1
mano_param_smooth_l1_beta: 0.05
mano_param_per_sample_clip: 1.0
```

说明：

1. `global_orient` 与 `hand_pose` 权重适中，避免参数项压过 3D 关节监督
2. `w_betas` 是轻量 beta anchor
3. `w_shape` 提供额外形状正则
4. `Smooth L1 + per-sample clip` 用来抑制异常样本

### 6.3 总体损失表达

在当前推荐配置下，可以写成：

```text
L_total =
  1.0 * L_2d
  + 5.0 * L_3d_joint
  + 1.0 * L_bone
  + 0.10 * L_3d_vert
  + 0.4 * L_global_orient
  + 0.4 * L_hand_pose
  + 0.01 * L_betas
  + 0.05 * L_shape
```

一致性项在当前配置中关闭，因此不进入总损失。

## 7. 评估与 checkpoint 选择

### 7.1 评估指标

常规评估指标仍包括：

1. `MPJPE`
2. `PA-MPJPE`
3. `MPVPE`
4. `PA-MPVPE`
5. `F-score@5mm / @15mm`

此外还有 FreiHAND mesh 一致性过滤阈值：

`mesh_fallback_kp_consistency_thr_mm: 40.0`

### 7.2 最佳 checkpoint 选择策略

当前配置不是按单独 `PA-MPJPE` 选最佳模型，而是：

`best_ckpt_metric: joint_mesh`

代码中的打分方式是：

```python
joint_mesh_avg = 0.5 * (PA-MPJPE + PA-MPVPE)
```

因此当前推荐配置在模型选择上明确追求 joint 和 mesh 的折中，而不是只优化关节点。

## 8. 当前推荐配置摘要

### 8.1 模型配置

| 模块 | 当前配置 | 说明 |
|------|------|------|
| Backbone | WiLoR ViT-L | RGB 主干 |
| Geometry Prior | MoGe2 neck | 冻结几何先验 |
| GeoSideAdapter | 开启 | map-level 几何侧分支 |
| GeoTokenizer | pooling 到 16x12 | 与 RGB patch 数对齐 |
| Fusion | `sum + channel_concat` | 保守注入几何信息 |
| MANO Head | 6-layer decoder + 3 IEF | 主回归头 |
| Refiner | SJTA, 2 steps | 后验参数细化 |
| GeoSideTuning | 关闭 | 当前不使用 token-level 侧调优 |

### 8.2 关键超参数

| 参数 | 值 |
|------|------|
| `moge2_num_tokens` | 400 |
| `sum_geo_gate_init` | -1.2 |
| `fusion_proj_zero_init` | true |
| `geo_branch_dropout_prob` | 0.04 |
| `geo_side_adapter.side_channels` | 128 |
| `geo_side_adapter.depth` | 2 |
| `geo_side_adapter.dropout` | 0.05 |
| `sjta_num_steps` | 2 |
| `batch_size` | 64 |
| `epochs` | 60 |

### 8.3 训练配置

| 项目 | 当前配置 |
|------|------|
| Optimizer | AdamW |
| Base LR | 2e-5 |
| Weight Decay | 1e-4 |
| Warmup | 8 epochs |
| Min LR | 5e-7 |
| Geo Fusion Ramp | 20 epochs |
| Side Branch Ramp | 12 epochs |
| Gradient Clip | 1.0 |
| EMA | 关闭 |
| RGB Token Mask | 关闭 |
| Consistency Loss | 关闭 |
| Best CKPT Metric | `joint_mesh` |

## 9. 为什么当前方案强调 GeoSideAdapter

如果把当前推荐模型的设计重点压缩成一句话，就是：

“在不破坏原始 WiLoR 主干和冻结 MoGe2 先验的前提下，用一个轻量的 map-level 几何侧分支去补充几何局部结构，再通过保守融合和 SJTA 做两阶段细化。”

`GeoSideAdapter` 是这条设计线里非常关键的一步，因为它解决了两个实际问题：

1. 仅靠原始 MoGe2 token 直接融合时，几何信息表达偏硬，且对 token 对齐较敏感
2. 仅做 token-level `side_tuning` 时，可调空间主要在 token 通道维，空间局部模式利用不够充分

新增 `GeoSideAdapter` 后，几何增强先在 feature map 级完成，再交给 tokenizer 统一压缩到和 RGB 一致的 token 空间，整体更符合“先局部建模、再跨模态对齐”的处理顺序。

## 10. 使用方式

### 10.1 训练

```bash
python train.py \
  --config /root/code/hand_reconstruction/GPGFormer/configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml
```

多卡：

```bash
torchrun --nproc_per_node=4 train.py \
  --config /root/code/hand_reconstruction/GPGFormer/configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml
```

### 10.2 评估或推理

如需评估，请将 `--config` 保持为同一份配置，并配合对应输出目录下的 `gpgformer_best.pt` 使用。当前配置的输出目录为：

`/root/code/vepfs/GPGFormer/checkpoints/freihand_loss_beta_mesh_target51_recommended_sidetuning_20260317/freihand`

## 11. 总结

当前这版 GPGFormer 推荐方案的关键结论如下：

1. 主干仍然是 WiLoR + 冻结 MoGe2 的双模态框架
2. 当前最终推荐结构明确启用了 `GeoSideAdapter`，并关闭了 token-level `side_tuning`
3. `GeoSideAdapter` 是一个发生在 `GeoTokenizer` 之前的 map-level 侧分支模块，输出通过 channel concat 保留原几何信息并追加补充信息
4. 多模态融合采用保守的 `sum + channel_concat + low initial gate + zero-init projection`
5. 后端用 `MANO Head + SJTA(2 steps)` 做两阶段回归与细化
6. 训练上不再依赖 `rgb_token_mask` 和 `consistency loss`
7. 最佳模型按 `joint_mesh = 0.5 * (PA-MPJPE + PA-MPVPE)` 选择，更符合当前 joint/mesh 平衡目标

---

**报告更新时间**: 2026-03-17
**对应配置**: `config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`

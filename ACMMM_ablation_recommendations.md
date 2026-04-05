# GPGFormer 面向 ACMMM 的消融实验建议

## 目标

当前最优配置文件为：

- `/root/code/hand_reconstruction/GPGFormer/configs/config_freihand_loss_beta_mesh_target51_recommended_sidetuning.yaml`

面向 ACMMM 投稿时，消融实验部分的目标不是“尽可能多做实验”，而是用尽可能少的实验，支撑论文中最核心的贡献点。最重要的原则是：

- 只消融相互独立的关键模块
- 避免在主文中进行大规模超参数扫参
- 固定一套强训练配方，每次只改一个因素

从当前最优模型来看，真正值得写进消融实验的核心成分主要有四个：

1. 几何先验分支：`use_geo_prior`、`MoGe2 neck`、`sum + channel_concat` 融合
2. map-level 的 `GeoSideAdapter`
3. `SJTA` 特征细化模块
4. 结构化损失设计：bone length、指尖加权、mesh/beta/shape 正则

因此，最适合的设计是“两张主表 + 少量补充实验”。

## 推荐的主消融实验

### 表 A：模型组件消融

这张表应该作为论文中的主消融表。

| 编号 | 几何先验 | GeoSideAdapter | SJTA | 目的 |
|---|---|---|---|---|
| A0 | x | x | x | 纯 RGB 基线 |
| A1 | check | x | x | 验证几何先验是否有效 |
| A2 | check | check | x | 验证 map-level side adaptation 是否有效 |
| A3 | check | check | SJTA-1 | 验证轻量 refinement 是否有效 |
| A4 | check | check | SJTA-2 或最终版本 | 最终模型 |

推荐的具体实现方式：

- `A0`：`use_geo_prior: false`，`geo_side_adapter.enabled: false`，`feature_refiner.method: none`
- `A1`：保留当前 fusion 方式，但设 `geo_side_adapter.enabled: false`，`feature_refiner.method: none`
- `A2`：开启 `geo_side_adapter`，关闭 `feature_refiner`
- `A3`：开启 `geo_side_adapter`，设 `feature_refiner.method: sjta`，`sjta_num_steps: 1`
- `A4`：使用当前最优配置，即 `geo_side_adapter side_channels=128 depth=2`，`sjta_num_steps=2`

为什么这张表最重要：

- 它可以直接回答“性能提升到底来自哪里”
- 它和当前模型的设计思路完全一致
- 它避免把篇幅浪费在一些辨识度不高的负结果上，例如过多 fusion 变体比较

如果篇幅特别紧张，主文中甚至可以只保留 `A0/A1/A2/A4` 四行，把 `A3` 放到 supplementary。

### 表 B：损失函数设计消融

第二张表建议只保留最容易讲清楚、最容易让审稿人理解的两个损失项。

| 编号 | 完整损失 | 去掉 Bone Length | 去掉 Tip Weight | 目的 |
|---|---|---|---|---|
| B0 | check | x | x | 最终训练目标 |
| B1 | x | check | x | 验证结构约束有效性 |
| B2 | x | x | check | 验证指尖强调有效性 |

具体建议：

- `B0`：当前最优配置
- `B1`：设置 `w_bone_length: 0.0`
- `B2`：设置 `joint_3d_tip_weight: 1.0`

优先选择这两个损失项的原因：

- 解释非常直观，审稿人一眼就能理解
- 与 hand structure 和 fingertip quality 直接相关
- 比大规模扫 `w_shape`、`w_betas`、`w_3d_vert` 更适合写在主文中

除非你的论文把某个正则项本身作为核心贡献，否则不建议把 `w_shape`、`w_betas`、`w_3d_vert` 的扫参放进主文。

## 最小但最有说服力的实验集合

如果你希望把实验数压到尽可能少，同时又能支撑论文主结论，我最推荐只做下面 7 个实验：

1. `A0`：纯 RGB 基线
2. `A1`：RGB + Geo Prior
3. `A2`：RGB + Geo Prior + GeoSideAdapter
4. `A4`：RGB + Geo Prior + GeoSideAdapter + SJTA-final
5. `B1`：最终模型去掉 bone length loss
6. `B2`：最终模型去掉 fingertip weighting
7. 最终模型在 HO3D 上的迁移/泛化实验

这套组合在“实验成本”和“论文说服力”之间，基本是收益最高的方案。

## 可选但优先级较低的实验

这些实验只有在核心表格已经完成、且还有算力预算时才建议补充。

### O1：SJTA 步数/层数消融

这是最适合作为补充实验的小型超参数消融，因为你已经在做相关配置。

推荐比较：

- `SJTA-1`
- `SJTA-2`
- 如有余力再加 `SJTA-3`

呈现方式建议：

- 可以作为一个 2 到 3 行的小表
- 或者在正文里用一句话说明：一轮 refinement 已经有效，两轮在性能与复杂度之间达到最佳平衡

不要把它扩展成大规模扫参。

### O2：GeoSideAdapter 深度消融

你已经有 `layer_1`，而当前最优大致对应 `depth=2`。

如果 `GeoSideAdapter` 被你作为论文中的一个重要创新点，那么建议只做：

- `depth=0`，即关闭 adapter
- `depth=1`
- `depth=2`

这样已经足够。不要在主文里继续扫 `side_channels`、`dropout`、`norm_groups`。

## 不建议放在主文中的消融方向

下面这些方向很容易消耗大量算力，但在 ACMMM 篇幅受限时，性价比通常不高：

- 对过多 fusion mode 做比较，例如 `concat`、`sum`、`cross_attn`、`weighted`、`normalized`、`channel_concat`
- 扫 `moge2_num_tokens`
- 对多个正则权重进行大规模 sweep，例如 `w_shape`、`w_betas`、`w_3d_vert`
- 同时大量比较 token-level `side_tuning` 和 map-level `geo_side_adapter`
- 对 FreiHAND 和 HO3D 的每一个消融项都重复完整实验

如果 token-level `side_tuning` 明显不如 map-level `geo_side_adapter`，建议在 supplementary 或 appendix 中简单作为负结果说明即可，不需要在主文里展开很多行。

## 推荐的论文叙事方式

整篇论文的实验部分，建议围绕以下三条主结论展开：

1. 几何先验相对于纯 RGB 重建是有效的
2. map-level side adaptation 能改善冻结几何先验在 tokenization 前的特征质量
3. 轻量 SJTA refinement 和结构化损失可以进一步提升 joint 和 mesh 质量

这样，消融实验就会和论文主线自然对齐：

- 表 A 支撑结论 1 和结论 2，并部分支撑结论 3
- 表 B 支撑结论 3 中关于结构监督的部分
- 额外的 HO3D 结果支撑泛化能力

这种写法会比把 architecture、fusion、loss 全部塞进一张大表更清晰。

## 公平性检查清单

为了让消融实验更经得住审稿人的质疑，建议除非某一行实验就是专门研究该因素，否则以下设置全部固定：

- 相同的训练划分和 `trainval_seed`
- 相同的 `epochs`
- 相同的 optimizer 和学习率调度
- 相同的 `deterministic` 和 `seed`
- 相同的图像尺寸和数据增强
- 相同的评测协议和指标

结合你现在的设置，还需要特别注意一点：

- `GeoSideAdapter` 会改变显存开销
- 如果某些实验不得不使用更小的 batch size，最好通过 gradient accumulation 保持等效 batch size 一致，或者在论文中明确说明差异

否则审稿人很容易质疑性能差异是否来自优化设置，而不是模块本身。

## 推荐的最终表格布局

### 主文

- 表 1：组件消融，`A0-A4`
- 表 2：损失消融，`B0-B2`
- 额外一个结果或小表：最终模型在 HO3D 上的迁移/泛化表现

### 补充材料

- SJTA 步数对比，`1/2/3`
- GeoSideAdapter 深度对比，`off/1/2`
- token-level side tuning 的负结果说明，如有需要

## 最终推荐

如果只能选择一套最适合当前 GPGFormer 版本、同时又适合 ACMMM 篇幅限制的消融实验方案，我最推荐你跑：

1. 纯 RGB 基线
2. `+ Geo Prior`
3. `+ GeoSideAdapter`
4. `+ GeoSideAdapter + SJTA-final`
5. 最终模型去掉 bone-length loss
6. 最终模型去掉 fingertip weighting
7. 最终模型在 HO3D 上的结果

这套实验数量不多，但已经足够清楚地说明性能提升来自哪里。

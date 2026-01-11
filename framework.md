## 语言版“算法流程图”（按数据流箭头写）

### 节点与箭头（从左到右）
1. **输入节点：RGB 图像**
   - 输入：整张 RGB 图 \(I \in \mathbb{R}^{H_0 \times W_0 \times 3}\)

2. **手部检测器 detector**
   - 操作：在整张图上检测手的边界框（或训练时使用 GT bbox）
   - 输出：手框 \(b = (x,y,w,h)\)

3. **裁剪节点 crop hand**
   - 操作：根据 \(b\) 从 \(I\) 中裁剪并 resize 得到手部局部图（并对齐到 WiLoR 期望的输入比例）
   - 输出：手部裁剪图 \(I_h \in \mathbb{R}^{H \times W \times 3}\)

4. **分支 A（外观/纹理分支）：Tokenizer1（WiLoR ViT patch embedding）**
   - 输入：\(I_h\)
   - 操作：将 \(I_h\) 划分为 patch 并做线性投影（ViT patch embedding）
   - 输出：外观 token 序列
     \[
       T_{\text{img}} \in \mathbb{R}^{N_1 \times D},\quad
       N_1 = \frac{H}{P}\cdot\frac{W}{P}
     \]

5. **分支 B（几何先验分支）：geometry prior（MoGe2, frozen）**
   - 输入：\(I_h\)（MoGe2 内部会做归一化等）
   - 操作：使用 MoGe2 提取几何先验特征
   - 输出：MoGe2 Conv neck 特征图（target features）
     \[
       F_g \in \mathbb{R}^{H_g \times W_g \times C_g}
     \]

6. **Tokenizer2（GeoTokenizer）：把 \(F_g\) 转成 token**
   - 输入：\(F_g\)
   - 操作：pool/flatten + 1x1 projection，把几何特征映射到与 ViT 相同嵌入维度 \(D\)
   - 输出：几何 token 序列
     \[
       T_{\text{geo}} \in \mathbb{R}^{N_2 \times D}
     \]

7. **Token 融合（geo + img，一起送入 ViT）**
   - 操作：concat 两种 token（当前实现 **geo 在前，img 在后**）：
     \[
       T = [T_{\text{geo}}; T_{\text{img}}] \in \mathbb{R}^{(N_2+N_1)\times D}
     \]
   - 编码：
     - 位置编码 \(E_{\text{pos}}\)：img patch token 使用 ViT pos embed；geo token 使用坐标位置编码
     - 类型编码 \(E_{\text{type}}\)：区分 img / geo
   - **重要变化（架构更新）**：当前 GPGFormer **不再向 ViT 输入任何 MANO special tokens**
     - 不使用 pose special tokens
     - 不使用 shape special token
     - 不使用 camera special token

8. **Transformer（WiLoR ViT-L blocks，仅作为特征提取器）**
   - 输入：\(T_0 = T + E_{\text{pos}} + E_{\text{type}}\)
   - 输出：上下文化后的 token：
     \[
       T_L \in \mathbb{R}^{(N_2+N_1)\times D}
     \]

9. **从 token 恢复空间特征（conditioning_feats）**
   - 操作：取出 \(T_L\) 的 patch token 部分（最后 \(N_1\) 个 token），reshape 成特征图：
     \[
       F \in \mathbb{R}^{H_p \times W_p \times D}
     \]
     实现中为 \((B,D,H_p,W_p)\)。
   - 解释：patch token 已经与 geo token 通过 self-attention 融合，所以 \(F\) 是 **geometry-guided** 的外观特征图。

10. **HaMeR-style MANO 回归头（MANOTransformerDecoderHead + IEF）**
   - 输入：conditioning_feats 特征图 \(F\)（flatten 后作为 context tokens）
     \[
       C = \text{flatten}(F) \in \mathbb{R}^{(H_pW_p)\times D}
     \]
   - Query token（1 个 token）：
     - `TRANSFORMER_INPUT=mean_shape`：用当前迭代的 \((\theta,\beta,\text{cam})\) 拼成 token
     - 或 `TRANSFORMER_INPUT=zero`：用全 0 token
   - 结构：cross-attention `TransformerDecoder`（query ↔ context） + 线性头 `decpose/decshape/deccam`
   - IEF（Iterative Error Feedback）：迭代 \(K\) 次，每次预测 residual 并累加：
     - \(\theta \leftarrow \theta + \Delta\theta\)
     - \(\beta \leftarrow \beta + \Delta\beta\)
     - \(\text{cam} \leftarrow \text{cam} + \Delta\text{cam}\)
   - 初始化：从 `MANO.MEAN_PARAMS` 提供的均值 `pose/shape/cam` 初始化（若 mean pose 为 6D，会转成 AA）
   - 输出：
     1) **MANO 参数**：global_orient + hand_pose（axis-angle aa → rotmat）+ betas
     2) **weak-perspective cam**：`pred_cam ∈ R^3`

11. **从 weak-persp cam 计算 camera translation（perspective t）**
   - 使用 dataloader 提供的每样本相机内参 focal（像素）：\((f_x,f_y)\)
   - 计算：
     \[
       t_z = \frac{2 f_x}{\text{image\_size}\cdot s + \epsilon}
     \]
     其中 \(s=\text{pred\_cam}[0]\)，\(t_x=\text{pred\_cam}[1]\)，\(t_y=\text{pred\_cam}[2]\)。
   - 输出：camera translation \(t \in \mathbb{R}^3\)

12. **（最终重建）用 MANO 层生成网格/关节**
   - 操作：将预测的 MANO 参数送入 MANO layer
   - 输出：3D 手网格顶点 \(V \in \mathbb{R}^{M \times 3}\)、3D 关节 \(J \in \mathbb{R}^{K \times 3}\)
   - 配合相机平移 \(t\) 可将 \(V/J\) 从 MANO 局部坐标系搬到相机坐标系，并可投影得到 2D 关键点用于监督或可视化

### 要求（必须执行）
1. 检测模型使用 WiLoR 训练好的手部检测器，权重已拷贝到：`GPGFormer/weights/wilor/detector.pt`
2. ViT 与 Tokenizer1 使用 WiLoR 的 ViT-Large，并使用 WiLoR 预训练权重初始化：`GPGFormer/weights/wilor/wilor_final.ckpt`
3. 使用 MoGe2 预训练模型（huggingface：Ruicheng/moge-2-vitl-normal），相关代码已最小化拷贝进：`GPGFormer/third_party/moge_min`
4. 几何先验的 target features 使用 MoGe2 的 Conv neck 输出
5. MoGe2 不参与学习更新（冻结参数）
6. 参考 UniHandFormer 的 dataloader 逻辑，实现 Dex-YCB / HO3D / FreiHAND 三个数据集的数据加载器（不修改 UniHandFormer 代码）
7. 监督信号为 3D mesh、3D joints、2D keypoints（loss/metric 参考 UniHandFormer，不修改 UniHandFormer 代码）
8. 涉及到的权重全部存放到 `GPGFormer/weights/`；第三方依赖最小化代码存放到 `GPGFormer/third_party/`
9. 严禁修改 `GPGFormer/` 之外的任何代码

### 当前实现补充说明（对应最新架构）
- **ViT 输入 token**：仅包含 `geo tokens + image patch tokens`，不包含任何 MANO pose/shape/cam special token。
- **MANO 参数回归**：采用 HaMeR-style `MANOTransformerDecoderHead`（cross-attention decoder + IEF），均值参数来自 `MANO.MEAN_PARAMS`。
- **相机内参**：优先使用 dataloader 提供的每样本 `cam_param=(fx,fy,cx,cy)`，不再依赖固定 focal_length 常量。


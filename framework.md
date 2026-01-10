## 语言版“算法流程图”（按数据流箭头写）

### 节点与箭头（从左到右）
1. **输入节点：RGB 图像**
   - 输入：整张 RGB 图 $I \in \mathbb{R}^{H_0 \times W_0 \times 3}$

2. **手部检测器 detector**
   - 操作：在整张图上检测手的边界框（或关键点后再取框）
   - 输出：手框 $b = (x,y,w,h)$（可含置信度、左右手类别等）

3. **裁剪节点 crop hand**
   - 操作：根据 $b$ 从 $I$ 中裁剪并 resize 得到手部局部图
   - 输出：手部裁剪图 $I_h \in \mathbb{R}^{H \times W \times 3}$

4. **分支 A（外观/纹理分支）：Tokenizer1（对裁剪手图做 token）**
   - 输入：$I_h$
   - 操作：将 $I_h$ 划分为 patch 并做线性投影（类似 ViT patch embedding）
   - 输出：外观 token 序列  
     $T_{\text{img}} \in \mathbb{R}^{N_1 \times D}$  
     其中 $N_1 = \frac{H}{P}\cdot\frac{W}{P}$（若 patch size 为 $P$），$D$ 为嵌入维度

5. **分支 B（几何先验分支）：geometry prior（图中写 use MoGe2）**
   - 输入：来自 **crop hand** 的信息（通常就是 $I_h$，也可能包含 $b$ 或相机内参等）
   - 操作：用某个“几何先验估计器”（图里标注 MoGe2）从手部裁剪图中提取/估计几何相关表征  
     （例如：深度/法向/3D 关键点粗估/网格粗形状/几何特征图等——图上统称为 geometry prior）
   - 输出：**target features**（目标几何特征）  
     用一个特征张量表示：$F_g \in \mathbb{R}^{H_g \times W_g \times C_g}$（或任意你定义的几何特征格式）

6. **Tokenizer2（把 target features 转成 token）**
   - 输入：$F_g$
   - 操作：将几何特征图（或几何特征集合）序列化/patchify/flatten，再映射到同一嵌入维度 $D$
   - 输出：几何 token 序列  
     $T_{\text{geo}} \in \mathbb{R}^{N_2 \times D}$

7. **Token 融合（图中两个“tokens”一起送入 Transformer）**
   - 操作：将两种 token 合并成一个序列，常见做法是 concat：  
     $T = [T_{\text{img}}; T_{\text{geo}}] \in \mathbb{R}^{(N_1+N_2)\times D}$
   - 建议同时加入：
     - 位置编码：$E_{\text{pos}}$
     - 模态/类型编码（区分 img / geo）：$E_{\text{type}}$
   - 得到最终输入 token：  
     $T_0 = T + E_{\text{pos}} + E_{\text{type}}$

8. **Transformer（图中写 use ViT）**
   - 输入：$T_0$
   - 操作：用 ViT-style Transformer encoder 对 token 做多层自注意力建模，让外观与几何先验相互对齐、互相补充
   - 输出：上下文化后的 token：  
     $T_L \in \mathbb{R}^{(N_1+N_2)\times D}$

9. **回归头（输出 MANO + camera translation）**
   - 输入：$T_L$（可用 [CLS] token、mean pooling、或取部分 token 聚合）
   - 输出（图里明确给了两类）：
     1) **MANO 参数**（手部参数化网格模型参数，至少包含 pose 与 shape；也可能包含 global rotation/scale 等，依你实现）  
     2) **camera translation**（相机平移向量）$t \in \mathbb{R}^3$

10. **（隐含的最终重建）用 MANO 层生成网格/关节**
   - 操作：将预测的 MANO 参数送入 MANO layer
   - 输出：3D 手网格顶点 $V \in \mathbb{R}^{M \times 3}$、3D 关节 $J \in \mathbb{R}^{K \times 3}$  
   - 配合相机平移 $t$（以及你设定的相机模型）可投影得到 2D 关键点/轮廓用于监督或可视化

### 要求（必须执行）：
1. 检测模型使用WiLoR训练好的手部检测器，权重路径为：/data0/users/Robert/linweiquan/WiLoR/pretrained_models/detector.pt，调研代码实例为：/data0/users/Robert/linweiquan/WiLoR/demo.py
2. ViT和tokenizer1使用与WiLoR相同的backbone（ViT-Large），并使用WiLoR的预训练权重进行初始化，权重路径为/data0/users/Robert/linweiquan/WiLoR/pretrained_models/wilor_final.ckpt
3. 使用MoGe2预训练模型（hugging face模型为Ruicheng/moge-2-vitl-normal）MoGe2代码路径为：/data0/users/Robert/linweiquan/MoGe.
4. 几何先验的target features使用MoGe2中的Conv neck输出。
5. 几何先验模型（这里使用MoGe2）不参与学习更新（需要冻结参数）
6. 参考/data0/users/Robert/linweiquan/UniHandFormer/dataloader中的dex_ycb_dataset.py、freihand_dataset.py和ho3d_dataset.py代码实现GPGFormer三个手部数据集的数据加载器。
7. GPGFormer的监督信号为3D mesh、3D joints、2D keypoints，参考/data0/users/Robert/linweiquan/UniHandFormer/src/losses/loss.py相关代码实现。
8. 涉及到的权重参数全部拷贝，存放到/data0/users/Robert/linweiquan/GPGFormer中合适的文件夹。
9. 严禁修改除了/data0/users/Robert/linweiquan/GPGFormer之外，其它文件的代码
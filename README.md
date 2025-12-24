# GPGFormer

GPGFormer 是一个双分支手部重建框架：  
- **分支 A（Hand Appearance）**：对 RGB 手部裁剪 patch 做 ViT patch embedding，提取局部纹理与轮廓细节。  
- **分支 B（Geometry Prior）**：用 MoGe-2 在整图上估计几何特征，再用同一 bbox 做 ROI 裁剪，对齐手部几何先验。  
两路 token 拼接后进入同一个 Transformer 编码器，输出 **MANO 参数** 与 **camera translation**。

## 核心流程
1. 输入 RGB。  
2. 使用 WiLoR detector 生成手部 bbox（或直接提供 bbox）。  
3. bbox 同时用于：  
   - RGB 手部 patch crop（外观分支）  
   - MoGe-2 几何特征图上的 ROI crop（几何先验分支）  
4. 两路 token concat → Transformer → 输出：  
   - MANO pose/shape  
   - camera translation (tx, ty, tz)

## 目录结构
- `config/` 配置文件  
- `src/` 模型与工具  
- `train.py` 训练  
- `metrics/evaluate.py` 测试  
- `visualization/visualize_model.py` 可视化  

## 依赖
- Python 3.8+  
- PyTorch  
- timm  
- smplx  
- ultralytics  
- opencv-python  
- moge (MoGe-2)  

## 权重与模型路径
1. **WiLoR detector**  
   - 路径：`/data0/users/Robert/linweiquan/GPGFormer/pretrained_model/detection/detector.pt`  
   - 用法参考：`/data0/users/Robert/linweiquan/WiLoR/demo.py`
2. **WiLoR ViT 预训练权重**  
   - 路径：`/data0/users/Robert/linweiquan/GPGFormer/pretrained_model/backbone/wilor_final.ckpt`  
3. **MoGe-2 模型**  
   - 默认从 Hugging Face 拉取：`Ruicheng/moge-2-vitl-normal`  
   - 如无外网，请下载后改为本地路径

## 训练
```bash
cd /data0/users/Robert/linweiquan/GPGFormer
python train.py --config config/config_xxx.yaml
```

## 多卡分布式训练
```bash
cd /data0/users/Robert/linweiquan/GPGFormer
torchrun --nproc_per_node=4 train.py --config config/config_xxx.yaml
```

## 指定GPU卡号
单卡示例：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/config_xxx.yaml
```
多卡示例（使用 0,1,2,3 号卡）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config config/config_xxx.yaml
```

## 测试
```bash
python metrics/evaluate.py --config config/config.yaml --checkpoint checkpoints/gpgformer_epoch_0.pt
```

## 可视化
```bash
python visualization/visualize_model.py --config config/config.yaml --checkpoint checkpoints/gpgformer_epoch_0.pt --out_dir vis_out
```

## 数据格式
默认支持两种数据来源：  
- **image_folder**：只需要图像文件，适合快速测试（无监督 loss 会被跳过）。  
- **custom_json**：提供 bbox 和 MANO/相机 GT。示例格式：
```json
[
  {
    "image_path": "/path/to/image.jpg",
    "bbox": [x1, y1, x2, y2],
    "mano_pose": [..],
    "mano_shape": [..],
    "cam_t": [tx, ty, tz]
  }
]
```

## 说明
- **MoGe-2 特征**来自 Conv neck 输出（`MoGeModel.neck`），再用 bbox 做 ROI crop。  
- **ViT 初始化**会从 WiLoR 权重中加载 `patch_embed` 与 `blocks`。  
- 若数据集中已有 bbox，可将 `dataset.use_detector=false` 并直接在 JSON 中提供 bbox。  
- 为了 batch 训练稳定，`dataset.input_size` 会将输入统一 resize。  

# 数据集对齐计划

## 目标
将 dex_ycb_dataset.py 和 ho3d_dataset.py 与 freihand_dataset_v2.py 对齐

## 对齐要点

### 1. 数据增强逻辑
- ✅ 使用 `get_example()` 函数进行 WiLoR-style 增强
- ✅ 添加中心抖动（center jitter）
- ✅ 添加亮度/对比度调整（可选）

### 2. 返回数据格式
```python
return {
    'rgb': imgRGB,  # torch.Tensor, ImageNet normalized
    'keypoints_2d': torch.Tensor,  # (21, 2)
    'keypoints_3d': torch.Tensor,  # (21, 3)
    'mano_params': dict,  # {'global_orient', 'hand_pose', 'betas'}
    'cam_param': torch.Tensor,  # (4,) [fx, fy, cx, cy]
    'box_center': torch.Tensor,  # (2,)
    'box_size': torch.Tensor,  # scalar
    'bbox_expand_factor': torch.Tensor,  # scalar
    '_scale': torch.Tensor,  # (2,)
    'mano_params_is_axis_angle': dict,
    'xyz_valid': int,  # 0 or 1
    'uv_valid': np.ndarray,  # (21,)
    'hand_type': str,  # 'left' or 'right'
    'is_right': float  # 0.0 or 1.0
}
```

### 3. 需要保留的数据集特有逻辑
- DexYCB: 左右手处理、DexYCB2MANO 映射
- HO3D: 特定的数据加载方式

## 修改步骤

### Phase 1: dex_ycb_dataset.py
1. 添加新的初始化参数
2. 修改 __getitem__ 使用 get_example()
3. 添加中心抖动和亮度/对比度调整
4. 统一返回格式
5. 添加 main 函数验证

### Phase 2: ho3d_dataset.py
重复 Phase 1 的步骤


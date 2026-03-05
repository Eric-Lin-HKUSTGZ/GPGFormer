import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

# ==========================================
# 第一部分：模拟手部重建模型预测顶点
# ==========================================

class DummyHandReconstructor(nn.Module):
    """
    模拟一个手部重建网络
    实际场景中，这里可能是 MANO Layer 或 Graph-CNN
    """
    def __init__(self):
        super().__init__()
        # 假设我们预测 778 个顶点 (MANO 模型的顶点数)
        self.num_vertices = 778
        
        # 在实际应用中，这里会有复杂的神经网络层
        # 这里我们随机初始化一个“手部”形状用于演示
        # 真实场景下输入是图像特征，输出是 vertices
        self.fc = nn.Linear(512, self.num_vertices * 3)
        
        # 加载预定义的面片
        # 注意：在手部重建中，faces 通常是固定的 (Fixed Topology)
        # 这里为了演示，我们生成一个简单的球体拓扑代替手部拓扑
        # 实际使用时请替换为 MANO 的 .obj 文件中的 faces
        self.faces = self._create_dummy_faces()

    def _create_dummy_faces(self):
        """生成虚拟的面片连接关系 (模拟 Mesh 拓扑)"""
        # 在真实项目中，通常这样做：
        # import trimesh; obj = trimesh.load('mano_hand.obj'); faces = torch.tensor(obj.faces, dtype=torch.long)
        # 这里生成一个简单的网格球体作为占位符
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        faces = []
        # 简化的三角面片生成逻辑 (仅为演示 Mesh 构造)
        # 这里的逻辑不对应真实手部，只是为了代码能跑通
        for i in range(len(phi)-1):
            for j in range(len(theta)-1):
                p1 = i * len(theta) + j
                p2 = p1 + 1
                p3 = p1 + len(theta)
                p4 = p3 + 1
                faces.append([p1, p3, p2])
                faces.append([p2, p3, p4])
        return torch.tensor(faces, dtype=torch.long)

    def forward(self, x):
        """
        输入: 图像特征
        输出: 预测的顶点坐标 (Batch, N, 3)
        """
        # 模拟预测过程
        verts = self.fc(x)
        verts = verts.view(-1, self.num_vertices, 3) # [Batch, 778, 3]
        return verts, self.faces.unsqueeze(0).repeat(x.shape[0], 1, 1)


# ==========================================
# 第二部分：构建可微渲染器
# ==========================================

def get_renderer(device, image_size=256):
    """
    初始化 PyTorch3D 渲染器
    这是目前学术界将预测 Mesh 渲染回图像空间的标准做法
    """
    # 初始化相机位置：从 Z 轴上方往下看
    R, T = look_at_view_transform(2.0, 0.0, 0.0) 
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    
    # 光照设置
    lights = PointLights(
        device=device, 
        location=((0.0, 0.0, -2.0),), # 光源位置
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
    )
    
    # 组合渲染器
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    return renderer.to(device)


# ==========================================
# 第三部分：完整执行流程
# ==========================================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型
    model = DummyHandReconstructor().to(device)
    
    # 2. 模拟输入 (Batch=2)
    dummy_input = torch.randn(2, 512).to(device)
    
    # 3. 获取预测顶点和面片
    # pred_verts: [2, 778, 3]
    # faces: [2, M, 3] (M 是面片数量)
    pred_verts, faces = model(dummy_input)
    
    # 4. 定义顶点颜色 (模拟手部皮肤颜色)
    # PyTorch3D 需要 Textures 对象
    # 这里我们将所有顶点设置为淡粉色
    verts_rgb = torch.ones_like(pred_verts) * 0.7 # [Batch, 778, 3]
    verts_rgb[:, :, 0] = 1.0  # R
    verts_rgb[:, :, 1] = 0.7  # G
    verts_rgb[:, :, 2] = 0.6  # B
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    
    # 5. 构建 Mesh 对象
    # 这是将顶点转为 Mesh 的关键步骤
    hand_mesh = Meshes(
        verts=pred_verts,
        faces=faces,
        textures=textures
    )
    
    # 6. 渲染
    renderer = get_renderer(device)
    images = renderer(hand_mesh)
    
    # 7. 可视化结果
    # images 是 [Batch, H, W, 4] (RGBA)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Rendered Hand Mesh 1")
    plt.imshow(images[0, ..., :3].cpu().detach().numpy())
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Rendered Hand Mesh 2")
    plt.imshow(images[1, ..., :3].cpu().detach().numpy())
    plt.axis("off")
    
    plt.show()
    
    # 8. (可选) 导出为 .obj 文件以便查看
    # 使用 trimesh 或 pytorch3d 的 IO 接口
    from pytorch3d.io import save_obj
    save_obj("predicted_hand.obj", pred_verts[0], faces[0])
    print("Mesh 已保存为 predicted_hand.obj")

if __name__ == "__main__":
    main()
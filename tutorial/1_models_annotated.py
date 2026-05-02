# =============================================================================
# 1_models_annotated.py — MLP 与 SimpleCNN 模型定义（带详细注释）
# =============================================================================
# 这是 digit16-cnn-experiment/src/models.py 的教学注释版。
# 对照阅读顺序：先读 MLP，再读 SimpleCNN，最后看 get_model。
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# MLP — 多层感知机 Baseline
# ═════════════════════════════════════════════════════════════════════════════
#
# MLP = Multi-Layer Perceptron，也叫前馈神经网络 (Feedforward NN)。
# 它是最基本的神经网络结构：每一层都与上一层的所有神经元相连。
#
# 在这个项目中的结构：
#   输入 256 维 (16×16 展平)
#     ↓ FC(256→128) + ReLU + Dropout(0.2)
#   128 维
#     ↓ FC(128→64)  + ReLU + Dropout(0.2)
#   64 维
#     ↓ FC(64→10)
#   10 维输出 (0-9 各类别的 logits)
#
# 参数量计算：
#   FC1: 256×128 + 128(bias) = 32,896
#   FC2: 128×64  + 64(bias)  =  8,256
#   FC3: 64×10   + 10(bias)  =    650
#   总计:                      41,802
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    参数:
        input_dim:  输入维度，默认 256 (16×16 展平)
        num_classes: 输出类别数，默认 10 (数字 0-9)
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 10):
        # super().__init__() 是 PyTorch 要求的——调用父类 nn.Module 的构造函数。
        # 这一行让 PyTorch 知道这个类的实例是一个神经网络模块，
        # 它会自动追踪所有 nn.Linear, nn.Dropout 等子模块的参数。
        super().__init__()

        # nn.Linear(in_features, out_features)
        # 定义: y = x @ W^T + b
        # - W 的形状: [out_features, in_features]
        # - b 的形状: [out_features]
        # PyTorch 自动初始化 W 和 b（默认用 Kaiming uniform）
        self.fc1 = nn.Linear(input_dim, 128)   # 256 → 128
        self.fc2 = nn.Linear(128, 64)           # 128 → 64
        self.fc3 = nn.Linear(64, num_classes)   # 64 → 10

        # nn.Dropout(p)
        # 训练时随机将 p 比例的神经元输出置 0。
        # 作用：防止过拟合。相当于每次前向传播训练一个不同的"子网络"，
        # 多个子网络取平均的结果更泛化。
        # 评估模式 (model.eval()) 时 Dropout 自动关闭。
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：定义数据从输入到输出的路径。

        PyTorch 的约定：你用 forward 定义计算逻辑，
        PyTorch 自动在反向传播时计算梯度（通过 autograd 引擎）。

        输入 x: [batch_size, 1, 16, 16]
        输出:   [batch_size, 10]
        """

        # 第一步：把 2D 图像展平成 1D 向量
        # x.view(x.size(0), -1) 的含义:
        #   x.size(0) = batch_size（保持 batch 维度不变）
        #   -1 = 自动计算剩下的维度大小
        #   对于 [B, 1, 16, 16] → [B, 256]
        # 为什么 MLP 要展平而 CNN 不需要？因为 MLP 的 Linear 层只能处理 1D 输入。
        # CNN 的 Conv2d 可以直接处理 2D (甚至 3D) 输入，保留了空间结构。
        x = x.view(x.size(0), -1)  # [B, 256]

        # 第二步：第一个隐藏层
        # fc1: 线性变换 256 → 128
        # F.relu: 逐元素 max(0, x)。非线性激活函数。
        #   没有 ReLU，多层线性变换 = 一层线性变换（线性叠线性还是线性）。
        #   ReLU 的导数：x>0 时导数为 1，x≤0 时导数为 0。
        #   梯度为 0 的神经元不再更新 = "死亡 ReLU"，所以有 ReLU 变体 (LeakyReLU 等)。
        # dropout: 随机丢弃 20% 的神经元
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 第三步：第二个隐藏层
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 第四步：输出层（不做 softmax）
        # 输出 10 个 logits（原始分数），不在这做 softmax。
        # 原因：CrossEntropyLoss 内部自动包含 softmax，
        # 如果在 forward 里做了 softmax，再喂给 CrossEntropyLoss 会 double-softmax。
        x = self.fc3(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# SimpleCNN — 小型卷积神经网络
# ═════════════════════════════════════════════════════════════════════════════
#
# CNN 和 MLP 的核心区别：
#   1. 卷积 (Conv): 神经元只连到局部区域（局部感受野），不是全连接
#   2. 权值共享: 同一个卷积核在整个图像上滑动，参数被所有位置共享
#   3. 池化 (Pool): 降采样，提供平移不变性
#
# 结构:
#   输入 [B, 1, 16, 16]
#     ↓ Conv1: 1→8, 3×3, pad=1  → [B, 8, 16, 16]
#     ↓ ReLU
#     ↓ MaxPool(2)                 → [B, 8, 8, 8]
#     ↓ Conv2: 8→16, 3×3, pad=1  → [B, 16, 8, 8]
#     ↓ ReLU
#     ↓ MaxPool(2)                 → [B, 16, 4, 4]
#     ↓ Flatten                    → [B, 256]
#     ↓ FC(256→64) + ReLU + Dropout(0.25)
#     ↓ FC(64→10)                  → [B, 10]
#
# 参数量计算（关键——理解"权值共享省了多少参数"）:
#   Conv1: 1×3×3×8 + 8(bias)   =    80
#   Conv2: 8×3×3×16 + 16(bias) = 1,168
#   FC1:   256×64 + 64(bias)    = 16,448
#   FC2:   64×10 + 10(bias)     =    650
#   总计:                         18,346
#
#   如果 Conv1 不用权值共享（即全连接）:
#   256×8 + 8 = 2,056（每个位置都需要独立参数），大了 25 倍！
# ═════════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Conv1: 1 通道 → 8 通道，3×3 卷积核 ──
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding, ...)
        # - in_channels=1:  输入是灰度图（1 个通道）
        # - out_channels=8: 输出 8 个特征图（8 个不同的卷积核）
        # - kernel_size=3:  3×3 的卷积窗口
        # - padding=1:      输入边缘补一圈 0
        #
        # 有 padding vs 无 padding:
        #   无 padding: 16×16 → (16-3+1)×(16-3+1) = 14×14（尺寸缩小）
        #   有 padding=1: 16×16 → 16×16（尺寸不变，因为边缘补了 0）
        #
        # 为什么用 3×3 而不是 5×5 或 7×7？
        #   - 3×3 是最小的能捕捉中心/上下/左右关系的核
        #   - 两个 3×3 的感受野 ≈ 一个 5×5，但参数更少（18 vs 25）
        #   - VGG 论文论证了"小核堆叠 > 大核单层"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)

        # ── Conv2: 8 通道 → 16 通道 ──
        # in_channels=8 意味着每个输出通道的卷积核实际上是一个 3×3×8 的体。
        # 它同时在 8 个输入通道上做卷积然后求和。
        # 这 8 个输入通道是 Conv1 学到的 8 种"低级特征"（边缘、纹路等），
        # Conv2 的任务是在这些特征上寻找组合模式（拐角、交叉等）。
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # ── MaxPool: 2×2 窗口，步长 2 ──
        # kernel_size=2, stride=2:
        #   每 2×2 的窗口取最大值，窗口之间不重叠（步长=2）。
        #   效果：16×16 → 8×8（每个维度减半）
        #
        # 为什么用 MaxPool 而不是 AvgPool？
        #   MaxPool: 保留最强信号。"这里有一个很强的边缘"
        #   AvgPool: 平均化。"这里总体偏亮"
        #   对于分类任务，MaxPool 通常更好——我们关心"有没有特征"，而不是"平均亮度"。
        self.pool = nn.MaxPool2d(2, 2)

        # ── 全连接层 ──
        # Conv2 + MaxPool 后，特征图尺寸是 [B, 16, 4, 4]
        # 展平 → 16×4×4 = 256 维
        self.fc1 = nn.Linear(16 * 4 * 4, 64)   # 256 → 64
        self.fc2 = nn.Linear(64, num_classes)    # 64 → 10

        # Dropout(0.25) —— CNN 通常用比 MLP 更低的 dropout 率
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。输入 [B, 1, 16, 16]，输出 [B, 10]。

        跟踪形状变化（以 batch_size=64 为例）:
          x = [64, 1, 16, 16]           输入
          x = [64, 8, 8, 8]             Conv1 → ReLU → Pool
          x = [64, 16, 4, 4]            Conv2 → ReLU → Pool
          x = [64, 256]                  Flatten
          x = [64, 64]                  FC1 → ReLU → Dropout
          x = [64, 10]                  FC2 → 输出
        """

        # 第一步：第一个卷积块
        # conv1: [B, 1, 16, 16] → [B, 8, 16, 16]
        #   padding=1 保持了尺寸不变
        # ReLU: 负值 → 0
        # pool:  [B, 8, 16, 16] → [B, 8, 8, 8]
        x = self.pool(F.relu(self.conv1(x)))

        # 第二步：第二个卷积块
        # conv2: [B, 8, 8, 8] → [B, 16, 8, 8]
        # pool:  [B, 16, 8, 8] → [B, 16, 4, 4]
        x = self.pool(F.relu(self.conv2(x)))

        # 第三步：展平
        # 把 [B, 16, 4, 4] 变成 [B, 256]
        # x.size(0) = batch_size, -1 = 自动计算 = 16×4×4 = 256
        x = x.view(x.size(0), -1)

        # 第四步：全连接分类头
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# get_model — 模型工厂函数
# ═════════════════════════════════════════════════════════════════════════════

def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    根据名称创建模型。

    用法:
        mlp = get_model("mlp")
        cnn = get_model("simple_cnn")

    工厂函数的好处:
        在 run_experiment.py 里只需要一行代码就能切换模型，
        不需要到处 import 和修改代码。
    """
    models = {
        "mlp": MLP,
        "simple_cnn": SimpleCNN,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(models.keys())}."
        )
    return models[model_name](**kwargs)

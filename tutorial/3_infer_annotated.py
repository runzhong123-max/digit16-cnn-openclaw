# =============================================================================
# 3_infer_annotated.py — 模型推理（带详细注释）
# =============================================================================
# 这是 digit16-cnn-experiment/src/infer.py 的教学注释版。
# 解释：训练好的模型如何被加载和使用，为什么要用和训练时相同的预处理。
# =============================================================================

import numpy as np
import torch


# ═════════════════════════════════════════════════════════════════════════════
# 归一化常量 — 必须和训练时一模一样！
# ═════════════════════════════════════════════════════════════════════════════
#
# 这两个数字是整个 MNIST 训练集的全局统计值：
#   像素均值 (mean)     = 0.1307
#   像素标准差 (std)    = 0.3081
#
# 为什么推理时必须用相同的值？
# → 模型在训练时见到的是归一化后的数据：输入 ≈ N(0, 1)
#   如果推理时不做归一化，输入 ≈ [0, 1]，分布完全不同。
#   模型学到的卷积核权重是针对 N(0,1) 调整好的，
#   换一种分布输入，"最优权重"就变成了"随便调的权重"。
#
# 类比：你学会了在米制单位下投篮（篮筐在 3.05m），
#       如果突然让你用英制数字（10 英尺 = 3.05m 但数字不同），
#       你就"不认识"了。
# ═════════════════════════════════════════════════════════════════════════════

_MNIST_MEAN = 0.1307
_MNIST_STD  = 0.3081


# ═════════════════════════════════════════════════════════════════════════════
# DigitInferer — 模型推理器
# ═════════════════════════════════════════════════════════════════════════════
#
# 职责：
#   1. 从 .pt 文件加载训练好的模型权重
#   2. 接收 16×16 numpy 数组（用户画的手写数字）
#   3. 做和训练时完全一样的预处理（归一化）
#   4. 前向传播得到 10 类 softmax 概率
#
# 使用示例:
#   inferer = DigitInferer("results/.../simple_cnn_best.pt")
#   probs = inferer.predict(grid_16x16)  # grid: np.array, shape (16,16), 值 ∈ [0,1]
#   digit, conf = inferer.predict_top(grid_16x16)
# ═════════════════════════════════════════════════════════════════════════════

class DigitInferer:

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: .pt 权重文件的路径
            device: "cpu" 或 "cuda"
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model_path = model_path

    def _load_model(self, model_path: str):
        """
        从磁盘加载模型，三步：
          1. 创建一个空的 SimpleCNN 实例（随机权重）
          2. 从 .pt 文件读取训练好的权重
          3. 把训练好的权重"覆盖"到空模型上
          4. 切换到 eval 模式（关闭 Dropout）
        """
        # 延迟导入——这样 demo 脚本不需要关心 src/ 的目录结构
        from src.models import SimpleCNN

        model = SimpleCNN(num_classes=10)

        # torch.load 读取保存的 state_dict（一个字典，key=参数名，value=张量）
        # weights_only=True：PyTorch 2.6+ 的安全参数，
        #   只加载权重张量，拒绝执行任意 Python 代码（防止恶意 .pt 文件攻击）
        # map_location=device：如果权重是在 GPU 上保存的，强制加载到 CPU
        state = torch.load(model_path, map_location=self.device, weights_only=True)

        # model.load_state_dict(state)
        # 把 state 里的每个参数（conv1.weight, fc1.bias, ...）
        # 按名字匹配，覆盖到 model 的对应层上。
        # 如果 state 里的参数名和 model 不匹配（比如模型结构不同），
        # 会抛 RuntimeError 并提示具体哪个参数不匹配。
        model.load_state_dict(state)

        model.to(self.device)

        # model.eval() — 评估模式
        # 关闭 Dropout（不再随机丢弃神经元）
        # 如果有 BatchNorm，会使用训练时累积的全局均值和方差
        model.eval()

        return model

    def predict(self, grid: np.ndarray) -> np.ndarray:
        """
        核心推理函数。

        输入: grid — (16, 16) numpy 数组，值 ∈ [0.0, 1.0]
              1.0 = 用户画过的格子
              0.0 = 空白格子

        输出: probs — (10,) numpy 数组，10 个类别的 softmax 概率
        """

        # ── 输入检查 ──
        if grid.shape != (16, 16):
            raise ValueError(f"Expected grid shape (16, 16), got {grid.shape}")
        if grid.min() < 0.0 or grid.max() > 1.0:
            raise ValueError("Grid values must be in [0.0, 1.0]")

        # ── 预处理：numpy → tensor ──
        # 用户画的 [16, 16] → [1, 1, 16, 16]
        #   unsqueeze(0) 加 batch 维度
        #   再 unsqueeze(0) 加 channel 维度（灰度图 = 1 通道）
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # ── 归一化：和训练数据完全一致 ──
        # 训练时的 transforms.Normalize((0.1307), (0.3081)) 等价于:
        #   tensor ← (tensor - 0.1307) / 0.3081
        #
        # 为什么要除以 std（标准差）？
        #   标准化 = (x - mean)/std，使数据变成均值 0、标准差 1 的分布。
        #   如果不除以 std，只用 (x - mean)，数据的方差还是原来的，
        #   而不同的输入维度可能有不同的方差，导致梯度更新步长不一致。
        tensor = (tensor - _MNIST_MEAN) / _MNIST_STD

        tensor = tensor.to(self.device)

        # ── 前向传播 ──
        # torch.no_grad() — 告诉 PyTorch 不要构建计算图
        # 推理不需要反向传播，关闭 autograd 可以:
        #   1. 节省内存（不存储中间结果用于梯度计算）
        #   2. 加速（跳过 autograd 引擎的开销）
        with torch.no_grad():
            logits = self.model(tensor)           # [1, 10] — 原始分数
            probs = torch.softmax(logits, dim=1)  # [1, 10] — 概率

        # 去掉 batch 维度 → (10,) numpy 数组
        return probs.cpu().numpy().squeeze(0)

    def predict_top(self, grid: np.ndarray) -> tuple[int, float]:
        """
        返回最可能的数字和置信度。

        用法: digit, confidence = inferer.predict_top(grid)
        """
        probs = self.predict(grid)
        top_idx = int(np.argmax(probs))  # 最大概率的索引
        return top_idx, float(probs[top_idx])

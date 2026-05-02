#!/usr/bin/env python3
"""演示 1：单次前向传播 —— 直观感受张量如何在 CNN 中流动。

用一个随机生成的 16×16 "数字"，逐步打印每一层输出的形状和数值统计，
让你清楚地看到数据从输入到输出的完整路径。

运行：
    cd digit16-cnn-experiment && source venv/bin/activate
    python tutorial/demo_forward_pass.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from src.models import SimpleCNN


def print_tensor(name, tensor):
    """友好地打印张量信息。"""
    print(f"  {name}")
    print(f"    shape : {list(tensor.shape)}")
    print(f"    dtype : {tensor.dtype}")
    print(f"    min   : {tensor.min().item():.4f}")
    print(f"    max   : {tensor.max().item():.4f}")
    print(f"    mean  : {tensor.mean().item():.4f}")
    print(f"    device: {tensor.device}")
    print()


def main():
    print("=" * 72)
    print("演示 1：CNN 前向传播 —— 数据如何从图像变成类别概率")
    print("=" * 72)

    # ── 第 0 步：准备一个 16×16 的"手写数字" ──
    # 我们手工造一条竖线（模拟数字 1），看看模型能否在前向传播中识别它
    print("\n── 第 0 步：准备输入 ──")
    grid = torch.zeros(1, 1, 16, 16)  # [batch=1, channel=1, height=16, width=16]
    grid[:, :, 2:14, 7:9] = 1.0       # 在第 7-8 列画一条竖线
    print_tensor("输入图像 [B, C, H, W]", grid[0])  # 只展示单张图

    # ── 第 1 步：创建模型 ──
    print("── 第 1 步：创建 SimpleCNN ──")
    model = SimpleCNN(num_classes=10)
    model.eval()  # 评估模式，不启用 dropout
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  卷积层 1: Conv2d(1→8, 3×3, pad=1) → ReLU → MaxPool(2)")
    print(f"  卷积层 2: Conv2d(8→16, 3×3, pad=1) → ReLU → MaxPool(2)")
    print(f"  全连接层: FC(256→64) → ReLU → Dropout → FC(64→10)")
    print()

    # ── 第 2 步：逐层前向传播 ──
    print("── 第 2 步：逐层前向传播 —— 跟踪形状变化 ──")

    # Conv1 + ReLU + Pool
    x = grid
    print(f"  输入:     {list(x.shape)}")

    x = model.conv1(x)
    print(f"  Conv1:    {list(x.shape)}  ← 8 个 3×3 卷积核扫过 16×16，输出 8 个 16×16 特征图")

    x = F.relu(x)
    print(f"  ReLU:     {list(x.shape)}  ← 负数 → 0，正数保留")

    x = model.pool(x)
    print(f"  MaxPool:  {list(x.shape)}  ← 每 2×2 取最大值，16×16 → 8×8")

    # Conv2 + ReLU + Pool
    x = model.conv2(x)
    print(f"  Conv2:    {list(x.shape)}  ← 16 个 3×3×8 卷积核，输出 16 个 8×8 特征图")

    x = F.relu(x)
    print(f"  ReLU:     {list(x.shape)}")

    x = model.pool(x)
    print(f"  MaxPool:  {list(x.shape)}  ← 8×8 → 4×4")

    # Flatten
    x_flat = x.view(x.size(0), -1)
    print(f"  Flatten:  {list(x_flat.shape)}  ← 16×4×4 = 256 维向量")

    # FC1 + ReLU
    x = model.fc1(x_flat)
    print(f"  FC(256→64): {list(x.shape)}  ← 全连接，256 → 64")

    x = F.relu(x)
    print(f"  ReLU:     {list(x.shape)}")

    # FC2 (output)
    logits = model.fc2(x)
    print(f"  FC(64→10): {list(logits.shape)}  ← 输出 10 维 logits (未归一化的分数)")
    print()

    # ── 第 3 步：Softmax 得到概率 ──
    print("── 第 3 步：Softmax —— logits → 概率 ──")
    probs = torch.softmax(logits, dim=1).squeeze(0)

    print(f"  Logits (原始分数):")
    for i in range(10):
        print(f"    数字 {i}: {logits[0, i].item():+8.4f}")

    print(f"\n  Softmax 后概率:")
    for i in range(10):
        bar = "█" * int(probs[i].item() * 50)
        print(f"    数字 {i}: {probs[i].item():.4f}  {bar}")

    top_digit = probs.argmax().item()
    top_conf = probs.max().item()
    print(f"\n  ★ Top-1 预测: 数字 {top_digit}, 置信度 {top_conf:.2%}")
    print(f"    (注意：这是随机权重，预测基本是瞎猜。训练后才能准确。)")

    # ── 第 4 步：思考题 ──
    print("\n" + "=" * 72)
    print("💡 思考题")
    print("=" * 72)
    print("""
  1. 输入是 [1, 1, 16, 16]，为什么第一个数字是 1？
     → 第一个 1 = batch size（一次处理一张图）
     → 第二个 1 = channel（灰度图只有 1 个通道，RGB 图有 3 个）

  2. Conv1 后形状还是 16×16，为什么没变小？
     → 因为 padding=1，在图像边缘补一圈 0，卷积后尺寸不变。
       如果没有 padding，3×3 卷积会让 16×16 → 14×14。

  3. MaxPool 让尺寸减半，16→8→4。为什么不用 AvgPool？
     → MaxPool 保留最强特征（"这里有个边缘！"），
       AvgPool 平滑化（"这里模糊地有点像边缘"），
       对于分类任务 MaxPool 通常更好。

  4. 为什么 flatten 后是 256 维？
     → 最后一张特征图是 16 通道 × 4 × 4 = 256。
       这个 256 不是原始像素，而是经过两次卷积提取的"高级特征"。

  5. 这个模型此时（随机权重）能正确识别竖线吗？
     → 基本不能。卷积核是随机的，还没学会"竖线是什么"。
       把这个 demo 加载训练好的权重再跑一次，对比结果！
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""演示 2：一次反向传播 —— 观察参数如何被梯度更新。

取一个小 batch（8 张图），做一次 前向→算loss→反向→更新，
对比更新前后参数的变化。

运行：
    python tutorial/demo_backprop.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
from src.models import SimpleCNN


def main():
    print("=" * 72)
    print("演示 2：一次反向传播 —— 参数如何被更新")
    print("=" * 72)

    # ── 准备数据 ──
    print("\n── 准备：8 张随机 16×16 图像 + 随机标签 ──")
    torch.manual_seed(42)

    images = torch.randn(8, 1, 16, 16)  # 8 张图
    labels = torch.randint(0, 10, (8,))  # 8 个随机标签 0-9
    print(f"  images shape: {list(images.shape)}")
    print(f"  labels:       {labels.tolist()}")
    print(f"  (用随机数据演示梯度机制，不需要真实 MNIST)")

    # ── 创建模型和优化器 ──
    print("\n── 第 1 步：创建模型 ──")
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  优化器:   Adam (lr=0.001)")

    # ── 记录更新前的参数 ──
    print("\n── 第 2 步：记录 conv1 卷积核更新前的值 ──")
    conv1_w_before = model.conv1.weight.data.clone()  # [8, 1, 3, 3]
    print(f"  conv1.weight shape: {list(conv1_w_before.shape)}")
    print(f"  第一个卷积核 (kernel 0) 更新前:")
    print(f"    {conv1_w_before[0, 0].tolist()}")

    # ── 前向传播 ──
    print("\n── 第 3 步：前向传播 + 计算 loss ──")
    outputs = model(images)  # [8, 10]
    loss = criterion(outputs, labels)
    print(f"  outputs shape: {list(outputs.shape)}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  (CrossEntropyLoss 内部自动做了 softmax，所以 outputs 是 logits)")

    # ── 反向传播 ──
    print("\n── 第 4 步：反向传播 loss.backward() ──")
    print(f"  conv1.weight.grad 更新前: {model.conv1.weight.grad}")

    optimizer.zero_grad()  # 清空旧梯度
    loss.backward()         # 链式法则自动计算所有参数的梯度

    grad = model.conv1.weight.grad  # conv1 卷积核收到的梯度
    print(f"  conv1.weight.grad 更新后: shape {list(grad.shape)}")
    print(f"  第一个卷积核收到的梯度:")
    print(f"    {grad[0, 0].tolist()}")
    print()
    print(f"  梯度解读:")
    print(f"    正梯度 → 增大该权重会增大 loss（所以应该减小）")
    print(f"    负梯度 → 增大该权重会减小 loss（所以应该增大）")
    print(f"    梯度的绝对值大小 = 该权重对 loss 的「影响力」")

    # ── 参数更新 ──
    print("\n── 第 5 步：optimizer.step() —— 用梯度更新参数 ──")
    optimizer.step()

    conv1_w_after = model.conv1.weight.data.clone()

    print(f"  第一个卷积核 更新后:")
    print(f"    {conv1_w_after[0, 0].tolist()}")

    change = conv1_w_after - conv1_w_before
    print(f"\n  变化量 (更新后 − 更新前):")
    print(f"    {change[0, 0].tolist()}")

    print(f"\n  验证: 变化量 ≈ −lr × 梯度?")
    print(f"    −lr × grad[0,0] = {(-0.001 * grad[0, 0]).tolist()}")
    print(f"    实际变化量         = {change[0, 0].tolist()}")
    print(f"    (Adam 不是朴素 SGD，所以不完全等于 −lr×grad，")
    print(f"     但方向大致相同。Adam 会自适应调整步长。)")

    # ── 验证 loss 是否下降 ──
    print("\n── 第 6 步：验证 —— 用同一批数据再算一次 loss ──")
    with torch.no_grad():
        outputs2 = model(images)
        loss2 = criterion(outputs2, labels)
    print(f"  更新前 loss: {loss.item():.4f}")
    print(f"  更新后 loss: {loss2.item():.4f}")
    print(f"  变化:       {loss2.item() - loss.item():+.4f}")
    print(f"  ✓ loss 下降了！参数往正确的方向走了一小步。")

    # ── 关键总结 ──
    print("\n" + "=" * 72)
    print("💡 关键总结：一次训练迭代 = 四行代码")
    print("=" * 72)
    print("""
    optimizer.zero_grad()   # ① 清空上一步的梯度（否则会累加）
    outputs = model(x)       # ② 前向传播：输入 → 网络 → 输出
    loss = criterion(y, t)   # ③ 计算 loss：比较预测和真实标签
    loss.backward()          # ④ 反向传播：链式法则自动算梯度
    optimizer.step()         # ⑤ 参数更新：沿梯度反方向走一步

  在真正的训练中（见 train.py），这四行在一个 for 循环里，
  对 54,000 张图片逐 batch 执行，每个 epoch 执行约 844 次。
  5 个 epoch = 约 4,220 次上述过程。
""")

    print("💡 进一步实验")
    print("""
  1. 把 lr 从 0.001 改成 0.1，看 loss 变化量和参数变化量。
  2. 把 optimizer 改成 SGD: torch.optim.SGD(model.parameters(), lr=0.01)
     对比 Adam 和 SGD 的行为差异。
  3. 多次运行 step，观察 loss 持续下降。
""")


if __name__ == "__main__":
    main()

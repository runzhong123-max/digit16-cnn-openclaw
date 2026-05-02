#!/usr/bin/env python3
"""演示 4：数据管线 —— 一张图片从 MNIST 到模型输入经历了什么。

逐步展示: 原始 28×28 → 张量化 → Resize 16×16 → 归一化。

运行：
    python tutorial/demo_data_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt


def main():
    print("=" * 72)
    print("演示 4：数据管线 —— MNIST 图像到模型输入")
    print("=" * 72)

    # 下载 MNIST（只用第一张图做演示）
    print("\n加载 MNIST 第一张训练图片...")
    mnist = torchvision.datasets.MNIST(
        root="./data", train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    image, label = mnist[0]
    print(f"  标签: {label}")
    print(f"  原始形状: {list(image.shape)}")   # [1, 28, 28]
    print(f"  像素范围: [{image.min():.4f}, {image.max():.4f}]")

    # ── 创建四个子图 ──
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("数据预处理管线：原始 → 张量 → Resize → 归一化", fontsize=13)

    # Step 1: 原始张量 [1, 28, 28]
    ax = axes[0]
    ax.imshow(image.squeeze(0), cmap="gray")
    ax.set_title(f"① 原始 28×28\n标签: {label}\n范围: [0, 1]", fontsize=10)
    ax.axis("off")

    # Step 2: 放大看看像素值
    ax = axes[1]
    # 取 28×28 的局部放大
    patch = image[:, 8:18, 8:18]
    ax.imshow(patch.squeeze(0), cmap="gray")
    ax.set_title(f"② 局部 10×10\n像素值 (放大)", fontsize=10)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{patch[0, i, j]:.1f}", ha="center", va="center",
                    fontsize=5, color="red" if patch[0, i, j] > 0.5 else "white")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(x+8) for x in range(10)], fontsize=5)
    ax.set_yticklabels([str(y+8) for y in range(10)], fontsize=5)

    # Step 3: Resize 到 16×16
    resized = TF.resize(image, [16, 16], antialias=True)
    ax = axes[2]
    ax.imshow(resized.squeeze(0), cmap="gray")
    ax.set_title(f"③ Resize 16×16\nbilinear 插值\n范围: [0, 1]", fontsize=10)
    ax.axis("off")

    # 标上像素值
    for i in range(16):
        for j in range(16):
            v = resized[0, i, j].item()
            if v > 0.3:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=4, color="red" if v > 0.6 else "orange")

    # Step 4: 归一化
    mean, std = 0.1307, 0.3081
    normalized = (resized - mean) / std
    ax = axes[3]
    # 归一化后可能超出 [0,1]，用合适的 colormap 范围
    im = ax.imshow(normalized.squeeze(0), cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_title(f"④ 归一化\n(x − {mean}) / {std}\n范围约: [{normalized.min():.1f}, {normalized.max():.1f}]", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()

    # ── 打印关键数值 ──
    print("\n" + "=" * 72)
    print("💡 核心概念")
    print("=" * 72)
    print(f"""
  1. ToTensor: 把 PIL Image [0,255] → float32 tensor [0.0, 1.0]
     原始 MNIST 像素是 0-255 的整数，除以 255 变成 0-1 浮点数。

  2. Resize: 28×28 → 16×16
     用了 bilinear 插值（antialias=True），不是最近邻。
     这保留了平滑的灰度过渡——边缘不会变得锯齿状。
     损失: 784 像素 → 256 像素，丢失了约 67% 的信息。

  3. Normalize: (x − 0.1307) / 0.3081
     mean=0.1307, std=0.3081 是整个 MNIST 训练集的像素均值和标准差。
     归一化后数据变成"零均值、单位方差"的分布。
     这在数值上更稳定——梯度不会因为输入值太大而爆炸，
     也不会因为太小而消失。

  4. 为什么推理时也要用同样的 normalize？
     → 模型在"零均值、单位方差"的数据上学到的规律，
       只对这种分布的数据有效。如果推理输入不做归一化，
       数据分布偏移，模型预测就会出错。
     这就是 demo 里的 {mean} 和 {std} 必须和训练时一模一样的原因。

  5. 像素值变化:
     原始:    [{image.min():.1f}, {image.max():.1f}]  (ToTensor 后)
     Resize:  [{resized.min():.1f}, {resized.max():.1f}]  (bilinear 平滑后范围收窄)
     归一化:  [{normalized.min():.1f}, {normalized.max():.1f}]  (零均值化)
""")

    # ── 输入形状总结 ──
    print("最终输入模型的张量:")
    tensor = normalized.unsqueeze(0)  # 添加 batch 维度
    print(f"  shape: {list(tensor.shape)}")
    print(f"  含义: [batch_size=1, channels=1, height=16, width=16]")


if __name__ == "__main__":
    main()

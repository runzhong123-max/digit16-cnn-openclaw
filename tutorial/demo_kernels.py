#!/usr/bin/env python3
"""演示 3：可视化训练好的卷积核 —— 看 CNN 学到了什么。

加载训练好的 SimpleCNN，取出 Conv1 的 8 个 3×3 卷积核，
用热力图展示它们各自检测什么模式。

运行：
    python tutorial/demo_kernels.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from src.models import SimpleCNN


def find_latest_model():
    """自动找到最新的 SimpleCNN 权重。"""
    pattern = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "experiment_*", "simple_cnn", "simple_cnn_best.pt"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def visualize_kernel(ax, kernel, title, cmap="RdBu_r"):
    """在 subplot 上画一个 3×3 卷积核的热力图。"""
    k = kernel.numpy()  # [3, 3]
    im = ax.imshow(k, cmap=cmap, norm=CenteredNorm(), aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # 在格子里标数值
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{k[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(k[i, j]) < 0.3 else "white")
    return im


def main():
    print("=" * 72)
    print("演示 3：可视化训练好的卷积核")
    print("=" * 72)

    # 加载模型
    model_path = find_latest_model()
    if model_path is None:
        print("错误：找不到训练好的模型权重。")
        print("请先运行: python scripts/run_experiment.py")
        return

    print(f"\n加载模型: {model_path}")

    model = SimpleCNN(num_classes=10)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 取出 Conv1 的卷积核
    kernels = model.conv1.weight.data  # [8, 1, 3, 3]

    # ── 可视化 ──
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle("Conv1 的 8 个 3×3 卷积核（训练后）", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.85)

    descriptions = [
        "Kernel 0", "Kernel 1", "Kernel 2", "Kernel 3",
        "Kernel 4", "Kernel 5", "Kernel 6", "Kernel 7",
    ]

    for i in range(8):
        ax = axes[i // 4, i % 4]
        visualize_kernel(ax, kernels[i, 0], descriptions[i])

    plt.tight_layout()
    plt.show()

    # ── 解读 ──
    print("\n" + "=" * 72)
    print("💡 如何解读卷积核")
    print("=" * 72)
    print("""
  颜色含义 (RdBu_r 配色):
    红色 (正值)  →  该位置像素越亮，这个核的响应越强
    蓝色 (负值)  →  该位置像素越亮，这个核的响应越弱
    白色 (0)     →  该位置像素不影响这个核

  常见模式:
    ┌          ┐    ┌          ┐    ┌          ┐
    │+ 0 −│    │+ + +│    │+ − 0│
    │+ 0 −│    │ 0 0 0│    │− 0 +│
    │+ 0 −│    │− − −│    │0 + −│
    └          ┘    └          ┘    └          ┘
    竖线检测器      横线检测器      斜线检测器
    (左亮右暗)     (上亮下暗)     (对角线)

  如果卷积核看起来像随机噪声（没有明显模式），可能的原因:
    1. 训练轮数不够（epochs 太少）
    2. 这个核是冗余的——模型不需要全部 8 个核
    3. 这个核学到的模式对人类来说不直观

  Conv2 的卷积核是 3×3×8 的（8 个输入通道），无法直接可视化为 3×3 矩阵。
  但它的逻辑类似——在 Conv1 输出的 8 种"低级特征"上寻找组合模式。
""")

    # ── 打印数值 ──
    print("Conv1 卷积核数值:")
    for i in range(8):
        k = kernels[i, 0]
        print(f"\n  Kernel {i}:")
        for row in range(3):
            vals = [f"{k[row, col]:+7.3f}" for col in range(3)]
            print(f"    {'  '.join(vals)}")


if __name__ == "__main__":
    main()

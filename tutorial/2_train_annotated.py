# =============================================================================
# 2_train_annotated.py — 训练循环（带详细注释）
# =============================================================================
# 这是 digit16-cnn-experiment/src/train.py 的教学注释版。
# 聚焦于核心训练逻辑：一个 epoch 里发生了什么，参数如何被更新。
# =============================================================================

import os, time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 非交互模式——用于保存图片，不需要 GUI
import matplotlib.pyplot as plt

from src.utils import save_json, save_csv


# ═════════════════════════════════════════════════════════════════════════════
# train_one_epoch — 一个 epoch 的训练
# ═════════════════════════════════════════════════════════════════════════════
#
# 一个 epoch = 把训练集的所有图片都过一遍。
# 在这个项目里，54,000 张训练图，batch_size=64，
# 所以一个 epoch = 54,000 / 64 ≈ 844 次迭代。
#
# 每次迭代做四件事（就是之前 demobackprop 演示的）:
#   ① optimizer.zero_grad()  — 清空旧梯度
#   ② outputs = model(x)     — 前向传播
#   ③ loss.backward()        — 反向传播（链式法则）
#   ④ optimizer.step()       — 参数更新
#
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,  # 训练数据加载器
    criterion: nn.Module,                  # 损失函数 (CrossEntropyLoss)
    optimizer: torch.optim.Optimizer,      # 优化器 (Adam)
    device: torch.device,                  # CPU 或 CUDA
) -> tuple[float, float]:
    """
    返回: (平均 loss, 准确率)
    """

    # model.train() — 切换到训练模式
    # 这会启用 Dropout（在 MLP/CNN 中随机丢弃神经元）
    # 如果忘了调用 .train() 而一直用 .eval()，Dropout 会一直关闭
    model.train()

    running_loss = 0.0  # 累积 loss，用于计算平均值
    correct = 0          # 累积正确预测数
    total = 0            # 累积总样本数

    # DataLoader 每次迭代返回一个 batch
    # images: [batch_size, 1, 16, 16]
    # labels: [batch_size] — 整数标签 0-9
    for images, labels in loader:
        # 把数据移到目标设备（CPU 或 GPU）
        # 在这个项目中 device 是 CPU，所以这一步实际上什么都没做。
        # 但在 GPU 训练中，.to(device) 会把张量从 CPU 内存复制到 GPU 显存。
        images, labels = images.to(device), labels.to(device)

        # ① 清空梯度 — 这是 PyTorch 设计的一个"坑"
        # PyTorch 的梯度默认是累加的（方便 RNN 等场景）。
        # 如果不 zero_grad()，每个 batch 的梯度会叠加上一个 batch 的，
        # 导致更新量错误。所以每个 batch 开始前必须清空。
        optimizer.zero_grad()

        # ② 前向传播
        # model(images) 调用的是 model.forward(images)
        # 输出 [batch_size, 10] — 每个样本 10 个 logits
        outputs = model(images)

        # ③ 计算 loss
        # criterion = nn.CrossEntropyLoss()
        # 它内部做了：
        #   1. softmax(logits) → 概率
        #   2. -log(真实类别的概率)
        #   3. 所有样本取平均
        loss = criterion(outputs, labels)

        # ④ 反向传播
        # loss.backward() 做的事：
        #   从 loss 这个标量出发，沿计算图反向传播，
        #   用链式法则计算每个参数的梯度 ∂loss/∂param，
        #   梯度存在 param.grad 里。
        #
        # 这一步执行后：
        #   model.conv1.weight.grad 包含了卷积核 1 的梯度
        #   model.fc1.weight.grad    包含了 FC1 的梯度
        #   ... 所有 18,346 个参数都有梯度了
        loss.backward()

        # ⑤ 参数更新
        # optimizer.step() 做的事：
        #   对于每个参数: param = param - lr * Adam(梯度, 历史统计)
        #   Adam 不是简单减 lr×grad，而是：
        #     - 维护梯度的一阶矩（动量）和二阶矩（方差）
        #     - 用这两个矩自适应调整每个参数的学习率
        #     - 对稀疏梯度更友好（梯度小的参数会给大一点的学习率）
        optimizer.step()

        # ── 统计指标 ──
        # loss.item() 把 0-维 tensor 转换成 Python float
        # images.size(0) = batch_size（当前 batch 的样本数）
        running_loss += loss.item() * images.size(0)

        # outputs.max(1) 返回 (最大值, 索引)
        # .max(1) 的 1 表示沿第 1 维（类别维）取最大值
        # predicted 是每个样本预测的类别 (0-9)
        _, predicted = outputs.max(1)

        # predicted.eq(labels) — 逐元素比较，返回 bool tensor
        # .sum().item() — 计数正确的数量
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    # 返回 epoch 的平均 loss 和准确率
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ═════════════════════════════════════════════════════════════════════════════
# evaluate_epoch — 验证/测试评估
# ═════════════════════════════════════════════════════════════════════════════
#
# 和 train_one_epoch 几乎一样，但有三个区别：
#   1. 用 model.eval() 而不是 model.train()
#       → Dropout 关闭，BatchNorm（如果有）用全局统计
#   2. @torch.no_grad() 装饰器
#       → 关闭 autograd 计算图，节省内存，加速推理
#   3. 没有 optimizer.zero_grad() / loss.backward() / optimizer.step()
#       → 只前向传播，不更新参数
#
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()  # 评估模式（关闭 Dropout）
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ═════════════════════════════════════════════════════════════════════════════
# train_model — 完整训练流程
# ═════════════════════════════════════════════════════════════════════════════
#
# 职责：
#   1. 创建损失函数 (CrossEntropyLoss) 和优化器 (Adam)
#   2. 循环 epochs 次，每次调用 train_one_epoch + evaluate_epoch
#   3. 记录每个 epoch 的 train/val loss 和 accuracy
#   4. 保存最佳模型（val accuracy 最高的那个）
#   5. 保存 metrics.csv + 训练曲线图
#
# ═════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: str,
    model_name: str = "model",
) -> dict:

    # ── 损失函数 ──
    # CrossEntropyLoss = softmax + NLLLoss 的组合
    # 输入: logits [B, 10] + labels [B]
    # 输出: 标量 loss
    # 内部自动对 logits 做 softmax，所以模型的 forward 不要做 softmax
    criterion = nn.CrossEntropyLoss()

    # ── 优化器 ──
    # Adam 的核心思想：
    #   每个参数有自己的学习率，根据历史梯度自动调整。
    #   m_t = β₁·m_{t-1} + (1-β₁)·g_t     (一阶矩 = 梯度的指数移动平均)
    #   v_t = β₂·v_{t-1} + (1-β₂)·g_t²    (二阶矩 = 梯度平方的指数移动平均)
    #   更新: θ_t = θ_{t-1} − lr · m_t / (√v_t + ε)
    # 相比 SGD:
    #   SGD: θ = θ - lr·g        (所有参数同一个学习率)
    #   Adam: 每个参数自适应学习率，通常收敛更快
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── 训练历史记录 ──
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, f"{model_name}_best.pt")

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Device: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    start_time = time.time()

    # ── Epoch 循环 ──
    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证一个 epoch
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device
        )

        # 记录
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 如果当前验证准确率 > 历史最佳，保存模型
        # 为什么按 val_acc 而不是 train_acc 选最佳模型？
        #   train_acc 可能因为过拟合而虚高——模型背下了训练集但泛化差。
        #   val_acc 反映的是模型在未见数据上的真实能力。
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time

    # 保存 CSV
    csv_rows = [{k: history[k][i] for k in history} for i in range(epochs)]
    save_csv(csv_rows, os.path.join(output_dir, f"{model_name}_metrics.csv"))

    # 保存训练曲线图
    _plot_training_curve(history, os.path.join(output_dir, f"{model_name}_training_curve.png"))

    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Training time: {elapsed:.1f}s")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": history["val_acc"].index(best_val_acc) + 1,
        "epochs_trained": epochs,
        "training_time_s": round(elapsed, 1),
    }


def _plot_training_curve(history: dict, save_path: str) -> None:
    """画训练曲线：左边 loss，右边 accuracy。"""
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss 曲线 ──
    # train loss > val loss 是正常的——val 没有 Dropout，loss 自然更低。
    # 如果 val loss 上升而 train loss 下降 → 过拟合。
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy 曲线 ──
    # 理想情况：训练和验证准确率一起上升，最后趋于平稳。
    # 如果训练准确率 >> 验证准确率 → 过拟合。
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

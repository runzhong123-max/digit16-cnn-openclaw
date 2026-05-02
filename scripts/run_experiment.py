"""Full experiment: train MLP and SimpleCNN, evaluate, compare, report."""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.utils import set_seed, get_device, make_timestamped_dir, load_yaml_config, save_json, save_csv
from src.data import get_mnist_dataloaders
from src.models import get_model
from src.train import train_model
from src.evaluate import evaluate_on_test


def main():
    print("=" * 60)
    print("FULL EXPERIMENT — Digit16 CNN Experiment")
    print("=" * 60)

    # Load config
    config = load_yaml_config("configs/default.yaml")

    device = get_device(config.get("device", "cpu"))
    set_seed(config["seed"])

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    image_size = config["image_size"]
    seed = config["seed"]

    print(f"\nDevice: {device}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {image_size}×{image_size}")

    # Load data
    print("\n[1/5] Loading MNIST data (28×28 → 16×16)...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        seed=seed,
        use_subset=False,
    )
    print(
        f"  Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # Create timestamped output directory
    output_dir = make_timestamped_dir(config["output_dir"], prefix="experiment")

    # Train MLP
    print("\n[2/5] Training MLP baseline...")
    mlp = get_model("mlp").to(device)
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"  Parameters: {n_mlp:,}")

    mlp_dir = os.path.join(output_dir, "mlp")
    os.makedirs(mlp_dir, exist_ok=True)

    mlp_result = train_model(
        model=mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        device=device,
        output_dir=mlp_dir,
        model_name="mlp",
    )

    # Evaluate MLP on test set
    mlp.load_state_dict(
        torch.load(
            mlp_result["best_model_path"],
            map_location=device,
            weights_only=True,
        )
    )
    mlp_test = evaluate_on_test(
        model=mlp,
        test_loader=test_loader,
        device=device,
        output_dir=mlp_dir,
        model_name="mlp",
    )

    # Train CNN
    print("\n[3/5] Training SimpleCNN...")
    cnn = get_model("simple_cnn").to(device)
    n_cnn = sum(p.numel() for p in cnn.parameters())
    print(f"  Parameters: {n_cnn:,}")

    cnn_dir = os.path.join(output_dir, "simple_cnn")
    os.makedirs(cnn_dir, exist_ok=True)

    cnn_result = train_model(
        model=cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        device=device,
        output_dir=cnn_dir,
        model_name="simple_cnn",
    )

    # Evaluate CNN on test set
    cnn.load_state_dict(
        torch.load(
            cnn_result["best_model_path"],
            map_location=device,
            weights_only=True,
        )
    )
    cnn_test = evaluate_on_test(
        model=cnn,
        test_loader=test_loader,
        device=device,
        output_dir=cnn_dir,
        model_name="simple_cnn",
    )

    # Compare
    print("\n[4/5] Building comparison...")
    comparison = [
        {
            "model": "MLP",
            "parameters": n_mlp,
            "best_val_acc": mlp_result["best_val_acc"],
            "test_accuracy": mlp_test["test_accuracy"],
            "test_macro_f1": mlp_test["test_macro_f1"],
            "training_time_s": mlp_result["training_time_s"],
            "best_epoch": mlp_result["best_epoch"],
        },
        {
            "model": "SimpleCNN",
            "parameters": n_cnn,
            "best_val_acc": cnn_result["best_val_acc"],
            "test_accuracy": cnn_test["test_accuracy"],
            "test_macro_f1": cnn_test["test_macro_f1"],
            "training_time_s": cnn_result["training_time_s"],
            "best_epoch": cnn_result["best_epoch"],
        },
    ]
    save_csv(
        comparison,
        os.path.join(output_dir, "comparison.csv"),
    )
    save_json(
        comparison,
        os.path.join(output_dir, "comparison.json"),
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<12} {'Params':>8} {'Val Acc':>8} {'Test Acc':>8} {'Macro F1':>8} {'Time(s)':>8}")
    print("-" * 60)
    for row in comparison:
        print(
            f"{row['model']:<12} {row['parameters']:>8,} "
            f"{row['best_val_acc']:>8.4f} {row['test_accuracy']:>8.4f} "
            f"{row['test_macro_f1']:>8.4f} {row['training_time_s']:>8.1f}"
        )
    print("=" * 60)

    # Generate report
    print("\n[5/5] Generating experiment report...")
    _generate_report(
        config=config,
        comparison=comparison,
        mlp_dir=mlp_dir,
        cnn_dir=cnn_dir,
        output_dir=output_dir,
        n_mlp=n_mlp,
        n_cnn=n_cnn,
    )

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


def _generate_report(
    config: dict,
    comparison: list,
    mlp_dir: str,
    cnn_dir: str,
    output_dir: str,
    n_mlp: int,
    n_cnn: int,
) -> None:
    """Generate experiment_report.md."""
    import datetime

    mlp = comparison[0]
    cnn = comparison[1]

    report = f"""# 16×16 Digit CNN Recognition — Experiment Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 实验目标

验证一个小型 CNN 在 16×16 灰度手写数字图像上的识别能力，并与一个简单 MLP baseline 进行对比。

核心问题:
- 在极小分辨率(16×16)下，CNN 的局部特征提取是否优于 MLP 的全局全连接？
- 在参数相近或更少的情况下，模型表现如何？

---

## 2. 数据集说明

- **来源:** MNIST (torchvision.datasets.MNIST)
- **原始大小:** 28×28 灰度图
- **类别:** 0–9 (10 类)
- **划分:**
  - Train: 54,000 张 (原始 train 的 90%)
  - Validation: 6,000 张 (原始 train 的 10%)
  - Test: 10,000 张 (官方 test set)

---

## 3. 输入图像处理

1. 原始 28×28 → Resize 到 16×16 (antialias=True, bilinear interpolation)
2. ToTensor: [0, 255] → [0.0, 1.0]
3. Normalize: mean=0.1307, std=0.3081 (MNIST 全局统计值)
4. 最终输入形状: `[batch_size, 1, 16, 16]`

---

## 4. 模型结构

### MLP Baseline

```
Input:  256 (flattened 16×16)
  ↓
FC(256 → 128) + ReLU + Dropout(0.2)
  ↓
FC(128 → 64) + ReLU + Dropout(0.2)
  ↓
FC(64 → 10)
```

参数量: {n_mlp:,}

### SimpleCNN

```
Input: 1×16×16
  ↓
Conv(1→8, 3×3, pad=1) + ReLU + MaxPool(2)  → 8×8×8
  ↓
Conv(8→16, 3×3, pad=1) + ReLU + MaxPool(2) → 16×4×4
  ↓
Flatten → 256
  ↓
FC(256 → 64) + ReLU + Dropout(0.25)
  ↓
FC(64 → 10)
```

参数量: {n_cnn:,}

---

## 5. 超参数

| Parameter    | Value       |
|-------------|-------------|
| Seed        | {config['seed']} |
| Epochs      | {config['epochs']} |
| Batch size  | {config['batch_size']} |
| Optimizer   | Adam |
| Learning rate | {config['learning_rate']} |
| Loss        | CrossEntropyLoss |
| Image size  | {config['image_size']}×{config['image_size']} |
| Device      | {config.get('device', 'cpu')} |

---

## 6. 实验结果

| Model      | Params   | Best Val Acc | Test Acc | Macro F1 | Train Time |
|-----------|----------|-------------|----------|----------|------------|
| MLP       | {n_mlp:,} | {mlp['best_val_acc']:.4f} | {mlp['test_accuracy']:.4f} | {mlp['test_macro_f1']:.4f} | {mlp['training_time_s']:.1f}s |
| SimpleCNN | {n_cnn:,} | {cnn['best_val_acc']:.4f} | {cnn['test_accuracy']:.4f} | {cnn['test_macro_f1']:.4f} | {cnn['training_time_s']:.1f}s |

**差异:**
- Test Accuracy 差异: {cnn['test_accuracy'] - mlp['test_accuracy']:+.4f}
- Macro F1 差异: {cnn['test_macro_f1'] - mlp['test_macro_f1']:+.4f}

---

## 7. 训练曲线

训练曲线保存在各模型的结果目录下:

- MLP: `{os.path.relpath(mlp_dir, output_dir)}/mlp_training_curve.png`
- SimpleCNN: `{os.path.relpath(cnn_dir, output_dir)}/simple_cnn_training_curve.png`

---

## 8. 混淆矩阵

混淆矩阵保存在各模型的结果目录下:

- MLP: `{os.path.relpath(mlp_dir, output_dir)}/mlp_confusion_matrix.png`
- SimpleCNN: `{os.path.relpath(cnn_dir, output_dir)}/simple_cnn_confusion_matrix.png`

---

## 9. 结论

- **MLP baseline** 在 16×16 输入上达到了 {mlp['test_accuracy']:.2%} 的测试准确率。
- **SimpleCNN** 在 16×16 输入上达到了 {cnn['test_accuracy']:.2%} 的测试准确率。
- CNN 相比 MLP: {'提升了' if cnn['test_accuracy'] > mlp['test_accuracy'] else '没有提升'}测试准确率，{'表明卷积网络的局部特征提取能力在小分辨率图像上仍然有效。' if cnn['test_accuracy'] > mlp['test_accuracy'] else '说明在极低分辨率下，MLP 也能学到足够的模式。'}

---

## 10. 局限性

1. **模型极小:** CNN 只有 2 个卷积层，参数量很少，上限有限。
2. **分辨率损失:** 28×28 → 16×16 丢失了约 67% 的像素信息。
3. **训练轮数少:** 仅 5 个 epoch，模型可能未完全收敛。
4. **无数据增强:** 未使用旋转、平移等增强手段。
5. **单次运行:** 仅一次 random seed 结果，可能有随机波动。

---

## 11. 下一步实验建议

1. **增加模型容量:** 尝试更多卷积层或更多通道数。
2. **增加训练轮数:** 训练到收敛 (10–20 epochs)。
3. **添加数据增强:** 随机平移、小幅度旋转。
4. **多次运行:** 使用多个 seed 计算均值和标准差。
5. **不同分辨率对比:** 对比 8×8, 16×16, 24×24, 28×28 的表现。
6. **正则化探索:** BatchNorm、不同 Dropout 率。
7. **学习率调度:** ReduceLROnPlateau 或 CosineAnnealing。
8. **混淆矩阵分析:** 针对易混淆数字对的针对性改进。
"""
    report_path = os.path.join(output_dir, "..", "reports", os.path.basename(output_dir) + "_report.md")
    # Actually save under results dir for consistency
    report_path = os.path.join(output_dir, "experiment_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()

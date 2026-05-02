# 16×16 Digit CNN Recognition — Experiment Report

**Generated:** 2026-05-02 18:15:02

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

参数量: 41,802

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

参数量: 18,346

---

## 5. 超参数

| Parameter    | Value       |
|-------------|-------------|
| Seed        | 42 |
| Epochs      | 5 |
| Batch size  | 64 |
| Optimizer   | Adam |
| Learning rate | 0.001 |
| Loss        | CrossEntropyLoss |
| Image size  | 16×16 |
| Device      | cpu |

---

## 6. 实验结果

| Model      | Params   | Best Val Acc | Test Acc | Macro F1 | Train Time |
|-----------|----------|-------------|----------|----------|------------|
| MLP       | 41,802 | 0.9712 | 0.9758 | 0.9757 | 17.0s |
| SimpleCNN | 18,346 | 0.9805 | 0.9828 | 0.9826 | 31.1s |

**差异:**
- Test Accuracy 差异: +0.0070
- Macro F1 差异: +0.0070

---

## 7. 训练曲线

训练曲线保存在各模型的结果目录下:

- MLP: `mlp/mlp_training_curve.png`
- SimpleCNN: `simple_cnn/simple_cnn_training_curve.png`

---

## 8. 混淆矩阵

混淆矩阵保存在各模型的结果目录下:

- MLP: `mlp/mlp_confusion_matrix.png`
- SimpleCNN: `simple_cnn/simple_cnn_confusion_matrix.png`

---

## 9. 结论

- **MLP baseline** 在 16×16 输入上达到了 97.58% 的测试准确率。
- **SimpleCNN** 在 16×16 输入上达到了 98.28% 的测试准确率。
- CNN 相比 MLP: 提升了测试准确率，表明卷积网络的局部特征提取能力在小分辨率图像上仍然有效。

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

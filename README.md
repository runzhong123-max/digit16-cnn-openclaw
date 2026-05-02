# Digit16 CNN Experiment

**16×16 手写数字识别：MLP vs CNN 对比实验**

## 项目目标

验证小型 CNN 是否能在 16×16 灰度手写数字图像上完成识别任务，并与 MLP baseline 对比。

## 环境要求

- Python 3.9+
- CPU 即可运行（可选 CUDA）

## 安装依赖

```bash
cd digit16-cnn-experiment
pip install -r requirements.txt
```

## 运行 Smoke Test

验证全流程能跑通（使用极小数据子集，1 个 epoch）：

```bash
cd digit16-cnn-experiment
python scripts/smoke_test.py
```

## 运行正式实验

```bash
cd digit16-cnn-experiment
python scripts/run_experiment.py
```

## 运行交互式 Demo

启动 16×16 手写数字识别 GUI 演示器（使用 matplotlib，无需额外 GUI 框架）：

```bash
cd digit16-cnn-experiment
source venv/bin/activate

# 自动发现最新模型权重
python demo/matplotlib_digit_demo.py

# 或指定模型路径
python demo/matplotlib_digit_demo.py --model-path results/experiment_.../simple_cnn/simple_cnn_best.pt

# 调整画笔大小
python demo/matplotlib_digit_demo.py --brush-size 2
```

### Demo 操作说明

| 操作 | 方式 |
|------|------|
| 画数字 | **左键拖拽** 在 16×16 网格上 |
| 擦除 | **右键拖拽** |
| 预测 | 点击 `[Predict]` 按钮 或按键盘 `p` |
| 清空 | 点击 `[Clear]` 按钮 或按键盘 `c` |
| 保存 | 点击 `[Save]` 按钮 或按键盘 `s`，保存为 .npy |

右侧显示:
- Top-1 预测数字 + 置信度
- 0-9 每个类别的概率柱状图

### ⚠️ 分布差异注意

用户在 16×16 格子上直接画出的数字与 MNIST 下采样图像在分布上有差异：
- MNIST 图像是 28×28 扫描原件经 bilinear 下采样到 16×16，像素有平滑过渡
- 本 demo 的输入只有 0 或 1（二值），且手写轨迹与扫描数字风格不同
- **因此预测准确率可能低于测试集报告值**，这是数据分布差异导致的正常现象

## 输出结果

每次实验会创建带时间戳的结果目录：

```
results/
  experiment_YYYYMMDD_HHMMSS/
    mlp/
      mlp_best.pt                 # MLP 最佳模型权重
      mlp_metrics.csv             # 训练曲线数据
      mlp_training_curve.png      # 训练曲线图
      mlp_test_metrics.json       # 测试指标
      mlp_confusion_matrix.png    # 混淆矩阵
    simple_cnn/
      simple_cnn_best.pt
      simple_cnn_metrics.csv
      simple_cnn_training_curve.png
      simple_cnn_test_metrics.json
      simple_cnn_confusion_matrix.png
    comparison.csv               # 模型对比表
    comparison.json              # 模型对比 (JSON)
    experiment_report.md         # 完整实验报告
```

## 配置

所有超参数在 `configs/default.yaml` 中统一管理：

- `seed`: 随机种子
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `image_size`: 输入图像大小
- `use_subset_for_debug`: 是否使用数据子集

## 项目结构

```
digit16-cnn-experiment/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── utils.py          # 工具函数（seed, device, IO）
│   ├── data.py           # 数据加载（MNIST → 16×16）
│   ├── models.py         # 模型定义（MLP, SimpleCNN）
│   ├── train.py          # 训练循环
│   ├── evaluate.py       # 测试评估
│   └── infer.py          # 推理接口（供 demo 调用）
├── demo/
│   ├── tkinter_digit_demo.py       # Tkinter 版本（需系统 Tcl/Tk）
│   └── matplotlib_digit_demo.py    # matplotlib 版本（推荐，跨平台）
├── scripts/
│   ├── smoke_test.py     # 冒烟测试
│   └── run_experiment.py # 正式实验
└── results/              # 实验结果
```

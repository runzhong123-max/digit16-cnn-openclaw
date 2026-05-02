# Digit16 CNN

一个麻雀虽小五脏俱全的 CNN 实验项目——在 16×16 像素上做手写数字识别，顺带一个能直接用鼠标画数字的交互 demo。

## 快速开始

```bash
git clone git@github.com:runzhong123-max/digit16-cnn-openclaw.git
cd digit16-cnn-openclaw
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 三个命令

```bash
# 1. 跑冒烟测试（1 分钟，验证一切正常）
python scripts/smoke_test.py

# 2. 跑正式实验（2 分钟，训练 MLP 和 CNN，出报告）
python scripts/run_experiment.py

# 3. 开交互 demo（用鼠标画数字，看模型实时预测）
python demo/matplotlib_digit_demo.py
```

## 有什么

| 东西 | 在哪 |
|------|------|
| MLP baseline (42k 参数，97.6%) | `src/models.py` |
| SimpleCNN (18k 参数，98.3%) | `src/models.py` |
| 训练 + 验证 + 测试 | `scripts/run_experiment.py` |
| 交互画板 demo | `demo/matplotlib_digit_demo.py` |
| 实验报告（含曲线和混淆矩阵） | `results/` |
| 逐行注释的教学版源码 | `tutorial/` |

## 拿来就改

改 `configs/default.yaml` 里的参数，重跑实验：

```yaml
epochs: 10          # 多练几轮
learning_rate: 0.0005
batch_size: 32
```

改模型？打开 `src/models.py`，加层或改通道数，再跑 `run_experiment.py`。

## 学了什么读完什么

`tutorial/` 里有四个带逐行注释的源码 + 四个独立可跑的演示程序，从「张量怎么流」到「梯度怎么传」到「卷积核长什么样」，一步步拆解。

```bash
python tutorial/demo_forward_pass.py   # 跟踪每层张量形状
python tutorial/demo_backprop.py       # 看参数更新前后变化
python tutorial/demo_kernels.py        # 卷积核可视化
python tutorial/demo_data_pipeline.py  # 预处理四步走
```

## 依赖

Python 3.9+，纯 CPU 运行。`requirements.txt` 就七个包：torch, torchvision, numpy, pandas, scikit-learn, matplotlib, pyyaml。

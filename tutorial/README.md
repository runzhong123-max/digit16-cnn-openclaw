# 阅读资料：CNN 手写数字识别核心源码精读

本文件夹是为理解 `digit16-cnn-experiment` 项目核心机制而准备的阅读材料。

## 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `1_models_annotated.py` | 带注释源码 | 模型定义——逐行解释 MLP 和 SimpleCNN |
| `2_train_annotated.py` | 带注释源码 | 训练循环——逐行解释参数如何被更新 |
| `3_infer_annotated.py` | 带注释源码 | 推理——模型训练好后怎么用 |
| `4_data_annotated.py` | 带注释源码 | 数据管线——数据怎么进来 |
| `demo_forward_pass.py` | 演示程序 | 单次前向传播：逐步打印张量形状和数值 |
| `demo_backprop.py` | 演示程序 | 一次反向传播：观察参数更新前后的变化 |
| `demo_kernels.py` | 演示程序 | 可视化训练好的卷积核长什么样 |
| `demo_data_pipeline.py` | 演示程序 | 逐步展示数据预处理的效果 |

## 阅读顺序

```
第一步：demo_forward_pass.py  →  跑一遍，感受数据流动
第二步：1_models_annotated.py  →  理解模型结构
第三步：demo_backprop.py      →  跑一遍，感受参数更新
第四步：2_train_annotated.py  →  理解训练循环
第五步：demo_kernels.py       →  看卷积核学到了什么
第六步：demo_data_pipeline.py →  理解数据预处理
第七步：3_infer_annotated.py  →  理解推理
第八步：4_data_annotated.py   →  理解数据管线细节
```

## 运行方式

```bash
cd digit16-cnn-experiment
source venv/bin/activate

# 演示程序（独立运行，不依赖训练）
python tutorial/demo_forward_pass.py
python tutorial/demo_backprop.py
python tutorial/demo_kernels.py
python tutorial/demo_data_pipeline.py
```

## 学习建议

1. **先跑演示程序，再看带注释源码。** 数据流动的感受比抽象理解重要。
2. **用纸笔跟踪张量形状变化。** [B, 1, 16, 16] → [B, 8, 8, 8] → ... 每一步写下形状。
3. **改参数看变化。** 把 kernel_size 从 3 改成 5，看形状怎么变。把通道数从 8 改成 16。
4. **对照 loss 曲线理解。** 跑 `demo_backprop.py` 时观察 loss 下降，理解梯度下降在做什么。

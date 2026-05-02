# =============================================================================
# 4_data_annotated.py — 数据加载和预处理（带详细注释）
# =============================================================================
# 这是 digit16-cnn-experiment/src/data.py 的教学注释版。
# 解释：数据怎么从磁盘到模型、transform 每一步在做什么、
#       train/val/test 怎么划分。
# =============================================================================

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ═════════════════════════════════════════════════════════════════════════════
# get_mnist_dataloaders — 主函数
# ═════════════════════════════════════════════════════════════════════════════
#
# 职责：
#   1. 下载/加载 MNIST 数据集
#   2. 应用 transform（预处理）
#   3. 划分 train/val/test
#   4. 包装成 DataLoader
#
# 返回值：train_loader, val_loader, test_loader
# ═════════════════════════════════════════════════════════════════════════════

def get_mnist_dataloaders(
    batch_size: int = 64,
    image_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    use_subset: bool = False,
    subset_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    # ── transform — 预处理流水线 ──
    # transforms.Compose 把多个 transform 串联起来，按顺序应用。
    # 每张图从磁盘加载后会依次经过这三个变换：
    #
    # ① ToTensor()
    #    - 把 PIL Image [0, 255] 转成 float32 tensor [0.0, 1.0]
    #    - 同时把形状从 [H, W] 变成 [C, H, W]（灰度图 C=1）
    #    - 等效于: tensor = img / 255.0
    #
    # ② Resize((16, 16), antialias=True)
    #    - 把 28×28 缩放为 16×16
    #    - antialias=True: 使用抗锯齿滤波器，避免缩放后出现锯齿伪影
    #    - 默认使用 bilinear 插值（不是最近邻）
    #    - bilinear: 新像素值 = 周围 4 个像素的加权平均（平滑过渡）
    #    - 最近邻: 新像素值 = 最近的那个原始像素（会产生锯齿）
    #
    # ③ Normalize(mean=0.1307, std=0.3081)
    #    - 把 [0, 1] 范围的像素变成零均值单位方差
    #    - 公式: x_norm = (x - 0.1307) / 0.3081
    #    - 0.1307 和 0.3081 是整个 MNIST 训练集的全局统计值
    #    - 效果：输入 ≈ N(0, 1) 分布 → 梯度下降更稳定
    #
    # 注意：Normalize 必须放在 Resize 之后！
    # 因为 0.1307 和 0.3081 是原始 28×28 图像的统计值，
    # Resize 后像素分布会略微变化，但我们仍沿用相同参数。
    # 这是深度学习中常见的近似处理。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # ── 加载数据集 ──
    # train=True: 加载 MNIST 官方训练集（60,000 张）
    # download=True: 如果本地没有，自动从 torchvision 镜像下载
    # 下载后数据存在 ./data/MNIST/ 目录
    #
    # MNIST 数据集结构：
    #   train: 60,000 张 28×28 灰度图 (我们把它再划分成 train + val)
    #   test:  10,000 张 28×28 灰度图 (官方测试集)
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # ── 划分 train / validation ──
    # val_ratio = 0.1，所以:
    #   val:   60,000 × 0.1 = 6,000 张
    #   train: 60,000 − 6,000 = 54,000 张
    #
    # random_split 随机划分，保证 train 和 val 的类别分布大致均匀。
    #
    # torch.Generator().manual_seed(seed) 确保每次运行都得到相同的划分——
    # 这是可复现性的关键。如果不用 seed，每次划分不同，实验结果无法对比。
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # ── Debug 模式：使用极小子集 ──
    # smoke_test.py 会启用这个模式，只用 256 张训练图跑 1 个 epoch。
    # 这让我们在开发时快速验证代码是否正确，
    # 不需要等完整的 5 epochs × 54,000 张图。
    if use_subset:
        train_set = _trim_subset(train_set, subset_size, seed)
        val_set = _trim_subset(val_set, max(subset_size // 8, 16), seed)
        test_set = _trim_subset(test_set, max(subset_size // 4, 32), seed)

    # ── DataLoader — 把数据集包装成可迭代的 batch 加载器 ──
    # DataLoader 的职责：
    #   1. shuffle: 每个 epoch 随机打乱数据顺序（防止模型记住顺序）
    #   2. batch: 把单张图片合并成 [batch_size, C, H, W]
    #   3. num_workers: 多进程加载（0 = 主进程加载）
    #
    # shuffle=True for train: 每个 epoch 的顺序不同，防止模型学习"样本顺序"而不是"样本内容"
    # shuffle=False for val/test: 顺序不重要，而且我们可能需要按顺序做分析
    #
    # num_workers=0: 在这个项目里用 0 就够了（CPU 训练，数据加载不是瓶颈）。
    # 如果 GPU 训练中数据加载成为瓶颈，可以设为 4 或 8。
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# _trim_subset — 从数据集中随机抽取子集
# ═════════════════════════════════════════════════════════════════════════════
#
# 用于 debug 模式。随机选取 size 个样本，包装成 Subset。
#
# torch.randperm(n) 生成 [0, n-1] 的随机排列。
# [:size] 取前 size 个索引。
#
# Subset 是 PyTorch 内置类——它接收一个数据集和索引列表，
# 只暴露索引列表中的样本。底层数据不复制，节省内存。
# ═════════════════════════════════════════════════════════════════════════════

def _trim_subset(dataset, size: int, seed: int) -> Subset:
    n = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)

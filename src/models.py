"""Model definitions: MLP baseline and SimpleCNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP baseline for 16×16 digit recognition.

    Input:  flattened 256-dim vector
    Architecture:  256 → 128 → 64 → 10
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, 16, 16]
        x = x.view(x.size(0), -1)  # flatten → [B, 256]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """Small CNN for 16×16 digit recognition.

    Architecture:
        Conv(1→8, 3×3, pad=1) → ReLU → MaxPool(2)
        Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2)
        Flatten → FC(16*4*4=256 → 64) → ReLU → Dropout → FC(64 → 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 16→8→4

        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 16, 16]
        x = self.pool(F.relu(self.conv1(x)))  # [B, 8, 8, 8]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 4, 4]
        x = x.view(x.size(0), -1)             # [B, 256]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function: return a model by name."""
    models = {
        "mlp": MLP,
        "simple_cnn": SimpleCNN,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(models.keys())}."
        )
    return models[model_name](**kwargs)

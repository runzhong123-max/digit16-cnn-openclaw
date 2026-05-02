"""Model inference for 16×16 digit recognition.

Loads a trained model and runs inference on a 16×16 numpy array.

Normalization must match the training pipeline:
    ToTensor → Resize(16,16) → Normalize(mean=0.1307, std=0.3081)

Usage as module:
    from src.infer import DigitInferer
    inferer = DigitInferer("path/to/model.pt")
    probs = inferer.predict(grid_16x16)  # grid_16x16: np.ndarray shape (16,16), values 0.0–1.0
    top_digit, top_conf = inferer.predict_top(grid_16x16)
"""

import numpy as np
import torch

# MNIST training normalization constants — must match src/data.py
_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


class DigitInferer:
    """Load a SimpleCNN model and run inference on 16×16 hand-drawn digits."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: Path to a saved SimpleCNN state_dict (.pt file).
            device: torch device string (default "cpu").
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model_path = model_path

    def _load_model(self, model_path: str):
        """Import SimpleCNN from src.models, load state_dict, set eval mode."""
        # Lazy import so that demo scripts don't need to know about src layout
        from src.models import SimpleCNN

        model = SimpleCNN(num_classes=10)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, grid: np.ndarray) -> np.ndarray:
        """
        Run inference on a 16×16 numpy array.

        Args:
            grid: (16, 16) numpy array, values in [0.0, 1.0].
                  Typically 1.0 where the user drew, 0.0 elsewhere.

        Returns:
            probs: (10,) numpy array of softmax probabilities.
        """
        if grid.shape != (16, 16):
            raise ValueError(
                f"Expected grid shape (16, 16), got {grid.shape}"
            )
        if grid.min() < 0.0 or grid.max() > 1.0:
            raise ValueError("Grid values must be in [0.0, 1.0]")

        # Convert to tensor: [H, W] → [1, 1, H, W]
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Apply the SAME normalization used during training
        tensor = (tensor - _MNIST_MEAN) / _MNIST_STD

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)          # [1, 10]
            probs = torch.softmax(logits, dim=1)  # [1, 10]

        return probs.cpu().numpy().squeeze(0)    # (10,)

    def predict_top(self, grid: np.ndarray) -> tuple[int, float]:
        """Return (predicted_digit, confidence) as (int, float)."""
        probs = self.predict(grid)
        top_idx = int(np.argmax(probs))
        return top_idx, float(probs[top_idx])

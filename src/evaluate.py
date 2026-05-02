"""Test-set evaluation: accuracy, macro-F1, confusion matrix."""

import os
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.utils import save_json


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    model_name: str = "model",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate a trained model on the test set.

    Saves:
        - test_metrics.json
        - confusion_matrix.png

    Returns metrics dict.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None).tolist()
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "model_name": model_name,
        "test_accuracy": round(test_acc, 6),
        "test_macro_f1": round(float(macro_f1), 6),
        "per_class_f1": {str(i): round(v, 4) for i, v in enumerate(per_class_f1)},
    }

    # Save JSON
    json_path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    save_json(metrics, json_path)

    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, model_name, cm_path)

    print(f"\n{model_name} Test Results:")
    print(f"  Accuracy:   {test_acc:.4f}")
    print(f"  Macro F1:   {macro_f1:.4f}")
    print(f"  Saved to:   {json_path}")
    print(f"  Confusion:  {cm_path}")

    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

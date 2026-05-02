"""Utility functions: seed, device, directory creation, IO helpers."""

import os
import json
import csv
import random
import datetime
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cpu") -> torch.device:
    """Return the torch device. Falls back to CPU if CUDA not available."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_timestamped_dir(base_dir: str, prefix: str = "run") -> str:
    """Create a timestamped results directory and return its path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{prefix}_{timestamp}"
    path = os.path.join(base_dir, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: dict, path: str) -> None:
    """Save a dict as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict], path: str) -> None:
    """Save a list of dicts as CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file. Requires PyYAML."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

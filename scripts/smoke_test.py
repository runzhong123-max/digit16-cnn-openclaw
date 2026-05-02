"""Smoke test: verify the full pipeline on a tiny data subset (1 epoch)."""

import os
import sys
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, make_timestamped_dir, load_yaml_config
from src.data import get_mnist_dataloaders
from src.models import get_model
from src.train import train_model
from src.evaluate import evaluate_on_test


def main():
    print("=" * 60)
    print("SMOKE TEST — Digit16 CNN Experiment")
    print("=" * 60)

    # Load config (with overrides for smoke test)
    config = load_yaml_config("configs/default.yaml")

    device = get_device(config.get("device", "cpu"))
    set_seed(config["seed"])

    # Smoke test: use tiny subset, 1 epoch
    batch_size = min(config["batch_size"], 32)
    epochs = 1
    use_subset = True
    subset_size = config.get("debug_subset_size", 128)

    print(f"\nDevice: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Subset size: {subset_size}")

    # Load data
    print("\n[1/4] Loading data (MNIST → 16×16, subset mode)...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        image_size=config["image_size"],
        num_workers=0,
        seed=config["seed"],
        use_subset=use_subset,
        subset_size=subset_size,
    )
    print(
        f"  Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # Create output dir
    output_dir = make_timestamped_dir(config["output_dir"], prefix="smoke_test")

    # Test MLP
    print("\n[2/4] Testing MLP model...")
    mlp = get_model("mlp").to(device)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP params: {n_params:,}")

    # Forward pass check
    dummy = torch.randn(batch_size, 1, 16, 16).to(device)
    out = mlp(dummy)
    assert out.shape == (batch_size, 10), f"MLP output shape error: {out.shape}"
    print(f"  Forward pass OK, output shape: {out.shape}")

    # Train
    train_model(
        model=mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=config["learning_rate"],
        device=device,
        output_dir=output_dir,
        model_name="mlp",
    )

    # Evaluate
    mlp.load_state_dict(
        torch.load(
            os.path.join(output_dir, "mlp_best.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    evaluate_on_test(
        model=mlp,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_name="mlp",
    )

    # Test CNN
    print("\n[3/4] Testing SimpleCNN model...")
    cnn = get_model("simple_cnn").to(device)
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"  CNN params: {n_params:,}")

    # Forward pass check
    out = cnn(dummy)
    assert out.shape == (batch_size, 10), f"CNN output shape error: {out.shape}"
    print(f"  Forward pass OK, output shape: {out.shape}")

    # Train
    train_model(
        model=cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=config["learning_rate"],
        device=device,
        output_dir=output_dir,
        model_name="simple_cnn",
    )

    # Evaluate
    cnn.load_state_dict(
        torch.load(
            os.path.join(output_dir, "simple_cnn_best.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    evaluate_on_test(
        model=cnn,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_name="simple_cnn",
    )

    # Summary
    print("\n" + "=" * 60)
    print("[4/4] SMOKE TEST PASSED ✓")
    print(f"Results saved to: {output_dir}")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

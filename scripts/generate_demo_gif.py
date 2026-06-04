"""
Generate a demo GIF showing model predictions on test samples.
Run: python scripts/generate_demo_gif.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

from src.data import get_mnist_dataloaders as get_data_loaders
from src.models import SimpleCNN, MLP
from src.utils import set_seed

def generate_gif(model_name='simple_cnn', num_samples=10, output_path='docs/figures/demo.gif'):
    set_seed(42)
    device = torch.device('cpu')

    # Load model
    results_root = 'results'
    import glob, re
    exp_dirs = sorted(glob.glob(f'{results_root}/experiment_*'))
    if not exp_dirs:
        print("No experiment results found. Run scripts/run_experiment.py first.")
        return

    latest = exp_dirs[-1]
    model_dir = f'{latest}/{model_name}'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_best.pt')]
    if not model_files:
        print(f"No model weights found in {model_dir}")
        return

    model_path = f'{model_dir}/{model_files[0]}'
    state = torch.load(model_path, map_location='cpu', weights_only=False)

    if model_name == 'mlp':
        model = MLP()
    else:
        model = SimpleCNN()

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Get test samples
    _, _, test_loader = get_data_loaders(batch_size=64, image_size=16)
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if len(torch.cat(all_images)) >= num_samples:
            break

    images = torch.cat(all_images)[:num_samples]
    labels = torch.cat(all_labels)[:num_samples]

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    def update(frame):
        ax = axes[frame]
        ax.clear()

        img = images[frame].squeeze().numpy()
        label = labels[frame].item()

        with torch.no_grad():
            logits = model(images[frame].unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = np.argmax(probs)

        color = 'green' if pred == label else 'red'
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'True: {label} | Pred: {pred}\nConf: {probs[pred]:.2%}',
                     color=color, fontsize=10)
        ax.axis('off')

        # Small bar chart at bottom
        bar_ax = ax.inset_axes([0.0, -0.35, 1.0, 0.3])
        bar_ax.bar(range(10), probs, color=['green' if i == pred else 'gray' for i in range(10)], alpha=0.7)
        bar_ax.set_xticks(range(10))
        bar_ax.set_ylim(0, 1)
        bar_ax.set_title('Probabilities', fontsize=8)

    for i in range(num_samples):
        update(i)

    plt.tight_layout()
    os.makedirs('docs/figures', exist_ok=True)
    plt.savefig('docs/figures/demo_predictions.png', dpi=150, bbox_inches='tight')
    print(f"Saved demo_predictions.png")

    # Simple GIF using saved frames
    from PIL import Image
    frames_list = []
    for i in range(num_samples):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        img = images[i].squeeze().numpy()
        label = labels[i].item()
        with torch.no_grad():
            logits = model(images[i].unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = np.argmax(probs)

        color = 'green' if pred == label else 'red'
        ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Input 16×16', fontsize=10)
        ax1.axis('off')

        bars = ax2.bar(range(10), probs, color=['#2ecc71' if i == pred else '#bdc3c7' for i in range(10)])
        ax2.set_xticks(range(10))
        ax2.set_ylim(0, 1)
        ax2.set_title(f'Prediction: {pred}  (True: {label})', color=color, fontsize=11)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Confidence')

        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{prob:.1%}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        frame_path = f'/tmp/demo_frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=120)
        frames_list.append(Image.open(frame_path))
        plt.close(fig)

    frames_list[0].save(output_path, save_all=True, append_images=frames_list[1:],
                        duration=1500, loop=0)
    print(f"Saved {output_path} ({len(frames_list)} frames)")

    for f in os.listdir('/tmp'):
        if f.startswith('demo_frame_'):
            os.remove(f'/tmp/{f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple_cnn', choices=['simple_cnn', 'mlp'])
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    generate_gif(args.model, args.samples)

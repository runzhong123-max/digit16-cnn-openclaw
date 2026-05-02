#!/usr/bin/env python3
"""Matplotlib-based 16×16 hand-drawn digit recognizer demo.

Draw digits on a 16×16 pixel grid with the mouse,
then predict the digit using a trained SimpleCNN.

Uses only matplotlib (already in requirements.txt) — no Tcl/Tk needed.

Usage:
    python demo/matplotlib_digit_demo.py
    python demo/matplotlib_digit_demo.py --model-path results/.../simple_cnn_best.pt
    python demo/matplotlib_digit_demo.py --brush-size 2
"""

import argparse
import os
import sys
import glob
import datetime

import numpy as np
import matplotlib
matplotlib.use("MacOSX")  # macOS native Cocoa backend (no Tcl/Tk needed)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.infer import DigitInferer

# ── Constants ────────────────────────────────────────────────────
GRID_SIZE = 16
CELL_SIZE = 1.0  # logical cell size in axes coords
CANVAS_SIZE = GRID_SIZE * CELL_SIZE

DRAW_COLOR = "#1a1a1a"
BG_COLOR = "#f5f5f5"
GRID_LINE_COLOR = "#cccccc"


# ── App ──────────────────────────────────────────────────────────


class DigitDrawerApp:
    def __init__(self, model_path: str, brush_size: int = 1):
        self.brush_size = brush_size
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Load model
        self._init_model(model_path)

        # Build figure
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("16×16 Digit Recognizer")

        # Disable default navigation (pan/zoom) so left-drag draws instead
        toolbar = self.fig.canvas.manager.toolbar
        if toolbar is not None:
            toolbar.mode = ''  # neutral mode — no pan, no zoom

        # Layout: [grid_area | prob_area]
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.2],
                                    left=0.05, right=0.95,
                                    top=0.92, bottom=0.08, wspace=0.3)

        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_prob = self.fig.add_subplot(gs[0, 1])

        self._build_grid()
        self._build_prob_panel()
        self._build_buttons()

        # Mouse state
        self._drawing = False
        self._erasing = False

        # Disconnect matplotlib's default keybindings that conflict with ours
        for key in ('keymap.pan', 'keymap.zoom', 'keymap.all_axes'):
            if key in plt.rcParams:
                plt.rcParams[key] = ''  # clear default shortcuts

        # Connect our custom event handlers
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Instructions
        self.fig.suptitle(
            "Left-drag to draw  |  Right-drag to erase  |  c = clear  |  p = predict  |  s = save",
            fontsize=10, color="#666666", y=0.97,
        )

        plt.show()

    # ── Model ─────────────────────────────────────────────────

    def _init_model(self, model_path: str):
        if not os.path.isfile(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print("Please train SimpleCNN first or specify --model-path.")
            sys.exit(1)
        try:
            self.inferer = DigitInferer(model_path, device="cpu")
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            sys.exit(1)

    # ── Grid canvas ──────────────────────────────────────────

    def _build_grid(self):
        self.ax_grid.set_xlim(0, CANVAS_SIZE)
        self.ax_grid.set_ylim(0, CANVAS_SIZE)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.invert_yaxis()  # row 0 at top
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.set_title("Draw a digit (0–9)", fontsize=13, pad=8)

        # Draw grid cells
        self._cell_patches = {}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = Rectangle(
                    (c * CELL_SIZE, r * CELL_SIZE),
                    CELL_SIZE, CELL_SIZE,
                    linewidth=0.3,
                    edgecolor=GRID_LINE_COLOR,
                    facecolor=BG_COLOR,
                )
                self.ax_grid.add_patch(rect)
                self._cell_patches[(r, c)] = rect

    # ── Probability panel ────────────────────────────────────

    def _build_prob_panel(self):
        self.ax_prob.set_xlim(0, 10)
        self.ax_prob.set_ylim(0, 100)
        self.ax_prob.set_xticks(range(10))
        self.ax_prob.set_xticklabels([str(i) for i in range(10)])
        self.ax_prob.set_xlabel("Digit", fontsize=11)
        self.ax_prob.set_ylabel("Probability (%)", fontsize=11)
        self.ax_prob.set_title("Prediction", fontsize=13, pad=8)

        self._bars = self.ax_prob.bar(
            range(10), [0] * 10, color="#cccccc", edgecolor="#aaaaaa", linewidth=0.5
        )

        # Annotation text for top-1
        self._ann_text = self.ax_prob.text(
            5, 95, "", ha="center", va="top",
            fontsize=14, fontweight="bold", color="#2196F3",
        )

    # ── Buttons (clickable text patches) ─────────────────────

    def _build_buttons(self):
        # Place buttons as text below the grid
        btn_y = -1.5
        spacing = 2.5

        self._btn_predict = self.ax_grid.text(
            0 * spacing, btn_y, "[Predict]", fontsize=10,
            color="#4CAF50", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#4CAF50"),
            picker=True,
        )
        self._btn_clear = self.ax_grid.text(
            1 * spacing, btn_y, "[Clear]", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE", edgecolor="#999999"),
            picker=True,
        )
        self._btn_save = self.ax_grid.text(
            2 * spacing, btn_y, "[Save]", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE", edgecolor="#999999"),
            picker=True,
        )

    # ── Drawing logic ─────────────────────────────────────────

    def _cell_from_event(self, event):
        """Convert mouse event to (row, col) in the grid axes."""
        if event.inaxes != self.ax_grid:
            return None
        c = int(event.xdata / CELL_SIZE)
        r = int(event.ydata / CELL_SIZE)
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None

    def _fill_cells(self, r, c, value):
        half = self.brush_size // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    self.grid[rr, cc] = value
                    color = DRAW_COLOR if value > 0.5 else BG_COLOR
                    self._cell_patches[(rr, cc)].set_facecolor(color)

    def _on_press(self, event):
        # Check if a button was clicked
        if hasattr(event, "artist") and event.artist is not None:
            self._handle_button(event.artist)
            return

        cell = self._cell_from_event(event)
        if cell is None:
            return

        if event.button is MouseButton.LEFT:
            self._drawing = True
            self._fill_cells(*cell, 1.0)
        elif event.button is MouseButton.RIGHT:
            self._erasing = True
            self._fill_cells(*cell, 0.0)

    def _on_motion(self, event):
        cell = self._cell_from_event(event)
        if cell is None:
            return
        if self._drawing:
            self._fill_cells(*cell, 1.0)
            self.fig.canvas.draw_idle()
        elif self._erasing:
            self._fill_cells(*cell, 0.0)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._drawing = False
        self._erasing = False

    def _on_key(self, event):
        if event.key == "c":
            self._clear()
        elif event.key == "p":
            self._predict()
        elif event.key == "s":
            self._save_matrix()

    def _handle_button(self, artist):
        if artist == self._btn_predict:
            self._predict()
        elif artist == self._btn_clear:
            self._clear()
        elif artist == self._btn_save:
            self._save_matrix()

    # ── Actions ───────────────────────────────────────────────

    def _predict(self):
        if self.grid.max() == 0.0:
            self._ann_text.set_text("Draw something first!")
            self.fig.canvas.draw_idle()
            return

        probs = self.inferer.predict(self.grid)
        top_digit = int(np.argmax(probs))
        top_conf = float(probs[top_digit])

        # Update bars
        for d in range(10):
            p = float(probs[d]) * 100
            self._bars[d].set_height(p)
            color = "#4CAF50" if d == top_digit else "#90CAF9"
            self._bars[d].set_facecolor(color)

        # Update annotation
        self._ann_text.set_text(
            f"Predicted: {top_digit}\nConfidence: {top_conf * 100:.2f}%"
        )

        # Update title
        self.ax_prob.set_title(
            f"Prediction: {top_digit}  ({top_conf * 100:.1f}%)",
            fontsize=13, pad=8, color="#2196F3",
        )

        self.fig.canvas.draw_idle()

    def _clear(self):
        self.grid.fill(0.0)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._cell_patches[(r, c)].set_facecolor(BG_COLOR)

        # Reset bars
        for bar in self._bars:
            bar.set_height(0)
            bar.set_facecolor("#cccccc")
        self._ann_text.set_text("")
        self.ax_prob.set_title("Prediction", fontsize=13, pad=8, color="black")

        self.fig.canvas.draw_idle()

    def _save_matrix(self):
        """Save grid to a timestamped file in current directory."""
        path = f"digit16_grid_{datetime.datetime.now():%Y%m%d_%H%M%S}.npy"
        np.save(path, self.grid)
        print(f"Saved: {path}")
        self._ann_text.set_text(f"Saved: {path}")
        self.fig.canvas.draw_idle()


# ── Model auto-discovery ─────────────────────────────────────


def _find_latest_model() -> str:
    pattern = os.path.join(
        _PROJECT_ROOT, "results", "experiment_*", "simple_cnn", "simple_cnn_best.pt"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else ""


# ── Entry point ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="16×16 Digit Recognizer (matplotlib GUI)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to SimpleCNN weights. Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--brush-size", type=int, default=1, choices=[1, 2, 3],
        help="Brush size (default: 1).",
    )
    args = parser.parse_args()

    model_path = args.model_path or _find_latest_model()
    if not model_path:
        print("ERROR: No model weights found.")
        print("Run: python scripts/run_experiment.py  first.")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Brush size: {args.brush_size}")
    print("Shortcuts: Left-drag=draw | Right-drag=erase | c=clear | p=predict | s=save")

    DigitDrawerApp(model_path=model_path, brush_size=args.brush_size)


if __name__ == "__main__":
    main()

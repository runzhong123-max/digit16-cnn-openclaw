#!/usr/bin/env python3
"""Tkinter-based 16×16 hand-drawn digit recognizer demo.

Draw digits on a 16×16 pixel grid with the mouse,
then predict the digit using a trained SimpleCNN.

Usage:
    # Use default model path
    python demo/tkinter_digit_demo.py

    # Specify model path
    python demo/tkinter_digit_demo.py --model-path results/experiment_.../simple_cnn/simple_cnn_best.pt

    # Change brush size
    python demo/tkinter_digit_demo.py --brush-size 2

Defaults:
    Model path: auto-discovers the most recent simple_cnn_best.pt in results/
    Brush size: 1
"""

import argparse
import os
import sys
import glob
from tkinter import Tk, Canvas, Frame, Button, Label, StringVar, LEFT, RIGHT, TOP, BOTH, X, Y, YES, BOTH as FILL_BOTH

import numpy as np

# Ensure project root is on sys.path so src imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.infer import DigitInferer


# ---- Constants ----
GRID_SIZE = 16           # logical pixels
CELL_PX = 28             # screen pixels per cell (including 1px gap)
GAP_PX = 1               # gap between cells
CANVAS_SIZE = GRID_SIZE * CELL_PX + 1  # total canvas pixels

DRAW_COLOR = "#000000"   # filled cell color
BG_COLOR = "#FFFFFF"     # empty cell color
GRID_COLOR = "#CCCCCC"   # grid line color

# ── GUI Application ──────────────────────────────────────────────


class DigitDrawerApp:
    def __init__(
        self,
        root: Tk,
        model_path: str,
        brush_size: int = 1,
    ):
        self.root = root
        self.root.title("16×16 Digit Recognizer")
        self.root.resizable(False, False)

        self.brush_size = brush_size
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Load inference engine
        self._init_model(model_path)

        # Build UI
        self._build_canvas()
        self._build_controls()
        self._build_results()

        # Mouse state
        self._drawing = False
        self._erasing = False  # right-click erase

    # ── Model loading ────────────────────────────────────────────

    def _init_model(self, model_path: str):
        if not os.path.isfile(model_path):
            print(f"Error: Model file not found at '{model_path}'")
            print("Please train SimpleCNN first or specify a valid --model-path.")
            print("Example: python demo/tkinter_digit_demo.py "
                  "--model-path results/experiment_.../simple_cnn/simple_cnn_best.pt")
            sys.exit(1)

        try:
            self.inferer = DigitInferer(model_path, device="cpu")
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model from '{model_path}': {e}")
            sys.exit(1)

    # ── Canvas (16×16 grid) ──────────────────────────────────────

    def _build_canvas(self):
        frame = Frame(self.root)
        frame.pack(padx=10, pady=10)

        self.canvas = Canvas(
            frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack()

        # Store rectangle ids keyed by (row, col)
        self._cells = {}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = c * CELL_PX + GAP_PX
                y1 = r * CELL_PX + GAP_PX
                x2 = x1 + CELL_PX - GAP_PX
                y2 = y1 + CELL_PX - GAP_PX
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=BG_COLOR,
                    outline=GRID_COLOR,
                    width=0,
                )
                self._cells[(r, c)] = rect

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<Button-2>", self._on_right_down)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_right_up)

        # macOS right-click via Control+click
        self.canvas.bind("<Control-Button-1>", self._on_right_down)
        self.canvas.bind("<Control-B1-Motion>", self._on_right_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self._on_right_up)

    # ── Controls ─────────────────────────────────────────────────

    def _build_controls(self):
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=(0, 5))

        Button(btn_frame, text="Predict", command=self._predict,
               width=10, bg="#4CAF50", fg="white").pack(side=LEFT, padx=4)
        Button(btn_frame, text="Clear", command=self._clear,
               width=10).pack(side=LEFT, padx=4)
        Button(btn_frame, text="Save Matrix", command=self._save_matrix,
               width=10).pack(side=LEFT, padx=4)
        Button(btn_frame, text="Quit", command=self.root.destroy,
               width=10).pack(side=LEFT, padx=4)

    # ── Results display ──────────────────────────────────────────

    def _build_results(self):
        result_frame = Frame(self.root)
        result_frame.pack(padx=10, pady=(0, 10), fill=X)

        self._pred_label = Label(
            result_frame,
            text="Prediction: —",
            font=("Helvetica", 18, "bold"),
        )
        self._pred_label.pack()

        self._conf_label = Label(
            result_frame,
            text="Confidence: —",
            font=("Helvetica", 14),
        )
        self._conf_label.pack(pady=(2, 8))

        # Per-class probabilities grid
        prob_frame = Frame(result_frame)
        prob_frame.pack()

        self._prob_labels = []
        self._prob_bars = []
        self._prob_vals = []

        # Two columns of 5 digits each
        for col in range(2):
            col_frame = Frame(prob_frame)
            col_frame.pack(side=LEFT, padx=10)

            for i in range(5):
                digit = col * 5 + i
                row_frame = Frame(col_frame)
                row_frame.pack(anchor="w", pady=1)

                lbl = Label(row_frame, text=f"{digit}:", width=3, anchor="e",
                            font=("Courier", 12))
                lbl.pack(side=LEFT)

                bar = Canvas(row_frame, width=120, height=14, bg="#EEEEEE",
                             highlightthickness=0)
                bar.pack(side=LEFT, padx=(4, 4))

                val = Label(row_frame, text="0.00%", width=8, anchor="e",
                            font=("Courier", 11))
                val.pack(side=LEFT)

                self._prob_labels.append(lbl)
                self._prob_bars.append(bar)
                self._prob_vals.append(val)

    # ── Drawing logic ─────────────────────────────────────────────

    def _cell_from_xy(self, event_x, event_y):
        """Convert canvas pixel coordinates to (row, col). Returns None if outside."""
        c = event_x // CELL_PX
        r = event_y // CELL_PX
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None

    def _fill_cells(self, r, c, value: float):
        """Fill cells in brush_size×brush_size area centered at (r, c)."""
        half = self.brush_size // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    self.grid[rr, cc] = value
                    color = DRAW_COLOR if value > 0.5 else BG_COLOR
                    self.canvas.itemconfig(self._cells[(rr, cc)], fill=color)

    def _on_left_down(self, event):
        self._drawing = True
        cell = self._cell_from_xy(event.x, event.y)
        if cell:
            self._fill_cells(*cell, 1.0)

    def _on_left_drag(self, event):
        if not self._drawing:
            return
        cell = self._cell_from_xy(event.x, event.y)
        if cell:
            self._fill_cells(*cell, 1.0)

    def _on_left_up(self, event):
        self._drawing = False

    def _on_right_down(self, event):
        self._erasing = True
        cell = self._cell_from_xy(event.x, event.y)
        if cell:
            self._fill_cells(*cell, 0.0)

    def _on_right_drag(self, event):
        if not self._erasing:
            return
        cell = self._cell_from_xy(event.x, event.y)
        if cell:
            self._fill_cells(*cell, 0.0)

    def _on_right_up(self, event):
        self._erasing = False

    # ── Actions ───────────────────────────────────────────────────

    def _predict(self):
        if self.grid.max() == 0.0:
            self._pred_label.config(text="Prediction: (draw something first)")
            self._conf_label.config(text="Confidence: —")
            self._clear_probs()
            return

        probs = self.inferer.predict(self.grid)
        top_digit = int(np.argmax(probs))
        top_conf = float(probs[top_digit])

        self._pred_label.config(
            text=f"Prediction: {top_digit}",
            fg="#2196F3",
        )
        self._conf_label.config(
            text=f"Confidence: {top_conf * 100:.2f}%",
        )

        # Update per-class bars and labels
        for digit in range(10):
            p = float(probs[digit])
            # Bar fill
            bar_width = int(p * 118)
            self._prob_bars[digit].delete("all")
            if bar_width > 0:
                color = "#4CAF50" if digit == top_digit else "#90CAF9"
                self._prob_bars[digit].create_rectangle(
                    0, 0, bar_width, 14, fill=color, outline="",
                )
            # Value text
            self._prob_vals[digit].config(text=f"{p * 100:6.2f}%")

            # Bold the top-1 digit label
            if digit == top_digit:
                self._prob_labels[digit].config(font=("Courier", 12, "bold"), fg="#2196F3")
            else:
                self._prob_labels[digit].config(font=("Courier", 12), fg="black")

    def _clear(self):
        self.grid.fill(0.0)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.canvas.itemconfig(self._cells[(r, c)], fill=BG_COLOR)
        self._pred_label.config(text="Prediction: —", fg="black")
        self._conf_label.config(text="Confidence: —")
        self._clear_probs()

    def _clear_probs(self):
        for digit in range(10):
            self._prob_bars[digit].delete("all")
            self._prob_vals[digit].config(text="0.00%")
            self._prob_labels[digit].config(font=("Courier", 12), fg="black")

    def _save_matrix(self):
        """Save the current 16×16 grid as a numpy .npy file."""
        from tkinter import filedialog
        import datetime

        default_name = f"digit16_grid_{datetime.datetime.now():%Y%m%d_%H%M%S}.npy"
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if path:
            np.save(path, self.grid)
            print(f"Grid saved to: {path}")


# ── Model path auto-discovery ────────────────────────────────────


def _find_latest_model() -> str:
    """Find the most recent simple_cnn_best.pt in results/ directory."""
    pattern = os.path.join(
        _PROJECT_ROOT, "results", "experiment_*", "simple_cnn", "simple_cnn_best.pt"
    )
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return ""


# ── Entry point ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="16×16 hand-drawn digit recognizer (Tkinter GUI)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained SimpleCNN weights (.pt). "
             "Auto-discovers latest in results/ if omitted.",
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Brush size in logical pixels (default: 1).",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = _find_latest_model()

    if not model_path:
        print("ERROR: No model weights found.")
        print("Auto-discovery searched: results/experiment_*/simple_cnn/simple_cnn_best.pt")
        print("\nPlease either:")
        print("  1) Run 'python scripts/run_experiment.py' to train a model, or")
        print("  2) Specify a model path: "
              "python demo/tkinter_digit_demo.py --model-path <path>")
        sys.exit(1)

    print(f"Using model: {model_path}")
    print(f"Brush size: {args.brush_size}")
    print("\nInstructions:")
    print("  • Left-click + drag to draw")
    print("  • Right-click (or Ctrl+click) + drag to erase")
    print("  • Click 'Predict' to classify")
    print("  • Click 'Clear' to reset the canvas")
    print("  • Click 'Save Matrix' to export the 16×16 grid as .npy")

    root = Tk()
    app = DigitDrawerApp(
        root=root,
        model_path=model_path,
        brush_size=args.brush_size,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

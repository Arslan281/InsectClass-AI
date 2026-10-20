r"""
plot_training.py - accuracy and loss curves (train vs validation) for every
*_history.json found in results\metrics (improved runs) and original_models (your copies).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import METRICS_DIR, ORIG_DIR, PLOTS_DIR

files = sorted(METRICS_DIR.glob("*_history.json")) + sorted(ORIG_DIR.glob("*history*.json"))
if not files:
    raise SystemExit("No history files found.")

for path in files:
    with open(path) as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, h["accuracy"], label="Training")
    axes[0].plot(epochs, h["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, h["loss"], label="Training")
    axes[1].plot(epochs, h["val_loss"], label="Validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.3)

    prefix = "original_" if path.parent == ORIG_DIR else ""
    name = path.stem.replace("_history", "")
    fig.suptitle(f"{prefix}{name}")
    fig.tight_layout()
    out = PLOTS_DIR / f"{prefix}{name}_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

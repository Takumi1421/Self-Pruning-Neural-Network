from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "cover_image.png"


def add_text(ax, x, y, text, size=14, weight="normal", color="#111827", ha="left", va="top"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        family="DejaVu Sans",
    )


def card(ax, x, y, w, h, fc, ec="#d1d5db"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)


def main():
    ASSETS.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14, 7.2), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#f8fafc")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f8fafc"))
    ax.add_patch(plt.Circle((0.88, 0.82), 0.22, color="#dbeafe", alpha=0.9))
    ax.add_patch(plt.Circle((0.10, 0.15), 0.18, color="#dcfce7", alpha=0.9))

    add_text(ax, 0.07, 0.88, "Self-Pruning Neural Network", size=28, weight="bold")
    add_text(ax, 0.07, 0.81, "CIFAR-10 lambda sweep on Apple Silicon", size=16, color="#475569")

    add_text(
        ax,
        0.07,
        0.70,
        "Measured 10-epoch experiment with a custom PrunableLinear layer.\n"
        "Result: stable accuracy near 78%, but no effective pruning yet.",
        size=16,
        color="#1f2937",
    )

    card(ax, 0.07, 0.30, 0.22, 0.18, "#ffffff")
    add_text(ax, 0.10, 0.44, "Best Accuracy", size=12, weight="bold", color="#64748b")
    add_text(ax, 0.10, 0.39, "78.52%", size=24, weight="bold", color="#0f766e")
    add_text(ax, 0.10, 0.34, "lambda = 1e-7", size=13, color="#334155")

    card(ax, 0.33, 0.30, 0.22, 0.18, "#ffffff")
    add_text(ax, 0.36, 0.44, "Best Sparsity", size=12, weight="bold", color="#64748b")
    add_text(ax, 0.36, 0.39, "0.0%", size=24, weight="bold", color="#7c3aed")
    add_text(ax, 0.36, 0.34, "threshold = 0.01", size=13, color="#334155")

    card(ax, 0.59, 0.30, 0.27, 0.18, "#ffffff")
    add_text(ax, 0.62, 0.44, "Sweep", size=12, weight="bold", color="#64748b")
    add_text(ax, 0.62, 0.39, "0, 1e-7, 1e-6, 1e-5, 5e-5", size=16, weight="bold", color="#1f2937")
    add_text(ax, 0.62, 0.34, "10 epochs on MPS", size=13, color="#334155")

    add_text(ax, 0.07, 0.18, "Generated from measured run results in this repository", size=12, color="#64748b")

    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

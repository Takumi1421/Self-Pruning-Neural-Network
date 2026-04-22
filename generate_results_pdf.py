from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_PDF = Path("self_pruning_results_report.pdf")
ASSETS_DIR = Path("assets")

SWEEP_ROWS = [
    ["0", "78.48%", "0.0%"],
    ["1e-7", "78.52%", "0.0%"],
    ["1e-6", "78.14%", "0.0%"],
    ["1e-5", "78.11%", "0.0%"],
    ["5e-5", "78.19%", "0.0%"],
]


def add_text(ax, x, y, text, size=12, weight="normal", color="#111111", ha="left"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va="top",
        family="DejaVu Sans",
    )


def build_summary_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor("white")

    add_text(ax, 0.08, 0.95, "Self-Pruning CIFAR-10 Report", size=24, weight="bold")
    add_text(ax, 0.08, 0.915, "Measured 10-epoch lambda sweep on Apple Silicon", size=12, color="#555555")

    ax.add_patch(plt.Rectangle((0.08, 0.79), 0.84, 0.09, color="#f4f7fb", ec="#d8e0ea"))
    add_text(ax, 0.11, 0.855, "Best Accuracy", size=12, weight="bold", color="#4a5568")
    add_text(ax, 0.11, 0.823, "78.52%", size=26, weight="bold")
    add_text(ax, 0.39, 0.855, "Best Lambda", size=12, weight="bold", color="#4a5568")
    add_text(ax, 0.39, 0.823, "1e-7", size=26, weight="bold")
    add_text(ax, 0.63, 0.855, "Best Sparsity", size=12, weight="bold", color="#4a5568")
    add_text(ax, 0.63, 0.823, "0.0%", size=26, weight="bold")

    add_text(ax, 0.08, 0.73, "Sweep Configuration", size=16, weight="bold")
    lines = [
        "Dataset: CIFAR-10",
        "Device: mps",
        "Epochs per run: 10",
        "Batch size: 128",
        "Lambdas tested: 0, 1e-7, 1e-6, 1e-5, 5e-5",
    ]
    y = 0.69
    for line in lines:
        add_text(ax, 0.10, y, f"- {line}", size=12)
        y -= 0.035

    add_text(ax, 0.08, 0.49, "Main Finding", size=16, weight="bold")
    finding_lines = [
        "All runs reached about 78% test accuracy after 10 epochs.",
        "No lambda produced measurable sparsity under the 0.01 pruning threshold.",
        "This sweep demonstrates stable classification learning, but not successful pruning yet.",
    ]
    y = 0.45
    for line in finding_lines:
        add_text(ax, 0.10, y, f"- {line}", size=12)
        y -= 0.05

    add_text(ax, 0.08, 0.25, "Sweep Results", size=16, weight="bold")
    table = ax.table(
        cellText=SWEEP_ROWS,
        colLabels=["Lambda", "Test Accuracy", "Sparsity Level (%)"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.08, 0.07, 0.84, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cfd8e3")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f9fbfd")
        else:
            cell.set_facecolor("#ffffff")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_graphs_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    add_text(ax, 0.08, 0.95, "Essential Graphs", size=22, weight="bold")
    add_text(ax, 0.08, 0.915, "Visual summary of the measured lambda sweep", size=12, color="#555555")

    lambda_sweep = ASSETS_DIR / "lambda_sweep.png"
    best_run = ASSETS_DIR / "best_run_overview.png"

    if lambda_sweep.exists():
        ax1 = fig.add_axes([0.08, 0.49, 0.84, 0.30])
        ax1.imshow(plt.imread(lambda_sweep))
        ax1.axis("off")

    if best_run.exists():
        ax2 = fig.add_axes([0.08, 0.11, 0.84, 0.28])
        ax2.imshow(plt.imread(best_run))
        ax2.axis("off")

    add_text(
        ax,
        0.08,
        0.43,
        "Accuracy stays tightly grouped across lambda values, while sparsity remains flat at 0.0% for every run.",
        size=11.5,
    )
    add_text(
        ax,
        0.08,
        0.06,
        "The best run by accuracy is lambda = 1e-7, but the sweep overall suggests that 10 epochs is not enough to induce pruning.",
        size=11.5,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    with PdfPages(OUTPUT_PDF) as pdf:
        build_summary_page(pdf)
        build_graphs_page(pdf)
    print(f"Created {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()

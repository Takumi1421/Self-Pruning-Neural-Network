import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
RESULTS_DIR = ROOT / "results"

SWEEP_RESULTS = [
    {"lambda": 0.0, "accuracy_percent": 78.48, "sparsity_percent": 0.0},
    {"lambda": 1e-7, "accuracy_percent": 78.52, "sparsity_percent": 0.0},
    {"lambda": 1e-6, "accuracy_percent": 78.14, "sparsity_percent": 0.0},
    {"lambda": 1e-5, "accuracy_percent": 78.11, "sparsity_percent": 0.0},
    {"lambda": 5e-5, "accuracy_percent": 78.19, "sparsity_percent": 0.0},
]

BEST_LAMBDA = 1e-7

HISTORIES = {
    "0.0": [
        {"epoch": 1, "loss": 1.4536, "accuracy_percent": 60.67, "sparsity_percent": 0.0, "time_seconds": 21.3},
        {"epoch": 2, "loss": 1.1246, "accuracy_percent": 65.39, "sparsity_percent": 0.0, "time_seconds": 20.2},
        {"epoch": 3, "loss": 1.0210, "accuracy_percent": 69.92, "sparsity_percent": 0.0, "time_seconds": 19.4},
        {"epoch": 4, "loss": 0.9548, "accuracy_percent": 72.36, "sparsity_percent": 0.0, "time_seconds": 18.1},
        {"epoch": 5, "loss": 0.8960, "accuracy_percent": 72.01, "sparsity_percent": 0.0, "time_seconds": 16.8},
        {"epoch": 6, "loss": 0.8471, "accuracy_percent": 75.24, "sparsity_percent": 0.0, "time_seconds": 16.0},
        {"epoch": 7, "loss": 0.8006, "accuracy_percent": 76.65, "sparsity_percent": 0.0, "time_seconds": 16.5},
        {"epoch": 8, "loss": 0.7596, "accuracy_percent": 77.47, "sparsity_percent": 0.0, "time_seconds": 17.6},
        {"epoch": 9, "loss": 0.7293, "accuracy_percent": 78.29, "sparsity_percent": 0.0, "time_seconds": 18.2},
        {"epoch": 10, "loss": 0.7140, "accuracy_percent": 78.48, "sparsity_percent": 0.0, "time_seconds": 22.4},
    ],
    "1e-07": [
        {"epoch": 1, "loss": 1.4910, "accuracy_percent": 58.85, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 2, "loss": 1.1616, "accuracy_percent": 66.76, "sparsity_percent": 0.0, "time_seconds": 22.2},
        {"epoch": 3, "loss": 1.0384, "accuracy_percent": 69.99, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 4, "loss": 0.9621, "accuracy_percent": 71.29, "sparsity_percent": 0.0, "time_seconds": 22.3},
        {"epoch": 5, "loss": 0.8905, "accuracy_percent": 73.47, "sparsity_percent": 0.0, "time_seconds": 19.0},
        {"epoch": 6, "loss": 0.8462, "accuracy_percent": 74.86, "sparsity_percent": 0.0, "time_seconds": 16.0},
        {"epoch": 7, "loss": 0.7997, "accuracy_percent": 76.84, "sparsity_percent": 0.0, "time_seconds": 16.4},
        {"epoch": 8, "loss": 0.7579, "accuracy_percent": 77.86, "sparsity_percent": 0.0, "time_seconds": 15.6},
        {"epoch": 9, "loss": 0.7334, "accuracy_percent": 78.26, "sparsity_percent": 0.0, "time_seconds": 15.8},
        {"epoch": 10, "loss": 0.7103, "accuracy_percent": 78.52, "sparsity_percent": 0.0, "time_seconds": 15.9},
    ],
    "1e-06": [
        {"epoch": 1, "loss": 1.4605, "accuracy_percent": 61.59, "sparsity_percent": 0.0, "time_seconds": 15.9},
        {"epoch": 2, "loss": 1.1303, "accuracy_percent": 66.00, "sparsity_percent": 0.0, "time_seconds": 15.9},
        {"epoch": 3, "loss": 1.0263, "accuracy_percent": 69.13, "sparsity_percent": 0.0, "time_seconds": 16.2},
        {"epoch": 4, "loss": 0.9554, "accuracy_percent": 72.71, "sparsity_percent": 0.0, "time_seconds": 16.0},
        {"epoch": 5, "loss": 0.8940, "accuracy_percent": 72.63, "sparsity_percent": 0.0, "time_seconds": 15.9},
        {"epoch": 6, "loss": 0.8367, "accuracy_percent": 74.66, "sparsity_percent": 0.0, "time_seconds": 16.1},
        {"epoch": 7, "loss": 0.7932, "accuracy_percent": 75.86, "sparsity_percent": 0.0, "time_seconds": 16.4},
        {"epoch": 8, "loss": 0.7600, "accuracy_percent": 77.33, "sparsity_percent": 0.0, "time_seconds": 15.8},
        {"epoch": 9, "loss": 0.7345, "accuracy_percent": 78.11, "sparsity_percent": 0.0, "time_seconds": 17.9},
        {"epoch": 10, "loss": 0.7084, "accuracy_percent": 78.14, "sparsity_percent": 0.0, "time_seconds": 23.1},
    ],
    "1e-05": [
        {"epoch": 1, "loss": 1.4681, "accuracy_percent": 60.22, "sparsity_percent": 0.0, "time_seconds": 22.0},
        {"epoch": 2, "loss": 1.1362, "accuracy_percent": 68.42, "sparsity_percent": 0.0, "time_seconds": 22.2},
        {"epoch": 3, "loss": 1.0244, "accuracy_percent": 69.18, "sparsity_percent": 0.0, "time_seconds": 23.1},
        {"epoch": 4, "loss": 0.9476, "accuracy_percent": 72.39, "sparsity_percent": 0.0, "time_seconds": 22.7},
        {"epoch": 5, "loss": 0.8909, "accuracy_percent": 73.23, "sparsity_percent": 0.0, "time_seconds": 22.4},
        {"epoch": 6, "loss": 0.8407, "accuracy_percent": 75.14, "sparsity_percent": 0.0, "time_seconds": 22.0},
        {"epoch": 7, "loss": 0.7933, "accuracy_percent": 76.39, "sparsity_percent": 0.0, "time_seconds": 21.8},
        {"epoch": 8, "loss": 0.7592, "accuracy_percent": 77.12, "sparsity_percent": 0.0, "time_seconds": 22.2},
        {"epoch": 9, "loss": 0.7362, "accuracy_percent": 78.05, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 10, "loss": 0.7154, "accuracy_percent": 78.11, "sparsity_percent": 0.0, "time_seconds": 21.9},
    ],
    "5e-05": [
        {"epoch": 1, "loss": 1.4737, "accuracy_percent": 62.95, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 2, "loss": 1.1386, "accuracy_percent": 66.44, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 3, "loss": 1.0210, "accuracy_percent": 70.59, "sparsity_percent": 0.0, "time_seconds": 22.4},
        {"epoch": 4, "loss": 0.9548, "accuracy_percent": 71.01, "sparsity_percent": 0.0, "time_seconds": 21.7},
        {"epoch": 5, "loss": 0.8951, "accuracy_percent": 73.76, "sparsity_percent": 0.0, "time_seconds": 22.0},
        {"epoch": 6, "loss": 0.8430, "accuracy_percent": 75.43, "sparsity_percent": 0.0, "time_seconds": 22.7},
        {"epoch": 7, "loss": 0.7968, "accuracy_percent": 76.22, "sparsity_percent": 0.0, "time_seconds": 21.7},
        {"epoch": 8, "loss": 0.7661, "accuracy_percent": 77.47, "sparsity_percent": 0.0, "time_seconds": 21.9},
        {"epoch": 9, "loss": 0.7315, "accuracy_percent": 77.78, "sparsity_percent": 0.0, "time_seconds": 21.7},
        {"epoch": 10, "loss": 0.7151, "accuracy_percent": 78.19, "sparsity_percent": 0.0, "time_seconds": 21.5},
    ],
}


def ensure_dirs():
    ASSETS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def write_sweep_files():
    with (RESULTS_DIR / "lambda_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(SWEEP_RESULTS, f, indent=2)
    with (RESULTS_DIR / "histories_by_lambda.json").open("w", encoding="utf-8") as f:
        json.dump(HISTORIES, f, indent=2)

    with (RESULTS_DIR / "lambda_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "accuracy_percent", "sparsity_percent"])
        writer.writeheader()
        writer.writerows(SWEEP_RESULTS)


def plot_lambda_sweep():
    labels = [str(row["lambda"]) for row in SWEEP_RESULTS]
    accs = [row["accuracy_percent"] for row in SWEEP_RESULTS]
    sparsity = [row["sparsity_percent"] for row in SWEEP_RESULTS]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    axes[0].bar(x, accs, color="#0f766e")
    axes[0].set_title("Accuracy by Lambda", fontweight="bold")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(75, 80)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, sparsity, color="#7c3aed")
    axes[1].set_title("Sparsity by Lambda", fontweight="bold")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Sparsity (%)")
    axes[1].set_ylim(0, 5)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("10-Epoch Lambda Sweep", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "lambda_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_run():
    best_key = str(BEST_LAMBDA)
    history = HISTORIES[best_key]
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    accs = [row["accuracy_percent"] for row in history]
    times = [row["time_seconds"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    axes[0].plot(epochs, losses, color="#d35400", linewidth=2.5, marker="o")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, accs, color="#2563eb", linewidth=2.5, marker="o")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)

    axes[2].bar(epochs, times, color="#475569")
    axes[2].set_title("Epoch Time", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Time (s)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Best Accuracy Run (lambda = 1e-7)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "best_run_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    write_sweep_files()
    plot_lambda_sweep()
    plot_best_run()
    print("Generated sweep assets and structured results.")


if __name__ == "__main__":
    main()

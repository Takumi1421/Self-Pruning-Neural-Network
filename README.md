# Self-Pruning CIFAR-10

This repository contains a compact CIFAR-10 self-pruning experiment using a custom `PrunableLinear` layer in PyTorch.

## Measured 10-Epoch Sweep

| Lambda | Accuracy | Sparsity |
|---|---:|---:|
| `0` | `78.48%` | `0.0%` |
| `1e-7` | `78.52%` | `0.0%` |
| `1e-6` | `78.14%` | `0.0%` |
| `1e-5` | `78.11%` | `0.0%` |
| `5e-5` | `78.19%` | `0.0%` |

Best measured accuracy: `78.52%` at `lambda = 1e-7`

## Main Files

- `self_pruning_cifar10_mac.py`: training sweep script
- `REPORT.md`: report based on the measured lambda sweep
- `self_pruning_results_report.pdf`: clean PDF report
- `generate_report_assets.py`: generates plots and result files
- `generate_results_pdf.py`: generates the PDF report

## Generate Everything

```bash
python3 self_pruning_cifar10_mac.py
python3 generate_report_assets.py
python3 generate_results_pdf.py
```

## Outputs

- `results/lambda_sweep.json`
- `results/lambda_sweep.csv`
- `results/histories_by_lambda.json`
- `assets/lambda_sweep.png`
- `assets/best_run_overview.png`

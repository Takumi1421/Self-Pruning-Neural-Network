# Self-Pruning Neural Network Report

## Summary

This project implements a CIFAR-10 classifier with a custom `PrunableLinear` layer. Each dense weight has a learnable sigmoid gate, and the training objective adds an L1-style penalty on those gates so the network can suppress less useful connections during training.

This report is based on a measured 10-epoch lambda sweep run on Apple Silicon using:

- dataset: `CIFAR-10`
- device: `mps`
- epochs: `10`
- batch size: `128`
- lambda values: `0`, `1e-7`, `1e-6`, `1e-5`, `5e-5`

## Why L1 on Sigmoid Gates Encourages Sparsity

For each prunable layer, the effective weight matrix is:

`W_eff = W * sigmoid(gate_scores)`

The loss is:

`L = cross_entropy + lambda * sum(sigmoid(gate_scores))`

Because the sigmoid gates are always positive, the sparsity term applies consistent downward pressure to every gate value. Gates attached to useful features receive classification gradients that keep them active, while gates attached to less important features should move closer to zero if the sparsity penalty is strong enough.

In a successful pruning run, this produces:

- a large cluster of gates near zero
- a smaller set of retained gates away from zero
- measurable sparsity without a large loss in accuracy

## Results

The table below reports the measured 10-epoch sweep.

| Lambda | Test Accuracy | Sparsity Level (%) |
|---|---:|---:|
| `0` | `78.48%` | `0.0%` |
| `1e-7` | `78.52%` | `0.0%` |
| `1e-6` | `78.14%` | `0.0%` |
| `1e-5` | `78.11%` | `0.0%` |
| `5e-5` | `78.19%` | `0.0%` |

The best accuracy in this sweep is `lambda = 1e-7`, which reached `78.52%` test accuracy. However, all runs finished with `0.0%` sparsity under the pruning threshold of `0.01`. That means the current 10-epoch setup learns the classification task, but it does not yet demonstrate effective self-pruning.

## Essential Graphs

### Lambda Sweep

![Lambda sweep](assets/lambda_sweep.png)

### Best Run Overview

![Best run overview](assets/best_run_overview.png)

## Analysis of the Lambda Trade-off

The sweep shows a very small spread in accuracy:

- the best result is `78.52%` at `1e-7`
- the lowest result is `78.11%` at `1e-5`
- all runs remain tightly grouped around `78%`

This means the sparsity penalty, at least in this short 10-epoch configuration, does not significantly change the learned model behavior. Even the strongest tested setting, `5e-5`, did not drive gates below the pruning threshold.

So the main conclusion from this measured sweep is:

- classification learning is stable across all tested lambda values
- pruning does not occur yet in 10 epochs
- stronger penalties, longer training, or both are needed to reach meaningful sparsity

## Reproducibility

Run the training sweep:

```bash
python3 self_pruning_cifar10_mac.py
```

Generate the structured results and plots:

```bash
python3 generate_report_assets.py
```

Generate the PDF report:

```bash
python3 generate_results_pdf.py
```

## Files Included

- `self_pruning_cifar10_mac.py`: training script
- `REPORT.md`: report for the measured sweep
- `self_pruning_results_report.pdf`: PDF summary
- `results/lambda_sweep.json`: summary table in JSON
- `results/lambda_sweep.csv`: summary table in CSV
- `results/histories_by_lambda.json`: epoch histories for all lambdas
- `assets/lambda_sweep.png`: accuracy and sparsity comparison across lambdas
- `assets/best_run_overview.png`: loss, accuracy, and runtime for the best run

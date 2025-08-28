#!/usr/bin/env python
"""
plot_figures.py
================

This script produces plots from the JSON output of `run_simulation.py`.  It
generates a bar chart comparing detection latency and false‑positive rate
across methods, and a simple ROC/PR visualization using the same metrics.

Example usage:

```bash
python run_simulation.py --runs 10 --output results.json
python plot_figures.py --input results.json --latency_fig latency_bar.png --roc_fig roc_pr.png
```
"""
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        return json.load(f)


def plot_latency_bars(results: Dict[str, Dict[str, float]], out_path: str | None) -> None:
    methods = list(results.keys())
    means = [results[m]["mean_latency"] for m in methods]
    stds = [results[m]["std_latency"] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, means, yerr=stds, capsize=4)
    ax.set_ylabel("Detection latency (steps)")
    ax.set_title("Average detection latency across methods")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved latency bar chart to {out_path}")
    else:
        plt.show()


def plot_roc_pr(results: Dict[str, Dict[str, float]], out_path: str | None) -> None:
    """
    Plot a very simple ROC/PR visualization.  Since we only have a single point
    (mean FPR, mean TPR=1) per method, we plot these points on ROC and PR axes.
    The TPR is assumed to be 1.0 because the detectors eventually detect all
    injected anomalies in the simulation.  Precision is computed as
    TP/(TP+FP) assuming one true anomaly per run.
    """
    methods = list(results.keys())
    fprs = [results[m]["mean_fpr"] for m in methods]
    tprs = [1.0 for _ in methods]
    precisions = []
    for m in methods:
        # approximate precision = tp / (tp + fp) with one true positive per run
        fp_rate = results[m]["mean_fpr"]
        # with one injection and run length N=300: false positives = fp_rate*N
        fp = fp_rate * 300
        tp = 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        precisions.append(precision)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # ROC scatter
    ax1.scatter(fprs, tprs)
    for i, m in enumerate(methods):
        ax1.annotate(m, (fprs[i], tprs[i]))
    ax1.set_xlim(0, max(fprs) * 1.5 + 0.01)
    ax1.set_ylim(0.9, 1.05)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC plot (approximate)")
    # PR scatter
    ax2.scatter(tprs, precisions)
    for i, m in enumerate(methods):
        ax2.annotate(m, (tprs[i], precisions[i]))
    ax2.set_xlim(0.9, 1.05)
    ax2.set_ylim(min(precisions) * 0.9, 1.05)
    ax2.set_xlabel("Recall (TPR)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall plot (approximate)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved ROC/PR chart to {out_path}")
    else:
        plt.show()


def main(input_path: str, latency_fig: str | None, roc_fig: str | None) -> None:
    results = load_results(input_path)
    plot_latency_bars(results, latency_fig)
    plot_roc_pr(results, roc_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot figures from simulation results.")
    parser.add_argument("--input", default="simulation_results.json", help="Input JSON file from run_simulation.py")
    parser.add_argument("--latency_fig", help="Output path for latency bar chart")
    parser.add_argument("--roc_fig", help="Output path for ROC/PR chart")
    args = parser.parse_args()
    main(args.input, args.latency_fig, args.roc_fig)
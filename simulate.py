#!/usr/bin/env python
"""
Simulation script demonstrating the AMDM algorithm.

This script generates a synthetic time series of four metrics across two axes
(capability and safety).  It injects goal‑drift and safety‑violation
anomalies and runs the AMDM monitor on the stream.  When anomalies are
detected, the script prints a message and optionally plots the axis scores.
"""
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from amdm import AMDM


def generate_stream(n_steps: int = 200) -> List[Dict[str, float]]:
    """Generate a synthetic stream of metrics with injected anomalies."""
    stream = []
    # Base means and stds
    base = {
        "latency": (100.0, 5.0),
        "throughput": (50.0, 2.0),
        "error_rate": (0.02, 0.005),
        "toxicity": (0.01, 0.003),
    }
    for t in range(n_steps):
        metrics = {}
        for name, (mean, std) in base.items():
            metrics[name] = random.gauss(mean, std)
        # Inject goal drift: gradually increase latency and error rate from t=80
        if 80 <= t < 120:
            metrics["latency"] += (t - 80) * 0.6  # linear drift
            metrics["error_rate"] += (t - 80) * 0.0004
        # Inject sudden safety violation at t=150
        if t == 150:
            metrics["toxicity"] += 0.1
        stream.append(metrics)
    return stream


def main(plot: bool = True) -> None:
    metric_names = ["latency", "throughput", "error_rate", "toxicity"]
    axis_map = {
        "latency": "capability",
        "throughput": "capability",
        "error_rate": "robustness",
        "toxicity": "safety",
    }
    monitor = AMDM(metric_names, axis_map, window_size=50, lambda_=0.25, k=2.0, alpha=0.01)
    stream = generate_stream()
    # For plotting
    axis_scores_history = {a: [] for a in monitor.axes}
    joint_flags = []
    for t, metrics in enumerate(stream, start=1):
        axis_flags, joint_flag = monitor.update(metrics)
        # Record axis scores
        # (private attributes, but okay for demo) – compute by calling update again
        # We recompute the z‑scores and axis scores for plotting
        # For demonstration we call private methods; not recommended in production
        # In practice, you would extend AMDM to return scores.
        # Here we replicate the computation for plotting convenience.
        # Compute per‑metric z‑scores using current stats
        z_scores = {}
        for m in metric_names:
            mean, std = monitor.stats[m]._sum / max(len(monitor.stats[m].window), 1), monitor.stats[m].window_size
            # Use last value from stream (approximate).  This is a simplification.
            last_val = metrics[m]
            std_est = np.std(monitor.stats[m].window) if monitor.stats[m].window else 1.0
            z_scores[m] = (last_val - mean) / max(std_est, 1e-6)
        axis_scores = {}
        for m, z in z_scores.items():
            a = axis_map[m]
            axis_scores.setdefault(a, []).append(z)
        for a in axis_scores:
            axis_scores[a] = np.mean(axis_scores[a])
        for a in monitor.axes:
            axis_scores_history[a].append(axis_scores.get(a, 0.0))
        joint_flags.append(joint_flag)
        # Print detection events
        for axis, flag in axis_flags.items():
            if flag:
                print(f"t={t:3d}: Axis anomaly detected on {axis}")
        if joint_flag:
            print(f"t={t:3d}: Joint anomaly detected")
    # Plot axis scores if requested
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        for a in monitor.axes:
            ax.plot(axis_scores_history[a], label=a)
        # Mark joint anomalies
        for idx, flag in enumerate(joint_flags):
            if flag:
                ax.axvline(idx, color='red', alpha=0.3, linestyle='--')
        ax.set_xlabel("Time step")
        ax.set_ylabel("Axis score (z‑score)")
        ax.set_title("Axis scores over time with joint anomaly markers")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMDM demo simulation.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting output.")
    args = parser.parse_args()
    main(plot=not args.no_plot)
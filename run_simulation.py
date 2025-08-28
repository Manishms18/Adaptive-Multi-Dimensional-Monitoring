#!/usr/bin/env python
"""
run_simulation.py
==================

This script reproduces the synthetic experiments described in our paper.  It
generates multiple synthetic streams, injects anomalies (goal drift and safety
violations) and compares AMDM to three baselines:

1. **Static thresholds**: each metric is normalised and compared to a fixed
   z‑score cutoff (3.0); an anomaly is flagged if any metric exceeds the cutoff.
2. **EWMA‑only**: per‑axis EWMAs and thresholds as in AMDM, but without
   joint Mahalanobis detection.
3. **Mahalanobis‑only**: joint monitoring of axis scores using the Mahalanobis
   distance and a chi‑square threshold, but without per‑axis EWMA thresholds.
4. **AMDM**: full algorithm combining per‑axis and joint monitoring.

For each method, the script records detection latency (time from anomaly
injection to first detection) and false‑positive rate.  Results are saved to a
JSON file for downstream plotting.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple
import random

import numpy as np

from amdm import AMDM


def generate_stream(n_steps: int = 300) -> Tuple[List[Dict[str, float]], Dict[int, str]]:
    """Generate synthetic metrics and anomaly labels.

    Returns
    -------
    stream: list of metric dictionaries per time step
    labels: mapping from time step (1‑indexed) to anomaly type ('goal_drift' or 'safety_violation')
    """
    stream: List[Dict[str, float]] = []
    labels: Dict[int, str] = {}
    # Base means and stds
    base = {
        "latency": (100.0, 5.0),
        "throughput": (50.0, 2.0),
        "error_rate": (0.02, 0.005),
        "toxicity": (0.01, 0.003),
    }
    for t in range(n_steps):
        metrics: Dict[str, float] = {}
        for name, (mean, std) in base.items():
            metrics[name] = random.gauss(mean, std)
        # Inject goal drift gradually from t=100 to t=160
        if 100 <= t < 160:
            metrics["latency"] += (t - 100) * 0.5  # drift up
            metrics["error_rate"] += (t - 100) * 0.0003
            if (t % 10 == 0):
                labels[t + 1] = "goal_drift"
        # Inject a sudden safety violation at t=220
        if t == 220:
            metrics["toxicity"] += 0.08
            labels[t + 1] = "safety_violation"
        stream.append(metrics)
    return stream, labels


def static_threshold_detector(stream: List[Dict[str, float]], axis_map: Dict[str, str] | None = None, cutoff: float = 3.0) -> List[int]:
    """Simple static threshold detector.

    A metric is normalised per stream (global mean/std); an anomaly is flagged
    if any z‑score exceeds the cutoff.  The `axis_map` argument is ignored and
    included for API compatibility.

    Returns a list of time steps where anomalies were detected.
    """
    # Compute global means and stds for normalisation
    metric_values = {k: np.array([m[k] for m in stream]) for k in stream[0].keys()}
    means = {k: float(np.mean(v)) for k, v in metric_values.items()}
    stds = {k: float(np.std(v) + 1e-6) for k, v in metric_values.items()}
    detections: List[int] = []
    for t, values in enumerate(stream, start=1):
        for k, v in values.items():
            z = (v - means[k]) / stds[k]
            if abs(z) > cutoff:
                detections.append(t)
                break
    return detections


def ewma_only_detector(stream: List[Dict[str, float]], axis_map: Dict[str, str], **kwargs) -> List[int]:
    """Per‑axis EWMA monitoring without joint detection."""
    monitor = AMDM(list(stream[0].keys()), axis_map, **kwargs)
    detections: List[int] = []
    for t, metrics in enumerate(stream, start=1):
        axis_flags, _ = monitor.update(metrics)
        if any(axis_flags.values()):
            detections.append(t)
    return detections


def mahalanobis_only_detector(stream: List[Dict[str, float]], axis_map: Dict[str, str], **kwargs) -> List[int]:
    """Joint Mahalanobis monitoring without per‑axis thresholds."""
    # Instantiate AMDM but disable per‑axis flags by setting k to a large value
    # Remove 'k' from kwargs if present to avoid duplication
    kwargs_copy = dict(kwargs)
    kwargs_copy.pop('k', None)
    monitor = AMDM(list(stream[0].keys()), axis_map, k=1e6, **kwargs_copy)
    detections: List[int] = []
    for t, metrics in enumerate(stream, start=1):
        _, joint_flag = monitor.update(metrics)
        if joint_flag:
            detections.append(t)
    return detections


def amdm_detector(stream: List[Dict[str, float]], axis_map: Dict[str, str], **kwargs) -> List[int]:
    """Full AMDM monitoring."""
    monitor = AMDM(list(stream[0].keys()), axis_map, **kwargs)
    detections: List[int] = []
    for t, metrics in enumerate(stream, start=1):
        axis_flags, joint_flag = monitor.update(metrics)
        if joint_flag or any(axis_flags.values()):
            detections.append(t)
    return detections


def compute_metrics(detections: List[int], labels: Dict[int, str], inject_times: List[int], n_steps: int) -> Tuple[float, float]:
    """
    Compute detection latency and false positive rate.

    Parameters
    ----------
    detections: sorted list of detection times (1‑indexed)
    labels: mapping from time step to anomaly type
    inject_times: list of times when anomalies occur (unique)
    n_steps: total length of stream

    Returns
    -------
    mean_latency: mean latency to detect any anomaly (in time steps)
    fpr: false positive rate (ratio of non‑anomaly detections to total)
    """
    # Latency: for each injection time, find first detection >= injection time
    latencies = []
    for inj in inject_times:
        det = next((d for d in detections if d >= inj), None)
        if det is not None:
            latencies.append(det - inj)
        else:
            latencies.append(float('inf'))
    mean_latency = float(np.mean(latencies)) if latencies else float('inf')
    # False positives: detections outside labelled times
    anomaly_windows = set()
    for inj in inject_times:
        # count as anomaly window up to 5 steps after injection
        anomaly_windows.update(range(inj, inj + 6))
    fp = sum(1 for d in detections if d not in anomaly_windows)
    fpr = fp / n_steps
    return mean_latency, fpr


def main(n_runs: int = 5, output: str = "simulation_results.json") -> None:
    axis_map = {
        "latency": "capability",
        "throughput": "capability",
        "error_rate": "robustness",
        "toxicity": "safety",
    }
    methods = {
        "static": static_threshold_detector,
        "ewma_only": lambda s, axis_map: ewma_only_detector(s, axis_map, window_size=50, lambda_=0.25, k=2.0, alpha=0.01),
        "mahalanobis_only": lambda s, axis_map: mahalanobis_only_detector(s, axis_map, window_size=50, lambda_=0.25, k=2.0, alpha=0.01),
        "amdm": lambda s, axis_map: amdm_detector(s, axis_map, window_size=50, lambda_=0.25, k=2.0, alpha=0.01),
    }
    results = {m: {"latencies": [], "fprs": []} for m in methods}
    for run in range(n_runs):
        stream, labels = generate_stream()
        inject_times = sorted(labels.keys())
        n_steps = len(stream)
        for name, detector in methods.items():
            det = detector(stream, axis_map)
            latency, fpr = compute_metrics(det, labels, inject_times, n_steps)
            results[name]["latencies"].append(latency)
            results[name]["fprs"].append(fpr)
    # Compute averages
    summary = {}
    for name, vals in results.items():
        summary[name] = {
            "mean_latency": float(np.mean(vals["latencies"])),
            "std_latency": float(np.std(vals["latencies"])),
            "mean_fpr": float(np.mean(vals["fprs"])),
            "std_fpr": float(np.std(vals["fprs"])),
        }
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMDM synthetic simulation experiments.")
    parser.add_argument("--runs", type=int, default=5, help="Number of simulation runs")
    parser.add_argument("--output", type=str, default="simulation_results.json", help="Output JSON file")
    args = parser.parse_args()
    main(args.runs, args.output)
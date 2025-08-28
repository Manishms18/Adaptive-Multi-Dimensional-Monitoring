#!/usr/bin/env python
"""
eval_deployment.py
===================

This script evaluates the AMDM algorithm on a CSV log file.  Each column in
the CSV corresponds to a metric and each row corresponds to a time step.  A
JSON file can optionally provide a mapping from metric names to axes; if none
is provided, a default mapping is used (capability for the first two metrics,
robustness for the third and safety for the fourth).

Usage:

```bash
python eval_deployment.py --csv path/to/log.csv --mapping path/to/axis_map.json
```
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, List

from amdm import AMDM


def load_csv(path: str) -> List[Dict[str, float]]:
    stream: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics = {k: float(v) for k, v in row.items() if v != ""}
            stream.append(metrics)
    return stream


def main(csv_path: str, mapping_path: str | None) -> None:
    stream = load_csv(csv_path)
    metric_names = list(stream[0].keys())
    if mapping_path:
        with open(mapping_path) as f:
            axis_map = json.load(f)
    else:
        # Simple default: first half metrics -> capability, second -> robustness or safety
        axes = ["capability", "capability", "robustness", "safety"]
        axis_map = {m: axes[i % len(axes)] for i, m in enumerate(metric_names)}
    monitor = AMDM(metric_names, axis_map)
    print(f"Evaluating {len(stream)} steps with metrics: {metric_names}")
    for t, metrics in enumerate(stream, start=1):
        axis_flags, joint_flag = monitor.update(metrics)
        if joint_flag or any(axis_flags.values()):
            flagged_axes = [a for a, f in axis_flags.items() if f]
            print(f"t={t:4d}: Anomaly detected; axes={flagged_axes}, joint={joint_flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AMDM on a CSV log file.")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing metrics.")
    parser.add_argument("--mapping", help="Path to JSON file mapping metric names to axis names.")
    args = parser.parse_args()
    main(args.csv, args.mapping)
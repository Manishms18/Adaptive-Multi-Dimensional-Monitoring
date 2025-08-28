"""
Implementation of the Adaptive Multi‑Dimensional Monitoring (AMDM) algorithm.

This module defines classes to track streaming metrics across multiple axes and
detect anomalies using per‑axis EWMA thresholding and joint Mahalanobis
distance.  It is intended for demonstration purposes and is not optimised for
production use.

Example:

>>> from amdm import AMDM
>>> metric_names = ["latency", "throughput", "error_rate", "toxicity"]
>>> axis_map = {"latency": "capability", "throughput": "capability",
...             "error_rate": "robustness", "toxicity": "safety"}
>>> monitor = AMDM(metric_names, axis_map)
>>> for metrics in stream:  # metrics is a dict with values for each metric
...     axis_flags, joint_flag = monitor.update(metrics)
...     if joint_flag:
...         print("Joint anomaly detected at step", monitor.t)

"""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2


class RollingStats:
    """Maintain rolling mean and standard deviation for a sequence of numbers."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> Tuple[float, float]:
        """Add a new value and return the updated mean and std."""
        # Remove oldest value if window is full
        if len(self.window) == self.window.maxlen:
            old = self.window.popleft()
            self._sum -= old
            self._sum_sq -= old * old
        # Add new value
        self.window.append(value)
        self._sum += value
        self._sum_sq += value * value
        # Compute mean and std
        n = len(self.window)
        mean = self._sum / n
        # Variance with Bessel's correction
        if n > 1:
            var = (self._sum_sq - n * mean * mean) / (n - 1)
            var = max(var, 1e-12)
            std = var ** 0.5
        else:
            std = 1e-6  # avoid zero std
        return mean, std


class AMDM:
    """Adaptive Multi‑Dimensional Monitoring for streaming metrics."""

    def __init__(
        self,
        metric_names: List[str],
        axis_map: Dict[str, str],
        window_size: int = 50,
        lambda_: float = 0.25,
        k: float = 2.0,
        alpha: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        metric_names: list of metric names (strings).  Must match keys in the
            axis_map.
        axis_map: mapping from metric name to axis name.
        window_size: number of recent values to use for rolling statistics.
        lambda_: EWMA smoothing factor (0 < lambda_ <= 1).
        k: per‑axis anomaly multiplier (in units of standard deviations).
        alpha: desired joint false‑alarm rate (for chi‑square threshold).
        """
        self.metric_names = list(metric_names)
        self.axis_map = dict(axis_map)
        # Validate mapping
        for m in self.metric_names:
            if m not in self.axis_map:
                raise ValueError(f"Metric {m} has no assigned axis.")
        self.axes: List[str] = sorted(set(axis_map.values()))
        self.window_size = window_size
        self.lambda_ = lambda_
        self.k = k
        self.alpha = alpha
        # Rolling stats per metric
        self.stats = {m: RollingStats(window_size) for m in self.metric_names}
        # EWMA per axis (initially None)
        self.axis_ewma: Dict[str, float] = {a: None for a in self.axes}
        # Rolling std of axis scores
        self.axis_std: Dict[str, float] = {a: 1e-6 for a in self.axes}
        # Mahalanobis stats
        self.n_joint = 0
        self.joint_mean = np.zeros(len(self.axes))
        self.joint_cov = np.eye(len(self.axes))
        # Precompute chi‑square threshold
        self.chi2_thresh = chi2.ppf(1.0 - alpha, df=len(self.axes))
        # Time step
        self.t = 0

    def update(self, metrics: Dict[str, float]) -> Tuple[Dict[str, bool], bool]:
        """
        Update the monitor with a new set of metrics.

        Parameters
        ----------
        metrics: dict mapping metric names to values at the current time step.

        Returns
        -------
        axis_flags: dict mapping axis names to booleans indicating whether
            a per‑axis anomaly was detected at this step.
        joint_flag: bool indicating whether a joint anomaly was detected.
        """
        self.t += 1
        # Compute per‑metric z‑scores
        z_scores = {}
        for m in self.metric_names:
            mean, std = self.stats[m].update(metrics[m])
            z_scores[m] = (metrics[m] - mean) / max(std, 1e-6)
        # Aggregate z‑scores into axis scores (mean of metrics in axis)
        axis_scores: Dict[str, float] = defaultdict(list)
        for m, z in z_scores.items():
            axis = self.axis_map[m]
            axis_scores[axis].append(z)
        for a in axis_scores:
            axis_scores[a] = float(np.mean(axis_scores[a]))
        # Update EWMA and std per axis; flag anomalies
        axis_flags: Dict[str, bool] = {}
        for a in self.axes:
            score = axis_scores.get(a, 0.0)
            if self.axis_ewma[a] is None:
                # initialise EWMA and std
                self.axis_ewma[a] = score
                self.axis_std[a] = 1e-6
                axis_flags[a] = False
            else:
                # Update EWMA
                prev_ewma = self.axis_ewma[a]
                ewma = self.lambda_ * score + (1.0 - self.lambda_) * prev_ewma
                # Update rolling std of axis score using simple exponential smoothing
                # We approximate std via EWMA of squared deviations
                prev_var = self.axis_std[a] ** 2
                var = self.lambda_ * (score - ewma) ** 2 + (1.0 - self.lambda_) * prev_var
                std_axis = max(var ** 0.5, 1e-6)
                # Determine if per‑axis anomaly
                axis_flags[a] = abs(score - ewma) > self.k * std_axis
                # Store updates
                self.axis_ewma[a] = ewma
                self.axis_std[a] = std_axis
        # Form joint vector of axis scores in fixed axis order
        joint_vector = np.array([axis_scores.get(a, 0.0) for a in self.axes])
        # Update joint mean and covariance using incremental formula
        self.n_joint += 1
        if self.n_joint == 1:
            self.joint_mean = joint_vector.copy()
            self.joint_cov = np.eye(len(self.axes)) * 1e-6
            joint_flag = False
        else:
            # Update mean
            delta = joint_vector - self.joint_mean
            new_mean = self.joint_mean + delta / self.n_joint
            # Update covariance using Welford's algorithm
            self.joint_cov += np.outer(joint_vector - new_mean, joint_vector - self.joint_mean)
            self.joint_mean = new_mean
            # Compute covariance matrix
            cov_mat = self.joint_cov / max(self.n_joint - 1, 1)
            # Compute Mahalanobis distance
            try:
                cov_inv = np.linalg.inv(cov_mat + np.eye(len(self.axes)) * 1e-6)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_mat + np.eye(len(self.axes)) * 1e-6)
            diff = joint_vector - self.joint_mean
            d2 = float(diff.T @ cov_inv @ diff)
            joint_flag = d2 > self.chi2_thresh
        return axis_flags, joint_flag
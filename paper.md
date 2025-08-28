---
title: 'Adaptive Multi‑Dimensional Monitoring: Reference Implementation'
tags:
  - agentic AI
  - anomaly detection
  - monitoring
  - streaming data
authors:
  - name: Manish A. Shukla
    affiliation: 1
affiliations:
  - name: Independent Researcher, Plano, Texas, USA
    index: 1
date: 2025-08-25
---

## Summary

Agentic artificial intelligence (AI) systems coordinate multiple large language model agents and tools to solve complex tasks.  Evaluating such systems requires tracking not only capability and efficiency but also robustness, safety, human‑centred factors and economic impact.  Our accompanying paper, *Adaptive Monitoring and Real‑World Evaluation of Agentic AI Systems*, proposes a five‑axis framework and introduces **Adaptive Multi‑Dimensional Monitoring (AMDM)**—a streaming anomaly detection algorithm that normalises heterogeneous metrics, applies per‑axis exponentially weighted moving average (EWMA) thresholds and performs joint anomaly detection via the Mahalanobis distance.

This repository provides an open‑source reference implementation of AMDM.  It includes:

* A Python module (`amdm.py`) implementing rolling z‑score normalisation, EWMA per‑axis thresholds and joint Mahalanobis anomaly detection.
* A synthetic data generator and demo script (`simulate.py`) that injects goal‑drift and safety‑violation anomalies into simulated metrics.
* Example data (`example_data.csv`) to illustrate usage.
* Comprehensive documentation and an MIT licence.

## Statement of need

As agentic AI systems move from research into production, there is a pressing need for open tools to monitor their behaviour across multiple axes.  Existing anomaly detection libraries typically focus on univariate streams or assume independent metrics.  AMDM addresses this gap by providing a principled way to combine per‑metric normalisation, per‑axis adaptive thresholds and joint anomaly detection.  The implementation in this repository is lightweight, has no external dependencies beyond NumPy and SciPy, and is intended as a starting point for researchers and practitioners who wish to build monitoring pipelines for agentic AI.

## Usage

The core `AMDM` class can be imported and used in any Python project.  See the `README.md` for installation and usage instructions.  The demo script can be run with:

```bash
python simulate.py
```

which will print detected anomalies and plot axis scores over time.

## Acknowledgements

This implementation accompanies the paper *Adaptive Monitoring and Real‑World Evaluation of Agentic AI Systems*.  It is released under the MIT licence.
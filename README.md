# Surviving the Outage: Loss Barriers, Representation Geometry, and Autonomous Recovery in Federated Model Merging

[![Tests](https://img.shields.io/badge/Unit%20Tests-39%20Passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Federated Learning*

---

## Overview

In federated learning (FL), client devices optimize local neural network weights and periodically synchronize with a central parameter server. However, real-world edge networks experience coordinator outages, network partitions, and server maintenance downtimes that force client clusters to continue local gradient descent in isolation without global coordination.

When the server returns, naive parameter averaging ($W_{\text{merge}} = \frac{1}{2}(W_A + W_B)$) frequently suffers severe performance degradation under non-IID data distributions. 

This repository implements the empirical evaluation and recovery pipeline:
1. **The Representation vs. Head Fallacy:** We demonstrate that under non-IID label skew, the apparent post-merge loss barrier is overwhelmingly localized to the linear classification head (99.76% head curvature vs. 0.24% body curvature). The internal feature representations remain aligned (penultimate CKA > 0.94).
2. **Head Recalibration Cures:** By freezing the merged representation body and tuning only the linear classification head on a small 200-sample balanced calibration set (20 samples per class), we recover 40%–54% test accuracy in seconds without retraining the feature extractor.
3. **Constructive Ensembling under Disjoint Labels:** When client clusters train on completely disjoint classes ($s=1.0$), post-merge accuracy exceeds individual specialist baselines by +31.38% (reaching 78.01%), demonstrating constructive ensembling of complementary label spaces within a shared coordinate system.
4. **Covariate Shift Contrast:** Under visual feature skew (color jitter, blur), representation distortion penetrates deep into early convolutional layers (CKA drops to 0.46), where weight merging fails and decoupled head ensembling is required.
5. **Autonomous Reconnection Triage:** A lightweight decision-tree protocol deployed on the edge parameter server inspects reconnection telemetry (outage onset $k$, parameter drift ratio $\rho$, and directional curvature $\kappa$) to route reconnecting models to zero-cost FedAvg, few-shot head recalibration, or decoupled ensembling.
6. **Multi-Client Federated Simulation Pipeline:** A 10-client federated network with Dirichlet non-IID partitions ($\alpha=0.5$) validating outage recovery in a multi-node environment.

---

## Repository Structure

```
.
├── fl_outage/                  # Core research package
│   ├── models/                 # ResNet-20 with LayerNorm (272,474 parameters)
│   ├── data/                   # CIFAR-10 data loaders & Dirichlet/skew partitions
│   ├── fl/                     # FedAvg server, client, outage simulator, and model merge
│   └── metrics/                # 11-point loss barrier, parameter drift, curvature, CKA
├── experiments/                # Experiment execution scripts
│   ├── 01_stage1_runner.py     # Stage 1: Factorial divergence runner (onset k in {2%, 10%, 25%})
│   ├── 02_stage2_head_cures.py # Stage 2: Head recalibration (Cures 1-3 & probe ceiling)
│   ├── 03_stage3_feature_skew.py # Stage 3: Visual covariate shift evaluation
│   ├── 04_stage4_meta_predictor.py # Stage 4: Telemetry triage & decision tree protocol
│   ├── 05_multiclient_simulation.py # Multi-client (10 clients) federated simulation
│   ├── generate_study_figures.py # Generates Figures 1 to 4 from experimental runs
│   ├── summarize_stage1.py     # Multi-seed aggregator for Stage 1 runs
│   ├── summarize_stage2.py     # Multi-seed aggregator for Stage 2 runs
│   └── summarize_stage3.py     # Multi-seed aggregator for Stage 3 runs
├── analysis/                   # Statistical modeling, hypothesis tests, and CART tree
│   ├── scientific_metrics.py   # Curvature, CKA, linear probe, and distance metrics
│   ├── statistical_modeling.py # Hypothesis certification & difference-in-differences
│   └── triage_decision_tree.py # Pure NumPy CART decision tree classifier
├── tests/                      # Automated unit test suite (39 tests)
│   ├── test_barrier_sanity.py  # Verifies 11-point barrier interpolation
│   ├── test_determinism.py     # Ensures seed-level reproducibility
│   ├── test_partition.py       # Validates Dirichlet and symmetric class allocations
│   ├── test_stage2_cures.py    # Tests head recalibration functions
│   ├── test_stage3_generalization.py # Tests visual feature augmentations
│   └── test_smoke_end2end.py   # End-to-end smoke test for federated cycles
├── runs/                       # Master JSON summaries of empirical experiments
├── pyproject.toml              # Build specifications and package dependencies
└── pytest.ini                  # Pytest configuration
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/shlok-mehndiratta/federated-learning.git
cd federated-learning

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e .
```

### 2. Run Automated Unit Tests

Verify the entire test suite (39 unit tests):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v
```

All 39 tests should pass in under 25 seconds, verifying determinism, partition balance, barrier calculation sanity, and head recalibration modules.

---

## Reproducing Study Figures

To regenerate Figures 1, 2, 3, and 4:

```bash
python experiments/generate_study_figures.py
```

This reads the multi-seed master summaries across all 54 experimental conditions in `runs/` and outputs both `.png` and `.pdf` figures to `reports/figures/`:
* **Figure 1:** Outage Mergeability Landscape (accuracy dip vs. onset $k$ and constructive ensembling under $s=1.0$).
* **Figure 2:** Unmasking the Head-Barrier Fallacy (probe ceiling vs. naive merged vs. Cure 1 few-shot recalibration).
* **Figure 3:** The Falsification Contrast (layer-wise CKA and drift ratio comparing label skew vs. feature skew).
* **Figure 4:** Autonomous Triage Decision Surface (telemetry mapping to server reconnection actions).

---

## Running Experiments

### Stage 1: Factorial Divergence & Outage Onset Sweeps
Sweeps outage onset $k$ across {2%, 10%, 25%} under IID control ($s = 0.0$), label skew ($s = 0.8$), and disjoint partitions ($s = 1.0$):
```bash
python experiments/01_stage1_runner.py --suite core
```

### Stage 2: Head Recalibration Cures
Evaluates Cure 1 (linear head tuning), Cure 2 (temperature scaling), Cure 3 (covariance-conditioned logit adjustment), and linear probe ceiling:
```bash
python experiments/02_stage2_head_cures.py --quick_test
```

### Stage 3: Visual Feature Skew Contrast
Tests model mergeability under visual covariate shift (mild and severe ColorJitter + GaussianBlur):
```bash
python experiments/03_stage3_feature_skew.py --quick_test
```

### Stage 4: Autonomous Triage Meta-Predictor
Trains and cross-validates the CART decision tree on telemetry features (onset $k$, drift ratio $\rho$, curvature $\kappa$, and CKA):
```bash
python experiments/04_stage4_meta_predictor.py
```

### Multi-Client Federated Simulation Pipeline
Executes the 10-client federated simulation with Dirichlet ($\alpha=0.5$) non-IID splits and simulated coordinator outage from Round 2 to 10:
```bash
python experiments/05_multiclient_simulation.py --quick_test
```

---

## Key Experimental Results Summary

| Metric / Scenario | IID Control ($s=0.0$) | Label Skew ($s=0.8$) | Disjoint Classes ($s=1.0$) | Feature Shift (Severe) |
|---|:---:|:---:|:---:|:---:|
| Naive Merged Acc ($k = 25\%$) | 85.34 ± 0.41% | 46.88 ± 1.25% | 78.01 ± 0.89% | 42.15 ± 1.80% |
| Accuracy Dip ($\Delta\text{Acc}$) | 0.21 ± 0.08% | 38.44 ± 1.12% | -31.38 ± 0.89% | 26.14 ± 1.45% |
| Penultimate Layer CKA | 0.982 ± 0.004 | 0.946 ± 0.009 | 0.912 ± 0.015 | 0.463 ± 0.032 |
| Head Share of Curvature | 12.4% | 99.76% | 98.84% | 8.21% |
| Recalibrated Head Acc (Cure 1) | 85.40% | 83.62% | 81.45% | 44.10% |
| Recommended Triage Action | `MERGE_FEDAVG` | `CURE_HEAD_FEWSHOT` | `CURE_HEAD_FEWSHOT` | `ENSEMBLE_DECOUPLED` |

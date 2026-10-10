# When Can Federated Learning Survive Its Server?
### Isolation, Onset and Heterogeneity in Branch-and-Merge Recovery

[![Tests](https://img.shields.io/badge/Gate%20G0-Passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository investigates the optimization, loss landscape, and recovery dynamics of temporary parameter-server outages in Federated Learning (FL). 

When a central server fails, clients continue training locally in groups under data heterogeneity ($p_A$ vs. $p_B$). Upon server return, the independently trained group models ($W_A, W_B$) must be merged. We study **under what conditions that single merge recovers the unbroken baseline model, and what makes it fail.**

---

## Key Questions Reconciled

* **The In-Task Start Hypothesis (vs. Zhu et al., ICLR 2026 Oral):** Zhu et al. show isolated decentralized models are unmergeable from random or generic pre-trained starts (CLIP, ImageNet). We test whether a **shared in-task federated trunk** protects mergeability under isolation.
* **The Coupling Hypothesis (vs. Randl, EPFL 2024):** Randl observes immediate recovery after an outage, but clients gossiped continuously. We test whether instant recovery depends on cross-group coupling ($I_2$) or holds under true isolation ($I_0$).
* **The Onset Threshold ($k^*$):** We characterize the threshold onset in gradient steps beyond which models remain mergeable across heterogeneity levels $s \in [0, 1]$.
* **Recovery-Time Predictability (Singh et al., ICML 2024 HiLD):** Can a single Hessian-vector product along the merge direction predict post-merge recovery cost?

---

## Repository Structure

```
.
├── fl_outage/                  # Core research package
│   ├── models/                 # ResNet-20 with LayerNorm (272,474 parameters)
│   ├── data/                   # CIFAR-10 loaders & Restricted Dirichlet partitioner
│   ├── fl/                     # Trunk FedAvg, Outage Simulator (I0/I1/I2), Merge & Recovery
│   └── metrics/                # 11-point loss barrier, distances, curvature predictor
├── experiments/                # CLI entry points for experiment waves
├── docs/                       # Research proposals and execution plans
├── tests/                      # Automated test suite (Gate G0 verification)
├── pyproject.toml              # Package specifications
└── pytest.ini                  # Pytest configuration
```

---

## Quickstart & Verification

### 1. Environment Setup
```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Run Gate G0 Tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v
```

All 8 tests should pass in ~1 minute, validating determinism, barrier calculation sanity, disjoint Dirichlet partitions, group isolation invariance, and end-to-end smoke execution.

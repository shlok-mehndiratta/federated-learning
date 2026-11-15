# When Can Federated Learning Survive Its Server?
### Branch-and-Merge Recovery After Coordinator Outages

**Course Project:** Distributed Systems / Machine Learning  
**Team Members:** Shlok Mehndiratta, Prathamesh Lokhande, Nikhil Kumar  

When a central parameter server in federated learning experiences a temporary outage, disconnected client cohorts continue local training on non-identical data distributions. When the server returns, the diverging cohort weights are merged by simple model averaging. In this project, we investigate whether and when simple weight merging recovers the accuracy of an unbroken federated baseline, how much damage concentrates in the final classification layer versus the representation backbone, and whether retraining only the final layer on a small server-held dataset is sufficient for recovery.

---

## 1. What We Studied

We study what happens to a convolutional neural network (ResNet-20) when training splits into two isolated client cohorts during a server outage. We systematically vary when the outage starts during training, how long the outage lasts, and the degree of data heterogeneity between cohorts (including IID data, Dirichlet label skew, and synthetic feature skew). At reconnection, we merge the two diverging branches by averaging their weights and evaluate the performance penalty relative to an uninterrupted reference model trained for the exact same total budget. We inspect where damage accumulates within the network using linear probes and layer-wise canonical correlation analysis (CKA). Finally, we evaluate lightweight reconnection remedies—specifically retraining only the final classification layer, full-network fine-tuning, and output ensembling—and validate findings on CIFAR-10, CIFAR-100, and a 10-client federated averaging setup.

---

## 2. How to Set Up

### Environment
The project requires Python 3.10+ and a CUDA-compatible GPU.

```bash
# Clone repository
git clone https://github.com/shlok-mehndiratta/federated-learning.git
cd federated-learning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Directory
CIFAR-10 and CIFAR-100 datasets are downloaded automatically on first run to `data/cache/`. You can specify a custom data path by setting `DATA=/path/to/cache`.

---

## 3. How to Reproduce the Results

All experiments, verification steps, and reporting pipelines are automated through `run_all.sh`. Execute the phases in the following order:

```bash
# 1. Quick check (~3-5 min)
# Tests every code path with a minimal training budget on CIFAR-10 and CIFAR-100
DATA=data/cache bash run_all.sh quick_check

# 2. Repeatability check (~2 min)
# Verifies that two runs with the same random seed produce bitwise identical outputs
DATA=data/cache bash run_all.sh repeat_check

# 3. Check saved models from earlier experiments (~5 min, no GPU training)
# Evaluates saved model checkpoints with corrected diagnostics
DATA=data/cache bash run_all.sh recheck_saved_models

# 4. Main experiment sweep (~6-8 hours across 5 GPU seeds)
# 5 data distributions x 6 outage start times x 5 seeds, fixed outage length L=5000 (150 runs)
DATA=data/cache bash run_all.sh main

# 5. Outage length sweep (~2-3 hours)
# Sweeps outage duration L in {1000, 2500, 10000, 20000} at fixed start k=2% across 3 seeds (36 runs)
DATA=data/cache bash run_all.sh outage_length

# 6. CIFAR-100 experiment (~2-3 hours)
# Evaluates scaling to 100 classes across 3 distributions and 4 outage starts across 3 seeds (36 runs)
DATA=data/cache bash run_all.sh cifar100

# 7. Federated learning client hyperparameter test (~15 min)
# Compares client learning rates and momentum values for 10-client FedAvg
DATA=data/cache bash run_all.sh fedavg_learning_rate

# 8. Federated learning outage experiment (~4 hours)
# 10-client FedAvg outage experiment (40 outage rounds) across 3 seeds (12 runs)
LR=0.03 MOM=0.9 DATA=data/cache bash run_all.sh fedavg

# 9. Reference backbone probe calculation (~1.5 hours across 4 shards)
# Retrains the no-outage reference model to compute reference backbone probes for clean damage separation
DATA=data/cache bash run_all.sh reference_probe

# 10. Generate all tables, figures, and hypothesis test summaries (~1 min)
DATA=data/cache bash run_all.sh results
```

---

## 4. Folder Layout

```
.
├── experiments/      # Experiment runner scripts, model re-evaluations, and plotting code
├── fl_outage/        # ResNet-20 model architecture, LayerNorm, and dataset splitting utilities
├── runs/             # Raw experiment run outputs (JSON files and cached model trunks)
│   ├── main/         # Main sweep (5 distributions x 6 outage starts x 5 seeds)
│   ├── outage_length/# Outage duration sweep runs (L = 1000 to 20000)
│   ├── cifar100/     # CIFAR-100 scaling runs
│   └── fedavg/       # 10-client federated averaging outage runs
├── results/          # Compiled tables, figures, and statistical test outputs
│   ├── main/         # LaTeX tables, summary JSON, and figures for main experiments
│   └── fedavg/       # LaTeX tables for federated averaging experiments
├── tests/            # Automated test suite (determinism, parameter mapping, split isolation)
├── data/             # Cached datasets (CIFAR-10, CIFAR-100)
└── docs/             # Supporting documentation, notes, and progress reports
```

---

## 5. Where Each Report Figure and Table Comes From

Every figure and table in the final report is generated programmatically from raw run JSONs:

| Report Artifact | Producing Script | Source Run Folders | Description |
| :--- | :--- | :--- | :--- |
| `table_core.tex` | `experiments/make_tables_and_figures.py` | `runs/main/`, `runs/outage_length/`, `runs/cifar100/` | Main results table: reference accuracy, naive merge accuracy, merge penalty, and final layer gain/residual with 95% CIs. |
| `table_cures.tex` | `experiments/make_tables_and_figures.py` | `runs/main/`, `runs/outage_length/`, `runs/cifar100/` | Reconnection remedies comparison: naive merge, final layer retraining (Cure 2), full fine-tuning, ensembling, and Holm-adjusted $p$-values. |
| `table_policy.tex` | `experiments/make_tables_and_figures.py` | `runs/main/`, `runs/cifar100/` | Reconnection decision policy regret evaluated in-sample on CIFAR-10 and held out on CIFAR-100. |
| `table_fl_main.tex` | `experiments/summarize_fedavg.py` | `runs/fedavg/` | 10-client FedAvg outage results: gap to uninterrupted baseline, naive merge, and final layer recalibration gain. |
| `fig_onset_cifar10.pdf` | `experiments/make_tables_and_figures.py` | `runs/main/` | 3-panel figure: merge penalty, recoverable final layer share, and final layer retraining gain vs. outage start $k$ on CIFAR-10. |
| `fig_onset_cifar100.pdf` | `experiments/make_tables_and_figures.py` | `runs/cifar100/` | 3-panel figure: merge penalty, recoverable final layer share, and final layer retraining gain vs. outage start $k$ on CIFAR-100. |
| `fig_length_cifar10.pdf` | `experiments/make_tables_and_figures.py` | `runs/outage_length/` | Merge penalty as a function of outage duration $L \in [1000, 20000]$ steps. |
| `fig_mechanism_cifar10.pdf`| `experiments/make_tables_and_figures.py` | `runs/main/` | 4-panel diagnostic figure: linear weight interpolation paths, layer-wise CKA alignment across depth, stage-wise partial merges, and recovery curves. |
| `prereg_tests.json` | `experiments/check_hypotheses.py` | `runs/main/`, `runs/outage_length/`, `runs/cifar100/` | Formal statistical tests (paired $t$-tests, CIs, regression slopes, Holm corrections) for all planned project hypotheses. |

---

## 6. Names Used in the Code

To prevent confusion between plain terms in the report and internal identifiers in the JSON files, refer to this mapping:

| Plain Report Term | Code / JSON Identifier | Explanation |
| :--- | :--- | :--- |
| Outage start time ($k$) | `onset_pct` / `onset_rounds` | Percentage of reference training completed before the server outage begins. |
| Outage length ($L$) | `L` | Number of gradient steps (or federated rounds) during which cohorts train in isolation. |
| No-outage model | `acc_ref` / `acc_no_outage` | Test accuracy of the unbroken reference model trained for the same total steps without an outage. |
| Naive model merge | `acc_merge` / `acc_naive` | Test accuracy of the simple weight average $(W_A + W_B) / 2$ immediately upon reconnection. |
| Merge penalty | `penalty` | Accuracy drop from the naive merge relative to the no-outage model (`acc_ref - acc_merge`). |
| Final classification layer | `head` / `fc` | Final linear projection layer (`fc.weight`, `fc.bias`). |
| Rest of the network | `body` / `trunk` | All convolutional blocks and normalization layers preceding the classification head. |
| Retrain final layer (few-shot) | `acc_cure2` | Reconnection remedy: retraining only the final layer on 200 server images per class using L-BFGS. |
| Converged final layer probe | `acc_probe_converged` / `acc_probe` | Diagnostic ceiling: retraining the final layer on all clean training data while keeping the merged body fixed. |
| Reference backbone probe | `acc_probe_ref` | Diagnostic probe trained on the unbroken no-outage reference body to isolate backbone damage. |
| Full fine-tuning | `acc_cure2_fullft` | Fine-tuning the entire network on the small server dataset with a reduced learning rate. |
| Output ensemble | `acc_ensemble_full` / `acc_ensemble` | Softmax probability average of both branch models: $\frac{1}{2}(\text{softmax}(f_A(x)) + \text{softmax}(f_B(x)))$. |
| Cohort weight divergence | `d_pair` | Normalized Euclidean distance between branch weights: $\|W_A - W_B\| / \frac{1}{2}(\|W_A\| + \|W_B\|)$. |
| Final layer vs. body drift ratio | `rho` ($\rho$) | Relative drift of final layer compared to body: $(\|W_{fc,A} - W_{fc,B}\| / \|W_{fc}\|) / (\|W_{body,A} - W_{body,B}\| / \|W_{body}\|)$. |

---

## 7. Code from the First Submission

Our original preliminary submission included exploratory code that has since been superseded:
- `science_harness.py`: Initial benchmarking runner for early exploratory runs.
- `experiments/01_preliminary_metrics.py` through `experiments/05_reconnection_policy.py`: Preliminary experiment scripts from earlier project milestones; the consolidated runner in `experiments/run_outage_experiment.py` replaces them.
- `fl_outage/fl/`: Legacy federated simulation scaffolding; multi-client validation is now handled directly by `experiments/run_fedavg_outage.py`.

These legacy files are retained for project history, but all reported results in this submission are produced by the scripts in `experiments/`.

---

## 8. What Changed Since Our First Submission

We made several important corrections and methodology improvements between our first submission and this final report:
1. **Fixed Outage Duration ($L$):** In the first submission, outage length was inadvertently tied to outage start time ($L = 20000 - \text{onset}$), which confounded start time with duration. We fixed outage duration to a constant $L = 5000$ steps across all start times.
2. **Proper No-Outage Comparison Model:** We introduced an uninterrupted reference model trained for the exact same total step budget, replacing previous comparisons against pre-outage checkpoints.
3. **Disjoint Server Validation Split:** Reconnection calibration now uses a dedicated 2,000-image balanced dataset (200 images/class on CIFAR-10, 20 images/class on CIFAR-100) strictly disjoint from both client training data and test sets.
4. **Guaranteed Repeatability:** Added explicit PyTorch and CUDA deterministic workspace settings (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) and unified seed generators so that any experiment produces exact bitwise identical results when rerun.
5. **Corrected Diagnostic Analysis:** Fixed a parameter mapping error in partial merges that previously mapped all body parameters to stage 0, and replaced unweighted ratio averaging with robust paired difference tests and confidence intervals.
6. **Data Partitioning Fix:** Resolved an indexing issue in the earlier asymmetric label-skew implementation to guarantee clean class separation between cohorts.

---

## 9. Limitations & Tools Used

### Limitations
- **Cohort-Level Simulation:** Most of our experiments simulate two parallel client cohorts training via continuous SGD rather than dozens of edge devices performing local federated epochs.
- **Under-Trained FedAvg Benchmark:** In the 10-client FedAvg experiment, runs trained for 42 to 82 rounds due to compute limits, reaching ~55% accuracy on CIFAR-10. This is reported honestly as an exploratory boundary rather than a fully converged production deployment.
- **Balanced Server Calibration Data:** Our final-layer recalibration remedy (Cure 2) assumes the server holds a small balanced validation set (200 images per class), which may not always be available in strictly private or unbalanced settings.

### Tools Used
We utilized an AI coding assistant (Antigravity / Claude) to assist with experiment scripting, test generation, and LaTeX table automation. All scientific hypotheses, experimental results, and written interpretations were designed, evaluated, and verified by the authors.

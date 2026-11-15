# After a Server Outage in Federated Learning: Which Part of the Accuracy Lost by Merging Recovers, and Which Part Remains

Course project, Federated Learning, IISER Bhopal
Team: Shlok Mehndiratta, Prathamesh Lokhande, Nikhil Kumar

In federated learning, a server repeatedly averages models trained by many clients. If the server is down for a while, groups of clients (cohorts) may keep training on their own data, and when the server returns their models are averaged. This project measures how much accuracy that average loses compared with a model that never had an outage. It also measures what the loss depends on, where in the network it comes from, and which part of it disappears once training resumes.

## What we studied

- **Simulation.** We simulate an outage with ResNet-20 (with layer normalisation) on CIFAR-10 and CIFAR-100. One model is trained on all clients until the *outage start*. It is then copied three times:
  - cohort A trains on its own data;
  - cohort B trains on its own data;
  - a *no-outage model* continues on all clients.

  All three train for the same number of steps, the *outage length*.
- **What we measure.** At reconnection, A and B are averaged into the *merged model*. The *merge loss* is the no-outage accuracy minus the merged accuracy. We split it into:
  - the part removed by retraining only the final layer (*final-layer loss*);
  - the part that remains in the earlier layers (*earlier-layer loss*).
- **Repairs.** We compare simple repairs (retraining the final layer or the whole model on 200 images held by the server, keeping one or both cohort models). We also measure the *remaining gap* after 1,000 steps of resumed training.
- **How the cohorts' data differ.** Five settings: same class mix, label skew (90/10), disjoint classes, and mild or strong image shift (contrast, blur and noise on one cohort's images).

## Setup

Python 3.10 or newer and a CUDA GPU.

```bash
git clone https://github.com/shlok-mehndiratta/federated-learning.git
cd federated-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 and CIFAR-100 are downloaded to `data/cache/` on first use. Set `DATA=/path/to/cache` to use another folder.

## Reproducing the results

Every stage is one command in `run_all.sh`. Times are for one RTX PRO 6000 GPU with several runs in parallel.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `bash run_all.sh quick_check` | Runs every code path with a tiny budget | ~5 min |
| 2 | `bash run_all.sh repeat_check` | Runs one setting twice and checks the results are identical | ~5 min |
| 3 | `bash run_all.sh main` | CIFAR-10: 5 cohort settings × 6 outage starts × 5 seeds, outage length 5,000 steps (150 runs) | ~8–10 h |
| 4 | `bash run_all.sh outage_length` | CIFAR-10: outage lengths 1,000–20,000 steps at a 2% start, 3 seeds (36 runs) | ~5 h |
| 5 | `bash run_all.sh cifar100` | CIFAR-100: 3 cohort settings × 4 outage starts × 3 seeds (36 runs) | ~4 h |
| 6 | `bash run_all.sh fedavg_learning_rate` | Short learning-rate trial for the ten-client simulation | ~30 min |
| 7 | `LR=0.03 MOM=0.9 bash run_all.sh fedavg` | Ten clients in two cohorts with standard federated averaging, 3 seeds (12 runs) | ~3 h |
| 8 | `bash run_all.sh reference_probe` | Retrains the final layer of each no-outage model (needed for the loss split); repeats the no-outage training and checks it matches the saved result | ~1.5 h |
| 9 | `bash run_all.sh results` | Tables, statistical tests and diagnostic figures | ~1 min |
| 10 | `bash run_all.sh report_figures` | The two figures used in the report | ~1 min |

## Folder layout

```
experiments/   experiment and analysis scripts (see below)
fl_outage/     ResNet-20 model, layer normalisation, data loading
runs/          one JSON file per run (main/, outage_length/, cifar100/, fedavg/)
results/       tables, figures and test results produced from runs/
tests/         automated checks (data splits, layer grouping, repeatability)
PLANNED_TESTS.md     hypotheses and the tests used to judge them
CHANGES_FROM_PLAN.md every change made to that plan, with reasons
```

| Script | Purpose |
|---|---|
| `experiments/run_outage_experiment.py` | One outage experiment per run: shared training, outage, merge, measurements, repairs, resumed training |
| `experiments/add_reference_probe.py` | Adds the final-layer retraining of the no-outage model to finished runs |
| `experiments/run_fedavg_outage.py` | Ten-client federated averaging version of the experiment |
| `experiments/check_hypotheses.py` | All statistical tests reported in the report |
| `experiments/make_tables_and_figures.py` | Summary tables and diagnostic figures |
| `experiments/make_report_figures.py` | The figures used in the report |
| `experiments/summarize_fedavg.py` | Tables for the ten-client simulation |

## Where each part of the report comes from

| Report item | Source |
|---|---|
| Figure 2 (merge loss vs. outage start and length) | `results/report_figures/fig_outage_start_and_length.pdf` from `make_report_figures.py` |
| Figure 3 (resumed training; remaining gap vs. earlier-layer loss) | `results/report_figures/fig_lasting_gap.pdf` from `make_report_figures.py` |
| Table 3 (merge loss and its two parts) | `results/main/summary.json` and `check_hypotheses.py` output |
| Table 4 (layer exchange, averaging block by block) | `swap` entries in `results/main/summary.json` |
| Table 5 (repairs at reconnection) | `results/main/table_cures.tex` and `check_hypotheses.py` output |
| All p-values and confidence intervals in the text | `check_hypotheses.py` output (`results/main/prereg_tests.json`) |
| Ten-client simulation | `runs/fedavg/` and `summarize_fedavg.py` |

## Names used in the code

| Report term | Name in code / JSON | Meaning |
|---|---|---|
| Outage start | `onset_pct` | Percentage of a 25,000-step reference budget trained before the outage |
| Outage length | `L` | Steps each cohort trains alone during the outage |
| No-outage model | `acc_ref` | Accuracy of the copy that kept training on all clients |
| Merged model | `acc_merge` | Accuracy of the parameter average of cohorts A and B |
| Merge loss | `penalty` | `acc_ref - acc_merge` |
| Final layer | `fc` | The last layer, mapping 64 features to class scores |
| Earlier layers | all parameters except `fc` | First convolution and the three residual blocks |
| Final layer retrained (merged model) | `acc_probe_converged` | Final layer retrained on all client images until it stops improving, earlier layers fixed |
| Final layer retrained (no-outage model) | `acc_probe_ref` | The same, on the no-outage model |
| Earlier-layer loss | — | `acc_probe_ref - acc_probe_converged` |
| Final-layer loss | — | merge loss minus earlier-layer loss |
| Retrain final layer (repair) | `acc_cure2` | Final layer retrained on the 200 server images (20 per class on CIFAR-10), 5 epochs |
| Retrain whole model (repair) | `acc_cure2_fullft` | Whole model retrained on the same 200 images, 5 epochs |
| Keep both models | `acc_ensemble_full` | Predictions of cohorts A and B averaged |
| Keep the better cohort | `acc_best_branch` | Cohort model with higher validation accuracy |
| Layer exchange | `path_head_only_acc[0]`, `[-1]` / `path_body_only_acc[0]`, `[-1]` | Averaged earlier layers with A's / B's final layer; A's / B's earlier layers with the averaged final layer |
| Averaging block by block | `stagewise_merge_acc` | Accuracy after averaging the network up to each block |
| Resumed training | `recovery_merge`, `recovery_cure2`, `recovery_ref` | Accuracy after 50, 100, 250, 500, 1,000 steps on all clients |
| Remaining gap | — | `recovery_ref[-1] - recovery_merge[-1]` |
| Distance between cohort models | `d_pair`, `d_head`, `d_body` | Root-mean-square difference per parameter (all, final layer, earlier layers) |

## Limitations

- **Cohort-level simulation.** Each cohort is simulated as one training run on its combined data.
- **Scope.** One network (ResNet-20) and two datasets.
- **Server data.** The repairs need 200 labelled images held by the server.
- **Ten-client simulation.** It was under-trained (no-outage accuracy 40–56%), so it does not confirm the main results.

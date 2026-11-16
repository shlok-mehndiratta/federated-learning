# After a Server Outage in Federated Learning

**Which part of the accuracy lost by merging recovers, and which part remains**

Course project, Federated Learning, IISER Bhopal: Shlok Mehndiratta, Prathamesh Lokhande, Nikhil Kumar

When a federated learning server is unavailable, groups of clients (cohorts) may keep training on their own. When it returns, their models are averaged. This project measures how much accuracy that average loses compared with a model that never had an outage. It also looks at where in the network the loss comes from, and what recovers once training resumes. Experiments use ResNet-20 on CIFAR-10 and CIFAR-100, plus a ten-client federated averaging simulation.

---

## Repository layout

```
experiments/      scripts that run the experiments and analyse the results
fl_outage/        model (ResNet-20 with layer normalisation) and data loading
analysis/         reference analysis and visualization utilities (main results use experiments/)
runs/             one result file (JSON) per training run
  main/             CIFAR-10: 5 cohort settings x 6 outage starts x 5 seeds
  outage_length/    CIFAR-10: outage lengths from 1,000 to 20,000 steps
  cifar100/         CIFAR-100: 3 cohort settings x 4 outage starts x 3 seeds
  fedavg_tuning/    federated simulation: choice of training settings
  fedavg_long/      federated simulation: main runs
results/          tables, figures and statistics produced from runs/
tests/            automated checks (data splits, layer grouping, repeatability)
run_all.sh        one command per stage of the study
requirements.txt  Python packages
```

## Scripts

| Script | What it does |
|---|---|
| `experiments/run_outage_experiment.py` | Runs one outage experiment: shared training, outage, merge of the two cohort models, measurements, repairs and resumed training |
| `experiments/add_reference_probe.py` | Retrains the final layer of each no-outage model; needed to split the loss into two parts |
| `experiments/run_fedavg_outage.py` | Same experiment with ten individual clients and federated averaging (optional server momentum) |
| `experiments/check_hypotheses.py` | Statistical tests for the main experiments |
| `experiments/check_fedavg.py` | Statistical tests for the federated simulation |
| `experiments/make_tables_and_figures.py` | Summary tables and diagnostic figures |
| `experiments/make_report_figures.py` | Generates summary and publication figures |
| `experiments/summarize_fedavg.py` | Summary tables for the federated simulation |

`fl_outage/fl/`, `fl_outage/metrics/`, and `analysis/` provide reference implementations and metrics; experiment workflows are run directly via the scripts above.

## Setup

```bash
git clone https://github.com/shlok-mehndiratta/federated-learning.git
cd federated-learning
pip install -r requirements.txt
```

CIFAR-10 and CIFAR-100 are downloaded to `data/cache/` the first time they are needed. A CUDA GPU is required for the training runs.

## Reproducing the results

```bash
bash run_all.sh quick_check      # runs every code path with a tiny budget
bash run_all.sh repeat_check     # runs one setting twice and checks the results are identical
bash run_all.sh main             # main CIFAR-10 experiments
bash run_all.sh outage_length    # outage-length experiments
bash run_all.sh cifar100         # CIFAR-100 experiments
bash run_all.sh reference_probe  # final-layer retraining of the no-outage models
bash run_all.sh results          # tables, statistics, diagnostic figures
bash run_all.sh report_figures   # summary and publication figures
```

Federated simulation, with the chosen settings (server momentum 0.7, client momentum 0.9, learning rate 0.03, 50 local steps per round):

```bash
for s in 42 43 44; do for src in iid label0.8; do for k in 2 25; do
  python experiments/run_fedavg_outage.py --sources $src --onsets $k --L 80 --R 10       --batch_size 128       --server_momentum 0.7 --momentum 0.9 --lr 0.03 --local_steps 50 --seeds $s --out_dir runs/fedavg_long
done; done; done
python experiments/check_fedavg.py --in_dir runs/fedavg_long --out results/fedavg_long/fedavg_checks.json
```

## Key figures and tables

| Item | Source |
|---|---|
| Figure 2: merge loss vs. outage start and length | `results/report_figures/fig_outage_start_and_length.pdf` |
| Figure 3: merge loss split into two parts | `results/report_figures/fig_damage_split.pdf` |
| Figure 4: resumed training and remaining gap | `results/report_figures/fig_lasting_gap.pdf` |
| Tables 3–5 and the statistics in the text | `results/main/summary.json`, `results/main/table_cures.tex`, output of `check_hypotheses.py` |
| Table 6: federated simulation | `results/fedavg_long/fedavg_checks.json` |

## Key terms

| Term | Meaning |
|---|---|
| Outage start | Point in training at which the cohorts are separated (`onset_pct`) |
| Outage length | Number of steps each cohort trains alone (`L`) |
| No-outage model | Copy that keeps training on all clients for the same number of steps (`acc_ref`) |
| Merged model | Parameter average of the two cohort models (`acc_merge`) |
| Merge loss | No-outage accuracy minus merged accuracy (`penalty`) |
| Final-layer retraining | Retraining only the last layer on 200 images held by the server (`acc_cure2`) |

#!/usr/bin/env bash
# Experiment runner. Run from the repository root: bash run_all.sh <phase>
# Phases: quick_check | repeat_check | recheck_saved_models | main | outage_length | cifar100 | fedavg_learning_rate | fedavg | reference_probe | results
# Long phases start background jobs with nohup; check logs with: tail -f logs/*.log
set -euo pipefail
DATA=${DATA:-experiments/data/cache}      # CIFAR cache directory
PY=${PY:-python}
mkdir -p logs runs results/main results/fedavg
phase=${1:-help}

case "$phase" in
quick_check)          # ~3-5 min: test every code path once with a small training budget
  $PY experiments/run_outage_experiment.py --sources label1.0 feature_severe --onsets 2 --seeds 1 --L 100 --total_steps 2000 \
      --recovery_checkpoints 20 50 --out_dir runs/quick_check --trunk_cache runs/quick_check_cache --data_dir $DATA
  $PY experiments/run_outage_experiment.py --dataset cifar100 --sources label1.0 --onsets 2 --seeds 1 --L 100 --total_steps 2000 \
      --recovery_checkpoints 20 50 --out_dir runs/quick_check --trunk_cache runs/quick_check_cache --data_dir $DATA
  ;;
repeat_check)        # verify that two runs with the same seed produce identical outputs
  for d in a b; do
    $PY experiments/run_outage_experiment.py --sources label0.8 --onsets 2 --seeds 42 --L 1000 \
        --out_dir runs/repeat_check_$d --trunk_cache runs/repeat_check_cache_$d --data_dir $DATA
  done
  $PY - <<'EOF'
import json, glob
a = json.load(open(glob.glob("runs/repeat_check_a/*.json")[0]))
b = json.load(open(glob.glob("runs/repeat_check_b/*.json")[0]))
for k in ["acc_A", "acc_B", "acc_ref", "acc_merge", "acc_cure2", "acc_probe_converged"]:
    print(f"{k:22s} {a[k]:.4f} {b[k]:.4f} {'OK' if abs(a[k]-b[k])<1e-4 else 'MISMATCH'}")
EOF
  ;;
recheck_saved_models) # evaluate saved checkpoints from earlier experiments
  nohup $PY experiments/recheck_saved_models.py --model_dirs runs/stage1_science runs/stage3_feature_skew \
      --data_dir $DATA > logs/recheck_saved_models.log 2>&1 &
  ;;
main)                 # main sweep: 5 sources x 6 outage starts x 5 seeds, fixed length L=5000
  for s in 42 43 44 45 46; do
    nohup $PY experiments/run_outage_experiment.py --sources iid label0.8 label1.0 feature_mild feature_severe \
        --onsets 0.5 1 2 5 10 25 --L 5000 --seeds $s --out_dir runs/main --data_dir $DATA > logs/main_s$s.log 2>&1 &
  done
  ;;
outage_length)        # sweep outage duration at fixed outage start k=2%
  for s in 42 43 44; do
    nohup bash -c "for L in 1000 2500 10000 20000; do $PY experiments/run_outage_experiment.py --sources iid label0.8 label1.0 \
        --onsets 2 --L \$L --seeds $s --out_dir runs/outage_length --data_dir $DATA; done" > logs/outage_length_s$s.log 2>&1 &
  done
  ;;
cifar100)             # CIFAR-100 experiment across 100 classes
  for s in 42 43 44; do
    nohup $PY experiments/run_outage_experiment.py --dataset cifar100 --sources iid label0.8 label1.0 --onsets 1 2 10 25 \
        --L 5000 --seeds $s --out_dir runs/cifar100 --data_dir $DATA > logs/cifar100_s$s.log 2>&1 &
  done
  ;;
fedavg_learning_rate) # test client learning rate and momentum settings
  for cfg in "0.1 0.9" "0.03 0.9" "0.01 0.9" "0.1 0.0" "0.05 0.0"; do
    set -- $cfg
    nohup $PY experiments/run_fedavg_outage.py --sources label0.8 label1.0 --onsets 10 --L 20 --R 1 --seeds 1 \
        --lr $1 --momentum $2 --local_steps 20 --batch_size 128 --data_dir $DATA \
        --out_dir runs/fedavg_lrtest_lr$1_m$2 > logs/fedavg_lrtest_lr$1_m$2.log 2>&1 &
  done
  ;;
fedavg)               # 10-client federated averaging outage experiment
  LR=${LR:-0.03}; MOM=${MOM:-0.9}; WALPHA=${WALPHA:-}
  extra=""; [ -n "$WALPHA" ] && extra="--within_alpha $WALPHA"
  for s in 42 43 44; do
    nohup $PY experiments/run_fedavg_outage.py --sources label0.8 iid --onsets 2 25 \
        --L 40 --R 10 --local_steps 20 --batch_size 128 --lr $LR --momentum $MOM $extra --seeds $s --data_dir $DATA \
        --out_dir runs/fedavg > logs/fedavg_s$s.log 2>&1 &
  done
  ;;
reference_probe)      # retrain the reference model to compute reference backbone probes
  for sh in 0 1 2 3; do
    nohup $PY experiments/add_reference_probe.py --in_dirs runs/main runs/outage_length runs/cifar100 \
        --data_dir $DATA --shard $sh --nshards 4 > logs/reference_probe_$sh.log 2>&1 &
  done
  ;;
results)              # generate tables, figures, and hypothesis test summaries
  $PY experiments/make_tables_and_figures.py --in_dirs runs/main runs/outage_length runs/cifar100 --out_dir results/main
  $PY experiments/check_hypotheses.py --in_dirs runs/main runs/outage_length runs/cifar100 --out results/main/prereg_tests.json
  [ -d runs/fedavg ] && $PY experiments/summarize_fedavg.py --in_dir runs/fedavg --out_dir results/fedavg || true
  ;;
*) echo "usage: bash run_all.sh {quick_check|repeat_check|recheck_saved_models|main|outage_length|cifar100|fedavg_learning_rate|fedavg|reference_probe|results}";;
esac

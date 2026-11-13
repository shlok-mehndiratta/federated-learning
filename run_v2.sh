#!/usr/bin/env bash
# v2 experiment launcher. Run from the repo root:  bash run_v2.sh <phase>
# Phases: smoke | determinism | old | core | lsweep | c100 | fl_lrtest | fl_main | backfill | analyze
# All long phases start background jobs (nohup) and return immediately; follow with:  tail -f logs/v2_*.log
set -euo pipefail
DATA=${DATA:-experiments/data/cache}      # CIFAR cache (already downloaded on the workstation)
PY=${PY:-python}
mkdir -p logs runs
phase=${1:-help}

case "$phase" in
smoke)       # ~3-5 min: every code path once, tiny budget
  $PY experiments/07_outage_v2.py --sources label1.0 feature_severe --onsets 2 --seeds 1 --L 100 --total_steps 2000 \
      --recovery_checkpoints 20 50 --out_dir runs/v2_smoke --trunk_cache runs/v2_smoke_cache --data_dir $DATA
  $PY experiments/07_outage_v2.py --dataset cifar100 --sources label1.0 --onsets 2 --seeds 1 --L 100 --total_steps 2000 \
      --recovery_checkpoints 20 50 --out_dir runs/v2_smoke --trunk_cache runs/v2_smoke_cache --data_dir $DATA
  ;;
determinism) # two identical runs must agree exactly
  for d in a b; do
    $PY experiments/07_outage_v2.py --sources label0.8 --onsets 2 --seeds 42 --L 1000 \
        --out_dir runs/v2_det_$d --trunk_cache runs/v2_det_cache_$d --data_dir $DATA
  done
  $PY - <<'EOF'
import json,glob
a=json.load(open(glob.glob("runs/v2_det_a/*.json")[0])); b=json.load(open(glob.glob("runs/v2_det_b/*.json")[0]))
for k in ["acc_A","acc_B","acc_ref","acc_merge","acc_cure2","acc_probe_converged"]:
    print(f"{k:22s} {a[k]:.4f} {b[k]:.4f} {'OK' if abs(a[k]-b[k])<1e-4 else 'MISMATCH'}")
EOF
  ;;
old)         # re-analyse existing Stage 1 / Stage 3 endpoints (no training)
  nohup $PY experiments/08_reanalyze_old_endpoints.py --model_dirs runs/stage1_science runs/stage3_feature_skew \
      --data_dir $DATA > logs/v2_old.log 2>&1 &
  ;;
core)        # T1-B + T1-C: 5 sources x 6 onsets x 5 seeds, fixed L=5000 (one process per seed)
  for s in 42 43 44 45 46; do
    nohup $PY experiments/07_outage_v2.py --sources iid label0.8 label1.0 feature_mild feature_severe \
        --onsets 0.5 1 2 5 10 25 --L 5000 --seeds $s --out_dir runs/v2_core --data_dir $DATA > logs/v2_core_s$s.log 2>&1 &
  done
  ;;
lsweep)      # T1-D: outage length sweep at k=2% (L=5000 comes from 'core')
  for s in 42 43 44; do
    nohup bash -c "for L in 1000 2500 10000 20000; do $PY experiments/07_outage_v2.py --sources iid label0.8 label1.0 \
        --onsets 2 --L \$L --seeds $s --out_dir runs/v2_lsweep --data_dir $DATA; done" > logs/v2_lsweep_s$s.log 2>&1 &
  done
  ;;
c100)        # T2-C: CIFAR-100 generalisation
  for s in 42 43 44; do
    nohup $PY experiments/07_outage_v2.py --dataset cifar100 --sources iid label0.8 label1.0 --onsets 1 2 10 25 \
        --L 5000 --seeds $s --out_dir runs/v2_cifar100 --data_dir $DATA > logs/v2_c100_s$s.log 2>&1 &
  done
  ;;
fl_lrtest)   # T2-A: pick client lr / momentum for genuine FedAvg (each ~2-4 min)
  for cfg in "0.1 0.9" "0.03 0.9" "0.01 0.9" "0.1 0.0" "0.05 0.0"; do
    set -- $cfg
    nohup $PY experiments/06_fl_outage_validation.py --sources label0.8 label1.0 --onsets 10 --L 20 --R 1 --seeds 1 \
        --lr $1 --momentum $2 --local_steps 20 --batch_size 128 --data_dir $DATA \
        --out_dir runs/fl_lrtest_lr$1_m$2 > logs/fl_lrtest_lr$1_m$2.log 2>&1 &
  done
  ;;
fl_main)     # T2-B: genuine FedAvg replication; set LR/MOM from fl_lrtest, WALPHA=0.5 for Dirichlet inside cohorts
  LR=${LR:?set LR}; MOM=${MOM:?set MOM}; WALPHA=${WALPHA:-}
  extra=""; [ -n "$WALPHA" ] && extra="--within_alpha $WALPHA"
  for s in 42 43 44; do
    nohup $PY experiments/06_fl_outage_validation.py --sources iid label0.8 label1.0 feature --onsets 2 10 25 \
        --L 40 --R 10 --local_steps 20 --batch_size 128 --lr $LR --momentum $MOM $extra --seeds $s --data_dir $DATA \
        --out_dir runs/v2_fedavg > logs/v2_fedavg_s$s.log 2>&1 &
  done
  ;;
backfill)    # reference-backbone probe for finished runs (run AFTER core/lsweep/c100 finish); 4 shards in parallel
  for sh in 0 1 2 3; do
    nohup $PY experiments/09_backfill_probe_ref.py --in_dirs runs/v2_core runs/v2_lsweep runs/v2_cifar100 \
        --data_dir $DATA --shard $sh --nshards 4 > logs/v2_backfill_$sh.log 2>&1 &
  done
  ;;
analyze)
  $PY experiments/summarize_v2.py --in_dirs runs/v2_core runs/v2_lsweep runs/v2_cifar100 --out_dir docs/v2
  $PY experiments/prereg_tests_v2.py --in_dirs runs/v2_core runs/v2_lsweep runs/v2_cifar100 --out docs/v2/prereg_tests.json
  [ -d runs/v2_fedavg ] && $PY experiments/summarize_stage5.py --in_dir runs/v2_fedavg --out_dir docs/v2_fedavg || true
  ;;
*) echo "usage: bash run_v2.sh {smoke|determinism|old|core|lsweep|c100|fl_lrtest|fl_main|backfill|analyze}";;
esac

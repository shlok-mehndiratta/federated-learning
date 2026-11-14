"""
Add reference backbone probes to finished experiment runs.

This script retrains the no-outage reference model for completed runs and computes
the reference probe accuracy (acc_probe_ref). This allows cleanly separating
damage to the final layer from damage to the rest of the network:
    damage to rest of network = probe(no-outage) - probe(merged)
    damage to final layer     = penalty - damage to rest of network

Usage:
    python experiments/add_reference_probe.py --in_dirs runs/main runs/outage_length runs/cifar100 --data_dir data/cache
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, glob, importlib.util, json, sys
from types import SimpleNamespace
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
spec = importlib.util.spec_from_file_location("outage_exp", os.path.join(HERE, "run_outage_experiment.py"))
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
from fl_outage.models.resnet import ResNet20LN


def main():
    ap = argparse.ArgumentParser(description="Add reference backbone probe accuracy to completed runs.")
    ap.add_argument("--in_dirs", nargs="+", default=["runs/main", "runs/outage_length", "runs/cifar100"])
    ap.add_argument("--data_dir", default="./data/cache")
    ap.add_argument("--shard", type=int, default=0, help="process only files with index % nshards == shard")
    ap.add_argument("--nshards", type=int, default=1)
    cli = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); v2.set_determinism()
    files = sorted(f for d in cli.in_dirs for f in glob.glob(os.path.join(d, "*.json")))
    files = [f for i, f in enumerate(files) if i % cli.nshards == cli.shard]
    data_cache = {}
    for f in files:
        r = json.load(open(f))
        if "acc_merge" not in r or "acc_probe_ref" in r:
            continue
        a = SimpleNamespace(**r["args"])
        if r["dataset"] not in data_cache:
            data_cache = {r["dataset"]: v2.load_data(r["dataset"], cli.data_dir, dev)}
        X, Y, Xte, Yte = data_cache[r["dataset"]]
        C = 10 if r["dataset"] == "cifar10" else 100
        seed, onset_pct, source = r["seed"], r["onset_pct"], r["source"]
        s, level = v2.SOURCES[source]; P = v2.Pipe(r["dataset"], dev)
        spc, vpc = (100, 200) if C == 10 else (10, 20)
        server, val, pool = v2.make_splits(Y, C, seed, spc, vpc)
        A_idx, B_idx = v2.cohort_split(Y, pool, C, s, seed)
        T = lambda v: torch.tensor(v, device=dev, dtype=torch.long)
        pool_t, B_t = T(pool), T(B_idx)
        bad = None
        if level:
            bad = torch.zeros(len(Y), dtype=torch.bool, device=dev); bad[B_t] = True
        onset_step = int(round(onset_pct / 100 * a.total_steps)); end = onset_step + a.L
        trunk_sd, trunk_opt = v2.get_trunk(a, X, Y, pool_t, P, dev, seed, onset_step, C)
        m = ResNet20LN(num_classes=C).to(dev); m.load_state_dict(trunk_sd); opt = v2.new_opt(m, a, trunk_opt)
        gen = torch.Generator(device=dev).manual_seed(seed * 1000 + 4 * 10 + int(onset_pct * 10))
        v2.train_arm(m, opt, X, Y, pool_t, a.L, onset_step, end, a, P, gen, bad_mask=bad, level=level)
        sdR = v2.clone_sd(m.state_dict())
        acc_ref = v2.acc_loss(v2.predict(m, Xte, P), Yte)[0]
        r["backfill_acc_ref_reproduced"] = acc_ref
        r["backfill_ref_matches"] = abs(acc_ref - r["acc_ref"]) < 1e-4
        r["acc_probe_ref"] = v2.probe_on(m, sdR, X, Y, Xte, Yte, pool_t, P, C)
        json.dump(r, open(f, "w"), indent=1)
        print(f"{os.path.basename(f)} ref stored={r['acc_ref']*100:.2f} reproduced={acc_ref*100:.2f} "
              f"{'OK' if r['backfill_ref_matches'] else 'MISMATCH'} probe_ref={r['acc_probe_ref']*100:.2f} "
              f"probe_merge={r['acc_probe_converged']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()

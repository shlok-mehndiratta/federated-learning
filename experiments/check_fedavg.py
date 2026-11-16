"""
Statistics for the ten-client federated averaging (FedAvg) runs, computed directly from the run JSON files
written by run_fedavg_outage.py. Mirrors the tests used for the main experiments:
  - merge loss (no-outage minus merged) per setting and outage start, with 95% confidence intervals
  - effect of outage start: merge loss at the early start minus at the late start, paired by seed
  - gain of retraining the final layer on the server images, paired by seed
  - share of the merge loss recovered by a final layer retrained on all client data (ratio of means)
  - keeping both cohort models (prediction average)
  - resumed training: accuracy of each model after every resumed round, and the gap left at the end

Reads: run JSON files from --in_dir.
Writes: summary JSON file with test statistics to --out.

Example:
  python experiments/check_fedavg.py --in_dir runs/fedavg_long --out results/fedavg_long/fedavg_checks.json
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

def paired(a, b):
    d = 100 * (np.asarray(a, float) - np.asarray(b, float)); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    t = d.mean() / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 0.0
    return dict(n=n, mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), t=float(t), p=float(p))

def mean_ci(v):
    v = 100 * np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h))

def fmt(r):
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" + (f" (t={r['t']:.2f}, p={r['p']:.3g})" if "p" in r else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="runs/fedavg_long")
    ap.add_argument("--out", default="results/fedavg_long/fedavg_checks.json")
    a = ap.parse_args()
    R = [json.load(open(f)) for f in glob.glob(os.path.join(a.in_dir, "*.json"))]
    g = defaultdict(list)
    for r in R: g[(r["source"], r["onset_rounds"])].append(r)
    out = {}
    print(f"{len(R)} runs")
    for key in sorted(g):
        S = sorted(g[key], key=lambda r: r["seed"]); f = lambda k: [r[k] for r in S]
        e = dict(n=len(S), no_outage=mean_ci(f("acc_no_outage")), merged=mean_ci(f("acc_naive")),
                 merge_loss=paired(f("acc_no_outage"), f("acc_naive")),
                 final_layer_gain=paired(f("acc_cure2"), f("acc_naive")),
                 retrained_on_all_minus_merged=paired(f("acc_probe"), f("acc_naive")),
                 both_kept=mean_ci(f("acc_ensemble")), d_pair=float(np.mean(f("d_pair"))))
        e["share_fixed_by_final_layer_pct"] = 100 * e["retrained_on_all_minus_merged"]["mean"] / e["merge_loss"]["mean"]
        rn, rc, rr = (np.array([r[k] for r in S]) for k in ("recovery_naive", "recovery_cure2", "recovery_no_outage"))
        e["recovery_rounds"] = len(rn[0])
        e["recovery_merged_mean_by_round"] = list(100 * rn.mean(0))
        e["recovery_final_layer_mean_by_round"] = list(100 * rc.mean(0))
        e["recovery_no_outage_mean_by_round"] = list(100 * rr.mean(0))
        e["recovery_final_layer_per_seed_by_round"] = (100 * rc).tolist()
        e["gap_after_resumed_training"] = paired(rr[:, -1], rn[:, -1])
        e["final_layer_minus_merged_after_resumed_training"] = paired(rc[:, -1], rn[:, -1])
        out[f"{key[0]}|{key[1]}"] = e
        print(f"\n{key[0]} start={key[1]} (n={len(S)}): no-outage {e['no_outage']['mean']:.2f}, merged {e['merged']['mean']:.2f}, "
              f"both kept {e['both_kept']['mean']:.2f}, D_pair {e['d_pair']:.4f}")
        print(f"  merge loss {fmt(e['merge_loss'])}")
        print(f"  final-layer retraining gain {fmt(e['final_layer_gain'])}")
        print(f"  final layer retrained on all clients minus merged {fmt(e['retrained_on_all_minus_merged'])} "
              f"-> {e['share_fixed_by_final_layer_pct']:.1f}% of the merge loss")
        print("  resumed training, mean accuracy by round (merged / final layer retrained / no-outage):")
        for i in range(e["recovery_rounds"]):
            print(f"    round {i + 1:2d}: {e['recovery_merged_mean_by_round'][i]:6.2f} / {e['recovery_final_layer_mean_by_round'][i]:6.2f} / "
                  f"{e['recovery_no_outage_mean_by_round'][i]:6.2f}")
        print(f"  final-layer-retrained run per seed by round: " +
              " | ".join(" ".join(f"{x:.1f}" for x in row) for row in e["recovery_final_layer_per_seed_by_round"]))
        print(f"  gap to no-outage after {e['recovery_rounds']} resumed rounds {fmt(e['gap_after_resumed_training'])}")
    print("\nEffect of outage start (merge loss at the early start minus at the late start, paired by seed):")
    onset = []
    for src in sorted({k[0] for k in g}):
        ks = sorted(k[1] for k in g if k[0] == src)
        if len(ks) < 2: continue
        lo, hi = {r["seed"]: r for r in g[(src, ks[0])]}, {r["seed"]: r for r in g[(src, ks[-1])]}
        s = sorted(set(lo) & set(hi))
        res = paired([lo[i]["acc_no_outage"] - lo[i]["acc_naive"] for i in s], [hi[i]["acc_no_outage"] - hi[i]["acc_naive"] for i in s])
        res.update(source=src, early=ks[0], late=ks[-1]); onset.append(res)
    ps = np.array([r["p"] for r in onset]); order = np.argsort(ps); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[idx])); onset[idx]["p_holm"] = float(run)
    for r in onset:
        print(f"  {r['source']}: start {r['early']} vs {r['late']} rounds: {fmt(r)}, Holm p={r['p_holm']:.3g}")
    out["onset_effect"] = onset
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nWrote {a.out}")

if __name__ == "__main__":
    main()

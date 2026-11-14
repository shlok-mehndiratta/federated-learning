"""
Pre-registered tests H1, H2, H3, H5, H6, H8 computed directly from the v2 run JSONs (Tier A).
Every number reported for a hypothesis in the paper must come from this script's output.

Usage:
  python experiments/prereg_tests_v2.py --in_dirs runs/v2_core runs/v2_lsweep runs/v2_cifar100 --out docs/v2/prereg_tests.json
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

ACTIONS = {"merge": "acc_merge", "cure2": "acc_cure2", "cure2_fullft": "acc_cure2_fullft", "cure3": "acc_cure3_decoupled"}


def paired(a, b):
    """Paired difference a - b (points): mean, 95% CI, t, two-sided p, n."""
    d = 100 * (np.asarray(a, float) - np.asarray(b, float)); n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else np.nan, lo=np.nan, hi=np.nan, t=np.nan, p=np.nan)
    se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    t = d.mean() / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 0.0
    return dict(n=n, mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), t=float(t), p=float(p))


def holm(results, key="p"):
    ps = np.array([r[key] for r in results]); order = np.argsort(ps); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[idx])); results[idx]["p_holm"] = float(run)
    return results


def slope_log(xs, ys):
    """OLS slope of y (points) on log10(x) using all seed-level points, with 95% CI."""
    lr = stats.linregress(np.log10(xs), 100 * np.asarray(ys)); h = stats.t.ppf(0.975, len(xs) - 2) * lr.stderr
    return dict(slope_per_decade=float(lr.slope), lo=float(lr.slope - h), hi=float(lr.slope + h), p=float(lr.pvalue), n_points=len(xs))


def by_seed(runs, field):
    return {r["seed"]: r[field] for r in runs}


def fmt(r):
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] (n={r['n']}, t={r['t']:.2f}, p={r['p']:.2g}" + \
           (f", p_Holm={r['p_holm']:.2g})" if "p_holm" in r else ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="docs/v2/prereg_tests.json")
    ap.add_argument("--core_L", type=int, default=5000)
    ap.add_argument("--h2_onset", type=float, default=2.0)
    a = ap.parse_args()
    R = [json.load(open(f)) for d in a.in_dirs for f in glob.glob(os.path.join(d, "*.json"))]
    R = [r for r in R if "acc_merge" in r]
    cell = defaultdict(list)
    for r in R: cell[(r["dataset"], r["source"], float(r["onset_pct"]), r["L"])].append(r)
    out = {}

    # ---------------- H1: penalty(k_min) - penalty(k=25) at fixed L, per dataset x source; slope vs log k
    print("\n=== H1 (primary): onset effect at fixed L =", a.core_L)
    h1 = []
    for ds, src in sorted({(k[0], k[1]) for k in cell if k[3] == a.core_L}):
        ks = sorted(k[2] for k in cell if k[0] == ds and k[1] == src and k[3] == a.core_L)
        if len(ks) < 2: continue
        lo_k, hi_k = ks[0], ks[-1]
        A, B = by_seed(cell[(ds, src, lo_k, a.core_L)], "penalty"), by_seed(cell[(ds, src, hi_k, a.core_L)], "penalty")
        s = sorted(set(A) & set(B)); res = paired([A[i] for i in s], [B[i] for i in s])
        pts = [(r["onset_pct"], r["penalty"]) for k in ks for r in cell[(ds, src, k, a.core_L)]]
        res.update(dataset=ds, source=src, k_low=lo_k, k_high=hi_k, slope=slope_log([p[0] for p in pts], [p[1] for p in pts]))
        h1.append(res)
    holm(h1)
    for r in h1:
        print(f"{r['dataset']:8s} {r['source']:15s} penalty(k={r['k_low']:g}) - penalty(k={r['k_high']:g}) = {fmt(r)} | "
              f"slope {r['slope']['slope_per_decade']:+.2f} pts/decade [{r['slope']['lo']:+.2f}, {r['slope']['hi']:+.2f}]")
    out["H1"] = h1

    # ---------------- H2: penalty(L_max) - penalty(L_min) at k=h2_onset; slope vs log L
    print(f"\n=== H2 (exploratory): outage length at k={a.h2_onset:g}%")
    h2 = []
    for ds, src in sorted({(k[0], k[1]) for k in cell if k[2] == a.h2_onset}):
        Ls = sorted(k[3] for k in cell if k[0] == ds and k[1] == src and k[2] == a.h2_onset)
        if len(Ls) < 3: continue
        A, B = by_seed(cell[(ds, src, a.h2_onset, Ls[-1])], "penalty"), by_seed(cell[(ds, src, a.h2_onset, Ls[0])], "penalty")
        s = sorted(set(A) & set(B)); res = paired([A[i] for i in s], [B[i] for i in s])
        pts = [(r["L"], r["penalty"]) for L in Ls for r in cell[(ds, src, a.h2_onset, L)] if r["seed"] in s]
        res.update(dataset=ds, source=src, L_low=Ls[0], L_high=Ls[-1], slope=slope_log([p[0] for p in pts], [p[1] for p in pts]))
        h2.append(res)
    holm(h2)
    for r in h2:
        print(f"{r['dataset']:8s} {r['source']:15s} penalty(L={r['L_high']}) - penalty(L={r['L_low']}) = {fmt(r)} | "
              f"slope {r['slope']['slope_per_decade']:+.2f} pts/decade [{r['slope']['lo']:+.2f}, {r['slope']['hi']:+.2f}]")
    out["H2"] = h2

    # ---------------- H3: in every cell with mean penalty >= 2: probe - merge > 0 AND ref - probe > 0
    print("\n=== H3 (primary): 0 < head share < 1 (cells with penalty >= 2 pts)")
    h3 = []
    for k in sorted(cell):
        S = cell[k]
        if 100 * np.mean([r["penalty"] for r in S]) < 2: continue
        hg = paired([r["acc_probe_converged"] for r in S], [r["acc_merge"] for r in S])
        rs = paired([r["acc_ref"] for r in S], [r["acc_probe_converged"] for r in S])
        e = dict(cell="|".join(map(str, k)), head_gain=hg, residual=rs, both=bool(hg["lo"] > 0 and rs["lo"] > 0))
        if all("acc_probe_ref" in r for r in S):
            e["backbone_damage"] = paired([r["acc_probe_ref"] for r in S], [r["acc_probe_converged"] for r in S])
        h3.append(e)
    nb = sum(e["both"] for e in h3)
    print(f"cells with penalty >= 2: {len(h3)}; both CIs exclude 0: {nb}")
    for e in h3:
        if not e["both"]:
            print(f"  NOT both: {e['cell']}: head_gain {fmt(e['head_gain'])}; residual {fmt(e['residual'])}")
    out["H3"] = dict(n_cells=len(h3), n_both=nb, cells=h3)

    # ---------------- H5: Cure 2 - merge, pre-registered family = label-skew, k <= 2%, L = core_L, per dataset
    print("\n=== H5 (primary): Cure 2 - merge; family = label-skew sources, k <= 2%, L =", a.core_L, "(Holm within dataset)")
    h5 = {}
    for ds in sorted({k[0] for k in cell}):
        fam = []
        for k in sorted(cell):
            if k[0] == ds and k[1] in ("label0.8", "label1.0") and k[2] <= 2.0 and k[3] == a.core_L:
                S = cell[k]; res = paired([r["acc_cure2"] for r in S], [r["acc_merge"] for r in S]); res["cell"] = "|".join(map(str, k)); fam.append(res)
        if fam:
            holm(fam); h5[ds] = fam
            for r in fam: print(f"{r['cell']:28s} {fmt(r)}")
    out["H5"] = h5

    # ---------------- H6: Cure 2 vs full fine-tune; recovery steps for resumed training to match Cure 2
    print("\n=== H6 (exploratory): Cure 2 - full fine-tune; steps of resumed training for the merge to reach Cure 2's accuracy")
    h6 = []
    for k in sorted(cell):
        S = cell[k]
        if 100 * np.mean([r["penalty"] for r in S]) < 2: continue
        d = paired([r["acc_cure2"] for r in S], [r["acc_cure2_fullft"] for r in S])
        steps = []
        for r in S:
            cps = r["recovery_checkpoints"]; hit = [c for c, acc in zip(cps, r["recovery_merge"]) if acc >= r["acc_cure2"]]
            steps.append(hit[0] if hit else np.inf)
        gap1000 = paired([r["recovery_cure2"][-1] for r in S], [r["recovery_merge"][-1] for r in S])
        e = dict(cell="|".join(map(str, k)), cure2_minus_fullft=d, steps_to_match=[float(x) for x in steps],
                 median_steps=float(np.median(steps)), cure2_minus_merge_after_last_recovery=gap1000)
        h6.append(e)
        print(f"{e['cell']:28s} cure2-fullft {fmt(d)} | median steps for merge to reach cure2: {e['median_steps']:g} | "
              f"after {S[0]['recovery_checkpoints'][-1]} steps cure2-merge {gap1000['mean']:+.2f} [{gap1000['lo']:+.2f}, {gap1000['hi']:+.2f}]")
    out["H6"] = h6

    # ---------------- H6b (exploratory): does the outage leave a persistent cost after resumed training?
    # persistent gap = ref_resumed(last) - merge_resumed(last); compared with the reconnection-time decomposition
    print("\n=== H6b (exploratory): persistent gap after resumed training vs. backbone damage at reconnection")
    h6b, xs, ys = [], [], []
    for k in sorted(cell):
        S = cell[k]
        if 100 * np.mean([r["penalty"] for r in S]) < 2: continue
        gap = paired([r["recovery_ref"][-1] for r in S], [r["recovery_merge"][-1] for r in S])
        e = dict(cell="|".join(map(str, k)), persistent_gap=gap, penalty=100 * float(np.mean([r["penalty"] for r in S])))
        if all("acc_probe_ref" in r for r in S):
            bb = 100 * float(np.mean([r["acc_probe_ref"] - r["acc_probe_converged"] for r in S]))
            e["backbone_damage_mean"] = bb; xs.append(bb); ys.append(gap["mean"])
        h6b.append(e)
        print(f"{e['cell']:28s} penalty {e['penalty']:6.2f} | backbone dmg {e.get('backbone_damage_mean', float('nan')):6.2f} | "
              f"persistent gap after {S[0]['recovery_checkpoints'][-1]} steps {fmt(gap)}")
    if len(xs) > 2:
        rho, pr = stats.spearmanr(xs, ys); lr = stats.linregress(xs, ys)
        print(f"across cells: Spearman(backbone damage, persistent gap) = {rho:.2f} (p={pr:.2g}); "
              f"OLS slope {lr.slope:.2f}, intercept {lr.intercept:.2f}, R^2 {lr.rvalue**2:.2f}")
        out["H6b_correlation"] = dict(spearman=float(rho), p=float(pr), slope=float(lr.slope), intercept=float(lr.intercept), r2=float(lr.rvalue ** 2))
    out["H6b"] = h6b

    # ---------------- H8: held-out regret, Always Cure 2 vs Always merge (paired over runs), also with ensemble as an action
    print("\n=== H8 (exploratory): held-out (non-cifar10) regret")
    for tag, acts in [("without_ensemble", ACTIONS), ("with_ensemble", {**ACTIONS, "ensemble": "acc_ensemble_full"})]:
        te = [r for r in R if r["dataset"] != "cifar10"]
        if not te: continue
        best = np.array([max(r[v] for v in acts.values()) for r in te])
        reg = {p: 100 * (best - np.array([r[acts[p]] for r in te])) for p in acts}
        d = reg["merge"] - reg["cure2"]; n = len(d); se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
        out[f"H8_{tag}"] = dict(n_runs=n, mean_regret={p: float(v.mean()) for p, v in reg.items()},
                                max_regret={p: float(v.max()) for p, v in reg.items()},
                                merge_minus_cure2=dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h)))
        print(f"[{tag}] n={n} mean regret: " + ", ".join(f"{p}={v.mean():.2f}" for p, v in reg.items()) +
              f" | merge - cure2 regret = {d.mean():+.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

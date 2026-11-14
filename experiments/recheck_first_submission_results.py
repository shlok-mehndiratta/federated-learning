"""
Re-analyze saved results from the first submission without GPU training.

Computes corrected statistics, comparisons, and decision policy accuracy
on saved run JSON files from the first submission.

Usage:
    python experiments/recheck_first_submission_results.py [runs_dir]
"""
import json, glob, os, sys
import numpy as np
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = []
for f in glob.glob(f"{ROOT}/stage2_head_cures/result_cures_*.json") + glob.glob(f"{ROOT}/stage3_feature_skew/result_cures_*.json"):
    d = json.load(open(f)); c = d["config"]
    if c["source"] == "R3-F": cond = f"R3-F {c['corruption_level']}"
    elif c["source"] == "R2b": cond = "R2b IID"
    elif c.get("asymmetric_mode"): cond = "R3-L asym s=0.8"
    else: cond = f"R3-L s={c['s']}"
    dist = d["distances"]
    rows.append(dict(cond=cond, k=c["onset_pct"], seed=c["seed"],
        naive=d["endpoints"]["acc_naive_merged"], accA=d["endpoints"]["acc_A"], accB=d["endpoints"]["acc_B"],
        c1=d["cure1_zero_shot_logit_adj"]["best_acc"], c2=d["cure2_few_shot_recalibration"]["acc"],
        c3=d["cure3_decoupled_head_ensemble"]["prob_ens_acc"], probe=d["cure4_linear_probe_upper_bound"]["probe_acc"],
        dpair=dist["d_pair"], ratio=dist["d_head"]/dist["d_body"], cka=d["layerwise_cka"]))
print(f"Loaded {len(rows)} runs\n")
conds = sorted({(r["cond"], r["k"]) for r in rows})
ms = lambda v: f"{100*np.mean(v):5.2f}±{100*np.std(v, ddof=1):4.2f}"

print("=== Table (mean ± SD, ddof=1, n=3) ===")
print(f"{'condition':18s} {'k':>5s} | {'naive':>11s} {'cure2':>11s} {'cure3':>11s} {'probe':>11s} | probe<naive?")
for cond, k in conds:
    R = [r for r in rows if r["cond"] == cond and r["k"] == k]
    g = lambda key: [r[key] for r in R]
    flag = "YES (not a ceiling)" if np.mean(g("probe")) < np.mean(g("naive")) else ""
    print(f"{cond:18s} {k:5.1f} | {ms(g('naive'))} {ms(g('c2'))} {ms(g('c3'))} {ms(g('probe'))} | {flag}")

print("\n=== Cure 2 vs naive: paired t-test (df=2), 95% CI, d_z, Holm over all cells ===")
tests = []
for cond, k in conds:
    R = sorted([r for r in rows if r["cond"] == cond and r["k"] == k], key=lambda r: r["seed"])
    diff = np.array([r["c2"] - r["naive"] for r in R]) * 100
    t, p = stats.ttest_rel([r["c2"] for r in R], [r["naive"] for r in R])
    se = diff.std(ddof=1) / np.sqrt(len(diff)); tc = stats.t.ppf(0.975, len(diff) - 1)
    tests.append((cond, k, diff.mean(), diff.mean() - tc*se, diff.mean() + tc*se, t, p, diff.mean()/diff.std(ddof=1)))
ps = np.array([x[6] for x in tests]); order = np.argsort(ps); m = len(ps); holm = np.empty(m); run = 0
for i, idx in enumerate(order):
    run = max(run, min(1, (m - i) * ps[idx])); holm[idx] = run
for x, h in zip(tests, holm):
    print(f"{x[0]:18s} k={x[1]:4.1f}  gain={x[2]:+6.2f} pts  CI[{x[3]:+6.2f},{x[4]:+6.2f}]  t={x[5]:6.2f} p={x[6]:.4f} holm={h:.3f} d_z={x[7]:5.2f}")

print("\n=== Head share vs matched R2b probe ceiling (k=2%) ===")
r2b = {k: np.mean([r["probe"] for r in rows if r["cond"] == "R2b IID" and r["k"] == k]) for _, k in conds}
for cond, k in conds:
    if cond.startswith("R3-L") and k == 2.0:
        R = [r for r in rows if r["cond"] == cond and r["k"] == k]
        nv, pr = np.mean([r["naive"] for r in R]), np.mean([r["probe"] for r in R])
        print(f"{cond:18s} total gap={100*(r2b[k]-nv):5.2f}  head-recoverable={100*(pr-nv):5.2f}  share={100*(pr-nv)/(r2b[k]-nv):4.1f}%")

print("\n=== Layer-wise CKA (A vs B), mean ===")
for cond, k in conds:
    R = [r for r in rows if r["cond"] == cond and r["k"] == k]
    keys = list(R[0]["cka"].keys())
    print(f"{cond:18s} k={k:4.1f} " + " ".join(f"{kk.split('_')[0][:6]}={np.mean([r['cka'][kk] for r in R]):.2f}" for kk in keys))

print("\n=== Oracle action (best of naive/cure2/cure3; tie tol 0.5pt -> cheapest) vs tree label ===")
ACT = ["MERGE", "CURE2", "ENSEMBLE"]
def oracle(r):
    best = max(r["naive"], r["c2"], r["c3"])
    for a, key in zip(ACT, ["naive", "c2", "c3"]):
        if r[key] >= best - 0.005: return a
def tree(r):
    if r["k"] <= 6:
        if r["dpair"] > 0.272: return "ENSEMBLE"
        return "CURE2" if r["ratio"] > 3.551 else "MERGE"
    return "MERGE"
key = {"MERGE": "naive", "CURE2": "c2", "ENSEMBLE": "c3"}
mism, reg_tree, reg_always = 0, [], []
for r in rows:
    o, t = oracle(r), tree(r)
    best = max(r["naive"], r["c2"], r["c3"])
    reg_tree.append(best - r[key[t]]); reg_always.append(best - r["c2"])
    mism += (o != t)
print(f"Tree label disagrees with oracle on {mism}/{len(rows)} runs")
print(f"Mean regret: tree={100*np.mean(reg_tree):.2f} pts (max {100*np.max(reg_tree):.2f}); "
      f"'always Cure 2'={100*np.mean(reg_always):.2f} pts (max {100*np.max(reg_always):.2f}); "
      f"'always merge'={100*np.mean([max(r['naive'],r['c2'],r['c3'])-r['naive'] for r in rows]):.2f} pts")
from collections import Counter
print("Oracle action counts:", Counter(oracle(r) for r in rows))
for cond, k in conds:
    R = [r for r in rows if r["cond"] == cond and r["k"] == k]
    print(f"  {cond:18s} k={k:4.1f} oracle={Counter(oracle(r) for r in R).most_common()} tree={tree(R[0])}")

# ---------- LaTeX export (paste into final_report.tex) ----------
tex = [r"\begin{table*}[t]\centering\small",
       r"\caption{Head cures recomputed from per-seed result files (mean $\pm$ SD, $n=3$, ddof$=1$). $\Delta$: paired Cure~2 gain with 95\% CI (paired $t$, df$=2$); $p_{\text{Holm}}$ corrected over all 18 cells. Stage~3 probe values omitted (see text).}",
       r"\label{tab:recomputed}\resizebox{\textwidth}{!}{\begin{tabular}{llccccc c}\toprule",
       r"Condition & $k$ & Naive merge & Cure 2 (head, 200 samples) & Decoupled heads & Probe (head-only) & $\Delta$ Cure 2 [95\% CI] & $p_{\text{Holm}}$\\\midrule"]
pm = lambda v: f"${100*np.mean(v):.2f} \\pm {100*np.std(v, ddof=1):.2f}$"
for (cond, k), x, h in zip(conds, tests, holm):
    R = [r for r in rows if r["cond"] == cond and r["k"] == k]
    g = lambda key: [r[key] for r in R]
    probe = "--" if cond.startswith("R3-F") else pm(g("probe"))
    tex.append(f"{cond} & {k:g}\\% & {pm(g('naive'))} & {pm(g('c2'))} & {pm(g('c3'))} & {probe} & ${x[2]:+.2f}$ [{x[3]:+.2f}, {x[4]:+.2f}] & {h:.3f}\\\\")
tex += [r"\bottomrule\end{tabular}}\end{table*}"]
open("recomputed_tables.tex", "w").write("\n".join(tex))
print("\nWrote recomputed_tables.tex")

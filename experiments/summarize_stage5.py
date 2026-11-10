"""Summarize Stage 5 (federated validation) runs into LaTeX tables + figures.
Usage: python experiments/summarize_stage5.py --in_dir runs/stage5_fl_validation --out_dir docs/figures
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"iid": "IID cohorts", "label0.8": "Label skew $s{=}0.8$", "label1.0": "Label skew $s{=}1.0$", "feature": "Feature skew"}
ORDER = ["iid", "label0.8", "label1.0", "feature"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="runs/stage5_fl_validation")
    ap.add_argument("--out_dir", default="docs/figures")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    R = [json.load(open(f)) for f in glob.glob(os.path.join(a.in_dir, "fl_*.json"))]
    print(f"Loaded {len(R)} runs")
    cells = sorted({(r["source"], r["onset_rounds"]) for r in R}, key=lambda c: (ORDER.index(c[0]), c[1]))
    pct = lambda v: 100 * np.asarray(v)
    ms = lambda v: f"${pct(v).mean():.2f} \\pm {pct(v).std(ddof=1) if len(v) > 1 else 0:.2f}$"

    # ---------------- Table A: merge outcomes and cures
    lines = [r"\begin{table*}[t]\centering\small",
             r"\caption{Federated outage experiments (10 clients, 2 cohorts of 5, FedAvg; outage of $L$ rounds after $k$ pre-outage rounds). Mean $\pm$ SD over seeds. \emph{No outage}: global FedAvg continued for the same $L$ rounds. \emph{Cure 2}: head-only recalibration on 200 server-held images disjoint from all client data. \emph{Head ceiling}: converged head on all client data (diagnostic only). $\Delta$: paired Cure 2 gain over naive merge with 95\% CI.}",
             r"\label{tab:fl_main}\resizebox{\textwidth}{!}{\begin{tabular}{lccccccccc}\toprule",
             r"Cohort data & $k$ & No outage & Naive merge & Cure 2 & Output ens. & Head ceiling & Head share & $\Delta$ Cure 2 [95\% CI] & CKA$_{A,B}$\\\midrule"]
    summary = []
    for src, k in cells:
        S = sorted([r for r in R if r["source"] == src and r["onset_rounds"] == k], key=lambda r: r["seed"])
        g = lambda key: [r[key] for r in S]
        d = pct(g("acc_cure2")) - pct(g("acc_naive"))
        if len(d) > 1:
            se = d.std(ddof=1) / np.sqrt(len(d)); tc = stats.t.ppf(0.975, len(d) - 1)
            ci = f"${d.mean():+.2f}$ [{d.mean() - tc * se:+.2f}, {d.mean() + tc * se:+.2f}]"
            p = stats.ttest_rel(g("acc_cure2"), g("acc_naive")).pvalue
        else:
            ci, p = f"${d.mean():+.2f}$", float("nan")
        gap = np.mean(g("acc_no_outage")) - np.mean(g("acc_naive"))
        share = (np.mean(g("acc_probe")) - np.mean(g("acc_naive"))) / gap if gap > 0.01 else float("nan")
        share_s = f"{100 * share:.0f}\\%" if np.isfinite(share) else "--"
        lines.append(f"{NAMES[src]} & {k} & {ms(g('acc_no_outage'))} & {ms(g('acc_naive'))} & {ms(g('acc_cure2'))} & "
                     f"{ms(g('acc_ensemble'))} & {ms(g('acc_probe'))} & {share_s} & {ci} & {np.mean(g('cka_penult')):.2f}\\\\")
        summary.append(dict(src=src, k=k, gap=100 * gap, gain=d.mean(), p=p, share=share,
                            ratio=np.mean(g("head_body_ratio")), dpair=np.mean(g("d_pair"))))
    lines += [r"\bottomrule\end{tabular}}\end{table*}"]
    open(os.path.join(a.out_dir, "table_fl_main.tex"), "w").write("\n".join(lines))

    print("\nsource      k  | gap-to-no-outage  cure2 gain  p      head share  rho   D_pair")
    for s in summary:
        print(f"{s['src']:9s} {s['k']:3d} | {s['gap']:8.2f}        {s['gain']:+6.2f}   {s['p']:.4f}  "
              f"{100 * s['share'] if np.isfinite(s['share']) else float('nan'):6.1f}%   {s['ratio']:.2f}  {s['dpair']:.4f}")

    # ---------------- Table B: reconnection policies (regret vs per-run oracle)
    acts = {"Merge (FedAvg)": "acc_naive", "Cure 2 (head recal.)": "acc_cure2", "Output ensemble": "acc_ensemble"}
    best = np.array([max(r[v] for v in acts.values()) for r in R])
    pol = {}
    for name, key in acts.items():
        reg = 100 * (best - np.array([r[key] for r in R])); pol[name] = (reg.mean(), reg.max())
    # telemetry rule: cure only when head drift dominates (threshold from a leave-one-seed-out fit)
    seeds = sorted({r["seed"] for r in R}); regs = []
    for s_out in seeds:
        tr = [r for r in R if r["seed"] != s_out]; te = [r for r in R if r["seed"] == s_out]
        cands = sorted({r["head_body_ratio"] for r in tr})
        def loss(th, rs): return np.mean([max(r[v] for v in acts.values()) - (r["acc_cure2"] if r["head_body_ratio"] > th else r["acc_naive"]) for r in rs])
        th = min(cands, key=lambda t: loss(t, tr)) if cands else 0
        regs += [100 * (max(r[v] for v in acts.values()) - (r["acc_cure2"] if r["head_body_ratio"] > th else r["acc_naive"])) for r in te]
    pol["Telemetry rule ($\\rho$ threshold, leave-one-seed-out)"] = (np.mean(regs), np.max(regs))
    oracle = Counter(max(acts, key=lambda n: r[acts[n]]) for r in R)
    tl = [r"\begin{table}[t]\centering\small",
          r"\caption{Reconnection policies: regret (accuracy points lost vs.\ the best action for each run). Oracle-best action counts: " +
          ", ".join(f"{k}: {v}" for k, v in oracle.items()) + ".}",
          r"\label{tab:fl_policy}\begin{tabular}{lcc}\toprule Policy & Mean regret & Max regret\\\midrule"]
    tl += [f"{n} & {m:.2f} & {x:.2f}\\\\" for n, (m, x) in pol.items()]
    tl += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(a.out_dir, "table_fl_policy.tex"), "w").write("\n".join(tl))
    print("\nPolicy regret:", {k: (round(m, 2), round(x, 2)) for k, (m, x) in pol.items()}, "\nOracle:", dict(oracle))

    # ---------------- Figure: mergeability vs onset + recovery curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for src in ORDER:
        ks = sorted({k for s, k in cells if s == src})
        if not ks: continue
        y = [np.mean([100 * (r["acc_no_outage"] - r["acc_naive"]) for r in R if r["source"] == src and r["onset_rounds"] == k]) for k in ks]
        e = [np.std([100 * (r["acc_no_outage"] - r["acc_naive"]) for r in R if r["source"] == src and r["onset_rounds"] == k]) for k in ks]
        ax[0].errorbar(ks, y, yerr=e, marker="o", capsize=3, label=NAMES[src])
    ax[0].axhline(0, color="gray", lw=0.8); ax[0].set_xlabel("Pre-outage FedAvg rounds $k$")
    ax[0].set_ylabel("Merge penalty vs. no outage (pts)"); ax[0].legend(fontsize=8); ax[0].set_title("Outage onset governs the merge penalty")
    kmin = min(k for _, k in cells)
    for src, ls in [("label1.0", "-"), ("feature", "--")]:
        S = [r for r in R if r["source"] == src and r["onset_rounds"] == kmin]
        if not S: continue
        for key, lab in [("recovery_naive", "naive merge"), ("recovery_cure2", "Cure 2"), ("recovery_no_outage", "no outage")]:
            c = 100 * np.array([r[key] for r in S]).mean(0)
            ax[1].plot(range(1, len(c) + 1), c, ls, marker=".", label=f"{NAMES[src]}: {lab}")
    ax[1].set_xlabel("Rounds after reconnection"); ax[1].set_ylabel("Test accuracy (%)")
    ax[1].set_title(f"Recovery after reconnection ($k={kmin}$)"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_fl_validation.pdf")); plt.savefig(os.path.join(a.out_dir, "fig_fl_validation.png"), dpi=150)
    print(f"\nWrote table_fl_main.tex, table_fl_policy.tex, fig_fl_validation.pdf to {a.out_dir}")


if __name__ == "__main__":
    main()

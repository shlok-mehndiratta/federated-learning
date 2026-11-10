"""
Analysis for v2 runs (07_outage_v2.py; optionally FedAvg runs from 06_fl_outage_validation.py).
Every number and figure in the paper should come from this script.

Usage:
  python experiments/summarize_v2.py --in_dirs runs/v2_core runs/v2_lsweep runs/v2_cifar100 --out_dir docs/v2
Outputs (out_dir): table_core.tex, table_cures.tex, table_policy.tex, summary.json, fig_*.pdf
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_NAME = {"iid": "IID", "label0.8": "Label skew $s{=}0.8$", "label1.0": "Label skew $s{=}1.0$",
            "feature_mild": "Feature shift (mild)", "feature_severe": "Feature shift (severe)"}
ACTIONS = {"merge": "acc_merge", "cure2": "acc_cure2", "cure2_fullft": "acc_cure2_fullft", "cure3": "acc_cure3_decoupled"}


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[idx])); out[idx] = run
    return out


def load(dirs):
    R = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            if "acc_merge" in r: R.append(r)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="docs/v2")
    ap.add_argument("--train_dataset", default="cifar10", help="policy fit on this dataset, evaluated on the others")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    R = load(a.in_dirs); print(f"{len(R)} runs")
    cell = defaultdict(list)
    for r in R: cell[(r["dataset"], r["source"], r["onset_pct"], r["L"])].append(r)
    keys = sorted(cell, key=lambda k: (k[0], list(SRC_NAME).index(k[1]), k[3], k[2]))
    summary = {}

    # ---------------- core table: reference, merge, penalty, head share
    rows, pvals, cure_rows = [], [], []
    for k in keys:
        S = sorted(cell[k], key=lambda r: r["seed"]); g = lambda f: [100 * r[f] for r in S]
        pen = ci95(g("penalty")); hs = [r["head_share"] for r in S if r["head_share"] is not None]
        hsm = ci95([100 * h for h in hs]) if hs else (np.nan, np.nan, np.nan)
        d2 = np.array(g("acc_cure2")) - np.array(g("acc_merge")); dft = np.array(g("acc_cure2")) - np.array(g("acc_cure2_fullft"))
        p2 = stats.ttest_rel(g("acc_cure2"), g("acc_merge")).pvalue if len(S) > 1 else np.nan
        pvals.append(p2)
        summary["|".join(map(str, k))] = dict(n=len(S), ref=ci95(g("acc_ref")), merge=ci95(g("acc_merge")), penalty=pen,
                                                head_share=hsm, cure2_gain=ci95(d2), cure2_minus_fullft=ci95(dft), p_cure2=p2,
                                                rho=ci95([r["rho"] for r in S]), d_pair=ci95([r["d_pair"] for r in S]))
        rows.append((k, S, pen, hsm)); cure_rows.append((k, S, ci95(d2), ci95(dft)))
    ph = holm([p if np.isfinite(p) else 1.0 for p in pvals])
    fmt = lambda m: f"{m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}]" if np.isfinite(m[1]) else f"{m[0]:.2f}"
    t = [r"\begin{table*}[t]\centering\small", r"\caption{Merge penalty relative to the no-outage reference at the same step (mean [95\% CI] over seeds). Head share: fraction of the penalty recovered by a converged head-only probe on the frozen merged backbone.}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrr}\toprule Data & Cohort data & $k$ (\%) & $L$ & $n$ & No outage & Merge & Penalty (pts) & Head share (\%)\\\midrule"]
    for (k, S, pen, hsm) in rows:
        g = lambda f: np.mean([100 * r[f] for r in S])
        t.append(f"{k[0]} & {SRC_NAME[k[1]]} & {k[2]:g} & {k[3]} & {len(S)} & {g('acc_ref'):.2f} & {g('acc_merge'):.2f} & {fmt(pen)} & {fmt(hsm)}\\\\")
    t.append(r"\bottomrule\end{tabular}}\end{table*}")
    open(os.path.join(a.out_dir, "table_core.tex"), "w").write("\n".join(t))

    t = [r"\begin{table*}[t]\centering\small", r"\caption{Recovery at reconnection (accuracy \%, mean over seeds). $\Delta$: paired Cure 2 gain over the merge [95\% CI]; $p_{\mathrm{Holm}}$ over all cells. Calibration uses 200 server-held images disjoint from all client data.}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrr}\toprule Data & Cohort data & $k$ & Merge & Best branch & Ensemble & Cure 1 & Cure 2 & Cure 2 (val) & Full FT & $\Delta$ Cure 2 & $p_{\mathrm{Holm}}$\\\midrule"]
    for (k, S, d2, dft), p in zip(cure_rows, ph):
        g = lambda f: np.mean([100 * r[f] for r in S])
        t.append(f"{k[0]} & {SRC_NAME[k[1]]} & {k[2]:g} & {g('acc_merge'):.2f} & {g('acc_best_branch'):.2f} & {g('acc_ensemble_full'):.2f} & "
                 f"{g('acc_cure1'):.2f} & {g('acc_cure2'):.2f} & {g('acc_cure2_tuned'):.2f} & {g('acc_cure2_fullft'):.2f} & {fmt(d2)} & {p:.3f}\\\\")
    t.append(r"\bottomrule\end{tabular}}\end{table*}")
    open(os.path.join(a.out_dir, "table_cures.tex"), "w").write("\n".join(t))

    # ---------------- policy regret: in-sample (train dataset) and out-of-sample (other datasets)
    def regret(rs, policy):
        out = []
        for r in rs:
            best = max(r[v] for v in ACTIONS.values()); out.append(100 * (best - r[ACTIONS[policy(r)]]))
        return np.array(out)
    tr = [r for r in R if r["dataset"] == a.train_dataset]; te = [r for r in R if r["dataset"] != a.train_dataset]
    rhos = sorted({round(r["rho"], 3) for r in tr})
    def fit_rho():
        if not rhos: return None
        return min(rhos, key=lambda th: regret(tr, lambda r: "cure2" if r["rho"] > th else "merge").mean())
    th = fit_rho()
    policies = {"Always merge": lambda r: "merge", "Always Cure 2": lambda r: "cure2", "Always full fine-tune": lambda r: "cure2_fullft",
                f"Cure 2 iff $\\rho >$ {th}": (lambda r: "cure2" if r["rho"] > th else "merge") if th is not None else None}
    t = [r"\begin{table}[t]\centering\small", r"\caption{Reconnection policy regret (points lost vs.\ best action per run). Thresholds fitted on " + a.train_dataset + r" only.}",
         r"\begin{tabular}{lcccc}\toprule Policy & In-sample mean & In-sample max & Held-out mean & Held-out max\\\midrule"]
    for name, pol in policies.items():
        if pol is None: continue
        ri = regret(tr, pol) if tr else np.array([np.nan]); ro = regret(te, pol) if te else np.array([np.nan])
        t.append(f"{name} & {np.nanmean(ri):.2f} & {np.nanmax(ri):.2f} & {np.nanmean(ro):.2f} & {np.nanmax(ro):.2f}\\\\")
        summary[f"policy|{name}"] = dict(in_mean=float(np.nanmean(ri)), in_max=float(np.nanmax(ri)), out_mean=float(np.nanmean(ro)), out_max=float(np.nanmax(ro)))
    t.append(r"\bottomrule\end{tabular}\end{table}")
    open(os.path.join(a.out_dir, "table_policy.tex"), "w").write("\n".join(t))

    # ---------------- figures (all from data)
    for ds in sorted({k[0] for k in keys}):
        Ls = sorted({k[3] for k in keys if k[0] == ds}); Lmain = max(Ls, key=lambda L: sum(1 for k in keys if k[0] == ds and k[3] == L))
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for src in SRC_NAME:
            ks = sorted([k for k in keys if k[0] == ds and k[1] == src and k[3] == Lmain], key=lambda k: k[2])
            if not ks: continue
            x = [k[2] for k in ks]
            m = [ci95([100 * r["penalty"] for r in cell[k]]) for k in ks]
            ax[0].errorbar(x, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], marker="o", capsize=3, label=SRC_NAME[src])
            h = [ci95([100 * r["head_share"] for r in cell[k] if r["head_share"] is not None]) for k in ks]
            ax[1].plot(x, [v[0] for v in h], marker="o", label=SRC_NAME[src])
            d = [ci95([100 * (r["acc_cure2"] - r["acc_merge"]) for r in cell[k]]) for k in ks]
            ax[2].errorbar(x, [v[0] for v in d], yerr=[[v[0] - v[1] for v in d], [v[2] - v[0] for v in d]], marker="o", capsize=3, label=SRC_NAME[src])
        for i, (yl, tt) in enumerate([("Penalty vs. no outage (pts)", f"Merge penalty at fixed L={Lmain}"), ("Head share (%)", "Fraction recoverable by the head"), ("Cure 2 gain (pts)", "Few-shot head recalibration")]):
            ax[i].set_xscale("log"); ax[i].set_xlabel("Outage onset k (% of reference budget)"); ax[i].set_ylabel(yl); ax[i].set_title(tt); ax[i].axhline(0, color="gray", lw=0.8)
        ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, f"fig_onset_{ds}.pdf")); plt.close()

        # outage-length sweep
        ksL = [k for k in keys if k[0] == ds and len({kk[3] for kk in keys if kk[0] == ds and kk[1] == k[1] and kk[2] == k[2]}) > 1]
        if ksL:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for (src, on) in sorted({(k[1], k[2]) for k in ksL}):
                kk = sorted([k for k in ksL if k[1] == src and k[2] == on], key=lambda k: k[3])
                m = [ci95([100 * r["penalty"] for r in cell[k]]) for k in kk]
                ax.errorbar([k[3] for k in kk], [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], marker="o", capsize=3, label=f"{SRC_NAME[src]}, k={on:g}%")
            ax.set_xscale("log"); ax.set_xlabel("Outage length L (steps)"); ax.set_ylabel("Penalty vs. no outage (pts)"); ax.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, f"fig_length_{ds}.pdf")); plt.close()

        # interpolation paths, CKA depth, stage-wise merges, recovery for the earliest onset per source
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
        for src in SRC_NAME:
            ks = sorted([k for k in keys if k[0] == ds and k[1] == src and k[3] == Lmain], key=lambda k: k[2])
            if not ks: continue
            k0 = ks[0]; S = cell[k0]
            al = np.linspace(0, 1, 11); P = 100 * np.array([r["path_full_acc"] for r in S])
            ax[0].plot(al, P.mean(0), label=f"{SRC_NAME[src]} (k={k0[2]:g}%)"); ax[0].fill_between(al, P.min(0), P.max(0), alpha=0.15)
            st = ["s0", "s1", "s2", "s3", "pen"]
            ax[1].plot(range(5), [np.mean([r[f"cka_AB_{s}"] for r in S]) for s in st], marker="o", label=SRC_NAME[src])
            ax[2].plot(range(5), [100 * np.mean([r["stagewise_merge_acc"][f"prefix_{j}"] for r in S]) for j in range(5)], marker="o", label=SRC_NAME[src])
            cps = S[0]["recovery_checkpoints"]
            for tag, ls in [("merge", "-"), ("cure2", "--"), ("ref", ":")]:
                ax[3].plot([0] + cps, [100 * np.mean([r["acc_merge" if tag == "merge" else ("acc_cure2" if tag == "cure2" else "acc_ref")] for r in S])] +
                           list(100 * np.mean([r[f"recovery_{tag}"] for r in S], 0)), ls, label=f"{SRC_NAME[src]}: {tag}")
        ax[0].set_xlabel("α"); ax[0].set_ylabel("Test accuracy (%)"); ax[0].set_title("Interpolation path (mean, min–max over seeds)")
        ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["stem", "stage 1", "stage 2", "stage 3", "penult."]); ax[1].set_ylabel("Linear CKA (A vs. B)"); ax[1].set_title("Representation alignment")
        ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["stem", "+st.1", "+st.2", "+st.3", "+head"]); ax[2].set_ylabel("Accuracy (%)"); ax[2].set_title("Stage-wise merge (prefix averaged)")
        ax[3].set_xlabel("Joint training steps after reconnection"); ax[3].set_ylabel("Test accuracy (%)"); ax[3].set_title("Recovery"); ax[3].legend(fontsize=6)
        ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, f"fig_mechanism_{ds}.pdf")); plt.close()

    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(f"{k}: penalty={v['penalty'][0]:.2f} head_share={v['head_share'][0]:.1f} cure2_gain={v['cure2_gain'][0]:+.2f}"
                    for k, v in summary.items() if not k.startswith("policy")))
    print({k: v for k, v in summary.items() if k.startswith("policy")})


if __name__ == "__main__":
    main()

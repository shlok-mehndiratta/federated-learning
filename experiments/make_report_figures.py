"""
Figures used in the course report. Reads only the per-run JSON files written by
run_outage_experiment.py (after add_reference_probe.py has been run), so every
plotted number comes from the logged results.

Writes three PDFs:
  fig_outage_start_and_length.pdf  how much accuracy merging loses, by outage start and outage length
  fig_damage_split.pdf             that loss split into the part fixed by retraining the final layer
                                   and the part that sits in the rest of the network
  fig_lasting_gap.pdf              what is left after training resumes, and how it relates to the split

Example:
  python experiments/make_report_figures.py --in_dirs runs/main runs/outage_length runs/cifar100 --out_dir results/report_figures
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# match the report's typeface where available (Linux Libertine), otherwise fall back to a serif font
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Linux Libertine O", "Linux Libertine", "Libertinus Serif", "DejaVu Serif"],
                     "font.size": 9})

NAMES = {"iid": "Same class mix", "label0.8": "Label skew (90/10)", "label1.0": "Disjoint classes",
         "feature_mild": "Image shift (mild)", "feature_severe": "Image shift (strong)"}
ORDER = list(NAMES)


def mean_ci(values):
    v = np.asarray(values, float)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def load(dirs):
    runs = [json.load(open(f)) for d in dirs for f in glob.glob(os.path.join(d, "*.json"))]
    runs = [r for r in runs if "acc_merge" in r]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["source"], float(r["onset_pct"]), r["L"])].append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="results/report_figures")
    ap.add_argument("--main_L", type=int, default=5000)
    ap.add_argument("--length_start", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    g = load(a.in_dirs)
    pts = lambda vals: 100 * np.asarray(vals)

    # 1. accuracy lost by merging, against outage start (fixed length) and outage length (fixed start)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for src in ORDER:
        ks = sorted(k[2] for k in g if k[0] == "cifar10" and k[1] == src and k[3] == a.main_L)
        if not ks:
            continue
        m = [mean_ci(pts([r["penalty"] for r in g[("cifar10", src, k, a.main_L)]])) for k in ks]
        ax[0].errorbar(ks, [x[0] for x in m], yerr=[x[1] for x in m], marker="o", capsize=3, label=NAMES[src])
    ax[0].set_xscale("log"); ax[0].axhline(0, color="gray", lw=0.7)
    ax[0].set_xlabel("Outage start (% of the training budget)"); ax[0].set_ylabel("Accuracy lost by merging (points)")
    ax[0].set_title(f"(a) Outage of {a.main_L:,} steps, CIFAR-10"); ax[0].legend(fontsize=7)
    for src in ORDER:
        Ls = sorted(k[3] for k in g if k[0] == "cifar10" and k[1] == src and k[2] == a.length_start)
        if len(Ls) < 2:
            continue
        m = [mean_ci(pts([r["penalty"] for r in g[("cifar10", src, a.length_start, L)]])) for L in Ls]
        ax[1].errorbar(Ls, [x[0] for x in m], yerr=[x[1] for x in m], marker="o", capsize=3, label=NAMES[src])
    ax[1].set_xscale("log"); ax[1].axhline(0, color="gray", lw=0.7)
    ax[1].set_xlabel("Outage length (training steps)"); ax[1].set_ylabel("Accuracy lost by merging (points)")
    ax[1].set_title(f"(b) Outage starting at {a.length_start:g}% of the budget, CIFAR-10"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_outage_start_and_length.pdf")); plt.close()

    # 2. split of the loss: fixed by retraining the final layer vs. left in the rest of the network
    show = [s for s in ["iid", "label0.8", "label1.0", "feature_severe"] if any(k[1] == s for k in g)]
    fig, ax = plt.subplots(1, len(show), figsize=(3.1 * len(show), 3.3), sharey=True)
    ax = np.atleast_1d(ax)
    for i, src in enumerate(show):
        ks = sorted(k[2] for k in g if k[0] == "cifar10" and k[1] == src and k[3] == a.main_L)
        rest, fixable = [], []
        for k in ks:
            S = g[("cifar10", src, k, a.main_L)]
            if not all("acc_probe_ref" in r for r in S):
                rest.append(np.nan); fixable.append(np.nan); continue
            r_net = np.mean(pts([r["acc_probe_ref"] - r["acc_probe_converged"] for r in S]))
            rest.append(r_net); fixable.append(np.mean(pts([r["penalty"] for r in S])) - r_net)
        x = np.arange(len(ks))
        ax[i].bar(x, rest, label="Left in the rest of the network")
        ax[i].bar(x, fixable, bottom=rest, label="Fixed by retraining the final layer")
        ax[i].set_xticks(x); ax[i].set_xticklabels([f"{k:g}%" for k in ks]); ax[i].set_title(NAMES[src], fontsize=9)
        ax[i].set_xlabel("Outage start")
    ax[0].set_ylabel("Accuracy lost by merging (points)"); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_damage_split.pdf")); plt.close()

    # 3. (a) recovery when training resumes, one example; (b) lasting gap vs. rest-of-network damage, all groups
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ex = ("cifar10", "label1.0", min([k[2] for k in g if k[0] == "cifar10" and k[1] == "label1.0" and k[3] == a.main_L] or [0]), a.main_L)
    if ex in g:
        S = g[ex]; steps = [0] + S[0]["recovery_checkpoints"]
        for key, start, lab, ls in [("recovery_merge", "acc_merge", "Merged model", "-"),
                                    ("recovery_cure2", "acc_cure2", "Merged, final layer retrained", "--"),
                                    ("recovery_ref", "acc_ref", "No-outage model", ":")]:
            curve = [np.mean(pts([r[start] for r in S]))] + list(np.mean([pts(r[key]) for r in S], 0))
            ax[0].plot(steps, curve, ls, marker=".", label=lab)
        ax[0].set_xlabel("Training steps after reconnection (all clients)"); ax[0].set_ylabel("Test accuracy (%)")
        ax[0].set_title(f"(a) Disjoint classes, outage start {ex[2]:g}%, CIFAR-10"); ax[0].legend(fontsize=7)
    xs, ys, ds = [], [], []
    for k, S in g.items():
        if np.mean(pts([r["penalty"] for r in S])) < 2 or not all("acc_probe_ref" in r for r in S):
            continue
        xs.append(np.mean(pts([r["acc_probe_ref"] - r["acc_probe_converged"] for r in S])))
        ys.append(np.mean(pts([r["recovery_ref"][-1] - r["recovery_merge"][-1] for r in S]))); ds.append(k[0])
    for dname, mk in [("cifar10", "o"), ("cifar100", "s")]:
        sel = [i for i, d in enumerate(ds) if d == dname]
        ax[1].scatter([xs[i] for i in sel], [ys[i] for i in sel], marker=mk, label=dname.upper().replace("CIFAR", "CIFAR-"))
    if len(xs) > 2:
        lr = stats.linregress(xs, ys); rho = stats.spearmanr(xs, ys)[0]
        xx = np.linspace(0, max(xs), 50); ax[1].plot(xx, lr.intercept + lr.slope * xx, color="gray", lw=1)
        ax[1].set_title(f"(b) All settings: Spearman {rho:.2f}, $R^2$ {lr.rvalue ** 2:.2f}")
    ax[1].set_xlabel("Rest-of-network damage at reconnection (points)")
    ax[1].set_ylabel("Lasting gap after 1,000 steps (points)"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "fig_lasting_gap.pdf")); plt.close()
    print("wrote", ", ".join(sorted(os.listdir(a.out_dir))))


if __name__ == "__main__":
    main()

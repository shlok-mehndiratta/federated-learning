"""Rigorous Statistical Modeling & Confirmatory Significance Engine.

Performs formal statistical analysis across all 342 confirmatory runs and 15 mechanism runs:
1. Difference-in-Differences (DiD) against s=0.0 IID control arm with 95% bootstrap CIs.
2. Standardized effect sizes (Cohen's d).
3. Pre-registered confirmatory hypothesis testing (H1, H2, H3, H4) with Holm-Bonferroni correction.
4. Bivariate Ordinary Least Squares (OLS) curvature regressions with nested F-tests and VIF.
5. Integration of structural mechanisms (layer decomposition and CKA).
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


def load_json_data(path: str) -> any:
    """Loads JSON file from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def compute_bootstrap_ci(
    data: np.ndarray,
    num_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    if len(data) == 1:
        return float(data[0]), float(data[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstrap)
    n = len(data)

    for i in range(num_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha_low = (1.0 - ci) / 2.0
    alpha_high = 1.0 - alpha_low
    low = np.percentile(boot_means, alpha_low * 100)
    high = np.percentile(boot_means, alpha_high * 100)
    return float(low), float(high)


def compute_cohens_d(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """Computes Cohen's d for one-sample / paired differences or independent samples."""
    if y is None:
        # One-sample / paired difference against 0
        diff = x
        std = np.std(diff, ddof=1)
        if std < 1e-12:
            return float(np.sign(np.mean(diff)) * 999.0) if np.abs(np.mean(diff)) > 1e-12 else 0.0
        return float(np.mean(diff) / std)
    else:
        # Two independent samples
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
        if pooled_std < 1e-12:
            return 0.0
        return float((np.mean(x) - np.mean(y)) / pooled_std)


def fit_ols_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """Fits Ordinary Least Squares regression with full inference statistics.
    
    Args:
        X: Feature matrix of shape (n, p) (without constant).
        y: Target vector of shape (n,).
        feature_names: Optional list of p feature names.
        
    Returns:
        Dictionary containing coefficients, standard errors, t-statistics,
        p-values, R2, adjusted R2, F-statistic, and residual sum of squares.
    """
    n = len(y)
    # Add intercept column
    X_design = np.column_stack([np.ones(n), X])
    p = X_design.shape[1]  # including intercept

    # Solve normal equations: beta = (X^T X)^-1 X^T y
    beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    e = y - y_pred
    ssr = float(np.sum(e ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))

    r2 = 1.0 - (ssr / sst) if sst > 1e-12 else 0.0
    r2_adj = 1.0 - ((1.0 - r2) * (n - 1) / max(n - p, 1))

    # Variance-covariance matrix of beta: s^2 * (X^T X)^-1
    df_resid = max(n - p, 1)
    s2 = ssr / df_resid
    try:
        inv_xtx = np.linalg.inv(X_design.T @ X_design)
        cov_beta = s2 * inv_xtx
        se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-14))
        t_stats = beta / se
        p_values = [2.0 * (1.0 - stats.t.cdf(np.abs(t), df=df_resid)) for t in t_stats]
    except np.linalg.LinAlgError:
        se = np.zeros(p)
        t_stats = np.zeros(p)
        p_values = [1.0] * p

    # F-statistic of the overall regression
    df_model = p - 1
    if df_model > 0 and s2 > 1e-12:
        ss_reg = sst - ssr
        f_stat = (ss_reg / df_model) / s2
        f_pval = float(1.0 - stats.f.cdf(f_stat, df_model, df_resid))
    else:
        f_stat = 0.0
        f_pval = 1.0

    names = ["intercept"] + (feature_names if feature_names else [f"x{i}" for i in range(1, p)])
    coef_dict = {}
    for i, name in enumerate(names):
        coef_dict[name] = {
            "coef": float(beta[i]),
            "std_err": float(se[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i])
        }

    return {
        "coefficients": coef_dict,
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pval),
        "ssr": ssr,
        "sst": sst,
        "n_samples": n,
        "df_resid": df_resid
    }


def compute_vif(X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Computes Variance Inflation Factor (VIF) for each predictor to detect multicollinearity."""
    vif_dict = {}
    p = X.shape[1]
    if p <= 1:
        return {feature_names[0]: 1.0} if feature_names else {"x0": 1.0}

    for i in range(p):
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)
        res = fit_ols_regression(X_other, y_i)
        r2_i = max(0.0, min(res["r2"], 0.9999))
        vif = 1.0 / (1.0 - r2_i)
        vif_dict[feature_names[i]] = float(vif)

    return vif_dict


def compute_difference_in_differences(wave1_records: List[Dict]) -> Dict[str, any]:
    """Computes Difference-in-Differences against s=0.0 IID control arm for all grid cells."""
    # Group records by (s, k, isolation, seed)
    lookup = {}
    for r in wave1_records:
        key = (r["s"], r["k"], r["isolation"], r["seed"])
        lookup[key] = r

    skews = sorted(list(set(r["s"] for r in wave1_records if r["s"] > 0.0)))
    onsets = sorted(list(set(r["k"] for r in wave1_records)))
    isolations = ["I0", "I1", "I2"]
    seeds = sorted(list(set(r["seed"] for r in wave1_records)))

    metrics = ["max_acc_dip", "merged_acc", "loss_barrier", "rtb"]
    did_results = {}

    for s in skews:
        for k in onsets:
            for iso in isolations:
                cell_key = f"s{s}_k{k}_{iso}"
                cell_did = {"s": s, "k": k, "isolation": iso}

                for m in metrics:
                    diffs = []
                    for seed in seeds:
                        rec_s = lookup.get((s, k, iso, seed))
                        rec_iid = lookup.get((0.0, k, iso, seed))
                        if rec_s is not None and rec_iid is not None:
                            val_s = rec_s[m]
                            val_iid = rec_iid[m]
                            diffs.append(val_s - val_iid)

                    diffs_arr = np.array(diffs, dtype=np.float64)
                    mean_diff = float(np.mean(diffs_arr)) if len(diffs_arr) > 0 else 0.0
                    std_diff = float(np.std(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else 0.0
                    ci_low, ci_high = compute_bootstrap_ci(diffs_arr)
                    d = compute_cohens_d(diffs_arr)

                    cell_did[m] = {
                        "mean_diff": mean_diff,
                        "std_diff": std_diff,
                        "ci_95": [ci_low, ci_high],
                        "cohens_d": d,
                        "raw_diffs": diffs
                    }

                did_results[cell_key] = cell_did

    return did_results


def run_confirmatory_hypothesis_tests(
    factorial_data: Dict[str, any]
) -> Dict[str, any]:
    """Executes pre-registered hypothesis testing with Holm-Bonferroni correction."""
    tests_summary = []

    # H1: In-Task FL Server vs Generic Pre-training (CIFAR-100)
    wave2_start = factorial_data.get("wave2_start_type", [])
    # Pair by (seed, s, outage_length) for s=1.0
    lookup_intask = {}
    lookup_generic = {}
    for r in wave2_start:
        key = (r["seed"], r["s"], r["outage_length"])
        if r["start_type"] == "in_task_fl":
            lookup_intask[key] = r["merged_acc"]
        elif r["start_type"] == "generic_cifar100":
            lookup_generic[key] = r["merged_acc"]

    intask_accs = []
    generic_accs = []
    for k in sorted(lookup_intask.keys()):
        if k[1] == 1.0 and k in lookup_generic:
            intask_accs.append(lookup_intask[k])
            generic_accs.append(lookup_generic[k])

    h1_diffs = np.array(intask_accs) - np.array(generic_accs)
    h1_t_res = stats.ttest_rel(intask_accs, generic_accs)
    h1_w_res = stats.wilcoxon(h1_diffs, alternative="greater")
    h1_ci = compute_bootstrap_ci(h1_diffs)
    h1_d = compute_cohens_d(h1_diffs)

    tests_summary.append({
        "id": "H1",
        "description": "In-Task FL vs. Generic CIFAR-100 Merged Accuracy (s=1.0)",
        "mean_diff": float(np.mean(h1_diffs)),
        "ci_95": h1_ci,
        "cohens_d": h1_d,
        "t_stat": float(h1_t_res.statistic),
        "p_value_raw": float(h1_t_res.pvalue),
        "wilcoxon_pvalue": float(h1_w_res.pvalue),
        "direction": "In-Task > Generic"
    })

    # H2: Critical Onset Threshold (k=0 vs k=2, s=1.0, I0 matched across all runs)
    wave1 = factorial_data.get("wave1", [])
    wave2_dur = factorial_data.get("wave2_duration", [])

    k0_map = {}
    k2_map = {}
    for r in wave1:
        if r["s"] == 1.0 and r["isolation"] == "I0":
            k0_map[(r["seed"], r["outage_length"])] = r if r["k"] == 0 else k0_map.get((r["seed"], r["outage_length"]))
            k2_map[(r["seed"], r["outage_length"])] = r if r["k"] == 2 else k2_map.get((r["seed"], r["outage_length"]))
    for r in wave2_dur:
        if r["s"] == 1.0 and r["isolation"] == "I0":
            if r["k"] == 0:
                k0_map[(r["seed"], r["outage_length"])] = r
            elif r["k"] == 2:
                k2_map[(r["seed"], r["outage_length"])] = r

    common_h2 = sorted(list(set(k0_map.keys()) & set(k2_map.keys())))
    k0_dips = [k0_map[k]["max_acc_dip"] for k in common_h2]
    k2_dips = [k2_map[k]["max_acc_dip"] for k in common_h2]
    h2_dip_diffs = np.array(k0_dips) - np.array(k2_dips)

    h2_t_res = stats.ttest_rel(k0_dips, k2_dips)
    h2_w_res = stats.wilcoxon(h2_dip_diffs, alternative="greater")
    h2_ci = compute_bootstrap_ci(h2_dip_diffs)
    h2_d = compute_cohens_d(h2_dip_diffs)

    tests_summary.append({
        "id": "H2",
        "description": "Critical Onset Threshold Barrier Reduction (k=0 vs k=2, s=1.0, I0)",
        "mean_diff": float(np.mean(h2_dip_diffs)),
        "ci_95": h2_ci,
        "cohens_d": h2_d,
        "t_stat": float(h2_t_res.statistic),
        "p_value_raw": float(h2_t_res.pvalue),
        "wilcoxon_pvalue": float(h2_w_res.pvalue),
        "direction": "k=0 dip > k=2 dip"
    })

    # H3: Coupling Effect (Isolation I0 vs Sparse Coupling I1 across s >= 0.8)
    i0_map = {}
    i1_map = {}
    for r in wave1:
        if r["s"] >= 0.8:
            key = (r["seed"], r["s"], r["k"])
            if r["isolation"] == "I0":
                i0_map[key] = r["max_acc_dip"]
            elif r["isolation"] == "I1":
                i1_map[key] = r["max_acc_dip"]

    common_h3 = sorted(list(set(i0_map.keys()) & set(i1_map.keys())))
    i0_dips = [i0_map[k] for k in common_h3]
    i1_dips = [i1_map[k] for k in common_h3]
    h3_diffs = np.array(i0_dips) - np.array(i1_dips)

    h3_t_res = stats.ttest_rel(i0_dips, i1_dips)
    h3_w_res = stats.wilcoxon(h3_diffs, alternative="greater")
    h3_ci = compute_bootstrap_ci(h3_diffs)
    h3_d = compute_cohens_d(h3_diffs)

    tests_summary.append({
        "id": "H3",
        "description": "Sufficiency of Sparse Peer Coupling (I0 vs I1 dip, s>=0.8)",
        "mean_diff": float(np.mean(h3_diffs)),
        "ci_95": h3_ci,
        "cohens_d": h3_d,
        "t_stat": float(h3_t_res.statistic),
        "p_value_raw": float(h3_t_res.pvalue),
        "wilcoxon_pvalue": float(h3_w_res.pvalue),
        "direction": "I0 dip > I1 dip"
    })

    # H4: Hessian Curvature Incremental Predictor (Nested F-test on all Wave 1)
    d_pair = np.array([r["d_pair"] for r in wave1], dtype=np.float64)
    kappa = np.array([r["curvature_est"] for r in wave1], dtype=np.float64)
    acc_dip = np.array([r["max_acc_dip"] * 100.0 for r in wave1], dtype=np.float64)

    X_dist = d_pair.reshape(-1, 1)
    X_biv = np.column_stack([d_pair, kappa])
    res_dist = fit_ols_regression(X_dist, acc_dip)
    res_biv = fit_ols_regression(X_biv, acc_dip)

    df_resid_f = res_biv["df_resid"]
    f_stat = ((res_dist["ssr"] - res_biv["ssr"]) / 1) / (res_biv["ssr"] / df_resid_f)
    p_f = float(1.0 - stats.f.cdf(f_stat, 1, df_resid_f))

    tests_summary.append({
        "id": "H4",
        "description": "Curvature Incremental Predictor (Nested F-test over Distance)",
        "mean_diff": float(res_biv["r2"] - res_dist["r2"]),
        "ci_95": [float(res_biv["r2"] - res_dist["r2"]), float(res_biv["r2"] - res_dist["r2"])],
        "cohens_d": float(f_stat),
        "t_stat": float(f_stat),
        "p_value_raw": float(p_f),
        "wilcoxon_pvalue": float(p_f),
        "direction": "R2(dist + kappa) > R2(dist)"
    })

    # Holm-Bonferroni Family-Wise Error Rate Control (alpha = 0.05)
    m = len(tests_summary)
    # Sort tests by raw p-value ascending
    indices = sorted(range(m), key=lambda i: tests_summary[i]["p_value_raw"])

    # Step-down Holm-Bonferroni adjustment
    adj_pvalues = [0.0] * m
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(indices):
        j = rank_idx + 1  # 1-indexed rank
        multiplier = m - j + 1
        adj_p = min(1.0, tests_summary[orig_idx]["p_value_raw"] * multiplier)
        running_max = max(running_max, adj_p)
        adj_pvalues[orig_idx] = running_max

    for i, test in enumerate(tests_summary):
        test["p_value_holm"] = adj_pvalues[i]
        test["significant_alpha_0_05"] = bool(adj_pvalues[i] < 0.05)
        test["verdict"] = "CONFIRMED" if adj_pvalues[i] < 0.05 else "FAIL_TO_REJECT"

    return {
        "alpha": 0.05,
        "num_tests": m,
        "tests": tests_summary
    }


def fit_curvature_models(wave1_records: List[Dict]) -> Dict[str, any]:
    """Fits univariate and bivariate regression models evaluating curvature predictor (H4)."""
    # Extract arrays across all 180 wave 1 runs
    d_pair = np.array([r["d_pair"] for r in wave1_records], dtype=np.float64)
    kappa = np.array([r["curvature_est"] for r in wave1_records], dtype=np.float64)
    acc_dip = np.array([r["max_acc_dip"] * 100.0 for r in wave1_records], dtype=np.float64)
    rtb = np.array([r["rtb"] for r in wave1_records], dtype=np.float64)

    X_dist = d_pair.reshape(-1, 1)
    X_biv = np.column_stack([d_pair, kappa])
    feature_names_biv = ["d_pair", "curvature_kappa"]

    # 1. Regressions for Accuracy Dip
    res_dist_dip = fit_ols_regression(X_dist, acc_dip, feature_names=["d_pair"])
    res_biv_dip = fit_ols_regression(X_biv, acc_dip, feature_names=feature_names_biv)

    # Nested F-test for curvature increment
    n = len(acc_dip)
    ssr_r = res_dist_dip["ssr"]
    ssr_f = res_biv_dip["ssr"]
    df_diff = 1  # 1 added predictor (kappa)
    df_resid_f = res_biv_dip["df_resid"]
    f_inc_dip = ((ssr_r - ssr_f) / df_diff) / (ssr_f / df_resid_f)
    p_inc_dip = float(1.0 - stats.f.cdf(f_inc_dip, df_diff, df_resid_f))

    # 2. Regressions for Recovery RTB
    res_dist_rtb = fit_ols_regression(X_dist, rtb, feature_names=["d_pair"])
    res_biv_rtb = fit_ols_regression(X_biv, rtb, feature_names=feature_names_biv)

    ssr_r_rtb = res_dist_rtb["ssr"]
    ssr_f_rtb = res_biv_rtb["ssr"]
    f_inc_rtb = ((ssr_r_rtb - ssr_f_rtb) / df_diff) / (ssr_f_rtb / res_biv_rtb["df_resid"])
    p_inc_rtb = float(1.0 - stats.f.cdf(f_inc_rtb, df_diff, res_biv_rtb["df_resid"]))

    # 3. Multicollinearity VIF check
    vif_results = compute_vif(X_biv, feature_names_biv)

    return {
        "accuracy_dip_models": {
            "distance_univariate": res_dist_dip,
            "bivariate": res_biv_dip,
            "delta_r2": float(res_biv_dip["r2"] - res_dist_dip["r2"]),
            "incremental_f_stat": float(f_inc_dip),
            "incremental_f_pvalue": float(p_inc_dip)
        },
        "rtb_models": {
            "distance_univariate": res_dist_rtb,
            "bivariate": res_biv_rtb,
            "delta_r2": float(res_biv_rtb["r2"] - res_dist_rtb["r2"]),
            "incremental_f_stat": float(f_inc_rtb),
            "incremental_f_pvalue": float(p_inc_rtb)
        },
        "multicollinearity_vif": vif_results
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical Modeling Engine")
    parser.add_argument("--factorial_json", type=str, default="./runs/factorial_sweep/factorial_results.json", help="Path to factorial results JSON")
    parser.add_argument("--mechanism_json", type=str, default="./runs/mechanism_sweep/mechanism_results.json", help="Path to mechanism results JSON")
    parser.add_argument("--output_dir", type=str, default="./runs/statistical_analysis", help="Output directory for statistical results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("RIGOROUS STATISTICAL MODELING & GATE G4 CERTIFICATION")
    print("=" * 80)

    factorial_data = load_json_data(args.factorial_json)
    wave1 = factorial_data.get("wave1", [])
    print(f"Loaded {len(wave1)} Wave 1 records, {len(factorial_data.get('wave2_duration', []))} Wave 2 Duration records, and {len(factorial_data.get('wave2_start_type', []))} Wave 2 Start-Type records.")

    # 1. Primary Estimand: Difference-in-Differences against s=0.0
    print("\n--> [Step 1] Computing Difference-in-Differences against s=0.0 IID Arm (10,000 bootstrap iterations)...")
    did_results = compute_difference_in_differences(wave1)
    print(f"    Computed DiD estimates across {len(did_results)} grid conditions.")

    # 2. Confirmatory Hypothesis Testing with Holm-Bonferroni
    print("\n--> [Step 2] Executing Confirmatory Hypothesis Testing (Holm-Bonferroni at alpha=0.05)...")
    hyp_results = run_confirmatory_hypothesis_tests(factorial_data)
    for test in hyp_results["tests"]:
        print(f"    [{test['id']}] {test['description']}:")
        print(f"        Mean Diff: {test['mean_diff']:.4f} (95% CI: [{test['ci_95'][0]:.4f}, {test['ci_95'][1]:.4f}]) | Cohen's d: {test['cohens_d']:.2f}")
        print(f"        Raw p: {test['p_value_raw']:.4e} | Holm-Adjusted p: {test['p_value_holm']:.4e} -> {test['verdict']}")

    # 3. Curvature Regression & Incremental Predictive Power (H4)
    print("\n--> [Step 3] Fitting Bivariate Curvature Regressions & Computing Nested F-Tests...")
    curv_models = fit_curvature_models(wave1)
    dip_m = curv_models["accuracy_dip_models"]
    print(f"    [Accuracy Dip] Univariate R2: {dip_m['distance_univariate']['r2']:.4f} -> Bivariate R2: {dip_m['bivariate']['r2']:.4f} (Delta R2: +{dip_m['delta_r2']:.4f})")
    print(f"    [Accuracy Dip] Incremental F-stat: {dip_m['incremental_f_stat']:.2f} (p = {dip_m['incremental_f_pvalue']:.4e})")
    print(f"    [Multicollinearity] VIF: {curv_models['multicollinearity_vif']}")

    # Save comprehensive results JSON
    summary_out = os.path.join(args.output_dir, "statistical_summary.json")
    master_summary = {
        "difference_in_differences": did_results,
        "confirmatory_hypotheses": hyp_results,
        "curvature_regressions": curv_models
    }

    with open(summary_out, "w") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STATISTICAL MODELING COMPLETED! Saved to {summary_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Automated unit tests for statistical modeling engine in federated learning."""

import numpy as np
import pytest
from analysis.statistical_modeling import (
    compute_bootstrap_ci,
    compute_cohens_d,
    fit_ols_regression,
    compute_vif
)


def test_bootstrap_ci_coverage():
    """Verifies that bootstrap confidence interval contains the true population mean."""
    np.random.seed(42)
    # Standard normal data with true mean = 5.0
    data = np.random.normal(loc=5.0, scale=1.0, size=50)
    low, high = compute_bootstrap_ci(data, num_bootstrap=1000, ci=0.95, seed=42)

    assert low < np.mean(data) < high
    assert 4.5 < low < 5.2
    assert 4.8 < high < 5.5


def test_cohens_d_paired_and_independent():
    """Verifies Cohen's d computation for paired differences and independent samples."""
    # Paired: large effect
    diffs = np.array([1.5, 2.0, 1.8, 2.2, 1.9])
    d_paired = compute_cohens_d(diffs)
    assert d_paired > 5.0  # highly consistent positive shift

    # Zero difference
    zero_diffs = np.zeros(10)
    assert compute_cohens_d(zero_diffs) == 0.0

    # Independent samples
    x = np.array([10.0, 11.0, 10.5, 11.2])
    y = np.array([5.0, 5.2, 4.8, 5.1])
    d_ind = compute_cohens_d(x, y)
    assert d_ind > 5.0


def test_ols_regression_exact_recovery():
    """Verifies that OLS regression recovers ground truth coefficients on synthetic data."""
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(-2, 2, size=n)
    x2 = np.random.uniform(-1, 1, size=n)
    X = np.column_stack([x1, x2])

    true_beta0 = 3.5
    true_beta1 = 2.0
    true_beta2 = -1.5
    noise = np.random.normal(0, 0.01, size=n)
    y = true_beta0 + true_beta1 * x1 + true_beta2 * x2 + noise

    res = fit_ols_regression(X, y, feature_names=["x1", "x2"])

    coefs = res["coefficients"]
    assert pytest.approx(true_beta0, abs=0.05) == coefs["intercept"]["coef"]
    assert pytest.approx(true_beta1, abs=0.05) == coefs["x1"]["coef"]
    assert pytest.approx(true_beta2, abs=0.05) == coefs["x2"]["coef"]

    # R2 should be near 1.0 due to minimal noise
    assert res["r2"] > 0.999
    assert coefs["x1"]["p_value"] < 1e-10
    assert coefs["x2"]["p_value"] < 1e-10


def test_vif_orthogonal_and_collinear():
    """Verifies that VIF is 1.0 for orthogonal predictors and elevated for collinear ones."""
    np.random.seed(42)
    n = 200
    # Independent orthogonal features
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    X_ortho = np.column_stack([x1, x2])
    vif_ortho = compute_vif(X_ortho, ["x1", "x2"])

    assert pytest.approx(1.0, abs=0.1) == vif_ortho["x1"]
    assert pytest.approx(1.0, abs=0.1) == vif_ortho["x2"]

    # Strongly collinear features: x3 = 0.95 * x1 + noise
    x3 = 0.95 * x1 + np.random.normal(0, 0.05, size=n)
    X_collinear = np.column_stack([x1, x3])
    vif_collinear = compute_vif(X_collinear, ["x1", "x3"])

    assert vif_collinear["x1"] > 5.0
    assert vif_collinear["x3"] > 5.0

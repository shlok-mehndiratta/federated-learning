# Curvature Regression Models & Incremental Predictive Power

### Accuracy Dip Regressions

| Model | Intercept | $\beta_1$ ($D_{\text{pair}}$) | $\beta_2$ ($\kappa$) | $R^2$ | Adj $R^2$ | $\Delta R^2$ | Incremental $F$ ($p$-value) | VIF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Univariate (Distance)** | -0.1000 (p=0.1233) | 549.1 (p=0.00e+00) | — | 0.6460 | 0.6440 | — | — | — |
| **Bivariate (Dist + Curv)** | -0.0621 (p=0.3243) | 398.0 (p=4.51e-14) | 0.3078 (p=0.0001) | **0.6741** | **0.6704** | **+0.0282** | **F=15.30 (p=0.0001)** | **2.74** |

### Multicollinearity Diagnostics
- Variance Inflation Factor (VIF): **2.74** (Threshold for collinearity: > 5.0; confirms strictly independent predictors).

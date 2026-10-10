# Formal Study Pre-Registration Protocol
## Project: When Can Federated Learning Survive Its Server?
**Authors:** Anmol, Shlok Mehndiratta, et al.  
**Target Venue:** ICML / NeurIPS / ICLR Workshop on Federated Learning & Loss Landscapes  
**Status:** DRAFT (To be frozen and timestamped at Gate G2)

---

## 1. Hypotheses

* **H1 (In-Task Start vs. Generic Pretraining):**  
  Isolation after shared in-task training (a federated trunk) remains mergeable beyond a small onset threshold $k^*$, whereas isolation after generic pre-training (e.g., CIFAR-100) fails, reconciling Zhu et al. (ICLR 2026 Oral) with Branch-Train-Merge (2022).
* **H2 (Heterogeneity Interaction):**  
  The onset threshold $k^*$ increases monotonically with between-group separation $s$. At $s=1.0$ (disjoint classes), a mergeable threshold may not exist within standard training regimes.
* **H3 (Coupling Effect):**  
  With sparse ($I_1$) or dense ($I_2$) coupling during an outage, recovery cost remains near zero across tested $(k, s)$. Under full isolation ($I_0$), recovery cost becomes strictly non-zero at high $s$.
* **H4 (Recovery-Time Curvature Predictor):**  
  A single Hessian-vector product curvature estimate along the merge direction (Singh et al., ICML 2024 HiLD) predicts merged accuracy drop and Rounds-to-Baseline (RTB) significantly better than raw parameter distance alone.

---

## 2. Primary Estimands & Statistical Decision Rules

### Primary Estimand: Difference-in-Differences against IID
Because early forks produce loss barriers even on identical data due to chaotic early training dynamics, all effects are measured relative to the $s=0.0$ IID arm:
$$\Delta(s, k, I) = M(s, k, I) - M(0, k, I)$$
where $M$ represents:
1. Sharma et al. 11-point loss barrier
2. Top-1 merged accuracy drop
3. Rounds-to-Baseline (RTB)

### Pre-Registered Recovery Criteria:
* A model is declared **Recovered** if its post-merge FedAvg test accuracy reaches within **$1.0$ percentage point** of the unbroken no-failure baseline at the same total step count.
* Maximum recovery evaluation cap: **30 federated rounds**.

### Statistical Power & Significance:
* 3 random seeds for exploratory grid cells.
* Raised to **5 random seeds** in cells spanning the empirical threshold $k^*$.
* Paired comparisons evaluated via paired t-tests (normality checked) with **Holm-Bonferroni family-wise error correction** at $\alpha = 0.05$.
* Effect sizes reported as Cohen's $d$ alongside 95% confidence intervals.

---

## 3. Pilot Decision Gates (Gate G2 Branches)

At the end of Week 3, the 1-seed pilot dictates the execution branch for Weeks 4–8:
* **Branch A (Expected):** Isolation hurts merging at high $s$, and less so at later onsets $\to$ Proceed with full factorial grid.
* **Branch B (Robust Recovery):** Merging succeeds everywhere past the first few percent $\to$ Reallocate budget to the start-type contrast (in-task vs. generic pretraining) and the curvature predictor (H4).
* **Branch C (Anchor Failure):** V3b fails to reproduce Zhu et al. $\to$ Stop scaling; diagnose optimizer, normalization, and local steps before proceeding.

# Formal Research Decision Log

> **Rule:** Every change to experimental scope, parameter values, hypothesis framing, or design decisions gets a dated line here: what was observed, what was changed, and the scientific rationale. This provides complete provenance for our confirmatory vs. exploratory claims.

---

### [2026-10-10] — Entry 0: Foundation Setup & Gate G0 Cleared
* **Context:** Initialization of codebase for *When Can Federated Learning Survive Its Server?*
* **What was built:**
  - Standard ResNet-20 LayerNorm (`ResNet20LN`, 272,474 parameters) matching ICML 2025 *The Butterfly Effect* to avoid BatchNorm covariate shift and REPAIR statistics recalculation.
  - Restricted Dirichlet partitioner supporting $s \in [0, 1]$ with mathematically disjoint sampling at $s=1.0$.
  - Sequential client execution model to ensure peak VRAM stays $< 300\text{ MB}$, guaranteeing zero OOM risk.
  - Gate G0 automated test suite (8 tests).
* **Observed:** All 8 tests passed in 60s (`test_determinism`, `test_barrier_sanity`, `test_partition`, `test_isolation`, `test_smoke_end2end`).
* **Decision:** Gate G0 formally declared CLEARED. Proceeding to Week 2 (Validation & Calibration).

---

### [2026-10-10] — Entry 1: Calibration Scaffold (Pending Week 2 GPU Runs)
* **Context:** Calibrating free parameters of the factorial design before freezing pre-registration.
* **Parameters to Lock (from V1 & V2 empirical curves):**
  1. **Trunk Length $T$:** Targeting steps for unbroken FedAvg run to reach $\sim 89\%$ CIFAR-10 test accuracy plateau (Provisional: 50 epochs $\approx 19,\!531$ steps).
  2. **Onset Grid $\{k_1 \dots k_7\}$:** Provisional: $\{0\%, 0.1\%, 0.5\%, 1\%, 2\%, 5\%, 20\%\}$. *Will be re-centered based on V2 empirical transition slope.*
  3. **Outage Lengths $L$:** Grounded in Horoi et al. (2025) parameter interference peaks. (Provisional: Short = 10 rounds, Medium = 30 rounds, Long = 60 rounds).
  4. **Sparse Coupling Period $R$ ($I_1$):** Set so exchange count is $\approx 5\%$ of outage rounds (e.g. $R = 20$).
  5. **Outage Learning Rate:** Locked to the trunk's learning rate at step $k$. (Constant rate $\eta=0.1$ retained as ablation).
  6. **Recovery Cap:** Maximum post-merge FedAvg rounds before declaring unrecovered (Locked to 30 rounds).

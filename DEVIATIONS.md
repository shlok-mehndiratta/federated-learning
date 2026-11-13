# Protocol Deviations & Adjustments Log (v2)

**Document Date:** November 16, 2026  
**Repository Branch:** `v2`  
**Purpose:** Formal audit trail of any departures from or refinements to `PREREG_v2.md` and `WORKSHOP_READINESS_PLAN.md`.

---

## 1. FedAvg Learning Rate Selection (Corrected)
- **Deviation:** In the initial execution of Step 6 (`fl_main`), `LR=0.01` was mistakenly selected for the multi-client FedAvg runs.
- **Pre-Registered Protocol Rule:** §4 Step 6 dictates: *"Pick the FedAvg config from `fl_lrtest` with the highest no-outage accuracy at `label0.8`."*
- **Correction:** The `fl_lrtest` pilot logs established that `LR=0.03, MOM=0.9` achieved the highest no-outage accuracy (**37.15%**) under `label0.8` (compared to 36.60% for `LR=0.01`). The run was stopped and relaunched strictly with `LR=0.03, MOM=0.9`.

## 2. Within-Cohort Dirichlet Partitioning (`WALPHA`) Reclassification
- **Initial Action:** The first multi-client FedAvg run was launched with `--within_alpha 0.5` (`WALPHA=0.5`).
- **Correction & Reclassification:** Introducing Dirichlet partitioning within cohorts increases intra-cohort heterogeneity rather than reducing client drift. As per the primary experimental protocol, the main FedAvg evaluation must isolate between-cohort isolation by keeping clients within each cohort IID (`within_alpha = None`).
- **Disposition:** The initial `WALPHA=0.5` run was archived to `runs/v2_fedavg_walpha0.5_lr0.01_secondary/` (and its summaries to `docs/v2_fedavg_secondary_walpha/`). It is designated strictly as an exploratory secondary variant and will not be reported as the primary FedAvg baseline in the paper.

## 3. Treatment of Orthogonal Label Skew ($s=1.0$) in Multi-Client FedAvg
- **Finding:** Under extreme disjoint class partitioning ($s=1.0$), decentralized 10-client FedAvg exhibits low no-outage accuracy (~17%–27%) due to severe orthogonal client drift across decentralized local update steps.
- **Reporting Status:** In accordance with the pre-registered scenario planning (§5, H7-b), $s=1.0$ in multi-client FedAvg is formally documented and reported as a pathological edge-case boundary where standard FedAvg requires specialized drift-correction (e.g., SCAFFOLD/FedProx). The primary multi-client FedAvg conclusions focus on IID and moderate label skew ($s \le 0.8$).

## 4. Operationalisation of Head Share (Hypothesis H3)
- **Observation:** Head share computed as an unweighted mean of per-seed ratios $(\text{probe}_i - \text{merge}_i) / (\text{ref}_i - \text{merge}_i)$ exhibited numerical instability (producing spurious ratios exceeding 100% when the denominator penalty is small or near-zero).
- **Refinement:** Following observation of preliminary results ($n=3\text{--}5$), Hypothesis H3 is formally operationalised via two robust paired Student-$t$ tests:
  1. $\text{probe} - \text{merge} > 0$ (verifying that the classifier head absorbs non-zero recoverable damage), and
  2. $\text{ref} - \text{probe} > 0$ (verifying that non-zero residual damage remains permanently in the representation body).
  Head share is reported strictly as a ratio of means ($\frac{\overline{\text{probe}} - \overline{\text{merge}}}{\overline{\text{ref}} - \overline{\text{merge}}}$), and evaluated only in cells where the empirical merge penalty is non-negligible ($\ge 2.0$ points).

## 5. Specification Correction: Validation Split Sample Counts
- **Correction:** In `PREREG_v2.md` §1, the text inadvertently described the validation split as "200 images (20/class)". The true protocol implemented in code (`make_splits`, `val_pc=200` for CIFAR-10 and `val_pc=20` for CIFAR-100) carves out **200 images per class** for CIFAR-10 (2,000 images total) and **20 images per class** for CIFAR-100 (2,000 images total).

## 6. Addition of Converged Probe on Reference Backbone (`acc_probe_ref`)
- **Addition:** Added a converged head-only probe on the no-outage reference backbone (`acc_probe_ref`).
- **Rationale:** A converged head can exceed the standard SGD-trained head even without any outage occurring (resulting in negative residuals in short-outage IID cells). Therefore, the head/backbone decomposition is reported as:
  - $\text{Backbone Damage} = \text{probe}(\text{ref}) - \text{probe}(\text{merge})$, and
  - $\text{Head Damage} = \text{Penalty} - \text{Backbone Damage}$.
- **Disposition:** Supplementary to the pre-registered H3 test. Added after viewing preliminary results ($n=3\text{--}5$).

## 7. FedAvg Grid Reduction for Compute Feasibility (Hypothesis H7)
- **Adjustment:** Multi-client FedAvg grid reduced from $\{iid, label0.8, label1.0, feature\} \times \{k=2, 10, 25\} \times 3\text{ seeds}$ (36 runs, ~8 hours) to $\{label0.8, iid\} \times \{k=2, 25\} \times 3\text{ seeds}$ (12 runs total, of which the 3 completed IID $k=2$ runs are preserved and skipped).
- **Rationale:** Runtime per condition was ~40 minutes under shared GPU load, making the full 36-run grid impractical. $s=1.0$ (pathological label skew) and synthetic feature skew are deferred. Initial IID results showed a low no-outage accuracy (~49%) at 42 rounds; a longer-budget FedAvg run will be evaluated after observing the `label0.8` results.
- **Hypothesis Coverage:** This reduced grid fully preserves the pre-registered tests for Hypothesis H7: directional replication of onset penalty reduction (H1: $k=2$ vs $k=25$) and head recalibration efficacy (H5: Cure 2 vs naive merge under `label0.8`, with `iid` as negative control).


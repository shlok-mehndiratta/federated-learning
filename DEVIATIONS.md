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

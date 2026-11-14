# Pre-Registration Specification: Federated Learning Outage Merging (v2)

**Date:** November 16, 2026  
**Authors:** Shlok Mehndiratta, Prathamesh Lokhande, Nikhil Kumar  
**Target:** Workshop on Federated / Distributed Learning / Model Merging at Major ML Conference  
**Repository Branch:** `v2` (derived from `extension-work`)  
**Commit Status:** Pre-registered and committed prior to executing main empirical runs.

---

## 1. Scope & Scientific Intent

This document formalizes the pre-registered research questions, statistical hypotheses, and primary test definitions for the v2 study on federated learning coordinator outages. Pre-registering these analyses ensures that all findings are evaluated against pre-specified criteria and avoids post-hoc narrative fitting.

All empirical evaluations are conducted under a strict two-tier experimental design:
- **Tier A (Controlled Cohort Study):** Two-branch continuous SGD on partitioned data with fixed outage length $L$, controlled onset $k$, and an un-diverged no-outage reference model trained concurrently.
- **Tier B (Genuine Multi-Client FedAvg):** 10-client federated network organized into two cohorts of 5 clients with local SGD, periodic server aggregation, and uncoordinated intra-cohort training during coordinator blackouts.

Data splits strictly isolate calibration data:
- **Server Pool:** 100 images/class (CIFAR-10) or 10 images/class (CIFAR-100) carved out prior to client partitioning.
- **Validation Split:** 200 images (20/class) or 20 images (2/class) carved out prior to client partitioning for hyperparameter tuning.
- **Client Pool:** The remaining training set, partitioned into client/cohort shards with zero overlap with server data.

---

## 2. Pre-Specified Hypotheses and Primary Tests

| ID | Research Question | Formal Hypothesis | Primary Statistical Test (Pre-Specified) | Status |
|---|---|---|---|---|
| **H1** | Does onset timing matter at fixed outage length? | The merge penalty ($\text{Acc}_{\text{ref}} - \text{Acc}_{\text{merge}}$) decreases monotonically as onset $k$ increases, holding outage duration $L$ constant. | Per data source: Paired difference $\text{Penalty}(k=0.5\%) - \text{Penalty}(k=25\%)$ across matched seeds, evaluating whether the 95% confidence interval is strictly positive ($CI > 0$); and linear regression slope of penalty against $\log k$. | **Primary** |
| **H2** | Does outage length matter at fixed onset? | The merge penalty increases monotonically with outage duration $L$ at fixed early onset ($k=2\%$). | Paired difference $\text{Penalty}(L=20\text{k}) - \text{Penalty}(L=1\text{k})$ across matched seeds, evaluating whether the 95% confidence interval is strictly positive ($CI > 0$). | Exploratory |
| **H3** | Where is representational damage structurally localized? | Under label distribution skew, the head-recoverable share is strictly bounded ($0 < \text{HeadShare} < 1$) at early onset, and stage-wise prefix merging exhibits the largest marginal accuracy collapse at the final residual stage (Stage 3). | 95% confidence interval of $\text{HeadShare} = \frac{\text{Acc}_{\text{probe}} - \text{Acc}_{\text{merge}}}{\text{Acc}_{\text{ref}} - \text{Acc}_{\text{merge}}}$ excludes both $0.0$ and $1.0$; and the marginal drop in prefix-merge accuracy is largest when transitioning from Stage 2 to Stage 3 / head. | **Primary** |
| **H4** | Does the classification head absorb the local class prior? | Under label distribution skew, the pairwise difference between Branch A and Branch B classification logits linearly tracks the theoretical log class-prior ratio ($\log \frac{p_A(y)}{p_B(y)}$), whereas under IID partitions the relationship is null. | Linear regression slope $\beta$ and coefficient of determination $R^2$ of logit difference against log-prior ratio, contrasting label skew ($R^2 > 0.90$) against IID control ($R^2 \approx 0$). | Exploratory |
| **H5** | Is server-side head recalibration beneficial upon reconnection? | With a disjoint server-held calibration set, Few-Shot Head Recalibration (Cure 2) strictly outperforms naive parameter averaging at early onset ($k \le 2\%$) under label skew. | Paired seed-matched difference $\text{Acc}_{\text{Cure2}} - \text{Acc}_{\text{merge}}$ evaluated at $k \le 2\%$ for label skew sources, with family-wise significance evaluated under the Holm-Bonferroni correction at $\alpha = 0.05$. | **Primary** |
| **H6** | Is head-only recalibration superior to alternative reconnection actions? | Recalibrating only the head on 200 server-held samples offers competitive accuracy compared to full-model fine-tuning, and provides an immediate accuracy boost that requires multiple rounds ($r$ steps) of resumed training to match. | Paired confidence intervals of $\text{Acc}_{\text{Cure2}} - \text{Acc}_{\text{FullFT}}$ on the identical 200 images; and empirical quantification of the recovery horizon $r^* = \min \{r : \text{Acc}_{\text{resume}}(r) \ge \text{Acc}_{\text{Cure2}}\}$. | Exploratory |
| **H7** | Do cohort-level findings replicate in genuine multi-client FedAvg? | The core phenomena confirmed in Tier A (H1: onset penalty reduction; H5: head recalibration efficacy) replicate qualitatively in 10-client federated averaging. | Directional replication of H1 and H5 paired tests on the multi-client FedAvg execution runs. | **Primary** |
| **H8** | Does a simple reconnection decision policy generalize out-of-distribution? | An "Always Cure 2" policy incurs lower out-of-distribution regret than "Always Merge" when transferred to CIFAR-100 and FedAvg environments. | Mean and maximum policy regret computed on held-out CIFAR-100 and multi-client FedAvg benchmark sweeps, with decision rules fixed strictly on CIFAR-10 Tier A data. | Exploratory |

---

## 3. Analysis & Inference Rules

1. **Hierarchy of Claims:** Hypotheses **H1, H3, H5, and H7** are designated as **Primary**. Hypotheses H2, H4, H6, and H8 are designated as **Exploratory** and will be explicitly labeled as such in the text.
2. **Multiple Testing Correction:** The Holm-Bonferroni step-down procedure is applied strictly within each hypothesis family (e.g., across the family of $k \le 2\%$ label-skew cells for H5). No cross-family penalty is applied.
3. **Statistical Reporting Standard:** All reported scalar metrics must state the empirical mean $\pm$ 95% Student-$t$ confidence interval across seeds along with exact sample size ($n$). Effect sizes are reported as paired difference in accuracy points ($\Delta \text{Acc}$).
4. **Data Integrity Standard:** Every number, table entry, and figure data point presented in the manuscript must be programmatically derived from logged JSON artifacts generated by `experiments/make_tables_and_figures.py`. No manual transcription or hard-coded arrays are permitted.

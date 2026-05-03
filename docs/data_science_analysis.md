# Data Science Analysis — ETS MARL Thesis

This document describes the analyses we run on simulation outputs and where each one is implemented. It is organised RQ-first: every analysis is tied to a specific thesis claim and to the notebook section that produces the figures/tables behind it. We default to curriculum methods (CBS Data Science) and reach for specialised methods only when they answer a question better.

The single home for the implementation is **`notebooks/ets_marl - Data Science Analysis.ipynb`**, with cross-references to the operational/RQ notebooks where helpful (`ets_marl - Default RQ Analysis.ipynb`, `ets_marl - Sweep Analysis.ipynb`, `ets_marl - Thesis Experiments Analysis.ipynb`).

---

## 0. Core principle

Every analysis in this plan exists to support a specific claim we want to make in the thesis. Method choice follows the question, not the other way around. We use ML curriculum methods by default because that's what we know best and what the rubric rewards, and we reach for specialized methods only when they materially help answer something.

Strategy is the focus. RQ3 (how financial vs environmental objectives shape strategies, and how they aggregate) is the most distinctive thing this simulation produces and the part most worth analysing in depth. RQ2 (regulatory sensitivity) and RQ1 (simulation credibility) get lighter, more targeted treatment.

---

## 1. The thesis claims we want to support

| RQ | Claim we want to make in Ch. 7/8 |
|---|---|
| **RQ3.a** | "Distinguishable strategic archetypes emerge in the converged market." |
| **RQ3.b** | "Removing the environmental objective from agents' rewards visibly changes the strategy space — agents converge to a narrower, more uniform financial play." |
| **RQ3.c** | "Agents respond rationally to economic incentives: green-investment timing tracks the LCOE crossover, and high-fossil agents transition faster than low-fossil ones." |
| **RQ2.a** | "Regulatory levers (LRF, MSR, cap level) shift both strategies and aggregate outcomes — and they do so through identifiable channels." |
| **RQ2.b** | "Stricter regulation changes the *type* of compliance failure, not just the amount." |
| **RQ1.a** | "Agents converge to stable policies within X episodes." |
| **RQ1.b** | "Findings replicate across seeds; observed cross-variant differences exceed cross-seed noise." |

Seven claims, seven analyses. Each analysis below is the work needed to defend one claim.

---

## 2. Strategy analyses (RQ3 — primary focus)

### S1. Strategy archetypes in the converged market
**Supports claim RQ3.a.**

What strategies emerge once training has converged? Are they distinguishable, and do they map onto the agent archetypes we designed?

**Method.** Per-agent feature vectors over the converged window: mean `bid_price_A{i}`, mean `avg_bid_mult_A{i}`, mean `avg_sec_qty_A{i}`, `inv_onshore_share_A{i}`, `inv_offshore_share_A{i}`, `inv_solar_share_A{i}`, terminal `green_frac_A{i}`, mean `udbc_*_total_A{i}` shares. Standardize (Lecture 2). Cluster with **K-means** with k chosen via **Silhouette + Elbow** (Lecture 3) and confirm with **hierarchical clustering** (dendrogram, Lecture 3). Visualize in 2D with **PCA** (Lecture 5).

**The actual analytical work, not just running the methods:** compare emergent clusters against the designed archetypes (coal-heavy / balanced / green-heavy). If the clustering recovers the archetypes, the simulation is producing differentiated strategies. If it collapses them or mixes them, that's a finding too — and we need to explain why.

**Thesis output.** One PCA scatter plot coloured by cluster, one dendrogram, one table cross-tabulating designed archetype × emergent cluster. Section in Ch. 7.2.

---

### S2. How the financial-vs-environmental objective shapes strategy
**Supports claim RQ3.b.** This is the cleanest direct hit on RQ3.

We have a sweep variant where all agents are set to `reward_weights = [1.0, 0.0]` (pure financial — ESG component zeroed out). Compare the strategy space under that variant against the default (mixed reward weights).

**Method.** Run S1's clustering pipeline on the all-financial variant separately. Then compare:

- **Number and shape of clusters.** Does the strategy space collapse (fewer clusters, tighter spread within clusters) when the green objective is removed? Silhouette score and PCA spread give us the comparison.
- **Cluster centroids.** What do the all-financial centroids look like vs. the default centroids in the same feature space? Project both sets of centroids into the same PCA space (fit PCA on default, transform all-financial) so the comparison is visually direct.
- **Aggregate outcomes by cluster.** Compute mean `green_frac`, `clearing_price`, `penalty` per cluster in each variant. Does removing ESG weighting flatten green investment across the board, or does it just flatten it for the agents who would have been green-leaning?

**Why this is the headline RQ3 analysis.** It directly tests the causal story we want to tell: the environmental component of the reward is what produces strategic heterogeneity. If the strategy space collapses when we remove it, that's evidence. If it doesn't collapse much, that's also a finding — agents found other reasons to differentiate (e.g. cost-structure heterogeneity, urgency scalars).

**Thesis output.** Two-panel PCA figure (default vs all-financial) on the same axes. Cluster summary table with side-by-side mean outcomes. Section in Ch. 7.2 — likely the most-cited figure in the thesis.

---

### S3. Are strategies economically rational?
**Supports claim RQ3.c.** Tier 2 — we ship S3 if S1 and S2 are clean. This section keeps only a compact anchor-alignment credibility check from the old clearing-price-prediction direction (not the removed prediction model).

We have two specific predictions from theory in Ch. 3 plus one credibility check.

**H1 (LCOE crossover).** "Agents invest in green when carbon price exceeds ~50 €/t." Per (agent, year), label = 1 if `invest_cost_A{i} > threshold`. Features: lagged `clearing_price`, agent archetype, `bank_start_A{i}`, year. Fit **logistic regression** with L1/LASSO (Lectures 4 + 8) and a **Random Forest** (Lecture 5) for non-linear effects. We're looking for: (i) positive significant coefficient on lagged clearing price in the logistic, and (ii) a step-up in RF partial-dependence around the 50 €/t mark.

**H2 (heterogeneity).** "High-fossil agents transition faster." Label = 1 if agent invested meaningfully in green by year 6. Features: archetype dummies, initial mix vector, mean clearing price the agent saw. Same model setup. Coefficient sign on archetype dummies is the test.

**Anchor check (mini, embedded here).** Does the converged-window mean clearing price track the fundamental anchor we computed in `src/utils/price_anchor.py`? Plot the two trajectories together over years `0..(n_years-1)` (default 0–11 in this repo) and report the correlation coefficient and mean absolute deviation. This is one figure and two numbers, not a full predictive model — that's the right scope. If they align, the simulation is producing economically grounded prices, which feeds RQ1 too (cross-referenced in C1/C2 as a credibility signal).

**Thesis output.** One table summarising H1 and H2 (estimate, sign, evaluation metric), one partial-dependence plot for H1, one anchor-vs-realised figure. Section in Ch. 7.3.

---

## 3. Regulatory analyses (RQ2)

### R1. Do regulatory levers shift strategies and outcomes?
**Supports claim RQ2.a.**

This is where most of the sweep variants pay off (LRF 2.2 / 4.3 / 6.0, MSR on/off, cap levels). The question is: do these regulatory dials change agent strategies, or do agents just absorb the change without restructuring how they play?

**Method.** Two angles, both at the strategy and outcome level.

**Strategy angle.** Re-run S1's clustering on each variant. Track how cluster structure changes: does tighter LRF push agents into a "more aggressive green investor" cluster? Does MSR-off let a "free-rider" cluster emerge? Project all variants' agents into the default-fitted PCA space and visualise where each variant's agents land.

**Outcome angle.** Mean ± std across seeds for headline outcomes (clearing price, green frac final, penalty total, secondary volume), per variant. Boxplots side-by-side, with IQR-based outlier flagging (Lecture 7) so a single rogue seed doesn't drive conclusions.

**Why both.** We claim regulation matters when *both* outcomes shift *and* the underlying strategies shift. If outcomes shift but strategies don't, agents are just being squeezed — boring. If strategies shift, agents are *adapting* — that's the interesting story, and it's exactly what RQ2 is asking about.

**Thesis output.** Multi-panel PCA figure (one panel per variant in the same projected space). Outcome comparison table. Section in Ch. 7.4.

---

### R2. UDBC compliance pathways
**Supports claim RQ2.b.**

The `udbc_*` columns already classify each (agent, year) into {U, D, M, B, C} — five compliance failure/success modes. We exploit this directly.

**Method.** Two analyses, both pre-existing in v2.

- **Multinomial logistic regression** (Lecture 4 explicitly covers this) predicting UDBC class from agent + market features (`bank_start_A{i}`, `cap`, `clearing_price`, archetype, year, variant). Coefficients tell us which conditions push toward each failure mode. **Random Forest** (Lecture 5) as the non-linear robustness check.
- **Transition heatmaps** per variant showing year-t → year-t+1 UDBC transitions. Descriptive (frequency tables, Lecture 2). The interesting comparison: does tighter LRF produce more U-state recoveries, or more cascades into D?

**Class imbalance.** Compliant (C) dominates. Apply **SMOTE or ADASYN** (Lecture 7) before fitting the multinomial, report results with and without — the comparison is itself informative (if SMOTE results differ a lot, the model was being driven by the majority class, which is honest to acknowledge).

**Thesis output.** One coefficient table from the multinomial logit, one transition-heatmap figure with one panel per variant. Section in Ch. 7.4 alongside R1.

---

## 4. Credibility analyses (RQ1)

These are short — they open Ch. 7 and establish that the rest of the analyses sit on a stable foundation. We don't want to spend too much page count here, but we can't skip them.

### C1. Convergence
**Supports claim RQ1.a.**

**Method.** Two detectors on episode-level reward and `ep_mean_clearing_price`:
- **Rolling-mean plateau check** (Lecture 2) — first episode where the rolling mean stays inside a tolerance band for K consecutive windows.
- **PELT change-point detection** (specialized, `ruptures` library) — objective change-point.

Use both, report the convergence episode where they agree. If they disagree, use the later cutoff as the conservative boundary and flag the run as detector-discordant in the methods notes. The resulting cutoff is what we use for everything in §2 and §3.

**Thesis output.** Two figures (reward and price trajectories with marked convergence point) and a short paragraph. Maybe 1.5 pages in Ch. 7.1.

---

### C2. Reproducibility across seeds
**Supports claim RQ1.b.**

**Method.** For six headline metrics (`quality_score`, `ep_mean_clearing_price`, `clearing_price_last`, mean `green_frac_A{i}`, mean `penalty_A{i}`, `secondary_volume`), compute mean ± std across seeds within episode bins. Fan charts (median + ± 1 std band). For the converged window, a small table of (metric, mean across seeds, std across seeds). Lecture-2-style EDA on the seed dimension.

This also serves as the validity floor for R1: if cross-seed std on the default config is comparable to cross-variant differences, we have to caveat R1 heavily. Reporting both side-by-side is the honest move.

**Thesis output.** One fan-chart figure, one summary table. ~1 page in Ch. 7.1.

---

## 5. Method inventory & curriculum mapping

For Ch. 5 Methodology. Demonstrates every method is curriculum-grounded except where we explicitly justify otherwise. The "Implemented in" column points at the notebook section that produces the figures and tables.

| Analysis | Methods | Lecture | Implemented in |
|---|---|---|---|
| S1 strategy clustering | K-means + Hierarchical + Silhouette/Elbow + PCA | L3 + L5 | `Data Science Analysis.ipynb` §S1 |
| S2 reward-weight effect on strategy | S1 pipeline applied to the variant; centroid comparison | L3 + L5 + L2 | `Data Science Analysis.ipynb` §S2 (variant logs from `configs/sweeps/thesis_experiments.yaml` `all_fin` / `balanced`) |
| S3 rationality tests | Logistic Regression (LASSO) + Random Forest | L4 + L5 + L8 | `Data Science Analysis.ipynb` §S3; anchor check cross-referenced from `Default RQ Analysis.ipynb` |
| R1 regulatory shifts | S1 pipeline per variant; descriptive comparison + IQR outlier check | L3 + L5 + L7 + L2 | `Thesis Experiments Analysis.ipynb` (cross-config) + `Sweep Analysis.ipynb` §5 (per-variant cluster shifts) |
| R2 UDBC pathways | Multinomial Logistic Regression + Random Forest + SMOTE/ADASYN; transition heatmaps | L4 + L5 + L7 + L2 | `Data Science Analysis.ipynb` §R2 (UDBC columns produced by `scripts/train.py`) |
| C1 convergence | Rolling-mean plateau + PELT change-point | L2 + Specialized | `Data Science Analysis.ipynb` §C1 |
| C2 reproducibility | Cross-seed mean ± std, fan charts | L2 | `Data Science Analysis.ipynb` §C2; cross-checked in `Default RQ Analysis.ipynb` |
| Cross-cutting | Standardisation, train/val/test, k-fold CV, F1 / AUC-ROC / R² / RMSE | L2 | helpers in `src/utils/` (e.g. `quality_metric.py`); features built in-notebook |

Specialized method used: PELT change-point detection. Justified in Methodology.

---

## 6. On specialized methods

We default to curriculum methods because that's what the rubric grades us on and what we can defend in oral. Specialized methods are not banned — they come in where (a) they answer a specific question better than what we know, and (b) we can explain them clearly.

Currently the only specialized method we plan to use is PELT for convergence detection (C1), and even then we pair it with a curriculum-native plateau check. If during the analysis we find a question where, say, a survival model genuinely fits better than a binary-classification reframe, we'll add it — but each addition needs an explicit defense in Methodology, not a smuggled-in citation.

This is a softer stance than v2's "what we are NOT doing" list. Keeping the door open is better than over-prescribing.

---

## 7. Multi-method robustness on key claims

For the headline claims, supported by ≥2 methods:

| Claim | Method 1 | Method 2 |
|---|---|---|
| "Strategies cluster into K archetypes" | K-means + Silhouette/Elbow | Hierarchical with dendrogram |
| "Removing ESG weight collapses strategies" | Cluster count comparison | PCA spread comparison |
| "Carbon price drives green investment" | Logistic regression coefficient | RF partial dependence |
| "Variants produce distinguishable markets" | Boxplot comparison + IQR | Strategy cluster shifts (S1 on each variant) |
| "Compliance pathway depends on archetype" | Multinomial logit | RF feature importance |
| "Agents converge by episode N" | PELT change-point | Rolling-mean plateau |

Goes in Ch. 5.

---

## 8. Priority

**Tier 1 — required for thesis quality:** S1, S2, R1, R2, C1, C2. Six analyses, every one tied to a specific claim above. This is the floor.

**Tier 2 — strongly preferred, ship if Tier 1 is clean:** S3 (rationality tests + anchor check). Adds the theory-coherence story that the CBS rubric explicitly rewards.

**Tier 3 — only if there's spare time at the end:** Anything else. Pathology classification, autoencoder anomaly detection, network analysis of secondary trades — all have been considered and dropped from the active plan. If a sweep run produces something genuinely surprising, we revisit.

Tier 1 + Tier 2 = 7 analyses, every one producing one or two thesis figures and a paragraph that maps to a specific claim. That's enough.

---

## 9. Sequencing

1. **Now (before final runs finish):** prototype C1 + C2 on the default seeds we already have. Establishes the converged-window cutoff for everything else and gets the credibility scaffolding in place.
2. **Default seeds finalised:** S1 first. The clustering pipeline becomes the reusable engine for S2 and R1.
3. **Reward-weight variant available:** S2. This is the highest-value single analysis in the plan.
4. **Other sweep variants available:** R1, R2, iteratively.
5. **Last 2 weeks:** S3. The rationality tests benefit from settled converged data.

---

## 10. Where it all lands in the thesis

- **Ch. 5 Methodology §5.x Data analysis methods:** §5 (curriculum mapping) + §6 (specialized methods) + §7 (multi-method robustness). One clean methodology section that demonstrably ticks the CBS rubric.
- **Ch. 7 Results:**
  - 7.1 Convergence and reproducibility (C1 + C2)
  - 7.2 Strategic archetypes and how objectives shape them (S1 + S2)
  - 7.3 Rationality of emergent strategies (S3)
  - 7.4 Sensitivity to regulation (R1 + R2)
- **Ch. 8 Discussion:** weaves the seven claims into the larger argument, drawing on the multi-method robustness table for any non-trivial claim.

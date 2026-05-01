# Data Science Analysis Plan — ETS MARL Thesis (v2)

Internal working doc for Daniel & Alessio.
Purpose: pin down DS analyses we run on simulation output, **anchored explicitly in the CBS Machine Learning curriculum** so nothing in Ch. 7 looks pulled from nowhere.

**v2 changes from v1:** every analysis now maps to a specific lecture in our ML class. Specialized methods (change-point detection) are explicitly flagged and justified rather than smuggled in. Inferential statistics that we never covered (Cox PH, Granger, GAMs, cluster-robust SEs, ICC, bootstrap CIs as the *primary* tool) have been replaced with curriculum-native equivalents.

---

## 0. Why this plan exists

CBS's master-thesis learning objectives explicitly require us to *"apply relevant Data Science methods"* and to *"document the analysis through the selection and processing of data sources"*. The presentation deck is more pointed: **"Don't go for the simple methods and mediocre analysis. Try to use different methods to do a robust analysis."**

The MARL system itself is a Data Science method — it *is* the artifact in our DSR framing — but the assessor will also want to see classical DS analysis applied to the data the simulation produces. This plan is how we deliver that, **using methods we actually learned**, which keeps the argument tight under examiner questions.

We exploit two facts about our setup:

1. The `training_log_*.csv` and `year_log_*.csv` files form a panel dataset: ~270 + ~300 columns, ~10M rows over all seeds. That's a real DS-grade dataset.
2. We have ≥4 default seeds plus several sweep variants, which gives enough comparative leverage for descriptive comparison, supervised classification of variants, and feature importance work.

---

## 1. Two analytic tracks

### Track A — In-training behaviour (transient dynamics)

Treats episode index as the time axis. Asks: *how does the policy/market evolve as agents learn?* Output: convergence diagnostics, training-stability evidence.

Primarily serves **Sub-RQ 1**.

### Track B — Converged behaviour (steady-state)

Restricts to the last X% of episodes (X calibrated by Track A). Asks: *what equilibrium has the market reached, and what does it tell us about the EU ETS?*

Primarily serves **Sub-RQ 2** and **Sub-RQ 3**.

---

## 2. Track A: in-training analyses

### A1. Convergence detection
**Lecture mapping:** Specialized (flagged); validated against Lecture 2 (rolling-mean / plateau check).

**Method:** Run two parallel detectors on episode-level reward trajectories per agent and on `ep_mean_clearing_price`:

1. **Change-point detection (PELT)** via the `ruptures` library — the specialized method.
2. **Plateau check using rolling mean + std** — the curriculum-native sanity check. Define convergence as the first episode where `rolling_mean(±5%)` stays inside a tolerance band for K consecutive windows.

We use both and report the convergence episode where they agree. This dual-method framing is exactly the multi-method robustness the CBS deck asks for.

**Why the specialized method:** PELT is the standard objective tool for this in time-series. It's the one place in the thesis we step outside the syllabus, and we justify it explicitly in Methodology: rolling-mean plateau detection is intuitive but threshold-dependent, while PELT gives a reproducible cutoff. We're transparent that this is beyond the standard curriculum.

**Columns:** `reward_A{i}`, `quality_score`, `ep_mean_clearing_price`, `actor_loss_A{i}`.

**Output:** convergence episode per (seed, agent), median + range across seeds. Two figures, one table.

**Maps to:** Sub-RQ 1; provides the cutoff used everywhere downstream.

---

### A2. Cross-seed reproducibility
**Lecture mapping:** Lecture 2 (descriptive statistics, EDA, train/test split philosophy applied to seeds-as-runs).

**Method:** For each headline metric, compute mean and std across seeds within episode bins. Visualize as fan charts (median line + ± 1 std shaded band). For the converged window, report `mean(metric) ± std(metric)` across seeds for ~6 metrics.

We do **not** introduce inferential statistics we didn't cover (no ICC, no bootstrap CI as the primary tool). We frame this as **descriptive cross-run stability** — exactly how a Lecture 2 EDA-style analysis would treat repeated runs.

**Why this matters for CBS:** Directly addresses the rubric *"discuss the quality of the analyzed sources or solutions, including their suitability and validity"*. If between-seed std is large relative to between-variant differences in B3, our findings don't generalize and we say so.

**Columns:** `clearing_price_last`, `quality_score`, `ep_mean_clearing_price`, mean `green_frac_A{i}`, mean `penalty_A{i}`, `secondary_volume`.

**Output:** Fan-chart figure (one per metric), seed-stability summary table.

**Maps to:** Sub-RQ 1; the validity gate for every other Track-B claim.

---

### A3. Pathological-episode detection (anomaly detection)
**Lecture mapping:** Lecture 7 (Isolation Forest, LOF, IQR), Lecture 12 (Autoencoders for anomaly detection), Lecture 5 (PCA for visualization).

**Method:** Treat each episode as a point in a feature space defined by per-episode aggregates (clearing price, mean reward, mean penalty, mean green frac, warning counts, default counts). Detect anomalous episodes with three complementary methods:

1. **IQR rule** on each feature individually — the simple baseline (Lecture 7).
2. **Isolation Forest** for multivariate anomaly detection (Lecture 7).
3. **Undercomplete autoencoder** trained on the bulk of episodes; reconstruction error flags anomalies (Lecture 12).

Visualize the anomalies with **PCA** projected to 2D (Lecture 5) so the reader can see the structure.

**Why three methods:** This is the textbook multi-method comparison that Lecture 7 actually motivated — different outlier detectors flag different things. The autoencoder is a slightly more advanced choice that uses Lecture 12 directly and gives us a deep-learning method without overreaching.

**Practical note on imbalance:** If we want to *classify* "did this episode trigger a pathological warning?" rather than just detect outliers, the warning class is rare. We apply **SMOTE or ADASYN** (Lecture 7) before training a Random Forest classifier (Lecture 5) to predict warning triggers from earlier-episode features. Performance: F1, AUC-ROC, confusion matrix (Lecture 2). Feature importance from Random Forest (Lecture 5) tells us which early signals predict pathological convergence.

**Columns:** `warn_*`, `streak_*_A{i}` (target candidates), all per-agent episode aggregates as features.

**Maps to:** Sub-RQ 1.

---

## 3. Track B: converged-behaviour analyses

All Track B analyses use the converged window from A1, pooled across seeds.

### B1. Strategy clustering (agent taxonomy)
**Lecture mapping:** Lecture 3 (K-means, Hierarchical, Silhouette/Elbow), Lecture 5 (PCA for visualization).

**Method:** Per-agent feature vectors aggregated over the converged window. Cluster with two complementary algorithms from Lecture 3:

1. **K-means** with optimal k selected via **Silhouette Score and Elbow Method** in tandem (the comparison the lecture explicitly calls for).
2. **Hierarchical clustering** with dendrogram and lifetime analysis (Lecture 3 again — the dendrogram lets us visually identify natural cluster cuts).

**DBSCAN** as a third pass if k-means and hierarchical disagree, since DBSCAN handles arbitrary shapes (Lecture 3).

Visualize cluster structure in 2D with **PCA** (Lecture 5). Compare emergent clusters against the *designed* archetypes (coal-heavy, balanced, green-heavy) — does each policy converge into its archetype's expected role?

**Standardize features first** (Lecture 2 — explicitly called out as required before clustering).

**Feature vector per (seed, agent):** mean `bid_price_A{i}`, mean `avg_bid_mult_A{i}`, mean `avg_sec_qty_A{i}`, `inv_onshore_share_A{i}`, `inv_offshore_share_A{i}`, `inv_solar_share_A{i}`, terminal `green_frac_A{i}`, mean of `udbc_*_total_A{i}` shares.

**Why this many methods:** Lecture 3 explicitly compares K-means, hierarchical, and DBSCAN as alternatives. Using all three with proper k-selection (Silhouette + Elbow) is exactly the multi-method robustness pattern from the lectures.

**Maps to:** Sub-RQ 3.

---

### B2. Predictive model of clearing price
**Lecture mapping:** Lecture 4 (Linear Regression), Lecture 5 (Random Forest + feature importance), Lecture 6 (Gradient Boosting / XGBoost / LightGBM), Lecture 2 (cross-validation, R² / RMSE / MAE), Lecture 8 (L1/L2 regularization).

**Method:** Predict year-level `clearing_price` from market state. Three nested models:

1. **Linear Regression** with L2 (Ridge) regularization (Lectures 4 + 8) — interpretable baseline.
2. **Random Forest** (Lecture 5) — handles non-linearity, gives feature importance.
3. **Gradient Boosting** with XGBoost or LightGBM (Lecture 6) — strongest predictive performance.

Standard ML pipeline (Lecture 2): standardize features → 80/10/10 train/val/test split → cross-validation (k=5) for hyperparameter selection → final eval on held-out test. Report R², RMSE, MAE for each model (all Lecture 2). Compare model accuracy: does the GBM beat linear by a margin that justifies the complexity? — that's the **bias/variance tradeoff** discussion (Lecture 9 + Lecture 2 model performance).

**Feature importance:** From Random Forest (Lecture 5) and from the gradient boosting model (Lecture 6). This is the *curriculum-native* substitute for SHAP — the lectures explicitly cover RF feature importance, and it's interpretable.

**Theory test:** Compute the fundamental anchor offline (we already have the formula) and add it as a feature. If RF feature importance ranks anchor highly, the simulation reproduces auction theory — that's a finding.

**Features:** `tnac`, `cap`, `auction_volume`, `msr_reserve`, lagged mean `bid_price_A*`, mean `bank_start_A*`, year, fundamental anchor.

**Maps to:** Sub-RQ 1 + Sub-RQ 3.

---

### B3. Sensitivity analysis across sweep variants
**Lecture mapping:** Lecture 2 (descriptive comparison + EDA), Lecture 5 (Random Forest), Lecture 6 (Gradient Boosting), Lecture 7 (outlier handling).

**Method:** Two complementary angles.

**B3a — Descriptive comparison.** For each sweep variant (LRF 2.2 / 4.3 / 6.0%, MSR on/off, cap levels, all-financial reward weights), compute mean ± std across seeds for the headline outcomes from A2. Display as a comparison table and as boxplots side-by-side. Boxplots also flag outlier seeds via the IQR rule (Lecture 7) so we don't get fooled by one bad seed.

This is a Lecture-2-style EDA comparison. We do **not** invoke causal inference language we didn't cover. We say "variant X shows higher mean clearing price than the default, with non-overlapping IQR boxes" rather than "treatment effect is statistically significant."

**B3b — Variant classification (creative reframe).** Train a Random Forest (Lecture 5) and Gradient Boosting model (Lecture 6) to **predict the variant label from outcome features**. If the classifier achieves high F1 (Lecture 2), the variants produce distinguishable outcomes — that's our evidence the regulatory mechanism matters. Feature importance tells us *which* outcomes differ most across variants.

**Why this is good:** B3b turns sensitivity analysis into a supervised classification problem, which is exactly Lecture 5/6 territory. It gives us a clean, defensible quantification ("the regulator's choice of LRF is detectable from market outcomes with F1 = X") without inventing statistical machinery we didn't learn.

**Maps to:** Sub-RQ 2 — direct hit. Likely the headline analysis.

---

### B4. Theory-prediction tests (reframed as supervised classification)
**Lecture mapping:** Lecture 4 (Logistic Regression — binary AND multinomial), Lecture 5 (Random Forest), Lecture 2 (F1, AUC-ROC, confusion matrix).

**Method:** We have two theoretical predictions from Ch. 3. We turn each into a binary or multinomial classification problem.

**H1 (LCOE crossover):** *"Agents invest in green when carbon price exceeds ~50 €/t."*
- Binary classification: per (agent, year), label = 1 if `invest_cost_A{i} > threshold`, 0 otherwise.
- Features: lagged `clearing_price`, agent archetype, `bank_start_A{i}`, `year`.
- Models: **Logistic Regression** (Lecture 4) with L1 (LASSO) for interpretable coefficients, plus **Random Forest** for non-linear effect detection.
- If logistic regression's coefficient on `clearing_price` is positive and the RF confirms a step-up around 50 €/t (visible in partial dependence plots, which fall under Lecture 5 RF interpretability), H1 is supported.

**H2 (heterogeneity):** *"High-fossil agents transition to green faster than low-fossil ones."*
- Binary classification: label = 1 if agent invested meaningfully in green by year 6 (mid-episode), 0 otherwise.
- Features: agent archetype (one-hot), initial mix vector, mean clearing price.
- Same model setup.
- Coefficient sign on archetype dummies tests H2 directly.

**Why this is curriculum-native:** Lecture 4 covers binary logistic regression explicitly. We're not inventing survival analysis — we're discretizing the question into a year-6 cutoff and using methods we learned. The cutoff choice is documented as a robustness check (Lecture 2 — cross-validation across cutoff years).

**Performance:** F1, AUC-ROC, confusion matrix (Lecture 2).

**Maps to:** Sub-RQ 3 + theory chapter.

---

### B5. UDBC compliance-pathway analysis
**Lecture mapping:** Lecture 4 (Multinomial Logistic Regression — explicitly covered), Lecture 5 (Random Forest), Lecture 2 (descriptive transitions).

**Method:** The `udbc_*` columns classify each (agent, year) into {U, D, M, B, C}. Two analyses.

**B5a — Multinomial logistic regression.** Predict the UDBC class from agent and market features (`bank_start_A{i}`, `cap`, `clearing_price`, agent archetype, year). Lecture 4 explicitly covers multinomial logistic for unordered multi-class problems. Coefficients tell us which features push toward each compliance failure mode. Compare against **Random Forest** as the non-linear alternative for robustness, with feature importance.

**B5b — Descriptive transition tables.** Per variant, count (UDBC class at year t) → (UDBC class at year t+1) transitions and display as heatmaps. This is descriptive — no Markov-chain formalism, just frequency tables, fully Lecture-2 EDA territory. The interesting question is whether high-LRF variants show more U → recovery transitions vs default.

**Class imbalance:** UDBC classes are imbalanced (most agent-years are compliant). Apply **SMOTE or ADASYN** (Lecture 7) before training the multinomial logit / RF if needed. We compare results with and without resampling — Lecture 7 makes a point of warning about ambiguous synthetic samples, so showing both is honest.

**Why this is good:** UDBC is a categorical labelling system already in our logs. Multinomial logit is in Lecture 4. RF is in Lecture 5. SMOTE is in Lecture 7. Three distinct curriculum touchpoints.

**Maps to:** Sub-RQ 2 + Sub-RQ 3.

---

## 4. Curriculum-mapping table

For Methodology Ch. 5, this is the table we put in the methods section to demonstrate every analytic choice traces to a course we took.

| Analysis | Primary method | Lecture |
|---|---|---|
| A1 convergence | PELT change-point + rolling-mean plateau | Specialized + L2 |
| A2 reproducibility | mean ± std across seeds, fan charts | L2 |
| A3 pathology | IQR + Isolation Forest + Autoencoder; SMOTE if classifying | L7 + L12 + L7 |
| B1 clustering | K-means + Hierarchical + Silhouette/Elbow + PCA viz | L3 + L5 |
| B2 price prediction | Linear (Ridge) + RF + Gradient Boosting + R²/RMSE/MAE | L4 + L5 + L6 + L8 + L2 |
| B3 sensitivity | Descriptive comparison + RF/GBM variant classifier | L2 + L5 + L6 |
| B4 theory tests | Binary Logistic Regression (LASSO) + RF | L4 + L5 + L8 |
| B5 UDBC | Multinomial Logistic Regression + RF + SMOTE/ADASYN | L4 + L5 + L7 |
| Cross-cutting | Cross-validation, F1, AUC-ROC, confusion matrix, R²/RMSE | L2 |

Specialized (non-curriculum) methods used: **PELT change-point detection only.** Justified explicitly in Methodology with reference to its standard role in time-series segmentation.

---

## 5. Cross-cutting methodological notes

### 5.1 Validation philosophy

Standard ML pipeline from Lecture 2 applied throughout: **standardize → train/val/test split → cross-validation for hyperparameters → final test set used once.** Where the unit of analysis is the seed (A2, B3a), we report mean ± std across seeds and avoid pretending we have inferential power we don't. Where it's per (agent, year) row (B2, B4, B5), proper k-fold CV with the seed as a grouping variable to prevent leakage.

### 5.2 Multi-method robustness

For each headline claim, support with at least two methods from the curriculum.

| Claim | Method 1 | Method 2 |
|---|---|---|
| "Agents converge by episode N" | PELT change-point (specialized) | Rolling-mean plateau (L2) |
| "Strategies cluster into K archetypes" | K-means + Silhouette/Elbow (L3) | Hierarchical clustering with dendrogram (L3) |
| "Carbon price drives green investment" | Logistic Regression coefficient sign (L4) | Random Forest + feature importance (L5) |
| "Regulatory variants produce distinguishable markets" | Descriptive boxplot comparison (L2) | RF/GBM variant classifier F1 (L5/L6) |
| "Agent type predicts compliance pathway" | Multinomial logit (L4) | Random Forest feature importance (L5) |
| "Fundamental anchor predicts clearing price" | Linear regression coefficient (L4) | RF feature importance ranking (L5) |

Goes in Methodology as a concrete table.

### 5.3 Tooling

Standard scientific-Python stack — every package corresponds to methods from the curriculum:

- `pandas`, `numpy`, `matplotlib`, `seaborn` (L2 EDA)
- `scikit-learn` for K-means, hierarchical, DBSCAN, PCA, LDA, Linear/Logistic Regression, Decision Tree, Random Forest, Isolation Forest, LOF, train/test split, cross-validation, all metrics (L2–L7)
- `xgboost` and `lightgbm` (L6)
- `imbalanced-learn` for SMOTE/ADASYN (L7)
- `tensorflow.keras` or `pytorch` for the autoencoder (L12)
- `ruptures` for PELT change-point (specialized; one place in the thesis)

Notebooks under `notebooks/`. Each Track B subsection gets its own notebook to keep them reviewable.

### 5.4 What we are NOT doing

Stating explicitly to avoid scope creep:

- **No exotic statistics we didn't cover** — no Cox PH, no Granger causality, no GAMs, no cluster-robust SEs, no ICC, no permutation tests, no Markov-chain formalism. Where the underlying question is interesting, we reframe it into a supervised classification or descriptive comparison.
- **No deep learning beyond MARL itself + the autoencoder.** No CNN, no LSTM for prediction (we have it as a curriculum tool but no good use case here), no GAN.
- **No NLP / text analysis.** No text data.
- **No network analysis.** No clean curriculum hook.
- **No real-EU-ETS empirical validation** beyond the qualitative price-range check already in Ch. 6.

Each absence is named in §5.1 of Methodology as a deliberate scope decision.

---

## 6. Mapping to Research Questions

| RQ | Primary | Supporting |
|---|---|---|
| **Sub-RQ 1** (env. & algorithm) | A1, A2, A3 | B2 (price-anchor recovery) |
| **Sub-RQ 2** (regulatory sensitivity) | B3 | B5 |
| **Sub-RQ 3** (strategies & aggregation) | B1, B4 | B5 |

---

## 7. Priority tiers

If time is tight, drop bottom-up.

**Tier 1 — must ship (these alone clear the CBS bar):**
A1, A2, B1, B3, B5. Five analyses across nine lectures' worth of methods (L2, L3, L4, L5, L6, L7, L12 + specialized).

**Tier 2 — strongly preferred:**
B2 (predictive model), B4 (theory tests), A3 (pathology / anomaly).

**Tier 3 — stretch:**
DBSCAN as third clustering pass in B1, autoencoder in A3 if Isolation Forest already gives clean results, SVM (L6) as additional classifier in B3b.

Tier 1 + Tier 2 = 8 analyses spanning lectures 2–7 plus 12 plus the one specialized method. That's the full breadth of the ML curriculum applied, which is exactly what the rubric asks for.

---

## 8. Sequencing

1. **Now (before final runs finish):** prototype A1 + A2 on existing seeds. Establishes converged-window cutoff.
2. **Default seeds finalised:** B1, B5 first.
3. **First sweep variant in:** B3 on that variant. Iterate per variant.
4. **Last 2 weeks:** B2, B4. Highest-effort, highest-payoff.
5. **Final week:** Tier 3 only if time permits.

---

## 9. Ch. 7 / Ch. 5 mapping

Ch. 5 (Methodology) §5.x: Data Analysis Methods. Direct lift of §4 (curriculum-mapping table) plus §5 here. This is the section that demonstrably ticks the CBS box on "applied Data Science methods".

Ch. 7 (Results):
- 7.1 Training dynamics → Track A
- 7.2 Converged market behaviour → B1, B2, B5
- 7.3 Strategic behaviour and theory tests → B4
- 7.4 Sensitivity to regulation → B3

Ch. 8 (Discussion) leans on §5.2 multi-method robustness table when claiming any non-trivial finding.

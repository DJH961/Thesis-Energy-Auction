# Statistics Appendix: v7.11.0 Run (Seed 42, 70k Episodes)

## Unit of Analysis

- **Episode-level metrics** (training_log): N = 70,000 episodes, each summarizing 12 years of simulation
- **Year-level metrics** (year_log): N = 840,000 rows (70,000 episodes x 12 years)
- **Single seed**: seed 42 only. No multi-seed replication.

**Statistical limitation**: With a single seed, all reported statistics describe the distribution of outcomes within one training trajectory. No between-seed variance is available. Inferential tests below compare windows within the same trajectory (non-independent samples), providing evidence of within-run trends only.

---

## Descriptive Statistics

### Clearing Price by Training Phase

| Phase | N | Mean | Std | Median | Min | Max | %_at_floor |
|-------|---|------|-----|--------|-----|-----|------------|
| Early (0-5k) | 5,000 | 78.15 | 71.83 | 45.00 | 45.00 | 494.60 | 65.3% |
| Mid (20k-30k) | 10,000 | 47.49 | 18.52 | 45.00 | 45.00 | 413.98 | 95.2% |
| Late (60k-70k) | 10,000 | 45.15 | 4.45 | 45.00 | 45.00 | 322.39 | 99.8% |
| Last 5k | 5,000 | 45.11 | 4.26 | 45.00 | 45.00 | 322.39 | 99.8% |

### Per-Agent Bid Price (Last 5,000 Episodes)

| Agent | Mean | Std | Median | P5 | P95 |
|-------|------|-----|--------|-----|-----|
| A1 | 53.74 | 15.28 | 45.00 | 45.0 | 84.7 |
| A2 | 53.28 | 15.11 | 45.00 | 45.0 | 87.3 |
| A3 | 54.04 | 16.98 | 45.00 | 45.0 | 93.0 |
| A4 | 54.42 | 16.86 | 45.00 | 45.0 | 92.8 |
| A5 | 54.23 | 13.88 | 47.92 | 45.0 | 82.5 |
| A6 | 56.96 | 19.51 | 47.61 | 45.0 | 100.7 |
| A7 | 329.54 | 48.11 | 327.94 | 253.2 | 411.5 |
| A8 | 81.35 | 23.90 | 77.93 | 48.1 | 126.1 |

Note: A7 is a clear outlier with mean bid 6x higher than all others. A8 also elevated. A1-A6 median at or near floor.

### Per-Agent Green Fraction (End-of-Episode, Last 5,000 Episodes)

| Agent | Mean | Std | Initial |
|-------|------|-----|---------|
| A1 | 0.328 | 0.044 | 0.22 |
| A2 | 0.375 | 0.056 | 0.23 |
| A3 | 0.602 | 0.057 | 0.43 |
| A4 | 0.600 | 0.057 | 0.43 |
| A5 | 0.980 | 0.036 | 0.73 |
| A6 | 0.981 | 0.035 | 0.73 |
| A7 | 1.000 | 0.001 | 0.93 |
| A8 | 1.000 | 0.000 | 0.93 |

### Per-Agent Reward (Last 5,000 Episodes)

| Agent | Mean Total | Std | Mean Base | Mean Penalty (M EUR) |
|-------|-----------|-----|-----------|---------------------|
| A1 | -6.558 | 1.110 | -6.558 | 2,025 |
| A2 | -1.538 | 1.431 | -1.538 | 307 |
| A3 | -3.818 | 0.829 | -3.818 | 1,225 |
| A4 | -0.158 | 1.337 | -0.158 | 1,010 |
| A5 | -0.901 | 0.563 | -0.901 | 153 |
| A6 | +3.644 | 1.579 | +3.644 | 425 |
| A7 | -0.103 | 0.218 | -0.103 | 36 |
| A8 | +5.560 | 0.932 | +5.560 | 69 |

Note: Shaping component = 0 for all agents in the last 5k (shaping decayed by episode ~42,000).

### Diagnostic Scores (Last 5,000 Episodes)

| Agent | S_financial | S_green | S_composite |
|-------|------------|---------|-------------|
| A1 | 0.576 | 0.131 | 0.876 |
| A2 | 0.404 | 0.166 | 0.585 |
| A3 | 0.701 | 0.262 | 1.001 |
| A4 | 0.677 | 0.260 | 0.769 |
| A5 | 0.852 | 0.543 | 1.152 |
| A6 | 0.792 | 0.542 | 0.967 |
| A7 | 0.972 | 0.566 | 1.272 |
| A8 | 0.971 | 0.566 | 1.069 |

---

## Inferential Tests

### Test 1: Clearing Price Trend (Phase Comparison)

**Question**: Did the clearing price change significantly between training phases?

**Method**: Mann-Whitney U test (non-parametric, no normality assumption; samples within same trajectory so not truly independent -- interpret as descriptive of within-run shift).

| Comparison | U statistic | p-value | Delta_mean (EUR) | Direction |
|------------|------------|---------|-----------------|-----------|
| Early (0-5k) vs Mid (20k-30k) | 32,647,194 | < 1e-300 | -30.66 | Decrease |
| Mid (20k-30k) vs Late (60k-70k) | 52,354,005 | 2.50e-97 | -2.34 | Decrease |

**Interpretation**: The clearing price declined significantly throughout training. The largest drop occurred early (early-to-mid: -30.66 EUR), corresponding to exploration decay. The mid-to-late drop (-2.34 EUR) is statistically significant but practically negligible -- the price was already pinned to the floor.

**Effect size**: The clearing price converged to within 0.3% of the floor (45.15 vs 45.00 EUR) in the late phase. Practical significance is nil -- the "improvement" is from exactly-at-floor to very-slightly-above-floor.

### Test 2: Green Fraction Stasis (Brown Agents)

**Question**: Did brown agents (A1-A4) improve their green fraction between mid and late training?

**Method**: Welch's t-test on episode-level green_frac values (unequal variance assumed).

| Agent | Mid Mean | Late Mean | t-stat | p-value | Cohen's d |
|-------|---------|-----------|--------|---------|-----------|
| A1 | 0.332 | 0.328 | 4.41 | 1.03e-5 | 0.09 |
| A2 | 0.378 | 0.375 | 2.88 | 0.004 | 0.06 |
| A3 | 0.605 | 0.602 | 2.54 | 0.011 | 0.05 |
| A4 | 0.603 | 0.600 | 2.47 | 0.013 | 0.05 |

**Interpretation**: Green fractions for brown agents are statistically significantly *lower* in the late phase than the mid phase (tiny negative trend). Cohen's d < 0.10 for all: negligible effect size. The brown agents did not learn to invest in green technology. The slight decreases may reflect stochastic drift or reduced exploration.

---

## Bid Strategy Analysis (Year-Level, Last 5,000 Episodes)

### Bid-to-Reserve Ratio by Agent

| Agent | Mean | Std |
|-------|------|-----|
| A1 | 1.16 | 0.99 |
| A2 | 1.14 | 0.92 |
| A3 | 1.15 | 0.96 |
| A4 | 1.16 | 0.97 |
| A5 | 1.47 | 1.79 |
| A6 | 1.34 | 1.44 |
| A7 | 6.43 | 4.59 |
| A8 | 2.13 | 2.38 |

A1-A4 bid near the reserve. A7 bids 6.4x reserve. High std across all agents reflects year-to-year variation within episodes.

### Floor Streak Statistics (Last 5,000 Episodes)

| Agent | Mean Streak | Max Streak |
|-------|------------|------------|
| A1 | 1.3 | 13 |
| A2 | 1.4 | 13 |
| A3 | 1.4 | 13 |
| A4 | 1.3 | 13 |
| A5 | 0.7 | 8 |
| A6 | 0.8 | 8 |
| A7 | 0.0 | 0 |
| A8 | 0.0 | 2 |

A1-A4 are stuck at floor for ~1.3 consecutive years on average, with max streaks of 13 years (i.e., entire episodes). A7 never hits floor.

### Secondary Market Participation

| Agent | Buy % | Sell % | Avg Volume (Mt) |
|-------|-------|--------|-----------------|
| A1 | 4.2% | 95.8% | 0.043 |
| A2 | 97.6% | 2.4% | 0.769 |
| A3 | 4.9% | 95.1% | 0.055 |
| A4 | 97.4% | 2.6% | 0.066 |
| A5 | 2.6% | 97.4% | 0.384 |
| A6 | 2.5% | 97.5% | 0.048 |
| A7 | 3.1% | 96.9% | 0.151 |
| A8 | 2.5% | 97.5% | 0.116 |

A2 and A4 are dominant buyers; A2 trades 0.769 Mt/yr (the largest volume). All others are net sellers.

---

## Market Scarcity (Year-Level, Last 5,000 Episodes)

| Year | Emissions (Mt) | Cap (Mt) | Scarcity Ratio | Clearing Price (EUR) |
|------|---------------|----------|----------------|---------------------|
| 0 | 21.07 | 20.25 | 1.040 | 45.00 |
| 3 | 17.54 | 17.60 | 0.996 | 45.00 |
| 6 | 15.71 | 14.93 | 1.052 | 45.00 |
| 9 | 14.27 | 12.25 | 1.164 | 45.06 |
| 11 | 13.48 | 10.47 | 1.287 | 45.11 |

The market is over-subscribed in 10 of 12 years. Year 3 is approximately balanced (0.996x). Scarcity increases steadily to 1.29x by year 11. Yet the clearing price remains indistinguishable from the floor.

---

## Explicit Blockers and Limitations

1. **Single seed**: All analysis from seed 42 only. Effect sizes and convergence patterns may differ under other seeds.
2. **Non-independent samples**: Phase comparison tests (Mann-Whitney U) compare windows from the same training trajectory. Observations within each window are serially correlated. P-values should be interpreted as descriptive, not as independent-sample hypothesis tests.
3. **No baseline comparison**: Without a parallel run (e.g., with heuristic bots, different pricing rule, or structural changes), we cannot isolate which design choices caused the failure vs. which merely failed to prevent it.
4. **Shaping already decayed**: The green shaping reward decayed to zero by ~episode 42k. Last-5k statistics reflect base reward only; any earlier shaping effects are not captured.
5. **WTP columns contain mixed types**: The `wtp_binding` column contains string values ("budget", "economic") mixed with numeric entries. These were handled by coercion but some per-agent WTP analysis may be imprecise.

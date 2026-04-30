# ESG Reward — Design, Calibration, and Stress Scenarios

This document describes the ESG component of the per-agent reward signal:
why it has the structure it does, how it is balanced against the financial
channel, and how it behaves under a range of stress scenarios.

It complements `docs/design.md` §7 (Reward Design); the formula and config
knobs themselves live in `src/environment/ets_environment.py`
(`_compute_rewards`) and `configs/default.yaml` (`esg:` block).

---

## 1. Formula

For agent $i$ in year $t$:

$$
\text{esg\_raw}_t = \text{scale} \cdot
\Big(
  w_{\text{stock}} \cdot \text{ef\_ratio}_t
  + w_{\text{flow}} \cdot \text{ef\_ratio}_t \cdot \frac{\text{anchor\_real}_t}{\text{anchor\_real}_0}
  + \text{speed\_coef} \cdot \max(0, \Delta\text{green}_t)
\Big)
$$

$$
\text{esg\_signal}_t =
\begin{cases}
\text{esg\_raw}_t \cdot \text{compliance\_gate}_t & \text{if } \text{esg\_raw}_t \ge 0 \\
\text{esg\_raw}_t & \text{otherwise (defensive branch)}
\end{cases}
$$

with

- $\text{ef\_ratio}_t = \max\!\left(0,\; \dfrac{ef_0 - ef_t}{ef_0}\right)$ — cumulative
  emission-factor improvement vs. the agent's year-0 mix.
- $\text{anchor\_real}_t = \text{compute\_fundamental\_anchor}(t) / \text{infl}(t)$
  — fundamental anchor in real terms; it encodes the cap trajectory and
  inflation, and is the same anchor used by the budget gate, the
  bid-head reward floor, and the quality metric volatility component.
- $\Delta\text{green}_t = \max(0,\; \text{green\_frac}_t - \text{green\_frac}_{t-1})$
  — current-year transition motion.
- $\text{compliance\_gate}_t$ — smooth blend approaching linear
  `coverage_frac` above `compliance_gate_blend_threshold` (0.90 default)
  and softening below; only attenuates non-negative `esg_raw`.

### What each term does

- **Stock term** (`stock_weight × ef_ratio`) — pays every year the agent
  maintains a high green share. A company that decarbonised early keeps
  earning ESG every subsequent year; this prevents a "front-load and
  forget" pattern where investments would only pay through `Δgreen`.
- **Flow term** (`flow_weight × ef_ratio × anchor_real_t / anchor_real_0`)
  — algebraically equal to *(saved Mt this year × anchor_real_t)*
  divided by *(initial baseline emissions × anchor_real_0)*. It values
  avoided carbon at the **live social shadow price**, so a green agent
  earns more in scarcity-heavy years than in early years. This matches
  the social-value perspective: the same Mt of saved CO₂ is worth more
  when the EU-wide cap binds harder.
- **Speed bonus** (`speed_coef × max(0, Δgreen)`) — small bonus for
  actual motion this year. `speed_coef` is uniform across the episode
  (default `speed_coef = speed_coef_late = 0.3`); we do not front-load
  the speed bonus.
- **Compliance gate** — ESG rewards never bypass compliance: an
  under-covered agent has its positive ESG attenuated. The gate is
  asymmetric (only positive ESG is gated) so future subtractive variants
  can still send a negative signal to a poorly-greened agent.

### What was removed (and why)

An earlier formulation centred ESG on a linear horizon penalty
$\text{ef\_baseline} = t / (n_\text{years} - 1)$. That term made late
investment net-negative — a year-11 agent at $\text{ef\_ratio} = 0.5$
received $\text{ef\_centered} = 0.5 - 11/11 = -0.5$, equivalent to
"the world ends at year 12 so investment after year 6 is bad". Late
investment is bad because of TVM, not because there's no future. The
saved-carbon hybrid preserves TVM organically (more years of stock+flow
accrual for earlier investment) without an artificial horizon penalty.

---

## 2. Balance with the financial channel

Per-agent reward composition (simplified):

```
base = w_cost × (-cost_norm_centered) + w_green × esg_signal
        \________________________/      \____________________/
              FINANCIAL                          ESG
```

For balanced agents (`w_cost = w_green = 0.5`, the A2/A4/A6/A8
archetypes), the goal is a roughly even contribution from each channel
across an episode. The ESG channel is positive-only and the financial
channel is centred on the fair-price baseline, so `esg.scale` is the
single knob used to calibrate magnitudes.

### Empirical sweep

Deterministic compliant rollout (8 learning agents, 0 bots, all bid
80 EUR/t × 1.0×need, green agents invest 5 %/yr in solar; tail-window
sum across the last 6 years):

| `esg.scale` | Σ \|ESG\| | Σ \|FIN\| | %ESG | Comment |
|---|---|---|---|---|
| 0.15 | 1.42 | 2.25 | 39 % | financial-heavy |
| 0.20 | 1.89 | 2.25 | 46 % | slight financial bias |
| **0.25** | **2.37** | **2.25** | **51 %** | **calibrated, ≈50/50** |
| 0.30 | 2.84 | 2.25 | 56 % | slight ESG bias |
| 0.50 | 4.74 | 2.25 | 68 % | ESG-dominated |
| 1.00 | 9.47 | 2.25 | 81 % | ESG-dominated |

`scale = 0.25` is the default. A regression test
(`tests/test_rewards.py::test_esg_balance_with_financial_50_50`) pins
the balance to `[35 %, 65 %]` so future changes to either channel will
trip the test. **Re-tune `scale` (not `stock_weight` / `flow_weight`)
if the balance drifts.**

### Per-year trajectory at `scale = 0.25`

```
Year   ESG/yr   FIN/yr   %ESG   ef_ratio   anchor_ratio
 0     +0.00    +0.07    0%     0.44       1.00
 1     +0.00    +0.30    0%     0.44       1.02
 2     +0.00    +0.27    0%     0.44       1.07
 3     +0.01    +0.12    7%     0.44       1.09     ← compliance gate kicks in
 4     +0.01    +0.01   62%     0.44       1.11
 5     +0.02    +0.02   54%     0.44       1.18
 6     +0.02    +0.05   33%     0.44       1.24     ← typical mid-game
 7     +0.03    +0.22   12%     0.44       1.28
 8     +0.02    +0.31    7%     0.44       1.33
 9     +0.01    +0.41    3%     0.44       1.37     ← scarcity-cost peak
10     +0.05    -0.11   30%     0.44       1.44
11     +0.03    -0.08   29%     0.44       1.47
                              tail-window: 51 % ESG
```

The shape is intentional: compliance dominates the early signal (ESG
gated to ≈0 when coverage_frac < 0.9), ESG meaningful in the middle,
financial costs of cap-tightening dominate near the peak (years 7–9),
then both balance again in the final two years.

---

## 3. Strengths

1. **Late investment rewarded by TVM, not horizon penalty.** A year-11
   agent at `ef_ratio = 0.5` receives $+0.39$ at the default scale.
   Earlier investment is still preferred because more years of stock+flow
   accrual remain — TVM is preserved organically.
2. **Sustained green share earns reward every year.** Stock term ≠ 0
   for any positive `ef_ratio`. Stops the front-load-then-stop pattern.
3. **Saved carbon is monetised at the live anchor.** Flow term =
   `(saved_Mt × anchor_real_t) / (initial_baseline_emiss × anchor_real_0)`
   — the social shadow price of avoided CO₂. A real ESG-mandated CFO
   would value avoided emissions at exactly this benchmark.
4. **Compliance always comes first.** The compliance gate
   `coverage_frac^(1+blend)` zeroes ESG when the agent is under-covered.
5. **Anchor-driven, not exogenous.** The `anchor_real_t / anchor_real_0`
   ratio is endogenous to the cap trajectory; no manual schedule needed.
6. **Speed bonus is uniform across the episode** — no artificial
   front-loading. Late investment is bad only because of TVM.

---

## 4. Weaknesses & known limitations

1. **Compliance gate creates a ~89 % → 91 % discontinuity** in
   coverage. An agent at 89 % coverage gets ESG strongly attenuated; at
   91 % it's near-full. Mitigated by `gate_blend_threshold=0.90` and
   `gate_blend_width=0.30` smoothing, but a sharp gradient remains
   around the threshold. *Acceptable* — this is the desired
   "compliance first" behaviour.
2. **No subtractive baseline for poor performance.** ESG ≥ 0 always.
   A coal-heavy agent (`ef_ratio = 0`) gets ESG = 0, not negative. By
   design — the financial channel already independently penalises poor
   green outcomes via the carbon-priced compliance cost.
3. **Anchor ratio is bounded by the cap trajectory.** Over the EU-ETS
   12-year horizon the anchor ratio reaches ~1.47 by year 11 (real
   terms), so the flow term contribution is bounded ~1.5× the stock
   term. *Acceptable* — matches reality (carbon has a bounded social
   value over a 12-year window).

---

## 5. Stress scenarios

| Scenario | ESG behaviour | Verdict |
|---|---|---|
| **Always coal** (`ef_ratio = 0`) | ESG = 0 every year. Financial channel still pays compliance cost. | ✓ correct — penalty is on the financial side |
| **Always green** (`ef_ratio = 1`) | Peak ≈ `0.25 × (1 + 1.5 × 1.47 + 0) = 0.80` per year; cumulative ≈ 3.5 over 12 years. | ✓ upper bound of green reward in steady state |
| **Late-starter** (invests entirely in year 6) | Years 0–6 ESG = 0; years 7–11 `ef_ratio` rises to ~0.7. Cumulative ESG ≈ 1.5. **Better than NEVER** (cumulative 0). | ✓ pre-redesign would have given negative cumulative reward, pushing the policy toward "never invest" |
| **Front-loader** (invests entirely in year 0) | `ef_ratio = 0.7` from year 0 onwards; stock term contributes every year, flow term ramps with anchor. Cumulative ESG ≈ 5.5. | ✓ TVM-correct preference for early investment preserved |
| **Deceptive bidder** (high bid_p but only 90 % coverage) | Compliance gate cuts ESG to ~70 % of nominal. Financial channel still pays the compliance cost. | ✓ gaming defeated |
| **Compliance threshold cliff** (89 % → 91 % coverage) | Sharp ~30 %-of-ESG gradient at the threshold. | Acceptable trade-off — softening would dilute the "compliance first" signal |
| **High-inflation regime** | `anchor_real_t / anchor_real_0` is built from real anchors so the flow term is unaffected by inflation. | ✓ inflation-invariant by construction |

---

## 6. Realism

The hybrid mirrors how real ESG-rated utilities are evaluated:

- **Stock component** corresponds to operating-emission-intensity factors
  used by S&P Global / MSCI / Sustainalytics utility ratings — a
  measure of how green the operating capacity is *right now*.
- **Flow component** corresponds to TCFD-style scenario analysis and
  internal carbon pricing: avoided emissions valued at the live carbon
  price benchmark, which is the framework Ørsted, EDF, and Engie use
  for ESG-NPV internal accounting.
- **Anchor as social shadow price**: the EU-ETS fundamental anchor *is*
  the marginal-abatement-cost benchmark used in those accounting
  frameworks. The flow term is therefore directly interpretable as a
  "saved-CO₂ ESG dividend at the social cost of carbon".
- **50/50 split for `[w_cost = 0.5, w_green = 0.5]`** corresponds to a
  CFO whose long-term incentive plan is 50 %-EBITDA-linked and
  50 %-ESG-linked — common at mid-cap European utilities in 2024–2025.

---

## 7. Logged channels

Every year, `_last_reward_channels[i]` records:

| Field | Meaning |
|---|---|
| `esg_signal` | Final ESG term, post-compliance-gate |
| `esg_stock_term` | `ef_ratio` (raw stock factor) |
| `esg_flow_term` | `ef_ratio × anchor_real_t / anchor_real_0` |
| `esg_anchor_ratio` | `anchor_real_t / anchor_real_0` (the live multiplier) |
| `compliance_gate` | The smooth `coverage_frac^(1+blend)` gate value |
| `gate_activation` | `1 + blend` (the gate exponent) |

These flow into `year_log` and from there into the analysis notebooks
(`Full Run & Analysis` §A6 and `Sweep Analysis` §5). When tuning
`esg.scale`, inspect `esg_stock_term` vs. `esg_flow_term` per year to
see which sub-component is dominating; `esg_anchor_ratio` shows how
much "scarcity-leverage" the flow term is currently providing.

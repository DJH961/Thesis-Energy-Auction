# ESG Reward — Design, Calibration, and Stress Scenarios

This document describes the ESG component of the per-agent reward signal:
why it has the structure it does, how it is balanced against the financial
channel, and how it behaves under a range of stress scenarios.

It complements `docs/design.md` §7 (Reward Design); the formula and config
knobs themselves live in `src/environment/ets_environment.py`
(`_compute_rewards`) and `configs/default.yaml` (`esg:` block).

---

## 1. Formula

For agent *i* in year *t*:

```
esg_raw_t = scale × ( w_stock × ef_ratio_t
                    + w_flow  × ef_ratio_t × (anchor_real_t / anchor_real_0)
                    + speed_coef × max(0, Δgreen_t) )

esg_signal_t = esg_raw_t × compliance_gate_t   if esg_raw_t ≥ 0
             = esg_raw_t                       otherwise (defensive branch)
```

with

- `ef_ratio_t = max(0, (ef_0 − ef_t) / ef_0)` — cumulative emission-factor
  improvement vs. the agent's year-0 mix.
- `anchor_real_t = compute_fundamental_anchor(t) / infl(t)` — fundamental
  anchor in real terms; encodes the cap trajectory and inflation, and is
  the same anchor used by the budget gate, the bid-head reward floor,
  and the quality metric volatility component.
- `Δgreen_t = max(0, green_frac_t − green_frac_{t−1})` — current-year
  transition motion.
- `compliance_gate_t` — smooth blend approaching linear `coverage_frac`
  above `compliance_gate_blend_threshold` (0.90 default) and softening
  below; only attenuates non-negative `esg_raw`.

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
across an episode. The financial channel is centred on the fair-price
baseline (`expected_compliance_norm = 1.0`), so a policy that bids at
the fundamental anchor pays cost_norm ≈ 1 and the **signed** financial
reward is ≈ 0 by construction. What drives the policy gradient is the
year-over-year *swing* of the financial channel (positive when the
clearing dips below anchor, negative when it spikes), so the right
calibration target is the **absolute** per-(year, agent) magnitude
on each side. ESG is positive-only and `esg.scale` is the single knob
used to match magnitudes.

### Why the previous "flat 80 EUR/t" sweep was wrong

The first calibration of this PR bid at a flat 80 EUR/t every year,
which is well below the mid-game anchor (~120–150 EUR/t real). That
gave cost_norm < 1.0 throughout the tail and left the financial
channel sitting at a spurious `+2.25` over the last 6 years — a
"buy-cheap windfall" no realistic policy can earn. A trained PPO
policy converges to bidding at or above the anchor (under-bidding
loses the auction → debt cascade), where signed Σ FIN ≈ 0. Re-running
the sweep against an **anchor-tracking** rollout (the steady-state of a
trained policy) shifts the calibration significantly.

### Empirical sweep — anchor-tracking rollout

8 learning agents, 0 bots, all bid the per-year fundamental anchor ×
1.0 × need; green agents invest 5 %/yr in solar; tail-window absolute
sum across the last 6 years:

| `esg.scale` | Σ \|ESG\| | Σ \|FIN\| | %\|ESG\| | Comment |
|---|---|---|---|---|
| 0.10 | 0.94 | 4.39 | 18 % | financial-heavy |
| 0.15 | 1.41 | 4.39 | 24 % | financial-heavy |
| 0.20 | 1.88 | 4.39 | 30 % | financial bias |
| 0.25 | 2.35 | 4.39 | 35 % | edge of window |
| 0.30 | 2.83 | 4.39 | 39 % | financial bias |
| **0.50** | **4.71** | **4.39** | **52 %** | **calibrated, ≈50/50** |
| 1.00 | 9.42 | 4.39 | 68 % | ESG-dominated |

`scale = 0.50` is the new default (was `0.25`, calibrated against the
flat-80 rollout that overstated FIN). The regression test
(`tests/test_rewards.py::test_esg_balance_with_financial_50_50`) now
runs the anchor-tracking rollout and pins the balance to `[35 %, 65 %]`.
**Re-tune `scale` (not `stock_weight` / `flow_weight`) if the balance
drifts.**

### Per-year trajectory at `scale = 0.50` (anchor-tracking, balanced agents)

```
Year   ESG/yr   FIN/yr    %|ESG|   ef     anc_r   cov_frac
 0     +0.000   −0.589      0 %   0.00   1.00     1.00
 1     +0.000   −0.582      0 %   0.00   1.02     1.00
 2     +0.084   −0.542     14 %   0.03   1.07     1.00
 3     +0.205   −0.485     30 %   0.08   1.09     1.00
 4     +0.349   −0.416     46 %   0.13   1.11     1.00
 5     +0.481   −0.363     57 %   0.17   1.18     1.00
 6     +0.588   −0.192     75 %   0.20   1.24     0.50    ← typical mid-game
 7     +0.653   +0.009     99 %   0.27   1.28     0.25
 8     +0.661   +0.046     93 %   0.32   1.33     0.75
 9     +0.430   +0.914     32 %   0.37   1.37     0.17    ← scarcity-cost peak
10     +1.125   −1.281     47 %   0.41   1.44     0.52
11     +1.251   +0.393     76 %   0.46   1.47     0.75
                                          tail-window: 52 % ESG
```

The shape: financial channel is large and *negative* in the early
years (capital cost of green build-out depresses cost_norm-centered
into negative territory), ESG ramps as `ef_ratio` and anchor scarcity
both grow, and both channels swing freely in the late game as scarcity
binds. Signed Σ FIN ≈ 0 across the tail (years 6–11); the |abs|
contribution is what the policy gradient sees.

---

## 3. Strengths

1. **Late investment rewarded by TVM, not horizon penalty.** A year-11
   agent at `ef_ratio = 0.5` receives ≈ `+0.79` at the default
   `scale = 0.50` (`0.50 × (1×0.5 + 1.5×0.5×1.47 + 0)`). Earlier
   investment is still preferred because more years of stock+flow
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
| **Always green** (`ef_ratio = 1`) | Peak ≈ `0.50 × (1 + 1.5 × 1.47 + 0) = 1.60` per year; cumulative ≈ 7 over 12 years. | ✓ upper bound of green reward in steady state |
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

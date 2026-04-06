# Design Document — ETS MARL

## Overview

This document describes the **ETS MARL** simulation: a stylised multi-agent reinforcement-learning model of the EU Emissions Trading System (EU ETS). It is intended as a self-contained reference for new readers — no prior knowledge of the codebase is assumed.

**What the project models:** A simplified carbon allowance market in which energy companies bid for emission permits, invest in cleaner technology, trade allowances between themselves, and learn bidding and investment strategies over thousands of simulated years.

**Why it exists:** To study emergent market behaviour (pricing dynamics, banking incentives, green-investment timing) under regulatory mechanisms (cap trajectory, Market Stability Reserve) using modern multi-agent RL.

**Who participates:** 16 market participants — 8 learning agents trained with PPO/HAPPO and 8 heuristic rule-based bots. Agents span four archetypes (coal-heavy, gas-dominant, transitioner, green-leader), each paired into one financially-motivated and one ESG-balanced company.

**How an episode works:** Each episode simulates 12 years. Every year, participants bid in a sealed-bid uniform-price auction for CO2 allowances, then trade in a bilateral secondary market, and choose how much to invest in renewable capacity. Penalties fall on those without enough allowances to cover emissions. The government cap shrinks by ~4.3–4.4 % annually, creating increasing scarcity that forces decarbonisation.

**Learning objective:** Learning agents maximise a reward signal combining net financial cost (after compliance, trading, investment, and operations) with an optional ESG component (saved-carbon-years). Over 100,000 episodes, agents converge on market strategies.

---

## 1. Scope and Purpose

This document describes the active architecture in `ets_marl_happo_current`.
It supersedes earlier DDPG-era notes and reflects the current HAPPO/PPO setup,
two-phase decision process, technology-resolved companies, bot participants,
and reward/economic mechanisms used in training and evaluation.

The model is a stylized EU ETS micro-market with 16 participants:
- 8 learning agents (PPO with HAPPO-style sequential updates)
- 8 heuristic bot agents (rule-based, not trained)

Each episode simulates 12 years.

## 2. Market Architecture

### 2.1 Auction type (v8.0+)

The primary market uses a **3-Tranche Bid Ladder**: each participant submits 3 independent
(price, quantity) pairs forming a demand curve at multiple price levels. This mirrors the
**demand curves** used in real EEX/ICE call auctions.

- Action space: `[p1, q1, p2, q2, p3, q3, invest_frac, tech_logit0, tech_logit1, tech_logit2]` (10D)
- All three tranches are expanded into separate bid rows and passed to `market_clearing_ets()`.
- **B1 invariant (v8.1)**: Tranches are sorted ascending by price immediately after action
  extraction, before all budget/collateral checks. This applies to **both RL agents and
  heuristic bots**:
  - RL agents: common `raw_tranches.sort()` in `step_auction()`.
  - Bots: also pre-sorted by construction (T1=0.9×, T2=1.0×, T3=1.1× mid price).
- **Phase 2 observation**: D1/D2 feedback dims (tranche fill ratios + price-vs-clearing)
  are stored after B1 sorting. Agents observe which tranche positions filled and by how
  much. This does not affect Phase 1 obs (built before bidding).

Valid bids are sorted by descending price and accepted until auction supply is exhausted.
All winners pay the same uniform clearing price (the marginal accepted bid).

### 2.2 Collateral enforcement (v8.1, E2/E4)

Before each auction, a collateral affordability check scales down bids if:
```
collateral = collateral_fraction × wavg_price × total_qty
           > max_collateral_budget_share × remaining_budget
```
- `collateral_fraction: 0.10` (10% — mid-range of real EUA exchange initial margin 5–15%)
- `max_collateral_budget_share: 0.50` (collateral can use at most 50% of remaining budget)
- E4 config also registers `leverage_multiplier: 3.0`, `suspension_length: 2`,
  `carry_forward_defaults: true` for future enforcement.

### 2.3 Practical clearing details in code

- Bids below effective reserve are rejected.
- Tie-breaks at identical prices are randomized (not pro-rata).
- Optional per-agent holding limit is supported via `max_agent_share`.
- If enabled, under-subscription can cancel the auction; default behavior is to clear partial demand.
- Unsold volume is either:
  - rolled into next year's auction supply, or
  - absorbed into MSR reserve (configurable).

## 3. Cap and Supply Dynamics

### 3.1 Cap path

From v7.0 onward, year-0 cap is calibrated dynamically from active participant
emissions instead of being manually hardcoded in the default config:

$$
cap_{0} = E_{system} \cdot (1 + overhead)
$$

where $E_{system}$ is the sum of initial emissions across all learning agents
and currently active bots. A manual override (`cap_year_0_override`) is still
supported for controlled experiments.

Annual cap follows a linear LRF schedule:

$$
cap_t = cap_0 - \sum_{k=0}^{t-1} lrf_k \cdot cap_0
$$

- Phase 1 (years 0–1, `lrf_phase_switch=2`): LRF = 4.3%
- Phase 2 (years 2+): LRF = 4.4%

Each year's cap declines by a fixed absolute amount (`lrf_k × cap_0`), not by a
compounding fraction. This matches the EU ETS Linear Reduction Factor mechanics
(EU Directive 2003/87/EC Art. 9), which specifies a constant annual absolute reduction.

### 3.2 MSR logic

MSR operates on auction volume (not cap) using TNAC proxy (sum of all banks):

**Three-band withholding (v8.2)** based on legislative TNAC proportions from Decision (EU) 2015/1814 (400:833:1096 Mt scaled to simulation size):
- If TNAC > upper threshold: withhold `0.24 × TNAC` (24% of total TNAC, not just the excess).
- If mid threshold ≤ TNAC ≤ upper threshold: withhold `TNAC − mid` (tapered intake).
- If lower threshold ≤ TNAC < mid threshold: no TNAC-triggered intake.
- If TNAC < lower threshold: release fixed volume from reserve.
- Threshold scaling preserves lower:mid:upper = 400:833:1096 when mapped to simulation scale.
  (`tnac_upper_ratio=0.36`, `tnac_mid_ratio=0.2737`, `tnac_lower_ratio=0.1314` of CAP_0.)
- Release fraction = 6.4% of CAP_0 per year.
- In v8.1, TNAC bounds and release amounts are specified as ratios of
  calibrated year-0 cap (`tnac_*_ratio`, `release_frac`, `emergency_release_frac`)
  and materialized at environment init/reset.
- **1-year TNAC lag (v8.1):** MSR uses *prior-year* TNAC (`_prev_tnac`), not the
  current year's holdings. This follows Decision (EU) 2015/1814 Art. 1(5), with
  intake regime updates from Decision (EU) 2023/852. Year 0
  has no MSR intervention unless `force_msr=True` (burn-in calibration mode).

**Price-responsive safeguards (P9):**
Current implementation includes price-responsive triggers to prevent procyclical supply withdrawal:
- **Containment trigger** (70% of penalty rate or 200 EUR/t absolute): When prices are elevated,
  suppress normal TNAC-triggered withdrawal even if TNAC > upper threshold.
- **Emergency release trigger** (85% of penalty rate or 300 EUR/t absolute, v8.1+):
  Emergency release requires *both* an absolute threshold breach *and* a MA3 price spike
  > 2.5× the prior year's MA3 (smoothed trigger, A4). Prevents procyclical flash releases.

**MSR cancellation mechanism (EU ETS post-2023 reform):**
At the start of each year, MSR holdings exceeding the previous year's auction volume are permanently cancelled. This implements the real EU ETS Directive cancellation rule:
```
excess = max(0, msr_reserve - prev_auction_volume)
msr_reserve -= excess
total_cancelled += excess  # cumulative tracker
```

In this micro-ETS, cancellation rarely triggers due to short 12-year episodes and moderate TNAC levels, but is included for regulatory completeness and long-run realism.

### 3.3 Reserve price mode

Auction reserve can be:
- static (`reserve_price`), or
- dynamic (`max(absolute_floor, discount * MA3_price)`) with fallback behavior when no valid clears exist.

## 4. Participants and Company Model

### 4.1 Agent population

- Learning agents A1-A8: PPO/HAPPO-trained.
- Bot agents B1-B8: heuristic policy for both auction and secondary market.

From v7.0, bot behavior also supports:
- `enhanced_noise` (higher valuation/urgency variance plus optional budget-stress quantity cuts),
- `fade_schedule` (episode-based retirement of bots in reverse index order),
- dynamic cap/MSR recalibration when active bot count changes.

Archetypes are mirrored between learners and bots:
- coal-heavy
- gas-dominant
- transitioner
- green-leader

All participants produce 10 TWh/year.

### 4.2 Technology-resolved generation mix

Each company has a 5-technology portfolio:
- coal
- gas
- onshore wind
- offshore wind
- solar

Model tracks technology-specific:
- emission factors
- capex
- capacity factors
- deploy delays
- operational costs
- decommission costs

### 4.3 Investment and queue dynamics

Agents choose annual invest fraction and target green technology.
Investment is queued and materializes with delay.

Important mechanics:
- Greening-only transition (fossil retired first, then green added).
- Investment failure risk depends on fossil exposure and experience.
- Construction jitter (Poisson delay), cancellation risk, and capex recovery on cancellation.
- Capacity-factor noise can alter realized emissions each year.

v7.0 adds optional **green finance** support:
- extra green-only annual loan headroom,
- extra green-only capex-throughput headroom,
- annual interest cost on utilized green loan,
- interest charged in reward cost accounting (not budget-spent accounting).

### 4.4 Compliance and carry-forward

Compliance is settled against realized emissions plus prior carry-forward obligation:

$$
shortfall_i = \max(0, E_i^{realized} + CF_i^{old} - A_i^{held})
$$

Penalty paid:

$$
penalty_i = shortfall_i \cdot effectivePenaltyRate_t
$$

If enabled, shortfall carries to next year, with optional cap multiplier to prevent runaway debt spirals.

### 4.5 MAC fuel-switching

If carbon price exceeds MAC threshold, company can temporarily switch part of coal dispatch to gas.
This lowers emissions in-year but adds MAC cost. It does not permanently alter long-run technology mix.

## 5. Two-Phase Yearly Decision Process

Each simulation year is split into two decisions.

### Phase 1: Auction + Investment

Action vector (6D):
1. Bid price (EUR/t)
2. Quantity multiplier on estimated need
3. Invest fraction
4. Onshore logit
5. Offshore logit
6. Solar logit

Technology choice is `argmax(logits)`.

### Phase 2: Secondary market

Action vector (2D):
1. Secondary price (absolute EUR/t)
2. Secondary quantity (positive buy, negative sell)

Secondary market uses bilateral double-auction matching with spread tolerance.
Participants can sell from current allocation plus bank (no short selling beyond holdings).

## 6. Observation Spaces

### 6.1 Phase 1 observation

Base dimension: **24** (v8.1, after Phase G consolidation from 28D).

Consolidated from 28D by removing:
- `expected_price_ar1` (redundant with MA3 + time signal)
- 2 raw technology fraction dims (5 → 3 summary fracs)
- `predicted_msr_withholding` (derivable from TNAC proxy)

Includes:
- `[0]` time (normalized by n_years)
- `[1]` cap (normalized)
- `[2]` 3-year MA3 clearing price (normalized by price_max)
- `[3]` green_frac = onshore + offshore + solar fraction (mix[2]+mix[3]+mix[4])
- `[4]` coal_frac (mix[0])
- `[5]` gas_frac (mix[1])
- `[6]` emissions (normalized)
- `[7]` expected annual emissions (normalized, no risk buffer — v8.2)
- `[8]` risk factor (p_fail)
- `[9]` investment experience (consecutive successes)
- `[10]` auction gap (banked allowances normalized)
- `[11–13]` construction queue (onshore, offshore, solar)
- `[14]` weighted emission factor
- `[15]` last secondary price (normalized)
- `[16]` last secondary volume (normalized)
- `[17]` carry-forward obligation (Mt / 5)
- `[18]` TNAC proxy (total bank / cap, clipped [0,3])
- `[19]` effective reserve price (normalized)
- `[20]` auction volume ratio (last_auction_volume / cap_t)
- `[21]` MSR reserve signal (msr_reserve / cap_t)
- `[22]` own bank ratio (clipped [0,5], normalized /5)
- `[23]` budget headroom (1 – budget_spent / annual_budget)

If opponent modeling is enabled:

$$
\text{obsDimPhase1} = 24 + 5 \cdot (N_{total} - 1)
$$

Each opponent contributes a public 5D tuple:
- normalized emissions
- normalized carry-forward
- green fraction
- fossil fraction
- total queue size

With 16 total participants:
- Phase 1 dimension = 24 + 5×15 = **99D**

### 6.2 Phase 2 observation

Phase 2 appends 13 features to Phase 1:

**Standard 7 dims:**
- allocation / 5
- clearing price / price_max
- net compliance position: (bank + allocation − emissions − carry_forward) / 5
- emission shock (P5)
- auction savings proxy: (allocation × 100 − payment) / 1000
- coverage ratio: (bank + allocation) / obligation, clipped [0,3], /3
- normalized carry-forward: carry_forward / estimated_need, clipped [0,3]

**D1/D2 — 6 per-tranche feedback dims (v8.1, auction only):**
- `[base+7]` tranche 1 fill ratio (0 = no fill, 1 = full fill)
- `[base+8]` tranche 2 fill ratio
- `[base+9]` tranche 3 fill ratio
- `[base+10]` (tranche 1 price − clearing price) / price_norm
- `[base+11]` (tranche 2 price − clearing price) / price_norm
- `[base+12]` (tranche 3 price − clearing price) / price_norm

$$
\text{obsDimPhase2} = \text{obsDimPhase1} + 13
$$

With 16 total participants (no opponent modeling): 24D Phase 1, **37D** Phase 2.

## 7. Reward Design (Current)

Per-agent reward is split into a **base reward** and a **shaping reward** that decays
over training:

$$
R_i = \underbrace{w_{cost,i}(-\text{costNorm}_i) + w_{green,i}(\text{esgScale}_i\cdot \text{esgRaw}_i)
  - \text{penalty\_norm}_i - \text{oppCost}_i}_{\text{base reward}}
  + \underbrace{(\text{greenBonus}_i + \text{efficiencyBonus}_i) \cdot \text{shapingWeight}}_{\text{shaping reward}}
  + \text{terminalValues}_i
$$

Where:
- `costNorm` is total non-penalty cost, scaled by `annual_budget` (v8.3: changed from /1000).
- Costs include auction, secondary, investment, OPEX delta (v8.3), budget penalties, capex throughput penalties, and MAC cost.
- **OPEX delta** (v8.3): Only the change from baseline OPEX enters the cost signal: `opex_delta = current_opex - baseline_opex`. Positive delta = costs rose; negative = OPEX savings from greening.
- `penalty_norm` is the compliance penalty at full strength, normalized by `annual_budget`.
- `greenBonus` rewards positive green share change with shaping decay over training.
- `efficiencyBonus` (v8.1) rewards emission-factor improvement vs initial EF, scaled by
  remaining time and carbon price. Decays with `shaping_weight` — does not permanently
  distort the financial reward channel.
- `esgRaw` uses saved-carbon-years style term before weighting.
- `esgScale_i` (v8.3) is per-agent: `base_esg_scale × (1000 / annual_budget)`, compensating
  for the divisor change to preserve the ESG-to-cost balance.
- `oppCost` is a cost-of-capital term on post-compliance banked allowances:
  $\text{oppCost}_i = holdings_i \cdot price_t \cdot r_{opp} / \text{annual\_budget}$.
- **Collateral normalization** (v8.3): Collateral cost in the reward uses `/annual_budget`,
  consistent with `collateral_load_last` obs[29] which already normalizes by annual_budget.

This structure makes objective weights explicit:
- Financial agents (`w_cost=1.0`, `w_green=0.0`) optimize pure cost.
- ESG agents (`w_cost=0.5`, `w_green=0.5`) are guaranteed an exact 50/50 split
  between financial and environmental reward channels (excluding transient shaping terms).

Terminal values in final year (configurable):
- bank terminal value with diminishing returns:

$$
V^{bank}_i = \log\left(1 + \frac{B_i}{\max(\hat{E}_i, 0.1)}\right)
\cdot \hat{E}_i \cdot \frac{P_T}{\text{annual\_budget}}
$$

where $B_i$ is banked allowances, $\hat{E}_i$ is annual estimated need,
and $P_T$ is the terminal price anchor.

Thesis justification: this specification preserves monotonicity (more prudent
banking still increases value) while imposing economically meaningful
diminishing marginal value on very large stocks. A one-year hedge remains
valuable, but speculative multi-year hoarding is discounted relative to a
linear payoff, improving market realism by encouraging secondary-market release
instead of end-horizon stockpile accumulation.
- queue terminal value (discounted future emissions savings from queued projects)

Policy-timing note for reward interpretation: the MSR 1-year TNAC lag (v8.1) is
retained when reading early-episode rewards. This is intentional: MSR decisions at
year t use the prior year's TNAC, matching the real governance calendar. Year-0 rewards
are not affected by MSR unless `force_msr=True`.

Terminal price anchor uses max of:
- auction clearing
- secondary clearing
- 80% of inflation-adjusted penalty rate

### 7.1 Diagnostic Scores (v8.1)

`compute_diagnostic_score()` returns interpretable per-agent metrics independent of
reward normalization artifacts:

| Score | Formula | Meaning |
|---|---|---|
| `S_financial` | `max(0, 1 - budget_spent / annual_budget)` | Cost efficiency [0,1] |
| `S_green` | `ef_improvement / initial_ef` | EF progress [0,1] |
| `S_penalty` | `1` (proxy; override with logged penalties) | Compliance quality [0,1] |
| `S_composite` | `w_cost × S_fin + w_green × S_grn + 0.3 × S_pen` | Weighted blend |

Scores are logged to year-level CSV as `diag_S_*_Ai` columns, and
episode-mean scores are printed in the training console.

## 8. Learning System

### 8.1 Policy/critic structure

Each learning company has:
- auction policy network
- secondary policy network
- value network

Actors are decentralized; critic can be centralized (MAPPO mode) over concatenated multi-agent state.

### 8.2 HAPPO/PPO training behavior

- On-policy episode rollouts.
- Sequential policy updates enabled by HAPPO option.
- GAE + clipped PPO objective.
- KL early-stopping and optional KL anchor to frozen BC policy.

### 8.3 Stabilization features

- Behavioral cloning warm-start from heuristic policy (optional).
- Hidden heuristic burn-in warm-start (optional) to initialize banks, MSR reserve,
  and MA3 price history before visible year 0.
- Reward normalization per agent.
- Entropy decay schedule.
- Epsilon-greedy exploration in physical action space with anchored Gaussian sampling.
- Historical Policy Pool (periodic snapshots and random swaps for opponent diversity).
- Diagnostics for stuck-market or degenerate-policy regimes.

v7.0 adds a configurable **tabula-rasa mode** for ablation:
- disables BC pretraining and KL anchor,
- removes action anchors,
- switches epsilon exploration to uniform,
- overrides warmup/decay schedules from `n_episodes` fractions.

## 9. Economic and Financial Layers

### 9.1 Inflation path

Inflation is sampled per episode (shared by all companies) and compounds annually.
It scales penalty, capex, OPEX, decommissioning, MAC, and power price base.

### 9.2 Electricity revenue channel

Electricity revenue can be enabled:

$$
P_{elec} = P_{base} + \text{passthrough} \cdot P_{carbon} \cdot EF_{system}
$$

Revenue offsets cost signal and links carbon prices to generation margins.

### 9.3 Unified budget envelope

Each company has an annual spending envelope for all major outlays.
Separate capex throughput constraint models physical delivery bottlenecks.

### 9.4 Budget Hardening Regime (v8.4)

Annual spending is subject to a tiered penalty regime:
- **Below 100%** (`soft_zone_start`): No penalty.
- **100–115%** (`soft_zone_start` → `hard_cap_fraction`): Quadratic penalty
  that scales with overshoot amount: `coef × (normalized²) × overshoot_abs`.
- **Above 115%**: Penalty continues to grow steeply (normalized > 1).
- **Investment hard gate**: When enabled, `step_auction()` scales down
  investment fraction if total projected spending would exceed the hard cap.
  This applies to the 10D action format with 3-tranche auction bids.

This replaces the previous 3-tier contingency/quadratic system and provides
clearer economic semantics: spending is free up to budget, incurs increasing
opportunity cost in the soft zone, and is structurally prevented from running
far above the hard cap.

## 10. Current Simplifications

The model remains stylized despite expanded realism:
- Single annual bid per participant (no multi-step bid curve).
- Simplified secondary market matching (no full limit-order book dynamics).
- Fixed annual output per company (10 TWh).
- Closed system (no cross-market linkage or imports).
- AR(1)-style expected price signal instead of full forward curve equilibrium.

These are deliberate to keep MARL tractable while preserving key strategic channels.

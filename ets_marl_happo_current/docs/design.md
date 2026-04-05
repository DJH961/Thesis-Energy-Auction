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

### 2.1 Auction type

The primary market is a uniform-price, sealed-bid, multi-unit buyer auction.
Each participant submits one annual bid tuple:
- bid price (EUR/t)
- bid quantity (Mt)

Valid bids are sorted by descending price and accepted until auction supply is exhausted.
All winners pay the same clearing price (the marginal accepted bid).

### 2.2 Practical clearing details in code

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
(EU Directive 2003/87/EC Art. 9).

### 3.2 MSR logic

MSR operates on auction volume (not cap) using TNAC proxy (sum of all banks):
**Three-band withholding (v7.5)** based on legislative TNAC proportions from Decision (EU) 2015/1814 (400:833:1096 Mt scaled to simulation size):
- If TNAC > upper threshold: withhold `0.24 × TNAC` (24% of total TNAC, not just the excess).
- If mid threshold ≤ TNAC ≤ upper threshold: withhold `TNAC − mid` (tapered intake).
- If lower threshold ≤ TNAC < mid threshold: no TNAC-triggered intake.
- If TNAC < lower threshold: release fixed volume from reserve.
- Threshold scaling preserves lower:mid:upper = 400:833:1096 when mapped to simulation scale.
  (`tnac_upper_ratio=0.36`, `tnac_mid_ratio=0.2737`, `tnac_lower_ratio=0.1314` of CAP_0.)
- Release fraction = 6.4% of CAP_0 per year (v7.4).
- **1-year TNAC lag (v7.4):** MSR uses *prior-year* TNAC (`_prev_tnac`), not the
  current year's holdings. This follows Decision (EU) 2015/1814 Art. 1(5), with
  intake regime updates from Decision (EU) 2023/852.
  Year 0 has no MSR intervention unless `force_msr=True` (burn-in mode).

**Price-responsive safeguards (P9):**
- **Containment trigger** (70% of penalty rate or 200 EUR/t absolute): When prices are elevated, suppress TNAC-triggered withdrawal.
- **Emergency release trigger (v7.4 A4, smoothed)**: Emergency release requires *both* an absolute threshold breach (≥ 85% of penalty rate or ≥ 300 EUR/t) *and* a MA3 price spike > 2.5× the prior year's MA3.

**MSR cancellation mechanism (EU ETS post-2023 reform):**
At the start of each year, MSR holdings exceeding the previous year's auction volume are permanently cancelled:
```
excess = max(0, msr_reserve - prev_auction_volume)
msr_reserve -= excess
total_cancelled += excess  # cumulative tracker
```

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

Base dimension: 28.

Includes:
- time and cap
- price signals (MA3, expected AR(1), last secondary)
- full technology mix and emissions/need/risk indicators
- queue state
- carry-forward obligation
- TNAC proxy
- effective reserve signal
- secondary volume and profit signal

If opponent modeling is enabled:

$$
obsDimPhase1 = 28 + 5 (N_{total} - 1)
$$

Each opponent contributes public 5D tuple:
- normalized emissions
- normalized carry-forward
- green fraction
- fossil fraction
- total queue size

With 16 total participants:
- phase 1 dimension = 103

### 6.2 Phase 2 observation

Phase 2 appends 7 auction-result features to phase 1:
- allocation
- clearing price
- net compliance position
- emission shock
- auction savings proxy
- coverage ratio
- normalized carry-forward

$$
obsDimPhase2 = obsDimPhase1 + 7
$$

With 16 total participants:
- phase 2 dimension = 110

## 7. Reward Design (v7.4)

Per-agent reward is split into a **base reward** and a **shaping reward** that decays
over training:

$$
R_i = \underbrace{w_{cost,i}(-\text{costNorm}_i) + w_{green,i}(\text{esgScale}_i\cdot \text{esgRaw}_i)
  - \text{penalty\_norm}_i - \text{oppCost}_i}_{\text{base reward}}
  + \underbrace{(\text{greenBonus}_i + \text{efficiencyBonus}_i) \cdot \text{shapingWeight}}_{\text{shaping reward}}
  + \text{terminalValues}_i
$$

Where:
- `costNorm` is total non-penalty cost, scaled by `annual_budget` (v7.6: changed from /1000).
- Costs include auction, secondary, investment, OPEX delta (v7.6), budget penalties, capex throughput, and MAC cost.
- **OPEX delta** (v7.6): Only the change from baseline OPEX enters the cost signal: `opex_delta = current_opex - baseline_opex`. Positive delta = costs rose; negative = OPEX savings from greening.
- `penalty_norm` is the compliance penalty at full strength, normalized by `annual_budget`.
- `greenBonus` rewards positive green share change with shaping decay over training.
- `efficiencyBonus` (v7.4) rewards emission-factor improvement vs initial EF, scaled by
  remaining time and carbon price. Decays with `shaping_weight`.
- `esgRaw` uses saved-carbon-years style term before weighting.
- `esgScale_i` (v7.6) is per-agent: `base_esg_scale × (1000 / annual_budget)`, compensating
  for the divisor change to preserve the ESG-to-cost balance.
- `oppCost` is a cost-of-capital term on post-compliance banked allowances:
  $\text{oppCost}_i = holdings_i \cdot price_t \cdot r_{opp} / \text{annual\_budget}$.

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

Policy-timing note for reward interpretation: the MSR 1-year TNAC lag (v7.4) means
year-0 rewards are not affected by MSR. Year 1+ rewards reflect prior-year TNAC decisions,
matching EU ETS governance calendar.

Terminal price anchor uses max of:
- auction clearing
- secondary clearing
- 80% of inflation-adjusted penalty rate

### 7.1 Diagnostic Scores (v7.4)

`compute_diagnostic_score()` returns interpretable per-agent metrics:

| Score | Formula | Meaning |
|---|---|---|
| `S_financial` | `max(0, 1 - budget_spent / annual_budget)` | Cost efficiency [0,1] |
| `S_green` | `ef_improvement / initial_ef` | EF progress [0,1] |
| `S_composite` | `w_cost × S_fin + w_green × S_grn + 0.3 × S_pen` | Weighted blend |

Scores are logged to year-level CSV as `diag_S_*_Ai` and printed in training console.

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

## 10. Current Simplifications

The model remains stylized despite expanded realism:
- Single annual bid per participant (no multi-step bid curve).
- Simplified secondary market matching (no full limit-order book dynamics).
- Fixed annual output per company (10 TWh).
- Closed system (no cross-market linkage or imports).
- AR(1)-style expected price signal instead of full forward curve equilibrium.

These are deliberate to keep MARL tractable while preserving key strategic channels.

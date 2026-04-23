# Design Document — ETS MARL

## Overview

This document describes the **ETS MARL** simulation: a stylised multi-agent reinforcement-learning model of the EU Emissions Trading System (EU ETS). It is intended as a self-contained reference for new readers — no prior knowledge of the codebase is assumed.

**What the project models:** A simplified carbon allowance market in which energy companies bid for emission permits, invest in cleaner technology, trade allowances between themselves, and learn bidding and investment strategies over thousands of simulated years.

**Why it exists:** To study emergent market behaviour (pricing dynamics, banking incentives, green-investment timing) under regulatory mechanisms (cap trajectory, Market Stability Reserve) using modern multi-agent RL.

**Who participates:** The current default profile (`v7.12`) is pure MARL with 8 learning agents trained with PPO/HAPPO and 0 bots. The environment still supports optional heuristic bot participants for ablation/calibration runs.

**How an episode works:** Each episode simulates 12 years. Every year, participants bid in a sealed-bid uniform-price auction for CO2 allowances, then trade in a bilateral secondary market, and choose how much to invest in renewable capacity. Penalties fall on those without enough allowances to cover emissions. The government cap shrinks by ~4.3–4.4 % annually, creating increasing scarcity that forces decarbonisation.

**Learning objective:** Learning agents maximise a reward signal combining net financial cost (after compliance, trading, investment, and operations) with an optional ESG component (saved-carbon-years). Over 100,000 episodes, agents converge on market strategies.

---

## 1. Scope and Purpose

This document describes the active architecture in `ets_marl_happo_current`.
It supersedes earlier DDPG-era notes and reflects the current HAPPO/PPO setup,
two-phase decision process, technology-resolved companies, bot participants,
and reward/economic mechanisms used in training and evaluation.

The model is a stylized EU ETS micro-market with configurable participants:
- 8 learning agents (PPO with HAPPO-style sequential updates)
- 0-8 heuristic bot agents (rule-based, not trained)

Default configuration in `v7.12` is:
- 8 learning agents
- 0 bot agents (`n_bot_agents: 0`)

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
- A collateral safety-net clip rescales bids when
  `collateral_fraction × max(0, bid_price − max(effective_reserve, price_ma3)) × bid_qty`
  exceeds `max_collateral_budget_share × remaining_budget`.
- The collateral clip is for training stability during exploration (not an economic mechanism);
  bot-only runs should have near-zero clip events, and non-zero events indicate
  heuristic/environment mismatch.
- If enabled, under-subscription can cancel the auction; default behavior is to clear partial demand.
- Unsold volume is configurable:
  - rolled into next year's auction supply (default in `v7.12`), or
  - absorbed into MSR reserve.

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

**Phantom-bidder-aware scarcity calibration (v7.13):**

When the phantom bidder is enabled, the cap overhead must be calibrated jointly
with expected phantom demand so that compliance agents face *minor* supply
scarcity in year 0, not excess scarcity. The phantom takes a fraction
$f_{\text{ph}}$ of auction supply each year:

$$
f_{\text{ph}} = \frac{qty\_frac\_lo + qty\_frac\_hi}{2} \times P(\text{bid} \geq \text{reserve}) \approx 0.125 \times 0.90 \approx 0.113
$$

The effective compliance supply in year 0 is therefore:

$$
S_{\text{eff},0} = cap_0 \times (1 - f_{\text{ph}}) = E_{system} \times (1 + overhead) \times 0.887
$$

For a target net shortfall $\delta$ (e.g., 4%):

$$
overhead = \frac{1 - \delta}{1 - f_{\text{ph}}} - 1 = \frac{0.96}{0.887} - 1 \approx +0.08
$$

The default `cap_overhead_pct: 0.08` (+8%) yields $S_{\text{eff},0} \approx 0.958 \times E_{system}$, a ~4% compliance supply shortfall in year 0. The LRF then compounds scarcity at ~4.3-4.4% per year, creating meaningful compliance pressure by mid-episode while keeping year-0 incentives moderate.

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
- Bot agents B1-B8 (optional): heuristic policy for both auction and secondary market.

Current default (`v7.12`) uses no bots (`n_bot_agents=0`).

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

### 4.6 Revenue-based dynamic budget

When `budget.mode` is set to `revenue_based`, annual budgets are computed dynamically
from electricity revenue rather than being fixed at episode start:

$$
\text{budget}_t = \text{EMA}\bigl(\text{revenue}_t - \text{opex}_t + \text{debt\_headroom}_i,\;\alpha\bigr)
$$

where `revenue` comes from `Company.compute_revenue()` (electricity sales with carbon-cost
passthrough using MA3-smoothed carbon price and system-average emission factor), and
`debt_headroom` is an archetype-specific buffer configured per agent. EMA smoothing
(`ema_alpha`, default 0.3) prevents erratic year-to-year budget swings.

When `budget.mode` is `fixed` (default), the original static annual budget is used unchanged.

### 4.7 Emergency loan system

When enabled (`budget.emergency_loan.enabled`), agents facing auction default receive an
emergency loan instead of immediate suspension:

- **Trigger**: Shortfall at auction settlement exceeds remaining budget but falls within
  `max_loan_fraction × annual_budget`.
- **Mechanics**: `apply_emergency_loan(shortfall)` adds the shortfall (plus accrued interest
  at `loan_interest_rate`, default 8%) to `_loan_outstanding`. Annual repayment is deducted
  at year start via `apply_loan_repayment()`.
- **Tracking**: `_loan_outstanding`, `_loan_repayment_annual`, `_years_under_loan` are
  maintained on the `Company` object and exposed in observations (see §6).
- **Heuristic loan-awareness**: Bots with outstanding loans reduce auction quantity (−30%),
  investment (−50%), and secondary buy volume (−40%).

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

Base dimension: **33**.

Includes:
- time and cap
- price signals (MA3, expected AR(1), last secondary)
- full technology mix and emissions/need/risk indicators
- queue state
- carry-forward obligation
- TNAC proxy
- effective reserve signal
- secondary volume and profit signal
- `[27]` budget headroom (1 – budget_spent / annual_budget)
- `[28]` safety/collateral dims
- `[29]` collateral load last
- `[30]` bid_affordability_last: last year's bid total / remaining budget (clipped [0,1])
- `[31]` loan_outstanding_norm: emergency loan / annual_budget
- `[32]` years_under_loan_norm: years under active loan / 5

If opponent modeling is enabled:

$$
obsDimPhase1 = 33 + 5 (N_{total} - 1)
$$

Each opponent contributes public 5D tuple:
- normalized emissions
- normalized carry-forward
- green fraction
- fossil fraction
- total queue size

With `N_total` total participants:
- phase 1 dimension = `33 + 5 x (N_total - 1)`

Default `v7.12` profile (`N_total=8`):
- phase 1 dimension = `33 + 5 x 7 = 68`

### 6.2 Phase 2 observation

Phase 2 appends **10** auction-result and compliance-awareness features to phase 1:
- allocation
- clearing price
- net compliance position
- emission shock
- auction savings proxy
- coverage ratio
- normalized carry-forward
- collateral_locked_norm: this year's collateral locked / annual_budget
- budget_remaining_phase2_norm: remaining annual budget after auction / annual_budget
- compliance_liability_norm: (emissions + carry_forward − bank − allocation) / annual_budget

$$
obsDimPhase2 = obsDimPhase1 + 10
$$

With `N_total` total participants:
- phase 2 dimension = `obsDimPhase1 + 10`

Default `v7.12` profile (`N_total=8`):
- phase 2 dimension = `68 + 10 = 78`

## 7. Reward Design (v7.12)

**Reward channel logging:** After each year, `_last_reward_channels` and
`_last_auction_reward_channels` dicts are populated with named components. These are for
debugging/analysis only and do not affect reward computation.

Per-agent reward uses a fixed-scale base formulation (no shaping channels):

$$
R_i = w_{cost,i}(-\text{costNorm}_i) + w_{green,i}(\text{esgSignal}_i)
  - \text{penaltyNorm}_i
  + \text{terminalValues}_i
$$

Where:
- `costNorm` is total non-penalty cost scaled by `REWARD_SCALE = 1000`.
- Costs include auction, secondary, investment, OPEX delta, restored soft budget penalty,
  restored soft capex penalty, MAC cost, collateral cost, and green-loan interest.
- `penaltyNorm` is compliance penalty at full strength, scaled by `REWARD_SCALE`.
- `esgSignal` is saved-carbon-years style improvement signal (global ESG scale).
- No shaping reward terms are active in the reward path.

This keeps objective weights explicit:
- Financial agents (`w_cost=1.0`, `w_green=0.0`) optimize pure cost.
- ESG-balanced agents (`w_cost=0.5`, `w_green=0.5`) trade off cost and ESG signal.

Terminal values in final year (configurable):
- bank terminal value with diminishing returns:

$$
V^{bank}_i = \log\left(1 + \frac{B_i}{\max(\hat{E}_i, 0.1)}\right)
\cdot \hat{E}_i \cdot \frac{P_T}{\text{REWARD\_SCALE}}
$$

where $B_i$ is banked allowances, $\hat{E}_i$ is annual estimated need,
and $P_T$ is the terminal price anchor.

- queue terminal value (discounted future emissions savings from queued projects)
  with a completion-fraction discount to prevent end-of-episode gaming of
  long-lead projects.

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

### 9.4 Budget Hardening Regime

Annual spending is subject to a tiered penalty regime:
- **Below 100%** (`soft_zone_start`): No penalty.
- **100–115%** (`soft_zone_start` → `hard_cap_fraction`): Quadratic penalty
  that scales with overshoot amount: `coef × (normalized²) × overshoot_abs`.
- **Above 115%**: Penalty continues to grow steeply (normalized > 1).
- **Investment hard gate**: When enabled, `step_auction()` scales down
  investment fraction if total projected spending would exceed the hard cap.

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

## 11. Phantom Bidder (v7.13)

### 11.1 Motivation

In a uniform-price auction, a symmetric Nash equilibrium exists where all
compliance agents bid exactly at the reserve price: the lowest possible price
that still guarantees inclusion. This "floor-bidding equilibrium" is stable but
economically pathological — the clearing price never rises above the reserve
and there is no incentive to bid truthfully.

The phantom bidder breaks this equilibrium by introducing **stochastic supply
scarcity**: because a random fraction of supply is consumed before compliance
agents' bids are processed, a floor bid sometimes wins full allocation,
sometimes partial, and sometimes nothing. This destroys the certainty that
a floor bid is "always safe".

### 11.2 Economic basis

In the real EU ETS, approximately 40% of primary auction demand comes from
financial intermediaries — banks, hedge funds, and proprietary traders — who
are permitted under EU Regulation 1031/2010 to bid at primary auctions
without a compliance obligation. These participants hold allowances
speculatively and do not surrender them for compliance. The phantom bidder
models this demand pool.

### 11.3 Mechanism

**Implementation** (`src/environment/phantom_bidder.py`):

The `PhantomBidder` is injected as a synthetic participant with reserved
`agent_id = n_total` (one beyond all real agents). Each year's auction
proceeds in three steps:

1. Phantom samples `(bid_price, bid_qty)` from its distributions.
2. The phantom's bid row is prepended to the bids array before clearing.
3. `market_clearing_ets` is called with `n_agents = n_total + 1`. After
   clearing, index `n_total` is stripped from the allocations/payments array.
   The phantom's allocation represents supply consumed by financial traders and
   is intentionally discarded (not credited to any agent's holdings).

**Price distribution:**

$$
\log(p_{\text{phantom}}) \sim \mathcal{N}(\log(\text{anchor}),\; \sigma^2)
$$

where $\text{anchor} = \max(MA3,\; \text{reserve} + \delta_{\min})$ and
$\sigma = 0.45$ (high variance, EU ETS calibrated). The median price equals the
anchor; the arithmetic mean is $\approx 1.11 \times \text{anchor}$ due to the
right skew of the lognormal. The price is clipped to
$[\text{reserve} - 5,\; 0.65 \times \text{effective penalty rate}]$.

Because the anchor tracks MA3, when agents bid above floor and the MA3 rises,
the phantom also rises — the distribution self-stabilises.

**Quantity distribution:**

$$
q_{\text{phantom}} \sim \mathcal{U}(qty\_frac\_lo,\; qty\_frac\_hi) \times q_{\text{auction}}
$$

Default range [5%, 20%] of auction supply. Expected fraction consumed
(accounting for below-reserve rejection probability) ≈ 11%.

**Below-reserve draws:** When the sampled price falls below the reserve, the
bid is submitted but rejected by the normal reserve-price filter in
`market_clearing_ets`. Compliance agents observe full auction supply that year.
This teaches agents that even without phantom squeezing, it is not guaranteed
to always be the clearing price setter.

### 11.4 What the phantom does NOT do

- Does **not** participate in compliance, secondary market, or MSR.
- Does **not** appear in any reward computation.
- Does **not** affect TNAC accounting (phantom allocation is discarded).
- Does **not** have special information or see other bids.

### 11.5 Supply-scarcity interaction

The phantom is calibrated jointly with `cap_overhead_pct` (§3.1). The intended
operating point: compliance agents face **minor net scarcity** in year 0, not
excess. With `cap_overhead_pct: 0.08` and phantom's expected 11% consumption:

| Year | cap / E  | After phantom / E | Net shortage |
|------|----------|-------------------|--------------|
| 0    | 1.08     | 0.958             | ~4%          |
| 3    | 0.94     | 0.834             | ~17%         |
| 6    | 0.80     | 0.710             | ~29%         |
| 12   | 0.52     | 0.461             | ~54%         |

(Before green investment, which reduces agents' effective emission needs.)

### 11.6 Configuration

```yaml
phantom_bidder:
  enabled: true
  qty_frac_lo: 0.05                 # 5% of auction supply minimum
  qty_frac_hi: 0.20                 # 20% of auction supply maximum
  price_lognormal_sigma: 0.45       # lognormal sigma; median=anchor, mean≈1.11×anchor
  price_min_above_reserve: 2.0      # lower bound of anchor above reserve (EUR/t)
  price_max_frac_penalty: 0.65      # upper bound: 65% of effective penalty rate
  price_min_below_reserve_buffer: 5.0  # max below-reserve draw allowed (EUR/t)
```

### 11.7 Diagnostics

- `phantom_bid_price`, `phantom_bid_qty`, `phantom_active` are logged to the
  year-level episode log (`env.episode_log[-1]`).
- `phantom_active_pct` is written to the episode CSV: percentage of years
  the phantom was active (bid ≥ reserve) in that episode.
- The training console `Bid/yr` line shows `(+Ph X%)` suffix when the
  phantom was active for at least one year in the current episode.

## 12. Equilibrium-Breaking Mechanisms (v7.13)

Several mechanisms work together to prevent degenerate floor-bidding equilibria:

| Mechanism | What it does |
|---|---|
| **Phantom Bidder** (§11) | Stochastic supply squeeze; floor bids sometimes fail to acquire any permits |
| **ESG Compliance Gate** | `esg_signal × coverage_frac²` — non-compliant agents lose ESG bonus |
| **Private Urgency Scalars** | Per-episode LogNormal penalty multiplier; destroys symmetric cost structure |
| **HPP Heuristic Seeding** | BC-trained WTP-bidding opponents always present in HPP pool |
| **Liquidity Pool Floor** | Secondary market price ≥ 25% of penalty rate; always a meaningful sell side |

These are configurable and independently enable/disableable in `default.yaml`.

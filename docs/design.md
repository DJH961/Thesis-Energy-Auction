# Design Document — ETS MARL

## Overview

This document describes the **ETS MARL** simulation: a stylised multi-agent reinforcement-learning model of the EU Emissions Trading System (EU ETS). It is intended as a self-contained reference — no prior knowledge of the codebase is assumed.

**What the project models:** A simplified carbon allowance market in which energy companies bid for emission permits, invest in cleaner technology, trade allowances between themselves, and learn bidding and investment strategies over thousands of simulated years.

**Why it exists:** To study emergent market behaviour (pricing dynamics, banking incentives, green-investment timing) under regulatory mechanisms (cap trajectory, Market Stability Reserve) using modern multi-agent RL.

**Who participates:** The current default profile is pure MARL with 8 learning agents trained with PPO/HAPPO and 0 bots. The environment still supports optional heuristic bot participants for ablation/calibration runs.

**How an episode works:** Each episode simulates 12 years. Every year, participants bid in a sealed-bid uniform-price auction for CO2 allowances, then trade in a bilateral secondary market, and choose how much to invest in renewable capacity. Penalties fall on those without enough allowances to cover emissions. The government cap shrinks by ~4.3–4.4% annually, creating increasing scarcity that forces decarbonisation.

**Learning objective:** Learning agents maximise a reward signal combining net financial cost (after compliance, trading, investment, and operations) with an optional ESG component (saved-carbon-years). Over 100,000 episodes, agents converge on market strategies.

---

## 1. Scope and Purpose

This document describes the active architecture in `ets_marl_happo_current`.

The model is a stylized EU ETS micro-market with configurable participants:
- 8 learning agents (PPO with HAPPO-style sequential updates)
- 0–8 heuristic bot agents (rule-based, not trained)

Default configuration uses 8 learning agents and 0 bot agents (`n_bot_agents: 0`). Each episode simulates 12 years.

## 2. Market Architecture

### 2.1 Auction type

The primary market is a uniform-price, sealed-bid, multi-unit buyer auction.
Each participant submits one annual bid tuple:
- bid price (EUR/t)
- bid quantity (Mt)

Valid bids are sorted by descending price and accepted until auction supply is exhausted.
All winners pay the same clearing price (the marginal accepted bid).

### 2.2 Practical clearing details

- Bids below effective reserve are rejected.
- Tie-breaks at identical prices are randomized (not pro-rata).
- Optional per-agent holding limit is supported via `max_agent_share`.
- A collateral safety-net clip rescales bids when
  `collateral_fraction × max(0, bid_price − max(effective_reserve, price_ma3)) × bid_qty`
  exceeds `max_collateral_budget_share × remaining_budget`.
- The collateral clip is for training stability during exploration (not an economic mechanism);
  non-zero clip events indicate heuristic/environment mismatch.
- If enabled, under-subscription can cancel the auction; default behavior is to clear partial demand.
- Unsold volume is configurable: rolled into next year's auction supply (default) or absorbed into the MSR reserve.

## 3. Cap and Supply Dynamics

### 3.1 Cap path

Year-0 cap is calibrated dynamically from active participant emissions:

$$
cap_{0} = E_{system} \cdot (1 + overhead)
$$

where $E_{system}$ is the sum of initial emissions across all active participants. A manual override (`cap_year_0_override`) is supported for controlled experiments. The default `cap_overhead_pct: 0.02` (+2%) creates a small initial supply surplus that compounds into meaningful scarcity by mid-episode via the LRF.

Annual cap follows a linear LRF schedule:

$$
cap_t = cap_0 - \sum_{k=0}^{t-1} lrf_k \cdot cap_0
$$

- Years 0–1 (`lrf_phase_switch=2`): LRF = 4.3%
- Years 2+: LRF = 4.4%

Each year's cap declines by a fixed absolute amount (`lrf_k × cap_0`), not by a compounding fraction. This matches the EU ETS Linear Reduction Factor mechanics (EU Directive 2003/87/EC Art. 9).

### 3.2 MSR logic

MSR operates on auction volume (not cap) using a TNAC proxy (sum of all agent banks).

Three-band withholding based on legislative TNAC proportions from Decision (EU) 2015/1814 (400:833:1096 Mt scaled to simulation size):
- If TNAC > upper threshold: withhold `0.24 × TNAC` (24% of total TNAC).
- If mid threshold ≤ TNAC ≤ upper threshold: withhold `TNAC − mid` (tapered intake).
- If lower threshold ≤ TNAC < mid threshold: no TNAC-triggered intake.
- If TNAC < lower threshold: release fixed volume from reserve.
- Threshold scaling preserves lower:mid:upper = 400:833:1096 when mapped to simulation scale
  (`tnac_upper_ratio=0.36`, `tnac_mid_ratio=0.2737`, `tnac_lower_ratio=0.1314` of CAP_0).
- Release fraction = 6.4% of CAP_0 per year.

**1-year TNAC lag:** MSR uses *prior-year* TNAC (`_prev_tnac`), not the current year's holdings. This follows Decision (EU) 2015/1814 Art. 1(5), with intake regime updates from Decision (EU) 2023/852. Year 0 has no MSR intervention unless `force_msr=True` (burn-in mode).

**Price-responsive safeguards:**
- **Containment trigger** (70% of penalty rate or 200 EUR/t absolute): When prices are elevated, suppresses TNAC-triggered withdrawal.
- **Emergency release trigger**: Emergency release requires *both* an absolute threshold breach (≥ 85% of penalty rate or ≥ 300 EUR/t) *and* a MA3 price spike > 2.5× the prior year's MA3.

**MSR cancellation mechanism (EU ETS post-2023 reform):**
At the start of each year, MSR holdings exceeding the previous year's auction volume are permanently cancelled:
```
excess = max(0, msr_reserve - prev_auction_volume)
msr_reserve -= excess
total_cancelled += excess
```

### 3.3 Reserve price mode

Auction reserve can be:
- static (`reserve_price`), or
- dynamic (`max(absolute_floor, discount * MA3_price)`) with fallback behavior when no valid clears exist.

## 4. Participants and Company Model

### 4.1 Agent population

- **Learning agents A1–A8:** PPO/HAPPO-trained.
- **Bot agents B1–B8 (optional):** heuristic policy for both auction and secondary market.

Default configuration uses no bots (`n_bot_agents=0`).

Agents are arranged as 4 archetype pairs, each consisting of one pure-financial and one ESG-balanced agent:

| Agent | Green fraction | Emission factor   | w_cost | w_green | Type            |
|-------|---------------|-------------------|--------|---------|-----------------|
| A1    | 45%           | ≈0.355 tCO2/MWh   | 1.0    | 0.0     | Pure financial  |
| A2    | 45%           | ≈0.355 tCO2/MWh   | 0.5    | 0.5     | ESG-balanced    |
| A3    | 50%           | ≈0.299 tCO2/MWh   | 1.0    | 0.0     | Pure financial  |
| A4    | 50%           | ≈0.299 tCO2/MWh   | 0.5    | 0.5     | ESG-balanced    |
| A5    | 65%           | ≈0.210 tCO2/MWh   | 1.0    | 0.0     | Pure financial  |
| A6    | 65%           | ≈0.210 tCO2/MWh   | 0.5    | 0.5     | ESG-balanced    |
| A7    | 75%           | ≈0.143 tCO2/MWh   | 1.0    | 0.0     | Pure financial  |
| A8    | 75%           | ≈0.143 tCO2/MWh   | 0.5    | 0.5     | ESG-balanced    |

All participants produce 10 TWh/year. Bot agents (when active) follow the same 4-archetype structure but use heavier fossil mixes (40–55% coal+gas). Bot behavior supports `enhanced_noise` (higher valuation/urgency variance and optional budget-stress quantity cuts) and a `fade_schedule` for episode-based retirement.

### 4.2 Technology-resolved generation mix

Each company has a 5-technology portfolio:
- coal
- gas
- onshore wind
- offshore wind
- solar

The model tracks technology-specific:
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

**Construction jitter:** Each investment project experiences Poisson-sampled additional delay beyond the base deploy delay. Per-technology mean extra delay and capacity-factor uncertainty:

| Technology    | λ (Poisson extra years) | cf_sigma |
|--------------|------------------------|----------|
| Coal         | 1.0                    | 0.00     |
| Gas          | 1.0                    | 0.00     |
| Onshore wind | 2.0                    | 0.08     |
| Offshore wind| 3.0                    | 0.08     |
| Solar        | 1.5                    | 0.05     |

Cancellation probability is `p_cancel=0.03` per year in the construction queue. If cancelled, the agent recovers `recovery_rate=0.40` (40%) of sunk capex. Capacity-factor noise (`cf_sigma`) applies each operating year, randomizing realized output and emissions.

### 4.4 Demand and Emission Uncertainty

With `uncertainty.enabled=true`, each agent receives a stochastic demand/emission shock each year. Shocks are drawn from a correlated multivariate normal distribution:

- Per-agent shock standard deviation: `sigma_demand=0.07` (7% of expected emissions).
- Cross-agent correlation: `corr_rho=0.40` — shocks are positively correlated, simulating system-wide demand fluctuations (weather, economic cycles) while preserving agent-level heterogeneity.

The shock multiplies each agent's realized emissions that year. Combined with capacity-factor noise from construction jitter, agents face uncertain compliance needs even when holding the expected number of allowances.

### 4.5 Compliance and carry-forward

Compliance is settled against realized emissions plus prior carry-forward obligation:

$$
shortfall_i = \max(0, E_i^{realized} + CF_i^{old} - A_i^{held})
$$

Penalty paid:

$$
penalty_i = shortfall_i \cdot effectivePenaltyRate_t
$$

Shortfall carries forward to the next year (`carry_forward: true`). With `carry_forward_cap: 0.0` (no cap), agents that consistently under-purchase face compounding obligations.

### 4.6 MAC fuel-switching

If carbon price exceeds the MAC threshold (`coal_to_gas_cost=48 EUR/t`), a company can temporarily switch up to `max_switch_frac=20%` of coal dispatch to gas. This lowers emissions in-year but adds MAC cost. It does not permanently alter the long-run technology mix.

### 4.7 Revenue-based dynamic budget

With `budget.mode = revenue_based`, annual budgets are computed dynamically from electricity revenue:

$$
\text{budget}_t = \text{EMA}\bigl(\text{revenue}_t - \text{opex}_t + \text{debt\_headroom}_i,\;\alpha\bigr)
$$

where `revenue` comes from `Company.compute_revenue()` (electricity sales with carbon-cost passthrough using MA3-smoothed carbon price and system-average emission factor), and `debt_headroom` is an archetype-specific buffer. EMA smoothing (`ema_alpha=0.3`) prevents erratic year-to-year budget swings.

### 4.8 Emergency loan system

When agents face auction default, an emergency loan is issued instead of immediate suspension:

- **Trigger**: Shortfall at auction settlement exceeds remaining budget but falls within `max_loan_fraction × annual_budget` (15%).
- **Mechanics**: `apply_emergency_loan(shortfall)` adds the shortfall plus accrued interest (`loan_interest_rate=8%`) to `_loan_outstanding`. Annual repayment is deducted at year start via `apply_loan_repayment()`.
- **Tracking**: `_loan_outstanding`, `_loan_repayment_annual`, `_years_under_loan` are maintained on the `Company` object and exposed in observations (see §6).
- **Heuristic loan-awareness**: Bots with outstanding loans reduce auction quantity (−30%), investment (−50%), and secondary buy volume (−40%).

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

Base dimension: **36**.

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
- `[33]` last_cover_ratio
- `[34]` own_last_secondary_buy_price (WTP anchor)
- `[35]` treasury_norm: treasury_reserve / annual_budget (clipped [0,1])

With opponent modeling enabled:

$$
obsDimPhase1 = 36 + 7 (N_{total} - 1)
$$

Each opponent contributes a **7D lagged tuple** (year t−1 snapshot, read from `_opponent_snapshots_prev`):

| Dim | Signal | Normalisation |
|-----|--------|---------------|
| 0 | `verified_emissions` | `/ 10.0` |
| 1 | `green_frac` | raw [0,1] |
| 2 | `fossil_frac` | raw [0,1] |
| 3 | `queue_signal` + N(0,σ) | clipped [0,1] |
| 4 | `bank_norm` = holdings / annual_need | clipped [0,3] / 3 |
| 5 | `net_secondary_norm` = (sec_bought − sec_sold) / annual_need | clipped [−1,1] |
| 6 | `lagged_compliance_gap_norm` = prior (emissions − surrendered) / annual_need | clipped [−1,1] |

**Timing:** Phase 1 reads `_opponent_snapshots_prev` (year t−1). At end of `step_secondary()`, current-year data is written into `_opponent_snapshots`. At the start of the next year's Phase 1, `_opponent_snapshots_prev` holds year t data.

```
Year t Phase 1 obs reads _opponent_snapshots_prev (year t−1).
End of year t step_secondary() saves prev, then writes new current snapshot.
Year t+1 Phase 1 obs reads updated _opponent_snapshots_prev (year t).
```

Default (`N_total=8`): phase 1 dimension = `36 + 7 × 7 = 85`.

### 6.2 Phase 2 observation

Phase 2 appends **11** auction-result and compliance-awareness features to phase 1:
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
- compliance_gap_norm: signed (realized_emissions − surrendered) / annual_need (clipped [−1,1])

$$
obsDimPhase2 = obsDimPhase1 + 11
$$

Default (`N_total=8`): phase 2 dimension = `85 + 11 = 96`.

## 7. Reward Design

**Reward channel logging:** After each year, `_last_reward_channels` and `_last_auction_reward_channels` dicts are populated with named components for debugging and analysis.

### 7.1 Core reward formulation

Per-agent reward:

$$
R_i = w_{cost,i}(-\text{costNorm}_i) + w_{green,i}(\text{esgSignal}_i)
  - \text{penaltyNorm}_i
  + \text{shapingTerms}_i
  + \text{terminalValues}_i
$$

Where:
- `costNorm` is total non-penalty cost scaled by `REWARD_SCALE = 1000`. Costs include auction, secondary, investment, OPEX delta, soft budget penalty, soft capex penalty, MAC cost, collateral cost, and green-loan interest.
- `penaltyNorm` is compliance penalty at full strength, scaled by `REWARD_SCALE`.
- `esgSignal` is the ESG bonus (see below).
- `shapingTerms` are active shaping channels (see §7.2), decaying to zero over training.

Objective weight structure:
- Financial agents (`w_cost=1.0`, `w_green=0.0`): optimize pure cost.
- ESG-balanced agents (`w_cost=0.5`, `w_green=0.5`): trade off cost and ESG signal.

**ESG signal formula:**

$$
esg_i = w_{green,i} \cdot esg\_scale \cdot \frac{ef_{0,i} - ef_{t,i}}{ef_{0,i}} \cdot \frac{t}{T} \cdot coverage\_frac_i^2
$$

where `ef_ratio` is the agent's emission factor improvement since episode start, `time_ratio` is elapsed episode fraction, and `coverage_frac²` is the ESG compliance gate — non-compliant agents (low coverage) receive a suppressed ESG bonus. `esg_scale=2.0`.

**Terminal values** in final year:
- Bank terminal value with diminishing returns:

$$
V^{bank}_i = \log\left(1 + \frac{B_i}{\max(\hat{E}_i, 0.1)}\right)
\cdot \hat{E}_i \cdot \frac{P_T}{\text{REWARD\_SCALE}}
$$

where $B_i$ is banked allowances, $\hat{E}_i$ is annual estimated need, and $P_T$ is the terminal price anchor.

- Queue terminal value (discounted future emissions savings from queued projects) with a completion-fraction discount to prevent end-of-episode gaming of long-lead projects.

Terminal price anchor uses `max(auction_clearing, secondary_clearing, 80% of inflation-adjusted penalty rate)`.

### 7.2 Shaping channels

Two shaping channels are active during training; both decay toward zero so the equilibrium reward is pure:

**Opportunity cost shaping** (`opportunity_cost_shaping.enabled=true`, `scale=1.0`): Rewards agents in proportion to the cost premium paid on secondary market purchases relative to the auction price. This creates an early-training incentive to win enough allowances at auction rather than overpaying on the secondary market.

**Coverage gap shaping** (`coverage_gap_shaping.enabled=true`, `scale=0.5`): Provides an immediate negative signal when an agent's auction allocation falls short of its compliance need. This bootstraps compliance-seeking behavior before the agent has accumulated enough penalty experiences to learn from them.

Both shaping terms decay with the `shaping_weight` schedule (auto: 40% of n_episodes, floor=0.0).

### 7.3 Diagnostic Scores

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

Actors are decentralized; the critic is centralized (MAPPO mode) over concatenated multi-agent state with `critic_hidden_size=512`.

### 8.2 HAPPO/PPO training

- On-policy episode rollouts.
- Sequential policy updates (HAPPO enabled).
- GAE + clipped PPO objective.
- KL early-stopping; KL anchor to frozen BC policy disabled by default (`kl_anchor_beta=0.0`).

### 8.3 Stabilization features

**Fundamental price anchor** (`src/utils/price_anchor.py`): `compute_fundamental_anchor(year, config)` returns the economically grounded expected clearing price derived from MAC cost, cap scarcity (LRF-based linear approximation), and effective penalty rate:

$$
\text{anchor}_t = \underbrace{MAC \times \text{mult}}_{\text{banking premium}} + \text{scarcity}_t \times (\text{eff\_penalty}_t - MAC \times \text{mult})
$$

where $\text{scarcity}_t = 1 - cap_t / cap_0$, ranging from 0 at yr0 to ~0.4 at yr11. Default output: ~67 EUR/t at yr0, ~101 EUR/t at yr11. Used as: (1) initial `price_head.bias` via `PPOAgent.inject_fundamental_anchor(year)`, and (2) AR(1) mean-reversion floor in `ETSEnvironment`.

**Warm-start:** Before visible year 0, `n_burnin_years=4` hidden years run to initialize market state:
- Agent banks are seeded to 15–35% of annual allowance need (`bank_seed_min=0.15`, `bank_seed_max=0.35`).
- Construction queues are pre-populated with Poisson-distributed projects: `μ_onshore=1.5`, `μ_offshore=0.5`, `μ_solar=2.0`.
- MA3 price history is initialized at `mean=70 EUR/t, std=10`.

This avoids cold-start artifacts where agents begin with zero banks and empty queues, which creates unrealistic year-0 compliance pressure and distorts early-episode learning.

**Private urgency scalars** (`urgency_scalars.enabled=true`): Each agent draws a per-episode LogNormal(0, σ=0.30) multiplier applied to its effective penalty rate. This creates private heterogeneous compliance pressure — two agents with identical holdings face different effective penalties in the same year. Scalars are re-drawn each episode, destroying the symmetric cost structure that sustains floor-bidding equilibria.

**Epsilon-greedy exploration** (`mode="uniform"`): With probability ε (decaying from 0.25 to 0.05 over 75% of training), bid prices are sampled from Uniform([price_min, price_max]) rather than from the policy distribution. This prevents early lock-in to floor prices by ensuring exploration covers the full price range.

**Historical Policy Pool (HPP):** Anti-regression mechanism maintaining a pool of 10 past actor snapshots. Each episode, each agent is independently swapped to a historical snapshot with probability 0.20. This ensures agents always face a diverse opponent distribution, preventing coordination on degenerate equilibria. `seed_heuristic=false` — BC-seeding of the pool is disabled since behavioral cloning pretraining is off by default.

**Entropy decay schedule:** Entropy coefficient decays from 0.08 to 0.025 over training to shift from exploration to exploitation.

**Reward normalization:** Per-agent EMA-based reward normalizer with `gae_min_std=0.1` prevents degenerate advantage estimates.

## 9. Economic and Financial Layers

### 9.1 Inflation path

Inflation is sampled per episode (shared by all companies) and compounds annually.
It scales penalty, capex, OPEX, decommissioning, MAC, and power price base.

### 9.2 Electricity revenue channel

Electricity revenue is enabled (`electricity.enabled=true`):

$$
P_{elec} = P_{base} + \text{passthrough} \cdot P_{carbon} \cdot EF_{system}
$$

With `base_price=55 EUR/MWh` and `carbon_passthrough=0.90`, carbon costs are substantially passed through to electricity prices, creating a revenue stream that partially offsets compliance costs for all generators. This links carbon price levels to agent profitability, making compliance cost management strategically important.

### 9.3 Unified budget envelope

Each company has an annual spending envelope for all major outlays.
A separate capex throughput constraint models physical delivery bottlenecks.

### 9.4 Budget Hardening Regime

Annual spending is subject to a tiered penalty regime:
- **Below 100%** (`soft_zone_start`): No penalty.
- **100–115%** (`soft_zone_start` → `hard_cap_fraction`): Quadratic penalty
  that scales with overshoot amount: `coef × (normalized²) × overshoot_abs`.
- **Above 115%**: Penalty continues to grow steeply (normalized > 1).
- **Investment hard gate**: When enabled, `step_auction()` scales down
  investment fraction if total projected spending would exceed the hard cap.

## 10. Current Simplifications

The model remains stylized despite expanded realism:
- Single annual bid per participant (no multi-step bid curve).
- Simplified secondary market matching (no full limit-order book dynamics).
- Fixed annual output per company (10 TWh).
- Closed system (no cross-market linkage or imports).
- AR(1)-style expected price signal instead of full forward curve equilibrium.

These are deliberate to keep MARL tractable while preserving key strategic channels.

## 11. Equilibrium-Breaking Mechanisms

Several mechanisms work together to prevent degenerate floor-bidding equilibria, where all agents bid at the reserve price and the clearing price never rises above the floor.

| Mechanism | What it does | Config |
|---|---|---|
| **Private Urgency Scalars** | Per-episode LogNormal(0, 0.30) penalty multiplier per agent; destroys symmetric cost structure, making floor bids risky for some agents even when others can afford them | `urgency_scalars.enabled: true` |
| **ESG Compliance Gate** | `esg_signal × coverage_frac²` — non-compliant agents lose ESG bonus, coupling compliance to the ESG reward stream for balanced agents | `esg.enabled: true` |
| **Epsilon-greedy Uniform Exploration** | Early-training price bids sampled from Uniform([price_min, price_max]); seeds diverse price history in MA3 and HPP pool | `exploration.mode: "uniform"` |
| **Historical Policy Pool (HPP)** | Periodic snapshots of past policies; random opponent swap each episode maintains diverse bidding history as opponents | `hpp.enabled: true` |
| **Coverage Gap Shaping** | Immediate reward signal for compliance shortfall in early training; decays to zero at equilibrium | `coverage_gap_shaping.enabled: true` |

These mechanisms are complementary: urgency scalars break symmetric compliance costs, the ESG gate creates compliance-reward coupling for balanced agents, and exploration plus HPP seed diverse price histories that the MA3 anchor preserves.

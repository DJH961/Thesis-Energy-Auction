# Design Document - ETS MARL (Current v6.0)

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

Annual cap follows an LRF schedule:

$$
cap_{t+1} = cap_t (1 - LRF_t)
$$

- Phase 1 LRF: 4.3%
- Phase 2 LRF: 4.4%

### 3.2 MSR logic

MSR operates on auction volume (not cap) using TNAC proxy (sum of all banks):
- If TNAC > upper threshold: withhold share of excess into reserve.
- If TNAC < lower threshold: release fixed volume from reserve.

Current implementation also includes price-responsive safeguards:
- If price ratio reaches containment trigger: suppress normal withdrawal.
- If price ratio reaches emergency trigger: force emergency release.

This avoids a procyclical loop where high prices and high TNAC jointly reduce supply further.

### 3.3 Reserve price mode

Auction reserve can be:
- static (`reserve_price`), or
- dynamic (`max(absolute_floor, discount * MA3_price)`) with fallback behavior when no valid clears exist.

## 4. Participants and Company Model

### 4.1 Agent population

- Learning agents A1-A8: PPO/HAPPO-trained.
- Bot agents B1-B8: heuristic policy for both auction and secondary market.

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

Base dimension: 23.

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
obsDimPhase1 = 23 + 5 (N_{total} - 1)
$$

Each opponent contributes public 5D tuple:
- normalized emissions
- normalized carry-forward
- green fraction
- fossil fraction
- total queue size

With 16 total participants:
- phase 1 dimension = 98

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
- phase 2 dimension = 105

## 7. Reward Design (Current)

Per-agent reward is:

$$
R_i = -\text{costNorm}_i + \text{greenBonus}_i + \text{queueBonus}_i + \text{esgSignal}_i + \text{terminalValues}_i
$$

Where:
- `costNorm` is net cost after electricity revenue, scaled.
- Costs include auction, secondary, investment, OPEX, budget penalties, capex throughput penalties, MAC cost, and compliance penalty.
- `greenBonus` rewards positive green share change with shaping decay over training.
- `queueBonus` rewards maintaining active construction pipeline.
- `esgSignal` uses saved-carbon-years style term, gated by ESG weight.

Terminal values in final year (configurable):
- bank terminal value (bank * terminal price)
- queue terminal value (discounted future emissions savings from queued projects)

Terminal price anchor uses max of:
- auction clearing
- secondary clearing
- 80% of inflation-adjusted penalty rate

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
- Reward normalization per agent.
- Entropy decay schedule.
- Epsilon-greedy exploration in physical action space with anchored Gaussian sampling.
- Historical Policy Pool (periodic snapshots and random swaps for opponent diversity).
- Diagnostics for stuck-market or degenerate-policy regimes.

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

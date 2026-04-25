# Design Document — ETS MARL

## 1. Auction Mechanism

### Why uniform-price sealed-bid?
The EU ETS power sector uses a single-round, sealed-bid, uniform-price auction
(Commission Regulation 1031/2010). Every winner pays the same clearing price —
the marginal accepted bid. This is the opposite of the energy seller market in
the base repo (ckrk/bidding_learning), where producers bid minimum acceptable
prices ascending. Here companies are **buyers** bidding maximum willingness-to-pay,
sorted descending.

### Clearing algorithm (adapted from ckrk/bidding_learning)
1. Filter bids below reserve price.
2. Sort descending by bid price (tie-break: ascending agent_id for determinism).
3. Walk sorted bids, accumulating quantity until Q_cap is reached.
4. Clearing price = price of the last accepted bid (marginal buyer).
5. Tie-break at marginal price: pro-rata allocation by bid quantity.
6. All winners pay the uniform clearing price.

## 2. Cap Trajectory

### LRF
`cap(t+1) = cap(t) * (1 - LRF_t)`
- LRF = 4.3% for years 1–4 (EU ETS 2024–2027)
- LRF = 4.4% for years 5+ (EU ETS 2028+)

### MSR
The MSR adjusts the **auction volume** (not the cap) based on TNAC:
- TNAC > 8.33 Mt: withhold 24% of excess into MSR reserve
- TNAC < 4.00 Mt: release 0.10 Mt/year from reserve into auction
- TNAC in [4.00, 8.33]: no adjustment

TNAC proxy: sum of all company banks. This approximates the EU definition
(issued allowances minus surrendered allowances) under the simplifying
assumption that all non-surrendered allowances are banked.

## 3. Company State

### Emission factor choice
All fossil generation modelled as gas CC (EF = 0.37 tCO₂/MWh, IPCC AR5 median).
Rationale: EU power mix 2024 is predominantly gas as marginal fossil fuel.
Coal is effectively being phased out. This is conservative but defensible.
Extension: differentiate coal/gas within fossil% with separate EFs.

### Investment dynamics
- Decision: continuous delta_green ∈ [0, max_delta_green] per year
- Cost: convex, `C = a*Δg + b*Δg²` (M€), with a=50, b=10
- Delay: 2 years (permitting + construction)
- Stochastic failure: `p_fail = base_fail * fossil_frac * (1 - success_rate)`
  - Brown companies fail more often (less green know-how)
  - Success rate updated as rolling 10-year average of past investments

### Risk factor
`risk_factor = base_risk * fossil_frac * (1 - success_rate)`

A company with risk_factor=0.45 bids for 45% more allowances than its estimated
need, as a precautionary buffer against investment failures. As a company greens
and builds success history, this buffer shrinks.

## 4. Reward Function

`reward_i = -(w_cost_i * total_cost_i + w_green_i * fossil_frac_i * 100)`

- `total_cost_i` = auction payments + |trading costs| + penalties + switching costs (all M€)
- `fossil_frac_i` ∈ [0,1], scaled by 100 to bring to comparable magnitude
- Weights encode company orientation:
  - A1 (90% fossil): cares mostly about cost (w_cost=0.80)
  - A4 (10% fossil): cares mostly about greening (w_green=0.80)

## 5. Multi-Agent Setup

Each company is an **independent DDPG agent** with its own:
- Actor network: obs (8D) → action (4D)
- Critic network: (obs, action) → Q-value
- Replay buffer: 50,000 transitions
- OU noise for exploration

Independent learning (no parameter sharing, no communication) is the baseline
following Wang et al. (2024) and ckrk/bidding_learning. This allows emergent
strategic behaviour without coordination assumptions.

## 6. Observation Space (per agent, 8D)

| Index | Feature | Normalisation |
|-------|---------|--------------|
| 0 | year / 10 | [0, 1] |
| 1 | cap_t / output_twh | [0, ~1.5] |
| 2 | last_clearing_price / 200 | [0, 1] |
| 3 | green_frac | [0, 1] |
| 4 | bank / output_twh | [0, ~1] |
| 5 | estimate_need / output_twh | [0, ~2] |
| 6 | success_rate | [0, 1] |
| 7 | risk_factor | [0, ~0.5] |

## 7. Action Space (per agent, 4D)

| Index | Feature | Range |
|-------|---------|-------|
| 0 | price_bid | [0, 200] €/t |
| 1 | quantity_bid | [0, 5.0] Mt |
| 2 | delta_green | [0, 0.03] p.p./year |
| 3 | trade_qty | [-5.0, 5.0] Mt (buy/sell) |

## 8. Known Simplifications and Future Extensions

1. **Single bid per company per auction** — real ETS allows step-function bids.
   Extension: K-step bid curve (as in Di Persio et al., I=3 steps).

2. **Secondary market** — simple bilateral clearing at clearing_price ± ε.
   Extension: order book with price discovery.

3. **Homogeneous output** — all companies produce 10 TWh/year.
   Extension: stochastic output, capacity expansion decisions.

4. **No banking limits** — EU ETS has no quantitative banking cap.
   This is correctly modelled (unlimited banking).

5. **No international linkage** — closed micro-ETS.

6. **Carbon price expectation** — AR(1) process, not forward-looking.
   Extension: rational expectations, futures market.

# ETS MARL — Agent Observation Space Dictionary

Each learning agent receives a continuous observation vector in **two
phases per year**, mirroring the two-phase action layout. Phase 1 is
read **before** the primary auction clears (input to bid + investment
decisions); Phase 2 is read **after** the auction clears (input to the
secondary-market trade decision).

Both observations are pre-normalised — every dimension is engineered to
sit roughly in `[-1, 1]` (or `[0, 1]` / `[0, 3]` for ratio-style
features) so the policy network does not need to learn its own input
scaling. The construction code lives in
`src/environment/company.py:841–1099` (`get_observation_phase1`,
`get_observation_phase2`, `obs_dim_phase1`, `obs_dim_phase2`); opponent
features are assembled in
`src/environment/ets_environment.py:2760–2781` and read at obs-build
time from `src/environment/ets_environment.py:3578–3615`.

---

## Phase 1 — Pre-auction (44 base dims + opponent block)

Submitted at the start of every year, **before** the primary auction
clears. Drives the 6-D Phase 1 action (`bid_price`, `qty_mult`,
`invest_frac`, three tech logits — see `docs/action_space.md`).

### Base 44 dims

`pn` below denotes the price normalisation constant `price_max`
(`auction.price_max`, default `250.0` €/tCO₂).

| dim | Name | Raw range | Normalisation | Semantics |
|---:|---|---|---|---|
| 0  | `time`                          | year ∈ {0…n_years−1}        | `/ 12.0`                | Episode progress (constant denominator; not `n_years` to keep ablations comparable). |
| 1  | `cap_norm`                      | Mt                          | `/ 30.0`                | Year-`t` allowance cap, scaled for an 8-agent baseline. |
| 2  | `price_ma3_norm`                | €/tCO₂                      | `/ pn`                  | 3-year moving average of clearing price (falls back to last clearing if MA3 unavailable). |
| 3  | `expected_price_norm`           | €/tCO₂                      | `/ pn`                  | AR(1) one-step forecast of clearing price. |
| 4–8 | `tech_mix[5]`                  | fraction                    | raw [0,1]               | Generation-share vector `[coal, gas, onshore, offshore, solar]`. |
| 9  | `emissions_norm`                | Mt                          | `/ 10.0`                | Current-year baseline emissions (pre-shock). |
| 10 | `estimate_need_norm`            | Mt                          | `/ 10.0`                | Deterministic compliance need (no risk buffer) — same quantity that scales `qty_mult`. |
| 11 | `p_fail`                        | [0, 1]                      | raw                     | Risk factor / failure probability from `compute_risk_factor()`. |
| 12 | `invest_experience`             | successes                   | `/ 5.0`                 | Consecutive successful investment cycles (skill proxy). |
| 13 | `auction_gap_norm`              | Mt                          | `/ 5.0`                 | Last year's banked-allowances gap = supply − demand at clearing. |
| 14 | `queue_onshore`                 | fraction of capacity        | raw [0,1]               | Onshore-wind plant fraction currently under construction. |
| 15 | `queue_offshore`                | fraction of capacity        | raw [0,1]               | Offshore-wind plant fraction currently under construction. |
| 16 | `queue_solar`                   | fraction of capacity        | raw [0,1]               | Solar plant fraction currently under construction. |
| 17 | `weighted_emission_factor`      | tCO₂/MWh                    | normalised in property  | Capacity-weighted average EF of current generation mix. |
| 18 | `last_sec_price_norm`           | €/tCO₂                      | `/ pn`                  | Last year's secondary-market VWAP. |
| 19 | `last_sec_volume_norm`          | Mt                          | `/ 10.0`                | Last year's secondary-market traded volume. |
| 20 | `carry_forward_norm`            | Mt                          | `/ 5.0`                 | Outstanding compliance debt rolled into year `t`. |
| 21 | `tnac_proxy`                    | TNAC / cap                  | clipped [0, 3]          | Total banked allowances ÷ cap — drives MSR withholding. |
| 22 | `effective_reserve_norm`        | €/tCO₂                      | `/ pn`                  | Auction reserve price for year `t` (post inflation / PCL). |
| 23 | `auction_volume_ratio`          | Mt / Mt                     | `/ cap_t`               | This year's MSR-adjusted auction supply ÷ cap (preview). |
| 24 | `msr_reserve_norm`              | Mt / Mt                     | `/ cap_t`               | Allowances currently parked in the MSR. |
| 25 | `own_bank_ratio`                | bank / own_need             | clipped [0, 5] / 5      | Own banked allowances expressed as multiple of annual need. |
| 26 | `predicted_withhold`            | Mt / cap                    | clipped [0, 1]          | Anticipated MSR withhold share of next year's cap. |
| 27 | `budget_headroom`               | 1 − spent / annual_budget   | clipped [-0.5, 1.0]     | 1.0 = fresh budget, 0 = at limit, negative = overspent. |
| 28 | `collateral_load_last`          | locked / annual_budget      | clipped [0, 1]          | Last year's collateral lock (overbid-risk signal). |
| 29 | `bid_affordability_last`        | bid_total / budget_remaining| clipped [0, 1]          | Last year's bid notional ÷ remaining budget. |
| 30 | `loan_outstanding_norm`         | loan / annual_budget        | clipped via property    | Emergency-loan principal outstanding. |
| 31 | `years_under_loan_norm`         | years / n_years             | raw                     | Remaining loan-repayment years. |
| 32 | `last_cover_ratio`              | supply / demand             | clipped [0, 3] / 3      | Last year's auction cover ratio (low → high competition → bid higher). |
| 33 | `own_last_sec_buy_price`        | €/tCO₂                      | clipped [0, pn] / pn    | Own last secondary-buy price (WTP anchor). |
| 34 | `cumulative_coverage_ratio`     | Σ alloc / Σ emissions       | clipped [0, 2] / 2      | Long-run compliance signal (<1 → persistently under-buying). |
| 35 | `treasury_norm`                 | treasury / annual_budget    | clipped [0, 2] / 2      | Treasury reserve relative to annual budget. |
| 36 | `cap_ahead_3y_ratio`            | cap(t+3) / cap(t)           | clipped [0, 1]          | 3-year scarcity lookahead. |
| 37 | `cap_ahead_6y_ratio`            | cap(t+6) / cap(t)           | clipped [0, 1]          | 6-year scarcity lookahead. |
| 38 | `pcl_headroom_norm`             | (pcl_ceiling − MA3) / pn    | clipped [0, 1]          | Headroom to bid-change-limit ceiling (1.0 = unconstrained). |
| 39 | `last_bid_price_clip`           | (actual − requested) / pn   | clipped [-1, 1] (signed)| PCL clip only; negative = bid clipped down by bid-change limit. |
| 40 | `last_budget_price_clip`        | (actual − requested) / pn   | clipped [-1, 1] (signed)| Budget-gate clip only; negative = clipped down by joint budget gate. |
| 41 | `last_bid_qty_clip_ratio`       | actual_q / requested_q      | clipped [0, 1]          | 1.0 = no qty gate; <1 = leverage / collateral / budget gate fired. |
| 42 | `last_invest_clip_ratio`        | actual / requested          | clipped [0, 1]          | 1.0 = no cap; <1 = capex hard gate fired. |
| 43 | `compliance_affordability`      | need × E[clearing] / cash   | clipped [0, 3] / 3      | Forward-looking budget anchor: 0 ≈ trivially affordable, 0.33 (raw 1.0) = compliance consumes all cash, 1.0 (raw 3.0) = unaffordable from cash alone. `expected_clearing = max(reserve, MA3, anchor)` — same formula the joint budget gate uses. Replaces purely lagged clip-feedback so agents don't need to learn affordability through repeated clip events. |

### Opponent block (optional, `7 × (N_total − 1)` dims)

Appended only when `opponent_modeling.enabled = true`. Each opponent
contributes a **7D lagged tuple** (year `t−1` snapshot, read from
`_opponent_snapshots_prev`):

| dim | Name | Raw range | Normalisation | Semantics |
|---:|---|---|---|---|
| 0 | `verified_emissions`            | Mt                          | `/ 10.0`                | Public emissions report. |
| 1 | `green_frac`                    | fraction                    | raw [0, 1]              | Renewable share of opponent's mix. |
| 2 | `fossil_frac`                   | fraction                    | raw [0, 1]              | Fossil share of opponent's mix. |
| 3 | `queue_noisy`                   | Σ frac_delta + 𝒩(0, σ)      | clipped [0, 1]          | Queue signal with `opponent_obs.queue_noise_sigma` jitter (default 0.15). |
| 4 | `tnac_share_norm`               | own_holdings / Σ holdings   | clipped [0, 1]          | Share of total banked allowances (respects EU ETS confidentiality). |
| 5 | `net_secondary_norm`            | (bought − sold) / own_need  | clipped [-1, 1]         | Last year's net secondary-market position. |
| 6 | `lagged_compliance_gap_norm`    | (emissions − surrendered) / own_need | clipped [-1, 1] | Prior-year compliance gap (signed). |

`opponent_obs.mode` switches between `"lagged"` (default — reads
`_opponent_snapshots_prev`, year `t−1`) and `"full_info"` (reads each
opponent's current `get_public_info()` directly — used as an oracle
upper-bound in ablations).

**Timing:** Phase 1 in year `t` reads `_opponent_snapshots_prev`, which
holds the year `t−1` snapshot. At the end of `step_secondary()` the
current-year snapshot is committed: `_opponent_snapshots_prev` ← old
`_opponent_snapshots`, then `_opponent_snapshots` ← year-`t` data.

### Total Phase 1 dimension

```
obs_dim_phase1 = 44 + opponent_obs.dims_per_opponent × (N_total − 1)
              = 44                                  if opponent_modeling disabled
              = 44 + 7 × (N_total − 1)              if opponent_modeling enabled (default 7)
```

Default (`N_total = 8`, opponent modeling on): `44 + 7 × 7 = 93`.

---

## Phase 2 — Post-auction (12 extra dims)

Read **after** the primary auction has cleared and the agent has
observed its allocation. Drives the 2-D Phase 2 action
(`sec_price_abs`, `sec_qty` — see `docs/action_space.md`).

Phase 2 = Phase 1 vector concatenated with **12** auction-result and
compliance-awareness features:

| dim (rel.) | Name | Raw range | Normalisation | Semantics |
|---:|---|---|---|---|
| +0 | `allocation_norm`              | Mt                          | `/ 5.0`                 | Allowances won at the primary auction. |
| +1 | `clearing_price_norm`          | €/tCO₂                      | `/ pn`                  | Realised auction clearing price. |
| +2 | `net_compliance_position`      | Mt                          | `/ 5.0`                 | `bank + allocation − emissions − carry_forward`; <0 → still short after using all holdings. |
| +3 | `emission_shock`               | fractional deviation        | raw                     | Realised emission shock vs. baseline (revealed only in Phase 2). |
| +4 | `auction_savings`              | M€                          | `/ 1000`                | `(allocation × 100 − payment) / 1000` — penalty value avoided minus cost paid. |
| +5 | `coverage_ratio`               | (bank + alloc) / obligation | clipped [0, 3] / 3      | How well-covered the agent is for compliance this year. |
| +6 | `carry_forward_norm`           | carry / estimate_need       | clipped [0, 3] / 3      | Outstanding debt relative to annual need. |
| +7 | `collateral_locked_norm`       | locked / annual_budget      | clipped [0, 1]          | This year's collateral lock (immediate over-commitment feedback). |
| +8 | `budget_remaining_phase2_norm` | (budget − spent) / budget   | clipped [-0.5, 1.0]     | Post-auction budget headroom. |
| +9 | `compliance_liability_norm`    | shortfall × penalty / budget| clipped [0, 2.0]        | Penalty exposure if no further allowances secured. |
| +10| `compliance_gap_norm`          | (emis + carry − holdings) / need | clipped [-2, 2] / 2 | Signed gap; >0 = short, <0 = surplus. |
| +11| `last_sec_qty_clip_ratio`      | actual / requested          | clipped [-1, 1]         | Last year's secondary-market qty clip signal; 1.0 = no constraint, <1 = buy clipped, negative = sell side. |

### Total Phase 2 dimension

```
obs_dim_phase2 = obs_dim_phase1 + 12
```

Default (`N_total = 8`, opponent modeling on): `93 + 12 = 105`.

---

## Bots

Bots are instantiated from the same `Company` class and **see the same
Phase 1 / Phase 2 observation vectors** as learning agents, but they
ignore most of the vector — the rule-based heuristic in
`src/agents/heuristic_policy.py` derives its bid / invest / trade
decisions from a small subset of fundamentals (MAC vs. penalty,
NPV-gated investment, target-bank-trajectory trading).

Only the first `n_agents` companies actually receive observations from
`_get_obs_phase1()`; bot observations are short-circuited inside their
heuristic policy.

---

## Architectural notes

* **Pre-auction information only.** Phase 1 deliberately excludes the
  current year's emission shock and the realised clearing price;
  exposing those would make the pre-auction decision trivially
  optimal. Both surface in Phase 2 (`emission_shock`,
  `clearing_price_norm`).
* **Clip-feedback dims.** Dims 28, 29, 39, 40, 41, 42 (Phase 1) and
  Phase 2 dim +11 are *gradient channels*: they expose how much each
  environment-side gate (PCL / budget / collateral / capex / sec-market
  qty) bit into last year's action. Year 0 always reads as
  unconstrained (no prior year exists).
* **Forward-looking compliance.** Dim 43
  (`compliance_affordability`) is the only Phase 1 dim that uses the
  budget gate's *prospective* expected clearing
  (`max(reserve, MA3, anchor)`) rather than a lagged outcome. It lets
  the policy anticipate affordability before the gate fires.
* **Confidentiality.** Opponent dims expose only signals public under
  EU ETS reporting rules: verified emissions, mix shares, a noisy
  build-queue signal, TNAC market share, and lagged compliance/sec
  positions. No private cash, treasury, or loan state leaks across
  agents.
* **Determinism.** Opponent queue noise uses
  `_opponent_obs_rng` (seeded independently from policy / market RNGs)
  so observation noise is reproducible at fixed seed regardless of
  policy stochasticity.

---

## Where the obs is built in code

```python
# src/environment/company.py:1089
@property
def obs_dim_phase1(self) -> int:
    opp_dims = self.config.get("opponent_obs", {}).get("dims_per_opponent", 7)
    if self._opponent_modeling and self._n_total > 1:
        return 44 + opp_dims * (self._n_total - 1)
    return 44

@property
def obs_dim_phase2(self) -> int:
    return self.obs_dim_phase1 + 12
```

Per-step assembly:

* Phase 1 → `ETSEnvironment._get_obs_phase1()`
  (`src/environment/ets_environment.py:3578–3615`) calls
  `Company.get_observation_phase1(...)` for each learning agent,
  passing the opponent block read from `_opponent_snapshots_prev`.
* Phase 2 → `Company.get_observation_phase2(obs_phase1, ...)`
  concatenates the 12 auction-result dims onto the cached Phase 1
  vector returned at the start of the year.

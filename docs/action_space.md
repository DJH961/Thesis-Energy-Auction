# ETS MARL — Agent Action Space Dictionary

Each learning agent emits a continuous action vector in **two phases per
year**. Both phases use a clipped-Gaussian policy: the network outputs
a per-dimension mean and log-std, samples a `raw ∈ ℝ`, clamps it to
`[-1, 1]`, then linearly rescales the squashed value into the physical
range `[low_d, high_d]` for each dimension.

The bounds below are pulled from `configs/default.yaml` and from the
action-space construction code at `scripts/train.py:80–106`.

---

## Phase 1 — Auction + Investment (6-D)

Submitted at the start of every year, **before** the primary auction
clears.

| dim | Name | Physical range | Source of bounds | Semantics |
|---:|---|---|---|---|
| 0 | `bid_price` | `[auction.price_min, auction.price_max]` €/tCO₂ — default `[45.0, 250.0]` | `auction.price_min`, `auction.price_max` | Sealed-bid limit price submitted into the uniform-price auction. May be further clipped by the **bid-change limit** (year-over-year change capped at `auction.bid_change_limit.value`, anchored on `max(price_ma3, fundamental_anchor)`) and by the **joint budget gate** (cf. `docs/config_dictionary.md` § auction). |
| 1 | `qty_mult` | `[auction.qty_mult_low, auction.qty_mult_high]` — default `[0.5, 2.0]` | `auction.qty_mult_low/high` | Multiplier on the agent's deterministic `estimate_need` (Mt). The submitted bid quantity is `qty_mult × estimate_need`, then capped at `auction.quantity_max`, the agent's per-auction `max_agent_share`, and the budget gate. |
| 2 | `invest_frac` | `[0.0, investment.max_invest_frac]` — default `[0.0, 0.20]` | `investment.max_invest_frac` | Fraction of this year's available budget allocated to renewable investment. After action emission it is hard-clipped by the **budget hard gate** (`budget.investment_hard_gate=true`) which also reads the agent's emergency-loan headroom. |
| 3 | `logit_onshore` | `[-1.0, 1.0]` (raw logit) | hard-coded | Pre-softmax logit for **onshore wind** (deploy delay 4 y). |
| 4 | `logit_offshore` | `[-1.0, 1.0]` (raw logit) | hard-coded | Pre-softmax logit for **offshore wind** (deploy delay 7 y). |
| 5 | `logit_solar` | `[-1.0, 1.0]` (raw logit) | hard-coded | Pre-softmax logit for **solar** (deploy delay 2 y). |

Tech-mix from logits: `softmax([logit_onshore, logit_offshore,
logit_solar])` partitions the year's `invest_frac × budget` across the
three buildable technologies. Only renewables can be added (greening-
only constraint); fossil generation can only decrease via natural
attrition / cancellation.

### Architectural detail (PPO `AuctionPolicy`)

The 6-D mean is produced by **conditioned heads** in
`src/agents/actor_critic.py`:

```
hidden       = MLP(LayerNorm(obs_phase1))                                 # 256-D
price_mean   = price_head(hidden)                                         # 1
qty_mean     = qty_head(concat[hidden, price_mean.detach()])              # 1, depends on price
rest_mean    = rest_head(hidden)                                          # 4
mean         = concat[price_mean, qty_mean, rest_mean]                    # 6
```

The detach on `price_mean` keeps gradients from flowing back through
the price branch when training the qty head — i.e. price is the
primary decision and quantity adapts to it.

`log_std` is a learnable 6-vector clipped to
`[ppo.log_std_min, ppo.log_std_max]` (default `[-3.0, 0.0]`); dim 0
(`bid_price`) is initialised tighter at `log_std = -1.5`.

---

## Phase 2 — Secondary Market (2-D)

Submitted **after** the auction clears and the agent observes its
allocation. The secondary market is a continuous double-auction over
this year's allowance volume.

| dim | Name | Physical range | Source of bounds | Semantics |
|---:|---|---|---|---|
| 0 | `sec_price_abs` | `[trading.sec_price_min, trading.sec_price_max_mult × max_penalty_rate]` €/tCO₂ — default low = `45.0`, high ≈ `2.0 × penalty.rate × (1 + infl)^n_years` | `trading.sec_price_min`, `trading.sec_price_max_mult`, `penalty.rate`, `penalty.inflation_rate`, `simulation.n_years` | Absolute limit price submitted to the secondary market — interpreted as a **buy** ceiling if `sec_qty > 0` and a **sell** floor if `sec_qty < 0`. |
| 1 | `sec_qty` | `[-auction.quantity_max, +auction.quantity_max]` Mt — default `[-3.0, +3.0]` | `auction.quantity_max` | Signed trade intent: positive = buy, negative = sell, |·| is the Mt size. The order is filled against opposing intents that lie within `trading.spread_tolerance`. |

A trade clears when the agent's price lies on the right side of the
counter-party's price within `trading.spread_tolerance` (default 12 %).
Filled volume is settled at the volume-weighted matched price; transaction
cost `trading.transaction_cost` (€/Mt) is debited from each side.

### Architectural detail (PPO `SecondaryPolicy`)

Standard 2-layer MLP (LayerNorm → FC256 → FC256) feeding a single 2-D
mean head. No conditioning between dims; both means come from the same
hidden vector. `log_std` is a learnable 2-vector with the same global
clipping as Phase 1.

---

## Action-space derivation in code

```python
# scripts/train.py:80–106
auction_low  = [price_min,  qty_mult_low,  0.0,                -1, -1, -1]
auction_high = [price_max,  qty_mult_high, max_invest_frac,    +1, +1, +1]

max_penalty   = penalty_rate × (1 + infl_rate)^n_years
secondary_low  = [sec_price_min,                       -quantity_max]
secondary_high = [sec_price_max_mult × max_penalty,    +quantity_max]
```

The bounds are **fixed for the duration of a run** and stored as
non-trainable `action_scale` / `action_bias` buffers in each policy
network so policies are robust to checkpoint reload.

---

## Bots

Bots use the same `Company` class and submit actions in the same 6-D /
2-D layout as learning agents, but their actions come from the
rule-based heuristic in `src/agents/heuristic_policy.py`
(MAC→penalty bidding, NPV-gated investment, target-bank-trajectory
trading) rather than a neural policy. Bot actions are **clipped to the
same physical bounds** as learning-agent actions before entering the
auction / secondary market.

---

## Post-action environment-side modifications

The raw policy output is not necessarily what the auction sees. After
action emission the environment may further constrain each agent's
bid via:

1. **Bid-change limit (`auction.bid_change_limit`)** — clips
   `bid_price` to within `±value` of last year's anchor
   (`max(price_ma3, fundamental_anchor)`).
2. **Joint budget gate (`auction.budget_gate`)** — sizes `bid_q` against
   the cash buffer
   `op_cash + treasury_fraction × treasury (+ optional loan headroom)`
   and the expected settlement
   `(safety_mult × max(reserve, MA3, anchor)) × bid_q + collateral(bid_p)`.
   Two-stage protocol: (1) shrink qty toward `need`; (2) reduce
   `bid_p` toward `max(reserve, MA3_inflated)`; (3) last-resort shrink
   qty below need.
3. **Investment hard gate (`budget.investment_hard_gate`)** — clips
   `invest_frac` so capex throughput plus mandatory compliance spending
   do not exceed `available_budget × hard_cap_fraction`.
4. **Lot rounding (`auction.lot_size`)** — rounds quantities to the
   minimum lot size before clearing.

These are recorded in the year-level log as `bid_qty_mult`,
`invest_frac_pre_clip` / `invest_frac_post_clip`, etc. — see
`docs/data_dictionary.md` § 3.

# Heuristic Bot — Behavioural Reference

This document describes **how the rule-based bot policy acts**. The
environment, action space, observation space, auction mechanics, MSR,
penalties, budget machinery, and reward channels are documented in
`design.md`, `action_space.md`, `observation_space.md`,
`reward_function.md`, and `config_dictionary.md`; nothing in those
references is repeated here. A reader who already knows the environment
should be able to use this file alone to reproduce or report on a
bots-only baseline.

The behaviour is implemented in `src/agents/heuristic_policy.py` and
dispatched by the environment in
`src/environment/ets_environment.py::_generate_bot_auction_actions /
_generate_bot_secondary_actions`. The bot has no policy network, no
critic, no learning step, and no exploration noise; it is a
deterministic function of the public market state, the agent's own
finances, two persistent per-episode random scalars (§5), and an
optional budget-stress flag.

The text below is purely descriptive — it states what the rules do, not
whether they are realistic, well-calibrated, or competitive with the
learning agents.

---

## 1. Inputs the heuristic reads

At each call the bot is passed (or reads from its `Company`):

- **Market state**: 3-year moving average of clearing price `price_ma3`,
  the dynamic reserve price, the inflation factor for the current year,
  the MSR-adjusted auction volume, and the current annual cap.
- **Own finances**: bank, allocation (Phase-2 only), operating cash
  (`annual_budget − budget_spent_this_year`), treasury balance,
  outstanding emergency loan, last year's collateral load, capex
  already spent this year, carry-forward debt.
- **Own physical state**: weighted emission factor, generation mix,
  `output_mwh`, per-tech `emission_factors`, `compute_estimate_need()`.
- **Fixed config**: `mac.coal_to_gas_cost`, `penalty.rate +
  inflation`, `auction.{price_min, price_max, qty_mult_low,
  qty_mult_high}`, `trading.{sec_price_min, sec_price_max_mult}`,
  `investment.{max_invest_frac, discount_rate}`, the technology arrays,
  `reward.terminal_payoff_years`.
- **Per-bot persistent scalars** (drawn once per episode, §5):
  `valuation_noise`, `urgency_multiplier`, `urgency_denom`.

The bot does not read other agents' bids, other agents' holdings, or
any private regulator state.

## 2. Auction phase — `auction_action`

The auction action is `[bid_price, qty_mult, invest_frac, logit_onshore,
logit_offshore, logit_solar]`, all in physical space. Each component is
computed independently.

### 2.1 Market anchor and urgency

The bot first forms two scalars used everywhere downstream:

- `anchor = max(mac.coal_to_gas_cost, price_ma3) + valuation_noise`.
  This is the bot's reference for "what the market is worth right now"
  — the MAC of coal-to-gas switching is treated as a floor, and the
  moving average of clearing prices is the upper reference. The
  per-episode `valuation_noise` (Gaussian) shifts this reference
  uniformly across all decisions for that bot in that episode.
- `coverage_ratio = bank / annual_need`.
- `urgency_raw = max(0, 1 − coverage_ratio / urgency_denom)`. Lower
  bank or higher denom → urgency near 0; bank far below need → urgency
  near 1.
- A **supply-scarcity bump** is added when MSR-adjusted auction volume
  is below 80 % of the annual cap: `+ max(0, 1 − supply_ratio) × 0.3`.
- The result is multiplied by the per-episode `urgency_multiplier` and
  clipped to `[0, 1]`.

### 2.2 Bid price — dual willingness-to-pay ceiling

The bid price is the minimum of two ceilings, then floored at the
reserve.

1. **Economic ceiling** (what the bot is willing to pay):
   `wtp_economic = anchor + urgency × (penalty_rate − anchor)`, then
   capped at `penalty_rate − 1`. At urgency 0 the bot bids at the
   anchor; at urgency 1 it bids just under the inflation-adjusted
   penalty rate. The relationship is linear in urgency.

2. **Budget ceiling** (what the bot can afford to pay per tonne):
   `wtp_budget = max_compliance_share × available_cash / qty_target`,
   with `available_cash = operating_cash + treasury_fraction ×
   treasury` (treasury fraction taken from `auction.budget_gate`;
   emergency loan headroom is excluded). `qty_target` is computed in
   §2.3 below.

3. **Final**: `bid_price = max(min(wtp_economic, wtp_budget), reserve
   + 1)`, then clipped to `[auction.price_min, auction.price_max]`.

The binding ceiling (`"economic"` or `"budget"`) is recorded on the
`Company` for diagnostics.

### 2.3 Quantity multiplier

```
qty_target = annual_need × (1 + 0.1 × urgency)
qty_mult   = clip(qty_target / annual_need,
                  auction.qty_mult_low, auction.qty_mult_high)
```

`annual_need` already includes carry-forward debt, so a bot starting
the year short asks for the full deficit plus a small urgency buffer.
The 10 % multiplier is a fixed safety margin; it does not scale with
the cap, the year, or the agent's mix.

### 2.4 Investment fraction

The bot picks **one** buildable technology per year and computes an
investment fraction against it.

**Technology choice.** Among onshore, offshore, solar, pick the tech
maximising the effective payoff metric

```
score(t) = (remaining_years − deploy_delays[t] + terminal_payoff_years)
           × capacity_factors[t] / capex[t].
```

Techs whose `(remaining_years − deploy_delays[t] + terminal_horizon)`
is non-positive are skipped. The chosen tech is encoded as a hard
one-hot in the three logit slots (`+1` for the winner, `−1` for the
others); the bot never mixes investment across technologies in a
single year.

**Reference NPV.** At a test fraction `frac_test` (0.07 if the bot is
green-tagged, 0.03 otherwise — see §6), the bot estimates avoided
emissions over the project's effective horizon and discounts them as an
annuity at `investment.discount_rate`:

```
ef_saved          = max(0, weighted_emission_factor − ef[best_tech])
effective_horizon = max(0, remaining_years − deploy_delays[best_tech]
                          + terminal_payoff_years)
annual_reduction  = frac_test × output_mwh × ef_saved / 1e6     [Mt]
annuity_factor    = (1 − (1+r)^−effective_horizon) / r          (or
                    effective_horizon if r = 0)
avoided_NPV       = annual_reduction × price_ma3 × annuity_factor
invest_cost       = compute_investment_cost(best_tech, frac_test, year)
npv_ratio         = avoided_NPV / invest_cost
```

**Branch on tag.** The npv_ratio drives `invest_frac` differently for
the two tags:

- **Green-tagged** bots invest proportionally to the NPV ratio with a
  floor:
  `invest_frac = clip(frac_test × min(npv_ratio, 2)/2 + 0.02, 0.02,
  investment.max_invest_frac)`.
- **Financial-tagged** bots only invest when `npv_ratio > 1` and use a
  lower floor:
  `invest_frac = clip(frac_test × min(npv_ratio, 2)/2, 0.005,
  max_invest_frac)` if `npv_ratio > 1`, else `0.005`.

**Capex throughput clip.** If the estimated invest cost exceeds
remaining capex throughput for the year, `invest_frac` is scaled down
linearly so the cost matches the remaining throughput.

**Compliance-priority clip.** After the throughput clip, the bot
computes a "post-compliance" budget headroom

```
expected_settlement = max(reserve, mac.coal_to_gas_cost)
expected_compliance = qty_mult × annual_need × expected_settlement
safety_reserve      = 0.05 × annual_budget
post_compliance     = max(0, available_cash − expected_compliance
                                              − safety_reserve)
```

and re-clips `invest_frac` so the implied capex is at most
`post_compliance`. This is the rule that makes cash-tight bots invest
less without explicit coordination.

**Loan and EMA smoothing.** If outstanding emergency loan / annual
budget > 0.05, `invest_frac` is multiplied by `(1 − 0.5 ×
loan_pressure)` (loan_pressure clipped at 1). Finally an EMA against
the previous year's value:
`invest_frac ← 0.5 × invest_frac + 0.5 × prev_invest_frac`, then
clipped to `[0, max_invest_frac]`. The previous value is stored on the
`Company`.

### 2.5 Loan-aware quantity reduction

After §2.3, if outstanding loan / annual budget > 0.05, `qty_mult` is
multiplied by `(1 − min(0.20, 0.3 × loan_pressure))` — i.e. up to a
20 % cut in compliance buying while a loan is outstanding.

### 2.6 Optional budget-stress event

When `bots.enhanced_noise.enabled` is on, each bot draws a Bernoulli
"stressed" flag at episode start. Stressed bots have their final
`qty_mult` multiplied by `budget_stress_qty_mult` (default 0.65) and
re-clipped. Nothing else about the bot's behaviour changes.

## 3. Secondary market phase — `secondary_action`

The secondary action is `[sec_price, sec_qty]` with `sec_qty > 0`
meaning buy, `< 0` meaning sell.

### 3.1 Trade direction and size

The bot targets a **forward-looking bank buffer** that linearly
shrinks to zero in the last year:

```
target_bank      = annual_need × min(remaining_years − 1, 2) × 0.3
current_position = bank + allocation − annual_need
trade_target     = 0.5 × (target_bank − current_position)
```

i.e. each year the bot closes half the gap to a buffer of ~0.6 ×
`annual_need` (clamped above by 2 years).

Two overrides:

- **Final-years aggression.** When `remaining_years ≤ 2` and
  `current_position < 0`, `trade_target` is bumped up to
  `min(1.5 × |current_position|, quantity_max)` so the last two years
  prioritise covering compliance.
- **No selling under carry-forward debt.** If `carry_forward > 0.01`,
  `trade_target` is clipped at 0 — the bot will only buy, never sell.

### 3.2 Buy-side budget cap

If `trade_target > 0`, the bot caps the buy size by remaining budget:

```
spend_frac = 0.9 if carry_forward > 0.01     # aggressive recovery
           = 0.6 if current_position < 0     # plain shortfall
           = 0.3 otherwise                   # routine top-up
max_spend  = spend_frac × (annual_budget − budget_spent_this_year)
trade_target = min(trade_target, max_spend / max(clearing_price, 1))
```

Outstanding loans further scale the buy size down by up to 40 %
(`× (1 − 0.4 × loan_pressure)`).

### 3.3 Price

The price uses the same anchor logic as the auction, but with the
**current clearing price** in place of `price_ma3`:

```
anchor   = max(mac.coal_to_gas_cost, clearing_price) + valuation_noise
coverage = (bank + allocation) / annual_need
urgency  = (1 − coverage / urgency_denom) × urgency_multiplier, clipped
severity = |trade_target| / max(annual_need, 0.1)
```

For **buys**:
`price_frac = urgency + 0.2 × min(severity, 1)`, and
`sec_price = anchor + price_frac × (penalty_rate − anchor)`.

For **sells**:
`price_frac = max(0.10, urgency) + 0.15 × min(severity, 1)`.
This gives surplus holders a small spread above the anchor that
widens with severity.

For zero trade: `sec_price = clearing_price`, `sec_qty = 0`.

After computation, the price is capped at `1.8 × penalty_rate` and
clipped to `[trading.sec_price_min, trading.sec_price_max_mult ×
penalty_rate]`; `sec_qty` is clipped to `±auction.quantity_max`.

## 4. Reaction to other state

The bot **does not** look at peer bids, peer holdings, recent reward
realisations, or the cap trajectory beyond what enters via
`price_ma3`, `auction_volume`, `cap_t`, and the dynamic reserve. It
**does** react to:

- inflation (via the inflation-adjusted penalty rate used as the WTP
  cap),
- supply scarcity (via the auction_volume / cap_t ratio bump in §2.1),
- its own collateral load and outstanding loan,
- carry-forward debt (no-sell rule + aggressive buy-spend fraction),
- the year index (via `remaining_years` in target-bank, NPV horizon,
  and final-year aggression).

## 5. Per-episode randomness

Three scalars are drawn at every `reset()` from the environment's
dedicated `_bot_rng` stream, fixed for the rest of the episode:

| Scalar                      | Distribution                                | Where it enters                             |
|-----------------------------|---------------------------------------------|---------------------------------------------|
| `valuation_noise[b]`        | `Normal(0, valuation_noise_std)` (EUR/t)    | Additive to the anchor in §2.1 and §3.3.    |
| `urgency_multiplier[b]`     | `Uniform(urgency_mult_low, urgency_mult_high)` | Multiplies urgency before clipping.      |
| `budget_stressed[b]` (opt.) | `Bernoulli(budget_stress_prob)`             | Cuts auction `qty_mult` (§2.6).             |

`urgency_denom` is fixed per slot via `bots.urgency_denominators` (not
random). Under `enhanced_noise`, the std and the multiplier bounds are
widened (default `15 EUR/t`, `[0.6, 1.5]`) and the stress event is
enabled. Conditional on these scalars and the public state, the bot is
deterministic.

## 6. The green / financial tag

The investment branch in §2.4 is selected by **`w_green > 0.25`**,
read from the agent's reward weights. This is inferred from
`bot_reward_weights` (or `reward_weights` for a learning agent the
heuristic is being used to seed) rather than from the agent index, so
re-ordering the weight list re-tags the bots consistently.

The tag affects only:

- the test fraction `frac_test` used to size avoided-emissions NPV
  (0.07 green, 0.03 financial), and
- the gating of `invest_frac` against `npv_ratio` (green: proportional
  + floor; financial: hard `npv_ratio > 1` gate).

Auction price, qty multiplier, secondary direction, and secondary
pricing do **not** depend on the tag.

## 7. Properties relevant for a bots-only baseline

The following are direct consequences of the rules above; they are
useful when reporting a bots-only run but are not by themselves
endorsements of realism.

- **Price formation is anchored to fundamentals.** Bids and secondary
  quotes are linear interpolations between
  `max(mac_cost, price_ma3 | clearing)` and the inflation-adjusted
  penalty rate, with the interpolation coefficient set by an
  urgency/severity statistic. There is no learned drift, no banking-
  premium term beyond the buffer in §3.1, and the penalty rate is a
  strict (minus-1) cap on willingness-to-pay.
- **Quantity is need-driven, not price-elastic.** `qty_mult` is a
  function of `annual_need` and urgency only; the bot does not reduce
  quantity in response to a high `wtp_economic` (the joint budget gate
  on the env side may still do so).
- **Investment is gated by NPV and cash.** Both tags route through the
  same NPV computation, the same capex-throughput clip, and the same
  post-compliance budget headroom clip. Under cash stress the
  compliance-priority clip dominates and `invest_frac` collapses
  toward zero regardless of NPV. Year-to-year smoothing damps
  oscillation.
- **One technology per year.** Tech logits are a hard one-hot on the
  argmax of `(effective_years × capacity_factor / capex)`. Diversified
  build-out across techs only emerges over multiple years.
- **Final-year compliance bias.** In the last 2 years the bot will buy
  to cover any shortfall up to `quantity_max`; under carry-forward
  debt it cannot sell. Combined, these produce a structural late-
  episode net demand from any bot entering year 11 short.
- **Loan and stress are dampeners only.** Outstanding loans
  monotonically reduce buy quantity, invest fraction, and secondary
  buy volume; the budget-stress event monotonically reduces auction
  `qty_mult`. None of these mechanisms can flip the trade direction
  or increase aggressiveness.
- **No peer modelling.** Bots do not condition on each other. Any
  "coordination" observed in a bots-only run is the joint product of
  shared market state (`price_ma3`, reserve, supply ratio, penalty)
  and the per-bot persistent scalars.

## 8. Implementation pointers

| File                                          | Contents                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------|
| `src/agents/heuristic_policy.py`              | `auction_action`, `secondary_action` — all rules in §§2–3.                |
| `src/environment/ets_environment.py`          | Per-episode scalar draw, fade resolution, dispatch into the policy.       |
| `configs/default.yaml` (`bots:` block)        | The four noise/urgency knobs, `enhanced_noise`, `fade_schedule`.          |
| `docs/config_dictionary.md`                   | One-line reference for each knob mentioned above.                         |

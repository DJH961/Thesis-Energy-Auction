# Heuristic Bot — Configuration & Setup

Reference for the rule-based **heuristic bot** participants (`B1…B8`). The
environment, auction, secondary market, budget, MSR, ESG reward, etc. are
documented in `design.md`, `config_dictionary.md`, `action_space.md` and
`reward_function.md` — **this doc only covers what is different from a
HAPPO learning agent**. If a knob is not mentioned here, the bot inherits
the same environment-side behaviour as a learning agent.

> Bots are off by default (`companies.n_bot_agents: 0`). They are intended
> for ablation, calibration and opponent-diversity runs. The mirror config
> arrays (`bot_*`) are kept synchronised even when bots are inactive.

---

## 1. What stays the same

Bots are full `Company` instances and live in the same simulation loop as
learning agents. They share:

- the same action space (6 Phase-1 dims, 2 Phase-2 dims), same bounds,
  same lot/budget/collateral/MSR/penalty machinery;
- the same observation construction (the `Company` builds the obs even
  though the bot never reads it);
- the same revenue passthrough, treasury reserve, emergency loan,
  carry-forward, ESG reward channels;
- the same default reward weight pattern `[1.0, 0.0]` (pure financial) /
  `[0.5, 0.5]` (balanced ESG) alternating across the 8 archetype slots.

The bot does **not** have a policy, value net, GAE buffer, BC loss, HPP
snapshot, or any gradient flow — it is invoked synchronously by the
environment at bid time and returns a deterministic action conditioned on
two per-episode random scalars.

## 2. Indexing & wiring

- Bot indices are `n_agents … n_agents + n_bot_agents − 1`. Learning
  agents come first; bots are appended.
- The environment auto-extends the per-agent config arrays with the
  `bot_*` mirrors before constructing `Company` objects, so every
  per-agent knob (`initial_mix`, `reward_weights`, `annual_budgets`,
  `debt_headrooms`, `capex_throughputs`) has a bot counterpart of the
  same shape.
- Market calibration (cap, emissions, MSR thresholds) is recomputed each
  episode against `n_agents + active_bots`, so bot fade-out shrinks the
  cap accordingly. See `market_calibration.py`.
- Bot actions are produced inside `step_auction` / `step_secondary` via
  `_generate_bot_auction_actions` and `_generate_bot_secondary_actions`,
  which call `src/agents/heuristic_policy.py`. The external `step()`
  interface receives only learning-agent actions.

## 3. Archetype defaults (`companies.bot_*`)

| Slot | `bot_initial_mix` (coal/gas/onshore/offshore/solar) | Archetype     | `bot_reward_weights` |
|------|------------------------------------------------------|---------------|----------------------|
| B1   | 0.40 / 0.40 / 0.10 / 0.05 / 0.05                     | coal-heavy    | [1.0, 0.0] financial |
| B2   | 0.40 / 0.40 / 0.10 / 0.05 / 0.05                     | coal-heavy    | [0.5, 0.5] balanced  |
| B3   | 0.15 / 0.45 / 0.20 / 0.10 / 0.10                     | gas-dominant  | [1.0, 0.0]           |
| B4   | 0.15 / 0.45 / 0.20 / 0.10 / 0.10                     | gas-dominant  | [0.5, 0.5]           |
| B5   | 0.05 / 0.25 / 0.35 / 0.20 / 0.15                     | transitioner  | [1.0, 0.0]           |
| B6   | 0.05 / 0.25 / 0.35 / 0.20 / 0.15                     | transitioner  | [0.5, 0.5]           |
| B7   | 0.00 / 0.10 / 0.30 / 0.35 / 0.25                     | green-leader  | [1.0, 0.0]           |
| B8   | 0.00 / 0.10 / 0.30 / 0.35 / 0.25                     | green-leader  | [0.5, 0.5]           |

`bot_annual_budgets`, `bot_debt_headrooms`, `bot_capex_throughputs`
mirror the learning-agent arrays element-by-element (`[880, 880, 800,
800, 820, 820, 780, 780]` etc.). Each pair B(2k−1)/B(2k) is identical at
the resource level — the only differentiator is the reward weight, which
the heuristic reads to switch its investment branch.

**Green vs. financial split is inferred from `w_green > 0.25`**, not from
index parity, so re-ordering `bot_reward_weights` re-tags the bots
correctly.

## 4. Behavioural model (`heuristic_policy.py`)

### 4.1 Auction action `[bid_price, qty_mult, invest_frac, logits×3]`

- **Market anchor**: `max(mac.coal_to_gas_cost, price_ma3) +
  valuation_noise`. `price_ma3` is the env's 3-yr MA of clearing.
- **Urgency**: `1 − coverage_ratio / urgency_denom`, clipped to `[0, 1]`,
  scaled by the bot's persistent `urgency_multiplier`. Boosted by up to
  `0.3 × (1 − supply_ratio)` when MSR-adjusted auction volume is below
  80 % of the year's cap (supply-scarcity awareness).
- **Bid price — dual WTP ceiling**:
  - *Economic ceiling*: `anchor + urgency × (penalty_rate − anchor)`,
    capped strictly below the inflation-adjusted penalty rate. No
    `price_ma3` feedback term — that was removed to prevent a runaway
    MA3 loop.
  - *Budget ceiling*: `max_compliance_share × available_cash /
    target_qty`. `available_cash = operating_cash + treasury_fraction ×
    treasury` (mirrors the env-side joint budget gate; loan headroom is
    excluded, treasury is meant to absorb spikes not size routine bids).
  - Result: `max(min(economic, budget), reserve + 1)` then clipped to
    `[auction.price_min, auction.price_max]`.
- **Quantity multiplier**: `qty_mult = clip((need + 0.1 × need × urgency)
  / need, qty_mult_low, qty_mult_high)`. `need` includes carry-forward
  debt. The 10 % extra is a safety buffer that scales with urgency.
- **Investment fraction (NPV-gated)**:
  - Pick the buildable tech maximising `(remaining_years − deploy_delay
    + terminal_horizon) × capacity_factor / capex`.
  - Compute avoided-carbon NPV with the **annuity discount factor**
    `(1 − (1+r)⁻ⁿ) / r`, `r = investment.discount_rate` (5 %).
  - Branch:
    - **Green** (`w_green > 0.25`): `invest_frac ≈ 0.07 × min(NPV/cost,
      2)/2 + 0.02`, floor 2 %.
    - **Financial**: only invest if `NPV/cost > 1`, otherwise hold a
      0.5 % floor.
  - Clip down to `capex_throughput` headroom and then to
    *post-compliance budget headroom* (cash minus expected compliance
    cost at the MAC anchor minus a 5 % safety reserve). This is the
    "compliance-priority" clip — coal-heavy bots invest less under stress.
  - **EMA smoothing**: `invest_frac ← 0.5 × invest_frac + 0.5 ×
    prev_invest_frac` to avoid on/off oscillation year-to-year.
- **Loan awareness**: when `loan_outstanding / annual_budget > 0.05`,
  `qty_mult` is scaled down up to −20 % and `invest_frac` up to −50 %
  proportional to loan pressure.
- **Tech logits**: hard one-hot on the winning tech (`+1`, `−1`, `−1`),
  i.e. the bot never mixes — it commits to one renewable per year.

### 4.2 Secondary action `[sec_price, sec_qty]`

- **Target bank trajectory**: `target_bank = need × min(remaining_years
  − 1, 2) × 0.3` (≈ a ~0.6-year buffer mid-episode, ramping to 0 at the
  end). Trade target is half the gap to `target_bank`.
- **Final-year aggression**: in the last 2 years, if currently short,
  the trade target is bumped to cover the shortfall (capped at
  `quantity_max`).
- **Compliance safeguard**: while `_carry_forward > 0.01`, the bot is
  forbidden from selling (`trade_target` clipped at 0).
- **Spend caps**: max share of remaining budget allowed for a buy is
  `0.9` if in carry-forward debt, `0.6` if just short, else `0.3`.
- **Loan awareness**: outstanding loan scales buy size down up to −40 %.
- **Price**: `anchor + price_frac × (penalty_rate − anchor)` with
  `price_frac = urgency + 0.2 × severity` for buys, and a tighter
  `max(0.10, urgency) + 0.15 × severity` for sells. Clipped to
  `[trading.sec_price_min, sec_price_max_mult × penalty_rate]`, with a
  hard `1.8 × penalty_rate` cap before clip.

### 4.3 What is **not** modelled

- No banking-premium or expected-future-price term beyond the
  fundamentals anchor (the env handles banking incentives via the
  reward function).
- No strategic withholding, collusion, or learned response to peer
  behaviour — bots react only to public market state and their own
  finances.
- No exploration noise, action noise, or epsilon-greedy on top of the
  rules; all randomness is the two persistent scalars in §5.1 and the
  optional `enhanced_noise` qty-stress draw.

## 5. `bots:` config block

| Key                          | Purpose                                                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `valuation_noise_std`        | Per-episode Gaussian on the **anchor** for each bot. Persistent within an episode, redrawn at `reset()`. Default **5 EUR/t**.   |
| `urgency_mult_low/high`      | Per-episode `Uniform(low, high)` scalar multiplying the bot's urgency term. Default **[0.8, 1.2]**.                             |
| `urgency_denominators`       | Per-slot list of denominators in `urgency = 1 − coverage / denom`. Lower → more aggressive. Default `[1.3, 1.7] × 4`.            |
| `max_compliance_share`       | Fraction of `available_cash` the heuristic is willing to commit to compliance in a single auction. Default **0.70**.            |

### 5.1 `bots.enhanced_noise` — opponent diversity boost

Inflates the persistent noise draws and adds a budget-stress event so
the bot population isn't lock-step. Defaults:

| Key                       | Default | Effect                                                                            |
|---------------------------|---------|-----------------------------------------------------------------------------------|
| `enabled`                 | `true`  | If on, overrides the four base noise knobs.                                       |
| `valuation_noise_std`     | `15.0`  | Wider anchor noise.                                                               |
| `urgency_mult_low/high`   | `0.6 / 1.5` | Wider urgency dispersion.                                                     |
| `budget_stress_prob`      | `0.15`  | Per-episode, per-bot Bernoulli. Stressed bots' `qty_mult` is cut at bid time.     |
| `budget_stress_qty_mult`  | `0.65`  | Multiplier applied to `qty_mult` (then re-clipped to `[qty_mult_low, high]`).     |

### 5.2 `bots.fade_schedule` — episode-based retirement

Turns bots off in stages as training progresses. Disabled by default.

| Key        | Default                                          | Effect                                                                  |
|------------|--------------------------------------------------|-------------------------------------------------------------------------|
| `enabled`  | `false`                                          | Master switch.                                                          |
| `schedule` | `[[0, 8], [30000, 6], [60000, 4], [80000, 2]]`   | `[episode_threshold, active_bot_count]` step function (non-decreasing in threshold). |

When a bot is "faded", it submits a no-op action (`bid_price=price_min,
qty=0, invest=0`, zero secondary action) but its `Company` slot remains
allocated. Market calibration is **rerun** at episode start whenever the
active bot count changes so cap, MSR thresholds and emissions targets
re-scale to the smaller participant pool.

## 6. Determinism & RNG

- Bot persistent noise draws (valuation noise, urgency multiplier,
  budget-stress mask) come from `self._bot_rng`, an independent RNG
  stream so changes elsewhere don't shift bot behaviour for a fixed
  seed.
- Given the two episode-start scalars and the optional stress flag, the
  heuristic is fully deterministic — same env state → same action.

## 7. Files at a glance

| File                                          | Role                                                          |
|-----------------------------------------------|---------------------------------------------------------------|
| `src/agents/heuristic_policy.py`              | `auction_action`, `secondary_action` (the rules above).       |
| `src/environment/ets_environment.py`          | Persistent-noise draw, fade resolution, bot-action dispatch.  |
| `src/environment/market_calibration.py`       | Re-scales cap / MSR when active bot count changes.            |
| `configs/default.yaml` (`bots:`, `bot_*`)     | All knobs documented above.                                   |
| `src/utils/preflight.py`                      | Validates `bot_*` array lengths against `n_bot_agents`.       |

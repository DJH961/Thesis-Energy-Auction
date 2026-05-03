# ETS MARL — Configuration Dictionary

`configs/default.yaml` exposes **38 top-level subsystems** and several
hundred tunable parameters in ~590 lines. This document is a one-stop
reference: each subsystem gets a one-line purpose plus the handful of
knobs that materially change behaviour. For the full parameter list
read the YAML directly — it is heavily commented.

> **Convention.** Many schedule parameters use `0` (or `-1`) to mean
> *"auto-scale relative to `simulation.n_episodes`"* — the auto formula
> is documented next to each knob in `default.yaml` and resolved at
> the top of `scripts/train.py`. The companion `tabula_rasa:` block is
> kept as an ablation reference but `tabula_rasa.enabled=true` raises
> `ValueError` — it is not a runtime override.

---

## Top-level meta

| Key | Purpose |
|---|---|
| `version` | Run banner. Bump in lockstep with `pyproject.toml`. |
| `device` | `auto` → CUDA if available, else CPU. |

## `simulation` — episode shape

`n_years=12`, `n_episodes=120000`, `seeds=[1729, 8191, 6561, 5041]`. A
12-year episode mirrors EU-ETS Phase 4 (2025 → 2036).

## `technologies` — 5 generation techs

Names + per-tech vectors of `emission_factors`, `capex` (€/kW),
`capacity_factors`, `deploy_delays` (yr; **0/0/4/7/2** for
coal/gas/onshore/offshore/solar — the strategic core of the game),
`operational_costs`, `decommission_costs`, `is_green`, `is_buildable`
(only renewables can be built).

## `ets` — cap, MSR, reserve price

| Key group | Purpose |
|---|---|
| `cap_year_0_override`, `initial_bank_fraction`, `cap_overhead_pct` | Calibrate year-0 cap to participant emissions. `null` ⇒ auto from data. |
| `lrf_phase1=0.043`, `lrf_phase2=0.044`, `lrf_phase_switch=2` | Linear reduction factor. Two-phase EU-ETS schedule. |
| `msr.{enabled, tnac_upper_ratio, withhold_rate, release_frac, activation_year, …}` | Market Stability Reserve with 1-year TNAC lag, cancellation mechanism, and price-containment / -release thresholds. |
| `banking`, `reserve_price`, `reserve_price_mode`, `reserve_discount`, `reserve_initial` | Banking on/off + auction floor mechanics. |
| `unsold_to_msr`, `max_rollover_multiplier`, `price_history_anchor` | Unsold-volume routing & MA(3) source. |

## `companies` — the agent population

| Key | Purpose |
|---|---|
| `n_agents=8` | Learning agents. Default: 4 reward archetypes × 2 seats. |
| `output_twh=10.0` | Each company produces 10 TWh/yr. |
| `initial_mix` | 5-vector tech-share per agent. The 8 default mixes span 45 %→75 % green at episode start. |
| `reward_weights` | `[w_cost, w_green]` per agent. Default alternates `[1,0]` and `[0.5,0.5]`. |
| `n_bot_agents=0` | Heuristic bots indexed after learning agents. **Default 0**; rule-based bot config (`bot_initial_mix`, `bot_reward_weights`) is kept synchronised in case bots are switched on. |

## `investment` / `risk` — capex & FID risk

`max_invest_frac=0.20`, `convexity_alpha=0.20`, `discount_rate=0.05`
(WACC). `risk.{p_fail_min=0.08, p_fail_max=0.40, p_fail_alpha=0.7,
experience_discount=0.10}` — FID failure probability rises with
queue-relative volume; agents with ≥ 2 completions get a discount.

## `auction` — primary market

| Key | Purpose |
|---|---|
| `price_min=45`, `price_max=250` | Action-space bounds for `bid_price`. |
| `quantity_max=3.0`, `qty_mult_low=0.5`, `qty_mult_high=2.0` | Quantity-multiplier action bounds. |
| `bid_change_limit.{enabled, value=75}` | Year-over-year bid change cap; anchored on `max(price_ma3, fundamental_anchor)`. |
| `pricing_rule="uniform"`, `lot_size=0.0005` | Uniform-price clearing & lot rounding. |
| `collateral.*` | Collateral on bid value, opportunity-cost shaping, bidder budget cap. |
| `budget_gate.{enabled, safety_mult=1.2, notional_safety_mult=5.0, protect_need_floor=true, inflation_aware_ma3=true, treasury_fraction=0.33, include_loan_headroom=false, shock_aware_need_floor=false}` | Joint budget gate — sizes `bid_q` against expected settlement using `op_cash + treasury_fraction × treasury` plus optional loan headroom. Two-stage shrink: qty → price → qty. |

## `trading` / `secondary` — secondary market

`enabled=true`, `transaction_cost=0.5` (€/Mt), `spread_tolerance=0.12`,
`sec_price_min=45`, `sec_price_max_mult=2.0` (× max-penalty).
`secondary.liquidity_pool` is an off-by-default ELP-style residual
liquidity facility.

## `budget` — financial constraints & loans

| Key | Purpose |
|---|---|
| `mode="revenue_based"` | Annual budget = revenue passthrough + base. |
| `annual_budgets`, `debt_headrooms` | Per-agent caps. Coal-heavy archetypes get higher headroom; pure-green archetype tightest. |
| `dynamic_budget_ceiling_multiplier=2.5` | Cap on dynamic budget growth — = multiplier × `annual_budgets[i]`. |
| `overspend_penalty_coef`, `tiered_penalty_coef`, `hard_cap_fraction=1.15`, `soft_zone_start=1.00` | Soft + tiered + hard penalties on budget overspend. |
| `capex_throughputs`, `capex_overspend_coef` | Annual capex throughput cap (M€/yr) per archetype. |
| `investment_hard_gate=true` | Pre-emption: clip `invest_frac` if it would breach the hard cap. |
| `emergency_loan.{max_loan_fraction=0.15, interest_rate=0.08, repayment_years=3, leverage_premium_*, capex_squeeze_floor, origination_sting_coef}` | Last-resort loan at end-of-year settlement; tiered cost + capex squeeze + small reward sting. |
| `treasury_reserve.{retention_fraction=0.60, cap_multiple=1.5, decay_rate=0.05, terminal_value_rate=0.30}` | Routes a fraction of revenue surplus into a buffer that decays each year, paid out as a terminal payoff. |
| `suspension_length=1` | # episodes a busted agent is suspended. |

## `green_finance` (default off) / `bots`

Optional concessional loan facility for green capex. `bots.*` controls
heuristic-bot valuation noise and urgency multipliers; only relevant
when `companies.n_bot_agents > 0`.

## `penalty` — non-compliance

`rate=138.75` (€/tCO₂ base 2026), `inflation_rate=0.020`,
`inflation_random_std=0.015`, `carry_forward=true`,
`carry_forward_cap=0.0` (uncapped). Inflation drift makes
late-episode default progressively more expensive.

## `price` — price formation knobs

`initial_expected=70`, `banking_premium_mult=1.4` (anchor lift for
banking premium), `burnin_std=10`, `ar1_persistence=0.85`,
`volatility_std=0.15` (real-time AR(1) on top of fundamental anchor).

## `mac` / `electricity` — abatement & revenue

`mac.{enabled, coal_to_gas_cost=48, max_switch_frac=0.20}` — coal-to-gas
fuel switching capped at 20 % per year. `electricity.{base_price=55,
carbon_passthrough=0.90}` — 90 % carbon-cost passthrough into wholesale
electricity revenue.

## `reward` — shaping & normalisation

| Key | Purpose |
|---|---|
| `normalizer_alpha=0.02`, `gae_min_std=0.15` | Per-agent EMA reward normaliser. |
| `clip_min/max=±10`, `shaping_beta=7.0`, `shaping_gamma=1.0`, `shaping_decay_episode=0`, `shaping_decay_frac=0.10`, `shaping_weight_floor=0.0` | Shaping-weight schedule (decays from 1 → floor over 10 % of training by default). |
| `terminal_bank_value`, `terminal_queue_value`, `terminal_payoff_years=5`, `treasury_terminal_value` | Terminal payoffs (year-T only). |
| `anchor_normalize_cost_only`, `budget_norm_anchor`, `budget_norm_budget_0` | Reward-bucket normalisation anchor. |
| `opportunity_cost_shaping`, `coverage_gap_shaping`, `banking_signal`, `sec_proxy` | Decaying shaping channels. |
| `compliance_gate_blend_threshold=0.90`, `compliance_gate_blend_width=0.30` | Coverage-frac → ESG gate exponent. |

See **`docs/reward_function.md`** for the full mathematical statement.

## `esg` — saved-carbon hybrid

`enabled=true`, `scale=0.50` (calibrated for ~50/50 fin/ESG balance for
`[0.5, 0.5]` agents — see `docs/esg_reward_design.md`),
`stock_weight=1.0`, `flow_weight=1.5`, `speed_coef=0.3`,
`speed_coef_late=0.3` (uniform — no front-loading).

## `opponent_modeling` / `opponent_obs`

`enabled=true`, `mode="lagged"`, `lag_years=1`,
`queue_noise_sigma=0.15`, `dims_per_opponent=7`. The 7 lagged
opponent observations are: emissions, green frac, fossil frac, queue
(with noise), TNAC share, net secondary trade, lagged compliance gap.

## `agent_cycling` (default off)

Optional soft cycling that scales a "frozen" agent's LR by
`soft_lr_scale=0.3` rather than fully freezing it.

## `ppo` — learner

| Group | Purpose |
|---|---|
| `hidden_size=256`, `lr=2e-4`, `gamma=0.99`, `gae_lambda=0.97`, `clip_eps=0.15` | Standard PPO knobs. |
| `entropy_coef=0.08 → entropy_coef_final=0.015` over `entropy_decay_frac=0.70` | Entropy decay schedule. |
| `value_coef=0.5`, `max_grad_norm=0.5`, `n_epochs=3`, `mini_batch_size=64`, `episodes_per_update=32` | Update-loop sizing. |
| `short_run_overrides.*` | Auto-applied if `n_episodes ≤ long_run_episode_threshold`. |
| `log_std_min=-3.0`, `log_std_max=0.0` | Policy std clipping. |
| `centralized_critic=true`, `split_invest_head=true`, `happo=true`, `happo_dynamic_order=true`, `happo_order_metric="advantage"` | Centralised critic with split bid/invest heads + sequential HAPPO update. |
| `critic_lr=5e-4`, `critic_hidden_size=512`, `critic_warmup_episodes=0`, `critic_extra_epochs=6`, `critic_huber=true`, `critic_huber_delta=4.0`, `clip_value=true` | Critic-side schedule and stabilisation. |
| `lr_decay="cosine"`, `lr_min=1e-5`, `critic_lr_decay="cosine"`, `critic_lr_min=5e-5` | LR annealing. |
| `target_kl=0.015` | KL early-stop threshold. |
| `dual_clip_c=3.0`, `log_ratio_clip=10.0` | Dual-clip PPO + numerical safety. |
| `kl_anchor_beta=0.0`, `kl_anchor_decay_episodes=0` | (Off by default) KL anchor toward BC policy. |

## Curriculum / shocks

| Subsystem | Purpose |
|---|---|
| `curriculum` (off) | Optional shorter-episode warm-up. |
| `cap_curriculum`, `budget_curriculum` (off) | Front-load loose cap / budget then ramp tight. |
| `uncertainty.{sigma_demand=0.07, corr_rho=0.40}` | Multiplicative emissions shock with cross-agent correlation. |
| `construction_jitter.{poisson_lambdas, p_cancel_per_tech, recovery_rate=0.40, cf_sigma}` | Poisson per-tech project counts, per-project cancellation, partial-capex recovery, capacity-factor noise. |

## `warm_start` — episode-0 priors

Burn-in years (default 4) seed `price_history` (mean 70 ± 10 €/t),
opening bank (15–35 % of need), construction queue (Poisson means per
tech), and the MSR reserve (`msr_initial_reserve_frac=0.23`) so episode
0 is not a cold-start zero state.

## `logging`

| Key | Purpose |
|---|---|
| `log_interval=100`, `save_interval=500`, `csv_flush_interval=1000`, `snapshot_interval=2500` | Console / disk cadence. |
| `snapshot_keep_recent=1` | # rolling snapshot pairs retained per seed (linear footprint). |
| `snapshot_delete_on_finish=true` | Purge `snapshots/` on clean exit. |
| `results_dir="results/"` | Output root. |
| `checkpoint_pruning.{enabled, online=true, n_keep_milestones=20, n_keep_recent=5}` | Bound `checkpoints_*/` size at all times. |

## `pretrain` (default off)

Optional behavioural-cloning warm-start from the heuristic policy.

## `diagnostics`

Window sizes and thresholds for the warning detectors that produce the
`warn_*` columns in `training_log_*.csv`.

## `hpp` — Historical Policy Pool

`pool_size=10`, `swap_prob=0.20`, `warmup_episodes=0` (auto: 20 % of
n_episodes). With probability `swap_prob` an agent's actor weights are
swapped with a randomly drawn historical snapshot for one episode —
non-stationarity training trick.

## `exploration`

`mode="anchored"`, `epsilon_start=0.25 → epsilon_final=0.02` over
`epsilon_decay_frac=0.45`. Optional `anchor_snap.{enabled=false,
prob_per_episode=0.02}` periodically snaps `price_head.bias` to the
fundamental anchor (Tier-3 mitigation for basin lock-in).

## `phantom_bidder` (default off)

Adds a synthetic bid each year drawn from a price log-normal × penalty
fraction. Used for ablation studies of price discovery without it.

## `urgency_scalars`

`enabled=true`, `lognormal_sigma=0.30`. Per-agent per-episode lognormal
draw multiplied into the penalty term — folded into every reward.

---

## How sweeps reuse this file

`scripts/sweep.py` reads a sweep spec (e.g.
`configs/sweeps/example_sweep.yaml`), copies `default.yaml` for each
variant, applies dotted-path overrides
(e.g. `ets.cap_overhead_pct: -0.02`), writes the resolved YAML into
the variant directory, and launches `train.py --config <variant.yaml>
--seed S --run-tag <variant>` for each `(variant, seed)` pair. So
**every parameter above is overridable** without editing `default.yaml`.

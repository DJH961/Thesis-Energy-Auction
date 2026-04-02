# Changelog — ETS MARL (`ets_marl_happo_current`)

Version numbers reflect the `# ETS MARL — Configuration vX.Y` header in `configs/default.yaml`
and, from v6.1.0 onwards, the `version` field in `pyproject.toml`.

---

## v7.0.0

**Tabula-Rasa Training Mode**

- Added `tabula_rasa` config block for cold-start ablation mode with explicit fraction-based overrides.
- `scripts/train.py` now applies tabula-rasa overrides right after per-seed config deep-copy and before any auto-schedule resolution.
- Enabling tabula-rasa now:
	- disables BC pretraining and KL anchor,
	- switches exploration mode to uniform,
	- removes auction/secondary action anchors,
	- overrides epsilon, entropy, critic warmup, shaping decay, and HPP warmup schedules from `n_episodes` fractions.
- `PPOAgent` now supports `exploration.mode` and switches epsilon-random actions between `anchored` and `uniform` sampling.

**Budget Relaxation + Green Finance**

- Coal-heavy budgets and capex-throughput values were relaxed in `configs/default.yaml`.
- Added `green_finance` config section (`enabled`, `loan_budget_boost`, `loan_interest_rate`, `capex_throughput_boost`).
- `Company` now tracks green-loan utilization with:
	- `record_green_loan`,
	- `compute_green_loan_cost`,
	- `green_loan_headroom`,
	- `green_capex_headroom`.
- `ETSEnvironment.step_auction` now supports green-finance recovery after normal budget/capex clipping.
- Reward computation now includes annual green-loan interest in `total_cost_ex_penalty` (reward cost channel only; not budget spending).

**Dynamic Market Calibration (Emission-Weighted) + Bot Features**

- Added `src/environment/market_calibration.py`:
	- `compute_system_emissions(...)`
	- `compute_market_params(...)`
- Cap/MSR calibration is now derived from active-participant emissions using ratio-based config keys.
- Environment init now writes derived values into runtime config and prints calibration summary.
- Added backward-compat deprecation path for old-style hardcoded cap/MSR configs.
- `CapSchedule` gained runtime `update_calibration(...)` support for fade-triggered recalibration.
- Added bot `enhanced_noise` and `fade_schedule` controls in config and runtime behavior.
- Fade now supports reducing active bots by schedule and recalibrating cap/MSR accordingly.
- Retired bots are fully removed from market dynamics (zero auction demand, no secondary activity, zero emissions/compliance contribution).

**Configuration Refactor**

- ETS config moved to ratio-based inputs:
	- `cap_year_0_override`, `cap_overhead_pct`
	- `msr.tnac_upper_ratio`, `msr.tnac_lower_ratio`
	- `msr.release_frac`, `msr.emergency_release_frac`
- Added exploration mode key: `exploration.mode` (`anchored` or `uniform`).

**Tests**

- Added:
	- `tests/test_tabula_rasa.py`
	- `tests/test_green_finance.py`
	- `tests/test_market_calibration.py`
	- `tests/test_bot_features.py`
- Existing core integration and cap schedule tests remain valid under both new ratio-based and backward-compatible paths.

## v6.4.0

**Reward Decomposition for Post-Hoc Analysis**

- Separated reward logging into two explicit components while keeping learning behavior unchanged:
	- `reward_base_A*`: constant reward terms (cost/revenue, ESG, penalties, opportunity cost, efficiency, and terminal values)
	- `reward_shaping_A*`: decaying shaping bonuses (`green_bonus + queue_bonus`)
	- `reward_A*`: unchanged total reward used for optimization (`base + shaping`)
- Environment year logs now emit `rewards_base` and `rewards_shaping` arrays alongside total rewards.
- Training pipeline writes the new base/shaping reward columns to both `year_log_s*.csv` and `training_log_s*.csv`.
- Training convergence diagnostics cell (D6) now visualizes system base-vs-shaping-vs-total reward trajectories and uses base rewards for per-agent constant trajectory plots when available.


## v6.3.0

**Burn-In Calibration and Warm-Start Upgrade**

- Warm-start now supports a hidden heuristic burn-in pre-period (`warm_start.burnin_enabled`) that initializes holdings, MSR reserve, construction queues, and price history jointly before visible year 0.
- Added new warm-start controls: `n_burnin_years`, `burnin_price_seed_mean`, and `burnin_price_seed_std`.
- Warm-start bank seeding recalibrated from `[0.5, 1.5]` to `[0.2, 0.4]`.

**MSR and Cap Calibration**

- MSR TNAC thresholds recalibrated to `tnac_upper=18.0` and `tnac_lower=9.0`.
- MSR emergency release amount increased from `0.90` to `4.0` Mt.
- `CapSchedule.get_cap()` now supports negative years (used by burn-in) via backward LRF extrapolation.
- `CapSchedule.get_auction_volume()` adds `force_msr` to bypass activation-year gating for hidden burn-in years.

**Reward and Observation Extensions**

- Added `reward.opportunity_cost_rate` (default `0.05`) and applied a cost-of-capital term on post-compliance banked allowances in reward computation.
- Phase-1 observation base expanded from 25 to 28 dimensions:
	- own bank ratio,
	- predicted MSR withholding signal,
	- auction volume change vs cap.
- Derived dimensions update: with 16 participants and opponent modeling, phase-1 is now 103D and phase-2 is 110D.

**Training Profile Infrastructure**

- Default long-run training config updated to:
	- `simulation.n_episodes=100000`
	- `ppo.lr=0.0003`
	- `ppo.entropy_coef_final=0.015`
	- `ppo.episodes_per_update=16`
- Added long-vs-short profile resolution in training:
	- long-run values apply when `n_episodes > 20000`
	- short-run overrides restore prior values (`lr=0.0002`, `entropy_coef_final=0.03`, `episodes_per_update=8`) for shorter runs.


## v6.2.0

**Bot Stochastic Valuation and Differentiation**

- Each bot now samples a persistent `valuation_noise` (drawn from N(0, σ)) and an `urgency_multiplier` (drawn from U[low, high]) at episode start. These persist for the entire episode, giving bots heterogeneous but stable bidding personalities across years.
- Bot pairs are differentiated by archetype via `urgency_denominators`: even-indexed bots use a denominator of 1.3 (more aggressive coverage), odd-indexed bots use 1.7 (more conservative).
- New config section `bots:` controls `valuation_noise_std`, `urgency_mult_low`, `urgency_mult_high`, and `urgency_denominators`.

**MSR Price-Containment Fix**

- Replaced ratio-based MSR triggers with dynamic absolute thresholds derived from the inflation-adjusted effective penalty rate.
- Containment trigger fires at `1.8 × effective_penalty_rate` (or `price_containment_absolute` as a hard fallback); emergency release at `2.5 × effective_penalty_rate` (or `price_release_absolute`).
- Fixes an always-dormant containment bug where the old ratio trigger (70 % of price_max) was almost never reached.
- MSR trigger events (containment, emergency release) are now logged. Removed deprecated ratio params from config.

**Terminal Bank Value Cap**

- Effective bank for terminal valuation capped at `min(holdings, 2.0 × annual_need)`. Holdings beyond a 2-year reserve receive zero additional terminal credit, making secondary-market selling rational.

**Secondary Revenue as Budget Credit**

- Negative `secondary_cost` (revenue from selling) now reduces the agent's budget spending for the year, freeing headroom for investment.

**Permanent Efficiency Bonus**

- Cost-efficiency improvement bonus (`0.3 × ef_improvement_ratio × time_weight × price_weight`) now applies to all agents regardless of `w_green`, giving coal-heavy agents a gradient for early green investment.

**PPO Value Function Clipping**

- Critic loss implements value clipping (`v_pred` clamped to `old_values ± clip_eps`) when `clip_value=true`. `old_values` tensor is now passed through `compute_gae()` into both `update()` and `update_happo()`.

**Config Changes**

- `penalty.carry_forward_cap` reduced from 2.0 to 1.0.
- `ppo.clip_value` set to `true` by default.
- Removed `msr.msr_activation_year` alias and `penalty.inflation_random_window` legacy fallback.

---

## v6.1.0

*Commit: `ca087d7` — "MSR enhancements, logging updates, and documentation (v6.1.0)"*

- MSR cancellation mechanism: MSR holdings exceeding the previous year's auction volume are permanently cancelled each year (EU ETS post-2023 reform). Tracked via `msr_total_cancelled` in year-level CSV.
- MSR price-responsive triggers now reference the inflation-adjusted penalty rate instead of `price_max`, preventing procyclical withdrawal at moderate prices.
- MSR activation lag (`activation_year: 2`) added to mirror EU ETS lagged TNAC observation logic.
- `ValueNetwork` improvements and PPO agent stabilisation.
- Bot behaviour console output enhanced: added green fraction progression and detailed shortfall tracking per bot.
- Version bumped to `6.1.0` in `pyproject.toml` with proper project description.
- All 192 tests passing at release.

---

## v6.0

*Commit: `abe60a5` — "Refactor secondary market pricing and reward structure"*

- Secondary market prices switched from relative multipliers to absolute fundamentals-based prices anchored to MAC cost and penalty rates.
- ESG reward signal introduced: saved-carbon-years formula with terminal queue valuation.
- Terminal bank value improved with diminishing-returns log formula.
- Fix for erratic clearing prices and distorted MA3 price signal (PR #3, commit `04eabad`): auction price history now uses only successful clearing prices (`price_history_anchor: "auction"`), preventing reserve-price pollution from failed auctions.
- Heuristic NPV calculation fixed (undiscounted → properly discounted).
- Minimum auction volume constraint added to `CapSchedule` to prevent zero-auction edge cases.
- Budget constraints and investment scaling enhancements (commit `650d4bf`): budget hard-cap multiplier, contingency zone, per-run schedule isolation to prevent config mutation across seeds.
- `config: cap_year_0` recalibrated to 57 Mt for 16-participant market.

---

## v5.5

*Commit: `09989ac` — "Enhanced capex throughput management and reward differentiation, better heuristic"*

- Capex throughput constraint added to `Company` class: separate per-agent `capex_throughput` cap on annual construction spend, independent of the unified budget.
- Penalty added for capex overspend; `bot_capex_throughputs` added to config.
- Reward differentiation improved based on ESG weighting.
- Heuristic policy updated to respect capex throughput limits when computing investment fractions.

---

## v5.4

*Commit: `49960d7` — "More Bots, better Heuristic - V5.4"*

- Expanded from 12 to **16 total participants** (8 learning agents + 8 heuristic bots).
- Heuristic policy updated with valuation-based bidding and target-bank trajectory trading for secondary market.
- ETS parameters (cap, MSR thresholds) recalibrated for 16-participant market.
- Enhanced training logging: bid multipliers, intent shares, investment technology choices added to year-level CSV.

---

## v5.3

*Commit: `4bffed9` — "Enhance inflation handling in ETS environment and agents"*

- Shared inflation path built per episode: all agents use the same compounded inflation rate each year.
- Heuristic policy updated to apply dynamic inflation-adjusted penalty rates to auction actions.
- KL anchor and critic warmup logging improved; inflation metrics added to year log.
- HPP logging refactored; penalty pricing mechanism updated.

---

## v5.2

*Commit: `533e808` — "Removed price anchor and added inflation"*

- Static price anchor removed; AR(1) price model now drives expected price observations.
- Inflation added economy-wide: penalty rate, CAPEX, OPEX, MAC, and electricity base price all compound annually at a configurable rate.

---

## v5.1

*Commit: `1ce26c4` — "Added bots"*

- Introduced 4 heuristic bot agents (12 total participants). Bots use the same `Company` class and participate in auction and secondary market alongside learning agents.
- Cap and MSR thresholds rescaled for 12-participant market.
- Q-learning baseline added for comparison (commit `e883439`).

---

## v5.0

*Commit: `a2686ca` — "Simplify main layout: legacy PPO, legacy test, and current HAPPO folders"*

This was the first stable version of `ets_marl_happo_current`, consolidating the codebase into the current folder structure.

- **8 learning agents** organised into 4 archetypes (coal-heavy, gas-dominant, transitioner, green-leader), each pair with financial and ESG reward weights.
- **5-technology model**: coal, gas, onshore wind, offshore wind, solar — with distinct emission factors, CAPEX, capacity factors, construction delays (0–7 years), and operational/decommissioning costs.
- **MAC fuel-switching**: coal-to-gas dispatch switching when carbon price exceeds MAC threshold.
- **Electricity revenue channel**: `P_elec = P_base + passthrough × P_carbon × EF_system`.
- **Carry-forward non-compliance**: shortfalls carry to the following year with a configurable cap multiplier.
- **Unified budget envelope**: single annual spending cap for compliance, CAPEX, and MAC.
- **Two-phase yearly decision**: Phase 1 (auction bid + investment), Phase 2 (secondary market), each with its own observation vector and policy network.
- **Terminal payouts**: terminal bank value and queue value added (commit `aa2a08b`).
- **Construction risk**: investment failure probability, Poisson delay jitter, project cancellation risk, capacity-factor noise.
- **Training infrastructure**: pre-training routine (commit `6310d7d`), market-collapse warnings (commit `3bfc11e`), Historical Policy Pool (HPP).
- Uncertainty (demand and emission shocks) reactivated (commit `3e8bb27`).

# ETS MARL Repository — Audit Report

This report compiles diagnostic findings from a static audit of `src/`, `scripts/`, `configs/`, and `tests/`. **No code was modified.** Line numbers are approximate; verify before patching.

---

## 1. Critical & High Severity Bugs

### 1.1 `src/environment/ets_environment.py:2095–2113` — Carry-forward double-subtraction (Critical, Bug)
`old_carry_forward` is captured before `settle_compliance_realized()` runs, but is then subtracted again from holdings when computing the new bank. Since `settle_compliance_realized()` already surrenders the carry-forward obligation internally, holdings get debited **twice**.
- **Impact**: Agents with any carry-forward debt lose double the allowances; produces an artificial death-spiral.
- **Fix hint**: After settlement, recompute `bank = max(0, holdings_after_settlement - new_carry_forward)`; do not re-subtract the captured `old_carry_forward`.

### 1.2 `src/environment/company.py:756–765` — Carry-forward can compound exponentially (Critical, Bug)
The function sets `self._carry_forward = shortfall` where `shortfall = max(0, realized_emissions + old_cf - allowances_held)`. If the agent surrenders nothing, the new CF = old_cf + realized_emissions, then next year emissions are added on top again.
- **Impact**: Runaway, super-linear obligation accumulation, unbounded penalty payments.
- **Fix hint**: Apply available allowances first to old CF, then to current emissions; new CF should equal residual unmet obligation only after surrender, not the full unmet need.

### 1.3 `src/environment/ets_environment.py:1613, 1622, 1262` — Possible double-counting of defaulted volume (High, Bug)
Defaulted volume is added to `_defaulted_volume_pending` AND subtracted from `unsold` when computing rollover. The two paths must be mutually exclusive; if `allocations.sum()` already excludes defaulters (from `settle_auction`), the subtraction is redundant and could cause asymmetric accounting depending on settlement flag combinations.
- **Impact**: Either inflated supply or lost allowances over multi-year horizons; affects long-run scarcity.
- **Fix hint**: Trace `settle_auction()` allocation semantics; remove the `- defaulted_volume` term iff it's already excluded; document invariant.

### 1.4 `src/environment/company.py:246–258` — MAC fuel-switch cost unit mismatch (High, Bug)
`emissions_reduction` is in Mt and `mac_cost` is EUR/MWh, but the code multiplies them together to get a "M€" cost. Mt × EUR/MWh ≠ M€ — formula is off by ~1000× (and the scaling depends on emission factor).
- **Impact**: MAC switching is essentially free or wildly mispriced, distorting agent fuel-switching incentives.
- **Fix hint**: `cost_M€ = switched_mwh * mac_cost / 1e6`. Cross-check with how fuel costs are computed elsewhere.

### 1.5 `src/agents/ppo_agent.py:622–623, 927–928` — Advantage normalization NaN path (Critical, Bug)
Both `update()` and `compute_gae()` divide by `adv.std() + 1e-8`. When advantages are uniform (early training, strong shaping), std ≈ 0 and `1e-8` is too small relative to advantage scale, producing huge normalized advantages or NaNs.
- **Impact**: Silent NaN propagation → gradients explode → training collapse. The configured `gae_min_std` floor exists but is not applied here.
- **Fix hint**: Use `torch.clamp(adv.std(), min=self.gae_min_std)` everywhere advantages are normalized.

### 1.6 `src/agents/ppo_agent.py:696–710` — Pre-clamp of `log_ratio` corrupts PPO clipped objective (High, Bug)
`log_ratio` is clamped to `[-2, 2]` (so `ratio ∈ [0.135, 7.39]`) **before** the PPO clipped surrogate compares it to `(1-ε, 1+ε)`. The clamp short-circuits the clip and breaks the importance-ratio semantics.
- **Impact**: Algorithm is no longer standard PPO; trust-region guarantees void; can mute or amplify updates inconsistently.
- **Fix hint**: Remove the log_ratio clamp; if a numerical safety net is needed, use a far wider bound (e.g. ±20) only as a NaN guard.

### 1.7 `src/agents/ppo_agent.py:871–885` — Phase-aware reward normalization is non-causal (High, Logic Bug)
Auction- and secondary-rewards are normalized using full-episode statistics computed *after* the trajectory completes, then fed into GAE. Reward at year *t* is normalized using future years' rewards, breaking the Markov assumption GAE relies on.
- **Impact**: Biased advantage estimates; on-policy assumptions violated; harder convergence.
- **Fix hint**: Drop episode-wide normalization or use only causal/running statistics from a `RewardNormalizer` updated as you walk the trajectory.

### 1.8 `scripts/train.py:~1163` — `RewardNormalizer.normalize_reward()` value is discarded (High, Bug)
`agent.normalize_reward(rewards[i])` is called but the return value is not stored; the raw reward is what gets pushed into the buffer.
- **Impact**: The configured `reward.normalizer_alpha` is decorative — per-agent reward normalization claimed in the design doesn't actually affect training.
- **Fix hint**: Either store the normalized reward in the buffer or remove the normalizer (and config key) to avoid misleading future readers.

### 1.9 `scripts/train.py:611–612` — Python `random` not seeded (Medium-High, Risk)
Only `numpy` and `torch` are seeded. `random.seed()` is not called.
- **Impact**: Any use of `random.choice/random.random` (HPP swaps, dataloaders, shuffles) is non-reproducible — silently breaks seed-based replay.
- **Fix hint**: `import random; random.seed(seed)` at the same place.

### 1.10 `scripts/train.py:1437–1449` — Cosine LR decay only updated on update episodes (High, Bug)
LR decay is computed inside the `if is_update_episode:` block. When `episodes_per_update > 1`, LR jumps in steps instead of decaying smoothly per episode.
- **Impact**: Discrete LR jumps near update boundaries cause optimizer oscillation; deviates from intended cosine schedule.
- **Fix hint**: Move LR scheduler step out of the update gate so it advances every episode (or use total gradient steps as the schedule clock).

### 1.11 `src/environment/ets_environment.py:404–434` — Dynamic reserve mode reads non-existent config keys (High, Misconfiguration)
`_compute_dynamic_reserve` reads `reserve_discount` / `reserve_initial`, neither of which is defined in `configs/default.yaml` or `configs/smoke_100.yaml`. Both configs use `reserve_price_mode: "static"`, so the branch is dead — but enabling dynamic mode crashes with KeyError or silently uses code defaults.
- **Impact**: Dynamic reserve mode is broken; pretends to be a feature.
- **Fix hint**: Either add the two keys with documented defaults in YAML or delete the dynamic branch.

### 1.12 `configs/smoke_100.yaml` — Missing ~48 keys present in `default.yaml` (Critical, Misconfiguration)
Including: `auction.budget_price_clip`, all of `budget.treasury_reserve.*`, all of `opponent_obs.*`, `ppo.happo_dynamic_order`, `ppo.critic_compliance_features`, `reward.banking_signal.*`, `exploration.anchor_boost`, `ppo.entropy_decay_frac`, `pricing_curriculum.*`, `budget.dynamic_budget_ceiling_multiplier`, etc.
- **Impact**: Smoke test runs with implicit code defaults that may diverge from `default.yaml`. Smoke is therefore not a faithful proxy for production training; some new v8.x features (treasury, opponent obs mode, anchor_boost) are entirely uncovered. **Critic input dimension** changes (compliance features = 4 extra dims/agent) make checkpoint portability broken between configs.
- **Fix hint**: Either copy missing keys verbatim, or document explicit reliance on code defaults and add a runtime assertion that flags missing keys.

### 1.13 `configs/default.yaml:~105` — Bot arrays of length 8 with `n_bot_agents: 0` (High, Misconfiguration)
With bots disabled, the eight-element bot arrays (`bot_initial_mix`, `bot_reward_weights`, …) are dead. If a user later sets `n_bot_agents = 4` without truncating, only the first four entries are used and entries 4–7 silently become wrong defaults.
- **Impact**: Configuration foot-gun on ablations.
- **Fix hint**: Either truncate to length 0 or add a strong comment + length-vs-n_bot_agents assertion in code at load time.

### 1.14 `configs/default.yaml:558` vs `configs/smoke_100.yaml:400` — `phantom_bidder.enabled` disagrees (High, Inconsistency)
False in default, true in smoke. Phantom bidder represents ~40% of EU ETS volume. Production runs without phantom; smoke runs with it.
- **Impact**: Smoke and production exercise different supply-demand regimes; bugs in either path can hide from the other.
- **Fix hint**: Decide canonical setting; if both modes are valid, add a smoke variant for each.

### 1.15 `configs/default.yaml:527` vs `configs/smoke_100.yaml:375` — `exploration.mode` differs (High, Inconsistency)
`anchored` (WTP-centered Gaussian) in default vs `uniform` in smoke. The new anchored exploration is the production v8.4 feature but is never exercised by smoke tests.
- **Impact**: Anchored mode has zero CI coverage; bugs there only surface on full runs.
- **Fix hint**: Add a second smoke config that uses anchored mode.

### 1.16 `src/agents/ppo_agent.py:267` vs `configs/default.yaml:303` — `normalize_returns` default contradicts config (High, Inconsistency)
Code default `True`, YAML sets `False`. If config key is removed/renamed, behavior silently flips.
- **Impact**: Critic value scale changes; misconfigurations are silent.
- **Fix hint**: Align code default with intended config value (`False`); or assert key presence.

---

## 2. Medium-Severity Bugs and Risks

### 2.1 `src/agents/ppo_agent.py:906–908` — Hardcoded year-1/2 advantage floor
Advantages at indices 2 and 3 are floor-clamped to `-1.0`. This is a structural patch for early-year reward bias and breaks if `n_years` changes.
- **Fix hint**: Address root cause via per-timestep value baseline or normalized rewards; remove the magic indices.

### 2.2 `src/agents/ppo_agent.py:1366–1388` — HAPPO sequential order disrupted when buffers don't match
When `T_j != expected_T` (e.g., HPP swap cleared a buffer), agents are excluded from the M-factor product chain — breaking HAPPO's monotonic-improvement assumption.
- **Fix hint**: Either skip HPP swaps for the current worst agent, or include skipped agents with `M = 1`.

### 2.3 `src/agents/ppo_agent.py:983–988` — HAPPO weighted-advantage re-normalization wipes M-factor scale
Re-normalizing `(weighted_adv - mean) / std` after applying M removes the cooperative credit-assignment magnitude.
- **Fix hint**: Skip the renorm or use a softer scale (e.g., divide by mean abs).

### 2.4 `src/agents/ppo_agent.py:738, 1071` — KL anchor uses KL(new ‖ bc), not KL(bc ‖ new)
Forward KL leaves modes uncovered by the BC policy un-penalized; can permit drift away from BC support.
- **Fix hint**: Use reverse KL or symmetric.

### 2.5 `src/environment/ets_environment.py:1535–1536, src/environment/cap_schedule.py:450` — Two different inflation formulas
Phantom uses `_inflation_factor(year)` while MSR uses `(1+rate)**year`. They should coincide if `_inflation_factor` is correctly implemented, but maintaining two code paths invites drift.
- **Fix hint**: Single helper function used in all places.

### 2.6 `src/environment/cap_schedule.py:122 vs 307` — Hardcoded constant `833/1096` duplicates `TNAC_MID_OVER_UPPER`
Same numeric value repeated in observation logic; if reference constants change, observations desync from MSR logic.
- **Fix hint**: Import the named constant everywhere.

### 2.7 `src/environment/ets_environment.py:3060` — Obs preview shows pre-cap auction volume
`this_year_auction_volume` includes unsold + defaulted **before** `max_rollover_multiplier` is applied. Agents then observe a volume larger than the auction will actually offer.
- **Fix hint**: Apply the cap before exposing to obs, or document the discrepancy.

### 2.8 `src/agents/ppo_agent.py:759–760, 1091–1092` — Wasted critic backward when grads are non-finite
`backward()` runs first, then a finite check zeroes the grads. Cheaper to check `loss` finiteness before backward.

### 2.9 `scripts/train.py:1304–1313` — HPP buffer clear pollutes `agent_perf_ema`
After buffers are cleared for swapped agents, the EMA update still uses `total_rewards[i]` from policies that have been replaced. HAPPO dynamic order uses this EMA → wrong update sequence.
- **Fix hint**: Zero out swapped agents' rewards before the EMA update.

### 2.10 `scripts/train.py:~1341` — Buffer overflow risk
"Don't clear buffers between episodes within a batch" comment is correct but no upper bound exists if episode length swells.
- **Fix hint**: Add a soft cap that forces a flush.

### 2.11 `src/environment/ets_environment.py:2431–2433` — Secondary market allows agents to sell into non-compliance
The `max_sell` formula doesn't subtract obligation, so agents can short themselves and trigger penalties. May be intentional but undocumented.
- **Fix hint**: Add explicit comment, or add a strict no-short toggle.

### 2.12 `tests/test_compliance_validation.py:60–84` — Tautological test
Asserts bots don't default; bots are coded not to default. This validates bot policy quality, not environment compliance logic.
- **Fix hint**: Replace with a test that injects a deliberately non-compliant agent and checks penalty/CF accounting.

### 2.13 `tests/test_cap_schedule.py:17–36` — Tests use hardcoded values that diverge from `default.yaml`
E.g. `lrf_phase_switch: 5` here vs `2` in default. Logic is tested, but production config trajectory is not.
- **Fix hint**: Add an integration test that loads `default.yaml` and validates the first 12 years.

### 2.14 `tests/test_training_smoke.py:24–53` — Logging interval contradicts row-count assertion
`log_interval = max(10, n_episodes) = 10` with `n_episodes = 10` should log only ~once, but the test asserts `len(rows) == 10`.
- **Fix hint**: Verify what `log_interval` semantics actually are and align test/comment.

### 2.15 `tests/test_environment.py:73–98` — Invest action-mapping test only checks monotonicity
Doesn't verify the executed `invest_frac` is close to the action value.
- **Fix hint**: Add a numeric tolerance check.

### 2.16 Missing tests for new v8 features
- No `test_treasury_reserve.py` (treasury retention/decay/terminal NPV).
- `test_banking_signal.py` may not cover `imputed_cap_factor`, decayed shaping weight, or magnitude relative to other reward components.

### 2.17 `configs/scenarios/*.yaml` — `simulation.n_episodes: 1000` triggers `short_run_overrides` silently
Scenario users may not realize hyperparameters are overridden under `long_run_episode_threshold`.
- **Fix hint**: Document inside scenario YAML.

### 2.18 `configs/default.yaml:48` vs `configs/smoke_100.yaml:37` — `cap_overhead_pct` differs (10% vs 8%) AND default comment says "+12%"
Comment/value mismatch; smoke uses different scarcity regime.

### 2.19 `configs/default.yaml:45` vs `configs/smoke_100.yaml:36` — `initial_bank_fraction` 0.5 vs 0.10
Smoke starts tight, production starts loose — different MSR/banking dynamics.

---

## 3. Configuration Issues (Config↔Code, Cross-Config)

| # | Location | Issue |
|---|---|---|
| 3.1 | `configs/smoke_100.yaml` vs `default.yaml` | `bid_change_limit.value` 50.0 vs 75.0 (high — different exploration regime) |
| 3.2 | `auction.suspension_length` | 2 in smoke, 0 in default; smoke still exercises deprecated suspension |
| 3.3 | `exploration.epsilon_start/final/decay_frac/anchor_boost` | Present in default, missing in smoke → smoke drifts on undocumented code defaults |
| 3.4 | `ppo.critic_compliance_features` | Present (true) in default, absent in smoke — changes critic input dim, breaks checkpoint portability |
| 3.5 | `pricing_curriculum.*` | Present in default with `enabled: false`; missing in smoke. Dead by default. |
| 3.6 | `reward.budget_norm_budget_0` | Only used when `budget_norm_anchor: "fixed"`; default is `"dynamic"` → dead key. |
| 3.7 | `budget.dynamic_budget_ceiling_multiplier` | Defined in default, missing in smoke → smoke runs effectively uncapped budget. |
| 3.8 | Auto-scale sentinels | Mixed conventions: `0` in some keys, `-1` in others (`entropy_decay_start`). Confusing. |
| 3.9 | `price.initial_expected` | Set in scenario configs, but only read with hardcoded fallbacks (70/80) elsewhere; not in default.yaml. Hidden tunable. |
| 3.10 | `auction.carry_forward_defaults` | Boolean toggle that silently loses defaulted volume if false; no warning. |

---

## 4. Numerical Stability / Robustness Concerns

- **Advantage std → 0 path** (1.5 above): also re-occurs anywhere `(x - x.mean()) / x.std()` is used without min-clamp.
- **`adv.std() + 1e-8`** is too tight for typical advantage scales (often ~tens to hundreds in M€-denominated rewards). Use `gae_min_std`.
- **Phase-specific reward normalization** uses non-causal episode statistics (1.7 above).
- **`market_clearing_ets.py:122–137`** — when *all* bids fall below reserve, reserve price is returned as clearing price. If `_reserve_anchor == "auction"`, this can pollute the price MA3. Verify the upstream gate at `ets_environment.py:1637` actually skips appending on `auction_failed`.
- **`company.py:428`** — defensive `+ 1e-6` in capacity-factor denominator: never triggers in practice, but masks any future bug that would zero the mix.
- **`ppo_agent.py:467–469, 522–523`** — epsilon-greedy random actions still go through the policy network to compute log-prob; minor compute waste.
- **CUDA detach/CPU transfer (`train.py:~1402-1404`)**: cumulative HAPPO ratio is moved to CPU each step.
- **No NaN/Inf guard** on observations or rewards before they enter the buffer (skim `ets_environment.step_*`); a single bad year poisons GAE.

---

## 5. Tests Issues

| # | Test | Issue |
|---|---|---|
| 5.1 | `test_tabula_rasa.py:52–59` | Doesn't verify `default.yaml` itself has `enabled: false`. Companion test partially covers this. |
| 5.2 | `test_training_smoke.py:24–53` | Log-interval semantics unclear — assertion `len(rows) == n_episodes` may be wrong. |
| 5.3 | `test_compliance_validation.py:60–84` | Tautological (bots designed not to default). |
| 5.4 | `test_heuristic.py:72–83` | Tests heuristic in isolation; doesn't exercise Company integration. |
| 5.5 | `test_environment.py:73–98` | Only checks monotonicity, not numeric correctness of invest mapping. |
| 5.6 | `test_market_clearing.py:89–100` | Doesn't assert `total_allocated == 1.0` for under-subscribed case. |
| 5.7 | `test_bid_change_limit.py:48–53` | Hardcoded `bcl_value=50.0` doesn't reflect production default 75.0. |
| 5.8 | `test_phantom_bidder.py:20–61` | Verifies MA3-independence but not that prices stay within `[reserve, penalty]`. |
| 5.9 | `test_bot_features.py:88–100` | Bot fade ordering not asserted by ID. |
| 5.10 | `test_rewards.py:143–150` | `test_collateral_cost_logged_matches_formula` should compute expected cost from first principles, not just check >0. (Verify.) |
| 5.11 | Missing tests | `treasury_reserve`, full `banking_signal` coverage, `anchored exploration mode`, dynamic-order HAPPO. |

---

## 6. Documentation/Comment vs Code Mismatches

- **`configs/default.yaml:48`** — comment "+12% cap overhead" but value is `0.10` (10%).
- **`configs/smoke_100.yaml:8`** — header comment "no BC, no KL anchor" but `pretrain.enabled` is just set to `false` (BC is supported, just disabled).
- **`configs/smoke_100.yaml:237`** — "auto: 40% of n_episodes" stale; actual fraction comes from `shaping_decay_frac` (0.10 in default).
- **`src/environment/phantom_bidder.py:97`** — `price_ma3` parameter docstring says "deprecated and ignored", still passed by environment at line ~1539.
- **`src/environment/ets_environment.py:2432`** — "no-short-selling" comment is misleading; the formula does allow selling into non-compliance.
- **`scripts/train.py:833`** — `EntropyConditionTracker` name implies condition-based logic; implementation is purely time-based.
- **`src/agents/ppo_agent.py:156`** — `seed if seed is not None else 0 + agent_id` — operator precedence trap (parses as `seed if seed is not None else (0 + agent_id)` which works but reads wrong).

---

## 7. Low / Nit Observations

- **`src/environment/ets_environment.py:750`** — burn-in queue seeding doesn't pass `current_year`, so inflation factor defaults to 1.0 instead of the burn-in (negative) year value.
- **`src/environment/ets_environment.py:1713–1733`** — `compute_investment_cost` recalculated 3× per branch after `invest_frac` rescaling. Risk of formula drift; consider a helper.
- **`src/agents/ppo_agent.py:223 / 197–198`** — raw action anchors clipped to `[-0.95, 0.95]`; arbitrary and reduces initial exploration coverage.
- **`scripts/evaluate.py:96`** — `select_auction_action(..., deterministic=True)` doesn't explicitly pass `epsilon=0.0`; relies on default. Train-eval gap risk.
- **`src/agents/ppo_agent.py:632–655`** — critic warmup loop duplicates main-loop minibatch iteration; maintenance burden.
- **`configs/scenarios/*.yaml`** — `n_episodes: 1000` activates `short_run_overrides` silently.
- **`reward.budget_norm_budget_0`** — dead key when `budget_norm_anchor: "dynamic"`.
- **Auto-scale conventions** — `0` vs `-1` sentinel inconsistency.
- **No checkpoint schema versioning** — silent breakage when critic input dim changes (e.g. `critic_compliance_features` toggle).
- **Resource cleanup** — no obvious file-handle close pattern in CSV logging in `train.py`; relies on `with` blocks (not verified at every call site).

---

## Summary Counts (deduplicated)

| Severity | Count |
|---|---|
| Critical | 6 |
| High | ~13 |
| Medium | ~18 |
| Low / Nit | ~25 |

**Top priorities to investigate first**:
1. Carry-forward double subtraction (1.1) and exponential CF accumulation (1.2) — directly corrupt compliance accounting.
2. PPO log-ratio pre-clamp (1.6) and advantage normalization NaN path (1.5) — algorithmically unsound.
3. Phase-aware reward normalization breaks GAE causality (1.7) — biases all training updates.
4. Discarded `RewardNormalizer` output in `train.py` (1.8) — reward normalization is currently a no-op.
5. Cosine LR decay only on update episodes (1.10) — discrete LR jumps.
6. MAC fuel-switch cost units (1.4) — investment incentives off by ~1000×.
7. Smoke ↔ default config divergence (1.12, 1.14, 1.15, plus §3) — smoke is not a faithful production proxy.
8. Possible defaulted-volume double-count (1.3) — long-horizon supply drift.
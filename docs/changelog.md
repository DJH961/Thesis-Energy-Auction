# Changelog

All notable changes to the ETS-MARL simulator (`ets_marl_happo_current`).
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
the project follows semantic-ish versioning tied to `pyproject.toml`,
and entries focus on what each release means for the simulation —
the market, the agents, the reward, the training loop — not on
documentation or notebook bookkeeping.

The pre-rewrite, exhaustive history (including documentation/notebook
churn) is preserved in [`changelog_archive.md`](changelog_archive.md).

---

## 8.6.5 — Faster checkpoint archiving

End-of-training checkpoint compression switches from `tar.xz` to
`tar.zst`. On a real seed that's roughly fifteen minutes down to
thirty seconds with the same ~1.7 GB on-disk footprint, which adds up
fast on full sweeps.

- `compress_checkpoints` now picks zstandard when the optional
  `zstandard` package is importable, falls back to `tar.gz` (level 1)
  on the stdlib, and still produces `tar.xz` on explicit request for
  cold-archival use.
- `decompress_checkpoints` sniffs the suffix and dispatches
  automatically, so old `.tar.xz` archives keep working untouched.
- New knob: `logging.compress_on_finish.checkpoints_codec` (default
  `"auto"`); CLI flag `--codec` on `scripts/compress_results.py`.
- `zstandard>=0.22.0` added to `requirements.txt` and the bootstrap
  self-heal probe so Azure ML curated environments pick it up.

No migration needed.

---

## 8.6.4 — Logs and checkpoints compress themselves on clean finish

Disk usage was getting out of hand on multi-seed sweeps. Logs and
checkpoints are now compressed automatically when `train_one_seed`
finishes cleanly: a 5 GB year-log shrinks to roughly 0.5–1 GB of
zstd-parquet, and a 2.5 GB checkpoint directory becomes a 0.5–1.5 GB
`.tar.xz`. A four-seed sweep drops from ~22 GB to ~3–5 GB.

- Persistent parquet cache for training/year logs in
  `src/utils/run_data.py`. Streaming CSV→parquet build (chunked at
  250k rows so memory stays bounded), zstd compression, atomic
  `tmp`→`rename`, `(size, mtime_ns)` cache key.
- `load_run_csv` and `glob_run_logs` transparently fall back to the
  parquet sibling once the CSV is deleted, so analysis code doesn't
  notice.
- New knobs: `logging.compress_on_finish.{logs, checkpoints,
  delete_csv, delete_checkpoint_dir}`, all default `true`.
- New script: `scripts/compress_results.py` — idempotent migration for
  existing results trees; `--dry-run`, `--keep-source`, `--logs-only`,
  `--checkpoints-only`.
- `decompress_checkpoints(archive)` ships as the symmetric helper.
- `configs/sweeps/thesis_experiments.yaml` — 16-job thesis sweep
  (5 structural variants × 3 seeds + 1 high-inflation cell on seed
  1729).

Run `scripts/compress_results.py` once to back-fill an existing
results tree; no retraining required.

---

## 8.6.3 — Seed stability across config variants

Same seed, same simulation — even when you change the agents'
behaviour, the LRF, the MSR, or the reward weights. Previously every
random draw funnelled through one shared RNG, so anything that
shifted the consumption order silently shifted subsequent emission
shocks, capacity-factor noise, AR(1) draws, and inflation paths.
Cross-variant comparisons now hold the world fixed.

- `ETSEnvironment` builds named sub-streams via
  `np.random.SeedSequence(seed).spawn(...)` for inflation, shocks,
  AR(1) prices, bot heterogeneity, urgency, warm-start draws,
  opponent-obs noise, auction tiebreak, and phantom bidder draws.
- One independent generator per `Company` for action-conditional
  draws (investment success, jitter delay, cancellations).
- `self.rng` and `self._env_rng` survive as backward-compatible
  aliases pointing at the shock stream.
- New regression tests pin invariance of inflation/shock paths
  across action and config differences.

Numerics for a given seed change versus 8.6.2 — re-run experiments
for clean comparisons.

---

## 8.6.2 — Heuristic bots brought back to parity

The heuristic bot policy had been left behind when the agent space
moved to PPO/HAPPO; it now runs on the same calibration as the
learning agents and ships with a config + driver to use it as a
no-learning credibility floor.

- Bots size their auction cash buffer the same way the env-side
  budget gate does (`operating + treasury_fraction × treasury`,
  default 0.5), so they no longer ignore the corporate treasury.
- Capex throughput reads `effective_capex_throughput` (loan-squeeze
  and revenue-modulation aware); green/financial classification reads
  `company.w_green` instead of inferring from agent-id parity.
- Secondary-market sell-side floor relaxed from `0.3 × (penalty −
  anchor)` to `0.10 ×` for non-urgent sellers — the old floor parked
  surplus 30% above market and suppressed secondary volume.
- `ETSEnvironment` now actually passes `loan_outstanding_norm` to
  bots; the loan-aware code paths in the heuristic were dead-coded
  before.
- New `configs/bots_only.yaml` mirrors `default.yaml` with
  `n_agents=0, n_bot_agents=8`.
- New `scripts/run_bots_only.py` — multi-seed driver, writes logs in
  the PPO column superset (per-agent reward, green fraction,
  shortfall, penalty, compliance rate, `quality_score`).

PPO runs with `n_bot_agents = 0` are bit-for-bit unchanged.

---

## 8.6.1 — Richer diagnostics and a signed quality score

A logging-coverage release. The training pipeline now exposes a much
richer set of per-year and per-episode diagnostics that previously
had to be reconstructed (often imperfectly) from holdings deltas, and
the headline `quality_score` moves to a signed `[-5, +5]` so
run-to-run progress is actually readable.

**Year-log additions** (`year_log_*.csv`)
- Auction internals flattened from `auction_stats`:
  `auction_total_demand`, `auction_unsold`, `auction_hhi`,
  `auction_max_agent_share`, `auction_failed`, `auction_defaults`,
  `auction_defaulted_volume`, `effective_reserve_price`.
- Secondary participation counts: intent vs executed, buyers vs
  sellers.
- Exogenous state: `common_emission_shock` (system-wide η_t × σ
  before idiosyncratic mixing), `fundamental_anchor`.
- Per-agent compliance trail: `carry_forward_start_A{i}`,
  `carry_forward_end_A{i}`, `coverage_gap_A{i}`,
  `effective_penalty_rate_A{i}`.
- Per-agent credit state: `treasury_reserve_A{i}`,
  `treasury_drawn_A{i}`, `loan_outstanding_A{i}`.

**Training-log additions** (`training_log_*.csv`)
- `year0_tnac`, `yearT_tnac`, `ep_total_unsold`,
  `ep_auction_failures`, `ep_total_defaults`,
  `peak_loan_outstanding_A{i}`, `peak_carry_forward_A{i}`,
  `final_treasury_reserve_A{i}`.

**Quality score**
- `quality_score` is now signed in `[-5, +5]`. Components
  (`Q_compliance`, `Q_price_realism`, `Q_saved_carbon`, `Q_cost_eff`,
  `Q_volatility`) keep their `[0, 1]` semantics; the top-level score
  rescales them via `2x − 1` (or `1 − 2v` for volatility), weighted
  `{0.30, 0.25, 0.25, 0.10, 0.10}`, and multiplied by 5.
- `compute_quality_score` factored into `src/utils/quality_metric.py`
  and unit-tested.

**Q-learning baseline**
- Re-aligned with the PPO trainer's environment surface: multi-seed
  driver (`--seeds`), `--run-tag`, year-log column superset
  overlapping the PPO schema, the same anchor-invariant
  `quality_score`, bot-aware logging.

**Snapshots auto-cleanup**
- Mid-run snapshot pairs in `results/snapshots/` are deleted at clean
  end-of-run (the live cumulative log strictly supersedes them).
  Opt out via `logging.snapshot_delete_on_finish: false`.

Older logs stay readable; new columns gracefully NaN-fill.

---

## 8.6.0 — Saved-carbon ESG, joint budget gate, episode quality metric

The big one. The ESG reward is rebuilt around a saved-carbon hybrid,
the auction bid sizing gets a proper last-line-of-defence budget
gate, and every episode now produces an anchor-invariant quality
score.

**ESG reward — saved-carbon hybrid.** Replaces the prior linear
horizon penalty. The new shape is

```
esg_raw = scale × ( stock_w × ef_ratio
                  + flow_w  × ef_ratio × (anchor_real_t / anchor_real_0)
                  + speed_coef × max(0, Δgreen) )
```

The stock term pays sustained green share every year; the flow term
monetises avoided carbon at the live anchor (a social shadow price);
the speed term is uniform across the episode (no front-loading). The
compliance gate is retained. `esg.scale=0.50` is calibrated for a
~50/50 financial-vs-ESG split (in absolute per-(year, agent)
magnitude) for `[w_cost=0.5, w_green=0.5]` agents on an
anchor-tracking rollout; a regression test pins balance to
`[35%, 65%]`.

**Joint budget gate — two-stage.** Bid quantity is sized against
*expected* settlement cost (uniform clearing × allocation +
collateral), not the agent's own `bid_p × bid_q`, so a high
willingness-to-pay no longer artificially shrinks qty in expectation.
When expected settlement exceeds available cash, the gate runs:

1. Shrink `bid_q` toward `need` (compliance floor — never below
   `need` if the original was ≥ `need`).
2. If still over budget, reduce `bid_p` toward
   `max(reserve, MA3_inflated)`. Below that floor, lowering price
   only loses the auction in expectation.
3. Last resort: shrink `bid_q` below `need`.

Cash buffer is `operating + treasury_fraction × treasury`
(`treasury_fraction = 0.5`) — treasury absorbs price spikes and is
only partially exposed to routine sizing. Emergency-loan headroom is
opt-in (`include_loan_headroom = false`); the loan is a
settlement-time safety net, not a sizing buffer.

**U/D/M/B/C compliance attribution.** Mutually-exclusive partition of
non-compliant years per agent: `U` (allocation < emissions, no
inherited debt), `D` (allocation ≥ emissions, inherited carry-
forward — the debt cascade), `M` (both). `B / C` are
compliant-with-stress: bank-covered / sec-covered. Written as
`udbc_{U,D,M,B,C}_total_A*` in the training log.

**Quality metric.** Per-episode `quality_score`, `Q_compliance`,
`Q_price_realism`, `Q_saved_carbon`, `Q_cost_eff`, `Q_volatility`
written to `training_log_*.csv`. Volatility uses `std/mean` of the
**clearing/anchor ratio** so a trajectory that perfectly tracks the
fundamental scores ≈ 0; saved-carbon is monetised at the per-year
fundamental anchor. Computed on the tail window, never fed back into
training.

**Observation.** New `obs[43] = compliance_affordability =
(need × expected_clearing) / cash`, clipped `[0, 3]` then divided by
3, with `expected_clearing = max(reserve, MA3, anchor)`.
`obs_dim_phase1 = 44`.

**Reward bid-head fair-price baseline.** `compliance_norm_excess =
(auction_cost + mac + collateral − baseline_cost) / compliance_denom`
with `baseline_cost = need × last_clearing_price / infl`. Buying
exactly `need` at the clearing price is reward-neutral; the gap
penalty is priced at a remediation rate that mixes the effective
penalty, the secondary EMA, and the anchor.

**Calibration changes carried into the default config.**
- `risk.p_fail_max`: 0.65 → 0.40 (aligns with policy-supported FID
  risk for offshore wind / solar).
- `budget.dynamic_budget_ceiling_multiplier`: 1.5 → 2.5.
- `ets.unsold_to_msr` flipped to `true` by default.

Re-train recommended for any `w_green > 0` agent.

---

## 8.5.6 — Per-agent RNG and an opt-in anchor snap

Two targeted fixes for seed-driven price-basin lock-in. The
exploration RNG was global, so agents' `ε`-greedy draws were
correlated; and once a policy fell into a low-price basin there was
no mechanism to climb back out. Both are now addressed.

- `PPOAgent` carries `self._rng`; the eight `np.random.*` calls in
  the auction/secondary action heads now route through it.
- New `snap_price_head_to_anchor()` / `restore_price_head()` give a
  probabilistic per-episode reset of the price head toward the
  fundamental anchor. Off by default
  (`exploration.anchor_snap.enabled: false`,
  `prob_per_episode: 0.02`).

---

## 8.5.5 — Disk usage on long sweeps stays bounded

No training, numerical, or RNG-path changes. Just stops sweeps from
filling the disk.

- New `logging.snapshot_keep_recent` (default 1): after each snapshot
  copy, older pairs for the same `(run_tag, seed)` are deleted, so
  snapshots become a rolling pair instead of a cumulative pile.
- New `logging.checkpoint_pruning.online` (default `true`):
  `prune_checkpoints` runs after every periodic `.pt` save, capping
  on-disk checkpoint count at all times. `agent_*_best.pt` and the
  episode-0 baseline are always preserved.

CSV content and RNG paths are byte-identical to 8.5.4.

---

## 8.5.4 — Sweep launcher and architecture-aware threading

Hand-launched per-variant runs replaced by a structured sweep
launcher, and CPU thread counts are now picked based on the host
instead of the PyTorch defaults.

- `src/utils/compute_setup.py` — `detect_architecture()` and
  `configure_compute()` set `torch.set_num_threads`, OMP/MKL/OpenBLAS
  env vars, and `KMP_BLOCKTIME=0`. Auto policy: ≤2 cores → all,
  3–4 → 2, 5–32 → 4, 33+ → 6. Override with `ETS_NUM_THREADS`.
- `scripts/train.py` — calls `configure_compute()` at startup; new
  `--parallel-seeds N` (`ProcessPoolExecutor`); new `--run-tag`
  prefixes all output filenames.
- `src/utils/sweep.py` — spec schema, `deep_merge` (list-replace),
  dotted-path expansion, variant-config resolution, cartesian job
  materialisation.
- `scripts/sweep.py` — process-pool launcher; per-job stdout
  captured to `.log`; background heartbeat thread emits a structured
  one-liner per running job from a tail-read of the training CSV.
  `--dry-run` supported.
- New tests: `test_compute_setup.py`, `test_sweep.py`,
  `test_sweep_launcher.py`.

`--run-tag` is a no-op when omitted; single-config runs are
byte-identical.

---

## 8.5.3 — Bid head no longer rewarded for floor-bidding

The fair-price baseline was retired in 8.5; this puts it back so that
buying exactly the compliance need at the clearing price is
reward-neutral and over/under-bidding decay symmetrically. The
secondary-price proxy used by the gap penalty was also too noisy and
is now an EMA, capped at 1.5× the effective penalty.

- `compute_auction_rewards()` —
  `compliance_norm_excess = (auction_cost + mac + collateral −
  baseline_cost) / compliance_denom`, with `baseline_cost = need ×
  clearing_price / infl`.
- Gap penalty rate now
  `min(max(eff_pen, sec_ema, anchor), cap_mult × eff_pen)`. The
  `sec_ema` (alpha 0.30) updates only on non-zero secondary volume.
  New config block: `reward.sec_proxy.*`.
- The year-0 BCL exemption is removed; the bid-change limit is now
  active in year 0 anchored on `max(price_ma3_early,
  fundamental_anchor) ± value`.

---

## 8.5.2 — Coverage-gap pricing and per-sub-head KL early-stop fix

The bid head was under-pricing forced secondary buying when coverage
gaps occurred, and the per-sub-head KL early-stop allowed one
auction sub-head to mask the other's trust-region violation.

- `compute_auction_rewards()` — `gap_penalty` rate is `max(eff_pen_rate,
  last_secondary_price, anchor_t) / infl` deflated to real terms; new
  diagnostic `expected_remediation_rate_real`.
- `update_happo()` with `split_invest_head=True`: KL early-stop uses
  `kl_auc_eff = max(kl_bid_mb, kl_inv_mb)` and
  `mb_kl = max(kl_auc_eff, kl_sec)`, so neither sub-head can hide
  behind the average.

---

## 8.5.1 — Dual-clip PPO

Actor losses were spiking to ~`1e8` in training logs. Dual-clip PPO
(Ye et al. 2020) bounds the surrogate when advantages are negative
and ratios are extreme.

- New `_ppo_clipped_loss(ratio, adv)` helper wired into all five
  surrogate sites; `c·adv` floor for negative-advantage rows.
  `log_ratio` clamp tightened from `±20` to `±10`.
- New config: `ppo.dual_clip_c=3.0`, `ppo.log_ratio_clip=10.0`.
  Schedule retune: `entropy_coef_final` 0.005 → 0.015,
  `log_std_min` −3.5 → −3.0, `n_epochs` 5 → 3.
- New per-agent `Why(U/D/B/C)` column in the year log partitioning
  each year into under-bought / debt-cascade / bank-covered /
  sec-covered.
- New `tests/test_dual_clip_ppo.py`: positive-adv unchanged,
  negative-adv bounded, vanilla recovered at `c=0`, finite loss
  under extreme ratios.

---

## 8.5 — Phase-1 split into bid and investment sub-heads

Phase-1 actions used to share an actor. The bid sub-head and the
investment sub-head now train against independent advantage streams
and have their own value networks, which removes a long-standing
gradient interference between compliance bidding and capacity
investment.

- `ppo.split_invest_head=true`. Bid sub-head (action dims 0–1) and
  invest sub-head (dims 2–5) each have their own causal phase
  reward normaliser; invest sub-head gets its own `value_net_invest`.
- `compute_auction_rewards()` returns `(joint, bid, invest)` triple.
- Action mapping: `auction_actions[i, 3:6]` is now a softmax split
  across the three green technologies
  (`investment.tech_softmax_temperature=1.0`).
- Per-tech cancellation rates in `construction_jitter.p_cancel_per_tech`
  replace the single scalar.
- `Company.effective_capex_throughput` scaled by
  `clip(0.7 + 0.3 × revenue_t / baseline_revenue, 0.5, 1.5)`.
- Terminal queue NPV is a proper DCF annuity over
  `reward.terminal_asset_lifetime_years=20` at WACC; terminal bank
  collapses to a single discounted hold formula.
- ESG signal centred on a linear baseline:
  `esg_raw = scale × ((ef_ratio − year/n_years) + speed_bonus)`.
- HAPPO ordering: `ppo.happo_order_metric` default switches to
  `"advantage"`.

`investment.discount_rate` raised to 0.05; existing checkpoints are
incompatible with split-head loading.

---

## 8.4.2 — Multi-file bug sweep

Numerics, GAE normalisation, cap-schedule effective-penalty, and a
gap-penalty denominator that was actively rewarding agents for
leaving coverage gaps.

- PPO: advantage std floor uses `torch.clamp(std, min=gae_min_std)`;
  `log_ratio` clamp widened `±2 → ±20`; causal per-phase reward
  normalisation in `compute_gae()`; `normalize_returns` default
  aligned to YAML (`False`).
- Reward: `gap_penalty` denominator changed from `budget_real` to
  `compliance_denom` (now matches `compliance_norm` — missing 1 Mt
  is strictly more expensive than buying it). Budget gate scales
  `bid_q` smoothly instead of hard-zeroing. The
  `coverage_frac_auction` gate is applied only on the savings branch,
  not the cost branch.
- Config: `banking_signal.imputed_cap_factor` 2.0 → 5.0;
  `reserve_discount`, `reserve_initial`, `price.initial_expected`
  added.
- Trainer: cosine LR decay advances every episode; Python `random`
  is now seeded alongside numpy/torch; `agent_perf_ema` skips
  HPP-swapped agents.
- `configs/smoke_100.yaml` synced with all non-`tabula_rasa` keys
  from `default.yaml`.

---

## 8.4.1 — PCL anchor floor and decoupled actor optimisers

Three independent fixes whose combined effect is calmer training:
the price-change-limit reference no longer drifts below the
fundamental anchor, the budget price clip becomes its own observable
signal, and the auction and secondary actors stop fighting over a
shared optimiser state.

- PCL reference is now
  `max(price_ma3, compute_fundamental_anchor(year, config,
  cap_t_actual=cap_t))` using the real MSR-adjusted cap.
- New obs dim `[40] = last_budget_price_clip`; obs dims [40]–[41]
  shift to [41]–[42]. `obs_dim_phase1` 42 → 43.
- `PPOAgent` carries separate `auction_optimizer` and
  `secondary_optimizer`; `actor_optimizer` is kept as a backward-
  compatible alias. Independent zero/clip/step per head.
- `compute_gae()` clamps year-1 advantages to a floor of −1.0
  pre-normalisation; returns `per_year_adv_mean` in `buf_tensors`.
- Big hyperparameter retune: `ppo.critic_lr_decay: cosine`,
  `critic_lr_min: 0.00005`, BCL 50 → 75, `mini_batch` 32 → 64,
  `episodes_per_update` 16 → 32, `clip_eps` 0.20 → 0.15, `target_kl`
  0.02 → 0.015, etc.

`load()` falls back to the old `actor_optimizer` key for legacy
checkpoints.

---

## 8.4 — Clip-feedback observations and a fixed BCL

Agents now see when their actions get clipped — by the bid-change
limit, by the budget price clip, by the qty clip, or by the
investment clip — instead of silently learning around invisible
constraints. The bid-change limit itself is also simplified.

- Phase-1 base obs grows 40 → 42: `pcl_headroom_norm`,
  `last_bid_price_clip`, `last_bid_qty_clip_ratio`,
  `last_invest_clip_ratio`. Phase-2 base grows +11 → +12 with
  `last_sec_qty_clip_ratio`. All reset to 0/1 at episode start.
- BCL restructured to a single `auction.bid_change_limit.{enabled,
  value}` key. Fixed 50 EUR/t, MA3-referenced (not last clearing
  price). Decay schedule removed; `set_bid_change_limit()` becomes
  a no-op.
- `compute_fundamental_anchor` gains a `cap_t_actual` parameter to
  bypass the LRF approximation; the env passes the actual cap at all
  three call sites.
- `tests/test_bid_change_limit.py` — 26 new tests.

Old `bid_change_limit_start`/`final` config keys replaced by the
nested `bid_change_limit:` block.

---

## 8.3.2 — MSR warm-start and BCL prelude *(folded into 8.4)*

- `_run_burnin()` seeds the MSR reserve to `msr_initial_reserve_frac
  × cap_year_0` (default 0.23), clamped to `tnac_lower`.
- Per-episode BCL decaying `bid_change_limit_start →
  bid_change_limit_final`; year 0 always unconstrained; clipping
  reference and limit exposed as obs dims [38]/[39].
  `obs_dim_phase1` 38 → 40 base.
- ESG `speed_coef` linearly interpolated `esg.speed_coef →
  esg.speed_coef_late` (new key, default 0.8).
- Exploration anchor now `compute_fundamental_anchor(current_year)
  × anchor_boost`; new `exploration.anchor_boost: 1.14`.

---

## 8.3.1 — Penalty denominator fix (the one that mattered)

For long enough that the bug had its own folklore, `penalty_realized`
and `remediation_cost` were normalised by `budget_real` while
compliance costs were normalised by `compliance_denom`. Outcome:
non-compliance was cheaper than compliance for some agents. Now
fixed.

- `penalty_realized` and `remediation_cost` denominators changed
  from `budget_real` to `infl × compliance_denom`. Penalty is now
  ~3.4× larger relative to compliance costs (penalty_rate 138.75 vs
  anchor ~80).
- Two cap-scarcity lookahead obs dims added: `[36]
  cap_ahead_3y_ratio`, `[37] cap_ahead_6y_ratio` (cap(t+k)/cap(t),
  clipped [0, 1]). `obs_dim_phase1` 36 → 38 base.
- All decay schedules now fraction-based via config:
  `epsilon_decay_frac=0.20`, `entropy_decay_frac=0.30`,
  `shaping_decay_frac=0.10`. Exploration mode default flips
  `"uniform"` → `"anchored"`. `epsilon_final` 0.05 → 0.03.

Penalty magnitudes change; re-training recommended.

---

## 8.3.0 — Banking signal closes the zero-bidding exploit

Agents had learned to bid zero, ride the bank, and never participate
in price formation. The reward now marks the bank to market: drawing
down counts as buying at today's clearing price, with a separate
timing P&L term that rewards buying cheap and drawing when prices
are high.

- `compliance_norm = compliance_norm_cash + w_imputed ×
  imputed_bank_norm`, where `imputed_bank_norm = min(drawdown ×
  clearing_price / infl, cap) / compliance_denom`.
- `banking_signal = w_banking × drawdown × (clearing_price −
  cost_basis) / (infl × compliance_denom)`.
- New per-agent `_bank_cost_basis[i]` array (weighted-average
  acquisition price), initialised at `fundamental_anchor(0) × 0.80`
  and updated after every auction/secondary purchase.
- Config: `reward.banking_signal.{enabled, w_banking, w_imputed,
  imputed_cap_factor, initial_bank_cost_factor}`.
- `tests/test_banking_signal.py` — 20 new tests.

Existing checkpoints behave reasonably but should be re-evaluated
under the new signal.

---

## 8.2.0 — Reward audit

Six bugs ranging from penalty double-counting to ESG over-multiplication
to phase-gradient imbalance. The reward function now does what the
docstring says it does.

- `penalty_realized` split into a real-terms realised payment plus a
  prospective `remediation_cost = carry_forward × next-year anchor ×
  scarcity`.
- `capital_norm` denominator changed `compliance_denom → budget_real`.
- `esg_anchor_ratio` removed as ESG multiplier (fixed at 1.0); ESG
  compliance gate linearised (`coverage_frac`, not `coverage_frac²`).
- `cost_norm` centred at 1.0 so a fair purchase is reward-neutral.
- `compute_auction_rewards()` normalisation aligned with
  `_compute_rewards`: phase-1 and phase-2 gradient scales are now
  proportionally consistent.
- Terminal queue guard smoothed: `remaining_scale = min(1.0,
  remaining/2.0)` replacing the hard `< 1.0` cliff. Terminal bank
  is piecewise (linear below need, log above).
- `loan_sting` normaliser fixed to `budget_real` (was nominal
  `annual_budget`).
- `esg.scale` raised to 2.0.

Reward scale changes substantially; re-train recommended.

---

## 8.1.1 — Inflation-invariant reward

Compounded inflation was leaking into the reward across years,
making the same physical decision look better or worse purely as a
function of when it happened. All cost buckets are now in real
terms.

- `compliance_cost_real`, `capital_cost_real`, `soft_penalty_real`
  divided by `company.inflation_factor(current_year)` before
  normalisation; `budget_real = annual_budget / infl`.
- Prospective penalty: `shortfall × penalty_rate × (1 + scarcity_t) ×
  urgency_scalar / budget_real`, with `scarcity_t = max(0, 1 −
  cap_t / cap_0)`.
- ESG: `esg_speed_coef × max(0, green_frac − prev_green_frac)` added
  inside `esg_raw_unanchored`. `esg_anchor_ratio =
  min(compliance_denom / budget_real, 2.0)` caps ESG for small-budget
  agents.
- Fixed `REWARD_SCALE = 1000.0` divisor removed; terminal bank/queue
  values scaled by `budget_real_t`.
- `_last_reward_channels` extended with 11 new diagnostic fields.

---

## 8.1.0 — Treasury, emergency loans, anchor-normalised cost reward

A major overhaul of the financial side of the simulation. Companies
now keep a corporate treasury, can take an emergency loan to settle
auctions, and the cost reward is normalised by what it actually
costs to comply at the live anchor.

- `Company` retains 60% of unspent budget as a treasury (1.5× cap,
  5% annual decay, terminal NPV).
- Emergency loan with capex covenant squeeze
  (`effective_capex_throughput`).
- `cost_norm = total_cost / (anchor_t × estimated_need)`.
- `ETSEnvironment` — 7-D lagged opponent obs (emissions, green_frac,
  fossil_frac, queue_noisy, bank_norm, net_secondary_norm,
  `lagged_compliance_gap_norm`) with a one-year two-buffer lag.
- Suspension replaced by a budget-based bid gate (cash <10% notional
  → qty zeroed). Canonical settlement waterfall:
  operating → treasury → loan → default.
- Phase-1 obs: `36 + 7 × (N − 1)` dims; Phase-2 obs: Phase-1 + 11
  dims; budget price clipping (soft 1.5× max affordable).

Old checkpoints are incompatible (obs dims change).

---

## 8.0.1 — EU ETS 2026 calibration

`market_calibration` was anchored to a stylised state that no longer
matches reality, and the MSR cancellation threshold drifted with the
auction volume instead of holding to the legislative TNAC band. Both
fixed.

- `_apply_msr` cancels holdings above `self.tnac_lower` (was
  `max(prev_auction_vol, prev_cap)`); same fix in
  `preview_auction_volume`.
- Recalibrated to EU ETS 2026 actuals: `msr.price_release_enabled:
  false`, `msr.price_containment_absolute` 350 → 175,
  `msr.price_release_absolute` 450 → 212, `ets.cap_overhead_pct`
  0.02 → 0.12, `msr.tnac_upper_ratio` 0.36 → 0.68,
  `ets.initial_bank_fraction` 0.10 → 0.81 (TNAC starts at
  1.05 × tnac_upper).
- `budget.dynamic_budget_ceiling_multiplier: 1.5` caps
  `compute_dynamic_budget` output.
- Burn-in MSR reserve clamped to `tnac_lower` after the burn-in
  loop.
- `_collateral_clip_events` surfaced in the trainer's Warnings line.

---

## 8.0.0 — Tabula-rasa retired, fundamental anchor everywhere

Cold-start training without prior knowledge had become more of a
research curiosity than a useful default. The block stays in config
for ablation reference but enabling it now raises an error. The
default trainer instead seeds the price head from the fundamental
anchor and uses that anchor as the AR(1) floor.

- `tabula_rasa.enabled=true` → `ValueError`.
- New `src/utils/price_anchor.py` —
  `compute_fundamental_anchor(year, config)` returns MAC plus a
  banking-premium scarcity term (~67 EUR/t yr 0 → ~101 EUR/t yr 11).
  `PPOAgent.inject_fundamental_anchor(year)` seeds `price_head.bias`.
  AR(1) floor switches from a static value to the per-year anchor at
  all three call sites.
- ε-random auction bids sample 50/50 below/above a WTP anchor
  (`mac + 0.5 × (penalty − mac) ≈ 93 EUR/t`). BC and KL anchor
  disabled by default.
- `auction.price_max` halved 500 → 250.
- `coverage_shaping`/`coverage_credit_weight`/`gap_closure_weight`
  removed in favour of a single unified `reward.coverage_gap_shaping`
  block.

---

## 7.13.1 — Phantom anchor and gate ramp-in

The phantom bidder was anchoring on the rolling MA3 of clearing
prices, which fed back into itself in collapsed-price regimes. It
now anchors on the effective penalty rate. The ESG compliance gate
also ramps in over the shaping schedule instead of biting from
year 0.

- `phantom_bidder` price anchor: `max(price_fundamental_frac ×
  effective_penalty_rate, reserve + min_above_reserve)`. Defaults
  retuned: `qty_frac_lo/hi: 0.15/0.35`,
  `price_lognormal_sigma: 0.35`, `price_fundamental_frac: 0.60`.
- `compute_gae()` uses `reward.gae_min_std` (default 0.1) in both
  phase normalisation blocks.
- ESG gate: `gate_activation = clamp(1 − shaping_weight/0.5, 0, 1)`,
  `compliance_gate = coverage_frac^(2 × gate_activation)`.
  `gate_activation` added to reward channels.
- Phantom bid stats logged (`phantom_avg_bid_price`,
  `phantom_avg_bid_qty` per episode; `phantom_bid_price/qty/active`
  per year).

---

## 7.13.0 — Phantom bidder and private urgency

Real EU-ETS auctions clear ~40% of volume to financial intermediaries
that don't operate emitting facilities. A `PhantomBidder` now
represents that demand, and a private per-agent urgency scalar
breaks the symmetric cost structure that was driving identical
bids across archetypes.

- `src/environment/phantom_bidder.py` (new) — `PhantomBidder` with
  LogNormal price (anchored to MA3, σ=0.45) and Uniform[5%, 20%] qty
  fraction; injected as `agent_id = n_total` row in `step_auction()`;
  allocation discarded.
- Private `LogNormal(0, 0.30)` urgency scalar per learning agent
  multiplied into the effective penalty in reward.
- ESG compliance gate becomes `coverage_frac²`.
- `_settle_double_auction()` — `_liquidity_ref_ema =
  max(_liquidity_ref_ema, floor_frac × penalty_rate)` (default
  `floor_frac=0.25`) prevents the secondary market from dragging the
  double-auction reference into collapse.
- `scripts/train.py` — BC snapshot seeds into the HPP pool after
  pretraining (`hpp.seed_count: 2`); `esg.scale` lowered 3.5 → 2.0.
- `configs/default.yaml` — `ets.cap_overhead_pct` −0.20 → +0.08.

Market dynamics change substantially with phantom demand; the
`cap_overhead_pct` sign reverses.

---

## 7.12.0 — Rollback of non-approved 7.11 defaults

Several 7.11 defaults had been pushed without scope approval and
were rolled back here. Soft budget and capex penalties returned;
the `log1p` terminal-bank valuation and the queue terminal value
came with them.

- `auction.qty_mult_low` 0.85 → 0.5, `qty_mult_high` 1.5 → 2.0;
  `ets.unsold_to_msr` true → false; `penalty.carry_forward_cap`
  0.5 → 1.0; `reward.shaping_weight_floor` 0.10 → 0.0; dead reward
  config keys removed.
- Soft budget/capex penalties restored in `cost_norm`. Terminal
  bank reverts to `log1p` with a 2× annual-need cap. Terminal queue
  value restored with a completion-fraction discount.
- New regression test: `test_terminal_queue_completion_fraction_discount`.

---

## 7.11.0 — Pure MARL default, scarcity from year 0

The default participant mix flips from `8 learning + 8 bots` to pure
MARL (`n_bot_agents = 0`), the cap goes scarce from year 0, and the
reward is simplified to a fixed global scale with hard mechanical
gating instead of soft penalties. Several of these defaults are
rolled back in 7.12; this entry is preserved as the version label
that actually shipped.

- `companies.n_bot_agents` 8 → 0.
- `ets.cap_overhead_pct` +0.02 → −0.10; `auction.reserve_price`
  and `price_min` 30 → 45; `qty_mult_low/high` tightened;
  `ets.unsold_to_msr: true`; `penalty.carry_forward_cap: 0.5`;
  bank seed `[0.15, 0.35]`.
- `REWARD_SCALE = 1000` fixed normalisation; baseline-cost
  subtraction removed; green and efficiency bonuses removed from
  the reward path; soft budget/capex penalties dropped; bank
  terminal switched to linear; queue terminal removed.
- `auction.pricing_rule: uniform | pay_as_bid` configurable in
  `market_clearing_ets.py`.
- Periodic CSV snapshots (`logging.snapshot_interval: 2500`).

The split into 7.9 / 7.10 below reconstructs the iterations behind
this label after the fact.

---

## 7.10.0 — Scarcity-first market calibration

Mid-cycle pass between the 7.9 reward simplification and the 7.11
pure-MARL flip. Tightened the market once the reward surface had
been cleaned up.

- `ets.cap_overhead_pct` +0.02 → −0.10; reserve and `price_min`
  30 → 45; `qty_mult_low/high` 0.3/2.0 → 0.85/1.5;
  `ets.unsold_to_msr: true`; `penalty.carry_forward_cap: 0.5`;
  bank seed `[0.15, 0.35]`.
- Exploration retune: `epsilon_start` 0.50 → 0.30,
  `epsilon_decay_frac` 0.50 → 0.80, `critic_warmup_frac` 0.10 →
  0.03, `shaping_decay_frac` 0.33 → 0.60,
  `reward.shaping_weight_floor` 0.00 → 0.10.
- New `auction.pricing_rule: uniform | pay_as_bid` knob.

---

## 7.9.0 — Reward simplification

Stripped the reward to four core channels and migrated soft
budget/capex penalties to hard mechanical gating.

- `REWARD_SCALE = 1000` fixed normalisation; baseline-cost
  subtraction removed; green and efficiency bonuses dropped from
  the active reward path.
- Soft budget/capex penalties replaced by hard gating/clipping at
  the env layer.
- Terminal bank switched to linear; queue terminal removed.
- `S_financial` clamp bounded `[0, 1]`.
- Active reward channels: `cost_norm`, `penalty_norm`, `esg_signal`,
  `base_reward`.

---

## 7.8.0 — Dual-ceiling WTP heuristic

Heuristic bid pricing was a single MAC-based number; it now uses a
dual ceiling that mixes a marginal economic willingness-to-pay with
a budget-derived ceiling, so cash-constrained bots stop paying their
way into bankruptcy.

- `_compute_marginal_ef()` becomes an instance method;
  `compute_revenue()` takes a `marginal_ef` parameter;
  `compute_dynamic_budget()` gets a new signature.
- Heuristic dual-ceiling WTP:
  `wtp_economic = MAC + urgency × expected-future-penalty gap`,
  capped by `wtp_budget = max_compliance_share × available / qty`.
  `bid_price = max(min(wtp_economic, wtp_budget), reserve + 1.0)`.
- Compliance-priority investment; `spend_frac=0.9` when
  `carry_forward > 0.01`.
- `_collateral_clip_events` per-episode counter; expected-clearing-
  based collateral sizing.
- Year-log gains 7 per-agent fields (WTP components, invest fraction
  pre/post clip, budget share metrics).
- Calibration: `electricity.base_price` 50 → 55,
  `carbon_passthrough` 0.80 → 0.90, `bots.max_compliance_share:
  0.70`, debt headrooms revised.

---

## 7.7.0 — Revenue-based dynamic budget and emergency loans

Up to here, the budget was a hard fixed cap that didn't move with
electricity revenue. From this release on, agents see a realistic
financial envelope that updates from realised revenue and can take
an emergency loan instead of defaulting outright.

- `Company.compute_dynamic_budget()` — EMA-smoothed (alpha=0.3) from
  electricity revenue minus OPEX plus archetype debt headroom.
  Emergency loan tracking: `_loan_outstanding`,
  `_loan_repayment_annual`, `_years_under_loan`, 8% interest.
- `settle_auction()` accepts `max_loan_budgets`; shortfall within
  the limit triggers `apply_emergency_loan()` instead of
  default/suspension.
- Phase-1 obs 30 → 33: `bid_affordability_last`,
  `loan_outstanding_norm`, `years_under_loan_norm`.
  Phase-2 obs +8 → +10 with budget remaining and compliance
  liability.
- Tiered budget penalty: zero below `soft_zone_start`, quadratic
  ramp to `hard_cap_fraction`. Investment hard gate scales
  `invest_frac` pre-commit.
- `heuristic_policy` — qty/invest/secondary-buy reductions when
  `loan_outstanding_norm > 0.01`.

Phase-1 obs dims change 30 → 33; old checkpoints incompatible.

---

## 7.6.1 — Year-0 calibration patch and phase-aware GAE

- `market_calibration` — `cap_year_0_override = 47.0 Mt`,
  `initial_bank_fraction = 0.10`. TNAC/MSR targets:
  `tnac_upper = 25.5`, `tnac_lower = 12.0`, intake 0.24, release
  3.0 Mt.
- `compute_gae()` normalises rewards by phase before clipping; PPO
  reward clip 2.0 → 10.0; `update()` applies explicit phase masks
  for auction vs secondary rows.
- `compute_auction_rewards()` includes loan interest and projected
  capex pressure; year-1 TNAC out-of-range warning is emission-
  guarded.

---

## 7.6.0 — Per-agent budget normalisation and two-phase credit assignment

The reward function is now denominated in each agent's own annual
budget instead of a fixed `/1000`, and auction and secondary
decisions get separate buffer entries so credit is no longer
smeared between them.

- All cost/penalty/terminal divisors changed from fixed `/1000.0`
  to `/company.annual_budget`. `esg_scale_i = base_esg_scale ×
  (1000 / annual_budget)` preserves the ESG-to-cost ratio.
- `compute_auction_rewards()` returns `r_auction` before secondary
  execution. `baseline_opex` is snapshotted; the heuristic
  `_auc_weight` is removed.
- Two buffer entries per year-step; phase tagging via
  `buffer.phases` and `is_auction` tensor; phase-split actor losses
  in `update_happo()`; phase-masked `compute_post_update_ratio()`.
- Raw rewards stored in buffer; batch normalisation in
  `compute_gae()` (`(r − μ)/σ`, clip `[−10, 10]`).
- HAPPO ordering by `initial_ef` descending (highest emitters
  first, deterministic).

`expected_T = 2 × n_years × episodes_per_update`; old checkpoints
incompatible.

---

## 7.5.0 — MSR three-band withholding

The MSR formula was a single linear band; the legislative formula
(Decision 2015/1814) is three bands. The `unsold_to_msr` accounting
also double-counted in the rollover. Both fixed.

- `_compute_tnac_withholding()`:
  - TNAC > upper → 24% × TNAC,
  - mid ≤ TNAC ≤ upper → TNAC − mid,
  - TNAC < mid → 0.
  - `tnac_lower_ratio` 0.22 → 0.1314; `tnac_mid_ratio: 0.2737`.
- Double-count fix: `unsold = auction_volume − allocations.sum()
  − defaulted_volume`.
- New telemetry: `_last_unsold_rollover_in`, `_last_msr_withheld`,
  `_last_msr_released`.
- `compute_estimate_need()` returns bare `compute_emissions()` (no
  risk buffer).
- `qty_mult_high` 1.3 → 2.0.

---

## 7.4.0 — Linear LRF and 1-year TNAC lag

The LRF cap schedule was decaying *exponentially*; the legislative
formula is linear. MSR was also reading TNAC without the one-year
lag the regulation specifies.

- `cap_t = cap_0 − Σ lrf_k × cap_0` (was exponential).
- 1-year `_prev_tnac` lag for MSR; lower threshold 18 → 22%;
  `release_frac` 0.016 → 0.064; smoothed price trigger requires
  MA3 spike > 2.5× prior year.
- Final-year urgency boost ×3 in heuristic; no selling under
  compliance debt; budget headroom cap on secondary buying.
- Collateral enforcement: `collateral_fraction` corrected to 0.10;
  `leverage_multiplier: 3.0`, `suspension_length: 2`,
  `carry_forward_defaults: true`.
- `compute_diagnostic_score()` logs `S_financial`, `S_green`,
  `S_composite` per agent to year-level CSV.

---

## 7.3.0 — Electricity revenue out of cost normalisation

Electricity revenue had been subtracted from `cost_norm`, which made
the reward depend on something agents don't bid on directly. It is
now reported only as a diagnostic.

- `cost_norm_ex_penalty = total_cost_ex_penalty / 1000.0` (revenue
  no longer subtracted).
- `queue_bonus` shaping term (`γ × n_active_queue × 0.1 ×
  shaping_weight`) deleted; shaping is `green_bonus` only.

Reward gradient changes for all agents.

---

## 7.2.1 — Reward normaliser, RNG, and dead-zone fixes

A grab bag of convergence fixes. Most users will care about the
return double-normalisation removal and the no-short-selling gate;
the rest are quiet correctness fixes.

- `RewardNormalizer.update_and_normalize()` saves `old_mu` before
  update to fix the EMA variance bias.
- `normalize_returns: false` now default in config.
- Auction policy loss weighted by phase-1 advantage proxy from
  `obs2[base+4]`.
- Efficiency bonus coefficient 0.3 → 1.5.
- HAPPO cumulative ratio uses a single tensor (dimension mismatch
  fix).
- `np.random.shuffle` replaced by a seeded `numpy.random.Generator`.
- `max_sell` subtracts `realized_emissions + carry_forward`
  (no-short-selling).
- `invest_frac` uses continuous `((action+1)/2) × max_invest_frac`
  mapping (dead-zone fix).
- Terminal carry-forward penalty `carry_forward × terminal_price ×
  1.5 / 1000`.
- Q-tables now use `np.load`/`np.savez` instead of `pickle`.
- CI added (`.github/workflows/test.yml`).

---

## 7.2.0 — Collateral affordability clip

The auction was happily accepting bids whose collateral the agent
couldn't actually post. A two-step clip fixes that without starving
compliance: shrink qty first, then drop the bid price only if qty
clipping would put the agent below their compliance need.

- `step_auction()` — collateral affordability: step 1 clips qty at
  current bid price if ≥ `min_qty_floor_frac × emissions_need`;
  step 2 reduces bid price if qty clipping would starve coverage.
- New `auction.collateral.min_qty_floor_frac: 0.5`.
- Obs dim `[27] = budget_headroom = clip(1 − budget_spent /
  annual_budget, −0.5, 1.0)`.
- Tabula-rasa expected-price fallback updated to ~80 EUR/t.

---

## 7.1.0 — Bid collateral opportunity cost

EU-ETS auctions tie up capital between bid submission and clearing,
which is a real financial cost the simulator wasn't pricing. It is
now a reward channel.

- New config block: `auction.collateral.{enabled,
  opportunity_cost_rate, hold_fraction}`. `hold_fraction` corrected
  0.08 → 0.02 (weekly-cycle annualised proxy).
- `step_secondary()` computes `rate × hold_fraction × max(0,
  bid_price − clearing_price) × allocation` and passes it into
  `_compute_rewards()` in `total_cost_ex_penalty`.
- Year log includes `collateral_costs`; year CSV gains
  `collateral_cost_A*`.
- Tabula-rasa uniform mode samples 50/50 below/above expected price
  within each side range, fixing a structural overbid bias.

---

## 7.0.0 — Tabula-rasa, dynamic budget, ratio-based ETS calibration

Major release. Cold-start training as a first-class mode, a
revenue-based dynamic budget, emergency loans for green investment,
and a ratio-based market calibration that derives cap and MSR
parameters from the participant set instead of hard-coded numbers.

- `tabula_rasa` config block disables BC and KL anchor, switches
  exploration to uniform, overrides all schedule fractions from
  `n_episodes`. Applied right after the per-seed config deep-copy.
- `src/environment/market_calibration.py` (new) —
  emission-weighted cap/MSR calibration:
  `compute_system_emissions()`, `compute_market_params()`. Env init
  writes derived values into the runtime config;
  `CapSchedule.update_calibration()` for fade-triggered
  recalibration.
- `green_finance` section in `Company`: `record_green_loan`,
  `compute_green_loan_cost`, `green_loan_headroom`,
  `green_capex_headroom`. `step_auction()` supports green-finance
  recovery.
- Ratio-based ETS inputs: `cap_overhead_pct`, `msr.tnac_upper/
  lower_ratio`, `msr.release_frac`. Bot `enhanced_noise` and
  `fade_schedule`. `exploration.mode`.
- New tests: `test_tabula_rasa.py`, `test_green_finance.py`,
  `test_market_calibration.py`, `test_bot_features.py`.

---

## 6.4.0 — Reward decomposition

`reward_A*` is unchanged in total; year and episode logs now also
record `reward_base_A*` and `reward_shaping_A*` separately so
post-hoc analysis can tell what drove a given trajectory.

- Year logs emit `rewards_base` and `rewards_shaping` arrays.
- `reward_base_A*` and `reward_shaping_A*` columns written to both
  `year_log` and `training_log` CSVs.
- Learning behaviour unchanged.

---

## 6.3.0 — Hidden burn-in pre-period

The simulator started year 0 with empty banks, an empty MSR reserve,
and no construction queues. Episodes spent their first three years
recovering from that artificial state. The new burn-in runs a hidden
pre-period that seeds them all to plausible values before year 0
begins.

- `warm_start.burnin_enabled`: hidden burn-in loop seeds bank, MSR
  reserve, construction queues, price history. New keys:
  `n_burnin_years`, `burnin_price_seed_mean/std`. New
  `_calibrate_post_init_bank`.
- `cap_schedule.get_cap()` supports negative years via backward LRF
  extrapolation; `get_auction_volume()` gains a `force_msr` flag.
- MSR recalibrated: `tnac_upper=18.0`, `tnac_lower=9.0`, emergency
  release 0.90 → 4.0 Mt; bank seed `[0.5, 1.5]` → `[0.2, 0.4]`.
- Phase-1 obs 25 → 28: own bank ratio, predicted MSR withholding
  signal, auction volume change vs cap.
- `simulation.n_episodes = 100000`. Long/short profile resolution
  in training.

---

## 6.2.0 — Persistent bot heterogeneity and inflation-aware MSR triggers

Bots had been bidding identically every episode, collapsing the
auction into a flat clearing price. They now carry persistent
per-episode valuation noise and urgency multipliers. The MSR
price-containment trigger is also re-anchored on the
inflation-adjusted penalty rate (the previous version was
effectively dormant).

- `heuristic_policy` — per-episode `valuation_noise` (N(0, σ)) and
  `urgency_multiplier` (U[low, high]) drawn at episode start.
  Even/odd bots use different `urgency_denominators` (1.3/1.7).
  New `bots.{valuation_noise_std, urgency_mult_lo/hi,
  urgency_denominators}`.
- MSR triggers use `1.8 × effective_penalty_rate` (containment) and
  `2.5 × effective_penalty_rate` (emergency release). MSR events
  logged.
- Terminal bank capped at `min(holdings, 2.0 × annual_need)`.
- Secondary sell revenue reduces budget spending.
- Efficiency bonus `0.3 × ef_improvement_ratio × time_weight ×
  price_weight` applies to all agents (not just green).
- Critic value clipping: `v_pred` clamped to `old_values ±
  clip_eps`. `ppo.clip_value: true` by default.

---

## 6.1.0 — Annual MSR cancellation

The post-2023 EU ETS reform permanently cancels MSR holdings above
the previous year's auction volume. The simulator now does the same.

- MSR holdings exceeding the previous year's auction volume are
  permanently cancelled each year. `msr_total_cancelled` written to
  the year-level CSV.
- MSR price-responsive triggers reference inflation-adjusted penalty
  rate (not `price_max`). MSR `activation_year: 2`.
- `ValueNetwork` improvements and PPO stabilisation in `ppo_agent`.

---

## 6.0 — Fundamentals-anchored secondary, saved-carbon ESG

Secondary prices stop being relative offsets and become absolute,
anchored to MAC and penalty rates. The ESG reward is rewritten
around saved-carbon-years; the terminal bank gets diminishing
returns.

- Secondary prices now absolute, anchored to MAC cost and penalty
  rates. Price history uses only successful clearing prices
  (`price_history_anchor: "auction"`) so failed-auction reserve
  prices stop polluting the MA3.
- ESG signal: saved-carbon-years formula with terminal queue
  valuation.
- Terminal bank: diminishing-returns `log` formula.
- `heuristic_policy` NPV properly discounted.
- Minimum auction volume constraint added.
- `Company` — budget hard-cap multiplier, contingency zone, per-run
  schedule isolation.
- `cap_year_0` recalibrated to 57 Mt for the 16-participant market.

---

## 5.5 — Capex throughput cap

A single budget cap let cash-rich agents commit a year's worth of
capex in one shot. A separate per-agent annual construction-spend
cap is now enforced independently of the operating budget.

- New `Company.capex_throughput` property; separate capex-spend cap
  per agent. `compute_capex_penalty()` for overspend.
  `bot_capex_throughputs` exposed in config.
- Heuristic investment fraction respects the cap.
- ESG-weighted reward differentiation improved.

---

## 5.4 — 16 participants

Expanded to 8 learning + 8 heuristic-bot agents (16 total). Cap and
MSR thresholds rescaled to match.

- ETS cap/MSR thresholds recalibrated for the 16-participant market.
- `heuristic_policy` — valuation-based bid price; target-bank
  trajectory logic in the secondary market.
- Year-level CSV gains bid multipliers, intent shares, and
  investment technology choices.

---

## 5.3 — Shared inflation path

Per-agent inflation paths were close to identical but not exactly
identical, which made cross-agent reward comparisons noisy. All
agents now compound the same annual rate from a single per-episode
draw.

- Single shared inflation path per episode; all agents use the
  same compounded rate each year.
- Inflation metrics added to the year log.
- `heuristic_policy` uses dynamic inflation-adjusted penalty rates
  in auction actions.

---

## 5.2 — AR(1) prices and economy-wide inflation

The static price anchor is replaced by an AR(1) price model, and
inflation now compounds penalty rate, CAPEX, OPEX, MAC, and
electricity base price annually at a configurable rate.

- AR(1) price model drives expected-price observations; static
  anchor removed.
- Inflation compounds across penalty rate, CAPEX, OPEX, MAC, and
  electricity base price.

---

## 5.1 — Heuristic bots and Q-learning baseline

Four heuristic-bot agents (12 total participants) and a Q-learning
baseline so the PPO agents have something to be measured against.

- Bots use the same `Company` class; participate in auction and
  secondary market.
- New `src/train_qlearning.py` Q-learning baseline.
- Cap and MSR thresholds rescaled for the 12-participant market.

---

## 5.0 — First stable `ets_marl_happo_current`

The codebase that everything since builds on: 8 learning agents
across 4 archetypes, a 5-technology generation model with MAC
fuel-switching, two-phase yearly decisions (auction + invest, then
secondary trading), and a real training infrastructure with BC
pre-training and a Historical Policy Pool.

- 8 agents in 4 archetypes (coal-heavy, gas-dominant, transitioner,
  green-leader) with distinct financial/ESG reward weights.
- 5-technology model: coal, gas, onshore wind, offshore wind, solar
  — emission factors, CAPEX, capacity factors, construction delays
  0–7 yr, OPEX/decommissioning.
- MAC fuel-switching; electricity revenue
  `P_base + passthrough × P_carbon × EF_system`.
- Carry-forward non-compliance; unified budget envelope.
- Two-phase yearly decision: Phase 1 (auction + invest), Phase 2
  (secondary).
- Terminal payouts: bank value + queue value.
- Construction risk: failure probability, Poisson jitter,
  cancellation, capacity-factor noise.
- Training: BC pre-training, Historical Policy Pool (HPP),
  market-collapse warnings; demand and emission shocks reactivated.

---

## 4.1 — Repo cleanup

Three cleanly separated branches replace the duplicated project
folders. No simulation behaviour changes.

- Duplicate `ets_marl test/` removed; `ets_marl ppo/` renamed to
  `ets_marl_happo`.
- Repository reorganised into `legacy_ppo/`, `legacy_test/`,
  `ets_marl_happo_current/`.

---

## 4.0 — 8 differentiated agents

The first 8-agent market with cost/ESG-differentiated archetypes,
a strengthened penalty system, a heuristic warm-start that
actually survives the first gradient step, and a market-collapse
warning that flags degenerate clearing-price regimes.

- 8 learning agents with differentiated reward weights (seed of the
  four-archetype split).
- Stronger non-compliance penalty; broken-window threshold;
  shortfall tracking in the year log.
- Heuristic actor/critic warm-start before PPO updates; retention
  fixes that previously wiped the warm-start on the first gradient
  step.
- Market-collapse warning system in trainer + console.

---

## 3.1 — Secondary trading and shocks

A secondary double-auction runs after each primary so agents can
rebalance allowances within a year, and per-agent emission noise
plus episode-level demand shocks make the world non-deterministic
in a useful way.

- Secondary double-auction after each primary auction.
- Per-agent emission noise and episode-level demand shocks.
- PPO loop reworked with separate auction and secondary decision
  handling.
- Selling-side incentive fix to keep banked allowances liquid.

---

## 3.0 — Multi-technology simulator

The first version that takes generation-stack heterogeneity
seriously. `Company` carries distinct emission factors, CAPEX,
OPEX, and dispatch logic per fuel; `ETSEnvironment` is reworked to
match.

- Richer technology mix with per-fuel emission factors, CAPEX, OPEX,
  dispatch logic.
- Cap, MSR, and clearing logic generalised; reserve price introduced.
- `configs/default.yaml` rewritten end-to-end; single-fuel legacy
  parameters retired.

---

## 2.1 — Auction-mechanic prototypes

A parallel `ets_marl test/` project to prototype auction mechanics
and reward variants without disturbing the v2.0 baseline. No code
changes to the main project; the prototype project is removed at
the v4.1 / v5.0 cleanup.

---

## 2.0 — First end-to-end MARL build

`Company` + `ETSEnvironment` + `PPOAgent` (pure PPO). Config-driven
entry points and the first test suite.

- `scripts/train.py`, `scripts/evaluate.py` — config-driven entry
  points.
- `configs/default.yaml` plus `baseline_2024` / `low_price_2020`
  scenario overlays.
- `company.py`, `ets_environment.py`, `agents/ppo_agent.py`
  (ActorCritic) — bare compliance bidding under a uniform-price
  auction.
- `tests/test_environment.py` — first smoke tests; `requirements.txt`
  and `pyproject.toml` packaging.

---

## 1.0 — Repository scaffold

Bare project skeleton and a placeholder training script. No
learning loop yet.

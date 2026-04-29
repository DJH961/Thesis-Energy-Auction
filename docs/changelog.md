# Changelog — ETS MARL (`ets_marl_happo_current`)

Version numbers reflect the `# ETS MARL — Configuration vX.Y[.Z]` header in `configs/default.yaml`
and, from v6.1.0 onwards, the `version` field in `pyproject.toml`.

---

## [8.5.4]

Bug-fix / infrastructure release. Pure efficiency change — **no
training-quality, RNG, or numerical paths altered**. Two back-to-back
runs of `--seed 42` produce byte-identical `training_log_s42.csv` and
`year_log_s42.csv` before and after the change.

### Fix 1 (v8.5.4) — architecture-aware CPU thread configuration

**Background.** On a 16 vCPU / 64 GB Azure `Standard_D16ds_v5` the
trainer was averaging ~24% CPU and ~3.4 GB RAM. The workload is a
single Python process driving small MLPs (`hidden_size=256`,
`mini_batch=64`); BLAS thread throughput plateaus around ~4 threads on
matmuls of this size, after which OMP barrier sync dominates and extra
cores sit idle. PyTorch's default of `torch.get_num_threads() ==
n_physical_cores` therefore *under-utilises* large cloud VMs (the BLAS
pool oversubscribes a single training process while 12 of 16 cores idle).

**Fix.** New `src/utils/compute_setup.py`:

* `detect_architecture()` — cross-platform detection of logical /
  physical cores + RAM, via `psutil` with a `/proc/cpuinfo` fallback.
* `configure_compute(num_threads=None, total_threads=None)` — sets
  `torch.set_num_threads`, `torch.set_num_interop_threads(1)`, and the
  `OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS /
  NUMEXPR_NUM_THREADS / VECLIB_MAXIMUM_THREADS` env vars plus
  `KMP_BLOCKTIME=0` (prevents Intel OMP from spinning idle workers,
  which is the failure mode that produces the "low CPU % but high
  context-switch" pattern on large VMs).
* Auto-policy calibrated to this codebase's matmul size:
  ≤2 cores → use what we have; 3–4 → 2; 5–32 → 4; 33+ → 6.
  Override with `ETS_NUM_THREADS` env var or `--num-threads` flag.

`scripts/train.py` now calls `configure_compute()` once at the top of
`main()`, before any tensor work.

### Fix 2 (v8.5.4) — `--parallel-seeds N` launcher for multi-seed runs

**Background.** The single-process bottleneck above is structural: even
perfectly tuned BLAS can't parallelise the Python env loop. The right
way to use a 16-core box is to run multiple seeds *in parallel
processes*, each with its own ~4-thread BLAS budget.

**Fix.** New `--parallel-seeds N` flag on `scripts/train.py`. When N>1
and multiple seeds are supplied, the launcher uses
`concurrent.futures.ProcessPoolExecutor` to spawn one subprocess per
seed; each child re-executes `train.py --seed S --parallel-seeds 1`
with `ETS_NUM_THREADS = total_cores // N`. Each child still runs the
unchanged `train_one_seed` path, so per-seed RNG, optimizer order, and
numerical results are bit-identical to a sequential invocation. Default
is `--parallel-seeds 1` → existing behaviour exactly.

Recommended invocation on `Standard_D16ds_v5`:

```bash
python scripts/train.py --config configs/default.yaml \
       --seed 1 2 3 4 --parallel-seeds 4
```

→ 4 processes × 4 BLAS threads ≈ all 16 vCPUs doing useful work.
Average CPU utilisation rises from ~24% to ~80–90% with no change to
any seed's training trajectory.

### Reproducibility note

Within a fixed `--num-threads` setting, runs are byte-identical CSV-for-
CSV. Switching between very different thread counts (e.g. 1 vs 8) on
the same seed *can* surface ~1e-7 floating-point reordering in BLAS
reductions; this does not affect training trajectories at any decimal
that matters but may change the last printed digit in some logs. The
default and `--parallel-seeds` paths both keep `num_threads` constant
per process, so this caveat does not apply to standard usage.

### Fix 3 (v8.5.4) — multi-variant × multi-seed sweep launcher

**Background.** With `--parallel-seeds` it became easy to run many seeds
of *one* config in parallel, but ablations across config variants
(scarcity, MSR on/off, reserve price, reward weights, …) still needed
hand-launched commands per variant, and per-variant outputs collided on
filenames once they were copied into a single analysis folder. The
unchanged trainer also dumps a *lot* of per-episode console output —
fine for one process, but a mess when several variants are running
concurrently in the same terminal.

**Fix.** New `scripts/sweep.py` launcher that reads a sweep spec YAML
and runs the cartesian product of variants × seeds in a process pool.

* **Spec schema** (full schema + validation in `src/utils/sweep.py`):

  ```yaml
  base_config: configs/default.yaml
  output_dir: results/sweeps/scarcity_msr
  seeds: [1, 2, 3]                 # default; variants may override
  parallel_workers: 4
  threads_per_worker: null         # auto = total_cores / workers
  variants:
    - name: tight_cap
      overrides: {ets.cap_overhead_pct: -0.02}
    - name: msr_off
      overrides: {ets.msr.enabled: false}
    - name: tight_cap_msr_off
      overrides:
        ets.cap_overhead_pct: -0.02
        ets.msr.enabled: false
      seeds: [1, 2]                # per-variant seed override
  ```

  Each variant's overrides are deep-merged onto the base config (lists
  *replace*, not concatenate, to avoid silent hyperparameter doubling).
  Dotted-path keys (`ets.msr.enabled`) and nested mappings are both
  accepted. Variant names are validated as filesystem-safe.

* **Per-variant output isolation.** Each variant's resolved YAML is
  written to `<output_dir>/_resolved/<variant>.yaml`. Its
  `logging.results_dir` is forced to `<output_dir>/<variant>/`, and
  `train.py` is invoked with a new `--run-tag <variant>` flag that
  rewrites every output filename so they remain unique even when
  copied into a single folder for analysis:

  | Without tag (existing)              | With `--run-tag tight_cap`               |
  |-------------------------------------|------------------------------------------|
  | `training_log_s42.csv`              | `training_log_tight_cap_s42.csv`         |
  | `year_log_s42.csv`                  | `year_log_tight_cap_s42.csv`             |
  | `checkpoints_s42/`                  | `checkpoints_tight_cap_s42/`             |
  | `snapshots/training_log_s42_ep*.csv`| `snapshots/training_log_tight_cap_s42_ep*.csv` |

  `--run-tag` is a no-op when omitted, so existing single-config runs
  produce byte-identical filenames to v8.5.3.

* **Per-job log capture, terse parent terminal.** Each subprocess's
  stdout+stderr is redirected to
  `<output_dir>/<variant>/run_<variant>_s<seed>.log` instead of being
  printed to the terminal where the launcher runs. The full per-episode
  diagnostic output is preserved on disk for later inspection. The
  parent terminal only prints structured progress lines:

  ```
  [sweep] [START 1/17] reference s=1  → results/.../reference/run_reference_s1.log
  [sweep] [LIVE  reference s=1] Ep 1200/100000 (1.2%) | px 75→142 (μ128) | R̄ -3.2→-1.8 | comp 87% | green 31→44%
  [sweep] [DONE  1/17 OK ] reference s=1  (results/.../run_reference_s1.log)
  ```

  A background heartbeat thread fires every `--heartbeat-interval`
  seconds (default 60) and emits **one structured summary line per
  running job**, parsed directly from the per-episode CSV
  (`training_log_<variant>_s<seed>.csv`):

  | Field | Meaning |
  |---|---|
  | `Ep N/total (X.X%)` | Last episode logged, with progress through `simulation.n_episodes`. |
  | `px init→recent (μy)` | Last-year auction clearing price: episode-0 value vs the mean over the most recent 50 logged episodes. `μ` is the within-episode year-mean over the same window. |
  | `R̄ init→recent` | Mean of `reward_A*` across all agents, episode-0 value vs recent mean (tracks convergence). |
  | `comp X%` | Compliance rate over the recent window: fraction of `(agent, episode)` cells where `shortfall_A* ≈ 0`. |
  | `green init→recent%` | Mean `green_frac_A*` across all agents, episode-0 value vs recent mean. |

  When the CSV does not exist yet (e.g. during the behavioural-cloning
  pretraining phase, before the trainer's CSV writer has emitted any
  rows), the heartbeat falls back to the last informative line of the
  captured log file. Pure-separator lines (`═══`, `───`, `=====`, …)
  are filtered out of that fallback. Use `--quiet` to suppress
  heartbeats entirely.

  The summary is computed with a tail-window read (last 64 KB of the
  CSV) and a cached "first-ever row" anchor, so re-summarising remains
  cheap even at 100k+ episodes.

* **Reproducibility.** Each `(variant, seed)` job runs the unchanged
  `train_one_seed` path inside a fresh
  `train.py --config <variant>.yaml --seed S --parallel-seeds 1
  --run-tag <variant>` subprocess, so per-seed RNG, optimiser order,
  and CSV outputs are bit-identical to a sequential, hand-launched
  invocation; only the process layout (and on-disk filename infix)
  differ.

Usage:

```bash
# Validate spec and inspect the job plan without launching anything:
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml --dry-run

# Run the full sweep:
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml
```

### Tests

* `tests/test_compute_setup.py` — 14 cases covering `detect_architecture`,
  the default-thread policy across core counts (incl. 16-core
  `D16ds_v5`), explicit / env-var / `total_threads` overrides.
* `tests/test_sweep.py` — 30 cases covering `deep_merge` (incl.
  list-replace and no-mutate invariants), dotted-path expansion, spec
  validation (missing fields, duplicate / unsafe variant names,
  bool-vs-int seed types, per-variant seed override rules),
  variant-config resolution, `build_jobs` cartesian expansion, and
  YAML-on-disk round-trips.
* `tests/test_sweep_launcher.py` — 20 cases covering the launcher's
  `_job_log_path`, `_tail_last_meaningful_line` (incl. comment skip,
  separator skip, large-file tail window), `_summarize_csv` (header-only
  files, partial trailing lines, compliance-rate edges, ≤200-char output
  guarantee, initial-row caching), and the `_Heartbeat` background
  thread (CSV-preferred summary, log-tail fallback, emit / remove /
  truncate / idempotent stop).
* End-to-end smoke: a one-variant one-seed sweep against
  `configs/smoke_100.yaml` produces the renamed CSVs / checkpoint dir
  and a `run_<variant>_s<seed>.log` capture, with the parent terminal
  only emitting `[sweep] [START …]` / `[LIVE …]` / `[DONE …]` lines.
* Full suite: 445 passed, 1 skipped (no regressions; +50 new tests
  vs v8.5.3).

### Files touched

* `src/utils/compute_setup.py` (new)
* `src/utils/sweep.py` (new) — spec schema, deep-merge, override
  expansion, variant-config resolution, job materialisation.
* `scripts/sweep.py` (new) — launcher: process pool, per-job log
  capture, heartbeat thread, plan / dry-run output.
* `configs/sweeps/example_sweep.yaml` (new) — worked example covering
  scarcity, MSR on/off, reserve price.
* `tests/test_compute_setup.py` (new)
* `tests/test_sweep.py` (new)
* `tests/test_sweep_launcher.py` (new)
* `scripts/train.py` — `main()` calls `configure_compute()`; new
  `--parallel-seeds`, `--num-threads`, `--run-tag` flags;
  `_run_seed_subprocess` helper; output filenames / checkpoint dir /
  snapshot filenames now carry the `--run-tag` infix when set.
* `README.md` — new "Running Sweeps" subsection.
* `configs/default.yaml`, `pyproject.toml` — version bump to 8.5.4.

---

## [8.5.3]

Bug-fix release. Three targeted changes addressing the v8.5 floor-bidding
regression head-on (Fix 1 was the missing structural piece) and refining
v8.5.2's Fix 3 with the rolling/capped sec-price proxy the user
correctly identified as needed.

### Fix 1 (v8.5.3) — restore `baseline_cost` subtraction in `compute_auction_rewards()`

**Background.** v8.4.2 subtracted a `baseline_cost = need × clearing_price`
fair-price reference from the auction reward so the bid head saw
*deviations* from the clearing price, not absolute spend. v8.5 removed
this. The result: every euro of `auction_cost` became pure negative
reward to the bid head, regardless of whether the agent bought exactly
its compliance need at the prevailing clearing price (a *fair* purchase)
or over-paid. The bid head's gradient unambiguously points to floor-
bidding (which minimises `auction_cost`), even when supply is scarce.
This is the structural cause of the v8.5 floor-bidding pathology.

**Fix.** Restore the baseline:

```python
# src/environment/ets_environment.py — compute_auction_rewards()
clearing_price_nom = float(self.last_clearing_price)
baseline_cost = (need * clearing_price_nom) / max(infl, 1e-9)
compliance_norm_excess = (
    (auction_cost + mac_cost_i + collateral_cost_i - baseline_cost)
    / compliance_denom
)
r_auction_bid[i] = -(compliance_norm_excess) - gap_penalty
```

Properties:
- Bidding to win exactly `need` at the clearing price → `auction_cost ≈
  baseline_cost` → bid-head reward ≈ 0 (neutral). The bid head no longer
  has a structural preference for floor-bidding.
- Over-buying → `auction_cost > baseline_cost` → small positive cost.
- Under-buying → `auction_cost < baseline_cost` → small "saving" from
  baseline, but the v8.5.3 `gap_penalty` (with rolling capped remediation
  rate, see Fix 3 v2 below) dominates because the remediation rate
  (≈ `eff_penalty` to `1.5·eff_penalty`) is several times the clearing
  price. Net effect on under-buyers is still strongly negative.

The `compliance_norm` legacy diagnostic (without baseline) is still
written to `_last_auction_reward_channels` for backward compatibility;
the new `compliance_norm_excess` field is the actual reward signal, and
`baseline_cost` is exposed as its own diagnostic.

### Fix 3 v2 (v8.5.3) — rolling, capped expected-remediation rate

**Why v8.5.2 was too aggressive.** v8.5.2 used `max(eff_penalty,
last_secondary_price, anchor)` per missing Mt. In normal markets,
secondary clearings sit *persistently* above the penalty rate because
carry-forward demand makes secondary buys preferable to default+
penalty+carry-forward chain. As the user observed, `last_secondary_price
> eff_penalty` is the typical case, not the price-spike edge case the
v8.5.2 design assumed. Two failure modes:
1. A single thin-liquidity year where sec clears at, e.g., 400 EUR/t
   would set the next year's gap_penalty rate ≈ 400, multiplying that
   reward channel by ~3× vs the legacy formula (`compliance_denom ≈
   anchor·need ≈ 57·need` ⇒ amplification factor 400/57 ≈ 7).
2. The bid head's effective coverage-gap penalty therefore had a
   stochastic, year-to-year scale that destabilised training rather
   than providing a clean trust-region signal.

**Fix.**
1. Maintain a per-episode EMA `_sec_price_ema` of secondary clearings,
   updated only when `secondary_volume > 0` (no-trade years cannot
   pollute the EMA with the stale auction clearing price). Default
   `reward.sec_proxy.ema_alpha = 0.30` (≈ 2-3 year half-life).
2. Cap the rate at `reward.sec_proxy.cap_mult × eff_penalty_rate`
   (default 1.5×). The bid head sees a stable proxy in the
   `[eff_penalty, 1.5·eff_penalty]` band regardless of sec dynamics.
3. Floor still `max(eff_penalty, sec_ema, anchor)` to handle year 0
   gracefully (sec_ema is `None` until the first sec clearing).

```python
sec_ema_nom              = self._sec_price_ema  # None until first sec clear
sec_proxy_nom            = float(sec_ema_nom) if sec_ema_nom is not None else float(anchor_t)
remediation_floor_nom    = max(eff_pen_rate_nom, sec_proxy_nom, anchor_t)
remediation_cap_nom      = self._sec_proxy_cap_mult * eff_pen_rate_nom
effective_remediation_nom = min(remediation_floor_nom, remediation_cap_nom)
```

New diagnostic `sec_price_ema` exposed in `_last_auction_reward_channels`.
New config block:

```yaml
reward:
  sec_proxy:
    ema_alpha: 0.30
    cap_mult: 1.5
```

### Year-0 bid-change-limit anchoring

**Issue.** `bid_change_limit` (BCL) was only active for `year > 0`;
year-0 bids were unconstrained, so policies that hadn't yet learned a
sensible price routinely emitted `price_max = 250 EUR/t`. The first
auction's clearing then polluted both the MA3 and the AR(1) expected-
price forecast with an unrealistic anchor for the rest of the episode.

**Fix.** The `year > 0` guard is removed in `step_auction()`. After
`warm_start.burnin`, `_price_history` is seeded with synthetic +
heuristic clearings (~70 EUR/t), so `price_ma3_early` is well-defined;
if it isn't, `_compute_price_ma3()` falls back to `expected_price`
(≈ `_price_initial`). The clipping reference is then
`max(price_ma3_early, fundamental_anchor)` ± `value` (default ±75 EUR/t),
matching the year > 0 path. With default config, year-0 bids land in
roughly `[0, ~145]` instead of `[0, 250]`.

### Verification of earlier fixes

- **Fix 2 (v8.5.1, dual-clip PPO + log_ratio clamp ±10):** verified by
  the 5 passing tests in `tests/test_dual_clip_ppo.py`. Actor loss is
  bounded by `c·|adv|` (default `c = 3.0`) for any importance ratio.
  No further change required.
- **Fix 4 (v8.5.2, per-sub-head KL early stop):** verified correct.
  When `split_invest_head=True`, `mb_kl = max(kl_bid, kl_inv, kl_sec)`
  drives the early-stop signal; neither sub-head can dilute the other.
  Joint-head fallback (`split_invest_head=False`) preserves legacy
  averaging. No change required.

### Version
- `pyproject.toml`         8.5.2 → 8.5.3
- `configs/default.yaml`   header banner v8.5.2 → v8.5.3

---

## [8.5.2]

Bug-fix release targeting the v8.5 floor-bidding / two-faction failure mode
observed in `archive/std_log.txt` (price stuck at €45 in the auction while
the secondary clears at €280–€350, with `ALoss` saturating at 1e7–1e8 on
ceiling-bidding agents and `comply` regressing from ~85% to ~70%).

Two targeted fixes addressing the diagnostic items the user flagged as
"buying in the market should truly make more sense than not buying" and
the split-head invest critic interaction with KL early stopping.

### Fix 3 — bid head now prices the cost of forced Phase-2 buying

`compute_auction_rewards()` (`src/environment/ets_environment.py`) priced
the coverage gap at `company.penalty_rate` only:

```python
gap_penalty = (coverage_gap * company.penalty_rate) / compliance_denom
```

In price-spike regimes where the secondary clears above the (inflation-
adjusted) penalty rate, this systematically *under-states* the true cost
of leaving an undercoverage gap — a rational agent will buy on secondary
rather than default, paying `sec_price` per missing Mt. The bid head
therefore developed a perverse preference for floor-bidding because the
arbitrage profit from buying cheap allowances and re-selling on the
secondary leaks into the value bootstrap (shared trunk + shared main
critic) while the penalty-only `gap_penalty` provided no countervailing
signal at auction time.

**Fix.** The bid head now prices each missing Mt at the maximum of three
nominal-EUR/t quantities, deflated to real terms:

```python
expected_remediation_rate_real = (
    max(eff_pen_rate_nom, sec_proxy_nom, anchor_t) / infl
)
gap_penalty = (coverage_gap * expected_remediation_rate_real) / compliance_denom
```

- `eff_pen_rate_nom`  — `Company.effective_penalty_rate(year)`
  (also fixes a pre-existing minor unit inconsistency: the old formula
  used the *base* `penalty_rate` against a real-terms denominator, so
  `gap_penalty` was slightly under-stated in late episodes).
- `sec_proxy_nom`     — `self.last_secondary_price`, a 1-year-lagged
  proxy for the secondary clearing the agent will face this year.
  Initialised to `_price_initial`, so year 0 falls back to the anchor.
- `anchor_t`          — fundamental anchor at the current cap;
  conservative lower bound when sec history is uninformative.

Behaviour in normal markets is essentially unchanged (the max picks
`eff_pen_rate ≈ 138`, matching the legacy `penalty_rate` to floating-
point precision). In price-spike regimes (sec > penalty), the bid head
correctly internalises the secondary-buy cost.

A new diagnostic field `expected_remediation_rate_real` is added to
`_last_auction_reward_channels`. Existing `coverage_gap_penalty` keeps
its name and downstream consumers.

### Fix 4 — per-sub-head KL early stopping when `split_invest_head=True`

`update_happo()` (`src/agents/ppo_agent.py`) computed a single
**joint-6D** auction KL even when the bid head (dims 0–1) and the
investment head (dims 2–5) were trained against separate advantage
streams. Two failure modes:

1. A runaway invest head could push joint KL well above `target_kl`
   while the bid head was still happily inside its trust region —
   early-stopping the *whole* update and freezing the bid head at a
   suboptimal point.
2. Conversely, a quiet invest head could average down a high bid-head
   KL so target_kl never tripped, allowing the bid head to drift past
   the intended trust region.

**Fix.** When `split_invest_head=True`, per-sub-head KL is computed
under `no_grad` from the *already-existing* `lr_bid`/`r_bid` and
`lr_inv`/`r_inv` tensors (the same ones used to build the dual-clip
losses), and the early-stop signal is

```python
kl_auc_eff = torch.max(kl_bid_mb, kl_inv_mb)
mb_kl      = torch.max(kl_auc_eff, kl_sec)
```

Using `max` (not `mean`) ensures that *either* sub-head exceeding
`target_kl` triggers early-stop, while neither sub-head can dilute the
other's signal. The legacy averaging path (`0.5 * (kl_auc + kl_sec)`)
is retained for `split_invest_head=False` to preserve back-compat.

The synchronous `update()` path (no HAPPO weighting, no split-head
support) is unchanged.

### Version
- `pyproject.toml`         8.5.1 → 8.5.2
- `configs/default.yaml`   header banner v8.5.1 → v8.5.2

---

## [8.5.1]

Bug-fix release. Three areas:

### PPO numerics — dual-clip surrogate (fix `actor_loss → 1e6–1e9`)
- The standard PPO clipped surrogate is unbounded below for negative-
  advantage rows once the importance ratio drifts above `1+ε`: both
  `r·A` and `clip(r)·A` are large negatives, so `min` returns `r·A`
  and the displayed `-min` grows without bound. Combined with the
  v8.4.2 widening of the `log_ratio` clamp to `±20`, the actor loss
  was reaching `1e8` in v8.5 training logs.
- New `_ppo_clipped_loss(ratio, adv)` helper implements **dual-clip
  PPO** (Ye et al. 2020). For `adv < 0` rows it adds a `c·adv` floor;
  positive-advantage rows are unchanged. Wired into all five surrogate
  sites in `update()` and `update_happo()` (auction joint, secondary,
  bid head, invest head, single-adv joint auction).
- `log_ratio` safety clamp tightened from `±20` → `±10` (still ratio
  cap ≈ 22 000, well outside any reasonable trust region; bounds the
  *displayed* loss too).
- New config knobs:
  - `ppo.dual_clip_c` (default `3.0`; `0` disables dual-clip)
  - `ppo.log_ratio_clip` (default `10.0`)

### Convergence-safe schedule tweaks (`configs/default.yaml`)
- `ppo.entropy_coef_final` `0.005 → 0.015` — small lift to prevent
  late-training std collapse without disturbing the existing decay
  shape.
- `ppo.log_std_min` `−3.5 → −3.0` — minimum raw σ rises from ≈0.030
  to ≈0.050.
- `ppo.n_epochs` `5 → 3` — combined with `target_kl=0.015` early-stop
  and the new dual-clip, this materially reduces ratio drift per
  update.

### Per-agent compliance attribution diagnostic (`scripts/train.py`)
- New `Why(U/D/B/C)` column on the per-agent and bot tables,
  partitioning each year of the episode into mutually-exclusive
  buckets:
  - **U** Under-bought, non-compliant — `obligation > alloc` AND
    non-compliant
  - **D** Debt-cascade, non-compliant — `obligation ≤ alloc` AND
    non-compliant (prior-year carry-forward exceeded auction
    surplus + bank)
  - **B** Bank-covered — `obligation > alloc` AND compliant AND no
    net secondary buy
  - **C** Sec-Covered — `obligation > alloc` AND compliant AND net
    secondary buy
  - Identity by construction: `U + D == Sf`.
- Selling-into-shortfall is **impossible** by environment rules
  (`_settle_double_auction` enforces
  `max_sell = alloc + bank − emiss − carry_forward`), so no separate
  "sold-into-short" bucket is reported.
- Year log gains a new `old_carry_forward` field
  (`ets_environment.py`) — the carry-forward debt at year start,
  required to compute the obligation-aware `U/D` split. Used only
  by the diagnostic; no behavioural impact.

### Tests
- New `tests/test_dual_clip_ppo.py` (5 tests) pins down the dual-clip
  semantics: positive-adv unchanged, negative-adv bounded by
  `c·|adv|`, vanilla recovered at `dual_clip_c=0`, mixed-sign batch
  correctness, finite displayed loss under extreme ratios.
- Full suite: 377 passed.

---

## [8.5]

Restructures Phase-1 credit assignment, refines the terminal/ESG reward
shape, and tightens budget accounting. Highlights:

### Reward / critic split (Phase 1)
- **Split-head investment critic** (`ppo.split_invest_head`, default `true`).
  The Phase-1 policy is now trained as two sub-heads sharing a trunk:
  - bid sub-head (dims 0–1: bid price, qty multiplier) is trained against a
    "main" advantage stream that sees compliance, secondary financials,
    penalty, banking signal, and terminal-bank value;
  - investment sub-head (dims 2–5: invest_frac, tech logits) is trained
    against a separate "invest" advantage stream that sees capital cost,
    centered ESG signal, and the discounted-NPV terminal-queue value.
  Each stream has its own value network (`value_net` and
  `value_net_invest`) and its own causal reward normalizer per phase.
- `compute_auction_rewards()` now returns `(joint, bid, invest)` so the
  train loop can route each sub-stream into the right buffer slot.
  `_compute_rewards()` exposes `_last_invest_reward_phase2` for the same
  purpose at the secondary-phase timestep.

### Tech-portfolio investment
- `auction_actions[i, 3:6]` are no longer routed through `argmax`: instead
  a softmax over the three buildable green technologies splits
  `invest_frac` proportionally, so each year's investment can diversify
  across techs (`investment.tech_softmax_temperature`, default `1.0`).
- Per-tech construction-phase cancellation rates
  (`construction_jitter.p_cancel_per_tech`) replace the single
  `p_cancel` scalar; defaults align with industry data
  (~1.2 %/yr onshore, ~0.8 %/yr offshore, ~2.0 %/yr solar).

### Capex throughput linked to revenue
- `Company.effective_capex_throughput` is now scaled by
  `clip(0.7 + 0.3 × revenue_t / baseline_revenue, 0.5, 1.5)`, where
  `baseline_revenue` is captured the first time the env reports revenue
  in an episode. High-revenue years gain construction headroom; low-
  revenue years see it shrink.

### Terminal-value reshape
- **Terminal queue NPV (#3).** `terminal_queue_value` now computes a
  proper discounted-cash-flow on each queued project: annual carbon
  savings × terminal price, valued as an annuity over
  `reward.terminal_asset_lifetime_years` (default 20 yr) at
  `investment.discount_rate` and discounted from the project's
  completion year. Late-episode investments are now valued at their
  economic worth instead of decaying linearly to zero.
- **Terminal bank discounted hold (#6).** The end-of-episode allowance-
  bank kicker drops the linear-below-need / log-above-need split. The
  value is now a single discounted hold:
  `holdings × terminal_price × (1 + invest_rate)^(-terminal_payoff_years)
  / budget_real`.

### ESG signal centred on a linear baseline (#13)
- ESG signal is now `esg_scale × ((ef_ratio − year/n_years) + speed_bonus)`,
  so do-nothing decarbonization yields zero-mean signal across the
  episode and only progress *ahead of* the linear trajectory pays. The
  compliance gate is only applied to non-negative signals so a
  behind-trajectory agent's negative signal isn't flipped under low
  coverage.
- `esg.speed_coef` is now front-loaded (`speed_coef` > `speed_coef_late`):
  early decarbonization receives the bigger speed bonus, since late-
  episode projects already pay back through the new terminal-queue NPV.

### Opponent observations (#16)
- `opponent_snapshots`/`opponent_obs` replaces the per-firm
  `bank_norm` (own holdings ÷ own need) with `tnac_share_norm`
  (own holdings ÷ Σ holdings). This signal is publicly inferable from
  aggregate TNAC reports and respects EU ETS confidentiality rules
  while still letting agents triangulate market share.

### Budget envelope (#1, #2)
- `penalty_cost` is now charged through `Company.record_spending`, so
  the penalty also counts toward `budget_spent_this_year` and the
  hard-cap penalty channel. Per-agent `budget.debt_headrooms` are sized
  so compliance-respecting agents never need them; non-compliant agents
  absorb the penalty before exhausting their cash buffer.
- Single source of truth for the budget hard cap. `budget.hard_cap_multiplier`
  is removed; `budget.hard_cap_fraction` (default 1.15) drives both
  `step_auction()`'s investment hard gate and `compute_budget_penalty()`.

### HAPPO ordering (#15)
- `ppo.happo_order_metric` selects the EMA used for HAPPO sequential
  ordering. The new default `"advantage"` tracks per-agent mean GAE
  advantage so the update order is robust against different reward
  floors across mixed reward functions; `"reward"` keeps the prior
  behaviour.

### Misc.
- `investment.discount_rate` raised to `0.05` (5 % WACC, standard
  utility-sector hurdle rate).
- Tests covering ESG positivity, terminal-bank diminishing returns and
  the auction-reward signature were updated to reflect the new shapes.

---

## [8.4.2]

Bug-fix release addressing defects identified in the under-bidding analysis
(`docs/BUGS_AND_ISSUES.md`, sections 1–6).

### PPO numerics (`src/agents/ppo_agent.py`)
- **1.5** Advantage / return / HAPPO weighted-advantage normalization now uses
  `torch.clamp(std, min=gae_min_std)` instead of `+ 1e-8` (the configured floor was
  previously unused).
- **1.6** Widened `log_ratio` pre-clamp from `[-2, 2]` to `[-20, 20]` so the PPO
  clipped surrogate `clamp(ratio, 1±ε)` is solely responsible for the trust region.
- **1.7** Phase-aware reward normalization in `compute_gae()` is now causal: two
  per-phase running EMA `RewardNormalizer`s walk the trajectory in temporal order.
- **1.16** `normalize_returns` code default aligned with YAML (`True` → `False`).
- **6.7** Parenthesized `default_rng(seed if … else 0 + agent_id)` for readability.

### Training loop (`scripts/train.py`)
- **1.8** Removed the discarded `RewardNormalizer.normalize_reward()` call (1.7
  makes it redundant).
- **1.9** Seed Python `random` alongside `numpy` / `torch` in `train_one_seed`.
- **1.10** Cosine LR decay moved out of the `is_update_episode` gate so it advances
  every episode.
- **2.9** Skip `agent_perf_ema` update for HPP-swapped agents (their rollout came
  from a historical policy).

### Cap schedule (`src/environment/cap_schedule.py`, `ets_environment.py`)
- **2.5** Unified MSR effective-penalty calculation behind `_effective_penalty()`.
  New optional `inflation_factor` parameter takes precedence over `(1+rate)**year`
  compounding so MSR thresholds are correct under
  `penalty.inflation_random_std` / `inflation_random_window`.

### Environment (`src/environment/ets_environment.py`)
- **3.10** One-shot warning when `auction.carry_forward_defaults=false` silently
  drops defaulted volume from supply.

### Config alignment
- **1.11 / 3.9** Added `ets.reserve_discount`, `ets.reserve_initial`, and
  `price.initial_expected` to `configs/default.yaml` with documented defaults.
- **2.18** Corrected `+12%`→`+10%` cap-overhead comment.
- **1.13** Stronger inline warning on bot-array length vs `n_bot_agents`.
- **1.12** Synced every non-`tabula_rasa` key from `default.yaml` into
  `configs/smoke_100.yaml` (treasury_reserve, banking_signal, opponent_obs, plus
  PPO schedule keys).
- **1.14 / 1.15 / 2.19 / 3.1 / 3.2 / 3.4 / 3.7** Re-aligned smoke with default for
  `phantom_bidder.enabled`, `exploration.mode`, `ets.initial_bank_fraction`,
  `auction.bid_change_limit.value`, `auction.suspension_length`,
  `auction.budget_price_clip`, `budget.dynamic_budget_ceiling_multiplier`,
  `ppo.critic_compliance_features`.

### Tests (`tests/`)
- **5.5 / 5.6 / 5.7** Tightened three assertions: under-subscribed
  `alloc.sum() == 1.0`; invest action `executed ≤ requested`; `_make_env` default
  `bcl_value` 50→75 to match production.

### Confirmed false positives (no change)
- **1.1 / 1.2** `settle_compliance_realized()` does not modify holdings; the env's
  `holdings - total_obligation` is the single surrender step. CF growth is linear.
- **1.3** `settle_auction()` zeros defaulter allocations, so
  `auction_volume - allocations.sum() - defaulted` is correctly mutually exclusive
  with `_defaulted_volume_pending`.
- **1.4** `mac_cost` is in EUR/tCO2 (per its comparison to `carbon_price`); units
  `MtCO2 × EUR/tCO2 = M€` ✓.

---

### Fixed — `gap_penalty` denominator mismatch (auction-phase reward)

`compute_auction_rewards()` divided `gap_penalty` by `budget_real` while `compliance_norm`
was divided by `compliance_denom (= anchor_real × need)`. With `budget_real ≈ 10 ×
compliance_denom`, a missed Mt saved ~1.0 of `compliance_norm` but cost only ~0.2 of
`gap_penalty` — an explicit gradient toward leaving coverage gaps.

**Fix:** `gap_penalty` now uses `compliance_denom`, matching `compliance_norm`. Because the
numerator still uses `penalty_rate` (≈ 138.75) and the implicit per-Mt buy cost is
`anchor_real` (≈ 67), a missed Mt is now strictly more expensive than buying it.

**Files changed:** `src/environment/ets_environment.py` (`compute_auction_rewards()`).

### Fixed — Soft 10% budget gate (Phase-1 cash-coverage check)

`step_auction()` zeroed `bid_q` whenever cash < 10% × `bid_p` × `bid_q`. The hard zero
created a gradient discontinuity, and the resulting `bid_qty_clip_ratio` observation
(dim [41]) fed the truncation back to the policy, which converged on whatever fit under
the cliff.

**Fix:** The gate now scales `bid_q` down to `cash / (bid_p × 0.10)` instead of zeroing
it. The agent still respects the cash constraint but retains a smooth gradient.

**Files changed:** `src/environment/ets_environment.py` (`step_auction()` budget gate
block), `tests/test_environment.py` (`test_budget_gate_scales_qty_for_cash_poor_agents`,
renamed and updated for soft-scale semantics).

### Fixed — `coverage_frac_auction` only gates savings, not costs

`_compute_rewards()` set `financial_reward = coverage_frac_auction × w_cost ×
(-cost_norm_centered)`. The multiplicative gate was symmetric: it scaled both the
positive savings branch (when `cost_norm_centered < 0`) and the negative cost branch
(when `cost_norm_centered > 0`). The asymmetry that emerged in expectation —
"skip the auction" → reward 0 vs. "win at clearing ≥ anchor" → reward < 0 — incentivised
under-bidding.

**Fix:** the gate is now applied **only** when `cost_norm_centered < 0` (agent under-spent
relative to expected cost). On the cost branch, `financial_reward = w_cost ×
(-cost_norm_centered)` is unscaled, so over-buying is no longer subsidised by partial
coverage. (No additive penalty added — per design directive.)

**Files changed:** `src/environment/ets_environment.py` (`_compute_rewards()` financial-
reward block).

### Changed — `banking_signal.imputed_cap_factor` raised 2.0 → 5.0

With `imputed_cap_factor = 2.0` and `compliance_denom = anchor × need`, the imputed
bank-drawdown norm saturated whenever `clearing_price > 2 × anchor` (≈ 134 EUR at
year 0). Above that threshold, drawing from the bank cost less in reward terms than
buying the same Mt at clearing — inverting the buy-vs-draw incentive in price spikes
and biasing policies toward larger banks and smaller auction bids.

**Fix:** raise the cap to `5.0 × compliance_denom`. The cap now binds only above
~5 × anchor (≈ 335 EUR), well into emergency price-containment territory, so the
imputed cost tracks clearing price linearly across the realistic auction range.
The `min(...)` clamp is retained to bound reward in pathological spikes, which is
already further bounded by the standard reward clip.

**Files changed:** `configs/default.yaml` (`reward.banking_signal.imputed_cap_factor`).

---

## [8.4.1]

### Changed — PCL reference floored at fundamental anchor

The bid price change limit (PCL) used `price_ma3` as its reference, which could drift
below the fundamental equilibrium price during low-price regimes, compressing the
allowed bidding range to an unrealistically narrow band.

**Fix:** The PCL reference is now `max(price_ma3, compute_fundamental_anchor(year, config, cap_t_actual=cap_t))`.
The actual MSR-adjusted cap (`cap_t`) is passed so the anchor reflects true scarcity,
not the linear LRF approximation.

**Files changed:** `src/environment/ets_environment.py` (`step_auction()` PCL block).

### Added — Budget price clip tracked as separate observation dim [40]

The soft budget clip (bid price clamped to 1.5× max affordable price) was silently
modifying actions with no gradient signal back to the agent.

**Fix:** `_last_budget_price_clip` (shape `n_total`, reset to 0 each episode) stores
`clipped_price − original_bid` (negative when clipped down). The Phase 1 observation
now exposes both clip signals as independent dims:

| Index | Name | Formula | Source |
|---|---|---|---|
| [39] | `last_bid_price_clip` | `(actual − requested) / price_max`, signed [-1,1] | PCL gate only |
| [40] | `last_budget_price_clip` | `(actual − requested) / price_max`, signed [-1,1] | Budget gate only |

Dims [40]–[41] (qty clip, invest clip) shift to [41]–[42]. **Phase 1 base: 42 → 43 dims.**

**Files changed:** `src/environment/ets_environment.py` (`__init__`, `reset()`,
`step_auction()`, `_get_obs_phase1()` call site), `src/environment/company.py`
(`get_observation_phase1()` signature, docstring, array, `obs_dim_phase1` property),
`tests/test_bid_change_limit.py` (dim references, new `test_obs_40_budget_price_clip_signal_range`),
`tests/test_company.py`, `tests/test_environment.py` (42 → 43 base dim references).

### Changed — Decoupled actor optimizers (auction and secondary stepped independently)

A single `actor_optimizer` covering both policy networks caused gradients from one
policy head to bleed into the other during the shared backward pass, corrupting
per-head gradient norms.

**Fix:** Two separate optimizers replace the combined one:

- `auction_optimizer` — tracks `auction_policy` parameters only
- `secondary_optimizer` — tracks `secondary_policy` parameters only
- `actor_optimizer` — backwards-compatibility alias for `auction_optimizer`

Each optimizer zeroes, clips, and steps independently. `save()` / `load()` updated
accordingly; `load()` falls back to the old `actor_optimizer` key for legacy checkpoints.

**Files changed:** `src/agents/ppo_agent.py` (`__init__`, `update()`, `update_happo()`,
`save()`, `load()`), `tests/test_mappo.py` (`test_separate_optimizers_step`).

### Added — Year-1 advantage floor and per-year advantage diagnostics

Year-1 transitions often receive large negative advantages early in training (insufficient
banked allowances, no price history), causing excessively aggressive policy updates that
destabilise year-0 learning.

**Fix:** In `compute_gae()`, advantages at buffer indices 2 and 3 (year 1 auction and
secondary) are clamped to a floor of −1.0 before normalisation. This is pre-normalisation
so the floor is in the same units as the raw advantage distribution.

`compute_gae()` also now returns `per_year_adv_mean` (shape `[n_years]`) in `buf_tensors`,
computed pre-normalisation as `mean(adv[2*yr], adv[2*yr+1])` for each year.
`train.py` logs `adv_yr1_mean_A1` (agent 0, year 1) to the episode CSV.

**Files changed:** `src/agents/ppo_agent.py` (`compute_gae()`), `scripts/train.py`
(capture + CSV field).

### Changed — Separate critic LR decay schedule

Previously the critic LR cosine decay used the same floor (`lr_min`) as the actor. A
lower floor for the critic is needed to sustain value accuracy late in training without
throttling actor exploration.

**New config keys:**
```yaml
ppo:
  critic_lr_decay: cosine     # independent decay mode for critic
  critic_lr_min: 0.00005      # floor; actor lr_min unchanged
```

`scripts/train.py` reads these separately and applies them to `critic_optimizer`
param groups independently of the actor decay.

**Files changed:** `configs/default.yaml`, `scripts/train.py`.

### Changed — Hyperparameter tuning (default.yaml)

| Parameter | Before | After | Reason |
|---|---|---|---|
| `auction.bid_change_limit.value` | 50.0 | 75.0 | More headroom for price discovery |
| `reward.normalizer_alpha` | 0.01 | 0.02 | Faster EMA response to reward scale shifts |
| `reward.gae_min_std` | 0.1 | 0.15 | Reduces over-confidence in low-variance regimes |
| `exploration.epsilon_final` | 0.03 | 0.02 | Tighter exploitation at convergence |
| `exploration.epsilon_decay_frac` | 0.40 | 0.45 | Slower exploration decay |
| `ppo.lr` | 0.0003 | 0.0002 | Lower actor LR for more stable late-training updates |
| `ppo.clip_eps` | 0.20 | 0.15 | Tighter trust region |
| `ppo.entropy_coef_final` | 0.01 | 0.005 | Less residual entropy at convergence |
| `ppo.entropy_decay_frac` | 0.90 | 0.70 | Faster entropy decay to encourage earlier exploitation |
| `ppo.mini_batch_size` | 32 | 64 | Larger batches reduce gradient variance |
| `ppo.episodes_per_update` | 16 | 32 | More on-policy data per update |
| `ppo.short_run_overrides.episodes_per_update` | 8 | 16 | Consistent 2× ratio with main setting |
| `ppo.log_std_min` | −2.5 | −3.5 | Allow slightly more deterministic policies |
| `ppo.critic_lr` | 0.001 | 0.0005 | Lower critic LR reduces value overfitting |
| `ppo.target_kl` | 0.02 | 0.015 | Tighter KL constraint |
| `ppo.critic_extra_epochs` | 4 | 6 | More critic epochs per update |
| `ppo.critic_huber_delta` | 10.0 | 4.0 | Narrower Huber region, less tolerance for large TD errors |

---

## [8.4]

### Added — Clip feedback signals in observation (4 new dims)

Agents previously had no way to know whether their requested action was modified by
internal gates (bid change limit, leverage cap, collateral budget, capex limit, secondary
market depth). Four new dimensions are now exposed so agents receive direct gradient
signal from constraint violations.

**Phase 1 observation base: 40 → 42 dims**

| Index | Name | Formula | Range |
|---|---|---|---|
| [38] | `pcl_headroom_norm` | `(pcl_ceiling − price_ma3) / price_max`, clipped [0,1] | [0, 1] |
| [39] | `last_bid_price_clip` | `(actual_bid − requested_bid) / price_max`, signed | [−1, 1] |
| [40] | `last_bid_qty_clip_ratio` | `actual_qty / requested_qty` | [0, 1] |
| [41] | `last_invest_clip_ratio` | `actual_invest_frac / requested_invest_frac` | [0, 1] |

The old dims [38]–[39] (`last_clearing_price_ref` and `bid_change_limit_norm`) are
replaced by these four. `pcl_ceiling` = `min(price_ma3 + bcl_value, price_max)`.

**Phase 2 observation: +11 → +12 extra dims**

| Index | Name | Formula | Range |
|---|---|---|---|
| [base+11] | `last_sec_qty_clip_ratio` | `actual_sec_qty / requested_sec_qty`, previous year | [−1, 1] |

All four clip arrays (`_last_bid_price_clip`, `_last_bid_qty_clip_ratio`,
`_last_invest_clip_ratio`, `_last_sec_qty_clip_ratio`) are stored on `ETSEnvironment`
(shape `n_total`) and reset to 0 / 1 at episode start.

**Files changed:** `src/environment/company.py` (`get_observation_phase1()`, `get_observation_phase2()`,
`obs_dim_phase1`, `obs_dim_phase2`), `src/environment/ets_environment.py` (`__init__`, `reset()`,
`step_auction()`, `step_secondary()`, `_get_obs_phase1()`, `get_observation_phase2()` call sites),
`tests/test_bid_change_limit.py` (26 new tests).

### Changed — Bid change limit: fixed value, MA3 reference, config restructure

The decaying-schedule BCL (two config keys `bid_change_limit_start` / `bid_change_limit_final`
set via `set_bid_change_limit()` in `train.py`) is replaced by a simpler, more robust design:

- **Fixed at 50 EUR/t** for the full run — no decay schedule.
- **MA3 reference**: the clipping window is now centred on the 3-year moving average of
  clearing prices (`price_ma3`) instead of `last_clearing_price`, making it more
  outlier-resistant.
- **Config restructure**: single nested key replaces the old pair.

```yaml
# Before (v8.3.2):
auction:
  bid_change_limit_start: 100.0
  bid_change_limit_final: 50.0

# After (v8.4):
auction:
  bid_change_limit:
    enabled: true
    value: 50.0      # fixed EUR/t; Year 0 always unconstrained
```

`set_bid_change_limit()` is now a no-op (kept for backwards compatibility).
The per-episode BCL schedule code in `scripts/train.py` is removed.

**Files changed:** `configs/default.yaml`, `configs/smoke_100.yaml`, `scripts/train.py`,
`src/environment/ets_environment.py` (`set_bid_change_limit()`, `step_auction()` bid clipping
block).

### Changed — `compute_fundamental_anchor` accepts actual cap (bypasses LRF approximation)

When the MSR is actively withholding, the effective cap falls below the linear-LRF
approximation (`cap_0 × (1 − lrf × year)`), causing `compute_fundamental_anchor` to
underestimate scarcity and the AR(1) price floor to drift low.

**Fix:** New optional parameter `cap_t_actual: float | None = None`. When provided,
it replaces the LRF linear estimate with the true MSR-adjusted cap from `CapSchedule`:

```python
def compute_fundamental_anchor(year, config, banking_premium_mult=None, cap_t_actual=None):
    ...
    if cap_t_actual is not None:
        cap_t = float(cap_t_actual)
    else:
        cap_t = max(cap_0 * (1.0 - lrf * year), cap_0 * 0.01)
```

`ETSEnvironment` now passes `cap_t_actual` at all three call sites:
`compute_auction_rewards()`, the AR(1) price floor update, and `_compute_rewards()`.

**Files changed:** `src/utils/price_anchor.py`, `src/environment/ets_environment.py`
(3 call sites).

---

## [8.3.2] (incorporated into 8.4)

### Added — MSR warm-start (realistic reserve at episode start)

During burn-in the Market Stability Reserve never triggers because TNAC stays well
below `tnac_upper`, leaving `_msr_reserve = 0` at the start of every episode. This
meant agents never encountered MSR withholding pressure during early years.

**Fix:** New config key `warm_start.msr_initial_reserve_frac` (default `0.0`; set to
`0.23` in `configs/default.yaml`). After burn-in completes, `_run_burnin()` seeds the
reserve to `msr_initial_reserve_frac × cap_year_0`, then clamps to `tnac_lower` (the
natural upper bound during normal operation).

At the default setting this pre-loads ≈ 0.23 × cap_year_0 Mt into the reserve,
matching realistic EU ETS reserve levels at the start of a 12-year window.

**Files changed:** `src/environment/ets_environment.py` (`_run_burnin()`),
`configs/default.yaml` (`warm_start.msr_initial_reserve_frac`).

### Added — Bid price change limit (stabilised price discovery)

Unconstrained bid jumps between years allow agents to escape any price-discovery
regime instantly, preventing coordinated convergence. A per-episode hard limit on
year-over-year bid price changes is now enforced in `step_auction()`.

The limit decays linearly from `bid_change_limit_start` to `bid_change_limit_final`
over the same episode window as epsilon decay:

| Config key | Default | Meaning |
|---|---|---|
| `auction.bid_change_limit_start` | `100.0` | EUR/t cap early in training |
| `auction.bid_change_limit_final` | `50.0` | EUR/t cap at convergence |

Year 0 within an episode is always unconstrained (no prior clearing price).
The current reference price and current limit are exposed to agents as two new
observation dimensions:

| Index | Name | Formula |
|---|---|---|
| [38] | `last_clearing_price_ref` | last clearing price / `price_max` |
| [39] | `bid_change_limit_norm` | current limit / `price_max` |

`obs_dim_phase1` updated from 38 to 40 base dims (total: 40 + 7×(N−1)).

**Files changed:** `src/environment/ets_environment.py` (`__init__`, `step_auction()`,
new `set_bid_change_limit()`), `src/environment/company.py` (`get_observation_phase1()`,
`obs_dim_phase1`), `scripts/train.py` (limit schedule + `set_bid_change_limit()` call),
`configs/default.yaml`.

### Changed — ESG speed bonus interpolated by year

The ESG signal `esg_raw = esg_scale × (ef_ratio + speed_coef × green_delta)` used a
single fixed `speed_coef`. Because `ef_ratio` is front-loaded by construction (year-0
investment earns 12 years of reward vs year-11 earning 1 year), late investors were
systematically under-rewarded for speed.

**Fix:** `speed_coef` is now linearly interpolated from `esg.speed_coef` at year 0 to
`esg.speed_coef_late` at the final year. New config key `esg.speed_coef_late: 0.8`
(up from the uniform `0.5`), compensating late investors with a higher speed bonus.

**Files changed:** `src/environment/ets_environment.py` (`_compute_rewards()`),
`configs/default.yaml` (`esg.speed_coef_late`).

### Fixed — Exploration price anchor is now year-adjusted

Epsilon-greedy exploration was centred on a static WTP estimate
(`mac + 0.5 × (penalty − mac) ≈ 93 EUR/t`) that ignored year progression, causing
over-exploration in early years and under-exploration in late years.

**Fix:** Both the uniform and anchored exploration modes now centre on
`compute_fundamental_anchor(current_year) × anchor_boost`, where `anchor_boost`
is a new config key (`exploration.anchor_boost: 1.14`). This gives:

| Year | Anchor | × 1.14 |
|---|---|---|
| 0 | ~67 EUR/t | ~76 EUR/t |
| 1 | ~70 EUR/t | ~80 EUR/t |
| 11 | ~101 EUR/t | ~115 EUR/t |

**Files changed:** `src/agents/ppo_agent.py` (`select_auction_action()`),
`configs/default.yaml` (`exploration.anchor_boost`).

---

## [8.3.1]

### Added — Cap scarcity lookahead in observation (Fix 5)

Two new dimensions added to the Phase 1 observation base vector (36 → 38D base):

| Index | Name | Formula | Purpose |
|---|---|---|---|
| [36] | `cap_ahead_3y_ratio` | `cap(t+3) / cap(t)`, clipped [0,1] | 3-year tightening signal |
| [37] | `cap_ahead_6y_ratio` | `cap(t+6) / cap(t)`, clipped [0,1] | 6-year tightening signal |

Values near 1.0 mean low near-term scarcity; values below 0.8 signal sustained cap tightening
that makes early banking or abatement investment rational. Computed in `ets_environment.py` via
`cap_schedule.get_cap()` and passed to `company.get_observation_phase1()`.

`obs_dim_phase1` property updated from `36 + opp_dims*(N-1)` to `38 + opp_dims*(N-1)`.

**Files changed:** `src/environment/company.py`, `src/environment/ets_environment.py`,
`tests/test_company.py` (3 asserts), `tests/test_environment.py:514`.

### Changed — Exploration and schedule auto-scaling (Fix 3 & 4)

All three decay schedules now read their auto-scale fraction from `configs/default.yaml` so the
schedule adapts proportionally to any episode count. Setting a value to `0` (auto) uses the
fraction; set it to an explicit integer to hard-override.

| Parameter | Config key | Old auto frac | New auto frac | At 100k eps |
|---|---|---|---|---|
| `epsilon_decay_episodes` | `exploration.epsilon_decay_frac` | 0.75 | **0.20** | 20 000 |
| `entropy_decay_window` | `ppo.entropy_decay_frac` | 0.90 | **0.30** | 30 000 |
| `shaping_decay_episode` | `reward.shaping_decay_frac` | 0.40 | **0.10** | 10 000 |

Additional exploration changes:

- `exploration.mode`: `"uniform"` → `"anchored"` (Gaussian centered on WTP, Fix 3)
- `exploration.epsilon_final`: 0.05 → 0.03 (lower mature noise floor, Fix 3)

**Files changed:** `configs/default.yaml` (new `*_frac` keys), `scripts/train.py`
(`EntropyConditionTracker`, epsilon and shaping resolution calls).

### Fixed — Penalty normalization denominator (non-compliance incentive bug)

`penalty_realized` and `remediation_cost` were both divided by `budget_real`
(annual budget / infl ≈ 800–1500 M€), while `compliance_norm_cash` is divided by
`compliance_denom` (anchor_real × need ≈ 200–400 M€). The 3–5× mismatch in scale
meant the reward signal from non-compliance was systematically weaker than the
reward signal from compliance, creating a marginal incentive to skip compliance.

**Example (1 Mt shortfall, anchor=80 €/t, need=3 Mt, budget=800 M€):**

| Term | Old formula | Value | New formula | Value |
|---|---|---|---|---|
| `compliance_norm_cash` (1 Mt at anchor) | cost / compliance_denom | 0.333 | unchanged | 0.333 |
| `penalty_realized` (1 Mt shortfall) | 138.75 / budget_real | **0.173** | 138.75 / (infl × compliance_denom) | **0.578** |

Old: skipping compliance nets +0.16 reward per Mt → agents non-compliant 1–6 of 12 years.
New: skipping compliance costs −0.245 reward per Mt → strong deterrent.

**Fix:** Changed denominator of `penalty_realized` from `budget_real` to
`infl × compliance_denom` (= `anchor_t × need`, the nominal compliance scale).
Changed denominator of `remediation_cost` from `budget_real` to `compliance_denom`
(`anchor_next_real` already carries the `/infl`, so the ratio stays inflation-invariant).

Both penalties are now on the same reward scale as compliance costs and correctly
exceed the market price (138.75 > 80 €/t), matching EU ETS design intent.

**Files changed:** `src/environment/ets_environment.py` (lines ~2633, ~2639).

---

## [8.3.0]

### Added — Banking timing signal

Agents previously had a structural incentive to zero-bid at auction: drawing from
their bank to cover compliance cost nothing in the reward, so skipping the auction
was always rational. This release closes that loophole and adds an explicit
intertemporal timing signal.

#### Root cause (exploit)

`compliance_norm` was computed purely from cash payments. An agent holding banked
allowances could satisfy its full compliance obligation at zero cash cost, receiving
no penalty and no reward cost — making zero-bidding a dominated strategy that left
unsold allowances rolling over into the next year's supply, depressing prices and
amplifying the exploit.

#### Fix A — Imputed bank drawdown in `compliance_norm`

Bank drawdown (the portion of compliance met by pre-existing holdings rather than
fresh market purchases) is now marked to the current clearing price and added to
`compliance_norm` via a new `imputed_bank_norm` term:

```
compliance_norm = compliance_norm_cash + w_imputed × imputed_bank_norm
```

where `imputed_bank_norm = min(drawdown × clearing_price / infl, cap) / compliance_denom`.

An agent drawing 3 Mt from its bank when the market clears at €80/t now faces the
same compliance cost as an agent that bought those 3 Mt fresh. Zero-bidding is no
longer costless.

#### Fix B — Banking timing P&L signal

A per-agent `banking_signal` rewards (or penalises) good intertemporal allocation:

```
banking_signal = w_banking × drawdown × (clearing_price − cost_basis) / (infl × compliance_denom)
```

where `cost_basis` is the weighted-average price paid for currently held allowances.
Agents that accumulated allowances cheaply and draw them when prices are high receive
a positive signal; agents that banked expensively and draw when the market is cheap
receive a negative signal. The two fixes are orthogonal: Fix A removes the zero-bid
exploit; Fix B adds a pure timing gradient on top.

#### Double-reward prevention

By construction the two signals do not interact: Fix A makes the compliance cost
identical whether allowances were bought or drawn, so no "budget savings" bonus
exists. Fix B then adds only the timing P&L — the value of having bought at a
different point in time.

#### Cost basis tracking

- `_bank_cost_basis[i]` (new `np.ndarray`, `n_total`) — per-agent weighted-average
  acquisition cost, updated after each auction allocation and secondary purchase.
- Initialised at `reset()` to `fundamental_anchor(year=0) × initial_bank_cost_factor`
  (default 0.80), reflecting that pre-episode holdings were accumulated when prices
  were historically lower.

### Changed

- `compliance_norm` now includes the imputed bank term when `banking_signal.enabled=True`.
  `compliance_norm_cash` (the pure cash component) is exposed separately for diagnostics.

### Config (`reward.banking_signal`)

```yaml
banking_signal:
  enabled: true
  w_banking: 0.3           # weight on timing P&L signal
  w_imputed: 1.0           # weight on imputed bank drawdown in compliance_norm
  imputed_cap_factor: 2.0  # cap imputed_bank_norm at N × compliance_denom
  initial_bank_cost_factor: 0.80  # pre-banked cost = fundamental_anchor(0) × factor
```

### Diagnostic channel changes (`_last_reward_channels`)

| Key | Status | Notes |
|---|---|---|
| `compliance_norm_cash` | NEW | cash-only component of `compliance_norm` |
| `imputed_bank_norm` | NEW | mark-to-market imputed cost of bank drawdown |
| `bank_drawdown` | NEW | Mt drawn from bank to cover compliance |
| `bank_cost_basis` | NEW | weighted-average acquisition price of current holdings |
| `banking_signal` | NEW | timing P&L reward term |

### Tests

`tests/test_banking_signal.py` — 20 new tests covering: cost basis init and update
mechanics, drawdown calculation, imputed cost equality, cap behaviour, signal
direction (positive/negative/zero), disabled mode, channel presence and finiteness,
and scale sanity (signal does not dominate the gradient).

---

## [8.2.0]

Full reward audit — six bugs identified and fixed across `_compute_rewards`
and `compute_auction_rewards`. Documented in `archive/docs/reward_redesign_v8_2.md`.

### Bugs fixed

- **Bug 1 — Penalty double-counted.** `penalty_norm` previously summed
  `penalty_prospective + penalty_realized` where both were driven by the same
  shortfall at the same penalty rate, charging non-compliance at 2×–3× the real
  economic cost. (Fix A)

- **Bug 2 — `capital_norm` wrong denominator.** Investment costs were divided by
  `compliance_denom` (≈€175M) instead of `budget_real` (≈€724M), over-penalizing
  green capex by ~4× and systematically discouraging the green transition the
  simulation is designed to teach. (Fix B)

- **Bug 3 — No revenue baseline (documented, not implemented).** Revenue had never
  entered the reward despite being listed in the design intent. This created a
  structural reward ceiling of zero for pure-financial agents. Fix C (re-adding
  revenue) was designed but deliberately excluded: electricity revenue is a
  pass-through that agents cannot materially influence by changing bidding or
  investment strategy, so including it adds a large constant without providing a
  useful gradient. The positive-reward problem is addressed instead via `esg.scale`
  (see Fix D). Revenue continues to be logged in `per_agent_diag` for diagnostics.

- **Bug 4 — `esg_anchor_ratio` silently muted ESG, breaking 50:50 balance.**
  `esg_anchor_ratio = min(compliance_denom / budget_real, 2.0)` evaluated to
  ≈0.07–0.24 in practice, scaling the ESG signal down by ~5× and producing an
  effective 80:20 cost:ESG weighting instead of the intended 50:50. (Fix D)

- **Bug 5 — Inflation invariance violated.** `penalty_realized` carried inflation
  quadratically: the inflated `penalty_cost` numerator was divided by the deflated
  `budget_real` denominator, resulting in a net `infl²` factor. By year 11
  (cumulative infl ≈ 1.24) penalties were ~1.54× more punishing than year 0 —
  directly contradicting the stated design goal of year-invariant reward scale. (Fix A)

- **Bug 6 — Phase-1 and Phase-2 rewards on different scales.** `compute_auction_rewards`
  used a fixed `/1000` scale while `_compute_rewards` used budget-relative
  normalization. Subtracting them in `train.py` to get `r_secondary` was
  dimensionally inconsistent, producing a ~7× gradient imbalance between the
  auction and secondary policy heads. (Fix E)

### Changed

- **Fix A — Penalty split into realized payment + forward remediation.**
  `penalty_realized` is now reconstructed in real terms (base rate, no inflation
  multiplication); `remediation_cost` represents carry-forward debt valued at
  next-year market anchor under scarcity. The scarcity "catch-up is harder" intent
  is preserved without double-charging the penalty rate.

- **Fix B — `capital_norm` divided by `budget_real`.** Investment now competes
  against the agent's actual budget envelope rather than its allowance bill.

- **Fix D — `esg_anchor_ratio` removed from ESG formula.** `esg_scale` is now the
  sole calibration knob. `esg_anchor_ratio` is retained in `_last_reward_channels`
  as `1.0` for log backward-compatibility but no longer multiplied into `esg_raw`.
  Default `esg.scale` raised to `2.0` so that a mid-journey ESG agent
  (`ef_ratio ≈ 0.5`) contributes roughly equal ESG and financial weight. Fully
  greened compliant agents (`ef_ratio ≈ 1.0`) can achieve a slightly positive net
  reward without requiring revenue in the signal.

- **Fix E — `compute_auction_rewards` budget-relative normalization.** Auction
  rewards now use the same `compliance_denom` / `budget_real` denominators as
  `_compute_rewards`. Phase-1 and Phase-2 gradients are proportionally consistent.

- **ESG compliance gate linearized.** Replaced `coverage_frac^(2 × gate_activation)`
  with `compliance_gate = coverage_frac` (linear). The quadratic form suppressed
  the ESG signal exactly when scarcity is high and greening matters most (late
  years, low coverage).

- **Terminal queue guard made smooth.** Replaced hard `if effective_remaining < 1.0:
  continue` with `remaining_scale = min(1.0, effective_remaining / 2.0)` multiplied
  into the queue value. Projects taper continuously: 0 years → 0% credit, 1 year →
  50%, ≥2 years → 100%. Removes the cliff while still discouraging last-minute
  investments.

- **Terminal bank formula piecewise.** Replaced `log(1 + ratio)` with a piecewise
  formula: linear below annual need (ratio < 1), log above. The linear lower branch
  provides a stronger bidding incentive under terminal scarcity; the log upper
  branch preserves diminishing returns for overbanking.

- **`loan_sting` normalizer fixed.** Emergency-loan origination cost now divided by
  `budget_real` (inflation-deflated) instead of `company.annual_budget` (nominal),
  consistent with all other reward normalization.

- **`cost_norm` centered for reward comparability.** `base_reward` now uses
  `w_cost × (−cost_norm_centered)` where `cost_norm_centered = cost_norm − 1.0`.
  By construction, `compliance_denom = anchor_real × annual_need`, so a fully-compliant
  agent buying exactly at the anchor pays `compliance_norm = 1.0`. Subtracting 1.0
  centers the cost contribution at zero for normal operation: zero = perfectly efficient,
  positive = under-spent, negative = over-spent. This makes financial and ESG reward
  scales comparable — a financial agent at normal operation achieves R ≈ 0 (the same
  break-even as a mid-journey ESG agent) without changing the ESG signal or adding any
  agent-type–specific conditional logic. `cost_norm_centered` and
  `expected_compliance_norm` added to `_last_reward_channels` for diagnostics.

### Diagnostic channel changes (`_last_reward_channels`)

| Key | Status | Notes |
|---|---|---|
| `compliance_norm` | unchanged | |
| `capital_norm` | redefined | now `/ budget_real` (was `/ compliance_denom`) |
| `soft_norm` | unchanged | |
| `cost_norm` | unchanged | sum of above three + `loan_sting` |
| `revenue_norm` | NEW (always 0.0) | revenue removed from reward; key retained for log compatibility |
| `penalty_norm` | redefined | now `penalty_realized + remediation_cost` |
| `penalty_realized` | redefined | real-terms realized penalty (base rate, inflation-invariant) |
| `penalty_prospective` | alias | now equals `remediation_cost` (kept for back-compat) |
| `remediation_cost` | NEW | carry-forward debt × next-year anchor × scarcity |
| `anchor_next_real` | NEW | expected next-year market anchor (real terms) |
| `carry_forward_debt` | NEW | `company._carry_forward` after compliance |
| `esg_signal` | redefined | no longer multiplied by `esg_anchor_ratio` |
| `esg_anchor_ratio` | redefined | always `1.0` (legacy log channel) |
| `base_reward` | redefined | revenue term removed |

`_last_auction_reward_channels`: all keys now record budget-normalized values
instead of `value / 1000`. Two new keys: `compliance_norm`, `capital_norm`.

### Removed

- `price_ma3_now` computation inside `_compute_rewards` (no longer needed once
  revenue is removed; `step_secondary` computes its own copy for `per_agent_diag`).
- `gate_activation` re-assignment inside the ESG block; variable retains its
  initialised value of `1.0` and is still logged for back-compatibility.
- `esg_anchor_ratio` as an active ESG multiplier (retained as a `1.0` log channel).

---

## [8.1.1]

### Added
- Inflation-deflated three cost buckets: `compliance_cost_real`, `capital_cost_real`, `soft_penalty_real` — all costs divided by `company.inflation_factor(current_year)` before normalisation.
- `budget_real` anchor for penalty and ESG normalisation: dynamically equals `company.annual_budget / infl` (mode `"dynamic"`) or a fixed reference budget (mode `"fixed"`). Config keys: `reward.budget_norm_anchor`, `reward.budget_norm_budget_0`.
- Scarcity-amplified prospective penalty: `shortfall × penalty_rate × (1 + scarcity_t) × urgency_scalar / budget_real`, where `scarcity_t = max(0, 1 − cap_t / cap_0)`.
- ESG speed bonus: `esg_speed_coef × max(0, green_frac − prev_green_frac)` added inside `esg_raw_unanchored`. Config key: `esg.speed_coef` (default 0.5).
- ESG fragility cap: `esg_anchor_ratio = min(compliance_denom / budget_real, 2.0)` prevents ESG from dominating for small-budget agents.
- Extended `_last_reward_channels` with diagnostic fields: `compliance_norm`, `capital_norm`, `soft_norm`, `penalty_prospective`, `penalty_realized`, `scarcity_amp`, `esg_anchor_ratio`, `anchor_real`, `budget_real`, `infl`, `shortfall`.

### Changed
- Removed fixed `REWARD_SCALE = 1000.0` divisor from `_compute_rewards`; all costs are now normalised by economically meaningful denominators (`compliance_denom` for compliance/capital, `budget_real` for soft/penalty/ESG).
- ESG signal no longer carries a `time_ratio` decay factor; ESG is equally valued in early and late years. Formula: `esg_raw_unanchored = esg_scale × (ef_ratio + speed_bonus)`.
- Terminal bank and queue values now scaled by per-agent `budget_real_t` instead of the fixed `REWARD_SCALE`.

### Breaking
- Old checkpoints produce identical network behaviour but reward scale changes; re-running diagnostics recommended.
- `_last_reward_channels` keys `cost_norm`, `penalty_norm` are still present; new keys added alongside them.

---

## [8.1.0]

### Added
- Corporate treasury reserve (unspent budget retention at 60%, 1.5× cap, 5% decay, terminal NPV)
- Leverage-scaled emergency loan with capex covenant squeeze (`effective_capex_throughput`)
- Anchor-normalised cost reward (`cost_norm = total_cost / (anchor_t × estimated_need)`; `penalty_norm` still uses static `REWARD_SCALE`)
- 7D lagged opponent observation with 1-year two-buffer lag (emissions, green_frac, fossil_frac, queue_noisy, bank_norm, net_secondary_norm, lagged_compliance_gap_norm)
- Phase 2 signed compliance gap dimension (`compliance_gap_norm` at `base+10`)
- Budget price clipping: soft clip at 1.5× max_affordable_price, collateral-aware

### Changed
- Suspension mechanism replaced by budget-based bid gate (cash < 10% of notional → qty zeroed)
- Settlement logic unified under canonical waterfall (operating → treasury → loan → default)
- Opponent obs default mode: `lagged` (7D); `full_info` (6D) available for ablation
- Year-end order: `settle_treasury_year_end()` → `apply_loan_repayment()` → `reset_budget()` → `reset_capex_budget()`

### Breaking
- Phase 1 obs: `36 + 7*(N−1)` dims (was `36 + 6*(N−1)`)
- Phase 2 obs: Phase 1 + 11 dims (was Phase 1 + 10)
- Old checkpoints incompatible unless `opponent_obs.mode: full_info` and legacy reward config

---

## v8.0.1

**Bug fix: MSR cancellation threshold, emergency release gate, burn-in reserve clamp, dynamic budget ceiling, EU ETS calibration**

### MSR cancellation threshold fix (`src/environment/cap_schedule.py`)
- **Bug**: `_apply_msr` cancelled MSR holdings above `max(prev_auction_vol, prev_cap)`, which varied with rollover-distorted auction volumes and could suppress legitimate reserves or cause premature cancellation.
- **Fix**: Cancel holdings above `self.tnac_lower` instead. This anchors cancellation to the legislative lower TNAC band (a stable, config-driven threshold) and removes the `prev_auction_vol` / `prev_cap` dependency entirely.
- `preview_auction_volume` mirror updated identically: `msr_snap = max(0.0, self._msr_reserve - max(0.0, self._msr_reserve - self.tnac_lower))`.

### Emergency price-release gate (`src/environment/cap_schedule.py`, `configs/default.yaml`)
- Added `msr.price_release_enabled: false` config flag (default `true` for backward compat).
- When `false`, the combined emergency-release block is skipped entirely in both `_apply_msr` and `preview_auction_volume`.
- Disabled by default in v8.0.1 to prevent procyclical reserve injection during early low-price episodes.

### Absolute price threshold tightening (`configs/default.yaml`)
- `msr.price_containment_absolute`: 350 → **175 EUR/t**
- `msr.price_release_absolute`: 450 → **212 EUR/t**
- Both now sit below `auction.price_max = 250 EUR/t`; price_max sanity-check warnings added to `CapSchedule.__init__` via `warnings.warn`.

### Burn-in MSR reserve clamp (`src/environment/ets_environment.py`)
- After the hidden burn-in loop and before `_calibrate_post_init_bank`, clamp `cap_schedule._msr_reserve = min(_msr_reserve, tnac_lower)`.
- Prevents burn-in from overfilling the reserve beyond the lower band, matching the real 2026 EU ETS starting state.

### Dynamic budget ceiling (`src/environment/company.py`, `configs/default.yaml`)
- Added `budget.dynamic_budget_ceiling_multiplier: 1.5`.
- `compute_dynamic_budget` now caps its output at `min(dynamic_budget, annual_budget × ceiling_mult)`, preventing revenue windfalls from inflating available budgets unboundedly.

### EU ETS calibration anchoring (`configs/default.yaml`)
- `ets.cap_overhead_pct`: 0.02 → **0.12** (reflects real 2026 EU ETS ~15% overhead).
- `msr.tnac_upper_ratio`: 0.36 → **0.68** (derived from band_width / (LRF × cap_0) ≈ 10 years, matching EU ETS design intent).
- `ets.initial_bank_fraction`: 0.10 → **0.81** — jointly calibrated with `tnac_upper_ratio` so TNAC starts at 1.05 × tnac_upper, replicating the real 2026 EU ETS observed ratio (1,148 / 1,096 = 1.05).
- All downstream parameters (`tnac_lower`, `tnac_mid`, `release_amount`) auto-derive from `tnac_upper_ratio` in `market_calibration.py` — no further code changes needed.

### Training log: collateral clip events (`scripts/train.py`)
- Per-episode `_collateral_clip_events` total now surfaced in the Warnings line when non-zero: `collateralClip=<N>`.
- Legend updated to document the new counter.

---

## v8.0.0

**Tabula-Rasa Retirement, Fundamental Price Anchor, WTP-Uniform Exploration, Coverage Shaping Unification**

### Tabula-rasa retirement (`scripts/train.py`, `configs/default.yaml`)
- `tabula_rasa.enabled=true` now raises `ValueError` in `train_one_seed()` — it is no longer a valid runtime override.
- The `tabula_rasa` block is kept in `default.yaml` with `enabled: false` as an ablation reference only.
- Rationale: tabula-rasa accumulated too many conflicting overrides over v5-v7. The intended "start from scratch" behavior is now the default (no BC pretraining, no KL anchor, uniform exploration), making the override redundant.

### Behavioral cloning and KL anchor disabled by default (`configs/default.yaml`, `src/agents/ppo_agent.py`)
- `pretrain.enabled: false` — BC warm-start is off in the default profile.
- `ppo.kl_anchor_beta: 0.0` — KL regularization toward frozen BC policy is off.
- Both remain configurable for controlled ablation studies.

### Fundamental price anchor (new file `src/utils/price_anchor.py`)
- **`compute_fundamental_anchor(year, config)`**: Economically grounded expected clearing price derived from MAC cost (~48 EUR/t), banking premium multiplier (1.4×), cap scarcity (linear LRF approximation), and inflation-adjusted effective penalty rate.
  - Formula: `mac_anchored + scarcity × (eff_penalty − mac_anchored)` where `scarcity = 1 − cap_t / cap_0`.
  - Default output: ~67 EUR/t at yr0, ~101 EUR/t at yr11.
  - No dependency on `CapSchedule` — resolves `cap_year_0` via three-priority fallback: (1) `ets["cap_year_0"]` runtime-calibrated value, (2) `ets["cap_year_0_override"]` YAML override, (3) estimate from `initial_mix × output_twh × emission_factors × (1 + cap_overhead_pct)`.
- **`PPOAgent.inject_fundamental_anchor(year)`**: Seeds `price_head.bias` so the initial policy mean ≈ fundamental anchor for year `year`. Called once per episode at episode start.
- **AR(1) mean-reversion floor** (`src/environment/ets_environment.py`): Static `ar1_floor` replaced with `compute_fundamental_anchor(year, config)` at the three AR(1) model update sites, so the expected price signal tracks economically grounded values as the cap tightens.
- **`price.initial_expected` removed** from `default.yaml`: This static scalar is no longer needed; `ets_environment.py` falls back to `70.0` via `.get("initial_expected", 70.0)` for backward compatibility with any existing checkpoints.

### WTP-uniform exploration (`src/agents/ppo_agent.py`)
- Epsilon-random auction bids now sample 50/50 below/above the agent's **willingness-to-pay (WTP) anchor** rather than uniformly across the full price range.
- WTP anchor: `wtp_economic = mac + 0.5 × (penalty − mac) ≈ 93 EUR/t`. Incorporates budget headroom from observation dim 27 when available.
- Prevents systematic over-exploration above economic ceiling; keeps random bids interpretable as feasible market prices.
- `exploration.mode: "uniform"` is the only supported mode (tabula-rasa override removed).
- `exploration.auction_anchors` retained as optional per-head anchor override for ablation.

### Coverage shaping unification (`configs/default.yaml`, `src/environment/ets_environment.py`)
- Removed root `coverage_shaping` block (was a duplicate, never properly wired to the reward path).
- Removed `reward.coverage_credit_weight` and `reward.gap_closure_weight` keys (dead config fields).
- Single unified block: `reward.coverage_gap_shaping` with sub-keys `enabled`, `weight`, `target_coverage`.
- Code reads exclusively from `reward.coverage_gap_shaping`; no fallback to old keys.

### Price range reduction (`configs/default.yaml`, `configs/smoke_100.yaml`)
- `auction.price_max`: 500 → **250 EUR/t**.
- Rationale: 500 EUR/t is nearly 4× the penalty rate (138.75 EUR/t) — no rational compliance agent bids there. The tighter range reduces exploration waste and aligns the action space with economically plausible prices.
- `secondary.sec_price_max_mult` adjusted accordingly in both configs.

### Config cleanup (`configs/default.yaml`)
- Stale comments and deprecated `price.initial_expected` key removed.
- `tabula_rasa` block retained with `enabled: false` and a clear ablation-reference comment.
- All `coverage_shaping`, `coverage_credit_weight`, `gap_closure_weight` references purged.

### Tests added / updated
- **`tests/test_price_anchor.py`** (new, 5 tests): `compute_fundamental_anchor()` calibration — year-0 value, monotonic increase, clipping to price bounds, config parameter sensitivity, `banking_premium_mult` kwarg override.
- **`tests/test_anchors.py`**: Added `test_inject_fundamental_anchor_sets_price_near_anchor` — verifies `inject_fundamental_anchor(0)` sets policy mean within ±5 EUR/t of the year-0 anchor.
- **`tests/test_tabula_rasa.py`** (rewritten): Tests now verify (1) `tabula_rasa.enabled=true` raises `ValueError`, (2) `enabled=false` is a no-op, (3) WTP-uniform exploration is side-balanced around WTP anchor (not midpoint), (4) null anchors fall back to WTP anchor not midpoint.
- **`tests/test_environment.py`**: Fixed `test_defaulted_volume_not_double_counted_with_unsold_rollover` for v8 price_max change (budget override now triggers defaults under 250 EUR/t price cap).
- All **279 tests pass**.

---

## v7.13.1

**Phantom anchor decoupling + GAE std-floor hardening + delayed ESG gate activation**

### Phantom bidder (`src/environment/phantom_bidder.py`, configs)
- Phantom price anchor now uses `max(price_fundamental_frac × effective_penalty_rate, reserve + min_above_reserve)`,
  decoupling phantom demand from MA3 drift.
- Added config key `phantom_bidder.price_fundamental_frac: 0.60`.
- Updated default phantom parameters:
  - `qty_frac_lo: 0.15`
  - `qty_frac_hi: 0.35`
  - `price_lognormal_sigma: 0.35`
- Added regression test `tests/test_phantom_bidder.py::test_phantom_anchor_independent_of_ma3`.

### PPO / GAE normalization (`src/agents/ppo_agent.py`, configs)
- Added config key `reward.gae_min_std` (default `0.1`).
- `compute_gae()` now uses this std floor in both auction/secondary phase reward normalization blocks
  (replacing the previous near-zero floor).
- Added regression test `tests/test_rewards.py::test_gae_produces_nonzero_advantages_constant_reward`.

### Reward function gate scheduling (`src/environment/ets_environment.py`)
- ESG compliance gate now ramps in with shaping decay:
  - `gate_activation = clamp(1 - shaping_weight / 0.5, 0, 1)`
  - `compliance_gate = coverage_frac ** (2 * gate_activation)`
- Added `gate_activation` to `reward_channels` diagnostics.

### Training diagnostics (`scripts/train.py`)
- Added phantom bidder fields to logs:
  - Episode CSV: `phantom_avg_bid_price`, `phantom_avg_bid_qty` (alongside `phantom_active_pct`)
  - Year CSV: `phantom_bid_price`, `phantom_bid_qty`, `phantom_active`
- Console `Bid/yr` line now prints phantom active rate plus average phantom price/qty.

---

## v7.13.0

**Equilibrium-Breaking Mechanisms + Phantom-Bidder-Aware Scarcity Calibration**

### Phantom Bidder (new file `src/environment/phantom_bidder.py`)
- **`PhantomBidder` class**: Models financial intermediary demand (~40% of real EU ETS
  auction volume; Regulation 1031/2010). Bids with `LogNormal(log(anchor), σ=0.45)` price
  anchored to MA3 and `Uniform[5%,20%]` quantity fraction of auction supply.
- **Integration (`ets_environment.py` `step_auction()`)**: Phantom injected as `agent_id=n_total`
  row; `market_clearing_ets` called with `n_agents=n_total+1`; phantom slot stripped from
  returned allocations/payments. Phantom allocation discarded (not credited to any agent).
- **Logging**: `phantom_bid_price`, `phantom_bid_qty`, `phantom_active` written to year log.
  `phantom_active_pct` written to episode CSV. Console `Bid/yr` line shows `(+Ph X%)` suffix.
- **Config block** (`default.yaml`, `smoke_100.yaml`): `phantom_bidder.enabled`,
  `qty_frac_lo/hi`, `price_lognormal_sigma`, `price_min_above_reserve`,
  `price_max_frac_penalty`, `price_min_below_reserve_buffer`.

### Scarcity recalibration (`configs/default.yaml`, `configs/smoke_100.yaml`)
- **`ets.cap_overhead_pct`**: `-0.20 → +0.08` in default; `+0.11 → +0.08` in smoke.
  Rationale: the prior -20% overhead yielded extreme scarcity even before the phantom.
  With phantom expected to consume ~11% of auction supply, `+0.08` gives compliance agents
  a net ~4% supply shortfall in year 0 — "just minorly scarcer than what they need".
  The LRF compounds scarcity naturally over the 12-year episode.

### ESG Compliance Gate (`ets_environment.py` `_compute_rewards()`)
- `compliance_gate = (coverage_frac)²` multiplies into `esg_signal`.
  Non-compliant agents lose ESG credit proportionally (80% coverage → 64% ESG signal).
- `compliance_gate` and `esg_vs_penalty_ratio` added to `_last_reward_channels[i]`
  and stored in the year-level log (`log["reward_channels"]`).
- `esg.scale` lowered `3.5 → 2.0` to compensate for gate interaction.
- Episode CSV: `esg_vs_penalty_ratio_A{i}` and `compliance_gate_A{i}` per agent.
- Console warning fires when `esg_vs_penalty_ratio > 1.0` for a non-compliant agent
  for `diagnostics.esg_over_penalty_warn_window` (default 50) consecutive episodes.

### Liquidity Pool Floor (`ets_environment.py` `_settle_double_auction()`)
- After EMA update: `_liquidity_ref_ema = max(_liquidity_ref_ema, floor_frac × penalty_rate)`.
  Default `floor_frac = 0.25` prevents secondary market from collapsing to reserve when
  primary auctions fail repeatedly.
- Config key: `secondary.liquidity_pool.floor_fraction_of_penalty: 0.25`.

### HPP Heuristic Seeding (`scripts/train.py`)
- After BC pretraining, `seed_count` copies of BC snapshot are seeded into each agent's
  HPP pool. These BC-trained WTP-bidding opponents remain present as diversity anchors
  for the entire training run.
- `hpp_min_pool_sizes` protects BC seeds from immediate eviction.
- Config: `hpp.seed_heuristic: true`, `hpp.seed_count: 2`.

### Private Urgency Scalars (`ets_environment.py` `reset()`, `_compute_rewards()`)
- Per-episode `LogNormal(0, σ=0.30)` scalar per learning agent multiplied into effective
  penalty in reward. `E[scalar]=1.0`, range ≈ 0.5–2.0. Breaks symmetric cost structure
  that supports the floor-bidding equilibrium.
- Config: `urgency_scalars.enabled: true`, `urgency_scalars.lognormal_sigma: 0.30`.

### Diagnostics (`scripts/train.py`)
- Episode CSV additions: `esg_vs_penalty_ratio_A{i}`, `compliance_gate_A{i}`,
  `phantom_active_pct`.
- `_esg_over_penalty_streak` per-agent counter triggers console warnings.
- Year-level log now stores `log["reward_channels"]` dict for CSV/diagnostic access.

### Tests (`tests/test_market_clearing.py`)
- Three new phantom bidder tests:
  - `test_phantom_above_reserve_squeezes_floor_bidders`: phantom consumes supply, floor bidders get less.
  - `test_phantom_below_reserve_rejected_floor_bidders_unaffected`: below-reserve phantom is harmless.
  - `test_phantom_fills_entire_supply_floor_bidders_zero`: 100% consumption case.
- All 272 tests pass.

---

## v7.12.0

**Rollback of Non-Approved v7.11 Defaults + Reward Corrections**

### Market/config rollbacks (`configs/default.yaml`)
- **Auction quantity multiplier range widened** to preserve auction/secondary strategy space:
  - `auction.qty_mult_low`: `0.85 -> 0.5`
  - `auction.qty_mult_high`: `1.5 -> 2.0`
- **Unsold handling reverted to rollover**:
  - `ets.unsold_to_msr`: `true -> false`
  - Unsold allowances are now rescheduled into next-year supply by default.
- **Carry-forward cap reverted**:
  - `penalty.carry_forward_cap`: `0.5 -> 1.0`
- **Shaping floor reset**:
  - `reward.shaping_weight_floor`: `0.10 -> 0.0`
- **Dead reward config declarations removed** from default profile:
  - `reward.terminal_queue_value`
  - `reward.coverage_credit_weight`
  - `reward.gap_closure_weight`

### Reward-function rollbacks (`src/environment/ets_environment.py`)
- **Soft budget/capex penalties restored** inside `cost_norm`:
  - `budget_penalty = company.compute_budget_penalty()`
  - `capex_penalty = company.compute_capex_penalty()`
  - Both terms are added to total cost before normalization.
- **Terminal bank valuation reverted to diminishing returns**:
  - Restored `log1p` form (with existing 2x annual-need cap) instead of linear valuation.
- **Terminal queue valuation restored** with anti-gaming control:
  - Queue value is back in final-year reward.
  - Added **completion-fraction discount** so long-lead projects started too late receive little/no terminal credit.

### Documentation/overlay cleanup
- Removed redundant `terminal_queue_value` overrides from:
  - `configs/qlearning.yaml`
  - `configs/smoke_100.yaml`

### Tests
- Added regression test for completion-discount queue valuation:
  - `tests/test_rewards.py::TestTerminalQueueValue::test_terminal_queue_completion_fraction_discount`
- Updated shaping-floor expectation test to match zero floor.
- Full suite validation:
  - `265 passed, 12 warnings` (`python -m pytest`)

---

## v7.11.0

**Pure MARL Default + Scarcity-First Market Calibration + Reward Simplification**

### Market structure and policy defaults (`configs/default.yaml`)
- **Participant mix switched to pure MARL by default**:
  - `companies.n_bot_agents`: `8 -> 0`
- **Structural scarcity introduced from year 0**:
  - `ets.cap_overhead_pct`: `+0.02 -> -0.10`
- **Price floor raised near MAC anchor**:
  - `auction.reserve_price`: `30.0 -> 45.0`
  - `auction.price_min`: `30.0 -> 45.0`
  - `secondary.sec_price_min`: `30.0 -> 45.0`
- **Auction quantity range tightened**:
  - `auction.qty_mult_low`: `0.3 -> 0.85`
  - `auction.qty_mult_high`: `2.0 -> 1.5`
- **Unsold volume moved to MSR path**:
  - `ets.unsold_to_msr`: `false -> true`
- **Carry-forward tolerance tightened**:
  - `penalty.carry_forward_cap`: `1.0 -> 0.5`
- **Initial bank seed increased**:
  - `stochastic.bank_seed_min/max`: `0.05/0.15 -> 0.15/0.35`
  - `ets.initial_bank_fraction`: `0.10 -> 0.25`

### Reward and training behavior changes (`src/environment/ets_environment.py`)
- **Reward normalization moved to fixed global scale** (`REWARD_SCALE=1000`) instead of per-agent budget divisors.
- **Removed baseline-cost subtraction** from reward.
- **Removed shaping channels from active reward path**:
  - green bonus
  - efficiency bonus
- **Removed soft budget/capex penalties from reward path** (constraints shifted to hard mechanical gating/clipping).
- **Terminal valuation changes**:
  - bank value switched from diminishing `log1p` form to linear value
  - terminal queue value removed from reward path
- **Reward channel diagnostics reduced** to active core channels (`cost_norm`, `penalty_norm`, `esg_signal`, `base_reward`).
- **Diagnostic score clamp fix**: `S_financial` bounded in `[0, 1]`.

### Exploration schedule updates (`configs/default.yaml`)
- `tabular.epsilon_start`: `0.50 -> 0.30`
- `tabular.epsilon_decay_frac`: `0.50 -> 0.80`
- `tabular.critic_warmup_frac`: `0.10 -> 0.03`
- `tabular.shaping_decay_frac`: `0.33 -> 0.60`
- `reward.shaping_weight_floor`: `0.00 -> 0.10`

### Auction mechanism extension (`src/auction/market_clearing_ets.py`)
- Added configurable pricing mode support:
  - `auction.pricing_rule: uniform | pay_as_bid`
- Default remained `uniform`; `pay_as_bid` added for comparative market-design experiments.

### Tooling and tests
- Added periodic training CSV snapshots (`logging.snapshot_interval: 2500`) in `scripts/train.py`.
- Adapted tests to the new default profile and reward channel set (bot-count overrides, schedule expectations, reward-channel keys).

### Known config-only declarations in v7.11
- Declared but unused at runtime in that release:
  - `reward.coverage_credit_weight`
  - `reward.gap_closure_weight`
  - `reward.terminal_queue_value`

Historical note:
- Several v7.11 defaults above were intentionally rolled back in `v7.12.0` based on approval scope.

---

## v7.8.0

**Dual-Ceiling WTP Heuristic, Marginal EF Revenue (instance method), Compliance-Priority Investment, Collateral Warning Counter, Enhanced Diagnostics, Config Updates, Smoke Tests**

### M1 — Marginal EF for Revenue Computation (`ets_environment.py`, `company.py`)
- **`_compute_marginal_ef()`**: Moved from module-level function to ETSEnvironment instance
  method. Computes the emission factor of the most carbon-intensive technology with ≥5%
  system-wide capacity share (soft blend 3–8%). Stores result as `self._current_marginal_ef`.
  Reference: Fabra & Reguant (2014) AER; Sijm et al. (2006) Energy Policy.
- **`compute_revenue()` signature updated**: Parameters renamed to `(marginal_ef, carbon_price,
  inflation_factor)`. Formula: `marginal_price = base_price + passthrough * carbon_price * marginal_ef`;
  `revenue = output_twh * marginal_price * inflation_factor`.
- **`compute_dynamic_budget()` updated**: New signature `(carbon_price, marginal_ef, current_year)`.
  `carbon_price_for_budget = self._price_ma3 if > 0 else config["price"]["initial_expected"]`.
- **Logging**: `self._last_system_ef`, `self._last_marginal_ef`, `self._current_marginal_ef` stored.

### Heuristic Refactor — `heuristic_policy.py`
- **Dual-ceiling WTP bid price**: Replaces the previous single-ceiling formula with two ceilings:
  - `wtp_economic = market_anchor + urgency * max(0, penalty_rate + expected_future_price - market_anchor)`
    capped at `(penalty_rate + expected_future_price - 1.0)`.
  - `wtp_budget = max_compliance_share * available / max(qty_clipped, 1e-6)` where
    `max_compliance_share = config["bots"]["max_compliance_share"]` (default 0.70).
  - `bid_price = max(min(wtp_economic, wtp_budget), reserve_price + 1.0)`.
  - Stores `_last_wtp_economic`, `_last_wtp_budget`, `_last_wtp_binding` on company.
- **Qty target computed before bid price** to provide denominator for `wtp_budget` ceiling.
- **Physical-need quantity**: `qty_target = annual_need + carry_fwd + 0.1 * annual_need * urgency`.
- **Compliance-priority investment**: `invest_frac` scaled down by post-compliance budget headroom.
- **Secondary**: `spend_frac = 0.9` when `carry_forward > 0.01`.

### E2 — Collateral Clip Safety Net (`ets_environment.py`)
- Collateral clip gating in `step_auction()` uses expected-clearing sizing:
  `expected_clearing = max(effective_reserve, price_ma3)`,
  `expected_collateral = collateral_fraction * max(0, bid_price - expected_clearing) * bid_qty`.
- If `expected_collateral > max_collateral_budget_share * budget_remaining`, bids are rescaled.
- `self._collateral_clip_events` (agent_id → count) is reset each episode, incremented when clip
  fires, and logged at episode end via `year_log["collateral_clip_events_episode"]` and
  `year_log["collateral_clip_rate_episode"]`.
- Validation expectation: bot-only runs should have ~0 clip events; non-zero indicates a
  heuristic/environment mismatch.

### Config Updates (`configs/default.yaml`)
- `electricity.base_price`: 50.0 → 55.0
- `electricity.carbon_passthrough`: 0.80 → 0.90
- `bots.max_compliance_share: 0.70` added (dual-ceiling WTP budget fraction).
- **Debt headrooms**: `[400, 400, 230, 230, 50, 50, -100, -100]`.
- **BC pretrain max**: 2000 episodes.

### Diagnostics (`ets_environment.py`, `train.py`)
- **`per_agent_diag`** extended with: `wtp_economic`, `wtp_budget`, `wtp_binding`,
  `available_budget`, `compliance_cost_share_of_budget`.
- **`log["marginal_ef_used"]`** added to step log.
- **Year-log CSV** (`train.py`): 7 new per-agent fields per bot per year:
  `wtp_economic`, `wtp_budget`, `wtp_binding`,
  `invest_frac_pre_clip`, `invest_frac_post_clip`,
  `available_budget`, `compliance_share_of_available`.
  Plus env-level: `marginal_ef_used`.

### Tests (`tests/test_compliance_validation.py`)
- New smoke test: runs 1 heuristic-only episode, asserts coal-bot coverage ≥ 0.85,
  zero defaults, clearing price above reserve, and within [30, 200] EUR/t.

### Item 7 — Loan Ordering Verified (`auction_settlement.py`)
- Confirmed: loan eligibility check fires **before** the default/suspension branch.
  No code change required; ordering is already correct.

### Held for Next Iteration (not enabled) — `configs/default.yaml`
- `coverage_shaping` block (disabled): post-auction coverage bonus decaying over 30% of training.
- `cap_curriculum` block (disabled): cap multiplier 1.3→1.0 over 30% of training.
- `budget_curriculum` block (disabled): budget multiplier 1.5→1.0 over 30% of training.

---

## v7.7.0

**Revenue-Based Budget, Emergency Loans, Observation Enrichment, Budget Hardening, Reward Channels, Heuristic Loan-Awareness**

### Phase A — Revenue-Based Dynamic Budget (`company.py`, `ets_environment.py`, `default.yaml`)
- **Dynamic budget mode**: `budget.mode: revenue_based` computes annual budgets from
  `Company.compute_revenue()` (electricity revenue with carbon-cost passthrough) minus
  operating costs plus archetype-specific `debt_headroom`. EMA smoothing (`ema_alpha=0.3`)
  prevents erratic year-to-year swings.
- **`compute_dynamic_budget()`**: Called in `step_auction()` before any budget-gated decisions.
  Uses MA3-smoothed carbon price and system-average emission factor for revenue estimation.
- **Config**: `budget.mode`, `budget.debt_headrooms` (per-agent), `budget.bot_debt_headrooms`,
  `budget.ema_alpha`.

### Phase B — Emergency Loan System (`company.py`, `market_clearing_ets.py`, `ets_environment.py`)
- **Loan-backed default prevention**: When an agent would default at auction settlement,
  an emergency loan covers the shortfall (up to `max_loan_fraction × annual_budget`).
  Prevents immediate suspension while imposing financial cost.
- **Loan tracking**: `Company._loan_outstanding`, `_loan_repayment_annual`,
  `_years_under_loan`. Interest accrues at `loan_interest_rate` (default 8%).
  Annual repayment deducted at year start via `apply_loan_repayment()`.
- **`settle_auction()` integration**: Accepts `max_loan_budgets` array. Shortfall within
  loan limit triggers `apply_emergency_loan()` instead of default/suspension.
- **Config**: `budget.emergency_loan.enabled`, `budget.emergency_loan.max_loan_fraction`,
  `budget.emergency_loan.interest_rate`.

### Phase C — Pre-Bid Warning and Observation Enrichment (`company.py`)
- **Phase 1 obs extended** from 30D to 33D with three new financial-awareness dims:
  - `[30]` `bid_affordability_last`: last year's bid total / remaining budget (clipped [0,1])
  - `[31]` `loan_outstanding_norm`: emergency loan / annual_budget
  - `[32]` `years_under_loan_norm`: years under active loan / 5
- **Phase 2 obs extended** from +8 to +10 with two compliance-awareness dims:
  - `budget_remaining_phase2_norm`: remaining annual budget after auction / annual_budget
  - `compliance_liability_norm`: (emissions + carry_forward − bank − allocation) / annual_budget

### Phase E — Compliance Reserve Signaling
- Compliance liability signal included in Phase 2 enrichment (see Phase C above).
  Allows secondary market policy to see impending shortfall before compliance settlement.

### Phase D — Reward Channels (`ets_environment.py`)
- **Structured reward logging**: `_last_reward_channels` and `_last_auction_reward_channels`
  dicts populated after each year. Each dict contains named reward components
  (cost_norm, penalty_norm, green_bonus, esg_signal, efficiency_bonus, opp_cost,
  budget_penalty, capex_penalty, loan_interest, base_reward, shaping_reward) for
  debugging and analysis. No change to reward computation.

### Phase F — Heuristic Loan-Awareness (`heuristic_policy.py`)
- **Loan-aware bidding**: When `loan_outstanding_norm > 0.01`, heuristic bots reduce
  auction qty (−30%), investment (−50%), and secondary buy volume (−40%) proportional
  to loan pressure. Prevents bots from over-extending when emergency loans are outstanding.
- **`train.py` integration**: BC warm-start callsites pass `loan_outstanding_norm`.

### Phase H — Config Tuning (`default.yaml`)
- **Budget hardening parameters**: Added `hard_cap_fraction` (1.15), `soft_zone_start` (1.0),
  `tiered_penalty_coef` (2.0), `investment_hard_gate` (true) to budget config section.

### Phase I — Environment Fixes (`company.py`, `ets_environment.py`)
- **Tiered budget penalty**: Replaced 3-tier contingency/quadratic/hard system with
  clean soft-zone quadratic: zero below `soft_zone_start`, quadratic ramp in
  [soft_zone_start, hard_cap_fraction], steep growth above. Penalty scales with
  overshoot amount, not full budget.
- **Investment hard gate**: Pre-investment check in `step_auction()` scales down
  `invest_frac` if total spending would exceed `hard_cap_fraction × annual_budget`.

### Phase J — Tests
- **14 new tests** per codebase covering: reward channel population (4), heuristic
  loan-awareness (3), tiered budget penalty (5), investment hard gate (1),
  reward channels in integration (1). All ported to auction codebase.
- **Current**: 260/260 tests pass.
- **Auction**: 265/265 tests pass.

## v7.6.1

**Calibration/Diagnostics + Phase-Aware Reward Pipeline Patch**

### Market calibration and policy parameters
- Year-0 cap override set to **47.0 Mt** via `cap_year_0_override`.
- Initial bank seeding made explicit in config (`initial_bank_fraction=0.10`) and consumed by environment reset logic.
- TNAC/MSR policy set to target values:
  - `tnac_upper=25.5 Mt`, `tnac_lower=12.0 Mt`
  - intake rate `0.24`
  - MSR release `3.0 Mt`
- `market_calibration.py` now respects explicit mid/lower TNAC ratios when supplied.

### Diagnostics and long-run logging
- Added a one-time Year-1 TNAC diagnostic warning for out-of-range startup states (`[1.0, 8.0] Mt`).
- Warning is emission-guarded to avoid log spam in very long runs.

### Reward and learning pipeline
- `compute_auction_rewards()` now includes loan-interest pressure and projected capex-pressure term.
- `compute_gae()` now normalizes rewards by phase (auction vs secondary) before clipping.
- PPO reward clip fallback raised from `2.0` to `10.0`.
- IPPO `update()` now applies explicit phase masks so auction policy gradients use auction rows and secondary policy gradients use secondary rows.

### Validation
- Updated reward tests to assert phase tagging/masking behavior in GAE.
- Focused updated tests pass for this patch.

## v7.6.0

**Reward Function Overhaul, Batch GAE Normalization, HAPPO Emission Ordering**

### Phase A — Reward Function Changes (`ets_environment.py`, `company.py`)
- **Baseline OPEX**: Added `baseline_opex` snapshot in `Company.__init__()`, computed at
  `current_year=0`. The reward function now uses `opex_delta = current_opex - baseline_opex`
  instead of absolute operational cost. Positive delta = costs rose; negative = OPEX savings
  from greening. This ensures agents aren't penalized for unavoidable base operating costs.
- **Per-agent financial-scale normalization**: All cost, penalty, opportunity cost, terminal
  bank value, and terminal debt penalty divisors changed from fixed `/1000.0` to
  `/company.annual_budget`. This ensures reward magnitudes are proportional to each agent's
  financial capacity, giving small-budget and large-budget agents comparable gradient signals.
- **Per-agent ESG scale**: `esg_scale_i = base_esg_scale × (1000.0 / annual_budget)`
  compensates for the divisor change to preserve the ESG-to-cost ratio that was calibrated
  with the original `/1000` scaling. Maintains the 50/50 balance for ESG agents (w_green=0.5).
- **Terminal values updated**: Bank value, debt liquidation penalty, and opportunity cost all
  use `/annual_budget` instead of `/1000`.

### Phase B — GAE/Normalization Changes (`ppo_agent.py`, `train.py`)
- **Raw rewards in buffer**: `train.py` now stores raw (un-normalized) rewards directly in
  the rollout buffer. `RewardNormalizer.update_and_normalize()` is still called for
  monitoring/logging, but its output is no longer used in the learning path.
- **Batch normalization in `compute_gae()`**: Rewards are standardized per-batch at the top
  of `compute_gae()`: `rewards = (rewards - mu) / max(std, 1e-8)`, then clipped to `[-10, 10]`.
  This replaces per-step EMA normalization, giving the critic a consistent target scale
  and eliminating cold-start bias artifacts.
- **`RewardNormalizer` preserved**: Class and `update_and_normalize()` retained for tracking
  reward scale in logs; no longer on the critical path.

### Phase C — _auc_weight Removal (`ppo_agent.py`)
- **Removed `_auc_weight` heuristic** from both `update()` and `update_happo()`. With
  per-agent budget normalization, the advantage signal is properly scaled without needing
  the observation-derived auction-savings weight. The auction policy loss now uses raw
  advantages identically to the secondary policy.

### Phase D — HAPPO Ordering (`train.py`)
- **Fixed emission-intensity ordering**: Replaced `episode_rng.permutation(n_agents)` with
  `sorted(range(n_agents), key=lambda i: env.companies[i].initial_ef, reverse=True)`.
  Highest emitters are updated first, receiving the cleanest advantages before cumulative
  importance ratio drift from earlier agents' updates. This is deterministic and aligns
  learning priority with where abatement decisions matter most.

### Phase E — Split Rewards / Two-Phase (`ets_environment.py`, `train.py`)
- **`compute_auction_rewards()` method**: New method computes per-agent intermediate reward
  after `step_auction()` completes: `r_auction = -(auction_cost + collateral + investment
  + opex_delta + mac_cost) / annual_budget`. Returns before secondary market execution.
- **Two transitions per year-step**: `train.py` now stores two buffer entries per year:
  (1) auction-phase transition with `r_auction`, `done=False`; (2) secondary-phase transition
  with `r_secondary = total_reward - r_auction`, `done=terminated`. This doubles T from
  `n_years` to `2 × n_years` per episode, providing proper credit assignment to each phase.
- **`expected_T` updated**: HAPPO ratio chain now expects `2 × n_years × episodes_per_update`.

### Phase F — Tests
- **New tests**: `test_opex_delta_zero_for_unchanged_mix`, `test_esg_cost_balance_preserved`,
  `test_batch_normalization_replaces_ema`, `test_split_rewards_sum_to_total`.
- **Updated**: `test_terminal_bank_uses_1000_divisor` → `test_terminal_bank_uses_budget_divisor`
  (references `/annual_budget` instead of `/1000`).
- **All 245 tests pass**.

### Phase G — Phase-Split Policy Gradient Backport (`ppo_agent.py`, `train.py`)

*Backport of the correctness fix from `ets_marl_happo_auction` Phase G.*

#### G1 — RolloutBuffer phase tagging (`ppo_agent.py`)
- **`phases` list**: `RolloutBuffer.clear()` now initialises `self.phases = []`. The `push()`
  method accepts a `phase='secondary'` keyword argument that tags each stored transition as
  either `'auction'` or `'secondary'`.

#### G2 — `store_transition()` phase param (`ppo_agent.py`)
- `store_transition()` now accepts and forwards `phase='secondary'` to `buffer.push()`.

#### G3 — Phase-split actor losses in `compute_gae()` + `update_happo()` (`ppo_agent.py`)
- **`compute_gae()`**: Builds `is_auction_t: BoolTensor[T]` from `buffer.phases`, exposed as
  `buf_tensors["is_auction"]` for reuse in `update_happo()` and `compute_post_update_ratio()`.
- **`update_happo()`**: Actor losses are now split per mini-batch:
  - `auc_policy_loss` — computed only on `is_auc_mb` rows (obs1-space).
  - `sec_policy_loss` — computed only on `~is_auc_mb` rows (obs2-space).
  - BC-KL penalty also applies phase masking to avoid evaluating policies on the wrong
    observation space.

#### G4 — Phase-masked `compute_post_update_ratio()` (`ppo_agent.py`)
- Replaced joint `(new_auc_lp − old_auc_lp) + (new_sec_lp − old_sec_lp)` formula with
  per-row phase assignment: auction rows receive the auction log-ratio (obs1-space); secondary
  rows receive the secondary log-ratio (obs2-space). Prevents cross-obs contamination in the
  HAPPO cumulative M-factor chain.

#### G5 — `train.py` phase tagging
- Auction-phase `store_transition()` call now passes `phase='auction'`.
- Secondary-phase `store_transition()` call now passes `phase='secondary'`.

### Config / Metadata
- `pyproject.toml`: version 7.6.0
- `default.yaml` header: v7.6
- `train.py` banner updated
- `README.md`: v7.6 improvements documented

---

## v7.5.0

**MSR Three-Band Withholding, Rollover Accounting Fix, Unbuffered Need, Heuristic Cleanup**

*Backport of v8.2 changes from `ets_marl_happo_auction`. All phases apply equally to both variants.*

### Phase A — MSR Three-Band Withholding (`cap_schedule.py`, `market_calibration.py`)
- **`_compute_tnac_withholding()` helper**: Extracted TNAC withholding into a dedicated
  method with three distinct regimes matching Decision (EU) 2015/1814 and its 2023 amendment:
  - `TNAC > upper`: withhold `24% × TNAC` (% of *total* TNAC, corrects prior excess-only formula).
  - `mid ≤ TNAC ≤ upper`: withhold `TNAC − mid` (tapered intake above mid-threshold).
  - `TNAC < mid`: no MSR intake.
- **Legislative TNAC proportions**: `TNAC_LOWER_REF=400`, `TNAC_MID_REF=833`,
  `TNAC_UPPER_REF=1096` scaled to simulation cap; preserves 400:833:1096 proportions.
- **`tnac_mid` propagated** through return dict, environment init/reset, and attribute
  assignment branches.
- **Config**: `tnac_lower_ratio` corrected 0.22 → 0.1314; `tnac_mid_ratio: 0.2737` added.

### Phase B — Rollover Accounting Fix (`ets_environment.py`, `cap_schedule.py`)
- **CapSchedule telemetry attrs**: `_last_unsold_rollover_in`, `_last_msr_withheld`,
  `_last_msr_released` set inside `get_auction_volume()`.
- **`msr_withhold_this_year` / `msr_release_this_year`** year-log entries now accurate.
- **Double-count fix**: `unsold = auction_volume − allocations.sum() − defaulted_volume`.
- **Pre-obs estimate** includes both pending rollover channels with cap.

### Phase C — Unbuffered Estimate Need (`company.py`)
- **`compute_estimate_need()` simplified**: Returns bare `compute_emissions()`, no risk
  buffer. Agents learn their own coverage buffer through bidding.
- `_compute_p_fail()` comment clarified as investment execution risk only.

### Phase D — Heuristic Simplification (`heuristic_policy.py`)
- **Green-agent seller discount removed**: `is_green` variable and 50% sell-rate reduction
  eliminated from `secondary_action()`.

### Phase E — Tests and Minor Fixes
- **New tests**: Cover three-band TNAC regimes and rollover interaction.
- **Updated tests**: Withholding formula, `tnac_mid` fixture, `tnac_lower` value.
- **`test_company`**: Added check that `compute_estimate_need()` returns bare emissions.
- **`test_environment`** / **`test_market_calibration`**: Updated for `tnac_mid` and
  rollover accounting.
- **`qty_mult_high` default**: Corrected 1.3 → 2.0.

### Config / Metadata
- Version bumped to `7.5.0` in `pyproject.toml`, `configs/default.yaml`.

---

## v7.4.0

**Plan v8.1 Backport: LRF/MSR Realism, Heuristic Rewrite, Collateral Enforcement, Reward Shaping**

*Note: Phases B, D, G apply only to `ets_marl_happo_auction`. This version receives
Phases A, C2/C3, E2/E4, and F.*

### Phase A — TNAC/MSR Realism
- **Linear LRF**: Fixed exponential-decay bug. Cap declines by equal absolute steps:
  `cap_t = cap_0 − Σ lrf_k × cap_0`. `lrf_phase_switch` set to year 2.
- **1-year TNAC lag** (`_prev_tnac`): MSR uses prior-year TNAC, matching EU ETS Decision
  2015/1814. Year 0 has no MSR unless `force_msr=True`.
- **MSR thresholds**: Lower threshold 18% → 22%; `release_frac` 0.016 → 0.064.
- **Smoothed price trigger (A4)**: Emergency release requires absolute threshold breach
  *and* MA3 spike > 2.5× prior year's MA3.

### Phase C — Heuristic Rewrite
- **C2/C3 smarter secondary**: Final-year urgency boost (×3), no selling when in
  compliance debt, green agents sell at half rate, budget headroom cap on buying.

### Phase E — Collateral Enforcement
- **E2 revised collateral**: `collateral_fraction` corrected to 0.10 (10%), previously
  relying on old `opportunity_cost_rate × hold_fraction = 0.001`. Updated
  `max_collateral_budget_share` to 0.50.
- **E4 leverage/suspension config**: Added `leverage_multiplier: 3.0`,
  `suspension_length: 2`, `carry_forward_defaults: true` to auction config section.

### Phase F — Reward Interpretability
- **Efficiency bonus as shaping**: `efficiency_bonus` decays with `shaping_weight`.
  Renamed from `efficiency_shaping` for consistency.
- **`compute_diagnostic_score()`**: New method logging S_financial, S_green, S_composite
  per agent. Year-level CSV now includes `diag_S_*_Ai` columns.
- **Console output**: Episode-mean diagnostic scores printed:
  `Diag(Sfin/Sgrn/Scomp): A1: 0.72/0.15/0.52 │ A2: ...`

### Config / Metadata
- Version bumped to `7.4.0` in `pyproject.toml`, `configs/default.yaml`.

---

## v7.3.0

**Reward Simplification: Remove Revenue, Prune Queue Bonus**

### Reward Changes
- **Electricity revenue removed from cost normalisation**: `cost_norm_ex_penalty` is now
  `total_cost_ex_penalty / 1000.0` instead of `(total_cost_ex_penalty - revenue) / 1000.0`.
  Revenue from electricity sales no longer offsets compliance costs in the reward signal,
  giving agents a cleaner cost-minimisation gradient.
- **Queue bonus removed from shaping rewards**: The `queue_bonus` term
  (`gamma_shaping × n_active_queue × 0.1 × shaping_weight`) has been deleted.
  Shaping rewards now consist solely of `green_bonus` (diminishing-returns bonus for
  green investment progress), reducing reward complexity and removing a signal that
  could incentivise queue-stuffing rather than genuine decarbonisation.

### Config / Metadata
- Version bumped to `7.3.0` in `pyproject.toml`, `configs/default.yaml`, `README.md`,
  and `train.py` banner.

---

## v7.2.1

**Bug Fixes, Security Hardening, and Code Quality**

### Critical Fixes
- **EMA variance bias**: `RewardNormalizer.update_and_normalize()` now saves `old_mu` before
  updating the mean, then uses `old_mu` in the variance update. Eliminates systematic
  variance underestimation that biased early-training reward normalization.
- **Return normalization removed**: Set `normalize_returns: false` in config. The per-agent
  `RewardNormalizer` already stabilizes reward scale; double-normalizing returns distorted
  the critic's value targets.
- **Phase-specific credit assignment**: Auction policy loss is now weighted by a phase-1-only
  advantage proxy derived from `obs2[base+4]` (auction_savings). This provides the auction
  policy with a gradient signal specific to auction performance rather than the blended
  year-level advantage.
- **Efficiency bonus amplified**: Coefficient increased from 0.3 to 1.5, price_weight
  changed from `clearing_price / 1000` to `clearing_price / 100`. Financial agents now
  receive a meaningful gradient for emission-factor improvement.

### Significant Fixes
- **HAPPO cumulative ratio**: Replaced `cumulative_ratio_by_T` dict with a single tensor
  sized to the expected trajectory length. Agents with mismatched T (e.g. HPP-cleared
  buffers) are excluded from the M-factor chain entirely, preventing ratio dimension
  mismatches and preserving sequential dependency.
- **Opportunity cost**: Added explicit documentation that `self.holdings[i]` is already
  the post-compliance bank when `_compute_rewards` is called.
- **No-short-selling**: `max_sell` in secondary market now subtracts expected compliance
  need (`realized_emissions + carry_forward`), preventing agents from selling allowances
  they need for compliance.

### Moderate Fixes
- **Burn-in double apply_matured_investments**: Removed the redundant call at the top of
  the burn-in loop. Only the bottom-of-loop call (year + 1) is retained.
- **Price history fallback**: `_compute_price_ma3()` now returns `self.expected_price`
  (AR(1) forecast, ~80€) when price history is empty, instead of falling back to
  `last_clearing_price` which may be the reserve price after a failed auction.
- **weights_only=True**: `torch.load()` now uses `weights_only=True` for security.
- **Action space dead zone fixed**: `invest_frac` now uses continuous linear mapping
  `((action + 1) / 2) * max_invest_frac` instead of `np.clip`, eliminating the dead
  zone where negative policy outputs all mapped to zero investment.
- **Terminal debt liquidation**: Final-year carry-forward debt is now aggressively
  penalized: `rewards -= carry_forward * terminal_price * 1.5 / 1000`.

### Security
- Replaced `pickle.load` for Q-tables with `np.load`/`np.savez` in `train_qlearning.py`,
  `evaluate_qlearning.py`, and `qlearning_analysis.py`.
- CSV file handles in `train.py` registered with `atexit` for cleanup on crash.

### Reproducibility
- `np.random.seed(seed)` and `torch.manual_seed(seed)` set at start of `train_one_seed()`.
- `np.random.shuffle` in PPO agent replaced with per-agent seeded `numpy.random.Generator`.

### Environment / Portability
- `requirements.txt` aligned with `pyproject.toml` exact pins.
- `install.bat` updated to use `==` pins matching `pyproject.toml`.

### Code Quality
- Removed dead legacy `logger.py` and `replay_buffer.py`.
- Removed stub `main.py` (real entry point is `scripts/train.py`).
- Added `OBS1_EXPECTED_PRICE_IDX` named constant replacing magic `obs1[3]`.
- Added obs-dim assertion in `evaluate.py`; config saved alongside checkpoints.
- Added `.github/workflows/test.yml` for CI (runs pytest on push/PR).

### Documentation
- Documented the "one reward per year-step for two decision phases" design choice
  (see phase-specific credit assignment above).

## v7.2.0

**Budget-Aware Collateral Affordability + Observation Headroom**

- `configs/default.yaml`:
	- Version header bumped to `v7.2`.
	- Added `auction.collateral.min_qty_floor_frac: 0.5`.
	- Updated collateral rationale comments for `hold_fraction: 0.02` (~7 days annual-step proxy balancing T+2 realism and yearly simulation granularity).
- `ETSEnvironment.step_auction` now applies a pre-auction two-step collateral affordability clip after quantity expansion:
	- Step 1: clip quantity at current bid price if affordability still preserves at least `min_qty_floor_frac × emissions_need`.
	- Step 2: if quantity clipping would starve coverage, reduce bid price instead to keep quantity.
- Added an explicit note above reward collateral-cost usage clarifying the `hold_fraction=0.02` interpretation.
- `Company.get_observation_phase1` gained optional inputs `budget_spent` and `annual_budget` and now computes:
	- `budget_headroom = clip(1 - budget_spent / annual_budget, -0.5, 1.0)`.
	- Observation dim `[27]` now carries `budget_headroom` (replacing auction-volume-change signal).
- `ETSEnvironment._get_obs_phase1()` now passes each agent's `budget_spent_this_year` and `annual_budget` into `get_observation_phase1(...)`.

**Tabula-Rasa 80 EUR Anchor Integration**

- Includes the tabula-rasa expected-price fallback updates around **80 EUR/t** from the current branch work (agent fallback + related tests), packaged with this release as `v7.2.0`.

## v7.1.0

**EU ETS Bid Collateral Cost + Tabula-Rasa Price-Side Balancing**

- Added `auction.collateral` config block in `configs/default.yaml`:
	- `enabled`
	- `opportunity_cost_rate`
	- `hold_fraction`
- Corrected `auction.collateral.hold_fraction` from `0.08` to `0.02` (weekly-cycle annualized proxy), with updated cost examples.
- `ETSEnvironment.step_secondary` now computes per-agent collateral opportunity cost using
	`rate * hold_fraction * max(0, bid_price - clearing_price) * allocation`.
- Collateral costs are passed into `_compute_rewards(...)` and included in
	`total_cost_ex_penalty` as a real financial cost channel (not penalty shaping).
- Year-level environment logs now include `collateral_costs`.
- Year-level CSV outputs now include per-agent collateral columns:
	- PPO/HAPPO training: `collateral_cost_A*` in `year_log_s*.csv`
	- Q-learning training: `collateral_cost_A*` in `ql_year_log_s*.csv`
- Tabula-rasa `exploration.mode="uniform"` auction-price sampling is now side-balanced:
	- 50/50 underbid vs overbid around expected price,
	- uniform sampling within each side range,
	- prevents structural overbid bias from asymmetric price bounds.
- Updated tabula-rasa exploration tests to validate over/under side balance.
- Version metadata updated to v7.1 (`configs/default.yaml`, `pyproject.toml`, banners/docs).

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

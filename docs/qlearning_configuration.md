# Q-Learning Configuration and Set-Up

Companion document to `design.md`. This file assumes the reader already
understands the ETS MARL environment — auction, secondary market,
investment/MAC loop, cap/MSR, ESG reward, observation and action
spaces. It only describes what the Q-learning trainer does
differently from the HAPPO/PPO trainer. Everything not mentioned here
is identical to the default PPO run (same env, same warm-start, same
year cadence, same reward, same logs).

Entry point: `src/train_qlearning.py`
Config: `configs/qlearning.yaml` (override on top of `configs/default.yaml`)
Eval: `scripts/evaluate_qlearning.py`

---

## 1. What is replaced relative to HAPPO

| Aspect | HAPPO | Q-learning |
|---|---|---|
| Policy | Per-agent PPO actor with HAPPO-style sequential updates | Independent tabular Q-table per agent |
| State | 44-dim continuous Phase-1 obs (+ Phase-2 obs) | 5 features × 3 bins = **243 discrete states** |
| Phase-1 action | 6-D continuous (bid price, qty mult, invest frac, 3 tech logits) | **6 discrete profiles** |
| Phase-2 action | 2-D continuous (sec price, sec qty) | **4 discrete profiles** |
| Joint action space | Continuous | **6 × 4 = 24** profile pairs per state |
| Exploration | Stochastic policy + entropy bonus | **ε-greedy**, linear decay |
| Update | Clipped surrogate (PPO) + GAE | **One-step tabular Q-update** |
| Bots | Same env heuristic bots, untouched | Same env heuristic bots, untouched |

---

## 2. State Discretisation

Implemented in `StateDiscretizer` (`src/agents/q_learning_agent.py`).

Five features extracted from the **Phase-1 observation vector** plus
the agent's `Company` object, each binned into `{0, 1, 2}` and packed
into a single integer in `[0, 242]` via mixed-radix encoding.

| # | Feature | Source | Bin edges | Meaning |
|---|---|---|---|---|
| 0 | `time_progress`     | `obs[0]`             | `[0.33, 0.67]` | early / mid / late episode |
| 1 | `carbon_price_norm` | `obs[2]` (`price_ma3 / price_max`) | `[0.30, 0.55]` (≈ 75 € / 137 € at `price_max=250`) | low / medium / high carbon price |
| 2 | `green_fraction`    | `company.green_frac` | `[0.35, 0.65]` | dirty / mixed / green mix |
| 3 | `compliance_gap`    | `obs[13]` (`auction_gap/5`) | `[0.10, 0.60]` | deficit / balanced / surplus |
| 4 | `carry_forward`     | `obs[20]` (`cf/5`)   | `[0.01, 0.20]` | none / some / heavy carry-forward |

Binning is right-closed: `value ≤ edges[0] → 0`, `≤ edges[1] → 1`,
else `→ 2`. Observation dimensions outside this list (cash, MSR
signals, opponent encodings, BCL / gate clip flags, compliance
affordability, etc.) are not part of the state.

Total state space size: `3^5 = 243` states per agent.

---

## 3. Action Profiles

Implemented in `ActionProfileMapper`. The mapper converts a
`(profile_idx, company, price_ma3, config, year)` tuple into the same
continuous action vector the env consumes from a HAPPO actor, so the
env code path is identical.

### 3.1 Phase-1 auction profiles (6)

The bid price is built from an **anchor**
`anchor = clip(price_ma3, price_min, penalty_rate)` where
`penalty_rate` is the inflation-adjusted rate via
`company.effective_penalty_rate(year)`.

| idx | Name | Bid (× anchor, except #5) | qty mult | invest frac | Tech (logits) |
|---|---|---|---|---|---|
| 0 | Conservative   | 0.90 | 1.0 | 0.01            | solar    |
| 1 | Moderate       | 1.10 | 1.0 | 0.03            | solar    |
| 2 | Aggressive buy | 1.30 | 1.5 | 0.06            | onshore  |
| 3 | Green push     | 1.20 | 1.0 | `max_invest_frac` | onshore |
| 4 | Financial      | 0.80 | 0.7 | 0.0             | solar    |
| 5 | Panic buy      | `1.20 × penalty_rate` | `qty_mult_high` | 0.03 | solar |

All values are clipped against the same env bounds HAPPO sees
(`price_min/max`, `qty_mult_low/high`, `max_invest_frac`).
Tech logits are one-hot-ish (`+1, −1, −1`) so the env's argmax picks
the intended technology.

### 3.2 Phase-2 secondary profiles (4)

Anchor: `max(clearing_price, sec_price_min)`.

| idx | Name | Price (× anchor) | qty (Mt) |
|---|---|---|---|
| 0 | Hold           | anchor (irrelevant; qty=0) | 0 |
| 1 | Sell surplus   | 1.10 | −min(1.0, `quantity_max`) |
| 2 | Buy shortfall  | 1.20 | +min(1.0, `quantity_max`) |
| 3 | Aggressive buy | 1.50 | +min(2.0, `quantity_max`) |

The env clips the resulting price into
`[sec_price_min, sec_price_max]` exactly as for HAPPO.

### 3.3 Joint action

The Q-table is indexed as **`Q(s, a1, a2)`** with shape
`(243, 6, 4)`. Auction and secondary profiles are selected
sequentially within the same year (a1 first, then a2), but a single
Bellman update is applied per (year, agent) on the joint cell — see §5.

---

## 4. ε-greedy Selection

Linear decay over a fraction of training:

```
ε(ep) = ε_start + (ε_end − ε_start) · (ep / ε_decay_episodes)   for ep < ε_decay_episodes
ε(ep) = ε_end                                                    otherwise
ε_decay_episodes = max(1, int(n_episodes · ε_decay_frac))
```

Defaults (`configs/qlearning.yaml`): `ε_start=1.0`, `ε_end=0.05`,
`ε_decay_frac=0.7`. With `n_episodes=10000` this is 7000 episodes of
linear decay followed by 3000 episodes at the floor.

Per agent, per phase:
- **Auction (a1):** with prob ε, sample uniformly from 6 profiles;
  else `argmax_{a1} max_{a2} Q(s, a1, a2)`.
- **Secondary (a2):** uses the already-picked `a1` to condition the
  greedy lookup: `argmax_{a2} Q(s, a1, a2)`.

ε-random vs greedy picks are counted per `log_interval` window for
the window-diagnostic counters (see §6).

---

## 5. Q-Update

Standard one-step tabular Q-learning, applied once per (year, agent)
right after `env.step_secondary` returns:

```
target = reward                                       if terminated
       = reward + γ · max_{a1',a2'} Q(s', a1', a2')   otherwise
Q(s, a1, a2) ← Q(s, a1, a2) + α · (target − Q(s, a1, a2))
```

- `s` is the discretised Phase-1 obs the agent saw before bidding.
- `s'` is the discretised next-year Phase-1 obs returned by
  `step_secondary`.
- `reward` is the env's per-agent reward for that year — the same
  scalar HAPPO sees (financial + ESG, with all gates / penalties /
  MAC costs already baked in).
- `terminated` is the env's terminal flag (end of year 12).

Defaults: `α = 0.1`, `γ = 0.95`. The bootstrap target uses the unconditional
max over the joint `(a1', a2')` cell, not the on-policy a1' choice.

Each update also bumps `visit_counts[s, a1, a2]` and accumulates the
absolute TD-error for window diagnostics.

---

## 6. Training Loop

`src/train_qlearning.py` mirrors `scripts/train.py` on everything
env-facing:

- Per-seed env reset uses `seed + episode * 1000` (same recipe as PPO).
- Same Phase-1 → Phase-2 cadence per simulated year, same
  `set_episode(ep)` warm-start hook.
- Bots are unchanged: the env's internal heuristic policy drives
  them; only the first `n_agents` agents have Q-tables.
- The trainer holds one `QLearningAgent` instance per learning agent,
  each with its own Q-table and its own
  `np.random.default_rng(seed + i)`. Agents are independent: no
  centralised critic, no parameter sharing, no joint exploration
  schedule beyond the shared ε.

There is no rollout buffer, no minibatching, no gradient clipping, no
GAE, no value head, no entropy bonus. Updates are applied online,
in-place.

`--run-tag <name>` is supported for sweep launchers and is used as a
filename infix (`ql_training_log_<tag>_s<seed>.csv`,
`qtables_<tag>_s<seed>_final.npz`).

### 6.1 Hyperparameters (`configs/qlearning.yaml`)

| Block | Key | Default | Notes |
|---|---|---|---|
| `simulation` | `n_episodes` | `10000` | Episodes per seed for the QL loop. |
| `qlearning` | `n_episodes`       | `10000` | Mirrors `simulation.n_episodes` for the QL loop. |
| `qlearning` | `alpha`            | `0.1`   | Learning rate. |
| `qlearning` | `gamma`            | `0.95`  | Discount factor. |
| `qlearning` | `epsilon_start`    | `1.0`   | Fully random at episode 0. |
| `qlearning` | `epsilon_end`      | `0.05`  | Floor for the final 30 % of training. |
| `qlearning` | `epsilon_decay_frac` | `0.7` | Fraction of `n_episodes` over which ε decays linearly. |
| `evaluation` | `n_episodes`      | `100`   | Greedy episodes after each seed. |
| `evaluation` | `epsilon`         | `0.0`   | Fully greedy in eval. |
| `logging` | `log_interval`        | `100`   | Console + window-diagnostic cadence. |
| `logging` | `save_interval`       | `500`   | Q-table checkpoint cadence. |
| `logging` | `csv_flush_interval`  | `500`   | CSV flush cadence. |
| `logging` | `results_dir`         | `results/qlearning/` | Output directory. |

Everything else (env, companies, auction, MSR, penalty, ESG,
investment, warm-start, …) is inherited unchanged from
`configs/default.yaml` via a deep merge in `merge_configs()`.

---

## 7. Logging

The QL trainer reuses the PPO CSV schemas so the analysis notebooks
load it the same way:

- `ql_training_log[_<tag>]_s<seed>.csv` — episode-level rollup.
  Includes the same anchor-invariant `quality_score` and the five
  `Q_compliance / Q_price_realism / Q_saved_carbon / Q_cost_eff /
  Q_volatility` components computed via
  `src.utils.quality_metric.compute_episode_quality`. Per-agent
  fields are the env-derived ones (reward, green_frac, compliance,
  shortfall, penalty, bid_price, …); learner-internal HAPPO fields
  (`actor_loss_*`, `entropy_coef`, …) are omitted. Two extra
  columns: `a1_profile_A{i}`, `a2_profile_A{i}` — the last-year
  profile each agent picked (≥ 0 for learners, `-1` for bots).
- `ql_year_log[_<tag>]_s<seed>.csv` — per-year flattening of
  `env.year_log`. Schema is a superset of `year_log_*.csv`, fully
  comparable.
- `qtables[_<tag>]_s<seed>_ep<N>.npz` — periodic Q-table checkpoints
  (one array per agent, key `agent_<i>`), saved every
  `save_interval` episodes. Final tables go to
  `qtables[_<tag>]_s<seed>_final.npz`.

Per-window diagnostic counters maintained on each `QLearningAgent`
(reset at the end of every `log_interval` window via
`reset_window_stats()`): mean `|TD|`, Q-table state coverage, mean
`max_{a1,a2} Q(s,·,·)` across visited states, share of ε-random
picks, and the distinct greedy `(a1, a2)` pairs across visited
states.

---

## 8. Evaluation

`scripts/evaluate_qlearning.py` loads a `qtables_…_final.npz`,
restores one `QLearningAgent` per learning slot, and rolls out
`evaluation.n_episodes` episodes with `epsilon=0.0` (fully greedy)
against the same env config. It writes the same CSV schemas with an
`eval_` prefix in the output directory.

# ETS MARL — Sweep Output Data Dictionary

This document describes every artefact produced by a training run or
sweep, so that downstream notebooks (and external readers of the
thesis) can interpret each file without having to read `train.py`.

A single training invocation (`scripts/train.py --config <cfg> --seed S`)
writes its outputs into the directory given by `logging.results_dir`
(default `results/`). A *sweep* (`scripts/sweep.py --spec <sweep.yaml>`)
runs many `(variant, seed)` jobs in parallel; each variant gets its own
sub-directory under `<spec.output_dir>/<variant_name>/`, and filenames
are disambiguated by an injected `--run-tag <variant_name>`.

---

## 1. Directory layout

For a sweep variant `tight_cap` with seeds `[1, 2, 3]`:

```
<output_dir>/tight_cap/
├── training_log_tight_cap_s1.csv      # episode-level log (one row per episode)
├── training_log_tight_cap_s2.csv
├── training_log_tight_cap_s3.csv
├── year_log_tight_cap_s1.csv          # year-level log (one row per (episode, year))
├── year_log_tight_cap_s2.csv
├── year_log_tight_cap_s3.csv
├── checkpoints_tight_cap_s1/
│   ├── agent_0_ep0.pt … agent_7_ep0.pt        # cold-start baseline
│   ├── agent_0_ep<save_interval>.pt …          # periodic snapshots
│   └── agent_0_best.pt … agent_7_best.pt       # best-reward snapshot
├── checkpoints_tight_cap_s2/
├── checkpoints_tight_cap_s3/
├── snapshots/                                  # rolling mid-run CSV copies
│   ├── training_log_tight_cap_s1_ep<N>.csv
│   └── year_log_tight_cap_s1_ep<N>.csv
├── run_tight_cap_s1.log                # captured stdout/stderr from train.py
├── run_tight_cap_s2.log
└── run_tight_cap_s3.log
```

A single-config run (no `--run-tag`) drops the `_<tag>` infix and writes
`training_log_s<seed>.csv`, `year_log_s<seed>.csv`, `checkpoints_s<seed>/`.

---

## 2. `training_log_*.csv` — episode-level log

One row per episode. Holds aggregate metrics, per-agent end-of-episode
diagnostics, and warning counts. Roughly **(8 + 26·N + 6·n_agents + 30)**
columns for `N = n_total_agents`. With the default 8-learning-agent
config, that is ≈ 270 columns.

Field groups (every `_A{i}` suffix runs `i = 1 … n_total_agents`,
i.e. learning agents first, then bots):

### Header / scheduling

| Column | Meaning |
|---|---|
| `episode` | Zero-based episode index. |
| `clearing_price_last` | Auction clearing price in the **final year** of the episode (€/tCO₂). |
| `cap_last` | Emission cap in the final year (Mt). |
| `entropy_coef` | Current actor entropy coefficient (decayed over training). |
| `shaping_weight` | Current global shaping weight applied to opportunity-cost / coverage-gap shaping (decays from 1 → `shaping_weight_floor`). |
| `entropy_decay_triggered` | `1` once the entropy decay schedule has begun, else `0`. |
| `active_agent` | HAPPO sequential-update index for this episode (which agent's policy is being updated). |
| `epsilon` | Current ε-greedy exploration rate. |

### Per-agent episode aggregates (all `_A{i}`)

| Column | Meaning |
|---|---|
| `reward_A{i}` | Total per-episode reward (Σ over years, post-shaping). |
| `reward_base_A{i}` | Total per-episode **base** reward (financial + ESG − penalty + banking; pre-shaping, pre-terminal). |
| `reward_shaping_A{i}` | Total shaping component (`reward − reward_base`). |
| `green_frac_A{i}` | Final-year green generation share. |
| `delta_green_A{i}` | Final-year minus year-0 green share. |
| `penalty_A{i}` | Total non-compliance penalties paid this episode (M€, nominal). |
| `shortfall_A{i}` | Total realised compliance shortfall this episode (Mt). |
| `queue_size_A{i}` | Construction queue length at end of episode (# pipeline projects). |
| `actor_loss_A{i}`, `critic_loss_A{i}` | PPO actor / critic losses from the most recent update for this agent. |
| `bid_price_A{i}` | Mean Phase-1 bid price across the episode (€/tCO₂). |

### Secondary-market & investment per agent (`_A{i}`)

| Column | Meaning |
|---|---|
| `sec_buy_vol_A{i}` / `sec_sell_vol_A{i}` | Mt bought / sold on secondary market across the episode. |
| `sec_buy_avg_px_A{i}` / `sec_sell_avg_px_A{i}` | Volume-weighted average secondary price for buys / sells (€/tCO₂). |
| `sec_buy_years_A{i}` / `sec_sell_years_A{i}` | # years (out of 12) in which the agent was a net buyer / seller. |
| `avg_sec_mult_A{i}` | Mean Phase-2 price action[0] (absolute €/tCO₂). |
| `avg_sec_qty_A{i}` | Mean Phase-2 qty action[1] (Mt; signed: + buy, − sell). |
| `avg_bid_mult_A{i}` | Mean Phase-1 quantity multiplier on `estimate_need`. |
| `avg_bid_coverage_A{i}` | Mean (alloc / need) coverage ratio after auction. |
| `sec_buy_intent_share_A{i}` / `sec_sell_intent_share_A{i}` | Share of years the agent submitted buy- / sell-side intent on the secondary market. |
| `inv_onshore_share_A{i}`, `inv_offshore_share_A{i}`, `inv_solar_share_A{i}` | Episode shares of total invested capacity by tech. |
| `udbc_U_total_A{i}` | # years agent was non-compliant **purely** from auction underbid (no inherited carry-forward). |
| `udbc_D_total_A{i}` | # years non-compliant **purely** from inherited debt cascade (alloc ≥ emiss but cf > 0). |
| `udbc_M_total_A{i}` | # years non-compliant from **both** causes simultaneously. |
| `udbc_B_total_A{i}` | # years auction-short but compliant via own bank drawdown. |
| `udbc_C_total_A{i}` | # years auction-short but compliant via secondary buy. |

### Episode-mean diagnostics

| Column | Meaning |
|---|---|
| `secondary_volume`, `secondary_avg_price`, `secondary_match_rate` | Episode-totals across the secondary market. |
| `ep_mean_clearing_price` | Mean uniform-price auction clearing across the 12 years. |
| `ep_mean_coal_coverage_ratio` | Mean (alloc / need) coverage averaged over coal-heavy archetype agents. |
| `ep_default_count` | # (agent, year) defaults this episode. |
| `ep_mean_bid_qty_mult` | Mean Phase-1 qty multiplier across all agent-years. |
| `ep_mean_coal_budget_headroom` | Mean budget headroom for coal-heavy agents (M€). |
| `quality_score` | Aggregate anchor-invariant quality metric (notebook §5.8). Higher = better. |
| `Q_compliance`, `Q_price_realism`, `Q_saved_carbon`, `Q_cost_eff`, `Q_volatility` | Five subscores in [0, 1] composing `quality_score`. |
| `price_start`, `price_peak`, `price_std` | Year-0 clearing, episode-max clearing, std of clearings across years. |
| `secondary_price` | Episode-mean secondary clearing price (alias for the year-log column). |

### Per-agent allocation, shocks & MAC

| Column | Meaning |
|---|---|
| `ep_start_bank_A{i}` | Bank holdings at start of year 0 (post-warmstart). Mt. |
| `mean_alloc_A{i}` | Mean auction allocation across years (Mt). |
| `mean_shock_A{i}`, `max_shock_A{i}` | Mean / max realised emission shock (multiplicative factor on baseline). |
| `mean_cf_shock_A{i}` | Mean realised capacity-factor noise shock. |
| `total_cancels_A{i}` | # construction projects cancelled this episode. |
| `total_mac_reduction_A{i}` | Total emissions abated via MAC fuel-switching (Mt). |
| `invest_cost_A{i}` | Episode-summed investment cost (M€, nominal). |

### Diagnostic warnings (counts of years triggering each detector)

`warn_lowAlloc, warn_priceFloor, warn_priceCeil, warn_auctFail,
warn_lowDemand, warn_noInvest, warn_debtSpiral, warn_bidCluster,
warn_overBank, warn_1sideSec, warn_noTrade, warn_cornering,
warn_rsvReject, warn_agents_stuck_ceiling, warn_agents_stuck_floor,
warn_agents_stuck_zeroQty`. Cumulative (multi-year) counters for the
single episode.

### Per-learning-agent stuck-streaks & diagnostic scores

| Column | Meaning |
|---|---|
| `streak_ceil_A{i}` | Consecutive-episode count of mean bid ≥ 99 % of `auction.price_max`. |
| `streak_floor_A{i}` | Consecutive-episode count of mean bid ≤ 102 % of `auction.price_min`. |
| `streak_zeroqty_A{i}` | Consecutive-episode count of avg `qty_mult ≤ zero_qty_threshold`. |
| `diag_S_financial_A{i}`, `diag_S_green_A{i}`, `diag_S_composite_A{i}` | Episode-mean of three normalised diagnostic scores in [0, 1]. |
| `esg_vs_penalty_ratio_A{i}` | `(w_green · esg_signal) / penalty_norm` averaged across years. |
| `compliance_gate_A{i}` | Episode-mean coverage-based ESG gate (`coverage_frac^(1+blend)`). |

### Phantom bidder (only meaningful if `phantom_bidder.enabled=true`)

`phantom_active_pct, phantom_avg_bid_price, phantom_avg_bid_qty`.

### Diagnostics: advantages & split-head losses

| Column | Meaning |
|---|---|
| `adv_yr1_mean_A1` … `adv_yr1_mean_A{n_agents}` | Year-1 mean (pre-normalisation) advantage per learning agent — diagnostic for HAPPO ordering. |
| `actor_loss_invest_A{i}`, `critic_loss_invest_A{i}` | Split-head loss for the **investment** sub-head of the auction policy. |
| `actor_loss_secondary_A{i}`, `critic_loss_secondary_A{i}` | Loss for the secondary-market policy. |

### Episode-level credit / debt-cascade aggregates

Reductions over the per-year series in `year_log_*.csv`, pre-aggregated
so notebooks don't have to roll them up:

| Column | Meaning |
|---|---|
| `year0_tnac` | TNAC at end of the first year (Mt). |
| `yearT_tnac` | TNAC at end of the final year (Mt). |
| `ep_total_unsold` | Sum of `auction_unsold` across the episode (Mt). |
| `ep_auction_failures` | Number of years for which the primary auction failed (`auction_failed=1`). |
| `ep_total_defaults` | Sum of post-clearing settlement defaults (sum of `auction_defaults` across years). |
| `peak_loan_outstanding_A{i}` | Maximum emergency-loan balance held by agent *i* during the episode (M€). |
| `peak_carry_forward_A{i}` | Maximum end-of-year carry-forward debt for agent *i* during the episode (Mt). |
| `final_treasury_reserve_A{i}` | Treasury balance held by agent *i* at the end of the final year (M€). |

---

## 3. `year_log_*.csv` — year-level log

One row per `(episode, year)` pair. With `n_episodes = 120 000` and
`n_years = 12` this is ~1.4 M rows per seed and is the largest CSV
written by the simulator. Roughly 17 + 35·N columns ⇒ ~300 columns at
the default 8-agent config.

### Market state (one row per year)

| Column | Meaning |
|---|---|
| `episode`, `year` | Indices, both zero-based. |
| `cap` | This year's emission cap (Mt). |
| `auction_volume` | Mt offered in the primary auction (after MSR withhold/release and unsold rollover). |
| `tnac` | Total Number of Allowances in Circulation at start of year — drives MSR. |
| `clearing_price` | Uniform-price auction clearing (€/tCO₂, nominal). |
| `secondary_price` | Volume-weighted secondary-market clearing price for the year. |
| `msr_reserve` | MSR reserve balance (Mt). |
| `msr_total_cancelled` | Cumulative cancellations under the MSR cancellation mechanism. |
| `msr_withhold_this_year`, `msr_release_this_year` | MSR flow this year (Mt). |
| `inflation_rate`, `inflation_factor` | Year-on-year and cumulative inflation factors. |
| `phantom_bid_price`, `phantom_bid_qty`, `phantom_active` | Phantom-bidder injection state. |
| `marginal_ef_used` | Average emission factor of the marginal MWh dispatched this year (system-wide). |

#### Auction & secondary-market scalars (single value per year)

Year-level scalars surfaced from the `auction_stats` sub-dict in
`info["year_log"]`, plus exogenous-state diagnostics:

| Column | Meaning |
|---|---|
| `auction_total_demand` | Total Mt of valid bid demand submitted (incl. agents and phantom) before clearing. |
| `auction_unsold` | Mt offered but not allocated this year. |
| `auction_hhi` | Herfindahl-Hirschman index of the *allocation* shares (`Σ (share×100)²`). |
| `auction_max_agent_share` | Largest single-agent share of allocations this year. |
| `auction_failed` | `1` if the auction failed (no valid bids / under-subscribed cancellation), else `0`. |
| `auction_defaults` | Number of agents whose post-clearing settlement defaulted this year. |
| `auction_defaulted_volume` | Mt that defaulted at settlement (rolled into next year's q_cap unless `carry_forward_defaults=false`). |
| `effective_reserve_price` | Reserve price actually used by the clearer this year (€/tCO₂). |
| `secondary_n_buyers_intent` | Number of agents that submitted a positive (buy) signed quantity to the secondary market. |
| `secondary_n_sellers_intent` | Number of agents that submitted a negative (sell) signed quantity. |
| `secondary_n_buyers_executed` | Number of agents that ended up with a positive realised secondary trade. |
| `secondary_n_sellers_executed` | Number of agents that ended up with a negative realised secondary trade. |
| `common_emission_shock` | System-wide common component of the correlated emission shock (`η_t × σ`). |
| `fundamental_anchor` | MAC-scarcity-penalty fundamental anchor used as the AR(1) floor and observation reference (€/tCO₂, nominal). |

### Per-agent year-level state (`_A{i}`)

| Column | Meaning |
|---|---|
| `bank_start_A{i}` | Allowance bank at start of year (Mt). |
| `alloc_A{i}` | Auction allowances won this year (Mt, post-collateral / share cap). |
| `emissions_A{i}` | Realised emissions (Mt). |
| `trade_qty_A{i}` | Net secondary trade (Mt; +buy, −sell). |
| `trade_cost_A{i}` | Net secondary cash flow (M€; +pay, −receive). |
| `green_frac_A{i}`, `delta_green_A{i}` | Generation green share and YoY change. |
| `shortfall_A{i}` | Mt non-compliance shortfall this year (post-secondary, post-bank). |
| `penalty_A{i}` | M€ penalty paid this year (= shortfall × `effective_penalty_rate`). |
| `reward_A{i}`, `reward_base_A{i}`, `reward_shaping_A{i}` | Per-year reward and its decomposition. |
| `holdings_A{i}` | Allowance holdings at end of year (Mt). |
| `invest_cost_A{i}` | Investment cash outflow this year (M€). |
| `collateral_cost_A{i}` | Collateral opportunity cost on locked auction collateral (M€). |
| `bid_price_A{i}` | Phase-1 bid price submitted (€/tCO₂). |
| `queue_size_A{i}` | Construction queue length end-of-year (# projects). |
| `emission_shock_A{i}`, `cf_shock_A{i}` | Realised demand and capacity-factor shocks this year. |
| `cancellation_A{i}` | # construction projects cancelled this year. |
| `auction_cost_A{i}`, `secondary_net_A{i}` | Cost-bucket breakdown (M€). |
| `compliance_surplus_A{i}` | (alloc + bank + buy) − (emiss + cf), Mt. |
| `bank_end_A{i}` | Bank holdings at end of year (Mt). |
| `mac_reduction_A{i}`, `mac_cost_A{i}` | MAC fuel-switching abated tonnes / cost (Mt, M€). |
| `terminal_bank_value_A{i}` | Year-T-only: discounted terminal value of unused bank (M€). |
| `terminal_queue_value_A{i}` | Year-T-only: NPV of pipeline projects at episode end (M€). |
| `terminal_liquidation_value_A{i}` | Year-T-only: bank + queue + treasury terminal payoff. |
| `sec_price_mult_A{i}` | Phase-2 action[0] (absolute €/tCO₂ price ask/bid). |
| `sec_qty_action_A{i}` | Phase-2 action[1] (Mt, signed). |
| `sec_action_side_A{i}` | −1 = sell intent, 0 = hold, +1 = buy intent. |
| `bid_qty_mult_A{i}` | Phase-1 action[1]: multiplier on `estimate_need`. |
| `estimate_need_A{i}` | Pre-auction deterministic need estimate (Mt). |
| `bid_coverage_A{i}` | alloc / need ratio after auction (clipped to 1.0). |
| `bid_to_reserve_A{i}` | Bid price / reserve price. |
| `invest_tech_choice_A{i}` | Argmax of softmax tech logits (0 = onshore, 1 = offshore, 2 = solar). |
| `diag_S_financial_A{i}`, `diag_S_green_A{i}`, `diag_S_composite_A{i}` | Year-level diagnostic scores, [0,1]. |
| `wtp_economic_A{i}`, `wtp_budget_A{i}`, `wtp_binding_A{i}` | Willingness-to-pay diagnostics: economic ceiling (next-year anchor × scarcity), budget ceiling, and which one binds. |
| `invest_frac_pre_clip_A{i}`, `invest_frac_post_clip_A{i}` | Phase-1 action[2] before / after the budget hard-gate clip. |
| `available_budget_A{i}` | Effective annual budget at start of year (M€). |
| `compliance_share_of_available_A{i}` | (auction + secondary + MAC) / `available_budget`. |

#### Per-agent compliance & credit state (year-level)

| Column | Meaning |
|---|---|
| `carry_forward_start_A{i}` | Carry-forward debt inherited at the **start** of the year (Mt). Equal to `old_carry_forward[i]` in `info["year_log"]`. |
| `carry_forward_end_A{i}` | Carry-forward debt rolling into the **next** year (Mt). After-compliance value of `company._carry_forward`. |
| `coverage_gap_A{i}` | Pre-secondary `max(0, need − alloc)` for agent *i*, in Mt. Drives the gap-penalty term in the bid-head reward. |
| `effective_penalty_rate_A{i}` | Inflation-adjusted nominal penalty rate the agent faces this year (€/tCO₂). |
| `treasury_reserve_A{i}` | Treasury balance at end of year (M€). |
| `treasury_drawn_A{i}` | M€ drawn from treasury this year (settlement waterfall stage B). |
| `loan_outstanding_A{i}` | Outstanding emergency-loan balance at end of year (M€). |

---

## 4. `checkpoints_*/agent_{i}_ep{N}.pt` and `agent_{i}_best.pt`

Per-agent PyTorch checkpoint written by `PPOAgent.save()`. Saved with
`torch.save(...)` as a single dictionary loadable with
`torch.load(..., weights_only=True)`. Filename conventions:

| Filename | When written |
|---|---|
| `agent_{i}_ep0.pt` | At episode 0, **before** any training (cold-start baseline). |
| `agent_{i}_ep{N}.pt` | Every `logging.save_interval` episodes. |
| `agent_{i}_best.pt` | Whenever this agent achieves a new best episode reward. |

Online pruning (`logging.checkpoint_pruning.online=true`, default)
keeps only the `n_keep_recent` newest periodic checkpoints + the
`n_keep_milestones` log-spaced milestones + ep0 + `_best.pt`.

### Checkpoint contents

| Key | Type | Description |
|---|---|---|
| `auction_policy` | `state_dict` | Phase-1 policy: layernorm, FC layers, conditioned `price_head` / `qty_head` / `rest_head`, `log_std` parameter. Output: 6-D Gaussian over `[bid_price, qty_mult, invest_frac, logit_onshore, logit_offshore, logit_solar]`. See `docs/action_space.md`. |
| `secondary_policy` | `state_dict` | Phase-2 policy: layernorm, FC, mean head, `log_std`. Output: 2-D Gaussian over `[sec_price_abs, sec_qty]`. |
| `value_net` | `state_dict` | Centralised critic on the global state — produces `V(s)` for the bid-stream advantage. |
| `value_net_invest` | `state_dict` | Second critic for the investment / Phase-2 stream (split-head HAPPO). |
| `auction_optimizer`, `secondary_optimizer`, `critic_optimizer`, `critic_invest_optimizer` | optimiser `state_dict` | Adam moments for resumable training. |

`PPOAgent.load()` accepts the legacy combined-actor checkpoint key
`actor_optimizer` for backward compatibility.

---

## 5. `snapshots/` — rolling intermediate CSV copies

Every `logging.snapshot_interval` episodes, the live cumulative
`training_log` and `year_log` are *copied* into
`snapshots/training_log{tag}_s{seed}_ep{N}.csv` and
`snapshots/year_log{tag}_s{seed}_ep{N}.csv`. Used to snapshot training
progress without interrupting the live writer. Retention is bounded by
`logging.snapshot_keep_recent` (default `1`, oldest pairs deleted as new
ones are written). When `logging.snapshot_delete_on_finish=true`
(default) the snapshot directory is purged on clean run completion —
the live cumulative CSVs supersede every snapshot.

The snapshot CSVs share the **exact same schema** as the live
training-log / year-log files described in §2 and §3 above; they are
truncated at episode `N` rather than at the final episode.

---

## 6. `run_<variant>_s<seed>.log` — captured stdout

A sweep launcher tees the entire stdout/stderr of each `train.py`
subprocess into this file. Useful for debugging crashed seeds without
re-running. Format is whatever `train.py` printed: header banner,
config summary, periodic episode lines (`Ep 1200/100000 …`), warning
detector messages, and the final summary block.

---

## 7. Cross-referencing with notebooks

The notebooks in `notebooks/` consume these files as follows:

* **`ets_marl - Full Run & Analysis.ipynb`** — primary consumer of
  `training_log_*.csv` (episode plots) and `year_log_*.csv` (per-year
  trajectory plots, market dynamics, compliance attribution).
* **`ets_marl - Sweep Analysis.ipynb`** — aggregates across
  `<output_dir>/<variant>/training_log_<variant>_s*.csv` to compare
  variants on `quality_score`, mean clearing prices, default counts,
  and the `udbc_*` compliance attribution buckets.

The CSVs are append-only during a run and committed on close — no
cross-process locking is needed; sweep workers each own their own
output directory.

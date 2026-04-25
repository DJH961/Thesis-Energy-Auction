# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

ETS MARL (v8.0.0) — a multi-agent reinforcement learning simulation of the EU Emissions Trading System. 8 learning agents (PPO/HAPPO) and up to 8 heuristic bots compete in a carbon allowance market over 12-year episodes. Agents bid in uniform-price auctions, trade on a secondary market, and invest in renewable energy capacity. This is a PhD thesis project.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/test_market_clearing.py -v --tb=short

# Run a specific test
python -m pytest tests/test_environment.py::test_unsold_volume_rolls_over_to_next_year -v

# Quick smoke test (100 episodes, fast)
python scripts/train.py --config configs/smoke_100.yaml --seed 42

# Full training run (100k episodes, hours)
python scripts/train.py --config configs/default.yaml --seed 42

# Evaluate trained checkpoint
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt

# Install dependencies (uv preferred)
uv sync
# or: pip install -r requirements.txt
```

## Architecture

### Two-Phase Episode Loop

Each year within an episode has two decision phases, managed by `ETSEnvironment`:

1. **Phase 1 (Auction + Investment)**: agents observe market state → output 6D continuous action `[bid_price, qty_mult, invest_frac, tech_logit_onshore, tech_logit_offshore, tech_logit_solar]` → auction clears → investments queued
2. **Phase 2 (Secondary Market)**: agents observe auction results → output 2D continuous action `[sec_price_abs, sec_qty]` → secondary market clears → compliance check → rewards

### Key Modules

- **`src/environment/ets_environment.py`** — `ETSEnvironment` class: the main simulation loop orchestrating auctions, trading, investment, compliance, and reward computation. This is the largest and most complex file.
- **`src/environment/company.py`** — `Company` class: per-agent state (energy mix, allowance bank, construction queue, budget, carry-forward obligations, emergency loans). Handles observation vector construction (`obs_dim_phase1`, `obs_dim_phase2`).
- **`src/environment/cap_schedule.py`** — `CapSchedule` class: emission cap trajectory with linear reduction factor (LRF), Market Stability Reserve (MSR) with 1-year TNAC lag, cancellation mechanism, and emergency price containment.
- **`src/auction/market_clearing_ets.py`** — Uniform-price sealed-bid auction clearing. `build_bids()` → `market_clearing_ets()` → `settle_auction()`.
- **`src/agents/ppo_agent.py`** — `PPOAgent` with `RewardNormalizer`. Wraps two policy networks and a centralized critic. Handles HAPPO sequential updates, optional behavioral cloning pretraining (disabled by default in v8), and KL-anchored policy regularization. `inject_fundamental_anchor(year)` seeds the auction price head bias to the fundamental anchor each episode.
- **`src/utils/price_anchor.py`** — `compute_fundamental_anchor(year, config)`: MAC-scarcity-penalty price anchor (no CapSchedule dependency). Drives AR(1) mean-reversion floor in `ETSEnvironment` and initial price-head bias in `PPOAgent`. Also `_resolve_cap_year_0(config)` with three-priority fallback.
- **`src/agents/actor_critic.py`** — Neural network architectures: `AuctionPolicy` (6D, conditioned heads: qty depends on price), `SecondaryPolicy` (2D), `ValueNetwork` (centralized critic with global state input).
- **`src/agents/heuristic_policy.py`** — Rule-based bot policy: fundamentals-based MAC→penalty bidding, NPV-gated investment, target-bank trajectory trading.
- **`scripts/train.py`** — Training entry point. Handles schedule auto-scaling, behavioral cloning warm-start, HPP (Historical Policy Pool), epsilon-greedy exploration decay, entropy decay, diagnostics, CSV logging, and checkpointing.

### Config System

YAML-based (not Hydra in this repo). Config files in `configs/`:
- `default.yaml` — full 100k-episode training config (8 learning agents, 0 bots by default in current version)
- `smoke_100.yaml` — 100-episode smoke test config (8 learning + 8 bots)
- `scenarios/` — scenario variants (baseline_2024, low_price_2020)

Many schedule parameters use `0` to mean "auto-scale relative to `n_episodes`" (e.g., `epsilon_decay_episodes: 0` → 35% of n_episodes). The `tabula_rasa` block is kept in config as an ablation reference but **`tabula_rasa.enabled=true` raises `ValueError`** in v8 — it is no longer a runtime override.

### Observation Space

Phase 1 obs includes: year/cap/price signals, agent-specific state (mix, bank, emissions, budget headroom, carry-forward, loan state), and opponent modeling (6D per opponent: bid, green_frac, bank, emissions, budget, compliance). Phase 2 obs = Phase 1 + auction results (clearing price, won qty, coverage ratio, etc.).

Dimensions depend on total agent count: `obs_dim_phase1 = 33 + 6*(N_total-1)`.

### Reward Structure

Reward combines: electricity revenue, compliance costs, trading costs, investment costs, MAC costs, penalty costs, bid collateral costs, budget overspend penalties, green investment shaping (decaying), ESG signal (saved-carbon-years), and terminal values (bank + construction queue NPV). Per-agent `RewardNormalizer` with EMA stabilizes scale.

### Training Pipeline

1. Optional behavioral cloning pretraining from heuristic policy
2. Critic warmup phase (critic trains alone, actors frozen)
3. Main HAPPO loop with: epsilon-greedy exploration (decaying), entropy regularization (decaying), KL anchor to BC policy (decaying), green shaping (decaying), HPP policy swaps
4. Cosine LR annealing for both actor and critic optimizers

### Bot System

Bots use the same `Company` class but actions come from `heuristic_policy.py` instead of neural networks. They are indexed after learning agents (indices 8-15 when `n_bot_agents=8`). Bot arrays in config (`bot_initial_mix`, `bot_annual_budgets`, `bot_capex_throughputs`, `bot_debt_headrooms`) must stay in sync with `n_bot_agents`.

## Conventions

- Units: emissions in Mt (megatonnes), prices in EUR/tonne, energy in TWh, budgets/costs in M€
- Technologies indexed as: `[coal=0, gas=1, onshore_wind=2, offshore_wind=3, solar=4]`
- Agent IDs: learning agents 0..n_agents-1, bots n_agents..n_agents+n_bot_agents-1
- Config values marked "auto" (set to 0) are computed as fractions of `n_episodes` at runtime
- `ets.cap_year_0` is dynamically calibrated from participant emissions unless `cap_year_0_override` is set

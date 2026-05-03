# CLAUDE.md

Short orientation file for any Claude Code (or other LLM) session
working in this repo. Points at canonical references rather than
duplicating them.

## What this is

ETS MARL — a multi-agent reinforcement-learning simulation of the EU
Emissions Trading System. Eight learning agents (PPO with HAPPO-style
sequential updates) bid in annual uniform-price auctions, trade on a
secondary market, and invest in renewable capacity over 12-year
episodes. Master-thesis project.

## Where to look

| Question | File |
|---|---|
| Big-picture architecture and the env loop | `docs/design.md` |
| Every config knob, grouped by subsystem | `docs/config_dictionary.md` |
| Every column in `training_log_*.csv` / `year_log_*.csv` | `docs/data_dictionary.md` |
| Action-space layout, bounds, derivation | `docs/action_space.md` |
| Reward formula (cost buckets, ESG, terminal payoffs) | `docs/reward_function.md`, `docs/esg_reward_design.md` |
| Research questions and which notebook addresses each | `docs/research_questions.md` |
| Methodology for the data-science chapter | `docs/data_science_analysis.md` |
| Parquet log cache + end-of-run compression | `docs/run_data_cache.md` |
| Release history | `docs/changelog.md` |

The current release version lives in `pyproject.toml` and
`configs/default.yaml` (`version:`). Don't pin version numbers anywhere
else.

## Useful commands

```bash
# Test suite
python -m pytest tests/ -v --tb=short

# 100-episode smoke (8 learning + 8 bots)
python scripts/train.py --config configs/smoke_100.yaml --seed 42

# Full run (default 120 000 episodes, 8 learning agents, 0 bots)
python scripts/train.py --config configs/default.yaml --seed 1729

# Sweep launcher
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml --dry-run

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt
```

## Conventions

- Units: emissions in Mt, prices in EUR/tonne, energy in TWh, money in M€.
- Technologies indexed `[coal=0, gas=1, onshore=2, offshore=3, solar=4]`.
- Agent indices: learning agents `0 … n_agents-1`, bots after them.
- Schedule knobs marked `0` (or `-1`) auto-scale relative to
  `simulation.n_episodes`; the formula is in the YAML next to the knob.
- The `tabula_rasa` block is an ablation reference; setting
  `tabula_rasa.enabled=true` raises `ValueError` (no longer a runtime
  override).
- Code/config comments are evergreen: describe what and why, not which
  release added it. Versioned history lives in `docs/changelog.md`.

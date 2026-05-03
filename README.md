# ETS MARL — A Multi-Agent Simulation of the EU Emissions Trading System

Master-thesis project by **Daniel Henke** and **Alessio Desideri** at
**Copenhagen Business School (CBS)**, MSc in Business Administration and
Data Science.

A multi-agent reinforcement-learning (PPO with HAPPO-style sequential
updates) simulation of the post-2023 EU ETS. Eight learning power-sector
companies — split across four archetypes (coal-heavy, gas-dominant,
transitioner, green-leader) and two reward weightings (pure financial vs
balanced ESG) — bid in annual uniform-price auctions, trade allowances on
a secondary market, and invest in renewable capacity over 12-year
episodes. The aim is to use the simulator as a Data-Science artefact:
study how the regulatory mechanisms of the EU ETS shape company
behaviour, and how individual strategies aggregate into market-level
outcomes.

Detailed design notes, config schemas, reward derivations, and data
dictionaries live under [`docs/`](docs/).

---

## Research Questions

The thesis is organised around one main research question and three
sub-questions. The full framing — including which notebook addresses
which RQ — lives in
[`docs/research_questions.md`](docs/research_questions.md).

> **Main RQ.** How can multi-agent reinforcement learning be used to
> simulate the EU Emissions Trading System, and what does the resulting
> simulation reveal about the system's regulatory mechanisms and the
> strategic behaviour of power-sector participants balancing financial
> and environmental objectives?

> **Sub-RQ 1.** What environment and algorithm design choices are
> required to build a stable and behaviourally credible simulation of a
> carbon market?

> **Sub-RQ 2.** How sensitive are market outcomes such as price
> stability, compliance, and the pace of decarbonisation to the
> regulatory mechanisms of the EU ETS?

> **Sub-RQ 3.** How do financial and environmental objectives shape the
> strategies of power companies in this market, and how do these
> individual strategies aggregate into market-level outcomes?

Cross-config sensitivity (RQ2) is driven by the sweep specs under
`configs/sweeps/`; the corresponding analysis lives in
`notebooks/ets_marl - Thesis Experiments Analysis.ipynb`.

---

## How the Simulation Works

Each episode simulates **12 years** of an EU-ETS-style carbon market.
Every year the regulator releases a shrinking emission cap and runs a
**uniform-price sealed-bid auction**; companies then trade on a
**secondary market**, optionally **invest** in onshore wind / offshore
wind / solar (with multi-year construction delays), and either comply or
pay a **penalty** (base €138.75/t in 2026, indexed by inflation). A
**Market Stability Reserve (MSR)** absorbs surplus allowances using the
post-2023 EU-ETS rules (1-year TNAC lag, three-band withholding,
cancellations).

The default profile uses **8 learning agents** (PPO/HAPPO) and **no
heuristic bots**:

| Archetype | Energy mix | A* (financial) | A* (ESG) |
|---|---|---|---|
| Coal-heavy | ~55 % fossil | A1 `[1.0, 0.0]` | A2 `[0.5, 0.5]` |
| Gas-dominant | ~50 % fossil | A3 `[1.0, 0.0]` | A4 `[0.5, 0.5]` |
| Transitioner | ~35 % fossil | A5 `[1.0, 0.0]` | A6 `[0.5, 0.5]` |
| Green-leader | ~25 % fossil | A7 `[1.0, 0.0]` | A8 `[0.5, 0.5]` |

Even-indexed agents (A1, A3, A5, A7) optimise pure financial reward;
odd-indexed agents (A2, A4, A6, A8) balance financial cost against an
ESG signal. All companies produce **10 TWh/yr** of electricity.
`configs/smoke_100.yaml` is a 100-episode smoke variant that additionally
activates 8 fixed-policy heuristic bots for short validation runs.

**Action space.** Phase 1 (auction + investment): bid price, bid quantity
multiplier, investment fraction, three tech logits (onshore / offshore /
solar). Phase 2 (secondary): absolute price, signed quantity (positive =
buy, negative = sell). Full schema in
[`docs/action_space.md`](docs/action_space.md).

**Reward.** Per-agent
`R = w_cost · (−cost_norm) + w_green · esg_signal − penalty_norm + shaping + terminal_values`.
Electricity revenue is logged but **not** in the gradient (agents cannot
materially influence it within the year). Full derivation, ESG
calibration, and terminal-value formulas in
[`docs/reward_function.md`](docs/reward_function.md) and
[`docs/esg_reward_design.md`](docs/esg_reward_design.md).

**Learning algorithm.** HAPPO with split-head actors (a bid sub-head and
an investment sub-head share an auction trunk; each has its own critic),
fundamental price anchor seeded each episode, WTP-anchored ε-greedy
exploration, decaying entropy and KL anchors, and an optional Historical
Policy Pool. Full design in [`docs/design.md`](docs/design.md).

**Key regulatory and economic mechanisms.** MSR with cancellations,
inflation path, MAC fuel-switching, joint budget gate, bid-change limit,
emergency loans, corporate treasury reserve, urgency scalars, optional
phantom bidder. Each is documented in
[`docs/design.md`](docs/design.md); the configurable knobs live in
[`docs/config_dictionary.md`](docs/config_dictionary.md).

---

## Project Structure

```
Thesis-Energy-Auction/
├── src/
│   ├── environment/      # ETS simulation: env loop, company state, cap/MSR, phantom bidder
│   ├── agents/           # PPO / HAPPO, actor-critic networks, heuristic policy, Q-learning
│   ├── auction/          # Uniform-price auction clearing
│   ├── analysis/         # Post-hoc analysis helpers
│   └── utils/            # Fundamental price anchor, preflight, sweep, parquet log cache
├── configs/
│   ├── default.yaml      # Full training config (120 000 episodes, 8 learning agents)
│   ├── smoke_100.yaml    # 100-episode smoke test (8 learning + 8 bots)
│   ├── bots_only.yaml    # Heuristic-bot-only baseline
│   ├── qlearning.yaml    # Tabular Q-learning baseline config
│   ├── scenarios/        # Scenario variants
│   └── sweeps/           # Sweep specs (incl. the thesis experiments)
├── scripts/
│   ├── train.py          # Main training entry point
│   ├── sweep.py          # Sweep launcher (variants × seeds in a process pool)
│   ├── evaluate.py       # Evaluate trained checkpoints
│   ├── evaluate_qlearning.py  # Q-learning evaluator
│   ├── run_bots_only.py  # Bots-only multi-seed driver
│   └── compress_results.py    # One-shot CSV→parquet + checkpoint archiver
├── tests/                # pytest suite (env, agents, rewards, sweeps, run-data, …)
├── notebooks/            # Analysis notebooks (sweep, default-RQ, full-run, debug, baselines)
├── docs/                 # Design, changelog, dictionaries, ESG design, analysis methodology
└── archive/              # Legacy code and historical planning docs
```

---

## Setup

Requires **Python 3.11+**.

```bash
# Linux / Mac
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: bash setup.sh

# Windows
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt        # or: install.bat
```

`uv sync` is also supported for faster installs.

---

## Running

```bash
# Quick smoke test (100 episodes)
python scripts/train.py --config configs/smoke_100.yaml --seed 42

# Full training run (default = 120 000 episodes)
python scripts/train.py --config configs/default.yaml --seed 1729

# Multiple seeds in parallel (bit-identical to sequential runs)
python scripts/train.py --config configs/default.yaml --seed 1729 8191 6561 5041 --parallel-seeds 4

# Sweeps over variants × seeds (writes to <spec>.output_dir)
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml --dry-run
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml

# Heuristic-bot-only baseline (no learning)
python scripts/run_bots_only.py --config configs/bots_only.yaml --seeds 42 123 456

# Evaluate a trained checkpoint
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt

# Test suite
python -m pytest tests/ -v --tb=short
```

Sweep workers tee per-seed stdout to
`<output_dir>/<variant>/run_<variant>_s<seed>.log`; the parent terminal
prints short `[START]` / `[LIVE]` heartbeats / `[DONE]` lines and an
aggregate ETA. Sweep CSV / checkpoint filenames are tagged with the
variant name (`training_log_<variant>_s<seed>.csv`, etc.) so files stay
unique when collected into a single folder. See
[`configs/sweeps/example_sweep.yaml`](configs/sweeps/example_sweep.yaml)
and [`src/utils/sweep.py`](src/utils/sweep.py) for the spec schema.

At clean end-of-run, training logs are converted to zstd parquet and
checkpoint directories are bundled into a single `tar.xz`; notebooks load
either form transparently. See
[`docs/run_data_cache.md`](docs/run_data_cache.md) for the loader API and
the one-shot migration script.

---

## Documentation

The [`docs/`](docs/) directory holds the thesis-supporting reference
material — environment and algorithm design, reward derivations, the
config dictionary, the data dictionary for the training-log / year-log
CSVs, ESG calibration notes, the data-science methodology, the changelog,
and the project-stats snapshot. Start with
[`docs/design.md`](docs/design.md) for environment and algorithm design,
[`docs/research_questions.md`](docs/research_questions.md) for the RQ
framing, and [`docs/config_dictionary.md`](docs/config_dictionary.md) for
every config knob and its effect.

The `notebooks/` directory contains analysis notebooks that consume the
training-log / year-log CSVs documented in
[`docs/data_dictionary.md`](docs/data_dictionary.md) (operational
diagnostics, sweep analysis, per-RQ thesis notebooks, and Q-learning /
bots-only credibility baselines).

---

## Acknowledgements

- Auction clearing adapted from [ckrk/bidding_learning](https://github.com/ckrk/bidding_learning) (MIT License)
- ETS parameters from the EU ETS Handbook, EU Carbon Market Report 2025, IRENA 2024, and IPCC AR5
- PPO / HAPPO algorithms based on Schulman et al. (2017) and Kuba et al. (2022)

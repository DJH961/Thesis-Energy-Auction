# ETS MARL — Current Version (HAPPO/PPO) v5.4

This is the **active, main version** of the carbon market simulation. It uses modern reinforcement learning (PPO/HAPPO) to simulate energy companies competing in a simplified EU Emissions Trading System, now with **8 heuristic bot agents** that mirror all learning agent archetypes and add realistic market demand.

## What Does This Code Do?

At a high level, this project:

1. **Creates a virtual carbon market** that mimics the real EU ETS (auctions, caps, penalties, banking, etc.)
2. **Places 8 AI-controlled (learning) companies** and **8 heuristic bot companies** into that market, each with a different energy mix
3. **Lets the learning companies learn** over thousands of simulated episodes how to bid in auctions, trade allowances, and invest in clean energy — while bots provide realistic background demand
4. **Records everything** so we can analyze what strategies emerge and what they tell us about real carbon markets

## How the Simulation Works

### The Carbon Market

Each "episode" simulates **12 years** of a carbon market. Every year:

1. The government sets a **cap** — the total CO2 allowed. This cap **shrinks each year** (by about 4.3-4.4%) to push companies toward cleaner energy. Over 12 years, the cap drops from 57.0 Mt (~45 Mt initial emissions for 16 participants).
2. Companies participate in an **auction** where they bid for emission allowances (each allowance = right to emit 1 tonne of CO2).
3. The auction uses a **uniform price** — everyone pays the same price, which is the lowest winning bid. This is how the real EU ETS works.
4. Companies that don't have enough allowances to cover their emissions face a **penalty** (base **€138.75/t in 2026**, indexed from €132.06 in 2024; plus carry-forward obligations).
5. Companies can **bank** (save) unused allowances for future years.
6. A **Market Stability Reserve (MSR)** automatically adjusts the auction supply — if too many allowances are floating around, it pulls some out; if prices spike, it releases extras. Price-responsive triggers prevent procyclical hoarding.

### Terminal Value Rewards

At the end of each episode (final year), agents receive terminal value bonuses that account for the beyond-horizon value of their decisions:

- **Bank terminal value**: Banked allowances are valued at the current clearing price, rewarding agents for prudent stockpiling.
- **Queue terminal value**: In-construction green projects are valued based on their discounted future carbon savings over a configurable payoff horizon (default: 5 years). This prevents the end-of-episode problem where agents stop investing because projects won't complete within the episode.

Both can be independently enabled/disabled via config flags (`reward.terminal_bank_value`, `reward.terminal_queue_value`).

### The Companies (Agents)

There are **16 total market participants**: 8 learning agents (PPO/HAPPO) and 8 heuristic bot agents.

**Learning Agents (A1-A8)** — organized into 4 archetypes (2 of each — one financially-motivated, one ESG-balanced):

| Archetype | Energy Mix | Reward Weights [w_cost, w_green] |
|-----------|-----------|----------------------------------|
| **Coal-heavy** (A1, A2) | ~80% fossil (40% coal, 40% gas) | A1: [1.0, 0.0] pure financial; A2: [0.5, 0.5] balanced ESG |
| **Gas-dominant** (A3, A4) | ~60% gas, 40% green | A3: [1.0, 0.0] pure financial; A4: [0.5, 0.5] balanced ESG |
| **Transitioner** (A5, A6) | ~30% fossil, 70% green | A5: [1.0, 0.0] pure financial; A6: [0.5, 0.5] balanced ESG |
| **Green-leader** (A7, A8) | ~10% gas, 90% renewable | A7: [1.0, 0.0] pure financial; A8: [0.5, 0.5] balanced ESG |

Even-indexed agents (A1, A3, A5, A7) have pure financial reward weights [1.0, 0.0]; odd-indexed agents (A2, A4, A6, A8) have balanced ESG weights [0.5, 0.5].

**Heuristic Bot Agents (B1-B8)** — fixed-policy agents using rule-based strategies from `heuristic_policy.py`, mirroring all 8 learning agent archetypes:

| Bot | Energy Mix | Description |
|-----|-----------|-------------|
| **B1, B2** | ~80% fossil (40% coal, 40% gas) | Coal-heavy mirrors |
| **B3, B4** | ~60% gas, 40% green | Gas-dominant mirrors |
| **B5, B6** | ~30% fossil, 70% green | Transitioner mirrors |
| **B7, B8** | ~10% gas, 90% renewable | Green-leader mirrors |

Bots use the same `Company` class and participate identically in auction clearing and secondary market matching. They are **not trained** — their actions come from the heuristic policy (valuation-based bidding, NPV-gated investment, target-bank trajectory trading). Bots are indexed after learning agents (indices 8-15) and are excluded from PPO updates. Bot `reward_weights` in the config are for evaluation logging only — bots use `heuristic_policy`, not rewards.

All companies produce **10 TWh/year** of electricity — the same output, but very different carbon footprints.

### What Each Agent Decides

Every year, each AI agent makes **6 decisions** (Phase 1) plus **2 more** (Phase 2):

**Phase 1 — Auction & Investment:**
- **Bid price**: How much to offer per allowance (€/tonne, range: 5-500)
- **Bid quantity**: Coverage multiplier on estimated annual need (0.3-1.3x)
- **Investment fraction**: What share of capacity to convert to green (0-10%)
- **Technology choice**: Where to invest — onshore wind (3yr delay), offshore wind (5yr delay), or solar (1yr delay)

**Phase 2 — Secondary Market:**
- **Secondary price**: Price multiplier on clearing price (0.8-1.3x)
- **Secondary quantity**: How many allowances to trade (positive = buy, negative = sell)

### How Agents Learn

The agents use **HAPPO (Heterogeneous-Agent PPO)**, a multi-agent reinforcement learning algorithm with centralized critics.

- Agents are pre-trained with **behavioral cloning** from a heuristic policy to seed sensible initial strategies
- Each agent has a centralized critic that sees the global state, enabling coordinated learning
- **Epsilon-greedy exploration** decays from 25% to 5% over training, preventing policy collapse
- **Historical Policy Pool** maintains past policy snapshots for opponent diversity
- **Auto-scaled schedules**: warmup, pretraining, exploration decay, and HPP timing can scale automatically with `n_episodes`
- Over 70,000 episodes, agents converge on sophisticated market strategies

The reward signal balances:
- **Revenue** from selling electricity (including carbon cost passthrough)
- **Costs** of buying allowances, trading, investing, and operations
- **Penalties** for non-compliance (with carry-forward obligations)
- **Emissions intensity** — penalizes higher emission factors
- **Green investment shaping** — bonus for increasing green fraction (decays over training)
- **Terminal values** — end-of-episode valuation of banked allowances and in-construction projects

### Key Mechanisms

- **Inflation path**: Annual inflation is sampled from historical calibration **N(μ=2.0%, σ=1.5%)**, then applied economy-wide to nominal costs
- **MAC fuel-switching**: When carbon prices exceed €65/t, companies automatically switch up to 20% of coal dispatch to gas (short-run operational change, not investment)
- **Electricity revenue**: Companies earn revenue from electricity sales, with carbon costs partially passed through to electricity prices (80%). Green generators benefit from the same revenue with lower carbon costs.
- **Dynamic reserve price**: Auction floor price adapts based on a 3-year moving average of secondary market prices
- **Carry-forward**: Non-compliance shortfall is added to next year's obligation (capped at 2.0x, allowing larger debt accumulation)

## Project Structure

```
ets_marl_happo_current/
│
├── src/                          # All the simulation code
│   ├── environment/
│   │   ├── ets_environment.py    # The main simulation loop (auction → trade → invest → repeat)
│   │   ├── company.py            # Each company's state: portfolio, allowances, budget, etc.
│   │   └── cap_schedule.py       # How the emission cap shrinks + MSR logic
│   │
│   ├── agents/
│   │   ├── ppo_agent.py          # The PPO/HAPPO learning algorithm
│   │   ├── actor_critic.py       # The neural networks (the "brains")
│   │   ├── heuristic_policy.py   # Rule-based heuristic for behavioral cloning warm-start
│   │   └── noise.py              # Exploration noise (helps agents try new things)
│   │
│   ├── auction/
│   │   └── market_clearing_ets.py  # Runs the uniform-price auction
│   │
│   └── utils/
│       ├── logger.py             # Logs training metrics
│       └── replay_buffer.py      # Stores past experiences for learning
│
├── configs/
│   └── default.yaml              # All simulation parameters (v5.0)
│
├── scripts/
│   ├── train.py                  # Starts a training run
│   └── evaluate.py               # Tests trained agents
│
├── tests/                        # Automated tests
│   ├── test_market_clearing.py   # Tests the auction works correctly
│   ├── test_cap_schedule.py      # Tests the cap schedule and MSR
│   ├── test_environment.py       # Tests the full simulation
│   ├── test_rewards.py           # Tests reward components and terminal values
│   ├── test_company.py           # Tests company state management
│   ├── test_heuristic.py         # Tests heuristic policy
│   ├── test_mappo.py             # Tests MAPPO/HAPPO integration
│   ├── test_clipped_gaussian.py  # Tests clipped Gaussian policy
│   └── test_anchors.py           # Tests action anchor initialization
│
├── notebooks/
│   └── ets_marl_colab.ipynb      # Google Colab notebook for training in the cloud
│
├── docs/
│   └── design.md                 # Detailed design decisions
│
├── main.py                       # Alternative entry point
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata
├── setup.sh                      # Linux/Mac setup script
└── install.bat                   # Windows setup script
```

## Setup & Installation

### Prerequisites

- **Python 3.11+** (3.12 also works)
- **pip** (Python's package manager — comes with Python)
- (Optional) **uv** for faster dependency installation

### Step-by-Step Setup

**On Windows:**
```bash
# 1. Open a terminal in this folder

# 2. (Recommended) Create a virtual environment so packages don't conflict:
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies:
pip install -r requirements.txt

# Or use the provided batch file:
install.bat
```

**On Linux/Mac:**
```bash
# 1. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies:
pip install -r requirements.txt

# Or use the provided script:
bash setup.sh
```

**On Google Colab (no local setup needed):**
Open `notebooks/ets_marl_colab.ipynb` in Google Colab — it handles installation automatically.

### Running Training

```bash
# Basic training run with default settings:
python scripts/train.py --config configs/default.yaml --seed 42

# Run with multiple seeds for statistical robustness:
python scripts/train.py --config configs/default.yaml --seed 42
python scripts/train.py --config configs/default.yaml --seed 123
python scripts/train.py --config configs/default.yaml --seed 456
```

Training runs 70,000 episodes of 12-year simulations. Results are saved to a `results/` folder.

### Running Tests

```bash
# Run all tests to make sure everything works:
python -m pytest tests/ -v
```

### Evaluating Trained Agents

```bash
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt
```

## Key Configuration (configs/default.yaml)

The most important settings you might want to change:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `simulation.n_episodes` | 70,000 | How many episodes to train for (more = better but slower) |
| `simulation.n_years` | 12 | How many years each episode simulates |
| `companies.n_agents` | 8 | Number of learning agents (PPO) |
| `companies.n_bot_agents` | 8 | Number of heuristic bot agents |
| `ets.cap_year_0` | 57.0 Mt | Starting emission cap (scaled for 16 participants) |
| `auction.price_max` | 500 | Maximum bid price (€/tonne) |
| `penalty.rate` | 138.75 | Fine per excess tonne of CO2 (€), base level at simulation year-0 (2026) |
| `penalty.inflation_rate` | 0.020 | Mean annual inflation for nominal indexing (μ) |
| `penalty.inflation_random_std` | 0.015 | Annual inflation standard deviation (σ), sampled with a normal distribution |
| `penalty.inflation_random_window` | 0.0 | Legacy fallback: uniform ±window (used only if `inflation_random_std = 0`) |
| `reward.terminal_bank_value` | true | Value banked allowances at episode end |
| `reward.terminal_queue_value` | true | Value in-construction projects at episode end |
| `reward.terminal_payoff_years` | 5 | Horizon for terminal queue value NPV calculation |

## Understanding the Output

After training, you'll find in `results/`:
- **Training logs**: Episode rewards, carbon prices, emissions over time
- **Model checkpoints**: Saved neural network weights (so you can resume or evaluate later)
- **Plots**: Visualizations of learning curves and market dynamics

## Acknowledgements

- Auction clearing adapted from [ckrk/bidding_learning](https://github.com/ckrk/bidding_learning) (MIT License)
- ETS parameters from EU ETS Handbook, EU Carbon Market Report 2025, IRENA 2024, IPCC AR5
- PPO/HAPPO algorithms based on Schulman et al. (2017) and Kuba et al. (2022)

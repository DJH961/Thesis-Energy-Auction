# ETS MARL — Current Version (HAPPO/PPO)

This is the **active, main version** of the carbon market simulation. It uses modern reinforcement learning (PPO/HAPPO) to simulate 8 energy companies competing in a simplified EU Emissions Trading System.

## What Does This Code Do?

At a high level, this project:

1. **Creates a virtual carbon market** that mimics the real EU ETS (auctions, caps, penalties, banking, etc.)
2. **Places 8 AI-controlled companies** into that market, each with a different energy mix (some burn mostly coal, others are mostly green)
3. **Lets the companies learn** over thousands of simulated episodes how to bid in auctions, trade allowances, and invest in clean energy
4. **Records everything** so we can analyze what strategies emerge and what they tell us about real carbon markets

## How the Simulation Works

### The Carbon Market

Each "episode" simulates **12 years** of a carbon market (configurable, set to 20 in config but effectively 12 meaningful years). Every year:

1. The government sets a **cap** — the total CO2 allowed. This cap **shrinks each year** (by about 4.3%) to push companies toward cleaner energy.
2. Companies participate in an **auction** where they bid for emission allowances (each allowance = right to emit 1 tonne of CO2).
3. The auction uses a **uniform price** — everyone pays the same price, which is the lowest winning bid. This is how the real EU ETS works.
4. Companies that don't have enough allowances to cover their emissions face a **penalty** (€100 per excess tonne, plus they must make up the shortfall next year).
5. Companies can **bank** (save) unused allowances for future years.
6. A **Market Stability Reserve (MSR)** automatically adjusts the auction supply — if too many allowances are floating around, it pulls some out; if prices spike, it releases extras.

### The Companies (Agents)

There are **8 companies**, organized into 4 archetypes (2 of each):

| Archetype | Energy Mix | Description |
|-----------|-----------|-------------|
| **Coal-heavy** | ~70-80% coal, some gas, little green | High emissions, high cost pressure from carbon pricing |
| **Gas-dominant** | ~60% gas, some green | Medium emissions, more flexible |
| **Mixed** | Balanced fossil/green | Moderate emissions, moderate flexibility |
| **Near-green** | ~70-80% renewable | Low emissions, less need for allowances |

All companies produce **10 TWh/year** of electricity — the same output, but very different carbon footprints.

### What Each Agent Decides

Every year, each AI agent makes **6 decisions** (Phase 1) plus **2 more** (Phase 2):

**Phase 1 — Auction & Investment:**
- **Bid price**: How much to offer per allowance (€/tonne)
- **Bid quantity**: How many allowances to try to buy
- **Investment fraction**: What share of budget to spend on building new green capacity
- **Technology choice**: Where to invest — onshore wind, offshore wind, or solar (3 outputs that get converted to probabilities)

**Phase 2 — Secondary Market:**
- **Secondary price**: At what price to buy/sell allowances from other companies
- **Secondary quantity**: How many to trade (positive = buy, negative = sell)

### How Agents Learn

The agents use **PPO (Proximal Policy Optimization)**, a reinforcement learning algorithm. 

- Each agent tries random strategies at first
- After each episode, it looks at what worked (made money, avoided penalties) and what didn't
- It gradually adjusts its strategy, trying more of what worked
- Over thousands of episodes, it converges on a sophisticated strategy

The reward signal balances:
- **Revenue** from selling electricity (including carbon cost passthrough)
- **Costs** of buying allowances and investing
- **Penalties** for non-compliance
- **Emissions intensity** — a bonus for being cleaner

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
│   │   ├── ppo_agent.py          # The PPO learning algorithm
│   │   ├── actor_critic.py       # The neural networks (the "brains")
│   │   ├── heuristic_policy.py   # Simple rule-based agents for comparison
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
├── tests/                        # Automated tests (38 tests)
│   ├── test_market_clearing.py   # Tests the auction works correctly
│   ├── test_cap_schedule.py      # Tests the cap schedule and MSR
│   └── test_environment.py       # Tests the full simulation
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

Training will take a while (thousands of episodes). Results are saved to a `results/` folder.

### Running Tests

```bash
# Run all tests to make sure everything works:
python -m pytest tests/ -v
```

All 38 tests should pass.

### Evaluating Trained Agents

```bash
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt
```

## Key Configuration (configs/default.yaml)

The most important settings you might want to change:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `simulation.n_episodes` | 30,000 | How many episodes to train for (more = better but slower) |
| `simulation.n_years` | 20 | How many years each episode simulates |
| `companies.n_agents` | 8 | Number of companies in the market |
| `ets.cap_year_0` | 23.5 Mt | Starting emission cap |
| `auction.price_max` | 500 | Maximum bid price (€/tonne) |
| `penalty.base` | 100 | Fine per excess tonne of CO2 (€) |

## Understanding the Output

After training, you'll find in `results/`:
- **Training logs**: Episode rewards, carbon prices, emissions over time
- **Model checkpoints**: Saved neural network weights (so you can resume or evaluate later)
- **Plots**: Visualizations of learning curves and market dynamics

## Acknowledgements

- Auction clearing adapted from [ckrk/bidding_learning](https://github.com/ckrk/bidding_learning) (MIT License)
- ETS parameters from EU ETS Handbook, EU Carbon Market Report 2025, IRENA 2024, IPCC AR5
- PPO algorithm based on Schulman et al. (2017)

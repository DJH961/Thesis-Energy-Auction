# ETS MARL — Auction (HAPPO/PPO)

This is the **auction-focused** carbon market simulation. It introduces a **3-tranche bid ladder** for the primary auction and replaces the bilateral secondary market with a **Uniform-Price Call Auction** (clearinghouse mechanism), mirroring how real EEX/ICE daily fixing works.

Release-specific changes are tracked in `docs/changelog.md`.

This mathematically guarantees maximum social surplus and finds the exact market equilibrium a Continuous Double Auction would discover over a longer time horizon.

## What Does This Code Do?

At a high level, this project:

1. **Creates a virtual carbon market** that mimics the real EU ETS (auctions, caps, penalties, banking, etc.)
2. **Places 8 AI-controlled (learning) companies** and **8 heuristic bot companies** into that market, each with a different energy mix
3. **Lets the learning companies learn** over thousands of simulated episodes how to bid in auctions, trade allowances, and invest in clean energy — while bots provide realistic background demand
4. **Records everything** so we can analyze what strategies emerge and what they tell us about real carbon markets

## How the Simulation Works

### The Carbon Market

Each "episode" simulates **12 years** of a carbon market. Every year:

1. The government sets a **cap** — the total CO2 allowed. This cap **shrinks each year** (by about 4.3-4.4%) to push companies toward cleaner energy. Over 12 years, the cap drops from 50.0 Mt (~11% surplus over ~45 Mt initial emissions for 16 participants).
2. Companies participate in an **auction** where they bid for emission allowances (each allowance = right to emit 1 tonne of CO2).
3. The auction uses a **uniform price** — everyone pays the same price, which is the lowest winning bid. This is how the real EU ETS works.
4. Companies that don't have enough allowances to cover their emissions face a **penalty** (base **€138.75/t in 2026**, indexed from €132.06 in 2024; plus carry-forward obligations).
5. Companies can **bank** (save) unused allowances for future years.
6. A **Market Stability Reserve (MSR)** automatically adjusts the auction supply based on the total number of allowances in circulation (TNAC). The MSR implements the EU ETS post-2023 reform including:
   - **1-year TNAC lag**: MSR uses the *prior year's* TNAC, matching EU ETS Decision 2015/1814 (Art. 1(5)). Year 0 has no MSR intervention.
   - **Three-band withholding**: Uses legislative TNAC proportions lower:mid:upper = 400:833:1096. Above upper: withhold 24% of total TNAC. Between mid and upper: withhold TNAC − mid. Below mid: no intake.
   - **Updated thresholds**: Proportions corrected to match Decision (EU) 2015/1814 Mt values: `tnac_lower_ratio` 0.1314 (400/1096), `tnac_mid_ratio` 0.2737 (833/1096), `tnac_upper_ratio` 0.36 (anchor).
   - **Cancellation mechanism**: MSR holdings exceeding the previous year's auction volume are permanently cancelled.
   - **Smoothed price trigger**: Emergency release requires both an absolute threshold breach (≥ 85% of penalty / 300 EUR/t) *and* a MA3 price spike > 2.5× prior year's MA3. Prevents procyclical flash releases.

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

Bots use the same `Company` class and participate identically in auction clearing and secondary market matching. They are **not trained** — their actions come from the heuristic policy (fundamentals-based MAC→penalty bidding, NPV-gated investment, target-bank trajectory trading with absolute-price secondary market). Bots are indexed after learning agents (indices 8-15) and are excluded from PPO updates. Bot `reward_weights` in the config are for evaluation logging only — bots use `heuristic_policy`, not rewards.

All companies produce **10 TWh/year** of electricity — the same output, but very different carbon footprints.

### What Each Agent Decides

Every year, each AI agent makes **6 decisions** (Phase 1) plus **2 more** (Phase 2):

**Phase 1 — Auction & Investment:**
- **Bid price**: How much to offer per allowance (€/tonne, range: 5-500)
- **Bid quantity**: Coverage multiplier on estimated annual need (0.3-2.0x)
- **Investment fraction**: What share of capacity to convert to green (0-20%, subject to capex throughput cap)
- **Technology choice**: Where to invest — onshore wind (4yr delay), offshore wind (7yr delay), or solar (2yr delay)

**Phase 2 — Secondary Market:**
- **Secondary price**: Absolute price in €/tonne (range: reserve price to 2× effective penalty rate)
- **Secondary quantity**: How many allowances to trade (positive = buy, negative = sell)

### How Agents Learn

The agents use **HAPPO (Heterogeneous-Agent PPO)**, a multi-agent reinforcement learning algorithm with centralized critics.

- Agents are pre-trained with **behavioral cloning** from a heuristic policy to seed sensible initial strategies
- Each agent has a centralized critic that sees the global state, enabling coordinated learning
- **Epsilon-greedy exploration** decays from 25% to 5% over training, preventing policy collapse
- **Historical Policy Pool** maintains past policy snapshots for opponent diversity
- **Auto-scaled schedules**: warmup, pretraining, exploration decay, and HPP timing can scale automatically with `n_episodes`
- Hidden burn-in warm-start initializes banks, MSR reserve, and price history before visible year 0
- Over 100,000 episodes, agents converge on sophisticated market strategies

The reward signal balances:
- **Revenue** from selling electricity (including carbon cost passthrough)
- **Total costs** including allowances, trading, investing, operations, MAC, and penalties (folded into one cost signal)
- **Bid collateral opportunity cost** on overbidding spread (`auction.collateral`): `rate × hold_fraction × max(0, bid - clearing) × qty_won`
- **Opportunity cost of capital** on post-compliance banked allowances (`reward.opportunity_cost_rate`)
- **Green investment shaping** — bonus for increasing green fraction (decays over training), scaled by (0.2 + w_green)
- **ESG signal** — saved-carbon-years formula: `w_green × ef_ratio × time_ratio × (budget/1000)`, rewarding early emission reductions more than late ones
- **Terminal values** — bank value (/1000 scaling) and ESG terminal queue value with γ^years_late discount

### Key Mechanisms

- **Inflation path**: Annual inflation is sampled from historical calibration **N(μ=2.0%, σ=1.5%)**, then applied economy-wide to nominal costs
- **MAC fuel-switching**: When carbon prices exceed €48/t (ICIS mid-range switching cost), companies automatically switch up to 20% of coal dispatch to gas (short-run operational change, not investment)
- **Electricity revenue**: Companies earn revenue from electricity sales, with carbon costs partially passed through to electricity prices (80%). Green generators benefit from the same revenue with lower carbon costs.
- **Unified financial envelope**: Each company has a single annual budget covering all spending (compliance + capex + MAC), calibrated to realistic revenue retention (~€724M for 10 TWh). Coal-heavy companies have the tightest budgets due to higher fuel OPEX.
- **Capex throughput cap**: Organizational constraint on annual construction spend (M€), modelling permitting pipeline capacity, EPC contractor access, and management bandwidth. Independent of the financial budget — a company can afford more investment than it can physically deliver.
- **Static reserve price**: Auction floor price at €30/t (matching price_min)
- **Auction bid collateral**: overbids above clearing incur a real capital lock-up cost on awarded quantity (`auction.collateral.enabled`)
- **Collateral affordability guardrail**: if collateral lock-up is unaffordable, bids are clipped in two steps (quantity first, then price if needed) to preserve feasible participation
- **Budget headroom observation**: Phase-1 dim `[27]` reports current annual budget headroom (`1.0` fresh, `0.0` at limit, negative overspend)
- **Carry-forward**: Non-compliance shortfall is added to next year's obligation (capped at 2.0x, allowing larger debt accumulation)
- **Hidden burn-in warm-start**: A configurable pre-period (`warm_start.burnin_enabled`) seeds realistic bank holdings, MSR reserve, and MA3 history before year 0
- **Fundamentals-based heuristic**: Bot bidding uses MAC→penalty gradient (`mac_cost + urgency × (penalty - mac_cost)`), removing dependence on price moving average
- **Absolute-price secondary market**: Secondary prices are expressed in €/t (not as multipliers), clipped to [sec_price_min, 2× effective penalty rate]
- **ESG signal**: Saved-carbon-years formula rewards emission factor improvements proportional to remaining time, gated by w_green
- **Tabula-rasa balanced price-side exploration**: epsilon-random auction prices in `uniform` mode are sampled 50/50 under vs over expected price (uniform within each side), preventing bias from asymmetric price bounds

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
│   └── default.yaml              # All simulation parameters
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

Training runs 100,000 episodes of 12-year simulations by default. Results are saved to a `results/` folder.

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
| `simulation.n_episodes` | 100,000 | How many episodes to train for (more = better but slower) |
| `simulation.n_years` | 12 | How many years each episode simulates |
| `companies.n_agents` | 8 | Number of learning agents (PPO) |
| `companies.n_bot_agents` | 8 | Number of heuristic bot agents |
| `ets.cap_overhead_pct` | 0.11 | Year-0 cap overhead over total initial emissions (dynamic calibration) |
| `ets.cap_year_0_override` | `null` | Optional hard override for year-0 cap |
| `auction.price_max` | 500 | Maximum bid price (€/tonne) |
| `penalty.rate` | 138.75 | Fine per excess tonne of CO2 (€), base level at simulation year-0 (2026) |
| `tabula_rasa.enabled` | false | Enables no-anchor tabula-rasa overrides with side-balanced under/over expected-price exploration |
| `auction.collateral.enabled` | true | Enables EU ETS-style bid collateral opportunity-cost term on overbids |
| `auction.collateral.opportunity_cost_rate` | 0.05 | Annualized cost-of-capital rate applied to locked collateral |
| `auction.collateral.hold_fraction` | 0.02 | Fraction of year collateral lock-up used in annualized model (~7 days) |
| `auction.collateral.min_qty_floor_frac` | 0.5 | Minimum coverage floor used by pre-auction collateral affordability clip |
| `green_finance.enabled` | false | Enables green-only loan/capex throughput boost during investment clipping |
| `bots.fade_schedule.enabled` | false | Enables episode-based bot retirement and automatic market recalibration |
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

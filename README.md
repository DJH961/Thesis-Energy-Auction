# ETS MARL — Multi-Agent Reinforcement Learning for the EU Emissions Trading System

This repository implements a multi-agent reinforcement learning simulation of the EU Emissions Trading System (EU ETS). Agents represent energy companies that must balance electricity generation costs, green investment, and carbon compliance through auction bidding, secondary market trading, and technology investment decisions.

## Repository Structure

```
Thesis-Energy-Auction/
│
├── ets_marl ppo/                 # <-- ACTIVE PROJECT (use this)
│   ├── configs/
│   │   └── default.yaml          # All hyperparameters and environment settings
│   ├── src/
│   │   ├── environment/
│   │   │   ├── ets_environment.py # Two-phase environment (auction + secondary market)
│   │   │   ├── company.py         # Agent energy companies (mix, emissions, investment)
│   │   │   └── cap_schedule.py    # EU ETS cap trajectory + MSR logic
│   │   ├── agents/
│   │   │   ├── ppo_agent.py       # PPO agent with HAPPO support
│   │   │   ├── actor_critic.py    # Neural networks (AuctionPolicy, SecondaryPolicy, ValueNetwork)
│   │   │   ├── heuristic_policy.py# Rule-based policy for behavioral cloning warm-start
│   │   │   └── noise.py           # Exploration noise utilities
│   │   ├── auction/
│   │   │   └── market_clearing_ets.py  # Uniform-price auction clearing with holding limits
│   │   └── utils/
│   ├── scripts/
│   │   ├── train.py               # Main training loop (BC warm-start + PPO)
│   │   └── evaluate.py            # Evaluation script
│   ├── tests/                     # 66 tests (pytest)
│   ├── notebooks/
│   │   └── ets_marl_colab_HAPPO.ipynb  # Interactive notebook (works in VS Code + Colab)
│   ├── requirements.txt
│   └── main.py
│
├── ets_marl ppo copia 2/         # Old snapshot (pre-HAPPO changes) — kept for reference
├── ets_marl test/                # Test/scratch area
└── README.md                     # This file
```

> **Which folder do I use?** Always work from `ets_marl ppo/`. The `copia 2` folder is a historical snapshot and should not be modified.

## Branch: `HAPPO_compliant`

This branch simplifies the reward function and action space to be compatible with HAPPO (Heterogeneous-Agent Proximal Policy Optimisation). The goal is to remove artificial guardrails that interfere with multi-agent credit assignment and let real economic signals (the 100 EUR/t penalty + carry-forward) drive agent behavior.

### What Changed (vs. previous P14 version)

| Area | Before (P14) | After (HAPPO_compliant) |
|------|-------------|------------------------|
| **Bid price** | Markup multiplier on reference anchor: `action[0]` in [0.85, 1.25] scaled by blended reference price | Direct bid price: `action[0]` maps to [60, 500] EUR/t |
| **Sell gating** | `sell_coverage_floor: 1.05` blocked selling when coverage < 105% | Removed. Agents sell freely (only constraint: no short selling) |
| **Coverage penalty** | `coverage_weight: 1.0` added a bounded [0,1] penalty when holdings < obligation | Removed. The 100 EUR/t penalty + carry-forward is the signal |
| **Holding cost** | Convex banking cost: `rate * (ratio-1)^1.5` included in reward | `banking_holding_cost: 0.0` — kept as diagnostic only, not in reward |
| **Green shaping** | `shaping_weight_floor: 1.0` (never decays) | `shaping_weight_floor: 0.0` (decays fully to zero) |
| **Penalty norm** | `penalty_cost / 500.0` (100M EUR penalty = 0.2 signal) | `penalty_cost / 100.0` (100M EUR penalty = 1.0 signal) |
| **KL anchor decay** | 5000 episodes | 8000 episodes (BC-taught behavior persists longer) |
| **Phase 2 obs** | 5 extra dims (alloc, price, compliance_pos, shock, savings) | 7 extra dims (+coverage_ratio, +carry_forward_norm) |
| **Holding limit** | `max_agent_share: 0.25` in config but unverified | Verified: correctly enforced in `market_clearing_ets.py` |

### Reward Function

```
R_i = -cost_norm - emissions_intensity - penalty_norm + green_bonus
```

Four signals, no artificial guards:
- **cost_norm**: (auction + secondary + invest + ops + MAC - electricity revenue) / 1000
- **emissions_intensity**: penalisable emission factor / 0.82
- **penalty_norm**: penalty_cost / 100.0 (strong signal for non-compliance)
- **green_bonus**: diminishing-returns bonus for green investment (decays to zero)

## How It Works

### Environment: Two-Phase Decision Making

Each year of the simulation has two phases:

**Phase 1 — Auction + Investment** (6D action):
1. `bid_price`: How much to bid for carbon allowances (60–500 EUR/t)
2. `qty_multiplier`: Coverage ratio of estimated compliance need to bid for (0.3–1.3x)
3. `invest_frac`: Fraction of output to shift from fossil to renewable (0–10%)
4. `tech_logits[3]`: Which green technology to invest in (onshore/offshore/solar)

**Phase 2 — Secondary Market** (2D action):
1. `price_multiplier`: Price relative to clearing price for secondary trade (0.8–1.3x)
2. `quantity`: How much to buy (+) or sell (-) on the secondary market

### Agent Archetypes (8 agents)

| Agents | Archetype | Initial Green | Cost Weight | Green Weight |
|--------|-----------|---------------|-------------|--------------|
| A1, A2 | Coal-heavy | 20% | 0.90 | 0.10 |
| A3, A4 | Gas-dominant | 40% | 0.75 | 0.25 |
| A5, A6 | Mixed transitioner | 70% | 0.50 | 0.50 |
| A7, A8 | Near-green leader | 90% | 0.25 | 0.75 |

### Training Pipeline

1. **Behavioral Cloning** (300 episodes): Pre-train policies with a rule-based heuristic
2. **Critic Warmup** (100 episodes): Train value network only (no actor gradients)
3. **HAPPO Training** (30,000 episodes): Sequential PPO updates with centralized critics
   - KL anchor prevents drift from BC-taught behavior (decays over 8000 episodes)
   - Epsilon-greedy exploration (25% → 5% over 7000 episodes)
   - Historical Policy Pool (HPP) maintains behavioral diversity
   - Curriculum learning: 4yr → 20yr episodes over 3000 episodes

## Quick Start

### Local (VS Code)

```bash
cd "ets_marl ppo"
pip install -r requirements.txt
python -m pytest tests/ -q          # Run tests (should pass 66/66)
python main.py                       # Train with default config
```

Or use the notebook: open `notebooks/ets_marl_colab_HAPPO.ipynb` in VS Code and run cells.

### Google Colab

Open the notebook `ets_marl_colab_HAPPO.ipynb` in Colab. Cell 1 auto-detects the Colab environment, clones the repo, and sets up the project.

### Running Tests

```bash
cd "ets_marl ppo"
python -m pytest tests/ -v
```

All 66 tests should pass. Key test files:
- `test_environment.py` — Environment mechanics, obs dimensions, compliance
- `test_market_clearing.py` — Auction clearing, holding limits, reserve price
- `test_mappo.py` — MAPPO/HAPPO centralized critic, global state construction
- `test_clipped_gaussian.py` — Action space clipping behavior

## Key Configuration (`configs/default.yaml`)

```yaml
# Core settings
simulation:
  n_years: 20
  n_episodes: 30000

# ETS cap trajectory
ets:
  cap_year_0: 23.5          # Mt (slight surplus at start)
  reserve_price: 60.0       # Auction floor price

# Auction
auction:
  price_min: 60.0            # Direct bid lower bound (EUR/t)
  price_max: 500.0           # Direct bid upper bound (EUR/t)
  max_agent_share: 0.25      # California-style holding limit

# Penalty
penalty:
  rate: 100.0                # EUR/t non-compliance penalty
  carry_forward: true        # Shortfall rolls to next year

# PPO / HAPPO
ppo:
  happo: true
  centralized_critic: true
  kl_anchor_decay_episodes: 8000
```

See `configs/default.yaml` for the complete configuration with inline documentation.

## Observation Space

**Phase 1** (21 base + 14 opponent = 35 dims with 8 agents):
- Time, cap, price signals (MA3, expected), technology mix (5D), emissions, risk, construction queue, carry-forward, secondary market signals, opponent modeling (2 dims per opponent)

**Phase 2** (Phase 1 + 7 = 42 dims):
- Allocation, clearing price, net compliance position, emission shock, auction savings, **coverage ratio**, **carry-forward normalized**

## Validation Checklist

After training ~200 episodes, verify:
- Agents that are short (coverage_ratio < 1.0) should buy on secondary market, not sell
- If agents still sell while short after 5000 episodes, investigate reward normalization clipping
- Penalty signal should produce reward magnitude ~1.0 for 100M EUR penalty (check with `penalty_cost / 100.0`)

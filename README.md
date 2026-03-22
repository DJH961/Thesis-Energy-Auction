# ETS MARL â€” Multi-Agent Reinforcement Learning for the EU Emissions Trading System

This repository implements a multi-agent reinforcement learning simulation of the EU Emissions Trading System (EU ETS). Agents represent energy companies that must balance electricity generation costs, green investment, and carbon compliance through auction bidding, secondary market trading, and technology investment decisions.

## Repository Structure

```
Thesis-Energy-Auction/
â”‚
â”œâ”€â”€ ets_marl_happo/                 # <-- ACTIVE PROJECT (use this)
â”‚   â”œâ”€â”€ configs/
â”‚   â”‚   â””â”€â”€ default.yaml          # All hyperparameters and environment settings
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ environment/
â”‚   â”‚   â”‚   â”œâ”€â”€ ets_environment.py # Two-phase environment (auction + secondary market)
â”‚   â”‚   â”‚   â”œâ”€â”€ company.py         # Agent energy companies (mix, emissions, investment)
â”‚   â”‚   â”‚   â””â”€â”€ cap_schedule.py    # EU ETS cap trajectory + MSR logic
â”‚   â”‚   â”œâ”€â”€ agents/
â”‚   â”‚   â”‚   â”œâ”€â”€ ppo_agent.py       # PPO agent with HAPPO support
â”‚   â”‚   â”‚   â”œâ”€â”€ actor_critic.py    # Neural networks (AuctionPolicy, SecondaryPolicy, ValueNetwork)
â”‚   â”‚   â”‚   â”œâ”€â”€ heuristic_policy.py# Rule-based policy for behavioral cloning warm-start
â”‚   â”‚   â”‚   â””â”€â”€ noise.py           # Exploration noise utilities
â”‚   â”‚   â”œâ”€â”€ auction/
â”‚   â”‚   â”‚   â””â”€â”€ market_clearing_ets.py  # Uniform-price auction clearing with holding limits
â”‚   â”‚   â””â”€â”€ utils/
â”‚   â”œâ”€â”€ scripts/
â”‚   â”‚   â”œâ”€â”€ train.py               # Main training loop (BC warm-start + PPO)
â”‚   â”‚   â””â”€â”€ evaluate.py            # Evaluation script
â”‚   â”œâ”€â”€ tests/                     # 68 tests (pytest)
â”‚   â”œâ”€â”€ notebooks/
â”‚   â”‚   â””â”€â”€ ets_marl_colab_HAPPO.ipynb  # Interactive notebook (works in VS Code + Colab)
â”‚   â”œâ”€â”€ requirements.txt
â”‚   â””â”€â”€ main.py
â”‚
â”œâ”€â”€ ets_marl_happo copia 2/         # Old snapshot (pre-HAPPO changes) â€” kept for reference
â”œâ”€â”€ ets_marl test/                # Test/scratch area
â””â”€â”€ README.md                     # This file
```

> **Which folder do I use?** Always work from `ets_marl_happo/`. The `copia 2` folder is a historical snapshot and should not be modified.

## Branch: `HAPPO_compliant`

This branch simplifies the reward function and action space to be compatible with HAPPO (Heterogeneous-Agent Proximal Policy Optimisation). The goal is to remove artificial guardrails that interfere with multi-agent credit assignment and let real economic signals (the 100 EUR/t penalty + carry-forward) drive agent behavior.

### What Changed (vs. previous P14 version)

| Area | Before (P14) | After (HAPPO_compliant) |
|------|-------------|------------------------|
| **Bid price** | Markup multiplier on reference anchor: `action[0]` in [0.85, 1.25] scaled by blended reference price | Direct bid price: `action[0]` maps to [60, 500] EUR/t |
| **Sell gating** | `sell_coverage_floor: 1.05` blocked selling when coverage < 105% | Removed. Agents sell freely (only constraint: no short selling) |
| **Coverage penalty** | `coverage_weight: 1.0` added a bounded [0,1] penalty when holdings < obligation | Removed. The 100 EUR/t penalty + carry-forward is the signal |
| **Holding cost** | Convex banking cost: `rate * (ratio-1)^1.5` included in reward | `banking_holding_cost: 0.0` â€” kept as diagnostic only, not in reward |
| **Green shaping** | `shaping_weight_floor: 1.0` (never decays) | `shaping_weight_floor: 0.0` (decays fully to zero) |
| **Penalty norm** | `penalty_cost / 500.0` (100M EUR penalty = 0.2 signal) | `penalty_cost / 100.0` (100M EUR penalty = 1.0 signal) |
| **KL anchor decay** | 5000 episodes | 8000 episodes (BC-taught behavior persists longer) |
| **Phase 2 obs** | 5 extra dims (alloc, price, compliance_pos, shock, savings) | 7 extra dims (+coverage_ratio, +carry_forward_norm) |
| **Holding limit** | `max_agent_share: 0.25` in config but unverified | `max_agent_share: 1.0` â€” removed; agents compete freely |
| **Reserve price** | Static `reserve_price: 60.0` = `price_min` | Dynamic: `max(5.0, 0.80 Ã— MA3)` with initial fallback of 50.0 |
| **price_min** | 60.0 (= reserve_price) | 5.0 (low floor; dynamic reserve provides effective floor) |
| **Opponent modeling** | 2D per opponent (avg_bid/200, green_frac) | 5D per opponent (emissions, carry_forward, green_frac, fossil_frac, queue_total) |
| **Phase 1 obs** | 21 base + 14 opponent = 35D | 22 base + 35 opponent = 57D (+TNAC proxy at [21]) |
| **Phase 2 obs** | Phase 1 + 7 = 42D | Phase 1 + 7 = 64D |
| **Auction stats** | Basic (clearing_price, demand, allocated) | +HHI, +max_agent_share_actual |
| **Cancel under-subscribed** | Always disabled | Enabled after episode 1000 (schedule) |

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

**Phase 1 â€” Auction + Investment** (6D action):
1. `bid_price`: How much to bid for carbon allowances (5â€“500 EUR/t, effective floor set by dynamic reserve)
2. `qty_multiplier`: Coverage ratio of estimated compliance need to bid for (0.3â€“1.3x)
3. `invest_frac`: Fraction of output to shift from fossil to renewable (0â€“10%)
4. `tech_logits[3]`: Which green technology to invest in (onshore/offshore/solar)

**Phase 2 â€” Secondary Market** (2D action):
1. `price_multiplier`: Price relative to clearing price for secondary trade (0.8â€“1.3x)
2. `quantity`: How much to buy (+) or sell (-) on the secondary market

### Agent Archetypes (8 agents â€” 4 archetypes Ã— 2 objectives)

Each archetype pair shares the same energy mix but differs in objective:
odd-numbered agents (A2, A4, A6, A8) are green-objective, even-numbered (A1, A3, A5, A7) are financial.

| Agent | Archetype | Objective | Initial Green | Cost Weight | Green Weight |
|-------|-----------|-----------|---------------|-------------|--------------|
| A1 | Coal-heavy | Financial | 20% | 0.75 | 0.25 |
| A2 | Coal-heavy | Green | 20% | 0.25 | 0.75 |
| A3 | Gas-dominant | Financial | 40% | 0.75 | 0.25 |
| A4 | Gas-dominant | Green | 40% | 0.25 | 0.75 |
| A5 | Transitioner | Financial | 70% | 0.75 | 0.25 |
| A6 | Transitioner | Green | 70% | 0.25 | 0.75 |
| A7 | Green-leader | Financial | 90% | 0.75 | 0.25 |
| A8 | Green-leader | Green | 90% | 0.25 | 0.75 |

### Training Pipeline

1. **Behavioral Cloning** (300 episodes): Pre-train policies with a rule-based heuristic
2. **Critic Warmup** (100 episodes): Train value network only (no actor gradients)
3. **HAPPO Training** (30,000 episodes): Sequential PPO updates with centralized critics
   - KL anchor prevents drift from BC-taught behavior (decays over 8000 episodes)
   - Epsilon-greedy exploration (25% â†’ 5% over 7000 episodes)
   - Historical Policy Pool (HPP) maintains behavioral diversity
   - Curriculum learning: 4yr â†’ 20yr episodes over 3000 episodes

## Quick Start

### Local (VS Code)

```bash
cd "ets_marl_happo"
pip install -r requirements.txt
python -m pytest tests/ -q          # Run tests (should pass 68/68)
python main.py                       # Train with default config
```

Or use the notebook: open `notebooks/ets_marl_colab_HAPPO.ipynb` in VS Code and run cells.

### Google Colab

Open the notebook `ets_marl_colab_HAPPO.ipynb` in Colab. Cell 1 auto-detects the Colab environment, clones the repo, and sets up the project.

### Running Tests

```bash
cd "ets_marl_happo"
python -m pytest tests/ -v
```

All 68 tests should pass. Key test files:
- `test_environment.py` â€” Environment mechanics, obs dimensions, compliance
- `test_market_clearing.py` â€” Auction clearing, holding limits, reserve price
- `test_mappo.py` â€” MAPPO/HAPPO centralized critic, global state construction
- `test_clipped_gaussian.py` â€” Action space clipping behavior

## Key Configuration (`configs/default.yaml`)

```yaml
# Core settings
simulation:
  n_years: 20
  n_episodes: 30000

# ETS cap trajectory + dynamic reserve
ets:
  cap_year_0: 23.5              # Mt (slight surplus at start)
  reserve_price: 5.0            # Absolute floor (EUR/t)
  reserve_price_mode: "dynamic" # max(abs_floor, discount Ã— MA3)
  reserve_discount: 0.80        # 80% of 3-year moving average
  reserve_initial: 50.0         # Fallback before MA3 history exists

# Auction
auction:
  price_min: 5.0               # Low floor â€” dynamic reserve provides effective floor
  price_max: 500.0             # Direct bid upper bound (EUR/t)
  max_agent_share: 1.0         # No holding limit (agents compete freely)
  cancel_under_subscribed_after: 1000  # Enable cancel-under-subscribed after ep 1000

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

**Phase 1** (22 base + 35 opponent = 57 dims with 8 agents):
- Time, cap, price signals (MA3, expected), technology mix (5D), emissions, risk, construction queue, carry-forward, secondary market signals, **TNAC proxy** (total banked / cap)
- Opponent modeling: 5D per opponent (emissions, carry_forward, green_frac, fossil_frac, queue_total)

**Phase 2** (Phase 1 + 7 = 64 dims):
- Allocation, clearing price, net compliance position, emission shock, auction savings, **coverage ratio**, **carry-forward normalized**

## Validation Checklist

After training ~200 episodes, verify:
- Agents that are short (coverage_ratio < 1.0) should buy on secondary market, not sell
- If agents still sell while short after 5000 episodes, investigate reward normalization clipping
- Penalty signal should produce reward magnitude ~1.0 for 100M EUR penalty (check with `penalty_cost / 100.0`)


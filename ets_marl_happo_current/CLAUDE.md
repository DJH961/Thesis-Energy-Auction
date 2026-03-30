# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ETS MARL HAPPO v5.5** — A microsimulation of the EU Emissions Trading System (ETS) where 8 PPO/HAPPO learning agents and 8 heuristic bot agents compete in a carbon allowance market over 70,000 training episodes of 12 simulated years each.

## Installation & Setup

```bash
# Recommended (tries uv first, falls back to pip):
bash setup.sh

# Manual:
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch 2.2.2, Gymnasium 1.2.3+.

## Common Commands

```bash
# Train (single seed):
python scripts/train.py --config configs/default.yaml --seed 42

# Evaluate trained agents:
python scripts/evaluate.py --config configs/default.yaml --checkpoint results/seed_42/

# Run all tests:
python -m pytest tests/ -v

# Run a single test file:
python -m pytest tests/test_market_clearing.py -v
```

Output lands in `results/seed_XX/` (logs, checkpoints, plots).

## Architecture

### Agents & Companies

16 companies total — 8 PPO learning agents (A1–A8) + 8 heuristic bots (B1–B8), grouped into 4 archetypes by energy mix (coal-heavy → gas-dominant → transitioner → green-leader). Two A-agents per archetype differ by reward weights: even-indexed = pure financial [1.0, 0.0], odd-indexed = balanced ESG [0.5, 0.5]. All produce 10 TWh/year but have different emission intensities.

### Episode Structure (12 years/episode)

Each year has two phases:
1. **Phase 1 (Auction + Investment):** Agents choose bid price, bid quantity, green investment fraction, technology
2. **Auction clears** via uniform-price sealed-bid mechanism
3. **Phase 2 (Secondary Market):** Agents trade at negotiated prices
4. **Compliance & Rewards:** Penalties for shortfall, carry-forward of obligations

### Learning Algorithm: HAPPO

- PPO with centralized critic (MAPPO paradigm), sequential per-agent updates (HAPPO)
- Pre-training via behavioral cloning from `heuristic_policy.py`
- ε-greedy exploration decays 25%→5% over training
- Historical policy pool for opponent diversity
- `scripts/train.py` handles the full loop: pretraining → episode loop → PPO updates → checkpointing

### ETS Market Mechanics

- **Cap trajectory:** ~4.3–4.4%/year LRF reduction mimicking real EU ETS
- **MSR (Market Stability Reserve):** Auto-adjusts auction volume based on total banked allowances (TNAC); too many → withdraw, too few → release
- **MAC fuel-switching:** At ~€65/tonne, temporary coal→gas operational dispatch
- **Investment queue:** Technology-specific delays (solar 2yr, onshore wind 4yr, offshore wind 7yr)
- **Terminal value rewards:** Banked allowances and in-queue projects valued at episode end to prevent horizon collapse

### Key Source Files

| File | Role |
|------|------|
| `src/environment/ets_environment.py` | Main simulation loop (phase 1 → auction → phase 2 → rewards) |
| `src/environment/company.py` | Per-company state: mix, budget, allowances, investment queue |
| `src/environment/cap_schedule.py` | Cap trajectory + MSR logic |
| `src/agents/ppo_agent.py` | PPO/HAPPO implementation with centralized critic |
| `src/agents/actor_critic.py` | Neural networks: AuctionPolicy, SecondaryPolicy, ValueNetwork |
| `src/agents/heuristic_policy.py` | Rule-based warm-start for behavioral cloning |
| `src/auction/market_clearing_ets.py` | Uniform-price auction clearing |
| `scripts/train.py` | Main training loop (~2000 lines) |
| `configs/default.yaml` | All hyperparameters (300+ lines) |

### Agent Actions (per year)

- **Phase 1:** `bid_price` (€5–500/t), `bid_quantity` (0.3–2.0× need), `invest_frac` (0–20%), `tech_choice` (0=onshore, 1=offshore, 2=solar)
- **Phase 2:** `secondary_price` (0.8–1.4× clearing price), `secondary_quantity` (±3.0 Mt buy/sell)

### Key Config Parameters (`configs/default.yaml`)

```yaml
simulation.n_episodes: 70000   # Training length
simulation.n_years: 12         # Per-episode horizon
companies.n_agents: 8          # PPO learning agents
companies.n_bot_agents: 8      # Heuristic bots
ppo.centralized_critic: true   # MAPPO
ppo.happo: true                # Sequential per-agent updates
ppo.hidden_size: 256
ppo.lr: 0.0002
ppo.episodes_per_update: 8
```

### Tests

9 test modules in `tests/`: market clearing, cap schedule, company state, environment integration, rewards, heuristic policy, MAPPO updates, clipped Gaussian actions, action anchors.

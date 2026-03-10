# EU ETS Carbon Market Simulation with MARL

A multi-agent reinforcement learning simulation of the EU Emissions Trading System (EU ETS),
modelling strategic bidding behaviour of energy companies in a uniform-price sealed-bid allowance auction.

## Overview

This project simulates 4 energy companies with heterogeneous green/fossil mixes competing
in an EU ETS-style carbon allowance auction. Each company is a DDPG agent that learns to
optimise its bidding strategy, secondary market trading, and green investment decisions
over a multi-year horizon.

The environment faithfully implements:
- **Uniform-price sealed-bid auction** (single-round, EU Auctioning Regulation)
- **Declining cap** via Linear Reduction Factor (LRF: 4.3% → 4.4%)
- **Market Stability Reserve (MSR)** adjusting auction volume based on TNAC
- **Banking** of unused allowances across years
- **Penalty** of €100/tCO₂ + mandatory make-up in following year
- **Green investment** with stochastic success, 2-year delay, and risk factor tied to know-how

## Repository Structure

```
ets_marl/
│
├── src/
│   ├── auction/
│   │   ├── __init__.py
│   │   └── market_clearing_ets.py      # Uniform-price buyer auction (fork of ckrk/bidding_learning)
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── ets_environment.py          # Main multi-year ETS environment (Gymnasium)
│   │   ├── cap_schedule.py             # LRF + MSR cap/auction volume logic
│   │   └── company.py                  # Company state: green%, banking, investments
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ddpg_agent.py               # DDPG agent (Actor-Critic, continuous action)
│   │   ├── actor_critic.py             # Neural network architectures
│   │   └── noise.py                    # Ornstein-Uhlenbeck exploration noise
│   │
│   └── utils/
│       ├── __init__.py
│       ├── replay_buffer.py            # Experience replay
│       └── logger.py                   # Training metrics and episode logging
│
├── tests/
│   ├── test_market_clearing.py         # Unit tests for auction clearing
│   ├── test_cap_schedule.py            # Unit tests for LRF + MSR
│   └── test_environment.py             # Integration tests for ETS environment
│
├── configs/
│   ├── default.yaml                    # Default hyperparameters
│   └── scenarios/
│       ├── baseline_2024.yaml          # EUA price ~65 €/t
│       └── low_price_2020.yaml         # EUA price ~25 €/t
│
├── scripts/
│   ├── train.py                        # Main training entry point
│   └── evaluate.py                     # Load trained agents and run evaluation
│
├── results/                            # Auto-generated: training logs, plots, checkpoints
│
├── docs/
│   └── design.md                       # Detailed design decisions and parameter choices
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Agents

| Agent | Fossil % | Green % | w_cost | w_green | Risk Factor (init) |
|-------|----------|---------|--------|---------|-------------------|
| A1    | 90%      | 10%     | 0.80   | 0.20    | 0.45              |
| A2    | 60%      | 40%     | 0.65   | 0.35    | 0.30              |
| A3    | 40%      | 60%     | 0.35   | 0.65    | 0.20              |
| A4    | 10%      | 90%     | 0.20   | 0.80    | 0.05              |

All companies produce **10 TWh/year**, giving comparable revenues and making
the strategic differences purely a function of energy mix and objectives.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Cap year 0 | 14.60 MtCO₂ | Scaled from EU ETS (proportional to 4-firm emissions) |
| LRF years 1–4 | 4.3% | EU ETS post-2023 reform |
| LRF years 5+ | 4.4% | EU ETS post-2023 reform |
| MSR upper threshold | 8.33 Mt | EU threshold 833M scaled |
| MSR lower threshold | 4.00 Mt | EU threshold 400M scaled |
| Emission factor (gas CC) | 0.37 tCO₂/MWh | IPCC AR5 Annex III median |
| Penalty | €100/tCO₂ | EU ETS Directive Art. 16 |
| Investment delay | 2 years | Permitting + construction lag |
| Max Δgreen%/year | 3 p.p. | Physical-organisational constraint |
| EUA price (baseline) | €65/t | EU Carbon Market Report 2025 |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/ets_marl.git
cd ets_marl
pip install -r requirements.txt

# 2. Run training with default config
python scripts/train.py --config configs/default.yaml --seed 42

# 3. Run with multiple seeds (for robustness)
for seed in 42 123 456; do
    python scripts/train.py --config configs/default.yaml --seed $seed
done

# 4. Evaluate trained agents
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt
```

## Reproducibility

All experiments are seeded. To reproduce the baseline results:

```bash
python scripts/train.py --config configs/baseline_2024.yaml --seed 42
```

Results (training logs, episode statistics, model checkpoints) are saved to `results/`.

## Acknowledgements

The auction clearing mechanism is adapted from:
- [ckrk/bidding_learning](https://github.com/ckrk/bidding_learning) (MIT License, 2024)
  — uniform-price market clearing on price-quantity bids

The DDPG implementation follows:
- Di Persio, Garbelli, Giordano (2024) — *RL for Bidding Strategy Optimization in Day-Ahead Energy Market*
- Lillicrap et al. (2015) — *Continuous control with deep reinforcement learning*

ETS parameters sourced from:
- EU ETS Handbook (European Commission, 2017)
- EU Carbon Market Report 2025 (European Commission)
- IRENA Renewable Power Generation Costs 2023
- IPCC AR5 Annex III

## License

MIT License

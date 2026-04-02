# Thesis-Energy-Auction

## What Is This Project About?

This is the code behind a **Master's thesis** that investigates a deceptively simple question:

> **If energy companies are self-interested and strategic, how does the EU's carbon market (the EU ETS) actually perform — and can AI agents help us understand that?**

The **EU Emissions Trading System (EU ETS)** is Europe's main tool for fighting climate change. It works like this: the EU sets a cap on how much CO2 industry can emit each year, then companies must buy "emission allowances" at auction — one allowance = the right to emit one tonne of CO2. The cap shrinks over time, making pollution more expensive and nudging companies toward cleaner energy.

But here's the catch: real companies don't just passively buy allowances. They **strategize**. They hoard allowances, time their investments, and try to game the system. Understanding this strategic behaviour is hard because you can't run experiments on the real EU ETS.

### Our Solution: AI Agents Playing the Carbon Market

We built a **computer simulation** where AI agents (powered by reinforcement learning) play the role of energy companies. Each agent:

1. **Bids in carbon auctions** — deciding how much to pay for emission allowances
2. **Trades on a secondary market** — buying/selling allowances with other companies
3. **Invests in green energy** — deciding when to build wind farms or solar panels vs. keep burning fossil fuels

The agents learn through trial and error over thousands of simulated years, eventually developing sophisticated strategies — just like real companies would.

### Why Does This Matter?

- **For policymakers**: See how carbon market rules affect company behaviour before changing real policy
- **For researchers**: A testbed for studying strategic interaction in environmental markets
- **For the climate**: Better-designed carbon markets = faster, cheaper emissions reductions

## Repository Structure

This repo currently contains **three versions** of the project, reflecting how it evolved over time:

| Folder | What It Is | Status |
|--------|-----------|--------|
| [ets_marl_happo_current](ets_marl_happo_current) | **The main, current version (v7.1).** Uses PPO/HAPPO agents, 8 learning + 8 bot companies, hidden burn-in warm-start, dynamic emission-weighted cap/MSR calibration, EU ETS-style bid collateral costs, side-balanced tabula-rasa start-price exploration, optional green-finance budget/capex relief, and bot enhanced-noise/fade controls. This is where all active development happens. | **Active** |
| [ets_marl_legacy_ppo](ets_marl_legacy_ppo) | An earlier version that used PPO agents. Kept for reference only. | Archived |
| [ets_marl_legacy_test](ets_marl_legacy_test) | The original prototype using DDPG agents (4 companies, simpler setup). Kept for historical comparison. | Archived |

### Q-Learning Baseline

The active version includes a **tabular Q-learning baseline** for comparison with PPO/HAPPO. This provides a lower bound on what a simple RL approach can achieve in the same environment.

- **State space**: 5 features (time progress, carbon price, green fraction, compliance gap, carry-forward) discretized into 3 bins each = 243 states
- **Action space**: 6 predefined auction profiles (Conservative, Moderate, Aggressive, GreenPush, Financial, PanicBuy) x 4 secondary market profiles (Hold, Sell, Buy, AggressiveBuy)
- **Algorithm**: Standard Q-learning with epsilon-greedy exploration

Key files:
| File | Purpose |
|------|---------|
| `ets_marl_happo_current/src/agents/q_learning_agent.py` | StateDiscretizer, ActionProfileMapper, QLearningAgent |
| `ets_marl_happo_current/src/train_qlearning.py` | Training loop + greedy evaluation |
| `ets_marl_happo_current/configs/qlearning.yaml` | Q-learning hyperparameters (disables reward shaping) |
| `ets_marl_happo_current/src/analysis/qlearning_analysis.py` | Q-table heatmaps, strategy histograms, PPO comparison plots |
| `ets_marl_happo_current/notebooks/ets_marl - Q-Learning Baseline.ipynb` | Interactive training and analysis notebook |

Run the baseline:
```bash
cd ets_marl_happo_current
python src/train_qlearning.py --config configs/default.yaml --ql-config configs/qlearning.yaml --seed 42
```

### Where Should I Start?

**Go to [ets_marl_happo_current/](ets_marl_happo_current).** That folder has its own detailed README with setup instructions, architecture explanations, and how to run experiments.

The legacy folders are only useful if you want to see how the project evolved or reproduce earlier results. They are read-only references.

## Quick Glossary

If you're new to this domain, here are the key terms you'll encounter:

| Term | Meaning |
|------|---------|
| **EU ETS** | The EU Emissions Trading System — Europe's carbon market |
| **Allowance (EUA)** | A permit to emit 1 tonne of CO2. Companies must surrender enough allowances to cover their emissions each year. |
| **Cap** | The total number of allowances available. It shrinks each year to reduce total emissions. |
| **MARL** | Multi-Agent Reinforcement Learning — multiple AI agents learning simultaneously in the same environment |
| **PPO / HAPPO** | Proximal Policy Optimization / Heterogeneous-Agent PPO — the AI algorithms the agents use to learn |
| **DDPG** | Deep Deterministic Policy Gradient — an older AI algorithm used in the legacy versions |
| **MSR** | Market Stability Reserve — an EU mechanism that automatically adds/removes allowances to stabilize prices |
| **LRF** | Linear Reduction Factor — the annual percentage by which the cap decreases |
| **Banking** | Companies can save unused allowances for future years |

## License

MIT License

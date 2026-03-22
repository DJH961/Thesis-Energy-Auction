# ETS MARL — Legacy Test Version (DDPG, 4 Agents)

> **Status: Archived.** This was the original prototype. For the current version, go to [ets_marl_happo_current](../ets_marl_happo_current).

## What Is This?

This is the **first version** of the carbon market simulation, built to test whether the core idea works. It uses:

- **4 companies** (instead of 8 in the current version)
- **DDPG** (Deep Deterministic Policy Gradient) as the learning algorithm (the current version uses PPO, which turned out to work better)
- A **simpler energy model** — companies just have a "fossil %" and "green %" without the full 5-technology breakdown

Think of this as the **proof of concept** that showed the approach was viable before building the more sophisticated current version.

## How It Differs From the Current Version

| Feature | This Version (Legacy) | Current Version (HAPPO) |
|---------|----------------------|------------------------|
| Number of companies | 4 | 8 |
| Learning algorithm | DDPG | PPO/HAPPO |
| Technologies | 2 (fossil vs green) | 5 (coal, gas, onshore wind, offshore wind, solar) |
| Cap (starting) | 14.6 Mt | 23.5 Mt |
| Max bid price | €200 | €500 |
| MAC switching | No | Yes (coal-to-gas switching) |
| Electricity revenue | No | Yes |
| Carry-forward penalties | No | Yes |

## Project Structure

```
ets_marl_legacy_test/
│
├── src/
│   ├── environment/
│   │   ├── ets_environment.py    # Simulation loop
│   │   ├── company.py            # Company state
│   │   └── cap_schedule.py       # Cap + MSR logic
│   ├── agents/
│   │   ├── ddpg_agent.py         # DDPG learning algorithm
│   │   ├── actor_critic.py       # Neural networks
│   │   └── noise.py              # Ornstein-Uhlenbeck exploration noise
│   ├── auction/
│   │   └── market_clearing_ets.py
│   └── utils/
│       ├── logger.py
│       └── replay_buffer.py
│
├── tests/                        # Unit and integration tests
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── docs/
│   └── design.md
├── main.py
├── requirements.txt
└── pyproject.toml
```

## Can I Still Run It?

Yes, it should still work:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train.py --config configs/default.yaml --seed 42

# Run tests
python -m pytest tests/ -v
```

However, all new development and experiments should use the [current version](../ets_marl_happo_current) instead.

## Why Was It Replaced?

- **DDPG struggled with multi-agent stability** — agents' policies would oscillate rather than converge
- **4 agents gave each company too much market power**, making results less realistic
- **The 2-technology model was too simple** to capture real-world fuel-switching dynamics (e.g., coal-to-gas switching when carbon prices rise)

These lessons directly shaped the design of the current version.

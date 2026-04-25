# ETS MARL — Legacy PPO Version

> **Status: Archived.** This was an intermediate version. For the current active version (v7.2), go to [ets_marl_happo_current](../ets_marl_happo_current).

## What Is This?

This is the **second version** of the carbon market simulation. It was the first version to use **PPO (Proximal Policy Optimization)** as the learning algorithm, replacing DDPG from the original prototype. It still used the simpler 4-agent, 2-technology setup but proved that PPO was the right algorithmic direction.

This version served as the bridge between the original DDPG prototype and the current 8-agent HAPPO version.

## How It Differs From the Other Versions

| Feature | Legacy Test (DDPG) | This Version (Legacy PPO) | Current (HAPPO) |
|---------|-------------------|--------------------------|-----------------|
| Learning algorithm | DDPG | **PPO** | PPO/HAPPO |
| Number of companies | 4 | 4 | 8 |
| Technologies | 2 (fossil vs green) | 2 (fossil vs green) | 5 (coal, gas, onshore, offshore, solar) |
| Heuristic baselines | No | **Yes** | Yes |
| Cap (starting) | 14.6 Mt | 14.6 Mt | 23.5 Mt |

## Project Structure

```
ets_marl_legacy_ppo/
│
├── src/
│   ├── environment/
│   │   ├── ets_environment.py    # Simulation loop
│   │   ├── company.py            # Company state
│   │   └── cap_schedule.py       # Cap + MSR logic
│   ├── agents/
│   │   ├── ppo_agent.py          # PPO learning algorithm
│   │   ├── actor_critic.py       # Neural networks
│   │   ├── heuristic_policy.py   # Rule-based baseline agents
│   │   └── noise.py              # Exploration noise
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
├── notebooks/
│   ├── ets_marl_colab.ipynb       # Training notebook
│   └── ets_marl_colab_HAPPO.ipynb # Early HAPPO experiments
├── docs/
│   └── design.md
├── main.py
├── requirements.txt
└── pyproject.toml
```

## Can I Still Run It?

Yes:

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml --seed 42
python -m pytest tests/ -v
```

However, all new development and experiments should use the [current version](../ets_marl_happo_current) instead.

## Why Was It Replaced?

- PPO worked well, but the **4-agent / 2-technology model was still too simple** to capture realistic market dynamics
- The current version scaled up to **8 agents and 5 technologies**, adding fuel-switching, electricity revenue, and carry-forward penalties
- The HAPPO variant allows **heterogeneous agent updates**, which is better suited for agents with very different energy mixes

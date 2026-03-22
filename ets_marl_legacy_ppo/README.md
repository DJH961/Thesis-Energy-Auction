# ETS MARL — Legacy PPO Version

> **Status: Archived.** This was an intermediate version. For the current version, go to [ets_marl_happo_current](../ets_marl_happo_current).

## What Is This?

This folder contains an **earlier snapshot** of the project from when PPO was first adopted as the learning algorithm (replacing DDPG from the original prototype). It exists purely as a historical reference point.

This version only contains a **notebook** — the full codebase was not preserved separately because the changes were carried forward into the current version.

## Contents

```
ets_marl_legacy_ppo/
└── notebooks/
    └── ets_marl_colab_HAPPO.ipynb   # Colab notebook from this era
```

## How It Fits In the Timeline

1. **Legacy Test** (`ets_marl_legacy_test/`) — First prototype, DDPG, 4 agents, 2 technologies
2. **Legacy PPO** (`ets_marl_legacy_ppo/`) — **This version.** Switched to PPO, intermediate experiments
3. **Current HAPPO** (`ets_marl_happo_current/`) — Full version with 8 agents, 5 technologies, PPO/HAPPO, all policy improvements

## Should I Look at This?

Probably not. Unless you want to compare the notebook with the current Colab notebook to see how the approach evolved, there's nothing here that isn't done better in the current version.

Go to [ets_marl_happo_current](../ets_marl_happo_current) for the real thing.

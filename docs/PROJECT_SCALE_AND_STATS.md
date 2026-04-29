# ETS MARL — Project Scale & Stats

A presentation-ready snapshot of the size, depth, and complexity of the
**ETS MARL** thesis codebase: a multi-agent reinforcement-learning
simulation of the EU Emissions Trading System (v8.5.3).

> All figures measured directly from the repository at the head of
> branch `copilot/add-project-facts-and-figures-document`
> (April 2026). Numbers are rounded only where indicated.

---

## 1. The headline numbers

| | |
|---|---:|
| **Total tracked files** (excl. `.git`) | **211** |
| **Total lines across all source/config/docs/notebooks** | **89,300** |
| **Python files** | **128** |
| **Python LOC** (entire repo) | **44,684** |
| **Python LOC in `src/`** (production code) | **9,667** |
| **Python LOC in `tests/`** | **8,371** |
| **Python LOC in `scripts/`** (training & evaluation) | **2,859** |
| **Python LOC in `archive/`** (legacy versions) | **23,787** |
| **Markdown documentation** | **9,022 lines** across 22 files |
| **Jupyter notebooks** | **23** (`.ipynb`) — 545 cells, 283 of them code |
| **YAML config files** | **17** — 995 lines total |
| **Released versions** in `docs/changelog.md` | **13** (latest: **8.5.3**) |
| **TODO / FIXME / HACK comments** | **0** ✨ |

---

## 2. The single biggest file (the beating heart)

`src/environment/ets_environment.py` — the simulation core.

| | |
|---|---:|
| Lines of code | **3,311** |
| Top contender (`scripts/train.py`) | 2,450 LOC |
| Third place (`src/agents/ppo_agent.py`) | 1,490 LOC |

Together, the **top three files alone are 7,251 lines** of simulation
and learning logic.

---

## 3. Production code anatomy (`src/`)

| Metric | Count |
|---|---:|
| Modules | 5 sub-packages (`agents`, `auction`, `environment`, `analysis`, `utils`) |
| Classes | **16** |
| Top-level functions | **41** |
| Methods | **159** |
| **Total callables in `src/`** | **216** |
| Distinct top-level imported packages | **19** |

### Largest production files

| File | LOC |
|---|---:|
| `src/environment/ets_environment.py` | 3,311 |
| `src/agents/ppo_agent.py` | 1,490 |
| `src/environment/company.py` | 1,118 |
| `src/environment/cap_schedule.py` | 586 |
| `src/train_qlearning.py` | 519 |
| `src/analysis/qlearning_analysis.py` | 454 |
| `src/agents/q_learning_agent.py` | 440 |
| `src/agents/heuristic_policy.py` | 407 |
| `src/auction/market_clearing_ets.py` | 352 |
| `src/agents/actor_critic.py` | 261 |

---

## 4. Test suite

| | |
|---|---:|
| Test files | **22** |
| Test functions | **382** |
| Assertion calls | **587** |
| Total test LOC | **8,371** |
| Test-to-production LOC ratio | **0.87 : 1** |

Test files cover *every* major subsystem: market clearing, cap schedule,
PPO numerics (incl. dual-clip), HAPPO updates, bid-change limits,
phantom bidders, banking signal, green finance, opponent modelling,
MAC curve calibration, preflight validation, behavioral rewards, and
end-to-end smoke runs.

---

## 5. The simulation itself: scale at training time

A single training run with the **default config** (`configs/default.yaml`):

| Knob | Value |
|---|---:|
| Episodes | **120,000** |
| Years per episode | **12** |
| Decision phases per year | **2** (auction + secondary market) |
| Learning agents | **8** |
| Bots (default config) | 0 (smoke config: 8) |
| Continuous action dims per agent per year | **6 + 2 = 8** |
| Observation dim, phase 1 (8 agents) | **43 + 7×7 = 92** |
| Observation dim, phase 2 (8 agents) | **92 + 12 = 104** |

### Derived "crazy" figures per training run

| | |
|---|---:|
| Agent-years simulated | 120,000 × 12 × 8 = **11,520,000** |
| Agent-decisions made (both phases) | 11,520,000 × 2 = **23,040,000** |
| Continuous action scalars produced | 23,040,000 × 4 ≈ **92,160,000** † |
| Auction bids cleared | 120,000 × 12 = **1,440,000 auction rounds**, each with up to 9 bidders incl. the phantom intermediary |
| Secondary-market clearings | another **1,440,000** double-auction rounds |
| Compliance checks | 120,000 × 12 × 8 = **11,520,000** |

† Counting per-phase action vectors: 6D auction + 2D secondary = 8D per
agent-year, but each phase is an independent decision (so 4 scalars per
"decision" on average).

A single training run therefore makes **23 million policy decisions**
inside **2.88 million market clearings**.

---

## 6. The configuration surface

`configs/default.yaml` alone:

| | |
|---|---:|
| Lines | **481** |
| Top-level subsystems | **38** |
| Tunable parameters (`key: value` lines) | **330** |

Top-level subsystems (38):

```
version, simulation, device, technologies, ets, companies, investment,
risk, auction, trading, secondary, budget, green_finance, bots, penalty,
price, mac, electricity, reward, esg, opponent_modeling, opponent_obs,
agent_cycling, ppo, curriculum, uncertainty, construction_jitter,
warm_start, logging, pretrain, diagnostics, hpp, exploration,
cap_curriculum, budget_curriculum, phantom_bidder, urgency_scalars,
tabula_rasa
```

That's **38 distinct mechanism toggles** in a single config file.

---

## 7. Reward function complexity

The reward function is composed of **dozens** of named channels and
shaping terms inside `ets_environment.py`. A grep for `r_*` variables in
the environment alone yields **49 distinct reward-related identifiers**,
covering:

- `r_auction_bid` / `r_auction_invest` / `r_auction_volume`
- `r_clearing`, `r_bid`, `r_phantom`
- `r_bank`, `r_invest`, `r_budget`
- `r_end` (terminal bank + queue NPV)
- ESG / green-finance shaping with **compliance gating**
- Coverage-gap remediation-rate proxy with **per-episode EMA**
- Per-agent **lognormal urgency scalar** (privately sampled each episode)

The reward config block alone defines normalisation,
GAE std floor, opportunity-cost shaping, coverage-gap shaping, banking
signal (with imputed-bank value), terminal payoffs (bank + queue),
treasury value, and a `sec_proxy` smoother — each with their own
parameters.

---

## 8. Energy-system fidelity

5 generation technologies, each with **8** physical/economic parameters:

| Tech | Capex (€/kW) | CF | Emission factor (tCO₂/MWh) | Deploy delay (yr) | Op cost (€/MWh) |
|---|---:|---:|---:|---:|---:|
| Coal | 3,000 | 0.65 | 0.820 | 0 | 72.0 |
| Gas | 1,150 | 0.60 | 0.490 | 0 | 55.0 |
| Onshore wind | 1,350 | 0.35 | 0.011 | **4** | 17.0 |
| Offshore wind | 3,250 | 0.47 | 0.012 | **7** | 47.0 |
| Solar | 750 | 0.17 | 0.048 | 2 | 10.0 |

On top of which the simulation models:

- ETS **cap schedule** with two-phase **LRF** (4.3% / 4.4%), **MSR** with
  1-year TNAC lag, **cancellation mechanism**, and unsold-volume
  rollover
- **Marginal Abatement Cost** (MAC) curve with coal-to-gas
  fuel-switching at 48 €/tCO₂
- **Bid-change limit** anchored on price MA(3) ∨ fundamental anchor
- **Phantom bidder** representing financial intermediaries (~11% of
  expected demand)
- **Construction queue** with project-level deploy delays and queued
  capex
- **Carry-forward obligations**, **emergency loans**, and **budget
  overspend** penalties
- **Inflation** indexing of nominal vs real prices throughout

---

## 9. Learning machinery

Implemented from scratch in PyTorch:

| | |
|---|---:|
| Hidden size of every actor / critic | **256** |
| Actor architecture | LayerNorm → 2× FC → **conditional heads** (price → quantity → rest) |
| Algorithms supported | **PPO**, **HAPPO** (sequential MARL), **Q-learning baseline** |
| Numerical safeguards | dual-clip PPO, ±10 log-ratio clamp, KL early-stop, per-sub-head KL when split-invest-head is on |
| Pre-training options | behavioral cloning from heuristic, critic-only warmup |
| Auxiliary mechanisms | epsilon-greedy exploration decay, entropy decay, Historical Policy Pool (HPP), KL-anchor to BC policy, reward normaliser with EMA, cosine LR annealing |

---

## 10. Engineering footprint

| | |
|---|---:|
| Notebooks for experiments & analysis | **10** active (each ~63 cells) + **13** in `archive/` |
| Largest single notebook | **10.5 MB** (`Full Run & Analysis HIGHER BID RANGE.ipynb`) |
| Direct dependencies (requirements.txt) | 8 (numpy, pandas, torch, gymnasium, matplotlib, seaborn, pytest, pyyaml) |
| Total transitive packages locked in `uv.lock` | **48** |
| Documentation pages | 4 in `docs/`: BUGS_AND_ISSUES, changelog (**2,186 lines**), changes_v83_to_v85, design |
| Versions released (changelog) | **13** |

---

## 11. The "wow" one-liners (for slides)

- **3,311 lines** in a single environment file — and it's all one
  coherent simulator.
- **23 million** agent decisions across **2.88 million** market clearings
  in a *single* default training run.
- **587 assertions** across **382 tests** — almost as much test code as
  production code (**8.4k vs 9.7k LOC**).
- **38 independently tunable subsystems** in `default.yaml`, exposing
  **330 parameters** to the experimenter.
- **49 distinct reward-channel identifiers** combined into a single
  scalar reward — including a privately sampled lognormal **urgency
  scalar** per agent per episode.
- **13 released versions** documented across **2,186 lines** of
  changelog.
- **Zero** TODO / FIXME / HACK comments in the production codebase.
- **Five** generation technologies modelled with deploy delays up to
  **7 years**, sitting inside a 12-year episode — agents must commit
  capital before they know how the cap will evolve.
- The environment, the auction clearing engine, the PPO/HAPPO learner,
  the MAC curve, the MSR, the phantom bidder, the heuristic bot, and
  the Q-learning baseline are **all written from scratch** — no RL
  framework dependency beyond PyTorch and Gymnasium.

---

*Generated by direct measurement of the repository — no estimates.*

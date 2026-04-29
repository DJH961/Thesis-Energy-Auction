# ETS MARL — Project Scale & Stats

A presentation-ready snapshot of the size, depth, and complexity of the
**ETS MARL** thesis codebase: a multi-agent reinforcement-learning
simulation of the EU Emissions Trading System (v8.5.3).

> Numbers are measured directly from the repository. **The `archive/`
> directory (legacy code from prior versions) is excluded from every
> count below**, and notebook sizes are computed **without execution
> outputs** so that they reflect authored content only.

---

## 1. The headline numbers (current codebase only)

| | |
|---|---:|
| **Tracked source/config/test/doc files** (excl. `archive/`, notebooks, lockfile) | **67** |
| **Lines of authored content** (Python + YAML + Markdown + Toml + Txt, excl. `archive/`) | **26,048** |
| **Python files** (current code) | **47** |
| **Python LOC** (current code) | **20,897** |
| **Python LOC in `src/`** (production simulation + agents) | **9,667** |
| **Python LOC in `tests/`** | **8,371** |
| **Python LOC in `scripts/`** (training & evaluation) | **2,859** |
| **Markdown documentation** | **9,022 lines** across 22 files |
| **YAML config files** | **5** active configs — 995 lines total |
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
| Sub-packages | 5 (`agents`, `auction`, `environment`, `analysis`, `utils`) |
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

Test files cover *every* major active subsystem: market clearing, cap
schedule, PPO numerics (incl. dual-clip), HAPPO updates, bid-change
limits, banking signal, green finance, opponent modelling, MAC curve
calibration, preflight validation, behavioural rewards, and end-to-end
smoke runs.

---

## 5. The simulation itself: scale at training time

A single training run with the **default config** (`configs/default.yaml`):

| Knob | Value |
|---|---:|
| Episodes | **120,000** |
| Years per episode | **12** |
| Decision phases per year | **2** (auction + secondary market) |
| Learning agents | **8** |
| Continuous action dims per agent per year | **6 + 2 = 8** |
| Observation dim, phase 1 (8 agents) | **43 + 7×7 = 92** |
| Observation dim, phase 2 (8 agents) | **92 + 12 = 104** |

### Derived "crazy" figures per training run

| | |
|---|---:|
| Agent-years simulated | 120,000 × 12 × 8 = **11,520,000** |
| Agent-decisions made (both phases) | 11,520,000 × 2 = **23,040,000** |
| Auction rounds cleared | 120,000 × 12 = **1,440,000** uniform-price sealed-bid clearings |
| Secondary-market clearings | another **1,440,000** double-auction rounds |
| Compliance checks | 120,000 × 12 × 8 = **11,520,000** |

A single training run therefore makes **23 million policy decisions**
inside **2.88 million market clearings**.

---

## 6. The configuration surface

`configs/default.yaml` alone:

| | |
|---|---:|
| Lines | **481** |
| Top-level subsystems present | **38** |
| **Active subsystems** (after stripping `enabled: false` blocks) | **31** |
| Tunable parameters (`key: value` lines) | **330** |

The 31 active subsystems include the ETS cap schedule, MSR, MAC curve,
auction & secondary-market mechanics, HAPPO/PPO learner, ESG signal,
opponent modelling, urgency scalars, banking signal, warm-start
burn-in, HPP (Historical Policy Pool), exploration schedule,
construction jitter, and reward shaping — each with its own parameter
block.

---

## 7. Reward function complexity

The reward computation in `ets_environment.py` aggregates **dozens** of
named shaping channels into a single per-agent scalar reward. The
config block alone defines:

- Phase-wise auction / secondary reward normalisation with EMA
- GAE std floor (per-phase)
- Opportunity-cost shaping
- Coverage-gap shaping
- Banking signal (with imputed bank value, decaying)
- ESG signal with **compliance gating** (`(coverage_frac)²` multiplier)
- `sec_proxy` EMA smoother for the bid-head gap-penalty rate
- Per-agent **lognormal urgency scalar** (privately sampled each
  episode, σ = 0.30)
- Terminal payoffs: bank value, queue NPV, treasury value
- Compliance-gate blend with configurable threshold and width

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

- ETS **cap schedule** with two-phase **LRF** (4.3% / 4.4%), **MSR**
  with 1-year TNAC lag, **cancellation mechanism**, and unsold-volume
  rollover
- **Marginal Abatement Cost** (MAC) curve with coal-to-gas
  fuel-switching at 48 €/tCO₂
- **Bid-change limit** anchored on price MA(3) ∨ fundamental anchor
- **Construction queue** with project-level deploy delays and queued
  capex
- **Carry-forward obligations**, **emergency loans**, and **budget
  overspend** penalties
- **Construction jitter** (Poisson project counts, per-tech
  cancellation probabilities, capacity-factor noise)
- **Inflation** indexing of nominal vs real prices throughout

---

## 9. Learning machinery

Implemented from scratch in PyTorch:

| | |
|---|---:|
| Hidden size of every actor / critic | **256** |
| Actor architecture | LayerNorm → 2× FC → **conditional heads** (price → quantity → rest) |
| Algorithms supported | **PPO**, **HAPPO** (sequential MARL), **Q-learning baseline** |
| Numerical safeguards | dual-clip PPO (`c=3.0`), ±10 log-ratio clamp, KL early-stop, per-sub-head KL when split-invest-head is on |
| Auxiliary mechanisms | **HPP** (Historical Policy Pool), epsilon-greedy exploration with anchored decay, entropy decay, KL-anchor to BC policy, reward normaliser with EMA, cosine LR annealing, opponent modelling (7D per opponent) |

---

## 10. Engineering footprint

### Notebooks (authored content only — no execution outputs counted)

The two notebooks that constitute the actual experimental record:

| Notebook | Cells | Code cells | Code lines | Markdown lines | Source-only size |
|---|---:|---:|---:|---:|---:|
| `ets_marl - Full Run & Analysis.ipynb` | 63 | 32 | **2,588** | 174 | 141 KB |
| `ets_marl - Q-Learning Baseline.ipynb` | 37 | 24 | **476** | 35 | 25 KB |
| **Total** | **100** | **56** | **3,064** | **209** | **166 KB** |

### Versions / releases

The `docs/changelog.md` documents the recent **v8.x line in detail —
13 versions** (v8.1.0 → v8.5.3) across **2,186 lines** of release
notes.

But the changelog only captures the latest major series. Counting the
naming evidence elsewhere in the repo, the project has been through
**roughly 25+ release iterations**:

- **10 major version generations** visible as branches:
  `second_version`, `third-version`, `version_four`, `fifth_version`,
  `Version_six`, `Version_seven`, `Version_eight`, `Version_nine`,
  `Version_ten` (and the v1 baseline).
- **1 git tag** preserving an older release: `v7.2`.
- **11 HAPPO experiment branches** (`HAPPO_1` … `HAPPO_8`,
  `HAPPO_BEST`, `HAPPO_LSTM`, `HAPPO_compliant`) representing
  algorithm-level iterations.
- **13 documented sub-versions** within v8.x (the changelog).
- **3 superseded codebases** preserved under `archive/legacy/`
  (`ets_marl_happo_auction`, `ets_marl_legacy_ppo`,
  `ets_marl_legacy_test`) — evidence of three prior full rewrites.

So the **conservative best estimate is ≥ 25 development iterations
shipped to a "release" branch**, on top of countless feature branches.

### Other footprint metrics

| | |
|---|---:|
| Direct dependencies (`requirements.txt`) | 8 (numpy, pandas, torch, gymnasium, matplotlib, seaborn, pytest, pyyaml) |
| Total transitive packages locked in `uv.lock` | **48** |
| Documentation pages | 4 in `docs/`: BUGS_AND_ISSUES, changelog (**2,186 lines**), changes_v83_to_v85, design |

---

## 11. The "wow" one-liners (for slides)

- **3,311 lines** in a single environment file — and it's all one
  coherent simulator.
- **23 million** agent decisions across **2.88 million** market clearings
  in a *single* default training run.
- **587 assertions** across **382 tests** — almost as much test code as
  production code (**8.4k vs 9.7k LOC**).
- **31 active mechanism subsystems** in `default.yaml`, exposing **330
  tunable parameters** to the experimenter.
- A privately sampled **lognormal urgency scalar** (σ = 0.30) is drawn
  per agent per episode and folded into every reward computation.
- **≥ 25 development iterations** estimated from branches, tags, and
  the changelog — including three *full rewrites* preserved in
  `archive/legacy/`.
- **Zero** TODO / FIXME / HACK comments in the production codebase.
- **Five** generation technologies modelled with deploy delays up to
  **7 years**, sitting inside a 12-year episode — agents must commit
  capital before they know how the cap will evolve.
- The environment, the auction clearing engine, the PPO/HAPPO learner,
  the MAC curve, the MSR, the heuristic bot, and the Q-learning
  baseline are **all written from scratch** — no RL framework
  dependency beyond PyTorch and Gymnasium.

---

*Generated by direct measurement of the repository — no estimates
except where explicitly labelled (release count).*

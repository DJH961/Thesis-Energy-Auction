# ETS MARL (HAPPO/PPO)

This is the **active, main version** of the carbon market simulation. It uses modern reinforcement learning (PPO/HAPPO) to simulate energy companies competing in a simplified EU Emissions Trading System. The default profile runs **8 pure learning agents** with no heuristic bots.

Recent changes and release-specific details are tracked in `docs/changelog.md`.

## What Does This Code Do?

At a high level, this project:

1. **Creates a virtual carbon market** that mimics the real EU ETS (auctions, caps, penalties, banking, etc.)
2. **Places 8 AI-controlled (learning) companies** into that market, each with a different energy mix
3. **Lets the learning companies learn** over thousands of simulated episodes how to bid in auctions, trade allowances, and invest in clean energy
4. **Records everything** so we can analyze what strategies emerge and what they tell us about real carbon markets

An optional heuristic bot layer (8 fixed-policy agents, `smoke_100.yaml`) is available for validation and comparison.

## How the Simulation Works

### The Carbon Market

Each "episode" simulates **12 years** of a carbon market. Every year:

1. The government sets a **cap** — the total CO2 allowed. This cap **shrinks each year** (by about 4.3-4.4%) to push companies toward cleaner energy. The cap starts above total initial emissions by `ets.cap_overhead_pct` (default 0.10), and the actual year-0 cap is calibrated dynamically from the active participants' emissions unless `ets.cap_year_0_override` is set.
2. Companies participate in an **auction** where they bid for emission allowances (each allowance = right to emit 1 tonne of CO2).
3. The auction uses a **uniform price** — everyone pays the same price, which is the lowest winning bid. This is how the real EU ETS works.
4. Companies that don't have enough allowances to cover their emissions face a **penalty** (base **€138.75/t in 2026**, indexed from €132.06 in 2024; plus carry-forward obligations).
5. Companies can **bank** (save) unused allowances for future years.
6. A **Market Stability Reserve (MSR)** automatically adjusts the auction supply based on the total number of allowances in circulation (TNAC). The MSR implements the EU ETS post-2023 reform including:
   - **1-year TNAC lag**: MSR uses the *prior year's* TNAC, matching EU ETS Decision 2015/1814 (Art. 1(5)). Year 0 has no MSR intervention.
   - **Three-band withholding**: Uses legislative TNAC proportions lower:mid:upper = 400:833:1096. Above upper: withhold 24% of total TNAC. Between mid and upper: withhold TNAC − mid. Below mid: no intake.
   - **Threshold derivation**: `tnac_upper_ratio` is set in config; `tnac_mid_ratio` and `tnac_lower_ratio` default to `null` and are derived from `tnac_upper_ratio` so the legislative 400:833:1096 proportions are preserved at any simulation scale.
   - **Cancellation mechanism**: MSR holdings exceeding the previous year's auction volume are permanently cancelled.
   - **Smoothed price trigger**: Emergency release requires both an absolute threshold breach *and* a MA3 price spike > 2.5× the prior year's MA3 (disabled by default in `default.yaml`).

### Terminal Value Rewards

At the end of each episode (final year), agents receive terminal value bonuses that account for the beyond-horizon value of their decisions:

- **Bank terminal value**: Banked allowances are valued at the current clearing price, rewarding agents for prudent stockpiling.
- **Queue terminal value**: In-construction green projects are valued based on their discounted future carbon savings over a configurable payoff horizon (default: 5 years). This prevents the end-of-episode problem where agents stop investing because projects won't complete within the episode.

Both can be independently enabled/disabled via config flags (`reward.terminal_bank_value`, `reward.terminal_queue_value`).

### The Companies (Agents)

The default profile uses **8 learning agents** (PPO/HAPPO) and **no heuristic bots** (`n_bot_agents: 0`). The smoke/ablation config (`smoke_100.yaml`) includes 8 additional heuristic bot agents for validation.

**Learning Agents (A1-A8)** — organized into 4 archetypes (2 of each — one financially-motivated, one ESG-balanced):

| Archetype | Energy Mix | Reward Weights [w_cost, w_green] |
|-----------|-----------|----------------------------------|
| **Coal-heavy** (A1, A2) | ~55% fossil (25% coal, 30% gas), 45% green | A1: [1.0, 0.0] pure financial; A2: [0.5, 0.5] balanced ESG |
| **Gas-dominant** (A3, A4) | ~50% fossil (15% coal, 35% gas), 50% green | A3: [1.0, 0.0] pure financial; A4: [0.5, 0.5] balanced ESG |
| **Transitioner** (A5, A6) | ~35% fossil (10% coal, 25% gas), 65% green | A5: [1.0, 0.0] pure financial; A6: [0.5, 0.5] balanced ESG |
| **Green-leader** (A7, A8) | ~25% fossil (5% coal, 20% gas), 75% green | A7: [1.0, 0.0] pure financial; A8: [0.5, 0.5] balanced ESG |

Even-indexed agents (A1, A3, A5, A7) have pure financial reward weights [1.0, 0.0]; odd-indexed agents (A2, A4, A6, A8) have balanced ESG weights [0.5, 0.5].

**Heuristic Bot Agents (B1-B8)** — available in the smoke config, fixed-policy agents using rule-based strategies from `heuristic_policy.py`, mirroring all 8 learning agent archetypes:

| Bot | Energy Mix | Description |
|-----|-----------|-------------|
| **B1, B2** | ~80% fossil (40% coal, 40% gas) | Coal-heavy mirrors |
| **B3, B4** | ~60% fossil (15% coal, 45% gas) | Gas-dominant mirrors |
| **B5, B6** | ~30% fossil (5% coal, 25% gas) | Transitioner mirrors |
| **B7, B8** | ~10% fossil (0% coal, 10% gas) | Green-leader mirrors |

Bots use the same `Company` class and participate identically in auction clearing and secondary market matching. They are **not trained** — their actions come from the heuristic policy (fundamentals-based MAC→penalty bidding, NPV-gated investment, target-bank trajectory trading with absolute-price secondary market). Bots are indexed after learning agents (indices 8-15) and are excluded from PPO updates.

All companies produce **10 TWh/year** of electricity — the same output, but very different carbon footprints.

### What Each Agent Decides

Every year, each AI agent makes **6 decisions** (Phase 1) plus **2 more** (Phase 2):

**Phase 1 — Auction & Investment:**
- **Bid price**: How much to offer per allowance (€/tonne, range: 45-250)
- **Bid quantity**: Coverage multiplier on estimated annual need (0.5-2.0x)
- **Investment fraction**: What share of capacity to convert to green (0-20%, subject to capex throughput cap)
- **Technology choice**: Where to invest — onshore wind (4yr delay), offshore wind (7yr delay), or solar (2yr delay)

**Phase 2 — Secondary Market:**
- **Secondary price**: Absolute price in €/tonne (range: reserve price to 2× effective penalty rate)
- **Secondary quantity**: How many allowances to trade (positive = buy, negative = sell)

### How Agents Learn

The agents use **HAPPO (Heterogeneous-Agent PPO)**, a multi-agent reinforcement learning algorithm with centralized critics.

- Each agent has a centralized critic that sees the global state, enabling coordinated learning. The Phase-1 policy is split into a **bid sub-head** (price, qty multiplier) and an **investment sub-head** (invest fraction + tech logits) with separate advantage streams and separate value networks, so capital decisions don't bias the auction-bidding gradient.
- **Fundamental price anchor** (`src/utils/price_anchor.py`): Each episode, the auction policy price head is seeded to an economically grounded anchor (~67 EUR/t at yr0, rising to ~101 EUR/t by yr11) derived from MAC cost, cap scarcity, and penalty rate. This replaces behavioral cloning as the initialization mechanism.
- **WTP-anchored exploration** (`exploration.mode: "anchored"`): Epsilon-random auction prices are sampled from a Gaussian centered on the agent's willingness-to-pay anchor (~MAC + 0.5×(penalty − MAC)), keeping exploration economically grounded rather than flat-uniform across the full price range. A `"uniform"` mode (side-balanced around the WTP anchor) is also available as an ablation.
- **Epsilon-greedy exploration** decays from 25% to 2% over training, preventing policy collapse
- **Historical Policy Pool** maintains past policy snapshots for opponent diversity
- **Auto-scaled schedules**: warmup, exploration decay, and HPP timing scale automatically with `n_episodes`
- Hidden burn-in warm-start initializes banks, MSR reserve, and price history before visible year 0
- Behavioral cloning pretraining and KL-anchor regularization are **disabled by default** (config: `pretrain.enabled: false`, `ppo.kl_anchor_beta: 0.0`)
- Over 100,000 episodes, agents converge on sophisticated market strategies

The reward signal is a pure cost+ESG+penalty formulation — electricity revenue is **not** included in the reward gradient (it is logged separately but cannot be influenced by bidding strategy):

`R = w_cost × (−cost_norm) + w_green × esg_signal − penalty_norm + shaping + terminal_values`

- **cost_norm**: inflation-adjusted compliance + capital + soft-budget + loan costs, normalised by economic denominators (`anchor × need` for compliance, `annual_budget` for capital/soft buckets). The Phase-1 bid-head reward subtracts a fair-price baseline (`need × clearing_price`) so that buying exactly the compliance need at the clearing price is reward-neutral; over-buying is a small positive cost and under-buying is dominated by the coverage-gap penalty.
- **Coverage-gap penalty**: missing Mt are priced at the expected cost of remediation — `max(effective_penalty_rate, secondary_price_ema, fundamental_anchor)` per Mt, capped at `cap_mult × effective_penalty_rate` (default 1.5×) so single-year secondary spikes can't dominate. See `reward.sec_proxy` (toggle via `enabled`).
- **Banking signal**: imputes a cost basis on bank drawdowns at the clearing price, eliminating the zero-bid free-compliance exploit and rewarding good intertemporal timing (`reward.banking_signal`)
- **ESG signal** — saved-carbon-years: `esg_scale × (ef_ratio + speed_coef × Δgreen) × compliance_gate` (compliance gate is a smooth blend that approaches linear above the threshold and softer below)
- **penalty_norm**: non-compliance penalty normalised by `budget_real`, including a prospective scarcity-amplified expected-future-penalty term
- **Coverage gap shaping**: optional per-agent reward bonus for closing the gap between auction allocation and compliance need (decays to zero)
- **Opportunity cost shaping**: rewards agents in proportion to the cost premium paid on the secondary vs auction (decays to zero)
- **Green investment shaping**: bonus for increasing green fraction (decays over training), scaled by `(0.2 + w_green)`
- **Terminal bank value**: discounted hold — banked allowances valued at the terminal price discounted at the investment rate over `terminal_payoff_years`, normalised by `budget_real`
- **Terminal queue value**: in-construction projects valued as a present-value annuity over `terminal_asset_lifetime_years` (default 20 yr) at `investment.discount_rate`, discounted from each project's completion year
- **Treasury terminal value**: corporate treasury reserve (year-end operating surplus retained at `retention_fraction`, decaying at `decay_rate`) is valued at episode end via `treasury_terminal_value`

### Key Mechanisms

- **Inflation path**: Annual inflation is sampled from historical calibration **N(μ=2.0%, σ=1.5%)**, then applied economy-wide to nominal costs
- **MAC fuel-switching**: When carbon prices exceed €48/t (ICIS mid-range switching cost), companies automatically switch up to 20% of coal dispatch to gas (short-run operational change, not investment)
- **Electricity revenue**: Companies earn revenue from electricity sales, with carbon costs partially passed through to electricity prices (90%). Green generators benefit from the same revenue with lower carbon costs.
- **Unified financial envelope**: Each company has a single annual budget covering all spending (compliance + capex + MAC). With `budget.mode: revenue_based`, the budget is computed from `Company.compute_revenue()` (electricity revenue with carbon-cost passthrough) minus operating costs plus an archetype-specific debt headroom, with EMA smoothing.
- **Capex throughput cap**: Organizational constraint on annual construction spend (M€), modelling permitting pipeline capacity, EPC contractor access, and management bandwidth. Independent of the financial budget — a company can afford more investment than it can physically deliver.
- **Static reserve price**: Auction floor price at €45/t (just below MAC cost, prevents degenerate floor equilibrium)
- **Phantom bidder** (available, disabled by default): A synthetic financial intermediary participant can be enabled (`phantom_bidder.enabled: true`) to represent financial sector demand (~40% of EU ETS volume). When enabled, it bids at each primary auction with a LogNormal price anchored near 60% of the effective penalty and 15–35% of supply as quantity, consuming supply that would otherwise be available to compliance agents. See `docs/design.md §11`.
- **ESG compliance gate**: ESG bonus is multiplied by a smooth gate that approaches linear `coverage_frac` above the threshold (`compliance_gate_blend_threshold`) and softens below — non-compliant agents receive proportionally reduced ESG credit without a hard cliff under late-year scarcity.
- **Private urgency scalars**: Per-episode LogNormal scalar multiplied into each agent's effective penalty, creating heterogeneous compliance pressure and breaking symmetric equilibria.
- **Auction bid collateral**: overbids above clearing incur a real capital lock-up cost on awarded quantity (`auction.collateral.enabled`)
- **Collateral affordability guardrail**: if collateral lock-up is unaffordable, bids are clipped in two steps (quantity first, then price if needed) to preserve feasible participation
- **Bid change limit (BCL)**: Year-over-year bid price changes are capped at `auction.bid_change_limit.value` EUR/t (active in year 0 too). The BCL reference is `max(price_ma3, fundamental_anchor(year))` so it doesn't drift below the equilibrium price during low-price regimes. Clip signals are exposed as observation dimensions for gradient feedback.
- **Budget headroom observation**: Phase-1 observation reports current annual budget headroom (`1.0` fresh, `0.0` at limit, negative on overspend), plus separate dims for bid-price clip, soft-budget price clip, qty clip ratio, and invest-clip ratio.
- **Emergency loan system**: When a company faces auction default, an emergency loan covers the shortfall (up to `max_loan_fraction × annual_budget`) instead of immediate suspension. Loans carry interest (default 8%) plus a leverage premium, with annual repayment deducted at year start. While a loan is outstanding the effective capex throughput is squeezed to a configurable floor.
- **Corporate treasury reserve**: Year-end operating surplus is retained at `retention_fraction` (capped at `cap_multiple × annual_budget`, decaying at `decay_rate`) and provides a buffer that smooths year-to-year compliance shocks. Valued at episode end via `treasury_terminal_value`.
- **Budget hardening**: Tiered penalty regime — free spending up to 100% of budget, quadratic penalty in [100%, 115%], steep growth above. Investment hard gate scales down `invest_frac` if total projected spending would exceed the hard cap.
- **Heuristic loan-awareness**: Bots with emergency loans reduce auction quantity (−30%), investment (−50%), and secondary buy volume (−40%) proportional to loan pressure.
- **Carry-forward**: Non-compliance shortfall is added to next year's obligation (capped at 1.0×, modelling standard carry-forward)
- **Hidden burn-in warm-start**: A configurable pre-period (`warm_start.burnin_enabled`) seeds realistic bank holdings, MSR reserve, and MA3 history before year 0
- **Fundamentals-based heuristic**: Bot bidding uses MAC→penalty gradient (`mac_cost + urgency × (penalty - mac_cost)`), removing dependence on price moving average
- **Absolute-price secondary market**: Secondary prices are expressed in €/t (not as multipliers), clipped to [sec_price_min, 2× effective penalty rate]
- **Tabula-rasa is retired**: setting `tabula_rasa.enabled: true` raises an error; the block is kept in `default.yaml` purely as an ablation reference.
- **Coverage gap shaping** (`reward.coverage_gap_shaping`): optional per-agent reward bonus for closing the gap between allowance coverage and emissions; single unified block

## Project Structure

```
Thesis-Energy-Auction/
│
├── src/                          # All the simulation code
│   ├── environment/
│   │   ├── ets_environment.py    # The main simulation loop (auction → trade → invest → repeat)
│   │   ├── company.py            # Each company's state: portfolio, allowances, budget, etc.
│   │   ├── cap_schedule.py       # How the emission cap shrinks + MSR logic
│   │   ├── phantom_bidder.py     # Financial intermediary demand (disabled by default)
│   │   └── market_calibration.py # Cap/MSR calibration from active participant emissions
│   │
│   ├── agents/
│   │   ├── ppo_agent.py          # The PPO/HAPPO learning algorithm
│   │   ├── actor_critic.py       # The neural networks (the "brains")
│   │   ├── heuristic_policy.py   # Rule-based policy for bot agents
│   │   ├── q_learning_agent.py   # Q-learning baseline agent
│   │   └── noise.py              # Exploration noise utilities
│   │
│   ├── auction/
│   │   └── market_clearing_ets.py  # Runs the uniform-price auction
│   │
│   └── utils/
│       ├── price_anchor.py       # Fundamental price anchor (MAC + scarcity + penalty)
│       └── preflight.py          # Config validation run before training starts
│
├── configs/
│   ├── default.yaml              # Full 100k-episode training config (8 learning agents, 0 bots)
│   ├── smoke_100.yaml            # 100-episode smoke test (8 learning + 8 bots)
│   └── scenarios/                # Scenario variants (baseline_2024, low_price_2020)
│
├── scripts/
│   ├── train.py                  # Starts a training run
│   ├── evaluate.py               # Tests trained agents
│   └── evaluate_qlearning.py     # Evaluates the Q-learning baseline
│
├── tests/                        # Automated tests
│   ├── test_preflight.py         # Config validation that runs before training
│   ├── test_market_clearing.py   # Tests the auction works correctly
│   ├── test_cap_schedule.py      # Tests the cap schedule and MSR
│   ├── test_environment.py       # Tests the full simulation
│   ├── test_rewards.py           # Tests reward components and terminal values
│   ├── test_company.py           # Tests company state management
│   ├── test_heuristic.py         # Tests heuristic policy
│   ├── test_mappo.py             # Tests MAPPO/HAPPO integration
│   ├── test_clipped_gaussian.py  # Tests clipped Gaussian policy
│   ├── test_anchors.py           # Tests action anchor initialization and fundamental anchor injection
│   ├── test_price_anchor.py      # Tests compute_fundamental_anchor() calibration
│   ├── test_banking_signal.py    # Tests bank-drawdown imputation and timing P&L
│   ├── test_bid_change_limit.py  # Tests year-over-year bid change limit + obs dims
│   ├── test_bot_features.py      # Tests bot enhanced noise and fade schedule
│   ├── test_compliance_validation.py # Bot-only compliance smoke test
│   ├── test_green_finance.py     # Tests optional green-finance loan boost
│   ├── test_market_calibration.py    # Tests emission-weighted cap calibration
│   ├── test_phantom_bidder.py    # Tests phantom-bidder anchor independence
│   ├── test_qlearning_actions.py # Tests Q-learning baseline action mapping
│   ├── test_tabula_rasa.py       # Tests tabula-rasa retirement and exploration sampling
│   └── test_training_smoke.py    # End-to-end short training run smoke test
│
├── notebooks/
│   ├── ets_marl - Full Run & Analysis.ipynb   # Full training run and results analysis
│   ├── ets_marl - Q-Learning Baseline.ipynb   # Q-learning comparison baseline
│   └── ets_marl - Step-by-Step Debug.ipynb    # Step-by-step environment debugging
│
├── docs/
│   ├── design.md                 # Detailed design decisions
│   ├── changelog.md              # Version history and release notes
│   └── tnac_msr_simulation_guide.md  # MSR/TNAC calibration methodology
│
├── archive/                      # Legacy versions and deprecated scripts
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata
├── setup.sh                      # Linux/Mac setup script
└── install.bat                   # Windows setup script
```

## Setup & Installation

### Prerequisites

- **Python 3.11+** (3.12 also works)
- **pip** (Python's package manager — comes with Python)
- (Optional) **uv** for faster dependency installation

### Step-by-Step Setup

**On Windows:**
```bash
# 1. Open a terminal in this folder

# 2. (Recommended) Create a virtual environment so packages don't conflict:
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies:
pip install -r requirements.txt

# Or use the provided batch file:
install.bat
```

**On Linux/Mac:**
```bash
# 1. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies:
pip install -r requirements.txt

# Or use the provided script:
bash setup.sh
```

### Running Training

```bash
# Quick smoke test (100 episodes, fast):
python scripts/train.py --config configs/smoke_100.yaml --seed 42

# Full training run with default settings (100k episodes):
python scripts/train.py --config configs/default.yaml --seed 42

# Run with multiple seeds for statistical robustness, in parallel processes
# (each seed remains numerically identical to a sequential run):
python scripts/train.py --config configs/default.yaml --seed 1 2 3 4 --parallel-seeds 4
```

Training runs 100,000 episodes of 12-year simulations by default. Results are saved to a `results/` folder.

### Running Sweeps (multiple configs × seeds)

For ablations over multiple config variants (scarcity, MSR on/off, reserve
price, etc.) use the sweep launcher. Define a *sweep spec* once, listing the
overrides for each variant; the launcher resolves one full config per
variant, then runs every (variant × seed) job as an independent `train.py`
subprocess in a process pool.

```bash
# Validate the spec and inspect the job plan without launching anything:
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml --dry-run

# Run the full sweep:
python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml
```

Each variant writes to its own `<output_dir>/<variant>/` results directory,
and **its CSV / checkpoint filenames are tagged with the variant name**
(`training_log_<variant>_s<seed>.csv`, `year_log_<variant>_s<seed>.csv`,
`checkpoints_<variant>_s<seed>/`) — so files stay unique even if you copy
them all into one folder for analysis.

To keep the parent terminal readable while many jobs run concurrently,
each subprocess's full stdout+stderr is captured to
`<output_dir>/<variant>/run_<variant>_s<seed>.log`. The terminal only
shows short progress lines (`[START]` / `[LIVE]` heartbeat / `[DONE]`).

Each `[LIVE]` line is a one-line training summary per running job, parsed
from the per-episode CSV — episode progress, clearing-price trajectory,
mean reward, compliance rate, green-investment progress:

```
[sweep] [LIVE  reference s=1] Ep 1200/100000 (1.2%) | px 75→142 (μ128) | R̄ -3.2→-1.8 | comp 87% | green 31→44%
```

Heartbeats fire every `--heartbeat-interval` seconds (default 60); pass
`--quiet` to disable them.

Per-seed RNG and CSVs are bit-identical to a sequential
`train.py --config <variant>.yaml --seed S --run-tag <variant>`
invocation; only the process layout differs. See
`configs/sweeps/example_sweep.yaml` for the schema (also documented in
`src/utils/sweep.py`).

### Running Tests

```bash
# Run all tests to make sure everything works:
python -m pytest tests/ -v --tb=short
```

### Evaluating Trained Agents

```bash
python scripts/evaluate.py --checkpoint results/seed_42/best_model.pt
```

## Key Configuration (configs/default.yaml)

The most important settings you might want to change:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `simulation.n_episodes` | 100,000 | How many episodes to train for (more = better but slower) |
| `simulation.n_years` | 12 | How many years each episode simulates |
| `companies.n_agents` | 8 | Number of learning agents (PPO) |
| `companies.n_bot_agents` | 0 | Number of heuristic bot agents (0 = pure MARL; `smoke_100.yaml` uses 8) |
| `ets.cap_overhead_pct` | 0.10 | Year-0 cap overhead over total initial emissions |
| `ets.cap_year_0_override` | `null` | Optional hard override for year-0 cap |
| `ets.msr.tnac_upper_ratio` | 0.68 | TNAC upper-band threshold as fraction of cap_year_0; mid/lower derived to preserve 400:833:1096 |
| `auction.price_max` | 250 | Maximum bid price (€/tonne) |
| `auction.bid_change_limit.value` | 75 | Max year-over-year bid price change (€/t); year 0 unconstrained |
| `penalty.rate` | 138.75 | Fine per excess tonne of CO2 (€), base level at simulation year-0 (2026) |
| `pretrain.enabled` | false | Behavioral cloning warm-start from heuristic policy (disabled by default) |
| `ppo.kl_anchor_beta` | 0.0 | KL-anchor regularization toward BC policy (disabled by default) |
| `phantom_bidder.enabled` | false | Synthetic financial intermediary demand (disabled by default) |
| `urgency_scalars.enabled` | true | Per-agent LogNormal penalty multiplier (creates private heterogeneity) |
| `exploration.mode` | `anchored` | `anchored` Gaussian around WTP, or `uniform` side-balanced around WTP (ablation) |
| `exploration.epsilon_start` | 0.25 | Initial epsilon-greedy exploration rate |
| `exploration.epsilon_final` | 0.02 | Mature epsilon noise floor |
| `auction.collateral.enabled` | true | Enables EU ETS-style bid collateral opportunity-cost term on overbids |
| `auction.collateral.collateral_rate` | 0.15 | Annualized margin rate applied to locked collateral |
| `auction.collateral.collateral_fraction` | 0.20 | Fraction of notional bid value posted as collateral |
| `auction.collateral.hold_fraction` | 0.02 | Fraction of year collateral lock-up used in annualized model (~7 days) |
| `auction.collateral.min_qty_floor_frac` | 0.5 | Minimum coverage floor used by pre-auction collateral affordability clip |
| `green_finance.enabled` | false | Enables green-only loan/capex throughput boost during investment clipping |
| `bots.fade_schedule.enabled` | false | Enables episode-based bot retirement and automatic market recalibration |
| `penalty.inflation_rate` | 0.020 | Mean annual inflation for nominal indexing (μ) |
| `penalty.inflation_random_std` | 0.015 | Annual inflation standard deviation (σ), sampled with a normal distribution |
| `reward.banking_signal.enabled` | true | Imputes bank drawdowns at clearing price + timing P&L |
| `reward.opportunity_cost_shaping.enabled` | true | Decaying bonus for low secondary-vs-auction premium |
| `reward.coverage_gap_shaping.enabled` | true | Decaying signal when auction allocation falls short of need |
| `reward.terminal_bank_value` | true | Value banked allowances at episode end |
| `reward.terminal_queue_value` | true | Value in-construction projects at episode end |
| `reward.terminal_payoff_years` | 5 | Horizon for terminal queue value NPV calculation |
| `reward.treasury_terminal_value` | true | Value retained corporate treasury at episode end |
| `budget.mode` | `revenue_based` | Budget calculation method (`revenue_based` or `fixed`) |
| `budget.emergency_loan.enabled` | `true` | Emergency loans prevent auction defaults |
| `budget.emergency_loan.max_loan_fraction` | 0.15 | Max loan as fraction of annual budget |
| `budget.emergency_loan.interest_rate` | 0.08 | Annual interest rate on emergency loans |
| `budget.treasury_reserve.enabled` | true | Retain a fraction of year-end operating surplus as a buffer |
| `budget.treasury_reserve.retention_fraction` | 0.60 | Fraction of operating surplus retained each year |
| `budget.treasury_reserve.cap_multiple` | 1.5 | Treasury cap as multiple of `annual_budget` |
| `budget.hard_cap_fraction` | 1.15 | Hard cap on spending as fraction of budget |
| `budget.soft_zone_start` | 1.0 | Budget fraction where soft penalty begins |
| `budget.investment_hard_gate` | `true` | Scale investment if it would exceed the hard cap |

## Understanding the Output

After training, you'll find in `results/`:
- **Training logs**: Episode rewards, carbon prices, emissions over time
- **Model checkpoints**: Saved neural network weights (so you can resume or evaluate later)
- **Plots**: Visualizations of learning curves and market dynamics

## Acknowledgements

- Auction clearing adapted from [ckrk/bidding_learning](https://github.com/ckrk/bidding_learning) (MIT License)
- ETS parameters from EU ETS Handbook, EU Carbon Market Report 2025, IRENA 2024, IPCC AR5
- PPO/HAPPO algorithms based on Schulman et al. (2017) and Kuba et al. (2022)

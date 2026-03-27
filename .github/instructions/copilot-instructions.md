# EU ETS Multi-Agent Reinforcement Learning Simulation

## Project Context

This is a master's thesis (MSc Data Science, Copenhagen Business School) simulating the EU Emissions Trading System using multi-agent reinforcement learning. The goal is emergent agent behavior that informs real-world EU ETS policy. Research framing: Design Science Research (Peffers et al. 2007, Hevner et al. 2004).

The simulation features 8 heterogeneous learning agents (A1–A8, PPO/HAPPO-trained) and 4 heuristic bots (B9+) representing power company archetypes: coal-heavy, gas-dominant, transitioner, green-leader. Each archetype has one financially-motivated agent (w_cost=0.75/w_green=0.25) and one green-motivated agent (w_cost=0.25/w_green=0.75), forming a 4×2 factorial design.

Agents participate in carbon allowance auctions, secondary market trading, and green energy investment across 12-year episodes with annual timesteps.

## Locked Design Decisions — Do Not Suggest Alternatives

These decisions are final. Do not propose changes to any of the following:

- **Power sector only** with 100% auctioning (no free allocation)
- **Annual timesteps** — quarterly auction rounds were considered but not implemented
- **Agent-to-agent secondary market only** — NO external liquidity pool, NO market makers, NO financial intermediaries. This has been explicitly and permanently rejected. Never suggest it.
- **Penalty rate:** €100/t plus make-up obligation (matching real EU ETS; inflation-indexed via EICP/HICP)
- **No curriculum learning** — BC pretraining and critic warmup handle cold-start
- **Independent agent learning (not MAPPO)** — realism is the priority
- **Uniform-price sealed-bid auction format**
- **Heuristic bots as permanent market participants** (structural fix for thin-market pathologies)
- **Unsold allowances roll over** to next year's auction supply (matching EU ETS Auctioning Regulation), NOT routed to MSR
- **Static reserve price (~30 €/t)** — dynamic reserve requires liquid market price signal this simulation deliberately excludes
- **`cancel_under_subscribed` permanently disabled**
- **No per-agent 25% holding limit** — no basis in EU ETS rules; replaced with HHI diagnostic

## Hard Rules — The Environment Must Obey These

### Primary Auction
- Uniform-price sealed-bid: all winners pay the clearing price (lowest accepted bid)
- Reserve price must exist; bids below are rejected
- No artificial bid caps near the penalty rate — agents may bid above €100/t
- No forced minimum bid quantities
- No mechanism guaranteeing all agents receive allowances

### Cap and MSR
- Annual cap declines via Linear Reduction Factor (LRF), consistent with Phase 4
- Starting cap calibrated so initial emissions are at or slightly below cap (mild surplus, never structural deficit from year zero)
- MSR intake: when TNAC exceeds upper threshold, withhold percentage from next auction into MSR
- MSR release: when TNAC below lower threshold, release from MSR into next auction
- MSR cancellation: allowances in MSR above previous year's auction volume are permanently cancelled
- **Current MSR thresholds:** `tnac_upper` = 48.0 Mt, `tnac_lower` = 20.0 Mt (~1.2× mid-episode annual emissions), `withhold_rate` = 0.24 (matching real EU ETS post-2023 reform)

### Secondary Market
- Agents must not sell allowances they do not hold (no short selling)
- Transaction costs must apply
- Price bounds: MAC floor, 2× effective penalty rate ceiling; `spread_tolerance` = 0.12
- No price-fixing mechanism forcing secondary price to equal auction price

### Compliance
- End of each period: surrender allowances = verified emissions
- €100/t penalty + make-up obligation for shortfall (carry forward to next period)
- No additional multipliers, escalating penalties, or artificial punishment
- Banking surplus allowances across periods is allowed and must not be penalized
- Borrowing future-vintage allowances is NOT allowed
- Configurable cap on accumulated carry-forward is permitted for training stability only

### Green Investment and Energy Mix
- Technology-specific parameters: distinct CapEx, OpEx, emission factors, capacity factors
- All costs grounded in real citable data (IRENA 2024 LCOE, IEA WEO, IPCC AR5/AR6)
- Renewables: high upfront CapEx (€600–4,000/kW), near-zero marginal cost
- Fossil: lower CapEx, significant ongoing fuel + ETS costs
- Emission factors: fossil 400–910 gCO₂e/kWh, renewables 7–56 gCO₂e/kWh (lifecycle)
- Capacity factors: renewables 15–50%, fossil 50–85%
- **Construction delays are mandatory:** solar PV 1.5–3yr, onshore wind 4–7yr, offshore wind 6–12yr (may be compressed but ordering/relative magnitudes preserved)
- CapEx deducted at decision time; capacity enters queue, no emission reductions same period
- **No instant greening** — even solar requires minimum 1 timestep
- Dirtiest capacity (coal) displaced before cleaner (gas)
- Decommissioning costs: €100–200/kW (not free)
- Green fraction = share of annual MWh from low-emission sources, weighted by capacity factors
- No forced investment minimums, no artificial subsidies, no mandatory retirement schedule

### Agent Action Space
- Agents MUST be able to: bid in auction, buy/sell secondary market, invest in new capacity, fuel-switch, do nothing
- Agents MUST be allowed to: bid above penalty rate, choose non-compliance, hoard allowances, pursue any legal EU ETS strategy
- Block ONLY: selling unheld allowances, borrowing future vintages, physical/economic nonsense (negative investment, bids > 5× penalty)

### Revenue Model
- Electricity revenue must reflect carbon cost pass-through (higher carbon → higher electricity price → low-emission generators benefit)
- No artificial penalty for banking
- No signal making non-compliance structurally profitable long-run
- No penalty or cost without real-world analog

## Key Calibration Values

- `mac.coal_to_gas_cost`: 48.0 EUR/tCO2 (ICIS 2020–2025 front-month switching cost, average-efficiency plants)
- Target equilibrium price: ~80 €/t (real EU ETS ~€84/t late 2025, forecasts €100–145/t 2027–2030)
- `log_std_min` for PPO: adjusted per implementation plan
- Critic architecture: must be deeper than 256 units for ~680D centralized input

## Core Principles

1. **Structural fixes over parameter tuning.** Never add artificial constraints (bid caps, arbitrary floors, forced compliance). If agents misbehave, diagnose WHY and fix the structural cause. Good behavior must emerge naturally.
2. **All parameter choices must be cited and defensible.** No arbitrary values. Real EU ETS data, IRENA LCOE, ICIS switching costs, EU directive text are the standard.
3. **If agents exploit a loophole, verify whether it exists in the real EU ETS.** If yes → finding. If no → environment bug.
4. **Reward shaping must not reference endogenous market prices.** The `price_anchor_delta` AR(1) disaster proved this: self-reinforcing floor-bidding collapse.
5. **Terminal value design matters for 12-year episodes.** Banked allowances valued at final clearing price; queued investments at NPV of avoided carbon costs (discounted by gamma^years_remaining).
6. **Reward structure is general-sum, not zero-sum.** Agents coupled through shared market mechanisms but rewards computed independently. Compatible with HAPPO's monotonic improvement guarantee.

## Anti-Patterns — Never Do These

- Never propose an external liquidity pool or market maker mechanism
- Never propose quarterly auctions (considered and rejected)
- Never add artificial bid caps, compliance floors, or forced behavior
- Never use reward shaping that references endogenous prices (auction clearing, secondary market)
- Never compress emission factor gaps between fossil and renewables
- Never allow instant capacity deployment (skip construction delays)
- Never use WidthType.PERCENTAGE for docx tables
- Never add costs/penalties without a real EU ETS analog
- Never treat banking as pathological — it's a legitimate strategy

## Tech Stack

- **Language:** Python
- **RL framework:** PPO / HAPPO with centralized critic, BC pretraining with KL anchor penalty
- **Simulation:** Custom `ETSEnvironment`, `market_clearing_ets.py`, `heuristic_policy.py`, `company.py`
- **Config:** YAML (`default.yaml`, `configs/`)
- **Key files:** `src/agents/`, `src/environment/`, `src/training/`, `configs/`
- **Thesis documents:** Word documents via docx npm library / Node.js scripts

## Data Sources (Required for Citations)

- ICIS front-month gas switching cost data (MAC calibration)
- IRENA 2024 Renewable Power Generation Costs (LCOE, CapEx)
- EU ETS Auctioning Regulation (1031/2010, replaced by 2023/2830)
- EU Directive inflation-adjustment mechanism (EICP/HICP)
- EEX trading conditions
- IPCC AR5/AR6 (emission factors)
- IEA World Energy Outlook (technology costs, capacity factors)

## Diagnostic Indicators (Expected Emergent Behavior)

If the environment is correct, these should emerge without being forced:
- Prices trend upward as cap tightens (~80 €/t target), neither floor-collapse nor ceiling-hit
- Modest strategic banking (1–3 periods buffer)
- Progressive renewable shift accelerating in later years; high-fossil agents green faster
- Majority compliance; occasional non-compliance is realistic, persistent non-compliance is a bug
- Active secondary market trading; high-fossil agents buy, high-green agents sell

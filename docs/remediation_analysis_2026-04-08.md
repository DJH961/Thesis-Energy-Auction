# ETS MARL Remediation Analysis — 2026-04-08

## Problem Statement

After 60,000 episodes of training, all 8 learning agents converged to a **degenerate floor-price equilibrium**:

| Metric | Value | Problem |
|--------|-------|---------|
| Clearing price | 30.0 ± 0.0 EUR/t | Pinned at reserve floor — zero variance |
| Agent bid coverage | 0.34x | Agents barely participating in auction |
| Bot bid coverage | 1.1x | Bots absorb all supply |
| System reward | -37.8 ± 9.2 | Chronic non-compliance |
| debtSpiral warnings | 3,649 | Agents trapped in compliance debt |
| Agent bid price trend | 114 → 73 → 59 EUR/t | Actively learning to bid **lower** |

The agents were not broken — they were rationally discovering the Nash equilibrium of a market with structural oversupply.

---

## Root Cause Chain

Three independent gradient-killing mechanisms operate simultaneously:

### 1. Supply Oversaturation (Market Design)

```
cap_overhead_pct: 0.02 (surplus)
  + unsold_to_msr: false (snowball recirculation)
    → Chronic oversupply (~41 Mt supply vs ~35 Mt demand)
      → Clearing price pinned at reserve floor (30 EUR/t)
        → No price variance → no learning signal from price
```

With `unsold_to_msr: false`, unsold allowances recirculated into the next auction year, creating a self-reinforcing oversupply loop. This is also **unrealistic** — the real EU ETS routes unsold volume into the MSR where the cancellation mechanism destroys excess.

### 2. Baseline Cost Subtraction (Reward Engineering)

```python
# ets_environment.py:2348-2349 (BEFORE fix)
baseline_cost = company.compute_estimate_need() * clearing_price / budget_divisor
cost_norm_ex_penalty -= baseline_cost
```

When `clearing_price == reserve_price` (which it always was):
- `baseline_cost ≈ actual_auction_cost`
- Therefore `cost_norm ≈ 0`
- The agent sees **zero cost gradient** regardless of bid price

This is the single most critical RL failure. The per-agent RewardNormalizer (EMA α=0.01) already handles reward scale — the baseline was double-normalizing.

### 3. Bot Subsidy Effect (Market Structure)

Bots bid 1.1x coverage at ~110 EUR/t, absorbing most auction supply. They then become net sellers on the secondary market (bot sec_net: -3.9 Mt vs agent sec_net: +3.9 Mt at ~209 EUR/t). Agents learned they could:
- Skip the auction (bid 0.34x at floor)
- Patch compliance gaps via secondary market purchases from bots
- Defer remaining shortfall via carry-forward (cap: 1.0x, no interest)

This created a **free-rider equilibrium** with no incentive to learn competitive bidding.

---

## Changes Implemented

### Category 1: Market Design (ETS-Rational)

| # | Parameter | Before | After | File | Rationale |
|---|-----------|--------|-------|------|-----------|
| M1 | `unsold_to_msr` | `false` | `true` | default.yaml | Real EU ETS routes unsold to MSR. Breaks oversupply snowball |
| M2 | `cap_overhead_pct` | `0.02` | `-0.05` | default.yaml | Phase 4 EU ETS trajectory. Moderate scarcity (-5% below emissions) |
| M3 | `reserve_price` | `30.0` | `45.0` | default.yaml | Just below MAC (48 EUR/t). Prevents degenerate floor equilibrium |
| M3 | `price_min` | `30.0` | `45.0` | default.yaml | Consistent with reserve_price |
| M3 | `sec_price_min` | `30.0` | `45.0` | default.yaml | Consistent with reserve_price |
| M4 | `carry_forward_cap` | `1.0` | `0.5` | default.yaml | Prevents infinite deferral. 50% is realistic compliance flexibility |
| M5 | `qty_mult_low/high` | `0.3/2.0` | `0.7/1.5` | default.yaml | Compliance-centered. No rational installation bids 30% of need |
| M6 | `n_bot_agents` | `8` | `0` | default.yaml | See detailed argument below |

### Category 2: Reward Engineering (RL-Critical)

| # | Change | File | Rationale |
|---|--------|------|-----------|
| R1 | Remove baseline cost subtraction | ets_environment.py | Gradient killer. Set `baseline_cost = 0.0` in both `step_auction` and `_compute_rewards` |
| R2 | `clip_max: 10.0` | default.yaml | Already symmetric (was 10.0). Confirmed |
| R3 | `shaping_weight_floor: 0.10` | default.yaml | 10% residual keeps green/efficiency signals alive |
| R4 | S_financial clamp bugfix | ets_environment.py | Added `min(1.0, ...)` to prevent S_financial > 1.0 on refunds |

### Category 3: Hyperparameter Tuning

| # | Parameter | Before | After | Rationale |
|---|-----------|--------|-------|-----------|
| H1 | `critic_warmup_frac` | `0.10` | `0.03` | 10% warmup anchors critic to degenerate floor-price values |
| H2 | `shaping_decay_frac` | `0.33` | `0.60` | Extend guiding signals through 60% of training |
| H3 | `epsilon_start` | `0.50` | `0.30` | Less initial noise flooding value function |
| H3 | `epsilon_decay_frac` | `0.50` | `0.80` | Longer exploration window |
| H4 | `initial_bank_fraction` | `0.10` | `0.05` | Less initial buffer → more bidding urgency |
| H4 | `bank_seed_min/max` | `0.05/0.15` | `0.03/0.08` | Consistent with reduced bank fraction |

---

## Why Remove Bots

The bots create three pathologies that block learning:

### 1. Free-Rider Equilibrium
Bots provide guaranteed secondary market liquidity. Agents learn to rely on bot surplus instead of competing at auction. With 16 participants, each agent controls ~6% of the market — too small for a strong causal link between actions and outcomes. With 8 agents, each controls ~12.5%.

### 2. Critic Contamination
The centralized HAPPO critic sees all agents' states. Half (bots) follow a fixed policy. The critic spends capacity modeling bot behavior rather than learning meaningful value estimates for learning agents.

### 3. Artificial Demand Floor
Bots always bid ~1.1x at fundamentals-based prices (~22 Mt guaranteed demand). This makes agent participation optional. Without bots, supply-demand balance depends entirely on agent behavior.

**Mitigation**: The cap auto-calibrates via `compute_market_params()` based on active participants. Removing bots automatically rescales the market to 8 agents with proportional cap/MSR.

---

## What Was NOT Changed (and why)

| Decision | Reason |
|----------|--------|
| No carry-forward interest | Compounds exponentially → death spirals (Chapter 2 finding) |
| No curriculum learning | 12-year episodes are fine; problem was market design |
| No network architecture changes | 256/512 hidden dims adequate for this obs/action space |
| No MAC/electricity/inflation changes | Working as designed |
| No budget changes | Revenue-based budget is correctly calibrated |

---

## Expected Outcomes

With these changes, the market should exhibit:

1. **Real clearing price above floor**: -5% cap overhead + unsold→MSR creates moderate scarcity. Agents must compete.
2. **Meaningful cost gradients**: Without baseline subtraction, agents see actual cost differences between bid strategies.
3. **Self-referential learning**: Without bots, the market is purely agent-driven. Each agent's actions directly affect clearing price.
4. **Compliance urgency**: 0.5x carry-forward cap + tighter qty bounds ([0.7, 1.5]) prevent indefinite deferral.
5. **Sustained exploration**: 30% → 5% epsilon over 80% of training prevents premature convergence.

---

## Files Modified

| File | Changes |
|------|---------|
| `configs/default.yaml` | 12 parameter changes (M1-M6, R3, H1-H4) |
| `src/environment/ets_environment.py` | Baseline removal (R1, 2 locations) + S_financial clamp (R4) |
| `tests/test_environment.py` | Rollover test: explicit `unsold_to_msr=False` override |
| `tests/test_market_calibration.py` | 3 tests: explicit `n_bot_agents=8` for bot-dependent tests |
| `tests/test_bot_features.py` | `_base_cfg()`: explicit `n_bot_agents=8` |
| `tests/test_rewards.py` | Shaping floor assertion relaxed to ≤ 0.20 |
| `tests/test_tabula_rasa.py` | Schedule fractions updated: (0.03, 0.60, 0.80) |

**All 264 tests pass.**

---

## Run Configuration (v2 — after Chapter 3 fix)

```
n_episodes: 100,000 (or 600,000 for long run)
Critic warmup: 3% of n_episodes
Epsilon: 0.30 → 0.05 over 80% of training
Shaping: decays to 10% floor over 60% of training
Entropy: 0.08 → 0.025 over 95% of training
Cap overhead: -10% (guarantees over-subscription)
Qty multiplier: [0.85, 1.5] (compliance-centered, min > supply)
Coverage shaping: enabled (weight=0.5, decays by 30%)
Carry-forward: 0.5x cap, no interest
Bots: 0 (pure MARL)
Reserve price: 45 EUR/t
```

---

## Chapter 3: Post-Run Diagnosis (71k episodes, 2026-04-08)

### Observed Failure

The 71k-episode run with Chapter 1+2 changes converged to a **new floor-price equilibrium at 45 EUR/t** (the raised reserve price) instead of discovering competitive pricing above floor.

| Metric | Value | Problem |
|--------|-------|---------|
| Clearing price (ep 70k) | 45.0 ± 0.0 EUR/t | Pinned at new reserve floor |
| Agent coverage (early) | 0.7–0.9x | Under-subscribing the auction |
| Two-class split | Green (A5-A8) overbid, Dirty (A1-A4) underbid | Coordination failure |
| Secondary market | Green sell at ~280 EUR/t, Dirty buy | Free-rider dynamic persists |

### Root Cause: Structural Under-Subscription

The Chapter 1+2 parameter combination (`cap_overhead=-0.05`, `qty_mult_low=0.70`) was **mathematically incapable** of producing over-subscription:

```
8 agents, total emissions ≈ 22.50 Mt

Supply calculation:
  cap = 22.50 × (1 + (-0.05)) = 21.375 Mt
  auction_vol ≈ 21.375 × 0.90 = 19.24 Mt

Minimum demand (all agents bid qty_mult_low):
  min_demand = 22.50 × 0.70 = 15.75 Mt

Ratio: 15.75 / 19.24 = 0.82x → GUARANTEED UNDER-SUBSCRIPTION
```

In a uniform-price sealed-bid auction, under-subscription means **everyone gets what they bid at the lowest submitted price**. With 8 agents all discovering this, they rationally converge to bidding at the floor — the Nash equilibrium of an oversupplied uniform-price auction.

### Why The Original Changes Were Insufficient

The Chapter 1+2 changes correctly identified three root causes (baseline subtraction, unsold snowball, bot subsidy) but missed the fourth: **the action space lower bound must force over-subscription for competitive pricing to emerge**. The cap tightening from +2% to -5% was not aggressive enough given the wide qty_mult range [0.7, 1.5].

### Fix Applied (Chapter 3)

| # | Parameter | Before (Ch.2) | After (Ch.3) | Rationale |
|---|-----------|---------------|--------------|-----------|
| M7 | `cap_overhead_pct` | `-0.05` | `-0.10` | Reduces supply by 10% below emissions |
| M8 | `qty_mult_low` | `0.70` | `0.85` | Min demand (19.1 Mt) > supply (18.2 Mt) |
| M9 | `coverage_shaping.enabled` | `false` | `false` (kept off) | Config existed but had NO implementation in code. Also violates ETS rationality: real companies have no external incentive to bid a specific coverage ratio |

**Verification**:
```
Supply: 22.50 × 0.90 × 0.90 ≈ 18.2 Mt
Min demand: 22.50 × 0.85 = 19.1 Mt
Ratio: 19.1 / 18.2 = 1.05x → OVER-SUBSCRIBED AT MINIMUM BIDS
```

At any bid coverage ≥ 0.85x, the auction is over-subscribed. This means some bids will be rationed, the clearing price will exceed the floor, and agents receive a meaningful price signal to learn from.

### Expected Outcome

With guaranteed over-subscription:
1. Clearing price > 45 EUR/t from episode 1
2. Price variance provides learning gradient
3. Agents must compete on price (not just quantity)
4. Coverage shaping guides early exploration toward ~1.0x (rational coverage)
5. Two-class polarization should dissolve as all agents face real auction competition

**All 264 tests pass after these changes.**

---

## Chapter 4: Uniform-Price Demand Reduction (13k episodes, 2026-04-09)

### Observed Failure

The 13k-episode run with Chapter 3 structural over-subscription fix showed:

| Metric | Early (0-1k) | Late (10k-13k) | Problem |
|--------|-------------|----------------|---------|
| Episode-mean clearing price | 82.8 ± 18.5 | 46.8 ± 4.8 | Converging to floor |
| Price at floor (45€) | 37% of episodes | 87% of episodes | Getting worse |
| A7/A8 bid prices | ~163€ | 427/406€ | Bidding at ceiling |
| A1-A4 bid prices | ~161€ | 82-84€ (many at 45€) | Converging to floor |
| warn_priceFloor | - | 136,166 cumulative | Chronic |
| warn_debtSpiral | - | 81,089 cumulative | Chronic |

### Root Cause: Uniform-Price Demand Reduction

The structural fix correctly guaranteed over-subscription (demand/supply ratio 1.05-1.6x). **But the clearing price still pinned to 45 EUR/t in 87% of episodes.**

This is the well-known **demand reduction equilibrium** in uniform-price auctions (Ausubel & Cramton, 2002; Klemperer, 2002). In a uniform-price auction:
- Every winner pays the **marginal** (lowest winning) bid
- A bidder's own bid only affects **whether** they win, not **what** they pay
- The dominant strategy is to bid at the reserve floor to minimize the risk of pushing up the marginal price
- With 8 agents, each discovers: "I can bid 45€, get allocated at 45€, while others bid 500€ and also pay 45€"

Year-level evidence (episode 12000):
```
yr=0: A1=45  A2=477  A3=346  A4=45  A5=45  A6=74  A7=500  A8=52 → cp=45
yr=5: A1=45  A2=45   A3=45   A4=45  A5=45  A6=45  A7=49   A8=500 → cp=45
```
Even with demand 1.45x supply, the clearing price is 45€ because the marginal winner bid 45€.

The agents split into two groups:
- **Green agents (A7-A8)**: Learned to overbid massively (400-500€) to guarantee allocation. Under uniform pricing, overbidding is costless — they still pay 45€.
- **Dirty agents (A1-A4)**: Learned to bid at floor. With demand > supply, some floor-bidders get rationed, but the ones that DO win pay 45€ — a better outcome than bidding higher.

### Why This Is The Correct Nash Equilibrium

This is not a learning failure. The agents correctly discovered that uniform-price auctions with a reserve floor have a degenerate equilibrium at the floor price. This is a known theoretical result.

### Fix Applied (Chapter 4): Pay-As-Bid Pricing

| # | Change | Before | After | Rationale |
|---|--------|--------|-------|-----------|
| M10 | `auction.pricing_rule` | `"uniform"` (implicit) | `"pay_as_bid"` | Each winner pays their own bid price |

**Pay-as-bid** (discriminatory pricing) fundamentally changes the game theory:
- Bid too high → overpay for allowances (direct cost penalty)
- Bid too low → don't get allocation (compliance penalty)
- Optimal → estimate the clearing price accurately

This creates a **direct causal link** between bid price and cost, giving agents a rich learning gradient. The UK ETS uses pay-as-bid pricing on the ICE Endex platform, so this is an ETS-rational choice.

**Implementation**: Added `pricing_rule` parameter to `market_clearing_ets()`. When `"pay_as_bid"`, each winner's payment = their allocation × their bid price (not the marginal price).

**All 264 tests pass.**

---

## Chapter 5: Credit Assignment & Initial Bank (2026-04-10)

### Problem Restatement

Chapter 4's pay-as-bid proposal was rejected: the EU ETS uses uniform pricing, and the model must respect that. The uniform-price demand reduction equilibrium (floor-bidding) is the **correct Nash equilibrium** for this auction format. The real problem is not the auction — it's that agents cannot learn the REST of the game:

1. All action dimensions produce near-identical total rewards (advantage ≈ 0 → no gradient)
2. Agents don't discover that buying on secondary closes their compliance gap
3. Agents start with near-zero bank (5% of need), making them immediately dependent on auction success in a game where floor-bidding is rational

### Root Cause: Collapsed Credit Assignment

The reward is computed as one scalar per year combining auction cost + secondary cost + penalty + investment + ESG. When the auction clears at floor price (as it rationally should), and secondary market trades are thin, the total reward is dominated by penalty and fixed costs. The **marginal contribution** of each action dimension is invisible.

The existing phase-split architecture (r_auction and r_secondary stored separately) was already in place but:
- `r_auction` = pure negative cost signal, no coverage feedback
- `r_secondary` = residual (total - auction), dominated by penalty, no gap-closure feedback

### Changes Implemented

#### D1: Realistic Initial Bank (Carry-Over)

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `initial_bank_fraction` | 0.05 | 0.25 | 25% of annual need — represents carry-over from prior compliance period |
| `bank_seed_min` | 0.03 | 0.15 | Consistent with raised fraction (warm-start burnin) |
| `bank_seed_max` | 0.08 | 0.35 | Consistent with raised fraction (warm-start burnin) |

**Rationale**: Real EU ETS companies enter each compliance period with banked allowances from prior years. Starting at 5% created immediate compliance desperation — agents had to win big at auction in year 1 or face penalties. At 25%, agents have breathing room to explore secondary market strategies before hitting compliance walls. This also makes the secondary market meaningful: agents with bank > need become natural sellers, agents with bank < need become natural buyers.

#### D2: Coverage Credit in Auction Reward

Added to `compute_auction_rewards()`:
```
coverage_credit = coverage_credit_weight × min(allocation / annual_need, 1.5) × shaping_weight
r_auction += coverage_credit
```

- `coverage_credit_weight: 0.3` — moderate signal weight
- Capped at 1.5× to prevent rewarding excessive over-bidding
- Decays with `shaping_weight` so agents eventually rely on pure cost signal
- Gives the auction policy a **direct gradient**: bid effectively → win more → positive credit

#### D3: Gap Closure Credit in Secondary Reward

Added to `_compute_rewards()`:
```
if deficit > 0 and bought > 0:
    closure_ratio = min(bought / deficit, 1.0)
    gap_closure_credit = gap_closure_weight × closure_ratio × shaping_weight
```

- `gap_closure_weight: 0.5` — strong signal for deficit agents
- Only activates when agent has compliance deficit AND buys on secondary
- `deficit = emissions + carry_forward - pre_trade_holdings`
- Decays with `shaping_weight` so agents eventually learn from penalty alone
- Gives the secondary policy a **direct gradient**: "buy when short → positive reward"

### Why These Changes Work Together

1. **Initial bank** creates a heterogeneous starting position: some agents are natural sellers (green, low emissions, proportionally larger bank), others are natural buyers (dirty, high emissions). This gives the secondary market real volume and price discovery.

2. **Coverage credit** makes the auction policy trainable even when clearing price pins at floor. The agent learns: "winning more at auction is good" independently of the price paid.

3. **Gap closure credit** makes the secondary policy trainable for deficit agents. Instead of discovering through trial-and-error over thousands of episodes that buying closes the gap → reduces penalty, the credit provides a direct shaping signal that decays as the agent internalizes the strategy.

4. Both credits decay via `shaping_weight`, so the long-run equilibrium depends on actual costs and penalties — the credits are training wheels, not permanent distortions.

### Files Modified

| File | Changes |
|------|---------|
| `configs/default.yaml` | v7.11.0: initial_bank_fraction 0.05→0.25, bank_seed_min/max 0.03/0.08→0.15/0.35, coverage_credit_weight=0.3, gap_closure_weight=0.5 |
| `src/environment/ets_environment.py` | coverage_credit in compute_auction_rewards(), gap_closure_credit in _compute_rewards(), trade_qtys parameter added to _compute_rewards() |
| `tests/test_environment.py` | test_unsold_volume_rolls_over: explicit qty_mult_low=0.1 override for rollover test |

**All 264 tests pass.**

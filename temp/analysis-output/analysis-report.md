# Analysis Report: v7.11.0 Training Run (Seed 42, 70k Episodes)

## Analysis Question

**Did the 8-agent HAPPO system learn a functioning carbon allowance market with realistic price discovery, green transition, and compliance behavior over 70,000 episodes?**

## Run Metadata

| Parameter | Value |
|-----------|-------|
| Version | v7.11.0 |
| Seed | 42 |
| Episodes | 70,000 |
| Years/episode | 12 |
| Agents | 8 learning (A1-A8), 0 bots |
| Algorithm | HAPPO (sequential update by emission intensity) |
| Exploration | Uniform epsilon-greedy, epsilon 0.30 -> 0.05 over 80% of training |
| Price floor | 45 EUR/tCO2 |
| Penalty rate | 138.75 EUR/tCO2 (base, with inflation) |

---

## Key Findings

### Finding 1: Complete Price Discovery Failure

The clearing price collapsed to the floor (45 EUR) and remained there for the entire training run. This is the dominant failure mode.

- **Early (0-5k)**: mean 78.15 EUR, 65.3% at floor -- exploration drives some above-floor prices
- **Mid (20k-30k)**: mean 47.49 EUR, 95.2% at floor -- exploration decaying, floor locking in
- **Late (60k-70k)**: mean 45.15 EUR, 99.8% at floor -- fully locked

The price never recovered. The clearing price MA-1000 hit the floor before episode 10,000 and stayed there for 60,000 subsequent episodes (Figure 1). The floor-streak counts (Figure 1 lower panel) show agents A1-A6 accumulating year-long streaks of floor-price bids by the late phase.

**Evidence**: Figure 1, Figure 6a (year-level boxplots all compressed to 45 EUR).

### Finding 2: Scarcity Exists But Produces No Price Signal

The market IS over-subscribed in the converged state. Total emissions exceed the cap from year 0 onward:

| Year | Emissions (Mt) | Cap (Mt) | Ratio |
|------|---------------|----------|-------|
| 0 | 21.07 | 20.25 | 1.04x |
| 6 | 15.71 | 14.93 | 1.05x |
| 9 | 14.27 | 12.25 | 1.16x |
| 11 | 13.48 | 10.47 | 1.29x |

Scarcity increases over the 12-year episode (reaching 1.29x by year 11). Yet the clearing price remains at 45 EUR in all years (Figure 6a). This confirms the diagnosis: the uniform-price auction mechanism produces zero policy gradient for bid_price because all winners pay the marginal (lowest winning) bid, not their own bid.

**Evidence**: Figure 6a, Figure 6b.

### Finding 3: Heterogeneous Agent Stratification is Frozen

Green fractions are essentially static throughout training:

| Agent | Initial Green | Converged Green (last 5k) | Change |
|-------|--------------|--------------------------|--------|
| A1 | 22% | 32.8% | +10.8% |
| A2 | 23% | 37.5% | +14.5% |
| A3 | 43% | 60.2% | +17.2% |
| A4 | 43% | 60.0% | +17.0% |
| A5 | 73% | 98.0% | +25.0% |
| A6 | 73% | 98.1% | +25.1% |
| A7 | 93% | 100.0% | +7.0% |
| A8 | 93% | 100.0% | +7.0% |

A5-A8 reached near-full green by episode ~5,000 and stayed there. A1-A4 ("brown" agents) made minimal green progress (Figure 4). The brown agents' green fractions plateau around episode 5,000 and show no improvement for the remaining 65,000 episodes.

Investment behavior (Figure 8) confirms stasis: invest_frac is constant across all 12 years for each agent, with no year-to-year adaptation. A7 invests at ~3%, A5 at ~2%, A1-A4 at <1%.

**Evidence**: Figure 4, Figure 8.

### Finding 4: Extreme Reward Inequality

The reward distribution is bimodal, split by initial green fraction:

| Agent | Total Reward (last 5k) | Annual Penalty (M EUR) |
|-------|----------------------|----------------------|
| A1 | -6.56 | 2,025 |
| A2 | -1.54 | 307 |
| A3 | -3.82 | 1,225 |
| A4 | -0.16 | 1,010 |
| A5 | -0.90 | 153 |
| A6 | +3.64 | 425 |
| A7 | -0.10 | 36 |
| A8 | +5.56 | 69 |

A1 receives total reward of -6.56 and pays 2,025 M EUR in penalties per episode. A8 earns +5.56 and pays only 69 M EUR. The 75x penalty differential between A1 and A7 is driven entirely by initial energy mix, not learned strategy. A1 cannot escape its penalty burden within the 12-year horizon.

**Evidence**: Figure 3, Figure 5.

### Finding 5: A7 Discovered a Ceiling Bid Strategy

A7 is the only agent that learned a non-trivial bid strategy: bidding at 330 EUR/t (mean), roughly 7.3x the reserve price. A7's bid-to-reserve ratio is 6.4x while all other agents are near 1.0-1.5x (Figure 10a).

However, this strategy has no market impact because under uniform pricing, A7 still pays 45 EUR (the marginal bid). A7's high bids only increase its collateral costs without changing the clearing price.

This is a pathological artifact: A7, being already 100% green, has near-zero emissions and near-zero compliance need. It bids aggressively because the cost signal from collateral is too weak (at the old 5% rate) to penalize overbidding, and the reward landscape has no downside for high bids when you don't need the allowances.

**Evidence**: Figure 2 (A7 panel), Figure 10a, Figure 10b.

### Finding 6: Secondary Market Shows Role Specialization

The secondary market shows clear buyer/seller specialization:
- **Net sellers (95%+ sell)**: A1, A3, A5, A6, A7, A8
- **Net buyers (97%+ buy)**: A2, A4

This is surprising given that A1 (the brownest agent) is a net seller, not a buyer. The secondary market is active but prices are high (200-300 EUR, Figure 6c), far above the auction floor. This disconnect (auction at 45 EUR, secondary at 200+ EUR) represents a persistent arbitrage opportunity that agents never exploit by bidding higher at auction.

**Evidence**: Secondary market stats, Figure 6c.

### Finding 7: Training Did Converge -- to a Bad Equilibrium

Actor and critic losses (Figure 7) show standard convergence patterns:
- Critic loss decreases from ~0.08 to ~0.02-0.04 and stabilizes
- Actor loss rises during exploration, peaks around episode 15k-20k, then decays to ~0.01

The networks learned successfully. The policies converged. The problem is that they converged to a suboptimal Nash equilibrium where all agents bid at floor. This is a stable equilibrium because no single agent can unilaterally improve its outcome by bidding higher (uniform pricing).

**Evidence**: Figure 7.

### Finding 8: Diagnostic Scores Show Financial vs Green Disconnect

- S_financial ranges from 0.40 (A2) to 0.97 (A7/A8): financial performance strongly correlates with initial green fraction
- S_green ranges from 0.13 (A1) to 0.57 (A7/A8): green transition scores are low across the board for brown agents
- S_composite shows A7 as the highest-scoring agent (1.27), A2 as lowest (0.59)

The diagnostic scores are essentially determined at initialization and never change meaningfully during training (Figure 9 is flat after episode 5,000).

**Evidence**: Figure 9.

---

## Root Cause Diagnosis

The 70,000-episode run demonstrates a **floor-price trap** caused by three interlocking mechanisms:

1. **Uniform-price auction zero-gradient**: Under uniform pricing, all winners pay the marginal bid. Bidding above the floor is individually irrational because it increases collateral but doesn't change the price paid. The policy gradient for bid_price is zero above the marginal bid.

2. **Exploration self-reinforcement**: The expected_price observation (obs[3]) reflects the historical clearing price, which is 45 EUR. The epsilon-greedy exploration anchors on this value, so even random exploration samples near the floor, preventing discovery of above-floor strategies.

3. **Initial heterogeneity trap**: Brown agents (A1-A4, starting 22-43% green) face unavoidable penalty costs that dominate their reward signal, swamping any bid-strategy gradient. Green agents (A5-A8, starting 73-93% green) have near-zero emissions and no compliance pressure, making bid strategy irrelevant to them.

The system converged to a competitive equilibrium that is Pareto-inferior: all agents would be better off if the clearing price were higher (reflecting true scarcity), but no individual agent has the unilateral incentive to bid higher.

---

## Caveats and Limitations

1. **Single seed**: All statistics are from seed 42 only. No multi-seed confidence intervals are possible.
2. **No counterfactual**: Without a comparison run (e.g., with bots, or with pay-as-bid), we cannot quantify how much of the failure is structural vs. algorithmic.
3. **Year-level statistics from converged phase only**: Year-level analysis uses the last 5,000 episodes. Earlier episodes are dominated by exploration noise and may not represent learned behavior.
4. **Reward scale**: Rewards are normalized by REWARD_SCALE=1000 M EUR, making absolute values hard to interpret without the normalization context.

---

## What Changed in Our Understanding

Before this analysis, the working hypothesis was that insufficient scarcity caused under-subscription. The data shows the opposite: **the market is over-subscribed (1.04-1.29x) throughout the converged phase**. The problem is not supply/demand -- it is that uniform pricing eliminates the bid-price signal.

The three fixes implemented (Option B: amplified collateral, Option C: cover_ratio observation, Option D: WTP-anchored exploration) directly target the three legs of this trap. Whether they are sufficient requires the next training run.

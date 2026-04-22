# Figure Catalog: v7.11.0 Run Analysis

---

## Figure 1: Auction Clearing Price Over Training

- **Filename**: `figures/figure-01-clearing-price.png`
- **Purpose**: Show the clearing price trajectory over 70k episodes to assess whether price discovery occurred.
- **Data source**: `training_log_s42.csv`, column `clearing_price_last` (MA-1000).
- **Variables plotted**: Upper: clearing price (MA-1000) + epsilon overlay. Lower: per-agent floor-streak count (MA-2000).
- **Error bars**: None (moving average of single-seed run).

### Caption Requirements
- State the MA window (1000 episodes)
- Note the floor line (45 EUR) and penalty reference (138.75 EUR)
- Note that epsilon (gray fill) decays from 0.30 to 0.05

### Key Observation
The clearing price starts near 120 EUR during high-exploration episodes, drops to 50 EUR by episode 5,000, and pins to 45 EUR (floor) by episode 10,000. It never recovers. The lower panel shows floor-streak counts rising for A1-A6 agents from episode 40,000 onward, confirming agents are increasingly stuck at the price floor.

### Interpretation
**Why exists**: Primary evidence of the price discovery failure.
**What to notice**: The price curve is monotonically decreasing toward the floor; no phase of price recovery or oscillation occurs after episode 10,000.
**What it changes**: Confirms that the HAPPO agents converged to a floor-price equilibrium. Any claim about "market price dynamics" in the thesis must be heavily qualified.

### Caveats
- MA-1000 smooths out rare above-floor spikes in the late phase (which occur in <0.2% of episodes)
- Single seed; different seeds may show different early-phase dynamics

---

## Figure 2: Per-Agent Mean Bid Price

- **Filename**: `figures/figure-02-bid-prices.png`
- **Purpose**: Show each agent's individual bid price trajectory to identify heterogeneous bidding strategies.
- **Data source**: `training_log_s42.csv`, columns `bid_price_A{1..8}` (MA-2000) + raw scatter for last 5k.
- **Variables plotted**: Per-agent bid price (smoothed line + raw dots in last 5k episodes).

### Key Observation
A1-A6 converge to bids near the floor (45-57 EUR). A8 learns a moderately elevated bid (~80 EUR). A7 diverges dramatically to ~330 EUR. The raw scatter in the last 5k shows A1-A4 have tight distributions around the floor with occasional above-floor spikes, while A7's distribution is centered far from the floor.

### Interpretation
**Why exists**: Reveals per-agent strategy differentiation.
**What to notice**: A7 is the only agent with a clearly non-floor bid strategy. A8 is partially elevated. A1-A6 are indistinguishable from floor bidders.
**What it changes**: A7's ceiling-seeking behavior (330 EUR vs WTP of ~168 EUR) is irrational under uniform pricing -- it pays extra collateral for no benefit. This highlights that the collateral gradient was too weak (at 5% rate) to constrain A7.

---

## Figure 3: Per-Agent Reward Decomposition

- **Filename**: `figures/figure-03-reward-decomposition.png`
- **Purpose**: Show how total reward, base reward, and shaping reward evolve per agent.
- **Data source**: `training_log_s42.csv`, columns `reward_{a}`, `reward_base_{a}`, `reward_shaping_{a}` (MA-2000).

### Key Observation
Shaping and base rewards converge together (shaping decays to zero). A6 and A8 achieve positive total rewards (+3.6, +5.6). A1 is stuck at -6.5 (the worst) with no improvement trend after episode 10,000. The reward trajectories for all agents flatten by episode ~15,000.

### Interpretation
**Why exists**: Confirms whether agents learned to improve their outcomes.
**What to notice**: A1's reward is constant and deeply negative. No agent shows a reward improvement trend after episode 15k.
**What it changes**: The flat reward curves after episode 15k suggest the system reached a stable equilibrium early. Additional training episodes (beyond 15k) added no value.

---

## Figure 4: Green Transition Progress

- **Filename**: `figures/figure-04-green-fraction.png`
- **Purpose**: Show whether agents learned to invest in renewable energy over training.
- **Data source**: `training_log_s42.csv`, columns `green_frac_A{1..8}` (MA-2000).

### Key Observation
A7/A8 reach 100% green by episode 2,000. A5/A6 reach ~98% by episode 5,000. A3/A4 plateau at ~60%, A1/A2 at ~33-37%. The four "tiers" are visible as four flat horizontal bands that form by episode 5,000 and never change.

### Interpretation
**Why exists**: Core metric for whether agents learn the green transition.
**What to notice**: The green fraction is determined by initial conditions, not by learning. The tier structure (22%->33%, 43%->60%, 73%->98%, 93%->100%) shows each agent made roughly the same proportional improvement (about +10-25 percentage points) regardless of training duration.
**What it changes**: This is evidence that the floor-price trap eliminates the economic incentive for green investment. Without price differentiation, there is no carbon cost signal to drive transition beyond the initial shaping-driven investment phase.

---

## Figure 5: Penalty Burden and Compliance Shortfall

- **Filename**: `figures/figure-05-penalty-shortfall.png`
- **Purpose**: Show penalty costs and compliance shortfalls by agent over training.
- **Data source**: `training_log_s42.csv`, columns `penalty_A{1..8}`, `shortfall_A{1..8}` (MA-2000).

### Key Observation
A1 pays ~2,000 M EUR/episode in penalties, A3 ~1,200 M EUR. These stabilize by episode 10,000 and show no improvement trend. Shortfall follows the same pattern: A1 faces ~15 Mt/episode shortfall, A3 ~8 Mt.

### Interpretation
**Why exists**: Quantifies the compliance burden differentials.
**What to notice**: Penalty costs for brown agents are stable (not decreasing), confirming they never learned to reduce their compliance gap through investment or trading.
**What it changes**: A1's penalty of 2,025 M EUR/episode represents ~43% of its 880 M EUR/yr x 12 yr budget. This agent is structurally insolvent from day one.

---

## Figure 6: Year-Level Market Dynamics (Converged Phase)

- **Filename**: `figures/figure-06-year-dynamics.png`
- **Purpose**: Show within-episode market structure at convergence.
- **Data source**: `year_log_s42.csv`, last 5,000 episodes.

### Key Observations

**6a (Clearing Price by Year)**: All year-level boxplots compressed to 45 EUR. No year-to-year price variation.

**6b (Emissions vs Cap)**: Total emissions decline from ~21 Mt (year 0) to ~13 Mt (year 11). Cap declines from ~20 Mt to ~10 Mt. Emissions exceed cap in most years (over-subscription). The gap widens in later years.

**6c (Secondary Price by Year)**: Secondary market prices are 200-300 EUR -- far above the auction clearing price of 45 EUR. The 5-6x price differential between primary and secondary markets represents a massive persistent arbitrage.

**6d (Bid Coverage by Year)**: A7's coverage ratio drops from ~1.5x to ~0.8x over the 12-year episode, suggesting it buys less as it runs out of need. Other agents remain near 1.0x.

### Interpretation
**Why exists**: Connects episode-level aggregate metrics to within-episode market structure.
**What to notice**: The auction-secondary price disconnect (45 vs 200-300 EUR) is the most striking feature. Agents are paying 200+ EUR on the secondary market when they could bid above 45 EUR at auction and pay less. This confirms agents never learned the fundamental arbitrage.
**What it changes**: The secondary market IS producing price variation and economic signals. The auction is the specific bottleneck.

---

## Figure 7: Actor and Critic Training Loss

- **Filename**: `figures/figure-07-training-losses.png`
- **Purpose**: Verify that neural network training converged properly.
- **Data source**: `training_log_s42.csv`, columns `actor_loss_A{1..8}`, `critic_loss_A{1..8}` (MA-2000, clipped).

### Key Observation
Critic loss decreases monotonically from ~0.08 to 0.02-0.04 and stabilizes. Actor loss shows the expected PPO pattern: rises during high-exploration phase, peaks at ~0.04 around episode 15-20k, then decays to ~0.01.

### Interpretation
**Why exists**: Rules out neural network training failure as the cause of poor market outcomes.
**What to notice**: Both networks converged normally. The learning infrastructure works -- the agents just learned a bad equilibrium.
**What it changes**: Confirms the problem is game-theoretic (equilibrium selection), not optimization-theoretic (gradient failure).

---

## Figure 8: Investment Behavior

- **Filename**: `figures/figure-08-investment.png`
- **Purpose**: Show investment patterns (queue sizes and investment fractions).
- **Data source**: Left: `training_log_s42.csv` `queue_size` (MA-2000). Right: `year_log_s42.csv` `invest_frac_post_clip` by year (last 10k eps).

### Key Observation
Queue sizes are stable after episode 5,000. Investment fractions are constant across all 12 years for each agent -- no year-to-year adaptation. A7 invests ~3%/yr, A5 ~2%/yr, A1-A4 < 1%/yr.

### Interpretation
**Why exists**: Tests whether agents learned dynamic investment strategies (investing more early, less after greening).
**What to notice**: Investment fraction is flat across years. Agents learned a single static investment rate, not a time-varying strategy.
**What it changes**: Confirms that the policy collapsed to a fixed-action equilibrium for the investment dimension as well.

---

## Figure 9: Diagnostic Scores

- **Filename**: `figures/figure-09-diagnostic-scores.png`
- **Purpose**: Track composite performance metrics over training.
- **Data source**: `training_log_s42.csv`, `diag_S_financial`, `diag_S_green`, `diag_S_composite` (MA-3000).

### Key Observation
All diagnostic scores are flat after episode 5,000. S_financial is stratified by initial green fraction (A7/A8 ~0.97, A1 ~0.58, A2 ~0.40). S_green is similarly stratified. No agent shows improvement in any diagnostic score after the initial exploration phase.

### Interpretation
**Why exists**: Provides a composite assessment of agent performance.
**What to notice**: Scores are initialization-determined. 65,000 episodes of training produced no measurable improvement in any agent's diagnostic score.
**What it changes**: This is the strongest single piece of evidence that training plateaued at episode ~5,000. The remaining 65,000 episodes were wasted compute.

---

## Figure 10: Bid Strategy Analysis

- **Filename**: `figures/figure-10-bid-strategy.png`
- **Purpose**: Compare bid prices to reserve price and WTP to identify strategic bidding.
- **Data source**: `year_log_s42.csv`, last 5,000 episodes.

### Key Observations

**10a (Bid-to-Reserve)**: A7 bids 6-11x reserve in early years, declining to ~2x by year 11. All other agents bid 1-1.5x reserve.

**10b (Bid vs WTP)**: A7's bid (solid ~330 EUR) far exceeds its WTP (dashed ~225 EUR). A1-A6 bid well below their WTP. A8 bids near its WTP.

### Interpretation
**Why exists**: Tests whether agents bid rationally relative to their fundamental valuation.
**What to notice**: A1-A6 underbid (bid < WTP) because the floor anchors their policy. A7 massively overbids (bid >> WTP) because it has no compliance pressure and weak collateral penalty. Both are irrational but stable.
**What it changes**: Confirms that the policy for bid_price is disconnected from economic fundamentals. The WTP-anchored exploration fix (Option D) directly targets this disconnect.

---

## Summary Table

| Figure | Primary Finding | Severity |
|--------|----------------|----------|
| 1 | Clearing price locked at floor after ep 10k | Critical |
| 2 | A7 ceiling-seeker, A1-A6 floor-stuck | Critical |
| 3 | Rewards flat after ep 15k | High |
| 4 | Green fraction frozen by initial conditions | High |
| 5 | Penalty burden unchanged by learning | High |
| 6 | Auction-secondary price disconnect (45 vs 250 EUR) | Critical |
| 7 | Networks converged normally | Neutral (good) |
| 8 | Investment fraction static across years | Medium |
| 9 | Diagnostic scores flat after ep 5k | High |
| 10 | Bids disconnected from WTP | Critical |

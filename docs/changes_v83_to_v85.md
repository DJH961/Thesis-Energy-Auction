# Core Changes: v8.3 → v8.5

This document summarises the most impactful changes introduced between version 8.3 and the current version (8.5). Changes are grouped by theme and ordered roughly by their importance to the simulation's learning outcomes.

---

## 1. Reward bugs that were teaching agents the wrong thing

These were the most critical problems: the reward signal is the only feedback agents have. If it is miscalibrated, they learn backwards no matter how many episodes they run.

### Agents had no reason to comply (v8.3.1 — penalty scale mismatch)

The penalty for missing compliance was charged using the agent's annual budget (≈ €800–1500 M) as the denominator, while the reward for *buying* allowances used a much smaller compliance cost figure (≈ €200–400 M) as the denominator. The two were 3–5× apart in scale, so skipping compliance looked cheap compared to buying — the opposite of how EU ETS works. Concretely, a 1 Mt shortfall cost only −0.17 in reward but buying that Mt cost −0.33; under-bidding was quietly optimal. Both denominators are now the same (compliance cost scale), making a missed tonne strictly more expensive than buying it.

### Penalty double-counted at 2×–3× rate (v8.2 — included for context)

A *prospective* penalty and a *realized* penalty were both driven by the same shortfall and summed together, effectively charging non-compliance at 2–3× the statutory rate. This was fixed in v8.2 by splitting them cleanly: one term for the actual payment, one for the economic cost of carry-forward debt.

### ESG reward was 80% silenced (v8.2)

An internal scaling factor (`esg_anchor_ratio`) was supposed to keep ESG from swamping the financial signal, but in practice it evaluated to 0.07–0.24 and muted ESG by ~5×. The result was a de-facto 80:20 financial-to-ESG weighting when 50:50 was intended. The broken multiplier was removed; `esg.scale` is now the sole calibration knob.

### Investment penalised 4× too heavily (v8.2)

Capital costs were divided by the compliance cost figure (≈ €175 M) rather than the agent's actual annual budget (≈ €725 M). This made green capex appear 4× more expensive in reward terms than it really is, systematically teaching agents not to invest — contradicting the entire point of the simulation. Denominator corrected.

### Over-spending on allowances was rewarded, not penalised (v8.4.2 — coverage gate asymmetry)

`_compute_rewards()` multiplied the financial reward by coverage fraction (how much of the needed allowances were actually secured). The intention was: "if you only bought half of what you needed, only credit you for half the savings." But the multiplication applied symmetrically — it also *discounted* the cost of over-buying. An agent that skipped the auction entirely got zero reward; an agent that won at a high price got a partial negative reward that was further shrunk by low coverage. Skipping the auction was better than bidding. Fixed: the coverage gate now applies **only** to the savings branch (when an agent spent less than expected). Over-buying is fully charged at face value.

### Gap penalty on the wrong scale (v8.4.2)

`compute_auction_rewards()` divided the gap penalty by `budget_real` while dividing the compliance reward by the smaller `compliance_denom`. Because `budget_real ≈ 10 × compliance_denom`, missing 1 Mt saved ~1.0 of compliance reward but cost only ~0.2 of gap penalty. Covering your obligations was explicitly irrational. Fixed: gap penalty now uses `compliance_denom`, so a missed tonne costs strictly more than buying it.

---

## 2. Closing zero-bid and bank-hoarding exploits

Even with correct reward scale, agents found structural loopholes that made passive strategies optimal.

### Zero-bidding was costless (v8.3.0 — banking signal)

An agent holding banked allowances could cover compliance at zero cash cost and receive no reward penalty. Skipping the auction was a dominant strategy: unsold allowances rolled into next year's supply, depressing prices and reinforcing the exploit. Two complementary fixes were introduced:

- **Imputed drawdown cost.** Bank drawdown is now "priced" at the current clearing price and added to the compliance cost, so drawing 3 Mt from the bank at €80/t is treated the same as buying 3 Mt fresh. The zero-bid loophole is closed.
- **Timing P&L signal.** On top of the imputed cost, a separate banking signal rewards agents that accumulated allowances cheaply and draw them when prices are high (and penalises the reverse). This adds a proper economic incentive for intertemporal arbitrage — banking is now a real strategy, not just a passive accumulation.

### `imputed_cap_factor` too low (v8.4.2 — banking signal cap raised 2× → 5×)

The cap on the imputed bank drawdown cost saturated whenever the clearing price exceeded 2 × anchor price (≈ €134/t). Above that threshold, drawing from the bank was artificially cheaper in reward terms than buying fresh — biasing agents toward larger banks and smaller bids in high-price scenarios. The cap was raised to 5 × anchor (≈ €335/t), well inside emergency price territory, so the imputed cost tracks the clearing price linearly across the realistic range.

---

## 3. Better information for agents: observation improvements

### Cap tightening visible 3 and 6 years ahead (v8.3.1)

Two new observation dimensions show how much the government cap will shrink over the next 3 and 6 years (`cap(t+3) / cap(t)` and `cap(t+6) / cap(t)`). A reading near 1.0 means no near-term scarcity; a reading below 0.8 signals sustained tightening that makes early banking or green investment rational. Without this, agents had no forward-looking scarcity signal.

### Agents can see when their own actions were clipped (v8.4 + v8.4.1)

Previously, when internal safety gates modified an agent's submitted bid (bid change limit, budget cap, collateral limit, secondary market depth, capex limit), the agent received no feedback — it could not tell whether its action was executed as submitted or silently changed. Four new observation dimensions were added:

- How close the current bid is to the price change ceiling (headroom)
- How much the bid price was adjusted by the price change limit (signed)
- How much the bid price was adjusted by the budget cap (signed, added in v8.4.1 as a separate dimension)
- Ratio of actual vs requested bid quantity
- Ratio of actual vs requested investment fraction
- Ratio of actual vs requested secondary market quantity

This is smart because a policy gradient can only learn to respect a constraint if it can observe when the constraint is active. Previously the gradient signal was simply cut off at the gate with no information flowing back.

### Exploration anchored to year-adjusted price (v8.3.2)

Random exploration was centred on a fixed price estimate (~€93/t) regardless of the simulation year. Early years (low scarcity) and late years (high scarcity) need very different centre points. Exploration now centres on the fundamental equilibrium price for the *current year*, scaled by a small boost factor. Year 0 explores around ~€76/t; year 11 around ~€115/t.

### Opponent market share replaces opponent bank level (v8.5)

The opponent observation previously included each competitor's bank size normalised by their own need (`bank / own_need`). This was replaced by `holdings / total_market_holdings` — each competitor's share of the total outstanding allowance pool. This is a better signal because:
1. It is what the real EU ETS publishes in aggregate TNAC reports — it is publicly inferable without requiring confidential data.
2. It lets agents triangulate price pressure: a small share of the pool suggests a competitor will need to buy more, pushing prices up.

---

## 4. Keeping the simulated market realistic

### MSR reserve pre-loaded at episode start (v8.3.2)

The Market Stability Reserve (MSR) is the EU's price stabilisation buffer: when total allowances outstanding (TNAC) is too high, the MSR withholds supply from auction. During the simulation burn-in period, TNAC was always far below the withholding threshold, so every episode started with an empty reserve. Agents never encountered MSR pressure in early years despite it being a real feature of the EU ETS. A warm-start seeds the reserve to ≈ 23% of the initial cap at episode start, matching realistic EU ETS reserve levels.

### Price change limit stabilised, then simplified (v8.3.2 → v8.4)

Agents could previously jump their bid by any amount between years. This allowed immediate escape from any price-discovery regime and prevented coordinated convergence. A per-year change limit was introduced in v8.3.2 with a decaying schedule (100 → 50 EUR/t over training). In v8.4 the schedule was replaced with a simpler fixed limit of 50 EUR/t for the whole run, centred on the 3-year moving average rather than last year's price. A moving average reference is more robust to one-year outliers.

### Fundamental price anchor uses real MSR-adjusted cap (v8.4)

The fundamental equilibrium price is estimated from how scarce the cap is relative to expected emissions. Previously this calculation used a linear approximation of the cap trajectory, which ignored MSR withholding. When the MSR was actively reducing supply, the anchor underestimated true scarcity and the price floor drifted too low. The anchor calculation now receives the actual current-year cap from `CapSchedule` — the number that accounts for MSR dynamics.

### Price change limit floored at fundamental anchor (v8.4.1)

The price change limit window was centred on the 3-year moving average (`price_ma3`). During low-price regimes, `price_ma3` could drift below the fundamental equilibrium, compressing the allowed bidding range to an unrealistically narrow band. The PCL reference is now `max(price_ma3, fundamental_anchor)`, so the window always extends at least to the economically justified price floor.

---

## 5. Training stability: PPO and HAPPO improvements

### Two separate policy optimisers (v8.4.1)

Previously, a single optimiser covered both the auction policy and the secondary-market policy. During a backward pass, gradients from one policy head bled into the other, corrupting per-head gradient norms. Two independent optimisers now step separately with independent gradient clipping. This is basic hygiene: the two policies see different observations and optimise different objectives; mixing their gradients was always a mistake.

### Year-1 has a hard floor on negative advantages (v8.4.1)

Year 1 within an episode is structurally difficult: agents have little price history and typically insufficient banked allowances. Early in training, year-1 transitions receive very large negative advantages, causing the policy to swing aggressively — destabilising the year-0 learning that was just starting to converge. Advantages at year-1 timesteps are now clamped to a floor of −1.0 before normalisation, dampening the most extreme early penalties without affecting other years.

### Causal reward normalisation (v8.4.2)

Phase-wise reward normalisation was using a running normaliser that saw all timesteps. This created a subtle look-ahead bias: early timestep rewards were being normalised by statistics that included future reward scales. The normaliser now walks the trajectory in temporal order (causal), using only past observations when computing the normalisation scale for each timestep.

### Advantage normalisation used a configured floor that was never applied (v8.4.2)

`compute_gae()` computed standard deviation with a `+ 1e-8` epsilon while a configured `gae_min_std` floor sat unused. If the reward variance in a batch was very low, advantages were being divided by near-zero variance — inflating them wildly. Fixed: `torch.clamp(std, min=gae_min_std)` is now used instead.

### Trust region was too loose (v8.4.2)

A `log_ratio` pre-clamp at `[-2, 2]` was applied before the PPO `clip_eps` constraint. Because `exp(2) ≈ 7.4`, this was allowing policy updates far outside the intended trust region before the clip had a chance to act. Widened to `[-20, 20]` so the PPO `clip_eps` alone controls the trust region, which is the correct design.

### Split-head investment critic (v8.5)

The Phase-1 policy has two very different jobs: decide how to bid in the auction (strategic, competitive) and decide how much to invest in green capacity (long-horizon, capital allocation). Previously, both were trained against the same advantage stream from a single value network. Now they have separate value networks and separate advantage streams:

- **Bid sub-head:** trained on compliance costs, secondary market financials, penalties, banking signal, and terminal bank value — the signals relevant to auction strategy.
- **Investment sub-head:** trained on capital cost, ESG signal, and the terminal value of projects under construction — the signals relevant to long-term capacity planning.

This is important because the timescales are different: auction bids pay off within the year; green investments take years to complete and decades to pay back. A single advantage stream averages these together and dilutes both gradients.

### HAPPO update order based on advantage, not reward (v8.5)

HAPPO (Heterogeneous-Agent PPO) updates agents in a specific sequence where each agent conditions on updates from prior agents. The order was previously determined by each agent's recent reward. If different agents have different reward scales (e.g. because their cost basis differs), this creates a systematic ordering bias unrelated to contribution quality. The new default orders agents by their mean GAE advantage, which is already normalised per-agent and reflects relative policy improvement rather than absolute reward level.

---

## 6. Investment mechanics made more realistic

### Investment can now spread across multiple technologies (v8.5)

Investment used to choose a single technology each year via `argmax` over three technology logits. This forced all-or-nothing allocation: either 100% onshore wind or 100% solar, never a split. Real companies diversify. The action is now passed through a `softmax`, distributing the investment budget proportionally across technologies. This also eliminates the discontinuity created by argmax, which is poorly behaved for gradient-based optimisation.

### Per-technology cancellation rates (v8.5)

Previously a single cancellation rate applied to all construction projects. Projects are now cancelled at technology-specific rates calibrated to industry data (≈ 1.2%/yr onshore wind, 0.8%/yr offshore, 2.0%/yr solar), reflecting the higher regulatory and planning risk of large-scale solar in the modelled period.

### Investment capacity scales with revenue (v8.5)

A company's construction throughput (how many concurrent projects it can manage) now scales with its current-year electricity revenue: high-revenue years gain construction headroom; low-revenue years see it shrink. This models the real-world constraint that capital programmes are bounded by balance-sheet capacity. Previously, throughput was fixed regardless of financial state.

### Terminal project value uses proper discounted cash flow (v8.5)

At the end of an episode, projects still under construction received a credit proportional to how much time remained in their useful life. This was a linear approximation that undervalued late-episode investments. The value is now computed as a proper discounted cash flow: each project's annual carbon savings (at terminal price) are valued as an annuity over a 20-year asset lifetime at a 5% WACC, discounted back from the expected completion date. A wind farm started in year 11 is now worth its economic value, not an arbitrary linear fraction.

### Terminal bank value simplified (v8.5)

The end-of-episode allowance bank was valued with a piecewise formula (linear below need, log above). This has been replaced by a single discounted-hold formula: `holdings × terminal_price × (1 + invest_rate)^(−payoff_years) / budget_real`. This is economically cleaner: banked allowances are worth their future value at the time they would realistically be surrendered, discounted back to today.

---

## 7. ESG signal redesigned

### ESG signal centred on a linear decarbonisation baseline (v8.5)

Previously, the ESG reward was proportional to absolute green fraction — every increase in renewables was equally rewarded regardless of how ambitious it was relative to a business-as-usual path. The signal is now **centred on a linear trajectory**: doing nothing (business-as-usual decarbonisation) earns zero ESG reward. Only progress *ahead of* the linear path pays a positive signal. Agents that fall behind the trajectory receive a negative signal. This is economically much more sensible: it rewards additionality rather than any green activity, and it avoids rewarding agents simply for being in the later years of an episode when their green fraction is naturally higher.

### ESG speed bonus front-loaded (v8.5)

The speed bonus (extra reward for increasing green fraction fast) was interpolated to be higher in late years to compensate late investors. With the new terminal-queue DCF valuation (Section 6 above), late-episode investments already receive fair economic credit through the terminal value. There is no longer a need to artificially boost their ESG speed bonus. The bonus is now front-loaded: early decarbonisation receives the higher speed bonus, which better reflects real policy intent.

---

## 8. Penalty flows through the budget system correctly (v8.5)

Compliance penalties were previously charged outside the regular budget waterfall. An agent could be penalised without that cost registering in `budget_spent_this_year`, meaning the penalty did not trigger the hard-cap penalty channel and was effectively invisible to the budget accounting system. Penalties now flow through `Company.record_spending()` like all other costs, so non-compliant agents face the full chain of consequences: penalty payment → potential hard-cap breach → hard-cap penalty. The `budget.hard_cap_multiplier` configuration key has been removed; a single `budget.hard_cap_fraction` now governs both the investment gate and the over-budget penalty, eliminating a parameter that could be set inconsistently.

---

## 9. Hyperparameter and config polish (v8.4.1 — v8.4.2)

Several training hyperparameters were tuned following the reward and architecture fixes above. The most notable:

| What changed | Why |
|---|---|
| Mini-batch size 32 → 64; episodes per update 16 → 32 | Larger batches reduce gradient variance; more on-policy data per update stabilises HAPPO |
| Actor LR 0.0003 → 0.0002; critic LR 0.001 → 0.0005 | Lower rates for more stable late-training updates; new independent critic LR floor prevents over-fitting the value function |
| PPO clip ε 0.20 → 0.15; target KL 0.02 → 0.015 | Tighter trust region matches the more precise reward signal |
| Entropy final 0.01 → 0.005; entropy decay 90% → 70% of training | Faster decay to exploitation once the reward signal is reliable |
| Bid change limit 50 → 75 EUR/t | More headroom for price discovery following the fundamental anchor correction |
| Separate critic LR decay schedule | Critic can sustain value accuracy late in training without throttling actor exploration |
| Cosine LR decay moved outside the update gate | Previously only advanced on update episodes, not every episode — a scheduling bug |
| Python `random` seeded alongside numpy/torch | Ensures full reproducibility for all random operations |

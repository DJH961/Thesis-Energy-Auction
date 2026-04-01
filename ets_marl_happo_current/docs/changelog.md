# Changelog — ETS MARL

All significant changes to the `ets_marl_happo_current` simulation are documented here.
For historical versions that predate this module, see the archived directories
`ets_marl_legacy_test` (DDPG prototype) and `ets_marl_legacy_ppo` (early PPO version).

---

## v6.2.0

**Bot Stochastic Valuation and Differentiation**

- Each bot now samples a persistent `valuation_noise` (drawn from N(0, σ)) and an `urgency_multiplier` (drawn from U[low, high]) at episode start. These persist for the entire episode, giving bots heterogeneous but stable bidding personalities across years.
- Bot pairs are differentiated by archetype via `urgency_denominators`: even-indexed bots use a denominator of 1.3 (more aggressive coverage), odd-indexed bots use 1.7 (more conservative). This creates distinct within-archetype behaviour.
- New config section `bots:` controls `valuation_noise_std`, `urgency_mult_low`, `urgency_mult_high`, and `urgency_denominators`.

**MSR Price-Containment Fix**

- Replaced ratio-based MSR triggers (which compared the clearing price to a fraction of `price_max`) with dynamic absolute thresholds derived from the inflation-adjusted effective penalty rate.
- Containment trigger fires at `1.8 × effective_penalty_rate` (or `price_containment_absolute` as a hard fallback). Emergency release fires at `2.5 × effective_penalty_rate` (or `price_release_absolute`).
- This fixes an always-below-clearing bug where the old ratio trigger (70 % of price_max = 350 EUR/t) was almost never reached, making the containment mechanism effectively dormant.
- MSR trigger events (containment, emergency release) are now logged.
- Removed deprecated `price_containment_trigger` and `price_release_trigger` ratio parameters from config.

**Terminal Bank Value Cap**

- Effective bank for terminal valuation is now capped at `min(holdings, 2.0 × annual_need)`. Holdings beyond a 2-year reserve receive zero additional terminal credit, making secondary-market selling rational for well-banked agents.

**Secondary Revenue as Budget Credit**

- Negative `secondary_cost` (i.e. revenue from selling allowances) now reduces the agent's budget spending for the year, freeing headroom for investment. Previously selling produced revenue that did not offset the budget constraint.

**Permanent Efficiency Bonus**

- A cost-efficiency improvement bonus (`0.3 × ef_improvement_ratio × time_weight × price_weight`) now applies to all agents regardless of `w_green`, giving coal-heavy agents a gradient for early green investment. Previously this bonus only benefited agents with non-zero ESG weight.

**PPO Value Function Clipping**

- Critic loss now implements value clipping: the predicted value is clamped to `old_values ± clip_eps` before computing the loss when `clip_value=true`. This stabilises critic training by preventing large value jumps between updates.
- `old_values` tensor is now passed through `compute_gae()` into both `update()` and `update_happo()`.

**Config Changes**

- `penalty.carry_forward_cap` reduced from 2.0 to 1.0 (tighter carry-forward debt cap).
- `ppo.clip_value` set to `true` by default.
- Removed legacy backward-compatible alias `msr.msr_activation_year`.
- Removed legacy `penalty.inflation_random_window` (replaced by `inflation_random_std`).

---

## v6.1.0

**Scale-Up to 16 Participants**

- Added 8 heuristic bot agents (B1–B8) mirroring all four learning-agent archetypes. Market now has 16 total participants.
- Cap year 0 recalibrated to 50 Mt (~11 % surplus over ~45 Mt total initial emissions for 16 agents). Previously 57 Mt (26 % surplus), which produced insufficient early scarcity.
- Bot annual budgets and capex throughputs added to config; bots participate identically in auction clearing and secondary matching.

**MSR Calibration and Stability Improvements**

- `tnac_upper` and `tnac_lower` recalibrated for the expanded 16-agent market.
- MSR activation lag (`activation_year: 2`) added: MSR is inactive for the first two years of each episode, mirroring the EU ETS lagged TNAC observation requirement (Decision 2015/1814, Art. 1(5)).
- MSR cancellation mechanism added: MSR holdings exceeding the previous year's auction volume are permanently cancelled each year (EU ETS post-2023 reform).
- Ratio-based price-responsive MSR safeguards introduced (containment at 70 % of price_max, emergency release at 85 % — subsequently replaced in v6.2).
- `price_history_anchor` set to `"auction"` to prevent reserve-price pollution of the MA3 when auctions fail.

**HAPPO and Centralised Critic**

- Training algorithm upgraded from independent PPO to HAPPO (Heterogeneous-Agent PPO) with sequential per-agent policy updates.
- Centralised critic added (concatenated multi-agent state, 680D input, 512-unit hidden layer).
- Separate critic learning rate (`critic_lr: 0.001`) and extra critic-only pre-training epochs added.
- Cosine annealing LR schedule for both actor and critic optimisers.

**Opponent Modeling**

- Each agent's observation now includes a 5D public state vector for every other participant (normalised emissions, carry-forward, green fraction, fossil fraction, queue size). Phase 1 observation dimension grows from 25 to 100 (16 agents); Phase 2 to 107.

**ESG Reward Signal**

- Saved-carbon-years ESG formula added: `w_green × ef_ratio × time_ratio × (budget/1000)`. ESG agents (w_green = 0.5) receive an exact 50/50 financial/environmental reward split.
- `esg.scale` calibrates ESG magnitude to match the cost channel.
- Terminal ESG queue value added (discounted by γ^years_late for delayed projects).

**Behavioral Cloning Warm-Start**

- Pre-training from heuristic policy (behavioral cloning) added to seed sensible initial strategies before RL exploration begins. Duration auto-scales with n_episodes.

**Historical Policy Pool (HPP)**

- Periodic snapshots of past actor policies maintained in a pool; random swaps during training keep opponent diversity and prevent synchronized policy collapse.

**Training Stability**

- Diagnostics module added (stuck-market detection, bid-ceiling / bid-floor warnings, zero-quantity warning, entropy boost trigger).
- `bank_seed_max` reduced from 2.0 to 1.5 to lower initial banking and tighten early scarcity.
- `reward.clip_max` raised to 10.0 (symmetric with clip_min) to avoid asymmetric gradient compression.
- Auto-scaled schedules: warmup, pretraining, exploration decay, and HPP timing all scale automatically with `n_episodes`.

---

## v5.0

*Corresponds to `ets_marl_legacy_ppo` — the first version to reach near-production quality.*

**Scale-Up to 8 Agents and 5 Technologies**

- Expanded from 4 agents to 8 learning agents organised into four archetypes (coal-heavy, gas-dominant, transitioner, green-leader), each pair containing one financially-motivated and one ESG-balanced agent.
- Technology model expanded from a simple fossil/green split to 5 explicit technologies: coal, gas, onshore wind, offshore wind, and solar — each with distinct emission factors, CAPEX, capacity factors, construction delays (2–7 years), operational costs, and decommissioning costs (sourced from IRENA 2024, IPCC AR5, IEA WEO 2024).

**PPO Replaces DDPG**

- Policy algorithm switched from DDPG to PPO. DDPG had exhibited multi-agent instability (oscillating rather than converging policies). PPO with clipped objectives proved more stable.
- Each agent has a decentralised actor and (initially independent) critic.
- GAE advantage estimation, entropy regularisation, and epsilon-greedy physical-action exploration introduced.

**MAC Fuel-Switching**

- Marginal Abatement Cost (MAC) mechanism added: companies can temporarily switch coal dispatch to gas when the carbon price exceeds the MAC threshold (48 EUR/tCO2). This lowers in-year emissions without permanently altering the technology mix.

**Electricity Revenue Channel**

- Companies now earn electricity revenue: `P_elec = P_base + passthrough × P_carbon × EF_system`. Revenue offsets cost signals and links carbon prices to generation margins.

**Carry-Forward Non-Compliance**

- Compliance shortfalls can optionally carry forward to the next year (with a configurable cap multiplier). This prevents runaway debt spirals while creating realistic multi-year compliance tension.

**Unified Budget Envelope**

- Annual budget caps all spending (compliance, CAPEX, MAC). A separate CAPEX throughput constraint models physical delivery bottlenecks independently of the financial limit.

**Heuristic Baseline Policy**

- Fundamentals-based heuristic policy introduced for baseline comparisons: MAC→penalty gradient bidding, NPV-gated investment, target-bank trajectory secondary trading.
- Behavioral cloning from this policy used as a warm-start for PPO agents.

**Two-Phase Yearly Decision**

- Decision process split into Phase 1 (auction bid + investment) and Phase 2 (secondary market), each with its own observation vector and policy network.

**Construction Risk and Delays**

- Investment failure probability depends on fossil exposure and accumulated success history.
- Construction jitter (Poisson delay), project cancellation risk, and capacity-factor noise added.

**AR(1) Expected Price Signal**

- AR(1) model generates an expected price observation, giving agents a forward-looking signal without requiring a full rational-expectations equilibrium.

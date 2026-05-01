# ETS MARL — Reward Function (Thesis Addendum)

This document gives a self-contained mathematical statement of the
per-agent, per-year reward used in `src/environment/ets_environment.py`
(`_compute_rewards`). It is intended to slot into the thesis as a
short reference appendix; it complements `docs/esg_reward_design.md`
(which is the long-form design document for the ESG term) and
`docs/config_dictionary.md` § `reward` and § `esg`.

Notation
:   `t` = current year (0…11). `i` = agent index. `infl_t` = cumulative
    inflation factor at year `t`. `anchor_t` = fundamental price anchor
    (MAC-scarcity-penalty mix); `anchor_real_t = anchor_t / infl_t`.
    `budget_real_t = annual_budget_i / infl_t`. `need_t` =
    deterministic compliance need estimate at obs time.

---

## 1. Top-level decomposition

For each `(i, t)` the reward is the sum of three streams plus optional
terminal payoffs at `t = T-1`:

```
r_{i,t} =  financial_reward_{i,t}              (compliance + capital + soft)
         + w_green · esg_signal_{i,t}          (saved-carbon hybrid)
         − penalty_norm_{i,t}                  (realised + prospective)
         + banking_signal_{i,t}                (bank-timing P&L)
         + opp_cost_shaping + coverage_gap_shaping     (decaying shaping)
         + 𝟙[t = T-1] · (terminal_bank + terminal_queue + terminal_treasury
                        − carry_forward_debt_penalty)
```

`reward_base_{i,t}` is the same expression minus the two shaping
channels and minus the terminal payoff. `reward_shaping_{i,t} = r −
reward_base`. Both decompositions are written to the year-level CSV.

---

## 2. Cost buckets (real terms)

All cash flows are deflated by `infl_t` before entering the reward, so
the agent's reward is inflation-invariant.

```
compliance_cost_real = (auction_cost + secondary_cost + mac_cost) / infl_t
capital_cost_real    = (investment_cost / infl_t) + opex_delta_real
soft_penalty_real    = (budget_penalty + capex_penalty + loan_interest) / infl_t
```

`opex_delta_real` is the **change** in real OPEX vs the year-0 baseline
mix. Re-deflating before subtracting prevents a phantom OPEX signal
in years where the mix is unchanged but inflation has moved.

---

## 3. Normalisation

Three buckets are normalised by two anchors:

```
compliance_denom  = max( anchor_real_t · need_t , 1 )      # value-of-need anchor
soft_denom        = max( budget_real_t , 1 )                # capital-/soft-budget anchor

compliance_norm_cash = compliance_cost_real / compliance_denom
imputed_bank_norm    = min( bank_drawdown · clearing_price / infl_t , imputed_cap_factor · compliance_denom ) / compliance_denom
compliance_norm      = compliance_norm_cash + w_imputed · imputed_bank_norm

capital_norm  = capital_cost_real / soft_denom
soft_norm     = soft_penalty_real / soft_denom

cost_norm     = compliance_norm + capital_norm + soft_norm + loan_sting
```

`imputed_bank_norm` charges drawn-down bank allowances at the current
clearing price — closing the "free compliance via bank" shortcut a
zero-bid agent would otherwise enjoy. `loan_sting` is a small reward
penalty when an emergency loan is actually drawn this year.

Centring at expected compliance:

```
cost_norm_centered = cost_norm − 1
```

A fully compliant agent buying exactly at the anchor scores
`compliance_norm = 1.0`, so subtracting 1 puts the financial reward at
`0` for "normal operation". Negative ⇒ outperforming the anchor;
positive ⇒ overspending.

---

## 4. Financial reward (with auction coverage gate)

`coverage_frac_auction = min(alloc_t / need_t , 1)`

```
                          ⎧ coverage_frac_auction · w_cost · (− cost_norm_centered)   if cost_norm_centered < 0
financial_reward_{i,t} =  ⎨
                          ⎩ w_cost · (− cost_norm_centered)                            otherwise
```

The asymmetric gate is intentional: **cost overruns apply at full
weight** but **savings are coverage-discounted**, removing the
"skip-the-auction → 0 financial reward" exploit (winning at clearing
≥ anchor would otherwise give a positive reward without compliance).

---

## 5. Penalty term (realised + prospective)

The penalty is split into two economically distinct components:

* **Realised** — money paid this year for shortfall, reconstructed
  in real terms from the nominal `penalty_cost`:

```
shortfall_realized = penalty_cost / effective_penalty_rate_t
penalty_realized   = shortfall_realized · penalty_rate · urgency_scalar_i
                     / max( infl_t · compliance_denom , 1 )
```

* **Remediation (prospective)** — carry-forward debt must be repaid
  next year by buying replacement allowances at the next-year anchor,
  amplified under scarcity:

```
scarcity_t  = max( 0 , 1 − cap_t / cap_0 )
scarcity_amp = 1 + scarcity_t

remediation_cost = cf_debt · anchor_next_real · scarcity_amp · urgency_scalar_i
                   / max( compliance_denom , 1 )
```

```
penalty_norm = penalty_realized + remediation_cost
```

`urgency_scalar_i ~ LogNormal(0, 0.30²)` is sampled **per agent per
episode** and folded into both branches; `cf_debt` is the agent's
current carry-forward obligation.

---

## 6. ESG term — saved-carbon hybrid

When `esg.enabled=true` and the agent's initial emission factor is
positive:

```
ef_ratio_t          = max( 0 , (initial_ef − weighted_ef_t) / initial_ef )           ∈ [0, 1]
green_delta_t       = max( 0 , green_frac_t − green_frac_{t-1} )
speed_bonus_t       = speed_coef · green_delta_t
anchor_real_ratio_t = anchor_real_t / max(anchor_real_0 , 1)

esg_raw = scale · ( stock_w · ef_ratio_t
                  + flow_w  · ef_ratio_t · anchor_real_ratio_t
                  + speed_bonus_t )

coverage_frac    = min( 1 , holdings_pre_compliance / need_t )
gate_blend       = clip01( (gate_threshold − coverage_frac) / gate_width )      ∈ [0, 1]
compliance_gate  = coverage_frac ^ (1 + gate_blend)

esg_signal       = compliance_gate · esg_raw          if esg_raw ≥ 0
                 = esg_raw                            otherwise   (ungated, defensive)
```

* The **stock term** `stock_w · ef_ratio` rewards *sustained* operation
  of low-EF capacity — pays every year, not just on the transition step.
* The **flow term** `flow_w · ef_ratio · anchor_real_ratio` is
  algebraically `(saved_Mt_t · anchor_real_t) /
  (initial_baseline_emiss · anchor_real_0)` — avoided carbon valued at
  the *live* social shadow price, normalised by the agent's year-0
  carbon liability. Rises with cap scarcity.
* The **speed bonus** is a small motion bonus, uniform across the
  episode (no front-loading bias).
* The **compliance gate** ensures ESG never out-pays compliance:
  `compliance_gate → 0` as `coverage_frac → 0`. The exponent
  `1 + gate_blend` sharpens the gate around `gate_threshold = 0.90`.

Default calibration: `scale=0.50, stock_w=1.0, flow_w=1.5,
speed_coef=0.3` (uniform). At these settings a `[w_cost=0.5,
w_green=0.5]` agent achieves ≈ 50/50 financial/ESG balance over a
fully compliant rollout — see `docs/esg_reward_design.md` for the full
calibration sweep and stress scenarios.

---

## 7. Banking timing signal

A small reward for "buying low, holding, drawing high":

```
bank_drawdown   = clip( bank_start , 0 , total_oblig − total_new )
imputed_real    = bank_drawdown · clearing_price / infl_t
banking_signal  = w_banking · bank_drawdown · (clearing_price − bank_cost_basis_i) / infl_t
                  / max( compliance_denom , 1 )
```

This makes a "store cheap, use expensive" inventory profitable and
penalises the reverse, mirroring the EU-ETS banking-premium intuition
without a crude price-spread reward. `bank_cost_basis_i` is an EMA of
the price the agent paid for its banked allowances.

---

## 8. Decaying shaping channels

Two early-training shaping channels with weight `shaping_weight_t`
that decays from `1 → shaping_weight_floor` (default `0`) over the
first `shaping_decay_frac · n_episodes` (default `0.10 · n_episodes`):

```
opp_cost_shaping       = − scale · max(0, sec_buy_cost_per_mt − clearing_price) · sec_buy_qty
                          · shaping_weight / max(budget_real, 1)
coverage_gap_shaping   = − scale · max(0, need − alloc) · penalty_rate
                          · shaping_weight / max(budget_real, 1)
```

Both go to zero by the end of the shaping window so the asymptotic
reward is unbiased. They live in `reward − reward_base`.

---

## 9. Terminal payoffs (year `T-1` only)

To remove the "stop investing in late years" structural bias and
to value un-spent allowances, the final year adds:

```
terminal_price = max( clearing_price_T , last_secondary_price_T , 0.8 · effective_penalty_T )

terminal_bank   = holdings_T · terminal_price · (1 + r_inv)^(−terminal_payoff_years)
                  / budget_real_T

terminal_queue  = Σ_{j ∈ queue}  delta_ef_j · frac_delta_j · output_MWh / 1e6
                                · terminal_price
                                · annuity( asset_lifetime, r_inv )
                                · (1 + r_inv)^(−years_to_completion_j)
                                / budget_real_T

terminal_treasury = treasury_reserve · treasury_terminal_rate / budget_real_T

carry_forward_debt_penalty = 1.5 · cf_debt · terminal_price / budget_real_T
```

`r_inv = investment.discount_rate = 0.05`, `asset_lifetime = 20 yr`
(default), `terminal_payoff_years = 5` (default).
`carry_forward_debt_penalty` is subtracted from `r` and `reward_base`
to ensure the policy cannot win by deferring all defaults to the
terminal step.

---

## 10. Per-agent reward weighting

Each agent has fixed `(w_cost_i, w_green_i)` from
`companies.reward_weights`. The four default learning archetypes use
`(1.0, 0.0)` (pure financial) and `(0.5, 0.5)` (balanced ESG); two
seats per archetype, total 8 agents. **Penalty, banking, shaping, and
terminal payoffs are not multiplied by `w_cost`/`w_green`** — only the
financial bucket and the ESG bucket are agent-weighted.

---

## 11. Numerical post-processing

Each agent maintains a `RewardNormalizer` (EMA, `α = 0.02`,
`gae_min_std = 0.15`) that returns a standardised reward to PPO's GAE
without altering the logged `reward_*` columns. The raw, log-scale
reward is what appears in `training_log_*.csv` and `year_log_*.csv`.

The reward is finally clipped to
`[reward.clip_min, reward.clip_max] = [−10, +10]` before being passed
to GAE — a numerical guard against single-step blowups, never reached
in normal operation under the default calibration.

---

## 12. Diagnostic channels logged per `(i, t)`

`_last_reward_channels[i]` (mirrored into the year-level CSV via
`info["year_log"]`) records: `compliance_norm,
compliance_norm_cash, imputed_bank_norm, bank_drawdown,
bank_cost_basis, banking_signal, capital_norm, soft_norm, cost_norm,
expected_compliance_norm, cost_norm_centered, coverage_frac_auction,
financial_reward, penalty_norm, penalty_realized, penalty_prospective,
remediation_cost, scarcity_amp, esg_signal, esg_anchor_ratio,
esg_stock_term, esg_flow_term, base_reward, opp_cost_shaping,
coverage_gap_shaping, gate_activation, compliance_gate,
esg_vs_penalty_ratio, anchor_t, anchor_real, anchor_next_real,
budget_real, infl, shortfall, carry_forward_debt`.

These per-channel diagnostics are how the post-hoc decomposition
plots (`notebooks/ets_marl - Full Run & Analysis.ipynb` § *Reward
decomposition* and § *ESG balance*) are produced.

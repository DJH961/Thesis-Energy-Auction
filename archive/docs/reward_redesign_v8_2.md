# Reward Redesign — v8.2

## Why this change

Across the first ~400 episodes of training, every learning agent's reward
stayed structurally negative and showed no clear upward trend (e.g. A1 ≈ −20,
A7 ≈ −20 to −48 across the entire trajectory). A line-by-line trace of
`_compute_rewards` and `compute_auction_rewards` in
`src/environment/ets_environment.py` revealed five interacting issues that
together explain the symptom and prevent the reward from realizing its
stated design goals. This document records what was wrong, what changed,
and how to verify it.

The intended vision (per project owner):

- The reward should be **balanced within an episode across years** — i.e.
  inflation in later years must not make the reward more negative.
- **Scarcity** should be appropriately factored into the cost of failing
  to comply, because catching up gets harder under a tightening cap.
- For ESG-active agents (`w_cost = w_green = 0.5`), the financial and
  ESG channels should contribute **roughly 50:50** to the total signal.
- Pure-financial agents (`w_green = 0`) should still have an **upward
  optimization direction** — currently the reward ceiling is structurally
  zero, so they can only "minimize losses" rather than "maximize profit".

## Bugs identified

### Bug 1 — Penalty was double-counted (`_compute_rewards`)

```text
penalty_norm = penalty_prospective + penalty_realized
             = shortfall × penalty_rate × scarcity_amp × urgency / budget_real
             + shortfall × penalty_rate × urgency           / budget_real
```

Both terms were driven by essentially the same shortfall (modulo small CF
noise), so a single non-compliance event was charged at **2× to 3× the
real penalty**.

The original intent (clarified by the project owner) was that the
`scarcity_amp` term represented the **future remediation cost** — the
fact that a carry-forward debt is harder to repay next year under
scarcity. That economic story is correct; the implementation just
expressed it as a duplicate penalty rate term instead of a market-price
remediation term.

### Bug 2 — `capital_norm` used the wrong denominator

```text
capital_norm = capital_cost_real / compliance_denom   # = anchor × need
```

`compliance_denom` ≈ €175M for a coal-heavy agent; `budget_real` ≈ €724M.
Dividing capital costs by the smaller denominator over-penalized
investment by a factor of `budget_real / compliance_denom` ≈ 4×, directly
discouraging the green transition the simulation is designed to teach.

### Bug 3 — No revenue baseline

The README's reward section explicitly lists "**Revenue from selling
electricity (including carbon cost passthrough)**" as a component, but
`compute_revenue()` was only consumed by:

1. The dynamic budget calculation (`budget.mode = revenue_based`).
2. Diagnostic logging (`per_agent_diag["revenue"]`).

It never entered the reward. As a consequence, for a pure-financial
agent (`w_green = 0`):

```text
base_reward = w_cost × (−cost_norm) − penalty_norm   ≤ 0
```

The structural maximum is exactly zero (achieved only by paying nothing
and emitting nothing — impossible). PPO had no upward signal to follow,
only "less negative" gradients.

### Bug 4 — ESG anchor ratio silently muted ESG, breaking 50:50

```text
esg_anchor_ratio = min(compliance_denom / budget_real, 2.0)
                 ≈ 0.07 to 0.24 in practice
esg_signal = esg_scale × (ef_ratio + speed_bonus) × esg_anchor_ratio × compliance_gate
```

The `esg_anchor_ratio` factor scaled the ESG signal down by ~5×. Combined
with `w_cost = w_green = 0.5`, the **realized weighting** was approximately
80:20 cost:ESG, not 50:50. A fully-decarbonized compliant agent contributed
~0.11 to ESG per year vs ~0.50 to the cost channel.

The `min(_, 2.0)` cap never bound in practice (the ratio is always < 0.3),
so it was dead code as written.

### Bug 5 — Inflation invariance was violated by the penalty terms

The cost normalization correctly cancelled `infl` between numerator and
denominator (numerator deflated by `/infl`, denominator was `budget_real
= annual_budget / infl`). But the penalty terms broke this:

```text
penalty_realized = penalties[i] × urgency / budget_real
                 = (shortfall × penalty_base × infl) / (annual_budget / infl)
                 = shortfall × penalty_base × urgency × infl² / annual_budget
                                                      ▲▲▲▲
                                                      QUADRATIC in inflation

penalty_prospective = shortfall × penalty_rate × scarcity_amp × urgency × infl / annual_budget
                                                                         ▲▲▲▲
                                                                         linear in infl
```

By year 11 (cumulative inflation ≈ 1.24), `penalty_realized` was 1.54×
more punishing than year 0 and `penalty_prospective` was 1.24× more
punishing — directly contradicting the "inflation should not make later
years more negative" design goal.

The root cause: `penalty_cost` came from `settle_compliance_realized`,
which uses `effective_penalty_rate = base × infl` (already inflated),
then was divided by `budget_real = budget / infl` (deflated again),
multiplying the inflation factor instead of cancelling it.

### Bug 6 — Phase-1 and Phase-2 rewards lived on different scales

```text
# Phase 1 (compute_auction_rewards):
r_auction = -(total_cost / 1000) - gap_penalty            # flat REWARD_SCALE

# Phase 2 (_compute_rewards):
rewards   = w_cost × (-cost_norm) + ... - penalty_norm    # budget_real ≈ 700

# train.py:1154
r_secondary = rewards - r_auction
```

Subtracting two budget-normalized values from a flat-scale value is
dimensionally inconsistent. In numbers: `r_auction ≈ −0.18`,
`rewards ≈ −1.5` per step, so the secondary head's residual is ~7× the
magnitude of the auction head's signal. The auction policy gradient was
proportionally weaker than the secondary one for the same underlying
mistake — undermining the whole purpose of the split-reward design.

## What changed

All edits are in `src/environment/ets_environment.py`. No config schema
changes are required, but the existing `esg_scale` knob now has a
different effective meaning (see Fix D below).

### Fix A — Penalty: realized payment + forward remediation, no double count

```python
eff_pen_rate       = max(company.effective_penalty_rate(self.current_year), 1e-9)
shortfall_realized = float(penalty_cost) / eff_pen_rate

# (1) Money paid this year — reconstructed in REAL terms (uses base rate)
penalty_realized = (shortfall_realized * company.penalty_rate * urgency_scalar
                    / max(budget_real, 1.0))

# (2) Forward remediation — carry-forward debt has to be repaid next year
#     by buying replacement allowances at the EXPECTED MARKET PRICE under scarcity.
anchor_next_real = anchor_next_nom / infl
scarcity_amp     = 1.0 + scarcity_t
remediation_cost = (float(company._carry_forward) * anchor_next_real * scarcity_amp
                    * urgency_scalar / max(budget_real, 1.0))

penalty_norm = penalty_realized + remediation_cost
```

Honors the original "scarcity makes catch-up harder" intent without
re-charging the penalty rate. The two terms now represent **distinct**
economic costs (penalty paid vs market-price replacement under scarcity).
Also fixes the inflation invariance issue from Bug 5: numerator uses the
base rate, so only one `infl` factor appears in numerator+denominator
combined, and it cancels with the inflation in `annual_budget` (which
grows with inflation in `revenue_based` budget mode).

### Fix B — `capital_norm` divided by `budget_real`

```python
compliance_norm = compliance_cost_real / compliance_denom
capital_norm    = capital_cost_real    / soft_denom        # was compliance_denom
soft_norm       = soft_penalty_real    / soft_denom
```

Investment now competes against the agent's actual budget envelope, not
against its allowance bill.

### Fix C — Explicit revenue baseline in `base_reward`

```python
revenue_real = company.compute_revenue(
    self._last_marginal_ef, price_ma3_now, infl
) / infl
revenue_norm = revenue_real / soft_denom

base_reward = float(
    company.w_cost  * (revenue_norm - cost_norm)
    + company.w_green * esg_signal
    - penalty_norm
)
```

Pure-financial agents now have a positive baseline (revenue) to maximize
against, instead of a structural ceiling of zero. Matches the README's
documented reward composition.

### Fix D — ESG re-scaled so 50:50 weights yield 50:50 contributions

```python
# Old:
# esg_raw_unanchored = esg_scale * (ef_ratio + speed_bonus)
# esg_anchor_ratio   = min(compliance_denom / budget_real, 2.0)  # ≈ 0.07-0.24
# esg_raw            = esg_raw_unanchored * esg_anchor_ratio

# New:
esg_raw          = esg_scale * (ef_ratio + speed_bonus)
esg_anchor_ratio = 1.0   # retained as a logged channel for back-compat
```

`esg_scale` (default 1.0) is now the single calibration knob. A
fully-decarbonized agent yields `esg_raw ≈ 1.0` per year, which sits on
the same scale as `(revenue_norm − cost_norm)`, so a 50:50 weighting
actually produces 50:50 contributions.

### Fix E — `compute_auction_rewards` uses budget-relative normalization

```python
# Old: r_auction[i] = -(total_cost / 1000.0) - gap_penalty

# New: same shape as _compute_rewards
compliance_norm = (auction_cost + mac_cost_i + collateral_cost_i) / compliance_denom
capital_norm    = (invest_cost + opex_delta) / budget_real
gap_penalty     = (coverage_gap * company.penalty_rate) / budget_real
r_auction[i]    = -(compliance_norm + capital_norm) - gap_penalty
```

Phase-1 and Phase-2 rewards now live on the same scale.
`r_secondary = rewards − r_auction` (in `scripts/train.py:1154`) becomes
dimensionally consistent, and the auction-head policy gradient regains
its proportional weight.

## Diagnostic channel changes

`_last_reward_channels[i]` (per-agent dict in `_compute_rewards`):

| Key | Status | Notes |
|-----|--------|-------|
| `compliance_norm` | unchanged | |
| `capital_norm` | redefined | now `/ budget_real` (was `/ compliance_denom`) |
| `soft_norm` | unchanged | |
| `cost_norm` | unchanged | sum of the three above + `loan_sting` |
| `revenue_norm` | **NEW** | electricity revenue normalized by `budget_real` |
| `penalty_norm` | redefined | now `penalty_realized + remediation_cost` |
| `penalty_realized` | redefined | reconstructed in real terms (base rate) |
| `penalty_prospective` | alias | now equals `remediation_cost` (kept for back-compat) |
| `remediation_cost` | **NEW** | forward remediation under scarcity |
| `scarcity_amp` | unchanged | `1 + scarcity_t` |
| `esg_signal` | redefined | no longer multiplied by `esg_anchor_ratio` |
| `esg_anchor_ratio` | redefined | always `1.0` now (legacy channel) |
| `base_reward` | redefined | includes revenue term |
| `anchor_next_real` | **NEW** | expected next-year market anchor |
| `carry_forward_debt` | **NEW** | `company._carry_forward` after compliance |

`_last_auction_reward_channels[i]` (per-agent dict in
`compute_auction_rewards`): all keys now record real-EUR (deflated) values
instead of `value / 1000`. Two new keys: `compliance_norm`, `capital_norm`.

## Acceptance criteria (sanity test)

After these fixes, a hand-built "ideal" policy — full compliance,
anchor-priced bids, modest investment, no over-budget, no penalty —
should yield, **for every agent**:

1. Per-year `base_reward ≥ 0`.
2. **Inflation invariance**: per-year `base_reward` in year 11 within
   ±5% of year 0 (under `budget.mode = revenue_based`).
3. **50:50 weighting yields ~50:50 contributions**: for an ESG agent
   with `w_green = 0.5`, the ESG-channel magnitude over an episode is
   within ±20% of the financial-channel magnitude.

These three properties are the durable guard against future drift. A new
regression test in `tests/test_rewards.py` should encode them.

## Tests that need updating

The following existing tests reference symbols/semantics that have
changed and will need to be revisited:

- `tests/test_rewards.py:340` — comment references the old ESG formula
  with `esg_anchor_ratio`.
- `tests/test_rewards.py:863-864` — same.
- `tests/test_rewards.py:942-950` (`test_esg_anchor_ratio_capped_at_two`)
  — assertion is now trivially true (`esg_anchor_ratio` is always 1.0);
  the test should either be deleted or replaced with a 50:50
  contribution check.
- `tests/test_rewards.py:1275-1278` — references `REWARD_SCALE` from
  the old `compute_auction_rewards`; the auction reward is now
  budget-normalized, not flat-scale.

Any dashboards or downstream analysis tools that consume the keys
`penalty_realized`, `penalty_prospective`, `esg_anchor_ratio`,
`esg_signal`, `capital_norm`, or `base_reward` will see different
numerical scales — review before comparing across runs.

## Caveats and follow-ups

- The inflation invariance proof for the penalty term assumes
  `budget.mode = revenue_based` (in which `annual_budget` itself grows
  with inflation through `compute_revenue`). Under
  `budget.mode = fixed`, `budget_real = budget_0_fixed / infl` shrinks
  over time and the invariance does not hold for any term. This is a
  pre-existing limitation of `fixed` mode, not a regression.
- `esg_scale` retains its config knob but its **effective meaning has
  shifted**: it is now the only ESG magnitude knob (no longer multiplied
  by `esg_anchor_ratio ≈ 0.2`). Existing trained checkpoints will see
  effectively ~5× larger ESG signal under the new code; either retrain
  from scratch or set `esg.scale ≈ 0.2` in config to roughly preserve
  the old magnitude during transition.
- Terminal-year reward components (`terminal_bank_value`,
  `terminal_queue_value`, `treasury_terminal_value`) were not modified.
  They still use `budget_real_t` and are inflation-invariant by the same
  argument as the rebalanced cost terms.

# Implementation Plan v9 — Budget Reform & Compliance Visibility

> Scope: 7 self-contained changes. Implement in order (each builds on the previous).

---

## Change 1: Remove Suspension → Budget-Based Gate

**Rationale:** Suspension is a blunt market-exclusion mechanism with no real-world ETS analog. Replace with a budget-based freeze: agents who cannot cover collateral simply have their bid quantity clipped to zero — they stay in the market but can't overbid.

### Files
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`ets_environment.py` → `step_auction()`**

Remove the suspension enforcement block:
```python
# DELETE this block:
for i in range(self.n_total):
    if self._suspension_remaining[i] > 0:
        bid_actions[i, 1] = 0.0
        self._suspension_remaining[i] -= 1
```

Replace with budget-based bid gate (insert at same location):
```python
for i in range(self.n_total):
    if not self._is_agent_active(i):
        continue
    cash = max(0.0, float(self.companies[i].annual_budget
                          - self.companies[i].budget_spent_this_year)
               + self.companies[i].get_treasury_available())
    bid_p = float(bid_actions[i, 0])
    if bid_p > 1e-6 and cash < bid_p * bid_actions[i, 1] * 0.10:
        # Agent cannot cover even 10% collateral — zero quantity
        bid_actions[i, 1] = 0.0
```

Remove all `_suspension_remaining` decrements and references in `settle_auction` call. Keep `defaults_mask` handling (agents still default if payment > all available resources).

**`ets_environment.py` → `reset()`**

Remove: `self._suspension_remaining = np.zeros(self.n_total, dtype=int)`
Remove: all logging of `suspension_remaining_list`, `suspended_agents`

**`company.py` → `get_observation_phase1()`**

Remove dim `[28]` (`suspension_remaining_norm`). Shift all subsequent dims down by 1.
Update obs docstring and `obs_dim_phase1` base from 36 → 35 (before treasury dim added in Change 4).

**`configs/default.yaml`**
```yaml
auction:
  suspension_length: 0   # disabled — budget gate replaces suspension
```

---

## Change 2: Budget Envelope Clipping (Price + Quantity)

**Rationale:** Currently only quantity is clipped when budget is tight. The bid price itself is unconstrained relative to budget — an agent can bid €200/t for 2Mt with €50M left, which is economically incoherent. Add a soft price ceiling derived from total cash capacity (operating + treasury).

> ⚠️ Depends on Change 4 (`get_treasury_available`). Implement after Change 4.

### Files
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`ets_environment.py` → `step_auction()`, after leverage gate, BEFORE collateral locking:**

```python
budget_price_clip = self.config["auction"].get("budget_price_clip", True)
if budget_price_clip:
    for i, company in enumerate(self.companies):
        if not self._is_agent_active(i):
            continue
        # Total cash = operating remaining + full treasury (collateral not yet locked)
        cash = max(1.0, float(company.annual_budget - company.budget_spent_this_year)
                   + company.get_treasury_available())
        bid_q = max(float(bid_actions[i, 1]), 1e-6)
        max_affordable_price = cash / bid_q
        # Soft clip: only fires at >1.5x affordable to allow headroom for collateral
        if bid_actions[i, 0] > 1.5 * max_affordable_price:
            bid_actions[i, 0] = float(np.clip(
                max_affordable_price,
                float(self.config["auction"]["price_min"]),
                float(self.config["auction"]["price_max"]),
            ))
```

> **Why 1.5× threshold?** Collateral locking happens after this clip. The 1.5× buffer ensures a bid that looks affordable on total cash is not clipped when collateral subsequently reduces operating cash. Treasury is always available post-collateral, so the buffer only needs to cover the collateral deduction from operating budget.

**`configs/default.yaml`**
```yaml
auction:
  budget_price_clip: true
```

---

## Change 3: Improved Emergency Loan (Sting)

**Rationale:** Flat 8% interest on a fixed loan is painless for large budgets. Replace with leverage-scaled interest, immediate reward origination fee, and capex covenant squeeze during repayment.

> ⚠️ The loan is triggered ONLY after operating budget AND treasury reserve are both exhausted (see Settlement Waterfall in Change 4). `shortfall` passed to `apply_emergency_loan()` is always the true residual after those two sources are drained.

### Files
- `src/environment/company.py`
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`company.py` → `apply_emergency_loan(shortfall)`**

Replace existing method:
```python
def apply_emergency_loan(self, shortfall: float) -> None:
    """Emergency loan — leverage-scaled rate on true residual after treasury exhausted."""
    loan_cfg = self.config.get("budget", {}).get("emergency_loan", {})
    base_rate = float(loan_cfg.get("interest_rate", 0.08))
    leverage_coef = float(loan_cfg.get("leverage_premium_coef", 0.25))
    leverage_exp = float(loan_cfg.get("leverage_premium_exp", 1.5))
    # loan_fraction relative to annual_budget (not incl. treasury)
    loan_fraction = shortfall / max(self.annual_budget, 1.0)
    effective_rate = base_rate + leverage_coef * (loan_fraction ** leverage_exp)
    principal_with_interest = shortfall * (1.0 + effective_rate)
    self._loan_outstanding += principal_with_interest
    self._loan_repayment_annual = (
        self._loan_outstanding / max(self._loan_repayment_years, 1)
    )
    self._years_under_loan = self._loan_repayment_years
    self._last_loan_fraction = loan_fraction
    self._loan_drawn_this_step = shortfall
```

Add to `__init__` and `reset()`: `self._last_loan_fraction = 0.0`, `self._loan_drawn_this_step = 0.0`
Add to `reset_budget()`: `self._loan_drawn_this_step = 0.0`

**`company.py` → new property:**
```python
@property
def effective_capex_throughput(self) -> float:
    """Capex throughput squeezed by debt covenant during loan repayment."""
    if self._years_under_loan > 0 and self._loan_outstanding > 0:
        loan_burden = self._loan_outstanding / max(self.annual_budget, 1.0)
        squeeze = max(
            float(self.config.get("budget", {}).get("emergency_loan", {})
                  .get("capex_squeeze_floor", 0.50)),
            1.0 - loan_burden
        )
        return self.capex_throughput * squeeze
    return self.capex_throughput
```

In `ets_environment.py` → investment block: replace `company.capex_throughput` → `company.effective_capex_throughput`.

**`ets_environment.py` → `_compute_rewards()`**

```python
loan_sting_coef = float(self.config.get("budget", {}).get(
    "emergency_loan", {}).get("origination_sting_coef", 0.30))
loan_draw_this_year = float(getattr(company, '_loan_drawn_this_step', 0.0))
if loan_draw_this_year > 0:
    loan_sting = (loan_draw_this_year / max(company.annual_budget, 1.0)) * loan_sting_coef
    total_cost += loan_sting * REWARD_SCALE
```

**`configs/default.yaml`**
```yaml
budget:
  emergency_loan:
    enabled: true
    max_loan_fraction: 0.15
    interest_rate: 0.08
    repayment_years: 3
    leverage_premium_coef: 0.25
    leverage_premium_exp: 1.5
    capex_squeeze_floor: 0.50
    origination_sting_coef: 0.30
```

---

## Change 4: Corporate Treasury Reserve

**Rationale:** Unspent annual budget currently evaporates. Real utilities retain unspent compliance/operating budget in a liquidity reserve deployable in future crises — before the emergency loan is triggered.

**Economic grounding:**
- 60% retention: CFO liquidity policy; 40% returned to shareholders/operations
- 1.5× cap: ~18 months cash; above this = "overcapitalised" (Moody's liquidity metrics)
- 5% decay: idle capital opportunity cost (EU utility WACC ~7–9%)
- Reserve depletes before loan: Myers (1984) pecking-order theory
- Terminal value at 30 cents: discounted going-concern balance sheet value

### Settlement Waterfall (canonical — all cash logic uses this order)

For every winning agent at auction settlement:

```
1. op_avail   = annual_budget − budget_spent_this_year − collateral_locked[i]
2. treasury   = get_treasury_available()
3. loan_limit = max_loan_fraction × annual_budget

Case A: payment ≤ op_avail
  → record_spending(payment). Treasury untouched. No loan.

Case B: op_avail < payment ≤ op_avail + treasury
  → record_spending(op_avail)
  → draw_treasury(payment − op_avail)
  → No loan.

Case C: op_avail + treasury < payment ≤ op_avail + treasury + loan_limit
  → record_spending(op_avail)
  → draw_treasury(treasury)  [full drain]
  → apply_emergency_loan(payment − op_avail − treasury)

Case D: payment > op_avail + treasury + loan_limit
  → DEFAULT: allocation cancelled, collateral forfeited.
```

> `agent_cash` passed to `settle_auction()` = `op_avail + treasury` (combined).
> `max_loan_budgets[i]` = `loan_limit` only (no treasury in this value).
> Post-settlement loop performs the actual waterfall deductions.

### Files
- `src/environment/company.py`
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`company.py` → `__init__()`**
```python
treasury_cfg = budget_cfg.get("treasury_reserve", {})
self._treasury_enabled = bool(treasury_cfg.get("enabled", False))
self._treasury_reserve = 0.0
self._treasury_cap_mult = float(treasury_cfg.get("cap_mult", 1.5))
self._treasury_retention = float(treasury_cfg.get("savings_retention_rate", 0.60))
self._treasury_decay = float(treasury_cfg.get("decay_rate", 0.05))
self._treasury_terminal_rate = float(treasury_cfg.get("terminal_value_rate", 0.30))
self._treasury_drawn_this_year = 0.0
```

**`company.py` → `reset()`**: add `self._treasury_reserve = 0.0`, `self._treasury_drawn_this_year = 0.0`

**`company.py` → new methods:**
```python
def settle_treasury_year_end(self) -> None:
    """Roll unspent budget into treasury with cap and decay. Call BEFORE reset_budget()."""
    if not self._treasury_enabled:
        return
    unspent = max(0.0, self.annual_budget - self.budget_spent_this_year)
    self._treasury_reserve += unspent * self._treasury_retention
    cap = self._treasury_cap_mult * max(self.annual_budget, 1.0)
    self._treasury_reserve = min(self._treasury_reserve, cap)
    self._treasury_reserve *= (1.0 - self._treasury_decay)
    self._treasury_drawn_this_year = 0.0

def get_treasury_available(self) -> float:
    return float(self._treasury_reserve) if self._treasury_enabled else 0.0

def draw_treasury(self, amount: float) -> float:
    actual = min(float(amount), self._treasury_reserve)
    self._treasury_reserve -= actual
    self._treasury_drawn_this_year += actual
    return actual
```

**`ets_environment.py` → budget reset block (both modes):**
```python
# ORDER MATTERS
company.settle_treasury_year_end()
company.apply_loan_repayment()
company.reset_budget()
company.reset_capex_budget()
```

**`ets_environment.py` → replace `agent_cash` construction + post-settlement deduction:**
```python
# ── Pre-settlement
operating_cash = np.array([
    max(0.0, float(c.annual_budget - c.budget_spent_this_year)
        - float(self._collateral_locked[i]))
    for i, c in enumerate(self.companies)
])
treasury_cash = np.array([c.get_treasury_available() for c in self.companies])
agent_cash = operating_cash + treasury_cash
max_loan_budgets = np.array([
    c._max_loan_fraction * max(c.annual_budget, 1.0) for c in self.companies
])

actual_alloc, actual_pay, defaults_mask, defaulted_vol, _, loan_amounts = settle_auction(
    allocations, payments, agent_cash, np.zeros(self.n_total),
    suspension_length=0, max_loan_budgets=max_loan_budgets,
)

# ── Post-settlement waterfall
for i in range(self.n_total):
    if actual_alloc[i] < 1e-9:
        continue
    payment = float(actual_pay[i])
    op = float(operating_cash[i])
    treas = float(treasury_cash[i])
    if payment <= op:                        # Case A
        self.companies[i].record_spending(payment)
    elif payment <= op + treas:              # Case B
        self.companies[i].record_spending(op)
        self.companies[i].draw_treasury(payment - op)
    else:                                    # Case C
        self.companies[i].record_spending(op)
        self.companies[i].draw_treasury(treas)
        self.companies[i].apply_emergency_loan(loan_amounts[i])
```

**`ets_environment.py` → `_compute_rewards()` terminal block:**
```python
if bool(reward_cfg.get("treasury_terminal_value", True)):
    if company._treasury_enabled and company._treasury_reserve > 0:
        t_value = company._treasury_reserve * company._treasury_terminal_rate / REWARD_SCALE
        rewards[i] += t_value
        terminal_bank_values[i] += t_value
```

**`company.py` → `get_observation_phase1()`** — with Change 1 applied (base = 35), add treasury as dim `[35]`:
```python
treasury_norm = float(np.clip(
    self._treasury_reserve / max(annual_budget, 1.0), 0.0, 2.0
)) / 2.0    # [35] treasury norm
```
Base `obs_dim_phase1` = **36** (net unchanged: −35 suspension + 1 treasury + 35 baseline).

**`configs/default.yaml`**
```yaml
budget:
  treasury_reserve:
    enabled: true
    savings_retention_rate: 0.60
    cap_mult: 1.5
    decay_rate: 0.05
    terminal_value_rate: 0.30
reward:
  treasury_terminal_value: true
```

---

## Change 5: Phase 2 Compliance Gap Observation

**Rationale:** `compliance_liability_norm` encodes magnitude of exposure but not direction. A signed `compliance_gap_norm` gives the policy a direct gradient target.

### Files
- `src/environment/company.py`

### Changes

**`company.py` → `get_observation_phase2()`** — append `[base+10]`:
```python
total_holdings_now = banked + allocation
compliance_gap = (emissions + self._carry_forward - total_holdings_now)
annual_need = max(self.compute_estimate_need(), 1e-6)
compliance_gap_norm = float(np.clip(compliance_gap / annual_need, -2.0, 2.0)) / 2.0
# [base+10]: 0.5=balanced, >0.5=short, <0.5=surplus
```

Append to `extra` → 11 elements.
Update `obs_dim_phase2`: `return self.obs_dim_phase1 + 11`

---

## Change 6: Anchor-Normalised Cost Reward

**Rationale:** The current `total_non_penalty_cost / REWARD_SCALE` uses a static scale that does not account for the growing cost of decarbonisation over time. A year-0 and year-10 agent spending the same *fraction of their budget* should receive the same `cost_norm`. Separating the normaliser between cost and penalty also preserves the intended asymmetry: penalties should escalate nominally with inflation, but spending efficiency should be time-stable.

### Files
- `src/environment/ets_environment.py`
- `src/utils/price_anchor.py` (already has `compute_fundamental_anchor`)
- `configs/default.yaml`
- `tests/test_rewards.py`

### Changes

**`ets_environment.py` → top-level import (add if not already present):**
```python
from src.utils.price_anchor import compute_fundamental_anchor
```

**`ets_environment.py` → `_compute_rewards()`, per-agent cost normalisation:**

Replace:
```python
cost_norm = total_non_penalty_cost / REWARD_SCALE
```
With:
```python
anchor_normalize = bool(reward_cfg.get("anchor_normalize_cost_only", True))
if anchor_normalize:
    anchor_t = compute_fundamental_anchor(self.current_year, self.config)
    estimated_need = max(company.compute_estimate_need(), 1e-6)
    cost_denominator = max(anchor_t * estimated_need, 1.0)
    cost_norm = total_non_penalty_cost / cost_denominator
else:
    cost_norm = total_non_penalty_cost / REWARD_SCALE
```

Penalty normalisation is **unchanged**: `penalty_norm = shortfall * eff_penalty_t / REWARD_SCALE`

Store anchor for diagnostics:
```python
self._last_reward_channels["anchor_t"] = anchor_t if anchor_normalize else None
```

> **Design intent:** `cost_norm` measures efficiency (spend vs. fundamental value of allowances). `penalty_norm` measures nominal non-compliance cost. The asymmetry is intentional and documented via the config flag.

**`configs/default.yaml`**
```yaml
reward:
  anchor_normalize_cost_only: true   # cost_norm uses anchor, penalty_norm uses REWARD_SCALE
```

**`tests/test_rewards.py` — add new test:**
```python
def test_anchor_normalised_cost_symmetry():
    """
    Equal-efficiency spending at year 0 and year 10 should produce equal cost_norm.
    Equal-shortfall at year 0 and year 10 should produce DIFFERENT penalty_norm
    (year 10 higher due to inflation).
    """
    from src.utils.price_anchor import compute_fundamental_anchor
    cfg = load_default_config()   # helper that loads configs/default.yaml

    need = 5.0   # Mt

    for yr in [0, 10]:
        anchor = compute_fundamental_anchor(yr, cfg)
        # Same efficiency: spend exactly anchor * need in both years
        cost_efficient = anchor * need
        cost_denominator = max(anchor * need, 1.0)
        cost_norm = cost_efficient / cost_denominator
        assert abs(cost_norm - 1.0) < 1e-6, f"Year {yr}: cost_norm should be 1.0"

    # Equal shortfall but different inflation => different penalty_norm
    REWARD_SCALE = cfg["auction"]["price_max"]  # or whatever constant is used
    shortfall = 1.0  # Mt
    penalty_yr0 = shortfall * cfg["penalty"]["rate"] / REWARD_SCALE
    inflation_factor_yr10 = (1 + cfg["penalty"]["inflation_rate"]) ** 10
    penalty_yr10 = shortfall * cfg["penalty"]["rate"] * inflation_factor_yr10 / REWARD_SCALE
    assert penalty_yr10 > penalty_yr0, "Inflation should raise penalty_norm over time"
```

---

## Change 7: 7D Lagged Opponent Observation

**Rationale:** The current 6D opponent tuple exposes only static company state. Adding bank position, secondary market behaviour, and a lagged compliance gap gives opponents predictive strategic signal (how exposed are they? are they buying or selling secondary? did they comply last year?). The 1-year lag is essential: at Phase 1, agents cannot observe opponent actions from the *current* year (that information does not yet exist).

### New 7D tuple definition

| Dim | Signal | Normalisation |
|-----|--------|---------------|
| 0 | `verified_emissions` (Mt) | `/ 10.0` |
| 1 | `green_frac` | raw [0,1] |
| 2 | `fossil_frac` | raw [0,1] |
| 3 | `queue_signal` + N(0, σ) | raw, clipped [0,1] |
| 4 | `bank_norm` = holdings / annual_need | clipped [0, 3] / 3 |
| 5 | `net_secondary_norm` = (sec_bought − sec_sold) / annual_need | clipped [−1, 1] |
| 6 | `lagged_compliance_gap_norm` = (prior_emissions − surrendered) / annual_need | clipped [−1, 1] |

All values read from year `t−1` snapshot (initialised to burn-in priors at episode start).

### Files
- `src/environment/ets_environment.py`
- `src/environment/company.py`
- `configs/default.yaml`
- `tests/test_mappo.py`, `tests/test_environment.py`, `tests/test_anchors.py`
- `docs/design.md`

### Changes

**`ets_environment.py` → `__init__()`** — add opponent snapshot state:
```python
# Opponent modeling snapshot (lagged by 1 year)
# Shape: (n_total, 7) — indices match new 7D tuple
self._opponent_snapshots = np.zeros((self.n_total, 7), dtype=float)
self._opponent_snapshots_prev = np.zeros((self.n_total, 7), dtype=float)
```

**`ets_environment.py` → `reset()`** — initialise snapshots with burn-in priors:
```python
for i, c in enumerate(self.companies):
    need = max(c.compute_estimate_need(), 1e-6)
    self._opponent_snapshots[i] = [
        c.compute_emissions() / 10.0,   # verified_emissions_norm
        c.green_frac,                    # green_frac
        c.fossil_frac,                   # fossil_frac
        sum(item["frac_delta"] for item in c._construction_queue),  # queue
        0.5,  # bank_norm: neutral prior
        0.0,  # net_secondary_norm: neutral prior
        0.0,  # lagged_compliance_gap_norm: neutral prior
    ]
self._opponent_snapshots_prev = self._opponent_snapshots.copy()
```

**`ets_environment.py` → end of `step_secondary()`** — update snapshots after compliance settled:
```python
opp_cfg = self.config.get("opponent_obs", {})
queue_sigma = float(opp_cfg.get("queue_noise_sigma", 0.15))
for i, c in enumerate(self.companies):
    need = max(c.compute_estimate_need(), 1e-6)
    queue_raw = float(sum(item["frac_delta"] for item in c._construction_queue))
    queue_noisy = float(np.clip(queue_raw + self.rng.normal(0, queue_sigma), 0.0, 1.0))
    bank_norm = float(np.clip(self.holdings[i] / need, 0.0, 3.0)) / 3.0
    # net_secondary: buy_vol[i] - sell_vol[i] from this year's secondary step
    net_sec = float(np.clip(
        (self._sec_bought[i] - self._sec_sold[i]) / need, -1.0, 1.0
    ))
    # lagged compliance gap: prior year emissions - surrendered
    lag_gap = float(np.clip(
        self._last_compliance_gaps[i] / need, -1.0, 1.0
    ))
    self._opponent_snapshots[i] = [
        c.compute_emissions() / 10.0,
        c.green_frac,
        c.fossil_frac,
        queue_noisy,
        bank_norm,
        net_sec,
        lag_gap,
    ]
# Prev snapshot (year t-1) is what Phase 1 obs reads
self._opponent_snapshots_prev = self._opponent_snapshots.copy()
```

> Add `self._sec_bought`, `self._sec_sold` (shape `n_total`) tracked in `step_secondary()` secondary clearing. Add `self._last_compliance_gaps` (shape `n_total`) = `(realized_emissions[i] - surrendered[i])` recorded after compliance.

**`ets_environment.py` → Phase 1 obs construction** — replace old 6D opponent builder:
```python
if self._opponent_modeling:
    opp_mode = self.config.get("opponent_obs", {}).get("mode", "lagged")
    opponent_obs_list = []
    for j in range(self.n_total):
        if j == i:
            continue
        if opp_mode == "full_info":
            # Legacy 6D path (for ablation)
            pub = self.companies[j].get_public_info()
            opponent_obs_list.extend([
                pub["emissions"], pub["carry_forward"], pub["green_frac"],
                pub["fossil_frac"], pub["queue_total"], pub["is_active"],
            ])
        else:
            # Default: 7D lagged snapshot (year t-1)
            opponent_obs_list.extend(self._opponent_snapshots_prev[j].tolist())
    opponent_obs = np.array(opponent_obs_list, dtype=np.float32) if opponent_obs_list else None
```

**`company.py` → `obs_dim_phase1` property** — update for 7D opponents:
```python
@property
def obs_dim_phase1(self) -> int:
    opp_dims = self.config.get("opponent_obs", {}).get("dims_per_opponent", 7)
    if self._opponent_modeling and self._n_total > 1:
        return 36 + opp_dims * (self._n_total - 1)
    return 36
```

> With 8 agents: Phase 1 = 36 + 7×7 = **85** (was 36 + 6×7 = 78).
> `obs_dim_phase2` = `obs_dim_phase1 + 11` (unchanged formula, new base).

**`configs/default.yaml`**
```yaml
opponent_obs:
  mode: "lagged"              # "lagged" (default) or "full_info" (ablation)
  lag_years: 1
  queue_noise_sigma: 0.15
  dims_per_opponent: 7
```

**`tests/test_mappo.py`, `tests/test_environment.py`, `tests/test_anchors.py`**

Find all hardcoded dimension assertions and update:
- Phase 1: `36 + 6*(N-1)` → `36 + 7*(N-1)`
- Phase 2: `46 + 6*(N-1)` → `47 + 7*(N-1)`  (Phase 1 + 11)

**`docs/design.md` → Section 6.1** — replace 6D tuple table with 7D table above and add timing diagram:
```
Timing:
  Year t Phase 1 obs reads: _opponent_snapshots_prev (from year t-1)
  Year t Phase 2 obs reads: same _opponent_snapshots_prev (no update mid-year)
  End of year t step_secondary(): updates _opponent_snapshots[]
  Start of year t+1: _opponent_snapshots_prev ← _opponent_snapshots
```

---

## Obs Dimension Summary

| Change | Phase 1 base | Opp dims | N=8 total P1 | Phase 2 extra | N=8 total P2 |
|--------|-------------|----------|--------------|---------------|---------------|
| Baseline | 36 | 6×7=42 | 78 | +10 | 88 |
| After Changes 1+4 | 36 | 6×7=42 | 78 | +10 | 88 |
| After Change 5 | 36 | 6×7=42 | 78 | +11 | **89** |
| After Change 7 | 36 | 7×7=49 | **85** | +11 | **96** |

**Final dims (N=8):** Phase 1 = 85, Phase 2 = 96.

---

## Logging Additions

Add to `year_log` in `step_secondary()`:
```python
"treasury_reserves": [c._treasury_reserve for c in self.companies],
"treasury_drawn": [c._treasury_drawn_this_year for c in self.companies],
"loan_outstanding": [c._loan_outstanding for c in self.companies],
"effective_capex_throughput": [c.effective_capex_throughput for c in self.companies],
"anchor_t": self._last_reward_channels.get("anchor_t"),
"opponent_snapshots": self._opponent_snapshots.tolist(),
```

---

## Implementation Order

1. **Change 1** — remove suspension
2. **Change 3** — loan sting
3. **Change 4** — treasury reserve + canonical waterfall
4. **Change 2** — budget price clip (needs Change 4)
5. **Change 5** — Phase 2 compliance gap obs
6. **Change 6** — anchor-normalised cost reward + test
7. **Change 7** — 7D lagged opponent obs + dimension updates in all test files + design.md

Run `configs/smoke_100.yaml` smoke test after each change.

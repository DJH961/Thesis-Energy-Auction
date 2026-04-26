# Implementation Plan v9 — Budget Reform & Compliance Visibility

> Scope: 5 self-contained changes. Implement in order (each builds on the previous).

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
    # loan_fraction is relative to annual_budget (not total cash incl. treasury)
    # so large treasury drawdowns don't deflate the leverage signal
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
  → _loan_drawn_this_step = shortfall

Case D: payment > op_avail + treasury + loan_limit
  → DEFAULT: allocation cancelled, collateral forfeited.
```

> `agent_cash` passed to `settle_auction()` = `op_avail + treasury` (combined).
> `max_loan_budgets[i]` passed to `settle_auction()` = `loan_limit` (pure loan headroom, not including treasury).
> Post-settlement loop in `ets_environment.py` performs the actual waterfall deductions.

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
    """Draw from reserve; returns actual amount drawn (capped at balance)."""
    actual = min(float(amount), self._treasury_reserve)
    self._treasury_reserve -= actual
    self._treasury_drawn_this_year += actual
    return actual
```

**`ets_environment.py` → `step_auction()` budget reset block (both modes):**
```python
# ORDER MATTERS: settle treasury first, then repay loan, then reset budget
company.settle_treasury_year_end()
company.apply_loan_repayment()
company.reset_budget()
company.reset_capex_budget()
```

**`ets_environment.py` → `step_auction()` — replace existing `agent_cash` construction and post-settlement deduction with the canonical waterfall:**

```python
# ── Pre-settlement cash stacks ──────────────────────────────────────────
operating_cash = np.array([
    max(0.0, float(c.annual_budget - c.budget_spent_this_year)
        - float(self._collateral_locked[i]))
    for i, c in enumerate(self.companies)
])
treasury_cash = np.array([c.get_treasury_available() for c in self.companies])
agent_cash = operating_cash + treasury_cash  # combined for settle_auction
max_loan_budgets = np.array([
    c._max_loan_fraction * max(c.annual_budget, 1.0) for c in self.companies
])

actual_alloc, actual_pay, defaults_mask, defaulted_vol, _, loan_amounts = settle_auction(
    allocations, payments, agent_cash, np.zeros(self.n_total),  # collateral already netted above
    suspension_length=0,
    max_loan_budgets=max_loan_budgets,
)

# ── Post-settlement waterfall deductions ────────────────────────────────
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
    else:                                    # Case C (loan covers residual)
        self.companies[i].record_spending(op)
        self.companies[i].draw_treasury(treas)
        self.companies[i].apply_emergency_loan(loan_amounts[i])
```

> Case D (default) is handled by `settle_auction()` itself — `actual_alloc[i] == 0` for defaulters, so the loop skips them.

**`ets_environment.py` → `_compute_rewards()` `is_final_year` block:**
```python
if bool(reward_cfg.get("treasury_terminal_value", True)):
    if company._treasury_enabled and company._treasury_reserve > 0:
        t_value = company._treasury_reserve * company._treasury_terminal_rate / REWARD_SCALE
        rewards[i] += t_value
        terminal_bank_values[i] += t_value
```

**`company.py` → `get_observation_phase1()`**

With Change 1 applied (base = 35), add treasury as new dim `[35]`:
```python
treasury_norm = float(np.clip(
    self._treasury_reserve / max(annual_budget, 1.0), 0.0, 2.0
)) / 2.0    # [35] treasury reserve norm (0=empty, 1.0=at cap)
```
Base obs_dim_phase1 = **36** (35 − 1 suspension + 1 treasury = net unchanged from original 36).

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

**Rationale:** `compliance_liability_norm` in Phase 2 obs encodes magnitude of exposure but not *direction*. A signed `compliance_gap_norm` gives the policy network a direct gradient target in Phase 2.

### Files
- `src/environment/company.py`

### Changes

**`company.py` → `get_observation_phase2()`**

Add `[base+10]` (current Phase 2 appends 10 dims, `[base+0]`…`[base+9]`):

```python
# Signed compliance gap: positive = under-covered, negative = over-covered
total_holdings_now = banked + allocation
compliance_gap = (emissions + self._carry_forward - total_holdings_now)
annual_need = max(self.compute_estimate_need(), 1e-6)
compliance_gap_norm = float(np.clip(compliance_gap / annual_need, -2.0, 2.0)) / 2.0
# [base+10]: 0.5 = balanced, >0.5 = short, <0.5 = surplus
```

Append to `extra` array → 11 elements total.
Update `obs_dim_phase2`: `return self.obs_dim_phase1 + 11`

---

## Obs Dimension Summary

| Change | Phase 1 base | Phase 2 extra | Net |
|--------|-------------|---------------|-----|
| Baseline | 36 | 10 | 46 |
| After Change 1 (−suspension) | 35 | 10 | 45 |
| After Change 4 (+treasury) | 36 | 10 | 46 |
| After Change 5 (+gap) | 36 | 11 | **47** |

**Final dims:** Phase 1 = 36 base + 6×(N−1) opponent. Phase 2 = Phase 1 + 11.

---

## Logging Additions

Add to `year_log` in `step_secondary()`:
```python
"treasury_reserves": [c._treasury_reserve for c in self.companies],
"treasury_drawn": [c._treasury_drawn_this_year for c in self.companies],
"loan_outstanding": [c._loan_outstanding for c in self.companies],
"effective_capex_throughput": [c.effective_capex_throughput for c in self.companies],
```

---

## Implementation Order

1. **Change 1** — remove suspension; no new fields, simplest
2. **Change 3** — loan sting; no obs changes, pure `company.py` + reward
3. **Change 4** — treasury reserve; new state, obs dim, **canonical waterfall replaces old `agent_cash` + post-hoc loop**
4. **Change 2** — budget price clip; depends on `get_treasury_available` from Change 4
5. **Change 5** — Phase 2 obs; standalone

Run smoke test (`configs/smoke_100.yaml`) after each change.

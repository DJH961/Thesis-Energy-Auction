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
                          - self.companies[i].budget_spent_this_year))
    # Include treasury reserve in available cash
    cash += self.companies[i].get_treasury_available()
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

**Rationale:** Currently only quantity is clipped when budget is tight. The bid price itself is unconstrained relative to budget — an agent can bid €200/t for 2Mt with €50M left, which is economically incoherent. Add a price ceiling derived from budget headroom.

### Files
- `src/environment/ets_environment.py`

### Changes

**`ets_environment.py` → `step_auction()`, after leverage gate, before collateral locking:**

```python
# Budget-envelope price clip: max bid price the agent can actually afford
# given remaining budget and minimum lot. Prevents price > budget / min_qty.
budget_price_clip = self.config["auction"].get("budget_price_clip", True)
if budget_price_clip:
    for i, company in enumerate(self.companies):
        if not self._is_agent_active(i):
            continue
        cash = max(1.0, float(company.annual_budget - company.budget_spent_this_year)
                   + company.get_treasury_available())
        bid_q = max(float(bid_actions[i, 1]), 1e-6)
        # Max supportable price = total available cash / bid quantity
        max_affordable_price = cash / bid_q
        # Soft clip: only apply if bid_price > 1.5x affordable (hard incoherence threshold)
        if bid_actions[i, 0] > 1.5 * max_affordable_price:
            bid_actions[i, 0] = float(np.clip(
                max_affordable_price,
                float(self.config["auction"]["price_min"]),
                float(self.config["auction"]["price_max"]),
            ))
```

**`configs/default.yaml`**
```yaml
auction:
  budget_price_clip: true
```

---

## Change 3: Improved Emergency Loan (Sting)

**Rationale:** Flat 8% interest on a fixed loan is painless for large budgets. Replace with leverage-scaled interest, immediate reward origination fee, and capex covenant squeeze during repayment.

### Files
- `src/environment/company.py`
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`company.py` → `apply_emergency_loan(shortfall)`**

Replace existing method:
```python
def apply_emergency_loan(self, shortfall: float) -> None:
    """Emergency loan with leverage-scaled interest rate (credit risk premium)."""
    loan_cfg = self.config.get("budget", {}).get("emergency_loan", {})
    base_rate = float(loan_cfg.get("interest_rate", 0.08))
    leverage_coef = float(loan_cfg.get("leverage_premium_coef", 0.25))
    leverage_exp = float(loan_cfg.get("leverage_premium_exp", 1.5))
    loan_fraction = shortfall / max(self.annual_budget, 1.0)
    effective_rate = base_rate + leverage_coef * (loan_fraction ** leverage_exp)
    principal_with_interest = shortfall * (1.0 + effective_rate)
    self._loan_outstanding += principal_with_interest
    self._loan_repayment_annual = (
        self._loan_outstanding / max(self._loan_repayment_years, 1)
    )
    self._years_under_loan = self._loan_repayment_years
    # Store loan fraction for capex squeeze
    self._last_loan_fraction = loan_fraction
```

Add `_last_loan_fraction = 0.0` to `__init__` and `reset()`.

**`company.py` → `compute_dynamic_budget()` (revenue_based mode) or wherever capex_throughput is used:**

Add capex covenant squeeze (call this at budget set time or expose via property):
```python
@property
def effective_capex_throughput(self) -> float:
    """Capex throughput reduced by debt covenant during loan repayment."""
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

In `ets_environment.py` → `step_auction()` investment block, replace all references to `company.capex_throughput` with `company.effective_capex_throughput`.

**`ets_environment.py` → `step_secondary()` → `_compute_rewards()`**

After `apply_emergency_loan` calls in `step_auction()`, record loan amounts. Then in `_compute_rewards()`, add origination fee to cost:
```python
# Loan origination fee (immediate sting at borrowing, not deferred)
loan_sting_coef = float(self.config.get("budget", {}).get(
    "emergency_loan", {}).get("origination_sting_coef", 0.30))
loan_draw_this_year = float(getattr(company, '_loan_drawn_this_step', 0.0))
if loan_draw_this_year > 0:
    loan_sting = (loan_draw_this_year / max(company.annual_budget, 1.0)) * loan_sting_coef
    total_cost += loan_sting * REWARD_SCALE  # unnorm before division below
```

Store `_loan_drawn_this_step` on company at loan origination; reset to 0 at `reset_budget()`.

**`configs/default.yaml`**
```yaml
budget:
  emergency_loan:
    enabled: true
    max_loan_fraction: 0.15
    interest_rate: 0.08
    repayment_years: 3
    leverage_premium_coef: 0.25   # NEW
    leverage_premium_exp: 1.5     # NEW
    capex_squeeze_floor: 0.50     # NEW: min capex throughput during repayment
    origination_sting_coef: 0.30  # NEW: immediate reward cost at borrowing
```

---

## Change 4: Corporate Treasury Reserve

**Rationale:** Unspent annual budget currently evaporates. Real utilities retain unspent compliance/operating budget in a liquidity reserve (subject to cap + opportunity cost decay) deployable in future crises — before the emergency loan.

**Economic grounding:**
- 60% retention: CFO liquidity policy; 40% returned to shareholders/operations
- 1.5× cap: ~18 months cash; above this = "overcapitalised" (Moody's liquidity metrics)
- 5% decay: idle capital opportunity cost (EU utility WACC ~7–9%)
- Reserve depletes before loan: Myers (1984) pecking-order theory
- Terminal value at 30%: discounted going-concern balance sheet value

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

**`company.py` → `reset()`**: add `self._treasury_reserve = 0.0` and `self._treasury_drawn_this_year = 0.0`

**`company.py` → new methods:**
```python
def settle_treasury_year_end(self) -> None:
    """Roll unspent budget into treasury reserve with cap and decay."""
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

**`ets_environment.py` → `step_auction()` budget reset block (both `revenue_based` and `fixed`):**
```python
# Call BEFORE apply_loan_repayment and reset_budget:
company.settle_treasury_year_end()
company.apply_loan_repayment()
company.reset_budget()
company.reset_capex_budget()
```

**`ets_environment.py` → `step_auction()` `agent_cash` array (before `settle_auction` call):**
```python
agent_cash = np.array([
    max(0.0, float(c.annual_budget - c.budget_spent_this_year))
    + c.get_treasury_available()
    for c in self.companies
])
```

**`ets_environment.py` → `step_auction()` after `settle_auction` returns:**
For each agent where payment exceeded raw budget, deduct from treasury:
```python
for i in range(self.n_total):
    if allocations[i] > 0 and payments[i] > 0:
        raw_budget = max(0.0, float(self.companies[i].annual_budget
                                    - self.companies[i].budget_spent_this_year))
        overflow = max(0.0, float(payments[i]) - raw_budget)
        if overflow > 0:
            self.companies[i].draw_treasury(overflow)
```

**`ets_environment.py` → `_compute_rewards()` `is_final_year` block:**
```python
if bool(reward_cfg.get("treasury_terminal_value", True)):
    if company._treasury_enabled and company._treasury_reserve > 0:
        t_value = company._treasury_reserve * company._treasury_terminal_rate / REWARD_SCALE
        rewards[i] += t_value
        terminal_bank_values[i] += t_value
```

**`company.py` → `get_observation_phase1()`**

Add as new dim after existing base dims (dim index depends on whether suspension dim was removed in Change 1 — with Change 1 applied, base is 35 dims; this becomes dim `[35]`):
```python
treasury_norm = float(np.clip(
    self._treasury_reserve / max(annual_budget, 1.0), 0.0, 2.0
)) / 2.0    # [35] treasury reserve norm (0=empty, 1.0=at cap)
```

Update `obs_dim_phase1` base to `36` (35 base - 1 suspension + 1 treasury = 35... wait: original 36 - 1 suspension + 1 treasury = **36**, unchanged base count).

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

**Rationale:** `compliance_liability_norm` in Phase 2 obs encodes magnitude of exposure but not *direction*. Agents must mentally derive the gap from 4+ inputs. A signed `compliance_gap_norm` gives the policy network a direct gradient target.

### Files
- `src/environment/company.py`

### Changes

**`company.py` → `get_observation_phase2()`**

Add one new dimension. Current Phase 2 appends 10 dims (`[base+0]` … `[base+9]`). Add `[base+10]`:

```python
# compliance_gap_norm: signed gap = (emissions + carry_forward - total_holdings) / annual_need
# Positive = under-covered (need to buy). Negative = over-covered (have surplus).
total_holdings_now = banked + allocation  # pre-secondary-trade holdings
compliance_gap = (emissions + self._carry_forward - total_holdings_now)
annual_need = max(self.compute_estimate_need(), 1e-6)
compliance_gap_norm = float(np.clip(compliance_gap / annual_need, -2.0, 2.0)) / 2.0
# [base+10] signed compliance gap (0.5=balanced, >0.5=short, <0.5=long)
```

Append `compliance_gap_norm` to `extra` array (making it 11 elements).

Update `obs_dim_phase2` property: `return self.obs_dim_phase1 + 11` (was `+ 10`).

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

1. **Change 1** — removes suspension state; simplest, no new fields
2. **Change 3** — loan sting (no obs changes, pure company.py + reward)
3. **Change 4** — treasury reserve (new state + obs dim shift from Change 1)
4. **Change 2** — budget price clip (depends on `get_treasury_available` from Change 4)
5. **Change 5** — Phase 2 obs (standalone, just add dim)

Run smoke test (`configs/smoke_100.yaml`) after each change.

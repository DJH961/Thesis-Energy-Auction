# Implementation Plan v8.1 — Budget Reform & Compliance Visibility

Fixes: section order matches implementation order; lag-buffer semantics corrected; collateral-aware price clipping; `_last_compliance_gaps` fully specified; loan sting reframed; treasury dim comment corrected; smoke test acceptance criteria added.
>
> **Scope: 8 changes. Sections are ordered to match implementation order — implement top to bottom.**

---

## Change 1: Remove Suspension → Budget-Based Gate

**Rationale:** Suspension is a blunt market-exclusion mechanism with no real-world ETS analog. Replace with a budget-based freeze: agents who cannot cover collateral simply have their bid quantity clipped to zero.

### Files
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`ets_environment.py` → `step_auction()`** — remove suspension block:
```python
# DELETE:
for i in range(self.n_total):
    if self._suspension_remaining[i] > 0:
        bid_actions[i, 1] = 0.0
        self._suspension_remaining[i] -= 1
```

Replace with budget-based gate (insert at same location):
```python
for i in range(self.n_total):
    if not self._is_agent_active(i):
        continue
    cash = max(0.0, float(self.companies[i].annual_budget
                          - self.companies[i].budget_spent_this_year)
               + self.companies[i].get_treasury_available())
    bid_p = float(bid_actions[i, 0])
    if bid_p > 1e-6 and cash < bid_p * bid_actions[i, 1] * 0.10:
        bid_actions[i, 1] = 0.0
```

**`ets_environment.py` → `reset()`**: remove `self._suspension_remaining = np.zeros(...)` and all logging of `suspension_remaining_list`, `suspended_agents`.

**`company.py` → `get_observation_phase1()`**: remove dim `[28]` (`suspension_remaining_norm`), shift subsequent dims down by 1. Update `obs_dim_phase1` base: 36 → 35 (temporary; restored to 36 after Change 4 adds treasury dim).

**`configs/default.yaml`**:
```yaml
auction:
  suspension_length: 0
```

---

## Change 3: Improved Emergency Loan (Sting)

> **Note:** Numbered Change 3 to preserve cross-reference consistency. Implement second.

**Rationale:** Replace flat 8% loan with leverage-scaled interest plus a small immediate reward-shaping sting and capex covenant squeeze.

**Design constraint — no double penalty:** The loan already imposes three consequences: (1) leverage-scaled interest on principal, (2) multi-year budget squeeze via `_loan_repayment_annual`, (3) capex throughput squeeze. The origination sting is a **lightweight RL salience term only** — it is not an additional accounting charge. Default `origination_sting_coef` is therefore set to `0.07` (not 0.30). Raising it risks making the policy excessively loan-averse beyond what the economic model justifies.

> ⚠️ The loan triggers ONLY after operating budget AND treasury are both exhausted (see Change 4 waterfall). `shortfall` passed to `apply_emergency_loan()` is always the true residual.

### Files
- `src/environment/company.py`
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`company.py` → `apply_emergency_loan(shortfall)`**:
```python
def apply_emergency_loan(self, shortfall: float) -> None:
    """Emergency loan — leverage-scaled rate. Call only after treasury is exhausted."""
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
    self._last_loan_fraction = loan_fraction
    self._loan_drawn_this_step = shortfall
```

Add to `__init__` and `reset()`: `self._last_loan_fraction = 0.0`, `self._loan_drawn_this_step = 0.0`
Add to `reset_budget()`: `self._loan_drawn_this_step = 0.0`

**`company.py` → new property:**
```python
@property
def effective_capex_throughput(self) -> float:
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

**`ets_environment.py` → `_compute_rewards()`** — sting is a small salience signal, not an accounting cost:
```python
# Origination sting: small immediate RL signal only. Economic cost already
# fully captured by principal+interest and multi-year repayment squeeze.
loan_sting_coef = float(self.config.get("budget", {}).get(
    "emergency_loan", {}).get("origination_sting_coef", 0.07))
loan_draw_this_year = float(getattr(company, '_loan_drawn_this_step', 0.0))
if loan_draw_this_year > 0:
    loan_sting = (loan_draw_this_year / max(company.annual_budget, 1.0)) * loan_sting_coef
    total_cost += loan_sting * REWARD_SCALE
```

**`configs/default.yaml`**:
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
    origination_sting_coef: 0.07   # small salience term; not additional accounting cost
```

---

## Change 4: Corporate Treasury Reserve

**Rationale:** Unspent annual budget currently evaporates. Real utilities retain a liquidity reserve deployable in future crises — before the emergency loan triggers.

**Economic grounding:** 60% retention (CFO liquidity policy); 1.5× cap (Moody's ~18-month cash); 5% decay (EU utility WACC ~7–9%); treasury before loan (Myers 1984 pecking order); terminal value at 30 cents on the dollar.

### Settlement Waterfall (canonical — all cash logic uses this order)

```
Inputs per winning agent i:
  op_avail   = annual_budget − budget_spent_this_year − collateral_locked[i]
  treasury   = get_treasury_available()
  loan_limit = max_loan_fraction × annual_budget

Case A: payment ≤ op_avail
  → record_spending(payment). Treasury untouched. No loan.

Case B: op_avail < payment ≤ op_avail + treasury
  → record_spending(op_avail)
  → draw_treasury(payment − op_avail)

Case C: op_avail + treasury < payment ≤ op_avail + treasury + loan_limit
  → record_spending(op_avail)
  → draw_treasury(treasury)            [full drain]
  → apply_emergency_loan(payment − op_avail − treasury)

Case D: payment > op_avail + treasury + loan_limit
  → DEFAULT: allocation cancelled, collateral forfeited.
```

> `agent_cash` passed to `settle_auction()` = `op_avail + treasury`.
> `max_loan_budgets[i]` = `loan_limit` (pure loan headroom only — treasury already in agent_cash).
> Case D is handled by `settle_auction()` — `actual_alloc[i] == 0`, post-settlement loop skips it.

### Files
- `src/environment/company.py`
- `src/environment/ets_environment.py`
- `configs/default.yaml`

### Changes

**`company.py` → `__init__()`**:
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
    """Roll unspent budget into treasury. Call BEFORE apply_loan_repayment() and reset_budget()."""
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

**`ets_environment.py` → budget reset block — ORDER MATTERS:**
```python
company.settle_treasury_year_end()   # 1st: capture unspent before reset
company.apply_loan_repayment()       # 2nd
company.reset_budget()               # 3rd
company.reset_capex_budget()         # 4th
```

**`ets_environment.py` → replace `agent_cash` construction and post-settlement deduction:**
```python
# ── Pre-settlement cash stacks (collateral already netted out of operating_cash)
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
    allocations, payments, agent_cash,
    np.zeros(self.n_total),   # collateral already netted above; pass zeros here
    suspension_length=0,
    max_loan_budgets=max_loan_budgets,
)

# ── Post-settlement waterfall (Cases A / B / C)
for i in range(self.n_total):
    if actual_alloc[i] < 1e-9:
        continue   # non-winner or defaulter (Case D)
    payment = float(actual_pay[i])
    op = float(operating_cash[i])
    treas = float(treasury_cash[i])
    if payment <= op:                  # Case A
        self.companies[i].record_spending(payment)
    elif payment <= op + treas:        # Case B
        self.companies[i].record_spending(op)
        self.companies[i].draw_treasury(payment - op)
    else:                              # Case C
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

**`company.py` → `get_observation_phase1()`** — add treasury as dim `[35]` (after Change 1 removed suspension):
```python
treasury_norm = float(np.clip(
    self._treasury_reserve / max(annual_budget, 1.0), 0.0, 2.0
)) / 2.0    # [35] treasury reserve norm
```

> **Dimension note:** Phase 1 base = 36. Derivation: original 36 − 1 (suspension removed in Change 1) + 1 (treasury added here) = **36**. Net count unchanged.

**`configs/default.yaml`**:
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

## Change 2: Budget Envelope Clipping (Price + Quantity)

> **Note:** Numbered Change 2 to preserve cross-reference consistency. Implement fourth (after Change 4).

**Rationale:** Bid price is unconstrained relative to budget. Add a soft price ceiling using the same collateral-aware cash basis as the settlement waterfall, ensuring consistency across all cash logic.

> ⚠️ **Depends on Change 4** (`get_treasury_available`, `_collateral_locked`). Do not implement before Change 4.

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
        # Use same collateral-aware basis as the settlement waterfall:
        # operating_remaining excludes already-locked collateral
        op_remaining = max(0.0, float(company.annual_budget - company.budget_spent_this_year)
                           - float(self._collateral_locked[i]))
        cash = op_remaining + company.get_treasury_available()
        cash = max(cash, 1.0)
        bid_q = max(float(bid_actions[i, 1]), 1e-6)
        max_affordable_price = cash / bid_q
        # Soft clip at 1.5x: small buffer for any residual collateral rounding
        if bid_actions[i, 0] > 1.5 * max_affordable_price:
            bid_actions[i, 0] = float(np.clip(
                max_affordable_price,
                float(self.config["auction"]["price_min"]),
                float(self.config["auction"]["price_max"]),
            ))
```

**`configs/default.yaml`**:
```yaml
auction:
  budget_price_clip: true
```

---

## Change 5: Phase 2 Compliance Gap Observation

**Rationale:** `compliance_liability_norm` encodes magnitude but not direction. A signed `compliance_gap_norm` gives the policy a direct gradient target.

### Files
- `src/environment/company.py`

### Changes

**`company.py` → `get_observation_phase2()`** — append `[base+10]`:
```python
total_holdings_now = banked + allocation
compliance_gap = (emissions + self._carry_forward - total_holdings_now)
annual_need = max(self.compute_estimate_need(), 1e-6)
compliance_gap_norm = float(np.clip(compliance_gap / annual_need, -2.0, 2.0)) / 2.0
# [base+10]: 0.5=balanced, >0.5=short (under-covered), <0.5=surplus
```

Append to `extra` → 11 elements total.
Update `obs_dim_phase2`: `return self.obs_dim_phase1 + 11`

---

## Change 6: Anchor-Normalised Cost Reward

**Rationale:** `total_non_penalty_cost / REWARD_SCALE` uses a static denominator that penalises late-year agents more harshly for the same spending efficiency as early-year agents. Normalise cost against the time-varying fundamental allowance value so equal-efficiency spending yields equal `cost_norm` regardless of year. Penalty is intentionally left on the static scale — non-compliance cost should escalate nominally with inflation.

### Files
- `src/environment/ets_environment.py`
- `configs/default.yaml`
- `tests/test_rewards.py`

### Changes

**`ets_environment.py` → top-level import:**
```python
from src.utils.price_anchor import compute_fundamental_anchor
```

**`ets_environment.py` → `_compute_rewards()` — replace `cost_norm` line:**
```python
anchor_normalize = bool(reward_cfg.get("anchor_normalize_cost_only", True))
if anchor_normalize:
    anchor_t = compute_fundamental_anchor(self.current_year, self.config)
    estimated_need = max(company.compute_estimate_need(), 1e-6)
    cost_denominator = max(anchor_t * estimated_need, 1.0)
    cost_norm = total_non_penalty_cost / cost_denominator
else:
    cost_norm = total_non_penalty_cost / REWARD_SCALE
# Penalty unchanged: penalty_norm = shortfall * eff_penalty_t / REWARD_SCALE
```

Store for diagnostics:
```python
self._last_reward_channels["anchor_t"] = anchor_t if anchor_normalize else None
```

**`configs/default.yaml`**:
```yaml
reward:
  anchor_normalize_cost_only: true
```

**`tests/test_rewards.py` — add:**
```python
def test_anchor_normalised_cost_symmetry():
    """Equal-efficiency spend → equal cost_norm. Equal shortfall → higher penalty_norm at yr 10."""
    from src.utils.price_anchor import compute_fundamental_anchor
    cfg = load_default_config()
    need = 5.0
    for yr in [0, 10]:
        anchor = compute_fundamental_anchor(yr, cfg)
        cost_norm = (anchor * need) / max(anchor * need, 1.0)
        assert abs(cost_norm - 1.0) < 1e-6, f"Year {yr}: cost_norm != 1.0"
    REWARD_SCALE = cfg["auction"]["price_max"]
    shortfall = 1.0
    p0 = shortfall * cfg["penalty"]["rate"] / REWARD_SCALE
    p10 = shortfall * cfg["penalty"]["rate"] * (1 + cfg["penalty"]["inflation_rate"]) ** 10 / REWARD_SCALE
    assert p10 > p0
```

---

## Change 7: 7D Lagged Opponent Observation

**Rationale:** The current 6D opponent tuple lacks bank position, secondary market behaviour, and lagged compliance gap — all strategically relevant signals. The 1-year lag is essential: Phase 1 cannot observe opponent actions from the current year.

### New 7D tuple

| Dim | Signal | Normalisation |
|-----|--------|---------------|
| 0 | `verified_emissions` | `/ 10.0` |
| 1 | `green_frac` | raw [0,1] |
| 2 | `fossil_frac` | raw [0,1] |
| 3 | `queue_signal` + N(0,σ) | clipped [0,1] |
| 4 | `bank_norm` = holdings / annual_need | clipped [0,3] / 3 |
| 5 | `net_secondary_norm` = (sec_bought − sec_sold) / annual_need | clipped [−1,1] |
| 6 | `lagged_compliance_gap_norm` = (prior_emissions − surrendered) / annual_need | clipped [−1,1] |

All dims read from `_opponent_snapshots_prev` (year t−1 data).

### Lag buffer protocol (two-buffer)

```
_opponent_snapshots_prev  — holds year t−1 data; read by Phase 1 obs
_opponent_snapshots       — holds year t data; written at end of step_secondary()

At end of step_secondary():
  1. self._opponent_snapshots_prev = self._opponent_snapshots.copy()  ← MUST happen BEFORE writing new data
  2. [write new year-t data into self._opponent_snapshots[i]]

At reset(): both buffers initialised with burn-in priors; _prev = current.copy()
```

> ⚠️ **Critical ordering:** `_opponent_snapshots_prev` must be assigned from the *old* snapshot **before** overwriting `_opponent_snapshots` with current-year data. Assigning after overwrites the lag and collapses both buffers to the same year.

### Files
- `src/environment/ets_environment.py`
- `src/environment/company.py`
- `configs/default.yaml`
- `tests/test_mappo.py`, `tests/test_environment.py`, `tests/test_anchors.py`
- `docs/design.md`

### Changes

**`ets_environment.py` → `__init__()`**:
```python
self._opponent_snapshots      = np.zeros((self.n_total, 7), dtype=float)
self._opponent_snapshots_prev = np.zeros((self.n_total, 7), dtype=float)
self._sec_bought              = np.zeros(self.n_total, dtype=float)
self._sec_sold                = np.zeros(self.n_total, dtype=float)
self._last_compliance_gaps    = np.zeros(self.n_total, dtype=float)
```

**`ets_environment.py` → `reset()`**:
```python
self._sec_bought           = np.zeros(self.n_total, dtype=float)
self._sec_sold             = np.zeros(self.n_total, dtype=float)
self._last_compliance_gaps = np.zeros(self.n_total, dtype=float)
for i, c in enumerate(self.companies):
    self._opponent_snapshots[i] = [
        c.compute_emissions() / 10.0,
        c.green_frac,
        c.fossil_frac,
        float(sum(item["frac_delta"] for item in c._construction_queue)),
        0.5,  # bank_norm: neutral prior
        0.0,  # net_secondary_norm: neutral prior
        0.0,  # lagged_compliance_gap_norm: neutral prior
    ]
self._opponent_snapshots_prev = self._opponent_snapshots.copy()
```

**`ets_environment.py` → `step_secondary()` secondary clearing** — track volumes:
```python
# After secondary trade fills, record per-agent volumes:
self._sec_bought[i] += float(secondary_buy_fill[i])
self._sec_sold[i]   += float(secondary_sell_fill[i])
```

**`ets_environment.py` → `step_secondary()` after compliance settled** — record gaps:
```python
# After compliance surrender:
self._last_compliance_gaps[i] = float(realized_emissions[i] - surrendered[i])
```

**`ets_environment.py` → END of `step_secondary()` — two-buffer update:**
```python
# ── Opponent snapshot update (two-buffer; _prev MUST be assigned before current is overwritten)
self._opponent_snapshots_prev = self._opponent_snapshots.copy()   # save year t−1
opp_cfg = self.config.get("opponent_obs", {})
queue_sigma = float(opp_cfg.get("queue_noise_sigma", 0.15))
for i, c in enumerate(self.companies):
    need = max(c.compute_estimate_need(), 1e-6)
    queue_raw = float(sum(item["frac_delta"] for item in c._construction_queue))
    queue_noisy = float(np.clip(queue_raw + self.rng.normal(0, queue_sigma), 0.0, 1.0))
    bank_norm = float(np.clip(self.holdings[i] / need, 0.0, 3.0)) / 3.0
    net_sec = float(np.clip((self._sec_bought[i] - self._sec_sold[i]) / need, -1.0, 1.0))
    lag_gap = float(np.clip(self._last_compliance_gaps[i] / need, -1.0, 1.0))
    self._opponent_snapshots[i] = [
        c.compute_emissions() / 10.0,
        c.green_frac, c.fossil_frac,
        queue_noisy, bank_norm, net_sec, lag_gap,
    ]
# Reset per-year secondary counters for next year
self._sec_bought[:] = 0.0
self._sec_sold[:]   = 0.0
```

**`ets_environment.py` → Phase 1 obs construction — mode-conditional opponent builder:**
```python
if self._opponent_modeling:
    opp_mode = self.config.get("opponent_obs", {}).get("mode", "lagged")
    opponent_obs_list = []
    for j in range(self.n_total):
        if j == i:
            continue
        if opp_mode == "full_info":
            pub = self.companies[j].get_public_info()
            opponent_obs_list.extend([
                pub["emissions"], pub["carry_forward"], pub["green_frac"],
                pub["fossil_frac"], pub["queue_total"], pub["is_active"],
            ])
        else:  # "lagged" (default)
            opponent_obs_list.extend(self._opponent_snapshots_prev[j].tolist())
    opponent_obs = np.array(opponent_obs_list, dtype=np.float32) if opponent_obs_list else None
```

**`company.py` → `obs_dim_phase1` property:**
```python
@property
def obs_dim_phase1(self) -> int:
    opp_dims = self.config.get("opponent_obs", {}).get("dims_per_opponent", 7)
    if self._opponent_modeling and self._n_total > 1:
        return 36 + opp_dims * (self._n_total - 1)
    return 36
# obs_dim_phase2 = obs_dim_phase1 + 11 (unchanged formula)
```

**`configs/default.yaml`**:
```yaml
opponent_obs:
  mode: "lagged"           # "lagged" (default) | "full_info" (ablation — restores 6D)
  lag_years: 1
  queue_noise_sigma: 0.15
  dims_per_opponent: 7
```

**Test dimension updates** — in `tests/test_mappo.py`, `tests/test_environment.py`, `tests/test_anchors.py`:
- `36 + 6*(N-1)` → `36 + 7*(N-1)`
- `46 + 6*(N-1)` → `47 + 7*(N-1)`

**`docs/design.md` → Section 6.1**: replace 6D table with 7D table above. Add timing note:
```
Year t Phase 1 obs reads _opponent_snapshots_prev (year t−1).
End of year t step_secondary() saves prev, then writes new current snapshot.
Year t+1 Phase 1 obs reads updated _opponent_snapshots_prev (year t).
```

---

## Change 8: Project Version Bump + Documentation Sync

**Rationale:** The combined scope of Changes 1–7 constitutes a minor version increment (new features, backward-incompatible observation dimensions). Bump from `8.0.1` → `8.1.0`.

> ⚠️ Existing trained checkpoints are **shape-incompatible** with the new observation dimensions unless `opponent_obs.mode: full_info` is set and reward config matches legacy settings. Record this in the migration note.

### Files
- `pyproject.toml` or `setup.py` (wherever version string lives)
- `src/__init__.py` (if version is also defined there)
- `README.md`
- `docs/design.md`
- `docs/changelog.md`

### Changes

**Version string**: `8.0.1` → `8.1.0` in all version-bearing files.

**`README.md`** — update:
- Observation dimensions: Phase 1 = `36 + 7*(N−1)`, Phase 2 = Phase 1 + 11
- Reward design: `cost_norm` uses anchor normalisation; `penalty_norm` uses static `REWARD_SCALE`
- Treasury/loan waterfall: operating → treasury → loan → default
- New config keys: `opponent_obs`, `budget.treasury_reserve`, `budget.emergency_loan` (full), `reward.anchor_normalize_cost_only`

**`docs/design.md`** — update:
- Section 6.1: 7D opponent tuple definition and timing diagram (done in Change 7)
- Reward design section: asymmetric normalisation rationale
- Budget/cash section: waterfall diagram

**`docs/changelog.md`** — prepend entry:
```markdown
## [8.1.0] — 2026-04-27
### Added
- Corporate treasury reserve (unspent budget retention, cap, decay, terminal value)
- Leverage-scaled emergency loan with capex covenant squeeze
- Anchor-normalised cost reward (`cost_norm` time-stable; `penalty_norm` nominally inflating)
- 7D lagged opponent observation with 1-year lag buffer
- Phase 2 signed compliance gap dimension
- Budget price clipping (soft, collateral-aware)

### Changed
- Suspension mechanism replaced by budget-based bid gate
- Settlement logic unified under canonical waterfall (operating → treasury → loan → default)
- Opponent obs default mode: `lagged` (7D); `full_info` (6D) available for ablation

### Breaking
- Phase 1 obs: `36 + 7*(N−1)` dims (was `36 + 6*(N−1)`)
- Phase 2 obs: Phase 1 + 11 dims (was Phase 1 + 10)
- Old checkpoints incompatible unless `opponent_obs.mode: full_info` and legacy reward config
```

---

## Obs Dimension Summary

| Stage | Phase 1 base | Opp dims (N=8) | P1 total | P2 extra | P2 total |
|-------|-------------|----------------|----------|----------|----------|
| Baseline (v8.0.1) | 36 | 6×7=42 | 78 | +10 | 88 |
| After Changes 1+4 | 36 | 6×7=42 | 78 | +10 | 88 |
| After Change 5 | 36 | 6×7=42 | 78 | +11 | 89 |
| After Change 7 (v8.1.0) | 36 | 7×7=49 | **85** | +11 | **96** |

---

## Smoke Test Acceptance Criteria

After each change, run `python -m pytest configs/smoke_100.yaml` (or equivalent). A change **passes** only if all of:

1. Episode completes 100 steps without exception or NaN/inf in obs, rewards, or prices.
2. `obs.shape == (n_agents, obs_dim_phase1)` at Phase 1; `(n_agents, obs_dim_phase2)` at Phase 2.
3. `treasury_reserve[i] >= 0` for all agents at all steps.
4. `loan_draw[i] <= max_loan_fraction * annual_budget[i]` at all steps.
5. After first year transition: `_opponent_snapshots_prev != _opponent_snapshots` (lag is non-trivial).
6. `_last_reward_channels["anchor_t"]` is a positive finite float when `anchor_normalize_cost_only: true`.
7. No agent's `budget_spent_this_year` exceeds `annual_budget + treasury_available + max_loan`.

---

## Logging Additions

Add to `year_log` in `step_secondary()`:
```python
"treasury_reserves":          [c._treasury_reserve for c in self.companies],
"treasury_drawn":             [c._treasury_drawn_this_year for c in self.companies],
"loan_outstanding":           [c._loan_outstanding for c in self.companies],
"effective_capex_throughput": [c.effective_capex_throughput for c in self.companies],
"anchor_t":                   self._last_reward_channels.get("anchor_t"),
"opponent_snapshots":         self._opponent_snapshots.tolist(),
```

---

## Implementation Order

| Step | Change | Key dependency |
|------|--------|----------------|
| 1 | Change 1 — remove suspension | none |
| 2 | Change 3 — loan sting | none |
| 3 | Change 4 — treasury + waterfall | Change 3 (loan called from waterfall) |
| 4 | Change 2 — budget price clip | Change 4 (`_collateral_locked`, `get_treasury_available`) |
| 5 | Change 5 — Phase 2 gap obs | none |
| 6 | Change 6 — anchor reward + test | none |
| 7 | Change 7 — 7D lagged opponent obs | none |
| 8 | Change 8 — version bump + docs | all above |

Run smoke acceptance checklist after each step.

"""
company.py
==========
Represents a single energy company with a technology-specific generation mix.

Each company produces a fixed 10 TWh/year from a mix of 5 technologies:
  [coal, gas, onshore_wind, offshore_wind, solar]

Key features:
  - Technology-specific emission factors, CapEx, capacity factors, and deployment delays
  - Construction queue: investments take technology-specific years to complete
  - Greening-only constraint: can only add renewables, fossil can only decrease
  - Real-data-grounded investment costs (CapEx × capacity needed)
  - Operational costs differ by technology (fuel + O&M)
  - Decommissioning costs when retiring fossil capacity
  - Risk model for investment failure
  - MAC fuel-switching: temporary coal→gas dispatch switching at marginal cost
  - Carry-forward obligations: non-compliance shortfall carries to next year
  - Construction jitter: Poisson delays and probabilistic cancellation with partial capex recovery
  - CF noise: capacity factor noise affects realized emissions each year
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional

N_TECHS = 5
TECH_NAMES = ["coal", "gas", "onshore_wind", "offshore_wind", "solar"]
TECH_IDX = {name: i for i, name in enumerate(TECH_NAMES)}
BUILDABLE_INDICES = [2, 3, 4]  # onshore_wind, offshore_wind, solar


class Company:

    def __init__(self, agent_id: int, config: dict, initial_mix: List[float], rng):
        self.agent_id = agent_id
        self.rng = rng
        self.config = config
        # n_total includes learning agents + bot agents (for opponent modeling obs dimension)
        self._n_total = (config["companies"]["n_agents"]
                         + config["companies"].get("n_bot_agents", 0))
        self._opponent_modeling = config.get("opponent_modeling", {}).get("enabled", False)

        co_cfg = config["companies"]
        tech_cfg = config["technologies"]
        inv_cfg = config["investment"]
        risk_cfg = config["risk"]
        pen_cfg = config["penalty"]

        self.output_twh = co_cfg["output_twh"]
        self.output_mwh = self.output_twh * 1e6  # 10 TWh = 10,000,000 MWh

        # Technology parameters (arrays of length 5)
        self.emission_factors = np.array(tech_cfg["emission_factors"], dtype=np.float64)  # tCO2/MWh
        self.capex = np.array(tech_cfg["capex"], dtype=np.float64)  # €/kW
        self.capacity_factors = np.array(tech_cfg["capacity_factors"], dtype=np.float64)
        self.deploy_delays = np.array(tech_cfg["deploy_delays"], dtype=np.int32)  # years (base)
        self.operational_costs = np.array(tech_cfg["operational_costs"], dtype=np.float64)  # €/MWh
        self.decommission_costs = np.array(tech_cfg["decommission_costs"], dtype=np.float64)  # €/kW
        self.is_green = np.array(tech_cfg["is_green"], dtype=bool)
        self.is_buildable = np.array(tech_cfg["is_buildable"], dtype=bool)

        # Investment parameters
        self.max_invest_frac = inv_cfg["max_invest_frac"]
        self.convexity_alpha = inv_cfg["convexity_alpha"]
        self.penalty_rate = pen_cfg["rate"]
        self._inflation_rate = pen_cfg.get("inflation_rate", 0.0)
        self._inflation_random_std = float(max(0.0, pen_cfg.get("inflation_random_std", 0.0)))
        self._inflation_random_window = float(max(0.0, pen_cfg.get("inflation_random_window", 0.0)))
        self._inflation_rates_by_year = {}
        self._inflation_factor_by_year = {0: 1.0}

        # Risk curve parameters
        self.p_fail_min = risk_cfg["p_fail_min"]
        self.p_fail_max = risk_cfg["p_fail_max"]
        self.p_fail_alpha = risk_cfg["p_fail_alpha"]
        self.exp_discount = risk_cfg["experience_discount"]
        self.exp_threshold = risk_cfg["experience_threshold"]

        # Reward weights
        w = co_cfg["reward_weights"][agent_id]
        self.w_cost = w[0]   # alpha: cost weight
        self.w_green = w[1]  # beta: emissions intensity weight

        # Budget constraint
        budget_cfg = config.get("budget", {})
        budgets = budget_cfg.get("annual_budgets", [1e9] * 4)
        self.annual_budget = budgets[agent_id] if agent_id < len(budgets) else 1e9
        self.overspend_coef = budget_cfg.get("overspend_penalty_coef", 2.0)
        self.budget_spent_this_year = 0.0
        self.prev_invest_frac = 0.0

        # Revenue-based dynamic budget
        self.debt_headroom = 0.0
        debt_headrooms = budget_cfg.get("debt_headrooms", [])
        if agent_id < len(debt_headrooms):
            self.debt_headroom = float(debt_headrooms[agent_id])
        self._budget_ema_alpha = float(budget_cfg.get("budget_ema_alpha", 0.3))
        self._budget_ema: Optional[float] = None
        self._dynamic_budget_ceiling_mult = float(
            budget_cfg.get("dynamic_budget_ceiling_multiplier", 1e9)
        )

        # Emergency loan facility
        loan_cfg = budget_cfg.get("emergency_loan", {})
        self._loan_enabled = bool(loan_cfg.get("enabled", False))
        self._max_loan_fraction = float(loan_cfg.get("max_loan_fraction", 0.15))
        self._loan_interest_rate = float(loan_cfg.get("interest_rate", 0.08))
        self._loan_repayment_years = int(loan_cfg.get("repayment_years", 3))
        self._loan_outstanding = 0.0
        self._loan_repayment_annual = 0.0
        self._years_under_loan = 0
        self._last_loan_fraction = 0.0
        self._loan_drawn_this_step = 0.0

        # Treasury reserve
        treasury_cfg = budget_cfg.get("treasury_reserve", {})
        self._treasury_enabled = bool(treasury_cfg.get("enabled", False))
        self._treasury_reserve = 0.0
        self._treasury_cap_mult = float(treasury_cfg.get("cap_mult", 1.5))
        self._treasury_retention = float(treasury_cfg.get("savings_retention_rate", 0.60))
        self._treasury_decay = float(treasury_cfg.get("decay_rate", 0.05))
        self._treasury_terminal_rate = float(treasury_cfg.get("terminal_value_rate", 0.30))
        self._treasury_drawn_this_year = 0.0

        # Capex throughput constraint (organizational construction spend cap)
        capex_tp = budget_cfg.get("capex_throughputs", [])
        self.capex_throughput = capex_tp[agent_id] if agent_id < len(capex_tp) else 1e9
        self.capex_overspend_coef = budget_cfg.get("capex_overspend_coef", 1.0)
        self.capex_spent_this_year = 0.0
        # Revenue-linked capex throughput scaler (set via update_capex_revenue_factor)
        self._baseline_revenue_for_capex = None
        self._revenue_capex_factor = 1.0

        # Green finance (optional): annual loan headroom for green capex only.
        gf_cfg = config.get("green_finance", {})
        self._gf_enabled = bool(gf_cfg.get("enabled", False))
        self._gf_loan_budget = float(gf_cfg.get("loan_budget_boost", 0.0))
        self._gf_interest_rate = float(gf_cfg.get("loan_interest_rate", 0.05))
        self._gf_capex_boost = float(gf_cfg.get("capex_throughput_boost", 0.0))
        self.green_loan_utilized = 0.0

        # Construction jitter config
        jitter_cfg = config.get("construction_jitter", {})
        self._jitter_enabled = jitter_cfg.get("enabled", False)
        poisson_lambdas = jitter_cfg.get("poisson_lambdas", [1.0, 1.0, 2.0, 3.0, 1.5])
        self._jitter_lambdas = np.array(poisson_lambdas, dtype=np.float64)
        # Per-tech construction-phase cancellation. p_cancel_per_tech (length 5)
        # takes precedence; the scalar p_cancel is used as a fallback default.
        self._p_cancel = float(jitter_cfg.get("p_cancel", 0.03))
        per_tech = jitter_cfg.get("p_cancel_per_tech", None)
        if per_tech is not None and len(per_tech) >= 5:
            self._p_cancel_per_tech = np.array(per_tech[:5], dtype=np.float64)
        else:
            self._p_cancel_per_tech = np.full(5, self._p_cancel, dtype=np.float64)
        self._recovery_rate = jitter_cfg.get("recovery_rate", 0.40)
        cf_sigma = jitter_cfg.get("cf_sigma", [0.0, 0.0, 0.08, 0.08, 0.05])
        self._cf_sigma = np.array(cf_sigma, dtype=np.float64)

        # State: technology mix vector [coal, gas, onshore, offshore, solar]
        self.mix = np.array(initial_mix, dtype=np.float64)
        assert abs(self.mix.sum() - 1.0) < 1e-6, f"Mix must sum to 1.0, got {self.mix.sum()}"
        # Hot-method memoization (invalidated on mix / queue / carry_forward /
        # consecutive_successes mutation via _invalidate_state_cache()).
        # Eliminates repeated arithmetic across observation construction, reward
        # computation, and heuristic policy calls.
        self._cache_emissions = None
        self._cache_estimate_need = None
        self._cache_p_fail = None
        self._cache_queue_capacity = None
        self.initial_ef = self.weighted_emission_factor  # snapshot for ESG signal
        self.baseline_opex = self.compute_operational_cost(current_year=0)  # snapshot of initial-mix OPEX
        self.prev_green_frac = self.green_frac

        # Construction queue: list of {tech_idx, frac_delta, completion_year, success, capex_spent}
        self._construction_queue: List[Dict] = []
        self._consecutive_successes = 0
        self.year_cost = 0.0

        # Carry-forward non-compliance obligation (Mt)
        self._carry_forward = 0.0
        self._carry_forward_enabled = config.get("penalty", {}).get("carry_forward", False)
        # Cap carry-forward at this multiple of base annual emissions (0 = no cap).
        # Prevents exponential death-spiral where accumulated shortfall makes
        # recovery impossible regardless of bidding strategy.
        self._carry_forward_cap = config.get("penalty", {}).get("carry_forward_cap", 0.0)

        # MAC fuel-switching config
        mac_cfg = config.get("mac", {})
        self._mac_enabled = mac_cfg.get("enabled", False)
        self._mac_cost = mac_cfg.get("coal_to_gas_cost", 65.0)
        self._mac_max_switch = mac_cfg.get("max_switch_frac", 0.20)

        # Price normalization constant (matches auction price_max)
        self._price_norm = config["auction"]["price_max"]


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def green_frac(self) -> float:
        return float(self.mix[self.is_green].sum())

    @property
    def fossil_frac(self) -> float:
        return float(self.mix[~self.is_green].sum())

    @property
    def weighted_emission_factor(self) -> float:
        """Company's current average emission factor (tCO2/MWh)."""
        return float(np.dot(self.mix, self.emission_factors))

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _invalidate_state_cache(self):
        """Clear memoized hot-method results. Call after any mutation of
        self.mix, self._construction_queue, self._carry_forward, or
        self._consecutive_successes."""
        self._cache_emissions = None
        self._cache_estimate_need = None
        self._cache_p_fail = None
        self._cache_queue_capacity = None

    # ------------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------------

    def compute_emissions(self) -> float:
        """Annual emissions in Mt CO2 (deterministic from current mix)."""
        if self._cache_emissions is None:
            self._cache_emissions = self.output_mwh * self.weighted_emission_factor / 1e6
        return self._cache_emissions

    def compute_emissions_with_cf_noise(self, cf_noise: np.ndarray) -> float:
        """
        Annual emissions with capacity factor noise applied to green technologies.
        Green output scales by (1 + η_t) per tech; fossil fills any deficit (or is displaced
        by surplus). Returns Mt CO2. Does NOT modify the mix permanently.
        """
        if cf_noise is None or np.all(cf_noise == 0):
            return self.compute_emissions()

        # Apply CF noise to green tech fractions
        realized_mix = self.mix.copy()
        for t in range(N_TECHS):
            if self.is_green[t]:
                realized_mix[t] = self.mix[t] * max(0.0, 1.0 + cf_noise[t])

        # Fossil fills deficit (or is displaced by green surplus)
        green_realized = realized_mix[self.is_green].sum()
        fossil_target = max(0.0, 1.0 - green_realized)
        fossil_actual = realized_mix[~self.is_green].sum()
        if fossil_actual > 1e-10:
            realized_mix[~self.is_green] *= fossil_target / fossil_actual
        else:
            # No fossil; cap green at 1.0
            if green_realized > 1.0:
                realized_mix[self.is_green] /= green_realized

        realized_mix = np.clip(realized_mix, 0.0, 1.0)
        s = realized_mix.sum()
        if s > 1e-6:
            realized_mix /= s

        return float(self.output_mwh * np.dot(realized_mix, self.emission_factors) / 1e6)

    def apply_mac_switching(self, carbon_price: float, current_year: int = 0) -> tuple:
        """
        MAC fuel-switching: temporarily switch coal dispatch to gas when
        carbon price exceeds the marginal abatement cost.

        Returns (emissions_reduction_Mt, cost_M€).
        Does NOT modify the permanent capacity mix.
        """
        mac_cost = self._mac_cost * self.inflation_factor(current_year)
        if not self._mac_enabled or carbon_price <= mac_cost:
            return 0.0, 0.0

        switchable = min(self.mix[0], self._mac_max_switch)
        if switchable < 1e-6:
            return 0.0, 0.0

        ef_reduction = self.emission_factors[0] - self.emission_factors[1]  # tCO2/MWh
        switched_mwh = switchable * self.output_mwh
        emissions_reduction = switched_mwh * ef_reduction / 1e6  # Mt
        cost = emissions_reduction * mac_cost  # M€
        return emissions_reduction, cost

    def compute_risk_factor(self) -> float:
        if self._cache_p_fail is None:
            self._cache_p_fail = self._compute_p_fail()
        return self._cache_p_fail

    def compute_estimate_need(self) -> float:
        # Intentionally unbuffered: expected emissions + compliance debt.
        # Coverage buffers are still learned through bid multipliers.
        if self._cache_estimate_need is None:
            self._cache_estimate_need = self.compute_emissions() + self._carry_forward
        return self._cache_estimate_need

    # ------------------------------------------------------------------
    # Operational costs
    # ------------------------------------------------------------------

    def compute_operational_cost(self, current_year: int = 0) -> float:
        """Annual operational cost in M€ (fuel + O&M, excl. ETS), inflation-indexed."""
        return self.output_mwh * float(np.dot(self.mix, self.operational_costs)) / 1e6 * self.inflation_factor(current_year)

    def compute_ets_fuel_cost(self, ets_price: float) -> float:
        """Annual ETS cost in M€ based on current mix and carbon price."""
        return self.compute_emissions() * ets_price

    # ------------------------------------------------------------------
    # Revenue-based dynamic budget
    # ------------------------------------------------------------------

    def compute_revenue(self, marginal_ef: float, carbon_price: float,
                        inflation_factor: float) -> float:
        """
        Electricity revenue in M€.

        Revenue = output_TWh × (base_price + passthrough × carbon_price × marginal_ef) × inflation_factor

        Parameters
        ----------
        marginal_ef : float
            Marginal emission factor of the price-setting technology (tCO2/MWh).
        carbon_price : float
            Carbon price used for cost pass-through (EUR/tCO2).
        inflation_factor : float
            Cumulative inflation from year 0.

        Returns M€.
        """
        elec_cfg = self.config.get("electricity", {})
        base_price = float(elec_cfg.get("base_price", 50.0))
        passthrough = float(elec_cfg.get("carbon_passthrough", 0.80))
        marginal_price = base_price + passthrough * carbon_price * marginal_ef
        return self.output_twh * marginal_price * inflation_factor

    def compute_dynamic_budget(self, carbon_price: float, marginal_ef: float,
                               current_year: int) -> float:
        """
        Revenue-based annual budget: max(1.0, revenue - opex + debt_headroom).

        Ensures every agent can afford at least minimal participation.
        """
        inf = self.inflation_factor(current_year)
        revenue = self.compute_revenue(marginal_ef, carbon_price, inf)
        opex = self.compute_operational_cost(current_year)
        dynamic_budget = max(1.0, revenue - opex + self.debt_headroom)
        ceiling = self.annual_budget * self._dynamic_budget_ceiling_mult
        return min(dynamic_budget, ceiling)

    def set_annual_budget(self, value: float) -> None:
        """Set annual budget and update exponential moving average."""
        self.annual_budget = value
        if self._budget_ema is None:
            self._budget_ema = value
        else:
            alpha = self._budget_ema_alpha
            self._budget_ema = alpha * value + (1.0 - alpha) * self._budget_ema

    # ------------------------------------------------------------------
    # Emergency loan facility
    # ------------------------------------------------------------------

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

    def apply_loan_repayment(self) -> None:
        """Deduct annual loan repayment from budget at year start."""
        if self._years_under_loan > 0:
            self.annual_budget = max(1.0, self.annual_budget - self._loan_repayment_annual)
            self._years_under_loan -= 1
            if self._years_under_loan == 0:
                self._loan_outstanding = 0.0
                self._loan_repayment_annual = 0.0

    def get_loan_outstanding_norm(self) -> float:
        """Loan outstanding normalized by annual budget."""
        return self._loan_outstanding / max(self.annual_budget, 1.0)

    @property
    def effective_capex_throughput(self) -> float:
        """Capex throughput available this year, scaled by emergency-loan
        squeeze and by realised electricity revenue (high revenue → more
        construction capacity, low revenue → less)."""
        base = self.capex_throughput
        if self._years_under_loan > 0 and self._loan_outstanding > 0:
            loan_burden = self._loan_outstanding / max(self.annual_budget, 1.0)
            squeeze = max(
                float(self.config.get("budget", {}).get("emergency_loan", {})
                      .get("capex_squeeze_floor", 0.50)),
                1.0 - loan_burden
            )
            base = base * squeeze
        return base * float(self._revenue_capex_factor)

    def update_capex_revenue_factor(self, revenue: float) -> None:
        """Update the revenue-based capex-throughput multiplier.

        Factor = clip(0.7 + 0.3 × revenue / baseline_revenue, 0.5, 1.5).
        Baseline revenue is captured the first time this method is called
        (i.e. the year-0 revenue with the initial technology mix).
        """
        rev = float(revenue)
        if self._baseline_revenue_for_capex is None or self._baseline_revenue_for_capex <= 0.0:
            self._baseline_revenue_for_capex = max(rev, 1.0)
        ratio = rev / max(self._baseline_revenue_for_capex, 1.0)
        self._revenue_capex_factor = float(np.clip(0.7 + 0.3 * ratio, 0.5, 1.5))

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

    # ------------------------------------------------------------------
    # Investment (greening-only)
    # ------------------------------------------------------------------

    def _compute_p_fail(self) -> float:
        # Investment execution risk only (project failure), not compliance uncertainty.
        p_base = self.p_fail_min + (self.p_fail_max - self.p_fail_min) * (self.fossil_frac ** self.p_fail_alpha)
        if self._consecutive_successes >= self.exp_threshold:
            p_base -= self.exp_discount
        return max(self.p_fail_min, p_base)

    def compute_investment_cost(self, tech_idx: int, frac_delta: float,
                               current_year: int = 0) -> float:
        """
        Compute real-data-grounded investment cost for adding renewable capacity.

        Cost = ΔP × CapEx × (1 + α × ΔP / TotalCapacity) × inflation_factor

        where ΔP = ΔMWh / (CF × 8760) is the new capacity in kW.

        Returns cost in M€.
        """
        if frac_delta <= 0:
            return 0.0

        delta_mwh = frac_delta * self.output_mwh  # MWh/year to add
        cf = self.capacity_factors[tech_idx]
        delta_mw = delta_mwh / (cf * 8760)  # MW of new capacity needed
        delta_kw = delta_mw * 1000.0         # convert to kW (CapEx is in €/kW)

        # Total company capacity (approximate, for convexity scaling)
        total_mw = self.output_mwh / (np.dot(self.mix, self.capacity_factors) * 8760 + 1e-6)
        total_kw = total_mw * 1000.0

        base_cost = delta_kw * self.capex[tech_idx]  # €
        convexity = 1.0 + self.convexity_alpha * delta_kw / max(total_kw, 1e-6)
        cost_eur = base_cost * convexity
        return cost_eur / 1e6 * self.inflation_factor(current_year)  # M€

    def compute_decommission_cost(self, fossil_tech_idx: int, frac_delta: float,
                                  current_year: int = 0) -> float:
        """Cost to decommission fossil capacity in M€, inflation-indexed."""
        if frac_delta <= 0 or self.decommission_costs[fossil_tech_idx] <= 0:
            return 0.0
        delta_mwh = frac_delta * self.output_mwh
        cf = self.capacity_factors[fossil_tech_idx]
        delta_mw = delta_mwh / (cf * 8760)
        delta_kw = delta_mw * 1000.0
        return delta_kw * self.decommission_costs[fossil_tech_idx] / 1e6 * self.inflation_factor(current_year)

    def plan_investment(self, tech_choice: int, invest_frac: float, current_year: int) -> float:
        """
        Plan a green investment.

        Parameters
        ----------
        tech_choice : int
            Index into buildable technologies (0=onshore, 1=offshore, 2=solar).
            Maps to tech indices [2, 3, 4].
        invest_frac : float
            Fraction of total output to shift from fossil to this green tech.
        current_year : int

        Returns
        -------
        total_cost : float
            Investment + decommissioning cost in M€ (paid upfront).
        """
        if tech_choice < 0 or tech_choice >= len(BUILDABLE_INDICES):
            tech_choice = 0
        tech_idx = BUILDABLE_INDICES[int(tech_choice)]

        frac = float(np.clip(invest_frac, 0.0, self.max_invest_frac))
        if frac < 1e-6:
            return 0.0

        # Can't invest more than remaining fossil fraction
        frac = min(frac, self.fossil_frac)
        if frac < 1e-6:
            return 0.0

        # Investment cost for new green capacity
        invest_cost = self.compute_investment_cost(tech_idx, frac, current_year)

        # Decommission cost: retire highest-emission fossil first
        decom_cost = 0.0
        frac_to_retire = frac
        for fossil_idx in [0, 1]:  # coal, gas
            if frac_to_retire <= 0:
                break
            available = self.mix[fossil_idx]
            retire_this = min(frac_to_retire, available)
            if retire_this > 1e-6:
                decom_cost += self.compute_decommission_cost(fossil_idx, retire_this, current_year)
                frac_to_retire -= retire_this

        total_cost = invest_cost + decom_cost

        # Risk check
        p_fail = self._compute_p_fail()
        success = self.rng.random() > p_fail

        if success:
            self._consecutive_successes += 1
        # v8.5.8: do NOT reset _consecutive_successes on a single failure.
        # The previous reset created a positive-feedback loop where lucky
        # agents (who hit 2 successes in a row early) unlocked the experience
        # discount permanently, while agents with one early failure had to
        # restart the streak from zero — driving 10× inter-agent invest
        # variance from tiny seed-luck differences. The streak now monotonic-
        # ally accumulates, so accumulated experience is no longer wiped out
        # by one failed project.

        if self._jitter_enabled:
            lam = float(self._jitter_lambdas[tech_idx])
            delay = int(self.rng.poisson(lam))
            delay = min(delay, int(lam) + 3)  # cap at λ + 3
        else:
            delay = int(self.deploy_delays[tech_idx])

        self._construction_queue.append({
            "tech_idx": tech_idx,
            "frac_delta": frac if success else 0.0,
            "completion_year": current_year + delay,
            "success": success,
            "capex_spent": total_cost,
        })
        self._invalidate_state_cache()

        return total_cost

    def cancel_queued_projects(self, rng) -> float:
        """
        Each queued project faces a per-year cancellation probability that
        depends on its target technology (``p_cancel_per_tech``). On
        cancellation, ``recovery_rate`` × spent capex is recovered.
        Returns recovered M€ (positive = money back).
        """
        if not self._jitter_enabled:
            return 0.0

        remaining = []
        recovered = 0.0
        for item in self._construction_queue:
            tech_idx = int(item.get("tech_idx", -1))
            if 0 <= tech_idx < len(self._p_cancel_per_tech):
                p_c = float(self._p_cancel_per_tech[tech_idx])
            else:
                p_c = float(self._p_cancel)
            if p_c > 0.0 and rng.random() < p_c:
                recovered += item.get("capex_spent", 0.0) * self._recovery_rate
            else:
                remaining.append(item)
        self._construction_queue = remaining
        self._invalidate_state_cache()
        return recovered

    def apply_matured_investments(self, current_year: int):
        """Apply completed construction projects. Retire fossil to make room."""
        self.prev_green_frac = self.green_frac

        matured = []
        remaining = []
        for item in self._construction_queue:
            if current_year >= item["completion_year"]:
                matured.append(item)
            else:
                remaining.append(item)
        self._construction_queue = remaining

        for item in matured:
            frac_delta = item["frac_delta"]
            if frac_delta <= 0:
                continue
            tech_idx = item["tech_idx"]

            # Retire fossil to make room (coal first, then gas)
            frac_to_retire = frac_delta
            for fossil_idx in [0, 1]:
                if frac_to_retire <= 0:
                    break
                available = self.mix[fossil_idx]
                retire = min(frac_to_retire, available)
                self.mix[fossil_idx] -= retire
                frac_to_retire -= retire

            # Add green capacity
            self.mix[tech_idx] += frac_delta

            # Normalize to handle floating point drift
            self.mix = np.clip(self.mix, 0.0, 1.0)
            self.mix /= self.mix.sum()

        self._invalidate_state_cache()

    # ------------------------------------------------------------------
    # Public info (for opponent modeling)
    # ------------------------------------------------------------------

    def get_public_info(self) -> dict:
        """Return publicly observable information about this company (6D)."""
        queue_total = sum(item["frac_delta"] for item in self._construction_queue)
        return {
            "emissions": self.compute_emissions() / 10.0,
            "carry_forward": self._carry_forward / 5.0,
            "green_frac": self.green_frac,
            "fossil_frac": self.fossil_frac,
            "queue_total": queue_total,
            "is_active": 1.0,
        }

    # ------------------------------------------------------------------
    # Construction queue info (for observation space)
    # ------------------------------------------------------------------

    def get_queue_capacity(self) -> np.ndarray:
        """MW under construction per green technology [onshore, offshore, solar]."""
        if self._cache_queue_capacity is not None:
            return self._cache_queue_capacity
        queue_frac = np.zeros(3)  # onshore, offshore, solar
        for item in self._construction_queue:
            tech_idx = item["tech_idx"]
            if tech_idx == 2:
                queue_frac[0] += item["frac_delta"]
            elif tech_idx == 3:
                queue_frac[1] += item["frac_delta"]
            elif tech_idx == 4:
                queue_frac[2] += item["frac_delta"]
        self._cache_queue_capacity = queue_frac
        return queue_frac

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def reset_budget(self):
        self.budget_spent_this_year = 0.0
        self.green_loan_utilized = 0.0
        self._loan_drawn_this_step = 0.0

    def record_spending(self, amount: float):
        self.budget_spent_this_year += float(amount)

    def record_green_loan(self, amount: float):
        self.green_loan_utilized += float(amount)

    def compute_green_loan_cost(self) -> float:
        return self._gf_interest_rate * self.green_loan_utilized

    @property
    def green_loan_headroom(self) -> float:
        if not self._gf_enabled:
            return 0.0
        return max(0.0, self._gf_loan_budget - self.green_loan_utilized)

    @property
    def green_capex_headroom(self) -> float:
        if not self._gf_enabled:
            return 0.0
        # Capex boost is annual headroom, not a cumulative state variable.
        return self._gf_capex_boost

    def compute_budget_penalty(self) -> float:
        """Tiered budget penalty: zero below soft_zone, quadratic in soft zone,
        steeper above hard cap, with an absolute ceiling to prevent explosive growth.

        Returns an absolute cost (M EUR) that is later divided by
        ``annual_budget`` in the reward function.  The magnitude is kept
        moderate by scaling with the *overshoot amount* rather than the
        full budget, so a 5 % overshoot on a 1 000 M EUR budget produces
        a penalty ≈ coef × (normalized²) × overshoot_abs.

        Capped at ``max_penalty_budget_mult × budget`` to prevent
        quadratic explosion from dominating the reward signal.
        """
        budget = max(self.annual_budget, 1.0)
        spend_ratio = self.budget_spent_this_year / budget
        budget_cfg = self.config.get("budget", {})
        soft_start = float(budget_cfg.get("soft_zone_start", 1.0))
        hard_cap = float(budget_cfg.get("hard_cap_fraction", 1.15))
        coef = float(budget_cfg.get("tiered_penalty_coef", 2.0))
        max_penalty_mult = float(budget_cfg.get("max_penalty_budget_mult", 2.0))

        if spend_ratio <= soft_start:
            return 0.0

        overshoot_abs = self.budget_spent_this_year - soft_start * budget
        zone_width = max(hard_cap - soft_start, 1e-6)
        normalized = (spend_ratio - soft_start) / zone_width

        if spend_ratio <= hard_cap:
            # Quadratic ramp within the soft zone
            raw = coef * (normalized ** 2) * overshoot_abs
        else:
            # Above hard cap: cubic-like (extra × normalized factor)
            raw = coef * (normalized ** 3) * overshoot_abs

        return min(raw, max_penalty_mult * budget)

    def get_budget_utilization(self) -> float:
        return self.budget_spent_this_year / max(self.annual_budget, 1e-6)

    # ------------------------------------------------------------------
    # Capex throughput
    # ------------------------------------------------------------------

    def reset_capex_budget(self):
        self.capex_spent_this_year = 0.0

    def record_capex_spending(self, amount: float):
        self.capex_spent_this_year += float(amount)

    def compute_capex_penalty(self) -> float:
        overshoot = max(0.0, self.capex_spent_this_year - self.capex_throughput)
        if overshoot < 1e-6:
            return 0.0
        ratio = overshoot / self.capex_throughput
        return self.capex_overspend_coef * (ratio ** 2) * self.capex_throughput

    def get_capex_utilization(self) -> float:
        return self.capex_spent_this_year / max(self.capex_throughput, 1e-6)

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------

    def set_inflation_path(self, annual_rates: List[float]):
        """Set shared episode inflation path (one rate per simulated year)."""
        self._inflation_rates_by_year = {i: float(r) for i, r in enumerate(annual_rates)}
        self._inflation_factor_by_year = {0: 1.0}

    def _inflation_rate_for_year(self, year_idx: int) -> float:
        if year_idx not in self._inflation_rates_by_year:
            rate = self._inflation_rate
            if self._inflation_random_std > 0.0:
                rate = float(self.rng.normal(self._inflation_rate, self._inflation_random_std))
                rate = max(-0.99, rate)
            elif self._inflation_random_window > 0.0:
                low = max(-0.99, self._inflation_rate - self._inflation_random_window)
                high = self._inflation_rate + self._inflation_random_window
                rate = float(self.rng.uniform(low, high))
            self._inflation_rates_by_year[year_idx] = float(rate)
        return float(self._inflation_rates_by_year[year_idx])

    def inflation_rate_for_year(self, current_year: int = 0) -> float:
        """Year-specific inflation rate used for this episode and year index."""
        return self._inflation_rate_for_year(max(0, int(current_year)))

    def inflation_factor(self, current_year: int = 0) -> float:
        """General inflation multiplier using compounded year-specific rates."""
        current_year = max(0, int(current_year))
        if current_year not in self._inflation_factor_by_year:
            start = max(self._inflation_factor_by_year.keys()) + 1
            for y in range(start, current_year + 1):
                prev_factor = self._inflation_factor_by_year[y - 1]
                prev_rate = self._inflation_rate_for_year(y - 1)
                self._inflation_factor_by_year[y] = prev_factor * (1.0 + prev_rate)
        return float(self._inflation_factor_by_year[current_year])

    def effective_penalty_rate(self, current_year: int = 0) -> float:
        """Penalty rate adjusted for inflation: base_rate × inflation_factor."""
        return self.penalty_rate * self.inflation_factor(current_year)

    def settle_compliance(self, allowances_held: float, current_year: int = 0) -> float:
        shortfall = max(0.0, self.compute_emissions() - allowances_held)
        return shortfall * self.effective_penalty_rate(current_year)

    def settle_compliance_realized(self, allowances_held: float, realized_emissions: float,
                                   current_year: int = 0) -> float:
        """
        Settle compliance against realized (shocked) emissions.
        With carry_forward enabled, shortfall is added to next year's obligation,
        optionally capped at ``carry_forward_cap × base_annual_emissions`` to
        prevent an exponential death-spiral in early training.
        """
        total_need = realized_emissions + self._carry_forward
        shortfall = max(0.0, total_need - allowances_held)
        if self._carry_forward_enabled:
            cap_mult = self._carry_forward_cap
            if cap_mult > 0:
                # Cap is defined against base annual emissions, not debt-inclusive need.
                base_emiss = max(self.compute_emissions(), 0.1)
                self._carry_forward = min(shortfall, cap_mult * base_emiss)
            else:
                self._carry_forward = shortfall
            self._invalidate_state_cache()
        return shortfall * self.effective_penalty_rate(current_year)

    # ------------------------------------------------------------------
    # Observations — Phase 1: 44D base (+7*(N-1) opponent) | Phase 2: +12
    # See get_observation_phase1 / get_observation_phase2 for the full layout.
    # ------------------------------------------------------------------

    def get_observation_phase1(self, year, cap_t, last_clearing_price,
                               expected_price, auction_gap=0.0,
                               last_secondary_price=0.0,
                               secondary_profit_signal=0.0,
                               price_ma3=None,
                               opponent_obs=None,
                               last_secondary_volume=0.0,
                               tnac_proxy=0.0,
                               effective_reserve=0.0,
                               last_auction_volume=0.0,
                               msr_reserve=0.0,
                               bank=0.0,
                               tnac_upper=28.0,
                               tnac_mid=None,
                               withhold_rate=0.24,
                               budget_spent: float = 0.0,
                               annual_budget: float = 1e9,
                               collateral_load_last: float = 0.0,
                               bid_affordability_last: float = 0.0,
                               n_years: int = 12,
                               last_cover_ratio: float = 1.0,
                               own_last_secondary_buy_price: float = 0.0,
                               cumulative_coverage_ratio: float = 1.0,
                               cap_ahead_3y_ratio: float = 1.0,
                               cap_ahead_6y_ratio: float = 1.0,
                               pcl_ceiling: float = 0.0,
                               last_bid_price_clip: float = 0.0,
                               last_budget_price_clip: float = 0.0,
                               last_bid_qty_clip_ratio: float = 1.0,
                               last_invest_clip_ratio: float = 1.0,
                               compliance_affordability: float = 0.0):
        """
        Phase 1 observation (pre-auction): 44D base + 7*(N-1) opponent dims.

        Base 44 dims:
        [0]  time (normalized)
        [1]  cap (normalized)
        [2]  3-year moving average of clearing price (normalized)
        [3]  expected price from AR(1) model (normalized)
        [4-8]  technology mix vector (5D)
        [9]  emissions (normalized)
        [10] expected annual emissions (no risk buffer)
        [11] p_fail
        [12] investment experience
        [13] auction gap (banked allowances)
        [14-16] construction queue (onshore, offshore, solar)
        [17] weighted emission factor (normalized)
        [18] last secondary market price (normalized)
        [19] last secondary market volume (normalized)
        [20] carry-forward obligation (Mt)
        [21] TNAC proxy (total banked allowances / cap, clipped to [0,3])
        [22] effective reserve price / price_max
        [23] auction volume ratio = THIS YEAR'S auction_volume / cap_t (MSR-adjusted preview)
        [24] MSR reserve normalized = msr_reserve / cap_t
        [25] own bank ratio (clipped [0, 5], normalized by /5)
        [26] predicted MSR withholding fraction of cap
        [27] budget_headroom (1.0=fresh, 0.0=at limit, negative=overspent)
        [28] collateral_load_last: last year's collateral locked / annual_budget
             (clipped [0,1]; high → overbid risk; agents learn to stay below budget)
        [29] bid_affordability_last: last year's bid_total / budget_remaining (clipped [0,1])
        [30] loan_outstanding_norm: emergency loan outstanding / annual_budget
        [31] years_under_loan_norm: remaining loan years / n_years
        [32] last_cover_ratio: auction_supply / total_demand (clipped [0,3], /3)
             WTP signal: low cover_ratio → high competition → should bid higher
        [33] own_last_secondary_buy_price: (own last sec buy price / price_max)
             WTP signal: high secondary cost → agent should bid more at auction
        [34] cumulative_coverage_ratio: cumul_alloc / cumul_emissions (clipped [0,2], /2)
             Long-run compliance signal: <1 means persistently under-buying
        [35] treasury_norm: treasury_reserve / annual_budget (clipped [0,2], /2)
        [36] cap_ahead_3y_ratio: cap(t+3) / cap(t) clipped [0,1] — 3-year scarcity lookahead
        [37] cap_ahead_6y_ratio: cap(t+6) / cap(t) clipped [0,1] — 6-year scarcity lookahead
        [38] pcl_headroom_norm: (pcl_ceiling - price_ma3) / price_max clipped [0,1]
             Headroom to upper bid-change bound; 1.0 = unconstrained
        [39] last_bid_price_clip: (actual_bid - requested_bid) / price_max, signed [-1,1]
             PCL clip only; negative if clipped down by bid-change limit; zero if unconstrained
        [40] last_budget_price_clip: (actual_bid - requested_bid) / price_max, signed [-1,1]
             Budget clip only; negative if clipped down by budget affordability gate; zero if unconstrained
        [41] last_bid_qty_clip_ratio: actual_qty / requested_qty clipped [0,1]
             1.0 = no qty gate fired; <1 = leverage/collateral/budget gate reduced qty
        [42] last_invest_clip_ratio: actual_invest_frac / requested_invest_frac clipped [0,1]
             1.0 = no cap applied; <1 = budget/capex gate reduced investment
        [43] compliance_affordability: forward-looking budget signal (v8.5.8).
             = (estimate_need × expected_clearing) / max(cash, 1), clipped [0, 3], normalized /3.
             Tells the agent how much of its remaining cash a need-covering bid at the
             expected clearing price would consume. 0 ≈ trivially affordable; 0.33 (raw 1.0)
             = entire cash needed for compliance; 1.0 (raw 3.0) = compliance unaffordable
             from cash alone. Replaces purely lagged clip-feedback with a forward-looking
             budget anchor so agents don't need to learn affordability through repeated
             clip events.

        Opponent dims (if opponent_modeling enabled, 7D per opponent):
        [44..] = (emissions/10, green_frac, fossil_frac, queue_noisy,
                  tnac_share_norm, net_secondary_norm,
                  lagged_compliance_gap_norm) per opponent
        """
        price_signal = (price_ma3 if price_ma3 is not None else last_clearing_price)
        queue = self.get_queue_capacity()
        pn = self._price_norm  # normalization constant (= price_max)
        own_need = max(self.compute_estimate_need(), 0.1)
        own_bank_ratio = float(np.clip(float(bank) / own_need, 0.0, 5.0)) / 5.0
        tnac_mid = float(tnac_mid) if tnac_mid is not None else float(tnac_upper) * (833.0 / 1096.0)
        predicted_tnac = float(tnac_proxy * cap_t)
        if predicted_tnac > float(tnac_upper):
            withheld_abs = float(withhold_rate) * predicted_tnac
        elif tnac_mid <= predicted_tnac <= float(tnac_upper):
            withheld_abs = predicted_tnac - tnac_mid
        else:
            withheld_abs = 0.0
        predicted_withhold = withheld_abs / max(cap_t, 1e-6)
        predicted_withhold = float(np.clip(predicted_withhold, 0.0, 1.0))
        budget_headroom = float(np.clip(
            1.0 - (budget_spent / max(annual_budget, 1e-6)),
            -0.5,
            1.0,
        ))
        treasury_norm = float(np.clip(
            self._treasury_reserve / max(annual_budget, 1.0), 0.0, 2.0
        )) / 2.0

        base = np.array([
            year / 12.0,                          # [0] normalized by n_years
            cap_t / 30.0,                         # [1] normalized for 8-agent cap
            price_signal / pn,                    # [2]
            expected_price / pn,                  # [3]
            self.mix[0],                          # [4] coal frac
            self.mix[1],                          # [5] gas frac
            self.mix[2],                          # [6] onshore frac
            self.mix[3],                          # [7] offshore frac
            self.mix[4],                          # [8] solar frac
            self.compute_emissions() / 10.0,      # [9]
            self.compute_estimate_need() / 10.0,  # [10]
            self.compute_risk_factor(),           # [11]
            self._consecutive_successes / 5.0,    # [12]
            auction_gap / 5.0,                    # [13]
            queue[0],                             # [14] onshore under construction
            queue[1],                             # [15] offshore under construction
            queue[2],                             # [16] solar under construction
            self.weighted_emission_factor,        # [17] avg EF
            last_secondary_price / pn,            # [18] secondary price signal
            last_secondary_volume / 10.0,         # [19] secondary volume signal
            self._carry_forward / 5.0,            # [20] carry-forward obligation (Mt)
            float(np.clip(tnac_proxy, 0.0, 3.0)), # [21] TNAC proxy
            effective_reserve / pn,               # [22] effective reserve signal
            last_auction_volume / max(cap_t, 1e-6),  # [23] THIS YEAR'S auction volume ratio
            msr_reserve / max(cap_t, 1e-6),          # [24] MSR reserve signal
            own_bank_ratio,                          # [25] own bank ratio
            predicted_withhold,                      # [26] predicted MSR withhold share
            budget_headroom,                         # [27] budget headroom signal
            float(np.clip(collateral_load_last, 0.0, 1.0)),       # [28] collateral load last year
            float(np.clip(bid_affordability_last, 0.0, 1.0)),     # [29] bid affordability
            self.get_loan_outstanding_norm(),                      # [30] loan outstanding norm
            float(self._years_under_loan / max(n_years, 1)),      # [31] years under loan norm
            float(np.clip(last_cover_ratio, 0.0, 3.0)) / 3.0,    # [32] WTP: auction cover ratio
            float(np.clip(own_last_secondary_buy_price, 0.0, pn)) / pn,  # [33] WTP: own sec buy price
            float(np.clip(cumulative_coverage_ratio, 0.0, 2.0)) / 2.0,  # [34] cumulative coverage
            treasury_norm,                                         # [35] treasury reserve norm
            float(np.clip(cap_ahead_3y_ratio, 0.0, 1.0)),        # [36] 3-year cap scarcity lookahead
            float(np.clip(cap_ahead_6y_ratio, 0.0, 1.0)),        # [37] 6-year cap scarcity lookahead
            float(np.clip((pcl_ceiling - price_signal) / pn, 0.0, 1.0)),   # [38] pcl headroom norm
            float(np.clip(last_bid_price_clip / pn, -1.0, 1.0)),          # [39] PCL price clip (signed)
            float(np.clip(last_budget_price_clip / pn, -1.0, 1.0)),       # [40] budget price clip (signed)
            float(np.clip(last_bid_qty_clip_ratio, 0.0, 1.0)),            # [41] bid qty clip ratio
            float(np.clip(last_invest_clip_ratio, 0.0, 1.0)),             # [42] invest frac clip ratio
            float(np.clip(compliance_affordability, 0.0, 3.0)) / 3.0,    # [43] forward-looking compliance affordability
        ], dtype=np.float32)
        if opponent_obs is not None and len(opponent_obs) > 0:
            return np.concatenate([base, opponent_obs])
        return base

    def get_observation_phase2(self, obs_phase1, allocation,
                                clearing_price, emissions, banked=0.0,
                                emission_shock=0.0, payment=0.0,
                                collateral_locked_norm: float = 0.0,
                                current_holdings: float = 0.0,
                                current_year: int = 0,
                                last_sec_qty_clip_ratio: float = 1.0):
        """
        Phase 2 observation (post-auction): obs_phase1 + 12 extra dims.
        Appends auction results, emission shock, auction savings, coverage and compliance signals.

        Extra dims:
        [base+0]  allocation / 5
        [base+1]  clearing_price / price_max
        [base+2]  net compliance position: (banked + allocation - emissions - carry_forward) / 5
                  <0 means the agent is still short after using all holdings
        [base+3]  emission_shock (realized deviation from base need)  -- P5
        [base+4]  auction_savings: (allocation × 100 - payment) / 1000
                  penalty-value avoided minus cost paid; encodes deal quality
        [base+5]  coverage_ratio: (banked + allocation) / max(emissions + carry_forward, 1e-6)
                  clipped to [0, 3], normalized by /3
        [base+6]  normalized carry_forward: carry_forward / max(estimated_need, 1e-6)
                  clipped to [0, 3]; agents need to see their debt
        [base+7]  collateral_locked_norm: this year's collateral locked / annual_budget
                  clipped to [0, 1]; immediate feedback on auction over-commitment risk
        [base+8]  budget_remaining_phase2_norm: (annual_budget - budget_spent) / annual_budget
                  clipped to [-0.5, 1.0]; post-auction budget headroom
        [base+9]  compliance_liability_norm: unfunded compliance cost / annual_budget
                  clipped to [0, 2.0]; signals penalty exposure
        [base+10] compliance_gap_norm: (emissions + carry_forward - holdings) / estimated_need
                  clipped to [-2, 2] /2; >0=short, <0=surplus
        [base+11] last_sec_qty_clip_ratio: actual_sec_qty / requested_sec_qty (previous year)
                  clipped to [-1, 1]; 1.0 = no constraint; <1 = buy clipped; negative = sell side
        """
        auction_savings = (allocation * 100.0 - payment) / 1000.0

        # Coverage ratio: how well-covered the agent is for compliance
        total_obligation = max(emissions + self._carry_forward, 1e-6)
        coverage_ratio = float(np.clip((banked + allocation) / total_obligation, 0.0, 3.0)) / 3.0

        # Normalized carry-forward: debt relative to estimated need
        estimated_need = max(self.compute_estimate_need(), 1e-6)
        carry_forward_norm = float(np.clip(self._carry_forward / estimated_need, 0.0, 3.0)) / 3.0

        budget_remaining_phase2_norm = float(np.clip(
            (self.annual_budget - self.budget_spent_this_year) / max(self.annual_budget, 1e-6),
            -0.5, 1.0,
        ))

        # Compliance liability: unfunded shortfall × penalty rate / budget
        shortfall = max(0.0, estimated_need - current_holdings)
        eff_penalty = self.effective_penalty_rate(current_year)
        compliance_liability_norm = float(np.clip(
            shortfall * eff_penalty / max(self.annual_budget, 1e-6),
            0.0, 2.0,
        ))

        total_holdings_now = banked + allocation
        compliance_gap = (emissions + self._carry_forward - total_holdings_now)
        compliance_gap_norm = float(np.clip(compliance_gap / estimated_need, -2.0, 2.0)) / 2.0
        # [base+10]: >0=short (under-covered), <0=surplus

        extra = np.array([
            allocation / 5.0,                                                       # [base+0]
            clearing_price / self._price_norm,                                      # [base+1]
            (banked + allocation - emissions - self._carry_forward) / 5.0,          # [base+2]
            float(emission_shock),                                                  # [base+3]
            float(auction_savings),                                                 # [base+4]
            coverage_ratio,                                                         # [base+5]
            carry_forward_norm,                                                     # [base+6]
            float(np.clip(collateral_locked_norm, 0.0, 1.0)),                      # [base+7]
            budget_remaining_phase2_norm,                                           # [base+8]
            compliance_liability_norm,                                              # [base+9]
            compliance_gap_norm,                                                    # [base+10]
            float(np.clip(last_sec_qty_clip_ratio, -1.0, 1.0)),                   # [base+11]
        ], dtype=np.float32)
        return np.concatenate([obs_phase1, extra])

    @property
    def obs_dim_phase1(self) -> int:
        """44 base dims + 7*(N_total-1) opponent dims. See get_observation_phase1 for full layout."""
        opp_dims = self.config.get("opponent_obs", {}).get("dims_per_opponent", 7)
        if self._opponent_modeling and self._n_total > 1:
            return 44 + opp_dims * (self._n_total - 1)
        return 44

    @property
    def obs_dim_phase2(self) -> int:
        """obs_dim_phase1 + 12 auction-result dims. See get_observation_phase2 for full layout."""
        return self.obs_dim_phase1 + 12

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_mix: List[float]):
        self.mix = np.array(initial_mix, dtype=np.float64)
        self._invalidate_state_cache()
        self.initial_ef = self.weighted_emission_factor
        self.prev_green_frac = self.green_frac
        self._construction_queue = []
        self._consecutive_successes = 0
        self.year_cost = 0.0
        self.budget_spent_this_year = 0.0
        self.capex_spent_this_year = 0.0
        self.green_loan_utilized = 0.0
        self.prev_invest_frac = 0.0
        self._carry_forward = 0.0
        self._inflation_rates_by_year = {}
        self._inflation_factor_by_year = {0: 1.0}
        self._budget_ema = None
        self._loan_outstanding = 0.0
        self._loan_repayment_annual = 0.0
        self._years_under_loan = 0
        self._last_loan_fraction = 0.0
        self._loan_drawn_this_step = 0.0
        self._treasury_reserve = 0.0
        self._treasury_drawn_this_year = 0.0
        # Reset revenue-linked capex multiplier so the next episode's year-0
        # revenue is what re-establishes the baseline.
        self._baseline_revenue_for_capex = None
        self._revenue_capex_factor = 1.0

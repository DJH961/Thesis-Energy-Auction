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

Roadmap improvements (P6):
  - Delay jitter: construction delays drawn from Poisson(λ_tech) instead of fixed
  - Cancellation risk: projects can be cancelled mid-queue (CapEx partially recovered)
  - CF noise: capacity factor noise affects realized emissions each year

Roadmap improvements (P8):
  - Phase 1 observation includes secondary market price and volume signals
  - Phase 2 observation includes emission shock for Phase 2 decisions
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

        # Emergency loan facility (B1)
        loan_cfg = budget_cfg.get("emergency_loan", {})
        self._loan_enabled = bool(loan_cfg.get("enabled", False))
        self._max_loan_fraction = float(loan_cfg.get("max_loan_fraction", 0.15))
        self._loan_interest_rate = float(loan_cfg.get("interest_rate", 0.08))
        self._loan_repayment_years = int(loan_cfg.get("repayment_years", 3))
        self._loan_outstanding = 0.0
        self._loan_repayment_annual = 0.0
        self._years_under_loan = 0

        # Capex throughput constraint (organizational construction spend cap)
        capex_tp = budget_cfg.get("capex_throughputs", [])
        self.capex_throughput = capex_tp[agent_id] if agent_id < len(capex_tp) else 1e9
        self.capex_overspend_coef = budget_cfg.get("capex_overspend_coef", 1.0)
        self.capex_spent_this_year = 0.0

        # Green finance (optional): annual loan headroom for green capex only.
        gf_cfg = config.get("green_finance", {})
        self._gf_enabled = bool(gf_cfg.get("enabled", False))
        self._gf_loan_budget = float(gf_cfg.get("loan_budget_boost", 0.0))
        self._gf_interest_rate = float(gf_cfg.get("loan_interest_rate", 0.05))
        self._gf_capex_boost = float(gf_cfg.get("capex_throughput_boost", 0.0))
        self.green_loan_utilized = 0.0

        # P6: Construction jitter config
        jitter_cfg = config.get("construction_jitter", {})
        self._jitter_enabled = jitter_cfg.get("enabled", False)
        poisson_lambdas = jitter_cfg.get("poisson_lambdas", [1.0, 1.0, 2.0, 3.0, 1.5])
        self._jitter_lambdas = np.array(poisson_lambdas, dtype=np.float64)
        self._p_cancel = jitter_cfg.get("p_cancel", 0.03)
        self._recovery_rate = jitter_cfg.get("recovery_rate", 0.40)
        cf_sigma = jitter_cfg.get("cf_sigma", [0.0, 0.0, 0.08, 0.08, 0.05])
        self._cf_sigma = np.array(cf_sigma, dtype=np.float64)

        # State: technology mix vector [coal, gas, onshore, offshore, solar]
        self.mix = np.array(initial_mix, dtype=np.float64)
        assert abs(self.mix.sum() - 1.0) < 1e-6, f"Mix must sum to 1.0, got {self.mix.sum()}"
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

        # P6: track capex spent per queued project (for partial recovery on cancellation)
        # Stored inside each queue item as "capex_spent"

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
    # Emissions
    # ------------------------------------------------------------------

    def compute_emissions(self) -> float:
        """Annual emissions in Mt CO2 (deterministic from current mix)."""
        return self.output_mwh * self.weighted_emission_factor / 1e6

    def compute_emissions_with_cf_noise(self, cf_noise: np.ndarray) -> float:
        """
        P6: Annual emissions with capacity factor noise applied to green technologies.

        When green CF is lower than expected, fossil backup fills the gap → more emissions.
        When green CF is higher, fossil is displaced → fewer emissions.

        cf_noise : array of shape (5,)
            Per-technology CF noise factor. Non-zero only for green techs.
            η ~ N(0, σ_cf) per tech; green output scales by (1 + η_t).

        Returns emissions in Mt CO2. Does NOT modify mix permanently.
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
        return self._compute_p_fail()

    def compute_estimate_need(self) -> float:
        # Intentionally unbuffered: agents can learn their own coverage buffer.
        return self.compute_emissions()

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
    # Revenue-based dynamic budget (A2)
    # ------------------------------------------------------------------

    def compute_revenue(self, smoothed_price: float, system_ef: float,
                        inflation_factor: float) -> float:
        """
        Electricity revenue in M€.

        Revenue = output_TWh × (base_price + passthrough × smoothed_price × system_ef) × inflation_factor

        Parameters
        ----------
        smoothed_price : float
            Moving-average carbon price (EUR/tCO2).
        system_ef : float
            System-wide average emission factor (tCO2/MWh).
        inflation_factor : float
            Cumulative inflation from year 0.

        Returns M€.
        """
        elec_cfg = self.config.get("electricity", {})
        base_price = float(elec_cfg.get("base_price", 50.0))
        passthrough = float(elec_cfg.get("carbon_passthrough", 0.80))
        eff_price = base_price + passthrough * smoothed_price * system_ef
        return self.output_twh * eff_price * inflation_factor

    def compute_dynamic_budget(self, smoothed_price: float, system_ef: float,
                               current_year: int) -> float:
        """
        Revenue-based annual budget: max(1.0, revenue - opex + debt_headroom).

        Ensures every agent can afford at least minimal participation.
        """
        inf = self.inflation_factor(current_year)
        revenue = self.compute_revenue(smoothed_price, system_ef, inf)
        opex = self.compute_operational_cost(current_year)
        return max(1.0, revenue - opex + self.debt_headroom)

    def set_annual_budget(self, value: float) -> None:
        """Set annual budget and update exponential moving average."""
        self.annual_budget = value
        if self._budget_ema is None:
            self._budget_ema = value
        else:
            alpha = self._budget_ema_alpha
            self._budget_ema = alpha * value + (1.0 - alpha) * self._budget_ema

    # ------------------------------------------------------------------
    # Emergency loan facility (B2)
    # ------------------------------------------------------------------

    def apply_emergency_loan(self, shortfall: float) -> None:
        """Record an emergency loan to cover auction settlement shortfall."""
        self._loan_outstanding += shortfall
        self._loan_repayment_annual = (
            self._loan_outstanding * (1.0 + self._loan_interest_rate)
            / max(self._loan_repayment_years, 1)
        )
        self._years_under_loan = self._loan_repayment_years

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
        else:
            self._consecutive_successes = 0

        # P6: Delay jitter — draw from Poisson(λ_tech) instead of fixed delay
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

        return total_cost

    def cancel_queued_projects(self, rng) -> float:
        """
        P6: Probabilistic cancellation of in-flight projects.

        Each queued project faces p_cancel chance of cancellation each year.
        On cancellation, a fraction (recovery_rate) of capex already spent is
        returned (negative cost = recovered funds).

        Returns recovered_funds in M€ (positive value = money returned).
        """
        if not self._jitter_enabled or self._p_cancel <= 0:
            return 0.0

        remaining = []
        recovered = 0.0
        for item in self._construction_queue:
            if rng.random() < self._p_cancel:
                # Project cancelled; partial capex recovery
                recovered += item.get("capex_spent", 0.0) * self._recovery_rate
                # frac_delta lost
            else:
                remaining.append(item)
        self._construction_queue = remaining
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
        queue_frac = np.zeros(3)  # onshore, offshore, solar
        for item in self._construction_queue:
            tech_idx = item["tech_idx"]
            if tech_idx == 2:
                queue_frac[0] += item["frac_delta"]
            elif tech_idx == 3:
                queue_frac[1] += item["frac_delta"]
            elif tech_idx == 4:
                queue_frac[2] += item["frac_delta"]
        return queue_frac

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def reset_budget(self):
        self.budget_spent_this_year = 0.0
        self.green_loan_utilized = 0.0

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
        steep above hard cap.

        Returns an absolute cost (M EUR) that is later divided by
        ``annual_budget`` in the reward function.  The magnitude is kept
        moderate by scaling with the *overshoot amount* rather than the
        full budget, so a 5 % overshoot on a 1 000 M EUR budget produces
        a penalty ≈ coef × (normalized²) × overshoot_abs.
        """
        budget = max(self.annual_budget, 1.0)
        spend_ratio = self.budget_spent_this_year / budget
        budget_cfg = self.config.get("budget", {})
        soft_start = float(budget_cfg.get("soft_zone_start", 1.0))
        hard_cap = float(budget_cfg.get("hard_cap_fraction", 1.15))
        coef = float(budget_cfg.get("tiered_penalty_coef", 2.0))

        if spend_ratio <= soft_start:
            return 0.0

        overshoot_abs = self.budget_spent_this_year - soft_start * budget
        zone_width = max(hard_cap - soft_start, 1e-6)
        normalized = (spend_ratio - soft_start) / zone_width

        if spend_ratio <= hard_cap:
            # Quadratic ramp within the soft zone
            return coef * (normalized ** 2) * overshoot_abs
        else:
            # Above hard cap: penalty grows steeply (cubic-like feel)
            return coef * (normalized ** 2) * overshoot_abs

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
                base_emiss = max(self.compute_estimate_need(), 0.1)
                self._carry_forward = min(shortfall, cap_mult * base_emiss)
            else:
                self._carry_forward = shortfall
        return shortfall * self.effective_penalty_rate(current_year)

    # ------------------------------------------------------------------
    # Observations — Phase 1: 33D base (+5*(N-1) opponent) | Phase 2: +10
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
                               suspension_remaining_norm: float = 0.0,
                               collateral_load_last: float = 0.0,
                               bid_affordability_last: float = 0.0):
        """
        Phase 1 observation (pre-auction): 33D base + 5*(N-1) opponent dims.

        Base 33 dims:
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
        [18] last secondary market price (normalized)  -- P8
        [19] last secondary market volume (normalized) -- P8
        [20] carry-forward obligation (Mt)
        [21] TNAC proxy (total banked allowances / cap, clipped to [0,3])
        [22] effective reserve price / price_max
        [23] auction volume ratio = THIS YEAR'S auction_volume / cap_t (MSR-adjusted preview)
        [24] MSR reserve normalized = msr_reserve / cap_t
        [25] own bank ratio (clipped [0, 5], normalized by /5)
        [26] predicted MSR withholding fraction of cap
        [27] budget_headroom (1.0=fresh, 0.0=at limit, negative=overspent)
        [28] suspension_remaining_norm: rounds still suspended / suspension_length
             (0=not suspended, 1=fully suspended; helps avoid bids that lead to default)
        [29] collateral_load_last: last year's collateral locked / annual_budget
             (clipped [0,1]; high → overbid risk; agents learn to stay below budget)
        [30] bid_affordability_last: last year's bid_total / budget_remaining (clipped [0,1])
        [31] loan_outstanding_norm: emergency loan outstanding / annual_budget
        [32] years_under_loan_norm: remaining loan years / 12

        Opponent dims (if opponent_modeling enabled, 5D per opponent):
        [33..] = (emissions/10, carry_forward/5, green_frac, fossil_frac, queue_total) per opponent
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
            last_secondary_price / pn,            # [18] P8: secondary price signal
            last_secondary_volume / 10.0,         # [19] P8: secondary volume signal
            self._carry_forward / 5.0,            # [20] carry-forward obligation (Mt)
            float(np.clip(tnac_proxy, 0.0, 3.0)), # [21] TNAC proxy
            effective_reserve / pn,               # [22] effective reserve signal
            last_auction_volume / max(cap_t, 1e-6),  # [23] THIS YEAR'S auction volume ratio
            msr_reserve / max(cap_t, 1e-6),          # [24] MSR reserve signal
            own_bank_ratio,                          # [25] own bank ratio
            predicted_withhold,                      # [26] predicted MSR withhold share
            budget_headroom,                         # [27] budget headroom signal
            float(np.clip(suspension_remaining_norm, 0.0, 1.0)),  # [28] suspension signal
            float(np.clip(collateral_load_last, 0.0, 1.0)),       # [29] collateral load last year
            float(np.clip(bid_affordability_last, 0.0, 1.0)),     # [30] bid affordability
            self.get_loan_outstanding_norm(),                      # [31] loan outstanding norm
            float(self._years_under_loan / 12.0),                 # [32] years under loan norm
        ], dtype=np.float32)
        if opponent_obs is not None and len(opponent_obs) > 0:
            return np.concatenate([base, opponent_obs])
        return base

    def get_observation_phase2(self, obs_phase1, allocation,
                                clearing_price, emissions, banked=0.0,
                                emission_shock=0.0, payment=0.0,
                                collateral_locked_norm: float = 0.0,
                                current_holdings: float = 0.0,
                                current_year: int = 0):
        """
        Phase 2 observation (post-auction): obs_phase1 + 10 extra dims.
        Appends auction results + P5 emission shock + auction_savings + coverage_ratio
        + carry_forward_norm + collateral_locked_norm + budget_remaining_phase2_norm
        + compliance_liability_norm.

        Extra dims:
        [base+0] allocation / 5
        [base+1] clearing_price / price_max
        [base+2] net compliance position: (banked + allocation - emissions - carry_forward) / 5
                 <0 means the agent is still short after using all holdings
        [base+3] emission_shock (realized deviation from base need)  -- P5
        [base+4] auction_savings: (allocation × 100 - payment) / 1000
                 penalty-value avoided minus cost paid; encodes deal quality
        [base+5] coverage_ratio: (banked + allocation) / max(emissions + carry_forward, 1e-6)
                 clipped to [0, 3], normalized by /3
        [base+6] normalized carry_forward: carry_forward / max(estimated_need, 1e-6)
                 clipped to [0, 3]; agents need to see their debt
        [base+7] collateral_locked_norm: this year's collateral locked / annual_budget
                 clipped to [0, 1]; immediate feedback on auction over-commitment risk
        [base+8] budget_remaining_phase2_norm: (annual_budget - budget_spent) / annual_budget
                 clipped to [-0.5, 1.0]; post-auction budget headroom
        [base+9] compliance_liability_norm: unfunded compliance cost / annual_budget
                 clipped to [0, 2.0]; signals penalty exposure
        """
        auction_savings = (allocation * 100.0 - payment) / 1000.0

        # Coverage ratio: how well-covered the agent is for compliance
        total_obligation = max(emissions + self._carry_forward, 1e-6)
        coverage_ratio = float(np.clip((banked + allocation) / total_obligation, 0.0, 3.0)) / 3.0

        # Normalized carry-forward: debt relative to estimated need
        estimated_need = max(self.compute_estimate_need(), 1e-6)
        carry_forward_norm = float(np.clip(self._carry_forward / estimated_need, 0.0, 3.0)) / 3.0

        # Budget remaining after auction phase (E1)
        budget_remaining_phase2_norm = float(np.clip(
            (self.annual_budget - self.budget_spent_this_year) / max(self.annual_budget, 1e-6),
            -0.5, 1.0,
        ))

        # Compliance liability: unfunded shortfall × penalty rate / budget (E2-E3)
        shortfall = max(0.0, estimated_need + self._carry_forward - current_holdings)
        eff_penalty = self.effective_penalty_rate(current_year)
        compliance_liability_norm = float(np.clip(
            shortfall * eff_penalty / max(self.annual_budget, 1e-6),
            0.0, 2.0,
        ))

        extra = np.array([
            allocation / 5.0,                                                       # [base+0]
            clearing_price / self._price_norm,                                      # [base+1]
            (banked + allocation - emissions - self._carry_forward) / 5.0,          # [base+2]
            float(emission_shock),                                                  # [base+3] P5
            float(auction_savings),                                                 # [base+4]
            coverage_ratio,                                                         # [base+5]
            carry_forward_norm,                                                     # [base+6]
            float(np.clip(collateral_locked_norm, 0.0, 1.0)),                      # [base+7]
            budget_remaining_phase2_norm,                                           # [base+8]
            compliance_liability_norm,                                              # [base+9]
        ], dtype=np.float32)
        return np.concatenate([obs_phase1, extra])

    @property
    def obs_dim_phase1(self) -> int:
        """33 base dims + 6*(N_total-1) opponent dims when opponent modeling is enabled.
        N_total = learning agents + bot agents (all market participants).
        Base dims include carry-forward at [20], TNAC proxy at [21],
        effective reserve at [22], THIS YEAR's auction volume ratio at [23],
        MSR reserve signal at [24], own bank ratio at [25], predicted
        MSR withholding at [26], budget headroom at [27],
        suspension_remaining_norm at [28], collateral_load_last at [29],
        bid_affordability_last at [30], loan_outstanding_norm at [31],
        years_under_loan_norm at [32].
        Opponent dims: emissions, carry_forward, green_frac, fossil_frac, queue_total, is_active."""
        if self._opponent_modeling and self._n_total > 1:
            return 33 + 6 * (self._n_total - 1)
        return 33

    @property
    def obs_dim_phase2(self) -> int:
        """obs_dim_phase1 + 10 (allocation, price, net_compliance_pos, emission_shock,
        auction_savings, coverage_ratio, carry_forward_norm, collateral_locked_norm,
        budget_remaining_phase2_norm, compliance_liability_norm)."""
        return self.obs_dim_phase1 + 10

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_mix: List[float]):
        self.mix = np.array(initial_mix, dtype=np.float64)
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

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
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional

N_TECHS = 5
TECH_NAMES = ["coal", "gas", "onshore_wind", "offshore_wind", "solar"]
TECH_IDX = {name: i for i, name in enumerate(TECH_NAMES)}


class Company:

    def __init__(self, agent_id: int, config: dict, initial_mix: List[float], rng):
        self.agent_id = agent_id
        self.rng = rng

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
        self.deploy_delays = np.array(tech_cfg["deploy_delays"], dtype=np.int32)  # years
        self.operational_costs = np.array(tech_cfg["operational_costs"], dtype=np.float64)  # €/MWh
        self.decommission_costs = np.array(tech_cfg["decommission_costs"], dtype=np.float64)  # €/kW
        self.is_green = np.array(tech_cfg["is_green"], dtype=bool)
        self.is_buildable = np.array(tech_cfg["is_buildable"], dtype=bool)

        # Investment parameters
        self.max_invest_frac = inv_cfg["max_invest_frac"]
        self.convexity_alpha = inv_cfg["convexity_alpha"]
        self.penalty_rate = pen_cfg["rate"]

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

        # State: technology mix vector [coal, gas, onshore, offshore, solar]
        self.mix = np.array(initial_mix, dtype=np.float64)
        assert abs(self.mix.sum() - 1.0) < 1e-6, f"Mix must sum to 1.0, got {self.mix.sum()}"
        self.prev_green_frac = self.green_frac

        # Construction queue: list of {tech_idx, frac_delta, completion_year, success}
        self._construction_queue: List[Dict] = []
        self._consecutive_successes = 0
        self.year_cost = 0.0

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
        """Annual emissions in Mt CO2."""
        # output_mwh × Σ(mix_T × EF_T) → MWh × tCO2/MWh = tCO2 → /1e6 = Mt
        return self.output_mwh * self.weighted_emission_factor / 1e6

    def compute_risk_factor(self) -> float:
        return self._compute_p_fail()

    def compute_estimate_need(self) -> float:
        return self.compute_emissions() * (1.0 + self.compute_risk_factor())

    # ------------------------------------------------------------------
    # Operational costs
    # ------------------------------------------------------------------

    def compute_operational_cost(self) -> float:
        """Annual operational cost in M€ (fuel + O&M, excl. ETS)."""
        # output_mwh × Σ(mix_T × opex_T) → MWh × €/MWh = € → /1e6 = M€
        return self.output_mwh * float(np.dot(self.mix, self.operational_costs)) / 1e6

    def compute_ets_fuel_cost(self, ets_price: float) -> float:
        """Annual ETS cost in M€ based on current mix and carbon price."""
        # emissions_Mt × price_€/t × 1e6 → M€... actually:
        # emissions in Mt = millions of tonnes, price in €/t
        # cost = Mt × 1e6 t/Mt × €/t = M€ × 1e0... let's be explicit:
        return self.compute_emissions() * ets_price  # Mt × €/t = M€ (since Mt=1e6t, €/t, result is M€)

    # ------------------------------------------------------------------
    # Investment (greening-only)
    # ------------------------------------------------------------------

    def _compute_p_fail(self) -> float:
        p_base = self.p_fail_min + (self.p_fail_max - self.p_fail_min) * (self.fossil_frac ** self.p_fail_alpha)
        if self._consecutive_successes >= self.exp_threshold:
            p_base -= self.exp_discount
        return max(self.p_fail_min, p_base)

    def compute_investment_cost(self, tech_idx: int, frac_delta: float) -> float:
        """
        Compute real-data-grounded investment cost for adding renewable capacity.

        Cost = ΔP × CapEx × (1 + α × ΔP / TotalCapacity)

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
        return cost_eur / 1e6  # M€

    def compute_decommission_cost(self, fossil_tech_idx: int, frac_delta: float) -> float:
        """Cost to decommission fossil capacity in M€."""
        if frac_delta <= 0 or self.decommission_costs[fossil_tech_idx] <= 0:
            return 0.0
        delta_mwh = frac_delta * self.output_mwh
        cf = self.capacity_factors[fossil_tech_idx]
        delta_mw = delta_mwh / (cf * 8760)
        delta_kw = delta_mw * 1000.0
        return delta_kw * self.decommission_costs[fossil_tech_idx] / 1e6

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
        # Map buildable choice to actual tech index
        buildable_indices = [2, 3, 4]  # onshore_wind, offshore_wind, solar
        if tech_choice < 0 or tech_choice >= len(buildable_indices):
            tech_choice = 0
        tech_idx = buildable_indices[int(tech_choice)]

        frac = float(np.clip(invest_frac, 0.0, self.max_invest_frac))
        if frac < 1e-6:
            return 0.0

        # Can't invest more than remaining fossil fraction
        frac = min(frac, self.fossil_frac)
        if frac < 1e-6:
            return 0.0

        # Investment cost for new green capacity
        invest_cost = self.compute_investment_cost(tech_idx, frac)

        # Decommission cost: retire highest-emission fossil first
        decom_cost = 0.0
        frac_to_retire = frac
        # Retire coal first, then gas
        for fossil_idx in [0, 1]:  # coal, gas
            if frac_to_retire <= 0:
                break
            available = self.mix[fossil_idx]
            retire_this = min(frac_to_retire, available)
            if retire_this > 1e-6:
                decom_cost += self.compute_decommission_cost(fossil_idx, retire_this)
                frac_to_retire -= retire_this

        total_cost = invest_cost + decom_cost

        # Risk check
        p_fail = self._compute_p_fail()
        success = self.rng.random() > p_fail

        if success:
            self._consecutive_successes += 1
        else:
            self._consecutive_successes = 0

        # Add to construction queue
        delay = self.deploy_delays[tech_idx]
        self._construction_queue.append({
            "tech_idx": tech_idx,
            "frac_delta": frac if success else 0.0,
            "completion_year": current_year + delay,
            "success": success,
        })

        return total_cost

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

    def record_spending(self, amount: float):
        self.budget_spent_this_year += float(amount)

    def compute_budget_penalty(self) -> float:
        overspend = max(0.0, self.budget_spent_this_year - self.annual_budget)
        if overspend < 1e-6:
            return 0.0
        ratio = overspend / self.annual_budget
        return self.overspend_coef * (ratio ** 2) * self.annual_budget

    def get_budget_utilization(self) -> float:
        return self.budget_spent_this_year / max(self.annual_budget, 1e-6)

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------

    def settle_compliance(self, allowances_held: float) -> float:
        shortfall = max(0.0, self.compute_emissions() - allowances_held)
        return shortfall * self.penalty_rate

    # ------------------------------------------------------------------
    # Observation — Phase 1: 18D, Phase 2: 21D
    # ------------------------------------------------------------------

    def get_observation_phase1(self, year, cap_t, last_clearing_price,
                               expected_price, auction_gap=0.0,
                               last_secondary_price=0.0,
                               secondary_profit_signal=0.0):
        """
        Phase 1 observation (pre-auction): 18 dimensions.

        [0]  time (normalized)
        [1]  cap (normalized)
        [2]  last auction price (normalized)
        [3]  expected price (normalized)
        [4-8]  technology mix vector (5D)
        [9]  emissions (normalized)
        [10] estimated need with risk buffer
        [11] p_fail
        [12] investment experience
        [13] auction gap
        [14-16] construction queue (onshore, offshore, solar)
        [17] weighted emission factor (normalized)
        """
        queue = self.get_queue_capacity()
        return np.array([
            year / 10.0,                          # [0]
            cap_t / 10.0,                         # [1]
            last_clearing_price / 200.0,          # [2]
            expected_price / 200.0,               # [3]
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
            self.weighted_emission_factor,        # [17] avg EF (already 0-1 range roughly)
        ], dtype=np.float32)

    def get_observation_phase2(self, obs_phase1, allocation,
                                clearing_price, emissions, banked=0.0):
        """
        Phase 2 observation (post-auction): 21 dimensions.
        Appends auction results to phase 1 obs.
        """
        extra = np.array([
            allocation / 5.0,                              # [18]
            clearing_price / 200.0,                        # [19]
            (banked + allocation - emissions) / 5.0,       # [20]
        ], dtype=np.float32)
        return np.concatenate([obs_phase1, extra])

    @property
    def obs_dim_phase1(self) -> int:
        return 18

    @property
    def obs_dim_phase2(self) -> int:
        return 21

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_mix: List[float]):
        self.mix = np.array(initial_mix, dtype=np.float64)
        self.prev_green_frac = self.green_frac
        self._construction_queue = []
        self._consecutive_successes = 0
        self.year_cost = 0.0
        self.budget_spent_this_year = 0.0

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
        self._n_agents = config["companies"]["n_agents"]
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

    def apply_mac_switching(self, carbon_price: float) -> tuple:
        """
        MAC fuel-switching: temporarily switch coal dispatch to gas when
        carbon price exceeds the marginal abatement cost.

        Returns (emissions_reduction_Mt, cost_M€).
        Does NOT modify the permanent capacity mix.
        """
        if not self._mac_enabled or carbon_price <= self._mac_cost:
            return 0.0, 0.0

        switchable = min(self.mix[0], self._mac_max_switch)
        if switchable < 1e-6:
            return 0.0, 0.0

        ef_reduction = self.emission_factors[0] - self.emission_factors[1]  # tCO2/MWh
        switched_mwh = switchable * self.output_mwh
        emissions_reduction = switched_mwh * ef_reduction / 1e6  # Mt
        cost = emissions_reduction * self._mac_cost  # M€
        return emissions_reduction, cost

    def compute_risk_factor(self) -> float:
        return self._compute_p_fail()

    def compute_estimate_need(self) -> float:
        return self.compute_emissions() * (1.0 + self.compute_risk_factor())

    # ------------------------------------------------------------------
    # Operational costs
    # ------------------------------------------------------------------

    def compute_operational_cost(self) -> float:
        """Annual operational cost in M€ (fuel + O&M, excl. ETS)."""
        return self.output_mwh * float(np.dot(self.mix, self.operational_costs)) / 1e6

    def compute_ets_fuel_cost(self, ets_price: float) -> float:
        """Annual ETS cost in M€ based on current mix and carbon price."""
        return self.compute_emissions() * ets_price

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
        invest_cost = self.compute_investment_cost(tech_idx, frac)

        # Decommission cost: retire highest-emission fossil first
        decom_cost = 0.0
        frac_to_retire = frac
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

    def settle_compliance_realized(self, allowances_held: float, realized_emissions: float) -> float:
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
        return shortfall * self.penalty_rate

    # ------------------------------------------------------------------
    # Observations — Phase 1: 20D base (+6 opponent) | Phase 2: +4
    # ------------------------------------------------------------------

    def get_observation_phase1(self, year, cap_t, last_clearing_price,
                               expected_price, auction_gap=0.0,
                               last_secondary_price=0.0,
                               secondary_profit_signal=0.0,
                               price_ma3=None,
                               opponent_obs=None,
                               last_secondary_volume=0.0):
        """
        Phase 1 observation (pre-auction): 20D base + 2*(N-1) opponent dims.

        Base 20 dims:
        [0]  time (normalized)
        [1]  cap (normalized)
        [2]  3-year moving average of clearing price (normalized)
        [3]  expected price from AR(1) model (normalized)
        [4-8]  technology mix vector (5D)
        [9]  emissions (normalized)
        [10] estimated need with risk buffer
        [11] p_fail
        [12] investment experience
        [13] auction gap (banked allowances)
        [14-16] construction queue (onshore, offshore, solar)
        [17] weighted emission factor (normalized)
        [18] last secondary market price (normalized)  -- P8
        [19] last secondary market volume (normalized) -- P8

        Opponent dims (if opponent_modeling enabled):
        [20..25] = (bid_j/200, green_j) for each other agent j
        """
        price_signal = (price_ma3 if price_ma3 is not None else last_clearing_price)
        queue = self.get_queue_capacity()
        pn = self._price_norm  # normalization constant (= price_max)
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
        ], dtype=np.float32)
        if opponent_obs is not None and len(opponent_obs) > 0:
            return np.concatenate([base, opponent_obs])
        return base

    def get_observation_phase2(self, obs_phase1, allocation,
                                clearing_price, emissions, banked=0.0,
                                emission_shock=0.0, payment=0.0):
        """
        Phase 2 observation (post-auction): obs_phase1 + 5 extra dims.
        Appends auction results + P5 emission shock + auction_savings signal.

        Extra dims:
        [base+0] allocation / 5
        [base+1] clearing_price / price_max
        [base+2] net compliance position: (banked + allocation - emissions - carry_forward) / 5
                 <0 means the agent is still short after using all holdings
        [base+3] emission_shock (realized deviation from base need)  -- P5
        [base+4] auction_savings: (allocation × 100 - payment) / 1000
                 penalty-value avoided minus cost paid; encodes deal quality
        """
        auction_savings = (allocation * 100.0 - payment) / 1000.0
        extra = np.array([
            allocation / 5.0,                                                       # [base+0]
            clearing_price / self._price_norm,                                      # [base+1]
            (banked + allocation - emissions - self._carry_forward) / 5.0,          # [base+2]
            float(emission_shock),                                                  # [base+3] P5
            float(auction_savings),                                                 # [base+4]
        ], dtype=np.float32)
        return np.concatenate([obs_phase1, extra])

    @property
    def obs_dim_phase1(self) -> int:
        """21 base dims + 2*(N-1) opponent dims when opponent modeling is enabled.
        Base dims include carry-forward obligation at index [20]."""
        if self._opponent_modeling and self._n_agents > 1:
            return 21 + 2 * (self._n_agents - 1)
        return 21

    @property
    def obs_dim_phase2(self) -> int:
        """obs_dim_phase1 + 5 (allocation, price, net_compliance_pos, emission_shock, auction_savings)."""
        return self.obs_dim_phase1 + 5

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
        self._carry_forward = 0.0

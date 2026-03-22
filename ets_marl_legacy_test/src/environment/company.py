"""
company.py
==========
Represents the state of a single energy company in the ETS simulation.

Tracks:
  - Energy mix (fossil%, green%)
  - Investment pipeline (delayed green investments)
  - Investment success history (drives risk factor)
  - Emissions and compliance state
"""

import numpy as np
from collections import deque


class Company:
    """
    State container for one energy company.

    Parameters
    ----------
    agent_id : int
    config : dict
        Full YAML config dict.
    initial_fossil_frac : float
        Starting fossil fraction (e.g. 0.90 for A1).
    rng : np.random.Generator
        Seeded random number generator (passed from environment).
    """

    def __init__(
        self,
        agent_id: int,
        config: dict,
        initial_fossil_frac: float,
        rng: np.random.Generator,
    ):
        self.agent_id = agent_id
        self.rng = rng

        # --- Config shortcuts ---
        co_cfg = config["companies"]
        inv_cfg = config["investment"]
        em_cfg = config["emissions"]
        risk_cfg = config["risk"]
        pen_cfg = config["penalty"]

        self.output_twh = co_cfg["output_twh"]               # TWh/year
        self.ef_gas = em_cfg["ef_gas_cc"]                    # tCO2/MWh
        self.max_delta_green = inv_cfg["max_delta_green_pp"] # p.p./year
        self.delay_years = inv_cfg["delay_years"]            # years
        self.cost_a = inv_cfg["switching_cost_a"]            # M€/p.p.
        self.cost_b = inv_cfg["switching_cost_b"]            # M€/p.p.^2
        self.base_fail_prob = inv_cfg["base_failure_prob"]
        self.base_risk = risk_cfg["base_risk"]
        self.init_success_rate = risk_cfg["initial_success_rate"]
        self.penalty_rate = pen_cfg["rate"]                  # €/tCO2

        # Reward weights [w_cost, w_green]
        w = config["companies"]["reward_weights"][agent_id]
        self.w_cost = w[0]
        self.w_green = w[1]

        # --- Initial state ---
        self.fossil_frac = initial_fossil_frac
        self.green_frac = 1.0 - initial_fossil_frac

        # Investment pipeline: deque of investment dicts
        self._investment_pipeline: deque = deque()

        # Success history for risk factor computation
        self._investment_history = []   # list of bool
        self.success_rate = self.init_success_rate

        # Running cost accumulator for current year
        self.year_cost = 0.0

    # ------------------------------------------------------------------
    # Emissions and compliance
    # ------------------------------------------------------------------

    def compute_emissions(self) -> float:
        """Annual CO2 emissions based on current energy mix. Returns Mt CO2."""
        return self.output_twh * self.fossil_frac * self.ef_gas

    def compute_risk_factor(self) -> float:
        """
        Risk factor: fraction by which the agent over-bids at auction.
        Higher for brown companies with low know-how.
        """
        return self.base_risk * self.fossil_frac * (1.0 - self.success_rate)

    def compute_estimate_need(self) -> float:
        """Carbon need estimate = actual_emissions * (1 + risk_factor)."""
        actual = self.compute_emissions()
        risk = self.compute_risk_factor()
        return actual * (1.0 + risk)

    # ------------------------------------------------------------------
    # Investment
    # ------------------------------------------------------------------

    def plan_investment(self, delta_green: float) -> float:
        """
        Queue a green investment.

        Returns switching_cost (M€).
        """
        delta = float(np.clip(delta_green, 0.0, self.max_delta_green))

        # Switching cost (convex): C = a*delta + b*delta^2
        cost = self.cost_a * delta + self.cost_b * (delta ** 2)

        # Stochastic failure: brown companies fail more
        p_fail = self.base_fail_prob * self.fossil_frac * (1.0 - self.success_rate)
        success = self.rng.random() > p_fail

        self._investment_pipeline.append({
            "delta_green": delta if success else 0.0,
            "success": success,
            "cost": cost,
        })

        # Update rolling 10-year success rate
        self._investment_history.append(success)
        self.success_rate = float(np.mean(self._investment_history[-10:]))

        return cost

    def apply_matured_investments(self, current_year: int):
        """Apply investments that have reached maturity. Called at year start."""
        matured = []
        remaining = deque()

        for inv in self._investment_pipeline:
            inv.setdefault("years_pending", 0)
            inv["years_pending"] += 1
            if inv["years_pending"] >= self.delay_years:
                matured.append(inv)
            else:
                remaining.append(inv)

        self._investment_pipeline = remaining

        for inv in matured:
            self.green_frac = float(
                np.clip(self.green_frac + inv["delta_green"], 0.0, 1.0)
            )
            self.fossil_frac = 1.0 - self.green_frac

    # ------------------------------------------------------------------
    # Allowance accounting — NO BANKING
    # ------------------------------------------------------------------

    def settle_compliance(self, allowances_held: float) -> float:
        """
        End-of-year compliance check.
        Allowances not used expire (no banking).
        Returns penalty cost (€).
        """
        actual_need = self.compute_emissions()
        shortfall = max(0.0, actual_need - allowances_held)
        penalty_cost = shortfall * self.penalty_rate
        # surplus expires — no banking, no makeup_deficit
        return penalty_cost

    # ------------------------------------------------------------------
    # Observation vector (11-dimensional)
    # ------------------------------------------------------------------

    def get_observation(
        self,
        year: int,
        cap_t: float,
        last_clearing_price: float,
        expected_price: float,
        auction_gap: float = 0.0,
        last_secondary_price: float = 0.0,
    ) -> np.ndarray:
        """
        Returns the 11-dimensional observation vector for this agent.

        Features:
          0  year_norm                : year / 10
          1  cap_norm                 : cap_t / 10
          2  last_clearing_price_norm : last_clearing_price / 200
          3  expected_price_norm      : expected_price / 200
          4  green_frac               : current green fraction [0,1]
          5  emissions_norm           : compute_emissions() / 10
          6  estimate_need_norm       : compute_estimate_need() / 10
          7  success_rate             : rolling investment success [0,1]
          8  risk_factor              : compute_risk_factor()
          9  auction_gap_norm         : gap post-auction / 5
          10 last_secondary_price_norm: last secondary market price / 200
        """
        obs = np.array([
            year / 10.0,
            cap_t / 10.0,
            last_clearing_price / 200.0,
            expected_price / 200.0,
            self.green_frac,
            self.compute_emissions() / 10.0,
            self.compute_estimate_need() / 10.0,
            self.success_rate,
            self.compute_risk_factor(),
            auction_gap / 5.0,
            last_secondary_price / 200.0,
        ], dtype=np.float32)
        return obs

    @property
    def obs_dim(self) -> int:
        return 11

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_fossil_frac: float):
        """Reset company state for a new episode."""
        self.fossil_frac = initial_fossil_frac
        self.green_frac = 1.0 - initial_fossil_frac
        self._investment_pipeline = deque()
        self._investment_history = []
        self.success_rate = self.init_success_rate
        self.year_cost = 0.0

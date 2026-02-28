"""
company.py
==========
Represents the state of a single energy company in the ETS simulation.

Risk model:
  p_fail = p_min + (p_max - p_min) * fossil_frac^alpha
  If the company has had `experience_threshold` consecutive successes,
  p_fail is reduced by `experience_discount` (floored at p_min).

Diminishing returns:
  delta_effective = delta_chosen * fossil_frac^beta
  As a company becomes greener, the remaining fossil capacity is harder
  to convert (marginal renewable resources are more expensive/difficult).
"""

import numpy as np
from collections import deque


class Company:

    def __init__(self, agent_id, config, initial_fossil_frac, rng):
        self.agent_id = agent_id
        self.rng = rng

        co_cfg   = config["companies"]
        inv_cfg  = config["investment"]
        em_cfg   = config["emissions"]
        risk_cfg = config["risk"]
        pen_cfg  = config["penalty"]

        self.output_twh        = co_cfg["output_twh"]
        self.ef_gas            = em_cfg["ef_gas_cc"]
        self.max_delta_green   = inv_cfg["max_delta_green_pp"]
        self.delay_years       = inv_cfg["delay_years"]
        self.cost_a            = inv_cfg["switching_cost_a"]
        self.cost_b            = inv_cfg["switching_cost_b"]
        self.penalty_rate      = pen_cfg["rate"]

        # Risk curve parameters
        self.p_fail_min       = risk_cfg["p_fail_min"]           # floor (e.g. 0.08)
        self.p_fail_max       = risk_cfg["p_fail_max"]           # ceiling (e.g. 0.65)
        self.p_fail_alpha     = risk_cfg["p_fail_alpha"]         # curvature (e.g. 0.7)
        self.exp_discount     = risk_cfg["experience_discount"]  # e.g. 0.10
        self.exp_threshold    = risk_cfg["experience_threshold"] # e.g. 2

        # Diminishing returns on green investment
        self.diminishing_beta = inv_cfg.get("diminishing_beta", 0.5)

        w = config["companies"]["reward_weights"][agent_id]
        self.w_cost  = w[0]
        self.w_green = w[1]

        self.fossil_frac     = initial_fossil_frac
        self.green_frac      = 1.0 - initial_fossil_frac
        self.prev_green_frac = self.green_frac  # for green_bonus in reward

        self._investment_pipeline: deque = deque()
        self._consecutive_successes = 0
        self.year_cost = 0.0

    # ------------------------------------------------------------------
    # Emissions
    # ------------------------------------------------------------------

    def compute_emissions(self) -> float:
        """Annual emissions in Mt CO2."""
        return self.output_twh * self.fossil_frac * self.ef_gas

    def compute_risk_factor(self) -> float:
        """Current failure probability for green investment."""
        return self._compute_p_fail()

    def compute_estimate_need(self) -> float:
        """Estimated allowance need including risk buffer."""
        return self.compute_emissions() * (1.0 + self.compute_risk_factor())

    # ------------------------------------------------------------------
    # Investment
    # ------------------------------------------------------------------

    def _compute_p_fail(self) -> float:
        """
        Failure probability based on fossil fraction + experience bonus.

        p_fail = p_min + (p_max - p_min) * fossil_frac^alpha
        Minus experience discount if enough consecutive successes.
        """
        p_base = self.p_fail_min + (self.p_fail_max - self.p_fail_min) * (self.fossil_frac ** self.p_fail_alpha)

        if self._consecutive_successes >= self.exp_threshold:
            p_base -= self.exp_discount

        return max(self.p_fail_min, p_base)

    def plan_investment(self, delta_green: float) -> float:
        """
        Plan a green investment for this year.

        The agent pays the switching cost regardless of success/failure
        (cost of attempting the transition). The actual green shift
        only happens if the investment succeeds AND after the delay period.

        Diminishing returns: the effective delta is scaled by fossil_frac^beta.
        When the company is already mostly green, further conversion is harder
        (the easy renewable resources have been exploited, only marginal ones remain).

        Returns
        -------
        cost : float
            Switching cost in M€.
        """
        delta = float(np.clip(delta_green, 0.0, self.max_delta_green))

        # Diminishing returns: harder to convert the last fossil capacity
        scale = self.fossil_frac ** self.diminishing_beta
        delta_effective = delta * scale

        cost = self.cost_a * delta + self.cost_b * (delta ** 2)  # cost based on intent

        p_fail  = self._compute_p_fail()
        success = self.rng.random() > p_fail

        self._investment_pipeline.append({
            "delta_green": delta_effective if success else 0.0,
            "success": success,
            "cost": cost,
        })

        # Update consecutive success counter
        if success:
            self._consecutive_successes += 1
        else:
            self._consecutive_successes = 0

        return cost

    def apply_matured_investments(self, current_year: int):
        """Apply matured investments. Saves prev_green_frac before update."""
        self.prev_green_frac = self.green_frac  # snapshot before maturation

        matured   = []
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
            self.green_frac  = float(np.clip(self.green_frac + inv["delta_green"], 0.0, 1.0))
            self.fossil_frac = 1.0 - self.green_frac

    # ------------------------------------------------------------------
    # Compliance — NO BANKING
    # ------------------------------------------------------------------

    def settle_compliance(self, allowances_held: float) -> float:
        shortfall = max(0.0, self.compute_emissions() - allowances_held)
        return shortfall * self.penalty_rate

    # ------------------------------------------------------------------
    # Observation — Phase 1: 12-dimensional, Phase 2: 15-dimensional
    # ------------------------------------------------------------------

    def get_observation_phase1(self, year, cap_t, last_clearing_price,
                               expected_price, auction_gap=0.0,
                               last_secondary_price=0.0,
                               secondary_profit_signal=0.0):
        """
        Phase 1 observation (pre-auction): 12 dimensions.
        Includes secondary_profit_signal so the auction actor knows
        that buying extra allowances can be sold at a profit.
        """
        return np.array([
            year / 10.0,                          # [0]  time
            cap_t / 10.0,                         # [1]  cap
            last_clearing_price / 200.0,          # [2]  last auction price
            expected_price / 200.0,               # [3]  expected price (AR1)
            self.green_frac,                      # [4]  own green fraction
            self.compute_emissions() / 10.0,      # [5]  own emissions
            self.compute_estimate_need() / 10.0,  # [6]  need with risk buffer
            self.compute_risk_factor(),           # [7]  p_fail
            self._consecutive_successes / 5.0,    # [8]  investment experience
            auction_gap / 5.0,                    # [9]  last year surplus/deficit
            last_secondary_price / 200.0,         # [10] last secondary price
            secondary_profit_signal / 50.0,       # [11] avg €/Mt profit from secondary
        ], dtype=np.float32)

    def get_observation_phase2(self, obs_phase1, allocation,
                                clearing_price, emissions):
        """
        Phase 2 observation (post-auction): 15 dimensions.
        Appends auction results so the secondary actor knows exactly
        how many allowances it has and what the surplus/deficit is.
        """
        extra = np.array([
            allocation / 5.0,                     # [12] allowances obtained
            clearing_price / 200.0,               # [13] this year's clearing price
            (allocation - emissions) / 5.0,       # [14] surplus (+) or deficit (-)
        ], dtype=np.float32)
        return np.concatenate([obs_phase1, extra])

    @property
    def obs_dim_phase1(self) -> int:
        return 12

    @property
    def obs_dim_phase2(self) -> int:
        return 15

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_fossil_frac: float):
        self.fossil_frac     = initial_fossil_frac
        self.green_frac      = 1.0 - initial_fossil_frac
        self.prev_green_frac = self.green_frac
        self._investment_pipeline = deque()
        self._consecutive_successes = 0
        self.year_cost    = 0.0

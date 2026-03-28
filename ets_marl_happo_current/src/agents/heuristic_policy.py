"""
heuristic_policy.py
===================
Rule-based heuristic actions for behavioral cloning warm-start.

Provides one function per decision phase that mirrors the agent's action space:

  auction_action(company, price_ma3, current_year, n_years, config)
      -> np.ndarray [bid_price, qty_multiplier, invest_frac,
                    logit_onshore, logit_offshore, logit_solar]

  secondary_action(company, bank, allocation, clearing_price, config)
      -> np.ndarray [sec_price_multiplier, sec_qty]

All outputs are in physical (action) space. The calling code in train.py
inverse-maps them through atanh for MSE supervision on the policy mean heads.

Agent objectives (4x2 factorial: archetype x objective)
-------------------------------------------------------
Even-indexed agents (0, 2, 4, 6) are *financial*: prioritize cost minimization,
invest conservatively, and sell surplus aggressively.
Odd-indexed agents (1, 3, 5, 7) are *green-objective*: invest more aggressively,
bid higher to guarantee allocation, and hold surplus rather than selling.

Design rationale
----------------
auction_action (valuation-based):
  - bid_price = min(penalty_rate_inflated,
      base_anchor * (1.4 - 0.3 * coverage_ratio))
    where coverage_ratio = bank / max(annual_need, 0.1).
    High bank -> lower bid (already covered), low bank -> bid near penalty.
    Green-objective agents add a 5% premium.
  - qty_mult: target-bank logic.
    target_bank = annual_need * min(remaining_years, 2) * 0.3
    qty_mult = clip((annual_need - bank + target_bank) / annual_need, low, high)
  - invest_frac: NPV-gated.
    avoided_carbon_npv = emission_reduction * price * effective_horizon
    invest proportional to NPV (higher for green agents).
  - tech choice: maximize (remaining_years - delay + terminal_horizon)
    * capacity_factor / capex  (effective payoff metric).

secondary_action (target-bank trajectory):
  - target_bank = annual_need * min(remaining_years - 1, 2) * 0.3
  - trade_target = (target_bank - bank) * 0.5
    Positive -> buy, negative -> sell.
  - Price multiplier scales by deficit/surplus severity.
"""

import numpy as np


# buildable tech indices inside company.mix: 2=onshore, 3=offshore, 4=solar
_TECH_ONSHORE = 2
_TECH_OFFSHORE = 3
_TECH_SOLAR = 4
_BUILDABLE = [_TECH_ONSHORE, _TECH_OFFSHORE, _TECH_SOLAR]


def auction_action(
    company,
    price_ma3: float,
    current_year: int,
    n_years: int,
    config: dict,
    reserve_price: float = None,
    inflation_factor: float = None,
    bank: float = None,
) -> np.ndarray:
    """
    Heuristic Phase-1 (auction + investment) action.

    Parameters
    ----------
    company : Company
        The company object for this agent.
    price_ma3 : float
        3-year moving average of clearing price (EUR/t).
    current_year : int
        Current year index (0-based).
    n_years : int
        Total episode length.
    config : dict
        Full training config.
    reserve_price : float, optional
        Dynamic reserve price override.
    inflation_factor : float, optional
        Cumulative inflation factor override.
    bank : float, optional
        Banked allowances (Mt) held by this agent. If None, defaults to 0.

    Returns
    -------
    action : np.ndarray, shape (6,)
        [bid_price, qty_mult, invest_frac, logit_onshore, logit_offshore, logit_solar]
        in physical space.
    """
    aq = config["auction"]
    inv = config["investment"]
    if reserve_price is None:
        reserve_price = config["ets"].get("reserve_price", 0.0)
    is_green = (company.agent_id % 2) == 1  # odd indices = green-objective

    # --- Penalty rate (valuation ceiling) ---
    # Use the company's effective_penalty_rate which follows the same
    # stochastic inflation path as the compliance penalty for learning agents.
    penalty_rate = company.effective_penalty_rate(current_year)

    # --- Coverage ratio ---
    annual_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
    if bank is None:
        bank = 0.0
    coverage_ratio = max(bank / annual_need, 0.0)

    # --- Bid price (valuation-based) ---
    # base_anchor blends penalty rate with MA3 signal (40/60 to avoid penalty-ceiling saturation)
    base_anchor = 0.4 * penalty_rate + 0.6 * max(price_ma3, reserve_price)
    # High coverage -> bid less aggressively; low coverage -> bid near penalty
    bid_price = min(penalty_rate, base_anchor * (1.4 - 0.3 * min(coverage_ratio, 3.0)))
    if is_green:
        bid_price *= 1.05  # green premium to ensure allocation
    bid_price = float(np.clip(
        max(reserve_price + 5.0, bid_price),
        aq["price_min"], aq["price_max"],
    ))

    # --- Quantity multiplier (target-bank logic) ---
    remaining_years = max(1, n_years - current_year)
    target_bank = annual_need * min(remaining_years, 2) * 0.3
    qty_mult = (annual_need - bank + target_bank) / max(annual_need, 0.1)
    qty_mult = float(np.clip(
        qty_mult, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0),
    ))

    # --- Investment fraction (NPV-gated) ---
    terminal_horizon = config.get("reward", {}).get("terminal_payoff_years", 5)
    tech_cfg = config["technologies"]
    deploy_delays = tech_cfg["deploy_delays"]
    capacity_factors = tech_cfg["capacity_factors"]
    capex_arr = tech_cfg["capex"]

    # Pick best buildable tech by effective payoff metric
    best_tech = _TECH_SOLAR  # default
    best_score = -1.0
    for t in _BUILDABLE:
        effective_years = remaining_years - deploy_delays[t] + terminal_horizon
        if effective_years <= 0:
            continue
        score = effective_years * capacity_factors[t] / max(capex_arr[t], 1.0)
        if score > best_score:
            best_score = score
            best_tech = t

    # NPV of avoided carbon
    ef_saved = max(0.0, company.weighted_emission_factor - company.emission_factors[best_tech])
    effective_horizon = max(0, remaining_years - deploy_delays[best_tech] + terminal_horizon)
    frac_test = 0.07 if is_green else 0.03
    annual_emission_reduction = frac_test * company.output_mwh * ef_saved / 1e6  # Mt
    avoided_carbon_npv = annual_emission_reduction * price_ma3 * effective_horizon  # M EUR
    invest_cost = company.compute_investment_cost(best_tech, frac_test, current_year)  # M EUR

    if is_green:
        # Green agents: invest proportionally to NPV ratio, minimum floor
        npv_ratio = avoided_carbon_npv / max(invest_cost, 1e-6)
        invest_frac = float(np.clip(
            frac_test * min(npv_ratio, 2.0) / 2.0 + 0.02,
            0.02, inv["max_invest_frac"],
        ))
    else:
        # Financial agents: strict NPV gate
        npv_ratio = avoided_carbon_npv / max(invest_cost, 1e-6)
        if npv_ratio > 1.0:
            invest_frac = float(np.clip(
                frac_test * min(npv_ratio, 2.0) / 2.0,
                0.005, inv["max_invest_frac"],
            ))
        else:
            invest_frac = 0.005

    # --- Capex throughput check ---
    # Scale down invest_frac if estimated cost exceeds remaining capex capacity
    capex_tp = getattr(company, 'capex_throughput', 1e9)
    capex_spent = getattr(company, 'capex_spent_this_year', 0.0)
    capex_remaining = max(0.0, capex_tp - capex_spent)
    est_cost = company.compute_investment_cost(best_tech, invest_frac, current_year)
    if est_cost > capex_remaining and est_cost > 1e-6:
        invest_frac *= capex_remaining / est_cost
        invest_frac = max(0.0, invest_frac)

    # --- Technology choice (logits) ---
    # Use the best_tech selected by effective payoff metric
    logits = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    tech_logit_idx = best_tech - _TECH_ONSHORE  # 0=onshore, 1=offshore, 2=solar
    logits[tech_logit_idx] = 1.0

    return np.array([bid_price, qty_mult, invest_frac, *logits], dtype=np.float32)


def secondary_action(
    company,
    bank: float,
    allocation: float,
    clearing_price: float,
    config: dict,
    current_year: int = 0,
    n_years: int = 12,
) -> np.ndarray:
    """
    Heuristic Phase-2 (secondary market) action.

    Parameters
    ----------
    company : Company
        The company object for this agent (used for need estimation).
    bank : float
        Banked allowances held at the start of this year (Mt), before compliance.
    allocation : float
        Allowances received at the primary auction (Mt).
    clearing_price : float
        Auction clearing price (EUR/t); used to scale the price multiplier target.
    config : dict
        Full training config.
    current_year : int
        Current year index (0-based).
    n_years : int
        Total episode length.

    Returns
    -------
    action : np.ndarray, shape (2,)
        [sec_price_multiplier, sec_qty] in physical space.
    """
    aq = config["auction"]
    qty_max = aq["quantity_max"]
    is_green = (company.agent_id % 2) == 1  # odd indices = green-objective

    need = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
    remaining_years = max(1, n_years - current_year)

    # Target-bank trajectory: hold a buffer of allowances for future years
    target_bank = need * min(remaining_years - 1, 2) * 0.3
    current_position = bank + allocation - need  # surplus after this year's compliance
    trade_target = (target_bank - current_position) * 0.5  # positive = buy, negative = sell

    # Compliance-urgency override: never sell if carrying forward debt
    if company._carry_forward > 0.01:
        trade_target = max(0.0, trade_target)

    # Scale price multiplier by deficit/surplus severity
    severity = abs(trade_target) / max(need, 0.1)  # normalized severity

    if trade_target > 0.01:
        # Need to buy
        buy_qty = min(abs(trade_target), qty_max)
        sec_qty = float(buy_qty)
        # More severe deficit -> higher premium (willing to pay more)
        if is_green:
            price_mult = 1.05 + 0.25 * min(severity, 1.0)  # 1.05-1.30
        else:
            price_mult = 1.10 + 0.20 * min(severity, 1.0)   # 1.10-1.30
    elif trade_target < -0.01:
        # Have excess -> sell
        sell_qty = min(abs(trade_target), qty_max)
        sec_qty = float(-sell_qty)
        # More surplus -> willing to sell cheaper (lower mult)
        if is_green:
            # Green agents hold more: only sell at premium
            price_mult = 1.15 + 0.10 * min(severity, 1.0)  # 1.15-1.25
        else:
            # Financial agents: sell above market (not below)
            price_mult = 1.15 + 0.10 * min(severity, 1.0)   # 1.15-1.25
    else:
        sec_qty = 0.0
        price_mult = 1.0

    sec_low = config.get("trading", {}).get("sec_mult_low", 0.8)
    sec_high = config.get("trading", {}).get("sec_mult_high", 1.3)
    price_mult = float(np.clip(price_mult, sec_low, sec_high))
    sec_qty = float(np.clip(sec_qty, -qty_max, qty_max))

    return np.array([price_mult, sec_qty], dtype=np.float32)

"""
heuristic_policy.py
===================
Rule-based heuristic actions for behavioral cloning warm-start.

Provides one function per decision phase that mirrors the agent's action space:

    auction_action(company, price_ma3, current_year, n_years, config)
            -> np.ndarray [bid_price, qty_multiplier, invest_frac,
                                        logit_onshore, logit_offshore, logit_solar]

    secondary_action(company, bank, allocation, clearing_price, config)
            -> np.ndarray [sec_price (absolute), sec_qty]

All outputs are in physical (action) space. The calling code in train.py
inverse-maps them through atanh for MSE supervision on the policy mean heads.

Agent objectives (4x2 factorial: archetype x objective)
-------------------------------------------------------
Even-indexed agents (0, 2, 4, 6) are financial: prioritize cost minimization,
invest conservatively, and sell surplus aggressively.
Odd-indexed agents (1, 3, 5, 7) are green-objective: invest more aggressively,
and hold surplus rather than selling.

Green/financial split affects investment behavior only, not auction bidding
or secondary pricing formulas.

Design rationale
----------------
auction_action (fundamentals-based):
    - market_anchor = max(mac_cost, price_ma3)
    - bid_price = market_anchor + urgency * (penalty_rate - market_anchor)
        where urgency = max(0.0, 1.0 - coverage_ratio / 1.5).
        If supply is restricted (auction_volume/cap_t < 0.8), urgency is
        boosted by max(0, 1 - supply_ratio) * 0.3.
        Covered agents bid near market anchor, desperate agents bid near penalty.
    - bid_price ceiling before clip: 1.8 * penalty_rate.
    - qty_mult: target-bank logic.
        target_bank = annual_need * min(remaining_years, 2) * 0.5
        qty_mult = clip((annual_need - bank + target_bank) / annual_need, low, high)
    - invest_frac: NPV-gated (properly discounted).
        annuity_factor = (1 - (1 + r)^-horizon) / r  where r = discount_rate
        avoided_carbon_npv = emission_reduction * price * annuity_factor
        invest proportional to NPV (higher for green agents).
    - tech choice: maximize (remaining_years - delay + terminal_horizon)
        * capacity_factor / capex  (effective payoff metric).

secondary_action (absolute-price, fundamentals-based):
    - target_bank = annual_need * min(remaining_years - 1, 2) * 0.3
    - trade_target = (target_bank - bank) * 0.5
        Positive -> buy, negative -> sell.
    - market_anchor = max(mac_cost, clearing_price)
    - Absolute price from fundamentals: market_anchor + f(urgency) *
        (penalty_rate - market_anchor).
    - Price ceiling before clip: 1.8 * penalty_rate.
    - Clipped to [sec_price_min, 2.0 * effective_penalty_rate].
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
    bank: float = 0.0,
    reserve_price: float = None,
    inflation_factor: float = None,
    auction_volume: float = None,
    cap_t: float = None,
    valuation_noise: float = 0.0,
    urgency_multiplier: float = 1.0,
    urgency_denom: float = 1.5,
    suspension_remaining: int = 0,
    suspension_length: int = 2,
    collateral_load_last: float = 0.0,
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
    auction_volume : float, optional
        THIS YEAR'S MSR-adjusted auction supply (Mt). Use the preview value
        from _get_obs_phase1 so the heuristic sees the same market size as the
        obs. Drives supply-scarcity urgency boost.
    cap_t : float, optional
        Current annual cap (Mt). Used with auction_volume for supply ratio.
    valuation_noise : float, optional
        Per-bot persistent valuation noise (EUR/t), added to market_anchor. Default: 0.0.
    urgency_multiplier : float, optional
        Per-bot persistent urgency multiplier. Default: 1.0.
    urgency_denom : float, optional
        Urgency denominator (replaces hardcoded 1.5 in coverage_ratio / 1.5). Default: 1.5.
    suspension_remaining : int, optional
        Rounds the agent is still suspended (0 = not suspended). When > 0 the
        environment forces a zero bid anyway; we return a zero bid here too so
        BC targets match enforced behaviour.
    suspension_length : int, optional
        Total suspension length in rounds (used for normalisation only).
    collateral_load_last : float, optional
        Last year's collateral locked / annual_budget [0, 1]. High values mean
        the agent over-committed; the heuristic scales qty_mult down to stay
        below the collateral budget limit and avoid future defaults.

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

    # Suspension guard: env will force zero bid, return matching zero target so
    # BC warm-start does not train the policy to bid when suspended.
    if suspension_remaining > 0:
        price_min = float(aq["price_min"])
        logits = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        return np.array([price_min, 0.0, 0.0, *logits], dtype=np.float32)

    # --- Penalty rate (valuation ceiling) ---
    pen_cfg = config.get("penalty", {})
    base_penalty = pen_cfg.get("rate", 100.0)
    if inflation_factor is None:
        infl = pen_cfg.get("inflation_rate", 0.0)
        penalty_rate = base_penalty * (1.0 + infl) ** current_year
    else:
        penalty_rate = base_penalty * float(inflation_factor)

    # --- Coverage ratio ---
    annual_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
    coverage_ratio = max(bank / annual_need, 0.0)

    # --- Bid price (fundamentals-based: MAC→penalty gradient) ---
    mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
    urgency = max(0.0, 1.0 - coverage_ratio / urgency_denom)
    urgency_boost = 0.0
    if auction_volume is not None and cap_t is not None and cap_t > 0:
        supply_ratio = float(auction_volume) / max(float(cap_t), 1e-6)
        if supply_ratio < 0.8:
            urgency_boost = max(0.0, 1.0 - supply_ratio) * 0.3
    urgency = min(1.0, (urgency + urgency_boost) * urgency_multiplier)
    market_anchor = max(mac_cost, price_ma3) + valuation_noise
    bid_price = market_anchor + urgency * (penalty_rate - market_anchor)
    bid_price = min(bid_price, 1.8 * penalty_rate)
    bid_price = float(np.clip(
        max(reserve_price + 5.0, bid_price),
        aq["price_min"], aq["price_max"],
    ))

    # --- Quantity multiplier (target-bank logic) ---
    remaining_years = max(1, n_years - current_year)
    target_bank = annual_need * min(remaining_years, 2) * 0.5
    qty_mult = (annual_need - bank + target_bank) / max(annual_need, 0.1)
    qty_mult = float(np.clip(
        qty_mult, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0),
    ))

    # E3: Pre-bid budget awareness — apply leverage gate and E2 collateral check.
    # Mirrors the environment-side E4/E2 enforcement so the heuristic respects
    # the same constraints and doesn't rely on silent post-hoc clipping.
    available_budget = max(0.0, float(company.annual_budget - company.budget_spent_this_year))
    lev_mult = float(aq.get("leverage_multiplier", 3.0))
    if lev_mult > 0.0 and bid_price > 1e-6 and available_budget > 0.0:
        max_notional_qty = lev_mult * available_budget / bid_price
        if qty_mult * annual_need > max_notional_qty:
            qty_mult = max_notional_qty / max(annual_need, 1e-6)
    coll_cfg_h = aq.get("collateral", {})
    h_coll_frac = float(coll_cfg_h.get("collateral_fraction",
                                        coll_cfg_h.get("opportunity_cost_rate", 0.05)
                                        * coll_cfg_h.get("hold_fraction", 0.02)))
    h_max_coll_share = float(coll_cfg_h.get("max_collateral_budget_share", 0.50))
    if h_coll_frac > 0.0 and bid_price > 1e-6 and available_budget > 0.0:
        projected_collateral = h_coll_frac * bid_price * qty_mult * annual_need
        max_collateral = h_max_coll_share * available_budget
        if projected_collateral > max_collateral:
            qty_mult *= max_collateral / projected_collateral

    # Collateral load safety: if last year's collateral locked was a large share
    # of the budget, scale back qty_mult proportionally to stay inside limits and
    # avoid a repeat default/suspension.  Linear fade: no reduction at 0.25,
    # full 50% reduction at 1.0.
    if collateral_load_last > 0.25:
        coll_penalty = min(0.5, (collateral_load_last - 0.25) / 0.75 * 0.5)
        qty_mult *= (1.0 - coll_penalty)

    # Final clip: clamp to [0, high] after budget constraints (budget constraint can reduce
    # below qty_mult_low when funds are tight; we allow 0 rather than force a default bid).
    qty_mult = float(np.clip(qty_mult, 0.0, aq.get("qty_mult_high", 2.0)))

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

    # NPV of avoided carbon (properly discounted using annuity formula)
    # Previously this was an undiscounted sum: `annual_reduction * price * horizon`.
    # A discount rate converts that into a true NPV so long-horizon investments
    # are not systematically over-valued relative to short-horizon ones.
    ef_saved = max(0.0, company.weighted_emission_factor - company.emission_factors[best_tech])
    effective_horizon = max(0, remaining_years - deploy_delays[best_tech] + terminal_horizon)
    frac_test = 0.07 if is_green else 0.03
    annual_emission_reduction = frac_test * company.output_mwh * ef_saved / 1e6  # Mt

    discount_rate = float(config.get("investment", {}).get("discount_rate", 0.05))
    if discount_rate > 0.0 and effective_horizon > 0:
        # Present value of an annuity: PV = PMT * (1 - (1+r)^-n) / r
        npv_factor = (1.0 - (1.0 + discount_rate) ** -effective_horizon) / discount_rate
    else:
        npv_factor = float(effective_horizon)
    avoided_carbon_npv = annual_emission_reduction * price_ma3 * npv_factor  # M EUR
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

    # Smooth year-to-year investment to avoid on/off oscillation.
    prev = getattr(company, "prev_invest_frac", 0.0)
    invest_frac = float(np.clip(0.5 * invest_frac + 0.5 * prev, 0.0, inv["max_invest_frac"]))

    # Write back to company for next iteration's EMA
    company.prev_invest_frac = invest_frac

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
    valuation_noise: float = 0.0,
    urgency_multiplier: float = 1.0,
    urgency_denom: float = 1.5,
) -> np.ndarray:
    """
    Heuristic Phase-2 (secondary market) action.

    C2/C3: Smarter secondary with compliance-risk awareness:
      - Never sells when carry_forward debt exists (C3)
      - Boosts buying in final years (C3: compliance risk)
      - Scales buy qty by budget headroom to avoid overspending (C2)
      - Green agents sell surplus at half rate (C2: strategic hold)

    Returns
    -------
    action : np.ndarray, shape (2,)
        [sec_price (absolute EUR/t), sec_qty] in physical space.
    """
    aq = config["auction"]
    qty_max = aq["quantity_max"]

    need = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
    remaining_years = max(1, n_years - current_year)
    is_green = (company.agent_id % 2) == 1

    # Target-bank trajectory: hold buffer for future years
    target_bank = need * min(remaining_years - 1, 2) * 0.3
    current_position = bank + allocation - need
    trade_target = (target_bank - current_position) * 0.5

    # C3: Final-year compliance urgency — aggressively buy in last 2 years
    if remaining_years <= 2 and current_position < 0:
        shortfall_boost = min(abs(current_position) * 1.5, qty_max)
        trade_target = max(trade_target, shortfall_boost)

    # C3: Never sell when carrying forward debt (compliance risk)
    if company._carry_forward > 0.01:
        trade_target = max(0.0, trade_target)

    # C2: Green agents are less aggressive sellers (hold strategic bank)
    if is_green and trade_target < -0.01:
        trade_target *= 0.5

    # C2: Budget headroom check — cap buy qty by remaining budget
    budget_remaining = max(
        0.0,
        float(company.annual_budget - company.budget_spent_this_year),
    )
    if trade_target > 0.01 and budget_remaining > 0:
        max_spend = budget_remaining * 0.3
        max_buy_at_price = max_spend / max(clearing_price, 1.0)
        if trade_target > max_buy_at_price:
            trade_target = max_buy_at_price

    # Fundamentals-based absolute price (MAC→penalty gradient)
    mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
    penalty_rate = company.effective_penalty_rate(current_year)
    trading_cfg = config.get("trading", {})
    sec_price_min = trading_cfg.get("sec_price_min", 30.0)
    sec_price_max_mult = trading_cfg.get("sec_price_max_mult", 2.0)
    sec_price_max = sec_price_max_mult * penalty_rate

    coverage_ratio = max((bank + allocation) / need, 0.0)
    urgency = max(0.0, (1.0 - coverage_ratio / urgency_denom) * urgency_multiplier)
    market_anchor = max(mac_cost, clearing_price) + valuation_noise
    severity = abs(trade_target) / max(need, 0.1)

    if trade_target > 0.01:
        buy_qty = min(abs(trade_target), qty_max)
        sec_qty = float(buy_qty)
        price_frac = urgency + 0.2 * min(severity, 1.0)
        sec_price = market_anchor + price_frac * (penalty_rate - market_anchor)
    elif trade_target < -0.01:
        sell_qty = min(abs(trade_target), qty_max)
        sec_qty = float(-sell_qty)
        price_frac = max(0.3, urgency) + 0.15 * min(severity, 1.0)
        sec_price = market_anchor + price_frac * (penalty_rate - market_anchor)
    else:
        sec_qty = 0.0
        sec_price = clearing_price

    sec_price = min(sec_price, 1.8 * penalty_rate)
    sec_price = float(np.clip(sec_price, sec_price_min, sec_price_max))
    sec_qty = float(np.clip(sec_qty, -qty_max, qty_max))

    return np.array([sec_price, sec_qty], dtype=np.float32)

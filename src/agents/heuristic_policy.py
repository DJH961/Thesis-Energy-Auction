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
auction_action (WTP-based):
    - market_anchor = max(mac_cost, price_ma3)
    - bid_price: WTP (willingness-to-pay) formula.
        wtp = min(0.95 * penalty_rate, market_anchor + urgency * (penalty_rate - market_anchor))
        bid_price = min(wtp, available / max(qty_for_price, 1e-6))
        where urgency = max(0.0, 1.0 - coverage_ratio / urgency_denom).
        If supply is restricted (auction_volume/cap_t < 0.8), urgency is
        boosted by max(0, 1 - supply_ratio) * 0.3.
        Budget-aware: bid price degrades gracefully when funds are tight.
    - qty_mult: physical compliance need + urgency safety buffer.
        annual_need = compute_estimate_need()  (includes carry-forward debt)
        qty_target = annual_need + 0.1 * annual_need * urgency
        qty_mult = clip(qty_target / annual_need, low, high)
    - invest_frac: NPV-gated (properly discounted) with compliance-priority clip.
        annuity_factor = (1 - (1 + r)^-horizon) / r  where r = discount_rate
        avoided_carbon_npv = emission_reduction * price * annuity_factor
        invest proportional to NPV (higher for green agents), then scaled down
        by post-compliance budget headroom to prevent overspending.
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
    collateral_load_last: float = 0.0,
    loan_outstanding_norm: float = 0.0,
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
    collateral_load_last : float, optional
        Last year's collateral locked / annual_budget [0, 1]. High values mean
        the agent over-committed; the heuristic scales qty_mult down to stay
        below the collateral budget limit and avoid future defaults.
    loan_outstanding_norm : float, optional
        Emergency loan outstanding / annual_budget [0, ∞). When > 0, the agent
        bids more conservatively and invests less to preserve cash for repayment.

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
    pen_cfg = config.get("penalty", {})
    base_penalty = pen_cfg.get("rate", 100.0)
    if inflation_factor is None:
        infl = pen_cfg.get("inflation_rate", 0.0)
        penalty_rate = base_penalty * (1.0 + infl) ** current_year
    else:
        penalty_rate = base_penalty * float(inflation_factor)

    # --- Coverage ratio ---
    annual_need = max(company.compute_estimate_need(), 0.1)
    coverage_ratio = max(bank / annual_need, 0.0)
    available = max(0.0, float(company.annual_budget - company.budget_spent_this_year))

    # --- Bid price (WTP-based: willingness-to-pay bounded by penalty cap) ---
    mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
    urgency = max(0.0, 1.0 - coverage_ratio / urgency_denom)
    urgency_boost = 0.0
    if auction_volume is not None and cap_t is not None and cap_t > 0:
        supply_ratio = float(auction_volume) / max(float(cap_t), 1e-6)
        if supply_ratio < 0.8:
            urgency_boost = max(0.0, 1.0 - supply_ratio) * 0.3
    urgency = min(1.0, (urgency + urgency_boost) * urgency_multiplier)
    market_anchor = max(mac_cost, price_ma3) + valuation_noise
    # --- C1: Target quantity (needed early for wtp_budget ceiling) ---
    qty_target_raw = annual_need + 0.1 * annual_need * urgency
    qty_mult_raw = qty_target_raw / max(annual_need, 1e-6)
    qty_mult_clipped = float(np.clip(qty_mult_raw, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0)))
    qty_clipped = qty_mult_clipped * annual_need  # EUR-denominator for wtp_budget

    # --- C1: Mid bid price — dual-ceiling WTP ---
    # Economic ceiling: penalty + expected future price incentivises buying before penalty
    expected_future_price = price_ma3
    wtp_economic = market_anchor + urgency * max(0.0, penalty_rate + expected_future_price - market_anchor)
    wtp_economic = min(wtp_economic, penalty_rate + expected_future_price - 1.0)
    # Budget ceiling: agent cannot commit more than max_compliance_share of available budget to compliance
    max_compliance_share = config.get("bots", {}).get("max_compliance_share", 0.70)
    wtp_budget = max_compliance_share * available / max(qty_clipped, 1e-6)
    bid_price = max(min(wtp_economic, wtp_budget), float(reserve_price) + 1.0)
    bid_price = float(np.clip(bid_price, aq["price_min"], aq["price_max"]))
    # Store for diagnostics
    company._last_wtp_economic = float(wtp_economic)
    company._last_wtp_budget = float(wtp_budget)
    company._last_wtp_binding = "economic" if wtp_economic <= wtp_budget else "budget"
    company._last_bid_price_heuristic = bid_price

    # --- Qty target: physical compliance need + urgency safety buffer ---
    remaining_years = max(1, n_years - current_year)
    qty_target = qty_target_raw  # already computed above for wtp_budget ceiling
    # carry_fwd is intentionally added again as an over-buying safety buffer when in debt
    # Clip to action-space bounds (never below zero unless suspended)
    qty_mult = qty_mult_clipped
    # Store for diagnostics
    company._last_qty_target = float(qty_target)

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

    # Compliance-priority investment: scale invest_frac down by post-compliance headroom.
    # Degrades investment gracefully under budget stress (coal bots invest less, funded bots invest fully).
    # Expected settlement ≈ mac_cost (market equilibrium anchor), not bid_price or price_ma3.
    invest_frac_pre_clip = invest_frac
    est_invest_cost = company.compute_investment_cost(best_tech, invest_frac, current_year)
    safety_reserve = 0.05 * float(company.annual_budget)
    expected_settlement_price = max(float(reserve_price), mac_cost)
    expected_compliance_cost = qty_mult * annual_need * expected_settlement_price
    post_compliance_headroom = max(0.0, available - expected_compliance_cost - safety_reserve)
    if est_invest_cost > 1e-6 and invest_frac > 1e-9:
        capex_per_unit_frac = est_invest_cost / invest_frac
        invest_frac = min(invest_frac, post_compliance_headroom / max(capex_per_unit_frac, 1e-9))
    invest_frac = max(0.0, invest_frac)
    # Store pre/post for diagnostics
    company._last_invest_frac_pre_compliance_clip = invest_frac_pre_clip
    company._last_invest_frac_post_compliance_clip = invest_frac

    # When emergency loan is outstanding, scale back qty and investment to preserve
    # cash for repayment.
    if loan_outstanding_norm > 0.05:
        loan_pressure = min(loan_outstanding_norm, 1.0)
        # Reduce qty by up to 20% (bounded) proportional to loan burden
        qty_mult *= (1.0 - min(0.20, 0.3 * loan_pressure))
        # Reduce investment by up to 50% proportional to loan burden
        invest_frac *= (1.0 - 0.5 * loan_pressure)

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
    loan_outstanding_norm: float = 0.0,
) -> np.ndarray:
    """
    Heuristic Phase-2 (secondary market) action.

        C2/C3: Smarter secondary with compliance-risk awareness:
      - Never sells when carry_forward debt exists (C3)
      - Boosts buying in final years (C3: compliance risk)
      - Scales buy qty by budget headroom to avoid overspending (C2)

    Returns
    -------
    action : np.ndarray, shape (2,)
        [sec_price (absolute EUR/t), sec_qty] in physical space.
    """
    aq = config["auction"]
    qty_max = aq["quantity_max"]

    need = max(company.compute_estimate_need(), 1e-6)
    remaining_years = max(1, n_years - current_year)

    # Target-bank trajectory: hold buffer for future years
    target_bank = need * min(remaining_years - 1, 2) * 0.3
    current_position = bank + allocation - need
    trade_target = (target_bank - current_position) * 0.5

    # Final-year compliance urgency — aggressively buy in last 2 years
    if remaining_years <= 2 and current_position < 0:
        shortfall_boost = min(abs(current_position) * 1.5, qty_max)
        trade_target = max(trade_target, shortfall_boost)

    # Never sell when carrying forward debt (compliance risk)
    if company._carry_forward > 0.01:
        trade_target = max(0.0, trade_target)

    # Budget headroom check — cap buy qty by remaining budget.
    # When already short (compliance debt), allow 60% of remaining budget to
    # accelerate debt recovery; otherwise keep the conservative 30% cap.
    budget_remaining = max(
        0.0,
        float(company.annual_budget - company.budget_spent_this_year),
    )
    if trade_target > 0.01 and budget_remaining > 0:
        if company._carry_forward > 0.01:
            spend_frac = 0.9  # aggressive recovery when carry_forward debt exists
        elif current_position < 0:
            spend_frac = 0.6
        else:
            spend_frac = 0.3
        max_spend = budget_remaining * spend_frac
        max_buy_at_price = max_spend / max(clearing_price, 1.0)
        if trade_target > max_buy_at_price:
            trade_target = max_buy_at_price

    # Reduce buying when emergency loan is outstanding
    if loan_outstanding_norm > 0.01 and trade_target > 0.01:
        loan_pressure = min(loan_outstanding_norm, 1.0)
        trade_target *= (1.0 - 0.4 * loan_pressure)

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

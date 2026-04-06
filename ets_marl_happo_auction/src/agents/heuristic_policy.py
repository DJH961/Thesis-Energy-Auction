"""
heuristic_policy.py
===================
Rule-based heuristic actions for behavioral cloning warm-start.

Provides one function per decision phase that mirrors the agent's action space:

    auction_action(company, price_ma3, current_year, n_years, config)
            -> np.ndarray shape (6,)
               [bid_price, qty_multiplier, invest_frac,
                logit_onshore, logit_offshore, logit_solar]
            (6D format used by bots; expanded to 10D in ets_environment.py)

    secondary_action(company, bank, allocation, clearing_price, config)
            -> np.ndarray [sec_price (absolute), sec_qty]

All outputs are in physical (action) space. The calling code in train.py
inverse-maps them through atanh for MSE supervision on the policy mean heads.

C1: NPV-aware 3-tranche demand curve
--------------------------------------
Instead of a single (price, qty) bid, the heuristic now constructs a
3-segment demand curve:
  - Tranche 1 (low price, highest priority): price near MAC/MA3, qty = 50% of
        target_bank_gap by default. Buys "cheap" allowances if market offers them.
  - Tranche 2 (mid price, core bid): price at market_anchor + urgency gradient,
        qty = 30% by default. The primary compliance bid.
    - Tranche 3 (high price, insurance): price near penalty, qty = 20% by default.
        Ensures compliance even in scarce markets at full cost.

The tranche quantity split and price spread are configurable via
config["bots"]["tranche_qty_split"] and config["bots"]["tranche_price_*"].

The (price, qty_mult) pairs are already sorted ascending by price, satisfying
the B1 tranche-sorting invariant.

C2/C3: Smarter secondary market + compliance-risk-awareness
------------------------------------------------------------
The secondary_action now accounts for:
  - Carry-forward debt: never sells when already in arrears
  - Remaining years: more aggressive buying in final years
    - Budget headroom: scales buy/sell targets to avoid overspending
"""

import numpy as np


# buildable tech indices inside company.mix: 2=onshore, 3=offshore, 4=solar
_TECH_ONSHORE = 2
_TECH_OFFSHORE = 3
_TECH_SOLAR = 4
_BUILDABLE = [_TECH_ONSHORE, _TECH_OFFSHORE, _TECH_SOLAR]


def _normalize_tranche_split(split_cfg) -> np.ndarray:
    """Return non-negative 3-way split that sums to 1."""
    split = np.array(split_cfg if split_cfg is not None else [1.0, 1.0, 1.0], dtype=float)
    if split.shape[0] != 3 or not np.all(np.isfinite(split)):
        split = np.array([1.0, 1.0, 1.0], dtype=float)
    split = np.maximum(split, 0.0)
    s = float(split.sum())
    if s <= 1e-9:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    return split / s


def _compute_tranche_urgency(bank: float, annual_need: float, current_year: int,
                             n_years: int, urgency_multiplier: float,
                             urgency_denom: float) -> float:
    """Coverage- and time-aware urgency in [0, 1] for tranche shaping."""
    need = max(float(annual_need), 1e-6)
    coverage_ratio = max(float(bank) / need, 0.0)
    coverage_urgency = max(0.0, 1.0 - coverage_ratio / max(float(urgency_denom), 1e-6))
    remaining_years = max(1, int(n_years) - int(current_year))
    late_episode_urgency = max(0.0, 1.0 - remaining_years / max(float(n_years), 1.0))
    urgency = max(coverage_urgency, late_episode_urgency)
    return float(np.clip(urgency * float(urgency_multiplier), 0.0, 1.0))


def _apply_per_tranche_budget_drop(prices: np.ndarray, qty_mults: np.ndarray,
                                   annual_need: float, available_budget: float,
                                   reserve_price: float, collateral_fraction: float) -> np.ndarray:
    """When cash-constrained, drop T1 first, then scale T2/T3 to fit budget."""
    need = max(float(annual_need), 1e-6)
    budget = max(0.0, float(available_budget))
    if budget <= 0.0:
        return np.zeros_like(qty_mults)

    q = np.maximum(qty_mults.astype(float), 0.0)
    p = prices.astype(float)

    def tranche_cost(q_mult_arr: np.ndarray, p_arr: np.ndarray) -> np.ndarray:
        qty_abs = q_mult_arr * need
        above_reserve = np.maximum(0.0, p_arr - float(reserve_price))
        unit = p_arr + float(collateral_fraction) * above_reserve
        return qty_abs * unit

    base_cost = float(np.sum(tranche_cost(q, p)))
    if base_cost <= budget + 1e-9:
        return q

    # Drop cheapest tranche first (T1 after ascending sort).
    q[0] = 0.0
    cost_after_t1 = float(np.sum(tranche_cost(q, p)))
    if cost_after_t1 <= budget + 1e-9:
        return q

    # If still too expensive, scale T2/T3 proportionally.
    rem_cost = float(np.sum(tranche_cost(q[1:], p[1:])))
    if rem_cost <= 1e-9:
        q[1:] = 0.0
        return q

    scale = float(np.clip(budget / rem_cost, 0.0, 1.0))
    q[1:] *= scale
    return q


def build_tranche_ladder(
    mid_price: float,
    total_qty_mult: float,
    config: dict,
    price_min: float,
    price_max: float,
    current_year: int,
    n_years: int,
    bank: float,
    annual_need: float,
    urgency_multiplier: float = 1.0,
    urgency_denom: float = 1.5,
    reserve_price: float = 0.0,
    available_budget: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3-tranche (price, qty_mult) arrays with configurable split/spread."""
    bot_cfg = config.get("bots", {})
    split = _normalize_tranche_split(bot_cfg.get("tranche_qty_split", [1.0, 1.0, 1.0]))

    low_mult = float(bot_cfg.get("tranche_price_low_mult", 0.90))
    high_base = float(bot_cfg.get("tranche_price_high_mult", 1.10))
    high_urgency_add = float(bot_cfg.get("tranche_price_high_urgency_add", 0.05))

    urgency = _compute_tranche_urgency(
        bank=bank,
        annual_need=annual_need,
        current_year=current_year,
        n_years=n_years,
        urgency_multiplier=urgency_multiplier,
        urgency_denom=urgency_denom,
    )

    prices = np.array([
        float(mid_price) * low_mult,
        float(mid_price),
        float(mid_price) * (high_base + high_urgency_add * urgency),
    ], dtype=float)
    prices = np.clip(prices, float(price_min), float(price_max))

    qty_mults = np.maximum(0.0, float(total_qty_mult)) * split

    if available_budget is not None:
        aq = config.get("auction", {})
        coll_cfg = aq.get("collateral", {})
        coll_frac = float(coll_cfg.get(
            "collateral_fraction",
            coll_cfg.get("opportunity_cost_rate", 0.05) * coll_cfg.get("hold_fraction", 0.02),
        ))
        qty_mults = _apply_per_tranche_budget_drop(
            prices=prices,
            qty_mults=qty_mults,
            annual_need=annual_need,
            available_budget=available_budget,
            reserve_price=reserve_price,
            collateral_fraction=coll_frac,
        )

    order = np.argsort(prices)
    return prices[order], qty_mults[order]


def _compute_npv_invest_frac(company, best_tech, frac_test, remaining_years,
                              terminal_horizon, price_ma3, current_year, config):
    """Shared NPV-gated investment fraction computation."""
    is_green = (company.agent_id % 2) == 1
    inv = config["investment"]
    tech_cfg = config["technologies"]
    deploy_delays = tech_cfg["deploy_delays"]

    ef_saved = max(0.0, company.weighted_emission_factor - company.emission_factors[best_tech])
    effective_horizon = max(0, remaining_years - deploy_delays[best_tech] + terminal_horizon)
    annual_emission_reduction = frac_test * company.output_mwh * ef_saved / 1e6  # Mt

    discount_rate = float(config.get("investment", {}).get("discount_rate", 0.05))
    if discount_rate > 0.0 and effective_horizon > 0:
        npv_factor = (1.0 - (1.0 + discount_rate) ** -effective_horizon) / discount_rate
    else:
        npv_factor = float(effective_horizon)
    avoided_carbon_npv = annual_emission_reduction * price_ma3 * npv_factor  # M EUR
    invest_cost = company.compute_investment_cost(best_tech, frac_test, current_year)  # M EUR

    if is_green:
        npv_ratio = avoided_carbon_npv / max(invest_cost, 1e-6)
        invest_frac = float(np.clip(
            frac_test * min(npv_ratio, 2.0) / 2.0 + 0.02,
            0.02, inv["max_invest_frac"],
        ))
    else:
        npv_ratio = avoided_carbon_npv / max(invest_cost, 1e-6)
        if npv_ratio > 1.0:
            invest_frac = float(np.clip(
                frac_test * min(npv_ratio, 2.0) / 2.0,
                0.005, inv["max_invest_frac"],
            ))
        else:
            invest_frac = 0.005

    # Capex throughput check
    capex_tp = getattr(company, 'capex_throughput', 1e9)
    capex_spent = getattr(company, 'capex_spent_this_year', 0.0)
    capex_remaining = max(0.0, capex_tp - capex_spent)
    est_cost = company.compute_investment_cost(best_tech, invest_frac, current_year)
    if est_cost > capex_remaining and est_cost > 1e-6:
        invest_frac *= capex_remaining / est_cost
        invest_frac = max(0.0, invest_frac)

    return invest_frac


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
    loan_outstanding_norm: float = 0.0,
) -> np.ndarray:
    """
    Heuristic Phase-1 (auction + investment) action.

    C1: Returns a 6D action that encodes a 3-segment demand curve:
      [mid_price, mid_qty_mult, invest_frac, logit_onshore, logit_offshore, logit_solar]

    The bot expansion in ets_environment.py converts this 6D to 10D by splitting
        into 3 tranches using configurable split/spread from config["bots"]:
            T1: price = mid_price × tranche_price_low_mult, qty via tranche_qty_split[0]
            T2: price = mid_price,                           qty via tranche_qty_split[1]
            T3: price = mid_price × (tranche_price_high_mult + tranche_price_high_urgency_add × urgency)
                    qty via tranche_qty_split[2]

        Under tight budgets, the ladder applies per-tranche dropping logic:
        T1 is zeroed first, then T2/T3 are scaled proportionally if still constrained.

    Parameters
    ----------
    company : Company
    price_ma3 : float  — 3-year moving average of clearing price (EUR/t)
    current_year : int — Current year index (0-based)
    n_years : int      — Total episode length
    config : dict      — Full training config
    bank : float       — Banked allowances at start of year (Mt)
    reserve_price : float, optional
    inflation_factor : float, optional
    auction_volume : float, optional — THIS YEAR'S MSR-adjusted supply (Mt)
    cap_t : float, optional
    valuation_noise : float, optional — Per-bot valuation noise (EUR/t)
    urgency_multiplier : float, optional
    urgency_denom : float, optional
    suspension_remaining : int, optional
        Rounds the agent is still suspended. Environment forces zero bid when
        suspended; heuristic returns a zero bid here too for BC target consistency.
    suspension_length : int, optional
        Total suspension length in rounds.
    collateral_load_last : float, optional
        Last year's collateral locked / annual_budget [0, 1]. High values mean
        the agent over-committed; heuristic scales qty_mult down to avoid repeat.
    loan_outstanding_norm : float, optional
        Emergency loan outstanding / annual_budget [0, ∞). When > 0, the agent
        bids more conservatively and invests less to preserve cash for repayment.

    Returns
    -------
    action : np.ndarray, shape (6,)
        [mid_price, total_qty_mult, invest_frac, logit_onshore, logit_offshore, logit_solar]
    """
    aq = config["auction"]
    inv = config["investment"]
    if reserve_price is None:
        reserve_price = config["ets"].get("reserve_price", 0.0)
    is_green = (company.agent_id % 2) == 1

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

    # --- Coverage ratio & urgency ---
    annual_need = max(company.compute_estimate_need() + company._carry_forward, 0.1)
    coverage_ratio = max(bank / annual_need, 0.0)
    mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
    urgency = max(0.0, 1.0 - coverage_ratio / urgency_denom)
    urgency_boost = 0.0
    if auction_volume is not None and cap_t is not None and cap_t > 0:
        supply_ratio = float(auction_volume) / max(float(cap_t), 1e-6)
        if supply_ratio < 0.8:
            urgency_boost = max(0.0, 1.0 - supply_ratio) * 0.3
    urgency = min(1.0, (urgency + urgency_boost) * urgency_multiplier)
    market_anchor = max(mac_cost, price_ma3) + valuation_noise

    # --- C1: Mid bid price for the core tranche ---
    # T2 = core compliance bid (MAC→penalty gradient scaled by urgency)
    bid_price = market_anchor + urgency * (penalty_rate - market_anchor)
    bid_price = min(bid_price, 1.8 * penalty_rate)
    bid_price = float(np.clip(
        max(reserve_price + 5.0, bid_price),
        aq["price_min"], aq["price_max"],
    ))

    # --- C1: Target quantity (total across all 3 tranches) ---
    remaining_years = max(1, n_years - current_year)
    target_bank = annual_need * min(remaining_years, 2) * 0.5
    # C3: Compliance risk — overshoot quantity in final years
    final_year_boost = 1.0 + 0.5 * max(0.0, 1.0 - remaining_years / max(n_years, 1))
    qty_mult = (annual_need - bank + target_bank) / max(annual_need, 0.1) * final_year_boost
    qty_mult = float(np.clip(
        qty_mult, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0),
    ))

    # E3: Pre-bid budget awareness.
    # Use a settlement-consistent cap so payment + collateral cannot exceed
    # available budget, preventing bot defaults by construction.
    available_budget = max(0.0, float(company.annual_budget - company.budget_spent_this_year))
    coll_cfg_h = aq.get("collateral", {})
    h_coll_frac = float(coll_cfg_h.get("collateral_fraction",
                                        coll_cfg_h.get("opportunity_cost_rate", 0.05)
                                        * coll_cfg_h.get("hold_fraction", 0.02)))
    if h_coll_frac > 0.0 and bid_price > 1e-6 and available_budget > 0.0:
        above_reserve = max(0.0, bid_price - float(reserve_price))
        denom = bid_price + h_coll_frac * above_reserve
        max_safe_qty = available_budget / max(denom, 1e-6)
        if qty_mult * annual_need > max_safe_qty:
            qty_mult = max_safe_qty / max(annual_need, 1e-6)

    # Collateral load safety: if last year's collateral locked was a large share
    # of the budget, scale back qty_mult proportionally to avoid a repeat default.
    # Linear fade: no reduction at 0.25, full 50% reduction at 1.0.
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

    best_tech = _TECH_SOLAR
    best_score = -1.0
    for t in _BUILDABLE:
        effective_years = remaining_years - deploy_delays[t] + terminal_horizon
        if effective_years <= 0:
            continue
        score = effective_years * capacity_factors[t] / max(capex_arr[t], 1.0)
        if score > best_score:
            best_score = score
            best_tech = t

    frac_test = 0.07 if is_green else 0.03
    invest_frac = _compute_npv_invest_frac(
        company, best_tech, frac_test, remaining_years,
        terminal_horizon, price_ma3, current_year, config,
    )

    # F1: Loan-awareness — when emergency loan outstanding, scale back qty and
    # investment to preserve cash for loan repayment.
    if loan_outstanding_norm > 0.05:
        loan_pressure = min(loan_outstanding_norm, 1.0)
        # Reduce qty by up to 20% (bounded) proportional to loan burden
        qty_mult *= (1.0 - min(0.20, 0.3 * loan_pressure))
        # Reduce investment by up to 50% proportional to loan burden
        invest_frac *= (1.0 - 0.5 * loan_pressure)

    # Smooth year-to-year investment to avoid on/off oscillation
    prev = getattr(company, "prev_invest_frac", 0.0)
    invest_frac = float(np.clip(0.5 * invest_frac + 0.5 * prev, 0.0, inv["max_invest_frac"]))
    company.prev_invest_frac = invest_frac

    # --- Technology logits ---
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
      - Scales buy/sell qty by budget headroom to avoid overspending (C2)

    Parameters
    ----------
    company : Company
    bank : float     — Banked allowances at start of year (Mt)
    allocation : float — Allowances from primary auction (Mt)
    clearing_price : float — Auction clearing price (EUR/t)
    config : dict
    current_year : int
    n_years : int
    valuation_noise : float, optional
    urgency_multiplier : float, optional
    urgency_denom : float, optional

    Returns
    -------
    action : np.ndarray, shape (2,)
        [sec_price (absolute EUR/t), sec_qty] in physical space.
    """
    aq = config["auction"]
    qty_max = aq["quantity_max"]

    need = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
    remaining_years = max(1, n_years - current_year)

    # Target-bank trajectory: hold buffer for future years
    target_bank = need * min(remaining_years - 1, 2) * 0.3
    current_position = bank + allocation - need  # surplus after this year's compliance
    trade_target = (target_bank - current_position) * 0.5

    # C3: Final-year compliance urgency — aggressively buy in last 2 years
    if remaining_years <= 2 and current_position < 0:
        shortfall_boost = min(abs(current_position) * 1.5, qty_max)
        trade_target = max(trade_target, shortfall_boost)

    # C3: Never sell when carrying forward debt (compliance risk)
    if company._carry_forward > 0.01:
        trade_target = max(0.0, trade_target)

    # C2: Budget headroom check — scale down buy qty if budget is tight
    budget_remaining = max(
        0.0,
        float(company.annual_budget - company.budget_spent_this_year),
    )
    if trade_target > 0.01 and budget_remaining > 0:
        max_spend = budget_remaining * 0.3  # use at most 30% of remaining budget in secondary
        max_buy_at_price = max_spend / max(clearing_price, 1.0)
        if trade_target > max_buy_at_price:
            trade_target = max_buy_at_price

    # F1: Loan-awareness — reduce buying when loan outstanding
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
        # Buy: price from fundamentals, higher with urgency
        buy_qty = min(abs(trade_target), qty_max)
        sec_qty = float(buy_qty)
        price_frac = urgency + 0.2 * min(severity, 1.0)
        sec_price = market_anchor + price_frac * (penalty_rate - market_anchor)
    elif trade_target < -0.01:
        # Sell excess: ask above MAC, modulated by severity
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

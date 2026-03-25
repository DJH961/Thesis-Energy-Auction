"""
heuristic_policy.py
===================
Rule-based heuristic actions for behavioral cloning warm-start.

Provides one function per decision phase that mirrors the agent's action space:

  auction_action(company, price_ma3, current_year, n_years, config)
      → np.ndarray [bid_price, qty_multiplier, invest_frac,
                    logit_onshore, logit_offshore, logit_solar]

  secondary_action(company, bank, allocation, clearing_price, config)
      → np.ndarray [sec_price_multiplier, sec_qty]

All outputs are in physical (action) space. The calling code in train.py
inverse-maps them through atanh for MSE supervision on the policy mean heads.

Agent objectives (4×2 factorial: archetype × objective)
-------------------------------------------------------
Even-indexed agents (0, 2, 4, 6) are *financial*: prioritize cost minimization,
invest conservatively, and sell surplus aggressively.
Odd-indexed agents (1, 3, 5, 7) are *green-objective*: invest more aggressively,
bid higher to guarantee allocation, and hold surplus rather than selling.

Design rationale
----------------
auction_action:
  - bid_price is floored at reserve_price + 5 to guarantee valid bids,
    anchored to 1.15× MA3 price (slight premium reflecting compliance urgency)
    and capped at penalty_rate (100 €/t).  In a uniform-price auction, bidding
    near true valuation is weakly dominant — you pay the clearing price
    regardless, so bidding higher guarantees allocation without raising cost.
    Green-objective agents bid a further 10% premium to ensure allocation
    for their transition strategy.
  - qty_multiplier = 1.0 normally; raised to 1.3 when carry-forward > 0
    (agent must cover the rolled-over shortfall).
  - invest_frac: financial agents use 0.03/0.005 (NPV-gated); green-objective
    agents use 0.07/0.02 (invest aggressively regardless of short-run NPV).
  - tech: solar (fast, 2yr) near end of episode; onshore wind (higher capacity)
    in the early years.

secondary_action:
  - Compute surplus = bank + allocation - need.
  - Financial agents: sell surplus aggressively (half at 1.1×), buy shortfall.
  - Green-objective agents: hold surplus as a buffer (only sell large excess),
    buy shortfall more aggressively (1.3× premium, up to 1.5× shortfall).
"""

import numpy as np


# buildable tech indices inside company.mix: 2=onshore, 3=offshore, 4=solar
_TECH_SOLAR = 4
_TECH_ONSHORE = 2


def auction_action(
    company,
    price_ma3: float,
    current_year: int,
    n_years: int,
    config: dict,
    reserve_price: float = None,
) -> np.ndarray:
    """
    Heuristic Phase-1 (auction + investment) action.

    Parameters
    ----------
    company : Company
        The company object for this agent.
    price_ma3 : float
        3-year moving average of clearing price (€/t).
    current_year : int
        Current year index (0-based).
    n_years : int
        Total episode length.
    config : dict
        Full training config.

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
    fossil_frac = company.fossil_frac        # high-emitters need more allowances

    # --- Bid price ---
    # In a uniform-price auction, bidding near your true valuation is
    # (weakly) dominant: you pay the clearing price regardless, so bidding
    # higher just guarantees allocation without raising your cost.
    # True valuation = min(penalty_rate, price_max).
    # We anchor near penalty_rate so BC warm-start seeds realistic compliance
    # prices from the first episodes, while still reacting to MA3.
    # High-emitters (fossil_frac > 0.5) bid more aggressively because their
    # compliance exposure is larger — they face higher penalties per Mt missed.
    penalty_rate = config.get("penalty", {}).get("rate", 100.0)

    # Urgency scaling: coal-heavy agents anchor closer to penalty rate
    if fossil_frac > 0.5:
        # High-emitters: weight penalty_rate more heavily (85%) to guarantee allocation
        penalty_weight = 0.85
        ma3_premium = 1.20
    else:
        penalty_weight = 0.70
        ma3_premium = 1.15

    ma3_anchor = price_ma3 * (ma3_premium if not is_green else ma3_premium + 0.05)
    near_penalty_anchor = penalty_weight * penalty_rate + (1.0 - penalty_weight) * ma3_anchor
    if is_green:
        near_penalty_anchor *= 1.05
    bid_price = float(np.clip(
        max(reserve_price + 5.0, min(penalty_rate * 1.05, near_penalty_anchor)),
        aq["price_min"], aq["price_max"],
    ))

    # --- Quantity multiplier ---
    # Cover full estimated need; increase when carry-forward exists or when
    # fossil fraction is high (high-emitters need a coverage buffer).
    if company._carry_forward > 1e-6:
        qty_mult = 1.5  # aggressive recovery from shortfall
    elif fossil_frac > 0.5:
        qty_mult = 1.2  # compliance buffer for high-emitters
    else:
        qty_mult = 1.0
    qty_mult = float(np.clip(
        qty_mult, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0),
    ))

    # --- Investment fraction ---
    # Simple NPV proxy: compare (years_left × annual carbon saving) to invest cost.
    # annual_carbon_saving = emission reduction from shifting invest_frac_test to solar,
    #                        valued at the current MA3 carbon price.
    # Green-objective agents invest more aggressively (higher frac, lower threshold).
    # Coal-heavy agents also benefit from higher investment to reduce future exposure.
    years_left = max(1, n_years - current_year)
    if is_green:
        frac_test = 0.07
    elif fossil_frac > 0.5:
        frac_test = 0.05  # coal-heavy financial: invest more than default to reduce exposure
    else:
        frac_test = 0.03

    invest_cost = company.compute_investment_cost(_TECH_SOLAR, frac_test)  # M€

    ef_saved = max(0.0, company.weighted_emission_factor - company.emission_factors[_TECH_SOLAR])
    annual_emission_reduction = frac_test * company.output_mwh * ef_saved / 1e6  # Mt
    annual_carbon_saving = annual_emission_reduction * price_ma3  # M€ (at MA3 price)

    if is_green:
        # Green agents invest aggressively: high frac when NPV positive, still
        # moderate when not (always push the transition).
        invest_frac = float(np.clip(
            0.07 if years_left * annual_carbon_saving > invest_cost * 0.5 else 0.02,
            0.0, inv["max_invest_frac"],
        ))
    elif fossil_frac > 0.5:
        # Coal-heavy financial: lower NPV threshold — future penalty avoidance
        # justifies early investment even at modest carbon prices.
        invest_frac = float(np.clip(
            0.05 if years_left * annual_carbon_saving > invest_cost * 0.7 else 0.02,
            0.0, inv["max_invest_frac"],
        ))
    else:
        # Financial agents invest conservatively: strict NPV gate.
        invest_frac = float(np.clip(
            0.03 if years_left * annual_carbon_saving > invest_cost else 0.005,
            0.0, inv["max_invest_frac"],
        ))

    # --- Technology choice (logits) ---
    # Solar (2yr delay) near the end when there is little time for onshore (5yr) to deliver.
    # Logits: [onshore, offshore, solar] — argmax selects the technology.
    if years_left < 5:
        logits = np.array([-1.0, -1.0,  1.0], dtype=np.float32)  # solar
    else:
        logits = np.array([ 1.0, -1.0, -1.0], dtype=np.float32)  # onshore wind

    return np.array([bid_price, qty_mult, invest_frac, *logits], dtype=np.float32)


def secondary_action(
    company,
    bank: float,
    allocation: float,
    clearing_price: float,
    config: dict,
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
        Auction clearing price (€/t); used to scale the price multiplier target.
    config : dict
        Full training config.

    Returns
    -------
    action : np.ndarray, shape (2,)
        [sec_price_multiplier, sec_qty] in physical space.
    """
    aq = config["auction"]
    qty_max = aq["quantity_max"]
    is_green = (company.agent_id % 2) == 1  # odd indices = green-objective
    fossil_frac = company.fossil_frac

    need = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
    surplus = bank + allocation - need

    if is_green:
        # Green-objective: hold surplus as compliance buffer; only sell large excess.
        if surplus > need * 0.3:
            sell_qty = min(surplus / 3.0, qty_max)
            sec_qty = float(-sell_qty)
            price_mult = 1.2  # demand higher price if selling
        elif surplus < 0:
            # Buy aggressively — cover 1.5× shortfall to build buffer
            buy_qty = min(abs(surplus) * 1.5, qty_max)
            sec_qty = float(buy_qty)
            price_mult = 1.3
        else:
            sec_qty = 0.0
            price_mult = 1.0
    else:
        # Financial: sell surplus aggressively for profit, buy shortfall.
        # High-emitters buy more aggressively to avoid carry-forward accumulation.
        if surplus > need * 0.1:
            sell_qty = min(surplus / 2.0, qty_max)
            sec_qty = float(-sell_qty)
            price_mult = 1.1
        elif surplus < 0:
            buy_multiplier = 1.3 if fossil_frac > 0.5 else 1.0
            buy_qty = min(abs(surplus) * buy_multiplier, qty_max)
            sec_qty = float(buy_qty)
            price_mult = 1.3
        else:
            sec_qty = 0.0
            price_mult = 1.0

    sec_low = config.get("trading", {}).get("sec_mult_low", 0.8)
    sec_high = config.get("trading", {}).get("sec_mult_high", 1.3)
    price_mult = float(np.clip(price_mult, sec_low, sec_high))
    sec_qty = float(np.clip(sec_qty, -qty_max, qty_max))

    return np.array([price_mult, sec_qty], dtype=np.float32)

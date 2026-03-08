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

Design rationale
----------------
auction_action:
  - bid_price is floored at reserve_price + 5 to guarantee valid bids,
    anchored to 1.15× MA3 price (slight premium reflecting compliance urgency)
    and capped at 90 to avoid overshooting early scarcity.
  - qty_multiplier = 1.0 normally; raised to 1.3 when carry-forward > 0
    (agent must cover the rolled-over shortfall).
  - invest_frac = 0.03 when the NPV-proxy of investing is positive over the
    remaining episode horizon, else 0.005 (always invest a little to explore).
  - tech: solar (fast, 2yr) near end of episode; onshore wind (higher capacity)
    in the early years.

secondary_action:
  - Compute surplus = bank + allocation - need.
  - Surplus > 10% of need  → sell half at 1.1× (take profit above clearing)
  - Deficit                → buy shortfall at 1.3× (willing to pay premium)
  - Near-balanced          → hold (neutral)
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
    reserve_price = config["ets"].get("reserve_price", 0.0)

    # --- Bid price ---
    # Floor: reserve_price + 5 ensures the bid is always accepted.
    # Ceiling: 90 avoids paying a large premium in early, low-scarcity years.
    bid_price = float(np.clip(
        max(reserve_price + 5.0, min(90.0, price_ma3 * 1.15)),
        aq["price_min"], aq["price_max"],
    ))

    # --- Quantity multiplier ---
    # Cover full estimated need; increase by 30% if there is a carry-forward.
    qty_mult = 1.3 if company._carry_forward > 1e-6 else 1.0
    qty_mult = float(np.clip(
        qty_mult, aq.get("qty_mult_low", 0.3), aq.get("qty_mult_high", 2.0),
    ))

    # --- Investment fraction ---
    # Simple NPV proxy: compare (years_left × annual carbon saving) to invest cost.
    # annual_carbon_saving = emission reduction from shifting invest_frac_test to solar,
    #                        valued at the current MA3 carbon price.
    years_left = max(1, n_years - current_year)
    frac_test = 0.03  # evaluate at 3% shift — matches candidate invest_frac

    invest_cost = company.compute_investment_cost(_TECH_SOLAR, frac_test)  # M€

    ef_saved = max(0.0, company.weighted_emission_factor - company.emission_factors[_TECH_SOLAR])
    annual_emission_reduction = frac_test * company.output_mwh * ef_saved / 1e6  # Mt
    annual_carbon_saving = annual_emission_reduction * price_ma3  # M€ (at MA3 price)

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

    need = max(company.compute_estimate_need() + company._carry_forward, 1e-6)
    surplus = bank + allocation - need

    if surplus > need * 0.1:
        # Oversupplied — sell half the surplus at a modest premium
        sell_qty = min(surplus / 2.0, qty_max)
        sec_qty = float(-sell_qty)   # negative = intent to sell
        price_mult = 1.1
    elif surplus < 0:
        # Short — buy the shortfall, willing to pay a premium
        buy_qty = min(abs(surplus), qty_max)
        sec_qty = float(buy_qty)     # positive = intent to buy
        price_mult = 1.3
    else:
        sec_qty = 0.0
        price_mult = 1.0

    price_mult = float(np.clip(price_mult, 0.5, 2.0))
    sec_qty = float(np.clip(sec_qty, -qty_max, qty_max))

    return np.array([price_mult, sec_qty], dtype=np.float32)

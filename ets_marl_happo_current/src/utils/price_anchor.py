"""
Fundamental price anchor for the EU ETS MARL simulation.

compute_fundamental_anchor(year, config) returns the economically grounded
expected clearing price for a given simulation year, derived from:
  - MAC cost (coal->gas fuel-switching threshold, ~48 EUR/t)
  - A banking premium multiplier on MAC (default 1.4x)
  - Cap scarcity: how far the cap has tightened relative to year 0
  - Effective penalty rate (inflation-adjusted ceiling)

Used as:
  1. The initial price_head bias in AuctionPolicy (replaces static action_anchors)
  2. The mean-reversion target in the AR(1) expected-price model (replaces
     static ar1_floor)

Formula
-------
  mac_anchored  = MAC × banking_premium_mult       # e.g. 48 × 1.4 = 67 EUR/t @ yr0
  scarcity      = 1 - cap(year) / cap(0)            # 0 at yr0, ~0.45 by yr11 @ 4.3%/yr LRF
  eff_penalty   = penalty_rate × (1 + inflation)^year
  anchor        = mac_anchored + scarcity × (eff_penalty - mac_anchored)

Calibration at default config values (MAC=48, mult=1.4, penalty=138.75,
inflation=2%, LRF~4.3%/yr, price_min=45, price_max=250):

  Year 0:  scarcity=0.00, anchor~67 EUR/t
  Year 4:  scarcity~0.16, anchor~79 EUR/t
  Year 8:  scarcity~0.30, anchor~90 EUR/t
  Year 11: scarcity~0.40, anchor~101 EUR/t

These track the EU ETS Phase 4 forward-curve shape.
"""
from __future__ import annotations


def _resolve_cap_year_0(config: dict) -> float:
    """
    Resolve cap_year_0 without requiring CapSchedule (which needs calibrated
    absolute values that are only written into config at environment init time).

    Priority:
      1. config["ets"]["cap_year_0"]        — calibrated value (set by ETSEnvironment)
      2. config["ets"]["cap_year_0_override"] — explicit YAML override
      3. Estimate from company initial_mix × output_twh × emission_factors
    """
    ets_cfg = config.get("ets", {})

    cap_calibrated = ets_cfg.get("cap_year_0")
    if cap_calibrated is not None:
        return float(cap_calibrated)

    cap_override = ets_cfg.get("cap_year_0_override")
    if cap_override is not None:
        return float(cap_override)

    # Estimate from company emissions
    companies = config.get("companies", {})
    n_agents = int(companies.get("n_agents", 8))
    output_twh = float(companies.get("output_twh", 10.0))
    default_mix = [[0.25, 0.30, 0.20, 0.15, 0.10]] * n_agents
    mixes = companies.get("initial_mix", default_mix)
    efs = config.get("technologies", {}).get(
        "emission_factors", [0.820, 0.490, 0.011, 0.012, 0.048]
    )
    total_mt = sum(
        sum(frac * output_twh * ef for frac, ef in zip(mix, efs))
        for mix in mixes
    )
    overhead = float(ets_cfg.get("cap_overhead_pct", 0.02))
    return total_mt * (1.0 + overhead)


def compute_fundamental_anchor(
    year: int,
    config: dict,
    banking_premium_mult: float | None = None,
) -> float:
    """
    Return the fundamental price anchor for the given simulation year.

    Parameters
    ----------
    year : int
        Simulation year index (0-based).
    config : dict
        Full training config dict.
    banking_premium_mult : float, optional
        Multiplier on MAC cost to account for banking premium.
        Defaults to config["price"]["banking_premium_mult"] (1.4).

    Returns
    -------
    float
        Anchor price in EUR/t, clipped to [price_min, price_max].
    """
    mult = banking_premium_mult
    if mult is None:
        mult = float(config.get("price", {}).get("banking_premium_mult", 1.4))

    penalty_base = float(config["penalty"]["rate"])
    inflation    = float(config["penalty"]["inflation_rate"])
    mac          = float(config["mac"]["coal_to_gas_cost"])

    ets_cfg = config.get("ets", {})
    lrf = float(ets_cfg.get("lrf_phase1", 0.043))

    cap_0 = _resolve_cap_year_0(config)
    # Linear LRF approximation (matches EU ETS mandate, good enough for anchor)
    cap_t = max(cap_0 * (1.0 - lrf * year), cap_0 * 0.01)
    scarcity = 1.0 - cap_t / max(cap_0, 1e-6)
    scarcity = float(max(0.0, min(1.0, scarcity)))

    eff_penalty  = penalty_base * ((1.0 + inflation) ** year)
    mac_anchored = mac * mult
    anchor = mac_anchored + scarcity * (eff_penalty - mac_anchored)

    price_min = float(config["auction"]["price_min"])
    price_max = float(config["auction"]["price_max"])
    return float(max(price_min, min(anchor, price_max)))

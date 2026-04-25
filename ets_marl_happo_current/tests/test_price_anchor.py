"""
test_price_anchor.py
====================
Tests for src/utils/price_anchor.py — compute_fundamental_anchor().

Covers:
  - Year-0 value is MAC × banking_premium_mult (scarcity = 0)
  - Anchor increases monotonically over the episode
  - Result is clipped to [price_min, price_max]
  - Config parameters (MAC, penalty rate, mult) are respected
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.price_anchor import compute_fundamental_anchor


# ---------------------------------------------------------------------------
# Shared minimal config matching default.yaml v8.0 values
# ---------------------------------------------------------------------------

def _make_config(
    mac=48.0,
    penalty_rate=138.75,
    inflation_rate=0.02,
    banking_premium_mult=1.4,
    price_min=45.0,
    price_max=250.0,
    lrf=0.043,
    cap_overhead_pct=0.02,
    n_agents=8,
):
    return {
        "mac": {"enabled": True, "coal_to_gas_cost": mac, "max_switch_frac": 0.20},
        "penalty": {"rate": penalty_rate, "inflation_rate": inflation_rate,
                    "carry_forward": True, "carry_forward_cap": 0.0},
        "price": {"banking_premium_mult": banking_premium_mult,
                  "ar1_persistence": 0.85, "volatility_std": 0.15, "burnin_std": 10.0},
        "auction": {"price_min": price_min, "price_max": price_max,
                    "quantity_max": 3.0, "qty_mult_low": 0.5, "qty_mult_high": 2.0,
                    "pricing_rule": "uniform"},
        "ets": {
            "cap_year_0_override": None,
            "initial_bank_fraction": 0.10,
            "cap_overhead_pct": cap_overhead_pct,
            "lrf_phase1": lrf,
            "lrf_phase2": lrf + 0.001,
            "lrf_phase_switch": 2,
            "msr": {
                "enabled": True,
                "tnac_upper_ratio": 0.36,
                "tnac_mid_ratio": None,
                "tnac_lower_ratio": None,
                "withhold_rate": 0.24,
                "release_frac": 0.0638297872,
                "activation_year": 1,
                "price_containment_absolute": 350,
                "price_release_absolute": 450,
                "emergency_release_frac": 0.064,
                "min_auction_frac": 0.10,
            },
            "banking": True,
            "reserve_price": 45.0,
            "reserve_price_mode": "static",
            "unsold_to_msr": False,
            "max_rollover_multiplier": 1.5,
            "price_history_anchor": "auction",
        },
        "companies": {
            "n_agents": n_agents,
            "output_twh": 10.0,
            "initial_mix": [[0.25, 0.30, 0.20, 0.15, 0.10]] * n_agents,
        },
        "technologies": {
            "names": ["coal", "gas", "onshore_wind", "offshore_wind", "solar"],
            "emission_factors": [0.820, 0.490, 0.011, 0.012, 0.048],
            "capacity_factors": [0.65, 0.60, 0.35, 0.47, 0.17],
            "capex": [3000, 1150, 1350, 3250, 750],
            "deploy_delays": [0, 0, 4, 7, 2],
            "operational_costs": [72.0, 55.0, 17.0, 47.0, 10.0],
            "decommission_costs": [200, 100, 0, 0, 0],
            "is_green": [False, False, True, True, True],
            "is_buildable": [False, False, True, True, True],
        },
        "simulation": {"n_years": 12, "n_episodes": 100},
    }


# ---------------------------------------------------------------------------
# Test 1: Year-0 value equals MAC × banking_premium_mult
# ---------------------------------------------------------------------------

def test_year_0_equals_mac_times_mult():
    """At year 0 scarcity = 0, so anchor = MAC × banking_premium_mult."""
    config = _make_config(mac=48.0, banking_premium_mult=1.4, price_min=45.0, price_max=250.0)
    anchor = compute_fundamental_anchor(0, config)
    expected = 48.0 * 1.4  # = 67.2 EUR/t
    assert abs(anchor - expected) < 1.0, (
        f"Year-0 anchor {anchor:.2f} != MAC×mult {expected:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Anchor increases monotonically with year
# ---------------------------------------------------------------------------

def test_anchor_increases_with_year():
    """As the cap tightens, scarcity rises and the anchor should increase each year."""
    config = _make_config()
    anchors = [compute_fundamental_anchor(y, config) for y in range(12)]

    # Allow for plateau near price_max but otherwise strictly non-decreasing
    for y in range(1, 12):
        assert anchors[y] >= anchors[y - 1] - 1e-6, (
            f"Anchor decreased from year {y-1} ({anchors[y-1]:.2f}) "
            f"to year {y} ({anchors[y]:.2f})"
        )

    # And should increase meaningfully over the full episode
    total_increase = anchors[-1] - anchors[0]
    assert total_increase > 10.0, (
        f"Anchor barely changed over 12 years: {anchors[0]:.1f} → {anchors[-1]:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Result is clipped to [price_min, price_max]
# ---------------------------------------------------------------------------

def test_anchor_clipped_to_price_bounds():
    """With extreme inputs the result must stay within [price_min, price_max]."""
    # Tight bounds — force clipping
    config_low = _make_config(mac=200.0, banking_premium_mult=5.0,
                               price_min=45.0, price_max=100.0)
    anchor_low = compute_fundamental_anchor(0, config_low)
    assert anchor_low <= 100.0, f"Anchor {anchor_low:.2f} exceeds price_max=100"

    config_high = _make_config(mac=1.0, banking_premium_mult=0.1,
                                price_min=60.0, price_max=250.0)
    anchor_high = compute_fundamental_anchor(0, config_high)
    assert anchor_high >= 60.0, f"Anchor {anchor_high:.2f} below price_min=60"


# ---------------------------------------------------------------------------
# Test 4: Config parameters are respected
# ---------------------------------------------------------------------------

def test_anchor_respects_config_params():
    """Different MAC and penalty rates should produce clearly different anchors."""
    config_low_mac = _make_config(mac=20.0, banking_premium_mult=1.0,
                                   penalty_rate=100.0)
    config_high_mac = _make_config(mac=80.0, banking_premium_mult=1.6,
                                    penalty_rate=200.0)

    anchor_low = compute_fundamental_anchor(0, config_low_mac)
    anchor_high = compute_fundamental_anchor(0, config_high_mac)

    assert anchor_high > anchor_low, (
        f"Higher MAC+penalty config should give higher anchor: "
        f"{anchor_high:.2f} vs {anchor_low:.2f}"
    )

    # Year 0: anchor_low = 20×1.0=20 (or clipped to price_min if below)
    # Year 0: anchor_high = 80×1.6=128
    assert anchor_high > 100.0, f"anchor_high {anchor_high:.2f} too low for MAC=80, mult=1.6"


# ---------------------------------------------------------------------------
# Test 5: banking_premium_mult override kwarg works
# ---------------------------------------------------------------------------

def test_banking_premium_mult_override():
    """Passing banking_premium_mult kwarg overrides the config value."""
    config = _make_config(mac=50.0, banking_premium_mult=1.0,
                           price_min=45.0, price_max=250.0)

    anchor_1x = compute_fundamental_anchor(0, config, banking_premium_mult=1.0)
    anchor_2x = compute_fundamental_anchor(0, config, banking_premium_mult=2.0)

    assert abs(anchor_1x - 50.0) < 1.0, f"1× mult: expected ~50, got {anchor_1x:.2f}"
    assert abs(anchor_2x - 100.0) < 1.0, f"2× mult: expected ~100, got {anchor_2x:.2f}"

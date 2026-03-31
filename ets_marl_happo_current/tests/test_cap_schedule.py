"""
tests/test_cap_schedule.py
==========================
Unit tests for LRF + MSR cap schedule logic.

Run with:
    pytest tests/test_cap_schedule.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.environment.cap_schedule import CapSchedule

BASE_CONFIG = {
    "ets": {
        "cap_year_0": 14.60,
        "lrf_phase1": 0.043,
        "lrf_phase2": 0.044,
        "lrf_phase_switch": 5,
        "reserve_price": 0.0,
        "msr": {
            "enabled": True,
            "tnac_upper": 8.18,   # 28.0 × 14.6/50.0 (scaled to match test cap)
            "tnac_lower": 4.09,   # 14.0 × 14.6/50.0 (scaled to match test cap)
            "withhold_rate": 0.24,
            "release_amount": 0.80,
            "min_auction_frac": 0.10,
            "activation_year": 0,
        },
    }
}


def make_schedule(msr_enabled=True, activation_year=0):
    cfg = BASE_CONFIG.copy()
    cfg["ets"] = dict(cfg["ets"])
    cfg["ets"]["msr"] = dict(cfg["ets"]["msr"])
    cfg["ets"]["msr"]["enabled"] = msr_enabled
    cfg["ets"]["msr"]["activation_year"] = activation_year
    return CapSchedule(cfg)


# ---------------------------------------------------------------------------
# LRF tests
# ---------------------------------------------------------------------------

def test_cap_year_0():
    s = make_schedule()
    assert s.get_cap(0) == pytest.approx(14.60)


def test_cap_year_1_lrf_phase1():
    s = make_schedule()
    expected = 14.60 * (1 - 0.043)
    assert s.get_cap(1) == pytest.approx(expected, rel=1e-6)


def test_cap_year_5_switches_to_phase2():
    """Year 5 should use LRF 4.4%, not 4.3%."""
    s = make_schedule()
    cap4 = s.get_cap(4)
    cap5 = s.get_cap(5)
    expected = cap4 * (1 - 0.044)
    assert cap5 == pytest.approx(expected, rel=1e-6)


def test_cap_strictly_decreasing():
    s = make_schedule()
    caps = [s.get_cap(t) for t in range(11)]
    for i in range(1, len(caps)):
        assert caps[i] < caps[i - 1], f"Cap did not decrease at year {i}"


def test_cap_year_10():
    """After 10 years cap should be roughly 35% lower than year 0."""
    s = make_schedule()
    cap10 = s.get_cap(10)
    reduction = 1 - cap10 / 14.60
    assert 0.30 < reduction < 0.45, f"Unexpected reduction: {reduction:.2%}"


# ---------------------------------------------------------------------------
# MSR tests
# ---------------------------------------------------------------------------

def test_msr_no_adjustment_within_band():
    """TNAC between thresholds → auction volume = cap."""
    s = make_schedule(msr_enabled=True)
    tnac = 6.0   # between 4.09 and 8.18
    vol = s.get_auction_volume(year=1, tnac=tnac)
    cap = s.get_cap(1)
    assert vol == pytest.approx(cap, rel=1e-6)


def test_msr_withhold_when_tnac_high():
    """TNAC > upper threshold → volume reduced, reserve grows."""
    s = make_schedule(msr_enabled=True)
    tnac = 10.0   # > 8.18
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    expected_withheld = min(0.24 * tnac, cap)
    expected_vol = max(cap - expected_withheld, 0.10 * cap)
    assert vol == pytest.approx(expected_vol, rel=1e-5)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-5)


def test_msr_inactive_before_activation_year():
    """MSR should not withhold/release/cancel before configured activation year."""
    s = make_schedule(msr_enabled=True, activation_year=2)
    tnac_high = 10.0

    cap0 = s.get_cap(0)
    vol0 = s.get_auction_volume(year=0, tnac=tnac_high)
    assert vol0 == pytest.approx(cap0, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(0.0, rel=1e-6)

    cap1 = s.get_cap(1)
    vol1 = s.get_auction_volume(year=1, tnac=tnac_high)
    assert vol1 == pytest.approx(cap1, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(0.0, rel=1e-6)

    cap2 = s.get_cap(2)
    vol2 = s.get_auction_volume(year=2, tnac=tnac_high)
    expected_withheld = min(0.24 * tnac_high, cap2)
    expected_vol = max(cap2 - expected_withheld, 0.10 * cap2)
    assert vol2 == pytest.approx(expected_vol, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-6)


def test_msr_activation_year_zero_means_immediate():
    """activation_year=0 should preserve legacy immediate MSR behavior."""
    s = make_schedule(msr_enabled=True, activation_year=0)
    tnac_high = 10.0
    cap = s.get_cap(0)
    vol = s.get_auction_volume(year=0, tnac=tnac_high)
    expected_withheld = min(0.24 * tnac_high, cap)
    expected_vol = max(cap - expected_withheld, 0.10 * cap)
    assert vol == pytest.approx(expected_vol, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-6)


def test_auction_volume_has_min_floor():
    """Auction volume is never below min_auction_frac * cap even with extreme TNAC."""
    s = make_schedule(msr_enabled=True)
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=1_000.0)
    assert vol >= 0.10 * cap - 1e-9


def test_msr_release_when_tnac_low():
    """TNAC < lower threshold → release from reserve into auction."""
    s = make_schedule(msr_enabled=True)
    # First build up some reserve
    s._msr_reserve = 1.00

    tnac = 3.0   # < 4.09
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    assert vol == pytest.approx(cap + 0.80, rel=1e-5)   # release_amount = 0.80
    assert s.msr_reserve() == pytest.approx(0.20, rel=1e-5)


def test_msr_disabled():
    """With MSR disabled, auction volume always equals cap."""
    s = make_schedule(msr_enabled=False)
    for tnac in [0.0, 5.0, 15.0]:
        s.reset()
        vol = s.get_auction_volume(year=1, tnac=tnac)
        cap = s.get_cap(1)
        assert vol == pytest.approx(cap, rel=1e-6)


def test_msr_volume_never_negative():
    """Auction volume cannot go negative even with extreme TNAC."""
    s = make_schedule(msr_enabled=True)
    vol = s.get_auction_volume(year=1, tnac=1000.0)
    assert vol >= 0.0


def test_msr_reset():
    """Reset clears the reserve and history."""
    s = make_schedule(msr_enabled=True)
    s.get_auction_volume(year=1, tnac=15.0)
    assert s.msr_reserve() > 0
    s.reset()
    assert s.msr_reserve() == 0.0
    assert s.cap_history == []


def test_msr_cancellation():
    """MSR cancellation: holdings above previous auction volume are cancelled."""
    s = make_schedule(msr_enabled=True)
    # Manually set reserve and volume history
    s._msr_reserve = 20.0
    s.volume_history = [10.0]

    # Call get_auction_volume to trigger cancellation
    vol = s.get_auction_volume(year=1, tnac=8.0)

    # Reserve should be reduced to previous auction volume
    assert s._msr_reserve == pytest.approx(10.0, rel=1e-5)
    # Total cancelled should equal the excess
    assert s._total_cancelled == pytest.approx(10.0, rel=1e-5)


def test_msr_no_cancellation_when_below():
    """No cancellation when MSR reserve is below previous auction volume."""
    s = make_schedule(msr_enabled=True)
    # Set reserve below previous auction volume
    s._msr_reserve = 5.0
    s.volume_history = [10.0]

    # Call get_auction_volume
    vol = s.get_auction_volume(year=1, tnac=8.0)

    # Reserve should be unchanged by cancellation (may change due to MSR logic)
    # But no cancellation should have occurred
    assert s._total_cancelled == pytest.approx(0.0, rel=1e-5)


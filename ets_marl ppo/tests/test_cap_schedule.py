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
            "tnac_upper": 8.33,
            "tnac_lower": 4.00,
            "withhold_rate": 0.24,
            "release_amount": 0.10,
        },
    }
}


def make_schedule(msr_enabled=True):
    cfg = BASE_CONFIG.copy()
    cfg["ets"] = dict(cfg["ets"])
    cfg["ets"]["msr"] = dict(cfg["ets"]["msr"])
    cfg["ets"]["msr"]["enabled"] = msr_enabled
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
    tnac = 6.0   # between 4.00 and 8.33
    vol = s.get_auction_volume(year=1, tnac=tnac)
    cap = s.get_cap(1)
    assert vol == pytest.approx(cap, rel=1e-6)


def test_msr_withhold_when_tnac_high():
    """TNAC > upper threshold → volume reduced, reserve grows."""
    s = make_schedule(msr_enabled=True)
    tnac = 10.0   # > 8.33
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    excess = tnac - 8.33
    expected_withheld = 0.24 * excess
    assert vol == pytest.approx(cap - expected_withheld, rel=1e-5)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-5)


def test_msr_release_when_tnac_low():
    """TNAC < lower threshold → release from reserve into auction."""
    s = make_schedule(msr_enabled=True)
    # First build up some reserve
    s._msr_reserve = 0.50

    tnac = 2.0   # < 4.00
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    assert vol == pytest.approx(cap + 0.10, rel=1e-5)   # release_amount = 0.10
    assert s.msr_reserve() == pytest.approx(0.40, rel=1e-5)


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

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
    "auction": {"price_max": 500.0},
    "ets": {
        "cap_year_0": 14.60,
        "lrf_phase1": 0.043,
        "lrf_phase2": 0.044,
        "lrf_phase_switch": 5,
        "reserve_price": 0.0,
        "msr": {
            "enabled": True,
            "tnac_upper": 5.26,   # 18.0 × 14.6/50.0 (scaled to match test cap)
            "tnac_mid": 4.00,     # 833/1096 of upper threshold
            "tnac_lower": 1.92,   # 400/1096 of upper threshold
            "withhold_rate": 0.24,
            "release_amount": 0.80,
            "min_auction_frac": 0.10,
            "activation_year": 0,
        },
    }
}


def make_schedule(msr_enabled=True, activation_year=0, seed_prev_tnac=None):
    cfg = BASE_CONFIG.copy()
    cfg["ets"] = dict(cfg["ets"])
    cfg["ets"]["msr"] = dict(cfg["ets"]["msr"])
    cfg["ets"]["msr"]["enabled"] = msr_enabled
    cfg["ets"]["msr"]["activation_year"] = activation_year
    s = CapSchedule(cfg)
    # Seed _prev_tnac for tests that want MSR active from the first call.
    # This simulates a prior year having completed (1-year TNAC lag).
    if seed_prev_tnac is not None:
        s._prev_tnac = float(seed_prev_tnac)
    return s


# ---------------------------------------------------------------------------
# LRF tests
# ---------------------------------------------------------------------------

def test_cap_year_0():
    s = make_schedule()
    assert s.get_cap(0) == pytest.approx(14.60)


def test_cap_negative_year():
    """Negative years should return caps higher than year 0 (linear extrapolation)."""
    s = make_schedule()
    cap_neg1 = s.get_cap(-1)
    cap_0 = s.get_cap(0)
    assert cap_neg1 > cap_0, "Cap at year -1 should exceed cap at year 0"
    # Linear backward: cap_0 + 1 * lrf_phase1 * cap_0
    expected = 14.60 * (1.0 + 0.043)
    assert cap_neg1 == pytest.approx(expected, rel=1e-4)


def test_cap_year_1_lrf_phase1():
    s = make_schedule()
    # Linear: cap_0 - 1 * lrf_phase1 * cap_0 = cap_0 * (1 - lrf_phase1)
    expected = 14.60 - 0.043 * 14.60
    assert s.get_cap(1) == pytest.approx(expected, rel=1e-6)


def test_cap_linear_equal_steps():
    """LRF is a LINEAR decline: each year removes the same absolute amount."""
    s = make_schedule()
    caps = [s.get_cap(t) for t in range(6)]
    # Phase 1 steps (t=1,2,...,switch-1=4) should all equal lrf_phase1 * cap_0
    step_size_phase1 = 0.043 * 14.60
    for t in range(1, min(s.lrf_switch, 6)):
        assert caps[t - 1] - caps[t] == pytest.approx(step_size_phase1, rel=1e-6), \
            f"Year {t}: step = {caps[t-1]-caps[t]:.6f}, expected {step_size_phase1:.6f}"


def test_cap_year_5_switches_to_phase2():
    """Year 5 should use LRF 4.4% (linear): step = 0.044 * cap_year_0."""
    s = make_schedule()
    cap4 = s.get_cap(4)
    cap5 = s.get_cap(5)
    # Linear: step = lrf_phase2 * cap_year_0 (not cap4 * lrf)
    step = cap4 - cap5
    expected_step = 0.044 * 14.60
    assert step == pytest.approx(expected_step, rel=1e-6)


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
    """TNAC between lower and mid thresholds → auction volume = cap."""
    tnac = 3.0   # between 1.92 and 4.00
    # Seed prev_tnac so MSR is active at year 1 (1-year lag)
    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    vol = s.get_auction_volume(year=1, tnac=tnac)
    cap = s.get_cap(1)
    assert vol == pytest.approx(cap, rel=1e-6)


def test_msr_withhold_in_middle_band():
    """Mid-threshold <= TNAC <= upper threshold → withhold TNAC - mid threshold."""
    tnac = 4.8   # between 4.00 and 5.26
    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    expected_withheld = min(tnac - s.tnac_mid, cap)
    expected_vol = max(cap - expected_withheld, 0.10 * cap)
    assert vol == pytest.approx(expected_vol, rel=1e-5)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-5)


def test_msr_withhold_when_tnac_high():
    """TNAC > upper threshold → volume reduced, reserve grows."""
    tnac = 10.0   # > 5.26
    # Seed prev_tnac so MSR sees this high TNAC from previous year
    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    expected_withheld = min(0.24 * tnac, cap)
    expected_vol = max(cap - expected_withheld, 0.10 * cap)
    assert vol == pytest.approx(expected_vol, rel=1e-5)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-5)


def test_msr_withheld_rate_above_upper_threshold():
    """Above upper threshold, withheld amount follows 24% of TNAC."""
    tnac = 8.5
    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    cap = s.get_cap(1)
    s.get_auction_volume(year=1, tnac=tnac)
    assert s._last_msr_withheld == pytest.approx(min(0.24 * tnac, cap), rel=1e-9)


def test_msr_withholding_reduces_final_supply_even_with_rollovers():
    """High-TNAC withholding must reduce final supply below raw cap + rollovers."""
    tnac = 10.0
    unsold_rollover = 0.60
    defaulted_rollover = 0.40

    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    raw_cap = s.get_cap(1)
    s._unsold_rollover_pending = unsold_rollover

    supply_before_defaults = s.get_auction_volume(year=1, tnac=tnac)
    final_supply = supply_before_defaults + defaulted_rollover

    assert s._last_msr_withheld > 0.0
    assert final_supply < raw_cap + unsold_rollover + defaulted_rollover


def test_msr_inactive_year_0_no_prior_tnac():
    """MSR should not activate at year 0 when no prior TNAC exists (1-year lag)."""
    s = make_schedule(msr_enabled=True)
    tnac_high = 10.0

    # Year 0: _prev_tnac is None → MSR is inactive regardless of current TNAC
    cap0 = s.get_cap(0)
    vol0 = s.get_auction_volume(year=0, tnac=tnac_high)
    assert vol0 == pytest.approx(cap0, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(0.0, rel=1e-6)
    # After year 0, _prev_tnac should be set for year 1
    assert s._prev_tnac == pytest.approx(tnac_high, rel=1e-9)


def test_msr_activates_year_1_after_prior_tnac():
    """MSR activates at year 1+ once _prev_tnac has been populated by year 0."""
    s = make_schedule(msr_enabled=True)
    tnac_high = 10.0

    # Year 0: populates _prev_tnac (no MSR action yet)
    s.get_auction_volume(year=0, tnac=tnac_high)

    # Year 1: MSR should fire using prev_tnac = 10.0 (high TNAC → withhold)
    cap1 = s.get_cap(1)
    vol1 = s.get_auction_volume(year=1, tnac=tnac_high)
    expected_withheld = min(0.24 * tnac_high, cap1)
    expected_vol = max(cap1 - expected_withheld, 0.10 * cap1)
    assert vol1 == pytest.approx(expected_vol, rel=1e-5)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-5)


def test_msr_activation_year_force_msr():
    """force_msr=True should activate MSR even without a prior TNAC (burn-in use)."""
    s = make_schedule(msr_enabled=True)
    tnac_high = 10.0
    cap = s.get_cap(0)

    # force_msr bypasses the _prev_tnac=None check (used during burn-in years)
    vol = s.get_auction_volume(year=0, tnac=tnac_high, force_msr=True)
    expected_withheld = min(0.24 * tnac_high, cap)
    expected_vol = max(cap - expected_withheld, 0.10 * cap)
    assert vol == pytest.approx(expected_vol, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-6)


def test_auction_volume_has_min_floor():
    """Auction volume is never below min_auction_frac * cap even with extreme TNAC."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=1_000.0)
    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=1_000.0)
    assert vol >= 0.10 * cap - 1e-9


def test_msr_release_when_tnac_low():
    """TNAC < lower threshold → release from reserve into auction."""
    tnac = 1.5   # < 1.92
    s = make_schedule(msr_enabled=True, seed_prev_tnac=tnac)
    # Build up some reserve
    s._msr_reserve = 1.00

    cap = s.get_cap(1)
    vol = s.get_auction_volume(year=1, tnac=tnac)

    assert vol == pytest.approx(cap + 0.80, rel=1e-5)   # release_amount = 0.80
    assert s.msr_reserve() == pytest.approx(0.20, rel=1e-5)


def test_msr_disabled():
    """With MSR disabled, auction volume always equals cap."""
    s = make_schedule(msr_enabled=False)
    for tnac in [0.0, 5.0, 15.0]:
        s.reset()
        s._prev_tnac = tnac  # seed prev so MSR gate is irrelevant
        vol = s.get_auction_volume(year=1, tnac=tnac)
        cap = s.get_cap(1)
        assert vol == pytest.approx(cap, rel=1e-6)


def test_msr_volume_never_negative():
    """Auction volume cannot go negative even with extreme TNAC."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=1_000.0)
    vol = s.get_auction_volume(year=1, tnac=1000.0)
    assert vol >= 0.0


def test_msr_reset():
    """Reset clears the reserve, history, and TNAC lag state."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=15.0)
    s.get_auction_volume(year=1, tnac=15.0)
    assert s.msr_reserve() > 0
    s.reset()
    assert s.msr_reserve() == 0.0
    assert s.cap_history == []
    assert s._prev_tnac is None
    assert s._prev_ma3 is None


def test_msr_cancellation():
    """MSR cancellation: holdings above tnac_lower are cancelled."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=4.0)
    # tnac_lower = 1.92 (from BASE_CONFIG)
    s._msr_reserve = 20.0

    vol = s.get_auction_volume(year=1, tnac=4.0)

    # Reserve should be clamped down to tnac_lower
    assert s._msr_reserve == pytest.approx(s.tnac_lower, rel=1e-5)
    # Total cancelled = excess above tnac_lower
    assert s._total_cancelled == pytest.approx(20.0 - s.tnac_lower, rel=1e-5)


def test_msr_no_cancellation_when_below():
    """No cancellation when MSR reserve is at or below tnac_lower."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=4.0)
    # Set reserve strictly below tnac_lower (1.92)
    s._msr_reserve = 1.5

    vol = s.get_auction_volume(year=1, tnac=4.0)

    # No cancellation should have occurred
    assert s._total_cancelled == pytest.approx(0.0, rel=1e-5)


def test_force_msr_bypasses_prev_tnac_gate():
    """force_msr=True should apply MSR logic even when _prev_tnac is None."""
    s = make_schedule(msr_enabled=True)
    tnac_high = 10.0
    cap0 = s.get_cap(0)

    # No seed_prev_tnac, but force_msr bypasses the gate
    vol = s.get_auction_volume(year=0, tnac=tnac_high, force_msr=True)
    expected_withheld = min(0.24 * tnac_high, cap0)
    expected_vol = max(cap0 - expected_withheld, 0.10 * cap0)

    assert vol == pytest.approx(expected_vol, rel=1e-6)
    assert s.msr_reserve() == pytest.approx(expected_withheld, rel=1e-6)


def test_preview_matches_live_msr_logic():
    """preview_auction_volume should match live get_auction_volume for the same lagged state."""
    s_preview = make_schedule(msr_enabled=True)
    s_live = make_schedule(msr_enabled=True)

    for s in (s_preview, s_live):
        s._prev_tnac = 10.0
        s._prev_ma3 = 70.0
        s._msr_reserve = 12.0
        s.volume_history = [9.0]
        s._unsold_rollover_pending = 0.0

    year = 1
    clearing_price = 420.0
    price_max = 500.0
    penalty_rate = 138.75
    inflation_rate = 0.02
    price_ma3 = 220.0

    preview_vol = s_preview.preview_auction_volume(
        year=year,
        clearing_price=clearing_price,
        price_max=price_max,
        penalty_rate=penalty_rate,
        inflation_rate=inflation_rate,
        price_ma3=price_ma3,
    )
    live_vol = s_live.get_auction_volume(
        year=year,
        tnac=8.0,  # ignored for lagged decision branch; _prev_tnac is used
        clearing_price=clearing_price,
        price_max=price_max,
        penalty_rate=penalty_rate,
        inflation_rate=inflation_rate,
        price_ma3=price_ma3,
    )

    assert preview_vol == pytest.approx(live_vol, rel=1e-9), (
        f"Preview/live mismatch: preview={preview_vol}, live={live_vol}"
    )


def test_msr_smoothed_price_trigger_a4():
    """A4: Emergency release fires when both absolute threshold and MA3 spike are met."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=4.0)
    # Build reserve for emergency release
    s._msr_reserve = 5.0
    # Set previous MA3 price (prev_ma3 = 100 EUR/t)
    s._prev_ma3 = 100.0
    # Base rate 138.75 → release_threshold = min(138.75 * 2.5, 450) = 346.875
    # clearing_price 400 > 346.875 → absolute threshold met
    # price_ma3 = 300 > 2.5 * 100 = 250 → smoothed spike met
    initial_reserve = s._msr_reserve
    vol = s.get_auction_volume(
        year=2, tnac=4.0, clearing_price=400.0,
        penalty_rate=138.75, inflation_rate=0.0,
        price_ma3=300.0,
    )
    assert s.msr_reserve() < initial_reserve, "Emergency release should have fired"
    assert s._msr_event_counts["emergency_release"] == 1


def test_msr_smoothed_price_trigger_no_spike():
    """A4: Emergency release does NOT fire if absolute threshold met but no MA3 spike."""
    s = make_schedule(msr_enabled=True, seed_prev_tnac=4.0)
    s._msr_reserve = 5.0
    s._prev_ma3 = 100.0
    # clearing_price 400 > release_threshold (absolute met)
    # price_ma3 = 110 < 2.5 * 100 = 250 → NO smoothed spike → should NOT fire
    initial_reserve = s._msr_reserve
    s.get_auction_volume(
        year=2, tnac=4.0, clearing_price=400.0,
        penalty_rate=138.75, inflation_rate=0.0,
        price_ma3=110.0,
    )
    # Emergency release should NOT have fired (but may have triggered containment)
    assert s._msr_event_counts["emergency_release"] == 0, \
        "Emergency release should not fire without smoothed spike"


def test_msr_tnac_lag_prev_tnac_update():
    """get_auction_volume updates _prev_tnac to the current tnac after each call."""
    s = make_schedule(msr_enabled=True)
    assert s._prev_tnac is None
    s.get_auction_volume(year=0, tnac=7.5)
    assert s._prev_tnac == pytest.approx(7.5, rel=1e-9)
    s.get_auction_volume(year=1, tnac=4.2)
    assert s._prev_tnac == pytest.approx(4.2, rel=1e-9)

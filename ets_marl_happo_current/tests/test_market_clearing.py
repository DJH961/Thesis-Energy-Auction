"""
tests/test_market_clearing.py
==============================
Unit tests for the EU ETS uniform-price buyer auction clearing.

Run with:
    pytest tests/test_market_clearing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings

import numpy as np
import pytest
from src.auction.market_clearing_ets import market_clearing_ets, build_bids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bids(agent_prices_qtys: list) -> np.ndarray:
    """[(price, qty), ...] -> bids array"""
    bids = []
    for i, (p, q) in enumerate(agent_prices_qtys):
        bids.append([i, q, p])
    return np.array(bids, dtype=float)


# ---------------------------------------------------------------------------
# Test 1: Correct clearing price
# ---------------------------------------------------------------------------

def test_clearing_price_basic():
    """
    3 agents, Q_cap = 2.0 Mt.
    Bids (price, qty): A0=(80, 1.0), A1=(70, 1.0), A2=(60, 1.0)
    Total demand = 3.0 > q_cap = 2.0 (over-subscribed).
    Sorted descending: A0=80 -> A1=70 -> A2=60
    Fill: 1.0 (A0) + 1.0 (A1) = 2.0 -> stop.
    Clearing price = 70 (last accepted bid).
    """
    bids = make_bids([(80, 1.0), (70, 1.0), (60, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0)

    assert price == pytest.approx(70.0), f"Expected 70, got {price}"
    assert alloc[0] == pytest.approx(1.0)
    assert alloc[1] == pytest.approx(1.0)
    assert alloc[2] == pytest.approx(0.0)
    assert stats["auction_failed"] is False


# ---------------------------------------------------------------------------
# Test 2: Quantity conservation
# ---------------------------------------------------------------------------

def test_quantity_conservation():
    """Total allocated must equal q_cap when over-subscribed."""
    bids = make_bids([(75, 2.0), (65, 3.0), (55, 2.0)])
    q_cap = 4.0
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=q_cap)

    assert alloc.sum() == pytest.approx(q_cap, abs=1e-6)
    assert stats["total_allocated"] == pytest.approx(q_cap, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Uniform price payment
# ---------------------------------------------------------------------------

def test_uniform_price_payment():
    """All winners pay the same clearing price per unit."""
    bids = make_bids([(90, 1.0), (80, 1.0), (70, 1.0)])
    price, alloc, pay, _ = market_clearing_ets(bids, q_cap=2.0)

    for i in range(3):
        if alloc[i] > 0:
            assert pay[i] == pytest.approx(alloc[i] * price, abs=1e-6), \
                f"Agent {i}: expected pay={alloc[i]*price}, got {pay[i]}"


# ---------------------------------------------------------------------------
# Test 4: Reserve price filters low bids
# ---------------------------------------------------------------------------

def test_reserve_price():
    """Bids below reserve price should be ignored.
    A0=80 (valid), A1=30 (invalid), A2=20 (invalid).
    Only 1 valid bid for 1.0 Mt vs q_cap=3.0.
    Default cancel_under_subscribed=False: sells 1.0 Mt to A0."""
    bids = make_bids([(80, 1.0), (30, 1.0), (20, 1.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=3.0, reserve_price=50.0)

    # Under-subscribed but default=False: auction proceeds, sells what was demanded
    assert stats["auction_failed"] is False
    assert alloc[0] == pytest.approx(1.0)
    assert alloc[1] == pytest.approx(0.0)
    assert alloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: All bids below reserve -> auction fails
# ---------------------------------------------------------------------------

def test_all_below_reserve():
    bids = make_bids([(10, 1.0), (20, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0, reserve_price=50.0)

    assert stats["auction_failed"] is True
    assert stats["fail_reason"] == "all_below_reserve"
    assert alloc.sum() == pytest.approx(0.0)
    assert pay.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 6: Single bidder (over-subscribed)
# ---------------------------------------------------------------------------

def test_single_bidder():
    """Single agent bids 3.0 Mt at 65, q_cap = 2.0. Over-subscribed."""
    bids = make_bids([(65, 3.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0)

    assert price == pytest.approx(65.0)
    assert alloc[0] == pytest.approx(2.0)   # capped at Q_cap
    assert pay[0] == pytest.approx(65.0 * 2.0)


# ---------------------------------------------------------------------------
# Test 7: Large bid (q >> Q_cap)
# ---------------------------------------------------------------------------

def test_large_bid():
    """A0 bids 100 Mt at 70, A1 bids 1 Mt at 60. Total=101 > q_cap=5."""
    bids = make_bids([(70, 100.0), (60, 1.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=5.0)

    assert alloc.sum() == pytest.approx(5.0, abs=1e-6)
    assert alloc[0] == pytest.approx(5.0)
    assert alloc[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 8: All bids at same price — random tie-break
# ---------------------------------------------------------------------------

def test_tie_break_random():
    """
    3 agents all bid at same price 70, each wants 2 Mt.
    Total demand = 6.0 > Q_cap = 3.0 (over-subscribed).
    Random tie-break: allocation is random but total = q_cap.
    With random ordering, one agent gets partial fill.
    """
    rng = np.random.default_rng(42)
    bids = make_bids([(70, 2.0), (70, 2.0), (70, 2.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=3.0, rng=rng)

    assert alloc.sum() == pytest.approx(3.0, abs=1e-6)
    # With random tie-break, at least one agent gets partial fill
    # and total is conserved. The exact allocation depends on rng.
    for a in alloc:
        assert a >= 0.0
        assert a <= 2.0 + 1e-6  # no agent exceeds their bid quantity


def test_tie_break_reproducible():
    """Same rng seed produces same allocation."""
    bids = make_bids([(70, 2.0), (70, 2.0), (70, 2.0)])
    rng1 = np.random.default_rng(99)
    _, alloc1, _, _ = market_clearing_ets(bids.copy(), q_cap=3.0, rng=rng1)
    rng2 = np.random.default_rng(99)
    _, alloc2, _, _ = market_clearing_ets(bids.copy(), q_cap=3.0, rng=rng2)

    np.testing.assert_array_almost_equal(alloc1, alloc2)


def test_tie_break_partial_fill():
    """
    Only the last successful bidder in random order gets partial fill.
    2 agents bid 2.0 Mt each at same price, q_cap=3.0 total demand = 4 > 3.
    First in random order gets 2.0 (full), second gets 1.0 (partial).
    """
    bids = make_bids([(70, 2.0), (70, 2.0)])
    rng = np.random.default_rng(42)
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=3.0, rng=rng)

    assert alloc.sum() == pytest.approx(3.0, abs=1e-6)
    # One agent gets full 2.0, the other gets 1.0
    sorted_alloc = sorted(alloc, reverse=True)
    assert sorted_alloc[0] == pytest.approx(2.0, abs=1e-6)
    assert sorted_alloc[1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 9: Non-negativity
# ---------------------------------------------------------------------------

def test_non_negativity():
    bids = make_bids([(80, 1.0), (60, 2.0), (40, 1.5)])
    # Total demand = 4.5 > q_cap = 2.0
    _, alloc, pay, _ = market_clearing_ets(bids, q_cap=2.0)

    assert (alloc >= 0).all()
    assert (pay >= 0).all()


# ---------------------------------------------------------------------------
# Test 10: build_bids helper
# ---------------------------------------------------------------------------

def test_build_bids():
    """build_bids should correctly reorder columns."""
    actions = np.array([[65.0, 1.5], [70.0, 2.0]])
    bids = build_bids(actions)
    assert bids.shape == (2, 3)
    assert bids[0, 0] == 0   # agent_id
    assert bids[0, 1] == pytest.approx(1.5)  # quantity
    assert bids[0, 2] == pytest.approx(65.0) # price


# ---------------------------------------------------------------------------
# Test 11: California-style holding limit (max_agent_share)
# ---------------------------------------------------------------------------

def test_holding_limit():
    """
    Agent A0 bids 200 for 10 Mt, A1 bids 100 for 5 Mt.
    Total demand = 15 > q_cap = 10 (over-subscribed).
    With max_agent_share=0.25, each agent is capped at 2.5 Mt.
    A0 gets 2.5 (capped), A1 gets 2.5 (capped), total = 5 Mt (some unsold).
    """
    bids = make_bids([(200, 10.0), (100, 5.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=10.0, max_agent_share=0.25)

    assert alloc[0] == pytest.approx(2.5, abs=1e-6), f"A0 should be capped at 2.5, got {alloc[0]}"
    assert alloc[1] == pytest.approx(2.5, abs=1e-6), f"A1 should be capped at 2.5, got {alloc[1]}"
    assert alloc.sum() == pytest.approx(5.0, abs=1e-6)
    assert stats["unsold"] == pytest.approx(5.0, abs=1e-6)


def test_holding_limit_no_effect_when_1():
    """With max_agent_share=1.0 (default), no cap is applied."""
    bids = make_bids([(200, 10.0), (100, 5.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=10.0, max_agent_share=1.0)

    assert alloc[0] == pytest.approx(10.0, abs=1e-6), f"A0 should get full 10, got {alloc[0]}"
    assert alloc[1] == pytest.approx(0.0, abs=1e-6)


def test_holding_limit_redistributes():
    """
    3 agents all bid at 80 for 5 Mt each. Total demand = 15 > Q_cap = 10.
    With max_agent_share=0.25 (cap=2.5): each gets 2.5, total = 7.5.
    """
    bids = make_bids([(80, 5.0), (80, 5.0), (80, 5.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=10.0, max_agent_share=0.25)

    for i in range(3):
        assert alloc[i] <= 2.5 + 1e-6, f"Agent {i} exceeds cap: {alloc[i]}"


# ---------------------------------------------------------------------------
# Test 12: Under-subscription -> auction cancelled
# ---------------------------------------------------------------------------

def test_under_subscription_cancels_when_enabled():
    """
    With cancel_under_subscribed=True:
    Total demand (3.0) < q_cap (5.0) -> auction cancelled.
    No allowances sold.
    """
    bids = make_bids([(80, 1.0), (70, 1.0), (60, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(
        bids, q_cap=5.0, cancel_under_subscribed=True)

    assert stats["auction_failed"] is True
    assert stats["fail_reason"] == "under_subscribed"
    assert alloc.sum() == pytest.approx(0.0)
    assert pay.sum() == pytest.approx(0.0)
    assert stats["unsold"] == pytest.approx(5.0)


def test_under_subscription_proceeds_by_default():
    """
    Default cancel_under_subscribed=False:
    Total demand (3.0) < q_cap (5.0) -> auction still proceeds,
    sells 3.0 Mt to all bidders.
    """
    bids = make_bids([(80, 1.0), (70, 1.0), (60, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=5.0)

    assert stats["auction_failed"] is False
    assert alloc.sum() == pytest.approx(3.0, abs=1e-6)
    assert stats["unsold"] == pytest.approx(2.0, abs=1e-6)


def test_exact_subscription_succeeds():
    """Total demand == q_cap: auction should succeed (either flag value)."""
    bids = make_bids([(80, 1.0), (70, 1.0), (60, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(
        bids, q_cap=3.0, cancel_under_subscribed=True)

    assert stats["auction_failed"] is False
    assert alloc.sum() == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 13: Empty bids safeguard
# ---------------------------------------------------------------------------

def test_empty_bids():
    """Empty bids array should not crash; returns failed auction."""
    bids = np.zeros((0, 3), dtype=float)
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0)

    assert stats["auction_failed"] is True
    assert stats["fail_reason"] == "no_bids"
    assert len(alloc) == 0
    assert len(pay) == 0


# ---------------------------------------------------------------------------
# Test 14: Unsold volume in stats
# ---------------------------------------------------------------------------

def test_unsold_in_stats():
    """Stats should report unsold volume when holding limits cause under-allocation."""
    bids = make_bids([(200, 10.0), (100, 5.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=10.0, max_agent_share=0.25)

    assert "unsold" in stats
    expected_unsold = 10.0 - alloc.sum()
    assert stats["unsold"] == pytest.approx(expected_unsold, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 15: User's worked example (Firm A, B, C)
# ---------------------------------------------------------------------------

def test_user_worked_example():
    """
    Firm A: 400 EUAs @ EUR 72, Firm B: 500 EUAs @ EUR 70, Firm C: 300 EUAs @ EUR 68.
    Supply = 1000 EUAs.  Total demand = 1200 > 1000 (over-subscribed).
    Sorted: A(72) -> B(70) -> C(68).
    Fill: 400 + 500 = 900, then 100 of C's 300.
    Clearing price = EUR 68.
    Allocations: A=400, B=500, C=100.
    All pay EUR 68 per unit.
    """
    bids = make_bids([(72.0, 400.0), (70.0, 500.0), (68.0, 300.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=1000.0)

    assert price == pytest.approx(68.0)
    assert alloc[0] == pytest.approx(400.0)   # Firm A
    assert alloc[1] == pytest.approx(500.0)   # Firm B
    assert alloc[2] == pytest.approx(100.0)   # Firm C (partial)
    assert alloc.sum() == pytest.approx(1000.0, abs=1e-6)

    # Uniform price payments
    assert pay[0] == pytest.approx(400.0 * 68.0)
    assert pay[1] == pytest.approx(500.0 * 68.0)
    assert pay[2] == pytest.approx(100.0 * 68.0)
    assert stats["auction_failed"] is False


# ---------------------------------------------------------------------------
# Test 16: fail_reason field present in stats
# ---------------------------------------------------------------------------

def test_stats_fail_reason():
    """Successful auction should NOT have a fail_reason key."""
    bids = make_bids([(80, 2.0), (70, 2.0)])
    _, _, _, stats = market_clearing_ets(bids, q_cap=3.0)
    assert stats["auction_failed"] is False
    assert "fail_reason" not in stats

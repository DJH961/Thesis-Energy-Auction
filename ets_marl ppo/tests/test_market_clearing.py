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
    """[(price, qty), ...] → bids array"""
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
    Sorted descending: A0=80 → A1=70 → A2=60
    Fill: 1.0 (A0) + 1.0 (A1) = 2.0 → stop.
    Clearing price = 70 (last accepted bid).
    """
    bids = make_bids([(80, 1.0), (70, 1.0), (60, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0)

    assert price == pytest.approx(70.0), f"Expected 70, got {price}"
    assert alloc[0] == pytest.approx(1.0)
    assert alloc[1] == pytest.approx(1.0)
    assert alloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2: Quantity conservation
# ---------------------------------------------------------------------------

def test_quantity_conservation():
    """Total allocated must equal min(total demand, Q_cap)."""
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
    """Bids below reserve price should be ignored."""
    bids = make_bids([(80, 1.0), (30, 1.0), (20, 1.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # expected under-allocation
        _, alloc, _, stats = market_clearing_ets(bids, q_cap=3.0, reserve_price=50.0)

    # Only A0 (price=80) is above reserve
    assert alloc[0] == pytest.approx(1.0)
    assert alloc[1] == pytest.approx(0.0)
    assert alloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: All bids below reserve → auction fails
# ---------------------------------------------------------------------------

def test_all_below_reserve():
    bids = make_bids([(10, 1.0), (20, 1.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0, reserve_price=50.0)

    assert stats["auction_failed"] is True
    assert alloc.sum() == pytest.approx(0.0)
    assert pay.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 6: Single bidder
# ---------------------------------------------------------------------------

def test_single_bidder():
    bids = make_bids([(65, 3.0)])
    price, alloc, pay, stats = market_clearing_ets(bids, q_cap=2.0)

    assert price == pytest.approx(65.0)
    assert alloc[0] == pytest.approx(2.0)   # capped at Q_cap
    assert pay[0] == pytest.approx(65.0 * 2.0)


# ---------------------------------------------------------------------------
# Test 7: Large bid (q >> Q_cap)
# ---------------------------------------------------------------------------

def test_large_bid():
    bids = make_bids([(70, 100.0), (60, 1.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=5.0)

    assert alloc.sum() == pytest.approx(5.0, abs=1e-6)
    assert alloc[0] == pytest.approx(5.0)
    assert alloc[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 8: All bids at same price (tie-break)
# ---------------------------------------------------------------------------

def test_tie_break_pro_rata():
    """
    3 agents all bid at same price 70, each wants 2 Mt.
    Q_cap = 3.0 Mt.
    Pro-rata: each gets 1.0 Mt.
    """
    bids = make_bids([(70, 2.0), (70, 2.0), (70, 2.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=3.0)

    assert alloc.sum() == pytest.approx(3.0, abs=1e-6)
    # Each should get roughly equal allocation (pro-rata)
    for a in alloc:
        assert a == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 9: Non-negativity
# ---------------------------------------------------------------------------

def test_non_negativity():
    bids = make_bids([(80, 1.0), (60, 2.0), (40, 1.5)])
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
    Q_cap = 10.  With max_agent_share=0.25, each agent is capped at 2.5 Mt.
    A0 gets 2.5 (capped), A1 gets 2.5 (capped), total = 5 Mt (some unsold).
    """
    bids = make_bids([(200, 10.0), (100, 5.0)])
    _, alloc, _, stats = market_clearing_ets(bids, q_cap=10.0, max_agent_share=0.25)

    assert alloc[0] == pytest.approx(2.5, abs=1e-6), f"A0 should be capped at 2.5, got {alloc[0]}"
    assert alloc[1] == pytest.approx(2.5, abs=1e-6), f"A1 should be capped at 2.5, got {alloc[1]}"
    assert alloc.sum() == pytest.approx(5.0, abs=1e-6)


def test_holding_limit_no_effect_when_1():
    """With max_agent_share=1.0 (default), no cap is applied."""
    bids = make_bids([(200, 10.0), (100, 5.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=10.0, max_agent_share=1.0)

    assert alloc[0] == pytest.approx(10.0, abs=1e-6), f"A0 should get full 10, got {alloc[0]}"
    assert alloc[1] == pytest.approx(0.0, abs=1e-6)


def test_holding_limit_redistributes():
    """
    3 agents all bid at 80 for 5 Mt each. Q_cap = 10.
    Without limit: pro-rata 10/15 * 5 = 3.33 each.
    With max_agent_share=0.25 (cap=2.5): each gets 2.5, total = 7.5.
    """
    bids = make_bids([(80, 5.0), (80, 5.0), (80, 5.0)])
    _, alloc, _, _ = market_clearing_ets(bids, q_cap=10.0, max_agent_share=0.25)

    for i in range(3):
        assert alloc[i] <= 2.5 + 1e-6, f"Agent {i} exceeds cap: {alloc[i]}"

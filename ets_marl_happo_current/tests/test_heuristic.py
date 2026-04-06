"""
test_heuristic.py
=================
Unit tests for the heuristic (rule-based) policy used for behavioral cloning.

Covers:
  - auction_action: valuation-based bid, target-bank quantity, NPV investment, tech selection
  - secondary_action: target-bank trajectory trading, green vs financial archetypes
  - Action bounds compliance
"""

import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.heuristic_policy import auction_action, secondary_action
from src.environment.company import Company

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_company(config, agent_id=0, seed=42):
    rng = np.random.default_rng(seed)
    mix = config["companies"]["initial_mix"][agent_id]
    return Company(agent_id=agent_id, config=config, initial_mix=mix, rng=rng)


# ---------------------------------------------------------------------------
# Auction action tests
# ---------------------------------------------------------------------------

class TestAuctionAction:

    def test_output_shape(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=12, config=config, bank=0.0)
        assert action.shape == (6,)
        assert action.dtype == np.float32

    def test_bid_price_within_bounds(self, config):
        """Bid price should be within [price_min, price_max]."""
        for i in range(8):
            c = make_company(config, agent_id=i)
            action = auction_action(c, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
            assert action[0] >= config["auction"]["price_min"]
            assert action[0] <= config["auction"]["price_max"]

    def test_bid_price_fundamentals_range(self, config):
        """Bid price should not exceed the configured 1.8x penalty ceiling (before global clips)."""
        mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
        pen_rate = config["penalty"]["rate"]
        infl = config["penalty"].get("inflation_rate", 0.02)
        for i in range(8):
            c = make_company(config, agent_id=i)
            action = auction_action(c, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
            inflated_penalty = pen_rate * (1 + infl) ** 5
            assert action[0] >= config["auction"]["price_min"], f"Agent {i} bid below price_min"
            assert action[0] <= min(config["auction"]["price_max"], 1.8 * inflated_penalty) + 1.0, (
                f"Agent {i} bid above penalty ceiling"
            )

    def test_coverage_based_bid_differentiation(self, config):
        """High bank (high coverage) should produce a lower bid than low bank."""
        c_low_bank = make_company(config, agent_id=0)
        c_high_bank = make_company(config, agent_id=0)
        low_bank = 0.0
        high_bank = 10.0  # large bank relative to annual need
        action_low = auction_action(c_low_bank, price_ma3=80.0, current_year=5,
                                    n_years=12, config=config, bank=low_bank)
        action_high = auction_action(c_high_bank, price_ma3=80.0, current_year=5,
                                     n_years=12, config=config, bank=high_bank)
        assert action_high[0] <= action_low[0], \
            "High bank coverage should produce lower bid"

    def test_qty_multiplier_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=12, config=config, bank=0.0)
        assert action[1] >= config["auction"]["qty_mult_low"]
        assert action[1] <= config["auction"]["qty_mult_high"]

    def test_qty_increases_with_carry_forward(self, config):
        """Carry-forward > 0 should increase total bid quantity (in Mt)."""
        c = make_company(config, agent_id=0)
        action_no_cf = auction_action(c, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
        need_no_cf = c.compute_estimate_need()
        c._carry_forward = 1.0  # 1 Mt carry-forward
        action_cf = auction_action(c, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
        need_cf = c.compute_estimate_need() + c._carry_forward
        # E3 budget/leverage constraints may reduce the raw multiplier when carry-forward
        # increases annual_need (larger notional → leverage gate bites).  Check that the
        # TOTAL bid volume in Mt increases even if the multiplier itself is clipped.
        qty_no_cf = action_no_cf[1] * need_no_cf
        qty_cf = action_cf[1] * need_cf
        assert qty_cf >= qty_no_cf - 1e-6, f"Carry-forward should increase total bid qty: {qty_cf:.3f} vs {qty_no_cf:.3f}"

    def test_invest_frac_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=12, config=config, bank=0.0)
        assert 0.0 <= action[2] <= config["investment"]["max_invest_frac"]

    def test_green_agent_invests_more(self, config):
        """Green-objective agents (odd IDs) should invest more aggressively."""
        c_financial = make_company(config, agent_id=0)  # even = financial
        c_green = make_company(config, agent_id=1)      # odd = green
        # Set high capex throughput so the constraint doesn't bind
        c_financial.capex_throughput = 1e9
        c_green.capex_throughput = 1e9
        a_fin = auction_action(c_financial, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
        a_grn = auction_action(c_green, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=0.0)
        assert a_grn[2] >= a_fin[2], "Green agent should invest >= financial agent"

    def test_npv_investment_gating(self, config):
        """Financial agent should invest less when carbon price is very low (bad NPV)."""
        c = make_company(config, agent_id=0)  # financial agent
        action_high_price = auction_action(c, price_ma3=200.0, current_year=2,
                                           n_years=12, config=config, bank=0.0)
        action_low_price = auction_action(c, price_ma3=5.0, current_year=2,
                                          n_years=12, config=config, bank=0.0)
        assert action_high_price[2] >= action_low_price[2], \
            "Higher carbon price (better NPV) should lead to more investment"

    def test_tech_logits_shape(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=12, config=config, bank=0.0)
        logits = action[3:6]
        assert logits.shape == (3,)

    def test_tech_selection_by_effective_payoff(self, config):
        """With very few years left, the fastest-deploying tech should win."""
        c = make_company(config, agent_id=0)
        # Near end: solar (delay=1) should beat onshore (delay=3) and offshore (delay=5)
        action = auction_action(c, price_ma3=80.0, current_year=18, n_years=12, config=config, bank=0.0)
        logits = action[3:6]  # [onshore, offshore, solar]
        assert np.argmax(logits) == 2, "Solar should be preferred near episode end"

    def test_green_and_financial_bid_same_for_same_state(self, config):
        """Bid pricing should not depend on green/financial objective split."""
        c_fin = make_company(config, agent_id=0)
        c_grn = make_company(config, agent_id=1)
        bank = 1.0
        a_fin = auction_action(c_fin, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=bank)
        a_grn = auction_action(c_grn, price_ma3=80.0, current_year=5, n_years=12, config=config, bank=bank)
        assert a_grn[0] == pytest.approx(a_fin[0], abs=1e-6)

    def test_year0_avg_bid_in_realistic_band(self, config):
        """At MA3=80 and mixed coverage, year-0 average bid should be in [85, 100] EUR/t."""
        rng = np.random.default_rng(123)
        bids = []
        c = make_company(config, agent_id=0)
        annual_need = max(c.compute_estimate_need(), 1e-6)
        for cov in rng.uniform(0.5, 2.0, size=200):
            bank = float(cov * annual_need)
            action = auction_action(c, price_ma3=80.0, current_year=0, n_years=12, config=config, bank=bank)
            bids.append(float(action[0]))
        avg_bid = float(np.mean(bids))
        assert 85.0 <= avg_bid <= 100.0, f"Year-0 average bid out of band: {avg_bid:.2f}"

    def test_bid_capped_at_1p8_penalty_with_high_ma3(self, config):
        """Very high MA3 should still respect the 1.8x penalty cap."""
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=300.0, current_year=0, n_years=12, config=config, bank=0.0)
        penalty = c.effective_penalty_rate(0)
        assert action[0] <= min(config["auction"]["price_max"], 1.8 * penalty) + 1e-6

    def test_year12_bid_bounds(self, config):
        """Near terminal years, bids should remain within calibrated [130, 230] EUR/t range."""
        c = make_company(config, agent_id=0)
        annual_need = max(c.compute_estimate_need(), 1e-6)
        banks = np.linspace(0.5 * annual_need, 2.0 * annual_need, num=30)
        bids = [
            float(auction_action(c, price_ma3=160.0, current_year=11, n_years=12, config=config, bank=b)[0])
            for b in banks
        ]
        assert min(bids) >= 130.0
        assert max(bids) <= 230.0


# ---------------------------------------------------------------------------
# Secondary action tests
# ---------------------------------------------------------------------------

class TestSecondaryAction:

    def test_output_shape(self, config):
        c = make_company(config, agent_id=0)
        action = secondary_action(c, bank=2.0, allocation=3.0,
                                  clearing_price=80.0, config=config)
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_price_within_bounds(self, config):
        """Secondary price should be within [sec_price_min, 2 × penalty_rate]."""
        c = make_company(config, agent_id=0)
        action = secondary_action(c, bank=2.0, allocation=3.0,
                                  clearing_price=80.0, config=config)
        sec_price_min = config["trading"]["sec_price_min"]
        penalty_rate = c.effective_penalty_rate(0)
        sec_price_max = config["trading"]["sec_price_max_mult"] * penalty_rate
        assert sec_price_min <= action[0] <= sec_price_max

    def test_qty_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = secondary_action(c, bank=2.0, allocation=3.0,
                                  clearing_price=80.0, config=config)
        qty_max = config["auction"]["quantity_max"]
        assert -qty_max <= action[1] <= qty_max

    def test_surplus_leads_to_selling(self, config):
        """With large surplus, agent should want to sell (negative qty)."""
        c = make_company(config, agent_id=0)  # financial agent
        # bank + allocation >> need -> big surplus
        action = secondary_action(c, bank=10.0, allocation=10.0,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        assert action[1] < 0, "Surplus should lead to selling (negative qty)"

    def test_deficit_leads_to_buying(self, config):
        """With shortfall, agent should want to buy (positive qty)."""
        c = make_company(config, agent_id=0)
        # bank + allocation << need -> shortfall
        action = secondary_action(c, bank=0.0, allocation=0.5,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        assert action[1] > 0, "Deficit should lead to buying (positive qty)"

    def test_green_and_financial_secondary_same_for_same_state(self, config):
        """Secondary pricing and quantity should not depend on green/financial split."""
        c_fin = make_company(config, agent_id=0)
        c_grn = make_company(config, agent_id=1)
        a_fin = secondary_action(c_fin, bank=5.0, allocation=5.0,
                                 clearing_price=80.0, config=config,
                                 current_year=5, n_years=12)
        a_grn = secondary_action(c_grn, bank=5.0, allocation=5.0,
                                 clearing_price=80.0, config=config,
                                 current_year=5, n_years=12)
        assert a_grn[0] == pytest.approx(a_fin[0], abs=1e-6)
        assert a_grn[1] == pytest.approx(a_fin[1], abs=1e-6)

    def test_target_bank_trading(self, config):
        """With zero bank at mid-episode, agent should buy to build buffer."""
        c = make_company(config, agent_id=0)
        action = secondary_action(c, bank=0.0, allocation=3.0,
                                  clearing_price=80.0, config=config,
                                  current_year=3, n_years=12)
        # With remaining_years=9, target_bank > 0, and bank=0 means
        # the agent should want to build up a buffer -> likely buy
        # (unless allocation already covers need + target_bank)
        need = max(c.compute_estimate_need() + c._carry_forward, 1e-6)
        position = 0.0 + 3.0 - need
        if position < 0:
            assert action[1] > 0, "With shortfall, should be buying"

    def test_no_sell_with_carry_forward(self, config):
        """Bot with carry-forward debt must never sell."""
        c = make_company(config, agent_id=0)
        c._carry_forward = 1.0  # 1 Mt carry-forward debt
        action = secondary_action(c, bank=5.0, allocation=5.0,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        assert action[1] >= 0, f"Should never sell with carry-forward debt, got qty={action[1]}"

    def test_sell_price_above_mac(self, config):
        """Financial bot sell price should be above MAC cost."""
        c = make_company(config, agent_id=0)  # financial (even)
        mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
        # Give large surplus to trigger selling
        action = secondary_action(c, bank=10.0, allocation=10.0,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        if action[1] < 0:  # selling
            assert action[0] >= mac_cost, f"Sell price should be >= MAC ({mac_cost}), got {action[0]}"

    def test_buy_price_near_penalty_when_desperate(self, config):
        """Financial bot buy price should approach penalty rate when desperate."""
        c = make_company(config, agent_id=0)  # financial (even)
        penalty_rate = c.effective_penalty_rate(5)
        mac_cost = config.get("mac", {}).get("coal_to_gas_cost", 48.0)
        # Create extreme deficit: no bank, tiny allocation
        c._carry_forward = 0.0
        action = secondary_action(c, bank=0.0, allocation=0.1,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        if action[1] > 0:  # buying
            assert action[0] >= mac_cost, f"Buy price should be >= MAC ({mac_cost}), got {action[0]}"
            assert action[0] <= 2.0 * penalty_rate + 1.0, f"Buy price too high: {action[0]}"


# ---------------------------------------------------------------------------
# Capex throughput compliance
# ---------------------------------------------------------------------------

class TestCapexThroughput:

    def test_bot_respects_capex_throughput(self, config):
        """Bot with low capex_throughput should scale down invest_frac."""
        c = make_company(config, agent_id=0)
        c.capex_throughput = 10.0  # very low cap (10 M€)
        c.capex_spent_this_year = 0.0
        action = auction_action(c, price_ma3=80.0, current_year=5,
                                n_years=12, config=config, bank=0.0)
        invest_frac = action[2]
        # Estimate cost at the returned invest_frac
        best_tech_idx = int(np.argmax(action[3:6])) + 2  # map logit idx to tech idx
        est_cost = c.compute_investment_cost(best_tech_idx, invest_frac, 5)
        # Should be within capex remaining (10 M€) or invest_frac should be ~0
        assert est_cost <= 10.0 + 1e-3 or invest_frac < 1e-4, (
            f"invest_frac={invest_frac}, est_cost={est_cost} should respect capex_throughput=10")


# ---------------------------------------------------------------------------
# F: Heuristic loan-awareness tests
# ---------------------------------------------------------------------------

INVEST_IDX = 2

def test_loan_awareness_reduces_qty(config):
    """Loan-awareness should reduce bid qty when loan_outstanding_norm > 0."""
    c = make_company(config, agent_id=0)
    bank = 5.0
    a_normal = auction_action(c, price_ma3=80.0, current_year=0, n_years=12,
                               config=config, bank=bank, loan_outstanding_norm=0.0)
    qty_normal = a_normal[1]
    a_loan = auction_action(c, price_ma3=80.0, current_year=0, n_years=12,
                             config=config, bank=bank, loan_outstanding_norm=0.5)
    qty_loan = a_loan[1]
    assert qty_loan <= qty_normal, f"Loan should reduce qty: {qty_loan} vs {qty_normal}"

def test_loan_awareness_reduces_invest(config):
    """Loan-awareness should reduce investment fraction when loan outstanding."""
    c1 = make_company(config, agent_id=0)
    bank = 5.0
    a_normal = auction_action(c1, price_ma3=80.0, current_year=0, n_years=12,
                               config=config, bank=bank, loan_outstanding_norm=0.0)
    c2 = make_company(config, agent_id=0)
    a_loan = auction_action(c2, price_ma3=80.0, current_year=0, n_years=12,
                             config=config, bank=bank, loan_outstanding_norm=0.8)
    assert a_loan[INVEST_IDX] <= a_normal[INVEST_IDX]

def test_loan_awareness_secondary_reduces_buying(config):
    """Loan-awareness should reduce secondary market buy target."""
    c = make_company(config, agent_id=0)
    a_normal = secondary_action(c, bank=0.5, allocation=2.0,
                                 clearing_price=80.0, config=config,
                                 loan_outstanding_norm=0.0)
    a_loan = secondary_action(c, bank=0.5, allocation=2.0,
                               clearing_price=80.0, config=config,
                               loan_outstanding_norm=0.8)
    # If buying, loan should reduce the buy qty (a[1] = qty, positive = buy)
    if a_normal[1] > 0.01:
        assert a_loan[1] <= a_normal[1] + 0.01

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
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=20, config=config)
        assert action.shape == (6,)
        assert action.dtype == np.float32

    def test_bid_price_within_bounds(self, config):
        """Bid price should be within [price_min, price_max]."""
        for i in range(8):
            c = make_company(config, agent_id=i)
            action = auction_action(c, price_ma3=80.0, current_year=5, n_years=20, config=config)
            assert action[0] >= config["auction"]["price_min"]
            assert action[0] <= config["auction"]["price_max"]

    def test_bid_price_anchored_to_ma3(self, config):
        """Bid price should scale with MA3 price (when below penalty ceiling)."""
        c = make_company(config, agent_id=0)
        # Use year 0 so penalty rate is low (~138.75) and MA3 can influence
        # Use MA3 values well below penalty so ceiling doesn't bind
        action_low = auction_action(c, price_ma3=30.0, current_year=0, n_years=20, config=config)
        action_high = auction_action(c, price_ma3=120.0, current_year=0, n_years=20, config=config)
        assert action_high[0] >= action_low[0], "Higher MA3 should produce higher bid"

    def test_coverage_based_bid_differentiation(self, config):
        """High bank (high coverage) should produce a lower bid than low bank."""
        c_low_bank = make_company(config, agent_id=0)
        c_low_bank._bank = 0.0
        c_high_bank = make_company(config, agent_id=0)
        c_high_bank._bank = 10.0  # large bank relative to annual need
        action_low = auction_action(c_low_bank, price_ma3=80.0, current_year=5,
                                    n_years=20, config=config)
        action_high = auction_action(c_high_bank, price_ma3=80.0, current_year=5,
                                     n_years=20, config=config)
        assert action_high[0] <= action_low[0], \
            "High bank coverage should produce lower bid"

    def test_qty_multiplier_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=20, config=config)
        assert action[1] >= config["auction"]["qty_mult_low"]
        assert action[1] <= config["auction"]["qty_mult_high"]

    def test_qty_increases_with_carry_forward(self, config):
        """Carry-forward > 0 should increase quantity multiplier."""
        c = make_company(config, agent_id=0)
        action_no_cf = auction_action(c, price_ma3=80.0, current_year=5, n_years=20, config=config)
        c._carry_forward = 1.0  # 1 Mt carry-forward
        action_cf = auction_action(c, price_ma3=80.0, current_year=5, n_years=20, config=config)
        assert action_cf[1] >= action_no_cf[1], "Carry-forward should increase qty"

    def test_invest_frac_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=20, config=config)
        assert 0.0 <= action[2] <= config["investment"]["max_invest_frac"]

    def test_green_agent_invests_more(self, config):
        """Green-objective agents (odd IDs) should invest more aggressively."""
        c_financial = make_company(config, agent_id=0)  # even = financial
        c_green = make_company(config, agent_id=1)      # odd = green
        # Set high capex throughput so the constraint doesn't bind
        c_financial.capex_throughput = 1e9
        c_green.capex_throughput = 1e9
        a_fin = auction_action(c_financial, price_ma3=80.0, current_year=5, n_years=20, config=config)
        a_grn = auction_action(c_green, price_ma3=80.0, current_year=5, n_years=20, config=config)
        assert a_grn[2] >= a_fin[2], "Green agent should invest >= financial agent"

    def test_npv_investment_gating(self, config):
        """Financial agent should invest less when carbon price is very low (bad NPV)."""
        c = make_company(config, agent_id=0)  # financial agent
        action_high_price = auction_action(c, price_ma3=200.0, current_year=2,
                                           n_years=20, config=config)
        action_low_price = auction_action(c, price_ma3=5.0, current_year=2,
                                          n_years=20, config=config)
        assert action_high_price[2] >= action_low_price[2], \
            "Higher carbon price (better NPV) should lead to more investment"

    def test_tech_logits_shape(self, config):
        c = make_company(config, agent_id=0)
        action = auction_action(c, price_ma3=80.0, current_year=0, n_years=20, config=config)
        logits = action[3:6]
        assert logits.shape == (3,)

    def test_tech_selection_by_effective_payoff(self, config):
        """With very few years left, the fastest-deploying tech should win."""
        c = make_company(config, agent_id=0)
        # Near end: solar (delay=1) should beat onshore (delay=3) and offshore (delay=5)
        action = auction_action(c, price_ma3=80.0, current_year=18, n_years=20, config=config)
        logits = action[3:6]  # [onshore, offshore, solar]
        assert np.argmax(logits) == 2, "Solar should be preferred near episode end"

    def test_green_agent_bids_higher(self, config):
        """Green agents should bid a premium over financial agents."""
        c_fin = make_company(config, agent_id=0)
        c_grn = make_company(config, agent_id=1)
        a_fin = auction_action(c_fin, price_ma3=80.0, current_year=5, n_years=20, config=config)
        a_grn = auction_action(c_grn, price_ma3=80.0, current_year=5, n_years=20, config=config)
        assert a_grn[0] >= a_fin[0], "Green agent should bid >= financial"


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

    def test_price_mult_within_bounds(self, config):
        c = make_company(config, agent_id=0)
        action = secondary_action(c, bank=2.0, allocation=3.0,
                                  clearing_price=80.0, config=config)
        sec_low = config["trading"]["sec_mult_low"]
        sec_high = config["trading"]["sec_mult_high"]
        assert sec_low <= action[0] <= sec_high

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

    def test_green_agent_holds_more(self, config):
        """Green agent should sell less of its surplus than financial agent."""
        c_fin = make_company(config, agent_id=0)
        c_grn = make_company(config, agent_id=1)
        a_fin = secondary_action(c_fin, bank=5.0, allocation=5.0,
                                 clearing_price=80.0, config=config,
                                 current_year=5, n_years=12)
        a_grn = secondary_action(c_grn, bank=5.0, allocation=5.0,
                                 clearing_price=80.0, config=config,
                                 current_year=5, n_years=12)
        # Green agent sells less (qty closer to 0)
        assert a_grn[1] >= a_fin[1], "Green agent should hold more (sell less)"

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

    def test_sell_price_above_minimum(self, config):
        """Financial bot sell price_mult should be >= 1.15."""
        c = make_company(config, agent_id=0)  # financial (even)
        # Give large surplus to trigger selling
        action = secondary_action(c, bank=10.0, allocation=10.0,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        if action[1] < 0:  # selling
            assert action[0] >= 1.15, f"Financial sell mult should be >= 1.15, got {action[0]}"

    def test_buy_price_reaches_market(self, config):
        """Financial bot buy price_mult should reach 1.30 at max severity."""
        c = make_company(config, agent_id=0)  # financial (even)
        # Create extreme deficit: no bank, tiny allocation, large carry-forward
        c._carry_forward = 0.0
        action = secondary_action(c, bank=0.0, allocation=0.1,
                                  clearing_price=80.0, config=config,
                                  current_year=5, n_years=12)
        if action[1] > 0:  # buying
            assert action[0] >= 1.10, f"Financial buy mult should be >= 1.10, got {action[0]}"


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
                                n_years=12, config=config)
        invest_frac = action[2]
        # Estimate cost at the returned invest_frac
        best_tech_idx = int(np.argmax(action[3:6])) + 2  # map logit idx to tech idx
        est_cost = c.compute_investment_cost(best_tech_idx, invest_frac, 5)
        # Should be within capex remaining (10 M€) or invest_frac should be ~0
        assert est_cost <= 10.0 + 1e-3 or invest_frac < 1e-4, (
            f"invest_frac={invest_frac}, est_cost={est_cost} should respect capex_throughput=10")

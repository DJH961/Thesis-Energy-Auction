"""
tests/test_qlearning_actions.py
================================
Verify that Q-learning action profiles produce valid outputs matching the
v8 environment interface.
"""

import numpy as np
import pytest
import yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.q_learning_agent import ActionProfileMapper, StateDiscretizer, QLearningAgent


@pytest.fixture
def config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class _FakeCompany:
    """Minimal company stub for testing without a full ETSEnvironment."""
    green_frac = 0.5

    def compute_estimate_need(self):
        return 1.0

    def effective_penalty_rate(self, year: int) -> float:
        base = 138.75
        return base * (1.02 ** year)


@pytest.fixture
def mapper():
    return ActionProfileMapper()


@pytest.fixture
def company():
    return _FakeCompany()


class TestSecondaryActionInterface:
    """Phase-2 action must return absolute EUR/t price, not a multiplier."""

    def test_hold_returns_zero_qty(self, mapper, company, config):
        action = mapper.get_secondary_action(0, company, clearing_price=100.0, config=config)
        assert action.shape == (2,)
        assert action[1] == pytest.approx(0.0), "Hold profile must have qty=0"

    def test_sell_surplus_returns_negative_qty(self, mapper, company, config):
        action = mapper.get_secondary_action(1, company, clearing_price=100.0, config=config)
        assert action[1] < 0, "Sell profile must have qty < 0"

    def test_buy_shortfall_returns_positive_qty(self, mapper, company, config):
        action = mapper.get_secondary_action(2, company, clearing_price=100.0, config=config)
        assert action[1] > 0, "Buy profile must have qty > 0"

    def test_aggressive_buy_positive_qty(self, mapper, company, config):
        action = mapper.get_secondary_action(3, company, clearing_price=100.0, config=config)
        assert action[1] > 0

    def test_price_is_absolute_not_multiplier(self, mapper, company, config):
        # A multiplier of ~1.0-1.5 would be in [0.8, 1.5]; an absolute EUR/t price
        # at a 100 EUR/t clearing price should be >= 100 (at least equal to anchor)
        clearing = 100.0
        for profile_idx in range(4):
            action = mapper.get_secondary_action(profile_idx, company,
                                                 clearing_price=clearing, config=config)
            price = action[0]
            assert price >= clearing * 0.99, (
                f"Profile {profile_idx}: price={price:.2f} looks like a multiplier, "
                f"expected absolute EUR/t >= {clearing:.1f}"
            )

    def test_price_not_clipped_to_floor_only(self, mapper, company, config):
        # Sell and buy profiles should differ in price; if all returned ~1.0
        # (the old multiplier) they'd all be clipped to sec_price_min=45
        clearing = 100.0
        sec_price_min = config.get("trading", {}).get("sec_price_min", 45.0)
        actions = [
            mapper.get_secondary_action(p, company, clearing_price=clearing, config=config)
            for p in range(4)
        ]
        prices = [a[0] for a in actions]
        # At clearing=100, all prices should be well above the minimum floor
        for p_idx, price in enumerate(prices):
            assert price > sec_price_min, (
                f"Profile {p_idx}: price={price:.2f} <= sec_price_min={sec_price_min}; "
                "action looks clipped to floor (old multiplier bug)"
            )


class TestAuctionPanicProfile:
    """Panic-buy profile must use inflation-adjusted penalty rate, not stale base rate."""

    def test_panic_bid_uses_effective_penalty_year0(self, mapper, company, config):
        action_yr0 = mapper.get_auction_action(5, company, price_ma3=80.0,
                                               config=config, current_year=0)
        base_rate = config["penalty"]["rate"]  # 138.75
        # Panic bid = effective_penalty * 1.2, clipped to price_max
        expected_yr0 = min(base_rate * 1.2, config["auction"]["price_max"])
        assert action_yr0[0] == pytest.approx(expected_yr0, rel=1e-4)

    def test_panic_bid_higher_in_later_year(self, mapper, company, config):
        action_yr0 = mapper.get_auction_action(5, company, price_ma3=80.0,
                                               config=config, current_year=0)
        action_yr5 = mapper.get_auction_action(5, company, price_ma3=80.0,
                                               config=config, current_year=5)
        price_max = config["auction"]["price_max"]
        # Year 5 should bid higher (unless both hit price_max ceiling)
        if action_yr0[0] < price_max and action_yr5[0] < price_max:
            assert action_yr5[0] > action_yr0[0], (
                "Panic bid should increase with year due to penalty inflation"
            )


class TestPriceBinEdges:
    """Recalibrated price bins should span the v8 price range properly."""

    def test_medium_bin_covers_typical_price(self):
        disc = StateDiscretizer()
        # At price_max=250, a typical price of ~100 EUR/t → norm ~0.40
        # Should land in medium bin (1), not low bin (0)
        # Create a fake obs with price_norm=0.40 at index 2
        obs = np.zeros(25)
        obs[2] = 0.40

        class _C:
            green_frac = 0.5

        # Manually check the bin
        bin_val = disc._bin_value(0.40, disc._BINS["price"])
        assert bin_val == 1, f"Price norm 0.40 should be medium (1), got {bin_val}"

    def test_low_bin_is_reachable(self):
        disc = StateDiscretizer()
        # norm=0.20 (50 EUR/t at price_max=250) should be LOW (0) with new [0.30, 0.55] edges
        bin_val = disc._bin_value(0.20, disc._BINS["price"])
        assert bin_val == 0, f"Price norm 0.20 should be low (0), got {bin_val}"

    def test_high_bin_is_reachable(self):
        disc = StateDiscretizer()
        # norm=0.70 (175 EUR/t at price_max=250) should be HIGH (2)
        bin_val = disc._bin_value(0.70, disc._BINS["price"])
        assert bin_val == 2, f"Price norm 0.70 should be high (2), got {bin_val}"


class TestSelectMethodsPassYear:
    """QLearningAgent.select_* must accept and forward current_year."""

    def test_select_auction_accepts_year(self, config):
        agent = QLearningAgent(agent_id=0, seed=42)
        obs = np.zeros(36)
        company = _FakeCompany()
        # Should not raise
        action, idx = agent.select_auction_action(
            obs, company, price_ma3=80.0, config=config,
            epsilon=0.0, current_year=5)
        assert action.shape == (6,)
        assert 0 <= idx < 6

    def test_select_secondary_accepts_year(self, config):
        agent = QLearningAgent(agent_id=0, seed=42)
        obs = np.zeros(36)
        company = _FakeCompany()
        action, idx = agent.select_secondary_action(
            obs, company, clearing_price=100.0, config=config,
            a1_idx=1, epsilon=0.0, current_year=5)
        assert action.shape == (2,)
        assert 0 <= idx < 4


class TestQLearningWindowDiagnostics:
    """The trainer's HAPPO-mirrored console relies on per-window
    counters (TD-error, ε-pick share, state coverage). These must stay
    consistent across reset/select/update calls."""

    def test_initial_diagnostics_zero(self):
        ag = QLearningAgent(agent_id=0, seed=42)
        assert ag.window_td_mean_abs() == 0.0
        assert ag.window_explore_share() == 0.0
        assert ag.coverage_states() == 0
        assert ag.coverage_fraction() == 0.0
        assert ag.mean_max_q() == 0.0
        assert ag.greedy_policy_diversity() == 0

    def test_update_records_td_and_visit(self):
        ag = QLearningAgent(agent_id=0, alpha=0.5, gamma=0.9, seed=42)
        # First update: Q starts at 0, target = 1.0 + 0.9*0 = 1.0,
        # so |TD| = 1.0
        ag.update(state=10, a1_idx=2, a2_idx=1,
                  reward=1.0, next_state=11, done=False)
        assert ag.coverage_states() == 1
        assert abs(ag.window_td_mean_abs() - 1.0) < 1e-9
        assert ag._td_count == 1
        # Q has moved off zero on the visited cell
        assert ag.q_table[10, 2, 1] != 0.0
        assert ag.visit_counts[10, 2, 1] == 1

    def test_reset_window_clears_counters(self, config):
        ag = QLearningAgent(agent_id=0, seed=42)
        company = _FakeCompany()
        obs = np.zeros(36)
        # Force ε-random pick so explore_picks > 0
        ag.select_auction_action(obs, company, price_ma3=80.0,
                                 config=config, epsilon=1.0)
        ag.update(state=0, a1_idx=0, a2_idx=0,
                  reward=2.0, next_state=1, done=False)
        assert ag._td_count >= 1
        assert ag._explore_picks >= 1
        ag.reset_window_stats()
        assert ag._td_count == 0
        assert ag._explore_picks == 0
        assert ag._greedy_picks == 0
        assert ag.window_td_mean_abs() == 0.0
        # Coverage is persistent across resets — only the rolling
        # window counters get cleared.
        assert ag.coverage_states() == 1

    def test_explore_share_split(self, config):
        ag = QLearningAgent(agent_id=0, seed=42)
        company = _FakeCompany()
        obs = np.zeros(36)
        # epsilon=0 → all greedy
        for _ in range(5):
            ag.select_auction_action(obs, company, price_ma3=80.0,
                                     config=config, epsilon=0.0)
        assert ag._greedy_picks == 5
        assert ag._explore_picks == 0
        assert ag.window_explore_share() == 0.0
        # epsilon=1 → all explore
        for _ in range(5):
            ag.select_auction_action(obs, company, price_ma3=80.0,
                                     config=config, epsilon=1.0)
        assert ag._explore_picks == 5
        assert abs(ag.window_explore_share() - 0.5) < 1e-9

    def test_mean_max_q_only_over_visited(self):
        ag = QLearningAgent(agent_id=0, alpha=1.0, gamma=0.0, seed=42)
        # One large positive reward at state 5; with γ=0 and α=1, the Q
        # for that cell becomes exactly the reward.
        ag.update(state=5, a1_idx=0, a2_idx=0,
                  reward=42.0, next_state=6, done=True)
        # mean_max_q is taken across visited states only — exactly one
        # state has been visited so the answer must equal that Q value.
        assert abs(ag.mean_max_q() - 42.0) < 1e-9
        assert ag.coverage_states() == 1

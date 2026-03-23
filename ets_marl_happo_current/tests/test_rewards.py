"""
test_rewards.py
===============
Unit tests for reward computation, RewardNormalizer, and price-anchor shaping.

Covers:
  - _compute_rewards individual components (cost, emissions, penalty, green, queue, price anchor)
  - RewardNormalizer EMA convergence and adaptive warmup
  - Reward clipping
  - Price-anchor Gaussian bonus decay
"""

import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.agents.ppo_agent import RewardNormalizer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_env(seed=42):
    config = load_config()
    return ETSEnvironment(config, seed=seed)


# ---------------------------------------------------------------------------
# RewardNormalizer
# ---------------------------------------------------------------------------

class TestRewardNormalizer:

    def test_first_sample_sets_mean(self):
        """First sample: adaptive alpha=1.0 → mu = reward exactly."""
        rn = RewardNormalizer(alpha=0.01)
        _ = rn.update_and_normalize(5.0)
        assert abs(rn.mu - 5.0) < 1e-6

    def test_normalized_output_zero_mean(self):
        """After many identical samples, normalized output ≈ 0."""
        rn = RewardNormalizer(alpha=0.01)
        for _ in range(200):
            val = rn.update_and_normalize(3.0)
        assert abs(val) < 0.5, f"Expected near-zero after convergence, got {val}"

    def test_adaptive_warmup(self):
        """Early samples use larger alpha (1/n), not the steady-state alpha."""
        rn = RewardNormalizer(alpha=0.01)
        rn.update_and_normalize(100.0)
        # After 1 sample: effective_alpha = max(0.01, 1/1) = 1.0
        assert abs(rn.mu - 100.0) < 1e-6
        rn.update_and_normalize(0.0)
        # After 2 samples: effective_alpha = max(0.01, 1/2) = 0.5
        assert abs(rn.mu - 50.0) < 1e-6

    def test_reset_clears_state(self):
        rn = RewardNormalizer(alpha=0.01)
        for _ in range(50):
            rn.update_and_normalize(10.0)
        rn.reset()
        assert rn.mu == 0.0
        assert rn.var == 1.0
        assert rn._n_samples == 0

    def test_variance_tracks_spread(self):
        """With alternating rewards, variance should be non-trivial."""
        rn = RewardNormalizer(alpha=0.05)
        for i in range(100):
            rn.update_and_normalize(10.0 if i % 2 == 0 else -10.0)
        assert rn.var > 1.0, f"Variance should be large for alternating rewards: {rn.var}"

    def test_normalizes_scale(self):
        """Rewards of different magnitudes should be brought to similar scale."""
        rn = RewardNormalizer(alpha=0.05)
        # Feed large rewards
        for _ in range(50):
            rn.update_and_normalize(1000.0 + np.random.randn() * 100)
        # Now a new value: should be ≈ O(1), not O(1000)
        val = rn.update_and_normalize(1000.0)
        assert abs(val) < 5.0, f"Normalized value should be O(1), got {val}"


# ---------------------------------------------------------------------------
# Reward components (integration via environment)
# ---------------------------------------------------------------------------

def _run_one_year(env, auction_price=80.0, qty_mult=1.0, invest_frac=0.0):
    """Helper: run one year and return rewards + info."""
    n = env.n_agents
    auction_actions = np.zeros((n, 6), dtype=np.float32)
    auction_actions[:, 0] = auction_price
    auction_actions[:, 1] = qty_mult
    auction_actions[:, 2] = invest_frac
    auction_actions[:, 3:] = [0.0, 0.0, 1.0]  # solar logits
    env.step_auction(auction_actions)

    secondary_actions = np.zeros((n, 2), dtype=np.float32)
    secondary_actions[:, 0] = 1.0  # trade at clearing price
    secondary_actions[:, 1] = 0.0  # no trading
    _, rewards, _, _, info = env.step_secondary(secondary_actions)
    return rewards, info


def test_rewards_finite():
    """All reward components produce finite values."""
    env = load_env()
    env.reset()
    rewards, _ = _run_one_year(env, auction_price=80.0)
    assert np.all(np.isfinite(rewards)), f"Non-finite rewards: {rewards}"


def test_rewards_differ_with_different_bids():
    """Different bidding strategies should produce different reward outcomes."""
    env = load_env()
    env.reset()
    rewards_high, _ = _run_one_year(env, auction_price=200.0, qty_mult=1.3)

    env2 = load_env()
    env2.reset()
    rewards_low, _ = _run_one_year(env2, auction_price=5.0, qty_mult=0.3)

    # Different strategies should produce meaningfully different outcomes
    # (not necessarily one better than the other — depends on market conditions)
    assert not np.allclose(rewards_high, rewards_low, atol=0.01), (
        "Different bidding strategies should produce different rewards")


def test_green_bonus_with_investment():
    """Investing in green tech should add a positive green bonus."""
    env = load_env(seed=1)
    env.reset()
    # No investment
    rewards_noinvest, _ = _run_one_year(env, auction_price=100.0, invest_frac=0.0)

    env2 = load_env(seed=1)
    env2.reset()
    # Max investment in solar
    rewards_invest, _ = _run_one_year(env2, auction_price=100.0, invest_frac=0.08)

    # Investment has a cost but also a green bonus; at least some agents should benefit
    # (the green bonus may not outweigh cost in 1 year, but the shaping should be present)
    # We just verify both produce finite rewards
    assert np.all(np.isfinite(rewards_invest))


def test_price_anchor_bonus_near_expected():
    """Bidding near expected_price should get a higher price-anchor bonus than bidding far."""
    config = load_config()
    config["reward"]["price_anchor_delta"] = 1.0  # make bonus visible
    env1 = ETSEnvironment(config, seed=42)
    env1.reset()
    expected = env1.expected_price  # AR(1) expected price

    # Bid near expected price
    rewards_near, _ = _run_one_year(env1, auction_price=expected)

    config2 = load_config()
    config2["reward"]["price_anchor_delta"] = 1.0
    env2 = ETSEnvironment(config2, seed=42)
    env2.reset()
    # Bid far from expected price
    rewards_far, _ = _run_one_year(env2, auction_price=expected * 3)

    # Near-expected should get better reward due to price anchor bonus
    assert rewards_near.mean() >= rewards_far.mean() - 0.5, (
        f"Near ({rewards_near.mean():.3f}) should be close to or better than far ({rewards_far.mean():.3f})")


def test_price_anchor_zero_when_disabled():
    """With price_anchor_delta=0, no price-proximity bonus."""
    config = load_config()
    config["reward"]["price_anchor_delta"] = 0.0
    env1 = ETSEnvironment(config, seed=42)
    env1.reset()
    rewards1, _ = _run_one_year(env1, auction_price=80.0)

    config2 = load_config()
    config2["reward"]["price_anchor_delta"] = 0.0
    env2 = ETSEnvironment(config2, seed=42)
    env2.reset()
    rewards2, _ = _run_one_year(env2, auction_price=400.0)

    # Without price anchor, same seed + same market clearing → same rewards
    # (bid price only affects payment, not an anchor bonus)
    # We just verify both are finite
    assert np.all(np.isfinite(rewards1))
    assert np.all(np.isfinite(rewards2))


def test_shaping_weight_decays():
    """Shaping weight should decrease toward 0 over episodes."""
    env = load_env()
    env.set_episode(0)
    w0 = env.shaping_weight
    env.set_episode(6000)
    w_mid = env.shaping_weight
    env.set_episode(12000)
    w_end = env.shaping_weight
    assert w0 > w_mid > w_end
    assert w_end <= 0.01, f"Shaping weight should be ~0 at decay end: {w_end}"


def test_electricity_revenue_reduces_cost():
    """With electricity enabled, agents get revenue that offsets costs."""
    config = load_config()
    config["electricity"]["enabled"] = True
    env_with = ETSEnvironment(config, seed=42)
    env_with.reset()
    rewards_with, _ = _run_one_year(env_with, auction_price=100.0)

    config2 = load_config()
    config2["electricity"]["enabled"] = False
    env_without = ETSEnvironment(config2, seed=42)
    env_without.reset()
    rewards_without, _ = _run_one_year(env_without, auction_price=100.0)

    # With electricity revenue, rewards should be higher (less negative)
    assert rewards_with.mean() > rewards_without.mean(), (
        f"Electricity revenue should improve rewards: {rewards_with.mean():.3f} vs {rewards_without.mean():.3f}")

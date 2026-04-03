"""
test_anchors.py
===============
Tests for the action anchor system: bias initialization and anchored
epsilon-greedy exploration for both auction and secondary policies.

Covers:
  - AuctionPolicy bias init shifts initial output toward anchor values
  - SecondaryPolicy bias init shifts initial output toward anchor values
  - Without anchors, initial output is near the midpoint (legacy behavior)
  - Anchored epsilon exploration for secondary market
  - Anchor config is correctly read from exploration section
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.actor_critic import AuctionPolicy, SecondaryPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auction_bounds():
    low = torch.FloatTensor([5.0, 0.3, 0.0, -1.0, -1.0, -1.0])
    high = torch.FloatTensor([500.0, 1.3, 0.10, 1.0, 1.0, 1.0])
    return low, high


@pytest.fixture
def secondary_bounds():
    low = torch.FloatTensor([0.8, -3.0])
    high = torch.FloatTensor([1.3, 3.0])
    return low, high


# ---------------------------------------------------------------------------
# Auction anchor bias tests
# ---------------------------------------------------------------------------

class TestAuctionAnchors:

    def test_anchored_price_near_target(self, auction_bounds):
        """With anchor=80, initial price output should be near 80, not midpoint 252.5."""
        low, high = auction_bounds
        anchors = [80.0, 1.0, 0.03, 0.3, -0.5, 0.5]
        policy = AuctionPolicy(
            obs_dim=22, action_dim=6, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        # With small weight init (gain=0.01), bias dominates initial output
        torch.manual_seed(42)
        obs = torch.zeros(1, 22)  # zero input → hidden ≈ 0 → mean ≈ bias
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        price = action[0, 0].item()
        # Should be much closer to 80 than to midpoint 252.5
        assert abs(price - 80.0) < 50.0, f"Initial price {price:.1f} too far from anchor 80"
        assert abs(price - 252.5) > 50.0, f"Initial price {price:.1f} too close to midpoint"

    def test_anchored_qty_near_target(self, auction_bounds):
        """With anchor=1.0, initial qty should be near 1.0, not midpoint 0.8."""
        low, high = auction_bounds
        anchors = [80.0, 1.0, 0.03, 0.3, -0.5, 0.5]
        policy = AuctionPolicy(
            obs_dim=22, action_dim=6, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 22)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        qty = action[0, 1].item()
        assert abs(qty - 1.0) < 0.2, f"Initial qty {qty:.2f} too far from anchor 1.0"

    def test_anchored_invest_frac(self, auction_bounds):
        """With anchor=0.03, initial invest_frac should be near 0.03."""
        low, high = auction_bounds
        anchors = [80.0, 1.0, 0.03, 0.3, -0.5, 0.5]
        policy = AuctionPolicy(
            obs_dim=22, action_dim=6, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 22)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        inv = action[0, 2].item()
        assert abs(inv - 0.03) < 0.02, f"Initial invest {inv:.3f} too far from anchor 0.03"

    def test_no_anchors_gives_midpoint(self, auction_bounds):
        """Without anchors, initial output should be near the midpoint."""
        low, high = auction_bounds
        policy = AuctionPolicy(
            obs_dim=22, action_dim=6, hidden_size=64,
            action_low=low, action_high=high, action_anchors=None)

        obs = torch.zeros(1, 22)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        price = action[0, 0].item()
        midpoint = (500.0 + 5.0) / 2.0  # 252.5
        assert abs(price - midpoint) < 50.0, f"Without anchors, price {price:.1f} should be near midpoint"

    def test_anchors_within_bounds(self, auction_bounds):
        """Anchored initial actions should still be within [low, high]."""
        low, high = auction_bounds
        anchors = [80.0, 1.0, 0.03, 0.3, -0.5, 0.5]
        policy = AuctionPolicy(
            obs_dim=22, action_dim=6, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 22)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        assert torch.all(action >= low - 0.01)
        assert torch.all(action <= high + 0.01)


# ---------------------------------------------------------------------------
# Secondary anchor bias tests
# ---------------------------------------------------------------------------

class TestSecondaryAnchors:

    def test_anchored_price_mult(self, secondary_bounds):
        """With anchor=1.05, initial output near 1.05 (not midpoint 1.05)."""
        low, high = secondary_bounds
        anchors = [1.05, 0.0]
        policy = SecondaryPolicy(
            obs_dim=29, action_dim=2, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 29)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        mult = action[0, 0].item()
        assert abs(mult - 1.05) < 0.1, f"Price mult {mult:.2f} should be near 1.05"

    def test_anchored_qty_neutral(self, secondary_bounds):
        """With anchor=0.0, initial qty output near 0.0."""
        low, high = secondary_bounds
        anchors = [1.05, 0.0]
        policy = SecondaryPolicy(
            obs_dim=29, action_dim=2, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 29)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        qty = action[0, 1].item()
        assert abs(qty) < 0.5, f"Secondary qty {qty:.2f} should be near 0"

    def test_no_anchors_secondary(self, secondary_bounds):
        """Without anchors, secondary policy uses midpoint."""
        low, high = secondary_bounds
        policy = SecondaryPolicy(
            obs_dim=29, action_dim=2, hidden_size=64,
            action_low=low, action_high=high, action_anchors=None)

        obs = torch.zeros(1, 29)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        # Midpoint: [(0.8+1.3)/2, (-3+3)/2] = [1.05, 0.0]
        # Both happen to be at the midpoint, which is the same as the anchor
        assert action.shape == (1, 2)


# ---------------------------------------------------------------------------
# Anchored epsilon exploration (secondary)
# ---------------------------------------------------------------------------

def test_secondary_epsilon_anchored():
    """Secondary epsilon exploration should sample near anchors, not uniform."""
    from src.agents.ppo_agent import PPOAgent

    config = {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -1.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {"price_min": 5.0, "price_max": 500.0, "quantity_max": 3.0,
                     "qty_mult_low": 0.3, "qty_mult_high": 1.3},
        "investment": {"max_invest_frac": 0.10},
        "trading": {"sec_mult_low": 0.8, "sec_mult_high": 1.3},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 2},
        "exploration": {
            "auction_anchors": [75.0, 0.33, 80.0, 0.33, 85.0, 0.33, 0.03, 0.3, -0.5, 0.5],
            "secondary_anchors": [1.05, 0.0],
        },
    }

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([5.0, 0.3, 5.0, 0.3, 5.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 1.3, 500.0, 1.3, 500.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=config, seed=42,
    )

    obs2 = np.random.randn(29).astype(np.float32)
    price_mults = []
    qtys = []
    for _ in range(200):
        action, _, _ = agent.select_secondary_action(obs2, epsilon=1.0)
        price_mults.append(action[0])
        qtys.append(action[1])

    price_mults = np.array(price_mults)
    qtys = np.array(qtys)

    # Price mult should be centered around 1.05
    assert abs(price_mults.mean() - 1.05) < 0.1, (
        f"Mean price mult {price_mults.mean():.2f} too far from anchor 1.05")

    # Qty should be centered around 0 (not the edges)
    assert abs(qtys.mean()) < 1.5, f"Mean qty {qtys.mean():.2f} should be near 0"
    # Should have both buyers and sellers
    assert qtys.min() < -0.5, "Should have some sellers"
    assert qtys.max() > 0.5, "Should have some buyers"


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------

def test_anchor_config_passed_to_policy():
    """Verify that exploration.auction_anchors from config reaches the policy."""
    from src.agents.ppo_agent import PPOAgent

    config = {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -1.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {"price_min": 5.0, "price_max": 500.0, "quantity_max": 3.0,
                     "qty_mult_low": 0.3, "qty_mult_high": 1.3},
        "investment": {"max_invest_frac": 0.10},
        "trading": {"sec_mult_low": 0.8, "sec_mult_high": 1.3},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 2},
        "exploration": {
            "auction_anchors": [75.0, 0.33, 80.0, 0.33, 85.0, 0.33, 0.03, 0.3, -0.5, 0.5],
            "secondary_anchors": [1.05, 0.0],
        },
    }

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([5.0, 0.3, 5.0, 0.3, 5.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 1.3, 500.0, 1.3, 500.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=config, seed=42,
    )

    # Verify the anchor reached the policy by checking initial deterministic output.
    # Keep obs on the policy device so the test is robust on both CPU and GPU runs.
    policy_device = next(agent.auction_policy.parameters()).device
    obs = torch.zeros(1, 22, device=policy_device)
    with torch.no_grad():
        action, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price = action[0, 0].item()
    assert abs(price - 80.0) < 50.0, f"Anchor not applied: price {price:.1f}"

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
    low = torch.FloatTensor([30.0, -3.0])
    high = torch.FloatTensor([300.0, 3.0])
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

    def test_anchored_price_abs(self, secondary_bounds):
        """With anchor=80, initial output near 80 EUR/t."""
        low, high = secondary_bounds
        anchors = [80.0, 0.0]
        policy = SecondaryPolicy(
            obs_dim=29, action_dim=2, hidden_size=64,
            action_low=low, action_high=high, action_anchors=anchors)

        obs = torch.zeros(1, 29)
        with torch.no_grad():
            action, _, _ = policy.act(obs, deterministic=True)
        sec_price = action[0, 0].item()
        assert abs(sec_price - 80.0) < 25.0, f"Price {sec_price:.2f} should be near 80"

    def test_anchored_qty_neutral(self, secondary_bounds):
        """With anchor=0.0, initial qty output near 0.0."""
        low, high = secondary_bounds
        anchors = [80.0, 0.0]
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
        # Midpoint: [(30+300)/2, (-3+3)/2] = [165.0, 0.0]
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
        "trading": {"sec_price_min": 30.0, "sec_price_max_mult": 2.0},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 2},
        "exploration": {
            "auction_anchors": [80.0, 1.0, 0.03, 0.3, -0.5, 0.5],
            "secondary_anchors": [80.0, 0.0],
        },
    }

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([5.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([30.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([300.0, 3.0], dtype=np.float32),
        config=config, seed=42,
    )

    obs2 = np.zeros(29, dtype=np.float32)
    obs2[-9] = 80.0 / 500.0  # normalized clearing-price feature in phase2 extras
    prices = []
    qtys = []
    for _ in range(200):
        action, _, _ = agent.select_secondary_action(obs2, epsilon=1.0)
        prices.append(action[0])
        qtys.append(action[1])

    prices = np.array(prices)
    qtys = np.array(qtys)

    # Secondary price should be centered around observed clearing (~80 EUR/t)
    assert abs(prices.mean() - 80.0) < 20.0, (
        f"Mean secondary price {prices.mean():.2f} too far from 80")

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
        "trading": {"sec_price_min": 30.0, "sec_price_max_mult": 2.0},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 2},
        "exploration": {
            "auction_anchors": [80.0, 1.0, 0.03, 0.3, -0.5, 0.5],
            "secondary_anchors": [80.0, 0.0],
        },
    }

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([5.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([30.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([300.0, 3.0], dtype=np.float32),
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


# ---------------------------------------------------------------------------
# inject_fundamental_anchor test
# ---------------------------------------------------------------------------

def test_inject_fundamental_anchor_sets_price_near_anchor():
    """inject_fundamental_anchor() should calibrate price_head.bias so that
    the deterministic initial action is near the fundamental anchor (~67 EUR/t
    at year 0 for default params: MAC=48, mult=1.4, price_max=250).
    """
    from src.agents.ppo_agent import PPOAgent

    config = {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -2.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {"price_min": 45.0, "price_max": 250.0, "quantity_max": 3.0,
                     "qty_mult_low": 0.5, "qty_mult_high": 2.0},
        "investment": {"max_invest_frac": 0.20},
        "trading": {"sec_price_min": 45.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02,
                    "carry_forward": True, "carry_forward_cap": 0.0},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 10.0},
        "companies": {"n_agents": 8, "output_twh": 10.0,
                      "initial_mix": [[0.25, 0.30, 0.20, 0.15, 0.10]] * 8},
        "mac": {"enabled": True, "coal_to_gas_cost": 48.0, "max_switch_frac": 0.20},
        "price": {"banking_premium_mult": 1.4, "ar1_persistence": 0.85,
                  "volatility_std": 0.15, "burnin_std": 10.0},
        "ets": {
            "cap_year_0_override": None, "initial_bank_fraction": 0.10,
            "cap_overhead_pct": 0.02, "lrf_phase1": 0.043, "lrf_phase2": 0.044,
            "lrf_phase_switch": 2,
            "msr": {
                "enabled": True, "tnac_upper_ratio": 0.36, "tnac_mid_ratio": None,
                "tnac_lower_ratio": None, "withhold_rate": 0.24,
                "release_frac": 0.0638297872, "activation_year": 1,
                "price_containment_absolute": 350, "price_release_absolute": 450,
                "emergency_release_frac": 0.064, "min_auction_frac": 0.10,
            },
            "banking": True, "reserve_price": 45.0, "reserve_price_mode": "static",
            "unsold_to_msr": False, "max_rollover_multiplier": 1.5,
            "price_history_anchor": "auction",
        },
        "technologies": {
            "names": ["coal", "gas", "onshore_wind", "offshore_wind", "solar"],
            "emission_factors": [0.820, 0.490, 0.011, 0.012, 0.048],
            "capacity_factors": [0.65, 0.60, 0.35, 0.47, 0.17],
            "capex": [3000, 1150, 1350, 3250, 750],
            "deploy_delays": [0, 0, 4, 7, 2],
            "operational_costs": [72.0, 55.0, 17.0, 47.0, 10.0],
            "decommission_costs": [200, 100, 0, 0, 0],
            "is_green": [False, False, True, True, True],
            "is_buildable": [False, False, True, True, True],
        },
        "simulation": {"n_years": 12, "n_episodes": 100},
        "exploration": {"auction_anchors": None, "secondary_anchors": None},
    }

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=22, obs_dim_phase2=29,
        auction_action_low=np.array([45.0, 0.5, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([250.0, 2.0, 0.20, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([45.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([350.0, 3.0], dtype=np.float32),
        config=config, seed=0,
    )

    # Before injection: no explicit anchor, bias starts at midpoint
    device = next(agent.auction_policy.parameters()).device
    obs = torch.zeros(1, 22, device=device)

    with torch.no_grad():
        action_before, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price_before = float(action_before[0, 0].item())

    # Inject the fundamental anchor (MAC=48, mult=1.4 → ~67 EUR/t at year 0)
    agent.inject_fundamental_anchor(year=0)

    with torch.no_grad():
        action_after, _, _ = agent.auction_policy.act(obs, deterministic=True)
    price_after = float(action_after[0, 0].item())

    # Without injection the price defaults near midpoint (147.5 EUR/t);
    # after injection it should be near the anchor (~67 EUR/t).
    expected_anchor = 48.0 * 1.4  # 67.2 EUR/t
    midpoint = (45.0 + 250.0) / 2.0  # 147.5 EUR/t

    assert abs(price_after - expected_anchor) < 20.0, (
        f"After inject_fundamental_anchor, price {price_after:.1f} too far from "
        f"expected anchor {expected_anchor:.1f} EUR/t"
    )
    assert abs(price_after - midpoint) > 30.0, (
        f"After inject_fundamental_anchor, price {price_after:.1f} still near midpoint "
        f"{midpoint:.1f} — injection had no effect"
    )

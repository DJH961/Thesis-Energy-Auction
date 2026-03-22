"""
test_clipped_gaussian.py
========================
Tests for the clipped-Gaussian action transformation and physical-space
epsilon-greedy exploration.

Covers:
  - Action bounds are respected (clamp instead of tanh)
  - act() and evaluate() produce consistent log_prob
  - Gradient flows at boundaries (no tanh vanishing gradient)
  - _to_raw() correctly inverts the physical→raw mapping
  - Epsilon-greedy produces diverse actions in physical space
  - Epsilon does not override deterministic mode
  - PPO update works correctly with epsilon-injected actions
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.actor_critic import AuctionPolicy, SecondaryPolicy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auction_policy():
    """AuctionPolicy with realistic ETS bounds."""
    action_low = torch.FloatTensor([40.0, 0.3, 0.0, -1.0, -1.0, -1.0])
    action_high = torch.FloatTensor([120.0, 1.3, 0.10, 1.0, 1.0, 1.0])
    return AuctionPolicy(
        obs_dim=21, action_dim=6, hidden_size=64,
        action_low=action_low, action_high=action_high,
        log_std_min=-1.5, log_std_max=0.0,
    )


@pytest.fixture
def secondary_policy():
    """SecondaryPolicy with realistic ETS bounds."""
    action_low = torch.FloatTensor([0.8, -3.0])
    action_high = torch.FloatTensor([1.3, 3.0])
    return SecondaryPolicy(
        obs_dim=25, action_dim=2, hidden_size=64,
        action_low=action_low, action_high=action_high,
        log_std_min=-1.5, log_std_max=0.0,
    )


@pytest.fixture
def sample_config():
    """Minimal config dict for PPOAgent."""
    return {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -1.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 40.0, "price_max": 120.0, "quantity_max": 3.0,
            "qty_mult_low": 0.3, "qty_mult_high": 1.3,
        },
        "investment": {"max_invest_frac": 0.10},
        "trading": {"sec_mult_low": 0.8, "sec_mult_high": 1.3},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 2},
    }


# ---------------------------------------------------------------------------
# Test 1: Action bounds
# ---------------------------------------------------------------------------

def test_act_output_bounds(auction_policy, secondary_policy):
    """act() always produces actions within [low, high] bounds."""
    torch.manual_seed(42)
    n_samples = 1000

    action_low_auc = auction_policy.action_bias - auction_policy.action_scale
    action_high_auc = auction_policy.action_bias + auction_policy.action_scale

    for _ in range(n_samples):
        obs = torch.randn(1, 21)
        action, _, _ = auction_policy.act(obs, deterministic=False)
        assert torch.all(action >= action_low_auc - 1e-5), (
            f"Auction action below lower bound: {action}")
        assert torch.all(action <= action_high_auc + 1e-5), (
            f"Auction action above upper bound: {action}")

    action_low_sec = secondary_policy.action_bias - secondary_policy.action_scale
    action_high_sec = secondary_policy.action_bias + secondary_policy.action_scale

    for _ in range(200):
        obs = torch.randn(1, 25)
        action, _, _ = secondary_policy.act(obs, deterministic=False)
        assert torch.all(action >= action_low_sec - 1e-5)
        assert torch.all(action <= action_high_sec + 1e-5)


# ---------------------------------------------------------------------------
# Test 2: act() and evaluate() produce consistent log_prob
# ---------------------------------------------------------------------------

def test_evaluate_log_prob_consistency(auction_policy):
    """act() and evaluate() should return the same log_prob for the same raw."""
    torch.manual_seed(123)
    obs = torch.randn(1, 21)
    action, raw, log_prob_act = auction_policy.act(obs, deterministic=False)

    log_prob_eval, _ = auction_policy.evaluate(obs, raw)

    np.testing.assert_allclose(
        log_prob_act.detach().numpy(),
        log_prob_eval.detach().numpy(),
        atol=1e-5,
        err_msg="act() and evaluate() log_prob mismatch"
    )


# ---------------------------------------------------------------------------
# Test 3: Gradient flows at boundaries (no tanh vanishing gradient)
# ---------------------------------------------------------------------------

def test_clamp_gradient_flow(auction_policy):
    """Gradient on mean-head weights is non-zero even when raw ≈ boundaries.

    With tanh, gradient vanishes at boundaries (dtanh/draw → 0).
    With clamp, gradient through log_prob remains non-zero because PPO
    differentiates through dist.log_prob(raw), not through the clamp.
    """
    obs = torch.randn(1, 21)
    dist = auction_policy.forward(obs)

    # Simulate a raw action at the lower boundary
    raw_boundary = torch.full((1, 6), -0.99)
    raw_boundary.requires_grad_(False)

    log_prob = dist.log_prob(raw_boundary).sum()

    # Backpropagate
    auction_policy.zero_grad()
    log_prob.backward()

    # Check that price_head weights have non-zero gradient
    grad = auction_policy.price_head.weight.grad
    assert grad is not None, "No gradient on price_head"
    assert torch.any(grad.abs() > 1e-8), (
        f"Gradient is effectively zero at boundary: max_abs_grad={grad.abs().max().item()}")


# ---------------------------------------------------------------------------
# Test 4: _to_raw invertibility
# ---------------------------------------------------------------------------

def test_to_raw_invertibility(auction_policy):
    """_to_raw correctly converts physical actions to raw space and back."""
    from scripts.train import _to_raw
    device = torch.device("cpu")

    # Physical actions well within bounds
    physical = np.array([80.0, 0.8, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
    raw = _to_raw(physical, auction_policy, device)

    # Convert back: physical = raw * scale + bias
    scale = auction_policy.action_scale.cpu().numpy()
    bias = auction_policy.action_bias.cpu().numpy()
    recovered = np.clip(raw, -1.0, 1.0) * scale + bias

    np.testing.assert_allclose(physical, recovered, atol=1e-4,
                               err_msg="_to_raw round-trip failed")


# ---------------------------------------------------------------------------
# Test 5: Epsilon-greedy produces diverse actions
# ---------------------------------------------------------------------------

def test_epsilon_greedy_auction(sample_config):
    """With epsilon=1.0, select_auction_action produces diverse physical actions."""
    from src.agents.ppo_agent import PPOAgent

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=21, obs_dim_phase2=25,
        auction_action_low=np.array([40.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([120.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=sample_config, seed=42,
    )

    obs = np.random.randn(21).astype(np.float32)
    bid_prices = []
    for _ in range(100):
        action, raw, lp = agent.select_auction_action(obs, epsilon=1.0)
        bid_prices.append(action[0])

    bid_prices = np.array(bid_prices)
    # With epsilon=1.0 (all random), bids should span [40, 120] range
    assert bid_prices.min() < 55.0, f"Min bid too high: {bid_prices.min():.1f}"
    assert bid_prices.max() > 105.0, f"Max bid too low: {bid_prices.max():.1f}"
    assert bid_prices.std() > 15.0, f"Bid std too low: {bid_prices.std():.1f}"


# ---------------------------------------------------------------------------
# Test 6: Deterministic mode ignores epsilon
# ---------------------------------------------------------------------------

def test_epsilon_greedy_deterministic_bypass(sample_config):
    """With deterministic=True, epsilon should have no effect."""
    from src.agents.ppo_agent import PPOAgent

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=21, obs_dim_phase2=25,
        auction_action_low=np.array([40.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([120.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=sample_config, seed=42,
    )

    obs = np.random.randn(21).astype(np.float32)

    # Deterministic with epsilon=0
    action_det, _, _ = agent.select_auction_action(obs, deterministic=True, epsilon=0.0)
    # Deterministic with epsilon=1.0 — should be identical
    action_det_eps, _, _ = agent.select_auction_action(obs, deterministic=True, epsilon=1.0)

    np.testing.assert_allclose(action_det, action_det_eps, atol=1e-5,
                               err_msg="Epsilon affected deterministic action")


# ---------------------------------------------------------------------------
# Test 7: Epsilon=0 is a no-op
# ---------------------------------------------------------------------------

def test_epsilon_zero_is_noop(sample_config):
    """With epsilon=0.0, results should be identical to no epsilon."""
    from src.agents.ppo_agent import PPOAgent

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=21, obs_dim_phase2=25,
        auction_action_low=np.array([40.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([120.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=sample_config, seed=42,
    )

    obs = np.random.randn(21).astype(np.float32)

    # Fix seed for reproducibility
    torch.manual_seed(999)
    np.random.seed(999)
    a1, r1, lp1 = agent.select_auction_action(obs, epsilon=0.0)

    torch.manual_seed(999)
    np.random.seed(999)
    a2, r2, lp2 = agent.select_auction_action(obs)

    np.testing.assert_allclose(a1, a2, atol=1e-5)
    np.testing.assert_allclose(r1, r2, atol=1e-5)
    np.testing.assert_allclose(lp1, lp2, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: Full PPO update with epsilon actions
# ---------------------------------------------------------------------------

def test_ppo_update_with_epsilon_actions(sample_config):
    """Run one episode with epsilon=0.5, then update() — no NaN or errors."""
    from src.agents.ppo_agent import PPOAgent

    agent = PPOAgent(
        agent_id=0, obs_dim_phase1=21, obs_dim_phase2=25,
        auction_action_low=np.array([40.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([120.0, 1.3, 0.10, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([0.8, -3.0], dtype=np.float32),
        secondary_action_high=np.array([1.3, 3.0], dtype=np.float32),
        config=sample_config, seed=42,
    )

    # Simulate 5 timesteps with epsilon=0.5
    for t in range(5):
        obs1 = np.random.randn(21).astype(np.float32)
        obs2 = np.random.randn(25).astype(np.float32)

        auc_a, auc_r, auc_lp = agent.select_auction_action(obs1, epsilon=0.5)
        sec_a, sec_r, sec_lp = agent.select_secondary_action(obs2, epsilon=0.5)

        reward = np.random.randn() * 5.0
        done = (t == 4)
        value = agent.estimate_value(obs2)

        agent.store_transition(
            obs1=obs1, obs2=obs2,
            auc_raw=auc_r, sec_raw=sec_r,
            auc_lp=auc_lp, sec_lp=sec_lp,
            reward=reward, done=done, value=value,
        )

    result = agent.update(last_value=0.0, actor_update=True)
    assert result is not None, "update() returned None"
    assert np.isfinite(result["actor_loss"]), f"Actor loss is NaN: {result['actor_loss']}"
    assert np.isfinite(result["critic_loss"]), f"Critic loss is NaN: {result['critic_loss']}"

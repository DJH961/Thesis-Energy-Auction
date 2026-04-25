"""
tests/test_environment.py
=========================
Integration tests for the ETSEnvironment.

Run with:
    pytest tests/test_environment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml
from src.environment.ets_environment import ETSEnvironment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")

def load_env(seed=42):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return ETSEnvironment(config, seed=seed)


# ---------------------------------------------------------------------------
# Test 1: Reset returns correct observation shape
# ---------------------------------------------------------------------------

def test_reset_obs_shape():
    env = load_env()
    obs, info = env.reset()
    assert obs.shape == (4, 8), f"Expected (4, 8), got {obs.shape}"


# ---------------------------------------------------------------------------
# Test 2: Step with random actions runs without error
# ---------------------------------------------------------------------------

def test_step_runs():
    env = load_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, rewards, terminated, truncated, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert len(rewards) == 4


# ---------------------------------------------------------------------------
# Test 3: Episode terminates after n_years steps
# ---------------------------------------------------------------------------

def test_episode_length():
    env = load_env()
    obs, _ = env.reset()
    steps = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)
        steps += 1
    assert steps == env.n_years


# ---------------------------------------------------------------------------
# Test 4: Cap decreases each year
# ---------------------------------------------------------------------------

def test_cap_decreases():
    env = load_env()
    env.reset()
    caps = []
    for _ in range(env.n_years):
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        caps.append(info["year_log"]["cap"])
        if terminated:
            break
    for i in range(1, len(caps)):
        assert caps[i] < caps[i - 1], f"Cap did not decrease at step {i}"


# ---------------------------------------------------------------------------
# Test 5: Rewards are finite
# ---------------------------------------------------------------------------

def test_rewards_finite():
    env = load_env()
    env.reset()
    for _ in range(env.n_years):
        action = env.action_space.sample()
        _, rewards, terminated, _, _ = env.step(action)
        assert np.all(np.isfinite(rewards)), f"Non-finite rewards: {rewards}"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 6: Banking accumulates surplus
# ---------------------------------------------------------------------------

def test_banking_accumulates():
    """If a company receives more allowances than it needs, bank should grow."""
    env = load_env()
    env.reset()

    # Force bids much higher than needed so agents always get excess
    # by using max price and max quantity
    aq = env.config["auction"]
    inv = env.config["investment"]
    action = np.tile(
        [aq["price_max"], aq["quantity_max"], 0.0, 0.0],
        (4, 1)
    ).astype(np.float32)

    _, _, _, _, info = env.step(action)
    banks_after_year1 = info["year_log"]["banks"]
    # At least some agents should have banked something
    assert sum(b > 0 for b in banks_after_year1) >= 1


# ---------------------------------------------------------------------------
# Test 7: Reproducibility with same seed
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Same seed → identical episode rewards."""
    rewards_run1 = _collect_episode_rewards(seed=42)
    rewards_run2 = _collect_episode_rewards(seed=42)
    np.testing.assert_array_almost_equal(rewards_run1, rewards_run2)


def _collect_episode_rewards(seed):
    env = load_env(seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    total = np.zeros(4)
    for _ in range(env.n_years):
        action = rng.uniform(
            env.action_space.low,
            env.action_space.high,
        ).astype(np.float32)
        _, rewards, terminated, _, _ = env.step(action)
        total += rewards
        if terminated:
            break
    return total


# ---------------------------------------------------------------------------
# Test 8: Different seeds give different results
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    rewards_42  = _collect_episode_rewards(seed=42)
    rewards_123 = _collect_episode_rewards(seed=123)
    assert not np.allclose(rewards_42, rewards_123), \
        "Different seeds produced identical results — check RNG usage"


# ---------------------------------------------------------------------------
# Test 9: Green fraction stays in [0, 1]
# ---------------------------------------------------------------------------

def test_green_fraction_bounds():
    env = load_env()
    env.reset()
    for _ in range(env.n_years):
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        for gf in info["year_log"]["green_fracs"]:
            assert 0.0 <= gf <= 1.0, f"Green fraction out of bounds: {gf}"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 10: Year log has expected keys
# ---------------------------------------------------------------------------

def test_year_log_keys():
    env = load_env()
    env.reset()
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    log = info["year_log"]
    for key in ["year", "cap", "tnac", "auction_volume", "clearing_price",
                "allocations", "penalties", "rewards", "green_fracs", "banks"]:
        assert key in log, f"Missing key in year_log: {key}"

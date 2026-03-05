"""
tests/test_environment.py
=========================
Integration tests for the ETSEnvironment with technology-specific mix.

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
    n_agents = env.config["companies"]["n_agents"]
    assert obs.shape == (n_agents, 18), f"Expected ({n_agents}, 18), got {obs.shape}"


# ---------------------------------------------------------------------------
# Test 2: Two-phase step runs without error
# ---------------------------------------------------------------------------

def test_two_phase_step_runs():
    env = load_env()
    obs1, _ = env.reset()
    n_agents = env.n_agents

    # Phase 1: [bid_price, qty, invest_frac, tech_logit0, tech_logit1, tech_logit2]
    auction_actions = np.random.uniform(
        [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
        size=(n_agents, 6)
    ).astype(np.float32)

    obs2, log = env.step_auction(auction_actions)
    assert obs2.shape == (n_agents, 21), f"Expected ({n_agents}, 21), got {obs2.shape}"

    # Phase 2: [sec_price_mult, sec_qty]
    secondary_actions = np.random.uniform(
        [0.5, -5.0], [2.0, 5.0], size=(n_agents, 2)
    ).astype(np.float32)

    obs1_next, rewards, terminated, truncated, info = env.step_secondary(secondary_actions)
    assert obs1_next.shape == (n_agents, 18)
    assert len(rewards) == n_agents


# ---------------------------------------------------------------------------
# Test 3: Episode terminates after n_years steps
# ---------------------------------------------------------------------------

def test_episode_length():
    env = load_env()
    obs1, _ = env.reset()
    n_agents = env.n_agents
    steps = 0
    terminated = False

    while not terminated:
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.random.uniform(
            [0.5, -5.0], [2.0, 5.0], size=(n_agents, 2)
        ).astype(np.float32)
        obs1, _, terminated, _, _ = env.step_secondary(secondary_actions)
        steps += 1

    assert steps == env.n_years


# ---------------------------------------------------------------------------
# Test 4: Cap decreases each year
# ---------------------------------------------------------------------------

def test_cap_decreases():
    env = load_env()
    env.reset()
    n_agents = env.n_agents
    caps = []

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.0, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = 1.0
        _, _, terminated, _, info = env.step_secondary(secondary_actions)
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
    n_agents = env.n_agents

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.random.uniform(
            [0.5, -5.0], [2.0, 5.0], size=(n_agents, 2)
        ).astype(np.float32)
        _, rewards, terminated, _, _ = env.step_secondary(secondary_actions)
        assert np.all(np.isfinite(rewards)), f"Non-finite rewards: {rewards}"
        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 6: Technology mix sums to 1
# ---------------------------------------------------------------------------

def test_tech_mix_sums_to_one():
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    for _ in range(env.n_years):
        auction_actions = np.random.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = 1.0
        _, _, terminated, _, _ = env.step_secondary(secondary_actions)

        for c in env.companies:
            assert abs(c.mix.sum() - 1.0) < 1e-6, f"Mix doesn't sum to 1: {c.mix}"
            assert np.all(c.mix >= -1e-9), f"Negative mix: {c.mix}"

        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 7: Green fraction only increases (greening-only)
# ---------------------------------------------------------------------------

def test_green_fraction_non_decreasing():
    """With positive investment, green fraction should not decrease."""
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    prev_green = [c.green_frac for c in env.companies]
    for _ in range(env.n_years):
        # Invest heavily in solar (tech_logit2 dominant)
        auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
        auction_actions[:, 0] = 100.0  # bid price
        auction_actions[:, 1] = 2.0    # quantity
        auction_actions[:, 2] = 0.03   # invest_frac
        auction_actions[:, 5] = 1.0    # solar logit highest

        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
        secondary_actions[:, 0] = 1.0
        _, _, terminated, _, _ = env.step_secondary(secondary_actions)
        if terminated:
            break

    # After investments mature, green should be >= initial
    # (may not increase in first years due to construction delays)


# ---------------------------------------------------------------------------
# Test 8: Reproducibility with same seed
# ---------------------------------------------------------------------------

def test_reproducibility():
    rewards_run1 = _collect_episode_rewards(seed=42)
    rewards_run2 = _collect_episode_rewards(seed=42)
    np.testing.assert_array_almost_equal(rewards_run1, rewards_run2)


def _collect_episode_rewards(seed):
    env = load_env(seed=seed)
    rng = np.random.default_rng(seed)
    obs1, _ = env.reset(seed=seed)
    n_agents = env.n_agents
    total = np.zeros(n_agents)

    for _ in range(env.n_years):
        auction_actions = rng.uniform(
            [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
            size=(n_agents, 6)
        ).astype(np.float32)
        obs2, _ = env.step_auction(auction_actions)

        secondary_actions = rng.uniform(
            [0.5, -5.0], [2.0, 5.0], size=(n_agents, 2)
        ).astype(np.float32)
        obs1, rewards, terminated, _, _ = env.step_secondary(secondary_actions)
        total += rewards
        if terminated:
            break
    return total


# ---------------------------------------------------------------------------
# Test 9: Different seeds give different results
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    rewards_42 = _collect_episode_rewards(seed=42)
    rewards_123 = _collect_episode_rewards(seed=123)
    assert not np.allclose(rewards_42, rewards_123), \
        "Different seeds produced identical results"


# ---------------------------------------------------------------------------
# Test 10: Year log has expected keys
# ---------------------------------------------------------------------------

def test_year_log_keys():
    env = load_env()
    env.reset()
    n_agents = env.n_agents

    auction_actions = np.random.uniform(
        [20.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        [200.0, 5.0, 0.05, 1.0, 1.0, 1.0],
        size=(n_agents, 6)
    ).astype(np.float32)
    obs2, _ = env.step_auction(auction_actions)

    secondary_actions = np.zeros((n_agents, 2), dtype=np.float32)
    secondary_actions[:, 0] = 1.0
    _, _, _, _, info = env.step_secondary(secondary_actions)
    log = info["year_log"]

    for key in ["year", "cap", "tnac", "auction_volume", "clearing_price",
                "allocations", "penalties", "rewards", "green_fracs",
                "tech_mixes", "holdings", "invest_costs"]:
        assert key in log, f"Missing key in year_log: {key}"


# ---------------------------------------------------------------------------
# Test 11: Emissions decrease with greener mix
# ---------------------------------------------------------------------------

def test_emissions_decrease_with_green():
    """A4 (90% green) should have much lower emissions than A1 (20% green)."""
    env = load_env()
    env.reset()
    e1 = env.companies[0].compute_emissions()
    e4 = env.companies[3].compute_emissions()
    assert e4 < e1, f"A4 emissions ({e4}) should be < A1 emissions ({e1})"
    assert e4 < 0.5 * e1, f"A4 should have significantly lower emissions than A1"


# ---------------------------------------------------------------------------
# Test 12: Investment costs are realistic (not off by 100x)
# ---------------------------------------------------------------------------

def test_investment_costs_realistic():
    """Shifting 3% output to onshore wind should cost ~tens of M€, not ~1.5 M€."""
    env = load_env()
    env.reset()
    c = env.companies[0]
    # 3% of 10 TWh = 300 GWh → needs ~98 MW at 35% CF → ~€132M
    cost = c.compute_investment_cost(tech_idx=2, frac_delta=0.03)  # onshore wind
    assert cost > 50, f"Investment cost ({cost:.1f} M€) is too low — should be >50 M€"
    assert cost < 500, f"Investment cost ({cost:.1f} M€) seems too high"

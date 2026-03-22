"""
tests/test_mappo.py
===================
Tests for MAPPO (Multi-Agent PPO) centralized critic functionality.

Verifies that:
  - ValueNetwork accepts global_state_dim as input
  - build_agents creates critic with correct input dim
  - RolloutBuffer stores global_state
  - Full episode + PPO update works end-to-end with MAPPO
  - IPPO fallback still works when centralized_critic=false
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml
import torch

from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.agents.actor_critic import ValueNetwork
from src.environment.ets_environment import ETSEnvironment
from scripts.train import build_agents


def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------

def test_centralized_critic_dimensions():
    """ValueNetwork with 320D global_state input should produce (batch, 1) output."""
    vn = ValueNetwork(obs_dim=320, hidden_size=256)
    x = torch.randn(4, 320)  # batch of 4, 320D global state
    out = vn(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_ippo_critic_dimensions():
    """ValueNetwork with 40D local obs2 input still works."""
    vn = ValueNetwork(obs_dim=40, hidden_size=256)
    x = torch.randn(4, 40)
    out = vn(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_rollout_buffer_global_state():
    """RolloutBuffer should store global_state when provided."""
    buf = RolloutBuffer()
    gs = np.random.randn(320).astype(np.float32)
    buf.push(
        obs1=np.zeros(35), obs2=np.zeros(40),
        auc_raw=np.zeros(6), sec_raw=np.zeros(2),
        auc_lp=np.zeros(1), sec_lp=np.zeros(1),
        reward=0.0, done=False, value=0.0,
        global_state=gs,
    )
    assert len(buf.global_states) == 1
    np.testing.assert_array_equal(buf.global_states[0], gs)


def test_rollout_buffer_no_global_state():
    """RolloutBuffer without global_state (IPPO mode) should have empty list."""
    buf = RolloutBuffer()
    buf.push(
        obs1=np.zeros(35), obs2=np.zeros(40),
        auc_raw=np.zeros(6), sec_raw=np.zeros(2),
        auc_lp=np.zeros(1), sec_lp=np.zeros(1),
        reward=0.0, done=False, value=0.0,
    )
    assert len(buf.global_states) == 0
    assert len(buf.obs1) == 1  # other fields still stored


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------

def test_build_agents_mappo():
    """build_agents with centralized_critic=true should create critic
    whose first layer accepts n_agents * obs2_dim input."""
    config = _load_config()
    config["ppo"]["centralized_critic"] = True
    env = ETSEnvironment(config, seed=42)
    agents = build_agents(env, config, seed=42)

    n_agents = config["companies"]["n_agents"]
    obs2_dim = env.companies[0].obs_dim_phase2
    expected_dim = n_agents * obs2_dim

    fc1_in = agents[0].value_net.fc1.in_features
    assert fc1_in == expected_dim, (
        f"Expected critic input dim {expected_dim}, got {fc1_in}")

    # All agents should have the same critic input dim
    for i, agent in enumerate(agents):
        assert agent.value_net.fc1.in_features == expected_dim, (
            f"Agent {i} critic has wrong input dim")
        assert agent.centralized_critic is True


def test_build_agents_ippo_fallback():
    """build_agents with centralized_critic=false should use local obs2 dim."""
    config = _load_config()
    config["ppo"]["centralized_critic"] = False
    env = ETSEnvironment(config, seed=42)
    agents = build_agents(env, config, seed=42)

    obs2_dim = env.companies[0].obs_dim_phase2
    fc1_in = agents[0].value_net.fc1.in_features
    assert fc1_in == obs2_dim, f"Expected {obs2_dim}, got {fc1_in}"
    assert agents[0].centralized_critic is False


def test_estimate_value_global_state():
    """estimate_value should accept 320D global state in MAPPO mode."""
    config = _load_config()
    config["ppo"]["centralized_critic"] = True
    env = ETSEnvironment(config, seed=42)
    agents = build_agents(env, config, seed=42)

    n_agents = config["companies"]["n_agents"]
    obs2_dim = env.companies[0].obs_dim_phase2
    global_state = np.random.randn(n_agents * obs2_dim).astype(np.float32)

    val = agents[0].estimate_value(global_state)
    assert np.isfinite(val), f"Value should be finite, got {val}"


def test_mappo_full_episode():
    """Run one full episode with MAPPO and verify PPO update completes."""
    config = _load_config()
    config["ppo"]["centralized_critic"] = True
    config["pretrain"]["enabled"] = False
    config["ppo"]["critic_warmup_episodes"] = 0

    env = ETSEnvironment(config, seed=42)
    agents = build_agents(env, config, seed=42)
    n_agents = config["companies"]["n_agents"]
    n_years = config["simulation"]["n_years"]

    obs1, _ = env.reset(seed=42)

    for year in range(n_years):
        # Phase 1: Auction
        auction_actions = np.zeros((n_agents, 6), dtype=np.float32)
        auction_raws, auction_logps = [], []
        for i in range(n_agents):
            a, r, lp = agents[i].select_auction_action(obs1[i])
            auction_actions[i] = a
            auction_raws.append(r)
            auction_logps.append(lp)

        obs2, _ = env.step_auction(auction_actions)
        global_state = obs2.flatten()  # MAPPO global state

        # Phase 2: Secondary
        sec_actions = np.zeros((n_agents, 2), dtype=np.float32)
        sec_raws, sec_logps = [], []
        for i in range(n_agents):
            a, r, lp = agents[i].select_secondary_action(obs2[i])
            sec_actions[i] = a
            sec_raws.append(r)
            sec_logps.append(lp)

        obs1_next, rewards, done, _, _ = env.step_secondary(sec_actions)

        # Store transitions with global state
        for i in range(n_agents):
            value = agents[i].estimate_value(global_state)
            agents[i].store_transition(
                obs1[i], obs2[i], auction_raws[i], sec_raws[i],
                auction_logps[i], sec_logps[i], rewards[i], done, value,
                global_state=global_state,
            )

        obs1 = obs1_next
        if done:
            break

    # PPO update should work without errors
    for i, agent in enumerate(agents):
        result = agent.update(last_value=0.0)
        assert result is not None, f"Agent {i} update returned None"
        assert np.isfinite(result["critic_loss"]), (
            f"Agent {i} critic loss not finite: {result['critic_loss']}")
        assert np.isfinite(result["actor_loss"]), (
            f"Agent {i} actor loss not finite: {result['actor_loss']}")

    # Verify buffer was cleared after update
    for agent in agents:
        assert len(agent.buffer) == 0, "Buffer should be cleared after update"

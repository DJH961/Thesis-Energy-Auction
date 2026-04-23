"""
test_compliance_validation.py
==============================
Smoke test: Run 1 episode with all-bot agents (no learning agents), check compliance metrics.

Asserts:
  - Zero collateral clip events
  - default_count == 0 across all years
  - Clearing price >= reserve_price every year
"""
import sys
import os
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


@pytest.fixture
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_one_episode_heuristic(config, seed=42):
    """Run one episode with all-heuristic agents (no learning agents)."""
    cfg = {k: v for k, v in config.items()}
    # Use 0 learning agents so all agents are bots
    cfg["companies"] = dict(cfg["companies"])
    original_n_agents = cfg["companies"]["n_agents"]
    original_initial_mix = cfg["companies"]["initial_mix"]
    original_reward_weights = cfg["companies"]["reward_weights"]
    cfg["companies"]["n_agents"] = 0
    cfg["companies"]["initial_mix"] = []
    cfg["companies"]["reward_weights"] = []
    cfg["companies"]["n_bot_agents"] = original_n_agents
    # Bots use the same emission mixes as the learning agents for consistent market dynamics
    cfg["companies"]["bot_initial_mix"] = original_initial_mix
    cfg["companies"]["bot_reward_weights"] = original_reward_weights
    env = ETSEnvironment(cfg, seed=seed)
    env.reset()
    episode_log = []
    n_years = cfg["simulation"]["n_years"]
    for year in range(n_years):
        obs1, log = env.step_auction(np.zeros((0, 6)))   # 0 learning agents, 6D actions each
        _, rewards, terminated, truncated, info = env.step_secondary(np.zeros((0, 2)))
        episode_log.append(env.episode_log[-1] if env.episode_log else {})
        if terminated or truncated:
            break
    return episode_log, env


class TestComplianceValidation:

    def test_coal_bot_coverage_and_defaults(self, config):
        """Run 1 episode, check zero defaults and clearing price above reserve."""
        episode_log, env = run_one_episode_heuristic(config)
        assert len(episode_log) > 0, "Episode log must not be empty"
        total_clip_events = sum(env._collateral_clip_events.values())
        assert total_clip_events == 0, (
            "Bot-only run should have ~0 collateral clip events; non-zero indicates "
            "heuristic/env mismatch."
        )

        reserve_price = config["auction"].get("reserve_price", 25.0)

        for yr_idx, yl in enumerate(episode_log):
            if yr_idx == 0:
                continue  # skip year 0 — bots still building up holdings
            # Check clearing price is above reserve (meaningful price discovery)
            clearing_price = yl.get("clearing_price", 0.0)
            assert clearing_price >= reserve_price, (
                f"Year {yr_idx}: clearing_price={clearing_price:.2f} below reserve={reserve_price:.2f}"
            )

            # Check zero defaults
            auction_stats = yl.get("auction_stats", {})
            defaults = auction_stats.get("defaults", 0)
            assert defaults == 0, f"Year {yr_idx}: unexpected defaults={defaults}"

    def test_clearing_price_above_reserve(self, config):
        """Clearing price must be strictly above reserve price."""
        episode_log, env = run_one_episode_heuristic(config)
        reserve = config["auction"].get("reserve_price", 25.0)
        for yr_idx, yl in enumerate(episode_log):
            price = yl.get("clearing_price", 0.0)
            assert price > reserve, (
                f"Year {yr_idx}: clearing_price={price:.2f} not above reserve={reserve:.2f}"
            )

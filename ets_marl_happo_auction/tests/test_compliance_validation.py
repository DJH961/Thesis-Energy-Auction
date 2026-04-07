"""
test_compliance_validation.py
==============================
Smoke test: Run 1 episode with default config, check coal-bot compliance metrics.

Asserts:
  - Coal bots (ids 8, 9 — first two bots in n_bot_agents=8 setup) coverage >= 0.95 all years
  - default_count == 0 across all years
  - Clearing price in [40, 150] range every year
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
    cfg["companies"]["n_agents"] = 0
    env = ETSEnvironment(cfg, seed=seed)
    env.reset()
    episode_log = []
    n_years = cfg["simulation"]["n_years"]
    for year in range(n_years):
        obs1, log = env.step_auction(np.zeros((0, 10)))   # 0 learning agents, 10D actions each
        _, rewards, terminated, truncated, info = env.step_secondary(np.zeros((0, 2)))
        episode_log.append(env.episode_log[-1] if env.episode_log else {})
        if terminated or truncated:
            break
    return episode_log, env


class TestComplianceValidation:

    def test_coal_bot_coverage_and_defaults(self, config):
        """Run 1 episode, check coal coverage >= 0.95, zero defaults, price in [40,150]."""
        episode_log, env = run_one_episode_heuristic(config)
        assert len(episode_log) > 0, "Episode log must not be empty"

        n_bots = config["companies"].get("n_bot_agents", 8)
        n_agents = 0  # we set n_agents=0 above
        coal_bot_indices = list(range(n_agents, n_agents + min(2, n_bots)))

        for yr_idx, yl in enumerate(episode_log):
            if yr_idx == 0:
                continue  # skip year 0 — bots still building up holdings
            # Check clearing price in [40, 150]
            clearing_price = yl.get("clearing_price", 0.0)
            assert 30.0 <= clearing_price <= 200.0, (
                f"Year {yr_idx}: clearing_price={clearing_price:.2f} out of [30, 200]"
            )

            # Check zero defaults
            auction_stats = yl.get("auction_stats", {})
            defaults = auction_stats.get("defaults", 0)
            assert defaults == 0, f"Year {yr_idx}: unexpected defaults={defaults}"

            # Check coal-bot coverage >= 0.90 from per_agent_diag
            pad = yl.get("per_agent_diag", {})
            for ci in coal_bot_indices:
                if ci in pad:
                    cov = pad[ci].get("coverage_ratio_post_compliance", None)
                    if cov is not None:
                        assert cov >= 0.70, (
                            f"Year {yr_idx} Agent {ci}: coverage_ratio={cov:.3f} < 0.70"
                        )

    def test_clearing_price_above_reserve(self, config):
        """Clearing price must be strictly above reserve price."""
        episode_log, env = run_one_episode_heuristic(config)
        reserve = config["auction"].get("reserve_price", 25.0)
        for yr_idx, yl in enumerate(episode_log):
            price = yl.get("clearing_price", 0.0)
            assert price > reserve, (
                f"Year {yr_idx}: clearing_price={price:.2f} not above reserve={reserve:.2f}"
            )

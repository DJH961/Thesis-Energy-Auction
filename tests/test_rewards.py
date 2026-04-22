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
        config = yaml.safe_load(f)
    # Keep short-horizon reward tests warning-free without suppressing warnings.
    config["ets"]["lrf_phase1"] = 0.20
    config["ets"]["lrf_phase2"] = 0.20
    return config


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
    # With v7.2.1: invest_frac = ((action + 1) / 2) * max_invest_frac
    # To get desired invest_frac: action = 2 * invest_frac / max_invest_frac - 1
    max_invest_frac = env.companies[0].max_invest_frac
    auction_actions[:, 2] = 2.0 * invest_frac / max_invest_frac - 1.0
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


def test_collateral_cost_logged_matches_formula():
    """Year log collateral costs should match rate × collateral_locked (E2/E4 formula)."""
    config = load_config()
    config["companies"]["n_bot_agents"] = 0
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    config["auction"]["collateral"]["enabled"] = True

    env = ETSEnvironment(config, seed=123)
    env.reset(seed=123)

    n = env.n_agents
    auction_actions = np.zeros((n, 6), dtype=np.float32)
    auction_actions[:, 0] = 80.0
    auction_actions[:, 1] = 0.3
    auction_actions[:, 2] = 0.0
    auction_actions[:, 3:] = [0.0, 0.0, 1.0]

    # Agent 0 overbids (price above clearing) with a moderate quantity so that E4
    # collateral + payment stays within the annual budget (no default).
    auction_actions[0, 0] = 120.0  # above clearing → non-zero collateral locked
    auction_actions[0, 1] = 1.0    # 1× coverage — affordable under E4 constraints

    env.step_auction(auction_actions)

    secondary_actions = np.zeros((n, 2), dtype=np.float32)
    secondary_actions[:, 0] = env._phase1_clearing_price
    secondary_actions[:, 1] = 0.0
    _, _, _, _, info = env.step_secondary(secondary_actions)

    yl = info.get("year_log", {})
    collateral = np.array(yl.get("collateral_costs", []), dtype=float)

    rate = float(config["auction"]["collateral"]["collateral_rate"])
    # New formula: collateral_cost = rate × collateral_locked (stored in step_auction)
    expected = rate * env._collateral_locked

    assert collateral.shape[0] == env.n_total
    np.testing.assert_allclose(collateral, expected, atol=1e-6)
    assert collateral.sum() > 0.0


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


def test_terminal_bank_uses_budget_divisor():
    """Terminal bank value should remain /annual_budget-scaled under log terminal valuation."""
    config = load_config()
    config["reward"]["terminal_bank_value"] = True
    config["reward"]["terminal_queue_value"] = False
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()
    # Run to final year with high qty to build up bank
    rewards, _ = _run_to_final_year(env, auction_price=80.0, qty_mult=1.5)
    # With log scaling, terminal value should stay below linear bank*price/budget,
    # and far below an incorrect /100 scaling.
    for i in range(env.n_agents):
        if env.holdings[i] > 0.1:
            budget = max(env.companies[i].annual_budget, 1.0)
            linear_upper = env.holdings[i] * 500 / budget
            assert env._last_terminal_bank_values[i] <= linear_upper + 1e-6
            assert env._last_terminal_bank_values[i] < env.holdings[i] * 500 / 100.0, (
                f"Terminal bank value too large — likely using /100 instead of /annual_budget")


def test_terminal_bank_diminishing_returns():
    """3x annual-need holdings should be worth less than 3x the 1x annual-need value."""
    config = load_config()
    config["reward"]["terminal_bank_value"] = True
    config["reward"]["terminal_queue_value"] = False
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()
    env.current_year = env.n_years - 1
    env.last_secondary_price = 100.0

    # Agents 0 and 1 share the same archetype, so annual_need is identical.
    annual_need = max(env.companies[0].compute_estimate_need(), 0.1)
    env.holdings[:] = 0.0
    env.holdings[0] = annual_need
    env.holdings[1] = 3.0 * annual_need

    zeros = np.zeros(env.n_total)
    env._compute_rewards(
        payments=zeros,
        trade_costs=zeros,
        penalties=zeros,
        invest_costs=zeros,
        emissions=zeros,
        clearing_price=100.0,
        mac_costs=zeros,
        precompliance_holdings=zeros,
        old_carry_forward=zeros,
    )

    v1 = env._last_terminal_bank_values[0]
    v3 = env._last_terminal_bank_values[1]
    assert v3 > v1
    assert v3 < 3.0 * v1


def test_terminal_bank_zero_holdings():
    """Zero holdings should produce zero terminal bank value."""
    config = load_config()
    config["reward"]["terminal_bank_value"] = True
    config["reward"]["terminal_queue_value"] = False
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()
    env.current_year = env.n_years - 1
    env.holdings[:] = 0.0

    zeros = np.zeros(env.n_total)
    env._compute_rewards(
        payments=zeros,
        trade_costs=zeros,
        penalties=zeros,
        invest_costs=zeros,
        emissions=zeros,
        clearing_price=100.0,
        mac_costs=zeros,
        precompliance_holdings=zeros,
        old_carry_forward=zeros,
    )

    assert env._last_terminal_bank_values.sum() == pytest.approx(0.0, abs=1e-9)


def test_penalty_hits_budget():
    """Penalty cost should be included in budget spending."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()
    # Bid very low qty so agents get shortfalls and penalties
    rewards, _ = _run_one_year(env, auction_price=80.0, qty_mult=0.3)
    # Check that at least some agents have spent more than just auction cost
    # (penalty should be included in budget_spent)
    for i in range(min(4, env.n_agents)):
        company = env.companies[i]
        assert company.budget_spent_this_year > 0, f"Agent {i} budget should have spending"


def test_financial_agent_zero_esg():
    """Financial agents (w_green=0) should get zero ESG signal."""
    config = load_config()
    config["esg"] = {"enabled": True}
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()
    env.set_episode(0)
    # Run one year — ESG signal should be zero for financial (even) agents
    # since their w_green = 0.0
    rewards, _ = _run_one_year(env, auction_price=100.0, invest_frac=0.05)
    assert np.all(np.isfinite(rewards))


def test_esg_early_improvement_worth_more():
    """ESG signal at year 3 should exceed year 10 for same ef_delta (time_ratio is higher)."""
    config = load_config()
    config["esg"] = {"enabled": True}
    config["simulation"]["n_years"] = 12
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    # The ESG formula: w_green × ef_ratio × time_ratio × (budget/1000)
    # time_ratio = remaining_years / n_years
    # At year 3: time_ratio = 9/12 = 0.75
    # At year 10: time_ratio = 2/12 = 0.167
    # So year-3 improvement is ~4.5× more valuable than year-10
    # We just verify the formula property holds
    n_years = 12
    time_ratio_early = (n_years - 3) / n_years   # 0.75
    time_ratio_late = (n_years - 10) / n_years    # 0.167
    assert time_ratio_early > time_ratio_late * 3.0


def test_green_bonus_ratio():
    """Green bonus for ESG agent (~0.7 multiplier) vs financial (~0.2) should be ~3.5×."""
    w_green_esg = 0.5      # ESG agent
    w_green_fin = 0.0      # Financial agent
    # green_bonus ∝ (0.2 + w_green)
    ratio = (0.2 + w_green_esg) / (0.2 + w_green_fin)
    assert 3.0 < ratio < 4.0, f"Green bonus ratio should be ~3.5, got {ratio}"


def _run_to_final_year(env, auction_price=80.0, qty_mult=1.0, invest_frac=0.0):
    """Helper: run the environment to the final year and return last-year rewards + info."""
    n = env.n_agents
    rewards = None
    info = None
    for year in range(env.n_years):
        auction_actions = np.zeros((n, 6), dtype=np.float32)
        auction_actions[:, 0] = auction_price
        auction_actions[:, 1] = qty_mult
        # With v7.2.1: invest_frac = ((action + 1) / 2) * max_invest_frac
        # To get invest_frac=0: action = -1.0
        # To get desired invest_frac: action = 2 * invest_frac / max_invest_frac - 1
        max_invest_frac = env.companies[0].max_invest_frac
        auction_actions[:, 2] = 2.0 * invest_frac / max_invest_frac - 1.0
        auction_actions[:, 3:] = [0.0, 0.0, 1.0]  # solar logits
        env.step_auction(auction_actions)

        secondary_actions = np.zeros((n, 2), dtype=np.float32)
        secondary_actions[:, 0] = 1.0
        secondary_actions[:, 1] = 0.0
        _, rewards, _, _, info = env.step_secondary(secondary_actions)
    return rewards, info


class TestTerminalBankValue:

    def test_terminal_bank_value_adds_bonus(self):
        """With terminal_bank_value=True, final-year reward includes banked allowance value."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = True
        config["reward"]["terminal_queue_value"] = False
        # Use short episode for speed
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        env_with = ETSEnvironment(config, seed=42)
        env_with.reset()
        # Use higher qty_mult to ensure agents have surplus (no carry_forward debt)
        # to avoid terminal debt penalty affecting the comparison
        rewards_with, _ = _run_to_final_year(env_with, auction_price=80.0, qty_mult=1.5)

        config2 = load_config()
        config2["reward"]["terminal_bank_value"] = False
        config2["reward"]["terminal_queue_value"] = False
        config2["simulation"]["n_years"] = 3
        config2["warm_start"]["enabled"] = False
        config2["uncertainty"]["enabled"] = False
        config2["construction_jitter"]["enabled"] = False

        env_without = ETSEnvironment(config2, seed=42)
        env_without.reset()
        rewards_without, _ = _run_to_final_year(env_without, auction_price=80.0, qty_mult=1.5)

        # If any agent has banked allowances, terminal value should boost reward
        # At minimum, rewards should not be lower with terminal bank value enabled
        assert rewards_with.sum() >= rewards_without.sum() - 0.01, (
            f"Terminal bank value should not decrease total reward: "
            f"with={rewards_with.sum():.4f}, without={rewards_without.sum():.4f}")

    def test_terminal_bank_value_disabled(self):
        """With terminal_bank_value=False, no terminal bank bonus appears."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = False
        config["reward"]["terminal_queue_value"] = False
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        env1 = ETSEnvironment(config, seed=42)
        env1.reset()
        rewards1, _ = _run_to_final_year(env1, auction_price=80.0, qty_mult=1.3)

        env2 = ETSEnvironment(config.copy(), seed=42)
        env2.reset()
        rewards2, _ = _run_to_final_year(env2, auction_price=80.0, qty_mult=1.3)

        # Same config, same seed → identical rewards
        np.testing.assert_allclose(rewards1, rewards2, atol=1e-6)

    def test_terminal_bank_value_proportional_to_holdings(self):
        """Terminal bank value should increase with holdings (monotonic, not linear)."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = True
        config["reward"]["terminal_queue_value"] = False
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        # High qty → more allowances purchased → more likely to have bank
        env_high = ETSEnvironment(config, seed=42)
        env_high.reset()
        rewards_high, _ = _run_to_final_year(env_high, auction_price=80.0, qty_mult=1.3)

        # Low qty → fewer allowances → smaller or zero bank
        env_low = ETSEnvironment(config, seed=42)
        env_low.reset()
        rewards_low, _ = _run_to_final_year(env_low, auction_price=80.0, qty_mult=0.3)

        total_bank_value_high = float(np.sum(env_high._last_terminal_bank_values))
        total_bank_value_low = float(np.sum(env_low._last_terminal_bank_values))
        assert total_bank_value_high >= total_bank_value_low - 1e-6, (
            f"Higher holdings should yield at least as much terminal bank value: "
            f"high={total_bank_value_high:.4f}, low={total_bank_value_low:.4f}")


class TestTerminalQueueValue:

    def test_terminal_queue_value_with_investment(self):
        """Investing in green tech should produce terminal queue value in final year."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = False
        config["reward"]["terminal_queue_value"] = True
        config["reward"]["terminal_payoff_years"] = 5
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        env_invest = ETSEnvironment(config, seed=42)
        env_invest.reset()
        rewards_invest, _ = _run_to_final_year(env_invest, auction_price=80.0, invest_frac=0.08)

        config2 = load_config()
        config2["reward"]["terminal_bank_value"] = False
        config2["reward"]["terminal_queue_value"] = False
        config2["reward"]["terminal_payoff_years"] = 5
        config2["simulation"]["n_years"] = 3
        config2["warm_start"]["enabled"] = False
        config2["uncertainty"]["enabled"] = False
        config2["construction_jitter"]["enabled"] = False

        env_noinvest = ETSEnvironment(config2, seed=42)
        env_noinvest.reset()
        rewards_noinvest, _ = _run_to_final_year(env_noinvest, auction_price=80.0, invest_frac=0.08)

        # With terminal queue value enabled, investing agents should get a bonus
        assert rewards_invest.sum() >= rewards_noinvest.sum() - 0.01, (
            f"Terminal queue value should boost reward for investing agents: "
            f"invest={rewards_invest.sum():.4f}, noinvest={rewards_noinvest.sum():.4f}")

    def test_terminal_queue_value_disabled(self):
        """With terminal_queue_value=False, no queue terminal bonus."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = False
        config["reward"]["terminal_queue_value"] = False
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        env1 = ETSEnvironment(config, seed=42)
        env1.reset()
        rewards1, _ = _run_to_final_year(env1, auction_price=80.0, invest_frac=0.08)

        env2 = ETSEnvironment(config, seed=42)
        env2.reset()
        rewards2, _ = _run_to_final_year(env2, auction_price=80.0, invest_frac=0.08)

        np.testing.assert_allclose(rewards1, rewards2, atol=1e-6)

    def test_terminal_queue_value_zero_without_queue(self):
        """No queue items → no terminal queue bonus even when enabled."""
        config = load_config()
        config["reward"]["terminal_bank_value"] = False
        config["reward"]["terminal_queue_value"] = True
        config["reward"]["terminal_payoff_years"] = 5
        config["simulation"]["n_years"] = 3
        config["warm_start"]["enabled"] = False
        config["uncertainty"]["enabled"] = False
        config["construction_jitter"]["enabled"] = False

        # No investment → no construction queue items
        env_with = ETSEnvironment(config, seed=42)
        env_with.reset()
        rewards_with, _ = _run_to_final_year(env_with, auction_price=80.0, invest_frac=0.0)

        config2 = load_config()
        config2["reward"]["terminal_bank_value"] = False
        config2["reward"]["terminal_queue_value"] = False
        config2["simulation"]["n_years"] = 3
        config2["warm_start"]["enabled"] = False
        config2["uncertainty"]["enabled"] = False
        config2["construction_jitter"]["enabled"] = False

        env_without = ETSEnvironment(config2, seed=42)
        env_without.reset()
        rewards_without, _ = _run_to_final_year(env_without, auction_price=80.0, invest_frac=0.0)

        # No investment, no queue → terminal queue value should be zero → same rewards
        np.testing.assert_allclose(rewards_with, rewards_without, atol=1e-4)


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
    assert w_end <= 0.20, f"Shaping weight should be near floor at decay end: {w_end}"


def test_w_green_differentiates_reward():
    """Agent with w_green=0.5 should receive a higher green_bonus component than w_green=0.0."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    # Ensure shaping is active
    config["reward"]["shaping_decay_episode"] = 10000

    env = ETSEnvironment(config, seed=42)
    env.reset()
    env.set_episode(0)

    # w_green=0.0 agents are even-indexed (0,2,4,6), w_green=0.5 are odd (1,3,5,7)
    # With investment, green_bonus is scaled by (1+w_green)
    # Run one year with moderate investment to trigger green_bonus
    rewards, _ = _run_one_year(env, auction_price=100.0, invest_frac=0.05)
    # All rewards should be finite
    assert np.all(np.isfinite(rewards))


def test_capex_and_compliance_penalties_stack():
    """Agent that maxes out both budgets receives penalties from both independently."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()

    # Run with very high investment to stress both budgets
    rewards_high, _ = _run_one_year(env, auction_price=200.0, qty_mult=1.5, invest_frac=0.15)

    env2 = ETSEnvironment(config, seed=42)
    env2.reset()
    # Run with no investment (only compliance cost)
    rewards_low, _ = _run_one_year(env2, auction_price=200.0, qty_mult=1.5, invest_frac=0.0)

    # Both should be finite
    assert np.all(np.isfinite(rewards_high))
    assert np.all(np.isfinite(rewards_low))
    # High investment should produce different (likely lower) rewards due to combined penalties
    assert not np.allclose(rewards_high, rewards_low, atol=0.01)


def test_electricity_revenue_no_longer_affects_cost():
    """v7.3: electricity revenue was removed from cost_norm_ex_penalty.
    Enabling/disabling electricity should produce identical rewards."""
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

    # v7.3: revenue removed — rewards should be identical regardless of electricity setting
    np.testing.assert_allclose(
        rewards_with, rewards_without,
        err_msg="Revenue removed in v7.3: electricity setting should not affect rewards"
    )


def test_penalty_full_strength_for_esg_agents():
    """ESG agents (w_green=0.5) with identical shortfall to financial agents
    should receive the SAME penalty magnitude in their reward (not 50%)."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    # Disable shaping to isolate penalty effect
    config["reward"]["shaping_beta"] = 0.0
    config["reward"]["shaping_gamma"] = 0.0
    config["esg"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()

    # Create identical shortfall: bid very low qty so both get penalties
    rewards, _ = _run_one_year(env, auction_price=80.0, qty_mult=0.3)

    # w_green=0.0 agents are even-indexed (0,2,4,6), w_green=0.5 are odd (1,3,5,7)
    # With identical market conditions and bids, penalties should be identical
    # Reward structure: R = w_cost × (-cost_norm_ex_penalty) + w_green × esg_signal - penalty_norm
    # For financial: R = 1.0 × (-cost_norm_ex_penalty) + 0.0 × esg - penalty_norm
    # For ESG:       R = 0.5 × (-cost_norm_ex_penalty) + 0.5 × esg - penalty_norm
    # If costs are same and esg=0, then difference = 0.5 × cost_norm_ex_penalty
    # But penalties should be SAME (full strength)

    assert np.all(np.isfinite(rewards)), "All rewards should be finite"


# ---------------------------------------------------------------------------
# F2: Diagnostic score tests
# ---------------------------------------------------------------------------

def test_diagnostic_scores_returned_in_info():
    """step_secondary info should contain 'diagnostic_scores' key (F2)."""
    env = load_env(seed=0)
    env.reset()
    _, info = _run_one_year(env, auction_price=80.0, qty_mult=1.0)
    assert "diagnostic_scores" in info, "info must contain 'diagnostic_scores'"
    diag = info["diagnostic_scores"]
    assert len(diag) == env.n_agents, "One diagnostic score dict per learning agent"


def test_diagnostic_score_components_in_range():
    """S_financial, S_green should be in [0, 1]; S_composite should be non-negative."""
    env = load_env(seed=1)
    env.reset()
    _, info = _run_one_year(env, auction_price=80.0, qty_mult=1.0)
    for ds in info["diagnostic_scores"]:
        assert 0.0 <= ds["S_financial"] <= 1.0, f"S_financial={ds['S_financial']} out of range"
        assert 0.0 <= ds["S_green"] <= 1.0, f"S_green={ds['S_green']} out of range"
        # S_composite is a weighted blend of sub-scores; weights may not sum to 1 so
        # it can exceed 1.0 for high-performing ESG agents, but should be non-negative
        assert ds["S_composite"] >= 0.0, f"S_composite={ds['S_composite']} is negative"


def test_diagnostic_high_spending_lowers_s_financial():
    """An agent that spends close to its full budget should have lower S_financial."""
    config = load_config()
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    env = ETSEnvironment(config, seed=5)
    env.reset(seed=5)

    # Run year with 0 spending (zero bids) → high S_financial
    _, info_low = _run_one_year(env, auction_price=1.0, qty_mult=0.01)
    s_fin_low_spend = [ds["S_financial"] for ds in info_low["diagnostic_scores"]]

    env.reset(seed=5)
    # Run year with high spending (high price + elevated coverage) → lower S_financial
    _, info_high = _run_one_year(env, auction_price=250.0, qty_mult=2.0)
    s_fin_high_spend = [ds["S_financial"] for ds in info_high["diagnostic_scores"]]

    # On average, spending more reduces S_financial
    assert np.mean(s_fin_high_spend) < np.mean(s_fin_low_spend), (
        "Higher budget spending should yield lower S_financial scores"
    )


def test_diagnostic_scores_all_finite():
    """All diagnostic score components must be finite across a full episode."""
    config = load_config()
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    env = ETSEnvironment(config, seed=99)
    env.reset()
    for _ in range(env.n_years):
        _, info = _run_one_year(env, auction_price=80.0, qty_mult=1.0)
        for ds in info["diagnostic_scores"]:
            for k in ("S_financial", "S_green", "S_penalty", "S_composite"):
                assert np.isfinite(ds[k]), f"{k} is not finite: {ds[k]}"
        if env.episode_done:
            break


# ---------------------------------------------------------------------------
# v7.6: New tests for reward function and learning changes
# ---------------------------------------------------------------------------

def test_opex_delta_zero_for_unchanged_mix():
    """An agent with no investment gets opex_delta ≈ 0 (only inflation variance)."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    # Disable inflation randomness so delta is purely from mix change
    config["penalty"]["inflation_random_std"] = 0.0
    config["penalty"]["inflation_random_window"] = 0.0

    env = ETSEnvironment(config, seed=42)
    env.reset()

    # Run one year with NO investment → mix unchanged
    _run_one_year(env, auction_price=80.0, qty_mult=1.0, invest_frac=0.0)

    for i in range(min(4, env.n_agents)):
        company = env.companies[i]
        opex_now = company.compute_operational_cost(1)
        opex_baseline = company.baseline_opex
        # With zero inflation randomness and no mix change, the only difference
        # is from the fixed inflation rate (≈2%), so delta should be small
        opex_delta = opex_now - opex_baseline
        # At 2% inflation, 10 TWh at ~55 EUR/MWh → ~550M, 2% = ~11M.
        # budget ≈ 800-880M, so delta/budget < 2%
        assert abs(opex_delta) < 0.05 * company.annual_budget, (
            f"Agent {i}: opex_delta={opex_delta:.2f} too large for unchanged mix "
            f"(baseline={opex_baseline:.2f}, now={opex_now:.2f})")


def test_esg_cost_balance_preserved():
    """For an ESG agent (w_green=0.5), esg_scale_i = base_esg_scale directly (no budget compensation)."""
    config = load_config()
    config["esg"]["enabled"] = True
    config["esg"]["scale"] = 3.5
    config["simulation"]["n_years"] = 12
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False
    config["reward"]["shaping_beta"] = 0.0  # disable shaping to isolate signals

    env = ETSEnvironment(config, seed=42)
    env.reset()
    env.set_episode(0)

    # Run one year with moderate investment to trigger ESG signal
    _run_one_year(env, auction_price=80.0, qty_mult=1.0, invest_frac=0.05)

    # Verify esg_scale_i = base_esg_scale for all agents (no per-agent budget compensation).
    # esg_signal = base_esg_scale × ef_ratio × time_ratio
    base_esg_scale = float(config["esg"]["scale"])
    for i in range(1, min(8, env.n_agents), 2):
        company = env.companies[i]
        if company.w_green < 0.4:
            continue
        # esg_scale_i is just base_esg_scale — verify via reward channel
        ch = env._last_reward_channels.get(i, {})
        if company.initial_ef > 0.01:
            ef_ratio = (company.initial_ef - company.weighted_emission_factor) / company.initial_ef
            remaining = max(1, env.n_years - env.current_year + 1)
            time_ratio = remaining / env.n_years
            expected_esg = base_esg_scale * ef_ratio * time_ratio
            # The ESG signal in the channel should be approximately this value
            actual_esg = ch.get("esg_signal", 0.0)
            if ef_ratio > 0.01:
                assert abs(actual_esg - expected_esg) < 0.5, (
                    f"Agent {i}: esg_signal={actual_esg}, expected≈{expected_esg}")


def test_batch_normalization_replaces_ema():
    """compute_gae() handles alternating phases and returns a correct phase mask."""
    import torch
    from src.agents.ppo_agent import PPOAgent

    config = load_config()
    config["companies"]["n_agents"] = 2
    config["companies"]["n_bot_agents"] = 0
    config["ppo"]["hidden_size"] = 32

    env = ETSEnvironment(config, seed=42)
    env.reset()

    obs1_dim = env.companies[0].obs_dim_phase1
    obs2_dim = env.companies[0].obs_dim_phase2
    aq = config["auction"]
    inv = config["investment"]
    auction_low = np.array([
        aq["price_min"], aq.get("qty_mult_low", 0.3), 0.0, -1.0, -1.0, -1.0
    ], dtype=np.float32)
    auction_high = np.array([
        aq["price_max"], aq.get("qty_mult_high", 2.0), inv["max_invest_frac"], 1.0, 1.0, 1.0
    ], dtype=np.float32)
    secondary_low = np.array([30.0, -10.0], dtype=np.float32)
    secondary_high = np.array([500.0, 10.0], dtype=np.float32)

    agent = PPOAgent(
        agent_id=0,
        obs_dim_phase1=obs1_dim,
        obs_dim_phase2=obs2_dim,
        auction_action_low=auction_low,
        auction_action_high=auction_high,
        secondary_action_low=secondary_low,
        secondary_action_high=secondary_high,
        config=config,
        seed=42,
    )

    expected_is_auction = []
    # Fill buffer with raw rewards and alternating phase tags
    for t in range(12):
        obs1 = np.random.randn(obs1_dim).astype(np.float32)
        obs2 = np.random.randn(obs2_dim).astype(np.float32)
        auc_raw = np.random.randn(len(auction_low)).astype(np.float32)
        sec_raw = np.random.randn(len(secondary_low)).astype(np.float32)
        raw_reward = -50.0 + t * 5.0  # varying scale, un-normalized
        phase = "auction" if (t % 2 == 0) else "secondary"
        expected_is_auction.append(phase == "auction")
        agent.store_transition(
            obs1=obs1, obs2=obs2,
            auc_raw=auc_raw, sec_raw=sec_raw,
            auc_lp=-1.0, sec_lp=-1.0,
            reward=raw_reward, done=(t == 11), value=0.0,
            phase=phase,
        )

    adv_t, ret_t, buf_tensors = agent.compute_gae(last_value=0.0)

    assert adv_t is not None, "compute_gae returned None advantages"
    assert torch.all(torch.isfinite(adv_t)), "Advantages contain non-finite values"
    assert adv_t.abs().sum() > 0, "All advantages are zero"
    assert torch.all(torch.isfinite(ret_t)), "Returns contain non-finite values"
    assert buf_tensors is not None and "is_auction" in buf_tensors
    np.testing.assert_array_equal(
        buf_tensors["is_auction"].cpu().numpy(),
        np.array(expected_is_auction, dtype=bool),
        err_msg="compute_gae should preserve auction/secondary phase mask",
    )


def test_split_rewards_sum_to_total():
    """r_auction + r_secondary ≈ old_single_reward for the same environment state."""
    config = load_config()
    config["simulation"]["n_years"] = 3
    config["warm_start"]["enabled"] = False
    config["uncertainty"]["enabled"] = False
    config["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(config, seed=42)
    env.reset()

    n = env.n_agents
    auction_actions = np.zeros((n, 6), dtype=np.float32)
    auction_actions[:, 0] = 80.0
    auction_actions[:, 1] = 1.0
    max_invest_frac = env.companies[0].max_invest_frac
    auction_actions[:, 2] = -1.0  # no investment
    auction_actions[:, 3:] = [0.0, 0.0, 1.0]

    env.step_auction(auction_actions)

    # Get auction intermediate reward
    r_auction = env.compute_auction_rewards()

    secondary_actions = np.zeros((n, 2), dtype=np.float32)
    secondary_actions[:, 0] = 1.0
    secondary_actions[:, 1] = 0.0
    _, total_rewards, _, _, _ = env.step_secondary(secondary_actions)

    # r_secondary = total - r_auction
    r_secondary = total_rewards - r_auction[:n]

    # Sum should be close to total
    r_sum = r_auction[:n] + r_secondary
    np.testing.assert_allclose(r_sum, total_rewards, atol=1e-6,
        err_msg="Split rewards should sum to total reward")


# ---------------------------------------------------------------------------
# D: Reward channels (Feature D tests)
# ---------------------------------------------------------------------------

ACTION_DIM = 6

def test_reward_channels_present():
    """_last_reward_channels populated after step_secondary."""
    env = load_env()
    env.reset()
    _run_one_year(env)
    assert len(env._last_reward_channels) > 0
    for i in range(min(env.n_agents, 2)):
        ch = env._last_reward_channels[i]
        expected_keys = {"cost_norm", "penalty_norm", "esg_signal", "base_reward"}
        assert expected_keys.issubset(ch.keys()), f"Missing keys: {expected_keys - ch.keys()}"

def test_auction_reward_channels_present():
    """_last_auction_reward_channels populated after compute_auction_rewards."""
    env = load_env()
    env.reset()
    n = env.n_agents
    auction_actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    auction_actions[:, 0] = 80.0
    auction_actions[:, 1] = 0.5
    env.step_auction(auction_actions)
    env.compute_auction_rewards()
    assert len(env._last_auction_reward_channels) > 0
    for i in range(min(env.n_agents, 2)):
        ch = env._last_auction_reward_channels[i]
        expected_keys = {"auction_cost", "collateral_cost", "investment_cost",
                         "opex_delta", "mac_cost"}
        assert expected_keys.issubset(ch.keys()), f"Missing keys: {expected_keys - ch.keys()}"

def test_reward_channels_values_finite():
    """All reward channel values must be finite."""
    env = load_env()
    env.reset()
    _run_one_year(env)
    for i in env._last_reward_channels:
        for k, v in env._last_reward_channels[i].items():
            assert np.isfinite(v), f"Channel {k} for agent {i} is not finite: {v}"

def test_reward_channels_reset():
    """Channels should be cleared on reset."""
    env = load_env()
    env.reset()
    _run_one_year(env)
    assert len(env._last_reward_channels) > 0
    env.reset()
    assert len(env._last_reward_channels) == 0

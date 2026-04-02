"""Tests for green-finance budget/capex relaxation behavior."""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.company import Company
from src.environment.ets_environment import ETSEnvironment


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _base_env_config():
    cfg = _load_config()
    cfg["companies"]["n_bot_agents"] = 0
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False
    cfg["simulation"]["n_years"] = 1
    return cfg


def _max_invest_actions(n_agents: int):
    actions = np.zeros((n_agents, 6), dtype=np.float32)
    actions[:, 0] = 100.0
    actions[:, 1] = 1.0
    actions[:, 2] = 0.20
    actions[:, 3:] = [0.0, 0.0, 1.0]  # solar
    return actions


def _run_one_year(env: ETSEnvironment):
    n = env.n_agents
    auction_actions = _max_invest_actions(n)
    _, log = env.step_auction(auction_actions)

    sec_actions = np.zeros((n, 2), dtype=np.float32)
    sec_actions[:, 0] = 80.0
    sec_actions[:, 1] = 0.0
    _, rewards, _, _, _ = env.step_secondary(sec_actions)
    return log, rewards


def test_green_finance_increases_investment():
    cfg_off = _base_env_config()
    cfg_off["green_finance"]["enabled"] = False

    cfg_on = _base_env_config()
    cfg_on["green_finance"]["enabled"] = True

    env_off = ETSEnvironment(cfg_off, seed=42)
    env_off.reset(seed=42)
    log_off, _ = _run_one_year(env_off)

    env_on = ETSEnvironment(cfg_on, seed=42)
    env_on.reset(seed=42)
    log_on, _ = _run_one_year(env_on)

    # Coal-heavy agents A1/A2 are typically most constrained.
    off_fracs = np.array(log_off["invest_fracs"][:2], dtype=float)
    on_fracs = np.array(log_on["invest_fracs"][:2], dtype=float)

    assert np.all(on_fracs >= off_fracs - 1e-9)
    assert np.any(on_fracs > off_fracs + 1e-5)


def test_green_finance_loan_cost_in_reward():
    cfg = _base_env_config()
    cfg["green_finance"]["enabled"] = True

    env = ETSEnvironment(cfg, seed=7)
    env.reset(seed=7)

    zeros = np.zeros(env.n_total)
    active_mask = np.ones(env.n_total, dtype=bool)

    c0 = env.companies[0]
    c0.reset_budget()
    c0.reset_capex_budget()
    c0.green_loan_utilized = 0.0
    r_no_loan = env._compute_rewards(
        payments=zeros,
        trade_costs=zeros,
        penalties=zeros,
        invest_costs=zeros,
        emissions=zeros,
        clearing_price=80.0,
        mac_costs=zeros,
        precompliance_holdings=zeros,
        old_carry_forward=zeros,
        active_mask=active_mask,
    )[0]

    c0.reset_budget()
    c0.reset_capex_budget()
    c0.green_loan_utilized = 100.0
    r_with_loan = env._compute_rewards(
        payments=zeros,
        trade_costs=zeros,
        penalties=zeros,
        invest_costs=zeros,
        emissions=zeros,
        clearing_price=80.0,
        mac_costs=zeros,
        precompliance_holdings=zeros,
        old_carry_forward=zeros,
        active_mask=active_mask,
    )[0]

    assert r_with_loan < r_no_loan


def test_green_finance_disabled_no_change():
    cfg_a = _base_env_config()
    cfg_a["green_finance"]["enabled"] = False

    cfg_b = _base_env_config()
    cfg_b["green_finance"]["enabled"] = False
    cfg_b["green_finance"]["loan_budget_boost"] = 999.0
    cfg_b["green_finance"]["capex_throughput_boost"] = 999.0

    env_a = ETSEnvironment(cfg_a, seed=11)
    env_b = ETSEnvironment(cfg_b, seed=11)

    env_a.reset(seed=11)
    env_b.reset(seed=11)

    log_a, rew_a = _run_one_year(env_a)
    log_b, rew_b = _run_one_year(env_b)

    np.testing.assert_allclose(log_a["invest_fracs"], log_b["invest_fracs"], atol=1e-9)
    np.testing.assert_allclose(rew_a, rew_b, atol=1e-9)


def test_green_loan_headroom_tracking():
    cfg = _base_env_config()
    cfg["green_finance"]["enabled"] = True
    rng = np.random.default_rng(42)

    c = Company(
        agent_id=0,
        config=cfg,
        initial_mix=cfg["companies"]["initial_mix"][0],
        rng=rng,
    )

    start_headroom = c.green_loan_headroom
    assert start_headroom == pytest.approx(cfg["green_finance"]["loan_budget_boost"])

    c.record_green_loan(50.0)
    assert c.green_loan_headroom == pytest.approx(start_headroom - 50.0)

    c.reset_budget()
    assert c.green_loan_headroom == pytest.approx(start_headroom)

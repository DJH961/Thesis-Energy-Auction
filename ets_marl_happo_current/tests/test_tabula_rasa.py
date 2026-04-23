"""Tests for tabula-rasa training mode overrides and exploration behavior."""

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import train as train_mod
from src.agents.ppo_agent import PPOAgent


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _capture_config_after_train_overrides(monkeypatch, config: dict):
    """Run train_one_seed until env construction and capture resolved config."""
    captured = {}

    class _StopEnv:
        def __init__(self, cfg, seed=None):
            captured["config"] = copy.deepcopy(cfg)
            raise RuntimeError("_stop_after_override_")

    monkeypatch.setattr(train_mod, "ETSEnvironment", _StopEnv)

    with pytest.raises(RuntimeError, match="_stop_after_override_"):
        train_mod.train_one_seed(config, seed=42)

    assert "config" in captured
    return captured["config"]


def test_tabula_rasa_disables_pretrain(monkeypatch):
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 1000
    cfg["tabula_rasa"]["enabled"] = True

    resolved = _capture_config_after_train_overrides(monkeypatch, cfg)
    assert resolved["pretrain"]["enabled"] is False
    assert resolved["ppo"]["kl_anchor_beta"] == 0.0


def test_tabula_rasa_schedule_overrides(monkeypatch):
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 1000
    cfg["tabula_rasa"]["enabled"] = True

    resolved = _capture_config_after_train_overrides(monkeypatch, cfg)
    n_ep = resolved["simulation"]["n_episodes"]

    assert resolved["ppo"]["critic_warmup_episodes"] == int(0.03 * n_ep)
    assert resolved["reward"]["shaping_decay_episode"] == int(0.60 * n_ep)
    assert resolved["exploration"]["epsilon_decay_episodes"] == int(0.90 * n_ep)
    assert resolved["exploration"]["mode"] == "uniform"


def test_tabula_rasa_no_anchors(monkeypatch):
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 1000
    cfg["companies"]["n_agents"] = 1
    cfg["tabula_rasa"]["enabled"] = True

    resolved = _capture_config_after_train_overrides(monkeypatch, cfg)

    class _DummyCompany:
        obs_dim_phase1 = 22
        obs_dim_phase2 = 29

    class _DummyEnv:
        companies = [_DummyCompany()]

    agents = train_mod.build_agents(_DummyEnv(), resolved, seed=1)
    agent = agents[0]

    obs = torch.zeros(1, 22, device=next(agent.auction_policy.parameters()).device)
    with torch.no_grad():
        action, _, _ = agent.auction_policy.act(obs, deterministic=True)

    price = float(action[0, 0].item())
    # With no explicit anchors, tabula-rasa now starts near expected price
    # rather than the midpoint of the full auction range.
    assert abs(price - 80.0) < 50.0
    assert abs(price - 265.0) > 50.0


def test_tabula_rasa_uniform_exploration():
    config = {
        "ppo": {
            "hidden_size": 64,
            "lr": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.02,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 2,
            "mini_batch_size": 4,
            "log_std_min": -1.5,
            "log_std_max": 0.0,
            "centralized_critic": False,
            "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 30.0,
            "price_max": 500.0,
            "quantity_max": 3.0,
            "qty_mult_low": 0.3,
            "qty_mult_high": 2.0,
        },
        "investment": {"max_invest_frac": 0.2},
        "trading": {"sec_price_min": 30.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02},
        "simulation": {"n_years": 12},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 1},
        "exploration": {
            "mode": "uniform",
            "auction_anchors": [80.0, 1.0, 0.03, 0.3, -0.5, 0.5],
            "secondary_anchors": [80.0, 0.0],
        },
    }

    agent = PPOAgent(
        agent_id=0,
        obs_dim_phase1=22,
        obs_dim_phase2=29,
        auction_action_low=np.array([30.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 2.0, 0.2, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([30.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([350.0, 3.0], dtype=np.float32),
        config=config,
        seed=42,
    )

    obs1 = np.zeros(22, dtype=np.float32)
    obs1[3] = 0.156  # expected_price = 78 EUR/t (used in fallback/anchored mode)
    np.random.seed(42)

    prices = []
    for _ in range(1000):
        action, _, _ = agent.select_auction_action(obs1, deterministic=False, epsilon=1.0)
        prices.append(float(action[0]))

    # WTP anchor: with small obs (22D < OBS1_BUDGET_HEADROOM_IDX=27), headroom defaults to 0.5.
    # WTP economic = MAC(48) + 0.5*(penalty(138.75)-MAC(48)) = ~93.4 EUR/t.
    # WTP budget = large (0.5*880 / 0.01). WTP anchor ≈ wtp_economic ≈ 93.4.
    # Compute actual WTP anchor so balance check is around the right reference.
    wtp_mac = 48.0
    wtp_penalty = 138.75
    wtp_budget_default = 880.0
    need_mt = max(0.01, float(obs1[10]) * 10.0)
    headroom_default = 0.5  # fallback (obs dim < 27)
    available = headroom_default * wtp_budget_default
    wtp_economic = wtp_mac + 0.5 * max(0.0, wtp_penalty - wtp_mac)
    wtp_base = min(wtp_economic, available / need_mt)
    price_min = config["auction"]["price_min"]
    price_max = config["auction"]["price_max"]
    wtp_anchor = float(np.clip(wtp_base, price_min, price_max))

    assert min(prices) >= price_min
    assert max(prices) <= price_max
    assert len(prices) > 0

    # WTP-uniform mode: bids are side-balanced around the WTP anchor.
    # Each epsilon sample draws with prob 0.5 from [price_min, wtp_anchor]
    # and prob 0.5 from [wtp_anchor, price_max].
    under_wtp = sum(p < wtp_anchor for p in prices)
    over_wtp = sum(p > wtp_anchor for p in prices)
    non_equal = under_wtp + over_wtp
    assert non_equal > 0
    under_share = under_wtp / non_equal
    assert 0.40 <= under_share <= 0.60, (
        f"Expected side-balanced sampling around WTP anchor {wtp_anchor:.1f}, "
        f"got under_share={under_share:.3f}"
    )


def test_tabula_rasa_uniform_exploration_fallback_expected_price():
    """Fallback expected-price anchor should default near 80 EUR/t, not midpoint."""
    config = {
        "ppo": {
            "hidden_size": 64,
            "lr": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.02,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 2,
            "mini_batch_size": 4,
            "log_std_min": -1.5,
            "log_std_max": 0.0,
            "centralized_critic": False,
            "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 30.0,
            "price_max": 500.0,
            "quantity_max": 3.0,
            "qty_mult_low": 0.3,
            "qty_mult_high": 2.0,
        },
        "investment": {"max_invest_frac": 0.2},
        "trading": {"sec_price_min": 30.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02},
        "simulation": {"n_years": 12},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 2.0},
        "companies": {"n_agents": 1},
        "price": {"initial_expected": 80.0},
        "exploration": {
            "mode": "uniform",
            "auction_anchors": None,
            "secondary_anchors": None,
        },
    }

    agent = PPOAgent(
        agent_id=0,
        obs_dim_phase1=3,  # intentionally omit expected-price feature index [3]
        obs_dim_phase2=6,
        auction_action_low=np.array([30.0, 0.3, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([500.0, 2.0, 0.2, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([30.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([350.0, 3.0], dtype=np.float32),
        config=config,
        seed=7,
    )

    obs1 = np.zeros(3, dtype=np.float32)
    np.random.seed(7)

    prices = []
    for _ in range(1000):
        action, _, _ = agent.select_auction_action(obs1, deterministic=False, epsilon=1.0)
        prices.append(float(action[0]))

    reference_price = float(config["price"]["initial_expected"])
    under = sum(p < reference_price for p in prices)
    over = sum(p > reference_price for p in prices)
    non_equal = under + over

    assert min(prices) >= 30.0
    assert max(prices) <= 500.0
    assert non_equal > 0

    # If midpoint fallback (265 EUR/t) sneaks back in, under-share vs 80 EUR/t
    # collapses far below this range.
    under_share = under / non_equal
    assert 0.40 <= under_share <= 0.60


def test_tabula_rasa_disabled_no_effect(monkeypatch):
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 1000
    cfg["tabula_rasa"]["enabled"] = False

    resolved = _capture_config_after_train_overrides(monkeypatch, cfg)
    assert resolved["pretrain"]["enabled"] is True
    assert resolved["ppo"]["kl_anchor_beta"] == pytest.approx(0.5)
    assert resolved["exploration"]["auction_anchors"] == [80.0, 1.2, 0.03, 0.3, -0.5, 0.5]

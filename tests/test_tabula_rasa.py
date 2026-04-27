"""Tests for v8 tabula-rasa retirement and exploration behavior.

v8 change: tabula_rasa.enabled=true raises ValueError (no longer a runtime
override). Config block is kept in default.yaml for ablation reference only,
always with enabled: false.
"""

import copy
import os
import sys

import numpy as np
import pytest
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
        def __init__(self, cfg, **_):
            captured["config"] = copy.deepcopy(cfg)
            raise RuntimeError("_stop_after_override_")

    monkeypatch.setattr(train_mod, "ETSEnvironment", _StopEnv)

    with pytest.raises(RuntimeError, match="_stop_after_override_"):
        train_mod.train_one_seed(config, seed=42)

    assert "config" in captured
    return captured["config"]


# ---------------------------------------------------------------------------
# enabled=true must raise ValueError
# ---------------------------------------------------------------------------

def test_tabula_rasa_enabled_raises_value_error():
    """tabula_rasa.enabled=true must raise ValueError (feature retired)."""
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 100
    cfg["tabula_rasa"]["enabled"] = True

    with pytest.raises(ValueError, match="tabula_rasa.enabled=true is no longer supported"):
        train_mod.train_one_seed(cfg, seed=42)


# ---------------------------------------------------------------------------
# enabled=false (default) must be a no-op — train proceeds normally
# ---------------------------------------------------------------------------

def test_tabula_rasa_disabled_is_noop(monkeypatch):
    """With tabula_rasa.enabled=false, config values must be unchanged."""
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 100
    cfg["tabula_rasa"]["enabled"] = False

    resolved = _capture_config_after_train_overrides(monkeypatch, cfg)

    # Expected defaults: pretrain off, KL anchor off, exploration anchored (WTP-centered)
    assert resolved["pretrain"]["enabled"] is False
    assert resolved["ppo"]["kl_anchor_beta"] == pytest.approx(0.0)
    assert resolved["exploration"]["mode"] == "anchored"
    assert resolved["exploration"]["auction_anchors"] is None


# ---------------------------------------------------------------------------
# tabula_rasa block must be present in default.yaml with enabled: false
# ---------------------------------------------------------------------------

def test_tabula_rasa_block_present_and_disabled():
    """default.yaml must contain a tabula_rasa block with enabled: false."""
    cfg = _load_config()
    assert "tabula_rasa" in cfg, "tabula_rasa block missing from default.yaml"
    assert cfg["tabula_rasa"]["enabled"] is False, (
        f"tabula_rasa.enabled should be false in default.yaml, "
        f"got {cfg['tabula_rasa']['enabled']}"
    )


# ---------------------------------------------------------------------------
# Standalone uniform exploration tests (independent of tabula-rasa flag)
# These test PPOAgent's WTP-anchored uniform exploration with price_max=250.
# ---------------------------------------------------------------------------

def test_uniform_exploration_side_balanced():
    """With epsilon=1.0, bids should be side-balanced around the WTP anchor."""
    config = {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -2.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 45.0,
            "price_max": 250.0,
            "quantity_max": 3.0,
            "qty_mult_low": 0.5,
            "qty_mult_high": 2.0,
        },
        "investment": {"max_invest_frac": 0.20},
        "trading": {"sec_price_min": 45.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02},
        "simulation": {"n_years": 12},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 10.0},
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
        auction_action_low=np.array([45.0, 0.5, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([250.0, 2.0, 0.20, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([45.0, -3.0], dtype=np.float32),
        secondary_action_high=np.array([350.0, 3.0], dtype=np.float32),
        config=config,
        seed=42,
    )

    obs1 = np.zeros(22, dtype=np.float32)
    obs1[3] = 0.156  # expected_price ~78 EUR/t (for reference in WTP calc)
    np.random.seed(42)

    prices = []
    for _ in range(1000):
        action, _, _ = agent.select_auction_action(obs1, deterministic=False, epsilon=1.0)
        prices.append(float(action[0]))

    # WTP anchor (obs dim < 27, so budget_headroom defaults to 0.5):
    wtp_mac = 48.0
    wtp_penalty = 138.75
    wtp_budget_default = 880.0
    need_mt = max(0.01, float(obs1[10]) * 10.0)
    headroom_default = 0.5
    available = headroom_default * wtp_budget_default
    wtp_economic = wtp_mac + 0.5 * max(0.0, wtp_penalty - wtp_mac)
    wtp_base = min(wtp_economic, available / need_mt)
    price_min = config["auction"]["price_min"]
    price_max = config["auction"]["price_max"]
    wtp_anchor = float(np.clip(wtp_base, price_min, price_max))

    assert min(prices) >= price_min
    assert max(prices) <= price_max

    # WTP-uniform: side-balanced sampling around the WTP anchor
    under_wtp = sum(p < wtp_anchor for p in prices)
    over_wtp = sum(p > wtp_anchor for p in prices)
    non_equal = under_wtp + over_wtp
    assert non_equal > 0
    under_share = under_wtp / non_equal
    assert 0.40 <= under_share <= 0.60, (
        f"Expected side-balanced sampling around WTP anchor {wtp_anchor:.1f}, "
        f"got under_share={under_share:.3f}"
    )


def test_uniform_exploration_fallback_not_at_midpoint():
    """WTP-uniform exploration with null anchors should sample around WTP, not the midpoint."""
    config = {
        "ppo": {
            "hidden_size": 64, "lr": 0.0003, "gamma": 0.99,
            "gae_lambda": 0.95, "clip_eps": 0.2, "entropy_coef": 0.02,
            "value_coef": 0.5, "max_grad_norm": 0.5, "n_epochs": 2,
            "mini_batch_size": 4, "log_std_min": -2.5, "log_std_max": 0.0,
            "centralized_critic": False, "critic_hidden_size": 64,
        },
        "auction": {
            "price_min": 45.0,
            "price_max": 250.0,
            "quantity_max": 3.0,
            "qty_mult_low": 0.5,
            "qty_mult_high": 2.0,
        },
        "investment": {"max_invest_frac": 0.20},
        "trading": {"sec_price_min": 45.0, "sec_price_max_mult": 2.0},
        "penalty": {"rate": 138.75, "inflation_rate": 0.02},
        "simulation": {"n_years": 12},
        "reward": {"normalizer_alpha": 0.01, "clip_min": -10.0, "clip_max": 10.0},
        "companies": {"n_agents": 1},
        "exploration": {
            "mode": "uniform",
            "auction_anchors": None,
            "secondary_anchors": None,
        },
    }

    agent = PPOAgent(
        agent_id=0,
        obs_dim_phase1=3,  # intentionally tiny obs
        obs_dim_phase2=6,
        auction_action_low=np.array([45.0, 0.5, 0.0, -1, -1, -1], dtype=np.float32),
        auction_action_high=np.array([250.0, 2.0, 0.20, 1, 1, 1], dtype=np.float32),
        secondary_action_low=np.array([45.0, -3.0], dtype=np.float32),
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

    mean_price = float(np.mean(prices))
    midpoint = (45.0 + 250.0) / 2.0  # 147.5

    assert min(prices) >= 45.0
    assert max(prices) <= 250.0

    # WTP anchor ~93 EUR/t (mac=48, penalty=138.75 → wtp_economic≈93).
    # Mean should be well below the midpoint (147.5), not near it.
    assert mean_price < midpoint - 20.0, (
        f"Expected mean price well below midpoint {midpoint:.1f} "
        f"(WTP anchor ~93 EUR/t), got mean={mean_price:.1f}"
    )

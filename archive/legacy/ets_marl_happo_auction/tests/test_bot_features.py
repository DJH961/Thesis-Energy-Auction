"""Tests for enhanced bot noise and fade-schedule behaviors."""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _base_cfg():
    cfg = _load_config()
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False
    return cfg


def test_enhanced_noise_wider_range():
    cfg = _base_cfg()
    cfg["bots"]["enhanced_noise"]["enabled"] = True
    cfg["bots"]["enhanced_noise"]["valuation_noise_std"] = 15.0

    env = ETSEnvironment(cfg, seed=42)

    samples = []
    for seed in range(80):
        env.set_episode(seed)
        env.reset(seed=seed)
        samples.extend(env._bot_valuation_noise.tolist())

    std = float(np.std(samples))
    assert 11.0 <= std <= 19.0


def test_enhanced_noise_disabled_default():
    cfg = _base_cfg()
    cfg["bots"]["enhanced_noise"]["enabled"] = False
    cfg["bots"]["valuation_noise_std"] = 5.0

    env = ETSEnvironment(cfg, seed=42)

    samples = []
    for seed in range(80):
        env.set_episode(seed)
        env.reset(seed=seed)
        samples.extend(env._bot_valuation_noise.tolist())

    std = float(np.std(samples))
    assert 3.5 <= std <= 6.5


def test_budget_stress_reduces_qty():
    cfg = _base_cfg()
    cfg["bots"]["enhanced_noise"]["enabled"] = True
    cfg["bots"]["enhanced_noise"]["budget_stress_prob"] = 0.0
    cfg["bots"]["enhanced_noise"]["budget_stress_qty_mult"] = 0.65

    env = ETSEnvironment(cfg, seed=9)
    env.reset(seed=9)

    cap0 = env.cap_schedule.get_cap(0)
    actions_unstressed = env._generate_bot_auction_actions(auction_volume=cap0, cap_t=cap0)

    env._bot_budget_stressed[:] = True
    actions_stressed = env._generate_bot_auction_actions(auction_volume=cap0, cap_t=cap0)

    qty_u = actions_unstressed[:, 1]
    qty_s = actions_stressed[:, 1]

    assert np.all(qty_s <= qty_u + 1e-9)
    assert np.any(qty_s < qty_u - 1e-6)


def test_fade_retires_bots_in_reverse_order():
    cfg = _base_cfg()
    cfg["bots"]["fade_schedule"]["enabled"] = True
    cfg["bots"]["fade_schedule"]["schedule"] = [[0, 8], [1, 6]]

    env = ETSEnvironment(cfg, seed=123)
    env.set_episode(2)
    env.reset(seed=123)

    cap0 = env.cap_schedule.get_cap(0)
    actions = env._generate_bot_auction_actions(auction_volume=cap0, cap_t=cap0)

    price_min = float(env.config["auction"]["price_min"])
    np.testing.assert_allclose(actions[6], np.array([price_min, 0.0, price_min, 0.0, price_min, 0.0,
                                                     0.0, 0.0, 0.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(actions[7], np.array([price_min, 0.0, price_min, 0.0, price_min, 0.0,
                                                     0.0, 0.0, 0.0, 0.0]), atol=1e-9)
    # Check that active bots have non-zero quantities (sum of 3 tranches)
    assert np.any(actions[:6, 1] + actions[:6, 3] + actions[:6, 5] > 0.0)


def test_fade_disabled_all_active():
    cfg = _base_cfg()
    cfg["bots"]["fade_schedule"]["enabled"] = False

    env = ETSEnvironment(cfg, seed=5)
    env.set_episode(99999)
    env.reset(seed=5)

    assert env._n_active_bots == env.n_bots


def test_retired_bots_no_secondary():
    cfg = _base_cfg()
    cfg["bots"]["fade_schedule"]["enabled"] = True
    cfg["bots"]["fade_schedule"]["schedule"] = [[0, 8], [1, 6]]

    env = ETSEnvironment(cfg, seed=55)
    env.set_episode(2)
    env.reset(seed=55)

    env._phase1_allocations = np.zeros(env.n_total)
    actions = env._generate_bot_secondary_actions(clearing_price=80.0)

    np.testing.assert_allclose(actions[6], np.array([0.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(actions[7], np.array([0.0, 0.0]), atol=1e-9)

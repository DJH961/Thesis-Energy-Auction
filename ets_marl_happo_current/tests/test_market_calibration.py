"""Tests for emission-weighted market calibration and backward compatibility."""

import copy
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.environment.market_calibration import compute_market_params


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_16_agents_matches_current_config():
    cfg = _load_config()
    params = compute_market_params(cfg)

    assert params["cap_year_0"] == pytest.approx(50.0, abs=0.1)
    assert params["tnac_upper"] == pytest.approx(18.0, abs=0.1)
    # tnac_lower_ratio updated to 0.22 (from 0.18): 50 × 0.22 ≈ 11.0
    assert params["tnac_lower"] == pytest.approx(11.0, abs=0.2)


def test_12_agents_scales_down():
    cfg = _load_config()
    cfg["companies"]["n_bot_agents"] = 4
    params = compute_market_params(cfg)

    # Emission-weighted scaling with first-4 bots retained (coal/gas-heavy mix)
    # yields a higher cap than participant-count-only scaling.
    assert params["cap_year_0"] == pytest.approx(44.5, abs=1.0)


def test_8_agents_scales_down():
    cfg = _load_config()
    cfg["companies"]["n_bot_agents"] = 0
    params = compute_market_params(cfg)

    assert params["cap_year_0"] == pytest.approx(25.0, abs=1.0)


def test_emission_weighted_not_participant_count():
    cfg = _load_config()

    # Remove two coal-heavy bots (first two in canonical ordering).
    cfg_remove_coal = copy.deepcopy(cfg)
    cfg_remove_coal["companies"]["bot_initial_mix"] = cfg["companies"]["bot_initial_mix"][2:]
    cfg_remove_coal["companies"]["n_bot_agents"] = 6
    cap_coal_removed = compute_market_params(cfg_remove_coal)["cap_year_0"]

    # Remove two green bots (last two in canonical ordering).
    cfg_remove_green = copy.deepcopy(cfg)
    cfg_remove_green["companies"]["bot_initial_mix"] = cfg["companies"]["bot_initial_mix"][:-2]
    cfg_remove_green["companies"]["n_bot_agents"] = 6
    cap_green_removed = compute_market_params(cfg_remove_green)["cap_year_0"]

    assert cap_coal_removed < cap_green_removed


def test_cap_override_ignores_formula():
    cfg = _load_config()
    cfg["ets"]["cap_year_0_override"] = 60.0
    cfg["companies"]["n_bot_agents"] = 0

    params = compute_market_params(cfg)
    assert params["cap_year_0"] == pytest.approx(60.0)


def test_backward_compat_no_overhead_key():
    cfg = _load_config()

    cfg["ets"].pop("cap_overhead_pct", None)
    cfg["ets"].pop("cap_year_0_override", None)
    cfg["ets"]["cap_year_0"] = 50.0
    cfg["ets"]["msr"]["tnac_upper"] = 18.0
    cfg["ets"]["msr"]["tnac_lower"] = 9.0
    cfg["ets"]["msr"]["release_amount"] = 0.8
    cfg["ets"]["msr"]["emergency_release_amount"] = 4.0

    with pytest.warns(DeprecationWarning):
        params = compute_market_params(cfg)

    assert params["cap_year_0"] == pytest.approx(50.0)
    assert params["tnac_upper"] == pytest.approx(18.0)
    assert params["tnac_lower"] == pytest.approx(9.0)


def test_fade_schedule_recalibrates():
    cfg = _load_config()
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False

    cfg["bots"]["fade_schedule"]["enabled"] = True
    cfg["bots"]["fade_schedule"]["schedule"] = [[0, 8], [1, 6]]

    env = ETSEnvironment(cfg, seed=123)

    env.set_episode(0)
    env.reset(seed=123)
    cap_before = env.cap_schedule.cap_year_0

    env.set_episode(2)
    env.reset(seed=123)
    cap_after = env.cap_schedule.cap_year_0

    assert env._n_active_bots == 6
    assert cap_after < cap_before

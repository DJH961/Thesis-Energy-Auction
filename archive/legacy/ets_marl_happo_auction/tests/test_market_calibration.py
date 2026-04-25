"""Tests for emission-weighted market calibration and backward compatibility."""

import copy
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.environment.market_calibration import (
    TNAC_LOWER_REF,
    TNAC_MID_REF,
    TNAC_UPPER_REF,
    compute_market_params,
    compute_system_emissions,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_16_agents_matches_current_config():
    cfg = _load_config()
    params = compute_market_params(cfg)
    total_emissions, _ = compute_system_emissions(cfg)
    expected_cap = total_emissions * (1.0 + cfg["ets"]["cap_overhead_pct"])
    expected_upper = expected_cap * cfg["ets"]["msr"]["tnac_upper_ratio"]

    assert params["cap_year_0"] == pytest.approx(expected_cap, rel=1e-9)
    assert params["tnac_upper"] == pytest.approx(expected_upper, rel=1e-9)
    # Preserve lower:mid:upper ~= 400:833:1096 when scaling to micro-ETS.
    assert params["tnac_mid"] / params["tnac_upper"] == pytest.approx(TNAC_MID_REF / TNAC_UPPER_REF, rel=1e-6)
    assert params["tnac_lower"] / params["tnac_upper"] == pytest.approx(TNAC_LOWER_REF / TNAC_UPPER_REF, rel=1e-6)


def test_12_agents_scales_down():
    cfg = _load_config()
    full_cap = compute_market_params(cfg)["cap_year_0"]
    cfg["companies"]["n_bot_agents"] = 4
    params = compute_market_params(cfg)

    assert params["cap_year_0"] < full_cap


def test_8_agents_scales_down():
    cfg = _load_config()
    cap_16 = compute_market_params(cfg)["cap_year_0"]
    cfg_12 = copy.deepcopy(cfg)
    cfg_12["companies"]["n_bot_agents"] = 4
    cap_12 = compute_market_params(cfg_12)["cap_year_0"]
    cfg["companies"]["n_bot_agents"] = 0
    params = compute_market_params(cfg)

    assert params["cap_year_0"] < cap_12 < cap_16


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
    cfg["ets"]["cap_overhead_pct"] = 999.0
    cfg["companies"]["n_bot_agents"] = 0

    params = compute_market_params(cfg)
    assert params["cap_year_0"] == pytest.approx(60.0)


def test_cap_schedule_uses_override_without_overhead_stack():
    cfg = _load_config()
    cfg["ets"]["cap_year_0_override"] = 60.0
    cfg["ets"]["cap_overhead_pct"] = 999.0
    cfg["warm_start"]["enabled"] = False
    cfg["uncertainty"]["enabled"] = False
    cfg["construction_jitter"]["enabled"] = False

    env = ETSEnvironment(cfg, seed=123)
    env.reset(seed=123)

    assert env.cap_schedule.cap_year_0 == pytest.approx(60.0)


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
    assert params["tnac_mid"] == pytest.approx(18.0 * 833.0 / 1096.0)
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

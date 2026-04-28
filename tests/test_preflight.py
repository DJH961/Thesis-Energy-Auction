"""
test_preflight.py
=================
Tests for ``src.utils.preflight``.

These checks run automatically at the start of every training run (see
``scripts/train.py::train_one_seed``) and are exercised here against the
default config plus a battery of common misconfigurations.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.preflight import (
	PreflightError,
	collect_preflight_errors,
	run_preflight_checks,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
SMOKE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "smoke_100.yaml")


def _load(path: str) -> dict:
	with open(path) as f:
		return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Default config — must always pass preflight
# --------------------------------------------------------------------------- #

def test_default_config_passes_preflight():
	cfg = _load(CONFIG_PATH)
	errors = collect_preflight_errors(cfg)
	assert errors == [], "default.yaml must pass preflight, got errors:\n" + "\n".join(errors)


def test_default_config_run_preflight_does_not_raise():
	run_preflight_checks(_load(CONFIG_PATH))


def test_smoke_config_passes_preflight():
	cfg = _load(SMOKE_CONFIG_PATH)
	errors = collect_preflight_errors(cfg)
	assert errors == [], "smoke_100.yaml must pass preflight, got errors:\n" + "\n".join(errors)


# --------------------------------------------------------------------------- #
# Misconfiguration regressions — each broken config should fail preflight
# --------------------------------------------------------------------------- #

def test_n_agents_mismatch_initial_mix_fails():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["initial_mix"] = cfg["companies"]["initial_mix"][:-1]
	with pytest.raises(PreflightError, match="initial_mix"):
		run_preflight_checks(cfg)


def test_n_agents_mismatch_reward_weights_fails():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["reward_weights"] = cfg["companies"]["reward_weights"][:-1]
	with pytest.raises(PreflightError, match="reward_weights"):
		run_preflight_checks(cfg)


def test_initial_mix_not_summing_to_one_fails():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["initial_mix"][0] = [0.5, 0.5, 0.5, 0.0, 0.0]  # sums to 1.5
	with pytest.raises(PreflightError, match="sums to"):
		run_preflight_checks(cfg)


def test_initial_mix_negative_entry_fails():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["initial_mix"][0] = [1.2, -0.2, 0.0, 0.0, 0.0]  # sums to 1 but negative
	with pytest.raises(PreflightError, match="negative entry"):
		run_preflight_checks(cfg)


def test_no_participants_fails():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["n_agents"] = 0
	cfg["companies"]["initial_mix"] = []
	cfg["companies"]["reward_weights"] = []
	cfg["companies"]["n_bot_agents"] = 0
	cfg["budget"]["annual_budgets"] = []
	cfg["budget"]["debt_headrooms"] = []
	cfg["budget"]["capex_throughputs"] = []
	with pytest.raises(PreflightError, match="no participants"):
		run_preflight_checks(cfg)


def test_bot_arrays_too_short_fail():
	cfg = _load(CONFIG_PATH)
	cfg["companies"]["n_bot_agents"] = 4
	cfg["companies"]["bot_initial_mix"] = cfg["companies"]["bot_initial_mix"][:2]
	with pytest.raises(PreflightError, match="bot_initial_mix"):
		run_preflight_checks(cfg)


def test_budget_array_too_short_fails():
	cfg = _load(CONFIG_PATH)
	cfg["budget"]["annual_budgets"] = cfg["budget"]["annual_budgets"][:-1]
	with pytest.raises(PreflightError, match="annual_budgets"):
		run_preflight_checks(cfg)


def test_technologies_array_length_mismatch_fails():
	cfg = _load(CONFIG_PATH)
	cfg["technologies"]["emission_factors"] = cfg["technologies"]["emission_factors"][:-1]
	with pytest.raises(PreflightError, match="emission_factors"):
		run_preflight_checks(cfg)


def test_auction_bounds_inverted_fails():
	cfg = _load(CONFIG_PATH)
	cfg["auction"]["price_max"] = cfg["auction"]["price_min"] - 1.0
	with pytest.raises(PreflightError, match="price_max"):
		run_preflight_checks(cfg)


def test_reserve_above_price_max_fails():
	cfg = _load(CONFIG_PATH)
	cfg["ets"]["reserve_price"] = cfg["auction"]["price_max"] + 10.0
	with pytest.raises(PreflightError, match="reserve_price"):
		run_preflight_checks(cfg)


def test_penalty_below_mac_fails():
	cfg = _load(CONFIG_PATH)
	cfg["penalty"]["rate"] = cfg["mac"]["coal_to_gas_cost"] - 1.0
	with pytest.raises(PreflightError, match="penalty.rate"):
		run_preflight_checks(cfg)


def test_zero_episodes_fails():
	cfg = _load(CONFIG_PATH)
	cfg["simulation"]["n_episodes"] = 0
	with pytest.raises(PreflightError, match="n_episodes"):
		run_preflight_checks(cfg)


def test_zero_years_fails():
	cfg = _load(CONFIG_PATH)
	cfg["simulation"]["n_years"] = 0
	with pytest.raises(PreflightError, match="n_years"):
		run_preflight_checks(cfg)


def test_msr_thresholds_inverted_fails():
	cfg = _load(CONFIG_PATH)
	cfg["ets"]["msr"]["enabled"] = True
	cfg["ets"]["msr"]["tnac_upper_ratio"] = 0.10
	cfg["ets"]["msr"]["tnac_mid_ratio"] = 0.50
	cfg["ets"]["msr"]["tnac_lower_ratio"] = 0.05
	with pytest.raises(PreflightError, match="tnac_mid_ratio"):
		run_preflight_checks(cfg)


def test_tabula_rasa_enabled_fails_in_preflight():
	cfg = _load(CONFIG_PATH)
	cfg["tabula_rasa"]["enabled"] = True
	with pytest.raises(PreflightError, match="tabula_rasa"):
		run_preflight_checks(cfg)


def test_hpp_seed_heuristic_without_pretrain_fails():
	cfg = _load(CONFIG_PATH)
	cfg["hpp"]["seed_heuristic"] = True
	cfg["pretrain"]["enabled"] = False
	with pytest.raises(PreflightError, match="seed_heuristic"):
		run_preflight_checks(cfg)


# --------------------------------------------------------------------------- #
# Multi-error reporting — one call should surface all problems
# --------------------------------------------------------------------------- #

def test_multiple_errors_reported_together():
	cfg = _load(CONFIG_PATH)
	cfg["simulation"]["n_episodes"] = 0
	cfg["simulation"]["n_years"] = 0
	cfg["companies"]["initial_mix"] = cfg["companies"]["initial_mix"][:-1]
	errors = collect_preflight_errors(cfg)
	# Expect at least one error from each of the three classes above.
	assert any("n_episodes" in e for e in errors)
	assert any("n_years" in e for e in errors)
	assert any("initial_mix" in e for e in errors)
	assert len(errors) >= 3


# --------------------------------------------------------------------------- #
# Integration: preflight runs before training
# --------------------------------------------------------------------------- #

def test_train_one_seed_invokes_preflight(monkeypatch):
	"""``train_one_seed`` must call preflight before constructing the env."""
	import scripts.train as train_mod

	cfg = _load(CONFIG_PATH)
	cfg["simulation"]["n_episodes"] = 1
	# Inject a misconfiguration that preflight catches.
	cfg["companies"]["initial_mix"] = cfg["companies"]["initial_mix"][:-1]

	def _fail_env(*_args, **_kwargs):  # pragma: no cover — must not be reached
		raise AssertionError("ETSEnvironment must not be constructed if preflight fails")

	monkeypatch.setattr(train_mod, "ETSEnvironment", _fail_env)

	with pytest.raises(PreflightError):
		train_mod.train_one_seed(cfg, seed=0)

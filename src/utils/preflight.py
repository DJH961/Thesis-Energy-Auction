"""
preflight.py
============
Preflight validation for ETS MARL training runs.

These checks catch common configuration errors *before* a training run starts,
turning silent corruption (mismatched array lengths, mix vectors that don't
sum to 1, off-by-one bot counts, etc.) into clear, early failures.

Usage
-----
The training entry point calls :func:`run_preflight_checks(config)` near the
top of ``train_one_seed``. Any error raises a :class:`PreflightError` with all
problems reported in a single message.

The same function is exercised by ``tests/test_preflight.py`` so the
default config is validated as part of the regular test suite, and so each
class of misconfiguration is regression-tested.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


class PreflightError(ValueError):
	"""Raised when one or more preflight validation checks fail."""


# --------------------------------------------------------------------------- #
# Individual check helpers — each appends human-readable strings to ``errors``.
# --------------------------------------------------------------------------- #

def _require(cond: bool, msg: str, errors: List[str]) -> None:
	if not cond:
		errors.append(msg)


def _check_simulation(config: dict, errors: List[str]) -> None:
	sim = config.get("simulation", {})
	n_episodes = int(sim.get("n_episodes", 0))
	n_years = int(sim.get("n_years", 0))
	_require(n_episodes > 0, f"simulation.n_episodes must be > 0 (got {n_episodes}).", errors)
	_require(n_years > 0, f"simulation.n_years must be > 0 (got {n_years}).", errors)


def _check_companies(config: dict, errors: List[str]) -> None:
	companies = config.get("companies", {})
	n_agents = int(companies.get("n_agents", 0))
	n_bots = int(companies.get("n_bot_agents", 0))
	_require(n_agents >= 0, f"companies.n_agents must be >= 0 (got {n_agents}).", errors)
	_require(n_bots >= 0, f"companies.n_bot_agents must be >= 0 (got {n_bots}).", errors)
	_require(n_agents + n_bots > 0,
		"companies.n_agents + n_bot_agents must be > 0 (no participants).",
		errors)

	# Learning agents: initial_mix and reward_weights must each have n_agents rows.
	initial_mix = companies.get("initial_mix", [])
	reward_weights = companies.get("reward_weights", [])
	_require(len(initial_mix) == n_agents,
		f"companies.initial_mix length {len(initial_mix)} != n_agents={n_agents}.",
		errors)
	_require(len(reward_weights) == n_agents,
		f"companies.reward_weights length {len(reward_weights)} != n_agents={n_agents}.",
		errors)

	# Each learning mix must sum to 1 and be non-negative.
	for i, mix in enumerate(initial_mix):
		s = float(np.sum(mix))
		_require(abs(s - 1.0) < 1e-6,
			f"companies.initial_mix[{i}] sums to {s:.6f}, expected 1.0.",
			errors)
		_require(min(mix) >= 0.0,
			f"companies.initial_mix[{i}] has a negative entry: {list(mix)}.",
			errors)

	for i, w in enumerate(reward_weights):
		_require(len(w) == 2,
			f"companies.reward_weights[{i}] must be [w_cost, w_green] (length 2), got {len(w)}.",
			errors)

	# Bot arrays — only validated against the *first* n_bots entries, since
	# unused bot rows are kept around in default.yaml as configuration shelf.
	bot_keys = (
		"bot_initial_mix",
		"bot_reward_weights",
	)
	for key in bot_keys:
		arr = companies.get(key, [])
		_require(len(arr) >= n_bots,
			f"companies.{key} length {len(arr)} < n_bot_agents={n_bots}.",
			errors)

	for i, mix in enumerate(companies.get("bot_initial_mix", [])[:n_bots]):
		s = float(np.sum(mix))
		_require(abs(s - 1.0) < 1e-6,
			f"companies.bot_initial_mix[{i}] sums to {s:.6f}, expected 1.0.",
			errors)
		_require(min(mix) >= 0.0,
			f"companies.bot_initial_mix[{i}] has a negative entry: {list(mix)}.",
			errors)

	# Budget arrays must align with their respective participant counts.
	budget = config.get("budget", {})
	for key, expected in (
		("annual_budgets", n_agents),
		("debt_headrooms", n_agents),
		("capex_throughputs", n_agents),
		("bot_annual_budgets", n_bots),
		("bot_debt_headrooms", n_bots),
		("bot_capex_throughputs", n_bots),
	):
		arr = budget.get(key, None)
		if arr is None:
			# Optional buckets — only complain if at least one participant exists.
			if expected > 0:
				errors.append(f"budget.{key} is required when corresponding count is {expected}.")
			continue
		_require(len(arr) >= expected,
			f"budget.{key} length {len(arr)} < expected {expected}.",
			errors)


def _check_technologies(config: dict, errors: List[str]) -> None:
	tech = config.get("technologies", {})
	expected_arrays = (
		"names",
		"emission_factors",
		"capex",
		"capacity_factors",
		"deploy_delays",
		"operational_costs",
		"decommission_costs",
		"is_green",
		"is_buildable",
	)
	lengths = []
	for key in expected_arrays:
		arr = tech.get(key)
		if arr is None:
			errors.append(f"technologies.{key} missing.")
			continue
		lengths.append((key, len(arr)))
	if lengths:
		ref_key, ref_len = lengths[0]
		for key, n in lengths[1:]:
			_require(n == ref_len,
				f"technologies.{key} length {n} != technologies.{ref_key} length {ref_len}.",
				errors)


def _check_auction(config: dict, errors: List[str]) -> None:
	au = config.get("auction", {})
	pmin = float(au.get("price_min", 0))
	pmax = float(au.get("price_max", 0))
	qmin = float(au.get("qty_mult_low", 0))
	qmax = float(au.get("qty_mult_high", 0))
	_require(pmin >= 0, f"auction.price_min must be >= 0 (got {pmin}).", errors)
	_require(pmax > pmin,
		f"auction.price_max ({pmax}) must be > auction.price_min ({pmin}).",
		errors)
	_require(qmax > qmin,
		f"auction.qty_mult_high ({qmax}) must be > auction.qty_mult_low ({qmin}).",
		errors)

	reserve = float(config.get("ets", {}).get("reserve_price", 0))
	_require(reserve <= pmax,
		f"ets.reserve_price ({reserve}) must be <= auction.price_max ({pmax}).",
		errors)
	_require(reserve >= pmin - 1e-9,
		f"ets.reserve_price ({reserve}) must be >= auction.price_min ({pmin}).",
		errors)


def _check_penalty(config: dict, errors: List[str]) -> None:
	pen = config.get("penalty", {})
	rate = float(pen.get("rate", 0))
	_require(rate > 0, f"penalty.rate must be > 0 (got {rate}).", errors)
	mac = float(config.get("mac", {}).get("coal_to_gas_cost", 0))
	_require(rate > mac,
		f"penalty.rate ({rate}) must exceed mac.coal_to_gas_cost ({mac}); "
		"otherwise non-compliance is cheaper than abatement.",
		errors)


def _check_msr(config: dict, errors: List[str]) -> None:
	msr = config.get("ets", {}).get("msr", {})
	if not msr.get("enabled", False):
		return
	upper = msr.get("tnac_upper_ratio")
	mid = msr.get("tnac_mid_ratio")
	lower = msr.get("tnac_lower_ratio")
	# ``mid`` and ``lower`` may be ``None`` (derived); only validate explicit values.
	if upper is not None:
		_require(0.0 < float(upper) < 1.0,
			f"ets.msr.tnac_upper_ratio must be in (0, 1) (got {upper}).",
			errors)
	if mid is not None and upper is not None:
		_require(float(mid) < float(upper),
			f"ets.msr.tnac_mid_ratio ({mid}) must be < tnac_upper_ratio ({upper}).",
			errors)
	if lower is not None and mid is not None:
		_require(float(lower) < float(mid),
			f"ets.msr.tnac_lower_ratio ({lower}) must be < tnac_mid_ratio ({mid}).",
			errors)


def _check_scenario_flags(config: dict, errors: List[str]) -> None:
	# Tabula rasa is retired — explicit guard so the failure message is friendly
	# and matches the runtime check in train.py.
	tr = config.get("tabula_rasa", {})
	if tr.get("enabled", False):
		errors.append(
			"tabula_rasa.enabled=true is no longer supported as a runtime override. "
			"Configure each schedule directly in its own section."
		)

	# HPP heuristic seeding requires BC pretraining to actually have run.
	hpp = config.get("hpp", {})
	pretrain = config.get("pretrain", {})
	if hpp.get("seed_heuristic", False) and not pretrain.get("enabled", False):
		errors.append(
			"hpp.seed_heuristic=true requires pretrain.enabled=true so behavioral "
			"cloning can populate the seed snapshots."
		)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def collect_preflight_errors(config: dict) -> List[str]:
	"""Run all preflight checks and return a list of error messages.

	Empty list means the config passes preflight. Useful for tests that want
	to inspect the failure reasons individually.
	"""
	errors: List[str] = []
	_check_simulation(config, errors)
	_check_companies(config, errors)
	_check_technologies(config, errors)
	_check_auction(config, errors)
	_check_penalty(config, errors)
	_check_msr(config, errors)
	_check_scenario_flags(config, errors)
	return errors


def run_preflight_checks(config: dict) -> None:
	"""Validate ``config`` and raise :class:`PreflightError` on any problem.

	All issues are reported in a single combined message so a misconfigured
	training run can be fixed in one round-trip rather than one error at a time.
	"""
	errors = collect_preflight_errors(config)
	if errors:
		bullet = "\n  - "
		raise PreflightError(
			"Preflight checks failed (" + str(len(errors)) + " issue(s)):"
			+ bullet + bullet.join(errors)
		)

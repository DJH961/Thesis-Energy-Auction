"""Utilities for emission-weighted market calibration.

This module computes cap/MSR calibration values from active participant
emissions so changing the active bot count automatically rescales market
parameters.
"""

from __future__ import annotations

import warnings

import numpy as np


TNAC_LOWER_REF = 400.0
TNAC_MID_REF = 833.0
TNAC_UPPER_REF = 1096.0


def _resolve_active_bots(config: dict, n_active_bots: int | None) -> int:
	"""Resolve active bot count with safe bounds."""
	if n_active_bots is None:
		n_active_bots = int(config["companies"].get("n_bot_agents", 0))
	return max(0, int(n_active_bots))


def compute_system_emissions(config: dict, n_active_bots: int | None = None):
	"""Sum initial emissions across all active participants.

	Parameters
	----------
	config : dict
	n_active_bots : int or None
		If None, uses config["companies"]["n_bot_agents"].
		Active bots are the FIRST n_active_bots from bot_initial_mix.

	Returns
	-------
	total_emissions_mt : float
	per_agent_emissions : list[float]
		Length n_agents + n_active_bots.
	"""
	ef = np.array(config["technologies"]["emission_factors"], dtype=float)
	output_mwh = float(config["companies"]["output_twh"]) * 1e6

	total = 0.0
	per_agent: list[float] = []

	for mix in config["companies"]["initial_mix"]:
		e = output_mwh * float(np.dot(mix, ef)) / 1e6
		total += e
		per_agent.append(e)

	n_active_bots = _resolve_active_bots(config, n_active_bots)
	bot_mixes = config["companies"].get("bot_initial_mix", [])
	for i in range(min(n_active_bots, len(bot_mixes))):
		e = output_mwh * float(np.dot(bot_mixes[i], ef)) / 1e6
		total += e
		per_agent.append(e)

	return float(total), per_agent


def compute_market_params(config: dict, n_active_bots: int | None = None):
	"""Compute cap_year_0 and MSR amounts from config.

	Returns dict with keys:
	  cap_year_0, tnac_upper, tnac_mid, tnac_lower,
	  release_amount, emergency_release_amount,
	  total_emissions, n_active_participants
	"""
	ets_cfg = config["ets"]
	msr_cfg = ets_cfg["msr"]

	total_emissions, _per_agent = compute_system_emissions(config, n_active_bots)
	n_agents = len(config["companies"]["initial_mix"])
	n_active_bots = _resolve_active_bots(config, n_active_bots)

	# Backward compatibility for old config shape that provides hardcoded values.
	if "cap_overhead_pct" not in ets_cfg:
		warnings.warn(
			"[market_calibration] Deprecated ETS config detected. "
			"Please migrate to cap_overhead_pct + MSR ratio fields.",
			DeprecationWarning,
			stacklevel=2,
		)
		cap_year_0 = float(ets_cfg.get("cap_year_0", 0.0))
		return {
			"cap_year_0": cap_year_0,
			"tnac_upper": float(msr_cfg.get("tnac_upper", 0.0)),
			"tnac_mid": float(
				msr_cfg.get(
					"tnac_mid",
					float(msr_cfg.get("tnac_upper", 0.0)) * (TNAC_MID_REF / TNAC_UPPER_REF),
				)
			),
			"tnac_lower": float(msr_cfg.get("tnac_lower", 0.0)),
			"release_amount": float(msr_cfg.get("release_amount", 0.0)),
			"emergency_release_amount": float(msr_cfg.get("emergency_release_amount", 0.0)),
			"total_emissions": float(total_emissions),
			"n_active_participants": int(n_agents + n_active_bots),
		}

	override = ets_cfg.get("cap_year_0_override")
	if override is not None:
		cap_year_0 = float(override)
	else:
		overhead = float(ets_cfg.get("cap_overhead_pct", 0.11))
		cap_year_0 = float(total_emissions) * (1.0 + overhead)

	tnac_upper_ratio = float(msr_cfg.get("tnac_upper_ratio", 0.36))
	# Preserve legislative TNAC band proportions when scaling to micro-ETS.
	ratio_scale = tnac_upper_ratio / TNAC_UPPER_REF
	tnac_upper = cap_year_0 * (TNAC_UPPER_REF * ratio_scale)
	tnac_mid = cap_year_0 * (TNAC_MID_REF * ratio_scale)
	tnac_lower = cap_year_0 * (TNAC_LOWER_REF * ratio_scale)
	release_amount = cap_year_0 * float(msr_cfg.get("release_frac", 0.016))
	emergency_release = cap_year_0 * float(msr_cfg.get("emergency_release_frac", 0.08))

	return {
		"cap_year_0": float(cap_year_0),
		"tnac_upper": float(tnac_upper),
		"tnac_mid": float(tnac_mid),
		"tnac_lower": float(tnac_lower),
		"release_amount": float(release_amount),
		"emergency_release_amount": float(emergency_release),
		"total_emissions": float(total_emissions),
		"n_active_participants": int(n_agents + n_active_bots),
	}


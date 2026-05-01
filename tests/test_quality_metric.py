"""tests/test_quality_metric.py
==============================
Sanity checks for the shared per-episode anchor-invariant quality
metric used by both ``scripts/train.py`` and the Q-learning baseline
(``src/train_qlearning.py``).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.quality_metric import (
    compute_episode_quality, compute_quality_score,
)


@pytest.fixture
def base_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..",
                            "configs", "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _yl(year, n_total, *, clearing=70.0, shortfall_per_agent=0.0,
         emissions_per_agent=2.0, payments_per_agent=140.0):
    return {
        "year": year,
        "clearing_price": clearing,
        "shortfalls": [shortfall_per_agent] * n_total,
        "emissions":  [emissions_per_agent] * n_total,
        "payments":   [payments_per_agent] * n_total,
        "trade_costs":      [0.0] * n_total,
        "invest_costs":     [0.0] * n_total,
        "penalties":        [0.0] * n_total,
        "collateral_costs": [0.0] * n_total,
        "mac_costs":        [0.0] * n_total,
    }


class TestComputeQualityScore:
    """Aggregator-level sanity checks. The util is exercised from
    both train.py and train_qlearning.py, so behaviour must stay
    consistent across NaN handling and weighting."""

    def test_perfect_inputs_score_max(self):
        # Compliance 1, realism 1, saved 1, cost_eff 1, volatility 0
        score = compute_quality_score(1.0, 1.0, 1.0, 1.0, 0.0)
        assert score == pytest.approx(5.0, abs=1e-6)

    def test_worst_inputs_score_min(self):
        # All bad: 0/0/0/0 + volatility 1
        score = compute_quality_score(0.0, 0.0, 0.0, 0.0, 1.0)
        assert score == pytest.approx(-5.0, abs=1e-6)

    def test_neutral_inputs_zero(self):
        score = compute_quality_score(0.5, 0.5, 0.5, 0.5, 0.5)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_all_nan_returns_nan(self):
        nan = float("nan")
        score = compute_quality_score(nan, nan, nan, nan, nan)
        assert np.isnan(score)

    def test_partial_nan_renormalises_weights(self):
        nan = float("nan")
        # Only compliance available, perfect — should saturate at +5
        # because the lone non-NaN component is renormalised to weight 1.
        score = compute_quality_score(1.0, nan, nan, nan, nan)
        assert score == pytest.approx(5.0, abs=1e-6)


class TestComputeEpisodeQuality:
    """End-to-end sanity for the episode-level component computation."""

    def test_perfect_compliant_episode(self, base_config):
        n_total = 8
        n_years = 12
        episode_log = [_yl(y, n_total) for y in range(n_years)]

        def cap_for_year(_y):
            return 22.0  # roughly default's cap_year_0; only used for the anchor

        out = compute_episode_quality(
            episode_log, base_config, cap_for_year, n_total, n_years,
        )
        # Every agent-year is compliant
        assert out["Q_compliance"] == pytest.approx(1.0)
        # All five components must be present (not NaN) for a populated log
        for key in ("Q_compliance", "Q_price_realism", "Q_saved_carbon",
                    "Q_cost_eff", "Q_volatility"):
            assert not np.isnan(out[key]), f"{key} should not be NaN"
        # Aggregate should be in [-5, +5]
        assert -5.0 <= out["quality_score"] <= 5.0

    def test_empty_log_returns_all_nan(self, base_config):
        out = compute_episode_quality([], base_config, lambda y: 22.0, 8, 12)
        for v in out.values():
            assert np.isnan(v)

    def test_non_compliant_lowers_compliance(self, base_config):
        n_total = 8
        n_years = 12
        log = []
        for y in range(n_years):
            # Year 0..2 short by 1.0 Mt for every agent
            sf = 1.0 if y < 3 else 0.0
            log.append(_yl(y, n_total, shortfall_per_agent=sf))
        out = compute_episode_quality(
            log, base_config, lambda _y: 22.0, n_total, n_years,
        )
        # 3/12 years are non-compliant for every agent ⇒ Q_compliance = 9/12 = 0.75
        assert out["Q_compliance"] == pytest.approx(9.0 / 12.0, abs=1e-6)

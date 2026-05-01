"""Tests for the per-episode convergence quality score.

The quality score is an analysis-only signed composite in ``[-5, +5]``
combining compliance, price realism, saved carbon, cost efficiency, and
clearing-vs-anchor volatility. ``compute_quality_score`` is the single
source of truth for the aggregation rule and is exercised here in
isolation so the math is pinned independently of full training runs.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from scripts.train import compute_quality_score  # noqa: E402


def _all_components_nan():
    return dict(
        Q_compliance=float("nan"),
        Q_price_realism=float("nan"),
        Q_saved_carbon=float("nan"),
        Q_cost_eff=float("nan"),
        Q_volatility=float("nan"),
    )


def test_perfect_trajectory_scores_plus_five():
    """All positive components at 1.0 and volatility at 0.0 → +5."""
    q = compute_quality_score(
        Q_compliance=1.0, Q_price_realism=1.0, Q_saved_carbon=1.0,
        Q_cost_eff=1.0, Q_volatility=0.0,
    )
    assert q == pytest.approx(5.0, abs=1e-9)


def test_worst_trajectory_scores_minus_five():
    """All positive components at 0.0 and volatility at 1.0 → -5."""
    q = compute_quality_score(
        Q_compliance=0.0, Q_price_realism=0.0, Q_saved_carbon=0.0,
        Q_cost_eff=0.0, Q_volatility=1.0,
    )
    assert q == pytest.approx(-5.0, abs=1e-9)


def test_neutral_trajectory_scores_zero():
    """Positive components at 0.5 and volatility at 0.5 → 0 (centre)."""
    q = compute_quality_score(
        Q_compliance=0.5, Q_price_realism=0.5, Q_saved_carbon=0.5,
        Q_cost_eff=0.5, Q_volatility=0.5,
    )
    assert q == pytest.approx(0.0, abs=1e-9)


def test_score_stays_within_bounds_for_random_inputs():
    """Score is clipped to [-5, +5] for any [0,1] input combination."""
    import random
    rng = random.Random(0)
    for _ in range(200):
        q = compute_quality_score(
            Q_compliance=rng.random(),
            Q_price_realism=rng.random(),
            Q_saved_carbon=rng.random(),
            Q_cost_eff=rng.random(),
            Q_volatility=rng.random(),
        )
        assert -5.0 - 1e-9 <= q <= 5.0 + 1e-9


def test_volatility_polarity_is_inverted():
    """High volatility hurts; low volatility helps (other components fixed)."""
    base = dict(
        Q_compliance=0.7, Q_price_realism=0.7,
        Q_saved_carbon=0.7, Q_cost_eff=0.7,
    )
    q_low_vol = compute_quality_score(**base, Q_volatility=0.0)
    q_high_vol = compute_quality_score(**base, Q_volatility=1.0)
    assert q_low_vol > q_high_vol
    # The two volatility extremes differ by exactly 2·w_vol·5 = 1.0.
    assert q_low_vol - q_high_vol == pytest.approx(1.0, abs=1e-9)


def test_typical_run_falls_in_signed_range():
    """A trajectory that previously scored ~0.55 in [0,1] now lands near 0
    on the new signed scale, demonstrating the visibility improvement.
    """
    # Components representative of a mid-quality run.
    q = compute_quality_score(
        Q_compliance=0.85, Q_price_realism=0.55, Q_saved_carbon=0.40,
        Q_cost_eff=0.55, Q_volatility=0.25,
    )
    # Old [0,1]-scale composite for these inputs was ≈ 0.55. The signed
    # rescale yields ≈ +1.2, comfortably away from both rails.
    assert -5.0 <= q <= 5.0
    assert 0.5 < q < 2.0


def test_nan_components_are_dropped_and_weights_renormalised():
    """A missing component should not push the score toward 0 by default;
    remaining weights are renormalised so the result stays comparable."""
    # Only compliance available, at the maximum → score should still
    # report the full +5 rather than collapsing to compliance·weight·5
    # (which would be 0.30·1·5 = 1.5).
    q = compute_quality_score(
        Q_compliance=1.0,
        Q_price_realism=float("nan"),
        Q_saved_carbon=float("nan"),
        Q_cost_eff=float("nan"),
        Q_volatility=float("nan"),
    )
    assert q == pytest.approx(5.0, abs=1e-9)


def test_all_nan_returns_nan():
    """No usable components → NaN (CSV writer treats NaN as empty)."""
    q = compute_quality_score(**_all_components_nan())
    assert math.isnan(q)


def test_none_components_are_treated_as_missing():
    """``None`` inputs must be tolerated alongside NaN."""
    q = compute_quality_score(
        Q_compliance=1.0, Q_price_realism=None, Q_saved_carbon=None,
        Q_cost_eff=None, Q_volatility=None,
    )
    assert q == pytest.approx(5.0, abs=1e-9)

"""
quality_metric.py
=================
Per-episode anchor-invariant convergence quality metric used by both the
PPO/HAPPO trainer (``scripts/train.py``) and the Q-learning baseline
(``src/train_qlearning.py``).

Composite of five components, each in ``[0, 1]``:

* ``Q_compliance``     1 → all agent-years compliant
* ``Q_price_realism``  1 → clearing tracks the fundamental anchor
* ``Q_saved_carbon``   1 → all year-0 emissions abated, valued at anchor
* ``Q_cost_eff``       1 → cost ≪ counterfactual cost at anchor
* ``Q_volatility``     0 → clearing/anchor ratio is constant year-to-year

The composite ``quality_score`` is a signed ``[-5, +5]`` aggregate. Both
the components and the aggregate are analysis-only — they never feed
back into training.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.utils.price_anchor import compute_fundamental_anchor


def compute_quality_score(
    Q_compliance: float,
    Q_price_realism: float,
    Q_saved_carbon: float,
    Q_cost_eff: float,
    Q_volatility: float,
) -> float:
    """Aggregate the per-episode quality components into a signed score.

    Each input ``Q_*`` is expected in ``[0, 1]`` (or NaN when the
    component cannot be computed for an episode). The composite is a
    weighted sum of *signed* versions (so the result is naturally
    centred at 0 and negative for poor trajectories), rescaled to
    ``[-5, +5]``:

      * positive components: signed = ``2·x − 1``  ∈ [-1, 1]
      * volatility (penalty): signed = ``1 − 2·v`` ∈ [-1, 1]

    Weights ``{compliance: 0.30, price_realism: 0.25, saved_carbon:
    0.25, cost_eff: 0.10, volatility: 0.10}`` sum to 1.0; multiplying
    by 5 yields the reported ``[-5, +5]`` range. NaN inputs are
    dropped and the remaining weights are renormalised.

    Returns ``float('nan')`` if every component is NaN.
    """
    weights = {
        "compliance": 0.30, "price_realism": 0.25, "saved_carbon": 0.25,
        "cost_eff": 0.10, "volatility": 0.10,
    }
    parts = {
        "compliance": Q_compliance, "price_realism": Q_price_realism,
        "saved_carbon": Q_saved_carbon, "cost_eff": Q_cost_eff,
        "volatility": Q_volatility,
    }
    w_used = 0.0
    signed_sum = 0.0
    for k, v in parts.items():
        if v is None:
            continue
        v = float(v)
        if np.isnan(v):
            continue
        w = weights[k]
        signed = (1.0 - 2.0 * v) if k == "volatility" else (2.0 * v - 1.0)
        signed_sum += w * signed
        w_used += w
    if w_used <= 0:
        return float("nan")
    signed_norm = signed_sum / w_used
    return float(np.clip(5.0 * signed_norm, -5.0, 5.0))


def compute_episode_quality(
    episode_log: Iterable[dict],
    config: dict,
    cap_for_year,
    n_total_agents: int,
    n_years: int,
) -> dict:
    """Compute per-episode quality components from an env episode log.

    Parameters
    ----------
    episode_log : iterable of year-log dicts
        Typically ``env.episode_log`` after a full episode rollout.
    config : dict
        Full simulation config (passed to ``compute_fundamental_anchor``).
    cap_for_year : callable
        Function ``year -> cap_t`` (e.g. ``env.cap_schedule.get_cap``);
        only consulted on the years present in the episode log.
    n_total_agents : int
        Number of companies (learning + bots) — must match the per-agent
        list lengths in the year logs.
    n_years : int
        Episode length, used to clip year indices when looking up caps.

    Returns
    -------
    dict with keys ``Q_compliance``, ``Q_price_realism``,
    ``Q_saved_carbon``, ``Q_cost_eff``, ``Q_volatility``,
    ``quality_score``. Any component that cannot be computed is NaN;
    ``quality_score`` is the renormalised signed aggregate.

    Never raises — defensive against missing keys / malformed logs;
    returns NaN components instead so logging callers don't crash.
    """
    out = {
        "Q_compliance":     float("nan"),
        "Q_price_realism":  float("nan"),
        "Q_saved_carbon":   float("nan"),
        "Q_cost_eff":       float("nan"),
        "Q_volatility":     float("nan"),
        "quality_score":    float("nan"),
    }

    try:
        ep_log = list(episode_log)
        if not ep_log:
            return out

        # Anchor cache (per year-of-episode)
        anchors_per_year: dict[int, float] = {}
        for yl_idx, yl in enumerate(ep_log):
            yr = int(yl.get("year", yl_idx))
            if yr not in anchors_per_year:
                # Clamp on year index for defensive robustness — episode
                # logs from short/curriculum runs occasionally carry a
                # year index past ``n_years - 1`` (e.g. terminal-payoff
                # bookkeeping). Clamping keeps the anchor lookup well-
                # defined; the underlying trajectory shape is unchanged.
                cap_t_a = float(cap_for_year(min(yr, max(0, n_years - 1))))
                anchors_per_year[yr] = float(compute_fundamental_anchor(
                    yr, config, cap_t_actual=cap_t_a
                ))

        # 1. Compliance (1 − non-compliance year share)
        total_years = 0
        compliant = 0
        for yl in ep_log:
            shorts = yl.get("shortfalls", [0.0] * n_total_agents)
            for i in range(n_total_agents):
                total_years += 1
                if i < len(shorts) and float(shorts[i]) <= 1e-6:
                    compliant += 1
        Q_compliance = (compliant / max(total_years, 1)) if total_years else float("nan")

        # 2. Price realism
        rel_errs = []
        for yl in ep_log:
            cp = float(yl.get("clearing_price", 0.0) or 0.0)
            yr = int(yl.get("year", 0))
            anc = float(anchors_per_year.get(yr, 0.0))
            if anc > 1e-6 and cp > 1e-6:
                rel_errs.append(min(5.0, abs(cp - anc) / anc))
        Q_price_realism = (
            float(np.clip(1.0 - float(np.mean(rel_errs)), 0.0, 1.0))
            if rel_errs else float("nan")
        )

        # 3. Saved carbon (and counterfactual reused for cost_eff)
        yr0_emiss = None
        for yl in ep_log:
            if int(yl.get("year", 0)) == 0:
                yr0_emiss = list(yl.get("emissions", [0.0] * n_total_agents))
                break

        ctrf_value = 0.0
        if yr0_emiss is not None and len(yr0_emiss) >= n_total_agents:
            saved_value = 0.0
            for yl in ep_log:
                yr = int(yl.get("year", 0))
                anc = float(anchors_per_year.get(yr, 0.0))
                if anc <= 1e-6:
                    continue
                emiss_y = yl.get("emissions", [0.0] * n_total_agents)
                for i in range(n_total_agents):
                    base_i = max(0.0, float(yr0_emiss[i]))
                    real_i = max(0.0, float(emiss_y[i] if i < len(emiss_y) else 0.0))
                    saved_value += max(0.0, base_i - real_i) * anc
                    ctrf_value += base_i * anc
            Q_saved_carbon = (
                float(np.clip(saved_value / ctrf_value, 0.0, 1.0))
                if ctrf_value > 1e-6 else float("nan")
            )
        else:
            Q_saved_carbon = float("nan")

        # 4. Cost efficiency. Sum realised costs from year-log per-agent
        # arrays (auction payments + trade + invest + penalty + collateral
        # + MAC). The PPO trainer additionally checks for already-
        # flattened per-agent column names; tabular Q-learning year logs
        # store raw per-agent lists, so we use those directly.
        if ctrf_value > 1e-6:
            total_cost = 0.0
            cost_keys = ("payments", "trade_costs", "invest_costs",
                         "penalties", "collateral_costs", "mac_costs")
            for yl in ep_log:
                for ck in cost_keys:
                    arr = yl.get(ck)
                    if arr is None:
                        continue
                    for i in range(min(n_total_agents, len(arr))):
                        total_cost += float(arr[i] or 0.0)
            cost_eff_raw = float(np.clip(1.0 - total_cost / ctrf_value, -1.0, 1.0))
            Q_cost_eff = float(np.clip(0.5 * (cost_eff_raw + 1.0), 0.0, 1.0))
        else:
            Q_cost_eff = float("nan")

        # 5. Volatility — std/mean of clearing/anchor ratio
        ratios = []
        for yl in ep_log:
            yr = int(yl.get("year", 0))
            cp = float(yl.get("clearing_price", 0.0) or 0.0)
            anc = float(anchors_per_year.get(yr, 0.0))
            if anc > 1e-6 and cp > 1e-6:
                ratios.append(cp / anc)
        if len(ratios) > 1:
            r_mu = float(np.mean(ratios))
            r_sd = float(np.std(ratios))
            Q_volatility = (
                float(np.clip(r_sd / r_mu, 0.0, 1.0)) if r_mu > 1e-6 else float("nan")
            )
        else:
            Q_volatility = float("nan")

        out["Q_compliance"]    = Q_compliance
        out["Q_price_realism"] = Q_price_realism
        out["Q_saved_carbon"]  = Q_saved_carbon
        out["Q_cost_eff"]      = Q_cost_eff
        out["Q_volatility"]    = Q_volatility
        out["quality_score"]   = compute_quality_score(
            Q_compliance, Q_price_realism, Q_saved_carbon,
            Q_cost_eff, Q_volatility,
        )
    except Exception:
        # Defensive: never crash the training loop on metric-calc failure.
        pass

    return out

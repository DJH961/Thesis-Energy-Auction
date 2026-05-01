"""
run_bots_only.py
================
Run a heuristic-bot-only EU ETS market for many episodes (no learning).

The bot policy lives in ``src/agents/heuristic_policy.py`` and is the
same code path the env uses when learning agents and bots co-exist;
this driver is the ``n_agents = 0`` rollout so a pure rule-based market
can be compared with the PPO/HAPPO and Q-learning baselines on
the same default-config calibration.

Outputs (per seed) match the schema produced by the PPO trainer so the
analysis notebooks can load them directly:

    <output_dir>/training_log_<tag>_s<seed>.csv
    <output_dir>/year_log_<tag>_s<seed>.csv

Per-row CSV fields are episode-summary metrics (clearing prices,
quality score, compliance, green frac …); year_log carries the per-
year traces for each episode.

Usage
-----
    python scripts/run_bots_only.py --config configs/bots_only.yaml \
        --seeds 42 123 456 --n-episodes 500
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from typing import Iterable, List

import numpy as np
import yaml

if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.ets_environment import ETSEnvironment
from src.utils.quality_metric import compute_episode_quality


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_python(value):
    """Coerce numpy scalars / arrays into JSON-friendly native types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_bots_only_seed(
    config: dict,
    seed: int,
    n_episodes: int,
    output_dir: str,
    run_tag: str | None = None,
    log_interval: int = 50,
) -> dict:
    """Run ``n_episodes`` heuristic-bot rollouts for a single seed."""
    n_years = int(config["simulation"]["n_years"])
    n_total = int(config["companies"].get("n_agents", 0)) + int(
        config["companies"].get("n_bot_agents", 0)
    )
    if n_total == 0:
        raise ValueError("n_agents + n_bot_agents must be > 0")

    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{run_tag}" if run_tag else ""
    train_path = os.path.join(output_dir, f"training_log{tag}_s{seed}.csv")
    year_path = os.path.join(output_dir, f"year_log{tag}_s{seed}.csv")

    # CSV schemas — kept small but compatible with the analysis notebooks.
    train_fields: List[str] = [
        "episode",
        "ep_mean_clearing_price", "mean_price",
        "price_start", "price_peak", "price_std",
        "secondary_avg_price", "secondary_volume_total",
        "compliance_rate", "shortfall_total",
        "mean_green_frac",
        "quality_score",
        "Q_compliance", "Q_price_realism", "Q_saved_carbon",
        "Q_cost_eff", "Q_volatility",
    ]
    for i in range(n_total):
        train_fields += [
            f"reward_A{i+1}", f"green_frac_A{i+1}",
            f"shortfall_A{i+1}", f"penalty_A{i+1}",
            f"compliance_rate_A{i+1}",
        ]

    year_fields: List[str] = [
        "episode", "year", "cap", "tnac", "msr_reserve",
        "auction_volume", "clearing_price", "effective_reserve",
        "secondary_clearing", "secondary_volume",
        "inflation_rate", "inflation_factor",
    ]
    for i in range(n_total):
        year_fields += [
            f"emissions_A{i+1}", f"allocation_A{i+1}",
            f"holdings_A{i+1}", f"shortfall_A{i+1}",
            f"penalty_A{i+1}", f"trade_qty_A{i+1}",
            f"trade_cost_A{i+1}", f"bid_price_A{i+1}",
            f"bid_qty_mult_A{i+1}", f"invest_cost_A{i+1}",
            f"reward_A{i+1}", f"green_frac_A{i+1}",
        ]

    train_csv = open(train_path, "w", newline="")
    year_csv = open(year_path, "w", newline="")
    train_w = csv.DictWriter(train_csv, fieldnames=train_fields)
    year_w = csv.DictWriter(year_csv, fieldnames=year_fields)
    train_w.writeheader()
    year_w.writeheader()

    env = ETSEnvironment(config, seed=seed)
    cap_for_year = env.cap_schedule.get_cap

    t0 = time.time()
    print(f"[bots-only seed={seed}] starting {n_episodes} episodes — log every {log_interval}")

    no_auc = np.zeros((0, 6), dtype=np.float32)
    no_sec = np.zeros((0, 2), dtype=np.float32)

    for ep in range(n_episodes):
        env.reset(seed=seed * 100003 + ep)
        ep_log: list[dict] = []
        clearing_prices: list[float] = []
        sec_prices: list[float] = []
        sec_volumes: list[float] = []
        per_agent_reward = np.zeros(n_total, dtype=float)
        per_agent_shortfall = np.zeros(n_total, dtype=float)
        per_agent_penalty = np.zeros(n_total, dtype=float)
        per_agent_compliant = np.zeros(n_total, dtype=int)

        for y in range(n_years):
            env.step_auction(no_auc)
            _, _, _, _, info = env.step_secondary(no_sec)
            log = info["year_log"]
            ep_log.append(log)

            cp = float(log.get("clearing_price", 0.0) or 0.0)
            clearing_prices.append(cp)
            sec_prices.append(float(log.get("secondary_clearing", 0.0) or 0.0))
            sec_volumes.append(float(log.get("secondary_volume", 0.0) or 0.0))

            rewards = np.asarray(log.get("rewards", [0.0] * n_total), dtype=float)
            shortfalls = np.asarray(log.get("shortfalls", [0.0] * n_total), dtype=float)
            penalties = np.asarray(log.get("penalties", [0.0] * n_total), dtype=float)
            per_agent_reward += rewards
            per_agent_shortfall += shortfalls
            per_agent_penalty += penalties
            per_agent_compliant += (shortfalls <= 1e-6).astype(int)

            year_row = {
                "episode": ep,
                "year": int(log.get("year", y)),
                "cap": _to_python(log.get("cap", 0.0)),
                "tnac": _to_python(log.get("tnac", 0.0)),
                "msr_reserve": _to_python(log.get("msr_reserve", 0.0)),
                "auction_volume": _to_python(log.get("auction_volume", 0.0)),
                "clearing_price": cp,
                "effective_reserve": _to_python(log.get("effective_reserve", 0.0)),
                "secondary_clearing": sec_prices[-1],
                "secondary_volume": sec_volumes[-1],
                "inflation_rate": _to_python(log.get("inflation_rate", 0.0)),
                "inflation_factor": _to_python(log.get("inflation_factor", 1.0)),
            }
            allocs = log.get("allocations", [0.0] * n_total)
            ems = log.get("emissions", [0.0] * n_total)
            holds = log.get("holdings", [0.0] * n_total)
            tcosts = log.get("trade_costs", [0.0] * n_total)
            tqtys = log.get("trade_qtys", [0.0] * n_total)
            bps = log.get("bid_prices", [0.0] * n_total)
            bqms = log.get("bid_qty_multipliers", [0.0] * n_total)
            ics = log.get("invest_costs", [0.0] * n_total)
            gfs = log.get("green_fracs", [0.0] * n_total)
            for i in range(n_total):
                year_row[f"emissions_A{i+1}"] = float(ems[i]) if i < len(ems) else 0.0
                year_row[f"allocation_A{i+1}"] = float(allocs[i]) if i < len(allocs) else 0.0
                year_row[f"holdings_A{i+1}"] = float(holds[i]) if i < len(holds) else 0.0
                year_row[f"shortfall_A{i+1}"] = float(shortfalls[i]) if i < len(shortfalls) else 0.0
                year_row[f"penalty_A{i+1}"] = float(penalties[i]) if i < len(penalties) else 0.0
                year_row[f"trade_qty_A{i+1}"] = float(tqtys[i]) if i < len(tqtys) else 0.0
                year_row[f"trade_cost_A{i+1}"] = float(tcosts[i]) if i < len(tcosts) else 0.0
                year_row[f"bid_price_A{i+1}"] = float(bps[i]) if i < len(bps) else 0.0
                year_row[f"bid_qty_mult_A{i+1}"] = float(bqms[i]) if i < len(bqms) else 0.0
                year_row[f"invest_cost_A{i+1}"] = float(ics[i]) if i < len(ics) else 0.0
                year_row[f"reward_A{i+1}"] = float(rewards[i]) if i < len(rewards) else 0.0
                year_row[f"green_frac_A{i+1}"] = float(gfs[i]) if i < len(gfs) else 0.0
            year_w.writerow(year_row)

        # Episode summary
        cps = np.asarray(clearing_prices, dtype=float)
        spv = np.asarray(sec_prices, dtype=float)
        svol = np.asarray(sec_volumes, dtype=float)
        # Volume-weighted secondary average (skip zero-volume years).
        sec_avg = float(np.average(spv, weights=svol)) if svol.sum() > 1e-9 else 0.0
        green_fracs_last = np.asarray(ep_log[-1].get("green_fracs", [0.0] * n_total), dtype=float)
        compliance_rate = float(per_agent_compliant.sum()) / float(max(n_years * n_total, 1))

        q = compute_episode_quality(
            ep_log, config, cap_for_year, n_total, n_years
        )

        train_row = {
            "episode": ep,
            "ep_mean_clearing_price": float(cps.mean()) if cps.size else 0.0,
            "mean_price": float(cps.mean()) if cps.size else 0.0,
            "price_start": float(cps[0]) if cps.size else 0.0,
            "price_peak": float(cps.max()) if cps.size else 0.0,
            "price_std": float(cps.std()) if cps.size else 0.0,
            "secondary_avg_price": sec_avg,
            "secondary_volume_total": float(svol.sum()),
            "compliance_rate": compliance_rate,
            "shortfall_total": float(per_agent_shortfall.sum()),
            "mean_green_frac": float(green_fracs_last.mean()),
            "quality_score": q.get("quality_score", float("nan")),
            "Q_compliance": q.get("Q_compliance", float("nan")),
            "Q_price_realism": q.get("Q_price_realism", float("nan")),
            "Q_saved_carbon": q.get("Q_saved_carbon", float("nan")),
            "Q_cost_eff": q.get("Q_cost_eff", float("nan")),
            "Q_volatility": q.get("Q_volatility", float("nan")),
        }
        for i in range(n_total):
            train_row[f"reward_A{i+1}"] = float(per_agent_reward[i])
            train_row[f"green_frac_A{i+1}"] = float(green_fracs_last[i]) if i < len(green_fracs_last) else 0.0
            train_row[f"shortfall_A{i+1}"] = float(per_agent_shortfall[i])
            train_row[f"penalty_A{i+1}"] = float(per_agent_penalty[i])
            train_row[f"compliance_rate_A{i+1}"] = float(per_agent_compliant[i]) / float(max(n_years, 1))
        train_w.writerow(train_row)

        if (ep + 1) % log_interval == 0 or ep == 0:
            qs = train_row["quality_score"]
            qs_str = f"{qs:+.2f}" if qs == qs else "nan"   # NaN-safe
            print(
                f"[bots-only s={seed}] ep {ep+1:5d}/{n_episodes} | "
                f"clr̄={train_row['ep_mean_clearing_price']:6.1f} | "
                f"comply={compliance_rate*100:5.1f}% | "
                f"green̄={train_row['mean_green_frac']*100:5.1f}% | "
                f"Q={qs_str}"
            )

    train_csv.close()
    year_csv.close()
    dt = time.time() - t0
    print(f"[bots-only seed={seed}] done in {dt:.1f}s → {train_path}")
    return {"train_path": train_path, "year_path": year_path, "elapsed": dt}


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run a heuristic-bots-only EU ETS market simulation."
    )
    parser.add_argument("--config", type=str, default="configs/bots_only.yaml")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Override seeds. Defaults to simulation.seeds in the config.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single-seed shortcut (overrides --seeds).",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Override simulation.n_episodes.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <results_dir>/bots_only/).",
    )
    parser.add_argument(
        "--run-tag", type=str, default=None,
        help="Filename infix for outputs.",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.seed is not None:
        seeds: list[int] = [int(args.seed)]
    elif args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [int(s) for s in config.get("simulation", {}).get("seeds", [42])]

    n_episodes = int(
        args.n_episodes
        if args.n_episodes is not None
        else config.get("simulation", {}).get("n_episodes", 500)
    )
    output_dir = args.output_dir or os.path.join(
        config.get("logging", {}).get("results_dir", "results/"), "bots_only"
    )

    print(
        f"Bots-only baseline | config={args.config} | seeds={seeds} | "
        f"episodes/seed={n_episodes} | output_dir={output_dir}"
    )
    for seed in seeds:
        run_bots_only_seed(
            config=config,
            seed=seed,
            n_episodes=n_episodes,
            output_dir=output_dir,
            run_tag=args.run_tag,
            log_interval=args.log_interval,
        )


if __name__ == "__main__":
    main()

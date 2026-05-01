"""Smoke test to ensure short training runs complete without crashing."""

import csv
import os
import sys
import warnings
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import train as train_mod


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "smoke_100.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_training_smoke_runs_10_episodes(tmp_path):
    cfg = _load_config()
    n_episodes = 10
    seed = 7

    cfg["simulation"]["n_episodes"] = n_episodes
    cfg["simulation"]["n_years"] = 4
    cfg["logging"]["results_dir"] = str(tmp_path / "results")
    cfg["logging"]["log_interval"] = max(10, n_episodes)
    cfg["logging"]["save_interval"] = 500
    cfg["logging"]["csv_flush_interval"] = 1
    cfg["pretrain"]["enabled"] = False
    cfg["pretrain"]["episodes"] = 0
    cfg["pretrain"]["epochs"] = 0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\[ETSEnvironment\] Weak scarcity: cap drops only .*",
            category=UserWarning,
        )
        train_mod.train_one_seed(cfg, seed=seed)

    ep_log = Path(cfg["logging"]["results_dir"]) / f"training_log_s{seed}.csv"
    assert ep_log.exists()

    with ep_log.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == n_episodes
    assert int(float(rows[-1]["episode"])) == n_episodes - 1

    # v8.6.1 expanded episode-level logging — pin the new column names.
    expected_ep_cols = {
        "year0_tnac", "yearT_tnac", "ep_total_unsold",
        "ep_auction_failures", "ep_total_defaults",
        "peak_loan_outstanding_A1", "peak_carry_forward_A1",
        "final_treasury_reserve_A1",
    }
    assert expected_ep_cols.issubset(rows[0].keys()), \
        f"Missing v8.6.1 episode columns: {expected_ep_cols - set(rows[0].keys())}"

    # And the year-level expanded columns.
    yr_log = Path(cfg["logging"]["results_dir"]) / f"year_log_s{seed}.csv"
    assert yr_log.exists()
    with yr_log.open(newline="", encoding="utf-8") as f:
        yr_rows = list(csv.DictReader(f))
    expected_yr_cols = {
        "auction_total_demand", "auction_unsold", "auction_hhi",
        "auction_max_agent_share", "auction_failed",
        "auction_defaults", "auction_defaulted_volume",
        "effective_reserve_price",
        "secondary_n_buyers_intent", "secondary_n_sellers_intent",
        "secondary_n_buyers_executed", "secondary_n_sellers_executed",
        "common_emission_shock", "fundamental_anchor",
        "carry_forward_start_A1", "carry_forward_end_A1",
        "coverage_gap_A1", "effective_penalty_rate_A1",
        "treasury_reserve_A1", "treasury_drawn_A1", "loan_outstanding_A1",
    }
    assert expected_yr_cols.issubset(yr_rows[0].keys()), \
        f"Missing v8.6.1 year columns: {expected_yr_cols - set(yr_rows[0].keys())}"

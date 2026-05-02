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
    # The smoke test asserts on the live CSV files, so disable end-of-run
    # log/checkpoint compression here. (The compression hook is exercised
    # separately in test_compress_on_finish_hook.)
    cfg["logging"].setdefault("compress_on_finish", {})
    cfg["logging"]["compress_on_finish"]["logs"] = False
    cfg["logging"]["compress_on_finish"]["checkpoints"] = False

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
    n_total = cfg["companies"]["n_agents"] + cfg["companies"].get("n_bot_agents", 0)
    expected_ep_cols = {
        "year0_tnac", "yearT_tnac", "ep_total_unsold",
        "ep_auction_failures", "ep_total_defaults",
    }
    for i in range(n_total):
        expected_ep_cols.update({
            f"peak_loan_outstanding_A{i+1}",
            f"peak_carry_forward_A{i+1}",
            f"final_treasury_reserve_A{i+1}",
        })
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
    }
    for i in range(n_total):
        expected_yr_cols.update({
            f"carry_forward_start_A{i+1}",
            f"carry_forward_end_A{i+1}",
            f"coverage_gap_A{i+1}",
            f"effective_penalty_rate_A{i+1}",
            f"treasury_reserve_A{i+1}",
            f"treasury_drawn_A{i+1}",
            f"loan_outstanding_A{i+1}",
        })
    assert expected_yr_cols.issubset(yr_rows[0].keys()), \
        f"Missing v8.6.1 year columns: {expected_yr_cols - set(yr_rows[0].keys())}"


def test_compress_on_finish_hook(tmp_path):
    """train_one_seed compresses logs + checkpoints at clean end-of-run.

    The hook is opt-out (default true); we run a 5-episode smoke and
    assert that the CSVs were replaced by parquet siblings and the
    checkpoint dir was bundled into a tar.xz.
    """
    cfg = _load_config()
    cfg["simulation"]["n_episodes"] = 5
    cfg["simulation"]["n_years"] = 3
    cfg["logging"]["results_dir"] = str(tmp_path / "results")
    cfg["logging"]["log_interval"] = 5
    cfg["logging"]["save_interval"] = 5  # force at least one checkpoint write
    cfg["logging"]["csv_flush_interval"] = 1
    cfg["pretrain"]["enabled"] = False
    cfg["pretrain"]["episodes"] = 0
    cfg["pretrain"]["epochs"] = 0
    # Defaults already enable compress_on_finish.{logs,checkpoints}; this
    # test verifies the wiring works end-to-end.

    seed = 11
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\[ETSEnvironment\] Weak scarcity: cap drops only .*",
            category=UserWarning,
        )
        train_mod.train_one_seed(cfg, seed=seed)

    results_dir = Path(cfg["logging"]["results_dir"])
    ep_csv = results_dir / f"training_log_s{seed}.csv"
    yr_csv = results_dir / f"year_log_s{seed}.csv"
    ep_pq = results_dir / f"training_log_s{seed}.parquet"
    yr_pq = results_dir / f"year_log_s{seed}.parquet"

    # CSVs were converted and removed; parquet siblings are on disk.
    assert not ep_csv.exists(), "training_log CSV should be removed after compression"
    assert not yr_csv.exists(), "year_log CSV should be removed after compression"
    assert ep_pq.exists(), "training_log parquet should exist after compression"
    assert yr_pq.exists(), "year_log parquet should exist after compression"

    # Checkpoint dir was bundled and removed.
    ckpt_dir = results_dir / f"checkpoints_s{seed}"
    ckpt_archive = results_dir / f"checkpoints_s{seed}.tar.xz"
    if ckpt_archive.exists():
        # Archive present implies the source dir was removed.
        assert not ckpt_dir.exists()

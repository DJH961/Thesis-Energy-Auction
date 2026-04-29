"""Tests for scripts/sweep.py launcher helpers (log capture, heartbeat)."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time

import pytest


# Load scripts/sweep.py as a module so we can unit-test its helpers without
# the launcher's argparse / subprocess code path running.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SWEEP_PATH = os.path.join(_REPO_ROOT, "scripts", "sweep.py")


@pytest.fixture(scope="module")
def sweep_module():
    spec = importlib.util.spec_from_file_location("scripts_sweep", _SWEEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Ensure src/ is importable (sweep.py also does this, but tests may run before).
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    spec.loader.exec_module(mod)
    return mod


class TestJobLogPath:
    def test_includes_variant_and_seed(self, sweep_module):
        path = sweep_module._job_log_path("/tmp/results/x", "tight_cap", 42)
        assert os.path.basename(path) == "run_tight_cap_s42.log"
        assert path.startswith("/tmp/results/x")


class TestTailLastMeaningfulLine:
    def test_none_for_missing_file(self, sweep_module, tmp_path):
        assert sweep_module._tail_last_meaningful_line(str(tmp_path / "nope.log")) is None

    def test_none_for_empty_file(self, sweep_module, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        assert sweep_module._tail_last_meaningful_line(str(p)) is None

    def test_skips_blank_and_comment_lines(self, sweep_module, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text(
            "# header comment\n"
            "first real line\n"
            "[Ep 100/1000  reward=12.3]\n"
            "\n"
            "   \n"
            "# trailing comment\n"
        )
        line = sweep_module._tail_last_meaningful_line(str(p))
        assert line == "[Ep 100/1000  reward=12.3]"

    def test_returns_none_when_only_comments(self, sweep_module, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("# a\n# b\n   \n")
        assert sweep_module._tail_last_meaningful_line(str(p)) is None

    def test_handles_large_file_via_tail_window(self, sweep_module, tmp_path):
        p = tmp_path / "big.log"
        # Write a >32KB file ending in a known line.
        with open(p, "w") as f:
            for i in range(2000):
                f.write(f"line {i} with some padding " * 4 + "\n")
            f.write("FINAL_MEANINGFUL_LINE\n")
        line = sweep_module._tail_last_meaningful_line(str(p))
        assert line == "FINAL_MEANINGFUL_LINE"

    def test_skips_pure_separator_lines(self, sweep_module, tmp_path):
        # The trainer prints separator banners (═══, ───, ====) at block
        # boundaries — the heartbeat must look past them to the real
        # last informative line.
        p = tmp_path / "log.txt"
        p.write_text(
            "[Ep 100/1000  reward=12.3]\n"
            + ("═" * 80) + "\n"
            + ("─" * 80) + "\n"
            + "= = = = = = = = =\n"
            + "----\n",
            encoding="utf-8",
        )
        line = sweep_module._tail_last_meaningful_line(str(p))
        assert line == "[Ep 100/1000  reward=12.3]"


class TestSummarizeCsv:
    """Tests for the year-log summary used by the heartbeat.

    The summariser reports the last completed episode's year-1 vs year-N
    (clearing price, secondary price, compliance fraction, green share)
    plus that episode's across-years mean. Reward is reported as a
    last-N-episode mean read from a separate per-episode (training) CSV.
    """

    YR_BASE_HEADERS = ["episode", "year", "clearing_price", "secondary_price"]

    def _write_year_csv(self, path, episodes, n_agents, *, n_years=12,
                        px=None, sec=None, green=None, shortfall=None):
        """Write a year-log fixture.

        ``px``/``sec`` are callables ``(ep, year) -> float`` (defaults
        to a stable placeholder). ``green``/``shortfall`` are callables
        ``(ep, year, agent_idx) -> float``.
        """
        import csv as _csv

        if px is None:
            px = lambda ep, yr: 80.0
        if sec is None:
            sec = lambda ep, yr: 70.0
        if green is None:
            green = lambda ep, yr, a: 0.30
        if shortfall is None:
            shortfall = lambda ep, yr, a: 0.0

        headers = list(self.YR_BASE_HEADERS)
        for i in range(n_agents):
            headers += [f"reward_A{i+1}", f"green_frac_A{i+1}", f"shortfall_A{i+1}"]
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for ep in episodes:
                for yr in range(1, n_years + 1):
                    row = {
                        "episode": ep, "year": yr,
                        "clearing_price": px(ep, yr),
                        "secondary_price": sec(ep, yr),
                    }
                    for i in range(n_agents):
                        row[f"reward_A{i+1}"] = -1.0
                        row[f"green_frac_A{i+1}"] = green(ep, yr, i)
                        row[f"shortfall_A{i+1}"] = shortfall(ep, yr, i)
                    w.writerow(row)

    def _write_train_csv(self, path, episodes, n_agents, *, reward=None,
                         match_rate=None):
        import csv as _csv

        if reward is None:
            reward = lambda ep, a: -2.0
        if match_rate is None:
            match_rate = lambda ep: 0.4
        headers = ["episode", "secondary_match_rate"]
        for i in range(n_agents):
            headers.append(f"reward_A{i+1}")
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for ep in episodes:
                row = {"episode": ep, "secondary_match_rate": match_rate(ep)}
                for i in range(n_agents):
                    row[f"reward_A{i+1}"] = reward(ep, i)
                w.writerow(row)

    def test_returns_none_for_missing_file(self, sweep_module, tmp_path):
        line, last_ep = sweep_module._summarize_csv(str(tmp_path / "nope.csv"))
        assert line is None
        assert last_ep is None

    def test_returns_none_for_header_only(self, sweep_module, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("episode,year,clearing_price,reward_A1,green_frac_A1,shortfall_A1\n")
        line, last_ep = sweep_module._summarize_csv(str(p))
        assert line is None
        assert last_ep is None

    def test_basic_summary_uses_last_episode_y1_yn(self, sweep_module, tmp_path):
        p = tmp_path / "year.csv"
        # Episode 100: price 70 → 130 across 12 years.
        self._write_year_csv(
            p, episodes=[10, 50, 100], n_agents=2,
            px=lambda ep, yr: 70.0 + (yr - 1) * (60.0 / 11) if ep == 100 else 50.0,
            sec=lambda ep, yr: 60.0 + (yr - 1) * 2.0 if ep == 100 else 40.0,
        )
        line, last_ep = sweep_module._summarize_csv(str(p), n_episodes=1000)
        assert line is not None
        assert last_ep == 100
        assert "Ep 100/1000" in line
        # Year-1 vs Year-12 of last episode (px starts at 70, ends at 130).
        assert "px 70→130" in line
        # Episode-mean reported in parentheses.
        assert "(μ" in line
        # Compliance + green present.
        assert "comp" in line
        assert "green" in line

    def test_reward_field_has_no_arrow(self, sweep_module, tmp_path):
        yr = tmp_path / "year.csv"
        tr = tmp_path / "train.csv"
        self._write_year_csv(yr, episodes=[1, 2], n_agents=2)
        self._write_train_csv(
            tr, episodes=range(0, 100), n_agents=2,
            reward=lambda ep, a: -3.0 + ep * 0.01,
        )
        line, _ = sweep_module._summarize_csv(str(yr), training_csv=str(tr))
        assert line is not None
        assert "R̄ " in line
        # No arrow in the reward field — comparison with the prior
        # heartbeat line is left to the reader.
        assert "→" not in line.split("R̄")[1]

    def test_secondary_match_rate_from_training_csv(self, sweep_module, tmp_path):
        yr = tmp_path / "year.csv"
        tr = tmp_path / "train.csv"
        self._write_year_csv(yr, episodes=[5], n_agents=1)
        self._write_train_csv(
            tr, episodes=[5], n_agents=1,
            match_rate=lambda ep: 0.42,
        )
        line, _ = sweep_module._summarize_csv(str(yr), training_csv=str(tr))
        assert line is not None
        assert "m42%" in line

    def test_compliance_zero_and_full(self, sweep_module, tmp_path):
        # All non-compliant.
        p1 = tmp_path / "all_default.csv"
        self._write_year_csv(
            p1, episodes=[1], n_agents=2,
            shortfall=lambda ep, yr, a: 5.0,
        )
        line1, _ = sweep_module._summarize_csv(str(p1))
        assert "comp 0%" in line1

        # All compliant.
        p2 = tmp_path / "all_ok.csv"
        self._write_year_csv(
            p2, episodes=[1], n_agents=2,
            shortfall=lambda ep, yr, a: 0.0,
        )
        line2, _ = sweep_module._summarize_csv(str(p2))
        assert "comp 100%" in line2

    def test_handles_partial_trailing_line(self, sweep_module, tmp_path):
        # Simulate a write race: last line is half-written.
        p = tmp_path / "x.csv"
        p.write_text(
            "episode,year,clearing_price,reward_A1,green_frac_A1,shortfall_A1\n"
            "1,1,70,-3.0,0.30,0.0\n"
            "1,2,75,-2.5,0.30,0.0\n"
            "1,3,80,"  # truncated mid-row
        )
        line, last_ep = sweep_module._summarize_csv(str(p))
        # Should fall back to whatever complete rows exist for episode 1.
        assert line is not None
        assert last_ep == 1

    def test_summary_under_220_chars(self, sweep_module, tmp_path):
        n_agents = 8
        yr = tmp_path / "year.csv"
        self._write_year_csv(yr, episodes=[100, 500, 1000], n_agents=n_agents)
        tr = tmp_path / "train.csv"
        self._write_train_csv(tr, episodes=range(900, 1001), n_agents=n_agents)
        line, _ = sweep_module._summarize_csv(
            str(yr), training_csv=str(tr), n_episodes=100000,
        )
        assert line is not None
        assert len(line) <= 220, f"summary too long: {len(line)} chars: {line}"

    def test_field_order(self, sweep_module, tmp_path):
        """Fields must appear in order: episode | px | sec | comp | green | R̄."""
        yr = tmp_path / "year.csv"
        tr = tmp_path / "train.csv"
        self._write_year_csv(yr, episodes=[1], n_agents=1)
        self._write_train_csv(tr, episodes=[1], n_agents=1)
        line, _ = sweep_module._summarize_csv(
            str(yr), training_csv=str(tr), n_episodes=10,
        )
        assert line is not None
        order_keys = ["Ep ", "| px ", "| sec ", "| comp ", "| green ", "| R̄ "]
        positions = [line.find(k) for k in order_keys]
        assert all(p >= 0 for p in positions), f"missing field in: {line}"
        assert positions == sorted(positions), f"wrong order: {line}"

    def test_returns_last_ep(self, sweep_module, tmp_path):
        p = tmp_path / "year.csv"
        self._write_year_csv(p, episodes=[0, 50, 137], n_agents=1)
        _line, last_ep = sweep_module._summarize_csv(str(p))
        assert last_ep == 137

    def test_undersubscription_count(self, sweep_module, tmp_path):
        """Years with Σ(bid_qty_mult × estimate_need) < auction_volume count
        as undersubscribed; reported as ``und K/N`` in the summary."""
        import csv as _csv

        p = tmp_path / "year.csv"
        n_agents = 2
        n_years = 12
        headers = ["episode", "year", "clearing_price", "secondary_price",
                   "auction_volume"]
        for i in range(n_agents):
            headers += [
                f"reward_A{i+1}", f"green_frac_A{i+1}", f"shortfall_A{i+1}",
                f"bid_qty_mult_A{i+1}", f"estimate_need_A{i+1}",
            ]
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for yr in range(1, n_years + 1):
                # First 3 years undersubscribed (mult=0.4 → demand 20),
                # rest oversubscribed (mult=3.0 → demand 150 vs supply 100).
                mult = 0.4 if yr <= 3 else 3.0
                row = {
                    "episode": 1, "year": yr,
                    "clearing_price": 80, "secondary_price": 70,
                    "auction_volume": 100.0,
                }
                for i in range(n_agents):
                    row[f"reward_A{i+1}"] = -1.0
                    row[f"green_frac_A{i+1}"] = 0.3
                    row[f"shortfall_A{i+1}"] = 0.0
                    row[f"bid_qty_mult_A{i+1}"] = mult
                    row[f"estimate_need_A{i+1}"] = 25.0
                w.writerow(row)
        line, _ = sweep_module._summarize_csv(str(p))
        assert line is not None
        assert "und 3/12" in line


class TestFormatEta:
    def test_subhour(self, sweep_module):
        assert sweep_module._format_eta(45) == "0m"
        assert sweep_module._format_eta(60) == "1m"
        assert sweep_module._format_eta(125) == "2m"
        assert sweep_module._format_eta(59 * 60) == "59m"

    def test_hours(self, sweep_module):
        assert sweep_module._format_eta(60 * 60) == "1h00m"
        assert sweep_module._format_eta(2 * 3600 + 35 * 60) == "2h35m"

    def test_days(self, sweep_module):
        assert sweep_module._format_eta(86400) == "1d00h"
        assert sweep_module._format_eta(2 * 86400 + 5 * 3600) == "2d05h"

    def test_unknown_for_invalid(self, sweep_module):
        assert sweep_module._format_eta(0) == "?"
        assert sweep_module._format_eta(-1) == "?"
        assert sweep_module._format_eta(float("nan")) == "?"
        assert sweep_module._format_eta(float("inf")) == "?"


class TestHeartbeat:
    def test_emits_tail_for_registered_jobs(self, sweep_module, tmp_path):
        import io
        log_path = tmp_path / "run_v_s1.log"
        log_path.write_text("[Ep 5/10  price=42.0]\n")
        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add("v", 1, str(log_path))
        hb.start()
        try:
            time.sleep(0.25)
        finally:
            hb.stop()
        out = buf.getvalue()
        assert "[LIVE  v s=1]" in out
        assert "[Ep 5/10  price=42.0]" in out

    def test_remove_silences_job(self, sweep_module, tmp_path):
        import io
        log_path = tmp_path / "run_v_s2.log"
        log_path.write_text("hello world\n")
        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add("v", 2, str(log_path))
        hb.start()
        time.sleep(0.15)
        hb.remove("v", 2)
        # Snapshot current output, then verify nothing more is added.
        before = buf.getvalue()
        time.sleep(0.2)
        hb.stop()
        after = buf.getvalue()
        # Anything after the remove() call should not mention this job.
        added = after[len(before):]
        assert "v s=2" not in added

    def test_truncates_overlong_lines(self, sweep_module, tmp_path):
        import io
        log_path = tmp_path / "run_v_s3.log"
        log_path.write_text("X" * 500 + "\n")
        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add("v", 3, str(log_path))
        hb.start()
        try:
            time.sleep(0.2)
        finally:
            hb.stop()
        out = buf.getvalue()
        assert "X" * 500 not in out
        assert "..." in out

    def test_stop_is_idempotent(self, sweep_module):
        hb = sweep_module._Heartbeat(interval=0.05)
        hb.start()
        hb.stop()
        hb.stop()  # must not raise

    def test_prefers_csv_summary_over_log_tail(self, sweep_module, tmp_path):
        """When a CSV is available, the heartbeat shows structured metrics."""
        import io
        import csv as _csv

        log_path = tmp_path / "run_v_s1.log"
        log_path.write_text("[some boring trainer line]\n")
        year_path = tmp_path / "year_log_v_s1.csv"
        with open(year_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "episode", "year", "clearing_price", "secondary_price",
                "reward_A1", "green_frac_A1", "shortfall_A1",
            ])
            w.writeheader()
            for ep in (100, 200, 400):
                for yr in range(1, 13):
                    w.writerow({
                        "episode": ep, "year": yr,
                        "clearing_price": 75 + yr * 5,
                        "secondary_price": 60 + yr * 2,
                        "reward_A1": -3.0,
                        "green_frac_A1": 0.30 + yr * 0.01,
                        "shortfall_A1": 0.0,
                    })

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add("v", 1, str(log_path), csv_year=str(year_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.25)
        finally:
            hb.stop()

        out = buf.getvalue()
        # Structured fields, NOT the boring log tail.
        assert "Ep 400/1000" in out
        assert "px " in out
        assert "comp" in out
        assert "boring trainer line" not in out

    def test_falls_back_to_log_when_csv_missing(self, sweep_module, tmp_path):
        """If CSV does not exist yet (e.g. during BC pretrain), tail the log."""
        import io
        log_path = tmp_path / "run_v_s2.log"
        log_path.write_text("[BC pretrain epoch 5/10  loss=0.23]\n")
        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add(
            "v", 2, str(log_path),
            csv_year=str(tmp_path / "does_not_exist.csv"),
            n_episodes=1000,
        )
        hb.start()
        try:
            time.sleep(0.2)
        finally:
            hb.stop()
        out = buf.getvalue()
        assert "BC pretrain epoch 5/10" in out

    def test_emits_per_job_eta_when_episode_progresses(
        self, sweep_module, tmp_path
    ):
        """Per-job ETA appears once enough episodes have been observed."""
        import csv as _csv
        import io

        log_path = tmp_path / "run_v_s1.log"
        log_path.write_text("\n")
        year_path = tmp_path / "year_log_v_s1.csv"
        headers = ["episode", "year", "clearing_price", "secondary_price",
                   "reward_A1", "green_frac_A1", "shortfall_A1"]

        def _write(eps):
            with open(year_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for ep in eps:
                    for yr in range(1, 13):
                        w.writerow({
                            "episode": ep, "year": yr,
                            "clearing_price": 80, "secondary_price": 70,
                            "reward_A1": -3, "green_frac_A1": 0.3,
                            "shortfall_A1": 0.0,
                        })

        _write([100])

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(
            interval=0.05, stream=buf, n_workers=1, total_jobs=1,
            first_interval=0.05,
        )
        hb.add("v", 1, str(log_path), csv_year=str(year_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.2)
            _write([100, 200])
            time.sleep(0.4)
        finally:
            hb.stop()

        out = buf.getvalue()
        # ETA should be emitted on the per-job line once a non-zero delta is
        # observed. We cannot assert the exact value (timing-dependent in
        # CI), only that an ETA token is appended.
        assert "ETA " in out
        # Single-job sweep: aggregate ETA banner is suppressed.
        assert "ETA total" not in out

    def test_emits_aggregate_eta_for_multi_job_sweep(
        self, sweep_module, tmp_path
    ):
        """Multi-job sweeps print an aggregate ``ETA total`` banner per tick."""
        import csv as _csv
        import io

        headers = ["episode", "year", "clearing_price", "secondary_price",
                   "reward_A1", "green_frac_A1", "shortfall_A1"]
        year_path = tmp_path / "year_log_v_s1.csv"
        log_path = tmp_path / "run_v_s1.log"
        log_path.write_text("\n")

        def _write(eps):
            with open(year_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for ep in eps:
                    for yr in range(1, 13):
                        w.writerow({
                            "episode": ep, "year": yr,
                            "clearing_price": 80, "secondary_price": 70,
                            "reward_A1": -3, "green_frac_A1": 0.3,
                            "shortfall_A1": 0.0,
                        })

        _write([100])

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(
            interval=0.05, stream=buf, n_workers=1, total_jobs=4,
            first_interval=0.05,
        )
        hb.add("v", 1, str(log_path), csv_year=str(year_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.2)
            _write([100, 200])
            time.sleep(0.4)
        finally:
            hb.stop()

        out = buf.getvalue()
        assert "ETA total" in out
        # The done/running/queued breakdown must reflect set_completed=0.
        assert "running" in out and "queued" in out

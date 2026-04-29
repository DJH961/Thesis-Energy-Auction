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
            + "----\n"
        )
        line = sweep_module._tail_last_meaningful_line(str(p))
        assert line == "[Ep 100/1000  reward=12.3]"


class TestSummarizeCsv:
    def _write_csv(self, path, rows, headers):
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def test_returns_none_for_missing_file(self, sweep_module, tmp_path):
        line, init, _last = sweep_module._summarize_csv(str(tmp_path / "nope.csv"))
        assert line is None
        assert init is None

    def test_returns_none_for_header_only(self, sweep_module, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("episode,clearing_price_last,reward_A1\n")
        line, _, _last = sweep_module._summarize_csv(str(p))
        assert line is None

    def test_basic_summary_shape(self, sweep_module, tmp_path):
        p = tmp_path / "x.csv"
        headers = [
            "episode", "clearing_price_last", "ep_mean_clearing_price",
            "reward_A1", "reward_A2",
            "green_frac_A1", "green_frac_A2",
            "shortfall_A1", "shortfall_A2",
        ]
        rows = []
        for ep in range(0, 1200, 100):
            rows.append({
                "episode": ep,
                "clearing_price_last": 75 + ep * 0.05,
                "ep_mean_clearing_price": 70 + ep * 0.04,
                "reward_A1": -3.5 + ep * 0.001,
                "reward_A2": -3.0 + ep * 0.001,
                "green_frac_A1": 0.30 + ep * 0.0001,
                "green_frac_A2": 0.32 + ep * 0.0001,
                "shortfall_A1": 0.0 if ep > 200 else 0.5,
                "shortfall_A2": 0.0,
            })
        self._write_csv(p, rows, headers)
        line, init, _last = sweep_module._summarize_csv(str(p), n_episodes=10000)
        assert line is not None
        assert "Ep 1100/10000" in line
        assert "%" in line  # progress percent
        assert "px 75→" in line  # initial→current arrow with rounded 75
        assert "R̄" in line
        assert "comp" in line  # compliance rate present
        assert "green" in line
        assert init is not None
        assert init["episode"] == "0"

    def test_handles_partial_trailing_line(self, sweep_module, tmp_path):
        # Simulate a write race: last line is half-written.
        p = tmp_path / "x.csv"
        p.write_text(
            "episode,clearing_price_last,reward_A1,green_frac_A1,shortfall_A1\n"
            "0,75,-3.0,0.30,0.0\n"
            "100,80,-2.5,0.35,0.0\n"
            "200,85,"  # truncated mid-row
        )
        line, _, _last = sweep_module._summarize_csv(str(p))
        assert line is not None
        # Should fall back to the last *complete* row (episode 100).
        assert "Ep 100" in line

    def test_compliance_rate_zero_and_full(self, sweep_module, tmp_path):
        headers = ["episode", "shortfall_A1", "shortfall_A2", "reward_A1", "reward_A2",
                   "clearing_price_last", "green_frac_A1", "green_frac_A2"]
        # All non-compliant.
        p1 = tmp_path / "all_default.csv"
        self._write_csv(p1, [
            {"episode": i, "shortfall_A1": 5.0, "shortfall_A2": 5.0,
             "reward_A1": -10.0, "reward_A2": -10.0,
             "clearing_price_last": 100, "green_frac_A1": 0.5, "green_frac_A2": 0.5}
            for i in range(50)
        ], headers)
        line1, _, _last = sweep_module._summarize_csv(str(p1))
        assert "comp 0%" in line1

        # All compliant.
        p2 = tmp_path / "all_ok.csv"
        self._write_csv(p2, [
            {"episode": i, "shortfall_A1": 0.0, "shortfall_A2": 0.0,
             "reward_A1": 1.0, "reward_A2": 1.0,
             "clearing_price_last": 100, "green_frac_A1": 0.5, "green_frac_A2": 0.5}
            for i in range(50)
        ], headers)
        line2, _, _last = sweep_module._summarize_csv(str(p2))
        assert "comp 100%" in line2

    def test_summary_under_200_chars(self, sweep_module, tmp_path):
        # With 8 agents (largest realistic shape), the summary still fits.
        n_agents = 8
        headers = ["episode", "clearing_price_last", "ep_mean_clearing_price"]
        for i in range(n_agents):
            headers += [f"reward_A{i+1}", f"green_frac_A{i+1}", f"shortfall_A{i+1}"]
        rows = []
        for ep in range(0, 5000, 100):
            row = {"episode": ep, "clearing_price_last": 100 + ep * 0.01,
                   "ep_mean_clearing_price": 95 + ep * 0.01}
            for i in range(n_agents):
                row[f"reward_A{i+1}"] = -2.0 + ep * 0.0001
                row[f"green_frac_A{i+1}"] = 0.40 + ep * 0.00005
                row[f"shortfall_A{i+1}"] = 0.0
            rows.append(row)
        p = tmp_path / "big.csv"
        self._write_csv(p, rows, headers)
        line, _, _last = sweep_module._summarize_csv(str(p), n_episodes=100000)
        assert line is not None
        assert len(line) <= 200, f"summary too long: {len(line)} chars: {line}"

    def test_initial_row_caching_anchors_arrow(self, sweep_module, tmp_path):
        # Even if the tail window does not contain row 0, the cached initial_row
        # should be respected to keep the "px X→Y" arrow stable across heartbeats.
        headers = ["episode", "clearing_price_last", "reward_A1",
                   "green_frac_A1", "shortfall_A1"]
        p = tmp_path / "x.csv"
        rows = [{"episode": i, "clearing_price_last": 50 + i * 0.5,
                 "reward_A1": -5.0 + i * 0.01,
                 "green_frac_A1": 0.20, "shortfall_A1": 0.0}
                for i in range(0, 3000)]
        self._write_csv(p, rows, headers)
        # Provide a synthetic "first-ever" cached row that differs from
        # whatever the tail window would pick.
        cached = {"episode": "0", "clearing_price_last": "10",
                  "reward_A1": "-99.0", "green_frac_A1": "0.05", "shortfall_A1": "0.0"}
        line, init, _last = sweep_module._summarize_csv(
            str(p), n_episodes=10000, initial_row=cached, tail_bytes=1024,
        )
        assert "px 10→" in line  # uses the cached initial value, not the tail
        assert init is cached  # returned unchanged for caller to keep caching

    def test_field_order_and_secondary_market(self, sweep_module, tmp_path):
        """Fields must appear in order: episode | px | sec | comp | green | R̄.

        Secondary-market columns (avg_price + match_rate) must be summarised
        when present, with format ``sec INIT→NOW (mXX%)``.
        """
        import csv as _csv
        p = tmp_path / "x.csv"
        headers = [
            "episode", "clearing_price_last", "ep_mean_clearing_price",
            "secondary_avg_price", "secondary_match_rate",
            "reward_A1", "green_frac_A1", "shortfall_A1",
        ]
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for ep in range(0, 500, 100):
                w.writerow({
                    "episode": ep,
                    "clearing_price_last": 75 + ep * 0.05,
                    "ep_mean_clearing_price": 70 + ep * 0.04,
                    "secondary_avg_price": 60 + ep * 0.03,
                    "secondary_match_rate": 0.40 + ep * 0.0001,
                    "reward_A1": -3.0 + ep * 0.001,
                    "green_frac_A1": 0.30 + ep * 0.0001,
                    "shortfall_A1": 0.0,
                })
        line, _, last_ep = sweep_module._summarize_csv(str(p), n_episodes=1000)
        assert line is not None
        assert last_ep == 400
        # Secondary market field present with both arrow and match-rate.
        assert "sec 60→" in line
        assert "(m" in line and "%)" in line
        # Required ordering: Ep < px < sec < comp < green < R̄.
        order_keys = ["Ep ", "| px ", "| sec ", "| comp ", "| green ", "| R̄ "]
        positions = [line.find(k) for k in order_keys]
        assert all(p >= 0 for p in positions), f"missing field in: {line}"
        assert positions == sorted(positions), f"wrong order: {line}"

    def test_returns_last_ep(self, sweep_module, tmp_path):
        import csv as _csv
        p = tmp_path / "x.csv"
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(
                f, fieldnames=["episode", "clearing_price_last", "reward_A1",
                               "green_frac_A1", "shortfall_A1"],
            )
            w.writeheader()
            for ep in (0, 50, 137):
                w.writerow({"episode": ep, "clearing_price_last": 80,
                            "reward_A1": -1.0, "green_frac_A1": 0.3,
                            "shortfall_A1": 0.0})
        _line, _init, last_ep = sweep_module._summarize_csv(str(p))
        assert last_ep == 137


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
        csv_path = tmp_path / "training_log_v_s1.csv"
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "episode", "clearing_price_last", "ep_mean_clearing_price",
                "reward_A1", "green_frac_A1", "shortfall_A1",
            ])
            w.writeheader()
            for ep in range(0, 500, 100):
                w.writerow({
                    "episode": ep, "clearing_price_last": 75 + ep * 0.1,
                    "ep_mean_clearing_price": 70 + ep * 0.1,
                    "reward_A1": -3.0 + ep * 0.001,
                    "green_frac_A1": 0.30 + ep * 0.0001,
                    "shortfall_A1": 0.0,
                })

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(interval=0.05, stream=buf)
        hb.add("v", 1, str(log_path), csv_path=str(csv_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.25)
        finally:
            hb.stop()

        out = buf.getvalue()
        # Structured fields, NOT the boring log tail.
        assert "Ep 400/1000" in out
        assert "px 75" in out
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
            csv_path=str(tmp_path / "does_not_exist.csv"),
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
        csv_path = tmp_path / "training_log_v_s1.csv"
        headers = ["episode", "clearing_price_last", "reward_A1",
                   "green_frac_A1", "shortfall_A1"]

        def _write(rows):
            with open(csv_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        # First snapshot: episode 100.
        _write([{"episode": 100, "clearing_price_last": 80, "reward_A1": -3,
                 "green_frac_A1": 0.3, "shortfall_A1": 0.0}])

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(
            interval=0.05, stream=buf, n_workers=1, total_jobs=1,
        )
        hb.add("v", 1, str(log_path), csv_path=str(csv_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.2)
            # Advance episode count so the heartbeat can compute a rate.
            _write([{"episode": 200, "clearing_price_last": 85, "reward_A1": -2,
                     "green_frac_A1": 0.32, "shortfall_A1": 0.0}])
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

        # Two jobs sharing one worker → queued > 0 for whichever isn't running.
        headers = ["episode", "clearing_price_last", "reward_A1",
                   "green_frac_A1", "shortfall_A1"]
        csv_path = tmp_path / "training_log_v_s1.csv"
        log_path = tmp_path / "run_v_s1.log"
        log_path.write_text("\n")

        def _write(rows):
            with open(csv_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        _write([{"episode": 100, "clearing_price_last": 80, "reward_A1": -3,
                 "green_frac_A1": 0.3, "shortfall_A1": 0.0}])

        buf = io.StringIO()
        hb = sweep_module._Heartbeat(
            interval=0.05, stream=buf, n_workers=1, total_jobs=4,
        )
        hb.add("v", 1, str(log_path), csv_path=str(csv_path), n_episodes=1000)
        hb.start()
        try:
            time.sleep(0.2)
            _write([{"episode": 200, "clearing_price_last": 85, "reward_A1": -2,
                     "green_frac_A1": 0.32, "shortfall_A1": 0.0}])
            time.sleep(0.4)
        finally:
            hb.stop()

        out = buf.getvalue()
        assert "ETA total" in out
        # The done/running/queued breakdown must reflect set_completed=0.
        assert "running" in out and "queued" in out

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

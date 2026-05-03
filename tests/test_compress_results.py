"""Tests for scripts/compress_results.py — focused on the new per-section /
total wall-clock timing instrumentation.

We don't exercise the full parquet/tarfile pipeline here (other test modules
already cover ``compress_logs`` / ``compress_checkpoints`` end-to-end). The
goal is just to lock in:

* ``_fmt_duration`` formats sub-second / minute / hour durations sensibly,
* ``main()`` prints a per-section timing line for both logs and checkpoints
  *and* a final ``total`` timing line on every run, even when there is
  nothing to compress.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "compress_results.py")


@pytest.fixture(scope="module")
def cr_module():
    spec = importlib.util.spec_from_file_location("scripts_compress_results", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    spec.loader.exec_module(mod)
    return mod


class TestFmtDuration:
    def test_subsecond_in_ms(self, cr_module):
        assert cr_module._fmt_duration(0.0) == "0ms"
        assert cr_module._fmt_duration(0.5) == "500ms"

    def test_seconds(self, cr_module):
        assert cr_module._fmt_duration(1.0) == "1.0s"
        assert cr_module._fmt_duration(45.7) == "45.7s"

    def test_minutes(self, cr_module):
        # 2m 05s — integer seconds, matching the hours-branch formatting.
        assert cr_module._fmt_duration(125.4) == "2m05s"

    def test_hours(self, cr_module):
        # 1h 02m 03s
        out = cr_module._fmt_duration(3600 + 2 * 60 + 3)
        assert out == "1h02m03s"


class TestMainPrintsTimings:
    def test_prints_section_and_total_timings_on_empty_tree(
        self, cr_module, tmp_path, capsys, monkeypatch
    ):
        """Even with nothing to compress, the script reports timings."""
        monkeypatch.setattr(sys, "argv", ["compress_results.py", str(tmp_path)])
        rc = cr_module.main()
        assert rc == 0
        out = capsys.readouterr().out
        # Both per-section banners are present.
        assert "logs total" in out
        assert "checkpoints total" in out
        # Final "total" line includes a combined wall-clock.
        assert "Total reclaimed" in out
        assert "total " in out  # part of the "(... = total Xs)" suffix

    def test_dry_run_does_not_print_timings(
        self, cr_module, tmp_path, capsys, monkeypatch
    ):
        """``--dry-run`` is a planning preview and exits before timing starts."""
        monkeypatch.setattr(
            sys, "argv", ["compress_results.py", str(tmp_path), "--dry-run"]
        )
        rc = cr_module.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total reclaimed" not in out

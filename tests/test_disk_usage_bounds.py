"""Tests for v8.5.5 training-output disk-usage caps.

Background
----------
Long Azure sweeps were hitting ``DiskFullError`` near the end of training
because two output paths grew without bound:

1. ``results/snapshots/`` accumulated one full copy of the cumulative
   ``training_log`` / ``year_log`` per ``snapshot_interval`` episodes,
   with episode-suffixed filenames → quadratic in episode count.
2. ``results/checkpoints_*/`` retained every periodic ``.pt`` file until
   clean exit because ``prune_checkpoints`` ran only at end-of-run.

These tests pin the new bounding behaviour:

* :func:`scripts.train.enforce_snapshot_retention` keeps only the
  ``keep_recent`` most-recent snapshot pairs for a given ``(tag_part,
  seed)`` and leaves unrelated files alone.
* :func:`scripts.train.prune_checkpoints` is invoked online so that
  during a long run the on-disk ``.pt`` count stays bounded.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from scripts.train import (  # noqa: E402  (path setup above)
    cleanup_snapshots_on_finish,
    enforce_snapshot_retention,
    prune_checkpoints,
)


def _touch(path: str) -> None:
    with open(path, "w") as f:
        f.write("")


# ---------------------------------------------------------------------------
# enforce_snapshot_retention
# ---------------------------------------------------------------------------


def test_enforce_snapshot_retention_keeps_only_most_recent(tmp_path):
    """Default keep_recent=1 leaves a single rolling pair per seed."""
    snap = tmp_path
    eps = [2500, 5000, 7500, 10000]
    for ep in eps:
        _touch(snap / f"training_log_s42_ep{ep}.csv")
        _touch(snap / f"year_log_s42_ep{ep}.csv")

    deleted = enforce_snapshot_retention(
        snap_dir=str(snap), tag_part="", seed=42, keep_recent=1
    )

    # 3 stale eps × 2 files per pair = 6 deletions.
    assert deleted == 6
    remaining = sorted(os.listdir(snap))
    assert remaining == [
        "training_log_s42_ep10000.csv",
        "year_log_s42_ep10000.csv",
    ]


def test_enforce_snapshot_retention_keep_recent_n(tmp_path):
    """keep_recent=K retains the K most-recent pairs."""
    snap = tmp_path
    eps = [1000, 2000, 3000, 4000, 5000]
    for ep in eps:
        _touch(snap / f"training_log_s7_ep{ep}.csv")
        _touch(snap / f"year_log_s7_ep{ep}.csv")

    enforce_snapshot_retention(
        snap_dir=str(snap), tag_part="", seed=7, keep_recent=2
    )

    remaining_eps = sorted(
        int(f.removeprefix("training_log_s7_ep").removesuffix(".csv"))
        for f in os.listdir(snap)
        if f.startswith("training_log_s7_ep")
    )
    assert remaining_eps == [4000, 5000]
    # Companion year_log files preserved for the same eps.
    for ep in remaining_eps:
        assert (snap / f"year_log_s7_ep{ep}.csv").exists()


def test_enforce_snapshot_retention_isolates_by_seed_and_tag(tmp_path):
    """Pruning a (tag, seed) pair must not touch other tags or seeds."""
    snap = tmp_path
    # Three groups co-located in the same snapshots dir.
    for ep in (1000, 2000, 3000):
        _touch(snap / f"training_log_s42_ep{ep}.csv")
        _touch(snap / f"year_log_s42_ep{ep}.csv")
        _touch(snap / f"training_log_s53_ep{ep}.csv")
        _touch(snap / f"year_log_s53_ep{ep}.csv")
        _touch(snap / f"training_log_msr_off_s42_ep{ep}.csv")
        _touch(snap / f"year_log_msr_off_s42_ep{ep}.csv")

    enforce_snapshot_retention(
        snap_dir=str(snap), tag_part="", seed=42, keep_recent=1
    )

    files = set(os.listdir(snap))
    # Target group: only ep3000 retained.
    assert "training_log_s42_ep3000.csv" in files
    assert "year_log_s42_ep3000.csv" in files
    assert "training_log_s42_ep1000.csv" not in files
    assert "training_log_s42_ep2000.csv" not in files
    # Other seed untouched.
    for ep in (1000, 2000, 3000):
        assert f"training_log_s53_ep{ep}.csv" in files
        assert f"year_log_s53_ep{ep}.csv" in files
    # Other tag untouched (tag_part="" must NOT match "_msr_off").
    for ep in (1000, 2000, 3000):
        assert f"training_log_msr_off_s42_ep{ep}.csv" in files


def test_enforce_snapshot_retention_with_run_tag(tmp_path):
    """tag_part='_msr_off' prunes only the matching tag's files."""
    snap = tmp_path
    for ep in (500, 1000, 1500):
        _touch(snap / f"training_log_msr_off_s5_ep{ep}.csv")
        _touch(snap / f"year_log_msr_off_s5_ep{ep}.csv")
        # Untagged sibling must survive.
        _touch(snap / f"training_log_s5_ep{ep}.csv")
        _touch(snap / f"year_log_s5_ep{ep}.csv")

    enforce_snapshot_retention(
        snap_dir=str(snap), tag_part="_msr_off", seed=5, keep_recent=1
    )

    files = set(os.listdir(snap))
    assert "training_log_msr_off_s5_ep1500.csv" in files
    assert "training_log_msr_off_s5_ep500.csv" not in files
    # Untagged group fully preserved.
    for ep in (500, 1000, 1500):
        assert f"training_log_s5_ep{ep}.csv" in files


def test_enforce_snapshot_retention_no_directory(tmp_path):
    """Missing directory is a no-op, not an error."""
    deleted = enforce_snapshot_retention(
        snap_dir=str(tmp_path / "does_not_exist"),
        tag_part="",
        seed=0,
        keep_recent=1,
    )
    assert deleted == 0


def test_enforce_snapshot_retention_clamps_keep_recent_to_one(tmp_path):
    """keep_recent < 1 is silently clamped to 1 to avoid wiping all history."""
    snap = tmp_path
    for ep in (100, 200):
        _touch(snap / f"training_log_s1_ep{ep}.csv")
        _touch(snap / f"year_log_s1_ep{ep}.csv")

    enforce_snapshot_retention(
        snap_dir=str(snap), tag_part="", seed=1, keep_recent=0
    )

    remaining = set(os.listdir(snap))
    # Most-recent pair survives.
    assert "training_log_s1_ep200.csv" in remaining
    assert "year_log_s1_ep200.csv" in remaining


# ---------------------------------------------------------------------------
# cleanup_snapshots_on_finish
# ---------------------------------------------------------------------------


def test_cleanup_snapshots_on_finish_deletes_all_pairs_for_seed(tmp_path):
    """Clean end-of-run wipes every snapshot pair for this seed/tag.

    The live cumulative training/year logs supersede mid-run snapshots
    once a run completes, so retaining them only wastes local disk and
    doubles the upload footprint when results/ is copied to Azure ML.
    """
    snap = tmp_path
    eps = [2500, 5000, 7500, 10000]
    for ep in eps:
        _touch(snap / f"training_log_s42_ep{ep}.csv")
        _touch(snap / f"year_log_s42_ep{ep}.csv")

    deleted = cleanup_snapshots_on_finish(
        snap_dir=str(snap), tag_part="", seed=42,
    )

    # All 4 pairs (8 files) gone.
    assert deleted == 8
    # Empty directory is removed too.
    assert not os.path.exists(str(snap))


def test_cleanup_snapshots_on_finish_isolates_other_seeds(tmp_path):
    """Other seeds' / tags' snapshots must not be touched."""
    snap = tmp_path
    # Target seed.
    for ep in (1000, 2000):
        _touch(snap / f"training_log_s1_ep{ep}.csv")
        _touch(snap / f"year_log_s1_ep{ep}.csv")
    # Different seed.
    _touch(snap / "training_log_s2_ep1000.csv")
    _touch(snap / "year_log_s2_ep1000.csv")
    # Different tag.
    _touch(snap / "training_log_msr_off_s1_ep1000.csv")
    _touch(snap / "year_log_msr_off_s1_ep1000.csv")

    cleanup_snapshots_on_finish(snap_dir=str(snap), tag_part="", seed=1)

    remaining = set(os.listdir(snap))
    # Target seed/tag wiped.
    assert "training_log_s1_ep1000.csv" not in remaining
    assert "year_log_s1_ep2000.csv" not in remaining
    # Other seed and tagged group preserved.
    assert "training_log_s2_ep1000.csv" in remaining
    assert "year_log_s2_ep1000.csv" in remaining
    assert "training_log_msr_off_s1_ep1000.csv" in remaining
    assert "year_log_msr_off_s1_ep1000.csv" in remaining
    # Directory not removed because other files remain.
    assert os.path.isdir(str(snap))


def test_cleanup_snapshots_on_finish_with_run_tag(tmp_path):
    """``tag_part`` selector only matches the requested tag group."""
    snap = tmp_path
    _touch(snap / "training_log_msr_off_s7_ep500.csv")
    _touch(snap / "year_log_msr_off_s7_ep500.csv")
    _touch(snap / "training_log_s7_ep500.csv")  # untagged group, keep
    _touch(snap / "year_log_s7_ep500.csv")

    cleanup_snapshots_on_finish(
        snap_dir=str(snap), tag_part="_msr_off", seed=7,
    )

    remaining = set(os.listdir(snap))
    assert "training_log_msr_off_s7_ep500.csv" not in remaining
    assert "year_log_msr_off_s7_ep500.csv" not in remaining
    assert "training_log_s7_ep500.csv" in remaining
    assert "year_log_s7_ep500.csv" in remaining


def test_cleanup_snapshots_on_finish_no_directory(tmp_path):
    """Missing snapshot directory is a no-op, not an error."""
    deleted = cleanup_snapshots_on_finish(
        snap_dir=str(tmp_path / "does_not_exist"),
        tag_part="",
        seed=0,
    )
    assert deleted == 0


# ---------------------------------------------------------------------------
# Online checkpoint pruning bounds peak file count.
# ---------------------------------------------------------------------------


def test_online_prune_checkpoints_bounds_peak_file_count(tmp_path):
    """Calling prune_checkpoints after each save keeps the dir bounded.

    Simulates a 100-save run with 8 agents and verifies that at no point
    after the first prune call does the file count exceed
    (n_keep_recent + n_keep_milestones + 1) × n_agents (the +1 covers the
    just-written most-recent checkpoint, which prune always keeps).
    """
    n_agents = 8
    n_keep_recent = 5
    n_keep_milestones = 20
    save_interval = 500
    n_saves = 100  # 50 000 episodes' worth

    bound = (n_keep_recent + n_keep_milestones) * n_agents
    peak = 0

    for s in range(n_saves):
        ep = s * save_interval
        for i in range(n_agents):
            _touch(tmp_path / f"agent_{i}_ep{ep}.pt")

        prune_checkpoints(
            ckpt_dir=str(tmp_path),
            n_agents=n_agents,
            n_keep_recent=n_keep_recent,
            n_keep_milestones=n_keep_milestones,
        )
        # Allow a small slack: prune_checkpoints may keep slightly more
        # than n_keep_milestones+n_keep_recent because the milestone-set
        # and recent-set can overlap or near-miss; we only require the
        # peak to stay O(1) in n_saves.
        peak = max(peak, len(os.listdir(tmp_path)))

    # Without online pruning, peak would be n_saves * n_agents = 800.
    # With online pruning, it must stay ≤ bound (small constant).
    assert peak <= bound, (
        f"peak={peak} files exceeded bound={bound}; "
        "online pruning is not bounding disk usage."
    )


def test_prune_checkpoints_preserves_best_files(tmp_path):
    """``agent_*_best.pt`` files are not touched by online pruning."""
    n_agents = 4
    for ep in (0, 500, 1000, 1500, 2000):
        for i in range(n_agents):
            _touch(tmp_path / f"agent_{i}_ep{ep}.pt")
    for i in range(n_agents):
        _touch(tmp_path / f"agent_{i}_best.pt")

    prune_checkpoints(
        ckpt_dir=str(tmp_path),
        n_agents=n_agents,
        n_keep_recent=1,
        n_keep_milestones=2,
    )

    best_files = sorted(
        f for f in os.listdir(tmp_path) if f.endswith("_best.pt")
    )
    assert best_files == [f"agent_{i}_best.pt" for i in range(n_agents)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

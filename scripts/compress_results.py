#!/usr/bin/env python
"""
compress_results.py — losslessly shrink a results tree.

Walks a directory and:

* converts every ``training_log_*.csv`` / ``year_log_*.csv`` into
  zstd-compressed parquet (typically 5–10× smaller). The source CSV is
  deleted only after the parquet's row count is verified against the CSV.
* bundles every ``checkpoints_*/`` directory into a single
  ``checkpoints_*.tar.xz`` archive (xz is stdlib, lossless, ~30–50 % of
  the source size on float weights). The source directory is deleted
  only after the archive is on disk.

Both steps are lossless modulo the documented float64→float32 / int64→int32
downcast applied by the parquet writer (see ``src/utils/run_data.py``).

Usage::

    # Migrate one sweep
    python scripts/compress_results.py results/sweeps/thesis_experiments

    # Dry-run first to see what would happen
    python scripts/compress_results.py results --dry-run

    # Keep the original CSV / checkpoint dir alongside the compressed copy
    python scripts/compress_results.py results --keep-source

This is the script to run *once* to migrate the 20 GB of existing CSVs and
2.5 GB of checkpoints to their compressed form. Future training runs do
this automatically at clean end-of-run via
``logging.compress_on_finish.{logs, checkpoints}`` (default true).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make the src/ package importable when this script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.utils.run_data import (  # noqa: E402
    cache_path_for,
    compress_checkpoints,
    compress_logs,
)

LOG_PREFIXES = ("training_log_", "year_log_", "ql_training_log_")


def _find_log_csvs(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(".csv"):
                continue
            if not any(f.startswith(p) for p in LOG_PREFIXES):
                continue
            out.append(os.path.join(dirpath, f))
    return sorted(out)


def _find_checkpoint_dirs(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames:
            if d.startswith("checkpoints"):
                # Skip dirs that are already siblings of an existing archive
                # (re-running compress_results should be a no-op).
                full = os.path.join(dirpath, d)
                archive = full.rstrip(os.sep) + ".tar.xz"
                if not os.path.exists(archive):
                    out.append(full)
    return sorted(out)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_duration(seconds: float) -> str:
    """Format a wall-clock duration as a short human-readable string.

    Sub-second durations are rendered in milliseconds; otherwise the output
    is ``XmYY.Zs`` or ``HhMMmSSs`` so that per-section and total timings are
    easy to scan at a glance.
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly compress an ETS MARL results tree."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="results",
        help="Directory to walk (default: 'results').",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the source CSVs / checkpoint dirs alongside the "
        "compressed copies. Default deletes them once each conversion is "
        "verified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without modifying any files.",
    )
    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Skip checkpoint compression.",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Skip log compression.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"compress_results: not a directory: {args.root!r}", file=sys.stderr)
        return 2
    if args.logs_only and args.checkpoints_only:
        print(
            "compress_results: --logs-only and --checkpoints-only are "
            "mutually exclusive",
            file=sys.stderr,
        )
        return 2

    do_logs = not args.checkpoints_only
    do_ckpts = not args.logs_only

    csvs = _find_log_csvs(args.root) if do_logs else []
    ckpts = _find_checkpoint_dirs(args.root) if do_ckpts else []

    print(f"Scanning {args.root!r}")
    print(f"  log CSVs to compress    : {len(csvs)}")
    print(f"  checkpoint dirs to bundle: {len(ckpts)}")

    if args.dry_run:
        for p in csvs:
            print(f"  [LOG ] {p} → {cache_path_for(p)}")
        for d in ckpts:
            print(f"  [CKPT] {d} → {d.rstrip(os.sep)}.tar.xz")
        return 0

    run_t0 = time.perf_counter()

    # ----- logs ---------------------------------------------------------
    logs_t0 = time.perf_counter()
    total_csv_bytes = 0
    total_pq_bytes = 0
    for csv_path in csvs:
        item_t0 = time.perf_counter()
        try:
            csv_size_before = os.path.getsize(csv_path)
            entries = compress_logs([csv_path], delete_csv=not args.keep_source)
        except Exception as e:
            print(f"  FAIL {csv_path}: {e}", file=sys.stderr)
            continue
        if not entries:
            continue
        e = entries[0]
        pq_size = os.path.getsize(e.parquet_path)
        total_csv_bytes += csv_size_before
        total_pq_bytes += pq_size
        ratio = pq_size / max(csv_size_before, 1) * 100
        print(
            f"  LOG  {csv_path}: "
            f"{_human(csv_size_before)} → {_human(pq_size)} ({ratio:.1f}%) "
            f"[{_fmt_duration(time.perf_counter() - item_t0)}]"
        )

    logs_elapsed = time.perf_counter() - logs_t0
    if csvs:
        ratio = total_pq_bytes / max(total_csv_bytes, 1) * 100
        print(
            f"  → logs total: {_human(total_csv_bytes)} → "
            f"{_human(total_pq_bytes)} ({ratio:.1f}%) "
            f"in {_fmt_duration(logs_elapsed)} "
            f"({len(csvs)} file{'s' if len(csvs) != 1 else ''})"
        )
    elif do_logs:
        print(f"  → logs total: nothing to do [{_fmt_duration(logs_elapsed)}]")

    # ----- checkpoints --------------------------------------------------
    ckpts_t0 = time.perf_counter()
    total_ckpt_bytes = 0
    total_archive_bytes = 0
    for ckpt_dir in ckpts:
        item_t0 = time.perf_counter()
        try:
            size_before = sum(
                os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(ckpt_dir)
                for f in fs
            )
            archive = compress_checkpoints(
                ckpt_dir, delete_dir=not args.keep_source
            )
        except Exception as e:
            print(f"  FAIL {ckpt_dir}: {e}", file=sys.stderr)
            continue
        if archive is None:
            continue
        size_after = os.path.getsize(archive)
        total_ckpt_bytes += size_before
        total_archive_bytes += size_after
        ratio = size_after / max(size_before, 1) * 100
        print(
            f"  CKPT {ckpt_dir}: "
            f"{_human(size_before)} → {_human(size_after)} ({ratio:.1f}%) "
            f"[{_fmt_duration(time.perf_counter() - item_t0)}]"
        )

    ckpts_elapsed = time.perf_counter() - ckpts_t0
    if ckpts:
        ratio = total_archive_bytes / max(total_ckpt_bytes, 1) * 100
        print(
            f"  → checkpoints total: {_human(total_ckpt_bytes)} → "
            f"{_human(total_archive_bytes)} ({ratio:.1f}%) "
            f"in {_fmt_duration(ckpts_elapsed)} "
            f"({len(ckpts)} dir{'s' if len(ckpts) != 1 else ''})"
        )
    elif do_ckpts:
        print(
            f"  → checkpoints total: nothing to do "
            f"[{_fmt_duration(ckpts_elapsed)}]"
        )

    saved = (total_csv_bytes - total_pq_bytes) + (
        total_ckpt_bytes - total_archive_bytes
    )
    total_elapsed = time.perf_counter() - run_t0
    print(
        f"Total reclaimed: {_human(saved)} "
        f"(logs {_fmt_duration(logs_elapsed)} + "
        f"checkpoints {_fmt_duration(ckpts_elapsed)} = "
        f"total {_fmt_duration(total_elapsed)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

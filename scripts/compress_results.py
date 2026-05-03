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

# Empirical throughput priors (bytes/sec of *input*) used for the upfront
# ETA when no measurements exist yet. Calibrated against recent compression
# runs on this codebase; conservative on purpose so the first guess errs
# on the slow side rather than overpromising. Updated *online* from
# observed throughput once each section starts producing data, so the ETA
# converges away from these priors after a few items.
_LOGS_BYTES_PER_SEC_PRIOR = 60 * 1024 * 1024     # ~60 MB/s CSV → parquet
_CKPTS_BYTES_PER_SEC_PRIOR = 8 * 1024 * 1024     # ~8 MB/s dir → tar.xz


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

    # Pre-walk file sizes so we can give an honest upfront ETA *before*
    # touching any data, and a rolling forward ETA on each per-item line.
    # Sizes here are uncompressed / source bytes — they are the work, not
    # the output, so they pair naturally with bytes-per-second throughputs.
    csv_sizes: list[int] = [os.path.getsize(p) for p in csvs]
    ckpt_sizes: list[int] = [
        sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, fs in os.walk(ck)
            for f in fs
        )
        for ck in ckpts
    ]
    total_csv_input = sum(csv_sizes)
    total_ckpt_input = sum(ckpt_sizes)

    print(f"Scanning {args.root!r}")
    print(
        f"  log CSVs to compress    : {len(csvs)} "
        f"({_human(total_csv_input)})"
    )
    print(
        f"  checkpoint dirs to bundle: {len(ckpts)} "
        f"({_human(total_ckpt_input)})"
    )

    if args.dry_run:
        for p in csvs:
            print(f"  [LOG ] {p} → {cache_path_for(p)}")
        for d in ckpts:
            print(f"  [CKPT] {d} → {d.rstrip(os.sep)}.tar.xz")
        return 0

    # Upfront forward-looking ETA — based on size-weighted prior throughput
    # so the user sees a *when-will-this-be-done* number before any item
    # has been processed. Refined per-item below as actual throughput is
    # observed, and backfilled into a final summary at the end.
    upfront_logs = (
        total_csv_input / _LOGS_BYTES_PER_SEC_PRIOR if total_csv_input else 0.0
    )
    upfront_ckpts = (
        total_ckpt_input / _CKPTS_BYTES_PER_SEC_PRIOR if total_ckpt_input else 0.0
    )
    upfront_total = upfront_logs + upfront_ckpts
    if upfront_total > 0:
        parts = []
        if upfront_logs > 0:
            parts.append(f"logs ~{_fmt_duration(upfront_logs)}")
        if upfront_ckpts > 0:
            parts.append(f"checkpoints ~{_fmt_duration(upfront_ckpts)}")
        print(
            f"  Estimated total time   : ~{_fmt_duration(upfront_total)} "
            f"({' + '.join(parts)}; refined as throughput is measured)"
        )

    run_t0 = time.perf_counter()

    # ----- logs ---------------------------------------------------------
    logs_t0 = time.perf_counter()
    total_csv_bytes = 0
    total_pq_bytes = 0
    bytes_done_so_far = 0
    for idx, csv_path in enumerate(csvs):
        item_t0 = time.perf_counter()
        try:
            csv_size_before = csv_sizes[idx]
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
        bytes_done_so_far += csv_size_before
        ratio = pq_size / max(csv_size_before, 1) * 100
        # Rolling forward ETA: extrapolate remaining bytes at the
        # throughput we've observed so far in this section.
        section_elapsed = time.perf_counter() - logs_t0
        bytes_left = max(0, total_csv_input - bytes_done_so_far)
        if section_elapsed > 0 and bytes_done_so_far > 0 and bytes_left > 0:
            sec_per_byte = section_elapsed / bytes_done_so_far
            eta_section = bytes_left * sec_per_byte
            eta_str = f"  → ETA {_fmt_duration(eta_section)} remaining"
        elif bytes_left == 0:
            eta_str = "  → ETA 0s (done)"
        else:
            eta_str = ""
        print(
            f"  LOG  {csv_path}: "
            f"{_human(csv_size_before)} → {_human(pq_size)} ({ratio:.1f}%) "
            f"[{_fmt_duration(time.perf_counter() - item_t0)}]"
            f"{eta_str}"
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
    bytes_done_so_far = 0
    for idx, ckpt_dir in enumerate(ckpts):
        item_t0 = time.perf_counter()
        try:
            size_before = ckpt_sizes[idx]
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
        bytes_done_so_far += size_before
        ratio = size_after / max(size_before, 1) * 100
        section_elapsed = time.perf_counter() - ckpts_t0
        bytes_left = max(0, total_ckpt_input - bytes_done_so_far)
        if section_elapsed > 0 and bytes_done_so_far > 0 and bytes_left > 0:
            sec_per_byte = section_elapsed / bytes_done_so_far
            eta_section = bytes_left * sec_per_byte
            eta_str = f"  → ETA {_fmt_duration(eta_section)} remaining"
        elif bytes_left == 0:
            eta_str = "  → ETA 0s (done)"
        else:
            eta_str = ""
        print(
            f"  CKPT {ckpt_dir}: "
            f"{_human(size_before)} → {_human(size_after)} ({ratio:.1f}%) "
            f"[{_fmt_duration(time.perf_counter() - item_t0)}]"
            f"{eta_str}"
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

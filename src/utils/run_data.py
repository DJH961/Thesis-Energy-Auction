"""
run_data.py
===========
Persistent parquet cache for ETS MARL training/year logs.

Why this exists
---------------
A full default-config seed writes ``training_log_*.csv`` (~10–30 MB,
``n_episodes`` rows × ~120 columns) and ``year_log_*.csv`` (multi-GB,
``n_episodes × 12`` rows × ~270–414 columns). Loading a single sweep with
4 seeds via ``pd.read_csv`` already costs >4 GiB of RAM (pandas defaults to
float64), and a thesis with 4–5 sweeps × 16 (variant, seed) cells turns
each notebook restart into a 20+ minute reload.

This module gives notebooks a one-line, low-memory replacement::

    from src.utils.run_data import load_run, load_sweep

    ep_df, yr_df = load_run(training_log_path, year_log_path)

What it does
------------
* On first call, the CSV is parsed in chunks, dtype-downcast (float64 →
  float32 and int64 → int32 where the values fit) and written next to the
  CSV as ``<name>.parquet`` with zstd compression. This is typically 5–10×
  smaller and 5–20× faster to reload than the source CSV.
* On subsequent calls, the parquet file is read directly via pyarrow.
* The cache key is ``(absolute_path, source_size, source_mtime_ns)`` stored
  inside the parquet metadata. If the source CSV changes the parquet is
  silently rebuilt.
* Optional ``columns=`` projection lets each notebook load only the few
  fields it actually plots — at parquet level this is essentially free
  (the rest is never decompressed) and finally makes 16-cell sweeps fit in
  notebook RAM.
* The cache survives across notebook kernels and across notebooks, so
  starting work in a new notebook is instant once the first one has
  populated the cache.

Design notes
------------
* No dependency on the rest of the package — this module only needs
  ``pandas`` and ``pyarrow`` so it can also be imported by lightweight
  analysis scripts.
* The training log is too small for a chunked read to matter, but the year
  log has been observed at >5 GB CSV / 1.44 M rows. We chunk both for
  uniformity; the chunk size is tuned so the in-flight DataFrame stays
  comfortably below 1 GiB even on float64 input.
* All numeric downcasts are *lossless* (range check + ``can_cast``). The
  ``episode``/``year`` columns stay int32 which is plenty for any
  realistic run length.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "src.utils.run_data requires pyarrow. Install via "
        "`pip install pyarrow` or `uv sync` (it is listed as a project "
        "dependency)."
    ) from e


__all__ = [
    "CacheEntry",
    "load_run_csv",
    "load_run",
    "load_sweep",
    "rebuild_cache",
    "cache_path_for",
]

_log = logging.getLogger(__name__)

# Default chunk size for CSV → parquet conversion. ~250k year-log rows of
# float64×270 cols ≈ 540 MiB in flight; we downcast to float32 immediately so
# steady-state RAM is roughly half of that.
_DEFAULT_CHUNKSIZE = 250_000

# Metadata keys embedded in the parquet file footer for staleness detection.
# Stored as bytes (parquet metadata is bytes-only).
_META_SOURCE_PATH = b"ets_marl.source_path"
_META_SOURCE_SIZE = b"ets_marl.source_size"
_META_SOURCE_MTIME_NS = b"ets_marl.source_mtime_ns"
_META_SCHEMA_VERSION = b"ets_marl.schema_version"

# Bump if the dtype-downcast / parquet-write logic changes in a way that
# requires existing caches to be rebuilt.
_SCHEMA_VERSION = b"1"

# Process-local lock so two threads in the same notebook don't race on
# building the same parquet. Cross-process safety is best-effort: parquet
# writes go to a tempfile and are atomically renamed.
_BUILD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """Result of a successful CSV→parquet conversion or cache hit."""

    csv_path: str
    parquet_path: str
    rows: int
    columns: int
    source_size: int
    source_mtime_ns: int
    rebuilt: bool  # True if we just (re)built the parquet on this call


def cache_path_for(csv_path: str) -> str:
    """Return the parquet sibling path for a given CSV path.

    A ``training_log_default_s42.csv`` file becomes
    ``training_log_default_s42.parquet`` in the same directory. We keep the
    cache next to the source so the user only has to manage one results
    tree, and so removing a sweep directory cleans up its cache too.
    """
    if not csv_path:
        raise ValueError("csv_path must be a non-empty string")
    base, _ = os.path.splitext(csv_path)
    return base + ".parquet"


def _stat_key(path: str) -> tuple[int, int]:
    st = os.stat(path)
    # mtime_ns is more precise than mtime and lets us detect rapid rewrites.
    return int(st.st_size), int(st.st_mtime_ns)


def _read_metadata(parquet_path: str) -> dict[bytes, bytes]:
    """Return the user-defined parquet metadata as a {bytes: bytes} dict."""
    meta = pq.read_metadata(parquet_path)
    kv = meta.metadata or {}
    return dict(kv)


def _is_cache_fresh(csv_path: str, parquet_path: str) -> bool:
    """True iff parquet exists and matches (size, mtime, schema)."""
    if not os.path.exists(parquet_path):
        return False
    try:
        meta = _read_metadata(parquet_path)
    except Exception as e:  # corrupt parquet → rebuild
        _log.warning("parquet cache unreadable (%s); rebuilding: %s", parquet_path, e)
        return False
    if meta.get(_META_SCHEMA_VERSION) != _SCHEMA_VERSION:
        return False
    try:
        size, mtime_ns = _stat_key(csv_path)
    except FileNotFoundError:
        # Source CSV gone but parquet exists — treat as fresh so users can
        # delete CSVs after caching to reclaim disk space. This is the
        # explicit on-disk-space workflow we want to support.
        return True
    if meta.get(_META_SOURCE_SIZE) != str(size).encode():
        return False
    if meta.get(_META_SOURCE_MTIME_NS) != str(mtime_ns).encode():
        return False
    return True


# ---------------------------------------------------------------------------
# Dtype downcasting
# ---------------------------------------------------------------------------


def _downcast_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Lossless float64→float32 / int64→int32 downcast for numeric columns.

    * float64 → float32 unconditionally. The values logged here (rewards,
      prices in EUR/t, allowance volumes in Mt) are domain-bounded well
      inside float32 range, and float32 NaN / ±inf semantics match
      float64. We don't range-check on the hot path; if a future log
      column starts emitting >3.4e38 values, switch back to float64 for
      that column.
    * int64 → int32 if min/max fit. The training log's episode counter
      tops out around 10⁵–10⁶ which fits comfortably.
    * Object / string columns are left untouched.
    """
    for col in df.columns:
        s = df[col]
        if s.dtype == np.float64:
            df[col] = s.astype(np.float32)
        elif s.dtype == np.int64:
            # Compute min/max ignoring NaN (int columns have no NaN, but be
            # defensive in case pandas inferred Int64 nullable upstream).
            try:
                mn = s.min()
                mx = s.max()
            except (TypeError, ValueError):
                continue
            if pd.notna(mn) and pd.notna(mx) and mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                df[col] = s.astype(np.int32)
    return df


# ---------------------------------------------------------------------------
# CSV → parquet conversion
# ---------------------------------------------------------------------------


def _build_parquet(
    csv_path: str,
    parquet_path: str,
    *,
    chunksize: int,
    compression: str,
) -> CacheEntry:
    """Stream a CSV into a parquet file with downcasting + zstd compression.

    The parquet is written to a sibling ``.tmp`` file and atomically renamed
    on success, so a crashed/interrupted build never leaves a corrupt cache
    that future calls would mistakenly accept.
    """
    size, mtime_ns = _stat_key(csv_path)
    tmp_path = parquet_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    writer: pq.ParquetWriter | None = None
    rows = 0
    n_cols = 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            _downcast_inplace(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                # Embed the cache key in the parquet metadata so a later
                # process can decide whether to rebuild without re-reading
                # the CSV.
                schema = table.schema.with_metadata(
                    {
                        _META_SOURCE_PATH: os.path.abspath(csv_path).encode(),
                        _META_SOURCE_SIZE: str(size).encode(),
                        _META_SOURCE_MTIME_NS: str(mtime_ns).encode(),
                        _META_SCHEMA_VERSION: _SCHEMA_VERSION,
                    }
                )
                writer = pq.ParquetWriter(tmp_path, schema, compression=compression)
                # Re-cast the first chunk against the metadata-tagged schema.
                table = table.cast(schema)
                n_cols = len(schema.names)
            else:
                # Subsequent chunks: align to the writer's schema (cheap).
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # CSV was empty (just a header). Write an empty parquet with the
        # right schema so downstream readers don't crash.
        empty = pd.read_csv(csv_path, nrows=0)
        _downcast_inplace(empty)
        table = pa.Table.from_pandas(empty, preserve_index=False)
        schema = table.schema.with_metadata(
            {
                _META_SOURCE_PATH: os.path.abspath(csv_path).encode(),
                _META_SOURCE_SIZE: str(size).encode(),
                _META_SOURCE_MTIME_NS: str(mtime_ns).encode(),
                _META_SCHEMA_VERSION: _SCHEMA_VERSION,
            }
        )
        pq.write_table(table.cast(schema), tmp_path, compression=compression)
        n_cols = len(schema.names)

    os.replace(tmp_path, parquet_path)
    return CacheEntry(
        csv_path=csv_path,
        parquet_path=parquet_path,
        rows=rows,
        columns=n_cols,
        source_size=size,
        source_mtime_ns=mtime_ns,
        rebuilt=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_run_csv(
    csv_path: str | os.PathLike | None,
    *,
    columns: Sequence[str] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
    compression: str = "zstd",
    rebuild: bool = False,
) -> pd.DataFrame | None:
    """Load one training-log / year-log CSV via a persistent parquet cache.

    Parameters
    ----------
    csv_path
        Path to the source CSV. ``None`` returns ``None`` (so notebook code
        can stay declarative even when one of the log files is missing).
    columns
        Optional iterable of column names to load. Pushed down into the
        parquet reader, so unused columns are never decompressed. Unknown
        column names are silently dropped so notebooks can request a
        superset across runs whose schemas have diverged over time.
    chunksize
        CSV-read chunk size when (re)building the cache. Default 250k rows.
    compression
        Parquet compression codec. Defaults to zstd (best size/speed trade
        for these logs).
    rebuild
        Force rebuilding the parquet even if the cache key matches.

    Returns
    -------
    pandas.DataFrame | None
        The loaded frame, with float64 columns downcast to float32 and int64
        downcast to int32 where lossless. Returns ``None`` iff
        ``csv_path is None``.
    """
    if csv_path is None:
        return None
    csv_path = os.fspath(csv_path)
    if not csv_path:
        return None

    parquet_path = cache_path_for(csv_path)
    csv_exists = os.path.exists(csv_path)
    pq_exists = os.path.exists(parquet_path)

    if not csv_exists and not pq_exists:
        raise FileNotFoundError(
            f"Neither CSV nor cached parquet found for {csv_path!r}"
        )

    needs_build = rebuild or not _is_cache_fresh(csv_path, parquet_path)
    if needs_build and not csv_exists:
        # Cache was stale but source CSV is gone — fall back to the existing
        # parquet rather than failing. This makes "delete CSVs after first
        # cache" a legitimate disk-space workflow.
        _log.info(
            "Source CSV missing for %s; using stale parquet cache", csv_path
        )
        needs_build = False

    if needs_build:
        with _BUILD_LOCK:
            # Re-check under the lock (another thread may have built it).
            if rebuild or not _is_cache_fresh(csv_path, parquet_path):
                _log.info("Building parquet cache: %s -> %s", csv_path, parquet_path)
                _build_parquet(
                    csv_path,
                    parquet_path,
                    chunksize=chunksize,
                    compression=compression,
                )

    cols = list(columns) if columns is not None else None
    if cols is not None:
        # Filter to columns that actually exist; silently drop unknowns so
        # notebooks can ask for a superset across heterogeneous schemas
        # (e.g. older runs that pre-date a new diagnostic column).
        available = set(pq.read_schema(parquet_path).names)
        cols = [c for c in cols if c in available]
        if not cols:
            # Asked for a column set that's entirely absent — return an
            # empty frame rather than crashing pyarrow.
            return pd.DataFrame()

    table = pq.read_table(parquet_path, columns=cols)
    return table.to_pandas()


def load_run(
    training_log: str | os.PathLike | None,
    year_log: str | os.PathLike | None,
    *,
    ep_columns: Sequence[str] | None = None,
    yr_columns: Sequence[str] | None = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
    rebuild: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the (training_log, year_log) pair for a single run.

    Convenience wrapper around :func:`load_run_csv` that mirrors the
    ``(ep_df, yr_df)`` tuple returned by the notebook helpers. Either path
    may be ``None`` to indicate that file is unavailable for this run.
    """
    ep_df = load_run_csv(
        training_log, columns=ep_columns, chunksize=chunksize, rebuild=rebuild
    )
    yr_df = load_run_csv(
        year_log, columns=yr_columns, chunksize=chunksize, rebuild=rebuild
    )
    return ep_df, yr_df


def rebuild_cache(csv_paths: Iterable[str], *, chunksize: int = _DEFAULT_CHUNKSIZE) -> list[CacheEntry]:
    """Force-rebuild the parquet cache for an iterable of CSV paths.

    Useful as a one-shot "warm the cache" cell at the top of a notebook,
    so the slow CSV → parquet conversion happens once and every later cell
    (and every later notebook) hits parquet directly.
    """
    out: list[CacheEntry] = []
    for p in csv_paths:
        if p is None:
            continue
        parquet_path = cache_path_for(p)
        with _BUILD_LOCK:
            entry = _build_parquet(
                p, parquet_path, chunksize=chunksize, compression="zstd"
            )
        out.append(entry)
    return out


def load_sweep(
    spec_path: str | os.PathLike,
    *,
    ep_columns: Sequence[str] | None = None,
    yr_columns: Sequence[str] | None = None,
    base_config_path: str | os.PathLike | None = None,
) -> dict[tuple[str, int], dict]:
    """Discover and load every (variant, seed) run for a sweep spec.

    The returned dict is keyed by ``(variant_name, seed)`` and each value
    has shape::

        {
            "ep_df": pd.DataFrame | None,
            "yr_df": pd.DataFrame | None,
            "training_log": str | None,
            "year_log": str | None,
            "config": dict,
            "results_dir": str,
        }

    Missing log files are tolerated — the corresponding ``*_df`` entry is
    ``None`` and the path is reported as ``None``. This matches the
    behaviour of the in-notebook discovery helpers but routes every load
    through the parquet cache, so reopening the notebook (or opening a
    second notebook) is near-instant.
    """
    import yaml

    from .sweep import build_jobs, load_sweep_spec  # local import to avoid cycles

    spec_path = os.fspath(spec_path)
    spec = load_sweep_spec(spec_path)
    base_path = (
        os.fspath(base_config_path)
        if base_config_path is not None
        else spec["base_config"]
    )
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    jobs = build_jobs(spec, base_cfg)

    out: dict[tuple[str, int], dict] = {}
    for job in jobs:
        variant = job.variant_name
        seed = job.seed
        # Look for both tagged and untagged filename layouts (run_tag is
        # optional; the sweep launcher uses tagged names by default).
        candidates = [
            (
                os.path.join(job.results_dir, f"training_log_{variant}_s{seed}.csv"),
                os.path.join(job.results_dir, f"year_log_{variant}_s{seed}.csv"),
            ),
            (
                os.path.join(job.results_dir, f"training_log_s{seed}.csv"),
                os.path.join(job.results_dir, f"year_log_s{seed}.csv"),
            ),
        ]
        tlog: str | None = None
        ylog: str | None = None
        for tp, yp in candidates:
            tp_ok = os.path.exists(tp) or os.path.exists(cache_path_for(tp))
            yp_ok = os.path.exists(yp) or os.path.exists(cache_path_for(yp))
            if tp_ok or yp_ok:
                tlog = tp if tp_ok else None
                ylog = yp if yp_ok else None
                break

        ep_df = load_run_csv(tlog, columns=ep_columns) if tlog else None
        yr_df = load_run_csv(ylog, columns=yr_columns) if ylog else None
        out[(variant, seed)] = {
            "ep_df": ep_df,
            "yr_df": yr_df,
            "training_log": tlog,
            "year_log": ylog,
            "config": job.config,
            "results_dir": job.results_dir,
        }
    return out

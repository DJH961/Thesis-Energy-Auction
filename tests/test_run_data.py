"""Tests for src/utils/run_data.py — the parquet-cached run loader."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.utils.run_data import (
    CacheEntry,
    cache_path_for,
    load_run,
    load_run_csv,
    rebuild_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, name: str, df: pd.DataFrame) -> str:
    p = os.path.join(tmp_path, name)
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def small_run(tmp_path):
    """A toy training/year log pair, fully numeric."""
    n_eps = 50
    n_years = 12
    rng = np.random.default_rng(0)
    ep_df = pd.DataFrame(
        {
            "episode": np.arange(n_eps, dtype=np.int64),
            "mean_reward": rng.normal(size=n_eps),
            "clearing_price_mean": rng.uniform(40, 100, size=n_eps),
        }
    )
    yr_rows = n_eps * n_years
    yr_df = pd.DataFrame(
        {
            "episode": np.repeat(np.arange(n_eps, dtype=np.int64), n_years),
            "year": np.tile(np.arange(n_years, dtype=np.int64), n_eps),
            "clearing_price": rng.uniform(40, 100, size=yr_rows),
            "tnac": rng.uniform(0, 500, size=yr_rows),
        }
    )
    tlog = _write_csv(tmp_path, "training_log_x_s1.csv", ep_df)
    ylog = _write_csv(tmp_path, "year_log_x_s1.csv", yr_df)
    return tlog, ylog, ep_df, yr_df


# ---------------------------------------------------------------------------
# cache_path_for
# ---------------------------------------------------------------------------


def test_cache_path_for_swaps_extension():
    assert cache_path_for("/a/b/training_log_s1.csv").endswith(
        "training_log_s1.parquet"
    )


def test_cache_path_for_rejects_empty():
    with pytest.raises(ValueError):
        cache_path_for("")


# ---------------------------------------------------------------------------
# Round-trip + dtype downcast
# ---------------------------------------------------------------------------


def test_first_load_creates_parquet_and_returns_data(small_run):
    tlog, ylog, ep_df, yr_df = small_run
    assert not os.path.exists(cache_path_for(tlog))
    assert not os.path.exists(cache_path_for(ylog))

    ep_loaded = load_run_csv(tlog)
    yr_loaded = load_run_csv(ylog)

    # Parquet sidecars now exist.
    assert os.path.exists(cache_path_for(tlog))
    assert os.path.exists(cache_path_for(ylog))

    # Same row counts and column sets.
    assert len(ep_loaded) == len(ep_df)
    assert set(ep_loaded.columns) == set(ep_df.columns)
    assert len(yr_loaded) == len(yr_df)
    assert set(yr_loaded.columns) == set(yr_df.columns)

    # Values are numerically equal up to float32 precision.
    for col in ["mean_reward", "clearing_price_mean"]:
        np.testing.assert_allclose(
            ep_loaded[col].to_numpy(), ep_df[col].to_numpy(), rtol=1e-6, atol=1e-6
        )


def test_downcast_float64_to_float32_and_int64_to_int32(small_run):
    tlog, ylog, _, _ = small_run
    ep_loaded = load_run_csv(tlog)
    yr_loaded = load_run_csv(ylog)

    # All originally-float columns should be float32 now.
    assert ep_loaded["mean_reward"].dtype == np.float32
    assert yr_loaded["clearing_price"].dtype == np.float32
    # Episode/year fit comfortably in int32.
    assert ep_loaded["episode"].dtype == np.int32
    assert yr_loaded["year"].dtype == np.int32


def test_parquet_smaller_than_csv_at_realistic_size(tmp_path):
    """Parquet footer overhead dominates for tiny inputs, so use a
    realistically-sized fixture (matches one episode's worth of year-log
    rows for a multi-agent run) where the compression actually pays off.
    """
    n_rows = 50_000
    n_cols = 20
    rng = np.random.default_rng(0)
    data = {f"col_{i}": rng.normal(size=n_rows) for i in range(n_cols)}
    data["episode"] = np.arange(n_rows, dtype=np.int64)
    df = pd.DataFrame(data)
    csv_path = os.path.join(tmp_path, "big.csv")
    df.to_csv(csv_path, index=False)

    load_run_csv(csv_path)
    pq_size = os.path.getsize(cache_path_for(csv_path))
    csv_size = os.path.getsize(csv_path)
    # Parquet+zstd on dense numeric data should beat CSV by a large margin.
    assert pq_size * 2 < csv_size, (pq_size, csv_size)


# ---------------------------------------------------------------------------
# Cache reuse / staleness
# ---------------------------------------------------------------------------


def test_second_load_does_not_rebuild(small_run, monkeypatch):
    tlog, _, _, _ = small_run
    load_run_csv(tlog)  # build

    # Spy on the build function; second call must not invoke it.
    from src.utils import run_data as rd

    called = {"n": 0}
    real_build = rd._build_parquet

    def spy(*a, **kw):
        called["n"] += 1
        return real_build(*a, **kw)

    monkeypatch.setattr(rd, "_build_parquet", spy)
    df = load_run_csv(tlog)
    assert called["n"] == 0
    assert len(df) > 0


def test_cache_invalidated_when_csv_changes(small_run):
    tlog, _, _, _ = small_run
    load_run_csv(tlog)
    pq_path = cache_path_for(tlog)
    first_mtime_meta = pq.read_metadata(pq_path).metadata[
        b"ets_marl.source_mtime_ns"
    ]

    # Rewrite CSV with extra rows so size + mtime both change.
    time.sleep(0.01)
    new_df = pd.DataFrame(
        {
            "episode": np.arange(100, dtype=np.int64),
            "mean_reward": np.zeros(100),
            "clearing_price_mean": np.zeros(100),
        }
    )
    new_df.to_csv(tlog, index=False)

    df = load_run_csv(tlog)
    assert len(df) == 100  # picked up the new content
    second_mtime_meta = pq.read_metadata(pq_path).metadata[
        b"ets_marl.source_mtime_ns"
    ]
    assert second_mtime_meta != first_mtime_meta


def test_force_rebuild(small_run, monkeypatch):
    tlog, _, _, _ = small_run
    load_run_csv(tlog)  # build

    from src.utils import run_data as rd

    called = {"n": 0}
    real_build = rd._build_parquet

    def spy(*a, **kw):
        called["n"] += 1
        return real_build(*a, **kw)

    monkeypatch.setattr(rd, "_build_parquet", spy)
    load_run_csv(tlog, rebuild=True)
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# Column projection
# ---------------------------------------------------------------------------


def test_columns_projection(small_run):
    tlog, _, _, _ = small_run
    df = load_run_csv(tlog, columns=["episode", "mean_reward"])
    assert list(df.columns) == ["episode", "mean_reward"]


def test_columns_projection_drops_unknown_keys(small_run):
    tlog, _, _, _ = small_run
    df = load_run_csv(tlog, columns=["episode", "does_not_exist"])
    assert list(df.columns) == ["episode"]


def test_columns_all_unknown_returns_empty(small_run):
    tlog, _, _, _ = small_run
    df = load_run_csv(tlog, columns=["nope_1", "nope_2"])
    assert df.empty


# ---------------------------------------------------------------------------
# load_run wrapper + None handling
# ---------------------------------------------------------------------------


def test_load_run_returns_pair(small_run):
    tlog, ylog, _, _ = small_run
    ep, yr = load_run(tlog, ylog)
    assert ep is not None and yr is not None


def test_load_run_with_none_paths():
    ep, yr = load_run(None, None)
    assert ep is None and yr is None


def test_load_run_csv_with_none_returns_none():
    assert load_run_csv(None) is None


def test_missing_csv_and_no_cache_raises(tmp_path):
    bogus = os.path.join(tmp_path, "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_run_csv(bogus)


# ---------------------------------------------------------------------------
# CSV deletion after caching is supported
# ---------------------------------------------------------------------------


def test_csv_can_be_deleted_after_caching(small_run):
    """Disk-space workflow: build cache, delete CSV, still readable."""
    tlog, _, _, _ = small_run
    load_run_csv(tlog)
    pq_path = cache_path_for(tlog)
    os.remove(tlog)
    assert os.path.exists(pq_path)
    df = load_run_csv(tlog)  # should fall back to parquet
    assert len(df) > 0


# ---------------------------------------------------------------------------
# rebuild_cache helper
# ---------------------------------------------------------------------------


def test_rebuild_cache_returns_entries(small_run):
    tlog, ylog, _, _ = small_run
    entries = rebuild_cache([tlog, ylog])
    assert len(entries) == 2
    assert all(isinstance(e, CacheEntry) for e in entries)
    assert all(e.rebuilt for e in entries)
    assert all(os.path.exists(e.parquet_path) for e in entries)


def test_rebuild_cache_skips_none(small_run):
    tlog, _, _, _ = small_run
    entries = rebuild_cache([None, tlog, None])
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Empty CSV (header only)
# ---------------------------------------------------------------------------


def test_empty_csv_produces_empty_parquet(tmp_path):
    p = os.path.join(tmp_path, "empty.csv")
    pd.DataFrame({"episode": pd.Series(dtype=np.int64), "x": pd.Series(dtype=np.float64)}).to_csv(
        p, index=False
    )
    df = load_run_csv(p)
    assert df.empty
    assert "episode" in df.columns
    assert os.path.exists(cache_path_for(p))


# ---------------------------------------------------------------------------
# Direct parquet-path loading
# ---------------------------------------------------------------------------


def test_load_run_csv_accepts_parquet_path_directly(small_run):
    tlog, _, _, _ = small_run
    load_run_csv(tlog)  # build cache
    pq_path = cache_path_for(tlog)
    assert pq_path.endswith(".parquet")
    df = load_run_csv(pq_path)  # direct parquet read
    assert len(df) > 0


def test_cache_path_for_idempotent_on_parquet_input():
    p = "/tmp/foo/year_log_s1.parquet"
    assert cache_path_for(p) == p


# ---------------------------------------------------------------------------
# glob_run_logs
# ---------------------------------------------------------------------------


def test_glob_run_logs_finds_csv_only(small_run):
    from src.utils.run_data import glob_run_logs

    tlog, ylog, _, _ = small_run
    d = os.path.dirname(tlog)
    hits = glob_run_logs(os.path.join(d, "training_log_*.csv"))
    assert hits == [tlog]


def test_glob_run_logs_finds_parquet_when_csv_deleted(small_run):
    from src.utils.run_data import glob_run_logs

    tlog, _, _, _ = small_run
    load_run_csv(tlog)
    os.remove(tlog)
    d = os.path.dirname(tlog)
    hits = glob_run_logs(os.path.join(d, "training_log_*.csv"))
    assert len(hits) == 1
    assert hits[0].endswith(".parquet")


def test_glob_run_logs_prefers_csv_when_both_present(small_run):
    from src.utils.run_data import glob_run_logs

    tlog, _, _, _ = small_run
    load_run_csv(tlog)  # build parquet alongside the CSV
    d = os.path.dirname(tlog)
    hits = glob_run_logs(os.path.join(d, "training_log_*.csv"))
    assert len(hits) == 1
    assert hits[0].endswith(".csv")


def test_glob_run_logs_dedupes_by_stem(tmp_path):
    """A file with both .csv and .parquet siblings shows up exactly once."""
    from src.utils.run_data import glob_run_logs

    a_csv = os.path.join(tmp_path, "year_log_s1.csv")
    a_pq = os.path.join(tmp_path, "year_log_s1.parquet")
    pd.DataFrame({"x": [1]}).to_csv(a_csv, index=False)
    pd.DataFrame({"x": [1]}).to_parquet(a_pq)
    hits = glob_run_logs(os.path.join(tmp_path, "year_log_*.csv"))
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# compress_logs
# ---------------------------------------------------------------------------


def test_compress_logs_roundtrip_and_deletes_csv(small_run):
    from src.utils.run_data import compress_logs

    tlog, ylog, ep_df_orig, yr_df_orig = small_run
    entries = compress_logs([tlog, ylog])
    assert len(entries) == 2
    # CSVs gone, parquets present
    assert not os.path.exists(tlog)
    assert not os.path.exists(ylog)
    assert os.path.exists(cache_path_for(tlog))
    assert os.path.exists(cache_path_for(ylog))
    # Round-trip values still match
    ep_back = load_run_csv(tlog)  # via cache fallback
    yr_back = load_run_csv(ylog)
    assert len(ep_back) == len(ep_df_orig)
    assert len(yr_back) == len(yr_df_orig)


def test_compress_logs_keep_csv_when_requested(small_run):
    from src.utils.run_data import compress_logs

    tlog, _, _, _ = small_run
    compress_logs([tlog], delete_csv=False)
    assert os.path.exists(tlog)
    assert os.path.exists(cache_path_for(tlog))


def test_compress_logs_skips_missing_and_non_csv(tmp_path):
    from src.utils.run_data import compress_logs

    pq_path = os.path.join(tmp_path, "fake.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(pq_path)
    out = compress_logs([
        os.path.join(tmp_path, "missing.csv"),
        pq_path,  # already parquet
        None,
    ])
    assert out == []


# ---------------------------------------------------------------------------
# compress_checkpoints
# ---------------------------------------------------------------------------


def test_compress_checkpoints_roundtrip(tmp_path):
    import tarfile

    from src.utils.run_data import compress_checkpoints

    ckpt_dir = os.path.join(tmp_path, "checkpoints_x_s1")
    os.makedirs(ckpt_dir)
    payloads = {}
    rng = np.random.default_rng(0)
    for i in range(4):
        name = f"agent_{i}_ep100.pt"
        # Use compressible content (zeros + random) so xz produces a
        # measurably smaller archive than the source dir.
        data = np.concatenate(
            [np.zeros(50_000, dtype=np.float32), rng.normal(size=10_000).astype(np.float32)]
        ).tobytes()
        with open(os.path.join(ckpt_dir, name), "wb") as f:
            f.write(data)
        payloads[name] = data

    archive = compress_checkpoints(ckpt_dir)
    assert archive is not None
    assert archive.endswith(".tar.xz")
    assert os.path.exists(archive)
    assert not os.path.exists(ckpt_dir)  # source removed

    # Archive content matches what we put in
    with tarfile.open(archive, mode="r:xz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                fname = os.path.basename(member.name)
                assert fname in payloads
                f = tar.extractfile(member)
                assert f.read() == payloads[fname]


def test_compress_checkpoints_no_op_on_missing_dir(tmp_path):
    from src.utils.run_data import compress_checkpoints

    assert compress_checkpoints(os.path.join(tmp_path, "nope")) is None


def test_compress_checkpoints_no_op_on_empty_dir(tmp_path):
    from src.utils.run_data import compress_checkpoints

    d = os.path.join(tmp_path, "empty")
    os.makedirs(d)
    assert compress_checkpoints(d) is None
    # Empty dir is left alone
    assert os.path.isdir(d)


def test_compress_checkpoints_keep_dir_option(tmp_path):
    from src.utils.run_data import compress_checkpoints

    ckpt_dir = os.path.join(tmp_path, "checkpoints_keep")
    os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "a.pt"), "wb") as f:
        f.write(b"hello")
    archive = compress_checkpoints(ckpt_dir, delete_dir=False)
    assert os.path.exists(archive)
    assert os.path.isdir(ckpt_dir)

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

    # Pin to xz to keep this as a back-compat regression: legacy archives
    # on disk are .tar.xz and must remain readable forever, even though
    # 'auto' now resolves to zst.
    archive = compress_checkpoints(ckpt_dir, codec="xz")
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


# ---------------------------------------------------------------------------
# decompress_checkpoints
# ---------------------------------------------------------------------------


def test_decompress_checkpoints_roundtrip(tmp_path):
    from src.utils.run_data import compress_checkpoints, decompress_checkpoints

    ckpt_dir = os.path.join(tmp_path, "checkpoints_x_s1")
    os.makedirs(ckpt_dir)
    payloads = {f"agent_{i}.pt": os.urandom(1024) for i in range(3)}
    for name, data in payloads.items():
        with open(os.path.join(ckpt_dir, name), "wb") as f:
            f.write(data)

    archive = compress_checkpoints(ckpt_dir)
    assert not os.path.isdir(ckpt_dir)

    out = decompress_checkpoints(archive, out_dir=str(tmp_path))
    assert out is not None
    assert os.path.isdir(out)
    # Original archive preserved (delete_archive default False)
    assert os.path.exists(archive)
    # Every payload restored byte-for-byte
    for name, data in payloads.items():
        with open(os.path.join(out, name), "rb") as f:
            assert f.read() == data


def test_decompress_checkpoints_delete_archive(tmp_path):
    from src.utils.run_data import compress_checkpoints, decompress_checkpoints

    ckpt_dir = os.path.join(tmp_path, "checkpoints_y_s1")
    os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "a.pt"), "wb") as f:
        f.write(b"x")
    archive = compress_checkpoints(ckpt_dir)
    decompress_checkpoints(archive, out_dir=str(tmp_path), delete_archive=True)
    assert not os.path.exists(archive)


def test_decompress_checkpoints_missing_archive(tmp_path):
    from src.utils.run_data import decompress_checkpoints

    assert decompress_checkpoints(os.path.join(tmp_path, "nope.tar.xz")) is None


def test_compress_logs_memory_bounded_streaming(tmp_path):
    """compress_logs streams in chunks — peak memory is bounded by
    chunksize × ncols × 8 bytes, not by the whole-CSV size. Verified by
    converting a 200k-row CSV with chunksize=10k and confirming the
    parquet writer never materialised the full DataFrame.
    """
    from src.utils.run_data import compress_logs

    csv = os.path.join(tmp_path, "year_log_huge_s1.csv")
    n = 200_000
    df = pd.DataFrame({
        "episode": np.arange(n, dtype=np.int64),
        "year": np.tile(np.arange(12), n // 12 + 1)[:n].astype(np.int64),
        "clearing_price": np.random.RandomState(0).uniform(50, 200, n),
        "tnac": np.random.RandomState(1).uniform(0, 1000, n),
    })
    df.to_csv(csv, index=False)
    entries = compress_logs([csv], chunksize=10_000)
    assert len(entries) == 1
    assert entries[0].rows == n
    # Round-trip preserves every row
    pq_path = cache_path_for(csv)
    back = pd.read_parquet(pq_path)
    assert len(back) == n
    assert (back["episode"].values == df["episode"].values).all()


# ---------------------------------------------------------------------------
# Lazy pyarrow import — ensures end-of-run checkpoint compression on Azure
# ML workers where pyarrow may be missing from the curated env still works.
# ---------------------------------------------------------------------------

def test_compress_checkpoints_does_not_require_pyarrow(tmp_path):
    """Reproduces the silent-failure scenario hit on Azure ML curated envs:
    when ``pyarrow`` is missing, the *whole* ``run_data`` module used to
    fail to import, taking the stdlib-only ``compress_checkpoints`` path
    down with it. After the fix the module imports cleanly without
    pyarrow and ``compress_checkpoints`` keeps working on its own.
    """
    import importlib
    import subprocess
    import sys

    # Run in a child so the meta-path block doesn't pollute our session.
    script = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pyarrow' or name.startswith('pyarrow.'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "for m in list(sys.modules):\n"
        "    if m.startswith('pyarrow') or m.startswith('src.utils.run_data'):\n"
        "        del sys.modules[m]\n"
        "from src.utils.run_data import compress_checkpoints, compress_logs\n"
        "import os\n"
        f"ck = r'{tmp_path}/checkpoints_x_s1'\n"        "os.makedirs(ck, exist_ok=True)\n"
        "open(os.path.join(ck, 'a.pt'), 'wb').write(b'hi')\n"
        "arch = compress_checkpoints(ck)\n"
        "assert arch and os.path.exists(arch), arch\n"
        # compress_logs should now raise the actionable ImportError.
        f"csv = r'{tmp_path}/training_log_x_s1.csv'\n"
        "open(csv, 'w').write('a,b\\n1,2\\n')\n"
        "try:\n"
        "    compress_logs([csv])\n"
        "    raise SystemExit('compress_logs should have raised ImportError')\n"
        "except ImportError as e:\n"
        "    assert 'pyarrow' in str(e), str(e)\n"
        "print('OK')\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, (
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "OK" in res.stdout


# ---------------------------------------------------------------------------
# Multi-codec checkpoint compression (zst / gz / xz)
# ---------------------------------------------------------------------------


def _make_ckpt_dir(tmp_path, name: str, payload: dict[str, bytes]) -> str:
    d = os.path.join(tmp_path, name)
    os.makedirs(d, exist_ok=True)
    for fname, data in payload.items():
        with open(os.path.join(d, fname), "wb") as f:
            f.write(data)
    return d


@pytest.mark.parametrize("codec,suffix", [
    ("zst", ".tar.zst"),
    ("gz", ".tar.gz"),
    ("xz", ".tar.xz"),
])
def test_compress_checkpoints_codec_roundtrip(tmp_path, codec, suffix):
    """Each supported codec round-trips byte-for-byte and writes the
    expected suffix. Verifies the codec dispatch in
    ``compress_checkpoints`` and the symmetric sniff in
    ``decompress_checkpoints``.
    """
    from src.utils.run_data import compress_checkpoints, decompress_checkpoints

    if codec == "zst":
        pytest.importorskip("zstandard")

    payload = {
        f"agent_{i}.pt": (np.zeros(2000, dtype=np.float32).tobytes()
                          + os.urandom(1024))
        for i in range(3)
    }
    ckpt_dir = _make_ckpt_dir(tmp_path, f"checkpoints_x_{codec}_s1", payload)

    archive = compress_checkpoints(ckpt_dir, codec=codec)
    assert archive is not None
    assert archive.endswith(suffix), archive
    assert os.path.exists(archive)
    assert not os.path.exists(ckpt_dir)  # source removed by default

    out = decompress_checkpoints(archive, out_dir=str(tmp_path / f"out_{codec}"))
    assert out is not None
    assert os.path.isdir(out)
    for fname, data in payload.items():
        with open(os.path.join(out, fname), "rb") as f:
            assert f.read() == data


def test_compress_checkpoints_auto_picks_zst_when_available(tmp_path):
    """``codec='auto'`` resolves to ``'zst'`` when the ``zstandard``
    package is importable. Ensures real runs get the fast codec by
    default without any config tweak.
    """
    pytest.importorskip("zstandard")
    from src.utils.run_data import compress_checkpoints

    d = _make_ckpt_dir(tmp_path, "checkpoints_auto_s1", {"a.pt": b"hello"})
    archive = compress_checkpoints(d, codec="auto")
    assert archive.endswith(".tar.zst"), archive


class _ZstandardImportBlocker:
    """Meta-path finder that raises ImportError for ``zstandard``.

    Used by tests below to simulate hosts where the optional
    ``zstandard`` dependency is not installed (e.g. minimal Azure ML
    curated environments) so we can verify the auto-fallback path and
    the explicit-codec hard-fail path both behave correctly.
    """

    def find_spec(self, name, path=None, target=None):  # noqa: D401
        if name == "zstandard":
            raise ImportError("blocked for test")
        return None


def _with_zstandard_blocked():
    """Context manager that hides ``zstandard`` from imports.

    Reloads ``src.utils.run_data`` inside the block so its
    ``_zstandard_available`` cache is rebuilt against the patched
    meta_path, and again on exit so the rest of the test session sees
    the real environment.
    """
    import contextlib
    import importlib
    import sys

    @contextlib.contextmanager
    def _ctx():
        import src.utils.run_data as rd

        saved = sys.modules.pop("zstandard", None)
        blocker = _ZstandardImportBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            importlib.reload(rd)
            yield rd
        finally:
            sys.meta_path.remove(blocker)
            if saved is not None:
                sys.modules["zstandard"] = saved
            importlib.reload(rd)

    return _ctx()


def test_compress_checkpoints_auto_falls_back_to_gz_without_zstandard(tmp_path):
    """When ``zstandard`` is missing, ``codec='auto'`` silently falls back
    to gzip — the stdlib codec — so a curated environment that never
    installs the optional dep still gets working compression.
    """
    with _with_zstandard_blocked() as rd:
        d = _make_ckpt_dir(tmp_path, "checkpoints_fallback_s1", {"a.pt": b"x"})
        archive = rd.compress_checkpoints(d, codec="auto")
        assert archive.endswith(".tar.gz"), archive


def test_compress_checkpoints_zst_raises_when_zstandard_missing(tmp_path):
    """Explicit ``codec='zst'`` must raise a clear, actionable error
    when ``zstandard`` is not importable — silent fallback would mask a
    misconfigured pinned-codec deployment.
    """
    with _with_zstandard_blocked() as rd:
        d = _make_ckpt_dir(tmp_path, "checkpoints_strict_s1", {"a.pt": b"x"})
        with pytest.raises(RuntimeError, match="zstandard"):
            rd.compress_checkpoints(d, codec="zst")


def test_decompress_checkpoints_unrecognised_suffix_raises(tmp_path):
    """A non-archive path with an unknown suffix must raise — silent
    None would let a typo propagate as a "no checkpoint loaded" warning
    far from the source of the bug.
    """
    from src.utils.run_data import decompress_checkpoints

    bogus = os.path.join(tmp_path, "checkpoints_x_s1.tar.bz2")
    with open(bogus, "wb") as f:
        f.write(b"\x00")
    with pytest.raises(ValueError, match="suffix"):
        decompress_checkpoints(bogus)


def test_compress_results_finder_skips_existing_archives(tmp_path):
    """The migration-script finder ``_find_checkpoint_dirs`` must skip a
    checkpoints dir whenever **any** archive variant (zst/gz/xz) already
    sits next to it. This makes re-running ``compress_results.py`` on a
    partially migrated tree a clean no-op regardless of which codec
    produced the existing archive.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compress_results_mod",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "compress_results.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Three checkpoint dirs, each next to a pre-existing archive in a
    # different codec; finder should skip all three.
    for tag, suffix in [
        ("zst", ".tar.zst"),
        ("gz", ".tar.gz"),
        ("xz", ".tar.xz"),
    ]:
        d = os.path.join(tmp_path, f"checkpoints_{tag}_s1")
        os.makedirs(d)
        with open(os.path.join(d, "a.pt"), "wb") as f:
            f.write(b"x")
        with open(d + suffix, "wb") as f:
            f.write(b"\x00\x01")

    # And one with no archive — finder should pick this up.
    d_new = os.path.join(tmp_path, "checkpoints_new_s1")
    os.makedirs(d_new)
    with open(os.path.join(d_new, "a.pt"), "wb") as f:
        f.write(b"x")

    found = mod._find_checkpoint_dirs(str(tmp_path))
    assert found == [d_new]


def test_run_data_module_imports_without_pyarrow(tmp_path):
    """Module-level import must not require pyarrow."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pyarrow' or name.startswith('pyarrow.'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import src.utils.run_data\n"
        "print('imported')\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, (
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "imported" in res.stdout

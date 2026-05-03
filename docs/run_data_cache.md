# Run-data loader: parquet cache for training/year logs

`src/utils/run_data.py` is a persistent parquet cache for the per-run CSVs
written by `scripts/train.py` (`training_log_*.csv` and `year_log_*.csv`).
It exists because pandas' default float64 + line-oriented CSV parsing
makes a single thesis-scale sweep eat >4 GiB of RAM and take many
minutes per notebook restart.

## What it does

1. **First load:** the source CSV is streamed in chunks, dtype-downcast
   (float64 → float32, int64 → int32 where lossless) and written next to
   the CSV as a `.parquet` sibling with zstd compression. This is
   typically 5–10× smaller and 5–20× faster to reload.
2. **Every later load** (in the same notebook or any other) reads the
   parquet directly — no CSV parsing, no full-frame allocation.
3. **Column projection** (`columns=[...]`) is pushed down to the parquet
   reader, so a notebook that only plots, say, `clearing_price` and
   `tnac` reads ~1 % of the original year-log payload and finishes in
   seconds.
4. **Cache invalidation** is automatic: the source CSV's size and
   `mtime_ns` are embedded in the parquet metadata. Rewriting the CSV
   forces a rebuild on the next call.
5. **Disk-space mode:** once the parquet exists you may delete the source
   CSV. The loader falls back to the parquet and continues to work, so
   you can free 80–90 % of the disk used by raw logs without losing
   reproducibility.

## Use in a notebook

Replace the typical

```python
ep_df = pd.read_csv(training_log_path)
yr_df = pd.read_csv(year_log_path)
```

with

```python
from src.utils.run_data import load_run

ep_df, yr_df = load_run(training_log_path, year_log_path)
```

`load_run_csv` accepts either a `.csv` or a `.parquet` path
interchangeably, so notebook code that points at the CSV keeps working
even after the source CSV has been deleted to free disk.

For column-selective loads (huge speed-up on the year log):

```python
from src.utils.run_data import load_run_csv

yr_df = load_run_csv(
    year_log_path,
    columns=["episode", "year", "clearing_price", "tnac"],
)
```

For glob-based discovery that survives CSV deletion:

```python
from src.utils.run_data import glob_run_logs

ppo_paths = glob_run_logs("results/sweep/*/training_log_s*.csv")
# Returns CSV paths if present, parquet paths otherwise — every run is
# discoverable either way.
```

For an entire sweep at once:

```python
from src.utils.run_data import load_sweep

runs = load_sweep("configs/sweeps/thesis_experiments.yaml")
for (variant, seed), r in runs.items():
    if r["ep_df"] is not None:
        ...
```

`runs` is keyed by `(variant_name, seed)` and each entry mirrors the
notebook's existing `runs` dict (`ep_df`, `yr_df`, `training_log`,
`year_log`, `config`, `results_dir`).

## End-of-training compression (cloud workers)

Long sweeps run on Azure / cloud workers and ship results back to your
laptop. To keep that download small, `scripts/train.py` runs two
lossless compression steps at clean end-of-run:

1. `training_log_*.csv` and `year_log_*.csv` are converted to zstd
   parquet via `compress_logs(...)` and the source CSVs are deleted.
2. The `checkpoints_<tag>_s<seed>/` directory is bundled into a single
   tar archive via `compress_checkpoints(...)` and the source dir is
   deleted. The codec is chosen by `logging.compress_on_finish.checkpoints_codec`
   (default `"auto"` — zstd when available, gzip fallback).

Both steps round-trip losslessly modulo the documented float64→float32 /
int64→int32 downcast in the parquet writer. The behaviour is controlled
by `logging.compress_on_finish` in the YAML config:

```yaml
logging:
  compress_on_finish:
    logs: true                      # CSV → parquet
    checkpoints: true               # checkpoints_*/ → tar archive
    delete_csv: true                # remove the source CSV after verify
    delete_checkpoint_dir: true
    checkpoints_codec: "auto"       # auto | zst | gz | xz
```

Set any of these to `false` to keep the originals alongside the
compressed copies.

### Codec choice

| codec | suffix      | compression speed     | ratio (typical, on `.pt` weights) | reader |
| ----- | ----------- | --------------------- | --------------------------------- | ------ |
| `zst` | `.tar.zst`  | **~30s @ 50 MB/s** (multi-threaded) | ~65–70% of source | needs `zstandard` (project dep) |
| `gz`  | `.tar.gz`   | ~3–4 min @ 7 MB/s     | ~75–80% of source                 | stdlib |
| `xz`  | `.tar.xz`   | ~15+ min @ ~2 MB/s    | ~60–65% of source                 | stdlib |

`"auto"` (the default) picks `zst` when the optional `zstandard`
package is importable and silently falls back to `gz` otherwise. Pin
to `"xz"` only for cold-archival uploads where read time is irrelevant
and you want the absolute smallest footprint.

**Existing `.tar.xz` archives on disk remain fully supported by
`decompress_checkpoints` and `scripts/evaluate.py` without any
migration step.** The reader sniffs the suffix and dispatches to the
right decoder, so swapping the writer codec is a transparent change.

### Memory profile

Both compression steps stream end-to-end:

* `compress_logs` reads the source CSV in 250k-row chunks (configurable
  via `chunksize=`) and writes each chunk as a parquet row group, so
  peak heap is bounded by `chunksize × n_columns × 8 bytes` — a few
  hundred MB even on a 5 GB year-log. Safe on Azure ML's standard
  D16ds_v5 (64 GB RAM).
* `compress_checkpoints` uses `tarfile`'s incremental writer; each
  ``.pt`` is read once into the codec's encoder buffer and never fully
  materialised in Python.

### Re-opening a compressed run

Logs are read transparently — `load_run_csv` accepts either the
original CSV path (auto-falls-back to the parquet sibling) or the
parquet path directly.

Checkpoints are extracted with the symmetric in-process helper, which
auto-sniffs the suffix:

```bash
# Shell — recommended for one-off use (pick the right flag for the codec)
tar -xf results/sweeps/.../checkpoints_lrf_low_s1729.tar.zst -C results/sweeps/.../

# In-process — handles all three suffixes uniformly
from src.utils.run_data import decompress_checkpoints
ckpt_dir = decompress_checkpoints(
    "results/sweeps/.../checkpoints_lrf_low_s1729.tar.zst"
)
# ckpt_dir now holds the same agent_*.pt layout the trainer wrote;
# pass it straight to scripts/evaluate.py --checkpoint <ckpt_dir>/agent_0_best.pt
```

`decompress_checkpoints` is non-destructive by default (it leaves the
archive on disk); pass `delete_archive=True` to reclaim the space once
you've extracted.

## Migrating existing results

Run-once script for the data you've already produced:

```bash
# Dry-run (lists what would happen, no writes)
python scripts/compress_results.py results --dry-run

# Migrate the whole results tree
python scripts/compress_results.py results

# Migrate a single sweep
python scripts/compress_results.py results/sweeps/thesis_experiments

# Keep the originals (e.g. while you spot-check the parquets)
python scripts/compress_results.py results --keep-source
```

The script walks the directory recursively, converts every `training_log_*` /
`year_log_*` / `ql_training_log_*` CSV to parquet (verifying row counts
before deleting the CSV) and bundles every `checkpoints_*/` directory
into a `tar.xz` archive. It is idempotent — re-running it skips runs
whose archives already exist.

## Warming the cache once

The first conversion of a 5 GB year-log CSV is still IO-bound. Run

```python
from src.utils.run_data import rebuild_cache
rebuild_cache([
    "results/sweeps/.../year_log_default_s1729.csv",
    ...
])
```

once at the top of any notebook (or in a small script) and every later
notebook restart hits the parquet cache directly. (If you've already run
`scripts/compress_results.py`, the cache is fully populated and no
warm-up is needed.)

## Disk-space sketch

For the current default-config sweep (4 seeds × 120 000 episodes) the
year log is ~5 GB CSV per seed. Empirically:

| File             | CSV (float64) | Compressed                        |
| ---------------- | ------------- | --------------------------------- |
| `year_log_*.csv` | ~5 GB         | ~0.5–1.0 GB parquet (zstd)        |
| `training_log_*.csv` | ~30 MB    | ~5–10 MB parquet (zstd)           |
| `checkpoints_*/` | ~2.5 GB       | ~1.6–1.8 GB `tar.zst` in 30–60 s  |
|                  |               | ~1.7–1.9 GB `tar.gz`  in 3–4 min  |
|                  |               | ~1.5–1.7 GB `tar.xz`  in 15+ min  |

A 4-seed sweep drops from ~22 GB on disk to ~3–5 GB; a 16-cell
thesis-experiments sweep stays well under one disk's worth of headroom.
The default `tar.zst` codec gets you the disk savings in tens of
seconds rather than minutes per seed, which adds up to multi-hour
wall-clock savings on a full sweep.

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
   `checkpoints_<tag>_s<seed>.tar.xz` via `compress_checkpoints(...)` and
   the source dir is deleted.

Both steps round-trip losslessly modulo the documented float64→float32 /
int64→int32 downcast in the parquet writer. The behaviour is controlled
by `logging.compress_on_finish` in the YAML config:

```yaml
logging:
  compress_on_finish:
    logs: true                # CSV → parquet
    checkpoints: true         # checkpoints_*/ → tar.xz
    delete_csv: true          # remove the source CSV after verify
    delete_checkpoint_dir: true
```

Set any of these to `false` to keep the originals alongside the
compressed copies.

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

| File             | CSV (float64) | Parquet (float32 + zstd) |
| ---------------- | ------------- | ------------------------ |
| `year_log_*.csv` | ~5 GB         | ~0.5–1.0 GB              |
| `training_log_*.csv` | ~30 MB    | ~5–10 MB                 |
| `checkpoints_*/` | ~2.5 GB       | ~0.5–1.5 GB tar.xz       |

A 4-seed sweep drops from ~22 GB on disk to ~3–5 GB; a 16-cell
thesis-experiments sweep stays well under one disk's worth of headroom.

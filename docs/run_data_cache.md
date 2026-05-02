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

For column-selective loads (huge speed-up on the year log):

```python
from src.utils.run_data import load_run_csv

yr_df = load_run_csv(
    year_log_path,
    columns=["episode", "year", "clearing_price", "tnac"],
)
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
notebook restart hits the parquet cache directly.

## Disk-space sketch

For the current default-config sweep (4 seeds × 120 000 episodes) the
year log is ~5 GB CSV per seed. Empirically:

| File             | CSV (float64) | Parquet (float32 + zstd) |
| ---------------- | ------------- | ------------------------ |
| `year_log_*.csv` | ~5 GB         | ~0.5–1.0 GB              |
| `training_log_*.csv` | ~30 MB    | ~5–10 MB                 |

So a 4-seed sweep drops from ~20 GB to ~3–4 GB on disk, and a 16-seed
thesis-experiments sweep stays well under one disk's worth of headroom.
If disk pressure remains an issue after caching, deleting the source
CSVs (the loader keeps reading the parquets) reclaims the rest.

"""
sweep.py — launch a multi-variant × multi-seed training sweep.

Reads a sweep spec YAML, materialises one resolved config per variant,
and runs every (variant, seed) job as an independent ``train.py``
subprocess via ``ProcessPoolExecutor``. Each subprocess writes its
``training_log_s{seed}.csv`` and checkpoints into the variant's own
``logging.results_dir`` (``<spec.output_dir>/<variant_name>``), so logs
from different variants never collide.

Usage
-----
    python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml

The spec file's ``parallel_workers`` and ``threads_per_worker`` settings
control concurrency; they can be overridden on the command line. Each
seed's RNG, optimiser order, and CSV output are bit-identical to a
sequential ``train.py --config <variant>.yaml --seed <S>`` invocation;
the only thing the sweep launcher does differently is run several such
invocations in parallel processes.

See ``src/utils/sweep.py`` for the spec schema and validation rules.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make ``src`` importable when this script is run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.compute_setup import detect_architecture  # noqa: E402
from src.utils.sweep import (  # noqa: E402
    iter_variant_yaml_paths,
    load_sweep_spec,
)


def _load_base_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _run_job(
    train_py: str,
    config_path: str,
    seed: int,
    threads: int,
) -> tuple[str, int, int]:
    """Run a single (variant_yaml, seed) training job in a subprocess.

    Returns ``(config_path, seed, returncode)``.
    """
    env = os.environ.copy()
    env["ETS_NUM_THREADS"] = str(max(1, int(threads)))
    cmd = [
        sys.executable,
        train_py,
        "--config", config_path,
        "--seed", str(seed),
        "--parallel-seeds", "1",  # we manage parallelism ourselves
    ]
    rc = subprocess.call(cmd, env=env)
    return config_path, seed, rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sweep of (config-variant × seed) training jobs in parallel. "
            "See src/utils/sweep.py for the spec schema."
        )
    )
    parser.add_argument(
        "--spec", required=True,
        help="Path to a sweep spec YAML file (see configs/sweeps/example_sweep.yaml).",
    )
    parser.add_argument(
        "--parallel-workers", type=int, default=None,
        help="Override spec's parallel_workers (number of concurrent training processes).",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=None,
        help=(
            "Override spec's threads_per_worker. When omitted and not set in the "
            "spec, an architecture-aware default is computed."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate spec, write resolved variant YAMLs, print the job plan, and exit.",
    )
    args = parser.parse_args()

    spec = load_sweep_spec(args.spec)
    base_config = _load_base_config(spec["base_config"])

    output_dir = spec["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Materialise one YAML per variant under <output_dir>/_resolved/.
    scratch_dir = os.path.join(output_dir, "_resolved")
    variant_jobs: list[tuple[str, str, list[int]]] = list(
        iter_variant_yaml_paths(spec, base_config, scratch_dir)
    )

    # Flatten to a (yaml_path, seed) list.
    jobs: list[tuple[str, int]] = []
    for variant_name, yaml_path, seeds in variant_jobs:
        for s in seeds:
            jobs.append((yaml_path, int(s)))

    n_jobs = len(jobs)
    if n_jobs == 0:
        print("[sweep] no jobs in spec — nothing to do", file=sys.stderr)
        return 0

    # Resolve concurrency.
    n_workers = args.parallel_workers or spec["parallel_workers"]
    n_workers = max(1, min(int(n_workers), n_jobs))

    threads = args.threads_per_worker or spec["threads_per_worker"]
    if threads is None:
        arch = detect_architecture()
        total = arch.get("physical_cores") or arch.get("logical_cores") or 1
        threads = max(1, total // n_workers)
    threads = max(1, int(threads))

    train_py = os.path.join(_REPO_ROOT, "scripts", "train.py")

    # Plan summary.
    print("=" * 72, file=sys.stderr)
    print(f"[sweep] spec:         {args.spec}", file=sys.stderr)
    print(f"[sweep] base_config:  {spec['base_config']}", file=sys.stderr)
    print(f"[sweep] output_dir:   {output_dir}", file=sys.stderr)
    print(f"[sweep] variants:     {len(variant_jobs)}", file=sys.stderr)
    print(f"[sweep] total jobs:   {n_jobs}", file=sys.stderr)
    print(
        f"[sweep] concurrency:  {n_workers} worker(s) x {threads} thread(s)",
        file=sys.stderr,
    )
    print("[sweep] plan:", file=sys.stderr)
    for variant_name, yaml_path, seeds in variant_jobs:
        seed_str = ", ".join(str(s) for s in seeds)
        print(f"  - {variant_name}: seeds=[{seed_str}]  ({yaml_path})", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    if args.dry_run:
        print("[sweep] --dry-run: exiting without launching jobs", file=sys.stderr)
        return 0

    # Launch.
    failures: list[tuple[str, int, int]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_run_job, train_py, yaml_path, seed, threads)
            for yaml_path, seed in jobs
        ]
        for fut in as_completed(futures):
            cfg_path, seed, rc = fut.result()
            completed += 1
            tag = "OK " if rc == 0 else f"FAIL({rc})"
            print(
                f"[sweep] [{completed}/{n_jobs}] {tag}  {os.path.basename(cfg_path)} seed={seed}",
                file=sys.stderr,
            )
            if rc != 0:
                failures.append((cfg_path, seed, rc))

    if failures:
        print(
            f"[sweep] {len(failures)} of {n_jobs} jobs failed:",
            file=sys.stderr,
        )
        for cfg_path, seed, rc in failures:
            print(
                f"  - {os.path.basename(cfg_path)} seed={seed} rc={rc}",
                file=sys.stderr,
            )
        return 1

    print(f"[sweep] all {n_jobs} jobs completed successfully", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

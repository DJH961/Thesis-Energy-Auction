"""
sweep.py — launch a multi-variant × multi-seed training sweep.

Reads a sweep spec YAML, materialises one resolved config per variant,
and runs every (variant, seed) job as an independent ``train.py``
subprocess via ``ProcessPoolExecutor``. Each subprocess:

* writes its CSVs / checkpoints under the variant's own
  ``logging.results_dir`` (``<spec.output_dir>/<variant_name>``), with
  filenames disambiguated by ``--run-tag <variant_name>`` so they stay
  unique even when copied into a single folder
  (``training_log_<variant>_s<seed>.csv`` etc.);
* has its stdout+stderr captured into
  ``<results_dir>/run_<variant>_s<seed>.log`` instead of being printed
  to the parent terminal — keeping the terminal readable while
  preserving the full per-job log on disk for later inspection.

The parent terminal only shows short, structured progress lines:

    [sweep] [START 1/17] reference s=1  → results/.../reference/run_reference_s1.log
    [sweep] [LIVE  reference s=1] [Ep 200/100000  ETA 03h12m  ...]
    [sweep] [DONE  1/17 OK ] reference s=1

Heartbeats are printed every ``--heartbeat-interval`` seconds (default
60s). Each heartbeat emits the most recent non-empty line from each
running job's log file, so you still see the meaty per-episode
diagnostics that ``train.py`` normally prints — just sampled, not
flooded.

Use ``--quiet`` to suppress heartbeats entirely.

Usage
-----
    python scripts/sweep.py --spec configs/sweeps/example_sweep.yaml

See ``src/utils/sweep.py`` for the spec schema and validation rules.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
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


def _job_log_path(results_dir: str, variant_name: str, seed: int) -> str:
    return os.path.join(results_dir, f"run_{variant_name}_s{seed}.log")


def _run_job(
    train_py: str,
    config_path: str,
    seed: int,
    threads: int,
    variant_name: str,
    results_dir: str,
) -> tuple[str, int, int, str]:
    """Run a single (variant, seed) training job in a subprocess.

    The child's stdout+stderr are redirected to a per-job log file inside
    ``results_dir`` so the parent terminal stays clean. The full output
    is preserved on disk for later inspection.

    Returns ``(variant_name, seed, returncode, log_path)``.
    """
    env = os.environ.copy()
    env["ETS_NUM_THREADS"] = str(max(1, int(threads)))
    # Force unbuffered child output so heartbeats see fresh lines.
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        train_py,
        "--config", config_path,
        "--seed", str(seed),
        "--parallel-seeds", "1",          # we manage parallelism ourselves
        "--run-tag", variant_name,         # disambiguate output filenames
    ]

    os.makedirs(results_dir, exist_ok=True)
    log_path = _job_log_path(results_dir, variant_name, seed)
    with open(log_path, "w", buffering=1) as log_f:
        log_f.write(
            f"# {variant_name} s={seed}  cmd: {' '.join(cmd)}\n"
            f"# threads={threads}  started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        log_f.flush()
        rc = subprocess.call(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return variant_name, seed, rc, log_path


def _tail_last_meaningful_line(path: str, max_bytes: int = 16384) -> str | None:
    """Return the last informative line from ``path`` (or None).

    Skips:
      * empty / whitespace-only lines,
      * comment lines starting with ``#``,
      * pure-separator lines (only a single repeated punctuation character
        like ``═``, ``─``, ``=``, ``-``, ``*``, ``·``), which the trainer
        prints at the boundaries of each diagnostic block and which would
        otherwise dominate the heartbeat output.
    """
    _SEP_CHARS = set("═─=-*·.━─_• ")
    try:
        size = os.path.getsize(path)
        if size == 0:
            return None
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Skip pure-separator lines (e.g. ═════ or -----).
            unique = set(stripped)
            if unique and unique.issubset(_SEP_CHARS):
                continue
            return stripped
    except (OSError, ValueError):
        return None
    return None


class _Heartbeat:
    """Background thread that periodically prints the tail of running job logs."""

    def __init__(self, interval: float, stream=None):
        # Floor the interval to a small positive value to keep the loop sane
        # at near-zero settings (e.g. unit tests) without blocking forever.
        self.interval = max(0.05, float(interval))
        self._lock = threading.Lock()
        self._jobs: dict[tuple[str, int], str] = {}  # (variant, seed) -> log_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Allow tests to inject a writable stream; default to current stderr
        # at print-time (so output still respects redirections).
        self._stream = stream

    def add(self, variant: str, seed: int, log_path: str) -> None:
        with self._lock:
            self._jobs[(variant, seed)] = log_path

    def remove(self, variant: str, seed: int) -> None:
        with self._lock:
            self._jobs.pop((variant, seed), None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="sweep-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                snapshot = list(self._jobs.items())
            if not snapshot:
                continue
            for (variant, seed), log_path in snapshot:
                line = _tail_last_meaningful_line(log_path)
                if line:
                    # Truncate very long lines so the terminal stays readable.
                    if len(line) > 200:
                        line = line[:197] + "..."
                    out = self._stream if self._stream is not None else sys.stderr
                    print(
                        f"[sweep] [LIVE  {variant} s={seed}] {line}",
                        file=out,
                        flush=True,
                    )


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
        "--heartbeat-interval", type=float, default=60.0,
        help=(
            "Seconds between live-progress heartbeats printed to the terminal "
            "(default: 60). Each heartbeat shows the last non-empty log line "
            "from every running job. Use --quiet to disable."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress heartbeat lines (start/finish lines are always printed).",
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

    # Flatten to a (variant_name, yaml_path, seed, results_dir) list.
    jobs: list[tuple[str, str, int, str]] = []
    for variant_name, yaml_path, seeds in variant_jobs:
        results_dir = os.path.join(output_dir, variant_name)
        for s in seeds:
            jobs.append((variant_name, yaml_path, int(s), results_dir))

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
    if not args.quiet:
        print(
            f"[sweep] heartbeat:    every {args.heartbeat_interval:.0f}s "
            f"(last log line from each running job)",
            file=sys.stderr,
        )
    print("[sweep] plan:", file=sys.stderr)
    for variant_name, yaml_path, seeds in variant_jobs:
        seed_str = ", ".join(str(s) for s in seeds)
        print(
            f"  - {variant_name}: seeds=[{seed_str}]  ({yaml_path})",
            file=sys.stderr,
        )
    print("=" * 72, file=sys.stderr)

    if args.dry_run:
        print("[sweep] --dry-run: exiting without launching jobs", file=sys.stderr)
        return 0

    # Launch.
    heartbeat = None
    if not args.quiet:
        heartbeat = _Heartbeat(args.heartbeat_interval)
        heartbeat.start()

    failures: list[tuple[str, int, int, str]] = []
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_meta = {}
            for idx, (variant_name, yaml_path, seed, results_dir) in enumerate(jobs, 1):
                fut = pool.submit(
                    _run_job, train_py, yaml_path, seed, threads, variant_name, results_dir
                )
                future_to_meta[fut] = (idx, variant_name, seed, results_dir)
                # Pre-register with heartbeat using the predicted log path so
                # the user gets live progress even before the child has flushed
                # much output.
                if heartbeat is not None:
                    heartbeat.add(
                        variant_name, seed,
                        _job_log_path(results_dir, variant_name, seed),
                    )
                print(
                    f"[sweep] [START {idx}/{n_jobs}] {variant_name} s={seed}  "
                    f"→ {_job_log_path(results_dir, variant_name, seed)}",
                    file=sys.stderr,
                    flush=True,
                )

            for fut in as_completed(future_to_meta):
                idx, variant_name, seed, results_dir = future_to_meta[fut]
                _v, _s, rc, log_path = fut.result()
                completed += 1
                if heartbeat is not None:
                    heartbeat.remove(variant_name, seed)
                tag = "OK " if rc == 0 else f"FAIL({rc})"
                print(
                    f"[sweep] [DONE  {completed}/{n_jobs} {tag}] {variant_name} s={seed}  "
                    f"({log_path})",
                    file=sys.stderr,
                    flush=True,
                )
                if rc != 0:
                    failures.append((variant_name, seed, rc, log_path))
    finally:
        if heartbeat is not None:
            heartbeat.stop()

    if failures:
        print(
            f"[sweep] {len(failures)} of {n_jobs} jobs failed:",
            file=sys.stderr,
        )
        for variant_name, seed, rc, log_path in failures:
            print(
                f"  - {variant_name} s={seed} rc={rc}  log: {log_path}",
                file=sys.stderr,
            )
        return 1

    print(
        f"[sweep] all {n_jobs} jobs completed successfully — per-job logs in "
        f"{output_dir}/<variant>/run_<variant>_s<seed>.log",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

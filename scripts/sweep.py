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
    [sweep] [LIVE  reference s=1] Ep 1200/100000 (1.2%) | px 75→142 (μ128) | und 2/12 | sec 60→78 (μ70, m41%) | comp 95% | green 31→44% | R̄ -1.8 | ETA 4h12m
    [sweep] [ETA total ≈ 18h33m, elapsed 0h45m] (3/17 done, 4 running, 10 queued)
    [sweep] [DONE  1/17 OK ] reference s=1

Heartbeats are printed every ``--heartbeat-interval`` seconds (default
300s = 5 min). The very first tick fires early (~60s) so the user gets
a quick confirmation things are running, then the loop settles into
the configured interval. Each heartbeat parses the per-year CSV
(``year_log_<variant>_s<seed>.csv``) to produce a single-line summary
of the **last completed episode** — year-1 vs year-N (clearing price,
secondary price, compliance, green share) plus that episode's
across-years mean — followed by a last-50-episode mean reward read
from ``training_log_<variant>_s<seed>.csv``. Field order — episode,
price, secondary, compliance, greening, reward — is intentional:
physical / market signals first, learning-quality reward last. A
per-job ``ETA`` is appended once enough episodes have elapsed to
estimate a rate; in multi-job sweeps an aggregate ``ETA total`` banner
(with sweep-wide elapsed wall time) is printed once per tick combining
the slowest running job with the queued backlog at the configured
worker count. When the year CSV does not exist yet (e.g. during
behavioural-cloning pretraining), the heartbeat falls back to the last
informative line of the captured log file.

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


def _read_csv_tail(
    csv_path: str, tail_bytes: int
) -> tuple[list[str] | None, list[dict]]:
    """Read header + tail of a CSV file.

    Returns ``(header_columns, parsed_rows)``. Returns ``(None, [])`` when
    the file is missing or empty. Skips partial trailing lines and any
    row whose first cell is not a signed integer (the ``episode`` column
    is always an int in our CSVs).
    """
    try:
        size = os.path.getsize(csv_path)
        if size == 0:
            return None, []
        with open(csv_path, "rb") as f:
            header_line = f.readline()
            if not header_line:
                return None, []
            header = header_line.decode("utf-8", errors="replace").rstrip("\r\n").split(",")
            if size <= len(header_line) + tail_bytes:
                body = f.read()
            else:
                f.seek(max(len(header_line), size - tail_bytes))
                f.readline()  # discard partial first line of window
                body = f.read()
        body_text = body.decode("utf-8", errors="replace")
    except OSError:
        return None, []

    import csv as _csv
    import io as _io

    rows: list[dict] = []
    for raw in _csv.reader(_io.StringIO(body_text)):
        if len(raw) != len(header):
            continue
        ep_cell = raw[0] if raw else ""
        if not ep_cell or ep_cell == "episode" or not ep_cell.lstrip("-").isdigit():
            continue
        rows.append(dict(zip(header, raw)))
    return header, rows


def _summarize_csv(
    year_csv: str,
    *,
    training_csv: str | None = None,
    n_episodes: int | None = None,
    tail_bytes: int = 524288,
    reward_window: int = 50,
) -> tuple[str | None, int | None]:
    """Build a one-line training-progress summary.

    Year-level signals (price, secondary price, compliance, green) are
    read from ``year_csv`` and report the **last completed episode**:
    year-1 vs year-12 plus the episode-mean across years 1..12. Reward is
    the mean across agents over the last ``reward_window`` episodes from
    ``training_csv`` (no arrow — the comparison is against the previous
    heartbeat line).

    Returns ``(line, last_ep)``. ``line`` is ``None`` when the year CSV
    has no usable rows yet; ``last_ep`` is the most recent episode index
    seen, or ``None``.
    """
    header, yr_rows = _read_csv_tail(year_csv, tail_bytes)
    if header is None or not yr_rows:
        return None, None

    def _f(d: dict, key: str, default: float = float("nan")) -> float:
        v = d.get(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _mean(vals: list[float]) -> float:
        clean = [v for v in vals if v == v]  # drop NaN
        return sum(clean) / len(clean) if clean else float("nan")

    def _ok(x: float) -> bool:
        return x == x  # not NaN

    # Group tail rows by episode, then pick the most-recent *complete* episode.
    last_row_ep = int(_f(yr_rows[-1], "episode", -1))
    if last_row_ep < 0:
        return None, None
    by_ep: dict[int, list[dict]] = {}
    for r in yr_rows:
        ep = int(_f(r, "episode", -1))
        if ep < 0:
            continue
        by_ep.setdefault(ep, []).append(r)
    if not by_ep:
        return None, None

    # Determine the expected episode length: the maximum number of year-rows
    # seen for any single episode in the tail window.  For a full 12-year
    # episode this will be 12; it may be smaller very early in training
    # before any episode has completed.
    expected_years = max(len(rows) for rows in by_ep.values())

    # Walk backwards through episodes (highest → lowest) and pick the first
    # one that is "complete": it has at least ``expected_years`` rows and
    # the year values form a contiguous run (no gaps / duplicate rows from
    # a mid-write snapshot).
    target_ep = max(by_ep.keys())  # fallback: most-recent started episode
    for ep in sorted(by_ep.keys(), reverse=True):
        rows = by_ep[ep]
        if len(rows) < expected_years:
            continue
        years_present = sorted(int(_f(r, "year", -1)) for r in rows)
        if years_present[0] < 0:
            continue
        if years_present[-1] == years_present[0] + len(years_present) - 1:
            target_ep = ep
            break

    ep_rows = sorted(by_ep[target_ep], key=lambda r: int(_f(r, "year", 0)))
    if not ep_rows:
        return None, None
    last_ep = target_ep

    # Discover agent-count from the year_log header.
    n_total = sum(
        1 for h in header
        if h.startswith("reward_A") and h[len("reward_A"):].isdigit()
    )

    y1_row = ep_rows[0]
    y12_row = ep_rows[-1]

    def _row_green(r: dict) -> float:
        if n_total == 0:
            return float("nan")
        return _mean([_f(r, f"green_frac_A{i+1}") for i in range(n_total)])

    def _row_comp(r: dict) -> float:
        if n_total == 0:
            return float("nan")
        compliant = 0
        total = 0
        for i in range(n_total):
            v = _f(r, f"shortfall_A{i+1}")
            if not _ok(v):
                continue
            total += 1
            if v <= 1e-6:
                compliant += 1
        return (compliant / total) if total > 0 else float("nan")

    # Year-1 / Year-N / episode-mean per metric.
    px_y1 = _f(y1_row, "clearing_price")
    px_yn = _f(y12_row, "clearing_price")
    px_mu = _mean([_f(r, "clearing_price") for r in ep_rows])

    sec_y1 = _f(y1_row, "secondary_price")
    sec_yn = _f(y12_row, "secondary_price")
    sec_mu = _mean([_f(r, "secondary_price") for r in ep_rows])

    g_y1 = _row_green(y1_row)
    g_yn = _row_green(y12_row)

    c_mu = _mean([_row_comp(r) for r in ep_rows])

    # Auction undersubscription count: years where total demand
    # (Σ bid_qty_mult_i × estimate_need_i) is below the year's
    # ``auction_volume`` supply. Reported as ``und K/N`` (K
    # undersubscribed years out of N years in the episode).
    und_count = 0
    und_total = 0
    for r in ep_rows:
        supply = _f(r, "auction_volume")
        if not _ok(supply) or supply <= 0:
            continue
        demand = 0.0
        any_bid = False
        for i in range(n_total):
            mult = _f(r, f"bid_qty_mult_A{i+1}")
            need = _f(r, f"estimate_need_A{i+1}")
            if _ok(mult) and _ok(need):
                demand += mult * need
                any_bid = True
        if not any_bid:
            continue
        und_total += 1
        if demand < supply:
            und_count += 1

    # Reward: last-N episodes mean across agents (from training CSV).
    r_now: float = float("nan")
    sec_match: float = float("nan")
    quality_score: float = float("nan")
    if training_csv is not None:
        _ep_header, ep_rows_train = _read_csv_tail(training_csv, tail_bytes)
        if _ep_header and ep_rows_train:
            n_total_train = sum(
                1 for h in _ep_header
                if h.startswith("reward_A") and h[len("reward_A"):].isdigit()
            )
            recent = ep_rows_train[-reward_window:]
            if n_total_train > 0:
                r_now_vals: list[float] = []
                for r in recent:
                    r_now_vals.append(
                        _mean([_f(r, f"reward_A{i+1}") for i in range(n_total_train)])
                    )
                r_now = _mean(r_now_vals)
            # Match rate and quality score from the latest episode of the training CSV.
            sec_match = _f(ep_rows_train[-1], "secondary_match_rate")
            quality_score = _f(ep_rows_train[-1], "quality_score")

    # Format. Field order: Ep | px | sec | comp | green | R̄.
    parts: list[str] = []
    if n_episodes and n_episodes > 0:
        pct = 100.0 * last_ep / n_episodes
        parts.append(f"Ep {last_ep}/{n_episodes} ({pct:.1f}%)")
    else:
        parts.append(f"Ep {last_ep}")

    if _ok(px_y1) and _ok(px_yn):
        s = f"px {px_y1:.0f}→{px_yn:.0f}"
        if _ok(px_mu):
            s += f" (μ{px_mu:.0f})"
        parts.append(s)

    if und_total > 0:
        parts.append(f"und {und_count}/{und_total}")

    if _ok(sec_y1) or _ok(sec_yn):
        if _ok(sec_y1) and _ok(sec_yn):
            s = f"sec {sec_y1:.0f}→{sec_yn:.0f}"
        else:
            s = f"sec {sec_yn if _ok(sec_yn) else sec_y1:.0f}"
        extra: list[str] = []
        if _ok(sec_mu):
            extra.append(f"μ{sec_mu:.0f}")
        if _ok(sec_match):
            extra.append(f"m{sec_match*100:.0f}%")
        if extra:
            s += f" ({', '.join(extra)})"
        parts.append(s)

    if _ok(c_mu):
        parts.append(f"comp {c_mu*100:.0f}%")

    if _ok(g_y1) and _ok(g_yn):
        parts.append(f"green {g_y1*100:.0f}→{g_yn*100:.0f}%")

    if _ok(r_now):
        parts.append(f"R̄ {r_now:+.1f}")

    if _ok(quality_score):
        parts.append(f"Q={quality_score:+.2f}")

    return " | ".join(parts), last_ep


def _format_eta(seconds: float) -> str:
    """Format a duration in seconds as a short ETA string.

    Returns ``"?"`` for non-positive / non-finite values (covers the early
    period of a job before any episode delta is observable).
    """
    if not (seconds == seconds) or seconds <= 0 or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


class _Heartbeat:
    """Background thread that periodically prints a condensed progress line.

    For each running job we prefer to summarise the per-episode CSV
    (``training_log_<variant>_s<seed>.csv``) so the line shows real
    training-quality signal — clearing price trajectory, mean reward,
    compliance rate, green-investment progress — instead of whatever
    happened to be the most recent log line.

    When the CSV does not yet exist (e.g. during behavioural-cloning
    pretraining, before the trainer's CSV writer has emitted its first
    row), we fall back to the last informative line of the captured
    stdout/stderr log.
    """

    def __init__(
        self,
        interval: float,
        stream=None,
        *,
        n_workers: int = 1,
        total_jobs: int | None = None,
        first_interval: float | None = None,
    ):
        # Floor the interval to a small positive value to keep the loop sane
        # at near-zero settings (e.g. unit tests) without blocking forever.
        self.interval = max(0.05, float(interval))
        # First-tick interval: a single early heartbeat so the user gets
        # confirmation things are moving without flooding the terminal.
        # Defaults to min(60s, interval); never larger than ``interval``.
        if first_interval is None:
            fi = min(60.0, self.interval)
        else:
            fi = max(0.05, float(first_interval))
        self.first_interval = min(fi, self.interval)
        self._lock = threading.Lock()
        # (variant, seed) -> {"log": str, "csv": str|None, "csv_year": str|None,
        #                     "n_eps": int|None,
        #                     "first_seen_t": float|None, "first_seen_ep": int|None,
        #                     "last_ep": int|None}
        self._jobs: dict[tuple[str, int], dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Allow tests to inject a writable stream; default to current stderr
        # at print-time (so output still respects redirections).
        self._stream = stream
        # Sweep-wide context for aggregate ETA. ``total_jobs`` is the number
        # of (variant × seed) jobs the launcher submitted; it is used together
        # with the count of jobs that have already completed to estimate the
        # wall-clock remaining for the *whole* sweep, not just the running
        # jobs. ``n_workers`` is the parallel-worker count.
        self._n_workers = max(1, int(n_workers))
        self._total_jobs = int(total_jobs) if total_jobs is not None else None
        self._completed_jobs = 0
        # Sweep wall-clock start; printed alongside the aggregate ETA banner
        # as ``elapsed`` so the user always sees how long the sweep has been
        # running, not just how much remains.
        self._start_time = time.time()

    def set_completed(self, completed: int) -> None:
        """Record how many jobs the launcher has finished (for ETA math)."""
        with self._lock:
            self._completed_jobs = int(completed)

    def add(
        self,
        variant: str,
        seed: int,
        log_path: str,
        csv_path: str | None = None,
        n_episodes: int | None = None,
        csv_year: str | None = None,
    ) -> None:
        with self._lock:
            self._jobs[(variant, seed)] = {
                "log": log_path,
                "csv": csv_path,
                "csv_year": csv_year,
                "n_eps": n_episodes,
                "first_seen_t": None,
                "first_seen_ep": None,
                "last_ep": None,
            }

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
        # Use a short ``first_interval`` for the very first tick so the user
        # gets a quick "things are running" confirmation, then settle into
        # the (longer) ``interval`` for subsequent heartbeats.
        wait = self.first_interval
        while not self._stop.wait(wait):
            wait = self.interval
            with self._lock:
                snapshot = [(key, dict(meta)) for key, meta in self._jobs.items()]
                completed = self._completed_jobs
                total_jobs = self._total_jobs
                n_workers = self._n_workers
                start_time = self._start_time
            if not snapshot:
                continue

            # Per-job ETA accumulator: list of (remaining_seconds_for_job,
            # estimated_full_job_seconds). We use these to estimate the
            # aggregate sweep ETA below.
            eta_running: list[float] = []
            full_job_estimates: list[float] = []

            now = time.time()
            for (variant, seed), meta in snapshot:
                line: str | None = None
                last_ep: int | None = None

                # 1) Preferred: structured summary from the year-level CSV
                #    (last episode: year-1 vs year-N + episode mean) plus
                #    a last-50-episode mean reward from the per-episode CSV.
                year_csv = meta.get("csv_year")
                train_csv = meta.get("csv")
                if year_csv and os.path.exists(year_csv):
                    summary, last_ep = _summarize_csv(
                        year_csv,
                        training_csv=train_csv if (train_csv and os.path.exists(train_csv)) else None,
                        n_episodes=meta.get("n_eps"),
                    )
                    if summary:
                        line = summary

                # 2) Fallback: last informative line of the captured log.
                if line is None:
                    line = _tail_last_meaningful_line(meta["log"])

                # Update per-job ETA tracking (only when we have a structured
                # episode count from the CSV — log-tail mode can't time-budget).
                eta_str = ""
                if last_ep is not None and last_ep > 0:
                    n_eps = meta.get("n_eps")
                    first_t: float | None = None
                    first_ep: int | None = None
                    with self._lock:
                        entry = self._jobs.get((variant, seed))
                        if entry is not None:
                            if entry.get("first_seen_t") is None:
                                entry["first_seen_t"] = now
                                entry["first_seen_ep"] = last_ep
                            entry["last_ep"] = last_ep
                            first_t = entry["first_seen_t"]
                            first_ep = entry["first_seen_ep"]
                    # ``entry`` may be None if the job was removed concurrently
                    # by the launcher between the snapshot read and now; in
                    # that case skip the ETA update (the job is already done).
                    if first_t is not None and first_ep is not None:
                        elapsed = now - first_t
                        delta_eps = last_ep - first_ep
                        if delta_eps > 0 and elapsed > 0 and n_eps:
                            sec_per_ep = elapsed / delta_eps
                            remaining = max(0, n_eps - last_ep) * sec_per_ep
                            eta_running.append(remaining)
                            full_job_estimates.append(n_eps * sec_per_ep)
                            eta_str = f" | ETA {_format_eta(remaining)}"

                if line:
                    line = line + eta_str
                    # Truncate very long lines so the terminal stays readable.
                    if len(line) > 220:
                        line = line[:217] + "..."
                    out = self._stream if self._stream is not None else sys.stderr
                    print(
                        f"[sweep] [LIVE  {variant} s={seed}] {line}",
                        file=out,
                        flush=True,
                    )

            # Aggregate sweep-wide ETA. Two contributions:
            #   (a) finishing the currently-running jobs — bounded below by
            #       the slowest running job (they execute in parallel).
            #   (b) clearing the queue of not-yet-started jobs at
            #       ``n_workers`` jobs in flight using the mean per-job
            #       runtime estimate.
            # We only print the aggregate ETA when there is more than one
            # job in the sweep so that single-job invocations stay quiet.
            if (
                eta_running
                and total_jobs is not None
                and total_jobs > 1
            ):
                running_count = len(eta_running)
                queued = max(0, total_jobs - completed - running_count)
                slowest_running = max(eta_running)
                if queued > 0 and full_job_estimates:
                    mean_full = sum(full_job_estimates) / len(full_job_estimates)
                    queued_wall = (queued * mean_full) / max(1, n_workers)
                else:
                    queued_wall = 0.0
                agg = slowest_running + queued_wall
                elapsed_total = max(0.0, now - start_time)
                done_msg = (
                    f"({completed}/{total_jobs} done, "
                    f"{running_count} running, {queued} queued)"
                )
                out = self._stream if self._stream is not None else sys.stderr
                print(
                    f"[sweep] [ETA total ≈ {_format_eta(agg)}, "
                    f"elapsed {_format_eta(elapsed_total)}] {done_msg}",
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
        "--heartbeat-interval", type=float, default=300.0,
        help=(
            "Seconds between live-progress heartbeats printed to the terminal "
            "(default: 300, i.e. one heartbeat every 5 minutes). The very "
            "first heartbeat fires earlier (after ~60s) so the user gets a "
            "quick confirmation things are running. Each heartbeat shows a "
            "one-line summary per running job: episode progress, year-1 vs "
            "year-N (clearing price, secondary, compliance, green) of the "
            "latest episode plus that episode's mean, and a last-50-episode "
            "mean reward. Use --quiet to disable."
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
            f"(first tick early; per-job: ep, last-episode Y1→YN px/sec/comp/green, R̄ last-50)",
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
        heartbeat = _Heartbeat(
            args.heartbeat_interval,
            n_workers=n_workers,
            total_jobs=n_jobs,
        )
        heartbeat.start()

    # Cache n_episodes per resolved variant YAML (one read per variant) so
    # the heartbeat can show "Ep N/total (XX%)".
    _n_eps_cache: dict[str, int | None] = {}

    def _get_n_episodes(yaml_path: str) -> int | None:
        if yaml_path in _n_eps_cache:
            return _n_eps_cache[yaml_path]
        try:
            cfg = _load_base_config(yaml_path)
            ne = int(cfg.get("simulation", {}).get("n_episodes", 0)) or None
        except (OSError, ValueError, TypeError, KeyError):
            ne = None
        _n_eps_cache[yaml_path] = ne
        return ne

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
                # much output. We also point the heartbeat at the per-episode
                # CSV so it can summarise structured training metrics
                # (clearing price, mean reward, compliance rate, greening).
                if heartbeat is not None:
                    csv_path = os.path.join(
                        results_dir, f"training_log_{variant_name}_s{seed}.csv"
                    )
                    csv_year = os.path.join(
                        results_dir, f"year_log_{variant_name}_s{seed}.csv"
                    )
                    heartbeat.add(
                        variant_name, seed,
                        _job_log_path(results_dir, variant_name, seed),
                        csv_path=csv_path,
                        n_episodes=_get_n_episodes(yaml_path),
                        csv_year=csv_year,
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
                    heartbeat.set_completed(completed)
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

"""
compute_setup.py
================
Architecture-aware compute configuration for ETS MARL training.

Tunes process-level threading for PyTorch / OpenMP / MKL so that runs make
reasonable use of the underlying machine without changing training quality.
Detection is portable: it adapts to small laptops, CI runners, and large
cloud VMs (e.g. Azure Standard_D16ds_v5 — 16 vCPU / 64 GB RAM) alike.

Why this is needed
------------------
The networks in this project are small (hidden_size=256, mini_batch=64). On
machines with many vCPUs, the BLAS thread pool can spend more time on
synchronisation than on useful matmuls, so PyTorch's default of
``torch.get_num_threads() == n_physical_cores`` actually *under-utilises* the
box (single-process, BLAS waiting on threads).

The right policy for this workload is:

* Cap intra-op threads per process to a small number (≤ 4 by default), which
  is where small-MLP throughput plateaus on x86.
* Reserve the remaining cores for parallel work at the *process* level —
  e.g. running multiple seeds concurrently from ``scripts/train.py
  --parallel-seeds N``. Each subprocess gets its own ≤ 4-thread budget,
  multiplying CPU utilisation without altering any single seed's RNG /
  training trajectory.

Behaviour summary
-----------------
* ``configure_compute()`` sets ``torch.set_num_threads`` /
  ``set_num_interop_threads`` and the standard env vars
  (``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
  ``NUMEXPR_NUM_THREADS``, ``KMP_BLOCKTIME``).
* If called with ``num_threads=None`` and ``total_threads=None``, picks a
  per-process budget automatically from the detected core count.
* Returns a small dict describing what was applied so callers can log it.
* All numerical / RNG state is left untouched — this only changes how many
  CPU threads the BLAS / OMP pools are allowed to use.

Nothing in this module changes training results. It only changes how many
worker threads the linear-algebra pools are allowed to spawn.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_architecture() -> dict:
    """Return a description of the host architecture.

    Keys
    ----
    logical_cores : int
        Number of logical CPUs (includes hyperthreads / SMT siblings).
    physical_cores : int or None
        Number of physical CPU cores if detectable, else ``None``.
    total_memory_gb : float or None
        Total system memory in GiB if detectable, else ``None``.
    machine : str
        ``platform.machine()`` (e.g. ``"x86_64"``, ``"arm64"``).
    system : str
        ``platform.system()`` (e.g. ``"Linux"``, ``"Windows"``).
    processor : str
        ``platform.processor()`` (best-effort, may be empty on Linux).
    """
    info: dict = {
        "logical_cores": os.cpu_count() or 1,
        "physical_cores": None,
        "total_memory_gb": None,
        "machine": platform.machine(),
        "system": platform.system(),
        "processor": platform.processor() or "",
    }

    # psutil is the only reliable way to get physical-core count + RAM
    # cross-platform. Treat as optional to avoid a hard dependency.
    try:
        import psutil  # type: ignore

        phys = psutil.cpu_count(logical=False)
        if phys:
            info["physical_cores"] = int(phys)
        vm = psutil.virtual_memory()
        info["total_memory_gb"] = round(vm.total / (1024 ** 3), 2)
    except Exception:
        # Linux fallback for physical-core count via /proc/cpuinfo.
        if info["system"] == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                core_ids = set()
                cur_phys = None
                for line in cpuinfo.splitlines():
                    if line.startswith("physical id"):
                        cur_phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id") and cur_phys is not None:
                        core_ids.add((cur_phys, line.split(":", 1)[1].strip()))
                if core_ids:
                    info["physical_cores"] = len(core_ids)
            except OSError:
                pass
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            info["total_memory_gb"] = round(kb / (1024 ** 2), 2)
                            break
            except OSError:
                pass

    return info


# ---------------------------------------------------------------------------
# Default thread budget
# ---------------------------------------------------------------------------

def _default_threads_per_process(arch: dict) -> int:
    """Pick a reasonable intra-op thread count for a single training process.

    Policy (calibrated for the small MLPs used here, hidden_size≈256):

    * 1–2 cores  : use what we have.
    * 3–4 cores  : 2 threads (leave room for IO / OS).
    * 5–8 cores  : 4 threads.
    * 9–32 cores : 4 threads (BLAS overhead dominates above ~4 for these
                   tiny matmuls; remaining cores are best used by running
                   parallel seeds in separate processes).
    * 33+ cores  : 6 threads.

    These are *defaults* — callers can pass ``num_threads`` or set
    ``ETS_NUM_THREADS`` to override.
    """
    cores = arch.get("physical_cores") or arch.get("logical_cores") or 1
    if cores <= 2:
        return max(1, cores)
    if cores <= 4:
        return 2
    if cores <= 32:
        return 4
    return 6


# ---------------------------------------------------------------------------
# Public configuration entry point
# ---------------------------------------------------------------------------

def configure_compute(
    num_threads: Optional[int] = None,
    total_threads: Optional[int] = None,
    quiet: bool = False,
) -> dict:
    """Configure the process-level CPU thread budget.

    Parameters
    ----------
    num_threads : int, optional
        Explicit intra-op thread count for this process. When omitted, falls
        back to the ``ETS_NUM_THREADS`` env var, then to a sensible default
        chosen from the detected architecture.
    total_threads : int, optional
        For multi-process launches (e.g. ``--parallel-seeds N``): the total
        number of threads available to the launcher. Each child then gets
        ``total_threads // N`` via the explicit ``num_threads`` arg. This
        argument is informational here and is recorded in the returned dict;
        the actual division is handled by the caller.
    quiet : bool
        If True, suppress the one-line banner.

    Returns
    -------
    dict
        Description of what was applied (architecture, num_threads,
        interop_threads, env vars set). Useful for logging.

    Notes
    -----
    * Safe to call multiple times — later calls overwrite earlier settings.
    * Must be called *before* the first heavy BLAS call to take effect on
      OMP / MKL pools. Calling it at the top of ``main()`` is fine; PyTorch
      reads ``set_num_threads`` lazily.
    * Does not change RNG state, network topology, or any numerical path.
    """
    arch = detect_architecture()

    # Resolve per-process thread count: explicit arg > env var > auto.
    if num_threads is None:
        env_override = os.environ.get("ETS_NUM_THREADS")
        if env_override:
            try:
                num_threads = max(1, int(env_override))
            except ValueError:
                num_threads = None
    if num_threads is None:
        num_threads = _default_threads_per_process(arch)

    num_threads = max(1, int(num_threads))

    # Set BLAS / OMP env vars. Do this before touching torch BLAS so the
    # underlying libraries pick them up. KMP_BLOCKTIME=0 prevents Intel OMP
    # from spinning idle threads, which is what causes the "low utilisation
    # but high context-switch" pattern on large Azure VMs.
    env_updates = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "KMP_BLOCKTIME": "0",
    }
    for k, v in env_updates.items():
        os.environ.setdefault(k, v)  # respect a user-set value

    # Apply to torch if it's already imported. Wrapped in a try because some
    # torch builds raise if interop threads are set after the first parallel
    # work item.
    interop_applied = None
    try:
        import torch  # local import: avoid forcing torch on callers

        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass
        try:
            # Interop = the queue feeding the intra-op pool. 1 is correct for
            # almost every PPO-style workload (no nested parallel regions).
            torch.set_num_interop_threads(1)
            interop_applied = 1
        except RuntimeError:
            # Already initialised — ignore. Not fatal.
            interop_applied = torch.get_num_interop_threads()
    except ImportError:
        pass

    summary = {
        "architecture": arch,
        "num_threads": num_threads,
        "interop_threads": interop_applied,
        "total_threads": total_threads,
        "env_updates": env_updates,
    }

    if not quiet:
        _print_banner(summary)

    return summary


def _print_banner(summary: dict) -> None:
    arch = summary["architecture"]
    cores_str = f"{arch['logical_cores']} logical"
    if arch.get("physical_cores"):
        cores_str += f" / {arch['physical_cores']} physical"
    mem_str = f"{arch['total_memory_gb']} GiB" if arch.get("total_memory_gb") else "?"
    line = (
        f"[compute] host={arch['system']}/{arch['machine']} cores={cores_str} "
        f"mem={mem_str} → torch.num_threads={summary['num_threads']} "
        f"interop={summary['interop_threads']}"
    )
    if summary.get("total_threads"):
        line += f" (parallel budget={summary['total_threads']})"
    print(line, file=sys.stderr)

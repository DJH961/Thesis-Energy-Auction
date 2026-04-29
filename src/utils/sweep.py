"""
sweep.py
========
Multi-config sweep helpers for ETS MARL.

This module is a small, dependency-free library that turns a *sweep spec*
YAML into a concrete list of training jobs, each consisting of:

* a fully-resolved config dict (base config + per-variant overrides),
* a unique variant name (used for the on-disk results directory),
* a seed.

The actual subprocess orchestration lives in ``scripts/sweep.py`` so this
module can be unit-tested without spinning up child processes.

Sweep spec schema (YAML)
------------------------
```yaml
base_config: configs/default.yaml          # required; relative to spec file or CWD
output_dir: results/sweeps/scarcity_msr    # required
seeds: [1, 2, 3]                           # default seeds (variants may override)
parallel_workers: 4                        # optional; default 1
threads_per_worker: null                   # optional; null => auto via ETS_NUM_THREADS

variants:                                  # required; non-empty list
  - name: tight_cap                        # required, unique, filesystem-safe
    overrides:                             # optional; nested dict of dotted-paths
      ets.cap_overhead_pct: -0.02
  - name: loose_cap_no_msr
    overrides:
      ets.cap_overhead_pct: 0.20
      ets.msr.enabled: false
    seeds: [1, 2]                          # optional; per-variant seed override
```

Override keys may use dotted paths (``ets.msr.enabled``) or nested mappings
(``{ets: {msr: {enabled: false}}}``); both are deep-merged into the base.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import yaml


# Filesystem-safe variant name: letters, digits, underscore, hyphen, dot.
# Disallow path separators and shell metachars to keep results_dir paths sane.
_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True)
class SweepJob:
    """One concrete training job in a sweep: (variant, seed, resolved_config)."""

    variant_name: str
    seed: int
    config: dict
    results_dir: str

    def display_name(self) -> str:
        return f"{self.variant_name}/s{self.seed}"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` and return a new dict.

    * Mappings are merged key-by-key.
    * Any non-mapping value in ``overrides`` replaces the corresponding value
      in ``base`` (including lists — they are *not* concatenated, to avoid
      surprising hyperparameter changes).
    * ``base`` and ``overrides`` are not mutated.
    """
    if not isinstance(base, dict):
        raise TypeError(f"deep_merge: base must be dict, got {type(base).__name__}")
    if not isinstance(overrides, dict):
        raise TypeError(
            f"deep_merge: overrides must be dict, got {type(overrides).__name__}"
        )

    result: dict = copy.deepcopy(base)
    for key, ov_val in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(ov_val, dict)
        ):
            result[key] = deep_merge(result[key], ov_val)
        else:
            result[key] = copy.deepcopy(ov_val)
    return result


def expand_dotted_overrides(overrides: dict) -> dict:
    """Expand any dotted-path keys (``a.b.c: v``) into nested dicts.

    Mixed flat and nested override dicts are accepted; they are merged
    together. Raises ``ValueError`` if two override entries collide on the
    same scalar slot with conflicting values.
    """
    if not isinstance(overrides, dict):
        raise TypeError(
            f"expand_dotted_overrides: expected dict, got {type(overrides).__name__}"
        )

    out: dict = {}
    for raw_key, value in overrides.items():
        if not isinstance(raw_key, str):
            raise TypeError(
                f"override key must be str, got {type(raw_key).__name__}: {raw_key!r}"
            )
        parts = raw_key.split(".") if "." in raw_key else [raw_key]
        if any(p == "" for p in parts):
            raise ValueError(f"empty segment in override key: {raw_key!r}")

        # Build a nested dict for this single override and merge it in.
        nested: Any = value
        for part in reversed(parts):
            nested = {part: nested}
        out = deep_merge(out, nested)
    return out


# ---------------------------------------------------------------------------
# Spec loading & validation
# ---------------------------------------------------------------------------

def _resolve_path(p: str, anchor_dir: str | None) -> str:
    """Resolve ``p`` either absolutely or relative to ``anchor_dir``."""
    if os.path.isabs(p):
        return p
    if anchor_dir and not os.path.exists(p):
        candidate = os.path.join(anchor_dir, p)
        if os.path.exists(candidate):
            return candidate
    return p


def load_sweep_spec(spec_path: str) -> dict:
    """Load and validate a sweep spec YAML file.

    Returns the validated dict (with required fields populated and types
    checked). Raises ``ValueError`` on any structural problem.
    """
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"sweep spec not found: {spec_path}")

    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(
            f"sweep spec must be a YAML mapping at top level, got {type(spec).__name__}"
        )

    spec_dir = os.path.dirname(os.path.abspath(spec_path))
    return validate_sweep_spec(spec, spec_dir=spec_dir)


def validate_sweep_spec(spec: dict, spec_dir: str | None = None) -> dict:
    """Validate a sweep spec dict and return a normalised copy.

    Pure function; does not read the base config file (callers do that so
    that this function remains test-friendly without filesystem fixtures).
    """
    if not isinstance(spec, dict):
        raise ValueError(f"sweep spec must be a dict, got {type(spec).__name__}")

    out = copy.deepcopy(spec)

    base = out.get("base_config")
    if not isinstance(base, str) or not base.strip():
        raise ValueError("sweep spec: 'base_config' (str) is required")
    out["base_config"] = _resolve_path(base, spec_dir)

    output_dir = out.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("sweep spec: 'output_dir' (str) is required")

    seeds = out.get("seeds", [])
    if seeds is not None and not isinstance(seeds, list):
        raise ValueError("sweep spec: 'seeds' must be a list of ints")
    if seeds:
        for s in seeds:
            if not isinstance(s, int) or isinstance(s, bool):
                raise ValueError(f"sweep spec: seed must be int, got {s!r}")
    out["seeds"] = list(seeds or [])

    pw = out.get("parallel_workers", 1)
    if not isinstance(pw, int) or isinstance(pw, bool) or pw < 1:
        raise ValueError("sweep spec: 'parallel_workers' must be a positive int")
    out["parallel_workers"] = pw

    tpw = out.get("threads_per_worker", None)
    if tpw is not None:
        if not isinstance(tpw, int) or isinstance(tpw, bool) or tpw < 1:
            raise ValueError(
                "sweep spec: 'threads_per_worker' must be a positive int or null"
            )
    out["threads_per_worker"] = tpw

    variants = out.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("sweep spec: 'variants' must be a non-empty list")

    seen_names: set[str] = set()
    norm_variants = []
    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            raise ValueError(f"sweep spec: variant #{i} must be a dict")
        name = v.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"sweep spec: variant #{i} missing 'name'")
        if not _VARIANT_NAME_RE.match(name):
            raise ValueError(
                f"sweep spec: variant name {name!r} is not filesystem-safe "
                f"(allowed: letters, digits, '_', '-', '.')"
            )
        if name in seen_names:
            raise ValueError(f"sweep spec: duplicate variant name: {name!r}")
        seen_names.add(name)

        overrides = v.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(
                f"sweep spec: variant {name!r} 'overrides' must be a dict"
            )

        v_seeds = v.get("seeds", None)
        if v_seeds is not None:
            if not isinstance(v_seeds, list) or not v_seeds:
                raise ValueError(
                    f"sweep spec: variant {name!r} 'seeds' must be a non-empty list"
                )
            for s in v_seeds:
                if not isinstance(s, int) or isinstance(s, bool):
                    raise ValueError(
                        f"sweep spec: variant {name!r} seed must be int, got {s!r}"
                    )

        norm_variants.append(
            {
                "name": name,
                "overrides": overrides,
                "seeds": list(v_seeds) if v_seeds is not None else None,
            }
        )

    out["variants"] = norm_variants

    # If no global seeds and no variant overrides them, that's an error.
    if not out["seeds"] and any(v["seeds"] is None for v in norm_variants):
        raise ValueError(
            "sweep spec: either top-level 'seeds' or per-variant 'seeds' "
            "must be provided for every variant"
        )

    return out


# ---------------------------------------------------------------------------
# Job materialisation
# ---------------------------------------------------------------------------

def resolve_variant_config(
    base_config: dict,
    variant: dict,
    output_dir: str,
) -> tuple[dict, str]:
    """Build the resolved config for a single variant.

    * Deep-merges the variant's overrides (after dotted-path expansion) onto
      a deep copy of ``base_config``.
    * Forces ``logging.results_dir`` to ``<output_dir>/<variant_name>`` so
      each variant writes to its own directory and seeds within a variant
      collide on filenames only if they share a seed (existing behaviour).

    Returns ``(resolved_config, results_dir)``.
    """
    name = variant["name"]
    overrides = expand_dotted_overrides(variant["overrides"])
    resolved = deep_merge(base_config, overrides)

    results_dir = os.path.join(output_dir, name)
    resolved.setdefault("logging", {})
    if not isinstance(resolved["logging"], dict):
        raise ValueError(
            f"base config 'logging' section must be a dict, got "
            f"{type(resolved['logging']).__name__}"
        )
    resolved["logging"]["results_dir"] = results_dir

    return resolved, results_dir


def build_jobs(spec: dict, base_config: dict) -> list[SweepJob]:
    """Materialise the full list of (variant, seed) jobs from a validated spec."""
    output_dir = spec["output_dir"]
    default_seeds: list[int] = spec["seeds"]

    jobs: list[SweepJob] = []
    for v in spec["variants"]:
        seeds = v["seeds"] if v["seeds"] is not None else default_seeds
        resolved, results_dir = resolve_variant_config(base_config, v, output_dir)
        for seed in seeds:
            jobs.append(
                SweepJob(
                    variant_name=v["name"],
                    seed=int(seed),
                    config=copy.deepcopy(resolved),
                    results_dir=results_dir,
                )
            )
    return jobs


def write_resolved_config(job_config: dict, dest_path: str) -> None:
    """Serialise a resolved config dict to YAML at ``dest_path``."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, "w") as f:
        yaml.safe_dump(job_config, f, sort_keys=False)


def iter_variant_yaml_paths(
    spec: dict, base_config: dict, scratch_dir: str
) -> Iterable[tuple[str, str, list[int]]]:
    """Write one YAML per variant to ``scratch_dir`` and yield the paths.

    Yields ``(variant_name, yaml_path, seeds)`` triples. Used by the
    subprocess launcher in ``scripts/sweep.py``: each variant only needs
    one YAML on disk regardless of how many seeds run against it.
    """
    output_dir = spec["output_dir"]
    default_seeds: list[int] = spec["seeds"]
    os.makedirs(scratch_dir, exist_ok=True)

    for v in spec["variants"]:
        seeds = v["seeds"] if v["seeds"] is not None else default_seeds
        resolved, _results_dir = resolve_variant_config(base_config, v, output_dir)
        yaml_path = os.path.join(scratch_dir, f"{v['name']}.yaml")
        write_resolved_config(resolved, yaml_path)
        yield v["name"], yaml_path, list(seeds)

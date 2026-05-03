"""Tests for src/utils/sweep.py: deep merge, override expansion, validation."""

from __future__ import annotations

import os

import pytest
import yaml

from src.utils.sweep import (
    SweepJob,
    build_jobs,
    deep_merge,
    expand_dotted_overrides,
    iter_variant_yaml_paths,
    load_sweep_spec,
    resolve_variant_config,
    validate_sweep_spec,
)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_merges_disjoint_keys(self):
        out = deep_merge({"a": 1}, {"b": 2})
        assert out == {"a": 1, "b": 2}

    def test_overrides_scalar(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_recursive_merge(self):
        base = {"ets": {"cap": 1.0, "msr": {"enabled": True, "rate": 0.24}}}
        ov = {"ets": {"msr": {"enabled": False}}}
        out = deep_merge(base, ov)
        assert out == {
            "ets": {"cap": 1.0, "msr": {"enabled": False, "rate": 0.24}}
        }

    def test_lists_are_replaced_not_concatenated(self):
        # Concatenating would silently double hyperparameters; replace instead.
        base = {"seeds": [1, 2, 3]}
        ov = {"seeds": [10]}
        assert deep_merge(base, ov) == {"seeds": [10]}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        ov = {"a": {"c": 2}}
        out = deep_merge(base, ov)
        assert base == {"a": {"b": 1}}
        assert ov == {"a": {"c": 2}}
        out["a"]["b"] = 999
        assert base == {"a": {"b": 1}}  # still unchanged

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError):
            deep_merge([1, 2], {"a": 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            deep_merge({"a": 1}, "nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# expand_dotted_overrides
# ---------------------------------------------------------------------------

class TestExpandDottedOverrides:
    def test_single_level(self):
        assert expand_dotted_overrides({"a": 1}) == {"a": 1}

    def test_dotted_path(self):
        assert expand_dotted_overrides({"ets.msr.enabled": False}) == {
            "ets": {"msr": {"enabled": False}}
        }

    def test_mixed_flat_and_dotted(self):
        out = expand_dotted_overrides({
            "ets.cap_overhead_pct": -0.02,
            "ets.msr.enabled": False,
            "logging": {"results_dir": "/tmp/x"},
        })
        assert out == {
            "ets": {"cap_overhead_pct": -0.02, "msr": {"enabled": False}},
            "logging": {"results_dir": "/tmp/x"},
        }

    def test_empty_segment_rejected(self):
        with pytest.raises(ValueError):
            expand_dotted_overrides({"ets..msr": True})
        with pytest.raises(ValueError):
            expand_dotted_overrides({".x": 1})

    def test_non_string_key_rejected(self):
        with pytest.raises(TypeError):
            expand_dotted_overrides({1: "v"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# validate_sweep_spec
# ---------------------------------------------------------------------------

def _minimal_spec(**extra) -> dict:
    base = {
        "base_config": "configs/default.yaml",
        "output_dir": "results/sweeps/x",
        "seeds": [1, 2],
        "variants": [{"name": "v1"}],
    }
    base.update(extra)
    return base


class TestValidateSweepSpec:
    def test_minimal_ok(self):
        out = validate_sweep_spec(_minimal_spec())
        assert out["seeds"] == [1, 2]
        assert out["parallel_workers"] == 1
        assert out["threads_per_worker"] is None
        assert out["variants"][0]["overrides"] == {}
        assert out["variants"][0]["seeds"] is None

    def test_missing_base_config(self):
        spec = _minimal_spec()
        del spec["base_config"]
        with pytest.raises(ValueError, match="base_config"):
            validate_sweep_spec(spec)

    def test_missing_output_dir(self):
        spec = _minimal_spec()
        del spec["output_dir"]
        with pytest.raises(ValueError, match="output_dir"):
            validate_sweep_spec(spec)

    def test_empty_variants_rejected(self):
        with pytest.raises(ValueError, match="variants"):
            validate_sweep_spec(_minimal_spec(variants=[]))

    def test_duplicate_variant_names(self):
        spec = _minimal_spec(variants=[{"name": "v1"}, {"name": "v1"}])
        with pytest.raises(ValueError, match="duplicate"):
            validate_sweep_spec(spec)

    def test_unsafe_variant_name(self):
        spec = _minimal_spec(variants=[{"name": "../escape"}])
        with pytest.raises(ValueError, match="filesystem-safe"):
            validate_sweep_spec(spec)

    def test_seeds_must_be_ints(self):
        with pytest.raises(ValueError):
            validate_sweep_spec(_minimal_spec(seeds=["one"]))
        with pytest.raises(ValueError):
            # bool is technically int — must be rejected.
            validate_sweep_spec(_minimal_spec(seeds=[True, False]))

    def test_parallel_workers_positive(self):
        with pytest.raises(ValueError):
            validate_sweep_spec(_minimal_spec(parallel_workers=0))
        with pytest.raises(ValueError):
            validate_sweep_spec(_minimal_spec(parallel_workers=-2))

    def test_per_variant_seeds_override_required(self):
        # No top-level seeds + variant without seeds => error.
        spec = {
            "base_config": "configs/default.yaml",
            "output_dir": "results/sweeps/x",
            "variants": [{"name": "v1"}],
        }
        with pytest.raises(ValueError, match="seeds"):
            validate_sweep_spec(spec)

    def test_per_variant_seeds_override_ok(self):
        spec = {
            "base_config": "configs/default.yaml",
            "output_dir": "results/sweeps/x",
            "variants": [{"name": "v1", "seeds": [7]}],
        }
        out = validate_sweep_spec(spec)
        assert out["variants"][0]["seeds"] == [7]


# ---------------------------------------------------------------------------
# resolve_variant_config / build_jobs
# ---------------------------------------------------------------------------

BASE_CFG = {
    "ets": {"cap_overhead_pct": 0.10, "msr": {"enabled": True, "rate": 0.24}},
    "logging": {"results_dir": "results/orig"},
    "simulation": {"n_episodes": 100},
}


class TestResolveVariantConfig:
    def test_overrides_applied(self):
        variant = {
            "name": "v1",
            "overrides": {"ets.cap_overhead_pct": -0.02, "ets.msr.enabled": False},
            "seeds": None,
        }
        cfg, results_dir = resolve_variant_config(BASE_CFG, variant, "results/sweeps/x")
        assert cfg["ets"]["cap_overhead_pct"] == -0.02
        assert cfg["ets"]["msr"]["enabled"] is False
        # untouched keys preserved
        assert cfg["ets"]["msr"]["rate"] == 0.24
        assert cfg["simulation"]["n_episodes"] == 100
        # results_dir overridden
        assert cfg["logging"]["results_dir"] == os.path.join("results/sweeps/x", "v1")
        assert results_dir == cfg["logging"]["results_dir"]

    def test_base_config_not_mutated(self):
        variant = {
            "name": "v1",
            "overrides": {"ets.msr.enabled": False},
            "seeds": None,
        }
        resolve_variant_config(BASE_CFG, variant, "results/sweeps/x")
        # Original is unchanged.
        assert BASE_CFG["ets"]["msr"]["enabled"] is True
        assert BASE_CFG["logging"]["results_dir"] == "results/orig"


class TestBuildJobs:
    def test_cartesian_with_default_seeds(self):
        spec = validate_sweep_spec(_minimal_spec(variants=[
            {"name": "a"},
            {"name": "b"},
        ]))
        jobs = build_jobs(spec, BASE_CFG)
        assert len(jobs) == 4  # 2 variants x 2 seeds
        names = sorted({j.variant_name for j in jobs})
        assert names == ["a", "b"]
        seeds = sorted({j.seed for j in jobs})
        assert seeds == [1, 2]
        assert all(isinstance(j, SweepJob) for j in jobs)

    def test_per_variant_seed_override(self):
        spec = validate_sweep_spec(_minimal_spec(
            seeds=[1, 2, 3],
            variants=[
                {"name": "a"},
                {"name": "b", "seeds": [99]},
            ],
        ))
        jobs = build_jobs(spec, BASE_CFG)
        a_seeds = sorted(j.seed for j in jobs if j.variant_name == "a")
        b_seeds = sorted(j.seed for j in jobs if j.variant_name == "b")
        assert a_seeds == [1, 2, 3]
        assert b_seeds == [99]

    def test_each_job_has_independent_config(self):
        spec = validate_sweep_spec(_minimal_spec(variants=[
            {"name": "a", "overrides": {"ets.msr.enabled": False}},
            {"name": "b"},
        ]))
        jobs = build_jobs(spec, BASE_CFG)
        a_cfg = next(j.config for j in jobs if j.variant_name == "a")
        b_cfg = next(j.config for j in jobs if j.variant_name == "b")
        assert a_cfg["ets"]["msr"]["enabled"] is False
        assert b_cfg["ets"]["msr"]["enabled"] is True
        # Mutating one job's config doesn't affect another.
        a_cfg["ets"]["msr"]["enabled"] = "MUTATED"
        assert b_cfg["ets"]["msr"]["enabled"] is True


# ---------------------------------------------------------------------------
# load_sweep_spec + iter_variant_yaml_paths (filesystem-touching)
# ---------------------------------------------------------------------------

class TestLoadSweepSpec:
    def test_loads_and_validates(self, tmp_path):
        spec_path = tmp_path / "sweep.yaml"
        spec_path.write_text(yaml.safe_dump({
            "base_config": "configs/default.yaml",
            "output_dir": str(tmp_path / "out"),
            "seeds": [1],
            "variants": [{"name": "v1"}],
        }))
        out = load_sweep_spec(str(spec_path))
        assert out["seeds"] == [1]
        assert out["variants"][0]["name"] == "v1"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sweep_spec(str(tmp_path / "nope.yaml"))

    def test_resolves_base_config_relative_to_spec_dir(self, tmp_path):
        # base_config "x.yaml" resolves to a sibling of the spec.
        sibling = tmp_path / "x.yaml"
        sibling.write_text("ets: {}\n")
        spec_path = tmp_path / "sweep.yaml"
        spec_path.write_text(yaml.safe_dump({
            "base_config": "x.yaml",
            "output_dir": str(tmp_path / "out"),
            "seeds": [1],
            "variants": [{"name": "v1"}],
        }))
        out = load_sweep_spec(str(spec_path))
        assert os.path.exists(out["base_config"])
        assert out["base_config"].endswith("x.yaml")


class TestIterVariantYamlPaths:
    def test_writes_one_yaml_per_variant(self, tmp_path):
        spec = validate_sweep_spec(_minimal_spec(
            seeds=[1, 2],
            variants=[
                {"name": "a", "overrides": {"ets.msr.enabled": False}},
                {"name": "b"},
            ],
        ))
        scratch = tmp_path / "_resolved"
        results = list(iter_variant_yaml_paths(spec, BASE_CFG, str(scratch)))
        assert len(results) == 2
        names = [r[0] for r in results]
        assert sorted(names) == ["a", "b"]
        for name, yaml_path, seeds in results:
            assert os.path.exists(yaml_path)
            assert seeds == [1, 2]
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["logging"]["results_dir"].endswith(os.path.join("x", name))
            if name == "a":
                assert cfg["ets"]["msr"]["enabled"] is False
            else:
                assert cfg["ets"]["msr"]["enabled"] is True


# ---------------------------------------------------------------------------
# scripts/sweep.py: parent-side post-job compression fallback
# ---------------------------------------------------------------------------

class TestPostJobCompression:
    """The parent sweep must compress logs/checkpoints even when the child
    train.py subprocess exits abnormally and never reaches its inline
    end-of-run compression block. Without this the user is left with multi-GB
    CSVs and checkpoint dirs after every crashed/interrupted sweep job."""

    def _setup_artifacts(self, tmp_path, *, with_compress_cfg=True):
        results_dir = tmp_path / "v"
        results_dir.mkdir()

        cfg = {}
        if with_compress_cfg:
            cfg = {"logging": {"compress_on_finish": {
                "logs": True, "checkpoints": True,
                "delete_csv": True, "delete_checkpoint_dir": True,
            }}}
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        for name in ("training_log_v_s7.csv", "year_log_v_s7.csv"):
            with open(results_dir / name, "w") as f:
                f.write("episode,reward\n")
                for i in range(50):
                    f.write(f"{i},{i*0.5}\n")

        ck = results_dir / "checkpoints_v_s7"
        ck.mkdir()
        for i in range(2):
            (ck / f"agent{i}.pt").write_bytes(b"x" * 256)

        return cfg_path, results_dir, ck

    def test_compresses_leftover_csvs_and_checkpoints(self, tmp_path):
        from scripts.sweep import _compress_job_artifacts
        from src.utils.run_data import resolve_checkpoint_codec

        cfg_path, results_dir, ck = self._setup_artifacts(tmp_path)
        log_f = open(results_dir / "run_v_s7.log", "w", encoding="utf-8")
        try:
            _compress_job_artifacts(
                config_path=str(cfg_path),
                results_dir=str(results_dir),
                variant_name="v", seed=7, log_f=log_f,
            )
        finally:
            log_f.close()

        # CSVs and checkpoint dir replaced by parquet + tar archive in
        # whichever codec the environment can actually produce (zst when
        # zstandard is installed, gz fallback otherwise).
        assert not (results_dir / "training_log_v_s7.csv").exists()
        assert not (results_dir / "year_log_v_s7.csv").exists()
        assert not ck.exists()
        assert (results_dir / "training_log_v_s7.parquet").exists()
        assert (results_dir / "year_log_v_s7.parquet").exists()
        archive_suffix = {"zst": ".tar.zst", "gz": ".tar.gz", "xz": ".tar.xz"}[
            resolve_checkpoint_codec("auto")
        ]
        assert (results_dir / f"checkpoints_v_s7{archive_suffix}").exists()

    def test_noop_when_no_artifacts_left(self, tmp_path):
        """Happy path: train.py already compressed everything; the
        parent-side fallback should be a silent no-op rather than error."""
        from scripts.sweep import _compress_job_artifacts

        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump({"logging": {"compress_on_finish": {
                "logs": True, "checkpoints": True,
            }}}, f)
        results_dir = tmp_path / "v"
        results_dir.mkdir()
        log_path = results_dir / "run_v_s7.log"
        with open(log_path, "w") as log_f:
            _compress_job_artifacts(
                config_path=str(cfg_path),
                results_dir=str(results_dir),
                variant_name="v", seed=7, log_f=log_f,
            )
        # No new artefacts and no errors.
        assert sorted(p.name for p in results_dir.iterdir()) == ["run_v_s7.log"]

    def test_respects_disabled_compression(self, tmp_path):
        """When compress_on_finish.logs/checkpoints are both false the
        helper must not touch any files."""
        from scripts.sweep import _compress_job_artifacts

        results_dir = tmp_path / "v"
        results_dir.mkdir()
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump({"logging": {"compress_on_finish": {
                "logs": False, "checkpoints": False,
            }}}, f)
        csv_path = results_dir / "training_log_v_s7.csv"
        csv_path.write_text("episode,reward\n0,0.0\n")
        ck = results_dir / "checkpoints_v_s7"
        ck.mkdir()
        (ck / "a.pt").write_bytes(b"x")

        with open(results_dir / "run_v_s7.log", "w") as log_f:
            _compress_job_artifacts(
                config_path=str(cfg_path),
                results_dir=str(results_dir),
                variant_name="v", seed=7, log_f=log_f,
            )
        assert csv_path.exists()
        assert ck.is_dir()
        assert not (results_dir / "training_log_v_s7.parquet").exists()
        # No archive of any supported codec should have been produced.
        for suffix in (".tar.zst", ".tar.gz", ".tar.xz"):
            assert not (results_dir / f"checkpoints_v_s7{suffix}").exists()

    def test_keeps_sources_when_delete_flags_false(self, tmp_path):
        """delete_csv=False / delete_checkpoint_dir=False must leave the
        originals on disk alongside the compressed copies."""
        from scripts.sweep import _compress_job_artifacts
        from src.utils.run_data import resolve_checkpoint_codec

        results_dir = tmp_path / "v"
        results_dir.mkdir()
        cfg_path = tmp_path / "cfg.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump({"logging": {"compress_on_finish": {
                "logs": True, "checkpoints": True,
                "delete_csv": False, "delete_checkpoint_dir": False,
            }}}, f)
        csv_path = results_dir / "training_log_v_s7.csv"
        csv_path.write_text("episode,reward\n0,0.0\n1,1.0\n")
        ck = results_dir / "checkpoints_v_s7"
        ck.mkdir()
        (ck / "a.pt").write_bytes(b"x" * 64)

        with open(results_dir / "run_v_s7.log", "w", encoding="utf-8") as log_f:
            _compress_job_artifacts(
                config_path=str(cfg_path),
                results_dir=str(results_dir),
                variant_name="v", seed=7, log_f=log_f,
            )
        assert csv_path.exists()
        assert ck.is_dir()
        assert (results_dir / "training_log_v_s7.parquet").exists()
        archive_suffix = {"zst": ".tar.zst", "gz": ".tar.gz", "xz": ".tar.xz"}[
            resolve_checkpoint_codec("auto")
        ]
        assert (results_dir / f"checkpoints_v_s7{archive_suffix}").exists()

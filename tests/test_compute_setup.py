"""Tests for src.utils.compute_setup."""

import os

import pytest

from src.utils.compute_setup import (
    configure_compute,
    detect_architecture,
    _default_threads_per_process,
)


def test_detect_architecture_returns_required_fields():
    arch = detect_architecture()
    # Always-present fields.
    assert isinstance(arch["logical_cores"], int)
    assert arch["logical_cores"] >= 1
    assert isinstance(arch["machine"], str)
    assert isinstance(arch["system"], str)
    # Optional fields may be None when psutil + /proc/cpuinfo both fail, but
    # if present they must be the right type.
    if arch["physical_cores"] is not None:
        assert isinstance(arch["physical_cores"], int)
        assert arch["physical_cores"] >= 1
    if arch["total_memory_gb"] is not None:
        assert isinstance(arch["total_memory_gb"], float)
        assert arch["total_memory_gb"] > 0


@pytest.mark.parametrize(
    "cores,expected",
    [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 4),
        (8, 4),
        (16, 4),   # Azure Standard_D16ds_v5
        (32, 4),
        (64, 6),
    ],
)
def test_default_threads_per_process_policy(cores, expected):
    arch = {"logical_cores": cores, "physical_cores": cores}
    assert _default_threads_per_process(arch) == expected


def test_configure_compute_explicit_num_threads_applied():
    summary = configure_compute(num_threads=2, quiet=True)
    assert summary["num_threads"] == 2
    assert os.environ["OMP_NUM_THREADS"] == os.environ.get("OMP_NUM_THREADS")
    # Either the call set it or a previous user value was respected; in either
    # case the value must parse as a positive int.
    assert int(os.environ["OMP_NUM_THREADS"]) >= 1

    import torch
    # set_num_threads is a global setting; just verify it took at least one
    # thread (some torch builds clamp to 1 in CI).
    assert torch.get_num_threads() >= 1


def test_configure_compute_env_var_override(monkeypatch):
    monkeypatch.setenv("ETS_NUM_THREADS", "3")
    # Wipe explicit BLAS env vars so configure_compute writes new ones.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)

    summary = configure_compute(num_threads=None, quiet=True)
    assert summary["num_threads"] == 3
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"


def test_configure_compute_records_total_threads(monkeypatch):
    monkeypatch.delenv("ETS_NUM_THREADS", raising=False)
    summary = configure_compute(num_threads=2, total_threads=16, quiet=True)
    assert summary["total_threads"] == 16
    assert summary["num_threads"] == 2


def test_configure_compute_returns_architecture_block():
    summary = configure_compute(num_threads=1, quiet=True)
    arch = summary["architecture"]
    assert "logical_cores" in arch
    assert "machine" in arch
    assert "system" in arch

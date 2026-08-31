#!/usr/bin/env python3
"""Author regression for the M672 path-identity repair only."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "map_m672_decoder_convtranspose_polyphase_workload_r3.py")
SPEC = importlib.util.spec_from_file_location("m672_mapper_r3", SCRIPT)
M672 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M672)


def write_one_bit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([int(value) & 1]))


def test_relative_name_consumes_the_validated_package_file(tmp_path,
                                                           monkeypatch):
    package = tmp_path / "package"
    work = tmp_path / "work"
    package.mkdir()
    work.mkdir()
    write_one_bit(package / "x.bitpack", 0)
    write_one_bit(work / "x.bitpack", 1)
    monkeypatch.chdir(work)

    matrices, _metadata = M672.materialize_polyphase(
        "x.bitpack", [1, 1, 1, 1, 1], trusted_root=package)
    assert sum(int(value.sum()) for value in matrices.values()) == 0
    accounting = M672.workload_accounting(
        "x.bitpack", [1, 1, 1, 1, 1], trusted_root=package)
    assert accounting["source_popcount"] == 0


def test_trusted_root_rejects_symlink_in_any_ancestor(tmp_path):
    real = tmp_path / "real"
    package = real / "package"
    package.mkdir(parents=True)
    write_one_bit(package / "x.bitpack", 0)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink|identities differ"):
        M672.validate_bitpack(
            link / "package/x.bitpack", [1, 1, 1, 1, 1],
            trusted_root=link / "package")


@pytest.mark.parametrize("root", ["relative", "/", "/tmp//x", "/tmp/./x",
                                  "/tmp/../x"])
def test_trusted_root_lexical_forms_are_fail_closed(root):
    with pytest.raises(RuntimeError):
        M672.trusted_root_all_components(root)


def test_absolute_validated_file_still_reconstructs_exactly(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    shape = [2, 1, 3, 2, 3]
    rng = np.random.default_rng(672)
    activation = rng.integers(0, 2, size=shape, dtype=np.uint8)
    packed = np.packbits(activation.reshape(-1), bitorder="little")
    payload = package / "activation.bitpack"
    payload.write_bytes(packed.tobytes())
    weight = rng.integers(-3, 4, size=(3, 2, 3, 3), dtype=np.int16)

    observed = M672.reconstruct_convtranspose(
        payload, shape, weight, trusted_root=package)
    expected = M672.R2.reconstruct_convtranspose(
        payload.resolve(), shape, weight, trusted_root=package.resolve())
    np.testing.assert_array_equal(observed, expected)


def test_m670_frozen_source_is_not_modified():
    assert M672.sha256(M672.R2.Path(M672.R2.__file__)) == (
        "875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b")

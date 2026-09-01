#!/usr/bin/env python3
"""M1598 different-author rehammer of the M1582/M1574 permit boundary.

This program issues and consumes only preload permits in a temporary directory.
It never constructs a producer, opens a capture payload, imports torch, reaches a
remote host, or invokes GPU/RTL/EDA work.  It deliberately treats same-process
Python reflection as adversarial because M1576 already admitted object.__new__
as an attack surface.
"""
from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
AUTHOR_TEST = HW / "tests/test_m1558_motion_ep34_s2_tsbg_reduced_binary_source.py"
M1574_CONTRACT = HW / "contracts/m1574_m1565_reduced_binary_permit_provenance_successor_source_contract_r1_20260901.json"
M1574_AUTHOR = HW / "reviews/m1574_m1565_reduced_binary_permit_provenance_successor_author_receipt_r1_20260901"
M1576_REVIEW = HW / "reviews/m1576_m1574_permit_provenance_independent_rehammer_r1_20260901"
M1582_CONTRACT = HW / "contracts/m1582_m1576_minted_instance_registry_successor_source_contract_r1_20260901.json"
M1582_AUTHOR = HW / "reviews/m1582_m1576_minted_instance_registry_successor_author_receipt_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089",
    AUTHOR_TEST: "d7ea365d2a4b0d26286a93d6061f91ca136dad19fd6613a558cd9f5f489e93d3",
    M1574_CONTRACT: "c86f8d656824aff89a5767c83b3fe7e9468fa7f2338a9053a9985f03a9d06a52",
    M1574_AUTHOR / "review.json": "caa944692d31067a7049209c2bc0bfc34e84daefd8e268b4973e42892774733c",
    M1574_AUTHOR / "SHA256SUMS": "5e5c85384e99ffb700399e60aca0395e128912676405100dc74a0c6e94815b6c",
    M1574_AUTHOR / "SHA256SUMS.seal.sha256": "73c851c6ff368e7841a93a335d589e6842ab35c7661ab3cda9cc2ba9ac87e334",
    M1576_REVIEW / "review.json": "43233ab3e583e6261c5abef40ce4c95ab8f85ab4be19d3248daf6bdf697f8cd7",
    M1576_REVIEW / "SHA256SUMS": "d354166de50388504c9086cd318f0246e897aab09b67bf46e62d6c1c89a6d816",
    M1576_REVIEW / "SHA256SUMS.seal.sha256": "167f10f972d82cdfadec783dfc9d0892062ced68e14f9877f2bb173bf52db20d",
    M1582_CONTRACT: "323f4585a914c68fda715b329fec412befcc2186a55e784ea4ebb55365fcdde8",
    Path(str(M1582_CONTRACT) + ".sha256"): "6b014f2fdce288edd250b2e550b09ebb4a8aabb4456e45056363702de990ae3e",
    Path(str(M1582_CONTRACT) + ".sha256.seal.sha256"): "846f4a8305b3815f6c3cbd83d32f29ffdb3455a4ecd4aa90e4bf0d24c78b282e",
    M1582_AUTHOR / "review.json": "44f2c1d8f088460d735481ec12e6e0e887b10aaf93fa82260c3daa71cb3c5b32",
    M1582_AUTHOR / "SHA256SUMS": "64651cd2b07dc80b39515ed3f97774c2d033ab408ff9f2ddd0b05b80625e01c2",
    M1582_AUTHOR / "SHA256SUMS.seal.sha256": "4e462c01498f14d914bba906fb0407e3e9b2a5efa008698e8a72a065f5ee21be",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def regular_exact(path, expected):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            digest(path) == expected, "identity drift: " + str(path))


def verify_sealed_tree(directory, manifest_digest, outer_file_digest):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular_exact(manifest, manifest_digest)
    regular_exact(outer, outer_file_digest)
    require(outer.read_text(encoding="ascii").split() ==
            [manifest_digest, "SHA256SUMS"], "outer seal content drift")
    names = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        member_digest, name = line.split(None, 1)
        name = name.strip()
        rel = Path(name)
        require(name == rel.as_posix() and not rel.is_absolute() and
                ".." not in rel.parts and name not in names,
                "manifest row invalid")
        names.add(name)
        regular_exact(directory / rel, member_digest)


for _path, _expected in EXPECTED.items():
    regular_exact(_path, _expected)
verify_sealed_tree(M1574_AUTHOR, EXPECTED[M1574_AUTHOR / "SHA256SUMS"],
                   EXPECTED[M1574_AUTHOR / "SHA256SUMS.seal.sha256"])
verify_sealed_tree(M1576_REVIEW, EXPECTED[M1576_REVIEW / "SHA256SUMS"],
                   EXPECTED[M1576_REVIEW / "SHA256SUMS.seal.sha256"])
verify_sealed_tree(M1582_AUTHOR, EXPECTED[M1582_AUTHOR / "SHA256SUMS"],
                   EXPECTED[M1582_AUTHOR / "SHA256SUMS.seal.sha256"])

MODULES_BEFORE_SOURCE = set(sys.modules)
SPEC = importlib.util.spec_from_file_location("m1598_exact_m1582", str(SOURCE))
require(SPEC is not None and SPEC.loader is not None, "cannot import source")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
MODULES_FROM_SOURCE = set(sys.modules) - MODULES_BEFORE_SOURCE


def closure_cells(function):
    closure = function.__closure__ or ()
    return dict(zip(function.__code__.co_freevars,
                    [cell.cell_contents for cell in closure]))


def mini_specs():
    return [{
        "layer_id": 0, "target": "FC1", "module_name": "m1598.fc1",
        "operator": "Linear", "operator_order": 0,
        "input_shape": (1, 1, 1, 1, 1),
        "output_shape": (1, 1, 1, 1, 1),
        "channel_axis": 4, "input_channels": 1, "output_channels": 1,
        "tokens_per_call": 1, "tokens_s40": 1,
        "input_elements_s40": 1, "input_active_s40": 0,
    }]


def must_reject(function):
    try:
        function()
    except (M.M1558Error, AssertionError, AttributeError, KeyError,
            pickle.PickleError, RuntimeError, TypeError, ValueError):
        return True
    raise AssertionError("attack unexpectedly accepted")


def forged_instance(cls):
    return object.__new__(cls)


def copy_attacks(permit, cls, output, inventory):
    copied = copy.copy(permit)
    deepcopied = copy.deepcopy(permit)
    require(type(copied) is cls and type(deepcopied) is cls,
            "copy did not retain exact type")
    require(must_reject(lambda: copied.consume(output, inventory)),
            "copy accepted")
    require(must_reject(lambda: deepcopied.consume(output, inventory)),
            "deepcopy accepted")
    require(must_reject(lambda: pickle.dumps(permit,
                                              protocol=pickle.HIGHEST_PROTOCOL)),
            "pickle accepted local permit type")


def class_equality_forgery(cls, victim, output, inventory):
    """Exploit mutable class equality/hash against dict keyed by object."""
    old_hash_value = hash(victim)
    previous = {}
    for name in ("__hash__", "__eq__"):
        previous[name] = (name in cls.__dict__, cls.__dict__.get(name))
    cls.__hash__ = lambda _self: old_hash_value
    cls.__eq__ = lambda _self, _other: True
    try:
        forged = forged_instance(cls)
        return forged.consume(output, inventory)
    finally:
        for name in ("__eq__", "__hash__"):
            existed, value = previous[name]
            if existed:
                setattr(cls, name, value)
            else:
                delattr(cls, name)


def direct_registry_forgery(cls, registry, output, inventory, estimate,
                            free_bytes):
    """Demonstrate that a Python closure cell is introspectable by caller code."""
    forged = forged_instance(cls)
    registry[forged] = (str(Path(output).resolve()), str(inventory),
                        dict(estimate), int(free_bytes))
    return forged.consume(output, inventory)


def run(output):
    contract = json.loads(M1582_CONTRACT.read_text(encoding="utf-8"))
    author = json.loads((M1582_AUTHOR / "review.json").read_text(encoding="utf-8"))
    require(contract["status"] ==
            "SUCCESSOR_SOURCE_ONLY__CLOSURE_MINTED_INSTANCE_REGISTRIES__INDEPENDENT_REHAMMER_REQUIRED__NO_REMOTE_NO_CAPTURE",
            "M1582 contract status drift")
    require(author["status"] ==
            "PASS_AUTHOR_DUAL_RUNTIME_MINTED_REGISTRY_REGRESSION__INDEPENDENT_REHAMMER_REQUIRED__NO_REMOTE_NO_CAPTURE",
            "M1582 author status drift")

    production_issue = closure_cells(M._issue_production_permit)
    synthetic_issue = closure_cells(M._issue_synthetic_permit)
    production_consume = closure_cells(M._ProductionPreloadPermit.consume)
    synthetic_consume = closure_cells(M._SyntheticPreloadPermit.consume)
    production_registry = production_issue["production_minted"]
    synthetic_registry = synthetic_issue["synthetic_minted"]

    require(production_registry is production_consume["production_minted"],
            "production issue/consume registry identity mismatch")
    require(synthetic_registry is synthetic_consume["synthetic_minted"],
            "synthetic issue/consume registry identity mismatch")
    require(production_registry is not synthetic_registry and
            len(production_registry) == 0 and len(synthetic_registry) == 0,
            "registries not separate and initially empty")

    source_text = SOURCE.read_text(encoding="utf-8")
    require(not any(name == "torch" or name.startswith("torch.")
                    for name in MODULES_FROM_SOURCE), "source import loaded torch")
    require(not any(name in MODULES_FROM_SOURCE for name in
                    ("paramiko", "requests", "socket", "subprocess")),
            "source import loaded remote/process module")
    require("import torch" not in source_text and "import subprocess" not in
            source_text and "import socket" not in source_text,
            "source gained GPU/remote/process import")
    description = M.describe()
    require(description["execution"] == {
        "gpu": False, "ssh": False, "capture": False,
        "release": False, "automatic_retry": False},
        "source execution boundary drift")
    require(must_reject(M.production_release), "production release callable")

    checks = []
    fatal = []
    real_queries = []
    real_disk_usage = shutil.disk_usage

    def record(name, value=True):
        require(value, name)
        checks.append(name)

    def spying_disk_usage(path):
        value = real_disk_usage(path)
        real_queries.append({"path": str(Path(path).resolve()),
                             "free": int(value.free)})
        return value

    with tempfile.TemporaryDirectory(prefix="m1598.",
                                     dir=str(Path(output).parent)) as directory:
        base = Path(directory)
        specs = mini_specs()
        inventory = M.canonical_sha(specs)
        estimate = M.estimate_from_specs(specs, 1)
        synthetic_free = (estimate["result_upper_bytes"] +
                          M.MIN_FREE_AFTER_BYTES + 1)
        production_specs = M.frozen_layer_specs()
        production_inventory = M.canonical_sha(production_specs)
        production_estimate = M.estimate_from_specs(production_specs, 40)

        record("closure_owned_production_registry")
        record("closure_owned_synthetic_registry")
        record("production_synthetic_registries_distinct")
        record("source_import_no_torch_gpu_remote_process")
        record("production_release_rejected")

        prod_forged = forged_instance(M._ProductionPreloadPermit)
        syn_forged = forged_instance(M._SyntheticPreloadPermit)
        record("plain_object_new_production_rejected", must_reject(
            lambda: prod_forged.consume(base / "plain_prod", production_inventory)))
        record("plain_object_new_synthetic_rejected", must_reject(
            lambda: syn_forged.consume(base / "plain_syn", inventory)))

        M.shutil.disk_usage = spying_disk_usage
        try:
            prod_copy_root = base / "prod_copy"
            prod_copy = M.issue_preload_permit(prod_copy_root)
        finally:
            M.shutil.disk_usage = real_disk_usage
        copy_attacks(prod_copy, M._ProductionPreloadPermit, prod_copy_root,
                     production_inventory)
        record("production_copy_rejected")
        record("production_deepcopy_rejected")
        record("production_pickle_rejected")
        prod_receipt = prod_copy.consume(prod_copy_root, production_inventory)
        record("production_original_after_copy_attacks_consumes")
        record("production_double_consume_rejected", must_reject(
            lambda: prod_copy.consume(prod_copy_root, production_inventory)))

        syn_copy_root = base / "syn_copy"
        syn_copy = M.issue_synthetic_permit(
            syn_copy_root, specs, 1, synthetic_free)
        copy_attacks(syn_copy, M._SyntheticPreloadPermit, syn_copy_root,
                     inventory)
        record("synthetic_copy_rejected")
        record("synthetic_deepcopy_rejected")
        record("synthetic_pickle_rejected")
        syn_receipt = syn_copy.consume(syn_copy_root, inventory)
        record("synthetic_original_after_copy_attacks_consumes")
        record("synthetic_double_consume_rejected", must_reject(
            lambda: syn_copy.consume(syn_copy_root, inventory)))

        M.shutil.disk_usage = spying_disk_usage
        try:
            prod_exception_root = base / "prod_exception"
            prod_exception = M.issue_preload_permit(prod_exception_root)
        finally:
            M.shutil.disk_usage = real_disk_usage
        before = len(production_registry)
        record("production_mismatch_raises", must_reject(
            lambda: prod_exception.consume(base / "wrong_prod",
                                           production_inventory)))
        record("production_exception_consumes_registry",
               len(production_registry) == before - 1)
        record("production_exception_retry_rejected", must_reject(
            lambda: prod_exception.consume(prod_exception_root,
                                           production_inventory)))

        syn_exception_root = base / "syn_exception"
        syn_exception = M.issue_synthetic_permit(
            syn_exception_root, specs, 1, synthetic_free)
        before = len(synthetic_registry)
        record("synthetic_mismatch_raises", must_reject(
            lambda: syn_exception.consume(base / "wrong_syn", inventory)))
        record("synthetic_exception_consumes_registry",
               len(synthetic_registry) == before - 1)
        record("synthetic_exception_retry_rejected", must_reject(
            lambda: syn_exception.consume(syn_exception_root, inventory)))

        occupied_root = base / "occupied_after_issue"
        occupied = M.issue_synthetic_permit(
            occupied_root, specs, 1, synthetic_free)
        occupied_root.mkdir()
        record("namespace_exception_raises", must_reject(
            lambda: occupied.consume(occupied_root, inventory)))
        occupied_root.rmdir()
        record("namespace_exception_retry_rejected", must_reject(
            lambda: occupied.consume(occupied_root, inventory)))

        # Fatal finding 1: both public exact classes are mutable.  Replacing
        # __hash__/__eq__ lets an object.__new__ forgery alias the victim dict
        # key and pop the legitimately minted record.
        M.shutil.disk_usage = spying_disk_usage
        try:
            prod_class_root = base / "prod_class_alias"
            prod_class_victim = M.issue_preload_permit(prod_class_root)
        finally:
            M.shutil.disk_usage = real_disk_usage
        prod_class_receipt = class_equality_forgery(
            M._ProductionPreloadPermit, prod_class_victim, prod_class_root,
            production_inventory)
        fatal.append("mutable_production_class_hash_eq_aliases_minted_key")

        syn_class_root = base / "syn_class_alias"
        syn_class_victim = M.issue_synthetic_permit(
            syn_class_root, specs, 1, synthetic_free)
        syn_class_receipt = class_equality_forgery(
            M._SyntheticPreloadPermit, syn_class_victim, syn_class_root,
            inventory)
        fatal.append("mutable_synthetic_class_hash_eq_aliases_minted_key")

        # Fatal finding 2: the registries are closure-owned but Python closure
        # cells are reflective state.  A caller can obtain the dict and insert
        # an unminted exact object directly, bypassing both issuers.
        prod_reflect_root = base / "prod_closure_reflect"
        prod_reflect_receipt = direct_registry_forgery(
            M._ProductionPreloadPermit, production_registry,
            prod_reflect_root, production_inventory, production_estimate,
            1 << 60)
        fatal.append("production_closure_registry_reflectively_mutable")

        syn_reflect_root = base / "syn_closure_reflect"
        syn_reflect_receipt = direct_registry_forgery(
            M._SyntheticPreloadPermit, synthetic_registry,
            syn_reflect_root, inventory, estimate, synthetic_free)
        fatal.append("synthetic_closure_registry_reflectively_mutable")

        require(len(production_registry) == 0 and len(synthetic_registry) == 0,
                "hammer left registry entries")
        record("registries_empty_after_hammer")
        record("production_disk_receipt_exact",
               prod_receipt["free_bytes_before"] == real_queries[0]["free"])
        record("synthetic_receipt_typed",
               syn_receipt["provenance"] == M.SYNTHETIC_PROVENANCE)

    result = {
        "schema": "m1598_m1582_m1574_tsbg_capture_permit_independent_rehammer_runtime_r1_v1",
        "status": "NO_GO_M1598_M1582_CLOSURE_REGISTRY_NOT_A_PYTHON_SECURITY_BOUNDARY__SUCCESSOR_FIX_ONLY__NO_CAPTURE",
        "runtime": {"implementation": sys.implementation.name,
                    "version": ".".join(str(value) for value in
                                        sys.version_info[:3])},
        "identity": {
            "source_sha256": EXPECTED[SOURCE],
            "test_sha256": EXPECTED[AUTHOR_TEST],
            "m1574_contract_sha256": EXPECTED[M1574_CONTRACT],
            "m1576_review_sha256": EXPECTED[M1576_REVIEW / "review.json"],
            "m1582_contract_sha256": EXPECTED[M1582_CONTRACT],
            "m1582_author_review_sha256": EXPECTED[M1582_AUTHOR / "review.json"],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "requested_attacks": {
            "passed_count": len(checks), "passed": checks,
            "plain_object_new_rejected": True,
            "copy_rejected": True, "deepcopy_rejected": True,
            "pickle_rejected": True, "double_consume_rejected": True,
            "exception_consumes_permit": True,
        },
        "fatal_findings": {
            "count": len(fatal), "names": fatal,
            "production_class_alias_receipt_provenance":
                prod_class_receipt["provenance"],
            "synthetic_class_alias_receipt_provenance":
                syn_class_receipt["provenance"],
            "production_reflective_registry_receipt_free_bytes":
                prod_reflect_receipt["free_bytes_before"],
            "synthetic_reflective_registry_receipt_provenance":
                syn_reflect_receipt["provenance"],
        },
        "real_disk": {"query_count": len(real_queries),
                      "all_paths": [row["path"] for row in real_queries]},
        "side_effects": {
            "permit_only_temporary_namespace": True,
            "producer_constructed": False, "payload_opened": False,
            "checkpoint_loaded": False, "torch_imported": False,
            "gpu": False, "ssh": False, "remote": False,
            "capture": False, "release": False,
            "tsbg_dse": False, "rtl": False, "eda": False,
        },
        "authorization": {
            "one_m1558_binary_capture": False,
            "successor_source_fix_only": True,
            "remote_wrapper": False, "gpu": False, "capture": False,
            "tsbg_dse": False, "rtl": False, "eda": False,
        },
    }
    output = Path(output)
    require(not output.exists(), "output already exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"] + " requested_checks={} fatal={}".format(
        len(checks), len(fatal)))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())

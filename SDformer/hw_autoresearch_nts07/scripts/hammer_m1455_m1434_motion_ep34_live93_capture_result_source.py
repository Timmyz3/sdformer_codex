#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only, fail-closed result hammer for the M1434 ep34 live-93 capture.

This source never launches capture work.  It validates a pre-existing,
recursively sealed result and keeps every hardware/performance claim false.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence


SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1401_SOURCE = HW / "scripts/hammer_m1401_m1349_motion_ep34_live105_capture_result_source.py"
M1401_SHA256 = "f55642429fe097fdb5c5fd860592d4b04652fc47c85526eb756dc005125e8a22"
M1434_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
M1434_SHA256 = "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed"
M1434_TEST = HW / "tests/test_m1434_motion_ep34_live93_runtime_successor.py"
M1434_TEST_SHA256 = "b05e122be8d3fb61b001be648ff8980a7341c2a19d29c401a9dc62ff5bafb8c2"
M1434_CONTRACT = HW / (
    "contracts/m1434_motion_ep34_live93_runtime_successor_source_"
    "contract_r1_20260831.json")
M1434_CONTRACT_SHA256 = "5e92af7c080f417fd94f190ce90c064a19fd70c02cfbd8fb6a2ad03d6f12e75e"
M1436_SOURCE = HW / (
    "system_simulator/scripts/build_m1436_ep34_decoder_capture_adapter_exact_graph.py")
M1436_SHA256 = "729279ca8a3c531e60363a5f1e5225b6fa84018931140af7b32767a862871eb6"
M1436_TEST = HW / (
    "system_simulator/tests/test_m1436_ep34_decoder_capture_adapter_exact_graph.py")
M1436_TEST_SHA256 = "35807cfb0f8ce3d4999a524530db830bf125e18e370b639a0213a222dc3110e7"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANONICAL_RESULT = HW / (
    "results/m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831")
CONTRACT = HW / (
    "contracts/m1455_m1434_motion_ep34_live93_capture_result_hammer_source_"
    "contract_r1_20260831.json")
TEST = HW / "tests/test_hammer_m1455_m1434_motion_ep34_live93_capture_result_source.py"
SOURCE_SCHEMA = "m1455_m1434_motion_ep34_live93_capture_result_hammer_source_r1_v1"
PASS_TOKEN = "PASS_M1455_SOURCE_SELF_CHECK__NO_CAPTURE_NO_REMOTE_NO_GPU"


class M1455Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1455Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1455Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1401 = load_exact("m1455_sealed_m1401", M1401_SOURCE, M1401_SHA256)
M1434 = load_exact("m1455_sealed_m1434", M1434_SOURCE, M1434_SHA256)
M1436 = load_exact("m1455_sealed_m1436", M1436_SOURCE, M1436_SHA256)
BASE = M1401.BASE


def strict_json(path: Path) -> dict[str, Any]:
    value = BASE.strict_file(path)
    require(type(value) is dict, "JSON root is not object")
    return value


def canonical_directory(root: Path) -> None:
    require(os.path.lexists(str(root)), "canonical result absent")
    mode = root.lstat().st_mode
    require(stat.S_ISDIR(mode) and not root.is_symlink(),
            "canonical result must be real directory")


def live_inventory() -> dict[str, list[str]]:
    policy = M1434.R1.strict_json(M1434.R1.SOURCE_CONTRACT)
    inventory = M1434.R1.frozen_non_atlif_inventory(policy)
    inventory["atlif"] = list(M1434.M1349.EXPECTED_ATLIF_NAMES)
    return M1434.expected_live93_inventory(inventory)


def validate_ordered(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        graph = M1436.validate_complete_ordered_graph(root)
    except Exception as error:
        raise M1455Error("complete ordered graph failed") from error
    path = root / "unified_ordered_records.jsonl"
    ordered = [BASE.strict_text(line) for line in
               path.read_text(encoding="utf-8").splitlines()]
    require(all(type(row) is dict for row in ordered), "ordered row is not object")
    inventory = live_inventory()
    try:
        audit = M1434.audit_with_live93(ordered, inventory, list(range(40)))
    except Exception as error:
        raise M1455Error("live93 ordered call matrix failed") from error
    require(audit.get("status") == "PASS" and audit.get("records") == 9880,
            "live93 ordered call matrix drift")
    reference = [(row.get("category"), row.get("name")) for row in ordered[:247]]
    for sample in range(40):
        rows = ordered[sample * 247:(sample + 1) * 247]
        require(len(rows) == 247 and all(
            type(row.get("global_sample_id")) is int and
            row["global_sample_id"] == sample for row in rows),
            "per-sample ordered slice drift")
        require([(row.get("category"), row.get("name")) for row in rows] == reference,
                "ordered module sequence differs across samples")
    return ordered, {"graph": graph, "call_matrix": audit,
                     "all_sample_sequences_equal": True}


def validate_manifest(manifest: dict[str, Any]) -> None:
    require(manifest.get("schema") ==
            "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1" and
            manifest.get("status") ==
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "M1434 manifest schema/status drift")
    identity = manifest.get("identity")
    require(type(identity) is dict, "manifest identity missing")
    require(identity.get("checkpoint_load_audit") ==
            {"missing_count": 0, "unexpected_count": 0},
            "checkpoint load audit drift")
    require(identity.get("module_counts") ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "static module count drift")
    selected = identity.get("selection", {}).get("selected", {})
    require(selected.get("candidate_id") == "resume_ep34" and
            type(selected.get("epoch")) is int and selected["epoch"] == 34 and
            selected.get("checkpoint", {}).get("sha256") == M1434.CHECKPOINT_SHA256 and
            selected.get("configuration", {}).get("sha256") == M1434.CONFIG_SHA256 and
            selected.get("profile", {}).get("sha256") == M1401.PROFILE_SHA256 and
            selected.get("profile", {}).get("samples") == 825,
            "selected ep34 identity drift")
    runtime = manifest.get("m1434_runtime_contract")
    dead = {"kind": "H60_STATIC_BUT_RUNTIME_BYPASSED", "count": 12,
            "names": list(M1434.DEAD_SN2_Q),
            "terminal_lf_sha256": M1434.DEAD_SN2_Q_SHA256}
    require(type(runtime) is dict and
            runtime.get("static_modules") == 259 and
            runtime.get("static_atlif") == 105 and
            runtime.get("static_atlif_terminal_lf_sha256") == M1434.STATIC_ATLIF_SHA256 and
            runtime.get("live_modules_per_sample") == 247 and
            runtime.get("live_atlif") == 93 and
            runtime.get("live_atlif_terminal_lf_sha256") == M1434.LIVE_ATLIF_SHA256 and
            runtime.get("dead_atlif") == dead and
            runtime.get("dead_calls_per_sample") == 0 and
            runtime.get("ordered_records") == 9880 and
            runtime.get("attention_records") == 480 and
            runtime.get("payload_files") == 640,
            "M1434 runtime contract drift")
    require(manifest.get("forensic_snapshots") == {
        "samples": 40, "atomic_per_sample": True,
        "failure_forensic_only": True, "automatic_canonical_promotion": False},
        "forensic snapshot policy drift")
    require(manifest.get("claim_boundary") == {
        "capture_only": True, "accuracy": False, "cycles": False,
        "speedup": False, "system_speedup": False, "energy": False,
        "rtl": False, "ppa": False, "fresh_result_hammer_required": True},
        "manifest claim boundary drift")


def validate_admission(admission: dict[str, Any]) -> None:
    dead = {"kind": "H60_STATIC_BUT_RUNTIME_BYPASSED", "count": 12,
            "names": list(M1434.DEAD_SN2_Q),
            "terminal_lf_sha256": M1434.DEAD_SN2_Q_SHA256}
    require(admission == {
        "schema": "m1434_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": 9880, "attention": 480, "payload_files": 640,
        "execution": 7360, "operator_rows": 79,
        "atlif_live_rows": 93, "atlif_static": 105,
        "static_atlif_terminal_lf_sha256": M1434.STATIC_ATLIF_SHA256,
        "live_atlif_terminal_lf_sha256": M1434.LIVE_ATLIF_SHA256,
        "dead_atlif": dead,
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False,
                           "energy": False, "ppa": False}},
        "M1434 admission drift")


def validate_result(root: Path = CANONICAL_RESULT) -> dict[str, Any]:
    canonical_directory(root)
    rows, seal = BASE.verify_recursive_seal(root)
    required = {"manifest.json", "m1434_admission.json",
                "unified_ordered_records.jsonl", "attention_qk/manifest.json",
                "execution_trace.json", "operator_runtime.json",
                "atlif_activity.json", "RUN_COMPLETE.txt"}
    require(required <= set(rows), "required sealed members missing")
    manifest = strict_json(root / "manifest.json")
    validate_manifest(manifest)
    validate_admission(strict_json(root / "m1434_admission.json"))
    expected = BASE.OLD.expected_cohort()
    observed = manifest.get("cohort", {}).get("samples")
    require(type(observed) is list and len(observed) == 40 and
            [{key: row[key] for key in expected[0]} for row in observed] == expected,
            "cohort identity/order drift")
    ordered, ordered_audit = validate_ordered(root)
    try:
        retained = M1401.M1338.validate_retained_payloads(root, rows, ordered)
        attention = M1401.M1338.OLD.validate_attention_geometry(root, rows)
        M1401.M1338.validate_attention_exact_archive(root)
        payloads = M1434.R1.validate_payload_population(root)
        M1434.validate_snapshot_population_live93(root)
    except Exception as error:
        raise M1455Error(
            "retained/attention/payload/forensic validation failed") from error
    require(len(payloads) == 640, "payload population is not 640")
    execution = BASE.strict_file(root / "execution_trace.json")
    operators = BASE.strict_file(root / "operator_runtime.json")
    atlif = BASE.strict_file(root / "atlif_activity.json")
    require(type(execution) is list and len(execution) == 7360,
            "execution population drift")
    require(type(operators) is list and len(operators) == 79 and
            len({row.get("name") for row in operators}) == 79 and
            all(type(row.get("calls")) is int and row["calls"] == 40
                for row in operators), "operator runtime drift")
    names = [row.get("name") for row in atlif] if type(atlif) is list else []
    require(len(names) == 93 and len(set(names)) == 93 and
            all(type(row.get("calls")) is int and row["calls"] == 40
                for row in atlif) and
            M1434.terminal_lf_digest(sorted(names)) == M1434.LIVE_ATLIF_SHA256 and
            not set(names) & set(M1434.DEAD_SN2_Q), "ATLIF live93 identity drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n",
            "completion token drift")
    return {"status": "PASS_M1455_M1434_EP34_LIVE93_CAPTURE_RESULT",
            "seal": seal,
            "population": {"ordered": ordered_audit["graph"]["rows"],
                           "retained": retained, "attention": attention,
                           "payload": len(payloads), "execution": len(execution),
                           "operator": len(operators), "atlif": len(atlif),
                           "forensic_snapshots": 40},
            "identity": {"checkpoint_sha256": M1434.CHECKPOINT_SHA256,
                         "config_sha256": M1434.CONFIG_SHA256,
                         "profile_sha256": M1401.PROFILE_SHA256},
            "claim_boundary": {"capture_only": True, "paper_result": False,
                               "cycles": False, "speedup": False,
                               "energy": False, "ppa": False}}


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1434_TEST, M1434_TEST_SHA256, "M1434 test")
    regular_exact(M1434_CONTRACT, M1434_CONTRACT_SHA256, "M1434 contract")
    regular_exact(M1436_TEST, M1436_TEST_SHA256, "M1436 test")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    M1434.verify_predecessors()
    M1434.validate_source_policy()
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") ==
            "SOURCE_ONLY__M1434_LIVE93_RESULT_HAMMER__NO_CAPTURE_NO_REMOTE",
            "source policy schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "source/test identity drift")
    require(policy.get("canonical_result") == str(CANONICAL_RESULT.relative_to(ROOT)) and
            policy.get("production_authorized") is False and
            policy.get("actual_result_seal_prefilled") is False and
            policy.get("claim_boundary") == {
                "source_only": True, "capture": False, "remote": False,
                "gpu": False, "controller_signal": False,
                "paper_result": False, "cycles": False, "speedup": False,
                "energy": False, "ppa": False, "system_speedup": False,
                "headline": False}, "source-only boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--validate-canonical-result", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        require(not os.path.lexists(str(CANONICAL_RESULT)),
                "source self-check refuses present canonical result")
        print(PASS_TOKEN)
        return 0
    print(json.dumps(validate_result(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1455Error as error:
        print("M1455_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

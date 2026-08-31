#!/opt/conda/envs/sdformerflow/bin/python
"""Additive ep34 capture successor for the measured H60 live-105 inventory.

M1329 failed after exact checkpoint restore because M1227 assumed twelve
``sn_v`` ATLIF modules existed but were dead.  The final Motion-C12 H60 graph
contains exactly 105 ATLIF modules and no ``sn_v`` module at all; consequently
all 105 are live and the per-sample unified inventory is 259, not 247.

This source preserves the sealed M1327 identity/cohort and patches only the
three inventory-dependent validators while delegating capture to the same
substrate.  It owns no attempt consumer or production CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1327_SOURCE = Path(__file__).with_name(
    "capture_m1327_motion_ep34_consumed_namespace_bridge_r1.py")
M1327_SOURCE_SHA256 = "2ab5024a11a81f7bb3ed75956114cc95e07dbe0782328414f2bd3c79342c3ac9"
M1329_RUNNER = HW / "scripts/run_m1329_m1327_motion_ep34_production_one_shot.py"
M1329_RUNNER_SHA256 = "14cec4e57a1818c332811083a142826c41a17e2f3bb5e95404f572a8840d3778"
FAILED_ROOT = HW / "results/m1343_m1329_failed_atlif_inventory_forensic_r1_20260831"
FAILED_ATTEMPT = FAILED_ROOT / "attempt_consumed"
FAILED_ATTEMPT_SHA256 = "0b673e6d8f61065e1920d6345277f1d71fea3ee20158c7601f96a588268cfd48"
FAILED_TEMP_LOG = FAILED_ROOT / "temp.log"
FAILED_TEMP_LOG_SHA256 = "ad94b85c5938ecae87065a95bc110ce53c5a99fb5c37a1dc500be8f69ae3dd0b"
OLD_ATTEMPT_TOKEN = b"M1329_M1327_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"

SOURCE_CONTRACT = HW / "contracts/m1343_motion_ep34_live105_inventory_successor_source_contract_r1_20260831.json"
TEST = HW / "tests/test_m1343_motion_ep34_live105_inventory_successor.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CANONICAL_RESULT = HW / "results/m1343_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831"
CANONICAL_ATTEMPT = HW / "results/.m1343_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831.attempt_consumed"
CANONICAL_LOG = HW / "results/.m1343_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831.production.log"

EXPECTED_ATLIF_COUNT = 105
EXPECTED_ATLIF_NAMES_SHA256 = "ca7dab07f7437608c9224d551f6315d2a7e97d30ed6d224db86ce69ebeb40265"
EXPECTED_LIVE_MODULES = 259
EXPECTED_ORDERED_RECORDS = 10360
EXPECTED_ATLIF_ROWS = 105
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
SOURCE_SCHEMA = "m1343_motion_ep34_live105_inventory_successor_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1329_FAILURE_CLOSED__LIVE105_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
PASS_TOKEN = "PASS_M1343_LIVE105_SOURCE_SELF_CHECK__NO_ATTEMPT_NO_GPU_NO_CAPTURE"


class M1343Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1343Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise M1343Error("missing " + label) from exc
    require(stat.S_ISREG(observed.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1343Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _load_m1327():
    regular_exact(M1327_SOURCE, M1327_SOURCE_SHA256, "sealed M1327 source")
    spec = importlib.util.spec_from_file_location("m1343_sealed_m1327", M1327_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1327")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1327 = _load_m1327()
M1249 = M1327.M1249
R1 = M1249.M1243.P.R1


def inventory_digest(names) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def expected_live105_inventory(static_inventory: dict[str, list[str]]) -> dict[str, list[str]]:
    require(set(static_inventory) == set(R1.EXPECTED_STATIC_COUNTS),
            "static inventory category set drift")
    counts = {key: len(value) for key, value in static_inventory.items()}
    require(counts == R1.EXPECTED_STATIC_COUNTS, "static inventory count drift")
    atlif = list(static_inventory["atlif"])
    require(len(atlif) == EXPECTED_ATLIF_COUNT, "ATLIF count is not 105")
    require(inventory_digest(atlif) == EXPECTED_ATLIF_NAMES_SHA256,
            "final ep34 ATLIF name-set SHA drift")
    require(not any(".sn_v" in name for name in atlif),
            "H60 final graph unexpectedly contains sn_v")
    result = {key: sorted(value) for key, value in static_inventory.items()}
    require(sum(len(value) for value in result.values()) == EXPECTED_LIVE_MODULES,
            "live inventory is not 259")
    return result


def validate_snapshot_population_live105(staging: Path) -> None:
    root = Path(staging) / "forensic_samples"
    R1.directory(root, "forensic snapshot root")
    expected_files = {
        "unified_ordered_sample.jsonl", "execution_sample.json",
        "operator_runtime_cumulative.json", "atlif_activity_cumulative.json",
    }
    for sample in range(40):
        sample_root = root / "sample_{:02d}".format(sample)
        R1.directory(sample_root, "atomic sample snapshot")
        manifest = R1.strict_json(sample_root / "snapshot_manifest.json")
        require(manifest["sample_id"] == sample and
                manifest["call_audit"]["status"] == "PASS" and
                manifest["call_audit"]["records"] == EXPECTED_LIVE_MODULES and
                manifest["call_audit"]["live_modules_per_sample"] == EXPECTED_LIVE_MODULES,
                "sample forensic live105 audit mismatch")
        require(set(manifest["files"]) == expected_files, "snapshot file population mismatch")
        for name, digest in manifest["files"].items():
            R1.regular(sample_root / name, "snapshot member")
            require(R1.sha256(sample_root / name) == digest, "snapshot member SHA mismatch")
    require(sorted(path.name for path in root.iterdir() if path.is_dir()) ==
            ["sample_{:02d}".format(sample) for sample in range(40)],
            "snapshot directory population mismatch")


def final_validate_and_seal_live105(staging, writer_type, selected_identity) -> None:
    staging = Path(staging)
    ordered = [json.loads(line) for line in
               (staging / "unified_ordered_records.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = R1.audit_call_matrix(ordered, writer_type.ACTIVE_WRITER.live_inventory, range(40))
    require(audit["status"] == "PASS" and len(ordered) == EXPECTED_ORDERED_RECORDS,
            "final ordered population is not 40x259")
    attention = R1.strict_json(staging / "attention_qk/manifest.json")
    require(len(attention["records"]) == 480, "attention population is not 480")
    R1.validate_payload_population(staging)
    validate_snapshot_population_live105(staging)
    execution = json.loads((staging / "execution_trace.json").read_text(encoding="utf-8"))
    operators = json.loads((staging / "operator_runtime.json").read_text(encoding="utf-8"))
    atlif = json.loads((staging / "atlif_activity.json").read_text(encoding="utf-8"))
    require(len(execution) == 7360, "execution population must be 40x184")
    require(len(operators) == 79 and all(int(row["calls"]) == 40 for row in operators),
            "operator runtime must contain 79 rows at 40 calls")
    require(len(atlif) == EXPECTED_ATLIF_ROWS and
            all(int(row["calls"]) == 40 for row in atlif),
            "ATLIF runtime must contain 105 live rows at 40 calls")
    require(inventory_digest(row["name"] for row in atlif) == EXPECTED_ATLIF_NAMES_SHA256,
            "ATLIF runtime name-set SHA drift")
    manifest_path = staging / "manifest.json"
    manifest = R1.strict_json(manifest_path)
    manifest.update({
        "schema": "m1343_motion_ep34_live105_unified_hardware_capture_r1_v1",
        "status": "CAPTURE_COMPLETE__FRESH_M1343_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
        "m1343_runtime_contract": {
            "static_modules": 259, "static_atlif": 105,
            "live_modules_per_sample": 259, "live_atlif": 105,
            "dead_sn_v": [], "dead_calls_per_sample": 0,
            "atlif_names_sha256": EXPECTED_ATLIF_NAMES_SHA256,
            "ordered_records": EXPECTED_ORDERED_RECORDS,
            "attention_records": 480, "payload_files": 640,
            "final_selection_identity": selected_identity,
        },
        "forensic_snapshots": {"samples": 40, "atomic_per_sample": True,
                               "failure_forensic_only": True,
                               "automatic_canonical_promotion": False},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    admission = {
        "schema": "m1343_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": EXPECTED_ORDERED_RECORDS, "attention": 480,
        "payload_files": 640, "execution": 7360, "operator_rows": 79,
        "atlif_live_rows": 105, "atlif_static": 105, "dead_sn_v": [],
        "atlif_names_sha256": EXPECTED_ATLIF_NAMES_SHA256,
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False, "energy": False, "ppa": False},
    }
    (staging / "m1343_admission.json").write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    R1.write_double_seal(staging)


@contextlib.contextmanager
def patched_live105_capture_chain() -> Iterator[None]:
    originals = {
        "dead": R1.DEAD_SN_V,
        "counts": R1.EXPECTED_LIVE_COUNTS,
        "inventory": R1.expected_live_inventory,
        "snapshots": R1.validate_snapshot_population,
        "final": R1.final_validate_and_seal,
        "result": M1249.CANONICAL_RESULT,
    }
    require(originals["dead"] and len(originals["dead"]) == 12,
            "sealed predecessor dead inventory already modified")
    try:
        R1.DEAD_SN_V = tuple()
        R1.EXPECTED_LIVE_COUNTS = dict(R1.EXPECTED_STATIC_COUNTS)
        R1.expected_live_inventory = expected_live105_inventory
        R1.validate_snapshot_population = validate_snapshot_population_live105
        R1.final_validate_and_seal = final_validate_and_seal_live105
        M1249.CANONICAL_RESULT = CANONICAL_RESULT
        yield
    finally:
        R1.DEAD_SN_V = originals["dead"]
        R1.EXPECTED_LIVE_COUNTS = originals["counts"]
        R1.expected_live_inventory = originals["inventory"]
        R1.validate_snapshot_population = originals["snapshots"]
        R1.final_validate_and_seal = originals["final"]
        M1249.CANONICAL_RESULT = originals["result"]


def verify_m1329_failure() -> None:
    regular_exact(M1329_RUNNER, M1329_RUNNER_SHA256, "M1329 runner")
    regular_exact(FAILED_ATTEMPT, FAILED_ATTEMPT_SHA256, "M1329 failed attempt")
    regular_exact(FAILED_TEMP_LOG, FAILED_TEMP_LOG_SHA256, "M1329 failed temp log")
    require(FAILED_ATTEMPT.read_bytes() == OLD_ATTEMPT_TOKEN, "M1329 attempt token drift")
    log = FAILED_TEMP_LOG.read_text(encoding="utf-8")
    require("installed ATLIF modules: 105" in log and
            "missing=0, unexpected=0" in log and
            "checkpoint_epoch34.pth" in log and
            "batch_norm=no_running modules=78" in log,
            "M1329 pre-failure evidence mismatch")
    require(not os.path.lexists(str(M1327.CANONICAL_RESULT)) and
            not os.path.lexists(str(M1327.CANONICAL_LOG)),
            "failed M1329 unexpectedly published canonical result/log")


def build_runtime() -> tuple[dict[str, Any], dict[str, Any]]:
    old_runtime, binding = M1327.validate_identity_and_project()
    require(binding["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
            binding["identity"]["config_sha256"] == CONFIG_SHA256,
            "ep34 checkpoint/config identity drift")
    runtime = copy.deepcopy(old_runtime)
    runtime["contract_path"] = str(SOURCE_CONTRACT.relative_to(ROOT))
    runtime["output"] = {"path": str(CANONICAL_RESULT.relative_to(ROOT))}
    require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
            runtime["capture"] == {"attention_windows_per_call": 100},
            "M1343 runtime projection drift")
    return runtime, binding


def delegate_for_future_release(runtime: dict[str, Any], binding: dict[str, Any], substrate: Any) -> Path:
    expected, rebound = build_runtime()
    require(runtime == expected and binding == rebound, "under-lease M1343 binding drift")
    with patched_live105_capture_chain():
        output = M1249.run_capture(runtime, binding, substrate=substrate)
    require(Path(output) == CANONICAL_RESULT, "capture returned non-M1343 result")
    return Path(output)


def require_fresh_namespaces() -> None:
    require(len({CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG}) == 3,
            "M1343 namespaces are not distinct")
    require(all(not os.path.lexists(str(path)) for path in
                (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)),
            "M1343 namespace is not fresh")


def validate_source_policy() -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "M1343 source policy mismatch")
    require(policy.get("source") == {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                                      "sha256": sha256(Path(__file__).resolve())},
            "M1343 source identity mismatch")
    require(policy.get("test") == {"path": str(TEST.relative_to(ROOT)),
                                    "sha256": sha256(TEST)}, "M1343 test identity mismatch")
    require(policy.get("production_authorized") is False, "source policy cannot authorize production")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1343 is source-only")
    validate_source_policy()
    verify_m1329_failure()
    build_runtime()
    require_fresh_namespaces()
    require(R1.DEAD_SN_V and len(R1.DEAD_SN_V) == 12,
            "source self-check must leave predecessor unmodified")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

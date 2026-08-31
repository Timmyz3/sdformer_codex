#!/opt/conda/envs/sdformerflow/bin/python
"""Source-only ep34 successor for the measured H60 live-93 ATLIF graph.

M1400 failed closed after sample 0 because M1349 promoted a read-only static
inventory of 105 ATLIF modules to a live inventory.  The H60 forward bypasses
all twelve ``sn2_q`` modules: 105 ATLIF modules exist, but exactly 93 execute.
This additive source preserves the sealed static authority and capture
substrate while replacing only the runtime dead-set and population validators.

There is deliberately no production CLI, attempt consumer, GPU operation,
remote operation, controller signal, or automatic retry in this source.
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
SOURCE = Path(__file__).resolve()
TEST = HW / "tests/test_m1434_motion_ep34_live93_runtime_successor.py"
SOURCE_CONTRACT = HW / (
    "contracts/m1434_motion_ep34_live93_runtime_successor_source_"
    "contract_r1_20260831.json")
M1349_SOURCE = Path(__file__).with_name(
    "capture_m1349_motion_ep34_live105_inventory_successor_r2.py")
M1349_SOURCE_SHA256 = "3fe0f51acf489cf2f4d1a65f83f872b49a5fde79401a2fdb525768e681fbbbe5"
M1349_TEST = HW / "tests/test_m1349_motion_ep34_live105_inventory_successor.py"
M1349_TEST_SHA256 = "b20e06bcecb9fab1a326701e40e7bb72c5f13a3204a9d52470b58237a747492f"
M1349_CONTRACT = HW / (
    "contracts/m1349_motion_ep34_live105_inventory_successor_source_"
    "contract_r1_20260831.json")
M1349_CONTRACT_SHA256 = "ce2f373eef512237a0e0ee087134176384c30663bd52d42aa68c68b05fbd4712"
M1349_AUTHOR = HW / (
    "reviews/m1349_motion_ep34_live105_inventory_successor_source_author_"
    "r1_20260831")
M1349_AUTHOR_REVIEW_SHA256 = "bd29fae08da4978416477bcc5cb93a36d254cee2456a489452a8e5ad4ea98c57"
M1349_AUTHOR_MANIFEST_SHA256 = "c46c15318b8a589ac20b17b8dd28b6687fd2a4eb9c68d318c6f3e16d063673a3"
M1349_AUTHOR_OUTER_SHA256 = "76cd24cc79e886e00e4dd82e8febfe22bdce23aecf353320e46b049da23a34ca"
M1353_BLIND = HW / (
    "reviews/m1353_m1349_motion_ep34_live105_inventory_successor_source_"
    "blind_hammer_r1_20260831")
M1353_REVIEW_SHA256 = "3a660e6c1608baf7e5f6b16383067539c21631f89c310d5aa13656cadcbdde2e"
M1353_MANIFEST_SHA256 = "7770775870e196d39eb213fc3b0bb5819ac1e5b595854065806ef792c2ea8bd7"
M1353_OUTER_SHA256 = "1e2c2f6a10f514770fab6bdf6666ba8d40a11d5393053310cd39014143aa0006"
M1400_RUNNER = HW / (
    "scripts/run_m1400_m1349_motion_ep34_live105_production_one_shot.py")
M1400_RUNNER_SHA256 = "c9d7e0e3d6eca16c710b8bbcf44be3154f1891eb8b3b8452d3fda1a5094668be"
M1412_RELEASE = HW / (
    "contracts/m1412_m1400_m1349_motion_ep34_live105_production_launch_"
    "release_r1_20260831.json")
M1412_RELEASE_SHA256 = "374c8a2e1aa770e1ee3868f5575db704ffd59b72c9518678979c480b890ab5ef"
M1430_FINAL = HW / (
    "reviews/m1430_m1412_m1400_m1349_motion_ep34_live105_production_final_"
    "launch_hammer_r1_20260831")
M1430_REVIEW_SHA256 = "f67aaa3e8a9885cefafa8c6ebc86b5c00e9fe46c03995483ebc8d293c5e21473"
M1430_MANIFEST_SHA256 = "38717b8f5a1e9d33c4796fb6386c4337578e78e1612aadb914af315adf8a316f"
M1430_OUTER_SHA256 = "d1b1f18104644752f73a694156cac7c55cf640b662f035a13a5a0460004e8821"
BSA_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/bsa_attention.py")
BSA_SOURCE_SHA256 = "0f77f66dbd331daa77a284199cda33125a1959a005b6f4d592e2e6cda5317187"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
STATIC_ATLIF_SHA256 = "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7"
DEAD_SN2_Q_SHA256 = "2f8e84f85d412008c45a67012da61d1ef7d248456aa64e8925af8aa57e6076a9"
LIVE_ATLIF_SHA256 = "f2dfcedab9ebe77b30b32d84bc38a2b1ea6511b0b3b359feb81a118ad2de252e"
EXPECTED_STATIC_MODULES = 259
EXPECTED_STATIC_ATLIF = 105
EXPECTED_LIVE_MODULES = 247
EXPECTED_LIVE_ATLIF = 93
EXPECTED_ORDERED_RECORDS = 9880
EXPECTED_ATTENTION = 480
EXPECTED_PAYLOAD = 640
EXPECTED_EXECUTION = 7360
EXPECTED_OPERATOR_ROWS = 79

DEAD_SN2_Q = tuple(
    "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.attn."
    "sn2_q.spiking_neuron".format(stage, block)
    for stage, blocks in enumerate((2, 2, 6, 2)) for block in range(blocks)
)

CANONICAL_RESULT = HW / (
    "results/m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831")
CANONICAL_ATTEMPT = HW / (
    "results/.m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_"
    "20260831.attempt_consumed")
CANONICAL_LOG = HW / (
    "results/.m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_"
    "20260831.production.log")
SOURCE_SCHEMA = "m1434_motion_ep34_live93_runtime_successor_source_r1_v1"
SOURCE_STATUS = (
    "SOURCE_ONLY__M1400_FAILURE_DIAGNOSED__DIFFERENT_AUTHOR_BLIND_REQUIRED__NO_GPU")
PASS_TOKEN = "PASS_M1434_LIVE93_SOURCE_SELF_CHECK__NO_ATTEMPT_NO_GPU_NO_CAPTURE"


class M1434Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1434Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_lf_digest(names: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1434Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(token: str):
        raise M1434Error("nonfinite JSON token: " + token)
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_double_seal(root: Path, review_sha: str, manifest_sha: str,
                       outer_sha: str) -> dict[str, Any]:
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(review, review_sha, root.name + " review")
    regular_exact(manifest, manifest_sha, root.name + " manifest")
    regular_exact(outer, outer_sha, root.name + " outer")
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            root.name + " outer content mismatch")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in rows, "duplicate sealed member")
        rows[name] = digest
    require(rows.get("review.json") == review_sha or
            rows.get(str(review.relative_to(ROOT))) == review_sha,
            root.name + " review is not sealed")
    return strict_json(review)


def _load_m1349():
    regular_exact(M1349_SOURCE, M1349_SOURCE_SHA256, "M1349 source")
    spec = importlib.util.spec_from_file_location("m1434_sealed_m1349", M1349_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1349")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    regular_exact(M1349_SOURCE, M1349_SOURCE_SHA256, "M1349 source after import")
    return module


M1349 = _load_m1349()
R1 = M1349.R1
M1249 = M1349.M1249


def expected_failure_errors() -> list[str]:
    return ["call_count:0:{}:0".format(name) for name in DEAD_SN2_Q] + [
        "record_count:247:259"]


def validate_h60_bypass_source() -> None:
    regular_exact(BSA_SOURCE, BSA_SOURCE_SHA256, "H60 bsa_attention source")
    text = BSA_SOURCE.read_text(encoding="utf-8")
    begin = text.index('elif cfg.mode in {"h60", "tx_sc_k_mag_no_carrier_shiftmax"}:')
    end = text.index('elif cfg.mode in {"h82",', begin)
    branch = text[begin:end]
    require("attn = k_orig.mul(gate)" in branch and "self.sn2_q(" not in branch,
            "pinned H60 branch no longer proves sn2_q bypass")


def verify_predecessors() -> None:
    regular_exact(M1349_TEST, M1349_TEST_SHA256, "M1349 test")
    regular_exact(M1349_CONTRACT, M1349_CONTRACT_SHA256, "M1349 contract")
    regular_exact(M1400_RUNNER, M1400_RUNNER_SHA256, "M1400 runner")
    regular_exact(M1412_RELEASE, M1412_RELEASE_SHA256, "M1412 release")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    author = verify_double_seal(M1349_AUTHOR, M1349_AUTHOR_REVIEW_SHA256,
                                M1349_AUTHOR_MANIFEST_SHA256,
                                M1349_AUTHOR_OUTER_SHA256)
    require(author.get("status") ==
            "PASS_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED",
            "M1349 author status mismatch")
    blind = verify_double_seal(M1353_BLIND, M1353_REVIEW_SHA256,
                               M1353_MANIFEST_SHA256, M1353_OUTER_SHA256)
    require(blind.get("status") ==
            "PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED",
            "M1353 blind status mismatch")
    final = verify_double_seal(M1430_FINAL, M1430_REVIEW_SHA256,
                               M1430_MANIFEST_SHA256, M1430_OUTER_SHA256)
    require(final.get("status") ==
            "PASS_M1400_M1349_EP34_LIVE105_FINAL_LAUNCH_AUTHORITY" and
            final.get("authorization") == {
                "launch": True, "runs": 1, "automatic_retry": False},
            "M1430 final launch authority mismatch")
    validate_h60_bypass_source()


def expected_live93_inventory(
        static_inventory: dict[str, list[str]]) -> dict[str, list[str]]:
    require(type(static_inventory) is dict and
            set(static_inventory) == set(R1.EXPECTED_STATIC_COUNTS),
            "static inventory category set drift")
    require(all(type(value) is list for value in static_inventory.values()),
            "static inventory categories must be lists")
    require({key: len(value) for key, value in static_inventory.items()} ==
            R1.EXPECTED_STATIC_COUNTS, "static inventory count drift")
    atlif = list(static_inventory["atlif"])
    require(atlif == list(M1349.EXPECTED_ATLIF_NAMES) and
            terminal_lf_digest(atlif) == STATIC_ATLIF_SHA256,
            "static ATLIF authority drift")
    require(len(atlif) == EXPECTED_STATIC_ATLIF and
            not any(".sn_v." in name for name in atlif),
            "static H60 ATLIF inventory mismatch")
    observed_dead = tuple(name for name in atlif if ".sn2_q." in name)
    require(observed_dead == DEAD_SN2_Q and
            terminal_lf_digest(observed_dead) == DEAD_SN2_Q_SHA256,
            "exact H60 dead sn2_q set drift")
    live_atlif = sorted(set(atlif) - set(DEAD_SN2_Q))
    require(len(live_atlif) == EXPECTED_LIVE_ATLIF and
            terminal_lf_digest(live_atlif) == LIVE_ATLIF_SHA256,
            "live93 ATLIF set drift")
    result = {key: sorted(value) for key, value in static_inventory.items()}
    result["atlif"] = live_atlif
    require({key: len(value) for key, value in result.items()} ==
            dict(R1.EXPECTED_STATIC_COUNTS, atlif=EXPECTED_LIVE_ATLIF),
            "live inventory counts drift")
    require(sum(len(value) for value in result.values()) == EXPECTED_LIVE_MODULES,
            "live inventory total is not 247")
    return result


def validate_failure_observation(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT) if policy is None else policy
    observed = policy.get("m1400_failure_observation")
    expected = {
        "attempt_sha256": "0c17499d2fec3a58272af40c22186f5537873d2c061da9c13ff2acceff968e37",
        "production_log_sha256": "45fe9d6054b3475a47baa3ef693d1a941e82e14077954feb27bdd20fa6b40ba5",
        "failed_json_sha256": "225fec4fce4d246a0deb851406ef44bdb3dbe657f56d94356b80888f7178849b",
        "sample0_snapshot_manifest_sha256":
            "d9dbd76ff51711518bdfeca9463c9829d4ba1e03de56e449b50b1d11ed58f23c",
        "sample0_ordered_sha256":
            "82e52f5fc69a53a2ff0ddf525adcc9981791564967470f8970c3a49e91fe3298",
        "staging_basename":
            ".m1349_motion_ep34_live105_unified_hardware_capture_s40_r1_"
            "20260831._jj9kq9x",
        "sample0_call_audit": {
            "status": "FAIL", "samples": 1,
            "live_modules_per_sample": 259, "records": 247,
            "expected_records": 259, "dead_modules": 0,
            "errors": expected_failure_errors(),
        },
        "sample0_category_counts": {
            "c1_conv3x3": 4, "decoder_convtranspose": 4, "atlif": 93,
            "fc1": 12, "fc2": 12, "patch_embed": 8,
            "batch_norm": 78, "qkv": 24, "attention": 12,
        },
        "controller_after_failure": {"pid": 3804343, "state": "T"},
        "automatic_retry": False,
        "controller_restore_permitted": False,
    }
    require(observed == expected, "M1400 failure observation drift")
    require(sum(observed["sample0_category_counts"].values()) == 247,
            "sample0 category population is not 247")
    return observed


def replay_sample0_forensic_summary(
        policy: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = validate_failure_observation(policy)
    audit = observed["sample0_call_audit"]
    require(audit["errors"] == expected_failure_errors(),
            "sample0 errors are not exactly the twelve bypassed sn2_q modules")
    return {
        "status": "PASS", "errors": [], "samples": 1,
        "live_modules_per_sample": EXPECTED_LIVE_MODULES,
        "records": audit["records"], "expected_records": EXPECTED_LIVE_MODULES,
        "dead_modules": len(DEAD_SN2_Q),
    }


def audit_with_live93(records: list[dict[str, Any]],
                      live_inventory: dict[str, list[str]],
                      sample_ids: list[int]) -> dict[str, Any]:
    old_dead = R1.DEAD_SN_V
    try:
        R1.DEAD_SN_V = DEAD_SN2_Q
        return R1.audit_call_matrix(records, live_inventory, sample_ids)
    finally:
        R1.DEAD_SN_V = old_dead


def validate_snapshot_population_live93(staging: Path) -> None:
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
        audit = manifest["call_audit"]
        require(manifest["sample_id"] == sample and audit["status"] == "PASS" and
                audit["records"] == EXPECTED_LIVE_MODULES and
                audit["live_modules_per_sample"] == EXPECTED_LIVE_MODULES and
                audit["dead_modules"] == len(DEAD_SN2_Q),
                "sample forensic live93 audit mismatch")
        require(set(manifest["files"]) == expected_files,
                "snapshot file population mismatch")
        for name, digest in manifest["files"].items():
            R1.regular(sample_root / name, "snapshot member")
            require(R1.sha256(sample_root / name) == digest,
                    "snapshot member SHA mismatch")
    require(sorted(path.name for path in root.iterdir() if path.is_dir()) ==
            ["sample_{:02d}".format(sample) for sample in range(40)],
            "snapshot directory population mismatch")


def final_validate_and_seal_live93(staging, writer_type, selected_identity) -> None:
    staging = Path(staging)
    ordered = [json.loads(line) for line in
               (staging / "unified_ordered_records.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    audit = audit_with_live93(ordered, writer_type.ACTIVE_WRITER.live_inventory,
                              list(range(40)))
    require(audit["status"] == "PASS" and
            len(ordered) == EXPECTED_ORDERED_RECORDS,
            "final ordered population is not 40x247")
    attention = R1.strict_json(staging / "attention_qk/manifest.json")
    require(len(attention["records"]) == EXPECTED_ATTENTION,
            "attention population is not 480")
    R1.validate_payload_population(staging)
    validate_snapshot_population_live93(staging)
    execution = json.loads((staging / "execution_trace.json").read_text(encoding="utf-8"))
    operators = json.loads((staging / "operator_runtime.json").read_text(encoding="utf-8"))
    atlif = json.loads((staging / "atlif_activity.json").read_text(encoding="utf-8"))
    require(len(execution) == EXPECTED_EXECUTION,
            "execution population must be 40x184")
    require(len(operators) == EXPECTED_OPERATOR_ROWS and
            all(int(row["calls"]) == 40 for row in operators),
            "operator runtime must contain 79 rows at 40 calls")
    require(len(atlif) == EXPECTED_LIVE_ATLIF and
            all(int(row["calls"]) == 40 for row in atlif),
            "ATLIF runtime must contain 93 live rows at 40 calls")
    names = sorted(row["name"] for row in atlif)
    require(terminal_lf_digest(names) == LIVE_ATLIF_SHA256 and
            not set(names) & set(DEAD_SN2_Q), "ATLIF live93 identity drift")
    dead_payload = {
        "kind": "H60_STATIC_BUT_RUNTIME_BYPASSED",
        "count": 12, "names": list(DEAD_SN2_Q),
        "terminal_lf_sha256": DEAD_SN2_Q_SHA256,
    }
    manifest_path = staging / "manifest.json"
    manifest = R1.strict_json(manifest_path)
    manifest.update({
        "schema": "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1",
        "status": "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
        "m1434_runtime_contract": {
            "static_modules": EXPECTED_STATIC_MODULES,
            "static_atlif": EXPECTED_STATIC_ATLIF,
            "static_atlif_terminal_lf_sha256": STATIC_ATLIF_SHA256,
            "live_modules_per_sample": EXPECTED_LIVE_MODULES,
            "live_atlif": EXPECTED_LIVE_ATLIF,
            "live_atlif_terminal_lf_sha256": LIVE_ATLIF_SHA256,
            "dead_atlif": dead_payload, "dead_calls_per_sample": 0,
            "ordered_records": EXPECTED_ORDERED_RECORDS,
            "attention_records": EXPECTED_ATTENTION,
            "payload_files": EXPECTED_PAYLOAD,
            "final_selection_identity": selected_identity,
        },
        "forensic_snapshots": {
            "samples": 40, "atomic_per_sample": True,
            "failure_forensic_only": True, "automatic_canonical_promotion": False,
        },
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    admission = {
        "schema": "m1434_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": EXPECTED_ORDERED_RECORDS, "attention": EXPECTED_ATTENTION,
        "payload_files": EXPECTED_PAYLOAD, "execution": EXPECTED_EXECUTION,
        "operator_rows": EXPECTED_OPERATOR_ROWS,
        "atlif_live_rows": EXPECTED_LIVE_ATLIF,
        "atlif_static": EXPECTED_STATIC_ATLIF,
        "static_atlif_terminal_lf_sha256": STATIC_ATLIF_SHA256,
        "live_atlif_terminal_lf_sha256": LIVE_ATLIF_SHA256,
        "dead_atlif": dead_payload,
        "claim_boundary": {
            "capture_only": True, "paper_result": False, "cycles": False,
            "speedup": False, "energy": False, "ppa": False,
        },
    }
    (staging / "m1434_admission.json").write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    R1.write_double_seal(staging)


@contextlib.contextmanager
def patched_live93_capture_chain() -> Iterator[None]:
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
        # R1's historical variable name is retained as an internal audit hook;
        # the public M1434 contract names the exact set dead_atlif/dead_sn2_q.
        R1.DEAD_SN_V = DEAD_SN2_Q
        R1.EXPECTED_LIVE_COUNTS = dict(R1.EXPECTED_STATIC_COUNTS,
                                       atlif=EXPECTED_LIVE_ATLIF)
        R1.expected_live_inventory = expected_live93_inventory
        R1.validate_snapshot_population = validate_snapshot_population_live93
        R1.final_validate_and_seal = final_validate_and_seal_live93
        M1249.CANONICAL_RESULT = CANONICAL_RESULT
        yield
    finally:
        R1.DEAD_SN_V = originals["dead"]
        R1.EXPECTED_LIVE_COUNTS = originals["counts"]
        R1.expected_live_inventory = originals["inventory"]
        R1.validate_snapshot_population = originals["snapshots"]
        R1.final_validate_and_seal = originals["final"]
        M1249.CANONICAL_RESULT = originals["result"]


def build_runtime() -> tuple[dict[str, Any], dict[str, Any]]:
    old_runtime, binding = M1349.build_runtime()
    identity = binding.get("identity", {})
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
            identity.get("config_sha256") == CONFIG_SHA256,
            "ep34 checkpoint/config identity drift")
    runtime = copy.deepcopy(old_runtime)
    runtime["contract_path"] = str(SOURCE_CONTRACT.relative_to(ROOT))
    runtime["output"] = {"path": str(CANONICAL_RESULT.relative_to(ROOT))}
    require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
            runtime["capture"] == {"attention_windows_per_call": 100},
            "M1434 runtime projection drift")
    return runtime, binding


def delegate_for_future_release(runtime: dict[str, Any], binding: dict[str, Any],
                                substrate: Any) -> Path:
    expected, rebound = build_runtime()
    require(runtime == expected and binding == rebound,
            "under-lease M1434 binding drift")
    with patched_live93_capture_chain():
        output = M1249.run_capture(runtime, binding, substrate=substrate)
    require(Path(output) == CANONICAL_RESULT, "capture returned non-M1434 result")
    return Path(output)


def require_fresh_namespaces() -> None:
    paths = (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)
    require(len(set(paths)) == 3 and all("m1434_" in path.name for path in paths),
            "M1434 namespaces are not distinct/fresh")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1434 namespace is not fresh")


def validate_source_policy(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT) if policy is None else policy
    require(policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") == SOURCE_STATUS, "M1434 source policy mismatch")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "M1434 source identity mismatch")
    require(policy.get("test") == {
        "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST),
        "passed": 22, "failed": 0}, "M1434 test identity/result mismatch")
    predecessors = policy.get("predecessors")
    require(type(predecessors) is dict and set(predecessors) == {
        "m1349_source", "m1349_test", "m1349_contract", "m1349_author_seal",
        "m1353_blind_seal", "m1400_runner", "m1412_release",
        "m1430_final_seal", "h60_bsa_source",
    }, "M1434 predecessor key set mismatch")
    require(predecessors["m1349_source"] == {
        "path": str(M1349_SOURCE.relative_to(ROOT)), "sha256": M1349_SOURCE_SHA256} and
        predecessors["m1349_test"] == {
            "path": str(M1349_TEST.relative_to(ROOT)), "sha256": M1349_TEST_SHA256} and
        predecessors["m1349_contract"] == {
            "path": str(M1349_CONTRACT.relative_to(ROOT)),
            "sha256": M1349_CONTRACT_SHA256} and
        predecessors["m1400_runner"] == {
            "path": str(M1400_RUNNER.relative_to(ROOT)), "sha256": M1400_RUNNER_SHA256} and
        predecessors["m1412_release"] == {
            "path": str(M1412_RELEASE.relative_to(ROOT)), "sha256": M1412_RELEASE_SHA256} and
        predecessors["h60_bsa_source"] == {
            "path": str(BSA_SOURCE.relative_to(ROOT)), "sha256": BSA_SOURCE_SHA256},
        "M1434 predecessor file projection mismatch")
    require(predecessors["m1349_author_seal"] == {
        "path": str(M1349_AUTHOR.relative_to(ROOT)),
        "review_sha256": M1349_AUTHOR_REVIEW_SHA256,
        "manifest_sha256": M1349_AUTHOR_MANIFEST_SHA256,
        "outer_file_sha256": M1349_AUTHOR_OUTER_SHA256} and
        predecessors["m1353_blind_seal"] == {
            "path": str(M1353_BLIND.relative_to(ROOT)),
            "review_sha256": M1353_REVIEW_SHA256,
            "manifest_sha256": M1353_MANIFEST_SHA256,
            "outer_file_sha256": M1353_OUTER_SHA256} and
        predecessors["m1430_final_seal"] == {
            "path": str(M1430_FINAL.relative_to(ROOT)),
            "review_sha256": M1430_REVIEW_SHA256,
            "manifest_sha256": M1430_MANIFEST_SHA256,
            "outer_file_sha256": M1430_OUTER_SHA256},
        "M1434 predecessor seal projection mismatch")
    require(policy.get("identity") == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_source_sha256": M1349.PROFILE_SOURCE_SHA256,
        "atlif_overlay_source_sha256": M1349.ATLIF_OVERLAY_SOURCE_SHA256,
        "h60_bsa_source_sha256": BSA_SOURCE_SHA256,
        "checkpoint_load": {"missing": 0, "unexpected": 0},
        "attention_mode": "h60"}, "M1434 bound identity mismatch")
    require(policy.get("inventory") == {
        "static_atlif_terminal_lf_sha256": STATIC_ATLIF_SHA256,
        "dead_sn2_q_terminal_lf_sha256": DEAD_SN2_Q_SHA256,
        "live_atlif_terminal_lf_sha256": LIVE_ATLIF_SHA256,
        "dead_reason": "H60 forward computes Shiftmax and attn=K*gate without "
                       "calling self.sn2_q; sn2_q remains a static ATLIF-installed "
                       "named module."}, "M1434 inventory identity mismatch")
    require(policy.get("population") == {
        "static_modules": 259, "static_atlif": 105,
        "live_modules_per_sample": 247, "live_atlif": 93,
        "dead_sn2_q": 12, "ordered_records": 9880,
        "attention_records": 480, "payload_files": 640,
    }, "M1434 population policy mismatch")
    require(policy.get("launch_authorized") is False and
            policy.get("automatic_retry") is False and
            policy.get("runs") == 0, "source policy cannot authorize production")
    require(policy.get("new_namespaces") == {
        "result": str(CANONICAL_RESULT.relative_to(ROOT)),
        "attempt": str(CANONICAL_ATTEMPT.relative_to(ROOT)),
        "log": str(CANONICAL_LOG.relative_to(ROOT))},
        "M1434 namespace projection mismatch")
    require(policy.get("claim_boundary") == {
        "source_and_tests_only": True, "different_author_blind_required": True,
        "gpu": False, "forward": False, "capture": False, "attempt": False,
        "remote": False, "controller_signal": False,
        "controller_restore": False, "production_release": False,
        "hardware_result": False}, "M1434 claim boundary mismatch")
    validate_failure_observation(policy)
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1434 is source-only")
    verify_predecessors()
    validate_source_policy()
    replay = replay_sample0_forensic_summary()
    require(replay["status"] == "PASS" and replay["records"] == 247,
            "sample0 forensic replay did not close")
    require_fresh_namespaces()
    require(R1.DEAD_SN_V != DEAD_SN2_Q,
            "source self-check must leave predecessor globals unchanged")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

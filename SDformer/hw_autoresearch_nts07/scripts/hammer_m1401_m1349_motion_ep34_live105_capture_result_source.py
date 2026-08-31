#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only, fail-closed result hammer for the final live-105 ep34 capture."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Sequence


SOURCE_FILE = Path(__file__).resolve()
ROOT = SOURCE_FILE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1338_SOURCE = HW / "scripts/hammer_m1338_m1327_final_ep34_capture_result_source.py"
M1338_SOURCE_SHA256 = "173452c9160347826f181ccf0e5865a90cb2e97c0d54a5af5afa8343ee12385a"
M1349_SOURCE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1349_motion_ep34_live105_inventory_successor_r2.py"
M1349_SOURCE_SHA256 = "3fe0f51acf489cf2f4d1a65f83f872b49a5fde79401a2fdb525768e681fbbbe5"
M1349_TEST = HW / "tests/test_m1349_motion_ep34_live105_inventory_successor.py"
M1349_TEST_SHA256 = "b20e06bcecb9fab1a326701e40e7bb72c5f13a3204a9d52470b58237a747492f"
M1349_CONTRACT = HW / "contracts/m1349_motion_ep34_live105_inventory_successor_source_contract_r1_20260831.json"
M1349_CONTRACT_SHA256 = "ce2f373eef512237a0e0ee087134176384c30663bd52d42aa68c68b05fbd4712"
M1349_AUTHOR = HW / "reviews/m1349_motion_ep34_live105_inventory_successor_source_author_r1_20260831"
M1353_BLIND = HW / "reviews/m1353_m1349_motion_ep34_live105_inventory_successor_source_blind_hammer_r1_20260831"
M1349_AUTHOR_SEAL = {
    "review": "bd29fae08da4978416477bcc5cb93a36d254cee2456a489452a8e5ad4ea98c57",
    "manifest": "c46c15318b8a589ac20b17b8dd28b6687fd2a4eb9c68d318c6f3e16d063673a3",
    "outer": "76cd24cc79e886e00e4dd82e8febfe22bdce23aecf353320e46b049da23a34ca",
}
M1353_BLIND_SEAL = {
    "review": "3a660e6c1608baf7e5f6b16383067539c21631f89c310d5aa13656cadcbdde2e",
    "manifest": "7770775870e196d39eb213fc3b0bb5819ac1e5b595854065806ef792c2ea8bd7",
    "outer": "1e2c2f6a10f514770fab6bdf6666ba8d40a11d5393053310cd39014143aa0006",
}
CANONICAL_RESULT = HW / "results/m1349_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1401_m1349_motion_ep34_live105_capture_result_hammer_source_contract_r1_20260831.json"
TEST = HW / "tests/test_hammer_m1401_m1349_motion_ep34_live105_capture_result_source.py"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
PROFILE_SHA256 = "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"
SELECTION_SHA256 = "4af7b7e1b4a174440331268fcfffda44896d86d02c7d20195e7a49d73eae6cd0"
ATLIF_NAMES_SHA256 = "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7"
EXPECTED_COUNTS = {
    "c1_conv3x3": 4,
    "decoder_convtranspose": 4,
    "atlif": 105,
    "fc1": 12,
    "fc2": 12,
    "patch_embed": 8,
    "batch_norm": 78,
    "qkv": 24,
    "attention": 12,
}
SOURCE_SCHEMA = "m1401_m1349_motion_ep34_live105_capture_result_hammer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__LIVE105_RESULT_HAMMER__CANONICAL_MUST_PREEXIST__NO_CAPTURE"
PASS_TOKEN = "PASS_M1401_SOURCE_SELF_CHECK__NO_CAPTURE_NO_CANONICAL_RESULT"


class M1401Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1401Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1401Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " not regular")
    require(sha256(path) == digest, label + " SHA drift")


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M1338 = load_exact("m1401_sealed_m1338", M1338_SOURCE, M1338_SOURCE_SHA256)
M1349 = load_exact("m1401_sealed_m1349", M1349_SOURCE, M1349_SOURCE_SHA256)
BASE = M1338.OLD.OLD


def strict_json(path: Path) -> dict[str, Any]:
    value = BASE.strict_file(path)
    require(type(value) is dict, "JSON root is not object")
    return value


def verify_authority_dir(root: Path, pins: dict[str, str], expected_status: str) -> None:
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    regular_exact(manifest, pins["manifest"], "authority manifest")
    regular_exact(outer, pins["outer"], "authority outer seal")
    require(outer.read_text().split() == [pins["manifest"], "SHA256SUMS"],
            "authority outer content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "malformed authority manifest row")
        digest, name = fields[0], fields[1].lstrip("*")
        pure = PurePosixPath(name)
        require(name not in rows and pure.parts and not pure.is_absolute() and
                ".." not in pure.parts, "unsafe/duplicate authority member")
        if pure.parts[0] in {"hw_autoresearch_nts07", "neuron_experiments"}:
            member = ROOT / pure
        else:
            member = root / pure
        regular_exact(member, digest, "authority member " + name)
        rows[name] = digest
    review_names = [name for name, digest in rows.items()
                    if name.endswith("review.json") and digest == pins["review"]]
    require(len(review_names) == 1, "authority review member drift: " + root.name)
    review = strict_json(root / "review.json")
    require(review.get("status") == expected_status, "authority status drift: " + root.name)


def canonical_absent(path: Path = CANONICAL_RESULT) -> None:
    require(not os.path.lexists(str(path)), "canonical namespace already exists")


def canonical_directory(path: Path = CANONICAL_RESULT) -> None:
    require(os.path.lexists(str(path)), "canonical result absent")
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not path.is_symlink(), "canonical result not real directory")


def terminal_lf_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def validate_ordered(ordered: list[Any]) -> dict[str, Any]:
    require(len(ordered) == 10360, "ordered population is not 40x259")
    sequences: list[list[tuple[str, str]]] = []
    atlif_names: list[str] | None = None
    for sample in range(40):
        rows = ordered[sample * 259:(sample + 1) * 259]
        require(len(rows) == 259 and all(type(row) is dict for row in rows),
                "sample ordered slice malformed")
        require(all(type(row.get("sample_id")) is int and row["sample_id"] == sample
                    for row in rows), "sample id/order drift")
        sequence: list[tuple[str, str]] = []
        counts = {key: 0 for key in EXPECTED_COUNTS}
        names_by_category = {key: [] for key in EXPECTED_COUNTS}
        for row in rows:
            category, name = row.get("category"), row.get("name")
            require(type(category) is str and category in EXPECTED_COUNTS and
                    type(name) is str and name, "ordered category/name malformed")
            require(type(row.get("input")) is dict and type(row.get("payload")) is dict,
                    "ordered input/payload missing")
            counts[category] += 1
            names_by_category[category].append(name)
            sequence.append((category, name))
        require(counts == EXPECTED_COUNTS, "per-sample category count drift")
        require(all(len(values) == len(set(values)) for values in names_by_category.values()),
                "per-category name population not unique")
        current_atlif = names_by_category["atlif"]
        require(current_atlif == sorted(current_atlif), "ATLIF sample order is not sorted")
        require(terminal_lf_digest(current_atlif) == ATLIF_NAMES_SHA256,
                "ATLIF live-105 identity drift")
        if atlif_names is None:
            atlif_names = current_atlif
        require(current_atlif == atlif_names, "ATLIF sequence differs across samples")
        sequences.append(sequence)
    require(all(sequence == sequences[0] for sequence in sequences),
            "full 259-row sequence differs across samples")
    require(atlif_names == list(M1349.EXPECTED_ATLIF_NAMES),
            "ordered ATLIF names differ from sealed M1349 authority")
    return {"ordered_rows": len(ordered), "samples": 40,
            "live_modules_per_sample": 259, "all_sample_sequences_equal": True}


def validate_admission(admission: dict[str, Any]) -> None:
    require(admission == {
        "schema": "m1343_final_capture_admission_r1_v1",
        "status": "PASS",
        "ordered": 10360,
        "attention": 480,
        "payload_files": 640,
        "execution": 7360,
        "operator_rows": 79,
        "atlif_live_rows": 105,
        "atlif_static": 105,
        "dead_sn_v": [],
        "atlif_names_sha256": ATLIF_NAMES_SHA256,
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False,
                           "energy": False, "ppa": False},
    }, "M1343 admission drift")


def validate_manifest(manifest: dict[str, Any]) -> None:
    require(manifest.get("schema") ==
            "m1343_motion_ep34_live105_unified_hardware_capture_r1_v1" and
            manifest.get("status") ==
            "CAPTURE_COMPLETE__FRESH_M1343_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "live105 manifest schema/status drift")
    identity = manifest.get("identity")
    require(type(identity) is dict, "manifest identity missing")
    load = identity.get("checkpoint_load_audit")
    require(load == {"missing_count": 0, "unexpected_count": 0},
            "checkpoint load audit drift")
    require(identity.get("module_counts") ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "module count drift")
    selected = identity.get("selection", {}).get("selected", {})
    require(selected.get("candidate_id") == "resume_ep34" and
            selected.get("epoch") == 34 and
            selected.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA256 and
            selected.get("configuration", {}).get("sha256") == CONFIG_SHA256 and
            selected.get("profile", {}).get("sha256") == PROFILE_SHA256 and
            selected.get("profile", {}).get("samples") == 825 and
            selected.get("profile", {}).get("module_counts") ==
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "selected ep34 identity drift")
    final = manifest.get("m1227_runtime_contract", {}).get("final_selection_identity", {})
    require(final == {"epoch": 34, "checkpoint_sha256": CHECKPOINT_SHA256,
                      "config_sha256": CONFIG_SHA256,
                      "profile_sha256": PROFILE_SHA256,
                      "selection_sha256": SELECTION_SHA256},
            "frozen final selection identity drift")
    runtime = manifest.get("m1343_runtime_contract")
    require(type(runtime) is dict and runtime.get("static_modules") == 259 and
            runtime.get("static_atlif") == 105 and
            runtime.get("live_modules_per_sample") == 259 and
            runtime.get("live_atlif") == 105 and runtime.get("dead_sn_v") == [] and
            runtime.get("dead_calls_per_sample") == 0 and
            runtime.get("atlif_names_sha256") == ATLIF_NAMES_SHA256 and
            runtime.get("ordered_records") == 10360 and
            runtime.get("attention_records") == 480 and
            runtime.get("payload_files") == 640,
            "M1343 runtime contract drift")
    require(manifest.get("claim_boundary") == {
        "capture_only": True, "accuracy": False, "cycles": False,
        "speedup": False, "system_speedup": False, "energy": False,
        "rtl": False, "ppa": False, "fresh_result_hammer_required": True,
    }, "manifest claim boundary drift")


def validate_result(root: Path = CANONICAL_RESULT) -> dict[str, Any]:
    canonical_directory(root)
    rows, seal = BASE.verify_recursive_seal(root)
    required = {"manifest.json", "m1343_admission.json",
                "unified_ordered_records.jsonl", "attention_qk/manifest.json",
                "execution_trace.json", "operator_runtime.json",
                "atlif_activity.json", "RUN_COMPLETE.txt"}
    require(required <= set(rows), "required sealed members missing")
    manifest = strict_json(root / "manifest.json")
    validate_manifest(manifest)
    validate_admission(strict_json(root / "m1343_admission.json"))
    expected_cohort = BASE.OLD.expected_cohort()
    observed = manifest.get("cohort", {}).get("samples")
    require(type(observed) is list and len(observed) == 40 and
            [{key: row[key] for key in expected_cohort[0]} for row in observed] == expected_cohort,
            "cohort identity/order drift")
    ordered = [BASE.strict_text(line) for line in
               (root / "unified_ordered_records.jsonl").read_text(encoding="utf-8").splitlines()]
    ordered_audit = validate_ordered(ordered)
    retained = M1338.validate_retained_payloads(root, rows, ordered)
    attention = M1338.OLD.validate_attention_geometry(root, rows)
    M1338.validate_attention_exact_archive(root)
    try:
        payloads = M1349.R1.validate_payload_population(root)
        M1349.validate_snapshot_population_live105(root)
    except Exception as error:
        raise M1401Error("payload/forensic snapshot validation failed") from error
    require(len(payloads) == 640, "payload population is not 640")
    execution = BASE.strict_file(root / "execution_trace.json")
    operators = BASE.strict_file(root / "operator_runtime.json")
    atlif = BASE.strict_file(root / "atlif_activity.json")
    require(type(execution) is list and len(execution) == 7360,
            "execution population drift")
    require(type(operators) is list and len(operators) == 79 and
            len({row.get("name") for row in operators}) == 79 and
            all(type(row.get("calls")) is int and row["calls"] == 40 for row in operators),
            "operator runtime drift")
    require(type(atlif) is list and len(atlif) == 105 and
            len({row.get("name") for row in atlif}) == 105 and
            all(type(row.get("calls")) is int and row["calls"] == 40 for row in atlif) and
            [row.get("name") for row in atlif] == list(M1349.EXPECTED_ATLIF_NAMES),
            "ATLIF runtime identity/order drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n",
            "completion token drift")
    return {
        "status": "PASS_M1401_M1349_EP34_LIVE105_CAPTURE_RESULT",
        "seal": seal,
        "population": {"ordered": ordered_audit["ordered_rows"],
                       "retained": retained, "attention": attention,
                       "payload": len(payloads), "execution": len(execution),
                       "operator": len(operators), "atlif": len(atlif)},
        "identity": {"checkpoint_sha256": CHECKPOINT_SHA256,
                     "config_sha256": CONFIG_SHA256,
                     "profile_sha256": PROFILE_SHA256},
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False, "energy": False},
    }


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1349_TEST, M1349_TEST_SHA256, "M1349 test")
    regular_exact(M1349_CONTRACT, M1349_CONTRACT_SHA256, "M1349 contract")
    regular_exact(PYTHON, PYTHON_SHA256, "pinned Python")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    verify_authority_dir(M1349_AUTHOR, M1349_AUTHOR_SEAL,
                         "PASS_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED")
    verify_authority_dir(M1353_BLIND, M1353_BLIND_SEAL,
                         "PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED")
    policy = strict_json(SOURCE_CONTRACT)
    expected = {
        "schema": SOURCE_SCHEMA,
        "status": SOURCE_STATUS,
        "date": "2026-08-31",
        "source": {"path": str(SOURCE_FILE.relative_to(ROOT)),
                   "sha256": sha256(SOURCE_FILE)},
        "test": {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST),
                 "passed": 12, "failed": 0},
        "canonical_result": str(CANONICAL_RESULT.relative_to(ROOT)),
        "m1349_authority": {
            "source_sha256": M1349_SOURCE_SHA256,
            "test_sha256": M1349_TEST_SHA256,
            "contract_sha256": M1349_CONTRACT_SHA256,
            "author_review_sha256": M1349_AUTHOR_SEAL["review"],
            "author_manifest_sha256": M1349_AUTHOR_SEAL["manifest"],
            "author_outer_file_sha256": M1349_AUTHOR_SEAL["outer"],
            "blind_review_sha256": M1353_BLIND_SEAL["review"],
            "blind_manifest_sha256": M1353_BLIND_SEAL["manifest"],
            "blind_outer_file_sha256": M1353_BLIND_SEAL["outer"],
        },
        "expected_population": {"samples": 40, "modules_per_sample": 259,
            "ordered": 10360, "atlif": 105, "operator": 79,
            "execution": 7360, "attention": 480, "payload": 640,
            "retained": 320},
        "production_authorized": False,
        "capture_executed": False,
        "actual_result_seal_prefilled": False,
        "claim_boundary": {"source_only": True, "capture_only": False,
            "paper_result": False, "cycles": False, "speedup": False,
            "energy": False, "ppa": False, "system_speedup": False,
            "headline": False},
    }
    require(policy == expected, "source policy exact-set/value drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check", action="store_true")
    group.add_argument("--validate-canonical-result", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.source_self_check:
        validate_source_policy()
        canonical_absent()
        print(PASS_TOKEN)
        return 0
    print(json.dumps(validate_result(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1401Error as error:
        print("M1401_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

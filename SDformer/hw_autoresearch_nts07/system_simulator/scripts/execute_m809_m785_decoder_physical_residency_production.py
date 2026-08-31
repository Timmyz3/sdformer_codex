#!/usr/bin/env python3
"""Fail-closed M809 recovery driver for the frozen M785 decoder model.

M809 is an additive source-only repair after the M808 failure audit.  It
preserves the frozen M785 address-timed scheduler and the two 40/120-record
populations, while adding a byte-exact flat attempt population and a sealed
post-consumption failure receipt.  It also retains the M798 repairs:

* module D1 remains charged but is excluded from every headline total/ratio;
* JSON duplicate keys are rejected;
* the future release is bound to the exact candidate/driver/runner reviewed by
  the fresh source hammer; and
* canonical publication uses Linux renameat2(RENAME_NOREPLACE), followed by a
  root-level four-member seal check.

This file does not authorize or launch production.  A separate exact-SHA true
release and one-shot runner invocation remain mandatory.
"""

import argparse
from collections import defaultdict
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M793_PATH = HERE / "execute_m793_m785_decoder_physical_residency_production.py"
M793_SHA256 = "c868eb5569d856f75d08e01c78d896eee774502df27a07301dc681c65410dd77"
M793_CANDIDATE = (
    HW / "contracts/m793_m785_decoder_physical_residency_production_release_candidate_r1_20260828.json"
)
M793_CANDIDATE_SHA256 = "2dc6b6dc0b110f124446cbd8c3b4b10d5e395f32d384be20098c1d8b6b7634aa"
CANDIDATE_SCHEMA = "m809_m785_decoder_production_recovery_candidate_v1"
RELEASE_SCHEMA = "m809_m785_decoder_production_true_release_v1"
SOURCE_HAMMER_DIR = (
    "reviews/m811_m809_m785_decoder_production_source_fresh_hammer_r1_20260829"
)
M808_REVIEW_DIR = (
    HW / "reviews/m808_m798_decoder_production_precompute_failure_hammer_r1_20260829"
)
M808_REVIEW_SHA256 = "bde011940c4999b3d9229b5cdd08900084a3c522accfccaaaa1b30fba42443d0"
M808_MANIFEST_SHA256 = "18f45f4b8dedf8c93a4ced9b364c4ec525f1aa88c08b736135a20c351a52cea3"
M808_OUTER_SEAL_FILE_SHA256 = "57b19ae72536f19a27e0f8ddee0ea76dd3688f36c065b537239a84b582c633ea"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")
HEADLINE_NUMERATOR = "TYPED_SIGNED_K8"
HEADLINE_DENOMINATOR = "EQUAL_SERVICE_K1X8"
D1_MODULE_INDEX = 1
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Failure("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def strict_json(path: Path) -> object:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            "JSON input must be a regular nonsymlink file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Failure("non-finite JSON constant: " + value)),
        )


def regular_exact(path: Path, expected: str, label: str) -> None:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            label + " is not a regular nonsymlink file")
    require(len(expected) == 64 and sha256(path) == expected,
            label + " SHA drift")


def load_m793():
    regular_exact(M793_PATH, M793_SHA256, "frozen M793 parent driver")
    spec = importlib.util.spec_from_file_location("m809_frozen_m793", M793_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M793 parent driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M793 = load_m793()
M785 = M793.M785


def verify_sealed(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory is not a regular directory: " + str(directory))
    try:
        return dict(M785.verify_sealed_directory(directory))
    except M785.Failure as error:
        raise Failure(str(error)) from error


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    return {
        name: HW / entry["path"]
        for name, entry in candidate["source_identity"].items()
    }


def _validate_candidate(candidate_path: Path, require_future_absent: bool,
                        attempt_required: bool = False) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict), "candidate must be an object")
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "candidate schema drift")
    require(candidate.get("status") ==
            "SOURCE_ONLY_M809_RECOVERY_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "candidate status drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate must not authorize production")
    require(candidate.get("authorization") == {
        "source_validation": True,
        "synthetic_self_test": True,
        "fresh_source_hammer": True,
        "production_replay": False,
        "result_directory": False,
        "cycles_or_speedup": False,
        "rtl_vcs_eda_gpu_remote": False,
    }, "candidate authorization is not closed")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")

    paths = _candidate_paths(candidate)
    required_sources = {
        "driver", "runner", "tests", "m793_parent_driver",
        "m793_parent_candidate", "analyzer", "storage_oracle",
        "m785_tests", "m785_contract", "recovery_contract",
    }
    require(set(paths) == required_sources, "candidate source set drift")
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate does not bind this exact driver")
    require(paths["runner"].resolve() ==
            HERE / "run_m809_m785_decoder_physical_residency_one_shot.sh",
            "candidate runner path drift")
    require(paths["tests"].resolve() ==
            HERE.parent / "tests/test_m809_m785_decoder_production_recovery.py",
            "candidate tests path drift")
    require(paths["m793_parent_driver"].resolve() == M793_PATH,
            "M793 parent driver path drift")
    require(paths["m793_parent_candidate"].resolve() == M793_CANDIDATE,
            "M793 parent candidate path drift")
    require(candidate["source_identity"]["m793_parent_candidate"]["sha256"] ==
            M793_CANDIDATE_SHA256, "M793 parent candidate SHA drift")
    parent_validation = M793.validate_candidate(M793_CANDIDATE)
    require(parent_validation["status"] ==
            "PASS_M793_SOURCE_CANDIDATE_IDENTITY__NO_PRODUCTION_RUN",
            "frozen M793 parent validation drift")

    m795 = candidate["m795_no_go_repair_basis"]
    m795_dir = HW / m795["directory"]
    m795_identity = verify_sealed(m795_dir)
    regular_exact(m795_dir / "review.json", m795["review_json_sha256"],
                  "M795 review")
    require(m795_identity["manifest_sha256"] == m795["manifest_sha256"] and
            m795_identity["outer_seal_file_sha256"] ==
            m795["outer_seal_file_sha256"], "M795 double seal drift")
    m795_review = strict_json(m795_dir / "review.json")
    require(m795_review.get("status") ==
            "NO_GO_M793_TRUE_RELEASE__P1_3__AUTHOR_ADDITIVE_REPAIR_REQUIRED" and
            m795_review.get("score") == 72 and
            m795_review.get("severity_counts") == {"p0": 0, "p1": 3, "p2": 2},
            "M795 repair authority drift")
    require(m795_review["review_target"]["candidate_sha256"] ==
            M793_CANDIDATE_SHA256 and
            m795_review["review_target"]["driver_sha256"] == M793_SHA256,
            "M795 did not review the frozen M793 basis")

    m808 = candidate["m808_failure_recovery_basis"]
    require((HW / m808["directory"]).resolve() == M808_REVIEW_DIR.resolve(),
            "M808 recovery directory drift")
    m808_identity = verify_sealed(M808_REVIEW_DIR)
    regular_exact(M808_REVIEW_DIR / "review.json",
                  M808_REVIEW_SHA256, "M808 failure audit")
    require(m808_identity["manifest_sha256"] == M808_MANIFEST_SHA256 and
            m808_identity["outer_seal_file_sha256"] ==
            M808_OUTER_SEAL_FILE_SHA256,
            "M808 failure-audit double seal drift")
    m808_review = strict_json(M808_REVIEW_DIR / "review.json")
    require(m808_review.get("status") ==
            "PASS100_FAILURE_AUDIT__M798_IDENTITY_PERMANENTLY_CONSUMED__AUTHORIZE_ADDITIVE_REPAIR_ONLY" and
            m808_review.get("score") == 100 and
            m808_review.get("severity_counts") ==
            {"p0": 1, "p1": 1, "p2": 0},
            "M808 additive-recovery authority drift")
    require(m808 == {
        "directory":
            "reviews/m808_m798_decoder_production_precompute_failure_hammer_r1_20260829",
        "review_json_sha256": M808_REVIEW_SHA256,
        "manifest_sha256": M808_MANIFEST_SHA256,
        "outer_seal_file_sha256": M808_OUTER_SEAL_FILE_SHA256,
        "m798_rerun_authorized": False,
        "new_identity_required": True,
    }, "candidate M808 recovery basis drift")

    recovery_contract = strict_json(paths["recovery_contract"])
    require(recovery_contract.get("schema") ==
            "m809_m798_decoder_production_recovery_contract_v1" and
            recovery_contract.get("status") ==
            "SOURCE_ONLY_RECOVERY_CONTRACT__NO_PRODUCTION_RELEASE" and
            recovery_contract.get("launch_now") is False and
            recovery_contract.get("release") is False and
            recovery_contract.get("m798_attempt_immutable") is True and
            recovery_contract.get("new_identity_required") is True,
            "M809 recovery contract drift")

    require(candidate["corrected_m699_identity"] == {
        "directory":
            "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828",
        "records": 120,
        "manifest_sha256":
            "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0",
        "outer_seal_file_sha256":
            "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c",
    }, "corrected M699 identity drift")
    require(len(candidate["corrected_m699_identity"][
        "outer_seal_file_sha256"]) == 64, "M699 outer SHA must be 64 hex")

    require(candidate["python_identity"] == {
        "path": "/opt/anaconda3/envs/pytorch310/bin/python3.10",
        "sha256":
            "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    }, "candidate Python identity drift")
    regular_exact(Path(candidate["python_identity"]["path"]),
                  candidate["python_identity"]["sha256"], "M809 Python")

    require(candidate["common_resource"] == {
        "lanes": 96,
        "onchip_sram_bytes_macro_rounded": 245760,
        "accumulator_bits": 24,
        "clock_ns": 3.0,
        "external_bytes_per_cycle": 192,
        "weight_bytes": 13824,
        "psum_bytes": 221184,
        "descriptor_control_bytes": 8192,
        "reserved_unallocated_bytes": 2560,
        "reserved_borrow_allowed": False,
        "resource_manifest_sha256":
            "a7400bddb174a00875298cd9bd8d2692e636727ff27b22ae580803383fdea0f3",
    }, "candidate common-resource tuple drift")
    require(candidate["production_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configurations": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "cold_start_and_drain_per_record_timestep": True,
        "records": "40_PLUS_120",
        "timesteps": 10,
        "only_legal_headline_ratio":
            "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8",
        "headline_total_excludes_modules": [1],
        "all_module_total_cycles_retained": True,
        "k8_vs_a1_headline_allowed": False,
        "d1": "COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE",
        "external_opportunity_artifact_candidate_input": False,
    }, "candidate production semantics drift")

    canonical = candidate["canonical"]
    require(canonical == {
        "result":
            "hw_autoresearch_nts07/results/m809_m785_h67_decoder_physical_residency_cycles_r1_20260829",
        "attempt":
            "hw_autoresearch_nts07/results/.m809_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release":
            "hw_autoresearch_nts07/contracts/m809_m785_decoder_physical_residency_production_true_release_r1_20260829.json",
        "failed_or_incomplete_prefix":
            "hw_autoresearch_nts07/results/m809_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_incomplete.",
    }, "candidate canonical-path drift")
    result = REPO / canonical["result"]
    attempt = REPO / canonical["attempt"]
    future = REPO / canonical["future_release"]
    require(not result.exists() and not result.is_symlink(),
            "canonical result already exists")
    if attempt_required:
        require(attempt.is_dir() and not attempt.is_symlink(),
                "attempt must be consumed immediately before production")
        verify_sealed(attempt)
    else:
        require(not attempt.exists() and not attempt.is_symlink(),
                "canonical attempt already exists")
    if require_future_absent:
        require(not future.exists() and not future.is_symlink(),
                "future release must be absent during source review")
    else:
        require(future.is_file() and not future.is_symlink(),
                "canonical true release is absent or nonregular")
    require(candidate["claim_boundary"] == {
        "source_only": True,
        "production_replay": False,
        "production_cycles": False,
        "production_speedup": False,
        "d1_headline": False,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
        "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "candidate claim boundary drift")
    require("m700" not in json.dumps(candidate, sort_keys=True).lower(),
            "external M700 artifact entered candidate identity")
    return {
        "status": "PASS_M809_REPAIRED_SOURCE_CANDIDATE__NO_PRODUCTION_RUN",
        "candidate_sha256": sha256(candidate_path),
        "parent_status": parent_validation["status"],
        "records": {"primary_m686": 40, "secondary_m699": 120},
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def validate_candidate(candidate_path: Path) -> Dict[str, object]:
    return _validate_candidate(candidate_path, require_future_absent=True,
                               attempt_required=False)


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool) -> Dict[str, object]:
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    candidate_validation = _validate_candidate(
        candidate_path, require_future_absent=False,
        attempt_required=attempt_required)
    release = strict_json(release_path)
    candidate = strict_json(candidate_path)
    require(isinstance(release, dict) and release.get("schema") == RELEASE_SCHEMA,
            "true release schema drift")
    require(release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_M809_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release.get("launch_now") is True and
            release.get("release") is True and
            release.get("max_attempts") == 1,
            "true release authorization drift")
    binding = release["candidate_binding"]
    require((HW / binding["path"]).resolve() == candidate_path and
            binding["sha256"] == candidate_validation["candidate_sha256"] and
            sha256(candidate_path) == binding["sha256"],
            "release candidate binding drift")
    require(release["source_identity"] == candidate["source_identity"],
            "release changed or extended candidate source identity")
    reviewed = release["reviewed_source_identity"]
    require(reviewed == {
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256": candidate["source_identity"]["runner"]["sha256"],
    }, "release reviewed-source identity drift")
    require(release["canonical"] == candidate["canonical"],
            "release changed canonical paths")

    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] == SOURCE_HAMMER_DIR,
            "release source-hammer directory drift")
    hammer_dir = HW / hammer["directory"]
    hammer_identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M811 source-hammer review")
    require(hammer_identity["manifest_sha256"] == hammer["manifest_sha256"] and
            hammer_identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"],
            "M811 source-hammer double seal drift")
    hammer_review = strict_json(hammer_dir / "review.json")
    require(hammer_review.get("status") ==
            "PASS100_M809_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            hammer_review.get("score") == 100 and
            hammer_review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0} and
            hammer_review.get("true_release_authorized") is True and
            hammer_review.get("production_launch_authorized") is False,
            "M811 source-hammer PASS100 semantics drift")
    target = hammer_review["review_target"]
    require(target["candidate_sha256"] == sha256(candidate_path) and
            target["driver_sha256"] ==
            candidate["source_identity"]["driver"]["sha256"] and
            target["runner_sha256"] ==
            candidate["source_identity"]["runner"]["sha256"],
            "source hammer did not review exact candidate/driver/runner")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
        "headline_excludes_module_indices": [1],
        "all_module_total_cycles_retained": True,
        "d1": "COMMON_CHARGED_DIAGNOSTIC_NONHEADLINE",
    }, "release runtime semantics drift")
    require(release["claim_boundary"] == {
        "decoder_component_cycles_after_result_hammer": True,
        "production_speedup_before_result_hammer": False,
        "d1_headline": False,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
        "system_speedup": False,
        "rtl_vcs_eda_energy_ppa": False,
    }, "release claim boundary drift")

    expected_sha = os.environ.get("M809_EXPECTED_RELEASE_SHA256", "")
    require(len(expected_sha) == 64 and sha256(release_path) == expected_sha,
            "caller did not supply exact independently reviewed release SHA")
    sidecar = Path(str(release_path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "true-release sidecars must be regular nonsymlink files")
    require(sidecar.read_text(encoding="utf-8") ==
            expected_sha + "  " + release_path.name + "\n",
            "true-release member sidecar content drift")
    require(outer.read_text(encoding="utf-8") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            "true-release outer sidecar content drift")

    attempt = REPO / candidate["canonical"]["attempt"]
    if attempt_required:
        require(attempt.is_dir() and not attempt.is_symlink(),
                "attempt must be consumed immediately before production")
        verify_sealed(attempt)
    else:
        require(not attempt.exists() and not attempt.is_symlink(),
                "release preflight requires unconsumed attempt")
    return {"release": release, "candidate": candidate,
            "candidate_validation": candidate_validation}


def validate_consumed_attempt(release_path: Path, candidate_path: Path,
                              attempt_path: Path) -> Dict[str, object]:
    """Execute the full post-consumption identity gate without scheduling."""
    gate = validate_true_release(
        release_path, candidate_path, attempt_required=True)
    candidate = gate["candidate"]
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    attempt_path = Path(attempt_path).resolve()
    require(attempt_path == canonical_attempt,
            "consumed-attempt path drift")
    identity = verify_sealed(attempt_path)
    receipt = strict_json(attempt_path / "attempt.json")
    require(receipt.get("schema") ==
            "m809_m785_decoder_production_attempt_v1" and
            receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY" and
            receipt.get("max_attempts") == 1 and
            receipt.get("candidate_sha256") == sha256(candidate_path) and
            receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            receipt.get("runner_sha256") ==
            candidate["source_identity"]["runner"]["sha256"] and
            receipt.get("release_sha256") == sha256(release_path),
            "consumed-attempt receipt identity drift")
    return {
        "status": "PASS_M809_CONSUMED_ATTEMPT_PREFLIGHT__NO_SCHEDULE_ROWS",
        "attempt_manifest_sha256": identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256":
            identity["outer_seal_file_sha256"],
        "production_replay": False,
        "scheduled_rows": 0,
        "production_cycles": None,
    }


def _sum_maps(target: Dict[str, int], source: Mapping[str, object]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def headline_ratio(per_config: Mapping[str, Mapping[str, int]]) -> float:
    denominator = int(per_config[HEADLINE_DENOMINATOR][
        "headline_total_cycles"])
    numerator = int(per_config[HEADLINE_NUMERATOR]["headline_total_cycles"])
    require(denominator > 0 and numerator > 0, "zero headline cycles")
    return denominator / numerator


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_production(release_path: Path, candidate_path: Path,
                   attempt_path: Path, output: Path) -> Dict[str, object]:
    gate = validate_true_release(
        release_path, candidate_path, attempt_required=True)
    candidate = gate["candidate"]
    canonical = candidate["canonical"]
    require(Path(attempt_path).resolve() ==
            (REPO / canonical["attempt"]).resolve(), "attempt path drift")
    attempt_identity = verify_sealed(Path(attempt_path))
    receipt = strict_json(Path(attempt_path) / "attempt.json")
    require(receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY" and
            receipt.get("release_sha256") == sha256(release_path) and
            receipt.get("candidate_sha256") == sha256(candidate_path),
            "attempt receipt identity drift")
    output = Path(output).resolve()
    canonical_result = (REPO / canonical["result"]).resolve()
    require(output.parent == canonical_result.parent and
            output.name.startswith(canonical_result.name + ".stage."),
            "driver output must be a fresh staging sibling")
    require(not output.exists() and not output.is_symlink(),
            "driver staging output already exists")
    output.mkdir(mode=0o700)

    contract_path = HW / candidate["source_identity"]["m785_contract"]["path"]
    contract = strict_json(contract_path)
    resource = M785.resource_from_contract(contract)
    mapper_row = contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(
        HW / mapper_row["path"], mapper_row["sha256"], "m809_m672_mapper")
    m712 = contract["inputs"]["m712_oracle"]
    m722 = contract["inputs"]["m722r2_oracle"]
    storage = contract["inputs"]["m785_storage_oracle"]
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])

    populations = (
        ("primary_m686", "M686_ZURICH_CITY_09_A_S10", 40),
        ("secondary_m699", "M699_DSEC_S3X10", 120),
    )
    detailed = []
    aggregate = {}
    for input_name, population_id, expected_records in populations:
        entry = contract["inputs"][input_name]
        payload_root = HW / entry["directory"]
        manifest = strict_json(payload_root / "manifest.json")
        records = M785.normalized_population_records(manifest, population_id)
        require(len(records) == expected_records,
                "production record count drift")
        per_config = {}
        for config in CONFIGS:
            total_cycles = 0
            headline_total_cycles = 0
            d1_total_cycles = 0
            requests = 0
            transactions = 0
            cycle_classes = defaultdict(int)
            address_hashes = []
            commit_hashes = []
            module_cycles = defaultdict(int)
            module_requests = defaultdict(int)
            for record in records:
                module_index = int(record["module_index"])
                for timestep in range(10):
                    schedule = M785.AddressTimedScheduler(resource).schedule(
                        M785.expand_transactions(M785.iter_record_transactions(
                            mapper, record, payload_root, population_id, config,
                            timestep, oracles)))
                    eligible = module_index != D1_MODULE_INDEX
                    row = {
                        "population_id": population_id,
                        "sequence": record["sequence"],
                        "sample_id": int(record["sample_id"]),
                        "module_index": module_index,
                        "timestep": timestep,
                        "config": config,
                        "headline_eligible": eligible,
                        "total_cycles": int(schedule["total_cycles"]),
                        "expanded_request_count": int(
                            schedule["expanded_request_count"]),
                        "compressed_transaction_count": int(
                            schedule["compressed_transaction_count"]),
                        "cycle_classes": {
                            key: int(value) for key, value in
                            schedule["cycle_classes"].items()
                        },
                        "transaction_address_sha256":
                            schedule["transaction_address_sha256"],
                        "commit_sequence_sha256":
                            schedule["commit_sequence_sha256"],
                    }
                    detailed.append(row)
                    total_cycles += row["total_cycles"]
                    if eligible:
                        headline_total_cycles += row["total_cycles"]
                    else:
                        d1_total_cycles += row["total_cycles"]
                    requests += row["expanded_request_count"]
                    transactions += row["compressed_transaction_count"]
                    _sum_maps(cycle_classes, row["cycle_classes"])
                    module_cycles[str(module_index)] += row["total_cycles"]
                    module_requests[str(module_index)] += row[
                        "expanded_request_count"]
                    address_hashes.append(row["transaction_address_sha256"])
                    commit_hashes.append(row["commit_sequence_sha256"])
                    del schedule
            require(total_cycles == headline_total_cycles + d1_total_cycles,
                    "D1/headline cycle partition mismatch")
            per_config[config] = {
                "total_cycles": total_cycles,
                "headline_total_cycles": headline_total_cycles,
                "diagnostic_d1_total_cycles": d1_total_cycles,
                "headline_excluded_module_indices": [D1_MODULE_INDEX],
                "expanded_request_count": requests,
                "compressed_transaction_count": transactions,
                "cycle_classes": dict(sorted(cycle_classes.items())),
                "module_cycles": dict(sorted(module_cycles.items())),
                "module_expanded_request_count": dict(sorted(
                    module_requests.items())),
                "ordered_address_hashes_sha256": canonical_sha256(address_hashes),
                "ordered_commit_hashes_sha256": canonical_sha256(commit_hashes),
                "resource_manifest_sha256": resource.identity()[
                    "resource_manifest_sha256"],
            }
        aggregate[population_id] = {
            "records": expected_records,
            "samples": expected_records // 4,
            "configs": per_config,
            "typed_k8_vs_equal_service_k1x8_headline":
                headline_ratio(per_config),
            "headline_ratio_source_field": "headline_total_cycles",
            "all_module_total_cycles_charged": True,
            "k8_vs_a1_headline_allowed": False,
            "d1_headline_eligible": False,
        }

    require(len(detailed) == (40 + 120) * 10 * len(CONFIGS),
            "detailed row count drift")
    result = {
        "schema": "m809_m785_decoder_physical_residency_production_result_v1",
        "date": "2026-08-29",
        "status": "PRODUCTION_REPLAY_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED",
        "source_identity": candidate["source_identity"],
        "candidate_sha256": sha256(candidate_path),
        "release_sha256": sha256(release_path),
        "attempt_outer_seal_file_sha256":
            attempt_identity["outer_seal_file_sha256"],
        "resource": resource.identity(),
        "execution_semantics": {
            "populations_separate": True,
            "cross_record_overlap_credited": False,
            "cross_population_overlap_credited": False,
            "record_timestep_cold_start_and_drain_charged": True,
            "configs": list(CONFIGS),
            "only_legal_headline_ratio":
                "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8",
            "headline_ratio_source_field": "headline_total_cycles",
            "headline_excluded_module_indices": [D1_MODULE_INDEX],
            "all_module_total_cycles_retained": True,
        },
        "populations": aggregate,
        "detailed_rows": len(detailed),
        "claim_boundary": {
            "production_cycles_generated": True,
            "production_speedup_admitted_before_result_hammer": False,
            "d1_headline": False,
            "decoder_complete": False,
            "full_network_completion": False,
            "table_a_insertion_allowed": False,
            "system_speedup": False,
            "rtl_vcs_eda_energy_ppa": False,
        },
    }
    _write_json(output / "result.json", result)
    _write_json(output / "detailed_rows.json", detailed)
    return result


def _rename_noreplace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    require(source.parent.resolve() == destination.parent.resolve(),
            "atomic publication requires sibling paths")
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "libc renameat2 is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    rc = function(AT_FDCWD, os.fsencode(source),
                  AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE)
    if rc != 0:
        observed = ctypes.get_errno()
        if observed == errno.EEXIST:
            raise Failure("atomic no-replace destination collision")
        raise Failure("renameat2(RENAME_NOREPLACE) failed errno=" +
                      str(observed))


def publish_no_replace(candidate_path: Path, stage: Path,
                       destination: Path) -> Dict[str, object]:
    candidate = strict_json(candidate_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "publication candidate schema drift")
    paths = _candidate_paths(candidate)
    regular_exact(paths["driver"],
                  candidate["source_identity"]["driver"]["sha256"],
                  "publication driver")
    regular_exact(paths["runner"],
                  candidate["source_identity"]["runner"]["sha256"],
                  "publication runner")
    canonical = (REPO / candidate["canonical"]["result"]).resolve()
    stage = Path(stage).resolve()
    destination = Path(destination).resolve()
    require(destination == canonical, "publication destination drift")
    require(stage.parent == canonical.parent and
            stage.name.startswith(canonical.name + ".stage."),
            "publication stage path drift")
    identity = verify_sealed(stage)
    members = {entry.name for entry in stage.iterdir()}
    require(members == {
        "result.json", "detailed_rows.json", "SHA256SUMS",
        "SHA256SUMS.seal.sha256",
    }, "publication stage/root member set drift")
    _rename_noreplace(stage, destination)
    require(destination.is_dir() and not destination.is_symlink() and
            not stage.exists() and not stage.is_symlink(),
            "atomic publication state transition failed")
    for name in ("result.json", "detailed_rows.json", "SHA256SUMS",
                 "SHA256SUMS.seal.sha256"):
        require((destination / name).is_file() and
                not (destination / name).is_symlink(),
                "canonical root member absent after publication: " + name)
    final_identity = verify_sealed(destination)
    require(final_identity == identity,
            "publication changed sealed result identity")
    return {
        "status": "PASS_M809_ATOMIC_NOREPLACE_CANONICAL_PUBLICATION",
        "manifest_sha256": identity["manifest_sha256"],
        "outer_seal_file_sha256": identity["outer_seal_file_sha256"],
    }


def seal_exact_members(directory: Path,
                       member_names: Sequence[str]) -> Dict[str, str]:
    """Create a flat root double seal and reject any unlisted member."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "seal target must be a regular directory")
    require(len(member_names) == len(set(member_names)) and member_names,
            "seal member list must be unique and nonempty")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(not manifest.exists() and not manifest.is_symlink() and
            not outer.exists() and not outer.is_symlink(),
            "seal outputs already exist")
    lines = []
    for name in member_names:
        require("/" not in name and name not in ("SHA256SUMS",
                                                  "SHA256SUMS.seal.sha256"),
                "flat seal requires root data members")
        member = directory / name
        require(member.is_file() and not member.is_symlink(),
                "flat seal member is absent or nonregular: " + name)
        lines.append(sha256(member) + "  " + name + "\n")
    with manifest.open("x", encoding="utf-8") as handle:
        handle.writelines(lines)
    with outer.open("x", encoding="utf-8") as handle:
        handle.write(sha256(manifest) + "  SHA256SUMS\n")
    members = {entry.name for entry in directory.iterdir()}
    require(members == set(member_names) | {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"},
        "flat sealed root member set drift")
    return verify_sealed(directory)


def write_flat_attempt(directory: Path, receipt: Mapping[str, object]) \
        -> Dict[str, str]:
    """Source-testable constructor for the M809 flat one-shot population."""
    directory = Path(directory)
    require(not directory.exists() and not directory.is_symlink(),
            "flat attempt destination already exists")
    directory.mkdir(mode=0o700)
    _write_json(directory / "attempt.json", dict(receipt))
    return seal_exact_members(directory, ("attempt.json",))


def _write_failure_receipt(output: Path, stdout_path: Path,
                           stderr_path: Path,
                           payload: Mapping[str, object]) -> Dict[str, str]:
    """Atomically shape a four-member, source-testable failure quarantine."""
    output = Path(output)
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    require(not output.exists() and not output.is_symlink(),
            "failure quarantine destination collision")
    require(stdout_path.is_file() and not stdout_path.is_symlink() and
            stderr_path.is_file() and not stderr_path.is_symlink(),
            "driver logs must be regular nonsymlink files")
    output.mkdir(mode=0o700)
    log_path = output / "driver.log"
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("===== STDOUT =====\n")
        handle.write(stdout_path.read_text(encoding="utf-8", errors="replace"))
        handle.write("\n===== STDERR =====\n")
        handle.write(stderr_path.read_text(encoding="utf-8", errors="replace"))
        handle.write("\n")
    failure = dict(payload)
    failure["driver_log_sha256"] = sha256(log_path)
    _write_json(output / "failure.json", failure)
    identity = seal_exact_members(output, ("failure.json", "driver.log"))
    require({entry.name for entry in output.iterdir()} == {
        "failure.json", "driver.log", "SHA256SUMS",
        "SHA256SUMS.seal.sha256"},
        "failure quarantine root must have exactly four members")
    return identity


def publish_failure_receipt(candidate_path: Path, release_path: Path,
                            attempt_path: Path, runner_path: Path,
                            stdout_path: Path, stderr_path: Path,
                            output: Path, expected_runner_sha256: str,
                            expected_release_sha256: str, return_code: int,
                            phase: str, partial_artifact: str) \
        -> Dict[str, object]:
    """Publish a sealed receipt after consumption without running schedules."""
    require(return_code != 0, "failure receipt refuses zero return code")
    require(phase and len(expected_runner_sha256) == 64 and
            len(expected_release_sha256) == 64,
            "failure receipt identity arguments are malformed")
    candidate_path = Path(candidate_path).resolve()
    release_path = Path(release_path).resolve()
    attempt_path = Path(attempt_path).resolve()
    runner_path = Path(runner_path).resolve()
    output = Path(output).resolve()
    candidate = strict_json(candidate_path)
    release = strict_json(release_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA and
            release.get("schema") == RELEASE_SCHEMA,
            "failure receipt candidate/release schema drift")
    paths = _candidate_paths(candidate)
    regular_exact(paths["driver"],
                  candidate["source_identity"]["driver"]["sha256"],
                  "failure receipt driver")
    regular_exact(runner_path, expected_runner_sha256,
                  "failure receipt runner")
    require(paths["runner"].resolve() == runner_path and
            candidate["source_identity"]["runner"]["sha256"] ==
            expected_runner_sha256,
            "failure receipt runner binding drift")
    regular_exact(release_path, expected_release_sha256,
                  "failure receipt release")
    require(release.get("candidate_binding", {}).get("sha256") ==
            sha256(candidate_path) and
            release.get("source_identity") == candidate.get("source_identity"),
            "failure receipt release binding drift")
    canonical_attempt = (REPO / candidate["canonical"]["attempt"]).resolve()
    canonical_result = (REPO / candidate["canonical"]["result"]).resolve()
    require(attempt_path == canonical_attempt,
            "failure receipt attempt path drift")
    attempt_identity = verify_sealed(attempt_path)
    attempt_receipt = strict_json(attempt_path / "attempt.json")
    require(attempt_receipt.get("candidate_sha256") == sha256(candidate_path) and
            attempt_receipt.get("driver_sha256") ==
            candidate["source_identity"]["driver"]["sha256"] and
            attempt_receipt.get("runner_sha256") == expected_runner_sha256 and
            attempt_receipt.get("release_sha256") == expected_release_sha256,
            "failure receipt attempt identity drift")
    require(output.parent == canonical_result.parent and
            output.name.startswith(canonical_result.name +
                                   ".failed_or_incomplete."),
            "failure quarantine canonical sibling/prefix drift")
    require(not output.exists() and not output.is_symlink(),
            "failure quarantine destination collision")
    payload = {
        "schema": "m809_m785_decoder_production_failure_receipt_v1",
        "date": "2026-08-29",
        "status": "FAILED_OR_INCOMPLETE__NO_CYCLES_CITABLE",
        "return_code": int(return_code),
        "phase": phase,
        "partial_artifact": partial_artifact,
        "candidate_sha256": sha256(candidate_path),
        "driver_sha256": candidate["source_identity"]["driver"]["sha256"],
        "runner_sha256": expected_runner_sha256,
        "release_sha256": expected_release_sha256,
        "attempt_json_sha256": sha256(attempt_path / "attempt.json"),
        "attempt_manifest_sha256": attempt_identity["manifest_sha256"],
        "attempt_outer_seal_file_sha256":
            attempt_identity["outer_seal_file_sha256"],
        "claim_boundary": {
            "production_complete": False,
            "cycles_citable": False,
            "speedup_citable": False,
            "decoder_complete": False,
            "table_a_insertion_allowed": False,
        },
    }
    identity = _write_failure_receipt(
        output, stdout_path, stderr_path, payload)
    return {
        "status": "PASS_M809_SEALED_FAILURE_RECEIPT",
        "manifest_sha256": identity["manifest_sha256"],
        "outer_seal_file_sha256": identity["outer_seal_file_sha256"],
    }


def self_test() -> Dict[str, object]:
    parent = M793.self_test()
    require(parent["status"] ==
            "PASS_M793_PRODUCTION_DRIVER_SYNTHETIC_SELF_TEST",
            "M793 parent self-test drift")

    duplicate_rejected = False
    with tempfile.TemporaryDirectory(prefix="m809_json_") as directory:
        path = Path(directory) / "duplicate.json"
        path.write_text('{"launch_now":true,"launch_now":false}\n',
                        encoding="utf-8")
        try:
            strict_json(path)
        except Failure:
            duplicate_rejected = True
    require(duplicate_rejected, "duplicate JSON key was accepted")

    flat_attempt_passed = False
    old_hierarchy_precise_failure = False
    with tempfile.TemporaryDirectory(prefix="m809_attempt_") as directory:
        root = Path(directory)
        flat = root / "flat"
        write_flat_attempt(flat, {
            "status": "SYNTHETIC_SOURCE_ONLY_ATTEMPT",
            "production_replay": False,
        })
        flat_attempt_passed = verify_sealed(flat)["manifest_sha256"] == \
            sha256(flat / "SHA256SUMS")

        old = root / "old_hierarchical"
        initial = old / "initial"
        initial.mkdir(parents=True)
        _write_json(initial / "attempt.json", {"synthetic": True})
        seal_exact_members(initial, ("attempt.json",))
        with (old / "SHA256SUMS").open("x", encoding="utf-8") as handle:
            handle.write(sha256(initial / "SHA256SUMS.seal.sha256") +
                         "  initial/SHA256SUMS.seal.sha256\n")
        with (old / "SHA256SUMS.seal.sha256").open(
                "x", encoding="utf-8") as handle:
            handle.write(sha256(old / "SHA256SUMS") +
                         "  SHA256SUMS\n")
        try:
            verify_sealed(old)
        except Failure as error:
            old_hierarchy_precise_failure = \
                str(error) == "sealed population mismatch"
    require(flat_attempt_passed,
            "flat attempt population did not pass generic verifier")
    require(old_hierarchy_precise_failure,
            "old hierarchical attempt did not fail precisely")

    failure_receipt_passed = False
    failure_collision_rejected = False
    with tempfile.TemporaryDirectory(prefix="m809_failure_") as directory:
        root = Path(directory)
        stdout = root / "stdout"
        stderr = root / "stderr"
        stdout.write_text("driver source-only stdout\n", encoding="utf-8")
        stderr.write_text("synthetic pre-stage failure\n", encoding="utf-8")
        quarantine = root / "result.failed_or_incomplete.synthetic"
        _write_failure_receipt(quarantine, stdout, stderr, {
            "schema": "m809_synthetic_failure_receipt_v1",
            "status": "FAILED_BEFORE_STAGE__SOURCE_TEST_ONLY",
            "return_code": 17,
            "production_replay": False,
        })
        failure_receipt_passed = (
            {entry.name for entry in quarantine.iterdir()} == {
                "failure.json", "driver.log", "SHA256SUMS",
                "SHA256SUMS.seal.sha256"} and
            verify_sealed(quarantine)["manifest_sha256"] ==
            sha256(quarantine / "SHA256SUMS"))
        before = {entry.name: sha256(entry) for entry in quarantine.iterdir()
                  if entry.is_file()}
        try:
            _write_failure_receipt(quarantine, stdout, stderr,
                                   {"status": "COLLISION"})
        except Failure:
            after = {entry.name: sha256(entry) for entry in
                     quarantine.iterdir() if entry.is_file()}
            failure_collision_rejected = before == after
    require(failure_receipt_passed,
            "pre-stage failure receipt was not exactly sealed")
    require(failure_collision_rejected,
            "failure quarantine collision modified existing evidence")

    base = {
        HEADLINE_DENOMINATOR: {
            "headline_total_cycles": 120,
            "total_cycles": 1120,
        },
        HEADLINE_NUMERATOR: {
            "headline_total_cycles": 80,
            "total_cycles": 2080,
        },
    }
    first = headline_ratio(base)
    base[HEADLINE_DENOMINATOR]["total_cycles"] += 1000000
    base[HEADLINE_NUMERATOR]["total_cycles"] += 7
    second = headline_ratio(base)
    require(first == 1.5 and second == first,
            "D1 diagnostic perturbation changed headline ratio")

    collision_rejected = False
    with tempfile.TemporaryDirectory(prefix="m809_publish_") as directory:
        parent_dir = Path(directory)
        stage = parent_dir / "result.stage.attack"
        destination = parent_dir / "result"
        stage.mkdir()
        destination.mkdir()
        try:
            _rename_noreplace(stage, destination)
        except Failure:
            collision_rejected = True
        require(collision_rejected and stage.is_dir() and destination.is_dir(),
                "destination race did not fail without moving stage")
        destination.rmdir()
        _rename_noreplace(stage, destination)
        require(destination.is_dir() and not stage.exists(),
                "no-replace success path failed")
    return {
        "status": "PASS_M809_REPAIRED_DRIVER_SYNTHETIC_SELF_TEST",
        "parent": parent["status"],
        "duplicate_json_rejected": True,
        "flat_attempt_population_passed": True,
        "old_hierarchical_attempt_precise_failure": True,
        "prestage_failure_receipt_four_member_sealed": True,
        "failure_quarantine_collision_rejected": True,
        "d1_headline_perturbation_invariant": True,
        "atomic_destination_race_rejected": True,
        "production_replay": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--validate-release-preflight", action="store_true")
    parser.add_argument("--validate-consumed-attempt", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--write-failure-receipt", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-release-sha256")
    parser.add_argument("--return-code", type=int)
    parser.add_argument("--phase")
    parser.add_argument("--partial-artifact", default="")
    args = parser.parse_args(argv)
    selected = sum(bool(value) for value in (
        args.self_test, args.validate_candidate,
        args.validate_release_preflight, args.validate_consumed_attempt,
        args.run_production,
        args.publish_no_replace, args.write_failure_receipt))
    require(selected == 1, "select exactly one driver mode")
    if args.self_test:
        require(not any((args.candidate, args.release, args.attempt,
                         args.output, args.publish_to, args.runner,
                         args.stdout_log, args.stderr_log)),
                "self-test refuses production paths")
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return 0
    require(args.candidate is not None, "candidate path is required")
    if args.validate_candidate:
        require(not any((args.release, args.attempt, args.output,
                         args.publish_to)),
                "source validation refuses production paths")
        print(json.dumps(validate_candidate(args.candidate), indent=2,
                         sort_keys=True, allow_nan=False))
        return 0
    if args.validate_release_preflight:
        require(args.release is not None and
                not any((args.attempt, args.output, args.publish_to)),
                "release preflight needs only candidate and release")
        value = validate_true_release(
            args.release, args.candidate, attempt_required=False)
        print(json.dumps({
            "status": "PASS_M809_TRUE_RELEASE_PREFLIGHT__ONE_SHOT_UNCONSUMED",
            "release_sha256": sha256(args.release),
            "candidate_sha256": value["candidate_validation"][
                "candidate_sha256"],
            "production_replay": False,
            "production_cycles": None,
        }, sort_keys=True, allow_nan=False))
        return 0
    if args.validate_consumed_attempt:
        require(args.release is not None and args.attempt is not None and
                not any((args.output, args.publish_to, args.runner,
                         args.stdout_log, args.stderr_log)),
                "consumed-attempt preflight needs candidate/release/attempt")
        print(json.dumps(validate_consumed_attempt(
            args.release, args.candidate, args.attempt),
            sort_keys=True, allow_nan=False))
        return 0
    if args.publish_no_replace:
        require(args.output is not None and args.publish_to is not None and
                not any((args.release, args.attempt)),
                "publication needs candidate, stage and destination only")
        print(json.dumps(publish_no_replace(
            args.candidate, args.output, args.publish_to),
            sort_keys=True, allow_nan=False))
        return 0
    if args.write_failure_receipt:
        require(all((args.release, args.attempt, args.output, args.runner,
                     args.stdout_log, args.stderr_log,
                     args.expected_runner_sha256,
                     args.expected_release_sha256,
                     args.return_code is not None, args.phase)) and
                args.publish_to is None,
                "failure receipt requires exact post-consumption inputs")
        print(json.dumps(publish_failure_receipt(
            args.candidate, args.release, args.attempt, args.runner,
            args.stdout_log, args.stderr_log, args.output,
            args.expected_runner_sha256, args.expected_release_sha256,
            args.return_code, args.phase, args.partial_artifact),
            sort_keys=True, allow_nan=False))
        return 0
    require(all((args.release, args.attempt, args.output)) and
            args.publish_to is None,
            "production requires release, attempt and staging output")
    value = run_production(args.release, args.candidate,
                           args.attempt, args.output)
    print(json.dumps({
        "status": value["status"],
        "result_sha256": sha256(Path(args.output) / "result.json"),
        "detailed_rows": value["detailed_rows"],
        "fresh_result_hammer_required": True,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exact one-shot production driver for the frozen M785 decoder model.

This additive driver does not alter the M785 analyzer.  Source validation and
the synthetic self-test are always available.  A production replay is accepted
only with a separately authored, exact-SHA true release and an already sealed
one-shot attempt receipt.  Primary and secondary populations are scheduled
independently and sequentially; no cross-record or cross-population overlap is
credited.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
M785_PATH = HERE / "analyze_m785_h67_decoder_physical_residency_repair.py"
M785_SHA256 = "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1"
CANDIDATE_SCHEMA = "m793_m785_decoder_production_release_candidate_v1"
RELEASE_SCHEMA = "m793_m785_decoder_production_true_release_v1"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CONFIGS = ("A1_OSG", "EQUAL_SERVICE_K1X8", "TYPED_SIGNED_K8")
HEADLINE_NUMERATOR = "TYPED_SIGNED_K8"
HEADLINE_DENOMINATOR = "EQUAL_SERVICE_K1X8"


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


def strict_json(path: Path) -> object:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            "JSON input must be a regular nonsymlink file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            Failure("non-finite JSON constant: " + value)))


def regular_exact(path: Path, expected: str, label: str) -> None:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            label + " is not a regular nonsymlink file")
    require(sha256(path) == expected, label + " SHA drift")


def load_m785():
    regular_exact(M785_PATH, M785_SHA256, "M785 analyzer")
    spec = importlib.util.spec_from_file_location("m793_frozen_m785", M785_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M785 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M785 = load_m785()


def verify_sealed(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory is not a regular directory: " + str(directory))
    return dict(M785.verify_sealed_directory(directory))


def _candidate_paths(candidate: Mapping[str, object]) -> Dict[str, Path]:
    rows = {}
    for name, entry in candidate["source_identity"].items():
        rows[name] = HW / entry["path"]
    return rows


def validate_candidate(candidate_path: Path) -> Dict[str, object]:
    candidate_path = Path(candidate_path).resolve()
    candidate = strict_json(candidate_path)
    require(isinstance(candidate, dict), "candidate must be an object")
    require(candidate.get("schema") == CANDIDATE_SCHEMA,
            "candidate schema drift")
    require(candidate.get("status") ==
            "SOURCE_ONLY_PRODUCTION_DRIVER_CANDIDATE__FRESH_HAMMER_REQUIRED",
            "candidate status drift")
    require(candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "source candidate must not authorize production")
    authorization = candidate.get("authorization", {})
    require(authorization == {
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
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    require(paths["driver"].resolve() == Path(__file__).resolve(),
            "candidate does not bind this exact driver")
    require(paths["analyzer"].resolve() == M785_PATH,
            "candidate analyzer path drift")

    source_contract = paths["m785_contract"]
    validation = M785.validate_source_contract(REPO, source_contract)
    require(validation["status"] ==
            "PASS_M785_SOURCE_IDENTITY_ONLY__NO_PRODUCTION_RUN",
            "M785 source validation drift")

    m790 = candidate["m790_pass100"]
    m790_dir = HW / m790["directory"]
    m790_identity = verify_sealed(m790_dir)
    regular_exact(m790_dir / "review.json", m790["review_json_sha256"],
                  "M790 review")
    require(m790_identity["manifest_sha256"] == m790["manifest_sha256"] and
            m790_identity["outer_seal_file_sha256"] ==
            m790["outer_seal_file_sha256"], "M790 double seal drift")
    review = strict_json(m790_dir / "review.json")
    require(review.get("status") ==
            "PASS100_M785_SOURCE_ONLY__SEPARATE_ADDITIVE_PRODUCTION_RELEASE_REQUIRED" and
            review.get("score") == 100 and
            review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
            "M790 PASS100 semantics drift")

    request = candidate["m786_source_request"]
    request_dir = HW / request["directory"]
    request_identity = verify_sealed(request_dir)
    regular_exact(request_dir / "request.json", request["request_json_sha256"],
                  "M786 request JSON")
    require(request_identity["manifest_sha256"] == request["manifest_sha256"] and
            request_identity["outer_seal_file_sha256"] ==
            request["outer_seal_file_sha256"], "M786 request double seal drift")

    contract = strict_json(source_contract)
    records = {}
    for name, population_id, expected in (
        ("primary_m686", "M686_ZURICH_CITY_09_A_S10", 40),
        ("secondary_m699", "M699_DSEC_S3X10", 120),
    ):
        row = contract["inputs"][name]
        directory = HW / row["directory"]
        identity = verify_sealed(directory)
        require(identity["outer_seal_file_sha256"] ==
                row["outer_seal_file_sha256"], name + " outer seal drift")
        manifest = strict_json(directory / "manifest.json")
        require(sha256(directory / "manifest.json") == row["manifest_sha256"],
                name + " manifest drift")
        normalized = M785.normalized_population_records(manifest, population_id)
        require(len(normalized) == expected, name + " record count drift")
        records[name] = len(normalized)

    production_inputs = candidate["production_inputs"]
    for candidate_name, contract_name, receipt_name, expected in (
        ("primary_m686", "primary_m686", None, 40),
        ("secondary_m699", "secondary_m699", None, 120),
        ("primary_m692_review", "primary_m692_review", "review_json_sha256", None),
        ("secondary_m705_review", "secondary_m705_review", "review_json_sha256", None),
    ):
        source = contract["inputs"][contract_name]
        frozen = production_inputs[candidate_name]
        require(frozen["directory"] == source["directory"] and
                frozen["outer_seal_file_sha256"] ==
                source["outer_seal_file_sha256"],
                candidate_name + " candidate/contract identity drift")
        if expected is not None:
            require(frozen["records"] == expected and
                    frozen["manifest_sha256"] == source["manifest_sha256"],
                    candidate_name + " record/manifest drift")
        else:
            require(frozen[receipt_name] == source[receipt_name],
                    candidate_name + " review identity drift")

    require(candidate["python_identity"] == {
        "path": "/opt/anaconda3/envs/pytorch310/bin/python3.10",
        "sha256": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    }, "candidate Python identity drift")
    regular_exact(Path(candidate["python_identity"]["path"]),
                  candidate["python_identity"]["sha256"], "M793 Python")
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
    require(validation["resource"]["resource_manifest_sha256"] ==
            candidate["common_resource"]["resource_manifest_sha256"],
            "candidate/source resource hash drift")
    require(candidate["production_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configurations": list(CONFIGS),
        "schedule":
            "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "cold_start_and_drain_per_record_timestep": True,
        "only_legal_headline_ratio":
            "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8",
        "k8_vs_a1_headline_allowed": False,
        "d1": "COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE",
        "external_opportunity_artifact_candidate_input": False,
    }, "candidate production semantics drift")

    canonical = candidate["canonical"]
    require(canonical == {
        "result":
            "hw_autoresearch_nts07/results/m793_m785_h67_decoder_physical_residency_cycles_r1_20260828",
        "attempt":
            "hw_autoresearch_nts07/results/.m793_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed",
        "future_release":
            "hw_autoresearch_nts07/contracts/m793_m785_decoder_physical_residency_production_true_release_r1_20260828.json",
        "failed_or_incomplete_prefix":
            "hw_autoresearch_nts07/results/m793_m785_h67_decoder_physical_residency_cycles_r1_20260828.failed_or_incomplete.",
    }, "candidate canonical-path drift")
    for key in ("result", "attempt", "future_release"):
        path = REPO / canonical[key]
        require(not path.exists() and not path.is_symlink(),
                key + " must remain absent before reviewed launch")
    boundary = candidate["claim_boundary"]
    require(boundary == {
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
        "status": "PASS_M793_SOURCE_CANDIDATE_IDENTITY__NO_PRODUCTION_RUN",
        "candidate_sha256": sha256(candidate_path),
        "m785_source_validation": validation["status"],
        "m790_score": 100,
        "records": records,
        "resource_manifest_sha256": validation["resource"][
            "resource_manifest_sha256"],
        "launch_now": False,
        "production_cycles": None,
        "production_speedup": None,
        "decoder_complete": False,
        "full_network_completion": False,
        "table_a_insertion_allowed": False,
    }


def validate_true_release(release_path: Path, candidate_path: Path,
                          attempt_required: bool = True) -> Dict[str, object]:
    release_path = Path(release_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    candidate_validation = validate_candidate_for_release(
        candidate_path, attempt_required=attempt_required)
    release = strict_json(release_path)
    require(isinstance(release, dict) and release.get("schema") == RELEASE_SCHEMA,
            "true release schema drift")
    require(release.get("status") ==
            "TRUE_RELEASE_AFTER_FRESH_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_REPLAY" and
            release.get("launch_now") is True and
            release.get("release") is True and release.get("max_attempts") == 1,
            "true release authorization drift")
    binding = release["candidate_binding"]
    require((HW / binding["path"]).resolve() == candidate_path and
            sha256(candidate_path) == binding["sha256"],
            "release candidate binding drift")
    for name in ("driver", "runner", "analyzer", "storage_oracle", "tests",
                 "m785_contract"):
        expected = release["source_identity"][name]
        candidate_entry = strict_json(candidate_path)["source_identity"][name]
        require(expected == candidate_entry,
                "release changed candidate source identity: " + name)
    require(release["canonical"] == strict_json(candidate_path)["canonical"],
            "release changed canonical paths")
    hammer = release["fresh_source_hammer"]
    require(hammer["directory"] ==
            "reviews/m795_m793_m785_decoder_production_source_fresh_hammer_r1_20260828",
            "release source-hammer directory drift")
    hammer_dir = HW / hammer["directory"]
    hammer_identity = verify_sealed(hammer_dir)
    regular_exact(hammer_dir / "review.json", hammer["review_json_sha256"],
                  "M795 source-hammer review")
    require(hammer_identity["manifest_sha256"] == hammer["manifest_sha256"] and
            hammer_identity["outer_seal_file_sha256"] ==
            hammer["outer_seal_file_sha256"],
            "M795 source-hammer double seal drift")
    hammer_review = strict_json(hammer_dir / "review.json")
    require(hammer_review.get("status") ==
            "PASS100_M793_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY" and
            hammer_review.get("score") == 100 and
            hammer_review.get("severity_counts") ==
            {"p0": 0, "p1": 0, "p2": 0},
            "M795 source-hammer PASS100 semantics drift")
    require(release["runtime_semantics"] == {
        "populations": "M686_40_AND_M699_120_SEPARATE",
        "configs": list(CONFIGS),
        "schedule": "RECORD_TIMESTEP_SEQUENTIAL_NO_CROSS_RECORD_OR_POPULATION_OVERLAP",
        "resource": "96_LANES_245760B_ACC24_3NS_192B_PER_CYCLE",
        "headline_ratio": "TYPED_SIGNED_K8_VS_EQUAL_SERVICE_K1X8_ONLY",
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
    expected_sha = os.environ.get("M793_EXPECTED_RELEASE_SHA256", "")
    require(len(expected_sha) == 64 and sha256(release_path) == expected_sha,
            "caller did not supply exact reviewed true-release SHA")
    release_sidecar = Path(str(release_path) + ".sha256")
    release_outer = Path(str(release_sidecar) + ".seal.sha256")
    regular_exact(release_sidecar, sha256(release_sidecar),
                  "true-release member sidecar")
    regular_exact(release_outer, sha256(release_outer),
                  "true-release outer sidecar")
    require(release_sidecar.read_text(encoding="utf-8") ==
            expected_sha + "  " + release_path.name + "\n",
            "true-release member sidecar content drift")
    require(release_outer.read_text(encoding="utf-8") ==
            sha256(release_sidecar) + "  " + release_sidecar.name + "\n",
            "true-release outer sidecar content drift")
    return {"release": release, "candidate": strict_json(candidate_path),
            "candidate_validation": candidate_validation}


def validate_candidate_for_release(candidate_path: Path,
                                   attempt_required: bool) -> Dict[str, object]:
    """Validate frozen candidate after future release exists.

    The pre-release validator requires future_release absent.  A true release
    must not weaken any other gate, so temporarily check the same identity with
    only that one expected state transition admitted.
    """
    candidate = strict_json(candidate_path)
    future = REPO / candidate["canonical"]["future_release"]
    require(future.is_file() and not future.is_symlink(),
            "canonical true release is absent or not regular")
    # Recheck all immutable fields directly.  Temporarily moving or hiding the
    # release would create a TOCTOU window, so this is deliberately explicit.
    require(candidate.get("schema") == CANDIDATE_SCHEMA and
            candidate.get("launch_now") is False and
            candidate.get("release") is False and
            candidate.get("max_attempts") == 0,
            "candidate immutable authorization drift")
    paths = _candidate_paths(candidate)
    for name, entry in candidate["source_identity"].items():
        regular_exact(paths[name], entry["sha256"], name)
    source_validation = M785.validate_source_contract(
        REPO, paths["m785_contract"])
    require(source_validation["status"] ==
            "PASS_M785_SOURCE_IDENTITY_ONLY__NO_PRODUCTION_RUN",
            "runtime M785 source/input identity drift")
    m790 = candidate["m790_pass100"]
    m790_dir = HW / m790["directory"]
    m790_identity = verify_sealed(m790_dir)
    regular_exact(m790_dir / "review.json", m790["review_json_sha256"],
                  "runtime M790 review")
    require(m790_identity["manifest_sha256"] == m790["manifest_sha256"] and
            m790_identity["outer_seal_file_sha256"] ==
            m790["outer_seal_file_sha256"], "runtime M790 seal drift")
    m786 = candidate["m786_source_request"]
    m786_dir = HW / m786["directory"]
    m786_identity = verify_sealed(m786_dir)
    regular_exact(m786_dir / "request.json", m786["request_json_sha256"],
                  "runtime M786 request")
    require(m786_identity["manifest_sha256"] == m786["manifest_sha256"] and
            m786_identity["outer_seal_file_sha256"] ==
            m786["outer_seal_file_sha256"], "runtime M786 seal drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    result = REPO / candidate["canonical"]["result"]
    attempt = REPO / candidate["canonical"]["attempt"]
    require(not result.exists() and not result.is_symlink(),
            "canonical result already exists")
    if attempt_required:
        require(attempt.is_dir() and not attempt.is_symlink(),
                "attempt must already be consumed before production")
        verify_sealed(attempt)
    else:
        require(not attempt.exists() and not attempt.is_symlink(),
                "preflight requires unconsumed one-shot")
    return {
        "status": "PASS_M793_FROZEN_CANDIDATE_WITH_TRUE_RELEASE_PRESENT",
        "candidate_sha256": sha256(candidate_path),
    }


def _sum_maps(target: Dict[str, int], source: Mapping[str, object]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


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
    receipt = strict_json(Path(attempt_path) / "initial/attempt.json")
    require(receipt.get("status") ==
            "CONSUMED_IMMEDIATELY_BEFORE_M793_PRODUCTION_REPLAY" and
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
        HW / mapper_row["path"], mapper_row["sha256"], "m793_m672_mapper")
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
        require(len(records) == expected_records, "production record count drift")
        per_config = {}
        for config in CONFIGS:
            total_cycles = 0
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
                    row = {
                        "population_id": population_id,
                        "sequence": record["sequence"],
                        "sample_id": int(record["sample_id"]),
                        "module_index": module_index,
                        "timestep": timestep,
                        "config": config,
                        "headline_eligible": module_index != 1,
                        "total_cycles": int(schedule["total_cycles"]),
                        "expanded_request_count": int(
                            schedule["expanded_request_count"]),
                        "compressed_transaction_count": int(
                            schedule["compressed_transaction_count"]),
                        "cycle_classes": {key: int(value) for key, value in
                                          schedule["cycle_classes"].items()},
                        "transaction_address_sha256":
                            schedule["transaction_address_sha256"],
                        "commit_sequence_sha256":
                            schedule["commit_sequence_sha256"],
                    }
                    detailed.append(row)
                    total_cycles += row["total_cycles"]
                    requests += row["expanded_request_count"]
                    transactions += row["compressed_transaction_count"]
                    _sum_maps(cycle_classes, row["cycle_classes"])
                    module_cycles[str(module_index)] += row["total_cycles"]
                    module_requests[str(module_index)] += row[
                        "expanded_request_count"]
                    address_hashes.append(row["transaction_address_sha256"])
                    commit_hashes.append(row["commit_sequence_sha256"])
                    del schedule
            per_config[config] = {
                "total_cycles": total_cycles,
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
        require(per_config[HEADLINE_DENOMINATOR]["total_cycles"] > 0 and
                per_config[HEADLINE_NUMERATOR]["total_cycles"] > 0,
                "zero headline cycles")
        aggregate[population_id] = {
            "records": expected_records,
            "samples": expected_records // 4,
            "configs": per_config,
            "typed_k8_vs_equal_service_k1x8":
                per_config[HEADLINE_DENOMINATOR]["total_cycles"] /
                per_config[HEADLINE_NUMERATOR]["total_cycles"],
            "k8_vs_a1_headline_allowed": False,
            "d1_headline_eligible": False,
        }

    require(len(detailed) == (40 + 120) * 10 * len(CONFIGS),
            "detailed row count drift")
    result = {
        "schema": "m793_m785_decoder_physical_residency_production_result_v1",
        "date": "2026-08-28",
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


def self_test() -> Dict[str, object]:
    value = M785.synthetic_self_test()
    require(value["status"] == "PASS_M785_SYNTHETIC_SOURCE_SELF_TEST",
            "M785 synthetic self-test drift")
    sample = defaultdict(int)
    _sum_maps(sample, {"a": 1, "b": 2})
    _sum_maps(sample, {"a": 3})
    require(dict(sample) == {"a": 4, "b": 2}, "aggregation self-test")
    require(M785.headline_ratio_allowed(HEADLINE_NUMERATOR,
                                       HEADLINE_DENOMINATOR),
            "headline pair drift")
    require(not M785.headline_ratio_allowed(HEADLINE_NUMERATOR, "A1_OSG"),
            "illegal K8/A1 headline admitted")
    return {
        "status": "PASS_M793_PRODUCTION_DRIVER_SYNTHETIC_SELF_TEST",
        "m785": value["status"],
        "production_replay": False,
        "production_cycles": None,
        "production_speedup": None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--validate-release-preflight", action="store_true")
    parser.add_argument("--run-production", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    selected = sum(bool(value) for value in (
        args.self_test, args.validate_candidate,
        args.validate_release_preflight, args.run_production))
    require(selected == 1, "select exactly one driver mode")
    if args.self_test:
        require(not any((args.candidate, args.release, args.attempt, args.output)),
                "self-test refuses production paths")
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return 0
    require(args.candidate is not None, "candidate path is required")
    if args.validate_candidate:
        require(not any((args.release, args.attempt, args.output)),
                "source validation refuses production paths")
        print(json.dumps(validate_candidate(args.candidate), indent=2,
                         sort_keys=True, allow_nan=False))
        return 0
    if args.validate_release_preflight:
        require(args.release is not None and
                not any((args.attempt, args.output)),
                "release preflight needs only candidate and release")
        value = validate_true_release(
            args.release, args.candidate, attempt_required=False)
        print(json.dumps({
            "status": "PASS_M793_TRUE_RELEASE_PREFLIGHT__ONE_SHOT_UNCONSUMED",
            "release_sha256": sha256(args.release),
            "candidate_sha256": value["candidate_validation"][
                "candidate_sha256"],
            "production_replay": False,
            "production_cycles": None,
        }, sort_keys=True, allow_nan=False))
        return 0
    require(all((args.release, args.attempt, args.output)),
            "production requires release, attempt and staging output")
    value = run_production(args.release, args.candidate, args.attempt, args.output)
    print(json.dumps({
        "status": value["status"],
        "result_sha256": sha256(Path(args.output) / "result.json"),
        "detailed_rows": value["detailed_rows"],
        "fresh_result_hammer_required": True,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

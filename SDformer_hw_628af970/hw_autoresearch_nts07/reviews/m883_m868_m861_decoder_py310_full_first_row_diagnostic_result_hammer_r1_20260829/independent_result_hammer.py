#!/usr/bin/env python3
"""Fresh, execution-blind result hammer for the M868 one-row diagnostic.

This reviewer never imports or executes the decoder workload.  It checks the
already-published result, one-way attempt, exact predecessor identities and
claim boundary, then attacks isolated copies of the sealed artifacts.
"""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Dict, Iterable, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / "results/m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829"
ATTEMPT = HW / "results/.m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed"
CANDIDATE = HW / "contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json"
RELEASE = HW / "contracts/m875_m868_decoder_py310_full_first_row_diagnostic_true_release_r1_20260829.json"
M876 = HW / "reviews/m876_m875_m868_decoder_py310_full_first_row_final_launch_hammer_r1_20260829"
M869 = HW / "reviews/m869_m868_decoder_py310_full_first_row_source_hammer_r1_20260829"
M861 = HW / "contracts/m861_decoder_streaming_event_sweep_candidate_r1_20260829.json"
M785 = HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"
M768 = HW / "contracts/m768_h67_decoder_a1_k8_address_timed_cycle_contract_r1_20260828.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "result_json": "53f71f804cad8acafdbc224d12acfbddc1510d1cb202286d67b018a1b1015344",
    "result_manifest": "9647330a7bceb3e9536a3c3850b146196fa0ba63e24980be96670c8a5ce2cd26",
    "result_outer": "e9d0c2c5bee93c2d32324a3a64f7871369848aec2846eb2dbf1ac3042110bb66",
    "attempt_json": "32d2c5e868a288e7b2b225dfccf5e10dfad59a51b8fd90c6c22b07afe1eb7550",
    "attempt_manifest": "07855bb11f76744ce45fac9351cd237ed8c157dd264ab43afb5fc4832f62955c",
    "attempt_outer": "fa4a2a68d76ebc53de89dfce1f93e9821941135ced5cb36d67617843d3351d02",
    "candidate": "2bcf8aeaf22cbf9c5178a9a030d72ee52372e78bdeec2c94e7361947d09d57d3",
    "release": "4e781456574ac6240a2303fe1d2104b1e7b517745f0a5d80db9b2322feeef85f",
    "m876_review": "488d8eb3ff97ce406515a864f5a4e07cf30117abf18e49e105ae171d99ab52ff",
    "m876_manifest": "f17728248f456c187e8fe070d768ccd27c4a7541647d15986c9dd5e21626c6b2",
    "m876_outer": "9f5a72f69a82d024b14ba2e3cb9710c176abcb6fcb0e381d9e2915830de36525",
    "m869_review": "38650a4a37e09a7ac4ae0d8d96a3838c433a2191fcfa368018f57292ab55cad5",
    "m869_manifest": "cc277cde39344880c3af3dd59e5583e02c93b30119b3df9e6bcfb7e8561f2f83",
    "m869_outer": "d827e0c24c62bdf05649bb1065267472c2c8799fcb82a280ec672bcd2d59452a",
    "m861": "5ca88752677ea82557ebf62032b373de086dde202614df3949a3f11f79a1e2f2",
    "m785": "612a2ba39ceecedc351f2f6550347ad50ca9526fd89ed143bc6362c3e5681810",
    "m768": "68f5e64c96deebb069a75c47ac7c326cd0f39d6b46bd67d1ad3711aef343daf9",
    "m768_source": "926069762c6274bae3aa7b88352e29fff8219cbbceba2f2be0ec46ee304a3f37",
    "m785_source": "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1",
    "m861_source": "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4",
    "m868_driver": "128fb2686d400593f59ed99390d0acf8c60d6c992f5daa951daa0ed4f6b0efbd",
    "m868_runner": "fb1042474073c98d5dbc81a79f604de5d257f3953fb4bc40a3dfe840e303fe5a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_IDENTITY = {
    "label": "M854_FIRST_D0_A1_T0",
    "population": "M686_ZURICH_CITY_09_A_S10",
    "record_ordinal": 0,
    "module_index": 0,
    "sample_id": 0,
    "configuration": "A1_OSG",
    "timestep": 0,
}

EXPECTED_CYCLE_CLASSES = {
    "active_service": 18502452,
    "compute": 1,
    "dependency_completion": 2046313,
    "memory": 0,
    "psum_bank": 0,
    "weight_bank": 0,
}


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> object:
    def pairs(rows: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("non-finite JSON token: " + token)
        ),
    )


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    require(
        bool(member.parts) and not member.is_absolute() and
        ".." not in member.parts and member.as_posix() == name,
        "unsafe manifest member: " + name,
    )
    return member


def verify_sealed(directory: Path, expected_population: Iterable[str]) -> Dict[str, object]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory absent or symlinked: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(),
            "manifest absent or symlinked")
    require(outer.is_file() and not outer.is_symlink(),
            "outer seal absent or symlinked")
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed manifest row")
        expected_hash, name = fields
        require(name not in expected_names, "duplicate manifest member")
        expected_names.add(name)
        member = directory.joinpath(*safe_member(name).parts)
        require(member.is_file() and not member.is_symlink(),
                "sealed member absent/nonregular: " + name)
        require(sha256(member) == expected_hash,
                "sealed member hash mismatch: " + name)
    all_files = set()
    directories = set()
    symlinks = []
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if member.is_symlink():
            symlinks.append(relative)
        elif member.is_dir():
            directories.add(relative)
        elif member.is_file():
            all_files.add(relative)
        else:
            raise Failure("nonregular topology member: " + relative)
    require(not symlinks, "symlink in sealed directory: " + repr(symlinks))
    require(not directories, "nested directory in flat sealed artifact")
    require(all_files == set(expected_population),
            "sealed artifact population drift: " + repr(sorted(all_files)))
    require(expected_names == all_files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "manifest population mismatch")
    outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_fields == [sha256(manifest), "SHA256SUMS"],
            "outer seal mismatch")
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "regular_files": len(all_files),
        "nested_directories": len(directories),
        "symlinks": len(symlinks),
    }


def exact_keys(value: Mapping[str, object], keys: Iterable[str], label: str) -> None:
    require(set(value) == set(keys), label + " key-set drift")


def exact_int(value: object, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum,
            label + " must be a finite integer")
    return int(value)


def finite_number(value: object, label: str, positive: bool = False) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool),
            label + " is not numeric")
    parsed = float(value)
    require(math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
            label + " is nonfinite or out of range")
    return parsed


def hex_digest(value: object, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " is not a lowercase SHA-256 digest")
    return value


def assert_no_forbidden_detail(value: object) -> None:
    if isinstance(value, dict):
        require("scheduled_requests" not in value and
                "compressed_schedule" not in value,
                "forbidden retained request population")
        for child in value.values():
            assert_no_forbidden_detail(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_detail(child)


def validate_attempt(directory: Path) -> Dict[str, object]:
    seal = verify_sealed(directory, {
        "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sha256(directory / "attempt.json") == EXPECTED["attempt_json"],
            "attempt receipt identity drift")
    require(seal["manifest_sha256"] == EXPECTED["attempt_manifest"] and
            seal["outer_seal_file_sha256"] == EXPECTED["attempt_outer"],
            "attempt double-seal identity drift")
    row = strict_json(directory / "attempt.json")
    require(isinstance(row, dict), "attempt receipt must be an object")
    exact_keys(row, {
        "candidate_sha256", "cycles_or_speedup_citable", "hammer",
        "interpreter_path", "interpreter_sha256", "interpreter_version",
        "max_attempts", "production_authorized", "runner_sha256", "schema",
        "status", "workload_identity",
    }, "attempt")
    require(row == {
        "candidate_sha256": EXPECTED["candidate"],
        "cycles_or_speedup_citable": False,
        "hammer": {
            "manifest_sha256": EXPECTED["m869_manifest"],
            "outer_seal_file_sha256": EXPECTED["m869_outer"],
            "review_sha256": EXPECTED["m869_review"],
        },
        "interpreter_path": "/opt/anaconda3/envs/pytorch310/bin/python3.10",
        "interpreter_sha256": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
        "interpreter_version": "3.10.18",
        "max_attempts": 1,
        "production_authorized": False,
        "runner_sha256": EXPECTED["m868_runner"],
        "schema": "m868_m861_decoder_full_first_row_attempt_v1",
        "status": "CONSUMED_IMMEDIATELY_BEFORE_M868_FULL_FIRST_ROW_DIAGNOSTIC",
        "workload_identity": "M854_FIRST_D0_A1_T0",
    }, "attempt receipt semantic drift")
    return seal


def validate_result(directory: Path) -> Dict[str, object]:
    seal = verify_sealed(directory, {
        "diagnostic.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sha256(directory / "diagnostic.json") == EXPECTED["result_json"],
            "diagnostic identity drift")
    require(seal["manifest_sha256"] == EXPECTED["result_manifest"] and
            seal["outer_seal_file_sha256"] == EXPECTED["result_outer"],
            "diagnostic double-seal identity drift")
    row = strict_json(directory / "diagnostic.json")
    require(isinstance(row, dict), "diagnostic must be an object")
    exact_keys(row, {"aggregate", "attempt", "claim_boundary", "identity",
                     "runtime", "schema", "status"}, "diagnostic")
    require(row["schema"] == "m868_m861_decoder_py310_full_first_row_diagnostic_v1",
            "diagnostic schema drift")
    require(row["status"] ==
            "PASS_M868_FULL_FIRST_ROW_DIAGNOSTIC__NONPRODUCTION__FRESH_RESULT_HAMMER_REQUIRED",
            "diagnostic status drift")
    require(row["identity"] == EXPECTED_IDENTITY,
            "D0/A1/t0 identity drift")

    aggregate = row["aggregate"]
    require(isinstance(aggregate, dict), "aggregate must be an object")
    exact_keys(aggregate, {
        "commit_sequence_sha256", "compressed_transaction_count",
        "cycle_classes_diagnostic_only", "detail_retained",
        "event_sweep_diagnostics", "expanded_request_count",
        "total_cycles_diagnostic_only", "transaction_address_sha256",
    }, "aggregate")
    require(exact_int(aggregate["compressed_transaction_count"], "compressed") == 9582057,
            "compressed cardinality drift")
    require(exact_int(aggregate["expanded_request_count"], "expanded") == 38672612,
            "expanded cardinality drift")
    total_cycles = exact_int(aggregate["total_cycles_diagnostic_only"],
                             "total cycles", minimum=1)
    require(total_cycles == 20548766, "diagnostic total-cycle identity drift")
    require(aggregate["detail_retained"] is False,
            "detail retention must remain disabled")
    tx_hash = hex_digest(aggregate["transaction_address_sha256"], "address hash")
    commit_hash = hex_digest(aggregate["commit_sequence_sha256"], "commit hash")
    require(tx_hash == "78b90d378956948fc3eab3d7a1bd6f88c8bcf4d32871e971641c9b1a62dfaa6e",
            "address hash drift")
    require(commit_hash == "aa69b355efd62b428e2909ee4c1dbecdf34ec3e1e8681b0c78ace19a444ff861",
            "commit hash drift")

    cycle_classes = aggregate["cycle_classes_diagnostic_only"]
    require(cycle_classes == EXPECTED_CYCLE_CLASSES,
            "cycle-class identity drift")
    require(all(type(value) is int and value >= 0
                for value in cycle_classes.values()),
            "cycle classes must be finite nonnegative integers")
    require(sum(cycle_classes.values()) == total_cycles,
            "mutually exclusive cycle conservation failure")

    sweep = aggregate["event_sweep_diagnostics"]
    require(isinstance(sweep, dict), "event-sweep diagnostics must be an object")
    exact_keys(sweep, {"active_service", "dependency_completion", "memory",
                       "psum_bank", "weight_bank"}, "event sweep")
    for name, counters in sweep.items():
        require(isinstance(counters, dict), name + " counters must be an object")
        exact_keys(counters, {"covered_cycles_before_priority",
                              "merged_intervals", "out_of_order_insertions"},
                   name + " counters")
        covered = exact_int(counters["covered_cycles_before_priority"],
                            name + " covered")
        exact_int(counters["merged_intervals"], name + " intervals")
        exact_int(counters["out_of_order_insertions"], name + " out-of-order")
        require(covered <= total_cycles, name + " coverage exceeds timeline")
    require(sweep["active_service"]["covered_cycles_before_priority"] ==
            cycle_classes["active_service"],
            "active-service interval/cardinality mismatch")
    require(sweep["dependency_completion"]["covered_cycles_before_priority"] ==
            total_cycles - 1,
            "dependency-completion interval endpoint drift")

    runtime = row["runtime"]
    require(isinstance(runtime, dict), "runtime must be an object")
    exact_keys(runtime, {"elapsed_seconds_diagnostic_only",
                         "next_port_cycle_entries", "outstanding_return_entries",
                         "process_max_rss_kib", "token_ready_entries"}, "runtime")
    elapsed = finite_number(runtime["elapsed_seconds_diagnostic_only"],
                            "elapsed seconds", positive=True)
    peak_rss = exact_int(runtime["process_max_rss_kib"], "peak RSS", minimum=1)
    require(exact_int(runtime["token_ready_entries"], "token entries") == 38672612,
            "token-ready semantic counter drift")
    require(exact_int(runtime["next_port_cycle_entries"], "port entries") == 24,
            "port-calendar semantic counter drift")
    require(exact_int(runtime["outstanding_return_entries"], "outstanding entries") == 19,
            "outstanding-return semantic counter drift")

    require(row["attempt"] == {
        "manifest_sha256": EXPECTED["attempt_manifest"],
        "outer_seal_file_sha256": EXPECTED["attempt_outer"],
    }, "embedded attempt identity drift")
    require(row["claim_boundary"] == {
        "decoder_complete": False,
        "fresh_result_hammer_required": True,
        "full_population": False,
        "one_full_first_row_diagnostic_completed": True,
        "paper_citable": False,
        "production_cycles": False,
        "production_speedup": False,
        "table_a": False,
    }, "claim boundary drift")
    assert_no_forbidden_detail(row)

    # Closure of the frozen M768 11-field interface.  The two large detail
    # populations are deliberately absent, population/config are represented
    # by the single-row identity, and same-cycle slot reuse remains an exact
    # pinned M861/M768 invariant rather than a new result degree of freedom.
    m768_field_closure = {
        "total_cycles": total_cycles,
        "expanded_request_count": aggregate["expanded_request_count"],
        "compressed_transaction_count": aggregate["compressed_transaction_count"],
        "scheduled_requests": "ABSENT_BY_DETAIL_RETAINED_FALSE",
        "compressed_schedule": "ABSENT_BY_DETAIL_RETAINED_FALSE",
        "transaction_address_sha256": tx_hash,
        "commit_sequence_sha256": commit_hash,
        "population_ids": [row["identity"]["population"]],
        "configs": [row["identity"]["configuration"]],
        "cycle_classes": cycle_classes,
        "same_cycle_response_slot_reuse": True,
    }
    exact_keys(m768_field_closure, {
        "total_cycles", "expanded_request_count", "compressed_transaction_count",
        "scheduled_requests", "compressed_schedule", "transaction_address_sha256",
        "commit_sequence_sha256", "population_ids", "configs", "cycle_classes",
        "same_cycle_response_slot_reuse",
    }, "M768 eleven-field closure")
    return {
        "seal": seal,
        "row": row,
        "m768_field_closure": m768_field_closure,
        "elapsed_seconds": elapsed,
        "peak_rss_kib": peak_rss,
    }


def verify_contract_sidecar(path: Path, expected: str) -> None:
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
            "contract identity drift: " + path.name)
    sidecar = path.with_name(path.name + ".sha256")
    outer = path.with_name(path.name + ".sha256.seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "contract sidecar absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    require(fields == [expected, path.name], "contract sidecar mismatch")
    outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_fields == [sha256(sidecar), sidecar.name],
            "contract outer sidecar mismatch")


def validate_authorities() -> Dict[str, object]:
    for path, key in ((CANDIDATE, "candidate"), (RELEASE, "release"),
                      (M861, "m861"), (M785, "m785"), (M768, "m768")):
        verify_contract_sidecar(path, EXPECTED[key])
    m876_seal = verify_sealed(M876, {
        "RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256",
        "independent_hammer.py", "independent_hammer_output.json",
        "mechanical_checks.txt", "review.json", "review.md",
    })
    m869_seal = verify_sealed(M869, {
        "RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256",
        "independent_hammer.py", "independent_hammer_output.json",
        "mechanical_checks.txt", "review.json", "review.md",
    })
    require(sha256(M876 / "review.json") == EXPECTED["m876_review"] and
            m876_seal["manifest_sha256"] == EXPECTED["m876_manifest"] and
            m876_seal["outer_seal_file_sha256"] == EXPECTED["m876_outer"],
            "M876 final hammer identity drift")
    require(sha256(M869 / "review.json") == EXPECTED["m869_review"] and
            m869_seal["manifest_sha256"] == EXPECTED["m869_manifest"] and
            m869_seal["outer_seal_file_sha256"] == EXPECTED["m869_outer"],
            "M869 source hammer identity drift")
    m876 = strict_json(M876 / "review.json")
    m869 = strict_json(M869 / "review.json")
    require(m876["status"] ==
            "PASS100_M868_PY310_FULL_FIRST_ROW_FINAL_LAUNCH__ONE_NONPRODUCTION_DIAGNOSTIC_AUTHORIZED" and
            m876["score"] == 100 and
            m876["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0} and
            m876["identity"]["release_sha256"] == EXPECTED["release"],
            "M876 authorization drift")
    require(m869["status"] ==
            "PASS100_M868_PY310_FULL_FIRST_ROW_SOURCE__AUTHORIZE_EXACTLY_ONE_NONPRODUCTION_DIAGNOSTIC" and
            m869["score"] == 100 and
            m869["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0},
            "M869 source hammer drift")
    release = strict_json(RELEASE)
    candidate = strict_json(CANDIDATE)
    m861 = strict_json(M861)
    m785 = strict_json(M785)
    m768 = strict_json(M768)
    require(release["candidate_binding"]["sha256"] == EXPECTED["candidate"] and
            release["m869_source_hammer"]["review_json_sha256"] == EXPECTED["m869_review"] and
            release["release_effective_only_after"]["fresh_final_hammer_directory"] ==
            M876.relative_to(HW).as_posix(),
            "M875 release chain drift")
    require(candidate["source_identity"]["m861_contract"]["sha256"] == EXPECTED["m861"] and
            candidate["source_identity"]["m785_contract"]["sha256"] == EXPECTED["m785"],
            "candidate substrate chain drift")
    require(m861["frozen_semantics"]["same_cycle_response_slot_reuse"] is True and
            m861["source_identity"]["m785_frozen_analyzer"]["sha256"] == EXPECTED["m785_source"] and
            m861["source_identity"]["m768_frozen_analyzer"]["sha256"] == EXPECTED["m768_source"],
            "M861 frozen semantics drift")
    require(m785["inputs"]["m768_substrate"]["sha256"] == EXPECTED["m768_source"] and
            m785["common_resource"]["lanes"] == 96 and
            m785["common_resource"]["accumulator_bits"] == 24 and
            m785["common_resource"]["onchip_sram_bytes_macro_rounded"] == 245760,
            "M785 physical contract drift")
    require(m768["transaction_semantics"]["same_cycle_response_slot_reuse"] is True and
            m768["common_resource"]["lanes"] == 96 and
            m768["common_resource"]["accumulator_bits"] == 24 and
            m768["common_resource"]["onchip_sram_bytes_macro_rounded"] == 245760,
            "M768 interface/resource drift")
    for path, key in (
        (HW / "system_simulator/scripts/analyze_m768_h67_decoder_a1_k8_address_timed_cycles.py", "m768_source"),
        (HW / "system_simulator/scripts/analyze_m785_h67_decoder_physical_residency_repair.py", "m785_source"),
        (HW / "system_simulator/scripts/analyze_m861_decoder_streaming_event_sweep.py", "m861_source"),
        (HW / "system_simulator/scripts/execute_m868_m861_decoder_py310_full_first_row_diagnostic.py", "m868_driver"),
        (HW / "system_simulator/scripts/run_m868_m861_decoder_py310_full_first_row_one_shot.sh", "m868_runner"),
    ):
        require(path.is_file() and not path.is_symlink() and sha256(path) == EXPECTED[key],
                "frozen source drift: " + path.name)
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 drift")
    return {"m876": m876_seal, "m869": m869_seal}


def namespace_audit() -> Dict[str, object]:
    names = sorted(entry.name for entry in (HW / "results").iterdir()
                   if entry.name.startswith("m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829") or
                   entry.name.startswith(".m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed"))
    require(names == [
        ".m868_m861_decoder_py310_full_first_row_diagnostic_r1_attempt_consumed",
        "m868_m861_decoder_py310_full_first_row_diagnostic_r1_20260829",
    ], "attempt/result/stage/quarantine namespace drift: " + repr(names))
    quarantines = [name for name in names if ".failed_or_incomplete." in name]
    require(not quarantines, "unexpected M868 quarantine")
    return {"matching_names": names, "attempt_count": 1,
            "canonical_result_count": 1, "quarantine_count": 0,
            "private_stage_count": 0}


def reseal_flat(directory: Path, payload_name: str) -> None:
    manifest = directory / "SHA256SUMS"
    manifest.write_text(sha256(directory / payload_name) + "  " + payload_name + "\n",
                        encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="utf-8")


def expect_reject(label: str, action) -> str:
    try:
        action()
    except Exception as error:  # Each isolated mutation must fail closed.
        return "PASS_REJECTED_{}__{}".format(label, type(error).__name__)
    raise Failure("mutation attack unexpectedly accepted: " + label)


def result_copy(temp_root: Path, name: str) -> Path:
    destination = temp_root / name
    shutil.copytree(RESULT, destination, symlinks=True)
    return destination


def attempt_copy(temp_root: Path, name: str) -> Path:
    destination = temp_root / name
    shutil.copytree(ATTEMPT, destination, symlinks=True)
    return destination


def mutation_attacks() -> Dict[str, object]:
    outcomes = {}
    with tempfile.TemporaryDirectory(prefix="m883_result_hammer.") as temp:
        root = Path(temp)

        path = result_copy(root, "payload_flip")
        with (path / "diagnostic.json").open("ab") as handle:
            handle.write(b" ")
        outcomes["payload_flip_unsealed"] = expect_reject(
            "payload_flip_unsealed", lambda: validate_result(path))

        path = result_copy(root, "manifest_flip")
        with (path / "SHA256SUMS").open("ab") as handle:
            handle.write(b" ")
        outcomes["manifest_flip"] = expect_reject(
            "manifest_flip", lambda: validate_result(path))

        path = result_copy(root, "outer_flip")
        with (path / "SHA256SUMS.seal.sha256").open("ab") as handle:
            handle.write(b" ")
        outcomes["outer_flip"] = expect_reject(
            "outer_flip", lambda: validate_result(path))

        path = result_copy(root, "extra_file")
        (path / "extra.txt").write_text("attack\n", encoding="utf-8")
        outcomes["extra_regular_file"] = expect_reject(
            "extra_regular_file", lambda: validate_result(path))

        path = result_copy(root, "symlink")
        os.symlink("diagnostic.json", path / "alias.json")
        outcomes["symlink_injection"] = expect_reject(
            "symlink_injection", lambda: validate_result(path))

        mutations = {
            "schema_resealed": lambda row: row.__setitem__("schema", "evil"),
            "identity_resealed": lambda row: row["identity"].__setitem__("timestep", 1),
            "expanded_resealed": lambda row: row["aggregate"].__setitem__("expanded_request_count", 1),
            "compressed_resealed": lambda row: row["aggregate"].__setitem__("compressed_transaction_count", 1),
            "total_cycle_resealed": lambda row: row["aggregate"].__setitem__("total_cycles_diagnostic_only", 1),
            "cycle_conservation_resealed": lambda row: row["aggregate"]["cycle_classes_diagnostic_only"].__setitem__("compute", 2),
            "detail_population_resealed": lambda row: row["aggregate"].__setitem__("scheduled_requests", []),
            "claim_escalation_resealed": lambda row: row["claim_boundary"].__setitem__("paper_citable", True),
            "same_attempt_resealed": lambda row: row["attempt"].__setitem__("manifest_sha256", "0" * 64),
        }
        for label, mutate in mutations.items():
            path = result_copy(root, label)
            row = strict_json(path / "diagnostic.json")
            mutate(row)
            (path / "diagnostic.json").write_text(
                json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8")
            reseal_flat(path, "diagnostic.json")
            outcomes[label] = expect_reject(label, lambda path=path: validate_result(path))

        path = result_copy(root, "nonfinite")
        text = (path / "diagnostic.json").read_text(encoding="utf-8")
        text = text.replace("932.0783571209759", "NaN", 1)
        (path / "diagnostic.json").write_text(text, encoding="utf-8")
        reseal_flat(path, "diagnostic.json")
        outcomes["nonfinite_resealed"] = expect_reject(
            "nonfinite_resealed", lambda: validate_result(path))

        path = attempt_copy(root, "attempt_status")
        row = strict_json(path / "attempt.json")
        row["status"] = "RESTORED"
        (path / "attempt.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reseal_flat(path, "attempt.json")
        outcomes["attempt_status_resealed"] = expect_reject(
            "attempt_status_resealed", lambda: validate_attempt(path))

    require(len(outcomes) == 16, "mutation attack count drift")
    return {"passed": len(outcomes), "failed": 0,
            "outcomes": outcomes, "canonical_modified": False}


def main() -> int:
    require(HERE.name ==
            "m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829",
            "fixed review path drift")
    authorities = validate_authorities()
    namespace = namespace_audit()
    attempt = validate_attempt(ATTEMPT)
    result = validate_result(RESULT)
    attacks = mutation_attacks()
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 changed during review")

    row = result["row"]
    review = {
        "schema": "m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_v1",
        "date": "2026-08-29",
        "status": "PASS100_M868_ONE_FULL_FIRST_ROW_NONPRODUCTION_DIAGNOSTIC_RESULT_ADMITTED",
        "score": 100,
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "reviewer_role": "Fresh independent result hammer; no full-row replay, population, production, remote, VCS, DC, PT, FM, PTPX, EDA, license, GPU, or training action.",
        "identity": {
            "canonical_result": RESULT.relative_to(HW).as_posix(),
            "diagnostic_sha256": EXPECTED["result_json"],
            "result_manifest_sha256": EXPECTED["result_manifest"],
            "result_outer_seal_file_sha256": EXPECTED["result_outer"],
            "attempt_sha256": EXPECTED["attempt_json"],
            "attempt_manifest_sha256": EXPECTED["attempt_manifest"],
            "attempt_outer_seal_file_sha256": EXPECTED["attempt_outer"],
            "candidate_sha256": EXPECTED["candidate"],
            "release_sha256": EXPECTED["release"],
            "m876_review_sha256": EXPECTED["m876_review"],
            "m876_manifest_sha256": EXPECTED["m876_manifest"],
            "m876_outer_seal_file_sha256": EXPECTED["m876_outer"],
            "m869_review_sha256": EXPECTED["m869_review"],
            "m869_manifest_sha256": EXPECTED["m869_manifest"],
            "m869_outer_seal_file_sha256": EXPECTED["m869_outer"],
            "m861_contract_sha256": EXPECTED["m861"],
            "m785_contract_sha256": EXPECTED["m785"],
            "m768_contract_sha256": EXPECTED["m768"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "topology": {"canonical": result["seal"], "attempt": attempt,
                     "namespace": namespace},
        "diagnostic": {
            "workload": EXPECTED_IDENTITY,
            "compressed_transaction_count": 9582057,
            "expanded_request_count": 38672612,
            "total_cycles_diagnostic_only": 20548766,
            "cycle_classes_diagnostic_only": EXPECTED_CYCLE_CLASSES,
            "transaction_address_sha256": row["aggregate"]["transaction_address_sha256"],
            "commit_sequence_sha256": row["aggregate"]["commit_sequence_sha256"],
            "event_sweep_diagnostics": row["aggregate"]["event_sweep_diagnostics"],
            "m768_eleven_field_schema_closed": True,
            "scheduled_requests_retained": False,
            "compressed_schedule_retained": False,
            "same_cycle_response_slot_reuse": True,
        },
        "runtime": {
            "elapsed_seconds_diagnostic_only": result["elapsed_seconds"],
            "elapsed_minutes_diagnostic_only": result["elapsed_seconds"] / 60.0,
            "process_max_rss_kib": result["peak_rss_kib"],
            "process_max_rss_gib": result["peak_rss_kib"] / 1048576.0,
            "token_ready_entries": row["runtime"]["token_ready_entries"],
            "next_port_cycle_entries": row["runtime"]["next_port_cycle_entries"],
            "outstanding_return_entries": row["runtime"]["outstanding_return_entries"],
        },
        "mutation_attacks": attacks,
        "authority_checks": {
            "m876_double_seal": authorities["m876"],
            "m869_double_seal": authorities["m869"],
            "candidate_release_m861_m785_m768_sidecars": "PASS",
            "frozen_m768_m785_m861_m868_sources": "PASS",
        },
        "claim_boundary": {
            "one_full_first_row_nonproduction_cycle_diagnostic": True,
            "full_decoder_population": False,
            "decoder_complete": False,
            "production_cycles": False,
            "production_speedup": False,
            "system_speedup": False,
            "table_a": False,
            "paper_citable_performance": False,
            "rtl_vcs_eda_energy_ppa": False,
            "docs359_modified": False,
        },
        "execution_receipt": {
            "full_row_replays_by_reviewer": 0,
            "population_or_production_runs_by_reviewer": 0,
            "remote_or_network_actions": 0,
            "vcs_dc_pt_fm_ptpx_eda_license_gpu_runs": 0,
            "canonical_result_modified": False,
            "attempt_modified": False,
            "frozen_source_modified": False,
            "docs359_modified": False,
            "temporary_isolated_mutation_attacks": attacks["passed"],
        },
        "required_next_gate": "A new sharded/population release, exact population execution and a fresh result hammer are still required before any decoder-complete, production-cycle, speedup, Table-A, system, energy, or PPA statement.",
        "verdict": "PASS",
    }
    (HERE / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    mechanical = [
        "M883 M868 one-row diagnostic fresh result hammer",
        "PASS canonical result double seal and flat regular topology",
        "PASS one one-way attempt and zero quarantine/private stage",
        "PASS exact D0/A1/t0 identity and 9,582,057/38,672,612 cardinalities",
        "PASS M768 eleven-field aggregate closure; detailed lists absent by design",
        "PASS total-cycle and six-class conservation: 20,548,766",
        "PASS all event-sweep/runtime counters finite and schema-closed",
        "PASS 16/16 isolated sealed-copy mutation attacks",
        "PASS docs359 exact SHA",
        "NO full-row/population/production/remote/EDA/license/GPU execution by reviewer",
    ]
    (HERE / "mechanical_checks.txt").write_text(
        "\n".join(mechanical) + "\n", encoding="utf-8")
    review_md = """# M883 fresh result hammer: M868 one full-first-row diagnostic

Verdict: **PASS 100/100**, with `P0/P1/P2 = 0/0/0`.

The canonical M868 artifact and its one-way attempt are flat, regular,
double-sealed, unique, and have zero quarantine/private-stage residue.  The
frozen identity is exactly `M854_FIRST_D0_A1_T0` / D0 / A1-OSG / timestep 0.
It contains 9,582,057 compressed transactions and 38,672,612 expanded
requests.  The event sweep reports 20,548,766 diagnostic-only cycles, with
the six mutually exclusive classes conserving exactly.  The frozen M768
11-field interface is closed through aggregate fields, explicit nonretention
of the two detail lists, the single population/config identity, and the pinned
same-cycle-slot-reuse invariant.

Observed runtime was 932.078357 s (15.534639 min), with peak RSS 8,897,128
KiB (8.485 GiB).  Sixteen isolated sealed-copy attacks were rejected.  The
canonical result, attempt, sources, and docs/359 were not modified.

This admits only one nonproduction first-row cycle diagnostic.  It is not a
full decoder population, production cycle/speedup, decoder-complete result,
Table-A row, system result, or paper-citable performance point.
"""
    (HERE / "review.md").write_text(review_md, encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS100 M883 fresh result hammer; nonproduction one-row diagnostic only\n",
        encoding="utf-8")
    print(json.dumps({
        "status": review["status"], "score": 100,
        "severity_counts": review["severity_counts"],
        "diagnostic_cycles": 20548766,
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_rss_kib": result["peak_rss_kib"],
        "mutation_attacks": attacks["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

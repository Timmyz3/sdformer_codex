#!/usr/bin/env python3
"""Receipt-blind M1073 hammer of the M1072 row-provenance source.

This audit does not read the M1072 author receipt and never advances the
51.84M-row production generator.  It attacks only tiny records/read windows,
re-derives the service ledger from M1016, and checks the frozen 1RW kernel.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
CHECKER = HW / "system_simulator/scripts/check_m1072_c1_row_provenance_exact_1rw_source.py"
TESTS = HW / "system_simulator/tests/test_m1072_c1_row_provenance_exact_1rw_source.py"
CONTRACT = HW / "contracts/m1072_m1065_c1_row_provenance_exact_1rw_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c"
CHECKER_SHA = "8017df21104fc87bd394c35618470eddfc6daabe4fb12aba56598b84789caa43"
TESTS_SHA = "051192f46e6fdd2d4803a44b56e556b8e2b54e409e30b07629d1820435707820"
CONTRACT_SHA = "017d5254346e54a24c3082cb9cd17f61e19d4f895ef6366e55345784e6b4ec03"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
SERVICE_DIGEST = "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def rejected(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (RuntimeError, TypeError, ValueError, OSError):
        return True
    return False


def load_source():
    require(sha256(SOURCE) == SOURCE_SHA, "M1072 source identity drift")
    spec = importlib.util.spec_from_file_location("m1073_receipt_blind_m1072", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1072")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resign(module, record):
    return replace(
        record,
        provenance_sha256=hashlib.sha256(
            module._canonical_provenance_payload(module.record_payload(record))
        ).hexdigest(),
    )


def rederive_service_ledger(module) -> tuple[dict[str, int], str]:
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    for task_id in range(module.TASKS):
        chunk = (task_id // module.M1064.PARTITIONS) % module.M1064.CHUNKS
        row_count = min(
            module.M1064.ROW_TILE,
            module.M1064.ROWS_PER_PHASE - chunk * module.M1064.ROW_TILE,
        )
        receipt = module.M1016.common_receipt(task_id, row_count)
        module.M1064.validate_receipt_exact(receipt, task_id, row_count)
        counts.update(receipt["counts"])
        digest.update(canonical(receipt))
        if (task_id + 1) % module.TASKS_PER_SAMPLE == 0:
            sample = (task_id + 1) // module.TASKS_PER_SAMPLE - 1
            commit = {
                "task": module.TASKS + sample,
                "counts": {
                    resource: (module.M1064.COMMIT_CYCLES_PER_SAMPLE
                               if resource == "commit" else 0)
                    for resource in module.M1064.RESOURCES
                },
                "sample_commit": sample,
            }
            counts.update(commit["counts"])
            digest.update(canonical(commit))
    return dict(counts), digest.hexdigest()


def main() -> dict[str, Any]:
    module = load_source()
    identity = {
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(CHECKER),
        "tests_sha256": sha256(TESTS),
        "contract_sha256": sha256(CONTRACT),
        "docs359_sha256": sha256(DOCS359),
        "canonical_rows_sha256": sha256(module.ROWS),
        "canonical_rows_bytes": module.ROWS.stat().st_size,
    }
    require(identity == {
        "source_sha256": SOURCE_SHA,
        "checker_sha256": CHECKER_SHA,
        "tests_sha256": TESTS_SHA,
        "contract_sha256": CONTRACT_SHA,
        "docs359_sha256": DOCS359_SHA,
        "canonical_rows_sha256": ROWS_SHA,
        "canonical_rows_bytes": 466_560_000,
    }, "frozen identity drift")

    production = module.iter_canonical_full_replay_results
    unique_boundary = {
        "generator": inspect.isgeneratorfunction(production),
        "zero_parameters": len(inspect.signature(production).parameters) == 0,
        "reader_zero_parameters": len(inspect.signature(module.CanonicalRowReader).parameters) == 0,
        "coverage_zero_parameters": len(inspect.signature(module.ProvenanceCoverage).parameters) == 0,
        "caller_record_argument_rejected": rejected(lambda: production([])),
        "caller_sample_argument_rejected": rejected(lambda: production(0)),
        "caller_work_argument_rejected": rejected(lambda: production(work=0)),
        "caller_preprocess_argument_rejected": rejected(lambda: production(preprocess=0)),
        "caller_capacity_argument_rejected": rejected(lambda: production(capacity={})),
        "caller_coverage_argument_rejected": rejected(lambda: production(coverage={})),
    }
    require(all(unique_boundary.values()), "caller-controlled production boundary")

    row_path = (HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/"
                "m410r2_h67_q32_runtime_rows_32.memh")
    row_info = os.lstat(row_path)
    canonical_reader = {
        "exact_path": module.ROWS.absolute() == row_path.absolute(),
        "regular_nonsymlink": stat.S_ISREG(row_info.st_mode) and not row_path.is_symlink(),
        "bytes": row_info.st_size,
        "sha256": identity["canonical_rows_sha256"],
        "pread_named_in_reader": "os.pread" in inspect.getsource(module.CanonicalRowReader),
        "nofollow_named_in_reader": "O_NOFOLLOW" in inspect.getsource(module.CanonicalRowReader),
        "final_hash_named_in_close": "final_hash=True" in inspect.getsource(module.CanonicalRowReader.close),
    }
    require(canonical_reader == {
        "exact_path": True,
        "regular_nonsymlink": True,
        "bytes": 466_560_000,
        "sha256": ROWS_SHA,
        "pread_named_in_reader": True,
        "nofollow_named_in_reader": True,
        "final_hash_named_in_close": True,
    }, "canonical M410 reader boundary drift")

    # Read two canonical tiny windows.  Independently recompute every task-0
    # cycle field through the frozen M1016 functions.
    with module.CanonicalRowReader() as reader:
        record0 = reader.derive(0)
        raw1, _ = reader.raw_for_task(1)
    with os.fdopen(os.open(row_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        raw0 = os.pread(stream.fileno(), 64 * module.M1064.BYTES_PER_LINE, 0)
    masks0 = [int(line, 16) & 0xFFFF for line in raw0.splitlines()]
    masks0_array = module.np.asarray(masks0, dtype=module.np.uint16)
    receipt0 = module.M1016.common_receipt(0, 64)
    independently_derived = {
        "common_receipt": receipt0,
        "preprocess": {},
        "works": {},
        "parents": {},
    }
    for design in module.DESIGNS:
        work, parent = module.M1016.parent_for_design(design, masks0_array)
        independently_derived["works"][design] = int(work)
        independently_derived["preprocess"][design] = int(
            module.M1016.preprocess_for_design(design, masks0_array, receipt0)
        )
        independently_derived["parents"][design] = {
            "reads": int(parent.get("reads", 0)),
            "writes": int(parent.get("writes", 0)),
            "forwards": int(parent.get("forwards", 0)),
            "work_cycles": int(work),
        }
    task0_rederivation = {
        "raw_row_bytes_sha256": hashlib.sha256(raw0).hexdigest(),
        "masks_le16_sha256": hashlib.sha256(b"".join(
            int(mask).to_bytes(2, "little") for mask in masks0
        )).hexdigest(),
        "shared_preprocess_cycles": max(independently_derived["preprocess"].values()),
        "works": independently_derived["works"],
        "parents": independently_derived["parents"],
        "common_receipt_matches": record0.common_receipt == receipt0,
        "record_matches_all_rederived_fields": (
            record0.raw_row_bytes_sha256 == hashlib.sha256(raw0).hexdigest()
            and record0.masks_le16_sha256 == hashlib.sha256(b"".join(
                int(mask).to_bytes(2, "little") for mask in masks0
            )).hexdigest()
            and record0.shared_preprocess_cycles == max(independently_derived["preprocess"].values())
            and dict(record0.works) == independently_derived["works"]
            and {name: dict(record0.parents[name]) for name in module.DESIGNS}
                == independently_derived["parents"]
            and record0.common_receipt == receipt0
        ),
        "provenance_self_digest_matches": (
            hashlib.sha256(module._canonical_provenance_payload(
                module.record_payload(record0)
            )).hexdigest() == record0.provenance_sha256
        ),
    }
    require(task0_rederivation["record_matches_all_rederived_fields"] and
            task0_rederivation["provenance_self_digest_matches"] and
            task0_rederivation["shared_preprocess_cycles"] == 210 and
            task0_rederivation["works"] == {
                "candidate": 1664, "strongest_zero": 4392,
                "same_coordinate_bit": 4392,
            }, "task-0 row-to-cycle rederivation drift")

    attacks: dict[str, bool] = {}
    forged = replace(
        record0,
        shared_preprocess_cycles=0,
        works={"candidate": 0, "strongest_zero": 999_999,
               "same_coordinate_bit": 999_999},
        parents={
            "candidate": {"reads": 0, "writes": 0, "forwards": 0,
                          "work_cycles": 0},
            "strongest_zero": {"reads": 0, "writes": 0, "forwards": 0,
                               "work_cycles": 999_999},
            "same_coordinate_bit": {"reads": 0, "writes": 0, "forwards": 0,
                                    "work_cycles": 999_999},
        },
    )
    forged = resign(module, forged)
    module.validate_record_shape(forged)
    attacks["manual_work_0_999999_preprocess_0"] = rejected(
        lambda: module.validate_external_records_against_frozen([forged])
    )
    zero = module.derive_record_from_exact_raw(0, b"00000000\n" * 64,
                                               record0.file_offset)
    attacks["all_zero_masks"] = rejected(
        lambda: module.validate_external_records_against_frozen([zero])
    )
    reordered = module.derive_record_from_exact_raw(0, raw1, record0.file_offset)
    attacks["row_reorder"] = rejected(
        lambda: module.validate_external_records_against_frozen([reordered])
    )
    wrong_offset = resign(module, replace(record0, file_offset=record0.file_offset + 9))
    attacks["wrong_offset"] = rejected(
        lambda: module.validate_external_records_against_frozen([wrong_offset])
    )
    wrong_raw_digest = resign(module, replace(record0, raw_row_bytes_sha256="0" * 64))
    attacks["wrong_raw_digest"] = rejected(
        lambda: module.validate_external_records_against_frozen([wrong_raw_digest])
    )
    wrong_mask_digest = resign(module, replace(record0, masks_le16_sha256="0" * 64))
    attacks["wrong_mask_digest"] = rejected(
        lambda: module.validate_external_records_against_frozen([wrong_mask_digest])
    )
    attacks["wrong_provenance_digest"] = rejected(
        lambda: module.validate_record_shape(replace(record0, provenance_sha256="0" * 64))
    )

    reader = module.CanonicalRowReader()
    try:
        with mock.patch.object(module.os, "pread", return_value=b""):
            attacks["short_pread"] = rejected(lambda: reader.raw_for_task(0))
    finally:
        reader.close()
    reader = module.CanonicalRowReader()
    try:
        actual = module.os.fstat(reader._fd)
        fake = SimpleNamespace(
            st_dev=actual.st_dev, st_ino=actual.st_ino, st_size=actual.st_size,
            st_mtime_ns=actual.st_mtime_ns + 1, st_ctime_ns=actual.st_ctime_ns,
            st_mode=actual.st_mode,
        )
        with mock.patch.object(module.os, "fstat", return_value=fake):
            attacks["file_stat_drift"] = rejected(
                lambda: reader._verify_unchanged(final_hash=False)
            )
    finally:
        reader.close()
    original_rows = module.ROWS
    try:
        with tempfile.NamedTemporaryFile("wb") as stream:
            stream.write(b"00000000\n")
            stream.flush()
            module.ROWS = Path(stream.name)
            attacks["canonical_path_drift"] = rejected(module.CanonicalRowReader)
    finally:
        module.ROWS = original_rows
    original_size = module.ROWS_BYTES
    try:
        module.ROWS_BYTES = original_size - 1
        attacks["canonical_size_drift"] = rejected(module.CanonicalRowReader)
    finally:
        module.ROWS_BYTES = original_size
    original_sha = module.ROWS_SHA
    try:
        module.ROWS_SHA = "0" * 64
        attacks["canonical_sha_drift"] = rejected(module.CanonicalRowReader)
    finally:
        module.ROWS_SHA = original_sha

    coverage0 = module.ProvenanceCoverage()
    population = {
        "empty": not coverage0.proof()["full_coverage_pass"],
    }
    coverage0.consume_internal(record0)
    population["partial"] = not coverage0.proof()["full_coverage_pass"]
    population["duplicate"] = rejected(lambda: coverage0.consume_internal(record0))
    population["out_of_order"] = rejected(
        lambda: module.ProvenanceCoverage().consume_internal(
            module.derive_record_from_exact_raw(1, raw1, 64 * module.M1064.BYTES_PER_LINE)
        )
    )
    population["constructor_override"] = rejected(
        lambda: module.ProvenanceCoverage(next_task_id=module.TASKS)
    )

    schema_attacks = {
        "bool_task_id": rejected(
            lambda: module.validate_record_shape(replace(record0, task_id=True))
        ),
        "bool_work": rejected(
            lambda: module.validate_record_shape(replace(
                record0,
                works={"candidate": True, "strongest_zero": 4392,
                       "same_coordinate_bit": 4392},
            ))
        ),
        "extra_work_key": rejected(
            lambda: module.validate_record_shape(replace(
                record0, works={**dict(record0.works), "extra": 0}
            ))
        ),
        "extra_parent_key": rejected(
            lambda: module.validate_record_shape(replace(
                record0,
                parents={**{name: dict(record0.parents[name])
                            for name in module.DESIGNS}, "extra": {}},
            ))
        ),
        "external_empty": rejected(
            lambda: module.validate_external_records_against_frozen([])
        ),
    }
    boolean_receipt = json.loads(json.dumps(record0.common_receipt))
    boolean_receipt["counts"]["dma"] = True
    extra_receipt = json.loads(json.dumps(record0.common_receipt))
    extra_receipt["extra"] = 0
    schema_attacks["bool_receipt"] = rejected(
        lambda: module.M1064.validate_receipt_exact(boolean_receipt, 0, 64)
    )
    schema_attacks["extra_receipt"] = rejected(
        lambda: module.M1064.validate_receipt_exact(extra_receipt, 0, 64)
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
        stream.write('{"schema":1,"schema":2}')
        stream.flush()
        schema_attacks["duplicate_json_key"] = rejected(
            lambda: module.strict_json(Path(stream.name))
        )

    require(all(attacks.values()), "row/file provenance attack escaped")
    require(all(population.values()), "population attack escaped")
    require(all(schema_attacks.values()), "boolean/schema attack escaped")

    service_counts, service_digest = rederive_service_ledger(module)
    require(module.TASKS == 812_160 and module.SAMPLES == 10 and
            service_counts == {
                "psum": 12_994_560, "weight": 70_853_184,
                "source": 51_840_000, "dma": 1_476_108, "commit": 960_000,
            } and service_digest == SERVICE_DIGEST,
            "independent service ledger drift")

    capacity = module.M1064.derive_physical_capacity()
    config = module.M1056.ArbiterConfig()
    config.validate()
    arbiter = module.M1056.small_oracle()
    resources = {
        "capacity_bytes": capacity["derived_total_bytes"],
        "capacity_margin_bytes": capacity["derived_margin_bytes"],
        "capacity_only_admitted": capacity["capacity_only_214912B_admitted"],
        "groups": config.groups,
        "ports_per_group": config.ports_per_group,
        "port_mode": config.port_mode,
        "different_address_same_port_conflict": arbiter["different_address_same_port_conflict"],
        "same_address_raw_enforced": arbiter["same_address_raw_enforced"],
        "delay_cascades": arbiter["delay_cascades_to_next_task_and_commit"],
        "cascade_nominal_cycles": arbiter["cascade"]["nominal_cycles"],
        "cascade_arbitrated_cycles": arbiter["cascade"]["arbitrated_cycles"],
    }
    require(resources == {
        "capacity_bytes": 214_912, "capacity_margin_bytes": 30_848,
        "capacity_only_admitted": False, "groups": 4,
        "ports_per_group": 1, "port_mode": "1RW",
        "different_address_same_port_conflict": True,
        "same_address_raw_enforced": True, "delay_cascades": True,
        "cascade_nominal_cycles": 20, "cascade_arbitrated_cycles": 22,
    }, "capacity/1RW cascade drift")

    source_text = SOURCE.read_text(encoding="utf-8")
    real_coverage = module.ProvenanceCoverage()
    real_coverage.consume_internal(record0)
    forged_coverage = module.ProvenanceCoverage()
    forged_coverage.consume_internal(forged)
    raw_digest_coverage = module.ProvenanceCoverage()
    raw_digest_coverage.consume_internal(wrong_raw_digest)
    expected_first_digest = hashlib.sha256(
        record0.provenance_sha256.encode()
    ).hexdigest()
    digest_binding = {
        "record_payload_has_task": '"task_id": record.task_id' in source_text,
        "record_payload_has_coordinate": '"coordinate": [record.sample' in source_text,
        "record_payload_has_offset": '"file_offset": record.file_offset' in source_text,
        "record_payload_has_raw_digest": '"raw_row_bytes_sha256": record.raw_row_bytes_sha256' in source_text,
        "record_payload_has_mask_digest": '"masks_le16_sha256": record.masks_le16_sha256' in source_text,
        "record_payload_has_preprocess": '"shared_preprocess_cycles": record.shared_preprocess_cycles' in source_text,
        "record_payload_has_work": '"works": dict(record.works)' in source_text,
        "record_payload_has_parent": '"parents": {name: dict(record.parents[name])' in source_text,
        "record_payload_has_receipt": '"common_receipt": dict(record.common_receipt)' in source_text,
        "coverage_consumes_provenance_digest": "self.execution_digest.update(record.provenance_sha256.encode())" in source_text,
        "coverage_requires_order": "record.task_id == self.next_task_id" in source_text,
        "coverage_caller_pass_false": '"caller_supplied_coverage_or_digest": False' in source_text,
        "first_record_digest_exact": (
            real_coverage.proof()["execution_provenance_digest_sha256"]
            == expected_first_digest
        ),
        "preprocess_work_mutation_changes_digest": (
            forged_coverage.proof()["execution_provenance_digest_sha256"]
            != real_coverage.proof()["execution_provenance_digest_sha256"]
        ),
        "raw_row_digest_mutation_changes_digest": (
            raw_digest_coverage.proof()["execution_provenance_digest_sha256"]
            != real_coverage.proof()["execution_provenance_digest_sha256"]
        ),
    }
    require(all(digest_binding.values()), "coverage/provenance digest binding drift")

    forbidden = [
        HW / "results/m1072_c1_row_provenance_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1072_c1_row_provenance_exact_1rw_full_replay_attempt_consumed",
        HW / "results/m1074_m1072_c1_row_provenance_exact_1rw_full_replay_r1_20260830",
        HW / "results/.m1074_m1072_c1_row_provenance_exact_1rw_full_replay_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden),
            "full replay namespace appeared during source hammer")
    require(sha256(DOCS359) == DOCS359_SHA, "docs359 drift after hammer")

    return {
        "schema": "m1073_m1072_c1_row_provenance_exact_1rw_source_hammer_mechanical_v1",
        "status": "PASS_M1073_M1072_C1_ROW_PROVENANCE_EXACT_1RW_SOURCE_HAMMER",
        "identity": identity,
        "scope": {
            "receipt_blind": True,
            "source_modified": False,
            "production_generator_advanced": False,
            "full_51840000_replay_executed": False,
            "eda_gpu_remote_used": False,
            "docs359_modified": False,
        },
        "unique_production_boundary": unique_boundary,
        "canonical_reader": canonical_reader,
        "task0_independent_m1016_rederivation": task0_rederivation,
        "m1065_and_file_attacks_rejected": attacks,
        "population_attacks_rejected": population,
        "boolean_schema_attacks_rejected": schema_attacks,
        "service_ledger": {
            "tasks": module.TASKS,
            "sample_commits": module.SAMPLES,
            "counts": service_counts,
            "digest_sha256": service_digest,
        },
        "capacity_and_exact_1rw": resources,
        "coverage_digest_binding": digest_binding,
        "claim_boundary": {
            "m1072_source_ready": True,
            "m1074_full_release_source_may_be_authored": True,
            "launch_now": False,
            "full_trace_port_feasibility": False,
            "capacity_only_214912B_admitted": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    output = main()
    target = HERE / "mechanical_checks.json"
    temporary = HERE / ".mechanical_checks.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(output["status"])

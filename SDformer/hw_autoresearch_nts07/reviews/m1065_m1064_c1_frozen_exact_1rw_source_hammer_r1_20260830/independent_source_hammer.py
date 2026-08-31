#!/opt/anaconda3/bin/python3.12
"""Receipt-blind M1065 hammer for the M1064 source-only boundary.

This test never calls ``iter_frozen_task_records`` or
``replay_frozen_sample``.  In particular, it does not execute the 51.84M-row
replay.  It re-derives the service ledger from the frozen M1016 function and
attacks the public record/replay boundary with tiny records only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1064_c1_frozen_exact_1rw_replay_source.py"
CHECKER = HW / "system_simulator/scripts/check_m1064_c1_frozen_exact_1rw_replay_source.py"
TESTS = HW / "system_simulator/tests/test_m1064_c1_frozen_exact_1rw_replay_source.py"
CONTRACT = HW / "contracts/m1064_m1057_c1_frozen_exact_1rw_replay_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "ecf2625ae60a9f7848fc32b852b67f8efd3439c5fb24b9904ef397d39aafed09"
CHECKER_SHA = "38c712097ba3119f046fcb4c5941995cd81514c8b25d71189c38fe610715a33e"
TESTS_SHA = "0956d82a8510a2307970b161f240df394d1bbd3e268f9f519997d7a205af864e"
CONTRACT_SHA = "203392094fed8dc29bcd65abd400a21a1a7a7607686fae77c1eb19e1eefeaa24"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SERVICE_DIGEST = "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    assert sha256(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location("m1065_receipt_blind_m1064", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (RuntimeError, TypeError, ValueError, SystemExit):
        return True
    return False


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def result_namespaces() -> list[str]:
    results = HW / "results"
    return sorted(
        path.name for path in results.iterdir()
        if "m1064" in path.name.lower() or "m1066" in path.name.lower()
    )


def project_processes() -> list[str]:
    run = subprocess.run(
        ["pgrep", "-af", "m1064|m1066"], text=True,
        capture_output=True, check=False,
    )
    return [line for line in run.stdout.splitlines()
            if "independent_source_hammer.py" not in line]


def rederive_service_ledger(module) -> tuple[dict[str, int], str]:
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    for task_id in range(module.TASKS):
        chunk = (task_id // module.PARTITIONS) % module.CHUNKS
        row_count = min(
            module.ROW_TILE,
            module.ROWS_PER_PHASE - chunk * module.ROW_TILE,
        )
        receipt = module.M1016.common_receipt(task_id, row_count)
        counts.update(receipt["counts"])
        digest.update(canonical(receipt))
        if (task_id + 1) % module.TASKS_PER_SAMPLE == 0:
            sample = (task_id + 1) // module.TASKS_PER_SAMPLE - 1
            commit = {
                "task": module.TASKS + sample,
                "counts": {
                    resource: (module.COMMIT_CYCLES_PER_SAMPLE
                               if resource == "commit" else 0)
                    for resource in module.RESOURCES
                },
                "sample_commit": sample,
            }
            counts.update(commit["counts"])
            digest.update(canonical(commit))
    return dict(counts), digest.hexdigest()


def main() -> dict[str, Any]:
    before_namespaces = result_namespaces()
    before_processes = project_processes()
    module = load_source()

    identity = {
        "source_sha256": sha256(SOURCE),
        "checker_sha256": sha256(CHECKER),
        "tests_sha256": sha256(TESTS),
        "contract_sha256": sha256(CONTRACT),
        "docs359_sha256": sha256(DOCS359),
    }
    assert identity == {
        "source_sha256": SOURCE_SHA,
        "checker_sha256": CHECKER_SHA,
        "tests_sha256": TESTS_SHA,
        "contract_sha256": CONTRACT_SHA,
        "docs359_sha256": DOCS359_SHA,
    }

    # Author tests are diagnostic only; the verdict below is independently
    # derived and does not consume the author receipt.
    directed = subprocess.run(
        [sys.executable, str(TESTS)], text=True, capture_output=True, check=False
    )
    assert directed.returncode == 0 and "Ran 15 tests" in directed.stderr

    capacity = module.derive_physical_capacity()
    capacity_pass = (
        len(inspect.signature(module.derive_physical_capacity).parameters) == 0
        and capacity["psum"]["groups"] == 4
        and capacity["psum"]["bytes"] == 122_880
        and capacity["weight"]["bytes"] == 49_152
        and capacity["parent_plus_other"]["bytes"] == 42_880
        and capacity["derived_total_bytes"] == 214_912
        and capacity["derived_margin_bytes"] == 30_848
        and capacity["capacity_bytes_pass"] is True
        and rejected(lambda: module.derive_physical_capacity(0))
        and list(inspect.signature(module.replay_frozen_sample).parameters)
        == ["records"]
    )
    assert capacity_pass

    config = module.M1056.ArbiterConfig()
    config.validate()
    arbiter = module.M1056.small_oracle()
    arbiter_pass = (
        config.groups == 4 and config.ports_per_group == 1
        and config.port_mode == "1RW"
        and arbiter["different_address_same_port_conflict"] is True
        and arbiter["same_address_raw_enforced"] is True
        and arbiter["delay_cascades_to_next_task_and_commit"] is True
        and arbiter["cascade"]["nominal_cycles"] == 20
        and arbiter["cascade"]["arbitrated_cycles"] == 22
    )
    assert arbiter_pass

    service_counts, service_digest = rederive_service_ledger(module)
    assert module.TASKS == 812_160 and module.SAMPLES == 10
    assert service_counts == {
        "psum": 12_994_560,
        "weight": 70_853_184,
        "source": 51_840_000,
        "dma": 1_476_108,
        "commit": 960_000,
    }
    assert service_digest == SERVICE_DIGEST

    record0 = module.build_frozen_record(0, [1] * 64)
    record2 = module.build_frozen_record(2, [1] * 64)
    coverage = module.FrozenCoverage()
    empty_rejected = not coverage.proof()["full_coverage_pass"]
    coverage.consume(record0)
    partial_rejected = not coverage.proof()["full_coverage_pass"]
    duplicate_rejected = rejected(lambda: coverage.consume(record0))
    out_of_order_rejected = rejected(
        lambda: module.FrozenCoverage().consume(record2)
    )

    receipt = json.loads(json.dumps(record0.design_receipts["candidate"].common_receipt))
    boolean_receipt = json.loads(json.dumps(receipt))
    boolean_receipt["counts"]["dma"] = True
    extra_receipt = json.loads(json.dumps(receipt))
    extra_receipt["coverage"] = True
    receipt_attacks = {
        "boolean_count": rejected(
            lambda: module.validate_receipt_exact(boolean_receipt, 0, 64)
        ),
        "extra_key": rejected(
            lambda: module.validate_receipt_exact(extra_receipt, 0, 64)
        ),
        "duplicate_json_key": rejected(lambda: module.parse_receipt_json_for_attack(
            json.dumps(receipt)[:-1] + ',"task":0}', 0, 64
        )),
    }

    base = record0.design_receipts
    mismatch_attacks: dict[str, bool] = {}
    for field_name, bad_receipt in {
        "task_id": replace(base["strongest_zero"], task_id=1),
        "row": replace(base["strongest_zero"], row=1),
        "row_count": replace(base["strongest_zero"], row_count=63),
        "preprocess": replace(base["strongest_zero"], preprocess_cycles=999),
    }.items():
        values = dict(base)
        values["strongest_zero"] = bad_receipt
        mismatch_attacks[field_name] = rejected(
            lambda values=values: module.validate_frozen_record(
                replace(record0, design_receipts=values)
            )
        )
    bad_common = json.loads(json.dumps(base["strongest_zero"].common_receipt))
    bad_common["counts"]["dma"] += 1
    values = dict(base)
    values["strongest_zero"] = replace(
        base["strongest_zero"], common_receipt=bad_common
    )
    mismatch_attacks["common_receipt"] = rejected(
        lambda: module.validate_frozen_record(
            replace(record0, design_receipts=values)
        )
    )
    assert all(receipt_attacks.values()) and all(mismatch_attacks.values())

    # Contract attacks do not modify the canonical contract.  Exact-path and
    # exact-hash binding must reject all alternate payloads before parsing.
    fake_contract_rejections: dict[str, bool] = {}
    payloads = {
        "unsealed": {"status": "PASS_M1064_SEALED_CONTRACT_SOURCE_ONLY__M1065_REQUIRED_NO_LAUNCH"},
        "extra": {**json.loads(CONTRACT.read_text()), "extra": 1},
        "bool": {**json.loads(CONTRACT.read_text()), "max_attempts_now": False},
    }
    with tempfile.TemporaryDirectory(prefix="m1065_contract_attack.") as tmp:
        for name, value in payloads.items():
            path = Path(tmp) / (name + ".json")
            path.write_text(json.dumps(value), encoding="utf-8")
            fake_contract_rejections[name] = rejected(
                lambda path=path: module.validate_sealed_contract(path)
            )
        duplicate = Path(tmp) / "duplicate.json"
        duplicate.write_text('{"schema":1,"schema":2}', encoding="utf-8")
        fake_contract_rejections["duplicate_key"] = rejected(
            lambda: module.strict_json(duplicate)
        )
    assert all(fake_contract_rejections.values())

    # P0 attack: validate_frozen_record does not bind preprocess/work to the
    # frozen row mask.  All named identities and common receipts stay exact,
    # yet arbitrary cycle-driving values are accepted.
    forged_receipts = {}
    forged_work = {
        "candidate": 0,
        "strongest_zero": 999_999,
        "same_coordinate_bit": 999_999,
    }
    for design in module.DESIGNS:
        old = base[design]
        forged_receipts[design] = module.DesignTaskReceipt(
            task_id=old.task_id,
            row=old.row,
            row_count=old.row_count,
            preprocess_cycles=0,
            common_receipt=old.common_receipt,
            plan=module.M1056.TaskPlan(
                old.task_id, 0, forged_work[design], old.row
            ),
        )
    forged_record = replace(
        record0, preprocess_cycles=0, design_receipts=forged_receipts
    )
    manual_work_preprocess_forgery_accepted = not rejected(
        lambda: module.validate_frozen_record(forged_record)
    )

    # Independent second form: real first-row masks and all-zero caller masks
    # both validate.  They produce different work/preprocess, but coverage sees
    # byte-identical service receipts and digest state.
    with module.ROWS.open("rb") as stream:
        raw = stream.read(64 * module.BYTES_PER_LINE)
    real_masks = [int(line, 16) & 0xFFFF for line in raw.splitlines()]
    real_record = module.build_frozen_record(0, real_masks)
    zero_record = module.build_frozen_record(0, [0] * 64)
    module.validate_frozen_record(real_record)
    module.validate_frozen_record(zero_record)
    real_cov, zero_cov = module.FrozenCoverage(), module.FrozenCoverage()
    real_cov.consume(real_record)
    zero_cov.consume(zero_record)
    mask_forgery_accepted = (
        real_record.preprocess_cycles != zero_record.preprocess_cycles
        and any(
            real_record.design_receipts[name].plan.work_cycles
            != zero_record.design_receipts[name].plan.work_cycles
            for name in module.DESIGNS
        )
        and real_cov.services == zero_cov.services
        and all(
            real_cov.digests[name].hexdigest()
            == zero_cov.digests[name].hexdigest()
            for name in module.DESIGNS
        )
    )
    assert manual_work_preprocess_forgery_accepted and mask_forgery_accepted

    fields = set(module.FrozenTaskRecord.__dataclass_fields__)
    no_row_provenance = not ({"masks", "row_payload_sha256", "row_source_token"} & fields)
    replay_accepts_caller_records = (
        list(inspect.signature(module.replay_frozen_sample).parameters)
        == ["records"]
    )
    assert no_row_provenance and replay_accepts_caller_records

    after_namespaces = result_namespaces()
    after_processes = project_processes()
    no_full_execution = before_namespaces == after_namespaces == []
    assert no_full_execution

    return {
        "schema": "m1065_m1064_c1_frozen_exact_1rw_source_hammer_mechanical_v1",
        "status": "STOP_M1065_M1064_C1_FROZEN_EXACT_1RW_SOURCE_HAMMER",
        "identity": identity,
        "positive_checks": {
            "directed_tests": 15,
            "empty_partial_duplicate_out_of_order_rejected": all([
                empty_rejected, partial_rejected,
                duplicate_rejected, out_of_order_rejected,
            ]),
            "receipt_attacks_rejected": receipt_attacks,
            "three_design_mismatch_attacks_rejected": mismatch_attacks,
            "contract_attacks_rejected": fake_contract_rejections,
            "capacity_internally_derived": capacity_pass,
            "four_group_one_1rw_raw_cascade_preserved": arbiter_pass,
            "task_count": module.TASKS,
            "sample_commits": module.SAMPLES,
            "service_counts": service_counts,
            "service_digest_sha256": service_digest,
        },
        "p0_counterexample": {
            "manual_work_preprocess_forgery_accepted": manual_work_preprocess_forgery_accepted,
            "arbitrary_mask_record_accepted": mask_forgery_accepted,
            "real_preprocess_cycles": real_record.preprocess_cycles,
            "zero_mask_preprocess_cycles": zero_record.preprocess_cycles,
            "real_work_cycles": {
                name: real_record.design_receipts[name].plan.work_cycles
                for name in module.DESIGNS
            },
            "zero_mask_work_cycles": {
                name: zero_record.design_receipts[name].plan.work_cycles
                for name in module.DESIGNS
            },
            "forged_work_cycles": forged_work,
            "coverage_service_and_digest_identical_for_real_vs_zero_mask": True,
            "frozen_record_has_no_mask_or_row_provenance": no_row_provenance,
            "replay_api_accepts_caller_records": replay_accepts_caller_records,
        },
        "scope": {
            "source_modified": False,
            "full_iterator_called": False,
            "replay_frozen_sample_called": False,
            "full_51840000_replay_executed": False,
            "eda_gpu_remote_used": False,
            "result_namespaces_before": before_namespaces,
            "result_namespaces_after": after_namespaces,
            "project_processes_before": before_processes,
            "project_processes_after": after_processes,
            "docs359_modified": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))

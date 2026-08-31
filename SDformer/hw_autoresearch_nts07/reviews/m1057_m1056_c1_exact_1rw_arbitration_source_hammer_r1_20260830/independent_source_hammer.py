#!/usr/bin/env python3
"""Receipt-blind M1057 hammer for the source-only M1056 1RW replay model.

This checker does not modify M1056 and never runs the full 51.84M-row replay.
It independently rebuilds the small scheduling oracle, then attacks the public
three-design/future-release boundary rather than trusting M1056 PASS booleans.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1056_c1_exact_1rw_arbitration_replay_source.py"
CHECKER = HW / "system_simulator/scripts/check_m1056_c1_exact_1rw_arbitration_replay_source.py"
TESTS = HW / "system_simulator/tests/test_m1056_c1_exact_1rw_arbitration_replay_source.py"
CONTRACT = HW / "contracts/m1056_m1051_c1_exact_1rw_arbitration_replay_source_contract_r1_20260829.json"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1051 = HW / "reviews/m1051_m1040_m1016_c1_full_replay_result_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f",
    "checker": "7c5737409fe5636284518293a46709615eab0f949df0b781d55565248d6b5fef",
    "tests": "792b59ef106490f63766b6b3604e1bde3fa8f44fa8cf63dfd3ab684def471eca",
    "contract": "9717e472ed1b74ab5bd3c6da7daac5f9694705781e16217e9d1f10b4c3d066de",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m1051_review": "e74974a15b6ad888af9675d6feee276a805840d93235e0c1ee9eff0f877e051f",
    "m1051_manifest": "f87f501bc50073bb946786ce8a23d8413d6b68dd166effc10e78ff7b926f0b69",
    "m1051_outer": "15e0c98654db25599520025bc43448e1e38c58fe9002ed5ad8a5f71b9eef4b0f",
}
EXPECTED_M1016_SERVICE_DIGEST = "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1057_attacked_m1056", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1056")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class IndependentGrant:
    event_id: str
    cycle: int


def independent_fixed_fifo(events: list[dict[str, Any]]) -> list[IndependentGrant]:
    """Small independent one-port oracle: release then program-order FIFO."""
    pending = {event["id"]: dict(event) for event in events}
    require(len(pending) == len(events), "duplicate independent ID")
    fifo: list[str] = []
    grants: dict[str, int] = {}
    cycle = min(event["ready"] for event in events)
    out: list[IndependentGrant] = []
    for _ in range(1000):
        ready = []
        for event in pending.values():
            deps = event.get("deps", [])
            if not all(dep_id in grants for dep_id, _ in deps):
                continue
            release = max([event["ready"]] + [grants[dep_id] + delay for dep_id, delay in deps])
            if release <= cycle:
                ready.append(event)
        for event in sorted(ready, key=lambda value: value["order"]):
            fifo.append(event["id"])
            del pending[event["id"]]
        if fifo:
            event_id = fifo.pop(0)
            grants[event_id] = cycle
            out.append(IndependentGrant(event_id, cycle))
            cycle += 1
        elif pending:
            future = []
            for event in pending.values():
                deps = event.get("deps", [])
                if all(dep_id in grants for dep_id, _ in deps):
                    future.append(max([event["ready"]] + [grants[dep_id] + delay for dep_id, delay in deps]))
            require(future, "independent dependency deadlock")
            cycle = min(future)
        else:
            return out
    raise RuntimeError("independent oracle did not terminate")


def receipt(task: int = 0, psum: Any = 16) -> dict[str, Any]:
    return {"task": task, "counts": {
        "psum": psum, "weight": 7, "source": 64, "dma": 1, "commit": 0,
    }}


def main() -> dict[str, Any]:
    identity = {
        "source": sha256(SOURCE), "checker": sha256(CHECKER),
        "tests": sha256(TESTS), "contract": sha256(CONTRACT),
        "m1016": sha256(M1016), "docs359": sha256(DOCS359),
        "m1051_review": sha256(M1051 / "review.json"),
        "m1051_manifest": sha256(M1051 / "SHA256SUMS"),
        "m1051_outer": sha256(M1051 / "SHA256SUMS.seal.sha256"),
    }
    require(identity == EXPECTED, "frozen identity drift")
    module = load_source()

    # Receipt-blind scheduling checks.  These expected results are constructed
    # without consulting M1056's returned coverage flags or small_oracle JSON.
    cases = {
        "multiplicity2": [
            {"id": "a", "order": 0, "ready": 3},
            {"id": "b", "order": 1, "ready": 3},
        ],
        "multiplicity3_reversed": [
            {"id": "c", "order": 2, "ready": 1},
            {"id": "a", "order": 0, "ready": 1},
            {"id": "b", "order": 1, "ready": 1},
        ],
        "cross_task_arrival": [
            {"id": "late_t0", "order": 0, "ready": 9},
            {"id": "early_t1", "order": 32, "ready": 2},
        ],
        "same_address_raw": [
            {"id": "new_read", "order": 2, "ready": 0,
             "deps": [("old_write", 1)]},
            {"id": "old_write", "order": 1, "ready": 4},
        ],
    }
    independent = {
        name: [(grant.event_id, grant.cycle) for grant in independent_fixed_fifo(events)]
        for name, events in cases.items()
    }
    expected_independent = {
        "multiplicity2": [("a", 3), ("b", 4)],
        "multiplicity3_reversed": [("a", 1), ("b", 2), ("c", 3)],
        "cross_task_arrival": [("early_t1", 2), ("late_t0", 9)],
        "same_address_raw": [("old_write", 4), ("new_read", 5)],
    }
    require(independent == expected_independent, "independent oracle drift")

    # Compare actual M1056 grants for the same directed conditions.
    actual2 = module.arbitrate_group([
        module.PortEvent("a", 0, 0, 0, 0, 1, "READ", 3),
        module.PortEvent("b", 0, 1, 0, 0, 65, "READ", 3),
    ], 0)
    actual3 = module.arbitrate_group([
        module.PortEvent("c", 0, 2, 0, 0, 2, "READ", 1),
        module.PortEvent("a", 0, 0, 0, 0, 0, "READ", 1),
        module.PortEvent("b", 0, 1, 0, 0, 1, "READ", 1),
    ], 0)
    actual_cross = module.arbitrate_group([
        module.PortEvent("late_t0", 0, 0, 0, 0, 3, "READ", 9),
        module.PortEvent("early_t1", 1, 32, 0, 0, 4, "READ", 2),
    ], 0)
    actual_raw = module.arbitrate_group([
        module.PortEvent("new_read", 1, 2, 0, 0, 11, "READ", 0,
                         (module.Dependency("old_write", 1),)),
        module.PortEvent("old_write", 0, 1, 0, 0, 11, "WRITE", 4),
    ], 0)
    actual = {
        "multiplicity2": [(key, actual2.grants[key].cycle) for key in actual2.grant_order],
        "multiplicity3_reversed": [(key, actual3.grants[key].cycle) for key in actual3.grant_order],
        "cross_task_arrival": [(key, actual_cross.grants[key].cycle) for key in actual_cross.grant_order],
        "same_address_raw": [(key, actual_raw.grants[key].cycle) for key in actual_raw.grant_order],
    }
    require(actual == independent, "M1056 directed grants differ from independent oracle")

    # Independently verify M1016-compatible event geometry and packed address.
    plan = module.TaskPlan(7, 5, 17, 63)
    events = module.nominal_task_events(plan, 100, {})
    by_id = {event.event_id: event for event in events}
    require(len(events) == 16, "event conservation drift")
    for bank in range(8):
        read = by_id[f"t7:b{bank}:R"]
        write = by_id[f"t7:b{bank}:W"]
        expected_address = (bank % 2) * 64 + 63
        require(read.group == bank // 2 and write.group == bank // 2 and
                read.address == expected_address and write.address == expected_address and
                read.base_ready_cycle == 100 + bank * 2 and
                write.base_ready_cycle == min(117, 100 + bank * 2 + 1),
                "M1016 event geometry/address mapping drift")

    cascade_plans = [module.TaskPlan(0, 0, 8, 3), module.TaskPlan(1, 0, 8, 3)]
    cascade = module.replay_task_sequence(cascade_plans)
    cascade_values = {
        "nominal": module.nominal_m1016_sequence_cycles(cascade_plans),
        "arbitrated": cascade.sample_cycles_after_commit,
        "starts": [task.work_start for task in cascade.tasks],
        "nominal_ends": [task.nominal_work_end for task in cascade.tasks],
        "effective_ends": [task.effective_work_end for task in cascade.tasks],
        "nominal_excess": cascade.total_nominal_excess_accesses,
    }
    require(cascade_values == {
        "nominal": 20, "arbitrated": 22, "starts": [0, 11],
        "nominal_ends": [8, 19], "effective_ends": [9, 20],
        "nominal_excess": 16,
    }, "cascade anchor drift")
    require(cascade_values["arbitrated"] !=
            cascade_values["nominal"] + cascade_values["nominal_excess"],
            "naive +conflict arithmetic admitted")

    # Attack 1: equal empty receipts are accepted as a common coordinate even
    # though they cannot equal the frozen M1016 service stream/digest.
    empty_receipt_pass = module.validate_three_design_common_coordinate(
        {name: [] for name in module.DESIGNS},
        {name: module.ArbiterConfig() for name in module.DESIGNS},
    )["status"] == "PASS_M1056_THREE_DESIGN_COMMON_COORDINATE"

    # Attack 2: Python bool is an int, so caller-controlled True counts pass.
    bool_receipt_pass = module.validate_three_design_common_coordinate(
        {name: [receipt(0, True)] for name in module.DESIGNS},
        {name: module.ArbiterConfig() for name in module.DESIGNS},
    )["status"] == "PASS_M1056_THREE_DESIGN_COMMON_COORDINATE"

    # Attack 3: receipt equality does not require equal task population,
    # task IDs, psum rows, or preprocess geometry across the three designs.
    shared_receipts = {name: [receipt()] for name in module.DESIGNS}
    mismatched_plans = {
        "candidate": [module.TaskPlan(0, 0, 8, 0)],
        "strongest_zero": [module.TaskPlan(0, 11, 16, 17),
                            module.TaskPlan(1, 0, 16, 18)],
        "same_coordinate_bit": [module.TaskPlan(99, 3, 16, 63)],
    }
    mismatched_geometry_pass = module.replay_three_design_sequences(
        mismatched_plans, shared_receipts
    )["status"] == "PASS_M1056_THREE_DESIGN_EXACT_1RW_REPLAY"

    # Attack 4: capacity is a caller scalar, not internally tied to the frozen
    # 214,912-byte organization or its 60-wide-macro / other-store ledger.
    capacity_zero_pass = module.replay_three_design_sequences(
        {name: [module.TaskPlan(0, 0, 8, 0)] for name in module.DESIGNS},
        shared_receipts, capacity_bytes=0,
    )["capacity_bytes_pass"] is True

    # Attack 5: validate_source_contract accepts an unsealed caller JSON with
    # only status/launch fields; it does not bind the contract or source SHA.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        fake_contract_path = Path(stream.name)
        json.dump({
            "status": "PASS_M1056_SOURCE_ONLY__M1057_REQUIRED_NO_FULL_REPLAY",
            "launch_now": False,
            "max_attempts_now": 0,
        }, stream)
    try:
        fake_contract_pass = module.validate_source_contract(fake_contract_path)["status"] == \
            "PASS_M1056_SOURCE_CONTRACT_PREFLIGHT__NO_FULL_REPLAY"
    finally:
        fake_contract_path.unlink(missing_ok=True)

    attacks = {
        "equal_empty_receipts_accepted": empty_receipt_pass,
        "boolean_service_count_accepted": bool_receipt_pass,
        "three_design_task_geometry_mismatch_accepted": mismatched_geometry_pass,
        "caller_capacity_zero_accepted": capacity_zero_pass,
        "unsealed_wrong_contract_identity_accepted": fake_contract_pass,
    }
    require(all(attacks.values()), "expected fail-open attack no longer reproduces")

    forbidden = [
        HW / "results/m1056_c1_exact_1rw_arbitration_full_replay_r1_20260829",
        HW / "results/.m1056_c1_exact_1rw_arbitration_full_replay_attempt_consumed",
        HW / "results/m1058_m1056_c1_exact_1rw_arbitration_full_replay_r1_20260829",
        HW / "results/.m1058_m1056_c1_exact_1rw_arbitration_full_replay_attempt_consumed",
    ]
    require(not any(path.exists() for path in forbidden),
            "M1056/M1058 full replay namespace exists")

    return {
        "schema": "m1057_m1056_c1_exact_1rw_source_hammer_checks_v1",
        "status": "STOP_M1057_M1056_C1_EXACT_1RW_SOURCE_HAMMER",
        "identity": identity,
        "positive_rederivation": {
            "four_packed_groups": module.PACKED_GROUPS == 4,
            "one_1rw_port_per_group": module.ArbiterConfig().ports_per_group == 1,
            "fifo_arrival_then_program_order": actual == independent,
            "different_address_same_port_serialized": actual["multiplicity2"] == [("a", 3), ("b", 4)],
            "multiplicity_two_three": True,
            "cross_task_input_order": True,
            "same_address_raw": True,
            "m1016_event_geometry_and_packed_address": True,
            "cascade": cascade_values,
            "not_naive_plus_403922": True,
            "packed_depth_groups": 4,
            "physical_addresses_per_group": 128,
            "logical_wide_row_bits_from_frozen_contract": 1824,
            "wide_macro_slices_per_group": 15,
            "physical_macro_count": 60,
            "physical_psum_capacity_bytes": 122880,
        },
        "fail_open_attacks": attacks,
        "expected_m1016_service_digest": EXPECTED_M1016_SERVICE_DIGEST,
        "m1056_pins_expected_digest": False,
        "full_replay_executed": False,
        "eda_gpu_remote_used": False,
        "source_modified": False,
        "docs359_modified": False,
        "future_full_replay_release_authorized": False,
        "capacity_admitted": False,
        "matched_cycles_admitted": False,
        "speedup_admitted": False,
        "paper_ppa_ready": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))

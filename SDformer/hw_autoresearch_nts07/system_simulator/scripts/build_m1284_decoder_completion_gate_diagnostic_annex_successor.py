#!/usr/bin/env python3
"""M1284 additive successor closing M1283's three M1278 P1 findings."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
SOURCE_FILE = Path(__file__).resolve()
PREDECESSOR = HW / "system_simulator/scripts/build_m1278_decoder_completion_gate_and_diagnostic_annex.py"
PREDECESSOR_SHA256 = "52c0829927fb32211df86e0781049f202b2ed63297b3743f121267a6bfa5471d"
PREDECESSOR_CONTRACT = HW / "contracts/m1278_decoder_completion_gate_diagnostic_annex_source_contract_r1_20260830.json"
PREDECESSOR_CONTRACT_SHA256 = "6987400c9adc638905675f1b1c3794095ec0ed2d63b887efadefa35ab105edfb"
CONTRACT = HW / "contracts/m1284_decoder_completion_gate_diagnostic_annex_successor_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ANNEX_NAME = "m1284_h67_ep35_decoder_only_diagnostic_annex_r1_20260830"
ANNEX_SCHEMA = "m1284_h67_ep35_decoder_only_diagnostic_annex_r1_v1"
ANNEX_STATUS = "PASS_M1284_EP35_DECODER_DIAGNOSTIC_ONLY__RESULT_HAMMER_REQUIRED"


class SuccessorError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise SuccessorError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = Path(path).lstat().st_mode
    require(stat.S_ISREG(mode) and not Path(path).is_symlink() and sha256(path) == expected,
            label + " identity drift")


def load_predecessor():
    regular(PREDECESSOR, PREDECESSOR_SHA256, "M1278 predecessor")
    regular(PREDECESSOR_CONTRACT, PREDECESSOR_CONTRACT_SHA256, "M1278 contract")
    name = "m1284_frozen_m1278"
    spec = importlib.util.spec_from_file_location(name, PREDECESSOR)
    require(spec is not None and spec.loader is not None, "cannot import M1278")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load_predecessor()


def canonical_layout():
    old = P.canonical_layout()
    return P.Layout(old.parent, old.result, old.attempt, old.lock, old.work,
                    old.parent / ANNEX_NAME)


def exact_int(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, label + " must be exact integer")
    return value


def exact_bool(value: Any, expected: bool, label: str) -> None:
    require(type(value) is bool and value is expected, label + " must be exact boolean")


def verify_attempt_types(layout, runner) -> None:
    receipt = P.strict_json(layout.attempt / "attempt.json")
    exact_int(receipt.get("maximum_attempts"), "attempt maximum_attempts", 1)
    require(receipt["maximum_attempts"] == 1, "attempt maximum must equal one")
    exact_bool(receipt.get("automatic_retry"), False, "attempt automatic_retry")
    exact_bool(receipt.get("canonical_payload_opened_before_attempt"), False,
               "attempt canonical_payload_opened_before_attempt")
    seal = runner.verify_atomic_seal(layout.attempt)
    exact_int(seal.get("members"), "attempt seal members", 1)
    require(seal["members"] == 1, "attempt seal must contain one payload")


def verify_call_counter_types(rows: list[dict[str, Any]]) -> None:
    exact_int(len(rows), "row count", 1)
    require(len(rows) == 120, "complete gate requires 120 rows")
    for index, row in enumerate(rows):
        for key in ("global_call_ordinal", "sequence_ordinal", "sequence_sample_id",
                    "module_ordinal", "transaction_ordinal_first",
                    "transaction_ordinal_last", "transaction_count", "cycle_start",
                    "cycle_end", "diagnostic_cycles"):
            exact_int(row.get(key), "row %d %s" % (index, key))
        exact_bool(row.get("d1_exact_theta"), index % 4 == 1,
                   "row %d d1_exact_theta" % index)
        exact_bool(row.get("d1_weight_folding"), False,
                   "row %d d1_weight_folding" % index)
        for key, value in row["diagnostic_traffic_bytes"].items():
            exact_int(value, "row %d traffic %s" % (index, key))
        for kind, summary in row["kind_summaries"].items():
            exact_int(summary["count"], "row %d %s count" % (index, kind))
            exact_int(summary["traffic_bytes"], "row %d %s bytes" % (index, kind))
            for name, count in summary["stall_events"].items():
                exact_int(count, "row %d %s stall %s" % (index, kind, name))
        for key, expected in (("diagnostic_only", True), ("speedup_admitted", False),
                              ("system_speedup_admitted", False),
                              ("paper_ppa_ready", False),
                              ("final_checkpoint_rebind_required", True)):
            exact_bool(row["claim_boundary"].get(key), expected,
                       "row %d claim %s" % (index, key))


def verify_result_counter_types(checked: dict[str, Any]) -> None:
    payload = checked["payload"]
    population = payload["population"]
    for key in ("calls", "timesteps_per_call", "transaction_count"):
        exact_int(population.get(key), "population " + key, 1)
    require(population["calls"] == 120 and population["timesteps_per_call"] == 10,
            "population counts drift")
    exact_int(payload["diagnostic"].get("cycles"), "diagnostic cycles", 1)
    for key, value in payload["diagnostic"]["traffic_bytes"].items():
        exact_int(value, "diagnostic traffic " + key)
    exact_int(checked.get("call_rows"), "checked call_rows", 1)
    exact_int(checked.get("transactions"), "checked transactions", 1)
    exact_int(checked.get("cycles"), "checked cycles", 1)
    exact_int(checked["seal"].get("members"), "result seal members", 1)
    require(checked["seal"]["members"] == 3, "result seal member count drift")
    claim = payload["claim_boundary"]
    expected_claim = {"decoder_only": True, "address_timed_transactions_complete": True,
        "same_resource_schedule_complete": True, "diagnostic_cycles_only": True,
        "diagnostic_traffic_only": True, "speedup_admitted": False,
        "system_speedup_admitted": False, "paper_ppa_ready": False,
        "paper_citable_performance": False, "final_checkpoint_rebind_required": True,
        "independent_result_hammer_required": True}
    require(set(claim) == set(expected_claim), "result claim key drift")
    for key, expected in expected_claim.items():
        exact_bool(claim[key], expected, "result claim " + key)


def validate_complete_gate(layout, runner, gate: dict[str, Any]) -> dict[str, Any]:
    require(type(gate) is dict and gate.get("state") == "COMPLETE",
            "completed gate state required")
    exact_bool(gate.get("published"), True, "gate published")
    exact_bool(gate.get("replay"), False, "gate replay")
    require(Path(gate.get("source_result")) == layout.result,
            "gate source-result path drift")
    verify_attempt_types(layout, runner)
    require(type(gate.get("checked")) is dict, "gate checked projection must be object")
    require(type(gate.get("rows")) is list, "gate row projection must be array")
    # Validate the supplied projection before Python's bool/int-coercing equality can
    # compare it with a canonical recomputation (False == 0 and True == 1).
    verify_result_counter_types(gate["checked"])
    verify_call_counter_types(gate["rows"])
    checked = runner.validate_publish_candidate(layout.result)
    require(checked == gate.get("checked"), "gate checked projection drift")
    rows = P.read_rows(layout.result / P.CALLS, runner, full=True)
    require(rows == gate.get("rows"), "gate row projection drift")
    verify_result_counter_types(checked)
    verify_call_counter_types(rows)
    return {"state": "COMPLETE", "published": True, "replay": False,
            "source_result": layout.result, "checked": checked, "rows": rows}


def _capability_closure():
    key = object()
    issued: set[object] = set()

    class Capability:
        __slots__ = ("gate", "key", "nonce", "consumed")
        def __init__(self, gate, supplied):
            require(supplied is key, "private capability constructor")
            self.gate = copy.deepcopy(gate); self.key = supplied
            self.nonce = object(); issued.add(self.nonce); self.consumed = False

    def complete(layout, runner,
                 alive: Callable[[int], bool] = P.pid_alive,
                 cmdline: Callable[[int], bytes] = P.pid_cmdline):
        gate = P.completion_gate(layout, runner, alive=alive, cmdline=cmdline)
        if gate.get("state") != "COMPLETE":
            raise P.Incomplete("M1284 incomplete; no capability and no output")
        return Capability(validate_complete_gate(layout, runner, gate), key)

    def consume(value):
        require(type(value) is Capability and value.key is key and
                value.nonce in issued and
                type(value.consumed) is bool and value.consumed is False,
                "valid unused completion capability required")
        issued.remove(value.nonce)
        value.consumed = True
        return copy.deepcopy(value.gate)

    return complete, consume


completion_capability, _consume_capability = _capability_closure()


def expected_identity(runner) -> dict[str, Any]:
    return {"checkpoint": "H67_ep35",
        "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
        "source_sha256": runner.SOURCE_SHA, "contract_sha256": runner.CONTRACT_ID[0],
        "m1110d_outer_seal_file_sha256": runner.M1110D_ID[2],
        "final_checkpoint_rebind_required": True}


def build_annex(layout, runner, gate: dict[str, Any]) -> dict[str, Any]:
    payload = P.annex_payload(gate)
    payload["schema"] = ANNEX_SCHEMA
    payload["status"] = ANNEX_STATUS
    validate_annex(layout, runner, gate["checked"], payload)
    return payload


def validate_annex(layout, runner, checked: dict[str, Any], payload: dict[str, Any]) -> None:
    expected_keys = {"schema", "status", "source_result", "identity", "population",
                     "common_resource", "diagnostic", "module_breakdown",
                     "sequence_breakdown", "claim_boundary"}
    require(type(payload) is dict and set(payload) == expected_keys,
            "annex top-level schema drift")
    require(payload["schema"] == ANNEX_SCHEMA and payload["status"] == ANNEX_STATUS,
            "annex schema/status drift")
    require(payload["identity"] == expected_identity(runner),
            "annex is not exact ep35 identity")
    require(payload["population"] == checked["payload"]["population"] and
            payload["common_resource"] == checked["payload"]["common_resource"] and
            payload["diagnostic"] == checked["payload"]["diagnostic"],
            "annex source projection drift")
    source = payload["source_result"]
    try:
        expected_source_path = str(layout.result.relative_to(REPO))
    except ValueError:
        expected_source_path = str(layout.result)
    require(set(source) == {"path", "payload_sha256", "call_schedule_sha256", "atomic_seal"} and
            source["path"] == expected_source_path and
            source["payload_sha256"] == sha256(layout.result / P.PAYLOAD) and
            source["call_schedule_sha256"] ==
                checked["payload"]["population"]["call_schedule_sha256"] and
            source["atomic_seal"] == checked["seal"], "annex source identity drift")
    claim = payload["claim_boundary"]
    expected_claim = {"ep35_only": True, "decoder_only": True,
        "diagnostic_only": True, "final_checkpoint_rebind_required": True,
        "ratio_or_speedup": False, "table_a": False, "full_network": False,
        "system_speedup": False, "energy": False, "ppa": False,
        "paper_headline": False, "independent_result_hammer_required": True}
    require(set(claim) == set(expected_claim), "annex claim key drift")
    for key, expected in expected_claim.items():
        exact_bool(claim[key], expected, "annex claim " + key)
    exact_int(len(payload["module_breakdown"]), "module row count", 1)
    exact_int(len(payload["sequence_breakdown"]), "sequence row count", 1)
    require(len(payload["module_breakdown"]) == 4 and
            len(payload["sequence_breakdown"]) == 3, "annex breakdown population drift")
    for row in payload["module_breakdown"]:
        exact_int(row["calls"], "module calls", 1)
        exact_int(row["diagnostic_cycles"], "module cycles", 1)
        for value in row["diagnostic_traffic_bytes"].values():
            exact_int(value, "module traffic")
    for row in payload["sequence_breakdown"]:
        exact_int(row["calls"], "sequence calls", 1)
        exact_int(row["cycles"], "sequence cycles", 1)


def publish_with_capability(layout, runner, capability):
    gate = _consume_capability(capability)
    gate = validate_complete_gate(layout, runner, gate)
    payload = build_annex(layout, runner, gate)
    validate_annex(layout, runner, gate["checked"], payload)
    return P.publish_annex(layout, payload)


def verify_static_authorities() -> None:
    regular(DOCS359, DOCS359_SHA256, "docs/359")
    contract = P.strict_json(CONTRACT)
    require(contract.get("schema") ==
            "m1284_decoder_completion_gate_diagnostic_annex_successor_source_contract_r1_v1",
            "M1284 contract schema drift")
    require(contract.get("source", {}).get("path") == str(SOURCE_FILE.relative_to(REPO)) and
            contract["source"]["sha256"] == sha256(SOURCE_FILE),
            "M1284 source binding drift")


def main() -> int:
    require(len(sys.argv) == 1, "M1284 accepts zero arguments")
    try:
        verify_static_authorities()
        runner = P.load_runner()
        capability = completion_capability(canonical_layout(), runner)
        result = publish_with_capability(canonical_layout(), runner, capability)
        print(json.dumps(result, sort_keys=True))
        return 0
    except P.Incomplete:
        sys.stderr.write("M1284_INCOMPLETE__NO_CAPABILITY_NO_OUTPUT_NO_REPLAY\n")
        return 75
    except BaseException as exc:
        sys.stderr.write("M1284_FAIL_CLOSED__NO_OUTPUT_NO_REPLAY: %s\n" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

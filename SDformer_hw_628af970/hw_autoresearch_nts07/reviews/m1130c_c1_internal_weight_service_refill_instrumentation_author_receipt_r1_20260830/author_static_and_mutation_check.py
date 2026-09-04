#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-side bounded checks for M1130C; different-author hammer remains required."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
CONTRACT = HW / "contracts/m1130c_c1_internal_weight_service_refill_instrumentation_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf",
    "contract": "20ff9026f8dbc25ad0e9813107a6e97a96f1e379244dcb26ffb51d3a972bcfab",
    "contract_side": "49c2e9599a2c87807717f87f7c117844ad056cefe660c71bffb564d0413de745",
    "contract_outer": "efc4bc08d3634531b99c1e45d1ce20c362bb5ca74249d9f2a6877b857af9352a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0
attacks: dict[str, str] = {}


class CheckFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise CheckFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == digest,
            "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing function: " + name)
    return node


def analyze(source: str, contract: dict[str, Any]) -> None:
    tree = ast.parse(source)
    audit = function_node(tree, "audit_frozen_internal_event_point")
    iterator = function_node(tree, "iter_canonical_internal_weight_service_refill_events")
    consume = function_node(tree, "instrument_real_event_inputs")
    oracle = function_node(tree, "source_small_oracle")
    audit_text = ast.get_source_segment(source, audit) or ast.unparse(audit)
    iterator_text = ast.get_source_segment(source, iterator) or ast.unparse(iterator)
    consume_text = ast.get_source_segment(source, consume) or ast.unparse(consume)
    oracle_text = ast.get_source_segment(source, oracle) or ast.unparse(oracle)
    require("CanonicalRowReader" not in source and
            "iter_canonical_full_replay_results(" not in source,
            "source may open canonical rows")
    require('audit["canonical_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_ready"] is True') < iterator_text.index("yield {}"),
            "STOP is not before yield")
    for token in ('"canonical_row_reader_opened": False', '"canonical_rows_read": 0',
                  '"canonical_events_emitted": 0', '"canonical_ready": False',
                  '"aggregate_expansion_allowed": False'):
        require(token in audit_text, "zero-event boundary drift: " + token)
    require('self.weight_runs.append((start, end, half_slot))' in audit_text and
            '"m1056_m1102_port_events": "psum only"' in audit_text,
            "upstream event-point audit drift")
    for field in contract["minimum_upstream_event_fields"]:
        require(field in source, "minimum event field absent from source: " + field)
    require('type(event) is InternalWeightServiceRefillEvent' in consume_text and
            "expected_beats" in consume_text and "seen_exact_ids" in consume_text and
            "schedule_native_one_rw" in consume_text and
            "validate_exact_once_and_conflicts" in consume_text,
            "direct-event instrumentation mechanics drift")
    require(oracle_text.count("InternalWeightServiceRefillEvent(") == 3 and
            '"events": 9' in oracle_text and '"writes": 6' in oracle_text and
            '"reads": 3' in oracle_text and
            '"explicitly_stalled_transactions": 3' in oracle_text,
            "bounded oracle drift")
    require(contract["status"] ==
            "SOURCE_ONLY_UPSTREAM_EVENT_OBJECT_ABSENT__CANONICAL_STOP__SYNTHETIC_INSTRUMENTATION_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "contract status drift")
    require(contract["frozen_upstream_audit"]["real_per_beat_weight_event_object_available"] is False and
            contract["frozen_upstream_audit"]["canonical_ready"] is False and
            contract["direct_event_rule"]["all_fields_must_be_producer_supplied"] is True and
            contract["direct_event_rule"]["count_weight_beat_first_interval_or_capacity_inference_forbidden"] is True,
            "contract event boundary drift")
    auth = contract["authorization"]
    require(auth["different_author_static_hammer_only"] is True and
            all(auth[key] is False for key in (
                "modify_frozen_m1102_or_m1016_now", "canonical_row_open_now",
                "full_51840000_replay_now", "runner_now", "eda_rtl_gpu_remote_now",
                "performance_or_energy_now")), "contract authorization escalation")
    boundary = contract["claim_boundary"]
    require(boundary["event_input_interface_and_synthetic_mechanics_only"] is True and
            all(value is False for key, value in boundary.items()
                if key != "event_input_interface_and_synthetic_mechanics_only"),
            "claim escalation")


def load_subject():
    spec = importlib.util.spec_from_file_location("m1130c_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def main() -> None:
    before = {path: sha256(path) for path in (SOURCE, CONTRACT, DOCS359)}
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    verify_regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    contract = strict_json(CONTRACT)
    source = SOURCE.read_text(encoding="utf-8")
    analyze(source, contract)
    module = load_subject()
    result = module.source_small_oracle()
    require(result["status"] ==
            "PASS_SYNTHETIC_DIRECT_EVENT_INSTRUMENTATION__CANONICAL_STOP" and
            result["synthetic"] == {
                "events": 9, "writes": 6, "reads": 3,
                "unique_exact_once_write_ids": 6,
                "explicitly_stalled_transactions": 3,
                "final_native_1rw_conflicts": 0,
                "final_weight_half_slot_overlaps": 0,
            } and result["canonical_iterator_stopped_before_row_open"] is True,
            "bounded runtime result drift")
    digest = "0" * 64
    event = module.InternalWeightServiceRefillEvent(
        "candidate", 0, 0, 5, "WRITE", 0, 0, 0, 0, tuple(range(8)), 128,
        (0xffff,) * 8, 8, 0, 0, module.exact_once_id("candidate", 0, 0, 0, 0), digest)
    rejected("wrong_op", lambda: replace(event, op="READ").validate())
    rejected("wrong_bank", lambda: replace(event, logical_bank=1).validate())
    rejected("wrong_local_row", lambda: replace(event, local_row=1).validate())
    rejected("wrong_slices", lambda: replace(event, native_slices=tuple(range(1, 9))).validate())
    rejected("wrong_bytes", lambda: replace(event, bytes=127).validate())
    rejected("wrong_byte_enable", lambda: replace(event, byte_enable_per_slice=(0xfffe,) * 8).validate())
    rejected("wrong_activation", lambda: replace(event, native_macro_activations=7).validate())
    rejected("wrong_exact_once_id", lambda: replace(event, service_event_exact_once_id="0" * 64).validate())
    rejected("duplicate_exact_once_id", lambda: module.instrument_real_event_inputs([
        event, replace(event, source_local_ordinal=1, store_transaction_ordinal=1)]))
    rejected("aggregate_not_event_type", lambda: module.instrument_real_event_inputs([
        {"count": 1, "weight_beat_first": 0, "half_slot": 0}]))
    mutated = copy.deepcopy(contract); mutated["authorization"]["full_51840000_replay_now"] = True
    rejected("contract_full_replay", lambda: analyze(source, mutated))
    mutated = copy.deepcopy(contract); mutated["claim_boundary"]["h67_transactions"] = True
    rejected("contract_h67_claim", lambda: analyze(source, mutated))
    rejected("source_canonical_ready", lambda: analyze(
        source.replace('"canonical_ready": False', '"canonical_ready": True', 1), contract))
    require(before == {path: sha256(path) for path in before},
            "author check modified subject or docs359")
    print(json.dumps({
        "schema": "m1130c_author_static_and_mutation_checks_v1",
        "status": "PASS_AUTHOR_SOURCE_CONTRACT_AND_BOUNDED_SYNTHETIC__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "checks": checks,
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "synthetic": result["synthetic"],
        "canonical_rows": 0,
        "canonical_events": 0,
        "full_51840000_replay": False,
        "eda_rtl_gpu_remote": False,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

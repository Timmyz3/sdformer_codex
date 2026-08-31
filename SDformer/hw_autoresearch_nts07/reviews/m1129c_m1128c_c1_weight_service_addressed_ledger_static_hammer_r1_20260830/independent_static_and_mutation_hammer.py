#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author bounded hammer for M1128C; never opens canonical rows."""
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
SOURCE = HW / "system_simulator/scripts/build_m1128c_c1_weight_service_addressed_ledger_source.py"
CONTRACT = HW / "contracts/m1128c_c1_weight_service_addressed_ledger_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1128c_c1_weight_service_addressed_ledger_author_receipt_r1_20260830"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "d25f9e4fdfda62f56e7efb120fe0c8f6108a4b23ba4eee712e3ec471b5fa493e",
    "contract": "69bcc952953a23d102ac021e2b67375ef0d539b47bf88c347081200fae1b9102",
    "contract_side": "f132061ad02a122b939cc3b1cad150b3acb4efda3d72c16d8027fd12b2c101e0",
    "contract_outer": "bb8eca6f7dd02546a9d8aed009e44212c89ed9fe90376ce83306128133786166",
    "author_review": "b4bd360904e99dc8b0457d3d07ead95ad0f529e96e73d2f3d3f1bc2fd8dc0300",
    "author_manifest": "248f908a0f9662dc5836cce8e447cbd6758dbd41d82e2c6704cc65d98be49b9d",
    "author_outer": "ccb5ac5836271577c95021d2afee63aa8300a873771a2e49eafabb0e439babd0",
    "m1102": "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == digest,
            "regular identity drift: " + str(path))


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "authority directory drift")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, review_sha)
    verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_sha)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        require(relative not in listed and "/" not in relative and relative not in (".", ".."),
                "bad manifest member")
        verify_regular(directory / relative, digest)
        listed.add(relative)
    expected, relative = outer.read_text(encoding="utf-8").split()
    require(relative == "SHA256SUMS" and expected == sha256(manifest), "outer drift")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    require(actual == listed | {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "live extra or missing sealed member")
    require(not any(path.is_symlink() for path in directory.iterdir()), "sealed symlink")


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing function: " + name)
    return node


def method_node(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    klass = next((item for item in tree.body
                  if isinstance(item, ast.ClassDef) and item.name == class_name), None)
    require(klass is not None, "missing class: " + class_name)
    node = next((item for item in klass.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing method: " + class_name + "." + name)
    return node


def calls(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Attribute):
                names.append(target.attr)
    return names


def analyze_subject(source_text: str, contract: dict[str, Any]) -> None:
    tree = ast.parse(source_text)
    audit = function_node(tree, "audit_frozen_service_interface")
    iterator = function_node(tree, "iter_canonical_weight_addressed_ledger")
    oracle = function_node(tree, "source_small_oracle")
    iterator_text = ast.get_source_segment(source_text, iterator) or ast.unparse(iterator)
    audit_text = ast.get_source_segment(source_text, audit) or ast.unparse(audit)
    oracle_text = ast.get_source_segment(source_text, oracle) or ast.unparse(oracle)
    require("CanonicalRowReader" not in source_text and
            "iter_canonical_full_replay_results(" not in source_text,
            "M1128C may open canonical rows")
    require(calls(iterator) == ["audit_frozen_service_interface", "require"],
            "canonical iterator call graph drift")
    require('audit["canonical_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_ready"] is True') < iterator_text.index("yield {}"),
            "canonical STOP is not before yield")
    for token in (
        '"canonical_row_reader_opened": False',
        '"full_51840000_rows_read": False',
        '"canonical_weight_transactions_emitted": 0',
        '"canonical_ready": False',
        '"count_or_weight_beat_first_expansion_allowed": False',
        '"capacity_geometry_expansion_allowed": False',
    ):
        require(token in audit_text, "canonical zero-boundary drift: " + token)
    for missing in (
        "native READ/WRITE operation", "logical weight bank", "native slice set",
        "local row", "bytes and byte enable", "native macro activation multiplicity",
        "service-beat to store-transaction exact-once relation",
    ):
        require(missing in audit_text, "missing native gap not explicit: " + missing)
    require("NATIVE_SLICES = 24" in source_text and "NATIVE_DEPTH = 128" in source_text and
            "SLICE_BYTES = 16" in source_text and "ROWS_PER_HALF = 16" in source_text and
            "REFILL_SLICES = 8" in source_text,
            "synthetic 24x128x128b geometry drift")
    require(oracle_text.count("AddressedWeightTransaction(") == 3 and
            '"transactions": 9' in oracle_text and
            '"refill_store_transactions": 6' in oracle_text and
            '"full_record_read_transactions": 3' in oracle_text and
            '"explicitly_stalled_transactions": 3' in oracle_text and
            '"final_native_1rw_conflicts": 0' in oracle_text and
            '"final_weight_half_slot_overlaps": 0' in oracle_text,
            "bounded synthetic oracle drift")
    require(contract["status"] ==
            "SOURCE_ONLY_CANONICAL_STOP_FROZEN_WEIGHT_EVENT_FIELDS_INSUFFICIENT__SYNTHETIC_MAPPING_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "contract status drift")
    frozen = contract["frozen_actual_service_interface"]
    require(frozen["canonical_ready"] is False and
            frozen["count_or_weight_beat_first_may_be_expanded"] is False and
            all(frozen["missing"].values()), "contract native-gap boundary drift")
    require(contract["fail_closed_behavior"] == {
        "canonical_iterator_stops_before_canonical_row_reader_open": True,
        "canonical_rows_read": 0,
        "canonical_weight_transactions_emitted": 0,
        "missing_fields_not_inferred_from_count": True,
        "missing_fields_not_inferred_from_weight_beat_first": True,
        "missing_fields_not_inferred_from_capacity_geometry": True,
        "synthetic_mapping_never_labeled_canonical": True,
    }, "contract fail-closed behavior drift")
    authorization = contract["authorization"]
    require(authorization["different_author_static_hammer_only"] is True and
            all(authorization[key] is False for key in (
                "canonical_export_now", "canonical_row_open_now",
                "full_51840000_replay_now", "runner_now",
                "eda_rtl_gpu_remote_now", "performance_or_energy_now")),
            "contract authorization escalation")
    boundary = contract["claim_boundary"]
    require(boundary["source_schema_and_synthetic_mechanics_only"] is True and
            all(value is False for key, value in boundary.items()
                if key != "source_schema_and_synthetic_mechanics_only"),
            "contract claim escalation")


def audit_real_m1016_interface(source_text: str) -> dict[str, Any]:
    tree = ast.parse(source_text)
    common = function_node(tree, "common_receipt")
    weight = method_node(tree, "PackingAudit", "weight_task")
    run_full = function_node(tree, "run_full")
    common_text = ast.get_source_segment(source_text, common) or ast.unparse(common)
    run_text = ast.get_source_segment(source_text, run_full) or ast.unparse(run_full)
    require([item.arg for item in weight.args.args] == ["self", "start", "beats", "half_slot"],
            "real PackingAudit.weight_task API drift")
    for field in ("task", "counts", "source_address_first", "source_address_count",
                  "weight_beat_first", "dma_first", "psum_addresses", "commit_first"):
        require(('"' + field + '"') in common_text, "real receipt lost field: " + field)
    require('receipt["counts"]["weight"]' in run_text and "index & 1" in run_text and
            "global_offsets[design] + start - preprocess" in run_text,
            "real weight_task call relation drift")
    forbidden = ("native_slice", "logical_bank", "local_row", "byte_enable",
                 "native_macro_activations", "store_transaction_ordinal")
    require(not any(token in common_text for token in forbidden) and
            not any(token in (ast.get_source_segment(source_text, weight) or ast.unparse(weight))
                    for token in forbidden),
            "real service interface unexpectedly became addressed")
    return {
        "receipt_fields": ["task", "counts", "source_address_first", "source_address_count",
                           "weight_beat_first", "dma_first", "psum_addresses", "commit_first"],
        "weight_interval_arguments": ["start", "beats", "half_slot"],
        "weight_call": "weight_task(global_offset + start - preprocess, receipt.counts.weight, index & 1)",
        "missing_native_fields": ["op", "logical_bank", "native_slices", "local_row",
                                  "bytes_and_byte_enable", "native_macro_activations",
                                  "service_beat_to_store_exact_once"],
    }


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise HammerFailure("attack accepted: " + label)


def load_subject():
    spec = importlib.util.spec_from_file_location("m1129c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load subject")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_requests(module):
    digest = "0" * 64
    rows = []
    expected = {axis: set() for axis in module.AXES}
    for axis_id, axis in enumerate(module.AXES):
        beat0, beat1 = 2 * axis_id, 2 * axis_id + 1
        expected[axis].update((beat0, beat1))
        base = 3 * axis_id
        rows.extend((
            module.AddressedWeightTransaction(axis, 5, 5, 0, "WRITE", 0, 0, 0, 0,
                tuple(range(8)), 128, (0xffff,) * 8, 8, beat0, base, 0, 0, digest),
            module.AddressedWeightTransaction(axis, 5, 5, 0, "WRITE", 1, 1, 0, 16,
                tuple(range(8)), 128, (0xffff,) * 8, 8, beat1, base + 1, 0, 1, digest),
            module.AddressedWeightTransaction(axis, 7, 7, 0, "READ", 0, 0, 0, 0,
                tuple(range(24)), 384, (0xffff,) * 24, 24, None, base + 2, 0, 2, digest),
        ))
    return rows, expected


def independent_schedule(rows) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda pair: (
        pair[1].requested_cycle, pair[1].source_task_id,
        pair[1].source_local_ordinal, pair[1].store_transaction_ordinal, pair[0]))
    next_cycle: dict[tuple[str, int], int] = {}
    output = []
    for _, row in ordered:
        cycle = max([row.requested_cycle] +
                    [next_cycle.get((row.axis, native_slice), 0)
                     for native_slice in row.native_slices])
        output.append({"axis": row.axis, "op": row.op, "cycle": cycle,
                       "stall_cycles": cycle - row.requested_cycle,
                       "half_slot": row.half_slot,
                       "native_slices": list(row.native_slices),
                       "service_beat_ordinal": row.service_beat_ordinal})
        for native_slice in row.native_slices:
            next_cycle[(row.axis, native_slice)] = cycle + 1
    return output


def main() -> None:
    before = {path: sha256(path) for path in (SOURCE, CONTRACT, M1102, M1016, DOCS359)}
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    verify_regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    verify_flat(AUTHOR, EXPECTED["author_review"], EXPECTED["author_manifest"], EXPECTED["author_outer"])
    verify_regular(M1102, EXPECTED["m1102"])
    verify_regular(M1016, EXPECTED["m1016"])
    verify_regular(DOCS359, EXPECTED["docs359"])

    source_text = SOURCE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    analyze_subject(source_text, contract)
    actual = audit_real_m1016_interface(M1016.read_text(encoding="utf-8"))
    require(author["status"] ==
            "PASS_M1128C_WEIGHT_SERVICE_ADDRESSED_LEDGER_SOURCE_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED__CANONICAL_STOP",
            "author status drift")
    require(author["identity"]["source_sha256"] == EXPECTED["source"] and
            author["identity"]["contract_sha256"] == EXPECTED["contract"],
            "author source/contract binding drift")

    module = load_subject()
    audit = module.audit_frozen_service_interface()
    require(audit["canonical_ready"] is False and
            audit["canonical_row_reader_opened"] is False and
            audit["full_51840000_rows_read"] is False and
            audit["canonical_weight_transactions_emitted"] == 0,
            "runtime canonical STOP boundary drift")
    canonical_stopped = False
    try:
        next(module.iter_canonical_weight_addressed_ledger())
    except module.Failure:
        canonical_stopped = True
    require(canonical_stopped, "canonical iterator escaped before addressed fields exist")

    requests, expected_beats = independent_requests(module)
    scheduled = module.schedule_native_one_rw(requests)
    summary = module.validate_exact_once_and_conflicts(scheduled, expected_beats)
    require(summary == {
        "transactions": 9, "unique_transaction_identities": 9,
        "refill_store_transactions": 6, "full_record_read_transactions": 3,
        "service_beats_expected": 6, "service_beats_exact_once": 6,
        "explicitly_stalled_transactions": 3,
        "final_native_1rw_conflicts": 0, "final_weight_half_slot_overlaps": 0,
    }, "subject bounded synthetic mismatch")
    independent = independent_schedule(requests)
    require(len(independent) == 9 and
            sum(item["op"] == "WRITE" for item in independent) == 6 and
            sum(item["op"] == "READ" for item in independent) == 3 and
            sum(item["stall_cycles"] > 0 for item in independent) == 3 and
            sorted({item["cycle"] for item in independent}) == [5, 6, 7],
            "independent 24-macro schedule mismatch")

    write = next(row for row in requests if row.op == "WRITE")
    read = next(row for row in requests if row.op == "READ")
    rejected("wrong_op_relation", lambda: replace(write, op="READ").validate())
    rejected("wrong_logical_bank", lambda: replace(write, logical_bank=1).validate())
    rejected("wrong_native_slice_set", lambda: replace(write, native_slices=tuple(range(1, 9))).validate())
    rejected("wrong_local_row", lambda: replace(write, local_row=1).validate())
    rejected("wrong_bytes", lambda: replace(write, bytes=127).validate())
    rejected("wrong_byte_enable", lambda: replace(write, byte_enable_per_slice=(0xfffe,) * 8).validate())
    rejected("wrong_macro_activation", lambda: replace(write, native_macro_activations=7).validate())
    rejected("read_carries_service_beat", lambda: replace(read, service_beat_ordinal=99).validate())
    rejected("duplicate_service_beat", lambda: module.validate_exact_once_and_conflicts(
        [scheduled[0], replace(scheduled[3], service_beat_ordinal=scheduled[0].service_beat_ordinal)],
        {axis: ({0, 1} if axis == "candidate" else set()) for axis in module.AXES}))
    rejected("missing_service_beat", lambda: module.validate_exact_once_and_conflicts(
        scheduled[1:], expected_beats))
    rejected("duplicate_transaction_identity", lambda: module.validate_exact_once_and_conflicts(
        [scheduled[0], replace(scheduled[3], source_local_ordinal=scheduled[0].source_local_ordinal,
                                store_transaction_ordinal=scheduled[0].store_transaction_ordinal)],
        {axis: ({0, 1} if axis == "candidate" else set()) for axis in module.AXES}))
    rejected("final_same_slice_1rw_conflict", lambda: module.validate_exact_once_and_conflicts(
        [write, replace(write, half_slot=1, logical_bank=1, local_row=16,
                        service_beat_ordinal=1, source_local_ordinal=1,
                        store_transaction_ordinal=1)],
        {axis: ({0, 1} if axis == "candidate" else set()) for axis in module.AXES}))
    available = {"counts", "weight_beat_first", "start", "beats", "half_slot"}
    required = {"op", "logical_bank", "native_slices", "local_row", "bytes",
                "byte_enable", "native_macro_activations", "exact_once_store_relation"}
    rejected("count_geometry_fabrication",
             lambda: require(required <= available, "summary fields are not addressed events"))

    mutations = []
    mutated = copy.deepcopy(contract); mutated["authorization"]["full_51840000_replay_now"] = True
    mutations.append(("contract_full_replay_escalation", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["frozen_actual_service_interface"]["canonical_ready"] = True
    mutations.append(("contract_canonical_ready_escalation", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["fail_closed_behavior"]["canonical_rows_read"] = 1
    mutations.append(("contract_row_open_escalation", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["claim_boundary"]["h67_transactions"] = True
    mutations.append(("contract_synthetic_claim_escalation", source_text, mutated))
    mutations.append(("source_canonical_ready_true",
                      source_text.replace('"canonical_ready": False', '"canonical_ready": True', 1),
                      contract))
    for label, candidate_source, candidate_contract in mutations:
        rejected(label, lambda text=candidate_source, con=candidate_contract:
                 analyze_subject(text, con))

    after = {path: sha256(path) for path in before}
    require(before == after, "hammer modified subject or authority")
    result = {
        "schema": "m1129c_m1128c_weight_service_addressed_ledger_static_hammer_r1_v1",
        "status": "PASS_DIFFERENT_AUTHOR_STATIC_AND_MUTATION_HAMMER__CANONICAL_STOP__GO_ADDITIVE_INTERNAL_EVENT_INSTRUMENTATION_SOURCE_ONLY",
        "score": 100,
        "checks": checks,
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "actual_frozen_interface": actual,
        "canonical_boundary": {
            "stopped_before_row_open": True,
            "canonical_rows_read": 0,
            "canonical_weight_transactions_emitted": 0,
            "full_51840000_replayed": False,
        },
        "bounded_synthetic": summary,
        "independent_schedule": independent,
        "authorization": {
            "next_additive_iterator_instrumentation_source_only": True,
            "must_derive_from_internal_real_service_refill_events": True,
            "may_infer_from_counts_weight_beat_first_or_capacity": False,
            "canonical_ledger_or_full_replay_now": False,
            "eda_rtl_gpu_remote_now": False,
        },
        "execution": {
            "bounded_import_and_synthetic_only": True,
            "full_51840000_replay": False,
            "eda": False, "rtl": False, "gpu": False, "remote": False,
            "subject_or_authority_modified": False, "docs359_modified": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

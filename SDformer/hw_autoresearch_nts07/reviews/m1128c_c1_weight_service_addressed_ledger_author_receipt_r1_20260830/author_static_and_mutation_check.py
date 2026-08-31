#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author static/mutation receipt check for source-only M1128C."""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1128c_c1_weight_service_addressed_ledger_source.py"
CONTRACT = HW / "contracts/m1128c_c1_weight_service_addressed_ledger_source_contract_r1_20260830.json"
M1126 = HW / "reviews/m1126c_c1_three_axis_storage_transaction_exporter_author_receipt_r1_20260830"
M1127 = HW / "reviews/m1127c_m1126c_c1_three_axis_storage_transaction_exporter_static_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "d25f9e4fdfda62f56e7efb120fe0c8f6108a4b23ba4eee712e3ec471b5fa493e",
    "contract": "69bcc952953a23d102ac021e2b67375ef0d539b47bf88c347081200fae1b9102",
    "contract_side": "f132061ad02a122b939cc3b1cad150b3acb4efda3d72c16d8027fd12b2c101e0",
    "contract_outer": "bb8eca6f7dd02546a9d8aed009e44212c89ed9fe90376ce83306128133786166",
    "m1126": ("5fea575ca6fce2bb3ca9831864a029e6cddd15b02b726a679eaef847512ca49e",
               "15a0236256bc9735936a474b08a3997bd5ad5084db31e20fc772cce8346487a2",
               "3254655b33067852d3a8f305e12d6c9fc408549b4a47b1a56b4f401a1d7df087"),
    "m1127": ("d93f72e5b045258155b09ec403d91a02282a30a757f0b2ea118a7dc1c40e135d",
               "1f539adf8b270925e54eb4938b3ab64930a5ec9c7f32f273374671025efbf971",
               "3bb7e99d668626a7455d3857f90d5fa7c5a40aebda269b64d731e8cfab7191b8"),
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


def verify_regular(path: Path, digest: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha256(path) == digest, "identity drift: " + str(path))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review, manifest, outer = directory / "review.json", directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed identity drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1); relative = relative.lstrip("*")
        require(relative not in listed, "duplicate manifest member")
        verify_regular(directory / relative, digest); listed.add(relative)
    digest, relative = outer.read_text(encoding="utf-8").split()
    require(relative == "SHA256SUMS" and digest == sha256(manifest), "outer drift")


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, node) or ast.unparse(node)


def analyze(source: str, contract: dict[str, Any]) -> None:
    ast.parse(source)
    audit = function_text(source, "audit_frozen_service_interface")
    iterator = function_text(source, "iter_canonical_weight_addressed_ledger")
    validator = function_text(source, "validate_exact_once_and_conflicts")
    require("CanonicalRowReader" not in source and
            "iter_canonical_full_replay_results(" not in source,
            "full row path referenced")
    require('audit["canonical_ready"] is True' in iterator and
            iterator.index('audit["canonical_ready"] is True') < iterator.index("yield {}"),
            "canonical STOP order drift")
    require('"canonical_row_reader_opened": False' in audit and
            '"full_51840000_rows_read": False' in audit and
            '"canonical_weight_transactions_emitted": 0' in audit and
            '"canonical_ready": False' in audit,
            "canonical zero boundary drift")
    require('receipt["counts"]["weight"]' in audit and 'index & 1' in audit and
            '"count_or_weight_beat_first_expansion_allowed": False' in audit and
            '"capacity_geometry_expansion_allowed": False' in audit,
            "actual frozen-interface gap drift")
    require('NATIVE_SLICES = 24' in source and 'NATIVE_DEPTH = 128' in source and
            'SLICE_BYTES = 16' in source and 'RECORD_BYTES = NATIVE_SLICES * SLICE_BYTES' in source,
            "24x128x128b mapping constants drift")
    require('"duplicate service beat mapping"' in validator and
            '"final native 1RW conflict"' in validator and
            '"final weight half-slot overlap"' in validator and
            'observed_beats == expected_service_beats' in validator,
            "exact-once/conflict gates drift")
    require(contract["status"] ==
            "SOURCE_ONLY_CANONICAL_STOP_FROZEN_WEIGHT_EVENT_FIELDS_INSUFFICIENT__SYNTHETIC_MAPPING_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "contract status drift")
    require(contract["frozen_actual_service_interface"]["canonical_ready"] is False and
            contract["frozen_actual_service_interface"]["count_or_weight_beat_first_may_be_expanded"] is False,
            "contract real-service fail-closed drift")
    require(contract["physical_mapping_contract"]["mapping_status"] ==
            "synthetic/proposed mapping contract only; not frozen canonical evidence" and
            contract["physical_mapping_contract"]["same_mapping_function_for_all_axes"] is True,
            "synthetic label or three-axis mapping drift")
    require(contract["fail_closed_behavior"] == {
        "canonical_iterator_stops_before_canonical_row_reader_open": True,
        "canonical_rows_read": 0,
        "canonical_weight_transactions_emitted": 0,
        "missing_fields_not_inferred_from_count": True,
        "missing_fields_not_inferred_from_weight_beat_first": True,
        "missing_fields_not_inferred_from_capacity_geometry": True,
        "synthetic_mapping_never_labeled_canonical": True,
    }, "contract STOP behavior drift")
    auth = contract["authorization"]
    require(auth["different_author_static_hammer_only"] is True and
            all(auth[key] is False for key in (
                "canonical_export_now", "canonical_row_open_now",
                "full_51840000_replay_now", "runner_now",
                "eda_rtl_gpu_remote_now", "performance_or_energy_now")),
            "authorization escalation")
    require(contract["claim_boundary"]["source_schema_and_synthetic_mechanics_only"] is True and
            all(value is False for key, value in contract["claim_boundary"].items()
                if key != "source_schema_and_synthetic_mechanics_only"),
            "claim escalation")


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def load_source():
    spec = importlib.util.spec_from_file_location("m1128c_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    before = sha256(SOURCE)
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    verify_regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    verify_flat(M1126, EXPECTED["m1126"]); verify_flat(M1127, EXPECTED["m1127"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    source = SOURCE.read_text(encoding="utf-8"); contract = strict_json(CONTRACT)
    analyze(source, contract)
    process = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(SOURCE), "--self-test"],
        cwd=HW, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False)
    require(process.returncode == 0 and process.stderr == "", "bounded runtime failed")
    runtime = json.loads(process.stdout)
    require(runtime["status"] == "PASS_SYNTHETIC_24X128X128B_MAPPING__CANONICAL_STOP" and
            runtime["frozen_interface_audit"]["canonical_weight_transactions_emitted"] == 0 and
            runtime["synthetic"] == {
                "transactions": 9, "unique_transaction_identities": 9,
                "refill_store_transactions": 6, "full_record_read_transactions": 3,
                "service_beats_expected": 6, "service_beats_exact_once": 6,
                "explicitly_stalled_transactions": 3,
                "final_native_1rw_conflicts": 0,
                "final_weight_half_slot_overlaps": 0},
            "bounded runtime payload drift")
    module = load_source(); digest = "0" * 64
    base = module.AddressedWeightTransaction(
        "candidate", 5, 5, 0, "WRITE", 0, 0, 0, 0, tuple(range(8)),
        128, (0xffff,) * 8, 8, 0, 0, 0, 0, digest)
    good_read = module.AddressedWeightTransaction(
        "candidate", 7, 7, 0, "READ", 0, 0, 0, 0, tuple(range(24)),
        384, (0xffff,) * 24, 24, None, 1, 0, 1, digest)
    rejected("duplicate_service_beat", lambda: module.validate_exact_once_and_conflicts(
        [base, replace(base, store_transaction_ordinal=2, source_local_ordinal=2)],
        {"candidate": {0}, "strongest_zero": set(), "same_coordinate_bit": set()}))
    rejected("missing_service_beat", lambda: module.validate_exact_once_and_conflicts(
        [base, good_read], {"candidate": {0, 1}, "strongest_zero": set(),
                            "same_coordinate_bit": set()}))
    rejected("wrong_native_slice_set", lambda: replace(base, native_slices=tuple(range(1, 9))).validate())
    rejected("wrong_bytes", lambda: replace(base, bytes=127).validate())
    rejected("wrong_byte_enable", lambda: replace(base, byte_enable_per_slice=(0xffff,) * 7).validate())
    rejected("wrong_activation_multiplicity", lambda: replace(base, native_macro_activations=7).validate())
    colliding = replace(base, service_beat_ordinal=1, store_transaction_ordinal=2,
                        source_local_ordinal=2, half_slot=1, logical_bank=1,
                        local_row=16)
    rejected("final_same_slice_1rw_conflict", lambda: module.validate_exact_once_and_conflicts(
        [base, colliding], {"candidate": {0, 1}, "strongest_zero": set(),
                            "same_coordinate_bit": set()}))
    rejected("count_expansion", lambda: require(
        runtime["frozen_interface_audit"]["count_or_weight_beat_first_expansion_allowed"] is True,
        "count expansion forbidden"))
    mutated = copy.deepcopy(contract); mutated["authorization"]["full_51840000_replay_now"] = True
    rejected("contract_full_replay", lambda: analyze(source, mutated))
    mutated = copy.deepcopy(contract); mutated["claim_boundary"]["traffic"] = True
    rejected("contract_traffic_claim", lambda: analyze(source, mutated))
    mutated = source.replace('"canonical_ready": False', '"canonical_ready": True', 1)
    rejected("source_canonical_ready", lambda: analyze(mutated, contract))
    mutated = source.replace('"canonical_weight_transactions_emitted": 0',
                             '"canonical_weight_transactions_emitted": 1', 1)
    rejected("source_real_tx_nonzero", lambda: analyze(mutated, contract))
    require(len(attacks) == 12, "all twelve attacks rejected")
    require(sha256(SOURCE) == before == EXPECTED["source"], "subject modified")
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 modified")
    print(json.dumps({
        "schema": "m1128c_weight_service_addressed_ledger_author_check_v1",
        "status": "PASS_M1128C_SOURCE_AND_SYNTHETIC__CANONICAL_STOP__NO_FULL_NO_EDA",
        "checks_passed": checks,
        "attacks_rejected": attacks,
        "source_sha256": EXPECTED["source"],
        "contract_sha256": EXPECTED["contract"],
        "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
        "canonical": {"row_reader_opened": False, "rows": 0,
                      "weight_transactions": 0, "ready": False},
        "synthetic": runtime["synthetic"],
        "authorization": {"different_author_hammer_only": True,
                          "full_replay": False, "eda": False},
        "docs359_sha256": EXPECTED["docs359"],
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent bounded hammer for M1126C; never opens the 51.84M rows."""
from __future__ import annotations

import ast
import copy
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
SOURCE = HW / "system_simulator/scripts/build_m1126c_c1_three_axis_storage_transaction_exporter_source.py"
CONTRACT = HW / "contracts/m1126c_c1_three_axis_storage_transaction_exporter_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1126c_c1_three_axis_storage_transaction_exporter_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1102_SOURCE = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
M1102_RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_RESULT_OUTER = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/.m1102_atomic_seal/SHA256SUMS.seal.sha256"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1123C = HW / "reviews/m1123c_m1122c_c1_path_c_common_charge_independent_hammer_r1_20260830"
M1125C = HW / "reviews/m1125c_c1_path_c_105macro_common_model_first_principles_audit_r1_20260830"

EXPECTED = {
    "source": "d54640b0bb85e7ba2e4222655a4325b23310aab8eb75b88c13ed00ad5ef12e27",
    "contract": "501406d91811e4808997cef94e0a0a07aeb039dae6282d39ce6d3f842b1e71df",
    "contract_side": "3f336d53a7520c05c75add2fae012ea0a913a1dd542ca8effde6396e13d017ca",
    "contract_outer": "24f0c43ff7fb557996dc5ca758abe79f704c47298f99041ca513426b25d44e07",
    "author_review": "5fea575ca6fce2bb3ca9831864a029e6cddd15b02b726a679eaef847512ca49e",
    "author_manifest": "15a0236256bc9735936a474b08a3997bd5ad5084db31e20fc772cce8346487a2",
    "author_outer": "3254655b33067852d3a8f305e12d6c9fc408549b4a47b1a56b4f401a1d7df087",
    "author_oracle": "20eeb5c54ef1f137f0dea535f1ce014a85ce853e9b8a16a227ce70b543769c4b",
    "m1102_source": "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    "m1102_result": "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    "m1102_result_outer": "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f",
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
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
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
    node = next((node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)
    require(node is not None, "missing function " + name)
    return node


def calls(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def analyze(source_text: str, contract: dict[str, Any]) -> None:
    tree = ast.parse(source_text)
    audit = function_node(tree, "audit_frozen_exportability")
    iterator = function_node(tree, "iter_canonical_transactions")
    small = function_node(tree, "source_small_oracle")
    iterator_text = ast.get_source_segment(source_text, iterator) or ast.unparse(iterator)
    audit_text = ast.get_source_segment(source_text, audit) or ast.unparse(audit)
    small_text = ast.get_source_segment(source_text, small) or ast.unparse(small)
    require("CanonicalRowReader" not in source_text and
            "iter_canonical_full_replay_results(" not in source_text,
            "exporter may open frozen rows")
    require(calls(iterator) == ["audit_frozen_exportability", "require"],
            "canonical iterator call graph drift")
    require('audit["canonical_export_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_export_ready"] is True') <
            iterator_text.index("yield {}"), "STOP is not before yield")
    require('"canonical_row_reader_opened": False' in audit_text and
            '"full_51840000_source_rows_read": False' in audit_text and
            '"transaction_rows_emitted": 0' in audit_text and
            '"canonical_export_ready": False' in audit_text,
            "zero-row/zero-transaction STOP drift")
    require('set(receipt) == {"task", "counts", "source_address_first",' in audit_text and
            '"source_address_count", "weight_beat_first", "dma_first",' in audit_text and
            '"psum_addresses", "commit_first"}' in audit_text,
            "frozen receipt-field boundary drift")
    for missing in (
        "native 1RW operation (READ versus WRITE)",
        "local 24-slice macro address",
        "logical bytes and byte-enable per on-chip access",
        "native-macro activation multiplicity per access",
        "exact-once relation between weight_beat service and on-chip weight store",
    ):
        require(missing in audit_text, "missing weight gap not encoded: " + missing)
    require('hasattr(m1102.M1072.M1016, "iter_parent_address_events")' in audit_text,
            "candidate parent boundary drift")
    require('"logical_bank", "address", "op", "base_ready_cycle"' in audit_text and
            '("cycle", "group", "address", "op")' in audit_text,
            "psum addressed/grant boundary drift")
    require('RESIDUAL_BYTES = 24_448' in source_text and
            '"residual_transactions_permitted": False' in audit_text and
            'self.storage_class in LIVE_CLASSES' in source_text and
            'LIVE_CLASSES = ("parent", "psum", "weight")' in source_text,
            "24448B residual prohibition drift")
    require(small_text.count("StorageTransaction(") == 6 and
            '"transactions": 5' in small_text and
            '"explicitly_stalled_transactions": 2' in small_text and
            '"final_1rw_conflicts": 0' in small_text and
            '"weight_half_slot_overlaps": 0' in small_text,
            "bounded oracle or attacks drift")
    require(contract["status"] ==
            "SOURCE_ONLY_FAIL_CLOSED_WEIGHT_TRANSACTION_PROVENANCE_GAP__DIFFERENT_AUTHOR_STATIC_HAMMER_ONLY",
            "contract status drift")
    capability = contract["frozen_capability_audit"]
    require(capability["candidate_parent_address_events_reconstructable"] is True and
            capability["baseline_parent_zero_aggregate_sealed"] is True and
            capability["psum_port_events_and_1rw_grants_reconstructable"] is True and
            capability["source_row_provenance_reconstructable"] is True,
            "parent/psum reconstructability boundary drift")
    require(all(capability[key] is False for key in (
        "weight_native_1rw_op_available",
        "weight_local_24_slice_macro_address_available",
        "weight_onchip_access_bytes_and_byte_enable_available",
        "weight_native_macro_activation_count_available",
        "weight_dram_beat_to_onchip_store_exact_once_mapping_available",
        "canonical_export_ready")), "weight gap falsely admitted")
    behavior = contract["fail_closed_behavior"]
    require(behavior == {
        "canonical_iterator_stops_before_canonical_row_reader_open": True,
        "canonical_rows_read": 0,
        "transaction_rows_emitted": 0,
        "no_weight_fields_inferred_from_capacity_geometry": True,
        "no_residual_accesses_invented": True,
        "no_partial_parent_or_psum_export_may_be_labeled_complete": True,
    }, "contract fail-closed behavior drift")
    require(contract["transaction_schema"]["residual_24448B_rule"] ==
            "identical conservative capacity denominator only; never synthesize an access",
            "contract residual rule drift")
    auth = contract["authorization"]
    require(auth["different_author_static_hammer_only"] is True and
            all(auth[key] is False for key in (
                "full_export_now", "canonical_row_open_now",
                "full_51840000_row_replay_now", "result_runner_now",
                "eda_rtl_gpu_remote_now", "new_performance_or_energy_now")),
            "authorization escalation")
    require(all(value is False for key, value in contract["claim_boundary"].items()
                if key != "source_only") and contract["claim_boundary"]["source_only"] is True,
            "claim escalation")


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise HammerFailure("attack accepted: " + label)


def independent_schedule() -> dict[str, int]:
    requests = [
        ("psum", 0, 7, 0, 0, None),
        ("psum", 0, 7, 0, 1, None),
        ("weight", 0, 9, 0, 2, 0),
        ("weight", 0, 9, 0, 3, 1),
        ("parent", 0, 10, 0, 4, None),
    ]
    next_cycle: dict[tuple[str, int], int] = {}
    seen = set()
    occupied = set()
    stalls = 0
    for storage_class, bank, requested, task, ordinal, half_slot in requests:
        identity = (storage_class, task, ordinal)
        require(identity not in seen and storage_class in ("parent", "psum", "weight"),
                "independent exact-once/residual failure")
        seen.add(identity)
        key = (storage_class, bank)
        cycle = max(requested, next_cycle.get(key, 0))
        stalls += int(cycle > requested)
        require((storage_class == "weight") == (half_slot in (0, 1)),
                "independent half-slot classification")
        require((storage_class, bank, cycle) not in occupied, "independent final conflict")
        occupied.add((storage_class, bank, cycle))
        next_cycle[key] = cycle + 1
    return {"transactions": len(requests), "unique_source_transactions": len(seen),
            "explicitly_stalled_transactions": stalls,
            "final_1rw_conflicts": 0, "weight_half_slot_overlaps": 0}


def load_subject():
    spec = importlib.util.spec_from_file_location("m1127c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load subject")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    before = sha256(SOURCE)
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    verify_regular(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_side"])
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    verify_flat(AUTHOR, EXPECTED["author_review"], EXPECTED["author_manifest"],
                EXPECTED["author_outer"])
    verify_regular(AUTHOR / "small_synthetic_oracle.json", EXPECTED["author_oracle"])
    verify_regular(M1102_SOURCE, EXPECTED["m1102_source"])
    verify_regular(M1102_RESULT, EXPECTED["m1102_result"])
    verify_regular(M1102_RESULT_OUTER, EXPECTED["m1102_result_outer"])
    verify_flat(M1000,
                "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
                "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
                "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5")
    verify_flat(M1123C,
                "b2752ce9e805bb1cbadab2229b48c287df4d7321b6f442a8b004dc904ab43e82",
                "8ead4a34f4c418fbca9343b984144808f9d785dfd39595e293801ea94ceef724",
                "4c1679005159d75f3fda75a9adceb7b6b17d6baae77949b312ec5ecf3a0d73ae")
    verify_flat(M1125C,
                "348e18ebdcf37f1740bcd8b977885ee86ea5b0a172232413866f2c739879d77c",
                "e306057ae9d3b52700d1221d764426d98fcc13221ab905129f0fb1aaacc3d8d1",
                "a0c3d3e137a07fc09294dfaf1e4e806ba9be11117506e0bd1e5d3e476ac094b1")
    source_text = SOURCE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    author_review = strict_json(AUTHOR / "review.json")
    author_oracle = strict_json(AUTHOR / "small_synthetic_oracle.json")
    analyze(source_text, contract)
    require(author_review["status"] ==
            "PASS_SOURCE_CONTRACT_AND_SMALL_SYNTHETIC__CANONICAL_EXPORT_FAILS_CLOSED__DIFFERENT_AUTHOR_STATIC_HAMMER_REQUIRED",
            "author status drift")
    require(author_review["identity"]["source_sha256"] == EXPECTED["source"] and
            author_review["identity"]["contract_sha256"] == EXPECTED["contract"],
            "author identity drift")
    require(author_oracle["exportability"]["canonical_row_reader_opened"] is False and
            author_oracle["exportability"]["full_51840000_source_rows_read"] is False and
            author_oracle["exportability"]["transaction_rows_emitted"] == 0,
            "sealed oracle canonical boundary drift")

    process = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-I", str(SOURCE), "--self-test"],
        cwd=HW, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        check=False)
    require(process.returncode == 0 and process.stderr == "", "bounded source selftest failed")
    runtime = json.loads(process.stdout)
    require(runtime == author_oracle, "runtime bounded oracle differs from sealed oracle")
    independent = independent_schedule()
    require(independent == author_oracle["synthetic"], "independent 5tx oracle mismatch")

    module = load_subject()
    digest = "0" * 64
    rejected("duplicate_exact_once", lambda: module.validate_exact_once([
        module.StorageTransaction(1, 1, 0, "candidate", "parent", 0, 0,
                                  "READ", 16, 1, None, 0, 0, digest),
        module.StorageTransaction(2, 2, 0, "candidate", "parent", 0, 1,
                                  "READ", 16, 1, None, 0, 0, digest)]))
    rejected("residual_24448_access", lambda: module.StorageTransaction(
        0, 0, 0, "candidate", "residual", 0, 0, "READ", 24448, 1,
        None, 0, 1, digest).validate())
    rejected("weight_half_slot_overlap", lambda: module.validate_exact_once([
        module.StorageTransaction(9, 9, 0, "candidate", "weight", 0, 0,
                                  "READ", 384, 24, 0, 0, 2, digest),
        module.StorageTransaction(9, 9, 0, "candidate", "weight", 0, 16,
                                  "READ", 384, 24, 1, 0, 3, digest)]))
    available = {"counts", "weight_beat_first"}
    required = {"op", "local_address", "bytes", "byte_enable",
                "native_macro_activations", "exact_once_store_relation"}
    rejected("count_and_weight_beat_first_fabrication",
             lambda: require(required <= available, "count is not addressed transaction"))

    mutations = []
    mutated = copy.deepcopy(contract); mutated["authorization"]["full_export_now"] = True
    mutations.append(("contract_full_export", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["frozen_capability_audit"]["canonical_export_ready"] = True
    mutations.append(("contract_weight_gap_bypass", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["fail_closed_behavior"]["canonical_rows_read"] = 1
    mutations.append(("contract_row_count_nonzero", source_text, mutated))
    mutated = copy.deepcopy(contract); mutated["transaction_schema"]["residual_24448B_rule"] = "use as cache"
    mutations.append(("contract_residual_promoted", source_text, mutated))
    mutations.append(("source_canonical_ready_true",
                      source_text.replace('"canonical_export_ready": False',
                                          '"canonical_export_ready": True', 1), contract))
    mutations.append(("source_tx_nonzero",
                      source_text.replace('"transaction_rows_emitted": 0',
                                          '"transaction_rows_emitted": 1', 1), contract))
    mutations.append(("source_reader_injected",
                      source_text.replace('audit = audit_frozen_exportability()',
                                          'CanonicalRowReader(); audit = audit_frozen_exportability()', 1),
                      contract))
    for label, text, candidate_contract in mutations:
        rejected(label, lambda text=text, candidate_contract=candidate_contract:
                 analyze(text, candidate_contract))
    require(len(attacks) == 11, "all eleven attacks rejected")
    require(sha256(SOURCE) == before == EXPECTED["source"], "subject modified")
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 modified")
    print(json.dumps({
        "schema": "m1127c_m1126c_three_axis_storage_exporter_static_hammer_v1",
        "status": "PASS_GAP_LOCALIZATION_AND_SYNTHETIC_SOURCE_ONLY__STOP_CANONICAL_EXPORT",
        "checks_passed": checks,
        "attacks_rejected": attacks,
        "canonical": {"rows_read": 0, "transactions_emitted": 0,
                      "canonical_row_reader_opened": False,
                      "full_export": False},
        "boundary": {
            "candidate_parent_address_events": "reconstructable",
            "baseline_parent": "sealed aggregate only; not addressed events",
            "psum_port_events_and_arbitrated_1rw_grants": "reconstructable",
            "weight_native_addressed_transactions": "not reconstructable",
            "residual_24448_bytes": "capacity denominator only; accesses forbidden",
        },
        "synthetic": independent,
        "execution": {"bounded_source_selftest": True, "full_51840000": False,
                      "eda": False, "rtl": False, "gpu": False, "remote": False},
        "source_sha256_before_after": EXPECTED["source"],
        "docs359_sha256": EXPECTED["docs359"],
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

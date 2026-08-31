#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1128C additive source for a C1 weight-service addressed ledger.

The actual frozen M1102/M1016 interface exposes only a weight service count,
global first-beat ordinal and an interval/half-slot summary.  It does not expose
native on-chip transactions.  The zero-argument canonical iterator therefore
stops before any payload row can be opened.  Only a bounded synthetic mapping
and arbitration oracle is executable in this source revision.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

M1126_SOURCE = HERE / "build_m1126c_c1_three_axis_storage_transaction_exporter_source.py"
M1126_SOURCE_SHA = "d54640b0bb85e7ba2e4222655a4325b23310aab8eb75b88c13ed00ad5ef12e27"
M1126_CONTRACT = HW / "contracts/m1126c_c1_three_axis_storage_transaction_exporter_source_contract_r1_20260830.json"
M1126_CONTRACT_OUTER_SHA = "24f0c43ff7fb557996dc5ca758abe79f704c47298f99041ca513426b25d44e07"
M1126_AUTHOR = HW / "reviews/m1126c_c1_three_axis_storage_transaction_exporter_author_receipt_r1_20260830"
M1126_AUTHOR_ID = (
    "5fea575ca6fce2bb3ca9831864a029e6cddd15b02b726a679eaef847512ca49e",
    "15a0236256bc9735936a474b08a3997bd5ad5084db31e20fc772cce8346487a2",
    "3254655b33067852d3a8f305e12d6c9fc408549b4a47b1a56b4f401a1d7df087",
)
M1127 = HW / "reviews/m1127c_m1126c_c1_three_axis_storage_transaction_exporter_static_hammer_r1_20260830"
M1127_ID = (
    "d93f72e5b045258155b09ec403d91a02282a30a757f0b2ea118a7dc1c40e135d",
    "1f539adf8b270925e54eb4938b3ab64930a5ec9c7f32f273374671025efbf971",
    "3bb7e99d668626a7455d3857f90d5fa7c5a40aebda269b64d731e8cfab7191b8",
)
M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1102_RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_RESULT_SHA = "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91"
M1102_RESULT_OUTER = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/.m1102_atomic_seal/SHA256SUMS.seal.sha256"
M1102_RESULT_OUTER_SHA = "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
M1016_SOURCE = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
CONTRACT = HW / "contracts/m1128c_c1_weight_service_addressed_ledger_source_contract_r1_20260830.json"
CONTRACT_SHA = "69bcc952953a23d102ac021e2b67375ef0d539b47bf88c347081200fae1b9102"
CONTRACT_OUTER_SHA = "bb8eca6f7dd02546a9d8aed009e44212c89ed9fe90376ce83306128133786166"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
NATIVE_SLICES = 24
NATIVE_DEPTH = 128
SLICE_BYTES = 16
ROWS_PER_HALF = 16
RECORD_BYTES = NATIVE_SLICES * SLICE_BYTES
REFILL_SLICES = 8
REFILL_BYTES = REFILL_SLICES * SLICE_BYTES


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == digest,
            "regular-file identity drift: " + str(path))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == expected,
                "sealed member drift: " + relative)
        listed.add(relative)
    expected, relative = outer.read_text(encoding="utf-8").split()
    require(relative == "SHA256SUMS" and expected == sha256(manifest),
            "outer seal drift: " + directory.name)


def load_m1102():
    verify_regular(M1102_SOURCE, M1102_SOURCE_SHA)
    spec = importlib.util.spec_from_file_location("m1128c_frozen_m1102", M1102_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1102")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = next((node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)
    require(node is not None, "missing function: " + name)
    return node


def method_node(source: str, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    klass = next((node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    require(klass is not None, "missing class: " + class_name)
    node = next((node for node in klass.body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)
    require(node is not None, "missing method: " + class_name + "." + name)
    return node


def audit_frozen_service_interface() -> dict[str, Any]:
    """Audit actual event definitions without constructing a row reader."""
    verify_regular(M1126_SOURCE, M1126_SOURCE_SHA)
    verify_regular(M1126_CONTRACT, "501406d91811e4808997cef94e0a0a07aeb039dae6282d39ce6d3f842b1e71df")
    verify_regular(Path(str(M1126_CONTRACT) + ".sha256.seal.sha256"),
                   M1126_CONTRACT_OUTER_SHA)
    verify_flat(M1126_AUTHOR, M1126_AUTHOR_ID)
    verify_flat(M1127, M1127_ID)
    verify_regular(M1102_RESULT, M1102_RESULT_SHA)
    verify_regular(M1102_RESULT_OUTER, M1102_RESULT_OUTER_SHA)
    verify_flat(M1000, M1000_ID)
    verify_regular(M1016_SOURCE, M1016_SOURCE_SHA)
    verify_regular(CONTRACT, CONTRACT_SHA)
    verify_regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), CONTRACT_OUTER_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    contract = strict_json(CONTRACT)
    require(contract["status"] ==
            "SOURCE_ONLY_CANONICAL_STOP_FROZEN_WEIGHT_EVENT_FIELDS_INSUFFICIENT__SYNTHETIC_MAPPING_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            contract["authorization"]["full_51840000_replay_now"] is False,
            "M1128C contract drift")
    m1102 = load_m1102()
    require(tuple(m1102.DESIGNS) == AXES and
            inspect.isgeneratorfunction(m1102.iter_canonical_full_replay_results) and
            sha256(Path(m1102.M1072.M1016.__file__)) == M1016_SOURCE_SHA,
            "M1102/M1016 actual iterator identity drift")
    receipt = m1102.M1072.M1016.common_receipt(0, 64)
    expected_fields = {"task", "counts", "source_address_first",
                       "source_address_count", "weight_beat_first", "dma_first",
                       "psum_addresses", "commit_first"}
    require(set(receipt) == expected_fields and
            set(receipt["counts"]) == {"psum", "weight", "source", "dma", "commit"},
            "actual common receipt field drift")
    source = M1016_SOURCE.read_text(encoding="utf-8")
    common = function_node(source, "common_receipt")
    weight_task = method_node(source, "PackingAudit", "weight_task")
    run_full = function_node(source, "run_full")
    require([arg.arg for arg in weight_task.args.args] ==
            ["self", "start", "beats", "half_slot"],
            "actual weight interval API drift")
    run_text = ast.get_source_segment(source, run_full) or ast.unparse(run_full)
    common_text = ast.get_source_segment(source, common) or ast.unparse(common)
    require('"weight_beat_first"' in common_text and
            'receipt["counts"]["weight"]' in run_text and 'index & 1' in run_text and
            not any(token in common_text for token in
                    ("native_slice", "local_row", "byte_enable", "native_macro_activations")),
            "actual frozen service semantics drift")
    missing = [
        "native READ/WRITE operation",
        "logical weight bank",
        "native slice set",
        "local row",
        "bytes and byte enable",
        "native macro activation multiplicity",
        "service-beat to store-transaction exact-once relation",
    ]
    return {
        "schema": "m1128c_frozen_weight_service_interface_audit_v1",
        "status": "STOP_BEFORE_PAYLOAD_OPEN__FROZEN_WEIGHT_EVENTS_NOT_ADDRESSED",
        "available_receipt_fields": sorted(receipt),
        "available_weight_interval_arguments": ["start", "beats", "half_slot"],
        "task_parity_half_slot_summary": True,
        "missing_native_fields": missing,
        "count_or_weight_beat_first_expansion_allowed": False,
        "capacity_geometry_expansion_allowed": False,
        "canonical_row_reader_opened": False,
        "full_51840000_rows_read": False,
        "canonical_weight_transactions_emitted": 0,
        "canonical_ready": False,
    }


@dataclass(frozen=True)
class AddressedWeightTransaction:
    axis: str
    requested_cycle: int
    cycle: int
    stall_cycles: int
    op: str
    logical_bank: int
    half_slot: int
    logical_row: int
    local_row: int
    native_slices: tuple[int, ...]
    bytes: int
    byte_enable_per_slice: tuple[int, ...]
    native_macro_activations: int
    service_beat_ordinal: int | None
    store_transaction_ordinal: int
    source_task_id: int
    source_local_ordinal: int
    source_row_provenance_sha256: str

    def validate(self) -> None:
        require(self.axis in AXES and type(self.requested_cycle) is int and
                type(self.cycle) is int and self.cycle >= self.requested_cycle >= 0 and
                self.stall_cycles == self.cycle - self.requested_cycle and
                self.op in ("READ", "WRITE") and self.logical_bank == self.half_slot and
                self.half_slot in (0, 1) and 0 <= self.logical_row < ROWS_PER_HALF and
                self.local_row == self.half_slot * ROWS_PER_HALF + self.logical_row and
                0 <= self.local_row < NATIVE_DEPTH and
                self.native_slices == tuple(sorted(set(self.native_slices))) and
                all(0 <= value < NATIVE_SLICES for value in self.native_slices) and
                self.bytes == len(self.native_slices) * SLICE_BYTES and
                self.byte_enable_per_slice == (0xffff,) * len(self.native_slices) and
                self.native_macro_activations == len(self.native_slices) and
                type(self.store_transaction_ordinal) is int and
                self.store_transaction_ordinal >= 0 and
                type(self.source_task_id) is int and self.source_task_id >= 0 and
                type(self.source_local_ordinal) is int and self.source_local_ordinal >= 0 and
                len(self.source_row_provenance_sha256) == 64 and
                all(char in "0123456789abcdef"
                    for char in self.source_row_provenance_sha256),
                "addressed weight transaction schema/mapping drift")
        if self.op == "WRITE":
            require(type(self.service_beat_ordinal) is int and
                    self.service_beat_ordinal >= 0 and
                    len(self.native_slices) == REFILL_SLICES and
                    self.bytes == REFILL_BYTES and
                    min(self.native_slices) % REFILL_SLICES == 0 and
                    self.native_slices == tuple(range(min(self.native_slices),
                                                      min(self.native_slices) + REFILL_SLICES)),
                    "refill beat mapping drift")
        else:
            require(self.service_beat_ordinal is None and
                    self.native_slices == tuple(range(NATIVE_SLICES)) and
                    self.bytes == RECORD_BYTES,
                    "full-record read mapping drift")


def schedule_native_one_rw(requests: Iterable[AddressedWeightTransaction]) -> list[AddressedWeightTransaction]:
    rows = list(requests)
    require(rows, "empty synthetic request list")
    for row in rows:
        row.validate()
        require(row.cycle == row.requested_cycle and row.stall_cycles == 0,
                "scheduler input must be nominal")
    ordered = sorted(enumerate(rows), key=lambda pair: (
        pair[1].requested_cycle, pair[1].source_task_id,
        pair[1].source_local_ordinal, pair[1].store_transaction_ordinal, pair[0]))
    next_cycle: dict[tuple[str, int], int] = {}
    output = []
    for _, row in ordered:
        cycle = max([row.requested_cycle] +
                    [next_cycle.get((row.axis, native_slice), 0)
                     for native_slice in row.native_slices])
        scheduled = replace(row, cycle=cycle,
                            stall_cycles=cycle - row.requested_cycle)
        scheduled.validate()
        output.append(scheduled)
        for native_slice in row.native_slices:
            next_cycle[(row.axis, native_slice)] = cycle + 1
    return output


def validate_exact_once_and_conflicts(
        rows: Sequence[AddressedWeightTransaction],
        expected_service_beats: Mapping[str, set[int]]) -> dict[str, int]:
    identities = set()
    observed_beats = {axis: set() for axis in AXES}
    occupied = set()
    half_slot_occupied: dict[tuple[str, int, int], int] = {}
    stalls = 0
    store_writes = 0
    reads = 0
    for row in rows:
        row.validate()
        identity = (row.axis, row.source_task_id, row.source_local_ordinal,
                    row.store_transaction_ordinal)
        require(identity not in identities, "duplicate store transaction identity")
        identities.add(identity)
        if row.op == "WRITE":
            require(row.service_beat_ordinal not in observed_beats[row.axis],
                    "duplicate service beat mapping")
            observed_beats[row.axis].add(int(row.service_beat_ordinal))
            store_writes += 1
        else:
            reads += 1
        for native_slice in row.native_slices:
            key = (row.axis, native_slice, row.cycle)
            require(key not in occupied, "final native 1RW conflict")
            occupied.add(key)
            prior_half = half_slot_occupied.get(key)
            require(prior_half is None or prior_half == row.half_slot,
                    "final weight half-slot overlap")
            half_slot_occupied[key] = row.half_slot
        stalls += int(row.stall_cycles > 0)
    require(observed_beats == expected_service_beats,
            "service beat to store transaction is not exact-once")
    return {
        "transactions": len(rows),
        "unique_transaction_identities": len(identities),
        "refill_store_transactions": store_writes,
        "full_record_read_transactions": reads,
        "service_beats_expected": sum(map(len, expected_service_beats.values())),
        "service_beats_exact_once": sum(map(len, observed_beats.values())),
        "explicitly_stalled_transactions": stalls,
        "final_native_1rw_conflicts": 0,
        "final_weight_half_slot_overlaps": 0,
    }


def iter_canonical_weight_addressed_ledger() -> Iterator[dict[str, Any]]:
    """Future production entry; intentionally STOP before payload open."""
    audit = audit_frozen_service_interface()
    require(audit["canonical_ready"] is True,
            "STOP: frozen weight service lacks native op/bank/slices/row/bytes/"
            "byte-enable/activation/exact-once store relation")
    if False:  # pragma: no cover
        yield {}


def source_small_oracle() -> dict[str, Any]:
    audit = audit_frozen_service_interface()
    digest = "0" * 64
    requests = []
    expected = {axis: set() for axis in AXES}
    for axis_id, axis in enumerate(AXES):
        beat0 = axis_id * 2
        beat1 = beat0 + 1
        expected[axis].update((beat0, beat1))
        base = axis_id * 3
        requests.extend((
            AddressedWeightTransaction(axis, 5, 5, 0, "WRITE", 0, 0, 0, 0,
                tuple(range(8)), 128, (0xffff,) * 8, 8, beat0, base,
                0, 0, digest),
            AddressedWeightTransaction(axis, 5, 5, 0, "WRITE", 1, 1, 0, 16,
                tuple(range(8)), 128, (0xffff,) * 8, 8, beat1, base + 1,
                0, 1, digest),
            AddressedWeightTransaction(axis, 7, 7, 0, "READ", 0, 0, 0, 0,
                tuple(range(24)), 384, (0xffff,) * 24, 24, None, base + 2,
                0, 2, digest),
        ))
    scheduled = schedule_native_one_rw(requests)
    summary = validate_exact_once_and_conflicts(scheduled, expected)
    require(summary == {
        "transactions": 9,
        "unique_transaction_identities": 9,
        "refill_store_transactions": 6,
        "full_record_read_transactions": 3,
        "service_beats_expected": 6,
        "service_beats_exact_once": 6,
        "explicitly_stalled_transactions": 3,
        "final_native_1rw_conflicts": 0,
        "final_weight_half_slot_overlaps": 0,
    }, "bounded synthetic result drift")
    canonical_stopped = False
    try:
        next(iter_canonical_weight_addressed_ledger())
    except Failure:
        canonical_stopped = True
    require(canonical_stopped, "canonical iterator escaped frozen field STOP")
    return {
        "schema": "m1128c_weight_service_addressed_ledger_small_oracle_v1",
        "status": "PASS_SYNTHETIC_24X128X128B_MAPPING__CANONICAL_STOP",
        "frozen_interface_audit": audit,
        "synthetic_mapping": {
            "status": "synthetic/proposed only; not canonical H67 evidence",
            "native_macros": 24,
            "native_depth_rows": 128,
            "native_width_bits": 128,
            "same_mapping_for_all_three_axes": True,
        },
        "synthetic": summary,
        "canonical_iterator_stopped_before_payload": canonical_stopped,
        "full_51840000_replayed": False,
        "eda_rtl_gpu_remote_executed": False,
    }


def main() -> None:
    require(sys.argv == [str(Path(sys.argv[0]))] or sys.argv[1:] == ["--self-test"],
            "only zero-argument or --self-test bounded source is allowed")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()

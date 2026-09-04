#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1126C source-only C1 three-axis storage transaction exporter.

The frozen M1102 chain is sufficient to reconstruct candidate parent accesses
and arbitrated psum accesses with source-row provenance.  It is not sufficient
to assign exact native 1RW operations and addresses to the weight service.
Therefore the canonical iterator fails before opening the 51.84M-row payload.
Only the bounded synthetic schema/arbitration oracle is executable here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Iterator, Mapping

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

M1102_SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1102_RESULT = (HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/"
                "m1102_c1_work8_exact_1rw_full_replay_result_r1.json")
M1102_RESULT_SHA = "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91"
M1102_RESULT_OUTER = (HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/"
                      ".m1102_atomic_seal/SHA256SUMS.seal.sha256")
M1102_RESULT_OUTER_SHA = "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f"

M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
M1123C = HW / "reviews/m1123c_m1122c_c1_path_c_common_charge_independent_hammer_r1_20260830"
M1123C_ID = (
    "b2752ce9e805bb1cbadab2229b48c287df4d7321b6f442a8b004dc904ab43e82",
    "8ead4a34f4c418fbca9343b984144808f9d785dfd39595e293801ea94ceef724",
    "4c1679005159d75f3fda75a9adceb7b6b17d6baae77949b312ec5ecf3a0d73ae",
)
M1125C = HW / "reviews/m1125c_c1_path_c_105macro_common_model_first_principles_audit_r1_20260830"
M1125C_ID = (
    "348e18ebdcf37f1740bcd8b977885ee86ea5b0a172232413866f2c739879d77c",
    "e306057ae9d3b52700d1221d764426d98fcc13221ab905129f0fb1aaacc3d8d1",
    "a0c3d3e137a07fc09294dfaf1e4e806ba9be11117506e0bd1e5d3e476ac094b1",
)
CONTRACT = HW / "contracts/m1126c_c1_three_axis_storage_transaction_exporter_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
LIVE_CLASSES = ("parent", "psum", "weight")
RESIDUAL_BYTES = 24_448
REQUIRED_TRANSACTION_FIELDS = (
    "cycle", "requested_cycle", "stall_cycles", "axis", "storage_class",
    "bank", "address", "op", "bytes", "native_macro_activations",
    "half_slot", "source_task_id", "source_local_ordinal",
    "source_row_provenance_sha256",
)


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
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            "regular-file identity drift: " + str(path))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review, manifest, outer = (directory / "review.json", directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
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
    spec = importlib.util.spec_from_file_location("m1126c_frozen_m1102", M1102_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1102")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class StorageTransaction:
    cycle: int
    requested_cycle: int
    stall_cycles: int
    axis: str
    storage_class: str
    bank: int
    address: int
    op: str
    bytes: int
    native_macro_activations: int
    half_slot: int | None
    source_task_id: int
    source_local_ordinal: int
    source_row_provenance_sha256: str

    def validate(self) -> None:
        require(self.axis in AXES and self.storage_class in LIVE_CLASSES and
                type(self.cycle) is int and type(self.requested_cycle) is int and
                self.cycle >= self.requested_cycle >= 0 and
                type(self.stall_cycles) is int and
                self.stall_cycles == self.cycle - self.requested_cycle and
                type(self.bank) is int and self.bank >= 0 and
                type(self.address) is int and self.address >= 0 and
                self.op in ("READ", "WRITE") and type(self.bytes) is int and
                self.bytes > 0 and type(self.native_macro_activations) is int and
                self.native_macro_activations > 0 and
                type(self.source_task_id) is int and self.source_task_id >= 0 and
                type(self.source_local_ordinal) is int and
                self.source_local_ordinal >= 0 and
                type(self.source_row_provenance_sha256) is str and
                len(self.source_row_provenance_sha256) == 64 and
                all(value in "0123456789abcdef"
                    for value in self.source_row_provenance_sha256),
                "transaction schema/value drift")
        require((self.storage_class == "weight" and self.half_slot in (0, 1)) or
                (self.storage_class != "weight" and self.half_slot is None),
                "half-slot classification drift")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in REQUIRED_TRANSACTION_FIELDS}


def schedule_one_rw(requests: Iterable[StorageTransaction]) -> list[StorageTransaction]:
    """Deterministic per-axis/class/bank 1RW serializer with explicit stalls."""
    rows = list(requests)
    require(rows, "empty synthetic request population")
    for row in rows:
        row.validate()
        require(row.cycle == row.requested_cycle and row.stall_cycles == 0,
                "input to 1RW scheduler must be nominal")
    ordered = sorted(enumerate(rows), key=lambda pair: (
        pair[1].requested_cycle, pair[1].source_task_id,
        pair[1].source_local_ordinal, pair[0]))
    next_cycle: dict[tuple[str, str, int], int] = {}
    output = []
    for _, row in ordered:
        key = (row.axis, row.storage_class, row.bank)
        cycle = max(row.requested_cycle, next_cycle.get(key, 0))
        scheduled = replace(row, cycle=cycle,
                            stall_cycles=cycle - row.requested_cycle)
        scheduled.validate()
        output.append(scheduled)
        next_cycle[key] = cycle + 1
    require(len({(row.axis, row.storage_class, row.bank, row.cycle)
                 for row in output}) == len(output), "1RW final conflict")
    return output


def validate_exact_once(rows: Iterable[StorageTransaction]) -> dict[str, int]:
    rows = list(rows)
    identities = set()
    half_slot_cycles = set()
    stalls = 0
    for row in rows:
        row.validate()
        identity = (row.axis, row.storage_class, row.source_task_id,
                    row.source_local_ordinal)
        require(identity not in identities, "duplicate source transaction")
        identities.add(identity)
        require(row.storage_class != "residual", "residual access fabricated")
        if row.storage_class == "weight":
            key = (row.axis, row.bank, row.cycle)
            require(key not in half_slot_cycles,
                    "weight half slots overlap on shared 1RW group")
            half_slot_cycles.add(key)
        stalls += int(row.stall_cycles > 0)
    return {"transactions": len(rows), "unique_source_transactions": len(identities),
            "explicitly_stalled_transactions": stalls,
            "final_1rw_conflicts": 0, "weight_half_slot_overlaps": 0}


def audit_frozen_exportability() -> dict[str, Any]:
    """Audit API sufficiency without opening the canonical 51.84M-row file."""
    verify_flat(M1000, M1000_ID)
    verify_flat(M1123C, M1123C_ID)
    verify_flat(M1125C, M1125C_ID)
    verify_regular(M1102_RESULT, M1102_RESULT_SHA)
    verify_regular(M1102_RESULT_OUTER, M1102_RESULT_OUTER_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    contract = strict_json(CONTRACT)
    require(contract.get("status") ==
            "SOURCE_ONLY_FAIL_CLOSED_WEIGHT_TRANSACTION_PROVENANCE_GAP__DIFFERENT_AUTHOR_STATIC_HAMMER_ONLY" and
            contract.get("authorization", {}).get("full_export_now") is False,
            "M1126C contract boundary drift")
    m1102 = load_m1102()
    require(inspect.isgeneratorfunction(m1102.iter_canonical_full_replay_results) and
            tuple(m1102.DESIGNS) == AXES, "M1102 production interface drift")
    port_event_fields = tuple(field.name for field in fields(m1102.M1056.PortEvent))
    grant_fields = tuple(field.name for field in fields(m1102.M1056.Grant))
    provenance_fields = tuple(field.name for field in fields(
        m1102.M1072.ProvenanceRecord))
    receipt = m1102.M1072.M1016.common_receipt(0, 64)
    require(set(receipt) == {"task", "counts", "source_address_first",
            "source_address_count", "weight_beat_first", "dma_first",
            "psum_addresses", "commit_first"} and
            set(receipt["counts"]) == {"psum", "weight", "source", "dma", "commit"},
            "frozen common receipt schema drift")
    missing_weight = (
        "native 1RW operation (READ versus WRITE)",
        "local 24-slice macro address",
        "logical bytes and byte-enable per on-chip access",
        "native-macro activation multiplicity per access",
        "exact-once relation between weight_beat service and on-chip weight store",
    )
    return {
        "schema": "m1126c_frozen_exportability_preflight_v1",
        "status": "STOP_BEFORE_CANONICAL_ROW_OPEN__WEIGHT_TRANSACTION_PROVENANCE_INSUFFICIENT",
        "canonical_row_reader_opened": False,
        "full_51840000_source_rows_read": False,
        "transaction_rows_emitted": 0,
        "candidate_parent_reconstructable": (
            hasattr(m1102.M1072.M1016, "iter_parent_address_events")),
        "baseline_parent_zero_aggregate_sealed": True,
        "psum_reconstructable_with_explicit_1rw_grants": (
            all(name in port_event_fields for name in
                ("logical_bank", "address", "op", "base_ready_cycle")) and
            all(name in grant_fields for name in ("cycle", "group", "address", "op"))),
        "source_row_provenance_reconstructable": all(name in provenance_fields for name in
            ("task_id", "raw_row_bytes_sha256", "provenance_sha256")),
        "weight_receipt_available_fields": sorted(receipt),
        "missing_weight_transaction_authorities": list(missing_weight),
        "weight_half_slot_interval_summary_is_not_an_addressed_transaction": True,
        "residual_capacity_bytes": RESIDUAL_BYTES,
        "residual_transactions_permitted": False,
        "canonical_export_ready": False,
        "fail_closed": True,
    }


def iter_canonical_transactions() -> Iterator[dict[str, Any]]:
    """Future zero-argument canonical entry; intentionally STOP before payload open."""
    audit = audit_frozen_exportability()
    require(audit["canonical_export_ready"] is True,
            "STOP: frozen weight service lacks op/address/bytes/native-activation/"
            "exact-once on-chip-store provenance")
    if False:  # pragma: no cover - keeps the production interface a generator.
        yield {}


def source_small_oracle() -> dict[str, Any]:
    audit = audit_frozen_exportability()
    digest = "0" * 64
    requests = [
        StorageTransaction(7, 7, 0, "candidate", "psum", 0, 3, "READ",
                           240, 15, None, 0, 0, digest),
        StorageTransaction(7, 7, 0, "candidate", "psum", 0, 67, "WRITE",
                           240, 15, None, 0, 1, digest),
        StorageTransaction(9, 9, 0, "candidate", "weight", 0, 0, "READ",
                           384, 24, 0, 0, 2, digest),
        StorageTransaction(9, 9, 0, "candidate", "weight", 0, 16, "READ",
                           384, 24, 1, 0, 3, digest),
        StorageTransaction(10, 10, 0, "candidate", "parent", 0, 5, "WRITE",
                           144, 9, None, 0, 4, digest),
    ]
    scheduled = schedule_one_rw(requests)
    exact = validate_exact_once(scheduled)
    require([row.cycle for row in scheduled if row.storage_class == "psum"] == [7, 8] and
            [row.cycle for row in scheduled if row.storage_class == "weight"] == [9, 10] and
            exact == {"transactions": 5, "unique_source_transactions": 5,
                      "explicitly_stalled_transactions": 2,
                      "final_1rw_conflicts": 0, "weight_half_slot_overlaps": 0},
            "synthetic 1RW/stall oracle drift")
    residual_rejected = False
    try:
        StorageTransaction(0, 0, 0, "candidate", "residual", 0, 0, "READ",
                           16, 1, None, 0, 5, digest).validate()
    except Failure:
        residual_rejected = True
    require(residual_rejected, "residual fabricated-access attack admitted")
    duplicate_rejected = False
    try:
        validate_exact_once([scheduled[0], scheduled[0]])
    except Failure:
        duplicate_rejected = True
    require(duplicate_rejected, "duplicate exact-once attack admitted")
    canonical_stop = False
    try:
        next(iter_canonical_transactions())
    except Failure:
        canonical_stop = True
    require(canonical_stop, "canonical iterator escaped weight provenance STOP")
    return {
        "schema": "m1126c_storage_transaction_exporter_small_oracle_v1",
        "status": "PASS_SMALL_SYNTHETIC__CANONICAL_EXPORT_FAILS_CLOSED",
        "required_transaction_fields": list(REQUIRED_TRANSACTION_FIELDS),
        "synthetic": exact,
        "residual_access_attack_rejected": residual_rejected,
        "duplicate_exact_once_attack_rejected": duplicate_rejected,
        "canonical_iterator_stopped_before_row_open": canonical_stop,
        "exportability": audit,
        "full_replay_executed": False,
        "eda_rtl_gpu_remote_executed": False,
    }


def main() -> None:
    require(sys.argv == [str(Path(sys.argv[0]))] or sys.argv[1:] == ["--self-test"],
            "only zero-argument or --self-test source oracle is allowed")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive per-beat addressed weight-refill event producer source.

The emitter is intended to be called at a real refill-beat creation point.  It
accepts and emits the complete M1130C event object; it has no aggregate/count
adapter.  Frozen canonical producers are not modified or hooked here, so the
canonical iterator remains fail-closed before opening any row.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Iterator

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1130_SOURCE = HERE / "build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1130_SOURCE_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1130_CONTRACT = HW / "contracts/m1130c_c1_internal_weight_service_refill_instrumentation_source_contract_r1_20260830.json"
M1130_CONTRACT_ID = (
    "20ff9026f8dbc25ad0e9813107a6e97a96f1e379244dcb26ffb51d3a972bcfab",
    "49c2e9599a2c87807717f87f7c117844ad056cefe660c71bffb564d0413de745",
    "efc4bc08d3634531b99c1e45d1ce20c362bb5ca74249d9f2a6877b857af9352a",
)
M1130_AUTHOR = HW / "reviews/m1130c_c1_internal_weight_service_refill_instrumentation_author_receipt_r1_20260830"
M1130_AUTHOR_OUTER = "f9ce60c54bc016378cd7c0727cb471b0629de4a2e43b24567a7aeb40163efa36"
M1131 = HW / "reviews/m1131c_m1130c_c1_internal_weight_service_refill_instrumentation_static_hammer_r1_20260830"
M1131_OUTER = "b6363d0ae8e9cb7c845af3463a85ed2fbb8ef1c2bab3a760166f4795494ab20f"
CONTRACT = HW / "contracts/m1132c_c1_upstream_weight_event_producer_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "8218699210c481a5a8d2ddfc7b2fe1091b24ef36b004716dc530d9b193acec91",
    "be85e9a08684691c964c78f0b441a85a43a61c69a3d4014ae608a7c123526b4f",
    "9592d136ea18b86c722fb69af3422ef8106d5d5d628d8badbf1e5b079f8d9f07",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)


class Failure(RuntimeError): pass


def require(value: bool, message: str) -> None:
    if not value: raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == digest,
            "identity drift: " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1]); verify_regular(outer, identity[2])
    require(side.read_text().split() == [identity[0], path.name] and
            outer.read_text().split() == [identity[1], side.name], "double-seal content")


def verify_flat(directory: Path, outer_sha: str) -> None:
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, outer_sha)
    require(directory.is_dir() and not directory.is_symlink() and
            outer.read_text().split() == [sha256(manifest), "SHA256SUMS"], "flat outer")
    listed = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None, "manifest row")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name not in listed and name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "manifest member")
        listed[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(listed), "sealed member set")
    for name, digest in listed.items(): verify_regular(directory / name, digest)


_M1130 = None


def load_m1130():
    global _M1130
    if _M1130 is None:
        verify_regular(M1130_SOURCE, M1130_SOURCE_SHA)
        spec = importlib.util.spec_from_file_location("m1132c_frozen_m1130", M1130_SOURCE)
        require(spec is not None and spec.loader is not None, "cannot load M1130C")
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module); _M1130 = module
    return _M1130


def source_preflight() -> dict[str, Any]:
    verify_regular(M1130_SOURCE, M1130_SOURCE_SHA)
    verify_double(M1130_CONTRACT, M1130_CONTRACT_ID)
    verify_flat(M1130_AUTHOR, M1130_AUTHOR_OUTER)
    verify_flat(M1131, M1131_OUTER)
    verify_double(CONTRACT, CONTRACT_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    m1130 = load_m1130(); audit = m1130.audit_frozen_internal_event_point()
    require(audit["real_internal_weight_service_refill_event_available"] is False and
            audit["canonical_rows_read"] == 0 and audit["canonical_events_emitted"] == 0,
            "frozen canonical producer unexpectedly changed")
    return {
        "status": "STOP_CANONICAL_HOOK_NOT_INTEGRATED__ADDITIVE_PRODUCER_SOURCE_ONLY",
        "producer_source_exists": True, "real_callsite_integrated": False,
        "canonical_ready": False, "canonical_rows": 0, "canonical_events": 0,
    }


class PerBeatAddressedWeightRefillProducer:
    """Emit one exact M1130C WRITE event for one real producer refill beat."""
    def __init__(self, sink: Callable[[Any], None]):
        require(callable(sink), "event sink must be callable")
        self._sink = sink
        self._exact_ids: set[tuple[str, str]] = set()
        self._beats: set[tuple[str, int]] = set()
        self._transactions: set[tuple[str, int, int, int]] = set()
        self.emitted = 0

    def emit_refill_event(self, *, axis: str, task_id: int,
                          source_local_ordinal: int, requested_cycle: int,
                          op: str, logical_bank: int, half_slot: int,
                          logical_row: int, local_row: int,
                          native_slices: tuple[int, ...], bytes: int,
                          byte_enable_per_slice: tuple[int, ...],
                          native_macro_activations: int,
                          service_beat_ordinal: int,
                          store_transaction_ordinal: int,
                          service_event_exact_once_id: str,
                          source_row_provenance_sha256: str):
        require(op == "WRITE", "refill producer emits WRITE only")
        m1130 = load_m1130()
        event = m1130.InternalWeightServiceRefillEvent(
            axis, task_id, source_local_ordinal, requested_cycle, op,
            logical_bank, half_slot, logical_row, local_row, native_slices,
            bytes, byte_enable_per_slice, native_macro_activations,
            service_beat_ordinal, store_transaction_ordinal,
            service_event_exact_once_id, source_row_provenance_sha256)
        event.validate()
        exact_key = (axis, service_event_exact_once_id)
        beat_key = (axis, service_beat_ordinal)
        transaction_key = (axis, task_id, source_local_ordinal, store_transaction_ordinal)
        require(exact_key not in self._exact_ids, "duplicate producer exact-once ID")
        require(beat_key not in self._beats, "duplicate producer service beat")
        require(transaction_key not in self._transactions, "duplicate producer transaction identity")
        self._sink(event)  # Exceptions propagate; no aggregate or fallback path exists.
        self._exact_ids.add(exact_key); self._beats.add(beat_key)
        self._transactions.add(transaction_key); self.emitted += 1
        return event


def iter_canonical_upstream_weight_refill_events() -> Iterator[Any]:
    audit = source_preflight()
    require(audit["canonical_ready"] is True,
            "STOP: additive producer exists but frozen real callsite hook is not integrated")
    if False:  # pragma: no cover
        yield None


def source_small_oracle() -> dict[str, Any]:
    preflight = source_preflight(); m1130 = load_m1130(); emitted = []
    producer = PerBeatAddressedWeightRefillProducer(emitted.append)
    provenance = hashlib.sha256(b"m1132c-bounded-synthetic").hexdigest()
    for axis_index, axis in enumerate(AXES):
        for local in range(2):
            beat = axis_index * 2 + local; transaction = axis_index * 2 + local
            producer.emit_refill_event(
                axis=axis, task_id=axis_index, source_local_ordinal=local,
                requested_cycle=5, op="WRITE", logical_bank=0, half_slot=0,
                logical_row=axis_index, local_row=axis_index,
                native_slices=tuple(range(8)), bytes=128,
                byte_enable_per_slice=(0xffff,) * 8, native_macro_activations=8,
                service_beat_ordinal=beat, store_transaction_ordinal=transaction,
                service_event_exact_once_id=m1130.exact_once_id(
                    axis, axis_index, local, beat, transaction),
                source_row_provenance_sha256=provenance)
    require(producer.emitted == 6 and len(emitted) == 6, "one-call one-event conservation")
    scheduled = m1130.instrument_real_event_inputs(emitted)
    summary = {
        "producer_write_events": len(emitted),
        "unique_exact_once_write_ids": len({row.service_event_exact_once_id for row in emitted}),
        "post_schedule_stalled_transactions": sum(row.stall_cycles > 0 for row in scheduled),
        "post_schedule_native_1rw_conflicts": 0,
    }
    require(summary == {"producer_write_events": 6, "unique_exact_once_write_ids": 6,
                        "post_schedule_stalled_transactions": 3,
                        "post_schedule_native_1rw_conflicts": 0}, "bounded producer oracle drift")
    stopped = False
    try: next(iter_canonical_upstream_weight_refill_events())
    except Failure: stopped = True
    require(stopped, "canonical iterator escaped unintegrated hook")
    return {
        "schema": "m1132c_upstream_weight_event_producer_small_oracle_v1",
        "status": "PASS_ADDITIVE_PER_BEAT_PRODUCER_SYNTHETIC__CANONICAL_STOP",
        "preflight": preflight, "synthetic": summary,
        "canonical_rows": 0, "canonical_events": 0,
        "full_51840000_replay": False, "eda_rtl_gpu_remote": False,
    }


def main() -> None:
    require(sys.argv[1:] in ([], ["--self-test"]), "only bounded self-test allowed")
    print(json.dumps(source_small_oracle(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()

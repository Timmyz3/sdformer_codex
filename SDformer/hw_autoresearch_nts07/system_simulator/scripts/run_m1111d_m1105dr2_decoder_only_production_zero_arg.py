#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1111D zero-argument decoder-only address-timed production runner.

SOURCE ONLY until a different-author final runner hammer pins this exact file,
its contract and its sealed author receipt.  Importing this module or calling
``source_static_self_test`` never opens the canonical M699 payload and never
creates a production namespace.
"""
from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HERE = SOURCE_FILE.parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
SOURCE = HERE / "build_m1105dr2_decoder_only_address_timed_source.py"
SOURCE_SHA = "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4"
SOURCE_CONTRACT = HW / "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json"
SOURCE_CONTRACT_ID = (
    "cdbae0362d3ea093dbcb318aa2efad04e70677f8d984a9908cda44b0de3b80a4",
    "37cdc8aa6b0c31103affa46f1aea80f073689540b16a40ea0eec68904a0fb4fe",
    "4f95a616e16530bc30f94b68235247f7c7abe1b32956fc981412b3b1576193d3",
)
SOURCE_RECEIPT = HW / "reviews/m1105dr2_decoder_source_trust_root_author_receipt_r1_20260830"
SOURCE_RECEIPT_ID = (
    "16a628bb69d12b41a421d16dc1af5a9da0ae7593cfeeb9105a71ebc57bd9f952",
    "e05ddc0c29c371e6a9b719a9e167b59ac2cecc33a51aac2959d0bd4b2a558cd8",
    "d16257e342be49f6e895bd1ca4b4c764eb6200da47bd27aa221abba7e6f6af25",
)
M1110D = HW / "reviews/m1110d_m1105dr2_decoder_source_contract_receipt_independent_hammer_r1_20260830"
M1110D_ID = (
    "feb6f554e15da36650d5d5d220d8bd75c2acb2f1c4a86dadb0d8359548285f7a",
    "96c4450b26ecec7c1a8ea5516a4d4e301eac3a3e25798aae99e67e38ed7ba65a",
    "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
)
MAPPER = HERE / "map_m672_decoder_convtranspose_polyphase_workload_r3.py"
MAPPER_SHA = "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254"
MAPPER_R2 = HERE / "map_m670_decoder_convtranspose_polyphase_workload_r2.py"
MAPPER_R2_SHA = "875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b"
CONTRACT = HW / "contracts/m1111d_m1105dr2_decoder_only_production_runner_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "82bba9ed495f8b1d316ea02647e9f28868c4845da69f95723132f7921a8535f6",
    "71f636ba43fc27321dda569e1133770cd77c9065974a2e974cd81b9b950df90e",
    "7a94c5ce60291f76e8a460886486271868d628e523d96c8c63080dea6409b4dc",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = HW / "results/m1111d_m1105dr2_decoder_only_address_timed_production_r1_20260830"
ATTEMPT = HW / "results/.m1111d_m1105dr2_decoder_only_production_attempt_consumed"
LOCK = HW / "results/.m1111d_m1105dr2_decoder_only_production.lock"
WORK_PREFIX = ".m1111d_m1105dr2_decoder_only_production_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
PAYLOAD = "m1111d_decoder_result.json"
CALLS = "m1111d_decoder_call_schedule.jsonl"
SEAL_DIR = ".m1111d_atomic_seal"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"

CONFIGURATION = "M1105DR2_EXACT_TYPED_K8"
KINDS = ("input_descriptor_read", "weight_read", "psum_read", "compute",
         "psum_write", "output_commit")
MODULE_GEOMETRY = {
    0: (1536, 384, 15, 20, 30, 40),
    1: (770, 192, 30, 40, 60, 80),
    2: (386, 96, 60, 80, 120, 160),
    3: (194, 96, 120, 160, 240, 320),
}
THETA_WORD = 1065353139
THETA_HEX = "b3ff7f3f"
EXPECTED_CALLS = 120
MIN_MEM_AVAILABLE_KIB = 4 * 1024 * 1024
MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024


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
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            "regular-file identity drift: " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double-seal content drift")


def verify_flat(directory: Path, identity: tuple[str, str, str], status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(),
            "flat authority directory drift")
    review = directory / "review.json"
    manifest = directory / MANIFEST
    outer = directory / OUTER
    require((sha256(review), sha256(manifest), sha256(outer)) == identity and
            outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "flat authority root drift")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in seen and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, "unsafe flat member")
        verify_regular(member, digest); seen.add(relative)
    require(strict_json(review).get("status") == status, "flat authority status drift")


def load_module(path: Path, expected: str, name: str):
    verify_regular(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_mapper():
    verify_regular(MAPPER_R2, MAPPER_R2_SHA)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return load_module(MAPPER, MAPPER_SHA, "m1111d_frozen_m672")


def validate_authorities(require_fresh: bool = True) -> dict[str, Any]:
    verify_regular(PYTHON, PYTHON_SHA)
    verify_regular(SOURCE, SOURCE_SHA)
    verify_regular(MAPPER, MAPPER_SHA)
    verify_regular(MAPPER_R2, MAPPER_R2_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    verify_double(SOURCE_CONTRACT, SOURCE_CONTRACT_ID)
    verify_double(CONTRACT, CONTRACT_ID)
    verify_flat(SOURCE_RECEIPT, SOURCE_RECEIPT_ID,
                "PASS_M1105DR2_FIXED_TRUST_SOURCE_AUTHOR_RECEIPT__INDEPENDENT_HAMMER_REQUIRED")
    verify_flat(M1110D, M1110D_ID,
                "PASS_M1110D_M1105DR2_FIXED_TRUST_SOURCE_HAMMER__RUNNER_AUTHORING_ONLY")
    contract = strict_json(CONTRACT)
    scope = contract["production_scope"]
    boundary = contract["claim_boundary"]
    require(contract["status"] ==
            "SOURCE_ONLY__DIFFERENT_AUTHOR_FINAL_RUNNER_HAMMER_REQUIRED__NO_PRODUCTION" and
            contract["runner"]["arguments"] == 0 and
            contract["runner"]["maximum_attempts_after_final_hammer"] == 1 and
            contract["runner"]["automatic_retry"] is False and
            scope["calls"] == EXPECTED_CALLS and
            scope["m700_external_input_allowed"] is False and
            scope["final_checkpoint_rebind_required"] is True and
            scope["d1"]["theta_word_uint32"] == THETA_WORD and
            scope["d1"]["theta_ieee754_le_hex"] == THETA_HEX and
            scope["d1"]["weight_folding_allowed"] is False and
            boundary["system_speedup_admitted"] is False and
            boundary["paper_ppa_ready"] is False,
            "M1111D contract semantic drift")
    if require_fresh:
        require(namespace_fresh(), "M1111D production namespace not fresh")
    return {"status": "PASS_M1111D_HARDCODED_AUTHORITIES_NO_PAYLOAD_NO_ATTEMPT",
            "calls": EXPECTED_CALLS, "m700_external_input": False,
            "final_checkpoint_rebind_required": True,
            "system_speedup_admitted": False, "paper_ppa_ready": False}


def namespace_fresh(ignore_lock: bool = False) -> bool:
    return (not RESULT.exists() and not RESULT.is_symlink() and
            not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            (ignore_lock or (not LOCK.exists() and not LOCK.is_symlink())) and
            not any(RESULT.parent.glob(WORK_PREFIX + "*")) and
            not any(RESULT.parent.glob(FAILURE_PREFIX + "*")))


def sanitize_environment() -> None:
    os.environ.clear()
    os.environ.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                       "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    require(all(key in values for key in ("MemAvailable", "CommitLimit", "Committed_AS")),
            "meminfo schema drift")
    return values


def resource_gate() -> dict[str, int]:
    info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB and
            headroom >= MIN_COMMIT_HEADROOM_KIB, "insufficient production resources")
    return {"mem_available_kib": info["MemAvailable"],
            "commit_headroom_kib": headroom}


def write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "renameat2 unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise Failure("atomic no-replace collision")
        raise OSError(code, os.strerror(code), str(destination))


def payload_files(directory: Path) -> list[Path]:
    files = []
    for item in sorted(directory.rglob("*")):
        relative = item.relative_to(directory)
        if relative.parts and relative.parts[0] == SEAL_DIR:
            continue
        require(not item.is_symlink(), "seal refuses symlink")
        if item.is_file():
            files.append(item)
    return files


def atomic_seal(directory: Path) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink() and
            not (directory / SEAL_DIR).exists(), "bad seal target")
    members = payload_files(directory)
    require(members, "empty seal target")
    stage = directory.parent / (directory.name + ".sealstage.%d.%d" %
                                (os.getpid(), time.time_ns()))
    stage.mkdir(mode=0o700)
    lines = [sha256(item) + "  " + item.relative_to(directory).as_posix()
             for item in members]
    write_exclusive(stage / MANIFEST, ("\n".join(lines) + "\n").encode())
    write_exclusive(stage / OUTER,
                    (sha256(stage / MANIFEST) + "  " + MANIFEST + "\n").encode())
    fsync_dir(stage); rename_noreplace(stage, directory / SEAL_DIR); fsync_dir(directory)
    return verify_atomic_seal(directory)


def verify_atomic_seal(directory: Path) -> dict[str, Any]:
    bundle = directory / SEAL_DIR
    manifest, outer = bundle / MANIFEST, bundle / OUTER
    require(bundle.is_dir() and not bundle.is_symlink() and
            manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink() and
            outer.read_text(encoding="utf-8") ==
                sha256(manifest) + "  " + MANIFEST + "\n", "atomic seal drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        member = directory / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == digest,
                "atomic member drift")
        listed[relative] = digest
    actual = {item.relative_to(directory).as_posix() for item in payload_files(directory)}
    require(set(listed) == actual, "atomic coverage drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer), "members": len(actual)}


@dataclass(frozen=True)
class Port:
    banks: int
    mode: str
    row_bytes: int
    read_latency: int
    write_latency: int
    initiation_interval: int
    outstanding: int


PORTS = {
    "weight": Port(8, "1R1W", 16, 4, 1, 1, 8),
    "psum": Port(6, "1RW", 48, 2, 1, 1, 8),
    "external": Port(1, "1RW", 192, 32, 3, 1, 16),
    "compute": Port(1, "1RW", 288, 1, 1, 1, 1),
}


@dataclass
class KindSummary:
    count: int = 0
    traffic_bytes: int = 0
    address_first: int | None = None
    address_last: int | None = None
    issue_first: int | None = None
    issue_last: int | None = None
    return_first: int | None = None
    return_last: int | None = None
    commit_first: int | None = None
    commit_last: int | None = None
    stalls: Counter = field(default_factory=Counter)

    def update(self, addresses: tuple[int, ...], bytes_charged: int,
               issue: int, returned: int, committed: int, stall: str) -> None:
        self.count += 1; self.traffic_bytes += bytes_charged
        if addresses:
            self.address_first = addresses[0] if self.address_first is None else self.address_first
            self.address_last = addresses[-1]
        self.issue_first = issue if self.issue_first is None else self.issue_first
        self.issue_last = issue
        self.return_first = returned if self.return_first is None else self.return_first
        self.return_last = returned
        self.commit_first = committed if self.commit_first is None else self.commit_first
        self.commit_last = committed
        self.stalls[stall] += 1

    def receipt(self) -> dict[str, Any]:
        return {"count": self.count, "traffic_bytes": self.traffic_bytes,
                "address_first": self.address_first, "address_last": self.address_last,
                "issue_first": self.issue_first, "issue_last": self.issue_last,
                "return_first": self.return_first, "return_last": self.return_last,
                "commit_first": self.commit_first, "commit_last": self.commit_last,
                "stall_events": dict(sorted(self.stalls.items()))}


@dataclass
class CallAudit:
    call: Mapping[str, Any]
    first_ordinal: int
    start_cycle: int
    kinds: dict[str, KindSummary] = field(
        default_factory=lambda: {name: KindSummary() for name in KINDS})
    address_digest: Any = field(default_factory=hashlib.sha256)
    dependency_digest: Any = field(default_factory=hashlib.sha256)
    schedule_digest: Any = field(default_factory=hashlib.sha256)
    transactions: int = 0
    last_ordinal: int = -1
    end_cycle: int = 0

    def consume(self, ordinal: int, kind: str, addresses: tuple[int, ...],
                banks: tuple[int, ...], width: int, dependencies: tuple[str, ...],
                produces: str, issue: int, returned: int, committed: int,
                stall: str, bytes_charged: int, identity: tuple[int, int, int, int]) -> None:
        require(kind in self.kinds and ordinal == self.first_ordinal + self.transactions,
                "call transaction order drift")
        for bank, address in zip(banks, addresses):
            self.address_digest.update(json.dumps(
                [ordinal, kind, bank, address, width], separators=(",", ":")).encode())
        self.dependency_digest.update(json.dumps(
            [ordinal, list(dependencies), produces], separators=(",", ":")).encode())
        self.schedule_digest.update(json.dumps(
            [ordinal, issue, returned, committed, stall, list(identity)],
            separators=(",", ":")).encode())
        self.kinds[kind].update(addresses, bytes_charged, issue, returned, committed, stall)
        self.transactions += 1; self.last_ordinal = ordinal
        self.end_cycle = max(self.end_cycle, committed + 1)

    def receipt(self) -> dict[str, Any]:
        call = self.call
        require(self.transactions > 0 and self.last_ordinal >= self.first_ordinal,
                "empty call schedule")
        traffic = {kind: row.traffic_bytes for kind, row in self.kinds.items()}
        return {
            "schema": "m1111d_decoder_address_timed_call_schedule_v1",
            "global_call_ordinal": int(call["global_ordinal"]),
            "sequence_ordinal": int(call["sequence_ordinal"]),
            "sequence": str(call["sequence"]),
            "sequence_sample_id": int(call["sequence_sample_id"]),
            "module_ordinal": int(call["module_ordinal"]),
            "module": str(call["module"]),
            "configuration": CONFIGURATION,
            "d1_exact_theta": (int(call["module_ordinal"]) == 1),
            "d1_theta_word_uint32": THETA_WORD if int(call["module_ordinal"]) == 1 else None,
            "d1_weight_folding": False,
            "transaction_ordinal_first": self.first_ordinal,
            "transaction_ordinal_last": self.last_ordinal,
            "transaction_count": self.transactions,
            "address_digest_sha256": self.address_digest.hexdigest(),
            "dependency_digest_sha256": self.dependency_digest.hexdigest(),
            "schedule_digest_sha256": self.schedule_digest.hexdigest(),
            "cycle_start": self.start_cycle,
            "cycle_end": self.end_cycle,
            "diagnostic_cycles": self.end_cycle - self.start_cycle,
            "diagnostic_traffic_bytes": {**traffic, "total": sum(traffic.values()),
                "external": traffic["input_descriptor_read"] + traffic["output_commit"],
                "onchip": traffic["weight_read"] + traffic["psum_read"] + traffic["psum_write"]},
            "kind_summaries": {kind: self.kinds[kind].receipt() for kind in KINDS},
            "claim_boundary": {"diagnostic_only": True, "speedup_admitted": False,
                "system_speedup_admitted": False, "paper_ppa_ready": False,
                "final_checkpoint_rebind_required": True}
        }


class Scheduler:
    def __init__(self) -> None:
        self.next_port: dict[tuple[str, int, str], int] = {}
        self.outstanding: dict[tuple[str, int], list[int]] = {}
        self.ordinal = 0
        self.end_cycle = 0

    @staticmethod
    def mapping(kind: str) -> tuple[str, str]:
        if kind == "input_descriptor_read": return "external", "read"
        if kind == "weight_read": return "weight", "read"
        if kind == "psum_read": return "psum", "read"
        if kind == "compute": return "compute", "write"
        if kind == "psum_write": return "psum", "write"
        require(kind == "output_commit", "unknown transaction kind")
        return "external", "write"

    def issue(self, audit: CallAudit, kind: str, addresses: Iterable[int],
              banks: Iterable[int], width: int, dependency_ready: int,
              dependencies: tuple[str, ...], produces: str,
              earliest: int, identity: tuple[int, int, int, int]) -> int:
        addresses = tuple(int(value) for value in addresses)
        banks = tuple(int(value) for value in banks)
        require(len(addresses) == len(banks) and banks and width > 0 and
                dependency_ready >= 0 and earliest >= 0 and produces,
                "transaction shape drift")
        resource_name, operation = self.mapping(kind)
        port = PORTS[resource_name]
        require(all(0 <= bank < port.banks for bank in banks) and
                len(banks) == len(set(banks)), "transaction bank drift")
        port_name = "rw" if port.mode == "1RW" else operation
        port_bound = max((self.next_port.get((resource_name, bank, port_name), 0)
                          for bank in banks), default=0)
        initial = max(earliest, dependency_ready, port_bound)
        outstanding_bound = initial
        changed = True
        while changed:
            changed = False
            for bank in banks:
                occupied = sorted(value for value in self.outstanding.get(
                    (resource_name, bank), []) if value > outstanding_bound)
                if len(occupied) >= port.outstanding:
                    proposed = occupied[len(occupied) - port.outstanding]
                    if proposed > outstanding_bound:
                        outstanding_bound = proposed; changed = True
        issue = max(initial, outstanding_bound)
        if issue == earliest: stall = "none"
        elif issue == dependency_ready: stall = "dependency_completion"
        elif issue == outstanding_bound and outstanding_bound > initial: stall = "outstanding"
        else: stall = resource_name + "_port"
        latency = port.read_latency if operation == "read" else port.write_latency
        denominator = 192 if resource_name == "external" else port.row_bytes
        beats = max(1, math.ceil(width / denominator))
        returned = issue + latency + beats - 1
        for bank in banks:
            self.next_port[(resource_name, bank, port_name)] = issue + max(
                port.initiation_interval, beats)
            self.outstanding[(resource_name, bank)] = [
                value for value in self.outstanding.get((resource_name, bank), [])
                if value > issue] + [returned]
        bytes_charged = width if resource_name == "external" else width * len(banks)
        ordinal = self.ordinal; self.ordinal += 1
        audit.consume(ordinal, kind, addresses, banks, width, dependencies,
                      produces, issue, returned, returned, stall, bytes_charged, identity)
        self.end_cycle = max(self.end_cycle, returned + 1)
        return returned


def bank_unique_groups(indices: Iterable[int], channels: int) -> list[tuple[int, ...]]:
    queues = [[] for _ in range(8)]
    for value in indices:
        value = int(value); require(value >= 0, "negative flat-K")
        queues[(value % channels) % 8].append(value)
    groups = [tuple(queue[index] for queue in queues if index < len(queue))
              for index in range(max((len(queue) for queue in queues), default=0))]
    require(sum(len(group) for group in groups) == sum(len(queue) for queue in queues),
            "source packing conservation drift")
    return groups


def execute_call(mapper, scheduler: Scheduler, canonical: Mapping[str, Any],
                 call: Mapping[str, Any]) -> dict[str, Any]:
    module = int(call["module_ordinal"])
    require(module in MODULE_GEOMETRY and int(call["global_ordinal"]) < EXPECTED_CALLS,
            "call coordinate drift")
    cin, cout, hin, win, hout, wout = MODULE_GEOMETRY[module]
    require(tuple(int(value) for value in call["input_shape"]) ==
            (10, 1, cin, hin, win), "call shape drift")
    payload = Path(canonical["trust_root"]["canonical_payload"]) / str(
        call["payload_relative_path"])
    verify_regular(payload, str(call["payload_sha256"]))
    call_start = scheduler.end_cycle
    audit = CallAudit(call, scheduler.ordinal, call_start)
    call_index = int(call["global_ordinal"])
    output_blocks = math.ceil(cout / 96)
    descriptor_ordinal = 0
    for timestep in range(10):
        previous: dict[tuple[int, int], tuple[str, int]] = {}
        for tile in mapper.iter_polyphase_tiles(
                payload, tuple(call["input_shape"]), tile_m=256,
                trusted_root=Path(canonical["trust_root"]["canonical_payload"]).resolve()):
            phase = int(tile["phase_bank"])
            values = tile["values"][timestep]
            for local_m, (dy, dx) in enumerate(zip(
                    tile["destination_y"], tile["destination_x"])):
                destination = int(dy) * wout + int(dx)
                active = [int(value) for value in values[local_m].nonzero()[0]]
                groups = bank_unique_groups(active, cin)
                for output_block in range(output_blocks):
                    key = (destination, output_block)
                    for group_ordinal, group in enumerate(groups):
                        prefix = "c%d:t%d:p%d:d%d:b%d:g%d" % (
                            call_index, timestep, phase, destination,
                            output_block, group_ordinal)
                        identity = (timestep, phase, destination, output_block)
                        desc_address = ((1 << 60) + (call_index << 32) +
                                        descriptor_ordinal * 16)
                        desc_token = prefix + ":descriptor"
                        desc_ready = scheduler.issue(audit, "input_descriptor_read",
                            (desc_address,), (0,), 16, call_start, (), desc_token,
                            call_start, identity)
                        descriptor_ordinal += 1
                        weight_addresses = tuple((2 << 60) + (module << 32) +
                            (output_block << 24) + value * 16 for value in group)
                        weight_banks = tuple((value % cin) % 8 for value in group)
                        weight_token = prefix + ":weight"
                        weight_ready = scheduler.issue(audit, "weight_read",
                            weight_addresses, weight_banks, 16, call_start, (),
                            weight_token, call_start, identity)
                        psum_base = ((3 << 60) + (call_index << 32) +
                            ((timestep * hout * wout * output_blocks +
                              destination * output_blocks + output_block) * 288))
                        predecessor_token, predecessor_ready = previous.get(
                            key, ("", call_start))
                        read_token = prefix + ":psum_read"
                        read_dependencies = ((predecessor_token,)
                                             if predecessor_token else ())
                        read_ready = scheduler.issue(audit, "psum_read",
                            tuple(psum_base + bank * 48 for bank in range(6)),
                            tuple(range(6)), 48, predecessor_ready,
                            read_dependencies, read_token, call_start, identity)
                        compute_token = prefix + ":compute"
                        compute_ready = scheduler.issue(audit, "compute",
                            (psum_base,), (0,), 288,
                            max(desc_ready, weight_ready, read_ready),
                            (desc_token, weight_token, read_token), compute_token,
                            call_start, identity)
                        write_token = prefix + ":psum_write"
                        write_ready = scheduler.issue(audit, "psum_write",
                            tuple(psum_base + bank * 48 for bank in range(6)),
                            tuple(range(6)), 48, compute_ready, (compute_token,),
                            write_token, call_start, identity)
                        previous[key] = (write_token, write_ready)
        for destination in range(hout * wout):
            phase = ((destination // wout) & 1) * 2 + (destination % wout & 1)
            for output_block in range(output_blocks):
                key = (destination, output_block)
                predecessor_token, predecessor_ready = previous.get(key, ("", call_start))
                identity = (timestep, phase, destination, output_block)
                address = ((4 << 60) + (call_index << 32) +
                    ((timestep * hout * wout * output_blocks +
                      destination * output_blocks + output_block) * 288))
                token = "c%d:t%d:d%d:b%d:commit" % (
                    call_index, timestep, destination, output_block)
                scheduler.issue(audit, "output_commit", (address,), (0,), 288,
                    predecessor_ready, ((predecessor_token,) if predecessor_token else ()),
                    token, call_start, identity)
    require(descriptor_ordinal * 16 < (1 << 32) and
            10 * hout * wout * output_blocks * 288 < (1 << 32) and
            output_blocks * (1 << 24) + 9 * cin * 16 < (1 << 32),
            "per-call/module address region overflow")
    receipt = audit.receipt()
    require(receipt["global_call_ordinal"] == call_index and
            receipt["configuration"] == CONFIGURATION and
            receipt["claim_boundary"]["system_speedup_admitted"] is False,
            "call receipt boundary drift")
    return receipt


def execute_production(work: Path) -> dict[str, Any]:
    require(ATTEMPT.is_dir() and not ATTEMPT.is_symlink() and
            not work.exists() and not RESULT.exists(), "production state drift")
    work.mkdir(mode=0o700)
    source = load_module(SOURCE, SOURCE_SHA, "m1111d_frozen_m1105dr2")
    mapper = load_mapper()
    canonical = source.build_canonical()
    require(canonical["status"] ==
            "PASS_M1105DR2_FIXED_TRUST_SOURCE_PREFLIGHT__PRODUCTION_NOT_RELEASED" and
            canonical["population"]["calls"] == EXPECTED_CALLS and
            canonical["external_baseline_rejection"]["m700_admitted"] is False and
            canonical["input_identity"]["final_checkpoint_rebind_required_if_changed"] is True and
            canonical["d1_exact_scaled_binary_miter"]["mismatches"] == 0,
            "canonical decoder source drift")
    scheduler = Scheduler()
    call_digest = hashlib.sha256()
    traffic = Counter()
    call_rows = 0
    call_path = work / CALLS
    with call_path.open("xb") as stream:
        for expected, call in enumerate(canonical["calls"]):
            require(int(call["global_ordinal"]) == expected, "120-call order drift")
            row = execute_call(mapper, scheduler, canonical, call)
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode()
            stream.write(encoded); call_digest.update(encoded)
            traffic.update(row["diagnostic_traffic_bytes"])
            call_rows += 1
        stream.flush(); os.fsync(stream.fileno())
    require(call_rows == EXPECTED_CALLS and scheduler.ordinal > 0 and
            scheduler.end_cycle > 0, "production population drift")
    result = {
        "schema": "m1111d_m1105dr2_decoder_only_address_timed_result_v1",
        "status": "PASS_M1111D_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED",
        "identity": {"checkpoint": "H67_ep35",
            "checkpoint_sha256": canonical["input_identity"]["checkpoint_sha256"],
            "source_sha256": SOURCE_SHA, "contract_sha256": CONTRACT_ID[0],
            "m1110d_outer_seal_file_sha256": M1110D_ID[2],
            "m700_external_input": False,
            "final_checkpoint_rebind_required": True},
        "population": {"calls": call_rows, "timesteps_per_call": 10,
            "transaction_count": scheduler.ordinal,
            "call_schedule_sha256": sha256(call_path),
            "call_row_stream_digest_sha256": call_digest.hexdigest()},
        "common_resource": strict_json(CONTRACT)["common_resource"],
        "diagnostic": {"cycles": scheduler.end_cycle,
            "traffic_bytes": dict(traffic), "ratios_or_speedups": None},
        "claim_boundary": {"decoder_only": True,
            "address_timed_transactions_complete": True,
            "same_resource_schedule_complete": True,
            "diagnostic_cycles_only": True, "diagnostic_traffic_only": True,
            "speedup_admitted": False, "system_speedup_admitted": False,
            "paper_ppa_ready": False, "paper_citable_performance": False,
            "final_checkpoint_rebind_required": True,
            "independent_result_hammer_required": True}
    }
    write_exclusive(work / PAYLOAD,
        (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    write_exclusive(work / "RUN_COMPLETE.txt",
        b"M1111D_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n")
    return {"result": result, "seal": atomic_seal(work)}


def consume_attempt() -> dict[str, Any]:
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and not RESULT.exists(),
            "attempt collision")
    ATTEMPT.mkdir(mode=0o700); fsync_dir(ATTEMPT.parent)
    receipt = {"schema": "m1111d_decoder_production_attempt_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": 1, "automatic_retry": False,
        "canonical_payload_opened_before_attempt": False,
        "runner_sha256": sha256(SOURCE_FILE),
        "contract_sha256": CONTRACT_ID[0]}
    write_exclusive(ATTEMPT / "attempt.json",
                    (json.dumps(receipt, sort_keys=True) + "\n").encode())
    return {"receipt": receipt, "seal": atomic_seal(ATTEMPT)}


def acquire_lock() -> None:
    try:
        LOCK.mkdir(mode=0o700)
    except FileExistsError as error:
        raise Failure("launch lock collision") from error
    write_exclusive(LOCK / "owner.json", (json.dumps({"pid": os.getpid(),
        "maximum_attempts": 1, "automatic_retry": False}, sort_keys=True) + "\n").encode())
    fsync_dir(LOCK.parent)


def release_lock() -> None:
    require(LOCK.is_dir() and not LOCK.is_symlink(), "lock identity drift")
    owner = LOCK / "owner.json"
    require(owner.is_file() and not owner.is_symlink(), "lock owner drift")
    owner.unlink(); LOCK.rmdir(); fsync_dir(LOCK.parent)


def quarantine_work(work: Path, quarantine: Path, phase: str) -> dict[str, Any]:
    require(quarantine.parent == RESULT.parent and
            quarantine.name.startswith(FAILURE_PREFIX) and
            not quarantine.exists() and not quarantine.is_symlink(),
            "quarantine path/collision drift")
    stage = Path(str(quarantine) + ".stage")
    require(not stage.exists() and not stage.is_symlink(), "quarantine stage collision")
    stage.mkdir(mode=0o700)
    if work.exists():
        rename_noreplace(work, stage / "partial_result")
    write_exclusive(stage / "failure.json", (json.dumps({
        "schema": "m1111d_decoder_failure_quarantine_v1",
        "status": "FAILED_OR_INTERRUPTED__NO_RETRY", "phase": phase,
        "attempt_consumed": True, "automatic_retry": False},
        sort_keys=True) + "\n").encode())
    seal = atomic_seal(stage); rename_noreplace(stage, quarantine)
    fsync_dir(RESULT.parent)
    return {"status": "PASS_M1111D_SEALED_FAILURE_QUARANTINE", "seal": seal}


def publish_result(work: Path) -> dict[str, Any]:
    require(work.parent == RESULT.parent and work.name.startswith(WORK_PREFIX) and
            not RESULT.exists() and not RESULT.is_symlink(), "publish path/collision drift")
    seal = verify_atomic_seal(work)
    payload = strict_json(work / PAYLOAD)
    require(payload["status"] ==
            "PASS_M1111D_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED" and
            payload["claim_boundary"]["system_speedup_admitted"] is False and
            payload["claim_boundary"]["paper_ppa_ready"] is False,
            "publish claim boundary drift")
    rename_noreplace(work, RESULT); fsync_dir(RESULT.parent)
    require(verify_atomic_seal(RESULT) == seal, "published seal drift")
    return {"status": payload["status"], "result": str(RESULT), "seal": seal}


def source_static_self_test() -> dict[str, Any]:
    identities = validate_authorities(require_fresh=True)
    require(not SOURCE_FILE.is_symlink() and SOURCE_FILE.is_file(), "runner path drift")
    scheduler = Scheduler()
    call = {"global_ordinal": 0, "sequence_ordinal": 0,
            "sequence": "synthetic", "sequence_sample_id": 0,
            "module_ordinal": 1, "module": "D1", "input_shape": [10, 1, 770, 30, 40]}
    audit = CallAudit(call, 0, 0)
    a = scheduler.issue(audit, "input_descriptor_read", (1 << 60,), (0,), 16,
                        0, (), "d", 0, (0, 3, 0, 0))
    b = scheduler.issue(audit, "weight_read", (2 << 60,), (0,), 16,
                        0, (), "w", 0, (0, 3, 0, 0))
    c = scheduler.issue(audit, "psum_read", (3 << 60,), (0,), 48,
                        0, (), "r", 0, (0, 3, 0, 0))
    d = scheduler.issue(audit, "compute", (3 << 60,), (0,), 288,
                        max(a, b, c), ("d", "w", "r"), "c", 0, (0, 3, 0, 0))
    e = scheduler.issue(audit, "psum_write", (3 << 60,), (0,), 48,
                        d, ("c",), "p", 0, (0, 3, 0, 0))
    scheduler.issue(audit, "output_commit", (4 << 60,), (0,), 288,
                    e, ("p",), "o", 0, (0, 3, 0, 0))
    receipt = audit.receipt()
    require(receipt["transaction_count"] == 6 and
            set(receipt["kind_summaries"]) == set(KINDS) and
            receipt["d1_exact_theta"] is True and
            receipt["d1_theta_word_uint32"] == THETA_WORD and
            receipt["d1_weight_folding"] is False and
            receipt["claim_boundary"]["system_speedup_admitted"] is False,
            "synthetic schedule oracle drift")
    return {"status": "PASS_M1111D_RUNNER_SOURCE_STATIC_SELF_TEST__NO_PRODUCTION",
            "identities": identities, "synthetic_transactions": 6,
            "all_six_kinds": True, "d1_theta_exact": True,
            "m700_external_input": False,
            "final_checkpoint_rebind_required": True,
            "launcher_main_called": False, "attempt_created": False,
            "canonical_payload_opened": False, "production_replay_executed": False,
            "system_speedup_admitted": False, "paper_ppa_ready": False}


def interrupted(signum, _frame) -> None:
    raise Failure("M1111D interrupted by signal %d" % int(signum))


def main() -> int:
    require(len(sys.argv) == 1, "M1111D accepts zero arguments")
    require(Path(sys.executable).resolve() == PYTHON and
            tuple(sys.version_info[:3]) == (3, 10, 18) and
            sys.flags.isolated == 1 and sys.flags.no_user_site == 1,
            "M1111D requires pinned isolated Python")
    validate_authorities(require_fresh=True)
    sanitize_environment()
    resource_gate()
    require(namespace_fresh(), "namespace changed before lock")
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    locked = False; attempted = False
    work = RESULT.parent / (WORK_PREFIX + "%d.%d" % (os.getpid(), time.time_ns()))
    quarantine = RESULT.parent / (FAILURE_PREFIX + "%d.%d.quarantine" %
                                  (os.getpid(), time.time_ns()))
    phase = "PRE_ATTEMPT"
    try:
        acquire_lock(); locked = True
        require(namespace_fresh(ignore_lock=True), "namespace changed under lock")
        phase = "CONSUME_ATTEMPT"
        consume_attempt(); attempted = True
        phase = "BUILD_120_CALL_ADDRESS_TIMED_SCHEDULE"
        execute_production(work)
        phase = "ATOMIC_NO_REPLACE_PUBLISH"
        published = publish_result(work)
        print(json.dumps(published, sort_keys=True))
        return 0
    except BaseException:
        failure = traceback.format_exc()
        if attempted:
            try:
                quarantine_work(work, quarantine, phase)
            except BaseException:
                sys.stderr.write("M1111D_QUARANTINE_FAILURE\n" + traceback.format_exc())
        sys.stderr.write("M1111D_FAIL_CLOSED phase=" + phase + "\n" + failure)
        return 1
    finally:
        if locked:
            release_lock()


if __name__ == "__main__":
    raise SystemExit(main())

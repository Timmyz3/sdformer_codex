#!/usr/bin/env python3
"""M1296 canonical, single-use, FD-rooted decoder-annex publisher.

This additive successor does not replay the decoder.  Its public production
surface is the zero-argument ``publish_canonical`` function.  Synthetic tests
may exercise the underscored implementation with temporary fixtures only.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Callable


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
HW = REPO / "hw_autoresearch_nts07"
OLD = HERE.with_name("build_m1284_decoder_completion_gate_diagnostic_annex_successor.py")
OLD_SHA = "a0b5747b63f857cda594765fb7ed1d4837295327af477f73ef27f5a36635eb02"
OLD_CONTRACT = HW / "contracts/m1284_decoder_completion_gate_diagnostic_annex_successor_source_contract_r1_20260830.json"
OLD_CONTRACT_SHA = "db774e9851343b1f79e9272199e933fe07a0fb837a6ccc1629e7e32add074008"
CONTRACT = HW / "contracts/m1296_decoder_completion_atomic_publisher_successor_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

NAME = "m1296_h67_ep35_decoder_only_diagnostic_annex_r1_20260830"
SCHEMA = "m1296_h67_ep35_decoder_only_diagnostic_annex_r1_v1"
STATUS = "PASS_M1296_EP35_DECODER_DIAGNOSTIC_ONLY__RESULT_HAMMER_REQUIRED"
TOKEN = "M1296_EP35_DECODER_DIAGNOSTIC_ANNEX_COMPLETE__RESULT_HAMMER_REQUIRED\n"
SEAL = ".m1296_atomic_seal"
MARKER = ".m1296_decoder_annex_publication_consumed"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class PublishError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise PublishError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _regular(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PublishError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            label + " identity drift")


_regular(OLD, OLD_SHA, "M1284 predecessor")
_regular(OLD_CONTRACT, OLD_CONTRACT_SHA, "M1284 predecessor contract")
_spec = importlib.util.spec_from_file_location("m1296_frozen_m1284", OLD)
require(_spec is not None and _spec.loader is not None, "cannot import M1284")
A = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = A
_spec.loader.exec_module(A)


def canonical_layout():
    old = A.canonical_layout()
    return A.P.Layout(old.parent, old.result, old.attempt, old.lock, old.work,
                      old.parent / NAME)


@dataclass
class Snapshot:
    parent_fd: int
    root_fd: int
    root_dev: int
    root_ino: int
    files: list[dict[str, Any]]

    @property
    def root_path(self) -> Path:
        return Path("/proc/self/fd") / str(self.root_fd)

    def close(self) -> None:
        for fd in (self.root_fd, self.parent_fd):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _open_file_at(parent_fd: int, relative: str) -> int:
    return os.open(relative, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                   dir_fd=parent_fd)


def _file_identity(parent_fd: int, relative: str) -> dict[str, Any]:
    fd = _open_file_at(parent_fd, relative)
    try:
        item = os.fstat(fd)
        require(stat.S_ISREG(item.st_mode), relative + " is not a regular file")
        return {"path": relative, "sha256": _sha_fd(fd), "size": item.st_size,
                "device": item.st_dev, "inode": item.st_ino}
    finally:
        os.close(fd)


def _snapshot(layout, runner) -> Snapshot:
    parent_fd = os.open(layout.parent, os.O_RDONLY | os.O_DIRECTORY |
                        os.O_CLOEXEC | os.O_NOFOLLOW)
    root_fd = -1
    try:
        # Cooperative writers serialize on the result parent; the result root
        # lock and inode checks additionally bind every validation to one object.
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        root_fd = os.open(layout.result.name, os.O_RDONLY | os.O_DIRECTORY |
                          os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        root = os.fstat(root_fd)
        require(stat.S_ISDIR(root.st_mode), "result root is not directory")
        top = set(os.listdir(root_fd))
        require(top == {A.P.PAYLOAD, A.P.CALLS, "RUN_COMPLETE.txt", runner.SEAL_DIR},
                "canonical result member set drift")
        seal_fd = os.open(runner.SEAL_DIR, os.O_RDONLY | os.O_DIRECTORY |
                          os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            require(set(os.listdir(seal_fd)) == {runner.MANIFEST, runner.OUTER},
                    "canonical result seal member set drift")
        finally:
            os.close(seal_fd)
        names = [A.P.PAYLOAD, A.P.CALLS, "RUN_COMPLETE.txt",
                 runner.SEAL_DIR + "/" + runner.MANIFEST,
                 runner.SEAL_DIR + "/" + runner.OUTER]
        files = [_file_identity(root_fd, name) for name in names]
        return Snapshot(parent_fd, root_fd, root.st_dev, root.st_ino, files)
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _same_path_identity(layout, snap: Snapshot) -> None:
    item = os.stat(layout.result.name, dir_fd=snap.parent_fd, follow_symlinks=False)
    require(stat.S_ISDIR(item.st_mode) and item.st_dev == snap.root_dev and
            item.st_ino == snap.root_ino, "canonical result root replacement detected")


def _full_identity(snap: Snapshot) -> dict[str, Any]:
    return {"root_device": snap.root_dev, "root_inode": snap.root_ino,
            "files": [dict(item) for item in snap.files]}


def _revalidate_snapshot(layout, runner, snap: Snapshot) -> dict[str, Any]:
    _same_path_identity(layout, snap)
    current = [_file_identity(snap.root_fd, item["path"]) for item in snap.files]
    require(current == snap.files, "FD-rooted canonical result identity drift")
    # The frozen runner intentionally rejects /proc/self/fd symlink spellings.
    # Run its semantic oracle through the canonical pathname while the parent
    # and result FDs remain locked, bracketing it with root-inode and full
    # FD-member checks.  Hash authority itself remains the held descriptors.
    checked = runner.validate_publish_candidate(layout.result)
    rows = A.P.read_rows(layout.result / A.P.CALLS, runner, full=True)
    gate = {"state": "COMPLETE", "rows": rows, "checked": checked,
            "source_result": layout.result, "published": True, "replay": False}
    gate = A.validate_complete_gate(layout, runner, gate)
    require((snap.root_path / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            A.P.COMPLETE, "FD-rooted completion token drift")
    _same_path_identity(layout, snap)
    return gate


EXPECTED_CLAIMS = {
    "ep35_only": True, "decoder_only": True, "diagnostic_only": True,
    "final_checkpoint_rebind_required": True, "ratio_or_speedup": False,
    "table_a": False, "full_network": False, "system_speedup": False,
    "energy": False, "ppa": False, "paper_headline": False,
    "canonical_single_use": True, "production_replay": False,
    "independent_result_hammer_required": True,
}


def _build_payload(layout, runner, gate: dict[str, Any], snap: Snapshot) -> dict[str, Any]:
    # Reuse the frozen, exact M1284 projection/type checker, then replace only
    # its source identity and protocol namespace with exact M1296 fields.
    canonical_gate = dict(gate)
    canonical_gate["source_result"] = layout.result
    payload = A.P.annex_payload(canonical_gate)
    payload["schema"] = SCHEMA
    payload["status"] = STATUS
    checked = gate["checked"]
    payload["source_result"] = {
        "path": str(layout.result.relative_to(REPO)) if layout.result.is_relative_to(REPO)
                else str(layout.result),
        "payload_sha256": sha256(snap.root_path / A.P.PAYLOAD),
        "call_schedule_sha256": checked["payload"]["population"]["call_schedule_sha256"],
        "completion_token_sha256": sha256(snap.root_path / "RUN_COMPLETE.txt"),
        "atomic_seal": checked["seal"],
        "fd_rooted_snapshot": _full_identity(snap),
    }
    payload["claim_boundary"] = dict(EXPECTED_CLAIMS)
    _validate_payload(layout, runner, gate, snap, payload)
    return payload


def _validate_payload(layout, runner, gate, snap: Snapshot, payload) -> None:
    expected_keys = {"schema", "status", "source_result", "identity", "population",
                     "common_resource", "diagnostic", "module_breakdown",
                     "sequence_breakdown", "claim_boundary"}
    require(type(payload) is dict and set(payload) == expected_keys,
            "M1296 annex top-level schema drift")
    require(payload["schema"] == SCHEMA and payload["status"] == STATUS,
            "M1296 annex schema/status drift")
    require(payload["claim_boundary"] == EXPECTED_CLAIMS and
            all(type(value) is bool for value in payload["claim_boundary"].values()),
            "M1296 annex exact claim boundary drift")
    source = payload["source_result"]
    require(set(source) == {"path", "payload_sha256", "call_schedule_sha256",
            "completion_token_sha256", "atomic_seal", "fd_rooted_snapshot"},
            "M1296 source-result schema drift")
    require(source["fd_rooted_snapshot"] == _full_identity(snap) and
            source["payload_sha256"] == sha256(snap.root_path / A.P.PAYLOAD) and
            source["completion_token_sha256"] == sha256(
                snap.root_path / "RUN_COMPLETE.txt") and
            source["atomic_seal"] == gate["checked"]["seal"],
            "M1296 FD-rooted source identity drift")
    # M1284 remains the exact scalar/schema oracle for all inherited fields.
    probe = dict(payload)
    probe["schema"] = A.ANNEX_SCHEMA
    probe["status"] = A.ANNEX_STATUS
    probe["source_result"] = {
        "path": source["path"], "payload_sha256": source["payload_sha256"],
        "call_schedule_sha256": source["call_schedule_sha256"],
        "atomic_seal": source["atomic_seal"]}
    probe["claim_boundary"] = {
        key: EXPECTED_CLAIMS[key] for key in (
            "ep35_only", "decoder_only", "diagnostic_only",
            "final_checkpoint_rebind_required", "ratio_or_speedup", "table_a",
            "full_network", "system_speedup", "energy", "ppa", "paper_headline",
            "independent_result_hammer_required")}
    A.validate_annex(layout, runner, gate["checked"], probe)


def _write_exclusive(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                 os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            require(count > 0, "short exclusive write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                 os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _seal_stage(stage: Path) -> dict[str, Any]:
    require({p.name for p in stage.iterdir()} == {"annex.json", "RUN_COMPLETE.txt"},
            "M1296 stage member set drift")
    bundle = stage / SEAL
    bundle.mkdir(mode=0o700)
    manifest = bundle / MANIFEST
    lines = [sha256(stage / name) + "  " + name + "\n"
             for name in ("RUN_COMPLETE.txt", "annex.json")]
    _write_exclusive(manifest, "".join(lines).encode())
    outer = bundle / OUTER
    _write_exclusive(outer, (sha256(manifest) + "  " + MANIFEST + "\n").encode())
    _fsync_dir(bundle); _fsync_dir(stage)
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer), "members": 2}


def _marker_write(fd: int, value: dict[str, Any]) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()
    os.ftruncate(fd, 0); os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(data)
    while view:
        count = os.write(fd, view); require(count > 0, "short marker write")
        view = view[count:]
    os.fsync(fd)


def _exact_contract() -> dict[str, Any]:
    return {
        "schema": "m1296_decoder_completion_atomic_publisher_successor_source_contract_r1_v1",
        "status": "ADDITIVE_ATOMIC_PUBLISHER_SOURCE__AUTHOR_TEST_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "date": "2026-08-30",
        "source": {"path": str(HERE.relative_to(REPO)), "sha256": sha256(HERE),
                   "arguments": 0},
        "frozen_m1284": {"source_path": str(OLD.relative_to(REPO)),
            "source_sha256": OLD_SHA, "contract_path": str(OLD_CONTRACT.relative_to(REPO)),
            "contract_sha256": OLD_CONTRACT_SHA, "modified": False},
        "publication": {"canonical_annex": str((HW / "results" / NAME).relative_to(REPO)),
            "persistent_marker": MARKER, "atomic_no_replace": True,
            "fd_rooted_snapshot": True, "parent_and_result_flock": True,
            "second_full_identity_check_after_stage": True,
            "native_token": TOKEN.rstrip("\n"), "native_seal": SEAL,
            "failure_marker_retained": True, "automatic_rollback": False,
            "automatic_retry": False},
        "claim_boundary": {"source_only": True, "production_executed": False,
            "canonical_annex_written": False, "gpu": False, "remote": False,
            "eda": False, "table_a": False, "system_speedup": False,
            "paper_ppa_ready": False, "different_author_hammer_required": True},
        "docs359_sha256": DOCS359_SHA,
    }


def verify_static_authorities() -> None:
    _regular(DOCS359, DOCS359_SHA, "docs/359")
    require(A.P.strict_json(CONTRACT) == _exact_contract(),
            "M1296 exact source contract drift")


def _publish_once(layout, runner, *, alive: Callable[[int], bool],
                  cmdline: Callable[[int], bytes],
                  after_stage: Callable[[Any, Snapshot, Path], None] | None = None,
                  check_static: bool = False) -> dict[str, Any]:
    require(layout.annex.name == NAME and layout.annex.parent == layout.parent,
            "M1296 destination is not the sole canonical annex")
    if check_static:
        verify_static_authorities()
    gate0 = A.P.completion_gate(layout, runner, alive=alive, cmdline=cmdline)
    if gate0.get("state") != "COMPLETE":
        raise A.P.Incomplete("M1296 incomplete; marker and output untouched")
    A.validate_complete_gate(layout, runner, gate0)
    snap = _snapshot(layout, runner)
    marker_fd = -1
    stage: Path | None = None
    committed = False
    try:
        gate = _revalidate_snapshot(layout, runner, snap)
        marker_path = layout.parent / MARKER
        marker_fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                            os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
        _marker_write(marker_fd, {"schema": "m1296_publication_marker_v1",
            "state": "PENDING", "canonical_annex": NAME,
            "result_root_device": snap.root_dev, "result_root_inode": snap.root_ino,
            "automatic_retry": False, "automatic_rollback": False})
        _fsync_dir(layout.parent)
        require(not layout.annex.exists() and not layout.annex.is_symlink(),
                "M1296 canonical annex namespace not fresh")
        payload = _build_payload(layout, runner, gate, snap)
        stage = layout.parent / ("." + NAME + ".stage.%d.%d" %
                                 (os.getpid(), time.time_ns()))
        stage.mkdir(mode=0o700)
        _write_exclusive(stage / "annex.json",
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        _write_exclusive(stage / "RUN_COMPLETE.txt", TOKEN.encode())
        seal = _seal_stage(stage)
        if after_stage is not None:
            after_stage(layout, snap, stage)
        # Required last-mile check: the exact same locked directory FD, every
        # opened member identity, full runner seal/schema, and canonical root
        # inode are revalidated only after the stage is sealed.
        final_gate = _revalidate_snapshot(layout, runner, snap)
        _validate_payload(layout, runner, final_gate, snap,
                          A.P.strict_json(stage / "annex.json"))
        marker_stat = os.fstat(marker_fd)
        marker_path_stat = os.stat(MARKER, dir_fd=snap.parent_fd,
                                   follow_symlinks=False)
        require((marker_stat.st_dev, marker_stat.st_ino) ==
                (marker_path_stat.st_dev, marker_path_stat.st_ino),
                "M1296 publication marker replacement detected")
        A.P.rename_noreplace(stage, layout.annex)
        committed = True
        _fsync_dir(layout.parent)
        _marker_write(marker_fd, {"schema": "m1296_publication_marker_v1",
            "state": "COMMITTED", "canonical_annex": NAME,
            "annex_json_sha256": sha256(layout.annex / "annex.json"),
            "seal": seal, "automatic_retry": False, "automatic_rollback": False})
        return {"status": STATUS, "path": str(layout.annex), "seal": seal,
                "marker": str(marker_path), "replay": False, "table_a": False}
    except BaseException as exc:
        # Deliberate semantics: the O_EXCL marker and any sealed stage survive.
        # There is no automatic rollback and no automatic retry.  If rename had
        # already committed, the annex also survives and requires forensic
        # resolution; evidence is never silently removed.
        if marker_fd >= 0:
            try:
                _marker_write(marker_fd, {"schema": "m1296_publication_marker_v1",
                    "state": "FAILED_AFTER_COMMIT" if committed else "FAILED",
                    "canonical_annex": NAME, "error_type": type(exc).__name__,
                    "stage_retained": None if stage is None else str(stage),
                    "annex_retained": committed, "automatic_retry": False,
                    "automatic_rollback": False})
            except BaseException:
                pass
        raise
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        snap.close()


def publish_canonical() -> dict[str, Any]:
    """The only public production publisher; accepts no paths or payloads."""
    return _publish_once(canonical_layout(), A.P.load_runner(), alive=A.P.pid_alive,
                         cmdline=A.P.pid_cmdline, check_static=True)


def main() -> int:
    require(len(sys.argv) == 1, "M1296 accepts zero arguments")
    try:
        print(json.dumps(publish_canonical(), sort_keys=True, allow_nan=False))
        return 0
    except A.P.Incomplete:
        sys.stderr.write("M1296_INCOMPLETE__NO_MARKER_NO_OUTPUT_NO_REPLAY\n")
        return 75
    except BaseException as exc:
        sys.stderr.write("M1296_FAIL_CLOSED__MARKER_AND_STAGE_RETAINED_NO_REPLAY: %s\n" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

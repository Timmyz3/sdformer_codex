#!/opt/conda/envs/sdformerflow/bin/python
"""M1253 fail-closed one-shot successor for the M1248 binder release.

Import is inert.  A future zero-argument production invocation snapshots all
eleven candidate inputs before atomically consuming one attempt.  The exact
M1241/M1234/M1228 execution bytes are copied into write-sealed memfd objects;
the child can therefore execute only those preflighted bytes.  The complete
four-candidate receipt is rebound to the pre-attempt snapshot and to a closed
claim boundary.  No retry, GPU, training, capture, remote, or EDA action exists.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Sequence


REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python")
PYTHON_VERSION = "3.10.20"

M1248_PINS = MappingProxyType({
    Path("hw_autoresearch_nts07/scripts/run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py"):
        "17827df585a76ebd6e2858e8142dafd77d9f0594e758bcfc5578b0d2c34c932c",
    Path("hw_autoresearch_nts07/tests/test_run_m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source.py"):
        "ace9c5a6334ebfc563b1e3844bcdd8639a5dae2c92c77630ddc2ca15496d1ed9",
    Path("hw_autoresearch_nts07/contracts/m1248_m1241_motion_cross_run_final_checkpoint_binder_one_shot_release_source_contract_r1_20260830.json"):
        "38ed07d027e67545e187f0a3d7d484e0d923dd31dfb39e61bc46d4ae147dd668",
})
M1241_SOURCE_REL = Path(
    "hw_autoresearch_nts07/scripts/build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py")
M1234_SOURCE_REL = Path(
    "hw_autoresearch_nts07/scripts/build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py")
M1228_SOURCE_REL = Path(
    "hw_autoresearch_nts07/scripts/build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py")
EXECUTION_PINS = MappingProxyType({
    M1241_SOURCE_REL: "10e97e31362064e63eea153ea087fc0e04379b172516bbe6061a6a249bce5f9b",
    M1234_SOURCE_REL: "570ff4a6762a2ec9822a6161fb2f666becd6706a26586fe137f81b16fb188d0b",
    M1228_SOURCE_REL: "9b2b43b4d36ed64741cbb39db0d9f5d75eb7bec09b00f4e496f3d52ce3ae5efe",
})
M1241_AUX_PINS = MappingProxyType({
    Path("hw_autoresearch_nts07/tests/test_build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py"):
        "c941406ee5e1de6d680d035f5496d5dddf2ea062fbd2c2c3f9399327948559b7",
    Path("hw_autoresearch_nts07/contracts/m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor_source_contract_r1_20260830.json"):
        "9493529d5399f1557d236cdab714fa4addb21666a9916db6ea3427f64924b02a",
})

M1251_REL = Path(
    "hw_autoresearch_nts07/reviews/m1251_m1248_production_binder_one_shot_release_source_hammer_r1_20260830")
M1251_MANIFEST_SHA256 = "7ee0ee7049c0070ab6f022651697638fa3d45dd98b51e86835df00740e920699"
M1251_OUTER_SHA256 = "79c0176d92b2d6fde4635765d5afe309a2d188a7425722c394fbdbaa6e2728ad"
M1251_SCHEMA = "m1251_m1248_production_binder_one_shot_release_source_hammer_r1_v1"
M1251_STATUS = "BLOCK_M1251_M1248_RELEASE__TOCTOU_AND_RESULT_AUTHORITY_GAPS__SUCCESSOR_REQUIRED"

DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

OLD_RUN_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/date_two_contribution_full30_20260826/c12_binary_motion_ttx")
NEW_RUN_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830")
OLD_CONFIG_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml")
NEW_CONFIG_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml")
NEW_MANIFEST_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.json")

OUTPUT_REL = Path(
    "hw_autoresearch_nts07/results/m1253_motion_cross_run_final_checkpoint_selection_r4_20260830")
ATTEMPT_REL = Path(
    "hw_autoresearch_nts07/results/.m1253_motion_cross_run_final_checkpoint_selection_r4_attempt_consumed")
LOG_REL = Path(
    "hw_autoresearch_nts07/results/m1253_motion_cross_run_final_checkpoint_selection_r4_20260830.launch.log")

MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RESULT_PAYLOADS = frozenset({
    "RUN_COMPLETE.txt", "e0_e8_activation_rebind_targets.json",
    "final_checkpoint_selection.json", "four_checkpoint_metrics.csv",
    "selected_checkpoint_and_config.json",
})
RESULT_SCHEMA = "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1"
RESULT_STATUS = (
    "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
    "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY")
RUN_COMPLETE = (
    "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
    "FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n")
CHILD_TOKEN = (
    "PASS_M1253_SEALED_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
    "FRESH_RESULT_HAMMER_REQUIRED")
EXACT_PAIRS = MappingProxyType({
    "legacy_ep29": 29, "resume_ep30": 30,
    "resume_ep32": 32, "resume_ep34": 34,
})
ERROR_METRIC_KEYS = (
    "AEE", "AAE", "AAE_Benchmark", "AEE_PE1", "AEE_PE2", "AEE_PE3",
    "AEE_outliers", "DSEC_Fl",
)
EXACT_CLAIM_BOUNDARY = MappingProxyType({
    "selection_bound_after_execution": True,
    "fresh_result_hammer_required": True,
    "hardware_rebind_authorized": False,
    "hardware_replay_complete": False,
    "hardware_speedup": False,
    "system_speedup": False,
    "power_or_energy": False,
    "checkpoint_copied": False,
    "gpu_started_by_binder": False,
    "remote_access_by_binder": False,
    "eda_started_by_binder": False,
})


SEALED_LAUNCHER = r'''
import os,sys,types
from pathlib import Path
def load(fd, name, filename):
    os.lseek(fd, 0, os.SEEK_SET)
    blocks=[]
    while True:
        block=os.read(fd, 1<<20)
        if not block: break
        blocks.append(block)
    module=types.ModuleType(name)
    module.__file__=filename
    module.__package__=""
    sys.modules[name]=module
    exec(compile(b"".join(blocks), filename, "exec"), module.__dict__)
    return module
m1241=load(int(sys.argv[1]), "m1253_sealed_m1241", "build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py")
m1234=load(int(sys.argv[2]), "m1253_sealed_m1234", "build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py")
m1228=load(int(sys.argv[3]), "m1253_sealed_m1228", "build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py")
m1234.load_predecessor=lambda: m1228
m1241.load_predecessor=lambda: m1234
original_freeze=m1241.freeze_file
def enriched_freeze(*args, **kwargs):
    frozen=original_freeze(*args, **kwargs)
    frozen.public_identity["device"]=frozen.physical_identity[0]
    frozen.public_identity["inode"]=frozen.physical_identity[1]
    return frozen
m1241.freeze_file=enriched_freeze
result=m1241.build(m1234.PRODUCTION_POLICY)
m1241.write_receipt(Path(sys.argv[4]), result)
print("PASS_M1253_SEALED_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__FRESH_RESULT_HAMMER_REQUIRED")
print("selected_candidate="+result["selected"]["candidate_id"])
print("selected_epoch="+str(result["selected"]["epoch"]))
'''


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key in {}: {}".format(label, key))
            value[key] = item
        return value
    def constant(value):
        raise ReleaseError("non-finite JSON constant in {}: {}".format(label, value))
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError("invalid {} JSON".format(label)) from exc
    require(isinstance(value, dict), label + " root must be object")
    return value


def strict_json(path: Path, label: str) -> dict[str, Any]:
    _, payload = snapshot_file(path, label)
    return strict_json_payload(payload, label)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _physical(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _open_chain(absolute: str, final_directory: bool = False) -> int:
    require(os.path.isabs(absolute) and os.path.normpath(absolute) == absolute,
            "path must be normalized absolute: " + absolute)
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open("/", dir_flags)
    parts = [part for part in Path(absolute).parts if part != "/"]
    require(bool(parts), "root is not a valid target")
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not last or final_directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            successor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = successor
        return current
    except Exception:
        os.close(current)
        raise


@dataclass(frozen=True)
class FileSnapshot:
    absolute_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    mode: int

    def receipt_identity(self) -> dict[str, Any]:
        return {
            "absolute_path": self.absolute_path, "sha256": self.sha256,
            "size_bytes": self.size_bytes, "mtime_ns": self.mtime_ns,
            "device": self.device, "inode": self.inode,
        }


def snapshot_file(path: Path, label: str) -> tuple[FileSnapshot, bytes]:
    absolute = os.fspath(path)
    try:
        before = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, absolute)) from exc
    require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
            label + " must be a non-symlink regular file")
    try:
        descriptor = _open_chain(absolute)
    except OSError as exc:
        raise ReleaseError("cannot descriptor-open " + label) from exc
    try:
        fd_before = os.fstat(descriptor)
        require(_identity(before) == _identity(fd_before), label + " pre-read identity mismatch")
        blocks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(absolute)
    require(_identity(fd_before) == _identity(fd_after) == _identity(after),
            label + " changed during snapshot")
    require(len(payload) == fd_after.st_size, label + " byte population mismatch")
    return FileSnapshot(
        absolute, sha256_bytes(payload), fd_after.st_size, fd_after.st_mtime_ns,
        fd_after.st_dev, fd_after.st_ino, fd_after.st_mode), payload


def directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode),
            label + " must be a non-symlink directory")


def executable(path: Path, label: str) -> None:
    try:
        target = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ReleaseError("invalid " + label) from exc
    require(stat.S_ISREG(target.stat().st_mode) and os.access(target, os.X_OK),
            label + " target must be executable regular file")


def verify_double_seal(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, Any]:
    directory(root, "M1251 review")
    manifest = root / MANIFEST
    outer = root / OUTER
    manifest_snapshot, manifest_payload = snapshot_file(manifest, "M1251 manifest")
    outer_snapshot, outer_payload = snapshot_file(outer, "M1251 outer")
    require(manifest_snapshot.sha256 == manifest_sha, "M1251 manifest SHA drift")
    require(outer_snapshot.sha256 == outer_sha, "M1251 outer SHA drift")
    require(outer_payload.decode("utf-8").split() == [manifest_sha, MANIFEST],
            "M1251 outer content mismatch")
    rows = {}
    review_payload = None
    for line in manifest_payload.decode("utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "invalid M1251 manifest row")
        name = fields[1].lstrip("*")
        require(Path(name).name == name and name not in rows, "invalid M1251 member")
        member_snapshot, member_payload = snapshot_file(root / name, "M1251 member " + name)
        require(member_snapshot.sha256 == fields[0], "M1251 member SHA drift: " + name)
        rows[name] = fields[0]
        if name == "review.json":
            review_payload = member_payload
    require("review.json" in rows, "M1251 review.json missing")
    require(review_payload is not None, "M1251 review payload missing")
    review = strict_json_payload(review_payload, "M1251 review")
    require(review.get("schema") == M1251_SCHEMA and review.get("status") == M1251_STATUS,
            "M1251 schema/status mismatch")
    authority = review.get("authority")
    require(isinstance(authority, dict) and
            authority.get("production_execution_authorized_now") is False and
            authority.get("future_execution_authorized_by_M1251") is False and
            authority.get("release_successor_authoring_required") is True and
            authority.get("fresh_different_author_successor_hammer_required") is True,
            "M1251 authority mismatch")
    return review


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    run_rel: Path
    config_key: str
    config_rel: Path
    epoch: int


@dataclass(frozen=True)
class Policy:
    repo: Path
    interpreter: Path
    python_version: str
    authority_pins: dict[Path, str]
    execution_pins: dict[Path, str]
    aux_pins: dict[Path, str]
    review_rel: Path
    review_manifest_sha256: str
    review_outer_sha256: str
    docs_rel: Path
    docs_sha256: str
    candidates: tuple[CandidateInput, ...]
    manifest_rel: Path
    output_rel: Path
    attempt_rel: Path
    log_rel: Path


PRODUCTION_POLICY = Policy(
    REPO, INTERPRETER, PYTHON_VERSION, dict(M1248_PINS), dict(EXECUTION_PINS),
    dict(M1241_AUX_PINS), M1251_REL, M1251_MANIFEST_SHA256, M1251_OUTER_SHA256,
    DOCS359_REL, DOCS359_SHA256,
    (
        CandidateInput("legacy_ep29", OLD_RUN_REL, "old", OLD_CONFIG_REL, 29),
        CandidateInput("resume_ep30", NEW_RUN_REL, "new", NEW_CONFIG_REL, 30),
        CandidateInput("resume_ep32", NEW_RUN_REL, "new", NEW_CONFIG_REL, 32),
        CandidateInput("resume_ep34", NEW_RUN_REL, "new", NEW_CONFIG_REL, 34),
    ),
    NEW_MANIFEST_REL, OUTPUT_REL, ATTEMPT_REL, LOG_REL,
)


def artifact_map(policy: Policy) -> dict[str, Path]:
    values = {"manifest": policy.repo / policy.manifest_rel}
    configs = {}
    for row in policy.candidates:
        configs[row.config_key] = policy.repo / row.config_rel
        run = policy.repo / row.run_rel
        values[row.candidate_id + ":checkpoint"] = run / (
            "checkpoint_epoch{}.pth".format(row.epoch))
        values[row.candidate_id + ":profile"] = run / "standard_valid825" / (
            "epoch{}".format(row.epoch)) / "spike_profile.json"
    for key, path in configs.items():
        values["config:" + key] = path
    return values


def make_sealed_memfd(name: str, payload: bytes) -> int:
    require(hasattr(os, "memfd_create") and hasattr(fcntl, "F_ADD_SEALS"),
            "sealed memfd support required")
    flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
    descriptor = os.memfd_create(name, flags)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        seals = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                 fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        require(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
                "memfd seal population mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@dataclass
class Prepared:
    policy: Policy
    snapshots: dict[str, FileSnapshot]
    source_fds: tuple[int, int, int]
    command: list[str]

    def close(self) -> None:
        for descriptor in self.source_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fresh(path: Path, label: str) -> None:
    require(not path.exists() and not path.is_symlink(),
            "fresh {} namespace required: {}".format(label, path))


def prepare(policy: Policy, executable_path: Path, version: str, cwd: Path) -> Prepared:
    require(executable_path == policy.interpreter, "interpreter path mismatch")
    require(version == policy.python_version, "interpreter version mismatch")
    require(cwd == policy.repo, "repository cwd mismatch")
    directory(policy.repo, "repository")
    executable(policy.interpreter, "production interpreter")
    for collection, label in ((policy.authority_pins, "M1248 authority"),
                              (policy.aux_pins, "M1241 auxiliary")):
        for relative, expected in collection.items():
            observed, _ = snapshot_file(policy.repo / relative, label + " " + str(relative))
            require(observed.sha256 == expected, label + " SHA drift: " + str(relative))
    verify_double_seal(policy.repo / policy.review_rel,
                       policy.review_manifest_sha256, policy.review_outer_sha256)
    docs, _ = snapshot_file(policy.repo / policy.docs_rel, "protected docs/359")
    require(docs.sha256 == policy.docs_sha256, "protected docs/359 SHA drift")
    pairs = tuple((row.candidate_id, row.epoch) for row in policy.candidates)
    require(pairs == tuple(EXACT_PAIRS.items()), "exact candidate pair/order mismatch")
    require(len({policy.repo / row.run_rel for row in policy.candidates}) == 2,
            "exactly two run roots required")
    require(len({policy.repo / row.config_rel for row in policy.candidates}) == 2,
            "exactly two configs required")
    for run in {policy.repo / row.run_rel for row in policy.candidates}:
        directory(run, "candidate run")
    snapshots = {}
    for key, path in artifact_map(policy).items():
        snapshots[key], _ = snapshot_file(path, "candidate artifact " + key)
    require(len(snapshots) == 11, "exact eleven candidate artifacts required")

    fds = []
    try:
        for relative in (M1241_SOURCE_REL, M1234_SOURCE_REL, M1228_SOURCE_REL):
            require(relative in policy.execution_pins, "missing execution pin: " + str(relative))
            observed, payload = snapshot_file(policy.repo / relative, "execution source " + str(relative))
            require(observed.sha256 == policy.execution_pins[relative],
                    "execution source SHA drift: " + str(relative))
            fds.append(make_sealed_memfd(relative.stem, payload))
        output = policy.repo / policy.output_rel
        attempt = policy.repo / policy.attempt_rel
        log = policy.repo / policy.log_rel
        require(len({str(output), str(attempt), str(log)}) == 3,
                "output/attempt/log namespaces must be distinct")
        require(output.parent == attempt.parent == log.parent,
                "output/attempt/log must share results parent")
        directory(output.parent, "results parent")
        _fresh(output, "output")
        _fresh(attempt, "attempt")
        _fresh(log, "log")
        command = [str(policy.interpreter), "-I", "-B", "-c", SEALED_LAUNCHER,
                   *(str(descriptor) for descriptor in fds), str(output)]
        return Prepared(policy, snapshots, tuple(fds), command)
    except Exception:
        for descriptor in fds:
            os.close(descriptor)
        raise


def consume_attempt(prepared: Prepared) -> None:
    attempt = prepared.policy.repo / prepared.policy.attempt_rel
    descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        snapshot_digest = hashlib.sha256()
        for key in sorted(prepared.snapshots):
            row = prepared.snapshots[key]
            snapshot_digest.update((key + "\0" + row.sha256 + "\0" +
                                    str(row.device) + "\0" + str(row.inode) + "\0").encode())
        body = (
            "M1253_PRODUCTION_BINDER_ATTEMPT_CONSUMED_BEFORE_SEALED_CHILD\n"
            "automatic_retry=false\ninput_snapshot_sha256={}\ncommand_sha256={}\n".format(
                snapshot_digest.hexdigest(),
                sha256_bytes("\0".join(prepared.command).encode())))
        os.write(descriptor, body.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_log(prepared: Prepared, completed: subprocess.CompletedProcess[str]) -> None:
    log = prepared.policy.repo / prepared.policy.log_rel
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        body = (
            "M1253_SEALED_CHILD_LOG\nreturncode={}\nstdout_sha256={}\n"
            "stderr_sha256={}\n".format(
                completed.returncode, sha256_bytes(completed.stdout.encode()),
                sha256_bytes(completed.stderr.encode())))
        os.write(descriptor, body.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exact_identity(observed: Any, expected: FileSnapshot, label: str) -> None:
    require(isinstance(observed, dict), label + " identity must be object")
    for key, value in expected.receipt_identity().items():
        require(observed.get(key) == value, "{} {} mismatch".format(label, key))


def _metric(value: Any, label: str) -> Decimal:
    require(type(value) in (str, int, float), label + " invalid metric type")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReleaseError(label + " invalid decimal") from exc
    require(result.is_finite() and result >= 0, label + " must be finite nonnegative")
    return result


def _verify_rows(result: dict[str, Any], prepared: Prepared) -> list[dict[str, Any]]:
    rows = result.get("candidate_population")
    require(isinstance(rows, list) and len(rows) == 4, "candidate population must be four")
    expected_row_keys = {
        "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
        "profile", "accuracy_metrics", "activity",
    }
    for row, candidate in zip(rows, prepared.policy.candidates):
        require(isinstance(row, dict) and set(row) == expected_row_keys,
                "candidate row key population mismatch")
        require((row.get("candidate_id"), row.get("epoch")) ==
                (candidate.candidate_id, candidate.epoch), "candidate pair mismatch")
        require(row.get("run_directory") == str(prepared.policy.repo / candidate.run_rel),
                "candidate run path mismatch")
        _exact_identity(row.get("checkpoint"),
                        prepared.snapshots[candidate.candidate_id + ":checkpoint"],
                        candidate.candidate_id + " checkpoint")
        _exact_identity(row.get("configuration"),
                        prepared.snapshots["config:" + candidate.config_key],
                        candidate.candidate_id + " configuration")
        _exact_identity(row.get("profile"),
                        prepared.snapshots[candidate.candidate_id + ":profile"],
                        candidate.candidate_id + " profile")
        profile = row["profile"]
        require(profile.get("samples") == 825 and type(profile.get("samples")) is int,
                "profile sample count mismatch")
        metrics = row.get("accuracy_metrics")
        require(isinstance(metrics, dict) and set(metrics) == set(ERROR_METRIC_KEYS),
                "accuracy metric key population mismatch")
        for key in ERROR_METRIC_KEYS:
            _metric(metrics[key], candidate.candidate_id + " " + key)
        require(isinstance(row.get("activity"), dict), "activity must be object")
    return rows


def verify_receipt(output: Path, prepared: Prepared) -> dict[str, Any]:
    directory(output, "M1241 selection receipt")
    observed = set()
    payloads = {}
    for member in output.iterdir():
        snapshot, payload = snapshot_file(member, "receipt member " + member.name)
        require(stat.S_ISREG(snapshot.mode), "receipt member must be regular")
        observed.add(member.name)
        payloads[member.name] = payload
    require(observed == RESULT_PAYLOADS | {MANIFEST, OUTER},
            "receipt member population mismatch")
    require(payloads[OUTER].decode("utf-8").split() ==
            [sha256_bytes(payloads[MANIFEST]), MANIFEST],
            "receipt outer seal mismatch")
    rows_by_name = {}
    for line in payloads[MANIFEST].decode("utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "invalid receipt manifest row")
        name = fields[1].lstrip("*")
        require(name in RESULT_PAYLOADS and Path(name).name == name and
                name not in rows_by_name, "invalid receipt member")
        require(sha256_bytes(payloads[name]) == fields[0],
                "receipt payload SHA drift: " + name)
        rows_by_name[name] = fields[0]
    require(set(rows_by_name) == RESULT_PAYLOADS, "receipt manifest population mismatch")
    require(payloads["RUN_COMPLETE.txt"].decode("utf-8") == RUN_COMPLETE,
            "receipt terminal mismatch")
    result = strict_json_payload(payloads["final_checkpoint_selection.json"],
                                 "selection result")
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == RESULT_STATUS,
            "selection schema/status mismatch")
    require(result.get("claim_boundary") == dict(EXACT_CLAIM_BOUNDARY),
            "selection exact claim boundary mismatch")
    _exact_identity(result.get("new_run_manifest"), prepared.snapshots["manifest"],
                    "new run manifest")
    candidate_rows = _verify_rows(result, prepared)
    expected_rule = {
        "candidate_ids": list(EXACT_PAIRS), "epochs": list(EXACT_PAIRS.values()),
        "primary": "minimum finite nonnegative standard-valid825 AEE",
        "tie_break": "lowest epoch", "all_four_candidates_required": True,
        "cross_run": True, "cross_config": True,
        "profile_hash_and_parse_same_immutable_bytes": True,
    }
    require(result.get("selection_rule") == expected_rule, "selection rule mismatch")
    winner = min(candidate_rows, key=lambda row: (
        _metric(row["accuracy_metrics"]["AEE"], "AEE"), row["epoch"]))
    expected_selected = {key: winner[key] for key in (
        "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
        "profile", "accuracy_metrics", "activity")}
    require(result.get("selected") == expected_selected,
            "selected row is not exact minimum-AEE candidate projection")
    require(EXACT_PAIRS.get(expected_selected["candidate_id"]) == expected_selected["epoch"],
            "selected candidate/epoch exact map mismatch")

    selected_file = strict_json_payload(payloads["selected_checkpoint_and_config.json"],
                                        "selected identity")
    require(selected_file == {
        "schema": "m1234_selected_checkpoint_and_config_r1_v1",
        **{key: expected_selected[key] for key in (
            "candidate_id", "epoch", "run_directory", "checkpoint",
            "configuration", "profile")}}, "selected identity payload mismatch")
    targets = json.loads(payloads["e0_e8_activation_rebind_targets.json"].decode("utf-8"))
    require(targets == result.get("e0_e8_activation_dependent_invalidation_and_rebind_targets"),
            "E0-E8 payload mismatch")

    csv_rows = list(csv.DictReader(io.StringIO(
        payloads["four_checkpoint_metrics.csv"].decode("utf-8"), newline="")))
    require(len(csv_rows) == 4, "metrics CSV row population mismatch")
    expected_header = ["candidate_id", "epoch", "config_sha256", "checkpoint_sha256",
                       "profile_sha256", "samples", *ERROR_METRIC_KEYS]
    require(list(csv_rows[0]) == expected_header, "metrics CSV header mismatch")
    for csv_row, row in zip(csv_rows, candidate_rows):
        require(csv_row["candidate_id"] == row["candidate_id"] and
                csv_row["epoch"] == str(row["epoch"]) and
                csv_row["config_sha256"] == row["configuration"]["sha256"] and
                csv_row["checkpoint_sha256"] == row["checkpoint"]["sha256"] and
                csv_row["profile_sha256"] == row["profile"]["sha256"] and
                csv_row["samples"] == "825", "metrics CSV identity mismatch")
        for key in ERROR_METRIC_KEYS:
            require(csv_row[key] == str(row["accuracy_metrics"][key]),
                    "metrics CSV value mismatch: " + key)
    return result


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path,
                   pass_fds: tuple[int, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, pass_fds=pass_fds,
                          env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})


def execute_once(policy: Policy, executable_path: Path, version: str, cwd: Path,
                 runner: Runner = default_runner) -> subprocess.CompletedProcess[str]:
    prepared = prepare(policy, executable_path, version, cwd)
    try:
        consume_attempt(prepared)
        completed = runner(prepared.command, policy.repo, prepared.source_fds)
        publish_log(prepared, completed)
        require(completed.returncode == 0,
                "single sealed M1241 child failed after attempt; no retry authorized")
        require(completed.stdout.count(CHILD_TOKEN) == 1,
                "single sealed M1241 child terminal stdout mismatch")
        verify_receipt(policy.repo / policy.output_rel, prepared)
        return completed
    finally:
        prepared.close()


def main() -> int:
    require(len(sys.argv) == 1, "production M1253 release accepts zero arguments")
    completed = execute_once(PRODUCTION_POLICY, Path(sys.executable),
                             platform.python_version(), Path.cwd())
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1253_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

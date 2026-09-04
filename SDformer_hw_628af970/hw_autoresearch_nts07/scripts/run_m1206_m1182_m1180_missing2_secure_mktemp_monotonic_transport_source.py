#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1206 inert secure-mktemp transport source for two M1180 dependencies.

This is the fail-closed successor to M1203/M1204.  It preserves the monotonic
ABSENT-or-EXACT repository state machine, but removes the fixed pre-SCP remote
pathname.  An authenticated remote process invokes ``/usr/bin/mktemp -d``;
the single returned anchored path and its owner/mode/type are verified before
SCP writes a fixed basename inside that exclusive directory.  A fresh,
different-author M1207 hammer is mandatory before any remote action.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source_contract_r1_20260830.json")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
M1197_REL = Path("hw_autoresearch_nts07/reviews/m1197_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_20260830/review.json")
M1201_REL = Path("hw_autoresearch_nts07/reviews/m1201_m1200_m1182_m1180_missing2_atomic_transport_repair_hammer_r1_20260830/review.json")
M1204_STOP_REL = Path("hw_autoresearch_nts07/reviews/m1204_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_hammer_r1_20260830/review.json")
M1184_REL = Path("hw_autoresearch_nts07/reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830/review.json")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1207_m1206_m1182_m1180_missing2_secure_mktemp_transport_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
REMOTE_MKTEMP = Path("/usr/bin/mktemp")
REMOTE_TEMP_TEMPLATE = "/tmp/m1206_m1180.XXXXXXXXXXXX"
REMOTE_TEMP_RE = re.compile(r"\A/tmp/m1206_m1180\.[A-Za-z0-9]{12}\Z")
REMOTE_ARCHIVE_BASENAME = "exact2.tar"
REMOTE_EXPECTED_UID = 0
LOCAL_ATTEMPT = HW / "results/.m1206_m1180_missing2_secure_mktemp_transport_r1_attempt_consumed"
LOCAL_RESULT = HW / "results/m1206_m1180_missing2_secure_mktemp_transport_r1_20260830"
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
PASS_TOKEN = "PASS_M1206_SECURE_MKTEMP_MONOTONIC_EXACT2__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


class TransportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise TransportError("missing {}: {}".format(label, path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be non-symlink regular file")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           TransportError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and
            str(path) == path.as_posix(), "unsafe repository-relative path")
    return path


EXPECTED = [
    {"label": "decoder_selection_authority",
     "path": "hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
     "size_bytes": 15961,
     "sha256": "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7",
     "inventory_disposition": "remote_existing_hash_verify",
     "remote_audit_state": "MISSING"},
    {"label": "dependency_event_inventory_authority",
     "path": "hw_autoresearch_nts07/results/h67_ep35_dependency_dag_s1_20260822/dependency_events.jsonl",
     "size_bytes": 34816039,
     "sha256": "e1d2007195a036eedcee1e49d960955b3508ffe590ba3d075a3877a501a62f6b",
     "inventory_disposition": "remote_existing_hash_verify",
     "remote_audit_state": "MISSING"},
]


def validate_expected(rows: Any) -> list[dict[str, Any]]:
    require(type(rows) is list and rows == EXPECTED, "exact-two identity/order drift")
    return rows


def exact_members(contract: dict[str, Any]) -> list[dict[str, Any]]:
    validate_expected(contract["missing2"])
    inventory = strict_json(ROOT / INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
            "inventory semantics drift")
    dependencies = inventory.get("dependencies", [])
    require(type(dependencies) is list and
            len({row.get("path") for row in dependencies}) == len(dependencies),
            "inventory duplicate path")
    by_path = {row.get("path"): row for row in dependencies}
    result = []
    for expected in EXPECTED:
        row = by_path.get(expected["path"])
        require(row is not None and row.get("label") == expected["label"] and
                row.get("disposition") == expected["inventory_disposition"] and
                row.get("size_bytes") == expected["size_bytes"] and
                row.get("sha256") == expected["sha256"], "inventory target drift")
        source = ROOT / repo_relative(expected["path"])
        regular(source, "exact-two source")
        require(source.stat().st_size == expected["size_bytes"] and
                sha256(source) == expected["sha256"], "local dependency drift")
        result.append({key: expected[key] for key in ("path", "size_bytes", "sha256")})
    return result


def exact_state(path: Path, row: dict[str, Any]) -> str:
    """Return ABSENT or EXACT; reject every other state without mutation."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "ABSENT"
    require(stat.S_ISREG(mode) and not path.is_symlink(), "target symlink/nonregular")
    require(path.stat().st_size == row["size_bytes"] and
            sha256(path) == row["sha256"], "target wrong size/SHA")
    return "EXACT"


def validate_temp_path_text(stdout: bytes) -> Path:
    """Accept exactly one newline-terminated anchored mktemp pathname."""
    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise TransportError("mktemp stdout must be ASCII") from error
    require(text.endswith("\n") and text.count("\n") == 1,
            "mktemp stdout must contain exactly one line")
    candidate = text[:-1]
    require(REMOTE_TEMP_RE.fullmatch(candidate) is not None,
            "mktemp stdout path is not anchored to the exclusive template")
    return Path(candidate)


def validate_temp_directory(path: Path, expected_uid: int = REMOTE_EXPECTED_UID) -> None:
    require(REMOTE_TEMP_RE.fullmatch(path.as_posix()) is not None,
            "remote temporary path anchor drift")
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise TransportError("remote temporary directory missing") from error
    require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(),
            "remote temporary path must be a non-symlink directory")
    require(info.st_uid == expected_uid, "remote temporary directory owner mismatch")
    require(stat.S_IMODE(info.st_mode) == 0o700,
            "remote temporary directory mode must be exactly 0700")


def validate_archive_path(path: Path, expected_size: int, expected_sha: str) -> None:
    require(path.name == REMOTE_ARCHIVE_BASENAME, "archive basename drift")
    regular(path, "remote archive")
    require(path.stat().st_size == expected_size, "remote archive size mismatch")
    require(sha256(path) == expected_sha, "remote archive SHA mismatch")


def validate_archive_to_stage(archive_path: Path, stage: Path,
                              rows: list[dict[str, Any]]) -> list[Path]:
    """Local test oracle mirroring the remote pre-extraction archive gate."""
    require(not stage.exists() and not stage.is_symlink(), "stage preexists")
    stage.mkdir(mode=0o700)
    staged: list[Path] = []
    try:
        regular(archive_path, "archive")
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            require([member.name for member in members] == [row["path"] for row in rows],
                    "archive extra/path/order attack")
            for member, row in zip(members, rows):
                require(member.isfile() and not member.issym() and not member.islnk() and
                        member.size == row["size_bytes"],
                        "archive link/type/size attack")
                output = stage / repo_relative(member.name)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                require(source is not None, "archive member unreadable")
                fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o444)
                digest = hashlib.sha256(); count = 0
                with os.fdopen(fd, "wb") as destination:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        count += len(block); digest.update(block); destination.write(block)
                    destination.flush(); os.fsync(destination.fileno())
                require(count == row["size_bytes"] and
                        digest.hexdigest() == row["sha256"],
                        "archive member SHA mismatch")
                staged.append(output)
        return staged
    except BaseException:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


def reconcile_exact_files(staged: list[Path], destinations: list[Path],
                          rows: list[dict[str, Any]], token: str,
                          publish: Callable[[Path, Path], None] = os.replace,
                          after_publish: Callable[[int], None] | None = None,
                          control_absent: Callable[[], None] | None = None,
                          cleanup: Callable[[Path], None] | None = None) -> list[str]:
    """Monotonic two-file publish with destination-local exclusive temporaries."""
    require(len(staged) == len(destinations) == len(rows) == 2, "exact-two required")
    require(re.fullmatch(r"[A-Za-z0-9]{12}", token) is not None, "publish token drift")
    if control_absent is not None:
        control_absent()
    states = [exact_state(path, row) for path, row in zip(destinations, rows)]
    temporary_paths: list[Path] = []
    primary: BaseException | None = None
    try:
        for index, (source, destination, row) in enumerate(zip(staged, destinations, rows)):
            if states[index] == "EXACT":
                continue
            regular(source, "staged member")
            require(source.stat().st_size == row["size_bytes"] and
                    sha256(source) == row["sha256"], "staged member drift")
            require(destination.parent.is_dir() and not destination.parent.is_symlink(),
                    "unsafe destination parent")
            temporary = destination.parent / ("." + destination.name +
                                                ".m1206." + token + ".publish.tmp")
            require(exact_state(temporary, row) == "ABSENT", "publish temporary preexists")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, "O_NOFOLLOW", 0), 0o444)
            temporary_paths.append(temporary)
            with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    dst.write(block)
                dst.flush(); os.fsync(dst.fileno())
            require(exact_state(temporary, row) == "EXACT", "publish temporary drift")
            if exact_state(destination, row) == "ABSENT":
                publish(temporary, destination)
            else:
                temporary.unlink()
            if after_publish is not None:
                after_publish(index)
            if control_absent is not None:
                control_absent()
            require(exact_state(destination, row) == "EXACT", "published state not exact")
            states[index] = "EXACT"
        if control_absent is not None:
            control_absent()
        require([exact_state(path, row) for path, row in zip(destinations, rows)] ==
                ["EXACT", "EXACT"], "final both-exact gate")
    except BaseException as error:
        primary = error
    cleanup_errors = []
    for temporary in temporary_paths:
        if temporary.exists() or temporary.is_symlink():
            try:
                (cleanup or Path.unlink)(temporary)
            except BaseException as error:
                cleanup_errors.append(repr(error))
    safe_states = [exact_state(path, row) for path, row in zip(destinations, rows)]
    require(all(state in ("ABSENT", "EXACT") for state in safe_states),
            "unsafe final reconciliation state")
    if control_absent is not None:
        control_absent()
    if primary is not None:
        raise primary
    require(not cleanup_errors,
            "cleanup failed; verified temporary artifacts may remain: " + repr(cleanup_errors))
    return safe_states


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    require(set(contract) == {"schema", "status", "date", "source", "test",
                              "inventory_authority", "stop_authorities",
                              "original_release_authority", "missing2", "remote_audit",
                              "transport", "reconciliation", "future_hammer",
                              "claim_boundary", "docs359_sha256"}, "contract keys drift")
    require(contract["schema"] ==
            "m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source_contract_r1_v1" and
            contract["status"] ==
            "INERT_SOURCE_ONLY__SECURE_MKTEMP_MONOTONIC_EXACT2__FRESH_M1207_HAMMER_REQUIRED",
            "contract schema/status drift")
    for label, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        path = ROOT / rel; regular(path, label)
        require(contract[label] == {"path": str(rel), "size_bytes": path.stat().st_size,
                                    "sha256": sha256(path)}, label + " identity drift")
    authorities = [
        (INVENTORY_REL, 42133, "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae"),
        (M1197_REL, 5442, "5b2355274564721c4df91067e74f7b5ba15635ff8b101a0fc2ffadcb961d1888"),
        (M1201_REL, 12599, "f2df5f0f3882118b6a91e493ebccd75d4e325d18bc74adc480d69bd709538d38"),
        (M1204_STOP_REL, 2943, "fa5ef5d25342993c152a5988557807c8eb8e0d4700f3234ef1e00702fa724748"),
        (M1184_REL, 4193, "adfda233ad020759cd0106f9e44d8ea7ec9ab97aa2882e1365ee18cf6ac88aa7"),
    ]
    for rel, size, digest in authorities:
        path = ROOT / rel; regular(path, "pinned authority")
        require(path.stat().st_size == size and sha256(path) == digest,
                "pinned authority drift: " + str(rel))
    require(contract["inventory_authority"] ==
            {"path": str(INVENTORY_REL), "size_bytes": 42133, "sha256": authorities[0][2]},
            "inventory authority contract drift")
    require(contract["stop_authorities"] == [
        {"path": str(M1197_REL), "size_bytes": 5442, "sha256": authorities[1][2],
         "status": "STOP_M1197_M1195_IDENTITY_AND_AUTHOR_SEAL_DRIFT"},
        {"path": str(M1201_REL), "size_bytes": 12599, "sha256": authorities[2][2],
         "status": "STOP_M1200_POST_PUBLICATION_CLEANUP_AND_LINK_WINDOW_NOT_ROLLBACK_CLEAN"},
        {"path": str(M1204_STOP_REL), "size_bytes": 2943, "sha256": authorities[3][2],
         "status": "STOP_M1203_PRE_SCP_ARCHIVE_PATH_NOT_PREFLIGHTED"}],
        "STOP authorities drift")
    require(contract["original_release_authority"] ==
            {"path": str(M1184_REL), "size_bytes": 4193, "sha256": authorities[4][2],
             "status": "PASS", "automatic_retry": False}, "M1184 authority drift")
    validate_expected(contract["missing2"])
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def verify_policy(contract: dict[str, Any]) -> None:
    require(contract["remote_audit"] == {"audited_remote_existing_rows": 55,
            "missing_rows": 2, "mismatched_rows": 0, "present_exact_rows": 53,
            "m1180_attempt_absent": True, "m1180_result_absent": True,
            "audit_read_only": True}, "remote audit drift")
    require(contract["transport"] == {
        "member_count": 2,
        "protocol": "SSH_REMOTE_MKTEMP_D_PLUS_VALIDATED_EXCLUSIVE_DIR_PLUS_SCP_FIXED_BASENAME_PLUS_REMOTE_PYTHON_MONOTONIC_RECONCILIATION",
        "local_ssh": str(SSH), "local_scp": str(SCP), "remote_host": REMOTE_HOST,
        "remote_port": int(REMOTE_PORT), "ssh_control_path": SSH_CONTROL_PATH,
        "remote_repository": str(REMOTE_REPO), "remote_interpreter": REMOTE_INTERPRETER,
        "remote_python_version": "3.10.20", "remote_mktemp": str(REMOTE_MKTEMP),
        "remote_temp_template": REMOTE_TEMP_TEMPLATE,
        "remote_temp_regex": REMOTE_TEMP_RE.pattern,
        "remote_temp_owner_uid": REMOTE_EXPECTED_UID, "remote_temp_mode": "0700",
        "remote_archive_basename": REMOTE_ARCHIVE_BASENAME,
        "shell": False, "fixed_argv": True, "automatic_retry": False,
        "gpu_launch": False, "capture_launch": False}, "transport policy drift")
    require(contract["reconciliation"] == {
        "allowed_target_states": ["ABSENT", "EXACT_REGULAR_SIZE_SHA"],
        "preexisting_exact_subset_accepted": True,
        "wrong_symlink_nonregular_rejected_without_mutation": True,
        "archive_regular_nonsymlink_size_sha_before_extract": True,
        "missing_installed_via_destination_local_exclusive_temp_publish": True,
        "published_exact_never_rolled_back": True,
        "partial_exact_state_safe_and_recoverable": True,
        "success_requires_both_exact": True,
        "cleanup_failure_may_leave_verified_temp_but_never_success": True,
        "handled_failure_requires_attempt_result_absent": True}, "reconciliation drift")
    require(contract["claim_boundary"] == {
        "source_only": True, "remote_transfer_executed_by_author": False,
        "gpu_or_capture_executed_by_author": False, "eda_executed_by_author": False,
        "m1180_attempt_consumed": False, "paper_result": False,
        "m1197_m1201_m1204_stops_or_m1184_modified": False,
        "docs359_modified": False}, "claim boundary drift")


def verify_future_hammer(contract: dict[str, Any]) -> None:
    future = contract["future_hammer"]
    require(future == {"directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1207_m1206_m1182_m1180_missing2_secure_mktemp_transport_hammer_r1_v1",
        "required_status": "PASS_M1206_SECURE_MKTEMP_MONOTONIC_EXACT2_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "review_env": "M1206_EXPECTED_HAMMER_REVIEW_SHA256",
        "manifest_env": "M1206_EXPECTED_HAMMER_MANIFEST_SHA256",
        "outer_env": "M1206_EXPECTED_HAMMER_OUTER_SHA256"}, "future hammer drift")
    paths = [ROOT / FUTURE_HAMMER_REL / name for name in
             ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")]
    for path in paths:
        regular(path, "fresh M1207 hammer")
    expected = [os.environ.get(future[key], "") for key in
                ("review_env", "manifest_env", "outer_env")]
    require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in expected),
            "fresh M1207 digest env absent")
    require([sha256(path) for path in paths] == expected, "fresh M1207 digest mismatch")
    require(paths[2].read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh M1207 recursive seal mismatch")
    review = strict_json(paths[0])
    require(review.get("schema") == future["required_schema"] and
            review.get("status") == future["required_status"],
            "fresh M1207 semantic admission mismatch")


def fixed_ssh_python_argv() -> list[str]:
    return [str(SSH), "-p", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", REMOTE_HOST, REMOTE_INTERPRETER, "-I", "-"]


def fixed_scp_argv(local_archive: Path, remote_temp: Path) -> list[str]:
    require(REMOTE_TEMP_RE.fullmatch(remote_temp.as_posix()) is not None,
            "unsafe SCP remote temporary path")
    destination = remote_temp / REMOTE_ARCHIVE_BASENAME
    return [str(SCP), "-P", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", str(local_archive),
            REMOTE_HOST + ":" + destination.as_posix()]


def build_archive(path: Path, members: list[dict[str, Any]]) -> tuple[int, str]:
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for row in members:
            source = ROOT / repo_relative(row["path"]); regular(source, "archive source")
            info = tarfile.TarInfo(row["path"]); info.size = row["size_bytes"]
            info.mode = 0o444; info.uid = info.gid = 0
            info.uname = info.gname = ""; info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    regular(path, "archive")
    return path.stat().st_size, sha256(path)


def mktemp_program() -> bytes:
    code = """import subprocess,sys
r=subprocess.run([MKTEMP,'-d',TEMPLATE],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=False,check=False)
if r.returncode!=0 or r.stderr: raise RuntimeError('remote mktemp failed')
sys.stdout.buffer.write(r.stdout)
"""
    return ("MKTEMP=" + repr(str(REMOTE_MKTEMP)) + "\nTEMPLATE=" +
            repr(REMOTE_TEMP_TEMPLATE) + "\n" + code).encode()


REMOTE_COMMON = r'''import hashlib,json,os,pathlib,re,stat,sys
def die(message): raise RuntimeError('M1206_REMOTE_FAIL: '+message)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def safe(text):
 p=pathlib.PurePosixPath(text)
 if p.is_absolute() or not p.parts or '..' in p.parts or str(p)!=text: die('unsafe path')
 return pathlib.Path(*p.parts)
def tempdir(path):
 if re.fullmatch(r'/tmp/m1206_m1180\.[A-Za-z0-9]{12}',str(path)) is None: die('temp anchor')
 info=path.lstat()
 if not stat.S_ISDIR(info.st_mode) or path.is_symlink(): die('temp symlink/non-directory')
 if info.st_uid!=EXPECTED_UID: die('temp owner')
 if stat.S_IMODE(info.st_mode)!=0o700: die('temp mode')
def absent_control(root,rel):
 p=root/safe(rel)
 if p.exists() or p.is_symlink(): die('M1180 attempt/result must remain absent')
def state(path,row):
 try: mode=path.lstat().st_mode
 except FileNotFoundError: return 'ABSENT'
 if not stat.S_ISREG(mode) or path.is_symlink(): die('target symlink/nonregular')
 if path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']: die('target wrong size/SHA')
 return 'EXACT'
'''


def remote_prefix(remote_temp: Path, members: list[dict[str, Any]],
                  archive_size: int = 0, archive_sha: str = "") -> str:
    require(REMOTE_TEMP_RE.fullmatch(remote_temp.as_posix()) is not None,
            "remote temp prefix anchor")
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    return "\n".join(("ROOT=" + repr(str(REMOTE_REPO)),
        "TEMPDIR=" + repr(remote_temp.as_posix()),
        "ARCHIVE_BASENAME=" + repr(REMOTE_ARCHIVE_BASENAME),
        "ARCHIVE_SIZE=" + repr(archive_size), "ARCHIVE_SHA=" + repr(archive_sha),
        "EXPECTED_UID=" + repr(REMOTE_EXPECTED_UID),
        "INTERPRETER=" + repr(REMOTE_INTERPRETER), "PYTHON_VERSION='3.10.20'",
        "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
        "M1180_RESULT=" + repr(str(M1180_RESULT_REL)), "ROWS=" + repr(rows), ""))


def temp_preflight_program(remote_temp: Path, members: list[dict[str, Any]]) -> bytes:
    code = r'''root=pathlib.Path(ROOT); td=pathlib.Path(TEMPDIR)
if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die('interpreter identity')
if not root.is_dir() or root.is_symlink(): die('unsafe root')
tempdir(td)
if list(td.iterdir()): die('exclusive temp directory not empty before SCP')
absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
states=[]
for row in json.loads(ROWS):
 dest=root/safe(row['path'])
 if not dest.parent.is_dir() or dest.parent.is_symlink(): die('unsafe parent')
 states.append(state(dest,row))
print(json.dumps({'status':'PASS_M1206_SECURE_TEMP_PREFLIGHT','temp':str(td),'states':states,'attempt_result_absent':True},sort_keys=True))
'''
    return (remote_prefix(remote_temp, members) + REMOTE_COMMON + code).encode()


REMOTE_RECONCILER = r'''import shutil,tarfile
def main():
 rows=json.loads(ROWS); root=pathlib.Path(ROOT); td=pathlib.Path(TEMPDIR)
 archive=td/ARCHIVE_BASENAME; stage=td/'stage'; destinations=[]; initial=[]; temps=[]
 primary=None; cleanup_errors=[]
 try:
  if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die('interpreter identity')
  if len(rows)!=2 or len({r['path'] for r in rows})!=2: die('exact-two drift')
  tempdir(td); absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
  for row in rows:
   dest=root/safe(row['path'])
   if not dest.parent.is_dir() or dest.parent.is_symlink(): die('unsafe destination parent')
   initial.append(state(dest,row)); destinations.append(dest)
  info=archive.lstat()
  if not stat.S_ISREG(info.st_mode) or archive.is_symlink(): die('archive symlink/nonregular')
  if info.st_uid!=EXPECTED_UID or info.st_size!=ARCHIVE_SIZE or digest(archive)!=ARCHIVE_SHA: die('archive owner/size/SHA')
  if stage.exists() or stage.is_symlink(): die('stage preexists')
  stage.mkdir(mode=0o700); staged=[]
  with tarfile.open(archive,'r:') as tf:
   items=tf.getmembers()
   if [m.name for m in items]!=[r['path'] for r in rows]: die('archive extra/path/order attack')
   for member,row in zip(items,rows):
    if not member.isfile() or member.issym() or member.islnk() or member.size!=row['size_bytes']: die('archive link/type/size attack')
    out=stage/safe(member.name); out.parent.mkdir(parents=True,exist_ok=True); src=tf.extractfile(member)
    if src is None: die('archive member unreadable')
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444); h=hashlib.sha256(); count=0
    with os.fdopen(fd,'wb') as dst:
     for block in iter(lambda:src.read(1<<20),b''): count+=len(block); h.update(block); dst.write(block)
     dst.flush(); os.fsync(dst.fileno())
    if count!=row['size_bytes'] or h.hexdigest()!=row['sha256']: die('archive member SHA mismatch')
    staged.append(out)
  token=td.name.split('.',1)[1]
  for src,dest,row in zip(staged,destinations,rows):
   if state(dest,row)=='EXACT': continue
   temporary=dest.parent/('.'+dest.name+'.m1206.'+token+'.publish.tmp')
   if temporary.exists() or temporary.is_symlink(): die('publish temporary preexists')
   fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444); temps.append(temporary)
   with src.open('rb') as inp,os.fdopen(fd,'wb') as out:
    for block in iter(lambda:inp.read(1<<20),b''): out.write(block)
    out.flush(); os.fsync(out.fileno())
   if state(temporary,row)!='EXACT': die('publish temporary drift')
   if state(dest,row)=='ABSENT': os.replace(temporary,dest)
   else: temporary.unlink()
   absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
   if state(dest,row)!='EXACT': die('published target not exact')
  absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
  final=[state(d,r) for d,r in zip(destinations,rows)]
  if final!=['EXACT','EXACT']: die('final both-exact gate')
 except BaseException as error: primary=error
 finally:
  for temporary in temps:
   if temporary.exists() or temporary.is_symlink():
    try: temporary.unlink()
    except BaseException as error: cleanup_errors.append(repr(error))
 final=[]
 if len(destinations)==2:
  final=[state(d,r) for d,r in zip(destinations,rows)]
 absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
 if primary is not None: raise primary
 if cleanup_errors: die('cleanup failed; verified temporary artifacts may remain: '+repr(cleanup_errors))
 if final!=['EXACT','EXACT']: die('final both-exact gate')
 print(json.dumps({'status':'PASS_M1206_REMOTE_MONOTONIC_EXACT2','initial':initial,'final':final,'attempt_result_absent':True},sort_keys=True))
main()
'''


def reconciler_program(remote_temp: Path, members: list[dict[str, Any]],
                       archive_size: int, archive_sha: str) -> bytes:
    return (remote_prefix(remote_temp, members, archive_size, archive_sha) +
            REMOTE_COMMON + REMOTE_RECONCILER).encode()


def cleanup_program(remote_temp: Path, members: list[dict[str, Any]]) -> bytes:
    code = r'''import shutil
root=pathlib.Path(ROOT); td=pathlib.Path(TEMPDIR)
if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die('interpreter identity')
tempdir(td); absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
final=[]
for row in json.loads(ROWS): final.append(state(root/safe(row['path']),row))
if any(item not in ('ABSENT','EXACT') for item in final): die('unsafe repository state before cleanup')
shutil.rmtree(td)
if td.exists() or td.is_symlink(): die('unique temp cleanup failed')
absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
print(json.dumps({'status':'PASS_M1206_UNIQUE_TEMP_CLEANUP','final':final,'attempt_result_absent':True},sort_keys=True))
'''
    return (remote_prefix(remote_temp, members) + REMOTE_COMMON + code).encode()


def parse_json_receipt(stdout: bytes, status: str) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
        require(text.endswith("\n") and text.count("\n") == 1,
                "remote receipt must be exactly one line")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransportError("remote receipt parse failed") from error
    require(type(value) is dict and value.get("status") == status,
            "remote receipt status drift")
    return value


def run_ssh(program: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(fixed_ssh_python_argv(), input=program, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=False, check=False)


def run() -> None:
    contract = load_contract(); verify_policy(contract); members = exact_members(contract)
    verify_future_hammer(contract)
    require(not LOCAL_ATTEMPT.exists() and not LOCAL_ATTEMPT.is_symlink() and
            not LOCAL_RESULT.exists() and not LOCAL_RESULT.is_symlink(),
            "M1206 namespace not fresh")

    created = run_ssh(mktemp_program())
    require(created.returncode == 0 and not created.stderr, "exclusive remote mktemp failed")
    remote_temp = validate_temp_path_text(created.stdout)
    validated = False; primary: BaseException | None = None; cleanup_error: str | None = None
    remote: dict[str, Any] | None = None
    try:
        preflight = run_ssh(temp_preflight_program(remote_temp, members))
        require(preflight.returncode == 0 and not preflight.stderr,
                "secure remote temp preflight failed")
        receipt = parse_json_receipt(preflight.stdout, "PASS_M1206_SECURE_TEMP_PREFLIGHT")
        require(receipt.get("temp") == remote_temp.as_posix() and
                receipt.get("attempt_result_absent") is True,
                "secure remote temp receipt drift")
        validated = True
        with tempfile.TemporaryDirectory(prefix="m1206_m1180_missing2_") as temporary:
            archive = Path(temporary) / REMOTE_ARCHIVE_BASENAME
            archive_size, archive_sha = build_archive(archive, members)
            LOCAL_ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write("M1206_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
            copied = subprocess.run(fixed_scp_argv(archive, remote_temp),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=False, check=False)
            require(copied.returncode == 0, "fixed-argv secure-directory SCP/SFTP failed")
            installed = run_ssh(reconciler_program(remote_temp, members,
                                                   archive_size, archive_sha))
            require(installed.returncode == 0,
                    "remote monotonic reconciliation failed: " +
                    installed.stderr.decode("utf-8", errors="replace")[-1200:])
            remote = parse_json_receipt(installed.stdout,
                                        "PASS_M1206_REMOTE_MONOTONIC_EXACT2")
            require(remote.get("final") == ["EXACT", "EXACT"] and
                    remote.get("attempt_result_absent") is True,
                    "remote reconciliation receipt drift")
    except BaseException as error:
        primary = error
    finally:
        if validated:
            cleaned = run_ssh(cleanup_program(remote_temp, members))
            if cleaned.returncode != 0:
                cleanup_error = "remote unique-directory cleanup failed: " + cleaned.stderr.decode(
                    "utf-8", errors="replace")[-1200:]
            else:
                try:
                    cleanup_receipt = parse_json_receipt(
                        cleaned.stdout, "PASS_M1206_UNIQUE_TEMP_CLEANUP")
                    require(cleanup_receipt.get("attempt_result_absent") is True,
                            "cleanup receipt attempt/result drift")
                except BaseException as error:
                    cleanup_error = str(error)
    if primary is not None:
        if cleanup_error:
            raise TransportError(str(primary) + "; " + cleanup_error)
        raise primary
    require(cleanup_error is None, cleanup_error or "cleanup failed")
    require(remote is not None, "remote receipt absent")
    receipt = {"schema": "m1206_m1180_missing2_secure_mktemp_transport_result_r1_v1",
        "status": PASS_TOKEN, "members": members,
        "initial_remote_states": remote["initial"], "final_remote_states": remote["final"],
        "remote_temp_cleaned": True, "m1180_attempt_consumed": False,
        "gpu_or_capture_launched": False, "paper_result": False,
        "docs359_sha256": DOCS359_SHA256}
    LOCAL_RESULT.mkdir()
    (LOCAL_RESULT / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(PASS_TOKEN)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print("M1206_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(1)

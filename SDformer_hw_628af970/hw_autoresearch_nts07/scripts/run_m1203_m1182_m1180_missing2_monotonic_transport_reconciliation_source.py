#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1203 inert monotonic exact-state reconciliation for two M1180 dependencies.

This additive successor implements the state machine required by M1201.  Each
repository target is either absent or an exact, non-symlink regular file.  An
exact pre-existing subset is accepted, a missing member is published through a
verified same-filesystem temporary, and success requires both members exact.
Partial exact publication is deliberately safe and recoverable; it is never
rolled back.  The source cannot execute until a fresh M1204 hammer is pinned.
It transfers dependencies only and never launches capture, GPU, or EDA work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source_contract_r1_20260830.json")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
M1197_REL = Path("hw_autoresearch_nts07/reviews/m1197_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_20260830/review.json")
M1201_REL = Path("hw_autoresearch_nts07/reviews/m1201_m1200_m1182_m1180_missing2_atomic_transport_repair_hammer_r1_20260830/review.json")
M1184_REL = Path("hw_autoresearch_nts07/reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830/review.json")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1204_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
REMOTE_ARCHIVE = Path("/tmp/m1203_m1180_missing2_monotonic_transport_r1.tar")
REMOTE_STAGE = REMOTE_REPO / ".m1203_m1180_missing2_monotonic_stage_r1"
LOCAL_ATTEMPT = HW / "results/.m1203_m1180_missing2_monotonic_transport_r1_attempt_consumed"
LOCAL_RESULT = HW / "results/m1203_m1180_missing2_monotonic_transport_r1_20260830"
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
PASS_TOKEN = "PASS_M1203_MONOTONIC_EXACT2_RECONCILIATION__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


class ReconcileError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconcileError(message)


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
        raise ReconcileError("missing {}: {}".format(label, path)) from error
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
                           ReconcileError("nonfinite JSON: " + token)))
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


def validate_archive_to_stage(archive_path: Path, stage: Path,
                              rows: list[dict[str, Any]]) -> list[Path]:
    """Strictly extract exact ordered regular members into a new local stage."""
    require(exact_state(stage, {"size_bytes": 0, "sha256": ""}) == "ABSENT",
            "stage preexists")
    stage.mkdir(mode=0o700)
    staged = []
    try:
        regular(archive_path, "archive")
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            require([member.name for member in members] == [row["path"] for row in rows],
                    "archive extra/path/order attack")
            for member, row in zip(members, rows):
                require(member.isfile() and not member.issym() and not member.islnk() and
                        member.size == row["size_bytes"], "archive link/type/size attack")
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
                require(count == row["size_bytes"] and digest.hexdigest() == row["sha256"],
                        "archive member SHA mismatch")
                staged.append(output)
        return staged
    except BaseException:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


def reconcile_exact_files(staged: list[Path], destinations: list[Path],
                          rows: list[dict[str, Any]],
                          publish: Callable[[Path, Path], None] = os.replace,
                          after_publish: Callable[[int], None] | None = None,
                          cleanup: Callable[[Path], None] | None = None) -> list[str]:
    """Monotonically reconcile: any published target is independently exact."""
    require(len(staged) == len(destinations) == len(rows) == 2, "exact-two required")
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
            temporary = destination.parent / ("." + destination.name + ".m1203.publish.tmp")
            require(exact_state(temporary, row) == "ABSENT", "publish temporary preexists")
            os.link(source, temporary)
            temporary_paths.append(temporary)
            require(exact_state(temporary, row) == "EXACT", "publish temporary drift")
            current = exact_state(destination, row)
            if current == "ABSENT":
                publish(temporary, destination)
            else:
                temporary.unlink()
            if after_publish is not None:
                after_publish(index)
            require(exact_state(destination, row) == "EXACT", "published state not exact")
            states[index] = "EXACT"
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
    # Never roll back an exact published target.  Reconcile safety is checked
    # independently after every handled failure; absent/exact subsets are valid.
    safe_states = [exact_state(path, row) for path, row in zip(destinations, rows)]
    require(all(state in ("ABSENT", "EXACT") for state in safe_states),
            "unsafe final reconciliation state")
    if primary is not None:
        raise primary
    require(not cleanup_errors, "cleanup failed after exact publication: " + repr(cleanup_errors))
    return safe_states


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    require(set(contract) == {"schema", "status", "date", "source", "test",
                              "inventory_authority", "stop_authorities",
                              "original_release_authority", "missing2", "remote_audit",
                              "transport", "reconciliation", "future_hammer",
                              "claim_boundary", "docs359_sha256"}, "contract keys drift")
    require(contract["schema"] ==
            "m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source_contract_r1_v1" and
            contract["status"] ==
            "INERT_SOURCE_ONLY__MONOTONIC_EXACT_STATE__FRESH_M1204_HAMMER_REQUIRED",
            "contract schema/status drift")
    for label, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        path = ROOT / rel; regular(path, label)
        require(contract[label] == {"path": str(rel), "size_bytes": path.stat().st_size,
                                    "sha256": sha256(path)}, label + " identity drift")
    authorities = [
        (INVENTORY_REL, 42133, "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae"),
        (M1197_REL, 5442, "5b2355274564721c4df91067e74f7b5ba15635ff8b101a0fc2ffadcb961d1888"),
        (M1201_REL, 12599, "f2df5f0f3882118b6a91e493ebccd75d4e325d18bc74adc480d69bd709538d38"),
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
         "status": "STOP_M1200_POST_PUBLICATION_CLEANUP_AND_LINK_WINDOW_NOT_ROLLBACK_CLEAN"}],
        "STOP authorities drift")
    require(contract["original_release_authority"] ==
            {"path": str(M1184_REL), "size_bytes": 4193, "sha256": authorities[3][2],
             "status": "PASS", "automatic_retry": False},
            "M1184 release authority drift")
    validate_expected(contract["missing2"])
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def verify_policy(contract: dict[str, Any]) -> None:
    require(contract["remote_audit"] == {"audited_remote_existing_rows": 55,
            "missing_rows": 2, "mismatched_rows": 0, "present_exact_rows": 53,
            "m1180_attempt_absent": True, "m1180_result_absent": True,
            "audit_read_only": True}, "remote audit drift")
    require(contract["reconciliation"] == {
        "allowed_target_states": ["ABSENT", "EXACT_REGULAR_SIZE_SHA"],
        "preexisting_exact_subset_accepted": True,
        "wrong_symlink_nonregular_rejected_without_mutation": True,
        "missing_installed_via_verified_archive_same_fs_temp_publish": True,
        "published_exact_never_rolled_back": True,
        "partial_exact_state_safe_and_recoverable": True,
        "success_requires_both_exact": True,
        "cleanup_failure_may_fail_command_but_exact_state_remains_valid": True,
        "handled_failure_requires_attempt_result_absent": True}, "reconciliation drift")
    require(contract["transport"]["automatic_retry"] is False and
            contract["transport"]["gpu_launch"] is False and
            contract["transport"]["capture_launch"] is False,
            "transport boundary drift")
    require(contract["claim_boundary"] == {
        "source_only": True, "remote_transfer_executed_by_author": False,
        "gpu_or_capture_executed_by_author": False, "eda_executed_by_author": False,
        "m1180_attempt_consumed": False, "paper_result": False,
        "m1195_m1200_stops_or_m1184_modified": False, "docs359_modified": False},
        "claim boundary drift")


def verify_future_hammer(contract: dict[str, Any]) -> None:
    future = contract["future_hammer"]
    require(future == {"directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1204_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_hammer_r1_v1",
        "required_status": "PASS_M1203_MONOTONIC_EXACT2_RECONCILIATION_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "review_env": "M1203_EXPECTED_HAMMER_REVIEW_SHA256",
        "manifest_env": "M1203_EXPECTED_HAMMER_MANIFEST_SHA256",
        "outer_env": "M1203_EXPECTED_HAMMER_OUTER_SHA256"}, "future hammer drift")
    paths = [ROOT / FUTURE_HAMMER_REL / name for name in
             ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")]
    for path in paths: regular(path, "fresh M1204 hammer")
    expected = [os.environ.get(future[key], "") for key in
                ("review_env", "manifest_env", "outer_env")]
    require(all(len(value) == 64 for value in expected), "fresh M1204 digest env absent")
    require([sha256(path) for path in paths] == expected, "fresh M1204 digest mismatch")
    require(paths[2].read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh M1204 recursive seal mismatch")
    review = strict_json(paths[0])
    require(review.get("schema") == future["required_schema"] and
            review.get("status") == future["required_status"],
            "fresh M1204 semantic admission mismatch")


def fixed_ssh_argv() -> list[str]:
    return [str(SSH), "-p", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", REMOTE_HOST, REMOTE_INTERPRETER, "-I", "-"]


def fixed_scp_argv(local_archive: Path) -> list[str]:
    return [str(SCP), "-P", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", str(local_archive),
            REMOTE_HOST + ":" + str(REMOTE_ARCHIVE)]


def build_archive(path: Path, members: list[dict[str, Any]]) -> str:
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for row in members:
            source = ROOT / repo_relative(row["path"]); regular(source, "archive source")
            info = tarfile.TarInfo(row["path"]); info.size = row["size_bytes"]
            info.mode = 0o444; info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            with source.open("rb") as stream: archive.addfile(info, stream)
    regular(path, "archive")
    return sha256(path)


def preflight_program(members: list[dict[str, Any]]) -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    code = r'''import hashlib,json,pathlib,stat,sys
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def safe(text):
 p=pathlib.PurePosixPath(text)
 if p.is_absolute() or not p.parts or '..' in p.parts or str(p)!=text: raise RuntimeError('unsafe path')
 return pathlib.Path(*p.parts)
def state(path,row):
 try: mode=path.lstat().st_mode
 except FileNotFoundError: return 'ABSENT'
 if not stat.S_ISREG(mode) or path.is_symlink(): raise RuntimeError('target symlink/nonregular')
 if path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']: raise RuntimeError('target wrong size/SHA')
 return 'EXACT'
root=pathlib.Path(ROOT)
if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: raise RuntimeError('interpreter identity')
if not root.is_dir() or root.is_symlink(): raise RuntimeError('unsafe root')
for rel in (M1180_ATTEMPT,M1180_RESULT):
 p=root/safe(rel)
 if p.exists() or p.is_symlink(): raise RuntimeError('M1180 attempt/result must remain absent')
states=[]
for row in json.loads(ROWS):
 dest=root/safe(row['path'])
 if not dest.parent.is_dir() or dest.parent.is_symlink(): raise RuntimeError('unsafe parent')
 states.append(state(dest,row))
print(json.dumps({'status':'PASS_M1203_REMOTE_PREFLIGHT_MONOTONIC','states':states,'attempt_result_absent':True},sort_keys=True))
'''
    prefix = "\n".join(("ROOT=" + repr(str(REMOTE_REPO)),
        "INTERPRETER=" + repr(REMOTE_INTERPRETER), "PYTHON_VERSION='3.10.20'",
        "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
        "M1180_RESULT=" + repr(str(M1180_RESULT_REL)), "ROWS=" + repr(rows), ""))
    return (prefix + code).encode()


REMOTE_RECONCILER = r'''import hashlib,json,os,pathlib,shutil,stat,sys,tarfile
def die(message): raise RuntimeError('M1203_REMOTE_FAIL: '+message)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def safe(text):
 p=pathlib.PurePosixPath(text)
 if p.is_absolute() or not p.parts or '..' in p.parts or str(p)!=text: die('unsafe path')
 return pathlib.Path(*p.parts)
def state(path,row):
 try: mode=path.lstat().st_mode
 except FileNotFoundError: return 'ABSENT'
 if not stat.S_ISREG(mode) or path.is_symlink(): die('target symlink/nonregular')
 if path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']: die('target wrong size/SHA')
 return 'EXACT'
def absent_control(root,rel):
 p=root/safe(rel)
 if p.exists() or p.is_symlink(): die('M1180 attempt/result must remain absent')
def main():
 rows=json.loads(ROWS); root=pathlib.Path(ROOT); archive=pathlib.Path(ARCHIVE); stage=pathlib.Path(STAGE)
 temps=[]; cleanup_errors=[]; primary=None; destinations=[]; initial=[]; stage_created=False
 try:
  if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die('interpreter identity')
  if len(rows)!=2 or len({r['path'] for r in rows})!=2: die('exact-two drift')
  absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
  for row in rows:
   dest=root/safe(row['path'])
   if not dest.parent.is_dir() or dest.parent.is_symlink(): die('unsafe destination parent')
   initial.append(state(dest,row)); destinations.append(dest)
  mode=archive.lstat().st_mode
  if not stat.S_ISREG(mode) or archive.is_symlink() or digest(archive)!=ARCHIVE_SHA: die('archive SHA/type')
  if stage.exists() or stage.is_symlink(): die('stage preexists')
  stage.mkdir(mode=0o700); stage_created=True
  staged=[]
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
  for index,(src,dest,row) in enumerate(zip(staged,destinations,rows)):
   if state(dest,row)=='EXACT': continue
   temporary=dest.parent/('.'+dest.name+'.m1203.publish.tmp')
   if temporary.exists() or temporary.is_symlink(): die('publish temporary preexists')
   os.link(src,temporary); temps.append(temporary)
   if state(temporary,row)!='EXACT': die('publish temporary drift')
   current=state(dest,row)
   if current=='ABSENT': os.replace(temporary,dest)
   else: temporary.unlink()
   if state(dest,row)!='EXACT': die('published target not exact')
  absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
  final=[state(dest,row) for dest,row in zip(destinations,rows)]
  if final!=['EXACT','EXACT']: die('final both-exact gate')
 except BaseException as error: primary=error
 finally:
  for temporary in temps:
   if temporary.exists() or temporary.is_symlink():
    try: temporary.unlink()
    except BaseException as error: cleanup_errors.append(repr(error))
  if stage_created and (stage.exists() or stage.is_symlink()):
   try: shutil.rmtree(stage)
   except BaseException as error: cleanup_errors.append(repr(error))
  if archive.exists() or archive.is_symlink():
   try: archive.unlink()
   except BaseException as error: cleanup_errors.append(repr(error))
 # Exact targets are monotonic and are intentionally not rolled back.
 final=[]
 if len(destinations)==2:
  for dest,row in zip(destinations,rows): final.append(state(dest,row))
 absent_control(root,M1180_ATTEMPT); absent_control(root,M1180_RESULT)
 if primary is not None: raise primary
 if cleanup_errors: die('cleanup failed after reconciliation: '+repr(cleanup_errors))
 if final!=['EXACT','EXACT']: die('final both-exact gate')
 print(json.dumps({'status':'PASS_M1203_REMOTE_MONOTONIC_EXACT2','initial':initial,'final':final,'attempt_result_absent':True},sort_keys=True))
main()
'''


def reconciler_program(members: list[dict[str, Any]], archive_sha: str) -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    prefix = "\n".join(("ROOT=" + repr(str(REMOTE_REPO)), "ARCHIVE=" + repr(str(REMOTE_ARCHIVE)),
        "STAGE=" + repr(str(REMOTE_STAGE)), "ARCHIVE_SHA=" + repr(archive_sha),
        "INTERPRETER=" + repr(REMOTE_INTERPRETER), "PYTHON_VERSION='3.10.20'",
        "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
        "M1180_RESULT=" + repr(str(M1180_RESULT_REL)), "ROWS=" + repr(rows), ""))
    return (prefix + REMOTE_RECONCILER).encode()


def run() -> None:
    contract = load_contract(); verify_policy(contract); members = exact_members(contract)
    verify_future_hammer(contract)
    require(not LOCAL_ATTEMPT.exists() and not LOCAL_ATTEMPT.is_symlink() and
            not LOCAL_RESULT.exists() and not LOCAL_RESULT.is_symlink(), "M1203 namespace not fresh")
    preflight = subprocess.run(fixed_ssh_argv(), input=preflight_program(members),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=False, check=False)
    require(preflight.returncode == 0 and b"PASS_M1203_REMOTE_PREFLIGHT_MONOTONIC" in preflight.stdout,
            "remote monotonic preflight failed")
    LOCAL_ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("M1203_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
    with tempfile.TemporaryDirectory(prefix="m1203_m1180_missing2_") as temporary:
        archive = Path(temporary) / "exact2.tar"; archive_sha = build_archive(archive, members)
        copied = subprocess.run(fixed_scp_argv(archive), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=False, check=False)
        require(copied.returncode == 0, "fixed-argv SCP/SFTP failed")
        installed = subprocess.run(fixed_ssh_argv(), input=reconciler_program(members, archive_sha),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=False, check=False)
        require(installed.returncode == 0, "remote monotonic reconciliation failed: " +
                installed.stderr.decode("utf-8", errors="replace")[-1200:])
        remote = json.loads(installed.stdout.decode("utf-8").strip().splitlines()[-1])
        require(remote.get("status") == "PASS_M1203_REMOTE_MONOTONIC_EXACT2" and
                remote.get("final") == ["EXACT", "EXACT"] and
                remote.get("attempt_result_absent") is True, "remote receipt drift")
    receipt = {"schema": "m1203_m1180_missing2_monotonic_transport_result_r1_v1",
        "status": PASS_TOKEN, "members": members, "initial_remote_states": remote["initial"],
        "final_remote_states": remote["final"], "m1180_attempt_consumed": False,
        "gpu_or_capture_launched": False, "paper_result": False,
        "docs359_sha256": DOCS359_SHA256}
    LOCAL_RESULT.mkdir(); (LOCAL_RESULT / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(PASS_TOKEN)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print("M1203_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(1)

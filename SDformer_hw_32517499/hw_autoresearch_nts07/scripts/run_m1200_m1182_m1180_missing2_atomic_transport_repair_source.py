#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1200 inert exact-two, rollback-clean transport repair for M1180.

M1197 stopped M1195 for identity/seal drift and sequential publication.  This
additive successor is a new identity.  It installs only the two inventory-bound
dependencies missing on the A800 host.  Publication uses same-filesystem hard
links and removes every published destination on any handled publication,
verification, or postcondition failure.  It is inert until a fresh independent
M1201 hammer is admitted by exact recursive-seal digests.
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
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1200_m1182_m1180_missing2_atomic_transport_repair_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1200_m1182_m1180_missing2_atomic_transport_repair_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1200_m1182_m1180_missing2_atomic_transport_repair_source_contract_r1_20260830.json")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
STOP_REL = Path("hw_autoresearch_nts07/reviews/m1197_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_20260830")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1201_m1200_m1182_m1180_missing2_atomic_transport_repair_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
REMOTE_ARCHIVE = Path("/tmp/m1200_m1180_missing2_atomic_transport_r1.tar")
REMOTE_STAGE = REMOTE_REPO / ".m1200_m1180_missing2_atomic_transport_stage_r1"
LOCAL_ATTEMPT = HW / "results/.m1200_m1180_missing2_atomic_transport_r1_attempt_consumed"
LOCAL_RESULT = HW / "results/m1200_m1180_missing2_atomic_transport_r1_20260830"
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
PASS_TOKEN = "PASS_M1200_EXACT2_ROLLBACK_CLEAN_INSTALL__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


class RepairError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


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
        raise RepairError("missing {}: {}".format(label, path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be non-symlink regular file")


def strict_json_bytes(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RepairError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    return strict_json_bytes(path.read_bytes())


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and
            str(path) == path.as_posix(), "unsafe repository-relative path")
    return path


def validate_expected_rows(rows: Any) -> list[dict[str, Any]]:
    require(type(rows) is list and len(rows) == 2, "exact2 cardinality drift")
    expected = [
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
    require(rows == expected, "exact2 target identity/order drift")
    return rows


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    require(set(contract) == {"schema", "status", "date", "source", "test",
                              "inventory_authority", "m1197_stop_authority", "missing2",
                              "remote_audit", "transport", "publication", "future_hammer",
                              "claim_boundary", "docs359_sha256"}, "contract exact keys drift")
    require(contract["schema"] ==
            "m1200_m1182_m1180_missing2_atomic_transport_repair_source_contract_r1_v1" and
            contract["status"] ==
            "INERT_SOURCE_ONLY__ROLLBACK_CLEAN_EXACT2__FRESH_M1201_HAMMER_REQUIRED",
            "contract schema/status drift")
    require(contract["date"] == "2026-08-30", "contract date drift")
    for label, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        path = ROOT / rel
        regular(path, label)
        require(contract[label] == {"path": str(rel), "size_bytes": path.stat().st_size,
                                    "sha256": sha256(path)}, label + " identity drift")
    inventory = contract["inventory_authority"]
    require(inventory == {"path": str(INVENTORY_REL), "size_bytes": 42133,
                          "sha256": "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae"},
            "inventory authority drift")
    path = ROOT / INVENTORY_REL
    regular(path, "M1182 inventory")
    require(path.stat().st_size == inventory["size_bytes"] and
            sha256(path) == inventory["sha256"], "inventory bytes drift")
    stop = contract["m1197_stop_authority"]
    require(stop == {"review_path": str(STOP_REL / "review.json"),
                     "review_sha256": "5b2355274564721c4df91067e74f7b5ba15635ff8b101a0fc2ffadcb961d1888",
                     "status": "STOP_M1197_M1195_IDENTITY_AND_AUTHOR_SEAL_DRIFT",
                     "p0_count": 2, "p1_count": 1, "m1195_execution_authorized": False},
            "M1197 STOP authority drift")
    stop_path = ROOT / STOP_REL / "review.json"
    regular(stop_path, "M1197 STOP review")
    require(sha256(stop_path) == stop["review_sha256"], "M1197 STOP bytes drift")
    validate_expected_rows(contract["missing2"])
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def exact_members(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = validate_expected_rows(contract["missing2"])
    inventory = strict_json(ROOT / INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
            "inventory semantics drift")
    dependencies = inventory.get("dependencies", [])
    by_path = {row.get("path"): row for row in dependencies}
    require(len(by_path) == len(dependencies), "inventory duplicate path")
    members: list[dict[str, Any]] = []
    for item in expected:
        row = by_path.get(item["path"])
        require(row is not None and row.get("label") == item["label"] and
                row.get("disposition") == item["inventory_disposition"] and
                row.get("size_bytes") == item["size_bytes"] and
                row.get("sha256") == item["sha256"], "inventory exact target drift")
        rel = repo_relative(item["path"])
        path = ROOT / rel
        regular(path, "exact2 source")
        require(path.stat().st_size == item["size_bytes"] and
                sha256(path) == item["sha256"], "exact2 local identity drift")
        members.append({"path": item["path"], "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"]})
    require(len(members) == 2 and len({row["path"] for row in members}) == 2,
            "exact2 uniqueness drift")
    return members


def verify_policy(contract: dict[str, Any]) -> None:
    require(contract["remote_audit"] == {
        "audited_remote_existing_rows": 55, "missing_rows": 2,
        "mismatched_rows": 0, "present_exact_rows": 53,
        "m1180_attempt_absent": True, "m1180_result_absent": True,
        "audit_read_only": True}, "remote audit statement drift")
    require(contract["publication"] == {
        "same_filesystem_hardlink_publish": True,
        "preexisting_destinations_rejected": True,
        "rollback_all_published_on_any_handled_failure": True,
        "second_publication_failure_test_required": True,
        "post_sha_failure_rolls_back_both": True,
        "m1180_postcondition_failure_rolls_back_both": True,
        "external_process_termination_atomicity_claimed": False}, "publication policy drift")
    transport = contract["transport"]
    require(transport == {
        "member_count": 2,
        "protocol": "LOCAL_TARFILE_PLUS_OPENSSH_SCP_DEFAULT_SFTP_PLUS_REMOTE_PYTHON_STDLIB_SAFE_EXTRACT",
        "local_ssh": {"path": "/usr/bin/ssh", "size_bytes": 775656,
                      "sha256": "3cbb1eb62b4fec407778373e84105378c1860648f8817086aa4176da11e93a88"},
        "local_scp": {"path": "/usr/bin/scp", "size_bytes": 105304,
                      "sha256": "35dc3481f433276e6071461500097c86dee5281fb7d64eed46bec8c79c45a666"},
        "remote_host": REMOTE_HOST, "remote_port": 10037,
        "ssh_control_path": SSH_CONTROL_PATH, "remote_repository": str(REMOTE_REPO),
        "remote_interpreter": REMOTE_INTERPRETER, "remote_python_version": "3.10.20",
        "remote_archive": str(REMOTE_ARCHIVE), "remote_stage": str(REMOTE_STAGE),
        "shell": False, "fixed_argv": True, "safe_archive_paths": True,
        "symlink_rejected": True, "post_install_size_sha_each_member": True,
        "remote_temp_cleanup": True, "m1180_attempt_result_absent_before_after": True,
        "automatic_retry": False}, "transport policy drift")
    for name in ("local_ssh", "local_scp"):
        row = transport[name]
        path = Path(row["path"])
        regular(path, name)
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                name + " executable identity drift")
    require(contract["claim_boundary"] == {
        "source_only": True, "remote_transfer_executed_by_author": False,
        "gpu_or_capture_executed_by_author": False, "eda_executed_by_author": False,
        "m1180_attempt_consumed": False, "paper_result": False,
        "m1195_or_m1197_modified": False, "docs359_modified": False},
        "claim boundary drift")


def verify_future_hammer(contract: dict[str, Any]) -> None:
    future = contract["future_hammer"]
    require(future == {
        "directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1201_m1200_m1182_m1180_missing2_atomic_transport_repair_hammer_r1_v1",
        "required_status": "PASS_M1200_ROLLBACK_CLEAN_EXACT2_TRANSPORT_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "review_env": "M1200_EXPECTED_HAMMER_REVIEW_SHA256",
        "manifest_env": "M1200_EXPECTED_HAMMER_MANIFEST_SHA256",
        "outer_env": "M1200_EXPECTED_HAMMER_OUTER_SHA256"}, "future hammer drift")
    paths = [ROOT / FUTURE_HAMMER_REL / name for name in
             ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")]
    for path in paths:
        regular(path, "fresh M1201 hammer")
    expected = [os.environ.get(future[key], "") for key in
                ("review_env", "manifest_env", "outer_env")]
    require(all(len(value) == 64 for value in expected), "fresh M1201 digest env absent")
    require([sha256(path) for path in paths] == expected, "fresh M1201 digest mismatch")
    require(paths[2].read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh M1201 recursive seal mismatch")
    review = strict_json(paths[0])
    require(review.get("schema") == future["required_schema"] and
            review.get("status") == future["required_status"],
            "fresh M1201 semantic admission mismatch")


def publish_exact2_atomic(staged: list[Path], destinations: list[Path],
                          verify: Callable[[], None],
                          link: Callable[[Path, Path], None] = os.link) -> None:
    """Publish two staged files, rolling back both on any handled failure."""
    require(len(staged) == len(destinations) == 2, "atomic publish exact2 required")
    require(all(not path.exists() and not path.is_symlink() for path in destinations),
            "atomic publish destination preexists")
    published: list[Path] = []
    try:
        for source, destination in zip(staged, destinations):
            link(source, destination)
            published.append(destination)
        verify()
    except BaseException:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
            except OSError as error:
                rollback_errors.append(str(error))
        require(not rollback_errors and
                all(not path.exists() and not path.is_symlink() for path in destinations),
                "atomic publish rollback failed")
        raise


def fixed_ssh_argv() -> list[str]:
    return [str(SSH), "-p", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", REMOTE_HOST, REMOTE_INTERPRETER, "-I", "-"]


def fixed_scp_argv(local_archive: Path) -> list[str]:
    return [str(SCP), "-P", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", str(local_archive),
            REMOTE_HOST + ":" + str(REMOTE_ARCHIVE)]


def build_archive(path: Path, members: list[dict[str, Any]]) -> str:
    require(len(members) == 2, "archive exact2 required")
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for row in members:
            source = ROOT / repo_relative(row["path"])
            regular(source, "archive source")
            require(source.stat().st_size == row["size_bytes"] and
                    sha256(source) == row["sha256"], "archive source drift")
            info = tarfile.TarInfo(row["path"])
            info.size = row["size_bytes"]
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
            require(sha256(source) == row["sha256"], "archive source changed during read")
    regular(path, "local archive")
    return sha256(path)


def preflight_program(members: list[dict[str, Any]], root: Path = REMOTE_REPO,
                      archive: Path = REMOTE_ARCHIVE, stage: Path = REMOTE_STAGE,
                      interpreter: str = REMOTE_INTERPRETER,
                      python_version: str = "3.10.20") -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    code = r'''import json,pathlib,sys
def safe(text):
 p=pathlib.PurePosixPath(text)
 assert not p.is_absolute() and p.parts and '..' not in p.parts and str(p)==text
 return pathlib.Path(*p.parts)
root=pathlib.Path(ROOT); archive=pathlib.Path(ARCHIVE); stage=pathlib.Path(STAGE)
assert sys.executable==INTERPRETER and sys.version.split()[0]==PYTHON_VERSION
assert root.is_dir() and not root.is_symlink()
assert not archive.exists() and not archive.is_symlink() and not stage.exists() and not stage.is_symlink()
assert not (root/safe(M1180_ATTEMPT)).exists() and not (root/safe(M1180_ATTEMPT)).is_symlink()
assert not (root/safe(M1180_RESULT)).exists() and not (root/safe(M1180_RESULT)).is_symlink()
for row in json.loads(ROWS):
 dest=root/safe(row['path'])
 assert dest.parent.is_dir() and not dest.parent.is_symlink()
 assert not dest.exists() and not dest.is_symlink()
print('PASS_M1200_REMOTE_PREFLIGHT__EXACT2_MISSING__M1180_ABSENT__NO_WRITE')
'''
    prefix = "\n".join(("ROOT=" + repr(str(root)),
                         "ARCHIVE=" + repr(str(archive)),
                         "STAGE=" + repr(str(stage)),
                         "INTERPRETER=" + repr(interpreter),
                         "PYTHON_VERSION=" + repr(python_version),
                         "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
                         "M1180_RESULT=" + repr(str(M1180_RESULT_REL)),
                         "ROWS=" + repr(rows), ""))
    return (prefix + code).encode("utf-8")


REMOTE_EXTRACTOR = r'''import hashlib,json,os,pathlib,shutil,stat,sys,tarfile
def die(message): raise RuntimeError('M1200_REMOTE_FAIL: '+message)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def safe(text):
 p=pathlib.PurePosixPath(text)
 if p.is_absolute() or not p.parts or '..' in p.parts or str(p)!=text: die('unsafe path')
 return pathlib.Path(*p.parts)
def absent(path,label):
 if path.exists() or path.is_symlink(): die(label+' must be absent')
def main():
 rows=json.loads(ROWS); root=pathlib.Path(ROOT); archive=pathlib.Path(ARCHIVE); stage=pathlib.Path(STAGE)
 attempt=root/safe(M1180_ATTEMPT); result=root/safe(M1180_RESULT)
 destinations=[]; published=[]
 try:
  if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die('interpreter identity')
  if not root.is_dir() or root.is_symlink(): die('unsafe root')
  if len(rows)!=2 or len({r['path'] for r in rows})!=2: die('extra/duplicate member')
  absent(attempt,'M1180 attempt'); absent(result,'M1180 result')
  mode=archive.lstat().st_mode
  if not stat.S_ISREG(mode) or archive.is_symlink() or digest(archive)!=ARCHIVE_SHA: die('archive SHA/type')
  absent(stage,'stage')
  for row in rows:
   dest=root/safe(row['path'])
   if not dest.parent.is_dir() or dest.parent.is_symlink(): die('unsafe destination parent')
   absent(dest,'preexisting destination'); destinations.append(dest)
  stage.mkdir(mode=0o700)
  with tarfile.open(archive,'r:') as tf:
   items=tf.getmembers()
   if [m.name for m in items]!=[r['path'] for r in rows]: die('extra/path/order member')
   for member,row in zip(items,rows):
    if not member.isfile() or member.issym() or member.islnk() or member.size!=row['size_bytes']: die('symlink/type/size')
    out=stage/safe(member.name); out.parent.mkdir(parents=True,exist_ok=True)
    src=tf.extractfile(member)
    if src is None: die('member unreadable')
    h=hashlib.sha256(); count=0
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444)
    with os.fdopen(fd,'wb') as dst:
     while True:
      block=src.read(1<<20)
      if not block: break
      dst.write(block); h.update(block); count+=len(block)
     dst.flush(); os.fsync(dst.fileno())
    if count!=row['size_bytes'] or h.hexdigest()!=row['sha256']: die('remote member SHA mismatch')
  absent(attempt,'M1180 attempt'); absent(result,'M1180 result')
  for row,dest in zip(rows,destinations):
   absent(dest,'destination race'); os.link(stage/safe(row['path']),dest); published.append(dest)
  for row,dest in zip(rows,destinations):
   mode=dest.lstat().st_mode
   if not stat.S_ISREG(mode) or dest.is_symlink() or dest.stat().st_size!=row['size_bytes'] or digest(dest)!=row['sha256']: die('post-install SHA/type')
  absent(attempt,'M1180 attempt postcondition'); absent(result,'M1180 result postcondition')
  print(json.dumps({'status':'PASS_M1200_REMOTE_ROLLBACK_CLEAN_EXACT2_INSTALL','members':2,'verified':2,'m1180_attempt_absent':True,'m1180_result_absent':True},sort_keys=True))
 except BaseException:
  rollback=[]
  for dest in reversed(published):
   try:
    if dest.exists() or dest.is_symlink(): dest.unlink()
   except BaseException as error: rollback.append(str(error))
  if rollback or any(d.exists() or d.is_symlink() for d in destinations):
   raise RuntimeError('M1200_REMOTE_ROLLBACK_FAILED: '+repr(rollback))
  raise
 finally:
  if stage.exists() and not stage.is_symlink(): shutil.rmtree(stage)
  if archive.exists() and not archive.is_symlink(): archive.unlink()
main()
'''


def remote_program(members: list[dict[str, Any]], archive_sha: str,
                   root: Path = REMOTE_REPO, archive: Path = REMOTE_ARCHIVE,
                   stage: Path = REMOTE_STAGE, interpreter: str = REMOTE_INTERPRETER,
                   python_version: str = "3.10.20") -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    prefix = "\n".join(("ROWS=" + repr(rows), "ROOT=" + repr(str(root)),
                         "ARCHIVE=" + repr(str(archive)),
                         "STAGE=" + repr(str(stage)),
                         "INTERPRETER=" + repr(interpreter),
                         "PYTHON_VERSION=" + repr(python_version),
                         "ARCHIVE_SHA=" + repr(archive_sha),
                         "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
                         "M1180_RESULT=" + repr(str(M1180_RESULT_REL)), ""))
    return (prefix + REMOTE_EXTRACTOR).encode("utf-8")


def consume_local_attempt() -> None:
    LOCAL_ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("M1200_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    require(len(sys.argv) == 1, "zero arguments required")
    contract = load_contract()
    verify_policy(contract)
    members = exact_members(contract)
    verify_future_hammer(contract)
    require(not LOCAL_ATTEMPT.exists() and not LOCAL_RESULT.exists(),
            "M1200 attempt/result not fresh")
    consume_local_attempt()
    preflight = subprocess.run(fixed_ssh_argv(), input=preflight_program(members),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               shell=False, check=False)
    require(preflight.returncode == 0 and
            preflight.stdout.decode("utf-8", "replace").strip() ==
            "PASS_M1200_REMOTE_PREFLIGHT__EXACT2_MISSING__M1180_ABSENT__NO_WRITE",
            "M1200 remote preflight failed")
    with tempfile.TemporaryDirectory(prefix="m1200_m1180_missing2_") as temporary:
        archive = Path(temporary) / "exact2.tar"
        archive_sha = build_archive(archive, members)
        copied = subprocess.run(fixed_scp_argv(archive), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=False, check=False)
        require(copied.returncode == 0, "M1200 fixed-argv SCP/SFTP failed")
        installed = subprocess.run(fixed_ssh_argv(), input=remote_program(members, archive_sha),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, check=False)
        require(installed.returncode == 0, "M1200 remote exact2 install failed")
        receipt = strict_json_bytes(installed.stdout.strip())
        require(receipt == {"m1180_attempt_absent": True, "m1180_result_absent": True,
                            "members": 2,
                            "status": "PASS_M1200_REMOTE_ROLLBACK_CLEAN_EXACT2_INSTALL",
                            "verified": 2}, "M1200 remote receipt drift")
    LOCAL_RESULT.mkdir(mode=0o755)
    receipt = {"schema": "m1200_m1180_missing2_atomic_transport_result_r1_v1",
               "status": PASS_TOKEN, "members": 2, "remote_post_sha_verified": True,
               "handled_failure_rollback_clean": True, "remote_temp_cleanup": True,
               "m1180_attempt_consumed": False, "gpu_or_capture_consumed": False,
               "paper_result": False}
    (LOCAL_RESULT / "transport_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (LOCAL_RESULT / "RUN_COMPLETE.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RepairError, OSError, ValueError, json.JSONDecodeError) as error:
        print("M1200_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

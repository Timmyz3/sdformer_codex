#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1195 inert exact-two transport repair for the M1182/M1180 preflight.

The sealed M1182 inventory marked two files as remote-existing, but a read-only
remote audit found them absent.  This additive adapter may install only those
two inventory-bound regular files.  It is inert until a fresh different-author
M1196 hammer is supplied through three exact digest environment variables.
It never launches M1180, GPU work, VCS, or DC.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1195_m1182_m1180_missing2_transport_repair_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1195_m1182_m1180_missing2_transport_repair_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1195_m1182_m1180_missing2_transport_repair_source_contract_r1_20260830.json")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1196_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
REMOTE_ARCHIVE = Path("/tmp/m1195_m1180_missing2_transport_r1.tar")
REMOTE_STAGE = REMOTE_REPO / ".m1195_m1180_missing2_transport_stage_r1"
LOCAL_ATTEMPT = HW / "results/.m1195_m1180_missing2_transport_r1_attempt_consumed"
LOCAL_RESULT = HW / "results/m1195_m1180_missing2_transport_r1_20260830"
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
PASS_TOKEN = "PASS_M1195_EXACT2_PREFLIGHT_DEPENDENCIES_INSTALLED__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


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


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    require(set(contract) == {"schema", "status", "date", "source", "test",
                              "inventory_authority", "missing2", "remote_audit",
                              "transport", "future_hammer", "claim_boundary",
                              "docs359_sha256"}, "contract exact keys drift")
    require(contract["schema"] ==
            "m1195_m1182_m1180_missing2_transport_repair_source_contract_r1_v1" and
            contract["status"] ==
            "INERT_SOURCE_ONLY__EXACT2__FRESH_M1196_HAMMER_REQUIRED",
            "contract schema/status drift")
    require(contract["date"] == "2026-08-30", "contract date drift")
    for label, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        row = contract[label]
        path = ROOT / rel
        regular(path, label)
        require(row == {"path": str(rel), "size_bytes": path.stat().st_size,
                        "sha256": sha256(path)}, label + " identity drift")
    inventory = contract["inventory_authority"]
    require(inventory == {"path": str(INVENTORY_REL), "size_bytes": 42133,
                          "sha256": "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae"},
            "inventory authority drift")
    inventory_path = ROOT / INVENTORY_REL
    regular(inventory_path, "M1182 dependency inventory")
    require(inventory_path.stat().st_size == inventory["size_bytes"] and
            sha256(inventory_path) == inventory["sha256"], "inventory bytes drift")
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def exact_members(contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = contract["missing2"]
    require(type(expected) is list and len(expected) == 2, "missing2 cardinality drift")
    inventory = strict_json(ROOT / INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
            "inventory semantics drift")
    rows = {row.get("path"): row for row in inventory.get("dependencies", [])}
    require(len(rows) == len(inventory.get("dependencies", [])), "inventory path uniqueness drift")
    members: list[dict[str, Any]] = []
    for expected_row in expected:
        require(set(expected_row) == {"label", "path", "size_bytes", "sha256",
                                      "inventory_disposition", "remote_audit_state"},
                "missing2 row exact keys drift")
        text = expected_row["path"]
        rel = repo_relative(text)
        row = rows.get(text)
        require(row is not None and row.get("label") == expected_row["label"] and
                row.get("disposition") == "remote_existing_hash_verify" and
                row.get("size_bytes") == expected_row["size_bytes"] and
                row.get("sha256") == expected_row["sha256"],
                "missing2 row not exactly bound by M1182 inventory")
        require(expected_row["inventory_disposition"] == "remote_existing_hash_verify" and
                expected_row["remote_audit_state"] == "MISSING",
                "missing2 audit semantics drift")
        path = ROOT / rel
        regular(path, "exact2 source")
        require(path.stat().st_size == expected_row["size_bytes"] and
                sha256(path) == expected_row["sha256"], "exact2 local identity drift")
        members.append({"path": text, "size_bytes": expected_row["size_bytes"],
                        "sha256": expected_row["sha256"]})
    require(len({row["path"] for row in members}) == 2, "missing2 uniqueness drift")
    return members


def verify_transport_contract(contract: dict[str, Any]) -> None:
    require(contract["remote_audit"] == {
        "audited_remote_existing_rows": 55, "missing_rows": 2,
        "mismatched_rows": 0, "present_exact_rows": 53,
        "m1180_attempt_absent": True, "m1180_result_absent": True,
        "audit_read_only": True}, "remote audit statement drift")
    require(contract["transport"] == {
        "member_count": 2, "protocol": "LOCAL_TARFILE_PLUS_OPENSSH_SCP_DEFAULT_SFTP_PLUS_REMOTE_PYTHON_STDLIB_SAFE_EXTRACT",
        "local_ssh": {"path": "/usr/bin/ssh", "size_bytes": 775656,
                      "sha256": "3cbb1eb62b4fec407778373e84105378c1860648f8817086aa4176da11e93a88"},
        "local_scp": {"path": "/usr/bin/scp", "size_bytes": 105304,
                      "sha256": "35dc3481f433276e6071461500097c86dee5281fb7d64eed46bec8c79c45a666"},
        "remote_host": REMOTE_HOST, "remote_port": 10037,
        "ssh_control_path": SSH_CONTROL_PATH, "remote_repository": str(REMOTE_REPO),
        "remote_interpreter": REMOTE_INTERPRETER, "remote_python_version": "3.10.20",
        "remote_archive": str(REMOTE_ARCHIVE), "remote_stage": str(REMOTE_STAGE),
        "shell": False, "fixed_argv": True, "safe_archive_paths": True,
        "symlink_rejected": True, "destinations_must_be_absent": True,
        "post_install_size_sha_each_member": True, "remote_temp_cleanup": True,
        "m1180_attempt_result_absent_before_after": True, "automatic_retry": False},
        "transport contract drift")
    for key in ("local_ssh", "local_scp"):
        row = contract["transport"][key]
        path = Path(row["path"])
        regular(path, key)
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                key + " executable identity drift")
    require(contract["claim_boundary"] == {
        "source_only": True, "remote_transfer_executed_by_author": False,
        "gpu_or_capture_executed_by_author": False, "eda_executed_by_author": False,
        "m1180_attempt_consumed": False, "paper_result": False,
        "original_authorities_modified": False, "docs359_modified": False},
        "claim boundary drift")


def verify_future_hammer(contract: dict[str, Any]) -> None:
    future = contract["future_hammer"]
    require(future == {
        "directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1196_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_v1",
        "required_status": "PASS_M1195_EXACT2_TRANSPORT_REPAIR_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "review_env": "M1195_EXPECTED_HAMMER_REVIEW_SHA256",
        "manifest_env": "M1195_EXPECTED_HAMMER_MANIFEST_SHA256",
        "outer_env": "M1195_EXPECTED_HAMMER_OUTER_SHA256"}, "future hammer drift")
    paths = [ROOT / FUTURE_HAMMER_REL / name for name in
             ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")]
    for path in paths:
        regular(path, "fresh M1196 hammer")
    expected = [os.environ.get(future[key], "") for key in
                ("review_env", "manifest_env", "outer_env")]
    require(all(len(value) == 64 for value in expected), "fresh M1196 digest env absent")
    require([sha256(path) for path in paths] == expected, "fresh M1196 digest mismatch")
    require(paths[2].read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh M1196 recursive seal mismatch")
    review = strict_json(paths[0])
    require(review.get("schema") == future["required_schema"] and
            review.get("status") == future["required_status"],
            "fresh M1196 semantic admission mismatch")


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
            source = ROOT / repo_relative(row["path"])
            regular(source, "archive source")
            require(source.stat().st_size == row["size_bytes"] and
                    sha256(source) == row["sha256"], "archive source identity drift")
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


def preflight_program(members: list[dict[str, Any]]) -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    code = r'''import json,pathlib,stat,sys
root=pathlib.Path(ROOT); archive=pathlib.Path(ARCHIVE); stage=pathlib.Path(STAGE)
def safe(text):
 p=pathlib.PurePosixPath(text)
 assert not p.is_absolute() and p.parts and '..' not in p.parts and str(p)==text
 return pathlib.Path(*p.parts)
assert sys.executable==INTERPRETER and sys.version.split()[0]=='3.10.20'
assert root.is_dir() and not root.is_symlink()
assert not archive.exists() and not archive.is_symlink() and not stage.exists() and not stage.is_symlink()
assert not (root/safe(M1180_ATTEMPT)).exists() and not (root/safe(M1180_ATTEMPT)).is_symlink()
assert not (root/safe(M1180_RESULT)).exists() and not (root/safe(M1180_RESULT)).is_symlink()
for row in json.loads(ROWS):
 rel=safe(row['path']); dest=root/rel; parent=dest.parent
 assert parent.is_dir() and not parent.is_symlink()
 assert not dest.exists() and not dest.is_symlink()
print('PASS_M1195_REMOTE_PREFLIGHT__EXACT2_MISSING__M1180_ABSENT__NO_WRITE')
'''
    prefix = "\n".join(("ROOT=" + repr(str(REMOTE_REPO)),
                         "ARCHIVE=" + repr(str(REMOTE_ARCHIVE)),
                         "STAGE=" + repr(str(REMOTE_STAGE)),
                         "INTERPRETER=" + repr(REMOTE_INTERPRETER),
                         "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
                         "M1180_RESULT=" + repr(str(M1180_RESULT_REL)),
                         "ROWS=" + repr(rows), ""))
    return (prefix + code).encode("utf-8")


REMOTE_EXTRACTOR = r'''import hashlib,json,os,pathlib,shutil,stat,sys,tarfile
def die(message): raise SystemExit('M1195_REMOTE_FAIL: '+message)
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
 if path.exists() or path.is_symlink(): die(label+' must remain absent')
def main():
 rows=json.loads(ROWS); root=pathlib.Path(ROOT); archive=pathlib.Path(ARCHIVE); stage=pathlib.Path(STAGE)
 attempt=root/safe(M1180_ATTEMPT); result=root/safe(M1180_RESULT)
 if sys.executable!=INTERPRETER or sys.version.split()[0]!='3.10.20': die('interpreter identity')
 if not root.is_dir() or root.is_symlink(): die('unsafe root')
 if len(rows)!=2 or len({r['path'] for r in rows})!=2: die('exact2 cardinality')
 absent(attempt,'M1180 attempt'); absent(result,'M1180 result')
 try:
  mode=archive.lstat().st_mode
 except FileNotFoundError: die('archive absent')
 if not stat.S_ISREG(mode) or archive.is_symlink() or digest(archive)!=ARCHIVE_SHA: die('archive identity')
 absent(stage,'stage')
 destinations=[]
 for row in rows:
  dest=root/safe(row['path'])
  if not dest.parent.is_dir() or dest.parent.is_symlink(): die('unsafe/missing destination parent')
  absent(dest,'destination'); destinations.append(dest)
 stage.mkdir(mode=0o700)
 installed=[]
 try:
  with tarfile.open(archive,'r:') as tf:
   items=tf.getmembers()
   if [m.name for m in items]!=[r['path'] for r in rows]: die('archive order/set')
   for member,row in zip(items,rows):
    if not member.isfile() or member.issym() or member.islnk() or member.size!=row['size_bytes']: die('member type/size')
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
    if count!=row['size_bytes'] or h.hexdigest()!=row['sha256']: die('member SHA')
  absent(attempt,'M1180 attempt'); absent(result,'M1180 result')
  for row,dest in zip(rows,destinations):
   absent(dest,'destination changed'); os.replace(stage/safe(row['path']),dest); installed.append(dest)
  for row,dest in zip(rows,destinations):
   mode=dest.lstat().st_mode
   if not stat.S_ISREG(mode) or dest.is_symlink() or dest.stat().st_size!=row['size_bytes'] or digest(dest)!=row['sha256']: die('post-install identity')
  absent(attempt,'M1180 attempt'); absent(result,'M1180 result')
  print(json.dumps({'status':'PASS_M1195_REMOTE_SAFE_EXACT2_INSTALL','members':2,'verified':2,'m1180_attempt_absent':True,'m1180_result_absent':True},sort_keys=True))
 finally:
  if stage.exists() and not stage.is_symlink(): shutil.rmtree(stage)
  if archive.exists() and not archive.is_symlink(): archive.unlink()
main()
'''


def remote_program(members: list[dict[str, Any]], archive_sha: str) -> bytes:
    rows = json.dumps(members, sort_keys=True, separators=(",", ":"))
    prefix = "\n".join(("ROWS=" + repr(rows), "ROOT=" + repr(str(REMOTE_REPO)),
                         "ARCHIVE=" + repr(str(REMOTE_ARCHIVE)),
                         "STAGE=" + repr(str(REMOTE_STAGE)),
                         "INTERPRETER=" + repr(REMOTE_INTERPRETER),
                         "ARCHIVE_SHA=" + repr(archive_sha),
                         "M1180_ATTEMPT=" + repr(str(M1180_ATTEMPT_REL)),
                         "M1180_RESULT=" + repr(str(M1180_RESULT_REL)), ""))
    return (prefix + REMOTE_EXTRACTOR).encode("utf-8")


def consume_local_attempt() -> None:
    LOCAL_ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("M1195_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    require(len(sys.argv) == 1, "zero arguments required")
    contract = load_contract()
    verify_transport_contract(contract)
    members = exact_members(contract)
    verify_future_hammer(contract)
    require(not LOCAL_ATTEMPT.exists() and not LOCAL_RESULT.exists(),
            "M1195 transport attempt/result not fresh")
    consume_local_attempt()
    preflight = subprocess.run(fixed_ssh_argv(), input=preflight_program(members),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               shell=False, check=False)
    require(preflight.returncode == 0 and
            preflight.stdout.decode("utf-8", "replace").strip() ==
            "PASS_M1195_REMOTE_PREFLIGHT__EXACT2_MISSING__M1180_ABSENT__NO_WRITE",
            "M1195 remote preflight failed")
    with tempfile.TemporaryDirectory(prefix="m1195_m1180_missing2_") as temporary:
        archive = Path(temporary) / "exact2.tar"
        archive_sha = build_archive(archive, members)
        copied = subprocess.run(fixed_scp_argv(archive), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=False, check=False)
        require(copied.returncode == 0, "M1195 fixed-argv SCP/SFTP failed")
        installed = subprocess.run(fixed_ssh_argv(), input=remote_program(members, archive_sha),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, check=False)
        require(installed.returncode == 0, "M1195 remote exact2 install failed")
        receipt = strict_json_bytes(installed.stdout.strip())
        require(receipt == {"m1180_attempt_absent": True, "m1180_result_absent": True,
                            "members": 2, "status": "PASS_M1195_REMOTE_SAFE_EXACT2_INSTALL",
                            "verified": 2}, "M1195 remote receipt drift")
    LOCAL_RESULT.mkdir(mode=0o755)
    receipt = {"schema": "m1195_m1180_missing2_transport_result_r1_v1",
               "status": PASS_TOKEN, "members": 2, "remote_post_sha_verified": True,
               "remote_temp_cleanup": True, "m1180_attempt_consumed": False,
               "gpu_or_capture_consumed": False, "paper_result": False}
    (LOCAL_RESULT / "transport_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (LOCAL_RESULT / "RUN_COMPLETE.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RepairError, OSError, ValueError, json.JSONDecodeError) as error:
        print("M1195_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

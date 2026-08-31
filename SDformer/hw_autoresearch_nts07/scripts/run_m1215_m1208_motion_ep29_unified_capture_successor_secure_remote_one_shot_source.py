#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Inert M1215 secure successor for the unchanged M1208 capture identity.

The source is unusable until a fresh M1216 release hammer binds this file,
the production launch contract, its transfer inventory, and the release-author
double seal.  Production uses an authenticated remote mktemp directory, SCP of
one exact tar archive, an ABSENT-or-EXACT monotonic publisher, full post-SHA
verification of the old M1180 dependency inventory, and exactly one M1208
launcher invocation.  Import is side-effect free.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1215_m1208_motion_ep29_unified_capture_successor_secure_remote_one_shot_source.py")
SOURCE_CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1215_m1208_motion_ep29_unified_capture_successor_secure_release_source_contract_r1_20260830.json")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1215_m1208_motion_ep29_unified_capture_successor_secure_remote_one_shot_source.py")
LAUNCHER_REL = Path("hw_autoresearch_nts07/scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py")
LAUNCH_CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1215_m1208_motion_ep29_unified_capture_successor_remote_dependency_inventory_r1_20260830.json")
TRANSFER_LIST_REL = Path("hw_autoresearch_nts07/contracts/m1215_m1208_motion_ep29_unified_capture_successor_remote_transfer_files_r1_20260830.txt")
OLD_INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
AUTHOR_REL = Path("hw_autoresearch_nts07/reviews/m1215_m1210_motion_ep29_unified_capture_successor_release_author_r1_20260830")
M1216_REL = Path("hw_autoresearch_nts07/reviews/m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_20260830")
FORENSIC_REL = Path("hw_autoresearch_nts07/reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
REMOTE_TEMP_TEMPLATE = "/tmp/m1215_m1208.XXXXXXXXXXXX"
REMOTE_TEMP_RE = re.compile(r"\A/tmp/m1215_m1208\.[A-Za-z0-9]{12}\Z")
REMOTE_ARCHIVE_BASENAME = "exact_release.tar"
LOCAL_ATTEMPT = HW / "results/.m1215_m1208_successor_secure_transfer_and_launch_r1_attempt_consumed"
M1210_FAILED_ATTEMPT = HW / "results/.m1210_m1208_secure_transfer_and_launch_r1_attempt_consumed"
M1210_FAILED_TOKEN = "M1210_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n"
M1210_FAILED_SHA = "b60af667912eae9f19fb93aaf201fc342cfdd22e9add4bfeac0e55c09268e5f6"
M1208_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1208_RESULT_REL = Path("hw_autoresearch_nts07/results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830")
M1208_LOG_REL = Path("hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
M1180_LOG_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
M1180_TOKEN = "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
M1216_SCHEMA = "m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_v1"
M1216_STATUS = "PASS_M1215_SUCCESSOR_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED"
PASS_TOKEN = "PASS_M1215_SUCCESSOR_TRANSFER_AND_M1208_ONE_SHOT_LAUNCH__RESULT_HAMMER_REQUIRED"


class ReleaseError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a non-symlink regular file")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ReleaseError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == text, "unsafe repository-relative path")
    return path


def exact_state(path: Path, row: dict[str, Any]) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "ABSENT"
    require(stat.S_ISREG(mode) and not path.is_symlink(), "target is symlink/nonregular")
    require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
            "target size/SHA drift: " + row["path"])
    return "EXACT"


def canonical_verify_double_seal(root: Path) -> dict[str, str]:
    regular(root / "SHA256SUMS", "manifest")
    regular(root / "SHA256SUMS.seal.sha256", "outer seal")
    outer = (root / "SHA256SUMS.seal.sha256").read_text(encoding="ascii").split()
    require(len(outer) == 2 and outer[1] == "SHA256SUMS" and
            outer[0] == sha256(root / "SHA256SUMS"), "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
                "malformed manifest")
        name = parts[1]
        require("/" not in name and name not in rows and name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
                "unsafe/duplicate manifest member")
        regular(root / name, "sealed member")
        require(sha256(root / name) == parts[0], "sealed member SHA mismatch")
        rows[name] = parts[0]
    require(rows, "empty seal")
    return rows


def load_release() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_contract = strict_json(ROOT / SOURCE_CONTRACT_REL)
    require(source_contract.get("schema") ==
            "m1215_m1208_motion_ep29_unified_capture_successor_secure_release_source_contract_r1_v1" and
            source_contract.get("status") ==
            "INERT_SOURCE_ONLY__FRESH_M1216_HAMMER_REQUIRED__NO_REMOTE_NO_GPU",
            "source contract semantic drift")
    require(source_contract["source"]["sha256"] == sha256(Path(__file__).resolve()) and
            source_contract["successor_launcher"]["sha256"] == sha256(ROOT / LAUNCHER_REL) and
            source_contract["test"]["sha256"] == sha256(ROOT / TEST_REL) and
            source_contract["launch_contract"]["sha256"] == sha256(ROOT / LAUNCH_CONTRACT_REL) and
            source_contract["inventory"]["sha256"] == sha256(ROOT / INVENTORY_REL) and
            source_contract["transfer_list"]["sha256"] == sha256(ROOT / TRANSFER_LIST_REL),
            "source contract exact binding mismatch")
    inventory = strict_json(ROOT / INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1215_m1208_motion_ep29_unified_capture_successor_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "EXACT_SUCCESSOR_TRANSFER__OLD_DEPENDENCIES_POST_SHA",
            "inventory semantic drift")
    old_authority = inventory.get("old_dependency_inventory", {})
    old_path = ROOT / OLD_INVENTORY_REL
    regular(old_path, "old M1182 dependency inventory")
    require(old_authority.get("path") == OLD_INVENTORY_REL.as_posix() and
            old_authority.get("size_bytes") == old_path.stat().st_size and
            old_authority.get("sha256") == sha256(old_path) and
            old_authority.get("row_count") == 95,
            "old dependency-inventory authority drift")
    old_rows = strict_json(old_path).get("dependencies")
    require(isinstance(old_rows, list) and len(old_rows) == 95 and
            len({row.get("path") for row in old_rows}) == 95,
            "old dependency population drift")
    for row in old_rows:
        repo_relative(row["path"])
        require(type(row.get("size_bytes")) is int and row["size_bytes"] >= 0 and
                re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None,
                "old dependency row identity malformed")
    fixed = inventory.get("transfer_required")
    require(isinstance(fixed, list) and fixed, "empty transfer inventory")
    listed = (ROOT / TRANSFER_LIST_REL).read_text(encoding="utf-8").splitlines()
    require(listed == [row["path"] for row in fixed] and len(listed) == len(set(listed)),
            "transfer list/inventory mismatch")
    for row in fixed:
        path = ROOT / repo_relative(row["path"])
        regular(path, "fixed transfer member")
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                "fixed transfer member identity drift")
    return source_contract, inventory, fixed


def load_m1216(source_contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = canonical_verify_double_seal(ROOT / M1216_REL)
    require("review.json" in rows, "M1216 lacks review.json")
    review = strict_json(ROOT / M1216_REL / "review.json")
    require(review.get("schema") == M1216_SCHEMA and review.get("status") == M1216_STATUS and
            review.get("verdict") == "GO" and isinstance(review.get("score"), int) and
            review.get("score") >= 95 and review.get("p0_count") == 0 and
            review.get("p1_count") == 0,
            "M1216 semantic admission mismatch")
    bindings = review.get("bindings", {})
    require(bindings.get("source_sha256") == sha256(Path(__file__).resolve()) and
            bindings.get("source_contract_sha256") == sha256(ROOT / SOURCE_CONTRACT_REL) and
            bindings.get("launch_contract_sha256") == sha256(ROOT / LAUNCH_CONTRACT_REL) and
            bindings.get("inventory_sha256") == sha256(ROOT / INVENTORY_REL) and
            bindings.get("transfer_list_sha256") == sha256(ROOT / TRANSFER_LIST_REL) and
            bindings.get("release_author_manifest_sha256") == sha256(ROOT / AUTHOR_REL / "SHA256SUMS") and
            bindings.get("release_author_outer_file_sha256") == sha256(ROOT / AUTHOR_REL / "SHA256SUMS.seal.sha256") and
            bindings.get("successor_launcher_sha256") == sha256(ROOT / LAUNCHER_REL) and
            bindings.get("m1210_failure_marker_sha256") == sha256(M1210_FAILED_ATTEMPT),
            "M1216 exact binding mismatch")
    auth = review.get("authorization", {})
    require(auth.get("secure_transfer") is True and auth.get("exact_remote_launch") is True and
            auth.get("launch_count") == 1 and auth.get("automatic_retry") is False,
            "M1216 authorization mismatch")
    hammer_members = []
    for name, digest in sorted(rows.items()):
        path = M1216_REL / name
        local = ROOT / path
        hammer_members.append({"path": path.as_posix(), "size_bytes": local.stat().st_size,
                               "sha256": digest})
    for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
        path = ROOT / M1216_REL / name
        hammer_members.append({"path": (M1216_REL / name).as_posix(),
                               "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return review, hammer_members


def validate_forensic() -> None:
    rows = canonical_verify_double_seal(ROOT / FORENSIC_REL)
    require(rows.get("review.json") ==
            "1bc5af3d81d1cc1ee8dd7a91871ba9d31edc31cc442c256d96e23bc1bd828e65" and
            sha256(ROOT / FORENSIC_REL / "SHA256SUMS") ==
            "a4cb9a3da26224b4f75bd2b3ea857c262155a81a094eb977e981ce60ea77cacb" and
            sha256(ROOT / FORENSIC_REL / "SHA256SUMS.seal.sha256") ==
            "a7868aec49fb721f495e40d80d98f0d5525391ea807608a1363c04465373f995",
            "M1215 forensic identity drift")
    review = strict_json(ROOT / FORENSIC_REL / "review.json")
    require(review.get("status") ==
            "PASS_FORENSIC__LOCAL_M1210_CONSUMED__REMOTE_M1208_UNCONSUMED__STATUS_MISMATCH_REPRODUCED" and
            review.get("successor_boundary", {}).get("remote_m1208_namespace_is_still_fresh") is True,
            "M1215 forensic semantic drift")


def make_archive(path: Path, rows: list[dict[str, Any]]) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for row in rows:
            source = ROOT / repo_relative(row["path"])
            info = archive.gettarinfo(str(source), arcname=row["path"])
            require(info.isfile() and not info.issym() and not info.islnk(), "unsafe archive member")
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mode = 0o444; info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)


REMOTE_HELPER = r'''
import base64,hashlib,json,os,pathlib,stat,sys,tarfile
repo=pathlib.Path(sys.argv[1]); temp=pathlib.Path(sys.argv[2]); archive=temp/sys.argv[3]
plan=json.loads(base64.b64decode(sys.argv[4]).decode('utf-8'))
def die(msg): raise SystemExit('M1215_REMOTE_FAIL:'+msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
st=temp.lstat()
if not stat.S_ISDIR(st.st_mode) or temp.is_symlink() or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700: die('temp')
rst=repo.lstat()
if not stat.S_ISDIR(rst.st_mode) or repo.is_symlink(): die('repo')
st=archive.lstat()
if not stat.S_ISREG(st.st_mode) or archive.is_symlink() or archive.stat().st_size!=plan['archive_size'] or sha(archive)!=plan['archive_sha256']: die('archive')
for row in plan['old_dependencies']:
 p=repo/pathlib.Path(row['path'])
 try: st=p.lstat()
 except FileNotFoundError: die('old_missing:'+row['path'])
 if not stat.S_ISREG(st.st_mode) or p.is_symlink() or p.stat().st_size!=row['size_bytes'] or sha(p)!=row['sha256']: die('old_drift:'+row['path'])
for rel,token in [(plan['m1180_attempt'],'M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n')]:
 p=repo/pathlib.Path(rel)
 if not p.is_file() or p.is_symlink() or p.read_text(encoding='ascii')!=token: die('m1180_attempt')
for rel in [plan['m1180_result'],plan['m1180_log'],plan['m1208_attempt'],plan['m1208_result'],plan['m1208_log']]:
 if os.path.lexists(repo/pathlib.Path(rel)): die('namespace:'+rel)
stage=temp/'stage'; stage.mkdir(mode=0o700)
with tarfile.open(archive,'r:') as tf:
 members=tf.getmembers()
 if [m.name for m in members]!=[r['path'] for r in plan['members']]: die('members')
 for m,row in zip(members,plan['members']):
  if not m.isfile() or m.issym() or m.islnk() or m.size!=row['size_bytes']: die('member_type')
  out=stage/pathlib.Path(m.name); out.parent.mkdir(parents=True,exist_ok=True)
  src=tf.extractfile(m); fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444)
  h=hashlib.sha256(); n=0
  with os.fdopen(fd,'wb') as dst:
   while True:
    b=src.read(1<<20)
    if not b: break
    n+=len(b); h.update(b); dst.write(b)
   dst.flush(); os.fsync(dst.fileno())
  if n!=row['size_bytes'] or h.hexdigest()!=row['sha256']: die('member_sha')
for row in plan['members']:
 rel=pathlib.Path(row['path']); src=stage/rel; dst=repo/rel; cursor=repo
 for part in rel.parts[:-1]:
  cursor=cursor/part
  try: pst=cursor.lstat()
  except FileNotFoundError:
   cursor.mkdir(mode=0o755); pst=cursor.lstat()
  if not stat.S_ISDIR(pst.st_mode) or cursor.is_symlink(): die('unsafe_parent:'+row['path'])
 try: st=dst.lstat()
 except FileNotFoundError: st=None
 if st is not None:
  if not stat.S_ISREG(st.st_mode) or dst.is_symlink() or dst.stat().st_size!=row['size_bytes'] or sha(dst)!=row['sha256']: die('target_drift:'+row['path'])
  continue
 tmp=dst.parent/('.'+dst.name+'.m1215.publish.tmp')
 if os.path.lexists(tmp): die('publish_tmp')
 fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444)
 with src.open('rb') as s,os.fdopen(fd,'wb') as d:
  while True:
   b=s.read(1<<20)
   if not b: break
   d.write(b)
  d.flush(); os.fsync(d.fileno())
 if tmp.stat().st_size!=row['size_bytes'] or sha(tmp)!=row['sha256']: die('publish_sha')
 os.replace(tmp,dst)
 if dst.stat().st_size!=row['size_bytes'] or sha(dst)!=row['sha256']: die('published')
print('PASS_M1215_REMOTE_EXACT_TRANSFER_PREFLIGHT')
'''


def command_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, **kwargs)


def execute_once(runner: Callable[..., subprocess.CompletedProcess[str]] = command_run) -> None:
    source_contract, inventory, fixed = load_release()
    regular(M1210_FAILED_ATTEMPT, "immutable failed M1210 attempt")
    require(sha256(M1210_FAILED_ATTEMPT) == M1210_FAILED_SHA and
            M1210_FAILED_ATTEMPT.read_text(encoding="ascii") == M1210_FAILED_TOKEN,
            "M1210 failure marker drift; retry of M1210 forbidden")
    validate_forensic()
    _, hammer = load_m1216(source_contract)
    members = fixed + hammer
    require(len({row["path"] for row in members}) == len(members), "duplicate transfer path")
    require(not os.path.lexists(LOCAL_ATTEMPT), "M1215 local attempt already consumed")
    with tempfile.TemporaryDirectory(prefix="m1215_m1208_local.") as local_name:
        archive = Path(local_name) / REMOTE_ARCHIVE_BASENAME
        make_archive(archive, members)
        remote_mktemp = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                                REMOTE_HOST, shlex.join(
                                    ["/usr/bin/mktemp", "-d", REMOTE_TEMP_TEMPLATE])])
        require(remote_mktemp.returncode == 0 and remote_mktemp.stdout.count("\n") == 1,
                "remote mktemp failed")
        remote_temp_text = remote_mktemp.stdout.rstrip("\n")
        require(REMOTE_TEMP_RE.fullmatch(remote_temp_text) is not None, "remote temp path drift")
        remote_archive = remote_temp_text + "/" + REMOTE_ARCHIVE_BASENAME
        copied = runner(["/usr/bin/scp", "-P", REMOTE_PORT,
                         "-o", "ControlPath=" + SSH_CONTROL_PATH,
                         str(archive), REMOTE_HOST + ":" + remote_archive])
        require(copied.returncode == 0, "SCP failed")
        old = strict_json(ROOT / OLD_INVENTORY_REL)["dependencies"]
        plan = {"archive_size": archive.stat().st_size, "archive_sha256": sha256(archive),
                "members": members, "old_dependencies": old,
                "m1180_attempt": M1180_ATTEMPT_REL.as_posix(),
                "m1180_result": M1180_RESULT_REL.as_posix(), "m1180_log": M1180_LOG_REL.as_posix(),
                "m1208_attempt": M1208_ATTEMPT_REL.as_posix(),
                "m1208_result": M1208_RESULT_REL.as_posix(), "m1208_log": M1208_LOG_REL.as_posix()}
        encoded = base64.b64encode(json.dumps(plan, separators=(",", ":")).encode()).decode()
        checked_command = shlex.join(
            [REMOTE_INTERPRETER, "-c", REMOTE_HELPER, str(REMOTE_REPO),
             remote_temp_text, REMOTE_ARCHIVE_BASENAME, encoded])
        checked = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                          REMOTE_HOST, checked_command])
        require(checked.returncode == 0 and
                checked.stdout.count("PASS_M1215_REMOTE_EXACT_TRANSFER_PREFLIGHT") == 1,
                "remote exact transfer/preflight failed")
        descriptor = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, b"M1215_SUCCESSOR_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        launch_code = ("import os,runpy; os.chdir(" + repr(str(REMOTE_REPO)) + "); "
                       "runpy.run_path(" + repr(str(REMOTE_REPO / LAUNCHER_REL)) + ",run_name='__main__')")
        launched = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                           REMOTE_HOST, shlex.join([REMOTE_INTERPRETER, "-c", launch_code])])
        require(launched.returncode == 0, "single M1208 remote launch failed; no retry authorized")
        require(launched.stdout.count("PASS_M1208_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED") == 1,
                "M1208 remote terminal token mismatch")
        sys.stdout.write(launched.stdout)
        print(PASS_TOKEN)


def main() -> int:
    require(len(sys.argv) == 1, "zero-argument production wrapper required")
    execute_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

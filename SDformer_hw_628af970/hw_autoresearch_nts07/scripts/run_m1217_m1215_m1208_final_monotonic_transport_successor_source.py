#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Inert final monotonic transport successor for the unchanged M1215 launcher."""
from __future__ import annotations

import ast
import base64
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
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1217_m1215_m1208_final_monotonic_transport_successor_source.py")
SOURCE_CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1217_m1215_m1208_final_monotonic_transport_source_contract_r1_20260830.json")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1217_m1215_m1208_final_monotonic_transport_successor_source.py")
LAUNCHER_REL = Path("hw_autoresearch_nts07/scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py")
INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1217_m1215_m1208_final_monotonic_transport_inventory_r1_20260830.json")
ROOTS_REL = Path("hw_autoresearch_nts07/contracts/m1217_m1215_m1208_final_monotonic_transport_roots_r1_20260830.txt")
OLD_INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
M1210_INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1210_m1208_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
DEPENDENCY_AUDIT_REL = Path("hw_autoresearch_nts07/reviews/m1217_m1215_m1208_remote_dependency_read_only_audit_r1_20260830")
AUTHOR_REL = Path("hw_autoresearch_nts07/reviews/m1217_m1215_m1208_final_monotonic_transport_author_r1_20260830")
M1218_REL = Path("hw_autoresearch_nts07/reviews/m1218_m1217_m1215_m1208_final_monotonic_transport_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
REMOTE_TEMP_TEMPLATE = "/tmp/m1217_m1208.XXXXXXXXXXXX"
REMOTE_TEMP_RE = re.compile(r"\A/tmp/m1217_m1208\.[A-Za-z0-9]{12}\Z")
REMOTE_ARCHIVE_BASENAME = "exact_requirement_closure.tar"
LOCAL_ATTEMPT = HW / "results/.m1217_m1215_final_monotonic_transport_and_launch_r1_attempt_consumed"
M1210_ATTEMPT = HW / "results/.m1210_m1208_secure_transfer_and_launch_r1_attempt_consumed"
M1210_TOKEN = "M1210_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n"
M1210_SHA = "b60af667912eae9f19fb93aaf201fc342cfdd22e9add4bfeac0e55c09268e5f6"
M1215_ATTEMPT = HW / "results/.m1215_m1208_successor_secure_transfer_and_launch_r1_attempt_consumed"
M1215_TOKEN = "M1215_SUCCESSOR_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n"
M1215_SHA = "805a84ae541cf9572f1059fd7eeac1cbdc94e331e4f6dd384ce5bab9cc41049c"
M1180_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1180_RESULT_REL = Path("hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830")
M1180_LOG_REL = Path("hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
M1208_ATTEMPT_REL = Path("hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed")
M1208_RESULT_REL = Path("hw_autoresearch_nts07/results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830")
M1208_LOG_REL = Path("hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log")
M1180_TOKEN = "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
M1218_SCHEMA = "m1218_m1217_m1215_m1208_final_monotonic_transport_hammer_r1_v1"
M1218_STATUS = "PASS_M1217_FINAL_MONOTONIC_REQUIREMENT_CLOSURE_AND_ONE_M1215_LAUNCH_AUTHORIZED"
PASS_TOKEN = "PASS_M1217_REQUIREMENT_CLOSURE_AND_M1215_ONE_SHOT_LAUNCH__RESULT_HAMMER_REQUIRED"

EXACT_REL_NAMES = {
    "CAPTURE_REL", "SOURCE_CONTRACT_REL", "TEST_REL", "LAUNCH_CONTRACT_REL",
    "SOURCE_HAMMER_REL", "RELEASE_HAMMER_REL", "SUCCESSOR_HAMMER_REL",
    "FORENSIC_REL", "DOCS359_REL",
}
ABSENT_REL_NAMES = {"ATTEMPT_REL", "RESULT_REL", "LOG_REL"}
OPTIONAL_REL_NAMES = {"LEASE_REL"}


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
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " must be regular")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ReleaseError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == text, "unsafe repository-relative path")
    return path


def row_for(relative: str) -> dict[str, Any]:
    path = ROOT / repo_relative(relative)
    regular(path, "row source")
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical_expand(directory: Path) -> list[dict[str, Any]]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory")
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest"); regular(outer, "outer seal")
    parts = outer.read_text(encoding="ascii").split()
    require(parts == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        rel = repo_relative(name)
        require(name not in listed and len(rel.parts) == 1 and
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "unsafe sealed member")
        member = directory / rel
        regular(member, "sealed member")
        require(sha256(member) == digest, "sealed member drift")
        listed[name] = digest
    actual = {p.name for p in directory.iterdir() if p.is_file() and not p.is_symlink()
              and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(listed), "sealed membership drift")
    paths = [p.relative_to(ROOT).as_posix() for p in directory.iterdir()
             if p.is_file() and not p.is_symlink()]
    return [row_for(path) for path in sorted(paths)]


def launcher_rel_constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 \
                or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.endswith("_REL"):
            continue
        require(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Path" and len(node.value.args) == 1,
                "nonliteral launcher REL " + name)
        try:
            value = ast.literal_eval(node.value.args[0])
        except Exception as exc:
            raise ReleaseError("nonconstant launcher REL " + name) from exc
        require(isinstance(value, str), "launcher REL not string")
        repo_relative(value)
        found[name] = value
    require(set(found) == EXACT_REL_NAMES | ABSENT_REL_NAMES | OPTIONAL_REL_NAMES,
            "launcher REL population drift")
    return found


def expand_transfer_roots(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    roots = inventory.get("transfer_roots")
    require(isinstance(roots, list) and len(roots) == 9, "transfer root population")
    expected_lines = [row["kind"] + "  " + row["path"] for row in roots]
    require((ROOT / ROOTS_REL).read_text(encoding="utf-8").splitlines() == expected_lines,
            "root list drift")
    expanded: list[dict[str, Any]] = []
    for root in roots:
        path = ROOT / repo_relative(root["path"])
        if root["kind"] == "file":
            row = row_for(root["path"])
            require(row["size_bytes"] == root["size_bytes"] and
                    row["sha256"] == root["sha256"], "file root drift")
            expanded.append(row)
        else:
            require(root["kind"] == "sealed_dir", "unknown root kind")
            rows = canonical_expand(path)
            require(len(rows) == root["expanded_file_count"] and
                    sha256(path / "review.json") == root["review_sha256"] and
                    sha256(path / "SHA256SUMS") == root["manifest_sha256"] and
                    sha256(path / "SHA256SUMS.seal.sha256") == root["outer_seal_file_sha256"],
                    "sealed root identity drift")
            expanded.extend(rows)
    require(len(expanded) == 40 and len({row["path"] for row in expanded}) == 40,
            "expanded transfer closure")
    return sorted(expanded, key=lambda row: row["path"])


def validate_launcher_coverage(inventory: dict[str, Any], expanded: list[dict[str, Any]],
                               old_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rels = launcher_rel_constants(ROOT / LAUNCHER_REL)
    frozen = inventory["launcher_required_rel"]
    require(frozen["all_rel_constant_count"] == 13 and
            frozen["exact_prerequisite_constants"] ==
            {name: rels[name] for name in EXACT_REL_NAMES} and
            frozen["runtime_absent_constants"] ==
            {name: rels[name] for name in ABSENT_REL_NAMES} and
            frozen["optional_runtime_state_constants"] ==
            {name: rels[name] for name in OPTIONAL_REL_NAMES},
            "inventory does not freeze every launcher REL")
    old = {row["path"]: row for row in old_rows}
    new = {row["path"]: row for row in expanded}
    require(not (set(old) & set(new)), "old/new coverage overlap")
    required: dict[str, dict[str, Any]] = {}
    sealed_roots: list[dict[str, Any]] = []
    for name in sorted(EXACT_REL_NAMES):
        relative = rels[name]; path = ROOT / relative
        if path.is_file():
            rows = [row_for(relative)]
        else:
            rows = canonical_expand(path)
            sealed_roots.append({"constant": name, "path": relative,
                                 "member_paths": [row["path"] for row in rows]})
        for row in rows:
            require(row["path"] not in required, "duplicate required file")
            required[row["path"]] = row
    launcher_row = row_for(LAUNCHER_REL.as_posix())
    required[launcher_row["path"]] = launcher_row
    for path, row in required.items():
        authority = old.get(path, new.get(path))
        require(authority is not None and
                {key: authority[key] for key in ("path", "size_bytes", "sha256")} == row,
                "uncovered/drifted launcher prerequisite " + path)
    require(set(new) == set(required) - set(old), "new package has missing/extra rows")
    require(len(required) == 41 and len(set(required) & set(old)) == 1 and len(new) == 40,
            "coverage cardinality")
    forensic = rels["FORENSIC_REL"]
    require(sum(path == forensic or path.startswith(forensic + "/") for path in new) == 9,
            "complete forensic double seal not transferred")
    return {"required_exact": [required[path] for path in sorted(required)],
            "sealed_roots": sealed_roots,
            "runtime_absent": [rels[name] for name in sorted(ABSENT_REL_NAMES)]}


def build_remote_authority(inventory: dict[str, Any], expanded: list[dict[str, Any]],
                           old_rows: list[dict[str, Any]],
                           m1210_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct the exact 143-file authority used by the read-only audit."""
    merged: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        frozen = {key: row[key] for key in ("path", "size_bytes", "sha256")}
        current = merged.get(frozen["path"])
        require(current is None or current == frozen,
                "conflicting remote authority " + frozen["path"])
        merged[frozen["path"]] = frozen

    for row in old_rows:
        add(row)
    for row in m1210_rows:
        add(row)
    roots = inventory["transfer_roots"]
    sealed_prefixes = [row["path"] + "/" for row in roots if row["kind"] == "sealed_dir"]
    for row in expanded:
        if any(row["path"].startswith(prefix) for prefix in sealed_prefixes):
            add(row)
    add(row_for(LAUNCHER_REL.as_posix()))
    add(row_for(DOCS359_REL.as_posix()))
    require(len(merged) == 143, "remote authority cardinality")
    forensic_prefix = inventory["launcher_required_rel"]["exact_prerequisite_constants"]["FORENSIC_REL"] + "/"
    require(sum(path.startswith(forensic_prefix) for path in merged) == 9,
            "remote authority forensic population")
    frozen = inventory["remote_dependency_authority"]
    require(frozen == {"unique_file_count": 143, "read_only_exact_before_publish": 134,
                       "read_only_missing_before_publish": 9, "read_only_drift": 0,
                       "post_publish_exact_required": 143},
            "remote authority audit cardinality drift")
    return [merged[path] for path in sorted(merged)]


def verify_marker(path: Path, digest: str, token: str, label: str) -> None:
    regular(path, label)
    require(sha256(path) == digest and path.read_text(encoding="ascii") == token and
            stat.S_IMODE(path.stat().st_mode) == 0o400,
            label + " identity drift")


def verify_sealed_authority(root: Path) -> dict[str, Any]:
    rows = canonical_expand(root)
    require(any(row["path"].endswith("/review.json") for row in rows), "authority review absent")
    return strict_json(root / "review.json")


def load_release() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]],
                            dict[str, Any], list[dict[str, Any]]]:
    contract = strict_json(ROOT / SOURCE_CONTRACT_REL)
    require(contract.get("schema") ==
            "m1217_m1215_m1208_final_monotonic_transport_source_contract_r1_v1" and
            contract.get("status") ==
            "INERT_SOURCE_ONLY__FRESH_M1218_HAMMER_REQUIRED__NO_REMOTE_NO_GPU",
            "source contract semantic drift")
    inventory = strict_json(ROOT / INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1217_m1215_m1208_final_monotonic_transport_inventory_r1_v1" and
            inventory.get("status") ==
            "EXACT_LAUNCHER_REQUIREMENT_CLOSURE__ABSENT_OR_EXACT_PACKAGE",
            "inventory semantic drift")
    require(contract["source"]["sha256"] == sha256(Path(__file__).resolve()) and
            contract["test"]["sha256"] == sha256(ROOT / TEST_REL) and
            contract["inventory"]["sha256"] == sha256(ROOT / INVENTORY_REL) and
            contract["roots"]["sha256"] == sha256(ROOT / ROOTS_REL) and
            contract["m1215_launcher"]["sha256"] == sha256(ROOT / LAUNCHER_REL),
            "source contract binding mismatch")
    audit_root = ROOT / DEPENDENCY_AUDIT_REL
    audit = verify_sealed_authority(audit_root)
    audit_binding = contract["remote_dependency_read_only_audit"]
    require(audit.get("status") ==
            "PASS_AUDIT__M1215_CONSUMED__REMOTE_M1208_FRESH__FORENSIC9_ONLY_MISSING" and
            audit_binding == {
                "path": DEPENDENCY_AUDIT_REL.as_posix(),
                "review_sha256": sha256(audit_root / "review.json"),
                "manifest_sha256": sha256(audit_root / "SHA256SUMS"),
                "outer_seal_file_sha256": sha256(audit_root / "SHA256SUMS.seal.sha256"),
                "authority_unique_files": 143, "remote_exact_files": 134,
                "remote_missing_files": 9, "remote_drift_files": 0},
            "dependency audit binding mismatch")
    old_path = ROOT / OLD_INVENTORY_REL
    old = strict_json(old_path).get("dependencies")
    require(isinstance(old, list) and len(old) == 95 and
            inventory["old_dependency_inventory"] == {
                "path": OLD_INVENTORY_REL.as_posix(), "size_bytes": old_path.stat().st_size,
                "sha256": sha256(old_path), "row_count": 95}, "old inventory authority")
    expanded = expand_transfer_roots(inventory)
    closure = validate_launcher_coverage(inventory, expanded, old)
    m1210_path = ROOT / M1210_INVENTORY_REL
    m1210 = strict_json(m1210_path).get("transfer_required")
    require(isinstance(m1210, list) and len(m1210) == 21 and
            inventory["m1210_transfer_inventory"] == {
                "path": M1210_INVENTORY_REL.as_posix(), "size_bytes": m1210_path.stat().st_size,
                "sha256": sha256(m1210_path), "row_count": 21},
            "M1210 transfer inventory authority")
    authority = build_remote_authority(inventory, expanded, old, m1210)
    return contract, inventory, expanded, closure, authority


def load_m1218() -> dict[str, Any]:
    review = verify_sealed_authority(ROOT / M1218_REL)
    require(review.get("schema") == M1218_SCHEMA and review.get("status") == M1218_STATUS and
            review.get("verdict") == "GO" and isinstance(review.get("score"), int) and
            review["score"] >= 95 and review.get("p0_count") == 0 and review.get("p1_count") == 0,
            "M1218 semantic admission")
    bindings = review.get("bindings", {})
    require(bindings.get("source_sha256") == sha256(Path(__file__).resolve()) and
            bindings.get("source_contract_sha256") == sha256(ROOT / SOURCE_CONTRACT_REL) and
            bindings.get("inventory_sha256") == sha256(ROOT / INVENTORY_REL) and
            bindings.get("roots_sha256") == sha256(ROOT / ROOTS_REL) and
            bindings.get("author_manifest_sha256") == sha256(ROOT / AUTHOR_REL / "SHA256SUMS") and
            bindings.get("author_outer_file_sha256") == sha256(ROOT / AUTHOR_REL / "SHA256SUMS.seal.sha256") and
            bindings.get("m1215_launcher_sha256") == sha256(ROOT / LAUNCHER_REL) and
            bindings.get("m1210_marker_sha256") == M1210_SHA and
            bindings.get("m1215_marker_sha256") == M1215_SHA and
            bindings.get("dependency_audit_review_sha256") ==
            sha256(ROOT / DEPENDENCY_AUDIT_REL / "review.json") and
            bindings.get("dependency_audit_manifest_sha256") ==
            sha256(ROOT / DEPENDENCY_AUDIT_REL / "SHA256SUMS") and
            bindings.get("dependency_audit_outer_file_sha256") ==
            sha256(ROOT / DEPENDENCY_AUDIT_REL / "SHA256SUMS.seal.sha256"),
            "M1218 exact binding mismatch")
    auth = review.get("authorization", {})
    require(auth.get("secure_transfer") is True and auth.get("exact_remote_launch") is True and
            auth.get("launch_count") == 1 and auth.get("automatic_retry") is False,
            "M1218 authorization mismatch")
    return review


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
def die(msg): raise SystemExit('M1217_REMOTE_FAIL:'+msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def exact(p,row):
 try: st=p.lstat()
 except FileNotFoundError:return False
 if not stat.S_ISREG(st.st_mode) or p.is_symlink() or p.stat().st_size!=row['size_bytes'] or sha(p)!=row['sha256']:die('drift:'+row['path'])
 return True
st=temp.lstat()
if not stat.S_ISDIR(st.st_mode) or temp.is_symlink() or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700:die('temp')
if not stat.S_ISDIR(repo.lstat().st_mode) or repo.is_symlink():die('repo')
if not exact(archive,{'path':'archive','size_bytes':plan['archive_size'],'sha256':plan['archive_sha256']}):die('archive')
for row in plan['old_dependencies']:
 if not exact(repo/pathlib.Path(row['path']),row):die('old_missing:'+row['path'])
p=repo/pathlib.Path(plan['m1180_attempt'])
if not p.is_file() or p.is_symlink() or p.read_text(encoding='ascii')!='M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n':die('m1180_attempt')
for rel in plan['runtime_absent']:
 if os.path.lexists(repo/pathlib.Path(rel)):die('runtime_namespace:'+rel)
with tarfile.open(archive,'r:') as tf:
 members=tf.getmembers()
 if [m.name for m in members]!=[r['path'] for r in plan['members']]:die('members')
 stage=temp/'stage';stage.mkdir(mode=0o700)
 for m,row in zip(members,plan['members']):
  if not m.isfile() or m.issym() or m.islnk() or m.size!=row['size_bytes']:die('member_type')
  out=stage/pathlib.Path(m.name);out.parent.mkdir(parents=True,exist_ok=True)
  src=tf.extractfile(m);fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444);h=hashlib.sha256();n=0
  with os.fdopen(fd,'wb') as dst:
   while True:
    b=src.read(1<<20)
    if not b:break
    n+=len(b);h.update(b);dst.write(b)
   dst.flush();os.fsync(dst.fileno())
  if n!=row['size_bytes'] or h.hexdigest()!=row['sha256']:die('member_sha')
# Validate every canonical destination before the first publication.
for row in plan['members']:
 p=repo/pathlib.Path(row['path'])
 if os.path.lexists(p):exact(p,row)
for row in plan['members']:
 rel=pathlib.Path(row['path']);src=stage/rel;dst=repo/rel;cursor=repo
 for part in rel.parts[:-1]:
  cursor=cursor/part
  try:pst=cursor.lstat()
  except FileNotFoundError:cursor.mkdir(mode=0o755);pst=cursor.lstat()
  if not stat.S_ISDIR(pst.st_mode) or cursor.is_symlink():die('unsafe_parent:'+row['path'])
 if exact(dst,row):continue
 tmp=dst.parent/('.'+dst.name+'.m1217.publish.tmp')
 fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444)
 with src.open('rb') as s,os.fdopen(fd,'wb') as d:
  while True:
   b=s.read(1<<20)
   if not b:break
   d.write(b)
  d.flush();os.fsync(d.fileno())
 if tmp.stat().st_size!=row['size_bytes'] or sha(tmp)!=row['sha256']:die('publish_tmp_sha')
 try:os.link(tmp,dst)
 except FileExistsError:exact(dst,row)
 os.unlink(tmp)
 exact(dst,row)
exact_count=0
for row in plan['post_publish_authority']:
 if not exact(repo/pathlib.Path(row['path']),row):die('authority_missing:'+row['path'])
 exact_count+=1
if exact_count!=143:die('authority_count')
for sealed in plan['sealed_roots']:
 root=repo/pathlib.Path(sealed['path']);manifest=root/'SHA256SUMS';outer=root/'SHA256SUMS.seal.sha256'
 if not manifest.is_file() or manifest.is_symlink() or not outer.is_file() or outer.is_symlink():die('seal_files')
 if outer.read_text(encoding='ascii').split()!=[sha(manifest),'SHA256SUMS']:die('outer_seal')
 listed=[]
 for line in manifest.read_text(encoding='ascii').splitlines():listed.append(line.split(None,1)[1].lstrip('*'))
 actual=sorted([p.name for p in root.iterdir() if p.is_file() and not p.is_symlink() and p.name not in {'SHA256SUMS','SHA256SUMS.seal.sha256'}])
 if sorted(listed)!=actual:die('seal_membership')
 if sorted([str((root/p).relative_to(repo)) for p in root.iterdir() if p.is_file() and not p.is_symlink()])!=sorted(sealed['member_paths']):die('sealed_plan')
for rel in plan['runtime_absent']:
 if os.path.lexists(repo/pathlib.Path(rel)):die('runtime_changed:'+rel)
print('PASS_M1217_REMOTE_REQUIREMENT_CLOSURE_PREFLIGHT__POST_VERIFY_143_EXACT')
'''


def command_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, **kwargs)


def execute_once(runner: Callable[..., subprocess.CompletedProcess[str]] = command_run) -> None:
    _, _, members, closure, authority = load_release()
    verify_marker(M1210_ATTEMPT, M1210_SHA, M1210_TOKEN, "consumed M1210 marker")
    verify_marker(M1215_ATTEMPT, M1215_SHA, M1215_TOKEN, "consumed M1215 marker")
    load_m1218()
    require(not os.path.lexists(LOCAL_ATTEMPT), "M1217 local attempt already consumed")
    with tempfile.TemporaryDirectory(prefix="m1217_m1208_local.") as local_name:
        archive = Path(local_name) / REMOTE_ARCHIVE_BASENAME
        make_archive(archive, members)
        made = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                       REMOTE_HOST, shlex.join(["/usr/bin/mktemp", "-d", REMOTE_TEMP_TEMPLATE])])
        require(made.returncode == 0 and made.stdout.count("\n") == 1, "remote mktemp failed")
        remote_temp = made.stdout.rstrip("\n")
        require(REMOTE_TEMP_RE.fullmatch(remote_temp) is not None, "remote temp path drift")
        remote_archive = remote_temp + "/" + REMOTE_ARCHIVE_BASENAME
        copied = runner(["/usr/bin/scp", "-P", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
                         str(archive), REMOTE_HOST + ":" + remote_archive])
        require(copied.returncode == 0, "SCP failed")
        old = strict_json(ROOT / OLD_INVENTORY_REL)["dependencies"]
        plan = {"archive_size": archive.stat().st_size, "archive_sha256": sha256(archive),
                "members": members, "old_dependencies": old,
                "required_exact": closure["required_exact"],
                "post_publish_authority": authority,
                "sealed_roots": closure["sealed_roots"],
                "runtime_absent": closure["runtime_absent"],
                "m1180_attempt": M1180_ATTEMPT_REL.as_posix()}
        encoded = base64.b64encode(json.dumps(plan, separators=(",", ":")).encode()).decode()
        checked = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                          REMOTE_HOST, shlex.join([REMOTE_INTERPRETER, "-c", REMOTE_HELPER,
                          str(REMOTE_REPO), remote_temp, REMOTE_ARCHIVE_BASENAME, encoded])])
        require(checked.returncode == 0 and
                checked.stdout.count(
                    "PASS_M1217_REMOTE_REQUIREMENT_CLOSURE_PREFLIGHT__POST_VERIFY_143_EXACT") == 1,
                "remote requirement closure failed")
        descriptor = os.open(LOCAL_ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor,
                     b"M1217_REQUIREMENT_CLOSURE_COMPLETE__M1215_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        launch_code = ("import os,runpy; os.chdir(" + repr(str(REMOTE_REPO)) + "); "
                       "runpy.run_path(" + repr(str(REMOTE_REPO / LAUNCHER_REL)) + ",run_name='__main__')")
        launched = runner(["/usr/bin/ssh", "-S", SSH_CONTROL_PATH, "-p", REMOTE_PORT,
                           REMOTE_HOST, shlex.join([REMOTE_INTERPRETER, "-c", launch_code])])
        require(launched.returncode == 0, "single M1215 remote launch failed; no retry authorized")
        require(launched.stdout.count("PASS_M1208_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED") == 1,
                "M1215 remote terminal token mismatch")
        sys.stdout.write(launched.stdout)
        print(PASS_TOKEN)


def main() -> int:
    require(len(sys.argv) == 1, "zero-argument production wrapper required")
    execute_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

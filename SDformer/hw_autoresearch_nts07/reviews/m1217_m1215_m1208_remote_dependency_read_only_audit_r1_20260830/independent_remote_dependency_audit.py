#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1217 audit of the M1215 successor remote prerequisites.

This program performs no remote publication and does not launch capture.  It
compares the authoritative local size/SHA identities with the existing remote
repository and reports the unique missing/drift population plus namespace
state.
"""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REMOTE_REPO = "/root/private_data/work/sdformer_codex/SDformer"
REMOTE_PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
SSH = ["/usr/bin/ssh", "-S", "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock",
       "-p", "10037", "root@ssh.sd5ai.scnet.cn"]
OLD_INVENTORY = HW / "contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json"
M1210_INVENTORY = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json"
SUCCESSOR_LAUNCHER = HW / "scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
SEALED_DIRS = [
    HW / "reviews/m1209_m1208_motion_ep29_unified_capture_source_hammer_r1_20260830",
    HW / "reviews/m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_20260830",
    HW / "reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830",
    HW / "reviews/m1216_m1215_m1208_motion_ep29_unified_capture_successor_release_hammer_r1_20260830",
]
M1180_ATTEMPT = "hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
M1180_TOKEN = "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
M1180_RESULT = "hw_autoresearch_nts07/results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830"
M1180_LOG = "hw_autoresearch_nts07/results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log"
M1208_ATTEMPT = "hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
M1208_RESULT = "hw_autoresearch_nts07/results/m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830"
M1208_LOG = "hw_autoresearch_nts07/results/.m1208_motion_ep29_unified_hardware_capture_s40_r1_20260830.production.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise RuntimeError("nonregular local authority: " + str(path))


def strict_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON root is not object: " + str(path))
    return value


def verify_seal(root: Path) -> list[dict]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    parts = outer.read_text(encoding="ascii").split()
    if len(parts) != 2 or parts[1] != "SHA256SUMS" or parts[0] != sha256(manifest):
        raise RuntimeError("outer seal mismatch: " + str(root))
    names: set[str] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        if (len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
                or "/" in fields[1] or fields[1] in names):
            raise RuntimeError("unsafe manifest: " + str(root))
        path = root / fields[1]
        regular(path)
        if sha256(path) != fields[0]:
            raise RuntimeError("member drift: " + str(path))
        names.add(fields[1])
    rows = []
    for path in sorted(root.iterdir()):
        regular(path)
        rel = path.relative_to(ROOT).as_posix()
        rows.append({"path": rel, "size_bytes": path.stat().st_size,
                     "sha256": sha256(path)})
    return rows


def add(rows: dict[str, dict], row: dict, source: str) -> None:
    candidate = {"path": row["path"], "size_bytes": row["size_bytes"],
                 "sha256": row["sha256"], "sources": [source]}
    old = rows.get(candidate["path"])
    if old is None:
        rows[candidate["path"]] = candidate
        return
    if (old["size_bytes"], old["sha256"]) != (candidate["size_bytes"], candidate["sha256"]):
        raise RuntimeError("conflicting local authorities: " + candidate["path"])
    old["sources"].append(source)


REMOTE_HELPER = r'''
import base64,hashlib,json,os,pathlib,stat,sys
repo=pathlib.Path(sys.argv[1]); plan=json.loads(base64.b64decode(sys.argv[2]).decode())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
out=[]
for row in plan['files']:
 p=repo/pathlib.Path(row['path'])
 try: st=p.lstat()
 except FileNotFoundError:
  out.append({'path':row['path'],'state':'MISSING'}); continue
 if not stat.S_ISREG(st.st_mode) or p.is_symlink():
  out.append({'path':row['path'],'state':'DRIFT_TYPE','remote_type':stat.S_IFMT(st.st_mode)}); continue
 size=p.stat().st_size
 if size!=row['size_bytes']:
  out.append({'path':row['path'],'state':'DRIFT_SIZE','remote_size_bytes':size}); continue
 digest=sha(p)
 out.append({'path':row['path'],'state':'EXACT' if digest==row['sha256'] else 'DRIFT_SHA','remote_sha256':digest})
p=repo/pathlib.Path(plan['m1180_attempt'])
try:
 st=p.lstat(); token=p.read_text(encoding='ascii') if stat.S_ISREG(st.st_mode) and not p.is_symlink() else None
 marker={'path':plan['m1180_attempt'],'state':'EXACT_TOKEN' if token==plan['m1180_token'] else 'DRIFT_TOKEN_OR_TYPE'}
except FileNotFoundError: marker={'path':plan['m1180_attempt'],'state':'MISSING'}
namespaces=[]
for rel in plan['must_absent']:
 namespaces.append({'path':rel,'state':'PRESENT' if os.path.lexists(repo/pathlib.Path(rel)) else 'ABSENT'})
print(json.dumps({'files':out,'m1180_marker':marker,'must_absent':namespaces},sort_keys=True,separators=(',',':')))
'''


def main() -> int:
    rows: dict[str, dict] = {}
    for row in strict_json(OLD_INVENTORY)["dependencies"]:
        add(rows, row, "m1182_old_inventory")
    for row in strict_json(M1210_INVENTORY)["transfer_required"]:
        add(rows, row, "m1210_transfer_inventory")
    for root in SEALED_DIRS:
        for row in verify_seal(root):
            add(rows, row, "recursive_seal:" + root.name)
    for path, source in [(SUCCESSOR_LAUNCHER, "m1215_successor_launcher"),
                         (DOCS359, "docs359")]:
        regular(path)
        add(rows, {"path": path.relative_to(ROOT).as_posix(),
                   "size_bytes": path.stat().st_size, "sha256": sha256(path)}, source)
    files = sorted(rows.values(), key=lambda row: row["path"])
    plan = {"files": files, "m1180_attempt": M1180_ATTEMPT, "m1180_token": M1180_TOKEN,
            "must_absent": [M1180_RESULT, M1180_LOG, M1208_ATTEMPT, M1208_RESULT, M1208_LOG]}
    encoded = base64.b64encode(json.dumps(plan, separators=(",", ":")).encode()).decode()
    command = SSH + [shlex.join([REMOTE_PYTHON, "-c", REMOTE_HELPER, REMOTE_REPO, encoded])]
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError("read-only remote audit failed rc={}: {}".format(
            completed.returncode, completed.stderr.strip()))
    remote = json.loads(completed.stdout)
    by_path = {row["path"]: row for row in remote["files"]}
    findings = []
    for row in files:
        state = by_path[row["path"]]["state"]
        if state != "EXACT":
            findings.append({"path": row["path"], "state": state,
                             "sources": row["sources"],
                             "expected_size_bytes": row["size_bytes"],
                             "expected_sha256": row["sha256"]})
    namespace_findings = [row for row in remote["must_absent"] if row["state"] != "ABSENT"]
    output = {
        "schema": "m1217_m1215_m1208_remote_dependency_read_only_audit_r1_v1",
        "status": "PASS_READ_ONLY_AUDIT__NO_REMOTE_WRITE_NO_LAUNCH",
        "local_authority": {"unique_files": len(files), "old_inventory_rows": 95,
                            "m1210_transfer_rows": 21, "sealed_directories": len(SEALED_DIRS)},
        "remote": {"exact_files": sum(row["state"] == "EXACT" for row in remote["files"]),
                   "missing_or_drift_files": len(findings), "findings": findings,
                   "m1180_marker": remote["m1180_marker"],
                   "must_absent_findings": namespace_findings,
                   "must_absent_states": remote["must_absent"]},
        "launch_admission": (len(findings) == 0 and
                             remote["m1180_marker"]["state"] == "EXACT_TOKEN" and
                             len(namespace_findings) == 0),
        "claim_boundary": {"remote_read_only": True, "remote_write": False,
                           "launch": False, "capture": False, "gpu": False,
                           "eda": False},
        "docs359_sha256": sha256(DOCS359),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

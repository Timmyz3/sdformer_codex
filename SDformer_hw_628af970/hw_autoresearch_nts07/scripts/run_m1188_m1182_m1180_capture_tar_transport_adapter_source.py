#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Inert M1188 tar/SFTP transport contingency for the sealed M1180 capture.

This source is additive: it never rewrites the original 42-row transfer list,
95-row dependency inventory, M1182 release, or M1184 hammer.  It is unusable
until a fresh different-author M1189 hammer is supplied through exact digests.
The production path uses only fixed subprocess argv with ``shell=False``.
"""
from __future__ import annotations

import hashlib
import io
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
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1188_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_20260830.json")
ORIGINAL_LIST_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_transfer_files_r1_20260830.txt")
ORIGINAL_INVENTORY_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_20260830.json")
ORIGINAL_RELEASE_REL = Path("hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json")
M1184_REL = Path("hw_autoresearch_nts07/reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1189_m1188_m1182_m1180_capture_tar_transport_adapter_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = "/opt/conda/envs/sdformerflow/bin/python"
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock"
SSH = Path("/usr/bin/ssh")
SCP = Path("/usr/bin/scp")
REMOTE_ARCHIVE = Path("/tmp/m1188_m1180_exact51_transport_r1.tar")
REMOTE_STAGE = REMOTE_REPO / ".m1188_m1180_exact51_transport_stage_r1"
ATTEMPT = HW / "results/.m1188_m1180_exact51_transport_r1_attempt_consumed"
RESULT = HW / "results/m1188_m1180_exact51_transport_r1_20260830"
PASS_TOKEN = "PASS_M1188_EXACT42_PLUS_M1184_SEALS_TRANSFER__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


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


def strict_json_bytes(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           TransportError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    return strict_json_bytes(path.read_bytes())


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise TransportError("missing {}: {}".format(label, path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file".format(label))


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
            "unsafe repository-relative path: " + text)
    require(str(path) == path.as_posix(), "non-canonical repository-relative path")
    return path


def parse_sha_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ")
        require(len(fields) == 2 and len(fields[0]) == 64 and
                all(c in "0123456789abcdef" for c in fields[0]),
                "malformed SHA256SUMS row")
        name = fields[1]
        require(Path(name).name == name and name not in {".", ".."},
                "unsafe SHA256SUMS member")
        rows.append((name, fields[0]))
    require(len(rows) == 7 and len({name for name, _ in rows}) == 7,
            "M1184 inner manifest cardinality/uniqueness drift")
    return rows


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    expected_keys = {"schema", "status", "date", "source", "test", "claim_boundary",
                     "prior_rsync_failure", "original_authorities", "m1184_hammer",
                     "transport", "future_hammer", "docs359_sha256"}
    require(set(contract) == expected_keys, "contract exact keys drift")
    require(contract["schema"] == "m1188_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_v1" and
            contract["status"] == "INERT_SOURCE_ONLY__FRESH_M1189_HAMMER_REQUIRED__NO_REMOTE_NO_TRANSFER_NO_GPU",
            "contract schema/status drift")
    require(contract["date"] == "2026-08-30", "contract date drift")
    for label, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        row = contract[label]
        require(set(row) == {"path", "size_bytes", "sha256"} and row["path"] == str(rel),
                label + " identity drift")
        path = ROOT / rel
        regular(path, label)
        require(type(row["size_bytes"]) is int and path.stat().st_size == row["size_bytes"] and
                sha256(path) == row["sha256"], label + " byte identity drift")
    require(contract["claim_boundary"] == {
        "adapter_only": True, "modifies_original_42_or_inventory_or_release_or_hammer": False,
        "remote_or_transfer_executed_by_author": False, "gpu_or_capture_executed_by_author": False,
        "m1180_attempt_consumed_by_transport": False, "paper_result": False},
        "claim boundary drift")
    require(contract["prior_rsync_failure"] == {
        "stage": "REMOTE_BEFORE_TRANSFER", "cause": "REMOTE_RSYNC_EXECUTABLE_ABSENT",
        "bytes_transferred": 0, "remote_namespace_created": False,
        "m1180_attempt_consumed": False, "gpu_consumed": False,
        "automatic_retry_performed": False}, "prior rsync failure record drift")
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def exact_members(contract: dict[str, Any]) -> list[dict[str, Any]]:
    authorities = contract["original_authorities"]
    expected_authority_keys = {"transfer_list", "dependency_inventory", "launch_release",
                               "exact_transfer_row_count", "inventory_transfer_required_count"}
    require(set(authorities) == expected_authority_keys and
            authorities["exact_transfer_row_count"] == 42 and
            authorities["inventory_transfer_required_count"] == 40,
            "original authority cardinality drift")
    authority_paths = {
        "transfer_list": ORIGINAL_LIST_REL,
        "dependency_inventory": ORIGINAL_INVENTORY_REL,
        "launch_release": ORIGINAL_RELEASE_REL,
    }
    for key, rel in authority_paths.items():
        row = authorities[key]
        require(set(row) == {"path", "size_bytes", "sha256"} and row["path"] == str(rel),
                key + " authority path drift")
        path = ROOT / rel
        regular(path, key)
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                key + " authority bytes drift")

    lines = (ROOT / ORIGINAL_LIST_REL).read_text(encoding="utf-8").splitlines()
    require(len(lines) == 42 and len(set(lines)) == 42 and all(lines),
            "original exact42 list drift")
    inventory = strict_json(ROOT / ORIGINAL_INVENTORY_REL)
    require(inventory.get("schema") ==
            "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1" and
            inventory.get("status") == "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
            "original inventory semantics drift")
    transfer_rows = [row for row in inventory.get("dependencies", [])
                     if row.get("disposition") == "transfer_required"]
    require(len(transfer_rows) == 40, "original inventory transfer population drift")
    by_path = {row.get("path"): row for row in transfer_rows}
    require(len(by_path) == 40 and set(lines[:5] + lines[7:]) <= set(lines),
            "original transfer uniqueness drift")
    require(set(lines) - set(by_path) == {str(ORIGINAL_INVENTORY_REL), str(ORIGINAL_LIST_REL)},
            "exact42 must equal inventory exact40 plus inventory/list")

    members: list[dict[str, Any]] = []
    for text in lines:
        rel = repo_relative(text)
        path = ROOT / rel
        regular(path, "exact42 member")
        if text in by_path:
            source = by_path[text]
            require(path.stat().st_size == source["size_bytes"] and sha256(path) == source["sha256"],
                    "inventory member bytes drift: " + text)
            size, digest = source["size_bytes"], source["sha256"]
        else:
            size, digest = path.stat().st_size, sha256(path)
        members.append({"path": text, "size_bytes": size, "sha256": digest,
                        "class": "ORIGINAL_EXACT42"})

    hammer = contract["m1184_hammer"]
    require(set(hammer) == {"directory", "file_count", "manifest_sha256", "outer_sha256",
                            "required_schema", "required_status"} and
            hammer["directory"] == str(M1184_REL) and hammer["file_count"] == 9 and
            hammer["required_schema"] ==
            "m1184_m1182_m1180_motion_unified_capture_launch_release_hammer_r1_v1" and
            hammer["required_status"] ==
            "PASS_M1184_M1182_M1180_UNIFIED_CAPTURE_RELEASE_HAMMER__EXACT_TRANSFER_AND_ONE_LAUNCH_AUTHORIZED__NO_RETRY__RESULT_HAMMER_REQUIRED",
            "M1184 hammer contract drift")
    manifest = ROOT / M1184_REL / "SHA256SUMS"
    outer = ROOT / M1184_REL / "SHA256SUMS.seal.sha256"
    regular(manifest, "M1184 manifest")
    regular(outer, "M1184 outer seal")
    require(sha256(manifest) == hammer["manifest_sha256"] and
            sha256(outer) == hammer["outer_sha256"] and
            outer.read_text(encoding="utf-8") == hammer["manifest_sha256"] + "  SHA256SUMS\n",
            "M1184 recursive seal drift")
    manifest_rows = parse_sha_manifest(manifest)
    expected_names = {name for name, _ in manifest_rows} | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    actual_names = {path.name for path in (ROOT / M1184_REL).iterdir()}
    require(actual_names == expected_names, "M1184 directory exact file set drift")
    for name, digest in manifest_rows:
        path = ROOT / M1184_REL / name
        regular(path, "M1184 sealed member")
        require(sha256(path) == digest, "M1184 sealed member drift: " + name)
    for name in sorted(expected_names):
        path = ROOT / M1184_REL / name
        members.append({"path": str(M1184_REL / name), "size_bytes": path.stat().st_size,
                        "sha256": sha256(path), "class": "M1184_EXACT_SEAL"})
    require(len(members) == 51 and len({row["path"] for row in members}) == 51,
            "exact42+exact9 population drift")
    return members


def verify_transport_contract(contract: dict[str, Any]) -> None:
    transport = contract["transport"]
    require(transport == {
        "member_count": 51, "original_exact42_count": 42, "m1184_exact_seal_count": 9,
        "local_ssh": {"path": str(SSH), "size_bytes": 775656,
                      "sha256": "3cbb1eb62b4fec407778373e84105378c1860648f8817086aa4176da11e93a88"},
        "local_scp": {"path": str(SCP), "size_bytes": 105304,
                      "sha256": "35dc3481f433276e6071461500097c86dee5281fb7d64eed46bec8c79c45a666"},
        "remote_host": REMOTE_HOST, "remote_port": 10037,
        "ssh_control_path": SSH_CONTROL_PATH, "remote_repository": str(REMOTE_REPO),
        "remote_interpreter": REMOTE_INTERPRETER, "remote_python_version": "3.10.20",
        "remote_archive": str(REMOTE_ARCHIVE), "remote_stage": str(REMOTE_STAGE),
        "protocol": "LOCAL_TARFILE_PLUS_OPENSSH_SCP_DEFAULT_SFTP_PLUS_REMOTE_PYTHON_STDLIB_SAFE_EXTRACT",
        "shell": False, "fixed_argv": True, "path_escape_rejected": True,
        "symlink_rejected": True, "post_transfer_size_sha_each_member": True,
        "automatic_retry": False}, "transport contract drift")
    for tool in (transport["local_ssh"], transport["local_scp"]):
        path = Path(tool["path"])
        regular(path, "bound local transport executable")
        require(path.stat().st_size == tool["size_bytes"] and sha256(path) == tool["sha256"],
                "local transport executable identity drift")
    future = contract["future_hammer"]
    require(future == {
        "directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1189_m1188_m1182_m1180_capture_tar_transport_adapter_hammer_r1_v1",
        "required_status": "PASS_M1188_TRANSPORT_ADAPTER_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "environment_review_sha256": "M1188_EXPECTED_HAMMER_REVIEW_SHA256",
        "environment_manifest_sha256": "M1188_EXPECTED_HAMMER_MANIFEST_SHA256",
        "environment_outer_sha256": "M1188_EXPECTED_HAMMER_OUTER_SHA256"},
        "future hammer contract drift")


def verify_future_hammer(contract: dict[str, Any]) -> None:
    review = ROOT / FUTURE_HAMMER_REL / "review.json"
    manifest = ROOT / FUTURE_HAMMER_REL / "SHA256SUMS"
    outer = ROOT / FUTURE_HAMMER_REL / "SHA256SUMS.seal.sha256"
    for path in (review, manifest, outer):
        regular(path, "future different-author hammer")
    expected = [os.environ.get(name, "") for name in (
        "M1188_EXPECTED_HAMMER_REVIEW_SHA256", "M1188_EXPECTED_HAMMER_MANIFEST_SHA256",
        "M1188_EXPECTED_HAMMER_OUTER_SHA256")]
    require(all(len(value) == 64 for value in expected), "fresh hammer digest environment absent")
    require([sha256(review), sha256(manifest), sha256(outer)] == expected,
            "fresh hammer digest mismatch")
    require(outer.read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh hammer recursive seal mismatch")
    value = strict_json(review)
    future = contract["future_hammer"]
    require(value.get("schema") == future["required_schema"] and
            value.get("status") == future["required_status"],
            "fresh hammer semantic admission mismatch")


def build_archive(path: Path, members: list[dict[str, Any]]) -> str:
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for row in members:
            source = ROOT / repo_relative(row["path"])
            regular(source, "archive member")
            require(source.stat().st_size == row["size_bytes"] and sha256(source) == row["sha256"],
                    "archive member changed before read")
            info = tarfile.TarInfo(row["path"])
            info.size = row["size_bytes"]
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
            require(sha256(source) == row["sha256"], "archive member changed during read")
    regular(path, "local archive")
    return sha256(path)


REMOTE_EXTRACTOR = r'''import hashlib, json, os, pathlib, shutil, stat, sys, tarfile
def die(message):
    raise SystemExit("M1188_REMOTE_FAIL: " + message)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def safe_rel(text):
    p=pathlib.PurePosixPath(text)
    if p.is_absolute() or not p.parts or ".." in p.parts or str(p)!=text: die("unsafe path")
    return pathlib.Path(*p.parts)
def safe_parents(root, rel_parent):
    current=root
    for part in rel_parent.parts:
        current=current/part
        try: mode=current.lstat().st_mode
        except FileNotFoundError:
            current.mkdir(mode=0o755)
            mode=current.lstat().st_mode
        if not stat.S_ISDIR(mode) or current.is_symlink(): die("unsafe parent")
def main():
    spec=json.loads(MANIFEST_JSON)
    archive=pathlib.Path(ARCHIVE)
    root=pathlib.Path(ROOT)
    stage=pathlib.Path(STAGE)
    if sys.executable!=INTERPRETER or sys.version.split()[0]!=PYTHON_VERSION: die("interpreter identity")
    if not root.is_dir() or root.is_symlink(): die("unsafe remote repository")
    try: mode=archive.lstat().st_mode
    except FileNotFoundError: die("archive absent")
    if not stat.S_ISREG(mode) or archive.is_symlink() or digest(archive)!=ARCHIVE_SHA: die("archive identity")
    if stage.exists() or stage.is_symlink(): die("staging namespace not fresh")
    expected={row["path"]:row for row in spec}
    if len(spec)!=51 or len(expected)!=51: die("manifest cardinality")
    stage.mkdir(mode=0o700)
    try:
        with tarfile.open(archive,"r:") as tf:
            items=tf.getmembers()
            if [m.name for m in items]!=[row["path"] for row in spec]: die("archive order/set")
            for member in items:
                if not member.isfile() or member.issym() or member.islnk(): die("non-regular archive member")
                row=expected[member.name]
                if member.size!=row["size_bytes"]: die("member size")
                rel=safe_rel(member.name)
                out=stage/rel
                out.parent.mkdir(parents=True,exist_ok=True)
                src=tf.extractfile(member)
                if src is None: die("member unreadable")
                h=hashlib.sha256(); count=0
                fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o444)
                with os.fdopen(fd,"wb") as dst:
                    while True:
                        block=src.read(1<<20)
                        if not block: break
                        dst.write(block); h.update(block); count+=len(block)
                    dst.flush(); os.fsync(dst.fileno())
                if count!=row["size_bytes"] or h.hexdigest()!=row["sha256"]: die("member SHA")
        for row in spec:
            rel=safe_rel(row["path"]); dest=root/rel
            safe_parents(root,rel.parent)
            if dest.exists() or dest.is_symlink():
                mode=dest.lstat().st_mode
                if not stat.S_ISREG(mode) or dest.is_symlink(): die("unsafe existing destination")
            os.replace(stage/rel,dest)
        for row in spec:
            dest=root/safe_rel(row["path"])
            mode=dest.lstat().st_mode
            if not stat.S_ISREG(mode) or dest.is_symlink() or dest.stat().st_size!=row["size_bytes"] or digest(dest)!=row["sha256"]: die("post-install identity")
        shutil.rmtree(stage)
        archive.unlink()
        print(json.dumps({"status":"PASS_M1188_REMOTE_SAFE_EXTRACT","members":51,"verified":51,"archive_removed":True,"stage_removed":True},sort_keys=True))
    except BaseException:
        raise
main()
'''


def remote_program(members: list[dict[str, Any]], archive_sha: str,
                   archive: Path = REMOTE_ARCHIVE, root: Path = REMOTE_REPO,
                   stage: Path = REMOTE_STAGE, interpreter: str = REMOTE_INTERPRETER,
                   python_version: str = "3.10.20") -> bytes:
    manifest = [{"path": row["path"], "size_bytes": row["size_bytes"],
                 "sha256": row["sha256"]} for row in members]
    prefix = "\n".join((
        "MANIFEST_JSON=" + repr(json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
        "ARCHIVE_SHA=" + repr(archive_sha), "ARCHIVE=" + repr(str(archive)),
        "ROOT=" + repr(str(root)), "STAGE=" + repr(str(stage)),
        "INTERPRETER=" + repr(interpreter),
        "PYTHON_VERSION=" + repr(python_version), ""))
    return (prefix + REMOTE_EXTRACTOR).encode("utf-8")


def fixed_ssh_argv() -> list[str]:
    return [str(SSH), "-p", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", REMOTE_HOST, REMOTE_INTERPRETER, "-I", "-"]


def fixed_scp_argv(local_archive: Path) -> list[str]:
    return [str(SCP), "-P", REMOTE_PORT, "-o", "ControlPath=" + SSH_CONTROL_PATH,
            "-o", "BatchMode=yes", str(local_archive),
            REMOTE_HOST + ":" + str(REMOTE_ARCHIVE)]


def preflight_program() -> bytes:
    code = """import pathlib,stat,sys\nroot=pathlib.Path({root!r}); archive=pathlib.Path({archive!r}); stage=pathlib.Path({stage!r})\nassert sys.executable=={interp!r} and sys.version.split()[0]=='3.10.20'\nassert root.is_dir() and not root.is_symlink()\nassert not archive.exists() and not archive.is_symlink() and not stage.exists() and not stage.is_symlink()\nprint('PASS_M1188_REMOTE_PREFLIGHT__NO_WRITE')\n""".format(root=str(REMOTE_REPO), archive=str(REMOTE_ARCHIVE), stage=str(REMOTE_STAGE),
           interp=REMOTE_INTERPRETER)
    return code.encode("utf-8")


def consume_attempt() -> None:
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("M1188_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    require(len(sys.argv) == 1, "zero arguments required")
    contract = load_contract()
    verify_transport_contract(contract)
    members = exact_members(contract)
    verify_future_hammer(contract)
    require(not ATTEMPT.exists() and not RESULT.exists(), "transport attempt/result namespace not fresh")
    consume_attempt()
    preflight = subprocess.run(fixed_ssh_argv(), input=preflight_program(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=False, check=False)
    require(preflight.returncode == 0 and preflight.stdout.decode("utf-8", "replace").strip() ==
            "PASS_M1188_REMOTE_PREFLIGHT__NO_WRITE", "remote preflight failed")
    with tempfile.TemporaryDirectory(prefix="m1188_m1180_transport_") as temporary:
        archive = Path(temporary) / "exact51.tar"
        archive_sha = build_archive(archive, members)
        copied = subprocess.run(fixed_scp_argv(archive), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=False, check=False)
        require(copied.returncode == 0, "fixed-argv SCP/SFTP transport failed")
        extracted = subprocess.run(fixed_ssh_argv(), input=remote_program(members, archive_sha),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, check=False)
        require(extracted.returncode == 0, "remote safe extractor failed")
        output = strict_json_bytes(extracted.stdout.strip())
        require(output == {"archive_removed": True, "members": 51, "stage_removed": True,
                           "status": "PASS_M1188_REMOTE_SAFE_EXTRACT", "verified": 51},
                "remote extractor receipt drift")
    RESULT.mkdir(mode=0o755)
    receipt = {"schema": "m1188_m1180_exact51_transport_result_r1_v1",
               "status": PASS_TOKEN, "members": 51, "original_exact42": 42,
               "m1184_exact_seals": 9, "m1180_attempt_consumed": False,
               "gpu_or_capture_consumed": False, "paper_result": False}
    (RESULT / "transport_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULT / "RUN_COMPLETE.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TransportError, OSError, ValueError, json.JSONDecodeError) as error:
        print("M1188_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

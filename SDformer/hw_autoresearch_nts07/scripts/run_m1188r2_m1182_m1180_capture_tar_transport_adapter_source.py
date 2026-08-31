#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1188R2 additive transport adapter with strict M1184 semantic admission.

This source does not replace M1188R1 or any M1180/M1182/M1184 authority.  It
reuses the sealed R1 exact51/SCP-default-SFTP/safe-extract primitives, but both
member construction and the production pre-attempt gate independently parse
and admit the exact M1184 schema, status, verdict, bindings and authorization.
It is inert until a fresh different-author M1190 hammer is supplied.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1188r2_m1182_m1180_capture_tar_transport_adapter_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1188r2_m1182_m1180_capture_tar_transport_adapter_source.py")
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1188r2_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_20260830.json")
R1_SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py")
R1_CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1188_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_20260830.json")
M1184_REL = Path("hw_autoresearch_nts07/reviews/m1184_m1182_m1180_motion_ep29_unified_capture_launch_release_hammer_r1_20260830")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1190_m1188r2_m1182_m1180_capture_tar_transport_adapter_hammer_r1_20260830")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
REMOTE_ARCHIVE = Path("/tmp/m1188r2_m1180_exact51_transport_r1.tar")
REMOTE_STAGE = Path("/root/private_data/work/sdformer_codex/SDformer/.m1188r2_m1180_exact51_transport_stage_r1")
ATTEMPT = HW / "results/.m1188r2_m1180_exact51_transport_r1_attempt_consumed"
RESULT = HW / "results/m1188r2_m1180_exact51_transport_r1_20260830"
PASS_TOKEN = "PASS_M1188R2_EXACT51_TRANSFER__STRICT_M1184_SEMANTICS__M1180_ATTEMPT_AND_GPU_UNTOUCHED"


class R2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R2Error(message)


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
        raise R2Error("missing {}: {}".format(label, path)) from error
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
                           R2Error("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    return strict_json_bytes(path.read_bytes())


def load_r1_module():
    path = ROOT / R1_SOURCE_REL
    regular(path, "M1188R1 source")
    spec = importlib.util.spec_from_file_location("m1188r1_sealed_transport", path)
    require(spec is not None and spec.loader is not None, "cannot import sealed M1188R1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1 = load_r1_module()


def load_contract() -> dict[str, Any]:
    contract = strict_json(ROOT / CONTRACT_REL)
    require(set(contract) == {"schema", "status", "date", "source", "test", "r1_unchanged",
                              "m1189_p0", "m1184_exact_semantics", "transport",
                              "future_hammer", "claim_boundary", "docs359_sha256"},
            "R2 contract exact keys drift")
    require(contract["schema"] ==
            "m1188r2_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_v1" and
            contract["status"] ==
            "INERT_SOURCE_ONLY__M1189_P0_REPAIRED__FRESH_M1190_HAMMER_REQUIRED",
            "R2 contract schema/status drift")
    require(contract["date"] == "2026-08-30", "R2 contract date drift")
    for key, rel in (("source", SOURCE_REL), ("test", TEST_REL)):
        row = contract[key]
        require(set(row) == {"path", "size_bytes", "sha256"} and row["path"] == str(rel),
                key + " contract identity drift")
        path = ROOT / rel
        regular(path, key)
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                key + " bytes drift")
    r1 = contract["r1_unchanged"]
    require(r1 == {
        "source_path": str(R1_SOURCE_REL),
        "source_sha256": "d8396d86146ed4b59ffd645c03222175b1f418fc8f9e26e6231d696a63c87970",
        "contract_path": str(R1_CONTRACT_REL),
        "contract_sha256": "9cffdb64db11145d9c69332f1d152db761f148a30fba9cb8353341afd40ffb1b",
        "overwritten": False}, "R1 predecessor binding drift")
    require(sha256(ROOT / R1_SOURCE_REL) == r1["source_sha256"] and
            sha256(ROOT / R1_CONTRACT_REL) == r1["contract_sha256"],
            "R1 predecessor bytes drift")
    require(contract["m1189_p0"] == {
        "finding": "M1184_SEMANTIC_ADMISSION_ABSENT",
        "repair": "STRICT_SCHEMA_STATUS_VERDICT_BINDINGS_AUTHORIZATION_AND_RECURSIVE_SEALS",
        "r1_execution_authorized": False}, "M1189 P0 repair statement drift")
    require(contract["transport"] == {
        "members": 51, "original_exact42": 42, "m1184_exact_seals": 9,
        "scp_default_sftp": True, "shell": False, "fixed_argv": True,
        "control_socket": "/tmp/codex_m714_ssh.MFUzxMzZ/control.sock",
        "safe_extract": True, "post_install_size_sha_each_member": True,
        "remote_archive": str(REMOTE_ARCHIVE), "remote_stage": str(REMOTE_STAGE),
        "automatic_retry": False}, "R2 transport drift")
    require(contract["claim_boundary"] == {
        "source_only": True, "remote": False, "transfer": False, "gpu": False,
        "capture": False, "paper_result": False, "m1180_attempt_consumed": False,
        "m1184_modified": False}, "R2 claim boundary drift")
    require(contract["docs359_sha256"] == DOCS359_SHA256 and
            sha256(ROOT / DOCS359_REL) == DOCS359_SHA256, "docs/359 drift")
    return contract


def validate_m1184_review(review: dict[str, Any], contract: dict[str, Any]) -> None:
    expected = contract["m1184_exact_semantics"]
    require(set(expected) == {"review_path", "review_sha256", "manifest_sha256",
                              "outer_sha256", "schema", "status", "verdict",
                              "bindings", "authorization"},
            "M1184 semantic contract exact keys drift")
    require(set(review) == {"schema", "status", "date", "scope", "verdict", "score",
                            "bindings", "technical_verification", "authorization",
                            "claim_boundary", "no_actions_performed", "docs359_sha256"},
            "M1184 review top-level exact keys drift")
    require(review["schema"] == expected["schema"], "M1184 schema semantic drift")
    require(review["status"] == "PASS" and review["status"] == expected["status"],
            "M1184 status must be exact PASS")
    require(review["verdict"] ==
            "PASS_EXACT_TRANSFER_AND_ONE_REMOTE_GPU_LAUNCH_AUTHORIZED__NO_AUTOMATIC_RETRY__FRESH_RESULT_HAMMER_REQUIRED" and
            review["verdict"] == expected["verdict"], "M1184 verdict semantic drift")
    require(type(review["bindings"]) is dict and review["bindings"] == expected["bindings"],
            "M1184 bindings semantic drift")
    require(type(review["authorization"]) is dict and
            review["authorization"] == expected["authorization"],
            "M1184 authorization semantic drift")


def strict_m1184_admission(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["m1184_exact_semantics"]
    review_path = ROOT / Path(expected["review_path"])
    manifest = ROOT / M1184_REL / "SHA256SUMS"
    outer = ROOT / M1184_REL / "SHA256SUMS.seal.sha256"
    for path in (review_path, manifest, outer):
        regular(path, "M1184 semantic/seal authority")
    require(sha256(review_path) == expected["review_sha256"] and
            sha256(manifest) == expected["manifest_sha256"] and
            sha256(outer) == expected["outer_sha256"], "M1184 authority SHA drift")
    require(outer.read_text(encoding="utf-8") ==
            expected["manifest_sha256"] + "  SHA256SUMS\n", "M1184 outer seal drift")
    rows = R1.parse_sha_manifest(manifest)
    by_name = {name: digest for name, digest in rows}
    require(by_name.get("review.json") == expected["review_sha256"],
            "M1184 manifest does not bind review.json")
    for name, digest in rows:
        member = ROOT / M1184_REL / name
        regular(member, "M1184 manifest member")
        require(sha256(member) == digest, "M1184 inner member seal drift")
    review = strict_json(review_path)
    validate_m1184_review(review, contract)
    return review


def exact_members(contract: dict[str, Any]) -> list[dict[str, Any]]:
    # R1 closes exact42 + exact9 bytes. R2 makes semantics a mandatory part of
    # member construction, so even source-only DSE cannot bypass the M1184 gate.
    r1_contract = R1.load_contract()
    R1.verify_transport_contract(r1_contract)
    members = R1.exact_members(r1_contract)
    strict_m1184_admission(contract)
    require(len(members) == 51, "R2 exact51 population drift")
    return members


def verify_future_hammer(contract: dict[str, Any]) -> None:
    future = contract["future_hammer"]
    require(future == {
        "directory": str(FUTURE_HAMMER_REL),
        "required_schema": "m1190_m1188r2_m1182_m1180_capture_tar_transport_adapter_hammer_r1_v1",
        "required_status": "PASS_M1188R2_TRANSPORT_ADAPTER_RELEASE__ONE_TRANSFER_AUTHORIZED",
        "review_env": "M1188R2_EXPECTED_HAMMER_REVIEW_SHA256",
        "manifest_env": "M1188R2_EXPECTED_HAMMER_MANIFEST_SHA256",
        "outer_env": "M1188R2_EXPECTED_HAMMER_OUTER_SHA256"},
        "future M1190 contract drift")
    paths = [ROOT / FUTURE_HAMMER_REL / name for name in
             ("review.json", "SHA256SUMS", "SHA256SUMS.seal.sha256")]
    for path in paths:
        regular(path, "fresh M1190 hammer")
    names = (future["review_env"], future["manifest_env"], future["outer_env"])
    expected = [os.environ.get(name, "") for name in names]
    require(all(len(value) == 64 for value in expected), "fresh M1190 digest env absent")
    require([sha256(path) for path in paths] == expected, "fresh M1190 digest mismatch")
    require(paths[2].read_text(encoding="utf-8") == expected[1] + "  SHA256SUMS\n",
            "fresh M1190 recursive seal mismatch")
    review = strict_json(paths[0])
    require(review.get("schema") == future["required_schema"] and
            review.get("status") == future["required_status"],
            "fresh M1190 semantic admission mismatch")


def fixed_scp_argv(local_archive: Path) -> list[str]:
    argv = R1.fixed_scp_argv(local_archive)
    argv[-1] = R1.REMOTE_HOST + ":" + str(REMOTE_ARCHIVE)
    return argv


def preflight_program() -> bytes:
    code = """import pathlib,sys\nroot=pathlib.Path({root!r}); archive=pathlib.Path({archive!r}); stage=pathlib.Path({stage!r})\nassert sys.executable=={interp!r} and sys.version.split()[0]=='3.10.20'\nassert root.is_dir() and not root.is_symlink()\nassert not archive.exists() and not archive.is_symlink() and not stage.exists() and not stage.is_symlink()\nprint('PASS_M1188R2_REMOTE_PREFLIGHT__NO_WRITE')\n""".format(root=str(R1.REMOTE_REPO), archive=str(REMOTE_ARCHIVE), stage=str(REMOTE_STAGE),
           interp=R1.REMOTE_INTERPRETER)
    return code.encode("utf-8")


def consume_attempt() -> None:
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("M1188R2_TRANSPORT_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    require(len(sys.argv) == 1, "zero arguments required")
    contract = load_contract()
    members = exact_members(contract)
    verify_future_hammer(contract)
    # Production pre-attempt semantic revalidation is deliberately separate
    # from exact_members; a semantic mutation between gates fails before marker.
    strict_m1184_admission(contract)
    require(not ATTEMPT.exists() and not RESULT.exists(), "R2 attempt/result not fresh")
    consume_attempt()
    preflight = subprocess.run(R1.fixed_ssh_argv(), input=preflight_program(),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               shell=False, check=False)
    require(preflight.returncode == 0 and
            preflight.stdout.decode("utf-8", "replace").strip() ==
            "PASS_M1188R2_REMOTE_PREFLIGHT__NO_WRITE", "R2 remote preflight failed")
    with tempfile.TemporaryDirectory(prefix="m1188r2_m1180_transport_") as temporary:
        archive = Path(temporary) / "exact51.tar"
        archive_sha = R1.build_archive(archive, members)
        copied = subprocess.run(fixed_scp_argv(archive), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=False, check=False)
        require(copied.returncode == 0, "R2 fixed-argv SCP/SFTP failed")
        program = R1.remote_program(members, archive_sha, REMOTE_ARCHIVE, R1.REMOTE_REPO,
                                    REMOTE_STAGE, R1.REMOTE_INTERPRETER, "3.10.20")
        extracted = subprocess.run(R1.fixed_ssh_argv(), input=program, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=False, check=False)
        require(extracted.returncode == 0, "R2 remote safe extract failed")
        output = strict_json_bytes(extracted.stdout.strip())
        require(output == {"archive_removed": True, "members": 51, "stage_removed": True,
                           "status": "PASS_M1188_REMOTE_SAFE_EXTRACT", "verified": 51},
                "R2 remote receipt drift")
    RESULT.mkdir(mode=0o755)
    receipt = {"schema": "m1188r2_m1180_exact51_transport_result_r1_v1",
               "status": PASS_TOKEN, "members": 51, "strict_m1184_semantics": True,
               "m1180_attempt_consumed": False, "gpu_or_capture_consumed": False,
               "paper_result": False}
    (RESULT / "transport_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULT / "RUN_COMPLETE.txt").write_text(PASS_TOKEN + "\n", encoding="utf-8")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (R2Error, R1.TransportError, OSError, ValueError, json.JSONDecodeError) as error:
        print("M1188R2_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)

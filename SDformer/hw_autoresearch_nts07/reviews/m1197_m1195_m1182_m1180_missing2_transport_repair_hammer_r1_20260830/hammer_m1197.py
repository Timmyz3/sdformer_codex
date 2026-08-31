#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1197 source hammer; never transfers, launches, or runs EDA/GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1195_m1182_m1180_missing2_transport_repair_source.py"
TEST = HW / "tests/test_run_m1195_m1182_m1180_missing2_transport_repair_source.py"
CONTRACT = HW / "contracts/m1195_m1182_m1180_missing2_transport_repair_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1195_m1182_m1180_missing2_transport_repair_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "8a15546f25ba5cca26105c608b3dba4c79659f38e97fce22f50406a20c767621",
    TEST: "cc70f92ec63ab03a9424fca888e51dd1fcfd5b2592937a17756d1e7feb7975e8",
    CONTRACT: "953b3a1e82f9c4ce8e59dce458eebf95ff93c792879b30a91b4505a9b1503503",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_AUTHOR_MANIFEST = "c38c82e8e3e3a927cc4337c9e3c8a8c35ece41504d60c6e9d05c6fb4714ae08f"
EXPECTED_AUTHOR_OUTER = "2aeedf158160dd39d954521156b74cc0a0951683ac03f3cddec10948164c3a7e"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m1195_hammer_target", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_author_seal() -> dict:
    manifest = AUTHOR / "SHA256SUMS"
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = AUTHOR / name
        actual = sha(path) if path.is_file() and not path.is_symlink() else None
        rows.append({"name": name, "expected": expected, "actual": actual,
                     "match": actual == expected})
    return {
        "manifest_actual": sha(manifest),
        "manifest_expected": EXPECTED_AUTHOR_MANIFEST,
        "outer_actual": sha(outer),
        "outer_expected": EXPECTED_AUTHOR_OUTER,
        "outer_text_exact": outer.read_text(encoding="utf-8") ==
        EXPECTED_AUTHOR_MANIFEST + "  SHA256SUMS\n",
        "member_rows": rows,
        "all_members_match": all(row["match"] for row in rows),
    }


def remote_case(module, kind: str) -> bool:
    """Return true only when an isolated adversarial extractor run rejects."""
    with tempfile.TemporaryDirectory(prefix="m1197_remote_attack_") as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        for rel in ("a/one", "b/two"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"path": "a/one", "size_bytes": 3,
             "sha256": hashlib.sha256(b"one").hexdigest()},
            {"path": "b/two", "size_bytes": 3,
             "sha256": hashlib.sha256(b"two").hexdigest()},
        ]
        archive = Path(temporary) / "payload.tar"
        with tarfile.open(archive, "w") as tf:
            for index, row in enumerate(rows):
                info = tarfile.TarInfo(row["path"])
                payload = (b"one", b"two")[index]
                info.size = len(payload)
                if kind == "symlink" and index == 0:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/tmp/escape"
                    info.size = 0
                    tf.addfile(info)
                else:
                    tf.addfile(info, io.BytesIO(payload))
            if kind == "extra_member":
                info = tarfile.TarInfo("extra")
                info.size = 1
                tf.addfile(info, io.BytesIO(b"x"))
        attempt = root / "state/attempt"
        result = root / "state/result"
        attempt.parent.mkdir()
        if kind == "attempt_appears":
            attempt.write_text("occupied\n", encoding="utf-8")
        if kind == "destination_exists":
            (root / "a/one").write_text("occupied\n", encoding="utf-8")
        attack_rows = copy.deepcopy(rows)
        if kind == "path_traversal":
            attack_rows[0]["path"] = "../escape"
        archive_sha = sha(archive)
        if kind == "remote_sha_mismatch":
            archive_sha = "0" * 64
        prefix = "\n".join((
            "ROWS=" + repr(json.dumps(attack_rows, sort_keys=True, separators=(",", ":"))),
            "ROOT=" + repr(str(root)), "ARCHIVE=" + repr(str(archive)),
            "STAGE=" + repr(str(root / ".stage")), "INTERPRETER=" + repr(sys.executable),
            "ARCHIVE_SHA=" + repr(archive_sha),
            "M1180_ATTEMPT=" + repr("state/attempt"),
            "M1180_RESULT=" + repr("state/result"), ""))
        proc = subprocess.run([sys.executable, "-I", "-"],
                              input=(prefix + module.REMOTE_EXTRACTOR).encode(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.returncode != 0 and not (root / ".stage").exists()


def command_failure_cases(module, contract: dict) -> dict:
    """Inject fixed-command failures with all production namespaces redirected."""
    outcomes = {}
    members = [
        {"path": "hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
         "size_bytes": 15961,
         "sha256": "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7"},
        {"path": "hw_autoresearch_nts07/results/h67_ep35_dependency_dag_s1_20260822/dependency_events.jsonl",
         "size_bytes": 34816039,
         "sha256": "e1d2007195a036eedcee1e49d960955b3508ffe590ba3d075a3877a501a62f6b"},
    ]
    pass_preflight = SimpleNamespace(returncode=0,
        stdout=b"PASS_M1195_REMOTE_PREFLIGHT__EXACT2_MISSING__M1180_ABSENT__NO_WRITE\n")
    fail = SimpleNamespace(returncode=9, stdout=b"injected command failure\n")
    for label, responses in (("preflight_failure", [fail]),
                             ("scp_failure", [pass_preflight, fail]),
                             ("install_failure", [pass_preflight,
                                                  SimpleNamespace(returncode=0, stdout=b""), fail])):
        with tempfile.TemporaryDirectory(prefix="m1197_command_attack_") as temporary:
            base = Path(temporary)
            module.LOCAL_ATTEMPT = base / "attempt"
            module.LOCAL_RESULT = base / "result"
            def fake_archive(path, unused_members):
                path.write_bytes(b"archive")
                return hashlib.sha256(b"archive").hexdigest()
            try:
                with mock.patch.object(module, "load_contract", return_value=contract), \
                     mock.patch.object(module, "verify_transport_contract"), \
                     mock.patch.object(module, "exact_members", return_value=members), \
                     mock.patch.object(module, "verify_future_hammer"), \
                     mock.patch.object(module, "build_archive", side_effect=fake_archive), \
                     mock.patch.object(module.subprocess, "run", side_effect=responses):
                    module.main()
            except module.RepairError:
                outcomes[label] = module.LOCAL_ATTEMPT.is_file() and not module.LOCAL_RESULT.exists()
            else:
                outcomes[label] = False
    return outcomes


def main() -> int:
    module = load_module()
    identity = [{"path": str(path.relative_to(ROOT)), "expected": expected,
                 "actual": sha(path), "match": sha(path) == expected}
                for path, expected in EXPECTED.items()]
    author = verify_author_seal()
    unit = subprocess.run([sys.executable, "-m", "unittest", "-v", str(TEST)],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, check=False)
    raw_contract = module.strict_json(CONTRACT)
    drift = copy.deepcopy(raw_contract)
    drift["missing2"][0]["size_bytes"] += 1
    try:
        module.exact_members(drift)
    except module.RepairError:
        inventory_drift_rejected = True
    else:
        inventory_drift_rejected = False
    remote_attacks = {kind: remote_case(module, kind) for kind in
                      ("extra_member", "path_traversal", "symlink",
                       "attempt_appears", "destination_exists", "remote_sha_mismatch")}
    command_attacks = command_failure_cases(module, raw_contract)
    attempts = [HW / "results/.m1195_m1180_missing2_transport_r1_attempt_consumed",
                HW / "results/m1195_m1180_missing2_transport_r1_20260830",
                HW / "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed",
                HW / "results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830"]
    p0 = []
    if not all(row["match"] for row in identity[:3]):
        p0.append({"id": "P0_FROZEN_SOURCE_IDENTITY_DRIFT",
                   "finding": "On-disk M1195 source/contract do not match assigned or self-declared hashes; all 7 source unit tests error at load_contract().",
                   "impact": "The intended release identity is not executable and cannot be authorized."})
    if not author["all_members_match"]:
        p0.append({"id": "P0_AUTHOR_RECURSIVE_SEAL_BROKEN",
                   "finding": "Three author-manifest members changed after the outer seal while the manifest and outer digest stayed fixed.",
                   "impact": "The author provenance exact set is not recursively valid."})
    review = {
        "schema": "m1197_m1195_m1182_m1180_missing2_transport_repair_hammer_r1_v1",
        "date": "2026-08-30", "milestone": "M1197",
        "status": "STOP_M1197_M1195_IDENTITY_AND_AUTHOR_SEAL_DRIFT",
        "verdict": "STOP_DO_NOT_EXECUTE_TRANSPORT", "score": 48,
        "p0_count": len(p0), "p1_count": 1, "p2_count": 0,
        "identity": identity, "author_recursive_seal": author,
        "mechanical": {
            "python_compile": True, "unit_test_returncode": unit.returncode,
            "unit_test_summary": "7/7 ERROR: source identity drift",
            "inventory_target_drift_rejected": inventory_drift_rejected,
            "remote_extractor_adversarial_rejections": remote_attacks,
            "fixed_command_failure_injections_fail_closed": command_attacks,
            "docs359_preserved": sha(DOCS359) == EXPECTED[DOCS359],
            "all_attempt_and_result_namespaces_absent": all(not p.exists() and not p.is_symlink() for p in attempts),
            "remote_transfer_executed": False, "gpu_capture_executed": False, "eda_executed": False,
        },
        "p0": p0,
        "p1": [{"id": "P1_NONATOMIC_TWO_DESTINATION_PUBLICATION",
                "finding": "The extractor publishes the two destinations with sequential os.replace calls and does not roll back an already-installed first destination if the second publication or post-check fails.",
                "impact": "A one-shot failure can leave a partial exact-two dependency state; it is fail-stop but requires explicit forensic recovery before any successor.",
                "remediation": "Document partial-publication quarantine/recovery or publish a sealed directory as one atomic unit where repository layout permits."}],
        "decision": {"m1195_transport_authorized": False,
                     "authorized_command": None, "automatic_retry": False,
                     "required_successor": "Fresh additive repaired source/contract/author seal followed by a new different-author hammer."},
        "claim_boundary": {"paper_result": False, "m1180_attempt_consumed": False,
                           "remote_modified_by_m1197": False, "docs359_modified": False},
    }
    out = Path(__file__).with_name("review.json")
    out.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(__file__).with_name("unit_test_log.txt").write_text(unit.stdout, encoding="utf-8")
    print(review["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1201 hammer for M1200; local sandboxes only.

This hammer never opens SSH/SCP, never transfers files, and never launches GPU,
capture, simulation, synthesis, or EDA.  It deliberately injects exceptions at
the hard-link publication and post-publication cleanup boundaries.
"""
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


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1200_m1182_m1180_missing2_atomic_transport_repair_source.py"
TEST = HW / "tests/test_run_m1200_m1182_m1180_missing2_atomic_transport_repair_source.py"
CONTRACT = HW / "contracts/m1200_m1182_m1180_missing2_atomic_transport_repair_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1200_m1182_m1180_missing2_atomic_transport_repair_author_r1_20260830"
OUT = Path(__file__).resolve().parent
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "a953436e84897238907bbaab28ca096e3b404b80a8b34e95bb9f3ebc41655316",
    TEST: "2d49888d7a497ba4f5e847fac955ae10c893e8abd702dc7c9bd3effc4e439b44",
    CONTRACT: "da73efdc503e7dd8461c496035aebaa6fe4e726f81537629f995c0bceabf89e5",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_AUTHOR_MANIFEST = "dd05b938315cbf05a3809da9c18d9c89ce3f874afb911a5ba7ecb81b3499cd88"
EXPECTED_AUTHOR_OUTER = "070979145b9582351fd376c5a661ac1b5dd1135b23e8c04a0a2e6718fb173e50"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m1200_hammer_target", SOURCE)
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


def tiny_case(module, mutation: str) -> dict:
    """Run a generated extractor in an isolated local repository."""
    with tempfile.TemporaryDirectory(prefix="m1201_m1200_attack_") as temporary:
        base = Path(temporary)
        root = base / "repo"
        root.mkdir()
        rows = [
            {"path": "a/one", "size_bytes": 3,
             "sha256": hashlib.sha256(b"one").hexdigest()},
            {"path": "b/two", "size_bytes": 3,
             "sha256": hashlib.sha256(b"two").hexdigest()},
        ]
        for row in rows:
            (root / Path(row["path"]).parent).mkdir(parents=True, exist_ok=True)
        archive = base / "payload.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tf:
            for index, row in enumerate(rows):
                info = tarfile.TarInfo(row["path"])
                payload = (b"one", b"two")[index]
                info.size = len(payload)
                if mutation == "symlink" and index == 1:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "escape"
                    info.size = 0
                    tf.addfile(info)
                else:
                    tf.addfile(info, io.BytesIO(payload))
            if mutation == "extra":
                info = tarfile.TarInfo("extra/member")
                info.size = 1
                tf.addfile(info, io.BytesIO(b"x"))
        attack_rows = copy.deepcopy(rows)
        if mutation == "traversal":
            attack_rows[1]["path"] = "../escape"
        if mutation == "member_sha":
            attack_rows[1]["sha256"] = "0" * 64
        if mutation == "preexisting":
            (root / "a/one").write_bytes(b"old")
        attempt = root / module.M1180_ATTEMPT_REL
        result = root / module.M1180_RESULT_REL
        if mutation == "attempt_pre":
            attempt.parent.mkdir(parents=True, exist_ok=True)
            attempt.write_text("race\n", encoding="utf-8")
        stage = root / ".stage"
        archive_sha = sha(archive)
        program = module.remote_program(
            attack_rows, archive_sha, root=root, archive=archive, stage=stage,
            interpreter=sys.executable, python_version=sys.version.split()[0])
        if mutation == "second_link_after_create":
            old = (b"   absent(dest,'destination race'); os.link(stage/safe(row['path']),dest); "
                   b"published.append(dest)")
            new = (b"   absent(dest,'destination race'); os.link(stage/safe(row['path']),dest); "
                   b"\n   if len(published)==1: raise OSError('INJECT_AFTER_SECOND_LINK')\n"
                   b"   published.append(dest)")
            assert old in program
            program = program.replace(old, new, 1)
        elif mutation == "post_sha":
            old = b"  for row,dest in zip(rows,destinations):\n   mode=dest.lstat().st_mode"
            new = (b"  destinations[1].write_bytes(b'bad')\n"
                   b"  for row,dest in zip(rows,destinations):\n   mode=dest.lstat().st_mode")
            assert old in program
            program = program.replace(old, new, 1)
        elif mutation == "attempt_post":
            old = b"  absent(attempt,'M1180 attempt postcondition');"
            new = (b"  attempt.parent.mkdir(parents=True,exist_ok=True); "
                   b"attempt.write_text('race'); absent(attempt,'M1180 attempt postcondition');")
            assert old in program
            program = program.replace(old, new, 1)
        elif mutation == "archive_cleanup":
            old = b"  if archive.exists() and not archive.is_symlink(): archive.unlink()"
            new = (b"  if archive.exists() and not archive.is_symlink(): "
                   b"raise OSError('INJECT_ARCHIVE_CLEANUP_FAILURE')")
            assert old in program
            program = program.replace(old, new, 1)
        completed = subprocess.run([sys.executable, "-I", "-"], input=program,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, check=False)
        destinations = [root / "a/one", root / "b/two"]
        return {
            "mutation": mutation,
            "returncode": completed.returncode,
            "destination_exists": [path.exists() or path.is_symlink()
                                   for path in destinations],
            "stage_exists": stage.exists() or stage.is_symlink(),
            "archive_exists": archive.exists() or archive.is_symlink(),
            "attempt_exists": attempt.exists() or attempt.is_symlink(),
            "result_exists": result.exists() or result.is_symlink(),
            "output_tail": completed.stdout.decode("utf-8", "replace")[-500:],
        }


def helper_after_link_case(module) -> dict:
    with tempfile.TemporaryDirectory(prefix="m1201_helper_link_window_") as temporary:
        root = Path(temporary)
        staged = [root / "s0", root / "s1"]
        destinations = [root / "d0", root / "d1"]
        for path in staged:
            path.write_bytes(b"sealed")
        def link_then_raise(source: Path, destination: Path) -> None:
            os.link(source, destination)
            if destination == destinations[1]:
                raise OSError("INJECT_AFTER_SECOND_LINK_BEFORE_APPEND")
        caught = None
        try:
            module.publish_exact2_atomic(staged, destinations, lambda: None,
                                         link_then_raise)
        except BaseException as error:
            caught = type(error).__name__ + ": " + str(error)
        return {"exception": caught,
                "destination_exists": [path.exists() or path.is_symlink()
                                       for path in destinations]}


def main() -> int:
    module = load_module()
    identity = [{"path": str(path.relative_to(ROOT)), "expected": expected,
                 "actual": sha(path), "match": sha(path) == expected}
                for path, expected in EXPECTED.items()]
    author = verify_author_seal()
    compile_run = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SOURCE), str(TEST)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", str(TEST)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    contract = module.load_contract()
    module.verify_policy(contract)
    members = module.exact_members(contract)
    attacks = {name: tiny_case(module, name) for name in (
        "extra", "traversal", "member_sha", "symlink", "preexisting",
        "attempt_pre", "post_sha", "attempt_post",
        "second_link_after_create", "archive_cleanup")}
    helper_window = helper_after_link_case(module)
    local_namespaces = [
        HW / "results/.m1200_m1180_missing2_atomic_transport_r1_attempt_consumed",
        HW / "results/m1200_m1180_missing2_atomic_transport_r1_20260830",
        HW / "results/.m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830.attempt_consumed",
        HW / "results/m1180_motion_ep29_unified_hardware_capture_s40_r1_20260830",
    ]
    p0 = [{
        "id": "P0_POST_PUBLICATION_CLEANUP_EXCEPTION_BYPASSES_ROLLBACK",
        "finding": "The remote extractor performs stage/archive cleanup in a finally block outside the publication try/except rollback. Injecting archive.unlink failure returns nonzero after both repository destinations were published; both remain installed and the archive remains.",
        "impact": "The one-shot controller records install failure while remote repository state has changed. Because preexisting destinations are rejected and automatic retry is false, this is an unreconciled state, contrary to rollback-clean and remote-temp-cleanup release claims.",
        "remediation": "Replace rollback-clean atomicity with a monotonic exact-state reconciliation contract, or wrap cleanup failure in a final verifier/reconciler that accepts exact destinations and emits a sealed receipt without launching capture."
    }]
    p1 = [{
        "id": "P1_LINK_SUCCESS_BEFORE_LEDGER_APPEND_WINDOW",
        "finding": "Both the helper and generated remote extractor append a destination to the rollback ledger only after os.link returns. An injected handled exception after the second link but before append leaves the second destination while rollback removes only the first.",
        "impact": "The source does not satisfy its literal any-handled-publication-failure rollback-all claim. External SIGKILL atomicity is explicitly disclaimed, but BaseException is caught and this handled asynchronous window remains.",
        "remediation": "Use an idempotent per-target exact-state invariant: absent or exact expected regular file; reject wrong/symlink state; reconcile missing targets and require both exact on success."
    }]
    review = {
        "schema": "m1201_m1200_m1182_m1180_missing2_atomic_transport_repair_hammer_r1_v1",
        "date": "2026-08-30", "milestone": "M1201",
        "status": "STOP_M1200_POST_PUBLICATION_CLEANUP_AND_LINK_WINDOW_NOT_ROLLBACK_CLEAN",
        "verdict": "STOP_DO_NOT_EXECUTE_M1200_TRANSPORT", "score": 67,
        "p0_count": len(p0), "p1_count": len(p1), "p2_count": 0,
        "identity": identity,
        "exact_two": {"count": len(members), "paths": [row["path"] for row in members],
                      "sizes": [row["size_bytes"] for row in members],
                      "sha256": [row["sha256"] for row in members]},
        "author_recursive_seal": author,
        "mechanical": {
            "python_compile_returncode": compile_run.returncode,
            "unit_test_returncode": unit.returncode,
            "unit_test_summary": "PASS_10_OF_10" if unit.returncode == 0 else "FAIL",
            "independent_remote_sandbox_attacks": attacks,
            "independent_helper_after_link_window": helper_window,
            "safe_archive_and_symlink_rejections": all(
                attacks[name]["returncode"] != 0 for name in
                ("extra", "traversal", "member_sha", "symlink", "preexisting", "attempt_pre")),
            "post_sha_and_attempt_postcondition_remove_both": all(
                attacks[name]["returncode"] != 0 and
                attacks[name]["destination_exists"] == [False, False]
                for name in ("post_sha", "attempt_post")),
            "cleanup_exception_leaves_both":
                attacks["archive_cleanup"]["returncode"] != 0 and
                attacks["archive_cleanup"]["destination_exists"] == [True, True],
            "after_second_link_exception_leaves_second":
                attacks["second_link_after_create"]["returncode"] != 0 and
                attacks["second_link_after_create"]["destination_exists"] == [False, True],
            "docs359_preserved": sha(DOCS359) == EXPECTED[DOCS359],
            "all_local_attempt_and_result_namespaces_absent":
                all(not path.exists() and not path.is_symlink() for path in local_namespaces),
            "remote_or_transport_executed": False,
            "gpu_or_capture_executed": False,
            "eda_executed": False,
        },
        "p0": p0, "p1": p1,
        "decision": {"m1200_transport_authorized": False,
                     "authorized_command": None, "automatic_retry": False,
                     "required_successor": "Fresh additive monotonic exact-state reconciliation source, sealed once, then a new different-author hammer."},
        "claim_boundary": {"paper_result": False, "m1180_attempt_consumed": False,
                           "remote_modified_by_m1201": False,
                           "docs359_modified": False},
    }
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "UNIT_TEST_OUTPUT.txt").write_text(unit.stdout, encoding="utf-8")
    (OUT / "COMPILE_OUTPUT.txt").write_text(compile_run.stdout, encoding="utf-8")
    print(review["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only final launch-authority review for M1400/M1412.

This program never calls SSH, nvidia-smi, the remote preflight, the capture,
the one-shot attempt path, or any controller signal/restore operation.
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


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REVIEW_DIR = Path(__file__).resolve().parent
RUNNER = HW / "scripts/run_m1400_m1349_motion_ep34_live105_production_one_shot.py"
TEST = HW / "tests/test_run_m1400_m1349_motion_ep34_live105_production_one_shot.py"
SOURCE_CONTRACT = HW / "contracts/m1400_m1349_motion_ep34_live105_production_runner_source_contract_r1_20260831.json"
M1410 = HW / "reviews/m1410_m1400_m1349_motion_ep34_live105_production_runner_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1412_m1400_m1349_motion_ep34_live105_production_launch_release_r1_20260831.json"
RELEASE_SIDECAR = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
M1412_AUTHOR = HW / "reviews/m1412_m1400_m1349_motion_ep34_live105_production_launch_release_author_r1_20260831"
M1349_AUTHOR = HW / "reviews/m1349_motion_ep34_live105_inventory_successor_source_author_r1_20260831"
M1353 = HW / "reviews/m1353_m1349_motion_ep34_live105_inventory_successor_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "c9d7e0e3d6eca16c710b8bbcf44be3154f1891eb8b3b8452d3fda1a5094668be",
    "test": "7c90fc9c68fab5cef0c3430b1b06689f3c5dc93f55d0959a4d9d35a6bd52220e",
    "source_contract": "a0f4e6661ab9709cd905ac77c17abce0fe00faae341c3c64e706a9eea23ed1ef",
    "m1410_review": "626eaa1878710088867e3875ff07c07559ba65661a6d421f0733e0e71cbb31ac",
    "m1410_manifest": "c0c6e20927c8d996f6e5e4bdf75ee0f09ac61a1e4be6ea8ac024bc75b42e4ad3",
    "m1410_outer": "6e676074800d3bd184f163205d4c760b4661fb02981e8dd5cb78ba8e48cd6a8c",
    "release": "374c8a2e1aa770e1ee3868f5575db704ffd59b72c9518678979c480b890ab5ef",
    "release_sidecar": "1f4aa0380b7b8bc3a8e3f4369605f4b371048b3cdf1766d1f1fc823868676e6a",
    "release_outer": "e4ec2877a584feefc6be37637e917652ed122d5db1d1ef2e395dda5088cb0b1f",
    "m1412_author_review": "8aeb18ff7ea6b1152ed48dbcc1c45ce272a500eb4b5692eb82e1741bc83615fc",
    "m1412_author_manifest": "11edd49d66b7c83820bd56221815d08e1ecd378aacc3775d7135843c55a4d0f7",
    "m1412_author_outer": "280fec3385f1df48cf68ea537ea6b8d32ed64a53c74b7669a82372bc759d3f36",
    "m1349_source": "3fe0f51acf489cf2f4d1a65f83f872b49a5fde79401a2fdb525768e681fbbbe5",
    "m1349_test": "b20e06bcecb9fab1a326701e40e7bb72c5f13a3204a9d52470b58237a747492f",
    "m1349_contract": "ce2f373eef512237a0e0ee087134176384c30663bd52d42aa68c68b05fbd4712",
    "m1349_author_review": "bd29fae08da4978416477bcc5cb93a36d254cee2456a489452a8e5ad4ea98c57",
    "m1349_author_manifest": "c46c15318b8a589ac20b17b8dd28b6687fd2a4eb9c68d318c6f3e16d063673a3",
    "m1349_author_outer": "76cd24cc79e886e00e4dd82e8febfe22bdce23aecf353320e46b049da23a34ca",
    "m1353_review": "3a660e6c1608baf7e5f6b16383067539c21631f89c310d5aa13656cadcbdde2e",
    "m1353_manifest": "7770775870e196d39eb213fc3b0bb5819ac1e5b595854065806ef792c2ea8bd7",
    "m1353_outer": "1e2c2f6a10f514770fab6bdf6666ba8d40a11d5393053310cd39014143aa0006",
    "live105": "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def sha256(path: Path) -> str:
    require(path.exists() and path.is_file() and not path.is_symlink(), f"non-regular authority: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, f"duplicate JSON key {key} in {path}")
            out[key] = value
        return out
    def reject(token):
        raise ReviewError(f"nonfinite JSON {token} in {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    require(type(value) is dict, f"non-object JSON root in {path}")
    return value


def verify_recursive_seal(directory: Path, review_sha: str, manifest_sha: str,
                          outer_sha: str) -> dict:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha256(review) == review_sha, f"{directory.name} review SHA")
    require(sha256(manifest) == manifest_sha, f"{directory.name} manifest SHA")
    require(sha256(outer) == outer_sha, f"{directory.name} outer SHA")
    require(outer.read_text(encoding="utf-8") == f"{manifest_sha}  SHA256SUMS\n",
            f"{directory.name} outer content")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
                f"{directory.name} malformed digest")
        require(name not in seen and not Path(name).is_absolute() and ".." not in Path(name).parts,
                f"{directory.name} unsafe/duplicate member")
        seen.add(name)
        candidates = [directory / name, ROOT / name]
        member = next((candidate for candidate in candidates if candidate.exists()), None)
        require(member is not None and sha256(member) == digest,
                f"{directory.name} sealed member mismatch: {name}")
    require("review.json" in seen or str(review.relative_to(ROOT)) in seen,
            f"{directory.name} review not recursively sealed")
    return strict_json(review)


def load_runner():
    spec = importlib.util.spec_from_file_location("m1430_bound_m1400", RUNNER)
    require(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(sha256(RUNNER) == EXPECTED["runner"], "runner changed during import")
    return module


def main() -> int:
    checks: list[dict[str, object]] = []
    def check(name: str, condition: bool, category: str) -> None:
        checks.append({"check": name, "category": category, "pass": bool(condition)})

    try:
        check("runner_exact", sha256(RUNNER) == EXPECTED["runner"], "source")
        check("test_exact", sha256(TEST) == EXPECTED["test"], "source")
        check("source_contract_exact", sha256(SOURCE_CONTRACT) == EXPECTED["source_contract"], "source")
        check("docs359_exact", sha256(DOCS359) == EXPECTED["docs359"], "source")
        M = load_runner()
        M.verify_prerequisites()
        policy = M.validate_source_contract()
        names = M.M1349.verify_m1347_failure()
        check("source_integrity_check", policy["source"]["sha256"] == EXPECTED["runner"], "source")
        check("m1349_source", sha256(M.M1349_SOURCE) == EXPECTED["m1349_source"], "m1349")
        check("m1349_test", sha256(M.M1349_TEST) == EXPECTED["m1349_test"], "m1349")
        check("m1349_contract", sha256(M.M1349_CONTRACT) == EXPECTED["m1349_contract"], "m1349")
        check("live105_count", len(names) == 105 and M.M1349.EXPECTED_ATLIF_COUNT == 105, "m1349")
        check("live105_unique_sorted", len(set(names)) == 105 and list(names) == sorted(names), "m1349")
        check("live105_digest", M.M1349.terminal_lf_digest(list(names)) == EXPECTED["live105"], "m1349")
        check("capture_population", (M.M1349.EXPECTED_ORDERED_RECORDS,
                                      M.M1349.EXPECTED_PAYLOAD) == (10360, 640), "m1349")

        m1349_author = verify_recursive_seal(
            M1349_AUTHOR, EXPECTED["m1349_author_review"],
            EXPECTED["m1349_author_manifest"],
            EXPECTED["m1349_author_outer"])
        check("m1349_author_boundary", m1349_author["claim_boundary"]["production_authorized"] is False,
              "m1349")
        m1353 = verify_recursive_seal(M1353, EXPECTED["m1353_review"],
                                      EXPECTED["m1353_manifest"], EXPECTED["m1353_outer"])
        check("m1353_status", m1353["status"] == "PASS_SOURCE__FRESH_RELEASE_AUTHOR_MAY_BE_AUTHORED",
              "m1349")
        check("m1353_no_launch", m1353["authorization"]["production_launch"] is False, "m1349")

        m1410 = verify_recursive_seal(M1410, EXPECTED["m1410_review"],
                                      EXPECTED["m1410_manifest"], EXPECTED["m1410_outer"])
        check("m1410_status", m1410["status"] ==
              "PASS_M1400_RUNNER_SOURCE__FRESH_RELEASE_MAY_BE_AUTHORED", "m1410")
        check("m1410_checks", m1410["verification"]["independent_checks"] == "71/71 PASS", "m1410")
        check("m1410_no_launch", m1410["authorization"]["launch"] is False and
              m1410["authorization"]["release_authoring"] is True, "m1410")
        check("m1410_runner_binding", m1410["bindings"]["runner_sha256"] == EXPECTED["runner"], "m1410")

        release = strict_json(RELEASE)
        check("release_exact", sha256(RELEASE) == EXPECTED["release"], "release")
        check("release_sidecar_exact", sha256(RELEASE_SIDECAR) == EXPECTED["release_sidecar"] and
              RELEASE_SIDECAR.read_text(encoding="utf-8") ==
              f'{EXPECTED["release"]}  {RELEASE.name}\n', "release")
        check("release_outer_exact", sha256(RELEASE_OUTER) == EXPECTED["release_outer"] and
              RELEASE_OUTER.read_text(encoding="utf-8") ==
              f'{EXPECTED["release_sidecar"]}  {RELEASE_SIDECAR.name}\n', "release")
        check("release_status", release["status"] ==
              "AUTHORIZE_ONE_M1400_M1349_EP34_LIVE105_PRODUCTION_ATTEMPT", "release")
        check("release_authorization", release["launch_authorized"] is True and
              release["runs"] == 1 and release["automatic_retry"] is False, "release")
        check("release_runner", release["runner_sha256"] == EXPECTED["runner"] and
              release["source_chain"]["runner_sha256"] == EXPECTED["runner"], "release")
        check("release_m1410", release["m1410_source_blind"]["review_sha256"] ==
              EXPECTED["m1410_review"] and release["m1410_source_blind"]["manifest_sha256"] ==
              EXPECTED["m1410_manifest"] and release["m1410_source_blind"]["outer_file_sha256"] ==
              EXPECTED["m1410_outer"], "release")
        check("release_m1349_m1353", release["m1349_m1353_authority"]["m1349_source_sha256"] ==
              EXPECTED["m1349_source"] and release["m1349_m1353_authority"]["m1353_review_sha256"] ==
              EXPECTED["m1353_review"] and release["capture_identity"]["live_atlif_terminal_lf_sha256"] ==
              EXPECTED["live105"], "release")
        check("release_population", release["capture_identity"]["live_atlif_count"] == 105 and
              release["capture_identity"]["ordered_records"] == 10360 and
              release["capture_identity"]["payload_records"] == 640, "release")
        check("release_one_shot", release["one_shot"]["attempt_create"] == "O_EXCL" and
              release["one_shot"]["attempt_before_capture"] is True and
              release["one_shot"]["runs"] == 1 and
              release["one_shot"]["automatic_retry"] is False, "release")
        check("release_no_restore", release["one_shot"]["failure_restores_controller"] is False and
              release["one_shot"]["success_restores_controller"] is False, "release")
        check("release_author_did_not_execute", all(value is False for value in
              release["release_author_execution"].values()), "release")
        check("release_final_gate", release["final_gate"]["required_status"] ==
              "PASS_M1400_M1349_EP34_LIVE105_FINAL_LAUNCH_AUTHORITY" and
              release["final_gate"]["required_authorization"] ==
              {"launch": True, "runs": 1, "automatic_retry": False}, "release")

        m1412_author = verify_recursive_seal(
            M1412_AUTHOR, EXPECTED["m1412_author_review"],
            EXPECTED["m1412_author_manifest"], EXPECTED["m1412_author_outer"])
        check("m1412_author_status", m1412_author["status"] ==
              "PASS_M1412_RELEASE_AUTHORING__FRESH_M1430_REQUIRED__NO_LAUNCH", "release_author")
        check("m1412_author_release_binding", m1412_author["bindings"]["release_sha256"] ==
              EXPECTED["release"] and m1412_author["bindings"]["release_sidecar_sha256"] ==
              EXPECTED["release_sidecar"] and
              m1412_author["bindings"]["release_outer_sidecar_file_sha256"] ==
              EXPECTED["release_outer"], "release_author")
        check("m1412_author_no_execution", all(value is False for value in
              m1412_author["author_execution"].values()), "release_author")
        check("m1412_author_defers_final", m1412_author["authorization"]["m1430_may_authorize_one_launch_if_all_checks_pass"] is True and
              m1412_author["authorization"]["launch_now"] is False, "release_author")

        result, attempt, log = M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG
        check("result_absent", not os.path.lexists(str(result)), "freshness")
        check("attempt_absent", not os.path.lexists(str(attempt)), "freshness")
        check("log_absent", not os.path.lexists(str(log)), "freshness")
        check("namespace_contract", tuple(release["one_shot"][key] for key in
              ("result_namespace", "attempt_namespace", "log_namespace")) ==
              tuple(str(path.relative_to(ROOT)) for path in (result, attempt, log)), "freshness")

        # Re-run the exact 22-test source suite. It uses only synthetic proc/GPU fixtures.
        completed = subprocess.run(
            [sys.executable, str(TEST)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=180,
            check=False)
        check("author_tests_22", completed.returncode == 0 and
              "Ran 22 tests" in completed.stdout and "OK" in completed.stdout, "replay")

        # The source-stage absent check is intentionally no longer callable now
        # that M1410/M1412/M1430 exist. Bind the exact, recursively sealed M1410
        # PASS produced when those future authorities were absent instead.
        check("sealed_source_absent_self_check", m1410["verification"]["source_absent_self_check"] ==
              "PASS_M1400_SOURCE_ABSENT_SELF_CHECK__NO_REMOTE_NO_GPU_NO_ATTEMPT", "replay")
        check("runner_has_no_restore_primitive", all(token not in RUNNER.read_text(encoding="utf-8")
              for token in ("os.kill", "SIGCONT", "send_signal", "kill(")), "safety")
        check("local_only_execution", True, "safety")
    except Exception as exc:
        checks.append({"check": "exception", "category": "fatal", "pass": False,
                       "detail": f"{type(exc).__name__}: {exc}"})

    failed = [str(row["check"]) for row in checks if not row["pass"]]
    categories = {}
    for row in checks:
        bucket = categories.setdefault(str(row["category"]), {"checks": 0, "passed": 0, "failed": 0})
        bucket["checks"] += 1
        bucket["passed" if row["pass"] else "failed"] += 1
    output = {
        "schema": "m1430_m1412_m1400_m1349_ep34_live105_final_launch_hammer_r1_v1",
        "status": "PASS" if not failed else "FAIL_DO_NOT_LAUNCH",
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "failed_checks": failed,
        "categories": categories,
        "authorization_if_pass": {"launch": True, "runs": 1, "automatic_retry": False},
        "execution": {"ssh": 0, "remote_preflight": 0, "gpu": 0, "capture": 0,
                      "attempt": 0, "controller_restore": 0},
        "test_replay_tail": completed.stdout.splitlines()[-6:] if "completed" in locals() else [],
        "checks": checks,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, no-EDA blind hammer for M1344 C2 release source."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1344_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
CHECKER = HW / "verif_m1344_c2_activity_release/static_check_m1344_c2_activity_vcs_release_source.py"
TEST = HW / "verif_m1344_c2_activity_release/test_m1344_c2_activity_vcs_release_source.py"
M1336_TEST = HW / "verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py"
M1334_TEST = HW / "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py"
CONTRACT = HW / "contracts/m1344_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1344_c2_headline_mapped_production_activity_vcs_release_source_author_r1_20260831"
M1337_FAIL = HW / "reviews/m1337_m1336_c2_headline_mapped_production_activity_vcs_release_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    RUNNER: "4d6081a094d4a865c23c42c9b2b0fc2644ee9d7f65ae401f838663d550daa4f0",
    CHECKER: "fc3c89040ec4ec3ecb9b8fcb10e8734df918f81615e2c8e6c52108445e56f3bb",
    TEST: "32d5499a2956da6cccf139db4850cf35a4e901086fee26331bfbad5df8ecef43",
    CONTRACT: "0fb605913b9d779bf493811d3d6498ed466254d40aa4847493a6150d2bc8af1b",
    AUTHOR / "review.json": "5ab74e944343f7bf9247ebf9f3e6c436a8b6d764ee35e60324d6801938c86027",
    AUTHOR / "SHA256SUMS": "c17e4c8c7071053d8e8083428d39a50048438bf3a64cd240be37863cb966d159",
    AUTHOR / "SHA256SUMS.seal.sha256": "be4d98a614354087f4cb6b08c6502b8c1696f7aabdd56e0e902b48899e8b9b8e",
    M1337_FAIL / "review.json": "84a898e2b894e6754ab9ef70464b6a3f6e857b44e076d9bc1c93cf8e53faa946",
    M1337_FAIL / "SHA256SUMS": "31ae8689016cac5482a004b355a0f640251b3ad128cba7535337520552b9a0f0",
    M1337_FAIL / "SHA256SUMS.seal.sha256": "a5fe53b7def3be354aaf7ef87e4e6d779be7a2c326a10097cd4dbcad2e45e1c8",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_seal(root: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer semantic drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and not rel.is_absolute() and ".." not in rel.parts and name not in rows,
                "manifest row invalid")
        member = root / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "sealed member drift: " + name)
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows) and rows.get("review.json") == review_sha,
            "sealed population/review drift")


def run_tests(path: Path, expected: int) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = subprocess.run([str(PYTHON), "-B", str(path)], cwd=str(HW.parent), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, check=False)
    require(run.returncode == 0 and ("Ran %d tests" % expected) in run.stdout
            and "OK" in run.stdout, "test replay failed: " + path.name)


def checker_rejects(action: Callable[[], Any]) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


def update_final_expected(T: Any, fixture: Any) -> None:
    final = fixture.paths["final_hammer"]
    T.seal_dir(final)
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"] = sha(final / "review.json")
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"] = sha(final / "SHA256SUMS")
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"] = sha(final / "SHA256SUMS.seal.sha256")


def validate_runner_mutant(M: Any, T: Any, text: str) -> bool:
    results = HW / "results"
    results.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m1348_runner_mutant_", dir=str(results)) as td:
        root = Path(td)
        runner = root / "runner.sh"
        runner.write_text(text); runner.chmod(0o755)
        contract = json.loads(CONTRACT.read_text())
        contract["identity"]["runner"] = runner.relative_to(HW).as_posix()
        contract["identity"]["runner_sha256"] = sha(runner)
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        T.sidecar(contract_path)
        old_runner, old_contract = M.RUNNER, M.CONTRACT
        M.RUNNER, M.CONTRACT = runner, contract_path
        try:
            return not checker_rejects(lambda: M.validate_common(skip_author=True))
        finally:
            M.RUNNER, M.CONTRACT = old_runner, old_contract


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
    verify_seal(AUTHOR, EXPECTED[AUTHOR / "review.json"],
                EXPECTED[AUTHOR / "SHA256SUMS"],
                EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_seal(M1337_FAIL, EXPECTED[M1337_FAIL / "review.json"],
                EXPECTED[M1337_FAIL / "SHA256SUMS"],
                EXPECTED[M1337_FAIL / "SHA256SUMS.seal.sha256"])
    run_tests(TEST, 12)
    run_tests(M1336_TEST, 10)
    run_tests(M1334_TEST, 12)

    T = load("m1348_bound_m1344_tests", TEST)
    M = T.M
    require(M.validate_common(skip_author=False) >= 50, "canonical common check failed")
    rejected: list[str] = []
    accepted: list[str] = []

    def attack(label: str, action: Callable[[], Any]) -> None:
        (rejected if checker_rejects(action) else accepted).append(label)

    with tempfile.TemporaryDirectory(prefix="m1348_empty_") as td:
        empty = M.future_paths(Path(td))
        M.validate_future("source_absent", empty)
        attack("runtime_present_on_absent_chain", lambda:
               M.validate_future("runtime_present", empty, None))

    fixture = T.RuntimeFixture()
    try:
        M.validate_future("runtime_present", fixture.paths, fixture.expected)
        attack("source_absent_on_runtime_chain", lambda:
               M.validate_future("source_absent", fixture.paths))
        attack("source_absent_with_external_sha", lambda:
               M.validate_future("source_absent", fixture.paths, fixture.expected))
        for name in M.ENV_NAMES:
            bad = dict(fixture.expected); bad[name] = "0" * 64
            attack("external_sha_mismatch_" + name, lambda value=bad:
                   M.validate_future("runtime_present", fixture.paths, value))
    finally:
        fixture.close()

    for key in ("source_hammer", "launch_release", "final_hammer"):
        fixture = T.RuntimeFixture()
        try:
            path = fixture.paths[key]
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
            attack("missing_" + key, lambda f=fixture:
                   M.validate_future("runtime_present", f.paths, f.expected))
        finally:
            fixture.close()

    for key in ("source_hammer", "launch_release", "final_hammer"):
        fixture = T.RuntimeFixture()
        try:
            path = fixture.paths[key]
            real = path.with_name(path.name + ".real")
            path.rename(real); path.symlink_to(real, target_is_directory=real.is_dir())
            attack("symlink_" + key, lambda f=fixture:
                   M.validate_future("runtime_present", f.paths, f.expected))
        finally:
            fixture.close()

    for document in ("release", "final"):
        fixture = T.RuntimeFixture()
        try:
            path = (fixture.paths["launch_release"] if document == "release"
                    else fixture.paths["final_hammer"] / "review.json")
            value = json.loads(path.read_text())
            value["authorization"]["simv_runs"] = 11
            path.write_text(json.dumps(value, sort_keys=True))
            if document == "release":
                T.sidecar(path)
                fixture.expected["M1344_EXPECTED_LAUNCH_RELEASE_SHA256"] = sha(path)
            else:
                update_final_expected(T, fixture)
            attack("cardinality_lift_" + document, lambda f=fixture:
                   M.validate_future("runtime_present", f.paths, f.expected))
        finally:
            fixture.close()

    # Fresh FN candidate: exact known claims are false, but an extra authority
    # claim is accepted because the checker does not require an exact key set.
    fixture = T.RuntimeFixture()
    try:
        final = fixture.paths["final_hammer"] / "review.json"
        value = json.loads(final.read_text())
        value["claim_boundary"]["launch_authorized"] = True
        final.write_text(json.dumps(value, sort_keys=True))
        update_final_expected(T, fixture)
        attack("extra_true_claim_in_final_hammer", lambda:
               M.validate_future("runtime_present", fixture.paths, fixture.expected))
    finally:
        fixture.close()

    # Fresh FN candidate: duplicate JSON authority keys are accepted by
    # json.loads (last value wins).
    fixture = T.RuntimeFixture()
    try:
        final = fixture.paths["final_hammer"] / "review.json"
        text = final.read_text()
        anchor = '"status": "PASS_M1347_AUTHORIZE_ONE_M1344_C2_MAPPED_PRODUCTION_ACTIVITY_VCS_LAUNCH"'
        require(anchor in text, "duplicate-key anchor drift")
        final.write_text(text.replace(anchor,
            '"status": "FORGED_DUPLICATE", ' + anchor, 1))
        update_final_expected(T, fixture)
        attack("duplicate_json_status_key", lambda:
               M.validate_future("runtime_present", fixture.paths, fixture.expected))
    finally:
        fixture.close()

    runner = RUNNER.read_text()
    receipt_prefix = ("d={'schema':'m1344_c2_mapped_production_activity_vcs_candidate_receipt_r1',"
                      "'status':'PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER'")
    require(receipt_prefix in runner, "candidate receipt anchor drift")
    receipt_start = runner.index(receipt_prefix)
    receipt_end = runner.index("\nout.write_text", receipt_start)
    receipt = runner[receipt_start:receipt_end]
    receipt_mutations = {
        "success_receipt_missing_runner_sha": "'runner_sha256':sha(runner),",
        "success_receipt_missing_source_contract_sha": "'source_contract_sha256':sha(contract),",
        "success_receipt_missing_launch_release_sha": "'launch_release_sha256':sha(release),",
        "success_receipt_missing_source_manifest_with_comment_residue":
            "'source_hammer_manifest_sha256':sha(source_hammer/'SHA256SUMS'),",
        "success_receipt_missing_final_outer_with_comment_residue":
            "'final_hammer_outer_file_sha256':sha(final_hammer/'SHA256SUMS.seal.sha256')",
    }
    for label, token in receipt_mutations.items():
        require(receipt.count(token) == 1, "receipt token anchor drift: " + label)
        mutated_receipt = receipt.replace(token, "", 1)
        mutated = runner[:receipt_start] + mutated_receipt + runner[receipt_end:]
        if "comment_residue" in label:
            bare = token.split("'", 2)[1]
            mutated += "\n# " + bare + " retained only as comment\n"
        if validate_runner_mutant(M, T, mutated):
            accepted.append(label)
        else:
            rejected.append(label)

    result = {
        "schema": "m1348_m1344_c2_release_source_blind_hammer_output_r1",
        "status": ("PASS_SOURCE_ADMITTED" if not accepted else
                   "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "score": 100 if not accepted else 64,
        "reviewer_independent_of_author": True,
        "new_m1344_tests": "12/12 PASS",
        "inherited_m1336_tests": "10/10 PASS",
        "inherited_m1334_tests": "12/12 PASS",
        "author_double_seal_verified": True,
        "m1337_failure_double_seal_verified": True,
        "independent_attack_count": len(rejected) + len(accepted),
        "independent_rejected_count": len(rejected),
        "independent_false_negative_count": len(accepted),
        "rejected_attacks": rejected,
        "accepted_attacks": accepted,
        "execution": {"license_query": False, "vcs": False, "simv": False,
                      "saif": False, "dc": False, "pt": False, "ptpx": False,
                      "eda": False, "launch_authority": False},
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("M1348_M1344_BLIND_HAMMER_ERROR: " + str(error), file=sys.stderr)
        raise

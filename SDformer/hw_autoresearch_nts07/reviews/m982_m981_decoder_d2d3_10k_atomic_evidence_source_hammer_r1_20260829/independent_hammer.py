#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M982 source hammer; never executes a real decoder prefix."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
CONTRACT = HW / "contracts/m981_m977_decoder_d2d3_10k_atomic_evidence_source_contract_r1_20260829.json"
DRIVER = HW / "system_simulator/scripts/execute_m981_m977_decoder_d2d3_10k_atomic_evidence_r1.py"
RUNNER = HW / "system_simulator/scripts/run_m985_m981_decoder_d2d3_10k_atomic_evidence_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m981_m977_decoder_atomic_evidence_source.py"
TESTS = HW / "system_simulator/tests/test_m981_m977_decoder_atomic_evidence_source.py"
M946 = HW / "system_simulator/scripts/analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
M896 = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
RECEIPT = HW / "reviews/m981_m977_decoder_atomic_evidence_source_receipt_r1_20260829"
RESULT = HW / "results/m985_m981_decoder_d2d3_10k_atomic_evidence_r1_20260829"
ATTEMPT = HW / "results/.m985_m981_decoder_d2d3_10k_atomic_evidence_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "contract": "61462516602d129f7a444603efce053691625d57c000177973c28b76fae5c1db",
    "driver": "dfd626e292077efc1d447ceb870a5c113e531c2086b0001ccbffbf1ec8ff86b2",
    "runner": "591809cd30f1f1b63ebc2fe3ebb9f9ad17a8b45cc931bda59f6764fb3dbb4be2",
    "checker": "13d46601b3ea6d255fba315194f518b5c4272ef8ad35e7491c35eaf0e59bbe6f",
    "tests": "ea5463e23b79dd40b7af66262cf94933debb205516517262b49b2f02dc47e049",
    "m946": "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac",
    "m896": "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    "receipt_review": "a0bd74dff1f62dfc8db9f00898bbbd794ba3bed129811261b0603ddd1d5c3d60",
    "receipt_manifest": "5431d9e16644658edf2c46a0007b72ab47f6fc65c89a186a5c8e0b6c4091bb17",
    "receipt_outer": "5de7306c719c4568a08630f225d74bab2a9e469933c576def56175f5f9f66e8c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_directory(directory, expected_manifest, expected_outer):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "receipt missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "receipt seals missing/symlink")
    require(sha(manifest) == expected_manifest and sha(outer) == expected_outer,
            "receipt seal identity drift")
    require(outer.read_text().split() == [expected_manifest, "SHA256SUMS"],
            "receipt outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate receipt path")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "receipt member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "receipt recursive exact-set drift")
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "receipt contains symlink")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def reproduce_attempt_retry_bypass(driver):
    """Interrupt the random attempt stage, then consume a second attempt."""
    original_attempt = driver.ATTEMPT
    original_result = driver.RESULT
    original_atomic_seal = driver.atomic_seal
    with tempfile.TemporaryDirectory(prefix="m982_attempt_retry_") as temporary:
        root = Path(temporary)
        driver.RESULT = root / "m985_result"
        driver.ATTEMPT = root / ".m985_attempt_consumed"
        stage1 = root / (driver.ATTEMPT.name + ".stage.111.1.1")
        stage2 = root / (driver.ATTEMPT.name + ".stage.222.2.2")
        authority = {"release_sha256": "synthetic-release",
                     "release_hammer_review_sha256": "synthetic-release-hammer"}

        def interrupt_after_manifest(directory, inject_fault=""):
            return original_atomic_seal(directory, "after_manifest")

        driver.atomic_seal = interrupt_after_manifest
        interrupted = False
        try:
            driver.consume_attempt(stage1, authority)
        except RuntimeError as error:
            interrupted = "interruption after manifest" in str(error)
        finally:
            driver.atomic_seal = original_atomic_seal

        first_receipt = stage1 / "attempt.json"
        partial_stages = driver.partial_seal_stages(stage1)
        canonical_absent_after_interrupt = not driver.ATTEMPT.exists()
        # This is the exact shell freshness shape for a later invocation: it
        # tests canonical result/attempt and only its newly randomized stage.
        next_exact_namespace_fresh = (
            not driver.RESULT.exists() and not driver.ATTEMPT.exists() and
            not stage2.exists()
        )
        second = driver.consume_attempt(stage2, authority)
        driver.verify_atomic_seal(driver.ATTEMPT)
        second_receipt = driver.ATTEMPT / "attempt.json"
        reproduced = (interrupted and first_receipt.is_file() and
                      bool(partial_stages) and canonical_absent_after_interrupt and
                      next_exact_namespace_fresh and second_receipt.is_file() and
                      stage1.is_dir() and second["receipt"]["max_attempts"] == 1)
        evidence = {
            "interruption_observed": interrupted,
            "first_random_stage_retained": stage1.is_dir(),
            "first_attempt_receipt_retained": first_receipt.is_file(),
            "first_partial_seal_stage_count": len(partial_stages),
            "canonical_attempt_absent_after_interrupt": canonical_absent_after_interrupt,
            "later_random_stage_passes_exact_freshness": next_exact_namespace_fresh,
            "second_attempt_consumed_and_published": second_receipt.is_file(),
            "two_attempt_receipts_exist": first_receipt.is_file() and second_receipt.is_file(),
            "p0_reproduced": reproduced,
        }
    driver.ATTEMPT = original_attempt
    driver.RESULT = original_result
    driver.atomic_seal = original_atomic_seal
    return evidence


def main():
    require(Path(sys.executable).resolve() == PYTHON and sha(PYTHON) == EXPECTED["python"],
            "exact Python drift")
    require(tuple(sys.version_info[:3]) == (3, 10, 18), "Python version drift")
    paths = {"contract": CONTRACT, "driver": DRIVER, "runner": RUNNER,
             "checker": CHECKER, "tests": TESTS, "m946": M946,
             "m896": M896, "docs359": DOC359}
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), key + " missing/symlink")
        require(sha(path) == EXPECTED[key], key + " SHA drift")
    receipt_seal = verify_directory(
        RECEIPT, EXPECTED["receipt_manifest"], EXPECTED["receipt_outer"])
    require(sha(RECEIPT / "review.json") == EXPECTED["receipt_review"],
            "M981 receipt review drift")
    receipt_review = json.loads((RECEIPT / "review.json").read_text())
    require(receipt_review["status"] ==
            "PASS_M981_ATOMIC_EVIDENCE_SOURCE_ONLY__NO_REAL_10K" and
            receipt_review["tests"]["real_10k_runs"] == 0,
            "M981 receipt scope drift")

    contract = json.loads(CONTRACT.read_text())
    require(contract["launch_now"] is False and contract["max_future_attempts"] == 1 and
            contract["authorization"]["retry"] is False,
            "M981 source authority drift")
    require(contract["canonical"]["source_hammer"].startswith(
            "hw_autoresearch_nts07/reviews/m982_"), "M982 chain drift")

    static_run = subprocess.run(
        [str(PYTHON), "-I", str(CHECKER), "--contract", str(CONTRACT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        timeout=30)
    unit_run = subprocess.run(
        [str(PYTHON), "-I", str(TESTS)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False, timeout=30)
    require(static_run.returncode == 0 and
            "PASS_M981_STATIC_SOURCE__NO_REAL_10K" in static_run.stdout,
            "M981 checker failed")
    require(unit_run.returncode == 0 and "Ran 7 tests" in unit_run.stderr and
            "OK" in unit_run.stderr, "M981 unit tests failed")

    before = (not RESULT.exists() and not RESULT.is_symlink() and
              not ATTEMPT.exists() and not ATTEMPT.is_symlink())
    inert = subprocess.run(
        [str(RUNNER)], cwd=str(HW.parent),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=10)
    after = (not RESULT.exists() and not RESULT.is_symlink() and
             not ATTEMPT.exists() and not ATTEMPT.is_symlink())
    require(before and after and inert.returncode == 1 and
            "M985 inert until M983 release" in inert.stderr,
            "M985 inert runner/freshness failed")

    runner_text = RUNNER.read_text()
    require('m985_attempt_stage="${m985_attempt}.stage.$$.$RANDOM.$RANDOM"' in
            runner_text, "random attempt stage token drift")
    require('[[ ! -e "${m985_result}"' in runner_text and
            '! -e "${m985_attempt}"' in runner_text and
            '! -e "${m985_work}"' in runner_text,
            "exact freshness token drift")
    require('M985_ATTEMPT_STAGE_RETAINED_NOT_MOVED' in runner_text,
            "attempt-stage retention token drift")
    require('find ' not in runner_text and 'flock ' not in runner_text and
            'mkdir -- "${m985_attempt}"' not in runner_text,
            "expected missing stale-stage/canonical-claim guard not reproduced")

    driver = load_module("m982_m981_driver", DRIVER)
    attack = reproduce_attempt_retry_bypass(driver)
    require(attack["p0_reproduced"], "attempt retry bypass not reproduced")

    return {
        "schema": "m982_m981_decoder_d2d3_10k_atomic_evidence_source_hammer_v1",
        "status": "STOP_M982_M981_ATTEMPT_CONSUMPTION_NOT_FAIL_CLOSED",
        "verdict": "STOP",
        "score_out_of_100": 82,
        "p0_count": 1, "p1_count": 0, "p2_count": 0,
        "pins": {key + "_sha256": EXPECTED[key] for key in
                 ("contract", "driver", "runner", "checker", "tests",
                  "m946", "m896", "docs359")},
        "m981_receipt_seal": receipt_seal,
        "positive": {
            "static_checker": "PASS_M981_STATIC_SOURCE__NO_REAL_10K",
            "unit_tests": "7/7 PASS",
            "source_receipt_recursive_exact_set": "PASS",
            "frozen_m946_m896_docs359": "PASS",
            "correct_m981_to_m985_numbering": "PASS",
            "atomic_payload_seal_tests": "PASS",
            "inert_runner_exit_code": inert.returncode,
            "real_10k_executed": False,
            "eda_gpu_remote_runs": 0,
        },
        "p0": {
            "id": "P0_INTERRUPTED_RANDOM_ATTEMPT_STAGE_ALLOWS_SECOND_ATTEMPT",
            "evidence": attack,
            "cause": ("The irreversible canonical ATTEMPT is claimed only after a random "
                      "stage is written and sealed; cleanup retains interrupted stages, "
                      "while later freshness ignores every stale .attempt.stage.* path."),
            "impact": ("A pre-D2 interruption can leave one consumed attempt receipt while "
                       "a second invocation consumes and publishes another, violating "
                       "max_future_attempts=1 and retry=false."),
            "required_repair": ("Atomically mkdir the canonical ATTEMPT as the irreversible "
                                "consumption point before writing/sealing it. Any interruption "
                                "must permanently block retry and preserve forensic payload."),
        },
        "decision": {
            "m983_release_authoring_authorized": False,
            "m985_real_10k_authorized": False,
            "automatic_retry": False,
            "legal_successor_chain": ["M994 source repair", "M995 source hammer",
                                      "M996 release", "M997 release hammer",
                                      "M998 sole D2-then-D3 10K run"],
        },
        "scope": {"source_modified": False, "real_10k_executed": False,
                  "eda_gpu_remote_runs": 0, "docs359_modified": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
                           "table_a_row": False, "system_speedup": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))

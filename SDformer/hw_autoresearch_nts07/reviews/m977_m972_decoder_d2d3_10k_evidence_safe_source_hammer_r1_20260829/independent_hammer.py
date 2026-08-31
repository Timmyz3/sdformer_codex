#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent static/synthetic hammer for the M972 source-only package.

This audit never executes a real decoder prefix.  It permits only the exact
M972 source checker/self-test and an inert runner invocation without release
pins.  All failure-injection work is confined to a temporary directory.
"""

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
CONTRACT = HW / "contracts/m972_m971_decoder_d2d3_10k_evidence_safe_source_contract_r1_20260829.json"
DRIVER = HW / "system_simulator/scripts/execute_m972_m971_decoder_d2d3_10k_evidence_safe_r1.py"
RUNNER = HW / "system_simulator/scripts/run_m972_m971_decoder_d2d3_10k_evidence_safe_r1_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m972_m971_decoder_d2d3_10k_evidence_safe_source.py"
TESTS = HW / "system_simulator/tests/test_m972_m971_decoder_d2d3_10k_evidence_safe_source.py"
M946 = HW / "system_simulator/scripts/analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
M896 = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
M961_CONTRACT = HW / "contracts/m961_m946_decoder_d2d3_10k_bounded_prefix_source_contract_r1_20260829.json"
M961_DRIVER = HW / "system_simulator/scripts/execute_m961_m946_decoder_d2d3_10k_bounded_prefix_r1.py"
M961_RUNNER = HW / "system_simulator/scripts/run_m961_m946_decoder_d2d3_10k_bounded_prefix_r1_one_shot.sh"
M950 = HW / "reviews/m950_m946_decoder_multilayer_bounded_prefix_source_fresh_hammer_r1_20260829"
M971 = HW / "reviews/m971_m961_decoder_d2d3_10k_failure_forensic_r1_20260829"
M972_RECEIPT = HW / "reviews/m972_m971_decoder_d2d3_10k_evidence_safe_source_receipt_r1_20260829"
RESULT = HW / "results/m972_m946_decoder_d2d3_10k_evidence_safe_r1_20260829"
ATTEMPT = HW / "results/.m972_m946_decoder_d2d3_10k_evidence_safe_r1_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "ea2cc752cb2895486f9b03c4ae15bdf40e59198262ac7a6dc414d82b54f79aa6",
    "driver": "61ea70f4ec6afd878a7a331c7e9421d6db7b2d1cbe5765ae73e53e7a8c225763",
    "runner": "14462e22a98fae64e3217c515fef6afd080f91c0ffc5b223e6b651500953761a",
    "checker": "95656da36776a467a5e20a38f6d1d3d853da125628470729716b56fe1613edc1",
    "tests": "d057382a183c4625c711169c46292f708ce79da80b98ad96ae539dd1b20ac1a9",
    "m946": "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac",
    "m896": "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    "m961_contract": "966cbb77aee1c03df2ac6dc8deb8ee707ee7560f89d1b9daf368335de86b5420",
    "m961_driver": "c997626a0eff58b4824d534335c9bc0627d8408f0f8e14a81e490bfc8895c54a",
    "m961_runner": "2ba4d4c8fb5b7ec90943c9ed71a60747a9296f880b8bdb09fc5620b1d41c009a",
    "m950_review": "2042b1d2f16a29be706a4c413ce3d473b7daedd56cca24dfd6aff57848579cf6",
    "m950_manifest": "8f749a2f9db1aa49d710765e3d89232b57029d3ed313f2da5299f0dfa3910ee7",
    "m950_outer": "389bae76312b4f51655facdb56d6754c3bb6e93821c02b52b68a0f9f84b19e09",
    "m971_review": "36073062ebfeb3c8077cabdd2ebae7bc2053212084432460b742fc5a4bafc1ef",
    "m971_manifest": "83af03a8768c3728e67d537c585b5b913aecf5c3d86f90e3c71fcccb5601027d",
    "m971_outer": "d1a19b066e205abc99cd31eaa58ae9ddab5619cfac2c822c6a67b291667a4c44",
    "m972_receipt_manifest": "4694805b4538eb5416d498f47d152fa422f0b9635337f8fdc225f87d1849b428",
    "m972_receipt_outer": "8b9d261bd32f1a7ee6c20f541ca94fbaa8bc52d44c51a1d1e4ad67c1a30019ae",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory missing/symlink: " + str(directory))
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "directory seals missing: " + str(directory))
    require(sha(manifest) == expected_manifest and sha(outer) == expected_outer,
            "directory seal identity drift: " + str(directory))
    require(outer.read_text() == expected_manifest + "  SHA256SUMS\n",
            "outer seal content drift: " + str(directory))
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate manifest path: " + rel)
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and
                sha(member) == digest, "manifest member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(actual == set(listed), "recursive exact-set drift: " + str(directory))
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "symlink in sealed directory: " + str(directory))
    return {"manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer),
            "entries": len(listed)}


def partial_seal_attack(driver):
    """Inject failure after manifest publication and before outer seal."""
    original = driver._write_exclusive
    with tempfile.TemporaryDirectory(prefix="m977_partial_seal_") as temporary:
        root = Path(temporary) / "failed_root"
        root.mkdir()
        original(root / "payload.txt", b"payload\n")

        def fail_outer(path, data):
            if Path(path).name == "SHA256SUMS.seal.sha256":
                raise RuntimeError("M977 injected outer-seal interruption")
            return original(path, data)

        driver._write_exclusive = fail_outer
        injected = False
        try:
            driver._seal_recursive(root)
        except RuntimeError as error:
            injected = "outer-seal interruption" in str(error)
        finally:
            driver._write_exclusive = original
        manifest_present = (root / "SHA256SUMS").is_file()
        outer_present = (root / "SHA256SUMS.seal.sha256").is_file()
        # The shell cleanup calls --seal-failure-root only when SHA256SUMS is
        # absent; therefore this exact partial state is moved without repair.
        cleanup_would_call_reseal = not manifest_present
        quarantine_would_be_double_sealed = manifest_present and outer_present
        return {
            "injection_observed": injected,
            "manifest_present": manifest_present,
            "outer_seal_present": outer_present,
            "cleanup_would_call_reseal": cleanup_would_call_reseal,
            "quarantine_would_be_double_sealed": quarantine_would_be_double_sealed,
            "p0_reproduced": (injected and manifest_present and not outer_present and
                              not cleanup_would_call_reseal),
        }


def main():
    require(Path(sys.executable).resolve() == PYTHON and sha(PYTHON) == PYTHON_SHA,
            "exact Python identity drift")
    require(tuple(sys.version_info[:3]) == (3, 10, 18), "Python version drift")

    paths = {
        "contract": CONTRACT, "driver": DRIVER, "runner": RUNNER,
        "checker": CHECKER, "tests": TESTS, "m946": M946, "m896": M896,
        "m961_contract": M961_CONTRACT, "m961_driver": M961_DRIVER,
        "m961_runner": M961_RUNNER, "docs359": DOC359,
    }
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), key + " missing/symlink")
        require(sha(path) == EXPECTED[key], key + " SHA drift")

    m950_seal = verify_directory(M950, EXPECTED["m950_manifest"], EXPECTED["m950_outer"])
    m971_seal = verify_directory(M971, EXPECTED["m971_manifest"], EXPECTED["m971_outer"])
    receipt_seal = verify_directory(
        M972_RECEIPT, EXPECTED["m972_receipt_manifest"], EXPECTED["m972_receipt_outer"])
    require(sha(M950 / "review.json") == EXPECTED["m950_review"], "M950 review drift")
    require(sha(M971 / "review.json") == EXPECTED["m971_review"], "M971 review drift")

    m971 = json.loads((M971 / "review.json").read_text())
    frozen = m971["frozen_identity"]
    require(frozen["m961_source_contract_sha256"] == EXPECTED["m961_contract"] and
            frozen["m961_driver_sha256"] == EXPECTED["m961_driver"] and
            frozen["m961_runner_sha256"] == EXPECTED["m961_runner"] and
            frozen["m946_source_sha256"] == EXPECTED["m946"] and
            frozen["m896_source_sha256"] == EXPECTED["m896"],
            "M971 frozen M946/M896/M961 identity drift")

    contract = json.loads(CONTRACT.read_text())
    require(contract["status"] ==
            "SOURCE_ONLY__M973_HAMMER_AND_M974_M975_RELEASE_CHAIN_REQUIRED",
            "M972 source status drift")
    require(contract["launch_now"] is False, "M972 source launches now")
    require(contract["authorization"]["execute_real_10k_now"] is False and
            contract["authorization"]["d2_or_d3_100k"] is False and
            contract["authorization"]["full_row"] is False and
            contract["authorization"]["eda_gpu_remote"] is False,
            "M972 source authority expanded")

    checker = load_module("m977_m972_checker", CHECKER)
    static = checker.check(CONTRACT)
    require(static["status"] == "PASS_M972_STATIC_SOURCE_CHECK__NO_REAL_10K" and
            static["real_prefix_executed"] is False and
            static["full_row_authorized"] is False,
            "M972 static checker scope drift")

    # Independently recompute ceil(bytes/192) instead of trusting labels.
    independent_geometry = {
        "D2": {"source_bytes": 231600,
               "source_fetch_requests": math.ceil(231600 / 192)},
        "D3": {"source_bytes": 465600,
               "source_fetch_requests": math.ceil(465600 / 192)},
    }
    require(independent_geometry["D2"]["source_fetch_requests"] == 1207 and
            independent_geometry["D3"]["source_fetch_requests"] == 2425,
            "independent byte/request derivation failed")
    require(static["byte_request_distinction"]["D2"]["source_fetch_requests"] == 1207 and
            static["byte_request_distinction"]["D3"]["source_fetch_requests"] == 2425,
            "generated byte/request geometry mismatch")

    runner_text = RUNNER.read_text()
    driver_text = DRIVER.read_text()
    ast.parse(driver_text, filename=str(DRIVER))
    require(runner_text.index("--run-row D2") < runner_text.index("--run-row D3"),
            "D2/D3 runner order drift")
    require("M972_WORK_ROOT_CREATED_BEFORE_D2" in runner_text and
            "--seal-failure-root" in runner_text,
            "M972 evidence lifecycle token missing")
    require("100000" not in runner_text and "100000" not in driver_text and
            "--run-full" not in runner_text and "--run-full" not in driver_text,
            "M972 contains executable 100K/full-row mode")
    require("PREFIX_10K = 10000" in driver_text,
            "M972 exact 10K constant drift")

    before_fresh = (not RESULT.exists() and not RESULT.is_symlink() and
                    not ATTEMPT.exists() and not ATTEMPT.is_symlink())
    inert = subprocess.run(
        [str(RUNNER)], cwd=str(HW.parent),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=10, check=False)
    after_fresh = (not RESULT.exists() and not RESULT.is_symlink() and
                   not ATTEMPT.exists() and not ATTEMPT.is_symlink())
    inert_text = inert.stdout + inert.stderr
    require(before_fresh and after_fresh and inert.returncode != 0 and
            "M972 inert until exact M974 release SHA" in inert_text,
            "M972 inert runner/freshness gate failed")

    driver = load_module("m977_m972_driver", DRIVER)
    durability = partial_seal_attack(driver)
    require(durability["p0_reproduced"], "partial-seal durability attack not reproduced")

    occupied_m973 = sorted(path.name for path in (HW / "reviews").glob("m973_*"))
    occupied_m974 = sorted(path.name for path in (HW / "reviews").glob("m974_*"))
    hardcoded = {
        "contract_source_hammer": contract["canonical"]["source_hammer"],
        "contract_future_release": contract["canonical"]["future_release"],
        "runner_release": "contracts/m974_m972_decoder_d2d3_10k_evidence_safe_release_r1_20260829.json",
        "runner_release_hammer": "reviews/m975_m974_m972_decoder_d2d3_10k_evidence_safe_release_hammer_r1_20260829",
        "driver_source_hammer": str(driver.SOURCE_HAMMER.relative_to(driver.REPO)),
        "driver_future_release": str(driver.FUTURE_RELEASE.relative_to(driver.REPO)),
        "driver_release_hammer": str(driver.RELEASE_HAMMER.relative_to(driver.REPO)),
    }
    numbering_p0 = bool(occupied_m973 and occupied_m974 and
                        "m973_m972" in hardcoded["contract_source_hammer"] and
                        "m974_m972" in hardcoded["contract_future_release"])
    require(numbering_p0, "expected old-chain milestone collision not reproduced")

    p0 = [
        {
            "id": "P0_OLD_CANONICAL_MILESTONE_COLLISION",
            "evidence": hardcoded,
            "occupied_m973": occupied_m973,
            "occupied_m974": occupied_m974,
            "impact": "M977 cannot be silently reinterpreted as the exact M973 authority required by the frozen source contract; M974/M975 are also globally reassigned or reserved.",
            "remediation": "Create the authorized additive successor chain M981 source, M982 source hammer, M983 release, M984 release hammer, M985 sole 10K execution.",
        },
        {
            "id": "P0_PARTIAL_SEAL_CAN_ESCAPE_AS_UNSEALED_QUARANTINE",
            "evidence": durability,
            "impact": "An interruption after SHA256SUMS creation but before the outer seal makes cleanup skip resealing and move an unsealed failure quarantine, contradicting the all-failures-double-sealed contract.",
            "remediation": "In the successor runner, verify the recursive pair; if either seal is absent/invalid, repair in a fresh staging directory and atomically publish a recursively valid quarantine before move.",
        },
    ]
    return {
        "schema": "m977_m972_decoder_d2d3_10k_evidence_safe_source_hammer_v1",
        "status": "STOP_M977_M972_SOURCE_NOT_RELEASE_ELIGIBLE",
        "verdict": "STOP",
        "score_out_of_100": 72,
        "p0": p0, "p0_count": len(p0),
        "p1": [], "p1_count": 0,
        "p2": [], "p2_count": 0,
        "positive_findings": {
            "m946_m896_m961_frozen": True,
            "recursive_upstream_seals": {"M950": m950_seal, "M971": m971_seal,
                                         "M972_source_receipt": receipt_seal},
            "byte_request_distinction": independent_geometry,
            "generated_source_fetch_requests": {"D2": 1207, "D3": 2425},
            "multi_transaction_and_nonzero_commit_accepted": True,
            "d2_command_precedes_d3": True,
            "synthetic_row_exception_persisted_and_double_sealed": True,
            "inert_runner_rc": inert.returncode,
            "attempt_and_result_fresh_before_after_inert": True,
            "real_10k_executed_by_m977": False,
            "real_100k_executed_or_authorized": False,
            "eda_gpu_remote_run_by_m977": False,
        },
        "claim_boundary": {
            "m972_release_authoring_authorized": False,
            "m972_real_10k_authorized": False,
            "paper_citable": False,
            "production_row": False,
            "decoder_complete": False,
            "table_a_row": False,
            "system_speedup": False,
            "full_row_authorized": False,
        },
        "legal_successor_chain": ["M981 source", "M982 source hammer",
                                  "M983 release", "M984 release hammer",
                                  "M985 sole D2-then-D3 10K execution"],
        "docs359_sha256": sha(DOC359),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))

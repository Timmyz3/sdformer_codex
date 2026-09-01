#!/usr/bin/env python3
"""Static, zero-EDA checks for the M1768 Python 3.12 wrapper source."""
import hashlib
import json
import os
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
WRAPPER = HW / "dc_handoff/scripts/run_m1768_m1753_c2_python312_wrapper_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1768_m1753_c2_python312_wrapper_source.py"
CONTRACT = HW / "contracts/m1768_m1767_m1753_c2_python312_wrapper_source_contract_r1_20260902.json"
M1767 = HW / "reviews/m1767_m1761_m1753_c2_python36_preparse_failure_receipt_r1_20260902"
M1753 = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
M1753_CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
PYTHON312 = Path("/usr/bin/python3.12")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

FIXED = {
    WRAPPER: "3a19b42593c22d5b756e9584e5cfd6a94fab9e03e614735f6db230fa1da4443c",
    M1767 / "receipt.json": "330e533d0f545439b7b0539a0c4816e8e77a6c89330620571162ec060c6b3729",
    M1767 / "SHA256SUMS": "ed9fbb6e5a3b30e77b81f74ee64861231336576ce562517a7a999f518c26d474",
    M1767 / "SHA256SUMS.seal.sha256": "80aaf88a542ed1fb9e754172d722ec8cdd7741bfe399ce02be212c30c60f2b71",
    M1753: "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e",
    M1753_CONTRACT: "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21",
    M1760 / "review.json": "987fccddbad6281bb31aa128987118ef4942e210d47201c528ab9be50055329c",
    M1760 / "SHA256SUMS": "e8921f4612f9b0b8532b43f441ccd2b93c2600e5dca861cefcb6ef293601afcf",
    M1760 / "SHA256SUMS.seal.sha256": "55caca70cf9670ee8e361c062f4c73e1272c399c990eb4b1e27771008f00830e",
    M1761: "bb5b32ead4bd2ff682abfbcedf242b645c20c89d71db0b7eeadc7c18f5191f5e",
    Path(str(M1761) + ".sha256"): "71b353b92b87c559b8c6501e4b8834e4c78383c5e29d1a367b1ae277423f7e3d",
    Path(str(M1761) + ".sha256.seal.sha256"): "47df6661b49128152093bce9cfaacf9868e4288d4647011f1b654f4783095074",
    PYTHON312: "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CLAIMS = dict((key, False) for key in (
    "launch_authorized", "launch_executed", "wrapper_attempt",
    "m1753_attempt", "mapped_vcs", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def validate_source_text(text):
    need(text.startswith("#!/usr/bin/python3.12\n"), "exact shebang")
    need(text.count("os.execve(") == 1, "one execve")
    need("subprocess" not in text, "subprocess forbidden")
    need("/opt/anaconda3" not in text and "python3.10" not in text,
         "borrowed interpreter forbidden")
    for marker in (
            'PYTHON312 = Path("/usr/bin/python3.12")',
            'PYTHON312_SHA = "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814"',
            'platform.python_version() != "3.12.13"',
            'M1768_EXPECTED_WRAPPER_SHA256',
            'M1768_EXPECTED_SOURCE_CONTRACT_SHA256',
            'M1768_EXPECTED_M1769_REVIEW_SHA256',
            'M1768_EXPECTED_M1769_MANIFEST_SHA256',
            'M1768_EXPECTED_M1769_OUTER_FILE_SHA256',
            'M1768_EXPECTED_M1770_RELEASE_SHA256',
            'M1753_EXPECTED_M1761_RELEASE_SHA256',
            'publish_no_replace(stage, ATTEMPT)',
            'os.execve(str(PYTHON312), [str(PYTHON312), str(M1753)], dict(os.environ))'):
        need(marker in text, "source marker: " + marker)
    main = text[text.index("def main():"):]
    order = [main.index(marker) for marker in (
        "verify_interpreter()", "verify_authority()", "namespaces_fresh()",
        "stage.mkdir()", "seal_dir(stage)", "publish_no_replace(stage, ATTEMPT)",
        "os.execve(")]
    need(order == sorted(order) and len(set(order)) == len(order), "execution order")
    need(main.index("publish_no_replace(stage, ATTEMPT)") < main.index("os.execve("),
         "attempt before execve")


def validate_contract(contract):
    need(contract.get("schema") ==
         "m1768_m1767_m1753_c2_python312_wrapper_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1767_FAILURE_BOUND__PYTHON312_WRAPPER__M1769_REVIEW_AND_M1770_RELEASE_REQUIRED__NO_EXECUTION",
         "contract status")
    identity = contract.get("source_identity", {})
    need(identity.get("wrapper_sha256") == FIXED[WRAPPER], "wrapper identity")
    need(identity.get("checker_sha256") == sha(CHECKER), "checker identity")
    need(identity.get("test_sha256") == sha(TEST), "test identity")
    interpreter = contract.get("interpreter_identity", {})
    need(interpreter == {
        "path": "/usr/bin/python3.12",
        "resolved_path": "/usr/bin/python3.12",
        "implementation": "CPython",
        "version": "3.12.13",
        "binary_sha256": FIXED[PYTHON312]}, "interpreter identity")
    authority = contract.get("bound_authority", {})
    need(authority.get("m1767_receipt_sha256") == FIXED[M1767 / "receipt.json"]
         and authority.get("m1767_manifest_file_sha256") == FIXED[M1767 / "SHA256SUMS"]
         and authority.get("m1767_outer_seal_file_sha256") == FIXED[M1767 / "SHA256SUMS.seal.sha256"]
         and authority.get("m1753_runner_sha256") == FIXED[M1753]
         and authority.get("m1753_source_contract_sha256") == FIXED[M1753_CONTRACT]
         and authority.get("m1760_review_sha256") == FIXED[M1760 / "review.json"]
         and authority.get("m1761_release_sha256") == FIXED[M1761],
         "bound authority")
    future = contract.get("future_authority", {})
    need(future == {
        "different_author_review_required": True,
        "review": "reviews/m1769_m1768_m1753_c2_python312_wrapper_source_hammer_r1_20260902",
        "review_status": "PASS_M1769_M1768_C2_PYTHON312_WRAPPER_SOURCE_HAMMER__AUTHORIZE_ONE_WRAPPER_ATTEMPT",
        "release": "contracts/m1770_m1769_m1768_m1753_c2_python312_wrapper_launch_release_r1_20260902.json",
        "release_status": "AUTHORIZE_ONE_M1768_C2_PYTHON312_WRAPPER_ATTEMPT"},
        "future authority")
    need(contract.get("future_budget") == {
        "m1768_wrapper_attempts": 1,
        "underlying_m1753_campaigns": 1,
        "automatic_retry": False}, "future budget")
    need(contract.get("claim_boundary") == CLAIMS, "claim boundary")
    need(contract.get("author_execution") == {
        "wrapper_runs": 0, "m1753_runs": 0, "license_queries": 0,
        "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
        "ptpx_runs": 0, "wrapper_attempts_created": 0,
        "m1753_attempts_created": 0, "results_created": 0},
        "author execution")


def validate_sources():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    validate_source_text(WRAPPER.read_text())
    validate_contract(strict_json(CONTRACT))
    for path in (
            HW / "results/.m1768_m1753_c2_python312_wrapper_attempt_consumed",
            HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.private_build.unsealed_do_not_cite"):
        need(not os.path.lexists(str(path)), "execution namespace exists")


def main():
    validate_sources()
    print(json.dumps({
        "schema": "m1768_m1753_c2_python312_wrapper_source_check_r1_v1",
        "status": "PASS_M1768_PYTHON312_WRAPPER_SOURCE_ONLY_NO_EXECUTION",
        "python_path": "/usr/bin/python3.12",
        "python_version": "3.12.13",
        "python_sha256": FIXED[PYTHON312],
        "atomic_wrapper_attempt_before_execve": True,
        "future_m1769_review_required": True,
        "future_m1770_release_required": True,
        "wrapper_runs": 0,
        "m1753_runs": 0,
        "eda_runs": 0,
        "license_queries": 0,
        "attempt_created": False,
        "result_created": False,
        "claim_boundary": CLAIMS}, sort_keys=True))


if __name__ == "__main__":
    main()

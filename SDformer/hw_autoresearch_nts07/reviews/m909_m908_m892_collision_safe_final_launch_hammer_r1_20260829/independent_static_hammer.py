#!/usr/bin/env python3
"""No-EDA static/fault-injection hammer for the additive M908 repair."""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/m909_m908_m892_collision_safe_final_launch_hammer_r1_20260829"
CONTRACT = ROOT / "contracts/m908_m892_dc_collision_safe_launch_contract_r1_20260829.json"
WRAPPER = ROOT / "dc_handoff/scripts/run_dc_m908_m892_collision_safe_exact_sha_r1.sh"
SHIM = ROOT / "dc_handoff/scripts/m908_collision_safe_path/rg"
M907 = ROOT / "reviews/m907_m892_same_uid_dc_collision_precheck_failure_audit_r1_20260829"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
RESULT = ROOT / "dc_handoff/runs/m892_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = ROOT / "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_attempt_consumed"
LOCK = ROOT / "dc_handoff/runs/.m892_m528_r21_macro_aware_product_dc_launch_lock"
PATTERN = r"(^|/)(dc_shell|dc_shell-t)([[:space:]]|$)|common_shell_exec -shell dc_shell"

EXPECTED = {
    CONTRACT: "dc6916a22073336417e7044907b5bc80d9bacf1ffb8845bc51acb463b3f5621c",
    WRAPPER: "cb0a12ca773cf15a13626104ce9053dd10fdbfc089b0323bf02edb6dc734666a",
    SHIM: "f8766318a8450d318f1a0cf86c7f57d51e6ce1318faf60f924b5a2d7c4f265fc",
    M907 / "review.json": "98f054ccf348b6bc135a3f579b2d4946646c6f53e1869335cb25584d9729035d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: %s" % key)
        out[key] = value
    return out


def nonfinite(value):
    raise ValueError("non-finite JSON value: %s" % value)


def strict_json(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"),
                          object_pairs_hook=unique, parse_constant=nonfinite)


def verify_tree(path):
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(path),
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=str(path), stdout=subprocess.DEVNULL)


for artifact, expected in EXPECTED.items():
    require(artifact.is_file() and not artifact.is_symlink(), "missing identity %s" % artifact)
    require(sha(artifact) == expected, "identity mismatch %s" % artifact)

verify_tree(M907)
subprocess.check_call(["sha256sum", "-c", CONTRACT.name + ".sha256"],
                      cwd=str(CONTRACT.parent), stdout=subprocess.DEVNULL)
subprocess.check_call(["sha256sum", "-c", CONTRACT.name + ".sha256.seal.sha256"],
                      cwd=str(CONTRACT.parent), stdout=subprocess.DEVNULL)
contract = strict_json(CONTRACT)
require(contract["schema"] == "m908_m892_dc_collision_safe_launch_contract_v1",
        "wrong contract schema")
require(contract["authorization"] == {
    "inherits_m892_attempt": True, "max_new_attempts": 0,
    "max_total_m892_attempts": 1, "run_dc": True,
    "run_formality": False, "run_pt": False, "run_ptpx": False,
    "run_remote": False, "run_saif": False, "run_vcs": False,
}, "authorization widened")
require(contract["collision_guard"]["argv_regex_used"] is False,
        "argv regex still admitted")
require(contract["collision_guard"]["proc_executable_or_comm_exact_match"] is True,
        "exact proc identity missing")
for relative, expected in contract["exact_files"].items():
    artifact = pathlib.Path(relative) if relative.startswith("/") else ROOT / relative
    require(sha(artifact) == expected, "contract exact-file mismatch %s" % relative)

wrapper_text = WRAPPER.read_text()
shim_text = SHIM.read_text()
subprocess.check_call(["bash", "-n", str(WRAPPER)])
subprocess.check_call(["bash", "-n", str(SHIM)])
require("M908_EXPECTED_WRAPPER_SHA256" in wrapper_text and
        "M908_EXPECTED_CONTRACT_SHA256" in wrapper_text and
        "M908_EXPECTED_FINAL_REVIEW_SHA256" in wrapper_text,
        "caller SHA pins missing")
require("[[ ! -e \"${m908_result}\" && ! -e \"${m908_attempt}\"" in wrapper_text,
        "pre-attempt one-shot identity gate missing")
require("m908_collision_present" in wrapper_text and "/proc/${m908_pid}/exe" not in wrapper_text,
        "unexpected collision implementation")
require("readlink -f \"${m908_proc}/exe\"" in wrapper_text and
        "<\"${m908_proc}/comm\"" in wrapper_text,
        "proc executable/comm identity reads missing")
require("/cmdline" not in wrapper_text and "-o args=" not in wrapper_text,
        "wrapper uses argv collision matching")
require("/cmdline" not in shim_text and "-o args=" not in shim_text,
        "shim uses argv collision matching")
require("exec /usr/bin/rg \"$@\"" in shim_text,
        "non-collision rg delegation missing")
require("exec env" in wrapper_text and "${m908_inner}" in wrapper_text,
        "sealed inner runner handoff missing")
require(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
        "M892 attempt/result/lock unexpectedly present before hammer")

# The old gate must reproduce its self-match without any Synopsys process.
old = subprocess.run(
    ["bash", "-c", "ps -u $(id -u) -o args= | /usr/bin/rg -q \"$1\"", "_", PATTERN],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
require(old.returncode == 0, "old argv-regex self-match did not reproduce")

# The shim must ignore arbitrary stdin that contains the pattern and report a
# clean host when no exact Synopsys executable/comm exists.
clean = subprocess.run([str(SHIM), "-q", PATTERN], input=(PATTERN + "\n").encode(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
require(clean.returncode == 1, "clean exact guard did not return no-match")

# A non-EDA /bin/sleep copy named dc_shell exercises the positive identity path.
with tempfile.TemporaryDirectory(prefix="m909_fake_identity_") as temp:
    fake = pathlib.Path(temp) / "dc_shell"
    shutil.copy2("/bin/sleep", fake)
    fake.chmod(0o755)
    proc = subprocess.Popen([str(fake), "30"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.1)
        collision = subprocess.run([str(SHIM), "-q", PATTERN], input=b"",
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        require(collision.returncode == 0,
                "exact fake dc_shell executable was not detected")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

delegated = subprocess.run([str(SHIM), "-Fxq", "m909_delegate"],
                           input=b"m909_delegate\n", stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
require(delegated.returncode == 0, "ordinary rg delegation failed")
require(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
        "hammer consumed/created M892 one-shot state")
require(sha(DOCS359) == EXPECTED[DOCS359], "docs359 changed")

review = {
    "schema": "m909_m908_m892_collision_safe_final_launch_hammer_v1",
    "status": "PASS100_M908_ADDITIVE_COLLISION_SAFE_FINAL_LAUNCH_HAMMER",
    "date": "2026-08-29",
    "verdict": "PASS",
    "score_100": 100,
    "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
    "identity": {
        "contract_sha256": EXPECTED[CONTRACT],
        "wrapper_sha256": EXPECTED[WRAPPER],
        "shim_sha256": EXPECTED[SHIM],
        "m907_audit_sha256": EXPECTED[M907 / "review.json"],
    },
    "fault_injection": {
        "old_argv_regex_self_match_reproduced": True,
        "clean_exact_guard_returned_no_collision": True,
        "fake_non_eda_dc_shell_executable_detected": True,
        "ordinary_rg_delegation_passed": True,
    },
    "decision": {
        "exactly_one_existing_m892_attempt_authorized": True,
        "new_attempts_authorized": 0,
        "launch_through_m908_wrapper_only": True,
        "dc_started_by_hammer": False,
        "license_query_started_by_hammer": False,
    },
    "claim_boundary": contract["claim_boundary"],
    "docs359_sha256": EXPECTED[DOCS359],
}
(OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True,
                                             allow_nan=False) + "\n")
(OUT / "review.md").write_text(
    "# M909 — M908 collision-safe final launch hammer\n\n"
    "**PASS 100/100**, P0/P1/P2 = 0/0/0.\n\n"
    "The old argv-regex self-match was reproduced. The additive guard returned "
    "clean on a clean host, detected a non-EDA executable named `dc_shell`, and "
    "delegated ordinary ripgrep behavior exactly. No EDA, license query, attempt "
    "token, lock, or result was created. The original M892 release remains the "
    "sole one-attempt authority.\n")
print("PASS100 M909 additive collision-safe final launch hammer")

#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent, read-only M1250/R11 release hammer (executing task M1254).

This script performs no VCS/simv/EDA/GPU/remote action and does not invoke the
production runner.  It rechecks immutable identities, recursive seals, source
checkers and adversarial release mutations before a separate review file can
authorize the single future invocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1250_m1232r11_m1162_c1_common_charge_protocol_exact_sha_r11.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs.f"
SOURCE_CONTRACT = HW / "contracts/m1250_m1247_m1246_c1_r11_vcs_release_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1250_m1247_m1246_c1_r11_vcs_launch_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1250_m1247_m1246_c1_r11_vcs_release_author_r1_20260830"
M1246_AUTHOR = HW / "reviews/m1246_m1242_c1_r11_checker_hardening_source_author_r1_20260830"
M1247 = HW / "reviews/m1247_m1246_c1_r11_checker_tests_independent_hammer_r1_20260830"
R11_CHECKER = HW / "verif_m1232r11_c1_common_charge_protocol/check_m1232r11_source.py"
R11_TESTS = HW / "verif_m1232r11_c1_common_charge_protocol/test_m1232r11_source.py"
RELEASE_CHECKER = HW / "verif_m1250_c1_r11_vcs_release/static_check_m1250_c1_r11_vcs_release_source.py"
RELEASE_TESTS = HW / "verif_m1250_c1_r11_vcs_release/test_m1250_c1_r11_vcs_release_source.py"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_attempt_consumed"
RESULT = RESULTS / "m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830"

EXPECTED = {
    RUNNER: "39edc512856693516f6bcf145ca184eb5254aa44bc7231d0d063e811f1b4393e",
    FILELIST: "87cc365423baa9cc2b99f9e2eac3f5a836fc8007f7136df2b671600437cab08e",
    SOURCE_CONTRACT: "820a3485da22cbf09815d8e41c540d14b8ad2230045b538c07418c8b39eb8172",
    RELEASE: "a0baab304bf7fa6fa250e6411387779b6426633efd7e51385186137fccb0a0ee",
    AUTHOR / "review.json": "db0e37a9e2333bcdad0d6753ede97ebf0b403f4a79f0f9247e95d59d31cccfe1",
    AUTHOR / "SHA256SUMS": "6f3cf8576bcc6b7c9665392052ee9d17f35359b1aaaa75a9379a6b9cb44bd6a3",
    AUTHOR / "SHA256SUMS.seal.sha256": "86cc477d55cb1ffe9eb37ece1587ba611150f22a3bbbb67a5d39444880784f58",
    M1246_AUTHOR / "SHA256SUMS": "a67fdce5307e7a35e84e162aa748404b1c56cccfc902458e1679b09b1acc1c52",
    M1246_AUTHOR / "SHA256SUMS.seal.sha256": "ff6d811aa64078feea2ad01fb30b19f10edf550d8da550ddb850939e5473f144",
    M1247 / "review.json": "32bdfcdafe3039eb9e44f318c2133e997cb182227fc0c18367d3ba9393bc807b",
    M1247 / "SHA256SUMS": "8440f0f6111f6df9df1cfe0f85847fc2743ce2b9cd6f857c33d2581fa6ec0132",
    M1247 / "SHA256SUMS.seal.sha256": "b9eb60767d829ecb0bde4e95bacc73c533c4da764702541803c69a4fa062c57d",
    R11_CHECKER: "154860a16dfa3e2175653e81c14db645da3718af2c8d659c35299d80248e68fd",
    R11_TESTS: "de89c87210e8782d38b84b8202d229a418ebb153583a02043f4080e25aac4605",
    RELEASE_CHECKER: "bd58cc1b9aa8a52c146702c0d44138cec0b5aadf101be9be52dfa45444aedc8f",
    RELEASE_TESTS: "4fb53108405cee7ed43cd64aa3ad95ec9750411f0ec6f6212356d82d4b35649b",
    HW / "verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv": "850881df0212a9461e47e36b6829a993b9cf25af2c9faa3b7921e08fa141c776",
    HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sums.read_text().split() == [sha(path), path.name], "bad sidecar: " + str(path))
    require(outer.read_text().split() == [sha(sums), sums.name], "bad outer sidecar: " + str(path))


def verify_dir(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), "not sealed dir: " + str(root))
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"], "outer mismatch: " + str(root))
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "unsafe manifest member")
        listed[name] = digest
    actual: set[str] = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink():
                continue
            if stat.S_ISREG(path.lstat().st_mode):
                actual.add(rel)
    require(actual == set(listed), "sealed population mismatch: " + str(root))
    for name, digest in listed.items():
        require(sha(root / name) == digest, "member drift: " + str(root / name))


def executable_lines(text: str, needle: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if needle in line and line.strip() and not line.lstrip().startswith("#")]


def audit_runner(text: str) -> list[str]:
    errors: list[str] = []
    def exact_raw(needle: str, count: int, label: str) -> None:
        if text.count(needle) != count:
            errors.append(label)
    def need(needle: str, label: str) -> None:
        if needle not in text:
            errors.append(label)

    exact_raw('"${VCS_BIN}" -full64', 1, "exactly one raw compile token")
    exact_raw('./simv -no_save', 1, "exactly one raw sim token")
    if len(executable_lines(text, '"${VCS_BIN}" -full64')) != 1:
        errors.append("exactly one executable compile")
    if len(executable_lines(text, './simv -no_save')) != 1:
        errors.append("exactly one executable simulation")
    exact_raw('/usr/bin/timeout --signal=TERM --kill-after=30s', 2, "two independent timeout invocations")
    if len(executable_lines(text, '/usr/bin/timeout --signal=TERM --kill-after=30s')) != 2:
        errors.append("two executable timeout invocations")
    need('COMPILE_TIMEOUT_SECONDS=1200', "compile timeout 1200")
    need('SIM_TIMEOUT_SECONDS=1800', "sim timeout 1800")
    need('"${COMPILE_TIMEOUT_SECONDS}s" \\\n  "${VCS_BIN}" -full64', "compile timeout directly guards VCS")
    need('"${SIM_TIMEOUT_SECONDS}s" ./simv -no_save', "sim timeout directly guards simv")
    exact_raw('/bin/mkdir -- "${ATTEMPT}"', 1, "single atomic attempt mkdir")
    exact_raw('/bin/mkdir -- "${WORK}"', 1, "single work mkdir")
    if '/bin/mkdir -p' in text or 'mkdir -p "${ATTEMPT}"' in text:
        errors.append("attempt is not O_EXCL mkdir")
    for needle, label in (
        ('ATTEMPT="${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_attempt_consumed"', "exact attempt namespace"),
        ('RESULT="${HW_ROOT}/results/m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830"', "exact result namespace"),
        ('WORK="${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_work.$$"', "fresh PID work namespace"),
        ('QUARANTINE="${RESULT}.failed_or_incomplete.$$.quarantine"', "unique quarantine namespace"),
        ('seal_dir "${WORK}" || true', "failure seal attempt"),
        ('mv -- "${WORK}" "${QUARANTINE}" || true', "failure quarantine move"),
        ("seal_dir \"${WORK}\"\nmv -- \"${WORK}\" \"${RESULT}\"",
         "success seal before canonical move"),
        ("automatic_retry=false", "no retry receipt"),
        ("if rg -qi '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\\$error|\\$fatal)([^[:alnum:]_]|$)' compile.log sim.log; then exit 35; fi", "Error/Fatal/Assertion rejection"),
        ('for phase in DIRECTED RESET_PENDING STICKY_ATTACKS SERVICE_ATTACKS RANDOM NORMAL_M935 CLEAN_RESET_PREP; do', "seven phases"),
        ('for index in $(seq 0 23); do', "24 random transactions"),
        ('normal_m935_rows=1 normal_m935_tasks=1', "normal M935 coverage"),
        ('COVERAGE_M1219R9_PROTOCOL weight_first=1', "protocol cover line"),
        ('COVERAGE_M1219R9_RESETS_ATTACKS reset_partial=1', "reset/attack cover line"),
        ('COVERAGE_M1219R9_SERVICE_ASSUMPTIONS weight_payload_mutation=1', "service cover line"),
        ('COVERAGE_M1219R9_FROZEN_M935 normal_issues=2', "M935 cover line"),
        ('[[ "$(rg -c "^${PASS_TOKEN} " sim.log || true)" -eq 1 ]]', "single exact PASS token"),
        ('if rg -q \'^TIMEOUT_M1219R9 \' sim.log; then exit 34; fi', "internal timeout rejection"),
        ('functional_vcs_verified=false', "failure claim boundary"),
    ):
        need(needle, label)
    for pin in ("M1250_EXPECTED_RELEASE_SHA256",
                "M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
                "M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
                "M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256"):
        if text.count(pin) < 3:
            errors.append("external digest pin absent/underused: " + pin)
    if "automatic_retry=true" in text or re.search(r"(^|\s)rm\s+-", text):
        errors.append("retry/destructive cleanup")
    try:
        author = text.index('for sealed in "${M1246_AUTHOR}"')
        hammer = text.index('sha_exact "${M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
        fresh = text.index('[[ ! -e "${ATTEMPT}"')
        attempt = text.index('/bin/mkdir -- "${ATTEMPT}"')
        work = text.index('/bin/mkdir -- "${WORK}"')
        compile_at = text.index('"${VCS_BIN}" -full64')
        sim_at = text.index('./simv -no_save')
        gates = text.index('for phase in DIRECTED')
        canonical = text.index('mv -- "${WORK}" "${RESULT}"')
        if not (author < hammer < fresh < attempt < work < compile_at < sim_at < gates < canonical):
            errors.append("fail-closed ordering")
    except ValueError:
        errors.append("ordering anchor absent")
    return errors


def changed_once(text: str, old: str, new: str) -> str:
    require(text.count(old) >= 1, "mutation anchor absent: " + old[:60])
    return text.replace(old, new, 1)


def mutation_suite(canonical: str) -> list[dict[str, object]]:
    mutations: list[tuple[str, str]] = []
    mutations.append(("compile_comment_decoy", canonical + '\n# "${VCS_BIN}" -full64 decoy\n'))
    mutations.append(("sim_comment_decoy", canonical + '\n# ./simv -no_save decoy\n'))
    mutations.append(("compile_disabled_with_decoy", changed_once(canonical, '  "${VCS_BIN}" -full64', '  # "${VCS_BIN}" -full64')))
    mutations.append(("sim_disabled_with_decoy", changed_once(canonical, './simv -no_save 2>&1', '# ./simv -no_save 2>&1')))
    mutations.append(("compile_timeout_removed", changed_once(canonical, '/usr/bin/timeout --signal=TERM --kill-after=30s', '/usr/bin/env', )))
    mutations.append(("compile_timeout_changed", changed_once(canonical, 'COMPILE_TIMEOUT_SECONDS=1200', 'COMPILE_TIMEOUT_SECONDS=1201')))
    mutations.append(("sim_timeout_changed", changed_once(canonical, 'SIM_TIMEOUT_SECONDS=1800', 'SIM_TIMEOUT_SECONDS=1801')))
    mutations.append(("sim_uses_compile_timeout", changed_once(canonical, '"${SIM_TIMEOUT_SECONDS}s" ./simv', '"${COMPILE_TIMEOUT_SECONDS}s" ./simv')))
    mutations.append(("attempt_mkdir_p", changed_once(canonical, '/bin/mkdir -- "${ATTEMPT}"', '/bin/mkdir -p -- "${ATTEMPT}"')))
    mutations.append(("attempt_result_alias", changed_once(canonical, 'ATTEMPT="${HW_ROOT}/results/.m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_attempt_consumed"', 'ATTEMPT="${HW_ROOT}/results/m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830"')))
    mutations.append(("work_namespace_not_pid_unique", changed_once(canonical, '_vcs_r11_work.$$"', '_vcs_r11_work"')))
    mutations.append(("quarantine_namespace_not_unique", changed_once(canonical, '.failed_or_incomplete.$$.quarantine"', '.failed_or_incomplete.quarantine"')))
    mutations.append(("failure_seal_removed", changed_once(canonical, 'seal_dir "${WORK}" || true', 'true # seal removed')))
    mutations.append(("failure_quarantine_removed", changed_once(canonical, 'mv -- "${WORK}" "${QUARANTINE}" || true', 'true # quarantine removed')))
    mutations.append(("success_seal_removed", changed_once(
        canonical, 'seal_dir "${WORK}"\nmv -- "${WORK}" "${RESULT}"',
        'true # success seal removed\nmv -- "${WORK}" "${RESULT}"')))
    mutations.append(("error_gate_removed", canonical.replace('Error|Fatal|Assertion', 'Warning')))
    mutations.append(("phase_population_reduced", changed_once(canonical, ' RANDOM NORMAL_M935 CLEAN_RESET_PREP', ' RANDOM CLEAN_RESET_PREP')))
    mutations.append(("random_population_reduced", changed_once(canonical, 'for index in $(seq 0 23); do', 'for index in $(seq 0 22); do')))
    mutations.append(("normal_gate_relaxed", changed_once(canonical, 'normal_m935_rows=1 normal_m935_tasks=1', 'normal_m935_rows=0 normal_m935_tasks=0')))
    mutations.append(("protocol_cover_relaxed", changed_once(canonical, 'COVERAGE_M1219R9_PROTOCOL weight_first=1', 'COVERAGE_M1219R9_PROTOCOL weight_first=0')))
    mutations.append(("pass_gate_removed", changed_once(canonical, '[[ "$(rg -c "^${PASS_TOKEN} " sim.log || true)" -eq 1 ]]', 'true # PASS gate removed')))
    mutations.append(("timeout_gate_removed", changed_once(canonical, "if rg -q '^TIMEOUT_M1219R9 ' sim.log; then exit 34; fi", 'true # internal timeout accepted')))
    mutations.append(("retry_enabled", canonical + '\nautomatic_retry=true\n'))
    mutations.append(("destructive_cleanup", canonical + '\nrm -rf "${WORK}"\n'))
    mutations.append(("hammer_review_pin_removed", changed_once(canonical, 'M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256', 'M1250_EXPECTED_UNUSED_SHA256')))
    mutations.append(("attempt_after_work", changed_once(canonical, '/bin/mkdir -- "${ATTEMPT}"\n', '') .replace('/bin/mkdir -- "${WORK}"\n', '/bin/mkdir -- "${WORK}"\n/bin/mkdir -- "${ATTEMPT}"\n', 1)))
    rows = []
    for name, text in mutations:
        errors = audit_runner(text)
        require(errors, "mutation accepted: " + name)
        rows.append({"name": name, "rejected": True, "first_error": errors[0]})
    return rows


def run_source_tool(path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([str(PYTHON), "-I", str(path)], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, check=False, env=env)


def no_live_runner() -> bool:
    own = {os.getpid(), os.getppid()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in own:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if RUNNER.name in argv:
            return False
    return True


def main() -> int:
    checks = 0
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
        checks += 1
    for path in (SOURCE_CONTRACT, RELEASE,
                 HW / "contracts/m1246_m1242_m1239_c1_r11_checker_source_contract_r1_20260830.json"):
        verify_sidecar(path)
        checks += 2
    for root in (M1246_AUTHOR, M1247, AUTHOR):
        verify_dir(root)
        checks += 1
    source_contract = json.loads(SOURCE_CONTRACT.read_text())
    release = json.loads(RELEASE.read_text())
    author = json.loads((AUTHOR / "review.json").read_text())
    prior = json.loads((M1247 / "review.json").read_text())
    require(source_contract["status"] == "M1250_C1_R11_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1251_REQUIRED__NO_VCS_NO_EDA", "source status")
    require(release["status"] == "AUTHORIZE_ONE_M1250_R11_UNIT_DELAY_VCS_ATTEMPT_AFTER_FRESH_M1251_HAMMER", "release status")
    require(author["status"] == "PASS_M1250_R11_ONE_SHOT_RELEASE_SOURCE__FRESH_M1251_HAMMER_REQUIRED", "author status")
    require(prior["status"] == "PASS_M1247_RELEASE_AUTHORING_GO" and prior["score"] == 100
            and prior["p0_count"] == prior["p1_count"] == prior["p2_count"] == 0, "M1247 authority")
    checks += 4
    runner_text = RUNNER.read_text()
    require(not audit_runner(runner_text), "canonical runner audit: " + repr(audit_runner(runner_text)))
    checks += 1
    expected_filelist = [
        "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v",
        str(HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"),
        str(HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"),
        str(HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"),
        str(HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"),
        str(HW / "verif_m1232r11_c1_common_charge_protocol/tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv"),
    ]
    require(FILELIST.read_text().splitlines() == expected_filelist, "filelist exact order/population")
    checks += 1
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    require(syntax.returncode == 0, "runner bash syntax")
    checks += 1
    tool_results = {}
    for label, path in (("m1246_checker", R11_CHECKER), ("m1246_tests", R11_TESTS),
                        ("m1250_checker", RELEASE_CHECKER), ("m1250_tests", RELEASE_TESTS)):
        done = run_source_tool(path)
        require(done.returncode == 0, label + " failed: " + done.stderr[-500:])
        tool_results[label] = {"returncode": done.returncode,
                               "stdout_tail": done.stdout[-1000:],
                               "stderr_tail": done.stderr[-1500:]}
        checks += 1
    require("Ran 24 tests" in tool_results["m1246_tests"]["stderr_tail"], "24 inherited tests absent")
    require("Ran 12 tests" in tool_results["m1250_tests"]["stderr_tail"], "12 author release tests absent")
    require('"checks_passed": 75' in tool_results["m1250_checker"]["stdout_tail"], "75 author checks absent")
    checks += 3
    mutations = mutation_suite(runner_text)
    checks += len(mutations)
    require(not os.path.lexists(ATTEMPT) and not os.path.lexists(RESULT), "attempt/result namespace already consumed")
    require(not list(RESULTS.glob(".m1250_m1232r11_m1162_c1_common_charge_protocol_vcs_r11_work.*")), "stale work namespace")
    require(not list(RESULTS.glob("m1250_m1232r11_m1162_c1_common_charge_protocol_unit_delay_vcs_r11_20260830.failed_or_incomplete.*")), "stale quarantine namespace")
    require(no_live_runner(), "M1250 runner already live")
    checks += 4
    for data in (source_contract, release, author):
        for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                    "ppa", "power", "energy", "system_speedup", "paper_citable"):
            require(data["claim_boundary"][key] is False, "inflated claim: " + key)
            checks += 1
    evidence = {
        "schema": "m1251_m1250_c1_r11_vcs_release_independent_hammer_mechanical_r1_v1",
        "executing_reviewer_task": "M1254",
        "status": "PASS_READ_ONLY_RELEASE_HAMMER_NO_VCS_NO_EDA",
        "checks_passed": checks,
        "identity_count": len(EXPECTED),
        "author_static_checks_replayed": 75,
        "author_release_mutations_replayed": 12,
        "inherited_checker_tests_replayed": 24,
        "additional_independent_release_mutations": len(mutations),
        "additional_mutations": mutations,
        "tool_results": tool_results,
        "fresh_namespace": True,
        "attempt_consumed": False,
        "live_runner": False,
        "vcs_runs": 0,
        "simv_runs": 0,
        "all_eda_runs": 0,
        "authorization_shape": {"vcs_compiles": 1, "simv_runs": 1,
                                "all_other_eda_runs": 0, "automatic_retry": False},
        "timeouts_seconds": {"compile": 1200, "simulation": 1800},
        "docs359_sha256": sha(DOCS359),
    }
    out = Path(__file__).with_name("mechanical_checks.json")
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: evidence[k] for k in (
        "schema", "status", "checks_passed", "author_static_checks_replayed",
        "author_release_mutations_replayed", "inherited_checker_tests_replayed",
        "additional_independent_release_mutations", "fresh_namespace", "attempt_consumed",
        "vcs_runs", "simv_runs", "all_eda_runs")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

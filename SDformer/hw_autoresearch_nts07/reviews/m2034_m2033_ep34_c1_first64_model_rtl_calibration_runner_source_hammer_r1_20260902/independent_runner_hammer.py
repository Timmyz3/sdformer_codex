#!/usr/bin/env python3
"""Independent static/mutation hammer for the exact M2033 VCS runner."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m2033_m2031_ep34_c1_first64_model_rtl_calibration_one_shot.sh"
M2032 = HW / "reviews/m2032_m2031_ep34_c1_first64_model_rtl_calibration_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "7a3f7340955edcdb5eb68e28c1b92a6fbf3f2fe2baeba8037f254978322ea41d"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED = {
    RUNNER: RUNNER_SHA,
    M2032 / "review.json": "f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
    M2032 / "SHA256SUMS": "987ff979ecaa505bcce1027fd4c3b255e3bad92fea23c812b50891d07dee8927",
    M2032 / "SHA256SUMS.seal.sha256": "103e6971327a58c8d800fac51d2c566f78c1ebd5a9e2a62494d7b01071032c25",
    DOCS359: DOCS359_SHA,
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def verify_seal(directory, expected_rows):
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    outer_digest, outer_name = outer.read_text().strip().split(maxsplit=1)
    require(outer_name.lstrip(" *") == "SHA256SUMS", "outer target")
    require(outer_digest == sha(manifest), "outer digest")
    rows = manifest.read_text().splitlines()
    require(len(rows) == expected_rows, "manifest row count")
    listed = set()
    for row in rows:
        digest, relative = row.split(maxsplit=1)
        relative = relative.lstrip(" *")
        require(relative not in listed, "duplicate manifest member")
        listed.add(relative)
        target = directory / relative
        require(target.is_file() and not target.is_symlink(), "manifest target")
        require(sha(target) == digest, "manifest payload digest")


def section(text, begin, end):
    left = text.index(begin)
    right = text.index(end, left)
    return text[left:right]


def validate(text):
    checks = {}
    checks["clean_exact_bash_shebang"] = text.startswith(
        "#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash\n")
    checks["startup_environment_scrub"] = (
        "unset BASH_ENV ENV CDPATH GLOBIGNORE" in text and
        "export PATH=/usr/bin:/bin LANG=C LC_ALL=C" in text)
    checks["strict_shell_and_umask"] = "set -euo pipefail" in text and "umask 077" in text
    checks["zero_arguments"] = "[[ $# -eq 0 ]]" in text
    checks["exact_input_pins"] = all(token in text for token in (
        "require_sha 726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
        "require_sha 8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
        "require_sha 8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1",
        "require_sha 4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3",
        "require_sha c3937a5d069f56cee3bd641eda0b78777acda8c15aae54e8650360e1105c485a",
        "require_sha f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
        "require_sha 8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
        "require_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
        "require_sha 873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
        "require_sha " + DOCS359_SHA))
    checks["predecessor_and_future_double_seals"] = (
        'verify_double_seal "${M2032_DIR}"' in text and
        'verify_double_seal "${M2034_DIR}"' in text)

    auth = section(text, '"${PYTHON}" -I - "${RUNNER}"', "audit_output=")
    checks["runner_self_sha_via_review_and_release"] = all(token in auth for token in (
        "runner_sha=sha(runner)",
        "review.get('runner_sha256') != runner_sha",
        "release.get('runner_sha256') != runner_sha"))
    checks["review_gate"] = all(token in auth for token in (
        "PASS_M2034_M2033_RUNNER_SOURCE_HAMMER",
        "review.get('score', 0) < 90",
        "review.get('severity_counts', {}).get('P0') != 0"))
    checks["release_gate"] = all(token in auth for token in (
        "AUTHORIZED_EXACTLY_ONE_M2033_VCS_COMPILE_AND_SIM",
        "release.get('review_sha256') != sha(review_path)",
        "release.get('result_path', '')",
        "release.get('attempt_path', '')",
        "{'vcs_compile_runs': 1, 'simv_runs': 1, 'automatic_retry': False}"))

    namespace = '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${STAGE}" ]]'
    attempt = 'mkdir -- "${ATTEMPT}"'
    stage = 'mkdir -- "${STAGE}"'
    compile_start = "/usr/bin/timeout --signal=TERM --kill-after=60s 900s"
    sim_start = "/usr/bin/timeout --signal=TERM --kill-after=30s 180s"
    checks["fresh_namespace_before_attempt"] = text.index(namespace) < text.index(attempt)
    checks["atomic_unique_attempt"] = text.count(attempt) == 1
    checks["attempt_before_stage_and_tools"] = (
        text.index(attempt) < text.index(stage) < text.index(compile_start) < text.index(sim_start))
    checks["automatic_retry_absent"] = (
        text.count(compile_start) == 1 and text.count(sim_start) == 1 and
        "automatic_retry':False" in text and
        "status=M2033_ATTEMPT_CONSUMED\\nvcs_compile_runs=1\\nsimv_runs=1\\nretry=false" in text and
        "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\\nexit_code=%s\\nretry=false" in text)

    collision = section(text, "reject_same_uid_vcs()", "stage_active=0")
    checks["collision_identity_axes"] = all(token in collision for token in (
        "blocked = {'vcs', 'vcs1', 'vlogan', 'simv'}",
        "path.stat().st_uid != os.getuid()", "comm =", "os.readlink",
        "exe = Path(os.readlink", "argv0 =", "comm.startswith('common_shell_ex')",
        "exe.startswith('common_shell_ex')", "'/vcs/' in joined"))
    checks["same_uid_lock"] = all(token in text for token in (
        'LOCK="/tmp/hw_autoresearch_m2033_vcs_uid_${RUN_UID}.lock"',
        'exec 9>"${LOCK}"', "/usr/bin/flock -n 9"))
    scan_positions = []
    cursor = 0
    while True:
        found = text.find("reject_same_uid_vcs\n", cursor)
        if found < 0:
            break
        scan_positions.append(found)
        cursor = found + 1
    checks["two_collision_scans"] = len(scan_positions) == 2
    checks["first_scan_before_attempt"] = scan_positions[0] < text.index(attempt)
    checks["second_scan_immediately_before_compile"] = (
        text.index(stage) < scan_positions[1] < text.index("set +e", scan_positions[1]) <
        text.index(compile_start))

    compile_block = section(text, compile_start, "compile_rc=$?")
    sim_block = section(text, sim_start, "sim_rc=$?")
    whitelist = (
        "/usr/bin/env -i PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin",
        "LANG=C LC_ALL=C TMPDIR=/tmp PWD=\"${STAGE}\"",
        "VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1",
        "SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo",
        "LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat")
    checks["compile_clean_whitelist"] = all(token in compile_block for token in whitelist)
    checks["sim_clean_whitelist"] = all(token in sim_block for token in whitelist)
    checks["home_not_set_or_inherited"] = (
        not __import__("re").search(r"(?:^|\s)HOME=", compile_block) and
        not __import__("re").search(r"(?:^|\s)HOME=", sim_block))
    checks["bounded_compile_and_sim"] = (
        compile_start in compile_block and sim_start in sim_block and
        '[[ "${compile_rc}" -eq 0 ]] || exit 3' in text and
        '[[ "${sim_rc}" -eq 0 ]] || exit 4' in text)
    checks["foundry_unit_delay_compile"] = all(token in compile_block for token in (
        "+define+UNIT_DELAY", '"${FOUNDRY_V}"', '"${MACRO}"', '"${TOP}"', '"${TB}"',
        "-top tb_m2031_ep34_c1_first64_model_rtl_calibration"))

    expected_pass = (
        "PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 "
        "input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 "
        "parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 "
        "forwards=4 deadline_holds=6 stalls=14 psum_commits=64 "
        "row_completions=64 numeric_commits=64 rtl_cycle_speedup=false "
        "full_network=false system_speedup=false")
    checks["exact_single_terminal"] = (
        expected_pass in text and 'grep -Fxc "${expected_pass}" sim.log' in text)
    checks["negative_log_gate"] = all(token in text for token in (
        "Error|Fatal|Assertion.*failed", "\\$fatal", "global watchdog expired",
        "counter mismatch", "numeric mismatch", "protocol_error"))
    checks["post_attempt_failure_quarantine"] = all(token in text for token in (
        "stage_active=1", "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
        'seal_dir "${STAGE}" || true', 'mv -T -- "${STAGE}" "${FAILED}" || true'))
    checks["result_double_seal"] = (
        'seal_dir "${STAGE}"' in text and
        'mv -T -- "${STAGE}" "${RESULT}"' in text)
    checks["payload_boundary"] = all(token in text for token in (
        "masks':'real ep34 sealed-ledger prefix",
        "signed12_values':'synthetic deterministic function of source index and lane'",
        "psum_prior':'all zero'", "real_weight_or_real_psum_numeric_calibration':False"))
    checks["claim_boundary"] = all(token in text for token in (
        "cpu_model_1p694510x_upgraded_to_rtl':False",
        "rtl_cycle_speedup':False", "same_area':False", "timing':False",
        "power':False", "energy':False", "full_network':False",
        "system_speedup':False", "headline':False"))
    checks["no_dynamic_shell_injection"] = all(token not in text for token in (
        "eval ", "source \"", "source $", "bash -c", "/bin/bash -c"))
    require(all(checks.values()), [key for key, value in checks.items() if not value])
    return checks


def replace_once(text, old, new):
    require(text.count(old) >= 1, "mutation source missing: " + old[:60])
    return text.replace(old, new, 1)


def main():
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), str(path))
        require(sha(path) == expected, "identity drift: " + str(path))
    verify_seal(M2032, 5)
    review = json.loads((M2032 / "review.json").read_text())
    require(review["status"] == "PASS_M2032_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_SOURCE_HAMMER", "M2032 status")
    require(review["severity_counts"]["p0"] == 0, "M2032 P0")

    source = RUNNER.read_text()
    checks = validate(source)
    compile_timeout = "/usr/bin/timeout --signal=TERM --kill-after=60s 900s"
    sim_timeout = "/usr/bin/timeout --signal=TERM --kill-after=30s 180s"
    clean_env = "/usr/bin/env -i PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"
    sim_clean = sim_timeout + " \\\n  " + clean_env
    sim_dirty = sim_timeout + " \\\n  /usr/bin/env PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"
    mutations = [
        ("dirty_shebang", "#!/usr/bin/env -S -i", "#!/usr/bin/env -S"),
        ("drop_startup_scrub", "unset BASH_ENV ENV CDPATH GLOBIGNORE", "true # scrub removed"),
        ("expand_argument_surface", "[[ $# -eq 0 ]]", "[[ $# -le 9 ]]"),
        ("top_sha", "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1", "0" * 64),
        ("tb_sha", "8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1", "1" * 64),
        ("fixture_sha", "4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3", "2" * 64),
        ("foundry_sha", "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d", "3" * 64),
        ("vcs_sha", "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287", "4" * 64),
        ("python_sha", "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161", "5" * 64),
        ("docs359_sha", DOCS359_SHA, "6" * 64),
        ("drop_m2034_seal", 'verify_double_seal "${M2034_DIR}"', "true # seal removed"),
        ("review_status", "PASS_M2034_M2033_RUNNER_SOURCE_HAMMER", "PASS_UNRELATED"),
        ("review_score", "review.get('score', 0) < 90", "review.get('score', 0) < 0"),
        ("review_p0", "get('P0') != 0", "get('P0') != 99"),
        ("review_runner_sha", "review.get('runner_sha256') != runner_sha", "False"),
        ("release_status", "AUTHORIZED_EXACTLY_ONE_M2033_VCS_COMPILE_AND_SIM", "AUTHORIZE_UNBOUNDED"),
        ("release_review_sha", "release.get('review_sha256') != sha(review_path)", "False"),
        ("release_result_path", "release.get('result_path', '')", "release.get('wrong_result_path', '')"),
        ("release_attempt_path", "release.get('attempt_path', '')", "release.get('wrong_attempt_path', '')"),
        ("release_retry", "'automatic_retry': False", "'automatic_retry': True"),
        ("drop_vcs1_collision", "'vcs1'", "'not_vcs1'"),
        ("drop_vlogan_collision", "'vlogan'", "'not_vlogan'"),
        ("drop_exe_axis", "exe = Path(os.readlink", "discarded = Path(os.readlink"),
        ("drop_argv0_axis", "argv0 = Path(parts[0])", "discarded = Path(parts[0])"),
        ("drop_wrapper_prefix", "comm.startswith('common_shell_ex')", "comm == 'common_shell_exec'"),
        ("drop_lock", "/usr/bin/flock -n 9", "true # flock removed"),
        ("drop_one_scan", "reject_same_uid_vcs\n\nmkdir", "true # scan removed\n\nmkdir"),
        ("compile_unbounded", compile_timeout, "/usr/bin/env"),
        ("sim_unbounded", sim_timeout, "/usr/bin/env"),
        ("compile_dirty_env", clean_env, "/usr/bin/env PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"),
        ("sim_dirty_env", sim_clean, sim_dirty),
        ("remove_unit_delay", "+define+UNIT_DELAY", "+define+NOT_UNIT_DELAY"),
        ("pass_not_exact", "grep -Fxc", "grep -Fq"),
        ("promote_rtl_speedup", "rtl_cycle_speedup':False", "rtl_cycle_speedup':True"),
        ("promote_real_numeric", "real_weight_or_real_psum_numeric_calibration':False", "real_weight_or_real_psum_numeric_calibration':True"),
        ("promote_system", "system_speedup':False", "system_speedup':True"),
        ("allow_retry_marker", "retry=false", "retry=true"),
        ("erase_failure_label", "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "FAILED_BUT_CITABLE"),
    ]

    rejected = []
    for name, old, new in mutations:
        mutated = replace_once(source, old, new)
        try:
            validate(mutated)
        except (AssertionError, ValueError, IndexError, KeyError):
            rejected.append(name)
    require(len(rejected) == len(mutations), "escaped mutations: " + repr(
        [name for name, _, _ in mutations if name not in rejected]))

    result = {
        "status": "PASS_M2034_INDEPENDENT_RUNNER_HAMMER",
        "runner_sha256": sha(RUNNER),
        "static_checks": len(checks),
        "mutations_rejected": len(rejected),
        "mutations_total": len(mutations),
        "mutation_names": rejected,
        "eda_launched": False,
        "license_query_launched": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

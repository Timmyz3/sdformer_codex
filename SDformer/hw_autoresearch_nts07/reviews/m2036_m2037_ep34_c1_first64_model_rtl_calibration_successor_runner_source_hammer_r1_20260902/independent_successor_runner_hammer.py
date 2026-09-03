#!/usr/bin/env python3
"""Independent, read-only source hammer for the exact M2037 VCS runner.

This checker never executes the reviewed runner and never invokes VCS, simv,
an EDA license utility, or a GPU program.  It validates immutable inputs,
predecessor evidence, fresh namespaces, fail-closed launch mechanics, the one
permitted VCS-generated symlink removal, and the M2036 release binding.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / (
    "dc_handoff/scripts/"
    "run_vcs_m2033_m2031_ep34_c1_first64_model_rtl_calibration_one_shot.sh"
)
REVIEW_DIR = Path(__file__).resolve().parent
REVIEW = REVIEW_DIR / "review.json"
RELEASE = REVIEW_DIR / "launch_release.json"
RESULT = HW / (
    "results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_"
    "vcs_successor_r1_20260902"
)
ATTEMPT = HW / (
    "results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_"
    "vcs_successor_attempt_consumed"
)

RUNNER_SHA256 = "9ecfea0331368385421c2b7bfbf84d00fe9bf6f4d793f8fc07bfa2b25fc047b3"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

M2032_DIR = HW / (
    "reviews/m2032_m2031_ep34_c1_first64_model_rtl_calibration_"
    "source_hammer_r1_20260902"
)
M2034_DIR = HW / (
    "reviews/m2034_m2033_ep34_c1_first64_model_rtl_calibration_"
    "runner_source_hammer_r1_20260902"
)
M2035_DIR = HW / (
    "reviews/m2035_m2033_ep34_c1_first64_vcs_seal_failure_"
    "hammer_r1_20260902"
)

PINNED_FILES = {
    RUNNER: RUNNER_SHA256,
    HW / "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv":
        "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv":
        "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    HW / "tb_m528_dw1rw/tb_m2031_ep34_c1_first64_model_rtl_calibration.sv":
        "8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1",
    HW / "tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh":
        "4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3",
    HW / "system_simulator/scripts/check_m2031_ep34_c1_first64_model_rtl_calibration_source.py":
        "c3937a5d069f56cee3bd641eda0b78777acda8c15aae54e8650360e1105c485a",
    M2032_DIR / "review.json":
        "f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
    M2034_DIR / "review.json":
        "3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544",
    M2035_DIR / "review.json":
        "e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654",
    Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
         "macro_assets/tsmc28_128x128_1rw_20260821/"
         "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v"):
        "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"):
        "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/anaconda3/bin/python3.12"):
        "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    HW / "docs/359_DATE终局冻结_20260813.md": DOCS359_SHA256,
}

SEALED_DIRS = {
    M2032_DIR: (
        "987ff979ecaa505bcce1027fd4c3b255e3bad92fea23c812b50891d07dee8927",
        "103e6971327a58c8d800fac51d2c566f78c1ebd5a9e2a62494d7b01071032c25",
    ),
    M2034_DIR: (
        "10d3145b15412ebec6d5552f7dd3e262a5e327ce2b43c0340c5450ad59ea01ba",
        "47ab2029d9247b8634016c5fef73397c93bc81828a5eb4a63119d81c38231680",
    ),
    M2035_DIR: (
        "3d77a4b6f7c70c3526557ccfe98d22ac65240b340b9aafd2f1cfe0a1458276dd",
        "4fdee7f53ad76535bd7dc89b0534fb4deb0165ad38c9c43a1ee0ed6ea2cab7d3",
    ),
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def reject(token: str) -> None:
        raise ValueError("non-standard JSON token: " + token)

    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be object")
    return value


def verify_sealed_directory(directory: Path, manifest_sha: str | None = None,
                            outer_sha: str | None = None) -> int:
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory is not a real directory: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest identity")
    require(outer.is_file() and not outer.is_symlink(), "outer identity")
    if manifest_sha is not None:
        require(sha256(manifest) == manifest_sha, "manifest SHA drift")
    if outer_sha is not None:
        require(sha256(outer) == outer_sha, "outer SHA drift")
    outer_fields = outer.read_text(encoding="utf-8").split()
    require(outer_fields == [sha256(manifest), "SHA256SUMS"], "outer seal content")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "malformed manifest row")
        digest, relative = fields
        relative = relative.lstrip("*")
        rel = Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts and relative not in listed,
                "unsafe or duplicate manifest path")
        target = directory / rel
        require(target.is_file() and not target.is_symlink(), "manifest member identity")
        require(sha256(target) == digest, "manifest member digest")
        listed.add(relative)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    require(not any(path.is_symlink() for path in directory.rglob("*")),
            "symlink in sealed review tree")
    require(actual == listed, "unsealed or missing review member")
    return len(listed)


def section(text: str, left: str, right: str) -> str:
    begin = text.index(left)
    end = text.index(right, begin)
    return text[begin:end]


def validate_source(text: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["clean_shebang"] = text.startswith(
        "#!/usr/bin/env -S -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/bash\n")
    checks["strict_shell_environment"] = all(token in text for token in (
        "set -euo pipefail", "unset BASH_ENV ENV CDPATH GLOBIGNORE",
        "export PATH=/usr/bin:/bin LANG=C LC_ALL=C", "umask 077"))
    checks["zero_argument_surface"] = "[[ $# -eq 0 ]]" in text
    checks["fresh_m2037_namespaces"] = all(token in text for token in (
        'RESULT="${HW_ROOT}/results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902"',
        'ATTEMPT="${HW_ROOT}/results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_attempt_consumed"',
        'STAGE="${HW_ROOT}/results/.m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_stage.$$"',
        'FAILED="${HW_ROOT}/results/m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902.failed_or_incomplete.$$.quarantine"'))
    checks["exact_predecessor_pins"] = all(token in text for token in (
        "require_sha f0b6ce291ec25b52815db25c0bc8e76d87162c9b3821fa9d3b7eb3577bfa238a",
        "require_sha 3eb091f8385e73745deea40e82cb4a04711b22f3b91e619692c5d0156b027544",
        "require_sha e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654"))
    checks["all_review_double_seals"] = all(
        'verify_double_seal "${%s_DIR}"' % milestone in text
        for milestone in ("M2032", "M2034", "M2035", "M2036"))

    gate = section(text, '"${PYTHON}" -I - "${RUNNER}"', "audit_output=")
    checks["m2036_self_gate"] = all(token in gate for token in (
        "PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER",
        "review.get('score', 0) < 90",
        "review.get('severity_counts', {}).get('P0') != 0",
        "review.get('runner_sha256') != runner_sha"))
    checks["m2036_release_gate"] = all(token in gate for token in (
        "AUTHORIZED_EXACTLY_ONE_M2037_SUCCESSOR_VCS_COMPILE_AND_SIM",
        "release.get('runner_sha256') != runner_sha",
        "release.get('review_sha256') != sha(review_path)",
        "release.get('result_path', '')", "release.get('attempt_path', '')",
        "{'vcs_compile_runs': 1, 'simv_runs': 1, 'automatic_retry': False}"))

    attempt = 'mkdir -- "${ATTEMPT}"'
    stage = 'mkdir -- "${STAGE}"'
    compile_timeout = "/usr/bin/timeout --signal=TERM --kill-after=60s 900s"
    sim_timeout = "/usr/bin/timeout --signal=TERM --kill-after=30s 180s"
    checks["one_shot_order"] = (
        text.index('[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" && ! -e "${STAGE}" ]]')
        < text.index(attempt) < text.index(stage) < text.index(compile_timeout)
        < text.index(sim_timeout))
    checks["one_compile_one_sim_no_retry"] = (
        text.count(compile_timeout) == 1 and text.count(sim_timeout) == 1 and
        text.count(attempt) == 1 and "retry=false" in text and
        "'automatic_retry':False" in text)
    checks["same_uid_lock"] = all(token in text for token in (
        'LOCK="/tmp/hw_autoresearch_m2037_vcs_uid_${RUN_UID}.lock"',
        'exec 9>"${LOCK}"', "/usr/bin/flock -n 9"))
    scan_positions = []
    cursor = 0
    while True:
        found = text.find("reject_same_uid_vcs\n", cursor)
        if found < 0:
            break
        scan_positions.append(found)
        cursor = found + 1
    checks["two_collision_scans"] = (
        len(scan_positions) == 2 and scan_positions[0] < text.index(attempt) and
        text.index(stage) < scan_positions[1] < text.index(compile_timeout))
    collision = section(text, "reject_same_uid_vcs()", "stage_active=0")
    checks["collision_identity_axes"] = all(token in collision for token in (
        "blocked = {'vcs', 'vcs1', 'vlogan', 'simv'}",
        "path.stat().st_uid != os.getuid()", "comm =", "exe = Path(os.readlink",
        "argv0 =", "comm.startswith('common_shell_ex')", "exe.startswith('common_shell_ex')"))

    clean = "/usr/bin/env -i PATH=/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"
    compile_block = section(text, compile_timeout, "compile_rc=$?")
    sim_block = section(text, sim_timeout, "sim_rc=$?")
    whitelist = (clean, "LANG=C LC_ALL=C TMPDIR=/tmp PWD=\"${STAGE}\"",
                 "VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1",
                 "SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo",
                 "LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat")
    checks["clean_compile_env"] = all(token in compile_block for token in whitelist)
    checks["clean_sim_env"] = all(token in sim_block for token in whitelist)
    checks["bounded_return_codes"] = all(token in text for token in (
        '[[ "${compile_rc}" -eq 0 ]] || exit 3',
        '[[ "${sim_rc}" -eq 0 ]] || exit 4'))
    checks["frozen_unit_delay_workload"] = all(token in compile_block for token in (
        "+define+UNIT_DELAY", '"${FOUNDRY_V}"', '"${MACRO}"', '"${TOP}"',
        '"${TB}"', "-top tb_m2031_ep34_c1_first64_model_rtl_calibration"))
    checks["exact_terminal_and_negative_gate"] = all(token in text for token in (
        'grep -Fxc "${expected_pass}" sim.log', "Error|Fatal|Assertion.*failed",
        "\\$fatal", "global watchdog expired", "counter mismatch",
        "numeric mismatch", "protocol_error"))

    symlink = section(text, '"${PYTHON}" -I - "${STAGE}" <<\'PY\'',
                      '"${PYTHON}" -I - "${STAGE}" "${RUNNER}"')
    checks["exactly_one_archive_link"] = all(token in symlink for token in (
        "if len(links) != 1", "link.parent != stage / 'csrc'",
        "re.fullmatch(r'_\\d+_archive_1\\.so', link.name)"))
    checks["raw_and_in_tree_regular_target"] = all(token in symlink for token in (
        "raw_target = os.readlink(str(link))", "target = link.resolve(strict=True)",
        "stage / 'simv.daidir' / link.name", "target != expected_target",
        "not target.is_file()", "target.is_symlink()"))
    checks["record_before_exact_unlink"] = all(token in symlink for token in (
        "'link_path':relative", "'raw_target':raw_target",
        "'resolved_target_path':target.relative_to(stage).as_posix()",
        "'target_size_bytes':target.stat().st_size", "'target_sha256':digest",
        "link.unlink()")) and symlink.count(".unlink()") == 1
    checks["zero_remaining_links"] = all(token in symlink for token in (
        "remaining = [p.relative_to(stage).as_posix()",
        "if remaining:", "symlinks remain after exact removal",
        "'remaining_symlinks_after_unlink':0"))
    checks["regular_record_written_after_unlink"] = (
        symlink.index("link.unlink()") < symlink.index("remaining =") <
        symlink.index("generated_symlink_removal.json"))
    checks["receipt_binds_removal_record"] = (
        "'generated_symlink_removal_sha256':sha(stage/'generated_symlink_removal.json')"
        in text)

    publish = section(text, 'seal_dir "${STAGE}"\nverify_double_seal "${STAGE}"',
                      "trap - EXIT INT TERM HUP")
    publish_tokens = (
        'seal_dir "${STAGE}"', 'verify_double_seal "${STAGE}"',
        'mv -T -n -- "${STAGE}" "${RESULT}"',
        '[[ ! -e "${STAGE}" && -d "${RESULT}" && ! -L "${RESULT}" ]]',
        'verify_double_seal "${RESULT}"', "stage_active=0")
    checks["verified_no_replace_publication"] = all(token in publish for token in publish_tokens)
    checks["publication_order"] = all(
        publish.index(publish_tokens[index]) < publish.index(publish_tokens[index + 1])
        for index in range(len(publish_tokens) - 1))
    checks["failure_quarantine"] = all(token in text for token in (
        "FAILED_OR_INCOMPLETE_DO_NOT_CITE", 'seal_dir "${STAGE}" || true',
        'mv -T -- "${STAGE}" "${FAILED}" || true'))
    checks["claim_boundary"] = all(token in text for token in (
        "'signed12_values':'synthetic deterministic function of source index and lane'",
        "'psum_prior':'all zero'", "'real_weight_or_real_psum_numeric_calibration':False",
        "'cpu_model_1p694510x_upgraded_to_rtl':False",
        "'rtl_cycle_speedup':False", "'same_area':False", "'timing':False",
        "'power':False", "'energy':False", "'full_network':False",
        "'system_speedup':False", "'headline':False"))
    checks["no_dynamic_shell_injection"] = all(token not in text for token in (
        "eval ", "source \"", "source $", "bash -c", "/bin/bash -c"))
    require(all(checks.values()), "failed static checks: " + repr(
        [key for key, value in checks.items() if not value]))
    return checks


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) >= 1, "missing mutation source: " + old[:70])
    return text.replace(old, new, 1)


def validate_release() -> None:
    review = strict_json(REVIEW)
    release = strict_json(RELEASE)
    require(review.get("status") ==
            "PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER", "review status")
    require(review.get("score", 0) >= 90, "review score")
    require(review.get("severity_counts", {}).get("P0") == 0, "review P0")
    require(review.get("runner_sha256") == RUNNER_SHA256, "review runner SHA")
    require(release.get("status") ==
            "AUTHORIZED_EXACTLY_ONE_M2037_SUCCESSOR_VCS_COMPILE_AND_SIM",
            "release status")
    require(release.get("runner_sha256") == RUNNER_SHA256, "release runner SHA")
    require(release.get("review_sha256") == sha256(REVIEW), "release review SHA")
    require(Path(release.get("result_path", "")).resolve() == RESULT.resolve(),
            "release result path")
    require(Path(release.get("attempt_path", "")).resolve() == ATTEMPT.resolve(),
            "release attempt path")
    require(release.get("execution_budget") == {
        "vcs_compile_runs": 1, "simv_runs": 1, "automatic_retry": False},
        "release execution budget")


def main() -> None:
    for path, expected in PINNED_FILES.items():
        require(path.is_file() and not path.is_symlink(), "pinned input identity: " + str(path))
        require(sha256(path) == expected, "pinned input SHA drift: " + str(path))
    for directory, seals in SEALED_DIRS.items():
        verify_sealed_directory(directory, *seals)

    predecessor_statuses = {
        M2032_DIR / "review.json":
            "PASS_M2032_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION_SOURCE_HAMMER",
        M2034_DIR / "review.json": "PASS_M2034_M2033_RUNNER_SOURCE_HAMMER",
        M2035_DIR / "review.json": "PASS_M2035_M2033_CANONICAL_SEAL_FAILURE_HAMMER",
    }
    for path, expected in predecessor_statuses.items():
        require(strict_json(path).get("status") == expected, "predecessor status drift")

    old_result = HW / "results/m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902"
    old_attempt = HW / "results/.m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_attempt_consumed"
    old_quarantine = list((HW / "results").glob(
        "m2033_m2031_ep34_c1_first64_model_rtl_calibration_vcs_r1_20260902."
        "failed_or_incomplete.*.quarantine"))
    require(not os.path.lexists(old_result), "old M2033 canonical result must remain absent")
    require(old_attempt.is_dir() and not old_attempt.is_symlink(), "old attempt marker")
    verify_sealed_directory(old_attempt)
    require(len(old_quarantine) == 1 and (old_quarantine[0] / "FAILED_DO_NOT_CITE").is_file(),
            "old M2033 quarantine population")
    old_links = [path for path in old_quarantine[0].rglob("*") if path.is_symlink()]
    require(len(old_links) == 1, "old M2033 symlink cardinality")
    old_link = old_links[0]
    require(old_link.relative_to(old_quarantine[0]).as_posix() ==
            "csrc/_2362104_archive_1.so", "old M2033 symlink path")
    require(os.readlink(str(old_link)) == ".//../simv.daidir//_2362104_archive_1.so",
            "old M2033 raw symlink target")
    old_target = old_link.resolve(strict=True)
    require(old_target ==
            (old_quarantine[0] / "simv.daidir/_2362104_archive_1.so").resolve(strict=True),
            "old M2033 resolved symlink target")
    require(old_target.is_file() and not old_target.is_symlink() and
            old_target.stat().st_size == 573944 and
            sha256(old_target) ==
            "6e63b0e29cf867d67d6eb68fbfd434cbed4b26a6bbf6176d3a20ec22995924c8",
            "old M2033 symlink target identity")

    require(not os.path.lexists(RESULT) and not os.path.lexists(ATTEMPT),
            "fresh M2037 canonical namespace already consumed")
    require(not list((HW / "results").glob(
        ".m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_stage.*")),
        "stale M2037 private stage")
    require(not list((HW / "results").glob(
        "m2037_m2031_ep34_c1_first64_model_rtl_calibration_vcs_successor_r1_20260902."
        "failed_or_incomplete.*.quarantine")), "stale M2037 quarantine")

    syntax = subprocess.run(["/bin/bash", "-n", str(RUNNER)], check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
    require(syntax.returncode == 0, "bash -n failed: " + syntax.stderr)
    source = RUNNER.read_text(encoding="utf-8")
    checks = validate_source(source)

    mutations = [
        ("result_namespace", "results/m2037_m2031", "results/m2033_m2031"),
        ("attempt_namespace", "results/.m2037_m2031", "results/.m2033_m2031"),
        ("drop_m2035_pin", "e3b8bffe5b9c0d33d326b5431ba79c9bcacec67527c4f996786cb5dd5f634654", "0" * 64),
        ("drop_m2036_seal", 'verify_double_seal "${M2036_DIR}"', "true # seal removed"),
        ("review_status", "PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER", "PASS_UNRELATED"),
        ("release_status", "AUTHORIZED_EXACTLY_ONE_M2037_SUCCESSOR_VCS_COMPILE_AND_SIM", "AUTHORIZE_UNBOUNDED"),
        ("allow_retry", "'automatic_retry': False", "'automatic_retry': True"),
        ("drop_lock", "/usr/bin/flock -n 9", "true # no lock"),
        ("drop_second_scan", "reject_same_uid_vcs\nset +e", "true # no second scan\nset +e"),
        ("compile_timeout", "/usr/bin/timeout --signal=TERM --kill-after=60s 900s", "/usr/bin/env"),
        ("sim_timeout", "/usr/bin/timeout --signal=TERM --kill-after=30s 180s", "/usr/bin/env"),
        ("dirty_compile_env", "/usr/bin/env -i PATH=/opt/synopsys/vcs", "/usr/bin/env PATH=/opt/synopsys/vcs"),
        ("dirty_sim_env", "/usr/bin/timeout --signal=TERM --kill-after=30s 180s \\\n  /usr/bin/env -i", "/usr/bin/timeout --signal=TERM --kill-after=30s 180s \\\n  /usr/bin/env"),
        ("pass_cardinality", "grep -Fxc", "grep -Fq"),
        ("link_count", "if len(links) != 1", "if len(links) < 1"),
        ("link_shape", "link.parent != stage / 'csrc'", "False"),
        ("target_location", "target != expected_target", "False"),
        ("target_regular", "not target.is_file()", "False"),
        ("raw_target_record", "'raw_target':raw_target", "'raw_target':'omitted'"),
        ("target_size_record", "'target_size_bytes':target.stat().st_size", "'target_size_bytes':0"),
        ("target_sha_record", "'target_sha256':digest", "'target_sha256':'omitted'"),
        ("broad_unlink", "link.unlink()", "for p in links: p.unlink()"),
        ("remaining_link_gate", "if remaining:", "if False:"),
        ("receipt_record_binding", "'generated_symlink_removal_sha256':sha(stage/'generated_symlink_removal.json')", "'generated_symlink_removal_sha256':'omitted'"),
        ("drop_stage_seal", 'seal_dir "${STAGE}"\nverify_double_seal "${STAGE}"', 'true # no stage seal\nverify_double_seal "${STAGE}"'),
        ("drop_stage_seal_verify", 'verify_double_seal "${STAGE}"', "true # no verify"),
        ("drop_no_clobber", "mv -T -n --", "mv -T --"),
        ("drop_publish_assert", '[[ ! -e "${STAGE}" && -d "${RESULT}" && ! -L "${RESULT}" ]]', "true # no assert"),
        ("drop_result_verify", 'verify_double_seal "${RESULT}"', "true # no result verify"),
        ("promote_rtl_speedup", "'rtl_cycle_speedup':False", "'rtl_cycle_speedup':True"),
        ("promote_real_numeric", "'real_weight_or_real_psum_numeric_calibration':False", "'real_weight_or_real_psum_numeric_calibration':True"),
        ("promote_system_speedup", "'system_speedup':False", "'system_speedup':True"),
    ]
    rejected = []
    for name, old, new in mutations:
        try:
            validate_source(replace_once(source, old, new))
        except (AssertionError, ValueError, IndexError):
            rejected.append(name)
    require(len(rejected) == len(mutations), "escaped mutations: " + repr(
        [name for name, _, _ in mutations if name not in rejected]))

    validate_release()
    verify_sealed_directory(REVIEW_DIR)
    print(json.dumps({
        "status": "PASS_M2036_M2037_SUCCESSOR_RUNNER_SOURCE_HAMMER",
        "runner_sha256": sha256(RUNNER),
        "static_checks": len(checks),
        "mutations_rejected": len(rejected),
        "mutations_total": len(mutations),
        "bash_syntax_pass": True,
        "fresh_result_attempt_stage_quarantine": True,
        "predecessor_double_seals_verified": 3,
        "eda_launched": False,
        "license_query_launched": False,
        "gpu_launched": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

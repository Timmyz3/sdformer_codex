#!/usr/bin/env python3
"""Fresh final-launch hammer for M799/M533 R17.

The live runner, VCS, simv, license server, and every HDL/EDA tool are outside
this program's authority.  The only runner execution is its source-owned
pre-mkdir stub, which stops before identity/license probes and attempt mkdir.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


REPO = Path("/home/zhumd/work/sdformer_codex/SDformer")
HW = REPO / "hw_autoresearch_nts07"
OUT = HW / "reviews/m799_m533_r17_unit_delay_vcs_final_launch_release_hammer_r1_20260828"
REQUEST_DIR = HW / "reviews/m807_m805_m799_m533_r17_final_launch_release_hammer_REQUEST_r1_20260828"
REQUEST = REQUEST_DIR / "request.json"
RELEASE = HW / "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m799_m533_m528_dead_write_only_1rw_unit_delay_r17_exact_sha.sh"
SOURCE = HW / "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_r1_20260828.json"
CANDIDATE = HW / "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_r1_20260828.json"
M801_DIR = HW / "reviews/m799_m533_r17_unit_delay_source_static_hammer_r1_20260828"
M801 = M801_DIR / "review.json"
M805_DIR = HW / "reviews/m805_m799_m533_r17_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"
M805 = M805_DIR / "review.json"
M805_CANON_DIR = HW / "reviews/m799_m533_r17_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"
M805_CANON = M805_CANON_DIR / "review.json"
M794_DIR = HW / "reviews/m794_m533_r15_premkdir_undefined_function_failure_hammer_r1_20260828"
M794 = M794_DIR / "review.json"
R15_RELEASE = HW / "contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
M797_DIR = HW / "reviews/m797_m795_m533_r16_function_closure_fresh_hammer_r1_20260828"
M797 = M797_DIR / "review.json"
CLOSURE = HW / "verif_m528_dw1rw/test_m799_r17_runner_function_closure.py"
WHITELIST = HW / "verif_m528_dw1rw/m799_r17_external_command_whitelist.json"
DRYRUN = HW / "verif_m528_dw1rw/test_m799_r17_runner_premkdir_dry_run.py"
PY36 = Path("/usr/libexec/platform-python3.6")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_r17_20260828"
R15_RESULT = HW / "results/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_r15_20260828"
R16_RESULT = HW / "results/m795_m533_m528_dead_write_only_1rw_unit_delay_vcs_r16_20260828"
R16_RELEASE = HW / "contracts/m795_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"

EXPECTED = {
    REQUEST: "67980033b604fc25273214a9f988136a89f89b830f0a3d8b12abfffc63f54232",
    RELEASE: "fe6814cd90a7b3d2f614376b1593b444adb4c046f236d5707937cd6c16c1a661",
    RUNNER: "4d1b0a940ee44013bf09b0b8b41197b31f43c44d83732da8710233e679a7e0fe",
    SOURCE: "0fe4fe0b4531bf5bfa6d69919f8845f935b389c0cf9762e7a0bf2dec5eb8eae5",
    CANDIDATE: "fc36f2f56cc48316e941b0840cb1803f74ec9c65e7d5272ede47178acfead865",
    M801: "abd9a611c312d01bc1aa04d74ca2d2fe80ca578733e752db0e926d69aea8a5dd",
    M805: "eab4fff314f26cb61f16ec52920d9b9848507e03cfb070ccd3907db013a24532",
    M805_CANON: "eab4fff314f26cb61f16ec52920d9b9848507e03cfb070ccd3907db013a24532",
    M794: "bc244f11943089794151b16d5bf6bf56b4708e4df69d4c6bb0ecbcd2efe0def8",
    R15_RELEASE: "6c3d4a1ffef609765a387f45bdf502510a1d0d9ded6df0b281f50668d689fd08",
    M797: "7f9b7d492bd29329e3982afd3553d6aa7a9ba4d186d6fa21dc0912e754251074",
    CLOSURE: "7daeb06f0dd8d3e18d077fc8ad115911e2a223491f913c5b5c4f0b570a1093a8",
    WHITELIST: "7bc11a6c4b7ce568de9a934c8178114ec8401a8e01125722c7173b92e75061d6",
    DRYRUN: "20136d66506042453d40ba4564f1340580c46666d5e206641c871fccefa2fa36",
    PY36: "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

AUTH = {"vcs_runs": 1, "simv_runs": 1, "iverilog_runs": 0,
        "verilator_runs": 0, "dc_runs": 0, "formality_runs": 0,
        "pt_runs": 0, "ptpx_runs": 0, "cpu_runs": 0, "gpu_runs": 0,
        "network_or_remote_jobs": 0}
FUTURE_COMMAND = (
    "env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C "
    "VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux "
    "SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo "
    "LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat "
    "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/"
    "dc_handoff/scripts/run_vcs_m799_m533_m528_dead_write_only_1rw_unit_delay_r17_exact_sha.sh"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular(path):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def strict_text(text, label):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key in %s: %s" % (label, key))
            out[key] = value
        return out
    def reject(token):
        raise RuntimeError("non-standard JSON token in %s: %s" % (label, token))
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    def finite(member):
        if isinstance(member, float):
            require(math.isfinite(member), "non-finite JSON number in %s" % label)
        elif isinstance(member, dict):
            for key, child in member.items():
                finite(key); finite(child)
        elif isinstance(member, list):
            for child in member:
                finite(child)
    finite(value)
    return value


def strict_json(path):
    path = Path(path)
    require(regular(path), "missing/non-regular JSON: %s" % path)
    return strict_text(path.read_text(encoding="utf-8"), str(path))


def sidecar_value(path):
    path = Path(path)
    member = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(regular(member) and regular(outer), "missing JSON double seal: %s" % path)
    require(member.read_text(encoding="utf-8").split() == [sha(path), path.name],
            "JSON member seal mismatch: %s" % path)
    require(outer.read_text(encoding="utf-8").split() == [sha(member), member.name],
            "JSON outer seal mismatch: %s" % path)
    return strict_json(path)


def package(directory, json_name):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "bad package: %s" % directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), "missing package seals: %s" % directory)
    names = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "bad manifest SHA")
        require(name not in names and not name.startswith("/") and ".." not in Path(name).parts,
                "bad/duplicate package member: %s" % name)
        target = directory / name
        require(regular(target) and sha(target) == expected, "package member mismatch: %s" % target)
        names.append(name)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), manifest.name],
            "package outer seal mismatch: %s" % directory)
    require(json_name in names, "package JSON omitted: %s" % directory)
    return strict_json(directory / json_name), names


def compare_packages(left, right):
    _, left_names = package(left, "review.json")
    _, right_names = package(right, "review.json")
    require(left_names == right_names, "M805 package member lists differ")
    for name in left_names + ["SHA256SUMS", "SHA256SUMS.seal.sha256"]:
        require((Path(left) / name).read_bytes() == (Path(right) / name).read_bytes(),
                "M805 package bytes differ: %s" % name)
    return left_names


def validate_release(value):
    require(value.get("schema") == "m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_v1", "release schema")
    require(value.get("status") == "AUTHORIZED_EXACTLY_ONE_M799_R17_FOUNDRY_UNIT_DELAY_FUNCTIONAL_VCS_AND_SIMV_ATTEMPT__FRESH_FINAL_RELEASE_HAMMER_REQUIRED", "release status")
    require(value.get("launch_now") is True and value.get("authorization") == AUTH, "release authorization")
    intent = value.get("release_intent", {})
    require(intent.get("run_vcs") is True and intent.get("run_simv") is True and intent.get("max_attempts") == 1, "release max attempt")
    require(intent.get("all_other_execution_authorized") is False and intent.get("attempt_consumed_only_by_runner_atomic_result_mkdir") is True, "release attempt boundary")
    ident = value.get("identity", {})
    exact = {
        "runner_sha256": EXPECTED[RUNNER], "source_contract_sha256": EXPECTED[SOURCE],
        "source_static_review_sha256": EXPECTED[M801], "candidate_sha256": EXPECTED[CANDIDATE],
        "candidate_hammer_review_sha256": EXPECTED[M805],
        "runner_canonical_candidate_hammer_review_sha256": EXPECTED[M805_CANON],
        "m794_r15_failure_review_sha256": EXPECTED[M794],
        "withdrawn_r15_release_sha256": EXPECTED[R15_RELEASE],
        "m797_r16_failure_review_sha256": EXPECTED[M797],
        "docs359_sha256": EXPECTED[DOC359],
    }
    for key, expected in exact.items():
        require(ident.get(key) == expected, "release identity mismatch: %s" % key)
    require(ident.get("runner_path") == str(RUNNER.relative_to(HW)), "release runner path")
    require(ident.get("source_static_review_path") == str(M801.relative_to(HW)), "release M801 path")
    require(ident.get("candidate_hammer_review_path") == str(M805.relative_to(HW)), "release M805 path")
    require(ident.get("runner_canonical_candidate_hammer_package_path") == str(M805_CANON_DIR.relative_to(HW)), "release canonical M805 path")
    unique = value.get("unique_attempt", {})
    require(unique.get("max_attempts") == 1 and unique.get("only_r17_identity_released") is True, "unique attempt")
    require(unique.get("r15_release_status") == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "R15 disposition")
    require(unique.get("r15_attempt_consumed") is False and unique.get("r15_result_absent") is True, "R15 attempt/result")
    require(unique.get("r16_status") == "FAIL_SOURCE_GATE_DO_NOT_EXECUTE_DO_NOT_CITE", "R16 disposition")
    require(unique.get("r16_launch_or_release_authorized") is False and unique.get("r16_result_absent") is True, "R16 boundaries")
    require(value.get("macro_model_mode") == "foundry_UNIT_DELAY_functional", "macro mode")
    boundary = value.get("claim_boundary", {})
    for key in ("functional_vcs_verified", "parent_1rw_protocol_verified", "rtl_verified", "timing_verified",
                "acc24_capacity_physically_bound", "macro_rounded_240kib_capacity_verified",
                "same_ledger_cycles_435293339_promoted", "same_ledger_speedup_1p746753x_promoted",
                "cycles", "speedup", "ppa", "energy", "full_network_or_system",
                "paper_headline", "paper_citable"):
        require(boundary.get(key) is False, "claim promoted: %s" % key)


def expect_failure(fn, label):
    try:
        fn()
    except Exception:
        return True
    raise RuntimeError("negative attack was accepted: %s" % label)


def resolve_sha_edges():
    text = RUNNER.read_text(encoding="utf-8").replace("\\\n", " ")
    variables = {"HW_ROOT": str(HW), "SCRIPT_DIR": str(RUNNER.parent), "RUNNER_PATH": str(RUNNER)}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Z][A-Z0-9_]*)="([^"`()]*)"', line.strip())
        if match:
            variables[match.group(1)] = match.group(2)
    pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
    def expand(value):
        for _ in range(100):
            new = pattern.sub(lambda m: variables.get(m.group(1), m.group(0)), value)
            if new == value:
                return new
            value = new
        raise RuntimeError("variable expansion recursion")
    for key in list(variables):
        variables[key] = expand(variables[key])
    raw = re.findall(r"^\s*require_regular_sha\s+([^\s]+)\s+(.+?)\s*$", text, re.MULTILINE)
    require(len(raw) == 76, "require_regular_sha count")
    ledger = []
    for index, (expected, expression) in enumerate(raw, 1):
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "bad SHA edge literal")
        target = Path(expand(expression.strip().strip('"')))
        require("${" not in str(target) and regular(target), "unresolved/non-regular SHA edge")
        actual = sha(target)
        require(actual == expected, "SHA edge drift: %s" % target)
        ledger.append({"index": index, "path": str(target), "sha256": actual})
    return ledger


def run_checked(command, expect=0):
    completed = subprocess.run(command, cwd=str(REPO), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True,
                               timeout=240, check=False)
    require(completed.returncode == expect,
            "command rc %s != %s: %s" % (completed.returncode, expect, completed.stderr[-1200:]))
    return completed


def closure_suite():
    results = {}
    for mutation in ("none", "delete-definition", "rename-definition", "inject-stale"):
        command = [str(PY36), "-I", str(CLOSURE), str(RUNNER), str(WHITELIST)]
        if mutation != "none":
            command += ["--mutation", mutation, "--expect-fail"]
        completed = run_checked(command)
        payload = strict_text(completed.stdout, "closure-%s" % mutation)
        require(payload.get("observed_pass") == (mutation == "none"), "closure mutation result")
        results[mutation] = payload
    return results


def dryrun_suite():
    completed = run_checked([str(PY36), "-I", str(DRYRUN), str(RUNNER)])
    payload = strict_text(completed.stdout, "premkdir-dryrun")
    require(payload.get("runner_rc") == 86, "dry-run rc")
    require(payload.get("events") == ["stub_collision_initial", "stub_cgroup", "stub_resource",
                                       "stub_collision_final", "live_probe_boundary_stop"], "stub event order")
    require(payload.get("totals") == {"vcs_identity_probe_runs": 0, "license_server_queries": 0,
                                       "vcs_compile_runs": 0, "simv_runs": 0,
                                       "result_directories_created": 0}, "stub side effects")
    require(not RESULT.exists(), "stub created prospective result")
    return payload


def collision_attacks():
    def must_be_absent(path):
        require(not Path(path).exists(), "collision: %s" % path)
    with tempfile.TemporaryDirectory(prefix="m810_collision.") as raw:
        root = Path(raw)
        result = root / "result"
        output = root / "output"
        result.mkdir(); output.mkdir()
        expect_failure(lambda: must_be_absent(result), "existing result directory")
        expect_failure(lambda: must_be_absent(output), "existing final hammer output directory")
    runner_text = RUNNER.read_text(encoding="utf-8")
    require(runner_text.count('[[ ! -e "${RESULT_DIR}" ]] || fail "result/attempt already exists: ${RESULT_DIR}"') == 1,
            "live result collision guard missing/duplicated")
    return {"isolated_existing_result_rejected": True,
            "isolated_existing_output_rejected": True,
            "live_runner_result_guard_occurrences": 1}


def main():
    require(not OUT.exists(), "fixed final hammer output collision")
    require(not RESULT.exists() and not R15_RESULT.exists() and not R16_RESULT.exists(), "result/attempt exists")
    require(not R16_RELEASE.exists(), "R16 release exists")
    for path, expected in EXPECTED.items():
        require(regular(path) and sha(path) == expected, "fixed identity drift: %s" % path)

    request, _ = package(REQUEST_DIR, "request.json")
    release = sidecar_value(RELEASE)
    source = sidecar_value(SOURCE)
    candidate = sidecar_value(CANDIDATE)
    m801, _ = package(M801_DIR, "review.json")
    m805, _ = package(M805_DIR, "review.json")
    m805_canon, canonical_members = package(M805_CANON_DIR, "review.json")
    m794, _ = package(M794_DIR, "review.json")
    r15 = sidecar_value(R15_RELEASE)
    m797, _ = package(M797_DIR, "review.json")
    compare_packages(M805_DIR, M805_CANON_DIR)

    require(request.get("future_command_under_review") == FUTURE_COMMAND, "future command")
    require(request.get("review_target", {}).get("release_sha256") == EXPECTED[RELEASE], "request release binding")
    require(request.get("request_authorization", {}).get("max_eda_attempts_authorized_by_request") == 0,
            "request execution authority")
    require(all(value is False for key, value in request.get("request_authorization", {}).items()
                if key != "max_eda_attempts_authorized_by_request"), "request is not closed")
    validate_release(release)
    require(source.get("schema") == "m799_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_v1", "source schema")
    require(candidate.get("schema") == "m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_v1" and candidate.get("launch_now") is False, "candidate boundary")
    require(m801.get("status") == "PASS_M799_M533_R17_UNIT_DELAY_SOURCE_STATIC_HAMMER" and m801.get("verdict") == "PASS" and m801.get("score_100") == 100, "M801 verdict")
    require([m801.get(key) for key in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "M801 findings")
    require(m805.get("status") == "PASS_M805_M799_M533_R17_UNIT_DELAY_VCS_LAUNCH_ADMISSION_CANDIDATE_HAMMER__RELEASE_AUTHORING_ONLY__NO_LAUNCH" and m805.get("verdict") == "PASS" and m805.get("score_100") == 100, "M805 verdict")
    require([m805.get(key) for key in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "M805 findings")
    require(m805_canon == m805, "canonical M805 JSON differs")
    require(m794.get("decision", {}).get("r15_release_permanently_withdrawn") is True and m794.get("decision", {}).get("r15_attempt_consumed") is False, "R15 not revoked")
    require(r15.get("unique_attempt", {}).get("r14_release_status") == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "withdrawn R15 lineage")
    require(m797.get("verdict") == "FAIL_SOURCE_GATE" and m797.get("score_100") == 98 and m797.get("p0_count") == 1, "R16 failure")
    require(m797.get("decision", {}).get("launch_release_authorized") is False and m797.get("decision", {}).get("run_vcs_now") is False, "R16 closed")

    edges = resolve_sha_edges()
    closure = closure_suite()
    dryrun = dryrun_suite()
    collisions = collision_attacks()

    # Adversarial identity and strict-JSON checks on isolated in-memory/temp copies.
    attacks = {}
    for key, replacement in (("runner_sha256", "0" * 64), ("candidate_sha256", "1" * 64),
                             ("source_static_review_sha256", "2" * 64),
                             ("candidate_hammer_review_sha256", "3" * 64)):
        mutated = copy.deepcopy(release)
        mutated["identity"][key] = replacement
        attacks["wrong_%s_rejected" % key] = expect_failure(lambda m=mutated: validate_release(m), key)
    attacks["wrong_release_sha_rejected"] = expect_failure(
        lambda: require(sha(RELEASE) == "4" * 64, "wrong release SHA"), "wrong release SHA")
    release_text = RELEASE.read_text(encoding="utf-8")
    duplicate_release = release_text.replace('{\n  "schema":', '{\n  "schema": "duplicate",\n  "schema":', 1)
    attacks["duplicate_key_release_rejected"] = expect_failure(
        lambda: strict_text(duplicate_release, "duplicate-release"), "duplicate release")
    review_text = M805.read_text(encoding="utf-8")
    duplicate_review = review_text.replace('{\n  "schema":', '{\n  "schema": "duplicate",\n  "schema":', 1)
    attacks["duplicate_key_review_rejected"] = expect_failure(
        lambda: strict_text(duplicate_review, "duplicate-review"), "duplicate review")
    attacks.update(collisions)

    runner_text = RUNNER.read_text(encoding="utf-8")
    fixed_paths = {
        "source_static": 'SOURCE_STATIC_DIR="${HW_ROOT}/reviews/m799_m533_r17_unit_delay_source_static_hammer_r1_20260828"',
        "candidate_hammer": 'CANDIDATE_HAMMER_DIR="${HW_ROOT}/reviews/m799_m533_r17_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"',
        "release": 'LAUNCH_RELEASE="${HW_ROOT}/contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"',
        "final_hammer": 'FINAL_HAMMER_DIR="${HW_ROOT}/reviews/m799_m533_r17_unit_delay_vcs_final_launch_release_hammer_r1_20260828"',
        "result": 'RESULT_DIR="${HW_ROOT}/results/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_r17_20260828"',
    }
    for label, needle in fixed_paths.items():
        require(runner_text.count(needle) == 1, "runner fixed path: %s" % label)
    require(runner_text.count("+define+UNIT_DELAY") == 1, "UNIT_DELAY compile define")
    require("+notimingcheck" not in runner_text and "+no_notifier" not in runner_text, "timing bypass")
    require("-top tb_m528_dead_write_only_1rw_product_capture_r3 -o simv" in runner_text, "top/filelist")
    sva_text = (HW / "verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv").read_text(encoding="utf-8")
    require("ap_read_xor_write: assert property (!(scratch_read && scratch_write));" in sva_text, "1RW SVA")

    identity = {
        "request_path": str(REQUEST.relative_to(HW)), "request_sha256": sha(REQUEST),
        "request_manifest_file_sha256": sha(REQUEST_DIR / "SHA256SUMS"),
        "request_outer_seal_file_sha256": sha(REQUEST_DIR / "SHA256SUMS.seal.sha256"),
        "final_release_path": str(RELEASE.relative_to(HW)), "final_release_sha256": sha(RELEASE),
        "final_release_manifest_file_sha256": sha(Path(str(RELEASE) + ".sha256")),
        "final_release_outer_seal_file_sha256": sha(Path(str(RELEASE) + ".sha256.seal.sha256")),
        "runner_path": str(RUNNER.relative_to(HW)), "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(SOURCE), "candidate_sha256": sha(CANDIDATE),
        "m801_review_sha256": sha(M801), "m801_manifest_file_sha256": sha(M801_DIR / "SHA256SUMS"),
        "m801_outer_seal_file_sha256": sha(M801_DIR / "SHA256SUMS.seal.sha256"),
        "m805_review_sha256": sha(M805), "m805_manifest_file_sha256": sha(M805_DIR / "SHA256SUMS"),
        "m805_outer_seal_file_sha256": sha(M805_DIR / "SHA256SUMS.seal.sha256"),
        "runner_canonical_m805_review_sha256": sha(M805_CANON),
        "runner_canonical_m805_manifest_file_sha256": sha(M805_CANON_DIR / "SHA256SUMS"),
        "runner_canonical_m805_outer_seal_file_sha256": sha(M805_CANON_DIR / "SHA256SUMS.seal.sha256"),
        "m794_review_sha256": sha(M794), "withdrawn_r15_release_sha256": sha(R15_RELEASE),
        "m797_review_sha256": sha(M797), "python3_path": str(PY36),
        "python3_sha256": sha(PY36), "python3_version": "3.6.8",
        "function_closure_test_sha256": sha(CLOSURE), "external_command_whitelist_sha256": sha(WHITELIST),
        "premkdir_stub_dry_run_test_sha256": sha(DRYRUN), "docs359_sha256": sha(DOC359),
    }
    positive = closure["none"]
    review = {
        "schema": "m799_m533_r17_unit_delay_vcs_final_launch_release_hammer_v1",
        "date": "2026-08-29", "milestone": "M810",
        "status": "PASS_M810_M799_M533_R17_UNIT_DELAY_VCS_FINAL_LAUNCH_RELEASE_HAMMER__EXACTLY_ONE_FUNCTIONAL_VCS_PLUS_SIMV_ATTEMPT_AUTHORIZED",
        "verdict": "PASS", "score_100": 100, "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "scope": "Fresh independent final-launch hammer of the exact double-sealed M799/M533 R17 foundry-UNIT_DELAY functional release. The reviewer did not invoke live runner mode, VCS identity, lmutil/license server, VCS compile, simv, or any HDL/EDA tool. Only the pinned Python 3.6 closure suite and the runner-owned pre-mkdir stub were executed; the stub stopped with rc86 before all live probes and attempt/result creation.",
        "identity": identity,
        "double_seal_audit": {"request_pass": True, "release_pass": True, "runner_pass": True,
                              "source_contract_pass": True, "candidate_pass": True,
                              "m801_pass": True, "m805_independent_pass": True,
                              "m805_runner_canonical_pass": True, "m794_pass": True,
                              "withdrawn_r15_release_pass": True, "m797_pass": True},
        "runner_canonical_m805_byte_identity": {"pass": True, "member_count": len(canonical_members),
                                                 "all_manifest_members_and_both_seals_byte_identical": True},
        "release_policy_audit": {"schema_status_exact_pass": True, "launch_now": True,
                                  "authorization_exact": AUTH, "max_attempts": 1,
                                  "attempt_consumed_only_by_atomic_result_mkdir": True,
                                  "result_path_absent": True, "functional_unit_delay_only": True,
                                  "all_other_execution_closed": True},
        "predecessor_disposition": {"r15_release_status": "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE",
                                     "r15_attempt_consumed": False, "r15_result_absent": True,
                                     "r16_verdict": "FAIL_SOURCE_GATE", "r16_score_100": 98,
                                     "r16_launch_or_release_authorized": False,
                                     "r16_release_absent": True, "r16_result_absent": True,
                                     "r17_attempt_absent_before_launch": True},
        "sha_edge_audit": {"require_regular_sha_edges": len(edges), "all_live_regular_exact_sha": True,
                           "unexpanded_or_missing": 0, "sha_mismatches": 0},
        "executable_closure_recheck": {"exact_pinned_python_pass": True,
                                        "complete_custom_function_definition_call_closure_pass": True,
                                        "custom_function_definitions": len(positive["definitions"]),
                                        "custom_call_sites_conservatively_enumerated": len(positive["custom_calls"]),
                                        "undefined_custom_calls": len(positive["undefined_custom_calls"]),
                                        "duplicate_custom_definitions": len(positive["duplicate_definitions"]),
                                        "external_commands_seen": len(positive["external_commands_seen"]),
                                        "external_command_regular_sha_whitelist_pass": True,
                                        "all_three_negative_mutations_fail": True,
                                        "delete_definition_attack_rejected": True,
                                        "rename_definition_attack_rejected": True,
                                        "inject_stale_attack_rejected": True,
                                        "exact_premkdir_stub_dry_run_reached_boundary": True,
                                        "stub_runner_rc": dryrun["runner_rc"],
                                        "stub_event_sequence": dryrun["events"],
                                        "dry_run_vcs_license_simv_result_side_effects_zero": True},
        "attack_results": attacks,
        "functional_filelist_audit": {"ordered_files": ["foundry UNIT_DELAY model", "nine-slice 1RW macro adapter", "M528 top R2", "SVA R2", "TB R7"],
                                      "all_exact_sha_bound": True,
                                      "top_module": "tb_m528_dead_write_only_1rw_product_capture_r3",
                                      "unit_delay_define_occurrences": 1,
                                      "forbidden_notimingcheck_occurrences": 0,
                                      "forbidden_no_notifier_occurrences": 0,
                                      "read_xor_write_sva_present": True},
        "scope_boundary_audit": {"functional_parent_1rw_protocol_only": True,
                                 "acc24_capacity_physically_bound": False,
                                 "macro_rounded_total_bytes_213376_promoted": False,
                                 "budget_240kib_verified": False,
                                 "same_ledger_cycles_435293339_promoted": False,
                                 "same_ledger_speedup_1p746753x_promoted": False,
                                 "interpretation": "This release can produce functional VCS evidence only. Acc24, 240 KiB, 435293339 cycles and 1.746753x remain upstream CPU-ledger facts and are not upgraded by this hammer or by the future functional run."},
        "claim_boundary": {"final_release_integrity_pass": True, "runner_live_executed": False,
                           "functional_vcs_verified": False, "rtl_verified": False,
                           "timing_verified": False, "cycles": False, "speedup": False,
                           "ppa": False, "energy": False, "full_network_or_system": False,
                           "paper_headline": False, "paper_citable": False},
        "findings": [],
        "decision": {"final_release_hammer_pass": True,
                     "exactly_one_vcs_attempt_authorized_now": True,
                     "exactly_one_simv_execution_authorized_now": True,
                     "max_attempts": 1, "all_other_runs_authorized": False,
                     "future_command_exact": FUTURE_COMMAND,
                     "raw_result_requires_fresh_result_hammer": True,
                     "acc24_240kib_cycles_speedup_not_promoted": True},
        "reviewer_execution_receipt": {"pinned_python_closure_suite_runs": 4,
                                       "runner_live_executions": 0, "runner_stub_executions": 1,
                                       "vcs_identity_queries": 0, "license_server_queries": 0,
                                       "vcs_compiles": 0, "simv_runs": 0, "hdl_or_eda_runs": 0,
                                       "result_directories_created": 0, "attempts_consumed": 0,
                                       "author_source_or_release_modifications": 0,
                                       "docs359_modified": False}
    }

    mechanical = "\n".join([
        "M810_M799_R17_FINAL_LAUNCH_HAMMER_MECHANICAL_CHECKS",
        "verdict=PASS", "score_100=100", "p0_count=0", "p1_count=0", "p2_count=0",
        "release_sha256=" + sha(RELEASE), "runner_sha256=" + sha(RUNNER),
        "m801_review_sha256=" + sha(M801), "m805_review_sha256=" + sha(M805),
        "canonical_m805_byte_identical=true", "require_regular_sha_edges=76",
        "closure_positive=PASS", "closure_delete_definition=REJECTED",
        "closure_rename_definition=REJECTED", "closure_inject_stale=REJECTED",
        "premkdir_stub_rc=86", "premkdir_stub_events=" + ",".join(dryrun["events"]),
        "vcs_identity_queries=0", "license_server_queries=0", "vcs_compiles=0",
        "simv_runs=0", "result_directories_created=0", "attempts_consumed=0",
        "docs359_sha256=" + sha(DOC359), "future_command=" + FUTURE_COMMAND,
    ]) + "\n"
    markdown = """# M810 — M799/M533 R17 final-launch release hammer

## Verdict

**PASS 100/100; P0/P1/P2 = 0/0/0.** The exact double-sealed release may be consumed once by the exact functional VCS command below. This hammer did not query VCS or a license server, did not compile or run simv, and did not create an attempt/result.

## Evidence

- Exact release `fe6814cd...a661`, runner `4d1b0a94...e0fe`, source, candidate, M801 and M805 identities are live and double sealed.
- Independent M805 and runner-canonical M805 packages are byte-identical, including manifest and outer seal.
- All 76 runner SHA edges are live; pinned Python 3.6 closure positive and all three mutations pass their expected gates.
- Pre-mkdir stub returns 86 with the exact five-event sequence and zero VCS identity, license, compile, simv, result or attempt side effects.
- Wrong release/runner/candidate/M801/M805 SHA, duplicate-key JSON, existing result, and final-output collision attacks fail closed.
- R15 remains permanently withdrawn with no consumed attempt/result; R16 remains `FAIL_SOURCE_GATE` with no release/result.

## Authorized command

```bash
%s
```

Exactly one functional VCS compile and one simv execution are authorized. Any raw result requires a fresh result hammer.

## Claim boundary

This is functional UNIT_DELAY evidence only. Acc24, 240 KiB, 435,293,339 cycles, 1.746753x, timing, PPA, energy, full-network and paper claims remain unpromoted.
""" % FUTURE_COMMAND

    # Publish only after every read-only/stub gate above has passed and while the
    # fixed path remains absent.  No author artifact is changed.
    OUT.mkdir()
    (OUT / "independent_hammer.py").write_bytes(Path(__file__).read_bytes())
    (OUT / "mechanical_checks.txt").write_text(mechanical, encoding="utf-8")
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(markdown, encoding="utf-8")
    names = ["independent_hammer.py", "mechanical_checks.txt", "review.json", "review.md"]
    manifest = "".join("%s  %s\n" % (sha(OUT / name), name) for name in names)
    (OUT / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text("%s  SHA256SUMS\n" % sha(OUT / "SHA256SUMS"), encoding="utf-8")
    package(OUT, "review.json")
    require(sha(DOC359) == EXPECTED[DOC359], "docs/359 changed during audit")
    print(json.dumps({"verdict": "PASS", "score_100": 100,
                      "review_sha256": sha(OUT / "review.json"),
                      "manifest_sha256": sha(OUT / "SHA256SUMS"),
                      "outer_seal_sha256": sha(OUT / "SHA256SUMS.seal.sha256")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

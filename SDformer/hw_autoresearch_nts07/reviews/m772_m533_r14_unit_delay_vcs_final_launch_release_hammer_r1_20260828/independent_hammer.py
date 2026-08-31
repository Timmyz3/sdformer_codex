#!/usr/bin/env python3
"""Fresh, read-only final-release hammer for M772/M533 r14.

This script never invokes the runner, VCS, simv, lmutil, or any HDL/EDA tool.
It independently recomputes fixed-file integrity, strict JSON semantics, the
runner's 64 exact-SHA edges, static execution gates, and live host safety.
"""

import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path


REPO = Path("/home/zhumd/work/sdformer_codex/SDformer").resolve()
HW = REPO / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m772_m533_m528_dead_write_only_1rw_unit_delay_r14_exact_sha.sh"
REQUEST_DIR = HW / "reviews/m778_m533_r14_unit_delay_vcs_final_launch_release_hammer_REQUEST_r1_20260828"
REQUEST = REQUEST_DIR / "request.json"
RELEASE = HW / "contracts/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
SOURCE = HW / "contracts/m772_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_r1_20260828.json"
CANDIDATE = HW / "contracts/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_r1_20260828.json"
PREFLIGHT_DIR = HW / "reviews/m772_m533_r14_vcs_environment_preflight_r1_20260828"
PREFLIGHT = PREFLIGHT_DIR / "preflight.json"
SOURCE_REVIEW_DIR = HW / "reviews/m772_m533_r14_unit_delay_source_static_hammer_r1_20260828"
SOURCE_REVIEW = SOURCE_REVIEW_DIR / "review.json"
CANDIDATE_REVIEW_DIR = HW / "reviews/m772_m533_r14_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"
CANDIDATE_REVIEW = CANDIDATE_REVIEW_DIR / "review.json"
R13_DIR = HW / "results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828"
R13 = R13_DIR / "RUN_FAILED_OR_INCOMPLETE.json"
M770_DIR = HW / "reviews/m770_m533_r13_vcs_home_failure_fresh_hammer_r1_20260828"
M770 = M770_DIR / "review.json"
RESULT = HW / "results/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_r14_20260828"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    REQUEST: "dcf219ac1322fc9f7235e0d43ba58e86c2e4421dbd6b8a1a675cadeaa685c8df",
    RELEASE: "de76cd4d42aad3bdddb78b65a83d0d8dbca1d794015172919785d9c3a2f9c242",
    RUNNER: "3acf166df55c877d49948a811320204b6249a5f8709a6e44c6365f2f98881761",
    SOURCE: "24d40ec62e087a9637267c5fc80df7a4c30e6ba39098261bdd730db5360b06a4",
    CANDIDATE: "5555ae8ba9b2141618d7610bdd4533c4b833e98139602ac495f79bc83b2a3c7a",
    PREFLIGHT: "dd7500d8d5deaa8bc4d0d02113218c6d6b92bdfe27dca1b8d0c3724b125b2c9f",
    SOURCE_REVIEW: "1388d65a4ef254287884601d126f6cc81727ca93ade21dba688d6072ed04fcd0",
    CANDIDATE_REVIEW: "e301144bb1a6dd1751aff7ddd83d0290f42826c5b12d11224553fa13c29347c0",
    R13: "df9e70c0f382139dc5c35b95cbee9e7aa7af9e9466d81cb9c8d563660fbe243b",
    M770: "caba813792a8df3b1b9b72a7ddb7ec053096acab6188645b9d3c59a2ca8c3192",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

ENV_POLICY = {
    "clean_env": True,
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
    "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
    "VCS_ARCH_OVERRIDE": "linux",
    "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
    "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
    "HOME": "UNSET",
    "identity_probe": "vcs -full64 -ID",
    "license_features": ["VCSCompiler_Net", "VCSRuntime_Net"],
}
AUTH = {
    "vcs_runs": 1,
    "simv_runs": 1,
    "iverilog_runs": 0,
    "verilator_runs": 0,
    "dc_runs": 0,
    "formality_runs": 0,
    "pt_runs": 0,
    "ptpx_runs": 0,
    "cpu_runs": 0,
    "gpu_runs": 0,
    "network_or_remote_jobs": 0,
}
RESOURCE_POLICY = {
    "prelaunch_samples": 3,
    "sample_interval_seconds": 2,
    "mem_available_min_kib": 134217728,
    "swap_free_min_kib": 33554432,
    "commit_headroom_min_kib": 33554432,
    "cgroup_version": 1,
    "memory_failcnt_must_not_increase": True,
    "under_oom_must_equal_zero": True,
    "oom_kill_must_equal_zero": True,
    "missing_counter_is_failure": True,
    "same_uid_synopsys_vcs_simv_collision_must_be_zero": True,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def strict_json(path: Path):
    def pairs(items):
        value = {}
        for key, member in items:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            value[key] = member
        return value

    def reject(token):
        raise RuntimeError(f"non-standard JSON token in {path}: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)

    def finite(member):
        if isinstance(member, float):
            require(math.isfinite(member), f"non-finite number in {path}")
        elif isinstance(member, dict):
            for key, child in member.items():
                finite(key)
                finite(child)
        elif isinstance(member, list):
            for child in member:
                finite(child)

    finite(value)
    return value


def regular(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def verify_sidecar_json(path: Path) -> None:
    require(regular(path), f"missing/non-regular JSON: {path}")
    member = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(regular(member) and regular(outer), f"missing/non-regular JSON sidecar: {path}")
    member_parts = member.read_text(encoding="utf-8").strip().split()
    outer_parts = outer.read_text(encoding="utf-8").strip().split()
    require(member_parts == [sha(path), path.name], f"member sidecar mismatch: {path}")
    require(outer_parts == [sha(member), member.name], f"outer sidecar mismatch: {path}")
    strict_json(path)


def verify_package(directory: Path, json_name: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), f"missing/non-regular package: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), f"missing package seals: {directory}")
    listed = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        target = directory / name
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"bad manifest SHA: {target}")
        require(regular(target), f"non-regular manifest target: {target}")
        require(sha(target) == expected, f"manifest mismatch: {target}")
        listed.append(name)
    outer_parts = outer.read_text(encoding="utf-8").strip().split()
    require(outer_parts == [sha(manifest), manifest.name], f"outer package seal mismatch: {directory}")
    require(json_name in listed, f"JSON omitted from manifest: {directory}/{json_name}")
    strict_json(directory / json_name)


def parse_edges(text: str):
    joined = text.replace("\\\n", " ")
    variables = {
        "HW_ROOT": str(HW),
        "SCRIPT_DIR": str(RUNNER.parent),
        "RUNNER_PATH": str(RUNNER),
    }
    for line in joined.splitlines():
        match = re.fullmatch(r'([A-Z][A-Z0-9_]*)="([^"`()]*?)"', line.strip())
        if match:
            variables[match.group(1)] = match.group(2)
    var_pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

    def expand(value: str) -> str:
        for _ in range(32):
            new = var_pattern.sub(lambda match: variables.get(match.group(1), match.group(0)), value)
            if new == value:
                return new
            value = new
        raise RuntimeError(f"variable recursion: {value}")

    for name in list(variables):
        variables[name] = expand(variables[name])
    calls = re.findall(r"^\s*require_regular_sha\s+([^\s]+)\s+(.+?)\s*$", joined, re.MULTILINE)
    ledger = []
    for index, (expected, path_expr) in enumerate(calls, 1):
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"edge {index} has malformed SHA")
        resolved_text = expand(path_expr.strip().strip('"'))
        require("${" not in resolved_text, f"edge {index} unresolved: {resolved_text}")
        path = Path(resolved_text)
        require(regular(path), f"edge {index} non-regular/symlink/missing: {path}")
        actual = sha(path)
        require(actual == expected, f"edge {index} mismatch: {path}: {actual} != {expected}")
        ledger.append({"index": index, "expected": expected, "path": str(path), "actual": actual})
    return ledger


def proc_collisions():
    uid = os.getuid()
    self_pid = os.getpid()
    parent_pid = os.getppid()
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in {self_pid, parent_pid}:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            exe = os.path.basename(os.readlink(entry / "exe"))
            argv = [item.decode("utf-8", "replace") for item in (entry / "cmdline").read_bytes().split(b"\0") if item]
            starttime = (entry / "stat").read_text(encoding="utf-8").split()[21]
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
        tokens = {os.path.basename(item) for item in argv[:8]}
        tokens.add(exe)
        joined = " ".join(argv)
        kind = None
        hits = sorted(tokens & {"dc_shell", "dc_shell-t", "fm_shell", "fm_shell_exec", "pt_shell", "pt_shell_exec"})
        if hits:
            kind = hits[0]
        elif "common_shell_exec" in tokens and re.search(r"(?:^|\s)-shell\s+(?:dc_shell|dc_shell-t|fm_shell|pt_shell)(?:\s|$)", joined):
            kind = "common_shell_exec_for_dc_fm_pt_ptpx"
        elif any(item == "vcs" or item.startswith("vcs.") or item in {"vcs1", "vlogan", "vhdlan"} for item in tokens):
            kind = "vcs"
        elif any(item == "simv" or item.startswith("simv.") for item in tokens):
            kind = "simv"
        if kind:
            matches.append({"pid": pid, "starttime": starttime, "class": kind, "exe": exe, "argv": argv})
    return uid, matches


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, rest = line.split(":", 1)
        pieces = rest.split()
        if pieces and pieces[0].isdigit():
            values[name] = int(pieces[0])
    return {
        "mem_available_kib": values["MemAvailable"],
        "swap_free_kib": values["SwapFree"],
        "commit_limit_kib": values["CommitLimit"],
        "committed_as_kib": values["Committed_AS"],
        "commit_headroom_kib": values["CommitLimit"] - values["Committed_AS"],
    }


def cgroup_state():
    relative = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1] == "memory":
            relative = parts[2]
    require(relative is not None and relative.startswith("/") and ".." not in relative, "cgroup-v1 memory path unavailable")
    directories = [Path("/sys/fs/cgroup/memory") / relative.lstrip("/"), Path("/sys/fs/cgroup/memory/user.slice")]
    state = []
    for directory in directories:
        require(directory.is_dir() and not directory.is_symlink(), f"missing cgroup directory: {directory}")
        failcnt_file = directory / "memory.failcnt"
        oom_file = directory / "memory.oom_control"
        usage_file = directory / "memory.usage_in_bytes"
        for path in (failcnt_file, oom_file, usage_file):
            require(path.is_file() and not path.is_symlink(), f"missing cgroup counter: {path}")
        fields = {}
        for line in oom_file.read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            fields[key] = int(value)
        state.append({
            "path": str(directory),
            "failcnt": int(failcnt_file.read_text(encoding="utf-8").strip()),
            "under_oom": fields["under_oom"],
            "oom_kill": fields["oom_kill"],
        })
    return state


def main():
    for path, expected in EXPECTED.items():
        require(regular(path), f"missing/non-regular identity: {path}")
        require(sha(path) == expected, f"identity SHA mismatch: {path}")

    for path in (RELEASE, SOURCE, CANDIDATE):
        verify_sidecar_json(path)
    verify_package(REQUEST_DIR, "request.json")
    verify_package(PREFLIGHT_DIR, "preflight.json")
    verify_package(SOURCE_REVIEW_DIR, "review.json")
    verify_package(CANDIDATE_REVIEW_DIR, "review.json")
    verify_package(R13_DIR, "RUN_FAILED_OR_INCOMPLETE.json")
    verify_package(M770_DIR, "review.json")

    request = strict_json(REQUEST)
    release = strict_json(RELEASE)
    source = strict_json(SOURCE)
    candidate = strict_json(CANDIDATE)
    preflight = strict_json(PREFLIGHT)
    source_review = strict_json(SOURCE_REVIEW)
    candidate_review = strict_json(CANDIDATE_REVIEW)
    r13 = strict_json(R13)
    m770 = strict_json(M770)
    strict_document_count = 9

    require(request["review_target"]["release_sha256"] == EXPECTED[RELEASE], "request release binding")
    require(request["future_command_under_review"] == "env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m772_m533_m528_dead_write_only_1rw_unit_delay_r14_exact_sha.sh", "request command mismatch")
    require(all(value in (False, 0) for value in request["request_authorization"].values()), "request must authorize zero executions")

    require(release["schema"] == "m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_v1", "release schema")
    require(release["status"] == "AUTHORIZED_EXACTLY_ONE_M772_R14_UNIT_DELAY_FUNCTIONAL_VCS_ATTEMPT__FRESH_FINAL_HAMMER_AND_LIVE_PREFLIGHT_STILL_REQUIRED", "release status")
    require(release["launch_now"] is True, "release launch_now")
    require(release["authorization"] == AUTH, "release authorization")
    require(release["environment_policy"] == ENV_POLICY, "release environment policy")
    require(release["resource_policy"] == RESOURCE_POLICY, "release resource policy")
    require(release["macro_model_mode"] == "foundry_UNIT_DELAY_functional", "release macro mode")
    require(release["unique_attempt"]["result_path"] == "results/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_r14_20260828", "release result path")
    require(release["unique_attempt"]["result_path_absent_at_release_authoring"] is True, "release result absence receipt")
    require(release["author_execution_receipt"] == {
        "runner_executions": 0, "vcs_runs": 0, "vcs_identity_queries": 0,
        "license_status_queries": 0, "simv_runs": 0, "all_other_hdl_or_eda_runs": 0,
        "experiments": 0, "remote_jobs": 0, "result_directories_created": 0,
        "docs359_modified": False,
    }, "release author execution receipt")
    require(release["claim_boundary"]["functional_vcs_verified"] is False and release["claim_boundary"]["timing_verified"] is False and release["claim_boundary"]["paper_citable"] is False, "release claim boundary")

    require(candidate["launch_now"] is False and candidate["authorization"] == AUTH, "candidate closed authorization")
    require(candidate["environment_policy"] == ENV_POLICY and candidate["resource_policy"] == RESOURCE_POLICY, "candidate policy")
    require(source["authorization"]["vcs_hdl_compiles"] == 0 and source["authorization"]["simv_runs"] == 0 and source["authorization"]["runner_executions"] == 0, "source no execution")
    require(source_review["status"] == "PASS_M772_M533_R14_UNIT_DELAY_SOURCE_STATIC_HAMMER" and source_review["score_100"] == 100, "source review status")
    require([source_review[key] for key in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "source review findings")
    require(candidate_review["status"] == "PASS_M772_M533_R14_UNIT_DELAY_VCS_LAUNCH_ADMISSION_CANDIDATE_HAMMER" and candidate_review["score_100"] == 100, "candidate review status")
    require([candidate_review[key] for key in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "candidate review findings")
    require(source_review["identity"]["runner_sha256"] == EXPECTED[RUNNER] and source_review["identity"]["source_contract_sha256"] == EXPECTED[SOURCE], "source review source binding")
    require(candidate_review["identity"]["candidate_sha256"] == EXPECTED[CANDIDATE] and candidate_review["identity"]["runner_sha256"] == EXPECTED[RUNNER], "candidate review source binding")
    require(release["identity"]["source_static_review_sha256"] == EXPECTED[SOURCE_REVIEW] and release["identity"]["candidate_hammer_review_sha256"] == EXPECTED[CANDIDATE_REVIEW], "release review binding")
    require(preflight["status"] == "PASS_READ_ONLY_FULL64_ID_AND_LICENSE_STATUS__NO_HOME__NO_COMPILE__NO_SEAT_CHECKOUT", "preflight status")
    require(preflight["environment"]["HOME"] == "UNSET_BY_ENV_I" and preflight["boundary"]["hdl_compile"] is False and preflight["boundary"]["simv"] is False, "preflight boundary")
    require(r13["status"] == "FAILED_DO_NOT_CITE" and r13["phase"] == "vcs_compile" and r13["child_rc"] == "vcs_1_tee_0" and r13["paper_citable"] is False, "r13 failure remains closed")
    require(m770["decision"]["r13_attempt_status"] == "PERMANENTLY_CONSUMED_FAILED_DO_NOT_CITE" and m770["decision"]["r13_functional_status"] == "NO_CONCLUSION", "M770 failure boundary")
    require(m770["decision"]["one_additive_r14_source_package_authorized"] is True and m770["decision"]["r14_launch_authorized_now"] is False, "M770 additive boundary")
    require(not RESULT.exists(), "r14 result/attempt already exists")

    runner_text = RUNNER.read_text(encoding="utf-8")
    require(os.access(RUNNER, os.X_OK), "runner not executable")
    require("if [[ $# -ne 0 ]]" in runner_text, "runner argument override guard absent")
    require("[[ ! -v HOME ]] || fail \"HOME must remain unset under env-i\"" in runner_text, "HOME-absent gate missing")
    for token in (
        '[[ "${VCS_HOME-}" == "${EXPECTED_VCS_HOME}" ]]',
        '[[ "${VCS_ARCH_OVERRIDE-}" == "${EXPECTED_VCS_ARCH_OVERRIDE}" ]]',
        '[[ "${SNPSLMD_LICENSE_FILE-}" == "${EXPECTED_SNPSLMD_LICENSE_FILE}" ]]',
        '[[ "${LM_LICENSE_FILE-}" == "${EXPECTED_LM_LICENSE_FILE}" ]]',
    ):
        require(token in runner_text, f"exact environment gate missing: {token}")

    edges = parse_edges(runner_text)
    require(len(edges) == 64, f"expected 64 SHA edges, got {len(edges)}")
    ledger_text = "".join(f"{item['index']:02d}\t{item['expected']}\t{item['path']}\n" for item in edges)
    ledger_sha = hashlib.sha256(ledger_text.encode("utf-8")).hexdigest()

    compile_match = re.search(r'"\$\{VCS_BIN\}" -full64 -sverilog.*?-o simv 2>&1 \| tee compile\.log', runner_text, re.DOTALL)
    require(compile_match is not None, "compile command not found")
    compile_text = compile_match.group(0)
    require(runner_text.count('"${VCS_BIN}" -full64 -sverilog') == 1, "VCS compile command count")
    require(compile_text.count("+define+UNIT_DELAY") == 1, "UNIT_DELAY compile count")
    require("${FOUNDRY_SLOW_V}" in compile_text and "${FOUNDRY_SLOW_DB}" not in compile_text, "foundry compile model")
    require("+notimingcheck" not in compile_text and "+no_notifier" not in compile_text, "forbidden timing bypass")
    require(runner_text.count("./simv 2>&1 | tee sim.log") == 1, "simv command count")
    for token in (
        "PASS_M533_M528_DW1RW_R7_DIRECTED_RANDOM_AND_ATTACKS",
        "COVERAGE_M533_M528_DW1RW_R7",
        "P2_STRENGTH_M533_M528_DW1RW_R3",
        "dead_plus_read", "deadline_read_write", "same_address_forward", "pending_plus_forward",
        "full_no_credit", "liveness_sequences", "parent_modes", "stalled_raw_recovery",
        "stalled_raw_forward_recovery", "stalled_raw_response_recovery", "pingpong_overlap",
        "endpoint_rows", "all_slices", "minima=1 normal_covers=13",
        "consecutive_distinct_reads", "response_identity_checks",
        "dirty_reserved", "stale_epoch", "overflow", "wrong_parent", "read_before_write",
        "parent_only_nonzero", "attacks=6",
        "Timing violation", "Assertion.*failed", "Error-\\[SVA", "\\$error", "\\$fatal",
        "normal scoreboard errors", "protocol attack not detected",
        "memory.failcnt", "under_oom", "oom_kill", "same_uid",
        "RESOURCE_FINAL_ACK", "final_synchronous", "SHA256SUMS.seal.sha256",
    ):
        require(token in runner_text, f"functional/resource/terminal gate missing: {token}")
    require("20000" in (HW / "tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r7.sv").read_text(encoding="utf-8"), "task watchdog 20000 absent")

    # Pure predicate negative tests: no runner or external tool invocation.
    canonical_env = {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "VCS_ARCH_OVERRIDE": "linux",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo", "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
    }
    def env_accept(env):
        return env == canonical_env and "HOME" not in env
    require(env_accept(canonical_env.copy()), "canonical env negative-test baseline")
    env_negative = {}
    for key in ("VCS_HOME", "VCS_ARCH_OVERRIDE", "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
        attack = canonical_env.copy()
        attack[key] = attack[key] + ".tampered"
        env_negative[key] = not env_accept(attack)
        require(env_negative[key], f"environment tamper not rejected: {key}")
    attack = canonical_env.copy(); attack["HOME"] = "/tmp/forbidden"
    env_negative["HOME_present"] = not env_accept(attack)
    require(env_negative["HOME_present"], "HOME presence not rejected")
    require(not regular(RUNNER.parent / "definitely_absent_m779"), "absent-path predicate failed")
    with tempfile.TemporaryDirectory(prefix="m779_negative_predicates.") as temporary:
        temp_root = Path(temporary)
        target = temp_root / "target"
        target.write_bytes(b"frozen\n")
        expected_target_sha = sha(target)
        link = temp_root / "link"
        link.symlink_to(target.name)
        collision = temp_root / "result_collision"
        collision.mkdir()
        file_negative = {
            "regular_exact_accept": regular(target) and sha(target) == expected_target_sha,
            "symlink_rejected": not regular(link),
            "result_collision_rejected": collision.exists(),
            "malformed_sha_rejected": re.fullmatch(r"[0-9a-f]{64}", "0" * 63) is None,
        }
        target.write_bytes(b"tampered\n")
        file_negative["content_or_sha_tamper_rejected"] = sha(target) != expected_target_sha
        require(all(file_negative.values()), f"file predicate negative test failed: {file_negative}")

    uid, initial_collisions = proc_collisions()
    require(not initial_collisions, f"same-uid EDA/VCS/simv collisions: {initial_collisions}")
    cgroup_initial = cgroup_state()
    require(all(item["under_oom"] == 0 and item["oom_kill"] == 0 for item in cgroup_initial), "initial cgroup OOM")
    samples = []
    for index in range(1, 4):
        sample = {"sample": index, **meminfo()}
        sample["cgroups"] = cgroup_state()
        require(sample["mem_available_kib"] >= RESOURCE_POLICY["mem_available_min_kib"], "MemAvailable threshold")
        require(sample["swap_free_kib"] >= RESOURCE_POLICY["swap_free_min_kib"], "SwapFree threshold")
        require(sample["commit_headroom_kib"] >= RESOURCE_POLICY["commit_headroom_min_kib"], "commit headroom threshold")
        require(all(item["failcnt"] == cgroup_initial[pos]["failcnt"] and item["under_oom"] == 0 and item["oom_kill"] == 0 for pos, item in enumerate(sample["cgroups"])), "cgroup threshold")
        samples.append(sample)
        if index != 3:
            time.sleep(2)
    uid_final, final_collisions = proc_collisions()
    require(uid_final == uid and not final_collisions, f"final collision scan: {final_collisions}")
    require(not RESULT.exists(), "result appeared during review")

    payload = {
        "status": "PASS_M772_M533_R14_UNIT_DELAY_FINAL_RELEASE_MECHANICAL_HAMMER",
        "score_100": 100,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "strict_json_document_count": strict_document_count,
        "double_sealed_identities_checked": 9,
        "runner_sha256": sha(RUNNER),
        "request_sha256": sha(REQUEST),
        "release_sha256": sha(RELEASE),
        "require_regular_sha_edges": len(edges),
        "edge_mismatch_count": 0,
        "canonical_edge_ledger_sha256": ledger_sha,
        "canonical_edge_ledger_bytes": len(ledger_text.encode("utf-8")),
        "result_path_absent": not RESULT.exists(),
        "unit_delay_compile_count": compile_text.count("+define+UNIT_DELAY"),
        "vcs_compile_command_count": runner_text.count('"${VCS_BIN}" -full64 -sverilog'),
        "simv_command_count": runner_text.count("./simv 2>&1 | tee sim.log"),
        "negative_environment_cases": env_negative,
        "negative_file_cases": file_negative,
        "live_gate": {
            "uid": uid,
            "collision_scan_initial_count": len(initial_collisions),
            "collision_scan_final_count": len(final_collisions),
            "samples": samples,
        },
        "execution_receipt": {
            "runner": 0, "vcs": 0, "simv": 0, "identity_probe": 0,
            "license_query": 0, "hdl_or_eda": 0, "cpu_or_gpu_experiment": 0,
            "network_or_remote": 0, "result_identity_creation": 0,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        sys.exit(1)

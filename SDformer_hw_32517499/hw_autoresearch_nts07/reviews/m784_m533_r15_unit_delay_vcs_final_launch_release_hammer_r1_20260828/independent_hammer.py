#!/usr/bin/env python3
"""Fresh, read-only final-launch hammer for M784/M533 r15.

This program never invokes the runner, VCS, simv, lmutil, a license server, or
any HDL/EDA tool.  It checks the frozen release graph, executes only the exact
embedded M770 Python predicate in isolation, and samples shared-host safety.
"""

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from pathlib import Path


REPO = Path("/home/zhumd/work/sdformer_codex/SDformer")
HW = REPO / "hw_autoresearch_nts07"
OUT = HW / "reviews/m784_m533_r15_unit_delay_vcs_final_launch_release_hammer_r1_20260828"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m784_m533_m528_dead_write_only_1rw_unit_delay_r15_exact_sha.sh"
R14_RUNNER = HW / "dc_handoff/scripts/run_vcs_m772_m533_m528_dead_write_only_1rw_unit_delay_r14_exact_sha.sh"
REQUEST_DIR = HW / "reviews/m789_m533_r15_unit_delay_vcs_final_launch_release_hammer_REQUEST_r1_20260828"
REQUEST = REQUEST_DIR / "request.json"
RELEASE = HW / "contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
SOURCE = HW / "contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_r1_20260828.json"
CANDIDATE = HW / "contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_r1_20260828.json"
SOURCE_REVIEW_DIR = HW / "reviews/m784_m533_r15_unit_delay_source_static_hammer_r1_20260828"
SOURCE_REVIEW = SOURCE_REVIEW_DIR / "review.json"
CANDIDATE_REVIEW_DIR = HW / "reviews/m784_m533_r15_unit_delay_vcs_launch_admission_candidate_hammer_r1_20260828"
CANDIDATE_REVIEW = CANDIDATE_REVIEW_DIR / "review.json"
M782_DIR = HW / "reviews/m782_m533_r14_premkdir_launch_boundary_failure_hammer_r1_20260828"
M782 = M782_DIR / "review.json"
R14_RELEASE = HW / "contracts/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
M779_DIR = HW / "reviews/m772_m533_r14_unit_delay_vcs_final_launch_release_hammer_r1_20260828"
M779 = M779_DIR / "review.json"
PREFLIGHT_DIR = HW / "reviews/m772_m533_r14_vcs_environment_preflight_r1_20260828"
PREFLIGHT = PREFLIGHT_DIR / "preflight.json"
R13_DIR = HW / "results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828"
R13 = R13_DIR / "RUN_FAILED_OR_INCOMPLETE.json"
M770_DIR = HW / "reviews/m770_m533_r13_vcs_home_failure_fresh_hammer_r1_20260828"
M770 = M770_DIR / "review.json"
PREDICATE_TEST = HW / "verif_m528_dw1rw/test_m784_r15_runner_m770_embedded_predicate.py"
RESULT = HW / "results/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_r15_20260828"
R14_RESULT = HW / "results/m772_m533_m528_dead_write_only_1rw_unit_delay_vcs_r14_20260828"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    REQUEST: "f140bae358ebd08609b15038dc3ceeca02ff1e49991b10cd6bdf19661a949b3e",
    RELEASE: "6c3d4a1ffef609765a387f45bdf502510a1d0d9ded6df0b281f50668d689fd08",
    RUNNER: "0bff34247859531757d80c597e48b0f978e6d29ff27f20ec8496dcc4d1548892",
    SOURCE: "d426deafd43278be97f2999c4dc41ade2a1439da244a1c5ed13074cf6f583ec8",
    CANDIDATE: "114412ba964e46cea4210201742f7e6daf44c03c38fe244fa135e44d6fbd7782",
    SOURCE_REVIEW: "7170972816ef9dd2083f708e7c6e985f87e0f8c3aa42e864bdd4f366ab0877d6",
    CANDIDATE_REVIEW: "3fcb2e4269a59a71063678f96b55e426c9cbb9d8f1676accc3be096483bb90b4",
    M782: "ff7498279990537c7e60f886d44a3a6ec919aeb39d2fe5a9294a049f9a79bf6b",
    R14_RELEASE: "de76cd4d42aad3bdddb78b65a83d0d8dbca1d794015172919785d9c3a2f9c242",
    M779: "7601eb19acf7c11fb9899f68ecc18b74d30938e7b3d271ec7ecf80bf8e720a27",
    PREFLIGHT: "dd7500d8d5deaa8bc4d0d02113218c6d6b92bdfe27dca1b8d0c3724b125b2c9f",
    R13: "df9e70c0f382139dc5c35b95cbee9e7aa7af9e9466d81cb9c8d563660fbe243b",
    M770: "caba813792a8df3b1b9b72a7ddb7ec053096acab6188645b9d3c59a2ca8c3192",
    PREDICATE_TEST: "f1dce780f7b8302b062e2ba85c6e72752952605105d04cee6a60796c73e79706",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

AUTH = {
    "vcs_runs": 1, "simv_runs": 1, "iverilog_runs": 0, "verilator_runs": 0,
    "dc_runs": 0, "formality_runs": 0, "pt_runs": 0, "ptpx_runs": 0,
    "cpu_runs": 0, "gpu_runs": 0, "network_or_remote_jobs": 0,
}
ENV_POLICY = {
    "clean_env": True, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
    "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "VCS_ARCH_OVERRIDE": "linux",
    "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
    "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat", "HOME": "UNSET",
    "identity_probe": "vcs -full64 -ID",
    "license_features": ["VCSCompiler_Net", "VCSRuntime_Net"],
}
RESOURCE_POLICY = {
    "prelaunch_samples": 3, "sample_interval_seconds": 2,
    "mem_available_min_kib": 134217728, "swap_free_min_kib": 33554432,
    "commit_headroom_min_kib": 33554432, "cgroup_version": 1,
    "memory_failcnt_must_not_increase": True, "under_oom_must_equal_zero": True,
    "oom_kill_must_equal_zero": True, "missing_counter_is_failure": True,
    "same_uid_synopsys_vcs_simv_collision_must_be_zero": True,
}


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


def no_symlink_ancestor(path):
    path = Path(path).absolute()
    return all(not parent.is_symlink() for parent in [path, *path.parents])


def strict_json(path):
    path = Path(path)
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f"duplicate JSON key in {path}: {key}")
            out[key] = value
        return out
    def reject(token):
        raise RuntimeError(f"non-standard JSON token in {path}: {token}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    def finite(member):
        if isinstance(member, float):
            require(math.isfinite(member), f"non-finite number in {path}")
        elif isinstance(member, dict):
            for key, child in member.items():
                finite(key); finite(child)
        elif isinstance(member, list):
            for child in member: finite(child)
    finite(value)
    return value


def verify_json_sidecars(path):
    path = Path(path)
    member = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(regular(path) and regular(member) and regular(outer), f"missing JSON seal: {path}")
    require(member.read_text().split() == [sha(path), path.name], f"member seal mismatch: {path}")
    require(outer.read_text().split() == [sha(member), member.name], f"outer seal mismatch: {path}")
    return strict_json(path)


def verify_package(directory, json_name):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), f"missing package: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), f"missing package seal: {directory}")
    names = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"bad manifest SHA: {directory}")
        target = directory / name
        require(regular(target) and no_symlink_ancestor(target), f"bad manifest target: {target}")
        require(sha(target) == expected, f"manifest mismatch: {target}")
        names.append(name)
    require(outer.read_text().split() == [sha(manifest), manifest.name], f"outer package seal mismatch: {directory}")
    require(json_name in names, f"JSON omitted from package: {directory}/{json_name}")
    return strict_json(directory / json_name)


def raw_edges(path):
    text = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    return re.findall(r"^\s*require_regular_sha\s+([^\s]+)\s+(.+?)\s*$", text, re.MULTILINE)


def resolve_edges(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    variables = {"HW_ROOT": str(HW), "SCRIPT_DIR": str(path.parent), "RUNNER_PATH": str(path)}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Z][A-Z0-9_]*)="([^"`()]*)"', line.strip())
        if match: variables[match.group(1)] = match.group(2)
    pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
    def expand(value):
        for _ in range(40):
            new = pattern.sub(lambda m: variables.get(m.group(1), m.group(0)), value)
            if new == value: return new
            value = new
        raise RuntimeError(f"variable recursion: {value}")
    for key in list(variables): variables[key] = expand(variables[key])
    ledger = []
    for index, (expected, expression) in enumerate(raw_edges(path), 1):
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"edge {index} malformed")
        resolved = expand(expression.strip().strip('"'))
        require("${" not in resolved, f"edge {index} unresolved: {resolved}")
        target = Path(resolved)
        require(regular(target) and no_symlink_ancestor(target), f"edge {index} bad target: {target}")
        actual = sha(target)
        require(actual == expected, f"edge {index} mismatch: {target}: {actual} != {expected}")
        ledger.append({"index": index, "expected": expected, "path": str(target), "actual": actual})
    return ledger


def execute_real_heredoc():
    start_line = '  python3 -I - "${R13_FAILED_RECEIPT}" "${M770_REVIEW}" "${M782_REVIEW}" "${AUTHOR_ENV_PREFLIGHT}" <<\'PY2\''
    lines = RUNNER.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line == start_line]
    require(len(starts) == 1, f"M770 heredoc start count={len(starts)}")
    start = starts[0] + 1
    ends = [i for i in range(start, len(lines)) if lines[i] == "PY2"]
    require(ends, "M770 heredoc unterminated")
    code = "\n".join(lines[start:ends[0]]) + "\n"
    require('audit.get("decision", {}).get("r14_launch_authorized_now") is False' in code,
            "real predicate lacks r14 key")
    require('audit.get("decision", {}).get("vcs_launch_authorized_now")' not in code,
            "withdrawn M770 key remains")
    def run(m770_path):
        return subprocess.run(
            ["python3", "-I", "-", str(R13), str(m770_path), str(M782), str(PREFLIGHT)],
            input=code, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    good = run(M770)
    require(good.returncode == 0, f"sealed heredoc failed: {good.stderr}")
    sealed = strict_json(M770)
    with tempfile.TemporaryDirectory(prefix="m792_m784_r15_heredoc.") as raw:
        tmp = Path(raw)
        missing = json.loads(json.dumps(sealed))
        del missing["decision"]["r14_launch_authorized_now"]
        missing_path = tmp / "missing.json"
        missing_path.write_text(json.dumps(missing), encoding="utf-8")
        missing_run = run(missing_path)
        require(missing_run.returncode != 0 and "M770 launch boundary" in missing_run.stderr,
                "missing-key attack not rejected by real predicate")
        wrong = json.loads(json.dumps(sealed))
        value = wrong["decision"].pop("r14_launch_authorized_now")
        wrong["decision"]["vcs_launch_authorized_now"] = value
        wrong_path = tmp / "wrong.json"
        wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
        wrong_run = run(wrong_path)
        require(wrong_run.returncode != 0 and "M770 launch boundary" in wrong_run.stderr,
                "wrong-key attack not rejected by real predicate")
    return {"exact_start_count": 1, "sealed_positive": "PASS",
            "missing_key": "FAIL_M770_LAUNCH_BOUNDARY",
            "wrong_key": "FAIL_M770_LAUNCH_BOUNDARY", "runner_executions": 0, "eda_runs": 0}


def proc_collisions():
    uid = os.getuid(); self_pid = os.getpid(); parent_pid = os.getppid(); matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        pid = int(entry.name)
        if pid in {self_pid, parent_pid}: continue
        try:
            if entry.stat().st_uid != uid: continue
            exe = os.path.basename(os.readlink(entry / "exe"))
            argv = [x.decode("utf-8", "replace") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
            starttime = (entry / "stat").read_text().split()[21]
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
        tokens = {exe, *(os.path.basename(x) for x in argv[:8])}
        joined = " ".join(argv)
        kind = None
        direct = sorted(tokens & {"dc_shell", "dc_shell-t", "fm_shell", "fm_shell_exec",
                                  "pt_shell", "pt_shell_exec", "ptpx", "vcs", "vcs1",
                                  "vlogan", "vhdlan", "simv"})
        if direct: kind = direct[0]
        elif "common_shell_exec" in tokens and re.search(r"(?:^|\s)-shell\s+(?:dc_shell|dc_shell-t|fm_shell|pt_shell)(?:\s|$)", joined):
            kind = "common_shell_exec_for_dc_fm_pt"
        elif any(t.startswith("vcs.") for t in tokens): kind = "vcs"
        elif any(t.startswith("simv.") for t in tokens): kind = "simv"
        if kind: matches.append({"pid": pid, "starttime": starttime, "class": kind, "exe": exe, "argv": argv})
    return uid, matches


def mem_sample():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, rest = line.split(":", 1); pieces = rest.split()
        if pieces and pieces[0].isdigit(): values[name] = int(pieces[0])
    return {"mem_available_kib": values["MemAvailable"], "swap_free_kib": values["SwapFree"],
            "commit_limit_kib": values["CommitLimit"], "committed_as_kib": values["Committed_AS"],
            "commit_headroom_kib": values["CommitLimit"] - values["Committed_AS"]}


def cgroup_paths():
    memory_rel = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, rel = line.split(":", 2)
        if "memory" in controllers.split(","): memory_rel = rel; break
    require(memory_rel is not None, "cgroup v1 memory controller absent")
    session = Path("/sys/fs/cgroup/memory") / memory_rel.lstrip("/")
    user = Path("/sys/fs/cgroup/memory/user.slice") / f"user-{os.getuid()}.slice"
    return session, user


def cgroup_read(path):
    failcnt_path = path / "memory.failcnt"; oom_path = path / "memory.oom_control"
    require(regular(failcnt_path) and regular(oom_path), f"missing cgroup counters: {path}")
    fields = {}
    for line in oom_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit(): fields[parts[0]] = int(parts[1])
    require("under_oom" in fields and "oom_kill" in fields, f"missing oom fields: {path}")
    return {"path": str(path), "failcnt": int(failcnt_path.read_text().strip()),
            "under_oom": fields["under_oom"], "oom_kill": fields["oom_kill"]}


def main():
    for path, expected in EXPECTED.items():
        require(regular(path), f"missing fixed identity: {path}")
        require(sha(path) == expected, f"fixed identity drift: {path}")

    request = verify_package(REQUEST_DIR, "request.json")
    release = verify_json_sidecars(RELEASE)
    source = verify_json_sidecars(SOURCE)
    candidate = verify_json_sidecars(CANDIDATE)
    source_review = verify_package(SOURCE_REVIEW_DIR, "review.json")
    candidate_review = verify_package(CANDIDATE_REVIEW_DIR, "review.json")
    m782 = verify_package(M782_DIR, "review.json")
    r14_release = verify_json_sidecars(R14_RELEASE)
    m779 = verify_package(M779_DIR, "review.json")
    preflight = verify_package(PREFLIGHT_DIR, "preflight.json")
    r13 = verify_package(R13_DIR, "RUN_FAILED_OR_INCOMPLETE.json")
    m770 = verify_package(M770_DIR, "review.json")

    require(request["status"] == "REQUEST_FRESH_INDEPENDENT_M784_R15_FINAL_RELEASE_HAMMER__NO_EXECUTION_AUTHORIZED_BY_REQUEST", "request status")
    require(request["request_authorization"] == {"run_runner": False, "run_vcs": False, "run_simv": False,
        "run_vcs_identity_probe": False, "run_license_query": False, "run_iverilog": False,
        "run_verilator": False, "run_dc": False, "run_formality": False, "run_pt": False,
        "run_ptpx": False, "run_cpu_or_gpu_experiment": False, "run_remote": False,
        "execute_future_command_now": False, "max_execution_attempts_authorized_by_request": 0}, "request closed authorization")
    require(release["schema"] == "m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_v1", "release schema")
    require(release["status"] == "AUTHORIZED_EXACTLY_ONE_M784_R15_UNIT_DELAY_FUNCTIONAL_VCS_ATTEMPT__FRESH_FINAL_HAMMER_REQUIRED__R14_PERMANENTLY_WITHDRAWN", "release status")
    require(release["launch_now"] is True and release["authorization"] == AUTH, "release authorization")
    require(release["environment_policy"] == ENV_POLICY and release["resource_policy"] == RESOURCE_POLICY, "release environment/resource policy")
    require(release["macro_model_mode"] == "foundry_UNIT_DELAY_functional", "release macro mode")
    ident = release["identity"]
    require(ident["runner_sha256"] == EXPECTED[RUNNER] and ident["source_contract_sha256"] == EXPECTED[SOURCE], "release source binding")
    require(ident["candidate_sha256"] == EXPECTED[CANDIDATE] and ident["source_static_review_sha256"] == EXPECTED[SOURCE_REVIEW], "release review binding")
    require(ident["candidate_hammer_review_sha256"] == EXPECTED[CANDIDATE_REVIEW] and ident["m782_failure_audit_sha256"] == EXPECTED[M782], "release hammer/M782 binding")
    require(ident["withdrawn_r14_release_sha256"] == EXPECTED[R14_RELEASE] and ident["superseded_m779_review_sha256"] == EXPECTED[M779], "release withdrawn chain binding")
    require(release["unique_attempt"]["only_r15_identity_released"] is True, "only r15 release")
    require(release["unique_attempt"]["r14_release_status"] == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "r14 release status in r15")
    require(release["author_execution_receipt"]["runner_executions"] == 0 and release["author_execution_receipt"]["vcs_runs"] == 0, "release author executions")
    require(source_review["verdict"] == "PASS" and source_review["score_100"] == 100 and
            [source_review[k] for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "source review admission")
    require(candidate_review["verdict"] == "PASS" and candidate_review["score_100"] == 100 and
            [candidate_review[k] for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "candidate review admission")
    require(candidate_review["decision"]["one_true_launch_release_authoring_authorized"] is True and
            candidate_review["decision"]["vcs_launch_authorized_now"] is False, "candidate release boundary")
    require(m782["verdict"] == "PASS_FAILURE_AUDIT" and m782["score_out_of_100"] == 100, "M782 audit")
    require(m782["r14_release_disposition"]["release_status_after_audit"] == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "M782 r14 disposition")
    require(m782["decision"]["m779_status"] == "SUPERSEDED_P1_BLOCKING_GAP_DO_NOT_USE_FOR_LAUNCH", "M779 disposition")
    require(m782["decision"]["r15_launch_authorized_now"] is False and m782["decision"]["one_additive_r15_source_package_authorized"] is True, "M782 additive boundary")
    require(r14_release["identity"]["runner_sha256"] == sha(R14_RUNNER), "r14 release identity")
    require(m779["decision"]["exactly_one_vcs_attempt_authorized_now"] is True, "historical M779 identity")
    require(r13["status"] == "FAILED_DO_NOT_CITE" and r13["paper_citable"] is False, "r13 immutable failure")
    require(m770["verdict"] == "PASS" and m770["decision"]["r14_launch_authorized_now"] is False, "M770 boundary")
    require(preflight["boundary"]["result_directory_created"] is False and preflight["boundary"]["r14_attempt_consumed"] is False, "historical preflight boundary")

    ledger = resolve_edges(RUNNER)
    require(len(ledger) == 67, f"r15 edge count={len(ledger)}")
    r14_raw, r15_raw = raw_edges(R14_RUNNER), raw_edges(RUNNER)
    require(len(r14_raw) == 64 and len(r15_raw) == 67, "r14/r15 edge cardinality")
    extras = [
        ("ff7498279990537c7e60f886d44a3a6ec919aeb39d2fe5a9294a049f9a79bf6b", '"${M782_REVIEW}"'),
        ("3cf622455ea68a5df7fe511ebc7897c2e78f68488d4696ffc23b4ade685d448b", '"${M782_DIR}/SHA256SUMS"'),
        ("e6dbb6250e913a56b58741374f1b8ac1ce5b20e0653713635c47f91fd2d5d740", '"${M782_DIR}/SHA256SUMS.seal.sha256"'),
    ]
    reduced = list(r15_raw)
    for extra in extras:
        require(reduced.count(extra) == 1, f"missing/duplicate appended M782 edge: {extra}")
        reduced.remove(extra)
    require(reduced == r14_raw, "original r14 64-edge sequence drift")

    heredoc = execute_real_heredoc()
    runner_text = RUNNER.read_text(encoding="utf-8")
    require(len(re.findall(r"^\s*\+define\+UNIT_DELAY(?:\s|$)", runner_text, re.MULTILINE)) == 1, "UNIT_DELAY compile occurrence")
    require("+notimingcheck" not in runner_text and "+no_notifier" not in runner_text, "timing bypass present")
    for token in ("pre_mkdir_collision_initial", "pre_mkdir_collision_final", "collision_postmkdir",
                  "RESOURCE_HEARTBEAT", "RESOURCE_FINAL_ACK",
                  "COVERAGE_M533_M528_DW1RW_R7", "attacks=6", "normal_covers=13",
                  "RUN_COMPLETE.json", "RUN_FAILED_OR_INCOMPLETE.json", "SHA256SUMS.seal.sha256"):
        require(token in runner_text, f"runner gate missing: {token}")
    for attack in ("dirty_reserved", "stale_epoch", "overflow", "wrong_parent",
                   "read_before_write", "parent_only_nonzero"):
        require(attack in runner_text, f"protocol attack gate missing: {attack}")
    tb_text = (HW / "tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r7.sv").read_text(encoding="utf-8")
    require("watchdog > 20000" in tb_text and '$fatal(1, "task timeout epoch=%0d", epoch)' in tb_text,
            "task watchdog drift")
    require('$fatal(1, "global watchdog expired")' in tb_text, "global watchdog drift")

    require(not RESULT.exists(), f"r15 result exists: {RESULT}")
    require(not list(RESULT.parent.glob(RESULT.name + "*")), "r15 result sibling/collision exists")
    require(not R14_RESULT.exists() and not list(R14_RESULT.parent.glob(R14_RESULT.name + "*")), "r14 result unexpectedly exists")
    tmp_count = len(list(Path("/tmp").glob("m784_m533_r15_unit_delay_vcs_preflight.*")))
    require(tmp_count == 0, f"stale r15 preflight dirs={tmp_count}")

    uid, initial_collision = proc_collisions()
    require(not initial_collision, f"initial same-UID EDA collision: {initial_collision}")
    session_path, user_path = cgroup_paths()
    cg_initial = {"session": cgroup_read(session_path), "user": cgroup_read(user_path)}
    for item in cg_initial.values():
        require(item["under_oom"] == 0 and item["oom_kill"] == 0, f"initial cgroup OOM: {item}")
    samples = []
    for index in range(3):
        sample = mem_sample(); sample["sample"] = index + 1; samples.append(sample)
        require(sample["mem_available_kib"] >= RESOURCE_POLICY["mem_available_min_kib"], f"MemAvailable sample {index+1}")
        require(sample["swap_free_kib"] >= RESOURCE_POLICY["swap_free_min_kib"], f"SwapFree sample {index+1}")
        require(sample["commit_headroom_kib"] >= RESOURCE_POLICY["commit_headroom_min_kib"], f"commit headroom sample {index+1}")
        if index != 2: time.sleep(2)
    cg_final = {"session": cgroup_read(session_path), "user": cgroup_read(user_path)}
    for key in ("session", "user"):
        require(cg_final[key]["failcnt"] == cg_initial[key]["failcnt"], f"cgroup failcnt changed: {key}")
        require(cg_final[key]["under_oom"] == 0 and cg_final[key]["oom_kill"] == 0, f"final cgroup OOM: {key}")
    _, final_collision = proc_collisions()
    require(not final_collision, f"final same-UID EDA collision: {final_collision}")
    require(not RESULT.exists() and not list(RESULT.parent.glob(RESULT.name + "*")), "r15 result appeared during review")
    require(sha(DOC359) == EXPECTED[DOC359], "docs359 drift after review")

    report = {
        "verdict": "PASS", "score_100": 100, "p0": 0, "p1": 0, "p2": 0,
        "identity": {"request_sha256": sha(REQUEST), "release_sha256": sha(RELEASE),
                     "runner_sha256": sha(RUNNER), "source_sha256": sha(SOURCE),
                     "candidate_sha256": sha(CANDIDATE), "source_review_sha256": sha(SOURCE_REVIEW),
                     "candidate_review_sha256": sha(CANDIDATE_REVIEW), "m782_sha256": sha(M782),
                     "docs359_sha256": sha(DOC359)},
        "sha_edges": {"r14": len(r14_raw), "r15": len(ledger), "appended_m782": 3,
                      "live_mismatch": 0, "r14_sequence_after_removing_m782": "BYTE_IDENTICAL"},
        "real_heredoc": heredoc,
        "absence": {"r15_result_and_siblings": True, "r14_result_and_siblings": True,
                    "r15_preflight_tmp_count": tmp_count},
        "live": {"uid": uid, "collision_initial": initial_collision, "collision_final": final_collision,
                 "samples": samples, "cgroup_initial": cg_initial, "cgroup_final": cg_final},
        "execution_receipt": {"runner": 0, "vcs": 0, "simv": 0, "vcs_identity": 0,
                              "license_queries": 0, "hdl_eda": 0, "result_dirs_created": 0},
        "exact_command": "env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 VCS_ARCH_OVERRIDE=linux SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m784_m533_m528_dead_write_only_1rw_unit_delay_r15_exact_sha.sh",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

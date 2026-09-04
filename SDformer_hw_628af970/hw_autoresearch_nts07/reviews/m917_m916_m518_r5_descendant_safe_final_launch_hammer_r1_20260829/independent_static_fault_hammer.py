#!/usr/bin/env python3
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_dc_m917_m518_r5_fixed_descendant_safe_setup_area_exact_sha.sh"
CONTRACT = HW / "contracts/m916_m518_r5_fixed_descendant_safe_setup_area_dc_contract_r1_20260829.json"
ADMISSION = HW / "contracts/m917_m518_r5_fixed_descendant_safe_setup_area_dc_launch_admission_r1_20260829.json"
M915 = HW / "reviews/m915_m518_r4_fixed_descendant_collision_quarantine_forensic_r1_20260829"
R4Q = HW / "dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828.failed_or_incomplete.2923446.quarantine"
R4A = HW / "dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed"
R5C = HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
R5A = HW / "dc_handoff/runs/.m917_m518_r5_fixed_descendant_safe_setup_area_attempt_consumed"

EXPECTED_RUNNER = "284f4066a1a719066e55bb4b71826e48cf6c1352b636ca30efdc5dfecc9350e8"
EXPECTED_CONTRACT = "144eee7090f769bf7670a77f3124a53f13f48d873bb48aa4ff536c961b61c86d"
EXPECTED_ADMISSION = "a5a70d9ad84983241b1a930e75ed7b654f6515c8e6bee62a5fcd47cec4254e8d"
EXPECTED_M915_OUTER = "49eca961a7ecddff84ab426ef757123dba7c840c8c283e1ef6223ace46ae3ef1"
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate key " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def double_seal(payload):
    digest_line = payload.with_name(payload.name + ".sha256").read_text().split()
    seal_path = payload.with_name(payload.name + ".sha256.seal.sha256")
    seal_line = seal_path.read_text().split()
    return digest_line[0] == sha(payload) and seal_line[0] == sha(payload.with_name(payload.name + ".sha256"))

def sealed_dir(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not manifest.is_file() or not outer.is_file() or outer.read_text().split()[0] != sha(manifest):
        return False
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if sha(root / rel[2:]) != digest:
            return False
    return True

def proc_tuple(pid):
    raw = Path("/proc") / str(pid) / "stat"
    text = raw.read_text()
    fields = text[text.rfind(") ") + 2:].split()
    return {
        "state": fields[0], "ppid": int(fields[1]), "pgrp": int(fields[2]),
        "session": int(fields[3]), "start": int(fields[19])
    }

def group_live(pgrp, session):
    members = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            p = proc_tuple(int(item.name))
        except (OSError, ValueError, IndexError):
            continue
        if p["pgrp"] == pgrp and p["session"] == session and p["state"] != "Z":
            members.append(int(item.name))
    return members

def predicate_result(functions, root, candidate, wrong_start=False):
    script = functions + r'''
root=$1
candidate=$2
m917_proc_identity "${root}" || exit 70
rs=${M917_P_START}; ru=${M917_P_UID}; re=${M917_P_EXE}; rp=${M917_P_PPID}
rg=${M917_P_PGRP}; rsession=${M917_P_SESSION}; rc=${M917_P_CMDHEX}
if [[ "${3}" == wrong ]]; then rs=$((rs + 1)); fi
if m917_is_exact_root_descendant "${candidate}" "${root}" "${rs}" "${ru}" "${re}" "${rp}" "${rg}" "${rsession}" "${rc}"; then
  printf 'DESCENDANT\n'
else
  printf 'EXTERNAL\n'
fi
'''
    with tempfile.TemporaryDirectory(prefix="m917_static_") as td:
        path = Path(td) / "predicate.sh"
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\nm917_uid=$(id -u)\n" + script)
        cp = subprocess.run(["/usr/bin/bash", str(path), str(root), str(candidate), "wrong" if wrong_start else "exact"],
                            universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=10)
        observation = cp.stdout.strip()
        if cp.returncode == 70 and not observation:
            observation = "EXTERNAL_ROOT_ABSENT"
        return cp.returncode, observation, cp.stderr.strip()

def static_invariants(text):
    required = [
        "m917_is_exact_root_descendant()",
        'if [[ -n "${root}" ]] && m917_is_exact_root_descendant',
        '"${m917_setsid}" "${m917_dc}" -f',
        '[[ "${M917_P_PGRP}" == "${pid}" && "${M917_P_SESSION}" == "${pid}" ]]',
        'm917_wait_job_empty "${pgrp}" "${session}" "${uid}" "${start}" 300',
        'm917_terminate_job "${m917_child_pgrp}" "${m917_child_session}"',
        'mkdir "${OUTPUT_DIR}" "${m917_work}/safe_home"',
        'chmod 700 "${m917_work}/safe_home"',
        'export HOME="${m917_work}/safe_home"',
        "job_tree_drained_before_seal=true",
        "job_tree_empty_before_ack=true",
        "compile_ultra_count=1",
        "hold_fix_command_count=0",
    ]
    forbidden = ["pt_shell -f", "fm_shell -f", "compile_ultra -incremental", "set_fix_hold"]
    return all(x in text for x in required) and not any(x in text for x in forbidden)

checks = {}
runner_text = RUNNER.read_text()
contract = strict_json(CONTRACT)
admission = strict_json(ADMISSION)
checks["runner_sha_exact"] = sha(RUNNER) == EXPECTED_RUNNER
checks["contract_sha_exact"] = sha(CONTRACT) == EXPECTED_CONTRACT
checks["admission_sha_exact"] = sha(ADMISSION) == EXPECTED_ADMISSION
checks["contract_double_seal"] = double_seal(CONTRACT)
checks["admission_double_seal"] = double_seal(ADMISSION)
checks["m915_recursive_seal"] = sealed_dir(M915) and sha(M915 / "SHA256SUMS.seal.sha256") == EXPECTED_M915_OUTER
checks["r4_attempt_recursive_seal"] = sealed_dir(R4A)
checks["r4_quarantine_do_not_cite"] = (R4Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().startswith("status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n")
checks["r5_identity_absent"] = not R5C.exists() and not R5A.exists() and not any((HW / "dc_handoff/runs").glob(".m917_m518_r5_fixed_descendant_safe_setup_area_work.*"))
checks["contract_runner_cross_edge"] = contract["identity"]["runner_sha256"] == EXPECTED_RUNNER
checks["admission_cross_edges"] = admission["identity"]["runner_sha256"] == EXPECTED_RUNNER and admission["identity"]["contract_sha256"] == EXPECTED_CONTRACT
checks["fixed_only_authorization"] = admission["authorization"]["run_dc"] and not any(admission["authorization"][k] for k in ["run_vcs", "run_formality", "run_pt", "run_ptpx", "run_remote", "run_rank3", "run_paired_comparison"])
checks["bash_syntax"] = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE).returncode == 0
checks["static_invariants"] = static_invariants(runner_text)
checks["fault_remove_descendant_call_detected"] = not static_invariants(runner_text.replace('if [[ -n "${root}" ]] && m917_is_exact_root_descendant', 'if false && m917_is_exact_root_descendant', 1))
checks["fault_remove_safe_home_detected"] = not static_invariants(runner_text.replace('export HOME="${m917_work}/safe_home"', 'true # HOME removed', 1))
checks["docs359_frozen"] = sha(HW / "docs/359_DATE终局冻结_20260813.md") == DOC_SHA

start = runner_text.index("m917_proc_identity() {")
end = runner_text.index('mkdir "${m917_preflight}"')
functions = runner_text[start:end]
root = None
dynamic = {}
try:
    root = subprocess.Popen(
        ["/usr/bin/setsid", "/usr/bin/bash", "-c",
         'sleep 60 & c=$!; /usr/bin/bash -c \'sleep 60 & echo $!; wait\' & b=$!; read gc < <(cat <&3); echo "$c $b $gc"; wait',
         "m917-tree"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, pass_fds=())
    # The compact shell above cannot pass fd3 portably; fall back to /proc tree discovery.
    time.sleep(0.3)
    root_pid = root.pid
    direct = []
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            try:
                if proc_tuple(int(item.name))["ppid"] == root_pid:
                    direct.append(int(item.name))
            except (OSError, ValueError, IndexError):
                pass
    # If the first construction exited because fd3 was absent, replace it with a simple, stable tree.
    if root.poll() is not None or len(direct) < 1:
        try:
            os.killpg(root_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        root = subprocess.Popen(
            ["/usr/bin/setsid", "/usr/bin/bash", "-c",
             'sleep 60 & /usr/bin/bash -c \'sleep 60 & wait\' & wait', "m917-tree"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        root_pid = root.pid
        direct = []
        for item in Path("/proc").iterdir():
            if item.name.isdigit():
                try:
                    if proc_tuple(int(item.name))["ppid"] == root_pid:
                        direct.append(int(item.name))
                except (OSError, ValueError, IndexError):
                    pass
    grandchildren = []
    for d in direct:
        for item in Path("/proc").iterdir():
            if item.name.isdigit():
                try:
                    if proc_tuple(int(item.name))["ppid"] == d:
                        grandchildren.append(int(item.name))
                except (OSError, ValueError, IndexError):
                    pass
    direct_pid = direct[0]
    grand_pid = grandchildren[0]
    dynamic["direct"] = predicate_result(functions, root_pid, direct_pid)[1]
    dynamic["grandchild"] = predicate_result(functions, root_pid, grand_pid)[1]
    dynamic["external_sibling"] = predicate_result(functions, root_pid, os.getpid())[1]
    dynamic["wrong_start"] = predicate_result(functions, root_pid, direct_pid, wrong_start=True)[1]
    checks["live_direct_child_excluded"] = dynamic["direct"] == "DESCENDANT"
    checks["live_grandchild_excluded"] = dynamic["grandchild"] == "DESCENDANT"
    checks["live_external_process_not_excluded"] = dynamic["external_sibling"] == "EXTERNAL"
    checks["root_starttime_mismatch_not_excluded"] = dynamic["wrong_start"] == "EXTERNAL"
    os.kill(root_pid, signal.SIGKILL)
    root.wait(timeout=5)
    time.sleep(0.2)
    dynamic["orphan"] = predicate_result(functions, root_pid, direct_pid)[1]
    checks["reparented_orphan_not_excluded"] = dynamic["orphan"] in {"EXTERNAL", "EXTERNAL_ROOT_ABSENT"}
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(50):
        if not group_live(root_pid, root_pid):
            break
        time.sleep(0.1)
    checks["isolated_job_group_drained"] = not group_live(root_pid, root_pid)
finally:
    if root is not None:
        try:
            os.killpg(root.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            root.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

before = sorted(p.name for p in (HW / "dc_handoff/runs").glob("*m917_m518_r5*"))
clean_env = {
    "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo", "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
    "M917_EXPECTED_DC_RUNNER_SHA256": "0" * 64,
    "M917_EXPECTED_ADMISSION_SHA256": EXPECTED_ADMISSION,
}
wrong = subprocess.run(["/usr/bin/bash", str(RUNNER)], cwd=str(HW), env=clean_env,
                       universal_newlines=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=10)
checks["wrong_sha_rejects_preeda"] = wrong.returncode == 3 and "caller must pin runner SHA" in wrong.stderr
home_env = dict(clean_env); home_env["HOME"] = "/tmp/m917_forbidden_home"
home = subprocess.run(["/usr/bin/bash", str(RUNNER)], cwd=str(HW), env=home_env,
                      universal_newlines=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, timeout=10)
checks["incoming_home_rejects_preeda"] = home.returncode == 3 and "incoming HOME absent" in home.stderr
after = sorted(p.name for p in (HW / "dc_handoff/runs").glob("*m917_m518_r5*"))
checks["negative_dry_runs_zero_formal_side_effect"] = before == after == [] and not R5A.exists()

eda = []
for item in Path("/proc").iterdir():
    if not item.name.isdigit():
        continue
    try:
        uid = int((item / "status").read_text().split("Uid:", 1)[1].split()[0])
        comm = (item / "comm").read_text().strip()
        exe = os.path.basename(os.path.realpath(item / "exe"))
    except (OSError, ValueError, IndexError):
        continue
    if uid == os.getuid() and (comm in {"dc_shell", "dc_shell-t", "fm_shell", "pt_shell", "vcs", "vcs1", "vlogan", "simv"} or exe == "common_shell_exec"):
        eda.append(int(item.name))
checks["live_same_uid_eda_collision_none"] = not eda

mem = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    mem[key] = int(value.split()[0])
headroom = mem["CommitLimit"] - mem["Committed_AS"]
checks["live_resource_gate_healthy"] = headroom >= 67108864 and mem["MemAvailable"] >= 134217728 and mem["SwapFree"] >= 33554432
checks["live_cgroup_gate_healthy"] = (Path("/sys/fs/cgroup/memory/user.slice/memory.failcnt").read_text().strip() == "0" and
    "under_oom 0" in Path("/sys/fs/cgroup/memory/user.slice/memory.oom_control").read_text() and
    "oom_kill 0" in Path("/sys/fs/cgroup/memory/user.slice/memory.oom_control").read_text())

failed = sorted(k for k, value in checks.items() if not value)
result = {
    "status": "PASS_M917_M916_M518_R5_DESCENDANT_SAFE_FINAL_LAUNCH_HAMMER" if not failed else "FAIL_M917_HAMMER",
    "score": 100 if not failed else max(0, 100 - 10 * len(failed)),
    "p0_p1_p2": [0, 0, 0] if not failed else [1, 0, 0],
    "checks": checks,
    "dynamic_fault_observations": dynamic,
    "failed": failed,
    "same_uid_eda_pids": eda,
    "commit_headroom_kib": headroom,
    "mem_available_kib": mem["MemAvailable"],
    "swap_free_kib": mem["SwapFree"],
    "eda_or_license_tools_executed": 0,
    "formal_attempt_consumed": False,
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)

#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
HW = SELF.parents[2]
REPO = HW.parent
DC = HW / "dc_handoff"
RUNS = DC / "runs"
WRAPPER = HW / "dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.sh"
HELPER = HW / "dc_handoff/scripts/check_m1311_python_symlink_entity.sh"
CONTRACT = HW / "contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_source_contract_r1_20260831.json"
M1302_WRAPPER = HW / "dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh"
M1302_ADMISSION = HW / "contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json"
M1302_CONTRACT = HW / "contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json"
M1288_RUNNER = HW / "dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh"
M1288_CONTRACT = HW / "contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json"
M1302_AUTHOR = HW / "reviews/m1302_m1288_c3_fixed_t10_ptsta_launch_author_receipt_r1_20260830"
M1308 = HW / "reviews/m1308_m1302_c3_fixed_t10_pt_launch_receipt_blind_hammer_r1_20260831"
M1310 = HW / "reviews/m1310_m1302_python_symlink_zero_attempt_forensic_r1_20260831"
M917 = HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
M928 = HW / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829"
M1285 = HW / "reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830"
M1288_CANONICAL = RUNS / "m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830"
M1288_WORK = Path(str(M1288_CANONICAL) + ".work")
M1288_ATTEMPT = RUNS / ".m1288_m917_fixed_t10_ptsta_attempt_consumed"
M1302_CANONICAL = RUNS / "m1302_m1288_fixed_t10_ptsta_adjudication_r1_20260830"
M1302_WORK = Path(str(M1302_CANONICAL) + ".work")
M1302_ATTEMPT = RUNS / ".m1302_m1288_fixed_t10_ptsta_attempt_consumed"
CANONICAL = RUNS / "m1311_m1288_fixed_t10_ptsta_adjudication_r1_20260831"
WORK = Path(str(CANONICAL) + ".work")
ATTEMPT = RUNS / ".m1311_m1288_fixed_t10_ptsta_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")


def sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    st = os.lstat(str(path))
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError("not regular: " + str(path))


def exact(obj, keys, name):
    if type(obj) is not dict or set(obj) != set(keys):
        raise RuntimeError(name + " keyset")


def bool_exact(obj, expected, name):
    exact(obj, expected.keys(), name)
    for key, value in expected.items():
        if type(obj[key]) is not bool or obj[key] is not value:
            raise RuntimeError(name + " boolean " + key)


def verify_payload(path):
    regular(path)
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(side); regular(outer)
    if side.read_text().split() != [sha(path), path.name]:
        raise RuntimeError("payload sidecar")
    if outer.read_text().split() != [sha(side), side.name]:
        raise RuntimeError("payload outer seal")


def verify_dir(directory):
    st = os.lstat(str(directory))
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError("sealed directory identity " + str(directory))
    regular(directory / "SHA256SUMS")
    regular(directory / "SHA256SUMS.seal.sha256")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise RuntimeError("outer seal " + str(directory))
    seen = set()
    for line in manifest.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError("manifest syntax")
        digest, rel = parts[0], parts[1].lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in seen or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise RuntimeError("manifest path")
        seen.add(rel)
        target = directory / rel
        regular(target)
        if sha(target) != digest:
            raise RuntimeError("manifest drift " + str(target))


def seal_dir(directory):
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        if path.is_symlink() or not path.is_file():
            if path.is_dir():
                continue
            raise RuntimeError("nonregular result member")
        rows.append("%s  %s" % (sha(path), path.relative_to(directory)))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("\n".join(rows) + "\n")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "%s  SHA256SUMS\n" % sha(manifest))
    verify_dir(directory)


def path_under_repo(value):
    if not value or not os.path.isabs(value):
        return False
    try:
        return os.path.commonpath([os.path.normpath(value), str(REPO)]) == str(REPO)
    except ValueError:
        return False


def process_is_repo_scoped(cwd, argv):
    if path_under_repo(cwd):
        return True
    return any(path_under_repo(arg) for arg in argv)


def collisions():
    uid = os.getuid()
    eda = {"pt_shell", "dc_shell", "dc_shell-t", "fm_shell", "vcs", "vcs1",
           "vlogan", "simv", "common_shell_exec", "common_shell_exe"}
    blocking, external = [], []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            pid = int(name)
            status = Path("/proc/%d/status" % pid).read_text().splitlines()
            puid = int(next(x for x in status if x.startswith("Uid:")).split()[1])
            raw = Path("/proc/%d/stat" % pid).read_text()
            rest = raw[raw.rfind(")") + 2:].split()
            state, start = rest[0], int(rest[19])
            comm = Path("/proc/%d/comm" % pid).read_text().strip()
            exe = os.path.basename(os.path.realpath("/proc/%d/exe" % pid))
            cwd = os.path.realpath("/proc/%d/cwd" % pid)
            raw_argv = Path("/proc/%d/cmdline" % pid).read_bytes().split(b"\0")
            argv = [item.decode("utf-8", "surrogateescape")
                    for item in raw_argv if item]
        except (OSError, StopIteration, ValueError):
            continue
        if puid == uid and state != "Z" and (comm in eda or exe in eda):
            item = {"pid": pid, "starttime": start, "comm": comm,
                    "exe": exe, "cwd": cwd}
            if process_is_repo_scoped(cwd, argv):
                blocking.append(item)
            else:
                external.append(item)
    return {"blocking": sorted(blocking, key=lambda x: (x["pid"], x["starttime"])),
            "external_record_only": sorted(external, key=lambda x: (x["pid"], x["starttime"]))}


def meminfo():
    data = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        match = re.match(r"^(MemAvailable|CommitLimit|Committed_AS):\s+(\d+)\s+kB$", line)
        if match:
            data[match.group(1)] = int(match.group(2))
    if set(data) != {"MemAvailable", "CommitLimit", "Committed_AS"}:
        raise RuntimeError("meminfo")
    return data


def validate_admission(data, args):
    exact(data, ("schema", "date", "milestone", "status", "objective", "identity",
          "exact_files", "tool", "python_entity", "preflight", "authorization",
          "result_adjudication", "claim_boundary"), "admission")
    if data["schema"] != "m1311_m1302_c3_fixed_t10_ptsta_python_entity_launch_admission_v1":
        raise RuntimeError("schema")
    if data["status"] != "AUTHORIZED_ONE_M1311_M1288_FIXED_T10_PTSTA_ATTEMPT_AFTER_HAMMER":
        raise RuntimeError("status")
    identity_keys = ("admission_path", "wrapper_path", "wrapper_sha256",
        "orchestrator_path", "orchestrator_sha256", "helper_path", "helper_sha256",
        "contract_path", "contract_sha256", "m1302_wrapper_sha256",
        "m1302_admission_sha256", "m1302_author_outer_seal_sha256",
        "m1308_outer_seal_sha256", "m1310_outer_seal_sha256", "m1288_runner_sha256",
        "m1288_result", "m1288_attempt", "m1311_result", "m1311_attempt")
    exact(data["identity"], identity_keys, "identity")
    if data["identity"]["wrapper_sha256"] != args.expected_wrapper_sha:
        raise RuntimeError("wrapper SHA")
    if data["identity"]["orchestrator_sha256"] != args.expected_orchestrator_sha:
        raise RuntimeError("orchestrator SHA")
    auth = {"launch_now": False, "launch_after_independent_hammer": True,
        "run_pt": True, "run_dc": False, "run_vcs": False,
        "run_formality": False, "run_ptpx": False, "run_remote": False,
        "query_license": True, "max_attempts_is_one": True,
        "strict_result_adjudication": True}
    bool_exact(data["authorization"], auth, "authorization")
    claims = {"launch_admission_only": True, "pt_executed": False,
        "setup_completed": False, "hold_closed": False, "coverage_closed": False,
        "unconstrained_paths_zero": False, "automatic_hold_fix": False,
        "power": False, "energy": False, "speedup": False, "system": False,
        "paper_ppa_ready": False, "headline": False}
    bool_exact(data["claim_boundary"], claims, "claims")
    expected_py = {"logical_path": "/usr/bin/python3",
        "link_targets": ["/etc/alternatives/python3", "/usr/bin/python3.6",
                         "/usr/libexec/platform-python3.6"],
        "resolved_path": "/usr/libexec/platform-python3.6", "device": 66313,
        "inode": 6442661434, "mode_octal": "0755", "size_bytes": 11872,
        "sha256": "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f",
        "fd_recheck_required": True}
    if data["python_entity"] != expected_py:
        raise RuntimeError("python entity")
    exact(data["preflight"], ("same_uid_collision_gate", "resource_gate",
          "license_gate", "fresh_namespaces", "order"), "preflight")
    if data["preflight"]["same_uid_collision_gate"] != {
            "blocking_scope": "same-UID EDA with cwd or an absolute argv path under exact repository root",
            "external_worktrees": "record only; never terminate and never block"}:
        raise RuntimeError("collision gate")
    if data["preflight"]["resource_gate"] != {"mem_available_min_kib": 8388608,
            "commit_headroom_min_kib": 8388608,
            "filesystem_available_min_kib": 4194304}:
        raise RuntimeError("resource gate")
    if data["preflight"]["license_gate"] != {"feature": "PrimeTime",
            "server": "27030@ic.ismd-nemo", "issued_gt_in_use_required": True,
            "query_before_attempt": True}:
        raise RuntimeError("license gate")
    expected_result = {"setup_state": "MET", "setup_slack_min_ns": 0.0,
        "hold_state": "MET", "hold_slack_min_ns": 0.0,
        "constraint_violated_paths": 0, "unconstrained_paths": 0,
        "required_coverage_rows": ["setup", "hold", "out_setup", "out_hold"],
        "coverage_rule": "each total>0 and met==total and violated==0 and untested==0",
        "fresh_result_hammer_required": True}
    if data["result_adjudication"] != expected_result:
        raise RuntimeError("result gate")


def validate_inputs(data):
    expected = set(data["exact_files"])
    if len(expected) != 26:
        raise RuntimeError("exact file cardinality")
    for relative, digest in data["exact_files"].items():
        if re.match(r"^[0-9a-f]{64}$", digest or "") is None:
            raise RuntimeError("digest syntax")
        path = HW / relative
        regular(path)
        if sha(path) != digest:
            raise RuntimeError("input drift " + relative)
    verify_payload(HW / data["identity"]["admission_path"])
    verify_payload(CONTRACT)
    verify_payload(M1302_ADMISSION)
    verify_payload(M1302_CONTRACT)
    verify_payload(M1288_CONTRACT)
    for directory in (M1302_AUTHOR, M1308, M1310, M917, M928, M1285):
        verify_dir(directory)
    exact(data["tool"], ("pt_shell_path", "pt_shell_sha256", "lmutil_path",
          "lmutil_sha256", "bash_path", "bash_sha256", "setsid_path",
          "setsid_sha256", "slow_db_path", "slow_db_sha256", "fast_db_path",
          "fast_db_sha256"), "tool")
    for key in ("pt_shell", "lmutil", "bash", "setsid", "slow_db", "fast_db"):
        path = Path(data["tool"][key + "_path"])
        regular(path)
        if sha(path) != data["tool"][key + "_sha256"]:
            raise RuntimeError("tool drift " + key)


def result_receipt():
    reports = M1288_CANONICAL / "reports"
    def text(name):
        return (reports / name).read_text(encoding="utf-8", errors="replace")
    def slack(name):
        match = re.search(r"slack \((MET|VIOLATED)\)\s+(-?\d+(?:\.\d+)?)", text(name))
        if not match:
            raise RuntimeError("missing slack " + name)
        return match.group(1), float(match.group(2))
    def row(name, coverage):
        match = re.search(r"^" + re.escape(name) +
            r"\s+(\d+)\s+(\d+) \([^\n]+?\)\s+(\d+) \([^\n]+?\)\s+(\d+) \(",
            coverage, re.M)
        if not match:
            raise RuntimeError("missing coverage " + name)
        return tuple(map(int, match.groups()))
    setup_state, setup = slack("timing_setup_slow.rpt")
    hold_state, hold = slack("timing_hold_fast.rpt")
    cov_text = text("analysis_coverage.rpt")
    coverage = {name: row(name, cov_text)
                for name in ("setup", "hold", "out_setup", "out_hold")}
    coverage_gate = all(total > 0 and met == total and violated == 0 and untested == 0
                        for total, met, violated, untested in coverage.values())
    check = text("check_timing.rpt")
    counts = []
    for pattern in (r"There (?:is|are)\s+(\d+)\s+input ports?.{0,240}?will be unconstrained",
                    r"There (?:is|are)\s+(\d+)\s+endpoints?.{0,240}?unconstrained"):
        counts.extend(int(x) for x in re.findall(pattern, check, re.I | re.S))
    unconstrained = sum(counts)
    violations = text("constraint_violators.rpt").count("slack (VIOLATED)")
    gate = (setup_state == "MET" and setup >= 0.0 and hold_state == "MET" and hold >= 0.0
            and violations == 0 and unconstrained == 0 and coverage_gate)
    status_value = ("PASS_M1311_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE"
                    if gate else
                    "STOP_M1311_M1288_FIXED_T10_PRELAYOUT_PTSTA_STRICT_TIMING_GATE")
    return {
        "schema": "m1311_m1288_fixed_t10_ptsta_adjudication_receipt_v1",
        "status": status_value,
        "setup": {"state": setup_state, "worst_slack_ns": setup,
                  "closed": setup_state == "MET" and setup >= 0.0},
        "hold": {"state": hold_state, "worst_slack_ns": hold,
                 "closed": hold_state == "MET" and hold >= 0.0,
                 "automatic_fix_performed": False},
        "constraint_violated_paths": violations,
        "unconstrained_paths": unconstrained,
        "analysis_coverage": {name: {"total": item[0], "met": item[1],
            "violated": item[2], "untested": item[3]}
            for name, item in coverage.items()},
        "coverage_gate_pass": coverage_gate,
        "strict_timing_gate_pass": gate,
        "python_entity_repair": {"logical_path": "/usr/bin/python3",
            "resolved_path": "/usr/libexec/platform-python3.6",
            "sha256": "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"},
        "scope": {"single_fixed_t10_component": True, "prelayout": True,
            "spef": False, "ideal_clock": True, "zero_wireload": True,
            "macro_count": 0, "mapped_identity_mutated": False},
        "claim_boundary": {"fresh_result_hammer_required": True, "power": False,
            "energy": False, "speedup": False, "system": False,
            "paper_ppa_ready": False, "headline": False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", required=True)
    parser.add_argument("--expected-admission-sha", required=True)
    parser.add_argument("--expected-wrapper-sha", required=True)
    parser.add_argument("--expected-orchestrator-sha", required=True)
    args = parser.parse_args()
    admission_path = Path(args.admission)
    if sha(admission_path) != args.expected_admission_sha:
        raise RuntimeError("admission SHA")
    data = json.loads(admission_path.read_text())
    validate_admission(data, args)
    validate_inputs(data)
    if sha(WRAPPER) != args.expected_wrapper_sha or sha(SELF) != args.expected_orchestrator_sha:
        raise RuntimeError("entry identity")
    if os.environ != {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                      "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
                      "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
                      "M1311_EXPECTED_WRAPPER_SHA256": args.expected_wrapper_sha,
                      "M1311_EXPECTED_ORCHESTRATOR_SHA256": args.expected_orchestrator_sha,
                      "M1311_EXPECTED_ADMISSION_SHA256": args.expected_admission_sha}:
        raise RuntimeError("environment")
    fresh = (M1288_CANONICAL, M1288_WORK, M1288_ATTEMPT,
             M1302_CANONICAL, M1302_WORK, M1302_ATTEMPT,
             CANONICAL, WORK, ATTEMPT)
    if any(path.exists() or path.is_symlink() for path in fresh):
        raise RuntimeError("namespace not fresh")
    first_collisions = collisions()
    if first_collisions["blocking"]:
        raise RuntimeError("same UID EDA collision")
    mem = meminfo()
    headroom = mem["CommitLimit"] - mem["Committed_AS"]
    disk = shutil.disk_usage(str(RUNS)).free // 1024
    if mem["MemAvailable"] < 8388608 or headroom < 8388608 or disk < 4194304:
        raise RuntimeError("resource gate")
    license_run = subprocess.run(
        [str(LMUTIL), "lmstat", "-c", "27030@ic.ismd-nemo", "-f", "PrimeTime"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    if license_run.returncode != 0:
        raise RuntimeError("license query")
    match = re.search(br"Users of PrimeTime:.*?Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use",
                      license_run.stdout, re.S)
    if not match:
        raise RuntimeError("license parse")
    issued, in_use = map(int, match.groups())
    if issued <= in_use:
        raise RuntimeError("no free PrimeTime license")
    second_collisions = collisions()
    if second_collisions["blocking"] or any(path.exists() or path.is_symlink() for path in fresh):
        raise RuntimeError("post-preflight collision/freshness")

    ATTEMPT.mkdir()
    attempted = True
    try:
        (ATTEMPT / "attempt.txt").write_text("status=M1311_ONE_SHOT_ATTEMPT_CONSUMED\n")
        WORK.mkdir()
        shutil.copy2(str(admission_path), str(WORK / "launch_admission.json"))
        shutil.copy2(str(CONTRACT), str(WORK / "source_contract.json"))
        (WORK / "preflight_summary.json").write_text(json.dumps({
            "mem_available_kib": mem["MemAvailable"], "commit_headroom_kib": headroom,
            "filesystem_available_kib": disk, "license_feature": "PrimeTime",
            "licenses_issued": issued, "licenses_in_use": in_use,
            "repo_scoped_same_uid_collision_count": 0,
            "external_same_uid_eda_record_only": second_collisions["external_record_only"]},
            sort_keys=True) + "\n")
        child_env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
            "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
            "M1288_EXPECTED_RUNNER_SHA256": "a7fa2c5b031a446562d0bdb8f6f80112d7348fff6be92efdbf5b12830f6b928c",
            "M1288_EXPECTED_ADMISSION_SHA256": "1ea53ea55a8cc2bbc992aa932f73e7865561f7dde16e53f5d74efe3a7b146e3e"}
        completed = subprocess.run(["/usr/bin/bash", str(M1288_RUNNER)], cwd=str(HW), env=child_env)
        if completed.returncode != 0:
            raise RuntimeError("M1288 exit %d" % completed.returncode)
        verify_dir(M1288_CANONICAL)
        receipt = result_receipt()
        (WORK / "m1311_adjudication_receipt_r1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (WORK / "RUN_COMPLETE.txt").write_text(receipt["status"] + "\n")
        (WORK / "GATE_EXIT_CODE.txt").write_text(
            ("0" if receipt["strict_timing_gate_pass"] else "10") + "\n")
        shutil.copy2(str(M1288_CANONICAL / "SHA256SUMS"), str(WORK / "m1288_result_SHA256SUMS"))
        shutil.copy2(str(M1288_CANONICAL / "SHA256SUMS.seal.sha256"),
                     str(WORK / "m1288_result_outer_seal.sha256"))
        if sha(DOC359) != data["exact_files"]["docs/359_DATE终局冻结_20260813.md"]:
            raise RuntimeError("docs359 post guard")
        seal_dir(ATTEMPT)
        seal_dir(WORK)
        os.replace(str(WORK), str(CANONICAL))
        return 0 if receipt["strict_timing_gate_pass"] else 10
    except BaseException as error:
        if attempted:
            if not WORK.exists():
                WORK.mkdir()
            failure = {"schema": "m1311_failure_receipt_v1",
                "status": "STOP_M1311_LAUNCH_OR_ADJUDICATION_FAILED",
                "error_type": type(error).__name__,
                "claim_boundary": {"timing_gate_pass": False, "power": False,
                    "energy": False, "speedup": False, "system": False,
                    "paper_ppa_ready": False, "headline": False}}
            (WORK / "m1311_failure_receipt_r1.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n")
            (WORK / "RUN_COMPLETE.txt").write_text(
                "STOP_M1311_LAUNCH_OR_ADJUDICATION_FAILED\n")
            seal_dir(WORK)
            quarantine = Path(str(CANONICAL) + ".failed_or_incomplete.%d.quarantine" % os.getpid())
            os.replace(str(WORK), str(quarantine))
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.stderr.write("M1311_STOP:%s\n" % type(exc).__name__)
        sys.exit(1)

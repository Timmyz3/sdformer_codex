#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
Q = HW / "dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828.failed_or_incomplete.2923446.quarantine"
A = HW / "dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed"
DOC = HW / "docs/359_DATE终局冻结_20260813.md"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_manifest(path):
    out = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out

checks = {}
current = parse_manifest(HERE / "quarantine_current_manifest.sha256")
live = {"./" + str(p.relative_to(Q)): sha(p) for p in sorted(Q.rglob("*")) if p.is_file()}
checks["quarantine_current_manifest_exact"] = live == current

historic = parse_manifest(Q / "SHA256SUMS")
checks["historic_outer_seal_still_binds_manifest"] = (
    (Q / "SHA256SUMS.seal.sha256").read_text().split()[0] == sha(Q / "SHA256SUMS")
)
mismatches = sorted(rel for rel, digest in historic.items() if not (Q / rel[2:]).is_file() or sha(Q / rel[2:]) != digest)
checks["historic_member_mismatch_only_dc_log"] = mismatches == ["./fixed/dc.log"]
checks["historic_dc_log_sha"] = historic["./fixed/dc.log"] == "1a527e4b44e53539c7e66309c7a9fedd38653c7ca08111c886d2ff09606d3bba"
checks["current_dc_log_sha"] = current["./fixed/dc.log"] == "e97921cc8046b75c5a255e763d727875a11958343bc7d202bcb29c60bb44b259"

child = dict(line.split("=", 1) for line in (Q / "fixed/dc_child_identity.txt").read_text().splitlines())
collision_lines = (Q / "fixed/resource_runtime_external_collisions.tsv").read_text().splitlines()
runtime7 = collision_lines[0].split("\t")
runtime_final = collision_lines[1].split("\t")
checks["root_pid_exact"] = child["pid"] == "3022873"
checks["runtime7_candidate_exact"] = runtime7[2] == "3206061"
checks["runtime7_candidate_parent_is_root"] = runtime7[3] == child["pid"]
checks["runtime7_candidate_same_uid"] = runtime7[4] == child["uid"] == "1913"
checks["runtime7_candidate_same_executable"] = runtime7[6] == child["exe"]
checks["runtime7_candidate_same_cmdline"] = runtime7[7] == child["cmdline_nul_hex"]
checks["runtime_final_worker_reparented_to_init"] = runtime_final[2] == runtime7[2] and runtime_final[3] == "1"

gate = (Q / "fixed/runtime_gate_every_snapshot.log").read_text().splitlines()
checks["first_six_runtime_samples_clean"] = len(gate) == 7 and all("gate=none" in x for x in gate[:6])
checks["runtime7_only_gate_is_collision"] = "gate=external_eda_collision_immediate" in gate[6]
resource = (Q / "fixed/resource_runtime.log").read_text().splitlines()
numbers = []
for line in resource:
    d = dict(item.split("=", 1) for item in line.split())
    numbers.append(d)
checks["resource_thresholds_healthy"] = all(
    int(d["commit_headroom_kib"]) >= 41943040
    and int(d["mem_available_kib"]) >= 134217728
    and int(d["swap_free_kib"]) >= 33554432
    and d["cgroup_failcnt"] == d["cgroup_under_oom"] == d["cgroup_oom_kill"] == "0"
    for d in numbers
)
failure = dict(line.split("=", 1) for line in (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().splitlines()[1:])
checks["quarantine_status_do_not_cite"] = (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text().startswith("status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n")
checks["failure_codes_match"] = failure["runner_exit_code"] == failure["child_exit_code"] == "11" and failure["monitor_exit_code"] == "1"
ack = (Q / "fixed/runtime_final_gate_ack.txt").read_text()
checks["monitor_final_ack_failed_only_collision"] = "reason=runtime_final_oom_or_collision" in ack and "status=FAIL_FINAL_GATE_ACK" in ack
dc_log = (Q / "fixed/dc.log").read_text(errors="replace")
checks["home_gui_error_observed"] = "no such variable" in dc_log and '::env(HOME)' in dc_log
checks["epipe_after_forced_tree_break_observed"] = dc_log.count("EPIPE") >= 1

attempt_manifest = parse_manifest(A / "SHA256SUMS")
checks["r4_attempt_member_manifest_valid"] = all(sha(A / rel[2:]) == digest for rel, digest in attempt_manifest.items())
checks["r4_attempt_outer_seal_valid"] = (A / "SHA256SUMS.seal.sha256").read_text().split()[0] == sha(A / "SHA256SUMS")
checks["r4_attempt_consumed_before_launch"] = (A / "ATTEMPT_CONSUMED.txt").read_text().startswith("status=CONSUMED_BEFORE_EXACT_POINT_DC_LAUNCH\n")
checks["docs359_frozen"] = sha(DOC) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

failed = sorted(k for k, v in checks.items() if not v)
result = {
    "status": "PASS_M915_M518_R4_FIXED_DESCENDANT_COLLISION_QUARANTINE_FORENSIC" if not failed else "FAIL_M915_FORENSIC",
    "score": 100 if not failed else 0,
    "p0_p1_p2": [0, 0, 0] if not failed else [1, 0, 0],
    "checks": checks,
    "failed": failed,
    "root_cause": "runtime collision scan excluded only the exact root PID, misclassified its exact direct child worker as external, terminated the root, then sealed before the reparented worker stopped writing dc.log",
    "required_successor": "additive Fixed-only runner with exact-root descendant exclusion, isolated setsid job tree drain before seal, and private safe HOME",
    "eda_executed": False,
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(1 if failed else 0)

#!/usr/bin/env python3
"""Read-only hammer for the consumed M1808 C3 mapped-energy attempt."""

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m1808_c3_mapped_energy_attempt_consumed"
FAILURE = RESULTS / "m1808_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
PREFLIGHT = RESULTS / (
    "m1808_c3_mapped_energy_r1_20260902."
    "preflight_rejected_source_chain_governance_quarantine"
)
PRIVATE = RESULTS / "m1808_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"
CANONICAL = RESULTS / "m1808_c3_mapped_energy_r1_20260902"
TB = HW / "dc_handoff/tb/tb_m1808_c3_m1454_fixed_t10_mapped_energy_reset_settling.sv"
CHECKER = HW / "system_simulator/scripts/check_m1808_c3_m1454_fixed_t10_mapped_energy_source.py"
RELEASE = HW / "contracts/m1841_m1840_m1839_m1808_c3_preflight_recovery_launch_release_r1_20260902.json"
M1842 = HW / "reviews/m1842_m1841_c3_preflight_recovery_launch_release_audit_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


EXPECTED = {
    "attempt_json": "391505ac79570617f7be75b87be7981455fe175c18c43e5185f253b5e07ebd11",
    "attempt_manifest": "310ce8741e3e62f082a5ad48249b19cce7d4ee3d9caf2424af5d2f0be2e83ea2",
    "attempt_outer_file": "dd25f2dcc87fa7b18a1561685616d761719647b9cc14d94bf7aa23bf5bbfcd52",
    "failure_json": "aea36bcd319f89be78ba7a6b26f0ec02acf17f095601ad227ac194759f063427",
    "failure_manifest": "ff0eaefd6ac92f539cbcd5d01f05592e73da568731f56584c17e06dc358ad2e2",
    "failure_outer_file": "bda296e738e3a6e8ad8791217c9ad3ed2706e53221796eab87fbd98086312b1a",
    "preflight_json": "ea9d08303dd29196a761c1e9927e5aa148a5f8746e1d5b4a64f354d66c74eda8",
    "preflight_manifest": "e243c0f10d810b1b5d39523ad479a1df2d751a3f139d7eae944072d2788eb856",
    "preflight_outer_file": "d9824a782b5ee5f1ba116abe2c7719a24579815798ee5b1d48b342de38784124",
    "compile_log": "cad754ee9aabe3edcecd348f652baea84173dae23bb09ec869c4e7248dd684b9",
    "mapped_sim_log": "ecd0fdb2141b701bf946f04ebd88821593491faac2f534c098ca3fc93bcc5365",
    "tb": "e3c39d9a80af0c17bbdc895c06231053b49d3c822bf075be66f4b92c3b154288",
    "checker": "cf36c026997e066871b9db68770e1dd6cf7a6ed3bf15ae1b858f91680206c498",
    "release": "68698e10cb2e625b949d98f157d70ca896546aef3149a96bae285ede2f09c6da",
    "release_sidecar": "3cb4c33b7daeabe28d3dda989925515d2af302880dc78425cfa64797786560d8",
    "release_outer_file": "7d31e71ab305ef0d37f2ab615bcac2676f64b24e7ea7a2f16af229dbabcbf53f",
    "m1842_review": "f6fd9c87bc8438b4c99915b832c78956224c67c4b7c27ece74032438b845d03c",
    "m1842_manifest": "dd14836172114a327f758c11c963bbc4e431a7efb29cfe813d29a53d567f13fd",
    "m1842_outer_file": "72b7cc647371862a05d66ea9da0bdb0bd4505fbf9db9ea020cfa28293af73c7a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_double_seal(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), "missing double seal " + str(root))
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        need(path.is_file() and sha(path) == digest, "manifest mismatch " + str(path))
    outer_digest, outer_name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    need(outer_name == "SHA256SUMS" and sha(manifest) == outer_digest,
         "outer seal mismatch " + str(root))


def verify_contract_sidecars(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    digest, name = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    need(name == path.name and digest == sha(path), "release sidecar mismatch")
    outer_digest, outer_name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    need(outer_name == sidecar.name and outer_digest == sha(sidecar),
         "release outer mismatch")


def exact_hash(path, key):
    need(Path(path).is_file() and sha(path) == EXPECTED[key], "identity drift " + key)


def main():
    for root in (ATTEMPT, FAILURE, PREFLIGHT, M1842):
        verify_double_seal(root)
    verify_contract_sidecars(RELEASE)

    exact_hash(ATTEMPT / "attempt.json", "attempt_json")
    exact_hash(ATTEMPT / "SHA256SUMS", "attempt_manifest")
    exact_hash(ATTEMPT / "SHA256SUMS.seal.sha256", "attempt_outer_file")
    exact_hash(FAILURE / "failure.json", "failure_json")
    exact_hash(FAILURE / "SHA256SUMS", "failure_manifest")
    exact_hash(FAILURE / "SHA256SUMS.seal.sha256", "failure_outer_file")
    exact_hash(PREFLIGHT / "failure.json", "preflight_json")
    exact_hash(PREFLIGHT / "SHA256SUMS", "preflight_manifest")
    exact_hash(PREFLIGHT / "SHA256SUMS.seal.sha256", "preflight_outer_file")
    exact_hash(PRIVATE / "build/compile.log", "compile_log")
    exact_hash(PRIVATE / "candidate/mapped_sim.log", "mapped_sim_log")
    exact_hash(TB, "tb")
    exact_hash(CHECKER, "checker")
    exact_hash(RELEASE, "release")
    exact_hash(Path(str(RELEASE) + ".sha256"), "release_sidecar")
    exact_hash(Path(str(RELEASE) + ".sha256.seal.sha256"), "release_outer_file")
    exact_hash(M1842 / "review.json", "m1842_review")
    exact_hash(M1842 / "SHA256SUMS", "m1842_manifest")
    exact_hash(M1842 / "SHA256SUMS.seal.sha256", "m1842_outer_file")
    exact_hash(DOCS359, "docs359")

    attempt = strict_json(ATTEMPT / "attempt.json")
    failure = strict_json(FAILURE / "failure.json")
    preflight = strict_json(PREFLIGHT / "failure.json")
    need(attempt.get("status") == "M1808_ATTEMPT_CONSUMED", "attempt status")
    need(attempt.get("budget") == {
        "ptpx_runs": 1, "saif_files": 1, "simv_runs": 1, "vcs_compiles": 1},
        "attempt budget")
    need(attempt.get("automatic_retry") is False and
         attempt.get("reuse_prior_simv_saif_ptpx") is False, "attempt retry/reuse")
    need(failure == {
        "attempt_consumed": True,
        "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 1, "vcs_compiles": 1},
        "error": "RuntimeError",
        "phase": "MAPPED_SIM_SAIF",
        "status": "FAILED_OR_INCOMPLETE"}, "production failure drift")
    need(preflight == {
        "attempt_consumed": False,
        "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure",
        "phase": "SOURCE_CHAIN",
        "status": "FAILED_OR_INCOMPLETE"}, "preflight failure drift")
    need(not CANONICAL.exists(), "canonical result unexpectedly exists")
    namespaces = sorted(path.name for path in RESULTS.glob("*m1808_c3_mapped_energy*")
                        if not path.name.startswith("."))
    need(namespaces == sorted([FAILURE.name, PREFLIGHT.name, PRIVATE.name]),
         "production namespace multiplicity")
    hidden = sorted(path.name for path in RESULTS.glob(".m1808_c3_mapped_energy*"))
    need(hidden == [ATTEMPT.name], "attempt latch multiplicity")

    compile_log = (PRIVATE / "build/compile.log").read_text(encoding="utf-8")
    sim_log = (PRIVATE / "candidate/mapped_sim.log").read_text(encoding="utf-8")
    tb_text = TB.read_text(encoding="utf-8")
    checker_text = CHECKER.read_text(encoding="utf-8")
    need(compile_log.count("Top Level Modules:") == 1 and
         compile_log.count("tb_m1808_c3_m1454_fixed_t10_mapped_energy") >= 2 and
         compile_log.count("CPU time:") == 1, "single VCS compile evidence")
    need(sim_log.count("Chronologic VCS simulator copyright") == 1,
         "single sim evidence")
    need(sim_log.count("M1808 debug counter X/Z at settling boundary") == 1,
         "mapped failure signature")
    need("reset_settling.sv\", 287:" in sim_log and
         "at time 31700 ps" in sim_log and
         "$finish at simulation time                31700" in sim_log,
         "mapped failure location/time")
    need("PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY" not in sim_log and
         "PASS_M1808_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD" not in sim_log and
         "M1808_RESET_SETTLING_GATE" not in sim_log and
         "M1808_SAIF_WINDOW_START" not in sim_log and
         "M1808_PUBLIC_RESULT_CHECK" not in sim_log,
         "post-fatal token unexpectedly present")
    need("if (post_reset_settle_cycles == 3) begin" in tb_text and
         "$isunknown({debug_config_beats, debug_raw_beats," in tb_text and
         '"M1808 debug counter X/Z at settling boundary"' in tb_text and
         "repeat (8) @(posedge clk_core);" in tb_text and
         "@(negedge clk_core); rst_core = 1'b0;" in tb_text and
         "repeat (3) @(posedge clk_core);" in tb_text,
         "TB reset-settling semantics")
    lines = tb_text.splitlines()
    need(lines[286].strip() == "$fatal(1," and
         "M1808 debug counter X/Z at settling boundary" in lines[287],
         "TB line-287 binding")
    need("runtime PASS count" in checker_text and
         "PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY" in checker_text,
         "checker runtime token semantics")

    saif_files = [path for path in PRIVATE.rglob("*")
                  if path.is_file() and path.suffix.lower() == ".saif"]
    ptpx_files = [path for path in PRIVATE.rglob("*")
                  if path.is_file() and "ptpx" in path.name.lower()]
    need(not saif_files and not ptpx_files, "unexpected SAIF/PTPX evidence")

    result = {
        "schema": "m1847_m1808_c3_mapped_energy_failure_mechanical_checks_r1_v1",
        "status": "PASS_INDEPENDENT_FAILURE_AUDIT__M1808_PRODUCTION_FAIL_CLOSED",
        "classification": "PERSISTENT_XZ_AT_THREE_CYCLE_RESET_SETTLING_BOUNDARY__NOT_TOKEN_ONLY__FUNCTIONAL_WORKLOAD_NOT_REACHED",
        "checks": {
            "docs359_exact_sha": "PASS",
            "m1841_release_double_seal": "PASS",
            "m1842_release_audit_double_seal": "PASS",
            "attempt_double_seal": "PASS",
            "failure_double_seal": "PASS",
            "preflight_governance_quarantine_double_seal": "PASS",
            "attempt_latches": 1,
            "production_failure_quarantines": 1,
            "preflight_quarantines": 1,
            "private_builds": 1,
            "canonical_results": 0,
            "vcs_compiles": 1,
            "simv_runs": 1,
            "saif_files": 0,
            "ptpx_runs": 0,
            "automatic_retry": False,
            "failure_time_ps": 31700,
            "failure_tb_line": 287,
            "three_post_release_settling_edges_consumed": True,
            "runtime_pass_tokens": 0,
            "numeric_workload_reached": False,
            "token_only_failure": False,
            "functional_arithmetic_failure_proven": False,
            "component_energy_admitted": False,
        },
        "execution": {"vcs": 0, "simv": 0, "saif": 0, "ptpx": 0,
                      "dc": 0, "formality": 0, "remote": 0, "gpu": 0},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

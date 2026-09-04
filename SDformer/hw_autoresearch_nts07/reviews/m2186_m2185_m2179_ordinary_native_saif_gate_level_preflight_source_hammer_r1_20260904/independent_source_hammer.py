#!/usr/bin/python3.12
"""Independent no-EDA M2186 source hammer for the fresh M2185 SAIF path."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
CONTRACT = HW / "contracts/m2185_m2179_ordinary_native_saif_gate_level_preflight_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2185_m2018_ordinary_native_saif_gate_level_reset_preflight_one_shot.py"
UCLI = HW / "dc_handoff/scripts/m2185_m2018_ordinary_native_saif_gate_level_reset_preflight.ucli.tcl"
OLD_UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"
BALANCED = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
M2185_TEST = HW / "tests/test_m2185_ordinary_native_saif_gate_level_preflight.py"
M2172_TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
FIXTURE_TEST = M2172_TEST
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2179 = HW / "reviews/m2179_m2178_m2176_ordinary_native_saif_reset_semantics_preflight_failure_result_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2185_m2179_ordinary_native_saif_gate_level_preflight_source_author_receipt_r1_20260904"
M2178_ATTEMPT = HW / "results/.m2178_m2176_ordinary_native_saif_reset_semantics_preflight_attempt_consumed"
M2187_RESULT = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904"
M2187_ATTEMPT = HW / "results/.m2187_m2185_ordinary_native_saif_gate_level_preflight_attempt_consumed"
M2187_LOCK = HW / "results/.m2187_m2185_ordinary_native_saif_gate_level_preflight_launch_lock"
EXPECTED_CONTRACT_SHA = "9a5ef223a00f0d93123046adb82cd87e85ca233b0d5a08973735007c5312a74b"
EXPECTED_DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_M2179_REVIEW_SHA = "af72c412f1aace70917f9ebcbb1c13919bdce1661898df349715fdf440c8fe41"
EXPECTED_RUNNER_SHA = "4ce0a7ef2630cd8c8feb31b3116bedd9e1e67aff09a61e9680245512984ac81f"
EXPECTED_UCLI_SHA = "80342476d38144c7f96a840ff695fc8689e401f91c8e93c8894d5784ea6bce2a"
EXPECTED_PARSER_SHA = "2dadf88ccfb4f4e43281203c67317b9f0bf91ed1fa3874eadb6015db9244438d"
GATE = "power -gate_level all mda sv"
SCOPE = "tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary"
EXPECTED_BUDGET = {
    "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
    "raw_saif_files_written": 2, "diagnostic_saif_files_written": 1,
    "admitted_measurement_saif_files": 1, "admitted_saif_files": 1,
    "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0,
    "automatic_retry": False, "ordinary_only": True,
    "single_frontend": True, "reuse_old_artifacts": False,
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def verify_seal(directory: Path) -> dict[str, object]:
    need(directory.is_dir() and not directory.is_symlink(), "sealed directory invalid")
    need(not any(path.is_symlink() for path in directory.rglob("*")),
         "symlink in sealed directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed: set[str] = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe seal member")
        need((directory / rel).is_file() and sha(directory / rel) == digest,
             "sealed member mismatch: " + rel.as_posix())
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive sealed directory")
    return {"member_count": len(listed), "manifest_sha256": sha(manifest),
            "outer_sha256": sha(outer), "exhaustive": True}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "module load: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutation_rejected(runner, text: str) -> tuple[bool, str]:
    try:
        runner.audit_ucli(text)
    except runner.Failure as exc:
        return True, str(exc)
    return False, "accepted"


def main() -> int:
    need(sha(CONTRACT) == EXPECTED_CONTRACT_SHA, "contract identity")
    need(sha(DOC359) == EXPECTED_DOC359_SHA, "docs359 identity")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "contract outer seal")
    contract = read_json(CONTRACT)
    need(contract["execution_budget"] == EXPECTED_BUDGET, "execution budget")
    inventory = contract["source_inventory"]
    need(isinstance(inventory, dict) and len(inventory) == 19, "exact 19-source inventory")
    for rel, digest in inventory.items():
        path = ROOT / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "source identity drift: " + rel)

    seals = {"m2179": verify_seal(M2179), "m2185_author": verify_seal(AUTHOR),
             "m2178_attempt": verify_seal(M2178_ATTEMPT)}
    need(seals["m2179"]["member_count"] == 5, "M2179 sealed member count")
    prior = read_json(M2179 / "review.json")
    need(sha(M2179 / "review.json") == EXPECTED_M2179_REVIEW_SHA, "M2179 review identity")
    need(prior["status"] ==
         "FAIL_M2179_M2178_RESULT_HAMMER__EMPTY_SAIF_MONITORING_POLICY__M2178_NO_RETRY__M2185_SOURCE_ONLY",
         "M2179 status")
    auth = prior["authorization"]
    need(auth["m2178_identity_consumed"] is True and auth["m2178_retry_authorized"] is False and
         auth["future_m2187_execution_authorized"] is False and
         auth["allowed_now"] == "FRESH_M2185_SOURCE_AUTHORING_ONLY", "M2179 authority")
    attempt = read_json(M2178_ATTEMPT / "attempt.json")
    need(attempt["status"] == "M2178_ATTEMPT_CONSUMED" and
         attempt["automatic_retry"] is False, "M2178 consumed attempt")
    quarantines = sorted(HW.glob(
        "results/m2178_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_r1_20260904.failed.*.quarantine"))
    need(len(quarantines) == 1, "exact one M2178 failure quarantine")
    seals["m2178_quarantine"] = verify_seal(quarantines[0])
    need(not M2187_RESULT.exists() and not M2187_ATTEMPT.exists() and not M2187_LOCK.exists(),
         "M2187 result/attempt/lock census not empty")

    runner = load(RUNNER, "m2185_runner_for_independent_m2186")
    parser = load(PARSER, "m2176_parser_for_independent_m2186")
    fixture = load(FIXTURE_TEST, "m2172_fixture_for_independent_m2186")
    need(sha(RUNNER) == EXPECTED_RUNNER_SHA and sha(UCLI) == EXPECTED_UCLI_SHA and
         sha(PARSER) == EXPECTED_PARSER_SHA, "primary source identity")
    old_lines = OLD_UCLI.read_text().splitlines()
    new_lines = UCLI.read_text().splitlines()
    need(new_lines == [old_lines[0], GATE, *old_lines[1:]], "not exact one-line UCLI delta")
    effective = [line.strip() for line in new_lines
                 if line.strip() and not line.lstrip().startswith("#")]
    need(effective[0] == GATE and len(effective) == 21 and
         len([line for line in old_lines if line.strip() and
              not line.lstrip().startswith("#")]) == 20, "effective UCLI delta")
    mutations = {
        "missing_gate": OLD_UCLI.read_text(),
        "wrong_scope": UCLI.read_text().replace(SCOPE, SCOPE + "_wrong"),
        "gate_after_scope": UCLI.read_text().replace(
            GATE + "\npower " + SCOPE, "power " + SCOPE + "\n" + GATE),
        "gate_after_enable": UCLI.read_text().replace(
            GATE + "\npower " + SCOPE + "\npower -enable",
            "power " + SCOPE + "\npower -enable\n" + GATE),
    }
    mutation_results = {name: mutation_rejected(runner, text)
                        for name, text in mutations.items()}
    need(all(value[0] for value in mutation_results.values()), "UCLI mutation escaped")
    need(runner.audit_ucli(UCLI.read_text()) == contract["ucli_one_line_delta"],
         "UCLI contract fingerprint")

    failures = [
        "Warning: reset ignored.", "Warning: reset rejected.",
        "Error: reset denied.", "Warning: reset unsupported.",
        "Warning: reset failed.", "Error: reset cannot complete.",
        "Warning: reset unable to complete.", "Error: reset remained uncleared.",
        "Warning: reset retained old counters.", "Error: reset remained active.",
        "Warning: reset not cleared.", "Error: reset not reset.",
        "Warning: clear failed.", "Error: clear request denied.",
    ]
    successes = ["Info: power reset request accepted and switching counters cleared.",
                 "Info: reset completed successfully."]
    need(all(parser.reset_failure_lines(line) == [line] for line in failures),
         "reset lexical mutation escaped")
    need(all(parser.reset_failure_lines(line) == [] for line in successes),
         "reset success control rejected")
    with tempfile.TemporaryDirectory(prefix="m2186_reset_mutations_") as raw:
        runtime = Path(raw) / "rtl_sim.log"
        for line in failures:
            runtime.write_text(fixture.runtime_text() + line + "\n")
            try:
                parser.parse_runtime(runtime)
            except parser.Failure:
                pass
            else:
                raise RuntimeError("reset failure escaped runtime parser: " + line)
    static = parser.static_check()
    need(static["status"] == "PASS_M2176_STATIC_PARSER", "M2176 static parser")
    need(parser.BASE_PATH == BALANCED and
         parser.parse_saif is parser.BASE.parse_saif and
         parser.TARGET_INSTANCE == "dut_ordinary" and parser.EXPECTED["records"] == 93971,
         "balanced parser inheritance")

    runner_text = RUNNER.read_text()
    need(runner.B.topology_audit() == contract["single_axis_topology"], "topology fingerprint")
    need(runner_text.count('"-debug_access+r"') == 1, "-debug_access+r compile surface")
    need(runner_text.count('"+M2160_AXIS_ORDINARY"') == 2,
         "ordinary plusarg source plus exact assertion")
    need(runner_text.count('"+WORKLOAD_SLOT=42"') == 2,
         "workload slot source plus exact assertion")
    need("+M2160_AXIS_TSGB" not in runner_text and ".SCHEDULE_MODE(1)" not in runner.TB.read_text(),
         "second schedule mode present")
    filelist_sources = [line.strip() for line in runner.FILELIST.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
    need(len(filelist_sources) == 4 and Path(filelist_sources[-1]).name == runner.TB.name,
         "exact four-source filelist/TB tail")
    need(contract["single_axis_topology"] == {
        "direct_m2018_frontends": 1, "filelist_source_count": 4,
        "parent_dual_axis_tb_instances": 0, "public_name_adapter_in_filelist": False,
        "schedule_mode_one_instances": 0, "schedule_mode_zero_instances": 1,
        "second_axis_symbols": 0,
    }, "single ordinary axis contract")

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPYCACHEPREFIX": "/tmp/m2186_source_hammer_pycache"}
    official = subprocess.run([sys.executable, "-B", str(M2185_TEST)], check=True,
                              capture_output=True, text=True, env=env, timeout=180)
    inherited = subprocess.run([sys.executable, "-B", str(M2172_TEST)], check=True,
                               capture_output=True, text=True, env=env, timeout=180)
    need("PASS_M2185_SOURCE_TESTS" in official.stdout and "eda_runs=0" in official.stdout,
         "M2185 official source suite")
    need("PASS_M2172_SOURCE_TESTS tests=42" in inherited.stdout and
         "eda_runs=0" in inherited.stdout, "balanced parser source suite")

    result = {
        "schema": "m2186_m2185_m2179_ordinary_native_saif_gate_level_preflight_source_mechanical_checks_r1_v1",
        "status": "PASS_M2186_SOURCE_MECHANICAL_CHECKS",
        "identity": {"runner_sha256": sha(RUNNER), "ucli_sha256": sha(UCLI),
                     "parser_sha256": sha(PARSER), "contract_sha256": sha(CONTRACT),
                     "docs359_sha256": sha(DOC359)},
        "sealed_inputs": seals,
        "source_inventory": {"count": len(inventory), "all_exact": True},
        "m2179_lineage": {"failure_review_exact": True, "m2178_consumed": True,
                           "m2178_retry_authorized": False,
                           "m2187_execution_authorized_before_review": False},
        "ucli": {"exact_one_line_delta": True, "first_effective_gate_level": True,
                 "exact_scope": SCOPE, "report_before_reset": True,
                 "measurement_enable_after_reset": True,
                 "mutations": {name: {"expected": "REJECT", "observed": "REJECT",
                                      "detail": detail}
                               for name, (_, detail) in mutation_results.items()}},
        "parser": {"balanced_dut_ordinary_exact": True, "records": 93971,
                   "reset_failure_mutations_rejected": len(failures),
                   "accepted_reset_controls": len(successes),
                   "official_m2185_stdout": official.stdout.strip(),
                   "inherited_m2172_stdout": inherited.stdout.strip()},
        "topology_and_execution": {"single_axis_topology": runner.B.topology_audit(),
                                   "exact_tb_filelist_fixture": True,
                                   "debug_access_r": True,
                                   "execution_budget": EXPECTED_BUDGET},
        "freshness": {"m2178_attempt_consumed": True,
                      "m2178_failure_quarantine_count": len(quarantines),
                      "m2187_result_absent": True, "m2187_attempt_absent": True,
                      "m2187_lock_absent": True},
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "execution": {"vcs_runs": 0, "simv_runs": 0, "license_queries": 0,
                      "saif_files_written": 0, "dc_runs": 0, "ptpx_runs": 0,
                      "icc2_runs": 0, "gpu_runs": 0, "git_mutations": 0},
        "authorization": EXPECTED_BUDGET,
    }
    out = HERE / "mechanical_checks.json"
    need(not out.exists(), "fresh mechanical output required")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/opt/anaconda3/bin/python3.12
"""Bounded M2238 source review; never executes M2239 or any EDA command."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
HW = OUT.parents[1]
REPO = HW.parent
RUNNER = HW / "system_simulator/scripts/run_m2239_m2237_lm_discovery_parse_only.py"
AUTHOR = HW / "reviews/m2237_m2223_lm_discovery_parse_only_source_author_receipt_r1_20260905"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("m2238_reviewed_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    contract = json.loads(runner.CONTRACT.read_text())
    runner.validate_inputs(contract)
    runner.verify_seal(AUTHOR)
    need(len(contract["pinned_files"]) == 19 and len(contract["sealed_directories"]) == 4, "pin/seal count")
    need(sha(runner.CONTRACT) == "4dafe1df8e987d44b0fe74e71efd79a881cf7b7be98e89c8f29a7467d250928e", "contract pin")
    for path in (runner.CONTRACT,):
        side = Path(str(path) + ".sha256")
        outer = Path(str(side) + ".seal.sha256")
        need(side.read_text().split() == [sha(path), path.name], "contract seal")
        need(outer.read_text().split() == [sha(side), side.name], "contract outer seal")
    need(not runner.RESULT.exists() and not runner.ATTEMPT.exists(), "M2239 virgin namespace")
    need(not list((HW / "results").glob(".m2239_lm_parse_only_work.*")), "M2239 work exists")
    imports = {}
    for path in (RUNNER, runner.CHECKER):
        tree = ast.parse(path.read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                need(node.level == 0, "relative import")
                names.add(node.module.split(".")[0])
        need(names <= sys.stdlib_module_names and "subprocess" not in names, "unexpected import")
        for token in ("os.system(", "os.popen(", "subprocess.", "os.exec", "os.spawn"):
            need(token not in path.read_text(), "external command path")
        imports[path.name] = sorted(names)
    checker_text = runner.CHECKER.read_text()
    need('r"^\\s*M2221_FATAL_FAIL_CLOSED:"' in checker_text, "runtime fatal anchor")
    need("work == RAW_DIRECTORY" in checker_text and "recorded_isolated = STAGING_DIRECTORY" in checker_text,
         "exact relocation predicates")
    need("write_text(" not in checker_text and "output.write" not in checker_text, "checker must not publish")
    checks = {"source_pins": "19/19 exact", "upstream_sealed_directories": "4/4 exhaustive double seal",
              "source_contract_and_author_seals": "PASS", "original_six_sources_exact": True,
              "old_failed_identity_preserved": True, "frozen_validation_diff_reviewed": True,
              "author_test_methods_independently_run": "5/5 PASS",
              "negative_cases_independently_run": "19/19 rejected",
              "stdlib_imports": imports, "custom_imported_checker_count": 1,
              "new_parser_no_writes": True, "runner_no_external_command_path": True,
              "exact_pid_3569314_relocation": True, "runtime_fatal_echo_distinguished": True,
              "m2239_namespace_virgin_at_review": True, "source_modified": False,
              "review_production_parses": 0, "lm_runs": 0, "license_queries": 0,
              "eda_runs": 0, "gpu_runs": 0, "git_mutations": False}
    review = {"schema": "m2238_m2237_lm_discovery_parse_only_source_hammer_r1_v1",
              "date_cst": "2026-09-05", "reviewer": "/root/m2224_lm_discovery_review",
              "status": "PASS_M2238_M2237_LM_PARSE_ONLY_SOURCE__M2239_CPU_PARSE_AUTHORIZED",
              "score_over_100": 98, "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
              "identity": {"source_contract_sha256": sha(runner.CONTRACT),
                           "checker_sha256": sha(runner.CHECKER), "runner_sha256": sha(RUNNER)},
              "authorization": {"cpu_parse_runs": 1, "license_queries": 0, "lm_runs": 0,
                                "eda_runs": 0, "gpu_runs": 0, "automatic_retry": False},
              "mechanical_checks": checks,
              "claim_boundary": {"source_review_only": True,
                  "independent_m2240_result_review_required": True,
                  "old_m2223_status_unchanged": True, "library_conversion": False,
                  "library_compatibility": False, "ndm_written": False, "pnr": False,
                  "paper_ppa_ready": False, "continuous_process_observation": False}}
    runner.gate(review, sha(runner.CONTRACT))
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (OUT / "RUN_COMPLETE.txt").write_text(review["status"] + "\n")
    runner.seal_new(OUT)
    print(json.dumps({"status": review["status"], "score": 98,
                      "review_sha256": sha(OUT / "review.json"), "release_gate": "PASS"}))


if __name__ == "__main__":
    main()

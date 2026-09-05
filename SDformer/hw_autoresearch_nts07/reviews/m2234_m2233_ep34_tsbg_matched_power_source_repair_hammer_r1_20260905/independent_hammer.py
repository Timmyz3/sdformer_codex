#!/opt/anaconda3/bin/python3
"""Bounded independent CPU review of M2233's complete parser dependency closure."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
HW = OUT.parents[1]
REPO = HW.parent
SOURCE = HW / "dc_handoff/scripts/run_m2233_ep34_tsbg_matched_power_repair_one_shot.py"
AUTHOR = HW / "reviews/m2233_ep34_tsbg_matched_power_dependency_closure_author_receipt_r1_20260905"
TEST = HW / "tests/test_m2233_ep34_tsbg_matched_power_dependency_closure.py"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    runner = load(SOURCE, "m2234_independent_runner")
    contract = runner.source_validation(False)
    runner.verify_seal(AUTHOR)
    need(len(contract["source_inventory"]) == 29, "source inventory count")
    author = json.loads((AUTHOR / "author_receipt.json").read_text())
    need(author["identity"]["runner_sha256"] == sha(SOURCE), "author runner binding")
    need(author["identity"]["contract_sha256"] == sha(runner.CONTRACT), "author contract binding")
    for path in (runner.RESULT, runner.ATTEMPT, runner.LOCK):
        need(not path.exists(), "M2235 namespace already consumed")
    need(not list((HW / "results").glob(".m2235_m2233_work.*")) and
         not list((HW / "results").glob(".m2235_m2233_stage.*")) and
         not list((HW / "results").glob("m2235_m2233*.quarantine")), "M2235 residual namespace")

    completed = subprocess.run([sys.executable, "-B", "-m", "unittest", str(TEST.relative_to(REPO)), "-v"],
                               cwd=REPO, capture_output=True, text=True)
    test_log = completed.stdout + completed.stderr
    need(completed.returncode == 0 and "Ran 10 tests" in test_log and "\nOK\n" in test_log,
         "ten author CPU tests")
    (OUT / "cpu_tests.log").write_text(test_log)

    helper_paths = (runner.STRUCT_HELPER, runner.POWER_HELPER, runner.BASE_HELPER)
    drift_rejections = []
    actual_sha = runner.sha
    for helper in helper_paths:
        def drift(path, selected=helper):
            return "0" * 64 if path == selected else actual_sha(path)
        with patch.object(runner, "sha", drift), patch.object(runner, "validate_dc_launcher",
                side_effect=RuntimeError("FORBIDDEN_LATER_GATE_REACHED")):
            try:
                runner.source_validation(False)
            except runner.Failure as exc:
                need("helper identity" in str(exc), "drift rejection reason")
                drift_rejections.append({"path": str(helper.relative_to(REPO)), "error": str(exc)})
            else:
                raise RuntimeError("helper drift admitted")

    # Observe the complete import graph independently of the author's test.
    imported = []
    original_spec = importlib.util.spec_from_file_location
    def traced_spec(name, path, *args, **kwargs):
        imported.append(Path(path).resolve())
        return original_spec(name, path, *args, **kwargs)
    with patch.object(importlib.util, "spec_from_file_location", traced_spec):
        parser = load(runner.PARSER, "m2234_parser_closure")
    expected = {runner.PARSER, *helper_paths}
    need(set(imported) == expected and len(imported) == 4, "recursive import closure")
    for path in imported:
        need(contract["source_inventory"][str(path.relative_to(REPO))] == sha(path), "import SHA closure")
    leaves = {}
    for path in (runner.BASE_HELPER, runner.POWER_HELPER):
        roots = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                need(node.level == 0, "leaf relative import")
                roots.add(node.module.split(".")[0])
        need(roots <= sys.stdlib_module_names, "nonstandard leaf import")
        for token in ("spec_from_file_location", "exec_module", "__import__", "import_module"):
            need(token not in path.read_text(), "leaf dynamic import")
        leaves[path.name] = sorted(roots)

    selected = runner.selections()
    need(tuple(selected) == ("low", "median", "high"), "fixed strata")
    need([selected[s]["global_slot"] for s in runner.STRATA] == [1606, 526, 1071], "fixed slots")
    need([selected[s]["selected_density_fraction"] for s in runner.STRATA] == [[0, 1], [9, 25], [117, 163]],
         "fixed densities")
    need(contract["selection"]["aggregate_estimand"].startswith("fixed one-third weighted index"), "estimand")
    model = parser.sram_model()
    need(model["capacity_bytes"] == 294912 and model["macro_count"] == 16 and
         model["area_um2"] == 558507.032 and
         model["dynamic_read_energy_pj_per_accepted_bank_activation"] == 22.213 and
         abs(model["leakage_power_mw"] - 3.826774326764422) < 1e-12 and
         model["identical_capacity_area_and_leakage_both_axes"], "same-capacity external model")
    mapping = contract["mapping_and_power"]
    need((mapping["dc_max_corner"], mapping["dc_min_corner"], mapping["ptpx_corner"]) ==
         ("SSG0P9V125C", "FFG1P05VM40C", "TT0P9V25C") and mapping["dc_to_ptpx_is_mixed_corner"],
         "mixed PVT")
    text = SOURCE.read_text()
    production = text[text.index("def production()") : text.index("def main()")]
    need(production.index("source_validation(require_review=True)") < production.index("LOCK.mkdir()") <
         production.index('counts["license_queries"] += 1') < production.index('for axis, mode in AXES.items()'),
         "authorization before attempt/tools")
    need("need(counts == COUNTS" in production and "automatic_retry" in production, "fixed complete budget")
    need('for stratum in STRATA:' in production and 'result["implementation_corners"]' in production and
         "FIXED_THREE_WINDOW_WEIGHTED_INDEX__NOT_POPULATION_MEAN" in production,
         "all six points and final boundary")

    checks = {"author_cpu_tests": "10/10 PASS", "source_inventory": "29/29 exact",
              "recursive_local_imports": [str(p.relative_to(REPO)) for p in imported],
              "leaf_stdlib_imports": leaves, "independent_helper_drift_rejections": drift_rejections,
              "preflight_precedes_attempt_and_all_tools": True,
              "m2235_namespace_virgin_at_review": True,
              "fixed_strata_slots": [1606, 526, 1071], "required_measured_points": 6,
              "fixed_index_not_population_mean": True, "mixed_corners_labeled": True,
              "sram_model": model, "source_modified": False, "eda_runs": 0,
              "license_queries": 0, "gpu_runs": 0, "git_mutations": False}
    identity = {"runner_sha256": sha(SOURCE), "contract_sha256": sha(runner.CONTRACT),
                "m2172_helper_sha256": sha(runner.STRUCT_HELPER),
                "m2117_helper_sha256": sha(runner.POWER_HELPER),
                "m2160_helper_sha256": sha(runner.BASE_HELPER),
                "parser_sha256": sha(runner.PARSER), "selection_sha256": sha(runner.SELECTION),
                "test_sha256": sha(TEST), "author_receipt_sha256": sha(AUTHOR / "author_receipt.json"),
                "docs359_sha256": sha(runner.DOC359), "independent_hammer_sha256": sha(Path(__file__))}
    review = {"schema": "m2234_m2233_ep34_tsbg_matched_power_source_repair_hammer_r1_v1",
              "date_cst": "2026-09-05", "reviewer": "/root/m2224_lm_discovery_review",
              "status": "PASS_M2234_M2233_MATCHED_POWER_SOURCE_REPAIR_RELEASE",
              "score_over_100": 98, "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
              "authorization": contract["execution_budget"], "identity": identity,
              "mechanical_checks": checks,
              "claim_boundary": {"source_review_only": True, "m2235_single_campaign_authorized": True,
                  "independent_m2236_result_hammer_required": True, "paper_result": False,
                  "post_layout": False, "hold_closed": False, "full_network": False,
                  "energy_per_frame": False, "silicon": False}}
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (OUT / "RUN_COMPLETE.txt").write_text(review["status"] + "\n")
    runner.seal_dir(OUT)
    runner.source_validation(True)
    print(json.dumps({"status": review["status"], "score": 98,
                      "review_sha256": sha(OUT / "review.json"), "actual_release_gate": "PASS"}))


if __name__ == "__main__":
    main()

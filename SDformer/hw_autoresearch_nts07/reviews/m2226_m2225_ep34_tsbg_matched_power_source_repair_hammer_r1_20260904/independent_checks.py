#!/opt/anaconda3/bin/python3
"""M2226 independent CPU-only review. Never invokes EDA or mutates sources."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2225_ep34_tsbg_matched_power_repair_one_shot.py"
CONTRACT = HW / "contracts/m2225_ep34_tsbg_matched_power_source_repair_contract_r1_20260904.json"
PARSER = HW / "system_simulator/scripts/parse_m2217_ep34_tsbg_matched_power.py"
BASE = HW / "system_simulator/scripts/parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    runner = load(RUNNER, "m2226_runner")
    contract = json.loads(CONTRACT.read_text())
    inventory = contract["source_inventory"]
    checks = {}
    checks["inventory_28_exact_hashes"] = len(inventory) == 28 and all(
        digest(REPO / name) == sha for name, sha in inventory.items())
    runner.source_validation(False)
    checks["static_source_validation_passes"] = True
    old = json.loads((HW / "contracts/m2217_ep34_tsbg_matched_power_source_contract_r1_20260904.json").read_text())
    checks["all_original_m2217_inventory_bytes_unchanged"] = all(
        digest(REPO / name) == sha for name, sha in old["source_inventory"].items())
    for helper, field in ((runner.STRUCT_HELPER, "STRUCT_HELPER_SHA"),
                          (runner.POWER_HELPER, "POWER_HELPER_SHA")):
        with patch.object(runner, field, "0" * 64):
            try:
                runner.source_validation(False)
            except runner.Failure as exc:
                checks[helper.stem + "_mutation_rejected"] = "helper identity" in str(exc)
            else:
                checks[helper.stem + "_mutation_rejected"] = False

    hashed = []
    original_sha = runner.sha
    def sha_spy(path):
        hashed.append(Path(path).resolve())
        return original_sha(path)
    with patch.object(runner, "sha", sha_spy):
        runner.source_validation(False)
    checks["m2160_absent_from_inventory"] = BASE.relative_to(REPO).as_posix() not in inventory
    checks["m2160_never_hashed_by_source_gate"] = BASE not in hashed

    imports = []
    original_spec = importlib.util.spec_from_file_location
    def traced_spec(name, path, *args, **kwargs):
        imports.append(str(Path(path).resolve()))
        return original_spec(name, path, *args, **kwargs)
    with patch.object(importlib.util, "spec_from_file_location", traced_spec):
        parser = load(PARSER, "m2226_parser_import_trace")
    checks["m2160_unconditionally_imported_in_production_parser"] = str(BASE) in imports

    class InMemoryMutation(importlib.abc.Loader):
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            raise RuntimeError("M2226_IN_MEMORY_UNPINNED_M2160_EXECUTED")
    def mutated_spec(name, path, *args, **kwargs):
        if Path(path).resolve() == BASE:
            return importlib.util.spec_from_loader(name, InMemoryMutation())
        return original_spec(name, path, *args, **kwargs)
    with patch.object(importlib.util, "spec_from_file_location", mutated_spec):
        runner.source_validation(False)
        try:
            load(PARSER, "m2226_mutated_parser_import")
        except RuntimeError as exc:
            checks["source_gate_passes_then_unpinned_module_executes"] = str(exc) == "M2226_IN_MEMORY_UNPINNED_M2160_EXECUTED"
        else:
            checks["source_gate_passes_then_unpinned_module_executes"] = False

    selector = load(runner.SELECTOR, "m2226_selector")
    selected = selector.select()
    frozen = json.loads(runner.SELECTION.read_text())
    checks["frozen_selected_rows_and_weights_reproduce_exactly"] = (
        selected["selections"] == frozen["selections"]
        and selected["aggregate_weights"] == frozen["aggregate_weights"]
        and selected["population"]["rows"] == frozen["population"]["rows"] == 2880)
    checks["fixed_tercile_weights"] = selected["aggregate_weights"] == {
        key: [1, 3] for key in ("low", "median", "high")}
    model = parser.sram_model()
    checks["sram_numbers_identical"] = (
        model["dynamic_read_energy_pj_per_accepted_bank_activation"] == 22.213
        and abs(model["leakage_power_mw"] - 3.826774326764422) < 1e-12
        and model["capacity_bytes"] == 294912 and model["macro_count"] == 16)
    checks["mixed_corner_explicit"] = (
        contract["mapping_and_power"]["dc_max_corner"] == "SSG0P9V125C"
        and contract["mapping_and_power"]["dc_min_corner"] == "FFG1P05VM40C"
        and contract["mapping_and_power"]["ptpx_corner"] == "TT0P9V25C"
        and contract["mapping_and_power"]["dc_to_ptpx_is_mixed_corner"]
        and model["leakage_proxy_corner_differs_from_logic_tt0p9v25c"])
    checks["serial_unchanged_budget"] = contract["execution_budget"] == {
        "license_queries": 1, "vcs_compiles": 2, "simv_runs": 6,
        "diagnostic_saif_files": 6, "measurement_saif_files": 6,
        "dc_runs": 2, "ptpx_runs": 6, "automatic_retry": False,
        "p1_serial": True, "reuse_m2203_raw": False}
    checks["m2227_namespace_unconsumed"] = not any(path.exists() for path in (
        runner.RESULT, runner.ATTEMPT, runner.LOCK))
    checks["old_attempt_not_input_or_output"] = (
        "results/m2219" not in RUNNER.read_text()
        and ".m2219_" not in RUNNER.read_text()
        and not any("m2219" in name for name in inventory))
    checks["all_investigative_checks_hold"] = all(checks.values())
    output = {
        "status": "REPRODUCED_P0_UNPINNED_TRANSITIVE_M2160_EXECUTION",
        "checks": checks,
        "checks_passed": sum(bool(value) for key, value in checks.items() if key != "all_investigative_checks_hold"),
        "checks_total": len(checks) - 1,
        "runtime_local_imports": imports,
        "unbound_dependency_sha256": digest(BASE),
        "source_files_modified": False,
        "eda_license_or_gpu_invoked": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if checks["all_investigative_checks_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

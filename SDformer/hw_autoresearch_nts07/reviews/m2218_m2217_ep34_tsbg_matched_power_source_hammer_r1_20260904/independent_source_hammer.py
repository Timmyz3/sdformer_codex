#!/opt/anaconda3/bin/python3
"""Independent, CPU-only M2218 source hammer.  Never invokes EDA or licenses."""
from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2217_ep34_tsbg_matched_power_source_contract_r1_20260904.json"
AUTHOR = HW / "reviews/m2217_ep34_tsbg_matched_power_source_author_receipt_r1_20260904"
M2204 = HW / "reviews/m2204_m2203_m2201_ordinary_native_saif_subtick_quantized_preflight_result_hammer_r1_20260904"
SELECTOR = HW / "system_simulator/scripts/select_m2217_ep34_tsbg_matched_power_windows.py"
PARSER = HW / "system_simulator/scripts/parse_m2217_ep34_tsbg_matched_power.py"
STRUCT = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
POWER = HW / "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py"
RUNNER = HW / "dc_handoff/scripts/run_m2217_ep34_tsbg_matched_power_one_shot.py"
TB = HW / "tb_m2018/tb_m2217_m2018_tsbg_matched_native_saif_power.sv"
UCLI = HW / "dc_handoff/scripts/m2217_m2018_single_dut_native_saif.ucli.tcl"
DC = HW / "dc_handoff/scripts/run_dc_m2217_m2018_matched_power_axis.tcl"
PT = HW / "dc_handoff/scripts/run_ptpx_m2217_m2018_matched_power_window.tcl"
SELECTION = HW / "tb_m2018/fixtures/m2217_ep34_tsbg_matched_power_windows.json"
MAPPING = HW / "reviews/tsmc28_sram_macro_audit_r1_20260827/tsmc28_sram_mapping_r1.json"
LOW_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
LOW_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
LOW_RESULT = HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/result.json"
HIGH_META = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json"
HIGH_RESULT = HW / "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_20260904/result.json"
TEST = HW / "tests/test_m2217_ep34_tsbg_matched_power_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            assert key not in out, (path, key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError((path, token))))


def verify_dir(root: Path) -> int:
    assert root.is_dir() and not root.is_symlink()
    assert not any(path.is_symlink() for path in root.rglob("*"))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, raw = line.split(maxsplit=1)
        rel = Path(raw.lstrip("*"))
        assert not rel.is_absolute() and ".." not in rel.parts
        path = root / rel
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
        assert rel.as_posix() not in rows
        rows[rel.as_posix()] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert actual == set(rows)
    return len(rows)


def canonical(meta: dict, population: str) -> tuple:
    return (meta["sequence"], int(meta["sample_id"]), int(meta["layer_id"]),
            int(meta["token_start"]), int(meta["slot"]), population)


def rebuild_selection() -> tuple[list[dict], dict]:
    low_meta = strict(LOW_META)["rows"]
    low_result = strict(LOW_RESULT)["rows"]
    high_meta = strict(HIGH_META)["rows"]
    high_result = strict(HIGH_RESULT)["rows"]
    assert len(low_meta) == len(low_result) == 1920
    assert len(high_meta) == len(high_result) == 960
    rows = []
    for population, metas, observed_rows in (
            ("m2051_g_le_48", low_meta, low_result),
            ("m2067_fc2_continuation", high_meta, high_result)):
        for meta, observed in zip(metas, observed_rows):
            assert int(meta["slot"]) == int(observed["workload_slot"])
            if population == "m2051_g_le_48":
                ordinary = int(meta["base_misses"]) * 12 * 8
                tsbg = int(meta["tsbg_misses"]) * 12 * 8
            else:
                ordinary = sum(int(row["ordinary_misses"])
                               for row in meta["chunk_rows"]) * 12 * 8
                tsbg = sum(int(row["tsbg_misses"])
                           for row in meta["chunk_rows"]) * 12 * 8
            assert 0 <= tsbg <= ordinary
            density = Fraction(ordinary - tsbg, ordinary) if ordinary else Fraction()
            rows.append((density, canonical(meta, population), population,
                         meta, observed, ordinary, tsbg))
    rows.sort(key=lambda row: (row[0], row[1]))
    assert len(rows) == 2880
    memh = LOW_MEMH.read_text().splitlines()
    assert len(memh) == 1920 * 192
    used = set()
    selected = []
    tie_counts = {}
    for index, label in enumerate(("low", "median", "high")):
        members = rows[index * 960:(index + 1) * 960]
        target = members[480][0]
        candidates = [row for row in members
                      if row[2] == "m2051_g_le_48" and row[5] > 0
                      and row[3]["sequence"] not in used]
        minimum = min(abs(row[0] - target) for row in candidates)
        nearest = [row for row in candidates if abs(row[0] - target) == minimum]
        tie_counts[label] = len(nearest)
        candidates.sort(key=lambda row: (abs(row[0] - target), -row[5], row[1]))
        density, _, _, meta, observed, ordinary, tsbg = candidates[0]
        used.add(meta["sequence"])
        begin = int(meta["slot"]) * 192
        descriptor = hashlib.sha256(
            ("\n".join(memh[begin:begin + 192]) + "\n").encode("ascii")
        ).hexdigest()
        selected.append({
            "stratum": label, "slot": int(meta["slot"]),
            "sample": int(meta["sample_id"]), "sequence": meta["sequence"],
            "layer": int(meta["layer_id"]), "density": [density.numerator, density.denominator],
            "descriptor_sha256": descriptor, "ordinary_requests": ordinary,
            "tsbg_requests": tsbg, "ordinary_cycles": int(observed["base_cycles"]),
            "tsbg_cycles": int(observed["tsbg_cycles"]),
        })
    return selected, {"population_rows": len(rows), "tie_counts": tie_counts,
                      "distinct_sequences": len(used)}


def invariant_errors(parts: dict[str, str], selection: dict) -> list[str]:
    errors = []
    tb, ucli, parser, runner, dc, pt = (parts[name] for name in
        ("tb", "ucli", "parser", "runner", "dc", "pt"))
    ucli_effective = "\n".join(line.strip() for line in ucli.splitlines()
                               if line.strip() and not line.lstrip().startswith("#"))
    if len(re.findall(r"m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend\s*#\s*\(", tb)) != 1:
        errors.append("single_dut")
    if ".SCHEDULE_MODE(SCHEDULE_MODE)" not in tb or "second_axis=0" not in tb:
        errors.append("single_axis")
    scope = "tb_m2217_m2018_tsbg_matched_native_saif_power.dut_axis"
    if ucli_effective.count("power " + scope) != 1 or ucli_effective.count("power -report") != 2:
        errors.append("saif_scope_roles")
    order = [ucli_effective.find(token) for token in ("power -enable", "\nrun\n", "power -disable",
        "M2217_PREHISTORY_SAIF_FILE", "power -reset", "action=measurement_enable",
        "action=second_run_returned", "M2217_MEASUREMENT_SAIF_FILE")]
    if any(value < 0 for value in order) or order != sorted(order):
        errors.append("saif_phase_order")
    if "need(tx_nonzero == 0" not in parser or "measurement record conservation" not in parser:
        errors.append("measurement_tx_conservation")
    if "len(records) == RECORDS" not in parser or "RECORDS = 93971" not in parser:
        errors.append("record_gate")
    if '"vcs_compiles": 2' not in runner or '"simv_runs": 6' not in runner \
            or '"dc_runs": 2' not in runner or '"ptpx_runs": 6' not in runner:
        errors.append("budget")
    if '"automatic_retry": False' not in runner or "shutil.rmtree(ATTEMPT" in runner:
        errors.append("one_shot")
    if "for axis, mode in AXES.items():" not in runner or "for stratum in STRATA:" not in runner:
        errors.append("six_points")
    if "saif_map -start" not in dc or "SCHEDULE_MODE=>$mode" not in dc or "compile_ultra" not in dc:
        errors.append("fresh_dc_map")
    if "read_saif -strip_path" not in pt or "ann_pct < 95.0" not in pt \
            or "leaf_pct < 95.0" not in pt or "toggle_pct < 20.0" not in pt:
        errors.append("pt_annotation")
    if "foreach cone {mem_req_valid mem_rsp_valid bridge_valid commit_valid" not in pt \
            or "mem_req_accept mem_rsp_accept bridge_accept commit_accept" not in pt:
        errors.append("pt_critical_cones")
    if selection.get("aggregate_weights") != {
            "low": [1, 3], "median": [1, 3], "high": [1, 3]}:
        errors.append("fixed_weights")
    sequences = [row.get("sequence") for row in selection.get("selections", [])]
    if len(sequences) != 3 or len(set(sequences)) != 3:
        errors.append("three_sequences")
    return errors


def main() -> int:
    contract = strict(CONTRACT)
    assert contract["status"] == "SOURCE_ONLY__M2218_REVIEW_REQUIRED__NO_EDA"
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(side) + ".seal.sha256")
    assert side.read_text().split() == [sha(CONTRACT), CONTRACT.name]
    assert outer.read_text().split() == [sha(side), side.name]
    author_members = verify_dir(AUTHOR)
    m2204_members = verify_dir(M2204)
    assert sha(DOC359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    for rel, digest in contract["source_inventory"].items():
        path = ROOT / rel
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, rel
    assert len(contract["source_inventory"]) == 26

    rebuilt, population = rebuild_selection()
    frozen = strict(SELECTION)
    frozen_key = [{"stratum": row["stratum"], "slot": row["global_slot"],
                   "sample": row["sample_id"], "sequence": row["sequence"],
                   "layer": row["layer_id"],
                   "density": row["selected_density_fraction"],
                   "descriptor_sha256": row["descriptor_text_sha256"],
                   "ordinary_requests": row["ordinary"]["accepted_bank_requests"],
                   "tsbg_requests": row["tsbg"]["accepted_bank_requests"],
                   "ordinary_cycles": row["ordinary"]["cycles"],
                   "tsbg_cycles": row["tsbg"]["cycles"]}
                  for row in frozen["selections"]]
    assert rebuilt == frozen_key

    parts = {"tb": TB.read_text(), "ucli": UCLI.read_text(),
             "parser": PARSER.read_text(), "runner": RUNNER.read_text(),
             "dc": DC.read_text(), "pt": PT.read_text()}
    assert not invariant_errors(parts, frozen)
    mutations = {
        "dual_dut": ({**parts, "tb": parts["tb"] +
            "\nm2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #() extra();\n"}, frozen),
        "axis_disconnect": ({**parts, "tb": parts["tb"].replace(
            ".SCHEDULE_MODE(SCHEDULE_MODE)", ".SCHEDULE_MODE(0)", 1)}, frozen),
        "scope_pollution": ({**parts, "ucli": parts["ucli"].replace(
            ".dut_axis", "", 1)}, frozen),
        "no_reset": ({**parts, "ucli": parts["ucli"].replace(
            "power -reset", "puts no_reset", 1)}, frozen),
        "measurement_as_diagnostic": ({**parts, "ucli": parts["ucli"].replace(
            "M2217_MEASUREMENT_SAIF_FILE", "M2217_PREHISTORY_SAIF_FILE")}, frozen),
        "remove_tx_gate": ({**parts, "parser": parts["parser"].replace(
            "need(tx_nonzero == 0", "need(tx_nonzero >= 0", 1)}, frozen),
        "remove_record_gate": ({**parts, "parser": parts["parser"].replace(
            "len(records) == RECORDS", "len(records) > 0", 1)}, frozen),
        "one_dc": ({**parts, "runner": parts["runner"].replace(
            '"dc_runs": 2', '"dc_runs": 1', 1)}, frozen),
        "retry": ({**parts, "runner": parts["runner"].replace(
            '"automatic_retry": False', '"automatic_retry": True')}, frozen),
        "no_saif_map": ({**parts, "dc": parts["dc"].replace(
            "saif_map -start", "# removed", 1)}, frozen),
        "weak_annotation": ({**parts, "pt": parts["pt"].replace(
            "ann_pct < 95.0", "ann_pct < 0.0", 1)}, frozen),
        "same_sequence": (parts, {**frozen, "selections": [
            {**row, "sequence": "one_sequence"} for row in frozen["selections"]]}),
    }
    for name, (mut_parts, mut_selection) in mutations.items():
        assert invariant_errors(mut_parts, mut_selection), name

    unit = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q", str(TEST)],
                          cwd=ROOT, text=True, capture_output=True, timeout=180)
    assert unit.returncode == 0, unit.stdout + unit.stderr
    static_runner = subprocess.run([sys.executable, "-B", str(RUNNER), "--static"],
        cwd=ROOT, text=True, capture_output=True, timeout=180)
    assert static_runner.returncode == 0 and "PASS_M2217_STATIC_RUNNER" in static_runner.stdout
    static_parser = subprocess.run([sys.executable, "-B", str(PARSER), "static"],
        cwd=ROOT, text=True, capture_output=True, timeout=180)
    assert static_parser.returncode == 0 and "PASS_M2217_STATIC_PARSER" in static_parser.stdout

    parser_tree = ast.parse(PARSER.read_text())
    imported_helpers = {STRUCT.relative_to(ROOT).as_posix(), POWER.relative_to(ROOT).as_posix()}
    inventory = set(contract["source_inventory"])
    missing_helpers = sorted(imported_helpers - inventory)
    assert missing_helpers == sorted(imported_helpers)
    assert "STRUCT = module(STRUCT_PATH" in PARSER.read_text()
    assert "POWER = module(POWER_PATH" in PARSER.read_text()
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "module" for node in ast.walk(parser_tree))

    fresh = [HW / "results/m2219_m2217_ep34_tsbg_matched_power_r1_20260904",
             HW / "results/.m2219_m2217_ep34_tsbg_matched_power_attempt_consumed",
             HW / "results/.m2219_m2217_ep34_tsbg_matched_power_launch_lock"]
    assert not any(path.exists() for path in fresh)
    mapping = strict(MAPPING)
    rows = [row for row in mapping["mappings"]
            if row["id"] == "C2_FC2_WEIGHT_BANKS_K1_K8_K1X8"]
    assert len(rows) == 1
    row = rows[0]
    assert row["macro_count"] == 16 and row["area_um2"] == 558507.032
    assert row["nominal_deep_segment_read_energy_pj_per_bank_request"] == 22.213

    result = {
        "status": "PASS_M2218_MECHANICAL_HAMMER__DECISIVE_P0_FOUND",
        "source_inventory_verified": "26/26",
        "author_seal_members": author_members,
        "m2204_seal_members": m2204_members,
        "selection": {"rebuilt_rows": population["population_rows"],
            "representatives": rebuilt, "distinct_sequences": population["distinct_sequences"],
            "nearest_tie_counts": population["tie_counts"],
            "selection_uses_measured_power_or_energy": False,
            "selection_uses_cycle_value_for_representative_choice": False},
        "matched_surface": {"axes": 2, "single_dut_per_compile": True,
            "strata": 3, "measurement_saif": 6, "diagnostic_saif": 6,
            "fresh_dc_maps": 2, "ptpx_points": 6,
            "same_ports_cache_clock_pvt": True},
        "activity_gates": {"dut_only": True, "records": 93971,
            "measurement_tx_zero": True, "measurement_conservation_exact": True,
            "duration_exact": True, "pt_annotation_net_leaf_min_percent": 95.0,
            "pt_nonzero_toggle_min_percent": 20.0, "critical_cones": 8},
        "sram_model": {"capacity_bytes_each_axis": 294912, "macro_count_each_axis": 16,
            "area_um2_each_axis": 558507.032,
            "dynamic_pj_per_actual_accepted_bank_activation": 22.213,
            "deep_segment_conservative": True,
            "leakage_mw_each_axis": 3.826774326764422,
            "mixed_corner_proxy_labeled": True},
        "tests": {"m2217_unit_tests": "9/9 PASS", "independent_mutations": "12/12 PASS",
            "static_runner": "PASS", "static_parser": "PASS"},
        "decisive_p0": {"name": "unpinned_transitive_parser_helpers",
            "missing_from_contract_source_inventory": missing_helpers,
            "production_imports_execute_these_helpers": True,
            "runner_source_validation_checks_them": False,
            "impact": "After M2218 sealing and before M2219, helper drift can change SAIF seal/conservation, transformation-map, annotation, critical-cone, and power arithmetic admission without tripping the 26/26 source gate."},
        "m2219_namespace_absent": True,
        "docs359_sha256": sha(DOC359),
        "execution": {"vcs": 0, "dc": 0, "ptpx": 0, "license_queries": 0,
            "gpu": 0, "git": 0}
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

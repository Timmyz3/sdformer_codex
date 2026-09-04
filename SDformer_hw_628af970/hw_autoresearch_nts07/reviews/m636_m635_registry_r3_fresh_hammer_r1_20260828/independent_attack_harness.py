#!/usr/bin/env python3
"""Independent CPU-only adversarial checks for the M635 registry r3.

This harness does not import the target unit tests and does not modify any
target.  It deliberately constructs its own evidence files under temporary,
repo-local directories because the registry refuses evidence outside the repo.
"""

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m635_h67_paper_metric_registry_r3.py"
CONFIG = HW_ROOT / "system_simulator/config/m635_h67_paper_metric_registry_r3_20260828.json"
REQUEST = HW_ROOT / "reviews/m636_m635_registry_r3_fresh_hammer_r1_REQUEST_20260828/request.json"

spec = importlib.util.spec_from_file_location("m636_target", str(BUILDER))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump_temp_json(value):
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
    with handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    return Path(handle.name)


def repo_json(directory, name, value):
    path = Path(directory) / name
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": "application/json",
    }


def run_mutation(base, mutate):
    value = copy.deepcopy(base)
    mutate(value)
    path = dump_temp_json(value)
    try:
        result = module.build(path)
        return {"accepted": True, "result": result}
    except module.RegistryError as exc:
        return {"accepted": False, "error": str(exc)}
    finally:
        path.unlink()


def make_complete_bundle(base, aggregate_override=None):
    """Build a self-consistent but entirely synthetic five-file bundle."""
    results_root = HW_ROOT / "results"
    reviews_root = HW_ROOT / "reviews"
    simulator_root = HW_ROOT / "system_simulator"
    results_temp = tempfile.TemporaryDirectory(dir=str(results_root))
    reviews_temp = tempfile.TemporaryDirectory(dir=str(reviews_root))
    simulator_temp = tempfile.TemporaryDirectory(dir=str(simulator_root))
    try:
        population_id = "m636_fake_population"
        workload_id = "m636_fake_workload"
        bundle_id = "direct_unified_m636_fully_fabricated"
        ordered = [item[0] for item in module.MANDATORY_ROW_SPECS]
        role_map = {item[0]: (item[1], item[2]) for item in module.MANDATORY_ROW_SPECS}
        totals = {
            "dense96_fixed_t10": 1200.0,
            "ptb_like_structured": 960.0,
            "exact_bit_k1": 840.0,
            "exact_bit_k1x8": 720.0,
            "exact_typed_k8": 660.0,
            "ours_exact": 600.0,
        }
        resource = {
            "schema": "m635.h67.common_resource_manifest.r1",
            "status": "FROZEN",
            "population_id": population_id,
            "workload_id": workload_id,
            # Deliberately shallow, author-declared and free of configuration,
            # simulator, checkpoint, trace, charge or fallback identities.
            "resource_tuple": {"author_declared": "not_measurement_rooted"},
        }
        resource_spec = repo_json(simulator_temp.name, "resource.json", resource)
        rows = []
        for index, row_id in enumerate(ordered):
            rows.append({
                "row_id": row_id,
                "role": role_map[row_id][0],
                "fidelity": role_map[row_id][1],
                "cycles": totals[row_id],
                "energy_mj": 100.0 + index,
                "area_mm2": 10.0 + index,
                "accuracy": 1.0 + index,
            })
        samples = []
        for view in module.VIEW_NAMES:
            for index, (sequence, weight, stratum) in enumerate((
                ("fake_seq_a", 0.2, "low"),
                ("fake_seq_b", 0.3, "mid"),
                ("fake_seq_c", 0.5, "high"),
            )):
                samples.append({
                    "sample_id": "%s_fake_%d" % (view, index),
                    "sequence_id": sequence,
                    "density_stratum": stratum,
                    "view": view,
                    "row_cycles": {row_id: totals[row_id] * weight for row_id in ordered},
                })
        aggregates = {
            "arithmetic_mean": 2.0,
            "geometric_mean": 2.0,
            "ratio_of_sums": 2.0,
            "minimum": 2.0,
            "maximum": 2.0,
        }
        if aggregate_override is not None:
            aggregates["geometric_mean"] = aggregate_override
        views = {"iso_lane": copy.deepcopy(aggregates), "iso_service": copy.deepcopy(aggregates)}
        result = {
            "schema": "m635.h67.direct_unified.result.r1",
            "status": "PASS_COMPLETE",
            "measurement_class": "DIRECT_UNIFIED_CYCLE_SIM",
            "population_id": population_id,
            "workload_id": workload_id,
            "resource_manifest_sha256": resource_spec["sha256"],
            "rows": rows,
            "samples": samples,
            "aggregates": aggregates,
            "views": views,
        }
        result_spec = repo_json(results_temp.name, "result.json", result)
        completion = {
            "schema": "m635.h67.direct_unified.completion_receipt.r1",
            "status": "PASS_COMPLETE",
            "direct_result_sha256": result_spec["sha256"],
            "resource_manifest_sha256": resource_spec["sha256"],
            "population_id": population_id,
            "workload_id": workload_id,
            "completed_row_ids": ordered,
            "closures": {key: True for key in module.REQUIRED_CLOSURES},
        }
        completion_spec = repo_json(results_temp.name, "completion.json", completion)
        sequence_specs = []
        for sequence in ("fake_seq_a", "fake_seq_b", "fake_seq_c"):
            sequence_specs.append(repo_json(results_temp.name, sequence + ".json", {
                "schema": "m635.h67.sequence_completion_receipt.r1",
                "status": "PASS",
                "sequence_id": sequence,
                "population_id": population_id,
                "workload_id": workload_id,
                "direct_result_sha256": result_spec["sha256"],
            }))
        coverage = {
            "schema": "m635.h67.coverage_receipt.r1",
            "status": "PASS",
            "direct_result_sha256": result_spec["sha256"],
            "population_id": population_id,
            "workload_id": workload_id,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "sequence_receipts": sequence_specs,
            "density_preregistration_receipt": None,
            "aggregates": aggregates,
            "views": views,
        }
        coverage_spec = repo_json(results_temp.name, "coverage.json", coverage)
        hammer = {
            "schema": "m635.h67.direct_unified.independent_hammer.r1",
            "status": "PASS",
            "direct_result_sha256": result_spec["sha256"],
            "completion_receipt_sha256": completion_spec["sha256"],
            "coverage_receipt_sha256": coverage_spec["sha256"],
            "resource_manifest_sha256": resource_spec["sha256"],
            "severity_counts": {"P0": 0, "P1": 0},
            "independence": {
                "author_receipt_used_as_authority": False,
                "raw_evidence_recomputed": True,
                "result_modified": False,
            },
            "recomputed_rows": rows,
            "recomputed_aggregates": aggregates,
            "recomputed_views": views,
            "authorization": {
                "table_a_methodology_admitted": True,
                "direct_unified_measurement_admitted": True,
            },
        }
        hammer_spec = repo_json(reviews_temp.name, "author_faked_hammer.json", hammer)
        value = copy.deepcopy(base)
        value["table_a_evidence_bundles"][bundle_id] = {
            "direct_result": result_spec,
            "completion_receipt": completion_spec,
            "resource_manifest": resource_spec,
            "coverage_receipt": coverage_spec,
            "independent_hammer_receipt": hammer_spec,
        }
        for row in value["table_a_schema"]["rows"][:6]:
            projected = next(item for item in rows if item["row_id"] == row["row_id"])
            row.update({
                "cycles": projected["cycles"],
                "energy_mj": projected["energy_mj"],
                "area_mm2": projected["area_mm2"],
                "accuracy": projected["accuracy"],
                "source_id": bundle_id,
                "measurement_class": "DIRECT_UNIFIED_CYCLE_SIM",
                "population_id": population_id,
                "workload_id": workload_id,
                "resource_manifest_sha256": resource_spec["sha256"],
                "completion_receipt_sha256": completion_spec["sha256"],
                "decoder_complete": True,
                "memory_timing_included": True,
                "full_network_completion": True,
                "logic_sram_dram_energy_closed": True,
                "logic_macro_area_closed": True,
                "sta_closed": True,
                "independent_hammer_pass": True,
                "blockers": [],
            })
        value["claim_boundary"]["table_a_admitted_rows"] = 6
        value["claim_boundary"]["paper_headline_admitted"] = True
        config_path = dump_temp_json(value)
        try:
            built = module.build(config_path)
            return {
                "accepted": True,
                "bundle_count": built["table_a_evidence_bundle_count"],
                "eligible_rows": built["headline_gate"]["eligible_row_count"],
                "headline": built["headline_gate"]["admitted"],
                "direct_speedup": built["headline_gate"]["direct_speedup"],
                "resource_tuple": resource["resource_tuple"],
            }
        except module.RegistryError as exc:
            return {"accepted": False, "error": str(exc)}
        finally:
            config_path.unlink()
    finally:
        simulator_temp.cleanup()
        reviews_temp.cleanup()
        results_temp.cleanup()


def strict_source_attack(base, name, payload):
    tests_root = HW_ROOT / "system_simulator/tests"
    with tempfile.TemporaryDirectory(dir=str(tests_root)) as directory:
        path = Path(directory) / (name + ".json")
        path.write_bytes(payload)
        value = copy.deepcopy(base)
        value["sources"][name] = {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "media_type": "application/json",
        }
        config_path = dump_temp_json(value)
        try:
            module.build(config_path)
            return {"accepted": True}
        except module.RegistryError as exc:
            return {"accepted": False, "error": str(exc)}
        finally:
            config_path.unlink()


def main():
    base = module.load_json(CONFIG, "M636 canonical")
    request = module.load_json(REQUEST, "M636 request")
    canonical = module.build(CONFIG)
    checks = {}
    checks["mandatory_delete"] = run_mutation(base, lambda value: (
        value["table_a_schema"]["required_row_ids"].remove("exact_bit_k1x8"),
        value["table_a_schema"].__setitem__("rows", [
            row for row in value["table_a_schema"]["rows"] if row["row_id"] != "exact_bit_k1x8"
        ]),
    ))
    checks["mandatory_rename"] = run_mutation(base, lambda value: (
        value["table_a_schema"]["required_row_ids"].__setitem__(3, "renamed_k1x8"),
        value["table_a_schema"]["rows"][3].__setitem__("row_id", "renamed_k1x8"),
    ))
    checks["role_mutation"] = run_mutation(base, lambda value: value["table_a_schema"]["rows"][0].__setitem__("role", "candidate"))
    checks["fidelity_mutation"] = run_mutation(base, lambda value: value["table_a_schema"]["rows"][5].__setitem__("fidelity", "lossy"))
    checks["anchor_mutation"] = run_mutation(base, lambda value: value["headline_policy"].__setitem__("strongest_same_page_baseline_row_id", "exact_bit_k1"))

    def promote_m618(value):
        for index, row in enumerate(value["table_a_schema"]["rows"][:6]):
            row.update({
                "cycles": 1000.0 if index == 0 else 200.0,
                "energy_mj": 1.0,
                "area_mm2": 1.0,
                "accuracy": 1.0,
                "source_id": "m618",
                "measurement_class": "DIRECT_UNIFIED_CYCLE_SIM",
                "population_id": "fake_population",
                "workload_id": "fake_workload",
                "resource_manifest_sha256": "1" * 64,
                "completion_receipt_sha256": "2" * 64,
                "decoder_complete": True,
                "memory_timing_included": True,
                "full_network_completion": True,
                "logic_sram_dram_energy_closed": True,
                "logic_macro_area_closed": True,
                "sta_closed": True,
                "independent_hammer_pass": True,
                "blockers": [],
            })
    checks["external_m618_promotion"] = run_mutation(base, promote_m618)
    checks["table_b_headline_promotion"] = run_mutation(base, lambda value: value["table_b_schema"]["rows"][0].__setitem__("headline_eligible", True))
    checks["table_c_ours_promotion"] = run_mutation(base, lambda value: value["table_c_schema"]["rows"][0].__setitem__("ours", True))
    checks["fake_bundle_missing_four_artifacts"] = run_mutation(base, lambda value: value["table_a_evidence_bundles"].__setitem__(
        "direct_unified_fake", {"direct_result": value["sources"]["m618"]}
    ))
    checks["duplicate_key_source"] = strict_source_attack(base, "m636_duplicate", b'{"metric":1,"metric":2}')
    checks["nan_source"] = strict_source_attack(base, "m636_nan", b'{"metric":NaN}')
    checks["infinity_source"] = strict_source_attack(base, "m636_inf", b'{"metric":Infinity}')
    checks["overflow_source"] = strict_source_attack(base, "m636_overflow", b'{"metric":1e999}')
    checks["full_fake_five_file_bundle"] = make_complete_bundle(base)
    checks["coordinated_bad_aggregate_across_three_files"] = make_complete_bundle(base, aggregate_override=3.0)

    source_checks = {}
    for source_id, source_spec in base["sources"].items():
        source_path = module.secure_repo_file(source_spec["path"])
        strict_doc = module.load_json(source_path, "M636 source " + source_id)
        source_checks[source_id] = {
            "sha_match": digest(source_path) == source_spec["sha256"],
            "strict_root_object": isinstance(strict_doc, dict),
            "media_type": source_spec["media_type"],
        }
    target_actual = {
        "builder": digest(BUILDER),
        "config": digest(CONFIG),
        "tests": digest(HW_ROOT / "system_simulator/tests/test_m635_h67_paper_metric_registry_r3.py"),
        "contract": digest(HW_ROOT / "contracts/m635_h67_paper_metric_registry_r3_contract_r1_20260828.json"),
        "docs359": digest(HW_ROOT / "docs/359_DATE终局冻结_20260813.md"),
    }
    target_expected = {
        "builder": request["targets"]["builder"]["sha256"],
        "config": request["targets"]["config"]["sha256"],
        "tests": request["targets"]["tests"]["sha256"],
        "contract": request["targets"]["contract"]["sha256"],
        "docs359": request["targets"]["docs359"]["sha256"],
    }
    output = {
        "schema": "m636.m635.registry_r3.independent_attack_harness.r1",
        "target_identity": {
            key: {"expected": target_expected[key], "actual": target_actual[key], "match": target_expected[key] == target_actual[key]}
            for key in target_actual
        },
        "code_constants": {
            "mandatory_row_specs": list(module.MANDATORY_ROW_SPECS),
            "optional_row_specs": list(module.OPTIONAL_ROW_SPECS),
            "fixed_numerator": module.FIXED_NUMERATOR_ROW_ID,
            "strongest_baseline": module.STRONGEST_BASELINE_ROW_ID,
            "candidate": module.CANDIDATE_ROW_ID,
        },
        "canonical": {
            "sources": len(canonical["source_hashes_validated"]),
            "bundles": canonical["table_a_evidence_bundle_count"],
            "eligible_rows": canonical["headline_gate"]["eligible_row_count"],
            "headline": canonical["headline_gate"]["admitted"],
            "analytical": canonical["analytical_diagnostic"]["admitted"],
        },
        "strict_source_checks": source_checks,
        "attacks": checks,
        "m518": {
            "bound_sha256": canonical["source_hashes_validated"]["m518"],
            "table_b_value": [row["value"] for row in canonical["table_b"] if row["metric_id"] == "m518_fixed_t10_directed_issue_cycle_anchor"][0],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

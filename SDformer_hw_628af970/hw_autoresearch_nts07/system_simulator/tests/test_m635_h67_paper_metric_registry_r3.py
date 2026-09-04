#!/usr/bin/env python3

import copy
import contextlib
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m635_h67_paper_metric_registry_r3.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m635_h67_paper_metric_registry_r3_20260828.json"
SPEC = importlib.util.spec_from_file_location("m635_registry", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class M635RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = MODULE.load_json(CONFIG, "test config")

    @staticmethod
    def remove_file(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def write_config(self, obj):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        with handle:
            json.dump(obj, handle, ensure_ascii=False, allow_nan=False)
        path = Path(handle.name)
        self.addCleanup(self.remove_file, path)
        return path

    @staticmethod
    def write_repo_json(directory, name, obj):
        path = Path(directory) / name
        payload = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        path.write_bytes(payload)
        return {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "media_type": "application/json",
        }

    @contextlib.contextmanager
    def synthetic_bundle_config(self, tamper_aggregate=False):
        results_root = REPO_ROOT / "hw_autoresearch_nts07/results"
        reviews_root = REPO_ROOT / "hw_autoresearch_nts07/reviews"
        simulator_root = REPO_ROOT / "hw_autoresearch_nts07/system_simulator"
        with tempfile.TemporaryDirectory(dir=str(results_root)) as results_dir, \
                tempfile.TemporaryDirectory(dir=str(reviews_root)) as reviews_dir, \
                tempfile.TemporaryDirectory(dir=str(simulator_root)) as resource_dir:
            population_id = "synthetic_population"
            workload_id = "synthetic_workload"
            bundle_id = "direct_unified_synthetic"
            resource = {
                "schema": "m635.h67.common_resource_manifest.r1",
                "status": "FROZEN",
                "population_id": population_id,
                "workload_id": workload_id,
                "resource_tuple": {"lanes": 96, "sram_bytes": 245760, "clock_ns": 3.0},
            }
            resource_spec = self.write_repo_json(resource_dir, "resource.json", resource)
            ordered_ids = [item[0] for item in MODULE.MANDATORY_ROW_SPECS]
            totals = {
                "dense96_fixed_t10": 1000,
                "ptb_like_structured": 800,
                "exact_bit_k1": 700,
                "exact_bit_k1x8": 600,
                "exact_typed_k8": 550,
                "ours_exact": 500,
            }
            rows = []
            specs = {item[0]: (item[1], item[2]) for item in MODULE.MANDATORY_ROW_SPECS}
            for row_id in ordered_ids:
                rows.append({
                    "row_id": row_id,
                    "role": specs[row_id][0],
                    "fidelity": specs[row_id][1],
                    "cycles": totals[row_id],
                    "energy_mj": 1.0,
                    "area_mm2": 1.0,
                    "accuracy": 1.0,
                })
            samples = []
            weights = (0.3, 0.3, 0.4)
            sequences = ("seq_a", "seq_b", "seq_c")
            for view in MODULE.VIEW_NAMES:
                for index, (sequence_id, weight) in enumerate(zip(sequences, weights)):
                    samples.append({
                        "sample_id": "%s_%d" % (view, index),
                        "sequence_id": sequence_id,
                        "density_stratum": ("low", "mid", "high")[index],
                        "view": view,
                        "row_cycles": {row_id: totals[row_id] * weight for row_id in ordered_ids},
                    })
            aggregates = {
                "arithmetic_mean": 2.0,
                "geometric_mean": 2.0,
                "ratio_of_sums": 2.0,
                "minimum": 2.0,
                "maximum": 2.0,
            }
            if tamper_aggregate:
                aggregates["geometric_mean"] = 3.0
            views = {
                "iso_lane": copy.deepcopy(aggregates),
                "iso_service": copy.deepcopy(aggregates),
            }
            result_doc = {
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
            result_spec = self.write_repo_json(results_dir, "result.json", result_doc)
            completion_doc = {
                "schema": "m635.h67.direct_unified.completion_receipt.r1",
                "status": "PASS_COMPLETE",
                "direct_result_sha256": result_spec["sha256"],
                "resource_manifest_sha256": resource_spec["sha256"],
                "population_id": population_id,
                "workload_id": workload_id,
                "completed_row_ids": ordered_ids,
                "closures": {name: True for name in MODULE.REQUIRED_CLOSURES},
            }
            completion_spec = self.write_repo_json(results_dir, "completion.json", completion_doc)
            sequence_specs = []
            for sequence_id in sequences:
                sequence_doc = {
                    "schema": "m635.h67.sequence_completion_receipt.r1",
                    "status": "PASS",
                    "sequence_id": sequence_id,
                    "population_id": population_id,
                    "workload_id": workload_id,
                    "direct_result_sha256": result_spec["sha256"],
                }
                sequence_specs.append(self.write_repo_json(results_dir, "%s.json" % sequence_id, sequence_doc))
            coverage_doc = {
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
            coverage_spec = self.write_repo_json(results_dir, "coverage.json", coverage_doc)
            hammer_doc = {
                "schema": "m635.h67.direct_unified.independent_hammer.r1",
                "status": "PASS",
                "direct_result_sha256": result_spec["sha256"],
                "completion_receipt_sha256": completion_spec["sha256"],
                "coverage_receipt_sha256": coverage_spec["sha256"],
                "resource_manifest_sha256": resource_spec["sha256"],
                "severity_counts": {"P0": 0, "P1": 0},
                "independence": {"author_receipt_used_as_authority": False, "raw_evidence_recomputed": True, "result_modified": False},
                "recomputed_rows": rows,
                "recomputed_aggregates": aggregates,
                "recomputed_views": views,
                "authorization": {"table_a_methodology_admitted": True, "direct_unified_measurement_admitted": True},
            }
            hammer_spec = self.write_repo_json(reviews_dir, "hammer.json", hammer_doc)
            obj = copy.deepcopy(self.base)
            obj["table_a_evidence_bundles"][bundle_id] = {
                "direct_result": result_spec,
                "completion_receipt": completion_spec,
                "resource_manifest": resource_spec,
                "coverage_receipt": coverage_spec,
                "independent_hammer_receipt": hammer_spec,
            }
            for row in obj["table_a_schema"]["rows"][:6]:
                evidence = next(item for item in rows if item["row_id"] == row["row_id"])
                row.update({
                    "cycles": evidence["cycles"],
                    "energy_mj": evidence["energy_mj"],
                    "area_mm2": evidence["area_mm2"],
                    "accuracy": evidence["accuracy"],
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
            obj["claim_boundary"]["table_a_admitted_rows"] = 6
            obj["claim_boundary"]["paper_headline_admitted"] = True
            yield self.write_config(obj)

    def test_01_canonical_passes_with_zero_eligible_and_no_headline(self):
        result = MODULE.build(CONFIG)
        self.assertEqual(12, len(result["source_hashes_validated"]))
        self.assertEqual(0, result["table_a_evidence_bundle_count"])
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["analytical_diagnostic"]["admitted"])

    def test_02_coordinated_mandatory_row_deletion_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["table_a_schema"]["rows"] = [
            row for row in obj["table_a_schema"]["rows"] if row["row_id"] != "exact_bit_k1x8"
        ]
        obj["table_a_schema"]["required_row_ids"].remove("exact_bit_k1x8")
        with self.assertRaisesRegex(MODULE.RegistryError, "code-level ladder"):
            MODULE.build(self.write_config(obj))

    def test_03_coordinated_mandatory_row_rename_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["table_a_schema"]["required_row_ids"][3] = "renamed_k1x8"
        obj["table_a_schema"]["rows"][3]["row_id"] = "renamed_k1x8"
        with self.assertRaisesRegex(MODULE.RegistryError, "code-level ladder"):
            MODULE.build(self.write_config(obj))

    def test_04_role_and_fidelity_are_code_fixed(self):
        obj = copy.deepcopy(self.base)
        obj["table_a_schema"]["rows"][0]["role"] = "candidate"
        with self.assertRaisesRegex(MODULE.RegistryError, "role/fidelity mutation"):
            MODULE.build(self.write_config(obj))

    def test_05_headline_anchors_are_code_fixed(self):
        obj = copy.deepcopy(self.base)
        obj["headline_policy"]["strongest_same_page_baseline_row_id"] = "exact_bit_k1"
        with self.assertRaisesRegex(MODULE.RegistryError, "code-level anchors"):
            MODULE.build(self.write_config(obj))

    def test_06_external_m618_coordinated_promotion_fails_closed(self):
        obj = copy.deepcopy(self.base)
        for index, row in enumerate(obj["table_a_schema"]["rows"][:6]):
            row.update({
                "cycles": 1000 if index == 0 else 200,
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
        result = MODULE.build(self.write_config(obj))
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        for row_id, reasons in result["headline_gate"]["row_failures"].items():
            if row_id != "ours_lossy":
                self.assertIn("table_b_or_c_source_structurally_forbidden", reasons)

    def test_07_fake_direct_bundle_name_without_five_artifacts_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["table_a_evidence_bundles"]["direct_unified_fake"] = {
            "direct_result": obj["sources"]["m618"]
        }
        with self.assertRaisesRegex(MODULE.RegistryError, "exactly five dedicated artifacts"):
            MODULE.build(self.write_config(obj))

    def test_08_sha_bound_duplicate_key_evidence_source_is_rejected(self):
        tests_dir = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        with tempfile.TemporaryDirectory(dir=str(tests_dir)) as temp_dir:
            evidence = Path(temp_dir) / "duplicate.json"
            payload = b'{"metric":1,"metric":2}'
            evidence.write_bytes(payload)
            obj = copy.deepcopy(self.base)
            obj["sources"]["coordinated_duplicate_attack"] = {
                "path": evidence.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": "application/json",
            }
            with self.assertRaisesRegex(MODULE.RegistryError, "duplicate JSON key"):
                MODULE.build(self.write_config(obj))

    def test_09_sha_bound_nonfinite_evidence_source_is_rejected(self):
        tests_dir = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        with tempfile.TemporaryDirectory(dir=str(tests_dir)) as temp_dir:
            evidence = Path(temp_dir) / "nonfinite.json"
            payload = b'{"metric":1e999}'
            evidence.write_bytes(payload)
            obj = copy.deepcopy(self.base)
            obj["sources"]["coordinated_nonfinite_attack"] = {
                "path": evidence.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": "application/json",
            }
            with self.assertRaisesRegex(MODULE.RegistryError, "non-finite JSON number"):
                MODULE.build(self.write_config(obj))

    def test_10_non_json_media_type_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["sources"]["m528"]["media_type"] = "text/plain"
        with self.assertRaisesRegex(MODULE.RegistryError, "requires application/json"):
            MODULE.build(self.write_config(obj))

    def test_11_m518_is_bound_to_actual_post_run_hammer(self):
        result = MODULE.build(CONFIG)
        self.assertEqual(
            "513c5d916859b0f48b9ffeced6853ad89a8ace5ea6a9b264baf05d1ed1966665",
            result["source_hashes_validated"]["m518"],
        )
        m518_rows = [row for row in result["table_b"] if row["metric_id"].startswith("m518_")]
        self.assertEqual(1, len(m518_rows))
        self.assertEqual(17, m518_rows[0]["value"])

    def test_12_m518_static_hammer_rebinding_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["sources"]["m518"] = {
            "path": "hw_autoresearch_nts07/reviews/m518_candidate_static_hammer_r11_independent_20260827/m518_candidate_static_hammer_r11_independent_20260827.json",
            "sha256": "13ddb58395083412338a6b314dc1c2c3b5c798a4305624d9d21a3dd13a4ce687",
            "media_type": "application/json",
        }
        with self.assertRaisesRegex(MODULE.RegistryError, "post-run receipt hammer SHA"):
            MODULE.build(self.write_config(obj))

    def test_13_symlink_source_component_is_rejected(self):
        tests_dir = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        with tempfile.TemporaryDirectory(dir=str(tests_dir)) as temp_dir:
            root = Path(temp_dir)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(MODULE.RegistryError, "symlink path component refused"):
                MODULE.secure_repo_file(link.relative_to(REPO_ROOT).as_posix())

    def test_14_claim_count_and_headline_are_executable_not_descriptive(self):
        for field, value in (("table_a_admitted_rows", 1), ("paper_headline_admitted", True)):
            obj = copy.deepcopy(self.base)
            obj["claim_boundary"][field] = value
            with self.assertRaisesRegex(MODULE.RegistryError, "disagrees with evidence gate"):
                MODULE.build(self.write_config(obj))

    def test_15_complete_five_artifact_bundle_has_a_reachable_admission_path(self):
        with self.synthetic_bundle_config() as config_path:
            result = MODULE.build(config_path)
        self.assertEqual(1, result["table_a_evidence_bundle_count"])
        self.assertEqual(6, result["headline_gate"]["eligible_row_count"])
        self.assertTrue(result["headline_gate"]["admitted"])
        self.assertEqual(2.0, result["headline_gate"]["direct_speedup"])

    def test_16_aggregate_cannot_be_coordinated_across_result_coverage_and_hammer(self):
        with self.synthetic_bundle_config(tamper_aggregate=True) as config_path:
            with self.assertRaisesRegex(MODULE.RegistryError, "does not recompute from raw samples"):
                MODULE.build(config_path)


if __name__ == "__main__":
    unittest.main()

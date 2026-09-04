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
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m653_h67_paper_metric_registry_r5.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m653_h67_paper_metric_registry_r5_20260828.json"
R4_TESTS = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests/test_m645_h67_paper_metric_registry_r4.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load("m653_registry", SCRIPT)
T4 = _load("m653_fixture_source_r4", R4_TESTS)


class M653RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = M.M635.load_json(CONFIG, "M653 test config")
        T4.M645RegistryTests.setUpClass()

    @staticmethod
    def _remove(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _config(self, value):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        with handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        path = Path(handle.name)
        self.addCleanup(self._remove, path)
        return path

    @staticmethod
    def _load_spec(spec):
        return json.loads((REPO_ROOT / spec["path"]).read_text(encoding="utf-8"))

    @staticmethod
    def _rewrite_json(spec, value):
        path = REPO_ROOT / spec["path"]
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                        encoding="utf-8")
        spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return spec

    @staticmethod
    def _rewrite_text(spec, value):
        path = REPO_ROOT / spec["path"]
        path.write_text(value, encoding="utf-8")
        spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return spec

    @staticmethod
    def _spec(path, media_type):
        return {"path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": media_type}

    def _seal_one(self, directory, document_spec):
        manifest_path = Path(directory) / "REQUEST_SHA256SUMS"
        member_name = Path(document_spec["path"]).name
        manifest_path.write_text("%s  %s\n" % (document_spec["sha256"], member_name), encoding="utf-8")
        manifest = self._spec(manifest_path, "text/plain")
        outer_path = Path(directory) / "REQUEST_SHA256SUMS.seal.sha256"
        outer_path.write_text("%s  %s\n" % (manifest["sha256"], manifest_path.name), encoding="utf-8")
        return manifest, self._spec(outer_path, "text/plain")

    def _seal_two(self, directory, first, second):
        manifest_path = Path(directory) / "SHA256SUMS"
        rows = sorted([(Path(first["path"]).name, first["sha256"]),
                       (Path(second["path"]).name, second["sha256"])])
        manifest_path.write_text("".join("%s  %s\n" % (digest, name) for name, digest in rows),
                                 encoding="utf-8")
        manifest = self._spec(manifest_path, "text/plain")
        outer_path = Path(directory) / "SHA256SUMS.seal.sha256"
        outer_path.write_text("%s  %s\n" % (manifest["sha256"], manifest_path.name), encoding="utf-8")
        return manifest, self._spec(outer_path, "text/plain")

    @contextlib.contextmanager
    def _rooted_positive_fixture(self):
        helper = T4.M645RegistryTests(methodName="test_01_canonical_is_zero_bundle_zero_authority_zero_headline")
        helper.base = T4.M645RegistryTests.base
        with helper._rooted_positive_fixture() as (_, old_bundle, _):
            bundle = copy.deepcopy(old_bundle)
            bundle_id = bundle["bundle_id"]

            population = self._load_spec(self._load_spec(bundle["measurement_identity"])["sequence_population_manifest"])
            for sample in population["samples"]:
                if sample["density_stratum"] == "mid":
                    sample["density_stratum"] = "medium"
            measurement_doc = self._load_spec(bundle["measurement_identity"])
            population_spec = measurement_doc["sequence_population_manifest"]
            self._rewrite_json(population_spec, population)
            measurement_doc["sequence_population_manifest"] = population_spec
            self._rewrite_json(bundle["measurement_identity"], measurement_doc)

            common = self._load_spec(bundle["common_resource_manifest"])
            common["measurement_identity_sha256"] = bundle["measurement_identity"]["sha256"]
            self._rewrite_json(bundle["common_resource_manifest"], common)

            for row_id, spec in bundle["configuration_manifests"].items():
                doc = self._load_spec(spec)
                doc["sequence_population_manifest_sha256"] = population_spec["sha256"]
                doc["common_resource_manifest_sha256"] = bundle["common_resource_manifest"]["sha256"]
                self._rewrite_json(spec, doc)

            raw_index = self._load_spec(bundle["raw_run_index"])
            run_specs = []
            for item in raw_index["runs"]:
                log = self._load_spec(item["log"])
                if log["density_stratum"] == "mid":
                    log["density_stratum"] = "medium"
                log["configuration_manifest_sha256"] = bundle["configuration_manifests"][log["row_id"]]["sha256"]
                log["sequence_population_manifest_sha256"] = population_spec["sha256"]
                self._rewrite_json(item["log"], log)
                run_specs.append(item["log"])
            raw_index["measurement_identity_sha256"] = bundle["measurement_identity"]["sha256"]
            self._rewrite_json(bundle["raw_run_index"], raw_index)

            energy = self._load_spec(bundle["energy_receipt"])
            energy["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            for row in energy["rows"]:
                row["configuration_manifest_sha256"] = bundle["configuration_manifests"][row["row_id"]]["sha256"]
            self._rewrite_json(bundle["energy_receipt"], energy)

            accuracy = self._load_spec(bundle["accuracy_receipt"])
            accuracy["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            accuracy["population_manifest_sha256"] = population_spec["sha256"]
            for row in accuracy["rows"]:
                row["configuration_manifest_sha256"] = bundle["configuration_manifests"][row["row_id"]]["sha256"]
            self._rewrite_json(bundle["accuracy_receipt"], accuracy)

            ppa = self._load_spec(bundle["ppa_receipt"])
            ppa["schema"] = "m653.h67.logic_macro_sta_power_ppa_receipt.r1"
            ppa["status"] = "PASS_RAW_PROJECTED"
            for row in ppa["rows"]:
                row["configuration_manifest_sha256"] = bundle["configuration_manifests"][row["row_id"]]["sha256"]
                self._rewrite_text(row["logic_report"], "logic_area_mm2 0.6\nlogic_power_mw 0.2\n")
                self._rewrite_text(row["sram_report"], "sram_macro_area_mm2 0.4\nsram_macro_power_mw 0.1\n")
                self._rewrite_text(row["sta_report"], "setup_wns_ns 0.0\nhold_wns_ns 0.0\n")
                row["logic_power_mw"] = 0.2
                row["sram_macro_power_mw"] = 0.1
                row["total_power_mw"] = 0.3
            self._rewrite_json(bundle["ppa_receipt"], ppa)

            result = self._load_spec(bundle["direct_result"])
            result["common_resource_manifest_sha256"] = bundle["common_resource_manifest"]["sha256"]
            result["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            result["ppa_receipt_sha256"] = bundle["ppa_receipt"]["sha256"]
            result["energy_receipt_sha256"] = bundle["energy_receipt"]["sha256"]
            result["accuracy_receipt_sha256"] = bundle["accuracy_receipt"]["sha256"]
            for sample in result["samples"]:
                if sample["density_stratum"] == "mid":
                    sample["density_stratum"] = "medium"
            self._rewrite_json(bundle["direct_result"], result)

            completion = self._load_spec(bundle["completion_receipt"])
            completion["direct_result_sha256"] = bundle["direct_result"]["sha256"]
            completion["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            completion["ppa_receipt_sha256"] = bundle["ppa_receipt"]["sha256"]
            completion["energy_receipt_sha256"] = bundle["energy_receipt"]["sha256"]
            completion["accuracy_receipt_sha256"] = bundle["accuracy_receipt"]["sha256"]
            self._rewrite_json(bundle["completion_receipt"], completion)

            coverage = self._load_spec(bundle["coverage_receipt"])
            coverage["direct_result_sha256"] = bundle["direct_result"]["sha256"]
            coverage["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            coverage["population_manifest_sha256"] = population_spec["sha256"]
            coverage["density_strata"] = ["high", "low", "medium"]
            self._rewrite_json(bundle["coverage_receipt"], coverage)

            reviews_dir = (REPO_ROOT / old_bundle["independent_hammer_receipt"]["path"]).parent
            results_dir = (REPO_ROOT / bundle["direct_result"]["path"]).parent
            numerator_path = results_dir / "fixed_numerator.json"
            operator_ids = measurement_doc["operator_ids"]
            numerator_value = {
                "schema": "m527_h67_fixed_throughput_numerator_receipt_v1",
                "status": "PASS_RECOMPUTED_FIXED_POPULATION",
                "m527_contract_sha256": M.M527_CONTRACT_SHA256,
                "checkpoint_sha256": M.CHECKPOINT_SHA256,
                "measurement_identity_sha256": bundle["measurement_identity"]["sha256"],
                "complete_trace_manifest_sha256": measurement_doc["complete_trace_manifest"]["sha256"],
                "sequence_population_manifest_sha256": population_spec["sha256"],
                "aggregation_weight_manifest_sha256": measurement_doc["aggregation_weight_manifest"]["sha256"],
                "population_scalar": 3,
                "population_unit": "frozen_frames_across_frozen_sequence_population",
                "frame_definition": measurement_doc["frame_definition"],
                "op_convention": {"multiply_ops": 1, "add_ops": 1, "mac_ops": 2,
                                  "comparison_ops": 1, "state_update_ops": 1,
                                  "normalization_ops": 1, "address_and_control_ops": 0},
                "included_operator_scope": operator_ids,
                "excluded_operator_scope_with_reason": {},
                "excluded_numerator_ops_cycles_energy_traffic_charged": True,
                "dense_equivalent_ops_scalar": 1000,
                "dense_equivalent_ops_unit": "ops_per_frozen_population",
                "original_useful_nonzero_ops_scalar": 100,
                "original_useful_nonzero_ops_unit": "ops_per_frozen_population",
                "configuration_ids": [M.ROW_TO_M527_CONFIGURATION[row] for row in M.MANDATORY_ROW_IDS],
            }
            numerator_path.write_text(json.dumps(numerator_value, separators=(",", ":")), encoding="utf-8")
            numerator = self._spec(numerator_path, "application/json")

            m527_spec = self._spec(M.M527_CONTRACT, "application/json")
            checkpoint_spec = {"path": M.CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
                               "sha256": M.CHECKPOINT_SHA256,
                               "media_type": "application/octet-stream"}
            bundle.update({"schema": "m653.h67.rooted_direct_bundle.r2",
                           "m527_contract": m527_spec, "checkpoint": checkpoint_spec,
                           "fixed_throughput_numerator_receipt": numerator})

            evidence = {"m527_contract": m527_spec["sha256"],
                        "checkpoint": checkpoint_spec["sha256"],
                        "fixed_throughput_numerator_receipt": numerator["sha256"],
                        "common_resource_manifest": bundle["common_resource_manifest"]["sha256"],
                        "measurement_identity": bundle["measurement_identity"]["sha256"],
                        "raw_run_index": bundle["raw_run_index"]["sha256"],
                        "direct_result": bundle["direct_result"]["sha256"],
                        "completion_receipt": bundle["completion_receipt"]["sha256"],
                        "coverage_receipt": bundle["coverage_receipt"]["sha256"],
                        "ppa_receipt": bundle["ppa_receipt"]["sha256"],
                        "energy_receipt": bundle["energy_receipt"]["sha256"],
                        "accuracy_receipt": bundle["accuracy_receipt"]["sha256"]}
            for row_id, spec in bundle["configuration_manifests"].items():
                evidence["configuration_manifest:" + row_id] = spec["sha256"]
            for index, spec in enumerate(run_specs):
                evidence["raw_log:%06d" % index] = spec["sha256"]
            evidence = {key: evidence[key] for key in sorted(evidence)}

            target_paths = {
                "registry_builder": M.Path(M.__file__), "registry_config": M.DEFAULT_CONFIG,
                "registry_tests": M.REGISTRY_TESTS, "registry_contract": M.REGISTRY_CONTRACT,
                "m527_contract": M.M527_CONTRACT, "checkpoint": M.CHECKPOINT,
            }
            reviewed_targets = {}
            for name, path in target_paths.items():
                media = "application/octet-stream" if name == "checkpoint" else (
                    "text/x-python" if name in ("registry_builder", "registry_tests") else "application/json")
                if name == "checkpoint":
                    reviewed_targets[name] = copy.deepcopy(checkpoint_spec)
                else:
                    reviewed_targets[name] = self._spec(path, media)
            reviewed_targets["direct_result"] = copy.deepcopy(bundle["direct_result"])
            reviewed_targets["fixed_throughput_numerator_receipt"] = copy.deepcopy(numerator)
            reviewed_target_shas = {key: reviewed_targets[key]["sha256"] for key in sorted(reviewed_targets)}

            authority_id = "test_only_m653_exact_review_authority"
            request_path = reviews_dir / "m653_request.json"
            request_value = {"schema": "m653.h67.direct_unified.hammer_request.r1",
                             "status": "FROZEN_BEFORE_REVIEW", "authority_id": authority_id,
                             "bundle_id": bundle_id, "reviewed_targets": reviewed_targets,
                             "bundle_evidence_sha256": evidence,
                             "complete_evidence_root_sha256": M._map_sha(evidence)}
            request_path.write_text(json.dumps(request_value, separators=(",", ":")), encoding="utf-8")
            request = self._spec(request_path, "application/json")
            request_manifest, request_outer = self._seal_one(reviews_dir, request)

            result_rows = result["rows"]
            aggregates = result["aggregates"]
            views = result["views"]
            hammer_path = reviews_dir / "m653_hammer.json"
            hammer_value = {
                "schema": "m653.h67.direct_unified.independent_hammer.r1", "status": "PASS_INDEPENDENT",
                "authority_id": authority_id, "request_outer_seal_sha256": request_outer["sha256"],
                "reviewed_targets_sha256": reviewed_target_shas, "bundle_evidence_sha256": evidence,
                "severity_counts": {"P0": 0, "P1": 0},
                "independence": {"author_receipt_used_as_authority": False,
                                 "raw_logs_rehashed_and_recomputed": True,
                                 "fixed_numerators_rehashed_and_recomputed": True,
                                 "typed_receipts_recomputed": True,
                                 "raw_ppa_reports_parsed_and_projected": True,
                                 "result_modified": False},
                "recomputed_fixed_throughput_numerators": {
                    "dense_equivalent_ops_scalar": 1000,
                    "original_useful_nonzero_ops_scalar": 100},
                "recomputed_rows": result_rows, "recomputed_aggregates": aggregates,
                "recomputed_views": views,
                "authorization": {"table_a_methodology_admitted": True,
                                  "direct_unified_measurement_admitted": True,
                                  "paper_headline_admitted": True}}
            hammer_path.write_text(json.dumps(hammer_value, separators=(",", ":")), encoding="utf-8")
            hammer = self._spec(hammer_path, "application/json")
            review_path = reviews_dir / "m653_review.json"
            review_value = {"schema": "m653.h67.direct_unified.hammer_review.r1",
                            "status": "COMPLETE", "authority_id": authority_id,
                            "request_outer_seal_sha256": request_outer["sha256"],
                            "reviewed_targets_sha256": reviewed_target_shas,
                            "bundle_evidence_sha256": evidence,
                            "complete_evidence_root_sha256": M._map_sha(evidence),
                            "receipt_sha256": hammer["sha256"],
                            "severity_counts": {"P0": 0, "P1": 0}, "verdict": "GO"}
            review_path.write_text(json.dumps(review_value, separators=(",", ":")), encoding="utf-8")
            review = self._spec(review_path, "application/json")
            review_manifest, review_outer = self._seal_two(reviews_dir, hammer, review)
            authority = {"request_document": request, "request_manifest": request_manifest,
                         "request_outer_seal": request_outer, "review_document": review,
                         "review_manifest": review_manifest, "review_outer_seal": review_outer,
                         "receipt": hammer}
            bundle["independent_hammer_authority_id"] = authority_id
            bundle["independent_hammer_receipt"] = hammer

            sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
            rows = copy.deepcopy(sealed["table_a_schema"]["rows"])
            for row in rows[:6]:
                evidence_row = next(item for item in result_rows if item["row_id"] == row["row_id"])
                row.update({"cycles": evidence_row["cycles"], "energy_mj": evidence_row["energy_mj"],
                            "area_mm2": evidence_row["area_mm2"], "accuracy": evidence_row["accuracy"],
                            "source_id": bundle_id, "measurement_class": M.M635.ALLOWED_MEASUREMENT_CLASS,
                            "population_id": "fixture_population", "workload_id": "fixture_workload",
                            "resource_manifest_sha256": bundle["common_resource_manifest"]["sha256"],
                            "completion_receipt_sha256": bundle["completion_receipt"]["sha256"],
                            "decoder_complete": True, "memory_timing_included": True,
                            "full_network_completion": True, "logic_sram_dram_energy_closed": True,
                            "logic_macro_area_closed": True, "sta_closed": True,
                            "independent_hammer_pass": True, "blockers": []})
            obj = copy.deepcopy(self.base)
            obj["table_a_evidence_bundles"][bundle_id] = bundle
            obj["table_a_rows"] = rows
            obj["claim_boundary"]["table_a_admitted_rows"] = 6
            obj["claim_boundary"]["paper_headline_admitted"] = True
            previous = M.TRUSTED_HAMMER_AUTHORITIES
            M.TRUSTED_HAMMER_AUTHORITIES = {authority_id: authority}
            try:
                yield self._config(obj), bundle, authority
            finally:
                M.TRUSTED_HAMMER_AUTHORITIES = previous

    def test_01_canonical_is_zero_and_headline_false(self):
        result = M.build(CONFIG)
        self.assertEqual(0, result["trusted_hammer_authority_count"])
        self.assertEqual(0, result["table_a_evidence_bundle_count"])
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["headline_gate"]["all_m527_independent_gates_pass"])

    def test_02_runtime_frozen_roots_and_docs359_are_exact(self):
        self.assertEqual(M.M527_CONTRACT_SHA256, M._sha256(M.M527_CONTRACT))
        self.assertEqual(M.CHECKPOINT_SHA256, M._sha256(M.CHECKPOINT))
        self.assertEqual(M.DOCS359_SHA256, M._sha256(M.DOCS359))
        M._runtime_m527_contract()

    def test_03_density_vocabulary_is_exact_m527_medium(self):
        with self._rooted_positive_fixture() as (config_path, bundle, _):
            result = M.build(config_path)
            self.assertTrue(result["headline_gate"]["admitted"])
            measurement = self._load_spec(bundle["measurement_identity"])
            population_spec = measurement["sequence_population_manifest"]
            population = self._load_spec(population_spec)
            population["samples"][1]["density_stratum"] = "mid"
            self._rewrite_json(population_spec, population)
            measurement["sequence_population_manifest"] = population_spec
            self._rewrite_json(bundle["measurement_identity"], measurement)
            with self.assertRaisesRegex(M.RegistryError, "low/medium/high"):
                M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)

    def test_04_future_exact_graph_reaches_six_rows(self):
        with self._rooted_positive_fixture() as (config_path, _, _):
            result = M.build(config_path)
        self.assertEqual(6, result["headline_gate"]["eligible_row_count"])
        self.assertTrue(result["headline_gate"]["admitted"])
        self.assertTrue(result["headline_gate"]["all_m527_independent_gates_pass"])
        self.assertEqual(2.0, result["headline_gate"]["direct_speedup"])

    def test_05_arbitrary_target_request_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, _, authority):
            request_path = REPO_ROOT / authority["request_document"]["path"]
            request_path.write_text('{"target":"fixture"}', encoding="utf-8")
            with self.assertRaises(M.RegistryError):
                M.build(config_path)

    def test_06_request_wrong_registry_contract_sha_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, _, authority):
            request_path = REPO_ROOT / authority["request_document"]["path"]
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["reviewed_targets"]["registry_contract"]["sha256"] = "0" * 64
            request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            with self.assertRaises(M.RegistryError):
                M.build(config_path)

    def test_07_review_nonzero_p1_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, _, authority):
            review_path = REPO_ROOT / authority["review_document"]["path"]
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["severity_counts"] = {"P0": 0, "P1": 1}
            review_path.write_text(json.dumps(review, separators=(",", ":")), encoding="utf-8")
            with self.assertRaises(M.RegistryError):
                M.build(config_path)

    def test_08_raw_logic_area_9999_vs_typed_point6_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, bundle, _):
            ppa = self._load_spec(bundle["ppa_receipt"])
            report = ppa["rows"][0]["logic_report"]
            self._rewrite_text(report, "logic_area_mm2 9999.0\nlogic_power_mw 0.2\n")
            self._rewrite_json(bundle["ppa_receipt"], ppa)
            obj = M.M635.load_json(config_path, "fixture")
            obj["table_a_evidence_bundles"][bundle["bundle_id"]]["ppa_receipt"] = bundle["ppa_receipt"]
            with self.assertRaisesRegex(M.RegistryError, "does not project raw report"):
                M.build(self._config(obj))

    def test_09_raw_power_mismatch_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, bundle, _):
            ppa = self._load_spec(bundle["ppa_receipt"])
            report = ppa["rows"][0]["sram_report"]
            self._rewrite_text(report, "sram_macro_area_mm2 0.4\nsram_macro_power_mw 77.0\n")
            self._rewrite_json(bundle["ppa_receipt"], ppa)
            obj = M.M635.load_json(config_path, "fixture")
            obj["table_a_evidence_bundles"][bundle["bundle_id"]]["ppa_receipt"] = bundle["ppa_receipt"]
            with self.assertRaisesRegex(M.RegistryError, "does not project raw report"):
                M.build(self._config(obj))

    def test_10_numerator_scalar_or_unit_mutation_is_rejected(self):
        for field, value in (("dense_equivalent_ops_scalar", 0),
                             ("dense_equivalent_ops_unit", "ops_per_frame")):
            with self._rooted_positive_fixture() as (config_path, bundle, _):
                numerator = self._load_spec(bundle["fixed_throughput_numerator_receipt"])
                numerator[field] = value
                self._rewrite_json(bundle["fixed_throughput_numerator_receipt"], numerator)
                obj = M.M635.load_json(config_path, "fixture")
                obj["table_a_evidence_bundles"][bundle["bundle_id"]]["fixed_throughput_numerator_receipt"] = bundle["fixed_throughput_numerator_receipt"]
                with self.assertRaises(M.RegistryError):
                    M.build(self._config(obj))

    def test_11_bundle_checkpoint_string_without_actual_file_spec_is_rejected(self):
        with self._rooted_positive_fixture() as (config_path, bundle, _):
            obj = M.M635.load_json(config_path, "fixture")
            del obj["table_a_evidence_bundles"][bundle["bundle_id"]]["checkpoint"]
            with self.assertRaisesRegex(M.RegistryError, "fields differ"):
                M.build(self._config(obj))

    def test_12_config_cannot_add_authority_and_canonical_map_is_empty(self):
        self.assertEqual({}, M.TRUSTED_HAMMER_AUTHORITIES)
        obj = copy.deepcopy(self.base)
        obj["trusted_hammer_authorities"] = {}
        with self.assertRaisesRegex(M.RegistryError, "fields differ"):
            M.build(self._config(obj))

    def test_13_nonfinite_is_rejected(self):
        path = Path(tempfile.mkstemp(suffix=".json")[1])
        self.addCleanup(self._remove, path)
        path.write_text('{"schema":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(M.RegistryError, "non-finite"):
            M.build(path)

    def test_14_analytical_range_stays_nonadmitted(self):
        result = M.build(CONFIG)
        self.assertFalse(result["analytical_diagnostic"]["admitted"])


if __name__ == "__main__":
    unittest.main()

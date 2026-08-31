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
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m645_h67_paper_metric_registry_r4.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m645_h67_paper_metric_registry_r4_20260828.json"
SPEC = importlib.util.spec_from_file_location("m645_registry", str(SCRIPT))
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M645RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = M.M635.load_json(CONFIG, "M645 test config")

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
    def _dummy_spec(path="hw_autoresearch_nts07/results/not_a_real_run.json"):
        return {"path": path, "sha256": "1" * 64, "media_type": "application/json"}

    @staticmethod
    def _write_json(directory, name, value):
        path = Path(directory) / name
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                        encoding="utf-8")
        return {"path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": "application/json"}

    @staticmethod
    def _write_text(directory, name, value, media_type="text/plain"):
        path = Path(directory) / name
        path.write_text(value, encoding="utf-8")
        return {"path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": media_type}

    @contextlib.contextmanager
    def _rooted_positive_fixture(self):
        results_root = REPO_ROOT / "hw_autoresearch_nts07/results"
        simulator_root = REPO_ROOT / "hw_autoresearch_nts07/system_simulator"
        contracts_root = REPO_ROOT / "hw_autoresearch_nts07/contracts"
        reviews_root = REPO_ROOT / "hw_autoresearch_nts07/reviews"
        with tempfile.TemporaryDirectory(dir=str(results_root)) as results_dir, \
                tempfile.TemporaryDirectory(dir=str(simulator_root)) as simulator_dir, \
                tempfile.TemporaryDirectory(dir=str(contracts_root)) as contracts_dir, \
                tempfile.TemporaryDirectory(dir=str(reviews_root)) as reviews_dir:
            producer = self._write_text(simulator_dir, "producer.py", "# frozen producer\n", "text/x-python")
            simulator = self._write_text(simulator_dir, "simulator.py", "# unified simulator\n", "text/x-python")
            trace_member = self._write_json(simulator_dir, "trace_member.json", {"records": 1})
            trace = self._write_json(simulator_dir, "trace.json", {
                "schema": "m645.h67.complete_trace_manifest.r1", "status": "FROZEN_COMPLETE",
                "checkpoint_sha256": M.CHECKPOINT_SHA256, "decoder_complete": True,
                "operator_scope": ["patch_embed", "Conv2d", "ConvTranspose2d", "fc1", "fc2",
                                   "dynamic_BN", "ATLIF", "attention", "prediction_head"],
                "record_count": 1, "trace_members": [trace_member],
            })
            population_rows = [
                {"sample_id": "sample_low", "sequence_id": "seq_a", "density_stratum": "low", "frame_count": 1},
                {"sample_id": "sample_mid", "sequence_id": "seq_b", "density_stratum": "mid", "frame_count": 1},
                {"sample_id": "sample_high", "sequence_id": "seq_c", "density_stratum": "high", "frame_count": 1},
            ]
            population = self._write_json(simulator_dir, "population.json", {
                "schema": "m645.h67.sequence_population_manifest.r1", "status": "FROZEN_COMPLETE",
                "checkpoint_sha256": M.CHECKPOINT_SHA256, "population_id": "fixture_population",
                "samples": population_rows,
            })
            weight_rows = []
            for sample in population_rows:
                for view in M.VIEW_NAMES:
                    weight_rows.append({"sample_id": sample["sample_id"], "view": view, "weight": 1.0 / 3.0})
            weights = self._write_json(simulator_dir, "weights.json", {
                "schema": "m645.h67.aggregation_weight_manifest.r1", "status": "FROZEN_COMPLETE",
                "checkpoint_sha256": M.CHECKPOINT_SHA256, "population_id": "fixture_population",
                "selection_frozen_before_results": True, "samples": weight_rows,
            })
            measurement = self._write_json(simulator_dir, "measurement.json", {
                "schema": "m645.h67.measurement_identity.r1", "status": "FROZEN_COMPLETE",
                "m527_contract_sha256": M.M527_CONTRACT_SHA256,
                "checkpoint_sha256": M.CHECKPOINT_SHA256,
                "complete_trace_manifest": trace,
                "sequence_population_manifest": population,
                "aggregation_weight_manifest": weights,
                "frame_definition": "one complete H67 invocation",
                "density_metric": "event_count_per_frame",
                "density_bin_boundaries": [0, 10, 20, 100],
                "operator_ids": ["patch", "conv", "decoder"],
            })
            invocation = self._write_json(contracts_dir, "invocation.json", {
                "schema": "m645.h67.direct_invocation_contract.r1", "status": "FROZEN_BEFORE_RUN",
                "producer_source_sha256": producer["sha256"],
                "simulator_source_sha256": simulator["sha256"],
                "checkpoint_sha256": M.CHECKPOINT_SHA256,
                "command_argv": ["python", "producer.py", "--direct-cycles"],
                "environment_allowlist": {}, "direct_cycles_required": True,
                "fallback_charged": True,
            })
            resource = {
                "technology_nm": 28, "clock_period_ns": 3.0, "source_lanes": 96,
                "service_width_sources_per_cycle": 8, "onchip_sram_bytes_total": 245760,
                "dram_bandwidth_bytes_per_second_decimal": 64000000000,
                "dram_bytes_per_cycle": 192, "accumulator_bits": 24,
                "source_queue_depth": 8, "completion_queue_depth": 8, "parent_queue_depth": 8,
                "weight_sram_bank_count": 8, "state_sram_bank_count": 8,
                "parent_scratch_bank_count": 1, "weight_sram_port_mode": "1R1W",
                "state_sram_port_mode": "1R1W", "parent_scratch_port_mode": "1R1W",
                "external_read_port_count": 1, "external_write_port_count": 1,
            }
            charge = {field: True for field in M.CHARGE_FIELDS}
            fallback_common = {"mode": "EXECUTE_UNSUPPORTED_WORK_IN_THE_SAME_UNIFIED_MODEL",
                               "must_charge_cycles": True, "must_charge_traffic": True,
                               "must_charge_energy": True, "must_charge_area": True,
                               "unsupported_operator_ids": []}
            common = self._write_json(simulator_dir, "common.json", {
                "schema": "m527_h67_common_resource_manifest_v1", "status": "FROZEN_EXECUTABLE",
                "m527_contract_sha256": M.M527_CONTRACT_SHA256,
                "checkpoint_sha256": M.CHECKPOINT_SHA256,
                "producer_source_sha256": producer["sha256"],
                "simulator_source_sha256": simulator["sha256"],
                "measurement_identity_sha256": measurement["sha256"],
                "resource_tuple": resource, "charge_policy": charge,
                "fallback_policy": fallback_common,
            })
            configs = {}
            for row_id in M.MANDATORY_ROW_IDS:
                config_id = M.ROW_TO_M527_CONFIGURATION[row_id]
                mechanisms = dict(zip(M.MECHANISM_FIELDS, M.EXPECTED_MECHANISMS[config_id]))
                source = self._write_json(simulator_dir, row_id + "_source.json", {
                    "schema": "m645.h67.executable_configuration_source.r1",
                    "configuration_id": config_id, "mechanism_enable_map": mechanisms,
                    "optimized_operator_ids": ["conv"],
                    "unsupported_operator_ids": ["patch", "decoder"],
                })
                fallback = copy.deepcopy(fallback_common)
                fallback["unsupported_operator_ids"] = ["patch", "decoder"]
                configs[row_id] = self._write_json(simulator_dir, row_id + "_manifest.json", {
                    "schema": "m527_h67_executable_configuration_manifest_v1",
                    "status": "FROZEN_EXECUTABLE", "configuration_id": config_id,
                    "configuration_source": source,
                    "producer_source_sha256": producer["sha256"],
                    "simulator_source_sha256": simulator["sha256"],
                    "invocation_contract_sha256": invocation["sha256"],
                    "checkpoint_sha256": M.CHECKPOINT_SHA256,
                    "complete_trace_manifest_sha256": trace["sha256"],
                    "sequence_population_manifest_sha256": population["sha256"],
                    "aggregation_weight_manifest_sha256": weights["sha256"],
                    "common_resource_manifest_sha256": common["sha256"],
                    "mechanism_enable_map": mechanisms, "optimized_operator_ids": ["conv"],
                    "resource_tuple": resource, "charge_policy": charge, "fallback_policy": fallback,
                })
            cycle_map = {"dense96_fixed_t10": 200, "ptb_like_structured": 180,
                         "exact_bit_k1": 160, "exact_bit_k1x8": 140,
                         "exact_typed_k8": 130, "ours_exact": 100}
            run_items = []
            run_logs = []
            for sample in population_rows:
                for view in M.VIEW_NAMES:
                    for row_id in M.MANDATORY_ROW_IDS:
                        run_id = "%s_%s_%s" % (sample["sample_id"], view, row_id)
                        log = self._write_json(results_dir, run_id + ".json", {
                            "schema": "m645.h67.direct_raw_run.r1", "status": "PASS_DIRECT",
                            "run_id": run_id, "sample_id": sample["sample_id"],
                            "sequence_id": sample["sequence_id"],
                            "density_stratum": sample["density_stratum"], "view": view,
                            "row_id": row_id, "configuration_manifest_sha256": configs[row_id]["sha256"],
                            "producer_source_sha256": producer["sha256"],
                            "simulator_source_sha256": simulator["sha256"],
                            "invocation_contract_sha256": invocation["sha256"],
                            "checkpoint_sha256": M.CHECKPOINT_SHA256,
                            "complete_trace_manifest_sha256": trace["sha256"],
                            "sequence_population_manifest_sha256": population["sha256"],
                            "aggregation_weight_manifest_sha256": weights["sha256"],
                            "direct_cycles": cycle_map[row_id], "logic_energy_nj": 100.0,
                            "sram_energy_nj": 200.0, "dram_energy_nj": 300.0,
                            "aee": 1.0, "dsec_fl_percent": 5.0,
                        })
                        run_items.append({"run_id": run_id, "sample_id": sample["sample_id"],
                                          "view": view, "row_id": row_id, "log": log})
                        run_logs.append(log)
            raw_index = self._write_json(results_dir, "raw_index.json", {
                "schema": "m645.h67.raw_run_index.r1", "status": "PASS_COMPLETE",
                "producer_source_sha256": producer["sha256"],
                "simulator_source_sha256": simulator["sha256"],
                "invocation_contract_sha256": invocation["sha256"],
                "measurement_identity_sha256": measurement["sha256"], "runs": run_items,
            })
            energy_rows = []
            accuracy_rows = []
            ppa_rows = []
            for row_id in M.MANDATORY_ROW_IDS:
                energy_rows.append({"row_id": row_id,
                                    "configuration_manifest_sha256": configs[row_id]["sha256"],
                                    "logic_energy_mj": 0.0001, "sram_energy_mj": 0.0002,
                                    "dram_energy_mj": 0.0003, "total_energy_mj": 0.0006})
                accuracy_rows.append({"row_id": row_id,
                                      "configuration_manifest_sha256": configs[row_id]["sha256"],
                                      "aee": 1.0, "dsec_fl_percent": 5.0})
                logic_report = self._write_text(results_dir, row_id + "_logic.rpt", "area 0.6\n")
                sram_report = self._write_text(results_dir, row_id + "_sram.rpt", "area 0.4\n")
                sta_report = self._write_text(results_dir, row_id + "_sta.rpt", "setup 0 hold 0\n")
                ppa_rows.append({"row_id": row_id,
                                 "configuration_manifest_sha256": configs[row_id]["sha256"],
                                 "logic_area_mm2": 0.6, "sram_macro_area_mm2": 0.4,
                                 "total_area_mm2": 1.0, "setup_wns_ns": 0.0, "hold_wns_ns": 0.0,
                                 "logic_report": logic_report, "sram_report": sram_report,
                                 "sta_report": sta_report})
            energy = self._write_json(results_dir, "energy.json", {
                "schema": "m645.h67.logic_sram_dram_energy_receipt.r1", "status": "PASS_TYPED",
                "raw_run_index_sha256": raw_index["sha256"], "rows": energy_rows})
            accuracy = self._write_json(results_dir, "accuracy.json", {
                "schema": "m645.h67.accuracy_receipt.r1", "status": "PASS_TYPED",
                "raw_run_index_sha256": raw_index["sha256"],
                "checkpoint_sha256": M.CHECKPOINT_SHA256,
                "population_manifest_sha256": population["sha256"], "rows": accuracy_rows})
            ppa = self._write_json(results_dir, "ppa.json", {
                "schema": "m645.h67.logic_macro_sta_ppa_receipt.r1", "status": "PASS_TYPED",
                "technology_nm": 28, "clock_period_ns": 3.0, "rows": ppa_rows})
            samples = []
            for view in M.VIEW_NAMES:
                for sample in population_rows:
                    samples.append({"sample_id": sample["sample_id"], "sequence_id": sample["sequence_id"],
                                    "density_stratum": sample["density_stratum"], "view": view,
                                    "row_cycles": copy.deepcopy(cycle_map)})
            aggregates = {"arithmetic_mean": 2.0, "geometric_mean": 2.0,
                          "ratio_of_sums": 2.0, "minimum": 2.0, "maximum": 2.0}
            result_rows = []
            roles = {item[0]: (item[1], item[2]) for item in M.M635.MANDATORY_ROW_SPECS}
            for row_id in M.MANDATORY_ROW_IDS:
                result_rows.append({"row_id": row_id, "role": roles[row_id][0], "fidelity": roles[row_id][1],
                                    "cycles": cycle_map[row_id] * 3, "energy_mj": 0.0006,
                                    "area_mm2": 1.0, "accuracy": 1.0})
            bundle_id = "direct_unified_rooted_test_fixture"
            direct_result = self._write_json(results_dir, "result.json", {
                "schema": "m645.h67.direct_unified.result.r1", "status": "PASS_COMPLETE",
                "bundle_id": bundle_id, "measurement_class": M.M635.ALLOWED_MEASUREMENT_CLASS,
                "population_id": "fixture_population", "workload_id": "fixture_workload",
                "common_resource_manifest_sha256": common["sha256"],
                "raw_run_index_sha256": raw_index["sha256"],
                "ppa_receipt_sha256": ppa["sha256"], "energy_receipt_sha256": energy["sha256"],
                "accuracy_receipt_sha256": accuracy["sha256"], "rows": result_rows,
                "samples": samples, "aggregates": aggregates,
                "views": {"iso_lane": copy.deepcopy(aggregates),
                          "iso_service": copy.deepcopy(aggregates)},
            })
            completion = self._write_json(results_dir, "completion.json", {
                "schema": "m645.h67.direct_unified.completion_receipt.r1",
                "status": "PASS_DERIVED_CLOSURES", "bundle_id": bundle_id,
                "direct_result_sha256": direct_result["sha256"],
                "raw_run_index_sha256": raw_index["sha256"],
                "ppa_receipt_sha256": ppa["sha256"], "energy_receipt_sha256": energy["sha256"],
                "accuracy_receipt_sha256": accuracy["sha256"],
                "completed_row_ids": list(M.MANDATORY_ROW_IDS),
            })
            coverage = self._write_json(results_dir, "coverage.json", {
                "schema": "m645.h67.coverage_receipt.r1", "status": "PASS_RECOMPUTED",
                "direct_result_sha256": direct_result["sha256"],
                "raw_run_index_sha256": raw_index["sha256"],
                "population_manifest_sha256": population["sha256"],
                "aggregation_manifest_sha256": weights["sha256"],
                "sample_ids": [row["sample_id"] for row in population_rows],
                "sequence_ids": ["seq_a", "seq_b", "seq_c"],
                "density_strata": ["high", "low", "mid"], "aggregates": aggregates,
                "views": {"iso_lane": copy.deepcopy(aggregates),
                          "iso_service": copy.deepcopy(aggregates)},
            })
            evidence = {"common_resource_manifest": common["sha256"],
                        "measurement_identity": measurement["sha256"],
                        "raw_run_index": raw_index["sha256"],
                        "direct_result": direct_result["sha256"],
                        "completion_receipt": completion["sha256"],
                        "coverage_receipt": coverage["sha256"],
                        "ppa_receipt": ppa["sha256"], "energy_receipt": energy["sha256"],
                        "accuracy_receipt": accuracy["sha256"]}
            for row_id in M.MANDATORY_ROW_IDS:
                evidence["configuration_manifest:" + row_id] = configs[row_id]["sha256"]
            for index, log in enumerate(run_logs):
                evidence["raw_log:%06d" % index] = log["sha256"]
            request_json = self._write_json(reviews_dir, "request.json", {"target": "fixture"})
            request_manifest_path = Path(reviews_dir) / "REQUEST_SHA256SUMS"
            request_manifest_path.write_text("%s  request.json\n" % request_json["sha256"], encoding="utf-8")
            request_manifest = {"path": request_manifest_path.relative_to(REPO_ROOT).as_posix(),
                                "sha256": hashlib.sha256(request_manifest_path.read_bytes()).hexdigest(),
                                "media_type": "text/plain"}
            request_outer_path = Path(reviews_dir) / "REQUEST_SHA256SUMS.seal.sha256"
            request_outer_path.write_text("%s  REQUEST_SHA256SUMS\n" % request_manifest["sha256"], encoding="utf-8")
            request_outer = {"path": request_outer_path.relative_to(REPO_ROOT).as_posix(),
                             "sha256": hashlib.sha256(request_outer_path.read_bytes()).hexdigest(),
                             "media_type": "text/plain"}
            authority_id = "test_only_code_injected_authority"
            hammer = self._write_json(reviews_dir, "hammer.json", {
                "schema": "m645.h67.direct_unified.independent_hammer.r1", "status": "PASS_INDEPENDENT",
                "authority_id": authority_id, "request_outer_seal_sha256": request_outer["sha256"],
                "bundle_evidence_sha256": {key: evidence[key] for key in sorted(evidence)},
                "severity_counts": {"P0": 0, "P1": 0},
                "independence": {"author_receipt_used_as_authority": False,
                                 "raw_logs_rehashed_and_recomputed": True,
                                 "typed_receipts_recomputed": True, "result_modified": False},
                "recomputed_rows": result_rows, "recomputed_aggregates": aggregates,
                "recomputed_views": {"iso_lane": copy.deepcopy(aggregates),
                                     "iso_service": copy.deepcopy(aggregates)},
                "authorization": {"table_a_methodology_admitted": True,
                                  "direct_unified_measurement_admitted": True,
                                  "paper_headline_admitted": True},
            })
            review_manifest_path = Path(reviews_dir) / "SHA256SUMS"
            review_manifest_path.write_text("%s  hammer.json\n" % hammer["sha256"], encoding="utf-8")
            review_manifest = {"path": review_manifest_path.relative_to(REPO_ROOT).as_posix(),
                               "sha256": hashlib.sha256(review_manifest_path.read_bytes()).hexdigest(),
                               "media_type": "text/plain"}
            review_outer_path = Path(reviews_dir) / "SHA256SUMS.seal.sha256"
            review_outer_path.write_text("%s  SHA256SUMS\n" % review_manifest["sha256"], encoding="utf-8")
            review_outer = {"path": review_outer_path.relative_to(REPO_ROOT).as_posix(),
                            "sha256": hashlib.sha256(review_outer_path.read_bytes()).hexdigest(),
                            "media_type": "text/plain"}
            authority = {"request_manifest": request_manifest, "request_outer_seal": request_outer,
                         "review_manifest": review_manifest, "review_outer_seal": review_outer,
                         "receipt": hammer}
            bundle = {"schema": "m645.h67.rooted_direct_bundle.r1", "bundle_id": bundle_id,
                      "m527_contract_sha256": M.M527_CONTRACT_SHA256,
                      "common_resource_manifest": common, "configuration_manifests": configs,
                      "producer": producer, "unified_simulator": simulator,
                      "invocation_contract": invocation, "measurement_identity": measurement,
                      "raw_run_index": raw_index, "direct_result": direct_result,
                      "completion_receipt": completion, "coverage_receipt": coverage,
                      "ppa_receipt": ppa, "energy_receipt": energy,
                      "accuracy_receipt": accuracy,
                      "independent_hammer_authority_id": authority_id,
                      "independent_hammer_receipt": hammer}
            sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
            rows = copy.deepcopy(sealed["table_a_schema"]["rows"])
            for row in rows[:6]:
                evidence_row = next(item for item in result_rows if item["row_id"] == row["row_id"])
                row.update({"cycles": evidence_row["cycles"], "energy_mj": evidence_row["energy_mj"],
                            "area_mm2": evidence_row["area_mm2"], "accuracy": evidence_row["accuracy"],
                            "source_id": bundle_id, "measurement_class": M.M635.ALLOWED_MEASUREMENT_CLASS,
                            "population_id": "fixture_population", "workload_id": "fixture_workload",
                            "resource_manifest_sha256": common["sha256"],
                            "completion_receipt_sha256": completion["sha256"],
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

    def _expanded_untrusted_bundle(self, bundle_id="direct_unified_fabricated_consistent"):
        spec = self._dummy_spec()
        bundle = {
            "schema": "m645.h67.rooted_direct_bundle.r1",
            "bundle_id": bundle_id,
            "m527_contract_sha256": M.M527_CONTRACT_SHA256,
            "common_resource_manifest": copy.deepcopy(spec),
            "configuration_manifests": {
                row_id: copy.deepcopy(spec) for row_id in M.MANDATORY_ROW_IDS
            },
            "producer": {"path": "hw_autoresearch_nts07/system_simulator/scripts/fake.py",
                         "sha256": "2" * 64, "media_type": "text/x-python"},
            "unified_simulator": {"path": "hw_autoresearch_nts07/system_simulator/scripts/fake_sim.py",
                                  "sha256": "3" * 64, "media_type": "text/x-python"},
            "invocation_contract": copy.deepcopy(spec),
            "measurement_identity": copy.deepcopy(spec),
            "raw_run_index": copy.deepcopy(spec),
            "direct_result": copy.deepcopy(spec),
            "completion_receipt": copy.deepcopy(spec),
            "coverage_receipt": copy.deepcopy(spec),
            "ppa_receipt": copy.deepcopy(spec),
            "energy_receipt": copy.deepcopy(spec),
            "accuracy_receipt": copy.deepcopy(spec),
            "independent_hammer_authority_id": "author_self_declared_pass",
            "independent_hammer_receipt": {
                "path": "hw_autoresearch_nts07/reviews/fabricated/review.json",
                "sha256": "4" * 64,
                "media_type": "application/json",
            },
        }
        return bundle

    def test_01_canonical_is_zero_bundle_zero_authority_zero_headline(self):
        result = M.build(CONFIG)
        self.assertEqual(12, len(result["source_hashes_validated"]))
        self.assertEqual(0, result["trusted_hammer_authority_count"])
        self.assertEqual(0, result["table_a_evidence_bundle_count"])
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["analytical_diagnostic"]["admitted"])

    def test_02_all_frozen_parent_identities_are_exact(self):
        self.assertEqual(M.M635_BUILDER_SHA256, hashlib.sha256(M.M635_BUILDER.read_bytes()).hexdigest())
        self.assertEqual(M.M635_CONFIG_SHA256, hashlib.sha256(M.M635_CONFIG.read_bytes()).hexdigest())
        self.assertEqual(M.M527_CONTRACT_SHA256, hashlib.sha256(M.M527_CONTRACT.read_bytes()).hexdigest())
        self.assertEqual(M.DOCS359_SHA256, hashlib.sha256(M.DOCS359.read_bytes()).hexdigest())

    def test_03_base_registry_sha_or_path_substitution_is_rejected(self):
        for field, value in (("sha256", "0" * 64), ("path", "hw_autoresearch_nts07/system_simulator/config/m628_h67_paper_metric_registry_r2_20260828.json")):
            obj = copy.deepcopy(self.base)
            obj["base_registry"][field] = value
            with self.assertRaises(M.RegistryError):
                M.build(self._config(obj))

    def test_04_old_five_file_internally_consistent_bundle_shape_is_rejected(self):
        # This is the exact authority topology accepted by M635/M636's attack:
        # result + completion + resource + coverage + author-labelled hammer.
        old = {
            "direct_result": self._dummy_spec(),
            "completion_receipt": self._dummy_spec(),
            "resource_manifest": self._dummy_spec("hw_autoresearch_nts07/system_simulator/fake_resource.json"),
            "coverage_receipt": self._dummy_spec(),
            "independent_hammer_receipt": self._dummy_spec("hw_autoresearch_nts07/reviews/fake/review.json"),
        }
        obj = copy.deepcopy(self.base)
        obj["table_a_evidence_bundles"]["direct_unified_old_five_file_attack"] = old
        with self.assertRaisesRegex(M.RegistryError, "fields differ"):
            M.build(self._config(obj))

    def test_05_expanded_synchronized_bundle_cannot_self_declare_hammer_authority(self):
        obj = copy.deepcopy(self.base)
        bundle_id = "direct_unified_fabricated_consistent"
        obj["table_a_evidence_bundles"][bundle_id] = self._expanded_untrusted_bundle(bundle_id)
        with self.assertRaisesRegex(M.RegistryError, "not code-trusted"):
            M.build(self._config(obj))

    def test_06_config_cannot_add_a_trusted_hammer_authority(self):
        obj = copy.deepcopy(self.base)
        obj["trusted_hammer_authorities"] = {
            "author_pass": {"request_outer_seal": self._dummy_spec(),
                            "review_outer_seal": self._dummy_spec(),
                            "receipt": self._dummy_spec()}
        }
        with self.assertRaisesRegex(M.RegistryError, "fields differ"):
            M.build(self._config(obj))

    def test_07_every_table_a_row_maps_to_a_distinct_m527_configuration(self):
        self.assertEqual(set(M.MANDATORY_ROW_IDS), set(M.ROW_TO_M527_CONFIGURATION))
        self.assertEqual(len(M.MANDATORY_ROW_IDS), len(set(M.ROW_TO_M527_CONFIGURATION.values())))
        self.assertEqual("c2_exact_typed_k8", M.ROW_TO_M527_CONFIGURATION["exact_typed_k8"])

    def test_08_external_table_b_source_still_cannot_be_promoted(self):
        obj = copy.deepcopy(self.base)
        sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
        rows = copy.deepcopy(sealed["table_a_schema"]["rows"])
        for row in rows[:6]:
            row.update({
                "cycles": 100, "energy_mj": 1.0, "area_mm2": 1.0, "accuracy": 1.0,
                "source_id": "m618", "measurement_class": M.M635.ALLOWED_MEASUREMENT_CLASS,
                "population_id": "fake", "workload_id": "fake",
                "resource_manifest_sha256": "1" * 64, "completion_receipt_sha256": "2" * 64,
                "decoder_complete": True, "memory_timing_included": True,
                "full_network_completion": True, "logic_sram_dram_energy_closed": True,
                "logic_macro_area_closed": True, "sta_closed": True,
                "independent_hammer_pass": True, "blockers": [],
            })
        obj["table_a_rows"] = rows
        result = M.build(self._config(obj))
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])

    def test_09_claim_boundary_is_executable_not_descriptive(self):
        for field, value in (("table_a_admitted_rows", 1), ("paper_headline_admitted", True)):
            obj = copy.deepcopy(self.base)
            obj["claim_boundary"][field] = value
            with self.assertRaisesRegex(M.RegistryError, "claim boundary disagrees"):
                M.build(self._config(obj))

    def test_10_nonfinite_overlay_is_rejected_by_strict_parser(self):
        path = Path(tempfile.mkstemp(suffix=".json")[1])
        self.addCleanup(self._remove, path)
        path.write_text('{"schema": NaN}', encoding="utf-8")
        with self.assertRaisesRegex(M.RegistryError, "non-finite"):
            M.build(path)

    def test_11_m527_resource_contract_is_code_fixed(self):
        valid = {
            "technology_nm": 28, "clock_period_ns": 3.0, "source_lanes": 96,
            "service_width_sources_per_cycle": 8, "onchip_sram_bytes_total": 245760,
            "dram_bandwidth_bytes_per_second_decimal": 64000000000,
            "dram_bytes_per_cycle": 192, "accumulator_bits": 24,
            "source_queue_depth": 8, "completion_queue_depth": 8, "parent_queue_depth": 8,
            "weight_sram_bank_count": 8, "state_sram_bank_count": 8,
            "parent_scratch_bank_count": 1, "weight_sram_port_mode": "1R1W",
            "state_sram_port_mode": "1R1W", "parent_scratch_port_mode": "1R1W",
            "external_read_port_count": 1, "external_write_port_count": 1,
        }
        M._validate_resource(valid)
        invalid = copy.deepcopy(valid)
        invalid["onchip_sram_bytes_total"] += 1
        with self.assertRaisesRegex(M.RegistryError, "onchip_sram_bytes_total"):
            M._validate_resource(invalid)

    def test_12_unpaid_charge_or_fallback_is_rejected(self):
        charge = {field: True for field in M.CHARGE_FIELDS}
        M._validate_charge(charge)
        charge["extra_sram_ports_charged"] = False
        with self.assertRaisesRegex(M.RegistryError, "must be charged"):
            M._validate_charge(charge)
        fallback = {"mode": "EXECUTE_UNSUPPORTED_WORK_IN_THE_SAME_UNIFIED_MODEL",
                    "must_charge_cycles": True, "must_charge_traffic": True,
                    "must_charge_energy": True, "must_charge_area": False,
                    "unsupported_operator_ids": ["decoder"]}
        with self.assertRaisesRegex(M.RegistryError, "must charge"):
            M._validate_fallback(fallback)

    def test_13_file_specs_reject_symlink_and_wrong_sha(self):
        tests = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        with tempfile.TemporaryDirectory(dir=str(tests)) as temp:
            root = Path(temp)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            spec = {"path": link.relative_to(REPO_ROOT).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "media_type": "application/json"}
            with self.assertRaisesRegex(M.RegistryError, "symlink"):
                M._file_spec(spec, "symlink attack")
            spec["path"] = target.relative_to(REPO_ROOT).as_posix()
            spec["sha256"] = "0" * 64
            with self.assertRaisesRegex(M.RegistryError, "SHA mismatch"):
                M._file_spec(spec, "SHA attack")

    def test_14_analytical_range_remains_nonadmitted(self):
        result = M.build(CONFIG)
        self.assertFalse(result["analytical_diagnostic"]["admitted"])
        self.assertEqual("1.7942940217026179000564835389", result["analytical_diagnostic"]["speedup_low"])
        self.assertEqual("1.8234548159105851413543721236", result["analytical_diagnostic"]["speedup_high"])

    def test_15_code_trust_root_is_empty_in_canonical_release(self):
        self.assertEqual({}, M.TRUSTED_HAMMER_AUTHORITIES)

    def test_16_fully_rooted_fixture_reaches_the_future_admission_path(self):
        # Test-only code injection models a later source release that pins a
        # real review authority.  No production fixture or trust root is kept.
        with self._rooted_positive_fixture() as (config_path, _, _):
            result = M.build(config_path)
        self.assertEqual(1, result["trusted_hammer_authority_count"])
        self.assertEqual(1, result["table_a_evidence_bundle_count"])
        self.assertEqual(6, result["headline_gate"]["eligible_row_count"])
        self.assertTrue(result["headline_gate"]["admitted"])
        self.assertEqual(2.0, result["headline_gate"]["direct_speedup"])

    def test_17_review_outer_seal_must_transitively_seal_hammer_receipt(self):
        with self._rooted_positive_fixture() as (config_path, _, authority):
            manifest_path = REPO_ROOT / authority["review_manifest"]["path"]
            manifest_path.write_text("0" * 64 + "  hammer.json\n", encoding="utf-8")
            with self.assertRaises(M.RegistryError):
                M.build(config_path)

    def test_18_typed_energy_component_cannot_hide_in_a_correct_total(self):
        with self._rooted_positive_fixture() as (config_path, bundle, _):
            energy_path = REPO_ROOT / bundle["energy_receipt"]["path"]
            energy = json.loads(energy_path.read_text(encoding="utf-8"))
            energy["rows"][0]["logic_energy_mj"] += 0.00005
            energy["rows"][0]["dram_energy_mj"] -= 0.00005
            energy_path.write_text(json.dumps(energy, separators=(",", ":")), encoding="utf-8")
            bundle["energy_receipt"]["sha256"] = hashlib.sha256(energy_path.read_bytes()).hexdigest()
            obj = M.M635.load_json(config_path, "fixture config")
            obj["table_a_evidence_bundles"][bundle["bundle_id"]]["energy_receipt"]["sha256"] = bundle["energy_receipt"]["sha256"]
            with self.assertRaises(M.RegistryError):
                M.build(self._config(obj))


if __name__ == "__main__":
    unittest.main()

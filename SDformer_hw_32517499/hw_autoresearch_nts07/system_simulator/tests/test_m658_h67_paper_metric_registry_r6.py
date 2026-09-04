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
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m658_h67_paper_metric_registry_r6.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m658_h67_paper_metric_registry_r6_20260828.json"
R5_TESTS = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests/test_m653_h67_paper_metric_registry_r5.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load("m658_registry", SCRIPT)
T5 = _load("m658_fixture_source_r5", R5_TESTS)


class M658RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = M.M635.load_json(CONFIG, "M658 test config")
        T5.M653RegistryTests.setUpClass()

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
        manifest_path.write_text("%s  %s\n" % (document_spec["sha256"], Path(document_spec["path"]).name),
                                 encoding="utf-8")
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

    @staticmethod
    def _report_texts(row_id):
        logic_library = "tcbn28hpcplusbwp30p140ssg0p81v125c.db"
        macro_library = "sram_128x128_1rw_ssg0p81v125c.lib"
        corner = "ssg0p81v125c"
        delimiter = "****************************************\n"
        dc = (delimiter + "Report : area\nTool : dc_shell\nVersion : T-2022.03-SP5\n" +
              "Library : %s\nOperating Conditions : %s\n" % (logic_library, corner) +
              "Design : %s\nTotal cell area (um2): 600000.000000\n" % row_id + delimiter)
        power = (delimiter + "Report : power\nTool : pt_shell\nVersion : T-2022.03-SP5\n" +
                 "Library : %s\nOperating Conditions : %s\n" % (logic_library, corner) +
                 "Design : %s\nLogic dynamic power (mW): 0.200000\n" % row_id +
                 "Memory dynamic power (mW): 0.100000\n" + delimiter)
        sta = (delimiter + "Report : timing\nTool : pt_shell\nVersion : T-2022.03-SP5\n" +
               "Library : %s\nOperating Conditions : %s\n" % (logic_library, corner) +
               "Design : %s\nSetup WNS (ns): 0.000000\nHold WNS (ns): 0.000000\n" % row_id + delimiter)
        macro = (delimiter + "Report : macro_characterization\nTool : memory_compiler\n" +
                 "Version : R-2022.09\nLibrary : %s\nCorner : %s\n" % (macro_library, corner) +
                 "Macro : %s_sram\nMacro area (um2): 400000.000000\n" % row_id + delimiter)
        return dc, power, sta, macro

    def _write_extraction(self, directory, row_id, configuration_sha, report_specs):
        paths = {name: REPO_ROOT / spec["path"] for name, spec in report_specs.items()}
        extracted = M.EX.extract(paths["dc_area"], paths["ptpx_power"], paths["pt_sta"], paths["sram_macro"])
        tools, libraries, corners = M._identity_projection(extracted)
        value = {
            "schema": "m658.h67.synopsys_ppa_extraction_receipt.r1",
            "status": "PASS_EXTRACTED_FROM_BOUND_REPORTS", "row_id": row_id,
            "configuration_manifest_sha256": configuration_sha,
            "extractor_source": {"path": M.EXTRACTOR.relative_to(REPO_ROOT).as_posix(),
                                 "sha256": M.EXTRACTOR_SHA256, "media_type": "text/x-python"},
            "extraction_argv": M._expected_argv(report_specs), "raw_reports": report_specs,
            "synopsys_tools": tools, "libraries": libraries, "corners": corners,
            "library_identity_sha256": M.R5._map_sha(libraries),
            "units": {"logic_area_mm2": "mm2", "logic_power_mw": "mW",
                      "sram_macro_area_mm2": "mm2", "sram_macro_power_mw": "mW",
                      "setup_wns_ns": "ns", "hold_wns_ns": "ns"},
            "extracted_values": extracted["values"],
        }
        path = Path(directory) / (row_id + "_extraction.json")
        path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        return self._spec(path, "application/json")

    @contextlib.contextmanager
    def _rooted_positive_fixture(self):
        helper = T5.M653RegistryTests(methodName="test_01_canonical_is_zero_and_headline_false")
        helper.base = T5.M653RegistryTests.base
        with helper._rooted_positive_fixture() as (_, old_bundle, _):
            bundle = copy.deepcopy(old_bundle)
            bundle_id = bundle["bundle_id"]
            required = M._required_operator_scope()

            measurement = self._load_spec(bundle["measurement_identity"])
            trace_spec = measurement["complete_trace_manifest"]
            trace = self._load_spec(trace_spec)
            trace["operator_scope"] = required
            self._rewrite_json(trace_spec, trace)
            measurement["complete_trace_manifest"] = trace_spec
            measurement["operator_ids"] = required
            self._rewrite_json(bundle["measurement_identity"], measurement)

            common = self._load_spec(bundle["common_resource_manifest"])
            common["measurement_identity_sha256"] = bundle["measurement_identity"]["sha256"]
            self._rewrite_json(bundle["common_resource_manifest"], common)

            unsupported = [item for item in required if item != "Conv2d"]
            for row_id, spec in bundle["configuration_manifests"].items():
                doc = self._load_spec(spec)
                source_spec = doc["configuration_source"]
                source = self._load_spec(source_spec)
                source["optimized_operator_ids"] = ["Conv2d"]
                source["unsupported_operator_ids"] = unsupported
                self._rewrite_json(source_spec, source)
                doc["configuration_source"] = source_spec
                doc["complete_trace_manifest_sha256"] = trace_spec["sha256"]
                doc["common_resource_manifest_sha256"] = bundle["common_resource_manifest"]["sha256"]
                doc["optimized_operator_ids"] = ["Conv2d"]
                doc["fallback_policy"]["unsupported_operator_ids"] = unsupported
                self._rewrite_json(spec, doc)

            raw_index = self._load_spec(bundle["raw_run_index"])
            run_specs = []
            for item in raw_index["runs"]:
                log = self._load_spec(item["log"])
                log["configuration_manifest_sha256"] = bundle["configuration_manifests"][log["row_id"]]["sha256"]
                log["complete_trace_manifest_sha256"] = trace_spec["sha256"]
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
            for row in accuracy["rows"]:
                row["configuration_manifest_sha256"] = bundle["configuration_manifests"][row["row_id"]]["sha256"]
            self._rewrite_json(bundle["accuracy_receipt"], accuracy)

            results_dir = (REPO_ROOT / bundle["direct_result"]["path"]).parent
            old_ppa = self._load_spec(bundle["ppa_receipt"])
            ppa_rows = []
            ppa_evidence_specs = {}
            for old_row in old_ppa["rows"]:
                row_id = old_row["row_id"]
                dc, power, sta, macro = self._report_texts(row_id)
                report_specs = {}
                for name, content in (("dc_area", dc), ("ptpx_power", power),
                                      ("pt_sta", sta), ("sram_macro", macro)):
                    path = results_dir / (row_id + "_" + name + ".rpt")
                    path.write_text(content, encoding="utf-8")
                    report_specs[name] = self._spec(path, "text/plain")
                extraction = self._write_extraction(results_dir, row_id,
                                                    bundle["configuration_manifests"][row_id]["sha256"],
                                                    report_specs)
                ppa_evidence_specs[row_id] = {"extraction": extraction, "reports": report_specs}
                ppa_rows.append({"row_id": row_id,
                                 "configuration_manifest_sha256": bundle["configuration_manifests"][row_id]["sha256"],
                                 "logic_area_mm2": 0.6, "logic_power_mw": 0.2,
                                 "sram_macro_area_mm2": 0.4, "sram_macro_power_mw": 0.1,
                                 "total_area_mm2": 1.0, "total_power_mw": 0.3,
                                 "setup_wns_ns": 0.0, "hold_wns_ns": 0.0,
                                 "extraction_receipt": extraction})
            ppa = {"schema": "m658.h67.synopsys_rooted_ppa_receipt.r1",
                   "status": "PASS_RAW_REPORT_EXTRACTED", "technology_nm": 28,
                   "clock_period_ns": 3.0, "rows": ppa_rows}
            self._rewrite_json(bundle["ppa_receipt"], ppa)

            result = self._load_spec(bundle["direct_result"])
            result["common_resource_manifest_sha256"] = bundle["common_resource_manifest"]["sha256"]
            result["raw_run_index_sha256"] = bundle["raw_run_index"]["sha256"]
            result["ppa_receipt_sha256"] = bundle["ppa_receipt"]["sha256"]
            result["energy_receipt_sha256"] = bundle["energy_receipt"]["sha256"]
            result["accuracy_receipt_sha256"] = bundle["accuracy_receipt"]["sha256"]
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
            self._rewrite_json(bundle["coverage_receipt"], coverage)

            numerator = self._load_spec(bundle["fixed_throughput_numerator_receipt"])
            numerator["measurement_identity_sha256"] = bundle["measurement_identity"]["sha256"]
            numerator["complete_trace_manifest_sha256"] = trace_spec["sha256"]
            numerator["included_operator_scope"] = required
            numerator["excluded_operator_scope_with_reason"] = {}
            self._rewrite_json(bundle["fixed_throughput_numerator_receipt"], numerator)

            bundle["schema"] = "m658.h67.rooted_direct_bundle.r3"
            evidence = {"m527_contract": bundle["m527_contract"]["sha256"],
                        "checkpoint": bundle["checkpoint"]["sha256"],
                        "fixed_throughput_numerator_receipt": bundle["fixed_throughput_numerator_receipt"]["sha256"],
                        "common_resource_manifest": bundle["common_resource_manifest"]["sha256"],
                        "measurement_identity": bundle["measurement_identity"]["sha256"],
                        "raw_run_index": bundle["raw_run_index"]["sha256"],
                        "direct_result": bundle["direct_result"]["sha256"],
                        "completion_receipt": bundle["completion_receipt"]["sha256"],
                        "coverage_receipt": bundle["coverage_receipt"]["sha256"],
                        "ppa_receipt": bundle["ppa_receipt"]["sha256"],
                        "energy_receipt": bundle["energy_receipt"]["sha256"],
                        "accuracy_receipt": bundle["accuracy_receipt"]["sha256"],
                        "ppa_extractor_source": M.EXTRACTOR_SHA256,
                        "m527_required_operator_scope": M.R5._map_sha(required),
                        "m527_complete_trace_manifest": trace_spec["sha256"]}
            for row_id, spec in bundle["configuration_manifests"].items():
                evidence["configuration_manifest:" + row_id] = spec["sha256"]
            for index, spec in enumerate(run_specs):
                evidence["raw_log:%06d" % index] = spec["sha256"]
            for row_id, info in ppa_evidence_specs.items():
                evidence["ppa_extraction_receipt:" + row_id] = info["extraction"]["sha256"]
                for name, spec in info["reports"].items():
                    evidence["ppa_raw_report:%s:%s" % (row_id, name)] = spec["sha256"]
            evidence = {key: evidence[key] for key in sorted(evidence)}

            reviews_dir = (REPO_ROOT / bundle["independent_hammer_receipt"]["path"]).parent
            fixed_targets = {"registry_builder": Path(M.__file__), "registry_config": M.DEFAULT_CONFIG,
                             "registry_tests": M.REGISTRY_TESTS, "registry_contract": M.REGISTRY_CONTRACT,
                             "m527_contract": M.M527_CONTRACT, "checkpoint": M.CHECKPOINT}
            targets = {}
            for name, path in fixed_targets.items():
                media = "application/octet-stream" if name == "checkpoint" else (
                    "text/x-python" if name in ("registry_builder", "registry_tests") else "application/json")
                if name == "checkpoint":
                    targets[name] = copy.deepcopy(bundle["checkpoint"])
                else:
                    targets[name] = self._spec(path, media)
            targets["direct_result"] = copy.deepcopy(bundle["direct_result"])
            targets["fixed_throughput_numerator_receipt"] = copy.deepcopy(bundle["fixed_throughput_numerator_receipt"])
            target_shas = {key: targets[key]["sha256"] for key in sorted(targets)}
            authority_id = "test_only_m658_exact_review_authority"
            request_path = reviews_dir / "m658_request.json"
            request_value = {"schema": "m653.h67.direct_unified.hammer_request.r1",
                             "status": "FROZEN_BEFORE_REVIEW", "authority_id": authority_id,
                             "bundle_id": bundle_id, "reviewed_targets": targets,
                             "bundle_evidence_sha256": evidence,
                             "complete_evidence_root_sha256": M.R5._map_sha(evidence)}
            request_path.write_text(json.dumps(request_value, separators=(",", ":")), encoding="utf-8")
            request = self._spec(request_path, "application/json")
            request_manifest, request_outer = self._seal_one(reviews_dir, request)

            hammer_path = reviews_dir / "m658_hammer.json"
            hammer_value = {"schema": "m653.h67.direct_unified.independent_hammer.r1",
                            "status": "PASS_INDEPENDENT", "authority_id": authority_id,
                            "request_outer_seal_sha256": request_outer["sha256"],
                            "reviewed_targets_sha256": target_shas,
                            "bundle_evidence_sha256": evidence, "severity_counts": {"P0": 0, "P1": 0},
                            "independence": {"author_receipt_used_as_authority": False,
                                             "raw_logs_rehashed_and_recomputed": True,
                                             "fixed_numerators_rehashed_and_recomputed": True,
                                             "typed_receipts_recomputed": True,
                                             "raw_ppa_reports_parsed_and_projected": True,
                                             "result_modified": False},
                            "recomputed_fixed_throughput_numerators": {
                                "dense_equivalent_ops_scalar": numerator["dense_equivalent_ops_scalar"],
                                "original_useful_nonzero_ops_scalar": numerator["original_useful_nonzero_ops_scalar"]},
                            "recomputed_rows": result["rows"], "recomputed_aggregates": result["aggregates"],
                            "recomputed_views": result["views"],
                            "authorization": {"table_a_methodology_admitted": True,
                                              "direct_unified_measurement_admitted": True,
                                              "paper_headline_admitted": True}}
            hammer_path.write_text(json.dumps(hammer_value, separators=(",", ":")), encoding="utf-8")
            hammer = self._spec(hammer_path, "application/json")
            review_path = reviews_dir / "m658_review.json"
            review_value = {"schema": "m653.h67.direct_unified.hammer_review.r1", "status": "COMPLETE",
                            "authority_id": authority_id, "request_outer_seal_sha256": request_outer["sha256"],
                            "reviewed_targets_sha256": target_shas, "bundle_evidence_sha256": evidence,
                            "complete_evidence_root_sha256": M.R5._map_sha(evidence),
                            "receipt_sha256": hammer["sha256"], "severity_counts": {"P0": 0, "P1": 0},
                            "verdict": "GO"}
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
                evidence_row = next(item for item in result["rows"] if item["row_id"] == row["row_id"])
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
                yield self._config(obj), bundle, authority, ppa_evidence_specs
            finally:
                M.TRUSTED_HAMMER_AUTHORITIES = previous

    def test_01_canonical_zero_authority_bundle_row_and_headline(self):
        result = M.build(CONFIG)
        self.assertEqual(0, result["trusted_hammer_authority_count"])
        self.assertEqual(0, result["table_a_evidence_bundle_count"])
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])

    def test_02_runtime_required_scope_is_exact_ten(self):
        self.assertEqual(["patch_embed", "Conv2d", "ConvTranspose2d", "fc1", "fc2",
                          "dynamic_BN", "ATLIF", "attention", "prediction_head",
                          "all_required_preprocess_and_completion"], M._required_operator_scope())

    def test_03_future_exact_graph_reaches_six_rows(self):
        with self._rooted_positive_fixture() as (config_path, _, _, _):
            result = M.build(config_path)
        self.assertEqual(6, result["headline_gate"]["eligible_row_count"])
        self.assertTrue(result["headline_gate"]["admitted"])
        self.assertTrue(result["headline_gate"]["m527_required_operator_scope_gate"])
        self.assertTrue(result["headline_gate"]["synopsys_ppa_provenance_gate"])

    def test_04_trace_scope_missing_fc1_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, _):
            measurement = self._load_spec(bundle["measurement_identity"])
            trace_spec = measurement["complete_trace_manifest"]
            trace = self._load_spec(trace_spec)
            trace["operator_scope"].remove("fc1")
            self._rewrite_json(trace_spec, trace)
            measurement["complete_trace_manifest"] = trace_spec
            self._rewrite_json(bundle["measurement_identity"], measurement)
            with self.assertRaisesRegex(M.RegistryError, "complete trace operator_scope"):
                M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)

    def test_05_measurement_scope_missing_fc1_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, _):
            measurement = self._load_spec(bundle["measurement_identity"])
            measurement["operator_ids"].remove("fc1")
            self._rewrite_json(bundle["measurement_identity"], measurement)
            with self.assertRaisesRegex(M.RegistryError, "measurement operator_ids"):
                M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)

    def test_06_numerator_scope_missing_fc1_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, _):
            measurement = M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)
            spec = bundle["fixed_throughput_numerator_receipt"]
            numerator = self._load_spec(spec)
            numerator["included_operator_scope"].remove("fc1")
            self._rewrite_json(spec, numerator)
            with self.assertRaises(M.RegistryError):
                M._validate_numerator_receipt(spec, measurement)

    def test_07_handwritten_three_line_numeric_report_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            extraction_spec = evidence[row_id]["extraction"]
            receipt = self._load_spec(extraction_spec)
            dc_spec = receipt["raw_reports"]["dc_area"]
            self._rewrite_text(dc_spec, "logic_area_mm2 0.6\nlogic_power_mw 0.2\nsetup_wns_ns 0.0\n")
            receipt["raw_reports"]["dc_area"] = dc_spec
            receipt["extraction_argv"] = M._expected_argv(receipt["raw_reports"])
            self._rewrite_json(extraction_spec, receipt)
            with self.assertRaisesRegex(M.RegistryError, "not an accepted Synopsys report"):
                M._validate_extraction_receipt(extraction_spec, row_id,
                                               bundle["configuration_manifests"][row_id]["sha256"])

    def test_08_tool_version_substitution_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["extraction"]
            receipt = self._load_spec(spec)
            receipt["synopsys_tools"]["dc_area"]["version"] = "UNKNOWN"
            self._rewrite_json(spec, receipt)
            with self.assertRaisesRegex(M.RegistryError, "tool/version/library/corner"):
                M._validate_extraction_receipt(spec, row_id,
                                               bundle["configuration_manifests"][row_id]["sha256"])

    def test_09_extraction_argv_substitution_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["extraction"]
            receipt = self._load_spec(spec)
            receipt["extraction_argv"] = ["python3", "unreviewed.py"]
            self._rewrite_json(spec, receipt)
            with self.assertRaisesRegex(M.RegistryError, "argv"):
                M._validate_extraction_receipt(spec, row_id,
                                               bundle["configuration_manifests"][row_id]["sha256"])

    def test_10_library_or_corner_substitution_rejects(self):
        for field in ("libraries", "corners"):
            with self._rooted_positive_fixture() as (_, bundle, _, evidence):
                row_id = M.MANDATORY_ROW_IDS[0]
                spec = evidence[row_id]["extraction"]
                receipt = self._load_spec(spec)
                receipt[field]["dc_area"] = "unbound"
                if field == "libraries":
                    receipt["library_identity_sha256"] = M.R5._map_sha(receipt[field])
                self._rewrite_json(spec, receipt)
                with self.assertRaisesRegex(M.RegistryError, "tool/version/library/corner"):
                    M._validate_extraction_receipt(spec, row_id,
                                                   bundle["configuration_manifests"][row_id]["sha256"])

    def test_11_extracted_value_substitution_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["extraction"]
            receipt = self._load_spec(spec)
            receipt["extracted_values"]["logic_area_mm2"] = 9999.0
            self._rewrite_json(spec, receipt)
            with self.assertRaisesRegex(M.RegistryError, "does not project bound raw report"):
                M._validate_extraction_receipt(spec, row_id,
                                               bundle["configuration_manifests"][row_id]["sha256"])

    def test_12_ppa_evidence_map_contains_all_bound_roots(self):
        with self._rooted_positive_fixture() as (_, bundle, _, _):
            evidence = M._collect_ppa_evidence(bundle["ppa_receipt"])
        self.assertEqual(1 + 6 * 5, len(evidence))
        self.assertEqual(M.EXTRACTOR_SHA256, evidence["ppa_extractor_source"])

    def test_13_canonical_authority_map_empty_and_config_cannot_add(self):
        self.assertEqual({}, M.TRUSTED_HAMMER_AUTHORITIES)
        obj = copy.deepcopy(self.base)
        obj["trusted_hammer_authorities"] = {}
        with self.assertRaisesRegex(M.RegistryError, "fields differ"):
            M.build(self._config(obj))

    def test_14_frozen_roots_and_docs359_unchanged(self):
        self.assertEqual(M.R5_BUILDER_SHA256, M._sha256(M.R5_BUILDER))
        self.assertEqual(M.EXTRACTOR_SHA256, M._sha256(M.EXTRACTOR))
        self.assertEqual(M.M527_CONTRACT_SHA256, M._sha256(M.M527_CONTRACT))
        self.assertEqual(M.CHECKPOINT_SHA256, M._sha256(M.CHECKPOINT))
        self.assertEqual(M.DOCS359_SHA256, M._sha256(M.DOCS359))

    def test_15_nonfinite_overlay_rejects(self):
        path = Path(tempfile.mkstemp(suffix=".json")[1])
        self.addCleanup(self._remove, path)
        path.write_text('{"schema":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(M.RegistryError, "non-finite"):
            M.build(path)


if __name__ == "__main__":
    unittest.main()

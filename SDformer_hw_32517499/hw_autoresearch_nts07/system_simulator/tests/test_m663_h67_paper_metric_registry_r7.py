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
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m663_h67_paper_metric_registry_r7.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m663_h67_paper_metric_registry_r7_20260828.json"
R6_TESTS = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests/test_m658_h67_paper_metric_registry_r6.py"
NATIVE_DC_AREA = REPO_ROOT / "hw_autoresearch_nts07/dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823/reports/area.rpt"
NATIVE_PT_SETUP = REPO_ROOT / "hw_autoresearch_nts07/dc_handoff/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826/ptsta/reports/timing_setup_slow.rpt"
NATIVE_PT_HOLD = REPO_ROOT / "hw_autoresearch_nts07/dc_handoff/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826/ptsta/reports/timing_hold_fast.rpt"
NATIVE_PTPX = REPO_ROOT / "hw_autoresearch_nts07/dc_handoff/runs/m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r4_20260826/reports/ptpx_power_primary_100ps.rpt"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load("m663_registry", SCRIPT)
T6 = _load("m663_fixture_source_r6", R6_TESTS)


class M663RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = M.M635.load_json(CONFIG, "M663 test config")
        T6.M658RegistryTests.setUpClass()

    @staticmethod
    def _remove(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _config(self, value):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                             delete=False)
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
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")), encoding="utf-8")
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
        manifest_path.write_text("%s  %s\n" %
                                 (document_spec["sha256"], Path(document_spec["path"]).name),
                                 encoding="utf-8")
        manifest = self._spec(manifest_path, "text/plain")
        outer_path = Path(directory) / "REQUEST_SHA256SUMS.seal.sha256"
        outer_path.write_text("%s  %s\n" % (manifest["sha256"], manifest_path.name),
                              encoding="utf-8")
        return manifest, self._spec(outer_path, "text/plain")

    def _seal_two(self, directory, first, second):
        manifest_path = Path(directory) / "SHA256SUMS"
        rows = sorted([(Path(first["path"]).name, first["sha256"]),
                       (Path(second["path"]).name, second["sha256"])])
        manifest_path.write_text("".join("%s  %s\n" % (digest, name)
                                         for name, digest in rows), encoding="utf-8")
        manifest = self._spec(manifest_path, "text/plain")
        outer_path = Path(directory) / "SHA256SUMS.seal.sha256"
        outer_path.write_text("%s  %s\n" % (manifest["sha256"], manifest_path.name),
                              encoding="utf-8")
        return manifest, self._spec(outer_path, "text/plain")

    @staticmethod
    def _native_texts(design, macro):
        dc_library = "tcbn28hpcplusbwp35p140ssg0p9v125c"
        dc = """****************************************
Report : area
Design : %s
Version: V-2023.12-SP3
Date   : Fri Aug 28 00:00:00 2026
****************************************

Library(s) Used:

    %s (File: /opt/tech/%s.db)

Number of cells: 1
Total cell area:                 600000.000000
Total area:                 undefined

1
""" % (design, dc_library, dc_library)
        power = """****************************************
Report : Averaged Power
	-significant_digits
	-nosplit
	-unit mW
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 00:00:00 2026
****************************************

                        Internal             Switching            Leakage              Total
Power Group             Power                Power                Power                Power               (     %%)   Attrs
----------------------------------------------------------------------------------------------------------------------------
clock_network           1.00000000e-01       2.00000000e-02       1.00000000e-02       1.30000000e-01      (39.39%%)
register                5.00000000e-02       1.00000000e-02       5.00000000e-03       6.50000000e-02      (19.70%%)
combinational           5.00000000e-02       2.00000000e-02       5.00000000e-03       7.50000000e-02      (22.73%%)
memory                  4.00000000e-02       1.00000000e-02       1.00000000e-02       6.00000000e-02      (18.18%%)

  Net Switching Power  = 6.00000000e-02        (18.18%%)
  Cell Internal Power  = 2.40000000e-01        (72.73%%)
  Cell Leakage Power   = 3.00000000e-02        ( 9.09%%)
                         -------------------------
Total Power            = 3.30000000e-01       (100.00%%)

1
""" % design
        setup = """****************************************
Report : timing
	-path_type full_clock_expanded
	-delay_type max
	-max_paths 1
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 00:00:00 2026
****************************************

  Path Type: max
  slack (MET)                                                    0.100000

1
""" % design
        hold = """****************************************
Report : timing
	-path_type full_clock_expanded
	-delay_type min
	-max_paths 1
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 00:00:00 2026
****************************************

  Path Type: min
  slack (MET)                                                    0.010000

1
""" % design
        macro_library = macro.lower()
        ds = """####*********************************************************************************************************************/
#### Software       : TSMC MEMORY COMPILER tsn28hpcpd127spsram_2012.02.00.d.180a */
#### Library Name   : %s (user specify : %s) */
#### Generated Time : 2026/08/28, 00:00:00 */
####*********************************************************************************************************************/

1. Area
  |       Width(um)      |       Height(um)        |    Area (um^2)     |
  |       1000.0000      |       400.0000          |      400000.0000   |

2. Timing Specification
   2.2 SRAM timing:(Slow, 0.9000, 125.0000 deg.)

4.1 Static Power
    Leakage Current                                             10.0000 (uA) 5.0000 (uA) 5.0000 (uA)

4.2 Dynamic Power - Average
    Read                                                   11.0000 (uA/MHz)
    Write                                                  12.0000 (uA/MHz)
""" % (macro_library, macro)
        return {"dc_area": dc, "ptpx_power": power, "pt_setup": setup,
                "pt_hold": hold, "sram_macro": ds}

    def _write_native_row(self, base_dir, row_id, configuration_sha):
        design = M._expected_design(row_id)
        macro = "M663_%s_SRAM" % row_id.upper()
        texts = self._native_texts(design, macro)
        report_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                         for name, text in texts.items()}
        run_id = M._expected_run_id(row_id, configuration_sha, report_hashes)
        run_dir = Path(base_dir) / run_id
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=False)
        report_specs = {}
        names = {"dc_area": "area.rpt", "ptpx_power": "power.rpt",
                 "pt_setup": "timing_setup.rpt", "pt_hold": "timing_hold.rpt",
                 "sram_macro": "macro.ds"}
        for name, text in texts.items():
            path = reports_dir / names[name]
            path.write_text(text, encoding="utf-8")
            report_specs[name] = self._spec(path, "text/plain")
        tools = {
            "dc_area": {"tool": "dc_shell", "version": "V-2023.12-SP3"},
            "ptpx_power": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
            "pt_setup": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
            "pt_hold": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
            "sram_macro": {"tool": "memory_compiler",
                           "version": "tsn28hpcpd127spsram_2012.02.00.d.180a"},
        }
        libraries = {"dc_area": "tcbn28hpcplusbwp35p140ssg0p9v125c",
                     "ptpx_power": "tcbn28hpcplusbwp30p140tt0p9v25c",
                     "pt_setup": "tcbn28hpcplusbwp35p140ssg0p9v125c",
                     "pt_hold": "tcbn28hpcplusbwp35p140ffg1p05vm40c",
                     "sram_macro": macro.lower()}
        corners = {"dc_area": "ssg0p9v125c", "ptpx_power": "tt0p9v25c",
                   "pt_setup": "ssg0p9v125c", "pt_hold": "ffg1p05vm40c",
                   "sram_macro": "slow_0p9v125c"}
        manifest = {
            "schema": "m663.h67.native_synopsys_run_manifest.r1",
            "status": "FROZEN_NATIVE_REPORTS", "row_id": row_id,
            "configuration_manifest_sha256": configuration_sha,
            "m527_configuration_id": M.ROW_TO_M527_CONFIGURATION[row_id],
            "operator_scope_sha256": M._map_sha(M._required_operator_scope()),
            "design_name": design, "macro_name": macro, "run_id": run_id,
            "raw_reports": report_specs, "tools": tools,
            "libraries": libraries, "corners": corners,
        }
        manifest_path = run_dir / "native_run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        manifest_spec = self._spec(manifest_path, "application/json")
        extracted = M.EX.extract_from_manifest(manifest_path)
        units = {field: ("mm2" if field.endswith("area_mm2") else
                         "ns" if field.endswith("wns_ns") else "mW")
                 for field in M.EXTRACTED_FIELDS}
        receipt = {
            "schema": "m663.h67.native_synopsys_ppa_extraction_receipt.r1",
            "status": "PASS_DIRECT_NATIVE_REPORT_PARSE", "row_id": row_id,
            "configuration_manifest_sha256": configuration_sha,
            "run_manifest": manifest_spec,
            "extractor_source": {"path": M.EXTRACTOR.relative_to(REPO_ROOT).as_posix(),
                                 "sha256": M.EXTRACTOR_SHA256,
                                 "media_type": "text/x-python"},
            "extraction_argv": M._expected_argv(manifest_spec),
            "raw_reports": report_specs, "native_identities": extracted["identities"],
            "tools": tools, "libraries": libraries, "corners": corners,
            "units": units, "extracted_values": extracted["values"],
        }
        receipt_path = run_dir / "native_extraction_receipt.json"
        receipt_path.write_text(json.dumps(receipt, separators=(",", ":")), encoding="utf-8")
        receipt_spec = self._spec(receipt_path, "application/json")
        values = dict(extracted["values"])
        values.update({"row_id": row_id, "configuration_manifest_sha256": configuration_sha,
                       "total_area_mm2": (values["logic_area_mm2"] +
                                          values["sram_macro_area_mm2"]),
                       "extraction_receipt": receipt_spec})
        return values, {"manifest": manifest_spec, "receipt": receipt_spec,
                        "reports": report_specs}

    @contextlib.contextmanager
    def _rooted_positive_fixture(self):
        helper = T6.M658RegistryTests(methodName="test_01_canonical_zero_authority_bundle_row_and_headline")
        helper.base = T6.M658RegistryTests.base
        with helper._rooted_positive_fixture() as (old_config, old_bundle, _, _):
            bundle = copy.deepcopy(old_bundle)
            bundle_id = bundle["bundle_id"]
            results_dir = (REPO_ROOT / bundle["ppa_receipt"]["path"]).parent
            row_evidence = {}
            ppa_rows = []
            for row_id in M.MANDATORY_ROW_IDS:
                row, evidence = self._write_native_row(
                    results_dir, row_id, bundle["configuration_manifests"][row_id]["sha256"])
                ppa_rows.append(row)
                row_evidence[row_id] = evidence
            ppa = {"schema": "m663.h67.native_synopsys_rooted_ppa_receipt.r1",
                   "status": "PASS_DIRECT_NATIVE_REPORT_PARSE", "technology_nm": 28,
                   "clock_period_ns": 3.0, "rows": ppa_rows}
            self._rewrite_json(bundle["ppa_receipt"], ppa)

            result = self._load_spec(bundle["direct_result"])
            result["ppa_receipt_sha256"] = bundle["ppa_receipt"]["sha256"]
            for row in result["rows"]:
                row["area_mm2"] = next(item["total_area_mm2"] for item in ppa_rows
                                        if item["row_id"] == row["row_id"])
            self._rewrite_json(bundle["direct_result"], result)

            completion = self._load_spec(bundle["completion_receipt"])
            completion["direct_result_sha256"] = bundle["direct_result"]["sha256"]
            completion["ppa_receipt_sha256"] = bundle["ppa_receipt"]["sha256"]
            self._rewrite_json(bundle["completion_receipt"], completion)
            coverage = self._load_spec(bundle["coverage_receipt"])
            coverage["direct_result_sha256"] = bundle["direct_result"]["sha256"]
            self._rewrite_json(bundle["coverage_receipt"], coverage)

            bundle["schema"] = "m663.h67.rooted_direct_bundle.r4"
            raw_index = self._load_spec(bundle["raw_run_index"])
            evidence = {
                "m527_contract": bundle["m527_contract"]["sha256"],
                "checkpoint": bundle["checkpoint"]["sha256"],
                "fixed_throughput_numerator_receipt":
                    bundle["fixed_throughput_numerator_receipt"]["sha256"],
                "common_resource_manifest": bundle["common_resource_manifest"]["sha256"],
                "measurement_identity": bundle["measurement_identity"]["sha256"],
                "raw_run_index": bundle["raw_run_index"]["sha256"],
                "direct_result": bundle["direct_result"]["sha256"],
                "completion_receipt": bundle["completion_receipt"]["sha256"],
                "coverage_receipt": bundle["coverage_receipt"]["sha256"],
                "ppa_receipt": bundle["ppa_receipt"]["sha256"],
                "energy_receipt": bundle["energy_receipt"]["sha256"],
                "accuracy_receipt": bundle["accuracy_receipt"]["sha256"],
                "m527_required_operator_scope": M._map_sha(M._required_operator_scope()),
            }
            measurement = self._load_spec(bundle["measurement_identity"])
            evidence["m527_complete_trace_manifest"] = measurement["complete_trace_manifest"]["sha256"]
            evidence.update(M._collect_ppa_evidence(bundle["ppa_receipt"]))
            for row_id, spec in bundle["configuration_manifests"].items():
                evidence["configuration_manifest:" + row_id] = spec["sha256"]
            for index, item in enumerate(raw_index["runs"]):
                evidence["raw_log:%06d" % index] = item["log"]["sha256"]
            evidence = {key: evidence[key] for key in sorted(evidence)}

            reviews_dir = (REPO_ROOT / bundle["independent_hammer_receipt"]["path"]).parent
            targets = {}
            fixed = {"registry_builder": Path(M.__file__), "registry_config": M.DEFAULT_CONFIG,
                     "registry_tests": M.REGISTRY_TESTS, "registry_contract": M.REGISTRY_CONTRACT,
                     "m527_contract": M.M527_CONTRACT, "checkpoint": M.CHECKPOINT}
            for name, path in fixed.items():
                if name == "checkpoint":
                    targets[name] = copy.deepcopy(bundle["checkpoint"])
                else:
                    media = "text/x-python" if name in ("registry_builder", "registry_tests") else "application/json"
                    targets[name] = self._spec(path, media)
            targets["direct_result"] = copy.deepcopy(bundle["direct_result"])
            targets["fixed_throughput_numerator_receipt"] = copy.deepcopy(
                bundle["fixed_throughput_numerator_receipt"])
            target_shas = {key: targets[key]["sha256"] for key in sorted(targets)}
            authority_id = "test_only_m663_native_registry_authority"
            request_path = reviews_dir / "m663_request.json"
            request_value = {"schema": "m653.h67.direct_unified.hammer_request.r1",
                             "status": "FROZEN_BEFORE_REVIEW", "authority_id": authority_id,
                             "bundle_id": bundle_id, "reviewed_targets": targets,
                             "bundle_evidence_sha256": evidence,
                             "complete_evidence_root_sha256": M._map_sha(evidence)}
            request_path.write_text(json.dumps(request_value, separators=(",", ":")), encoding="utf-8")
            request = self._spec(request_path, "application/json")
            request_manifest, request_outer = self._seal_one(reviews_dir, request)

            numerator = self._load_spec(bundle["fixed_throughput_numerator_receipt"])
            hammer_path = reviews_dir / "m663_hammer.json"
            hammer_value = {"schema": "m653.h67.direct_unified.independent_hammer.r1",
                            "status": "PASS_INDEPENDENT", "authority_id": authority_id,
                            "request_outer_seal_sha256": request_outer["sha256"],
                            "reviewed_targets_sha256": target_shas,
                            "bundle_evidence_sha256": evidence,
                            "severity_counts": {"P0": 0, "P1": 0},
                            "independence": {"author_receipt_used_as_authority": False,
                                             "raw_logs_rehashed_and_recomputed": True,
                                             "fixed_numerators_rehashed_and_recomputed": True,
                                             "typed_receipts_recomputed": True,
                                             "raw_ppa_reports_parsed_and_projected": True,
                                             "result_modified": False},
                            "recomputed_fixed_throughput_numerators": {
                                "dense_equivalent_ops_scalar": numerator["dense_equivalent_ops_scalar"],
                                "original_useful_nonzero_ops_scalar":
                                    numerator["original_useful_nonzero_ops_scalar"]},
                            "recomputed_rows": result["rows"],
                            "recomputed_aggregates": result["aggregates"],
                            "recomputed_views": result["views"],
                            "authorization": {"table_a_methodology_admitted": True,
                                              "direct_unified_measurement_admitted": True,
                                              "paper_headline_admitted": True}}
            hammer_path.write_text(json.dumps(hammer_value, separators=(",", ":")), encoding="utf-8")
            hammer = self._spec(hammer_path, "application/json")
            review_path = reviews_dir / "m663_review.json"
            review_value = {"schema": "m653.h67.direct_unified.hammer_review.r1",
                            "status": "COMPLETE", "authority_id": authority_id,
                            "request_outer_seal_sha256": request_outer["sha256"],
                            "reviewed_targets_sha256": target_shas,
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
                yield self._config(obj), bundle, authority, row_evidence
            finally:
                M.TRUSTED_HAMMER_AUTHORITIES = previous
        helper.doCleanups()

    def test_01_canonical_zero_authority_bundle_row_and_headline(self):
        result = M.build(CONFIG)
        self.assertEqual((0, 0, 0, False),
                         (result["trusted_hammer_authority_count"],
                          result["table_a_evidence_bundle_count"],
                          result["headline_gate"]["eligible_row_count"],
                          result["headline_gate"]["admitted"]))

    def test_02_runtime_required_scope_is_exact_ten(self):
        self.assertEqual(10, len(M._required_operator_scope()))
        self.assertIn("fc1", M._required_operator_scope())

    def test_03_repo_native_dc_area_parses_directly(self):
        identity, area = M.EX.parse_dc_area(NATIVE_DC_AREA)
        self.assertEqual("qfit_head_p48_signed_lane_fold", identity["design"])
        self.assertAlmostEqual(0.03545917173, area, places=12)

    def test_04_repo_native_pt_setup_and_hold_parse_directly(self):
        setup = M.EX.parse_pt_timing(NATIVE_PT_SETUP, "max")
        hold = M.EX.parse_pt_timing(NATIVE_PT_HOLD, "min")
        self.assertEqual("m433_exact_dualbank_coread_pwp_adapter", setup[0]["design"])
        self.assertAlmostEqual(0.841061, setup[1], places=6)
        self.assertAlmostEqual(0.017869, hold[1], places=6)

    def test_05_repo_native_ptpx_power_parses_leakage_and_total(self):
        identity, values = M.EX.parse_ptpx_power(NATIVE_PTPX)
        self.assertEqual("m405_q32_elastic_selected_slice", identity["design"])
        self.assertGreater(values["total_leakage_power_mw"], 0.0)
        self.assertAlmostEqual(values["total_power_mw"],
                               values["total_internal_power_mw"] +
                               values["total_switching_power_mw"] +
                               values["total_leakage_power_mw"], places=6)

    def test_06_future_native_graph_reaches_six_rows(self):
        with self._rooted_positive_fixture() as (config_path, _, _, _):
            result = M.build(config_path)
        self.assertEqual(6, result["headline_gate"]["eligible_row_count"])
        self.assertTrue(result["headline_gate"]["admitted"])

    def test_07_wrong_design_rejects_after_report_rehash(self):
        with self._rooted_positive_fixture() as (_, _, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            manifest_spec = evidence[row_id]["manifest"]
            manifest = self._load_spec(manifest_spec)
            dc_spec = manifest["raw_reports"]["dc_area"]
            path = REPO_ROOT / dc_spec["path"]
            text = path.read_text(encoding="utf-8").replace(M._expected_design(row_id),
                                                            "wrong_configuration")
            self._rewrite_text(dc_spec, text)
            manifest["raw_reports"]["dc_area"] = dc_spec
            self._rewrite_json(manifest_spec, manifest)
            with self.assertRaisesRegex(M.EX.ExtractionError, "manifest design"):
                M.EX.extract_from_manifest(REPO_ROOT / manifest_spec["path"])

    def test_08_cross_row_report_reuse_rejects(self):
        with self._rooted_positive_fixture() as (_, _, _, evidence):
            source = M.MANDATORY_ROW_IDS[0]
            target = M.MANDATORY_ROW_IDS[1]
            source_manifest = self._load_spec(evidence[source]["manifest"])
            target_spec = evidence[target]["manifest"]
            target_manifest = self._load_spec(target_spec)
            for name in ("dc_area", "ptpx_power", "pt_setup", "pt_hold"):
                target_manifest["raw_reports"][name] = source_manifest["raw_reports"][name]
            self._rewrite_json(target_spec, target_manifest)
            with self.assertRaisesRegex(M.EX.ExtractionError, "manifest design"):
                M.EX.extract_from_manifest(REPO_ROOT / target_spec["path"])

    def test_09_three_line_numeric_wrapper_rejects(self):
        path = Path(tempfile.mkstemp(suffix=".rpt")[1])
        self.addCleanup(self._remove, path)
        path.write_text("logic_area_mm2 0.6\nlogic_power_mw 0.2\nsetup_wns_ns 0.0\n",
                        encoding="utf-8")
        with self.assertRaises(M.EX.ExtractionError):
            M.EX.parse_dc_area(path)

    def test_10_missing_leakage_rejects(self):
        with self._rooted_positive_fixture() as (_, _, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            manifest_spec = evidence[row_id]["manifest"]
            manifest = self._load_spec(manifest_spec)
            power_spec = manifest["raw_reports"]["ptpx_power"]
            path = REPO_ROOT / power_spec["path"]
            text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                             if "Cell Leakage Power" not in line) + "\n"
            self._rewrite_text(power_spec, text)
            manifest["raw_reports"]["ptpx_power"] = power_spec
            self._rewrite_json(manifest_spec, manifest)
            with self.assertRaisesRegex(M.EX.ExtractionError, "leakage total"):
                M.EX.extract_from_manifest(REPO_ROOT / manifest_spec["path"])

    def test_11_total_power_arithmetic_drift_rejects(self):
        path = Path(tempfile.mkstemp(suffix=".rpt")[1])
        self.addCleanup(self._remove, path)
        text = self._native_texts("x", "X")["ptpx_power"].replace(
            "Total Power            = 3.30000000e-01", "Total Power            = 9.90000000e-01")
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(M.EX.ExtractionError, "internal\+switching\+leakage"):
            M.EX.parse_ptpx_power(path)

    def test_12_macro_identity_drift_rejects(self):
        with self._rooted_positive_fixture() as (_, _, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            manifest_spec = evidence[row_id]["manifest"]
            manifest = self._load_spec(manifest_spec)
            manifest["macro_name"] = "WRONG_MACRO"
            self._rewrite_json(manifest_spec, manifest)
            with self.assertRaisesRegex(M.EX.ExtractionError, "manifest macro"):
                M.EX.extract_from_manifest(REPO_ROOT / manifest_spec["path"])

    def test_13_run_manifest_report_set_omission_rejects(self):
        with self._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["manifest"]
            manifest = self._load_spec(spec)
            del manifest["raw_reports"]["pt_hold"]
            self._rewrite_json(spec, manifest)
            receipt_spec = evidence[row_id]["receipt"]
            receipt = self._load_spec(receipt_spec)
            receipt["run_manifest"] = spec
            self._rewrite_json(receipt_spec, receipt)
            with self.assertRaisesRegex(M.RegistryError, "raw report set"):
                M._validate_extraction_receipt(receipt_spec, row_id,
                                               bundle["configuration_manifests"][row_id]["sha256"])

    def test_14_typed_total_power_or_leakage_drift_rejects(self):
        for field in ("total_leakage_power_mw", "total_power_mw"):
            with self._rooted_positive_fixture() as (_, bundle, _, evidence):
                row_id = M.MANDATORY_ROW_IDS[0]
                spec = evidence[row_id]["receipt"]
                receipt = self._load_spec(spec)
                receipt["extracted_values"][field] += 1.0
                self._rewrite_json(spec, receipt)
                with self.assertRaisesRegex(M.RegistryError, "value drift"):
                    M._validate_extraction_receipt(
                        spec, row_id, bundle["configuration_manifests"][row_id]["sha256"])

    def test_15_evidence_map_binds_six_manifests_receipts_and_thirty_reports(self):
        with self._rooted_positive_fixture() as (_, bundle, _, _):
            evidence = M._collect_ppa_evidence(bundle["ppa_receipt"])
        self.assertEqual(1 + 6 * 7, len(evidence))
        self.assertEqual(6, len([key for key in evidence if key.startswith("ppa_run_manifest:")]))
        self.assertEqual(30, len([key for key in evidence if key.startswith("ppa_native_report:")]))

    def test_16_wrong_review_target_rejects(self):
        with self._rooted_positive_fixture() as (config_path, _, authority, _):
            path = REPO_ROOT / authority["request_document"]["path"]
            request = json.loads(path.read_text(encoding="utf-8"))
            request["reviewed_targets"]["registry_contract"] = self._spec(CONFIG, "application/json")
            path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            authority["request_document"] = self._spec(path, "application/json")
            authority["request_manifest"], authority["request_outer_seal"] = self._seal_one(
                path.parent, authority["request_document"])
            with self.assertRaisesRegex(M.RegistryError, "target path mismatch"):
                M.build(config_path)

    def test_17_frozen_roots_and_docs359_unchanged(self):
        self.assertEqual(M.R6_BUILDER_SHA256, M._sha256(M.R6_BUILDER))
        self.assertEqual(M.EXTRACTOR_SHA256, M._sha256(M.EXTRACTOR))
        self.assertEqual(M.M527_CONTRACT_SHA256, M._sha256(M.M527_CONTRACT))
        self.assertEqual(M.CHECKPOINT_SHA256, M._sha256(M.CHECKPOINT))
        self.assertEqual(M.DOCS359_SHA256, M._sha256(M.DOCS359))

    def test_18_nonfinite_overlay_rejects(self):
        path = Path(tempfile.mkstemp(suffix=".json")[1])
        self.addCleanup(self._remove, path)
        path.write_text('{"schema":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(M.RegistryError, "non-finite"):
            M.build(path)


if __name__ == "__main__":
    unittest.main()

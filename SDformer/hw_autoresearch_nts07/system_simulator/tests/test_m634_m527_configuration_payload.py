#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_verify_m634_m527_configuration_payload.py"
SPEC = importlib.util.spec_from_file_location("m634_payload", SCRIPT)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M634PayloadTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)
        self.trace = self._file("trace.json", {"rows": 3})
        self.population = self._file("population.json", {"frames": 3})
        self.weights = self._file("weights.json", {"weights": [1, 1, 1]})
        self.simulator = self.root / "simulator.py"
        self.simulator.write_text("# exact fixture simulator\n", encoding="utf-8")
        self.operator_ids = ["patch", "conv0", "decoder0"]
        self.measurement = self._file("measurement.json", {
            "schema": M.MEASUREMENT_SCHEMA,
            "m527_contract_sha256": M.M527_CONTRACT_SHA256,
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "complete_trace_manifest": self._ref(self.trace),
            "sequence_population_manifest": self._ref(self.population),
            "aggregation_weight_manifest": self._ref(self.weights),
            "frame_definition": "one frozen H67 network invocation",
            "density_metric": "event_count_per_frame",
            "density_bin_boundaries": [0, 10, 20, 100],
            "operator_ids": self.operator_ids,
        })
        self.common_obj = {
            "schema": M.COMMON_SOURCE_SCHEMA,
            "resource_tuple": {
                "technology_nm": 28, "clock_period_ns": 3.0,
                "source_lanes": 96, "service_width_sources_per_cycle": 8,
                "onchip_sram_bytes_total": 245760,
                "dram_bandwidth_bytes_per_second_decimal": 64000000000,
                "dram_bytes_per_cycle": 192, "accumulator_bits": 24,
                "source_queue_depth": 8, "completion_queue_depth": 8,
                "parent_queue_depth": 8, "weight_sram_bank_count": 8,
                "state_sram_bank_count": 8, "parent_scratch_bank_count": 1,
                "weight_sram_port_mode": "1R1W", "state_sram_port_mode": "1R1W",
                "parent_scratch_port_mode": "1R1W",
                "external_read_port_count": 1, "external_write_port_count": 1,
            },
            "charge_policy": {key: True for key in M.CHARGE_FIELDS},
            "fallback_policy": {
                "mode": "EXECUTE_UNSUPPORTED_WORK_IN_THE_SAME_UNIFIED_MODEL",
                "must_charge_cycles": True, "must_charge_traffic": True,
                "must_charge_energy": True, "must_charge_area": True,
            },
        }
        self.common = self._file("common.json", self.common_obj)
        self.configs = {}
        for config_id in M.CONFIGURATION_IDS:
            self.configs[config_id] = self._file(config_id + "_source.json", {
                "schema": M.CONFIG_SOURCE_SCHEMA,
                "configuration_id": config_id,
                "mechanism_enable_map": M.EXPECTED_MECHANISMS[config_id],
                "optimized_operator_ids": ["conv0"],
                "unsupported_operator_ids": ["patch", "decoder0"],
            })

    def tearDown(self):
        self.temp.cleanup()

    def _file(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _ref(path):
        return {"path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def build(self, output_name="payload"):
        output = self.root / output_name
        result = M.build_payload(self.measurement, self.common, self.simulator,
                                 self.configs, output)
        return output, result

    def test_complete_payload_builds_and_revalidates(self):
        output, result = self.build()
        self.assertEqual(5, result["configuration_count"])
        self.assertFalse(result["paper_headline"])
        again = M.verify_payload(output)
        self.assertEqual(result["member_manifest_sha256"], again["member_manifest_sha256"])
        b2 = json.loads((output / "b2_exact_bit_sparse_k1.json").read_text())
        self.assertEqual(8, b2["resource_tuple"]["service_width_sources_per_cycle"])
        self.assertEqual(1, b2["mechanism_enable_map"]["execution_service_limit_sources_per_cycle"])

    def test_null_measurement_sha_rejected_before_output(self):
        value = json.loads(self.measurement.read_text())
        value["complete_trace_manifest"]["sha256"] = None
        self.measurement.write_text(json.dumps(value) + "\n", encoding="utf-8")
        output = self.root / "must_not_exist"
        with self.assertRaisesRegex(M.PayloadError, "non-null lowercase SHA256"):
            M.build_payload(self.measurement, self.common, self.simulator, self.configs, output)
        self.assertFalse(output.exists())

    def test_measurement_source_mutation_rejected(self):
        self.trace.write_text('{"rows":4}\n', encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "complete_trace_manifest SHA mismatch"):
            self.build()

    def test_config_partition_gap_rejected(self):
        path = self.configs[M.CONFIGURATION_IDS[0]]
        value = json.loads(path.read_text())
        value["unsupported_operator_ids"] = ["patch"]
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "operator partition"):
            self.build()

    def test_resource_width_mismatch_rejected(self):
        value = copy.deepcopy(self.common_obj)
        value["resource_tuple"]["service_width_sources_per_cycle"] = 1
        self.common.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "service_width_sources_per_cycle mismatch"):
            self.build()

    def test_unpaid_added_resource_rejected(self):
        value = copy.deepcopy(self.common_obj)
        value["charge_policy"]["extra_matcher_area_charged"] = False
        self.common.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "every added resource"):
            self.build()

    def test_payload_tamper_rejected(self):
        output, _ = self.build()
        path = output / "c123_ours_exact.json"
        value = json.loads(path.read_text())
        value["claim_boundary"]["paper_headline"] = True
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "payload member SHA mismatch"):
            M.verify_payload(output)

    def test_resealed_unknown_overclaim_field_rejected(self):
        output, _ = self.build()
        path = output / "c123_ours_exact.json"
        value = json.loads(path.read_text())
        value["system_speedup_admitted"] = True
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        registry_path = output / "registry.json"
        registry = json.loads(registry_path.read_text())
        for entry in registry["configuration_manifests"]:
            if entry["configuration_id"] == "c123_ours_exact":
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        names = [item.name for item in output.iterdir()
                 if item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
        M._seal(output, names)
        with self.assertRaisesRegex(M.PayloadError, "fields drift"):
            M.verify_payload(output)

    def test_configuration_source_mutation_rejected_after_build(self):
        output, _ = self.build()
        path = self.configs[M.CONFIGURATION_IDS[-1]]
        value = json.loads(path.read_text())
        value["optimized_operator_ids"] = ["conv0", "patch"]
        value["unsupported_operator_ids"] = ["decoder0"]
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(M.PayloadError, "configuration source SHA drift"):
            M.verify_payload(output)


if __name__ == "__main__":
    unittest.main()

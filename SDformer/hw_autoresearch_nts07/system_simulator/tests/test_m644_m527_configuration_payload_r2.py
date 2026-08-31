#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_verify_m644_m527_configuration_payload_r2.py"
SPEC = importlib.util.spec_from_file_location("m644_payload", str(SCRIPT))
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M644PayloadTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        self.temp = tempfile.TemporaryDirectory(dir=str(parent))
        self.root = Path(self.temp.name)
        self.trace = self._file("trace.json", {
            "schema": "fixture_decoder_complete_trace_v1", "rows": 40,
            "decoder_rows": 40, "operators": ["patch", "conv0", "decoder0"],
        })
        self.population = self._file("population.json", {
            "schema": "fixture_population_v1", "frames": 3,
        })
        self.weights = self._file("weights.json", {
            "schema": "fixture_weights_v1", "weights": [1, 1, 1],
        })
        self.simulator = self.root / "simulator.py"
        self.simulator.write_text("# exact fixture unified simulator\n", encoding="utf-8")
        self.operator_ids = ["patch", "conv0", "decoder0"]
        self.upstream_obj = {
            "schema": M.UPSTREAM_SCHEMA,
            "status": M.UPSTREAM_STATUS,
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "complete_trace_manifest_sha256": self._ref(self.trace)["sha256"],
            "sequence_population_manifest_sha256": self._ref(self.population)["sha256"],
            "aggregation_weight_manifest_sha256": self._ref(self.weights)["sha256"],
            "population_scalar": 3,
            "population_unit": "frozen_frames_across_frozen_sequence_population",
            "frame_definition": "one frozen H67 decoder-complete network invocation",
            "density_metric": "event_count_per_frame",
            "density_bin_boundaries": [0, 10, 20, 100],
            "operator_ids": self.operator_ids,
            "verification": M.UPSTREAM_VERIFICATION,
            "claim_boundary": M.UPSTREAM_CLAIM_BOUNDARY,
        }
        self.upstream = self._file("upstream_receipt.json", self.upstream_obj)
        self.measurement_obj = {
            "schema": M.MEASUREMENT_SCHEMA,
            "m527_contract_sha256": M.M527_CONTRACT_SHA256,
            "checkpoint_sha256": self.upstream_obj["checkpoint_sha256"],
            "complete_trace_manifest": self._ref(self.trace),
            "sequence_population_manifest": self._ref(self.population),
            "aggregation_weight_manifest": self._ref(self.weights),
            "upstream_semantic_verification_receipt": self._ref(self.upstream),
            "frame_definition": self.upstream_obj["frame_definition"],
            "density_metric": self.upstream_obj["density_metric"],
            "density_bin_boundaries": self.upstream_obj["density_bin_boundaries"],
            "population_scalar": 3,
            "population_unit": self.upstream_obj["population_unit"],
            "operator_ids": self.operator_ids,
        }
        self.measurement = self._file("measurement.json", self.measurement_obj)
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
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _ref(path):
        return {"path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def build(self, name="payload"):
        output = self.root / name
        result = M.build_payload(self.measurement, self.common, self.simulator,
                                 self.configs, output)
        return output, result

    def verify(self, output):
        return M.verify_payload(output, self.measurement, self.common,
                                self.simulator, self.configs)

    def _load(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def _reseal_consistently(self, output):
        common_path = output / "common_resource_manifest.json"
        common_sha = hashlib.sha256(common_path.read_bytes()).hexdigest()
        registry_path = output / "registry.json"
        registry = self._load(registry_path)
        registry["common_resource_manifest"]["sha256"] = common_sha
        for config_id in M.CONFIGURATION_IDS:
            path = output / (config_id + ".json")
            value = self._load(path)
            value["common_resource_manifest_sha256"] = common_sha
            self._write(path, value)
            for entry in registry["configuration_manifests"]:
                if entry["configuration_id"] == config_id:
                    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write(registry_path, registry)
        names = [item.name for item in output.iterdir()
                 if item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
        M._seal(output, names)

    def test_positive_build_staging_publish_and_revalidate(self):
        output, result = self.build()
        self.assertEqual(5, result["configuration_count"])
        self.assertTrue(result["configuration_identity_payload_ready"])
        self.assertFalse(result["m527_configuration_registry_ready"])
        self.assertFalse(result["system_speedup"])
        self.assertEqual(result["member_manifest_sha256"],
                         self.verify(output)["member_manifest_sha256"])

    def test_a1_resealed_optimized_fallback_rebinding_attack_rejected(self):
        output, _ = self.build()
        path = output / "c123_ours_exact.json"
        value = self._load(path)
        value["optimized_operator_ids"] = list(self.operator_ids)
        value["fallback_policy"]["unsupported_operator_ids"] = []
        self._write(path, value)
        self._reseal_consistently(output)
        with self.assertRaisesRegex(M.PayloadError, "live-source reconstruction"):
            self.verify(output)

    def test_a2_resealed_waterfall_overclaim_rejected(self):
        output, _ = self.build()
        path = output / "c123_ours_exact.json"
        value = self._load(path)
        value["claim_boundary"]["waterfall_admitted"] = True
        self._write(path, value)
        self._reseal_consistently(output)
        with self.assertRaisesRegex(M.PayloadError, "live-source reconstruction"):
            self.verify(output)

    def test_a3_resealed_registry_identity_and_gate_attack_rejected(self):
        output, _ = self.build()
        path = output / "registry.json"
        value = self._load(path)
        value["m527_contract_sha256"] = "0" * 64
        value["common_resource_manifest"]["path"] = "missing.json"
        value["configuration_identity_payload_ready"] = False
        value["m527_contract_admission_gate_current_value"] = True
        self._write(path, value)
        names = [item.name for item in output.iterdir()
                 if item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
        M._seal(output, names)
        with self.assertRaisesRegex(M.PayloadError, "registry differs"):
            self.verify(output)

    def test_a4_resealed_receipt_overclaim_and_proof_attack_rejected(self):
        output, _ = self.build()
        path = output / "verification_receipt.json"
        value = self._load(path)
        value["status"] = "HEADLINE_ADMITTED"
        value["validated_configuration_ids"] = []
        value["all_live_source_paths_and_hashes_verified"] = False
        value["claim_boundary"]["system_speedup"] = True
        self._write(path, value)
        self._reseal_consistently(output)
        with self.assertRaisesRegex(M.PayloadError, "exact non-admission receipt"):
            self.verify(output)

    def test_a5_resealed_configuration_path_attack_rejected(self):
        output, _ = self.build()
        for config_id in M.CONFIGURATION_IDS:
            path = output / (config_id + ".json")
            value = self._load(path)
            value["configuration_source_path"] = "missing/config.json"
            value["simulator_source_path"] = "missing/simulator.py"
            value["complete_trace_manifest_path"] = "missing/trace.json"
            value["measurement_identity_manifest_path"] = "missing/measurement.json"
            self._write(path, value)
        self._reseal_consistently(output)
        with self.assertRaisesRegex(M.PayloadError, "live-source reconstruction"):
            self.verify(output)

    def test_a6_resealed_common_embedded_identity_attack_rejected(self):
        output, _ = self.build()
        path = output / "common_resource_manifest.json"
        value = self._load(path)
        value["measurement_identity"]["checkpoint_sha256"] = "0" * 64
        value["measurement_identity"]["operator_ids"] = []
        self._write(path, value)
        self._reseal_consistently(output)
        with self.assertRaisesRegex(M.PayloadError, "common resource manifest differs"):
            self.verify(output)

    def test_a7_upstream_population_identity_attack_rejected(self):
        value = copy.deepcopy(self.upstream_obj)
        value["population_scalar"] = 999
        self._write(self.upstream, value)
        measurement = copy.deepcopy(self.measurement_obj)
        measurement["upstream_semantic_verification_receipt"] = self._ref(self.upstream)
        self._write(self.measurement, measurement)
        with self.assertRaisesRegex(M.PayloadError, "upstream receipt identity drift: population_scalar"):
            self.build()

    def test_a7_upstream_schema_status_and_operator_universe_are_bound(self):
        attacks = (("schema", "wrong_schema"), ("status", "HEADLINE"),
                   ("operator_ids", ["patch"]))
        for index, (field, replacement) in enumerate(attacks):
            with self.subTest(field=field):
                value = copy.deepcopy(self.upstream_obj)
                value[field] = replacement
                path = self._file("upstream_attack_{}.json".format(index), value)
                measurement = copy.deepcopy(self.measurement_obj)
                measurement["upstream_semantic_verification_receipt"] = self._ref(path)
                measurement_path = self._file("measurement_attack_{}.json".format(index), measurement)
                with self.assertRaises(M.PayloadError):
                    M.build_payload(measurement_path, self.common, self.simulator,
                                    self.configs, self.root / "bad_upstream_{}".format(index))

    def test_a8_output_symlink_ancestor_rejected(self):
        real_parent = self.root / "real_parent"
        real_parent.mkdir()
        symlink_parent = self.root / "symlink_parent"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        output = symlink_parent / "payload"
        with self.assertRaisesRegex(M.PayloadError, "symlink path component refused"):
            M.build_payload(self.measurement, self.common, self.simulator,
                            self.configs, output)
        self.assertFalse((real_parent / "payload").exists())

    def test_output_outside_repo_and_dangling_leaf_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(M.PayloadError, "path escapes repository"):
                M.build_payload(self.measurement, self.common, self.simulator,
                                self.configs, Path(outside) / "payload")
        dangling = self.root / "dangling_payload"
        dangling.symlink_to(self.root / "does_not_exist", target_is_directory=True)
        with self.assertRaisesRegex(M.PayloadError, "output leaf symlink refused"):
            M.build_payload(self.measurement, self.common, self.simulator,
                            self.configs, dangling)

    def test_a9_post_publish_failure_quarantines_and_clears_canonical(self):
        output = self.root / "payload"
        with mock.patch.object(M, "_post_publish_verify",
                               side_effect=M.PayloadError("injected post-publish failure")):
            with self.assertRaisesRegex(M.PayloadError, "quarantined"):
                M.build_payload(self.measurement, self.common, self.simulator,
                                self.configs, output)
        self.assertFalse(output.exists())
        quarantines = list(self.root.glob("payload.m644_quarantine_*"))
        self.assertEqual(1, len(quarantines))
        failure = self._load(quarantines[0] / "POST_PUBLISH_FAILURE.json")
        self.assertTrue(failure["canonical_output_removed"])
        self.assertFalse(failure["claim_boundary"]["payload_ready"])


if __name__ == "__main__":
    unittest.main()

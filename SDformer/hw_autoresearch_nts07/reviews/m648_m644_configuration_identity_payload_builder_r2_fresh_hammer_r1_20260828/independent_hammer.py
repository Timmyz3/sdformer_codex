#!/usr/bin/env python3
"""Independent adversarial hammer for M644/r2.

Only repository-local temporary fixtures are created.  No production payload,
GPU, EDA, M511, remote service, or docs/359 mutation is performed.
"""

import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_verify_m644_m527_configuration_payload_r2.py"
SPEC = importlib.util.spec_from_file_location("m644_independent_target", str(BUILDER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import M644 builder")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    return path


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def expect_reject(label, fn):
    try:
        fn()
    except M.PayloadError as exc:
        return {"label": label, "outcome": "REJECT", "reason": str(exc)}
    raise AssertionError(label + " unexpectedly accepted")


class Fixture(object):
    def __init__(self, root, fabricated_upstream=False):
        self.root = root
        if fabricated_upstream:
            trace_body = {"schema": "fabricated_not_decoder_complete_trace_v0",
                          "rows": 0, "decoder_rows": 0, "operators": ["fake0"]}
            population_body = {"schema": "fabricated_population_v0", "frames": 999}
            weight_body = {"schema": "fabricated_weights_v0", "weights": [1]}
            self.operator_ids = ["fake0"]
            population_scalar = 999
        else:
            trace_body = {"schema": "hammer_decoder_complete_trace_v1",
                          "rows": 40, "decoder_rows": 40,
                          "operators": ["patch", "conv0", "decoder0"]}
            population_body = {"schema": "hammer_population_v1", "frames": 3}
            weight_body = {"schema": "hammer_weights_v1", "weights": [1, 1, 1]}
            self.operator_ids = ["patch", "conv0", "decoder0"]
            population_scalar = 3
        self.trace = write_json(root / "trace.json", trace_body)
        self.population = write_json(root / "population.json", population_body)
        self.weights = write_json(root / "weights.json", weight_body)
        self.simulator = root / "simulator.py"
        self.simulator.write_text("# independent hammer simulator fixture\n", encoding="utf-8")
        self.upstream_obj = {
            "schema": M.UPSTREAM_SCHEMA,
            "status": M.UPSTREAM_STATUS,
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "complete_trace_manifest_sha256": digest(self.trace),
            "sequence_population_manifest_sha256": digest(self.population),
            "aggregation_weight_manifest_sha256": digest(self.weights),
            "population_scalar": population_scalar,
            "population_unit": "frozen_frames_across_frozen_sequence_population",
            "frame_definition": "one asserted H67 decoder-complete network invocation",
            "density_metric": "event_count_per_frame",
            "density_bin_boundaries": [0, 10, 20, 1000],
            "operator_ids": self.operator_ids,
            "verification": copy.deepcopy(M.UPSTREAM_VERIFICATION),
            "claim_boundary": copy.deepcopy(M.UPSTREAM_CLAIM_BOUNDARY),
        }
        self.upstream = write_json(root / "upstream_receipt.json", self.upstream_obj)
        self.measurement_obj = {
            "schema": M.MEASUREMENT_SCHEMA,
            "m527_contract_sha256": M.M527_CONTRACT_SHA256,
            "checkpoint_sha256": self.upstream_obj["checkpoint_sha256"],
            "complete_trace_manifest": ref(self.trace),
            "sequence_population_manifest": ref(self.population),
            "aggregation_weight_manifest": ref(self.weights),
            "upstream_semantic_verification_receipt": ref(self.upstream),
            "frame_definition": self.upstream_obj["frame_definition"],
            "density_metric": self.upstream_obj["density_metric"],
            "density_bin_boundaries": self.upstream_obj["density_bin_boundaries"],
            "population_scalar": population_scalar,
            "population_unit": self.upstream_obj["population_unit"],
            "operator_ids": self.operator_ids,
        }
        self.measurement = write_json(root / "measurement.json", self.measurement_obj)
        self.common = write_json(root / "common.json", {
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
        })
        self.configs = {}
        optimized = [self.operator_ids[0]]
        unsupported = self.operator_ids[1:]
        for config_id in M.CONFIGURATION_IDS:
            self.configs[config_id] = write_json(root / (config_id + "_source.json"), {
                "schema": M.CONFIG_SOURCE_SCHEMA,
                "configuration_id": config_id,
                "mechanism_enable_map": copy.deepcopy(M.EXPECTED_MECHANISMS[config_id]),
                "optimized_operator_ids": optimized,
                "unsupported_operator_ids": unsupported,
            })

    def build(self, name):
        output = self.root / name
        result = M.build_payload(self.measurement, self.common, self.simulator,
                                 self.configs, output)
        return output, result

    def verify(self, output):
        return M.verify_payload(output, self.measurement, self.common,
                                self.simulator, self.configs)


def reseal(output):
    names = [path.name for path in output.iterdir()
             if path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
    M._seal(output, names)


def rebuild_dependent_hashes_and_reseal(output):
    common_sha = digest(output / "common_resource_manifest.json")
    registry_path = output / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["common_resource_manifest"]["sha256"] = common_sha
    for config_id in M.CONFIGURATION_IDS:
        path = output / (config_id + ".json")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["common_resource_manifest_sha256"] = common_sha
        write_json(path, value)
        for entry in registry["configuration_manifests"]:
            if entry["configuration_id"] == config_id:
                entry["sha256"] = digest(path)
    write_json(registry_path, registry)
    reseal(output)


def main():
    parent = ROOT / "hw_autoresearch_nts07/system_simulator/tests"
    outcomes = []
    # The fixture must be built after its directory exists; keep each attack in
    # an isolated temporary subtree to rule out cross-test state.
    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary)
        root.mkdir(exist_ok=True)
        f = Fixture(root)
        output, result = f.build("positive")
        assert result["configuration_count"] == 5
        assert result["m527_configuration_registry_ready"] is False
        configs = [json.loads((output / (item + ".json")).read_text(encoding="utf-8"))
                   for item in M.CONFIGURATION_IDS]
        resource_fingerprints = {
            json.dumps(item["resource_tuple"], sort_keys=True) for item in configs}
        charge_fingerprints = {
            json.dumps(item["charge_policy"], sort_keys=True) for item in configs}
        assert len(resource_fingerprints) == 1
        assert len(charge_fingerprints) == 1
        assert all(all(item["charge_policy"][key] is True for key in M.CHARGE_FIELDS)
                   for item in configs)
        b2 = [item for item in configs if item["configuration_id"] == "b2_exact_bit_sparse_k1"][0]
        assert b2["resource_tuple"]["service_width_sources_per_cycle"] == 8
        assert b2["mechanism_enable_map"]["execution_service_limit_sources_per_cycle"] == 1
        assert all(b2["charge_policy"].values())
        outcomes.append({"label": "positive_five_common_configs_and_b2_physical_charge",
                         "outcome": "PASS"})

        # A1: optimized/fallback partition is changed and all dependent hashes resealed.
        path = output / "c123_ours_exact.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["optimized_operator_ids"] = list(f.operator_ids)
        value["fallback_policy"]["unsupported_operator_ids"] = []
        write_json(path, value)
        rebuild_dependent_hashes_and_reseal(output)
        outcomes.append(expect_reject("A1_resealed_operator_partition", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("a2")
        path = output / "c123_ours_exact.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["claim_boundary"]["waterfall_admitted"] = True
        write_json(path, value); rebuild_dependent_hashes_and_reseal(output)
        outcomes.append(expect_reject("A2_resealed_waterfall", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("a3")
        path = output / "registry.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["m527_contract_sha256"] = "0" * 64
        value["configuration_identity_payload_ready"] = False
        value["m527_contract_admission_gate_current_value"] = True
        write_json(path, value); reseal(output)
        outcomes.append(expect_reject("A3_resealed_registry_gate", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("a4")
        path = output / "verification_receipt.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "HEADLINE"
        value["validated_configuration_ids"] = []
        value["all_live_source_paths_and_hashes_verified"] = False
        value["claim_boundary"]["system_speedup"] = True
        write_json(path, value); reseal(output)
        outcomes.append(expect_reject("A4_resealed_receipt_overclaim", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("a5")
        for config_id in M.CONFIGURATION_IDS:
            path = output / (config_id + ".json")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["configuration_source_path"] = "missing/source.json"
            value["simulator_source_path"] = "missing/simulator.py"
            value["complete_trace_manifest_path"] = "missing/trace.json"
            write_json(path, value)
        rebuild_dependent_hashes_and_reseal(output)
        outcomes.append(expect_reject("A5_resealed_member_paths", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("a6")
        path = output / "common_resource_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["measurement_identity"]["checkpoint_sha256"] = "0" * 64
        value["measurement_identity"]["operator_ids"] = []
        write_json(path, value); rebuild_dependent_hashes_and_reseal(output)
        outcomes.append(expect_reject("A6_resealed_embedded_measurement", lambda: f.verify(output)))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root)
        changed = copy.deepcopy(f.upstream_obj)
        changed["population_scalar"] = 999
        write_json(f.upstream, changed)
        measurement = copy.deepcopy(f.measurement_obj)
        measurement["upstream_semantic_verification_receipt"] = ref(f.upstream)
        write_json(f.measurement, measurement)
        outcomes.append(expect_reject("A7_upstream_receipt_mutation",
                                      lambda: f.build("a7")))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root)
        real = root / "real"; real.mkdir()
        alias = root / "alias"; alias.symlink_to(real, target_is_directory=True)
        outcomes.append(expect_reject("A8_output_symlink_ancestor",
                                      lambda: M.build_payload(f.measurement, f.common,
                                                              f.simulator, f.configs,
                                                              alias / "payload")))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root)
        dangling = root / "dangling_payload"
        dangling.symlink_to(root / "absent_target", target_is_directory=True)
        outcomes.append(expect_reject("output_dangling_leaf_symlink",
                                      lambda: M.build_payload(f.measurement, f.common,
                                                              f.simulator, f.configs,
                                                              dangling)))
        with tempfile.TemporaryDirectory(prefix="m648_outside_") as outside:
            outcomes.append(expect_reject("output_outside_repository",
                                          lambda: M.build_payload(f.measurement, f.common,
                                                                  f.simulator, f.configs,
                                                                  Path(outside) / "payload")))

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output = root / "staging_fail"
        before = set(root.glob(".m644_m527_payload_*"))
        with mock.patch.object(M, "verify_payload",
                               side_effect=M.PayloadError("m648 injected staging failure")):
            outcomes.append(expect_reject(
                "staging_failure_rejected",
                lambda: M.build_payload(f.measurement, f.common, f.simulator,
                                        f.configs, output)))
        after = set(root.glob(".m644_m527_payload_*"))
        assert not output.exists() and after == before
        outcomes.append({"label": "staging_failure_cleans_temporary_and_canonical",
                         "outcome": "PASS"})

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output = root / "a9"
        try:
            with mock.patch.object(M, "_post_publish_verify",
                                   side_effect=M.PayloadError("m648 injected postpublish failure")):
                M.build_payload(f.measurement, f.common, f.simulator, f.configs, output)
            raise AssertionError("A9 unexpectedly accepted")
        except M.PayloadError:
            pass
        quarantines = list(root.glob("a9.m644_quarantine_*"))
        assert not output.exists() and len(quarantines) == 1
        failure = json.loads((quarantines[0] / "POST_PUBLISH_FAILURE.json").read_text(encoding="utf-8"))
        assert failure["canonical_output_removed"] is True
        outcomes.append({"label": "A9_postpublish_cleanup_quarantine", "outcome": "PASS"})

    with tempfile.TemporaryDirectory(prefix="m648_hammer_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root); output, _ = f.build("live")
        source = f.configs["c123_ours_exact"]
        value = json.loads(source.read_text(encoding="utf-8"))
        value["optimized_operator_ids"] = list(f.operator_ids)
        value["unsupported_operator_ids"] = []
        write_json(source, value)
        outcomes.append(expect_reject("live_source_mutation_after_publish",
                                      lambda: f.verify(output)))

    # Fresh fake-receipt attack: all caller-controlled files are mutually
    # self-consistent, but the trace has a wrong schema and zero decoder rows.
    # Acceptance proves that M644 checks receipt contents, not receipt provenance.
    with tempfile.TemporaryDirectory(prefix="m648_fake_upstream_", dir=str(parent)) as temporary:
        root = Path(temporary); f = Fixture(root, fabricated_upstream=True)
        output, result = f.build("fabricated_payload")
        assert result["status"].startswith("PASS_M644")
        receipt = json.loads((output / "verification_receipt.json").read_text(encoding="utf-8"))
        assert receipt["upstream_decoder_complete_semantics_verified"] is True
        outcomes.append({
            "label": "F1_self_authored_fake_upstream_receipt_wrong_schema_zero_rows",
            "outcome": "ACCEPTED",
            "returned_status": result["status"],
            "payload_claim": "upstream_decoder_complete_semantics_verified=true",
        })

    print(json.dumps({
        "schema": "m648_m644_independent_hammer_result_v1",
        "author_regression_count_repeated": 9,
        "outcomes": outcomes,
        "fresh_finding": "F1_ACCEPTED_FAKE_UPSTREAM_RECEIPT_WITHOUT_PRODUCER_OR_SEAL_IDENTITY",
        "production_payload_generated": False,
        "gpu_used": False,
        "eda_used": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

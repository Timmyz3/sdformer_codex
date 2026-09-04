#!/usr/bin/env python3
"""CPU-only independent attack harness for the frozen M645 registry r4.

The production targets are never edited.  Temporary rooted fixtures are made
under the namespaces already required by the registry and are removed by the
fixture context.  The target test module is used only as a fixture factory for
the future-positive graph; every check and mutation below is independently
specified here.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m645_h67_paper_metric_registry_r4.py"
CONFIG = HW_ROOT / "system_simulator/config/m645_h67_paper_metric_registry_r4_20260828.json"
TESTS = HW_ROOT / "system_simulator/tests/test_m645_h67_paper_metric_registry_r4.py"
CONTRACT = HW_ROOT / "contracts/m645_h67_paper_metric_registry_r4_contract_r1_20260828.json"
REQUEST = HW_ROOT / "reviews/m646_m645_registry_r4_rooted_bundle_fresh_hammer_r1_REQUEST_20260828/request.json"
M527 = HW_ROOT / "contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module("m651_target", BUILDER)
T = load_module("m651_target_fixture_factory", TESTS)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def temp_config(value):
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
    with handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    return Path(handle.name)


def build_value(value):
    path = temp_config(value)
    try:
        try:
            result = M.build(path)
            return {"accepted": True, "headline": result["headline_gate"]["admitted"],
                    "eligible_rows": result["headline_gate"]["eligible_row_count"]}
        except (M.RegistryError, RuntimeError) as exc:
            return {"accepted": False, "error": str(exc)}
    finally:
        path.unlink()


def rewrite_json_spec(spec, mutate):
    path = REPO_ROOT / spec["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                    encoding="utf-8")
    spec["sha256"] = sha256(path)
    return value


def fixture_case():
    case = T.M645RegistryTests(methodName="test_01_canonical_is_zero_bundle_zero_authority_zero_headline")
    T.M645RegistryTests.setUpClass()
    return case


def with_fixture(action):
    case = fixture_case()
    previous = M.TRUSTED_HAMMER_AUTHORITIES
    try:
        with case._rooted_positive_fixture() as (config_path, bundle, authority):
            authority_id = bundle["independent_hammer_authority_id"]
            M.TRUSTED_HAMMER_AUTHORITIES = {authority_id: authority}
            return action(Path(config_path), bundle, authority)
    finally:
        M.TRUSTED_HAMMER_AUTHORITIES = previous
        case.doCleanups()


def roots(bundle):
    _, _, producer_sha = M._file_spec(bundle["producer"], "producer",
                                      "hw_autoresearch_nts07/system_simulator/",
                                      ("text/x-python", "text/plain"))
    _, _, simulator_sha = M._file_spec(bundle["unified_simulator"], "simulator",
                                       "hw_autoresearch_nts07/system_simulator/",
                                       ("text/x-python", "text/plain"))
    _, invocation, invocation_sha = M._file_spec(bundle["invocation_contract"], "invocation",
                                                 "hw_autoresearch_nts07/contracts/")
    measurement = M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)
    common = M._validate_common_resource(bundle["common_resource_manifest"], producer_sha,
                                         simulator_sha, measurement)
    configs = M._validate_configurations(bundle["configuration_manifests"], common, producer_sha,
                                         simulator_sha, invocation_sha, measurement)
    return producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs


def resolved_raw(bundle):
    producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs = roots(bundle)
    raw = M._validate_raw_runs(bundle["raw_run_index"], configs, producer_sha, simulator_sha,
                               invocation_sha, measurement)
    recomputed = M._recompute(raw["runs"], measurement)
    return producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs, raw, recomputed


def expect_reject(func):
    try:
        func()
    except (M.RegistryError, RuntimeError, ValueError) as exc:
        return {"accepted": False, "error": str(exc)}
    return {"accepted": True}


def old_five_file_attack(base):
    dummy = {"path": "hw_autoresearch_nts07/results/fake.json", "sha256": "1" * 64,
             "media_type": "application/json"}
    value = copy.deepcopy(base)
    value["table_a_evidence_bundles"]["direct_unified_old_five"] = {
        "direct_result": copy.deepcopy(dummy),
        "completion_receipt": copy.deepcopy(dummy),
        "resource_manifest": {"path": "hw_autoresearch_nts07/system_simulator/fake.json",
                              "sha256": "2" * 64, "media_type": "application/json"},
        "coverage_receipt": copy.deepcopy(dummy),
        "independent_hammer_receipt": {"path": "hw_autoresearch_nts07/reviews/fake/review.json",
                                       "sha256": "3" * 64, "media_type": "application/json"},
    }
    return build_value(value)


def expanded_self_signed_attack(base):
    dummy = {"path": "hw_autoresearch_nts07/results/fake.json", "sha256": "1" * 64,
             "media_type": "application/json"}
    row_specs = {row_id: copy.deepcopy(dummy) for row_id in M.MANDATORY_ROW_IDS}
    value = copy.deepcopy(base)
    bundle_id = "direct_unified_shape_complete_self_signed"
    value["table_a_evidence_bundles"][bundle_id] = {
        "schema": "m645.h67.rooted_direct_bundle.r1", "bundle_id": bundle_id,
        "m527_contract_sha256": M.M527_CONTRACT_SHA256,
        "common_resource_manifest": copy.deepcopy(dummy), "configuration_manifests": row_specs,
        "producer": {"path": "hw_autoresearch_nts07/system_simulator/fake.py",
                     "sha256": "2" * 64, "media_type": "text/x-python"},
        "unified_simulator": {"path": "hw_autoresearch_nts07/system_simulator/fake_sim.py",
                              "sha256": "3" * 64, "media_type": "text/x-python"},
        "invocation_contract": copy.deepcopy(dummy), "measurement_identity": copy.deepcopy(dummy),
        "raw_run_index": copy.deepcopy(dummy), "direct_result": copy.deepcopy(dummy),
        "completion_receipt": copy.deepcopy(dummy), "coverage_receipt": copy.deepcopy(dummy),
        "ppa_receipt": copy.deepcopy(dummy), "energy_receipt": copy.deepcopy(dummy),
        "accuracy_receipt": copy.deepcopy(dummy),
        "independent_hammer_authority_id": "self_signed_pass",
        "independent_hammer_receipt": {"path": "hw_autoresearch_nts07/reviews/self_signed/hammer.json",
                                       "sha256": "4" * 64, "media_type": "application/json"},
    }
    return build_value(value)


def strict_json_attack(payload):
    handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
    path = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        return expect_reject(lambda: M.M635.load_json(path, "strict attack"))
    finally:
        path.unlink()


def positive_path(config_path, bundle, authority):
    result = M.build(config_path)
    request_manifest = REPO_ROOT / authority["request_manifest"]["path"]
    request_rows = M._parse_sha256sums(request_manifest, "fixture request manifest")
    request_docs = {}
    for name in request_rows:
        member = request_manifest.parent / name
        try:
            request_docs[name] = json.loads(member.read_text(encoding="utf-8"))
        except Exception:
            request_docs[name] = None
    return {"accepted": True, "authorities": result["trusted_hammer_authority_count"],
            "bundles": result["table_a_evidence_bundle_count"],
            "eligible_rows": result["headline_gate"]["eligible_row_count"],
            "headline": result["headline_gate"]["admitted"],
            "direct_speedup": result["headline_gate"]["direct_speedup"],
            "request_sealed_members": request_rows, "request_documents": request_docs}


def inspect_future_roots(config_path, bundle, authority):
    producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs, raw, recomputed = resolved_raw(bundle)
    typed = M._validate_typed_receipts(bundle["bundle_id"], bundle, raw, configs,
                                       recomputed[0], measurement)
    result = M._validate_result_and_receipts(bundle["bundle_id"], bundle, configs, common,
                                             measurement, raw, recomputed, typed)
    request_rows, _, request_outer = M._validate_outer_seal(
        authority["request_manifest"], authority["request_outer_seal"], "request")
    review_rows, _, review_outer = M._validate_outer_seal(
        authority["review_manifest"], authority["review_outer_seal"], "review")
    config_docs = [M.M635.load_json(REPO_ROOT / bundle["configuration_manifests"][row_id]["path"], row_id)
                   for row_id in M.MANDATORY_ROW_IDS]
    return {
        "producer_sha256": producer_sha, "simulator_sha256": simulator_sha,
        "invocation_sha256": invocation_sha, "checkpoint_sha256": invocation["checkpoint_sha256"],
        "configuration_ids": [doc["configuration_id"] for doc in config_docs],
        "configuration_paths_distinct": len({bundle["configuration_manifests"][row_id]["path"]
                                               for row_id in M.MANDATORY_ROW_IDS}) == 6,
        "configuration_source_paths_distinct": len({doc["configuration_source"]["path"]
                                                      for doc in config_docs}) == 6,
        "common_resource_equal": all(doc["resource_tuple"] == common["doc"]["resource_tuple"]
                                     for doc in config_docs),
        "charge_equal": all(doc["charge_policy"] == common["doc"]["charge_policy"]
                            for doc in config_docs),
        "fallback_nonpartition_equal": all(
            {key: doc["fallback_policy"][key] for key in M.FALLBACK_FIELDS[:-1]} ==
            {key: common["doc"]["fallback_policy"][key] for key in M.FALLBACK_FIELDS[:-1]}
            for doc in config_docs),
        "population_samples": len(measurement["samples"]), "raw_runs": len(raw["runs"]),
        "expected_raw_runs": len(measurement["samples"]) * len(M.VIEW_NAMES) * len(M.MANDATORY_ROW_IDS),
        "typed_rows": {kind: len(typed[kind]) for kind in ("ppa", "energy", "accuracy")},
        "direct_rows": len(result["rows"]), "request_sealed_members": request_rows,
        "review_sealed_members": review_rows, "request_outer_sha256": request_outer,
        "review_outer_sha256": review_outer,
    }


def trace_decoder_attack(config_path, bundle, authority):
    measurement_path = REPO_ROOT / bundle["measurement_identity"]["path"]
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    trace_spec = measurement["complete_trace_manifest"]
    rewrite_json_spec(trace_spec, lambda value: value.__setitem__("decoder_complete", False))
    rewrite_json_spec(bundle["measurement_identity"],
                      lambda value: value.__setitem__("complete_trace_manifest", trace_spec))
    return expect_reject(lambda: M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256))


def producer_root_attack(config_path, bundle, authority):
    producer_path = REPO_ROOT / bundle["producer"]["path"]
    producer_path.write_text("# mutated producer\n", encoding="utf-8")
    bundle["producer"]["sha256"] = sha256(producer_path)
    return expect_reject(lambda: M._validate_bundle(bundle["bundle_id"], bundle))


def simulator_root_attack(config_path, bundle, authority):
    simulator_path = REPO_ROOT / bundle["unified_simulator"]["path"]
    simulator_path.write_text("# mutated simulator\n", encoding="utf-8")
    bundle["unified_simulator"]["sha256"] = sha256(simulator_path)
    return expect_reject(lambda: M._validate_bundle(bundle["bundle_id"], bundle))


def invocation_checkpoint_attack(config_path, bundle, authority):
    rewrite_json_spec(bundle["invocation_contract"],
                      lambda value: value.__setitem__("checkpoint_sha256", "0" * 64))
    return expect_reject(lambda: M._validate_bundle(bundle["bundle_id"], bundle))


def measurement_checkpoint_attack(config_path, bundle, authority):
    rewrite_json_spec(bundle["measurement_identity"],
                      lambda value: value.__setitem__("checkpoint_sha256", "0" * 64))
    return expect_reject(lambda: M._validate_measurement_identity(bundle["measurement_identity"],
                                                                  M.CHECKPOINT_SHA256))


def duplicate_configuration_source_attack(config_path, bundle, authority):
    row0, row1 = M.MANDATORY_ROW_IDS[:2]
    manifest0 = M.M635.load_json(REPO_ROOT / bundle["configuration_manifests"][row0]["path"], row0)
    rewrite_json_spec(bundle["configuration_manifests"][row1],
                      lambda value: value.__setitem__("configuration_source",
                                                      copy.deepcopy(manifest0["configuration_source"])))
    return expect_reject(lambda: roots(bundle))


def common_resource_mutation_attack(config_path, bundle, authority):
    rewrite_json_spec(bundle["common_resource_manifest"],
                      lambda value: value["resource_tuple"].__setitem__("onchip_sram_bytes_total", 245761))
    producer_sha, simulator_sha, _, _, measurement, _, _ = roots_without_common(bundle)
    return expect_reject(lambda: M._validate_common_resource(bundle["common_resource_manifest"],
                                                             producer_sha, simulator_sha, measurement))


def roots_without_common(bundle):
    _, _, producer_sha = M._file_spec(bundle["producer"], "producer",
                                      "hw_autoresearch_nts07/system_simulator/",
                                      ("text/x-python", "text/plain"))
    _, _, simulator_sha = M._file_spec(bundle["unified_simulator"], "simulator",
                                       "hw_autoresearch_nts07/system_simulator/",
                                       ("text/x-python", "text/plain"))
    _, invocation, invocation_sha = M._file_spec(bundle["invocation_contract"], "invocation",
                                                 "hw_autoresearch_nts07/contracts/")
    measurement = M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)
    return producer_sha, simulator_sha, invocation, invocation_sha, measurement, None, None


def population_medium_contract_probe(config_path, bundle, authority):
    measurement_path = REPO_ROOT / bundle["measurement_identity"]["path"]
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    population_spec = measurement["sequence_population_manifest"]
    rewrite_json_spec(population_spec,
                      lambda value: value["samples"][1].__setitem__("density_stratum", "medium"))
    rewrite_json_spec(bundle["measurement_identity"],
                      lambda value: value.__setitem__("sequence_population_manifest", population_spec))
    return expect_reject(lambda: M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256))


def aggregation_omission_attack(config_path, bundle, authority):
    measurement_path = REPO_ROOT / bundle["measurement_identity"]["path"]
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    weights_spec = measurement["aggregation_weight_manifest"]
    rewrite_json_spec(weights_spec, lambda value: value["samples"].pop())
    rewrite_json_spec(bundle["measurement_identity"],
                      lambda value: value.__setitem__("aggregation_weight_manifest", weights_spec))
    return expect_reject(lambda: M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256))


def raw_omission_attack(config_path, bundle, authority):
    producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs = roots(bundle)
    rewrite_json_spec(bundle["raw_run_index"], lambda value: value["runs"].pop())
    return expect_reject(lambda: M._validate_raw_runs(bundle["raw_run_index"], configs,
                                                      producer_sha, simulator_sha,
                                                      invocation_sha, measurement))


def raw_duplicate_attack(config_path, bundle, authority):
    producer_sha, simulator_sha, invocation, invocation_sha, measurement, common, configs = roots(bundle)
    def duplicate(value):
        value["runs"][-1] = copy.deepcopy(value["runs"][0])
    rewrite_json_spec(bundle["raw_run_index"], duplicate)
    return expect_reject(lambda: M._validate_raw_runs(bundle["raw_run_index"], configs,
                                                      producer_sha, simulator_sha,
                                                      invocation_sha, measurement))


def typed_energy_swap_attack(config_path, bundle, authority):
    _, _, _, _, measurement, _, configs, raw, recomputed = resolved_raw(bundle)
    def swap(value):
        value["rows"][0]["logic_energy_mj"] += 0.00005
        value["rows"][0]["dram_energy_mj"] -= 0.00005
    rewrite_json_spec(bundle["energy_receipt"], swap)
    return expect_reject(lambda: M._validate_typed_receipts(bundle["bundle_id"], bundle, raw,
                                                            configs, recomputed[0], measurement))


def typed_accuracy_attack(config_path, bundle, authority):
    _, _, _, _, measurement, _, configs, raw, recomputed = resolved_raw(bundle)
    rewrite_json_spec(bundle["accuracy_receipt"],
                      lambda value: value["rows"][0].__setitem__("aee", 99.0))
    return expect_reject(lambda: M._validate_typed_receipts(bundle["bundle_id"], bundle, raw,
                                                            configs, recomputed[0], measurement))


def typed_ppa_attack(config_path, bundle, authority):
    _, _, _, _, measurement, _, configs, raw, recomputed = resolved_raw(bundle)
    rewrite_json_spec(bundle["ppa_receipt"],
                      lambda value: value["rows"][0].__setitem__("total_area_mm2", 7.0))
    return expect_reject(lambda: M._validate_typed_receipts(bundle["bundle_id"], bundle, raw,
                                                            configs, recomputed[0], measurement))


def ppa_raw_report_semantic_probe(config_path, bundle, authority):
    _, _, _, _, measurement, _, configs, raw, recomputed = resolved_raw(bundle)
    ppa_path = REPO_ROOT / bundle["ppa_receipt"]["path"]
    ppa = json.loads(ppa_path.read_text(encoding="utf-8"))
    report_spec = ppa["rows"][0]["logic_report"]
    report_path = REPO_ROOT / report_spec["path"]
    report_path.write_text("area 9999.0\n", encoding="utf-8")
    report_spec["sha256"] = sha256(report_path)
    ppa_path.write_text(json.dumps(ppa, separators=(",", ":")), encoding="utf-8")
    bundle["ppa_receipt"]["sha256"] = sha256(ppa_path)
    try:
        M._validate_typed_receipts(bundle["bundle_id"], bundle, raw, configs,
                                   recomputed[0], measurement)
        return {"accepted": True, "note": "raw logic report says area 9999 while typed scalar remains 0.6"}
    except M.RegistryError as exc:
        return {"accepted": False, "error": str(exc)}


def result_role_attack(config_path, bundle, authority):
    _, _, _, _, measurement, common, configs, raw, recomputed = resolved_raw(bundle)
    typed = M._validate_typed_receipts(bundle["bundle_id"], bundle, raw, configs,
                                       recomputed[0], measurement)
    rewrite_json_spec(bundle["direct_result"],
                      lambda value: value["rows"][0].__setitem__("role", "candidate"))
    return expect_reject(lambda: M._validate_result_and_receipts(bundle["bundle_id"], bundle,
                                                                 configs, common, measurement,
                                                                 raw, recomputed, typed))


def outer_seal_attack(config_path, bundle, authority):
    manifest = REPO_ROOT / authority["review_manifest"]["path"]
    manifest.write_text("0" * 64 + "  hammer.json\n", encoding="utf-8")
    return expect_reject(lambda: M._validate_outer_seal(authority["review_manifest"],
                                                        authority["review_outer_seal"], "review"))


def symlink_attack():
    tests_root = HW_ROOT / "system_simulator/tests"
    with tempfile.TemporaryDirectory(dir=str(tests_root)) as directory:
        root = Path(directory)
        target = root / "target.json"
        target.write_text("{}", encoding="utf-8")
        link = root / "link.json"
        link.symlink_to(target)
        spec = {"path": link.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(target),
                "media_type": "application/json"}
        return expect_reject(lambda: M._file_spec(spec, "symlink"))


def external_m618_promotion(base):
    sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
    value = copy.deepcopy(base)
    rows = copy.deepcopy(sealed["table_a_schema"]["rows"])
    for index, row in enumerate(rows[:6]):
        row.update({"cycles": 1000 if index == 0 else 100, "energy_mj": 1.0,
                    "area_mm2": 1.0, "accuracy": 1.0, "source_id": "m618",
                    "measurement_class": M.M635.ALLOWED_MEASUREMENT_CLASS,
                    "population_id": "fake", "workload_id": "fake",
                    "resource_manifest_sha256": "1" * 64,
                    "completion_receipt_sha256": "2" * 64,
                    "decoder_complete": True, "memory_timing_included": True,
                    "full_network_completion": True,
                    "logic_sram_dram_energy_closed": True,
                    "logic_macro_area_closed": True, "sta_closed": True,
                    "independent_hammer_pass": True, "blockers": []})
    value["table_a_rows"] = rows
    return build_value(value)


def row_role_attack(base):
    sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
    value = copy.deepcopy(base)
    value["table_a_rows"] = copy.deepcopy(sealed["table_a_schema"]["rows"])
    value["table_a_rows"][0]["role"] = "candidate"
    return build_value(value)


def base_anchor_reseal_attack(base):
    sealed = M.M635.load_json(M.M635_CONFIG, "sealed base")
    sealed["headline_policy"]["fixed_numerator_row_id"] = "ours_exact"
    config_dir = HW_ROOT / "system_simulator/config"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                     dir=str(config_dir), delete=False) as handle:
        json.dump(sealed, handle, ensure_ascii=False, allow_nan=False)
        path = Path(handle.name)
    try:
        value = copy.deepcopy(base)
        value["base_registry"] = {"path": path.relative_to(REPO_ROOT).as_posix(),
                                  "sha256": sha256(path), "media_type": "application/json"}
        return build_value(value)
    finally:
        path.unlink()


def main():
    request = M.M635.load_json(REQUEST, "M646 request")
    base = M.M635.load_json(CONFIG, "M645 canonical")
    canonical = M.build(CONFIG)
    test_tree = ast.parse(TESTS.read_text(encoding="utf-8"))
    test_count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                     for node in ast.walk(test_tree))
    expected = request["targets"]
    target_paths = {
        "builder": BUILDER, "canonical_config": CONFIG, "unit_tests": TESTS,
        "contract": CONTRACT,
        "author_handoff_outer_seal": REPO_ROOT / expected["author_handoff_outer_seal"]["path"],
        "protected_docs359": DOCS359,
    }
    identities = {name: {"expected": expected[name]["sha256"], "actual": sha256(path),
                         "match": expected[name]["sha256"] == sha256(path)}
                  for name, path in target_paths.items()}
    config_authority = copy.deepcopy(base)
    config_authority["trusted_hammer_authorities"] = {"self": {"pass": True}}
    attacks = {
        "old_internally_consistent_five_file_topology": old_five_file_attack(base),
        "expanded_shape_complete_self_signed_authority": expanded_self_signed_attack(base),
        "config_authority_extension": build_value(config_authority),
        "duplicate_json_key": strict_json_attack(b'{"a":1,"a":2}'),
        "nan": strict_json_attack(b'{"a":NaN}'),
        "infinity": strict_json_attack(b'{"a":Infinity}'),
        "overflow_1e999": strict_json_attack(b'{"a":1e999}'),
        "symlink": symlink_attack(),
        "producer_source_mutation": with_fixture(producer_root_attack),
        "simulator_source_mutation": with_fixture(simulator_root_attack),
        "invocation_checkpoint_mutation": with_fixture(invocation_checkpoint_attack),
        "measurement_checkpoint_mutation": with_fixture(measurement_checkpoint_attack),
        "duplicate_configuration_source": with_fixture(duplicate_configuration_source_attack),
        "common_resource_mutation": with_fixture(common_resource_mutation_attack),
        "decoder_complete_false": with_fixture(trace_decoder_attack),
        "m527_medium_density_spelling": with_fixture(population_medium_contract_probe),
        "aggregation_cartesian_omission": with_fixture(aggregation_omission_attack),
        "raw_run_omission": with_fixture(raw_omission_attack),
        "raw_run_duplicate": with_fixture(raw_duplicate_attack),
        "typed_energy_component_swap_same_total": with_fixture(typed_energy_swap_attack),
        "typed_accuracy_scalar_mutation": with_fixture(typed_accuracy_attack),
        "typed_ppa_total_mutation": with_fixture(typed_ppa_attack),
        "raw_ppa_report_semantic_mismatch": with_fixture(ppa_raw_report_semantic_probe),
        "direct_result_role_mutation": with_fixture(result_role_attack),
        "review_outer_seal_mutation": with_fixture(outer_seal_attack),
        "external_m618_table_b_promotion": external_m618_promotion(base),
        "table_a_row_role_mutation": row_role_attack(base),
        "base_registry_anchor_reseal": base_anchor_reseal_attack(base),
    }
    m527 = M.M635.load_json(M527, "M527")
    output = {
        "schema": "m651.m645.registry_r4.independent_attack_harness.r1",
        "target_identities": identities,
        "test_count": {"expected": expected["unit_tests"]["exact_test_count"],
                       "actual": test_count, "match": test_count == expected["unit_tests"]["exact_test_count"]},
        "canonical": {"sources": len(canonical["source_hashes_validated"]),
                      "trusted_authorities": canonical["trusted_hammer_authority_count"],
                      "bundles": canonical["table_a_evidence_bundle_count"],
                      "eligible_rows": canonical["headline_gate"]["eligible_row_count"],
                      "headline": canonical["headline_gate"]["admitted"],
                      "analytical": canonical["analytical_diagnostic"]["admitted"]},
        "future_positive_fixture": with_fixture(positive_path),
        "future_root_inventory": with_fixture(inspect_future_roots),
        "attacks": attacks,
        "static_contract_gaps": {
            "m527_requires_fixed_numerator_before_headline":
                "measurement_identity.admission_gate.current_value" in m527["headline_policy"]["independent_required_gates"] and
                "fixed_throughput_numerators.admission_gate.current_value" in m527["headline_policy"]["independent_required_gates"],
            "m645_builder_mentions_fixed_throughput_numerator":
                "fixed_throughput_numerator" in BUILDER.read_text(encoding="utf-8"),
            "m527_density_strata": m527["aggregation"]["density_strata_required"],
            "m645_density_literals": ["low", "mid", "high"],
            "hammer_receipt_fields": ["schema", "status", "authority_id", "request_outer_seal_sha256",
                                      "bundle_evidence_sha256", "severity_counts", "independence",
                                      "recomputed_rows", "recomputed_aggregates", "recomputed_views", "authorization"],
            "hammer_receipt_has_reviewed_target_identity_field": False,
        },
        "docs359_sha256_after": sha256(DOCS359),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent CPU/static attacks for the sealed M653 registry-r5 target.

The author test module is used only to construct disposable, internally
consistent evidence graphs.  This harness defines its own assertions and
records whether a graph is accepted or rejected.  All fixture files live in
TemporaryDirectory instances and are removed before exit.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m653_h67_paper_metric_registry_r5.py"
TESTS = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests/test_m653_h67_paper_metric_registry_r5.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m653_h67_paper_metric_registry_r5_20260828.json"
DOCS359 = REPO_ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_module("m657_author_fixture_factory", TESTS)
T.M653RegistryTests.setUpClass()
# Use the exact target-module instance owned by the disposable fixture factory;
# its context manager installs the temporary code-level authority in this
# module's otherwise-empty trust map for the duration of each attack.
M = T.M


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def helper():
    item = T.M653RegistryTests(methodName="test_01_canonical_is_zero_and_headline_false")
    item.base = T.M653RegistryTests.base
    return item


def reject(label, fn):
    try:
        fn()
    except M.RegistryError as exc:
        return {"label": label, "rejected": True, "error": str(exc)}
    return {"label": label, "rejected": False, "error": None}


def reseal_request(h, authority):
    request_path = REPO_ROOT / authority["request_document"]["path"]
    authority["request_document"] = h._spec(request_path, "application/json")
    manifest, outer = h._seal_one(request_path.parent, authority["request_document"])
    authority["request_manifest"] = manifest
    authority["request_outer_seal"] = outer


def reseal_review(h, authority):
    review_path = REPO_ROOT / authority["review_document"]["path"]
    receipt_path = REPO_ROOT / authority["receipt"]["path"]
    authority["review_document"] = h._spec(review_path, "application/json")
    authority["receipt"] = h._spec(receipt_path, "application/json")
    manifest, outer = h._seal_two(review_path.parent, authority["receipt"], authority["review_document"])
    authority["review_manifest"] = manifest
    authority["review_outer_seal"] = outer


def canonical_check():
    value = M.build(CONFIG)
    return {
        "sources": len(value["source_hashes_validated"]),
        "trusted_authorities": value["trusted_hammer_authority_count"],
        "bundles": value["table_a_evidence_bundle_count"],
        "eligible_rows": value["headline_gate"]["eligible_row_count"],
        "headline": value["headline_gate"]["admitted"],
        "all_m527_gates": value["headline_gate"]["all_m527_independent_gates_pass"],
        "analytical": value["analytical_diagnostic"]["admitted"],
    }


def accepted_scope_and_ppa_probe():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, bundle, _):
            value = M.build(config_path)
            measurement = h._load_spec(bundle["measurement_identity"])
            numerator = h._load_spec(bundle["fixed_throughput_numerator_receipt"])
            m527 = M._runtime_m527_contract()
            required = m527["identity"]["required_operator_scope"]
            ppa = h._load_spec(bundle["ppa_receipt"])
            logic_report = REPO_ROOT / ppa["rows"][0]["logic_report"]["path"]
            sram_report = REPO_ROOT / ppa["rows"][0]["sram_report"]["path"]
            sta_report = REPO_ROOT / ppa["rows"][0]["sta_report"]["path"]
            return {
                "accepted": bool(value["headline_gate"]["admitted"]),
                "eligible_rows": value["headline_gate"]["eligible_row_count"],
                "measurement_operator_ids": measurement["operator_ids"],
                "numerator_included_operator_scope": numerator["included_operator_scope"],
                "m527_required_operator_scope": required,
                "required_scope_equals_numerator_scope": set(required) == set(numerator["included_operator_scope"]),
                "raw_ppa_reports": {
                    "logic": logic_report.read_text(encoding="utf-8"),
                    "sram": sram_report.read_text(encoding="utf-8"),
                    "sta": sta_report.read_text(encoding="utf-8"),
                },
                "ppa_receipt_has_tool_or_extractor_identity": any(
                    key in ppa for key in ("tool_identity", "extractor_identity", "command_sha256", "library_corner")
                ),
            }
    finally:
        h.doCleanups()


def attack_arbitrary_request():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, _, authority):
            path = REPO_ROOT / authority["request_document"]["path"]
            path.write_text('{"target":"fixture"}', encoding="utf-8")
            reseal_request(h, authority)
            return reject("resealed arbitrary request", lambda: M.build(config_path))
    finally:
        h.doCleanups()


def attack_wrong_reviewed_target():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, _, authority):
            path = REPO_ROOT / authority["request_document"]["path"]
            request = json.loads(path.read_text(encoding="utf-8"))
            request["reviewed_targets"]["registry_contract"] = h._spec(CONFIG, "application/json")
            path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            reseal_request(h, authority)
            return reject("resealed wrong registry-contract target", lambda: M.build(config_path))
    finally:
        h.doCleanups()


def attack_nonzero_review_p1():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, _, authority):
            path = REPO_ROOT / authority["review_document"]["path"]
            review = json.loads(path.read_text(encoding="utf-8"))
            review["severity_counts"] = {"P0": 0, "P1": 1}
            path.write_text(json.dumps(review, separators=(",", ":")), encoding="utf-8")
            reseal_review(h, authority)
            return reject("resealed nonzero review P1", lambda: M.build(config_path))
    finally:
        h.doCleanups()


def attack_incomplete_evidence_map():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, _, authority):
            path = REPO_ROOT / authority["request_document"]["path"]
            request = json.loads(path.read_text(encoding="utf-8"))
            del request["bundle_evidence_sha256"]["m527_contract"]
            request["complete_evidence_root_sha256"] = M._map_sha(request["bundle_evidence_sha256"])
            path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            reseal_request(h, authority)
            return reject("resealed incomplete evidence map", lambda: M.build(config_path))
    finally:
        h.doCleanups()


def attack_raw_ppa(field_kind):
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, bundle, _):
            ppa = h._load_spec(bundle["ppa_receipt"])
            if field_kind == "logic_area":
                report = ppa["rows"][0]["logic_report"]
                h._rewrite_text(report, "logic_area_mm2 9999.0\nlogic_power_mw 0.2\n")
            else:
                report = ppa["rows"][0]["sram_report"]
                h._rewrite_text(report, "sram_macro_area_mm2 0.4\nsram_macro_power_mw 77.0\n")
            h._rewrite_json(bundle["ppa_receipt"], ppa)
            obj = M.M635.load_json(config_path, "fixture")
            obj["table_a_evidence_bundles"][bundle["bundle_id"]]["ppa_receipt"] = bundle["ppa_receipt"]
            candidate = h._config(obj)
            return reject("raw PPA " + field_kind + " mismatch", lambda: M.build(candidate))
    finally:
        h.doCleanups()


def attack_density_mid():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _):
            measurement = h._load_spec(bundle["measurement_identity"])
            population_spec = measurement["sequence_population_manifest"]
            population = h._load_spec(population_spec)
            population["samples"][1]["density_stratum"] = "mid"
            h._rewrite_json(population_spec, population)
            measurement["sequence_population_manifest"] = population_spec
            h._rewrite_json(bundle["measurement_identity"], measurement)
            return reject(
                "legacy mid density",
                lambda: M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256),
            )
    finally:
        h.doCleanups()


def attack_numerator_population():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _):
            measurement = M._validate_measurement_identity(bundle["measurement_identity"], M.CHECKPOINT_SHA256)
            numerator = h._load_spec(bundle["fixed_throughput_numerator_receipt"])
            numerator["population_scalar"] += 1
            h._rewrite_json(bundle["fixed_throughput_numerator_receipt"], numerator)
            return reject(
                "fixed numerator population mismatch",
                lambda: M._validate_numerator_receipt(bundle["fixed_throughput_numerator_receipt"], measurement),
            )
    finally:
        h.doCleanups()


def attack_missing_checkpoint_spec():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, bundle, _):
            obj = M.M635.load_json(config_path, "fixture")
            del obj["table_a_evidence_bundles"][bundle["bundle_id"]]["checkpoint"]
            candidate = h._config(obj)
            return reject("missing checkpoint file spec", lambda: M.build(candidate))
    finally:
        h.doCleanups()


def main():
    result = {
        "schema": "m657.m653.registry_r5.independent_attack_summary.r1",
        "canonical": canonical_check(),
        "accepted_adversarial_probe": accepted_scope_and_ppa_probe(),
        "rejected_attacks": [
            attack_arbitrary_request(),
            attack_wrong_reviewed_target(),
            attack_nonzero_review_p1(),
            attack_incomplete_evidence_map(),
            attack_raw_ppa("logic_area"),
            attack_raw_ppa("sram_power"),
            attack_density_mid(),
            attack_numerator_population(),
            attack_missing_checkpoint_spec(),
        ],
        "frozen_roots": {
            "m527": sha256(M.M527_CONTRACT),
            "checkpoint": sha256(M.CHECKPOINT),
            "docs359": sha256(DOCS359),
        },
        "temporary_fixture_persisted": False,
    }
    if not all(item["rejected"] for item in result["rejected_attacks"]):
        raise RuntimeError("one or more intended negative attacks were admitted")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

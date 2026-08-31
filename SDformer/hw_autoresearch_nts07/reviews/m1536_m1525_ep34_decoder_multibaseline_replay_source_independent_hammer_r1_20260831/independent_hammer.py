#!/usr/bin/env python3
"""Independent, CPU-light hammer for the M1525 source-only replay ladder."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1525_ep34_decoder_multibaseline_replay_successor_source.py"
TEST = HW / "system_simulator/tests/test_m1525_ep34_decoder_multibaseline_replay_successor_source.py"
CONTRACT = HW / "contracts/m1525_ep34_decoder_multibaseline_replay_successor_source_contract_r1_20260831.json"
M1526 = HW / "contracts/m1526_ep34_decoder_int8_numeric_bridge_gate_source_contract_r1_20260831.json"
M1514 = HW / "contracts/m1514_ep34_decoder_weight_identity_export_source_contract_r1_20260831.json"
M1515 = HW / "reviews/m1515_m1514_ep34_decoder_weight_identity_source_independent_hammer_r1_20260831/review.json"
M1515_MANIFEST = M1515.parent / "SHA256SUMS"
M1515_OUTER = M1515.parent / "SHA256SUMS.seal.sha256"
M1521 = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831/manifest.json"
M1527 = HW / "reviews/m1527_m1521_ep34_decoder_positive_plane_actual_result_hammer_r1_20260831/review.json"
M1527_MANIFEST = M1527.parent / "SHA256SUMS"
M1527_OUTER = M1527.parent / "SHA256SUMS.seal.sha256"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    SOURCE: "d52fa8b4d7a0f4395a4214f4209f449fcfb404fab7e90f12f179efe33405141a",
    TEST: "1c7d0027157d8a0942733c47738312f09807f0b42fc170479d6cad7f8982a632",
    CONTRACT: "9b1a1d383b46aca7cdfa1b1085432848849f3e6f235fe594ceb4bf068a9671b9",
    M1526: "529151b5d4b682f8cde483678f853b8ea01f1364e48106b2ca8867d1de477a36",
    M1515: "37b0e20082bc28d713a28d50bfc80f0fa3eca6062c2558f470efb9ba771e4990",
    M1515_MANIFEST: "a7d1b3c57617d27c6115e52dd0120e1d697428d6616f3b2e1981846d83018219",
    M1515_OUTER: "445b3ce95402b6e206ca279bafd3ce755bc91ff088157f42c3935a99a6779d3a",
    M1521: "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304",
    M1527: "366068b725a16c42fc69adc29c463ce909b0f528f6a31c36eb25e6914366c714",
    M1527_MANIFEST: "2cf68cdca714a68d25ac6cbd9ea2f9a9f9cfa323bd3d39a401c7f6636dec8a94",
    M1527_OUTER: "37841dfbd4f6d83d4004efdfa5e80e011396c9581fc8e2c0985ffec85274bb22",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON token " + token)))


def require(value, message):
    if not value:
        raise AssertionError(message)


def rejects(module, function, label):
    try:
        function()
    except module.M1525Error:
        return label
    raise AssertionError(label + " was accepted")


def weight_authority(module):
    authority = load(M1514)
    rows = []
    for ordinal, item in enumerate(authority["weight_identities"]):
        rows.append({
            "module_ordinal": ordinal,
            "module": item["checkpoint_key"][:-len(".weight")],
            "shape": item["shape"],
            "dtype": "torch.float32",
            "layout": "C_ORDER_CONTIGUOUS",
            "byte_order": "little",
            "content_sha256": item["content_sha256"],
            "content_bytes": item["content_bytes"],
            "bias": item["bias"],
        })
    return {
        "status": "PASS_M1514_SOURCE_ONLY_DECODER_WEIGHT_IDENTITY__NO_EXPORT",
        "checkpoint": {"sha256": module.CHECKPOINT_SHA256,
                       "root_keys": ["model_state_dict"]},
        "weights": rows,
    }


def main():
    checks = []
    for path, expected in PINS.items():
        require(path.is_file() and not path.is_symlink(), "unsafe/missing " + str(path))
        require(sha(path) == expected, "SHA drift " + str(path))
        checks.append("pin:" + path.name)
    require(M1515_OUTER.read_text(encoding="utf-8").split() ==
            [PINS[M1515_MANIFEST], "SHA256SUMS"] and
            M1527_OUTER.read_text(encoding="utf-8").split() ==
            [PINS[M1527_MANIFEST], "SHA256SUMS"],
            "upstream outer-seal content drift")
    checks.append("upstream_outer_seals")

    spec = importlib.util.spec_from_file_location("m1536_frozen_m1525", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    contract = load(CONTRACT)
    m1526 = load(M1526)
    m1515 = load(M1515)
    m1527 = load(M1527)
    planes = load(M1521)
    weights = weight_authority(module)

    require(contract["schema"] == module.SCHEMA and contract["status"] == module.STATUS,
            "contract/source schema or status drift")
    require(contract["configurations_in_order"] == list(module.CONFIGS),
            "configuration order drift")
    require(sum(module.COMMON_RESOURCE["partitions"].values()) == 245760,
            "240 KiB partition accounting drift")
    require(module.COMMON_RESOURCE["lanes"] == 96 and
            module.COMMON_RESOURCE["accumulator_bits"] == 24 and
            module.COMMON_RESOURCE["clock_ns"] == 3.0 and
            module.COMMON_RESOURCE["external_bytes_per_cycle"] == 192,
            "common resource headline drift")
    checks.extend(["contract", "resource_sum", "resource_axes"])

    plan = module.build_replay_plan(planes, weights)
    require(plan["readiness"] == {
        "dense_bit_k1x8_k8_source_plan_ready": True,
        "product_capture_ready": False,
        "product_blocker": "EP34_INT8_WEIGHT_BYTES_PLUS_MITER_PLUS_ACC24_PROOF_MISSING",
        "production": False,
    }, "default readiness is not fail closed")
    require(plan["identity"]["planes"]["calls"] == 120 and
            plan["identity"]["weights"]["layers"] == 4,
            "actual authority population drift")
    require(plan["claim_boundary"] == {
        "source_only": True, "production": False, "transactions": False,
        "cycles": False, "traffic": False, "speedup": False,
        "system_speedup": False, "energy": False, "rtl": False,
        "eda": False, "ppa": False, "table_a": False,
    }, "claim boundary drift")
    require(plan["old_m1105dr2_reuse"]["allowed"] is False and
            "old_cycles_and_traffic_are_diagnostic_only" in
            plan["old_m1105dr2_reuse"]["reasons"], "M1105 reuse is not forbidden")
    checks.extend(["actual_m1521_manifest", "m1514_weight_projection",
                   "default_product_block", "claim_boundary", "old_m1105_forbidden"])

    ladder = plan["configurations"]
    require([row["resource_manifest_sha256"] for row in ladder].count(
        ladder[0]["resource_manifest_sha256"]) == 4, "resource digest differs")
    require(len({row["commit_policy"] for row in ladder}) == 1 and
            len({row["d0_d1_value_policy"] for row in ladder}) == 1,
            "commit/numeric comparator differs")
    require([row["frontend_area_matched"] for row in ladder] ==
            [True, False, True, True], "K1x8 area label drift")
    require([row["product_bridge_required"] for row in ladder] ==
            [False, False, False, True], "product gate assignment drift")
    checks.extend(["shared_resource_digest", "shared_commit_numeric",
                   "equal_service_not_area_matched", "product_gate_assignment"])

    require(m1515["status"].startswith("PASS_M1515_") and
            m1527["status"].startswith("PASS_M1527_") and
            m1527["authorization"]["address_timed_replay_successor_authoring"] is True and
            m1527["authorization"]["production_rerun"] is False,
            "upstream independent authorities drift")
    require(m1526["claim_boundary"]["m1525_int8_replay_admitted"] is False and
            m1526["algorithm_handoff_summary"]["if_any_field_missing"] ==
            "KEEP_M1525_INT8_REPLAY_BLOCKED",
            "M1526 product blocker drift")
    checks.extend(["m1515_authority", "m1527_authority", "m1526_blocker"])

    mutations = []
    altered = deepcopy(planes)
    altered["records"][0]["global_call_ordinal"] = 1
    mutations.append(rejects(module, lambda: module.validate_positive_plane_manifest(altered),
                             "plane_order"))
    altered = deepcopy(planes)
    altered["records"][0]["positive_output_sha256"] = "A" * 64
    mutations.append(rejects(module, lambda: module.validate_positive_plane_manifest(altered),
                             "uppercase_sha"))
    altered = deepcopy(weights)
    altered["weights"][0]["content_bytes"] += 4
    mutations.append(rejects(module, lambda: module.validate_weight_identity(altered),
                             "weight_extent"))
    altered = deepcopy(weights)
    altered["checkpoint"]["root_keys"].append("optimizer")
    mutations.append(rejects(module, lambda: module.validate_weight_identity(altered),
                             "checkpoint_root"))
    mutations.append(rejects(module, lambda: module.build_replay_plan(
        planes, weights, request_production=True), "production_request"))
    comparator = [{"configuration": name, "resource_manifest_sha256": "r",
                   "commit_address_hash": "c", "population_manifest_sha256": "p",
                   "checkpoint_sha256": module.CHECKPOINT_SHA256}
                  for name in module.CONFIGS]
    altered = deepcopy(comparator)
    altered[2]["commit_address_hash"] = "different"
    mutations.append(rejects(module, lambda: module.validate_comparator_rows(altered),
                             "commit_address_mismatch"))
    require(len(mutations) == 6, "mutation population drift")
    checks.append("mutations_6_of_6_rejected")

    # This deliberately records a safety gap rather than treating it as release.
    # M1526 supersedes this optional shorthand bridge and keeps product replay blocked.
    weak_bridge = {
        "checkpoint_sha256": module.CHECKPOINT_SHA256,
        "four_int8_payload_sha256": [str(index) * 64 for index in range(1, 5)],
        "quantization_policy": None,
        "fp32_to_int8_miter": True,
        "acc24_bound": True,
        "independent_hammer_pass": True,
    }
    weak_plan = module.build_replay_plan(planes, weights, weak_bridge)
    require(weak_plan["readiness"]["product_capture_ready"] is True,
            "expected optional-bridge negative-test gap changed")
    checks.append("p1_optional_bridge_under_specified_observed")

    output = {
        "status": "PASS_M1536_M1525_SOURCE_FOR_THREE_NONPRODUCT_CONFIGS__PRODUCT_BLOCKED_BY_M1526",
        "checks": len(checks),
        "check_names": checks,
        "mutations_rejected": mutations,
        "python_compatibility": "M1525 source/test independently run under CPython 3.6",
        "production_authorized": False,
        "successor_authoring": {
            "DENSE_TYPED_K8": True,
            "BIT_EQUAL_SERVICE_K1X8": True,
            "BIT_TYPED_K8": True,
            "PRODUCT_CAPTURE_TYPED_K8": False,
        },
        "p1": "M1525 optional bridge shorthand accepts a null quantization policy and is not a release authority; M1526 Q1-Q4 plus a fresh independent hammer must supersede it.",
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

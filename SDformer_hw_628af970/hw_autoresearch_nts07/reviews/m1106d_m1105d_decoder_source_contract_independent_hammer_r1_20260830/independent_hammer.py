#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1106D receipt-blind source/contract hammer; no production/EDA/RTL."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
SOURCE = HW / "system_simulator/scripts/build_m1105d_decoder_only_address_timed_source.py"
CONTRACT = HW / "contracts/m1105d_decoder_only_address_timed_source_contract_r1_20260830.json"
RECEIPT_DIR = HW / "results/m1105d_decoder_only_address_timed_source_preflight_r1_20260830"
PAYLOAD_DIR = HW / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE / "mechanical_checks.json"

EXPECTED = {
    "source": "d110d0f559b0947c1941f64ce863ece8f953b6ded15d45bb8d5ffdb68973d411",
    "contract": "afca3302e5a0bff9386bb53cbc2d6e72bd5932cd71ffc47f80a36a97bd5ab241",
    "contract_sidecar": "59cf1301ad8d30c1fbab7f2757615ffed72fcc358bbec0f721f9db26cc3d27b6",
    "contract_outer": "341894018b8120bff7c653799877f716c51de66832df7dec739ccb3085f0b62a",
    "receipt": "95e80734c76f57ebc4d239c6df31c9f0994981d2fdec31003e6db86e969994c0",
    "receipt_manifest": "949d24549283a1645f0f4db67c1339b269137d46d299932096bc209be9f46366",
    "receipt_outer": "1cab0368fba2e1284fde3f19ca37b2e0bce0f32baa664a1c66220c76caf7a003",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
SEQUENCES = ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"]
MODULE_NAMES = [
    "sttmultires_unet.decoders.0.deconv.0",
    "sttmultires_unet.decoders.1.deconv.0",
    "sttmultires_unet.decoders.2.deconv.0",
    "sttmultires_unet.decoders.3.deconv.0",
]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> dict[str, str]:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require((sha(path), sha(side), sha(outer)) == identity and
            side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double-seal drift")
    return {"primary_sha256": identity[0], "sidecar_sha256": identity[1],
            "outer_seal_file_sha256": identity[2]}


def verify_receipt() -> dict[str, Any]:
    receipt = RECEIPT_DIR / "receipt.json"
    manifest = RECEIPT_DIR / "SHA256SUMS"
    outer = RECEIPT_DIR / "SHA256SUMS.seal.sha256"
    require(RECEIPT_DIR.is_dir() and not RECEIPT_DIR.is_symlink() and
            (sha(receipt), sha(manifest), sha(outer)) ==
            (EXPECTED["receipt"], EXPECTED["receipt_manifest"],
             EXPECTED["receipt_outer"]), "receipt identity drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        member = RECEIPT_DIR / name
        require(name not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "receipt member drift")
        listed[name] = digest
    actual = {path.name for path in RECEIPT_DIR.iterdir()
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(set(listed) == actual and
            outer.read_text(encoding="utf-8").split() ==
                [EXPECTED["receipt_manifest"], "SHA256SUMS"],
            "receipt coverage/outer drift")
    return {"receipt_sha256": sha(receipt),
            "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer),
            "members": len(listed),
            "manifest_is_not_outer_file_sha256":
                EXPECTED["receipt_manifest"] != EXPECTED["receipt_outer"]}


def load_source():
    require(sha(SOURCE) == EXPECTED["source"] and not SOURCE.is_symlink(),
            "source identity drift")
    spec = importlib.util.spec_from_file_location("m1106d_frozen_m1105d", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_projection(receipt: dict[str, Any], contract: dict[str, Any],
                        manifest: dict[str, Any]) -> None:
    require(receipt.get("schema") == "m1105d_decoder_only_source_preflight_receipt_v1" and
            receipt.get("status") ==
                "PASS_SOURCE_AND_FULL_IDENTITY_PREFLIGHT__PRODUCTION_NOT_RELEASED" and
            receipt.get("contract_sha256") == EXPECTED["contract"],
            "receipt identity projection")
    population = receipt.get("population", {})
    require(population == {"sequences": 3, "samples": 30, "calls": 120,
                           "packed_bytes": 261090000,
                           "global_ordinals_contiguous": True,
                           "per_sample_module_order": ["D0", "D1", "D2", "D3"]},
            "population projection")
    calls = receipt.get("calls")
    require(type(calls) is list and len(calls) == 120, "call population")
    manifest_rows = manifest["records"]
    for ordinal, call in enumerate(calls):
        module = ordinal % 4
        sample = ordinal // 4
        row = manifest_rows[ordinal]
        require(call["global_ordinal"] == ordinal and
                call["global_sample_id"] == sample and
                call["module_ordinal"] == module and
                call["module"] == MODULE_NAMES[module] and
                call["sequence_ordinal"] == sample // 10 and
                call["sequence"] == SEQUENCES[sample // 10] and
                call["sequence_sample_id"] == sample % 10 and
                call["payload_relative_path"] == row["relative_path"],
                "call order/identity projection")
        stats = (row["statistics"]["scaled_binary_audit"] if module == 1
                 else row["statistics"])
        require(call["payload_sha256"] == stats["packed_sha256"],
                "payload SHA projection")
        numeric = call["numeric_source"]
        if module == 1:
            require(numeric == {"encoding": "bit_times_exact_theta_word",
                                "theta_word": 1065353139,
                                "theta_le_hex": "b3ff7f3f",
                                "weight_folding": False}, "D1 numeric projection")
        else:
            require(numeric == {"encoding": "exact_binary", "theta_word": None,
                                "weight_folding": False}, "binary numeric projection")
        addresses = call["address_regions"]
        stride = 1 << 32
        expected = {
            "input_descriptor_base": (1 << 60) + ordinal * stride,
            "weight_base": (2 << 60) + module * stride,
            "psum_base": (3 << 60) + ordinal * stride,
            "output_commit_base": (4 << 60) + ordinal * stride,
            "control_descriptor_base": (5 << 60) + ordinal * stride,
            "per_call_region_bytes": stride,
        }
        require(addresses == expected and len({value >> 60 for key, value in addresses.items()
                                               if key.endswith("_base")}) == 5,
                "address-region projection/overlap")
    miter = receipt.get("d1_exact_scaled_binary_miter", {})
    require(miter.get("theta_word") == 1065353139 and
            miter.get("theta_le_hex") == "b3ff7f3f" and
            miter.get("calls_checked") == 30 and miter.get("mismatches") == 0 and
            miter.get("folded_weights") is False and
            miter.get("coerced_to_one") is False and
            [row["global_call_index"] for row in miter.get("records", [])] ==
                list(range(1, 120, 4)) and
            all(row["mismatch"] is False and
                row["reconstructed_raw_fp32_sha256"] == row["expected_raw_fp32_sha256"]
                for row in miter["records"]), "D1 miter projection")
    resource = receipt.get("common_resource_schedule_schema")
    require(resource == contract["common_resource_schedule_schema"] and
            resource["lanes"] == 96 and resource["accumulator_bits"] == 24 and
            resource["clock_ns"] == 3.0 and
            resource["external_bytes_per_cycle"] == 192 and
            resource["onchip_sram_bytes_macro_rounded"] == 245760 and
            sum(resource["partitions"].values()) == 245760,
            "common resource projection")
    schema = receipt.get("transaction_event_schema")
    require(schema == contract["transaction_event_schema"] and
            schema["required_dependency_fields"] ==
                ["dependency_tokens", "produces_token"] and
            schema["required_time_fields"] == ["earliest_issue_cycle",
                "dependency_ready_cycle", "issue_cycle", "return_cycle",
                "commit_cycle", "stall_class"] and
            "remain absent" in schema["time_policy"],
            "dependency/time schema projection")
    release = receipt.get("release", {})
    boundary = receipt.get("claim_boundary", {})
    require(release == {"production_run_allowed": False,
                        "requires_different_author_hammer": True,
                        "production_cycles": None, "speedup": None,
                        "system_speedup_admitted": False} and
            all(boundary.get(key) is False for key in
                ("production_transactions", "cycles", "traffic", "speedup",
                 "system_speedup", "ours_performance", "rtl", "eda", "energy")) and
            receipt["input_identity"]["checkpoint_sha256"] ==
                contract["population"]["checkpoint_sha256"] and
            receipt["input_identity"]["final_checkpoint_rebind_required_if_changed"] is True and
            "m700" not in json.dumps(receipt, sort_keys=True).lower(),
            "release/checkpoint/M700 boundary")


def mutation_tests(receipt: dict[str, Any], contract: dict[str, Any],
                   manifest: dict[str, Any]) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    def run(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        value = copy.deepcopy(receipt)
        mutate(value)
        try:
            validate_projection(value, contract, manifest)
        except (RuntimeError, KeyError, IndexError, TypeError, ValueError):
            tests[name] = True
        else:
            tests[name] = False
    run("call_reorder", lambda value: value["calls"].__setitem__(slice(0, 2),
        [value["calls"][1], value["calls"][0]]))
    run("missing_d1", lambda value: value["calls"].pop(1))
    run("duplicate_global_ordinal", lambda value: value["calls"][1].__setitem__(
        "global_ordinal", 0))
    run("payload_path", lambda value: value["calls"][0].__setitem__(
        "payload_relative_path", "calls/forged.bitpack"))
    run("payload_sha", lambda value: value["calls"][0].__setitem__(
        "payload_sha256", "0" * 64))
    run("d1_endian", lambda value: value["calls"][1]["numeric_source"].__setitem__(
        "theta_le_hex", "3f7fffb3"))
    run("d1_theta_word", lambda value: value["calls"][1]["numeric_source"].__setitem__(
        "theta_word", 1065353216))
    run("d1_weight_folding", lambda value: value["calls"][1]["numeric_source"].__setitem__(
        "weight_folding", True))
    run("d1_force_one", lambda value: value["d1_exact_scaled_binary_miter"].__setitem__(
        "coerced_to_one", True))
    run("address_overlap", lambda value: value["calls"][0]["address_regions"].__setitem__(
        "psum_base", value["calls"][0]["address_regions"]["input_descriptor_base"]))
    run("missing_dependency_field", lambda value: value["transaction_event_schema"].__setitem__(
        "required_dependency_fields", ["dependency_tokens"]))
    run("timestamp_policy", lambda value: value["transaction_event_schema"].__setitem__(
        "time_policy", "timestamps may be caller supplied"))
    run("lanes", lambda value: value["common_resource_schedule_schema"].__setitem__("lanes", 95))
    run("acc24", lambda value: value["common_resource_schedule_schema"].__setitem__(
        "accumulator_bits", 23))
    run("sram240k", lambda value: value["common_resource_schedule_schema"].__setitem__(
        "onchip_sram_bytes_macro_rounded", 262144))
    run("external_192B", lambda value: value["common_resource_schedule_schema"].__setitem__(
        "external_bytes_per_cycle", 191))
    run("clock_3ns", lambda value: value["common_resource_schedule_schema"].__setitem__(
        "clock_ns", 4.0))
    run("M700_injected", lambda value: value.__setitem__("m700_speedup", 3.088))
    run("final_checkpoint_mislabeled", lambda value: value["input_identity"].__setitem__(
        "final_checkpoint_rebind_required_if_changed", False))
    return tests


def forged_contract_attack(module, contract: dict[str, Any]) -> dict[str, Any]:
    forged = copy.deepcopy(contract)
    forged["population"]["checkpoint"] = "FORGED_FINAL_CHECKPOINT"
    forged["population"]["checkpoint_sha256"] = "0" * 64
    forged["population"]["final_checkpoint_policy"] = "reuse ep35 without rebinding"
    forged["d1_numeric_contract"]["theta_word_uint32"] = 1065353216
    forged["d1_numeric_contract"]["theta_ieee754_le_hex"] = "0000803f"
    forged["d1_numeric_contract"]["weight_folding_allowed"] = True
    forged["d1_numeric_contract"]["coercion_to_binary_one_allowed"] = True
    resource = forged["common_resource_schedule_schema"]
    resource["lanes"] = 95
    resource["accumulator_bits"] = 23
    resource["clock_ns"] = 4.0
    resource["external_bytes_per_cycle"] = 191
    resource["onchip_sram_bytes_macro_rounded"] = 262144
    resource["address_regions"]["psum"] = resource["address_regions"]["input_descriptor"]
    schema = forged["transaction_event_schema"]
    schema["required_dependency_fields"] = []
    schema["required_time_fields"] = ["caller_timestamp"]
    schema["time_policy"] = "caller supplies timestamps"
    with tempfile.TemporaryDirectory(prefix="m1106d_forged_contract.") as temp:
        path = Path(temp) / "forged_contract.json"
        path.write_text(json.dumps(forged, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
        accepted = None
        error = None
        try:
            accepted = module.build(REPO, path)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        return {
            "accepted": accepted is not None,
            "error": error,
            "status_if_accepted": None if accepted is None else accepted.get("status"),
            "returned_mutated_resource": False if accepted is None else
                accepted.get("common_resource_schedule_schema") == resource,
            "returned_mutated_dependency_schema": False if accepted is None else
                accepted.get("transaction_event_schema") == schema,
            "canonical_contract_sha_required_by_source": False,
            "caller_contract_path_controls_semantic_schema": accepted is not None,
        }


def main() -> None:
    contract_identity = verify_double(CONTRACT, (EXPECTED["contract"],
        EXPECTED["contract_sidecar"], EXPECTED["contract_outer"]))
    receipt_identity = verify_receipt()
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359 drift")
    contract = strict_json(CONTRACT)
    receipt = strict_json(RECEIPT_DIR / "receipt.json")
    manifest = strict_json(PAYLOAD_DIR / "manifest.json")
    validate_projection(receipt, contract, manifest)
    mutations = mutation_tests(receipt, contract, manifest)
    require(all(mutations.values()), "independent projection mutation escaped")
    module = load_source()
    require(module.d1_scaled_binary_raw_sha and
            "os.environ" not in SOURCE.read_text(encoding="utf-8") and
            "getenv" not in SOURCE.read_text(encoding="utf-8"),
            "source/env static drift")
    wrong_theta_rejected = False
    try:
        module.d1_scaled_binary_raw_sha(
            PAYLOAD_DIR / manifest["records"][1]["relative_path"], 1, 1065353216)
    except Exception:
        wrong_theta_rejected = True
    require(wrong_theta_rejected, "wrong D1 theta accepted")
    forged = forged_contract_attack(module, contract)
    m700 = copy.deepcopy(contract)
    m700["external_m700_speedup"] = 3.088
    with tempfile.TemporaryDirectory(prefix="m1106d_m700.") as temp:
        path = Path(temp) / "m700.json"
        path.write_text(json.dumps(m700, sort_keys=True) + "\n", encoding="utf-8")
        try:
            module.build(REPO, path)
        except Exception:
            m700_rejected = True
        else:
            m700_rejected = False
    output = {
        "schema": "m1106d_m1105d_decoder_source_contract_hammer_checks_v1",
        "receipt_blind": True,
        "identity": {"source_sha256": sha(SOURCE),
                     "contract": contract_identity,
                     "receipt": receipt_identity,
                     "docs359_sha256": sha(DOCS359)},
        "race_observation": {
            "initial_discovery_preceded_receipt_seal": True,
            "receipt_used_only_after_exact_manifest_and_outer_file_identity_appeared": True,
        },
        "terminology": {
            "author_supplied_949d_called_outer": True,
            "actual_manifest_sha256": EXPECTED["receipt_manifest"],
            "actual_outer_seal_file_sha256": EXPECTED["receipt_outer"],
            "contract_misuses_manifest_as_outer_file_sha": False,
        },
        "canonical_projection_validation": "PASS",
        "mutation_rejections": mutations,
        "mutations_rejected": sum(mutations.values()),
        "mutations_total": len(mutations),
        "wrong_d1_theta_helper_rejected": wrong_theta_rejected,
        "m700_contract_injection_rejected": m700_rejected,
        "forged_caller_contract_attack": forged,
        "environment_authority": {
            "environment_read_by_source": False,
            "caller_repo_root_argument_present": True,
            "caller_contract_path_argument_present": True,
            "caller_output_path_argument_present": True,
        },
        "scope": {
            "canonical_source_or_contract_modified": False,
            "production_runner_created": False,
            "production_attempt_created": False,
            "production_transactions_enumerated": False,
            "eda_rtl_gpu_remote_used": False,
            "temporary_forged_contract_only": True,
            "docs359_modified": False,
        },
        "verdict": ("STOP_M1106D_CALLER_CONTRACT_CAN_REWRITE_RESOURCE_NUMERIC_"
                    "DEPENDENCY_ADDRESS_AND_CHECKPOINT_SCHEMA" if forged["accepted"] else
                    "GO_M1106D_SOURCE_CONTRACT_HAMMER__RUNNER_AUTHORING_ONLY"),
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True,
                              allow_nan=False) + "\n", encoding="utf-8")
    print(output["verdict"])


if __name__ == "__main__":
    main()

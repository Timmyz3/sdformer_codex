#!/usr/bin/env python3
"""M1290 additive repair of the frozen M1281 decoder surrogate adapter.

The public fixture and production APIs are deliberately separate.  The fixture
API can never admit an analytical annex.  The zero-argument production API
opens only a canonical, fully sealed M1111DR2 result and a separately sealed
M1291 different-author result hammer; it accepts no caller booleans, paths,
PASS strings, or naked SHA values as authority.
"""
from collections import Counter
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Dict, Iterable, List, Mapping, Tuple

getcontext().prec = 50

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PRODUCTION_RESULT = HW / "results/m1111dr2_m1105dr2_decoder_only_address_timed_production_r2_20260830"
PRODUCTION_HAMMER = HW / "reviews/m1291_m1111dr2_decoder_production_result_independent_hammer_r1_20260830"

PAYLOAD = "m1111dr2_decoder_result.json"
CALLS = "m1111dr2_decoder_call_schedule.jsonl"
COMPLETE = "RUN_COMPLETE.txt"
SEAL_DIR = ".m1111dr2_atomic_seal"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RESULT_FILES = (PAYLOAD, CALLS, COMPLETE)
KINDS = ("input_descriptor_read", "weight_read", "psum_read", "compute",
         "psum_write", "output_commit")
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
MODULES = (
    "sttmultires_unet.decoders.0.deconv.0",
    "sttmultires_unet.decoders.1.deconv.0",
    "sttmultires_unet.decoders.2.deconv.0",
    "sttmultires_unet.decoders.3.deconv.0",
)
LAYERS = ("D0", "D1", "D2", "D3")
CONFIGURATION = "M1105DR2_EXACT_TYPED_K8"
EXPECTED_CALLS = 120
EXPECTED_SAMPLES = 30
CALLS_PER_LAYER = 30
SLOPE = 4
MAX_RELATIVE_ERROR = Decimal("0.001")
EXPECTED_COMMIT_BYTES = (13824000, 27648000, 55296000, 221184000)
EXPECTED_COMMON_RESOURCE_SHA256 = "bfc3c19baec8ed8055d7274c022667f18523575563eba0f5f940944aee531137"
HAMMER_SCHEMA = "m1291_m1111dr2_decoder_production_result_independent_hammer_v1"
HAMMER_STATUS = "PASS_M1291_M1111DR2_DECODER_PRODUCTION_RESULT_INDEPENDENT_HAMMER__DIAGNOSTIC_ONLY"


class CalibrationError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CalibrationError(message)


def exact_keys(value: Any, expected: Iterable[str], label: str) -> None:
    require(type(value) is dict and set(value) == set(expected), label + " key drift")


def strict_json_text(text: str) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CalibrationError("nonfinite JSON: " + token)))


def strict_json_file(path: Path) -> Any:
    return strict_json_text(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    require(path.exists() and stat.S_ISREG(path.lstat().st_mode) and
            not path.is_symlink(), label + " must be regular non-symlink")


def lowercase_sha(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " SHA drift")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_result_seal(directory: Path) -> Dict[str, Any]:
    """Verify exact three payload files plus the nested manifest/outer seal."""
    require(directory.is_dir() and not directory.is_symlink(), "result directory drift")
    require({item.name for item in directory.iterdir()} == set(RESULT_FILES) | {SEAL_DIR},
            "result top-level set is not exact three files plus seal directory")
    for name in RESULT_FILES:
        regular(directory / name, "result " + name)
    bundle = directory / SEAL_DIR
    require(bundle.is_dir() and not bundle.is_symlink() and
            {item.name for item in bundle.iterdir()} == {MANIFEST, OUTER},
            "result seal bundle drift")
    manifest, outer = bundle / MANIFEST, bundle / OUTER
    regular(manifest, "result manifest"); regular(outer, "result outer seal")
    manifest_sha = sha256(manifest)
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  " + MANIFEST + "\n",
            "result outer seal content drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "result manifest syntax drift")
        digest, relative = fields
        lowercase_sha(digest, "result manifest member")
        require(relative in RESULT_FILES and relative not in listed,
                "result manifest coverage/path drift")
        regular(directory / relative, "result manifest member")
        require(sha256(directory / relative) == digest, "result member digest drift")
        listed[relative] = digest
    require(set(listed) == set(RESULT_FILES), "result manifest does not cover exact three files")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(outer), "members": listed}


def verify_hammer_seal(directory: Path, result_identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify a flat different-author hammer and its exact linkage to result."""
    require(directory.is_dir() and not directory.is_symlink(), "hammer directory drift")
    manifest, outer, review = directory / MANIFEST, directory / OUTER, directory / "review.json"
    regular(manifest, "hammer manifest"); regular(outer, "hammer outer seal")
    regular(review, "hammer review")
    manifest_sha = sha256(manifest)
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  " + MANIFEST + "\n",
            "hammer outer seal drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "hammer manifest syntax drift")
        digest, relative = fields
        lowercase_sha(digest, "hammer member")
        member = directory / relative
        require(relative not in listed and len(Path(relative).parts) == 1 and
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "hammer manifest path drift")
        regular(member, "hammer member")
        require(sha256(member) == digest, "hammer member digest drift")
        listed.add(relative)
    actual = {item.name for item in directory.iterdir() if item.is_file() and
              item.name not in {MANIFEST, OUTER}}
    require(actual == listed and "review.json" in listed, "hammer manifest coverage drift")
    value = strict_json_file(review)
    exact_keys(value, ("schema", "status", "identity", "verification", "claim_boundary"),
               "hammer review")
    require(value["schema"] == HAMMER_SCHEMA and value["status"] == HAMMER_STATUS,
            "hammer schema/status drift")
    exact_keys(value["identity"], ("result_manifest_sha256",
        "result_outer_seal_file_sha256", "result_payload_sha256",
        "result_calls_sha256", "result_completion_sha256"), "hammer identity")
    expected_identity = {
        "result_manifest_sha256": result_identity["manifest_sha256"],
        "result_outer_seal_file_sha256": result_identity["outer_seal_file_sha256"],
        "result_payload_sha256": result_identity["members"][PAYLOAD],
        "result_calls_sha256": result_identity["members"][CALLS],
        "result_completion_sha256": result_identity["members"][COMPLETE],
    }
    require(value["identity"] == expected_identity, "hammer/result cryptographic linkage drift")
    required_verification = {
        "exact_three_payload_files": True, "result_manifest_and_outer_seal": True,
        "strict_120_call_rows": True, "kind_summaries_and_digests": True,
        "diagnostic_claim_boundary": True,
    }
    required_boundary = {"diagnostic_only": True, "analytical_annex": False,
        "speedup": False, "system_speedup": False, "paper_ppa_ready": False}
    require(value["verification"] == required_verification and
            value["claim_boundary"] == required_boundary,
            "hammer sealed verification/claim drift")
    return {"review_sha256": sha256(review), "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(outer), "status": value["status"]}


SUMMARY_KEYS = ("count", "traffic_bytes", "address_first", "address_last",
                "issue_first", "issue_last", "return_first", "return_last",
                "commit_first", "commit_last", "stall_events")
ROW_KEYS = ("schema", "global_call_ordinal", "sequence_ordinal", "sequence",
    "sequence_sample_id", "module_ordinal", "module", "configuration",
    "d1_exact_theta", "d1_theta_word_uint32", "d1_weight_folding",
    "transaction_ordinal_first", "transaction_ordinal_last", "transaction_count",
    "address_digest_sha256", "dependency_digest_sha256", "schedule_digest_sha256",
    "cycle_start", "cycle_end", "diagnostic_cycles", "diagnostic_traffic_bytes",
    "kind_summaries", "claim_boundary")


def validate_summary(summary: Any, kind: str, traffic: int) -> Dict[str, Any]:
    exact_keys(summary, SUMMARY_KEYS, "kind summary " + kind)
    require(type(summary["count"]) is int and summary["count"] >= 0 and
            type(summary["traffic_bytes"]) is int and
            summary["traffic_bytes"] == traffic and
            type(summary["stall_events"]) is dict and
            all(type(key) is str and type(value) is int and value >= 0
                for key, value in summary["stall_events"].items()) and
            sum(summary["stall_events"].values()) == summary["count"],
            "kind summary count/traffic/stall drift: " + kind)
    for key in SUMMARY_KEYS[2:-1]:
        require(summary[key] is None or type(summary[key]) is int,
                "kind summary endpoint drift: " + kind)
    return summary


def project_call(row: Any, ordinal: int, expected_transaction: int,
                 expected_cycle: int) -> Tuple[Dict[str, Any], int, int]:
    exact_keys(row, ROW_KEYS, "M1111DR2 call row")
    sample = ordinal // 4; module = ordinal % 4; sequence_ordinal = sample // 10
    require(row["schema"] == "m1111dr2_decoder_address_timed_call_schedule_v2" and
            type(row["global_call_ordinal"]) is int and
            row["global_call_ordinal"] == ordinal and
            type(row["sequence_ordinal"]) is int and
            row["sequence_ordinal"] == sequence_ordinal and
            row["sequence"] == SEQUENCES[sequence_ordinal] and
            type(row["sequence_sample_id"]) is int and
            row["sequence_sample_id"] == sample % 10 and
            type(row["module_ordinal"]) is int and row["module_ordinal"] == module and
            row["module"] == MODULES[module] and row["configuration"] == CONFIGURATION,
            "call identity/order drift")
    require(row["d1_exact_theta"] is (module == 1) and
            row["d1_theta_word_uint32"] == (1065353139 if module == 1 else None) and
            row["d1_weight_folding"] is False, "D1 numeric identity drift")
    for name in ("address_digest_sha256", "dependency_digest_sha256",
                 "schedule_digest_sha256"):
        lowercase_sha(row[name], name)
    first, last, count = (row["transaction_ordinal_first"],
                          row["transaction_ordinal_last"], row["transaction_count"])
    start, end, cycles = row["cycle_start"], row["cycle_end"], row["diagnostic_cycles"]
    require(type(first) is int and first == expected_transaction and
            type(count) is int and count > 0 and type(last) is int and
            last == first + count - 1 and type(start) is int and
            start == expected_cycle and type(end) is int and end > start and
            type(cycles) is int and cycles == end - start, "call interval drift")
    traffic = row["diagnostic_traffic_bytes"]
    exact_keys(traffic, (*KINDS, "total", "external", "onchip"), "call traffic")
    require(all(type(traffic[key]) is int and traffic[key] >= 0 for key in traffic) and
            traffic["total"] == sum(traffic[kind] for kind in KINDS) and
            traffic["external"] == traffic["input_descriptor_read"] + traffic["output_commit"] and
            traffic["onchip"] == traffic["weight_read"] + traffic["psum_read"] +
                traffic["psum_write"], "traffic conservation drift")
    summaries = row["kind_summaries"]
    exact_keys(summaries, KINDS, "kind summaries")
    for kind in KINDS:
        validate_summary(summaries[kind], kind, traffic[kind])
    groups = summaries["input_descriptor_read"]["count"]
    require(groups > 0 and all(summaries[kind]["count"] == groups for kind in
            ("weight_read", "psum_read", "compute", "psum_write")),
            "descriptor/weight/psum/compute/write group-count drift")
    require(traffic["input_descriptor_read"] == 16 * groups and
            traffic["psum_read"] == 288 * groups and
            traffic["compute"] == 288 * groups and
            traffic["psum_write"] == 288 * groups and
            traffic["weight_read"] % 16 == 0, "group-derived traffic drift")
    terms = traffic["weight_read"] // 16
    require(groups <= terms <= 8 * groups,
            "required group_count <= active_source_terms <= 8*group_count violated")
    commit_bytes = summaries["output_commit"]["traffic_bytes"]
    require(commit_bytes == traffic["output_commit"] and
            summaries["output_commit"]["count"] * 288 == commit_bytes and
            commit_bytes == EXPECTED_COMMIT_BYTES[module],
            "per-call output_commit traffic drift")
    require(count == sum(summaries[kind]["count"] for kind in KINDS),
            "transaction count/kind count drift")
    claim = row["claim_boundary"]
    expected_claim = {"diagnostic_only": True, "speedup_admitted": False,
        "system_speedup_admitted": False, "paper_ppa_ready": False,
        "final_checkpoint_rebind_required": True}
    require(claim == expected_claim, "call claim boundary drift")
    projection = {
        "global_call_ordinal": ordinal,
        "sequence_ordinal": sequence_ordinal,
        "sequence": row["sequence"],
        "sequence_sample_id": row["sequence_sample_id"],
        "module_ordinal": module,
        "module": row["module"],
        "layer": LAYERS[module],
        "configuration": row["configuration"],
        "address_digest_sha256": row["address_digest_sha256"],
        "dependency_digest_sha256": row["dependency_digest_sha256"],
        "schedule_digest_sha256": row["schedule_digest_sha256"],
        "kind_summary_digest_sha256": hashlib.sha256(
            canonical_json(summaries).encode()).hexdigest(),
        "group_count": groups,
        "active_source_terms": terms,
        "measured_cycles": cycles,
        "traffic": {
            "descriptor_bytes": traffic["input_descriptor_read"],
            "weight_bytes": traffic["weight_read"],
            "psum_read_bytes": traffic["psum_read"],
            "compute_count": groups,
            "compute_bytes": traffic["compute"],
            "psum_write_bytes": traffic["psum_write"],
            "commit_bytes": commit_bytes,
            "total_transaction_bytes": traffic["total"],
        },
    }
    return projection, last + 1, end


def validate_result_payload(directory: Path, seal: Mapping[str, Any]) -> List[Dict[str, Any]]:
    require((directory / COMPLETE).read_bytes() ==
            b"M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n",
            "completion token drift")
    payload = strict_json_file(directory / PAYLOAD)
    exact_keys(payload, ("schema", "status", "identity", "population",
        "common_resource", "diagnostic", "claim_boundary"), "result payload")
    require(payload["schema"] == "m1111dr2_m1105dr2_decoder_only_address_timed_result_v2" and
            payload["status"] ==
              "PASS_M1111DR2_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED",
            "result schema/status drift")
    identity = payload["identity"]
    exact_keys(identity, ("checkpoint", "checkpoint_sha256", "source_sha256",
        "contract_sha256", "m1110d_outer_seal_file_sha256",
        "final_checkpoint_rebind_required"), "result identity")
    require(identity == {"checkpoint": "H67_ep35",
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "source_sha256": "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
            "contract_sha256": "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
            "m1110d_outer_seal_file_sha256":
                "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
            "final_checkpoint_rebind_required": True}, "result identity/rebind drift")
    claim = payload["claim_boundary"]
    require(claim.get("decoder_only") is True and
            claim.get("address_timed_transactions_complete") is True and
            claim.get("same_resource_schedule_complete") is True and
            claim.get("diagnostic_cycles_only") is True and
            claim.get("diagnostic_traffic_only") is True and
            claim.get("speedup_admitted") is False and
            claim.get("system_speedup_admitted") is False and
            claim.get("paper_ppa_ready") is False and
            claim.get("paper_citable_performance") is False and
            claim.get("final_checkpoint_rebind_required") is True and
            claim.get("independent_result_hammer_required") is True,
            "result claim boundary drift")
    population = payload["population"]
    exact_keys(population, ("calls", "timesteps_per_call", "transaction_count",
        "call_schedule_sha256", "call_row_stream_digest_sha256"), "result population")
    require(population["calls"] == EXPECTED_CALLS and
            population["timesteps_per_call"] == 10 and
            type(population["transaction_count"]) is int and population["transaction_count"] > 0 and
            population["call_schedule_sha256"] == seal["members"][CALLS] and
            population["call_row_stream_digest_sha256"] == seal["members"][CALLS],
            "result population/schedule identity drift")
    require(type(payload["common_resource"]) is dict and
            hashlib.sha256(canonical_json(payload["common_resource"]).encode()).hexdigest() ==
                EXPECTED_COMMON_RESOURCE_SHA256, "result common-resource projection drift")
    rows = []
    raw_digest = hashlib.sha256(); expected_transaction = 0; expected_cycle = 0
    aggregate = Counter()
    with (directory / CALLS).open("rb") as stream:
        for ordinal, raw in enumerate(stream):
            require(ordinal < EXPECTED_CALLS and raw.endswith(b"\n") and raw.strip(),
                    "call JSONL framing/count drift")
            text = raw.decode("utf-8"); row = strict_json_text(text)
            require(canonical_json(row) + "\n" == text, "call JSONL canonical encoding drift")
            projected, expected_transaction, expected_cycle = project_call(
                row, ordinal, expected_transaction, expected_cycle)
            aggregate.update(row["diagnostic_traffic_bytes"])
            rows.append(projected); raw_digest.update(raw)
    require(len(rows) == EXPECTED_CALLS and raw_digest.hexdigest() == seal["members"][CALLS] and
            population.get("transaction_count") == expected_transaction,
            "120-row stream/transaction identity drift")
    diagnostic = payload["diagnostic"]
    exact_keys(diagnostic, ("cycles", "traffic_bytes", "ratios_or_speedups"),
               "result diagnostic")
    exact_keys(diagnostic["traffic_bytes"], (*KINDS, "total", "external", "onchip"),
               "result diagnostic traffic")
    require(type(diagnostic["cycles"]) is int and diagnostic["cycles"] == expected_cycle and
            diagnostic["traffic_bytes"] == dict(aggregate) and
            diagnostic["ratios_or_speedups"] is None,
            "result diagnostic aggregate projection drift")
    samples = {(row["sequence_ordinal"], row["sequence"], row["sequence_sample_id"])
               for row in rows}
    require(len(samples) == EXPECTED_SAMPLES and
            {row["sequence"] for row in rows} == set(SEQUENCES) and
            {row["module"] for row in rows} == set(MODULES) and
            len({(row["sequence"], row["sequence_sample_id"], row["module"])
                 for row in rows}) == EXPECTED_CALLS,
            "required 30 sample identities/3 sequences/4 modules drift")
    for module in range(4):
        layer_rows = [row for row in rows if row["module_ordinal"] == module]
        observations = {(row["address_digest_sha256"], row["dependency_digest_sha256"],
            row["schedule_digest_sha256"], row["kind_summary_digest_sha256"],
            row["group_count"], row["active_source_terms"], row["measured_cycles"],
            row["traffic"]["commit_bytes"]) for row in layer_rows}
        require(len(layer_rows) == CALLS_PER_LAYER and len(observations) == CALLS_PER_LAYER,
                "each layer requires 30 distinct observations")
    return rows


def verify_production_authorities(result_dir: Path,
                                  hammer_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result_seal = verify_result_seal(result_dir)
    rows = validate_result_payload(result_dir, result_seal)
    hammer = verify_hammer_seal(hammer_dir, result_seal)
    return rows, {"result": result_seal, "hammer": hammer}


def fit_projected(rows: List[Dict[str, Any]], production_verified: bool,
                  synthetic_fixture: bool) -> Dict[str, Any]:
    require(type(production_verified) is bool and type(synthetic_fixture) is bool,
            "fixture/production flags must be exact bool")
    require(production_verified is not synthetic_fixture,
            "exactly one fixture/production domain must be active")
    require(len(rows) == EXPECTED_CALLS, "projected row count drift")
    layer_receipts = {}
    all_absolute = []; all_relative = []
    for module, layer in enumerate(LAYERS):
        selected = [row for row in rows if row["module_ordinal"] == module]
        require(len(selected) == CALLS_PER_LAYER, "layer call count drift")
        residuals = [Decimal(row["measured_cycles"] - SLOPE * row["group_count"])
                     for row in selected]
        constant = sum(residuals) / Decimal(len(residuals))
        absolute = []; relative = []
        for row in selected:
            predicted = Decimal(SLOPE * row["group_count"]) + constant
            measured = Decimal(row["measured_cycles"])
            error = abs(predicted - measured)
            absolute.append(error); relative.append(error / measured)
        all_absolute.extend(absolute); all_relative.extend(relative)
        layer_receipts[layer] = {
            "calls": len(selected), "fitted_constant": format(constant, "f"),
            "mean_absolute_error": format(sum(absolute) / Decimal(len(absolute)), "f"),
            "max_absolute_error": format(max(absolute), "f"),
            "mean_relative_error": format(sum(relative) / Decimal(len(relative)), "f"),
            "max_relative_error": format(max(relative), "f"),
        }
    gate = max(all_relative) <= MAX_RELATIVE_ERROR
    annex = bool(production_verified and not synthetic_fixture and gate)
    aggregate = Counter()
    for row in rows:
        aggregate.update(row["traffic"])
    return {
        "schema": "m1290_decoder_surrogate_calibration_result_v1",
        "status": ("PASS_M1290_PRODUCTION_CALIBRATION_ERROR_GATE__ANALYTICAL_ANNEX_ONLY"
                   if annex else "PASS_M1290_CALIBRATION_ONLY__ANNEX_FALSE"),
        "calibration_only": True,
        "population": {"calls": 120, "samples": 30, "sequences": 3,
                       "modules": 4, "distinct_observations_per_layer": 30},
        "cycle_surrogate": {"slope_cycles_per_group": 4, "layers": layer_receipts,
            "global_mean_absolute_error": format(sum(all_absolute) / Decimal(120), "f"),
            "global_max_absolute_error": format(max(all_absolute), "f"),
            "global_mean_relative_error": format(sum(all_relative) / Decimal(120), "f"),
            "global_max_relative_error": format(max(all_relative), "f"),
            "error_gate_lte": "0.001", "error_gate_pass": gate,
            "analytical_cycle_annex_allowed": annex},
        "exact_diagnostic_traffic": dict(sorted(aggregate.items())),
        "claim_boundary": {"calibration_only": True, "analytical_cycle_annex": annex,
            "traffic_model": False, "speedup_admitted": False,
            "system_speedup_admitted": False, "paper_ppa_ready": False,
            "energy": False},
    }


def calibrate_fixture(payload: Mapping[str, Any], synthetic_fixture: bool) -> Dict[str, Any]:
    """Fixture-only API.  Exact Boolean True is mandatory; annex is always false."""
    require(type(synthetic_fixture) is bool and synthetic_fixture is True,
            "synthetic_fixture must be exact bool True")
    exact_keys(payload, ("schema", "calls", "claim_boundary"), "fixture payload")
    require(payload["schema"] == "m1290_projected_fixture_v1" and
            payload["claim_boundary"] == {"synthetic_fixture": True,
                "analytical_cycle_annex": False}, "fixture claim drift")
    rows = payload["calls"]
    require(type(rows) is list, "fixture calls drift")
    return fit_projected(rows, production_verified=False, synthetic_fixture=True)


def calibrate_production() -> Dict[str, Any]:
    """Zero-argument production API; all authority is opened and verified here."""
    rows, _authority = verify_production_authorities(PRODUCTION_RESULT, PRODUCTION_HAMMER)
    return fit_projected(rows, production_verified=True, synthetic_fixture=False)


if __name__ == "__main__":
    raise SystemExit("M1290 is an import-only source; production has no CLI entry point")

#!/usr/bin/env python3
"""Read-only M1008 hammer of the published M998 bounded-prefix result."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HW / "system_simulator/scripts/execute_m994_m982_decoder_canonical_attempt_source_r1.py"
RUNNER = HW / "system_simulator/scripts/run_m998_m994_decoder_canonical_attempt_one_shot.sh"
RESULT = HW / "results/m998_m994_decoder_d2d3_10k_canonical_attempt_r1_20260829"
ATTEMPT = HW / "results/.m998_m994_decoder_d2d3_10k_canonical_attempt_consumed"
RELEASE_SHA = "7140608515b165db358c1ccee23c3a23712aff5abd0172b3793993c89bc6fc03"
M995_ID = (
    "cea74195cdcef8532e41e3dd6810bd5dbfc0cc225174d9a707383b3bd092f4b8",
    "8b3745a9c449438c6f0618e4514a28c39f05167f024e2cd860118b58724080ca",
    "9a6ea0f3fd321b6c23eb34246f75a6b4737607dc2bb2dc18bcf229981ba5b9c6",
)
M997_ID = (
    "7329320279faf673f5181c08f7869b30c9c680631e35ddbbce6f38d0fb8a91bc",
    "93e11e9001b1b261780d4eeac80ae14a462d5725fe97712afb025304cebfacbc",
    "645d8670e9071c9c9947a52eafea0971db1d3a9e7ea7c2657a856e46969c3707",
)
REQUIRED_EXACT_FIELDS = {
    "total_cycles", "expanded_request_count", "compressed_transaction_count",
    "scheduled_requests", "compressed_schedule", "transaction_address_sha256",
    "commit_sequence_sha256", "population_ids", "configs", "cycle_classes",
    "same_cycle_response_slot_reuse", "terminal_readiness",
    "terminal_readiness_sha256", "port_calendars",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def load_driver():
    spec = importlib.util.spec_from_file_location("m1008_readonly_m994", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load M994 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def payload_set(base, directory):
    return {item.relative_to(directory).as_posix()
            for item in base.payload_files(directory)}


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def validate_row(layer, payload, expected_geometry):
    require(payload.get("schema") == "m981_decoder_10k_row_v1" and
            payload.get("status") == "PASS_M981_ROW_EXACT__FRESH_HAMMER_REQUIRED",
            layer + " wrapper status drift")
    row = payload.get("row", {})
    identity = row.get("row_identity", {})
    require(identity.get("layer") == layer and identity.get("config") == "A1_OSG" and
            identity.get("sample_id") == 0 and identity.get("timestep") == 0 and
            identity.get("numerical_route") == "EXACT_BINARY_SUPPORT" and
            identity.get("decoder_complete") is False and
            identity.get("headline_eligible") is False,
            layer + " identity drift")
    require(row.get("prefix") == 10000 and
            row.get("status") == "PASS_M946_BOUNDED_PREFIX_EXACT_PREFLIGHT__NO_FULL_ROW",
            layer + " prefix status drift")
    exact = row.get("exact_miter", {})
    require(exact.get("status") == "PASS_M768_M861_M890_M896_EXACT_MITER" and
            exact.get("old_reference_status") ==
            "PASS_M768_M861_M890_M896_EXACT_MITER" and
            exact.get("expanded_request_count") == 10000,
            layer + " exact miter drift")
    require(set(exact.get("exact_fields", [])) == REQUIRED_EXACT_FIELDS,
            layer + " exact-field set drift")
    classes = exact.get("six_cycle_classes", {})
    total = sum(int(value) for value in classes.values())
    require(total == exact.get("total_cycles_diagnostic_only") and total > 0,
            layer + " cycle-class conservation drift")
    require(exact.get("compressed_transaction_count", 0) > 0 and
            exact["compressed_transaction_count"] <= 10000,
            layer + " transaction count invalid")
    for key in ("transaction_address_sha256", "compressed_transaction_order_sha256",
                "cycle_identity_sha256", "terminal_readiness_sha256",
                "commit_sequence_sha256"):
        require(valid_sha(exact.get(key)), layer + " invalid " + key)
    runs = exact.get("event_run_counts", {})
    require(all(int(runs.get(key, 0)) > 0 for key in
                ("active_service", "dependency_completion", "memory",
                 "psum_bank", "weight_bank")), layer + " event-run coverage drift")

    summary = payload.get("summary", {})
    require(summary.get("layer") == layer and
            summary.get("prefix_requests") == 10000 and
            summary.get("source_bytes") == expected_geometry["source_bytes"] and
            summary.get("source_fetch_requests") ==
            expected_geometry["source_fetch_requests"] and
            summary.get("requests_beyond_first_source_fetch") ==
            10000 - expected_geometry["source_fetch_requests"] and
            summary.get("observed_compressed_transaction_count") ==
            exact["compressed_transaction_count"] and
            summary.get("observed_commit_requests_in_prefix") ==
            exact["commit_requests_in_prefix"], layer + " summary drift")
    shape = identity.get("input_shape")
    require(isinstance(shape, list) and len(shape) == 5, layer + " shape drift")
    derived_bytes = math.ceil(int(shape[2]) * int(shape[3]) * int(shape[4]) / 8)
    require(derived_bytes == expected_geometry["source_bytes"] and
            math.ceil(derived_bytes / 192) == expected_geometry["source_fetch_requests"],
            layer + " source geometry recompute drift")
    boundary = row.get("claim_boundary", {})
    require(boundary.get("bounded_prefix_only") is True and
            boundary.get("prefix_cycle_diagnostic_only") is True and
            all(boundary.get(key) is False for key in
                ("decoder_complete", "full_row_authorized", "paper_citable",
                 "production_row", "system_speedup", "table_a_row")),
            layer + " claim boundary expanded")
    require(exact.get("commit_requests_in_prefix") == 0,
            layer + " unexpected complete-row commit in 10K prefix")
    return {
        "layer": layer,
        "numeric_prefix_exact_pass": True,
        "cycle_prefix_exact_pass": True,
        "expanded_requests": 10000,
        "compressed_transactions": exact["compressed_transaction_count"],
        "total_cycles_diagnostic_only": total,
        "commit_requests_in_prefix": 0,
        "source_bytes": expected_geometry["source_bytes"],
        "source_fetch_requests": expected_geometry["source_fetch_requests"],
        "cycle_complete_row_pass": False,
    }


def main():
    module = load_driver()
    authority = module.validate_authority(RUNNER, RELEASE_SHA, M995_ID, M997_ID)
    attempt_seal = module.B.verify_atomic_seal(ATTEMPT)
    attempt = strict_json(ATTEMPT / "attempt.json")
    require(payload_set(module.B, ATTEMPT) == {"attempt.json"},
            "attempt exact payload set drift")
    require(attempt.get("status") ==
            "CONSUMED_AT_CANONICAL_MKDIR_BEFORE_D2_MODEL_CALL" and
            attempt.get("release_sha256") == RELEASE_SHA and
            attempt.get("release_hammer_review_sha256") == M997_ID[0] and
            attempt.get("max_attempts") == 1 and attempt.get("retry") is False and
            attempt.get("d2_or_d3_100k_authorized") is False and
            attempt.get("full_row_authorized") is False,
            "attempt authority drift")
    module.validate_attempt({"release_sha256": RELEASE_SHA,
                             "release_hammer_review_sha256": M997_ID[0]}, ATTEMPT)

    root_seal = module.B.verify_atomic_seal(RESULT)
    expected_root = {"WORK_STARTED.txt", "RUN_COMPLETE.txt", "result.json"}
    for layer in ("D2", "D3"):
        expected_root.update({
            layer + "/ROW_STARTED.json", layer + "/ROW_COMPLETE.txt",
            layer + "/row.json", layer + "/row.log",
            layer + "/.m981_atomic_seal/SHA256SUMS",
            layer + "/.m981_atomic_seal/SHA256SUMS.seal.sha256",
        })
    require(payload_set(module.B, RESULT) == expected_root,
            "result root exact payload set drift")
    require((RESULT / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "M998_COMPLETE__HAMMER_REQUIRED\n", "completion token drift")
    result = strict_json(RESULT / "result.json")
    require(result.get("schema") ==
            "m998_decoder_d2d3_10k_canonical_attempt_result_v1" and
            result.get("status") ==
            "PASS_M998_D2_THEN_D3_10K__RESULT_HAMMER_REQUIRED" and
            result.get("release_sha256") == RELEASE_SHA,
            "M998 result identity drift")
    boundary = result.get("claim_boundary", {})
    require(all(boundary.get(key) is False for key in
                ("decoder_complete", "paper_citable", "system_speedup", "table_a_row")),
            "M998 result claim expansion")
    rows = result.get("rows", [])
    require(len(rows) == 2 and [row.get("summary", {}).get("layer") for row in rows] ==
            ["D2", "D3"], "D2-D3 result order drift")

    geometry = {"D2": {"source_bytes": 231600, "source_fetch_requests": 1207},
                "D3": {"source_bytes": 465600, "source_fetch_requests": 2425}}
    row_results = []
    row_seals = {}
    for index, layer in enumerate(("D2", "D3")):
        directory = RESULT / layer
        require(payload_set(module.B, directory) ==
                {"ROW_STARTED.json", "ROW_COMPLETE.txt", "row.json", "row.log"},
                layer + " exact payload set drift")
        seal = module.B.verify_atomic_seal(directory)
        require(result.get("row_seals", {}).get(layer) == seal,
                layer + " embedded seal drift")
        payload = strict_json(directory / "row.json")
        require(payload == rows[index], layer + " embedded row drift")
        row_results.append(validate_row(layer, payload, geometry[layer]))
        row_seals[layer] = seal

    require(not any(RESULT.parent.glob(RESULT.name + ".work.*")) and
            not any(RESULT.parent.glob(module.FAILURE_PREFIX + "*")),
            "post-publication work/failure namespace drift")
    return {
        "schema": "m1008_m998_independent_result_hammer_v1",
        "status": "PASS_M1008_M998_D2D3_10K_RESULT_HAMMER",
        "verdict": "ADMIT_D2_D3_BOUNDED_PREFIX_NUMERIC_AND_CYCLE_DIAGNOSTIC_ONLY",
        "authority": authority,
        "attempt_seal": attempt_seal,
        "root_seal": root_seal,
        "row_seals": row_seals,
        "rows": row_results,
        "numeric_prefix_exact_pass": True,
        "cycle_prefix_exact_pass": True,
        "cycle_complete_row_pass": False,
        "d1_common_charge": True,
        "decoder_complete": False,
        "system_speedup": False,
        "paper_citable": False,
        "rerun_performed": False,
        "docs359_sha256": sha(HW / "docs/359_DATE终局冻结_20260813.md"),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))

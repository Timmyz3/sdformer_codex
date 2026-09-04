#!/usr/bin/env python3
"""Read-only independent hammer for the sealed M1656 decoder prefix result.

This review never invokes the decoder runner and never opens the decoder
payload.  It verifies the sealed receipt, exact authority/source lineage,
fixed D0/call0 population, three configuration-bound session receipts,
request/byte ledgers, address and commit digests, and RSS gates.  Reported
ratios are independently recomputed from integer ledgers.  They remain a
42-destination prefix diagnostic: no L3, full-decoder, system, or paper claim.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import copy
from decimal import Decimal, getcontext
import hashlib
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / (
    "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.failed_no_retry")
SOURCE = HW / (
    "system_simulator/scripts/build_m1656_decoder_actual_prefix_"
    "authorization_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1656_decoder_actual_prefix_authorization_successor_"
    "source_contract_r1_20260901.json")
M1645 = HW / (
    "system_simulator/scripts/build_m1645_decoder_compact_actual_prefix_"
    "runner_source.py")
M1638 = HW / (
    "system_simulator/scripts/build_m1638_decoder_compact_l2_session_"
    "configuration_bound_successor_source.py")
M1657 = HW / (
    "reviews/m1657_m1656_decoder_actual_prefix_authorization_successor_"
    "source_independent_review_r1_20260901")
RELEASE = HW / (
    "contracts/m1658_m1657_m1656_decoder_actual_prefix_one_shot_"
    "release_r1_20260901.json")
M1521_MANIFEST = HW / (
    "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831/"
    "manifest.json")
M1527 = HW / (
    "reviews/m1527_m1521_ep34_decoder_positive_plane_actual_result_"
    "hammer_r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
KINDS = ("commit", "compute", "external_read", "psum_read",
         "psum_write", "weight_read", "weight_write")
HEX = frozenset("0123456789abcdef")
COMMIT_SHA = "b96c56fde350a6e8573a923806db1bff7c5c9df97c91d87c0228bdde1aa244e9"
FINAL_COMMIT_SHA = "43638a7021ba36d235ed0a6e5e3609471f2d976cbc8f3eec2101dc7f6050db46"
CHECKPOINT_SHA = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
RESOURCE_SHA = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
PAYLOAD_SHA = "37208563da5f5b218f3aff5b292f05e10a5db16b078672762b2cb9ed60678a1c"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

EXPECTED = {
    "source": "5e1930598b1f107f231b280de0a9dc73d4589171d790f3732ace218ff8c91429",
    "source_contract": "f0e679e976a339889fef7fbf06f29f3beb0bf285e2b6465309da313acb843590",
    "m1645": "0869bed30edbae34ed4d58a0959fa7f70962c3b78b383c80bbd96e4782e7d833",
    "m1638": "1b3961b0d0682980a035f5ad9ba880eb44929e56116f23f2e68cbb9e0a3fdecd",
    "m1657_review": "11e3652dda8a0cddbaacb6938108e953f8abc21bad836243396bf32208c430e7",
    "m1657_manifest": "9b36183daafa1247db5c138657302023c5b53a2aa5a7e8017481172c8ce6c500",
    "m1657_outer": "c9ee02e3ad5a41f5902f832e62399f0ef5a1528d84c755cddd79614ceb4421b5",
    "release": "6ece7ce619150d466f0b4ee85f3cdb01f8c0d65166c60408f564ddc2430e031b",
    "release_sidecar": "7d32d9c103179568645c19929432e22e67c2ab4ed9cc785c7345f85dbafdc42f",
    "release_outer": "420a4b5f59498f902782a9ea0389a5daf9b5c3def3deb70dc9b69fe851f5d9f0",
    "m1521_manifest": "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304",
    "m1527_review": "366068b725a16c42fc69adc29c463ce909b0f528f6a31c36eb25e6914366c714",
    "m1527_manifest": "2cf68cdca714a68d25ac6cbd9ea2f9a9f9cfa323bd3d39a401c7f6636dec8a94",
    "m1527_outer": "37841dfbd4f6d83d4004efdfa5e80e011396c9581fc8e2c0985ffec85274bb22",
    "attempt": "d46017eb67b9a304b2aa974ac61f0cb81c78fc502cda4614dd0ac7420ffd3c98",
    "result_json": "badb856d74beb9a4a618a8e2cfa53f17f7fc08b42d73c98ec026258b2dfe0eb5",
    "result_manifest": "53a6ed1f9cda116e56182cfa7a110312a95cabbcc7782df0015c83f0e9313477",
    "result_outer": "138093e6f19d5354a2cbbc28343a5a72121d355958a010c76aafc3b2c1f212a8",
}

EXPECTED_ROWS = {
    "DENSE_TYPED_K8": {
        "cycles": 1034451, "requests": 299688,
        "kinds": {"commit": 168, "compute": 49920,
            "external_read": 74880, "psum_read": 49920,
            "psum_write": 49920, "weight_read": 49920,
            "weight_write": 24960},
        "bytes": {"commit": 64512, "compute": 14376960,
            "external_read": 40734720, "psum_read": 14376960,
            "psum_write": 14376960, "weight_read": 38338560,
            "weight_write": 38338560},
        "address": "141c6d46fdf54d91e75d8abc6c9dc4d76ce0aeae1d37f36abb5343173247af09",
        "session": "305e65305b7c9f2968208f698e1cc4a0b64fe9d4fa09aa949529e63c324258c8",
        "state": "91d53175ecf5333a3d58ce07674e95cd72eb56bd71d17a2cae63985aa27683e7",
        "dense_coverage": True,
    },
    "BIT_EQUAL_SERVICE_K1X8": {
        "cycles": 519007, "requests": 172400,
        "kinds": {"commit": 168, "compute": 7336,
            "external_read": 75112, "psum_read": 7336,
            "psum_write": 7336, "weight_read": 45220,
            "weight_write": 29892},
        "bytes": {"commit": 64512, "compute": 2112768,
            "external_read": 46637632, "psum_read": 2112768,
            "psum_write": 2112768, "weight_read": 4341120,
            "weight_write": 45914112},
        "address": "d44a2520af1b69615905e162c9d2597824cd43c68496083b3f1081024319fa97",
        "session": "af73f48d393d921ced928a6c2d58913336bb8fc001207e12784ee95a6c4c0669",
        "state": "6cd6f140a3f495813a4623d2f1385d97f9a8f5239b964638975d01625b3d65c3",
        "dense_coverage": False,
    },
    "BIT_TYPED_K8": {
        "cycles": 481123, "requests": 96632,
        "kinds": {"commit": 168, "compute": 7336,
            "external_read": 37228, "psum_read": 7336,
            "psum_write": 7336, "weight_read": 7336,
            "weight_write": 29892},
        "bytes": {"commit": 64512, "compute": 2112768,
            "external_read": 46212368, "psum_read": 2112768,
            "psum_write": 2112768, "weight_read": 4341120,
            "weight_write": 45914112},
        "address": "b3d395291a84040175f05eefb4fbf53af17e18874f14851ab34e4a144ef3ba4a",
        "session": "27a09f9f3a74005bed84bf99b8f1fa1fd4cdc077a78a3dbc72fa258cd001ac4e",
        "state": "5e88641c683222b75ffa54579752acc8233636dbaa845f010e8b4772de9e0b72",
        "dense_coverage": False,
    },
}


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json_text(text):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(text, object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            HammerError("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def strict_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"))


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            label + " key topology drift")


def hex64(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in HEX for character in value),
            label + " must be lowercase hex64")


def verify_file_double_seal(path, expected_file, expected_sidecar,
                            expected_outer, label):
    regular_exact(path, expected_file, label)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular_exact(sidecar, expected_sidecar, label + " sidecar")
    regular_exact(outer, expected_outer, label + " outer")
    require(sidecar.read_text(encoding="ascii") ==
            expected_file + "  " + path.name + "\n",
            label + " sidecar content drift")
    require(outer.read_text(encoding="ascii") ==
            expected_sidecar + "  " + sidecar.name + "\n",
            label + " outer content drift")


def verify_review_tree(root, review_sha, manifest_sha, outer_sha, label):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), label + " root drift")
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(review, review_sha, label + " review")
    regular_exact(manifest, manifest_sha, label + " manifest")
    regular_exact(outer, outer_sha, label + " outer")
    require(outer.read_text(encoding="ascii") ==
            manifest_sha + "  SHA256SUMS\n", label + " outer content")
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, label + " malformed manifest")
        digest, name = fields
        relative = Path(name)
        hex64(digest, label + " manifest digest")
        require(name not in rows and not relative.is_absolute() and
                ".." not in relative.parts, label + " unsafe manifest")
        rows[name] = digest
        regular_exact(root / relative, digest, label + " member " + name)
    require(rows.get("review.json") == review_sha,
            label + " review absent from seal")
    return strict_json(review)


def verify_result_seal():
    require(RESULT.is_dir() and not RESULT.is_symlink(), "result root drift")
    members = sorted(path.name for path in RESULT.iterdir())
    require(members == ["SHA256SUMS", "SHA256SUMS.seal.sha256",
                        "result.json"], "result flat topology drift")
    for path in RESULT.iterdir():
        require(path.is_file() and not path.is_symlink(),
                "result member type drift")
    regular_exact(RESULT / "result.json", EXPECTED["result_json"],
                  "result.json")
    regular_exact(RESULT / "SHA256SUMS", EXPECTED["result_manifest"],
                  "result manifest")
    regular_exact(RESULT / "SHA256SUMS.seal.sha256",
                  EXPECTED["result_outer"], "result outer")
    require((RESULT / "SHA256SUMS").read_text(encoding="ascii") ==
            EXPECTED["result_json"] + "  result.json\n",
            "result manifest content drift")
    require((RESULT / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii") ==
            EXPECTED["result_manifest"] + "  SHA256SUMS\n",
            "result outer content drift")


def validate_fixed_population(row):
    exact_keys(row, ("call_ordinal", "configuration_order",
        "decoder_stage", "destinations", "module_ordinal",
        "output_blocks", "timestep"), "fixed population")
    require(row == {"call_ordinal": 0,
        "configuration_order": list(CONFIGS), "decoder_stage": "D0",
        "destinations": list(range(42)), "module_ordinal": 0,
        "output_blocks": list(range(4)), "timestep": 0},
        "fixed 42-destination population drift")


def validate_rss(row):
    exact_keys(row, ("absolute_limit_kib", "baseline_current_rss_kib",
        "baseline_hwm_rss_kib", "gate_calls", "increment_limit_kib",
        "max_current_rss_kib", "max_hwm_rss_kib"), "RSS")
    for field in row:
        require(type(row[field]) is int, "RSS field type drift")
    require(row["absolute_limit_kib"] == 2097152 and
            row["increment_limit_kib"] == 524288 and
            row["gate_calls"] == 129 and
            0 <= row["baseline_current_rss_kib"] <=
                row["max_current_rss_kib"] < row["absolute_limit_kib"] and
            0 <= row["baseline_hwm_rss_kib"] <=
                row["max_hwm_rss_kib"] < row["absolute_limit_kib"] and
            row["max_hwm_rss_kib"] >= row["max_current_rss_kib"] and
            row["max_hwm_rss_kib"] - row["baseline_current_rss_kib"] <
                row["increment_limit_kib"], "RSS gate drift")


def validate_metric(row, configuration):
    exact_keys(row, ("byte_counts", "configuration",
        "independent_hammer_pending", "kind_counts", "paper_result",
        "packed_commit_sequence_sha256",
        "packed_transaction_address_sha256", "request_count",
        "total_cycles"), "metric")
    expected = EXPECTED_ROWS[configuration]
    require(row["configuration"] == configuration and
            row["total_cycles"] == expected["cycles"] and
            row["request_count"] == expected["requests"] and
            row["kind_counts"] == expected["kinds"] and
            row["byte_counts"] == expected["bytes"] and
            row["packed_transaction_address_sha256"] ==
                expected["address"] and
            row["packed_commit_sequence_sha256"] == COMMIT_SHA and
            row["independent_hammer_pending"] is True and
            row["paper_result"] is False, "metric exact ledger drift")
    require(set(row["kind_counts"]) == set(KINDS) and
            set(row["byte_counts"]) == set(KINDS) and
            sum(row["kind_counts"].values()) == row["request_count"] and
            all(type(value) is int and value >= 0
                for value in row["byte_counts"].values()),
            "metric conservation/type drift")
    hex64(row["packed_transaction_address_sha256"], "address digest")
    hex64(row["packed_commit_sequence_sha256"], "commit digest")


def validate_session(row, configuration):
    exact_keys(row, ("commits", "configuration", "dense_cache_covered",
        "dense_psum_1rw_covered", "destinations", "final_commit_digest",
        "final_state_sha256", "requests", "resource_manifest_sha256",
        "schema", "session_identity"), "session")
    expected = EXPECTED_ROWS[configuration]
    dense = expected["dense_coverage"]
    require(row["configuration"] == configuration and
            row["schema"] ==
                "m1638_decoder_compact_l2_session_configuration_bound_successor_source_r1_v1" and
            row["resource_manifest_sha256"] == RESOURCE_SHA and
            row["destinations"] == 42 and row["commits"] == 168 and
            row["requests"] == expected["requests"] and
            row["dense_cache_covered"] is dense and
            row["dense_psum_1rw_covered"] is dense and
            row["final_commit_digest"] == FINAL_COMMIT_SHA and
            row["final_state_sha256"] == expected["state"] and
            row["session_identity"] == expected["session"],
            "session identity/miter receipt drift")
    for field in ("final_commit_digest", "final_state_sha256",
                  "session_identity"):
        hex64(row[field], "session " + field)


def validate_result(row):
    exact_keys(row, ("bytes_pending_hammer", "checkpoint",
        "checkpoint_sha256", "cycles_pending_hammer", "fixed_population",
        "full_decoder", "independent_hammer_pending", "l3", "metrics",
        "paper_result", "payload_fd_sha256", "payload_fd_size",
        "product_capture", "production", "release_sha256",
        "resource_manifest_sha256", "rss", "schema", "sessions",
        "source_sha256", "status"), "result")
    require(row["schema"] == "m1656_decoder_actual_prefix_result_r1_v1" and
            row["status"] ==
                "PREFIX_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED" and
            row["source_sha256"] == EXPECTED["source"] and
            row["release_sha256"] == EXPECTED["release"] and
            row["checkpoint"] == "motion_ep34_live93" and
            row["checkpoint_sha256"] == CHECKPOINT_SHA and
            row["resource_manifest_sha256"] == RESOURCE_SHA and
            row["payload_fd_sha256"] == PAYLOAD_SHA and
            row["payload_fd_size"] == 576000 and
            row["independent_hammer_pending"] is True and
            row["cycles_pending_hammer"] is True and
            row["bytes_pending_hammer"] is True and
            row["product_capture"] is False and row["l3"] is False and
            row["full_decoder"] is False and row["production"] is False and
            row["paper_result"] is False, "result identity/claim drift")
    validate_fixed_population(row["fixed_population"])
    require(type(row["metrics"]) is list and len(row["metrics"]) == 3 and
            type(row["sessions"]) is list and len(row["sessions"]) == 3,
            "three-row result topology drift")
    require([item.get("configuration") for item in row["metrics"]] ==
                list(CONFIGS) and
            [item.get("configuration") for item in row["sessions"]] ==
                list(CONFIGS), "configuration order drift")
    for metric, session, configuration in zip(
            row["metrics"], row["sessions"], CONFIGS):
        validate_metric(metric, configuration)
        validate_session(session, configuration)
        require(metric["request_count"] == session["requests"],
                "metric/session request cross-binding drift")
    require(len(set(item["session_identity"] for item in row["sessions"])) == 3
            and len(set(item["final_state_sha256"]
                        for item in row["sessions"])) == 3 and
            len(set(item["packed_transaction_address_sha256"]
                    for item in row["metrics"])) == 3 and
            len(set(item["packed_commit_sequence_sha256"]
                    for item in row["metrics"])) == 1,
            "session/address/commit cross-configuration invariant drift")
    validate_rss(row["rss"])


def ratio_payload(row):
    getcontext().prec = 50
    metrics = dict((item["configuration"], item) for item in row["metrics"])
    totals = dict((name, sum(item["byte_counts"].values()))
                  for name, item in metrics.items())

    def compare(left, right):
        left_cycles = metrics[left]["total_cycles"]
        right_cycles = metrics[right]["total_cycles"]
        left_bytes, right_bytes = totals[left], totals[right]
        return {"cycle_numerator": left_cycles,
            "cycle_denominator": right_cycles,
            "cycle_ratio": str(Decimal(left_cycles) /
                               Decimal(right_cycles)),
            "time_reduction": str(Decimal(1) -
                                  Decimal(right_cycles) /
                                  Decimal(left_cycles)),
            "byte_numerator": left_bytes,
            "byte_denominator": right_bytes,
            "byte_ratio": str(Decimal(left_bytes) / Decimal(right_bytes)),
            "byte_reduction": str(Decimal(1) -
                                  Decimal(right_bytes) /
                                  Decimal(left_bytes))}
    return {"modeled_transaction_bytes": totals,
        "all_three_modeled_transaction_bytes": sum(totals.values()),
        "all_three_cycles": sum(item["total_cycles"]
                                for item in metrics.values()),
        "dense_vs_bit_equal": compare(CONFIGS[0], CONFIGS[1]),
        "dense_vs_bit_k8": compare(CONFIGS[0], CONFIGS[2]),
        "bit_equal_vs_bit_k8": compare(CONFIGS[1], CONFIGS[2])}


def verify_source_miter_path():
    source_text = SOURCE.read_text(encoding="utf-8")
    predecessor = M1645.read_text(encoding="utf-8")
    require(source_text.index(
        "authority = verify_pre_payload_authorities(require_future=True)") <
        source_text.index("consume_attempt(authority[\"release_sha256\"])") <
        source_text.index("path, shape, payload_sha = P._selected_payload()"),
        "authorization/attempt/payload order drift")
    for needle in ("receipt, metric = P._schedule_prefix(configuration, plane, rss)",
                   "P.L2.validate_bundle(receipts)"):
        require(needle in source_text, "M1656 exact execution seam drift")
    for needle in ("self.miter.accept_request_pair(receipt, compact_receipt)",
                   "self.miter.accept_destination_pair(reference_state, compact_state)",
                   "actual-prefix packed address/commit miter failed"):
        require(needle in predecessor, "M1645 miter seam drift")


def mutation_hammer(original):
    attacks = []

    def reject(name, mutate):
        candidate = copy.deepcopy(original)
        mutate(candidate)
        try:
            validate_result(candidate)
        except (HammerError, KeyError, TypeError, ValueError):
            attacks.append(name)
            return
        raise HammerError("mutation accepted: " + name)

    reject("root_status", lambda row: row.update(status="PASS"))
    reject("root_l3", lambda row: row.update(l3=True))
    reject("root_paper", lambda row: row.update(paper_result=True))
    reject("fixed_destination", lambda row:
           row["fixed_population"]["destinations"].append(42))
    reject("configuration_order", lambda row:
           row["metrics"].reverse())
    reject("metric_cycle", lambda row:
           row["metrics"][2].update(total_cycles=481122))
    reject("metric_byte", lambda row:
           row["metrics"][2]["byte_counts"].update(external_read=1))
    reject("metric_request", lambda row:
           row["metrics"][2].update(request_count=96631))
    reject("address_digest", lambda row:
           row["metrics"][1].update(packed_transaction_address_sha256=
                                    "0" * 64))
    reject("commit_digest", lambda row:
           row["metrics"][0].update(packed_commit_sequence_sha256=
                                    "0" * 64))
    reject("session_alias", lambda row:
           row["sessions"][2].update(session_identity=
                                     row["sessions"][1]["session_identity"]))
    reject("session_state", lambda row:
           row["sessions"][0].update(final_state_sha256="0" * 64))
    reject("session_requests", lambda row:
           row["sessions"][0].update(requests=1))
    reject("dense_coverage", lambda row:
           row["sessions"][1].update(dense_cache_covered=True))
    reject("rss_absolute", lambda row:
           row["rss"].update(absolute_limit_kib=1))
    reject("rss_gate_calls", lambda row: row["rss"].update(gate_calls=0))
    reject("resource", lambda row: row.update(resource_manifest_sha256=
                                               "0" * 64))
    reject("hidden_alias", lambda row: row.update(system_speedup=2.15))
    try:
        strict_json_text('{"a":1,"a":2}')
    except HammerError:
        attacks.append("duplicate_json_key")
    try:
        strict_json_text('{"a":NaN}')
    except HammerError:
        attacks.append("nonfinite_json")
    require(len(attacks) == 20, "mutation rejection count drift")
    return attacks


def main():
    verify_result_seal()
    regular_exact(SOURCE, EXPECTED["source"], "M1656 source")
    regular_exact(SOURCE_CONTRACT, EXPECTED["source_contract"],
                  "M1656 source contract")
    regular_exact(M1645, EXPECTED["m1645"], "M1645 source")
    regular_exact(M1638, EXPECTED["m1638"], "M1638 source")
    verify_file_double_seal(RELEASE, EXPECTED["release"],
        EXPECTED["release_sidecar"], EXPECTED["release_outer"],
        "M1658 release")
    m1657 = verify_review_tree(M1657, EXPECTED["m1657_review"],
        EXPECTED["m1657_manifest"], EXPECTED["m1657_outer"], "M1657")
    require(m1657.get("status") ==
        "PASS_M1657_M1656_DECODER_ACTUAL_PREFIX_AUTHORIZATION_SUCCESSOR_SOURCE__AUTHORIZE_RELEASE_AUTHORING__NO_EXECUTION",
        "M1657 status drift")
    m1527 = verify_review_tree(M1527, EXPECTED["m1527_review"],
        EXPECTED["m1527_manifest"], EXPECTED["m1527_outer"], "M1527")
    require(m1527.get("status") ==
        "PASS_M1527_M1521_EP34_DECODER_POSITIVE_PLANE_ACTUAL_RESULT__ADDRESS_REPLAY_SUCCESSOR_ALLOWED",
        "M1527 status drift")
    regular_exact(M1521_MANIFEST, EXPECTED["m1521_manifest"],
                  "M1521 manifest")
    manifest = strict_json(M1521_MANIFEST)
    first = manifest.get("records", [None])[0]
    require(type(first) is dict and first.get("global_call_ordinal") == 0 and
            first.get("module_ordinal") == 0 and
            first.get("positive_output_sha256") == PAYLOAD_SHA and
            first.get("plane_bytes") == 576000 and
            first.get("shape") == [10, 1, 1536, 15, 20],
            "M1521 fixed payload manifest row drift")
    regular_exact(DOCS359, DOCS359_SHA, "protected docs359")
    regular_exact(ATTEMPT, EXPECTED["attempt"], "attempt marker")
    require(stat.S_IMODE(ATTEMPT.lstat().st_mode) == 0o400 and
            not ATTEMPT.is_symlink() and
            ATTEMPT.read_text(encoding="ascii") ==
            "M1656_ATTEMPT_CONSUMED__D0_CALL0_DEST0_41__THREE_CONFIGURATIONS__AUTOMATIC_RETRY_FALSE\n"
            "release_sha256=" + EXPECTED["release"] + "\n"
            "source_sha256=" + EXPECTED["source"] + "\n",
            "attempt marker content/mode drift")
    require(not os.path.lexists(str(WORK)) and
            not os.path.lexists(str(FAILURE)),
            "work/failure namespace must be absent after canonical publish")
    verify_source_miter_path()
    result = strict_json(RESULT / "result.json")
    validate_result(result)
    ratios = ratio_payload(result)
    attacks = mutation_hammer(result)
    output = {"schema":
        "m1666_m1656_decoder_actual_prefix_result_independent_hammer_r1_v1",
        "status":
        "PASS_M1666_M1656_DECODER_ACTUAL_PREFIX_RESULT__PREFIX_ONLY_DIAGNOSTIC__NO_L3_OR_PAPER_CLAIM",
        "sealed_result": True, "fixed_destinations": 42,
        "configurations": list(CONFIGS), "distinct_sessions": 3,
        "request_miter_bound": True, "destination_miter_bound": True,
        "address_digests_exact": True,
        "common_packed_commit_sequence_sha256": COMMIT_SHA,
        "common_final_commit_digest": FINAL_COMMIT_SHA,
        "rss_gate_exact": True, "ratio_of_sums": ratios,
        "mutations_rejected": attacks, "mutation_count": len(attacks),
        "payload_opened": False, "runner_executed": False,
        "l3": False, "full_decoder": False, "system_speedup": False,
        "paper_result": False}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

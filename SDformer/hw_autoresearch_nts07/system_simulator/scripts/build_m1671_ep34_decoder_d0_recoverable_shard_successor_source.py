#!/usr/bin/env python3
"""Source-only successor for recoverable full-D0 cycle replay.

The independently hammered M1656 result proves one actual D0/call0/t0
prefix.  M1671 generalizes exactly that reference/compact scheduling pair to
the complete frozen D0 population: 30 calls, ten timesteps, 1,200
destinations and four 96-lane output blocks.  Execution is partitioned into
fixed 42-destination shards (the final shard of a timestep has 24) so an
interruption never invalidates already sealed shards.

Each future shard must run all three configurations in this order:
``DENSE_TYPED_K8``, ``BIT_EQUAL_SERVICE_K1X8``, ``BIT_TYPED_K8``.  Every
request is compared between the frozen M1539 reference scheduler and M1610
compact engine before acceptance; cumulative state is compared after every
destination.  Only a complete 8,700-shard set may be reduced, using integer
ratio-of-sums.  The shard reset is an explicit simulator boundary, not a
monolithic full-call claim.

This revision cannot open a canonical payload, create an attempt, run a
shard, or reduce performance.  A different-author M1672 review and separate
M1673 execution release are required.  Python syntax is CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/"
    "test_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_contract_r1_20260901.json")
M1645_SOURCE = HERE / (
    "build_m1645_decoder_compact_actual_prefix_runner_source.py")
M1656_SOURCE = HERE / (
    "build_m1656_decoder_actual_prefix_authorization_successor_source.py")
M1656_RESULT = HW / (
    "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901")
M1666_REVIEW = HW / (
    "reviews/m1666_m1656_decoder_actual_prefix_result_independent_"
    "hammer_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_REVIEW = HW / (
    "reviews/m1672_m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1673_m1672_m1671_ep34_decoder_d0_recoverable_shard_"
    "execution_release_r1_20260901.json")

SCHEMA = "m1671_ep34_decoder_d0_recoverable_shard_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__FULL_D0_RECOVERABLE_SHARD_GRID__"
    "DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_PAYLOAD_NO_EXECUTION")
CHECKPOINT = "motion_ep34_live93"
CHECKPOINT_SHA256 = (
    "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
RESOURCE_SHA256 = (
    "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
M1645_SOURCE_SHA256 = (
    "0869bed30edbae34ed4d58a0959fa7f70962c3b78b383c80bbd96e4782e7d833")
M1656_SOURCE_SHA256 = (
    "5e1930598b1f107f231b280de0a9dc73d4589171d790f3732ace218ff8c91429")
M1656_RESULT_SHA256 = (
    "badb856d74beb9a4a618a8e2cfa53f17f7fc08b42d73c98ec026258b2dfe0eb5")
M1656_RESULT_MANIFEST_SHA256 = (
    "53a6ed1f9cda116e56182cfa7a110312a95cabbcc7782df0015c83f0e9313477")
M1656_RESULT_OUTER_FILE_SHA256 = (
    "138093e6f19d5354a2cbbc28343a5a72121d355958a010c76aafc3b2c1f212a8")
M1666_REVIEW_SHA256 = (
    "1acd2380365c1d89750f82cf1623d68ad77147355ebbba7b6d2c83597d6eda29")
M1666_MANIFEST_SHA256 = (
    "2bed52d666d9913562bf4370b33c6a9b6528200cd490c1ac3c3585e229213b65")
M1666_OUTER_FILE_SHA256 = (
    "d7a4edda6946b065948a85e0cf53bf90df4c06cd281f8069309422b6a685230b")

CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
MODULE_ORDINAL = 0
TIMESTEPS = 10
OUTPUT_HEIGHT = 30
OUTPUT_WIDTH = 40
DESTINATIONS = OUTPUT_HEIGHT * OUTPUT_WIDTH
OUTPUT_BLOCKS = 4
DESTINATIONS_PER_SHARD = 42
D0_CALL_ORDINALS = tuple(range(0, 120, 4))
SHARDS_PER_TIMESTEP = int(math.ceil(
    float(DESTINATIONS) / DESTINATIONS_PER_SHARD))
TOTAL_SHARDS = len(D0_CALL_ORDINALS) * TIMESTEPS * SHARDS_PER_TIMESTEP
RSS_ABSOLUTE_LIMIT_KIB = 2 * 1024 * 1024
RSS_INCREMENT_LIMIT_KIB = 512 * 1024


class M1671Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1671Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1671Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1671Error("nonfinite JSON: " + token)))


def verify_flat_tree(root, expected_review, expected_manifest,
                     expected_outer, label):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), label + " tree drift")
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(review, expected_review, label + " review")
    regular_exact(manifest, expected_manifest, label + " manifest")
    regular_exact(outer, expected_outer, label + " outer")
    require(outer.read_text(encoding="ascii") ==
            expected_manifest + "  SHA256SUMS\n",
            label + " outer content drift")
    names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and
                fields[1] not in names and "/" not in fields[1] and
                ".." not in fields[1], label + " manifest drift")
        names.add(fields[1])
        regular_exact(root / fields[1], fields[0],
                      label + " member " + fields[1])
    require("review.json" in names, label + " review is not sealed")
    return strict_json(review)


def verify_result_tree():
    root = M1656_RESULT
    require(root.is_dir() and not root.is_symlink(), "M1656 result tree drift")
    result = root / "result.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(result, M1656_RESULT_SHA256, "M1656 result")
    regular_exact(manifest, M1656_RESULT_MANIFEST_SHA256,
                  "M1656 result manifest")
    regular_exact(outer, M1656_RESULT_OUTER_FILE_SHA256,
                  "M1656 result outer")
    require(manifest.read_text(encoding="ascii") ==
            M1656_RESULT_SHA256 + "  result.json\n" and
            outer.read_text(encoding="ascii") ==
            M1656_RESULT_MANIFEST_SHA256 + "  SHA256SUMS\n",
            "M1656 result seal content drift")
    row = strict_json(result)
    require(row.get("schema") == "m1656_decoder_actual_prefix_result_r1_v1" and
            row.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
            row.get("resource_manifest_sha256") == RESOURCE_SHA256 and
            row.get("fixed_population", {}).get("decoder_stage") == "D0" and
            row.get("fixed_population", {}).get("call_ordinal") == 0 and
            row.get("fixed_population", {}).get("timestep") == 0 and
            row.get("fixed_population", {}).get("destinations") ==
                list(range(42)) and
            row.get("full_decoder") is False and
            row.get("paper_result") is False,
            "M1656 result semantic boundary drift")
    return row


def verify_m1666():
    row = verify_flat_tree(M1666_REVIEW, M1666_REVIEW_SHA256,
        M1666_MANIFEST_SHA256, M1666_OUTER_FILE_SHA256, "M1666")
    require(row.get("status") ==
            "PASS_M1666_M1656_DECODER_ACTUAL_PREFIX_RESULT__PREFIX_ONLY_DIAGNOSTIC__NO_L3_OR_PAPER_CLAIM" and
            row.get("verdict") == "PASS_PREFIX_DIAGNOSTIC_ONLY" and
            row.get("p0_count") == 0 and
            row.get("p1_count") == 0 and
            row.get("authorization", {}).get(
                "prefix_diagnostic_reporting") is True and
            row.get("authorization", {}).get("l3_expansion") is False and
            row.get("authorization", {}).get("paper_result") is False,
            "M1666 review status/authorization drift")
    return row


def load_m1645():
    regular_exact(M1645_SOURCE, M1645_SOURCE_SHA256, "exact M1645 source")
    spec = importlib.util.spec_from_file_location("m1671_exact_m1645",
                                                  str(M1645_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1645")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
            module.RESOURCE_SHA256 == RESOURCE_SHA256,
            "M1645 execution boundary drift")
    return module


P = load_m1645()
R = P.R
C = P.C


def shard_descriptor(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < TOTAL_SHARDS,
            "shard ordinal out of range")
    call_index, inner = divmod(ordinal, TIMESTEPS * SHARDS_PER_TIMESTEP)
    timestep, shard_in_timestep = divmod(inner, SHARDS_PER_TIMESTEP)
    start = shard_in_timestep * DESTINATIONS_PER_SHARD
    stop = min(start + DESTINATIONS_PER_SHARD, DESTINATIONS)
    return {"shard_ordinal": ordinal,
            "call_ordinal": D0_CALL_ORDINALS[call_index],
            "sample_ordinal": call_index, "module_ordinal": MODULE_ORDINAL,
            "timestep": timestep, "destination_start": start,
            "destination_stop_exclusive": stop,
            "destination_count": stop - start,
            "output_blocks": list(range(OUTPUT_BLOCKS)),
            "configuration_order": list(CONFIGS)}


def fixed_grid():
    first = shard_descriptor(0)
    last = shard_descriptor(TOTAL_SHARDS - 1)
    return {"decoder_stage": "D0", "calls": len(D0_CALL_ORDINALS),
            "call_ordinals": list(D0_CALL_ORDINALS),
            "timesteps_per_call": TIMESTEPS,
            "destinations_per_timestep": DESTINATIONS,
            "output_blocks": OUTPUT_BLOCKS,
            "nominal_destinations_per_shard": DESTINATIONS_PER_SHARD,
            "shards_per_timestep": SHARDS_PER_TIMESTEP,
            "total_shards": TOTAL_SHARDS,
            "first_shard": first, "last_shard": last,
            "configuration_order": list(CONFIGS),
            "ordering": "call_index,timestep,destination_start"}


def validate_grid():
    seen = set()
    expected = []
    for call in D0_CALL_ORDINALS:
        for timestep in range(TIMESTEPS):
            covered = []
            for shard_in_timestep in range(SHARDS_PER_TIMESTEP):
                ordinal = ((D0_CALL_ORDINALS.index(call) * TIMESTEPS +
                            timestep) * SHARDS_PER_TIMESTEP +
                           shard_in_timestep)
                row = shard_descriptor(ordinal)
                key = (row["call_ordinal"], row["timestep"],
                       row["destination_start"],
                       row["destination_stop_exclusive"])
                require(key not in seen, "duplicate shard coordinate")
                seen.add(key)
                covered.extend(range(row["destination_start"],
                                     row["destination_stop_exclusive"]))
                expected.append(ordinal)
            require(covered == list(range(DESTINATIONS)),
                    "D0 timestep destination coverage has a gap/overlap")
    require(expected == list(range(TOTAL_SHARDS)) and len(seen) == TOTAL_SHARDS,
            "shard ordinal/grid conservation failed")
    return {"calls": len(D0_CALL_ORDINALS), "timesteps": 300,
            "destinations": len(D0_CALL_ORDINALS) * TIMESTEPS * DESTINATIONS,
            "shards": TOTAL_SHARDS, "gap_count": 0, "overlap_count": 0}


def selected_record(shard):
    manifest = R.strict_json(R.M1521_MANIFEST)
    R.validate_population_manifest(manifest)
    call = shard["call_ordinal"]
    row = manifest["records"][call]
    require(row.get("global_call_ordinal") == call and
            row.get("module_ordinal") == MODULE_ORDINAL and
            tuple(row.get("shape", ())) == tuple(R.INPUT_SHAPES[0]) and
            row.get("positive_output_sha256") and
            row.get("negative_plane_all_zero") is True,
            "D0 shard payload record identity drift")
    return row


def _flag_and_subordinal(suffix):
    return P._flag_and_subordinal(suffix)


def actual_coordinate(configuration, row, request_ordinal, timestep,
                      destination, output_block):
    require(configuration in CONFIGS and row.get("config") == configuration,
            "request configuration drift")
    commit_id = "{}:commit:{}:{}".format(
        configuration, destination, output_block)
    # Future engine uses call/timestep-qualified commit identifiers.  Author
    # tests also accept the unqualified exact-M1656 form.
    if row.get("kind") == "commit":
        require(row.get("id") == commit_id or
                row.get("id", "").endswith(
                    ":commit:{}:{}".format(destination, output_block)),
                "commit identifier drift")
        flag, subordinal, group = C.FLAG_COMMIT, 0, C.U32_SENTINEL
    else:
        prefix = "{}:m0:t{}:d{}:ob{}:g".format(
            configuration, timestep, destination, output_block)
        identifier = row.get("id", "")
        require(identifier.startswith(prefix), "D0 request identifier drift")
        fields = identifier[len(prefix):].split(":", 1)
        require(len(fields) == 2 and fields[0].isdigit(),
                "D0 group identifier drift")
        group = int(fields[0])
        flag, subordinal = _flag_and_subordinal(fields[1])
    return (1, MODULE_ORDINAL, int(timestep), flag, int(destination),
            int(output_block), group, subordinal, int(request_ordinal))


class ShardSession(object):
    """Exact request miter plus a cumulative per-destination state miter."""
    def __init__(self, configuration, shard, rss):
        require(configuration in CONFIGS and configuration != FORBIDDEN_CONFIG,
                "configuration is not admitted")
        self.configuration = configuration
        self.shard = dict(shard)
        self.rss = rss
        self.reference = R.AddressTimedScheduler(configuration)
        self.compact = C.CompactScheduler(configuration)
        self.cache = P.MirroredWeightCache()
        self.tokens = {}
        self.last_psum_write_ready = [0] * OUTPUT_BLOCKS
        self.packed_reference = hashlib.sha256()
        self.packed_commit = hashlib.sha256()
        self.request_digest = hashlib.sha256()
        self.destination_digests = []

    def accept(self, row, destination, output_block):
        ordinal = self.compact.requests
        coordinate = actual_coordinate(self.configuration, row, ordinal,
            self.shard["timestep"], destination, output_block)
        missing = [token for token in row["dependencies"]
                   if token not in self.tokens]
        require(not missing, "shard unresolved dependency")
        dependency = max([self.tokens[token] for token in row["dependencies"]]
                         or [row["earliest_issue_cycle"]])
        port_ready = C.reference_port_ready(row, self.reference)
        reference_receipt = self.reference.schedule_one(row)
        self.compact.begin_addresses()
        for address, bank in zip(row["addresses"], row["banks"]):
            self.compact.push_address(address, bank)
        self.compact.schedule_loaded(
            C.kind_index(row["kind"]), row["width_bytes"],
            row["earliest_issue_cycle"], dependency, *coordinate)
        require((reference_receipt["dependency_ready_cycle"],
                 reference_receipt["issue_cycle"],
                 reference_receipt["return_cycle"], port_ready) ==
                (dependency, self.compact.last_issue,
                 self.compact.last_return, self.compact.last_port_ready),
                "per-request reference/compact cycle miter failed")
        require(self.compact.next_port ==
                    C.compact_next_port_projection(self.reference) and
                self.compact.last_cycle == self.reference.last_cycle and
                self.compact.requests == self.reference.requests,
                "cumulative scheduler miter failed")
        expected_outstanding, expected_counts = \
            C.compact_outstanding_projection(self.reference)
        layout = tuple((bank * 8, 8) for bank in range(8)) + \
            tuple((64 + bank * 8, 8) for bank in range(6)) + \
            ((112, 16), (128, 1))
        for queue, (base, capacity) in enumerate(layout):
            count = self.compact.outstanding_count[queue]
            require(count == expected_counts[queue] and count <= capacity and
                    sorted(self.compact.outstanding[base:base + count]) ==
                        expected_outstanding[base:base + count],
                    "outstanding queue miter failed")
        if row["produces"]:
            require(row["produces"] not in self.tokens,
                    "duplicate compact dependency token")
            self.tokens[row["produces"]] = self.compact.last_return
        if row["kind"] == "psum_write":
            self.last_psum_write_ready[output_block] = max(
                self.last_psum_write_ready[output_block],
                self.compact.last_return)
        for address, bank in zip(row["addresses"], row["banks"]):
            self.packed_reference.update(C.PACKED_ADDRESS.pack(
                coordinate[0], C.config_index(self.configuration),
                C.kind_index(row["kind"]), coordinate[1], coordinate[2],
                coordinate[3], coordinate[4], coordinate[5], coordinate[6],
                coordinate[7], coordinate[8], int(address), int(bank),
                int(row["width_bytes"])))
        if row["kind"] == "commit":
            for address in row["addresses"]:
                self.packed_commit.update(C.PACKED_COMMIT.pack(
                    self.reference.kind_counts.get("commit", 0) - 1,
                    int(address), int(row["width_bytes"])))
        event = {"coordinate": list(coordinate), "kind": row["kind"],
            "earliest": row["earliest_issue_cycle"],
            "dependency": dependency, "port_ready": port_ready,
            "issue": self.compact.last_issue,
            "return": self.compact.last_return,
            "width_bytes": row["width_bytes"],
            "addresses": list(row["addresses"]),
            "banks": list(row["banks"])}
        self.request_digest.update(json.dumps(event, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8"))

    def finish_destination(self, destination):
        summary = self.compact.summary()
        require(summary["request_count"] == self.reference.requests and
                summary["kind_counts"] == self.reference.kind_counts and
                summary["byte_counts"] == self.reference.byte_counts and
                summary["packed_transaction_address_sha256"] ==
                    self.packed_reference.hexdigest() and
                summary["packed_commit_sequence_sha256"] ==
                    self.packed_commit.hexdigest() and
                summary["total_cycles"] == self.reference.last_cycle + 1,
                "per-destination cumulative ledger miter failed")
        expected_flat, expected_counts = \
            C.compact_outstanding_projection(self.reference)
        layout = tuple((bank * 8, 8) for bank in range(8)) + \
            tuple((64 + bank * 8, 8) for bank in range(6)) + \
            ((112, 16), (128, 1))
        expected_active = []
        for queue, (base, _capacity) in enumerate(layout):
            expected_active.append(sorted(
                value for value in
                expected_flat[base:base + expected_counts[queue]]
                if value > self.reference.last_cycle))
        require(list(self.compact.next_port) ==
                    C.compact_next_port_projection(self.reference) and
                P._active_compact_queues(self.compact,
                    self.compact.last_cycle) == expected_active,
                "per-destination port/outstanding miter failed")
        cache_reference, cache_compact = self.cache.states(
            self.request_digest.hexdigest())
        require(cache_reference["state_sha256"] ==
                    cache_compact["state_sha256"],
                "per-destination cache-state miter failed")
        state = {"configuration": self.configuration,
            "destination": destination,
            "request_count": summary["request_count"],
            "last_cycle": summary["total_cycles"] - 1,
            "kind_counts": summary["kind_counts"],
            "byte_counts": summary["byte_counts"],
            "address_sha256": summary[
                "packed_transaction_address_sha256"],
            "commit_sha256": summary["packed_commit_sequence_sha256"],
            "cache_sha256": cache_reference["state_sha256"],
            "last_psum_write_ready": list(self.last_psum_write_ready)}
        self.destination_digests.append(hashlib.sha256(json.dumps(
            state, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest())
        self.tokens.clear()
        self.reference.tokens.clear()
        self.rss.sample()

    def finish(self):
        summary = self.compact.summary()
        count = self.shard["destination_count"]
        require(len(self.destination_digests) == count and
                summary["kind_counts"].get("commit") ==
                    count * OUTPUT_BLOCKS,
                "shard destination/commit population incomplete")
        return {"configuration": self.configuration,
            "resource_manifest_sha256": RESOURCE_SHA256,
            "total_cycles": summary["total_cycles"],
            "request_count": summary["request_count"],
            "kind_counts": summary["kind_counts"],
            "byte_counts": summary["byte_counts"],
            "packed_transaction_address_sha256":
                summary["packed_transaction_address_sha256"],
            "packed_commit_sequence_sha256":
                summary["packed_commit_sequence_sha256"],
            "destination_state_chain_sha256": hashlib.sha256(
                "".join(self.destination_digests).encode("ascii")).hexdigest(),
            "per_request_miter": True, "per_destination_miter": True,
            "shard_reset_boundary": True, "paper_result": False}


def validate_three_configuration_metrics(rows, shard):
    require(type(rows) is list and len(rows) == 3 and
            [row.get("configuration") for row in rows] == list(CONFIGS),
            "three-configuration order/population drift")
    require(all(row.get("resource_manifest_sha256") == RESOURCE_SHA256 and
                row.get("per_request_miter") is True and
                row.get("per_destination_miter") is True and
                row.get("shard_reset_boundary") is True and
                row.get("kind_counts", {}).get("commit") ==
                    shard["destination_count"] * OUTPUT_BLOCKS and
                type(row.get("total_cycles")) is int and
                row["total_cycles"] > 0 for row in rows),
            "shard metric boundary drift")
    require(len(set(row["packed_commit_sequence_sha256"]
                    for row in rows)) == 1,
            "cross-configuration commit sequence drift")
    dense, equal, typed = [row["total_cycles"] for row in rows]
    return {"dense_cycles": dense, "bit_equal_cycles": equal,
            "bit_typed_cycles": typed,
            "dense_to_bit_typed_numerator": dense,
            "dense_to_bit_typed_denominator": typed,
            "bit_equal_to_bit_typed_numerator": equal,
            "bit_equal_to_bit_typed_denominator": typed,
            "floating_ratio_deferred_to_complete_reducer": True}


def synthetic_shard():
    shard = shard_descriptor(0)
    shard.update(destination_stop_exclusive=3, destination_count=3)
    rss = P.RssGate()
    rows = []
    for configuration in CONFIGS:
        session = ShardSession(configuration, shard, rss)
        for destination in range(3):
            active_channels = (range(16) if configuration == CONFIGS[0]
                               else range(4))
            contributors = [(destination % 9, channel)
                            for channel in active_channels]
            for output_block in range(OUTPUT_BLOCKS):
                last = ""
                for row in R.destination_transactions(
                        configuration, MODULE_ORDINAL, 0, destination,
                        output_block, contributors, "", session.cache):
                    session.accept(row, destination, output_block)
                    if row["kind"] == "psum_write":
                        last = row["produces"]
                commit = R.request(
                    "{}:commit:{}:{}".format(
                        configuration, destination, output_block),
                    configuration, "commit",
                    [(4 << 60) | ((destination * OUTPUT_BLOCKS +
                                  output_block) * R.OUTPUT_COMMIT_BYTES)],
                    [0], R.OUTPUT_COMMIT_BYTES,
                    [last] if last else ())
                session.accept(commit, destination, output_block)
            session.finish_destination(destination)
        rows.append(session.finish())
    ratios = validate_three_configuration_metrics(rows, shard)
    require(rows[0]["request_count"] > rows[2]["request_count"] and
            rows[1]["kind_counts"].get("compute") ==
                rows[2]["kind_counts"].get("compute"),
            "synthetic dense/bit/equal-service ordering drift")
    return {"schema": SCHEMA,
            "status": "PASS_M1671_SYNTHETIC_SHARD__NO_PAYLOAD_NO_EXECUTION",
            "shard": shard, "metrics": rows, "integer_ratios": ratios,
            "actual_payload": False, "actual_execution": False,
            "attempt_writes": 0, "paper_result": False}


def validate_authorities():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1645_SOURCE, M1645_SOURCE_SHA256, "exact M1645 source")
    regular_exact(M1656_SOURCE, M1656_SOURCE_SHA256, "exact M1656 source")
    verify_result_tree()
    verify_m1666()
    validate_grid()
    selected_record(shard_descriptor(0))
    selected_record(shard_descriptor(TOTAL_SHARDS - 1))
    require(not FUTURE_REVIEW.exists() and
            not os.path.lexists(str(FUTURE_RELEASE)),
            "future M1672/M1673 authority must be absent at source stage")
    return {"m1656_result_sha256": M1656_RESULT_SHA256,
            "m1666_review_sha256": M1666_REVIEW_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "resource_manifest_sha256": RESOURCE_SHA256,
            "grid": validate_grid(), "actual_payload": False,
            "actual_execution": False}


def reduce_complete_shards(rows):
    """Pure reducer for future sealed rows; no filesystem or payload access."""
    require(type(rows) is list and len(rows) == TOTAL_SHARDS,
            "complete D0 reducer requires exactly 8700 shard rows")
    totals = dict((configuration, {"cycles": 0, "requests": 0,
                                   "bytes": {}})
                  for configuration in CONFIGS)
    seen = set()
    for ordinal, bundle in enumerate(rows):
        require(type(bundle) is dict and
                bundle.get("shard") == shard_descriptor(ordinal) and
                bundle.get("shard_ordinal") == ordinal and
                bundle.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
                bundle.get("resource_manifest_sha256") == RESOURCE_SHA256,
                "shard identity/order drift in reducer")
        key = json.dumps(bundle["shard"], sort_keys=True,
                         separators=(",", ":"))
        require(key not in seen, "duplicate shard in reducer")
        seen.add(key)
        metrics = bundle.get("metrics")
        validate_three_configuration_metrics(metrics, bundle["shard"])
        for metric in metrics:
            target = totals[metric["configuration"]]
            target["cycles"] += metric["total_cycles"]
            target["requests"] += metric["request_count"]
            for name, value in metric["byte_counts"].items():
                target["bytes"][name] = target["bytes"].get(name, 0) + value
    dense = totals[CONFIGS[0]]["cycles"]
    equal = totals[CONFIGS[1]]["cycles"]
    typed = totals[CONFIGS[2]]["cycles"]
    return {"configuration_totals": totals,
            "ratio_of_sums": {
                "dense_to_bit_typed": {"numerator": dense,
                    "denominator": typed},
                "bit_equal_to_bit_typed": {"numerator": equal,
                    "denominator": typed}},
            "complete_shards": TOTAL_SHARDS,
            "full_d0_population_covered": True,
            "shard_isolated_cycle_model": True,
            "monolithic_full_call": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result_pending_independent_hammer": True}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "checkpoint": CHECKPOINT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "resource_manifest_sha256": RESOURCE_SHA256,
            "fixed_grid": fixed_grid(),
            "execution_model": {
                "reference": "exact M1539 AddressTimedScheduler",
                "compact": "exact M1610 CompactScheduler",
                "configuration_order": list(CONFIGS),
                "per_request_miter": True,
                "per_destination_cumulative_miter": True,
                "immutable_payload_fd_hash_required": True,
                "rss_absolute_limit_kib": RSS_ABSOLUTE_LIMIT_KIB,
                "rss_increment_limit_kib": RSS_INCREMENT_LIMIT_KIB,
                "attempt_before_payload": True,
                "automatic_retry": False,
                "recoverable_unit": "one sealed shard",
                "reduction": "integer ratio-of-sums only",
                "shard_reset_boundary": True},
            "future_expansion": {
                "D2_D3": "same grid engine after separate exact source review; module geometry and output blocks rebound",
                "D1": "excluded until numeric bit-exact bridge is admitted; never infer from D0/D2/D3",
                "full_decoder": False, "system": False},
            "claim_boundary": {"source_only": True,
                "actual_payload": False, "execution": False,
                "attempt_creation": False, "cycles": False,
                "traffic": False, "speedup": False,
                "energy": False, "rtl": False, "eda": False,
                "full_d0_result": False, "full_decoder": False,
                "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M1671_SOURCE_PREFLIGHT__NO_PAYLOAD_NO_EXECUTION",
            "authorities": validate_authorities(),
            "claim_boundary": describe()["claim_boundary"]}
    elif args.synthetic_self_test:
        output = synthetic_shard()
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

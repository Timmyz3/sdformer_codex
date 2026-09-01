#!/usr/bin/env python3
"""M1619 source-only contract machinery for a future decoder L2 prefix.

M1615 authorizes only this separately named source/contract step.  This file
therefore does not contain an ep34 payload path, cannot open a payload, and
cannot execute L2.  It freezes the canonical D0/call0 prefix and supplies the
fail-closed request/destination miter interface that a later, independently
reviewed execution source must use.

The prefix is destination 0..41 at timestep 0.  Forty-two is the smallest
row-major D0 prefix containing (y,x) parity 00/01/10/11: destinations
0, 1, 40 and 41.  State is cumulative across the entire prefix; resetting a
scheduler, cache, dependency state, digest, or counter per destination is
explicitly rejected.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1610_SOURCE = HERE / "build_m1610_decoder_compact_cycle_simulator_source.py"
M1610_SOURCE_SHA256 = "73d4bade27612a3dfcbdc3e7417d7180397629a5be1f9e23587a58ea487b84ce"
M1615 = HW / "reviews/m1615_m1610_decoder_compact_l0_l1_independent_review_r1_20260901"
M1615_REVIEW_SHA256 = "ab87c20943052570a24b4e7beb2bee3be913fcb95c388597c7fee844b1fe5f4c"
M1615_MANIFEST_SHA256 = "fd13594b29891c46e203e14bcffab823aab999ff1263ed492081ee937a681360"
M1615_OUTER_FILE_SHA256 = "800f6c2e7a48ca90e1513aa6d0f7bd3691bc434a687de400874706fde0afef0d"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1619_decoder_compact_l2_canonical_prefix_source_r1_v1"
STATUS = "SOURCE_ONLY__L2_CANONICAL_PREFIX_INTERFACE__NO_PAYLOAD_NO_EXECUTION"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
CHECKPOINT = "motion_ep34_live93"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
DECODER_STAGE = "D0"
CALL_ORDINAL = 0
MODULE_ORDINAL = 0
TIMESTEP = 0
OUTPUT_WIDTH = 40
OUTPUT_HEIGHT = 30
OUTPUT_BLOCKS = 4
PREFIX_DESTINATIONS = 42
DESTINATIONS = tuple(range(PREFIX_DESTINATIONS))
EXPECTED_COMMITS_PER_CONFIG = PREFIX_DESTINATIONS * OUTPUT_BLOCKS
RESOURCE_SHA256 = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
RSS_ABSOLUTE_LIMIT_KIB = 2 * 1024 * 1024
RSS_INCREMENT_LIMIT_KIB = 512 * 1024
HEX = frozenset("0123456789abcdef")

REQUEST_FIELDS = (
    "configuration", "schema_version", "module", "timestep", "destination",
    "output_block", "group", "subordinal", "request_ordinal", "kind",
    "earliest_issue_cycle",
    "dependency_ready_cycle", "port_ready_cycle", "issue_cycle", "beats",
    "return_cycle", "width_bytes", "addresses", "banks",
    "packed_event_sha256")

PREFIX_EXACT_FIELDS = (
    "configuration", "destination", "last_cycle", "request_count",
    "kind_counts", "byte_counts", "packed_transaction_address_sha256",
    "packed_commit_sequence_sha256", "next_port_calendar",
    "outstanding_active_returns", "numeric_dependency_state", "cache",
    "coverage_counters", "commit_count", "reset_count",
    "resource_manifest_sha256")


class M1619Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1619Error(message)


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
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def verify_flat_seal(path, review_sha, manifest_sha, outer_file_sha):
    path = Path(path)
    regular_exact(path / "review.json", review_sha, "M1615 review")
    regular_exact(path / "SHA256SUMS", manifest_sha, "M1615 manifest")
    regular_exact(path / "SHA256SUMS.seal.sha256", outer_file_sha,
                  "M1615 outer seal")
    require((path / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() == [manifest_sha, "SHA256SUMS"],
            "M1615 outer content drift")
    review = json.loads((path / "review.json").read_text(encoding="utf-8"))
    require(review.get("status") ==
            "PASS_M1610_L0_L1_EXACT_SYNTHETIC_MITER__ONLY_L2_SOURCE_AUTHORING_AUTHORIZED" and
            review.get("single_next_authorization", {}).get("execution_gate") ==
            "The L2 source must receive a new independent review before any actual-prefix execution. L3 remains separately gated.",
            "M1615 decision drift")
    return {"review_sha256": review_sha,
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_file_sha}


def load_m1610():
    regular_exact(M1610_SOURCE, M1610_SOURCE_SHA256, "M1610 source")
    spec = importlib.util.spec_from_file_location("m1619_bound_m1610",
                                                  str(M1610_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M1610")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.describe()
    require(tuple(module.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
            description.get("implemented_miter_levels") == ["L0", "L1"] and
            description.get("missing_miter_levels") == ["L2", "L3"] and
            module.M.validate_resource() == RESOURCE_SHA256,
            "M1610 boundary drift")
    return module


M1615_SEAL = verify_flat_seal(M1615, M1615_REVIEW_SHA256,
                              M1615_MANIFEST_SHA256,
                              M1615_OUTER_FILE_SHA256)
regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
C = load_m1610()


def hex64(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in HEX for character in value),
            label + " is not lowercase hex64")


def prefix_geometry():
    parity_mask = 0
    coordinates = []
    for destination in DESTINATIONS:
        y = destination // OUTPUT_WIDTH
        x = destination % OUTPUT_WIDTH
        parity_mask |= 1 << (((y & 1) << 1) | (x & 1))
        coordinates.append((destination, y, x))
    require(parity_mask == 0xf and coordinates[-1] == (41, 1, 1),
            "canonical prefix parity/extent drift")
    require(coordinates[0] == (0, 0, 0) and
            any(y == 0 and 0 < x < OUTPUT_WIDTH - 1
                for _d, y, x in coordinates) and
            any(0 < y < OUTPUT_HEIGHT - 1 and
                0 < x < OUTPUT_WIDTH - 1 for _d, y, x in coordinates),
            "canonical prefix lacks corner/edge/interior")
    return {"count": len(coordinates), "first": coordinates[0],
            "last": coordinates[-1], "parity_mask": parity_mask,
            "corner": True, "edge": True, "interior": True}


def dense_cache_history_proof():
    """Payload-free geometry proof that one persistent cache is exercised."""
    reference = C.M.WeightTileCache()
    compact = C.NumericWeightTileCache()
    hits = 0
    misses = 0
    evictions = 0
    destination_end_ticks = []
    # Two consecutive destinations are sufficient for miss/hit/eviction;
    # the execution contract nevertheless keeps all 42 consecutive states.
    for destination in DESTINATIONS[:2]:
        output_y = destination // OUTPUT_WIDTH
        output_x = destination % OUTPUT_WIDTH
        contributors = []
        for _iy, _ix, tap in C.M.destination_sources(
                output_y, output_x, 15, 20):
            for channel in range(1536):
                contributors.append((tap, channel))
        groups = C.M.bank_unique_groups(contributors, 1536)
        for output_block in range(OUTPUT_BLOCKS):
            for group in groups:
                keys = [(0, output_block, tap, channel // 16)
                        for tap, channel in group]
                before = set(reference.key_to_slot)
                expected = reference.prepare(keys)
                compact.begin_group()
                for key in keys:
                    compact.push_key(*key)
                compact.prepare_loaded()
                actual = []
                for ordinal in range(compact.miss_count):
                    actual.append(((compact.miss_module[ordinal],
                                    compact.miss_output_block[ordinal],
                                    compact.miss_tap[ordinal],
                                    compact.miss_channel_tile[ordinal]),
                                   compact.miss_slot[ordinal]))
                require(actual == expected, "dense cache miss/slot miter drift")
                misses += len(expected)
                hits += len(set(keys)) - len(expected)
                evictions += len([key for key in before
                                  if key not in reference.key_to_slot])
        destination_end_ticks.append(reference.tick)
    require(hits > 0 and misses > 0 and evictions > 0 and
            len(reference.key_to_slot) == 9 and
            destination_end_ticks[1] > destination_end_ticks[0],
            "dense prefix does not prove persistent cache history")
    return {"destinations": 2, "hits": hits, "misses": misses,
            "evictions": evictions, "final_entries": 9,
            "destination_end_ticks": destination_end_ticks}


def validate_request_receipt(row, configuration, expected_ordinal):
    require(type(row) is dict and all(field in row for field in REQUEST_FIELDS),
            "request receipt fields missing")
    require(row["configuration"] == configuration and
            row["request_ordinal"] == expected_ordinal,
            "request configuration/ordinal drift")
    for field in ("schema_version", "module", "timestep", "destination",
                  "output_block", "group", "subordinal"):
        require(type(row[field]) is int and row[field] >= 0,
                field + " is not a nonnegative integer")
    require(type(row["kind"]) is str and row["kind"] in C.KIND_NAMES,
            "request kind drift")
    for field in ("earliest_issue_cycle", "dependency_ready_cycle",
                  "port_ready_cycle", "issue_cycle", "beats",
                  "return_cycle", "width_bytes"):
        require(type(row[field]) is int and row[field] >= 0,
                field + " is not a nonnegative integer")
    require(row["beats"] > 0 and row["width_bytes"] > 0 and
            row["issue_cycle"] >= max(row["earliest_issue_cycle"],
                                      row["dependency_ready_cycle"],
                                      row["port_ready_cycle"]) and
            row["return_cycle"] >= row["issue_cycle"],
            "request timing relation invalid")
    require(type(row["addresses"]) is list and row["addresses"] and
            type(row["banks"]) is list and
            len(row["addresses"]) == len(row["banks"]) and
            all(type(value) is int and value >= 0 for value in row["addresses"]) and
            all(type(value) is int and value >= 0 for value in row["banks"]),
            "request address/bank vector invalid")
    hex64(row["packed_event_sha256"], "request packed event digest")


def validate_rss(row):
    require(type(row) is dict and
            all(type(row.get(field)) is int and row[field] >= 0 for field in
                ("baseline_rss_kib", "current_rss_kib", "hwm_rss_kib")),
            "RSS telemetry invalid")
    require(row["current_rss_kib"] < RSS_ABSOLUTE_LIMIT_KIB and
            row["hwm_rss_kib"] < RSS_ABSOLUTE_LIMIT_KIB and
            row["hwm_rss_kib"] >= row["current_rss_kib"] and
            row["hwm_rss_kib"] >= row["baseline_rss_kib"] and
            row["hwm_rss_kib"] - row["baseline_rss_kib"] <
                RSS_INCREMENT_LIMIT_KIB,
            "L2 RSS gate exceeded")


def validate_prefix_state(row, configuration, destination):
    require(type(row) is dict and all(field in row for field in PREFIX_EXACT_FIELDS),
            "prefix state fields missing")
    require(row["configuration"] == configuration and
            row["destination"] == destination and
            row["resource_manifest_sha256"] == RESOURCE_SHA256 and
            row["reset_count"] == 0,
            "prefix identity/reset drift")
    require(type(row["last_cycle"]) is int and row["last_cycle"] >= 0 and
            type(row["request_count"]) is int and row["request_count"] > 0 and
            type(row["commit_count"]) is int and
            row["commit_count"] == (destination + 1) * OUTPUT_BLOCKS,
            "prefix cycle/request/commit count invalid")
    require(type(row["kind_counts"]) is dict and
            type(row["byte_counts"]) is dict and
            all(key in C.KIND_NAMES and type(value) is int and value >= 0
                for key, value in row["kind_counts"].items()) and
            all(key in C.KIND_NAMES and type(value) is int and value >= 0
                for key, value in row["byte_counts"].items()) and
            sum(row["kind_counts"].values()) == row["request_count"],
            "prefix kind/request conservation invalid")
    for field in ("packed_transaction_address_sha256",
                  "packed_commit_sequence_sha256"):
        hex64(row[field], field)
    require(type(row["next_port_calendar"]) is list and
            len(row["next_port_calendar"]) == 24 and
            all(type(value) is int and value >= 0
                for value in row["next_port_calendar"]),
            "24-entry port calendar invalid")
    outstanding = row["outstanding_active_returns"]
    capacities = (8,) * 8 + (8,) * 6 + (16, 1)
    require(type(outstanding) is list and len(outstanding) == 16,
            "outstanding queue projection invalid")
    for queue, capacity in zip(outstanding, capacities):
        require(type(queue) is list and len(queue) <= capacity and
                queue == sorted(queue) and
                all(type(value) is int and value >= 0 for value in queue),
                "active outstanding queue invalid")
    dependency = row["numeric_dependency_state"]
    require(type(dependency) is dict and
            type(dependency.get("source_ready_cycle")) is int and
            dependency["source_ready_cycle"] >= 0 and
            type(dependency.get("persistent_control_ready_cycle")) is int and
            dependency["persistent_control_ready_cycle"] >= 0 and
            type(dependency.get("last_psum_write_ready")) is list and
            len(dependency["last_psum_write_ready"]) == OUTPUT_BLOCKS and
            all(type(value) is int and value >= 0
                for value in dependency["last_psum_write_ready"]),
            "retained numeric dependency state invalid")
    cache = row["cache"]
    require(type(cache) is dict and
            all(type(cache.get(field)) is int and cache[field] >= 0 for field in
                ("valid_entries", "tick", "hits", "misses", "evictions")) and
            cache["valid_entries"] <= 9,
            "cache state invalid")
    hex64(cache.get("state_sha256"), "cache state digest")
    coverage = row["coverage_counters"]
    require(type(coverage) is dict and
            type(coverage.get("outstanding_full_waits")) is int and
            coverage["outstanding_full_waits"] >= 0 and
            type(coverage.get("shared_1rw_serializations")) is int and
            coverage["shared_1rw_serializations"] >= 0,
            "coverage counters invalid")
    validate_rss(row.get("rss"))


class CanonicalPrefixMiter(object):
    """Cumulative future-run validator; no payload or scheduler creation."""
    __slots__ = ("configuration", "next_destination", "next_request_ordinal",
                 "previous_request_count", "previous_last_cycle",
                 "last_request_return",
                 "previous_cache_tick", "previous_cache_hits",
                 "previous_cache_misses", "previous_cache_evictions",
                 "previous_shared_1rw", "previous_port_calendar",
                 "source_ready_cycle", "control_ready_cycle",
                 "previous_address_digest", "previous_commit_digest",
                 "dense_cache_covered", "dense_psum_1rw_covered")

    def __init__(self, configuration):
        require(configuration in CONFIGS and configuration != FORBIDDEN_CONFIG,
                "configuration is not admitted")
        self.configuration = configuration
        self.next_destination = 0
        self.next_request_ordinal = 0
        self.previous_request_count = 0
        self.previous_last_cycle = -1
        self.last_request_return = -1
        self.previous_cache_tick = 0
        self.previous_cache_hits = 0
        self.previous_cache_misses = 0
        self.previous_cache_evictions = 0
        self.previous_shared_1rw = 0
        self.previous_port_calendar = [0] * 24
        self.source_ready_cycle = -1
        self.control_ready_cycle = -1
        self.previous_address_digest = ""
        self.previous_commit_digest = ""
        self.dense_cache_covered = False
        self.dense_psum_1rw_covered = False

    def accept_request_pair(self, reference, compact):
        validate_request_receipt(reference, self.configuration,
                                 self.next_request_ordinal)
        validate_request_receipt(compact, self.configuration,
                                 self.next_request_ordinal)
        require(all(reference[field] == compact[field]
                    for field in REQUEST_FIELDS),
                "request issue/return/dependency/port/beats/address exact miter failed")
        self.last_request_return = reference["return_cycle"]
        self.next_request_ordinal += 1

    def accept_destination_pair(self, reference, compact):
        destination = self.next_destination
        require(destination < PREFIX_DESTINATIONS,
                "too many prefix destinations")
        validate_prefix_state(reference, self.configuration, destination)
        validate_prefix_state(compact, self.configuration, destination)
        require(all(reference[field] == compact[field]
                    for field in PREFIX_EXACT_FIELDS),
                "destination cumulative-state exact miter failed")
        require(reference["request_count"] == self.next_request_ordinal and
                reference["request_count"] > self.previous_request_count and
                reference["last_cycle"] >= self.previous_last_cycle and
                reference["last_cycle"] >= self.last_request_return,
                "destination history was reset or skipped")
        require(reference["packed_transaction_address_sha256"] !=
                    self.previous_address_digest and
                reference["packed_commit_sequence_sha256"] !=
                    self.previous_commit_digest,
                "cumulative digest failed to advance")
        cache = reference["cache"]
        coverage = reference["coverage_counters"]
        require(cache["tick"] >= self.previous_cache_tick and
                cache["hits"] >= self.previous_cache_hits and
                cache["misses"] >= self.previous_cache_misses and
                cache["evictions"] >= self.previous_cache_evictions and
                coverage["shared_1rw_serializations"] >=
                    self.previous_shared_1rw and
                all(current >= previous for current, previous in zip(
                    reference["next_port_calendar"],
                    self.previous_port_calendar)),
                "cache/port/coverage history moved backwards")
        dependency = reference["numeric_dependency_state"]
        if self.source_ready_cycle < 0:
            self.source_ready_cycle = dependency["source_ready_cycle"]
            self.control_ready_cycle = dependency[
                "persistent_control_ready_cycle"]
        require(dependency["source_ready_cycle"] == self.source_ready_cycle and
                dependency["persistent_control_ready_cycle"] ==
                    self.control_ready_cycle,
                "persistent source/control dependency changed")
        if (self.configuration == CONFIGS[0] and cache["hits"] > 0 and
                cache["misses"] > 0 and cache["evictions"] > 0):
            self.dense_cache_covered = True
        if (self.configuration == CONFIGS[0] and
                coverage["shared_1rw_serializations"] > 0):
            self.dense_psum_1rw_covered = True
        self.previous_request_count = reference["request_count"]
        self.previous_last_cycle = reference["last_cycle"]
        self.previous_cache_tick = cache["tick"]
        self.previous_cache_hits = cache["hits"]
        self.previous_cache_misses = cache["misses"]
        self.previous_cache_evictions = cache["evictions"]
        self.previous_shared_1rw = coverage["shared_1rw_serializations"]
        self.previous_port_calendar = list(reference["next_port_calendar"])
        self.previous_address_digest = reference[
            "packed_transaction_address_sha256"]
        self.previous_commit_digest = reference[
            "packed_commit_sequence_sha256"]
        self.next_destination += 1

    def finish(self):
        require(self.next_destination == PREFIX_DESTINATIONS and
                self.previous_request_count == self.next_request_ordinal,
                "canonical prefix is incomplete")
        if self.configuration == CONFIGS[0]:
            require(self.dense_cache_covered,
                    "dense prefix lacks cache miss/hit/eviction coverage")
            require(self.dense_psum_1rw_covered,
                    "dense prefix lacks psum shared-1RW coverage")
        return {"configuration": self.configuration,
                "destinations": self.next_destination,
                "requests": self.next_request_ordinal,
                "commits": EXPECTED_COMMITS_PER_CONFIG,
                "dense_cache_covered": self.dense_cache_covered,
                "dense_psum_1rw_covered": self.dense_psum_1rw_covered,
                "final_commit_digest": self.previous_commit_digest}


def validate_bundle(rows):
    require(type(rows) is list and len(rows) == 3 and
            [row.get("configuration") for row in rows] == list(CONFIGS),
            "L2 bundle configuration order drift")
    require(all(row.get("destinations") == PREFIX_DESTINATIONS and
                row.get("commits") == EXPECTED_COMMITS_PER_CONFIG
                for row in rows), "L2 bundle population drift")
    require(len(set(row.get("final_commit_digest") for row in rows)) == 1,
            "cross-configuration commit stream drift")
    return True


def _digest(label, value):
    return hashlib.sha256((label + ":" + str(value)).encode("ascii")).hexdigest()


def synthetic_state(configuration, destination, requests, dense):
    cache_tick = destination + 1 if dense else 0
    return {"configuration": configuration, "destination": destination,
            "last_cycle": 1000 + requests, "request_count": requests,
            "kind_counts": {"commit": requests},
            "byte_counts": {"commit": requests * 384},
            "packed_transaction_address_sha256":
                _digest("address-" + configuration, destination),
            "packed_commit_sequence_sha256": _digest("commit", destination),
            "next_port_calendar": [destination + 1] * 24,
            "outstanding_active_returns": [[] for _index in range(16)],
            "numeric_dependency_state": {"source_ready_cycle": 1,
                "persistent_control_ready_cycle": 2,
                "last_psum_write_ready": [destination + 3] * OUTPUT_BLOCKS},
            "cache": {"valid_entries": 9 if dense else 0,
                "tick": cache_tick, "hits": cache_tick,
                "misses": cache_tick, "evictions": cache_tick,
                "state_sha256": _digest("cache-" + configuration,
                                         destination)},
            "coverage_counters": {"outstanding_full_waits": 0,
                "shared_1rw_serializations":
                    destination + 1 if dense else 0},
            "commit_count": (destination + 1) * OUTPUT_BLOCKS,
            "reset_count": 0, "resource_manifest_sha256": RESOURCE_SHA256,
            "rss": {"baseline_rss_kib": 100000,
                "current_rss_kib": 100000 + destination,
                "hwm_rss_kib": 100100 + destination}}


def synthetic_request(configuration, ordinal):
    return {"configuration": configuration, "schema_version": 1,
            "module": 0, "timestep": 0, "destination": ordinal // 4,
            "output_block": ordinal % 4, "group": 0, "subordinal": 0,
            "request_ordinal": ordinal,
            "kind": "commit", "earliest_issue_cycle": ordinal,
            "dependency_ready_cycle": ordinal,
            "port_ready_cycle": ordinal, "issue_cycle": ordinal,
            "beats": 2, "return_cycle": ordinal + 4,
            "width_bytes": 384, "addresses": [(4 << 60) + ordinal * 384],
            "banks": [0], "packed_event_sha256":
                _digest("event-" + configuration, ordinal)}


def static_self_test():
    geometry = prefix_geometry()
    cache_proof = dense_cache_history_proof()
    rows = []
    for configuration in CONFIGS:
        miter = CanonicalPrefixMiter(configuration)
        for destination in DESTINATIONS:
            for _output_block in range(OUTPUT_BLOCKS):
                request = synthetic_request(configuration,
                                            miter.next_request_ordinal)
                miter.accept_request_pair(request, dict(request))
            state = synthetic_state(configuration, destination,
                                    miter.next_request_ordinal,
                                    configuration == CONFIGS[0])
            miter.accept_destination_pair(state, json.loads(json.dumps(state)))
        rows.append(miter.finish())
    validate_bundle(rows)
    attacks = 0
    try:
        CanonicalPrefixMiter(FORBIDDEN_CONFIG)
    except M1619Error:
        attacks += 1
    try:
        actual_prefix_release()
    except M1619Error:
        attacks += 1
    bad = synthetic_request(CONFIGS[0], 0)
    bad_compact = dict(bad); bad_compact["beats"] = 3
    try:
        CanonicalPrefixMiter(CONFIGS[0]).accept_request_pair(bad, bad_compact)
    except M1619Error:
        attacks += 1
    require(attacks == 3, "static attacks were not rejected")
    return {"schema": SCHEMA,
            "status": "PASS_M1619_L2_SOURCE_INTERFACE_STATIC_ONLY__NO_PAYLOAD_NO_EXECUTION",
            "geometry": geometry, "dense_cache_history": cache_proof,
            "synthetic_sessions": rows, "attacks_rejected": attacks,
            "actual_payload": False, "l2_executed": False,
            "l3_executed": False, "pilot": False, "production": False,
            "paper_result": False}


def actual_prefix_release(_provider=None, _token=None):
    raise M1619Error(
        "M1619 is source-only; a new independent review must authorize any actual L2 prefix execution")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "identity": {"checkpoint": CHECKPOINT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "decoder_stage": DECODER_STAGE, "call_ordinal": CALL_ORDINAL,
                "module_ordinal": MODULE_ORDINAL, "timestep": TIMESTEP,
                "configurations": list(CONFIGS),
                "forbidden_configuration": FORBIDDEN_CONFIG,
                "resource_manifest_sha256": RESOURCE_SHA256},
            "canonical_prefix": {"order": "row-major consecutive",
                "first_destination": 0,
                "last_destination_inclusive": PREFIX_DESTINATIONS - 1,
                "destination_count": PREFIX_DESTINATIONS,
                "output_blocks": OUTPUT_BLOCKS,
                "expected_commits_per_configuration":
                    EXPECTED_COMMITS_PER_CONFIG,
                "state_reset_per_destination": False,
                "coverage": ["corner", "edge", "interior",
                    "four_xy_parities", "dense_cache_miss_hit_eviction",
                    "psum_shared_1rw", "dense_commit"]},
            "future_exact_miter": {"per_request": list(REQUEST_FIELDS),
                "per_destination_prefix": list(PREFIX_EXACT_FIELDS),
                "retained_state": ["numeric dependencies",
                    "24-entry port calendar", "129-slot active projection",
                    "nine-entry cache", "cumulative digests and counters",
                    "RSS current/HWM/baseline"]},
            "bindings": {"m1610_source_sha256": M1610_SOURCE_SHA256,
                "m1615": M1615_SEAL, "docs359_sha256": DOCS359_SHA256},
            "authorization": {"source_only": True,
                "independent_review_required": True,
                "actual_payload": False, "l2_execution": False,
                "l3": False, "pilot": False, "production": False,
                "cycles": False, "traffic": False, "speedup": False,
                "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--static-self-test", action="store_true")
    args = parser.parse_args(argv)
    result = describe() if args.describe else static_self_test()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

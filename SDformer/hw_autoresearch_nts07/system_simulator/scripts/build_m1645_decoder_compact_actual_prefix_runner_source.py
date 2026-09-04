#!/usr/bin/env python3
"""Source-only actual-prefix runner for the frozen ep34 decoder payload.

M1639 authorizes authoring this runner, not executing it.  The implementation
binds the exact M1539 reference scheduler, M1610 compact engine and M1638
configuration-bound L2 miter.  A private canonical entry is fully specified
for a later independently reviewed one-shot launcher, while every CLI mode in
this source remains payload-free.

The admitted population is exactly D0/call0/module0/timestep0, destinations
0..41 and output blocks 0..3, in the three non-product configuration order.
Every scheduled request is compared before it is accepted by M1638 and every
destination supplies cumulative reference/compact state.  Scheduler, cache,
digests and dependency history persist across the complete prefix.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1539_SOURCE = HERE / "build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
M1610_SOURCE = HERE / "build_m1610_decoder_compact_cycle_simulator_source.py"
M1638_SOURCE = HERE / "build_m1638_decoder_compact_l2_session_configuration_bound_successor_source.py"
M1539_SOURCE_SHA256 = "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4"
M1610_SOURCE_SHA256 = "73d4bade27612a3dfcbdc3e7417d7180397629a5be1f9e23587a58ea487b84ce"
M1638_SOURCE_SHA256 = "1b3961b0d0682980a035f5ad9ba880eb44929e56116f23f2e68cbb9e0a3fdecd"
M1639 = HW / (
    "reviews/m1639_m1638_decoder_compact_l2_session_configuration_bound_"
    "source_independent_review_r1_20260901")
M1639_REVIEW_SHA256 = "2af2dc261a4986e261bb74423b009a9cadd4449b647313d67487b5c5bd6c2ce6"
M1639_MANIFEST_SHA256 = "c67fc715da69067be262c3bab4b5c7ba33fc5e8ef85f08e5eb586b0b7f7a24fb"
M1639_OUTER_FILE_SHA256 = "68ef86a7cc778bbafb18ac8bce9b9258f63bd1aa6643d0a44a35fa4f73eba6b9"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1645_decoder_compact_actual_prefix_runner_source_r1_v1"
STATUS = "SOURCE_ONLY__ACTUAL_PREFIX_RUNNER_AUTHORED__NO_PAYLOAD_NO_EXECUTION"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
CHECKPOINT = "motion_ep34_live93"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
RESOURCE_SHA256 = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
DECODER_STAGE = "D0"
CALL_ORDINAL = 0
MODULE_ORDINAL = 0
TIMESTEP = 0
PREFIX_DESTINATIONS = 42
OUTPUT_BLOCKS = 4
OUTPUT_WIDTH = 40
RSS_ABSOLUTE_LIMIT_KIB = 2 * 1024 * 1024
RSS_INCREMENT_LIMIT_KIB = 512 * 1024
FUTURE_REVIEW = HW / (
    "reviews/m1646_m1645_decoder_compact_actual_prefix_runner_source_"
    "independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1647_m1646_m1645_decoder_compact_actual_prefix_runner_"
    "release_r1_20260901.json")


class M1645Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1645Error(message)


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


def load_exact(path, digest, name):
    regular_exact(path, digest, name + " source")
    spec = importlib.util.spec_from_file_location("m1645_exact_" + name,
                                                  str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import exact " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_m1639():
    regular_exact(M1639 / "review.json", M1639_REVIEW_SHA256, "M1639 review")
    regular_exact(M1639 / "SHA256SUMS", M1639_MANIFEST_SHA256,
                  "M1639 manifest")
    regular_exact(M1639 / "SHA256SUMS.seal.sha256",
                  M1639_OUTER_FILE_SHA256, "M1639 outer seal")
    require((M1639 / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [M1639_MANIFEST_SHA256, "SHA256SUMS"],
            "M1639 outer content drift")
    row = json.loads((M1639 / "review.json").read_text(encoding="utf-8"))
    require(row.get("status") ==
            "PASS_M1639_M1638_CONFIGURATION_BOUND_SOURCE__AUTHORIZE_ACTUAL_PREFIX_RUNNER_SOURCE_AUTHORING__NO_EXECUTION" and
            row.get("authorization", {}).get(
                "actual_prefix_runner_source_authoring") is True and
            row.get("authorization", {}).get(
                "actual_prefix_runner_execution") is False and
            row.get("authorization", {}).get("actual_payload") is False,
            "M1639 authorization drift")
    return {"review_sha256": M1639_REVIEW_SHA256,
            "manifest_sha256": M1639_MANIFEST_SHA256,
            "outer_seal_file_sha256": M1639_OUTER_FILE_SHA256}


regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
M1639_SEAL = verify_m1639()
R = load_exact(M1539_SOURCE, M1539_SOURCE_SHA256, "m1539")
C = load_exact(M1610_SOURCE, M1610_SOURCE_SHA256, "m1610")
L2 = load_exact(M1638_SOURCE, M1638_SOURCE_SHA256, "m1638")
require(tuple(R.CONFIGS) == CONFIGS and tuple(C.CONFIGS) == CONFIGS and
        tuple(L2.CONFIGS) == CONFIGS and
        R.FORBIDDEN_CONFIG == C.FORBIDDEN_CONFIG ==
            L2.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
        R.validate_resource() == C.M.validate_resource() ==
            L2.RESOURCE_SHA256 == RESOURCE_SHA256 and
        L2.PREFIX_DESTINATIONS == PREFIX_DESTINATIONS and
        L2.OUTPUT_BLOCKS == OUTPUT_BLOCKS,
        "exact engine/L2 boundary drift")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def read_rss_kib():
    values = {}
    with Path("/proc/self/status").open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmRSS:") or line.startswith("VmHWM:"):
                fields = line.split()
                require(len(fields) >= 2 and fields[1].isdigit(),
                        "malformed /proc RSS field")
                values[fields[0][:-1]] = int(fields[1])
    require("VmRSS" in values and "VmHWM" in values,
            "Linux current/HWM RSS telemetry unavailable")
    return values["VmRSS"], values["VmHWM"]


class RssGate(object):
    __slots__ = ("baseline", "baseline_hwm", "max_current", "max_hwm",
                 "gate_calls")

    def __init__(self):
        current, hwm = read_rss_kib()
        self.baseline = current
        self.baseline_hwm = hwm
        self.max_current = current
        self.max_hwm = hwm
        self.gate_calls = 0
        self.sample()

    def sample(self):
        current, hwm = read_rss_kib()
        self.max_current = max(self.max_current, current)
        self.max_hwm = max(self.max_hwm, hwm)
        self.gate_calls += 1
        require(current < RSS_ABSOLUTE_LIMIT_KIB and
                hwm < RSS_ABSOLUTE_LIMIT_KIB and hwm >= current and
                hwm >= self.baseline and
                hwm - self.baseline < RSS_INCREMENT_LIMIT_KIB,
                "actual-prefix RSS gate exceeded")
        return {"baseline_rss_kib": self.baseline,
                "current_rss_kib": current, "hwm_rss_kib": hwm}

    def summary(self):
        return {"baseline_current_rss_kib": self.baseline,
                "baseline_hwm_rss_kib": self.baseline_hwm,
                "max_current_rss_kib": self.max_current,
                "max_hwm_rss_kib": self.max_hwm,
                "absolute_limit_kib": RSS_ABSOLUTE_LIMIT_KIB,
                "increment_limit_kib": RSS_INCREMENT_LIMIT_KIB,
                "gate_calls": self.gate_calls}


def _flag_and_subordinal(suffix):
    if suffix == "typed_desc":
        return C.FLAG_TYPED_DESC, 0
    if suffix.startswith("k1_desc"):
        return C.FLAG_K1_DESC, int(suffix[len("k1_desc"):])
    if suffix.startswith("refill") and suffix.endswith("_weight_write"):
        return C.FLAG_REFILL_WRITE, int(
            suffix[len("refill"):-len("_weight_write")])
    if suffix.startswith("refill"):
        return C.FLAG_REFILL, int(suffix[len("refill"):])
    if suffix == "typed_weight":
        return C.FLAG_TYPED_WEIGHT, 0
    if suffix.startswith("k1_weight"):
        return C.FLAG_K1_WEIGHT, int(suffix[len("k1_weight"):])
    fixed = {"psum_read": C.FLAG_PSUM_READ, "compute": C.FLAG_COMPUTE,
             "psum_write": C.FLAG_PSUM_WRITE}
    require(suffix in fixed, "unknown actual-prefix request suffix")
    return fixed[suffix], 0


def actual_coordinate(configuration, row, request_ordinal,
                      destination, output_block):
    """Encode D0 coordinates without M1610's module-3 synthetic adapter."""
    require(configuration in CONFIGS and row.get("config") == configuration,
            "request configuration drift")
    commit_id = "{}:commit:{}:{}".format(
        configuration, destination, output_block)
    if row.get("kind") == "commit":
        require(row.get("id") == commit_id, "commit identifier drift")
        flag, subordinal, group = C.FLAG_COMMIT, 0, C.U32_SENTINEL
    else:
        prefix = "{}:m0:t0:d{}:ob{}:g".format(
            configuration, destination, output_block)
        identifier = row.get("id", "")
        require(identifier.startswith(prefix), "D0 request identifier drift")
        tail = identifier[len(prefix):]
        fields = tail.split(":", 1)
        require(len(fields) == 2 and fields[0].isdigit(),
                "D0 group identifier drift")
        group = int(fields[0])
        flag, subordinal = _flag_and_subordinal(fields[1])
    return (1, MODULE_ORDINAL, TIMESTEP, flag, int(destination),
            int(output_block), group, subordinal, int(request_ordinal))


def _active_compact_queues(compact, last_cycle):
    output = []
    layout = tuple((bank * 8, 8) for bank in range(8)) + \
        tuple((64 + bank * 8, 8) for bank in range(6)) + \
        ((112, 16), (128, 1))
    for queue, (base, capacity) in enumerate(layout):
        count = compact.outstanding_count[queue]
        require(0 <= count <= capacity, "compact outstanding count drift")
        output.append(sorted(value for value in
                             compact.outstanding[base:base + count]
                             if value > last_cycle))
    return output


class MirroredWeightCache(object):
    """Feed the exact M1539 and M1610 cache implementations in lockstep."""
    __slots__ = ("reference", "compact", "hits", "misses", "evictions",
                 "previous_reference_sha", "previous_compact_sha")

    def __init__(self):
        self.reference = R.WeightTileCache()
        self.compact = C.NumericWeightTileCache()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        empty = hashlib.sha256(b"").hexdigest()
        self.previous_reference_sha = empty
        self.previous_compact_sha = empty

    def prepare(self, keys):
        keys = [tuple(int(value) for value in key) for key in keys]
        unique = []
        for key in keys:
            if key not in unique:
                unique.append(key)
        before = set(self.reference.key_to_slot)
        expected = self.reference.prepare(keys)
        self.compact.begin_group()
        for key in keys:
            self.compact.push_key(*key)
        self.compact.prepare_loaded()
        actual = []
        for ordinal in range(self.compact.miss_count):
            actual.append(((self.compact.miss_module[ordinal],
                            self.compact.miss_output_block[ordinal],
                            self.compact.miss_tap[ordinal],
                            self.compact.miss_channel_tile[ordinal]),
                           self.compact.miss_slot[ordinal]))
        require(actual == expected, "actual-prefix cache miss/slot miter failed")
        self.misses += len(expected)
        self.hits += len(unique) - len(expected)
        self.evictions += len([key for key in before
                               if key not in self.reference.key_to_slot])
        self._check_exact()
        return expected

    def slot(self, key):
        key = tuple(int(value) for value in key)
        reference = self.reference.slot(key)
        compact = self.compact.slot_for(*key)
        require(reference == compact, "actual-prefix cache slot miter failed")
        return reference

    def _reference_payload(self):
        entries = sorted([list(key) + [slot, self.reference.age[key]]
                          for key, slot in self.reference.key_to_slot.items()])
        return {"tick": self.reference.tick, "entries": entries}

    def _compact_payload(self):
        entries = []
        for slot in range(C.WEIGHT_CACHE_ENTRIES):
            if self.compact.valid[slot]:
                entries.append([self.compact.module[slot],
                    self.compact.output_block[slot], self.compact.tap[slot],
                    self.compact.channel_tile[slot], slot,
                    self.compact.age[slot]])
        return {"tick": self.compact.tick, "entries": sorted(entries)}

    def _check_exact(self):
        require(self._reference_payload() == self._compact_payload(),
                "actual-prefix cache state miter failed")

    def states(self, accepted_weight_sha256):
        self._check_exact()
        reference_sha = hashlib.sha256(canonical_bytes(
            self._reference_payload())).hexdigest()
        compact_sha = hashlib.sha256(canonical_bytes(
            self._compact_payload())).hexdigest()
        require(reference_sha == compact_sha, "cache state digest miter failed")
        common = {"valid_entries": len(self.reference.key_to_slot),
                  "tick": self.reference.tick, "hits": self.hits,
                  "misses": self.misses, "evictions": self.evictions,
                  "accepted_weight_request_sha256": accepted_weight_sha256}
        reference = dict(common)
        reference.update({"state_sha256": reference_sha,
                          "previous_state_sha256": self.previous_reference_sha})
        compact = dict(common)
        compact.update({"state_sha256": compact_sha,
                        "previous_state_sha256": self.previous_compact_sha})
        self.previous_reference_sha = reference_sha
        self.previous_compact_sha = compact_sha
        return reference, compact


class PrefixSession(object):
    """Cumulative exact reference/compact/L2 session for one configuration."""
    __slots__ = ("configuration", "rss", "reference", "compact", "cache",
                 "miter", "tokens", "last_psum_write_ready",
                 "packed_reference", "packed_commit", "finished")

    def __init__(self, configuration, rss):
        require(configuration in CONFIGS and configuration != FORBIDDEN_CONFIG,
                "configuration is not admitted")
        require(type(rss) is RssGate, "exact RSS gate required")
        self.configuration = configuration
        self.rss = rss
        self.reference = R.AddressTimedScheduler(configuration)
        self.compact = C.CompactScheduler(configuration)
        self.cache = MirroredWeightCache()
        self.miter = L2.CanonicalPrefixMiter(configuration)
        self.tokens = {}
        self.last_psum_write_ready = [0] * OUTPUT_BLOCKS
        self.packed_reference = hashlib.sha256()
        self.packed_commit = hashlib.sha256()
        self.finished = False

    def accept(self, row, destination, output_block):
        require(not self.finished and destination == self.miter.next_destination,
                "request destination/session state drift")
        ordinal = self.miter.next_request_ordinal
        coordinate = actual_coordinate(self.configuration, row, ordinal,
                                       destination, output_block)
        missing = [token for token in row["dependencies"]
                   if token not in self.tokens]
        require(not missing, "actual-prefix unresolved dependency")
        dependency = max([self.tokens[token]
                          for token in row["dependencies"]] or
                         [row["earliest_issue_cycle"]])
        port_ready = C.reference_port_ready(row, self.reference)
        reference_receipt = self.reference.schedule_one(row)
        require(reference_receipt["dependency_ready_cycle"] == dependency,
                "reference dependency differs from numeric dependency")
        self.compact.begin_addresses()
        for address, bank in zip(row["addresses"], row["banks"]):
            self.compact.push_address(address, bank)
        self.compact.schedule_loaded(
            C.kind_index(row["kind"]), row["width_bytes"],
            row["earliest_issue_cycle"], dependency, *coordinate)
        require((self.compact.last_issue, self.compact.last_return,
                 self.compact.last_dependency, self.compact.last_port_ready) ==
                (reference_receipt["issue_cycle"],
                 reference_receipt["return_cycle"],
                 reference_receipt["dependency_ready_cycle"], port_ready),
                "actual-prefix request cycle miter failed")
        require(self.compact.next_port ==
                    C.compact_next_port_projection(self.reference) and
                self.compact.last_cycle == self.reference.last_cycle and
                self.compact.requests == self.reference.requests,
                "actual-prefix cumulative scheduler miter failed")
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
                    "actual-prefix outstanding miter failed")
        if row["produces"]:
            require(row["produces"] not in self.tokens,
                    "duplicate numeric producer")
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
                 "beats": self.compact.last_beats,
                 "width_bytes": row["width_bytes"],
                 "addresses": list(row["addresses"]),
                 "banks": list(row["banks"])}
        receipt = {"configuration": self.configuration,
                   "schema_version": coordinate[0], "module": coordinate[1],
                   "timestep": coordinate[2], "destination": coordinate[4],
                   "output_block": coordinate[5], "group": coordinate[6],
                   "subordinal": coordinate[7],
                   "request_ordinal": coordinate[8], "kind": row["kind"],
                   "earliest_issue_cycle": row["earliest_issue_cycle"],
                   "dependency_ready_cycle": dependency,
                   "port_ready_cycle": port_ready,
                   "issue_cycle": self.compact.last_issue,
                   "beats": self.compact.last_beats,
                   "return_cycle": self.compact.last_return,
                   "width_bytes": row["width_bytes"],
                   "addresses": list(row["addresses"]),
                   "banks": list(row["banks"]),
                   "packed_event_sha256": hashlib.sha256(
                       canonical_bytes(event)).hexdigest()}
        compact_receipt = json.loads(canonical_bytes(receipt).decode("utf-8"))
        self.miter.accept_request_pair(receipt, compact_receipt)
        return reference_receipt

    def finish_destination(self, destination):
        require(destination == self.miter.next_destination and
                self.reference.requests == self.compact.requests ==
                    self.miter.next_request_ordinal,
                "destination cumulative request count drift")
        last_cycle = self.reference.last_cycle
        reference_cache, compact_cache = self.cache.states(
            self.miter.cache_request_digest.hexdigest())
        reference_state = {
            "configuration": self.configuration, "destination": destination,
            "last_cycle": last_cycle, "request_count": self.reference.requests,
            "kind_counts": dict(self.reference.kind_counts),
            "byte_counts": dict(self.reference.byte_counts),
            "packed_transaction_address_sha256":
                self.miter.address_digest.hexdigest(),
            "packed_commit_sequence_sha256":
                self.miter.commit_digest.hexdigest(),
            "next_port_calendar": C.compact_next_port_projection(self.reference),
            "outstanding_active_returns": [sorted(value for value in queue
                if value > last_cycle) for queue in
                self.miter.queue_returns],
            "numeric_dependency_state": {"source_ready_cycle": 0,
                "persistent_control_ready_cycle": 0,
                "last_psum_write_ready": list(self.last_psum_write_ready)},
            "cache": reference_cache,
            "coverage_counters": {"outstanding_full_waits":
                self.compact.full_waits, "shared_1rw_serializations":
                self.compact.shared_1rw_serializations},
            "commit_count": self.reference.kind_counts.get("commit", 0),
            "reset_count": 0, "resource_manifest_sha256": RESOURCE_SHA256,
            "rss": self.rss.sample()}
        compact_summary = self.compact.summary()
        require(compact_summary["packed_transaction_address_sha256"] ==
                    self.packed_reference.hexdigest() and
                compact_summary["packed_commit_sequence_sha256"] ==
                    self.packed_commit.hexdigest(),
                "actual-prefix packed address/commit miter failed")
        compact_state = json.loads(canonical_bytes(reference_state).decode(
            "utf-8"))
        compact_state.update({
            "last_cycle": self.compact.last_cycle,
            "request_count": self.compact.requests,
            "kind_counts": compact_summary["kind_counts"],
            "byte_counts": compact_summary["byte_counts"],
            "next_port_calendar": list(self.compact.next_port),
            "outstanding_active_returns": _active_compact_queues(
                self.compact, self.compact.last_cycle),
            "cache": compact_cache})
        self.miter.accept_destination_pair(reference_state, compact_state)
        self.tokens.clear()
        self.reference.tokens.clear()
        return {"destination": destination, "last_cycle": last_cycle,
                "requests": self.reference.requests}

    def finish(self):
        require(not self.finished and
                self.miter.next_destination == PREFIX_DESTINATIONS,
                "actual-prefix session incomplete or repeated")
        self.finished = True
        receipt = self.miter.finish()
        summary = self.compact.summary()
        return receipt, {"configuration": self.configuration,
            "total_cycles": summary["total_cycles"],
            "request_count": summary["request_count"],
            "kind_counts": summary["kind_counts"],
            "byte_counts": summary["byte_counts"],
            "packed_transaction_address_sha256":
                summary["packed_transaction_address_sha256"],
            "packed_commit_sequence_sha256":
                summary["packed_commit_sequence_sha256"],
            "independent_hammer_pending": True, "paper_result": False}


class ImmutablePrefixPlane(object):
    """Verify the canonical payload FD, retain only timestep-zero bytes."""
    __slots__ = ("path", "shape", "expected_sha256", "opened_sha256",
                 "opened_size", "snapshot")

    def __init__(self, path, shape, expected_sha256):
        self.path = Path(path)
        self.shape = tuple(int(value) for value in shape)
        self.expected_sha256 = str(expected_sha256)
        require(self.shape == tuple(R.INPUT_SHAPES[MODULE_ORDINAL]),
                "actual-prefix payload shape drift")
        mode = self.path.lstat().st_mode
        require(stat.S_ISREG(mode) and not self.path.is_symlink(),
                "actual-prefix payload must be regular non-symlink")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(self.path), flags)
        try:
            opened = os.fstat(descriptor)
            require(stat.S_ISREG(opened.st_mode), "opened payload is not regular")
            stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
            self.opened_sha256 = digest.hexdigest()
            self.opened_size = int(opened.st_size)
            require(self.opened_sha256 == self.expected_sha256,
                    "opened payload SHA drift")
            elements = self.shape[2] * self.shape[3] * self.shape[4]
            prefix_bytes = (elements + 7) // 8
            stream.seek(0)
            self.snapshot = bytes(stream.read(prefix_bytes))
            require(len(self.snapshot) == prefix_bytes,
                    "timestep-zero payload prefix truncated")
            stream.close()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def bit(self, channel, y, x):
        channels, height, width = self.shape[2:]
        require(0 <= channel < channels and 0 <= y < height and
                0 <= x < width, "actual-prefix bit index out of range")
        index = (channel * height + y) * width + x
        return (self.snapshot[index >> 3] >> (index & 7)) & 1


def _selected_payload():
    manifest = R.strict_json(R.M1521_MANIFEST)
    R.validate_population_manifest(manifest)
    row = manifest["records"][CALL_ORDINAL]
    require(row.get("global_call_ordinal") == CALL_ORDINAL and
            row.get("module_ordinal") == MODULE_ORDINAL and
            tuple(row.get("shape", ())) == tuple(R.INPUT_SHAPES[MODULE_ORDINAL]),
            "canonical call-zero D0 identity drift")
    path = (R.M1521_ROOT / row["positive_output"]).resolve()
    require(path.parent == (R.M1521_ROOT / "payloads").resolve(),
            "canonical payload path escaped payload directory")
    return path, tuple(row["shape"]), row["positive_output_sha256"]


def _schedule_prefix(configuration, plane, rss):
    session = PrefixSession(configuration, rss)
    cin, cout, hin, win, _hout, wout = R.GEOMETRY[MODULE_ORDINAL]
    require(wout == OUTPUT_WIDTH and (cout + 95) // 96 == OUTPUT_BLOCKS,
            "D0 prefix geometry drift")
    getter = lambda channel, y, x: plane.bit(channel, y, x)
    for destination in range(PREFIX_DESTINATIONS):
        oy, ox = divmod(destination, OUTPUT_WIDTH)
        contributors = R.contributors_for_destination(
            getter, configuration, cin, hin, win, oy, ox)
        for output_block in range(OUTPUT_BLOCKS):
            last = ""
            for row in R.destination_transactions(
                    configuration, MODULE_ORDINAL, TIMESTEP, destination,
                    output_block, contributors, "", session.cache):
                receipt = session.accept(row, destination, output_block)
                if row["kind"] == "psum_write":
                    last = row["produces"]
            commit_id = "{}:commit:{}:{}".format(
                configuration, destination, output_block)
            commit_address = ((4 << 60) | (MODULE_ORDINAL << 52) |
                (TIMESTEP << 44) |
                ((destination * OUTPUT_BLOCKS + output_block) *
                 R.OUTPUT_COMMIT_BYTES))
            commit = R.request(commit_id, configuration, "commit",
                [commit_address], [0], R.OUTPUT_COMMIT_BYTES,
                [last] if last else ())
            session.accept(commit, destination, output_block)
        session.finish_destination(destination)
    return session.finish()


def _run_bound_actual_prefix():
    """Private future-launch target; never called by this source or its tests."""
    require(FUTURE_REVIEW.exists() and FUTURE_RELEASE.exists(),
            "M1646 review and M1647 release are required before execution")
    R.validate_authorities(True)
    path, shape, payload_sha = _selected_payload()
    rss = RssGate()
    plane = ImmutablePrefixPlane(path, shape, payload_sha)
    rss.sample()
    receipts = []
    metrics = []
    for configuration in CONFIGS:
        receipt, metric = _schedule_prefix(configuration, plane, rss)
        receipts.append(receipt)
        metrics.append(metric)
    summaries = [receipt.as_dict() for receipt in receipts]
    L2.validate_bundle(receipts)
    rss.sample()
    return {"schema": SCHEMA, "status": "FUTURE_ONE_SHOT_PREFIX_CANDIDATE",
            "checkpoint": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA256,
            "decoder_stage": DECODER_STAGE, "call_ordinal": CALL_ORDINAL,
            "module_ordinal": MODULE_ORDINAL, "timestep": TIMESTEP,
            "destinations": PREFIX_DESTINATIONS,
            "configuration_order": list(CONFIGS), "sessions": summaries,
            "metrics": metrics, "rss": rss.summary(),
            "payload_fd_sha256": plane.opened_sha256,
            "payload_fd_size": plane.opened_size,
            "independent_hammer_pending": True, "cycles_pending_hammer": True,
            "bytes_pending_hammer": True, "product_capture": False,
            "l3": False, "full_decoder": False, "production": False,
            "paper_result": False}


def actual_prefix_release(_token=None):
    raise M1645Error(
        "M1645 is source-only; M1646 review and M1647 release are required")


def _synthetic_session(configuration, rss):
    session = PrefixSession(configuration, rss)
    for destination in range(PREFIX_DESTINATIONS):
        tap = destination % 9
        contributors = [(tap, channel) for channel in range(16)]
        for output_block in range(OUTPUT_BLOCKS):
            last = ""
            for row in R.destination_transactions(
                    configuration, MODULE_ORDINAL, TIMESTEP, destination,
                    output_block, contributors, "", session.cache):
                session.accept(row, destination, output_block)
                if row["kind"] == "psum_write":
                    last = row["produces"]
            commit_id = "{}:commit:{}:{}".format(
                configuration, destination, output_block)
            address = ((4 << 60) +
                       (destination * OUTPUT_BLOCKS + output_block) *
                       R.OUTPUT_COMMIT_BYTES)
            session.accept(R.request(commit_id, configuration, "commit",
                [address], [0], R.OUTPUT_COMMIT_BYTES,
                [last] if last else ()), destination, output_block)
        session.finish_destination(destination)
    return session.finish()


def static_self_test():
    require(not FUTURE_REVIEW.exists() and not FUTURE_RELEASE.exists(),
            "future review/release must remain absent during author tests")
    rss = RssGate()
    receipts = []
    metrics = []
    for configuration in CONFIGS:
        receipt, metric = _synthetic_session(configuration, rss)
        receipts.append(receipt); metrics.append(metric)
    summaries = [receipt.as_dict() for receipt in receipts]
    L2.validate_bundle(receipts)
    attacks = 0
    try:
        actual_prefix_release()
    except M1645Error:
        attacks += 1
    try:
        actual_coordinate(CONFIGS[0],
            R.request(CONFIGS[0] + ":commit:0:0", CONFIGS[0], "commit",
                      [0], [0], 384), 0, 1, 0)
    except M1645Error:
        attacks += 1
    try:
        actual_coordinate(FORBIDDEN_CONFIG, {}, 0, 0, 0)
    except M1645Error:
        attacks += 1
    require(attacks == 3, "source-only/coordinate attacks not rejected")
    return {"schema": SCHEMA,
            "status": "PASS_M1645_ACTUAL_PREFIX_RUNNER_SOURCE_STATIC_ONLY",
            "configurations": [row["configuration"] for row in summaries],
            "distinct_sessions": len(set(row["session_identity"]
                                         for row in summaries)),
            "metrics": metrics, "rss": rss.summary(),
            "attacks_rejected": attacks, "actual_payload": False,
            "actual_execution": False, "cycles_admitted": False,
            "bytes_admitted": False, "product_capture": False,
            "l3": False, "full_decoder": False, "production": False,
            "gpu": False, "eda": False, "paper_result": False}


def validate_authorities():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1539_SOURCE, M1539_SOURCE_SHA256, "M1539 source")
    regular_exact(M1610_SOURCE, M1610_SOURCE_SHA256, "M1610 source")
    regular_exact(M1638_SOURCE, M1638_SOURCE_SHA256, "M1638 source")
    verify_m1639()
    R.validate_authorities(False)
    _selected_payload()
    require(not FUTURE_REVIEW.exists() and not FUTURE_RELEASE.exists(),
            "future review/release unexpectedly present")
    return {"m1539_source_sha256": M1539_SOURCE_SHA256,
            "m1610_source_sha256": M1610_SOURCE_SHA256,
            "m1638_source_sha256": M1638_SOURCE_SHA256,
            "m1639": M1639_SEAL, "actual_payload": False,
            "actual_execution": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "bindings": {"checkpoint": CHECKPOINT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "resource_manifest_sha256": RESOURCE_SHA256,
                "m1539_source_sha256": M1539_SOURCE_SHA256,
                "m1610_source_sha256": M1610_SOURCE_SHA256,
                "m1638_source_sha256": M1638_SOURCE_SHA256,
                "m1639_review_sha256": M1639_REVIEW_SHA256,
                "docs359_sha256": DOCS359_SHA256},
            "fixed_population": {"decoder_stage": DECODER_STAGE,
                "call_ordinal": CALL_ORDINAL,
                "module_ordinal": MODULE_ORDINAL, "timestep": TIMESTEP,
                "destinations": list(range(PREFIX_DESTINATIONS)),
                "output_blocks": list(range(OUTPUT_BLOCKS)),
                "configuration_order": list(CONFIGS)},
            "exact_path": {"reference_scheduler": "M1539 AddressTimedScheduler",
                "compact_engine": "M1610 CompactScheduler",
                "cache_miter": "M1539 WeightTileCache vs M1610 NumericWeightTileCache",
                "per_request_miter": "M1638 accept_request_pair",
                "per_destination_miter": "M1638 accept_destination_pair",
                "session_finish_bundle": "three distinct M1638 sessions",
                "m1610_module3_synthetic_adapter_reused": False,
                "strict_d0_coordinate_encoder": True},
            "rss_gate": {"baseline_current_measured": True,
                "current_measured": True, "hwm_measured": True,
                "absolute_limit_kib": RSS_ABSOLUTE_LIMIT_KIB,
                "increment_limit_kib": RSS_INCREMENT_LIMIT_KIB},
            "future_result": {"one_shot_prefix_only": True,
                "cycles_pending_independent_hammer": True,
                "bytes_pending_independent_hammer": True,
                "product_capture": False, "l3": False,
                "full_decoder": False, "production": False,
                "paper_result": False},
            "future_gate": {"review": str(FUTURE_REVIEW.relative_to(HW)),
                "release": str(FUTURE_RELEASE.relative_to(HW)),
                "review_present": FUTURE_REVIEW.exists(),
                "release_present": FUTURE_RELEASE.exists()},
            "authorization": {"source_only": True,
                "different_author_review": True, "actual_payload": False,
                "actual_execution": False, "attempt_creation": False,
                "release_creation": False, "l2_execution": False,
                "l3": False, "full_decoder": False, "production": False,
                "cycles": False, "traffic": False, "energy": False,
                "speedup": False, "system_speedup": False, "gpu": False,
                "rtl": False, "eda": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        validate_authorities()
        output = describe()
    elif args.synthetic_self_test:
        output = static_self_test()
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

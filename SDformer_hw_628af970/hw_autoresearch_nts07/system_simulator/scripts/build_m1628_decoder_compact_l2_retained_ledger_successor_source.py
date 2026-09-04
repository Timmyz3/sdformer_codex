#!/usr/bin/env python3
"""M1628 source-only repair of the decoder compact L2 prefix interface.

This successor repairs exactly the three M1620 P1 findings.  It imports the
frozen M1619 source, but it does not contain or open an ep34 payload and cannot
execute an actual prefix.  Accepted request pairs are the sole authority for
the cumulative request, byte, transaction-address, commit, port, outstanding
return and psum-write ledgers.  Destination rows may only match those ledgers.

Finish receipts are one-time, per-session authenticated objects.  They bind
configuration, resource, population, coverage and final cumulative state.
Bundle validation accepts only three genuine, distinct, not-yet-consumed
receipts in the frozen configuration order.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1619_SOURCE = HERE / "build_m1619_decoder_compact_l2_canonical_prefix_source.py"
M1619_SOURCE_SHA256 = "12c57983dee200c6c2eda3c42c13b3e111ec1b2ade86309f4b4b65f1b90306a0"
M1620 = HW / (
    "reviews/m1620_m1619_decoder_compact_l2_canonical_prefix_source_"
    "independent_review_r1_20260901")
M1620_REVIEW_SHA256 = "38b2b52316eaa90d28b668dfbf07a14e1dc0b0ea3b2f283c0699cb3b6d256bd9"
M1620_MANIFEST_SHA256 = "8e2379b1a51a3f9e01f908fde0eb3f86ac516f500458a83f234888dbf74fcf76"
M1620_OUTER_FILE_SHA256 = "e67ee5a011904dcd773fac81d6c88b29a677cb8ad2ad3e41bc3c775bf0e02618"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1628_decoder_compact_l2_retained_ledger_successor_source_r1_v1"
STATUS = "SOURCE_ONLY__M1620_THREE_P1_REPAIRED__NO_PAYLOAD_NO_EXECUTION"
FUTURE_REVIEW = HW / (
    "reviews/m1629_m1628_decoder_compact_l2_retained_ledger_source_"
    "independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1633_m1629_m1628_decoder_compact_l2_actual_prefix_"
    "source_release_r1_20260901.json")


class M1628Error(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise M1628Error(message)


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def verify_m1620():
    regular_exact(M1620 / "review.json", M1620_REVIEW_SHA256,
                  "M1620 review")
    regular_exact(M1620 / "SHA256SUMS", M1620_MANIFEST_SHA256,
                  "M1620 manifest")
    regular_exact(M1620 / "SHA256SUMS.seal.sha256",
                  M1620_OUTER_FILE_SHA256, "M1620 outer seal")
    require((M1620 / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [M1620_MANIFEST_SHA256, "SHA256SUMS"],
            "M1620 outer seal content drift")
    review = json.loads((M1620 / "review.json").read_text(encoding="utf-8"))
    require(review.get("status") ==
            "NO_GO_M1619_ACTUAL_L2_RUNNER_SOURCE__THREE_P1_INTERFACE_GAPS__SUCCESSOR_SOURCE_REPAIR_ONLY" and
            [item.get("id") for item in review.get("findings", {}).get(
                "p1", [])] == [
                    "P1_CROSS_DESTINATION_STATE_CONTINUITY_NOT_PROVEN",
                    "P1_REQUEST_SCOPE_AND_PREFIX_LEDGER_NOT_BOUND",
                    "P1_CROSS_CONFIGURATION_FRESH_SESSION_PROOF_ABSENT"],
            "M1620 decision drift")


def load_m1619():
    regular_exact(M1619_SOURCE, M1619_SOURCE_SHA256, "M1619 source")
    spec = importlib.util.spec_from_file_location("m1628_bound_m1619",
                                                  str(M1619_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1619")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1619_decoder_compact_l2_canonical_prefix_source_r1_v1" and
            module.RESOURCE_SHA256 ==
            "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10",
            "M1619 semantic boundary drift")
    return module


regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
verify_m1620()
B = load_m1619()
CONFIGS = B.CONFIGS
FORBIDDEN_CONFIG = B.FORBIDDEN_CONFIG
RESOURCE_SHA256 = B.RESOURCE_SHA256
PREFIX_DESTINATIONS = B.PREFIX_DESTINATIONS
OUTPUT_BLOCKS = B.OUTPUT_BLOCKS
EXPECTED_COMMITS_PER_CONFIG = B.EXPECTED_COMMITS_PER_CONFIG
REQUEST_FIELDS = B.REQUEST_FIELDS
PREFIX_EXACT_FIELDS = B.PREFIX_EXACT_FIELDS


def bound_call(function, *arguments):
    try:
        return function(*arguments)
    except B.M1619Error as error:
        raise M1628Error(str(error))


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def update_digest(digest, value):
    encoded = canonical_bytes(value)
    digest.update(str(len(encoded)).encode("ascii") + b":" + encoded)


def queue_index(kind, bank):
    kind_id = bound_call(B.C.kind_index, kind)
    bound_call(B.C.validate_bank, kind_id, bank)
    if kind_id in (2, 3):
        return int(bank)
    if kind_id in (4, 5):
        return 8 + int(bank)
    if kind_id in (0, 1, 7):
        return 14
    require(kind_id == 6, "unmapped outstanding queue")
    return 15


def port_index(kind, bank):
    return bound_call(B.C.port_index_for, bound_call(B.C.kind_index, kind),
                      bank)


def commit_record(row):
    return {"schema_version": row["schema_version"],
            "module": row["module"], "timestep": row["timestep"],
            "destination": row["destination"],
            "output_block": row["output_block"],
            "group": row["group"], "subordinal": row["subordinal"],
            "kind": row["kind"], "width_bytes": row["width_bytes"],
            "addresses": list(row["addresses"]),
            "banks": list(row["banks"])}


class _FinishReceipt(object):
    __slots__ = ("_payload_json", "_tag", "_locked")

    def __init__(self, payload_json, tag):
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, _name, _value):
        raise M1628Error("finish receipt is immutable")

    def as_dict(self):
        return json.loads(self._payload_json)


def _build_session_gate():
    key = os.urandom(32)
    issued = {}
    pending = {}

    def new_session(owner):
        require(type(owner) is CanonicalPrefixMiter,
                "session owner must be an exact canonical miter")
        while True:
            identity = os.urandom(32).hex()
            if identity not in issued and identity not in pending:
                issued[identity] = owner
                return identity

    def finish_session(owner, payload):
        require(type(owner) is CanonicalPrefixMiter and owner.finished and
                owner.next_destination == PREFIX_DESTINATIONS and
                owner.next_request_ordinal > 0 and
                payload == owner._finish_payload(),
                "finish sealer requires the exact completed owner state")
        identity = payload.get("session_identity")
        require(issued.get(identity) is owner,
                "finish session identity/owner drift")
        payload_json = canonical_bytes(payload).decode("utf-8")
        tag = hmac.new(key, payload_json.encode("utf-8"),
                       hashlib.sha256).hexdigest()
        receipt = _FinishReceipt(payload_json, tag)
        del issued[identity]
        pending[identity] = (receipt, tag)
        return receipt

    def inspect_receipts(rows):
        require(type(rows) is list and len(rows) == len(CONFIGS),
                "L2 bundle requires exactly three finish receipts")
        values = []
        identities = []
        for receipt in rows:
            require(type(receipt) is _FinishReceipt,
                    "bundle row is not a genuine finish receipt")
            value = receipt.as_dict()
            identity = value.get("session_identity")
            entry = pending.get(identity)
            require(entry is not None and entry[0] is receipt and
                    hmac.compare_digest(entry[1], receipt._tag) and
                    hmac.compare_digest(
                        hmac.new(key, receipt._payload_json.encode("utf-8"),
                                 hashlib.sha256).hexdigest(), receipt._tag),
                    "finish receipt authenticity/freshness failed")
            values.append(value)
            identities.append(identity)
        require(len(set(identities)) == len(CONFIGS),
                "finish sessions are not distinct")
        return values, identities

    def consume(identities):
        for identity in identities:
            require(identity in pending, "finish session was already consumed")
        for identity in identities:
            del pending[identity]

    return new_session, finish_session, inspect_receipts, consume


(_new_session, _finish_session, _inspect_finish_receipts,
 _consume_finish_receipts) = _build_session_gate()


class CanonicalPrefixMiter(object):
    """Request-authoritative cumulative prefix validator."""
    __slots__ = (
        "configuration", "session_identity", "next_destination",
        "next_request_ordinal", "previous_request_count",
        "previous_last_cycle", "maximum_issued_return", "queue_returns",
        "derived_port_calendar", "derived_last_psum_write_ready",
        "kind_counts", "byte_counts", "address_digest", "commit_digest",
        "cache_request_digest", "previous_cache_request_digest",
        "commit_pairs", "previous_cache", "source_ready_cycle",
        "control_ready_cycle", "previous_shared_1rw",
        "previous_outstanding_full_waits", "dense_cache_covered",
        "dense_psum_1rw_covered", "last_state_digest", "final_cache_digest",
        "finished")

    def __init__(self, configuration):
        require(configuration in CONFIGS and configuration != FORBIDDEN_CONFIG,
                "configuration is not admitted")
        self.configuration = configuration
        self.session_identity = _new_session(self)
        self.next_destination = 0
        self.next_request_ordinal = 0
        self.previous_request_count = 0
        self.previous_last_cycle = -1
        self.maximum_issued_return = -1
        self.queue_returns = [[] for _index in range(16)]
        self.derived_port_calendar = [0] * 24
        self.derived_last_psum_write_ready = [0] * OUTPUT_BLOCKS
        self.kind_counts = dict((name, 0) for name in B.C.KIND_NAMES)
        self.byte_counts = dict((name, 0) for name in B.C.KIND_NAMES)
        self.address_digest = hashlib.sha256()
        self.commit_digest = hashlib.sha256()
        self.cache_request_digest = hashlib.sha256()
        self.previous_cache_request_digest = hashlib.sha256(b"").hexdigest()
        self.commit_pairs = set()
        self.previous_cache = None
        self.source_ready_cycle = None
        self.control_ready_cycle = None
        self.previous_shared_1rw = 0
        self.previous_outstanding_full_waits = 0
        self.dense_cache_covered = False
        self.dense_psum_1rw_covered = False
        self.last_state_digest = None
        self.final_cache_digest = None
        self.finished = False

    def _active_returns(self, last_cycle):
        return [sorted(value for value in queue if value > last_cycle)
                for queue in self.queue_returns]

    def _nonzero_counts(self, values):
        return dict((name, values[name]) for name in B.C.KIND_NAMES
                    if values[name])

    def accept_request_pair(self, reference, compact):
        require(not self.finished, "finished session cannot accept requests")
        bound_call(B.validate_request_receipt, reference, self.configuration,
                   self.next_request_ordinal)
        bound_call(B.validate_request_receipt, compact, self.configuration,
                   self.next_request_ordinal)
        require(all(reference[field] == compact[field]
                    for field in REQUEST_FIELDS),
                "request exact miter failed")
        require(reference["module"] == B.MODULE_ORDINAL and
                reference["timestep"] == B.TIMESTEP and
                reference["destination"] == self.next_destination and
                0 <= reference["output_block"] < OUTPUT_BLOCKS,
                "request is outside D0/module0/timestep0/current destination")
        require(len(set(reference["banks"])) == len(reference["banks"]),
                "one request repeats a bank")
        expected_port_ready = 0
        for bank in reference["banks"]:
            expected_port_ready = max(
                expected_port_ready,
                self.derived_port_calendar[port_index(reference["kind"], bank)])
        require(reference["port_ready_cycle"] == expected_port_ready,
                "request port-ready value is not derived from accepted history")
        for bank in reference["banks"]:
            queue = queue_index(reference["kind"], bank)
            self.queue_returns[queue].append(reference["return_cycle"])
            port = port_index(reference["kind"], bank)
            self.derived_port_calendar[port] = max(
                self.derived_port_calendar[port],
                reference["issue_cycle"] + max(1, reference["beats"]))
        self.maximum_issued_return = max(
            self.maximum_issued_return, reference["return_cycle"])
        kind = reference["kind"]
        self.kind_counts[kind] += 1
        self.byte_counts[kind] += (reference["width_bytes"] *
                                  len(reference["addresses"]))
        update_digest(self.address_digest, reference)
        if kind in ("weight_read", "weight_write"):
            update_digest(self.cache_request_digest, reference)
        if kind == "psum_write":
            block = reference["output_block"]
            self.derived_last_psum_write_ready[block] = max(
                self.derived_last_psum_write_ready[block],
                reference["return_cycle"])
        if kind == "commit":
            pair = (reference["destination"], reference["output_block"])
            require(pair not in self.commit_pairs,
                    "duplicate destination/output-block commit")
            self.commit_pairs.add(pair)
            update_digest(self.commit_digest, commit_record(reference))
        self.next_request_ordinal += 1

    def accept_destination_pair(self, reference, compact):
        require(not self.finished, "finished session cannot accept state")
        destination = self.next_destination
        require(destination < PREFIX_DESTINATIONS,
                "too many prefix destinations")
        bound_call(B.validate_prefix_state, reference, self.configuration,
                   destination)
        bound_call(B.validate_prefix_state, compact, self.configuration,
                   destination)
        require(all(reference[field] == compact[field]
                    for field in PREFIX_EXACT_FIELDS),
                "destination exact miter failed")
        require(reference["request_count"] == self.next_request_ordinal and
                reference["request_count"] > self.previous_request_count and
                reference["last_cycle"] >= self.previous_last_cycle,
                "destination request/cycle history reset or skipped")
        require(reference["kind_counts"] ==
                    self._nonzero_counts(self.kind_counts) and
                reference["byte_counts"] ==
                    self._nonzero_counts(self.byte_counts) and
                reference["packed_transaction_address_sha256"] ==
                    self.address_digest.hexdigest() and
                reference["packed_commit_sequence_sha256"] ==
                    self.commit_digest.hexdigest(),
                "destination ledger is not derived from accepted requests")
        expected_commits = set((destination, block)
                               for block in range(OUTPUT_BLOCKS))
        require(expected_commits.issubset(self.commit_pairs) and
                reference["commit_count"] == len(self.commit_pairs) ==
                    (destination + 1) * OUTPUT_BLOCKS,
                "destination commit population is not dense/derived")
        require(reference["outstanding_active_returns"] ==
                    self._active_returns(reference["last_cycle"]),
                "active outstanding returns are not request-derived")
        require(reference["next_port_calendar"] ==
                    self.derived_port_calendar,
                "port calendar is not request-derived")
        dependency = reference["numeric_dependency_state"]
        require(dependency["last_psum_write_ready"] ==
                    self.derived_last_psum_write_ready,
                "psum-write readiness is not request-derived/nondecreasing")
        if self.source_ready_cycle is None:
            self.source_ready_cycle = dependency["source_ready_cycle"]
            self.control_ready_cycle = dependency[
                "persistent_control_ready_cycle"]
        require(dependency["source_ready_cycle"] == self.source_ready_cycle and
                dependency["persistent_control_ready_cycle"] ==
                    self.control_ready_cycle,
                "persistent source/control dependency changed")
        cache = reference["cache"]
        require(cache.get("accepted_weight_request_sha256") ==
                    self.cache_request_digest.hexdigest(),
                "cache transition is not bound to accepted weight requests")
        if self.previous_cache is not None:
            prior = self.previous_cache
            require(cache.get("previous_state_sha256") ==
                        prior["state_sha256"],
                    "cache transition predecessor identity drift")
            require(all(cache[field] >= prior[field] for field in
                        ("tick", "hits", "misses", "evictions")),
                    "cache counters moved backwards")
            advanced = any(cache[field] > prior[field] for field in
                           ("tick", "hits", "misses", "evictions"))
            require(cache["valid_entries"] >= prior["valid_entries"],
                    "cache valid content was cleared")
            if not advanced:
                require(cache["valid_entries"] == prior["valid_entries"] and
                        cache["state_sha256"] == prior["state_sha256"],
                        "cache content changed without cache activity")
            if cache["state_sha256"] != prior["state_sha256"]:
                require(advanced and
                        cache["accepted_weight_request_sha256"] !=
                            self.previous_cache_request_digest,
                        "cache content changed without an accepted weight request")
            if advanced and prior["valid_entries"]:
                require(cache["valid_entries"] > 0,
                        "active cache history vanished")
        else:
            require(cache.get("previous_state_sha256") ==
                        hashlib.sha256(b"").hexdigest(),
                    "first cache predecessor identity drift")
        coverage = reference["coverage_counters"]
        require(coverage["shared_1rw_serializations"] >=
                    self.previous_shared_1rw and
                coverage["outstanding_full_waits"] >=
                    self.previous_outstanding_full_waits,
                "coverage history moved backwards")
        if (self.configuration == CONFIGS[0] and cache["hits"] > 0 and
                cache["misses"] > 0 and cache["evictions"] > 0 and
                cache["valid_entries"] > 0):
            self.dense_cache_covered = True
        if (self.configuration == CONFIGS[0] and
                coverage["shared_1rw_serializations"] > 0):
            self.dense_psum_1rw_covered = True
        state_binding = dict((field, reference[field])
                             for field in PREFIX_EXACT_FIELDS)
        state_binding["maximum_issued_return"] = self.maximum_issued_return
        self.last_state_digest = hashlib.sha256(
            canonical_bytes(state_binding)).hexdigest()
        self.final_cache_digest = cache["state_sha256"]
        self.previous_request_count = self.next_request_ordinal
        self.previous_last_cycle = reference["last_cycle"]
        self.previous_cache = json.loads(json.dumps(cache))
        self.previous_cache_request_digest = self.cache_request_digest.hexdigest()
        self.previous_shared_1rw = coverage["shared_1rw_serializations"]
        self.previous_outstanding_full_waits = coverage[
            "outstanding_full_waits"]
        self.next_destination += 1

    def _finish_payload(self):
        require(self.next_destination == PREFIX_DESTINATIONS and
                self.previous_request_count == self.next_request_ordinal and
                self.next_request_ordinal > 0 and
                len(self.commit_pairs) == EXPECTED_COMMITS_PER_CONFIG,
                "canonical prefix is incomplete or empty")
        if self.configuration == CONFIGS[0]:
            require(self.dense_cache_covered and self.dense_psum_1rw_covered,
                    "dense session lacks cache/1RW coverage")
        ledger = {"request_count": self.next_request_ordinal,
                  "kind_counts": self._nonzero_counts(self.kind_counts),
                  "byte_counts": self._nonzero_counts(self.byte_counts),
                  "address_digest": self.address_digest.hexdigest(),
                  "commit_digest": self.commit_digest.hexdigest(),
                  "port_calendar": self.derived_port_calendar,
                  "last_psum_write_ready":
                      self.derived_last_psum_write_ready,
                  "maximum_issued_return": self.maximum_issued_return,
                  "last_cycle": self.previous_last_cycle,
                  "last_state_digest": self.last_state_digest,
                  "final_cache_digest": self.final_cache_digest}
        return {"schema": SCHEMA,
                "session_identity": self.session_identity,
                "configuration": self.configuration,
                "resource_manifest_sha256": RESOURCE_SHA256,
                "destinations": self.next_destination,
                "requests": self.next_request_ordinal,
                "commits": len(self.commit_pairs),
                "dense_cache_covered": self.dense_cache_covered,
                "dense_psum_1rw_covered": self.dense_psum_1rw_covered,
                "final_commit_digest": self.commit_digest.hexdigest(),
                "final_state_sha256": hashlib.sha256(
                    canonical_bytes(ledger)).hexdigest()}

    def finish(self):
        require(not self.finished, "session finish is one-shot")
        payload = self._finish_payload()
        self.finished = True
        return _finish_session(self, payload)


def validate_bundle(rows):
    values, identities = _inspect_finish_receipts(rows)
    require([row.get("configuration") for row in values] == list(CONFIGS),
            "L2 bundle configuration order drift")
    require(all(row.get("schema") == SCHEMA and
                row.get("resource_manifest_sha256") == RESOURCE_SHA256 and
                row.get("destinations") == PREFIX_DESTINATIONS and
                type(row.get("requests")) is int and row["requests"] > 0 and
                row.get("commits") == EXPECTED_COMMITS_PER_CONFIG and
                type(row.get("final_state_sha256")) is str and
                len(row["final_state_sha256"]) == 64
                for row in values),
            "L2 bundle resource/population/final-state drift")
    require(values[0].get("dense_cache_covered") is True and
            values[0].get("dense_psum_1rw_covered") is True,
            "dense configuration coverage missing")
    require(len(set(row.get("final_commit_digest") for row in values)) == 1,
            "cross-configuration commit stream drift")
    _consume_finish_receipts(identities)
    return True


def synthetic_request(configuration, ordinal, kind="commit"):
    destination = ordinal // OUTPUT_BLOCKS
    output_block = ordinal % OUTPUT_BLOCKS
    kind_id = B.C.kind_index(kind)
    bank = 0
    return {"configuration": configuration, "schema_version": 1,
            "module": B.MODULE_ORDINAL, "timestep": B.TIMESTEP,
            "destination": destination, "output_block": output_block,
            "group": 0, "subordinal": 0, "request_ordinal": ordinal,
            "kind": kind, "earliest_issue_cycle": ordinal,
            "dependency_ready_cycle": ordinal,
            "port_ready_cycle": 0, "issue_cycle": ordinal,
            "beats": 2, "return_cycle": ordinal + B.C.latency_for(kind_id) + 1,
            "width_bytes": 384,
            "addresses": [(4 << 60) + ordinal * 384], "banks": [bank],
            "packed_event_sha256": hashlib.sha256(
                ("event:" + configuration + ":" + str(ordinal)).encode(
                    "ascii")).hexdigest()}


def synthetic_state(miter, dense, last_cycle=None):
    destination = miter.next_destination
    if last_cycle is None:
        last_cycle = 1000 + miter.next_request_ordinal
    tick = destination + 1
    cache_state = hashlib.sha256(
        ("cache:" + miter.configuration + ":" + str(destination)).encode(
            "ascii")).hexdigest()
    state = {"configuration": miter.configuration,
             "destination": destination, "last_cycle": last_cycle,
             "request_count": miter.next_request_ordinal,
             "kind_counts": miter._nonzero_counts(miter.kind_counts),
             "byte_counts": miter._nonzero_counts(miter.byte_counts),
             "packed_transaction_address_sha256":
                 miter.address_digest.hexdigest(),
             "packed_commit_sequence_sha256": miter.commit_digest.hexdigest(),
             "next_port_calendar": list(miter.derived_port_calendar),
             "outstanding_active_returns":
                 miter._active_returns(last_cycle),
             "numeric_dependency_state": {"source_ready_cycle": 1,
                 "persistent_control_ready_cycle": 2,
                 "last_psum_write_ready":
                     list(miter.derived_last_psum_write_ready)},
             "cache": {"valid_entries": 9,
                 "tick": tick, "hits": tick, "misses": tick,
                 "evictions": tick,
                 "state_sha256": cache_state,
                 "previous_state_sha256":
                     (miter.previous_cache["state_sha256"]
                      if miter.previous_cache is not None else
                      hashlib.sha256(b"").hexdigest()),
                 "accepted_weight_request_sha256":
                     miter.cache_request_digest.hexdigest()},
             "coverage_counters": {"outstanding_full_waits": 0,
                 "shared_1rw_serializations": tick},
             "commit_count": len(miter.commit_pairs), "reset_count": 0,
             "resource_manifest_sha256": RESOURCE_SHA256,
             "rss": {"baseline_rss_kib": 100000,
                 "current_rss_kib": 100000 + destination,
                 "hwm_rss_kib": 100100 + destination}}
    return state


def build_synthetic_session(configuration, address_bias=0):
    miter = CanonicalPrefixMiter(configuration)
    dense = configuration == CONFIGS[0]
    for destination in range(PREFIX_DESTINATIONS):
        ordinal = miter.next_request_ordinal
        weight = synthetic_request(configuration, ordinal, "weight_read")
        weight["destination"] = destination
        weight["output_block"] = 0
        weight["width_bytes"] = 16
        weight["addresses"] = [destination % 128]
        weight["banks"] = [destination % 8]
        weight["port_ready_cycle"] = max(
            miter.derived_port_calendar[port_index(
                weight["kind"], bank)] for bank in weight["banks"])
        weight["issue_cycle"] = max(
            weight["earliest_issue_cycle"],
            weight["dependency_ready_cycle"],
            weight["port_ready_cycle"])
        weight["return_cycle"] = (weight["issue_cycle"] +
                                  B.C.latency_for(B.C.kind_index(
                                      weight["kind"])) +
                                  weight["beats"] - 1)
        miter.accept_request_pair(weight, json.loads(json.dumps(weight)))
        for output_block in range(OUTPUT_BLOCKS):
            ordinal = miter.next_request_ordinal
            request = synthetic_request(configuration, ordinal)
            request["destination"] = destination
            request["output_block"] = output_block
            request["port_ready_cycle"] = max(
                miter.derived_port_calendar[port_index(
                    request["kind"], bank)] for bank in request["banks"])
            request["issue_cycle"] = max(
                request["earliest_issue_cycle"],
                request["dependency_ready_cycle"],
                request["port_ready_cycle"])
            request["return_cycle"] = (request["issue_cycle"] +
                                       B.C.latency_for(B.C.kind_index(
                                           request["kind"])) +
                                       request["beats"] - 1)
            if address_bias:
                request["addresses"] = [value + address_bias
                                        for value in request["addresses"]]
            miter.accept_request_pair(request,
                                      json.loads(json.dumps(request)))
        state = synthetic_state(miter, dense)
        miter.accept_destination_pair(state, json.loads(json.dumps(state)))
    return miter.finish()


def static_self_test():
    require(not FUTURE_REVIEW.exists() and not FUTURE_RELEASE.exists(),
            "future review/release must remain absent")
    rows = [build_synthetic_session(configuration)
            for configuration in CONFIGS]
    summaries = [row.as_dict() for row in rows]
    validate_bundle(rows)
    attacks = 0
    try:
        validate_bundle([dict(value) for value in summaries])
    except M1628Error:
        attacks += 1
    try:
        actual_prefix_release()
    except M1628Error:
        attacks += 1
    require(attacks == 2, "built-in source attacks were not rejected")
    return {"schema": SCHEMA,
            "status": "PASS_M1628_THREE_P1_SOURCE_REPAIR_STATIC_ONLY",
            "sessions": summaries, "attacks_rejected": attacks,
            "actual_payload": False, "l2_executed": False,
            "l3_executed": False, "gpu": False, "eda": False,
            "paper_result": False}


def actual_prefix_release(_provider=None, _token=None):
    raise M1628Error(
        "M1628 is source-only; M1629 review and M1633 release are absent")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "repair": {"request_authoritative_ledgers": True,
                "max_return_and_active_projection_derived": True,
                "psum_ready_nondecreasing_and_derived": True,
                "cache_clear_transition_rejected": True,
                "authenticated_one_time_finish_receipt": True,
                "three_distinct_fresh_sessions": True},
            "future_gate": {"review": str(FUTURE_REVIEW.relative_to(HW)),
                "release": str(FUTURE_RELEASE.relative_to(HW)),
                "review_present": FUTURE_REVIEW.exists(),
                "release_present": FUTURE_RELEASE.exists()},
            "bindings": {"m1619_source_sha256": M1619_SOURCE_SHA256,
                "m1620_review_sha256": M1620_REVIEW_SHA256,
                "resource_manifest_sha256": RESOURCE_SHA256,
                "docs359_sha256": DOCS359_SHA256},
            "authorization": {"source_only": True,
                "different_author_review": True,
                "actual_payload": False, "l2_execution": False,
                "l3": False, "pilot": False, "production": False,
                "cycles": False, "traffic": False, "energy": False,
                "speedup": False, "rtl": False, "eda": False,
                "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--static-self-test", action="store_true")
    args = parser.parse_args(argv)
    value = describe() if args.describe else static_self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

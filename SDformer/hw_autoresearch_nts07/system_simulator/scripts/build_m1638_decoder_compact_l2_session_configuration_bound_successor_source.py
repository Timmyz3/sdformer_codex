#!/usr/bin/env python3
"""M1638 source-only repair of M1629's session-configuration relabel P1.

The exact M1628 request/state/cache/ledger implementation is inherited.  This
successor rebuilds only session construction, finish authentication and bundle
coverage policy.  A hidden registry binds each exact owner to its immutable
initial configuration and checks it at request, state, payload and finish.

No actual ep34 payload is named or opened and no actual L2/L3 run is exposed.
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
M1628_SOURCE = HERE / "build_m1628_decoder_compact_l2_retained_ledger_successor_source.py"
M1628_SOURCE_SHA256 = "41bae4c11484c4bf3e5da9537225372d703c2f7206879ecf7814bc52c56c0df4"
M1629 = HW / (
    "reviews/m1629_m1628_decoder_compact_l2_retained_ledger_source_"
    "independent_review_r1_20260901")
M1629_REVIEW_SHA256 = "87493d5ad24a230ca5ce17bd6b8ab9177e0161aa98a407f5c5559ad46284f01e"
M1629_MANIFEST_SHA256 = "fbde5fe643ee0d0b85671e4f7a0ce50b01619f9b559ec121d081f897cadab8ca"
M1629_OUTER_FILE_SHA256 = "15387330c2457bfc32b5c165fad7f6f9592f0a2fdbfc6733414def9e8427fc65"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1638_decoder_compact_l2_session_configuration_bound_successor_source_r1_v1"
STATUS = "SOURCE_ONLY__M1629_CONFIGURATION_RELABEL_P1_REPAIRED__NO_PAYLOAD_NO_EXECUTION"
FUTURE_REVIEW = HW / (
    "reviews/m1639_m1638_decoder_compact_l2_session_configuration_bound_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1640_m1639_m1638_decoder_compact_l2_actual_prefix_"
    "source_release_r1_20260901.json")


def require(value, message):
    if not value:
        raise M1638Error(message)


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


def load_m1628():
    regular_exact(M1628_SOURCE, M1628_SOURCE_SHA256, "M1628 source")
    spec = importlib.util.spec_from_file_location("m1638_bound_m1628",
                                                  str(M1628_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1628")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1628_decoder_compact_l2_retained_ledger_successor_source_r1_v1" and
            module.RESOURCE_SHA256 ==
            "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10",
            "M1628 semantic boundary drift")
    return module


regular_exact(DOC359, DOC359_SHA256, "protected docs359")
regular_exact(M1629 / "review.json", M1629_REVIEW_SHA256, "M1629 review")
regular_exact(M1629 / "SHA256SUMS", M1629_MANIFEST_SHA256, "M1629 manifest")
regular_exact(M1629 / "SHA256SUMS.seal.sha256", M1629_OUTER_FILE_SHA256,
              "M1629 outer seal")
require((M1629 / "SHA256SUMS.seal.sha256").read_text(
            encoding="ascii").split() == [M1629_MANIFEST_SHA256, "SHA256SUMS"],
        "M1629 outer content drift")
_m1629_review = json.loads((M1629 / "review.json").read_text(encoding="utf-8"))
require(_m1629_review.get("status") ==
        "NO_GO_M1628_ACTUAL_L2_RUNNER_SOURCE__ONE_P1_SESSION_CONFIGURATION_BINDING_GAP__SUCCESSOR_REPAIR_ONLY" and
        [_item.get("id") for _item in _m1629_review.get("findings", {}).get(
            "p1", [])] == ["P1_SESSION_CONFIGURATION_RELABEL_NOT_BOUND_AT_CREATION"],
        "M1629 finding boundary drift")

P = load_m1628()
M1638Error = P.M1628Error
M1628Error = M1638Error
B = P.B
CONFIGS = P.CONFIGS
FORBIDDEN_CONFIG = P.FORBIDDEN_CONFIG
RESOURCE_SHA256 = P.RESOURCE_SHA256
PREFIX_DESTINATIONS = P.PREFIX_DESTINATIONS
OUTPUT_BLOCKS = P.OUTPUT_BLOCKS
EXPECTED_COMMITS_PER_CONFIG = P.EXPECTED_COMMITS_PER_CONFIG
REQUEST_FIELDS = P.REQUEST_FIELDS
PREFIX_EXACT_FIELDS = P.PREFIX_EXACT_FIELDS
queue_index = P.queue_index
port_index = P.port_index
canonical_bytes = P.canonical_bytes
synthetic_request = P.synthetic_request


class _FinishReceipt(object):
    __slots__ = ("_payload_json", "_tag", "_locked")

    def __init__(self, payload_json, tag):
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, _name, _value):
        raise M1638Error("finish receipt is immutable")

    def as_dict(self):
        return json.loads(self._payload_json)


def _build_session_gate():
    key = os.urandom(32)
    issued = {}
    pending = {}

    def new_session(owner, initial_configuration=None):
        require(type(owner) is CanonicalPrefixMiter and
                initial_configuration in CONFIGS and
                initial_configuration != FORBIDDEN_CONFIG,
                "session owner/configuration must be exact and admitted")
        while True:
            identity = os.urandom(32).hex()
            if identity not in issued and identity not in pending:
                issued[identity] = (owner, initial_configuration)
                return identity

    def assert_session(owner):
        require(type(owner) is CanonicalPrefixMiter,
                "session owner must be an exact canonical miter")
        entry = issued.get(owner.session_identity)
        require(entry is not None and entry[0] is owner,
                "session identity/owner drift")
        initial_configuration = entry[1]
        require(owner.configuration == initial_configuration,
                "session configuration differs from immutable initial binding")
        return initial_configuration

    def finish_session(owner, payload):
        initial_configuration = assert_session(owner)
        require(owner.finished and
                owner.next_destination == PREFIX_DESTINATIONS and
                owner.next_request_ordinal > 0 and
                payload == owner._finish_payload() and
                payload.get("configuration") == initial_configuration,
                "finish sealer requires exact completed initial configuration")
        identity = payload.get("session_identity")
        payload_json = canonical_bytes(payload).decode("utf-8")
        tag = hmac.new(key, payload_json.encode("utf-8"),
                       hashlib.sha256).hexdigest()
        receipt = _FinishReceipt(payload_json, tag)
        del issued[identity]
        pending[identity] = (receipt, tag, initial_configuration)
        return receipt

    def inspect_receipts(rows):
        require(type(rows) is list and len(rows) == len(CONFIGS),
                "L2 bundle requires exactly three finish receipts")
        values = []
        identities = []
        initial_configurations = []
        for receipt in rows:
            require(type(receipt) is _FinishReceipt,
                    "bundle row is not a genuine finish receipt")
            value = receipt.as_dict()
            identity = value.get("session_identity")
            entry = pending.get(identity)
            require(entry is not None and entry[0] is receipt and
                    value.get("configuration") == entry[2] and
                    hmac.compare_digest(entry[1], receipt._tag) and
                    hmac.compare_digest(
                        hmac.new(key, receipt._payload_json.encode("utf-8"),
                                 hashlib.sha256).hexdigest(), receipt._tag),
                    "finish receipt authenticity/configuration/freshness failed")
            values.append(value)
            identities.append(identity)
            initial_configurations.append(entry[2])
        require(len(set(identities)) == len(CONFIGS),
                "finish sessions are not distinct")
        require(initial_configurations == list(CONFIGS),
                "initial session configuration order drift")
        return values, identities

    def consume(identities):
        for identity in identities:
            require(identity in pending, "finish session was already consumed")
        for identity in identities:
            del pending[identity]

    return new_session, assert_session, finish_session, inspect_receipts, consume


(_new_session, _assert_session, _finish_session, _inspect_finish_receipts,
 _consume_finish_receipts) = _build_session_gate()


class CanonicalPrefixMiter(P.CanonicalPrefixMiter):
    """Exact M1628 miter with immutable hidden initial-configuration binding."""
    __slots__ = ()

    def __init__(self, configuration):
        require(configuration in CONFIGS and configuration != FORBIDDEN_CONFIG,
                "configuration is not admitted")
        self.configuration = configuration
        self.session_identity = _new_session(self, configuration)
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

    def accept_request_pair(self, reference, compact):
        initial_configuration = _assert_session(self)
        require(reference.get("configuration") == initial_configuration and
                compact.get("configuration") == initial_configuration,
                "request configuration differs from initial session binding")
        P.CanonicalPrefixMiter.accept_request_pair(self, reference, compact)
        _assert_session(self)

    def accept_destination_pair(self, reference, compact):
        initial_configuration = _assert_session(self)
        require(reference.get("configuration") == initial_configuration and
                compact.get("configuration") == initial_configuration,
                "state configuration differs from initial session binding")
        P.CanonicalPrefixMiter.accept_destination_pair(self, reference, compact)
        _assert_session(self)

    def _finish_payload(self):
        initial_configuration = _assert_session(self)
        payload = P.CanonicalPrefixMiter._finish_payload(self)
        require(payload.get("configuration") == initial_configuration,
                "finish payload configuration differs from initial binding")
        payload["schema"] = SCHEMA
        return payload

    def finish(self):
        _assert_session(self)
        require(not self.finished, "session finish is one-shot")
        payload = self._finish_payload()
        self.finished = True
        return _finish_session(self, payload)


def synthetic_state(miter, dense, last_cycle=None):
    _assert_session(miter)
    state = P.synthetic_state(miter, dense, last_cycle)
    require(state.get("configuration") == miter.configuration,
            "synthetic state configuration drift")
    return state


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
    coverage_policy = [
        (row.get("dense_cache_covered"), row.get("dense_psum_1rw_covered"))
        for row in values]
    require(coverage_policy == [(True, True), (False, False), (False, False)],
            "per-configuration dense coverage policy drift")
    require(len(set(row.get("final_commit_digest") for row in values)) == 1,
            "cross-configuration commit stream drift")
    _consume_finish_receipts(identities)
    return True


def build_synthetic_miter(configuration, address_bias=0):
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
        weight["issue_cycle"] = max(weight["earliest_issue_cycle"],
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
            request["issue_cycle"] = max(request["earliest_issue_cycle"],
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
    return miter


def build_synthetic_session(configuration, address_bias=0):
    return build_synthetic_miter(configuration, address_bias).finish()


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
    except M1638Error:
        attacks += 1
    for target in CONFIGS[1:]:
        relabelled = build_synthetic_miter(CONFIGS[0])
        relabelled.configuration = target
        try:
            relabelled.finish()
        except M1638Error:
            attacks += 1
    try:
        actual_prefix_release()
    except M1638Error:
        attacks += 1
    require(attacks == 4, "built-in source attacks were not rejected")
    return {"schema": SCHEMA,
            "status": "PASS_M1638_CONFIGURATION_BOUND_SOURCE_STATIC_ONLY",
            "sessions": summaries, "attacks_rejected": attacks,
            "configuration_relabel_rejected": 2,
            "actual_payload": False, "l2_executed": False,
            "l3_executed": False, "gpu": False, "eda": False,
            "paper_result": False}


def actual_prefix_release(_provider=None, _token=None):
    raise M1638Error(
        "M1638 is source-only; M1639 review and M1640 release are absent")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "repair": {"m1628_behavior_inherited": True,
                "hidden_initial_configuration_bound": True,
                "request_state_payload_finish_all_check_binding": True,
                "exact_bundle_coverage_policy": [
                    [True, True], [False, False], [False, False]],
                "three_dense_session_relabel_rejected": True},
            "future_gate": {"review": str(FUTURE_REVIEW.relative_to(HW)),
                "release": str(FUTURE_RELEASE.relative_to(HW)),
                "review_present": FUTURE_REVIEW.exists(),
                "release_present": FUTURE_RELEASE.exists()},
            "bindings": {"m1628_source_sha256": M1628_SOURCE_SHA256,
                "m1629_review_sha256": M1629_REVIEW_SHA256,
                "resource_manifest_sha256": RESOURCE_SHA256,
                "docs359_sha256": DOC359_SHA256},
            "authorization": {"source_only": True,
                "different_author_review": True,
                "actual_runner_source": False, "actual_payload": False,
                "l2_execution": False, "l3": False, "pilot": False,
                "production": False, "cycles": False, "traffic": False,
                "energy": False, "speedup": False, "rtl": False,
                "eda": False, "paper_result": False}}


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

#!/usr/bin/env python3
"""M1639 payload-free hammer for M1638's session-configuration repair."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys


sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1638_decoder_compact_l2_session_configuration_bound_successor_source.py"
TEST = HW / "system_simulator/tests/test_m1638_decoder_compact_l2_session_configuration_bound_successor_source.py"
CONTRACT = HW / "contracts/m1638_decoder_compact_l2_session_configuration_bound_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1638_decoder_compact_l2_session_configuration_bound_successor_source_author_receipt_r1_20260901"
M1628_SOURCE = HW / "system_simulator/scripts/build_m1628_decoder_compact_l2_retained_ledger_successor_source.py"
M1629 = HW / "reviews/m1629_m1628_decoder_compact_l2_retained_ledger_source_independent_review_r1_20260901"
M1629_HAMMER = M1629 / "independent_hammer.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_RELEASE = HW / "contracts/m1640_m1639_m1638_decoder_compact_l2_actual_prefix_source_release_r1_20260901.json"

EXPECTED = {
    SOURCE: "1b3961b0d0682980a035f5ad9ba880eb44929e56116f23f2e68cbb9e0a3fdecd",
    TEST: "2d3f222a9cf843e1d16d0547c63ad5e9c7a9bcb245236aa484bd097a0b36afcc",
    CONTRACT: "81b283f3f1a0127be3994dda926b66d921889380f2fd5f79bacca46cfaeb5cca",
    Path(str(CONTRACT) + ".sha256"): "c03db224441c1309bac9385a72e09212e8c9c223b7b6d20923e6291a9a968a01",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "3790ea98d85bc572871e323a3d36564c32ba135b0ce010299ac6cceb7b83879c",
    AUTHOR / "review.json": "ac2c375be73f4b0bbe9b7fa0d5d8418cc04c9b65e63302d4b2a965a2dcdf0f41",
    AUTHOR / "SHA256SUMS": "54f9759d86aac5878e1d24064ef55ad5139a1d2afc0d89c522343b3850830113",
    AUTHOR / "SHA256SUMS.seal.sha256": "1e0a62d02cb6840346f58e142beaa8fcd0f08de0c1961bb86297f34e70807d51",
    M1628_SOURCE: "41bae4c11484c4bf3e5da9537225372d703c2f7206879ecf7814bc52c56c0df4",
    M1629 / "review.json": "87493d5ad24a230ca5ce17bd6b8ab9177e0161aa98a407f5c5559ad46284f01e",
    M1629 / "SHA256SUMS": "fbde5fe643ee0d0b85671e4f7a0ce50b01619f9b559ec121d081f897cadab8ca",
    M1629 / "SHA256SUMS.seal.sha256": "15387330c2457bfc32b5c165fad7f6f9592f0a2fdbfc6733414def9e8427fc65",
    M1629_HAMMER: "641cf32a046f9b2c363e897ffe0b6032271aa0a14a01bdb0fbd573fca4a99ef8",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_COVERAGE = [(True, True), (False, False), (False, False)]


class Failure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("non-finite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_regular(path, digest):
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))
    require(sha(path) == digest, "identity drift: " + str(path))


def verify_file_seal(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii") ==
            sha(path) + "  " + path.name + "\n", "file inner seal mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha(sidecar) + "  " + sidecar.name + "\n", "file outer seal mismatch")


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "sealed tree absent")
    require(outer.read_text(encoding="ascii") ==
            sha(manifest) + "  SHA256SUMS\n", "tree outer seal mismatch")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        require(re.match(r"^[0-9a-f]{64}  (?:\./)?[^/\n][^\n]*$", row) is not None,
                "malformed tree manifest")
        digest, raw_name = row.split("  ", 1)
        name = raw_name[2:] if raw_name.startswith("./") else raw_name
        require(name not in listed and not Path(name).is_absolute() and
                all(part not in ("", ".", "..") for part in Path(name).parts),
                "unsafe/duplicate tree member")
        listed[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        for name in list(dirs) + list(files):
            path = Path(base) / name
            require(not path.is_symlink(), "symlink in sealed tree")
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(listed), "sealed tree topology drift")
    for name, digest in listed.items():
        verify_regular(root / name, digest)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(module, action):
    try:
        action()
    except module.M1638Error:
        return True
    return False


class Driver(object):
    def __init__(self, module):
        self.m = module

    def request(self, miter, kind, output_block, mutation=None):
        m = self.m
        ordinal = miter.next_request_ordinal
        row = m.synthetic_request(miter.configuration, ordinal, kind)
        row["destination"] = miter.next_destination
        row["output_block"] = output_block
        if kind in ("weight_read", "weight_write"):
            row["width_bytes"] = 16
            row["addresses"] = [miter.next_destination % 128]
            row["banks"] = [miter.next_destination % 8]
        elif kind in ("psum_read", "psum_write"):
            row["width_bytes"] = 48
            row["addresses"] = [output_block * 48]
            row["banks"] = [output_block % 6]
        row["port_ready_cycle"] = max(
            miter.derived_port_calendar[m.port_index(kind, bank)]
            for bank in row["banks"])
        row["issue_cycle"] = max(row["earliest_issue_cycle"],
                                 row["dependency_ready_cycle"],
                                 row["port_ready_cycle"])
        row["return_cycle"] = (row["issue_cycle"] +
                               m.B.C.latency_for(m.B.C.kind_index(kind)) +
                               row["beats"] - 1)
        if mutation is not None:
            mutation(row)
        miter.accept_request_pair(row, copy.deepcopy(row))
        return row

    def populate(self, miter, weight=True, psum=False, long_return=False):
        if weight:
            self.request(miter, "weight_read", 0)
        if psum:
            self.request(miter, "psum_write", 0)
        for block in range(self.m.OUTPUT_BLOCKS):
            mutation = None
            if long_return and block == 0:
                mutation = lambda row: row.update({"return_cycle": 5000})
            self.request(miter, "commit", block, mutation)

    def state(self, miter, last_cycle=None, mutation=None):
        row = self.m.synthetic_state(
            miter, miter.configuration == self.m.CONFIGS[0], last_cycle)
        if mutation is not None:
            mutation(row)
        miter.accept_destination_pair(row, copy.deepcopy(row))
        return row

    def completed(self, configuration, address_bias=0):
        miter = self.m.CanonicalPrefixMiter(configuration)
        for _destination in range(self.m.PREFIX_DESTINATIONS):
            self.request(miter, "weight_read", 0)
            for block in range(self.m.OUTPUT_BLOCKS):
                if address_bias:
                    self.request(miter, "commit", block,
                                 lambda row, bias=address_bias: row.update({
                                     "addresses": [value + bias
                                                   for value in row["addresses"]]}))
                else:
                    self.request(miter, "commit", block)
            self.state(miter)
        return miter


def audit_static_source(text, contract):
    for token in ("np.load", "numpy.load", "torch.load", ".npz", ".tar.zst",
                  "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
        require(token not in text, "payload token present: " + token)
    require("issued[identity] = (owner, initial_configuration)" in text and
            "pending[identity] = (receipt, tag, initial_configuration)" in text,
            "hidden owner/configuration binding absent")
    assert_block = text[text.index("    def assert_session(owner):"):
                        text.index("    def finish_session(owner, payload):")]
    require("entry[0] is owner" in assert_block and
            "initial_configuration = entry[1]" in assert_block and
            "owner.configuration == initial_configuration" in assert_block,
            "hidden session assertion drift")
    request = text[text.index("    def accept_request_pair(self, reference, compact):"):
                   text.index("    def accept_destination_pair(self, reference, compact):")]
    state = text[text.index("    def accept_destination_pair(self, reference, compact):"):
                 text.index("    def _finish_payload(self):")]
    payload = text[text.index("    def _finish_payload(self):"):
                   text.index("    def finish(self):")]
    finish = text[text.index("    def finish(self):"):
                  text.index("def synthetic_state(")]
    require(request.count("_assert_session(self)") == 2 and
            'reference.get("configuration") == initial_configuration' in request and
            'compact.get("configuration") == initial_configuration' in request,
            "request binding before/after drift")
    require(state.count("_assert_session(self)") == 2 and
            'reference.get("configuration") == initial_configuration' in state and
            'compact.get("configuration") == initial_configuration' in state,
            "state binding before/after drift")
    require("_assert_session(self)" in payload and
            'payload.get("configuration") == initial_configuration' in payload,
            "payload binding drift")
    require("_assert_session(self)" in finish and
            "_finish_session(self, payload)" in finish,
            "finish binding drift")
    inspect = text[text.index("    def inspect_receipts(rows):"):
                   text.index("    def consume(identities):")]
    require("value.get(\"configuration\") == entry[2]" in inspect and
            "initial_configurations.append(entry[2])" in inspect and
            "initial_configurations == list(CONFIGS)" in inspect,
            "bundle initial-configuration authentication drift")
    bundle = text[text.index("def validate_bundle(rows):"):
                  text.index("def build_synthetic_miter(")]
    require("coverage_policy == [(True, True), (False, False), (False, False)]" in bundle,
            "exact configuration coverage policy drift")
    require(contract.get("p1_repair", {}).get("exact_coverage_policy") == [
        {"configuration": "DENSE_TYPED_K8", "dense_cache_covered": True,
         "dense_psum_1rw_covered": True},
        {"configuration": "BIT_EQUAL_SERVICE_K1X8", "dense_cache_covered": False,
         "dense_psum_1rw_covered": False},
        {"configuration": "BIT_TYPED_K8", "dense_cache_covered": False,
         "dense_psum_1rw_covered": False}], "contract coverage policy drift")
    auth = contract.get("authorization", {})
    require(auth.get("different_author_review") is True and
            all(auth.get(key) is False for key in (
                "actual_prefix_runner_source", "actual_payload", "l2_execution",
                "l3", "pilot", "production", "gpu", "eda",
                "attempt_creation", "release_creation")),
            "source-only authorization drift")
    claims = contract.get("claim_boundary", {})
    require(claims.get("source_only") is True and
            all(claims.get(key) is False for key in (
                "actual_payload", "l2_execution", "l3", "cycles", "traffic",
                "speedup", "system_speedup", "energy", "rtl", "eda", "ppa",
                "table_a", "paper_result")), "claim boundary drift")


def run_prior_attacks(m, prior):
    d = Driver(m)
    survivors = {
        "S1_EARLIER_MAX_RETURN_CANNOT_BE_DROPPED": prior.survivor_max_return(m, d),
        "S2_CROSS_DESTINATION_FUTURE_RETURN_CANNOT_DISAPPEAR": prior.survivor_cross_destination(m, d),
        "S3_LAST_PSUM_WRITE_READY_CANNOT_MOVE_BACKWARD": prior.survivor_psum_ready(m, d),
        "S4_CACHE_CLEAR_OR_CONTENT_RESET_CANNOT_PASS": prior.survivor_cache_continuity(m, d),
        "S5_REQUEST_SCOPE_MODULE_TIMESTEP_DESTINATION_BLOCK_IS_BOUND": prior.survivor_request_scope(m, d),
        "S6_KIND_AND_BYTE_LEDGER_IS_INTERNALLY_DERIVED": prior.survivor_count_bytes(m, d),
        "S7_ADDRESS_AND_COMMIT_DIGESTS_ARE_INTERNALLY_DERIVED": prior.survivor_digests(m, d),
        "S8_HAND_WRITTEN_FINISH_ROWS_ARE_REJECTED": prior.survivor_forged_finish(m, d),
    }
    require(all(survivors.values()), "M1628/M1620 survivor attack escaped")

    rows = [d.completed(configuration).finish() for configuration in m.CONFIGS]
    values = [row.as_dict() for row in rows]
    require([(row["dense_cache_covered"], row["dense_psum_1rw_covered"])
             for row in values] == EXPECTED_COVERAGE,
            "genuine exact coverage policy mismatch")
    clone = object.__new__(type(rows[0]))
    object.__setattr__(clone, "_payload_json", rows[0]._payload_json)
    object.__setattr__(clone, "_tag", rows[0]._tag)
    object.__setattr__(clone, "_locked", True)
    extra = {
        "receipt_clone": rejected(m, lambda: m.validate_bundle(
            [clone, rows[1], rows[2]])),
        "duplicate_session": rejected(m, lambda: m.validate_bundle(
            [rows[0], rows[0], rows[2]])),
        "configuration_reorder": rejected(m, lambda: m.validate_bundle(
            [rows[1], rows[0], rows[2]])),
    }
    require(all(extra.values()), "clone/duplicate/order attack escaped")
    require(m.validate_bundle(rows) is True, "genuine bundle rejected")
    extra["consumed_replay"] = rejected(m, lambda: m.validate_bundle(rows))

    tag_rows = [d.completed(configuration).finish() for configuration in m.CONFIGS]
    object.__setattr__(tag_rows[0], "_tag", "0" * 64)
    extra["tag_mutation"] = rejected(m, lambda: m.validate_bundle(tag_rows))
    stream_rows = [d.completed(configuration, 4096 if index == 2 else 0).finish()
                   for index, configuration in enumerate(m.CONFIGS)]
    extra["shared_commit_stream_mismatch"] = rejected(
        m, lambda: m.validate_bundle(stream_rows))
    require(all(extra.values()), "M1629 receipt/stream/replay attack escaped")
    return d, survivors, extra


def run_configuration_binding_attacks(m, d):
    outcomes = {}
    request_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
    request_miter.configuration = m.CONFIGS[1]
    outcomes["request_owner_relabel"] = rejected(
        m, lambda: d.request(request_miter, "commit", 0))

    state_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(state_miter)
    state = m.synthetic_state(state_miter, True)
    state_miter.configuration = m.CONFIGS[1]
    outcomes["state_owner_relabel"] = rejected(
        m, lambda: state_miter.accept_destination_pair(state, copy.deepcopy(state)))

    request_payload_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
    reference = m.synthetic_request(m.CONFIGS[0], 0, "commit")
    reference["destination"] = 0
    reference["output_block"] = 0
    reference["configuration"] = m.CONFIGS[1]
    compact = copy.deepcopy(reference)
    outcomes["request_payload_relabel"] = rejected(
        m, lambda: request_payload_miter.accept_request_pair(reference, compact))

    state_payload_miter = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(state_payload_miter)
    bad_state = m.synthetic_state(state_payload_miter, True)
    bad_state["configuration"] = m.CONFIGS[1]
    outcomes["state_payload_relabel"] = rejected(
        m, lambda: state_payload_miter.accept_destination_pair(
            bad_state, copy.deepcopy(bad_state)))

    payload_miter = d.completed(m.CONFIGS[0])
    payload_miter.configuration = m.CONFIGS[1]
    outcomes["finish_payload_owner_relabel"] = rejected(
        m, payload_miter._finish_payload)

    finish_miter = d.completed(m.CONFIGS[0])
    finish_miter.configuration = m.CONFIGS[2]
    outcomes["finish_owner_relabel"] = rejected(m, finish_miter.finish)

    dense = [d.completed(m.CONFIGS[0]) for _index in range(3)]
    dense_receipt = dense[0].finish()
    require(dense_receipt.as_dict()["configuration"] == m.CONFIGS[0],
            "dense control receipt drift")
    for index, target in enumerate(m.CONFIGS[1:], 1):
        dense[index].configuration = target
        outcomes["dense3_relabel_to_" + target] = rejected(m, dense[index].finish)

    bundle_rows = [d.completed(configuration).finish()
                   for configuration in m.CONFIGS]
    original_value = bundle_rows[1].as_dict()
    original_value["configuration"] = m.CONFIGS[0]
    object.__setattr__(bundle_rows[1], "_payload_json",
                       m.canonical_bytes(original_value).decode("utf-8"))
    outcomes["bundle_payload_configuration_relabel"] = rejected(
        m, lambda: m.validate_bundle(bundle_rows))

    coverage_rows = [d.completed(configuration).finish()
                     for configuration in m.CONFIGS]
    changed = coverage_rows[1].as_dict()
    changed["dense_cache_covered"] = True
    changed["dense_psum_1rw_covered"] = True
    object.__setattr__(coverage_rows[1], "_payload_json",
                       m.canonical_bytes(changed).decode("utf-8"))
    outcomes["bundle_non_dense_coverage_relabel"] = rejected(
        m, lambda: m.validate_bundle(coverage_rows))
    require(all(outcomes.values()), "configuration-binding mutation escaped")
    require(len(outcomes) == 10, "configuration attack population drift")
    return outcomes


def main():
    for path, digest in EXPECTED.items():
        verify_regular(path, digest)
    verify_file_seal(CONTRACT)
    verify_tree(AUTHOR)
    verify_tree(M1629)
    require(not FUTURE_RELEASE.exists() and
            not Path(str(FUTURE_RELEASE) + ".sha256").exists() and
            not Path(str(FUTURE_RELEASE) + ".sha256.seal.sha256").exists(),
            "actual-prefix release exists before M1639 admission")
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    prior_review = strict_json(M1629 / "review.json")
    require(author.get("status") ==
            "PASS_AUTHOR_M1629_CONFIGURATION_RELABEL_P1_SOURCE_REPAIR__M1639_DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_EXECUTION" and
            author.get("score") >= 95 and author.get("paper_claims") == 0,
            "M1638 author receipt drift")
    require(prior_review.get("status") ==
            "NO_GO_M1628_ACTUAL_L2_RUNNER_SOURCE__ONE_P1_SESSION_CONFIGURATION_BINDING_GAP__SUCCESSOR_REPAIR_ONLY" and
            [row.get("id") for row in prior_review.get("findings", {}).get("p1", [])] ==
            ["P1_SESSION_CONFIGURATION_RELABEL_NOT_BOUND_AT_CREATION"],
            "M1629 P1 authority drift")
    source_text = SOURCE.read_text(encoding="utf-8")
    audit_static_source(source_text, contract)
    m = load(SOURCE, "m1639_bound_m1638")
    prior = load(M1629_HAMMER, "m1639_bound_m1629_hammer")
    require(m.CONFIGS == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8"),
        "configuration order drift")
    d, survivors, prior_extra = run_prior_attacks(m, prior)
    binding = run_configuration_binding_attacks(m, d)
    require(rejected(m, lambda: m.actual_prefix_release(lambda: True, object())),
            "actual prefix runner/payload path opened")
    describe = m.describe()
    require(describe["repair"]["exact_bundle_coverage_policy"] ==
            [[True, True], [False, False], [False, False]] and
            describe["repair"]["request_state_payload_finish_all_check_binding"] is True and
            all(describe["authorization"][key] is False for key in (
                "actual_runner_source", "actual_payload", "l2_execution", "l3",
                "pilot", "production", "cycles", "traffic", "energy", "speedup",
                "rtl", "eda", "paper_result")), "describe boundary drift")
    output = {
        "schema": "m1639_m1638_configuration_bound_independent_hammer_v1",
        "status": "PASS",
        "m1620_m1628_survivors": survivors,
        "m1629_additional_attacks": prior_extra,
        "configuration_binding_attacks": binding,
        "coverage_policy": [[True, True], [False, False], [False, False]],
        "attack_categories": len(survivors) + len(prior_extra) + len(binding) + 1,
        "all_attacks_rejected": True,
        "actual_payload_opened": False,
        "l2_executed": False,
        "l3_executed": False,
        "runner_authored": False,
        "release_created": False,
        "gpu": False,
        "eda": False,
        "source_sha256": sha(SOURCE),
        "test_sha256": sha(TEST),
        "contract_sha256": sha(CONTRACT),
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        raise SystemExit("FAIL_CLOSED_M1639: " + str(error))

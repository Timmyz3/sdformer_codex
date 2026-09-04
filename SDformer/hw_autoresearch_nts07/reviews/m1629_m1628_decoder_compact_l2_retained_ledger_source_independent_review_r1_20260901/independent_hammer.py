#!/usr/bin/env python3
"""Different-author synthetic-only hammer for the M1628 L2 source successor."""
from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1628_decoder_compact_l2_retained_ledger_successor_source.py"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1628_decoder_compact_l2_retained_ledger_successor_source.py"
CONTRACT = HW / "contracts/m1628_decoder_compact_l2_retained_ledger_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1628_decoder_compact_l2_retained_ledger_successor_source_author_receipt_r1_20260901"
M1620 = HW / "reviews/m1620_m1619_decoder_compact_l2_canonical_prefix_source_independent_review_r1_20260901"
M1619 = HW / "system_simulator/scripts/build_m1619_decoder_compact_l2_canonical_prefix_source.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / "contracts/m1633_m1629_m1628_decoder_compact_l2_actual_prefix_source_release_r1_20260901.json"

PINS = {
    SOURCE: "41bae4c11484c4bf3e5da9537225372d703c2f7206879ecf7814bc52c56c0df4",
    AUTHOR_TEST: "92ec0f60987566d42706c0b3b71be35d269485a9846a74bc2722b266d7aa68ba",
    CONTRACT: "591ef3e1a2e39bbeeacad9f7a291f400dd77a471339c8d953ee4a6699c9bdb65",
    AUTHOR / "review.json": "b3a1a630bbd0e1387a8d246db5bb7c068898587fd7cc47f08948b79aad8928ce",
    M1620 / "review.json": "38b2b52316eaa90d28b668dfbf07a14e1dc0b0ea3b2f283c0699cb3b6d256bd9",
    M1619: "12c57983dee200c6c2eda3c42c13b3e111ec1b2ade86309f4b4b65f1b90306a0",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text(encoding="ascii").split() ==
            [sha256(path), path.name], "file seal mismatch")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), manifest.name], "file outer mismatch")


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    require(outer.is_file() and not outer.is_symlink(), "outer absent")
    listed = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        if not row.strip():
            continue
        digest, name = row.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in listed, "unsafe manifest row")
        listed[name] = digest
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "outer content drift")
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        bp = Path(base)
        dirs[:] = [name for name in dirs if not (bp / name).is_symlink()]
        for name in files:
            path = bp / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                    "nonregular tree member")
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(listed), "tree topology drift")
    for name, digest in listed.items():
        require(sha256(root / name) == digest, "tree member drift " + name)


def load_source():
    spec = importlib.util.spec_from_file_location("m1629_bound_m1628",
                                                  str(SOURCE))
    require(spec is not None and spec.loader is not None, "source loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(module, action):
    try:
        action()
    except module.M1628Error:
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
                    self.request(
                        miter, "commit", block,
                        lambda row, bias=address_bias: row.update({
                            "addresses": [value + bias
                                          for value in row["addresses"]]}))
                else:
                    self.request(miter, "commit", block)
            self.state(miter)
        return miter

    def dense_completed_relabelled(self, target_configuration):
        miter = self.completed(self.m.CONFIGS[0])
        miter.configuration = target_configuration
        return miter


def survivor_max_return(m, d):
    mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(mi, weight=False, long_return=True)
    row = m.synthetic_state(mi, True, 100)
    require(5000 in row["outstanding_active_returns"][14], "long return absent")
    row["outstanding_active_returns"] = [[] for _index in range(16)]
    return rejected(m, lambda: mi.accept_destination_pair(row, copy.deepcopy(row)))


def survivor_cross_destination(m, d):
    mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(mi, weight=False, long_return=True)
    d.state(mi, 100)
    d.populate(mi, weight=False)
    row = m.synthetic_state(mi, True, 200)
    require(5000 in row["outstanding_active_returns"][14], "retained return absent")
    row["outstanding_active_returns"][14] = []
    return rejected(m, lambda: mi.accept_destination_pair(row, copy.deepcopy(row)))


def survivor_psum_ready(m, d):
    mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(mi, psum=True)
    first = d.state(mi)
    require(first["numeric_dependency_state"]["last_psum_write_ready"][0] > 0,
            "psum readiness not exercised")
    d.populate(mi)
    row = m.synthetic_state(mi, True)
    row["numeric_dependency_state"]["last_psum_write_ready"][0] = 0
    return rejected(m, lambda: mi.accept_destination_pair(row, copy.deepcopy(row)))


def survivor_cache_continuity(m, d):
    mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(mi)
    first = d.state(mi)
    d.populate(mi)
    cleared = m.synthetic_state(mi, True)
    cleared["cache"]["valid_entries"] = 0
    a = rejected(m, lambda: mi.accept_destination_pair(
        cleared, copy.deepcopy(cleared)))
    unchanged = m.synthetic_state(mi, True)
    for field in ("tick", "hits", "misses", "evictions"):
        unchanged["cache"][field] = first["cache"][field]
    b = rejected(m, lambda: mi.accept_destination_pair(
        unchanged, copy.deepcopy(unchanged)))
    return a and b


def survivor_request_scope(m, d):
    attacks = []
    for field, value in (("module", 99), ("timestep", 99),
                         ("destination", 41), ("output_block", 99)):
        mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
        attacks.append(rejected(
            m, lambda mi=mi, field=field, value=value:
            d.request(mi, "commit", 0,
                      lambda row: row.update({field: value}))))
    return all(attacks)


def survivor_count_bytes(m, d):
    mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
    d.populate(mi)
    row = m.synthetic_state(mi, True)
    row["kind_counts"] = {"external_read": row["request_count"]}
    row["byte_counts"] = {"external_read": 0}
    return rejected(m, lambda: mi.accept_destination_pair(row, copy.deepcopy(row)))


def survivor_digests(m, d):
    outcomes = []
    for field, value in (("packed_transaction_address_sha256", "1" * 64),
                         ("packed_commit_sequence_sha256", "2" * 64)):
        mi = m.CanonicalPrefixMiter(m.CONFIGS[0])
        d.populate(mi)
        row = m.synthetic_state(mi, True)
        row[field] = value
        outcomes.append(rejected(
            m, lambda mi=mi, row=row:
            mi.accept_destination_pair(row, copy.deepcopy(row))))
    return all(outcomes)


def survivor_forged_finish(m, _d):
    forged = [{"configuration": configuration,
               "destinations": m.PREFIX_DESTINATIONS,
               "requests": 1,
               "commits": m.EXPECTED_COMMITS_PER_CONFIG,
               "dense_cache_covered": True,
               "dense_psum_1rw_covered": True,
               "final_commit_digest": "f" * 64,
               "final_state_sha256": "e" * 64}
              for configuration in m.CONFIGS]
    return rejected(m, lambda: m.validate_bundle(forged))


def build():
    for path, digest in PINS.items():
        require(path.is_file(), "missing input " + str(path))
        require(sha256(path) == digest, "input SHA drift " + str(path))
    verify_file_seal(CONTRACT)
    for tree in (AUTHOR, M1620):
        verify_tree(tree)
    require(sha256(AUTHOR / "SHA256SUMS") ==
            "c2b14695c3760e9f089b64b8ed2c083395712b3c3b5daff98e80615a121b54c4",
            "author manifest identity")
    require(not RELEASE.exists(), "future release must remain absent")
    source_text = SOURCE.read_text(encoding="utf-8")
    for token in ("np.load", "numpy.load", "torch.load", ".npz", ".tar.zst",
                  "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
        require(token not in source_text, "actual payload token " + token)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["authorization"]["actual_prefix_runner_source"] is False and
            contract["authorization"]["actual_payload"] is False and
            contract["authorization"]["l2_execution"] is False and
            contract["authorization"]["eda"] is False,
            "contract execution boundary")

    m = load_source()
    d = Driver(m)
    require(rejected(m, lambda: m.actual_prefix_release(lambda: True, object())),
            "actual prefix release opened")
    survivors = {
        "S1_EARLIER_MAX_RETURN_CANNOT_BE_DROPPED": survivor_max_return(m, d),
        "S2_CROSS_DESTINATION_FUTURE_RETURN_CANNOT_DISAPPEAR":
            survivor_cross_destination(m, d),
        "S3_LAST_PSUM_WRITE_READY_CANNOT_MOVE_BACKWARD": survivor_psum_ready(m, d),
        "S4_CACHE_CLEAR_OR_CONTENT_RESET_CANNOT_PASS": survivor_cache_continuity(m, d),
        "S5_REQUEST_SCOPE_MODULE_TIMESTEP_DESTINATION_BLOCK_IS_BOUND":
            survivor_request_scope(m, d),
        "S6_KIND_AND_BYTE_LEDGER_IS_INTERNALLY_DERIVED": survivor_count_bytes(m, d),
        "S7_ADDRESS_AND_COMMIT_DIGESTS_ARE_INTERNALLY_DERIVED":
            survivor_digests(m, d),
        "S8_HAND_WRITTEN_FINISH_ROWS_ARE_REJECTED": survivor_forged_finish(m, d),
    }
    require(len(survivors) == 8 and all(survivors.values()),
            "one or more M1620 survivors still pass")

    # Genuine receipt/authenticity/replay controls.
    rows = [d.completed(configuration).finish() for configuration in m.CONFIGS]
    identities = [row.as_dict()["session_identity"] for row in rows]
    require(len(set(identities)) == 3, "fresh identities are not distinct")
    clone = object.__new__(type(rows[0]))
    object.__setattr__(clone, "_payload_json", rows[0]._payload_json)
    object.__setattr__(clone, "_tag", rows[0]._tag)
    object.__setattr__(clone, "_locked", True)
    clone_rejected = rejected(m, lambda: m.validate_bundle(
        [clone, rows[1], rows[2]]))
    duplicate_rejected = rejected(m, lambda: m.validate_bundle(
        [rows[0], rows[0], rows[2]]))
    reorder_rejected = rejected(m, lambda: m.validate_bundle(
        [rows[1], rows[0], rows[2]]))
    require(clone_rejected and duplicate_rejected and reorder_rejected,
            "receipt identity/order attack survived")
    require(m.validate_bundle(rows) is True, "genuine bundle rejected")
    replay_rejected = rejected(m, lambda: m.validate_bundle(rows))
    require(replay_rejected, "consumed bundle replay survived")

    tag_rows = [d.completed(configuration).finish()
                for configuration in m.CONFIGS]
    object.__setattr__(tag_rows[0], "_tag", "0" * 64)
    tag_rejected = rejected(m, lambda: m.validate_bundle(tag_rows))
    require(tag_rejected, "receipt tag mutation survived")

    stream_rows = [d.completed(configuration, 4096 if index == 2 else 0).finish()
                   for index, configuration in enumerate(m.CONFIGS)]
    shared_stream_rejected = rejected(m, lambda: m.validate_bundle(stream_rows))
    require(shared_stream_rejected, "cross-configuration commit mismatch survived")

    # New first-principles attack: session configuration is mutable and is not
    # bound by the hidden issued-session registry. Three DENSE sessions can be
    # relabelled after all requests/state were accepted, then pass as the three
    # frozen configurations. This must survive to substantiate the P1 verdict.
    relabelled = [d.dense_completed_relabelled(configuration).finish()
                  for configuration in m.CONFIGS]
    relabelled_values = [row.as_dict() for row in relabelled]
    relabelled_accepted = False
    try:
        relabelled_accepted = m.validate_bundle(relabelled) is True
    except m.M1628Error:
        relabelled_accepted = False
    require(relabelled_accepted,
            "configuration relabel attack unexpectedly rejected; reassess verdict")

    return {
        "schema": "m1629_m1628_decoder_compact_l2_retained_ledger_independent_hammer_v1",
        "status": "AUDIT_COMPLETE__EIGHT_SURVIVORS_REJECTED__ONE_NEW_P1_SURVIVED",
        "m1620_survivors": survivors,
        "m1620_survivor_categories": 8,
        "m1620_survivor_categories_rejected": 8,
        "additional_rejections": {
            "receipt_clone": clone_rejected,
            "duplicate_session": duplicate_rejected,
            "configuration_reorder": reorder_rejected,
            "tag_mutation": tag_rejected,
            "shared_commit_stream_mismatch": shared_stream_rejected,
            "consumed_bundle_replay": replay_rejected,
        },
        "new_p1": {
            "id": "P1_SESSION_CONFIGURATION_RELABEL_NOT_BOUND_AT_CREATION",
            "mutation_survived": relabelled_accepted,
            "actual_session_configuration": m.CONFIGS[0],
            "reported_configuration_order": [row["configuration"]
                                               for row in relabelled_values],
            "all_dense_coverage_flags": [row["dense_cache_covered"]
                                          for row in relabelled_values],
            "required_repair": "Bind immutable configuration to the issued session at new_session and require the same value at every request, state, finish and bundle receipt; reject extra dense-only coverage on non-dense configurations or bind per-configuration coverage policy."
        },
        "identity": {"source_sha256": sha256(SOURCE),
                     "test_sha256": sha256(AUTHOR_TEST),
                     "contract_sha256": sha256(CONTRACT),
                     "author_review_sha256": sha256(AUTHOR / "review.json"),
                     "docs359_sha256": sha256(DOC359)},
        "execution": {"actual_payload_opened": False,
                      "actual_l2_executed": False, "l3_executed": False,
                      "pilot": False, "production": False, "gpu": False,
                      "eda": False, "attempt_created": False,
                      "release_created": False, "paper_result": False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    value = build()
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

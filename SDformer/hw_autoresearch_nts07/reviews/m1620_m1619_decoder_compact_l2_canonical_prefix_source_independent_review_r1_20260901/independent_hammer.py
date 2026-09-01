#!/usr/bin/env python3
"""Different-author, payload-free hammer of the exact M1619 L2 source.

This review imports only the sealed source-only interface and runs synthetic
mutations.  It intentionally has no actual-payload path and never invokes an
L2/L3 runner.  Surviving mutations are evidence against authorizing the next
actual-prefix source.
"""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1619_decoder_compact_l2_canonical_prefix_source.py"
TEST = HW / "system_simulator/tests/test_m1619_decoder_compact_l2_canonical_prefix_source.py"
CONTRACT = HW / "contracts/m1619_decoder_compact_l2_canonical_prefix_source_contract_r1_20260901.json"
CONTRACT_MANIFEST = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT_MANIFEST) + ".seal.sha256")
AUTHOR = HW / "reviews/m1619_decoder_compact_l2_canonical_prefix_source_author_receipt_r1_20260901"
M1610_SOURCE = HW / "system_simulator/scripts/build_m1610_decoder_compact_cycle_simulator_source.py"
M1539_SOURCE = HW / "system_simulator/scripts/build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
M1615 = HW / "reviews/m1615_m1610_decoder_compact_l0_l1_independent_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "m1619_source": "12c57983dee200c6c2eda3c42c13b3e111ec1b2ade86309f4b4b65f1b90306a0",
    "m1619_test": "c98dbbd90c56290734ea42e8bb8172f318a0ec6980c14c694b4ffc7c00110779",
    "m1619_contract": "ad9e4dbc496cca78e098a85bced3766c94bb2fc3f7f0e78a272f7d9f95cda8bb",
    "m1619_contract_manifest": "649a74e957238b85c7237c00e07ebbd5cdad50fc6377749cfc6a7fca14a81cc3",
    "m1619_contract_outer_file": "c32f30a9839595c59f205d795064681edb95b4089aa7df4843717ef5ebf1b21c",
    "m1619_author_review": "95dd36cb55fdba725bb6454b0f9717fe832f4bf302a5b4100fe7d8b18d7b082f",
    "m1619_author_manifest": "25146cdd2c34185ecf27dc6c1245a3f1544ec4b22d6c419eb1a6e2e1a77583c7",
    "m1619_author_outer_file": "e52403c2f7895f8d0bd4ed58872532e553f01ada13d5825741a6182cea22bc0c",
    "m1610_source": "73d4bade27612a3dfcbdc3e7417d7180397629a5be1f9e23587a58ea487b84ce",
    "m1539_source": "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4",
    "m1615_review": "ab87c20943052570a24b4e7beb2bee3be913fcb95c388597c7fee844b1fe5f4c",
    "m1615_manifest": "fd13594b29891c46e203e14bcffab823aab999ff1263ed492081ee937a681360",
    "m1615_outer_file": "800f6c2e7a48ca90e1513aa6d0f7bd3691bc434a687de400874706fde0afef0d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " is not a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def flat_seal(directory, review_sha, manifest_sha, outer_file_sha, label):
    regular_exact(directory / "review.json", review_sha, label + " review")
    regular_exact(directory / "SHA256SUMS", manifest_sha,
                  label + " manifest")
    regular_exact(directory / "SHA256SUMS.seal.sha256", outer_file_sha,
                  label + " outer seal")
    require((directory / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() == [manifest_sha, "SHA256SUMS"],
            label + " outer seal content drift")


def bind_inputs():
    regular_exact(SOURCE, EXPECTED["m1619_source"], "M1619 source")
    regular_exact(TEST, EXPECTED["m1619_test"], "M1619 test")
    regular_exact(CONTRACT, EXPECTED["m1619_contract"], "M1619 contract")
    regular_exact(CONTRACT_MANIFEST, EXPECTED["m1619_contract_manifest"],
                  "M1619 contract manifest")
    regular_exact(CONTRACT_OUTER, EXPECTED["m1619_contract_outer_file"],
                  "M1619 contract outer seal")
    require(CONTRACT_MANIFEST.read_text(encoding="ascii").split() ==
            [EXPECTED["m1619_contract"], CONTRACT.name],
            "M1619 contract manifest content drift")
    require(CONTRACT_OUTER.read_text(encoding="ascii").split() ==
            [EXPECTED["m1619_contract_manifest"], CONTRACT_MANIFEST.name],
            "M1619 contract outer content drift")
    flat_seal(AUTHOR, EXPECTED["m1619_author_review"],
              EXPECTED["m1619_author_manifest"],
              EXPECTED["m1619_author_outer_file"], "M1619 author")
    regular_exact(M1610_SOURCE, EXPECTED["m1610_source"], "M1610 source")
    regular_exact(M1539_SOURCE, EXPECTED["m1539_source"], "M1539 source")
    flat_seal(M1615, EXPECTED["m1615_review"], EXPECTED["m1615_manifest"],
              EXPECTED["m1615_outer_file"], "M1615")
    regular_exact(DOCS359, EXPECTED["docs359"], "docs359")


def load_source():
    spec = importlib.util.spec_from_file_location("m1620_bound_m1619",
                                                  str(SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1619")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejects(module, action, label):
    try:
        action()
    except module.M1619Error:
        return label
    raise AssertionError("mutation survived unexpectedly: " + label)


def accept_four_requests(module, miter, mutator=None):
    accepted = []
    for _index in range(module.OUTPUT_BLOCKS):
        ordinal = miter.next_request_ordinal
        row = module.synthetic_request(miter.configuration, ordinal)
        if mutator is not None:
            mutator(row, ordinal)
        miter.accept_request_pair(row, copy.deepcopy(row))
        accepted.append(row)
    return accepted


def request_exact_mutations(module):
    base = module.synthetic_request(module.CONFIGS[0], 0)
    base.update({"earliest_issue_cycle": 100,
                 "dependency_ready_cycle": 100,
                 "port_ready_cycle": 100,
                 "issue_cycle": 110,
                 "return_cycle": 210})
    other_kind = next(name for name in module.C.KIND_NAMES
                      if name != base["kind"])
    changes = {
        "configuration": module.CONFIGS[1],
        "schema_version": 2,
        "module": 1,
        "timestep": 1,
        "destination": 1,
        "output_block": 1,
        "group": 1,
        "subordinal": 1,
        "request_ordinal": 1,
        "kind": other_kind,
        "earliest_issue_cycle": 99,
        "dependency_ready_cycle": 99,
        "port_ready_cycle": 99,
        "issue_cycle": 111,
        "beats": 3,
        "return_cycle": 211,
        "width_bytes": 385,
        "addresses": [base["addresses"][0] + 384],
        "banks": [1],
        "packed_event_sha256": "a" * 64,
    }
    rejected = []
    for field in module.REQUEST_FIELDS:
        reference = copy.deepcopy(base)
        compact = copy.deepcopy(base)
        compact[field] = copy.deepcopy(changes[field])
        rejected.append(rejects(
            module,
            lambda reference=reference, compact=compact:
                module.CanonicalPrefixMiter(module.CONFIGS[0]).accept_request_pair(
                    reference, compact),
            "request_pair_" + field))
    require(len(rejected) == len(module.REQUEST_FIELDS),
            "not every request field was mutation-tested")
    return rejected


def state_for_first_destination(module):
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    accept_four_requests(module, miter)
    state = module.synthetic_state(module.CONFIGS[0], 0,
                                   miter.next_request_ordinal, True)
    return miter, state


def prefix_exact_mutations(module):
    alternate_kind = next(name for name in module.C.KIND_NAMES
                          if name != "commit")
    rejected = []
    for field in module.PREFIX_EXACT_FIELDS:
        miter, reference = state_for_first_destination(module)
        compact = copy.deepcopy(reference)
        if field == "configuration":
            compact[field] = module.CONFIGS[1]
        elif field == "destination":
            compact[field] = 1
        elif field == "last_cycle":
            compact[field] += 1
        elif field == "request_count":
            compact[field] += 1
        elif field == "kind_counts":
            compact[field] = {alternate_kind: compact["request_count"]}
        elif field == "byte_counts":
            compact[field] = {"commit": compact["byte_counts"]["commit"] + 1}
        elif field in ("packed_transaction_address_sha256",
                       "packed_commit_sequence_sha256"):
            compact[field] = "b" * 64
        elif field == "next_port_calendar":
            compact[field][0] += 1
        elif field == "outstanding_active_returns":
            compact[field][0] = [10000]
        elif field == "numeric_dependency_state":
            compact[field]["last_psum_write_ready"][0] += 1
        elif field == "cache":
            compact[field]["state_sha256"] = "c" * 64
        elif field == "coverage_counters":
            compact[field]["outstanding_full_waits"] += 1
        elif field == "commit_count":
            compact[field] += 1
        elif field == "reset_count":
            compact[field] = 1
        elif field == "resource_manifest_sha256":
            compact[field] = "d" * 64
        else:
            raise AssertionError("unhandled prefix field: " + field)
        rejected.append(rejects(
            module,
            lambda miter=miter, reference=reference, compact=compact:
                miter.accept_destination_pair(reference, compact),
            "destination_pair_" + field))
    require(len(rejected) == len(module.PREFIX_EXACT_FIELDS),
            "not every prefix exact field was mutation-tested")
    return rejected


def structural_and_history_rejections(module):
    rejected = []

    miter, state = state_for_first_destination(module)
    missing_rss = copy.deepcopy(state)
    del missing_rss["rss"]
    rejected.append(rejects(
        module, lambda: miter.accept_destination_pair(missing_rss,
                                                       copy.deepcopy(missing_rss)),
        "missing_destination_rss"))

    miter, state = state_for_first_destination(module)
    state["rss"]["hwm_rss_kib"] = module.RSS_ABSOLUTE_LIMIT_KIB
    rejected.append(rejects(
        module, lambda: miter.accept_destination_pair(state, copy.deepcopy(state)),
        "rss_absolute_limit"))

    miter, state = state_for_first_destination(module)
    state["reset_count"] = 1
    rejected.append(rejects(
        module, lambda: miter.accept_destination_pair(state, copy.deepcopy(state)),
        "per_destination_reset"))

    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    accept_four_requests(module, miter)
    skipped = module.synthetic_state(module.CONFIGS[0], 1,
                                     miter.next_request_ordinal, True)
    rejected.append(rejects(
        module, lambda: miter.accept_destination_pair(skipped,
                                                       copy.deepcopy(skipped)),
        "skipped_destination"))

    incomplete = module.CanonicalPrefixMiter(module.CONFIGS[0])
    rejected.append(rejects(module, incomplete.finish, "incomplete_prefix"))

    def two_destination_backward(mutator, label):
        m = module.CanonicalPrefixMiter(module.CONFIGS[0])
        accept_four_requests(module, m)
        first = module.synthetic_state(module.CONFIGS[0], 0,
                                       m.next_request_ordinal, True)
        m.accept_destination_pair(first, copy.deepcopy(first))
        accept_four_requests(module, m)
        second = module.synthetic_state(module.CONFIGS[0], 1,
                                        m.next_request_ordinal, True)
        mutator(first, second)
        return rejects(module,
                       lambda: m.accept_destination_pair(second,
                                                          copy.deepcopy(second)),
                       label)

    rejected.append(two_destination_backward(
        lambda first, second: second["cache"].update(
            {"tick": first["cache"]["tick"] - 1}),
        "cache_tick_backward"))
    rejected.append(two_destination_backward(
        lambda first, second: second.update(
            {"next_port_calendar": [0] * 24}),
        "port_calendar_backward"))
    rejected.append(two_destination_backward(
        lambda _first, second: second["numeric_dependency_state"].update(
            {"source_ready_cycle": 99}),
        "source_dependency_changed"))
    rejected.append(two_destination_backward(
        lambda _first, second: second["numeric_dependency_state"].update(
            {"persistent_control_ready_cycle": 99}),
        "control_dependency_changed"))
    return rejected


def surviving_mutations(module):
    survivors = []

    # last_request_return is overwritten rather than accumulated as a max, so
    # an earlier long-latency request can remain in flight past last_cycle.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    def long_first_return(row, ordinal):
        if ordinal == 0:
            row["return_cycle"] = 5000
    accept_four_requests(module, miter, long_first_return)
    state = module.synthetic_state(module.CONFIGS[0], 0,
                                   miter.next_request_ordinal, True)
    require(state["last_cycle"] < 5000,
            "synthetic max-return proof invalid")
    miter.accept_destination_pair(state, copy.deepcopy(state))
    survivors.append("earlier_request_return_after_destination_last_cycle_not_tracked")

    # A future active return and psum dependency disappear/move backward even
    # though the second prefix last_cycle has not reached them.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    accept_four_requests(module, miter)
    first = module.synthetic_state(module.CONFIGS[0], 0,
                                   miter.next_request_ordinal, True)
    first["outstanding_active_returns"][0] = [5000]
    first["numeric_dependency_state"]["last_psum_write_ready"] = [5000] * 4
    miter.accept_destination_pair(first, copy.deepcopy(first))
    accept_four_requests(module, miter)
    second = module.synthetic_state(module.CONFIGS[0], 1,
                                    miter.next_request_ordinal, True)
    second["outstanding_active_returns"][0] = []
    second["numeric_dependency_state"]["last_psum_write_ready"] = [0] * 4
    require(second["last_cycle"] < 5000, "synthetic future-return proof invalid")
    miter.accept_destination_pair(second, copy.deepcopy(second))
    survivors.extend(["future_outstanding_return_disappears",
                      "last_psum_dependency_moves_backward"])

    # The cache contents can be reset while tick/counters remain monotonic.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    accept_four_requests(module, miter)
    first = module.synthetic_state(module.CONFIGS[0], 0,
                                   miter.next_request_ordinal, True)
    require(first["cache"]["valid_entries"] == 9,
            "cache-reset attack precondition missing")
    miter.accept_destination_pair(first, copy.deepcopy(first))
    accept_four_requests(module, miter)
    second = module.synthetic_state(module.CONFIGS[0], 1,
                                    miter.next_request_ordinal, True)
    second["cache"]["valid_entries"] = 0
    second["cache"]["state_sha256"] = "e" * 64
    miter.accept_destination_pair(second, copy.deepcopy(second))
    survivors.append("cache_contents_reset_between_destinations")

    # Requests can be from the wrong canonical scope, and the cumulative
    # counts/bytes/digests need not be derived from accepted request receipts.
    miter = module.CanonicalPrefixMiter(module.CONFIGS[0])
    def wrong_scope(row, _ordinal):
        row["module"] = 99
        row["timestep"] = 99
        row["destination"] = 41
        row["output_block"] = 99
    accepted = accept_four_requests(module, miter, wrong_scope)
    state = module.synthetic_state(module.CONFIGS[0], 0,
                                   miter.next_request_ordinal, True)
    alternate_kind = next(name for name in module.C.KIND_NAMES
                          if name != "commit")
    state["kind_counts"] = {alternate_kind: len(accepted)}
    state["byte_counts"] = {alternate_kind: 0}
    state["packed_transaction_address_sha256"] = "1" * 64
    state["packed_commit_sequence_sha256"] = "2" * 64
    miter.accept_destination_pair(state, copy.deepcopy(state))
    survivors.extend(["request_scope_not_bound_to_canonical_destination",
                      "prefix_kind_and_byte_ledger_not_derived_from_requests",
                      "prefix_digests_not_derived_from_requests"])

    # validate_bundle accepts forged rows that were never produced by a fresh
    # completed CanonicalPrefixMiter session.
    forged = []
    for configuration in module.CONFIGS:
        forged.append({"configuration": configuration,
                       "destinations": module.PREFIX_DESTINATIONS,
                       "requests": 0,
                       "commits": module.EXPECTED_COMMITS_PER_CONFIG,
                       "dense_cache_covered": False,
                       "dense_psum_1rw_covered": False,
                       "final_commit_digest": "f" * 64})
    require(module.validate_bundle(forged) is True,
            "forged cross-configuration bundle unexpectedly rejected")
    survivors.append("cross_configuration_bundle_has_no_fresh_session_proof")
    return survivors


def boundary_and_claim_hammer(module, contract):
    source_text = SOURCE.read_text(encoding="utf-8")
    for token in ("np.load", "numpy.load", "torch.load", ".npz",
                  ".tar.zst",
                  "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
        require(token not in source_text,
                "actual payload access token present: " + token)
    tree = ast.parse(source_text)
    require(not any(isinstance(node, ast.Constant) and
                    isinstance(node.value, str) and
                    node.value.startswith("--actual") for node in ast.walk(tree)),
            "actual execution CLI mode present")
    for provider, token in ((None, None), (lambda: True, object())):
        rejects(module,
                lambda provider=provider, token=token:
                    module.actual_prefix_release(provider, token),
                "actual_prefix_release_closed")
    description = module.describe()
    forbidden_claims = ("actual_payload", "l2_execution", "l3", "pilot",
                        "production", "cycles", "traffic", "speedup",
                        "paper_result")
    require(all(description["authorization"][key] is False
                for key in forbidden_claims),
            "source description opens paper/execution claim")
    require(all(contract["claim_boundary"][key] is False for key in
                ("actual_payload", "l2_execution", "l3", "pilot",
                 "production", "cycles", "traffic", "speedup",
                 "system_speedup", "energy", "rtl", "eda", "ppa",
                 "table_a", "paper_result")),
            "contract opens paper/execution claim")
    forged_claim = copy.deepcopy(contract)
    forged_claim["claim_boundary"]["paper_result"] = True
    try:
        require(all(forged_claim["claim_boundary"][key] is False for key in
                    ("paper_result", "system_speedup")),
                "forged paper claim")
    except AssertionError:
        claim_attack_rejected = True
    else:
        claim_attack_rejected = False
    require(claim_attack_rejected, "hammer failed to reject forged paper claim")
    return {"actual_release_calls_rejected": 2,
            "actual_cli_modes": 0,
            "forbidden_payload_tokens": 0,
            "forged_paper_claim_rejected": True}


def main():
    bind_inputs()
    contract = strict_json(CONTRACT)
    require(contract["status"] ==
            "SOURCE_ONLY__L2_CANONICAL_PREFIX_INTERFACE__NO_PAYLOAD_NO_EXECUTION",
            "M1619 contract status drift")
    module = load_source()
    geometry = module.prefix_geometry()
    require(geometry == {"count": 42, "first": (0, 0, 0),
                         "last": (41, 1, 1), "parity_mask": 15,
                         "corner": True, "edge": True, "interior": True},
            "canonical geometry/parity/corner-edge-interior drift")
    built_in = module.static_self_test()
    require(built_in["actual_payload"] is False and
            built_in["l2_executed"] is False and
            built_in["l3_executed"] is False and
            built_in["paper_result"] is False,
            "built-in test crossed source-only boundary")
    request_rejected = request_exact_mutations(module)
    prefix_rejected = prefix_exact_mutations(module)
    history_rejected = structural_and_history_rejections(module)
    survived = surviving_mutations(module)
    require(survived == [
        "earlier_request_return_after_destination_last_cycle_not_tracked",
        "future_outstanding_return_disappears",
        "last_psum_dependency_moves_backward",
        "cache_contents_reset_between_destinations",
        "request_scope_not_bound_to_canonical_destination",
        "prefix_kind_and_byte_ledger_not_derived_from_requests",
        "prefix_digests_not_derived_from_requests",
        "cross_configuration_bundle_has_no_fresh_session_proof",
    ], "surviving mutation set drift")
    output = {
        "schema": "m1620_m1619_decoder_compact_l2_canonical_prefix_independent_hammer_r1_v1",
        "status": "NO_GO_M1619_ACTUAL_L2_RUNNER_SOURCE__SUCCESSOR_INTERFACE_REPAIR_REQUIRED",
        "python": sys.version,
        "input_sha256": EXPECTED,
        "geometry": geometry,
        "built_in_static_sessions": len(built_in["synthetic_sessions"]),
        "rejected_mutations": {
            "per_request_exact_fields": len(request_rejected),
            "per_destination_exact_fields": len(prefix_rejected),
            "prefix_geometry_reset_rss_and_monotonic_history":
                len(history_rejected),
            "total": len(request_rejected) + len(prefix_rejected) +
                     len(history_rejected),
        },
        "surviving_mutations": survived,
        "boundary": boundary_and_claim_hammer(module, contract),
        "p0": 0,
        "p1": 3,
        "actual_payload_opened": False,
        "l2_executed": False,
        "l3_executed": False,
        "eda_gpu_executed": False,
        "paper_result": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

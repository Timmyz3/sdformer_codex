#!/usr/bin/env python3
"""Payload-free independent hammer for the M1645 actual-prefix source.

This review exercises only the synthetic scheduler/miter path.  It never calls
the private actual-prefix runner and never opens the frozen decoder payload.
The review directory necessarily exists while this file runs, so the author
time ``FUTURE_REVIEW`` absence gate is rebound to a fresh, absent sentinel only
for the synthetic regression; the real future release must remain absent.
"""
from __future__ import print_function

import ast
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
SOURCE = HW / "system_simulator/scripts/build_m1645_decoder_compact_actual_prefix_runner_source.py"
TEST = HW / "system_simulator/tests/test_m1645_decoder_compact_actual_prefix_runner_source.py"
CONTRACT = HW / "contracts/m1645_decoder_compact_actual_prefix_runner_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1645_decoder_compact_actual_prefix_runner_source_author_receipt_r1_20260901"
M1539 = HW / "system_simulator/scripts/build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
M1610 = HW / "system_simulator/scripts/build_m1610_decoder_compact_cycle_simulator_source.py"
M1638 = HW / "system_simulator/scripts/build_m1638_decoder_compact_l2_session_configuration_bound_successor_source.py"
M1639 = HW / "reviews/m1639_m1638_decoder_compact_l2_session_configuration_bound_source_independent_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
REVIEW = Path(__file__).resolve().parent
FUTURE_RELEASE = HW / "contracts/m1647_m1646_m1645_decoder_compact_actual_prefix_runner_release_r1_20260901.json"
ABSENT_REVIEW_SENTINEL = HW / "reviews/.m1646_absent_review_sentinel_for_independent_synthetic_test"

EXPECTED = {
    SOURCE: "0869bed30edbae34ed4d58a0959fa7f70962c3b78b383c80bbd96e4782e7d833",
    TEST: "bf0796b01da592b4e206ac3dee48773a325aeed9da70c7dd360d6067e53f48d8",
    CONTRACT: "8beeebe22bdb9d22c2032450dd79fb1578351fb11c55039bc5a533062912f957",
    Path(str(CONTRACT) + ".sha256"): "4eb762ae9aae7eb8e3267ea9de9e6204121d1ff7ddd8a0a890e57a7999e9182e",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "e20f8508885406b6c84404e4cf263663b104c707befe3f7a9e5194943c9f8dc0",
    AUTHOR / "review.json": "c565b966e6ce91aa2baf3ef1f8d0353e434c5c61b2cae9075a65c930e8a13ec6",
    AUTHOR / "SHA256SUMS": "bf18952af2b6e933fda5b7da658b9ca2bdb230428c6550a2e0b03f120cd5655e",
    AUTHOR / "SHA256SUMS.seal.sha256": "07c74a7158f6cd57c5cc81430ad5a70a5bafabdb2ed5efb633b6e91f9779e9ef",
    M1539: "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4",
    M1610: "73d4bade27612a3dfcbdc3e7417d7180397629a5be1f9e23587a58ea487b84ce",
    M1638: "1b3961b0d0682980a035f5ad9ba880eb44929e56116f23f2e68cbb9e0a3fdecd",
    M1639 / "review.json": "2af2dc261a4986e261bb74423b009a9cadd4449b647313d67487b5c5bd6c2ce6",
    M1639 / "SHA256SUMS": "c67fc715da69067be262c3bab4b5c7ba33fc5e8ef85f08e5eb586b0b7f7a24fb",
    M1639 / "SHA256SUMS.seal.sha256": "68ef86a7cc778bbafb18ac8bce9b9258f63bd1aa6643d0a44a35fa4f73eba6b9",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("non-finite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_regular(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular: " + str(path))
    require(sha(path) == digest, "identity drift: " + str(path))


def verify_file_seal(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii") ==
            sha(path) + "  " + path.name + "\n", "inner seal mismatch")
    require(outer.read_text(encoding="ascii") ==
            sha(sidecar) + "  " + sidecar.name + "\n", "outer seal mismatch")


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "sealed tree absent")
    require(outer.read_text(encoding="ascii") ==
            sha(manifest) + "  SHA256SUMS\n", "tree outer seal mismatch")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        require(re.match(r"^[0-9a-f]{64}  (?:\./)?[^/\n][^\n]*$", row),
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
            if path.is_file() and rel not in (
                    "SHA256SUMS", "SHA256SUMS.seal.sha256"):
                actual.add(rel)
    require(actual == set(listed), "sealed tree topology drift")
    for name, digest in listed.items():
        verify_regular(root / name, digest)


def load_source():
    spec = importlib.util.spec_from_file_location("m1646_exact_m1645", str(SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1645")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(module, action):
    try:
        action()
    except module.M1645Error:
        return True
    return False


def audit_static(text, contract):
    tree = ast.parse(text)
    private_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id == "_run_bound_actual_prefix":
            private_calls.append(node.lineno)
    require(private_calls == [], "private actual runner has a call site")
    main = text[text.index("def main(argv=None):"):]
    require("_run_bound_actual_prefix" not in main and
            "actual_prefix_release(" not in main,
            "CLI reaches private execution")
    require("self.miter.accept_request_pair(receipt, compact_receipt)" in text,
            "per-request M1638 miter call missing")
    require("self.miter.accept_destination_pair(reference_state, compact_state)" in text,
            "per-destination M1638 miter call missing")
    require("L2.validate_bundle(receipts)" in text,
            "three-session M1638 bundle gate missing")
    require("self.tokens.clear()" in text and
            "self.reference.tokens.clear()" in text,
            "destination token retirement missing")
    require("self.reference = R.AddressTimedScheduler(configuration)" in text and
            "self.compact = C.CompactScheduler(configuration)" in text and
            "self.cache = MirroredWeightCache()" in text and
            "self.miter = L2.CanonicalPrefixMiter(configuration)" in text,
            "persistent session objects missing")
    require("for destination in range(PREFIX_DESTINATIONS):" in text and
            "for output_block in range(OUTPUT_BLOCKS):" in text,
            "fixed prefix traversal missing")
    fixed = contract["fixed_population"]
    require(fixed["decoder_stage"] == "D0" and
            fixed["call_ordinal"] == 0 and fixed["module_ordinal"] == 0 and
            fixed["timestep"] == 0 and fixed["destinations"] ==
            "0..41 inclusive in exact row-major order" and
            fixed["output_blocks"] == "0..3 inclusive" and
            fixed["commits_per_configuration"] == 168,
            "contract population drift")
    require(contract["runner_invariants"][
                "every_accepted_request_calls_m1638_accept_request_pair"] and
            contract["runner_invariants"][
                "every_destination_calls_m1638_accept_destination_pair"] and
            contract["claim_boundary"]["paper_result"] is False,
            "contract miter/claim boundary drift")
    private = text[text.index("def _run_bound_actual_prefix():"):
                   text.index("def actual_prefix_release(")]
    require("FUTURE_REVIEW.exists() and FUTURE_RELEASE.exists()" in private and
            "regular_exact(FUTURE_REVIEW" not in private and
            "regular_exact(FUTURE_RELEASE" not in private and
            "verify_tree" not in private and "SHA256SUMS" not in private,
            "presence-only future execution gate characterization drift")


def synthetic_with_tracker(module):
    miter_class = module.L2.CanonicalPrefixMiter
    original_init = miter_class.__init__
    original_request = miter_class.accept_request_pair
    original_destination = miter_class.accept_destination_pair
    original_finish = miter_class.finish
    trackers = []

    def tracked_init(self, configuration):
        original_init(self, configuration)
        trackers.append({"owner": self, "configuration": configuration,
                         "request_pairs": 0, "destination_pairs": 0,
                         "finish_calls": 0})

    def find(self):
        return next(row for row in trackers if row["owner"] is self)

    def tracked_request(self, reference, compact):
        find(self)["request_pairs"] += 1
        return original_request(self, reference, compact)

    def tracked_destination(self, reference, compact):
        find(self)["destination_pairs"] += 1
        return original_destination(self, reference, compact)

    def tracked_finish(self):
        find(self)["finish_calls"] += 1
        return original_finish(self)

    miter_class.__init__ = tracked_init
    miter_class.accept_request_pair = tracked_request
    miter_class.accept_destination_pair = tracked_destination
    miter_class.finish = tracked_finish
    try:
        rss = module.RssGate()
        rows = [module._synthetic_session(configuration, rss)
                for configuration in module.CONFIGS]
    finally:
        miter_class.__init__ = original_init
        miter_class.accept_request_pair = original_request
        miter_class.accept_destination_pair = original_destination
        miter_class.finish = original_finish
    require(len(trackers) == 3, "three persistent sessions missing")
    summaries = []
    for tracker, (receipt, metric), configuration in zip(
            trackers, rows, module.CONFIGS):
        row = receipt.as_dict()
        require(tracker["configuration"] == configuration and
                tracker["request_pairs"] == metric["request_count"] ==
                    row["requests"] and
                tracker["destination_pairs"] == module.PREFIX_DESTINATIONS ==
                    row["destinations"] and
                tracker["finish_calls"] == 1 and row["commits"] == 168 and
                metric["kind_counts"]["commit"] == 168,
                "request/destination/session count drift")
        summaries.append({"configuration": configuration,
            "request_pairs": tracker["request_pairs"],
            "destination_pairs": tracker["destination_pairs"],
            "commits": row["commits"], "session_identity":
            row["session_identity"], "total_cycles_synthetic_only":
            metric["total_cycles"]})
    require(len(set(row["session_identity"] for row in summaries)) == 3,
            "configuration-bound sessions are not distinct")
    return summaries, rss.summary()


def mutation_rejected(module, target, mutation):
    miter_class = module.L2.CanonicalPrefixMiter
    original_request = miter_class.accept_request_pair
    original_destination = miter_class.accept_destination_pair
    fired = [False]

    def mutate_request(self, reference, compact):
        if target == "request" and not fired[0]:
            fired[0] = True
            compact = copy.deepcopy(compact)
            mutation(compact)
        return original_request(self, reference, compact)

    def mutate_destination(self, reference, compact):
        if target == "destination" and not fired[0]:
            fired[0] = True
            compact = copy.deepcopy(compact)
            mutation(compact)
        return original_destination(self, reference, compact)

    miter_class.accept_request_pair = mutate_request
    miter_class.accept_destination_pair = mutate_destination
    try:
        try:
            module._synthetic_session(module.CONFIGS[0], module.RssGate())
        except (module.M1645Error, module.L2.M1638Error):
            return fired[0]
    finally:
        miter_class.accept_request_pair = original_request
        miter_class.accept_destination_pair = original_destination
    return False


def rss_mutations(module):
    original = module.read_rss_kib
    outcomes = []
    try:
        sequence = [(100, 100), (120, 130)]
        module.read_rss_kib = lambda: sequence.pop(0)
        gate = module.RssGate()
        require(gate.summary()["gate_calls"] == 1,
                "RSS normal sequence drift")
        outcomes.append("normal_pass")

        sequence = [(100, 100), (module.RSS_ABSOLUTE_LIMIT_KIB,
                                  module.RSS_ABSOLUTE_LIMIT_KIB)]
        module.read_rss_kib = lambda: sequence.pop(0)
        require(rejected(module, lambda: module.RssGate()),
                "absolute RSS overflow accepted")
        outcomes.append("absolute_rejected")

        sequence = [(100, 100),
                    (100, 100 + module.RSS_INCREMENT_LIMIT_KIB)]
        module.read_rss_kib = lambda: sequence.pop(0)
        require(rejected(module, lambda: module.RssGate()),
                "increment RSS overflow accepted")
        outcomes.append("increment_rejected")
    finally:
        module.read_rss_kib = original
    return outcomes


def main():
    for path, digest in EXPECTED.items():
        verify_regular(path, digest)
    verify_file_seal(CONTRACT)
    verify_tree(AUTHOR)
    verify_tree(M1639)
    require(not FUTURE_RELEASE.exists(), "future release already exists")
    require(not ABSENT_REVIEW_SENTINEL.exists(), "review sentinel exists")

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    m1639 = strict_json(M1639 / "review.json")
    audit_static(SOURCE.read_text(encoding="utf-8"), contract)
    require(author["status"].startswith("PASS_AUTHOR_M1645_") and
            author["paper_claims"] == 0,
            "M1645 author receipt boundary drift")
    require(m1639["status"].startswith("PASS_M1639_M1638_") and
            m1639["authorization"][
                "actual_prefix_runner_source_authoring"] is True and
            m1639["authorization"]["actual_prefix_runner_execution"] is False,
            "M1639 authorization drift")

    module = load_source()
    original_review = module.FUTURE_REVIEW
    require(original_review == REVIEW and REVIEW.exists(),
            "M1646 review identity drift")
    module.FUTURE_REVIEW = ABSENT_REVIEW_SENTINEL
    try:
        result = module.static_self_test()
        summaries, rss = synthetic_with_tracker(module)
    finally:
        module.FUTURE_REVIEW = original_review
    require(result["distinct_sessions"] == 3 and
            result["actual_payload"] is False and
            result["actual_execution"] is False and
            result["cycles_admitted"] is False and
            result["bytes_admitted"] is False and
            result["paper_result"] is False,
            "source-only synthetic boundary drift")

    request_cycle = mutation_rejected(module, "request",
        lambda row: row.update({"issue_cycle": row["issue_cycle"] + 1}))
    request_coordinate = mutation_rejected(module, "request",
        lambda row: row.update({"module": 3}))
    destination_cycle = mutation_rejected(module, "destination",
        lambda row: row.update({"last_cycle": row["last_cycle"] + 1}))
    destination_configuration = mutation_rejected(module, "destination",
        lambda row: row.update({"configuration": module.CONFIGS[1]}))
    require(all((request_cycle, request_coordinate, destination_cycle,
                 destination_configuration)), "M1638 mutation accepted")

    config = module.CONFIGS[0]
    good = module.R.request(config + ":commit:7:2", config, "commit",
                            [0], [0], 384)
    coordinate = module.actual_coordinate(config, good, 11, 7, 2)
    require(coordinate[1] == 0 and coordinate[2] == 0 and
            coordinate[4:6] == (7, 2), "D0 coordinate encoder drift")
    require(rejected(module, lambda: module.actual_coordinate(
                config, good, 11, 8, 2)) and
            rejected(module, lambda: module.actual_coordinate(
                module.FORBIDDEN_CONFIG, {}, 0, 0, 0)) and
            rejected(module, module.actual_prefix_release),
            "coordinate/product/public-release attack accepted")

    output = {
        "schema": "m1646_m1645_decoder_actual_prefix_source_independent_hammer_r1_v1",
        "status": "PASS_M1646_M1645_SOURCE_ONLY_NO_ACTUAL_EXECUTION",
        "python": sys.version.split()[0],
        "identities_checked": len(EXPECTED),
        "contract_seal": "PASS", "author_tree_seal": "PASS",
        "m1639_tree_seal": "PASS",
        "fixed_population": {"decoder_stage": "D0", "call_ordinal": 0,
            "module_ordinal": 0, "timestep": 0,
            "destinations": "0..41", "output_blocks": "0..3",
            "configuration_order": list(module.CONFIGS)},
        "session_summaries": summaries,
        "rss": rss,
        "rss_mutations": rss_mutations(module),
        "miter_mutations": {"request_cycle": request_cycle,
            "request_coordinate": request_coordinate,
            "destination_cycle": destination_cycle,
            "destination_configuration": destination_configuration},
        "precreate_review_absence_witnessed_before_review_directory": True,
        "future_release_absent": not FUTURE_RELEASE.exists(),
        "private_actual_runner_call_sites": 0,
        "p1_presence_only_private_execution_gate": True,
        "release_authorized": False,
        "successor_source_repair_only": True,
        "actual_payload_opened": False, "actual_prefix_executed": False,
        "actual_cycles_admitted": False, "paper_result": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

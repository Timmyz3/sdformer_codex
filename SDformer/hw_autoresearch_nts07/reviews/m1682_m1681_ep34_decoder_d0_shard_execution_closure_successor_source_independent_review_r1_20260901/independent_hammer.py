#!/usr/bin/env python3
"""Read-only different-author hammer for M1681 decoder D0 closure.

The hammer uses only source metadata and synthetic temporary fixtures.  It
never opens a canonical payload, runs a decoder replay, creates an M1683
release, or invokes GPU/EDA work.  Python 3.6 compatible.
"""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / (
    "system_simulator/scripts/build_m1681_ep34_decoder_d0_shard_"
    "execution_closure_successor_source.py")
TEST = HW / (
    "system_simulator/tests/test_m1681_ep34_decoder_d0_shard_"
    "execution_closure_successor_source.py")
CONTRACT = HW / (
    "contracts/m1681_ep34_decoder_d0_shard_execution_closure_successor_"
    "source_contract_r1_20260901.json")
AUTHOR = HW / (
    "reviews/m1681_ep34_decoder_d0_shard_execution_closure_successor_"
    "source_author_receipt_r1_20260901")
M1671 = HW / (
    "system_simulator/scripts/build_m1671_ep34_decoder_d0_recoverable_"
    "shard_successor_source.py")
M1671_TEST = HW / (
    "system_simulator/tests/test_m1671_ep34_decoder_d0_recoverable_"
    "shard_successor_source.py")
M1671_CONTRACT = HW / (
    "contracts/m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_contract_r1_20260901.json")
M1672 = HW / (
    "reviews/m1672_m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_independent_review_r1_20260901")
M1666 = HW / (
    "reviews/m1666_m1656_decoder_actual_prefix_result_independent_"
    "hammer_r1_20260901")
M1683 = HW / (
    "contracts/m1683_m1682_m1681_ep34_decoder_d0_shard_execution_"
    "campaign_release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "006535679b38e2aa207fadde05e9207d2e72dae0464315dceea4a3c96da77a6f",
    "test": "e80c432a88397dc2c10495f8e019be0452fa0e64c150ee05b74d500de57e5721",
    "contract": "3056b9ab52a24e86a98f565cdfe59f3c15f063aaf346477990190a3a9fedddfb",
    "contract_sidecar": "c19ff563afc1c7c6249dbea9f7fb595eabc6dbc5b512749669d9272474ef52e8",
    "contract_outer": "8e52fc9cadb9a135c35edae098e74c8130953871ee5d01016a39bfe81313e59d",
    "author_review": "ec92c4b126497b97b41db33f20d9dae8899552dc1d84786e5e850ecfd5029df5",
    "author_manifest": "64cec1857d1cafb9e32f795357e94e5f1c315d15e04efe35d2b2b35a17dedc50",
    "author_outer": "f966f901d53453f91a24ee641cc2c8017f9238c68fc4b5b3cfa443054e53c24d",
    "m1671": "f6f99909265acac768acf3f1f6340e25d422bde2726cc19b60b4a30c602b8e02",
    "m1671_test": "db1a64ae42b2885f7ebe7bfc7542cab695b63a7e24275da8858d52d98b2675f5",
    "m1671_contract": "5745fd1d1c44507cc20208144c78533bdc6838265cd0611b04cfed23eb90aa6f",
    "m1672_review": "f9d9a1290e8a616940a14db60cc1d50c9f1e2492a0a9a98ee3538991b90b404d",
    "m1672_manifest": "b154ba678a2a4850e3c5665fb734da03dbf74a405cb14fac4cdd5400a81efa5f",
    "m1672_outer": "7608fa6da9dd0ec7a7d33ddfce5645da58aba28ee25934dc689b608a95398e7e",
    "m1666_review": "1acd2380365c1d89750f82cf1623d68ad77147355ebbba7b6d2c83597d6eda29",
    "m1666_manifest": "2bed52d666d9913562bf4370b33c6a9b6528200cd490c1ac3c3585e229213b65",
    "m1666_outer": "d7a4edda6946b065948a85e0cf53bf90df4c06cd281f8069309422b6a685230b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def must(value, message):
    if not value:
        raise HammerError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    must(stat.S_ISREG(mode) and not path.is_symlink(),
         label + " must be regular non-symlink")
    must(sha(path) == expected, label + " SHA drift")


def verify_file_seal(path, payload_sha, sidecar_sha, outer_sha):
    regular_exact(path, payload_sha, "sealed payload")
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular_exact(sidecar, sidecar_sha, "sealed sidecar")
    regular_exact(outer, outer_sha, "sealed outer")
    must(sidecar.read_text(encoding="ascii") ==
         payload_sha + "  " + path.name + "\n", "sidecar content")
    must(outer.read_text(encoding="ascii") ==
         sidecar_sha + "  " + sidecar.name + "\n", "outer content")


def verify_dir_seal(root, review_sha, manifest_sha, outer_sha):
    root = Path(root)
    regular_exact(root / "review.json", review_sha, "review")
    regular_exact(root / "SHA256SUMS", manifest_sha, "manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha, "outer")
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(root),
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=str(root), stdout=subprocess.DEVNULL)


def load_source():
    spec = importlib.util.spec_from_file_location("m1682_exact_m1681",
                                                  str(SOURCE))
    must(spec is not None and spec.loader is not None, "cannot import M1681")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_source()


def synthetic_metric(configuration, shard, cycles, requests, address_digit):
    commits = shard["destination_count"] * M.G.OUTPUT_BLOCKS
    must(requests > commits, "synthetic request population")
    row = {"configuration": configuration,
        "resource_manifest_sha256": M.G.RESOURCE_SHA256,
        "total_cycles": cycles, "request_count": requests,
        "kind_counts": {"commit": commits, "compute": requests - commits},
        "byte_counts": {"commit": commits * M.G.R.OUTPUT_COMMIT_BYTES,
                        "compute": 0},
        "packed_transaction_address_sha256": address_digit * 64,
        "packed_commit_sequence_sha256": "a" * 64,
        "destination_state_chain_sha256": "b" * 64,
        "per_request_miter": True, "per_destination_miter": True,
        "shard_reset_boundary": True, "paper_result": False}
    row["final_state_sha256"] = M.metric_final_state(row, shard)
    return row


def synthetic_receipt(ordinal, attempt_sha, release_sha="c" * 64):
    shard = M.G.shard_descriptor(ordinal)
    commits = shard["destination_count"] * M.G.OUTPUT_BLOCKS
    metrics = [synthetic_metric(configuration, shard, 100 + index * 10,
                                commits + 4 + index, str(index + 1))
               for index, configuration in enumerate(M.CONFIGS)]
    return {"schema": M.RESULT_SCHEMA, "status": M.RESULT_STATUS,
        "source_sha256": sha(SOURCE), "release_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "checkpoint_sha256": M.G.CHECKPOINT_SHA256,
        "resource_manifest_sha256": M.G.RESOURCE_SHA256,
        "shard_ordinal": ordinal, "shard": shard,
        "configuration_order": list(M.CONFIGS), "metrics": metrics,
        "integer_ratio_inputs":
            M.G.validate_three_configuration_metrics(metrics, shard),
        "payload_fd_sha256": "d" * 64, "payload_fd_size": 576000,
        "rss": {"absolute_limit_kib": M.G.RSS_ABSOLUTE_LIMIT_KIB,
            "increment_limit_kib": M.G.RSS_INCREMENT_LIMIT_KIB,
            "gate_calls": 1},
        "automatic_retry": False, "shard_isolated": True,
        "monolithic_full_call": False, "full_decoder": False,
        "system_speedup": False, "paper_result": False,
        "independent_result_hammer_pending": True}


def fixed_namespace_audit():
    all_names = set()
    for ordinal in range(M.G.TOTAL_SHARDS):
        paths = M.namespace_strings(ordinal)
        must(set(paths) == {"result", "attempt", "work", "failure"},
             "namespace keyset")
        for value in paths.values():
            must(value not in all_names, "cross-shard namespace collision")
            all_names.add(value)
    must(len(all_names) == 34800, "fixed namespace population")
    must("0000" in M.namespace_strings(0)["result"] and
         "8699" in M.namespace_strings(8699)["result"],
         "first/last namespace identity")
    return len(all_names)


def call_path_audit():
    text = SOURCE.read_text(encoding="utf-8")
    order = ["release_sha = validate_future_review_and_release()",
        "paths = require_fresh_shard(ordinal)",
        "attempt_sha = consume_attempt(ordinal, release_sha)",
        "G.R.validate_authorities(True)", "record = G.selected_record(shard)",
        "plane = ImmutableTimestepPlane(",
        "metrics = _schedule_actual_shard(shard, plane, rss)",
        "seal_work_tree(paths[\"work\"])",
        "paths[\"work\"].rename(paths[\"result\"])"]
    positions = [text.index(token) for token in order]
    must(positions == sorted(positions) and len(set(positions)) == len(order),
         "authority/attempt/payload/shard/seal/publish reachability order")
    schedule = text[text.index("def _schedule_actual_shard"):
                    text.index("def _run_authorized_shard")]
    for token in ("plane.bit", "contributors_for_destination",
                  "destination_transactions", "session.accept",
                  "session.finish_destination", "session.finish"):
        must(token in schedule, "actual payload-to-shard seam " + token)
    tree = ast.parse(text)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    main_text = "\n".join(text.splitlines()[main.lineno - 1:])
    must("_run_authorized_shard(" not in main_text and
         "reduce_complete_sealed_shards(" not in main_text,
         "private execution target leaked to CLI")
    return order


def immutable_fd_audit():
    shape = tuple(M.G.R.INPUT_SHAPES[0])
    size = 576000
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "synthetic.bitpack"
        raw = bytearray(size)
        timestep_bytes = size // 10
        raw[2 * timestep_bytes] = 1
        path.write_bytes(bytes(raw))
        expected = sha(path)
        plane = M.ImmutableTimestepPlane(path, shape, expected, 2)
        must(plane.opened_sha256 == expected and plane.opened_size == size and
             plane.bit(0, 0, 0) == 1, "immutable FD snapshot")
        rejected = 0
        for bad in ("0" * 64, expected[:-1] + "0"):
            try:
                M.ImmutableTimestepPlane(path, shape, bad, 2)
            except M.M1681Error:
                rejected += 1
        must(rejected == 2, "FD SHA mutations")
    return rejected


def reducer_request_conservation():
    rows = []
    for ordinal in range(2):
        attempt = (str(ordinal + 1) * 64)[:64]
        rows.append(synthetic_receipt(ordinal, attempt))
    old_total, old_verify = M.G.TOTAL_SHARDS, M.verify_sealed_shard
    M.G.TOTAL_SHARDS = 2
    M.verify_sealed_shard = lambda ordinal: {
        "ordinal": ordinal, "row": rows[ordinal],
        "seal": {"manifest_sha256": str(ordinal) * 64},
        "attempt_sha256": rows[ordinal]["attempt_sha256"]}
    try:
        reduced = M.reduce_complete_sealed_shards()
    finally:
        M.G.TOTAL_SHARDS, M.verify_sealed_shard = old_total, old_verify
    expected = dict((configuration, sum(row["metrics"][index]["request_count"]
        for row in rows)) for index, configuration in enumerate(M.CONFIGS))
    actual = dict((configuration,
        reduced["configuration_totals"][configuration]["requests"])
        for configuration in M.CONFIGS)
    must(actual == expected, "reducer request conservation drift")
    return {"expected": expected, "actual": actual,
            "conserved": True}


def sealed_fixture(root, ordinal=0):
    root = Path(root)
    paths = {"result": root / "result", "attempt": root / "attempt",
             "work": root / "work", "failure": root / "failure"}
    paths["attempt"].write_text("attempt\n", encoding="ascii")
    attempt_sha = sha(paths["attempt"])
    row = synthetic_receipt(ordinal, attempt_sha)
    paths["result"].mkdir()
    (paths["result"] / "result.json").write_text(json.dumps(
        row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    M.seal_work_tree(paths["result"])
    return paths, row


def ambiguous_namespace_reproduction():
    with tempfile.TemporaryDirectory() as directory:
        paths, _row = sealed_fixture(directory)
        paths["failure"].mkdir()
        original = M.namespace_paths
        M.namespace_paths = lambda _ordinal: paths
        try:
            accepted = M.verify_sealed_shard(0)
            resume_rejected = False
            old_total = M.G.TOTAL_SHARDS
            M.G.TOTAL_SHARDS = 1
            try:
                M.resume_state()
            except M.M1681Error:
                resume_rejected = True
            finally:
                M.G.TOTAL_SHARDS = old_total
        finally:
            M.namespace_paths = original
        must(accepted["ordinal"] == 0 and resume_rejected,
             "ambiguous namespace reproduction")
    return {"verify_sealed_shard_accepted_result_attempt_failure": True,
            "resume_state_rejected_same_topology": True,
            "reducer_calls_resume_state": False}


def fail_closed_mutations():
    shard = M.G.shard_descriptor(0)
    commits = shard["destination_count"] * M.G.OUTPUT_BLOCKS
    base = synthetic_metric(M.CONFIGS[0], shard, 100, commits + 4, "1")
    cases = []

    def reject(name, mutate):
        row = copy.deepcopy(base)
        mutate(row)
        try:
            M.validate_metric(row, M.CONFIGS[0], shard)
        except M.M1681Error:
            cases.append(name)
            return
        raise HammerError("metric mutation accepted: " + name)

    reject("zero_cycles", lambda row: row.update(total_cycles=0))
    reject("zero_requests", lambda row: row.update(request_count=0))
    reject("negative_requests", lambda row: row.update(request_count=-1))
    reject("boolean_requests", lambda row: row.update(request_count=True))
    reject("negative_kind", lambda row: row["kind_counts"].update(compute=-1))
    reject("kind_conservation", lambda row: row["kind_counts"].update(compute=1))
    reject("negative_byte", lambda row: row["byte_counts"].update(compute=-1))
    reject("commit_count", lambda row: row["kind_counts"].update(commit=0))
    reject("commit_bytes", lambda row: row["byte_counts"].update(commit=0))
    reject("address_digest", lambda row:
           row.update(packed_transaction_address_sha256="bad"))
    reject("commit_digest", lambda row:
           row.update(packed_commit_sequence_sha256="bad"))
    reject("destination_chain", lambda row:
           row.update(destination_state_chain_sha256="bad"))
    reject("final_state", lambda row: row.update(final_state_sha256="0" * 64))

    with tempfile.TemporaryDirectory() as directory:
        paths, _row = sealed_fixture(directory)
        original = M.namespace_paths
        M.namespace_paths = lambda _ordinal: paths
        try:
            M.verify_sealed_shard(0)
            (paths["result"] / "extra").write_text("x", encoding="ascii")
            try:
                M.verify_sealed_shard(0)
            except M.M1681Error:
                cases.append("unsealed_extra_file")
            (paths["result"] / "extra").unlink()
            cache = paths["result"] / "__pycache__"
            cache.mkdir()
            (cache / "x.pyc").write_bytes(b"x")
            try:
                M.verify_sealed_shard(0)
            except M.M1681Error:
                cases.append("result_pycache")
        finally:
            M.namespace_paths = original
    must(len(cases) == 15, "fail-closed mutation count")
    return cases


def main():
    identities = {"source": SOURCE, "test": TEST, "contract": CONTRACT,
        "m1671": M1671, "m1671_test": M1671_TEST,
        "m1671_contract": M1671_CONTRACT,
        "m1672_review": M1672 / "review.json",
        "m1666_review": M1666 / "review.json", "docs359": DOCS359}
    for name, path in identities.items():
        regular_exact(path, EXPECTED[name], name)
    verify_file_seal(CONTRACT, EXPECTED["contract"],
                     EXPECTED["contract_sidecar"], EXPECTED["contract_outer"])
    verify_dir_seal(AUTHOR, EXPECTED["author_review"],
                    EXPECTED["author_manifest"], EXPECTED["author_outer"])
    verify_dir_seal(M1672, EXPECTED["m1672_review"],
                    EXPECTED["m1672_manifest"], EXPECTED["m1672_outer"])
    verify_dir_seal(M1666, EXPECTED["m1666_review"],
                    EXPECTED["m1666_manifest"], EXPECTED["m1666_outer"])
    must(not os.path.lexists(str(M1683)) and
         not os.path.lexists(str(Path(str(M1683) + ".sha256"))) and
         not os.path.lexists(str(Path(str(M1683) + ".sha256.seal.sha256"))),
         "M1683 release must remain absent")
    must(M.G.validate_grid() == {"calls": 30, "timesteps": 300,
        "destinations": 360000, "shards": 8700, "gap_count": 0,
        "overlap_count": 0}, "fixed grid drift")
    namespace_count = fixed_namespace_audit()
    call_order = call_path_audit()
    fd_rejections = immutable_fd_audit()
    request_conservation = reducer_request_conservation()
    namespace_bug = ambiguous_namespace_reproduction()
    mutations = fail_closed_mutations()
    output = {"schema":
        "m1682_m1681_decoder_d0_execution_closure_independent_hammer_r1_v1",
        "status": "FAIL_M1682_M1681_DECODER_D0_EXECUTION_CLOSURE_SOURCE",
        "python": sys.version.split()[0], "exact_identities": len(identities),
        "fixed_namespace_count": namespace_count,
        "actual_payload_to_shard_call_order": call_order,
        "immutable_fd_sha_mutations_rejected": fd_rejections,
        "fail_closed_mutations_rejected": mutations,
        "reducer_request_conservation": request_conservation,
        "p1_ambiguous_namespace": namespace_bug,
        "canonical_payload_opened": False, "replay_executed": False,
        "gpu": False, "eda": False, "release_created": False}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

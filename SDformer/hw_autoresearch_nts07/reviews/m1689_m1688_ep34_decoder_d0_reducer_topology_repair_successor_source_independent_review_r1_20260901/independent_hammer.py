#!/usr/bin/env python3
"""No-payload different-author hammer for the M1688 topology repair."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py"
TEST = HW / "system_simulator/tests/test_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py"
CONTRACT = HW / "contracts/m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_author_receipt_r1_20260901"

EXPECTED = {
    SOURCE: "2ae2725e24c46972f46c54ae71260a8fc637e85c4de0b90f9f91bc42da76abba",
    TEST: "7a331143f6d486939ed77eb34eef60610e450d131313f6df3340cd76290662cb",
    CONTRACT: "10f44a589f986c06f560b0353224b83f5ca6f44e5a0ac73599bd40a8dc85271f",
    Path(str(CONTRACT) + ".sha256"): "d9a08a246a0e5d447b5a30da8cdc2aed41399d185fe94ac536f2bf9a699dc4c1",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "610e6bed1acf43ffb76d9c68db9f5490eb6b1abcd524eea2912b5e43b61b4069",
    AUTHOR / "review.json": "d7a84e8bf33750fbaa2c770e5e787573edc5a307fafc91c5864eaa108875cf28",
    AUTHOR / "SHA256SUMS": "54691aba427f05ce02df1e634abbc4d1cd08f5cb5b2eb96413325b7e8f547a21",
    AUTHOR / "SHA256SUMS.seal.sha256": "54ec8b679df684385b3084676fee27c5b02675c0c2c42848517aea671aefd23a",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    need(spec is not None and spec.loader is not None, "import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(action):
    try:
        action()
    except Exception:
        return True
    return False


def bind(module, mapping):
    old = module.B.namespace_paths
    module.B.namespace_paths = lambda ordinal: mapping[ordinal]
    return old


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift: " + str(path))
    module = load(SOURCE, "m1689_exact_m1688")
    helpers = load(TEST, "m1689_exact_m1688_test_helpers")
    module.verify_m1682_disposition()
    need(module.B.G.TOTAL_SHARDS == 8700 and tuple(module.CONFIGS) == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8"),
         "grid/config drift")
    need(not os.path.lexists(str(module.FORBIDDEN_M1683_RELEASE)),
         "M1683 release revived")

    topology = {}
    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        old = bind(module, {0: paths})
        try:
            verified = module.verify_sealed_shard(0)
            topology["exact_result_attempt_only"] = verified["ordinal"] == 0
        finally:
            module.B.namespace_paths = old

    for name in ("failure", "work"):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = helpers.build_shard(directory, 0)
            paths[name].mkdir()
            old = bind(module, {0: paths})
            try:
                topology["extra_" + name] = rejected(
                    lambda: module.verify_sealed_shard(0))
            finally:
                module.B.namespace_paths = old

    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        paths["attempt"].unlink()
        old = bind(module, {0: paths})
        try:
            topology["missing_attempt"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old

    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        result = paths["result"]
        for child in result.iterdir():
            child.unlink()
        result.rmdir()
        old = bind(module, {0: paths})
        try:
            topology["missing_result"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old

    attempt_attacks = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths, _ = helpers.build_shard(root, 0)
        target = root / "attempt_target"
        paths["attempt"].rename(target)
        paths["attempt"].symlink_to(target)
        old = bind(module, {0: paths})
        try:
            attempt_attacks["symlink"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old
    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        os.chmod(str(paths["attempt"]), 0o600)
        old = bind(module, {0: paths})
        try:
            attempt_attacks["mode_0600"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old
    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        paths["attempt"].unlink()
        paths["attempt"].mkdir()
        os.chmod(str(paths["attempt"]), 0o400)
        old = bind(module, {0: paths})
        try:
            attempt_attacks["directory_0400"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old

    with tempfile.TemporaryDirectory() as directory:
        paths0, _ = helpers.build_shard(directory, 0)
        paths1, _ = helpers.build_shard(directory, 1)
        old_paths = bind(module, {0: paths0, 1: paths1})
        old_total = module.B.G.TOTAL_SHARDS
        module.B.G.TOTAL_SHARDS = 2
        try:
            reduced = module.reduce_complete_sealed_shards()
        finally:
            module.B.G.TOTAL_SHARDS = old_total
            module.B.namespace_paths = old_paths
    requests = dict((config,
        reduced["configuration_totals"][config]["requests"])
        for config in module.CONFIGS)
    need(requests == {"DENSE_TYPED_K8": 344,
                       "BIT_EQUAL_SERVICE_K1X8": 346,
                       "BIT_TYPED_K8": 348}, "request conservation")

    shard = module.B.G.shard_descriptor(0)
    base = [helpers.make_metric(config, shard, request, 200 - index * 20)
            for index, (config, request) in enumerate(zip(
                module.CONFIGS, (172, 173, 174)))]
    module.B.validate_metric_bundle(base, shard)
    metric_mutations = [
        lambda rows: rows[0].update(total_cycles=0),
        lambda rows: rows[0].update(request_count=0),
        lambda rows: rows[0].update(request_count=-1),
        lambda rows: rows[0].update(request_count=True),
        lambda rows: rows[0]["kind_counts"].update(compute=-1),
        lambda rows: rows[0]["kind_counts"].update(compute=5),
        lambda rows: rows[0]["byte_counts"].update(compute=-1),
        lambda rows: rows[0]["kind_counts"].update(commit=167),
        lambda rows: rows[0]["byte_counts"].update(commit=0),
        lambda rows: rows[0].update(packed_transaction_address_sha256="bad"),
        lambda rows: rows[1].update(packed_commit_sequence_sha256="7" * 64),
        lambda rows: rows[0].update(destination_state_chain_sha256="bad"),
        lambda rows: rows[0].update(final_state_sha256="0" * 64),
    ]
    rejected_metrics = 0
    for mutation in metric_mutations:
        rows = copy.deepcopy(base)
        mutation(rows)
        if rejected(lambda: module.B.validate_metric_bundle(rows, shard)):
            rejected_metrics += 1
    need(rejected_metrics == 13, "metric mutation escaped")

    seal_attacks = {}
    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        (paths["result"] / "unsealed").write_text("x")
        old = bind(module, {0: paths})
        try:
            seal_attacks["extra_file"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old
    with tempfile.TemporaryDirectory() as directory:
        paths, _ = helpers.build_shard(directory, 0)
        cache = paths["result"] / "__pycache__"
        cache.mkdir(); (cache / "x.pyc").write_bytes(b"x")
        old = bind(module, {0: paths})
        try:
            seal_attacks["result_pycache"] = rejected(
                lambda: module.verify_sealed_shard(0))
        finally:
            module.B.namespace_paths = old

    tree = ast.parse(SOURCE.read_text())
    reducer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "reduce_complete_sealed_shards")
    calls = [node for node in ast.walk(reducer) if isinstance(node, ast.Call)]
    need(any(isinstance(call.func, ast.Name) and
             call.func.id == "verify_sealed_shard" for call in calls),
         "strong reducer call absent")
    need(not any(isinstance(call.func, ast.Attribute) and
                 isinstance(call.func.value, ast.Name) and
                 call.func.value.id == "B" and
                 call.func.attr == "verify_sealed_shard" for call in calls),
         "weak verifier called")
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "main")
    main_text = "\n".join(SOURCE.read_text().splitlines()[main_node.lineno - 1:])
    need("verify_sealed_shard(" not in main_text and
         "reduce_complete_sealed_shards(" not in main_text,
         "payload/reducer CLI reachability")

    need(all(topology.values()) and all(attempt_attacks.values()) and
         all(seal_attacks.values()), "topology/attempt/seal attack escaped")
    result = {
        "schema": "m1689_m1688_decoder_topology_repair_independent_hammer_r1_v1",
        "status": "PASS_M1689_M1688_DECODER_D0_REDUCER_TOPOLOGY_REPAIR_SOURCE__AUTHORIZE_NEWLY_NUMBERED_RELEASE_AUTHORING_ONLY__NO_EXECUTION",
        "score": 100, "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "verified": {
            "exact_topology": {"result": True, "attempt": True,
                               "work": False, "failure": False},
            "topology_attacks": topology,
            "attempt_attacks": attempt_attacks,
            "request_conservation": requests,
            "existing_metric_attacks_rejected": rejected_metrics,
            "existing_seal_attacks": seal_attacks,
            "strong_verifier_called_by_reducer": True,
            "total_shards_frozen": 8700,
            "m1683_release_forbidden": True,
            "payload_opened": False, "replay_executed": False,
            "reducer_executed_on_real_shards": False,
            "release_created": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

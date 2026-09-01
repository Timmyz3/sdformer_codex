#!/usr/bin/env python3
"""Independent zero-analysis hammer for M1763 layer-private S2 witness source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/analyze_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
TEST = HW / "system_simulator/tests/test_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
CONTRACT = HW / "contracts/m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_author_receipt_r1_20260902"
M1762_FAILURE = HW / "results/m1762_m1756_m1754_m1747_tsbg_shape_failure_receipt_r1_20260901.json"
M1762_REVIEW = HW / "reviews/m1762_m1756_m1754_m1747_tsbg_shape_failure_independent_diagnosis_r1_20260901"
M1747 = HW / "system_simulator/scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py"
M1748 = HW / "reviews/m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_source_hammer_r1_20260901"
M1749 = HW / "contracts/m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_analysis_release_r1_20260901.json"
M1744 = HW / "reviews/m1744_m1707_ep34_tsbg_capture_result_independent_hammer_r1_20260901"
M1765 = HW / "contracts/m1765_m1764_m1763_ep34_tsbg_layer_private_s2_witness_analysis_release_r1_20260902.json"
RESULT = HW / "results/m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902"
WORK = HW / "results/.m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902.work"
ATTEMPT = HW / "results/.m1763_m1707_ep34_tsbg_layer_private_s2_witness_attempt_consumed"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JSON nonregular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_file_seal(path, payload_sha, sidecar_sha, outer_sha):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(path.is_file() and not path.is_symlink(), "payload nonregular")
    need(sha(path) == payload_sha and sha(sidecar) == sidecar_sha
         and sha(outer) == outer_sha, "file seal identity " + str(path))
    need(sidecar.read_text().split() == [payload_sha, path.name], "sidecar")
    need(outer.read_text().split() == [sidecar_sha, sidecar.name], "outer")


def verify_dir(root, review_sha, manifest_sha, outer_sha, primary="review.json"):
    root = Path(root); manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root")
    need(sha(root / primary) == review_sha and sha(manifest) == manifest_sha
         and sha(outer) == outer_sha, "sealed triple " + str(root))
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer seal")
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1); need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             "manifest member drift " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed directory")
        if (path.is_file() and "__pycache__" not in path.parts and path.name not in
                {"SHA256SUMS", "SHA256SUMS.seal.sha256"}):
            actual.add(path.relative_to(root).as_posix())
    need(actual == listed, "sealed population drift")


def load_source():
    spec = importlib.util.spec_from_file_location("m1764_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def must_fail(function):
    try:
        function()
    except Exception:
        return 1
    raise RuntimeError("negative mutation survived")


def geometry_policy(group_counts, output_blocks):
    need(tuple(group_counts) == (6, 12, 24, 48), "G16 geometry mutation")
    need(tuple(output_blocks) == (24, 48, 96, 192),
         "output-block geometry mutation")
    return True


def seal_dir(root, module):
    names = sorted(path.name for path in root.iterdir()
                   if path.is_file() and path.name not in
                   ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(module.sha256(root / name), name)
                                for name in names))
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(module.sha256(manifest)))


def seal_file(path, module):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    sidecar.write_text("{}  {}\n".format(module.sha256(path), path.name))
    outer.write_text("{}  {}\n".format(module.sha256(sidecar), sidecar.name))


def fixtures(module, np):
    group_counts = (6, 12, 24, 48)
    output_blocks = (24, 48, 96, 192)
    layer_ids = (8, 12, 16, 28)
    layers = []
    betas = {}
    samples = []
    calls = []
    sample_id = 0
    for index, (groups, blocks, layer_id) in enumerate(
            zip(group_counts, output_blocks, layer_ids)):
        output_channels = blocks * module.BASE.S2_OUTPUT_TILE
        output_tiles = module.BASE.ceil_div(
            output_channels, module.BASE.TSBG_OUTPUT_TILE)
        row_bytes = module.BASE.GROUP_WIDTH * module.BASE.TSBG_OUTPUT_TILE * 4
        layers.append({"layer_id": layer_id, "target": "FC1",
            "module_name": "fc1_{}".format(layer_id),
            "input_channels": groups * module.BASE.GROUP_WIDTH,
            "output_channels": output_channels, "tokens_per_call": 1,
            "weight_layout": {"row_bytes": row_bytes,
                "base_address": index * 64 * row_bytes,
                "source_group_count": groups,
                "output_tile_count": output_tiles,
                "bank_count": module.BASE.WEIGHT_BANKS}})
        betas[layer_id] = [1] * blocks
        for sequence in ("seq_a", "seq_b"):
            for code in (1, 127):
                samples.append({"global_sample_id": sample_id,
                                "sequence": sequence})
                value = np.zeros((1, groups * module.BASE.GROUP_WIDTH),
                                 dtype=np.int8)
                value[0, 0] = code
                calls.append((sample_id, layer_id, value))
                sample_id += 1
    return layers, samples, betas, calls, group_counts, output_blocks, layer_ids


def direct_decision_hashes(module, layers, betas, calls, np):
    rows = dict((row["layer_id"], row) for row in layers)
    hashes = dict((epsilon, hashlib.sha256())
                  for epsilon in module.BASE.S2_EPSILON_RATIO)
    for sample_id, layer_id, value in calls:
        shaped = value.reshape(value.shape[0], -1, module.BASE.GROUP_WIDTH)
        nnz = (shaped != 0).sum(axis=2).astype(np.int16)
        active = nnz > 0
        magnitude = np.abs(shaped.astype(np.int16)).sum(axis=2).astype(np.int32)
        for epsilon in module.BASE.S2_EPSILON_RATIO:
            metric = module.BASE.s2_fc1_pair_metrics(
                active, nnz, magnitude, rows[layer_id]["output_channels"],
                betas[layer_id], epsilon, np)
            hashes[epsilon].update(struct.pack("<IId", sample_id, layer_id,
                                               float(epsilon)))
            hashes[epsilon].update(metric["decision_payload"])
    return dict((epsilon, value.hexdigest()) for epsilon, value in hashes.items())


def main():
    verify_file_seal(CONTRACT,
        "f9c5bb34025a596f1981e812f0155c87c42442d64aee322318e0567d5a50cc9c",
        "dd0f49a63e9ac36a58f0a3ae04d88a04577ddac85294e0de540165c37df9566a",
        "4eebc370d75dce18a421e5e17f55327f52d9c83c222243965886ab57c4b0b79a")
    verify_dir(AUTHOR,
        "651cf4e31475edbeee4f2ddfb68aebc9cb8d3c1beac40fdae7e9631b0319ee78",
        "c8a87760c0c9e4f667b33b3f7b9a5b2a0ee0aea40f6e32ddd0ed9f83f3b9cbb8",
        "e47c16ad986d6040a23c7842b07f1a2204c8eb34cd07e75e7b2afc411114a8d8",
        primary="author_receipt.json")
    verify_file_seal(M1762_FAILURE,
        "42c6771cf1b585174e0d9b3198392bc6761b18d5324eeb26033043f738d559d7",
        "9d2f2bce905184b56ac41fbdf84e4e40cac869fb8a1d25558cb7aacd952c5987",
        "e0357977836659782990ba193641dfff5425bfe070820212648bede1f1c1e501")
    verify_dir(M1762_REVIEW,
        "ecf3fbfc595efb56b404699f0eacdfb278aa5a7008cf4166005bfacfca0642ff",
        "7e6935f74d9100407695fe7892ab075561dfbf483c587ca6e240c73c3538ba00",
        "18f0862dbca4ce4761e4290448c2a7102f9689203c28531b7236fb45c6863e39")
    verify_dir(M1748,
        "f9c3e152bb10d67a1e0b2421565e0f72469804fab4330dae9c00518b684e1c47",
        "10683d2a63035841ef17572a5ca8b57a98eb260cb5b8c39d8d5eabbfb132e594",
        "d1ba7c36dff713385fc30817877f3228516f9a6fa862805a44e5f7d6355e07cc")
    verify_file_seal(M1749,
        "6114020ab8d4da7c9a7c6f149496ee3efb1e7d19aeff5e34becaf60c1d465806",
        "36b76521e4e994158b30e79eddb15e93150f27aaae7aed2a3ff97ac7eeb6c5fe",
        "29939160f02b0b7b4d548bf055e38a5f45367df65634d1317e18297fcbbb5e0f")
    verify_dir(M1744,
        "d237b3a64cf47313873a84a4749465b7cc7361bd8cf57dde5a0b6275f336dbc7",
        "df15fe385bc7f5eccde2fecd19f5fe478dbc0480653cec5aab208c59a8a6b1f4",
        "40c3e5f2c4a98be985bf225fe6cf3a3cda88c3a32047a372c84ca0608baaf1d2")
    need(sha(M1747) ==
         "3bc48502ab1cccf579cfc65dc0cba2747e5bd38a8a4df82dda3f626f7283683b",
         "M1747 source")
    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 drift")

    contract = strict_json(CONTRACT); author = strict_json(AUTHOR / "author_receipt.json")
    failure = strict_json(M1762_FAILURE); diagnosis = strict_json(M1762_REVIEW / "review.json")
    capture_review = strict_json(M1744 / "review.json")
    predecessor_review = strict_json(M1748 / "review.json")
    predecessor_release = strict_json(M1749)
    need(contract["source"] == {"path": str(SOURCE.relative_to(HW.parent)),
                                "sha256": sha(SOURCE)}
         and contract["test"] == {"path": str(TEST.relative_to(HW.parent)),
                                  "sha256": sha(TEST)}, "source inventory")
    need(author["status"] ==
         "PASS_M1763_AUTHOR_SOURCE_ONLY__READY_FOR_M1764_DIFFERENT_AUTHOR_REVIEW__NO_ANALYSIS",
         "author status")
    need(failure["status"] ==
         "FAILED_CLOSED_DURING_PAYLOAD_REPLAY__S2_HETEROGENEOUS_LAYER_WITNESS_AGGREGATION_BUG__CAPTURE_NOT_AT_FAULT__NO_RETRY"
         and failure["absence_and_budget"]["m1756_authority_consumed"] is True
         and failure["absence_and_budget"]["result_publications"] == 0
         and failure["root_cause"]["s2_cross_layer_witness_aggregation_implicated"] is True
         and failure["root_cause"]["tsbg_algorithm_implicated"] is False,
         "M1762 failure semantics")
    need(diagnosis["status"] ==
         "PASS_FAILURE_DIAGNOSIS__ANALYZER_FALSE_NEGATIVE__CAPTURE_REUSE_ALLOWED__NO_RETRY_AUTHORITY"
         and diagnosis["authorization"]["successor_source_authoring"] is True
         and diagnosis["authorization"]["successor_analysis"] is False,
         "M1762 review semantics")
    need(predecessor_review["status"] ==
         "PASS_M1748_M1747_SOURCE_HAMMER__M1749_RELEASE_MAY_BE_CREATED"
         and predecessor_release["status"] ==
         "AUTHORIZE_ONE_M1747_EP34_TSBG_SCHEMA_IDENTITY_SUCCESSOR_ANALYSIS",
         "M1748/M1749 semantics")
    need(capture_review["status"] ==
         "PASS_M1744_M1707_EP34_TSBG_CAPTURE_RESULT__AUTHORIZE_M1727_ANALYSIS_ONLY"
         and capture_review["verified"]["samples"] == 40
         and capture_review["verified"]["fc_frames"] == 11040
         and capture_review["verified"]["all_frame_headers_order_and_extent"] is True
         and capture_review["verified"]["all_zlib_eof_raw_length_and_crc"] is True
         and capture_review["bindings"]["capture_sha256sums_sha256"] ==
             "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f"
         and capture_review["bindings"]["capture_outer_seal_file_sha256"] ==
             "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85",
         "M1707/M1744 binding")
    need(not any(os.path.lexists(str(path)) for path in
                 (M1765, RESULT, WORK, ATTEMPT)), "premature authority/result namespace")

    module = load_source()
    source_check = module.source_self_check()
    need(source_check["status"] == "PASS_M1763_SOURCE_SELF_CHECK__NO_ANALYSIS"
         and source_check["tsbg_math_changed"] is False
         and source_check["analysis_runs"] == 0
         and source_check["paper_result"] is False,
         "source self-check")
    need(module.BASE.tsbg_pair_metrics is module.M1747.BASE.tsbg_pair_metrics
         and module.DecisionAccumulator.finalize_tsbg_rows is
             module._BASE_DECISION_ACCUMULATOR.finalize_tsbg_rows,
         "TSBG implementation/finalizer drift")

    try:
        import numpy as np
    except ImportError:
        result = {"schema": "m1764_source_only_interpreter_check_r1_v1",
            "status": "PASS_M1764_SOURCE_ONLY_NO_NUMPY",
            "python": sys.version.split()[0], "source_self_check": True,
            "m1762_m1747_m1748_m1749_m1744_m1707_bound": True,
            "fresh_namespaces": True, "analysis_runs": 0,
            "capture_reads": 0, "gpu_runs": 0, "network": False, "eda_runs": 0}
    else:
        layers, samples, betas, calls, groups, blocks, layer_ids = fixtures(module, np)
        accumulator = module.DecisionAccumulator(layers, samples, betas, np)
        for sample_id, layer_id, value in calls:
            accumulator.consume_pair(sample_id, layer_id, value)
        s2_rows = accumulator.finalize_s2_rows()
        tsbg_rows = accumulator.finalize_tsbg_rows()
        need({key: accumulator.s2_hash[key].hexdigest()
              for key in accumulator.s2_hash} ==
             direct_decision_hashes(module, layers, betas, calls, np),
             "S2 decision hash drift")
        need(len(tsbg_rows) > 0 and all(row["compute_work_changed"] is False
             and row["same_capacity_ordinary_lru_baseline"] is True
             for row in tsbg_rows), "TSBG finalization boundary")

        layer_rows = [row for row in s2_rows if row["epsilon_ratio"] == 0.01
                      and row["scope_type"] == "layer"]
        all_row = [row for row in s2_rows if row["epsilon_ratio"] == 0.01
                   and row["scope_type"] == "all"][0]
        sequence_rows = [row for row in s2_rows if row["epsilon_ratio"] == 0.01
                         and row["scope_type"] == "sequence"]
        direct_layer = dict(("fc1_{}".format(layer_id), block)
                            for layer_id, block in zip(layer_ids, blocks))
        need(dict((row["scope"], row["dynamic_same_block_keep_drop_witness_count"])
                  for row in layer_rows) == direct_layer,
             "layer-local witness/output-block weighting")
        need(all_row["dynamic_same_block_keep_drop_witness_count"] == sum(blocks)
             and all_row["layer_private_witness_layers"] == 4,
             "all witness integer sum")
        need(len(sequence_rows) == 2 and all(
             row["dynamic_same_block_keep_drop_witness_count"] == sum(blocks)
             and row["layer_private_witness_layers"] == 4 for row in sequence_rows),
             "sequence witness integer sum")
        seen_groups = sorted(set(int(value["drop"].shape[0])
                                 for value in accumulator.s2_seen.values()))
        need(seen_groups == list(groups), "G16 heterogeneous identity")
        need(all(len(key) == 4 and key[3] in layer_ids
                 for key in accumulator.s2_seen), "layer_id absent from witness key")

        geometry_policy(groups, blocks)
        mutations = {"group_shape": 0, "output_block_geometry": 0,
                     "output_block_isolation": 0, "cross_layer_or": 0,
                     "authority": 0}
        for index in range(4):
            for delta in (-1, 1):
                mutant = list(groups); mutant[index] += delta
                mutations["group_shape"] += must_fail(
                    lambda value=mutant: geometry_policy(value, blocks))
                mutant = list(blocks); mutant[index] += delta
                mutations["output_block_geometry"] += must_fail(
                    lambda value=mutant: geometry_policy(groups, value))
        for index, block in enumerate(blocks):
            key = next(key for key in accumulator.s2_seen
                       if key[0] == 0.01 and key[1] == "all" and key[3] == layer_ids[index])
            before = accumulator.finalize_s2_rows()
            before_count = next(row for row in before if row["epsilon_ratio"] == 0.01
                                and row["scope_type"] == "all")[
                                    "dynamic_same_block_keep_drop_witness_count"]
            accumulator.s2_seen[key]["output_blocks"] = block + 1
            after_count = next(row for row in accumulator.finalize_s2_rows()
                               if row["epsilon_ratio"] == 0.01
                               and row["scope_type"] == "all")[
                                   "dynamic_same_block_keep_drop_witness_count"]
            need(after_count - before_count == 1, "own-layer block mutation isolation")
            accumulator.s2_seen[key]["output_blocks"] = block
            mutations["output_block_isolation"] += 1
        padded_first_multiplier = 4 * blocks[0]
        need(padded_first_multiplier != sum(blocks),
             "padding/first-multiplier mutation unexpectedly equivalent")
        mutations["cross_layer_or"] += 1
        need("np.pad(drop" not in SOURCE.read_text()
             and "np.pad(keep" not in SOURCE.read_text(), "padding repair survived")

        ids = module.identities()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); review = root / "review"; review.mkdir()
            review_doc = {"schema": module.REVIEW_SCHEMA,
                "status": module.REVIEW_STATUS, "identity": ids,
                "authorization": {"m1765_release_may_be_created": True,
                    "analysis_run": False, "capture_verify": False},
                "claim_boundary": {"paper_result": False}}
            (review / "review.json").write_text(json.dumps(review_doc, sort_keys=True))
            seal_dir(review, module); binding = module.validate_future_review(review, ids)
            release = root / "release.json"; release_ids = dict(ids)
            release_ids.update({"m1764_review_sha256": binding["review_sha256"],
                "m1764_review_outer_seal_file_sha256":
                    binding["outer_seal_file_sha256"]})
            release_doc = {"schema": module.RELEASE_SCHEMA,
                "status": module.RELEASE_STATUS, "identity": release_ids,
                "authorization": {"analysis_runs": 1, "capture_verifications": 1,
                    "result_publications": 1, "attempts": 1,
                    "automatic_retry": False, "gpu_runs": 0, "eda_runs": 0,
                    "all_other_runs": 0}, "claim_boundary": {"paper_result": False}}
            release.write_text(json.dumps(release_doc, sort_keys=True)); seal_file(release, module)
            module.validate_future_release(release, binding, ids)
            for key, value in (("attempts", 2), ("analysis_runs", 2),
                               ("capture_verifications", 0), ("result_publications", 2)):
                mutant = json.loads(json.dumps(release_doc))
                mutant["authorization"][key] = value
                release.write_text(json.dumps(mutant, sort_keys=True)); seal_file(release, module)
                mutations["authority"] += must_fail(
                    lambda: module.validate_future_release(release, binding, ids))
        need(mutations == {"group_shape": 8, "output_block_geometry": 8,
                           "output_block_isolation": 4,
                           "cross_layer_or": 1, "authority": 4},
             "mutation count")
        result = {"schema": "m1764_numeric_interpreter_hammer_r1_v1",
            "status": "PASS_M1764_M1763_SOURCE_HAMMER__M1765_RELEASE_MAY_BE_CREATED",
            "python": sys.version.split()[0], "numpy": np.__version__,
            "source_self_check": True,
            "m1762_m1747_m1748_m1749_m1744_m1707_bound": True,
            "source_groups_16": list(groups), "output_blocks_16": list(blocks),
            "layer_ids": list(layer_ids), "pairs": len(calls),
            "tsbg_pair_math_identity": True, "tsbg_finalize_identity": True,
            "s2_decision_hash_identity": True,
            "layer_private_witness_identity": True,
            "all_witness": all_row["dynamic_same_block_keep_drop_witness_count"],
            "sequence_witnesses": [row["dynamic_same_block_keep_drop_witness_count"]
                                   for row in sequence_rows],
            "mutations_rejected": mutations,
            "fresh_namespaces": True, "analysis_runs": 0, "capture_reads": 0,
            "gpu_runs": 0, "network": False, "eda_runs": 0}
    output = HERE / ("cpython" + str(sys.version_info[0]) +
                     str(sys.version_info[1]) + "_hammer.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

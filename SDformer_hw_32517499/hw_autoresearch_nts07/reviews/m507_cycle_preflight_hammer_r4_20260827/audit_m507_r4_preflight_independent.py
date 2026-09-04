#!/usr/bin/env python3
"""Independent receipt-blind static audit for M507 r4.

This program never imports or executes the production analyzer and never opens
or decompresses trace payloads.  It audits source/contract identity, frozen
metadata and review seals, required source structure, and an independently
implemented resource/cycle algebra on small synthetic boundary cases.
"""

import ast
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / (
    "system_simulator/scripts/"
    "analyze_m507_h67_apec_g2_same_resource_cycle_fastkill_r4.py")
CONTRACT = ROOT / (
    "contracts/"
    "m507_h67_apec_g2_same_resource_cycle_fastkill_contract_r4_20260827.json")

EXPECTED_ANALYZER_SHA256 = (
    "13db92a7094ba6acce168be0f0c070318c76726edb28fb4bfa3db903302e4968")
EXPECTED_CONTRACT_SHA256 = (
    "241ae6c8a5f2194e14a0573099a9d574003197c1c4a9c01626ecf1e81f2f3a5a")
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def verify_review_manifest(review_dir):
    failures = []
    members = []
    manifest = review_dir / "SHA256SUMS"
    if not manifest.is_file():
        return {"complete": False, "failures": ["missing SHA256SUMS"],
                "members": []}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = raw.split(None, 1)
        name = name.strip()
        local_member = review_dir / name
        root_member = ROOT / name
        member = local_member if local_member.is_file() else root_member
        members.append(name)
        if not member.is_file():
            failures.append("missing:" + name)
        elif sha256(member) != digest:
            failures.append("sha:" + name)
    return {"complete": not failures, "failures": failures,
            "members": members}


def blocks(taps, model):
    return math.ceil(model["output_channels"] * taps /
                     model["compute_lanes"])


def logical_vector_bytes(taps, model):
    return math.ceil(model["output_channels"] * taps *
                     model["accumulator_bits"] / 8)


def commit_cycles(taps, model):
    slot = blocks(taps, model)
    sink = math.ceil(logical_vector_bytes(taps, model) /
                     (model["output_banks"] *
                      model["output_bank_bytes_per_cycle"]))
    return max(slot, sink) + model["destination_slot_sync_read_latency_cycles"]


def stream_service(events, taps, model):
    compute = events * blocks(taps, model)
    weight = math.ceil(events * model["output_channels"] * taps /
                       model["weight_bytes_per_cycle"])
    return compute, weight


def boundary_group(count0, count1, common, taps0, taps1, union_taps,
                   model):
    if not (0 <= common <= min(count0, count1)):
        raise RuntimeError("invalid synthetic overlap")
    b_compute0, b_weight0 = stream_service(count0, taps0, model)
    b_compute1, b_weight1 = stream_service(count1, taps1, model)
    b_events = count0 + count1
    b_exec = max(b_compute0 + b_compute1,
                 b_weight0 + b_weight1 + (1 if b_events else 0))
    b_materialize = ((blocks(taps0, model) if count0 else 0) +
                     (blocks(taps1, model) if count1 else 0))
    b_commit = ((commit_cycles(taps0, model) if count0 else 0) +
                (commit_cycles(taps1, model) if count1 else 0))
    baseline = model["bitmap_pair_read_cycles"] + b_exec + \
        b_materialize + b_commit

    residual0 = count0 - common
    residual1 = count1 - common
    c_compute0, c_weight0 = stream_service(residual0, taps0, model)
    c_compute1, c_weight1 = stream_service(residual1, taps1, model)
    c_computec, c_weightc = stream_service(common, union_taps, model)
    c_events = residual0 + residual1 + common
    c_exec = max(c_compute0 + c_compute1 + c_computec,
                 c_weight0 + c_weight1 + c_weightc +
                 (1 if c_events else 0))
    c_materialize = b_materialize
    c_commit = b_commit
    scratch = 0
    if common:
        block_count = blocks(union_taps, model)
        transfer = block_count * math.ceil(
            model["compute_lanes"] * model["accumulator_bits"] / 8 /
            model["scratch_bytes_per_cycle"])
        one_read = (transfer + block_count *
                    model["overlap_scratch_sync_read_latency_cycles"])
        scratch = transfer + 2 * one_read
    candidate = (model["bitmap_pair_read_cycles"] +
                 model["exact_compare_cycles"] + c_exec + scratch +
                 c_materialize + c_commit)
    return {"baseline": baseline, "candidate": candidate,
            "ratio": baseline / candidate}


def main():
    assert sha256(ANALYZER) == EXPECTED_ANALYZER_SHA256
    assert sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256
    source = ANALYZER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required_functions = {
        "destination_slot_terms", "vector_transfer_terms",
        "destination_block_write_terms", "lane_block_terms",
        "scratch_block_terms", "build_resource_ledger", "record_cycles",
        "analyze_cohort", "reconcile_m501_cohort", "write_seal", "main",
    }
    assert required_functions <= function_names
    contract = strict_json(CONTRACT)
    assert contract["schema"].endswith("contract_v4")
    assert contract["inputs"]["analyzer"]["sha256"] == sha256(ANALYZER)

    outer_sha = {}
    seals = {}
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        assert path.is_file(), (name, path)
        observed = sha256(path)
        assert observed == spec["sha256"], (name, observed, spec["sha256"])
        outer_sha[name] = observed
        if "sealed_manifest_sha256" in spec:
            token = path.read_text(encoding="utf-8").split()[0]
            assert token == spec["sealed_manifest_sha256"], name
            seals[name] = {
                "outer_file_sha256": observed,
                "inner_manifest_sha256": token,
            }
    assert outer_sha["docs359"] == EXPECTED_DOCS359_SHA256

    for review in ("r1", "r2", "r3"):
        seal_name = "m507_{}_preflight_review_seal".format(review)
        review_dir = ROOT / (
            "reviews/m507_cycle_preflight_hammer_{}_20260827".format(review))
        verification = verify_review_manifest(review_dir)
        seals[seal_name]["current_inner_manifest_complete"] = verification[
            "complete"]
        seals[seal_name]["current_inner_manifest_failures"] = verification[
            "failures"]
    assert not seals["m507_r1_preflight_review_seal"][
        "current_inner_manifest_complete"]
    assert seals["m507_r2_preflight_review_seal"][
        "current_inner_manifest_complete"]
    assert seals["m507_r3_preflight_review_seal"][
        "current_inner_manifest_complete"]

    model = contract["cycle_model"]
    full_blocks = blocks(9, model)
    bitmap = math.ceil(2 * model["input_channels"] / 8)
    block_logical = math.ceil(model["compute_lanes"] *
                              model["accumulator_bits"] / 8)
    scratch_words_per_block = math.ceil(
        block_logical / model["scratch_bytes_per_cycle"])
    scratch = (full_blocks * scratch_words_per_block *
               model["scratch_bytes_per_cycle"])
    one_destination = (full_blocks * model["destination_slot_banks"] *
                       model["destination_slot_bank_bytes_per_cycle"])
    destinations = 2 * one_destination
    payload = model["common_total_sram_bytes"] - bitmap - scratch - destinations
    assert bitmap == model["pair_bitmap_buffer_bytes"] == 192
    assert scratch == model["reserved_overlap_scratch_bytes"] == 18432
    assert destinations == model["destination_vector_slots_bytes"] == 36864
    assert payload == model["payload_and_weight_window_bytes"] == 190272
    assert bitmap + scratch + destinations + payload == 245760
    lane_bank_logical = math.ceil(
        (model["compute_lanes"] // model["destination_slot_banks"]) *
        model["accumulator_bits"] / 8)
    assert lane_bank_logical == 29
    assert lane_bank_logical <= model["destination_slot_bank_bytes_per_cycle"]
    assert logical_vector_bytes(9, model) == 16416
    assert one_destination == 18432
    assert commit_cycles(9, model) == 130

    required_source_anchors = [
        "cycles = max(slot[\"cycles\"], sink_cycles) + read_tail",
        "event_read_modify_write\": False",
        "first_product_initializes_nonempty_block\": True",
        "one_read_cycles\": transfer_cycles + blocks * read_latency",
        "validation_rows + train_rows",
        "staging_dir = Path(tempfile.mkdtemp(",
        "write_seal(staging_dir, [result_name, csv_name, seq_name, readme_name,",
        "completion_name])",
        "os.replace(staging_dir, args.output_dir)",
        "require(not args.output_dir.exists(), \"refusing M507 overwrite\")",
    ]
    for anchor in required_source_anchors:
        assert anchor in source, anchor
    assert "import analyze_m507" not in source

    cases = {
        "empty_interior": boundary_group(0, 0, 0, 9, 9, 9, model),
        "one_each_full_overlap_interior": boundary_group(
            1, 1, 1, 9, 9, 9, model),
        "one_each_no_overlap_interior": boundary_group(
            1, 1, 0, 9, 9, 9, model),
        "asymmetric_partial_overlap_interior": boundary_group(
            2, 3, 1, 9, 9, 9, model),
        "one_each_full_overlap_left_border": boundary_group(
            1, 1, 1, 6, 9, 9, model),
        "one_each_full_overlap_top_left": boundary_group(
            1, 1, 1, 4, 6, 6, model),
    }
    print(json.dumps({
        "identity": {
            "analyzer_sha256": sha256(ANALYZER),
            "contract_sha256": sha256(CONTRACT),
            "docs359_sha256": outer_sha["docs359"],
            "all_contract_input_outer_sha_match": True,
        },
        "review_seals": seals,
        "resource_recompute": {
            "pair_bitmap_bytes": bitmap,
            "overlap_cache_bytes": scratch,
            "one_destination_slot_bytes": one_destination,
            "two_destination_slots_bytes": destinations,
            "payload_and_weight_window_bytes": payload,
            "total_bytes": bitmap + scratch + destinations + payload,
            "full_blocks": full_blocks,
            "lane_block_logical_bytes": block_logical,
            "lane_block_physical_scratch_bytes": (
                scratch_words_per_block * model["scratch_bytes_per_cycle"]),
            "lane_demand_logical_bytes_per_destination_bank":
                lane_bank_logical,
            "full_vector_logical_bytes": logical_vector_bytes(9, model),
            "full_vector_physical_destination_bytes": one_destination,
            "full_vector_commit_cycles": commit_cycles(9, model),
        },
        "independent_boundary_cycles": cases,
        "scope": {
            "production_imported": False,
            "production_main_executed": False,
            "payload_opened_or_decompressed": False,
            "vcs_dc_gpu_started": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

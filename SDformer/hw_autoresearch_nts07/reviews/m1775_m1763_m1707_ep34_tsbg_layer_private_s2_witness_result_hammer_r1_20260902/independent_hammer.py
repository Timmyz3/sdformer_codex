#!/usr/bin/env python3
"""Read-only M1775 hammer for the sealed M1763 diagnostic result.

This script does not replay M1763 analysis.  It independently verifies the
transferred result tree, its authority chain, CSV/JSON equivalence, integer
aggregation, roofline/state formulas, the epsilon axis, and the epsilon-zero
decision hash.  Nonzero-epsilon bit decisions require the absent M1707 binary
capture; the limitation is deliberately emitted rather than papered over.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902"
ATTEMPT = HW / "results/.m1763_m1707_ep34_tsbg_layer_private_s2_witness_attempt_consumed"
SOURCE = HW / "system_simulator/scripts/analyze_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
TEST = HW / "system_simulator/tests/test_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
CONTRACT = HW / "contracts/m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_contract_r1_20260902.json"
RELEASE = HW / "contracts/m1765_m1764_m1763_ep34_tsbg_layer_private_s2_witness_analysis_release_r1_20260902.json"
M1764 = HW / "reviews/m1764_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_hammer_r1_20260902"
M1744 = HW / "reviews/m1744_m1707_ep34_tsbg_capture_result_independent_hammer_r1_20260901"
M1763_AUTHOR = HW / "reviews/m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_author_receipt_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "f86bbb02da1e259626539e83969c9e42cf9b34b7ee84c5072ffb9f6c3f70646c",
    "test": "110b7030512df840678fe265c12529d6b35b3b4752c9de4fe818102537739271",
    "contract": "f9c5bb34025a596f1981e812f0155c87c42442d64aee322318e0567d5a50cc9c",
    "release": "8fb394cf6bff01e743a2527cc38dd593defa4ac10628185a8145671f956368b8",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m1764": (
        "16b6825344526643d5cbcc2fa3eb79c9d5b75c369410dae28c79f0070e8e5730",
        "9eb74750cff3b95cc05e29a9f3f7d6b73ad4e821d887fa3be3e07409270bdd55",
        "b071027ef31dd80c0ab3ca957ea51e227b2784d71a5aeaa6bc71a5bd843cd881",
    ),
    "m1744": (
        "d237b3a64cf47313873a84a4749465b7cc7361bd8cf57dde5a0b6275f336dbc7",
        "df15fe385bc7f5eccde2fecd19f5fe478dbc0480653cec5aab208c59a8a6b1f4",
        "40c3e5f2c4a98be985bf225fe6cf3a3cda88c3a32047a372c84ca0608baaf1d2",
    ),
    "m1763_author": (
        "651cf4e31475edbeee4f2ddfb68aebc9cb8d3c1beac40fdae7e9631b0319ee78",
        "c8a87760c0c9e4f667b33b3f7b9a5b2a0ee0aea40f6e32ddd0ed9f83f3b9cbb8",
        "e47c16ad986d6040a23c7842b07f1a2204c8eb34cd07e75e7b2afc411114a8d8",
    ),
    "result": (
        "722aa302c983b63eae4e40816cffd123d0da34b09df56b872d52502d18cee961",
        "e70b0f837bdaa24b0345193e5c0048f2ce14cb803b9106871c91aac2b8d48332",
        "9e08414fcb185fe8cd1251ca758d8319dab84fd7dd8367b55f29f2b02da8ff2e",
    ),
}

SEQUENCES = (
    "interlaken_01_a", "thun_01_b", "zurich_city_09_a",
    "zurich_city_12_a",
)
EPSILON = (0.0, 0.01, 0.02, 0.05, 0.10)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          HammerError("nonfinite JSON: " + value)))


def regular(path, expected=None):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            str(path) + " is not a regular non-symlink")
    actual = sha256(path)
    if expected is not None:
        require(actual == expected, str(path) + " SHA drift")
    return actual


def verify_sidecar(path, expected=None):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    digest = regular(path, expected)
    regular(sidecar)
    regular(outer)
    require(sidecar.read_text(encoding="ascii").split() == [digest, path.name],
            "sidecar mismatch: " + str(path))
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sidecar), sidecar.name], "outer sidecar mismatch")
    return (digest, sha256(sidecar), sha256(outer))


def verify_sealed_dir(root, expected=None, payload="review.json",
                      exact_population=False):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "sealed dir missing")
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(sums)
    regular(outer)
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sums), sums.name], "directory outer seal mismatch")
    names = []
    for line in sums.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and HEX64.match(fields[0]),
                "malformed SHA256SUMS line")
        name = fields[1].strip().lstrip("*")
        require(name not in names and name not in (sums.name, outer.name) and
                not name.startswith("/") and ".." not in Path(name).parts,
                "unsafe/duplicate manifest member")
        regular(root / name, fields[0])
        names.append(name)
    actual_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                          if p.is_file() and p.name not in (sums.name, outer.name))
    if exact_population:
        require(sorted(names) == actual_files, "manifest population mismatch")
    triple = (sha256(root / payload), sha256(sums), sha256(outer))
    if expected is not None:
        require(triple == expected, "sealed directory triple drift")
    return {"payload_sha256": triple[0], "manifest_sha256": triple[1],
            "outer_seal_file_sha256": triple[2], "members": names}


def parse_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) ==
                len(set(reader.fieldnames)), "CSV duplicate header")
        rows = list(reader)
    require(all(set(row) == set(reader.fieldnames) for row in rows),
            "CSV row width drift")
    return reader.fieldnames, rows


def compare_csv_value(text, expected):
    if expected is None:
        return text == ""
    if type(expected) is bool:
        return text.lower() == str(expected).lower()
    if type(expected) is int:
        return int(text) == expected
    if type(expected) is float:
        return math.isfinite(float(text)) and math.isclose(
            float(text), expected, rel_tol=1e-14, abs_tol=1e-14)
    return text == str(expected)


def crosscheck_csv(csv_rows, json_rows, key_fields):
    by_key = {}
    for row in json_rows:
        key = tuple(str(row[key]) for key in key_fields)
        require(key not in by_key, "duplicate JSON row key")
        by_key[key] = row
    require(len(csv_rows) == len(by_key), "CSV/JSON population mismatch")
    for row in csv_rows:
        key = tuple(row[key] for key in key_fields)
        require(key in by_key, "CSV row key absent from JSON")
        expected = by_key[key]
        for field, value in row.items():
            require(field in expected and compare_csv_value(value, expected[field]),
                    "CSV/JSON mismatch: {} {}".format(key, field))


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def verify_tsbg(decision):
    rows = decision["tsbg"]["rows"]
    require(decision["tsbg"]["bundles"] == [4, 8] and len(rows) == 62,
            "TSBG population drift")
    additive = (
        "tokens", "bundle_count", "baseline_weight_row_accesses",
        "baseline_weight_row_hits", "baseline_weight_row_fetches",
        "candidate_weight_row_accesses", "candidate_weight_row_hits",
        "candidate_weight_row_fetches", "baseline_weight_fetch_bytes",
        "candidate_weight_fetch_bytes", "compute_issue_cycles",
        "commit_cycles", "baseline_weight_cycles", "candidate_weight_cycles",
        "baseline_schedule_cycles", "candidate_schedule_cycles",
        "baseline_roofline_cycles", "candidate_roofline_cycles",
    )
    report = []
    for bundle in (4, 8):
        all_row = [row for row in rows if row["bundle"] == bundle and
                   row["scope_type"] == "all"]
        sequences = [row for row in rows if row["bundle"] == bundle and
                     row["scope_type"] == "sequence"]
        require(len(all_row) == 1 and len(sequences) == 4 and
                tuple(sorted(row["scope"] for row in sequences)) ==
                tuple(sorted(SEQUENCES)), "TSBG sequence cohort drift")
        all_row = all_row[0]
        for field in additive:
            require(all_row[field] == sum(row[field] for row in sequences),
                    "TSBG integer aggregate drift: " + field)
        for field in ("baseline_weight_bank_bytes", "candidate_weight_bank_bytes"):
            require(len(all_row[field]) == 8 and all_row[field] == [
                sum(row[field][index] for row in sequences) for index in range(8)],
                "TSBG bank aggregate drift")
        checked = []
        for row in [all_row] + sorted(sequences, key=lambda item: item["scope"]):
            require(row["baseline_weight_row_accesses"] ==
                    row["baseline_weight_row_hits"] +
                    row["baseline_weight_row_fetches"], "baseline LRU drift")
            require(row["candidate_weight_row_accesses"] ==
                    row["candidate_weight_row_hits"] +
                    row["candidate_weight_row_fetches"], "candidate LRU drift")
            require(row["baseline_weight_fetch_bytes"] ==
                    sum(row["baseline_weight_bank_bytes"]) and
                    row["candidate_weight_fetch_bytes"] ==
                    sum(row["candidate_weight_bank_bytes"]), "bank byte drift")
            for prefix in ("baseline", "candidate"):
                weight_cycles = max(
                    ceil_div(row[prefix + "_weight_fetch_bytes"], 128),
                    max(ceil_div(value, 16) for value in
                        row[prefix + "_weight_bank_bytes"]))
                # Scope rows sum per-pair bank-limited cycles.  Re-applying
                # max() after aggregating banks is only a lower bound because
                # max and sum do not commute.
                require(row[prefix + "_weight_cycles"] >= weight_cycles,
                        prefix + " aggregate weight-cycle lower-bound drift")
                roof = max(row["compute_issue_cycles"], row["commit_cycles"],
                           row[prefix + "_weight_cycles"],
                           row[prefix + "_schedule_cycles"])
                require(row[prefix + "_roofline_cycles"] == roof,
                        prefix + " roofline drift")
            reduction = 1.0 - (float(row["candidate_weight_fetch_bytes"]) /
                               float(row["baseline_weight_fetch_bytes"]))
            ratio = (float(row["baseline_weight_fetch_bytes"]) /
                     float(row["candidate_weight_fetch_bytes"]))
            speedup = (float(row["baseline_roofline_cycles"]) /
                       float(row["candidate_roofline_cycles"]))
            require(math.isclose(row["weight_fetch_reduction"], reduction,
                                 rel_tol=1e-14) and
                    math.isclose(row["weight_fetch_ratio"], ratio,
                                 rel_tol=1e-14) and
                    math.isclose(row["roofline_cycle_speedup"], speedup,
                                 rel_tol=1e-14), "TSBG ratio drift")
            baseline_acc = 96 * 3
            baseline_fifo = 16
            candidate_acc = bundle * 96 * 3
            candidate_fifo = bundle * 16
            incremental = candidate_acc + candidate_fifo - baseline_acc - baseline_fifo
            require(row["baseline_acc24_context_bytes_lower_bound"] == baseline_acc and
                    row["baseline_source_fifo_bytes_lower_bound"] == baseline_fifo and
                    row["candidate_b_token_acc24_context_bytes_lower_bound"] == candidate_acc and
                    row["candidate_b_token_source_fifo_bytes_lower_bound"] == candidate_fifo and
                    row["candidate_incremental_state_bytes_lower_bound"] == incremental,
                    "TSBG state lower bound drift")
            require(row["same_resource_claim"] is False and
                    row["screening_only"] is True and
                    row["context_tag_and_broadcast_control_priced"] is False and
                    row["full_area_energy_pricing_complete"] is False and
                    row["fetch_ratio_is_cycle_speedup"] is False,
                    "TSBG claim boundary drift")
            checked.append({
                "scope": row["scope"],
                "baseline_weight_fetch_bytes": row["baseline_weight_fetch_bytes"],
                "candidate_weight_fetch_bytes": row["candidate_weight_fetch_bytes"],
                "baseline_roofline_cycles": row["baseline_roofline_cycles"],
                "candidate_roofline_cycles": row["candidate_roofline_cycles"],
                "roofline_ratio_screening_only": speedup,
                "candidate_incremental_state_bytes_lower_bound": incremental,
            })
        report.append({"bundle": bundle, "all_and_four_sequences": checked})
    require(all(item["cycle_path_admitted"] is False and
                item["energy_only_path_eligible"] is False
                for item in decision["tsbg"]["decisions"]),
            "TSBG decision unexpectedly admitted")
    return report


def epsilon_zero_hash():
    # Exact M1558 inventory, independently transcribed and bound by its frozen
    # source through M1747/M1763.  Drop payload is identically zero at epsilon 0.
    fc1 = (
        (8, 96, 192000), (10, 96, 192000),
        (12, 192, 48000), (14, 192, 48000),
        (16, 384, 12000), (18, 384, 12000),
        (20, 384, 12000), (22, 384, 12000),
        (24, 384, 12000), (26, 384, 12000),
        (28, 768, 3000), (30, 768, 3000),
    )
    digest = hashlib.sha256()
    for sample_id in range(40):
        for layer_id, channels, tokens in fc1:
            groups = ceil_div(channels, 16)
            digest.update(struct.pack("<IId", sample_id, layer_id, 0.0))
            digest.update(b"\x00" * (tokens * ceil_div(groups, 8)))
    return digest.hexdigest()


def verify_s2(decision):
    s2 = decision["s2"]
    rows = s2["fc1_rows"]
    require(s2["geometry"] == "16x16" and
            tuple(s2["epsilon_ratio_axis"]) == EPSILON and len(rows) == 85,
            "S2 axis/population drift")
    sum_fields = (
        "tokens", "baseline_nonzero_blocks", "kept_blocks", "dropped_blocks",
        "metadata_bytes", "baseline_weight_bytes", "saved_weight_bytes",
        "baseline_nonzero_products", "saved_nonzero_products",
        "saved_psum_update_events", "sum_abs_output_code_debt",
        "dynamic_same_block_keep_drop_witness_count",
    )
    summaries = []
    canonical_row_hashes = {}
    for epsilon in EPSILON:
        subset = [row for row in rows if row["epsilon_ratio"] == epsilon]
        all_rows = [row for row in subset if row["scope_type"] == "all"]
        sequences = [row for row in subset if row["scope_type"] == "sequence"]
        layers = [row for row in subset if row["scope_type"] == "layer"]
        require(len(all_rows) == 1 and len(sequences) == 4 and len(layers) == 12 and
                tuple(sorted(row["scope"] for row in sequences)) ==
                tuple(sorted(SEQUENCES)), "S2 scope population drift")
        all_row = all_rows[0]
        for field in sum_fields:
            require(all_row[field] == sum(row[field] for row in sequences) and
                    all_row[field] == sum(row[field] for row in layers),
                    "S2 integer aggregate drift: " + field)
        for field in ("max_dropped_block_abs_output_code_debt",
                      "max_accumulated_abs_output_code_debt_per_token"):
            require(all_row[field] == max(row[field] for row in sequences) and
                    all_row[field] == max(row[field] for row in layers),
                    "S2 maximum aggregate drift: " + field)
        threshold = int(math.floor(epsilon * 16 * 127.0 + 1.0e-12))
        require(all(row["threshold_abs_code_sum"] == threshold for row in subset),
                "S2 threshold drift")
        for row in subset:
            require(row["paired_aee_present"] is False and
                    row["overall_delta_aee"] is None and
                    row["max_sequence_delta_aee"] is None and
                    row["same_resource_cycle_speedup"] is None and
                    row["passes_fixed_gate"] is False and
                    row["paper_admission"] is False and
                    row["layer_private_witness_identity"] is True,
                    "S2 admission boundary drift")
        if epsilon == 0.0:
            require(all_row["dropped_blocks"] == 0 and
                    all_row["kept_blocks"] == all_row["baseline_nonzero_blocks"] and
                    all_row["metadata_bytes"] == 0,
                    "S2 epsilon-zero subset drift")
        else:
            # M1744 independently established that all nonzero captured codes
            # are unit magnitude.  Thus a 16-wide group has magnitude <=16,
            # strictly below the smallest nonzero threshold (20).
            require(threshold >= 20 and
                    all_row["max_dropped_block_abs_output_code_debt"] <= 16 and
                    all_row["kept_blocks"] == 0 and
                    all_row["dropped_blocks"] ==
                    all_row["baseline_nonzero_blocks"] and
                    all_row["drop_fraction_of_remaining_nonzero_blocks"] == 1.0 and
                    all_row["saved_nonzero_products"] ==
                    all_row["baseline_nonzero_products"] and
                    all_row["dynamic_same_block_keep_drop_witness_count"] == 0,
                    "S2 unit-code all-drop implication drift")
        encoded = json.dumps(subset, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("ascii")
        canonical_row_hashes[str(epsilon)] = hashlib.sha256(encoded).hexdigest()
        summaries.append({
            "epsilon_ratio": epsilon,
            "threshold_abs_code_sum": threshold,
            "baseline_nonzero_blocks": all_row["baseline_nonzero_blocks"],
            "kept_blocks": all_row["kept_blocks"],
            "dropped_blocks": all_row["dropped_blocks"],
            "drop_fraction": all_row["drop_fraction_of_remaining_nonzero_blocks"],
            "max_dropped_group_abs_code_sum":
                all_row["max_dropped_block_abs_output_code_debt"],
            "paired_aee_present": False,
            "paper_admission": False,
        })
    hashes = s2["fc1_decision_sha256"]
    require(set(hashes) == {str(value) for value in EPSILON} and
            all(HEX64.match(value) for value in hashes.values()) and
            len(set(hashes.values())) == len(EPSILON),
            "S2 decision hash map malformed")
    zero = epsilon_zero_hash()
    require(hashes["0.0"] == zero, "independent epsilon-zero hash mismatch")
    map_digest = hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    require(s2["paired_aee_present"] is False and
            s2["paper_admission"] is False and
            s2["patch"]["keep_drop_claim"] is False and
            s2["fc2"]["evaluated_again"] is False,
            "S2 top-level claim drift")
    return {
        "epsilon_rows": summaries,
        "producer_decision_hashes": hashes,
        "epsilon_zero_decision_hash_independently_recomputed": zero,
        "producer_decision_hash_map_sha256_recomputed": map_digest,
        "canonical_epsilon_row_sha256_recomputed": canonical_row_hashes,
        "nonzero_epsilon_payload_hash_replay":
            "NOT_POSSIBLE_FROM_TRANSFERRED_RESULT_WITHOUT_M1707_FC_FRAMES",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    regular(SOURCE, EXPECTED["source"])
    regular(TEST, EXPECTED["test"])
    verify_sidecar(CONTRACT, EXPECTED["contract"])
    release_triple = verify_sidecar(RELEASE, EXPECTED["release"])
    regular(DOCS359, EXPECTED["docs359"])
    m1764 = verify_sealed_dir(M1764, EXPECTED["m1764"])
    m1744 = verify_sealed_dir(M1744, EXPECTED["m1744"])
    m1763_author = verify_sealed_dir(
        M1763_AUTHOR, EXPECTED["m1763_author"], "author_receipt.json")
    result = verify_sealed_dir(RESULT, EXPECTED["result"], "decision.json",
                               exact_population=True)
    require(result["members"] == [
        "RUN_COMPLETE.txt", "decision.json", "s2_fc1_rows.csv",
        "s2_paired_aee_required_fields.json", "tsbg_b4_b8_rows.csv"],
        "result exact member population drift")

    decision = strict_json(RESULT / "decision.json")
    require(decision["schema"] ==
            "m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_v1" and
            decision["status"] ==
            "DIAGNOSTIC_SCREENING_ONLY__LAYER_PRIVATE_S2_WITNESS__TSBG_UNCHANGED__NO_PAPER_RESULT",
            "result schema/status drift")
    identity = decision["identity"]
    require(identity["analyzer_sha256"] == EXPECTED["source"] and
            identity["m1763_contract_sha256"] == EXPECTED["contract"] and
            identity["m1764_review_sha256"] == EXPECTED["m1764"][0] and
            identity["m1765_release_sha256"] == EXPECTED["release"] and
            identity["docs359_sha256"] == EXPECTED["docs359"] and
            identity["m1707_capture_inner_manifest_sha256"] ==
            "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f" and
            identity["m1707_capture_outer_seal_file_sha256"] ==
            "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85",
            "result identity drift")
    require(decision["population"] == {
        "samples": 40, "layers": 32, "fc_pairs": 960,
        "fc_frames": 11040, "fc_tokens": 44640000,
        "captured_nonzero_codes": 872855874,
        "patch_histogram_rows": 320,
    }, "result population drift")
    boundary = decision["claim_boundary"]
    require(boundary["decision_only"] is True and
            boundary["captured_codeword_and_contributor_scope_only"] is True and
            all(boundary[key] is False for key in (
                "hardware_quantization_authority", "model_bit_exact",
                "paired_aee", "rtl", "vcs", "eda", "energy",
                "system_speedup", "paper_result")),
            "result claim boundary drift")
    require(decision["s2_witness_repair"] == {
        "layer_private_identity": True, "cross_layer_padding": False,
        "tsbg_changed": False,
    }, "S2 repair identity drift")

    tsbg_fields, tsbg_csv = parse_csv(RESULT / "tsbg_b4_b8_rows.csv")
    s2_fields, s2_csv = parse_csv(RESULT / "s2_fc1_rows.csv")
    del tsbg_fields, s2_fields
    crosscheck_csv(tsbg_csv, decision["tsbg"]["rows"],
                   ("bundle", "scope_type", "scope"))
    crosscheck_csv(s2_csv, decision["s2"]["fc1_rows"],
                   ("epsilon_ratio", "scope_type", "scope"))

    paired = strict_json(RESULT / "s2_paired_aee_required_fields.json")
    require(paired["status"] == "INPUT_REQUIRED__NO_AEE_RESULT" and
            paired["paper_admission"] is False and
            paired["component_speedup_multiplication_allowed"] is False and
            paired["cohort_required"] == {
                "samples": 40,
                "exact_sample_order_sha256":
                    "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773",
            }, "paired AEE requirement drift")

    release = strict_json(RELEASE)
    require(release["authorization"]["attempts"] == 1 and
            release["authorization"]["automatic_retry"] is False and
            release["production_requirements"]["fresh_attempt_namespace"] ==
            str(ATTEMPT.relative_to(ROOT)) and
            "attempt_receipt" not in release["production_requirements"],
            "M1765 attempt contract drift")
    attempt_mode = ATTEMPT.lstat().st_mode
    require(stat.S_ISDIR(attempt_mode) and not ATTEMPT.is_symlink() and
            list(ATTEMPT.iterdir()) == [], "attempt semaphore drift")
    source_text = SOURCE.read_text(encoding="utf-8")
    require("ATTEMPT.mkdir()" in source_text and
            "require(not os.path.lexists(str(RESULT))" in source_text,
            "attempt atomic-mkdir source semantics drift")

    tsbg = verify_tsbg(decision)
    s2 = verify_s2(decision)
    output = {
        "schema": "m1775_m1763_m1707_ep34_tsbg_layer_private_s2_witness_result_hammer_mechanical_r1_v1",
        "status": "PASS_DIAGNOSTIC_RESULT_MECHANICS__NO_PAPER_ADMISSION",
        "result_triple": result,
        "authority": {
            "m1765_release_triple": {
                "payload_sha256": release_triple[0],
                "sidecar_sha256": release_triple[1],
                "outer_seal_file_sha256": release_triple[2],
            },
            "m1764": m1764, "m1744": m1744,
            "m1763_author": m1763_author,
        },
        "population": decision["population"],
        "sealed_sequence_count": 4,
        "sealed_sequences": list(SEQUENCES),
        "tsbg_recomputation": tsbg,
        "s2_recomputation": s2,
        "attempt_semantics": {
            "classification": "NOT_P0__VALID_ATOMIC_MKDIR_ONE_SHOT_SEMAPHORE",
            "empty_directory": True,
            "release_promised_sealed_attempt_receipt": False,
            "automatic_retry": False,
            "provenance_weakness": "P2_NON_SELF_DESCRIBING_CONSUMED_TOKEN",
        },
        "execution": {"analysis_runs": 0, "gpu_runs": 0, "eda_runs": 0,
                      "network_access": False, "result_tree_writes": 0,
                      "docs359_writes": 0},
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

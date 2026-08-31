#!/usr/bin/env python3
"""Independent CPU-only hammer for the sealed M463R2 result.

This auditor intentionally does not import or execute either M463 analyzer.  It
verifies the frozen identities and seals, rehashes the 80 M40 payload files,
reconstructs the cycle ledger from sealed CSVs, and independently recomputes the
beta16 static drop-set cover from the four frozen INT8 weight payloads.
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_double_seal(directory, expected_manifest=None, expected_outer=None):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(),
            "double-seal files missing: {}".format(directory))
    manifest_sha = sha256(manifest)
    outer_sha = sha256(outer)
    if expected_manifest is not None:
        require(manifest_sha == expected_manifest,
                "manifest identity drift: {}".format(directory))
    if expected_outer is not None:
        require(outer_sha == expected_outer,
                "outer-seal identity drift: {}".format(directory))
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS" and
            outer_fields[0] == manifest_sha,
            "outer seal does not bind SHA256SUMS: {}".format(directory))
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 1)
        require(len(fields) == 2, "malformed manifest row")
        expected = fields[0]
        name = fields[1].strip()
        candidate = directory / name
        require(candidate.is_file() and candidate.parent.resolve() ==
                directory.resolve(), "unsafe or missing manifest member")
        require(sha256(candidate) == expected,
                "manifest member hash mismatch: {}".format(candidate))
        entries += 1
    return {"manifest_sha256": manifest_sha, "outer_sha256": outer_sha,
            "entries_verified": entries}


def int_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {key: int(value) for key, value in row.items()}


def minimum_drop_cover(drop_sets):
    full = (1 << 96) - 1
    union = 0
    for value in drop_sets:
        union |= value
    if union != full:
        return None, False
    for size in range(1, 5):
        for indices in itertools.combinations(range(16), size):
            candidate = 0
            for index in indices:
                candidate |= drop_sets[index]
            if candidate == full:
                return size, True
    return None, True


def weight_cover_census(weight_paths):
    counts = Counter()
    maximum_retained_distribution = Counter()
    full = (1 << 96) - 1
    for weight_path in weight_paths:
        raw = np.fromfile(str(weight_path), dtype=np.int8)
        require(raw.size == 432 * 16 * 8 * 96,
                "frozen INT8 weight geometry drift")
        weight = raw.reshape(432, 16, 8, 96)
        for partition in range(432):
            for output_block in range(8):
                block = weight[partition, :, output_block, :]
                drop_sets = []
                for source in range(16):
                    mask = 0
                    for lane, value in enumerate(block[source]):
                        if abs(int(value)) <= 16:
                            mask |= 1 << lane
                    require(mask <= full, "drop-set mask overflow")
                    drop_sets.append(mask)
                minimum, reachable = minimum_drop_cover(drop_sets)
                counts["blocks"] += 1
                if not reachable:
                    counts["uncoverable"] += 1
                elif minimum is None:
                    counts["reachable_only_above_4"] += 1
                else:
                    counts["minimum_{}".format(minimum)] += 1
                    if minimum <= 4:
                        counts["cumulative_le_4"] += 1
                retained_per_lane = np.sum(
                    np.abs(block.astype(np.int16)) > 16, axis=0)
                maximum_retained_distribution[int(np.max(
                    retained_per_lane))] += 1
    return counts, maximum_retained_distribution


def write_double_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    outer = output_dir / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                     encoding="utf-8")
    return sha256(manifest), sha256(outer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing hammer overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m463r2_independent_destination_stationary_hammer_contract_v1" and
            contract.get("status") == "FROZEN_BEFORE_RAW_PAYLOAD_REHASH",
            "independent hammer contract identity drift")
    root = args.contract.resolve().parents[1]
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "frozen input identity drift: {}".format(name))
        identities[name] = {"path": spec["path"],
                            "sha256": spec["sha256"]}
    require((root / contract["inputs"]["hammer_script"]["path"]).resolve() ==
            Path(__file__).resolve(), "hammer self-path drift")

    subject_contract = strict_json(
        root / contract["inputs"]["subject_contract"]["path"])
    subject_result = strict_json(
        root / contract["inputs"]["subject_result"]["path"])
    require(subject_contract["status"] ==
            "FROZEN_PREPAYLOAD_RECOVERY_BEFORE_UNIQUE_M40_REPLAY" and
            subject_result["status"] ==
            "PASS_M463R2_CPU_ONLY_EXACT_SCHEDULE_DSE",
            "subject status drift")
    require(subject_result["identity"]["analyzer"]["sha256"] ==
            contract["inputs"]["subject_analyzer"]["sha256"],
            "subject analyzer/result binding mismatch")
    for name, spec in subject_contract["inputs"].items():
        candidate = root / spec["path"]
        require(candidate.is_file() and sha256(candidate) == spec["sha256"],
                "subject contract input drift: {}".format(name))
        require(subject_result["identity"][name]["sha256"] == spec["sha256"],
                "subject result identity mismatch: {}".format(name))
    upstream_seals = {}
    for name in subject_contract["outer_seals_to_verify"]:
        seal_path = root / subject_contract["inputs"][name]["path"]
        upstream_seals[name] = verify_double_seal(seal_path.parent)
    result_dir = (root / contract["inputs"]["subject_result"]["path"]).parent
    result_seal = verify_double_seal(
        result_dir, contract["expected"]["subject_manifest_sha256"],
        contract["inputs"]["subject_outer_seal"]["sha256"])

    trace = strict_json(root / subject_contract["inputs"]["m40_trace"]["path"])
    trace_dir = (root / subject_contract["inputs"]["m40_trace"]["path"]).parent
    payload_files = 0
    payload_bytes = 0
    payload_names = set()
    for record in trace["records"]:
        for key, hash_key in (("packed_file", "packed_file_sha256"),
                              ("value_payload_file", "value_payload_sha256")):
            name = record[key]
            require(name not in payload_names, "duplicate M40 payload path")
            payload_names.add(name)
            path = trace_dir / name
            require(path.is_file() and sha256(path) == record[hash_key],
                    "M40 payload hash mismatch: {}".format(name))
            payload_files += 1
            payload_bytes += path.stat().st_size
    require(payload_files == 80 and payload_bytes == 42346309,
            "M40 payload extent drift")

    phase_path = root / contract["inputs"]["subject_phase_csv"]["path"]
    phase_count = 0
    phase_totals = Counter()
    for row in int_rows(phase_path):
        phase_count += 1
        for key in ("active_rows", "pwp_rows", "fallback_rows",
                    "nonzero_correction_rows", "dense_keep_correction_work",
                    "beta0_correction_work", "beta16_correction_work",
                    "early_matcher"):
            phase_totals[key] += row[key]
    require(phase_count == 17280, "phase extent drift")
    expected = contract["expected"]
    require(phase_totals["dense_keep_correction_work"] == 304443912 and
            phase_totals["beta0_correction_work"] == 304443912 and
            phase_totals["beta16_correction_work"] == 304415265 and
            phase_totals["pwp_rows"] == 15909646 and
            phase_totals["nonzero_correction_rows"] == 22256814 and
            phase_totals["early_matcher"] == 67912100,
            "independent phase aggregate mismatch")
    active_phases = sum(1 for row in int_rows(phase_path)
                        if row["active_rows"] > 0)
    dense_active_compute = (8 * phase_totals["pwp_rows"] +
                            phase_totals["dense_keep_correction_work"])
    beta16_active_compute = (8 * phase_totals["pwp_rows"] +
                             phase_totals["beta16_correction_work"])
    require(dense_active_compute == 431721080 and
            beta16_active_compute == 431692433 and active_phases == 17280,
            "active-compute reconstruction mismatch")

    dense = subject_result["cycle_points"]["dense_keep_control"]
    m430 = strict_json(root / contract["inputs"]["m430_result"]["path"])
    component_keys = (
        "config_data", "config_command", "matcher", "bitmap_seal",
        "tile0_dma_data", "tile0_dma_commands", "tile1_dma_exposed",
        "replay0", "replay1", "tail", "commit")
    for key in component_keys + ("active_compute",
                                 "descriptor_sram_startup"):
        require(dense["components"][key] == m430["component_ledger"][key],
                "M430 dense component mismatch: {}".format(key))
    dense_cycles = sum(dense["components"][key] for key in component_keys)
    require(dense_cycles == dense["cycles"] ==
            m430["comparisons"]["m430_catalog_dual_cycles"] == 517041352,
            "M430 dense cycle reconstruction mismatch")

    timestamp_path = root / contract["inputs"]["subject_timestamp_csv"]["path"]
    timestamps = list(int_rows(timestamp_path))
    require(len(timestamps) == 17280, "timestamp extent drift")
    timestamp_totals = Counter()
    by_sample = defaultdict(list)
    for row in timestamps:
        by_sample[row["sample"]].append(row)
        for key in ("tile0_compute", "tile1_compute", "replay0", "replay1",
                    "tile_dma", "tile1_dma_exposed", "tail"):
            timestamp_totals[key] += row[key]
    beta16_cycles_from_time = 0
    for sample in range(10):
        rows = sorted(by_sample[sample], key=lambda item: item["phase_index"])
        require(len(rows) == 1728 and rows[0]["phase_start"] == 0,
                "per-sample timestamp extent drift")
        for previous, current in zip(rows, rows[1:]):
            require(previous["phase_end"] == current["phase_start"],
                    "timestamp continuity mismatch")
        beta16_cycles_from_time += rows[-1]["phase_end"] + 96000
    beta16_free = subject_result["cycle_points"][
        "beta16_free_selector_optimistic"]["cycles"]
    require(beta16_cycles_from_time == beta16_free == 517012705 and
            timestamp_totals["tile0_compute"] +
            timestamp_totals["tile1_compute"] == beta16_active_compute,
            "beta16 timestamp reconstruction mismatch")
    beta0_cycles = dense_cycles
    block_local_s1_f0 = beta16_free + 8 * phase_totals[
        "nonzero_correction_rows"]
    shared_s1_f0 = beta16_free + phase_totals[
        "nonzero_correction_rows"]
    hard_gate = dense_cycles * 10 // 11
    require(beta0_cycles == 517041352 and
            block_local_s1_f0 == 695067217 and
            shared_s1_f0 == 539269519 and hard_gate == 470037592,
            "named cycle-point reconstruction mismatch")

    sensitivity_path = root / contract["inputs"][
        "subject_sensitivity_csv"]["path"]
    sensitivity_rows = []
    with sensitivity_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            setup = int(row["selector_setup_cycles"])
            fill = int(row["fill_drain_cycles_per_active_tile"])
            if row["setup_scope"] == "block_local":
                setup_charge = 8 * phase_totals[
                    "nonzero_correction_rows"] * setup
            else:
                require(row["setup_scope"] == "shared_row_optimistic",
                        "unknown setup scope")
                setup_charge = phase_totals["nonzero_correction_rows"] * setup
            predicted = beta16_free + setup_charge + 2 * active_phases * fill
            require(int(row["cycles"]) == predicted,
                    "selector/fill recurrence mismatch")
            require((row["passes_1p10_m430_integer_gate"] == "True") ==
                    (predicted <= hard_gate), "gate predicate mismatch")
            sensitivity_rows.append(row)
    require(len(sensitivity_rows) == 32, "sensitivity grid extent drift")

    weight_paths = [root / subject_contract["inputs"][
        "weight_o{}".format(index)]["path"] for index in range(4)]
    cover, retained_distribution = weight_cover_census(weight_paths)
    require(cover["blocks"] == 13824 and cover["minimum_3"] == 119 and
            cover["minimum_4"] == 2694 and
            cover["cumulative_le_4"] == 2813 and
            cover["uncoverable"] == 891 and
            cover["reachable_only_above_4"] == 10120,
            "independent beta16 drop-set cover mismatch")
    require(dict(retained_distribution) ==
            {11: 15, 12: 335, 13: 2369, 14: 6119, 15: 4095, 16: 891},
            "legacy retained-source distribution drift")
    cover_csv_path = root / contract["inputs"]["subject_cover_csv"]["path"]
    cover_csv = Counter()
    cover_rows = 0
    with cover_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["beta_q"]) != 16:
                continue
            cover_rows += 1
            cover_csv["minimum_3"] += int(row["minimum_cover_eq_3"])
            cover_csv["minimum_4"] += int(row["minimum_cover_eq_4"])
            cover_csv["cumulative_le_4"] += int(row["cover_le_4"])
            cover_csv["uncoverable"] += int(
                row["uncoverable_even_with_all_16"])
    require(cover_rows == 13824 and all(cover_csv[key] == cover[key]
            for key in ("minimum_3", "minimum_4", "cumulative_le_4",
                        "uncoverable")), "cover CSV semantic mismatch")

    error_csv_path = root / contract["inputs"]["subject_error_csv"]["path"]
    error_rows = list(csv.DictReader(error_csv_path.open(
        "r", encoding="utf-8", newline="")))
    exact_signed_sum = sum(int(row[
        "exact_signed_error_sum_over_final_accumulators_q"])
                           for row in error_rows)
    final_accumulators = sum(int(row["final_accumulators"])
                             for row in error_rows)
    global_error = subject_result["numeric_error"][
        "integer_accumulator_global"]
    require(len(error_rows) == 4 and exact_signed_sum == 717519077 and
            final_accumulators == 92160000 and
            global_error["exact_signed_error_sum_over_final_accumulators_q"]
            == exact_signed_sum and
            global_error[
                "final_accumulator_abs_error_histogram_or_quantiles_computed"]
            is False,
            "raw integer error summary mismatch")

    old_anchor = 437037880
    old_anchor_would_pass = old_anchor <= hard_gate
    actual_free_passes = beta16_free <= hard_gate
    require(old_anchor_would_pass and not actual_free_passes and
            subject_result["decision"]["decision"] ==
            "NO_GO_SELECTOR_RESOURCE_GATE",
            "legacy-anchor attack or final decision mismatch")

    review = {
        "schema": "m463r2_independent_destination_stationary_hammer_review_v1",
        "status": "PASS_SEALED_RESULT_AND_UNIQUE_NO_GO",
        "score": 92,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 2},
        "scope": ("Independent CPU-only sealed-result audit of four frozen "
                  "H67 ep35 bottleneck Conv3x3 operators; no GPU, SSH, raw "
                  "payload decode, analyzer replay, RTL, VCS, Synopsys, "
                  "resource-normalized, full-network/system, accuracy, "
                  "power/energy, paper-PPA, or DATE-headline claim."),
        "identity": identities,
        "seal_audit": {
            "subject_result": result_seal,
            "upstream_outer_seals": upstream_seals,
            "subject_contract_inputs_rehashed": len(
                subject_contract["inputs"]),
            "subject_contract_input_mismatches": 0,
            "docs359_sha256": identities["docs359"]["sha256"],
        },
        "payload_audit": {
            "review_pass_payload_files_rehashed": payload_files,
            "review_pass_payload_bytes_rehashed": payload_bytes,
            "payload_sha_mismatches": 0,
            "payload_decoded": False,
            "analyzer_executed_or_imported": False,
        },
        "independent_cycles": {
            "m430_dense_components": dense["components"],
            "m430_dense_cycles": dense_cycles,
            "beta0_cycles": beta0_cycles,
            "beta16_free_cycles": beta16_free,
            "beta16_free_saved_cycles_vs_m430": dense_cycles - beta16_free,
            "beta16_free_speedup_vs_m430": dense_cycles / float(beta16_free),
            "block_local_setup1_fill0_cycles": block_local_s1_f0,
            "shared_row_setup1_fill0_cycles": shared_s1_f0,
            "hard_gate_cycles": hard_gate,
            "all_block_local_nonzero_setup_points_pass_gate": False,
            "phase_rows": phase_count,
            "active_phases": active_phases,
        },
        "drop_set_semantics": {
            "blocks": cover["blocks"],
            "minimum_cover_eq_3": cover["minimum_3"],
            "minimum_cover_eq_4": cover["minimum_4"],
            "cumulative_cover_le_4": cover["cumulative_le_4"],
            "reachable_only_above_4": cover[
                "reachable_only_above_4"],
            "uncoverable_even_with_all_16": cover["uncoverable"],
            "legacy_maximum_retained_source_distribution": {
                str(key): retained_distribution[key]
                for key in sorted(retained_distribution)},
            "interpretation": ("A union cover asks whether selected dropped-"
                               "source sets touch all 96 lanes. It is neither "
                               "maximum retained-source capacity nor a token "
                               "skip/cycle fraction."),
        },
        "legacy_anchor_attack": {
            "rejected_old_exact_anchor_cycles": old_anchor,
            "actual_exact_destination_stationary_cycles": beta16_free,
            "difference_cycles": beta16_free - old_anchor,
            "old_anchor_would_pass_1p10_gate": old_anchor_would_pass,
            "actual_exact_point_passes_1p10_gate": actual_free_passes,
            "old_anchor_implied_speedup_vs_m430":
                dense_cycles / float(old_anchor),
            "actual_free_speedup_vs_m430": dense_cycles / float(beta16_free),
            "admission": ("The old scalar is rejected as an exact anchor and "
                          "cannot authorize hardware. M463R2 correctly keeps "
                          "it out of the decision recurrence."),
        },
        "claim_boundary_review": {
            "pwp_direct_miter": ("The checked arithmetic identity is valid for "
                                 "the frozen trace: x*Wkeep = c*Wkeep + "
                                 "(x-c)*Wkeep. It does not prove selector RTL, "
                                 "cycle delivery, post-BN accuracy, or AEE."),
            "sumabs_bound": ("For binary integer support, triangle inequality "
                             "gives |sum x_i*qdrop_i| <= sum |qdrop_i|. The "
                             "sealed run did not compute final-accumulator "
                             "absolute-error samples or quantiles, so zero "
                             "'violations' is a theorem-side statement, not "
                             "an empirical distribution check."),
            "token_cycle_overclaim_attack": {
                "cover_le_4_fraction": cover["cumulative_le_4"] /
                float(cover["blocks"]),
                "beta16_correction_work_reduction":
                    phase_totals["dense_keep_correction_work"] -
                    phase_totals["beta16_correction_work"],
                "beta16_free_cycle_reduction": dense_cycles - beta16_free,
                "interpretation": ("The static 20.35% cover statistic yields "
                                   "neither 20.35% token skipping nor cycle "
                                   "reduction. Exact free-selector replay saves "
                                   "only 28,647 cycles before any resource "
                                   "charge."),
            },
        },
        "verified": [
            "All frozen contract inputs, six inherited double seals, and the M463R2 result double seal verify with zero mismatch.",
            "The review pass rehashed 80 distinct M40 payload files totaling 42,346,309 bytes with zero mismatch and did not decode them or execute/import an M463 analyzer.",
            "The independent CSV recurrence exactly reconstructs M430 dense 517,041,352, beta0 517,041,352, beta16 free 517,012,705, block-local s1/f0 695,067,217, shared-row s1/f0 539,269,519, and hard gate 470,037,592 cycles.",
            "The four frozen INT8 weights independently reproduce exact3=119, exact4=2,694, cumulative<=4=2,813, reachable-only-above4=10,120, and uncoverable=891 over 13,824 blocks.",
            "The final decision is robust: even the physically free selector point misses the gate; every block-local point with nonzero setup is slower still.",
        ],
        "findings": [
            {
                "severity": "P1",
                "title": "The legacy 437,037,880 scalar is not a valid exact anchor",
                "detail": ("It is 79,974,825 cycles below the exact M463R2 "
                           "destination-stationary replay. Treating it as exact "
                           "would create 1.183x and reverse the hard gate. "
                           "M463R2 correctly uses 517,012,705 for all decisions; "
                           "the legacy scalar must remain non-admitting and "
                           "uncitable as measured performance."),
            },
            {
                "severity": "P2",
                "title": "Drop-set cover is a static weight property, not token or cycle sparsity",
                "detail": ("2,813/13,824 blocks have a <=4 dropped-source union "
                           "cover, but exact correction work falls by only "
                           "28,647 and exact free-selector cycles by only 28,647. "
                           "The cover percentage must not be presented as a "
                           "skip rate or speedup."),
            },
            {
                "severity": "P2",
                "title": "The sumabs bound is analytic rather than an observed absolute-error audit",
                "detail": ("The triangle-inequality bound is sound for the raw "
                           "binary integer support, but final-accumulator "
                           "absolute errors and quantiles were explicitly not "
                           "computed. Keep the claim to a worst-case raw-Conv "
                           "bound and aggregate signed sum; do not infer BN/AEE "
                           "accuracy or an empirical zero-violation result."),
            },
        ],
        "decision": {
            "unique_decision": "NO_GO_SELECTOR_RESOURCE_GATE",
            "beta16_cycle_line": "NO_GO",
            "rtl_vcs_synopsys": "NO_GO",
            "quote_437037880_as_exact_or_measured": "NO_GO",
            "quote_cover_fraction_as_token_skip_or_speedup": "NO_GO",
            "paper_system_or_date_headline": "NO_GO",
            "retained_evidence": ("Static lossy-weight census and raw-integer "
                                  "error-bound appendix evidence only."),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    review_name = "m463r2_independent_destination_stationary_hammer_review.json"
    markdown_name = "m463r2_independent_destination_stationary_hammer_review.md"
    (args.output_dir / review_name).write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (args.output_dir / markdown_name).write_text(
        "# M463R2 independent hammer\n\n"
        "Score: **92/100**; P0/P1/P2 = **0/1/2**.\n\n"
        "Unique decision: **NO_GO_SELECTOR_RESOURCE_GATE**.  The exact beta16 "
        "free-selector replay is 517,012,705 cycles, not the rejected legacy "
        "437,037,880 anchor.  The first block-local nonzero-setup point is "
        "695,067,217 cycles against the frozen 470,037,592 gate.\n\n"
        "The independently recomputed cover counts are exact3=119, exact4=2,694, "
        "cumulative<=4=2,813, and uncoverable=891.  They are static weight-cover "
        "statistics, not token skip or cycle speedup.  PWP-direct equality and "
        "the sumabs bound remain raw integer arithmetic claims only; they do not "
        "establish RTL delivery, post-BN/valid825 accuracy, AEE, PPA, energy, or "
        "system performance.\n",
        encoding="utf-8")
    manifest_sha, outer_sha = write_double_seal(
        args.output_dir, [review_name, markdown_name])
    print("PASS_M463R2_INDEPENDENT_HAMMER score=92 P0=0 P1=1 P2=2 "
          "decision=NO_GO_SELECTOR_RESOURCE_GATE manifest={} outer={}".format(
              manifest_sha, outer_sha), flush=True)


if __name__ == "__main__":
    main()

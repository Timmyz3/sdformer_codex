#!/opt/anaconda3/envs/pytorch310/bin/python
"""Independent, fail-closed audit for the sealed M470 producer DSE.

This checker deliberately consumes only the sealed producer JSON/CSV artifacts
and the corrected r2 admission receipt.  It does not import the producer model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_RESULT_SHA = "7817460e7c13e73c20b80de4224ffac285d3a604edf258c182e3c8c78a9ad165"
EXPECTED_PRODUCER_SEAL_SHA = "e4697fb47c0c4ab311ad9e7a272d0a97918bd834f85e60a446b1b9dab7b8c8a8"
EXPECTED_DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SPILL_BYTES_PER_BOUNDARY = 5_472_000
SRAM_LIMIT_BYTES = 245_760
STRONGEST_ZERO_EXTERNAL = 742_148_386
M468R3_STORED_EXTERNAL = 872_452_768


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def require(condition: bool, label: str, failures: list[str], checks: Counter[str]) -> None:
    checks["performed"] += 1
    if not condition:
        failures.append(label)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    hw = repo / "hw_autoresearch_nts07"
    producer = hw / "results/m470_h67_partition_window_payload_stationary_r1b_20260826"
    admission_path = hw / "results/m470_h67_partition_window_payload_stationary_admission_r2_20260826/m470_producer_admission_receipt_r2.json"
    result_path = producer / "m470_h67_partition_window_payload_stationary_result_r1.json"
    points_path = producer / "m470_cycle_traffic_capacity_points.csv"
    comparisons_path = producer / "m470_materiality_comparisons.csv"
    infeasible_path = producer / "m470_infeasible_capacity_points.csv"
    seal_path = producer / "SHA256SUMS.seal.sha256"
    sums_path = producer / "SHA256SUMS"
    docs359 = hw / "docs/359_DATE终局冻结_20260813.md"

    failures: list[str] = []
    checks: Counter[str] = Counter()

    result_sha = sha256(result_path)
    producer_seal_sha = sha256(seal_path)
    docs359_sha = sha256(docs359)
    require(result_sha == EXPECTED_RESULT_SHA, "producer result SHA mismatch", failures, checks)
    require(producer_seal_sha == EXPECTED_PRODUCER_SEAL_SHA, "producer seal SHA mismatch", failures, checks)
    require(docs359_sha == EXPECTED_DOCS359_SHA, "docs359 SHA mismatch", failures, checks)

    # Verify both levels of the producer's sha256sum seal without invoking it.
    for line in sums_path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        require(sha256(producer / relative) == digest, f"producer SHA256SUMS mismatch: {relative}", failures, checks)
    seal_digest, seal_relative = seal_path.read_text().strip().split(maxsplit=1)
    require(seal_relative.lstrip("* ") == "SHA256SUMS", "unexpected producer seal target", failures, checks)
    require(sha256(sums_path) == seal_digest, "producer outer seal mismatch", failures, checks)

    result = json.loads(result_path.read_text())
    admission = json.loads(admission_path.read_text())
    require(admission["identity"]["producer_result"]["sha256"] == result_sha, "r2 result SHA does not bind producer", failures, checks)
    require(admission["identity"]["producer_seal"]["sha256"] == producer_seal_sha, "r2 seal SHA does not bind producer", failures, checks)
    require(admission["r1_fail_closed"]["correct_sha256"] == result_sha, "r2 corrected SHA mismatch", failures, checks)
    require(admission["r1_fail_closed"]["wrong_sha256_recorded_by_r1"] != result_sha, "r1 wrong SHA unexpectedly equals producer", failures, checks)
    require(admission["r1_fail_closed"]["r1_admission_must_not_be_cited"] is True, "r1 was not explicitly revoked", failures, checks)

    with points_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with comparisons_path.open(newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    with infeasible_path.open(newline="") as stream:
        infeasible = list(csv.DictReader(stream))

    require(len(rows) == 147, f"point count {len(rows)} != 147", failures, checks)
    require(len(result["points"]) == 147, f"JSON point count {len(result['points'])} != 147", failures, checks)
    csv_keys = [
        (r["mode"], int(r["partition_window_p"]), int(r["resident_block_banks"]), r["bandwidth_bytes_per_cycle"], r["generator_source_lanes_k"])
        for r in rows
    ]
    require(len(set(csv_keys)) == 147, "CSV DSE point key is not unique", failures, checks)

    # Independently reconcile JSON points to CSV at their actual DSE grain.
    json_by_key = {}
    for point in result["points"]:
        bw = str(point["bandwidth_bytes_per_cycle"])
        lane = "" if point["generator_source_lanes_k"] is None else str(point["generator_source_lanes_k"])
        key = (point["mode"], point["partition_window_p"], point["resident_block_banks"], bw, lane)
        json_by_key[key] = point
    require(set(csv_keys) == set(json_by_key), "JSON/CSV DSE grains differ", failures, checks)

    for index, row in enumerate(rows):
        prefix = f"point[{index}] {row['mode']}/P{row['partition_window_p']}/B{row['resident_block_banks']}/BW{row['bandwidth_bytes_per_cycle']}/K{row['generator_source_lanes_k'] or '-'}"
        boundaries = as_int(row, "operator_window_boundary_count")
        spill = boundaries * SPILL_BYTES_PER_BOUNDARY
        require(as_int(row, "psum_spill_write_bytes") == spill, prefix + " spill-write invariant", failures, checks)
        require(as_int(row, "psum_reload_read_bytes") == spill, prefix + " reload-read invariant", failures, checks)
        require(as_int(row, "spill_reload_dram_bytes") == 2 * spill, prefix + " spill DRAM sum", failures, checks)
        require(as_int(row, "spill_reload_sram_bytes") == 2 * spill, prefix + " spill SRAM sum", failures, checks)

        execution = (
            as_int(row, "matcher_or_popcount_cycles")
            + as_int(row, "replay_or_source_issue_cycles")
            + as_int(row, "task_fill_cycles")
            + as_int(row, "task_drain_cycles")
            + as_int(row, "descriptor_latency_cycles")
        )
        total = (
            execution
            + as_int(row, "payload_fill_cycles")
            + as_int(row, "spill_reload_cycles")
            + as_int(row, "final_commit_cycles")
            + as_int(row, "generator_cycles")
        )
        require(as_int(row, "execution_cycles") == execution, prefix + " execution component sum", failures, checks)
        require(as_int(row, "total_cycles") == total, prefix + " total cycle component sum", failures, checks)
        logical = as_int(row, "logical_sram_bytes")
        macro = as_int(row, "macro_rounded_sram_bytes")
        require(logical <= SRAM_LIMIT_BYTES, prefix + " logical SRAM exceeds 240 KiB", failures, checks)
        require(macro <= SRAM_LIMIT_BYTES, prefix + " macro-rounded SRAM exceeds 240 KiB", failures, checks)
        require(row["fits_240k_logical"] == "True", prefix + " logical-fit flag false", failures, checks)
        require(row["fits_240k_macro_rounded"] == "True", prefix + " macro-fit flag false", failures, checks)
        require(row["fits_both_240k_gates"] == "True", prefix + " dual-fit flag false", failures, checks)
        require(as_int(row, "m40_payload_reads") == 0, prefix + " M40 payload reread", failures, checks)
        require(row["performance_admitted"] == "False", prefix + " performance admitted", failures, checks)
        require(row["system_speedup"] == "False", prefix + " system speedup asserted", failures, checks)

        key = csv_keys[index]
        point = json_by_key[key]
        require(point["total_cycles"] == total, prefix + " JSON/CSV total cycle mismatch", failures, checks)
        require(point["capacity"]["logical_total_bytes"] == logical, prefix + " JSON/CSV logical capacity mismatch", failures, checks)
        require(point["capacity"]["macro_rounded_total_bytes"] == macro, prefix + " JSON/CSV macro capacity mismatch", failures, checks)
        require(point["capacity"]["every_macro_item_ge_logical"] is True, prefix + " macro item under logical item", failures, checks)

    # Every published comparison ratio is recomputed from its two cycle counts.
    for index, row in enumerate(comparisons):
        candidate = int(row["candidate_cycles"])
        same_tile = int(row["same_tile_strong_zero_cycles"])
        same_resource = int(row["same_resource_strong_zero_cycles"])
        require(close(float(row["same_tile_speedup"]), same_tile / candidate), f"comparison[{index}] same-tile ratio", failures, checks)
        require(close(float(row["same_resource_speedup"]), same_resource / candidate), f"comparison[{index}] same-resource ratio", failures, checks)
        require(row["performance_admitted"] == "False", f"comparison[{index}] admitted", failures, checks)

    stored_128 = [
        r for r in comparisons
        if r["candidate"] == "stored_pwp" and r["bandwidth_bytes_per_cycle"] == "128"
    ]
    best_all = max(stored_128, key=lambda r: float(r["same_resource_speedup"]))
    requested = [r for r in stored_128 if int(r["partition_window_p"]) in {1, 2, 4, 8}]
    best_requested = max(requested, key=lambda r: float(r["same_resource_speedup"]))
    require(int(best_all["partition_window_p"]) == 5, "best all P is not 5", failures, checks)
    require(int(best_all["candidate_cycles"]) == 892_869_158, "best all candidate cycles", failures, checks)
    require(int(best_all["same_resource_strong_zero_cycles"]) == 1_148_674_816, "best all zero cycles", failures, checks)
    require(close(float(best_all["same_resource_speedup"]), 1_148_674_816 / 892_869_158), "best all ratio", failures, checks)
    require(int(best_requested["partition_window_p"]) == 4, "best requested P is not 4", failures, checks)
    require(close(float(best_requested["same_resource_speedup"]), 1_218_613_216 / 964_742_918), "best requested ratio", failures, checks)

    p8_stored_infeasible = [r for r in infeasible if r["mode"] == "stored_pwp" and r["partition_window_p"] == "8"]
    p8_stored_feasible = [r for r in rows if r["mode"] == "stored_pwp" and r["partition_window_p"] == "8"]
    require(len(p8_stored_infeasible) == 2, "stored P8 missing infeasible records for both bank counts", failures, checks)
    require(not p8_stored_feasible, "stored P8 unexpectedly feasible", failures, checks)

    candidate_cycles = int(best_all["candidate_cycles"])
    audit = {
        "schema": "m470_independent_hammer_audit_v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "checks_performed": checks["performed"],
        "failure_count": len(failures),
        "failures": failures,
        "identity": {
            "producer_result_sha256": result_sha,
            "producer_seal_file_sha256": producer_seal_sha,
            "docs359_sha256": docs359_sha,
            "r1_admission_revoked": admission["r1_fail_closed"]["r1_admission_must_not_be_cited"],
            "r2_corrected_identity_pass": admission["identity"]["producer_result"]["sha256"] == result_sha,
        },
        "recomputed": {
            "points": len(rows),
            "spill_bytes_per_boundary": SPILL_BYTES_PER_BOUNDARY,
            "all_cycle_component_sums_pass": not any("component sum" in f for f in failures),
            "all_capacity_dual_gates_pass": not any("SRAM exceeds" in f or "fit flag" in f for f in failures),
            "best_stored_128Bpc": {
                "partition_window_p": int(best_all["partition_window_p"]),
                "resident_block_banks": int(best_all["resident_block_banks"]),
                "candidate_cycles": candidate_cycles,
                "same_resource_zero_cycles": int(best_all["same_resource_strong_zero_cycles"]),
                "same_resource_speedup": int(best_all["same_resource_strong_zero_cycles"]) / candidate_cycles,
            },
            "best_requested_p_1_2_4_8": {
                "partition_window_p": int(best_requested["partition_window_p"]),
                "candidate_cycles": int(best_requested["candidate_cycles"]),
                "same_resource_zero_cycles": int(best_requested["same_resource_strong_zero_cycles"]),
                "same_resource_speedup": int(best_requested["same_resource_strong_zero_cycles"]) / int(best_requested["candidate_cycles"]),
            },
            "stored_pwp_p8_feasible": False,
            "diagnostic_speedup_vs_strongest_zero_742148386": STRONGEST_ZERO_EXTERNAL / candidate_cycles,
            "diagnostic_speedup_vs_m468r3_stored_872452768": M468R3_STORED_EXTERNAL / candidate_cycles,
        },
        "verdict": {
            "score_out_of_100": 78 if not failures else 0,
            "rtl_nominated": False,
            "performance_admitted": False,
            "decision": "KILL_M470_RTL_AXIS_KEEP_CPU_DSE_AS_NEGATIVE_EVIDENCE" if not failures else "FAIL_CLOSED_NO_CLAIM",
            "reason": "The internal same-schedule 1.286498x point is exact within the CPU model, but is 0.831195x versus the stronger frozen zero baseline and 0.977134x versus M468R3 stored; it does not justify RTL.",
        },
        "claim_boundary": {
            "four_frozen_h67_bottleneck_conv3x3_only": True,
            "cpu_cycle_simulator": True,
            "rtl_measured_speedup": False,
            "synopsys": False,
            "energy": False,
            "full_network_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

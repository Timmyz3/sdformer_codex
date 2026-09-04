#!/usr/bin/env python3
"""RTL-edge-aware W64 PWP/correction/accumulator schedule replay."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M108_R1_SCRIPT = HW / (
    "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py")
M108_R1_RESULT = HW / (
    "results/m108_w64_fused_pwp_accumulator_schedule_r1_20260824/"
    "m108_w64_fused_pwp_accumulator_schedule.json")
M107_REVIEW = HW / (
    "reviews/m107_w64_pingpong_schedule_replay_independent_hammer_r1_20260824/"
    "m107_w64_pingpong_schedule_replay_independent_hammer_review.json")
M106_R2_CONTRACT = HW / (
    "contracts/m106_r2_standard_streaming_grace_vcs_contract_r1_20260824.json")
M106_R2_RUN = HW / (
    "dc_handoff/runs/m106_r2_standard_streaming_grace_vcs_r1_sealed_20260824/"
    "RUN_COMPLETE.txt")

EXPECTED_SHA256 = {
    "m108_r1_script": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_r1_result": "358640e62c2e52f859b7143f0bac957d6988ed1bd7c56e5dd54d21bc01344318",
    "m107_review": "256f2c049ac09c57428f7dcfe93343efab41c09f4e1b128023b6479cf082d873",
    "m106_r2_contract": "984ca6558ebbf3a58135e60b4aa889b7726532b8a4fc872acf7156f50d7d8196",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def load_r1():
    spec = importlib.util.spec_from_file_location("m108_frozen_r1", M108_R1_SCRIPT)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen M108 r1 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate_rtl_edge_window_major(m108, sequence):
    bank_free = [0, 0]
    producer_end = 0
    service_end = 0
    service_idle = 0
    producer_stall = 0
    descriptor_fill_cycles = 0
    correction_tokens = 0
    pwp_tokens = 0
    commit_cycles = 0
    pipeline_flush_cycles = 0
    dispatch_edges = 0
    bank_reacquire_boundaries = 0
    exposed_post_pwp_wait = 0
    max_fill = 0
    max_service = 0

    for index, row in enumerate(sequence):
        bank = index & 1
        if index == 0 or producer_end > bank_free[bank]:
            fill_start = producer_end
        else:
            fill_start = bank_free[bank] + 1
            bank_reacquire_boundaries += 1
        producer_stall += fill_start - producer_end
        fill_cycles = row["events"] + 1
        fill_end = fill_start + fill_cycles
        producer_end = fill_end
        descriptor_fill_cycles += fill_cycles
        dispatch_edges += 1
        dispatch_ready = fill_end + 1

        # PWP uses the shared 256-bit service lane while the independent M106
        # descriptor bank fills and dispatches.  Thus the controller edge costs
        # a cycle only when it is not hidden under PWP service.
        pwp_start = max(service_end, fill_start)
        service_idle += pwp_start - service_end
        pwp_end = pwp_start + row["pwp_tokens"]
        pwp_tokens += row["pwp_tokens"]

        correction = row["events"] + 3 * row["groups"]
        correction_tokens += correction
        if correction:
            correction_start = max(pwp_end, dispatch_ready)
            exposed = correction_start - pwp_end
            exposed_post_pwp_wait += exposed
            service_idle += exposed
            correction_end = correction_start + correction
            bank_free[bank] = correction_end
            service_end = correction_end
        else:
            # An empty bank is released on the dispatch edge without consuming
            # the shared service lane.  Its control latency can overlap PWP.
            bank_free[bank] = dispatch_ready
            service_end = pwp_end

        max_fill = max(max_fill, fill_cycles)
        max_service = max(max_service, row["pwp_tokens"] + correction)

        if row["partition"] == m108.PARTITIONS - 1:
            # The final descriptor must be dispatched/released before commit,
            # even when it is empty and has no correction service token.
            window_ready = max(service_end, bank_free[bank])
            service_idle += window_ready - service_end
            service_end = window_ready + 1
            pipeline_flush_cycles += 1
            rows_here = min(m108.WIN_ROWS,
                            m108.ROWS - row["window"] * m108.WIN_ROWS)
            commit = rows_here * m108.OUTPUT_BLOCKS
            service_end += commit
            commit_cycles += commit

    require(correction_tokens
            == m108.EXPECTED_EVENTS + 3 * m108.EXPECTED_GROUPS,
            "correction token conservation failed")
    require(pwp_tokens == m108.EXPECTED_PWP_TOKENS,
            "PWP token conservation failed")
    common_tail = commit_cycles + pipeline_flush_cycles
    baseline_cycles = m108.BASELINE_SERVICE_TOKENS + common_tail
    require(service_end == pwp_tokens + correction_tokens + service_idle
            + common_tail, "M108 r2 service cycle conservation failed")
    return {
        "descriptors": len(sequence),
        "descriptor_fill_cycles": descriptor_fill_cycles,
        "producer_bank_stall_cycles": producer_stall,
        "controller_dispatch_edges": dispatch_edges,
        "bank_reacquire_boundaries": bank_reacquire_boundaries,
        "exposed_post_pwp_fill_or_dispatch_wait_cycles": exposed_post_pwp_wait,
        "service_idle_cycles": service_idle,
        "pwp_service_tokens": pwp_tokens,
        "correction_service_tokens": correction_tokens,
        "accumulator_pipeline_flush_cycles": pipeline_flush_cycles,
        "accumulator_commit_cycles": commit_cycles,
        "candidate_cycles": service_end,
        "fair_fixed8_baseline_cycles": baseline_cycles,
        "same_clock_service_island_ratio": baseline_cycles / float(service_end),
        "headroom_to_two_x_cycles": baseline_cycles // 2 - service_end,
        "maximum_descriptor_fill_cycles": max_fill,
        "maximum_descriptor_service_tokens": max_service,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M108 r2 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in {
        "m108_r1_script": M108_R1_SCRIPT,
        "m108_r1_result": M108_R1_RESULT,
        "m107_review": M107_REVIEW,
        "m106_r2_contract": M106_R2_CONTRACT,
    }.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)
    require(M106_R2_RUN.read_text(encoding="utf-8").splitlines()[0]
            == "status=PASS_M106_R2_STANDARD_STREAMING_GRACE_DIRECTED_VCS_SVA",
            "M106 r2 sealed VCS admission missing")

    m108 = load_r1()
    m105 = m108.load_m105_module()
    manifest = m108.strict_json(m105.M40_MANIFEST)
    m72 = m108.strict_json(m105.M72_RESULT)
    m41 = m108.strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"] if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]),
    )
    require(len(heldout) == 20, "heldout record extent drift")
    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, weight_shas, _ = m105.build_width_catalog(m72, m41)

    record_counts = {}
    total_width_uses = Counter()
    totals = Counter()
    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        counts = m108.descriptor_counts(
            m105, masks, record["operator_index"], centers, widths, popcount)
        key = (record["sample_id"], record["operator_index"])
        record_counts[key] = counts
        total_width_uses.update(counts["width_uses"])
        for name in ("events", "groups", "pwp_tokens", "pwp_updates",
                     "positive_events", "negative_events"):
            totals[name] += int(counts[name].sum())
        totals["coefficient_checks"] += counts["coefficient_checks"]
        print("[M108R2 RECORD] {}/20 sample={} op={}".format(
            record_index + 1, key[0], key[1]), flush=True)
    require(totals["events"] == m108.EXPECTED_EVENTS,
            "event conservation drift")
    require(totals["groups"] == m108.EXPECTED_GROUPS,
            "group conservation drift")
    require(totals["pwp_tokens"] == m108.EXPECTED_PWP_TOKENS,
            "PWP conservation drift")
    require(dict(sorted(total_width_uses.items())) == m108.EXPECTED_PWP_USES,
            "PWP width-use drift")

    sequence = []
    for sample in range(5, 10):
        for op in range(4):
            counts = record_counts[(sample, op)]
            for window in range(m108.WINDOWS_PER_PHASE):
                for partition in range(m108.PARTITIONS):
                    sequence.append({
                        "sample": sample,
                        "operator": op,
                        "window": window,
                        "partition": partition,
                        "events": int(counts["events"][partition, window]),
                        "groups": int(counts["groups"][partition, window]),
                        "pwp_tokens": int(counts["pwp_tokens"][partition, window]),
                    })
    require(len(sequence) == 406080, "descriptor extent drift")
    schedule = simulate_rtl_edge_window_major(m108, sequence)
    r1 = strict_json(M108_R1_RESULT)
    require(schedule["candidate_cycles"] >= r1["schedule"]["candidate_cycles"],
            "RTL-edge r2 unexpectedly beats fluid r1")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M108 r2 analyzer changed during execution")

    payload = {
        "schema": "m108_r2_rtl_edge_fused_schedule_result_v1",
        "status": "PASS_RTL_EDGE_AWARE_PRECOMPACTED_MODULE_SCHEDULE_PORT_CUTS_REMAIN",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "m108_r1_script_sha256": EXPECTED_SHA256["m108_r1_script"],
            "m108_r1_result_sha256": EXPECTED_SHA256["m108_r1_result"],
            "m107_review_sha256": EXPECTED_SHA256["m107_review"],
            "m106_r2_contract_sha256": EXPECTED_SHA256["m106_r2_contract"],
            "m106_r2_run_complete_sha256": sha256(M106_R2_RUN),
            "weight_payload_sha256": weight_shas,
        },
        "work_conservation": {
            "events": totals["events"],
            "positive_events": totals["positive_events"],
            "negative_events": totals["negative_events"],
            "active_groups": totals["groups"],
            "pwp_updates": totals["pwp_updates"],
            "pwp_service_tokens": totals["pwp_tokens"],
            "pwp_uses_by_width": dict(sorted(total_width_uses.items())),
            "source_coefficient_checks": totals["coefficient_checks"],
        },
        "fluid_r1_reference": r1["schedule"],
        "rtl_edge_schedule": schedule,
        "model_boundary": {
            "standard_streaming_m106_r2": True,
            "ready_dispatch_and_blocked_reacquire_edges": True,
            "pwp_seed_before_correction": True,
            "accumulator_flush_and_commit": True,
            "commercial_small_stream_cycle_miter": False,
            "precompaction_schedule": False,
            "shared_weight_sram_schedule": False,
            "full_lane_numeric_miter": False,
            "macro_inclusive_ppa": False,
        },
        "admission": {
            "rtl_edge_aware_software_cycle_schedule": True,
            "scheduled_precompacted_module_cycle_ratio": True,
            "actual_m106_controller_cycle_miter": False,
            "physical_speedup": False,
            "equal_area": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M108R2 ratio={:.12f} candidate={} headroom={} output={}".format(
        schedule["same_clock_service_island_ratio"],
        schedule["candidate_cycles"], schedule["headroom_to_two_x_cycles"],
        args.output), flush=True)


if __name__ == "__main__":
    main()

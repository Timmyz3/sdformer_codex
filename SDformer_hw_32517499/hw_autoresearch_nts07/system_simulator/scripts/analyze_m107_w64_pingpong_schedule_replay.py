#!/usr/bin/env python3
"""Replay M105 heldout events through a two-bank W64 fill/drain schedule."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M105_DIR = HW / "reviews/m105_bounded_row_transpose_preflight_independent_hammer_r1_20260824"
M105_SCRIPT = M105_DIR / "audit_m105_bounded_row_transpose.py"
M105_RESULT = M105_DIR / "m105_bounded_row_transpose_preflight.json"
M106_CONTRACT = HW / "contracts/m106_w64_bounded_bitmap_transpose_vcs_contract_r1_20260824.json"
M106_RUN = HW / "dc_handoff/runs/m106_w64_bounded_bitmap_transpose_vcs_r1_sealed_20260824"

EXPECTED_SHA256 = {
    "m105_script": "5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5",
    "m105_result": "3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b",
    "m106_contract": "881491f58543f2c6b0b5b3c1d07d7b170cdbfb4190153a18929bdddd83a39999",
}
WIN_ROWS = 64
WINDOWS_PER_PHASE = 47
EXPECTED_EVENTS = 188148490
EXPECTED_GROUPS = 35140002
PWP_TOKENS = 226222255
BASELINE_TOKENS = 1114383288


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

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )


def load_m105_module():
    spec = importlib.util.spec_from_file_location("m105_frozen_auditor", M105_SCRIPT)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen M105 auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_counts(event_masks, popcount):
    starts = np.arange(0, event_masks.shape[1], WIN_ROWS, dtype=np.intp)
    row_events = popcount[event_masks].sum(axis=2, dtype=np.uint16)
    prefix = np.concatenate((
        np.zeros((event_masks.shape[0], 1), dtype=np.uint32),
        np.cumsum(row_events, axis=1, dtype=np.uint32),
    ), axis=1)
    ends = np.minimum(starts + WIN_ROWS, event_masks.shape[1])
    events = prefix[:, ends] - prefix[:, starts]
    unions = np.bitwise_or.reduceat(event_masks, starts, axis=1)
    groups = popcount[unions].sum(axis=2, dtype=np.uint16)
    require(events.shape == groups.shape == (432, WINDOWS_PER_PHASE),
            "W64 phase schedule shape drift")
    return events.astype(np.int64), groups.astype(np.int64)


def simulate_pingpong(sequence):
    bank_free = [0, 0]
    producer_end = 0
    drain_end = 0
    correction_tokens = 0
    descriptor_fill_cycles = 0
    service_idle_cycles = 0
    producer_bank_stall_cycles = 0
    empty_windows = 0
    nonempty_windows = 0
    maximum_fill_cycles = 0
    maximum_drain_tokens = 0

    for index, (events, groups) in enumerate(sequence):
        require(events >= groups >= 0, "event/group relation violated")
        if groups == 0:
            require(events == 0, "events without an active key")
            empty_windows += 1
        else:
            nonempty_windows += 1
        bank = index & 1
        fill_start = max(producer_end, bank_free[bank])
        producer_bank_stall_cycles += fill_start - producer_end
        fill_cycles = events + 1  # one accepted event/cycle plus close
        fill_end = fill_start + fill_cycles
        producer_end = fill_end
        drain_start = max(drain_end, fill_end)
        service_idle_cycles += drain_start - drain_end
        drain_tokens = events + 3 * groups
        drain_end = drain_start + drain_tokens
        bank_free[bank] = drain_end
        correction_tokens += drain_tokens
        descriptor_fill_cycles += fill_cycles
        maximum_fill_cycles = max(maximum_fill_cycles, fill_cycles)
        maximum_drain_tokens = max(maximum_drain_tokens, drain_tokens)

    require(drain_end == correction_tokens + service_idle_cycles,
            "ping-pong cycle conservation failed")
    return {
        "windows": len(sequence),
        "empty_windows": empty_windows,
        "nonempty_windows": nonempty_windows,
        "correction_service_tokens": correction_tokens,
        "descriptor_fill_cycles": descriptor_fill_cycles,
        "producer_bank_stall_cycles": producer_bank_stall_cycles,
        "service_idle_cycles_from_initial_fill_or_bank_not_ready": service_idle_cycles,
        "correction_schedule_cycles": drain_end,
        "maximum_window_fill_cycles": maximum_fill_cycles,
        "maximum_window_drain_tokens": maximum_drain_tokens,
        "pwp_tokens_serially_charged": PWP_TOKENS,
        "combined_service_island_cycles": drain_end + PWP_TOKENS,
        "fixed8_baseline_tokens": BASELINE_TOKENS,
        "same_clock_service_island_ratio":
            BASELINE_TOKENS / float(drain_end + PWP_TOKENS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M107 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    require(sha256(M105_SCRIPT) == EXPECTED_SHA256["m105_script"],
            "M105 auditor identity drift")
    require(sha256(M105_RESULT) == EXPECTED_SHA256["m105_result"],
            "M105 result identity drift")
    require(sha256(M106_CONTRACT) == EXPECTED_SHA256["m106_contract"],
            "M106 contract identity drift")
    require((M106_RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
            .splitlines()[0]
            == "status=PASS_M106_W64_BOUNDED_BITMAP_TRANSPOSE_DIRECTED_VCS_SVA",
            "M106 sealed VCS admission missing")

    m105 = load_m105_module()
    for label, path in {
        "m40_manifest": m105.M40_MANIFEST,
        "m72_result": m105.M72_RESULT,
        "m78_result": m105.M78_RESULT,
        "m41_result": m105.M41_RESULT,
    }.items():
        require(sha256(path) == m105.EXPECTED_SHA256[label],
                "frozen source identity drift: " + label)
    manifest = strict_json(m105.M40_MANIFEST)
    m72 = strict_json(m105.M72_RESULT)
    m41 = strict_json(m105.M41_RESULT)
    heldout = sorted(
        (row for row in manifest["records"] if row["sample_id"] in range(5, 10)),
        key=lambda row: (row["sample_id"], row["operator_index"]),
    )
    require(len(heldout) == 20, "heldout record extent drift")

    popcount = np.fromiter(
        (int(value).bit_count() for value in range(1 << 16)),
        dtype=np.uint8,
        count=1 << 16,
    )
    centers = m105.centers_array(m72)
    widths, weight_shas, _ = m105.build_width_catalog(m72, m41)
    record_counts = {}
    total_events = 0
    total_groups = 0
    for record_index, record in enumerate(heldout):
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, _, _ = m105.build_record_events(
            masks, record["operator_index"], centers, widths, popcount)
        events, groups = window_counts(event_masks, popcount)
        record_counts[(record["sample_id"], record["operator_index"])] = (
            events, groups)
        total_events += int(events.sum())
        total_groups += int(groups.sum())
        print("[M107 RECORD] {}/20 sample={} op={} events={} groups={}".format(
            record_index + 1, record["sample_id"],
            record["operator_index"], int(events.sum()), int(groups.sum())),
            flush=True)
    require(total_events == EXPECTED_EVENTS, "M107 event conservation drift")
    require(total_groups == EXPECTED_GROUPS, "M107 group conservation drift")

    partition_major = []
    window_major = []
    for sample in range(5, 10):
        for operator in range(4):
            events, groups = record_counts[(sample, operator)]
            for partition in range(432):
                for window in range(WINDOWS_PER_PHASE):
                    partition_major.append((int(events[partition, window]),
                                            int(groups[partition, window])))
            for window in range(WINDOWS_PER_PHASE):
                for partition in range(432):
                    window_major.append((int(events[partition, window]),
                                         int(groups[partition, window])))
    require(len(partition_major) == len(window_major) == 406080,
            "M107 window sequence extent drift")
    require(sorted(partition_major) == sorted(window_major),
            "loop reorder changed window multiset")
    partition_result = simulate_pingpong(partition_major)
    window_result = simulate_pingpong(window_major)

    frozen = strict_json(M105_RESULT)
    w64 = next(row for row in frozen["window_results"]
               if row["window_rows"] == WIN_ROWS)
    require(w64["correction_fallback_events"] == total_events
            and w64["active_groups_total"] == total_groups,
            "M105 W64 ledger mismatch")
    require(partition_result["correction_service_tokens"]
            == window_result["correction_service_tokens"]
            == w64["correction_fallback_token_envelope"],
            "loop order changed service work")
    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M107 analyzer changed during execution")

    payload = {
        "schema": "m107_w64_pingpong_schedule_replay_result_v1",
        "status": "PASS_W64_TWO_BANK_DESCRIPTOR_FILL_DRAIN_REPLAY_ACCUMULATOR_PORT_CUT",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "m105_auditor_sha256": EXPECTED_SHA256["m105_script"],
            "m105_result_sha256": EXPECTED_SHA256["m105_result"],
            "m106_contract_sha256": EXPECTED_SHA256["m106_contract"],
            "m106_run_complete_sha256": sha256(M106_RUN / "RUN_COMPLETE.txt"),
            "weight_payload_sha256": weight_shas,
        },
        "scope": {
            "heldout_samples": [5, 6, 7, 8, 9],
            "operators": 4,
            "partitions_per_operator": 432,
            "rows_per_phase": 3000,
            "window_rows": WIN_ROWS,
            "windows_per_phase": WINDOWS_PER_PHASE,
            "windows": len(partition_major),
            "descriptor_ingress": "one correction/fallback event per cycle plus one close cycle per window",
            "correction_drain": "three load cycles per active key plus one event cycle per event",
            "banks": 2,
            "pwp_policy": "all frozen cap11 PWP service tokens conservatively serialized in addition to correction schedule",
        },
        "work_conservation": {
            "events": total_events,
            "groups": total_groups,
            "correction_tokens": total_events + 3 * total_groups,
            "pwp_tokens": PWP_TOKENS,
            "baseline_tokens": BASELINE_TOKENS,
        },
        "partition_major_schedule": partition_result,
        "window_major_schedule": window_result,
        "accumulator_contract": {
            "selected_order": "window_major",
            "signed_bits": 24,
            "dense_bound_signed_bits": 21,
            "double_window_bytes": 294912,
            "bank_and_rmw_replay": False,
            "finite_width_miter": False,
        },
        "admission": {
            "exact_heldout_event_and_group_replay": True,
            "two_bank_fill_drain_cycle_schedule": True,
            "pwp_tokens_serially_charged": True,
            "accumulator_schedule": False,
            "shared_weight_sram_schedule": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS M107 partition_ratio={:.12f} window_ratio={:.12f} output={}".format(
        partition_result["same_clock_service_island_ratio"],
        window_result["same_clock_service_island_ratio"], args.output),
        flush=True)


if __name__ == "__main__":
    main()

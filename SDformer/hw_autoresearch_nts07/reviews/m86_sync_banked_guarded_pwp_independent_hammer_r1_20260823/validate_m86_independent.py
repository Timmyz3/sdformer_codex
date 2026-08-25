#!/usr/bin/env python3
"""Independent M86 identity, binary-address, and cycle-accounting oracle.

This checker intentionally does not import any M78--M86 producer module.  It
parses the frozen records, offset table, and canonical 74-byte metadata image
directly and reconstructs the bank/row sequence from the mathematical layout.
"""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import List


PHASES = 1728
ENTRIES = 128
ROWS = 460
WORDS = {0: 24, 1: 27, 2: 30, 3: 33, 4: 0}
FETCH_WORDS = {0: 24, 1: 32, 2: 32, 3: 40, 4: 0}
BEATS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 1}
EXPECTED_SHA = {
    "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
        "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "rtl_m85/guarded_wordpacked_pwp_stream.sv":
        "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "rtl_m86/sync_banked_guarded_pwp_frontend.sv":
        "edb06b7f4e891d4b00c8b49ace547efdf8daf84dc19716c710a6a343dc97f781",
    "verif_m86/sync_banked_guarded_pwp_frontend_assertions.sv":
        "8733048482c677e77be88044d55215bea98d75007f9cb4b7aba83d23e6ce0dd3",
    "tb_m86/tb_sync_banked_guarded_pwp_frontend.sv":
        "1ae9433a031a772d215d4ac032255042cffb757cd8c0a9e4f3934a76641b1386",
    "dc_handoff/filelists/date_m86_sync_banked_guarded_pwp_frontend_vcs.f":
        "4038e7629d90957e23ed36387f8cbdbf3c2e161df1955b91047cdeab13f25230",
    "dc_handoff/scripts/run_vcs_m86_sync_banked_guarded_pwp_actual_records_sva.sh":
        "10fb629d811160a12eff26d377d122f4def3f51134a897201774597d397bc3d5",
    "contracts/m86_sync_banked_guarded_pwp_frontend_vcs_contract_r1_20260823.json":
        "d7bb4929abca9d3f9562c3a7d85bdaa769734a877e064c50eaa6173fc519578a",
    "results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin":
        "52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0",
}
EXPECTED_EXTERNAL_SHA = {
    "records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_little_fields(raw: bytes, count: int, width: int) -> List[int]:
    value = int.from_bytes(raw, "little")
    return [(value >> (index * width)) & ((1 << width) - 1)
            for index in range(count)]


def parse_pass_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"PASS M86 sync-bank actual-record replay phases=(\d+) "
        r"descriptors=(\d+) outputs=(\d+) beats=(\d+) "
        r"always_ready_ii_checks=(\d+) stress_phases=(\d+) "
        r"backpressure_cycles=(\d+) fifo_full_cycles=(\d+) "
        r"duplicate_row_attacks=(\d+)"
    )
    match = pattern.search(text)
    require(match is not None, "sealed M86 PASS line missing")
    require(not re.search(
        r"failed at|Offending|^Error|^Fatal|watchdog timeout", text,
        flags=re.IGNORECASE | re.MULTILINE), "failure signature in sealed log")
    names = ("phases", "descriptors", "outputs", "beats",
             "always_ready_ii_checks", "stress_phases",
             "backpressure_cycles", "fifo_full_cycles",
             "duplicate_row_attacks")
    return dict(zip(names, map(int, match.groups())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--sealed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = {}
    for relative, expected in EXPECTED_SHA.items():
        observed = sha256(args.hw_root / relative)
        require(observed == expected, f"exact-SHA drift: {relative}")
        identities[relative] = observed
    external_identity = {
        "records": sha256(args.records),
        "offsets": sha256(args.offsets),
    }
    require(external_identity == EXPECTED_EXTERNAL_SHA,
            f"external input drift: {external_identity}")

    complete = {}
    for line in (args.sealed_run / "RUN_COMPLETE.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            complete[key] = value
    require(complete.get("status") ==
            "PASS_M86_SYNC_BANK_ACTUAL_RECORD_VCS_SVA",
            "sealed status is not PASS")
    for field in ("compiled_sram_macro", "real_escape_fallback",
                  "rtl_cycle_speedup", "paper_ppa_ready",
                  "system_speedup", "headline"):
        require(complete.get(field) == "false",
                f"claim boundary drift: {field}")
    pass_log = parse_pass_log(args.sealed_run / "sim.raw.log")

    offsets_raw = args.offsets.read_bytes()
    records_raw = args.records.read_bytes()
    metadata_raw = (args.hw_root / (
        "results/m85_canonical_74b_phase_metadata_r1_20260823/"
        "m85_phase_metadata_74b.bin")).read_bytes()
    require(len(offsets_raw) == (PHASES + 1) * 4,
            "offset-table byte count mismatch")
    require(len(metadata_raw) == PHASES * 74,
            "metadata byte count mismatch")
    offsets = struct.unpack(f"<{PHASES + 1}I", offsets_raw)
    require(offsets[0] == 0 and offsets[-1] == len(records_raw),
            "offset endpoints mismatch")
    require(all(left < right for left, right in zip(offsets, offsets[1:])),
            "offsets not strictly increasing")

    width_counts = Counter()
    beat_counts = Counter()
    cross_row_fetches = 0
    regular_fetches = 0
    maximum_row = -1
    maximum_fetch_end = 0
    maximum_terminal = 0
    payload_words_sum = 0
    payload_rows_needed_sum = 0
    escape_locations = []

    for phase in range(PHASES):
        record = records_raw[offsets[phase]:offsets[phase + 1]]
        metadata = metadata_raw[phase * 74:(phase + 1) * 74]
        require(record[:48] == metadata[:48],
                f"record/metadata header mismatch phase={phase}")
        codes = unpack_little_fields(metadata[:48], ENTRIES, 3)
        bases = unpack_little_fields(metadata[48:], 16, 13)
        cursor = 0
        for pattern in range(16):
            require(bases[pattern] == cursor,
                    f"noncanonical base phase={phase} pattern={pattern}")
            for block in range(8):
                code = codes[pattern * 8 + block]
                require(code in WORDS, f"reserved code phase={phase}")
                if code == 4:
                    escape_locations.append([phase, pattern, block])
                    width_counts[12] += 1
                    beat_counts[12] += 1
                    continue
                width = 8 + code
                width_counts[width] += 1
                beat_counts[width] += BEATS[code]
                maximum_fetch_end = max(maximum_fetch_end,
                                        cursor + FETCH_WORDS[code])
                require(cursor + FETCH_WORDS[code] <= ROWS * 8,
                        f"fetch overflow phase={phase} cursor={cursor}")
                for beat in range(BEATS[code]):
                    logical = cursor + beat * 8
                    base_row, base_bank = divmod(logical, 8)
                    rows = [base_row + (bank < base_bank)
                            for bank in range(8)]
                    require(max(rows) < ROWS,
                            f"bank row OOB phase={phase} logical={logical}")
                    maximum_row = max(maximum_row, max(rows))
                    cross_row_fetches += base_bank != 0
                    regular_fetches += 1
                cursor += WORDS[code]
        require(len(record) >= 48 + cursor * 4,
                f"short record payload phase={phase}")
        maximum_terminal = max(maximum_terminal, cursor)
        payload_words_sum += cursor
        payload_rows_needed_sum += (cursor + 7) // 8

    require(sum(width_counts.values()) == PHASES * ENTRIES,
            "descriptor total mismatch")
    total_beats = sum(beat_counts.values())
    require(total_beats == regular_fetches + len(escape_locations),
            "beat reconstruction mismatch")
    require(total_beats == 835383, "frozen beat count mismatch")
    require(escape_locations == [[1242, 5, 5]],
            f"escape identity drift: {escape_locations}")
    require(maximum_row == 459 and maximum_fetch_end == 3680,
            "bank boundary mismatch")

    loader_cycles = PHASES * ROWS
    phase_commit_cycles = PHASES
    serialized_frontend_service = loader_cycles + phase_commit_cycles + total_beats
    useful_row_cycles = payload_rows_needed_sum
    forced_zero_row_cycles = loader_cycles - useful_row_cycles
    actual_ii_extent = 1600 * (ENTRIES - 1)
    all_phase_ii_extent = PHASES * (ENTRIES - 1)
    require(actual_ii_extent == 203200, "II extent arithmetic mismatch")
    require(pass_log == {
        "phases": 1728,
        "descriptors": 221184,
        "outputs": 221184,
        "beats": 835383,
        "always_ready_ii_checks": 203200,
        "stress_phases": 128,
        "backpressure_cycles": 5261,
        "fifo_full_cycles": 4940,
        "duplicate_row_attacks": 1,
    }, f"sealed PASS counters drift: {pass_log}")

    result = {
        "schema": "m86_independent_identity_bank_and_cycle_oracle_v1",
        "status": "PASS_M86_SCOPED_FUNCTIONAL_EVIDENCE_WITH_UNCHARGED_LOADER",
        "identity": {
            "reviewed_exact_sha256": identities,
            "external_sha256": external_identity,
            "sealed_run_receipt_sha256": sha256(
                args.sealed_run / "RUN_COMPLETE.txt"),
            "sealed_sim_log_sha256": sha256(
                args.sealed_run / "sim.raw.log"),
        },
        "independent_binary_reconstruction": {
            "phases": PHASES,
            "descriptors": PHASES * ENTRIES,
            "width_counts": {str(key): width_counts[key]
                             for key in sorted(width_counts)},
            "beats_by_width": {str(key): beat_counts[key]
                               for key in sorted(beat_counts)},
            "bank_read_beats_including_escape_control": total_beats,
            "regular_bank_fetches": regular_fetches,
            "cross_row_fetches": cross_row_fetches,
            "maximum_bank_row": maximum_row,
            "maximum_fetch_end_words": maximum_fetch_end,
            "maximum_terminal_words": maximum_terminal,
            "escape_locations": escape_locations,
        },
        "cycle_scope_audit": {
            "reported_always_ready_ii_checks": actual_ii_extent,
            "all_1728_phase_within_phase_intervals": all_phase_ii_extent,
            "intervals_excluded_by_128_stress_phases":
                all_phase_ii_extent - actual_ii_extent,
            "first_start_and_cross_phase_intervals_not_in_ii_counter": True,
            "mandatory_loader_row_accepts": loader_cycles,
            "phase_commit_accepts": phase_commit_cycles,
            "bank_read_issues": total_beats,
            "serialized_loader_commit_read_lower_bound_cycles":
                serialized_frontend_service,
            "loader_plus_commit_overhead_vs_bank_read_issues":
                (loader_cycles + phase_commit_cycles) / total_beats,
            "loader_share_of_serialized_lower_bound":
                loader_cycles / serialized_frontend_service,
            "useful_payload_words": payload_words_sum,
            "minimum_rows_for_actual_terminals": useful_row_cycles,
            "forced_zero_row_accepts_due_to_fixed_460_row_completion":
                forced_zero_row_cycles,
            "fixed_row_loader_overhead_vs_minimum_rows":
                forced_zero_row_cycles / useful_row_cycles,
            "interpretation": (
                "The 203200 checks prove only consecutive descriptor starts "
                "inside the 1600 always-ready phases. They do not charge the "
                "460 payload writes and one phase commit required before each "
                "phase, nor the 128 stressed phases, phase boundaries, escape "
                "fallback, or downstream accumulation."
            ),
        },
        "sealed_vcs": pass_log,
        "claim_boundary_confirmed": {
            "one_cycle_registered_read_interface_functional": True,
            "compiled_sram_macro": False,
            "real_escape_fallback": False,
            "rtl_cycle_speedup": False,
            "m78_shared32_1p409x_re_admitted": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "beats": total_beats,
        "max_row": maximum_row,
        "loader_cycles": loader_cycles,
        "serialized_lower_bound": serialized_frontend_service,
        "forced_zero_rows": forced_zero_row_cycles,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

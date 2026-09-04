#!/usr/bin/env python3
"""Independent, non-producer validation for the M79 directed VCS milestone."""

import hashlib
import json
import math
import os
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
SEALED = HW / "dc_handoff/runs/m79_precision_elastic_pwp_vcs_r1_sealed_20260823"
RERUN = REVIEW / "independent_vcs_rerun"
LATENCY = REVIEW / "independent_latency_vcs"
M78 = HW / "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json"

INPUTS = {
    "rtl_m79/precision_elastic_pwp_beat_assembler.sv":
        "00bf98d682759906a932c5518561393c5fc74104407e9df35ec3af42835fcad7",
    "verif_m79/precision_elastic_pwp_beat_assembler_assertions.sv":
        "dbea10891e951a8b502f02776e14915e2cf67c5d70ff024530c5a5599ecadad7",
    "tb_m79/tb_precision_elastic_pwp_beat_assembler.sv":
        "62ca1d6dd375c2a0307eb81d91cb5eb54466589fbd07789d479f374f45e92b87",
    "dc_handoff/filelists/date_m79_precision_elastic_pwp_directed_vcs.f":
        "1403f615fed184aaa669df5689dace47ab9b9329e999a0eae4fc7288ef76d7c2",
    "contracts/m79_precision_elastic_pwp_assembler_vcs_contract_r1_20260823.json":
        "7ac5121c2c01885fbb227fd6c386f626b39d9440469d0f8548b41a8122a7ae7a",
    "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json":
        "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
}
EXPECTED_PASS = (
    "PASS M79 directed transactions=136 beats=512 escapes=8 stalls=12 "
    "lanes=96 protocol_attacks=2 widths=8,9,10,11,12")
EXPECTED_LATENCY_PASS = (
    "PASS M79 independent hammer checks=12 signed_extremes=4 widths=4 "
    "padding_attacks=3 missing_last=1")


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
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def signed_extend(raw, width, out_width=12):
    raw &= (1 << width) - 1
    if raw & (1 << (width - 1)):
        raw -= 1 << width
    require(-(1 << (out_width - 1)) <= raw < (1 << (out_width - 1)),
            "signed extension exceeds canonical output")
    return raw


def replay_directed_payloads():
    by_width = {}
    total_beats = 0
    for width in (8, 9, 10, 11):
        seen = set()
        transactions = 0
        beat_count = int(math.ceil(96 * width / 256.0))
        padding = beat_count * 256 - 96 * width
        for txid in range(128):
            if 8 + (txid % 4) != width:
                continue
            transactions += 1
            packed = 0
            golden = []
            for lane in range(96):
                unsigned = (txid * 17 + lane * 13 + width * 7) % (1 << width)
                signed = signed_extend(unsigned, width)
                seen.add(signed)
                golden.append(signed)
                packed |= unsigned << (lane * width)
            require(packed >> (96 * width) == 0,
                    "nonzero generated padding width {}".format(width))
            beats = [(packed >> (beat * 256)) & ((1 << 256) - 1)
                     for beat in range(beat_count)]
            rebuilt = sum(value << (beat * 256)
                          for beat, value in enumerate(beats))
            for lane, expected in enumerate(golden):
                raw = (rebuilt >> (lane * width)) & ((1 << width) - 1)
                require(signed_extend(raw, width) == expected,
                        "unpack mismatch width {} lane {}".format(width, lane))
            require(rebuilt >> (96 * width) == 0,
                    "rebuilt padding mismatch width {}".format(width))
            total_beats += beat_count
        require(transactions == 32, "directed transaction count drift")
        by_width[str(width)] = {
            "transactions": transactions,
            "payload_bits": 96 * width,
            "payload_bytes": 96 * width // 8,
            "beats": beat_count,
            "padding_bits": padding,
            "unique_signed_codewords_hit": len(seen),
            "codeword_space": 1 << width,
            "minimum_hit": min(seen),
            "maximum_hit": max(seen),
            "both_signed_extremes_hit": (
                -(1 << (width - 1)) in seen and
                (1 << (width - 1)) - 1 in seen),
        }
    require(total_beats == 512, "directed beat total drift")
    return by_width, total_beats


def validate_vcs_run(run_dir):
    require((run_dir / "compile.rc").read_text().strip() == "0",
            "compile rc nonzero: {}".format(run_dir))
    require((run_dir / "sim.rc").read_text().strip() == "0",
            "sim rc nonzero: {}".format(run_dir))
    sim = (run_dir / "sim.raw.log").read_text(encoding="utf-8")
    compile_log = (run_dir / "compile.raw.log").read_text(encoding="utf-8")
    require(sim.count(EXPECTED_PASS) == 1,
            "expected directed PASS line missing or repeated")
    for cover in ("cp_width8", "cp_width9", "cp_width10", "cp_width11",
                  "cp_escape12", "cp_output_stall", "cp_protocol_fault"):
        require(cover in sim, "missing cover report {}".format(cover))
    forbidden = ("failed at", "Offending", "\nError", "\nFatal")
    require(not any(token in sim for token in forbidden),
            "simulation failure signature")
    require("Warning-[" not in compile_log and "Error-[" not in compile_log,
            "compile warning/error signature")
    return {
        "compile_rc": 0,
        "sim_rc": 0,
        "pass_line_count": sim.count(EXPECTED_PASS),
        "cover_names_observed": 7,
        "sim_log_sha256": sha256(run_dir / "sim.raw.log"),
        "compile_log_sha256": sha256(run_dir / "compile.raw.log"),
    }


def main():
    observed_inputs = {}
    for relative, expected in INPUTS.items():
        observed = sha256(HW / relative)
        require(observed == expected, "input SHA drift: {}".format(relative))
        observed_inputs[relative] = observed

    contract = strict_json(HW / next(
        path for path in INPUTS if path.startswith("contracts/")))
    require(contract["geometry"]["beats_by_width"] ==
            {"8": 3, "9": 4, "10": 4, "11": 5, "12": 0},
            "contract beat geometry drift")

    geometry, total_beats = replay_directed_payloads()
    sealed = validate_vcs_run(SEALED)
    rerun = validate_vcs_run(RERUN)

    latency_log = (LATENCY / "sim.raw.log").read_text(encoding="utf-8")
    require(latency_log.count(EXPECTED_LATENCY_PASS) == 1,
            "independent latency PASS line missing or repeated")
    observed_ii = {}
    for width, beats in ((8, 3), (9, 4), (10, 4), (11, 5)):
        expected = ("M79_INDEPENDENT_II width={} beats={} "
                    "command_ii_cycles={}").format(width, beats, beats + 1)
        require(expected in latency_log, "missing II result: {}".format(expected))
        observed_ii[str(width)] = beats + 1

    m78 = strict_json(M78)
    cap11 = next(item for item in m78["configurations"]
                 if item["signed_width_cap"] == 11)
    shared32 = next(item for item in cap11["cycle_simulations"]
                    if item["port"] == "SHARED_32B")
    pwp_uses = sum(cap11["heldout"]["pwp_uses_by_width"].values())
    no_overlap_cycles = shared32["candidate_cycles"] + pwp_uses
    no_overlap_speedup = shared32["bit_sparse_cycles"] / float(no_overlap_cycles)

    producer_mode = oct(os.stat(str(SEALED)).st_mode & 0o777)
    producer_file_modes = sorted(set(
        oct(os.stat(str(path)).st_mode & 0o777)
        for path in SEALED.iterdir() if path.is_file()))
    runner = HW / "dc_handoff/scripts/run_vcs_m79_precision_elastic_pwp_directed_sva.sh"

    payload = {
        "schema": "m79_precision_elastic_pwp_vcs_independent_hammer_v1",
        "status": "PASS_ISOLATED_FUNCTION_NO_GO_SHARED32_OR_DATE_HEADLINE",
        "identity": {
            "production_inputs": observed_inputs,
            "production_runner_sha256_not_self_pinned": sha256(runner),
            "independent_tb_sha256": sha256(REVIEW / "tb_m79_independent_hammer.sv"),
        },
        "independent_bit_arithmetic": {
            "regular_transactions_recomputed": 128,
            "accepted_regular_beats_recomputed": total_beats,
            "widths": geometry,
            "escape12": {
                "payload_beats": 0,
                "canonical_output_zero": True,
                "meaning": "control token only; fallback weight delivery is outside M79",
            },
            "padding_attack_coverage": {
                "producer": [9],
                "independent_vcs": [9, 10, 11],
                "width8_has_no_padding": True,
            },
        },
        "vcs": {
            "producer_sealed_run": sealed,
            "exact_sha_independent_rerun": rerun,
            "independent_latency_and_negative_test": {
                "pass": True,
                "observed_minimum_back_to_back_command_ii_cycles": observed_ii,
                "model_beat_cycles": {"8": 3, "9": 4, "10": 4, "11": 5},
                "padding_attacks": [9, 10, 11],
                "missing_final_last_attack": True,
                "sim_log_sha256": sha256(LATENCY / "sim.raw.log"),
                "compile_log_sha256": sha256(LATENCY / "compile.raw.log"),
            },
        },
        "m78_shared32_audit": {
            "reported_candidate_cycles": shared32["candidate_cycles"],
            "reported_bit_sparse_cycles": shared32["bit_sparse_cycles"],
            "reported_speedup_vs_bit_sparse": shared32["speedup_vs_bit_sparse"],
            "regular_pwp_uses": pwp_uses,
            "catalog_escape_entries": cap11["ineligible_output_block_entries"],
            "heldout_escape_uses": cap11["heldout"]["block_local_escape_rows"],
            "rtl_observation": (
                "single-buffer assembler minimum consecutive command II is beat_count+1; "
                "M78 charges beat_count and has no ordered schedule proving the extra cycle hidden"),
            "one_unhidden_cycle_per_pwp_sensitivity": {
                "candidate_cycles": no_overlap_cycles,
                "speedup_vs_bit_sparse": no_overlap_speedup,
                "not_a_corrected_headline": True,
            },
        },
        "evidence_hygiene": {
            "producer_directory_mode": producer_mode,
            "producer_top_file_modes": producer_file_modes,
            "producer_named_sealed_but_write_bits_present": True,
            "runner_itself_absent_from_pinned_input_set": True,
            "producer_output_hash_manifest_present": False,
        },
        "decision": {
            "isolated_directed_function": "GO",
            "m78_shared32_cycle_assumption": "NO_GO",
            "finite_queue_and_macro_integration": "NO_GO",
            "date_performance_or_ppa_claim": "NO_GO",
        },
    }
    output = REVIEW / "m79_independent_hammer.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    manifest_paths = (
        REVIEW / "README.md",
        REVIEW / "validate_m79_independent_hammer.py",
        REVIEW / "tb_m79_independent_hammer.sv",
        output,
        RERUN / "input_sha256.txt",
        RERUN / "compile.raw.log",
        RERUN / "sim.raw.log",
        RERUN / "RUN_COMPLETE.txt",
        LATENCY / "compile.raw.log",
        LATENCY / "sim.raw.log",
    )
    manifest = {
        str(path.relative_to(REVIEW)): sha256(path) for path in manifest_paths
    }
    (REVIEW / "review_artifact_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M79 independent validator output={}".format(output))


if __name__ == "__main__":
    main()

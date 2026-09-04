#!/usr/bin/env python3
"""Independent validation of M82 zero-bubble PWP stream RTL and evidence."""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
SEALED = HW / "dc_handoff/runs/m82_zero_bubble_pwp_stream_vcs_r1_sealed_20260823"
RERUN = REVIEW / "independent_vcs_rerun"
MIXED = REVIEW / "independent_mixed_vcs_r4"
M79 = HW / "reviews/m79_precision_elastic_pwp_vcs_independent_hammer_r1_20260823/m79_independent_hammer.json"
M81_RESULT = HW / "results/m81_interleaved_word_pwp_buffer_valid825_internal_dev_r2_20260823/m81_interleaved_word_pwp_buffer.json"
M81_REVIEW = HW / "reviews/m81_interleaved_word_pwp_buffer_independent_hammer_r2_20260823/m81_interleaved_word_pwp_buffer_independent_hammer_r2_review.json"

INPUTS = {
    "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
        "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "verif_m82/zero_bubble_elastic_pwp_stream_assertions.sv":
        "159bd670e82109fce9f5fa3f27570996a40e79067f63e7f79dbb25eab01013f9",
    "tb_m82/tb_zero_bubble_elastic_pwp_stream.sv":
        "9bd7f53880c57c0d9b2b86cdb1350020d27d90c14d0c3a81652c34a451a2e3a7",
    "dc_handoff/filelists/date_m82_zero_bubble_pwp_stream_directed_vcs.f":
        "c34fdc9bf75d76360c9e49a0b1fcf4fcc8a9876fa1294336177c1db3ef651ce2",
    "contracts/m82_zero_bubble_elastic_pwp_stream_vcs_contract_r1_20260823.json":
        "b7003149cd7ef80871239b582a04ffa07817c04eecd03bcab2cf07dd406e9272",
    "reviews/m79_precision_elastic_pwp_vcs_independent_hammer_r1_20260823/m79_independent_hammer.json":
        "85d15ef419b73fd130986fbbbe0aab09488dde055bafea17eb493d286a89c958",
}
M81_RESULT_SHA = "515e023421a2650077b61fb620b06428786eb180dd3d36d05eccec1c8d2fabad"
M81_REVIEW_SHA = "91240d28d9696a269fc70786e7dbcb9491fc6daae7ebb30af9c50770ef24804b"
EXPECTED_PASS = (
    "PASS M82 zero-bubble regular=129 escapes=8 starts=139 ii_checks=135 "
    "stalls=1 lanes=96 protocol_attacks=3 service=3,4,4,5")
EXPECTED_INDEPENDENT_PASS = (
    "PASS M82 independent hammer normal=11 escapes=3 mixed_ii=11 "
    "stall_cycles=3 attacks=8 signed_extremes=8,9,10,11 "
    "service=3,4,4,5 escape_service=1")


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


def validate_producer_style_run(run_dir):
    require((run_dir / "compile.rc").read_text().strip() == "0",
            "compile rc nonzero")
    require((run_dir / "sim.rc").read_text().strip() == "0", "sim rc nonzero")
    compile_log = (run_dir / "compile.raw.log").read_text(encoding="utf-8")
    sim_log = (run_dir / "sim.raw.log").read_text(encoding="utf-8")
    require(sim_log.count(EXPECTED_PASS) == 1,
            "producer PASS missing or repeated")
    for cover in ("cp_width8", "cp_width9", "cp_width10", "cp_width11",
                  "cp_escape12", "cp_zero_bubble_boundary",
                  "cp_output_stall", "cp_protocol_fault"):
        require(cover in sim_log, "missing producer cover {}".format(cover))
    require("Warning-[" not in compile_log and "Error-[" not in compile_log,
            "compile warning/error signature")
    require(not any(token in sim_log for token in
                    ("failed at", "Offending", "\nError", "\nFatal",
                     "watchdog timeout")), "simulation failure signature")
    return {
        "compile_rc": 0,
        "sim_rc": 0,
        "pass_line_count": 1,
        "cover_names_observed": 8,
        "compile_log_sha256": sha256(run_dir / "compile.raw.log"),
        "sim_log_sha256": sha256(run_dir / "sim.raw.log"),
    }


def recompute_directed_vectors():
    widths = {}
    total_beats = 0
    for width in (8, 9, 10, 11):
        seen = set()
        beats = int(math.ceil(96 * width / 256.0))
        for txid in range(128):
            if 8 + txid % 4 != width:
                continue
            packed = 0
            golden = []
            for lane in range(96):
                unsigned = (txid * 17 + lane * 13 + width * 7) % (1 << width)
                signed = unsigned
                if signed >= (1 << (width - 1)):
                    signed -= 1 << width
                packed |= unsigned << (lane * width)
                golden.append(signed)
                seen.add(signed)
            rebuilt = sum(
                ((packed >> (beat * 256)) & ((1 << 256) - 1))
                << (beat * 256) for beat in range(beats))
            require(rebuilt >> (96 * width) == 0,
                    "nonzero directed padding")
            for lane, expected in enumerate(golden):
                raw = (rebuilt >> (lane * width)) & ((1 << width) - 1)
                if raw >= (1 << (width - 1)):
                    raw -= 1 << width
                require(raw == expected, "directed signed unpack mismatch")
            total_beats += beats
        widths[str(width)] = {
            "transactions": 32,
            "payload_bits": 96 * width,
            "payload_bytes": 12 * width,
            "service_beats": beats,
            "padding_bits": beats * 256 - 96 * width,
            "unique_signed_codewords_hit": len(seen),
            "codeword_space": 1 << width,
            "both_signed_extremes_hit": (
                -(1 << (width - 1)) in seen and
                (1 << (width - 1)) - 1 in seen),
        }
    require(total_beats == 512, "directed regular beat count drift")
    return widths


def validate_independent_mixed():
    compile_log = (MIXED / "compile.raw.log").read_text(encoding="utf-8")
    sim_log = (MIXED / "sim.raw.log").read_text(encoding="utf-8")
    require(sim_log.count(EXPECTED_INDEPENDENT_PASS) == 1,
            "independent mixed PASS missing or repeated")
    require("Warning-[" not in compile_log and "Error-[" not in compile_log,
            "independent compile warning/error")
    require(not any(token in sim_log for token in
                    ("failed at", "Offending", "\nError", "\nFatal")),
            "independent sim failure")
    observed = Counter()
    service = {8: 3, 9: 4, 10: 4, 11: 5, 12: 1}
    for width, cycles in service.items():
        line = "M82_INDEPENDENT_II previous_width={} observed_cycles={}".format(
            width, cycles)
        observed[str(width)] = sim_log.count(line)
    require(sum(observed.values()) == 11, "mixed II count drift")
    require(all(observed[str(width)] > 0 for width in service),
            "mixed II misses a width/escape class")
    require("cp_protocol_fault, 124 attempts, 8 match" in sim_log,
            "independent protocol fault coverage drift")
    return {
        "pass": True,
        "mixed_ii_checks": 11,
        "ii_match_counts_by_previous_width": dict(sorted(observed.items())),
        "observed_service_cycles": {str(k): v for k, v in service.items()},
        "signed_extreme_widths": [8, 9, 10, 11],
        "adjacent_escape_checked": True,
        "stall_cycles_held": 3,
        "protocol_attacks": {
            "start_mid_transaction": 1,
            "premature_last": 1,
            "missing_final_last": 1,
            "nonzero_padding": 3,
            "malformed_escape": 2,
        },
        "compile_log_sha256": sha256(MIXED / "compile.raw.log"),
        "sim_log_sha256": sha256(MIXED / "sim.raw.log"),
    }


def main():
    observed_inputs = {}
    for relative, expected in INPUTS.items():
        observed = sha256(HW / relative)
        require(observed == expected, "input SHA drift: {}".format(relative))
        observed_inputs[relative] = observed
    require(sha256(M81_RESULT) == M81_RESULT_SHA, "M81 result SHA drift")
    require(sha256(M81_REVIEW) == M81_REVIEW_SHA, "M81 review SHA drift")

    contract = strict_json(HW / next(
        path for path in INPUTS if path.startswith("contracts/")))
    require(contract["geometry"]["regular_beats"] ==
            {"8": 3, "9": 4, "10": 4, "11": 5},
            "contract service geometry drift")
    require(contract["geometry"]["escape_control_beats"] == 1,
            "escape service drift")

    sealed = validate_producer_style_run(SEALED)
    rerun = validate_producer_style_run(RERUN)
    independent = validate_independent_mixed()
    widths = recompute_directed_vectors()

    m79 = strict_json(M79)
    m81 = strict_json(M81_RESULT)
    m81_review = strict_json(M81_REVIEW)
    m79_audit = m79["m78_shared32_audit"]
    original = m79_audit["reported_speedup_vs_bit_sparse"]
    sensitivity = m79_audit["one_unhidden_cycle_per_pwp_sensitivity"]
    escape_cycles = m79_audit["heldout_escape_uses"]
    candidate_with_escape = m79_audit["reported_candidate_cycles"] + escape_cycles
    speedup_with_escape = (
        m79_audit["reported_bit_sparse_cycles"] / float(candidate_with_escape))

    producer_mode = oct(os.stat(str(SEALED)).st_mode & 0o777)
    producer_file_modes = sorted(set(
        oct(os.stat(str(path)).st_mode & 0o777)
        for path in SEALED.iterdir() if path.is_file()))
    runner = HW / "dc_handoff/scripts/run_vcs_m82_zero_bubble_pwp_stream_directed_sva.sh"

    payload = {
        "schema": "m82_zero_bubble_pwp_stream_vcs_independent_hammer_v1",
        "status": "PASS_ISOLATED_ZERO_BUBBLE_NO_GO_BANK_FALLBACK_SYSTEM_DATE",
        "identity": {
            "production_inputs": observed_inputs,
            "m81_result_sha256": M81_RESULT_SHA,
            "m81_independent_review_sha256": M81_REVIEW_SHA,
            "production_runner_sha256_not_self_pinned": sha256(runner),
            "independent_tb_sha256": sha256(REVIEW / "tb_m82_independent_hammer.sv"),
        },
        "independent_bit_arithmetic": {
            "widths": widths,
            "escape12": {
                "stream_beats": 1,
                "beat_must_be_last": True,
                "beat_data_must_be_zero": True,
                "output_payload_zero": True,
                "real_fallback_execution_inside_m82": False,
            },
        },
        "vcs": {
            "producer_sealed_run": sealed,
            "exact_sha_independent_rerun": rerun,
            "independent_mixed_stream": independent,
        },
        "m79_to_m82_throughput": {
            "m79_observed_regular_command_ii": {"8": 4, "9": 5, "10": 5, "11": 6},
            "m82_observed_regular_start_ii": {"8": 3, "9": 4, "10": 4, "11": 5},
            "extra_cycle_removed_per_regular_pwp": True,
            "conditional_recovered_cycles_vs_m79_no_overlap_sensitivity":
                m79_audit["regular_pwp_uses"],
            "m79_no_overlap_sensitivity_speedup": sensitivity["speedup_vs_bit_sparse"],
            "m78_reported_speedup_restored_at_isolated_stream_boundary": original,
            "not_re_admitted_as_module_or_system_speedup": True,
            "reason": (
                "M82 proves an always-ready stream boundary, but does not prove M81 bank "
                "response timing, ordered correction arbitration, fallback, or finite queues"),
        },
        "escape_control_cycle_sensitivity": {
            "m78_heldout_escape_uses": escape_cycles,
            "m82_one_control_beat_each_not_explicitly_charged_by_m78": True,
            "candidate_cycles_if_all_362_control_beats_are_unhidden": candidate_with_escape,
            "speedup_vs_bit_sparse": speedup_with_escape,
            "materiality": "numerically negligible but must be charged in integrated replay",
        },
        "m81_boundary": {
            "rtl_bank_integrated": False,
            "payload_banks_assumed": m81["architecture"]["payload_banks"],
            "bits_per_bank_word": m81["architecture"]["bits_per_bank_word"],
            "cross_row_fraction": m81_review["retained_r1_bank_evidence"]["cross_row_fraction"],
            "independent_row_addresses_required": True,
            "barrel_reorder_sta_area_energy_available": False,
            "finite_queue_or_synchronous_response_proof": False,
        },
        "evidence_hygiene": {
            "producer_directory_mode": producer_mode,
            "producer_top_file_modes": producer_file_modes,
            "producer_named_sealed_but_write_bits_present": True,
            "producer_output_hash_manifest_present": False,
            "runner_itself_absent_from_pinned_input_set": True,
        },
        "decisions": {
            "isolated_zero_bubble_stream_rtl": "GO",
            "retire_m79_command_separated_frontend": "GO",
            "m81_bank_integrated_zero_bubble": "NO_GO",
            "real_escape_fallback_and_ordering": "NO_GO",
            "m78_shared32_speedup_re_admission": "NO_GO",
            "date_performance_ppa_or_best_paper": "NO_GO",
        },
        "findings": {
            "scoped_p0": [],
            "promotion_p0": [
                "M81 synchronous eight-bank address/response/barrel path is absent",
                "real bit-sparse escape fallback and accumulator ordering are absent",
                "frozen ordered finite-queue replay and Synopsys PPA are absent",
            ],
            "p1": [
                "zero-bubble interval is checked procedurally, not asserted in SVA",
                "SVA lacks input hold-under-backpressure and beat-count/tag conservation properties",
                "sealed output is writable and runner/output identities are not fully sealed",
            ],
        },
    }
    output = REVIEW / "m82_independent_hammer.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    manifest_paths = (
        REVIEW / "README.md",
        REVIEW / "validate_m82_independent_hammer.py",
        REVIEW / "tb_m82_independent_hammer.sv",
        output,
        RERUN / "input_sha256.txt",
        RERUN / "compile.raw.log",
        RERUN / "sim.raw.log",
        RERUN / "RUN_COMPLETE.txt",
        MIXED / "compile.raw.log",
        MIXED / "sim.raw.log",
    )
    manifest = {
        str(path.relative_to(REVIEW)): sha256(path) for path in manifest_paths
    }
    (REVIEW / "review_artifact_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M82 independent validator output={}".format(output))


if __name__ == "__main__":
    main()

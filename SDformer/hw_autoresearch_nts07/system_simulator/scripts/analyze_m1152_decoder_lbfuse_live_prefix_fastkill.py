#!/usr/bin/env python3
"""M1152 read-only LB-FUSE fast-kill on a frozen live M1111DR2 prefix.

The live producer is never opened for writing and only the exact contracted
prefix is consumed.  The candidate is compared with the already legal
source-order A1 schedule at the same 96 lanes, 240 KiB, six-bank 1RW Acc24
storage, and dense output commit.  Moving that same state into an ordinary
three-row line buffer does not remove its read/modify/write operations.
"""

from collections import defaultdict
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import sys


getcontext().prec = 40
HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m1152_decoder_lbfuse_live_prefix_fastkill_contract_r1_20260830.json"
OUT = HW / "results/m1152_decoder_lbfuse_live_prefix_fastkill_r1_20260830"
MODULES = {
    0: ("D0", 40, 384),
    1: ("D1", 80, 192),
    2: ("D2", 160, 96),
    3: ("D3", 320, 96),
}


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("non-finite JSON: " + token)))


def strict_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"))


def ratio(num, den):
    require(int(den) > 0, "zero denominator")
    return format(Decimal(int(num)) / Decimal(int(den)), ".12f")


def seal(directory):
    members = sorted(p for p in directory.iterdir() if p.is_file() and
                     p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(p), p.name)
                                for p in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                     encoding="utf-8")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def main():
    require(len(sys.argv) == 1, "zero arguments only")
    contract = strict_json(CONTRACT)
    require(contract["schema"] ==
            "m1152_decoder_lbfuse_live_prefix_fastkill_contract_v1",
            "contract schema")
    auth = contract["authorities"]
    require(sha256(HW / auth["m1111dr2_runner_contract"]) ==
            auth["m1111dr2_runner_contract_sha256"], "runner contract drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            auth["docs359_sha256"], "docs359 drift")
    m722 = HW / auth["m722r2_result"]
    require(sha256(m722 / "SHA256SUMS.seal.sha256") ==
            auth["m722r2_outer_seal_file_sha256"], "M722 outer drift")

    frozen = contract["frozen_prefix"]
    live = HW / frozen["path"]
    require(live.is_file() and not live.is_symlink(), "live input unavailable")
    raw_rows = []
    with live.open("rb") as stream:
        for _ in range(frozen["rows"]):
            raw = stream.readline()
            require(raw and raw.endswith(b"\n"), "short/torn frozen prefix")
            raw_rows.append(raw)
    prefix = b"".join(raw_rows)
    require(len(prefix) == frozen["bytes"] and
            sha256_bytes(prefix) == frozen["sha256"], "frozen prefix drift")

    rows = []
    aggregate = defaultdict(int)
    per_module = {name: defaultdict(int) for name, _w, _c in MODULES.values()}
    expected_cycle = 0
    expected_tx = 0
    for ordinal, raw in enumerate(raw_rows):
        text = raw.decode("utf-8")
        row = strict_json_text(text)
        require((json.dumps(row, sort_keys=True, separators=(",", ":"),
                            allow_nan=False) + "\n") == text,
                "noncanonical row")
        require(row["global_call_ordinal"] == ordinal and
                row["cycle_start"] == expected_cycle and
                row["transaction_ordinal_first"] == expected_tx,
                "ordinal continuity")
        require(row["cycle_end"] - row["cycle_start"] ==
                row["diagnostic_cycles"], "cycle projection")
        require(row["transaction_ordinal_last"] + 1 ==
                row["transaction_ordinal_first"] + row["transaction_count"],
                "transaction projection")
        expected_cycle = row["cycle_end"]
        expected_tx = row["transaction_ordinal_last"] + 1
        claim = row["claim_boundary"]
        require(claim == {"diagnostic_only": True,
                "final_checkpoint_rebind_required": True,
                "paper_ppa_ready": False, "speedup_admitted": False,
                "system_speedup_admitted": False}, "claim drift")
        kinds = row["kind_summaries"]
        count = kinds["compute"]["count"]
        require(kinds["psum_read"]["count"] == count ==
                kinds["psum_write"]["count"], "RMW count mismatch")
        require(kinds["psum_read"]["traffic_bytes"] == count * 288 and
                kinds["psum_write"]["traffic_bytes"] == count * 288,
                "Acc24/96 traffic mismatch")
        require(kinds["compute"]["commit_first"] >=
                kinds["psum_read"]["return_first"] and
                kinds["psum_write"]["commit_first"] >=
                kinds["compute"]["return_first"], "dependency order")
        module = "D{}".format(row["module_ordinal"])
        require(module in per_module, "module ordinal")
        for key, value in row["diagnostic_traffic_bytes"].items():
            aggregate[key] += int(value)
            per_module[module][key] += int(value)
        for key, value in (("calls", 1), ("cycles", row["diagnostic_cycles"]),
                           ("compute_events", count),
                           ("commit_events", kinds["output_commit"]["count"])):
            aggregate[key] += int(value)
            per_module[module][key] += int(value)
        rows.append(row)

    resource = contract["resource"]
    capacities = {}
    for index, (name, width, _cout) in MODULES.items():
        acc24 = 3 * width * 96 * 3
        acc16 = 3 * width * 96 * 2
        half24 = 3 * width * 48 * 3
        capacities[name] = {
            "three_rows_acc24_full96_bytes": acc24,
            "three_rows_acc24_full96_fits_240kib": acc24 <= resource["onchip_sram_bytes_macro_rounded"],
            "three_rows_acc24_full96_fits_frozen_psum_partition": acc24 <= resource["psum_bytes"],
            "three_rows_acc16_full96_bytes": acc16,
            "three_rows_acc16_full96_fits_240kib": acc16 <= resource["onchip_sram_bytes_macro_rounded"],
            "acc16_numeric_admission_on_this_prefix": False,
            "three_rows_acc24_cout48_bytes": half24,
            "three_rows_acc24_cout48_fits_240kib": half24 <= resource["onchip_sram_bytes_macro_rounded"],
            "cout48_passes": 2,
        }

    # Fair same-port result: the current schedule already keeps psums on chip.
    # A line-buffer address remap leaves every exact RMW and dense commit intact.
    baseline_cycles = aggregate["cycles"]
    candidate_lower_bound_cycles = baseline_cycles
    baseline_psum_bytes = aggregate["psum_read"] + aggregate["psum_write"]
    candidate_same_port_psum_bytes = baseline_psum_bytes
    d3 = per_module["D3"]
    d3_split = {
        "calls_in_prefix": d3["calls"],
        "extra_input_descriptor_read_bytes_if_not_retained": d3["input_descriptor_read"],
        "input_descriptor_read_multiplier": "2.000000000000",
        "logical_weight_payload_multiplier": "1.000000000000",
        "weight_refill_command_overhead_may_increase": True,
        "compute_output_lane_passes": 2,
        "same_96_lane_throughput_claim_allowed": False,
    }

    old = strict_json(m722 / "report.json")
    require(old["status"] == "KILL_NO_RTL__FAIR_A1_ZERO_PSUM_SPILL" and
            old["decision"]["fair_a1_zero_offchip_psum_spill"] is True,
            "M722 decision drift")
    result = {
        "schema": "m1152_decoder_lbfuse_live_prefix_fastkill_result_v1",
        "status": "KILL_NO_RTL__SAME_PORT_LINE_BUFFER_IS_ADDRESS_REMAP_ONLY",
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "docs359_sha256": auth["docs359_sha256"],
            "live_prefix_rows": frozen["rows"],
            "live_prefix_bytes": frozen["bytes"],
            "live_prefix_sha256": frozen["sha256"],
            "m722r2_outer_seal_file_sha256": auth["m722r2_outer_seal_file_sha256"],
        },
        "population": {
            "calls": aggregate["calls"],
            "complete_samples": 5,
            "additional_d0_call": 1,
            "sequence": "interlaken_01_a",
            "partial_prefix": True,
            "checkpoint": "H67_ep35",
            "final_checkpoint_rebind_required": True,
        },
        "first_principles": {
            "current_psum_is_onchip": True,
            "current_offchip_psum_spill_bytes": 0,
            "exact_update_requires_read_compute_write": True,
            "same_1rw_port_preserves_rmw_event_count": True,
            "same_dense_commit_preserves_commit_event_count": True,
            "line_buffer_changes_address_lifetime_not_arithmetic_work": True,
            "source_direct_grouping_not_observable_in_compressed_prefix": True,
            "source_direct_speedup_fail_closed": True,
            "prior_complete_m722_directional_only": {
                "a1_over_lb_headline": old["decision"]["headline_a1_over_lb"],
                "lb_over_osg_groups": old["totals"]["headline_d0_d2_d3"]["lb_over_osg_groups"],
                "may_not_be_relabelled_as_current_prefix_or_final_checkpoint": True
            }
        },
        "capacity": capacities,
        "prefix_totals": dict(aggregate),
        "per_module": {key: dict(value) for key, value in per_module.items()},
        "fair_same_port_candidate": {
            "baseline_cycles": baseline_cycles,
            "candidate_executable_cycle_lower_bound": candidate_lower_bound_cycles,
            "baseline_over_candidate_speedup_upper_bound": ratio(baseline_cycles, candidate_lower_bound_cycles),
            "baseline_onchip_psum_rmw_bytes": baseline_psum_bytes,
            "candidate_onchip_psum_rmw_bytes": candidate_same_port_psum_bytes,
            "onchip_psum_byte_reduction_fraction": "0.000000000000",
            "offchip_psum_byte_reduction_fraction": "0.000000000000",
            "dense_output_commit_bytes_equal": True,
            "d3_acc24_cout48_two_pass_cost": d3_split,
        },
        "decision": {
            "cycle_gate_required": "1.200000000000",
            "cycle_gate_observed_upper_bound": "1.000000000000",
            "cycle_gate_pass": False,
            "traffic_gate_requires_cycle_regression_le": "0.050000000000",
            "traffic_gate_requires_psum_reduction_ge": "0.300000000000",
            "traffic_gate_observed_psum_reduction": "0.000000000000",
            "traffic_gate_pass": False,
            "rtl_authorized": False,
            "verdict": "KILL_LB_FUSE_AS_PERFORMANCE_MECHANISM__KEEP_EXISTING_A1_OSG_SCHEDULE",
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(not OUT.exists(), "result namespace collision")
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(result, indent=2,
        sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        "KILL_M1152_LBFUSE_NO_RTL__PARTIAL_DIAGNOSTIC_ONLY\n", encoding="utf-8")
    sealed = seal(OUT)
    print(json.dumps({"status": result["status"], "cycles": baseline_cycles,
        "psum_rmw_bytes": baseline_psum_bytes, "seal": sealed}, sort_keys=True))


if __name__ == "__main__":
    main()

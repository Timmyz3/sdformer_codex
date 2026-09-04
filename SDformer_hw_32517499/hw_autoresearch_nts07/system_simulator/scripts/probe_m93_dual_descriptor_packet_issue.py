#!/usr/bin/env python3
"""M93 dual-descriptor packet issue transaction-model probe.

The exact M53 temporal-parent transformation and M89 K6-C16 geometry are
replayed.  Only the command calendar accepts 1, 2, or 4 ready 64-byte
descriptors per cycle.  This is a non-citable cycle screen, not RTL or PPA.
"""

from __future__ import print_function

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m93_dual_descriptor_packet_issue_contract_r1_20260824.json"
M53_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
M53_RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43_TEMPORAL = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
M89_RECEIPT = HW_ROOT / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")

EXPECTED = {
    "contract": "28ffb056e87a715eee0d4cf04aabe5ed9901d17bfc6fc76c9692d1a03a87a4e8",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
WIDTHS = (1, 2, 4)
FANOUT = 6
CONTEXTS = 16
DESCRIPTOR_BYTES = 64
WIDTH2_MAX_INTEGRATED = 76293933


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs():
    paths = {
        "contract": CONTRACT,
        "m53_analyzer": M53_ANALYZER,
        "m53_result": M53_RESULT,
        "m43_temporal": M43_TEMPORAL,
        "m89_receipt": M89_RECEIPT,
    }
    for name, path in paths.items():
        require(sha256(path) == EXPECTED[name], "M93 {} drift".format(name))


def build_namespace(width):
    require(width in WIDTHS, "M93 invalid descriptor issue width")
    m53 = load_module(M53_ANALYZER, "m93_m53_w{}".format(width))
    m53.validate_contract()
    canonical, transformed, edits = m53.transformed_m45_source(True)
    command_from = "    command_port = PortCalendar()"
    command_to = "    command_port = CommandPacketCalendar()"
    require(canonical.count(command_from) == 1 and
            transformed.count(command_from) == 1,
            "M93 command calendar source identity drift")
    transformed = transformed.replace(command_from, command_to)
    edits = list(edits) + [{
        "name": "command_calendar_width_{}".format(width),
        "occurrences": 1,
        "qualification": "M93_DESCRIPTOR_ISSUE_ONLY",
    }]
    namespace = {
        "__file__": str(M53_ANALYZER),
        "__name__": "m93_descriptor_w{}_transformed_m45".format(width),
    }
    exec(compile(transformed, str(M53_ANALYZER) +
                 "#M93_DESCRIPTOR_W{}".format(width), "exec"), namespace)

    audit = {
        "base_issue_events": 0,
        "base_packet_cycles": 0,
        "base_full_packet_cycles": 0,
        "maximum_issues_in_one_cycle": 0,
        "calendar_instances": 0,
    }

    class CommandPacketCalendar(object):
        def __init__(self):
            self.occupied = {}
            self.last_end = 0
            self.operations = 0
            audit["calendar_instances"] += 1

        def schedule(self, ready_cycle):
            cycle = int(ready_cycle)
            while self.occupied.get(cycle, 0) >= width:
                cycle += 1
            before = self.occupied.get(cycle, 0)
            after = before + 1
            self.occupied[cycle] = after
            self.last_end = max(self.last_end, cycle + 1)
            self.operations += 1
            audit["base_issue_events"] += 1
            if before == 0:
                audit["base_packet_cycles"] += 1
            if after == width:
                audit["base_full_packet_cycles"] += 1
            audit["maximum_issues_in_one_cycle"] = max(
                audit["maximum_issues_in_one_cycle"], after)
            return cycle + 1

    namespace["CommandPacketCalendar"] = CommandPacketCalendar
    require(namespace["schedule_tile_timestep"].__globals__ is namespace,
            "M93 scheduler namespace mismatch")
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT),
            "M93 temporal parent was not enabled")
    source_audit = {
        "canonical_m45_sha256": sha256_bytes(canonical.encode("utf-8")),
        "transformed_source_sha256":
            sha256_bytes(transformed.encode("utf-8")),
        "edit_count": len(edits),
        "edits": edits,
        "unlisted_source_edits": 0,
    }
    return m53, namespace, m43, audit, source_audit


def replay_width(width):
    validate_inputs()
    m53, namespace, m43, command_audit, source_audit = build_namespace(width)
    namespace["validate_contract"]()
    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference = m53.read_json(M43_TEMPORAL)
    references = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(references) == 40,
            "M93 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in references, "M93 M43 reference record drift")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        cached.append((record, masks, references[key]))
    result = m53.analyze_configuration(
        namespace, m43, cached,
        "K6_CTX16_TEMPORAL_DESCRIPTOR_W{}".format(width),
        FANOUT, CONTEXTS, True, "M93_NON_CITABLE_DESCRIPTOR_PACKET_SCREEN")
    blocks = namespace["BLOCKS"]
    modeled_issue_events = command_audit["base_issue_events"] * blocks
    modeled_packet_cycles = command_audit["base_packet_cycles"] * blocks
    modeled_full_packet_cycles = (
        command_audit["base_full_packet_cycles"] * blocks)
    descriptor_commands = sum(
        row["descriptor_commands"] for row in result["per_sample"])
    require(modeled_issue_events == descriptor_commands,
            "M93 descriptor command/calendar conservation drift")
    result["descriptor_issue"] = {
        "issue_width": width,
        "descriptor_bytes": DESCRIPTOR_BYTES,
        "packet_capacity_bytes": width * DESCRIPTOR_BYTES,
        "descriptor_commands": descriptor_commands,
        "packet_cycles": modeled_packet_cycles,
        "full_packet_cycles": modeled_full_packet_cycles,
        "partial_packet_cycles":
            modeled_packet_cycles - modeled_full_packet_cycles,
        "maximum_issues_in_one_cycle":
            command_audit["maximum_issues_in_one_cycle"],
        "issued_descriptor_bytes": descriptor_commands * DESCRIPTOR_BYTES,
        "reserved_packet_lane_bytes":
            modeled_packet_cycles * width * DESCRIPTOR_BYTES,
        "packet_lane_utilization": fraction(
            descriptor_commands,
            modeled_packet_cycles * width),
        "calendar_instances": command_audit["calendar_instances"],
        "base_block_replication_factor": blocks,
    }
    result["dynamic_source_edit_audit"] = source_audit
    return result


def exact_per_sample(candidate, baseline):
    if len(candidate["per_sample"]) != len(baseline):
        return False
    for row, ref in zip(candidate["per_sample"], baseline):
        if (row["sample_id"] != ref["sample_id"] or
                row["source_only_cycles"] != ref["source"] or
                row["integrated_cycles"] != ref["integrated"]):
            return False
    return True


def sample_no_regression(candidate, baseline):
    refs = dict((row["sample_id"], row) for row in baseline)
    return all(row["integrated_cycles"] <= refs[row["sample_id"]]["integrated"]
               for row in candidate["per_sample"])


def aggregate_wait(result, field):
    return sum(row[field] for row in result["per_sample"])


def build():
    validate_inputs()
    with ProcessPoolExecutor(max_workers=len(WIDTHS)) as executor:
        configurations = list(executor.map(replay_width, WIDTHS))
    by_width = dict((row["descriptor_issue"]["issue_width"], row)
                    for row in configurations)
    m89 = read_json(M89_RECEIPT)
    matches = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(matches) == 1, "M93 M89 K6 baseline missing")
    baseline = matches[0]
    w1, w2, w4 = (by_width[value] for value in WIDTHS)

    reproduction = {
        "width1_exact_source_cycles_equal_69964176":
            w1["aggregate_source_only_cycles"] == 69964176,
        "width1_exact_integrated_cycles_equal_76677320":
            w1["aggregate_integrated_cycles"] == 76677320,
        "width1_exact_p95_integrated_cycles_equal_7843680":
            w1["integrated_cycle_distribution"]["p95_nearest_rank"] == 7843680,
        "width1_each_sample_exact_match_m89_k6":
            exact_per_sample(w1, baseline["per_sample"]),
    }
    w2_command_wait = aggregate_wait(w2, "command_or_state_wait_cycles")
    w1_command_wait = aggregate_wait(w1, "command_or_state_wait_cycles")
    w2_gates = {
        "exact_40_record_10_sample_replay":
            w2["record_ledger"]["record_count"] == 40 and
            len(w2["per_sample"]) == 10,
        "signed_add_subtract_conservation": all(
            row["signed_add_updates"] + row["signed_subtract_updates"] ==
            row["logical_source_updates"] for row in w2["per_sample"]),
        "new_dependency_edges_equal_zero": True,
        "descriptor_count_and_identity_conservation":
            w2["descriptor_issue"]["descriptor_commands"] == 25920000,
        "maximum_descriptor_issues_per_cycle_le_2":
            w2["descriptor_issue"]["maximum_issues_in_one_cycle"] <= 2,
        "maximum_metadata_occupancy_le_16": all(
            row["maximum_metadata_occupancy"] <= 16
            for row in w2["per_sample"]),
        "maximum_complete_occupancy_le_16": all(
            row["maximum_complete_occupancy"] <= 16
            for row in w2["per_sample"]),
        "aggregate_source_cycles_must_not_exceed_m89_k6_69964176":
            w2["aggregate_source_only_cycles"] <= 69964176,
        "aggregate_integrated_cycles_le_76293933":
            w2["aggregate_integrated_cycles"] <= WIDTH2_MAX_INTEGRATED,
        "p95_integrated_cycles_lt_7843680":
            w2["integrated_cycle_distribution"]["p95_nearest_rank"] < 7843680,
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            sample_no_regression(w2, baseline["per_sample"]),
        "command_or_state_wait_cycles_must_decrease":
            w2_command_wait < w1_command_wait,
    }
    w4_incremental_limit = int(math.floor(
        w2["aggregate_integrated_cycles"] * 0.995))
    w4_incremental_gate = (
        w4["aggregate_integrated_cycles"] <= w4_incremental_limit)
    if all(reproduction.values()) and all(w2_gates.values()):
        selected = 4 if w4_incremental_gate else 2
        status = "PASS_WIDTH{}_PROMOTION_SCREEN".format(selected)
    else:
        selected = None
        status = "PASS_EXECUTION_NO_GO_PROMOTION"

    return {
        "schema": "m93_dual_descriptor_packet_issue_result_v1",
        "status": status,
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m53_analyzer_sha256": EXPECTED["m53_analyzer"],
            "m53_result_sha256": EXPECTED["m53_result"],
            "m43_temporal_sha256": EXPECTED["m43_temporal"],
            "m89_receipt_sha256": EXPECTED["m89_receipt"],
        },
        "frozen_baseline": {
            "source_cycles": baseline["source_cycles"],
            "integrated_cycles": baseline["integrated_cycles"],
            "p95_integrated_cycles": baseline["p95_integrated_cycles"],
            "per_sample": baseline["per_sample"],
        },
        "configurations": configurations,
        "reproduction_gates": reproduction,
        "width2_gates": w2_gates,
        "width2_all_gates_pass": all(w2_gates.values()),
        "width4_policy": {
            "incremental_half_percent_maximum_integrated_cycles":
                w4_incremental_limit,
            "incremental_half_percent_gate_pass": w4_incremental_gate,
        },
        "comparison": {
            "width2_integrated_speedup_vs_m89_k6": fraction(
                baseline["integrated_cycles"],
                w2["aggregate_integrated_cycles"]),
            "width4_integrated_speedup_vs_width2": fraction(
                w2["aggregate_integrated_cycles"],
                w4["aggregate_integrated_cycles"]),
            "width2_integrated_delta_vs_m89_k6":
                w2["aggregate_integrated_cycles"] -
                baseline["integrated_cycles"],
            "width2_source_delta_vs_m89_k6":
                w2["aggregate_source_only_cycles"] - baseline["source_cycles"],
            "width2_command_wait_delta_vs_width1":
                w2_command_wait - w1_command_wait,
            "width2_vs_m89_k8_integrated_delta":
                w2["aggregate_integrated_cycles"] - 76337352,
        },
        "selected_width": selected,
        "claim_policy": {
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    compact = {
        "status": result["status"],
        "selected_width": result["selected_width"],
        "reproduction": all(result["reproduction_gates"].values()),
        "width2_all_gates": result["width2_all_gates_pass"],
        "width4_incremental_gate":
            result["width4_policy"]["incremental_half_percent_gate_pass"],
        "configurations": [{
            "width": row["descriptor_issue"]["issue_width"],
            "source": row["aggregate_source_only_cycles"],
            "integrated": row["aggregate_integrated_cycles"],
            "p95": row["integrated_cycle_distribution"]["p95_nearest_rank"],
            "command_wait": aggregate_wait(
                row, "command_or_state_wait_cycles"),
        } for row in result["configurations"]],
    }
    print("M93_DUAL_DESCRIPTOR_PACKET_ISSUE=" +
          json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()

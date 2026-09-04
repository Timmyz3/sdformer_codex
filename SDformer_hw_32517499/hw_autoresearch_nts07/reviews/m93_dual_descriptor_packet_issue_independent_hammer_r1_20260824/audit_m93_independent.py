#!/usr/bin/env python3
"""Independent M93 dual-descriptor issue screening audit.

This audit never imports or executes the M93 producer.  It verifies every
sealed input hash, rebuilds all sample/aggregate ledgers from the raw 40-record
results, checks the logs and receipt, and statically audits the sole dynamic
scheduler-source edit plus the wide-calendar implementation.
"""

from __future__ import print_function

from fractions import Fraction
import difflib
import hashlib
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULT_DIR = HW / "results/m93_dual_descriptor_packet_issue_probe_r1_20260824"
CONTRACT = HW / "contracts/m93_dual_descriptor_packet_issue_contract_r1_20260824.json"
PROBE = HW / "system_simulator/scripts/probe_m93_dual_descriptor_packet_issue.py"
RAW = RESULT_DIR / "remote_artifacts/m93_dual_descriptor_packet_issue_probe_r1_20260824.json"
FAIL_LOG = RESULT_DIR / "remote_artifacts/m93_dual_descriptor_packet_issue_probe_r1_20260824.log"
COMPLETE_LOG = RESULT_DIR / "remote_artifacts/m93_dual_descriptor_packet_issue_probe_r2_20260824.log"
RECEIPT = RESULT_DIR / "m93_dual_descriptor_packet_issue_probe_receipt_r1.json"
M45 = HW / "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py"
M53_ANALYZER = HW / "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py"
M53_RESULT = HW / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43_TEMPORAL = HW / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
M89_RECEIPT = HW / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")
OUTPUT = HERE / "m93_independent_audit.json"

EXPECTED = {
    "contract": "28ffb056e87a715eee0d4cf04aabe5ed9901d17bfc6fc76c9692d1a03a87a4e8",
    "probe": "832042a48996c56b3709fe1104990b2e6f8895d48eb593f27be04c64ac5ed883",
    "raw": "7345e006f052bf00520800f9fbf8d2792747a2686cdb0e15c61d9703f1cac7e9",
    "failure_log_r1": "846a83037fdfcfc27af873174517bc11446b859eefb3db3a65dfd38a79ba9641",
    "complete_log_r2": "e1ec609cc54e115d945c1bd2d0454dabd34c21e9292e3a0176bb08fd611c5e83",
    "receipt": "b07fa6872a1eebe5f98db07c5e6502902030a32d33a480cc898cddbcecd536a9",
    "m45_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}

WIDTHS = (1, 2, 4)
SUM_FIELDS = (
    "source_only_cycles", "integrated_cycles", "logical_source_updates",
    "unique_weight_issues", "descriptor_commands", "parent_partial_reads",
    "parent_partial_writes", "final_accumulator_reads",
    "final_accumulator_writes", "completed_outputs", "fusion_groups",
    "zero_source_groups", "parent_wait_cycles",
    "command_or_state_wait_cycles", "response_or_context_wait_cycles",
    "weight_dma_wait_cycles", "fusion_hold_wait_cycles", "late_join_groups",
    "signed_add_updates", "signed_subtract_updates", "weight_dma_bytes",
    "final_accumulator_read_bytes", "final_accumulator_write_bytes",
    "completed_output_bytes",
)
MAX_FIELDS = (
    "maximum_metadata_occupancy", "maximum_complete_occupancy",
    "maximum_resident_occupancy",
)
PROGRESS = re.compile(
    r"^\[M53 K6_CTX16_TEMPORAL_DESCRIPTOR_W([124])\] "
    r"([0-9]+)/40 sample=([0-9]+) operator=(\S+)$")


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


def strict_json_text(text, label):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result,
                    "duplicate JSON key {} in {}".format(key, label))
            result[key] = value
        return result

    return json.loads(
        text, object_pairs_hook=pairs_hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant {} in {}".format(raw, label))))


def read_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"), str(path))


def compare(left, right, label="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right),
                "{} key drift: {} != {}".format(
                    label, sorted(left), sorted(right)))
        for key in left:
            compare(left[key], right[key], label + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), label + " length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + "[{}]".format(index))
    elif isinstance(left, float) or isinstance(right, float):
        require(abs(float(left) - float(right)) <=
                1e-12 * max(1.0, abs(float(right))), label + " float drift")
    else:
        require(left == right,
                "{} drift: {} != {}".format(label, left, right))


def raw_fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    return {"numerator": int(numerator), "denominator": int(denominator)}


def reduced_fraction(numerator, denominator):
    require(denominator > 0, "zero denominator")
    value = Fraction(int(numerator), int(denominator))
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def distribution(values):
    ordered = sorted(values)
    require(ordered, "empty distribution")

    def nearest_rank(percent):
        rank = (percent * len(ordered) + 99) // 100
        return ordered[rank - 1]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean_exact": raw_fraction(sum(ordered), len(ordered)),
        "p50_nearest_rank": nearest_rank(50),
        "p95_nearest_rank": nearest_rank(95),
        "p99_nearest_rank": nearest_rank(99),
    }


def validate_identity():
    paths = {
        "contract": CONTRACT,
        "probe": PROBE,
        "raw": RAW,
        "failure_log_r1": FAIL_LOG,
        "complete_log_r2": COMPLETE_LOG,
        "receipt": RECEIPT,
        "m45_analyzer": M45,
        "m53_analyzer": M53_ANALYZER,
        "m53_result": M53_RESULT,
        "m43_temporal": M43_TEMPORAL,
        "m89_receipt": M89_RECEIPT,
    }
    identity = {}
    for name, path in paths.items():
        actual = sha256(path)
        require(actual == EXPECTED[name], name + " SHA drift")
        identity[name] = {
            "path": str(path), "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return identity


def audit_logs(raw):
    failure = FAIL_LOG.read_text(encoding="utf-8")
    require("FileNotFoundError" in failure and
            "m53_adaptive_temporal_parent_k4_ctx16_dse.json" in failure and
            "M93_DUAL_DESCRIPTOR_PACKET_ISSUE=" not in failure,
            "r1 failure log did not fail closed on missing M53 result")

    complete = COMPLETE_LOG.read_text(encoding="utf-8")
    progress = dict((width, []) for width in WIDTHS)
    for line in complete.splitlines():
        match = PROGRESS.match(line)
        if match:
            width = int(match.group(1))
            progress[width].append((int(match.group(2)),
                                    int(match.group(3)), match.group(4)))
    progress_summary = {}
    for width in WIDTHS:
        rows = progress[width]
        require([row[0] for row in rows] == list(range(1, 41)),
                "W{} progress is not exactly 1..40".format(width))
        identities = [(row[1], row[2]) for row in rows]
        require(len(set(identities)) == 40 and
                set(row[0] for row in identities) == set(range(10)) and
                all(sum(1 for sample, _ in identities if sample == value) == 4
                    for value in range(10)),
                "W{} progress identity population drift".format(width))
        progress_summary[str(width)] = {
            "records": 40, "samples": 10,
            "unique_sample_operator_pairs": 40,
        }
    markers = [line for line in complete.splitlines()
               if line.startswith("M93_DUAL_DESCRIPTOR_PACKET_ISSUE=")]
    require(len(markers) == 1, "r2 final marker population drift")
    marker = strict_json_text(markers[0].split("=", 1)[1], "r2 marker")
    compact = {
        "status": raw["status"],
        "selected_width": raw["selected_width"],
        "reproduction": all(raw["reproduction_gates"].values()),
        "width2_all_gates": raw["width2_all_gates_pass"],
        "width4_incremental_gate":
            raw["width4_policy"]["incremental_half_percent_gate_pass"],
        "configurations": [],
    }
    for row in raw["configurations"]:
        compact["configurations"].append({
            "width": row["descriptor_issue"]["issue_width"],
            "source": row["aggregate_source_only_cycles"],
            "integrated": row["aggregate_integrated_cycles"],
            "p95": row["integrated_cycle_distribution"]["p95_nearest_rank"],
            "command_wait": sum(
                sample["command_or_state_wait_cycles"]
                for sample in row["per_sample"]),
        })
    compare(compact, marker, "r2 compact marker")
    return {
        "r1": "FAIL_CLOSED_MISSING_M53_RESULT",
        "r2": "COMPLETE_EXACT_COMPACT_MATCH",
        "r2_progress": progress_summary,
    }


def static_source_audit(raw):
    canonical = M45.read_text(encoding="utf-8")
    m53 = canonical
    replacements = (
        (
            "require(1 <= fanout_k <= context_capacity <= 8,\n"
            "            \"invalid fanout/context geometry\")",
            "require(1 <= fanout_k <= context_capacity <= 16,\n"
            "            \"invalid fanout/context geometry\")",
        ),
        (
            "require(name in (\"local_zero\", \"left\", \"up\"),\n"
            "                \"temporal parent leaked into M45 primary\")",
            "require(name in (\"local_zero\", \"left\", \"up\",\n"
            "                         \"previous_timestep\"),\n"
            "                \"invalid M53 adaptive parent\")",
        ),
        ("module.ALLOW_TEMPORAL_PARENT = False",
         "module.ALLOW_TEMPORAL_PARENT = True"),
    )
    for old, new in replacements:
        require(m53.count(old) == 1, "M53 static replacement identity drift")
        m53 = m53.replace(old, new)
    command_from = "    command_port = PortCalendar()"
    command_to = "    command_port = CommandPacketCalendar()"
    require(m53.count(command_from) == 1 and m53.count(command_to) == 0,
            "M93 command-port source identity drift")
    transformed = m53.replace(command_from, command_to)
    source_diff = list(difflib.unified_diff(
        m53.splitlines(), transformed.splitlines(), lineterm=""))
    changed = [line for line in source_diff
               if (line.startswith("+") or line.startswith("-")) and
               not line.startswith("+++") and not line.startswith("---")]
    compare(changed, ["-" + command_from, "+" + command_to],
            "M93 sole transformed-scheduler edit")
    transformed_sha = sha256_bytes(transformed.encode("utf-8"))
    require(transformed_sha ==
            "7e697612f92a167aff0b8d908bf907c9ce05d1f6da6cb9c52e691a203034d842",
            "M93 reconstructed transformed-source SHA drift")
    for row in raw["configurations"]:
        audit = row["dynamic_source_edit_audit"]
        require(audit["canonical_m45_sha256"] == EXPECTED["m45_analyzer"] and
                audit["transformed_source_sha256"] == transformed_sha and
                audit["edit_count"] == 4 and
                audit["unlisted_source_edits"] == 0,
                "M93 raw dynamic-source audit drift")

    probe = PROBE.read_text(encoding="utf-8")
    required_calendar_snippets = (
        "self.occupied = {}",
        "cycle = int(ready_cycle)",
        "while self.occupied.get(cycle, 0) >= width:",
        "cycle += 1",
        "before = self.occupied.get(cycle, 0)",
        "after = before + 1",
        "self.occupied[cycle] = after",
        "return cycle + 1",
    )
    require(all(probe.count(snippet) == 1
                for snippet in required_calendar_snippets),
            "M93 wide-calendar static implementation drift")
    require(probe.count("command_audit[\"base_issue_events\"] * blocks") == 1 and
            probe.count("command_audit[\"base_packet_cycles\"] * blocks") == 1 and
            probe.count("command_audit[\"base_full_packet_cycles\"] * blocks") == 1,
            "M93 eight-block scaling source drift")
    require(canonical.count("record_counts[\"integrated_cycles\"] = block_counts[\"integrated_cycles\"] * BLOCKS") == 1 and
            canonical.count("add_counts(record_counts, block_counts, BLOCKS)") == 1,
            "M45 eight-block source scaling drift")
    require(canonical.count(
        "counts[\"maximum_metadata_occupancy\"], min(16, len(ready)))") == 1,
        "M45 metadata-occupancy clamp source drift")
    return {
        "canonical_m45_sha256": EXPECTED["m45_analyzer"],
        "m53_transformed_then_m93_sha256": transformed_sha,
        "m93_scheduler_source_diff": changed,
        "only_command_calendar_instantiation_changed_after_m53": True,
        "calendar_capacity_rule": "while occupancy[cycle] >= width, advance; then increment exactly one",
        "calendar_never_schedules_before_ready_since": True,
        "calendar_can_schedule_before_current_scheduler_now": True,
        "calendar_completion": "issue_cycle + 1",
        "all_widths_share_same_transformed_scheduler_sha": True,
        "width_semantics_live_in_unhashed_closure_class_relative_to_transformed_source_sha": True,
        "metadata_fifo_gate_is_vacuous_for_ready_heap_overflow": True,
        "metadata_fifo_gate_reason": (
            "reported occupancy is min(16,len(ready)) and is then checked <=16; "
            "the underlying ready heap is not bounded by that assertion"),
        "eight_block_scaling": {
            "rows_per_tile_timestep": 300,
            "timesteps": 10,
            "tiles": 27,
            "records": 40,
            "calendar_instances_per_width": 40 * 10 * 27,
            "base_descriptors_per_width": 40 * 10 * 27 * 300,
            "blocks": 8,
            "scaled_descriptors_per_width": 40 * 10 * 27 * 300 * 8,
        },
        "physical_packet_equivalence": False,
        "missing_physical_packet_semantics": [
            "descriptor payload bytes and addresses",
            "128/256-byte alignment and bus transfer",
            "per-lane valid and partial-packet encoding",
            "packet assembly queue and current-cycle admission constraint",
            "unpacker/decode/dispatch timing and backpressure",
            "wide wiring, arbitration, timing, area and energy",
        ],
    }


def rebuild_sample(records, sample_id):
    selected = [row for row in records if row["sample_id"] == sample_id]
    require(len(selected) == 4, "sample {} operator count drift".format(sample_id))
    require(len(set(row["operator"] for row in selected)) == 4,
            "sample {} operator identity drift".format(sample_id))
    result = {"sample_id": sample_id}
    for field in SUM_FIELDS:
        result[field] = sum(row[field] for row in selected)
    for field in MAX_FIELDS:
        result[field] = max(row[field] for row in selected)
    parent_names = sorted(selected[0]["parent_selection"][
        "parent_choice_by_tile"])
    result["parent_choice_by_tile"] = dict(
        (parent, sum(row["parent_selection"]["parent_choice_by_tile"][parent]
                     for row in selected)) for parent in parent_names)
    result["unfused_parent_delta_source_issue_cycles"] = sum(
        row["parent_selection"]["unfused_parent_delta_source_issue_cycles"]
        for row in selected)
    result["previous_timestep_choices_after_timestep_zero"] = sum(
        row["parent_selection"][
            "previous_timestep_choices_after_timestep_zero"]
        for row in selected)
    require(sum(row["parent_selection"][
        "previous_timestep_choices_at_timestep_zero"] for row in selected) == 0,
        "previous-timestep parent leaked into timestep zero")
    result["integrated_over_source_only"] = reduced_fraction(
        result["integrated_cycles"] - result["source_only_cycles"],
        result["source_only_cycles"])
    result["parent_wait_fraction"] = reduced_fraction(
        result["parent_wait_cycles"], result["integrated_cycles"])
    return result


def audit_configuration(row):
    width = row["descriptor_issue"]["issue_width"]
    records = row["record_ledger"]["records"]
    require(row["record_ledger"]["record_count"] == len(records) == 40,
            "W{} record count drift".format(width))
    identities = [(record["sample_id"], record["operator"])
                  for record in records]
    require(len(set(identities)) == 40 and
            set(sample for sample, _ in identities) == set(range(10)),
            "W{} record identity drift".format(width))
    canonical = (json.dumps(records, sort_keys=True, separators=(",", ":")) +
                 "\n").encode("utf-8")
    compare(sha256_bytes(canonical), row["record_ledger"]["canonical_sha256"],
            "W{} canonical record SHA".format(width))
    rebuilt_samples = [rebuild_sample(records, sample_id)
                       for sample_id in range(10)]
    compare(rebuilt_samples, row["per_sample"],
            "W{} per-sample reconstruction".format(width))
    source_values = [sample["source_only_cycles"]
                     for sample in rebuilt_samples]
    integrated_values = [sample["integrated_cycles"]
                         for sample in rebuilt_samples]
    compare(distribution(source_values), row["source_only_cycle_distribution"],
            "W{} source distribution".format(width))
    compare(distribution(integrated_values),
            row["integrated_cycle_distribution"],
            "W{} integrated distribution".format(width))
    source = sum(source_values)
    integrated = sum(integrated_values)
    compare(source, row["aggregate_source_only_cycles"],
            "W{} source aggregate".format(width))
    compare(integrated, row["aggregate_integrated_cycles"],
            "W{} integrated aggregate".format(width))
    compare(raw_fraction(integrated - source, source),
            row["aggregate_integrated_over_source_only"],
            "W{} aggregate overhead fraction".format(width))
    compare(raw_fraction(sum(sample["parent_wait_cycles"]
                             for sample in rebuilt_samples), integrated),
            row["aggregate_parent_wait_fraction"],
            "W{} parent wait fraction".format(width))

    issue = row["descriptor_issue"]
    descriptors = sum(sample["descriptor_commands"]
                      for sample in rebuilt_samples)
    require(descriptors == issue["descriptor_commands"] == 25920000,
            "W{} descriptor count drift".format(width))
    require(issue["packet_cycles"] ==
            issue["full_packet_cycles"] + issue["partial_packet_cycles"],
            "W{} packet-cycle partition drift".format(width))
    require(issue["maximum_issues_in_one_cycle"] <= width,
            "W{} issue capacity overflow".format(width))
    require(issue["issued_descriptor_bytes"] == descriptors * 64 and
            issue["reserved_packet_lane_bytes"] ==
            issue["packet_cycles"] * width * 64,
            "W{} packet byte arithmetic drift".format(width))
    compare(reduced_fraction(descriptors, issue["packet_cycles"] * width),
            issue["packet_lane_utilization"],
            "W{} lane utilization".format(width))
    require(issue["base_block_replication_factor"] == 8 and
            issue["calendar_instances"] == 10800 and
            descriptors % 8 == 0 and issue["packet_cycles"] % 8 == 0 and
            issue["full_packet_cycles"] % 8 == 0,
            "W{} block scaling drift".format(width))
    for sample in rebuilt_samples:
        require(sample["signed_add_updates"] +
                sample["signed_subtract_updates"] ==
                sample["logical_source_updates"],
                "W{} signed arithmetic conservation drift".format(width))
    return {
        "width": width,
        "records": 40,
        "samples": 10,
        "record_ledger_sha256": row["record_ledger"]["canonical_sha256"],
        "source_cycles": source,
        "integrated_cycles": integrated,
        "p95_integrated_cycles":
            row["integrated_cycle_distribution"]["p95_nearest_rank"],
        "waits": dict((field, sum(sample[field]
                                   for sample in rebuilt_samples))
                      for field in (
                          "command_or_state_wait_cycles",
                          "response_or_context_wait_cycles",
                          "parent_wait_cycles", "weight_dma_wait_cycles",
                          "fusion_hold_wait_cycles")),
        "packet": {
            "packet_capacity_bytes": issue["packet_capacity_bytes"],
            "descriptor_commands": descriptors,
            "packet_cycles": issue["packet_cycles"],
            "full_packet_cycles": issue["full_packet_cycles"],
            "partial_packet_cycles": issue["partial_packet_cycles"],
            "maximum_issues_in_one_cycle":
                issue["maximum_issues_in_one_cycle"],
            "packet_lane_utilization": issue["packet_lane_utilization"],
            "issued_descriptor_bytes": issue["issued_descriptor_bytes"],
            "reserved_packet_lane_bytes":
                issue["reserved_packet_lane_bytes"],
        },
        "per_sample": [{
            "sample_id": sample["sample_id"],
            "source_cycles": sample["source_only_cycles"],
            "integrated_cycles": sample["integrated_cycles"],
            "command_wait": sample["command_or_state_wait_cycles"],
            "response_wait": sample["response_or_context_wait_cycles"],
            "parent_wait": sample["parent_wait_cycles"],
        } for sample in rebuilt_samples],
        "logical_source_updates": sum(
            sample["logical_source_updates"] for sample in rebuilt_samples),
        "signed_add_updates": sum(
            sample["signed_add_updates"] for sample in rebuilt_samples),
        "signed_subtract_updates": sum(
            sample["signed_subtract_updates"] for sample in rebuilt_samples),
        "unique_weight_issues": sum(
            sample["unique_weight_issues"] for sample in rebuilt_samples),
        "fusion_groups": sum(
            sample["fusion_groups"] for sample in rebuilt_samples),
        "zero_source_groups": sum(
            sample["zero_source_groups"] for sample in rebuilt_samples),
        "maximum_metadata_occupancy": max(
            sample["maximum_metadata_occupancy"] for sample in rebuilt_samples),
        "maximum_complete_occupancy": max(
            sample["maximum_complete_occupancy"] for sample in rebuilt_samples),
    }


def delta(after, before):
    return after - before


def main():
    identity = validate_identity()
    contract = read_json(CONTRACT)
    raw = read_json(RAW)
    receipt = read_json(RECEIPT)
    m89 = read_json(M89_RECEIPT)
    require(raw["status"] == receipt["status"] ==
            "PASS_EXECUTION_NO_GO_PROMOTION" and
            raw["selected_width"] is None and
            receipt["decision"]["promotion"] == "NO_GO",
            "M93 fail-closed status drift")
    log_audit = audit_logs(raw)
    static_audit = static_source_audit(raw)
    configs = dict((row["descriptor_issue"]["issue_width"],
                    audit_configuration(row))
                   for row in raw["configurations"])
    require(set(configs) == set(WIDTHS), "M93 width population drift")

    # Every width must replay the same canonical record identities in the same order.
    raw_by_width = dict((row["descriptor_issue"]["issue_width"], row)
                        for row in raw["configurations"])
    identities_by_width = {}
    for width, row in raw_by_width.items():
        identities_by_width[width] = [
            (record["sample_id"], record["operator"])
            for record in row["record_ledger"]["records"]]
    compare(identities_by_width[1], identities_by_width[2],
            "W1/W2 record identity/order")
    compare(identities_by_width[1], identities_by_width[4],
            "W1/W4 record identity/order")

    k6_matches = [row for row in m89["configurations"]
                  if row["name"] == "K6"]
    require(len(k6_matches) == 1, "M89 K6 baseline population drift")
    baseline = k6_matches[0]
    w1 = raw_by_width[1]
    require(len(baseline["per_sample"]) == len(w1["per_sample"]) == 10,
            "W1 baseline sample population drift")
    w1_sample_exact = []
    for observed, expected in zip(w1["per_sample"], baseline["per_sample"]):
        exact = (observed["sample_id"] == expected["sample_id"] and
                 observed["source_only_cycles"] == expected["source"] and
                 observed["integrated_cycles"] == expected["integrated"])
        require(exact, "W1 M89 K6 per-sample mismatch")
        w1_sample_exact.append({
            "sample_id": observed["sample_id"],
            "source_cycles": observed["source_only_cycles"],
            "integrated_cycles": observed["integrated_cycles"],
            "exact_m89_k6_match": True,
        })

    c1, c2, c4 = (configs[width] for width in WIDTHS)
    w2_deltas = []
    w4_deltas = []
    for before, after in zip(c1["per_sample"], c2["per_sample"]):
        w2_deltas.append({
            "sample_id": before["sample_id"],
            "source_cycles": delta(after["source_cycles"],
                                   before["source_cycles"]),
            "integrated_cycles": delta(after["integrated_cycles"],
                                       before["integrated_cycles"]),
        })
    for before, after in zip(c2["per_sample"], c4["per_sample"]):
        w4_deltas.append({
            "sample_id": before["sample_id"],
            "source_cycles": delta(after["source_cycles"],
                                   before["source_cycles"]),
            "integrated_cycles": delta(after["integrated_cycles"],
                                       before["integrated_cycles"]),
        })
    compare(w2_deltas, receipt["width2_per_sample_deltas_candidate_minus_m89_k6"],
            "receipt W2 per-sample deltas")

    integrated_limit = int(math.floor(c1["integrated_cycles"] * 0.995))
    w4_limit = int(math.floor(c2["integrated_cycles"] * 0.995))
    reproduction_gates = {
        "width1_exact_source_cycles_equal_69964176":
            c1["source_cycles"] == 69964176,
        "width1_exact_integrated_cycles_equal_76677320":
            c1["integrated_cycles"] == 76677320,
        "width1_exact_p95_integrated_cycles_equal_7843680":
            c1["p95_integrated_cycles"] == 7843680,
        "width1_each_sample_exact_match_m89_k6":
            all(row["exact_m89_k6_match"] for row in w1_sample_exact),
    }
    w2_gates = {
        "exact_40_record_10_sample_replay":
            c2["records"] == 40 and c2["samples"] == 10,
        "signed_add_subtract_conservation":
            c2["signed_add_updates"] + c2["signed_subtract_updates"] ==
            c2["logical_source_updates"],
        "new_dependency_edges_equal_zero":
            static_audit["only_command_calendar_instantiation_changed_after_m53"],
        "descriptor_count_and_identity_conservation":
            c2["packet"]["descriptor_commands"] == 25920000 and
            identities_by_width[2] == identities_by_width[1],
        "maximum_descriptor_issues_per_cycle_le_2":
            c2["packet"]["maximum_issues_in_one_cycle"] <= 2,
        "maximum_metadata_occupancy_le_16":
            c2["maximum_metadata_occupancy"] <= 16,
        "maximum_complete_occupancy_le_16":
            c2["maximum_complete_occupancy"] <= 16,
        "aggregate_source_cycles_must_not_exceed_m89_k6_69964176":
            c2["source_cycles"] <= 69964176,
        "aggregate_integrated_cycles_le_76293933":
            c2["integrated_cycles"] <= integrated_limit,
        "p95_integrated_cycles_lt_7843680":
            c2["p95_integrated_cycles"] < 7843680,
        "each_sample_integrated_cycles_must_not_regress_vs_m89_k6":
            all(row["integrated_cycles"] <= 0 for row in w2_deltas),
        "command_or_state_wait_cycles_must_decrease":
            c2["waits"]["command_or_state_wait_cycles"] <
            c1["waits"]["command_or_state_wait_cycles"],
    }
    compare(reproduction_gates, raw["reproduction_gates"],
            "raw reproduction gates")
    compare(reproduction_gates, receipt["reproduction_gates"],
            "receipt reproduction gates")
    compare(w2_gates, raw["width2_gates"], "raw W2 gates")
    receipt_w2_gates = dict(receipt["width2_gates"])
    require(receipt_w2_gates.pop("all_width2_gates_pass") is False,
            "receipt W2 all-gate drift")
    compare(w2_gates, receipt_w2_gates, "receipt W2 gates")
    require(not all(w2_gates.values()) and
            raw["width2_all_gates_pass"] is False,
            "W2 must remain NO-GO")

    command_delta = delta(
        c2["waits"]["command_or_state_wait_cycles"],
        c1["waits"]["command_or_state_wait_cycles"])
    response_delta = delta(
        c2["waits"]["response_or_context_wait_cycles"],
        c1["waits"]["response_or_context_wait_cycles"])
    parent_delta = delta(
        c2["waits"]["parent_wait_cycles"],
        c1["waits"]["parent_wait_cycles"])
    source_delta = delta(c2["source_cycles"], c1["source_cycles"])
    integrated_delta = delta(c2["integrated_cycles"], c1["integrated_cycles"])
    overhead1 = c1["integrated_cycles"] - c1["source_cycles"]
    overhead2 = c2["integrated_cycles"] - c2["source_cycles"]
    overhead_delta = overhead2 - overhead1
    known_wait_delta = command_delta + response_delta + parent_delta
    residual_delta = overhead_delta - known_wait_delta
    require((command_delta, response_delta, parent_delta, source_delta,
             integrated_delta, overhead_delta, residual_delta) ==
            (-62880, 116984, 27160, -109080, -27624, 81456, 192),
            "W2 decomposition drift")
    require(source_delta + overhead_delta == integrated_delta,
            "W2 source/overhead decomposition does not close")
    compare(raw["comparison"]["width2_command_wait_delta_vs_width1"],
            command_delta, "raw command delta")
    compare(raw["comparison"]["width2_integrated_delta_vs_m89_k6"],
            integrated_delta, "raw integrated delta")
    compare(raw["comparison"]["width2_source_delta_vs_m89_k6"],
            source_delta, "raw source delta")

    config_receipt = dict((row["width"], row)
                          for row in receipt["configurations"])
    for width, config in configs.items():
        stored = config_receipt[width]
        expected_summary = {
            "width": width,
            "packet_capacity_bytes": config["packet"]["packet_capacity_bytes"],
            "source_cycles": config["source_cycles"],
            "integrated_cycles": config["integrated_cycles"],
            "p95_integrated_cycles": config["p95_integrated_cycles"],
            "non_source_overhead_cycles":
                config["integrated_cycles"] - config["source_cycles"],
            "command_or_state_wait_cycles":
                config["waits"]["command_or_state_wait_cycles"],
            "response_or_context_wait_cycles":
                config["waits"]["response_or_context_wait_cycles"],
            "parent_wait_cycles": config["waits"]["parent_wait_cycles"],
            "packet_lane_utilization":
                config["packet"]["packet_lane_utilization"]["decimal"],
        }
        if width != 1:
            expected_summary["full_packet_cycles"] = (
                config["packet"]["full_packet_cycles"])
            expected_summary["partial_packet_cycles"] = (
                config["packet"]["partial_packet_cycles"])
        compare(expected_summary, stored,
                "receipt W{} summary".format(width))

    w2_receipt = receipt["width2_comparison_vs_m89_k6"]
    compare(w2_receipt["source_cycle_delta"], source_delta,
            "receipt W2 source delta")
    compare(w2_receipt["integrated_cycle_delta"], integrated_delta,
            "receipt W2 integrated delta")
    compare(w2_receipt["command_wait_delta"], command_delta,
            "receipt W2 command delta")
    compare(w2_receipt["response_wait_delta"], response_delta,
            "receipt W2 response delta")
    compare(w2_receipt["parent_wait_delta"], parent_delta,
            "receipt W2 parent delta")
    compare(w2_receipt["non_source_overhead_delta"], overhead_delta,
            "receipt W2 overhead delta")
    compare(w2_receipt["cycles_above_half_percent_promotion_limit"],
            c2["integrated_cycles"] - integrated_limit,
            "receipt W2 promotion miss")
    compare(w2_receipt["integrated_delta_vs_m89_k8"],
            c2["integrated_cycles"] - 76337352,
            "receipt W2 vs K8 delta")
    compare(w2_receipt["integrated_speedup"],
            c1["integrated_cycles"] / float(c2["integrated_cycles"]),
            "receipt W2 speedup")
    compare(w2_receipt["integrated_cycle_reduction_fraction"],
            -integrated_delta / float(c1["integrated_cycles"]),
            "receipt W2 reduction fraction")

    w4_delta = c4["integrated_cycles"] - c2["integrated_cycles"]
    require(w4_delta == 36744 and c4["integrated_cycles"] >
            c2["integrated_cycles"], "W4 must be slower than W2")
    compare(receipt["width4_policy"], {
        "incremental_half_percent_maximum_integrated_cycles": w4_limit,
        "incremental_half_percent_gate_pass": False,
        "integrated_cycle_delta_width4_minus_width2": w4_delta,
        "decision": "KILL_WIDTH4",
    }, "receipt W4 policy")

    result = {
        "schema": "m93_dual_descriptor_packet_issue_independent_audit_v1",
        "status": "PASS_INDEPENDENT_ARITHMETIC_STATIC_AUDIT_NO_GO_CONFIRMED",
        "identity": identity,
        "log_audit": log_audit,
        "independence": {
            "producer_imported_or_executed": False,
            "raw_records_directly_reaggregated": True,
            "receipt_arithmetic_directly_recomputed": True,
            "producer_files_modified": False,
            "command_calendar_event_trace_available": False,
        },
        "static_source_audit": static_audit,
        "configurations": [configs[width] for width in WIDTHS],
        "w1_exact_m89_k6_per_sample": w1_sample_exact,
        "w2_per_sample_delta_candidate_minus_w1": w2_deltas,
        "w4_per_sample_delta_candidate_minus_w2": w4_deltas,
        "reproduction_gates": reproduction_gates,
        "width2_gates": w2_gates,
        "width2_all_gates_pass": False,
        "width4_incremental_half_percent_gate_pass": False,
        "cycle_decomposition": {
            "w2_minus_w1": {
                "source_cycles": source_delta,
                "command_wait_cycles": command_delta,
                "response_wait_cycles": response_delta,
                "parent_wait_cycles": parent_delta,
                "known_wait_delta_sum": known_wait_delta,
                "residual_non_source_overhead_delta": residual_delta,
                "total_non_source_overhead_delta": overhead_delta,
                "integrated_cycles": integrated_delta,
                "exact_equation":
                    "-109080 source + 81456 overhead = -27624 integrated",
            },
            "w4_minus_w2": {
                "source_cycles": c4["source_cycles"] - c2["source_cycles"],
                "command_wait_cycles":
                    c4["waits"]["command_or_state_wait_cycles"] -
                    c2["waits"]["command_or_state_wait_cycles"],
                "response_wait_cycles":
                    c4["waits"]["response_or_context_wait_cycles"] -
                    c2["waits"]["response_or_context_wait_cycles"],
                "parent_wait_cycles":
                    c4["waits"]["parent_wait_cycles"] -
                    c2["waits"]["parent_wait_cycles"],
                "integrated_cycles": w4_delta,
            },
            "interpretation": (
                "Wider admission changes which K6-ready tasks coexist and are "
                "fused. W2 saves source work and command wait, but earlier/burstier "
                "residency fills context/complete resources and shifts the bottleneck "
                "to response and parent waits. W4 intensifies this coupling."),
        },
        "promotion": {
            "decision": "NO_GO",
            "width2_integrated_reduction_fraction":
                -integrated_delta / float(c1["integrated_cycles"]),
            "width2_integrated_speedup":
                c1["integrated_cycles"] / float(c2["integrated_cycles"]),
            "width2_cycles_above_frozen_half_percent_limit":
                c2["integrated_cycles"] - integrated_limit,
            "width2_regressed_sample_ids": [
                row["sample_id"] for row in w2_deltas
                if row["integrated_cycles"] > 0],
            "width4_minus_width2_integrated_cycles": w4_delta,
            "must_not_relax_half_percent_gate": True,
            "must_not_report_source_only_gain_as_performance": True,
        },
        "next_minimum_direction": {
            "name": "METADATA_ONLY_BANK_CYCLE_MONOTONIC_K6_GROUPING",
            "keep_descriptor_issue_width": 1,
            "keep_descriptor_packet_bytes": 64,
            "keep_resident_contexts": 16,
            "keep_fanout": 6,
            "future_wait_or_hold_permitted": False,
            "additional_vector_payload_storage_bytes": 0,
            "selection_scope": "only currently prepared descriptors",
            "metadata": "reuse existing delta/bank signature metadata; do not read vector payload",
            "monotonic_gate": (
                "at every legacy K6 decision point, accept an alternate group only "
                "when its exact union bank-issue cycles are no greater than the "
                "legacy group's cycles; deterministic legacy tie fallback"),
            "predeclared_required_gates": [
                "selector-disabled exact W1 reproduction",
                "per-decision bank-cycle monotonicity",
                "descriptor/DAG/parent/signed-update conservation",
                "no additional command, response or parent waits",
                "each-record and each-sample source cycles non-regression",
                "each-sample integrated cycles non-regression",
                "aggregate integrated improvement at least frozen 0.5 percent",
            ],
        },
        "claim_policy": {
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M93 independent W1={} W2={} W4={} W2_delta={} W4_delta={} NO_GO"
          .format(c1["integrated_cycles"], c2["integrated_cycles"],
                  c4["integrated_cycles"], integrated_delta, w4_delta))


if __name__ == "__main__":
    main()

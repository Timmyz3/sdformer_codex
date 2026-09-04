#!/usr/bin/env python3
"""Independent static/mathematical preflight for M507.

This checker never imports or executes the production M507 analyzer.  It
locks the reviewed identities, verifies frozen input identities, searches the
source AST/text for accounting invariants, and evaluates three hand-derived
boundary groups from the frozen contract equation.
"""

import ast
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/m507_h67_apec_g2_same_resource_cycle_fastkill_contract_r1_20260827.json"
ANALYZER = ROOT / "system_simulator/scripts/analyze_m507_h67_apec_g2_same_resource_cycle_fastkill.py"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "a2646134822d2074bc810004576dc0ffc6be04a5f4417b08c477d9c2a8a90410",
    "analyzer": "213976d42c83b7f3512b62e35c2c9e6a7763e1953d67e618517dc5897291db92",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def service(events, taps, model):
    co = int(model["output_channels"])
    lanes = int(model["compute_lanes"])
    bw = int(model["weight_bytes_per_cycle"])
    compute = events * math.ceil(co * taps / lanes)
    weight = math.ceil(events * co * taps / bw)
    return compute, weight


def boundary_case(count0, count1, common, taps0, taps1, union_taps, model):
    bitmap = int(model["bitmap_pair_read_cycles"])
    compare = int(model["exact_compare_cycles"])
    startup = int(model["weight_startup_latency_cycles"])
    scratch_bw = int(model["scratch_bytes_per_cycle"])
    co = int(model["output_channels"])
    acc = int(model["accumulator_bits"])

    b0c, b0w = service(count0, taps0, model)
    b1c, b1w = service(count1, taps1, model)
    bevents = count0 + count1
    baseline = bitmap + max(b0c + b1c, b0w + b1w + (startup if bevents else 0))

    terms = ((count0 - common, taps0),
             (count1 - common, taps1),
             (common, union_taps))
    ccompute = sum(service(events, taps, model)[0] for events, taps in terms)
    cweight = sum(service(events, taps, model)[1] for events, taps in terms)
    cevents = count0 + count1 - common
    scratch_pass = 0
    if common:
        scratch_bytes = math.ceil(co * union_taps * acc / 8)
        scratch_pass = math.ceil(scratch_bytes / scratch_bw)
    candidate = (bitmap + compare +
                 max(ccompute, cweight + (startup if cevents else 0)) +
                 3 * scratch_pass)
    return {
        "baseline_cycles": baseline,
        "candidate_cycles": candidate,
        "speedup": baseline / candidate,
        "scratch_pass_cycles": scratch_pass,
        "production_scratch_cycles": 3 * scratch_pass,
        "minimum_serial_sync_1r1w_cycles_with_two_read_tails": (
            3 * scratch_pass + (2 if scratch_pass else 0)),
    }


def main():
    identities = {
        "contract": sha256(CONTRACT),
        "analyzer": sha256(ANALYZER),
        "docs359": sha256(DOCS359),
    }
    require(identities == EXPECTED, "reviewed identity drift: " + repr(identities))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = ANALYZER.read_text(encoding="utf-8")
    ast.parse(source)
    for name, spec in contract["inputs"].items():
        path = ROOT / spec["path"]
        require(path.is_file(), "missing input " + name)
        require(sha256(path) == spec["sha256"], "input SHA drift " + name)

    static = {
        "scratch_conflict_counter_has_increment":
            'counters["scratch_port_conflict_cycles"] +=' in source,
        "queue_occupancy_is_modeled": "queue_occupancy" in source,
        "weight_bank_conflict_is_modeled": "weight_bank_conflict" in source,
        "train_m501_ledger_is_compared": (
            'cohort["cohort"] == "train_calibration_s32"' in source),
        "baseline_has_destination_commit_term": (
            "total_baseline_cycles += bitmap_read_cycles + bexec +" in source),
        "candidate_has_serial_scratch_term": (
            "bitmap_read_cycles + compare_cycles + cexec + scratch_cycles" in source),
        "same_resource_gate_uses_declared_boolean": (
            'model["same_top_ports_frequency_lanes_and_sram"] is True' in source),
    }
    model = contract["cycle_model"]
    cases = {
        "interior_full_overlap_one_event": boundary_case(1, 1, 1, 9, 9, 9, model),
        "interior_no_overlap_one_each": boundary_case(1, 1, 0, 9, 9, 9, model),
        "top_edge_full_overlap_one_event": boundary_case(1, 1, 1, 6, 6, 6, model),
    }
    print(json.dumps({
        "schema": "m507_preflight_independent_static_audit_v1",
        "identities": identities,
        "static_findings": static,
        "boundary_cases": cases,
        "verdict": "NO_GO_REVISE_BEFORE_ONE_SHOT_EXECUTION",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seal the ten-sample M66 VCS/replay development receipt."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
RESULTS = HW / "results"
EXPECTED_SIMV_SHA256 = "839d599287f63b7a973688253c815d8549448a1a0f8078e9185d6f3d098333cf"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_manifest(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines, "empty SHA manifest: {}".format(path))
    for line in lines:
        expected, raw_path = line.split("  ", 1)
        target = Path(raw_path)
        require(target.is_file(), "manifest target missing: {}".format(target))
        require(sha256(target) == expected,
                "manifest SHA mismatch: {}".format(target))


def nearest_rank(values, numerator, denominator):
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def sample_run(sample_id):
    sid = "{:02d}".format(sample_id)
    run = RESULTS / (
        "m68_m66_all10_vcs_dev_s{}_r1_20260823/s{}".format(sid, sid))
    require(run.is_dir(), "M68 sample run missing: {}".format(sid))
    verify_sha_manifest(run / "output_manifest.sha256")
    prelaunch = (run / "prelaunch_input.sha256").read_text(
        encoding="utf-8").splitlines()
    require(prelaunch and prelaunch[0].split()[0] == EXPECTED_SIMV_SHA256,
            "M68 compiled simv identity drift")
    complete = (run / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
    require("PASS_M68_SAMPLE_{}_M66_LOOKAHEAD_VCS_REPLAY".format(sid) in complete,
            "M68 unique completion marker missing")
    require("SYSTEM_SPEEDUP_ADMITTED=false" in complete and
            "PAPER_PPA_READY=false" in complete,
            "M68 claim boundary drift")
    replay_path = run / "m68_s{}_ledger_replay.json".format(sid)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    require(replay["sample_id"] == sample_id, "M68 sample identity mismatch")
    require(replay["status"] == "PASS_M66_STREAMING_FIFO_TAG_ARITHMETIC_REPLAY",
            "M68 replay status mismatch")
    require(replay["functional_mismatch_count"] == 0,
            "M68 functional mismatch")
    require(replay["accepted_requests"] == replay["accepted_responses"],
            "M68 request/response conservation failure")
    require(replay["accepted_outputs"] == 2592000,
            "M68 command/output extent mismatch")
    require(replay["maximum_context_occupancy"] <= 8 and
            replay["maximum_complete_occupancy"] <= 8 and
            replay["maximum_metadata_occupancy"] <= 1,
            "M68 queue bound exceeded")
    require(replay["metadata_fifo_final_occupancy"] == 0,
            "M68 final metadata occupancy nonzero")
    require(replay["launch_phase"]["prelaunch_artificial_bubbles"] == 0,
            "M68 prelaunch artificial bubble present")
    return {
        "sample_id": sample_id,
        "run": str(run.relative_to(HW)),
        "replay_sha256": sha256(replay_path),
        "accepted_requests": replay["accepted_requests"],
        "accepted_responses": replay["accepted_responses"],
        "accepted_outputs": replay["accepted_outputs"],
        "rtl_cycles": replay["rtl_cycles"],
        "m53_transaction_model_cycles": replay["m53_transaction_model_cycles"],
        "rtl_minus_m53_transaction_cycles": (
            replay["rtl_minus_m53_transaction_cycles"]),
        "maximum_context_occupancy": replay["maximum_context_occupancy"],
        "maximum_complete_occupancy": replay["maximum_complete_occupancy"],
        "response_tag_wraps": replay["response_tag_wraps"],
        "seamless_launches": replay["seamless_launches"],
    }


def build(output):
    require(not output.exists(), "refusing M68 all10 receipt overwrite")
    rows = [sample_run(sample_id) for sample_id in range(10)]
    total_rtl = sum(row["rtl_cycles"] for row in rows)
    total_model = sum(row["m53_transaction_model_cycles"] for row in rows)
    total_delta = sum(row["rtl_minus_m53_transaction_cycles"] for row in rows)
    require(total_rtl - total_model == total_delta,
            "M68 aggregate cycle conservation failure")
    require(total_rtl == 79927612 and total_model == 79869808,
            "M68 frozen all10 cycle totals drift")
    require(sum(row["accepted_requests"] for row in rows) == 68847096,
            "M68 frozen request total drift")
    require(sum(row["accepted_outputs"] for row in rows) == 25920000,
            "M68 frozen output total drift")
    rtl_values = [row["rtl_cycles"] for row in rows]
    delta_values = [row["rtl_minus_m53_transaction_cycles"] for row in rows]
    payload = {
        "schema": "m68_m66_all10_vcs_exact_replay_development_receipt_v1",
        "status": "PASS_M68_ALL10_M66_VCS_EXACT_REPLAY_MEMORY_PHYSICAL_UNADMITTED",
        "identity": {
            "builder_sha256": sha256(Path(__file__).resolve()),
            "compiled_simv_sha256": EXPECTED_SIMV_SHA256,
        },
        "population": {
            "samples": 10,
            "accepted_requests": sum(row["accepted_requests"] for row in rows),
            "accepted_responses": sum(row["accepted_responses"] for row in rows),
            "accepted_outputs": sum(row["accepted_outputs"] for row in rows),
            "functional_mismatch_count": 0,
        },
        "cycle_observation": {
            "rtl_cycles_total": total_rtl,
            "m53_transaction_model_cycles_total": total_model,
            "rtl_minus_model_cycles_total": total_delta,
            "rtl_over_model": total_rtl / total_model,
            "model_over_rtl": total_model / total_rtl,
            "rtl_cycles_p50_nearest_rank": nearest_rank(rtl_values, 50, 100),
            "rtl_cycles_p95_nearest_rank": nearest_rank(rtl_values, 95, 100),
            "rtl_minus_model_maximum": max(delta_values),
            "rtl_minus_model_maximum_sample": max(
                rows, key=lambda row: row["rtl_minus_m53_transaction_cycles"])["sample_id"],
            "interpretation": (
                "phase-safe VCS makespan is 0.0724 percent above the precomputed "
                "transaction targets across all10; this is measured RTL schedule "
                "overhead, not a system or memory-physical speedup"),
        },
        "samples": rows,
        "claim_boundary": {
            "accepted": (
                "all10 exact arithmetic/tag/ready-valid replay of the frozen offline "
                "M53 descriptor streams through M66 RTL"),
            "offline_descriptor_fetch_bytes_charged": False,
            "online_selector_implemented": False,
            "weight_sram_ports_and_macros_implemented": False,
            "dram_address_timed": False,
            "full_network_or_system_cycles": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "date_headline": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M68 all10 rtl={} model={} delta={} requests={}".format(
        total_rtl, total_model, total_delta,
        payload["population"]["accepted_requests"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()

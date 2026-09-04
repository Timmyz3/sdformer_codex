#!/usr/bin/env python3
"""Producer validator for the sealed M54 exact-SHA VCS/SVA run."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


HW_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = HW_ROOT / "contracts/m54_r1_exact_sha_vcs_receipt_r1_20260823.json"
EXPECTED_RECEIPT_SHA256 = (
    "c5ba3b3ac468ef736a478c3eb65157d61653d629c5d0fd7c29cdb58dc0c74546")
RUN_DIR = Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
    "dc_handoff/runs/m54_k4_ctx16_atomic_exact_sha_vcs_r1_20260823")


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
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        value = {}
        for key, item in raw_pairs:
            require(key not in value, "duplicate key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def check_call(command, cwd, label):
    result = subprocess.run(command, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-1000:],
                result.stderr[-1000:]))
    return result


def validate(rerun_tools):
    require(RECEIPT.is_file() and sha256(RECEIPT) == EXPECTED_RECEIPT_SHA256,
            "M54 exact-SHA receipt drift")
    receipt = strict_json(RECEIPT)
    require(receipt["schema"] == "m54_r1_exact_sha_vcs_receipt_v1" and
            receipt["status"] ==
            "PASS_M54_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER",
            "M54 receipt identity/status mismatch")
    canonical = receipt["canonical_run"]
    require(Path(canonical["path"]) == RUN_DIR and
            canonical["tool"] == "Synopsys VCS V-2023.12-SP1_Full64" and
            canonical["compile_rc"] == 0 and
            canonical["simulation_rc"] == 0 and
            canonical["sealed_read_only"] is True and
            canonical["non_overwriting"] is True and
            canonical["dc_launched"] is False and
            canonical["open_source_simulator_used"] is False,
            "M54 canonical run identity mismatch")
    require(RUN_DIR.is_dir() and
            not (RUN_DIR.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP |
                                            stat.S_IWOTH)),
            "M54 run directory missing/writable")
    for path in RUN_DIR.rglob("*"):
        if path.is_symlink():
            continue
        require(not (path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP |
                                            stat.S_IWOTH)),
                "M54 sealed path writable: {}".format(path))

    for name, row in receipt["source_anchors"].items():
        path = (HW_ROOT / row["path"]).resolve()
        require(path.is_file() and sha256(path) == row["sha256"],
                "M54 source anchor drift: {}".format(name))
    evidence_names = {
        "compile_log_sha256": "compile.raw.log",
        "completion_seal_sha256": "completion_seal.sha256",
        "handshake_ledger_sha256": "m54_handshake_ledger.log",
        "input_manifest_sha256": "input_sha256.txt",
        "ledger_replay_result_sha256": "m54_ledger_replay.json",
        "local_seal_sha256": "run_local_seal.sha256",
        "miter_log_sha256": "miter.raw.log",
        "output_manifest_sha256": "output_sha256.txt",
        "preflight_receipt_sha256": "preflight_receipt.json",
        "preflight_sha_checks_sha256": "preflight_sha_checks.txt",
        "run_complete_sha256": "RUN_COMPLETE.txt",
        "runner_status_sha256": "runner_status.txt",
        "simulation_log_sha256": "sim.raw.log",
        "sva_cover_receipt_sha256": "sva_cover_matches.txt",
    }
    require(set(evidence_names) == set(receipt["canonical_evidence_anchors"]),
            "M54 canonical evidence population mismatch")
    for key, name in evidence_names.items():
        path = RUN_DIR / name
        require(path.is_file() and
                sha256(path) == receipt["canonical_evidence_anchors"][key],
                "M54 canonical evidence drift: {}".format(name))

    check_call(["sha256sum", "--strict", "-c",
                str(RUN_DIR / "input_sha256.txt")], HW_ROOT,
               "M54 input manifest")
    check_call(["sha256sum", "--strict", "-c",
                str(RUN_DIR / "output_sha256.txt")], RUN_DIR,
               "M54 output manifest")
    check_call(["sha256sum", "--strict", "-c", "run_local_seal.sha256"],
               RUN_DIR, "M54 local seal")
    check_call(["sha256sum", "--strict", "-c", "completion_seal.sha256"],
               RUN_DIR, "M54 completion seal")
    require((RUN_DIR / "compile.rc").read_text().strip() == "0" and
            (RUN_DIR / "sim.rc").read_text().strip() == "0" and
            (RUN_DIR / "RUN_COMPLETE.txt").read_text().splitlines() == [
                "M54_R1_RUN_COMPLETE=PASS",
                "all_functional_SVA_ledger_and_SHA_checks_complete=true",
                "claim_scope=STANDALONE_VCS_SVA_ONLY_PENDING_INDEPENDENT_HAMMER",
                "dc_launched=false",
            ], "M54 completion markers drift")

    results = receipt["canonical_results"]
    require(results["commands"] == results["outputs"] == 67 and
            results["groups"] == 24 and results["accepted_requests"] == 53 and
            results["physical_unique_weight_row_issues"] == 381 and
            results["logical_destination_updates"] == 450 and
            results["functional_mismatch_count"] == 0 and
            results["ledger_mismatch_count"] == 0 and
            results["sva_assertion_failure_count"] == 0 and
            all(results[name] is True for name in (
                "atomic_push4_reached", "complete13_pop_push4_reached",
                "complete16_reached", "context16_reached",
                "context_ids_finite_4b_and_reuse_checked",
                "metadata16_reached",
                "response_tags_monotonic_modulo_16b")),
            "M54 canonical result mismatch")
    covers = {}
    for line in (RUN_DIR / "sva_cover_matches.txt").read_text().splitlines():
        key, value = line.split("=", 1)
        require(key not in covers, "duplicate cover receipt key")
        covers[key] = int(value)
    require(covers == receipt["sva_cover_matches"] and len(covers) == 32 and
            all(value > 0 for value in covers.values()),
            "M54 SVA cover mismatch/nonpositive")
    replay = strict_json(RUN_DIR / "m54_ledger_replay.json")
    require(replay["status"] ==
            "PASS_STANDALONE_M54_K4_C16_EXACT_LEDGER" and
            replay["mismatch_count"] == 0 and
            replay["commands"] == replay["outputs"] == 67 and
            replay["groups"] == 24 and replay["requests"] == 53 and
            replay["physical_unique_weight_row_issues"] == 381 and
            replay["logical_destination_updates"] == 450 and
            replay["group_relations"] == receipt["group_relations"],
            "M54 replay result mismatch")
    sim = (RUN_DIR / "sim.raw.log").read_text(encoding="utf-8")
    require("M54_ASSERTION_MODULE_ACTIVE=1" in sim and
            "M54_SVA_BOUND=1" in sim and
            "PASS M54 K4_CTX16_ATOMIC_UNION commands=67 outputs=67 groups=24 requests=53 context16=1 meta16=1 complete16=1 push4=1 pop13push4=1"
            in sim and "Offending" not in sim and "failed at" not in sim,
            "M54 simulation pass/failure marker mismatch")
    require(receipt["diagnostic_disclosure"]["noncanonical_entries"] == 5 and
            len(receipt["negative_attacks_passed"]) == 10 and
            receipt["m52_closure_scope"]["M52-P1-01"].startswith("CLOSED") and
            "physical" in receipt["m52_closure_scope"]["M52-P2-03"].lower(),
            "M54 disclosure/closure boundary mismatch")

    if rerun_tools:
        preflight = (HW_ROOT /
                     receipt["source_anchors"]["preflight_validator"]["path"])
        miter = HW_ROOT / receipt["source_anchors"]["ledger_replay"]["path"]
        with tempfile.TemporaryDirectory(prefix="m54_validator_") as temp:
            temp_path = Path(temp)
            preflight_output = temp_path / "preflight.json"
            preflight_run = check_call(
                ["/usr/bin/python3.6", str(preflight), "--output",
                 str(preflight_output)], HW_ROOT, "M54 preflight rerun")
            require("PASS M54 preflight" in preflight_run.stdout and
                    sha256(preflight_output) ==
                    receipt["canonical_evidence_anchors"][
                        "preflight_receipt_sha256"],
                    "M54 preflight rerun drift")
            replay_output = temp_path / "replay.json"
            replay_run = check_call(
                ["/usr/bin/python3.6", str(miter), "--ledger",
                 str(RUN_DIR / "m54_handshake_ledger.log"), "--output",
                 str(replay_output)], HW_ROOT, "M54 replay rerun")
            require("PASS M54 LEDGER" in replay_run.stdout and
                    sha256(replay_output) ==
                    receipt["canonical_evidence_anchors"][
                        "ledger_replay_result_sha256"],
                    "M54 replay rerun drift")

    return {
        "schema": "m54_r1_exact_sha_vcs_producer_validator_result_v1",
        "status": "PASS_M54_R1_EXACT_SHA_VCS_SVA_PRODUCER_VALIDATED",
        "receipt_sha256": sha256(RECEIPT),
        "canonical_run": str(RUN_DIR),
        "rerun_tools": bool(rerun_tools),
        "commands": 67,
        "outputs": 67,
        "groups": 24,
        "accepted_requests": 53,
        "physical_unique_weight_row_issues": 381,
        "logical_destination_updates": 450,
        "cover_count": 32,
        "attack_count": 10,
        "functional_mismatch_count": 0,
        "ledger_mismatch_count": 0,
        "sva_assertion_failure_count": 0,
        "dc_launched": False,
        "M52_cycles_admitted_as_RTL_cycles": False,
        "pending_independent_hammer": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-tools", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.rerun_tools)
    if args.output is not None:
        require(not args.output.exists(), "refusing validator output overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS M54 producer exact-SHA VCS/SVA commands=67 requests=53 outputs=67 covers=32 pending-hammer")


if __name__ == "__main__":
    main()

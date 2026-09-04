#!/usr/bin/env python3
"""Validate the frozen M49-r1 exact-SHA VCS/SVA producer receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "contracts/m49_r1_exact_sha_vcs_receipt_r1_20260823.json"
EXPECTED_RECEIPT_SHA256 = (
    "30bc288a16a2b317467481a625cb739805c72d16f0f643ca69ab09b73a65e0bc")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def validate_sha_manifest(path, base):
    count = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        fields = raw.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed SHA manifest line")
        target = Path(fields[1].lstrip(" *"))
        if not target.is_absolute():
            target = Path(base) / target
        require(target.is_file(), "manifest target missing: {}".format(target))
        require(sha256(target) == fields[0],
                "manifest target SHA drift: {}".format(target))
        count += 1
    require(count > 0, "empty SHA manifest")
    return count


def build():
    require(sha256(RECEIPT) == EXPECTED_RECEIPT_SHA256,
            "M49 receipt identity drift")
    receipt = read_json(RECEIPT)
    require(receipt["status"] ==
            "PASS_M49_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER",
            "M49 receipt status drift")
    for name, item in receipt["source_anchors"].items():
        path = ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "M49 source anchor drift: {}".format(name))

    run_dir = Path(receipt["canonical_run"]["path"])
    require(run_dir.is_dir(), "canonical run missing")
    require(not os.access(str(run_dir), os.W_OK),
            "canonical run directory remains writable")
    require(not (run_dir / "RUN_FAILED_OR_INCOMPLETE.txt").exists(),
            "canonical run has failure marker")
    anchors = receipt["canonical_evidence_anchors"]
    evidence_names = {
        "compile_log_sha256": "compile.raw.log",
        "simulation_log_sha256": "sim.raw.log",
        "miter_log_sha256": "miter.raw.log",
        "preflight_sha_checks_sha256": "preflight_sha_checks.txt",
        "input_manifest_sha256": "input_sha256.txt",
        "output_manifest_sha256": "output_sha256.txt",
        "local_seal_sha256": "run_local_seal.sha256",
        "completion_seal_sha256": "completion_seal.sha256",
        "run_complete_sha256": "RUN_COMPLETE.txt",
        "runner_status_sha256": "runner_status.txt",
        "cover_receipt_sha256": "sva_cover_matches.txt",
        "handshake_ledger_sha256": "m49_handshake_ledger.log",
        "ledger_replay_result_sha256": "m49_ledger_replay.json",
    }
    for anchor_name, filename in evidence_names.items():
        path = run_dir / filename
        require(path.is_file() and sha256(path) == anchors[anchor_name],
                "canonical evidence drift: {}".format(filename))
        require(not os.access(str(path), os.W_OK),
                "canonical evidence remains writable: {}".format(filename))

    input_count = validate_sha_manifest(run_dir / "input_sha256.txt", ROOT)
    output_count = validate_sha_manifest(run_dir / "output_sha256.txt", ROOT)
    local_count = validate_sha_manifest(run_dir / "run_local_seal.sha256",
                                        run_dir)
    completion_count = validate_sha_manifest(
        run_dir / "completion_seal.sha256", run_dir)
    require(input_count == 10 and output_count >= 90
            and local_count == 5 and completion_count == 4,
            "canonical manifest population drift")
    require((run_dir / "compile.rc").read_text().strip() == "0"
            and (run_dir / "sim.rc").read_text().strip() == "0",
            "canonical compile/sim rc is not zero")

    compile_log = (run_dir / "compile.raw.log").read_text(
        encoding="utf-8", errors="replace")
    require("Parsing design file 'rtl_m49/" in compile_log
            and "Parsing design file 'verif_m49/" in compile_log
            and "Parsing design file 'tb_m49/" in compile_log,
            "canonical compile did not parse all isolated M49 sources")
    require(not re.search(
        r"^(Warning|Error)-|(^|[^A-Za-z])(warning|error|fatal)([^A-Za-z]|$)",
        compile_log, re.IGNORECASE | re.MULTILINE),
        "canonical compile warning/error/fatal signature")

    sim_log = (run_dir / "sim.raw.log").read_text(
        encoding="utf-8", errors="replace")
    required_lines = (
        "M49_ASSERTION_MODULE_ACTIVE=1",
        "M49_SVA_BOUND=1",
        "M49_ATTACKS reset_request_stall=1 reset_output_stall=1 "
        "unexpected_response=1 duplicate_launch_pair=1 duplicate_relaunch=1 "
        "response_context0_mismatch=1 response_context1_mismatch=1 "
        "response_bank_mismatch=1 overlapping_masks=1 positive_overflow=1 "
        "negative_overflow=1",
        "PASS M49 K2_CTX8_ATOMIC_DUAL_ENQUEUE legal_tags=28 outputs=28 "
        "requests=71 context8=1 meta16=1 complete16=1",
    )
    for line in required_lines:
        require(sim_log.splitlines().count(line) == 1,
                "canonical simulation marker mismatch: {}".format(line))
    require(not re.search(r"failed at|Offending|assertion.*(fail|error)",
                          sim_log, re.IGNORECASE),
            "canonical SVA failure signature")

    covers = {}
    for raw in (run_dir / "sva_cover_matches.txt").read_text().splitlines():
        name, value = raw.split("=", 1)
        covers[name] = int(value)
    require(covers == receipt["sva_cover_matches"],
            "canonical cover receipt mismatch")
    require(all(value > 0 for value in covers.values()),
            "one or more canonical covers are zero")

    replay = read_json(run_dir / "m49_ledger_replay.json")
    result = receipt["canonical_results"]
    require(replay["status"] ==
            "PASS_STANDALONE_M49_K2_UNION_EXACT_LEDGER"
            and replay["commands"] == result["legal_tags"] == 28
            and replay["outputs"] == result["legal_outputs"] == 28
            and replay["groups"] == result["groups"] == 15
            and replay["requests"] == result["accepted_requests"] == 71
            and replay["physical_unique_weight_row_issues"] ==
                result["physical_unique_weight_row_issues"] == 491
            and replay["logical_destination_updates"] ==
                result["logical_destination_updates"] == 789
            and replay["mismatch_count"] == 0,
            "canonical ledger replay metric drift")
    require(len(receipt["negative_attacks_passed"]) == 11,
            "negative attack population drift")
    require(len(receipt["noncanonical_diagnostic_disclosure"]) == 5,
            "noncanonical diagnostic disclosure missing")
    require(receipt["claim_boundary"]["not_admitted"],
            "claim boundary missing")
    return {
        "schema": "m49_r1_exact_sha_vcs_validator_result_v1",
        "status": "PASS_M49_R1_PRODUCER_RECEIPT_PENDING_INDEPENDENT_HAMMER",
        "receipt_sha256": sha256(RECEIPT),
        "canonical_run": str(run_dir),
        "input_manifest_entries": input_count,
        "output_manifest_entries": output_count,
        "local_seal_entries": local_count,
        "completion_seal_entries": completion_count,
        "legal_tags": 28,
        "legal_outputs": 28,
        "accepted_requests": 71,
        "physical_unique_weight_row_issues": 491,
        "logical_destination_updates": 789,
        "functional_ledger_sva_mismatch_count": 0,
        "covers_nonzero": len(covers),
        "negative_attacks_passed": 11,
        "independent_hammer_required": True,
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build()
    if args.output is not None:
        require(not args.output.exists(), "refusing to overwrite validator output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("PASS M49-r1 producer receipt tags=28 outputs=28 requests=71 covers=15 attacks=11")


if __name__ == "__main__":
    main()

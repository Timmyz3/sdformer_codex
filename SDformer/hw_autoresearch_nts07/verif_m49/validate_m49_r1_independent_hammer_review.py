#!/usr/bin/env python3
"""Fail-closed validator for the M49-r1 independent hammer review."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = (ROOT / "results/m49_r1_independent_hammer_20260823"
          / "m49_r1_independent_hammer_review.json")
EXPECTED_REVIEW_SHA256 = "25e4989b347f40c5512667dc2260d45a8601bcd56cc7ab1815a0e6ed0a92f3b4"
EXPECTED_RECEIPT_SHA256 = "30bc288a16a2b317467481a625cb739805c72d16f0f643ca69ab09b73a65e0bc"
EXPECTED_INDEPENDENT_RESULT_SHA256 = "9b2440b400df8b0b429b526a20991296e9f5a9d467f1e2ecd91f18499d8780d1"
EXPECTED_PRODUCER_REPLAY_SHA256 = "6081823d38fbeec2ee8426794d29c06158c0f7687b30110acb4e4a98478e3928"


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


def manifest_count(path, base):
    count = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        fields = raw.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed manifest: {}".format(path))
        target = Path(fields[1].lstrip(" *"))
        if not target.is_absolute():
            target = Path(base) / target
        require(target.is_file(), "manifest target missing: {}".format(target))
        require(sha256(target) == fields[0],
                "manifest target drift: {}".format(target))
        count += 1
    require(count > 0, "empty manifest")
    return count


def validate(rerun):
    require(sha256(REVIEW) == EXPECTED_REVIEW_SHA256, "review identity drift")
    review = read_json(REVIEW)
    require(review["status"] == "PASS_INDEPENDENT_HAMMER_STANDALONE_ONLY",
            "review status drift")
    require(review["verdict"] ==
            "GO_STANDALONE_M49_R1_EXACT_SHA_VCS_K2C8_ONLY",
            "review verdict drift")
    require(review["score_0_to_100"] == 94, "review score drift")
    require(review["severity_counts"] == {"P0": 0, "P1": 0, "P2": 5},
            "severity population drift")
    require(len(review["findings"]) == 5 and
            all(item["severity"] == "P2" for item in review["findings"]),
            "finding population drift")

    for name, item in review["candidate_anchors"].items():
        path = ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "candidate anchor drift: {}".format(name))
    receipt_path = ROOT / review["candidate_anchors"]["producer_receipt"]["path"]
    require(sha256(receipt_path) == EXPECTED_RECEIPT_SHA256,
            "producer receipt drift")
    receipt = read_json(receipt_path)
    require(receipt["status"] ==
            "PASS_M49_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER",
            "producer status drift")

    canonical = Path(receipt["canonical_run"]["path"])
    require(canonical.is_dir() and not os.access(str(canonical), os.W_OK),
            "canonical run missing or writable")
    require(not (canonical / "RUN_FAILED_OR_INCOMPLETE.txt").exists(),
            "canonical failure marker exists")
    evidence_files = {
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
    for anchor, filename in evidence_files.items():
        path = canonical / filename
        require(path.is_file() and
                sha256(path) == receipt["canonical_evidence_anchors"][anchor],
                "canonical evidence drift: {}".format(filename))
        require(not os.access(str(path), os.W_OK),
                "canonical evidence writable: {}".format(filename))
    require(manifest_count(canonical / "input_sha256.txt", ROOT) == 10,
            "input manifest population")
    require(manifest_count(canonical / "output_sha256.txt", ROOT) >= 90,
            "output manifest population")
    require(manifest_count(canonical / "run_local_seal.sha256", canonical) == 5,
            "local seal population")
    require(manifest_count(canonical / "completion_seal.sha256", canonical) == 4,
            "completion seal population")

    compile_log = (canonical / "compile.raw.log").read_text(
        encoding="utf-8", errors="replace")
    require("Parsing design file 'rtl_m49/" in compile_log and
            "Parsing design file 'verif_m49/" in compile_log and
            "Parsing design file 'tb_m49/" in compile_log,
            "canonical source isolation drift")
    require(not re.search(
        r"^(Warning|Error)-|(^|[^A-Za-z])(warning|error|fatal)([^A-Za-z]|$)",
        compile_log, re.I | re.M), "canonical compile signature")
    sim_log = (canonical / "sim.raw.log").read_text(
        encoding="utf-8", errors="replace")
    producer_pass = ("PASS M49 K2_CTX8_ATOMIC_DUAL_ENQUEUE legal_tags=28 "
                     "outputs=28 requests=71 context8=1 meta16=1 complete16=1")
    require(sim_log.splitlines().count(producer_pass) == 1,
            "canonical pass marker")
    require(not re.search(r"failed at|Offending|assertion.*(fail|error)",
                          sim_log, re.I), "canonical assertion failure")
    require((canonical / "compile.rc").read_text().strip() == "0" and
            (canonical / "sim.rc").read_text().strip() == "0",
            "canonical return code")

    diagnostics = receipt["noncanonical_diagnostic_disclosure"]
    require(len(diagnostics) == 5 and diagnostics[0]["raw_log_preserved"] is False,
            "noncanonical disclosure drift")
    preserved = {
        "smoke1_sva": (
            "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_smoke.UJyqAp/sim.log",
            "sim_log_sha256"),
        "smoke1_no_sva_isolation": (
            "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_nosva_smoke.h6U8Cb/sim.log",
            "sim_log_sha256"),
        "smoke2_sva": (
            "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_smoke2.lSnXzQ/sim.log",
            "sim_log_sha256"),
        "smoke3_sva": (
            "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_smoke3.UmuZ4q/sim.log",
            "sim_log_sha256"),
    }
    by_stage = dict((item["stage"], item) for item in diagnostics)
    for stage, (path_raw, hash_key) in preserved.items():
        path = Path(path_raw)
        require(path.is_file() and sha256(path) == by_stage[stage][hash_key],
                "noncanonical diagnostic drift: {}".format(stage))
    partial_ledger = Path(
        "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_smoke.UJyqAp/m49_handshake_ledger.log")
    require(sha256(partial_ledger) == by_stage["smoke1_sva"]["partial_ledger_sha256"],
            "noncanonical partial ledger drift")

    independent = review["independent_evidence"]
    for key in ("checker", "checker_result", "independent_vcs_testbench"):
        item = independent[key]
        path = ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "independent evidence drift: {}".format(key))
    independent_result_path = ROOT / independent["checker_result"]["path"]
    require(sha256(independent_result_path) == EXPECTED_INDEPENDENT_RESULT_SHA256,
            "independent result drift")
    independent_result = read_json(independent_result_path)
    require(independent_result["status"] ==
            "PASS_M49_R1_INDEPENDENT_LEDGER_STATIC_FIFO_HAMMER",
            "independent result status")
    require(independent_result["independent_ledger"] == {
        "commands": 28, "groups": 15, "logical_updates": 789,
        "outputs": 28, "physical_reads": 491,
        "relations": {"K1": 2, "K2_FULL_SHARE": 10,
                      "K2_NO_SHARE": 1, "K2_PARTIAL_SHARE": 2},
        "requests": 71, "shared_updates": 298,
    }, "independent ledger metrics")
    require(len(independent_result["negative_mutations_rejected"]) == 10 and
            independent_result["producer_modules_imported"] == [],
            "independent mutation/isolation drift")
    require(independent_result["fifo_state_enumeration"] == {
        "completion_transition_cases": 102,
        "count15_pop_push2_next_count": 16,
        "count16_pop_push1_next_count": 16,
        "metadata_transition_cases": 68,
        "pointer_transition_cases": 1632,
    }, "FIFO enumeration drift")

    edge = independent["independent_vcs_run"]
    edge_run = Path(edge["path"])
    require(edge_run.is_dir() and not os.access(str(edge_run), os.W_OK),
            "independent VCS run missing or writable")
    for field, filename in (("compile_log_sha256", "compile.log"),
                            ("simulation_log_sha256", "sim.log"),
                            ("simv_sha256", "simv")):
        path = edge_run / filename
        require(path.is_file() and sha256(path) == edge[field],
                "independent VCS evidence drift: {}".format(filename))
        require(not os.access(str(path), os.W_OK),
                "independent VCS evidence writable: {}".format(filename))
    edge_compile = (edge_run / "compile.log").read_text(
        encoding="utf-8", errors="replace")
    require("Parsing design file 'rtl_m49/" in edge_compile and
            "Parsing design file 'verif_m49/" in edge_compile and
            "Parsing design file 'results/m49_r1_independent_hammer_20260823/" in edge_compile,
            "independent VCS source parse drift")
    require(not re.search(
        r"^(Warning|Error)-|(^|[^A-Za-z])(warning|error|fatal)([^A-Za-z]|$)",
        edge_compile, re.I | re.M), "independent compile signature")
    edge_sim = (edge_run / "sim.log").read_text(
        encoding="utf-8", errors="replace")
    require(edge_sim.splitlines().count(edge["pass_marker"]) == 1,
            "independent VCS pass marker")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in edge_sim,
            "independent VCS tool identity")
    require(not re.search(r"failed at|Offending|assertion.*(fail|error)",
                          edge_sim, re.I), "independent VCS assertion failure")

    require(len(review["admitted_claims"]) == 3 and
            len(review["forbidden_claims"]) == 5,
            "claim-boundary population")
    forbidden_text = " ".join(review["forbidden_claims"])
    for token in ("298/789", "M45", "PPA", "3x", "DATE", "best-paper"):
        require(token in forbidden_text, "forbidden claim token absent: {}".format(token))

    if rerun:
        producer_validator = ROOT / review["candidate_anchors"]["producer_validator"]["path"]
        subprocess.check_call(["/usr/bin/python3.6", str(producer_validator)],
                              cwd=str(ROOT))
        with tempfile.TemporaryDirectory(prefix="m49_review_rerun_") as tmp:
            tmp_path = Path(tmp)
            producer_out = tmp_path / "producer_replay.json"
            subprocess.check_call([
                "/usr/bin/python3.6",
                str(ROOT / review["candidate_anchors"]["producer_ledger_replay"]["path"]),
                "--ledger", str(canonical / "m49_handshake_ledger.log"),
                "--output", str(producer_out)], cwd=str(ROOT))
            require(sha256(producer_out) == EXPECTED_PRODUCER_REPLAY_SHA256,
                    "producer replay rerun drift")
            independent_out = tmp_path / "independent_replay.json"
            subprocess.check_call([
                "/usr/bin/python3.6",
                str(ROOT / independent["checker"]["path"]),
                "--ledger", str(canonical / "m49_handshake_ledger.log"),
                "--rtl", str(ROOT / review["candidate_anchors"]["rtl"]["path"]),
                "--output", str(independent_out)], cwd=str(ROOT))
            require(sha256(independent_out) == EXPECTED_INDEPENDENT_RESULT_SHA256,
                    "independent rerun drift")

    return {
        "schema": "m49_r1_independent_hammer_review_validator_result_v1",
        "status": "PASS_M49_R1_INDEPENDENT_HAMMER_REVIEW",
        "review_sha256": sha256(REVIEW),
        "verdict": review["verdict"],
        "score_0_to_100": 94,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 5},
        "producer_validator_rerun": bool(rerun),
        "producer_and_independent_ledger_rerun": bool(rerun),
        "independent_vcs_edge_pass": True,
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.rerun)
    if args.output is not None:
        require(not args.output.exists(), "refusing to overwrite validator output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("PASS M49-r1 independent hammer score=94 P0=0 P1=0 P2=5 standalone-only")


if __name__ == "__main__":
    main()

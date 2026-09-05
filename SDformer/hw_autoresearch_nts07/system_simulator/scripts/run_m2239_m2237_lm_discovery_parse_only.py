#!/usr/bin/env python3
"""Publish one source-reviewed CPU-only recovery of immutable M2223 logs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
CONTRACT = HW / "contracts/m2237_m2223_lm_discovery_parse_only_source_contract_r1_20260905.json"
CHECKER = HW / "system_simulator/scripts/check_m2237_m2223_lm_discovery_parse_only.py"
REVIEW = HW / "reviews/m2238_m2237_lm_discovery_parse_only_source_hammer_r1_20260905"
RESULT = HW / "results/m2239_m2223_lm_discovery_parse_only_r1_20260905"
ATTEMPT = HW / "results/.m2239_m2223_lm_discovery_cpu_parse_attempt_consumed"
STATUS = "RAW_PASS_M2239_M2223_PARSE_ONLY_PENDING_M2240_RESULT_REVIEW"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), "missing/symlink input " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(directory):
    need(directory.is_dir() and not directory.is_symlink(), "invalid sealed directory")
    paths = list(directory.rglob("*"))
    need(not any(p.is_symlink() for p in paths), "sealed symlink")
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        need(name not in listed and not Path(name).is_absolute()
             and ".." not in Path(name).parts, "unsafe/duplicate seal entry")
        need(sha(directory / name) == digest, "sealed file drift " + name)
        listed[name] = digest
    actual = {str(p.relative_to(directory)) for p in paths if p.is_file()}
    need(actual == set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}, "nonexhaustive seal")
    need((directory / "SHA256SUMS.seal.sha256").read_text().split()
         == [sha(directory / "SHA256SUMS"), "SHA256SUMS"], "outer seal mismatch")


def seal_new(directory):
    files = sorted(p for p in directory.iterdir() if p.is_file()
                   and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (directory / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files))
    (directory / "SHA256SUMS.seal.sha256").write_text(sha(directory / "SHA256SUMS") + "  SHA256SUMS\n")
    verify_seal(directory)


def validate_inputs(contract):
    for relative, expected in contract["pinned_files"].items():
        need(sha(REPO / relative) == expected, "input drift " + relative)
    for relative in contract["sealed_directories"]:
        verify_seal(REPO / relative)
    need(sha(Path(sys.executable).resolve()) == contract["python_sha256"], "Python identity")
    old = json.loads((HW / contract["m2221_contract"]).read_text())
    for relative, expected in old["source_inventory"].items():
        need(sha(REPO / relative) == expected, "frozen M2221 source drift")
    forensic = json.loads((HW / contract["m2224_review"]).read_text())
    need(forensic["status"] == "PASS_M2224_FAILURE_FORENSICS__NEW_IDENTITY_PARSE_ONLY_RECOVERY_AUTHORIZED"
         and forensic["authorization"]["new_identity_parse_only_recovery"], "M2224 authorization")
    need(forensic["identity"]["raw_directory"] == contract["raw_directory"]
         and forensic["authorization"]["inputs_exact_raw_manifest_sha256"]
         == sha(REPO / contract["raw_directory"] / "SHA256SUMS"), "M2224 raw mapping identity")
    raw = REPO / contract["raw_directory"]
    need((raw / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
         == "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=2\nretry=false\n", "old failure changed")
    need(not (raw / "receipt.json").exists() and not (raw / "RUN_COMPLETE.txt").exists(), "old result upgraded")


def load_checker():
    spec = importlib.util.spec_from_file_location("m2237_lm_recovery", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(review, contract_sha):
    need(review["status"] == "PASS_M2238_M2237_LM_PARSE_ONLY_SOURCE__M2239_CPU_PARSE_AUTHORIZED", "M2238 status")
    need(review["score_over_100"] >= 95 and review["severity_counts"]["p0"] == 0
         and review["severity_counts"]["p1"] == 0, "M2238 decisive finding")
    need(review["identity"] == {"source_contract_sha256": contract_sha,
         "checker_sha256": sha(CHECKER), "runner_sha256": sha(Path(__file__).resolve())}, "M2238 identity")
    need(review["authorization"] == {"cpu_parse_runs": 1, "license_queries": 0,
         "lm_runs": 0, "eda_runs": 0, "gpu_runs": 0, "automatic_retry": False}, "M2238 budget")


def execute(contract, contract_sha, review_sha):
    need(sha(REVIEW / "review.json") == review_sha, "caller-pinned M2238 review")
    verify_seal(REVIEW)
    gate(json.loads((REVIEW / "review.json").read_text()), contract_sha)
    need(not any(p.exists() or p.is_symlink() for p in (RESULT, ATTEMPT)), "M2239 already consumed")
    work = HW / ("results/.m2239_lm_parse_only_work." + str(os.getpid()))
    need(not work.exists() and not work.is_symlink(), "work identity exists")
    ATTEMPT.mkdir()  # Atomic one-shot admission, even across competing CPU launchers.
    (ATTEMPT / "ATTEMPT_CONSUMED.txt").write_text("M2239_CPU_PARSE_CONSUMED\nretry=false\nlm_runs=0\neda_runs=0\n")
    seal_new(ATTEMPT)
    work.mkdir()
    try:
        with (work / "parser.stdout.log").open("w") as out, (work / "parser.stderr.log").open("w") as err:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    receipt = load_checker().validate(REPO / contract["raw_directory"])
                    validate_inputs(contract)
                    need(receipt["status"] == STATUS, "unexpected parse receipt")
                    print(STATUS)
                except BaseException:
                    traceback.print_exc()
                    raise
        receipt.update({"source_contract_sha256": contract_sha, "m2238_review_sha256": review_sha,
            "pinned_inputs": contract["pinned_files"], "parse_execution": {
                "cpu_parse_runs": 1, "license_queries": 0, "lm_runs": 0, "eda_runs": 0, "gpu_runs": 0},
            "claim_scope": "command/option discovery only; no conversion, library compatibility or P&R admission"})
        (work / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (work / "RUN_COMPLETE.txt").write_text(STATUS + "\n")
        seal_new(work)
        need(not RESULT.exists() and not RESULT.is_symlink(), "result identity appeared")
        work.rename(RESULT)
    except BaseException:
        if work.exists():
            (work / "RUN_FAILED.txt").write_text("FAILED_M2239_PARSE_ONLY_DO_NOT_CITE\nretry=false\n")
            seal_new(work)
            work.rename(Path(str(RESULT) + ".failed." + str(os.getpid()) + ".quarantine"))
        raise
    print(STATUS)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--contract-sha256")
    ap.add_argument("--source-review-sha256")
    args = ap.parse_args()
    if args.execute:
        need(args.contract_sha256 and args.source_review_sha256, "caller SHA pins required")
        need(sha(CONTRACT) == args.contract_sha256, "caller-pinned source contract")
    contract = json.loads(CONTRACT.read_text())
    validate_inputs(contract)
    if args.static:
        print("PASS_M2237_LM_PARSE_ONLY_STATIC__NO_M2239_ATTEMPT")
    else:
        execute(contract, args.contract_sha256, args.source_review_sha256)


if __name__ == "__main__":
    main()

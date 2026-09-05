#!/usr/bin/env python3
"""M2229 source: recover pinned M2215 raw logs under M2231, with zero EDA."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
CONTRACT = HW / "contracts/m2229_m2215_causal_parse_only_source_contract_r1_20260905.json"
Q = HW / "results/m2215_m2213_preread_postread_causal_directed_vcs_r1_20260904.failed_or_incomplete.3812622.quarantine"
SOURCE_REVIEW = HW / "reviews/m2230_m2229_causal_parse_only_source_hammer_r1_20260905"
RESULT = HW / "results/m2231_m2215_causal_parse_only_successor_r1_20260905"
ATTEMPT = HW / "results/.m2231_causal_parse_only_attempt_consumed"
LOCK = HW / "results/.m2231_causal_parse_only_lock"
TB_REL = "hw_autoresearch_nts07/tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv"
RAW_PASS = "RAW_PASS_M2215_M2213_PREREAD_POSTREAD_CAUSAL_DIRECTED"
STATUS = "RAW_PASS_M2231_M2215_PARSE_ONLY_PENDING_M2232_RESULT_REVIEW"
EXPECTED = dict(ordinary_reads=2304, postread_reads=2304, preread_reads=576,
    suppressed_reads=1728, ordinary_cycles=3386, postread_cycles=3386,
    preread_cycles=1119, rows=24, hits_post=18, hits_pre=18,
    real_postread_rows=18, postread_bundle_req=216, postread_bundle_rsp=216,
    postread_bank_req=1728, postread_bank_rsp=1728, identity_rsp=216,
    commits_each=24, products_each=4608, golden_mismatches=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), "missing/symlink file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(directory):
    require(directory.is_dir() and not directory.is_symlink(), "bad sealed directory")
    paths = list(directory.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "sealed directory contains symlink")
    manifest = directory / "SHA256SUMS"
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in entries and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "bad/duplicate manifest path")
        require(sha(directory / name) == digest, "sealed file drift: " + name)
        entries[name] = digest
    actual = {str(p.relative_to(directory)) for p in paths if p.is_file()}
    require(actual == set(entries) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}, "nonexhaustive seal")
    require((directory / "SHA256SUMS.seal.sha256").read_text().split()
            == [sha(manifest), "SHA256SUMS"], "outer seal mismatch")


def seal_new(directory):
    files = sorted(p for p in directory.iterdir() if p.is_file()
                   and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(not any(p.is_symlink() for p in directory.iterdir()), "new symlink")
    (directory / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files))
    (directory / "SHA256SUMS.seal.sha256").write_text(sha(directory / "SHA256SUMS") + "  SHA256SUMS\n")
    verify_seal(directory)


def parse_raw(comp, sim, rc, tb):
    require(rc.strip() == "0", "nonzero simulator return code")
    require("Chronologic VCS (TM)" in comp and "Version V-2023.12-SP1_Full64" in comp,
            "compiler identity missing")
    require("All of 7 modules done" in comp and "simv up to date" in comp
            and "CPU time:" in comp and "to link" in comp, "incomplete compiler/link log")
    require("Chronologic VCS simulator copyright" in sim
            and "Runtime version V-2023.12-SP1_Full64" in sim, "runtime identity missing")
    require(not re.search(r"Error(?:-|\[|:)|Fatal(?:-|:)|\$fatal|assertion\s+failed", comp + sim, re.I),
            "compile/runtime/assertion failure")
    warnings = re.findall(r"Warning-\[(\w+)\]", comp)
    require(len(warnings) == 27 and warnings.count("KUAI") == 26
            and warnings.count("LNX_OS_VERUN") == 1, "unreviewed compiler warning")
    require(len(re.findall(r"\bWarning(?:-|\[|:)", comp, re.I)) == 27,
            "unparsed compiler warning form")
    require("Linux version 'Rocky Linux release 8.10 (Green Obsidian)' is not supported" in comp,
            "changed platform warning")
    locations = re.findall(r"Warning-\[KUAI\] Keyword used as identifier\n([^\n]+), (\d+)\n  '([^']+)'", comp)
    require(len(locations) == 26, "unparsed KUAI warning")
    tb_lines = tb.splitlines()
    for path, line, identifier in locations:
        require(path == TB_REL and identifier == "context", "unreviewed keyword warning")
        require(0 < int(line) <= len(tb_lines)
                and re.search(r"\bcontext\b", tb_lines[int(line)-1]), "warning/source mismatch")
    require(not re.search(r"\bWarning(?:-|\[|:)", sim, re.I), "unreviewed runtime warning")
    passes = re.findall(r"^" + RAW_PASS + r" (.*)$", sim, re.M)
    covers = re.findall(r"^M2213_COVER (.*)$", sim, re.M)
    require(len(passes) == len(covers) == 1 and sim.count(RAW_PASS) == 1, "missing/duplicate raw token")
    tokens = (passes[0] + " " + covers[0]).split()
    require(all(re.fullmatch(r"\w+=\d+", token) for token in tokens), "malformed ledger")
    pairs = [token.split("=") for token in tokens]
    require(len({key for key, _ in pairs}) == len(pairs), "duplicate ledger field")
    ledger = {key: int(value) for key, value in pairs}
    require(ledger == EXPECTED, "directed ledger changed")
    require(ledger["postread_reads"] - ledger["preread_reads"]
            == ledger["postread_bank_req"] == ledger["postread_bank_rsp"], "causal conservation")
    matches = re.findall(r"sva_postread\.(cp_\w+), \d+ attempts, (\d+) match", sim)
    require(len(matches) == 3 and {key: int(value) for key, value in matches} == {
        "cp_real_postread_request": 552, "cp_real_postread_response": 1932,
        "cp_postread_commit_terminal": 4}, "missing/duplicate/changed SVA cover")
    require(sim.count('$finish called from file "' + TB_REL + '", line 683.') == 1
            and "Time: 10330500 ps" in sim and "V C S   S i m u l a t i o n   R e p o r t" in sim,
            "incomplete simulation finish")
    return {"raw_log_ledger": ledger, "sva_matches": dict((k, int(v)) for k, v in matches),
            "reviewed_warnings": {"tb_context_KUAI": 26, "Rocky_8p10_LNX_OS_VERUN": 1}}


def validate_inputs(contract):
    for relative, expected in contract["pinned_files"].items():
        require(sha(REPO / relative) == expected, "pinned input drift: " + relative)
    for relative in contract["sealed_directories"]:
        verify_seal(REPO / relative)
    original = json.loads((HW / contract["m2213_contract"]).read_text())
    for relative, expected in original["source_inventory"].items():
        require(sha(REPO / relative) == expected, "M2213 source drift: " + relative)
    require(sha(Path(sys.executable).resolve()) == contract["python_sha256"], "Python identity drift")
    m2216 = json.loads((HW / contract["m2216_review"]).read_text())
    require(m2216["status"] == "PASS_M2216_FAILURE_DIAGNOSIS__NEW_IDENTITY_PARSE_ONLY_SUCCESSOR_SUPPORTED"
            and m2216["recovery_authorization"]["new_identity_parse_only_successor_supported"],
            "M2216 recovery support missing")
    require((Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
            == "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n", "original failure altered")


def authorize(review, contract_sha):
    require(review["status"] == "PASS_M2230_M2229_PARSE_ONLY_SOURCE__M2231_CPU_PARSE_AUTHORIZED", "M2230 status")
    require(review["score_over_100"] >= 95 and review["severity_counts"]["p0"] == 0
            and review["severity_counts"]["p1"] == 0, "M2230 unresolved decisive finding")
    require(review["identity"]["source_contract_sha256"] == contract_sha
            and review["identity"]["parser_runner_sha256"] == sha(Path(__file__).resolve()), "M2230 source identity")
    require(review["authorization"] == {"cpu_parse_runs": 1, "license_queries": 0,
            "eda_runs": 0, "gpu_runs": 0, "automatic_retry": False}, "M2230 execution budget")


def run_once(contract, contract_sha, review_sha):
    require(sha(SOURCE_REVIEW / "review.json") == review_sha, "caller-pinned M2230 mismatch")
    verify_seal(SOURCE_REVIEW)
    authorize(json.loads((SOURCE_REVIEW / "review.json").read_text()), contract_sha)
    require(not any(p.exists() or p.is_symlink() for p in (RESULT, ATTEMPT, LOCK)), "M2231 identity consumed/busy")
    work = HW / ("results/.m2231_causal_parse_only_work." + str(os.getpid()))
    require(not work.exists(), "M2231 work exists")
    LOCK.mkdir()
    try:
        ATTEMPT.mkdir()
        (ATTEMPT / "ATTEMPT_CONSUMED.txt").write_text("M2231_CPU_PARSE_ATTEMPT_CONSUMED\nretry=false\neda_runs=0\n")
        seal_new(ATTEMPT)
        work.mkdir()
        try:
            with (work / "parser.stdout.log").open("w") as out, (work / "parser.stderr.log").open("w") as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        measured = parse_raw((Q / "vcs_compile.log").read_text(), (Q / "simv.log").read_text(),
                                             (Q / "simv.rc").read_text(), (REPO / TB_REL).read_text())
                        validate_inputs(contract)
                        print(STATUS)
                    except BaseException:
                        traceback.print_exc()
                        raise
            receipt = {"schema": "m2231_m2215_causal_parse_only_successor_r1_v1", "status": STATUS,
                "producer": "M2215 raw VCS, original postprocessing FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                "recovery": "M2229 parse-only; new receipt, no VCS retry or upstream mutation",
                "source_contract_sha256": contract_sha, "m2230_review_sha256": review_sha,
                "pinned_input_files": contract["pinned_files"], **measured,
                "directed_request_reduction": 0.75,
                "claim_boundary": {"raw_directed_function_and_causality_pending_m2232": True,
                    "rtl_function_admitted": False, "component_speedup": False, "full_population_speedup": False,
                    "system_speedup": False, "same_area": False, "area": False, "timing": False,
                    "hold": False, "power": False, "energy": False, "paper_citable": False, "headline": False},
                "execution": {"cpu_parse_runs": 1, "license_queries": 0, "eda_runs": 0, "gpu_runs": 0, "retry": False}}
            (work / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            (work / "RUN_COMPLETE.txt").write_text(STATUS + "\n")
            seal_new(work)
            require(not RESULT.exists(), "M2231 result appeared during parse")
            work.rename(RESULT)
        except BaseException:
            if work.exists():
                (work / "RUN_FAILED.txt").write_text("FAILED_M2231_PARSE_ONLY_DO_NOT_CITE\nretry=false\n")
                seal_new(work)
                work.rename(Path(str(RESULT) + ".failed." + str(os.getpid()) + ".quarantine"))
            raise
    finally:
        LOCK.rmdir()
    print(STATUS)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--contract-sha256")
    ap.add_argument("--source-review-sha256")
    args = ap.parse_args()
    contract = json.loads(CONTRACT.read_text())
    if args.execute:
        require(args.contract_sha256 and args.source_review_sha256, "execution requires caller-pinned source contract and M2230")
        require(sha(CONTRACT) == args.contract_sha256, "caller-pinned contract mismatch")
    validate_inputs(contract)
    if args.static:
        print("PASS_M2229_STATIC_IDENTITIES__NO_M2231_ATTEMPT_CREATED")
    else:
        run_once(contract, args.contract_sha256, args.source_review_sha256)


if __name__ == "__main__":
    main()

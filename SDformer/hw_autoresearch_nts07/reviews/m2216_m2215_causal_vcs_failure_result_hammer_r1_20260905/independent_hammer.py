#!/usr/bin/env python3
"""Read-only audit of frozen M2215 quarantine; invokes no EDA or license tool."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
Q = HW / "results/m2215_m2213_preread_postread_causal_directed_vcs_r1_20260904.failed_or_incomplete.3812622.quarantine"
A = HW / "results/.m2215_m2213_preread_postread_causal_vcs_attempt_consumed"
S = HW / "reviews/m2214_m2213_preread_postread_causal_ablation_source_hammer_r1_20260904"
C = HW / "contracts/m2213_preread_postread_causal_ablation_source_contract_r1_20260904.json"
P = HW / "system_simulator/scripts/parse_m2215_m2213_preread_postread_causal_directed_vcs.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(directory):
    assert directory.is_dir() and not directory.is_symlink()
    paths = list(directory.rglob("*"))
    assert not any(p.is_symlink() for p in paths)
    actual = {str(p.relative_to(directory)) for p in paths if p.is_file()}
    entries = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        assert name not in entries and not Path(name).is_absolute()
        assert ".." not in Path(name).parts
        assert sha(directory / name) == digest, name
        entries[name] = digest
    assert actual == set(entries) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    seal = (directory / "SHA256SUMS.seal.sha256").read_text().split()
    assert seal == [sha(directory / "SHA256SUMS"), "SHA256SUMS"]
    return {"files": len(entries), "sha256sums_sha256": sha(directory / "SHA256SUMS"),
            "outer_seal_sha256": sha(directory / "SHA256SUMS.seal.sha256")}


def main():
    seals = {"quarantine": sealed(Q), "attempt": sealed(A), "m2214": sealed(S)}
    contract = json.loads(C.read_text())
    for relative, expected in contract["source_inventory"].items():
        assert sha(REPO / relative) == expected, relative
    review = json.loads((S / "review.json").read_text())
    assert review["status"] == "PASS_M2214_M2213_SOURCE_HAMMER__M2215_ONE_SHOT_VCS_AUTHORIZED"
    assert review["identity"]["contract_sha256"] == sha(C)
    assert review["identity"]["parser_sha256"] == sha(P)
    assert review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
    assert (Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() == "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n"
    assert "status=M2215_ATTEMPT_CONSUMED" in (A / "ATTEMPT_CONSUMED.txt").read_text()
    assert not (HW / "results/m2215_m2213_preread_postread_causal_directed_vcs_r1_20260904").exists()
    assert not (Q / "receipt.json").exists() and not (Q / "RUN_COMPLETE.txt").exists()
    assert not (Q / "parser.log").read_bytes()
    comp = (Q / "vcs_compile.log").read_text()
    sim = (Q / "simv.log").read_text()
    assert (Q / "simv.rc").read_text().strip() == "0"
    assert "Chronologic VCS (TM)" in comp and "Version V-2023.12-SP1_Full64" in comp
    assert "All of 7 modules done" in comp and "simv up to date" in comp
    assert "CPU time:" in comp and "to link" in comp
    assert not re.search(r"Error-|Syntax error|Fatal", comp)
    assert "Chronologic VCS simulator copyright" not in comp
    assert "Chronologic VCS simulator copyright" in sim
    assert "Runtime version V-2023.12-SP1_Full64" in sim
    assert not re.search(r"\$fatal|Error:|assertion failed|timeout|mismatch", sim, re.I) or "golden_mismatches=0" in sim
    clean_sim = sim.replace("golden_mismatches=0", "")
    assert not re.search(r"\$fatal|Error:|assertion failed|timeout|mismatch", clean_sim, re.I)
    tokens = re.findall(r"^RAW_PASS_M2215_M2213_PREREAD_POSTREAD_CAUSAL_DIRECTED (.*)$", sim, re.M)
    covers = re.findall(r"^M2213_COVER (.*)$", sim, re.M)
    assert len(tokens) == len(covers) == 1
    values = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", tokens[0] + " " + covers[0])}
    expected = dict(ordinary_reads=2304, postread_reads=2304, preread_reads=576,
        suppressed_reads=1728, ordinary_cycles=3386, postread_cycles=3386,
        preread_cycles=1119, rows=24, hits_post=18, hits_pre=18,
        real_postread_rows=18, postread_bundle_req=216, postread_bundle_rsp=216,
        postread_bank_req=1728, postread_bank_rsp=1728, identity_rsp=216,
        commits_each=24, products_each=4608, golden_mismatches=0)
    assert values == expected
    assert values["postread_reads"] - values["preread_reads"] == values["postread_bank_req"]
    sva_matches = {name: int(matches) for name, matches in re.findall(
        r"sva_postread\.(cp_\w+), \d+ attempts, (\d+) match", sim)}
    assert sva_matches == {"cp_real_postread_request": 552,
        "cp_real_postread_response": 1932, "cp_postread_commit_terminal": 4}
    assert '$finish called from file "hw_autoresearch_nts07/tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv", line 683.' in sim
    warnings = re.findall(r"Warning-\[(\w+)\]", comp)
    assert set(warnings) == {"KUAI", "LNX_OS_VERUN"}
    locations = re.findall(r"Warning-\[KUAI\] Keyword used as identifier\n([^\n]+), (\d+)\n  '([^']+)'", comp)
    assert len(locations) == warnings.count("KUAI") == 26
    tb_lines = (HW / "tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv").read_text().splitlines()
    for path, line, identifier in locations:
        assert path == "hw_autoresearch_nts07/tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv"
        assert identifier == "context" and re.search(r"\bcontext\b", tb_lines[int(line)-1])
    bad_assert = [n for n in ast.walk(ast.parse(P.read_text())) if isinstance(n, ast.Assert)
        and ast.unparse(n.test) == "'Chronologic VCS simulator copyright' in compile_log"]
    assert len(bad_assert) == 1
    python = Path("/opt/anaconda3/bin/python3.12")
    assert sha(python) == review["identity"]["python_sha256"]
    replay = subprocess.run([str(python), "-B", str(P), "--compile-log", str(Q / "vcs_compile.log"),
        "--sim-log", str(Q / "simv.log"), "--sim-rc", str(Q / "simv.rc"), "--output", "/dev/null"],
        text=True, capture_output=True, check=False)
    assert replay.returncode == 1 and replay.stdout == ""
    assert 'assert "Chronologic VCS simulator copyright" in compile_log' in replay.stderr
    assert "AssertionError" in replay.stderr
    # Recheck every immutable input after the read-only reproduction.
    assert {"quarantine": sealed(Q), "attempt": sealed(A), "m2214": sealed(S)} == seals
    for relative, expected in contract["source_inventory"].items():
        assert sha(REPO / relative) == expected
    print(json.dumps({"status": "PASS_M2216_INDEPENDENT_FAILURE_DIAGNOSIS",
        "seals": seals, "contract_sha256": sha(C), "parser_sha256": sha(P),
        "source_inventory_entries_checked": len(contract["source_inventory"]),
        "raw_log_ledger": values, "sva_matches": sva_matches,
        "warning_counts": {kind: warnings.count(kind) for kind in sorted(set(warnings))},
        "diagnostic_parser_replay": {"rc": replay.returncode, "stdout_empty": True,
            "failure_line": bad_assert[0].lineno, "stderr": replay.stderr},
        "immutable_input_seals_rechecked_after_replay": True,
        "review_execution": {"read_only_parser_replays": 1, "eda_runs": 0,
            "license_queries": 0, "gpu_runs": 0, "git_mutations": 0}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

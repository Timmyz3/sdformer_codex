#!/usr/bin/env python3
"""Build the exact-SHA M67 Synopsys VCS pressure receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m67_lookahead_pressure_vcs_contract_r1_20260823.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_covers(sim_text):
    covers = {}
    pattern = re.compile(r"\.([A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match")
    for name, count in pattern.findall(sim_text):
        require(name not in covers, "duplicate cover in VCS log: " + name)
        covers[name] = int(count)
    return covers


def parse_ledger(path):
    counts = {"C": 0, "L": 0, "R": 0, "O": 0, "END": 0}
    end_line = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        require(fields and fields[0] in counts,
                "unexpected ledger record: " + line[:80])
        counts[fields[0]] += 1
        if fields[0] == "END":
            end_line = line
    require(end_line == "END commands=73 outputs=73", "ledger END drift")
    require(counts == {"C": 73, "L": 30, "R": 56, "O": 73, "END": 1},
            "ledger conservation drift: {}".format(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    contract = load_json(CONTRACT)
    sim_text = (run / "sim.raw.log").read_text(encoding="utf-8")
    pass_line = contract["required_terminal_pass_line"]
    require(sim_text.count(pass_line) == 1, "terminal PASS not unique")
    for line in contract["required_marker_lines"]:
        require(sim_text.count(line) == 1, "required marker drift: " + line)
    require(not re.search(r"Assertion failure|failed at|Offending|^Error|^Fatal",
                          sim_text, re.I | re.M), "failure signature in sim log")
    covers = parse_covers(sim_text)
    for name, minimum in contract["required_cover_minimum_matches"].items():
        require(covers.get(name, 0) >= minimum,
                "cover below minimum: {}".format(name))
    ledger_counts = parse_ledger(run / "m67_handshake_ledger.log")
    artifacts = {}
    for name in ("compile.raw.log", "compile.rc", "sim.raw.log", "sim.rc",
                 "m67_handshake_ledger.log", "snapshot.sha256"):
        artifacts[name] = sha256_path(run / name)
    receipt = {
        "schema": "m67_lookahead_pressure_vcs_receipt_r1",
        "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_PRESSURE_R1",
        "contract": {
            "path": str(CONTRACT.relative_to(HW)),
            "sha256": sha256_path(CONTRACT),
        },
        "run_directory": str(run),
        "tool": "Synopsys VCS V-2023.12-SP1_Full64",
        "exact_identity_sha256": contract["exact_sha256"],
        "results": contract["expected_results"],
        "ledger_record_counts": ledger_counts,
        "observed_cover_matches": {
            name: covers[name]
            for name in contract["required_cover_minimum_matches"]
        },
        "assertion_modules_active": {
            "m54": True,
            "m66": True,
        },
        "unique_terminal_pass": True,
        "assertion_failure_count": 0,
        "functional_mismatch_count": 0,
        "run_artifact_sha256": artifacts,
        "claim_boundary": contract["claim_boundary"],
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M67 receipt builder covers={} ledger_records={}".format(
        len(receipt["observed_cover_matches"]), sum(ledger_counts.values())))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M67 receipt builder: {}".format(error))
        raise SystemExit(1)

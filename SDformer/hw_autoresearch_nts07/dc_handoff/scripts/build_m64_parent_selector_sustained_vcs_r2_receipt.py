#!/usr/bin/env python3
"""Build the additive M64-r2 sustained-throughput VCS receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m64_parent_selector_sustained_vcs_contract_r2_20260823.json"
PASS_RE = re.compile(
    r"^PASS M64 R2 sustained tests=(\d+) inputs=(\d+) outputs=(\d+) "
    r"b2b_accepts=(\d+) full_cycles=(\d+) max_full_run=(\d+) "
    r"full_push_pop=(\d+) source256=(\d+) "
    r"parent_hits=(\d+),(\d+),(\d+),(\d+) ties=(\d+) "
    r"random_stalls=(\d+) output_stalls=(\d+) max_outstanding=(\d+) "
    r"valid_low=(\d+) mismatches=(\d+)$", re.MULTILINE)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def cover_matches(text, name):
    pattern = re.compile(
        r"r2_sva\.{}.*,\s*\d+ attempts,\s*(\d+) match".format(
            re.escape(name)))
    hits = [int(value) for value in pattern.findall(text)]
    if len(hits) != 1:
        raise ValueError("cover missing/duplicated: {}".format(name))
    return hits[0]


def resolve_identity_path(relative):
    if relative == "vcs_launcher_binary":
        return Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
    return HW / relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing M64-r2 receipt overwrite")
    run = args.run.resolve()
    contract = load_json(CONTRACT)
    sim_text = (run / "sim.raw.log").read_text(encoding="utf-8")
    compile_text = (run / "compile.raw.log").read_text(encoding="utf-8")
    matches = PASS_RE.findall(sim_text)
    if len(matches) != 1:
        raise ValueError("unique M64-r2 terminal PASS line missing")
    values = [int(value) for value in matches[0]]
    keys = [
        "tests", "accepted_inputs", "accepted_outputs",
        "back_to_back_input_accepts", "full_throughput_cycles",
        "maximum_full_throughput_run", "pipeline_full_push_pop_cycles",
        "source_count_256_outputs", "parent_zero", "parent_left",
        "parent_up", "parent_previous", "forced_tie_accepts",
        "random_output_stall_cycles", "total_output_stall_cycles",
        "maximum_outstanding", "sustained_valid_low_cycles",
        "functional_mismatches",
    ]
    results = dict(zip(keys, values))
    results["parent_hits"] = {
        "zero": results.pop("parent_zero"),
        "left": results.pop("parent_left"),
        "up": results.pop("parent_up"),
        "previous_timestep": results.pop("parent_previous"),
    }
    expected = contract["expected"]
    if results["tests"] != expected["tests"] or (
            results["accepted_inputs"] != expected["tests"]) or (
            results["accepted_outputs"] != expected["tests"]):
        raise ValueError("M64-r2 input/output conservation failure")
    if (run / "compile.rc").read_text().strip() != "0" or (
            run / "sim.rc").read_text().strip() != "0":
        raise ValueError("M64-r2 nonzero VCS rc")
    if "V-2023.12-SP1_Full64" not in compile_text or (
            "V-2023.12-SP1_Full64" not in sim_text):
        raise ValueError("M64-r2 VCS identity missing")
    if re.search(r"Warning-\[|Error-\[|^Error", compile_text,
                 re.IGNORECASE | re.MULTILINE):
        raise ValueError("M64-r2 compile diagnostic signature")
    if re.search(r"Assertion failure|failed at|Offending|\bFatal\b|\bError-\[",
                 sim_text, re.IGNORECASE):
        raise ValueError("M64-r2 simulation failure signature")

    identities = {}
    for relative, expected_sha in contract["exact_sha256"].items():
        path = resolve_identity_path(relative)
        actual = sha256_path(path)
        if actual != expected_sha:
            raise ValueError("identity drift: {}".format(relative))
        identities[relative] = actual

    covers = {}
    for name, minimum in contract["required_cover_minimum_matches"].items():
        observed = cover_matches(sim_text, name)
        if observed < minimum:
            raise ValueError("cover below minimum: {}".format(name))
        covers[name] = observed

    artifact_names = [
        "preflight.raw.log", "snapshot.sha256", "compile.command.txt",
        "compile.raw.log", "compile.rc", "sim.command.txt", "sim.raw.log",
        "sim.rc", "simv",
    ]
    artifacts = {}
    for name in artifact_names:
        path = run / name
        if not path.is_file():
            raise ValueError("run artifact missing: {}".format(name))
        artifacts[name] = sha256_path(path)

    receipt = {
        "schema": "m64_parent_selector_sustained_vcs_receipt_r2",
        "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_SUSTAINED_R2",
        "contract": {
            "path": str(CONTRACT.relative_to(HW)),
            "sha256": sha256_path(CONTRACT),
        },
        "run_directory": str(run),
        "tool": "Synopsys VCS V-2023.12-SP1_Full64",
        "exact_identity_sha256": identities,
        "run_artifact_sha256": artifacts,
        "results": results,
        "observed_cover_matches": covers,
        "assertion_module_active": sim_text.splitlines().count(
            contract["required_module_active_line"]) == 1,
        "unique_terminal_pass": len(matches) == 1,
        "assertion_failure_count": 0,
        "claim_boundary": contract["claim_boundary"],
        "admission": {
            "sustained_directed_vcs_sva_admitted": True,
            "system_speedup_admitted": False,
            "headline_admitted": False,
            "ppa_admitted": False,
            "power_energy_admitted": False,
            "all10_or_full_network_admitted": False,
            "random_or_formal_protocol_proof_admitted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M64-r2 receipt builder inputs={} outputs={} b2b={} full_run={}".format(
        results["accepted_inputs"], results["accepted_outputs"],
        results["back_to_back_input_accepts"],
        results["maximum_full_throughput_run"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M64-r2 receipt builder: {}".format(error))
        raise SystemExit(1)

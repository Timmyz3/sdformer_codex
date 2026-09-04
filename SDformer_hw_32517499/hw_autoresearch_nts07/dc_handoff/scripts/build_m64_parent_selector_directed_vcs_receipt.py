#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m64_online_adaptive_parent_selector_directed_vcs_contract_r1_20260823.json"
FILES = {
    "rtl": HW / "rtl_m64/qfit_adaptive_parent_selector_p256.sv",
    "sva": HW / "verif_m64/qfit_adaptive_parent_selector_p256_assertions.sv",
    "tb": HW / "tb_m64/tb_qfit_adaptive_parent_selector_p256.sv",
    "filelist": HW / "dc_handoff/filelists/date_m64_parent_selector_directed_vcs.f",
    "runner": HW / "dc_handoff/scripts/run_vcs_m64_parent_selector_directed_sva.sh",
    "builder": Path(__file__).resolve(),
    "validator": HW / "dc_handoff/scripts/validate_m64_parent_selector_directed_vcs.py",
}
PASS_RE = re.compile(
    r"^PASS M64 selector tests=(\d+) outputs=(\d+) "
    r"parent_hits=(\d+),(\d+),(\d+),(\d+) stalls=(\d+)$", re.M)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contract = strict_json(CONTRACT)
    compile_text = (args.run / "compile.raw.log").read_text()
    sim_text = (args.run / "sim.raw.log").read_text()
    match = PASS_RE.search(sim_text)
    if match is None:
        raise ValueError("M64 terminal PASS line missing")
    tests, outputs, hit0, hit1, hit2, hit3, stalls = map(int, match.groups())
    if [tests, outputs, hit0, hit1, hit2, hit3, stalls] != [
            4096, 4096, 1271, 974, 988, 863, 1074]:
        raise ValueError("M64 terminal counters drift")
    if "Error-[" in compile_text or "Warning-[" in compile_text:
        raise ValueError("M64 compile diagnostic signature present")
    if re.search(r"failed at|Offending|Error|Fatal", sim_text, re.I):
        raise ValueError("M64 simulation failure signature present")
    if (args.run / "compile.rc").read_text().strip() != "0" or (
            args.run / "sim.rc").read_text().strip() != "0":
        raise ValueError("M64 VCS nonzero rc")
    receipt = {
        "schema": "m64_online_adaptive_parent_selector_directed_vcs_receipt_v1",
        "status": "PASS_EXACT_SHA_DIRECTED_VCS_SVA",
        "identity": dict((name + "_sha256", sha(path))
                         for name, path in sorted(FILES.items())),
        "contract_sha256": sha(CONTRACT),
        "simv_sha256": sha(args.run / "simv"),
        "tool": "Synopsys VCS V-2023.12-SP1_Full64",
        "results": {
            "tests": tests,
            "outputs": outputs,
            "parent_hits": {
                "zero": hit0, "left": hit1, "up": hit2,
                "previous_timestep": hit3,
            },
            "output_stall_cycles": stalls,
            "functional_mismatches": 0,
            "assertion_failures": 0,
            "all_four_parent_covers_nonzero": min(hit0, hit1, hit2, hit3) > 0,
        },
        "architecture": contract["architecture"],
        "claim_boundary": contract["claim_boundary"],
        "admission": {
            "directed_vcs_sva_admitted": True,
            "all10_trace_admitted": False,
            "seed_sram_or_scheduler_admitted": False,
            "cycles_or_system_speedup_admitted": False,
            "dc_sta_formality_admitted": False,
            "power_energy_ppa_admitted": False,
            "headline_admitted": False,
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

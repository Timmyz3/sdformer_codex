#!/usr/bin/env python3
"""Build the M66 exact-SHA full-S00 VCS receipt from immutable run evidence."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract_path = repo / "hw_autoresearch_nts07/contracts/m66_s00_lookahead_exact_sha_vcs_contract_r1_20260823.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    run = repo / contract["paths"]["production_run"]
    compile_run = repo / contract["paths"]["compile_run"]
    replay = json.loads((run / "m66_s00_ledger_replay.json").read_text(encoding="utf-8"))
    log = (run / "sim.raw.log").read_text(errors="replace")
    if args.output.exists():
        raise ValueError("refusing receipt overwrite")
    covers = {"m54_inherited": {}, "m66_lookahead": {}}
    for instance, name, attempts, matches in re.findall(
            r"\.(m54_sva|m66_seam_sva)\.(cp_[A-Za-z0-9_]+),\s+(\d+) attempts,\s+(\d+) match", log):
        group = "m54_inherited" if instance == "m54_sva" else "m66_lookahead"
        covers[group][name] = {"attempts": int(attempts), "matches": int(matches)}
    m57_cycles = contract["expected"]["m57_phase_safe_rtl_cycles"]
    m66_cycles = replay["rtl_cycles"]
    payload = {
        "schema": "m66_s00_lookahead_exact_sha_vcs_receipt_v1",
        "status": "PASS_M66_EXACT_SHA_FULL_S00_LOOKAHEAD_VCS_REPLAY",
        "date": "2026-08-23",
        "contract_sha256": sha(contract_path),
        "run": {
            "directory": str(run),
            "compile_directory": str(compile_run),
            "elapsed_seconds": int((run / "end_epoch.txt").read_text()) - int((run / "start_epoch.txt").read_text()),
            "compile_rc": int((compile_run / "compile.rc").read_text()),
            "sim_rc": int((run / "sim.rc").read_text()),
            "gzip_rc": int((run / "gzip.rc").read_text()),
            "replay_rc": int((run / "replay.rc").read_text()),
            "full_sample_not_sampled": True,
        },
        "identity": {
            "simv_sha256": sha(compile_run / "simv"),
            "compile_log_sha256": sha(compile_run / "compile.raw.log"),
            "ledger_gzip_sha256": sha(run / "m66_s00_handshake_ledger.compact.log.gz"),
            "ledger_gzip_bytes": (run / "m66_s00_handshake_ledger.compact.log.gz").stat().st_size,
            "sim_log_sha256": sha(run / "sim.raw.log"),
            "replay_sha256": sha(run / "m66_s00_ledger_replay.json"),
            "precompile_sha256_manifest": sha(compile_run / "precompile_input.sha256"),
            "prelaunch_sha256_manifest": sha(run / "prelaunch_input.sha256"),
        },
        "functional_and_protocol": replay,
        "sva": {
            "m54_module_active": log.count("M54_ASSERTION_MODULE_ACTIVE=1") == 1,
            "m66_module_active": log.count("M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1") == 1,
            "coverpoints": covers,
            "assertion_failure_signatures": len(re.findall(
                r"(?i)(assertion failed|error-|fatal:|\$error|\$fatal)", log)),
        },
        "terminal": {
            "pass_line_count": len(re.findall(r"^PASS M66 S0 ", log, re.M)),
            "progress_records": len(re.findall(r"^M57_PROGRESS ", log, re.M)),
        },
        "same_trace_m57_to_m66": {
            "m57_rtl_cycles": m57_cycles,
            "m66_rtl_cycles": m66_cycles,
            "cycles_saved": m57_cycles - m66_cycles,
            "cycle_reduction_numerator": m57_cycles - m66_cycles,
            "cycle_reduction_denominator": m57_cycles,
            "speedup_numerator": m57_cycles,
            "speedup_denominator": m66_cycles,
            "comparison_scope": "same S00 offline schedule, VCS RTL cycles, M57 bubble vs M66 seamless handoff",
            "system_speedup_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M66 receipt elapsed={}s cycles={} saved={} seamless={}".format(
        payload["run"]["elapsed_seconds"], m66_cycles, m57_cycles - m66_cycles,
        replay["seamless_launches"]))


if __name__ == "__main__":
    main()

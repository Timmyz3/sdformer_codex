#!/usr/bin/env python3
"""Python-3.6-compatible additive release validator for the M66 r1 evidence."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicate_object,
                      parse_constant=reject_constant)


def read_sha_manifest(path):
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        require(match is not None, "bad SHA manifest line {} in {}".format(line_number, path))
        target = match.group(2)
        require(target not in result, "duplicate SHA manifest path: " + target)
        result[target] = match.group(1)
    return result


def require_exact_manifest(manifest_path, expected_paths):
    observed = read_sha_manifest(manifest_path)
    expected = {str(path): sha(path) for path in expected_paths}
    require(observed == expected, "manifest path/SHA set drift: " + str(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract_path = repo / "hw_autoresearch_nts07/contracts/m66_s00_lookahead_exact_sha_vcs_contract_r1_20260823.json"
    contract = load_json(contract_path)
    receipt = load_json(args.receipt)
    run = repo / contract["paths"]["production_run"]
    compile_run = repo / contract["paths"]["compile_run"]
    expected = contract["expected"]
    paths = {
        "lookahead_core_rtl": repo / "hw_autoresearch_nts07/rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv",
        "schedule_bridge_rtl": repo / "hw_autoresearch_nts07/rtl_m66/qfit_m66_m53_schedule_bridge_lookahead.sv",
        "inherited_assertions": repo / "hw_autoresearch_nts07/verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
        "lookahead_assertions": repo / "hw_autoresearch_nts07/verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv",
        "testbench": repo / "hw_autoresearch_nts07/tb_m66/tb_m66_m53_schedule_bridge_lookahead.sv",
        "filelist": repo / "hw_autoresearch_nts07/dc_handoff/filelists/date_m66_m53_schedule_bridge_lookahead_vcs.f",
        "schedule_stream": repo / "hw_autoresearch_nts07/dc_handoff/runs/m57_diagnostics_20260823/s00_sim_r2/input.bin",
        "schedule_manifest": repo / "hw_autoresearch_nts07/results/m57_h67_k4c16_temporal_vcs_r1_20260823/m57_s00_schedule_manifest.json",
        "ledger_replayer": repo / "hw_autoresearch_nts07/verif_m66/replay_m66_handshake_ledger.py",
        "receipt_builder": repo / "hw_autoresearch_nts07/verif_m66/build_m66_s00_exact_sha_receipt.py",
        "receipt_validator": repo / "hw_autoresearch_nts07/verif_m66/validate_m66_s00_exact_sha_receipt.py",
        "m57_reference_replay": repo / "hw_autoresearch_nts07/results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823/m57_s00_ledger_replay.json",
        "m57_reference_receipt": repo / "hw_autoresearch_nts07/results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823/m57_s00_phase_safe_exact_sha_vcs_receipt.json",
        "vcs_launcher": Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"),
    }
    require(receipt["contract_sha256"] == sha(contract_path), "contract identity drift")
    require(receipt["status"] == "PASS_M66_EXACT_SHA_FULL_S00_LOOKAHEAD_VCS_REPLAY", "bad receipt status")
    for name, path in paths.items():
        require(path.is_file(), name + " missing")
        require(sha(path) == contract["exact_sha256"][name], name + " SHA drift")
    require_exact_manifest(compile_run / "precompile_input.sha256",
                           [contract_path] + list(paths.values()))
    for name in ("compile.rc",):
        require(int((compile_run / name).read_text()) == 0, name + " nonzero")
    for name in ("sim.rc", "gzip.rc", "replay.rc"):
        require(int((run / name).read_text()) == 0, name + " nonzero")
    require(receipt["identity"]["simv_sha256"] == sha(compile_run / "simv"), "simv SHA drift")
    require(receipt["identity"]["compile_log_sha256"] == sha(compile_run / "compile.raw.log"), "compile log SHA drift")
    require(receipt["identity"]["ledger_gzip_sha256"] == sha(run / "m66_s00_handshake_ledger.compact.log.gz"), "ledger SHA drift")
    require(receipt["identity"]["sim_log_sha256"] == sha(run / "sim.raw.log"), "sim log SHA drift")
    require(receipt["identity"]["replay_sha256"] == sha(run / "m66_s00_ledger_replay.json"), "replay SHA drift")
    compile_command = "{} -full64 -sverilog -assert svaext -f {} -top tb_m66_m53_schedule_bridge_lookahead -Mdir={} -o {} -l {}".format(
        paths["vcs_launcher"], paths["filelist"], compile_run / "csrc",
        compile_run / "simv", compile_run / "compile.raw.log")
    require((compile_run / "compile.command.txt").read_text(encoding="utf-8") == compile_command + "\n",
            "compile command drift")
    sim_command = "{} +STREAM={} +LEDGER={} -assert report".format(
        compile_run / "simv", paths["schedule_stream"], run / "ledger.fifo")
    require((run / "sim.command.txt").read_text(encoding="utf-8") == sim_command + "\n",
            "simulation command drift")
    require_exact_manifest(run / "prelaunch_input.sha256", [
        contract_path, compile_run / "precompile_input.sha256", compile_run / "simv",
        compile_run / "compile.raw.log", compile_run / "compile.command.txt",
        paths["schedule_stream"], paths["schedule_manifest"], paths["ledger_replayer"],
        paths["receipt_builder"], paths["receipt_validator"],
    ])
    compile_log = (compile_run / "compile.raw.log").read_text(errors="replace")
    for source in ("qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv",
                   "qfit_m66_m53_schedule_bridge_lookahead.sv",
                   "qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
                   "qfit_k4_parent_delta_lookahead_assertions.sv",
                   "tb_m66_m53_schedule_bridge_lookahead.sv"):
        require("Parsing design file" in compile_log and source in compile_log,
                "compile log missing " + source)
    log = (run / "sim.raw.log").read_text(errors="replace")
    require(len(re.findall(r"^PASS M66 S0 ", log, re.M)) == 1, "missing unique M66 PASS")
    require(log.count("M54_ASSERTION_MODULE_ACTIVE=1") == 1, "inherited SVA inactive/duplicate")
    require(log.count("M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1") == 1, "lookahead SVA inactive/duplicate")
    require(not re.search(r"(?i)(assertion failed|error-|fatal:|\$error|\$fatal)", log),
            "assertion/fatal signature in simulation log")
    replay = receipt["functional_and_protocol"]
    stored_replay = load_json(run / "m66_s00_ledger_replay.json")
    require(replay == stored_replay, "receipt replay payload drift")
    with tempfile.TemporaryDirectory(prefix="m66_replay_validate_") as temporary_directory:
        rerun_path = Path(temporary_directory) / "replay.json"
        rerun = subprocess.run([
            sys.executable, str(paths["ledger_replayer"]),
            "--ledger", str(run / "m66_s00_handshake_ledger.compact.log.gz"),
            "--schedule-manifest", str(paths["schedule_manifest"]),
            "--output", str(rerun_path),
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        require(rerun.returncode == 0, "locked ledger replay rerun failed: " + rerun.stdout)
        require(load_json(rerun_path) == stored_replay, "locked ledger replay rerun differs")
    require(replay["identity"]["ledger_file_sha256"] ==
            sha(run / "m66_s00_handshake_ledger.compact.log.gz"), "replay ledger file SHA drift")
    require(replay["identity"]["schedule_manifest_sha256"] == sha(paths["schedule_manifest"]),
            "replay schedule-manifest SHA drift")
    require(replay["accepted_requests"] == replay["accepted_responses"] == expected["source_issue_cycles"],
            "request/response total drift")
    require(replay["accepted_outputs"] == expected["descriptor_commands"], "output total drift")
    require(replay["functional_mismatch_count"] == 0 and replay["metadata_fifo_final_occupancy"] == 0,
            "functional/FIFO failure")
    require(replay["rtl_cycles"] == expected["m66_rtl_cycles"] and
            replay["m53_transaction_model_cycles"] == expected["m53_transaction_model_cycles"] and
            replay["rtl_minus_m53_transaction_cycles"] == expected["m66_minus_model_cycles"] and
            replay["seamless_launches"] == expected["seamless_launches"],
            "M66 cycle/seam total drift")
    require(replay["event_lines"] > expected["source_issue_cycles"], "event ledger incomplete")
    phase = replay["launch_phase"]
    require(phase["prelaunch_artificial_bubbles"] == 0 and
            phase["direct_groups"] + phase["aligned_groups"] == expected["fusion_groups"],
            "launch phase conservation failure")
    comparison = receipt["same_trace_m57_to_m66"]
    require(comparison["m57_rtl_cycles"] == expected["m57_phase_safe_rtl_cycles"] and
            comparison["m66_rtl_cycles"] == expected["m66_rtl_cycles"] and
            comparison["cycles_saved"] == expected["cycles_saved"] and
            comparison["speedup_numerator"] > comparison["speedup_denominator"],
            "same-trace speed comparison drift")
    covers = receipt["sva"]["coverpoints"]["m66_lookahead"]
    required_positive_covers = ("cp_seam_k1", "cp_seam_k2", "cp_seam_k3", "cp_seam_k4",
                                "cp_zero_next_waits", "cp_seam_with_completion_push",
                                "cp_seam_with_output_accept")
    for name in required_positive_covers:
        require(name in covers and covers[name]["attempts"] > 0 and covers[name]["matches"] > 0,
                name + " not covered")
    require(receipt["sva"]["assertion_failure_signatures"] == 0, "receipt saw assertion failures")
    require(receipt["claim_boundary"]["system_speedup_admitted"] is False and
            receipt["claim_boundary"]["paper_ppa_ready"] is False and
            receipt["claim_boundary"]["power_or_energy_admitted"] is False,
            "claim boundary widened")
    print("PASS M66 full-S00 exact-SHA VCS receipt validator")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M66 validator: {}".format(error))
        raise SystemExit(1)


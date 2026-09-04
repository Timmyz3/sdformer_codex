#!/usr/bin/env python3
"""Build a fail-closed receipt from one completed M35-r5 VCS run."""

import argparse
import hashlib
import json
from pathlib import Path
import re


PASS_PATTERN = re.compile(
    r"^M35_R5_PASS packets=(?P<packets>[0-9]+) "
    r"all_lane_products=(?P<products>[0-9]+) "
    r"valid_products=(?P<valid_products>[0-9]+) "
    r"config_loads=(?P<loads>[0-9]+) "
    r"config_releases=(?P<releases>[0-9]+) "
    r"legal_ids=10 illegal_ids=6 "
    r"illegal_rejections=(?P<illegal_rejections>[0-9]+) "
    r"stalls=(?P<stalls>[0-9]+) "
    r"consecutive_full_rate=(?P<full_rate>[0-9]+) "
    r"masks_all=1 busy_release_rejects=(?P<busy_rejects>[0-9]+) "
    r"idA_pin_perturbations=(?P<id_a>[0-9]+) "
    r"reset_under_stall=(?P<reset_stall>[0-9]+) mismatches=0$",
    re.M,
)


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
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-standard JSON constant")),
    )


def read_status(path, label):
    raw = Path(path).read_text(encoding="ascii")
    require(re.fullmatch(r"0\n", raw) is not None, label + " exit status")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output.resolve()
    require(run.is_dir(), "run directory missing")
    require(output.parent == run and not output.exists(),
            "receipt must be a new direct member of run directory")
    compile_rc = read_status(run / "compile.exit_status", "compile")
    sim_rc = read_status(run / "sim.exit_status", "simulation")
    miter_rc = read_status(run / "miter.exit_status", "miter")
    review_rc = read_status(run / "review_validation.exit_status",
                            "review validation")
    compile_stdout = (run / "compile.stdout.log").read_text(
        encoding="utf-8", errors="replace")
    compile_stderr = (run / "compile.stderr.log").read_text(
        encoding="utf-8", errors="replace")
    sim_stdout = (run / "sim.stdout.log").read_text(
        encoding="utf-8", errors="replace")
    sim_stderr = (run / "sim.stderr.log").read_text(
        encoding="utf-8", errors="replace")
    require("Chronologic VCS" in compile_stdout + compile_stderr,
            "VCS compiler identity missing")
    require("M35_R5_SIMULATOR=Synopsys VCS" in sim_stdout,
            "VCS runtime marker missing")
    require("M35_R5_ASSERTIONS=enabled" in sim_stdout and
            "M35_R5_SVA_BOUND=1" in sim_stdout,
            "SVA marker missing")
    require("M35_R5_RANDOM_SEED=0x4d350105" in sim_stdout,
            "seed marker missing")
    match = PASS_PATTERN.search(sim_stdout)
    require(match is not None, "internal PASS marker missing or malformed")
    metrics = {key: int(value) for key, value in match.groupdict().items()}
    require(metrics["packets"] == 10240 and metrics["products"] == 81920 and
            metrics["loads"] == metrics["releases"] == 10 and
            metrics["illegal_rejections"] == 6 and
            metrics["stalls"] > 0 and metrics["full_rate"] >= 1270 and
            metrics["busy_rejects"] > 0 and metrics["id_a"] > 0 and
            metrics["reset_stall"] == 1,
            "PASS accounting boundary failed")
    failure_text = sim_stdout + "\n" + sim_stderr
    require(re.search(r"assertion[^\n]*(fail|error)|offending.*assert",
                      failure_text, re.I) is None,
            "assertion failure signature found")
    cover_matches = [int(value) for value in re.findall(
        r", [0-9]+ attempts, ([0-9]+) match", sim_stdout)]
    require(len(cover_matches) == 19 and min(cover_matches) > 0,
            "SVA cover population/match drift")
    miter = read_json(run / "m35_r5_trace_miter.json")
    require(miter.get("status") ==
            "PASS_ACTUAL_VCS_HANDSHAKE_TRACE_ZERO_MISMATCH",
            "trace miter status drift")
    product = miter["product_miter"]
    require(product["packets_mitered"] == 10240 and
            product["actual_dut_signed56_products_mitered"] == 81920 and
            product["mismatches"] == 0 and
            product["term_sign_miswire_sensitive"] is True,
            "trace miter accounting drift")
    ids = miter["descriptor_id_execution"]
    require(ids["descriptor_ids_executed"] == list(range(16)) and
            ids["illegal_ids_protocol_error"] == list(range(10, 16)) and
            ids["hex_A_alias_rejected"] is True,
            "descriptor execution population drift")
    evidence_names = [
        "input_sha256.txt", "review_validation.stdout.log",
        "review_validation.stderr.log", "review_validation.exit_status",
        "tool_version.stdout.log", "tool_version.stderr.log",
        "compile.stdout.log", "compile.stderr.log", "compile.exit_status",
        "sim.stdout.log", "sim.stderr.log", "sim.exit_status",
        "handshake_trace.csv", "miter.stdout.log", "miter.stderr.log",
        "miter.exit_status", "m35_r5_trace_miter.json",
    ]
    evidence = {}
    for name in evidence_names:
        path = run / name
        require(path.is_file(), "missing evidence member: " + name)
        evidence[name] = sha256(path)
    receipt = {
        "schema": "m35_r5_exact_sha_vcs_receipt_v1",
        "status": "PASS_EXACT_SHA_VCS_SVA_AND_ACTUAL_OUTPUT_MITER",
        "run_directory": str(run),
        "exact_source": {
            "m35_r4_rtl_sha256":
                "84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854",
            "m35_r4_independent_review_sha256":
                "8b0978b3158d780a0d5acee4ac0a780c32349e1dd45c1722f1421cb01b86fb6f",
            "m35_r4_independent_validator_sha256":
                "305f7ff80090fcd6fd2a957e4a3f07d8b0c53219c392719e243973942674d2e8",
        },
        "exit_status": {
            "review_validation": review_rc,
            "vcs_compile": compile_rc,
            "vcs_simulation": sim_rc,
            "independent_trace_miter": miter_rc,
        },
        "vcs_metrics": {
            "packets": metrics["packets"],
            "actual_dut_signed56_products_checked_in_tb": metrics["products"],
            "valid_mask_products": metrics["valid_products"],
            "legal_descriptor_ids": 10,
            "illegal_descriptor_ids": 6,
            "illegal_id_protocol_error_rejections":
                metrics["illegal_rejections"],
            "output_stall_cycles": metrics["stalls"],
            "consecutive_full_rate_transitions": metrics["full_rate"],
            "busy_release_rejections": metrics["busy_rejects"],
            "live_hex_A_pin_perturbations": metrics["id_a"],
            "reset_under_stall_events": metrics["reset_stall"],
            "tb_mismatches": 0,
            "sva_cover_properties": len(cover_matches),
            "sva_cover_minimum_matches": min(cover_matches),
            "sva_failure_signatures": 0,
        },
        "independent_trace_miter": product,
        "descriptor_id_execution": ids,
        "evidence_sha256": evidence,
        "claim_boundary": {
            "permitted": "Exact reviewed RTL source Synopsys VCS/SVA execution, all 16 descriptor IDs, ready/valid protocol stress, and actual signed56 DUT output trace miter.",
            "forbidden": "DC, STA, Formality, PPA, power, energy, integrated Local/Motion speedup, accuracy, external comparison, DATE headline, or best-paper claim."
        },
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M35-r5 VCS receipt")


if __name__ == "__main__":
    main()

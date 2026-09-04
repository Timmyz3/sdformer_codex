#!/usr/bin/env python3
"""Fail-closed preflight/full validator for additive M64-r2 VCS evidence."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m64_parent_selector_sustained_vcs_contract_r2_20260823.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def resolve_identity_path(relative):
    if relative == "vcs_launcher_binary":
        return Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
    return HW / relative


def validate_contract(contract):
    require(contract["schema"] ==
            "m64_parent_selector_sustained_vcs_contract_r2",
            "contract schema drift")
    require(contract["status"] == "FROZEN_EXACT_SHA_BEFORE_OFFICIAL_RUN",
            "contract status drift")
    require(contract["tool_policy"] == {
        "hdl_simulator": "Synopsys VCS V-2023.12-SP1 only",
        "open_source_hdl_tools_allowed": False,
        "dc_sta_formality_admitted": False,
    }, "tool policy drift")
    expected = contract["expected"]
    require(expected == {
        "tests": 2048,
        "minimum_back_to_back_input_accepts": 32,
        "minimum_full_throughput_cycles": 32,
        "minimum_maximum_full_throughput_run": 32,
        "minimum_pipeline_full_push_pop_cycles": 32,
        "minimum_source_count_256_outputs": 1,
        "forced_tie_accepts": 3,
        "minimum_random_output_stall_cycles": 1,
        "maximum_outstanding": 2,
        "sustained_valid_low_cycles": 0,
        "functional_mismatches": 0,
    }, "expected metric gate drift")
    require(contract["campaign"]["runtime_randomization"] is False and
            contract["campaign"]["sustained_valid"] is True and
            contract["campaign"]["all_output_oracle"] is True and
            contract["campaign"]["random_backpressure_seed_hex"] ==
            "0x64b25e11", "campaign contract drift")
    claim = contract["claim_boundary"]
    for key in ("system_speedup_admitted", "headline_admitted",
                "ppa_admitted", "power_energy_admitted",
                "all10_or_full_network_admitted",
                "random_or_formal_protocol_proof_admitted"):
        require(claim[key] is False, "claim promotion: {}".format(key))
    require(claim["sustained_directed_vcs_sva_admitted"] is True,
            "directed admission missing")
    require(len(contract["exact_sha256"]) == 11,
            "exact identity set size drift")
    require(len(contract["required_cover_minimum_matches"]) == 11,
            "cover requirement set size drift")


def validate_exact_identity(contract):
    observed = {}
    for relative, expected in contract["exact_sha256"].items():
        path = resolve_identity_path(relative)
        require(path.is_file(), "identity path missing: {}".format(relative))
        actual = sha256_path(path)
        require(actual == expected,
                "identity SHA drift {} observed={} expected={}".format(
                    relative, actual, expected))
        observed[relative] = actual
    return observed


def validate_snapshot(run):
    manifest = run / "snapshot.sha256"
    require(manifest.is_file(), "snapshot manifest missing")
    snapshot = run / "snapshot"
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip(" *")
        target = snapshot / relative
        require(target.is_file(), "snapshot file missing: {}".format(relative))
        require(sha256_path(target) == expected,
                "snapshot SHA drift: {}".format(relative))
        count += 1
    require(count == 11, "snapshot entry count drift")
    return count


def validate_receipt(receipt, contract, run):
    require(receipt["schema"] ==
            "m64_parent_selector_sustained_vcs_receipt_r2" and
            receipt["status"] ==
            "PASS_EXACT_SHA_SYNOPSYS_VCS_SUSTAINED_R2",
            "receipt terminal state drift")
    require(receipt["contract"]["sha256"] == sha256_path(CONTRACT),
            "receipt contract binding drift")
    require(Path(receipt["run_directory"]).resolve() == run.resolve(),
            "receipt run-directory drift")
    require(receipt["tool"] == "Synopsys VCS V-2023.12-SP1_Full64",
            "receipt tool drift")
    require(receipt["exact_identity_sha256"] == contract["exact_sha256"],
            "receipt identity ledger drift")
    results = receipt["results"]
    expected = contract["expected"]
    require(results["tests"] == expected["tests"] and
            results["accepted_inputs"] == expected["tests"] and
            results["accepted_outputs"] == expected["tests"],
            "receipt transaction conservation failure")
    require(results["back_to_back_input_accepts"] >=
            expected["minimum_back_to_back_input_accepts"],
            "back-to-back input acceptance below minimum")
    require(results["full_throughput_cycles"] >=
            expected["minimum_full_throughput_cycles"] and
            results["maximum_full_throughput_run"] >=
            expected["minimum_maximum_full_throughput_run"],
            "full-throughput interval below minimum")
    require(results["pipeline_full_push_pop_cycles"] >=
            expected["minimum_pipeline_full_push_pop_cycles"],
            "pipeline-full push+pop below minimum")
    require(results["source_count_256_outputs"] >=
            expected["minimum_source_count_256_outputs"],
            "source_count=256 output missing")
    require(results["forced_tie_accepts"] == expected["forced_tie_accepts"],
            "forced tie count drift")
    require(results["random_output_stall_cycles"] >=
            expected["minimum_random_output_stall_cycles"],
            "random output backpressure missing")
    require(results["maximum_outstanding"] == expected["maximum_outstanding"],
            "pipeline occupancy maximum drift")
    require(results["sustained_valid_low_cycles"] == 0 and
            results["functional_mismatches"] == 0,
            "sustained-valid/oracle failure")
    require(min(results["parent_hits"].values()) > 0,
            "one or more parent classes absent")
    require(receipt["assertion_module_active"] is True and
            receipt["unique_terminal_pass"] is True and
            receipt["assertion_failure_count"] == 0,
            "SVA/PASS receipt failure")
    for name, minimum in contract["required_cover_minimum_matches"].items():
        require(receipt["observed_cover_matches"][name] >= minimum,
                "receipt cover below minimum: {}".format(name))
    require(set(receipt["observed_cover_matches"]) ==
            set(contract["required_cover_minimum_matches"]),
            "receipt cover set drift")
    for name, expected_sha in receipt["run_artifact_sha256"].items():
        require(sha256_path(run / name) == expected_sha,
                "receipt run artifact drift: {}".format(name))
    require(receipt["claim_boundary"] == contract["claim_boundary"],
            "receipt claim boundary drift")
    require(receipt["admission"] == {
        "sustained_directed_vcs_sva_admitted": True,
        "system_speedup_admitted": False,
        "headline_admitted": False,
        "ppa_admitted": False,
        "power_energy_admitted": False,
        "all10_or_full_network_admitted": False,
        "random_or_formal_protocol_proof_admitted": False,
    }, "receipt admission drift")


def rejected(name, function, *args):
    try:
        function(*args)
    except Exception as error:
        return {"name": name, "result": "REJECTED", "reason": str(error)}
    raise ValueError("tamper accepted: {}".format(name))


def run_tamper_tests(contract, receipt, run):
    attacks = []
    bad = copy.deepcopy(contract)
    bad["claim_boundary"]["system_speedup_admitted"] = True
    attacks.append(rejected("system_speedup_promotion",
                            validate_contract, bad))
    bad = copy.deepcopy(contract)
    path = "rtl_m64/qfit_adaptive_parent_selector_p256.sv"
    bad["exact_sha256"][path] = "0" * 64
    attacks.append(rejected("rtl_identity_drift",
                            validate_exact_identity, bad))
    bad = copy.deepcopy(contract)
    bad["expected"]["minimum_back_to_back_input_accepts"] = 0
    attacks.append(rejected("throughput_gate_weakening",
                            validate_contract, bad))
    bad = copy.deepcopy(receipt)
    bad["results"]["pipeline_full_push_pop_cycles"] = 0
    attacks.append(rejected("push_pop_metric_tamper",
                            validate_receipt, bad, contract, run))
    bad = copy.deepcopy(receipt)
    bad["observed_cover_matches"]["cp_source_count_256"] = 0
    attacks.append(rejected("source256_cover_tamper",
                            validate_receipt, bad, contract, run))
    bad = copy.deepcopy(receipt)
    bad["run_artifact_sha256"]["sim.raw.log"] = "0" * 64
    attacks.append(rejected("run_artifact_binding_tamper",
                            validate_receipt, bad, contract, run))
    return attacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    contract = load_json(CONTRACT)
    validate_contract(contract)
    identities = validate_exact_identity(contract)
    if args.mode == "preflight":
        print("PASS M64-r2 exact-SHA preflight identities={}".format(
            len(identities)))
        return 0
    require(args.run is not None and args.receipt is not None,
            "full mode requires --run and --receipt")
    run = args.run.resolve()
    receipt = load_json(args.receipt)
    require((run / "compile.rc").read_text().strip() == "0" and
            (run / "sim.rc").read_text().strip() == "0", "nonzero run rc")
    require(not (run / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt").exists(),
            "failed/incomplete marker present")
    snapshot_entries = validate_snapshot(run)
    validate_receipt(receipt, contract, run)
    attacks = run_tamper_tests(contract, receipt, run)
    print("PASS M64-r2 full validator snapshot_entries={} tamper_rejected={} receipt_sha256={} system_speedup_admitted=false".format(
        snapshot_entries, len(attacks), sha256_path(args.receipt)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M64-r2 validator: {}".format(error))
        raise SystemExit(1)

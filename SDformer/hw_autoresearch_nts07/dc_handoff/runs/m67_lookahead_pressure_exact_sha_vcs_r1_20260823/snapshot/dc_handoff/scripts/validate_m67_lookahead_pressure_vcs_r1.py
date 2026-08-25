#!/usr/bin/env python3
"""Fail-closed validator for exact-SHA M67 Synopsys VCS evidence."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m67_lookahead_pressure_vcs_contract_r1_20260823.json"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
IDENTITY_PATHS = {
    "rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv",
    "verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
    "verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv",
    "tb_m67/tb_m67_qfit_k4_parent_delta_lookahead_pressure.sv",
    "dc_handoff/filelists/date_m67_lookahead_pressure_vcs.f",
    "dc_handoff/scripts/run_vcs_m67_lookahead_pressure_exact_sha_r1.sh",
    "dc_handoff/scripts/build_m67_lookahead_pressure_vcs_receipt_r1.py",
    "dc_handoff/scripts/validate_m67_lookahead_pressure_vcs_r1.py",
    "vcs_launcher_binary",
}
FALSE_CLAIMS = {
    "system_speedup_admitted",
    "full_network_cycles_admitted",
    "headline_admitted",
    "ppa_admitted",
    "power_energy_admitted",
    "online_scheduler_admitted",
    "sram_dram_timing_admitted",
    "seam_k2_k3_k4_pressure_admitted",
}


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
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def identity_path(relative):
    return VCS if relative == "vcs_launcher_binary" else HW / relative


def validate_contract(contract):
    require(contract["schema"] == "m67_lookahead_pressure_vcs_contract_r1",
            "contract schema drift")
    require(contract["status"] == "FROZEN_EXACT_SHA_BEFORE_OFFICIAL_RUN",
            "contract status drift")
    require(contract["tool_policy"] == {
        "hdl_simulator": "Synopsys VCS V-2023.12-SP1 only",
        "open_source_hdl_tools_allowed": False,
        "dc_sta_formality_admitted": False,
    }, "tool policy drift")
    require(set(contract["exact_sha256"]) == IDENTITY_PATHS,
            "exact identity path set drift")
    require(contract["expected_results"] == {
        "commands": 73,
        "outputs": 73,
        "groups": 30,
        "requests": 56,
        "functional_mismatches": 0,
        "protocol_attacks": 10,
        "four_way_seam_events": 1,
        "zero_next_wait_events": 1,
    }, "expected results drift")
    require(contract["required_terminal_pass_line"] ==
            "PASS M67 K4_CTX16_ATOMIC_UNION commands=73 outputs=73 groups=30 requests=56 context16=1 meta16=1 complete16=1 push4=1 pop13push4=1",
            "terminal PASS contract drift")
    require(set(contract["claim_boundary"]) ==
            FALSE_CLAIMS | {"directed_pressure_vcs_sva_admitted"},
            "claim boundary key set drift")
    require(contract["claim_boundary"]["directed_pressure_vcs_sva_admitted"]
            is True, "directed pressure admission missing")
    for key in FALSE_CLAIMS:
        require(contract["claim_boundary"][key] is False,
                "claim promotion: " + key)
    require(len(contract["required_cover_minimum_matches"]) == 32,
            "required cover set size drift")


def validate_identities(contract):
    for relative, expected in contract["exact_sha256"].items():
        path = identity_path(relative)
        require(path.is_file(), "identity path missing: " + relative)
        require(sha256_path(path) == expected,
                "identity SHA drift: " + relative)


def validate_snapshot(run):
    manifest = run / "snapshot.sha256"
    require(manifest.is_file(), "snapshot manifest missing")
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip(" *")
        target = run / "snapshot" / relative
        require(target.is_file(), "snapshot entry missing: " + relative)
        require(sha256_path(target) == expected,
                "snapshot SHA drift: " + relative)
        entries += 1
    require(entries == 9, "snapshot entry count drift")


def validate_receipt(receipt, contract, run):
    require(receipt["schema"] == "m67_lookahead_pressure_vcs_receipt_r1" and
            receipt["status"] == "PASS_EXACT_SHA_SYNOPSYS_VCS_PRESSURE_R1",
            "receipt status drift")
    require(receipt["contract"]["sha256"] == sha256_path(CONTRACT),
            "receipt contract binding drift")
    require(Path(receipt["run_directory"]).resolve() == run.resolve(),
            "receipt run path drift")
    require(receipt["tool"] == "Synopsys VCS V-2023.12-SP1_Full64",
            "receipt tool drift")
    require(receipt["exact_identity_sha256"] == contract["exact_sha256"],
            "receipt identity ledger drift")
    require(receipt["results"] == contract["expected_results"],
            "receipt results drift")
    require(receipt["ledger_record_counts"] ==
            {"C": 73, "L": 30, "R": 56, "O": 73, "END": 1},
            "ledger record counts drift")
    require(receipt["claim_boundary"] == contract["claim_boundary"],
            "receipt claim boundary drift")
    require(receipt["assertion_modules_active"] == {"m54": True, "m66": True}
            and receipt["unique_terminal_pass"] is True
            and receipt["assertion_failure_count"] == 0
            and receipt["functional_mismatch_count"] == 0,
            "receipt functional/SVA state drift")
    require(set(receipt["observed_cover_matches"]) ==
            set(contract["required_cover_minimum_matches"]),
            "receipt cover set drift")
    for name, minimum in contract["required_cover_minimum_matches"].items():
        require(receipt["observed_cover_matches"][name] >= minimum,
                "receipt cover below minimum: " + name)
    for name, expected in receipt["run_artifact_sha256"].items():
        require(sha256_path(run / name) == expected,
                "run artifact SHA drift: " + name)


def rejected(name, function, *args):
    try:
        function(*args)
    except Exception as error:
        return {"attack": name, "status": "REJECTED", "reason": str(error)}
    raise ValueError("tamper accepted: " + name)


def tamper_tests(contract, receipt, run):
    attacks = []
    bad = copy.deepcopy(contract)
    bad["claim_boundary"]["system_speedup_admitted"] = True
    attacks.append(rejected("claim_widening", validate_contract, bad))
    bad = copy.deepcopy(contract)
    del bad["exact_sha256"][
        "tb_m67/tb_m67_qfit_k4_parent_delta_lookahead_pressure.sv"]
    attacks.append(rejected("identity_path_substitution", validate_contract, bad))
    bad = copy.deepcopy(contract)
    bad["expected_results"]["functional_mismatches"] = 1
    attacks.append(rejected("result_gate_weakening", validate_contract, bad))
    bad = copy.deepcopy(receipt)
    bad["ledger_record_counts"]["O"] = 72
    attacks.append(rejected("ledger_conservation_tamper",
                            validate_receipt, bad, contract, run))
    bad = copy.deepcopy(receipt)
    bad["observed_cover_matches"]["cp_zero_next_waits"] = 0
    attacks.append(rejected("zero_wait_cover_tamper",
                            validate_receipt, bad, contract, run))
    bad = copy.deepcopy(receipt)
    bad["run_artifact_sha256"]["sim.raw.log"] = "0" * 64
    attacks.append(rejected("run_artifact_tamper",
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
    validate_identities(contract)
    if args.mode == "preflight":
        print("PASS M67 exact-SHA preflight identities=9")
        return 0
    require(args.run is not None and args.receipt is not None,
            "full mode requires run and receipt")
    run = args.run.resolve()
    require((run / "compile.rc").read_text().strip() == "0" and
            (run / "sim.rc").read_text().strip() == "0", "nonzero run rc")
    require(not (run / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt").exists(),
            "failed marker present")
    validate_snapshot(run)
    receipt = load_json(args.receipt)
    validate_receipt(receipt, contract, run)
    attacks = tamper_tests(contract, receipt, run)
    print("PASS M67 full validator tamper_rejected={} receipt_sha256={} system_speedup_admitted=false".format(
        len(attacks), sha256_path(args.receipt)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M67 validator: {}".format(error))
        raise SystemExit(1)

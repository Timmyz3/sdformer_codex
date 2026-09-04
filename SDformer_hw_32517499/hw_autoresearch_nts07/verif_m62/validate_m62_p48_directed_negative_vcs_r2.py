#!/usr/bin/env python3
"""Validate and seal the exact-SHA M62-r2 Synopsys VCS-only run."""

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REL = Path(
    "contracts/m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2_20260823.json"
)
RUNNER_REL = Path("dc_handoff/scripts/run_vcs_m62_p48_directed_negative_r2.sh")
VALIDATOR_REL = Path("verif_m62/validate_m62_p48_directed_negative_vcs_r2.py")
EXPECTED_PASS = (
    "PASS M62 R2 directed_negative legal_full8=6 lane_checks=576 attacks=5 "
    "attack_accepts=5 sticky_cycles=15 mismatches=0"
)
EXPECTED_COVERS = {
    "cp_legal_full8_0": 1,
    "cp_legal_full8_1": 1,
    "cp_legal_full8_2": 1,
    "cp_legal_full8_3": 1,
    "cp_legal_full8_4": 1,
    "cp_legal_full8_5": 1,
    "cp_near_positive_limit": 1,
    "cp_near_negative_limit": 1,
    "cp_five_cycle_stall_case": 5,
    "cp_attack_overlap": 1,
    "cp_attack_invalid_slot": 1,
    "cp_attack_reserved_negative_128": 1,
    "cp_attack_no_signed_work": 1,
    "cp_attack_accumulator_overflow": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicate_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=no_duplicate_object)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON object required: {}".format(path))
    return payload


def validate_contract(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if payload.get("schema") != (
        "m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2"
    ):
        errors.append("contract schema drift")
    policy = payload.get("tool_policy", {})
    if policy.get("hdl_simulator") != "Synopsys VCS V-2023.12-SP1 only":
        errors.append("VCS version policy drift")
    if policy.get("open_source_hdl_tools_allowed") is not False:
        errors.append("open-source HDL policy promotion")
    for key in ("dc_run_admitted", "formality_run_admitted", "sta_run_admitted"):
        if policy.get(key) is not False:
            errors.append("{} promotion".format(key))
    geometry = payload.get("geometry", {})
    if geometry != {
        "pixels": 48,
        "outputs": 2,
        "lanes": 96,
        "source_slots": 8,
        "weight_bits": 8,
        "accumulator_bits": 13,
    }:
        errors.append("geometry drift")
    if payload.get("required_pass_line") != EXPECTED_PASS:
        errors.append("required PASS line drift")
    if payload.get("required_cover_minimum_matches") != EXPECTED_COVERS:
        errors.append("required cover ledger drift")
    semantics = payload.get("fault_semantics", {})
    required_semantics = {
        "pre_accept_rejection": False,
        "accepted_event_required": True,
        "protocol_error_required_next_cycle": True,
        "sticky_until_reset": True,
    }
    for key, expected in required_semantics.items():
        if semantics.get(key) is not expected:
            errors.append("fault semantic drift: {}".format(key))
    claim = payload.get("claim_boundary", {})
    if claim.get("additive_r2_vcs_directed_negative_evidence") is not True:
        errors.append("r2 evidence admission missing")
    for key in (
        "rtl_modified_by_r2",
        "r1_replaced_or_rewritten",
        "dc_sta_formality_admitted",
        "ppa_power_energy_admitted",
        "accuracy_admitted",
        "system_speedup",
        "headline",
        "paper_ready",
    ):
        if claim.get(key) is not False:
            errors.append("claim promotion: {}".format(key))
    frozen = payload.get("frozen_inputs", {})
    rtl_sha = frozen.get("rtl_m62/qfit_head_p48_signed_lane_fold.sv")
    if rtl_sha != "4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f":
        errors.append("M62 RTL SHA drift")
    if len(frozen) != 5:
        errors.append("frozen input set drift")
    if len(payload.get("r1_immutable_bindings", {})) != 5:
        errors.append("r1 immutable binding set drift")
    return errors


def validate_source_bindings(contract: Dict[str, Any]) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for ledger_name in ("frozen_inputs", "r1_immutable_bindings"):
        ledger = contract[ledger_name]
        for relative, expected in ledger.items():
            path = ROOT / relative
            if not path.is_file():
                raise ValueError("bound input missing: {}".format(relative))
            actual = sha256(path)
            observed[relative] = actual
            if actual != expected:
                raise ValueError(
                    "bound input SHA drift: {} expected={} observed={}".format(
                        relative, expected, actual
                    )
                )
    return observed


def cover_matches(sim_text: str, name: str) -> int:
    pattern = re.compile(
        r"m62_r2_sva\.{}.*,\s*\d+ attempts,\s*(\d+) match".format(
            re.escape(name)
        )
    )
    hits = [int(value) for value in pattern.findall(sim_text)]
    if len(hits) != 1:
        raise ValueError("cover result missing or duplicated: {}".format(name))
    return hits[0]


def require_rc_zero(path: Path) -> None:
    if path.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("nonzero run code: {}".format(path))


def validate_run(run_dir: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "compile.command.txt",
        "compile.raw.log",
        "compile.rc",
        "sim.command.txt",
        "sim.raw.log",
        "sim.rc",
        "preflight_sha_checks.txt",
        "input_sha256.txt",
        "simv",
    ]
    for name in required:
        if not (run_dir / name).is_file():
            raise ValueError("run artifact missing: {}".format(name))
    require_rc_zero(run_dir / "compile.rc")
    require_rc_zero(run_dir / "sim.rc")
    compile_text = (run_dir / "compile.raw.log").read_text()
    sim_text = (run_dir / "sim.raw.log").read_text()
    command_text = (run_dir / "compile.command.txt").read_text()
    if "V-2023.12-SP1" not in compile_text:
        raise ValueError("VCS compile version identity missing")
    if "/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs" not in command_text:
        raise ValueError("compile command is not frozen Synopsys VCS")
    lowered = command_text.lower()
    for forbidden in ("verilator", "iverilog", "yosys"):
        if forbidden in lowered:
            raise ValueError("forbidden HDL tool in command: {}".format(forbidden))
    if sim_text.splitlines().count(EXPECTED_PASS) != 1:
        raise ValueError("exact M62-r2 PASS line missing or duplicated")
    active = contract["required_module_active_line"]
    if sim_text.splitlines().count(active) != 1:
        raise ValueError("M62-r2 SVA active marker missing or duplicated")
    failure_re = re.compile(
        r"(Assertion failure|failed at|Offending|\bFatal\b|\bError-\[)", re.I
    )
    if failure_re.search(compile_text + "\n" + sim_text):
        raise ValueError("compile/simulation failure signature found")
    covers: Dict[str, int] = {}
    for name, minimum in contract["required_cover_minimum_matches"].items():
        observed = cover_matches(sim_text, name)
        covers[name] = observed
        if observed < minimum:
            raise ValueError(
                "cover below minimum: {} observed={} minimum={}".format(
                    name, observed, minimum
                )
            )
    artifacts = {
        name: sha256(run_dir / name)
        for name in required
        if (run_dir / name).is_file()
    }
    return {"covers": covers, "run_artifact_sha256": artifacts}


def run_tamper_tests(contract: Dict[str, Any]) -> List[Dict[str, str]]:
    tests = []
    mutations = []
    promoted = copy.deepcopy(contract)
    promoted["claim_boundary"]["system_speedup"] = True
    mutations.append(("system_speedup_promotion", promoted))
    rtl_drift = copy.deepcopy(contract)
    rtl_drift["frozen_inputs"][
        "rtl_m62/qfit_head_p48_signed_lane_fold.sv"
    ] = "0" * 64
    mutations.append(("rtl_sha_drift", rtl_drift))
    cover_drift = copy.deepcopy(contract)
    cover_drift["required_cover_minimum_matches"]["cp_attack_overlap"] = 0
    mutations.append(("cover_ledger_drift", cover_drift))
    semantic_drift = copy.deepcopy(contract)
    semantic_drift["fault_semantics"]["pre_accept_rejection"] = True
    mutations.append(("pre_accept_semantic_forgery", semantic_drift))
    for name, mutated in mutations:
        errors = validate_contract(mutated)
        if not errors:
            raise ValueError("tamper mutation was not rejected: {}".format(name))
        tests.append({"name": name, "result": "REJECTED", "reason": errors[0]})
    return tests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    contract_path = ROOT / CONTRACT_REL
    contract = load_json(contract_path)
    contract_errors = validate_contract(contract)
    if contract_errors:
        raise SystemExit("; ".join(contract_errors))
    bindings = validate_source_bindings(contract)
    run_dir = args.run_dir.resolve()
    run_evidence = validate_run(run_dir, contract)
    tamper_tests = run_tamper_tests(contract)
    receipt = {
        "schema": "m62_p48_signed_lane_fold_directed_negative_vcs_receipt_r2",
        "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_ONLY",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": {
            "path": str(CONTRACT_REL),
            "sha256": sha256(contract_path),
        },
        "run_dir": str(run_dir),
        "tool_identity": "Synopsys VCS V-2023.12-SP1_Full64",
        "source_bindings_sha256": bindings,
        "runner": {"path": str(RUNNER_REL), "sha256": sha256(ROOT / RUNNER_REL)},
        "validator": {
            "path": str(VALIDATOR_REL),
            "sha256": sha256(ROOT / VALIDATOR_REL),
        },
        "required_pass_line": EXPECTED_PASS,
        "observed_cover_matches": run_evidence["covers"],
        "run_artifact_sha256": run_evidence["run_artifact_sha256"],
        "tamper_tests": tamper_tests,
        "claim_boundary": contract["claim_boundary"],
        "boundary_text": contract["boundary_text"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PASS M62-r2 receipt validator")
    print("receipt={}".format(args.receipt.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

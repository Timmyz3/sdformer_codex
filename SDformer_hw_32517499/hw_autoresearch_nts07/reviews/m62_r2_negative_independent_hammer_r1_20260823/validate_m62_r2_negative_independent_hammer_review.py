#!/usr/bin/env python3
"""Independent fail-closed audit of M62-r2 directed-negative VCS evidence."""

from __future__ import print_function

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m62_r2_negative_independent_hammer_review.json"
CONTRACT = HW / "contracts/m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2_20260823.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m62_p48_directed_negative_r2.sh"
PRODUCER_VALIDATOR = HW / "verif_m62/validate_m62_p48_directed_negative_vcs_r2.py"
RUN = HW / "dc_handoff/runs/m62_p48_directed_negative_vcs_r2_sealed_20260823"
PRODUCER_RECEIPT = HW / "results/m62_p48_directed_negative_vcs_r2_20260823/m62_p48_directed_negative_vcs_receipt_r2.json"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")

EXPECTED_PASS = (
    "PASS M62 R2 directed_negative legal_full8=6 lane_checks=576 attacks=5 "
    "attack_accepts=5 sticky_cycles=15 mismatches=0"
)
EXPECTED_ACTIVE = "M62_R2_NEGATIVE_ASSERTION_MODULE_ACTIVE=1"

EXPECTED_IDENTITY = {
    "contract": "431fd824352684d85ce54e5f36c78c48fb011477d44bdf419bf859b4a2f40698",
    "runner": "bd9a8f067e8a9ddeb74cfcb0e32e935ef87767fb495ddc1c2ed9dc24728d4da9",
    "producer_validator": "fdb6a8fc2666c8858701d2790b169b76e76c4858fa0a0617dd71aecd2b2d5498",
    "producer_receipt": "0ddd67bf883f17400ea8f24f2f6746831d575a6c3e4af17d4de418ee8f0439e2",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
}

EXPECTED_SOURCE = {
    "rtl_m62/qfit_head_p48_signed_lane_fold.sv": "4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f",
    "contracts/m62_p48_signed_lane_fold_directed_negative_inputs_r2_20260823.json": "48c855cb3333f8c5392fe47a95969dfaa65d8ada6928b039ce9e3d27446c123a",
    "verif_m62/qfit_head_p48_signed_lane_fold_negative_assertions_r2.sv": "585d8828b311deb323a5a4dc15fbad08ee5490182c318f09ef7b07a1f2e0663d",
    "tb_m62/tb_qfit_head_p48_signed_lane_fold_negative_r2.sv": "5b5286deee43273aaa04df5e95d5c05db3c9c43df0e962dc6f8d2471f94d985b",
    "dc_handoff/filelists/date_m62_p48_directed_negative_vcs_r2.f": "c3143b9a183bce0063f39a4b302539b2829a555665923281629d02fec58938bc",
}

EXPECTED_R1 = {
    "contracts/m62_p48_signed_lane_fold_directed_vcs_contract_r1_20260823.json": "cc70780bcd539eec5badf420f4b8c2e58e6c4bd6c402d9b74041cce836233b24",
    "verif_m62/qfit_head_p48_signed_lane_fold_assertions.sv": "16a7907340711ab722ce1f2ec978da776004befef791d03d8bc34893d128cd05",
    "tb_m62/tb_qfit_head_p48_signed_lane_fold.sv": "f6b9a4ad2967af302a093b16f0cef37a99b389486e1cfaa86568ca548a6392e8",
    "dc_handoff/filelists/date_m62_p48_directed_vcs.f": "65ee44ec7b0614c6619863dbdb60e56010d4d76f39f34f2ff02f3b1a5f006387",
    "results/m62_p48_directed_vcs_r1_20260823/m62_p48_directed_vcs_receipt_r1.json": "e003d9efe60c46323f4a7bc69350d0ab1b083dad868d4dae76d70b16f9c71a6a",
}

EXPECTED_RUN = {
    "compile.command.txt": "a3b1d66d418d94ce3f8fb48c1de25984a69abeeef2143741c314f37de642b908",
    "compile.raw.log": "971eecd5a610d8f425bfc8dbe7582d8506529288bffd82fde3d965da23c9dd92",
    "compile.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "input_sha256.txt": "8932539d78888536e1d12080e2e4e7a5ac3607978704495c7addf27ee433cbf0",
    "preflight_sha_checks.txt": "36fd7a62bcd4eda4e0c9f2d49aa83dc4947b7991dc17481e6bbd2c7aed511d99",
    "sim.command.txt": "e127c7bff5acb518235a874bd6e2ede54d1dd1974a6a23b183bebfa1bf6c9c60",
    "sim.raw.log": "86f32c6a794fd116463f36664da9e476436ad20bd15ef159d9b9837fa4018be2",
    "sim.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "simv": "d63903b3948c999fa2e05ac4027f80c97891b2b91ba44d8760c3839fceade869",
}

EXPECTED_COVERS = {
    "cp_legal_full8_0": 1,
    "cp_legal_full8_1": 1,
    "cp_legal_full8_2": 1,
    "cp_legal_full8_3": 1,
    "cp_legal_full8_4": 1,
    "cp_legal_full8_5": 1,
    "cp_near_positive_limit": 2,
    "cp_near_negative_limit": 2,
    "cp_five_cycle_stall_case": 6,
    "cp_attack_overlap": 1,
    "cp_attack_invalid_slot": 1,
    "cp_attack_reserved_negative_128": 1,
    "cp_attack_no_signed_work": 1,
    "cp_attack_accumulator_overflow": 1,
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


def no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=no_duplicate_object)
    require(isinstance(value, dict), "top-level object required: {}".format(path))
    return value


def check_hash(path, expected, label):
    require(Path(path).is_file(), "missing {}: {}".format(label, path))
    observed = sha256_path(path)
    require(observed == expected,
            "{} SHA drift observed={} expected={}".format(label, observed, expected))
    return observed


def cover_matches(sim_text, name):
    pattern = re.compile(
        r"m62_r2_sva\.{}.*,\s*\d+ attempts,\s*(\d+) match".format(
            re.escape(name)))
    hits = [int(value) for value in pattern.findall(sim_text)]
    require(len(hits) == 1, "cover missing/duplicated: {}".format(name))
    return hits[0]


def validate_review_core(review):
    require(review.get("schema") == "m62_r2_negative_independent_hammer_review_v1",
            "review schema drift")
    require(review.get("status") ==
            "PASS_ADMITTED_DIRECTED_SCOPE_WITH_DATE_PROTOCOL_BLOCKER",
            "review status drift")
    score = review["date_oriented_score"]
    require(score["total"] == 78 and score["maximum"] == 100,
            "review score drift")
    require(sum(score["subscores"].values()) == 78,
            "review subscores do not sum")
    require({key: len(review["issues"][key]) for key in ("P0", "P1", "P2")}
            == {"P0": 1, "P1": 4, "P2": 2}, "severity inventory drift")
    require(review["proof_boundary"] == {
        "deterministic_directed_transactions": 11,
        "runtime_randomization": False,
        "constrained_random_protocol_campaign_present": False,
        "formal_protocol_property_proof_present": False,
        "directed_coverage_substitutes_for_random_or_formal": False,
        "formality_equivalence_would_substitute_for_protocol_property_proof": False,
        "admitted_statement": "The exact frozen RTL passed the enumerated deterministic legal and malformed traces under VCS/SVA.",
        "forbidden_statement": "The protocol is exhaustively proven correct for arbitrary traffic, backpressure, reset timing, fault combinations, or temporal sequences.",
    }, "proof boundary drift")
    require(review["admission"]["date_protocol_correctness"] ==
            "NO_GO_UNTIL_P0_CLOSED", "DATE protocol gate widened")
    require(review["claim_boundary_audit"]["system_speedup"] is False and
            review["claim_boundary_audit"]["headline"] is False and
            review["claim_boundary_audit"]["paper_ready"] is False,
            "review claim promotion")
    identity = review["identity_audit"]
    require(identity["contract_sha256"] == EXPECTED_IDENTITY["contract"] and
            identity["runner_sha256"] == EXPECTED_IDENTITY["runner"] and
            identity["producer_validator_sha256"] ==
            EXPECTED_IDENTITY["producer_validator"] and
            identity["producer_receipt_sha256"] ==
            EXPECTED_IDENTITY["producer_receipt"] and
            identity["vcs_launcher_sha256"] == EXPECTED_IDENTITY["vcs"],
            "review identity ledger drift")
    seal = identity["official_seal_observations"]
    require(seal == {
        "run_complete_marker_present": False,
        "completion_seal_present": False,
        "output_manifest_present": False,
        "failed_or_incomplete_marker_present": False,
        "top_level_writable_file_count": 11,
        "run_directory_mode": "0775",
    }, "review seal observations drift")
    require(review["observed_cover_matches"] == EXPECTED_COVERS,
            "review cover ledger drift")


def validate_contract(contract):
    require(contract["schema"] ==
            "m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2",
            "contract schema drift")
    require(contract["tool_policy"] == {
        "hdl_simulator": "Synopsys VCS V-2023.12-SP1 only",
        "open_source_hdl_tools_allowed": False,
        "dc_run_admitted": False,
        "formality_run_admitted": False,
        "sta_run_admitted": False,
    }, "contract tool policy drift")
    require(contract["geometry"] == {
        "pixels": 48, "outputs": 2, "lanes": 96, "source_slots": 8,
        "weight_bits": 8, "accumulator_bits": 13,
    }, "contract geometry drift")
    require(contract["frozen_inputs"] == EXPECTED_SOURCE,
            "contract frozen input ledger drift")
    require(contract["r1_immutable_bindings"] == EXPECTED_R1,
            "contract r1 binding drift")
    stimulus = contract["deterministic_stimulus"]
    require(stimulus["runtime_randomization"] is False and
            stimulus["legal_full8_cases"] == 6 and
            stimulus["checked_lanes_per_legal_case"] == 96 and
            stimulus["total_lane_checks"] == 576 and
            stimulus["accepted_fail_closed_attacks"] == 5 and
            stimulus["sticky_cycles_per_attack"] == 3,
            "contract stimulus ledger drift")
    semantics = contract["fault_semantics"]
    require(semantics["pre_accept_rejection"] is False and
            semantics["accepted_event_required"] is True and
            semantics["protocol_error_required_next_cycle"] is True and
            semantics["sticky_until_reset"] is True and
            set(semantics["closed_after_fault"]) == {
                "command_ready", "event_ready", "output_valid",
                "command_accept", "event_accept", "output_accept"},
            "contract accepted/fail-closed semantics drift")
    required_min = dict(EXPECTED_COVERS)
    required_min["cp_near_positive_limit"] = 1
    required_min["cp_near_negative_limit"] = 1
    required_min["cp_five_cycle_stall_case"] = 5
    require(contract["required_cover_minimum_matches"] == required_min,
            "contract required cover ledger drift")
    claim = contract["claim_boundary"]
    require(claim["additive_r2_vcs_directed_negative_evidence"] is True,
            "contract directed evidence admission missing")
    for key in ("rtl_modified_by_r2", "r1_replaced_or_rewritten",
                "dc_sta_formality_admitted", "ppa_power_energy_admitted",
                "accuracy_admitted", "system_speedup", "headline",
                "paper_ready"):
        require(claim[key] is False, "contract claim promotion: {}".format(key))


def validate_runner_contract_binding(contract):
    text = RUNNER.read_text(encoding="utf-8")
    require('CONTRACT="contracts/m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2_20260823.json"' in text,
            "runner contract path drift")
    all_expected = dict(EXPECTED_SOURCE)
    all_expected["contracts/m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2_20260823.json"] = EXPECTED_IDENTITY["contract"]
    for relative, expected in all_expected.items():
        require(relative in text and expected in text,
                "runner exact-SHA binding missing: {}".format(relative))
    require("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs" in text and
            "-assert svaext" in text and
            "-top tb_qfit_head_p48_signed_lane_fold_negative_r2" in text,
            "runner VCS launch drift")
    require("if [[ -e \"$RUN_DIR\" ]]" in text and
            "if [[ -e \"$RECEIPT\" ]]" in text,
            "runner overwrite guard drift")
    # This is deliberately an audit finding, not a pass claim: the contract has
    # no runner/validator/VCS SHA fields of its own.
    require("runner_sha256" not in contract and "validator_sha256" not in contract,
            "contract/runner asymmetry observation changed")


def validate_producer_receipt(receipt):
    require(receipt["schema"] ==
            "m62_p48_signed_lane_fold_directed_negative_vcs_receipt_r2" and
            receipt["status"] == "PASS_EXACT_SHA_SYNOPSYS_VCS_ONLY",
            "producer receipt terminal state drift")
    require(receipt["contract"]["sha256"] == EXPECTED_IDENTITY["contract"],
            "producer receipt contract binding drift")
    require(receipt["runner"]["sha256"] == EXPECTED_IDENTITY["runner"] and
            receipt["validator"]["sha256"] ==
            EXPECTED_IDENTITY["producer_validator"],
            "producer receipt checker binding drift")
    require(receipt["source_bindings_sha256"] ==
            dict(list(EXPECTED_SOURCE.items()) + list(EXPECTED_R1.items())),
            "producer receipt source binding drift")
    require(receipt["run_artifact_sha256"] == EXPECTED_RUN,
            "producer receipt run binding drift")
    require(receipt["observed_cover_matches"] == EXPECTED_COVERS,
            "producer receipt cover ledger drift")
    require(receipt["required_pass_line"] == EXPECTED_PASS,
            "producer receipt PASS binding drift")
    require(len(receipt["tamper_tests"]) == 4 and
            all(item["result"] == "REJECTED" for item in receipt["tamper_tests"]),
            "producer tamper ledger drift")
    claim = receipt["claim_boundary"]
    require(claim["system_speedup"] is False and
            claim["headline"] is False and claim["paper_ready"] is False,
            "producer receipt claim promotion")


def validate_run():
    for name, expected in EXPECTED_RUN.items():
        check_hash(RUN / name, expected, "run artifact {}".format(name))
    require((RUN / "compile.rc").read_text().strip() == "0" and
            (RUN / "sim.rc").read_text().strip() == "0", "nonzero VCS rc")
    compile_text = (RUN / "compile.raw.log").read_text(encoding="utf-8")
    sim_text = (RUN / "sim.raw.log").read_text(encoding="utf-8")
    command_text = (RUN / "compile.command.txt").read_text(encoding="utf-8")
    require("Version V-2023.12-SP1_Full64" in compile_text and
            "Compiler version V-2023.12-SP1_Full64" in sim_text,
            "VCS version evidence missing")
    require("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs" in command_text and
            "-assert svaext" in command_text,
            "official compile command drift")
    require(sim_text.splitlines().count(EXPECTED_ACTIVE) == 1 and
            sim_text.splitlines().count(EXPECTED_PASS) == 1,
            "unique active/PASS evidence failure")
    require(re.search(r"Assertion failure|failed at|Offending|\bFatal\b|\bError-\[",
                      compile_text + "\n" + sim_text, re.I) is None,
            "VCS failure signature found")
    covers = {name: cover_matches(sim_text, name) for name in EXPECTED_COVERS}
    require(covers == EXPECTED_COVERS, "official cover result drift")
    for marker in ("RUN_COMPLETE.txt", "completion_seal.sha256",
                   "output_manifest.sha256", "RUN_FAILED_OR_INCOMPLETE.txt"):
        require(not (RUN / marker).exists(),
                "official seal state changed; re-review required: {}".format(marker))
    writable = []
    for path in RUN.iterdir():
        if path.is_file() and path.stat().st_mode & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable.append(path.name)
    require(len(writable) == 11, "official writable-file inventory changed")
    require(stat.S_IMODE(RUN.stat().st_mode) == 0o775,
            "official run directory mode changed")
    return covers, sorted(writable)


def validate_semantics_and_oracle():
    tb = (HW / "tb_m62/tb_qfit_head_p48_signed_lane_fold_negative_r2.sv").read_text(encoding="utf-8")
    sva = (HW / "verif_m62/qfit_head_p48_signed_lane_fold_negative_assertions_r2.sv").read_text(encoding="utf-8")
    rtl = (HW / "rtl_m62/qfit_head_p48_signed_lane_fold.sv").read_text(encoding="utf-8")
    require("for (int lane = 0; lane < LANES; lane++)" in tb and
            "lane_checks++;" in tb and "!== golden[lane]" in tb and
            "legal_groups != 6 || lane_checks != 576" in tb,
            "all-96 lane oracle structure drift")
    require("$urandom" not in tb and "$random" not in tb,
            "unexpected runtime randomization")
    for cause in ("mask_overlap", "invalid_slot_mask", "reserved_negative_weight",
                  "event_signed_count == 0", "accumulator_overflow"):
        require(cause in sva and cause in rtl,
                "attack cause absent from SVA/RTL: {}".format(cause))
    require("task automatic build_attack" in tb and
            "event_negative_mask[0] = 1'b1;" in tb and
            "event_positive_mask[PIXELS+1] = 1'b1;" in tb and
            "event_weight[0 +: 8] = 8'h80;" in tb and
            "Valid source and legal weights, but no signed mask work." in tb and
            "seed_value = (case_id == 4) ? 4090 : 0;" in tb,
            "five directed attack constructions drift")
    require("accepted_on_edge = event_accept;" in tb and
            "if (!accepted_on_edge)" in tb and
            "#1;\n            if (!protocol_error)" in tb and
            "repeat (3) begin" in tb and
            "command_ready || event_ready" in tb and
            "output_valid || command_accept || event_accept" in tb,
            "accepted-then-sticky TB semantic check drift")
    require(re.search(r"event_accept\s*&&\s*malformed_event\s*\|=>\s*protocol_error", sva),
            "next-cycle fault SVA missing")
    require(re.search(r"protocol_error\s*\|=>\s*protocol_error", sva),
            "sticky SVA missing")
    require("protocol_error\n        |-> !command_ready && !event_ready && !output_valid" in sva,
            "all-interface fail-closed SVA missing")
    require("if (event_accept) begin" in rtl and
            "faulted_q <= 1'b1;" in rtl and
            "assign command_ready = !faulted_q" in rtl and
            "assign event_ready = !faulted_q" in rtl and
            "assign output_valid = !faulted_q" in rtl,
            "RTL accepted/fail-closed structure drift")


def rejected(name, function, payload):
    try:
        function(payload)
    except Exception as error:
        return {"name": name, "result": "REJECTED", "reason": str(error)}
    raise ValueError("negative attack was accepted: {}".format(name))


def negative_attacks(review, contract, receipt):
    attacks = []
    bad = copy.deepcopy(review)
    bad["proof_boundary"]["directed_coverage_substitutes_for_random_or_formal"] = True
    attacks.append(rejected("directed_as_formal_forgery", validate_review_core, bad))
    bad = copy.deepcopy(review)
    bad["claim_boundary_audit"]["system_speedup"] = True
    attacks.append(rejected("system_speedup_promotion", validate_review_core, bad))
    bad = copy.deepcopy(review)
    bad["identity_audit"]["official_seal_observations"]["completion_seal_present"] = True
    attacks.append(rejected("seal_presence_forgery", validate_review_core, bad))
    bad = copy.deepcopy(review)
    bad["date_oriented_score"]["total"] = 90
    attacks.append(rejected("score_inflation", validate_review_core, bad))
    bad = copy.deepcopy(contract)
    bad["fault_semantics"]["pre_accept_rejection"] = True
    attacks.append(rejected("preaccept_semantic_forgery", validate_contract, bad))
    bad = copy.deepcopy(contract)
    bad["required_cover_minimum_matches"]["cp_attack_overlap"] = 0
    attacks.append(rejected("cover_requirement_weakening", validate_contract, bad))
    bad = copy.deepcopy(receipt)
    bad["runner"]["sha256"] = "0" * 64
    attacks.append(rejected("receipt_runner_binding_tamper", validate_producer_receipt, bad))
    bad = copy.deepcopy(receipt)
    bad["run_artifact_sha256"]["sim.raw.log"] = "0" * 64
    attacks.append(rejected("receipt_run_artifact_tamper", validate_producer_receipt, bad))
    bad = copy.deepcopy(receipt)
    bad["observed_cover_matches"]["cp_attack_accumulator_overflow"] = 0
    attacks.append(rejected("receipt_cover_tamper", validate_producer_receipt, bad))
    return attacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent receipt overwrite")

    review = load_json(REVIEW)
    contract = load_json(CONTRACT)
    producer_receipt = load_json(PRODUCER_RECEIPT)
    validate_review_core(review)
    check_hash(CONTRACT, EXPECTED_IDENTITY["contract"], "contract")
    check_hash(RUNNER, EXPECTED_IDENTITY["runner"], "runner")
    check_hash(PRODUCER_VALIDATOR, EXPECTED_IDENTITY["producer_validator"],
               "producer validator")
    check_hash(PRODUCER_RECEIPT, EXPECTED_IDENTITY["producer_receipt"],
               "producer receipt")
    check_hash(VCS, EXPECTED_IDENTITY["vcs"], "VCS launcher")
    for relative, expected in dict(list(EXPECTED_SOURCE.items()) +
                                   list(EXPECTED_R1.items())).items():
        check_hash(HW / relative, expected, "bound source {}".format(relative))
    validate_contract(contract)
    validate_runner_contract_binding(contract)
    validate_producer_receipt(producer_receipt)
    covers, writable = validate_run()
    validate_semantics_and_oracle()
    attacks = negative_attacks(review, contract, producer_receipt)

    payload = {
        "schema": "m62_r2_negative_independent_hammer_validation_receipt_v1",
        "status": "PASS_INDEPENDENT_READ_ONLY_AUDIT_WITH_DATE_P0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "review_sha256": sha256_path(REVIEW),
        "validator_sha256": sha256_path(Path(__file__)),
        "production_contract_sha256": EXPECTED_IDENTITY["contract"],
        "production_runner_sha256": EXPECTED_IDENTITY["runner"],
        "production_validator_sha256": EXPECTED_IDENTITY["producer_validator"],
        "production_receipt_sha256": EXPECTED_IDENTITY["producer_receipt"],
        "vcs_launcher_sha256": EXPECTED_IDENTITY["vcs"],
        "production_run_artifact_sha256": EXPECTED_RUN,
        "checks": {
            "contract_runner_exact_sha_current_bytes": True,
            "official_run_current_bytes_consistent": True,
            "official_run_immutably_sealed": False,
            "producer_receipt_binds_current_run": True,
            "producer_receipt_has_detached_self_seal": False,
            "all96_lane_oracle_checks": 576,
            "accepted_then_sticky_fail_closed_attacks": 5,
            "sticky_cycles_checked": 15,
            "directed_substitutes_for_random_or_formal": False,
            "production_evidence_modified": False,
        },
        "observed_cover_matches": covers,
        "official_writable_top_level_files": writable,
        "date_oriented_score": 78,
        "severity_counts": {"P0": 1, "P1": 4, "P2": 2},
        "negative_attacks": attacks,
        "admission": review["admission"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M62-r2 independent hammer score=78 P0/P1/P2=1/4/2")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M62-r2 independent hammer: {}".format(error))
        raise SystemExit(1)

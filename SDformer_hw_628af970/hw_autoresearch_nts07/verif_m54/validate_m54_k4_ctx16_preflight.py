#!/usr/bin/env python3
"""Static/exact-SHA producer preflight for the M54 VCS experiment."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import py_compile


HW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m54_k4_ctx16_atomic_union_exact_sha_vcs_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "f1b224843cba23f9929cee4147d18e72acda05d10e4c04a0dda086dc7b05dc08")
EXPECTED_COUNTS = {
    "commands": 67,
    "groups": 24,
    "accepted_requests": 53,
    "outputs": 67,
    "physical_unique_weight_row_issues": 381,
    "logical_destination_updates": 450,
    "ledger_mismatch_count": 0,
    "sva_assertion_failure_count": 0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        value = {}
        for key, item in raw_pairs:
            require(key not in value, "duplicate key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def resolve(row):
    return (HW_ROOT / row["path"]).resolve()


def validate(contract_path):
    require(contract_path.is_file() and
            sha256(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M54 frozen contract SHA mismatch")
    contract = strict_json(contract_path)
    require(contract["schema"] ==
            "m54_k4_ctx16_atomic_union_exact_sha_vcs_contract_v1" and
            contract["status"] ==
            "FROZEN_PRE_RUN_STANDALONE_VCS_SVA_ONLY_NO_DC",
            "M54 contract identity/status mismatch")
    require(contract["frozen_geometry"] == {
        "accumulator_bits_signed": 19,
        "complete_vector_entries": 16,
        "context_id_bits": 4,
        "destination_fanout_max": 4,
        "output_lanes": 96,
        "resident_contexts": 16,
        "response_metadata_entries": 16,
        "response_tag_bits": 16,
        "source_banks": 8,
        "tile_bits": 256,
        "weight_bits_signed": 8,
    }, "M54 geometry mismatch")
    for group in ("inputs", "history_anchors"):
        for name, row in contract[group].items():
            path = resolve(row)
            require(path.is_file() and sha256(path) == row["sha256"],
                    "M54 {} source drift: {}".format(group, name))
    for key, value in EXPECTED_COUNTS.items():
        require(contract["required_vcs_evidence"][key] == value,
                "M54 expected evidence mismatch: {}".format(key))
    require(len(contract["required_vcs_evidence"][
                "minimum_cover_matches"]) == 32 and
            all(value == 1 for value in contract["required_vcs_evidence"][
                "minimum_cover_matches"].values()),
            "M54 cover contract mismatch")
    require(len(contract["required_attacks"]) == 10,
            "M54 attack population mismatch")

    rtl = resolve(contract["inputs"]["rtl"]).read_text(encoding="utf-8")
    sva = resolve(contract["inputs"]["sva"]).read_text(encoding="utf-8")
    tb = resolve(contract["inputs"]["testbench"]).read_text(encoding="utf-8")
    filelist = resolve(contract["inputs"]["vcs_filelist"]).read_text(
        encoding="utf-8").splitlines()
    required_rtl = [
        "parameter int CONTEXTS = 16", "parameter int MAX_K = 4",
        "meta_tag_q [0:META_DEPTH-1]",
        "meta_count_contexts_q [0:META_DEPTH-1]",
        "meta_contexts_q [0:META_DEPTH-1]",
        "meta_bank_valid_q [0:META_DEPTH-1]",
        "meta_context_valid_q [0:META_DEPTH-1]",
        "meta_context_subtract_q [0:META_DEPTH-1]",
        "weight_response_tag == expected_tag",
        "weight_response_context_count == expected_count",
        "weight_response_contexts == expected_contexts",
        "weight_response_bank_valid == expected_bank_valid",
        "complete_credits >= expected_count",
        "complete_tail_q <= complete_tail_q + complete_push_count",
        "context_allocated_q[expected_context[slot]] <= 1'b0",
        "context_allocated_q[launch_context[slot]] <= 1'b0",
        "for (int row = 0; row < TILE_BITS/BANKS; row++)",
        "response_acc_overflow", "faulted_q <= 1'b1",
    ]
    for token in required_rtl:
        require(token in rtl, "missing RTL semantic token: {}".format(token))
    require("M52" not in rtl and "transaction" not in rtl.lower(),
            "M52 transaction cycles leaked into RTL")
    for name in contract["required_vcs_evidence"][
            "minimum_cover_matches"]:
        require((name + ": cover property") in sva,
                "missing SVA cover: {}".format(name))
    for attack in contract["required_attacks"]:
        require((attack + "=1") in tb,
                "missing TB attack marker: {}".format(attack))
    require(filelist == [
        "rtl_m54/qfit_k4_parent_delta_p8_l96_ctx16.sv",
        "verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv",
        "tb_m54/tb_qfit_k4_parent_delta_p8_l96_ctx16.sv",
    ], "M54 filelist mismatch")
    replay = resolve(contract["inputs"]["independent_ledger_replay"])
    py_compile.compile(str(replay), doraise=True)
    diagnostic = strict_json(resolve(contract["inputs"]["diagnostic_history"]))
    require(diagnostic["status"] ==
            "DIAGNOSTIC_ONLY_ALL_FIRST_FAILURES_DISCLOSED" and
            diagnostic["first_failures_overwritten"] is False and
            len(diagnostic["entries"]) == 5,
            "M54 diagnostic disclosure mismatch")
    return {
        "schema": "m54_k4_ctx16_preflight_receipt_v1",
        "status": "PASS_M54_EXACT_SHA_STATIC_READY_FOR_VCS_NO_DC",
        "contract_sha256": sha256(contract_path),
        "source_hashes": dict((name, row["sha256"])
                              for name, row in contract["inputs"].items()),
        "expected_vcs_evidence": dict(EXPECTED_COUNTS),
        "cover_count": 32,
        "attack_count": 10,
        "diagnostic_entries": 5,
        "m49_m52_sources_modified": False,
        "open_source_simulator_used": False,
        "dc_launched": False,
        "claim_boundary": (
            "standalone exact-SHA VCS experiment only; no M52 cycle, DC, "
            "PPA, power, energy, system speedup, DATE or best-paper claim"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.contract.resolve())
    if args.output is not None:
        require(not args.output.exists(), "refusing preflight output overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS M54 preflight exact-SHA sources=7 covers=32 attacks=10 no-DC")


if __name__ == "__main__":
    main()

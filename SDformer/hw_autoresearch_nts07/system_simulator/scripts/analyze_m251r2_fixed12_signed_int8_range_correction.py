#!/usr/bin/env python3
"""Fail-closed correction of the M251 fixed12 full signed-INT8 range."""

import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = load_json(args.contract)
    require(contract.get("schema") ==
            "m251r2_fixed12_signed_int8_range_correction_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        identities[name] = {"path": spec["path"], "sha256": observed}

    old = load_json(root / contract["inputs"]["m251_r1_result"]["path"])
    old_range = old["fixed12_pwp"]
    require(old_range["universal_minimum"] == -2032 and
            old_range["universal_maximum"] == 2032,
            "M251 r1 old range identity drift")
    domain = contract["correct_numeric_domain"]
    require(domain["terms_per_pwp"] * domain["full_signed_int8_minimum"] ==
            domain["pwp_sum_minimum"] and
            domain["terms_per_pwp"] * domain["full_signed_int8_maximum"] ==
            domain["pwp_sum_maximum"],
            "corrected PWP range arithmetic drift")
    require(domain["pwp_sum_minimum"] >= domain["signed12_minimum"] and
            domain["pwp_sum_maximum"] <= domain["signed12_maximum"],
            "full signed-INT8 PWP does not fit signed12")

    cycles = {row["port"]: row
              for row in old["same_resource_cycle_simulations"]}
    expected = contract["unchanged_cycle_values"]
    require(old["exact_natural_work"]["natural_vector_op_speedup_vs_bit_sparse"] ==
            expected["natural_vector_work_speedup"] and
            cycles["WIDE144_PWP_96_WEIGHT"]["speedup_vs_dense"] ==
                expected["wide144_speedup_vs_dense"] and
            cycles["WIDE144_PWP_96_WEIGHT"]["speedup_vs_bit_sparse"] ==
                expected["wide144_speedup_vs_bit_sparse"] and
            cycles["SHARED96"]["speedup_vs_dense"] ==
                expected["shared96_speedup_vs_dense"] and
            cycles["SHARED96"]["speedup_vs_bit_sparse"] ==
                expected["shared96_speedup_vs_bit_sparse"],
            "M251 r1 cycle identity drift")

    payload = {
        "schema": "m251r2_fixed12_signed_int8_range_correction_v1",
        "status": "PASS_M251_FIXED12_RANGE_CORRECTED_CYCLES_UNCHANGED",
        "identity": identities,
        "revocation": {
            "m251_r1_old_exact_range_admitted": False,
            "old_range": contract["supersedes"]["old_range"],
            "reason": contract["supersedes"]["reason"]
        },
        "corrected_full_signed_int8_pwp_range": {
            "terms": domain["terms_per_pwp"],
            "input_range": [domain["full_signed_int8_minimum"],
                            domain["full_signed_int8_maximum"]],
            "sum_range": [domain["pwp_sum_minimum"],
                          domain["pwp_sum_maximum"]],
            "signed12_range": [domain["signed12_minimum"],
                               domain["signed12_maximum"]],
            "signed12_safe": True,
            "negative_boundary_is_exact_signed12_rail": True
        },
        "unchanged_performance": {
            "reason": "the corrected numeric range does not change fixed12 vector bytes, service cycles, work selection, DMA traffic or any replayed cycle",
            "cycle_values": expected,
            "m251_r1_result_remains_cycle_source": True
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "Correction overlay for the fixed12 full signed-INT8 PWP range only. M251 cycle results remain unchanged. PAFT-versus-control hardware gain, checkpoint INT8 export/Acc19 proof, RTL-integrated cycles, energy, system speedup, paper PPA and headline remain unadmitted."
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m251r2_fixed12_signed_int8_range_correction_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M251R2_PASS range=[-2048,2032] signed12_safe=true cycles_unchanged=true")


if __name__ == "__main__":
    main()

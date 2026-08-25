#!/usr/bin/env python3
"""Fail-closed validator for the independent M74 hammer artifacts."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m74_dual_pwp_independent_hammer_review.json"
RECON = HERE / "m74_dual_pwp_independent_reconstruction.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate key {}".format(key))
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=hook, parse_constant=reject)


def main():
    review = read(REVIEW)
    recon = read(RECON)
    require(review["status"] ==
            "NO_GO_RTL_GO_ALGORITHM_FEEDBACK_BEAM1_GATED_REDESIGN",
            "review decision drift")
    require(recon["status"] == "PASS_EXACT_RECONSTRUCTION_NO_PRODUCTION_IMPORT",
            "reconstruction status drift")
    require(recon["identity"]["production_analyzer_imported"] is False,
            "production analyzer import admitted")
    require(recon["production_comparison"]["integer_and_operator_mismatch_count"] == 0,
            "production mismatch drift")

    expected = {
        "1": 1.5923286763822577,
        "2": 1.597795772692031,
        "4": 1.5991595504147487,
        "16": 1.59972581919019,
    }
    for beam, speedup in expected.items():
        row = recon["configurations"][beam]
        require(abs(row["dual_candidate_speedup"] - speedup) < 1e-15,
                "speedup drift beam {}".format(beam))
        require(row["dual_pwp_reads"] == 2 * row["dual_selected_vectors"],
                "dual read conservation drift beam {}".format(beam))
        require(row["single_pwp_reads"] + row["single_correction_vector_ops"]
                + row["dual_pwp_reads"] + row["dual_correction_vector_ops"]
                + row["bit_sparse_fallback_vector_ops"]
                == row["dual_candidate_vector_ops"],
                "operation conservation drift beam {}".format(beam))

    require(recon["configurations"]["16"]["partition_vectors"] == 25920000,
            "population drift")
    require(recon["configurations"]["16"][
        "selected_unique_pattern_bit_checks"] == 16532448,
        "bit-check population drift")
    require(all(item["plus_identity_ok"] and item["minus_identity_ok"]
                for item in recon["arithmetic_bit_truth_table"]),
            "truth-table identity drift")

    for _name, item in review["evidence"].items():
        path = HW / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "evidence identity drift: {}".format(path))
    require(review["scores_0_to_100"]["rtl_readiness"] <= 10 and
            review["scores_0_to_100"]["overall_current_milestone"] == 39,
            "score/decision inconsistency")
    print("PASS M74 independent hammer NO_GO_RTL beams=1/2/4/16 bit_checks=16532448")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M74 independent hammer: {}".format(error))
        raise SystemExit(1)

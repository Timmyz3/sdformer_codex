#!/usr/bin/env python3
"""Emit the review-corrected literal token model for the current M104 RTL."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m104_r2_literal_serial_token_correction_contract_r1_20260824.json"
REVIEW = HW / "reviews/m104_held_weight_correction_broadcaster_independent_hammer_r1_20260824/m104_held_weight_correction_broadcaster_independent_hammer_review.json"


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
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M104 r2 output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    contract = strict_json(CONTRACT)
    review = strict_json(REVIEW)
    inputs = contract["frozen_inputs"]
    for stem in ("m103_audit", "m102_ledger", "m104_r1_result",
                 "production_rtl", "production_dc_filelist"):
        path = HW / inputs[stem]
        require(sha256(path) == inputs[stem + "_sha256"],
                "identity drift: " + stem)
    require(sha256(REVIEW) == contract["trigger"]["review_sha256"],
            "M104 independent review identity drift")
    require(review["severity_counts"]["P0"] == 1,
            "M104 review P0 count drift")
    require(review["go_no_go"]["literal_published_m104_token_result"]
            == "NO_GO_UNTIL_P0_FIXED_OR_RELABELLED",
            "M104 review verdict drift")

    model = contract["literal_current_rtl_model"]
    events = model["events_E"]
    groups = model["groups_G"]
    correction = events + model["load_tokens_per_group"] * groups
    combined = correction + model["existing_pwp_tokens"]
    ratio = model["fixed8_baseline_tokens"] / float(combined)
    require(correction == 191466250 == model["correction_tokens"],
            "literal correction token mismatch")
    require(combined == 417688505 == model["combined_tokens"],
            "literal combined token mismatch")
    require(math.isclose(ratio, model["conditional_same_clock_token_ratio"],
                         rel_tol=0.0, abs_tol=1e-15),
            "literal token ratio mismatch")

    payload = {
        "schema": "m104_r2_literal_serial_token_correction_result_v1",
        "status": "PASS_REVIEW_P0_CORRECTED_LITERAL_SERIAL_TOKEN_MODEL",
        "identity": {
            "contract_sha256": sha256(CONTRACT),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "independent_review_sha256": sha256(REVIEW),
            "production_rtl_sha256": inputs["production_rtl_sha256"],
            "m104_r1_result_sha256": inputs["m104_r1_result_sha256"],
        },
        "literal_current_rtl_model": model,
        "fused_or_overlapped_design_target": contract[
            "fused_or_overlapped_design_target"],
        "correction": {
            "r1_undercharge_tokens": groups,
            "r1_ratio_relabelled_as_unimplemented_fused_or_overlapped_target": True,
            "r2_literal_serial_formula": "E+3G",
        },
        "claim_boundary": contract["claim_boundary"],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS M104 r2 literal serial token ratio={:.12f} scheduled=false physical=false".format(ratio))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the M35-r4 checkpoint-frozen canonical descriptor-ID boundary.

This is a Python 3.6 compatible, model/static-RTL audit.  It deliberately
does not invoke or claim VCS, synthesis, Formality, PPA, or system speedup.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = (
    HW_ROOT / "contracts/m35_canonical_descriptor_contract_r4_20260822.json"
)
RTL = HW_ROOT / "rtl_m35_r4/qfit_complement_csd8_canonical.sv"
EXPECTED_CONTRACT_SHA256 = (
    "28f4c9a8b6b9c28d1e10bf47397fb6da104e6d64f14d53a3410cd09c034b5ac6"
)
EXPECTED_DESCRIPTOR_ROWS_SHA256 = (
    "209d34c4df8d3babf2ad701ee6c1305b2be17eea8ac7cf2bb62d703c5d9caff7"
)
EXPECTED_FINGERPRINT64 = "209d34c4df8d3bab"
EXPECTED_DELTAS = [2, 15, 1, 21, 110, 18, 121, 144, 97, 588]
REGRESSION_SEED = 0x4D350004


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(raw):
    raise ValueError("non-standard JSON constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(type(key) is str, "non-string JSON key")
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            "{} key population drift".format(label))


def strict_int(value, label, minimum=None, maximum=None):
    require(type(value) is int, "{} must be an exact integer".format(label))
    if minimum is not None:
        require(value >= minimum, "{} below bound".format(label))
    if maximum is not None:
        require(value <= maximum, "{} above bound".format(label))
    return value


def strict_bool(value, label):
    require(type(value) is bool, "{} must be an exact bool".format(label))
    return value


def strict_string(value, label):
    require(type(value) is str, "{} must be an exact string".format(label))
    return value


def canonical_descriptor_bytes(rows):
    return json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def resolve_anchor(spec, label):
    exact_keys(spec, {"path", "sha256"}, label)
    relative = Path(strict_string(spec["path"], label + ".path"))
    require(not relative.is_absolute(), "{} path must be relative".format(label))
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("{} path escapes repository".format(label))
    expected = strict_string(spec["sha256"], label + ".sha256")
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            "{} SHA256 syntax drift".format(label))
    require(path.is_file(), "{} anchor missing".format(label))
    require(sha256(path) == expected, "{} anchor hash drift".format(label))
    return path


def validate_term(term, label):
    exact_keys(term, {"valid", "negative", "shift"}, label)
    valid = strict_bool(term["valid"], label + ".valid")
    negative = strict_bool(term["negative"], label + ".negative")
    shift = strict_int(term["shift"], label + ".shift", 0, 9)
    if not valid:
        require(not negative and shift == 0,
                "{} invalid-slot metadata must be zero".format(label))
    return {"valid": valid, "negative": negative, "shift": shift}


def validate_contract(contract_path=DEFAULT_CONTRACT):
    contract_path = Path(contract_path).resolve()
    require(contract_path == DEFAULT_CONTRACT.resolve(),
            "M35-r4 only admits the frozen canonical contract path")
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M35-r4 canonical contract hash drift")
    contract = read_json(contract_path)
    exact_keys(contract, {
        "schema", "status", "provenance", "frozen_model_identity",
        "interface", "descriptor_rows", "generality_tradeoff",
        "claim_boundary",
    }, "contract")
    require(contract["schema"] == "m35_canonical_descriptor_contract_v4",
            "contract schema drift")
    require(contract["status"] ==
            "FROZEN_H67_EP35_TEN_ENTRY_DESCRIPTOR_ID_ROM",
            "contract status drift")

    provenance = contract["provenance"]
    exact_keys(provenance, {
        "m35_r3_math_result", "m35_r3_independent_review",
        "m35_r3_independent_validator",
    }, "provenance")
    source_path = resolve_anchor(
        provenance["m35_r3_math_result"], "m35_r3_math_result")
    review_path = resolve_anchor(
        provenance["m35_r3_independent_review"],
        "m35_r3_independent_review")
    validator_path = resolve_anchor(
        provenance["m35_r3_independent_validator"],
        "m35_r3_independent_validator")

    identity = contract["frozen_model_identity"]
    exact_keys(identity, {
        "checkpoint_sha256", "canonical_serialization",
        "descriptor_rows_sha256", "rtl_fingerprint64",
    }, "frozen_model_identity")
    checkpoint_sha = strict_string(
        identity["checkpoint_sha256"], "checkpoint_sha256")
    require(re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is not None,
            "checkpoint SHA256 syntax drift")
    require(identity["canonical_serialization"] ==
            "UTF-8 JSON of descriptor_rows with sort_keys=true and separators=(',',':')",
            "canonical serialization drift")
    require(identity["descriptor_rows_sha256"] ==
            EXPECTED_DESCRIPTOR_ROWS_SHA256, "descriptor table hash drift")
    require(identity["rtl_fingerprint64"] == EXPECTED_FINGERPRINT64,
            "descriptor fingerprint drift")

    interface = contract["interface"]
    exact_keys(interface, {
        "descriptor_id_bits", "legal_descriptor_ids",
        "illegal_descriptor_ids", "runtime_descriptor_payload_present",
        "outputs_per_accepted_packet",
        "target_unstalled_initiation_interval_cycles", "result_width_bits",
        "runtime_integer_multiplier_target",
    }, "interface")
    require(strict_int(interface["descriptor_id_bits"],
                       "descriptor_id_bits") == 4,
            "descriptor ID width drift")
    require(type(interface["legal_descriptor_ids"]) is list and
            interface["legal_descriptor_ids"] == list(range(10)) and
            all(type(value) is int
                for value in interface["legal_descriptor_ids"]),
            "legal ID population drift")
    require(type(interface["illegal_descriptor_ids"]) is list and
            interface["illegal_descriptor_ids"] == list(range(10, 16)) and
            all(type(value) is int
                for value in interface["illegal_descriptor_ids"]),
            "illegal ID population drift")
    require(strict_bool(interface["runtime_descriptor_payload_present"],
                        "runtime_descriptor_payload_present") is False,
            "runtime descriptor payload must be absent")
    require(strict_int(interface["outputs_per_accepted_packet"],
                       "outputs_per_accepted_packet") == 8,
            "output count drift")
    require(strict_int(
        interface["target_unstalled_initiation_interval_cycles"],
        "target_unstalled_initiation_interval_cycles") == 1,
        "target II drift")
    require(strict_int(interface["result_width_bits"],
                       "result_width_bits") == 56,
            "result width drift")
    require(strict_int(interface["runtime_integer_multiplier_target"],
                       "runtime_integer_multiplier_target") == 0,
            "integer multiplier target drift")

    rows = contract["descriptor_rows"]
    require(type(rows) is list and len(rows) == 10,
            "descriptor row population drift")
    checked_rows = []
    for index, row in enumerate(rows):
        label = "descriptor_rows[{}]".format(index)
        exact_keys(row, {"descriptor_id", "delta",
                         "threshold_uq0p24_raw", "terms"}, label)
        descriptor_id = strict_int(row["descriptor_id"],
                                   label + ".descriptor_id", 0, 9)
        require(descriptor_id == index, "descriptor ID ordering drift")
        delta = strict_int(row["delta"], label + ".delta", 1, 1023)
        require(delta == EXPECTED_DELTAS[index], "delta ordering drift")
        raw = strict_int(row["threshold_uq0p24_raw"],
                         label + ".threshold_uq0p24_raw")
        require(raw == (1 << 24) - delta,
                "threshold/complement identity drift")
        require(type(row["terms"]) is list and len(row["terms"]) == 4,
                "{} term population drift".format(label))
        terms = [validate_term(term, "{}.terms[{}]".format(label, term_index))
                 for term_index, term in enumerate(row["terms"])]
        seen_invalid = False
        reconstructed = 0
        shifts = []
        for term in terms:
            if not term["valid"]:
                seen_invalid = True
                continue
            require(not seen_invalid,
                    "{} valid terms must precede invalid slots".format(label))
            require(term["shift"] not in shifts,
                    "{} shifts must be unique".format(label))
            require(not shifts or term["shift"] > shifts[-1],
                    "{} shifts must be strictly increasing".format(label))
            shifts.append(term["shift"])
            coefficient = -1 if term["negative"] else 1
            reconstructed += coefficient << term["shift"]
        require(reconstructed == delta, "{} reconstruction drift".format(label))
        checked_rows.append({
            "descriptor_id": descriptor_id,
            "delta": delta,
            "threshold_uq0p24_raw": raw,
            "terms": terms,
        })
    observed_rows_sha = hashlib.sha256(
        canonical_descriptor_bytes(checked_rows)).hexdigest()
    require(observed_rows_sha == EXPECTED_DESCRIPTOR_ROWS_SHA256,
            "canonical descriptor serialization hash mismatch")

    source = read_json(source_path)
    require(source.get("schema") == "m35_complement_csd_audit_v3" and
            source.get("status") ==
            "PASS_TEN_CHECKPOINT_THRESHOLDS_EXACT_UP_TO_FOUR_TERM_COMPLEMENT_CSD_SIGNED42",
            "M35-r3 source semantics drift")
    require(source.get("identity", {}).get("checkpoint_sha256") ==
            checkpoint_sha, "checkpoint binding drift")
    require(type(source.get("thresholds")) is list and
            len(source["thresholds"]) == 10,
            "M35-r3 threshold population drift")
    for index, (row, source_row) in enumerate(zip(checked_rows,
                                                  source["thresholds"])):
        require(type(source_row.get("delta")) is int and
                source_row["delta"] == row["delta"],
                "M35-r3 delta drift at {}".format(index))
        require(type(source_row.get("threshold_uq0p24_raw")) is int and
                source_row["threshold_uq0p24_raw"] ==
                row["threshold_uq0p24_raw"],
                "M35-r3 threshold drift at {}".format(index))
        expected_terms = [
            {"coefficient": -1 if term["negative"] else 1,
             "shift": term["shift"]}
            for term in row["terms"] if term["valid"]
        ]
        require(source_row.get("csd_terms") == expected_terms,
                "M35-r3 term drift at {}".format(index))

    review = read_json(review_path)
    review_count = review.get("rtl_protocol_audit", {}).get(
        "accepted_noncanonical_tuples_for_frozen_deltas")
    require(type(review_count) is int and review_count == 3577,
            "independent review noncanonical count drift")

    generality = contract["generality_tradeoff"]
    exact_keys(generality, {"scope", "benefit", "cost"},
               "generality_tradeoff")
    for key in sorted(generality):
        strict_string(generality[key], "generality_tradeoff." + key)
    boundary = contract["claim_boundary"]
    exact_keys(boundary, {"permitted", "forbidden"}, "claim_boundary")
    strict_string(boundary["permitted"], "claim_boundary.permitted")
    strict_string(boundary["forbidden"], "claim_boundary.forbidden")
    return contract, checked_rows, {
        "source_path": source_path,
        "review_path": review_path,
        "validator_path": validator_path,
        "checkpoint_sha256": checkpoint_sha,
    }


def raw_tuple_to_descriptor_id(slots, rows):
    """Compatibility admission: accept one exact packed canonical row only."""
    require(type(slots) is list and len(slots) == 4,
            "raw descriptor must have exactly four slots")
    checked = [validate_term(slot, "raw_slots[{}]".format(index))
               for index, slot in enumerate(slots)]
    matches = [row["descriptor_id"] for row in rows
               if row["terms"] == checked]
    require(len(matches) == 1,
            "raw descriptor is not one frozen canonical ROM row")
    return matches[0]


def numeric_slot(value):
    if value == 0:
        return {"valid": False, "negative": False, "shift": 0}
    magnitude = abs(value)
    require(magnitude & (magnitude - 1) == 0,
            "numeric slot is not a signed power of two")
    return {
        "valid": True,
        "negative": value < 0,
        "shift": magnitude.bit_length() - 1,
    }


def audit_legacy_tuple_space(rows):
    choices = [0] + [sign * (1 << shift)
                     for shift in range(10) for sign in (-1, 1)]
    canonical = {
        row["delta"]: tuple(
            (-1 if term["negative"] else 1) * (1 << term["shift"])
            for term in row["terms"] if term["valid"])
        for row in rows
    }
    frozen = set(canonical)
    old_frozen_accepted = 0
    old_review_canonical = 0
    old_review_noncanonical = 0
    r4_accepted = 0
    r4_rejected_review_noncanonical = 0
    r4_rejected_additional_order_or_hole = 0
    first_rejected = None
    for numeric in itertools.product(choices, repeat=4):
        delta = sum(numeric)
        if delta not in frozen:
            continue
        old_frozen_accepted += 1
        observed = tuple(value for value in numeric if value)
        was_review_canonical = observed == canonical[delta]
        if was_review_canonical:
            old_review_canonical += 1
        else:
            old_review_noncanonical += 1
        slots = [numeric_slot(value) for value in numeric]
        try:
            selected = raw_tuple_to_descriptor_id(slots, rows)
            require(rows[selected]["delta"] == delta,
                    "adapter selected wrong frozen delta")
            r4_accepted += 1
            require(was_review_canonical,
                    "r4 accepted a review-noncanonical tuple")
        except ValueError:
            if was_review_canonical:
                r4_rejected_additional_order_or_hole += 1
            else:
                r4_rejected_review_noncanonical += 1
                if first_rejected is None:
                    first_rejected = {
                        "delta": delta,
                        "numeric_slots": list(numeric),
                        "canonical_slots": list(canonical[delta]),
                    }
    require(old_frozen_accepted == 3620,
            "legacy frozen tuple population drift")
    require(old_review_canonical == 43 and
            old_review_noncanonical == 3577,
            "independent review tuple partition drift")
    require(r4_accepted == 10 and
            r4_rejected_review_noncanonical == 3577 and
            r4_rejected_additional_order_or_hole == 33,
            "r4 strict canonical rejection population drift")
    require(first_rejected is not None,
            "noncanonical rejection witness missing")
    return {
        "legacy_r3_frozen_delta_tuples": old_frozen_accepted,
        "legacy_review_canonical_order_agnostic_tuples":
            old_review_canonical,
        "legacy_review_noncanonical_tuples": old_review_noncanonical,
        "r4_exact_rows_accepted": r4_accepted,
        "r4_review_noncanonical_tuples_rejected":
            r4_rejected_review_noncanonical,
        "r4_additional_order_or_hole_variants_rejected":
            r4_rejected_additional_order_or_hole,
        "r4_total_legacy_frozen_tuples_rejected":
            (r4_rejected_review_noncanonical +
             r4_rejected_additional_order_or_hole),
        "first_rejected_witness": first_rejected,
        "invalid_descriptor_ids_rejected": list(range(10, 16)),
    }


def product_from_row(accumulator, row):
    strict_int(accumulator, "accumulator", -(1 << 31), (1 << 31) - 1)
    correction = 0
    for term in row["terms"]:
        if term["valid"]:
            value = accumulator << term["shift"]
            correction += -value if term["negative"] else value
    return (accumulator << 24) - correction


def audit_product_identity(rows):
    edges = [-(1 << 31), -(1 << 31) + 1, -1, 0, 1,
             (1 << 31) - 2, (1 << 31) - 1]
    generator = random.Random(REGRESSION_SEED)
    values = list(edges)
    values.extend(generator.randint(-(1 << 31), (1 << 31) - 1)
                  for _unused in range(10000))
    digest = hashlib.sha256()
    minimum = None
    maximum = None
    for row in rows:
        for accumulator in values:
            observed = product_from_row(accumulator, row)
            expected = accumulator * row["threshold_uq0p24_raw"]
            require(observed == expected,
                    "signed56 product identity mismatch")
            require(-(1 << 55) <= observed <= (1 << 55) - 1,
                    "product escapes signed56")
            minimum = observed if minimum is None else min(minimum, observed)
            maximum = observed if maximum is None else max(maximum, observed)
            digest.update(("{}:{}:{}\n".format(
                row["descriptor_id"], accumulator, observed)).encode("ascii"))
    return {
        "seed_hex": "0x{:08x}".format(REGRESSION_SEED),
        "descriptor_count": len(rows),
        "edge_values_per_descriptor": len(edges),
        "random_values_per_descriptor": 10000,
        "total_products": len(rows) * len(values),
        "mismatches": 0,
        "observed_product_range": [minimum, maximum],
        "signed56_range": [-(1 << 55), (1 << 55) - 1],
        "all_products_fit_signed56": True,
        "vector_and_product_sha256": digest.hexdigest(),
    }


def parse_rtl_rom(text):
    rows = []
    blocks = re.findall(r"4'd([0-9]+):\s*begin(.*?)\n\s*end", text, re.S)
    require(len(blocks) == 10, "RTL descriptor case population drift")
    for raw_id, block in blocks:
        descriptor_id = int(raw_id)
        valid_match = re.search(
            r"rom_term_valid\s*=\s*4'b([01]{4})", block)
        require(valid_match is not None,
                "RTL valid mask missing at ID {}".format(descriptor_id))
        valid_bits = int(valid_match.group(1), 2)
        negative_match = re.search(
            r"rom_term_negative\s*=\s*4'b([01]{4})", block)
        negative_bits = (int(negative_match.group(1), 2)
                         if negative_match is not None else 0)
        shifts = [0, 0, 0, 0]
        for raw_term, raw_shift in re.findall(
                r"rom_term_shift\[([0-3])\]\s*=\s*4'd([0-9]+)", block):
            shifts[int(raw_term)] = int(raw_shift)
        terms = []
        for term in range(4):
            terms.append({
                "valid": bool((valid_bits >> term) & 1),
                "negative": bool((negative_bits >> term) & 1),
                "shift": shifts[term],
            })
        rows.append((descriptor_id, terms))
    rows.sort()
    require([item[0] for item in rows] == list(range(10)),
            "RTL descriptor ID ordering drift")
    return rows


def audit_rtl(rows):
    require(RTL.is_file(), "M35-r4 RTL candidate missing")
    text = RTL.read_text(encoding="utf-8")
    require("CONFIG_FINGERPRINT64 = 64'h{}".format(
        EXPECTED_FINGERPRINT64) in text, "RTL fingerprint binding missing")
    require("unique case (config_descriptor_id)" in text and
            "default: begin" in text and "rom_legal = 1'b0" in text,
            "RTL fail-closed ID decode missing")
    require("uses_integer_multiplier = 1'b0" in text,
            "RTL multiplier-use flag drift")
    require("stage2_ready = !stage2_valid_q || output_ready" in text and
            "stage1_ready = !stage1_valid_q || stage2_ready" in text,
            "RTL elastic ready chain missing")
    forbidden_payload = re.search(
        r"\binput\s+logic[^;\n]*\bconfig_(delta|term_valid|term_negative|term_shift)\b",
        text)
    require(forbidden_payload is None,
            "RTL exposes a raw runtime descriptor payload")
    without_line_comments = re.sub(r"//[^\n]*", "", text)
    without_comments = re.sub(r"/\*.*?\*/", "", without_line_comments,
                              flags=re.S)
    require(re.search(r"(?<![/*])\*(?![/*])", without_comments) is None,
            "RTL candidate contains an integer multiplication operator")
    decoded = parse_rtl_rom(text)
    for (descriptor_id, terms), row in zip(decoded, rows):
        require(descriptor_id == row["descriptor_id"] and
                terms == row["terms"],
                "RTL ROM differs from contract at ID {}".format(
                    descriptor_id))
    return {
        "rtl_path": str(RTL.relative_to(ROOT)),
        "rtl_sha256": sha256(RTL),
        "descriptor_rom_matches_contract": True,
        "runtime_raw_descriptor_payload_absent": True,
        "invalid_id_default_reject_present": True,
        "integer_multiplication_operators_lexical": 0,
        "outputs_per_packet_design_target": 8,
        "unstalled_initiation_interval_design_target_cycles": 1,
        "verification_status": "STATIC_CANDIDATE_ONLY_NOT_VCS_OR_SYNTHESIS",
    }


def build_report(contract_path=DEFAULT_CONTRACT):
    contract, rows, anchors = validate_contract(contract_path)
    tuple_audit = audit_legacy_tuple_space(rows)
    product_audit = audit_product_identity(rows)
    rtl_audit = audit_rtl(rows)
    return {
        "schema": "m35_r4_canonical_descriptor_audit_v1",
        "status": "PASS_MODEL_AND_STATIC_RTL_CANDIDATE_ONLY",
        "identity": {
            "contract_path": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "rtl_sha256": rtl_audit["rtl_sha256"],
            "m35_r3_math_result_sha256": sha256(anchors["source_path"]),
            "m35_r3_independent_review_sha256":
                sha256(anchors["review_path"]),
            "m35_r3_independent_validator_sha256":
                sha256(anchors["validator_path"]),
            "checkpoint_sha256": anchors["checkpoint_sha256"],
            "descriptor_rows_sha256": EXPECTED_DESCRIPTOR_ROWS_SHA256,
            "rtl_fingerprint64": EXPECTED_FINGERPRINT64,
        },
        "descriptor_boundary": tuple_audit,
        "product_identity": product_audit,
        "rtl_candidate": rtl_audit,
        "generality_tradeoff": contract["generality_tradeoff"],
        "admission": {
            "ten_frozen_descriptor_id_membership_model_admitted": True,
            "legacy_3577_noncanonical_tuple_rejection_model_admitted": True,
            "signed56_integer_identity_model_admitted": True,
            "rtl_static_rom_identity_admitted": True,
            "vcs_admitted": False,
            "rtl_pipeline_ii_admitted": False,
            "synthesis_timing_area_admitted": False,
            "formality_admitted": False,
            "integrated_local_motion_speedup_admitted": False,
            "power_energy_admitted": False,
            "paper_ppa_ready": False,
            "headline_admitted": False,
            "best_paper_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M35-r4 audit")
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independently miter the M35-r5 VCS handshake trace against the frozen ROM."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/m35_canonical_descriptor_contract_r4_20260822.json"
RTL = ROOT / "rtl_m35_r4/qfit_complement_csd8_canonical.sv"
REVIEW = (ROOT / "results/m35_r4_independent_hammer_review_20260822" /
          "m35_r4_independent_hammer_review.json")
VALIDATOR = (ROOT / "results/m35_r4_independent_hammer_review_20260822" /
             "validate_m35_r4_independent_hammer_review.py")
EXPECTED = {
    "contract": "28f4c9a8b6b9c28d1e10bf47397fb6da104e6d64f14d53a3410cd09c034b5ac6",
    "rtl": "84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854",
    "review": "8b0978b3158d780a0d5acee4ac0a780c32349e1dd45c1722f1421cb01b86fb6f",
    "validator": "305f7ff80090fcd6fd2a957e4a3f07d8b0c53219c392719e243973942674d2e8",
}
HEADER = ["kind", "sequence", "descriptor_id", "epoch", "tag", "mask"] + [
    "lane{}".format(index) for index in range(8)
]
PACKETS_PER_DESCRIPTOR = 1024
PACKETS = 10 * PACKETS_PER_DESCRIPTOR


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


def parse_decimal(raw, label):
    require(re.fullmatch(r"[0-9]+", raw) is not None,
            "{} decimal syntax".format(label))
    return int(raw, 10)


def parse_hex(raw, bits, label):
    require(re.fullmatch(r"[0-9a-fA-F]+", raw) is not None,
            "{} hex syntax".format(label))
    value = int(raw, 16)
    require(value < (1 << bits), "{} exceeds {} bits".format(label, bits))
    return value


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def read_trace(path):
    path = Path(path)
    require(path.is_file(), "VCS handshake trace missing")
    events = []
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("empty VCS handshake trace")
        require(header == HEADER, "VCS handshake trace header drift")
        for line_number, row in enumerate(reader, 2):
            require(len(row) == len(HEADER),
                    "trace column count at line {}".format(line_number))
            require(row[0] in ("C", "E", "I", "O"),
                    "trace kind at line {}".format(line_number))
            events.append((line_number, dict(zip(HEADER, row))))
    require(events, "empty VCS handshake event population")
    return events


def validate_id_events(events):
    candidates = {}
    errors = {}
    for line_number, row in events:
        if row["kind"] not in ("C", "E"):
            continue
        candidate = parse_decimal(row["descriptor_id"],
                                  "descriptor ID line {}".format(line_number))
        require(parse_decimal(row["sequence"], "sequence") == candidate,
                "configuration event sequence mismatch")
        flag = parse_hex(row["mask"], 8, "configuration flag")
        require(all(raw == "0" for raw in
                    [row["tag"]] + [row["lane{}".format(i)] for i in range(8)]),
                "configuration event padding drift")
        if row["kind"] == "C":
            require(candidate not in candidates,
                    "duplicate configuration candidate event")
            candidates[candidate] = {
                "flag": flag,
                "epoch": parse_decimal(row["epoch"], "candidate epoch"),
            }
        else:
            require(candidate not in errors, "duplicate protocol error event")
            errors[candidate] = {
                "flag": flag,
                "epoch": parse_decimal(row["epoch"], "error epoch"),
            }
    require(set(candidates) == set(range(16)),
            "all 16 descriptor IDs were not executed")
    for descriptor_id in range(10):
        require(candidates[descriptor_id] == {
            "flag": 1, "epoch": descriptor_id + 1,
        }, "legal descriptor decode event mismatch")
    for descriptor_id in range(10, 16):
        expected = {"flag": 0, "epoch": 0x7000 + descriptor_id}
        require(candidates[descriptor_id] == expected,
                "illegal descriptor decode event mismatch")
    require(set(errors) == set(range(10, 16)),
            "all six illegal IDs did not reach protocol_error")
    for descriptor_id in range(10, 16):
        require(errors[descriptor_id] == {
            "flag": 1, "epoch": 0x7000 + descriptor_id,
        }, "illegal descriptor protocol_error event mismatch")
    return {
        "descriptor_ids_executed": list(range(16)),
        "legal_ids_observed": list(range(10)),
        "illegal_ids_observed": list(range(10, 16)),
        "illegal_ids_protocol_error": list(range(10, 16)),
        "hex_A_alias_rejected": True,
    }


def validate_products(events, rows):
    inputs = {}
    outputs = {}
    for line_number, row in events:
        if row["kind"] not in ("I", "O"):
            continue
        sequence = parse_decimal(row["sequence"],
                                 "sequence line {}".format(line_number))
        target = inputs if row["kind"] == "I" else outputs
        require(sequence not in target, "duplicate {} sequence {}".format(
            row["kind"], sequence))
        descriptor_id = parse_decimal(row["descriptor_id"], "descriptor ID")
        epoch = parse_decimal(row["epoch"], "epoch")
        tag = parse_hex(row["tag"], 48, "tag")
        mask = parse_hex(row["mask"], 8, "mask")
        lane_bits = 32 if row["kind"] == "I" else 56
        lanes = [signed(parse_hex(row["lane{}".format(lane)], lane_bits,
                                  "lane"), lane_bits) for lane in range(8)]
        target[sequence] = {
            "descriptor_id": descriptor_id,
            "epoch": epoch,
            "tag": tag,
            "mask": mask,
            "lanes": lanes,
        }
    require(set(inputs) == set(range(PACKETS)),
            "input handshake sequence population drift")
    require(set(outputs) == set(range(PACKETS)),
            "output handshake sequence population drift")

    digest = hashlib.sha256()
    mismatches = 0
    checked = 0
    mask_seen = set()
    extrema = {}
    observed_min = None
    observed_max = None
    negative_term_ids = set()
    for sequence in range(PACKETS):
        source = inputs[sequence]
        sink = outputs[sequence]
        descriptor_id = sequence // PACKETS_PER_DESCRIPTOR
        require(source["descriptor_id"] == sink["descriptor_id"] ==
                descriptor_id, "descriptor sequence mismatch")
        require(source["epoch"] == sink["epoch"] == descriptor_id + 1,
                "epoch propagation mismatch")
        require(source["tag"] == sink["tag"] == sequence,
                "tag/order propagation mismatch")
        require(source["mask"] == sink["mask"], "mask propagation mismatch")
        mask_seen.add(source["mask"])
        row = rows[descriptor_id]
        require(row["descriptor_id"] == descriptor_id,
                "contract descriptor ordering drift")
        require(row["threshold_uq0p24_raw"] ==
                (1 << 24) - row["delta"], "contract threshold drift")
        if any(term["valid"] and term["negative"]
               for term in row["terms"]):
            negative_term_ids.add(descriptor_id)
        extrema.setdefault(descriptor_id, set()).update(source["lanes"])
        for lane, (accumulator, observed) in enumerate(
                zip(source["lanes"], sink["lanes"])):
            correction = 0
            for term in row["terms"]:
                if term["valid"]:
                    shifted = accumulator << term["shift"]
                    correction += -shifted if term["negative"] else shifted
            expected = (accumulator << 24) - correction
            if observed != expected:
                mismatches += 1
            require(-(1 << 55) <= observed <= (1 << 55) - 1,
                    "observed DUT product outside signed56")
            digest.update(("{}:{}:{}:{}:{}\n".format(
                sequence, descriptor_id, lane, accumulator,
                observed)).encode("ascii"))
            checked += 1
            observed_min = observed if observed_min is None else min(
                observed_min, observed)
            observed_max = observed if observed_max is None else max(
                observed_max, observed)
    require(mismatches == 0, "actual DUT product miter mismatch")
    require(checked == PACKETS * 8, "product count drift")
    require(mask_seen == set(range(256)), "all masks were not observed")
    required_edges = {-(1 << 31), -(1 << 31) + 1, -1, 0, 1,
                      (1 << 31) - 2, (1 << 31) - 1}
    for descriptor_id in range(10):
        require(required_edges.issubset(extrema[descriptor_id]),
                "signed32 extrema missing at descriptor {}".format(
                    descriptor_id))
    require(negative_term_ids == {1, 4, 6, 8, 9},
            "negative-term descriptor coverage drift")
    return {
        "packets_mitered": PACKETS,
        "actual_dut_signed56_products_mitered": checked,
        "mismatches": mismatches,
        "all_256_masks_observed": True,
        "signed32_extrema_per_descriptor_observed": True,
        "negative_term_descriptor_ids_exercised": sorted(negative_term_ids),
        "term_sign_miswire_sensitive": True,
        "observed_signed56_range": [observed_min, observed_max],
        "observed_product_digest_sha256": digest.hexdigest(),
    }


def build_report(trace):
    for name, path in (("contract", CONTRACT), ("rtl", RTL),
                       ("review", REVIEW), ("validator", VALIDATOR)):
        require(path.is_file() and sha256(path) == EXPECTED[name],
                "{} exact SHA drift".format(name))
    contract = read_json(CONTRACT)
    require(contract.get("schema") == "m35_canonical_descriptor_contract_v4",
            "contract schema drift")
    rows = contract.get("descriptor_rows")
    require(type(rows) is list and len(rows) == 10,
            "contract descriptor population drift")
    events = read_trace(trace)
    ids = validate_id_events(events)
    products = validate_products(events, rows)
    return {
        "schema": "m35_r5_vcs_trace_miter_v1",
        "status": "PASS_ACTUAL_VCS_HANDSHAKE_TRACE_ZERO_MISMATCH",
        "identity": {
            "trace_sha256": sha256(trace),
            "miter_sha256": sha256(Path(__file__).resolve()),
            "contract_sha256": EXPECTED["contract"],
            "rtl_sha256": EXPECTED["rtl"],
            "m35_r4_independent_review_sha256": EXPECTED["review"],
            "m35_r4_independent_validator_sha256": EXPECTED["validator"],
        },
        "descriptor_id_execution": ids,
        "product_miter": products,
        "claim_boundary": {
            "permitted": "Exact-source VCS trace decode/protocol observations and actual signed56 DUT product miter only.",
            "forbidden": "DC, STA, Formality, PPA, power, energy, system speedup, accuracy, external comparison, DATE headline, or best-paper claim."
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite miter output")
    report = build_report(args.trace.resolve())
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS M35-r5 independent VCS trace miter")


if __name__ == "__main__":
    main()

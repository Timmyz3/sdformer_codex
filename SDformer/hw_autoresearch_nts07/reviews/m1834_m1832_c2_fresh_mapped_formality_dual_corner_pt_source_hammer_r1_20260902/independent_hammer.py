#!/usr/bin/env python3
"""Read-only reproducer for the M1834 fail-closed source review."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import sys


REVIEW_DIR = Path(__file__).resolve().parent
HW = REVIEW_DIR.parents[1]
CONTRACT = HW / "contracts/m1832_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1832_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_m1832_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m1832_m1809_c2_fresh_mapped_two_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1832_m1809_c2_fresh_mapped_dual_corner.tcl"
FILELIST = HW / "dc_handoff/filelists/iscas_m1809_c2_registered_fault_matched_k8_k1x8_logic_only_dc.f"
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    CONTRACT: "148571756f0ddf9a361de10968958e603c868b76233b5c8924833e32911b47e0",
    RUNNER: "6b8802bb84faa4cf7fe4b2629f8a22b1a8dd5b04c576867efb7008a7e8167c99",
    FM_TCL: "68afc0bc8d14dfaed5cc504a2fc912e988db2e02be5dacb20a3fea915ba696db",
    PT_TCL: "7ec5783073b9df82bc07015f9be8243d144556662520c79b149beaccb524029a",
    FILELIST: "1dc9703bafb12ed35dda1dc9b7248881145d600c06129b00b34b7308eaeaf661",
    M1830 / "review.json": "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON token: " + token)))


def verify_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if sha(manifest) != manifest_sha or sha(outer) != outer_sha:
        raise Failure("directory seal identity mismatch: " + str(root))
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal semantic mismatch: " + str(root))
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise Failure("bad manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in listed:
            raise Failure("unsafe/duplicate manifest member")
        path = root / rel
        if not path.is_file() or path.is_symlink() or sha(path) != fields[0]:
            raise Failure("manifest member mismatch: " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if listed != actual:
        raise Failure("non-exhaustive directory seal: " + str(root))


def segment(text, begin, end):
    start = text.index(begin)
    finish = text.index(end, start)
    return text[start:finish]


def main():
    for path, digest in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise Failure("exact identity mismatch: " + str(path))
    verify_directory(
        M1811,
        "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
        "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b")
    verify_directory(
        M1830,
        "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
        "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d")
    verify_directory(
        AUTHOR,
        "39e5519d29d30dce536c3769984bd4b82a22d2da8c23b30b3767a4030c20a08b",
        "726820d1cafe55e53eb04d8d587eb0df69b4f64c8fba414f0bffe1026ca6d143")

    contract = strict_json(CONTRACT)
    m1830 = strict_json(M1830 / "review.json")
    runner = RUNNER.read_text()
    pt = PT_TCL.read_text()
    rows = [line.strip() for line in FILELIST.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 13 or len(set(rows)) != 13:
        raise Failure("filelist is not 13 unique rows")
    frozen_sources = m1830["source_identity"]["sources"]
    if set(rows) != set(frozen_sources):
        raise Failure("M1830 source map and filelist differ")
    for row in rows:
        path = HW / row
        if sha(path) != frozen_sources[row]:
            raise Failure("current live RTL differs from M1830: " + row)

    authority = segment(runner, "def verify_authority():", "\ndef same_uid_eda():")
    if "len(rows) != 13" not in authority:
        raise Failure("row cardinality check missing")
    # Confirm the blocker: the authority function never verifies the resolved
    # 13 source contents against the already sealed M1830 SHA map.
    source_pin_calls = 0
    for row in rows:
        if row in authority and frozen_sources[row] in authority:
            source_pin_calls += 1
    if source_pin_calls != 0:
        raise Failure("review assumption stale: runner now pins live RTL")

    expected_identity = segment(
        authority, "expected_identity = {", "\n    expected_budget = {")
    if '"m1834_source_review_sha256": review_sha' not in expected_identity:
        raise Failure("M1834 review JSON is not release-pinned")
    if ("m1834_source_review_manifest_sha256" in expected_identity
            or "m1834_source_review_outer_seal_file_sha256" in expected_identity):
        raise Failure("review assumption stale: release now pins complete M1834 seals")
    for token in ("M1832_EXPECTED_M1834_SOURCE_REVIEW_MANIFEST_SHA256",
                  "M1832_EXPECTED_M1834_SOURCE_REVIEW_OUTER_SHA256"):
        if token not in authority:
            raise Failure("caller-side M1834 seal input unexpectedly absent")

    required_pt = segment(runner, "def verify_pt(axis, directory):", "\ndef write_attempt(")
    for missing in ("exceptions.rpt", "design.rpt", "wire_load.rpt"):
        if missing in required_pt:
            raise Failure("review assumption stale: PT required set now contains " + missing)
    if ("report_exceptions -ignored" not in pt
            or "report_analysis_coverage" not in pt
            or "report_constraint -all_violators" not in pt):
        raise Failure("PT report generation unexpectedly changed")

    if contract["future_execution_budget"] != {
            "max_attempts": 1, "formality_runs_exact": 2,
            "pt_runs_exact": 2, "all_other_eda_runs": 0,
            "automatic_retry": False, "axis_order": ["K8", "K1X8"]}:
        raise Failure("contract execution budget drift")
    for forbidden in (
            HW / "dc_handoff/runs/.m1832_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed",
            HW / "dc_handoff/runs/m1832_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902",
            HW / "contracts/m1836_m1834_m1832_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"):
        if forbidden.exists():
            raise Failure("unauthorized namespace exists: " + str(forbidden))

    print(json.dumps({
        "status": "FAIL_CLOSED_P1_2_P2_1_REPRODUCED",
        "current_live_rtl_matches": 13,
        "launch_time_live_rtl_exact_pins": 0,
        "release_pins_m1834_review_json": True,
        "release_pins_m1834_manifest": False,
        "release_pins_m1834_outer_seal": False,
        "future_attempts_authorized": 0,
        "eda_runs_by_hammer": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Failure, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "HAMMER_ERROR", "error": str(error)}, sort_keys=True))
        sys.exit(1)

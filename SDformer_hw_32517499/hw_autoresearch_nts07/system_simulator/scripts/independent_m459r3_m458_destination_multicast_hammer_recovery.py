#!/usr/bin/env python3
"""Second fail-closed recovery for M459 reviewer-only output defects.

The frozen R1 independent implementation remains unchanged.  This wrapper
verifies it, applies only the already documented per-block waste label fix and
adds three already-computed zero-mismatch diagnostics to the phase CSV field
list.  No M458 analyzer is imported or invoked.
"""

import hashlib
from pathlib import Path


BASE_SHA256 = "cd14da8a1332392c60176f01ec6f1a6456eec2b6285881bc072b41cb5dc09a38"
REPLACEMENTS = {
    '"m459_m458_independent_hammer_contract_v1"':
        '"m459r3_m458_independent_hammer_recovery_contract_v1"',
    '"FROZEN_BEFORE_INDEPENDENT_RAW_M40_SECOND_PASS"':
        '"FROZEN_BEFORE_FINAL_TECHNICAL_RECOVERY_RAW_M40_PASS"',
    '"M459_INDEPENDENT_RAW_M40_SECOND_PASS_AUTHORIZED.marker"':
        '"M459R3_FINAL_TECHNICAL_RECOVERY_RAW_M40_PASS_AUTHORIZED.marker"',
    'status=FROZEN_CONTRACT_AUTHORIZES_ONE_READ_ONLY_INDEPENDENT_SECOND_PASS':
        'status=FROZEN_CONTRACT_AUTHORIZES_ONE_READ_ONLY_FINAL_TECHNICAL_RECOVERY_PASS',
    'group_result[str(width)]["zero"]["wasted_slots"] * OUTPUT_BLOCKS':
        'group_result[str(width)]["zero"]["wasted_slots"]',
    '(\n                group_result[str(width)]["pwp"]["wasted_slots"] +\n                group_result[str(width)]["correction"]["wasted_slots"]) * OUTPUT_BLOCKS':
        '(\n                group_result[str(width)]["pwp"]["wasted_slots"] +\n                group_result[str(width)]["correction"]["wasted_slots"])',
    '"used_pwp_patterns", "used_center_runs", "early_matcher"]':
        '"used_pwp_patterns", "used_center_runs", "early_matcher",\n'
        '                    "reconstruction_mismatches", "residual_count_mismatches",\n'
        '                    "plus_minus_overlap_mismatches"]',
}


def build_patched_source():
    base = Path(__file__).with_name(
        "independent_m459_m458_destination_multicast_hammer.py")
    source = base.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise SystemExit("M459R3 frozen base auditor identity drift")
    text = source.decode("utf-8")
    for old, new in REPLACEMENTS.items():
        if text.count(old) != 1:
            raise SystemExit("M459R3 recovery patch target count drift: {}".format(old))
        text = text.replace(old, new)
    return base, text


if __name__ == "__main__":
    base, text = build_patched_source()
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
    exec(compile(text, str(base) + "<M459R3_RECOVERY>", "exec"), namespace)

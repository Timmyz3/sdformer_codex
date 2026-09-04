#!/usr/bin/env python3
"""Fail-closed recovery wrapper for the M459 reviewer-only waste-slot label defect.

The frozen R1 independent implementation is retained byte-for-byte.  This
wrapper checks that identity, changes only the recovery contract identity and
the two CSV waste-slot fields from erroneous output-block-scaled values to the
declared per-block values, then executes the independent implementation.  It
does not import or invoke the M458 analyzer.
"""

import hashlib
from pathlib import Path


BASE_SHA256 = "cd14da8a1332392c60176f01ec6f1a6456eec2b6285881bc072b41cb5dc09a38"
base = Path(__file__).with_name(
    "independent_m459_m458_destination_multicast_hammer.py")
source = base.read_bytes()
if hashlib.sha256(source).hexdigest() != BASE_SHA256:
    raise SystemExit("M459R2 frozen base auditor identity drift")
text = source.decode("utf-8")
replacements = {
    '"m459_m458_independent_hammer_contract_v1"':
        '"m459r2_m458_independent_hammer_recovery_contract_v1"',
    '"FROZEN_BEFORE_INDEPENDENT_RAW_M40_SECOND_PASS"':
        '"FROZEN_BEFORE_TECHNICAL_RECOVERY_RAW_M40_PASS"',
    '"M459_INDEPENDENT_RAW_M40_SECOND_PASS_AUTHORIZED.marker"':
        '"M459R2_TECHNICAL_RECOVERY_RAW_M40_PASS_AUTHORIZED.marker"',
    'status=FROZEN_CONTRACT_AUTHORIZES_ONE_READ_ONLY_INDEPENDENT_SECOND_PASS':
        'status=FROZEN_CONTRACT_AUTHORIZES_ONE_READ_ONLY_TECHNICAL_RECOVERY_PASS',
    'group_result[str(width)]["zero"]["wasted_slots"] * OUTPUT_BLOCKS':
        'group_result[str(width)]["zero"]["wasted_slots"]',
    '(\n                group_result[str(width)]["pwp"]["wasted_slots"] +\n                group_result[str(width)]["correction"]["wasted_slots"]) * OUTPUT_BLOCKS':
        '(\n                group_result[str(width)]["pwp"]["wasted_slots"] +\n                group_result[str(width)]["correction"]["wasted_slots"])',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit("M459R2 recovery patch target count drift: {}".format(old))
    text = text.replace(old, new)
namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
exec(compile(text, str(base) + "<M459R2_RECOVERY>", "exec"), namespace)

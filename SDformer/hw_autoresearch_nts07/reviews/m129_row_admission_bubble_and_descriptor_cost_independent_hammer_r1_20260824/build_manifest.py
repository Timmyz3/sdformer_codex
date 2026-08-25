#!/usr/bin/env python3
import hashlib
from pathlib import Path


review = Path(__file__).resolve().parent
files = [
    "README.md",
    "RUN_COMPLETE.txt",
    "audit_m129_independent.py",
    "build_manifest.py",
    "identity_negative_tests.raw.log",
    "independent_recompute.raw.log",
    "independent_recompute_m129.py",
    "independent_result/m129_independent_recompute.json",
    "m129_identity_negative_tests.json",
    "m129_row_admission_bubble_and_descriptor_cost_independent_audit.json",
    "negative_m109_transitive_identity_drift.py",
    "negative_m122_result_identity_drift.json",
    "negative_transitive_drift_output/m129_row_admission_bubble_and_descriptor_cost.json",
    "production_rerun.raw.log",
    "production_rerun/m129_row_admission_bubble_and_descriptor_cost.json",
    "run_m129_identity_negative_tests.py",
    "source_evidence.sha256",
]
lines = []
for relative in files:
    path = review / relative
    if not path.is_file():
        raise SystemExit("missing manifest input: " + relative)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(digest + "  " + relative)
(review / "manifest.sha256").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
print("PASS M129 manifest entries={}".format(len(lines)))

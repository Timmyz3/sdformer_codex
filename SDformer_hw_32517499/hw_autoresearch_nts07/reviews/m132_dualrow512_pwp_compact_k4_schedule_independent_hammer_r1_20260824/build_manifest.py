#!/usr/bin/env python3
import hashlib
from pathlib import Path


review = Path(__file__).resolve().parent
files = [
    "README.md",
    "RUN_COMPLETE.txt",
    "audit_m132_independent.py",
    "build_manifest.py",
    "identity_negative_tests.raw.log",
    "independent_recompute.raw.log",
    "independent_recompute_m132.py",
    "independent_result/m132_independent_recompute.json",
    "m132_dualrow512_pwp_compact_k4_schedule_independent_audit.json",
    "m132_identity_negative_tests.json",
    "negative_m109_result_transitive_identity_drift.json",
    "negative_m129_result_identity_drift.json",
    "negative_transitive_drift_output/m132_dualrow512_pwp_compact_k4_schedule.json",
    "production_rerun.raw.log",
    "production_rerun/m132_dualrow512_pwp_compact_k4_schedule.json",
    "run_m132_identity_negative_tests.py",
    "source_evidence.sha256",
]
lines = []
for relative in files:
    path = review / relative
    if not path.is_file():
        raise SystemExit("missing manifest input: " + relative)
    lines.append(hashlib.sha256(path.read_bytes()).hexdigest()
                 + "  " + relative)
(review / "manifest.sha256").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
print("PASS M132 manifest entries={}".format(len(lines)))

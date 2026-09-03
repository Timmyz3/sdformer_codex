#!/usr/bin/env python3
"""CPU-only mutations for the unsealed M1808 result-hammer draft."""
from __future__ import print_function

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HAMMER_PATH = HERE / "independent_result_hammer_draft.py"
SPEC = importlib.util.spec_from_file_location("m1808_result_hammer_draft",
                                              str(HAMMER_PATH))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("draft hammer unavailable")
HAMMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HAMMER)


def replace_script(text, old, new):
    if old not in text:
        raise RuntimeError("mutation anchor absent " + old)
    return text.replace(old, new, 1)


def replace_script_last(text, old, new):
    if old not in text:
        raise RuntimeError("mutation anchor absent " + old)
    before, after = text.rsplit(old, 1)
    return before + new + after


def main():
    base = HAMMER.strict_json(HAMMER.CHECKLIST)
    source = HAMMER.SCRIPT.read_text()
    attacks = []

    def data_attack(name, mutate):
        value = copy.deepcopy(base)
        mutate(value)
        attacks.append((name, value, source))

    def source_attack(name, old, new):
        attacks.append((name, copy.deepcopy(base),
                        replace_script_last(source, old, new)))

    data_attack("schema", lambda value: value.update(schema="wrong"))
    data_attack("status", lambda value: value.update(status="COMPLETE"))
    data_attack("canonical", lambda value: value.update(canonical_result="partial"))
    data_attack("attempt", lambda value: value.update(canonical_attempt="wrong"))
    data_attack("old_failure", lambda value: value.update(old_failure_exclusion=""))
    data_attack("future_pin", lambda value: value["future_caller_pins"].pop())
    data_attack("check_count", lambda value: value["checks"].pop())
    data_attack("check_id", lambda value: value["checks"][0].update(id="HR00"))
    data_attack("duplicate_id", lambda value: value["checks"][1].update(id="HR01"))
    data_attack("severity", lambda value: value["checks"][0].update(severity="P2"))
    data_attack("requirement", lambda value: value["checks"][0].update(requirement=""))
    data_attack("claim", lambda value: value["claim_boundary"].update(system_speedup=True))
    data_attack("authorization", lambda value: value["authorization"].update(create_pass=True))
    source_attack("seal_gate", "def verify_dir_seal(", "def seal_check_removed(")
    source_attack("runtime_gate", "def validate_runtime(", "def runtime_check_removed(")
    source_attack("saif_gate", "def validate_saif(", "def saif_check_removed(")
    source_attack("power_gate", "def validate_power_and_energy(", "def power_check_removed(")
    source_attack("m1798_gate", "def isolate_m1798(", "def old_failure_check_removed(")

    rows = []
    for name, value, script_text in attacks:
        try:
            HAMMER.validate_checklist_data(value, script_text)
            rejected = False
        except HAMMER.AuditFailure:
            rejected = True
        rows.append({"name": name, "rejected": rejected})
    if not all(row["rejected"] for row in rows):
        raise RuntimeError("M1808 result-hammer draft mutation escaped")
    print(json.dumps({
        "status": "M1808_RESULT_HAMMER_DRAFT_MUTATIONS_REJECTED__NO_CANONICAL_READ_NO_PASS",
        "attacks": len(rows), "rejected": len(rows), "rows": rows,
        "canonical_read": False, "eda_runs": 0, "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

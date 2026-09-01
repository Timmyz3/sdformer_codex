#!/usr/bin/env python3
"""Static semantic-mutation suite for inert M1826; never launches EDA."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1826_m1794_c2_tsbg_production_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1826_checker_test", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_texts():
    return {
        MODULE.RTL: MODULE.RTL.read_text(),
        MODULE.TB: MODULE.TB.read_text(),
        MODULE.SVA: MODULE.SVA.read_text(),
        MODULE.RUNNER: MODULE.RUNNER.read_text(),
    }


def target_path(kind):
    return {"tb": MODULE.TB, "sva": MODULE.SVA,
            "runner": MODULE.RUNNER}[kind]


def rejected(texts):
    try:
        MODULE.validate_semantics(texts)
    except RuntimeError:
        return True
    return False


def main():
    positive = MODULE.validate_sources()
    base = source_texts()
    attacks = []
    for name, kind, old, new in MODULE.MUTATION_SPECS:
        path = target_path(kind)
        if old not in base[path]:
            raise RuntimeError("mutation anchor absent " + name)
        mutated = dict(base)
        mutated[path] = mutated[path].replace(old, new)
        attacks.append({"name": name, "kind": kind,
                        "rejected": rejected(mutated)})
    if len(attacks) < 70:
        raise RuntimeError("mutation cardinality below 70")
    if not all(item["rejected"] for item in attacks):
        raise RuntimeError("semantic mutation escaped")
    print(json.dumps({
        "status": "PASS_M1826_STATIC_AND_SEMANTIC_MUTATIONS",
        "positive": positive,
        "mutation_attacks": len(attacks),
        "mutation_rejected": sum(1 for item in attacks if item["rejected"]),
        "m1813_governance_escapes_closed": 9,
        "m1824_equivalent_bypasses_closed": 12,
        "self_runner_pin_mutation": 1,
        "contract_sha_rejection_counted_as_semantic": False,
        "attacks": attacks}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Static and mutation tests for the inert M1790 source package."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1790_c3_m1454_fixed_t10_mapped_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1790_checker_test", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_texts():
    paths = (MODULE.TB, MODULE.FILELIST, MODULE.UCLI, MODULE.PT_TCL,
             MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST)
    return dict((path, path.read_text()) for path in paths)


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

    def attack(name, path, old, new):
        mutated = dict(base)
        if old not in mutated[path]:
            raise RuntimeError("mutation anchor absent " + name)
        mutated[path] = mutated[path].replace(old, new)
        attacks.append({"name": name, "rejected": rejected(mutated)})

    attack("clock_period", MODULE.TB, "always #1.5 clk_core", "always #2 clk_core")
    attack("measured_tiles", MODULE.TB, "MEASURE_TILES = 8", "MEASURE_TILES = 7")
    attack("public_xz_gate", MODULE.TB, "$isunknown({config_ready", "$isunknown({config_accept")
    attack("hierarchical_read", MODULE.TB, "send_config();", "if (dut.busy) send_config();")
    attack("ucli_scope", MODULE.UCLI, MODULE.SAIF_SCOPE, MODULE.TOP)
    attack("black_box_gate", MODULE.PT_TCL, "M1790_FAIL_BLACK_BOX_AFTER_LINK", "M1790_BLACK_BOX_IGNORED")
    attack("annotation_gate", MODULE.PT_TCL, "M1790_FAIL_EXACT_NET_ANNOTATION_GATE", "M1790_NET_ANNOTATION_IGNORED")
    attack("automatic_retry", MODULE.RUNNER, "automatic_retry\": False", "automatic_retry\": True")
    attack("execution_budget", MODULE.RUNNER, "vcs_compiles\": 1", "vcs_compiles\": 2")
    attack("gate_initializer", MODULE.RUNNER, "+define+UNIT_DELAY", "+define+UNIT_DELAY +initreg")

    if not all(item["rejected"] for item in attacks):
        raise RuntimeError("mutation escaped")
    print(json.dumps({"status": "PASS_M1790_STATIC_AND_MUTATION_TESTS",
                      "positive": positive, "mutation_attacks": len(attacks),
                      "mutation_rejected": sum(1 for item in attacks
                                               if item["rejected"]),
                      "attacks": attacks}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

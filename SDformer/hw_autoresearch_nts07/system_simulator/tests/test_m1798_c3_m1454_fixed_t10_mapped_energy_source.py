#!/usr/bin/env python3
"""Static and mutation tests for the inert additive M1798 source package."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1798_c3_m1454_fixed_t10_mapped_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1798_checker_test", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_texts():
    paths = (MODULE.TB_TAG, MODULE.FILELIST, MODULE.UCLI, MODULE.PT_TCL,
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

    attack("ordered_tile_done_comparison", MODULE.TB_TAG,
           "sampled_tile_done_tag !==", "sampled_tile_done_tag ===")
    attack("ordered_tile_done_queue", MODULE.TB_TAG,
           "expected_tile_done_tag[expected_read]",
           "sampled_tile_done_tag")
    attack("directed_input_tag_comparison", MODULE.TB_TAG,
           "sampled_raw_tag !== directed_tag(expected_write)",
           "sampled_raw_tag === directed_tag(expected_write)")
    attack("missing_tag_gate", MODULE.TB_TAG,
           "expected_write != EXPECTED_TOTAL_TAGS",
           "expected_write == EXPECTED_TOTAL_TAGS")
    attack("tag_pass_token", MODULE.TB_TAG,
           "PASS_M1798_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD",
           "PASS_M1798_TAG_CHECK_REMOVED")
    attack("filelist_tag_monitor", MODULE.FILELIST, str(MODULE.TB_TAG),
           str(MODULE.TB_BASE))
    attack("ucli_scope", MODULE.UCLI, MODULE.SAIF_SCOPE, MODULE.TOP)
    attack("ptpx_inherited_authority", MODULE.PT_TCL,
           str(MODULE.M1790_PT_TCL), "/tmp/unreviewed_ptpx.tcl")
    attack("automatic_retry", MODULE.RUNNER,
           "automatic_retry\": False", "automatic_retry\": True")
    attack("execution_budget", MODULE.RUNNER,
           "vcs_compiles\": 1", "vcs_compiles\": 2")
    attack("gate_initializer", MODULE.RUNNER,
           "+define+UNIT_DELAY", "+define+UNIT_DELAY +initreg")

    for index, token in enumerate(MODULE.RELEASE_BINDING_TOKENS):
        attack("release_binding_%02d" % index, MODULE.RUNNER, token,
               "MUTATED_RELEASE_BINDING_%02d" % index)

    if not all(item["rejected"] for item in attacks):
        raise RuntimeError("mutation escaped")
    print(json.dumps({
        "status": "PASS_M1798_STATIC_AND_MUTATION_TESTS",
        "positive": positive,
        "mutation_attacks": len(attacks),
        "mutation_rejected": sum(1 for item in attacks if item["rejected"]),
        "release_binding_mutations": len(MODULE.RELEASE_BINDING_TOKENS),
        "attacks": attacks}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

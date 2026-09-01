#!/usr/bin/env python3
"""Static and mutation tests for inert additive M1808 source package."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1808_c3_m1454_fixed_t10_mapped_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1808_checker_test", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_texts():
    paths = (MODULE.TB, MODULE.TB_TAG, MODULE.FILELIST, MODULE.UCLI,
             MODULE.PT_TCL, MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST)
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

    attack("immediate_architectural_gate", MODULE.TB,
           "M1808 architectural/control output contains X/Z",
           "M1808 architectural check removed")
    attack("settle_exact_three", MODULE.TB,
           "post_reset_settle_cycles == 3",
           "post_reset_settle_cycles == 2")
    attack("settle_overrun", MODULE.TB,
           "post_reset_settle_cycles > 3",
           "post_reset_settle_cycles > 4")
    attack("quiescent_activity", MODULE.TB,
           "M1808 activity during reset-settling",
           "M1808 quiescence removed")
    attack("debug_binary_boundary", MODULE.TB,
           "M1808 debug counter X/Z at settling boundary",
           "M1808 binary boundary removed")
    attack("debug_zero_boundary", MODULE.TB,
           "M1808 debug counter nonzero at settling boundary",
           "M1808 zero boundary removed")
    attack("full_gate_restore", MODULE.TB,
           "M1808 full public output contains X/Z",
           "M1808 full gate removed")
    attack("boundary_must_close", MODULE.TB,
           "if (!full_public_check_enabled || post_reset_settle_cycles != 3)",
           "if (full_public_check_enabled || post_reset_settle_cycles != 3)")
    attack("warmup_count", MODULE.TB,
           "repeat (8) @(posedge clk_core)",
           "repeat (7) @(posedge clk_core)")
    attack("measured_tiles", MODULE.TB,
           "tile < MEASURE_TILES", "tile <= MEASURE_TILES")
    attack("counter_conservation", MODULE.TB,
           "debug_stage1_issues-base_issues != 17*MEASURE_TILES",
           "debug_stage1_issues-base_issues == 17*MEASURE_TILES")
    attack("stall_nonvacuity", MODULE.TB,
           "result_stall_cycles == 0 || raw_stall_cycles == 0",
           "result_stall_cycles != 0 || raw_stall_cycles != 0")
    attack("ordered_tile_done_comparison", MODULE.TB_TAG,
           "sampled_tile_done_tag !==", "sampled_tile_done_tag ===")
    attack("ordered_tile_done_queue", MODULE.TB_TAG,
           "expected_tile_done_tag[expected_read]", "sampled_tile_done_tag")
    attack("directed_input_tag", MODULE.TB_TAG,
           "sampled_raw_tag !== directed_tag(expected_write)",
           "sampled_raw_tag === directed_tag(expected_write)")
    attack("missing_tag_gate", MODULE.TB_TAG,
           "expected_write != EXPECTED_TOTAL_TAGS",
           "expected_write == EXPECTED_TOTAL_TAGS")
    attack("filelist_new_tb", MODULE.FILELIST, str(MODULE.TB),
           str(MODULE.M1798.TB_BASE))
    attack("ucli_scope", MODULE.UCLI, MODULE.SAIF_SCOPE, MODULE.TOP)
    attack("ptpx_authority", MODULE.PT_TCL,
           str(MODULE.M1798.M1790_PT_TCL), "/tmp/unreviewed_ptpx.tcl")
    attack("automatic_retry", MODULE.RUNNER,
           "automatic_retry\": False", "automatic_retry\": True")
    attack("execution_budget", MODULE.RUNNER,
           "vcs_compiles\": 1", "vcs_compiles\": 2")
    attack("gate_initializer", MODULE.RUNNER,
           "\"-debug_access+r\"", "\"+initreg\", \"-debug_access+r\"")
    attack("misleading_delay", MODULE.RUNNER,
           "\"-debug_access+r\"", "\"+define+UNIT_DELAY\", \"-debug_access+r\"")

    for index, token in enumerate(MODULE.RELEASE_BINDING_TOKENS):
        attack("release_binding_%02d" % index, MODULE.RUNNER, token,
               "MUTATED_RELEASE_BINDING_%02d" % index)

    if not all(item["rejected"] for item in attacks):
        raise RuntimeError("mutation escaped")
    print(json.dumps({
        "status": "PASS_M1808_STATIC_AND_MUTATION_TESTS",
        "positive": positive,
        "mutation_attacks": len(attacks),
        "mutation_rejected": sum(1 for item in attacks if item["rejected"]),
        "release_binding_mutations": len(MODULE.RELEASE_BINDING_TOKENS),
        "attacks": attacks}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

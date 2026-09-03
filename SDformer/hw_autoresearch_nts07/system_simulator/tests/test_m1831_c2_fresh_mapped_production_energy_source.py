#!/usr/bin/env python3
"""Static mutation checks for the formal M1831 source package."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1831_c2_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1831_checker_test", str(CHECKER))
MODULE = importlib.util.module_from_spec(SPEC)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1831 checker unavailable")
SPEC.loader.exec_module(MODULE)


def texts():
    paths = (MODULE.RUNNER, MODULE.CHECKER, MODULE.TEST, MODULE.CONTRACT,
             MODULE.CORE, MODULE.FAULT, MODULE.TOP_TB, MODULE.MEM,
             MODULE.PROD_ASSERT, MODULE.M979, MODULE.UCLI, MODULE.PT_TCL,
             MODULE.FILELISTS["k8"], MODULE.FILELISTS["k1x8"])
    return dict((path, path.read_text()) for path in paths)


ATTACKS = (
    ("k8_derived_top", MODULE.CORE,
     "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0",
     "wrong_k8_top"),
    ("k1x8_derived_top", MODULE.CORE,
     "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1",
     "wrong_k1x8_top"),
    ("runner_k8_cycle", MODULE.RUNNER, "[51, 131, 486, 1231, 14]",
     "[50, 131, 486, 1231, 14]"),
    ("runner_k1x8_cycle", MODULE.RUNNER, "[53, 133, 499, 1246, 14]",
     "[52, 133, 499, 1246, 14]"),
    ("runner_compile_budget", MODULE.RUNNER, '"vcs_compiles": 2',
     '"vcs_compiles": 1'),
    ("runner_sim_budget", MODULE.RUNNER, '"simv_runs": 10',
     '"simv_runs": 9'),
    ("runner_saif_budget", MODULE.RUNNER, '"saif_files": 10',
     '"saif_files": 9'),
    ("runner_ptpx_budget", MODULE.RUNNER, '"ptpx_runs": 10',
     '"ptpx_runs": 9'),
    ("runner_saif_gate", MODULE.RUNNER,
     "all ten mapped SAIF coordinates required before PTPX",
     "partial SAIF accepted before PTPX"),
    ("runner_partial_binding", MODULE.RUNNER,
     "M1811 = HW /", "PARTIAL_MARKER = '.m1811_bad'\nM1811 = HW /"),
    ("runner_m1811_manifest", MODULE.RUNNER,
     "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
     "095050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"),
    ("runner_m1830_review", MODULE.RUNNER,
     "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
     "09e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"),
    ("runner_k8_netlist", MODULE.RUNNER,
     "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
     "03605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792"),
    ("runner_k1x8_sdc", MODULE.RUNNER,
     "1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a",
     "0631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a"),
    ("runner_source_review_path", MODULE.RUNNER,
     "reviews/m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902",
     "reviews/m1832_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902"),
    ("runner_source_review_env", MODULE.RUNNER,
     "M1831_EXPECTED_M1833_REVIEW_SHA256",
     "M1831_EXPECTED_M1832_REVIEW_SHA256"),
    ("runner_source_review_severity_gate", MODULE.RUNNER,
     "raise Failure(\"M1833 source review not admitted\")",
     "pass  # source review admission bypassed"),
    ("runner_launch_release_path", MODULE.RUNNER,
     "contracts/m1835_m1833_m1831_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json",
     "contracts/m1833_m1832_m1831_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json"),
    ("runner_launch_release_env", MODULE.RUNNER,
     "M1831_EXPECTED_M1835_RELEASE_SHA256",
     "M1831_EXPECTED_M1833_RELEASE_SHA256"),
    ("ucli_scope", MODULE.UCLI,
     "core.dut.implementation", "core.dut"),
    ("ptpx_derived_top0", MODULE.PT_TCL,
     "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0",
     "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"),
    ("ptpx_macro_boundary", MODULE.PT_TCL,
     'puts $scope_fp "macro_count=0"', 'puts $scope_fp "macro_count=16"'),
    ("fault_xz_gate", MODULE.FAULT,
     "$isunknown({protocol_error, numeric_overflow,",
     "$isunknown({numeric_overflow,"),
    ("k8_filelist_mapped_binding", MODULE.FILELISTS["k8"],
     "+define+M1831_AXIS_K8", "+define+M1831_AXIS_K8\n/tmp/partial_mapped.v"),
    ("contract_status", MODULE.CONTRACT,
     "SOURCE_COMPLETE__M1811_M1830_BOUND__M1833_REVIEW_AND_M1835_RELEASE_REQUIRED__NO_EDA",
     "INCOMPLETE_SOURCE_DRAFT"),
    ("contract_m1811_identity", MODULE.CONTRACT,
     "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
     "095050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"),
    ("contract_m1830_identity", MODULE.CONTRACT,
     "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
     "09e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"),
    ("contract_source_review_identity", MODULE.CONTRACT,
     "reviews/m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902",
     "reviews/m1832_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902"),
    ("contract_launch_release_identity", MODULE.CONTRACT,
     "contracts/m1835_m1833_m1831_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json",
     "contracts/m1833_m1832_m1831_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json"),
)


def main():
    positive = MODULE.validate_sources()
    base = texts(); rows = []
    for name, path, old, new in ATTACKS:
        if old not in base[path]:
            raise RuntimeError("attack anchor absent " + name)
        mutated = dict(base); mutated[path] = base[path].replace(old, new, 1)
        try:
            MODULE.validate_source_texts(mutated)
            rejected = False
        except RuntimeError:
            rejected = True
        rows.append({"name": name, "rejected": rejected})
    if not all(row["rejected"] for row in rows):
        raise RuntimeError("M1831 formal source mutation escaped")
    print(json.dumps({
        "status": "PASS_M1831_FORMAL_SOURCE_STATIC_MUTATIONS",
        "positive": positive, "attacks": len(rows),
        "rejected": sum(1 for row in rows if row["rejected"]),
        "rows": rows, "eda_runs": 0, "license_queries": 0,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

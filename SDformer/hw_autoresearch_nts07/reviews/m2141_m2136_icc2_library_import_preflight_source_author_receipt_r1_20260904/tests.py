#!/usr/bin/python3.12
"""Pure-Python mutation checks for M2141 source gates; invokes no EDA."""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m2141_selfcheck", HERE / "selfcheck.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

tcl = module.TCL.read_text()
runner = module.RUNNER.read_text()
assert module.text_errors(tcl, runner) == []

mutations = {
    "wrong_option_setter": (tcl.replace("set_app_options -name lib.configuration.local_output_dir", "set_app_var lib.configuration.local_output_dir", 1), runner),
    "overwrite_conversion": (tcl.replace("-output_directory $frame_dir]", "-output_directory $frame_dir -overwrite]", 1), runner),
    "rtl_import": (tcl.replace("    save_lib", "    read_verilog forbidden.v\n    save_lib", 1), runner),
    "pnr_command": (tcl.replace("    save_lib", "    route_auto\n    save_lib", 1), runner),
    "missing_collateral_absorption": (tcl, runner.replace('mv -- "${PRIOR_COLLATERAL}" "${WORK}/prior_m2135_collateral/icc2_output.txt"', ": # removed", 1)),
    "second_icc2": (tcl, runner + '\n"${ICC2}" -f "${TCL}"\n'),
    "second_license_query": (tcl, runner + '\n"${LMUTIL}" lmstat\n'),
}
for name, (candidate_tcl, candidate_runner) in mutations.items():
    errors = module.text_errors(candidate_tcl, candidate_runner)
    assert errors, f"mutation survived: {name}"
    print(f"PASS_MUTATION_REJECTED {name} errors={len(errors)}")

print("PASS_M2141_AUTHOR_MUTATION_TESTS total=7")

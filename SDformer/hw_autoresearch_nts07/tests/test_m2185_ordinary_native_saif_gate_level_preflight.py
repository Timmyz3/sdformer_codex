#!/opt/anaconda3/bin/python3
"""No-EDA mutation regression for M2185 gate-level SAIF successor."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m2185_m2018_ordinary_native_saif_gate_level_reset_preflight_one_shot.py"
BASE_TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
M2176_TEST = HW / "tests/test_m2176_ordinary_native_saif_reset_semantics_preflight.py"
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("m2185_runner_test", RUNNER)
P = load("m2176_parser_for_m2185_test", PARSER)
T = load("m2172_fixture_for_m2185_test", BASE_TEST)


def rejected(text: str) -> bool:
    try:
        R.audit_ucli(text)
    except R.Failure:
        return True
    return False


def main() -> None:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX="/tmp/m2185_test_pycache")
    inherited = subprocess.run([sys.executable, "-B", str(M2176_TEST)], check=True,
                               capture_output=True, text=True, env=env, timeout=180)
    assert "PASS_M2176_SOURCE_TESTS" in inherited.stdout
    source = R.UCLI.read_text()
    old = R.OLD_UCLI.read_text()
    assert R.audit_ucli(source)["added_effective_commands"] == 1
    assert rejected(old)
    assert rejected(source.replace(R.SCOPE, R.SCOPE + "_wrong"))
    assert rejected(source.replace(R.GATE_LEVEL + "\npower " + R.SCOPE,
                                   "power " + R.SCOPE + "\n" + R.GATE_LEVEL))
    assert rejected(source.replace(R.GATE_LEVEL + "\npower " + R.SCOPE + "\npower -enable",
                                   "power " + R.SCOPE + "\npower -enable\n" + R.GATE_LEVEL))

    failures = [
        "Warning: reset ignored.", "Warning: reset rejected.",
        "Error: reset denied.", "Warning: reset unsupported.",
        "Warning: reset failed.", "Error: reset cannot complete.",
        "Warning: reset unable to complete.", "Error: reset remained uncleared.",
        "Warning: reset retained old counters.", "Error: reset remained active.",
        "Warning: reset not cleared.", "Error: reset not reset.",
        "Warning: clear failed.", "Error: clear request denied.",
    ]
    assert all(P.reset_failure_lines(line) == [line] for line in failures)
    with tempfile.TemporaryDirectory(prefix="m2185_reset_semantics_") as raw:
        runtime = Path(raw) / "rtl_sim.log"
        for line in failures:
            runtime.write_text(T.runtime_text() + line + "\n")
            try:
                P.parse_runtime(runtime)
            except P.Failure:
                pass
            else:
                raise AssertionError(f"reset failure escaped M2176 parser: {line}")

    print("PASS_M2185_SOURCE_TESTS inherited_m2176=1 ucli_exact_delta=1 "
          "missing_gate_rejected=1 wrong_scope_rejected=1 order_mutations_rejected=2 "
          "reset_failure_mutations_rejected=14 eda_runs=0")


if __name__ == "__main__":
    main()

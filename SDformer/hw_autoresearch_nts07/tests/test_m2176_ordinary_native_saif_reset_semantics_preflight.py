#!/opt/anaconda3/bin/python3
"""No-EDA M2176 minimal reset-semantic regression."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[1]
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"
BASE_PARSER = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
BASE_TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_one_shot.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("m2176_parser_test", PARSER)
T = load("m2172_test_fixture_for_m2176", BASE_TEST)


def main() -> None:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX="/tmp/m2176_test_pycache")
    inherited = subprocess.run([sys.executable, "-B", str(BASE_TEST)], check=True,
                               capture_output=True, text=True, env=env, timeout=120)
    assert "PASS_M2172_SOURCE_TESTS tests=42" in inherited.stdout
    assert M.static_check()["status"] == "PASS_M2176_STATIC_PARSER"
    assert M.BASE_PATH == BASE_PARSER
    assert M.parse_saif is M.BASE.parse_saif
    source = RUNNER.read_text()
    assert all(token in source for token in ("M2176", "M2177", "M2178", "M2179"))

    failures = [
        "Warning: reset ignored.", "Warning: reset rejected.",
        "Error: reset denied.", "Warning: reset unsupported.",
        "Warning: reset failed.", "Error: reset cannot complete.",
        "Warning: reset unable to complete.", "Error: reset remained uncleared.",
        "Warning: reset retained old counters.", "Error: reset remained active.",
        "Warning: reset not cleared.", "Error: reset not reset.",
        "Warning: clear failed.", "Error: clear request denied.",
    ]
    for line in failures:
        assert M.reset_failure_lines(line) == [line]
    successes = [
        "Info: power reset request accepted and switching counters cleared.",
        "Info: reset completed successfully.",
    ]
    for line in successes:
        assert M.reset_failure_lines(line) == []

    with tempfile.TemporaryDirectory(prefix="m2176_runtime_") as raw:
        path = Path(raw) / "rtl_sim.log"
        path.write_text(T.runtime_text())
        assert M.parse_runtime(path)["completion_ledger"]["products"] == 29472
        for line in failures:
            path.write_text(T.runtime_text() + line + "\n")
            try:
                M.parse_runtime(path)
            except M.Failure:
                pass
            else:
                raise AssertionError(f"minimal failure escaped parse_runtime: {line}")
        path.write_text(T.runtime_text() + successes[0] + "\n")
        assert M.parse_runtime(path)["completion_ledger"]["products"] == 29472

    print("PASS_M2176_SOURCE_TESTS inherited_m2172=42 "
          "minimal_failure_mutations=14 accepted_controls=2 eda_runs=0")


if __name__ == "__main__":
    main()

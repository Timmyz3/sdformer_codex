#!/opt/anaconda3/bin/python3
"""No-EDA mutation regression for the narrow M2201 SAIF parser repair."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[1]
PARSER = HW / "system_simulator/scripts/parse_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight_one_shot.py"
BASE_TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
M2185_TEST = HW / "tests/test_m2185_ordinary_native_saif_gate_level_preflight.py"
OLD_Q = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904.failed.3245526.quarantine"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("m2201_parser_test", PARSER)
T = load("m2172_fixture_for_m2201", BASE_TEST)


def must_fail(callable_) -> None:
    try:
        callable_()
    except P.Failure:
        return
    raise AssertionError("mutation unexpectedly passed")


def main() -> None:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX="/tmp/m2201_source_test_pycache")
    inherited = subprocess.run([sys.executable, "-B", str(M2185_TEST)], check=True,
                               capture_output=True, text=True, env=env, timeout=180)
    assert "PASS_M2185_SOURCE_TESTS" in inherited.stdout
    assert P.static_check()["status"] == "PASS_M2201_STATIC_PARSER"

    # Immutable M2187 files are read-only regression inputs, never admitted/reused.
    old_pre_sha = P.sha256(OLD_Q / "rtl_prehistory.saif")
    old_measure_sha = P.sha256(OLD_Q / "rtl_measurement.saif")
    diagnostic = P.parse_saif(OLD_Q / "rtl_prehistory.saif",
                              role="diagnostic_prehistory")
    measurement = P.parse_saif(OLD_Q / "rtl_measurement.saif", role="measurement")
    assert diagnostic["record_count"] == 93971
    assert diagnostic["conservation_mode"] == "uniform_floor_subtick"
    assert abs(diagnostic["subtick_residual_raw"] - 0.01) <= 1e-6
    assert diagnostic["full_tick_error_accepted"] is False
    assert measurement["record_count"] == 93971
    assert measurement["conservation_failures"] == 0
    assert measurement["tx_nonzero_record_count"] == 0
    assert P.sha256(OLD_Q / "rtl_prehistory.saif") == old_pre_sha
    assert P.sha256(OLD_Q / "rtl_measurement.saif") == old_measure_sha

    A = P.audit_conservation_fields
    assert A([(7.0, 3.0, 0.0, 2.0)], duration_raw=10.0,
             role="diagnostic_prehistory")["mode"] == "exact"
    assert A([(7.0, 3.0, 0.0, 2.0), (6.0, 4.0, 0.0, 0.0)],
             duration_raw=10.01,
             role="diagnostic_prehistory")["mode"] == "uniform_floor_subtick"
    must_fail(lambda: A([(6.0, 3.0, 0.0, 2.0)], duration_raw=10.01,
                        role="diagnostic_prehistory"))  # residual 1.01 tick
    must_fail(lambda: A([(8.0, 3.0, 0.0, 2.0)], duration_raw=10.01,
                        role="diagnostic_prehistory"))  # ceil, residual -0.99
    must_fail(lambda: A([(7.0, 3.0, 0.0, 2.0), (6.0, 3.0, 0.0, 2.0)],
                        duration_raw=10.01,
                        role="diagnostic_prehistory"))  # nonuniform
    fractional_fields = [
        (7.5, 2.5, 0.0, 2.0),  # fractional T0/T1
        (7.0, 2.5, 0.5, 2.0),  # fractional TX
        (7.0, 3.0, 0.0, 2.5),  # fractional TC
        (6.5, 3.5, 0.0, 2.0),  # second independent T0/T1 shape
    ]
    for values in fractional_fields:
        must_fail(lambda values=values: A([values], duration_raw=10.01,
                                          role="diagnostic_prehistory"))
    must_fail(lambda: A([(7.0, 2.99, 0.0, 2.0)], duration_raw=10.0,
                        role="measurement"))  # measurement residual 0.01
    must_fail(lambda: A([(6.0, 3.0, 0.0, 2.0)], duration_raw=10.0,
                        role="measurement"))  # measurement residual 1 tick

    with tempfile.TemporaryDirectory(prefix="m2201_mutations_") as raw:
        root = Path(raw)
        path = root / "measurement.saif"
        P.EXPECTED["records"] = 32
        good = T.saif_text(role="measurement", count=32)
        path.write_text(good)
        T.seal_file(path)
        assert P.parse_saif(path, role="measurement")["record_count"] == 32

        mutations = {
            "measurement_residual_0p01": good.replace(
                "(T0 60875.00)", "(T0 60874.99)", 1),
            "measurement_tx": T.saif_text(role="measurement", count=32, tx_first=1),
            "wrong_hierarchy": T.saif_text(role="measurement", count=32,
                                            target="impostor"),
            "missing_record": T.saif_text(role="measurement", count=31),
            "missing_critical": T.saif_text(role="measurement", count=32,
                                             mute_first_critical=True),
        }
        for mutation in mutations.values():
            path.write_text(mutation)
            T.seal_file(path)
            must_fail(lambda: P.parse_saif(path, role="measurement"))

    runner = RUNNER.read_text()
    assert all(token in runner for token in ("M2201", "M2202", "M2203", "M2204"))
    assert "M2203_ATTEMPT_CONSUMED" in runner
    assert "m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1" not in runner
    assert runner.count('"-debug_access+r"') == 1
    assert runner.count('"+M2160_AXIS_ORDINARY"') == 2
    assert "reuse_old_artifacts\": False" in runner

    print("PASS_M2201_SOURCE_TESTS inherited_m2185=1 real_controls=2 "
          "conservation_mutations=9 full_saif_mutations=5 m2187_raw_modified=0 "
          "vcs_runs=0 license_queries=0 eda_runs=0")


if __name__ == "__main__":
    main()

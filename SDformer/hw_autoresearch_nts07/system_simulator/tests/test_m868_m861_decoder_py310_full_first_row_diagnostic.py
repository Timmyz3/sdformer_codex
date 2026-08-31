"""Source-only tests for the M868 nonproduction one-row gate.

These tests never enumerate the full row, consume the canonical attempt, or
write in the repository results namespace.
"""

import importlib.util
import json
from pathlib import Path

import pytest


HW = Path(__file__).resolve().parents[2]
DRIVER = (HW / "system_simulator/scripts/execute_m868_m861_decoder_py310_full_first_row_diagnostic.py")
RUNNER = (HW / "system_simulator/scripts/run_m868_m861_decoder_py310_full_first_row_one_shot.sh")
CANDIDATE = (HW / "contracts/m868_m861_decoder_py310_full_first_row_diagnostic_candidate_r1_20260829.json")


def load_driver():
    spec = importlib.util.spec_from_file_location("m868_test_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_identity_has_no_shebang_or_ambient_fallback():
    source = DRIVER.read_text(encoding="utf-8")
    assert not source.startswith("#!")
    module = load_driver()
    assert str(module.PYTHON_PATH) == "/opt/anaconda3/envs/pytorch310/bin/python3.10"
    assert module.PYTHON_SHA256 == "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
    assert module.PYTHON_VERSION == "3.10.18"


def test_candidate_is_source_only_and_exactly_one_row():
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["launch_now"] is False
    assert candidate["max_attempts"] == 1
    assert candidate["workload"]["identity"] == "M854_FIRST_D0_A1_T0"
    assert candidate["workload"]["rows_authorized"] == 1
    assert candidate["workload"]["population_rows_authorized"] == 0
    assert candidate["authorization"]["run_full_first_row_now"] is False
    assert candidate["authorization"]["run_full_population"] is False
    assert candidate["claim_boundary"]["paper_citable"] is False


def test_runner_has_explicit_no_work_mode_and_one_way_namespaces():
    source = RUNNER.read_text(encoding="utf-8")
    assert "--dry-run-no-work" in source
    assert "M868_EXPECTED_HAMMER_REVIEW_SHA256" in source
    assert "M868_EXPECTED_HAMMER_OUTER_SHA256" in source
    assert "attempt_consumed" in source
    assert "failed_or_incomplete" in source
    assert "PUBLISH_CANONICAL_NOREPLACE" in source
    assert "RUN_EXACT_ONE_FULL_FIRST_ROW_DIAGNOSTIC" in source
    assert "run-production" not in source


def test_private_seal_and_noreplace_are_fail_closed(tmp_path):
    module = load_driver()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "member.json").write_text("{}\n", encoding="utf-8")
    identity = module.seal_directory(stage, ("member.json",))
    assert module.verify_sealed(stage) == identity
    destination = tmp_path / "canonical"
    module._rename_noreplace(stage, destination)
    assert not stage.exists()
    assert module.verify_sealed(destination) == identity
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "member.json").write_text("{}\n", encoding="utf-8")
    module.seal_directory(replacement, ("member.json",))
    with pytest.raises(module.Failure):
        module._rename_noreplace(replacement, destination)


def test_full_row_gate_is_not_reachable_without_future_hammer():
    module = load_driver()
    with pytest.raises((module.Failure, FileNotFoundError)):
        module.validate_hammer("0" * 64, "0" * 64)

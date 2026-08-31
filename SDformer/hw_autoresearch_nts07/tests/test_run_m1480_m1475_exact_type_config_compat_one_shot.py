from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1480_m1475_exact_type_config_compat_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1480_tested", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("value", [
    {"launch": 1, "runs": 1, "automatic_retry": False, "controller_restore": False},
    {"launch": True, "runs": True, "automatic_retry": False, "controller_restore": False},
    {"launch": True, "runs": 1.0, "automatic_retry": False, "controller_restore": False},
    {"launch": True, "runs": 1, "automatic_retry": 0, "controller_restore": False},
    {"launch": True, "runs": 1, "automatic_retry": False, "controller_restore": 0},
])
def test_rejects_m1476_type_confusion(value):
    with pytest.raises(MODULE.M1480Error):
        MODULE.exact_authorization(value, True)


def test_accepts_exact_launch_authorization():
    MODULE.exact_authorization({
        "launch": True, "runs": 1, "automatic_retry": False,
        "controller_restore": False}, True)


def test_accepts_exact_nonlaunch_authorization():
    MODULE.exact_authorization({
        "launch": False, "runs": 0, "automatic_retry": False,
        "controller_restore": False}, False)


@pytest.mark.parametrize("key", [
    "launch", "runs", "automatic_retry", "controller_restore"])
def test_rejects_missing_authorization_field(key):
    value = {"launch": True, "runs": 1, "automatic_retry": False,
             "controller_restore": False}
    value.pop(key)
    with pytest.raises(MODULE.M1480Error):
        MODULE.exact_authorization(value, True)


def test_rejects_extra_authorization_field():
    value = {"launch": True, "runs": 1, "automatic_retry": False,
             "controller_restore": False, "extra": False}
    with pytest.raises(MODULE.M1480Error):
        MODULE.exact_authorization(value, True)


def test_reuses_m1475_narrow_compatibility_and_m1458_namespaces():
    assert MODULE.M1475.M1458.CANONICAL_RESULT.name.startswith("m1458_m1434_")
    assert MODULE.M1475.M1458.CANONICAL_ATTEMPT.name.startswith(".m1458_m1434_")
    assert MODULE.M1475.M1458.CANONICAL_LOG.name.startswith(".m1458_m1434_")
    assert MODULE.M1475.configuration_content_compatibility

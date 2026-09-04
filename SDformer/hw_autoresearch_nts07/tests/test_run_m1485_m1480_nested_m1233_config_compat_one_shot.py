from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1485_tested", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_nested_patch_is_label_narrow_and_restored(monkeypatch):
    frozen_original = MODULE.ORIGINAL_M1233_EXACT_IDENTITY
    extended_original = MODULE.ORIGINAL_EXTENDED_EXACT_IDENTITY
    frozen_calls = []
    extended_calls = []
    marker = {"frozen": True}

    def frozen_strict(value, label):
        frozen_calls.append((value, label))
        return marker

    def extended_strict(value, label):
        extended_calls.append((value, label))
        return marker

    monkeypatch.setattr(MODULE, "ORIGINAL_M1233_EXACT_IDENTITY", frozen_strict)
    monkeypatch.setattr(MODULE, "ORIGINAL_EXTENDED_EXACT_IDENTITY", extended_strict)
    monkeypatch.setattr(MODULE.FROZEN_M1233, "exact_identity", frozen_strict)
    monkeypatch.setattr(MODULE.M1319, "exact_extended_identity", extended_strict)
    monkeypatch.setattr(MODULE.M1480.M1475,
                        "verify_configuration_content_identity",
                        lambda value: dict(value))
    config = dict(MODULE.M1480.M1475.FROZEN_CONFIG_ENTITY)
    with MODULE.dual_configuration_compatibility():
        assert MODULE.M1319.exact_extended_identity(
            config, "selected configuration") == config
        assert MODULE.FROZEN_M1233.exact_identity(
            config, "selected configuration") == config
        assert MODULE.FROZEN_M1233.exact_identity(
            {}, "selected checkpoint") is marker
        assert MODULE.FROZEN_M1233.exact_identity(
            {}, "selected profile") is marker
    assert MODULE.FROZEN_M1233.exact_identity is frozen_strict
    assert MODULE.M1319.exact_extended_identity is extended_strict
    assert frozen_calls == [({}, "selected checkpoint"), ({}, "selected profile")]
    assert extended_calls == []


def test_nested_and_m1475_contexts_coexist(monkeypatch):
    config = dict(MODULE.M1480.M1475.FROZEN_CONFIG_ENTITY)
    monkeypatch.setattr(MODULE.M1480.M1475,
                        "verify_configuration_content_identity",
                        lambda value: dict(value))
    with MODULE.dual_configuration_compatibility():
        assert MODULE.M1480.M1475.M1319.exact_extended_identity(
            config, "selected configuration") == config
        assert MODULE.FROZEN_M1233.exact_identity(
            config, "selected configuration") == config


def test_context_restores_after_exception():
    frozen_original = MODULE.ORIGINAL_M1233_EXACT_IDENTITY
    extended_original = MODULE.ORIGINAL_EXTENDED_EXACT_IDENTITY
    with pytest.raises(RuntimeError):
        with MODULE.dual_configuration_compatibility():
            raise RuntimeError("attack")
    assert MODULE.FROZEN_M1233.exact_identity is frozen_original
    assert MODULE.M1319.exact_extended_identity is extended_original


def test_context_restores_both_after_inner_tamper():
    frozen_original = MODULE.ORIGINAL_M1233_EXACT_IDENTITY
    extended_original = MODULE.ORIGINAL_EXTENDED_EXACT_IDENTITY
    with pytest.raises(MODULE.M1485Error):
        with MODULE.dual_configuration_compatibility():
            MODULE.FROZEN_M1233.exact_identity = lambda value, label: value
            MODULE.M1319.exact_extended_identity = lambda value, label: value
    assert MODULE.FROZEN_M1233.exact_identity is frozen_original
    assert MODULE.M1319.exact_extended_identity is extended_original


@pytest.mark.parametrize("field,replacement", [
    ("size_bytes", 6481.0),
    ("mtime_ns", 1788081356000000000.0),
    ("device", 194.0),
    ("inode", 26561699333.0),
    ("mode", 33152.0),
    ("absolute_path", 1),
    ("sha256", True),
])
def test_frozen_config_rejects_type_confusion(field, replacement):
    value = dict(MODULE.M1480.M1475.FROZEN_CONFIG_ENTITY)
    value[field] = replacement
    with pytest.raises(MODULE.M1485Error):
        MODULE.verify_frozen_config_entity_exact_type(value)


@pytest.mark.parametrize("value", [
    {"launch": 1, "runs": 1, "automatic_retry": False,
     "controller_restore": False},
    {"launch": True, "runs": True, "automatic_retry": False,
     "controller_restore": False},
    {"launch": True, "runs": 1.0, "automatic_retry": False,
     "controller_restore": False},
    {"launch": True, "runs": 1, "automatic_retry": 0,
     "controller_restore": False},
    {"launch": True, "runs": 1, "automatic_retry": False,
     "controller_restore": 0},
])
def test_authorization_rejects_type_confusion(value):
    with pytest.raises(MODULE.M1485Error):
        MODULE.exact_authorization(value, True)


def test_authorization_accepts_exact_types():
    MODULE.exact_authorization({
        "launch": True, "runs": 1, "automatic_retry": False,
        "controller_restore": False}, True)

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1489_m1485_m1434_export_alias_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("m1489_tested", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exports_exact_aliases_and_restores():
    with MODULE.export_digest_aliases():
        assert MODULE.M1434.PROFILE_SOURCE_SHA256 == MODULE.PROFILE_SHA256
        assert MODULE.M1434.ATLIF_OVERLAY_SOURCE_SHA256 == MODULE.ATLIF_SHA256
    assert not hasattr(MODULE.M1434, "PROFILE_SOURCE_SHA256")
    assert not hasattr(MODULE.M1434, "ATLIF_OVERLAY_SOURCE_SHA256")


def test_restores_after_exception():
    with pytest.raises(RuntimeError):
        with MODULE.export_digest_aliases():
            raise RuntimeError("attack")
    assert not hasattr(MODULE.M1434, "PROFILE_SOURCE_SHA256")
    assert not hasattr(MODULE.M1434, "ATLIF_OVERLAY_SOURCE_SHA256")


def test_restores_before_tamper_error():
    with pytest.raises(MODULE.M1489Error):
        with MODULE.export_digest_aliases():
            MODULE.M1434.PROFILE_SOURCE_SHA256 = "0" * 64
    assert not hasattr(MODULE.M1434, "PROFILE_SOURCE_SHA256")
    assert not hasattr(MODULE.M1434, "ATLIF_OVERLAY_SOURCE_SHA256")


@pytest.mark.parametrize("name", [
    "PROFILE_SOURCE_SHA256", "ATLIF_OVERLAY_SOURCE_SHA256"])
def test_rejects_preinstalled_alias(monkeypatch, name):
    monkeypatch.setattr(MODULE.M1434, name, "0" * 64, raising=False)
    with pytest.raises(MODULE.M1489Error):
        MODULE.validate_bootstrap()

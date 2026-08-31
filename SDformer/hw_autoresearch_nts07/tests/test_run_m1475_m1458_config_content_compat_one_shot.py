from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1475_m1458_config_content_compat_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1475_tested", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen(path: Path, payload: bytes) -> dict:
    path.write_bytes(payload)
    observed = path.lstat()
    return {
        "absolute_path": str(path),
        "size_bytes": len(payload),
        "mtime_ns": 1788081356000000000,
        "sha256": MODULE.hashlib.sha256(payload).hexdigest(),
        "device": 194,
        "inode": 26561699333,
        "mode": 33152,
    }


def configure(monkeypatch, tmp_path, payload=b"a" * 6481):
    path = tmp_path / "config.yml"
    value = frozen(path, payload)
    monkeypatch.setattr(MODULE, "CONFIG_PATH", path)
    monkeypatch.setattr(MODULE, "CONFIG_ABSOLUTE", str(path))
    monkeypatch.setattr(MODULE, "CONFIG_SIZE", len(payload))
    monkeypatch.setattr(MODULE, "CONFIG_SHA256",
                        MODULE.hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(MODULE, "FROZEN_CONFIG_ENTITY", dict(value))
    return path, value


def test_accepts_content_exact_entity_drift(monkeypatch, tmp_path):
    path, value = configure(monkeypatch, tmp_path)
    path.chmod(0o644)
    assert MODULE.verify_configuration_content_identity(value) == value


@pytest.mark.parametrize("field,replacement", [
    ("absolute_path", "/tmp/wrong"), ("size_bytes", 6480),
    ("mtime_ns", 1), ("sha256", "0" * 64), ("device", 1),
    ("inode", 1), ("mode", stat.S_IFREG | 0o644),
])
def test_rejects_frozen_selection_drift(monkeypatch, tmp_path, field, replacement):
    _path, value = configure(monkeypatch, tmp_path)
    value[field] = replacement
    with pytest.raises(MODULE.M1475Error):
        MODULE.verify_configuration_content_identity(value)


def test_rejects_content_drift(monkeypatch, tmp_path):
    path, value = configure(monkeypatch, tmp_path)
    path.write_bytes(b"b" * 6481)
    with pytest.raises(MODULE.M1475Error):
        MODULE.verify_configuration_content_identity(value)


def test_rejects_size_drift(monkeypatch, tmp_path):
    path, value = configure(monkeypatch, tmp_path)
    path.write_bytes(b"a" * 6480)
    with pytest.raises(MODULE.M1475Error):
        MODULE.verify_configuration_content_identity(value)


def test_rejects_symlink(monkeypatch, tmp_path):
    target, value = configure(monkeypatch, tmp_path)
    link = tmp_path / "link.yml"
    link.symlink_to(target)
    monkeypatch.setattr(MODULE, "CONFIG_PATH", link)
    monkeypatch.setattr(MODULE, "CONFIG_ABSOLUTE", str(link))
    value["absolute_path"] = str(link)
    monkeypatch.setattr(MODULE, "FROZEN_CONFIG_ENTITY", value)
    with pytest.raises(MODULE.M1475Error):
        MODULE.verify_configuration_content_identity(value)


def test_patch_is_label_narrow_and_restored(monkeypatch, tmp_path):
    _path, value = configure(monkeypatch, tmp_path)
    calls = []

    def original(item, label):
        calls.append((item, label))
        return "strict"

    monkeypatch.setattr(MODULE, "ORIGINAL_EXACT_EXTENDED_IDENTITY", original)
    monkeypatch.setattr(MODULE.M1319, "exact_extended_identity", original)
    with MODULE.configuration_content_compatibility():
        assert MODULE.M1319.exact_extended_identity(value, "selected configuration") == value
        assert MODULE.M1319.exact_extended_identity({}, "selected checkpoint") == "strict"
    assert MODULE.M1319.exact_extended_identity is original
    assert calls == [({}, "selected checkpoint")]


def test_context_restores_after_exception(monkeypatch):
    original = MODULE.ORIGINAL_EXACT_EXTENDED_IDENTITY
    monkeypatch.setattr(MODULE.M1319, "exact_extended_identity", original)
    with pytest.raises(RuntimeError):
        with MODULE.configuration_content_compatibility():
            raise RuntimeError("attack")
    assert MODULE.M1319.exact_extended_identity is original

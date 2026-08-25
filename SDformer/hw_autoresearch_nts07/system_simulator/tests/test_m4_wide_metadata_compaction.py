from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m4_wide_metadata_compaction.py"
)
SPEC = spec_from_file_location("m4_wide_metadata_compaction", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_default_wide_metadata_bit_reduction() -> None:
    result = MODULE.analyze()
    assert result["metadata_bits_per_row"] == 91
    assert result["legacy_per_bank_metadata_bits"] == 69_888
    assert result["shared_wide_metadata_bits"] == 11_648
    assert result["metadata_bit_reduction"] == 58_240
    assert result["metadata_reduction_fraction"] == 5 / 6
    assert result["state_data_bits_unchanged"] == 393_216
    assert (
        result["persistent_destination_data_plus_metadata_reduction_fraction"]
        == 58_240 / 463_104
    )


def test_rejects_nonpositive_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        MODULE.analyze(banks=0)

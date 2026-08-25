from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_m4_wide_metadata_ab.py"
)
SPEC = spec_from_file_location("m4_wide_metadata_ab", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def point(cycles=100, stalls=(0, 0, 0, 0)):
    return {
        "streaming_cycles": cycles,
        "request_beats": 50,
        "bank_reads": 200,
        "request_stalls": stalls[0],
        "output_stalls": stalls[1],
        "rmw_backpressure_cycles": stalls[2],
        "rmw_backpressure_cover_max": stalls[3],
    }


def test_cycle_neutral_bit_reduction() -> None:
    result = MODULE.summarize(
        point(), point(), point(None, (4, 5, 6, 6)),
        point(None, (4, 5, 6, 6)),
        {
            "status": "PASS_M4_WIDE_METADATA_BIT_AUDIT_PRE_DC",
            "legacy_per_bank_metadata_bits": 69_888,
            "shared_wide_metadata_bits": 11_648,
            "metadata_bit_reduction": 58_240,
            "metadata_reduction_fraction": 5 / 6,
            "persistent_destination_data_plus_metadata_reduction_fraction":
                0.125,
        },
    )
    assert result["bounded_workload_functional_cycle_match"]
    assert result["metadata_bits"]["reduction"] == 58_240


def test_rejects_cycle_mismatch() -> None:
    with pytest.raises(ValueError, match="streaming_cycles"):
        MODULE.summarize(
            point(100), point(101), point(None, (4, 5, 6, 6)),
            point(None, (4, 5, 6, 6)),
            {
                "status": "PASS_M4_WIDE_METADATA_BIT_AUDIT_PRE_DC",
                "legacy_per_bank_metadata_bits": 1,
                "shared_wide_metadata_bits": 1,
                "metadata_bit_reduction": 0,
                "metadata_reduction_fraction": 0,
                "persistent_destination_data_plus_metadata_reduction_fraction": 0,
            },
        )


def test_rejects_random_ab_or_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="request_stalls"):
        MODULE.summarize(
            point(), point(), point(None, (4, 5, 6, 6)),
            point(None, (7, 5, 6, 6)),
            {
                "status": "PASS_M4_WIDE_METADATA_BIT_AUDIT_PRE_DC",
                "legacy_per_bank_metadata_bits": 1,
                "shared_wide_metadata_bits": 1,
                "metadata_bit_reduction": 0,
                "metadata_reduction_fraction": 0,
                "persistent_destination_data_plus_metadata_reduction_fraction": 0,
            },
        )


def test_rejects_bad_bit_audit_status() -> None:
    with pytest.raises(ValueError, match="bit audit"):
        MODULE.summarize(
            point(), point(), point(None, (4, 5, 6, 6)),
            point(None, (4, 5, 6, 6)),
            {"status": "FAIL"},
        )

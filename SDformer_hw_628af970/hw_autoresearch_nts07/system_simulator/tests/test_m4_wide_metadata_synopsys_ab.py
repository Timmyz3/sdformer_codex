from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_m4_wide_metadata_synopsys_ab.py"
SPEC = spec_from_file_location("m4_wide_metadata_synopsys_ab", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def point(area: float, sequential: int) -> dict:
    return {
        "identity": {"same": True},
        "metrics": {
            "total_cell_area_um2": area,
            "sequential_cells": sequential,
        },
    }


def test_reports_reduction_without_paper_admission() -> None:
    result = MODULE.summarize(point(40.0, 60), point(100.0, 120))
    assert result["status"] == "PASS_PREMACRO_LOGIC_ONLY_DC_FORMALITY_AB"
    assert not result["paper_ppa_admitted"]
    assert result["comparison"]["cell_area_reduction_fraction"] == 0.6
    assert result["comparison"]["sequential_cell_reduction"] == 60


def test_rejects_identity_mismatch() -> None:
    shared = point(40.0, 60)
    legacy = point(100.0, 120)
    legacy["identity"] = {"same": False}
    with pytest.raises(ValueError, match="identities"):
        MODULE.summarize(shared, legacy)


def test_rejects_non_improving_ab() -> None:
    with pytest.raises(ValueError, match="did not reduce"):
        MODULE.summarize(point(100.0, 120), point(100.0, 120))


def test_parameter_parser_is_fail_closed() -> None:
    assert MODULE.parse_parameters(
        "STATE_QUEUE_DEPTH=1,USE_SHARED_WIDE_METADATA=0"
    ) == {"STATE_QUEUE_DEPTH": "1", "USE_SHARED_WIDE_METADATA": "0"}
    with pytest.raises(ValueError, match="malformed"):
        MODULE.parse_parameters("STATE_QUEUE_DEPTH")

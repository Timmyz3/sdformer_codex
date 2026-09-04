from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_m4_state_queue_streaming_dse.py"
)
SPEC = spec_from_file_location("m4_state_queue_streaming_dse", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_streaming_dse_keeps_q1_as_area_candidate() -> None:
    result = MODULE.summarize([
        {"queue_depth": 1, "payload_bits": 3170,
         "streaming_cycles": 285_235},
        {"queue_depth": 2, "payload_bits": 6340,
         "streaming_cycles": 282_979},
    ])
    assert result["candidate"] == "Q1_PREMACRO_PENDING_LOGIC_AREA"
    assert result["throughput_ablation"] == 2
    assert result["points"][1]["speedup_vs_q1"] == 285_235 / 282_979
    assert result["points"][1]["cycle_reduction_vs_q1_fraction"] < 0.008

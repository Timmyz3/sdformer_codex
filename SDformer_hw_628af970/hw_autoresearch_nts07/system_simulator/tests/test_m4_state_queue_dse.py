from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_m4_state_queue_dse.py"
)
SPEC = spec_from_file_location("m4_state_queue_dse", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def point(depth, hybrid):
    return (
        depth,
        Path(f"depth{depth}.json"),
        {
            "status": "PASS_M4_STATEFUL_PAIRED_VCS_CORE_CYCLES",
            "population": {"sequences": 4},
            "overall": {
                "local_cycles": 120,
                "hybrid_cycles": hybrid,
                "hybrid_regression_pairs": 0,
            },
        },
    )


def test_queue_dse_prunes_deeper_equal_cycle_points(monkeypatch):
    monkeypatch.setattr(MODULE, "sha256", lambda _path: "0" * 64)
    result = MODULE.summarize([point(1, 105), point(2, 100), point(4, 100)])
    assert result["entry_bits"] == 3170
    assert result["smallest_area_proxy_candidate_depth"] == 1
    assert result["minimum_depth_at_best_cycles"] == 2
    assert result["points"][2]["dominated_by_shallower_equal_cycle_point"]

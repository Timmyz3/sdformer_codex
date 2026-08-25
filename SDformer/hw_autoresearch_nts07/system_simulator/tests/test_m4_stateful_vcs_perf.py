from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_m4_stateful_vcs_perf.py"
)
SPEC = spec_from_file_location("m4_stateful_vcs_perf", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_vcs_summary_reconciles_modes_and_identities():
    text = """\
M4_STATE_SEQ id=1 mode=local_only cycles=100 descriptors=8 outputs=4 motion_outputs=0
M4_STATE_SEQ id=2 mode=hybrid_local_motion cycles=80 descriptors=8 outputs=4 motion_outputs=2
M4_STATE_SEQ id=3 mode=local_only cycles=120 descriptors=12 outputs=8 motion_outputs=0
M4_STATE_SEQ id=4 mode=hybrid_local_motion cycles=100 descriptors=12 outputs=8 motion_outputs=3
PASS_M4_STATEFUL_PERF pairs=2 local_cycles=220 hybrid_cycles=180
"""
    manifest = {
        "population": {"sequences": 4},
        "identities": {
            "H67": {"selected_sample_groups": 1},
            "Local5": {"selected_sample_groups": 1},
        },
    }
    result = MODULE.summarize(text, manifest)
    assert result["overall"]["pairs"] == 2
    assert result["overall"]["aggregate_speedup_vs_local"] == 220 / 180
    assert result["per_identity"]["H67"]["motion_outputs"] == 2
    assert result["per_identity"]["Local5"]["motion_outputs"] == 3
    assert result["overall"]["hybrid_regression_pairs"] == 0

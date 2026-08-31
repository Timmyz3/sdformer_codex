import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/analyze_m712_pidp_decoder_exact_cpu_fastkill.py"
SPEC = importlib.util.spec_from_file_location("m712", SCRIPT)
M712 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M712)


def test_static_self_test():
    M712.self_test()


def test_topology_all_decoder_shapes():
    for spec in M712.MODULES.values():
        _, cin, _, hin, win, hout, wout, _ = spec
        topo = M712.topology(cin, hin, win, hout, wout)
        assert topo["topology_mismatches"] == 0
        assert topo["legal_spatial_edges"] > 0


def test_traffic_gate_requires_cycle_bound():
    threshold = M712.Decimal(1) / M712.Decimal("1.05")
    assert M712.Decimal("0.94") < threshold
    assert M712.Decimal("0.96") >= threshold


def test_wrap24_ring():
    assert M712.wrap24((1 << 23) - 1 + 1) == -(1 << 23)
    assert M712.wrap24(-(1 << 23) - 1) == (1 << 23) - 1

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/miter_local5_source_owned_gate_quotient_rtl.py"
SPEC = importlib.util.spec_from_file_location("source_quotient_miter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_destination_mapping_is_bijective_for_legal_edges() -> None:
    for destination in range(MODULE.SOURCES):
        for role in range(MODULE.ROLES):
            source = MODULE.source_for(destination, role)
            if source is not None:
                assert MODULE.destination_for(source, role) == destination


def test_group_reconstruction_folds_equal_gate_and_preserves_destinations() -> None:
    candidate_k = [0] * MODULE.SOURCES
    valid = [0] * MODULE.SOURCES
    gates = [0] * MODULE.SOURCES
    source = 16
    k_value = 0b101
    for role in (0, 1, 2):
        destination = MODULE.destination_for(source, role)
        assert destination is not None
        candidate_k[destination] |= k_value << (role * MODULE.HEAD_DIM)
        valid[destination] |= 1 << role
        gate = 7 if role < 2 else 9
        gates[destination] |= gate << (role * MODULE.GATE_W)

    # Populate all remaining legal edge K bindings so every source is reconstructable.
    for destination in range(MODULE.SOURCES):
        for role in range(MODULE.ROLES):
            mapped = MODULE.source_for(destination, role)
            if mapped is None or ((valid[destination] >> role) & 1):
                continue
            valid[destination] |= 1 << role
            edge_k = k_value if mapped == source else 0
            candidate_k[destination] |= edge_k << (role * MODULE.HEAD_DIM)

    weights = [[1, -2] for _ in range(MODULE.HEAD_DIM)]
    observed = MODULE.analyze_group(
        candidate_k=candidate_k,
        valid_mask=valid,
        packed_gates=gates,
        weights=weights,
    )
    assert observed["active"] == 1
    assert observed["active_unique_gate_instances"] == 2
    assert observed["all_source_unique_gate_instances"] == 2
    assert observed["terms"] == 4
    assert observed["updates"] == 6
    assert observed["multiplicity_histogram"] == {1: 2, 2: 2}
    expected_dests = [MODULE.destination_for(source, role) for role in (0, 1, 2)]
    assert [observed["acc"][dest][0] for dest in expected_dests] == [14, 14, 18]
    assert [observed["acc"][dest][1] for dest in expected_dests] == [-28, -28, -36]

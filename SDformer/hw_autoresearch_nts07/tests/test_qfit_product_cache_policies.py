import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "policies",
    ROOT / "scripts/evaluate_qfit_product_cache_policies.py",
)
assert SPEC and SPEC.loader
policies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policies)


def test_online_policies_and_oracle_ordering():
    rows = policies.load_rows()
    lru4 = policies.simulate_lru(rows, 4)
    first4 = policies.simulate_no_replace(rows, 4)
    oracle4 = policies.simulate_belady(rows, 4)
    static4 = policies.evaluate_static(
        rows, 4, policies.global_codebook(rows, 4), "static"
    )
    assert lru4["product_starts"] == 499
    assert first4["product_starts"] == 397
    assert first4["fills"] + first4["bypasses"] == 397
    assert oracle4["product_starts"] <= min(
        lru4["product_starts"],
        first4["product_starts"],
        static4["product_starts"],
    )


def test_same_trace_static_codebook_matches_existing_profile():
    rows = policies.load_rows()
    codebook = policies.global_codebook(rows, 4)
    value = policies.evaluate_static(rows, 4, codebook, "static")
    assert codebook == [31, 32, 29, 15]
    assert value["product_starts"] == 262


def test_exception_split_is_smaller_but_fixed_packet_is_not():
    rows = policies.load_rows()
    first4 = policies.simulate_no_replace(rows, 4)
    bits = policies.bundle_bits_dynamic(
        len(rows),
        int(first4["fills"]),
        int(first4["bypasses"]),
        4,
    )
    assert bits["fixed_packet_bits"] > bits["baseline_gate_bits"]
    assert bits["exception_split_bits"] < bits["baseline_gate_bits"]


def test_descriptor_gate_dictionary_matches_ordered_trace():
    rows = policies.load_rows()
    value = policies.descriptor_gate_dictionary(rows)
    assert value["descriptors"] == 36
    assert value["dictionary_entries"] == 91
    assert value["max_dictionary_entries"] == 4
    assert value["baseline_total_bits"] == 55_278
    assert value["ideal_dictionary_total_bits"] == 22_455
    assert value["variable_safe_total_bits"] == 22_743
    assert value["fixed_header_total_bits"] == 23_544


def test_gate_stationary_reorder_preserves_terms_and_cache_sequence():
    rows = policies.load_rows()
    reordered, stats = policies.gate_stationary_reorder(rows)
    assert len(reordered) == len(rows)
    assert stats["gate_stationary"]["gate_transitions"] < stats["original"][
        "gate_transitions"
    ]
    for ways in (4, 6, 8):
        original = policies.simulate_lru(rows, ways)
        changed = policies.simulate_lru(reordered, ways)
        assert original["product_starts"] == changed["product_starts"]

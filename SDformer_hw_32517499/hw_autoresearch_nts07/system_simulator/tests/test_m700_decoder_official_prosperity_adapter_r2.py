#!/usr/bin/env python3
"""Static/unit tests for M700; never call M672 or official Simulator.run_fc."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "hw_autoresearch_nts07" / "scripts" /
    "run_m700_h67_ep35_decoder_official_prosperity_iso_workload_r2.py"
)
SPEC = importlib.util.spec_from_file_location("m700_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def counters(seed: int = 1):
    result = {field: seed for field in M.COUNTER_FIELDS}
    result["total_cycles"] = seed * 2
    result["compute_cycles"] = seed
    result["memory_stall_cycles"] = seed
    return result


def sealed_review(tmp_path: Path, payload: dict):
    directory = tmp_path / "review"
    directory.mkdir()
    (directory / "review.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    M.write_double_seal(directory)
    return directory


def frozen_target():
    return {
        "runner_sha256": M.sha256_file(SCRIPT),
        "contract_sha256": M.sha256_file(M.CONTRACT),
        "test_sha256": M.sha256_file(Path(__file__).resolve()),
        "m686_manifest_sha256":
            "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33",
        "m692_review_sha256":
            "5088e36fa935536766f51f4e58c198d16f49ac3fe415b2f3d6432b184a36f49f",
        "m697_review_sha256":
            "f5fd5a172cd011654224aa0591df30518c0753d9b563f88535ff42ad39188dd1",
    }


def test_import_is_inert_for_mapper_and_official_api():
    assert M._MAPPER is None
    assert M._FC is None
    assert M._SIMULATOR is None
    assert M._ACCELERATOR is None


def test_frozen_constants():
    assert M.PHASE_ORDER == (3, 2, 1, 0)
    assert M.EXACT_MODULES == (0, 2, 3)
    assert M.DIAGNOSTIC_MODULES == (1,)
    assert (M.M_TILE, M.K_TILE, M.N_TILE, M.MEM_IF_WIDTH) == (256, 16, 128, 1024)


def test_frozen_preflight_binds_m692_m686_without_importing_engines():
    contract = M.strict_json(M.CONTRACT)
    identity, exact_records, diagnostic_records = M.preflight(contract)
    assert len(exact_records) == 30
    assert len(diagnostic_records) == 10
    assert identity["m692"]["review_sha256"] == \
        "5088e36fa935536766f51f4e58c198d16f49ac3fe415b2f3d6432b184a36f49f"
    assert identity["m686_package"]["manifest_sha256"] == \
        "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33"
    assert identity["m697_review"]["sha256"] == \
        "f5fd5a172cd011654224aa0591df30518c0753d9b563f88535ff42ad39188dd1"
    assert [row["module_index"] for row in exact_records[:3]] == [0, 2, 3]
    assert all(row["admission_role"] ==
               "SCALED_BINARY_OPPORTUNITY_DIAGNOSTIC_ONLY"
               for row in diagnostic_records)
    assert M._MAPPER is None and M._SIMULATOR is None


def test_tiny_non_square_polyphase_numeric_miter(tmp_path):
    contract = M.strict_json(M.CONTRACT)
    mapper = M.load_mapper(contract)
    shape = (2, 1, 2, 2, 3)
    values = np.asarray([
        [[[[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1]]]],
        [[[[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 1]]]],
    ], dtype=np.uint8)
    assert values.shape == shape
    payload = np.packbits(values.reshape(-1), bitorder="little")
    bitpack = tmp_path / "tiny.bitpack"
    bitpack.write_bytes(payload.tobytes())
    weight = (np.arange(2 * 3 * 3 * 3, dtype=np.int64).reshape(2, 3, 3, 3) % 7) - 3
    observed = mapper.reconstruct_convtranspose(
        bitpack, shape, weight, tile_m=2, trusted_root=tmp_path
    )
    expected = np.zeros((2, 3, 4, 6), dtype=np.int64)
    for t in range(shape[0]):
        for cin in range(shape[2]):
            for sy in range(shape[3]):
                for sx in range(shape[4]):
                    for ky in range(3):
                        for kx in range(3):
                            dy, dx = 2 * sy - 1 + ky, 2 * sx - 1 + kx
                            if 0 <= dy < 4 and 0 <= dx < 6:
                                expected[t, :, dy, dx] += (
                                    int(values[t, 0, cin, sy, sx]) *
                                    weight[cin, :, ky, kx]
                                )
    np.testing.assert_array_equal(observed, expected)
    assert M._SIMULATOR is None and M._FC is None


@pytest.mark.parametrize("member", ["../x", "/tmp/x", "a/../../b", ".", ""])
def test_safe_member_rejects_escape(member):
    with pytest.raises(RuntimeError):
        M.safe_member(member)


def test_safe_member_accepts_nested_payload():
    assert M.safe_member("calls/s00_d0.activation.le.bitpack").as_posix() == \
        "calls/s00_d0.activation.le.bitpack"


def test_strict_json_rejects_duplicate(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        M.strict_json(path)


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-standard"):
        M.strict_json(path)


def test_double_seal_round_trip(tmp_path):
    directory = tmp_path / "sealed"
    directory.mkdir()
    (directory / "a.txt").write_text("a\n", encoding="utf-8")
    nested = directory / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b\n", encoding="utf-8")
    M.write_double_seal(directory)
    result = M.verify_double_seal(directory)
    assert len(result["manifest_file_sha256"]) == 64
    assert len(result["outer_seal_file_sha256"]) == 64


def test_double_seal_rejects_tamper(tmp_path):
    directory = tmp_path / "sealed"
    directory.mkdir()
    target = directory / "a.txt"
    target.write_text("a\n", encoding="utf-8")
    M.write_double_seal(directory)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="member SHA"):
        M.verify_double_seal(directory)


def test_double_seal_rejects_unsealed_extra(tmp_path):
    directory = tmp_path / "sealed"
    directory.mkdir()
    (directory / "a.txt").write_text("a\n", encoding="utf-8")
    M.write_double_seal(directory)
    (directory / "extra.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="population"):
        M.verify_double_seal(directory)


def test_trusted_file_rejects_leaf_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("x", encoding="utf-8")
    (root / "link").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        M.trusted_file(root, "link", "attack")


def test_popcount_little_bit_and_tail(tmp_path):
    path = tmp_path / "bits"
    path.write_bytes(bytes([0b00010101]))
    assert M.popcount_file(path, 5) == (3, 1)
    path.write_bytes(bytes([0b10010101]))
    with pytest.raises(RuntimeError, match="tail"):
        M.popcount_file(path, 5)


def test_expand_n128_rejects_partial_n():
    with pytest.raises(RuntimeError, match="integral"):
        M.expand_exact_n128(counters(), m_dim=3000, k_dim=6144, n_dim=192)


def test_expand_n128_exact_counter_scaling():
    base = counters(3)
    base["dram_reads"] = 100000
    base["dram_writes"] = 0
    expanded = M.expand_exact_n128(base, m_dim=3000, k_dim=6144, n_dim=384)
    assert expanded["compute_cycles"] == base["compute_cycles"] * 3
    assert expanded["num_ops"] == base["num_ops"] * 3
    assert expanded["total_cycles"] == (
        expanded["compute_cycles"] + expanded["memory_stall_cycles"]
    )


def test_derived_requires_no_performance_interpretation():
    row = counters(4)
    row["num_ops"] = 384 * 7
    derived = M.derived(row, 384)
    assert derived["support_nnz"] == 7
    assert derived["support_nnz_divisible_by_output_N"] is True


def fake_result(sample: int, module: int, bit_cycles: int, product_cycles: int):
    phases = []
    for bank in M.PHASE_ORDER:
        bit = M.derived(counters(2), 96)
        product = M.derived(counters(1), 96)
        bit["total_cycles"] = bit_cycles + bank
        product["total_cycles"] = product_cycles + bank
        phases.append({
            "phase_bank": bank,
            "modes": {"bit": bit, "product": product},
            "support_accounting": {
                "active_tap_events": 10 + sample + module + bank,
                "valid_tap_slots_all_time": 100 + bank,
            },
            "product_vs_bit_speedup": bit["total_cycles"] /
                product["total_cycles"],
        })
    return {"sample_id": sample, "module_index": module, "phases": phases}


def test_aggregate_uses_ratio_of_summed_cycles():
    rows = [fake_result(0, 0, 20, 10), fake_result(1, 0, 30, 10)]
    result = M.aggregate(rows)
    assert result["support_calls_per_mode"] == 8
    expected = result["bit"]["total_cycles"] / result["product"]["total_cycles"]
    assert result["aggregate_cycle_ratio_speedup"] == expected


def test_phase_aggregates_are_separate_and_conserve_to_overall():
    rows = [fake_result(0, 0, 20, 10), fake_result(1, 2, 30, 11)]
    result = M.aggregate_breakdowns(rows)
    assert all(f"phase:{bank}" in result for bank in M.PHASE_ORDER)
    assert result["overall"]["support_calls_per_mode"] == 8
    assert all(result[f"phase:{bank}"]["support_calls_per_mode"] == 2
               for bank in M.PHASE_ORDER)
    for mode in ("bit", "product"):
        for field, expected in result["overall"][mode].items():
            assert sum(result[f"phase:{bank}"][mode][field]
                       for bank in M.PHASE_ORDER) == expected
    for field, expected in result["overall"]["mapped_support_accounting"].items():
        assert sum(result[f"phase:{bank}"]["mapped_support_accounting"][field]
                   for bank in M.PHASE_ORDER) == expected
    for bank in M.PHASE_ORDER:
        phase = result[f"phase:{bank}"]
        assert phase["phase_bank"] == bank
        expected = phase["bit"]["total_cycles"] / \
            phase["product"]["total_cycles"]
        assert phase["aggregate_cycle_ratio_speedup"] == expected


def test_execution_authorization_requires_exact_status(tmp_path):
    directory = sealed_review(tmp_path, {
        "status": "NO_GO", "severity": {"p0": 0, "p1": 0, "p2": 0},
        "go": False, "execution_authorized": False,
        "frozen_target": frozen_target(),
    })
    outer_sha = M.sha256_file(directory / "SHA256SUMS.seal.sha256")
    with pytest.raises(RuntimeError, match="does not authorize"):
        M.validate_execution_authorization(directory, outer_sha)


def test_execution_authorization_accepts_exact_go(tmp_path):
    directory = sealed_review(tmp_path, {
        "status": "GO_M700_FULL_OFFICIAL_CPU_REPLAY__P0_0_P1_0",
        "severity": {"p0": 0, "p1": 0, "p2": 1},
        "go": True, "execution_authorized": True,
        "frozen_target": frozen_target(),
    })
    outer_sha = M.sha256_file(directory / "SHA256SUMS.seal.sha256")
    result = M.validate_execution_authorization(directory, outer_sha)
    assert result["review_sha256"] == M.sha256_file(directory / "review.json")


def test_build_report_keeps_exact_decoder_complete_null():
    exact_rows = [fake_result(0, 0, 20, 10)]
    diag_rows = [fake_result(0, 1, 50, 13)]
    contract = {
        "official_configuration": {"type": "Prosperity"},
        "mapping": {"phase_order": [3, 2, 1, 0]},
        "claim_boundary": {
            "ours": False, "full_decoder_latency": False,
            "system_speedup": False,
        },
    }
    report = M.build_report(contract, {}, exact_rows, diag_rows, {})
    assert report["exact_decoder_complete"]["admitted"] is False
    assert report["exact_decoder_complete"]["total_cycles"] is None
    assert report["exact_decoder_complete"]["product_vs_bit_speedup"] is None
    assert report["d1_scaled_binary_opportunity_diagnostic"][
        "folded_weight_deployment_admitted"] is False
    assert "overall" in report["official_binary_support_subset"]["aggregates"]
    assert "record:s00_d0" in report["official_binary_support_subset"]["aggregates"]
    assert all(f"phase:{bank}" in
               report["official_binary_support_subset"]["aggregates"]
               for bank in M.PHASE_ORDER)
    assert all(f"phase:{bank}" in
               report["d1_scaled_binary_opportunity_diagnostic"]["aggregates"]
               for bank in M.PHASE_ORDER)
    assert report["official_binary_support_subset"]["aggregates"]["phase:3"][
        "bit"]["total_cycles"] != \
        report["d1_scaled_binary_opportunity_diagnostic"]["aggregates"][
            "phase:3"]["bit"]["total_cycles"]
    assert report["execution"]["workers"] == 3


def test_atomic_publish_is_non_overwriting_and_double_sealed(tmp_path):
    output = tmp_path / "canonical"
    report = {"schema": "test", "claim_boundary": {"cycles": False}}
    receipt = {"schema": "receipt", "status": "TEST"}
    lock_path = tmp_path / M.SINGLE_WRITER_LOCK.name
    lock = M.acquire_single_writer_lock(lock_path)
    try:
        M.atomic_publish(output, report, receipt, lock)
        M.verify_double_seal(output)
        with pytest.raises(RuntimeError, match="exists"):
            M.atomic_publish(output, report, receipt, lock)
    finally:
        M.release_single_writer_lock(lock)


def test_single_writer_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / M.SINGLE_WRITER_LOCK.name
    lock = M.acquire_single_writer_lock(lock_path)
    try:
        M.validate_single_writer_lock(lock)
        with pytest.raises(RuntimeError, match="already exists"):
            M.acquire_single_writer_lock(lock_path)
    finally:
        M.release_single_writer_lock(lock)
    assert not lock_path.exists()


def test_atomic_publish_rejects_a_dangling_output_symlink(tmp_path):
    output = tmp_path / "canonical"
    output.symlink_to(tmp_path / "does_not_exist")
    lock = M.acquire_single_writer_lock(tmp_path / M.SINGLE_WRITER_LOCK.name)
    try:
        with pytest.raises(RuntimeError, match="exists"):
            M.atomic_publish(
                output, {"schema": "test"}, {"schema": "receipt"}, lock
            )
    finally:
        M.release_single_writer_lock(lock)


def test_fault_injected_execute_records_failure_is_double_sealed(tmp_path):
    run_state = tmp_path / "fresh_failure_receipt"
    run_state.mkdir()
    stage = {"name": "EXECUTE_EXACT_D0_D2_D3_POPULATION"}

    def injected_execute_records(_records, *, workers):
        assert workers == 3
        raise RuntimeError("injected execute_records failure")

    with pytest.raises(RuntimeError, match="injected execute_records"):
        M.run_with_failure_receipt(
            run_state, stage,
            lambda: injected_execute_records([], workers=3),
        )
    M.verify_double_seal(run_state)
    failure = M.strict_json(run_state / "FAILED.json")
    assert failure["stage"] == "EXECUTE_EXACT_D0_D2_D3_POPULATION"
    assert failure["workers"] == 3
    assert failure["canonical_output_admitted"] is False
    assert failure["cycles_admitted"] is False


def test_execute_records_rejects_worker_drift_before_pool_creation():
    with pytest.raises(RuntimeError, match="exactly 3"):
        M.execute_records([], workers=2)


def test_source_has_no_m618_expand_import_or_eager_run():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "run_m618" not in source
    assert "if module == 0:" in source
    assert "M700 D0 direct-vs-N128x3 miter failed" in source
    assert "exact_decoder_complete" in source
    assert "product_vs_bit_speedup\": None" in source


def test_docs359_is_frozen():
    docs = ROOT / "hw_autoresearch_nts07" / "docs" / "359_DATE终局冻结_20260813.md"
    assert M.sha256_file(docs) == \
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

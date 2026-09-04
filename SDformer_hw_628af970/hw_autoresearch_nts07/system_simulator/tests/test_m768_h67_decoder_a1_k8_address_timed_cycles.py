#!/usr/bin/env python3
"""Author attacks for the M768 source-only address-timed analyzer."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m768_h67_decoder_a1_k8_address_timed_cycles.py")
CONTRACT = (ROOT / "contracts/"
            "m768_h67_decoder_a1_k8_address_timed_cycle_contract_r1_20260828.json")
SPEC = importlib.util.spec_from_file_location("m768_address_timed", SCRIPT)
M768 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M768)


def common_resource(psum_mode="1RW", outstanding=8):
    return M768.CommonResource(
        lanes=96,
        accumulator_bits=24,
        clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760,
        macro_round_bytes=128,
        weight_bytes_logical=13824,
        psum_bytes_logical=221184,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=2560,
        weight=M768.PortSpec(8, "1R1W", 16, 4, 1, 1, outstanding),
        psum=M768.PortSpec(6, psum_mode, 48, 2, 1, 1, outstanding),
        external=M768.PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=M768.PortSpec(1, "1RW", 288, 1, 1, 1, 1),
    )


def transaction(identifier, kind, bank=0, config="TYPED_SIGNED_K8",
                dependencies=(), produces="", address=0x1000,
                width=16):
    return M768.CompressedTransaction(
        transaction_id=identifier,
        population_id="SYNTHETIC_PRIMARY",
        config=config,
        kind=kind,
        base_address=address,
        address_stride_bytes=64,
        count=1,
        bank_pattern=(bank,),
        width_bytes=width,
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
    )


def seal_directory(path):
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    members = sorted(member for member in path.rglob("*") if member.is_file()
                     and member not in (manifest, outer))
    manifest.write_text("".join(
        "{}  {}\n".format(M768.sha256(member),
                          member.relative_to(path).as_posix())
        for member in members), encoding="utf-8")
    outer.write_text(
        "{}  SHA256SUMS\n".format(M768.sha256(manifest)), encoding="utf-8")


def test_common_resource_exact_capacity_and_capacity_cliff():
    resource = common_resource()
    resource.validate()
    assert resource.allocated_macro_rounded_bytes == 243200
    assert resource.reserved_unallocated_bytes == 2560
    broken = M768.CommonResource(
        **dict(resource.__dict__, descriptor_control_bytes_logical=10753,
               reserved_unallocated_bytes=0)
    )
    with pytest.raises(M768.Failure, match="budget|capacity|conserve"):
        broken.validate()


def test_weight_bank_conflict_serializes_but_distinct_banks_issue_together():
    resource = common_resource()
    same = [transaction("a", "weight_read", bank=0),
            transaction("b", "weight_read", bank=0)]
    result_same = M768.AddressTimedScheduler(resource).schedule(
        M768.expand_transactions(same))
    assert [row["issue_cycle"] for row in
            result_same["scheduled_requests"]] == [0, 1]
    different = [transaction("a", "weight_read", bank=0),
                 transaction("b", "weight_read", bank=1)]
    result_different = M768.AddressTimedScheduler(resource).schedule(
        M768.expand_transactions(different))
    assert [row["issue_cycle"] for row in
            result_different["scheduled_requests"]] == [0, 0]


def test_1rw_serializes_read_write_while_1r1w_keeps_independent_ports():
    requests = [transaction("read", "psum_read", bank=0, width=48),
                transaction("write", "psum_write", bank=0, width=48)]
    one_rw = M768.AddressTimedScheduler(common_resource("1RW")).schedule(
        M768.expand_transactions(requests))
    one_r_one_w = M768.AddressTimedScheduler(
        common_resource("1R1W")).schedule(M768.expand_transactions(requests))
    assert [row["issue_cycle"] for row in
            one_rw["scheduled_requests"]] == [0, 1]
    assert [row["issue_cycle"] for row in
            one_r_one_w["scheduled_requests"]] == [0, 0]


def test_outstanding_slot_is_reusable_on_the_response_cycle():
    requests = [transaction("first", "weight_read", bank=0),
                transaction("second", "weight_read", bank=0)]
    result = M768.AddressTimedScheduler(
        common_resource(outstanding=1)).schedule(
            M768.expand_transactions(requests))
    first, second = result["scheduled_requests"]
    assert first["return_cycle"] == 4
    assert second["issue_cycle"] == first["return_cycle"]
    assert result["same_cycle_response_slot_reuse"] is True


def test_dependency_issue_return_commit_timestamps_and_conservation():
    requests = [
        transaction("weight", "weight_read", bank=0, produces="weight_done"),
        transaction("compute", "compute", bank=0,
                    dependencies=("weight_done:0",), produces="compute_done",
                    width=288),
        transaction("commit", "commit", bank=0,
                    dependencies=("compute_done:0",), width=384),
    ]
    result = M768.AddressTimedScheduler(common_resource()).schedule(
        M768.expand_transactions(requests))
    weight, compute, commit = result["scheduled_requests"]
    assert compute["issue_cycle"] >= weight["return_cycle"]
    assert commit["issue_cycle"] >= compute["return_cycle"]
    for row in result["scheduled_requests"]:
        assert row["addresses"] and row["banks"]
        assert row["issue_cycle"] <= row["return_cycle"] <= row["commit_cycle"]
    assert sum(result["cycle_classes"].values()) == result["total_cycles"]
    assert result["compressed_transaction_count"] == 3


def test_commit_hash_ignores_config_and_timing_but_not_address():
    hashes = {}
    resource_hashes = {}
    fallbacks = {}
    for config in M768.CONFIGS:
        tx = transaction("commit", "commit", bank=0, config=config,
                         address=0xABC000, width=384)
        result = M768.AddressTimedScheduler(common_resource()).schedule(
            M768.expand_transactions([tx]))
        hashes[config] = result["commit_sequence_sha256"]
        resource_hashes[config] = common_resource().identity()[
            "resource_manifest_sha256"]
        fallbacks[config] = M768.route_for_record(1, config)[
            "fallback_policy"]
    M768.assert_fair_configs(resource_hashes, hashes, fallbacks)
    changed = transaction("commit", "commit", bank=0,
                          address=0xABC180, width=384)
    changed_hash = M768.AddressTimedScheduler(common_resource()).schedule(
        M768.expand_transactions([changed]))["commit_sequence_sha256"]
    assert changed_hash != next(iter(hashes.values()))


def test_population_mixing_is_fail_closed():
    assert M768.assert_population_isolation(["PRIMARY", "PRIMARY"]) == "PRIMARY"
    with pytest.raises(M768.Failure, match="never be mixed"):
        M768.assert_population_isolation(["PRIMARY", "SECONDARY"])


def test_d1_fallback_is_identical_and_headline_comparator_is_narrow():
    rows = [M768.route_for_record(1, config) for config in M768.CONFIGS]
    assert all(row["fallback"] is True for row in rows)
    assert len({row["effective_config"] for row in rows}) == 1
    assert len({row["fallback_policy"] for row in rows}) == 1
    assert M768.headline_ratio_allowed("TYPED_SIGNED_K8",
                                       "EQUAL_SERVICE_K1X8")
    assert not M768.headline_ratio_allowed("TYPED_SIGNED_K8", "A1_OSG")
    assert not M768.headline_ratio_allowed("TYPED_SIGNED_K8", "K1")


def test_dense_commit_address_hash_is_deterministic():
    addresses = M768.dense_commit_addresses(3, 9, 4, 6, 1)
    assert len(addresses) == 24
    assert len(set(addresses)) == 24
    assert M768.commit_address_hash(addresses) == M768.commit_address_hash(
        list(addresses))
    assert M768.commit_address_hash(addresses) != M768.commit_address_hash(
        list(reversed(addresses)))


def test_bank_unique_flattened_k_packing_conserves_sources():
    flat = [0, 8, 16, 1, 9, 2]
    groups = M768.bank_unique_groups(flat, channels=32, bank_count=8)
    assert sum(len(group) for group in groups) == len(flat)
    for group in groups:
        banks = [int(value) % 32 % 8 for value in group]
        assert len(banks) == len(set(banks))


def test_m672_exact_mapper_adapter_on_tiny_bitpack(tmp_path):
    mapper_path = (ROOT / "system_simulator/scripts/"
                   "map_m672_decoder_convtranspose_polyphase_workload_r3.py")
    mapper = M768.load_pinned_mapper(
        mapper_path,
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
    )
    package = tmp_path / "package"
    package.mkdir()
    shape = [1, 1, 3, 2, 3]
    activation = np.asarray(
        [[[[[1, 0, 1], [0, 1, 0]],
           [[0, 1, 0], [1, 0, 1]],
           [[1, 1, 0], [0, 0, 1]]]]], dtype=np.uint8)
    payload = package / "x.bitpack"
    payload.write_bytes(np.packbits(activation.reshape(-1),
                                    bitorder="little").tobytes())
    tiles = list(mapper.iter_polyphase_tiles(
        payload, shape, tile_m=2, trusted_root=package))
    assert tiles
    assert sum(int(tile["values"].sum()) for tile in tiles) > 0
    assert all("destination_y" in tile and "source_flat_index" in tile
               for tile in tiles)


def test_tiny_mapper_to_address_timed_scheduler_end_to_end(tmp_path):
    mapper_path = (ROOT / "system_simulator/scripts/"
                   "map_m672_decoder_convtranspose_polyphase_workload_r3.py")
    mapper = M768.load_pinned_mapper(
        mapper_path,
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
    )
    package = tmp_path / "package"
    calls = package / "calls"
    calls.mkdir(parents=True)
    shape = [10, 1, 3, 2, 3]
    activation = np.zeros(shape, dtype=np.uint8)
    activation[0, 0, 0, 0, 0] = 1
    activation[0, 0, 1, 1, 2] = 1
    payload = calls / "x.bitpack"
    payload.write_bytes(np.packbits(activation.reshape(-1),
                                    bitorder="little").tobytes())
    record = {
        "population_id": "TINY_PRIMARY",
        "module_index": 0,
        "input_shape": shape,
        "relative_path": "calls/x.bitpack",
        "packed_sha256": M768.sha256(payload),
    }
    geometry = {0: (3, 4, 2, 3, 4, 6)}
    transactions = list(M768.iter_record_transactions(
        mapper, record, package, "TINY_PRIMARY", "TYPED_SIGNED_K8", 0,
        tile_m=2, geometry=geometry))
    assert transactions
    assert {tx.kind for tx in transactions} >= {
        "weight_read", "psum_read", "compute", "psum_write", "commit"
    }
    result = M768.AddressTimedScheduler(common_resource()).schedule(
        M768.expand_transactions(transactions))
    commit_rows = [row for row in result["scheduled_requests"]
                   if row["kind"] == "commit"]
    assert len(commit_rows) == 4 * 6
    assert result["population_ids"] == ["TINY_PRIMARY"]
    assert all(row["issue_cycle"] <= row["return_cycle"] <=
               row["commit_cycle"] for row in commit_rows)
    expected_addresses = [
        (1 << 60) | address for address in
        M768.dense_commit_addresses(0, 0, 4, 6, 1)
    ]
    assert result["commit_sequence_sha256"] == M768.commit_address_hash(
        expected_addresses)


def test_sealed_directory_checks_nested_seals_and_rejects_mutation(tmp_path):
    package = tmp_path / "sealed"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (nested / "SHA256SUMS").write_text("nested member\n", encoding="utf-8")
    (nested / "payload.bin").write_bytes(b"payload")
    seal_directory(package)
    identity = M768.verify_sealed_directory(package)
    assert len(identity["manifest_sha256"]) == 64
    (nested / "payload.bin").write_bytes(b"mutated")
    with pytest.raises(M768.Failure, match="mismatch"):
        M768.verify_sealed_directory(package)


def test_real_manifests_normalize_without_population_mixing():
    primary_path = (ROOT / "system_handoff/outgoing/"
                    "m686r6_h67_ep35_layer_static_decoder_payload_s10_r1_20260828/"
                    "manifest.json")
    secondary_path = (ROOT / "system_handoff/outgoing/"
                      "m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828/"
                      "manifest.json")
    primary = M768.normalized_population_records(
        M768.strict_json(primary_path), "M686_PRIMARY")
    secondary = M768.normalized_population_records(
        M768.strict_json(secondary_path), "M699_SECONDARY")
    assert len(primary) == 40
    assert len(secondary) == 120
    assert M768.assert_population_isolation(
        row["population_id"] for row in primary) == "M686_PRIMARY"
    assert M768.assert_population_isolation(
        row["population_id"] for row in secondary) == "M699_SECONDARY"


def test_source_contract_forbids_production_and_has_no_external_candidate():
    text = CONTRACT.read_text(encoding="utf-8")
    contract = json.loads(text)
    assert contract["launch_now"] is False
    assert contract["production_speedup_allowed"] is False
    assert contract["table_a_feed_defaults"]["table_a_insertion_allowed"] is False
    assert "m700" not in text.lower()
    assert hashlib.sha256((ROOT / "docs/359_DATE终局冻结_20260813.md").read_bytes()
                          ).hexdigest() == M768.DOCS359_SHA256


def test_synthetic_self_test_never_emits_production_metrics():
    result = M768.synthetic_self_test()
    assert result["status"] == "PASS_M768_SYNTHETIC_SOURCE_SELF_TEST"
    assert result["production_cycles"] is None
    assert result["production_speedup"] is None

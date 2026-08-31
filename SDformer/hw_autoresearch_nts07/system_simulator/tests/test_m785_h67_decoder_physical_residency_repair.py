#!/usr/bin/env python3
"""Adversarial source-only tests for the M785 physical-residency repair."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m785_h67_decoder_physical_residency_repair.py")
CONTRACT = (ROOT / "contracts/"
            "m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json")
SPEC = importlib.util.spec_from_file_location("m785_decoder_repair", SCRIPT)
M785 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M785)


def common_resource(psum_bytes=221184):
    rounded_psum = ((psum_bytes + 127) // 128) * 128
    return M785.CommonResource(
        lanes=96, accumulator_bits=24, clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760, macro_round_bytes=128,
        weight_bytes_logical=13824, psum_bytes_logical=psum_bytes,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=245760 - 13824 - rounded_psum - 8192,
        weight=M785.PortSpec(8, "1R1W", 16, 4, 1, 1, 8),
        psum=M785.PortSpec(6, "1RW", 48, 2, 1, 1, 8),
        external=M785.PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=M785.PortSpec(1, "1RW", 288, 1, 1, 1, 1))


def tx(identifier, kind, base, banks, width, dependencies=(), produces="",
       earliest=0, count=1, stride=0, offsets=()):
    return M785.CompressedTransaction(
        identifier, "M785_SYNTHETIC", "TYPED_SIGNED_K8", kind, base,
        stride, count, tuple(banks), width,
        address_offsets=tuple(offsets),
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
        earliest_issue_cycle=earliest)


def load_mapper_oracles():
    mapper = M785.load_pinned_module(
        ROOT / "system_simulator/scripts/"
        "map_m672_decoder_convtranspose_polyphase_workload_r3.py",
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
        "m672_mapper")
    oracles = M785.load_pinned_oracles(
        ROOT / "system_simulator/scripts/"
        "analyze_m712_pidp_decoder_exact_cpu_fastkill.py",
        "87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc",
        ROOT / "system_simulator/scripts/"
        "analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py",
        "ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165")
    return mapper, oracles


def tiny_record(tmp_path, module_index=0, cout=96):
    package = tmp_path / "package"
    calls = package / "calls"
    calls.mkdir(parents=True)
    shape = [10, 1, 8, 2, 3]
    activation = np.zeros(shape, dtype=np.uint8)
    activation[0, 0, 0, 0, 0] = 1
    activation[0, 0, 1, 1, 2] = 1
    activation[0, 0, 7, 0, 2] = 1
    payload = calls / "x.bitpack"
    payload.write_bytes(np.packbits(
        activation.reshape(-1), bitorder="little").tobytes())
    record = {
        "population_id": "TINY", "module_index": module_index,
        "input_shape": shape, "relative_path": "calls/x.bitpack",
        "packed_sha256": M785.sha256(payload),
    }
    return package, record, {
        module_index: (8, cout, 2, 3, 4, 6)}, activation


def test_frozen_logical_goldens_and_physical_slot_mapping_are_distinct():
    assert M785.weight_bank_and_local_row(24, 8) == (0, 48)
    assert M785.weight_bank_and_local_row(25, 8) == (1, 48)
    assert M785.weight_slot_bank_and_local_row(0, 0, 8) == (0, 0)
    assert M785.weight_slot_bank_and_local_row(1, 0, 8) == (0, 192)
    assert M785.weight_slot_bank_and_local_row(8, 7, 8) == (7, 1536)


def test_dirty_psum_replacement_waits_for_local_read_and_external_evict():
    residency = M785.PsumResidency(M785.PSUM_VECTOR_BYTES)
    residency.acquire(0)
    residency.mark_dirty(0)
    slot, events = residency.acquire(1)
    assert slot == 0 and [event.kind for event in events] == ["evict"]
    moves = M785.residency_transactions(
        "m785", "M785_SYNTHETIC", "TYPED_SIGNED_K8", events,
        {0: "victim_done:0"})
    assert [row.kind for row in moves] == ["psum_read", "external_write"]
    assert moves[1].dependency_tokens == (M785._terminal_token(moves[0]),)
    victim = tx("victim", "psum_write", 0, range(6), 48,
                produces="victim_done", earliest=10)
    replacement = tx(
        "replacement", "psum_read", 0, range(6), 48,
        dependencies=(M785._terminal_token(moves[-1]),))
    scheduled = M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([victim] + moves + [replacement]))
    rows = {row["request_id"]: row for row in scheduled["scheduled_requests"]}
    evict_return = max(row["return_cycle"] for key, row in rows.items()
                       if "external_write" in key)
    assert rows["replacement:0"]["issue_cycle"] >= evict_return


def test_psum_restore_charges_external_then_six_bank_local_write():
    residency = M785.PsumResidency(M785.PSUM_VECTOR_BYTES)
    residency.acquire(0); residency.mark_dirty(0)
    residency.acquire(1); residency.mark_dirty(1)
    _slot, events = residency.acquire(0)
    assert [event.kind for event in events] == ["evict", "restore"]
    moves = M785.residency_transactions(
        "restore", "M785_SYNTHETIC", "TYPED_SIGNED_K8", events,
        {0: "old0:done:0", 1: "old1:done:0"})
    assert [row.kind for row in moves] == [
        "psum_read", "external_write", "external_read", "psum_write"]
    assert moves[-1].bank_pattern == tuple(range(6))
    assert moves[-1].dependency_tokens == (M785._terminal_token(moves[-2]),)


def test_weight_lru_owns_nine_bijective_physical_slots():
    cache = M785.WeightResidency()
    keys = [(0, block, 0, 0) for block in range(10)]
    accesses = [cache.access(key) for key in keys[:9]]
    assert [row.slot for row in accesses] == list(range(9))
    assert len(cache.key_to_slot) == len(cache.slot_to_key) == 9
    replacement = cache.access(keys[9])
    assert replacement.evicted_key == keys[0] and replacement.slot == 0
    assert cache.slot_to_key[0] == keys[9]


def test_weight_refill_external_return_local_write_and_read_are_hard_chained():
    rows = M785._weight_refill_transactions(
        "w", "M785_SYNTHETIC", "TYPED_SIGNED_K8", 0, 0, 0, 2,
        "source:0", ("prior_use0:0", "prior_use1:0"), 0)
    assert [row.kind for row in rows] == ["external_read", "weight_write"]
    assert rows[0].count == 8 and rows[1].count == 12
    assert rows[1].dependency_tokens == (
        M785._terminal_token(rows[0]), "prior_use0:0", "prior_use1:0")
    prior0 = tx("prior0", "weight_read", 0, (0,), 16,
                produces="prior_use0", earliest=10)
    prior1 = tx("prior1", "weight_read", 0, (1,), 16,
                produces="prior_use1", earliest=20, offsets=(0,))
    bank, offset = M785.weight_slot_bank_and_local_row(2, 0, 8)
    read = tx("read", "weight_read", offset, (bank,), 16,
              dependencies=(M785._terminal_token(rows[1]),),
              count=6, stride=16, produces="read_done")
    source = tx("source", "external_read", 1 << 60, (0,), 16,
                produces="source")
    schedule = M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([source, prior0, prior1] + rows + [read]))
    by_id = {row["request_id"]: row
             for row in schedule["scheduled_requests"]}
    refill_external_return = by_id[
        rows[0].transaction_id + ":7"]["return_cycle"]
    local_first_issue = by_id[
        rows[1].transaction_id + ":0"]["issue_cycle"]
    local_final_return = by_id[
        rows[1].transaction_id + ":11"]["return_cycle"]
    assert local_first_issue >= refill_external_return
    assert local_first_issue >= by_id["prior0:0"]["return_cycle"]
    assert local_first_issue >= by_id["prior1:0"]["return_cycle"]
    assert by_id["read:0"]["issue_cycle"] >= local_final_return


def test_weight_bank_range_rejects_one_byte_over_per_bank_partition():
    legal = tx("legal", "weight_read", 1728 - 16, (0,), 16)
    M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([legal]))
    illegal = tx("illegal", "weight_read", 1728 - 15, (0,), 16)
    with pytest.raises(M785.Failure, match="weight address"):
        M785.AddressTimedScheduler(common_resource()).schedule(
            M785.expand_transactions([illegal]))


def test_m722_line_buffer_plan_injection_is_rejected(tmp_path):
    _mapper, oracles = load_mapper_oracles()
    _package, _record, geometry, activation = tiny_record(tmp_path)
    bits = activation[0, 0]
    spec = geometry[0]
    counts = oracles.m722r2.R1.group_counts(bits, 1)
    original = oracles.m722r2.R1

    class BadR1:
        group_counts = staticmethod(original.group_counts)

        @staticmethod
        def a1_storage_plan(spec_arg):
            value = dict(original.a1_storage_plan(spec_arg))
            value["stripe_count"] = 999
            value["stripes"] = [[0, 1]] * 999
            return value

    class BadM722:
        R1 = BadR1

    with pytest.raises(M785.Failure, match="M722 line-buffer storage"):
        M785.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]),
            M785.OracleBundle(oracles.m712, BadM722(), oracles.storage))


def test_independent_global_vector_storage_injection_is_rejected(tmp_path):
    _mapper, oracles = load_mapper_oracles()
    _package, _record, geometry, activation = tiny_record(tmp_path)
    bits = activation[0, 0]
    spec = geometry[0]
    counts = oracles.m722r2.R1.group_counts(bits, 1)

    class BadStorage:
        @staticmethod
        def plan(geometry_arg, psum_arg):
            value = dict(oracles.storage.plan(geometry_arg, psum_arg))
            value["offchip_backing_address_span_bytes"] += 288
            return value

    with pytest.raises(M785.Failure, match="M785 independent storage"):
        M785.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]),
            M785.OracleBundle(oracles.m712, oracles.m722r2, BadStorage()))


def test_d3_independent_storage_is_100_stripes_and_not_m722_equivalent():
    value = M785.STORAGE.plan((96, 96, 120, 160, 240, 320), 221184)
    assert value["resident_capacity_vectors"] == 768
    assert value["stripe_count"] == 100
    assert value["stripes"][0] == [0, 768]
    assert value["stripes"][-1][1] == 240 * 320
    assert value["offchip_backing_address_span_bytes"] == 240 * 320 * 288
    assert value["m722_line_buffer_storage_equivalent"] is False


def test_tiny_end_to_end_has_local_moves_physical_slots_and_fair_commit(tmp_path):
    mapper, oracles = load_mapper_oracles()
    package, record, geometry, _activation = tiny_record(tmp_path, cout=192)
    schedules = {}
    projections = {}
    for config in M785.CONFIGS:
        rows = list(M785.iter_record_transactions(
            mapper, record, package, "TINY", config, 0, oracles,
            tile_m=2, geometry=geometry,
            psum_bytes=M785.PSUM_VECTOR_BYTES * 2))
        assert any(row.kind == "weight_write" for row in rows)
        expanded = list(M785.expand_transactions(rows))
        weight_addresses = {
            request.addresses for request in expanded
            if request.kind == "weight_read"
        }
        assert len(weight_addresses) > 1
        schedule = M785.AddressTimedScheduler(
            common_resource(M785.PSUM_VECTOR_BYTES * 2)).schedule(
                expanded)
        schedules[config] = schedule
        projections[config] = [
            (row["kind"], tuple(row["banks"]), row["width_bytes"],
             row["issue_cycle"], row["return_cycle"])
            for row in schedule["scheduled_requests"]]
    assert projections["A1_OSG"] != projections["EQUAL_SERVICE_K1X8"]
    assert projections["EQUAL_SERVICE_K1X8"] != projections["TYPED_SIGNED_K8"]
    assert len({value["commit_sequence_sha256"]
                for value in schedules.values()}) == 1


def test_psum_partition_still_fails_one_byte_over_boundary():
    legal = tx("legal", "psum_read", 221184 - 288, range(6), 48)
    M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([legal]))
    illegal = tx("illegal", "psum_read", 221184 - 287, range(6), 48)
    with pytest.raises(M785.Failure, match="psum address"):
        M785.AddressTimedScheduler(common_resource()).schedule(
            M785.expand_transactions([illegal]))


def test_d1_remains_common_full_shape_diagnostic(tmp_path):
    _package, record, geometry, _activation = tiny_record(tmp_path, 1)
    signatures = {}
    for config in M785.CONFIGS:
        rows = list(M785._d1_transactions(
            record, "TINY", config, 0, geometry[1]))
        signatures[config] = [(row.kind, row.count, row.width_bytes)
                              for row in rows]
        assert next(row.count for row in rows if row.kind == "compute") > 1
        assert M785.route_for_record(1, config)["headline_eligible"] is False
    assert len({tuple(value) for value in signatures.values()}) == 1


def test_only_k8_vs_equal_service_and_source_contract_is_fail_closed():
    assert M785.headline_ratio_allowed(
        "TYPED_SIGNED_K8", "EQUAL_SERVICE_K1X8")
    assert not M785.headline_ratio_allowed("TYPED_SIGNED_K8", "A1_OSG")
    assert not M785.headline_ratio_allowed("TYPED_SIGNED_K8", "K1")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["launch_now"] is False
    assert contract["production_speedup_allowed"] is False
    assert contract["claim_boundary"]["decoder_complete"] is False
    assert contract["claim_boundary"]["table_a_insertion_allowed"] is False
    assert M785.sha256(ROOT / "docs/359_DATE终局冻结_20260813.md") == \
        M785.DOCS359_SHA256


def test_self_test_contains_no_production_metric():
    value = M785.synthetic_self_test()
    assert value["status"] == "PASS_M785_SYNTHETIC_SOURCE_SELF_TEST"
    assert value["production_cycles"] is None
    assert value["production_speedup"] is None
    assert value["launch_now"] is False

#!/usr/bin/env python3
"""Adversarial source-only tests for the additive M777 repair."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m777_h67_decoder_a1_k8_address_timed_repair.py")
CONTRACT = (ROOT / "contracts/"
            "m777_h67_decoder_a1_k8_address_timed_repair_contract_r1_20260828.json")
SPEC = importlib.util.spec_from_file_location("m777_decoder_repair", SCRIPT)
M777 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M777)


def common_resource(psum_bytes=221184):
    reserve = 245760 - 13824 - psum_bytes - 8192
    return M777.CommonResource(
        lanes=96,
        accumulator_bits=24,
        clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760,
        macro_round_bytes=128,
        weight_bytes_logical=13824,
        psum_bytes_logical=psum_bytes,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=reserve,
        weight=M777.PortSpec(8, "1R1W", 16, 4, 1, 1, 8),
        psum=M777.PortSpec(6, "1RW", 48, 2, 1, 1, 8),
        external=M777.PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=M777.PortSpec(1, "1RW", 288, 1, 1, 1, 1),
    )


def transaction(identifier, kind, address, width=48, bank_pattern=(0,)):
    return M777.CompressedTransaction(
        transaction_id=identifier,
        population_id="SYNTHETIC",
        config="TYPED_SIGNED_K8",
        kind=kind,
        base_address=address,
        address_stride_bytes=0,
        count=1,
        bank_pattern=bank_pattern,
        width_bytes=width,
    )


def load_mapper_and_oracles():
    mapper = M777.load_pinned_module(
        ROOT / "system_simulator/scripts/"
        "map_m672_decoder_convtranspose_polyphase_workload_r3.py",
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
        "m672_mapper",
    )
    oracles = M777.load_pinned_oracles(
        ROOT / "system_simulator/scripts/"
        "analyze_m712_pidp_decoder_exact_cpu_fastkill.py",
        "87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc",
        ROOT / "system_simulator/scripts/"
        "analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py",
        "ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165",
    )
    return mapper, oracles


def tiny_record(tmp_path, module_index=0):
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
        "population_id": "TINY",
        "module_index": module_index,
        "input_shape": shape,
        "relative_path": "calls/x.bitpack",
        "packed_sha256": M777.sha256(payload),
    }
    geometry = {module_index: (8, 96, 2, 3, 4, 6)}
    return package, record, geometry, activation


def test_weight_bank_local_golden_k24_k25_and_tap_channel_div8():
    assert M777.weight_bank_and_local_row(24, 8) == (0, 48)
    assert M777.weight_bank_and_local_row(25, 8) == (1, 48)
    assert M777.weight_bank_and_local_row(8, 8) == (0, 16)
    banks, rows = M777.weight_group_layout([24, 25], 8)
    assert banks == (0, 1)
    assert rows == (48, 48)


def test_weight_group_rejects_duplicate_bank_but_a1_keeps_logical_osg():
    with pytest.raises(M777.Failure, match="collision"):
        M777.weight_group_layout([0, 8], 16)
    groups = M777.service_groups([(0, 0), (8, 8)], "A1_OSG", 16)
    assert groups == [((0, 0), (8, 8))]
    k8_groups = M777.service_groups([(0, 0), (8, 8)],
                                    "TYPED_SIGNED_K8", 16)
    assert len(k8_groups) == 2


def test_three_config_service_semantics_are_not_labels():
    contributors = [(0, 0), (1, 1), (8, 8), (9, 9), (16, 16)]
    groups = {config: M777.service_groups(contributors, config, 16)
              for config in M777.CONFIGS}
    assert sum(map(len, groups["A1_OSG"])) == len(contributors)
    assert sum(map(len, groups["EQUAL_SERVICE_K1X8"])) == len(contributors)
    assert sum(map(len, groups["TYPED_SIGNED_K8"])) == len(contributors)
    # Equal physical K1x8 and K8 grouping, but executable descriptor/weight
    # construction differs below; A1 grouping differs at bank collisions.
    assert groups["A1_OSG"] != groups["TYPED_SIGNED_K8"]
    prefix = "p"
    k1 = M777._descriptor_transactions(
        prefix, "POP", "EQUAL_SERVICE_K1X8",
        groups["EQUAL_SERVICE_K1X8"][0], 0, "source:0")
    k8 = M777._descriptor_transactions(
        prefix, "POP", "TYPED_SIGNED_K8",
        groups["TYPED_SIGNED_K8"][0], 0, "source:0")
    assert len(k1) == len(groups["EQUAL_SERVICE_K1X8"][0])
    assert len(k8) == 1
    assert [(row.kind, row.count, row.width_bytes) for row in k1] != [
        (row.kind, row.count, row.width_bytes) for row in k8]


def test_only_k8_vs_equal_service_k1x8_is_legal():
    assert M777.headline_ratio_allowed(
        "TYPED_SIGNED_K8", "EQUAL_SERVICE_K1X8")
    assert not M777.headline_ratio_allowed("TYPED_SIGNED_K8", "A1_OSG")
    assert not M777.headline_ratio_allowed("TYPED_SIGNED_K8", "K1")


def test_psum_partition_stripe_is_exact_and_d3_is_bounded():
    stripes = M777.psum_stripes(240 * 320, 221184)
    assert 221184 // M777.PSUM_VECTOR_BYTES == 768
    assert stripes[0].vector_lo == 0 and stripes[0].vector_hi == 768
    assert stripes[-1].vector_hi == 240 * 320
    assert all(row.vector_hi - row.vector_lo <= 768 for row in stripes)
    assert len(stripes) == 100


def test_scheduler_rejects_one_byte_over_physical_psum_partition():
    resource = common_resource()
    resource.validate()
    legal = transaction(
        "legal", "psum_read", 221184 - 288, 48, tuple(range(6)))
    M777.AddressTimedScheduler(resource).schedule(
        M777.expand_transactions([legal]))
    illegal = transaction(
        "illegal", "psum_read", 221184 - 287, 48, tuple(range(6)))
    with pytest.raises(M777.Failure, match="physical partition"):
        M777.AddressTimedScheduler(resource).schedule(
            M777.expand_transactions([illegal]))


def test_residency_evict_restore_and_backing_are_explicitly_charged():
    residency = M777.PsumResidency(M777.PSUM_VECTOR_BYTES * 2)
    residency.acquire(0); residency.mark_dirty(0)
    residency.acquire(1); residency.mark_dirty(1)
    _slot, evict = residency.acquire(2)
    assert [(row.kind, row.key) for row in evict] == [("evict", 0)]
    _slot, restore = residency.acquire(0)
    assert [(row.kind, row.key) for row in restore] == [("evict", 1),
                                                        ("restore", 0)]
    txs = M777.residency_transactions(
        "r", "POP", "TYPED_SIGNED_K8", evict + restore,
        {0: "psum0:done:0", 1: "psum1:done:0"})
    assert [row.kind for row in txs] == [
        "external_write", "external_write", "external_read"]
    assert all(row.count == 2 for row in txs)  # 288 B / 192 B/cycle
    assert txs[0].dependency_tokens == ("psum0:done:0",)


def test_weight_refill_cache_is_nine_tiles_not_unbounded():
    cache = M777.WeightResidency()
    keys = [(0, 0, 0, index) for index in range(10)]
    for key in keys[:9]:
        assert cache.access(key) == (True, None)
    miss, evicted = cache.access(keys[9])
    assert miss is True and evicted == keys[0]
    miss, evicted = cache.access(keys[0])
    assert miss is True and evicted == keys[1]


def test_real_m712_m722_oracles_are_called_and_mismatch_fails(tmp_path):
    _mapper, oracles = load_mapper_and_oracles()
    _package, _record, geometry_map, activation = tiny_record(tmp_path)
    bits = activation[0, 0]
    spec = geometry_map[0]
    blocks = 1
    counts = oracles.m722r2.R1.group_counts(bits, blocks)
    result = M777.verify_contributor_and_storage_oracles(
        bits, 0, spec, int(counts["contributors"]),
        int(counts["osg_groups"]), oracles)
    assert result["m712_contributors"] == result["m722_contributors"]

    class BadM712:
        @staticmethod
        def descriptor_counts(bits_arg, blocks_arg):
            rows = oracles.m712.descriptor_counts(bits_arg, blocks_arg)
            return rows[0], rows[1], rows[2] + 1, rows[3]

    bad = M777.OracleBundle(BadM712(), oracles.m722r2)
    with pytest.raises(M777.Failure, match="M712 contributor oracle"):
        M777.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]), bad)


def test_tiny_end_to_end_has_nonisomorphic_schedules_fetch_refill_and_same_commit(tmp_path):
    mapper, oracles = load_mapper_and_oracles()
    package, record, geometry, _activation = tiny_record(tmp_path)
    schedules = {}
    projections = {}
    for config in M777.CONFIGS:
        txs = list(M777.iter_record_transactions(
            mapper, record, package, "TINY", config, 0, oracles,
            tile_m=2, geometry=geometry))
        kinds = [row.kind for row in txs]
        ids = [row.transaction_id for row in txs]
        assert kinds.count("external_read") > 1
        assert any("source_fetch" in value for value in ids)
        assert any("descriptor" in value for value in ids)
        assert any("weight_refill" in value for value in ids)
        result = M777.AddressTimedScheduler(common_resource()).schedule(
            M777.expand_transactions(txs))
        schedules[config] = result
        projections[config] = [
            (row["kind"], tuple(row["banks"]), row["width_bytes"],
             row["issue_cycle"], row["return_cycle"])
            for row in result["scheduled_requests"]]
    assert projections["A1_OSG"] != projections["EQUAL_SERVICE_K1X8"]
    assert projections["EQUAL_SERVICE_K1X8"] != projections["TYPED_SIGNED_K8"]
    assert len({schedules[c]["commit_sequence_sha256"]
                for c in M777.CONFIGS}) == 1
    assert len({schedules[c]["total_cycles"] for c in M777.CONFIGS}) > 1


def test_d1_common_fallback_charges_full_shape_and_is_not_placeholder(tmp_path):
    _package, record, geometry, _activation = tiny_record(tmp_path, 1)
    rows_by_config = {}
    for config in M777.CONFIGS:
        rows = list(M777._d1_transactions(
            record, "TINY", config, 0, geometry[1]))
        rows_by_config[config] = rows
        compute = next(row for row in rows if row.kind == "compute")
        assert compute.count == (8 * 2 * 3 * 9 * 96) // 96
        assert compute.count > 1
        assert sum(row.count * row.width_bytes for row in rows
                   if row.kind == "external_read") > 8 * 2 * 3 * 4
    signatures = {
        config: [(row.kind, row.count, row.width_bytes)
                 for row in rows]
        for config, rows in rows_by_config.items()
    }
    assert len({tuple(value) for value in signatures.values()}) == 1
    assert all(M777.route_for_record(1, config)["headline_eligible"] is False
               for config in M777.CONFIGS)


def test_same_common_resource_and_commit_fairness_remains_enforced():
    hashes = {config: common_resource().identity()["resource_manifest_sha256"]
              for config in M777.CONFIGS}
    addresses = [(1 << 60) | value for value in
                 M777.dense_commit_addresses(3, 0, 4, 6, 1)]
    commits = {config: M777.commit_address_hash(addresses)
               for config in M777.CONFIGS}
    fallbacks = {config: M777.route_for_record(1, config)["fallback_policy"]
                 for config in M777.CONFIGS}
    M777.assert_fair_configs(hashes, commits, fallbacks)
    commits["A1_OSG"] = M777.commit_address_hash(list(reversed(addresses)))
    with pytest.raises(M777.Failure, match="commit"):
        M777.assert_fair_configs(hashes, commits, fallbacks)


def test_source_contract_remains_fail_closed_and_docs359_is_frozen():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["launch_now"] is False
    assert contract["production_speedup_allowed"] is False
    assert contract["claim_boundary"]["decoder_complete"] is False
    assert contract["claim_boundary"]["full_network_completion"] is False
    assert contract["claim_boundary"]["table_a_insertion_allowed"] is False
    assert M777.sha256(ROOT / "docs/359_DATE终局冻结_20260813.md") == \
        M777.DOCS359_SHA256


def test_cli_self_test_never_emits_production_metrics():
    result = M777.synthetic_self_test()
    assert result["status"] == "PASS_M777_SYNTHETIC_SOURCE_SELF_TEST"
    assert result["production_cycles"] is None
    assert result["production_speedup"] is None
    assert result["launch_now"] is False

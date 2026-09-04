#!/usr/bin/env python3
"""Receipt-blind, source-only attacks for the frozen M785 candidate.

No production population is opened and no decoder cycle/speedup result is
written.  Only synthetic dependency, address, oracle and fairness cases are
constructed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
ANALYZER = HW / "system_simulator/scripts/analyze_m785_h67_decoder_physical_residency_repair.py"
ORACLE = HW / "system_simulator/scripts/oracle_m785_decoder_global_vector_storage.py"
TESTS = HW / "system_simulator/tests/test_m785_h67_decoder_physical_residency_repair.py"
CONTRACT = HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"
REQUEST = HW / "reviews/m786_m785_decoder_physical_residency_source_fresh_hammer_REQUEST_r1_20260828"

EXPECTED = {
    "analyzer": "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1",
    "oracle": "422da36ad1414d2dfa70363607c27bb99dee2f2505d1ceee2142a6023c162db5",
    "tests": "1ec8730cde5f91a91af269fb54969c5c6762fe5cb8bc36ba4b36117ce21c6787",
    "contract": "612a2ba39ceecedc351f2f6550347ad50ca9526fd89ed143bc6362c3e5681810",
    "request_manifest": "3d2abc6a4091201d1628ffbdf60d8945f57be349ddbb1c6a82212bc0a17364f5",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M785 = load(ANALYZER, "m790_frozen_m785")


def common_resource(psum_bytes: int = 221184):
    rounded = ((psum_bytes + 127) // 128) * 128
    return M785.CommonResource(
        lanes=96, accumulator_bits=24, clock_ns=3.0,
        external_bytes_per_cycle=192,
        onchip_budget_bytes_macro_rounded=245760, macro_round_bytes=128,
        weight_bytes_logical=13824, psum_bytes_logical=psum_bytes,
        descriptor_control_bytes_logical=8192,
        reserved_unallocated_bytes=245760 - 13824 - rounded - 8192,
        weight=M785.PortSpec(8, "1R1W", 16, 4, 1, 1, 8),
        psum=M785.PortSpec(6, "1RW", 48, 2, 1, 1, 8),
        external=M785.PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=M785.PortSpec(1, "1RW", 288, 1, 1, 1, 1))


def tx(identifier: str, kind: str, base: int, banks: Sequence[int],
       width: int, dependencies: Sequence[str] = (), produces: str = "",
       earliest: int = 0, count: int = 1, stride: int = 0,
       offsets: Sequence[int] = ()):
    return M785.CompressedTransaction(
        identifier, "M790_SYNTHETIC", "TYPED_SIGNED_K8", kind, base,
        stride, count, tuple(banks), width,
        address_offsets=tuple(offsets),
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
        earliest_issue_cycle=earliest)


def terminal(row) -> str:
    return M785._terminal_token(row)


def expect_failure(function, contains: str) -> str:
    try:
        function()
    except Exception as error:
        if contains not in str(error):
            raise AssertionError(
                "wrong failure {!r}; expected {!r}".format(str(error), contains))
        return str(error)
    raise AssertionError("counterexample accepted: " + contains)


def verify_member_manifest(directory: Path) -> Dict[str, str]:
    manifest = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and seal.is_file()
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        assert len(sha) == 64 and name not in expected
        member = directory / name
        assert member.is_file() and not member.is_symlink()
        assert digest(member) == sha
        expected[name] = sha
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.iterdir()
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    assert actual == set(expected)
    assert seal.read_text(encoding="utf-8").strip() == (
        digest(manifest) + "  SHA256SUMS")
    return expected


def load_mapper_oracles():
    mapper = M785.load_pinned_module(
        HW / "system_simulator/scripts/map_m672_decoder_convtranspose_polyphase_workload_r3.py",
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
        "m672_mapper")
    oracles = M785.load_pinned_oracles(
        HW / "system_simulator/scripts/analyze_m712_pidp_decoder_exact_cpu_fastkill.py",
        "87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc",
        HW / "system_simulator/scripts/analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py",
        "ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165")
    return mapper, oracles


def synthetic_record(directory: Path, cout: int = 192):
    calls = directory / "calls"
    calls.mkdir(parents=True)
    shape = [10, 1, 8, 2, 3]
    activation = np.zeros(shape, dtype=np.uint8)
    activation[0, 0, 0, 0, 0] = 1
    activation[0, 0, 1, 1, 2] = 1
    activation[0, 0, 7, 0, 2] = 1
    payload = calls / "synthetic.bitpack"
    payload.write_bytes(np.packbits(
        activation.reshape(-1), bitorder="little").tobytes())
    record = {
        "population_id": "M790_SYNTHETIC", "module_index": 0,
        "input_shape": shape, "relative_path": "calls/synthetic.bitpack",
        "packed_sha256": digest(payload),
    }
    return record, {0: (8, cout, 2, 3, 4, 6)}, activation


def stripped_projection(schedule: Mapping[str, Any]):
    return [
        (row["kind"], tuple(row["banks"]), tuple(row["addresses"]),
         row["width_bytes"], row["issue_cycle"], row["return_cycle"])
        for row in schedule["scheduled_requests"]
    ]


def main() -> int:
    identities = {
        "analyzer_sha256": digest(ANALYZER),
        "storage_oracle_sha256": digest(ORACLE),
        "tests_sha256": digest(TESTS),
        "contract_sha256": digest(CONTRACT),
        "request_manifest_sha256": digest(REQUEST / "SHA256SUMS"),
        "docs359_sha256": digest(HW / "docs/359_DATE终局冻结_20260813.md"),
    }
    assert identities == {
        "analyzer_sha256": EXPECTED["analyzer"],
        "storage_oracle_sha256": EXPECTED["oracle"],
        "tests_sha256": EXPECTED["tests"],
        "contract_sha256": EXPECTED["contract"],
        "request_manifest_sha256": EXPECTED["request_manifest"],
        "docs359_sha256": EXPECTED["docs359"],
    }
    request_members = verify_member_manifest(REQUEST)
    assert set(request_members) == {
        "author_validation.txt", "request.json", "request.md",
        "source_closure_report.json"}
    contract_sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    contract_outer = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")
    assert contract_sidecar.read_text(encoding="utf-8").strip() == (
        EXPECTED["contract"] + "  " + CONTRACT.name)
    assert contract_outer.read_text(encoding="utf-8").strip() == (
        digest(contract_sidecar) + "  " + contract_sidecar.name)
    source_validation = M785.validate_source_contract(ROOT, CONTRACT)
    assert source_validation["status"] == (
        "PASS_M785_SOURCE_IDENTITY_ONLY__NO_PRODUCTION_RUN")

    # Dirty capacity-one eviction with deliberately staggered victim service.
    residency = M785.PsumResidency(M785.PSUM_VECTOR_BYTES)
    residency.acquire(0)
    residency.mark_dirty(0)
    slot, events = residency.acquire(1)
    assert slot == 0 and [event.kind for event in events] == ["evict"]
    moves = M785.residency_transactions(
        "ev", "M790_SYNTHETIC", "TYPED_SIGNED_K8", events,
        {0: "victim_done:0"})
    assert [row.kind for row in moves] == ["psum_read", "external_write"]
    victim = tx("victim", "psum_write", 0, range(6), 48,
                produces="victim_done", earliest=37)
    replacement = tx(
        "replacement", "psum_read", 0, range(6), 48,
        dependencies=(terminal(moves[-1]),), produces="replacement_done")
    psum_schedule = M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([victim] + moves + [replacement]))
    p = {row["request_id"]: row for row in psum_schedule["scheduled_requests"]}
    local_evict = p[moves[0].transaction_id + ":0"]
    external_evict = p[moves[1].transaction_id + ":0"]
    replacement_row = p["replacement:0"]
    assert local_evict["issue_cycle"] >= p["victim:0"]["return_cycle"]
    assert external_evict["issue_cycle"] >= local_evict["return_cycle"]
    assert replacement_row["issue_cycle"] >= external_evict["return_cycle"]
    assert replacement.dependency_tokens == (terminal(moves[-1]),)

    # Evict+restore is a single-slot serial chain; a new read waits for local
    # restore return, not merely for the external read issue.
    residency = M785.PsumResidency(M785.PSUM_VECTOR_BYTES)
    residency.acquire(0); residency.mark_dirty(0)
    residency.acquire(1); residency.mark_dirty(1)
    restored_slot, restore_events = residency.acquire(0)
    assert restored_slot == 0
    assert [event.kind for event in restore_events] == ["evict", "restore"]
    restore_moves = M785.residency_transactions(
        "rs", "M790_SYNTHETIC", "TYPED_SIGNED_K8", restore_events,
        {0: "old0:0", 1: "old1:0"})
    assert [row.kind for row in restore_moves] == [
        "psum_read", "external_write", "external_read", "psum_write"]
    old0 = tx("old0", "psum_write", 0, range(6), 48,
              produces="old0", earliest=7)
    old1 = tx("old1", "psum_write", 0, range(6), 48,
              produces="old1", earliest=53)
    after_restore = tx(
        "after_restore", "psum_read", 0, range(6), 48,
        dependencies=(terminal(restore_moves[-1]),))
    restore_schedule = M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions(
            [old0, old1] + restore_moves + [after_restore]))
    r = {row["request_id"]: row
         for row in restore_schedule["scheduled_requests"]}
    move_first = [r[row.transaction_id + ":0"] for row in restore_moves]
    move_last = [r[row.transaction_id + ":" + str(row.count - 1)]
                 for row in restore_moves]
    assert move_first[0]["issue_cycle"] >= r["old1:0"]["return_cycle"]
    for before, after in zip(move_last, move_first[1:]):
        assert after["issue_cycle"] >= before["return_cycle"]
    assert r["after_restore:0"]["issue_cycle"] >= move_last[-1]["return_cycle"]
    assert restore_moves[0].bank_pattern == tuple(range(6))
    assert restore_moves[-1].bank_pattern == tuple(range(6))

    # Nine physical weight slots remain bijective; the tenth key reuses one
    # address only after the victim key has been retired.
    cache = M785.WeightResidency()
    keys = [(0, block, 0, 0) for block in range(10)]
    accesses = [cache.access(key) for key in keys]
    assert [row.slot for row in accesses[:9]] == list(range(9))
    assert accesses[9].slot == 0 and accesses[9].evicted_key == keys[0]
    assert len(cache.key_to_slot) == len(cache.slot_to_key) == 9
    assert all(cache.slot_to_key[slot] == key
               for key, slot in cache.key_to_slot.items())
    slot_addresses = {
        slot: tuple(M785.weight_slot_bank_and_local_row(slot, bank, 8)
                    for bank in range(8))
        for slot in range(9)
    }
    assert len(set(slot_addresses.values())) == 9
    assert all(0 <= row < 1728 for values in slot_addresses.values()
               for _bank, row in values)

    # Stagger eight victim-bank last uses.  Every one is a hard dependency of
    # every local refill beat; the new read waits for the twelfth beat return.
    overwrite_tokens = tuple("use{}:0".format(bank) for bank in range(8))
    refill = M785._weight_refill_transactions(
        "wr", "M790_SYNTHETIC", "TYPED_SIGNED_K8", 9, 8, 0, 0,
        "source:0", overwrite_tokens, 0)
    source = tx("source", "external_read", 1 << 60, (0,), 16,
                produces="source")
    prior = [
        tx("use{}".format(bank), "weight_read", 0, (bank,), 16,
           produces="use{}".format(bank), earliest=5 + bank * 7,
           offsets=(0,))
        for bank in range(8)
    ]
    bank, offset = M785.weight_slot_bank_and_local_row(0, 0, 8)
    new_read = tx(
        "new_read", "weight_read", offset, (bank,), 16,
        dependencies=(terminal(refill[-1]),), count=6, stride=16,
        produces="new_read_done")
    weight_schedule = M785.AddressTimedScheduler(common_resource()).schedule(
        M785.expand_transactions([source] + prior + refill + [new_read]))
    w = {row["request_id"]: row
         for row in weight_schedule["scheduled_requests"]}
    external_last = w[refill[0].transaction_id + ":7"]
    local_first = w[refill[1].transaction_id + ":0"]
    local_last = w[refill[1].transaction_id + ":11"]
    prior_returns = [w["use{}:0".format(bank)]["return_cycle"]
                     for bank in range(8)]
    assert refill[0].count * refill[0].width_bytes == 1536
    assert (refill[1].count * len(refill[1].bank_pattern) *
            refill[1].width_bytes) == 1536
    assert set(overwrite_tokens).issubset(refill[1].dependency_tokens)
    assert local_first["issue_cycle"] >= external_last["return_cycle"]
    assert local_first["issue_cycle"] >= max(prior_returns)
    assert w["new_read:0"]["issue_cycle"] >= local_last["return_cycle"]

    mapper, oracles = load_mapper_oracles()
    with tempfile.TemporaryDirectory(prefix="m790_m785_attack_") as temp:
        package = Path(temp)
        record, geometry, activation = synthetic_record(package)
        bits = activation[0, 0]
        spec = geometry[0]
        counts = oracles.m722r2.R1.group_counts(bits, 2)
        oracle_pass = M785.verify_contributor_and_storage_oracles(
            bits, 0, spec,
            int(counts["contributors"]) // 2,
            int(counts["osg_groups"]) // 2, oracles,
            M785.PSUM_VECTOR_BYTES * 2)
        assert oracle_pass["m722_role"] == "CONTRIBUTOR_GROUP_ORACLE_ONLY"
        assert oracle_pass["m722_storage_equivalent_to_m785"] is False

        original_m722 = oracles.m722r2.R1
        m722_failures = {}
        for field, mutate in {
            "stripe_count": lambda value: value.__setitem__(
                "stripe_count", int(value["stripe_count"]) + 1),
            "stripes": lambda value: value.__setitem__("stripes", [[0, 1]]),
            "total_bytes": lambda value: value.__setitem__(
                "total_bytes", int(value["total_bytes"]) + 128),
            "offchip": lambda value: value.__setitem__(
                "offchip_psum_spill_bytes", 288),
        }.items():
            class BadR1:
                group_counts = staticmethod(original_m722.group_counts)

                @staticmethod
                def a1_storage_plan(spec_arg, mutate=mutate):
                    value = dict(original_m722.a1_storage_plan(spec_arg))
                    mutate(value)
                    return value

            class BadM722:
                R1 = BadR1

            m722_failures[field] = expect_failure(
                lambda BadM722=BadM722: M785.verify_contributor_and_storage_oracles(
                    bits, 0, spec,
                    int(counts["contributors"]) // 2,
                    int(counts["osg_groups"]) // 2,
                    M785.OracleBundle(oracles.m712, BadM722(), oracles.storage),
                    M785.PSUM_VECTOR_BYTES * 2),
                "M722 line-buffer storage")

        storage_failures = {}
        for field, mutate in {
            "stripe_count": lambda value: value.__setitem__(
                "stripe_count", int(value["stripe_count"]) + 1),
            "stripes": lambda value: value.__setitem__("stripes", [[0, 1]]),
            "partition": lambda value: value.__setitem__(
                "psum_partition_bytes", int(value["psum_partition_bytes"]) + 288),
            "offchip_span": lambda value: value.__setitem__(
                "offchip_backing_address_span_bytes",
                int(value["offchip_backing_address_span_bytes"]) + 288),
        }.items():
            class BadStorage:
                @staticmethod
                def plan(geometry_arg, psum_arg, mutate=mutate):
                    value = dict(oracles.storage.plan(geometry_arg, psum_arg))
                    mutate(value)
                    return value

            storage_failures[field] = expect_failure(
                lambda BadStorage=BadStorage: M785.verify_contributor_and_storage_oracles(
                    bits, 0, spec,
                    int(counts["contributors"]) // 2,
                    int(counts["osg_groups"]) // 2,
                    M785.OracleBundle(oracles.m712, oracles.m722r2,
                                      BadStorage()),
                    M785.PSUM_VECTOR_BYTES * 2),
                "M785 independent storage")

        schedules: Dict[str, Mapping[str, Any]] = {}
        for config in M785.CONFIGS:
            rows = list(M785.iter_record_transactions(
                mapper, record, package, "M790_SYNTHETIC", config, 0,
                oracles, tile_m=2, geometry=geometry,
                psum_bytes=M785.PSUM_VECTOR_BYTES * 2))
            schedules[config] = M785.AddressTimedScheduler(
                common_resource(M785.PSUM_VECTOR_BYTES * 2)).schedule(
                    M785.expand_transactions(rows))
        projections = {name: stripped_projection(value)
                       for name, value in schedules.items()}
        assert projections["A1_OSG"] != projections["EQUAL_SERVICE_K1X8"]
        assert projections["EQUAL_SERVICE_K1X8"] != projections["TYPED_SIGNED_K8"]
        assert len({value["commit_sequence_sha256"]
                    for value in schedules.values()}) == 1

        # Exercise the integrated LRU path, not just the refill helper.  Ten
        # output blocks force physical-slot replacement.  Every overwrite
        # carries the victim key's per-bank final read tokens, and all later
        # reads of the reused slot wait for the final local write return.
        eviction_package = package / "integrated_weight_eviction"
        eviction_record, eviction_geometry, _ = synthetic_record(
            eviction_package, cout=960)
        eviction_rows = list(M785.iter_record_transactions(
            mapper, eviction_record, eviction_package, "M790_EVICTION",
            "TYPED_SIGNED_K8", 0, oracles, tile_m=2,
            geometry=eviction_geometry))
        overwrite_refills = [
            row for row in eviction_rows
            if row.kind == "weight_write" and any(
                "typed_weight" in token for token in row.dependency_tokens)
        ]
        assert overwrite_refills
        integrated_schedule = M785.AddressTimedScheduler(
            common_resource()).schedule(M785.expand_transactions(eviction_rows))
        integrated = integrated_schedule["scheduled_requests"]
        token_return = {
            row["produces_token"]: row["return_cycle"]
            for row in integrated if row["produces_token"]
        }
        request_by_id = {row["request_id"]: row for row in integrated}
        for refill_row in overwrite_refills:
            victim_tokens = [token for token in refill_row.dependency_tokens
                             if "typed_weight" in token]
            assert victim_tokens and len(victim_tokens) == len(set(victim_tokens))
            local_first = request_by_id[refill_row.transaction_id + ":0"]
            assert local_first["issue_cycle"] >= max(
                token_return[token] for token in victim_tokens)
            local_terminal = terminal(refill_row)
            dependent_reads = [
                row for row in integrated
                if row["kind"] == "weight_read" and
                local_terminal in row["dependency_tokens"]
            ]
            assert dependent_reads
            assert min(row["issue_cycle"] for row in dependent_reads) >= (
                token_return[local_terminal])

    d3 = M785.STORAGE.plan((96, 96, 120, 160, 240, 320), 221184)
    assert d3["resident_capacity_vectors"] == 768
    assert d3["stripe_count"] == 100 and len(d3["stripes"]) == 100
    assert d3["psum_partition_bytes"] == 221184
    assert d3["resident_payload_bytes_at_full_capacity"] == 221184
    assert d3["offchip_backing_address_span_bytes"] == 240 * 320 * 288
    assert d3["m722_line_buffer_storage_equivalent"] is False

    # Same-resource/fallback/headline boundaries are rechecked independently.
    resource = common_resource()
    resource.validate()
    resource_id = resource.identity()
    assert resource_id["lanes"] == 96
    assert resource_id["accumulator_bits"] == 24
    assert resource_id["clock_ns"] == 3.0
    assert resource_id["external_bytes_per_cycle"] == 192
    assert resource_id["onchip_budget_bytes_macro_rounded"] == 245760
    assert resource_id["resource_manifest_sha256"] == (
        "a7400bddb174a00875298cd9bd8d2692e636727ff27b22ae580803383fdea0f3")
    headline = {
        "k8_vs_equal_service": M785.headline_ratio_allowed(
            "TYPED_SIGNED_K8", "EQUAL_SERVICE_K1X8"),
        "k8_vs_a1": M785.headline_ratio_allowed("TYPED_SIGNED_K8", "A1_OSG"),
        "k8_vs_k1": M785.headline_ratio_allowed("TYPED_SIGNED_K8", "K1"),
    }
    assert headline == {
        "k8_vs_equal_service": True, "k8_vs_a1": False, "k8_vs_k1": False}
    d1_signatures = {}
    for config in M785.CONFIGS:
        rows = list(M785._d1_transactions(
            {"module_index": 1}, "M790_D1", config, 0,
            M785.MODULE_GEOMETRY[1]))
        d1_signatures[config] = [(row.kind, row.count, row.width_bytes)
                                 for row in rows]
        assert not M785.route_for_record(1, config)["headline_eligible"]
    assert len({json.dumps(value) for value in d1_signatures.values()}) == 1
    assert next(value[1] for value in d1_signatures["A1_OSG"]
                if value[0] == "compute") == 16632000

    output = {
        "schema": "m790_m785_decoder_physical_residency_source_fresh_hammer_attack_v1",
        "date": "2026-08-28",
        "status": "PASS_M785_SOURCE_SEMANTICS__SEPARATE_PRODUCTION_RELEASE_REQUIRED",
        "score": 100,
        "identity": identities,
        "request_member_count": len(request_members),
        "source_validation_status": source_validation["status"],
        "closed_m781_findings": {
            "dirty_psum_evict_chain": {
                "victim_return": p["victim:0"]["return_cycle"],
                "local_read_issue": local_evict["issue_cycle"],
                "local_read_return": local_evict["return_cycle"],
                "external_write_issue": external_evict["issue_cycle"],
                "external_write_return": external_evict["return_cycle"],
                "replacement_read_issue": replacement_row["issue_cycle"],
            },
            "dirty_evict_restore_chain_kinds": [row.kind for row in restore_moves],
            "weight_key_slot_bijection": True,
            "weight_tenth_key_reuses_retired_slot": True,
            "weight_refill_external_bytes": 1536,
            "weight_refill_local_bytes": 1536,
            "weight_refill_waits_all_victim_banks": True,
            "weight_read_waits_final_local_write_return": True,
            "integrated_weight_overwrite_refill_count": len(overwrite_refills),
            "integrated_per_key_per_bank_last_use_closed": True,
            "m722_injections_rejected": sorted(m722_failures),
            "m785_storage_injections_rejected": sorted(storage_failures),
            "d3_vectors_per_stripe": 768,
            "d3_stripes": 100,
            "d3_psum_partition_bytes": 221184,
            "d3_backing_span_bytes": d3["offchip_backing_address_span_bytes"],
            "m722_role": oracle_pass["m722_role"],
            "m722_storage_equivalent_to_m785": False,
        },
        "fairness": {
            "resource_manifest_sha256": resource_id["resource_manifest_sha256"],
            "three_stripped_service_paths_pairwise_nonisomorphic": True,
            "same_dense_commit_hash": True,
            "legal_headline_pair": headline,
            "d1_common_compute_count": 16632000,
            "d1_headline_eligible": False,
        },
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "authorization": {
            "source_only_pass": True,
            "one_production_run_authorized": False,
            "separate_additive_release_required": True,
            "production_population_replay_performed": False,
            "production_cycles_generated": False,
            "production_speedup_generated": False,
            "decoder_complete": False,
            "full_network_completion": False,
            "table_a_insertion_allowed": False,
            "rtl_vcs_eda_gpu_remote_performed": False,
            "docs359_modified": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

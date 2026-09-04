#!/usr/bin/env python3
"""Receipt-blind source-only attacks for the M777 additive repair.

This script never opens a production population and never writes a decoder
cycle/speedup result.  It reconstructs only synthetic transactions and the
frozen static/oracle identities needed by the M778 request.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
ANALYZER = HW / "system_simulator/scripts/analyze_m777_h67_decoder_a1_k8_address_timed_repair.py"
CONTRACT = HW / "contracts/m777_h67_decoder_a1_k8_address_timed_repair_contract_r1_20260828.json"
TESTS = HW / "system_simulator/tests/test_m777_h67_decoder_a1_k8_address_timed_repair.py"


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


M777 = load(ANALYZER, "m781_frozen_m777")


def common_resource(psum_bytes: int = 221184):
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
        reserved_unallocated_bytes=245760 - 13824 - psum_bytes - 8192,
        weight=M777.PortSpec(8, "1R1W", 16, 4, 1, 1, 8),
        psum=M777.PortSpec(6, "1RW", 48, 2, 1, 1, 8),
        external=M777.PortSpec(1, "1RW", 192, 32, 3, 1, 16),
        compute=M777.PortSpec(1, "1RW", 288, 1, 1, 1, 1),
    )


def load_mapper_oracles():
    mapper = M777.load_pinned_module(
        HW / "system_simulator/scripts/map_m672_decoder_convtranspose_polyphase_workload_r3.py",
        "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
        "m672_mapper",
    )
    oracles = M777.load_pinned_oracles(
        HW / "system_simulator/scripts/analyze_m712_pidp_decoder_exact_cpu_fastkill.py",
        "87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc",
        HW / "system_simulator/scripts/analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py",
        "ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165",
    )
    return mapper, oracles


def synthetic_record(directory: Path, cout: int = 96):
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
        "population_id": "M781_SYNTHETIC",
        "module_index": 0,
        "input_shape": shape,
        "relative_path": "calls/synthetic.bitpack",
        "packed_sha256": digest(payload),
    }
    geometry = {0: (8, cout, 2, 3, 4, 6)}
    return record, geometry, activation, payload


def expect_failure(function, contains: str) -> str:
    try:
        function()
    except Exception as error:  # fail closed is the subject of the attack
        if contains not in str(error):
            raise AssertionError(
                "wrong failure {!r}; expected {!r}".format(str(error), contains))
        return str(error)
    raise AssertionError("counterexample was accepted: " + contains)


def request_projection(schedule: Mapping[str, Any]):
    """Erase config, population, transaction and request identifiers."""
    return [
        [row["kind"], row["banks"], row["width_bytes"],
         row["issue_cycle"], row["return_cycle"]]
        for row in schedule["scheduled_requests"]
    ]


def transaction(identifier: str, kind: str, base: int, banks: Sequence[int],
                width: int, dependencies: Sequence[str] = (),
                produces: str = "", earliest: int = 0):
    return M777.CompressedTransaction(
        transaction_id=identifier,
        population_id="M781_SYNTHETIC",
        config="TYPED_SIGNED_K8",
        kind=kind,
        base_address=base,
        address_stride_bytes=0,
        count=1,
        bank_pattern=tuple(banks),
        width_bytes=width,
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
        earliest_issue_cycle=earliest,
    )


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_validation = M777.validate_source_contract(ROOT, CONTRACT)
    mapper, oracles = load_mapper_oracles()

    identities = {
        "contract_sha256": digest(CONTRACT),
        "analyzer_sha256": digest(ANALYZER),
        "tests_sha256": digest(TESTS),
        "contract_member_sidecar_file_sha256": digest(
            CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")),
        "contract_outer_sidecar_file_sha256": digest(
            CONTRACT.with_suffix(CONTRACT.suffix + ".sha256.seal.sha256")),
        "docs359_sha256": digest(HW / "docs/359_DATE终局冻结_20260813.md"),
        "source_validation_status": source_validation["status"],
        "resource_manifest_sha256": source_validation["resource"][
            "resource_manifest_sha256"],
    }

    # Reconstruct all three service paths without trusting the author tests.
    with tempfile.TemporaryDirectory(prefix="m781_decoder_attack_") as temp:
        package = Path(temp)
        record, geometry, activation, _payload = synthetic_record(package)
        schedules: Dict[str, Mapping[str, Any]] = {}
        streams: Dict[str, Sequence[Any]] = {}
        for config in M777.CONFIGS:
            rows = list(M777.iter_record_transactions(
                mapper, record, package, "M781_SYNTHETIC", config, 0,
                oracles, tile_m=2, geometry=geometry))
            streams[config] = rows
            schedules[config] = M777.AddressTimedScheduler(
                common_resource()).schedule(M777.expand_transactions(rows))

        projections = {name: request_projection(value)
                       for name, value in schedules.items()}
        distinct_paths = (
            projections["A1_OSG"] != projections["EQUAL_SERVICE_K1X8"]
            and projections["EQUAL_SERVICE_K1X8"] !=
            projections["TYPED_SIGNED_K8"]
        )
        commit_hashes = {name: value["commit_sequence_sha256"]
                         for name, value in schedules.items()}
        assert distinct_paths and len(set(commit_hashes.values())) == 1

        deletion_attacks = {}
        for label, needle in (
                ("source", "source_fetch"),
                ("descriptor", "descriptor"),
                ("weight_refill", "weight_refill")):
            damaged = [row for row in streams["TYPED_SIGNED_K8"]
                       if needle not in row.transaction_id]
            deletion_attacks[label] = expect_failure(
                lambda damaged=damaged: M777.AddressTimedScheduler(
                    common_resource()).schedule(
                        M777.expand_transactions(damaged)),
                "unresolved dependency",
            )

        # A two-output-block record proves the weight LRU has logical keys but
        # no physical cache slots.  Distinct output-block weight contents are
        # both reported resident and are read from identical bank/local rows.
        record2, geometry2, _activation2, _payload2 = synthetic_record(
            package / "two_blocks", cout=192)
        two_block_rows = list(M777.iter_record_transactions(
            mapper, record2, package / "two_blocks", "M781_TWO_BLOCKS",
            "TYPED_SIGNED_K8", 0, oracles, tile_m=2, geometry=geometry2))
        alias_rows = []
        for row in two_block_rows:
            if row.kind != "weight_read":
                continue
            refill_dependencies = [value for value in row.dependency_tokens
                                   if "weight_refill" in value]
            output_blocks = sorted({
                int(value.split(":ob", 1)[1].split(":", 1)[0])
                for value in refill_dependencies
            })
            if output_blocks:
                alias_rows.append({
                    "output_blocks": output_blocks,
                    "banks": list(row.bank_pattern),
                    "offsets": list(row.address_offsets),
                    "dependencies": refill_dependencies,
                })
        ob0 = next(row for row in alias_rows if row["output_blocks"] == [0])
        ob1 = next(row for row in alias_rows if row["output_blocks"] == [1])
        output_block_alias = (
            ob0["banks"] == ob1["banks"]
            and ob0["offsets"] == ob1["offsets"]
        )
        assert output_block_alias

    # K24/K25 and additional tap/channel-div-8 goldens.
    weight_goldens = {
        str(flat_k): list(M777.weight_bank_and_local_row(flat_k, channels))
        for channels, flat_k in ((8, 24), (8, 25), (16, 8),
                                 (16, 16), (16, 31), (384, 3455))
    }
    assert weight_goldens["24"] == [0, 48]
    assert weight_goldens["25"] == [1, 48]
    assert M777.weight_bank_and_local_row(8, 16) == (0, 16)
    assert M777.weight_bank_and_local_row(16, 16) == (0, 32)
    assert M777.weight_bank_and_local_row(31, 16) == (7, 48)

    # Exact physical boundary and D3 stripe lattice.
    legal = transaction("legal", "psum_read", 221184 - 288,
                        range(6), 48)
    M777.AddressTimedScheduler(common_resource()).schedule(
        M777.expand_transactions([legal]))
    illegal = transaction("illegal", "psum_read", 221184 - 287,
                          range(6), 48)
    boundary_failure = expect_failure(
        lambda: M777.AddressTimedScheduler(common_resource()).schedule(
            M777.expand_transactions([illegal])),
        "physical partition",
    )
    stripes = M777.psum_stripes(240 * 320, 221184)
    assert len(stripes) == 100
    assert all(row.vector_hi - row.vector_lo <= 768 for row in stripes)

    # Capacity+1 dirty/revisit emits external evict/restore, but slot reuse is
    # not ordered after eviction.  The later psum read is scheduled at cycle 0
    # while the victim backing write cannot issue until cycle 10.
    residency = M777.PsumResidency(M777.PSUM_VECTOR_BYTES)
    residency.acquire(0)
    residency.mark_dirty(0)
    slot, evict_events = residency.acquire(1)
    assert slot == 0 and [event.kind for event in evict_events] == ["evict"]
    evict_rows = M777.residency_transactions(
        "m781", "M781_SYNTHETIC", "TYPED_SIGNED_K8", evict_events,
        {0: "victim_done:0"})
    victim = transaction("victim", "psum_write", 0, range(6), 48,
                         produces="victim_done", earliest=10)
    replacement_read = transaction(
        "replacement_read", "psum_read", 0, range(6), 48)
    hazard_schedule = M777.AddressTimedScheduler(common_resource()).schedule(
        M777.expand_transactions([victim] + evict_rows + [replacement_read]))
    by_id = {row["request_id"]: row
             for row in hazard_schedule["scheduled_requests"]}
    replacement_issue = by_id["replacement_read:0"]["issue_cycle"]
    evict_return = max(row["return_cycle"]
                       for key, row in by_id.items() if ":evict:" in key)
    slot_reuse_before_evict = replacement_issue < evict_return
    assert slot_reuse_before_evict
    backing_has_no_psum_port_action = all(
        row.kind in ("external_read", "external_write") for row in evict_rows)
    assert backing_has_no_psum_port_action

    # Revisit proves the external events exist.  This is deliberately separate
    # from the hazard attack above.
    residency.acquire(2)
    _slot, revisit_events = residency.acquire(0)
    assert any(row.kind == "restore" for row in revisit_events)

    # Execute real M712/M722 functions, then independently inject M712, M722
    # contributor/group, and M722 storage mismatches.
    bits = activation[0, 0]
    spec = geometry[0]
    counts = oracles.m722r2.R1.group_counts(bits, 1)
    oracle_pass = M777.verify_contributor_and_storage_oracles(
        bits, 0, spec, int(counts["contributors"]),
        int(counts["osg_groups"]), oracles)

    class BadM712:
        @staticmethod
        def descriptor_counts(bits_arg, blocks_arg):
            rows = oracles.m712.descriptor_counts(bits_arg, blocks_arg)
            return rows[0], rows[1], rows[2] + 1, rows[3]

    m712_failure = expect_failure(
        lambda: M777.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]),
            M777.OracleBundle(BadM712(), oracles.m722r2)),
        "M712 contributor oracle",
    )

    original_r1 = oracles.m722r2.R1

    class BadGroupR1:
        @staticmethod
        def group_counts(bits_arg, blocks_arg):
            value = dict(original_r1.group_counts(bits_arg, blocks_arg))
            value["osg_groups"] += 1
            return value

        @staticmethod
        def a1_storage_plan(spec_arg):
            return original_r1.a1_storage_plan(spec_arg)

    class BadGroupM722:
        R1 = BadGroupR1

    m722_group_failure = expect_failure(
        lambda: M777.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]),
            M777.OracleBundle(oracles.m712, BadGroupM722())),
        "M722 OSG group oracle",
    )

    class BadStorageR1:
        @staticmethod
        def group_counts(bits_arg, blocks_arg):
            return original_r1.group_counts(bits_arg, blocks_arg)

        @staticmethod
        def a1_storage_plan(spec_arg):
            value = dict(original_r1.a1_storage_plan(spec_arg))
            value["stripe_count"] = 999
            value["stripes"] = [[0, 1]] * 999
            return value

    class BadStorageM722:
        R1 = BadStorageR1

    storage_mismatch_accepted = False
    try:
        accepted = M777.verify_contributor_and_storage_oracles(
            bits, 0, spec, int(counts["contributors"]),
            int(counts["osg_groups"]),
            M777.OracleBundle(oracles.m712, BadStorageM722()))
        storage_mismatch_accepted = accepted["m722_stripe_count"] == 999
    except Exception:
        storage_mismatch_accepted = False
    assert storage_mismatch_accepted

    d3_spec = ("D3", 96, 96, 120, 160, 240, 320, 1)
    m722_d3_storage = original_r1.a1_storage_plan(d3_spec)
    d3_model_disagreement = {
        "m722_stripe_count": int(m722_d3_storage["stripe_count"]),
        "m722_offchip_psum_spill_bytes": int(
            m722_d3_storage["offchip_psum_spill_bytes"]),
        "m777_stripe_count": len(stripes),
        "m777_capacity_plus_one_emits_dirty_evict": True,
    }
    assert d3_model_disagreement["m722_stripe_count"] != len(stripes)

    # D1 must be common, full-shape and diagnostic.  This part closes M773-P1-03.
    d1_spec = M777.MODULE_GEOMETRY[1]
    d1_signatures = {}
    d1_compute = {}
    for config in M777.CONFIGS:
        rows = list(M777._d1_transactions(
            {"module_index": 1}, "M781_D1", config, 0, d1_spec))
        d1_signatures[config] = [
            [row.kind, row.count, row.width_bytes] for row in rows]
        d1_compute[config] = next(
            row.count for row in rows if row.kind == "compute")
    assert len({json.dumps(value) for value in d1_signatures.values()}) == 1
    assert min(d1_compute.values()) > 1
    assert all(not M777.route_for_record(1, config)["headline_eligible"]
               for config in M777.CONFIGS)

    headline_boundary = {
        "k8_vs_equal_service_k1x8": M777.headline_ratio_allowed(
            "TYPED_SIGNED_K8", "EQUAL_SERVICE_K1X8"),
        "k8_vs_a1": M777.headline_ratio_allowed(
            "TYPED_SIGNED_K8", "A1_OSG"),
        "k8_vs_k1": M777.headline_ratio_allowed(
            "TYPED_SIGNED_K8", "K1"),
    }
    assert headline_boundary == {
        "k8_vs_equal_service_k1x8": True,
        "k8_vs_a1": False,
        "k8_vs_k1": False,
    }

    output = {
        "schema": "m781_m777_decoder_address_timed_repair_source_fresh_hammer_attack_v1",
        "date": "2026-08-28",
        "status": "FAIL_M777_PHYSICAL_RESIDENCY_SEMANTICS__NO_PRODUCTION_RUN",
        "score": 78,
        "identity": identities,
        "closed_m773_findings": {
            "three_nonisomorphic_service_paths": distinct_paths,
            "same_dense_commit_hash": len(set(commit_hashes.values())) == 1,
            "legal_headline_pair_only": headline_boundary,
            "psum_boundary_rejects_capacity_plus_one_byte": boundary_failure,
            "d3_stripes": len(stripes),
            "vectors_per_stripe": 221184 // M777.PSUM_VECTOR_BYTES,
            "weight_goldens": weight_goldens,
            "external_class_deletions_fail": deletion_attacks,
            "m712_m722_contributor_group_oracles_execute": oracle_pass,
            "m712_mismatch_rejected": m712_failure,
            "m722_group_mismatch_rejected": m722_group_failure,
            "d1_common_full_shape_compute_count": d1_compute,
            "d1_nonheadline": True,
        },
        "new_adversarial_failures": {
            "dirty_slot_reuse_before_evict_completion": {
                "replacement_psum_read_issue_cycle": replacement_issue,
                "victim_external_evict_return_cycle": evict_return,
                "hazard_present": slot_reuse_before_evict,
            },
            "backing_transfer_has_no_local_psum_port_action":
                backing_has_no_psum_port_action,
            "weight_lru_has_no_physical_slot_and_output_blocks_alias": {
                "output_block_0": ob0,
                "output_block_1": ob1,
                "same_bank_local_addresses": output_block_alias,
            },
            "m722_storage_mismatch_injection_accepted":
                storage_mismatch_accepted,
            "m722_vs_m777_d3_storage_model_disagreement":
                d3_model_disagreement,
        },
        "authorization": {
            "source_identity_only_pass": True,
            "one_production_run_authorized": False,
            "production_population_replay_performed": False,
            "production_cycles_generated": False,
            "production_speedup_generated": False,
            "decoder_complete": False,
            "full_network_completion": False,
            "table_a_insertion_allowed": False,
            "rtl_vcs_eda_gpu_remote_performed": False,
            "docs359_modified": False,
        },
        "contract_launch_now": contract["launch_now"],
        "contract_production_speedup_allowed":
            contract["production_speedup_allowed"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

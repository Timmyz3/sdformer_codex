#!/usr/bin/env python3
"""Fresh, source-only adversarial audit of M768.

This program deliberately does not replay M686 or M699 through the cycle
generator.  It validates their sealed identities, reruns the author tests,
and uses synthetic payloads/requests to attack the execution semantics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
ANALYZER = HW / "system_simulator/scripts/analyze_m768_h67_decoder_a1_k8_address_timed_cycles.py"
TESTS = HW / "system_simulator/tests/test_m768_h67_decoder_a1_k8_address_timed_cycles.py"
CONTRACT = HW / "contracts/m768_h67_decoder_a1_k8_address_timed_cycle_contract_r1_20260828.json"
REQUEST = HW / "reviews/m769_m768_decoder_address_timed_source_fresh_hammer_r1_REQUEST_20260828/request.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m768_fresh_attack", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M768 = load_module()


def resource(psum_mode: str = "1RW", outstanding: int = 8):
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


def tx(identifier: str, kind: str, *, bank: int = 0, dependencies=(),
       produces: str = "", address: int = 0x1000, width: int = 16):
    return M768.CompressedTransaction(
        identifier, "SYNTHETIC_PRIMARY", "TYPED_SIGNED_K8", kind,
        address, 0, 1, (bank,), width,
        dependency_tokens=tuple(dependencies),
        produces_token_prefix=produces,
    )


def seal_directory(path: Path) -> None:
    members = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    )
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    manifest.write_text("".join(
        f"{sha256(item)}  {item.relative_to(path).as_posix()}\n"
        for item in members
    ), encoding="utf-8")
    outer.write_text(f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")


def semantic_projection(rows):
    return [
        (
            row.kind,
            row.base_address,
            row.address_stride_bytes,
            row.count,
            row.bank_pattern,
            row.width_bytes,
            row.address_offsets,
            len(row.dependency_tokens),
            bool(row.produces_token_prefix),
        )
        for row in rows
    ]


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    checks = {}

    checks["pinned_source_sha"] = {
        "contract": sha256(CONTRACT),
        "analyzer": sha256(ANALYZER),
        "tests": sha256(TESTS),
        "request": sha256(REQUEST),
        "matches_request": (
            sha256(CONTRACT) == request["pinned_sources"]["contract"]["sha256"]
            and sha256(ANALYZER) == request["pinned_sources"]["analyzer"]["sha256"]
            and sha256(TESTS) == request["pinned_sources"]["tests"]["sha256"]
        ),
    }

    sidecar = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    checks["contract_double_seal"] = {
        "member_sidecar_file_sha256": sha256(sidecar),
        "outer_seal_file_sha256": sha256(outer),
        "member_line_exact": sidecar.read_text(encoding="utf-8").strip()
        == f"{sha256(CONTRACT)}  {CONTRACT.name}",
        "outer_line_exact": outer.read_text(encoding="utf-8").strip()
        == f"{sha256(sidecar)}  {sidecar.name}",
    }

    docs359 = HW / "docs/359_DATE终局冻结_20260813.md"
    checks["docs359_sha256"] = sha256(docs359)

    sealed = {}
    for key in ("primary_m686", "primary_m692_review", "secondary_m699", "secondary_m705_review"):
        row = contract["inputs"][key]
        path = HW / row["directory"]
        identity = M768.verify_sealed_directory(path)
        sealed[key] = identity
    checks["sealed_inputs"] = sealed

    primary_manifest = M768.strict_json(
        HW / contract["inputs"]["primary_m686"]["directory"] / "manifest.json"
    )
    secondary_manifest = M768.strict_json(
        HW / contract["inputs"]["secondary_m699"]["directory"] / "manifest.json"
    )
    primary = M768.normalized_population_records(primary_manifest, "PRIMARY")
    secondary = M768.normalized_population_records(secondary_manifest, "SECONDARY")
    checks["population_lattices"] = {
        "primary_records": len(primary),
        "primary_unique_sequence_sample_module": len({
            (r["sequence"], r["sample_id"], r["module_index"]) for r in primary
        }),
        "secondary_records": len(secondary),
        "secondary_unique_sequence_sample_module": len({
            (r["sequence"], r["sample_id"], r["module_index"]) for r in secondary
        }),
        "secondary_sequences": sorted({r["sequence"] for r in secondary}),
    }

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TESTS)],
        cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    validation_run = subprocess.run(
        [
            sys.executable, str(ANALYZER), "--validate-source-contract",
            "--repo-root", str(REPO), "--contract", str(CONTRACT),
        ],
        cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    checks["author_tests"] = {
        "returncode": pytest_run.returncode,
        "output": pytest_run.stdout.strip(),
    }
    checks["source_identity_validation"] = {
        "returncode": validation_run.returncode,
        "status": (json.loads(validation_run.stdout)["status"]
                   if validation_run.returncode == 0 else None),
    }

    with tempfile.TemporaryDirectory(prefix="m773_seal_attack_") as tmp:
        package = Path(tmp) / "sealed"
        package.mkdir()
        member = package / "member.bin"
        member.write_bytes(b"original")
        seal_directory(package)
        M768.verify_sealed_directory(package)
        member.write_bytes(b"mutated")
        rejected = False
        try:
            M768.verify_sealed_directory(package)
        except M768.Failure:
            rejected = True
        checks["sealed_member_mutation_rejected"] = rejected

    common = resource()
    common.validate()
    over_capacity_rejected = False
    try:
        M768.CommonResource(
            **dict(common.__dict__, descriptor_control_bytes_logical=10753,
                   reserved_unallocated_bytes=0)
        ).validate()
    except M768.Failure:
        over_capacity_rejected = True
    checks["capacity_manifest"] = {
        "allocated": common.allocated_macro_rounded_bytes,
        "reserved": common.reserved_unallocated_bytes,
        "total": common.allocated_macro_rounded_bytes + common.reserved_unallocated_bytes,
        "one_byte_macro_round_cliff_rejected": over_capacity_rejected,
    }

    same_bank = M768.AddressTimedScheduler(resource()).schedule(
        M768.expand_transactions([tx("r0", "weight_read"), tx("r1", "weight_read")])
    )
    distinct_bank = M768.AddressTimedScheduler(resource()).schedule(
        M768.expand_transactions([tx("r0", "weight_read", bank=0), tx("r1", "weight_read", bank=1)])
    )
    rw_rows = [tx("read", "psum_read", width=48), tx("write", "psum_write", width=48)]
    one_rw = M768.AddressTimedScheduler(resource("1RW")).schedule(M768.expand_transactions(rw_rows))
    one_r_one_w = M768.AddressTimedScheduler(resource("1R1W")).schedule(M768.expand_transactions(rw_rows))
    slot = M768.AddressTimedScheduler(resource(outstanding=1)).schedule(
        M768.expand_transactions([tx("s0", "weight_read"), tx("s1", "weight_read")])
    )
    dependency_rejected = False
    try:
        M768.AddressTimedScheduler(resource()).schedule(
            M768.expand_transactions([tx("bad", "compute", dependencies=("missing",), width=288)])
        )
    except M768.Failure:
        dependency_rejected = True
    checks["scheduler_attacks"] = {
        "same_bank_issue": [r["issue_cycle"] for r in same_bank["scheduled_requests"]],
        "distinct_bank_issue": [r["issue_cycle"] for r in distinct_bank["scheduled_requests"]],
        "1rw_issue": [r["issue_cycle"] for r in one_rw["scheduled_requests"]],
        "1r1w_issue": [r["issue_cycle"] for r in one_r_one_w["scheduled_requests"]],
        "outstanding1_return_then_issue": [
            slot["scheduled_requests"][0]["return_cycle"],
            slot["scheduled_requests"][1]["issue_cycle"],
        ],
        "unresolved_dependency_rejected": dependency_rejected,
    }

    # An out-of-capacity address is accepted because scheduler resources carry
    # no range/stripe/eviction semantics.  This is an intentional attack, not a
    # proposed workload result.
    out_of_range = tx(
        "outside_psum", "psum_read", address=common.psum_bytes_logical + 48,
        width=48,
    )
    out_of_range_result = M768.AddressTimedScheduler(common).schedule(
        M768.expand_transactions([out_of_range])
    )
    checks["address_capacity_attack"] = {
        "address": common.psum_bytes_logical + 48,
        "logical_psum_bytes": common.psum_bytes_logical,
        "scheduler_accepted": out_of_range_result["expanded_request_count"] == 1,
        "range_or_stripe_gate_present": False,
    }

    with tempfile.TemporaryDirectory(prefix="m773_mapper_attack_") as tmp:
        package = Path(tmp) / "package"
        calls = package / "calls"
        calls.mkdir(parents=True)
        shape = [10, 1, 8, 1, 1]
        activation = np.zeros(shape, dtype=np.uint8)
        activation[0, 0, 0, 0, 0] = 1
        activation[0, 0, 1, 0, 0] = 1
        payload = calls / "x.bitpack"
        payload.write_bytes(np.packbits(activation.reshape(-1), bitorder="little").tobytes())
        record = {
            "population_id": "TINY_PRIMARY",
            "module_index": 0,
            "input_shape": shape,
            "relative_path": "calls/x.bitpack",
            "packed_sha256": sha256(payload),
        }
        mapper = M768.load_pinned_mapper(
            HW / contract["inputs"]["m672_mapper"]["path"],
            contract["inputs"]["m672_mapper"]["sha256"],
        )
        geometry = {0: (8, 4, 1, 1, 2, 2)}
        rows_by_config = {
            config: list(M768.iter_record_transactions(
                mapper, record, package, "TINY_PRIMARY", config, 0,
                tile_m=2, geometry=geometry,
            ))
            for config in M768.CONFIGS
        }
        projections = {
            config: semantic_projection(rows)
            for config, rows in rows_by_config.items()
        }
        cycles = {
            config: M768.AddressTimedScheduler(resource()).schedule(
                M768.expand_transactions(rows)
            )["total_cycles"]
            for config, rows in rows_by_config.items()
        }
        weight = next(
            row for row in rows_by_config["TYPED_SIGNED_K8"]
            if row.kind == "weight_read" and len(row.bank_pattern) >= 2
        )
        channels = 8
        observed_flat_k = tuple(offset // 16 for offset in weight.address_offsets[:2])
        expected_bank_local_offsets = tuple(
            (((flat_k // channels) * (channels // 8))
              + ((flat_k % channels) // 8)) * 16
            for flat_k in observed_flat_k
        )
        checks["configuration_semantics_attack"] = {
            "all_three_semantic_projections_identical": len({
                json.dumps(value, sort_keys=True) for value in projections.values()
            }) == 1,
            "total_cycles_by_config": cycles,
            "all_three_cycles_identical": len(set(cycles.values())) == 1,
            "generator_has_no_config_specific_work_model": True,
        }
        checks["weight_bank_local_address_attack"] = {
            "observed_flattened_k": list(observed_flat_k),
            "observed_first_two_offsets": list(weight.address_offsets[:2]),
            "expected_first_two_offsets_from_tap_then_source_channel_div8": list(expected_bank_local_offsets),
            "matches_frozen_address_order": tuple(weight.address_offsets[:2]) == expected_bank_local_offsets,
        }
        kinds = {row.kind for row in rows_by_config["TYPED_SIGNED_K8"]}
        checks["binary_input_traffic_attack"] = {
            "transaction_kinds": sorted(kinds),
            "has_external_input_or_weight_refill": "external_read" in kinds,
            "payload_or_descriptor_load_bytes_charged": False,
        }

    d3_psum_address_span = 240 * 320 * 1 * 384
    checks["d3_psum_residency_attack"] = {
        "logical_dense_address_span_bytes": d3_psum_address_span,
        "physical_psum_partition_bytes": common.psum_bytes_logical,
        "span_over_partition_ratio": d3_psum_address_span / common.psum_bytes_logical,
        "stripe_eviction_restore_transactions_implemented": False,
    }

    checks["claim_boundary"] = {
        "launch_now": contract["launch_now"],
        "production_speedup_allowed": contract["production_speedup_allowed"],
        "decoder_complete": contract["claim_boundary"]["decoder_complete"],
        "full_network_completion": contract["claim_boundary"]["full_network_completion"],
        "table_a_insertion_allowed": contract["claim_boundary"]["table_a_insertion_allowed"],
        "system_speedup": contract["claim_boundary"]["system_speedup"],
        "m700_candidate_input_absent": "m700" not in json.dumps(contract).lower(),
        "result_directory_absent": not any((HW / "results").glob("*m768*")),
    }

    findings = [
        {
            "id": "M773-P0-01",
            "severity": "P0",
            "title": "A1, equal-service K1x8 and typed K8 have no distinct executable semantics",
            "evidence": "Synthetic exact-mapper replay produces identical non-label transaction projections and identical cycles for all three configurations; iter_record_transactions has no configuration-specific work branch.",
            "impact": "A production replay could only measure a relabelled common path, so it cannot establish the authorized K8-vs-equal-service-K1x8 comparator.",
        },
        {
            "id": "M773-P0-02",
            "severity": "P0",
            "title": "The 245760-byte resource is declared but not enforced by address residency",
            "evidence": "A psum request beyond the 221184-byte psum partition is accepted. D3 dense destination span is 29491200 bytes (133.33x the psum partition), while no stripe, eviction, restore or external backing transaction exists.",
            "impact": "The current scheduler grants an unbounded logical psum store and omits the memory traffic whose timing M768 is intended to measure.",
        },
        {
            "id": "M773-P0-03",
            "severity": "P0",
            "title": "Frozen flattened-K bank-local address order is not implemented",
            "evidence": "For flattened K indices 24 and 25 (same tap, channels 0 and 1 on distinct banks), generated offsets are [384,400], while tap/source_channel_div_8 bank-local addressing requires [48,48].",
            "impact": "Address hashes and conflicts would describe a different physical weight layout from the common-resource contract.",
        },
        {
            "id": "M773-P1-01",
            "severity": "P1",
            "title": "Binary source/descriptor and weight-refill traffic is uncharged",
            "evidence": "The exact binary route emits weight_read, psum_read/write, compute and commit only; no external_read is emitted and no source SRAM is allocated. The 13824-byte weight partition has no refill behavior.",
            "impact": "memory_timing_included and DRAM/SRAM byte metrics cannot be asserted for a production component row.",
        },
        {
            "id": "M773-P1-02",
            "severity": "P1",
            "title": "Pinned contributor/storage oracles are identity-checked but never used as executable conservation oracles",
            "evidence": "M712 and M722R2 SHAs are validated, but the generator consumes only M672 tiles; source_flat_index is assigned and unused, and no contributor multiset or stripe/storage equality is checked.",
            "impact": "Exact mapper import does not prove contributor/address expansion or storage traffic conservation.",
        },
        {
            "id": "M773-P1-03",
            "severity": "P1",
            "title": "D1 common dense fallback has a placeholder compute charge",
            "evidence": "The D1 branch emits exactly one compute request after the input transfer regardless of layer geometry, then dense commits.",
            "impact": "D1 remains diagnostic as declared, but its cycle number would not be a defensible charged dense-FP32 fallback.",
        },
        {
            "id": "M773-P2-01",
            "severity": "P2",
            "title": "Cycle-class conservation is structural, not a bottleneck attribution proof",
            "evidence": "Any cycle with any issued request is active_service; no-issue cycles with any inflight request are classified dependency_completion. Per-port busy time is not reported.",
            "impact": "The required compute/weight-bank/psum-bank/memory/dependency breakdown may be misleading even though its sum equals total_cycles.",
        },
    ]

    output = {
        "schema": "m773_m768_decoder_address_timed_source_fresh_hammer_attack_v1",
        "date": "2026-08-28",
        "status": "FAIL_SOURCE_SEMANTICS__NO_PRODUCTION_REPLAY_AUTHORIZED",
        "score": 62,
        "severity_counts": {"p0": 3, "p1": 3, "p2": 1},
        "checks": checks,
        "findings": findings,
        "production_launch_authorized": False,
        "production_replay_performed": False,
        "production_cycles_generated": False,
        "production_speedup_generated": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

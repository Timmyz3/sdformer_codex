#!/usr/bin/env python3
"""Model C4 temporal residency that fuses Motion state and ATLIF stream commit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(tile_dir: Path) -> list[dict[str, str]]:
    path = tile_dir / "tile_records.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty tile records: {path}")
    return rows


def load_wall_model() -> Any:
    path = Path(__file__).with_name("analyze_m4_descriptor_resident_wall_cycles.py")
    spec = importlib.util.spec_from_file_location("m7_wall_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import M4 wall model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["row_id"], row["weight_group"],
    )


def geometry_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["sample_id"], row["sequence_key"], row["name"], row["operator"],
        row["operator_call_index"], row["weight_group"], row["source_width"],
        row["chunks_per_row"], row["output_channel_fanout"],
    )


def analyze_identity(label: str, tile_dir: Path, contexts: int, slots: int) -> dict[str, Any]:
    if contexts <= 0 or slots <= 0:
        raise ValueError("contexts and ATLIF slots must be positive")
    rows = read_rows(tile_dir)
    manifest_path = tile_dir / "manifest.json"
    packed_path = tile_dir / "packed_tiles.npz"
    records_path = tile_dir / "tile_records.csv"
    for path in (manifest_path, packed_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing tile identity: {path}")

    physical: dict[tuple[str, ...], dict[str, Any]] = {}
    by_geometry: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        key = physical_key(row)
        geometry = (
            int(row["source_width"]), int(row["chunks_per_row"]),
            int(row["output_channel_fanout"]),
        )
        entry = physical.setdefault(key, {"geometry": geometry, "steps": set(), "chunks": defaultdict(set)})
        if entry["geometry"] != geometry:
            raise ValueError(f"physical geometry changed: {key}")
        step = int(row["temporal_step"])
        entry["steps"].add(step)
        entry["chunks"][step].add(int(row["chunk_index"]))
        by_geometry[geometry_key(row)].add(key)

    temporal_steps = None
    for key, entry in physical.items():
        source_width, chunks, _fanout = entry["geometry"]
        del source_width
        steps = sorted(entry["steps"])
        if steps != list(range(len(steps))):
            raise ValueError(f"non-contiguous temporal state: {key}")
        if temporal_steps is None:
            temporal_steps = len(steps)
        elif temporal_steps != len(steps):
            raise ValueError("mixed temporal lengths require separate resident engines")
        for step in steps:
            if entry["chunks"][step] != set(range(chunks)):
                raise ValueError(f"incomplete row chunks: {key} step={step}")
    assert temporal_steps is not None

    groups: list[list[tuple[str, ...]]] = []
    for geometry in sorted(by_geometry):
        keys = sorted(by_geometry[geometry], key=lambda key: int(key[5]))
        groups.extend(keys[start:start + contexts] for start in range(0, len(keys), contexts))

    # Prove that temporal residency changes only batch order, not the C4 rows
    # sharing a compute batch.  This makes the existing temporal-fenced M4
    # wall-cycle evidence reusable without assuming an unverified repacking.
    resident_membership = Counter(
        (step, tuple(sorted(group)))
        for group in groups
        for step in range(temporal_steps)
    )
    wall = load_wall_model()
    ordered_bundles = wall.ordered_row_bundles(rows)
    fenced_groups: OrderedDict[tuple[str, ...], list[list[int]]] = OrderedDict()
    for bundle in ordered_bundles:
        fenced_groups.setdefault(
            wall.batch_key(rows[bundle[0]], "temporal_fenced"), []
        ).append(bundle)
    fenced_membership: Counter[tuple[int, tuple[tuple[str, ...], ...]]] = Counter()
    for bundles in fenced_groups.values():
        for start in range(0, len(bundles), contexts):
            batch = bundles[start:start + contexts]
            step = int(rows[batch[0][0]]["temporal_step"])
            fenced_membership[(step, tuple(sorted(physical_key(rows[item[0]]) for item in batch)))] += 1
    if resident_membership != fenced_membership:
        raise ValueError("temporal residency changed temporal-fenced C4 batch membership")
    membership_sha = hashlib.sha256(
        json.dumps(
            sorted((step, list(keys), count) for (step, keys), count in resident_membership.items()),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    resident_state = []
    global_by_sample: dict[str, dict[str, int]] = defaultdict(
        lambda: {"activation_bits": 0, "destination_bits": 0, "atlif_partial_bits": 0}
    )
    for key, entry in physical.items():
        source_width, _chunks, fanout = entry["geometry"]
        global_by_sample[key[0]]["activation_bits"] += source_width
        global_by_sample[key[0]]["destination_bits"] += fanout * 32
        global_by_sample[key[0]]["atlif_partial_bits"] += fanout * slots * 24
    for group in groups:
        activation = destination = atlif = 0
        for key in group:
            source_width, _chunks, fanout = physical[key]["geometry"]
            activation += source_width
            destination += fanout * 32
            atlif += fanout * slots * 24
        resident_state.append({
            "contexts": len(group),
            "activation_bits": activation,
            "destination_bits": destination,
            "atlif_partial_bits": atlif,
            "total_bits": activation + destination + atlif,
        })
    peak = max(resident_state, key=lambda item: item["total_bits"])
    global_peak = max(
        (values | {"total_bits": sum(values.values())} for values in global_by_sample.values()),
        key=lambda item: item["total_bits"],
    )
    batches = len(groups) * temporal_steps
    expected_batches = sum(
        math.ceil(len(keys) / contexts) * temporal_steps for keys in by_geometry.values()
    )
    if batches != expected_batches:
        raise ValueError("temporal resident batch accounting failed")
    return {
        "identity": label,
        "records": len(rows),
        "physical_rows": len(physical),
        "temporal_steps": temporal_steps,
        "spatial_context_groups": len(groups),
        "temporal_batches": batches,
        "partial_context_groups": sum(len(group) < contexts for group in groups),
        "batch_membership_equal_temporal_fenced": True,
        "batch_membership_sha256": membership_sha,
        "resident_peak": peak,
        "global_materialized_peak_per_sample": global_peak,
        "state_reduction_vs_global": global_peak["total_bits"] / peak["total_bits"],
        "schedule_contract": (
            "same C4 spatial identities remain resident while temporal_step sweeps 0..T-1; "
            "each individual compute batch contains one timestep only"
        ),
        "tile_records_sha256": sha256(records_path),
        "tile_manifest_sha256": sha256(manifest_path),
        "packed_tiles_sha256": sha256(packed_path),
    }


def packed_atlif_service(
    transactions: list[dict[str, str]], lanes: int, slots: int
) -> dict[str, Any]:
    """Exact per-invocation service for T10 plus cross-neuron-packed T2.

    A T2 invocation packs ``slots / 2`` independent neuron groups into the ten
    physical output slots.  This is the packing implemented by
    ``hitflow_dptme_array`` and checkpoint-replayed by the M7 L16 VCS miter.
    Keeping the rounding per invocation prevents unrelated layers from sharing
    otherwise unusable tail capacity.
    """
    if lanes <= 0 or slots <= 0:
        raise ValueError("ATLIF lanes and slots must be positive")
    service = 0
    equal_resource_service = 0
    dense_macs = 0
    live_invocations = 0
    by_temporal_steps: dict[int, dict[str, int]] = defaultdict(
        lambda: {"invocations": 0, "neurons": 0, "dense_macs": 0,
                 "packed_service_cycles": 0, "equal_resource_service_cycles": 0}
    )
    for row in transactions:
        if row.get("deployment_dead_result", "False").lower() == "true":
            continue
        temporal_steps = int(row["temporal_steps"])
        elements = int(row["elements_per_frame"])
        row_dense_macs = int(row["dense_macs_per_frame"])
        if temporal_steps <= 0 or slots % temporal_steps != 0 \
                or elements % temporal_steps != 0:
            raise ValueError(f"unsupported ATLIF invocation: {row.get('name', '')}")
        neurons = elements // temporal_steps
        if row_dense_macs != neurons * temporal_steps * temporal_steps:
            raise ValueError(f"ATLIF dense-MAC identity mismatch: {row.get('name', '')}")
        groups_per_command = slots // temporal_steps
        invocation_service = (
            math.ceil(neurons / (lanes * groups_per_command)) * temporal_steps
        )
        invocation_equal_resource = math.ceil(row_dense_macs / (lanes * slots))
        service += invocation_service
        equal_resource_service += invocation_equal_resource
        dense_macs += row_dense_macs
        live_invocations += 1
        bucket = by_temporal_steps[temporal_steps]
        bucket["invocations"] += 1
        bucket["neurons"] += neurons
        bucket["dense_macs"] += row_dense_macs
        bucket["packed_service_cycles"] += invocation_service
        bucket["equal_resource_service_cycles"] += invocation_equal_resource
    if live_invocations == 0:
        raise ValueError("ATLIF transaction ledger has no live invocation")
    return {
        "live_invocations": live_invocations,
        "physical_int8_multipliers": lanes * slots,
        "dense_macs": dense_macs,
        "packed_service_cycles": service,
        "equal_resource_service_cycles": equal_resource_service,
        "slot_packing_utilization": equal_resource_service / service,
        "by_temporal_steps": {
            str(key): value for key, value in sorted(by_temporal_steps.items())
        },
    }


def system_envelope(
    ledger: dict[str, Any], contract: dict[str, Any], m4: dict[str, Any],
    atlif_transactions: list[dict[str, str]], slots: int,
    stream_lanes: list[int],
) -> dict[str, Any]:
    if ledger.get("status") != "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE":
        raise ValueError("full-network ledger is not admitted")
    fixed = int(ledger["cycles_per_frame_model"]["fixed_total"])
    operator_cycles = int(ledger["cycles_per_frame_model"]["operator_activity_weighted"])
    atlif_cycles = int(ledger["cycles_per_frame_model"]["atlif_non_dead"])
    rqtb_attention = int(ledger["attention"]["rqtb_cycles_per_frame"])
    eligible = int(contract["coverage"]["eligible_cycles"])
    noneligible = operator_cycles - eligible
    categories = contract["coverage"]["categories"]
    qk_cycles = int(categories["attention_k_projection"]["eligible_cycles"]) + int(
        categories["attention_q_projection"]["eligible_cycles"]
    )
    m4_profiled_eligible = eligible - qk_cycles
    if m4_profiled_eligible <= 0:
        raise ValueError("M4-profiled eligible work is empty")
    local_item = m4["variants"]["local"]["per_identity"]["H67"]
    hybrid_item = m4["variants"]["hybrid"]["per_identity"]["H67"]
    local_speed = float(local_item["speedup_vs_p1_sparse_wall"])
    hybrid_speed_vs_local_p1 = local_speed * (
        int(local_item["m4_wall_cycles"]) / int(hybrid_item["m4_wall_cycles"])
    )
    variants = {}
    for line, speed in (("local", local_speed), ("hybrid", hybrid_speed_vs_local_p1)):
        # The admitted M4 tile population contains 31 non-Q/K operators.  Q/K
        # projections are therefore frozen at their ledger cost instead of
        # inheriting an unmeasured M4 or Motion speedup.
        optimized_operator = (
            noneligible + qk_cycles + math.ceil(m4_profiled_eligible / speed)
        )
        rows = []
        for lanes in sorted(set(stream_lanes)):
            service_item = packed_atlif_service(atlif_transactions, lanes, slots)
            service = int(service_item["packed_service_cycles"])
            equal_resource_service = int(service_item["equal_resource_service_cycles"])
            no_overlap = optimized_operator + service + rqtb_attention
            atlif_mac_matched_fixed_operator = (
                operator_cycles + equal_resource_service + rqtb_attention
            )
            rows.append({
                "stream_lanes": lanes,
                "physical_int8_multipliers": lanes * slots,
                "atlif_stream_service_cycles": service,
                "atlif_equal_resource_service_cycles": equal_resource_service,
                "atlif_ideal_compute_occupancy_no_bank_stalls": (
                    service_item["slot_packing_utilization"]
                ),
                "atlif_by_temporal_steps": service_item["by_temporal_steps"],
                "no_overlap_cycles": no_overlap,
                "speedup_vs_original_96mac_fixed": fixed / no_overlap,
                "atlif_mac_matched_fixed_operator_rqtb_cycles": (
                    atlif_mac_matched_fixed_operator
                ),
                "speedup_vs_atlif_mac_matched_fixed_operator_rqtb": (
                    atlif_mac_matched_fixed_operator / no_overlap
                ),
                "packing_compute_speedup_same_m4_rqtb": (
                    equal_resource_service / service
                ),
                "temporal_residency_cycle_gain_same_m4_rqtb": "UNMODELED",
            })
        variants[line] = {
            "effective_m4_speedup_vs_local_p1": speed,
            "optimized_operator_cycles": optimized_operator,
            "stream_points": rows,
        }
    return {
        "fixed_baseline_cycles": fixed,
        "operator_activity_cycles": operator_cycles,
        "eligible_cycles": eligible,
        "qk_projection_cycles_frozen_unprofiled": qk_cycles,
        "m4_profiled_eligible_cycles": m4_profiled_eligible,
        "noneligible_operator_cycles": noneligible,
        "standalone_atlif_cycles": atlif_cycles,
        "rqtb_attention_cycles": rqtb_attention,
        "variants": variants,
        "boundary": (
            "Amdahl/streaming envelope only. ATLIF-MAC-matched speedup compares only equal "
            "ATLIF INT8 multiplier count on an RQTB stack; it is not a whole-chip area/power "
            "match. Unprofiled Q/K projections are frozen. T2 service is rounded per invocation "
            "and uses checkpoint-proven five-group packing. Residency cycle gain remains "
            "unmodeled until address-timed SRAM/FIFO simulation; this is not FPS."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, required=True)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--atlif-slots", type=int, default=10)
    parser.add_argument("--atlif-stream-lanes", action="append", type=int, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--dual-line-contract", type=Path, required=True)
    parser.add_argument("--m4", type=Path, required=True)
    parser.add_argument("--atlif-transactions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities = [
        analyze_identity(label, Path(directory), args.contexts, args.atlif_slots)
        for label, directory in args.identity
    ]
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    contract = json.loads(args.dual_line_contract.read_text(encoding="utf-8"))
    m4 = json.loads(args.m4.read_text(encoding="utf-8"))
    atlif_transactions = list(
        csv.DictReader(args.atlif_transactions.open(encoding="utf-8"))
    )
    envelope = system_envelope(
        ledger, contract, m4, atlif_transactions, args.atlif_slots,
        args.atlif_stream_lanes
    )
    payload = {
        "schema": "m7_temporal_resident_fusion_envelope_v2",
        "status": "PASS_M7_TEMPORAL_RESIDENCY_SYSTEM_ENVELOPE_PRE_RTL",
        "claim_boundary": (
            "Exact resident-state sizing on admitted tile identities plus an Amdahl/ATLIF "
            "stream envelope. This does not yet prove fused RTL timing, memory ports, full-network "
            "cycle accuracy, energy, or accuracy."
        ),
        "architecture": {
            "contexts": args.contexts,
            "atlif_slots": args.atlif_slots,
            "innovation": (
                "sweep all temporal steps for the same C4 spatial rows, retaining previous "
                "activation, destination output, and ATLIF partials locally; Local replaces "
                "destination state while Motion applies a signed delta in place"
            ),
        },
        "identities": identities,
        "system_envelope": envelope,
        "sources": {
            "ledger_sha256": sha256(args.ledger),
            "dual_line_contract_sha256": sha256(args.dual_line_contract),
            "m4_sha256": sha256(args.m4),
            "atlif_transactions_sha256": sha256(args.atlif_transactions),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "temporal_resident_fusion.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = [
        "# M7 temporal-resident Local/Motion + ATLIF fusion envelope\n\n",
        "| identity | temporal batches | resident peak | global peak | state reduction |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for item in identities:
        report.append(
            f"| {item['identity']} | {item['temporal_batches']} | "
            f"{item['resident_peak']['total_bits']} bit | "
            f"{item['global_materialized_peak_per_sample']['total_bits']} bit | "
            f"{item['state_reduction_vs_global']:.3f}x |\n"
        )
    report.extend([
        "\n| line | ATLIF lanes | multipliers | ideal compute occ. | vs original 96-MAC | ATLIF-MAC-matched |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    for line, variant in envelope["variants"].items():
        for item in variant["stream_points"]:
            report.append(
                f"| {line} | {item['stream_lanes']} | "
                f"{item['physical_int8_multipliers']} | "
                f"{item['atlif_ideal_compute_occupancy_no_bank_stalls']:.6f} | "
                f"{item['speedup_vs_original_96mac_fixed']:.6f}x | "
                f"{item['speedup_vs_atlif_mac_matched_fixed_operator_rqtb']:.6f}x |\n"
            )
    report.append(
        "\nThese are bounded no-overlap system envelopes, not measured full-network speedups. "
        "The ATLIF-MAC-matched column matches only ATLIF multiplier count, not whole-chip "
        "resources. Packing has no modeled compute-cycle gain at these divisible shapes; "
        "residency benefit still requires address-timed memory simulation.\n"
    )
    (args.output / "REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

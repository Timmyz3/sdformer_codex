#!/usr/bin/env python3
"""Screen a streamed coefficient engine for H67 current-batch FFN BN.

This milestone deliberately separates coefficient service from the much larger
moment barrier and replay problem.  It proves only exact geometry and a
cycle-defined producer/consumer recurrence.  Reciprocal-square-root accuracy,
fixed-point formats and physical PPA remain outside the admission boundary.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PATHS = {
    "m159": HW / "results/m159_h67_full_ffn_subgraph_scope_r1_20260824/m159_h67_full_ffn_subgraph_scope.json",
    "m161": HW / "results/m161_h67_dynamic_bn_rank3_fusion_dse_r1_20260824/m161_h67_dynamic_bn_rank3_fusion_dse.json",
    "m161_csv": HW / "results/m161_h67_dynamic_bn_rank3_fusion_dse_r1_20260824/per_ffn_dynamic_bn_rank3_dse.csv",
    "m167_review": HW / "results/m167_independent_hammer_review_r1_20260824/m167_independent_hammer_review.json",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "m159": "6c67a75d052080cf58e558f960f23bea64d841087967de044fef898ad46c7f89",
    "m161": "07f731ddd55ad879bda6b01df47187c75f06674d66b312b2479aa659200b0aaa",
    "m161_csv": "6d2cc9252c21893438328d9a1aa20f77ed5f7d4bcac57cc10b9ee2d9c17600de",
    "m167_review": "d98c184b289b66107194e6464ea9ac1a11661b016871d919a273279faa65b983",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

T = 10
BN1_TILE_CHANNELS = 16
BN2_TILE_CHANNELS = 96
COEFFICIENT_BITS = 48  # Q24 alpha + Q24 offset candidate, not numeric admission.
SERIAL_COEFFICIENT_II = 16
ACCOUNTED_FFN_CYCLES = 205_384_111
GLOBAL_ENVELOPE_CYCLES = 620_302_905


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)),
        )


def overlap_recurrence(tile_count, producer_cycles, consumer_cycles):
    """Cycles and exposed bubbles for ping-pong producer/consumer tiles."""
    require(tile_count > 0 and producer_cycles > 0 and consumer_cycles > 0,
            "invalid recurrence input")
    exposed = producer_cycles
    if producer_cycles > consumer_cycles:
        exposed += (tile_count - 1) * (producer_cycles - consumer_cycles)
    total = tile_count * consumer_cycles + exposed
    return total, exposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M232 output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M232 frozen input identity drift")

    m159 = strict_json(PATHS["m159"])
    m161 = strict_json(PATHS["m161"])
    m167 = strict_json(PATHS["m167_review"])
    require(m159["accounted_compute_cycles_per_frame"][
        "full_ffn_subgraph_excluding_bn_residual"] == ACCOUNTED_FFN_CYCLES,
        "M159 FFN cycle identity drift")
    require(m161["frozen_semantics"]["bn_policy"] ==
            "no_running/current-batch", "M161 BN policy drift")
    require(m161["frozen_semantics"]["global_moment_barrier_per_module"],
            "M161 barrier semantics drift")
    require(m167["score"] == 81,
            "M167 independent review identity drift")

    module_rows = []
    with PATHS["m161_csv"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hidden = int(row["expanded_channels"])
            positions = int(row["positions_per_channel"])
            require(hidden % 4 == 0 and hidden % BN1_TILE_CHANNELS == 0,
                    "unexpected FFN expansion geometry")
            output_channels = hidden // 4
            require(output_channels % BN2_TILE_CHANNELS == 0,
                    "unexpected FFN output geometry")
            stage = int(row["stage"])

            bn1_tiles = hidden // BN1_TILE_CHANNELS
            bn2_tiles = output_channels // BN2_TILE_CHANNELS
            coefficient_ii = SERIAL_COEFFICIENT_II
            bn1_producer = BN1_TILE_CHANNELS * coefficient_ii
            bn2_producer = BN2_TILE_CHANNELS * coefficient_ii
            # M167 BACK reconstructs two time rows for 16 channels per issue.
            bn1_consumer = positions * (T // 2)
            # The M159 96-lane row boundary consumes one full output-channel
            # tile per cycle across each temporal/spatial position.
            bn2_consumer = positions * T
            bn1_total, bn1_exposed = overlap_recurrence(
                bn1_tiles, bn1_producer, bn1_consumer)
            bn2_total, bn2_exposed = overlap_recurrence(
                bn2_tiles, bn2_producer, bn2_consumer)

            require(bn1_producer <= bn1_consumer,
                    "II16 BN1 producer fails to hide under replay")
            require(bn2_producer <= bn2_consumer,
                    "II16 BN2 producer fails to hide under replay")
            require(bn1_exposed == bn1_producer and
                    bn2_exposed == bn2_producer,
                    "unexpected overlap bubble arithmetic")
            module_rows.append({
                "module": row["module"],
                "stage": stage,
                "positions_per_channel": positions,
                "bn1_channels": hidden,
                "bn2_channels": output_channels,
                "bn1_tiles": bn1_tiles,
                "bn2_tiles": bn2_tiles,
                "coefficient_output_interval_cycles": coefficient_ii,
                "bn1_coefficient_tile_fill_cycles": bn1_producer,
                "bn1_replay_cycles_per_tile": bn1_consumer,
                "bn1_coefficient_to_replay_rate_margin": (
                    float(bn1_consumer) / bn1_producer),
                "bn1_ideal_replay_cycles": bn1_tiles * bn1_consumer,
                "bn1_total_cycles_with_pingpong": bn1_total,
                "bn1_exposed_coefficient_cycles": bn1_exposed,
                "bn2_coefficient_tile_fill_cycles": bn2_producer,
                "bn2_replay_cycles_per_tile": bn2_consumer,
                "bn2_coefficient_to_replay_rate_margin": (
                    float(bn2_consumer) / bn2_producer),
                "bn2_ideal_replay_cycles": bn2_tiles * bn2_consumer,
                "bn2_total_cycles_with_pingpong": bn2_total,
                "bn2_exposed_coefficient_cycles": bn2_exposed,
            })

    require(len(module_rows) == 12, "FFN module count drift")
    total_bn1_channels = sum(row["bn1_channels"] for row in module_rows)
    total_bn2_channels = sum(row["bn2_channels"] for row in module_rows)
    require(total_bn1_channels == 17_664, "BN1 coefficient count drift")
    require(total_bn2_channels == 4_416, "BN2 coefficient count drift")
    total_coefficients = total_bn1_channels + total_bn2_channels
    require(total_coefficients == 22_080, "total coefficient count drift")

    no_overlap_cycles = total_coefficients * SERIAL_COEFFICIENT_II
    exposed_cycles = sum(
        row["bn1_exposed_coefficient_cycles"] +
        row["bn2_exposed_coefficient_cycles"] for row in module_rows)
    require(no_overlap_cycles == 353_280, "serial coefficient total drift")
    require(exposed_cycles == 21_504, "overlap exposure drift")

    stage_rows = []
    for stage in range(4):
        selected = [row for row in module_rows if row["stage"] == stage]
        stage_rows.append({
            "stage": stage,
            "blocks": len(selected),
            "bn1_channels": sum(row["bn1_channels"] for row in selected),
            "bn2_channels": sum(row["bn2_channels"] for row in selected),
            "no_overlap_serial_coefficient_cycles": sum(
                (row["bn1_channels"] + row["bn2_channels"])
                * SERIAL_COEFFICIENT_II for row in selected),
            "pingpong_exposed_coefficient_cycles": sum(
                row["bn1_exposed_coefficient_cycles"] +
                row["bn2_exposed_coefficient_cycles"] for row in selected),
            "minimum_bn1_rate_margin": min(
                row["bn1_coefficient_to_replay_rate_margin"]
                for row in selected),
            "minimum_bn2_rate_margin": min(
                row["bn2_coefficient_to_replay_rate_margin"]
                for row in selected),
        })

    ii_sensitivity = []
    for coefficient_ii in (1, 4, 8, 16, 24, 31, 32, 64):
        exposed = 0
        all_hidden = True
        for row in module_rows:
            for tile_count, tile_channels, consumer in (
                    (row["bn1_tiles"], BN1_TILE_CHANNELS,
                     row["bn1_replay_cycles_per_tile"]),
                    (row["bn2_tiles"], BN2_TILE_CHANNELS,
                     row["bn2_replay_cycles_per_tile"])):
                producer = tile_channels * coefficient_ii
                _, bubble = overlap_recurrence(
                    tile_count, producer, consumer)
                exposed += bubble
                all_hidden = all_hidden and producer <= consumer
        ii_sensitivity.append({
            "coefficient_output_interval_cycles": coefficient_ii,
            "all_24_bn_phases_rate_matched_after_first_tile": all_hidden,
            "exposed_coefficient_cycles_per_frame": exposed,
            "share_of_accounted_ffn_cycles": exposed / ACCOUNTED_FFN_CYCLES,
            "share_of_global_envelope_cycles": exposed / GLOBAL_ENVELOPE_CYCLES,
        })
    require(next(row for row in ii_sensitivity if
                 row["coefficient_output_interval_cycles"] == 31)[
                     "all_24_bn_phases_rate_matched_after_first_tile"],
            "II31 should fit the stage-3 BN2 boundary")
    require(not next(row for row in ii_sensitivity if
                     row["coefficient_output_interval_cycles"] == 32)[
                         "all_24_bn_phases_rate_matched_after_first_tile"],
            "II32 should miss the stage-3 BN2 boundary")

    full_coefficient_storage_bits = max(
        (row["bn1_channels"] + row["bn2_channels"]) * COEFFICIENT_BITS
        for row in module_rows)
    pingpong_storage_bits = 2 * BN2_TILE_CHANNELS * COEFFICIENT_BITS
    require(full_coefficient_storage_bits == 184_320,
            "largest block coefficient storage drift")
    require(pingpong_storage_bits == 9_216,
            "pingpong coefficient storage drift")

    payload = {
        "schema": "m232_dynamic_bn_coefficient_stream_screen_v1",
        "status": "PASS_CYCLE_DEFINED_COEFFICIENT_STREAM_RATE_SCREEN",
        "scope": "H67/Motion 12 FFN BN1+BN2 current-batch coefficient finalization and channel-major replay boundary",
        "frozen_semantics": {
            "bn_policy": "no_running/current-batch",
            "ffn_blocks": 12,
            "bn_phases": 24,
            "bn1_coefficients_per_frame": total_bn1_channels,
            "bn2_coefficients_per_frame": total_bn2_channels,
            "total_coefficients_per_frame": total_coefficients,
            "global_per_module_moment_barrier_retained": True,
        },
        "candidate": {
            "name": "barrier-drained coefficient streaming",
            "producer": "one scalar reciprocal-sqrt/affine coefficient stream with aggregate output interval 16 cycles",
            "consumer": "BN1 16-channel M167 BACK tiles and BN2 96-channel normalize/residual tiles",
            "storage": "two 96-channel coefficient tile banks shared across mutually exclusive BN1/BN2 phases",
            "schedule": "after each module moment barrier, fill tile 0; consume tile N while filling tile N+1 in channel-major replay order",
            "m167_reuse": "BN1 alpha-times-left and offset prefold products use the existing barrier-phase PREFOLD owner of the shared96 main pool; no second wide multiplier pool is admitted",
        },
        "cycle_screen": {
            "serial_coefficient_output_interval_cycles": SERIAL_COEFFICIENT_II,
            "serial_no_overlap_cycles_per_frame": no_overlap_cycles,
            "pingpong_exposed_first_tile_cycles_per_frame": exposed_cycles,
            "overlap_reduction": no_overlap_cycles / exposed_cycles,
            "exposed_share_of_accounted_ffn_cycles": exposed_cycles / ACCOUNTED_FFN_CYCLES,
            "exposed_share_of_global_envelope_cycles": exposed_cycles / GLOBAL_ENVELOPE_CYCLES,
            "minimum_bn1_producer_to_consumer_rate_margin": min(
                row["bn1_coefficient_to_replay_rate_margin"]
                for row in module_rows),
            "minimum_bn2_producer_to_consumer_rate_margin": min(
                row["bn2_coefficient_to_replay_rate_margin"]
                for row in module_rows),
            "maximum_rate_matched_output_interval_cycles": 31,
            "qualification": "Cycle-defined tile recurrence only. It assumes channel-major state replay and excludes moment accumulation, SRAM port stalls, coefficient arithmetic latency before the stated output interval, numeric accuracy and physical energy.",
        },
        "coefficient_storage_screen": {
            "candidate_format": "Q24 alpha plus Q24 offset per channel",
            "format_accuracy_proved": False,
            "largest_block_full_materialization_bits": full_coefficient_storage_bits,
            "shared_two_by_96_tile_bits": pingpong_storage_bits,
            "local_storage_reduction": full_coefficient_storage_bits / pingpong_storage_bits,
            "qualification": "Register/SRAM payload bits only; excludes ECC, tags, banking and control.",
        },
        "stage_rows": stage_rows,
        "ii_sensitivity": ii_sensitivity,
        "algorithm_feedback": {
            "required_capture": "export per-module current-batch mean/variance/epsilon/gamma/beta and pre/post-BN activation ranges from the frozen no-running graph on the A800 server",
            "required_training": "if Q24/Q8 fixed-point replay misses valid825, train with the exact streamed coefficient and channel-major hardware order",
            "do_not_spend_accuracy_budget_on": "approximating away dynamic BN before the deployment graph is trained to running/folded BN; M193 rejected the present frozen-stat recalibration",
        },
        "admission": {
            "exact_topology_and_coefficient_count": True,
            "cycle_defined_pingpong_rate_screen": True,
            "single_ii16_stream_rate_matched": True,
            "coefficient_numeric_rtl": False,
            "reciprocal_sqrt_accuracy": False,
            "fixed_point_checkpoint_equivalence": False,
            "moment_state_sram_and_address_schedule": False,
            "full_bn_cycles": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement": "For the 12 H67 FFNs, current-batch BN produces 22,080 channel coefficients per frame. Under a channel-major two-tile recurrence, a scalar coefficient stream with a 16-cycle output interval is rate-hidden after the first tile for every BN1/BN2 stage; only 21,504 first-tile cycles remain exposed, while candidate coefficient storage falls from 184,320 to 9,216 bits for the largest block. This is a cycle-defined service-boundary result, not fixed-point BN accuracy, physical speedup or system speedup.",
        "identity": {
            "inputs_sha256": observed,
            "analyzer_start_sha256": script_start,
            "docs359_sha256_unchanged": EXPECTED["docs359"],
        },
    }

    output.mkdir(parents=True)
    csv_path = output / "per_ffn_coefficient_stream.csv"
    fields = list(module_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(module_rows)
    result_path = output / "m232_dynamic_bn_coefficient_stream_screen_r1.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    readme = output / "README.md"
    readme.write_text(
        "# M232 dynamic-BN coefficient stream screen\n\n"
        "The frozen no-running FFN path retains 24 current-batch BN phases and "
        "22,080 per-channel coefficients per frame. A single coefficient stream "
        "at output interval 16 cycles can feed double-buffered 16-channel BN1 "
        "tiles and 96-channel BN2 tiles without a post-first-tile rate stall in "
        "any stage. The slowest boundary is stage-3 BN2: 1,536 coefficient-fill "
        "cycles versus 3,000 replay cycles.\n\n"
        "Without overlap, coefficient service is 353,280 cycles/frame. The exact "
        "ping-pong recurrence exposes only the first tile of each of 24 phases, "
        "21,504 cycles/frame (16.428571x lower), equal to 0.010470% of the "
        "205.384M accounted FFN cycles and 0.003467% of the existing global "
        "envelope. II31 still rate-matches every phase; II32 first misses stage-3 "
        "BN2. Therefore the performance-critical work is the moment barrier and "
        "state replay, not a large rsqrt farm.\n\n"
        "Using an illustrative Q24 alpha plus Q24 offset payload, the largest "
        "block would materialize 184,320 coefficient bits. Two shared 96-channel "
        "tile banks hold 9,216 bits, a 20x local payload reduction. BN1 prefold "
        "can use M167's mutually exclusive PREFOLD phase instead of adding a "
        "second wide multiplier pool.\n\n"
        "This milestone does not prove reciprocal-sqrt error, fixed-point "
        "equivalence, moment SRAM ports, complete BN cycles, PPA, energy or "
        "system speedup. The next numeric gate is an A800 capture of real "
        "mean/variance/gamma/beta and activation ranges.\n",
        encoding="utf-8")
    require(sha256(script_path) == script_start, "analyzer changed during run")
    manifest = output / "manifest.sha256"
    entries = [readme, result_path, csv_path]
    manifest.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in entries),
        encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "coefficients": total_coefficients,
        "no_overlap_cycles": no_overlap_cycles,
        "exposed_cycles": exposed_cycles,
        "max_rate_matched_ii": 31,
        "minimum_bn2_margin": payload["cycle_screen"][
            "minimum_bn2_producer_to_consumer_rate_margin"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

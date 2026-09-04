#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""M1554: destination-owned ep34 decoder screen for S2 CCBS.

M1547 incorrectly reset the epsilon debt at each source site.  This successor
selects output destinations and accumulates all legal K3/S2 spatial sources
and taps before making a source-group drop decision.  It remains a CPU-only
diagnostic: epsilon is not AEE, and no cycle, traffic, energy or RTL claim is
admitted.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
BASE_SOURCE = HW / "system_simulator/scripts/analyze_m1547_ep34_s2_ccbs_decoder_retained_fastkill.py"
BASE_SOURCE_SHA256 = "facf1831a29ee9b4db86b4899e82cf248e5fc9e1134c7c36eac89319fd9419d8"
CONTRACT = HW / "contracts/m1554_ep34_s2_ccbs_destination_debt_successor_contract_r1_20260831.json"
DEFAULT_OUTPUT = HW / "results/m1554_ep34_s2_ccbs_destination_debt_successor_r1_20260831"

GEOMETRY = (
    (1536, 384, 15, 20, 30, 40),
    (770, 192, 30, 40, 60, 80),
    (386, 96, 60, 80, 120, 160),
    (194, 96, 120, 160, 240, 320),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


require(sha256(BASE_SOURCE) == BASE_SOURCE_SHA256, "M1547 base source SHA drift")
SPEC = importlib.util.spec_from_file_location("m1554_bound_m1547", str(BASE_SOURCE))
require(SPEC is not None and SPEC.loader is not None, "cannot import M1547 base")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def destination_sources(output_y, output_x, height, width):
    """Exact input coordinates/taps for ConvTranspose K3/S2/P1/OP1."""
    result = []
    for ky in range(3):
        numerator_y = int(output_y) + 1 - ky
        if numerator_y % 2:
            continue
        input_y = numerator_y // 2
        if not 0 <= input_y < int(height):
            continue
        for kx in range(3):
            numerator_x = int(output_x) + 1 - kx
            if numerator_x % 2:
                continue
            input_x = numerator_x // 2
            if 0 <= input_x < int(width):
                result.append((input_y, input_x, ky, kx))
    require(1 <= len(result) <= 4, "destination contributor cardinality drift")
    require(len(set(result)) == len(result), "duplicate destination contributor")
    return tuple(result)


def unpack_selected_destinations(path, record, destinations_per_call):
    import numpy as np
    require(sha256(path) == record["positive_output_sha256"], "payload SHA drift")
    raw = np.fromfile(str(path), dtype=np.uint8)
    require(int(raw.size) == int(record["plane_bytes"]), "payload extent drift")
    lookup = np.array([[((value >> bit) & 1) for bit in range(8)]
                       for value in range(256)], dtype=np.uint8)
    bits = lookup[raw].reshape(-1)[:int(record["elements"])]
    shape = tuple(int(value) for value in record["shape"])
    module = int(record["module_ordinal"])
    cin, _cout, hin, win, hout, wout = GEOMETRY[module]
    require(shape == (10, 1, cin, hin, win), "payload geometry drift")
    plane = bits.reshape(shape)
    total = 10 * hout * wout
    indices = np.linspace(0, total - 1, destinations_per_call, dtype=np.int64)
    observations = []
    for flat in indices.tolist():
        timestep = int(flat) // (hout * wout)
        spatial = int(flat) % (hout * wout)
        output_y = spatial // wout
        output_x = spatial % wout
        sources = destination_sources(output_y, output_x, hin, win)
        activity = np.stack([plane[timestep, 0, :, iy, ix]
                             for iy, ix, _ky, _kx in sources], axis=0)
        require(activity.shape == (len(sources), cin), "activity shape drift")
        require(set(np.unique(activity).tolist()).issubset(set([0, 1])),
                "selected activity is not binary")
        observations.append({"timestep": timestep, "output_y": output_y,
            "output_x": output_x, "sources": sources, "activity": activity})
    require(len(observations) == destinations_per_call,
            "selected destination population drift")
    return observations


def analyze_config(weights, observations, group_size, output_tile):
    import numpy as np
    aggregate_bounds = []
    exact_ratios = []
    false_zero_collisions = 0
    exact_positive_bounds = 0
    epsilon_totals = dict((str(value), {"blocks": 0, "dropped": 0,
                                        "normalized_debt": []})
                          for value in B.EPSILON_GRID)
    witness_states = {}
    layer_rows = []
    aggregate_metadata = {"metadata_bytes": 0, "int8_weight_bytes": 0,
                          "old_g11_bytes": 0, "metadata_reads": 0}
    contributor_histogram = {}

    for layer_ordinal in range(4):
        weight = weights[layer_ordinal]
        layer_observations = observations[layer_ordinal]
        cin, cout, kh, kw = weight.shape
        require((cin, cout) == GEOMETRY[layer_ordinal][:2] and kh == 3 and kw == 3,
                "weight geometry drift")
        account = B.metadata_account(cin, cout, kh * kw,
                                     group_size, output_tile)
        aggregate_metadata["metadata_bytes"] += account["metadata_bytes"]
        aggregate_metadata["int8_weight_bytes"] += cin * cout * kh * kw
        aggregate_metadata["old_g11_bytes"] += account["old_g11_per_source_metadata_bytes"]
        g_blocks = account["g_blocks"]
        o_blocks = account["o_blocks"]
        aggregate_metadata["metadata_reads"] += (
            g_blocks * o_blocks * len(layer_observations))
        max_weight = float(np.abs(weight).max())
        require(max_weight > 0.0, "zero layer max weight")
        metadata = np.zeros((g_blocks, o_blocks), dtype=np.float64)
        for gb in range(g_blocks):
            gs = gb * group_size
            ge = min(cin, gs + group_size)
            for ob in range(o_blocks):
                os_ = ob * output_tile
                oe = min(cout, os_ + output_tile)
                metadata[gb, ob] = float(
                    np.abs(weight[gs:ge, os_:oe, :, :]).max())

        bounds_by_observation = []
        for observation in layer_observations:
            source_count = len(observation["sources"])
            contributor_histogram[str(source_count)] = (
                contributor_histogram.get(str(source_count), 0) + 1)
            activity = observation["activity"]
            padded = np.zeros((source_count, g_blocks * group_size),
                              dtype=np.uint8)
            padded[:, :cin] = activity
            counts = padded.reshape(source_count, g_blocks,
                                    group_size).sum(axis=(0, 2))
            bounds = counts[:, None].astype(np.float64) * metadata
            bounds_by_observation.append(bounds)
            aggregate_bounds.append(
                (bounds / (max_weight * float(group_size * source_count))).reshape(-1))

        for observation_ordinal, bounds in enumerate(bounds_by_observation):
            source_count = len(layer_observations[observation_ordinal]["sources"])
            reference = max_weight * float(cin * source_count)
            for epsilon in B.EPSILON_GRID:
                totals = epsilon_totals[str(epsilon)]
                for ob in range(o_blocks):
                    mask, debt = B.fixed_order_drop(
                        bounds[:, ob].tolist(), epsilon, reference)
                    totals["blocks"] += len(mask)
                    totals["dropped"] += sum(1 for value in mask if value)
                    totals["normalized_debt"].append(debt / reference)
                    if epsilon == B.DYNAMIC_WITNESS_EPSILON:
                        for gb, dropped in enumerate(mask):
                            key = (layer_ordinal, gb, ob)
                            witness_states[key] = witness_states.get(key, 0) | (
                                1 if dropped else 2)

        exact_observations = B.sampled_indices(len(layer_observations), 18)
        exact_g_blocks = B.sampled_indices(g_blocks, B.EXACT_G_BLOCKS_PER_LAYER)
        exact_o_blocks = B.sampled_indices(o_blocks, B.EXACT_O_BLOCKS_PER_LAYER)
        for observation_ordinal in exact_observations:
            observation = layer_observations[observation_ordinal]
            bounds = bounds_by_observation[observation_ordinal]
            for gb in exact_g_blocks:
                gs = gb * group_size
                ge = min(cin, gs + group_size)
                for ob in exact_o_blocks:
                    bound = float(bounds[gb, ob])
                    if bound <= 0.0:
                        continue
                    os_ = ob * output_tile
                    oe = min(cout, os_ + output_tile)
                    contribution = np.zeros((oe - os_,), dtype=np.float64)
                    for source_ordinal, (_iy, _ix, ky, kx) in enumerate(
                            observation["sources"]):
                        active = observation["activity"][source_ordinal,
                                                          gs:ge].astype(bool)
                        if bool(active.any()):
                            contribution += weight[gs:ge, os_:oe, ky, kx][active, :].sum(axis=0)
                    exact_value = float(np.abs(contribution).max())
                    exact_positive_bounds += 1
                    if exact_value <= 1.0e-12:
                        false_zero_collisions += 1
                    else:
                        exact_ratios.append(bound / exact_value)

        layer_rows.append({"module_ordinal": layer_ordinal, "cin": cin,
            "cout": cout, "selected_destinations": len(layer_observations),
            "metadata": account, "layer_max_abs_fp32_weight": max_weight,
            "exact_ratio_sample_policy": {
                "destinations": len(exact_observations),
                "g_blocks": len(exact_g_blocks),
                "o_blocks": len(exact_o_blocks)}})

    bound_stats = B.quantiles(np.concatenate(aggregate_bounds))
    ratio_stats = B.quantiles(exact_ratios)
    false_zero_fraction = float(false_zero_collisions) / float(exact_positive_bounds)
    metadata_ratio = (float(aggregate_metadata["metadata_bytes"]) /
                      float(aggregate_metadata["int8_weight_bytes"]))
    metadata_reduction = (float(aggregate_metadata["old_g11_bytes"]) /
                          float(aggregate_metadata["metadata_bytes"]))
    epsilon_rows = []
    for epsilon in B.EPSILON_GRID:
        totals = epsilon_totals[str(epsilon)]
        epsilon_rows.append({"epsilon_normalized": epsilon,
            "block_decisions": totals["blocks"],
            "drop_fraction": float(totals["dropped"]) / float(totals["blocks"]),
            "normalized_destination_debt": B.quantiles(
                totals["normalized_debt"]), "is_aee_budget": False})
    dynamic_count = B.dynamic_witness_count(witness_states)
    gates = {
        "bound_median_ratio_le_4": ratio_stats["median"] <= B.BOUND_MEDIAN_RATIO_MAX,
        "bound_p90_ratio_le_12": ratio_stats["p90"] <= B.BOUND_P90_RATIO_MAX,
        "false_zero_collision_le_1pct": false_zero_fraction <= B.BOUND_FALSE_ZERO_COLLISION_MAX,
        "metadata_le_2pct_int8_weight_bytes": metadata_ratio <= B.METADATA_WEIGHT_RATIO_MAX,
        "metadata_reduction_vs_old_g11_ge_8x": metadata_reduction >= B.METADATA_REDUCTION_MIN,
        "dynamic_same_block_keep_drop_witness": dynamic_count > 0,
        "zero_epsilon_only_zero_bound": epsilon_rows[0]["normalized_destination_debt"]["max"] == 0.0,
    }
    return {"block": {"source_group": group_size, "output_tile": output_tile},
        "debt_owner": "destination_x_output_tile",
        "spatial_contributor_histogram": contributor_histogram,
        "weight_domain": "exact ep34 FP32 local bound; no INT8 numeric authority",
        "metadata_charge_domain": "uint16 M directory versus hypothetical packed INT8 weight bytes",
        "metadata_aggregate": {"metadata_bytes": aggregate_metadata["metadata_bytes"],
            "int8_weight_bytes": aggregate_metadata["int8_weight_bytes"],
            "metadata_to_int8_weight_bytes": metadata_ratio,
            "old_g11_per_source_metadata_bytes": aggregate_metadata["old_g11_bytes"],
            "reduction_vs_old_g11": metadata_reduction,
            "metadata_reads_over_selected_destinations": aggregate_metadata["metadata_reads"]},
        "bound_normalized_to_layer_max_group_and_spatial_capacity": bound_stats,
        "bound_to_exact_destination_contribution_ratio_sample": ratio_stats,
        "positive_bound_exact_zero_collision_fraction": false_zero_fraction,
        "dynamic_witness": {"epsilon_normalized": B.DYNAMIC_WITNESS_EPSILON,
            "static_blocks_observed": len(witness_states),
            "blocks_with_both_keep_and_drop": dynamic_count,
            "fraction": float(dynamic_count) / float(len(witness_states))},
        "epsilon_diagnostics": epsilon_rows, "layers": layer_rows,
        "gates": gates, "passes_destination_screen": all(gates.values())}


def write_sealed_result(output, result):
    require(not output.exists(), "output already exists")
    output.mkdir(parents=True)
    result_path = output / "m1554_ep34_s2_ccbs_destination_debt_successor_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    report_path = output / "m1554_REPORT.md"
    report_path.write_text(
        "# M1554 destination-owned S2 CCBS re-screen\n\n"
        "Status: **{}**. Every decision accumulates all legal K3/S2 spatial "
        "contributors at one destination/output tile. This is a CPU-only local "
        "screen: epsilon is not AEE and no cycles, traffic, energy or RTL are "
        "admitted.\n".format(result["status"]))
    complete = output / "RUN_COMPLETE.txt"
    complete.write_text(result["status"] + "\n")
    members = [result_path, report_path, complete]
    sums = output / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                             for path in members))
    (output / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)))


def run(output):
    import numpy as np
    manifest = B.verify_inputs()
    contract = B.strict_json(CONTRACT)
    require(contract["block_configs"] == [list(row) for row in B.BLOCK_CONFIGS] and
            contract["epsilon_grid"] == list(B.EPSILON_GRID) and
            contract["required_accounting"]["owner"] == "destination_x_output_tile",
            "M1554 contract drift")
    selected, selected_samples = B.select_records(manifest)
    observations = dict((ordinal, []) for ordinal in range(4))
    selected_calls = []
    for record in selected:
        rows = unpack_selected_destinations(
            B.M1521_ROOT / record["positive_output"], record, B.SITES_PER_CALL)
        observations[int(record["module_ordinal"])].extend(rows)
        selected_calls.append({"global_call_ordinal": record["global_call_ordinal"],
            "sequence": record["sequence"],
            "replay_sample_ordinal": record["replay_sample_ordinal"],
            "module_ordinal": record["module_ordinal"],
            "positive_output_sha256": record["positive_output_sha256"]})
    require(all(len(rows) == 9 * B.SITES_PER_CALL
                for rows in observations.values()), "destination population drift")
    weights = B.load_weights()
    block_results = [analyze_config(weights, observations, group, output_tile)
                     for group, output_tile in B.BLOCK_CONFIGS]
    passing = [row for row in block_results if row["passes_destination_screen"]]
    status = ("PASS_DESTINATION_DEBT_SCREEN__INCREMENTAL_FC_PATCH_CAPTURE_REQUEST_ONLY__NO_AEE_PERFORMANCE_OR_RTL"
              if passing else
              "NO_GO_S2_DESTINATION_DEBT_SCREEN__NO_CAPTURE_AEE_PERFORMANCE_OR_RTL")
    result = {"schema": "m1554_ep34_s2_ccbs_destination_debt_successor_r1_v1",
        "status": status,
        "identity": {"checkpoint_sha256": B.EXPECTED[B.CHECKPOINT],
            "m1521_manifest_sha256": B.EXPECTED[B.M1521_MANIFEST],
            "m1547_base_source_sha256": BASE_SOURCE_SHA256,
            "contract_sha256": sha256(CONTRACT),
            "docs359_sha256": B.EXPECTED[B.DOCS359]},
        "population": {"selection": "per sequence positions 0,4,9; all decoder layers",
            "selected_samples": selected_samples, "selected_calls": len(selected_calls),
            "selected_destinations_per_call": B.SITES_PER_CALL,
            "selected_destinations_per_layer": 9 * B.SITES_PER_CALL,
            "calls": selected_calls},
        "accounting": {"debt_owner": "destination_x_output_tile",
            "all_legal_spatial_sources_and_taps_accumulated": True,
            "fixed_source_group_order": True, "runtime_sorter": False,
            "decision_before_weight_fetch": "structural_model_only_not_measured"},
        "block_results": block_results,
        "decision": {"passing_configs": [row["block"] for row in passing],
            "destination_screen": "PASS" if passing else "NO_GO",
            "next_authorized_request": ("compact incremental FC/patch capture only"
                                        if passing else "none"),
            "capture_executed": False, "rtl_authorized": False,
            "aee_authorized": False, "performance_claim_authorized": False},
        "limitations": [
            "decoder planes are binary and are not the final analog FC/patch scope",
            "epsilon/debt is a dimensionless local output bound, not AEE",
            "FP32 weights define the local bound; metadata bytes use a hypothetical INT8 capacity denominator",
            "no paired forward, address-timed replay, cycle, traffic, energy or RTL evaluation was run"],
        "claim_boundary": dict(contract["claim_boundary"])}
    write_sealed_result(output, result)
    print(status)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.preflight:
        manifest = B.verify_inputs()
        selected, _samples = B.select_records(manifest)
        require(CONTRACT.is_file(), "missing M1554 contract")
        require(len(selected) == 36, "preflight population drift")
        print("PASS_M1554_PREFLIGHT calls=36 destinations_per_call=64 gpu=false ssh=false rtl=false")
        return
    run(args.output)


if __name__ == "__main__":
    main()

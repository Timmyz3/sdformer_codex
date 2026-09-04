#!/usr/bin/env python3
"""Independent fail-closed hammer for M1554.

This script intentionally does not import the M1554/M1547 analyzers.  It
reconstructs the selected population, ConvTranspose2d destination mapping,
block bounds, debt decisions, metadata accounting and exact samples from the
sealed checkpoint and M1521 payloads.  No performance, AEE, traffic, energy or
RTL claim is produced.
"""

from __future__ import print_function

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
AUTHOR_SOURCE = HW / "system_simulator/scripts/analyze_m1554_ep34_s2_ccbs_destination_debt_successor.py"
AUTHOR_TEST = HW / "system_simulator/tests/test_m1554_ep34_s2_ccbs_destination_debt_successor.py"
AUTHOR_CONTRACT = HW / "contracts/m1554_ep34_s2_ccbs_destination_debt_successor_contract_r1_20260831.json"
AUTHOR_ROOT = HW / "results/m1554_ep34_s2_ccbs_destination_debt_successor_r1_20260831"
AUTHOR_RESULT = AUTHOR_ROOT / "m1554_ep34_s2_ccbs_destination_debt_successor_r1.json"
MANIFEST = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831/manifest.json"
PAYLOAD_ROOT = MANIFEST.parent
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    AUTHOR_SOURCE: "3dc49b9d8985b53f449f233d3cf9062077f6029724cc4537c539b9e0a2cf0752",
    AUTHOR_TEST: "408902855689260609d790ecfc31653ce1526da2f029f1ea86f6f4cb8f362888",
    AUTHOR_CONTRACT: "a93c5695dd9c752b3ee1abb63dcc1bf156adc773d8fd55c68b38fcf861b31dc1",
    AUTHOR_RESULT: "3763bbe9da77a0eb9c776044dcf4204cdc4c9f0886b3b757b2e60d1bb2c88d85",
    MANIFEST: "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304",
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
WEIGHT_SHA256 = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
GEOMETRY = (
    (1536, 384, 15, 20, 30, 40),
    (770, 192, 30, 40, 60, 80),
    (386, 96, 60, 80, 120, 160),
    (194, 96, 120, 160, 240, 320),
)
BLOCKS = ((8, 16), (16, 16), (32, 16))
EPSILONS = (0.0, 0.05, 0.10, 0.20, 0.30)
SAMPLE_POSITIONS = (0, 4, 9)
DESTINATIONS_PER_CALL = 64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def q(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and bool(np.isfinite(values).all()),
            "empty/nonfinite quantile population")
    return {"count": int(values.size), "min": float(values.min()),
            "median": float(np.quantile(values, 0.5)),
            "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max())}


def sample_indices(count, wanted):
    import numpy as np
    return sorted(set(int(value) for value in
                      np.linspace(0, count - 1, min(count, wanted), dtype=np.int64)))


def destination_sources(oy, ox, height, width):
    """Inverse of oy=2*iy-1+ky, ox=2*ix-1+kx."""
    result = []
    for ky in range(3):
        y = int(oy) + 1 - ky
        if y % 2 != 0:
            continue
        iy = y // 2
        if iy < 0 or iy >= int(height):
            continue
        for kx in range(3):
            x = int(ox) + 1 - kx
            if x % 2 != 0:
                continue
            ix = x // 2
            if 0 <= ix < int(width):
                result.append((iy, ix, ky, kx))
    require(1 <= len(result) <= 4, "illegal destination contributor count")
    require(len(result) == len(set(result)), "duplicate contributor")
    return tuple(result)


def forward_map(height, width):
    """Independent forward enumeration for K3/S2/P1/OP1."""
    mapping = {}
    for iy in range(height):
        for ix in range(width):
            for ky in range(3):
                oy = 2 * iy - 1 + ky
                if oy < 0 or oy >= 2 * height:
                    continue
                for kx in range(3):
                    ox = 2 * ix - 1 + kx
                    if 0 <= ox < 2 * width:
                        mapping.setdefault((oy, ox), []).append((iy, ix, ky, kx))
    return dict((key, tuple(value)) for key, value in mapping.items())


def fixed_drop(bounds, epsilon, reference):
    require(epsilon >= 0.0 and reference >= 0.0, "bad debt arguments")
    if reference == 0.0:
        require(all(float(value) == 0.0 for value in bounds),
                "zero reference with nonzero bound")
        return [True for _value in bounds], 0.0
    limit = epsilon * reference
    debt = 0.0
    mask = []
    for raw in bounds:
        value = float(raw)
        require(value >= 0.0 and math.isfinite(value), "bad block bound")
        drop = value == 0.0 or debt + value <= limit
        mask.append(drop)
        if drop:
            debt += value
    return mask, debt


def selected_records(manifest):
    records = manifest["records"]
    sequences = sorted(set(row["sequence"] for row in records))
    require(sequences == ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"],
            "sequence drift")
    selected = []
    for sequence in sequences:
        samples = sorted(set((row["replay_sample_ordinal"], row["global_sample_id"])
                             for row in records if row["sequence"] == sequence))
        require(len(samples) == 10, "sample population drift")
        for position in SAMPLE_POSITIONS:
            sample = samples[position]
            rows = [row for row in records
                    if row["sequence"] == sequence and
                    row["replay_sample_ordinal"] == sample[0]]
            rows.sort(key=lambda row: row["module_ordinal"])
            require([row["module_ordinal"] for row in rows] == [0, 1, 2, 3],
                    "module population drift")
            selected.extend(rows)
    require(len(selected) == 36, "selected call count drift")
    return selected


def unpack_destinations(record):
    import numpy as np
    path = PAYLOAD_ROOT / record["positive_output"]
    require(sha256(path) == record["positive_output_sha256"], "payload SHA drift")
    raw = np.fromfile(str(path), dtype=np.uint8)
    require(int(raw.size) == int(record["plane_bytes"]), "payload extent drift")
    bits = np.unpackbits(raw, bitorder="little")[:int(record["elements"])]
    shape = tuple(int(value) for value in record["shape"])
    layer = int(record["module_ordinal"])
    cin, _cout, hin, win, hout, wout = GEOMETRY[layer]
    require(shape == (10, 1, cin, hin, win), "payload shape drift")
    plane = bits.reshape(shape)
    flat_indices = np.linspace(0, 10 * hout * wout - 1,
                               DESTINATIONS_PER_CALL, dtype=np.int64)
    output = []
    for flat in flat_indices.tolist():
        timestep = int(flat) // (hout * wout)
        spatial = int(flat) % (hout * wout)
        oy, ox = spatial // wout, spatial % wout
        sources = destination_sources(oy, ox, hin, win)
        activity = np.stack([plane[timestep, 0, :, iy, ix]
                             for iy, ix, _ky, _kx in sources], axis=0)
        require(activity.shape == (len(sources), cin), "activity shape drift")
        output.append((timestep, oy, ox, sources, activity))
    require(len(output) == DESTINATIONS_PER_CALL, "destination count drift")
    return output


def load_weights():
    import numpy as np
    import torch
    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    require(type(wrapper) is dict and set(wrapper) == set(["model_state_dict"]),
            "checkpoint wrapper drift")
    state = wrapper["model_state_dict"]
    require(len(state) == 921, "checkpoint state count drift")
    weights = []
    for layer in range(4):
        key = "sttmultires_unet.decoders.{}.deconv.0.weight".format(layer)
        value = state[key].detach().cpu().contiguous().numpy()
        require(value.dtype == np.float32 and
                value.shape == (GEOMETRY[layer][0], GEOMETRY[layer][1], 3, 3),
                "ConvTranspose weight layout drift")
        require(hashlib.sha256(value.tobytes(order="C")).hexdigest() ==
                WEIGHT_SHA256[layer], "weight content drift")
        weights.append(value)
    return weights


def torch_layout_attack():
    """Use nonsymmetric taps/channels to reject flip/layout mistakes."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    source = torch.tensor([[[[1.0, -2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [-7.0, 8.0]]]], dtype=torch.float64)
    weight = torch.arange(2 * 3 * 3 * 3, dtype=torch.float64).reshape(2, 3, 3, 3)
    weight = (weight - 19.0) / 7.0
    expected = functional.conv_transpose2d(source, weight, stride=2,
                                           padding=1, output_padding=1)
    actual = np.zeros(tuple(expected.shape), dtype=np.float64)
    src = source.numpy()
    wgt = weight.numpy()
    for oy in range(expected.shape[2]):
        for ox in range(expected.shape[3]):
            for iy, ix, ky, kx in destination_sources(oy, ox, 2, 2):
                actual[0, :, oy, ox] += np.matmul(src[0, :, iy, ix],
                                                  wgt[:, :, ky, kx])
    require(np.array_equal(actual, expected.numpy()),
            "PyTorch ConvTranspose layout/tap mismatch")
    # A kernel flip must not accidentally be equivalent on this witness.
    flipped = np.zeros_like(actual)
    for oy in range(expected.shape[2]):
        for ox in range(expected.shape[3]):
            for iy, ix, ky, kx in destination_sources(oy, ox, 2, 2):
                flipped[0, :, oy, ox] += np.matmul(src[0, :, iy, ix],
                    wgt[:, :, 2 - ky, 2 - kx])
    require(not np.array_equal(flipped, expected.numpy()),
            "kernel-flip attack lacked discrimination")


def analyze(weights, observations, group, output_tile):
    import numpy as np
    author_exact_ratios = []
    capacity_utilization = []
    global_counts = dict((str(epsilon), [0, 0, []]) for epsilon in EPSILONS)
    active_counts = dict((str(epsilon), [0, 0, []]) for epsilon in EPSILONS)
    witness = {}
    metadata_bytes = 0
    old_bytes = 0
    weight_bytes = 0
    metadata_reads = 0
    exact_zero = 0
    exact_positive = 0
    contributor_hist = {}
    all_bounds_le_reference = True

    for layer in range(4):
        weight = weights[layer]
        cin, cout, _kh, _kw = weight.shape
        gblocks = (cin + group - 1) // group
        oblocks = (cout + output_tile - 1) // output_tile
        metadata_bytes += 2 * gblocks * oblocks
        old_bytes += 2 * cin * oblocks
        weight_bytes += cin * cout * 9
        metadata_reads += len(observations[layer]) * gblocks * oblocks
        max_weight = float(np.abs(weight).max())
        metadata = np.zeros((gblocks, oblocks), dtype=np.float64)
        for gb in range(gblocks):
            gs, ge = gb * group, min(cin, (gb + 1) * group)
            require(ge > gs and ge - gs <= group, "tail group drift")
            for ob in range(oblocks):
                os_, oe = ob * output_tile, min(cout, (ob + 1) * output_tile)
                metadata[gb, ob] = float(np.abs(weight[gs:ge, os_:oe, :, :]).max())

        bounds_per_observation = []
        for _t, _oy, _ox, sources, activity in observations[layer]:
            source_count = len(sources)
            contributor_hist[str(source_count)] = contributor_hist.get(str(source_count), 0) + 1
            padded = np.zeros((source_count, gblocks * group), dtype=np.uint8)
            padded[:, :cin] = activity
            counts = padded.reshape(source_count, gblocks, group).sum(axis=(0, 2))
            bounds = counts[:, None].astype(np.float64) * metadata
            bounds_per_observation.append(bounds)
            global_reference = max_weight * float(cin * source_count)
            for ob in range(oblocks):
                values = bounds[:, ob].tolist()
                active_reference = float(bounds[:, ob].sum())
                require(active_reference <= global_reference + 1e-9,
                        "block bounds exceed global capacity reference")
                all_bounds_le_reference = all_bounds_le_reference and (
                    active_reference <= global_reference + 1e-9)
                capacity_utilization.append(active_reference / global_reference)
                for epsilon in EPSILONS:
                    mask, debt = fixed_drop(values, epsilon, global_reference)
                    row = global_counts[str(epsilon)]
                    row[0] += len(mask)
                    row[1] += sum(1 for value in mask if value)
                    row[2].append(debt / global_reference)
                    amask, adebt = fixed_drop(values, epsilon, active_reference)
                    arow = active_counts[str(epsilon)]
                    arow[0] += len(amask)
                    arow[1] += sum(1 for value in amask if value)
                    arow[2].append(0.0 if active_reference == 0.0 else
                                   adebt / active_reference)
                    if epsilon == 0.1:
                        for gb, dropped in enumerate(mask):
                            key = (layer, gb, ob)
                            witness[key] = witness.get(key, 0) | (1 if dropped else 2)

        for destination_index in sample_indices(len(observations[layer]), 18):
            _t, _oy, _ox, sources, activity = observations[layer][destination_index]
            bounds = bounds_per_observation[destination_index]
            for gb in sample_indices(gblocks, 16):
                gs, ge = gb * group, min(cin, (gb + 1) * group)
                for ob in sample_indices(oblocks, 8):
                    bound = float(bounds[gb, ob])
                    if bound <= 0.0:
                        continue
                    os_, oe = ob * output_tile, min(cout, (ob + 1) * output_tile)
                    exact = np.zeros((oe - os_,), dtype=np.float64)
                    for source_index, (_iy, _ix, ky, kx) in enumerate(sources):
                        active = activity[source_index, gs:ge].astype(bool)
                        if bool(active.any()):
                            exact += weight[gs:ge, os_:oe, ky, kx][active, :].sum(axis=0)
                    exact_value = float(np.abs(exact).max())
                    exact_positive += 1
                    require(exact_value <= bound + 1e-6,
                            "certified block bound violated")
                    if exact_value <= 1e-12:
                        exact_zero += 1
                    else:
                        author_exact_ratios.append(bound / exact_value)

    def epsilon_rows(population):
        output = []
        for epsilon in EPSILONS:
            blocks, dropped, debts = population[str(epsilon)]
            output.append({"epsilon": epsilon, "block_decisions": blocks,
                           "drop_fraction": float(dropped) / float(blocks),
                           "normalized_debt": q(debts)})
        return output

    return {
        "block": {"source_group": group, "output_tile": output_tile},
        "contributor_histogram": contributor_hist,
        "metadata": {"bytes": metadata_bytes, "old_g11_bytes": old_bytes,
                     "hypothetical_int8_weight_bytes": weight_bytes,
                     "ratio": float(metadata_bytes) / float(weight_bytes),
                     "reduction": float(old_bytes) / float(metadata_bytes),
                     "reads": metadata_reads},
        "exact_ratio": q(author_exact_ratios),
        "exact_positive_bound_count": exact_positive,
        "exact_zero_collision_fraction": float(exact_zero) / float(exact_positive),
        "capacity_utilization": q(capacity_utilization),
        "global_capacity_reference": epsilon_rows(global_counts),
        "active_bound_mass_reference": epsilon_rows(active_counts),
        "dynamic_witness_count": sum(1 for value in witness.values() if value == 3),
        "dynamic_static_blocks": len(witness),
        "all_bounds_le_global_reference": all_bounds_le_reference,
    }


def compare(author, independent):
    import numpy as np
    expected_status = "PASS_DESTINATION_DEBT_SCREEN__INCREMENTAL_FC_PATCH_CAPTURE_REQUEST_ONLY__NO_AEE_PERFORMANCE_OR_RTL"
    require(author["status"] == expected_status, "author status drift")
    require(author["population"]["selected_calls"] == 36 and
            author["population"]["selected_destinations_per_call"] == 64,
            "author population drift")
    for authored, replayed in zip(author["block_results"], independent):
        require(authored["block"] == replayed["block"], "block order drift")
        require(authored["spatial_contributor_histogram"] ==
                replayed["contributor_histogram"], "contributor histogram mismatch")
        am = authored["metadata_aggregate"]
        rm = replayed["metadata"]
        require(am["metadata_bytes"] == rm["bytes"] and
                am["old_g11_per_source_metadata_bytes"] == rm["old_g11_bytes"] and
                am["int8_weight_bytes"] == rm["hypothetical_int8_weight_bytes"] and
                am["metadata_reads_over_selected_destinations"] == rm["reads"],
                "metadata accounting mismatch")
        require(np.isclose(am["metadata_to_int8_weight_bytes"], rm["ratio"], rtol=0, atol=1e-15) and
                np.isclose(am["reduction_vs_old_g11"], rm["reduction"], rtol=0, atol=1e-12),
                "metadata ratio mismatch")
        for key in ("count", "min", "median", "p90", "p99", "max"):
            require(np.isclose(authored["bound_to_exact_destination_contribution_ratio_sample"][key],
                               replayed["exact_ratio"][key], rtol=0, atol=1e-10),
                    "exact ratio mismatch: " + key)
        require(np.isclose(authored["positive_bound_exact_zero_collision_fraction"],
                           replayed["exact_zero_collision_fraction"], rtol=0, atol=1e-15),
                "zero collision mismatch")
        for authored_e, replayed_e in zip(authored["epsilon_diagnostics"],
                                          replayed["global_capacity_reference"]):
            require(authored_e["epsilon_normalized"] == replayed_e["epsilon"],
                    "epsilon order drift")
            require(authored_e["block_decisions"] == replayed_e["block_decisions"] and
                    np.isclose(authored_e["drop_fraction"], replayed_e["drop_fraction"],
                               rtol=0, atol=1e-15), "drop replay mismatch")
        require(authored["dynamic_witness"]["blocks_with_both_keep_and_drop"] ==
                replayed["dynamic_witness_count"], "dynamic witness mismatch")


def main():
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256(path) == expected,
                "identity drift: " + str(path))
    # Verify the author's nested seal from its own directory.
    sums = AUTHOR_ROOT / "SHA256SUMS"
    seal = AUTHOR_ROOT / "SHA256SUMS.seal.sha256"
    require(seal.read_text().split() == [sha256(sums), "SHA256SUMS"],
            "author outer seal drift")
    for line in sums.read_text().splitlines():
        digest, name = line.split(None, 1)
        require(sha256(AUTHOR_ROOT / name.strip()) == digest,
                "author member seal drift: " + name)

    contract = strict_json(AUTHOR_CONTRACT)
    require(contract["required_accounting"]["owner"] == "destination_x_output_tile" and
            contract["required_accounting"]["all_legal_spatial_sources_and_taps_accumulated_before_drop"] is True,
            "author contract debt scope drift")
    require(all(contract["claim_boundary"][key] is False for key in
                ["aee", "accuracy_admission", "capture_executed", "cycles",
                 "traffic", "speedup", "system_speedup", "energy", "rtl",
                 "vcs", "eda", "paper_headline"]), "claim boundary drift")

    # Exhaustively compare forward and inverse maps for every production shape.
    contributor_hist_all = {}
    for _cin, _cout, height, width, hout, wout in GEOMETRY:
        mapping = forward_map(height, width)
        require(len(mapping) == hout * wout, "forward map output coverage drift")
        for oy in range(hout):
            for ox in range(wout):
                inverse = destination_sources(oy, ox, height, width)
                require(set(inverse) == set(mapping[(oy, ox)]) and
                        len(inverse) == len(mapping[(oy, ox)]),
                        "forward/inverse mapping mismatch")
                key = str(len(inverse))
                contributor_hist_all[key] = contributor_hist_all.get(key, 0) + 1
    torch_layout_attack()

    manifest = strict_json(MANIFEST)
    selected = selected_records(manifest)
    observations = dict((layer, []) for layer in range(4))
    selected_call_ids = []
    for record in selected:
        observations[int(record["module_ordinal"])].extend(unpack_destinations(record))
        selected_call_ids.append(int(record["global_call_ordinal"]))
    require(all(len(observations[layer]) == 576 for layer in range(4)),
            "36-call x 64-destination population drift")
    require(len(set(selected_call_ids)) == 36, "selected call identity duplication")

    weights = load_weights()
    replay = [analyze(weights, observations, group, output_tile)
              for group, output_tile in BLOCKS]
    author = strict_json(AUTHOR_RESULT)
    compare(author, replay)

    # Independently reproduce the published gate pattern.
    require(replay[0]["metadata"]["reduction"] < 8.0, "8x16 strict tail gate changed")
    require(replay[1]["metadata"]["reduction"] >= 8.0 and
            replay[1]["exact_ratio"]["median"] <= 4.0 and
            replay[1]["exact_ratio"]["p90"] <= 12.0,
            "16x16 author gate changed")
    require(replay[2]["exact_ratio"]["median"] > 4.0,
            "32x16 ratio failure changed")

    epsilon01 = []
    for row in replay:
        global_row = [value for value in row["global_capacity_reference"]
                      if value["epsilon"] == 0.1][0]
        active_row = [value for value in row["active_bound_mass_reference"]
                      if value["epsilon"] == 0.1][0]
        epsilon01.append({"block": row["block"],
                          "global_capacity_drop_fraction": global_row["drop_fraction"],
                          "active_bound_mass_drop_fraction": active_row["drop_fraction"],
                          "capacity_utilization": row["capacity_utilization"]})

    result = {
        "schema": "m1555_m1554_s2_destination_debt_independent_hammer_r1_v1",
        "status": "PASS_RECOMPUTE__CONDITIONAL_CAPTURE_ONLY__REFERENCE_GATE_REPAIR_REQUIRED_BEFORE_S2_ADMISSION",
        "python": sys.version.split()[0],
        "attacks": {
            "count": 24,
            "passed": 24,
            "items": [
                "author SHA identities", "author nested seal", "docs359 identity",
                "strict JSON duplicate keys", "36-call identity", "3-sequence sample positions",
                "64 destinations per call", "four decoder layers", "little-endian bit unpack",
                "exhaustive forward/inverse K3S2P1OP1 mapping", "boundary destinations",
                "1/2/4 contributor cardinality", "PyTorch ConvTranspose layout",
                "nonsymmetric tap order", "kernel-flip rejection", "weight content SHA",
                "tail source groups", "tail output tiles", "destination-owned debt",
                "all-source/tap bound", "exact sampled bound", "metadata denominator/reads",
                "three block gate pattern", "claim-boundary denial"
            ]
        },
        "population": {"calls": 36, "destinations_per_call": 64,
                       "destinations_per_layer": 576,
                       "selected_destination_blocks": 36 * 64 * 3,
                       "exhaustive_geometry_contributor_histogram": contributor_hist_all},
        "independent_replay": replay,
        "reference_audit": {
            "global_reference_formula": "layer_max_abs_weight * Cin * spatial_contributor_count",
            "mathematically_safe_absolute_capacity_upper_bound": True,
            "all_active_bound_mass_le_reference": all(
                row["all_bounds_le_global_reference"] for row in replay),
            "scientifically_sufficient_relative_error_or_AEE_reference": False,
            "reason": "It normalizes each sparse destination by the hypothetical all-Cin-active layer maximum. The median active bound mass is only about five percent of this capacity, so epsilon=0.1 often permits dropping the entire observed destination. This is a valid absolute worst-case-capacity bound but not a calibrated relative-output, state or AEE budget.",
            "epsilon_0p1_comparison": epsilon01,
            "required_successor_gate": "Report both global-capacity and per-destination active-bound-mass references; precommit the latter (or a tighter safe layer/output scale) for retained-data screening, and prohibit the 99.2% global-capacity drop fraction from motivating performance, AEE or RTL."
        },
        "verdict": {
            "incremental_fc_patch_capture": "CONDITIONALLY_ALLOWED",
            "condition": "The compact producer must preserve enough group magnitude/code and weight-block identity to recompute an activity-relative safe reference. The capture is data acquisition only; M1554 epsilon=0.1 drop fractions are not an admission gate.",
            "s2_mechanism_admitted": False,
            "paired_aee_authorized": False,
            "performance_or_rtl_authorized": False,
            "author_16x16_pass_reinterpreted_as": "metadata/dynamic-witness survival only"
        }
    }
    output = Path(__file__).resolve().parent / "independent_recompute.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()

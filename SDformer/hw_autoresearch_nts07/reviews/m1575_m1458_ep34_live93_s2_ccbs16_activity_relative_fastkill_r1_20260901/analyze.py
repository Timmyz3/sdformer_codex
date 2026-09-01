#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""M1575: CPU-only 16x16 CCBS activity-relative fast-kill on M1458.

The analysis consumes all 30 decoder samples from the sealed ep34 live93
capture.  It intentionally produces no AEE, cycle, traffic, energy, RTL or
EDA claim.  A dropped source-group is charged to one destination/output-tile,
and its epsilon reference is the sum of observed active block bounds for that
same owner.  This repairs the dense global-capacity reference rejected by
M1555.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
MANIFEST = CAPTURE / "manifest.json"
ORDERED = CAPTURE / "unified_ordered_records.jsonl"
ADMISSION = CAPTURE / "m1434_admission.json"
CAPTURE_SUMS = CAPTURE / "SHA256SUMS"
CAPTURE_OUTER = CAPTURE / "SHA256SUMS.seal.sha256"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831/review.json"
M1513 = HW / "reviews/m1513_m1512_m1458_ep34_production_provenance_addendum_r1_20260831/review.json"
M1555 = HW / "reviews/m1555_m1554_s2_destination_debt_independent_hammer_r1_20260901/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    MANIFEST: "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    ORDERED: "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    ADMISSION: "441a61a41a7080b03f2ac10c557fbf6aaf4abcea0d9aad4e6b44c9f15075eef0",
    CAPTURE_SUMS: "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    CAPTURE_OUTER: "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    M1512: "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    M1513: "1eb36a76fac29d5d15607dbb4ee3f9a434c4b0686843acac11f18116b48c7aaa",
    M1555: "e9c2313bfb0f9d68e98e3bbb0a72d358991f43b1fd93eb1704153f24f03fc7c4",
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
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
GROUP = 16
OUTPUT_TILE = 16
EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.10)
DYNAMIC_EPSILON = 0.10
DESTINATIONS_PER_CALL = 64
EXACT_DESTINATIONS_PER_CALL = 8
EXACT_OUTPUT_TILES_PER_DESTINATION = 4

CLAIM_BOUNDARY = {
    "capture_reused": True,
    "checkpoint_bound": True,
    "decoder_only": True,
    "activity_relative_bound": True,
    "paired_aee": False,
    "accuracy_admission": False,
    "cycles": False,
    "traffic": False,
    "energy": False,
    "speedup": False,
    "system_speedup": False,
    "rtl": False,
    "vcs": False,
    "eda": False,
    "paper_headline": False,
}


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
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def q(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and bool(np.isfinite(values).all()),
            "empty or nonfinite quantile population")
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def sample_indices(count, wanted):
    import numpy as np
    require(count > 0 and wanted > 0, "bad sampling arguments")
    return sorted(set(int(v) for v in
                      np.linspace(0, count - 1, min(count, wanted), dtype=np.int64)))


def destination_sources(oy, ox, height, width):
    result = []
    for ky in range(3):
        y = int(oy) + 1 - ky
        if y % 2:
            continue
        iy = y // 2
        if iy < 0 or iy >= int(height):
            continue
        for kx in range(3):
            x = int(ox) + 1 - kx
            if x % 2:
                continue
            ix = x // 2
            if 0 <= ix < int(width):
                result.append((iy, ix, ky, kx))
    require(1 <= len(result) <= 4 and len(result) == len(set(result)),
            "bad K3/S2/P1/OP1 destination inverse")
    return tuple(result)


def fixed_drop(bounds, epsilon, reference):
    require(epsilon >= 0.0 and reference >= 0.0, "bad debt arguments")
    if reference == 0.0:
        require(all(float(value) == 0.0 for value in bounds),
                "zero reference with positive bound")
        return [True] * len(bounds), 0.0
    debt = 0.0
    limit = epsilon * reference
    mask = []
    for raw in bounds:
        value = float(raw)
        require(value >= 0.0 and math.isfinite(value), "bad block bound")
        drop = value == 0.0 or debt + value <= limit
        mask.append(drop)
        if drop:
            debt += value
    require(debt <= limit + max(1.0e-10, abs(limit) * 1.0e-12),
            "debt budget violated")
    return mask, debt


def verify_inputs():
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256(path) == expected,
                "identity drift: " + str(path))
    require(CAPTURE_OUTER.read_text(encoding="utf-8").split() ==
            [EXPECTED[CAPTURE_SUMS], "SHA256SUMS"], "capture outer seal drift")
    manifest = strict_json(MANIFEST)
    admission = strict_json(ADMISSION)
    m1512 = strict_json(M1512)
    m1513 = strict_json(M1513)
    require(manifest["schema"] == "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1",
            "capture schema drift")
    selected = manifest["identity"]["selection"]["selected"]
    require(selected["candidate_id"] == "resume_ep34" and
            selected["checkpoint"]["sha256"] == EXPECTED[CHECKPOINT],
            "capture checkpoint selection drift")
    require(admission["status"] == "PASS" and admission["payload_files"] == 640,
            "capture admission drift")
    require(m1512["status"] == "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT",
            "M1512 capture-content hammer drift")
    require(m1513["status"] == "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE",
            "M1513 production-provenance hammer drift")
    return manifest


def load_records():
    records = []
    with ORDERED.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("category") != "decoder_convtranspose" or row.get("cohort") != "decoder":
                continue
            match = re.fullmatch(r"sttmultires_unet\.decoders\.(\d)\.deconv\.0", row["name"])
            require(match is not None, "decoder name drift")
            row["layer"] = int(match.group(1))
            records.append(row)
    require(len(records) == 120, "decoder record count drift")
    by_sequence = {sequence: [] for sequence in SEQUENCES}
    identities = set()
    for row in records:
        require(row["sequence"] in by_sequence, "sequence drift")
        layer = row["layer"]
        cin, _cout, hin, win, _hout, _wout = GEOMETRY[layer]
        require(tuple(row["input"]["shape"]) == (10, 1, cin, hin, win),
                "decoder input shape drift")
        require(row["input"]["negative"] == 0 and row["input"]["positive"] == row["input"]["active"],
                "decoder input is not nonnegative binary support")
        require(row["payload"]["retained"] is True, "decoder payload not retained")
        key = (row["global_sample_id"], layer)
        require(key not in identities, "duplicate sample/layer")
        identities.add(key)
        by_sequence[row["sequence"]].append(row)
    for sequence, rows in by_sequence.items():
        require(len(rows) == 40 and len(set(r["global_sample_id"] for r in rows)) == 10,
                "per-sequence population drift: " + sequence)
        require(all(sum(1 for r in rows if r["global_sample_id"] == sid) == 4
                    for sid in set(r["global_sample_id"] for r in rows)),
                "sample layer population drift")
    records.sort(key=lambda row: (SEQUENCES.index(row["sequence"]),
                                  row["global_sample_id"], row["layer"]))
    return records


def load_weights():
    import numpy as np
    import torch
    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    require(type(wrapper) is dict and set(wrapper) == {"model_state_dict"},
            "checkpoint wrapper drift")
    state = wrapper["model_state_dict"]
    require(len(state) == 921, "checkpoint state population drift")
    weights = []
    for layer, geometry in enumerate(GEOMETRY):
        key = "sttmultires_unet.decoders.{}.deconv.0.weight".format(layer)
        value = state[key].detach().cpu().contiguous().numpy()
        require(value.dtype == np.float32 and
                value.shape == (geometry[0], geometry[1], 3, 3),
                "decoder weight shape/dtype drift")
        require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == WEIGHT_SHA256[layer],
                "decoder weight SHA drift")
        weights.append(value.copy())
    return weights


def layer_static(weight):
    import numpy as np
    cin, cout, kh, kw = weight.shape
    require((kh, kw) == (3, 3), "kernel geometry drift")
    gblocks = (cin + GROUP - 1) // GROUP
    oblocks = (cout + OUTPUT_TILE - 1) // OUTPUT_TILE
    maxima = np.zeros((gblocks, oblocks), dtype=np.float64)
    payload = np.zeros((gblocks, oblocks), dtype=np.int64)
    for gb in range(gblocks):
        gs, ge = gb * GROUP, min(cin, (gb + 1) * GROUP)
        for ob in range(oblocks):
            os_, oe = ob * OUTPUT_TILE, min(cout, (ob + 1) * OUTPUT_TILE)
            maxima[gb, ob] = float(np.abs(weight[gs:ge, os_:oe, :, :]).max())
            payload[gb, ob] = (ge - gs) * (oe - os_) * kh * kw
    metadata_bytes = 2 * gblocks * oblocks
    old_g11_bytes = 2 * cin * oblocks
    int8_weight_bytes = cin * cout * kh * kw
    return {
        "maxima": maxima,
        "payload": payload,
        "gblocks": gblocks,
        "oblocks": oblocks,
        "metadata_bytes": metadata_bytes,
        "old_g11_bytes": old_g11_bytes,
        "int8_weight_bytes": int8_weight_bytes,
    }


def payload_reader(record):
    import numpy as np
    relative = record["payload"]["support_sign"]
    path = CAPTURE / relative
    require(path.is_file() and sha256(path) == record["payload"]["support_sign_sha256"],
            "payload identity drift: " + relative)
    raw = np.fromfile(str(path), dtype=np.uint8)
    positive_bytes = int(record["payload"]["positive_plane_bytes"])
    negative_bytes = int(record["payload"]["negative_plane_bytes"])
    require(int(raw.size) == positive_bytes + negative_bytes,
            "support/sign payload extent drift")
    positive = raw[:positive_bytes]
    negative = raw[positive_bytes:]
    require(not bool(negative.any()), "decoder negative plane is nonzero")
    shape = tuple(int(v) for v in record["input"]["shape"])
    timestep, one, channels, height, width = shape
    require(one == 1 and timestep == 10, "payload tensor shape drift")
    elements = math.prod(shape)
    require(positive_bytes == (elements + 7) // 8, "positive plane size drift")
    spatial_stride = height * width

    def site(t, y, x):
        indices = (int(t) * channels * spatial_stride +
                   np.arange(channels, dtype=np.int64) * spatial_stride +
                   int(y) * width + int(x))
        return ((positive[indices >> 3] >> (indices & 7)) & 1).astype(np.uint8)
    return site


def empty_accumulator():
    return dict((str(epsilon), {
        "block_decisions": 0,
        "dropped": 0,
        "candidate_weight_bytes_proxy": 0,
        "dropped_weight_bytes_proxy": 0,
        "metadata_read_bytes_proxy": 0,
        "debt_sum": 0.0,
        "reference_sum": 0.0,
        "normalized_debt": [],
    }) for epsilon in EPSILONS)


def analyze_record(record, weight, static, accumulator, dynamic_states, quality):
    import numpy as np
    layer = record["layer"]
    cin, cout, hin, win, hout, wout = GEOMETRY[layer]
    site = payload_reader(record)
    destination_indices = sample_indices(10 * hout * wout, DESTINATIONS_PER_CALL)
    exact_destination_ordinals = set(sample_indices(len(destination_indices),
                                                    EXACT_DESTINATIONS_PER_CALL))
    exact_output_tiles = set(sample_indices(static["oblocks"],
                                            EXACT_OUTPUT_TILES_PER_DESTINATION))
    for destination_ordinal, flat in enumerate(destination_indices):
        t = flat // (hout * wout)
        spatial = flat % (hout * wout)
        oy, ox = spatial // wout, spatial % wout
        sources = destination_sources(oy, ox, hin, win)
        source_activity = np.stack([site(t, iy, ix)
                                    for iy, ix, _ky, _kx in sources], axis=0)
        padded = np.zeros((len(sources), static["gblocks"] * GROUP), dtype=np.uint8)
        padded[:, :cin] = source_activity
        counts = padded.reshape(len(sources), static["gblocks"], GROUP).sum(axis=(0, 2))
        bounds = counts[:, None].astype(np.float64) * static["maxima"]
        for ob in range(static["oblocks"]):
            values = bounds[:, ob].tolist()
            reference = float(bounds[:, ob].sum())
            payload_bytes = static["payload"][:, ob]
            dynamic_mask = None
            dynamic_debt = None
            for epsilon in EPSILONS:
                mask, debt = fixed_drop(values, epsilon, reference)
                row = accumulator[str(epsilon)]
                row["block_decisions"] += len(mask)
                row["dropped"] += sum(1 for value in mask if value)
                row["candidate_weight_bytes_proxy"] += int(payload_bytes.sum())
                row["dropped_weight_bytes_proxy"] += int(payload_bytes[np.asarray(mask, dtype=bool)].sum())
                row["metadata_read_bytes_proxy"] += 2 * len(mask)
                row["debt_sum"] += debt
                row["reference_sum"] += reference
                row["normalized_debt"].append(0.0 if reference == 0.0 else debt / reference)
                if epsilon == DYNAMIC_EPSILON:
                    dynamic_mask = np.asarray(mask, dtype=bool)
                    dynamic_debt = debt
                    for gb, dropped in enumerate(mask):
                        dynamic_states[gb, ob] |= (1 if dropped else 2)

            if (destination_ordinal not in exact_destination_ordinals or
                    ob not in exact_output_tiles):
                continue
            require(dynamic_mask is not None and dynamic_debt is not None,
                    "dynamic epsilon mask missing")
            os_, oe = ob * OUTPUT_TILE, min(cout, (ob + 1) * OUTPUT_TILE)
            full = np.zeros((oe - os_,), dtype=np.float64)
            dropped = np.zeros_like(full)
            drop_channels = np.repeat(dynamic_mask, GROUP)[:cin]
            for source_index, (_iy, _ix, ky, kx) in enumerate(sources):
                active = source_activity[source_index].astype(bool)
                if bool(active.any()):
                    source_weight = weight[:, os_:oe, ky, kx]
                    full += source_weight[active, :].sum(axis=0, dtype=np.float64)
                    dropped_active = active & drop_channels
                    if bool(dropped_active.any()):
                        dropped += source_weight[dropped_active, :].sum(axis=0,
                                                                       dtype=np.float64)
            exact_error = float(np.abs(dropped).max())
            full_linf = float(np.abs(full).max())
            tolerance = max(1.0e-6, abs(dynamic_debt) * 2.0e-6)
            require(exact_error <= dynamic_debt + tolerance,
                    "activity-relative certified debt violated")
            reference = float(bounds[:, ob].sum())
            quality["exact_error_over_activity_bound"].append(
                0.0 if reference == 0.0 else exact_error / reference)
            quality["certified_debt_over_activity_bound"].append(
                0.0 if reference == 0.0 else dynamic_debt / reference)
            if dynamic_debt > 0.0:
                quality["exact_error_over_certified_debt"].append(exact_error / dynamic_debt)
            if full_linf > 1.0e-6:
                quality["exact_error_over_full_output_linf"].append(exact_error / full_linf)
            else:
                quality["near_zero_full_output_count"] += 1
            quality["sampled_output_tiles"] += 1


def summarize_accumulator(accumulator):
    result = []
    for epsilon in EPSILONS:
        row = accumulator[str(epsilon)]
        blocks = row["block_decisions"]
        candidate = row["candidate_weight_bytes_proxy"]
        reference = row["reference_sum"]
        result.append({
            "epsilon": epsilon,
            "block_decisions": blocks,
            "keep": blocks - row["dropped"],
            "drop": row["dropped"],
            "drop_fraction": float(row["dropped"]) / float(blocks),
            "candidate_weight_bytes_proxy": candidate,
            "dropped_weight_bytes_proxy": row["dropped_weight_bytes_proxy"],
            "potential_weight_byte_suppression_fraction":
                float(row["dropped_weight_bytes_proxy"]) / float(candidate),
            "metadata_read_bytes_proxy": row["metadata_read_bytes_proxy"],
            "metadata_to_candidate_weight_byte_ratio":
                float(row["metadata_read_bytes_proxy"]) / float(candidate),
            "aggregate_certified_debt_over_activity_bound_mass":
                0.0 if reference == 0.0 else row["debt_sum"] / reference,
            "per_destination_output_tile_normalized_debt": q(row["normalized_debt"]),
            "paired_aee": False,
            "cycles": False,
            "traffic": False,
        })
    return result


def merge_accumulators(accumulators):
    merged = empty_accumulator()
    for source in accumulators:
        for epsilon in EPSILONS:
            key = str(epsilon)
            for field in ("block_decisions", "dropped", "candidate_weight_bytes_proxy",
                          "dropped_weight_bytes_proxy", "metadata_read_bytes_proxy"):
                merged[key][field] += source[key][field]
            for field in ("debt_sum", "reference_sum"):
                merged[key][field] += source[key][field]
            merged[key]["normalized_debt"].extend(source[key]["normalized_debt"])
    return merged


def summarize_quality(quality):
    output = {
        "sampled_output_tiles": quality["sampled_output_tiles"],
        "near_zero_full_output_count": quality["near_zero_full_output_count"],
        "definition": {
            "primary_certified_proxy": "Linf exact dropped contribution / sum_G M(G,O)*A(G)",
            "secondary_empirical_proxy": "Linf exact dropped contribution / Linf exact unpruned output for non-near-zero outputs",
            "network_metric": False,
            "paired_aee": False,
        },
    }
    for key in ("exact_error_over_activity_bound",
                "certified_debt_over_activity_bound",
                "exact_error_over_certified_debt",
                "exact_error_over_full_output_linf"):
        output[key] = q(quality[key]) if quality[key] else None
    return output


def main():
    import numpy as np
    manifest = verify_inputs()
    records = load_records()
    weights = load_weights()
    statics = [layer_static(weight) for weight in weights]

    cell_acc = {}
    cell_states = {}
    cell_quality = {}
    for sequence in SEQUENCES:
        for layer in range(4):
            key = (sequence, layer)
            cell_acc[key] = empty_accumulator()
            cell_states[key] = np.zeros((statics[layer]["gblocks"],
                                         statics[layer]["oblocks"]), dtype=np.uint8)
            cell_quality[key] = {
                "sampled_output_tiles": 0,
                "near_zero_full_output_count": 0,
                "exact_error_over_activity_bound": [],
                "certified_debt_over_activity_bound": [],
                "exact_error_over_certified_debt": [],
                "exact_error_over_full_output_linf": [],
            }

    for ordinal, record in enumerate(records):
        key = (record["sequence"], record["layer"])
        analyze_record(record, weights[record["layer"]], statics[record["layer"]],
                       cell_acc[key], cell_states[key], cell_quality[key])
        print("record {}/120 {} D{}".format(ordinal + 1, key[0], key[1]), flush=True)

    cells = []
    for sequence in SEQUENCES:
        for layer in range(4):
            key = (sequence, layer)
            states = cell_states[key]
            cells.append({
                "sequence": sequence,
                "layer": layer,
                "calls": 10,
                "sampled_destinations": 10 * DESTINATIONS_PER_CALL,
                "epsilon_rows": summarize_accumulator(cell_acc[key]),
                "dynamic_keep_drop_witness": {
                    "epsilon": DYNAMIC_EPSILON,
                    "static_blocks": int(states.size),
                    "both_keep_and_drop": int((states == 3).sum()),
                    "always_drop": int((states == 1).sum()),
                    "always_keep": int((states == 2).sum()),
                    "unobserved": int((states == 0).sum()),
                },
                "quality_proxy_epsilon_0p1": summarize_quality(cell_quality[key]),
            })

    layer_rows = []
    for layer in range(4):
        selected = [cell_acc[(sequence, layer)] for sequence in SEQUENCES]
        states = np.zeros_like(cell_states[(SEQUENCES[0], layer)])
        quality = {
            "sampled_output_tiles": 0, "near_zero_full_output_count": 0,
            "exact_error_over_activity_bound": [],
            "certified_debt_over_activity_bound": [],
            "exact_error_over_certified_debt": [],
            "exact_error_over_full_output_linf": [],
        }
        for sequence in SEQUENCES:
            states |= cell_states[(sequence, layer)]
            source = cell_quality[(sequence, layer)]
            for field in quality:
                if isinstance(quality[field], list):
                    quality[field].extend(source[field])
                else:
                    quality[field] += source[field]
        static = statics[layer]
        layer_rows.append({
            "layer": layer,
            "geometry": {"cin": GEOMETRY[layer][0], "cout": GEOMETRY[layer][1],
                         "hin": GEOMETRY[layer][2], "win": GEOMETRY[layer][3],
                         "hout": GEOMETRY[layer][4], "wout": GEOMETRY[layer][5]},
            "static_metadata": {
                "bytes": static["metadata_bytes"],
                "hypothetical_packed_int8_weight_bytes": static["int8_weight_bytes"],
                "metadata_to_weight_ratio": static["metadata_bytes"] / static["int8_weight_bytes"],
                "old_g11_bytes": static["old_g11_bytes"],
                "reduction_vs_old_g11": static["old_g11_bytes"] / static["metadata_bytes"],
            },
            "epsilon_rows": summarize_accumulator(merge_accumulators(selected)),
            "dynamic_keep_drop_witness": {
                "epsilon": DYNAMIC_EPSILON,
                "static_blocks": int(states.size),
                "both_keep_and_drop": int((states == 3).sum()),
            },
            "quality_proxy_epsilon_0p1": summarize_quality(quality),
        })

    global_acc = merge_accumulators(list(cell_acc.values()))
    global_eps = summarize_accumulator(global_acc)
    global_dynamic = [row for row in global_eps if row["epsilon"] == DYNAMIC_EPSILON][0]
    metadata_bytes = sum(row["metadata_bytes"] for row in statics)
    int8_weight_bytes = sum(row["int8_weight_bytes"] for row in statics)
    old_g11_bytes = sum(row["old_g11_bytes"] for row in statics)
    all_quality = {
        "sampled_output_tiles": 0, "near_zero_full_output_count": 0,
        "exact_error_over_activity_bound": [],
        "certified_debt_over_activity_bound": [],
        "exact_error_over_certified_debt": [],
        "exact_error_over_full_output_linf": [],
    }
    for source in cell_quality.values():
        for field in all_quality:
            if isinstance(all_quality[field], list):
                all_quality[field].extend(source[field])
            else:
                all_quality[field] += source[field]

    metadata_ratio = metadata_bytes / int8_weight_bytes
    metadata_reduction = old_g11_bytes / metadata_bytes
    gates = {
        "activity_reference_is_destination_output_tile_owned": True,
        "all_30_decoder_samples_consumed": True,
        "three_sequences_x_ten_samples_x_four_layers": len(records) == 120,
        "metadata_le_2pct_hypothetical_int8_weight_bytes": metadata_ratio <= 0.02,
        "metadata_reduction_vs_old_g11_ge_8x": metadata_reduction >= 8.0,
        "epsilon_0p1_potential_weight_byte_suppression_ge_20pct":
            global_dynamic["potential_weight_byte_suppression_fraction"] >= 0.20,
        "epsilon_0p1_certified_debt_max_le_0p1":
            global_dynamic["per_destination_output_tile_normalized_debt"]["max"] <= 0.100000000001,
        "sampled_exact_error_within_certified_debt":
            all_quality["exact_error_over_certified_debt"] and
            max(all_quality["exact_error_over_certified_debt"]) <= 1.000002,
        "paired_aee_available": False,
        "address_timed_bank_burst_suppression_available": False,
    }
    retention_gates = [key for key in gates if key not in
                       ("paired_aee_available", "address_timed_bank_burst_suppression_available")]
    retained = all(gates[key] for key in retention_gates)
    status = ("PASS_ACTIVITY_RELATIVE_FASTKILL__CONDITIONAL_RETAIN_FOR_PAIRED_AEE_AND_ADDRESS_TIMED_REPLAY__NO_RTL_OR_PERFORMANCE"
              if retained else
              "NO_GO_ACTIVITY_RELATIVE_FASTKILL__NO_RTL_OR_PERFORMANCE")

    result = {
        "schema": "m1575_m1458_ep34_live93_s2_ccbs16_activity_relative_fastkill_r1_v1",
        "status": status,
        "python": sys.version.split()[0],
        "identity": {
            "checkpoint_sha256": EXPECTED[CHECKPOINT],
            "capture_manifest_sha256": EXPECTED[MANIFEST],
            "capture_ordered_sha256": EXPECTED[ORDERED],
            "capture_sha256s_sha256": EXPECTED[CAPTURE_SUMS],
            "capture_outer_sha256": EXPECTED[CAPTURE_OUTER],
            "m1512_capture_content_hammer_sha256": EXPECTED[M1512],
            "m1513_production_provenance_sha256": EXPECTED[M1513],
            "m1555_reference_repair_source_sha256": EXPECTED[M1555],
            "docs359_sha256": EXPECTED[DOCS359],
            "selected_candidate": manifest["identity"]["selection"]["selected"]["candidate_id"],
        },
        "method": {
            "source_group": GROUP,
            "output_tile": OUTPUT_TILE,
            "debt_owner": "destination_x_output_tile",
            "reference": "sum_G M(G,O)*A(G) over all legal spatial contributors/taps",
            "weight_bound": "M(G,O)=max_abs_ep34_FP32_weight over source group, output tile and all K3 taps",
            "drop_order": "fixed ascending source-group order",
            "epsilon_grid": list(EPSILONS),
            "sampled_destinations_per_call": DESTINATIONS_PER_CALL,
            "exact_quality_destinations_per_call": EXACT_DESTINATIONS_PER_CALL,
            "exact_quality_output_tiles_per_destination": EXACT_OUTPUT_TILES_PER_DESTINATION,
            "payload_byte_proxy": "hypothetical packed INT8 static weight block; eligibility only, not measured traffic",
        },
        "population": {
            "sequences": list(SEQUENCES),
            "samples_per_sequence": 10,
            "calls": len(records),
            "layers": 4,
            "sampled_destinations_per_layer": 30 * DESTINATIONS_PER_CALL,
            "decoder_payload_sha_checked_per_call": True,
        },
        "static_metadata_global": {
            "bytes": metadata_bytes,
            "hypothetical_packed_int8_weight_bytes": int8_weight_bytes,
            "metadata_to_weight_ratio": metadata_ratio,
            "old_g11_bytes": old_g11_bytes,
            "reduction_vs_old_g11": metadata_reduction,
        },
        "global_epsilon_rows": global_eps,
        "per_layer": layer_rows,
        "per_sequence_layer": cells,
        "quality_proxy_epsilon_0p1": summarize_quality(all_quality),
        "gates": gates,
        "decision": {
            "offline_candidate": "S2_CCBS16",
            "retained_for_next_measurement": retained,
            "next_required": ["paired same-checkpoint AEE Pareto",
                              "address-timed bank/burst suppression with metadata charged"],
            "rtl_authorized": False,
            "performance_claim_authorized": False,
            "reason": ("activity-relative skip eligibility survives the CPU gate, but paired AEE and executable memory suppression are absent"
                       if retained else
                       "the activity-relative opportunity or metadata gate fails before paired AEE/RTL"),
        },
        "limitations": [
            "The reported byte suppression is only a weight-block eligibility proxy; cache reuse, ports, bursts and cycles are not modeled.",
            "The local certified error is relative to conservative observed active-bound mass, not the exact output magnitude or optical-flow AEE.",
            "No modified forward pass or paired AEE was run; accuracy_admission and paired_aee remain false.",
            "No RTL, VCS, synthesis, power or EDA tool was launched.",
            "Only decoder ConvTranspose inputs are screened; this result is not a whole-network speedup.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }

    result_path = OUT / "result.json"
    review_json_path = OUT / "review.json"
    review_md_path = OUT / "review.md"
    complete_path = OUT / "RUN_COMPLETE.txt"
    for path in (result_path, review_json_path, review_md_path, complete_path,
                 OUT / "SHA256SUMS", OUT / "SHA256SUMS.seal.sha256"):
        require(not path.exists(), "refuse overwrite: " + str(path))
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")

    review = {
        "schema": "m1575_s2_ccbs16_activity_relative_fastkill_review_r1_v1",
        "status": status,
        "verdict": result["decision"],
        "headline_legal": False,
        "paired_aee": False,
        "speedup": False,
        "key_numbers": {
            "epsilon_0_exact_drop_fraction": global_eps[0]["drop_fraction"],
            "epsilon_0p1_drop_fraction": global_dynamic["drop_fraction"],
            "epsilon_0p1_potential_weight_byte_suppression_fraction":
                global_dynamic["potential_weight_byte_suppression_fraction"],
            "epsilon_0p1_aggregate_certified_debt_over_activity_bound_mass":
                global_dynamic["aggregate_certified_debt_over_activity_bound_mass"],
            "metadata_to_weight_ratio": metadata_ratio,
            "metadata_reduction_vs_old_g11": metadata_reduction,
        },
        "required_next": result["decision"]["next_required"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    review_json_path.write_text(json.dumps(review, indent=2, sort_keys=True,
                                           allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# M1575 ep34 live93 S2 CCBS16 activity-relative fast-kill",
        "",
        "Status: **{}**.".format(status),
        "",
        "All 30 decoder samples from the sealed ep34 live93 capture were consumed "
        "(three DSEC sequences, ten samples each, four ConvTranspose layers). The "
        "debt owner is one destination/output-tile and the reference is that owner's "
        "observed active bound mass. This repairs the dense global-capacity denominator "
        "that made the prior 99.2% drop number scientifically unusable.",
        "",
        "## Global 16x16 screen",
        "",
        "| epsilon | keep | drop | drop fraction | weight-byte eligibility | aggregate bound debt |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_eps:
        lines.append("| {:.2f} | {:,} | {:,} | {:.3%} | {:.3%} | {:.3%} |".format(
            row["epsilon"], row["keep"], row["drop"], row["drop_fraction"],
            row["potential_weight_byte_suppression_fraction"],
            row["aggregate_certified_debt_over_activity_bound_mass"]))
    lines += [
        "",
        "Static uint16 directory: **{:,} B**, or **{:.4%}** of the hypothetical "
        "packed INT8 decoder weights; **{:.2f}x** smaller than the old per-source G11 metadata.".format(
            metadata_bytes, metadata_ratio, metadata_reduction),
        "",
        "## Per-sequence/layer at epsilon=0.10",
        "",
        "| sequence | layer | keep | drop | drop fraction | weight-byte eligibility | debt / active bound |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        row = [value for value in cell["epsilon_rows"] if value["epsilon"] == 0.10][0]
        lines.append("| {} | D{} | {:,} | {:,} | {:.3%} | {:.3%} | {:.3%} |".format(
            cell["sequence"], cell["layer"], row["keep"], row["drop"],
            row["drop_fraction"], row["potential_weight_byte_suppression_fraction"],
            row["aggregate_certified_debt_over_activity_bound_mass"]))
    quality_summary = result["quality_proxy_epsilon_0p1"]
    lines += [
        "",
        "## Verdict and red lines",
        "",
        "The candidate is **{}** for only the next two measurements: paired same-checkpoint "
        "AEE and address-timed bank/burst suppression with metadata charged. Exact sampled "
        "local errors stayed inside the certified debt; the primary proxy median/p90 is "
        "{:.3%}/{:.3%} of activity-bound mass. This is not optical-flow accuracy.".format(
            "retained" if retained else "killed",
            quality_summary["exact_error_over_activity_bound"]["median"],
            quality_summary["exact_error_over_activity_bound"]["p90"]),
        "",
        "`paired_aee=false`, `cycles=false`, `traffic=false`, `energy=false`, "
        "`speedup=false`, and `rtl=false`. The weight-byte number is eligibility, not "
        "measured DRAM/SRAM traffic and not a system acceleration.",
        "",
    ]
    review_md_path.write_text("\n".join(lines), encoding="utf-8")
    complete_path.write_text(status + "\n", encoding="utf-8")

    members = [Path(__file__).resolve(), result_path, review_json_path,
               review_md_path, complete_path]
    sums_path = OUT / "SHA256SUMS"
    sums_path.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                  for path in members), encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums_path)), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()

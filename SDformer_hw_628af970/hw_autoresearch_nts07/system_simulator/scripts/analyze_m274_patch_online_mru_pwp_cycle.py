#!/usr/bin/env python3
"""Replay a leakage-free online one-entry PWP memo on ordered patch masks."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


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
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + token)),
        )


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file() and sha256(path) == record["file_sha256"],
            "M51 payload identity drift: " + str(path))
    shape = tuple(int(value) for value in record["input_shape"])
    elements = int(np.prod(shape))
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]),
            "M51 packed byte extent drift")
    bits = np.unpackbits(packed, bitorder="little")[:elements]
    require(int(bits.sum(dtype=np.uint64)) == int(record["active_elements"]),
            "M51 active-bit population drift")
    return bits.reshape(shape).astype(np.uint8, copy=False)


def popcount_lut():
    values = np.arange(65536, dtype="<u2")
    octets = values.view(np.uint8).reshape(65536, 2)
    return np.unpackbits(octets, axis=1).sum(axis=1).astype(np.uint8)


def normalize_shapes(bits, output_shape):
    if bits.ndim == 4:
        bits = bits[:, np.newaxis, :, :, :]
    require(bits.ndim == 5, "expected T(B)CHW patch input")
    if len(output_shape) == 4:
        output_shape = [output_shape[0], 1] + list(output_shape[1:])
    require(len(output_shape) == 5, "expected T(B)CHW patch output")
    t_count, batch, channels, height, width = bits.shape
    out_t, out_b, out_c, out_h, out_w = (
        int(value) for value in output_shape)
    require((t_count, batch, out_c) == (out_t, out_b, 96),
            "patch output geometry drift")
    require(height % out_h == 0 and width % out_w == 0 and
            height // out_h == width // out_w and
            height // out_h in (1, 2) and channels % 16 == 0,
            "patch stride/channel geometry drift")
    return bits, height // out_h, out_h, out_w


def phase_online_mru(sequence, pop_lut):
    populations = pop_lut[sequence]
    bit_ops = int(populations.sum(dtype=np.uint64))
    singleton_ops = int(np.count_nonzero(populations == 1))
    expensive = sequence[populations >= 2]
    expensive_pop = pop_lut[expensive]
    if expensive.size:
        misses = np.empty(expensive.size, dtype=bool)
        misses[0] = True
        misses[1:] = expensive[1:] != expensive[:-1]
        miss_ops = int(expensive_pop[misses].sum(dtype=np.uint64))
        misses_count = int(np.count_nonzero(misses))
        hits = int(expensive.size - misses_count)
    else:
        miss_ops = 0
        misses_count = 0
        hits = 0
    memo_ops = singleton_ops + miss_ops + hits
    require(0 <= memo_ops <= bit_ops and
            hits + misses_count == int(expensive.size),
            "online memo work conservation drift")
    return {
        "rows": int(sequence.size),
        "bit_sparse_ops": bit_ops,
        "memo_ops": memo_ops,
        "eligible_rows": int(expensive.size),
        "memo_hits": hits,
        "memo_misses": misses_count,
        "singleton_rows": singleton_ops,
        "zero_rows": int(np.count_nonzero(populations == 0)),
    }


def replay_record(bits, output_shape, pop_lut):
    bits, stride, out_h, out_w = normalize_shapes(bits, output_shape)
    t_count, batch, channels, height, width = bits.shape
    padded = np.pad(bits, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")
    phases = []
    source_contributions = 0
    for kernel_y in range(3):
        for kernel_x in range(3):
            sampled = padded[:, :, :,
                kernel_y:kernel_y + stride * out_h:stride,
                kernel_x:kernel_x + stride * out_w:stride]
            require(sampled.shape ==
                    (t_count, batch, channels, out_h, out_w),
                    "receptive-field sample geometry drift")
            source_contributions += int(sampled.sum(dtype=np.uint64))
            channel_last = np.moveaxis(sampled, 2, -1)
            packed = np.packbits(channel_last, axis=-1, bitorder="little")
            require(packed.shape[-1] == channels // 8,
                    "packed source group extent drift")
            masks = (packed[..., 0::2].astype(np.uint16) |
                     (packed[..., 1::2].astype(np.uint16) << 8))
            for group in range(channels // 16):
                phases.append(phase_online_mru(
                    np.ascontiguousarray(masks[..., group]).reshape(-1),
                    pop_lut,
                ))

    require(len(phases) == 9 * channels // 16,
            "patch phase population drift")
    input_vectors = int(t_count * batch * height * width)
    output_tokens = int(t_count * batch * out_h * out_w)
    bit_cycles = 48
    memo_cycles = 48
    bit_bindings = Counter()
    memo_bindings = Counter()
    for index, phase in enumerate(phases):
        next_load = 48 if index + 1 < len(phases) else 0
        bit_service, bit_binding = max(
            (phase["bit_sparse_ops"], "compute"),
            (next_load, "weight_load"),
        )
        memo_service, memo_binding = max(
            (phase["memo_ops"], "compute"),
            (phase["eligible_rows"] + 2, "lookup"),
            (next_load, "weight_load"),
        )
        bit_cycles += bit_service + 2
        memo_cycles += memo_service + 2
        bit_bindings[bit_binding] += 1
        memo_bindings[memo_binding] += 1
    totals = Counter()
    for phase in phases:
        totals.update(phase)
    bit_total = input_vectors + bit_cycles + output_tokens
    memo_total = input_vectors + memo_cycles + output_tokens
    return {
        "stride": stride,
        "input_channels": channels,
        "partitions": len(phases),
        "input_scan_cycles": input_vectors,
        "output_commit_cycles": output_tokens,
        "valid_receptive_field_source_contributions": source_contributions,
        "bit_sparse_vector_ops": totals["bit_sparse_ops"],
        "memo_vector_ops": totals["memo_ops"],
        "eligible_rows": totals["eligible_rows"],
        "memo_hits": totals["memo_hits"],
        "memo_misses": totals["memo_misses"],
        "singleton_rows": totals["singleton_rows"],
        "zero_rows": totals["zero_rows"],
        "eligible_hit_rate": (totals["memo_hits"] /
                              float(totals["eligible_rows"])
                              if totals["eligible_rows"] else 0.0),
        "bit_sparse_cycles": bit_total,
        "online_mru_cycles": memo_total,
        "speedup_vs_bit_sparse": bit_total / float(memo_total),
        "binding_phases": {
            "bit_sparse": dict(sorted(bit_bindings.items())),
            "online_mru": dict(sorted(memo_bindings.items())),
        },
    }


def summarize(rows):
    result = []
    for index in sorted(rows):
        row = rows[index]
        result.append({
            "index": index,
            "bit_sparse_cycles": row["bit_sparse_cycles"],
            "online_mru_cycles": row["online_mru_cycles"],
            "speedup_vs_bit_sparse": (
                row["bit_sparse_cycles"] /
                float(row["online_mru_cycles"])
            ),
            "eligible_rows": row["eligible_rows"],
            "memo_hits": row["memo_hits"],
            "eligible_hit_rate": (
                row["memo_hits"] / float(row["eligible_rows"])
                if row["eligible_rows"] else 0.0
            ),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_start = sha256(Path(__file__).resolve())
    contract = strict_json(args.contract)
    require(contract["schema"] ==
            "m274_patch_online_mru_pwp_cycle_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "frozen input SHA drift {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    manifest = strict_json(paths["m51_manifest"])
    m222 = strict_json(paths["m222_patch_reference"])
    require(manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "M51 trace admission drift")
    selected = set(contract["source_geometry"][
        "selected_m51_module_indices"])
    records = [record for record in manifest["records"]
               if int(record["module_index"]) in selected]
    require(len(records) == 60, "selected patch record population drift")
    m222_rows = {(int(row["sample_id"]), int(row["module_index"])): row
                 for row in m222["per_record"]}
    pop_lut = popcount_lut()
    payload_root = paths["m51_manifest"].parent
    per_record = []
    aggregate = Counter()
    per_sample = defaultdict(Counter)
    per_module = defaultdict(Counter)
    fields = (
        "valid_receptive_field_source_contributions",
        "bit_sparse_vector_ops", "memo_vector_ops", "eligible_rows",
        "memo_hits", "memo_misses", "singleton_rows", "zero_rows",
        "bit_sparse_cycles", "online_mru_cycles",
    )
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        replay = replay_record(decode_record(record, payload_root),
                               record["output_shape"], pop_lut)
        key = (int(record["sample_id"]), int(record["module_index"]))
        require(key in m222_rows and
                replay["valid_receptive_field_source_contributions"] ==
                int(m222_rows[key][
                    "valid_receptive_field_source_contributions"]),
                "M222 receptive-field source conservation drift")
        replay.update({
            "ordinal": ordinal,
            "sample_id": key[0],
            "module_index": key[1],
            "name": record["name"],
            "payload_sha256": record["file_sha256"],
        })
        per_record.append(replay)
        for field in fields:
            aggregate[field] += replay[field]
            per_sample[key[0]][field] += replay[field]
            per_module[key[1]][field] += replay[field]
        print("[M274] {}/60 sample={} module={} hit={:.6f} speed={:.6f}".format(
            ordinal + 1, key[0], key[1], replay["eligible_hit_rate"],
            replay["speedup_vs_bit_sparse"]), flush=True)

    sample_rows = summarize(per_sample)
    module_rows = summarize(per_module)
    speedup = (aggregate["bit_sparse_cycles"] /
               float(aggregate["online_mru_cycles"]))
    hit_rate = (aggregate["memo_hits"] /
                float(aggregate["eligible_rows"]))
    all_faster = (all(row["speedup_vs_bit_sparse"] > 1.0
                      for row in per_record) and
                  all(row["speedup_vs_bit_sparse"] > 1.0
                      for row in sample_rows) and
                  all(row["speedup_vs_bit_sparse"] > 1.0
                      for row in module_rows))
    rtl_candidate = all_faster and speedup >= 1.5
    output = {
        "schema": "m274_patch_online_mru_pwp_cycle_v1",
        "status": ("PASS_ONLINE_MRU_RTL_PROMOTION_CANDIDATE"
                   if rtl_candidate else
                   "PASS_ONLINE_MRU_TRACE_SCREEN_NO_RTL_PROMOTION"),
        "identity": identities,
        "scope": {
            "samples": 10,
            "modules": 6,
            "records": 60,
            "source_partition": contract["source_geometry"][
                "source_partition"],
            "output_lanes": 96,
        },
        "exact_ordered_work": {
            "valid_receptive_field_source_contributions":
                aggregate["valid_receptive_field_source_contributions"],
            "bit_sparse_vector_ops": aggregate["bit_sparse_vector_ops"],
            "online_mru_vector_ops": aggregate["memo_vector_ops"],
            "natural_vector_op_speedup": (
                aggregate["bit_sparse_vector_ops"] /
                float(aggregate["memo_vector_ops"])
            ),
            "eligible_rows": aggregate["eligible_rows"],
            "memo_hits": aggregate["memo_hits"],
            "memo_misses": aggregate["memo_misses"],
            "eligible_hit_rate": hit_rate,
        },
        "same_resource_module_cycles": {
            "bit_sparse": aggregate["bit_sparse_cycles"],
            "online_mru": aggregate["online_mru_cycles"],
            "speedup_vs_bit_sparse": speedup,
            "all_60_records_all_10_samples_all_6_modules_faster": all_faster,
            "rtl_promotion_gate_1p5x": rtl_candidate,
        },
        "hardware_state": contract["online_memo_policy"],
        "per_sample": sample_rows,
        "per_module": module_rows,
        "per_record": per_record,
        "admission": {
            "exact_ordered_trace": True,
            "training_or_profile_selection": False,
            "isolated_patch_module_cycles": all_faster,
            "rtl_candidate": rtl_candidate,
            "pwp_numeric_rtl": False,
            "vcs": False,
            "dc": False,
            "cache_macro": False,
            "builder_area": False,
            "complete_patch_embed": False,
            "energy": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M274 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = (
        args.output_dir / "m274_patch_online_mru_pwp_cycle_r1.json"
    )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M274_PASS hit={:.6f} speed={:.6f} rtl_candidate={}".format(
        hit_rate, speedup, rtl_candidate), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent streaming reconstruction of the M517 FC2 density evidence.

This checker intentionally does not import or execute the production analyzer.
It derives the FC2 population from the sealed M51 manifest, streams the sealed
tar.zst, verifies each selected member, and reconstructs the relevant service
floors and distributions from the raw little-endian bit packs.
"""

import argparse
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd


KNOWN = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "archive": "aa261ebe64015bbd295f65f4b734efcb6b26c11c3dd0828e9e7a659433f6c3b4",
    "production": "",  # Filled after production evidence is read; not an input pin.
}
GEOMETRY = {0: (384, 96), 1: (768, 192), 2: (1536, 384), 3: (3072, 768)}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path):
    def reject_duplicate(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicate)


def blank():
    return {
        "records": 0,
        "tokens": 0,
        "tiles": 0,
        "events": 0,
        "full_floor": 0,
        "tile_floor": 0,
        "dense_floor": 0,
        "nnz_hist": np.zeros(97, dtype=np.int64),
        "bankmax_hist": np.zeros(13, dtype=np.int64),
    }


def add(dst, src):
    for key in ("records", "tokens", "tiles", "events", "full_floor",
                "tile_floor", "dense_floor"):
        dst[key] += int(src[key])
    dst["nnz_hist"] += src["nnz_hist"]
    dst["bankmax_hist"] += src["bankmax_hist"]


def quantile(hist, q):
    target = math.ceil(int(hist.sum()) * q)
    cumulative = 0
    for value, count in enumerate(hist.tolist()):
        cumulative += count
        if cumulative >= target:
            return value
    raise AssertionError("empty histogram")


def summarize(x):
    nonzero_nnz = np.flatnonzero(x["nnz_hist"])
    nonzero_bank = np.flatnonzero(x["bankmax_hist"])
    return {
        "records": x["records"],
        "tokens": x["tokens"],
        "tiles": x["tiles"],
        "events": x["events"],
        "empty_tiles": int(x["nnz_hist"][0]),
        "empty_fraction": float(x["nnz_hist"][0] / x["tiles"]),
        "full_vector_floor_cycles": x["full_floor"],
        "tile_partition_floor_cycles": x["tile_floor"],
        "dense_cycles": x["dense_floor"],
        "nnz": {
            "p50": quantile(x["nnz_hist"], 0.50),
            "p90": quantile(x["nnz_hist"], 0.90),
            "p95": quantile(x["nnz_hist"], 0.95),
            "p99": quantile(x["nnz_hist"], 0.99),
            "p99p9": quantile(x["nnz_hist"], 0.999),
            "max": int(nonzero_nnz[-1]),
        },
        "max_bank": {
            "p50": quantile(x["bankmax_hist"], 0.50),
            "p95": quantile(x["bankmax_hist"], 0.95),
            "p99": quantile(x["bankmax_hist"], 0.99),
            "p99p9": quantile(x["bankmax_hist"], 0.999),
            "max": int(nonzero_bank[-1]),
        },
        "tiles_ge_24": int(x["nnz_hist"][24:].sum()),
        "tiles_ge_48": int(x["nnz_hist"][48:].sum()),
        "tiles_ge_54": int(x["nnz_hist"][54:].sum()),
        "tiles_ge_72": int(x["nnz_hist"][72:].sum()),
        "dense_sparse_zero_tax_ties": int(x["bankmax_hist"][12]),
        "nnz_histogram": x["nnz_hist"].tolist(),
        "max_bank_histogram": x["bankmax_hist"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite output")

    identities = {
        "manifest": digest_file(args.manifest),
        "archive": digest_file(args.archive),
        "production": digest_file(args.production),
    }
    if identities["manifest"] != KNOWN["manifest"]:
        raise RuntimeError("manifest identity mismatch")
    if identities["archive"] != KNOWN["archive"]:
        raise RuntimeError("archive identity mismatch")

    manifest = load_json(args.manifest)
    production = load_json(args.production)
    chosen = {}
    module_names = set()
    sample_ids = set()
    for row in manifest["records"]:
        if row.get("operator") != "Linear" or ".mlp.fc2" not in row.get("name", ""):
            continue
        rel = row["relative_path"]
        if rel in chosen:
            raise RuntimeError("duplicate relative path in manifest: " + rel)
        chosen[rel] = row
        module_names.add(row["name"])
        sample_ids.add(int(row["sample_id"]))
    if len(chosen) != 120 or len(module_names) != 12 or sample_ids != set(range(10)):
        raise RuntimeError("FC2 population mismatch")

    aggregate = blank()
    by_sample = defaultdict(blank)
    by_stage = defaultdict(blank)
    seen = set()
    sha_pass = 0
    pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

    with args.archive.open("rb") as compressed:
        decompressed = zstd.ZstdDecompressor().stream_reader(compressed)
        with tarfile.open(fileobj=decompressed, mode="r|") as stream:
            for member in stream:
                rel = "/".join(member.name.split("/")[-2:])
                if rel not in chosen:
                    continue
                if rel in seen:
                    raise RuntimeError("duplicate archive member: " + rel)
                row = chosen[rel]
                fh = stream.extractfile(member)
                if fh is None:
                    raise RuntimeError("member is not readable: " + rel)
                raw = fh.read()
                if len(raw) != int(row["packed_bytes"]):
                    raise RuntimeError("extent mismatch: " + rel)
                if digest_bytes(raw) != row["file_sha256"]:
                    raise RuntimeError("payload digest mismatch: " + rel)
                sha_pass += 1

                stage = int(row["name"].split(".layers.", 1)[1].split(".", 1)[0])
                cin, cout = GEOMETRY[stage]
                shape = [int(v) for v in row["input_shape"]]
                outshape = [int(v) for v in row["output_shape"]]
                if shape[-1] != cin or outshape[-1] != cout or shape[:-1] != outshape[:-1]:
                    raise RuntimeError("geometry mismatch: " + rel)
                tokens = 1
                for extent in shape[:-1]:
                    tokens *= extent
                output_blocks = cout // 96
                byte_matrix = np.frombuffer(raw, dtype=np.uint8).reshape(tokens, cin // 8)

                # A bank is one bit position in each byte. This is derived from
                # the manifest's little-endian channel packing, independent of
                # the production implementation.
                full_counts = np.zeros((tokens, 8), dtype=np.int32)
                for bank in range(8):
                    full_counts[:, bank] = ((byte_matrix & (1 << bank)) != 0).sum(axis=1)
                tiles = byte_matrix.reshape(-1, 12)
                nnz = pop[tiles].sum(axis=1).astype(np.int16)
                tile_banks = np.zeros((tiles.shape[0], 8), dtype=np.int16)
                for bank in range(8):
                    tile_banks[:, bank] = ((tiles & (1 << bank)) != 0).sum(axis=1)
                bankmax = tile_banks.max(axis=1)

                item = blank()
                item["records"] = 1
                item["tokens"] = tokens
                item["tiles"] = int(tiles.shape[0])
                item["events"] = int(nnz.sum())
                item["full_floor"] = int(full_counts.max(axis=1).sum()) * output_blocks
                item["tile_floor"] = int(bankmax.sum()) * output_blocks
                item["dense_floor"] = int(tiles.shape[0]) * 12 * output_blocks
                item["nnz_hist"] = np.bincount(nnz, minlength=97).astype(np.int64)
                item["bankmax_hist"] = np.bincount(bankmax, minlength=13).astype(np.int64)
                if item["events"] != int(row["active_elements"]):
                    raise RuntimeError("active count mismatch: " + rel)
                if np.any(bankmax > 12) or np.any(nnz > 96):
                    raise RuntimeError("tile capacity violation")
                add(aggregate, item)
                add(by_sample[int(row["sample_id"])], item)
                add(by_stage[stage], item)
                seen.add(rel)

    if seen != set(chosen):
        raise RuntimeError("archive population incomplete")
    rebuilt = summarize(aggregate)
    sample_summary = {str(k): summarize(v) for k, v in sorted(by_sample.items())}
    stage_summary = {str(k): summarize(v) for k, v in sorted(by_stage.items())}

    required = {
        "records": 120,
        "tokens": 5_580_000,
        "tiles": 36_480_000,
        "events": 143_894_510,
        "full_vector_floor_cycles": 70_657_362,
        "tile_partition_floor_cycles": 118_651_292,
        "dense_cycles": 1_105_920_000,
        "tiles_ge_48": 1_922,
    }
    exact_checks = {key: rebuilt[key] == expected for key, expected in required.items()}
    exact_checks["maximum_nnz_55"] = rebuilt["nnz"]["max"] == 55
    exact_checks["payload_sha_120"] = sha_pass == 120

    prod = production["aggregate"]
    production_checks = {
        "nnz_histogram": rebuilt["nnz_histogram"] == prod["nnz_histogram_0_to_96"],
        "max_bank_histogram": rebuilt["max_bank_histogram"] == prod["max_bank_histogram_0_to_12"],
        "per_sample_events": all(
            sample_summary[str(k)]["events"] == production["per_sample"][str(k)]["events"]
            for k in range(10)),
        "per_sample_full_floor": all(
            sample_summary[str(k)]["full_vector_floor_cycles"] == production["per_sample"][str(k)]["full_vector_sparse_service_floor_cycles"]
            for k in range(10)),
        "per_sample_tile_floor": all(
            sample_summary[str(k)]["tile_partition_floor_cycles"] == production["per_sample"][str(k)]["tile_partition_sparse_service_floor_cycles"]
            for k in range(10)),
        "per_sample_high50": all(
            sample_summary[str(k)]["tiles_ge_48"] == production["per_sample"][str(k)]["thresholds"]["48"]["tiles"]
            for k in range(10)),
    }

    ev = np.array([sample_summary[str(k)]["events"] for k in range(10)], dtype=np.float64)
    empty = np.array([sample_summary[str(k)]["empty_fraction"] for k in range(10)], dtype=np.float64)
    tile_floor = np.array([sample_summary[str(k)]["tile_partition_floor_cycles"] for k in range(10)], dtype=np.float64)
    high50 = np.array([sample_summary[str(k)]["tiles_ge_48"] for k in range(10)], dtype=np.float64)
    stability = {
        "events_min": int(ev.min()), "events_max": int(ev.max()),
        "events_cv": float(ev.std(ddof=0) / ev.mean()),
        "empty_fraction_min": float(empty.min()), "empty_fraction_max": float(empty.max()),
        "tile_floor_min": int(tile_floor.min()), "tile_floor_max": int(tile_floor.max()),
        "tile_floor_cv": float(tile_floor.std(ddof=0) / tile_floor.mean()),
        "high50_min": int(high50.min()), "high50_max": int(high50.max()),
        "high50_cv": float(high50.std(ddof=0) / high50.mean()),
    }

    # Zero-tax proof: for every tile, max occupancy of eight 12-entry banks is
    # <= 12. Replacing that tile's sparse issue by a 12-cycle dense issue cannot
    # reduce work. Cross-tile aggregation only strengthens the sparse baseline:
    # max(a+b) <= max(a)+max(b), so segmenting can inflate, never deflate, its floor.
    fairness = {
        "zero_tax_dense_strict_wins": 0,
        "zero_tax_ties": rebuilt["dense_sparse_zero_tax_ties"],
        "tile_floor_over_full_vector_floor": rebuilt["tile_partition_floor_cycles"] / rebuilt["full_vector_floor_cycles"],
        "tile_floor_over_strong_m216_cycles": rebuilt["tile_partition_floor_cycles"] / 90_196_785,
        "proof": "For each tile max(bank_count)<=12=dense. For a full vector max(sum tile bank counts)<=sum max(tile bank counts); cross-tile aggregation can only improve the sparse reference.",
        "parallel_path_boundary": "A physically parallel dense path with independent banks is outside the declared same-eight-bank resource contract and would require matched area/bandwidth comparison.",
    }

    uniform_overhead = (90_196_785 - rebuilt["full_vector_floor_cycles"]) / (rebuilt["tiles"] - rebuilt["empty_tiles"])
    sensitivity = {}
    for overhead in (0.0, 1.0, 2.0, 4.0, 8.0, uniform_overhead):
        routed = 0
        saved = 0.0
        for stage, stage_data in by_stage.items():
            blocks = 1 << stage
            for bankmax, count in enumerate(stage_data["bankmax_hist"].tolist()):
                if bankmax == 0 or count == 0:
                    continue
                difference = bankmax * blocks + overhead - 12 * blocks
                if difference > 0:
                    routed += count
                    saved += difference * count
        sensitivity["{:.12g}".format(overhead)] = {
            "routed_tiles": int(routed),
            "saved_cycles_before_router_format_queue_tax": float(saved),
            "upper_speedup_if_subtracted_from_m216": 90_196_785 / (90_196_785 - saved),
        }

    output = {
        "schema": "m517_independent_reconstruction_v1",
        "status": "PASS" if all(exact_checks.values()) and all(production_checks.values()) else "FAIL",
        "identity": identities,
        "population": {"manifest_fc2_records": len(chosen), "module_names": len(module_names), "sample_ids": sorted(sample_ids), "payload_sha_pass": sha_pass},
        "required_exact_checks": exact_checks,
        "production_cross_checks": production_checks,
        "aggregate": rebuilt,
        "per_sample": sample_summary,
        "per_stage": stage_summary,
        "per_sample_stability": stability,
        "fairness_analysis": fairness,
        "uniform_overhead_challenge": {
            "observed_m216_minus_full_floor": 90_196_785 - rebuilt["full_vector_floor_cycles"],
            "nonzero_tiles": rebuilt["tiles"] - rebuilt["empty_tiles"],
            "cycles_per_nonzero_tile_if_uniform": uniform_overhead,
            "independent_sensitivity": sensitivity,
            "interpretation": "This attribution is not exact per-tile latency and cannot admit speedup; it is only an optimistic sensitivity. KILL does not depend on it because zero-tax dense never wins.",
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "exact": exact_checks, "production": production_checks}, sort_keys=True))


if __name__ == "__main__":
    main()

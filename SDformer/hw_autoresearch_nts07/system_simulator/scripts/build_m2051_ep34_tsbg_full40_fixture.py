#!/opt/anaconda3/bin/python
"""Build a performance-independent ep34 multilayer/token TSBG RTL fixture.

The fixed selection uses all forty captured samples from four DSEC sequences,
every FC1/FC2 layer supported by the existing G48 frontend, and the
first/middle/last aligned quartet of tokens.  Smaller layers are zero-padded to
the same physical G48 engine.  No cycle result is inspected or produced here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901"
)
M1558 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py"
)
OUT = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
STATS = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh"
META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"

M1558_SHA = "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089"
CAPTURE_MANIFEST_SHA = "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f"
CAPTURE_OUTER_SHA = "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85"
FC_FRAMES_SHA = "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1"
LAYERS_SHA = "bd40c213f075ea3198f7145d25e9c96988701f46d5572c1e40d36e008feab08a"
SAMPLE_ORDER_SHA = "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773"
TARGET_SAMPLES = tuple(range(40))
CONTEXTS = 4
MAX_GROUPS = 48
SOURCES = 16
SLICES = 6
CACHE_ROWS = 4
STAT_FIELDS = (
    "sample_id", "layer_id", "is_fc2", "token_start", "source_groups",
    "live_rows", "issues", "products", "base_misses", "base_hits",
    "base_evictions", "tsbg_misses", "tsbg_hits", "tsbg_evictions",
)


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module():
    need(sha256(M1558) == M1558_SHA, "M1558 source SHA drift")
    spec = importlib.util.spec_from_file_location("m2051_exact_m1558", M1558)
    need(spec is not None and spec.loader is not None, "cannot import M1558")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(sha256(M1558) == M1558_SHA, "M1558 source changed during import")
    return module


def verify_capture() -> None:
    manifest = CAPTURE / "SHA256SUMS"
    outer = CAPTURE / "SHA256SUMS.seal.sha256"
    need(sha256(manifest) == CAPTURE_MANIFEST_SHA, "capture manifest drift")
    need(sha256(outer) == CAPTURE_OUTER_SHA, "capture outer seal drift")
    need(outer.read_text(encoding="ascii").split() ==
         [CAPTURE_MANIFEST_SHA, "SHA256SUMS"], "capture outer contents drift")
    rows = dict(
        (name.strip().lstrip("*"), digest)
        for digest, name in (
            line.split(None, 1)
            for line in manifest.read_text(encoding="ascii").splitlines()
        )
    )
    expected = {
        "fc_frames.bin": FC_FRAMES_SHA,
        "layers.json": LAYERS_SHA,
        "sample_order.json": SAMPLE_ORDER_SHA,
    }
    for name, digest in expected.items():
        need(rows.get(name) == digest, f"{name} manifest identity drift")
        need(sha256(CAPTURE / name) == digest, f"{name} payload SHA drift")


def token_starts(tokens: int) -> tuple[int, int, int]:
    need(tokens >= 12 and tokens % CONTEXTS == 0, "token geometry drift")
    middle = (tokens // 2 // CONTEXTS) * CONTEXTS
    result = (0, middle, tokens - CONTEXTS)
    need(len(set(result)) == 3 and all(value % CONTEXTS == 0 for value in result),
         "quartet selection drift")
    return result


def select_inventory():
    layers_payload = json.loads((CAPTURE / "layers.json").read_text())
    samples_payload = json.loads((CAPTURE / "sample_order.json").read_text())
    layers = [row for row in layers_payload["layers"]
              if row["target"] in ("FC1", "FC2") and
              int(row["weight_layout"]["source_group_count"]) <= MAX_GROUPS]
    need(len(layers) == 16, "supported FC inventory cardinality drift")
    need(sum(row["target"] == "FC1" for row in layers) == 12 and
         sum(row["target"] == "FC2" for row in layers) == 4,
         "supported FC1/FC2 split drift")
    need([int(row["layer_id"]) for row in layers] ==
         [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30],
         "supported layer identity drift")
    sample_by_id = {int(row["global_sample_id"]): row
                    for row in samples_payload["samples"]}
    samples = [sample_by_id[value] for value in TARGET_SAMPLES]
    need([int(row["global_sample_id"]) for row in samples] == list(range(40)),
         "full capture sample order drift")
    sequence_counts = {}
    for row in samples:
        sequence_counts[row["sequence"]] = sequence_counts.get(row["sequence"], 0) + 1
    need(len(sequence_counts) == 4 and set(sequence_counts.values()) == {10},
         "four-sequence ten-sample cohort drift")
    return layers, samples


def extract(module, layers):
    layer_by_id = {int(row["layer_id"]): row for row in layers}
    wanted = {}
    for sample_id in TARGET_SAMPLES:
        for layer in layers:
            key = (sample_id, int(layer["layer_id"]))
            wanted[key] = token_starts(int(layer["tokens_per_call"]))
    found = {}
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while True:
            prefix = stream.read(module.FRAME_HEADER.size)
            if not prefix:
                break
            need(len(prefix) == module.FRAME_HEADER.size, "truncated frame header")
            values = module.FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             frame_start, token_count, channels, bitrow, nnz_total,
             raw_bytes, compressed_bytes, crc32) = values
            need(magic == module.FRAME_MAGIC and
                 version == module.FRAME_VERSION and
                 header_size == module.FRAME_HEADER.size,
                 "frame header identity drift")
            compressed = stream.read(compressed_bytes)
            need(len(compressed) == compressed_bytes, "truncated compressed frame")
            key = (int(sample_id), int(layer_id))
            if key not in wanted:
                continue
            selected = [start for start in wanted[key]
                        if int(frame_start) <= start and
                        start + CONTEXTS <= int(frame_start) + int(token_count)]
            if not selected:
                continue
            layer = layer_by_id[int(layer_id)]
            need(int(channels) == int(layer["input_channels"]) and
                 int(channels) ==
                 int(layer["weight_layout"]["source_group_count"]) * SOURCES,
                 "target layer channel geometry drift")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            need(decoder.eof and not decoder.unused_data and
                 not decoder.unconsumed_tail and len(raw) == raw_bytes and
                 (zlib.crc32(raw) & 0xFFFFFFFF) == crc32,
                 "target frame decompression drift")
            decoded = module.decode_frame_payload(
                raw, token_count, channels, bitrow, nnz_total, return_codes=True
            )
            codes = np.asarray(decoded["codes"], dtype=np.int8)
            need(set(int(value) for value in np.unique(codes)).issubset({-1, 0, 1}),
                 "descriptor code outside {-1,0,+1}")
            for start in selected:
                offset = start - int(frame_start)
                quartet = codes[offset:offset + CONTEXTS].copy()
                need(quartet.shape == (CONTEXTS, channels),
                     "selected quartet extent drift")
                out_key = (int(sample_id), int(layer_id), start)
                need(out_key not in found, "duplicate selected quartet")
                found[out_key] = quartet
    need(len(found) == len(TARGET_SAMPLES) * len(layers) * 3,
         "selected quartet population incomplete")
    return found


def cache_counts(accesses):
    # Match the RTL's nonblocking age semantics exactly.  A hit records the
    # pre-increment access_clock, while a completed miss fill records the
    # already-incremented clock.  This can create equal-age entries; the RTL
    # then breaks a victim tie by the lowest physical cache-row index.  A
    # conventional ordered-list LRU model is therefore not exact.
    valid = [False] * CACHE_ROWS
    group_at = [0] * CACHE_ROWS
    age = [0] * CACHE_ROWS
    access_clock = 0
    hits = misses = evictions = 0
    for group in accesses:
        hit = next((index for index in range(CACHE_ROWS)
                    if valid[index] and group_at[index] == group), None)
        if hit is not None:
            hits += 1
            age[hit] = access_clock
        else:
            misses += 1
            invalid = next((index for index in range(CACHE_ROWS)
                            if not valid[index]), None)
            if invalid is None:
                victim = min(range(CACHE_ROWS), key=lambda index: (age[index], index))
                evictions += 1
            else:
                victim = invalid
            group_at[victim] = group
            valid[victim] = True
            age[victim] = access_clock + 1
        access_clock += 1
    return misses, hits, evictions


def describe(values, groups):
    active = values != 0
    live = active.any(axis=2)
    base_accesses = [group for context in range(CONTEXTS)
                     for group in range(groups) if live[context, group]]
    tsbg_accesses = [group for group in range(groups)
                     for context in range(CONTEXTS) if live[context, group]]
    base = cache_counts(base_accesses)
    tsbg = cache_counts(tsbg_accesses)
    half_live = active.reshape(CONTEXTS, MAX_GROUPS, 2, 8).any(axis=3)
    return {
        "live_rows": int(live.sum()),
        "nonzero_codes": int(active.sum()),
        "negative_codes": int((values < 0).sum()),
        "issues": int(half_live.sum()) * SLICES,
        "products": int(active.sum()) * SLICES * 16,
        "base_misses": base[0], "base_hits": base[1],
        "base_evictions": base[2],
        "tsbg_misses": tsbg[0], "tsbg_hits": tsbg[1],
        "tsbg_evictions": tsbg[2],
    }


def pack_stats(row) -> str:
    value = 0
    for index, field in enumerate(STAT_FIELDS):
        item = int(row[field])
        need(0 <= item < (1 << 32), f"stat overflow: {field}")
        value |= item << (index * 32)
    return f"{value:0{len(STAT_FIELDS) * 8}x}\n"


def main() -> int:
    verify_capture()
    module = load_module()
    layers, samples = select_inventory()
    found = extract(module, layers)
    fixture_words = []
    stats_words = []
    rows = []
    token_roles = ("first", "middle", "last")
    for sample in samples:
        sample_id = int(sample["global_sample_id"])
        for layer in layers:
            layer_id = int(layer["layer_id"])
            groups = int(layer["weight_layout"]["source_group_count"])
            for role, start in zip(token_roles,
                                   token_starts(int(layer["tokens_per_call"]))):
                raw_values = found[(sample_id, layer_id, start)]
                values = np.zeros((CONTEXTS, MAX_GROUPS, SOURCES), dtype=np.int8)
                values[:, :groups, :] = raw_values.reshape(CONTEXTS, groups, SOURCES)
                row = {
                    "slot": len(rows), "sample_id": sample_id,
                    "sequence": sample["sequence"], "layer_id": layer_id,
                    "target": layer["target"],
                    "is_fc2": int(layer["target"] == "FC2"),
                    "token_role": role, "token_start": start,
                    "source_groups": groups,
                    "physical_source_groups": MAX_GROUPS,
                }
                row.update(describe(values, groups))
                for context in range(CONTEXTS):
                    for group in range(MAX_GROUPS):
                        source = values[context, group]
                        active = sum(1 << lane for lane, code in enumerate(source)
                                     if code != 0)
                        sign = sum(1 << lane for lane, code in enumerate(source)
                                   if code < 0)
                        need(sign & ~active == 0, "sign without activity")
                        fixture_words.append(f"{sign:04x}{active:04x}\n")
                stats_words.append(pack_stats(row))
                rows.append(row)
    need(len(rows) == 1920 and len(fixture_words) == 1920 * CONTEXTS * MAX_GROUPS,
         "fixture cardinality drift")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(fixture_words), encoding="ascii")
    STATS.write_text("".join(stats_words), encoding="ascii")
    unsupported = [int(row["layer_id"]) for row in
                   json.loads((CAPTURE / "layers.json").read_text())["layers"]
                   if row["target"] == "FC2" and
                   int(row["weight_layout"]["source_group_count"]) > MAX_GROUPS]
    metadata = {
        "schema": "m2051_ep34_tsbg_multilayer_token_fixture_r1_v1",
        "selection_rule": (
            "all forty captured samples in four sequences; all FC1/FC2 layers "
            "with source_group_count<=48; first/middle/last aligned B4 quartet"
        ),
        "selection_uses_performance": False,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA,
        "fc_frames_sha256": FC_FRAMES_SHA,
        "layers_sha256": LAYERS_SHA,
        "sample_order_sha256": SAMPLE_ORDER_SHA,
        "m1558_source_sha256": M1558_SHA,
        "geometry": {
            "workloads": 1920, "sequences": 4, "samples": 40,
            "supported_layers": 16, "fc1_layers": 12, "fc2_layers": 4,
            "quartets_per_layer_sample": 3, "contexts": CONTEXTS,
            "physical_source_groups": MAX_GROUPS, "sources_per_group": SOURCES,
        },
        "unsupported_fc2_layer_ids_over_g48": unsupported,
        "stat_fields_lsb_first_u32": list(STAT_FIELDS),
        "rows": rows,
        "fixture_sha256": sha256(OUT),
        "stats_sha256": sha256(STATS),
        "claim_boundary": {
            "real_ep34_activity_and_sign_descriptors": True,
            "same_physical_g48_engine_with_zero_padding": True,
            "hardware_weight_values": False,
            "cycle_result": False,
            "full_fc_population": False,
            "system_speedup": False,
        },
    }
    META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "workloads": len(rows),
        "fixture_sha256": metadata["fixture_sha256"],
        "stats_sha256": metadata["stats_sha256"],
        "metadata_sha256": sha256(META),
        "negative_codes": sum(row["negative_codes"] for row in rows),
        "unsupported_fc2_layers": unsupported,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

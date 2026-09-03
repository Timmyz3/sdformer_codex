#!/opt/anaconda3/bin/python
"""Build the fixed four-sequence ep34 G48 descriptor fixture for TSBG RTL.

The selection rule is fixed before looking at performance: for each captured
sequence, use its first sampled frame, layer 28 (the first 48-source-group FC1),
and tokens 0--3.  Only binary activity/sign descriptors are exported.  The
script does not estimate cycles and cannot select a favorable bundle.
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
OUT = HW / "tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.memh"
META = HW / "tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.json"

M1558_SHA = "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089"
CAPTURE_MANIFEST_SHA = "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f"
CAPTURE_OUTER_SHA = "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85"
FC_FRAMES_SHA = "dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1"
TARGET_LAYER = 28
TARGET_SAMPLES = (0, 10, 20, 30)
CONTEXTS = 4
GROUPS = 48
SOURCES = 16


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture_module():
    need(sha256(M1558) == M1558_SHA, "M1558 source SHA drift")
    spec = importlib.util.spec_from_file_location("m2046_exact_m1558", M1558)
    need(spec is not None and spec.loader is not None, "cannot import M1558")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(sha256(M1558) == M1558_SHA, "M1558 source changed during import")
    return module


def verify_capture() -> None:
    manifest = CAPTURE / "SHA256SUMS"
    outer = CAPTURE / "SHA256SUMS.seal.sha256"
    frames = CAPTURE / "fc_frames.bin"
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
    need(rows.get("fc_frames.bin") == FC_FRAMES_SHA, "frame manifest identity drift")
    need(sha256(frames) == FC_FRAMES_SHA, "frame payload SHA drift")


def extract(module):
    wanted = set((sample, TARGET_LAYER) for sample in TARGET_SAMPLES)
    found = {}
    with (CAPTURE / "fc_frames.bin").open("rb") as stream:
        while wanted - set(found):
            prefix = stream.read(module.FRAME_HEADER.size)
            need(prefix, "target frames missing")
            need(len(prefix) == module.FRAME_HEADER.size, "truncated frame header")
            values = module.FRAME_HEADER.unpack(prefix)
            (magic, version, header_size, layer_id, sample_id, frame_index,
             token_start, token_count, channels, bitrow, nnz_total,
             raw_bytes, compressed_bytes, crc32) = values
            need(magic == module.FRAME_MAGIC and
                 version == module.FRAME_VERSION and
                 header_size == module.FRAME_HEADER.size,
                 "frame header identity drift")
            compressed = stream.read(compressed_bytes)
            need(len(compressed) == compressed_bytes, "truncated compressed frame")
            key = (int(sample_id), int(layer_id))
            if key not in wanted or int(frame_index) != 0:
                continue
            need(int(token_start) == 0 and int(token_count) >= CONTEXTS and
                 int(channels) == GROUPS * SOURCES and
                 int(bitrow) == channels // 8, "target geometry drift")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed) + decoder.flush()
            need(decoder.eof and not decoder.unused_data and
                 not decoder.unconsumed_tail and len(raw) == raw_bytes and
                 (zlib.crc32(raw) & 0xFFFFFFFF) == crc32,
                 "target frame decompression drift")
            decoded = module.decode_frame_payload(
                raw, token_count, channels, bitrow, nnz_total, return_codes=True
            )
            codes = np.asarray(decoded["codes"][:CONTEXTS], dtype=np.int8)
            need(codes.shape == (CONTEXTS, GROUPS * SOURCES), "decoded shape drift")
            need(set(int(value) for value in np.unique(codes)).issubset({-1, 0, 1}),
                 "descriptor code outside {-1,0,+1}")
            found[key] = codes.reshape(CONTEXTS, GROUPS, SOURCES)
    return found


def main() -> int:
    verify_capture()
    module = load_capture_module()
    found = extract(module)
    words = []
    rows = []
    for slot, sample in enumerate(TARGET_SAMPLES):
        values = found[(sample, TARGET_LAYER)]
        live_rows = 0
        nonzero_codes = 0
        negative_codes = 0
        for context in range(CONTEXTS):
            for group in range(GROUPS):
                source = values[context, group]
                active = sum(1 << lane for lane, value in enumerate(source) if value != 0)
                sign = sum(1 << lane for lane, value in enumerate(source) if value < 0)
                need(sign & ~active == 0, "sign without activity")
                words.append(f"{sign:04x}{active:04x}\n")
                live_rows += int(active != 0)
                nonzero_codes += int(np.count_nonzero(source))
                negative_codes += int(np.count_nonzero(source < 0))
        rows.append({
            "slot": slot,
            "sample_id": sample,
            "layer_id": TARGET_LAYER,
            "token_start": 0,
            "token_count": CONTEXTS,
            "live_rows": live_rows,
            "nonzero_codes": nonzero_codes,
            "negative_codes": negative_codes,
        })
    need(len(words) == len(TARGET_SAMPLES) * CONTEXTS * GROUPS,
         "fixture cardinality drift")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(words), encoding="ascii")
    metadata = {
        "schema": "m2046_ep34_tsbg_g48_four_sequence_fixture_r1_v1",
        "selection_rule": (
            "first sample of each of four captured sequences; layer 28; tokens 0..3"
        ),
        "selection_uses_performance": False,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA,
        "fc_frames_sha256": FC_FRAMES_SHA,
        "m1558_source_sha256": M1558_SHA,
        "geometry": {"samples": 4, "contexts": 4, "groups": 48, "sources": 16},
        "rows": rows,
        "fixture_sha256": sha256(OUT),
        "claim_boundary": {
            "real_ep34_activity_and_sign_descriptors": True,
            "hardware_weight_values": False,
            "cycle_result": False,
            "rtl": False,
            "system_speedup": False,
        },
    }
    META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

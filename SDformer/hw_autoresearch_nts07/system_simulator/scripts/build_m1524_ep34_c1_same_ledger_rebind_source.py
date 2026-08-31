#!/usr/bin/env python3
"""Read-only source design for rebinding C1 same-ledger replay to ep34.

M1458 already retained the exact inputs of the four bottleneck Conv3x3
operators for ten C1 samples.  This adapter authenticates those records and
provides the deterministic mapping from their little-endian support planes to
the 432 partitions x 3000 rows consumed by M505/M528.  It does not materialize
the 51.84-million-row production ledger and it does not execute a cycle model.

The old M410/M528 row word contains q32 diagnostics above bit 15, but the
frozen M504/M505/M528 recurrence consumes only ``word & 0xffff``.  The future
ep34 ledger therefore uses the mechanically compatible line
``0000<support16>``.  A future production analyzer must recompute zero, bit,
and product-capture cycles from that one new ledger; none of the old ep35 cycle
anchors are reusable as ep34 measurements.
"""
from __future__ import annotations

from collections import OrderedDict
import argparse
import hashlib
import json
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Iterable
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1524_ep34_c1_same_ledger_rebind_source.py"
CONTRACT = HW / "contracts/m1524_ep34_c1_same_ledger_rebind_source_contract_r1_20260831.json"
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CAPTURE_MANIFEST = CAPTURE / "manifest.json"
ORDERED = CAPTURE / "unified_ordered_records.jsonl"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M528 = HW / "system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute.py"
M505 = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M504 = HW / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
OLD_M528_RESULT = HW / "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

CAPTURE_MANIFEST_SHA256 = "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d"
CAPTURE_SHA256SUMS_SHA256 = "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e"
CAPTURE_OUTER_FILE_SHA256 = "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed"
ORDERED_SHA256 = "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
M528_SHA256 = "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a"
M505_SHA256 = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
M504_SHA256 = "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
OLD_M528_RESULT_SHA256 = "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

MODULES = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
WEIGHT_KEYS = tuple(name + ".weight" for name in MODULES)
WEIGHT_SHAPE = (768, 768, 3, 3)
WEIGHT_BYTES = 21_233_664
WEIGHT_SHA256 = (
    "e1377479fcdfcb946b5f6d8f0344140f41953224cb999f8506d6f6e860c692c0",
    "f4620a355f6a13bd29cecb05fd3d31d5f3f40f6a1dd874018e3a345790ba32d0",
    "714d4e02223887174665ec4e685c6cc1854535d012a50de2974f7b8537356677",
    "58b96e585075b6da5d9ed0fdeef60a40063d2c80fa6c27894ae8d327f1be687e",
)
NONZERO_BITS = (0x3F7FFA25, 0x3F7FF852, 0x3F7FFA95, 0x3F7FF926)
NONZERO_COUNTS = (2_908_684, 1_322_881, 3_299_475, 1_744_577)
EXPECTED_SAMPLE_KEYS = tuple("zurich_city_09_a_{:04d}.npy".format(1 + 10 * index)
                             for index in range(10))

TIMESTEPS, CHANNELS, HEIGHT, WIDTH = 10, 768, 15, 20
ROWS = TIMESTEPS * HEIGHT * WIDTH
FEATURES = CHANNELS * 3 * 3
PARTITIONS = FEATURES // 16
OPERATORS, SAMPLES = 4, 10
PHASES = SAMPLES * OPERATORS * PARTITIONS
SOURCE_ROWS = PHASES * ROWS
ROW_LINE_BYTES = 9
ROW_STREAM_BYTES = SOURCE_ROWS * ROW_LINE_BYTES
CAPACITY_BYTES = 213_376
CAPACITY_BUDGET_BYTES = 245_760
EXPECTED_STATE_KEYS = 921

SCHEMA = "m1524_ep34_c1_same_ledger_rebind_source_audit_r1_v1"
STATUS = "PASS_M1524_SOURCE_ONLY_EP34_C1_REBIND_DESIGN__NO_PRODUCTION"
SOURCE_STATUS = "SOURCE_ONLY__EP34_C1_SAME_LEDGER_REBIND__PRODUCTION_FALSE"
FUTURE_OUTPUT = (
    "hw_autoresearch_nts07/results/"
    "m1525_ep34_c1_same_ledger_rows_and_three_baseline_replay_r1_20260831"
)


class M1524Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1524Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1524Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == digest, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1524Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def capture_manifest_entries() -> dict[str, str]:
    regular_exact(CAPTURE / "SHA256SUMS", CAPTURE_SHA256SUMS_SHA256,
                  "capture SHA256SUMS")
    regular_exact(CAPTURE / "SHA256SUMS.seal.sha256", CAPTURE_OUTER_FILE_SHA256,
                  "capture outer seal")
    require((CAPTURE / "SHA256SUMS.seal.sha256").read_text().split() ==
            [CAPTURE_SHA256SUMS_SHA256, "SHA256SUMS"],
            "capture outer seal content drift")
    entries: dict[str, str] = {}
    prefix = CAPTURE.relative_to(ROOT).as_posix() + "/"
    for line in (CAPTURE / "SHA256SUMS").read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "capture manifest row malformed")
        digest, name = fields
        name = name.lstrip("*")
        if name.startswith(prefix):
            name = name[len(prefix):]
        require(name not in entries and not Path(name).is_absolute() and
                ".." not in Path(name).parts,
                "unsafe or duplicate capture manifest member")
        entries[name] = digest
    return entries


def verify_capture_identity() -> tuple[dict[str, Any], dict[str, str]]:
    entries = capture_manifest_entries()
    regular_exact(CAPTURE_MANIFEST, CAPTURE_MANIFEST_SHA256, "capture manifest")
    regular_exact(ORDERED, ORDERED_SHA256, "capture ordered records")
    require(entries.get("manifest.json") == CAPTURE_MANIFEST_SHA256 and
            entries.get("unified_ordered_records.jsonl") == ORDERED_SHA256,
            "capture seal does not bind required metadata")
    manifest = strict_json(CAPTURE_MANIFEST)
    require(manifest.get("schema") ==
            "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1" and
            manifest.get("status") ==
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "capture schema/status drift")
    selected = manifest.get("identity", {}).get("selection", {}).get("selected", {})
    require(selected.get("candidate_id") == "resume_ep34" and
            selected.get("epoch") == 34 and
            selected.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA256,
            "selected checkpoint identity drift")
    require(manifest.get("module_inventory", {}).get("c1_conv3x3") == list(MODULES),
            "C1 module inventory drift")
    samples = [row for row in manifest.get("cohort", {}).get("samples", [])
               if row.get("cohort") == "c1"]
    require(len(samples) == SAMPLES and
            tuple(row.get("sample_key") for row in samples) == EXPECTED_SAMPLE_KEYS and
            tuple(row.get("global_sample_id") for row in samples) == tuple(range(SAMPLES)),
            "C1 sample identity/order drift")
    return manifest, entries


def collect_records() -> tuple[list[dict[str, Any]], dict[str, str]]:
    _, entries = verify_capture_identity()
    selected: list[dict[str, Any]] = []
    for line in ORDERED.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("cohort") == "c1" and row.get("category") == "c1_conv3x3":
            selected.append(row)
    require(len(selected) == SAMPLES * OPERATORS, "C1 retained population drift")
    for sample in range(SAMPLES):
        rows = selected[sample * OPERATORS:(sample + 1) * OPERATORS]
        require([row.get("name") for row in rows] == list(MODULES),
                "C1 module order drift")
        for operator, row in enumerate(rows):
            require(row.get("global_sample_id") == sample and
                    row.get("sample_key") == EXPECTED_SAMPLE_KEYS[sample] and
                    row.get("payload", {}).get("retained") is True,
                    "C1 sample/module binding drift")
            input_row = row.get("input", {})
            require(input_row.get("shape") == [10, 1, 768, 15, 20] and
                    input_row.get("dtype") == "torch.float32" and
                    input_row.get("negative") == 0 and
                    input_row.get("nonfinite") == 0 and
                    input_row.get("active") == input_row.get("positive"),
                    "C1 input semantics drift")
            payload = row["payload"]
            require(payload.get("positive_plane_bytes") == 288_000 and
                    payload.get("negative_plane_bytes") == 288_000,
                    "C1 support-plane extent drift")
            for key, sha_key in (("support_sign", "support_sign_sha256"),
                                 ("compressed_fp32", "compressed_sha256")):
                name = payload[key]
                path = CAPTURE / name
                require(entries.get(name) == payload[sha_key],
                        "capture seal/payload identity mismatch")
                regular_exact(path, payload[sha_key], "C1 payload")
            require(operator == MODULES.index(row["name"]),
                    "C1 operator ordinal drift")
    return selected, entries


def decode_support(record: dict[str, Any]) -> np.ndarray:
    """Return exact positive support as [T,C,H,W] bool."""
    payload = record["payload"]
    raw = (CAPTURE / payload["support_sign"]).read_bytes()
    plane_bytes = int(payload["positive_plane_bytes"])
    require(len(raw) == plane_bytes + int(payload["negative_plane_bytes"]),
            "support-sign extent drift")
    positive, negative = raw[:plane_bytes], raw[plane_bytes:]
    require(not any(negative), "C1 negative support is not empty")
    bits = np.unpackbits(np.frombuffer(positive, dtype=np.uint8), bitorder="little")
    elements = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    require(bits.size == elements and int(bits.sum()) == record["input"]["active"],
            "C1 positive support population drift")
    return bits.astype(np.bool_, copy=False).reshape(TIMESTEPS, CHANNELS, HEIGHT, WIDTH)


def phase_masks(support: np.ndarray, partition: int) -> np.ndarray:
    """Map one 16-feature partition to 3000 output-site source masks."""
    support = np.asarray(support, dtype=np.bool_)
    require(support.shape == (TIMESTEPS, CHANNELS, HEIGHT, WIDTH),
            "support tensor geometry drift")
    require(type(partition) is int and 0 <= partition < PARTITIONS,
            "partition out of range")
    masks = np.zeros((TIMESTEPS, HEIGHT, WIDTH), dtype=np.uint16)
    for bit in range(16):
        feature = partition * 16 + bit
        channel, kernel = divmod(feature, 9)
        kernel_y, kernel_x = divmod(kernel, 3)
        y0, x0 = kernel_y - 1, kernel_x - 1
        out_y0, out_y1 = max(0, -y0), min(HEIGHT, HEIGHT - y0)
        out_x0, out_x1 = max(0, -x0), min(WIDTH, WIDTH - x0)
        in_y0, in_y1 = out_y0 + y0, out_y1 + y0
        in_x0, in_x1 = out_x0 + x0, out_x1 + x0
        masks[:, out_y0:out_y1, out_x0:out_x1] |= (
            support[:, channel, in_y0:in_y1, in_x0:in_x1].astype(np.uint16)
            << np.uint16(bit)
        )
    return masks.reshape(ROWS)


def m528_compatible_lines(masks: np.ndarray) -> bytes:
    masks = np.asarray(masks, dtype=np.uint16).reshape(-1)
    require(masks.size == ROWS, "phase row population drift")
    return "".join("0000{:04x}\n".format(int(value)) for value in masks).encode("ascii")


def numeric_codebook_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Authenticate exact two-code C1 values without changing them."""
    output = []
    for operator, module in enumerate(MODULES):
        counts: dict[int, int] = {}
        for record in records:
            if record["name"] != module:
                continue
            compressed = (CAPTURE / record["payload"]["compressed_fp32"]).read_bytes()
            raw = zlib.decompress(compressed)
            require(hashlib.sha256(raw).hexdigest() ==
                    record["payload"]["raw_fp32_sha256"],
                    "raw C1 float payload SHA drift")
            require(len(raw) == TIMESTEPS * CHANNELS * HEIGHT * WIDTH * 4,
                    "raw C1 float payload extent drift")
            values, population = np.unique(np.frombuffer(raw, dtype="<u4"),
                                           return_counts=True)
            for value, count in zip(values, population):
                counts[int(value)] = counts.get(int(value), 0) + int(count)
        expected = {0: SAMPLES * TIMESTEPS * CHANNELS * HEIGHT * WIDTH -
                    NONZERO_COUNTS[operator],
                    NONZERO_BITS[operator]: NONZERO_COUNTS[operator]}
        require(counts == expected, "C1 exact two-code identity drift")
        output.append({
            "operator": operator,
            "module": module,
            "zero_bits_hex": "00000000",
            "nonzero_bits_hex": "{:08x}".format(NONZERO_BITS[operator]),
            "nonzero_float32": struct.unpack("<f", struct.pack("<I",
                                                                 NONZERO_BITS[operator]))[0],
            "nonzero_count": NONZERO_COUNTS[operator],
            "two_code_exact": True,
        })
    return output


def checkpoint_weight_audit() -> list[dict[str, Any]]:
    """Read-only identity audit; this is not an INT8 export/bridge."""
    import torch
    regular_exact(CHECKPOINT, CHECKPOINT_SHA256, "ep34 checkpoint")
    value = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    require(type(value) is dict and set(value) == {"model_state_dict"},
            "checkpoint root drift")
    state = value["model_state_dict"]
    require(type(state) is OrderedDict and len(state) == EXPECTED_STATE_KEYS,
            "checkpoint state dictionary drift")
    output = []
    for ordinal, key in enumerate(WEIGHT_KEYS):
        require(key in state and key[:-6] + "bias" not in state,
                "C1 weight/bias identity drift")
        tensor = state[key]
        require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and
                tensor.dtype == torch.float32 and tuple(tensor.shape) == WEIGHT_SHAPE and
                tensor.is_contiguous() and sys.byteorder == "little",
                "C1 weight tensor geometry/dtype drift")
        raw = tensor.detach().numpy().tobytes(order="C")
        require(len(raw) == WEIGHT_BYTES and
                hashlib.sha256(raw).hexdigest() == WEIGHT_SHA256[ordinal],
                "C1 weight content identity drift")
        output.append({
            "operator": ordinal,
            "module": MODULES[ordinal],
            "checkpoint_key": key,
            "shape": list(WEIGHT_SHAPE),
            "dtype": "torch.float32",
            "content_bytes": WEIGHT_BYTES,
            "content_sha256": WEIGHT_SHA256[ordinal],
            "bias": None,
        })
    return output


def audit(run_numeric: bool = False, run_checkpoint: bool = False) -> dict[str, Any]:
    for path, digest, label in ((M528, M528_SHA256, "old M528 analyzer"),
                                (M505, M505_SHA256, "old M505 analyzer"),
                                (M504, M504_SHA256, "old M504 analyzer"),
                                (OLD_M528_RESULT, OLD_M528_RESULT_SHA256,
                                 "old M528 result"),
                                (DOCS359, DOCS359_SHA256, "docs359")):
        regular_exact(path, digest, label)
    records, _ = collect_records()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "identity": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "samples": list(EXPECTED_SAMPLE_KEYS),
            "modules": list(MODULES),
            "retained_records": len(records),
        },
        "geometry": {
            "samples": SAMPLES,
            "operators": OPERATORS,
            "partitions": PARTITIONS,
            "rows_per_phase": ROWS,
            "phases": PHASES,
            "source_rows": SOURCE_ROWS,
            "future_memh_bytes": ROW_STREAM_BYTES,
            "phase_order": "sample,operator,partition",
            "row_order": "timestep,output_y,output_x",
            "line_format": "0000<support16_lowercase_hex>\\n",
        },
        "capacity_coordinate": {
            "candidate_macro_rounded_bytes": CAPACITY_BYTES,
            "budget_bytes": CAPACITY_BUDGET_BYTES,
            "unchanged_from_m528": True,
        },
        "rebind_decision": {
            "new_gpu_capture_required": False,
            "new_ep34_row_ledger_required": True,
            "zero_bit_product_must_share_new_ledger": True,
            "old_ep35_cycle_numerators_or_denominators_reusable": False,
            "old_cycle_values_are_historical_comparison_only": True,
            "ep34_c1_weight_identity_available": run_checkpoint,
            "ep34_int8_bridge_available": False,
            "int8_bridge_is_paper_promotion_prerequisite": True,
        },
        "future_action": {
            "output": FUTURE_OUTPUT,
            "exists_or_written_by_m1524": False,
            "production": False,
            "independent_release_required": True,
            "independent_result_hammer_required": True,
        },
        "claim_boundary": {
            "source_only": True,
            "read_only": True,
            "production": False,
            "cpu_cycle_replay": False,
            "gpu": False,
            "remote": False,
            "eda": False,
            "vcs": False,
            "rows_materialized": False,
            "cycles": False,
            "speedup": False,
            "system_speedup": False,
            "energy": False,
            "ppa": False,
            "paper_result": False,
        },
    }
    if run_numeric:
        result["numeric_codebook"] = numeric_codebook_audit(records)
    if run_checkpoint:
        result["c1_weights"] = checkpoint_weight_audit()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-audit", action="store_true")
    parser.add_argument("--checkpoint-audit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.numeric_audit, args.checkpoint_audit),
                     indent=2, sort_keys=True))
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

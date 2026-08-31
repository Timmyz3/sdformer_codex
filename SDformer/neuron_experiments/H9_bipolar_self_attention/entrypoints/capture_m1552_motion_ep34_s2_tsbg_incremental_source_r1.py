#!/usr/bin/env python3
"""Source-only ep34 S2/TSBG compact incremental-capture producer.

The producer is a streaming forward-hook writer intended for a future,
independently released successor of the existing M1434/M1174 model/sample
loop.  This file has no production CLI and never imports torch at module load.
It cannot load a checkpoint, start CUDA, use SSH, or publish a capture.

All emitted codes are diagnostic captured codewords.  They are exact only
with respect to the stored codeword/contributor stream; they are not a formal
INT8 authority or a model-bit-exact claim.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
M1434 = SOURCE.with_name(
    "capture_m1434_motion_ep34_live93_runtime_successor_r1.py")
M1227 = SOURCE.with_name(
    "capture_m1227_motion_final_checkpoint_unified_hardware_r1.py")
M1174 = SOURCE.with_name(
    "capture_m1174_motion_checkpoint_parametric_unified_hardware.py")
M1458_ROOT = HW / (
    "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831")
M1458_MANIFEST = M1458_ROOT / "manifest.json"
M1458_OPERATORS = M1458_ROOT / "operator_runtime.json"
M1458_ORDERED = M1458_ROOT / "unified_ordered_records.jsonl"
SAMPLE_ORDER = HW / (
    "system_handoff/m1544_ep34_sparse_capture_handoff_r1_20260831/sample_order.json")
M1544_VALIDATOR = HW / (
    "system_handoff/scripts/validate_m1544_ep34_sparse_capture_handoff.py")
M1548 = HW / (
    "reviews/m1548_m1544_sparse_capture_handoff_independent_hammer_r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    M1434: "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed",
    M1227: "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b",
    M1174: "b476fad6885be23aa63a6b5d8e690fb3e213421074270cbb25e8ec00c202080a",
    M1458_MANIFEST: "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    M1458_OPERATORS: "eb0cd40e701361f8acc08d6003680de0ca35626e8e75dcf56827c978899e8a8e",
    M1458_ORDERED: "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    SAMPLE_ORDER: "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773",
    M1544_VALIDATOR: "463fa7392fa090eda7fdb298fcc10ff896f91a961a0a529a013be2eec47ec240",
    M1548 / "review.json": "4f52f4fa7ccbeef3fe3b83e3ba4d69b61f670a0eb6195138138a171cf657719a",
    M1548 / "SHA256SUMS.seal.sha256": "24f2932cfe1ea35325155b43b70e68cb396b051295f69d5934ed04922fa07260",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
M1458_ORDER_SHA256 = "88db38f9cc3f3e0b89cf332ef84958ed87e7c84873075e4399a2a54d2ce64c47"
SOURCE_SCHEMA = "m1552_motion_ep34_s2_tsbg_incremental_producer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__STREAMING_HOOK_PRODUCER__NO_GPU_NO_CAPTURE_NO_RELEASE"
MAX_ESTIMATED_BYTES = 12 * 1024 * 1024 * 1024
MIN_FREE_AFTER_BYTES = 16 * 1024 * 1024 * 1024
GROUP_WIDTH = 16
OUTPUT_TILE_WIDTH = 96
TOKEN_CHUNK = 4096
TARGET_COUNTS = {"FC1": 12, "FC2": 12, "PATCH": 8}
MAGNITUDE_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128]

GATES = {
    "S1": {
        "metadata_plus_beta_over_saved_weight_bytes_veto": 0.25,
        "beta_port_cycle_regression_veto": 0.05,
        "mean_delta_aee_max": 0.02,
        "per_sequence_delta_aee_max": 0.03,
    },
    "S2": {
        "total_metadata_over_weight_bytes_max": 0.02,
        "metadata_reduction_vs_g11_min": 8.0,
        "dynamic_same_block_keep_drop_witness_required": True,
    },
    "TSBG": {
        "aggregate_fc1_fc2_cycle_speedup_min": 1.15,
        "every_sequence_cycle_speedup_min": 1.05,
        "energy_branch_cycle_regression_max": 0.05,
        "energy_branch_weight_byte_reduction_min": 0.30,
        "energy_branch_memory_energy_reduction_min": 0.20,
    },
}


class M1552Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1552Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1552Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path, root_type=dict):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1552Error("nonfinite JSON: " + token)))
    require(type(value) is root_type, "JSON root type mismatch")
    return value


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def project_m1434_sample(row):
    """Drop runtime-only path/cohort fields while retaining exact S40 identity."""
    required = ("global_sample_id", "sequence", "sequence_sample_id",
                "sample_key", "sha256")
    require(type(row) is dict and all(key in row for key in required),
            "M1434 sample projection missing identity field")
    return dict((key, row[key]) for key in required)


def verify_bindings():
    for path, expected in EXPECTED.items():
        regular_exact(path, expected, str(path.relative_to(ROOT)))
    review = strict_json(M1548 / "review.json")
    require(review.get("status") ==
            "PASS_M1548_EXACT_M1544_HANDOFF__TRANSFER_AND_PRODUCER_INTEGRATION_ONLY__NO_CAPTURE" and
            review.get("authorization", {}).get("remote_producer_integration") is True and
            review.get("authorization", {}).get("capture_execution") is False,
            "M1548 authorization boundary drift")
    samples = strict_json(SAMPLE_ORDER)
    require(samples["identity"]["checkpoint_sha256"] == CHECKPOINT_SHA256 and
            len(samples["samples"]) == 40 and
            [row["global_sample_id"] for row in samples["samples"]] == list(range(40)) and
            canonical_sha(samples["samples"]) == M1458_ORDER_SHA256,
            "M1458 S40 identity drift")
    return samples


def target_for(name, operator):
    if operator == "Linear" and name.endswith(".mlp.fc1"):
        return "FC1"
    if operator == "Linear" and name.endswith(".mlp.fc2"):
        return "FC2"
    if operator == "Conv2d" and ".patch_embed." in name:
        return "PATCH"
    return None


def frozen_layer_specs():
    verify_bindings()
    runtime = strict_json(M1458_OPERATORS, list)
    first_order = {}
    with M1458_ORDERED.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("global_sample_id") == 0 and row.get("category") in (
                    "fc1", "fc2", "patch_embed"):
                name = row["name"]
                require(name not in first_order, "target module fired twice in sample0")
                first_order[name] = int(row["global_order"])
    specs = []
    counts = dict((key, 0) for key in TARGET_COUNTS)
    for row in runtime:
        name = row.get("name")
        operator = row.get("operator")
        target = target_for(name, operator)
        if target is None:
            continue
        input_shape = tuple(int(value) for value in row["input_shape_first"])
        output_shape = tuple(int(value) for value in row["output_shape_first"])
        if operator == "Linear":
            input_channels = input_shape[-1]
            output_channels = output_shape[-1]
            channel_axis = len(input_shape) - 1
        else:
            require(len(input_shape) in (4, 5), "unsupported Conv2d input rank")
            channel_axis = 1 if len(input_shape) == 4 else 2
            input_channels = input_shape[channel_axis]
            output_channels = output_shape[channel_axis]
        require(name in first_order and int(row["calls"]) == 40,
                "target module missing exact sample0/call authority")
        specs.append({
            "target": target, "module_name": name,
            "operator": operator, "operator_order": first_order[name],
            "input_shape": input_shape, "output_shape": output_shape,
            "channel_axis": channel_axis,
            "input_channels": input_channels,
            "output_channels": output_channels,
        })
        counts[target] += 1
    specs.sort(key=lambda row: row["operator_order"])
    require(counts == TARGET_COUNTS and len(specs) == 32 and
            len(set(row["module_name"] for row in specs)) == 32 and
            len(set(row["operator_order"] for row in specs)) == 32,
            "frozen FC1/FC2/PATCH inventory drift")
    for index, row in enumerate(specs):
        row["layer_id"] = index
    return specs


def preflight_before_checkpoint_load(output, estimated_result_bytes,
                                     free_bytes=None):
    output = Path(output)
    estimate = int(estimated_result_bytes)
    require(0 < estimate <= MAX_ESTIMATED_BYTES,
            "estimated result exceeds strict 12 GiB cap")
    require(not os.path.lexists(str(output)), "fresh capture namespace required")
    parent = output.parent.resolve()
    require(parent.is_dir() and not parent.is_symlink(), "output parent invalid")
    available = shutil.disk_usage(str(parent)).free if free_bytes is None else int(free_bytes)
    require(available - estimate >= MIN_FREE_AFTER_BYTES,
            "capture would leave less than 16 GiB free")
    return {"estimated_result_bytes": estimate, "free_bytes_before": available,
            "free_bytes_after_lower_bound": available - estimate,
            "checkpoint_loaded": False}


def diagnostic_codebook():
    return {
        "width_bits": 8, "signed": True, "zero_point": 0, "unit_code": 1,
        "scale_numerator": 1, "scale_denominator": 1,
        "rounding": "nearest_even", "saturation": "signed_clamp",
        "authority": "diagnostic_fixed_point_codeword",
        "diagnostic_capture_only": True, "hardware_quant_authority": False,
    }


def module_dimensions(module):
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)
    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        return int(module.in_channels), int(module.out_channels)
    if hasattr(module, "m1552_input_channels") and hasattr(module, "m1552_output_channels"):
        return int(module.m1552_input_channels), int(module.m1552_output_channels)
    raise M1552Error("hook target lacks channel dimensions")


def weight_beta_by_tile(module, output_channels):
    tiles = (int(output_channels) + OUTPUT_TILE_WIDTH - 1) // OUTPUT_TILE_WIDTH
    if hasattr(module, "m1552_beta_by_tile"):
        values = [int(value) for value in module.m1552_beta_by_tile]
        require(len(values) == tiles and all(value > 0 for value in values),
                "synthetic beta vector malformed")
        return values
    weight = getattr(module, "weight", None)
    require(weight is not None and hasattr(weight, "detach"),
            "real hook target lacks checkpoint weight tensor")
    values = []
    detached = weight.detach()
    for tile in range(tiles):
        begin = tile * OUTPUT_TILE_WIDTH
        end = min((tile + 1) * OUTPUT_TILE_WIDTH, int(output_channels))
        maximum = detached[begin:end].abs().max().item()
        require(math.isfinite(float(maximum)), "nonfinite checkpoint weight")
        values.append(max(1, int(math.ceil(float(maximum)))))
    return values


def build_layer_rows(model, specs):
    named = dict(model.named_modules())
    rows = []
    next_address = 0
    betas = {}
    for spec in specs:
        name = spec["module_name"]
        require(name in named, "frozen target missing from model: " + name)
        module = named[name]
        observed = module_dimensions(module)
        require(observed == (spec["input_channels"], spec["output_channels"]),
                "model target channel drift: " + name)
        source_groups = (spec["input_channels"] + GROUP_WIDTH - 1) // GROUP_WIDTH
        output_tiles = (spec["output_channels"] + OUTPUT_TILE_WIDTH - 1) // OUTPUT_TILE_WIDTH
        row_bytes = GROUP_WIDTH * OUTPUT_TILE_WIDTH * 4
        base = next_address
        blocks = []
        for output_tile in range(output_tiles):
            for source_group in range(source_groups):
                index = output_tile * source_groups + source_group
                address = base + index * row_bytes
                blocks.append({
                    "source_group_id": source_group,
                    "output_tile_id": output_tile,
                    "address": address,
                    "bank_key": (address // row_bytes) % 8,
                    "row_buffer_key": "%d:%d:%d" % (
                        spec["layer_id"], output_tile, source_group),
                })
        next_address += source_groups * output_tiles * row_bytes
        rows.append({
            "layer_id": spec["layer_id"], "target": spec["target"],
            "module_name": name, "operator_order": spec["operator_order"],
            "input_channels": spec["input_channels"],
            "output_channels": spec["output_channels"],
            "group_width": GROUP_WIDTH, "output_tile_width": OUTPUT_TILE_WIDTH,
            "codebook": diagnostic_codebook(),
            "weight_layout": {
                "base_address": base, "bank_count": 8, "row_bytes": row_bytes,
                "address_formula":
                    "base_address+(output_tile_id*source_group_count+source_group_id)*row_bytes",
                "bank_formula": "(address//row_bytes)%bank_count",
                "row_buffer_baseline": "ordinary_same_capacity_LRU_weight_row_buffer",
                "blocks": blocks,
            },
            "s1_eligible": spec["target"] == "PATCH",
            "s1_magnitude_bin_edges_abs_code":
                list(MAGNITUDE_EDGES) if spec["target"] == "PATCH" else [],
        })
        betas[spec["layer_id"]] = weight_beta_by_tile(module, spec["output_channels"])
    return rows, betas


class CanonicalZlibJsonlWriter(object):
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("wb")
        self.compressor = zlib.compressobj(9)
        self.rows = 0

    def write(self, value):
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        payload = self.compressor.compress(raw)
        if payload:
            self.stream.write(payload)
        self.rows += 1

    def close(self):
        if self.stream is not None:
            self.stream.write(self.compressor.flush(zlib.Z_FINISH))
            self.stream.close()
            self.stream = None


class TorchTokenAdapter(object):
    """Chunked tensor adapter; never materializes a complete CPU tensor."""
    def __init__(self, torch_module):
        self.torch = torch_module

    def shape(self, tensor):
        return tuple(int(value) for value in tensor.shape)

    def chunks(self, tensor, spec, wanted=TOKEN_CHUNK):
        value = tensor.detach()
        axis = int(spec["channel_axis"])
        if axis != value.dim() - 1:
            value = value.movedim(axis, -1)
        value = value.reshape(-1, int(spec["input_channels"]))
        for begin in range(0, int(value.shape[0]), int(wanted)):
            chunk = value[begin:begin + int(wanted)].to(device="cpu")
            codes = self.torch.clamp(self.torch.round(chunk), -128, 127).to(
                dtype=self.torch.int8)
            yield codes.tolist()


class SyntheticTokenAdapter(object):
    def shape(self, tensor):
        return tuple(tensor.shape)

    def chunks(self, tensor, spec, wanted=TOKEN_CHUNK):
        del spec, wanted
        yield [list(map(int, row)) for row in tensor.rows]


def encode_group(codes):
    valid = len(codes)
    width = (valid + 7) // 8
    support = bytearray(width); signs = bytearray(width); nonunit = bytearray(width)
    nonzero = []
    for index, code in enumerate(codes):
        code = int(code)
        require(-128 <= code <= 127, "diagnostic code outside int8")
        if code:
            support[index // 8] |= 1 << (index % 8)
            if code < 0:
                signs[index // 8] |= 1 << (index % 8)
            if abs(code) != 1:
                nonunit[index // 8] |= 1 << (index % 8)
            nonzero.append(code & 0xff)
    if not nonzero:
        return None
    return {"valid_channels": valid, "support_hex": bytes(support).hex(),
            "sign_hex": bytes(signs).hex(), "nonunit_hex": bytes(nonunit).hex(),
            "nonzero_codes_le_hex": bytes(bytearray(nonzero)).hex()}


def token_coordinates(spec, token_order):
    shape = spec["input_shape"]
    axis = spec["channel_axis"]
    logical = list(shape[:axis]) + list(shape[axis + 1:])
    require(len(logical) in (3, 4), "unsupported logical token rank")
    value = int(token_order)
    x = value % logical[-1]; value //= logical[-1]
    y = value % logical[-2]; value //= logical[-2]
    window = value
    return int(window), int(y), int(x)


class SparseCaptureProducer(object):
    def __init__(self, model, tensor_adapter, root, specs, sample_order):
        self.model = model
        self.adapter = tensor_adapter
        self.root = Path(root)
        require(not os.path.lexists(str(self.root)), "producer output must be fresh")
        self.root.mkdir()
        self.specs = [dict(row) for row in specs]
        self.samples = list(sample_order["samples"])
        require(len(self.samples) == 40 and
                [row["global_sample_id"] for row in self.samples] == list(range(40)),
                "producer requires exact S40 order")
        self.layers, self.betas = build_layer_rows(model, self.specs)
        (self.root / "sample_order.json").write_bytes(SAMPLE_ORDER.read_bytes())
        (self.root / "layers.json").write_text(json.dumps({
            "schema": "m1544_ep34_sparse_capture_layers_r1_v1",
            "status": "STATIC_WEIGHT_LAYOUT_COMPLETE__NO_CYCLE_OR_ENERGY_CLAIM",
            "layers": self.layers,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.tokens = CanonicalZlibJsonlWriter(self.root / "token_source_groups.jsonl.zlib")
        self.s1 = CanonicalZlibJsonlWriter(self.root / "s1_histogram_debt.jsonl.zlib")
        self.global_order = 0
        self.handles = []
        self.active_sample = None
        self.expected_hook = 0
        self.s1_accum = {}
        self._attach()

    def _attach(self):
        named = dict(self.model.named_modules())
        for spec in self.specs:
            module = named[spec["module_name"]]
            def hook(_module, inputs, _output, _spec=spec):
                self._on_input(_spec, inputs)
            self.handles.append(module.register_forward_hook(hook))

    def begin_sample(self, sample):
        require(self.active_sample is None, "nested producer sample")
        expected = self.samples[int(sample["global_sample_id"])]
        projected = project_m1434_sample(sample)
        require(projected == expected, "producer sample identity/order drift")
        self.active_sample = projected
        self.expected_hook = 0
        self.s1_accum = {}

    def _on_input(self, spec, inputs):
        require(self.active_sample is not None, "target hook fired outside sample")
        require(self.expected_hook < len(self.specs) and
                spec["layer_id"] == self.specs[self.expected_hook]["layer_id"],
                "target hook order/duplicate drift")
        self.expected_hook += 1
        tensors = [value for value in inputs if hasattr(value, "shape")]
        require(len(tensors) == 1, "target hook requires exactly one tensor input")
        tensor = tensors[0]
        require(self.adapter.shape(tensor) == tuple(spec["input_shape"]),
                "target input shape drift: " + spec["module_name"])
        token_order = 0
        for chunk in self.adapter.chunks(tensor, spec):
            for codes in chunk:
                require(len(codes) == spec["input_channels"], "token channel drift")
                groups = []
                for begin in range(0, len(codes), GROUP_WIDTH):
                    encoded = encode_group(codes[begin:begin + GROUP_WIDTH])
                    if encoded is not None:
                        encoded["source_group_id"] = begin // GROUP_WIDTH
                        groups.append(encoded)
                window, y, x = token_coordinates(spec, token_order)
                sample = self.active_sample
                self.tokens.write({
                    "schema": "m1544_ep34_sparse_token_source_groups_r1_v1",
                    "global_order": self.global_order,
                    "sample_global_id": sample["global_sample_id"],
                    "sequence": sample["sequence"],
                    "sequence_sample_id": sample["sequence_sample_id"],
                    "sample_key": sample["sample_key"],
                    "operator_order": spec["operator_order"],
                    "layer_id": spec["layer_id"], "token_order": token_order,
                    "window_order": window, "spatial_y": y, "spatial_x": x,
                    "groups": groups,
                })
                self.global_order += 1
                if spec["target"] == "PATCH":
                    self._accumulate_s1(spec, codes)
                token_order += 1
        expected_tokens = 1
        for index, value in enumerate(spec["input_shape"]):
            if index != spec["channel_axis"]:
                expected_tokens *= int(value)
        require(token_order == expected_tokens, "token population drift")

    def _accumulate_s1(self, spec, codes):
        layer = spec["layer_id"]
        bins = len(MAGNITUDE_EDGES) - 1
        counts = [0] * bins
        magnitude_sum = [0] * bins
        for code in codes:
            magnitude = abs(int(code))
            if magnitude == 0:
                continue
            index = bins - 1
            for candidate in range(bins):
                if MAGNITUDE_EDGES[candidate] <= magnitude < MAGNITUDE_EDGES[candidate + 1]:
                    index = candidate; break
            counts[index] += 1
            magnitude_sum[index] += magnitude
        for tile, beta in enumerate(self.betas[layer]):
            key = (layer, tile)
            entry = self.s1_accum.setdefault(key, {
                "counts": [0] * bins, "debt": [0] * bins})
            for index in range(bins):
                entry["counts"][index] += counts[index]
                entry["debt"][index] += int(beta) * magnitude_sum[index]

    def end_sample(self):
        require(self.active_sample is not None and self.expected_hook == len(self.specs),
                "sample target hook population incomplete")
        sample_id = self.active_sample["global_sample_id"]
        for layer in self.layers:
            if not layer["s1_eligible"]:
                continue
            tiles = (layer["output_channels"] + OUTPUT_TILE_WIDTH - 1) // OUTPUT_TILE_WIDTH
            for tile in range(tiles):
                entry = self.s1_accum[(layer["layer_id"], tile)]
                self.s1.write({
                    "schema": "m1544_ep34_s1_histogram_debt_r1_v1",
                    "sample_global_id": sample_id, "layer_id": layer["layer_id"],
                    "output_tile_id": tile,
                    "count_by_magnitude_bin": entry["counts"],
                    "beta_abs_code_debt_by_magnitude_bin": entry["debt"],
                    "nonzero_source_count": sum(entry["counts"]),
                    "beta_rounding": "ceil_upper_bound",
                })
        self.active_sample = None
        self.s1_accum = {}

    def finalize_source_result(self):
        require(self.active_sample is None and self.tokens.rows > 0 and self.s1.rows > 0,
                "producer result incomplete")
        while self.handles:
            self.handles.pop().remove()
        self.tokens.close(); self.s1.close()
        manifest = {
            "schema": "m1544_ep34_sparse_incremental_capture_manifest_r1_v1",
            "status": "CAPTURE_COMPLETE__INDEPENDENT_HAMMER_REQUIRED__NO_PERFORMANCE_CLAIM",
            "identity": {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "m1458_manifest_sha256": EXPECTED[M1458_MANIFEST],
                "m1458_inner_manifest_sha256":
                    "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
                "m1458_outer_file_sha256":
                    "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
                "m1458_sample_order_sha256": M1458_ORDER_SHA256,
                "m1540_review_sha256":
                    "218e3d23fae126ddc4a8655f8e9cd7cb762276ab87c7494b7ad05f6e469730bb",
                "m1541_review_sha256":
                    "849fd69b735779057ea2d197985b1dc81183f62b6c49c569f490659cdef86365",
            },
            "population": {"samples": 40, "layers": len(self.layers),
                           "token_records": self.tokens.rows,
                           "s1_histogram_rows": self.s1.rows},
            "files": {"sample_order": "sample_order.json", "layers": "layers.json",
                      "tokens": "token_source_groups.jsonl.zlib",
                      "s1": "s1_histogram_debt.jsonl.zlib"},
            "encoding": {
                "token_container": "canonical_jsonl_zlib_level9",
                "zero_groups": "omitted_from_groups_but_token_record_retained",
                "support_sign_nonunit": "little_endian_channel_bitsets",
                "codes": "signed_little_endian_nonzero_only",
                "full_fp_tensor_saved": False,
                "static_weight_mapping_repeated_per_token": False},
            "coverage": {"all_40_samples": True, "all_layers_each_sample": True,
                         "targets": ["FC1", "FC2", "PATCH"],
                         "token_records": self.tokens.rows},
            "admission_gates": GATES,
            "claim_boundary": {
                "capture_only": True, "static_opportunity": False,
                "cycles": False, "speedup": False, "traffic": False,
                "energy": False, "aee": False, "rtl": False,
                "paper_headline": False, "hardware_quantization_authority": False,
                "model_bit_exact": False,
                "tsbg_exact_scope": "captured_codeword_and_contributor_only",
                "formal_int8_bridge_required": True},
        }
        (self.root / "capture_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.root / "RUN_COMPLETE.txt").write_text(
            "M1544_EP34_SPARSE_CAPTURE_COMPLETE__NO_HARDWARE_CLAIM\n", encoding="ascii")
        members = sorted({"capture_manifest.json", "sample_order.json", "layers.json",
                          "token_source_groups.jsonl.zlib", "s1_histogram_debt.jsonl.zlib",
                          "RUN_COMPLETE.txt"})
        sha_manifest = self.root / "SHA256SUMS"
        sha_manifest.write_text("".join(
            "{}  {}\n".format(sha256(self.root / name), name) for name in members),
            encoding="ascii")
        (self.root / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(sha256(sha_manifest)), encoding="ascii")
        return self.root


def load_validator():
    regular_exact(M1544_VALIDATOR, EXPECTED[M1544_VALIDATOR], "M1544 validator")
    spec = importlib.util.spec_from_file_location("m1552_bound_m1544", str(M1544_VALIDATOR))
    require(spec is not None and spec.loader is not None, "cannot import M1544 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def production_release(_token=None):
    raise M1552Error(
        "M1552 is source-only; independent source hammer and explicit one-shot "
        "remote release are required before checkpoint load or capture")


def describe():
    return {
        "schema": SOURCE_SCHEMA, "status": SOURCE_STATUS,
        "targets": TARGET_COUNTS, "samples": 40,
        "integration": {
            "runtime_binding": "M1434.build_runtime",
            "model_loader": "M1174 profile.load_config/build_model/validate_h9_load_audit",
            "sample_loader": "M1174 np.load+preprocess_chunk exact loop",
            "hook_point": "forward input of exact M1458 FC1/FC2/PATCH inventory"},
        "streaming": {"token_chunk": TOKEN_CHUNK,
                      "canonical_zlib_jsonl": True, "full_tensor_saved": False},
        "quantization": {"hardware_authority": False,
                         "exact_scope": "captured_codeword_and_contributor_only"},
        "pre_checkpoint_gates": {"estimate_max_bytes": MAX_ESTIMATED_BYTES,
                                 "free_after_min_bytes": MIN_FREE_AFTER_BYTES},
        "execution": {"gpu": False, "ssh": False, "capture": False,
                      "release": False, "automatic_retry": False}}


def source_self_check():
    samples = verify_bindings()
    specs = frozen_layer_specs()
    require(len(samples["samples"]) == 40 and
            {key: sum(row["target"] == key for row in specs)
             for key in TARGET_COUNTS} == TARGET_COUNTS,
            "source self-check population drift")
    require(describe()["execution"] == {
        "gpu": False, "ssh": False, "capture": False,
        "release": False, "automatic_retry": False},
        "source execution boundary drift")
    return {"status": "PASS_M1552_SOURCE_SELF_CHECK__NO_GPU_NO_CAPTURE",
            "layers": len(specs), "samples": 40,
            "hardware_quantization_authority": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args(argv)
    value = describe() if args.describe else source_self_check()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

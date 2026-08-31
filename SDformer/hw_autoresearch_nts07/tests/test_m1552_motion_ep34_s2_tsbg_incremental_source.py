#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1552_motion_ep34_s2_tsbg_incremental_source_r1.py")
SPEC = importlib.util.spec_from_file_location("m1552_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Handle(object):
    def __init__(self, hooks, hook):
        self.hooks = hooks; self.hook = hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class FakeModule(object):
    def __init__(self, input_channels, output_channels, beta=None):
        self.m1552_input_channels = int(input_channels)
        self.m1552_output_channels = int(output_channels)
        tiles = (int(output_channels) + M.OUTPUT_TILE_WIDTH - 1) // M.OUTPUT_TILE_WIDTH
        self.m1552_beta_by_tile = list(beta or ([2] * tiles))
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self.hooks, hook)

    def fire(self, tensor):
        for hook in list(self.hooks):
            hook(self, (tensor,), None)


class FakeModel(object):
    def __init__(self, modules):
        self.modules = dict(modules)

    def named_modules(self):
        return list(self.modules.items())


class FakeTensor(object):
    def __init__(self, shape, rows):
        self.shape = tuple(shape)
        self.rows = [list(row) for row in rows]


def rejects(function):
    try:
        function()
    except (M.M1552Error, AssertionError):
        return
    raise AssertionError("attack accepted")


def small_specs():
    return [
        {"layer_id": 0, "target": "PATCH", "module_name": "fixture.patch",
         "operator": "Conv2d", "operator_order": 0,
         "input_shape": (1, 1, 4, 1, 2), "output_shape": (1, 1, 4, 1, 2),
         "channel_axis": 2, "input_channels": 4, "output_channels": 4},
        {"layer_id": 1, "target": "FC1", "module_name": "fixture.mlp.fc1",
         "operator": "Linear", "operator_order": 1,
         "input_shape": (1, 1, 1, 2, 4), "output_shape": (1, 1, 1, 2, 4),
         "channel_axis": 4, "input_channels": 4, "output_channels": 4},
        {"layer_id": 2, "target": "FC2", "module_name": "fixture.mlp.fc2",
         "operator": "Linear", "operator_order": 2,
         "input_shape": (1, 1, 1, 2, 4), "output_shape": (1, 1, 1, 2, 4),
         "channel_axis": 4, "input_channels": 4, "output_channels": 4},
    ]


def small_model():
    return FakeModel({
        "fixture.patch": FakeModule(4, 4, [3]),
        "fixture.mlp.fc1": FakeModule(4, 4, [2]),
        "fixture.mlp.fc2": FakeModule(4, 4, [2]),
    })


def run_valid(root):
    samples = M.verify_bindings()
    model = small_model()
    producer = M.SparseCaptureProducer(
        model, M.SyntheticTokenAdapter(), root, small_specs(), samples)
    patch = FakeTensor((1, 1, 4, 1, 2), [[0, 0, 0, 0], [1, -2, 0, 3]])
    linear = FakeTensor((1, 1, 1, 2, 4), [[0, 0, 0, 0], [1, -2, 0, 3]])
    for sample in samples["samples"]:
        producer.begin_sample(sample)
        model.modules["fixture.patch"].fire(patch)
        model.modules["fixture.mlp.fc1"].fire(linear)
        model.modules["fixture.mlp.fc2"].fire(linear)
        producer.end_sample()
    result = producer.finalize_source_result()
    return M.load_validator().validate_capture(result), result


def main():
    attacks = []
    source = M.source_self_check()
    assert source["layers"] == 32 and source["samples"] == 40
    description = M.describe()
    assert description["execution"] == {
        "gpu": False, "ssh": False, "capture": False,
        "release": False, "automatic_retry": False}
    assert description["quantization"]["hardware_authority"] is False
    specs = M.frozen_layer_specs()
    assert {key: sum(row["target"] == key for row in specs)
            for key in M.TARGET_COUNTS} == M.TARGET_COUNTS
    assert specs[0]["module_name"] == (
        "sttmultires_unet.encoders.swin3d.patch_embed.head.conv.0")
    assert specs[-1]["module_name"].endswith("layers.3.swin_blocks.1.mlp.fc2")
    exact_fake = FakeModel(dict(
        (row["module_name"], FakeModule(row["input_channels"],
                                        row["output_channels"]))
        for row in specs))
    exact_layers, exact_betas = M.build_layer_rows(exact_fake, specs)
    assert len(exact_layers) == len(exact_betas) == 32
    assert [row["module_name"] for row in exact_layers] == [
        row["module_name"] for row in specs]

    with tempfile.TemporaryDirectory(prefix="m1552_test.") as directory:
        root = Path(directory)
        out = root / "valid"
        validated, result = run_valid(out)
        assert validated["samples"] == 40 and validated["layers"] == 3
        assert validated["token_records"] == 240
        assert validated["s1_histogram_rows"] == 40
        assert not any("tensor" in path.name for path in result.iterdir())
        manifest = json.loads((result / "capture_manifest.json").read_text())
        assert manifest["claim_boundary"]["hardware_quantization_authority"] is False
        assert manifest["claim_boundary"]["tsbg_exact_scope"] == (
            "captured_codeword_and_contributor_only")

        gate_out = root / "future"
        ok = M.preflight_before_checkpoint_load(
            gate_out, 1024, free_bytes=M.MIN_FREE_AFTER_BYTES + 1024)
        assert ok["checkpoint_loaded"] is False
        rejects(lambda: M.preflight_before_checkpoint_load(
            gate_out, M.MAX_ESTIMATED_BYTES + 1,
            free_bytes=M.MAX_ESTIMATED_BYTES + M.MIN_FREE_AFTER_BYTES + 1))
        attacks.append("estimate_12gib")
        rejects(lambda: M.preflight_before_checkpoint_load(
            gate_out, 1024, free_bytes=M.MIN_FREE_AFTER_BYTES + 1023))
        attacks.append("free_16gib")
        gate_out.mkdir()
        rejects(lambda: M.preflight_before_checkpoint_load(
            gate_out, 1024, free_bytes=M.MIN_FREE_AFTER_BYTES + 1024))
        attacks.append("fresh_namespace")

        samples = M.verify_bindings()
        runtime_row = dict(samples["samples"][0], cohort="c1", path="x",
                           resolved_path="/x", bytes=1)
        assert M.project_m1434_sample(runtime_row) == samples["samples"][0]
        model = small_model()
        producer = M.SparseCaptureProducer(
            model, M.SyntheticTokenAdapter(), root / "attack_order",
            small_specs(), samples)
        rejects(lambda: model.modules["fixture.patch"].fire(
            FakeTensor((1, 1, 4, 1, 2), [[0] * 4, [0] * 4])))
        attacks.append("hook_outside_sample")
        wrong = dict(samples["samples"][0]); wrong["sample_key"] = "wrong"
        rejects(lambda: producer.begin_sample(wrong)); attacks.append("sample_identity")
        producer.begin_sample(samples["samples"][0])
        rejects(lambda: model.modules["fixture.mlp.fc1"].fire(
            FakeTensor((1, 1, 1, 2, 4), [[0] * 4, [0] * 4])))
        attacks.append("hook_order")

        missing = FakeModel({"fixture.patch": FakeModule(4, 4),
                             "fixture.mlp.fc1": FakeModule(4, 4)})
        rejects(lambda: M.build_layer_rows(missing, small_specs()))
        attacks.append("missing_real_hook")
        drift = small_model(); drift.modules["fixture.mlp.fc2"].m1552_input_channels = 5
        rejects(lambda: M.build_layer_rows(drift, small_specs()))
        attacks.append("channel_drift")

    rejects(M.production_release); attacks.append("production_release")
    rejects(lambda: M.encode_group([128])); attacks.append("code_range")
    assert M.encode_group([0, 0, 0, 0]) is None
    encoded = M.encode_group([1, -2, 0, 3])
    assert encoded == {"valid_channels": 4, "support_hex": "0b",
                       "sign_hex": "02", "nonunit_hex": "0a",
                       "nonzero_codes_le_hex": "01fe03"}
    assert len(attacks) == 10
    print("PASS M1552 source tests attacks=10 synthetic_samples=40 tokens=240 no_gpu=1 no_capture=1")


if __name__ == "__main__":
    main()

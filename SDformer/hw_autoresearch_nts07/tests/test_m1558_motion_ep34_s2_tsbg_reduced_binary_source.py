#!/usr/bin/env python3
"""Synthetic roundtrip and M1574 permit-provenance attacks for M1558 source."""

from __future__ import print_function

import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
SPEC = importlib.util.spec_from_file_location("m1558_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Handle(object):
    def __init__(self, hooks, hook):
        self.hooks = hooks; self.hook = hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class FakeModule(object):
    def __init__(self, inputs, outputs, beta=None):
        self.m1552_input_channels = int(inputs)
        self.m1552_output_channels = int(outputs)
        tiles = (int(outputs) + M.OUTPUT_TILE_WIDTH - 1) // M.OUTPUT_TILE_WIDTH
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
    except (M.M1558Error, AssertionError, AttributeError, TypeError,
            ValueError, zlib.error):
        return
    raise AssertionError("attack accepted")


def fake_specs(sample_count=3):
    return [
        {"layer_id": 0, "target": "PATCH", "module_name": "fixture.patch",
         "operator": "Conv2d", "operator_order": 0,
         "input_shape": (1, 1, 4, 1, 2), "output_shape": (1, 1, 4, 1, 2),
         "channel_axis": 2, "input_channels": 4, "output_channels": 4,
         "tokens_per_call": 2, "tokens_s40": 2 * sample_count,
         "input_elements_s40": 8 * sample_count,
         "input_active_s40": 3 * sample_count},
        {"layer_id": 1, "target": "FC1", "module_name": "fixture.mlp.fc1",
         "operator": "Linear", "operator_order": 1,
         "input_shape": (1, 1, 1, 3, 5), "output_shape": (1, 1, 1, 3, 4),
         "channel_axis": 4, "input_channels": 5, "output_channels": 4,
         "tokens_per_call": 3, "tokens_s40": 3 * sample_count,
         "input_elements_s40": 15 * sample_count,
         "input_active_s40": 6 * sample_count},
        {"layer_id": 2, "target": "FC2", "module_name": "fixture.mlp.fc2",
         "operator": "Linear", "operator_order": 2,
         "input_shape": (1, 1, 1, 3, 5), "output_shape": (1, 1, 1, 3, 4),
         "channel_axis": 4, "input_channels": 5, "output_channels": 4,
         "tokens_per_call": 3, "tokens_s40": 3 * sample_count,
         "input_elements_s40": 15 * sample_count,
         "input_active_s40": 6 * sample_count},
    ]


def fake_model():
    return FakeModel({"fixture.patch": FakeModule(4, 4, [3]),
                      "fixture.mlp.fc1": FakeModule(5, 4, [2]),
                      "fixture.mlp.fc2": FakeModule(5, 4, [2])})


def sample_order(count=3):
    authority = M.M1552.verify_bindings()
    return dict(authority, samples=list(authority["samples"][:count]))


def tensors():
    patch = FakeTensor((1, 1, 4, 1, 2),
                       [[0, 0, 0, 0], [1, -2, 0, 3]])
    fc = FakeTensor((1, 1, 1, 3, 5),
                    [[0, 0, 0, 0, 0], [1, -2, 0, 3, 0],
                     [1, 0, -1, 0, 2]])
    return patch, fc


def permit_for(root, specs, samples):
    estimate = M.estimate_from_specs(specs, len(samples["samples"]))
    free = estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES + 1
    return M.issue_synthetic_permit(root, specs, len(samples["samples"]), free)


def forge_without_constructor(cls, output, inventory, estimate, free_bytes):
    """Populate every writable slot without invoking the private constructor."""
    permit = object.__new__(cls)
    prefix = "_{}__".format(cls.__name__)
    object.__setattr__(permit, prefix + "output", str(Path(output).resolve()))
    object.__setattr__(permit, prefix + "inventory", str(inventory))
    object.__setattr__(permit, prefix + "estimate", dict(estimate))
    object.__setattr__(permit, prefix + "free", int(free_bytes))
    object.__setattr__(permit, prefix + "consumed", False)
    return permit


def run_valid(root):
    specs = fake_specs(); samples = sample_order(); model = fake_model()
    permit = permit_for(root, specs, samples)
    producer = M.ReducedBinaryProducer(
        model, M.SyntheticBinaryAdapter(), root, specs, samples, permit,
        production_inventory=False)
    patch, fc = tensors()
    for sample in samples["samples"]:
        producer.begin_sample(sample)
        model.modules["fixture.patch"].fire(patch)
        model.modules["fixture.mlp.fc1"].fire(fc)
        model.modules["fixture.mlp.fc2"].fire(fc)
        producer.end_sample()
    result = producer.finalize_source_result()
    return M.validate_binary_result(result, specs, samples), result, specs, samples


def reseal(root):
    sums = root / "SHA256SUMS"
    rows = []
    for line in sums.read_text().splitlines():
        _digest, name = line.split(None, 1)
        name = name.strip()
        rows.append("{}  {}".format(M.sha256(root / name), name))
    sums.write_text("\n".join(rows) + "\n")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(M.sha256(sums)))


def first_frame(root):
    with (root / "fc_frames.bin").open("rb") as stream:
        header = stream.read(M.FRAME_HEADER.size)
        values = M.FRAME_HEADER.unpack(header)
        compressed = stream.read(values[-2])
    raw = zlib.decompress(compressed)
    return values, raw


def main():
    attacks = []
    source = M.source_self_check()
    assert source["layers"] == 32 and source["FC_layers"] == 24
    assert source["fc_tokens"] == 44640000
    assert source["patch_tokens_histogram_only"] == 430080000
    assert source["raw_fc_payload_upper_bytes"] == 7528535874
    description = M.describe()
    assert description["preload"]["raw_upper_bytes"] == 7528535874
    assert description["execution"] == {
        "gpu": False, "ssh": False, "capture": False,
        "release": False, "automatic_retry": False}
    text = SOURCE.read_text()
    assert "token_source_groups" not in text and ".tolist()" not in text
    assert "production_release" in text
    assert not hasattr(M, "_mint_permit")
    assert not hasattr(M, "_checked_issue_permit")
    assert "def mint(" not in text
    assert M._ProductionPreloadPermit is not M._SyntheticPreloadPermit
    assert list(inspect.signature(M.issue_preload_permit).parameters) == ["output"]
    assert list(inspect.signature(M._issue_production_permit).parameters) == ["output"]
    assert "production_minted = {}" in text
    assert "synthetic_minted = {}" in text
    assert "production_minted.pop(self)" in text
    assert "synthetic_minted.pop(self)" in text

    with tempfile.TemporaryDirectory(prefix="m1558_test.") as directory:
        base = Path(directory)
        validated, result, specs, samples = run_valid(base / "valid")
        assert validated == {
            "status": "PASS_M1558_INCREMENTAL_BINARY_VALIDATION",
            "frames": 6, "fc_tokens": 18, "zero_fc_tokens": 6,
            "nonzero_codes": 36, "patch_histogram_rows": 3,
            "hardware_quantization_authority": False}
        manifest = json.loads((result / "capture_manifest.json").read_text())
        assert manifest["encoding"]["patch_per_token_payload"] is False
        assert manifest["claim_boundary"]["hardware_quantization_authority"] is False
        assert manifest["claim_boundary"]["tsbg_exact_scope"] == (
            "captured_codeword_and_contributor_only")
        assert not any("token" in path.name for path in result.iterdir())

        values, raw = first_frame(result)
        decoded = M.decode_frame_payload(
            raw, values[7], values[8], values[9], values[10], return_codes=True)
        assert decoded["codes"].tolist() == [
            [0, 0, 0, 0, 0], [1, -2, 0, 3, 0], [1, 0, -1, 0, 2]]
        attacks.append("binary_roundtrip_zero_tail_sign_nonunit")

        estimate = M.estimate_from_specs(specs, 3)
        strict_free = estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES
        rejects(lambda: M.issue_synthetic_permit(
            base / "free_equal", specs, 3, strict_free))
        attacks.append("free_after_strict_gt_16gib")

        # Production has no caller-controlled free-space/inventory/sample seam.
        rejects(lambda: M.issue_preload_permit(
            base / "public_override", strict_free + 1))
        attacks.append("production_public_signature_no_free_override")
        rejects(lambda: M._issue_production_permit(
            base / "private_override", strict_free + 1))
        attacks.append("production_private_signature_no_free_override")

        production_specs = M.frozen_layer_specs()
        production_estimate = M.estimate_from_specs(production_specs, 40)
        production_equal = (production_estimate["result_upper_bytes"] +
                            M.MIN_FREE_AFTER_BYTES)
        disk_calls = []
        original_disk_usage = M.shutil.disk_usage

        class DiskResult(object):
            def __init__(self, free):
                self.free = int(free)

        def equal_disk(path):
            disk_calls.append(str(Path(path).resolve()))
            return DiskResult(production_equal)

        M.shutil.disk_usage = equal_disk
        try:
            rejects(lambda: M.issue_preload_permit(base / "production_equal"))
        finally:
            M.shutil.disk_usage = original_disk_usage
        assert disk_calls == [str(base.resolve())]
        attacks.append("production_real_disk_equal_free_rejected")

        disk_calls = []
        production_required = production_equal + 1

        def sufficient_disk(path):
            disk_calls.append(str(Path(path).resolve()))
            return DiskResult(production_required)

        M.shutil.disk_usage = sufficient_disk
        try:
            production_root = base / "production_real_disk"
            production_permit = M.issue_preload_permit(production_root)
        finally:
            M.shutil.disk_usage = original_disk_usage
        assert disk_calls == [str(base.resolve())]
        assert type(production_permit) is M._ProductionPreloadPermit
        production_receipt = production_permit.consume(
            production_root, M.canonical_sha(production_specs))
        assert production_receipt["provenance"] == M.PRODUCTION_PROVENANCE
        assert production_receipt["free_bytes_before"] == production_required
        assert production_receipt["free_bytes_after_upper"] == (
            M.MIN_FREE_AFTER_BYTES + 1)
        attacks.append("production_real_disk_only_and_provenance")

        exact_synthetic_root = base / "exact_synthetic"
        exact_synthetic = M.issue_synthetic_permit(
            exact_synthetic_root, production_specs, 40, production_required)
        assert type(exact_synthetic) is M._SyntheticPreloadPermit
        rejects(lambda: M.ReducedBinaryProducer(
            object(), object(), exact_synthetic_root, production_specs,
            {"samples": []}, exact_synthetic, production_inventory=True))
        attacks.append("production_rejects_exact_inventory_synthetic_permit")

        production_for_synthetic_root = base / "production_for_synthetic"
        M.shutil.disk_usage = sufficient_disk
        try:
            production_for_synthetic = M.issue_preload_permit(
                production_for_synthetic_root)
        finally:
            M.shutil.disk_usage = original_disk_usage
        rejects(lambda: M.ReducedBinaryProducer(
            object(), object(), production_for_synthetic_root, production_specs,
            {"samples": []}, production_for_synthetic,
            production_inventory=False))
        attacks.append("synthetic_rejects_production_permit")

        rejects(lambda: setattr(exact_synthetic, "provenance",
                                M.PRODUCTION_PROVENANCE))
        attacks.append("permit_provenance_not_caller_mutable")
        rejects(lambda: M._ProductionPreloadPermit(
            base / "direct_production", M.canonical_sha(production_specs),
            production_estimate, production_required, object()))
        attacks.append("production_constructor_private")
        rejects(lambda: M._SyntheticPreloadPermit(
            base / "direct_synthetic", M.canonical_sha(specs), estimate,
            strict_free + 1, object()))
        attacks.append("synthetic_constructor_private")

        forged_production_root = base / "forged_production_object_new"
        forged_production = forge_without_constructor(
            M._ProductionPreloadPermit, forged_production_root,
            M.canonical_sha(production_specs), production_estimate,
            production_required)
        rejects(lambda: forged_production.consume(
            forged_production_root, M.canonical_sha(production_specs)))
        attacks.append("object_new_slot_filled_production_not_minted")

        forged_synthetic_root = base / "forged_synthetic_object_new"
        forged_synthetic = forge_without_constructor(
            M._SyntheticPreloadPermit, forged_synthetic_root,
            M.canonical_sha(specs), estimate, strict_free + 1)
        rejects(lambda: forged_synthetic.consume(
            forged_synthetic_root, M.canonical_sha(specs)))
        attacks.append("object_new_slot_filled_synthetic_not_minted")

        # A real production issuer query remains the source of the receipt.
        # Caller-writable handle slots are deliberately not trusted after mint.
        actual_queries = []

        def actual_spy(path):
            value = original_disk_usage(path)
            actual_queries.append(value)
            return value

        M.shutil.disk_usage = actual_spy
        try:
            actual_root = base / "actual_disk_registry_receipt"
            actual_permit = M.issue_preload_permit(actual_root)
        finally:
            M.shutil.disk_usage = original_disk_usage
        prefix = "_{}__".format(type(actual_permit).__name__)
        object.__setattr__(actual_permit, prefix + "free", 1 << 62)
        object.__setattr__(actual_permit, prefix + "output",
                           str((base / "caller_slot_rewrite").resolve()))
        actual_receipt = actual_permit.consume(
            actual_root, M.canonical_sha(production_specs))
        assert len(actual_queries) == 1
        assert actual_receipt["free_bytes_before"] == int(actual_queries[0].free)
        assert actual_receipt["output"] == str(actual_root.resolve())
        rejects(lambda: actual_permit.consume(
            actual_root, M.canonical_sha(production_specs)))
        attacks.append("registry_record_preserves_real_disk_receipt_and_one_shot")

        huge = [dict(row) for row in specs]
        huge[1]["input_active_s40"] = M.MAX_RUNTIME_BYTES
        rejects(lambda: M.estimate_from_specs(huge, 3))
        attacks.append("estimate_under_12gib")
        occupied = base / "occupied"; occupied.mkdir()
        rejects(lambda: M.issue_synthetic_permit(
            occupied, specs, 3, M.MAX_RUNTIME_BYTES + M.MIN_FREE_AFTER_BYTES))
        attacks.append("fresh_namespace")

        wrong_root = base / "permit_a"
        permit = permit_for(wrong_root, specs, samples)
        rejects(lambda: M.ReducedBinaryProducer(
            fake_model(), M.SyntheticBinaryAdapter(), base / "permit_b",
            specs, samples, permit))
        attacks.append("permit_path_binding")
        assert type(permit) is M._SyntheticPreloadPermit
        permit = permit_for(base / "reuse", specs, samples)
        permit.consume(base / "reuse", M.canonical_sha(specs))
        rejects(lambda: permit.consume(base / "reuse", M.canonical_sha(specs)))
        attacks.append("permit_one_shot")
        rejects(lambda: M.ReducedBinaryProducer(
            fake_model(), M.SyntheticBinaryAdapter(), base / "forged",
            specs, samples, object()))
        attacks.append("permit_exact_type")

        model = fake_model(); out = base / "order"
        producer = M.ReducedBinaryProducer(
            model, M.SyntheticBinaryAdapter(), out, specs, samples,
            permit_for(out, specs, samples))
        patch, fc = tensors()
        rejects(lambda: model.modules["fixture.patch"].fire(patch))
        attacks.append("hook_outside_sample")
        wrong = dict(samples["samples"][0]); wrong["sample_key"] = "wrong"
        rejects(lambda: producer.begin_sample(wrong))
        attacks.append("sample_identity")
        producer.begin_sample(samples["samples"][0])
        rejects(lambda: model.modules["fixture.mlp.fc1"].fire(fc))
        attacks.append("hook_order")

        drift_specs = fake_specs(); drift = fake_model()
        drift.modules["fixture.mlp.fc2"].m1552_input_channels = 6
        drift_out = base / "drift"
        rejects(lambda: M.ReducedBinaryProducer(
            drift, M.SyntheticBinaryAdapter(), drift_out, drift_specs, samples,
            permit_for(drift_out, drift_specs, samples)))
        attacks.append("module_dimensions")

        bad_raw = bytearray(raw)
        bad_raw[0] |= 0x80
        rejects(lambda: M.decode_frame_payload(
            bytes(bad_raw), values[7], values[8], values[9], values[10]))
        attacks.append("tail_bit")
        matrix_bytes = values[7] * values[9]
        bad_raw = bytearray(raw); bad_raw[3 * matrix_bytes] += 1
        rejects(lambda: M.decode_frame_payload(
            bytes(bad_raw), values[7], values[8], values[9], values[10]))
        attacks.append("nnz_support")
        bad_raw = bytearray(raw); bad_raw[matrix_bytes] |= 0x02
        rejects(lambda: M.decode_frame_payload(
            bytes(bad_raw), values[7], values[8], values[9], values[10]))
        attacks.append("sign_support_code")
        bad_raw = bytearray(raw); bad_raw[2 * matrix_bytes] ^= 0x02
        rejects(lambda: M.decode_frame_payload(
            bytes(bad_raw), values[7], values[8], values[9], values[10]))
        attacks.append("nonunit_code")

        truncated = base / "truncated"; shutil.copytree(result, truncated)
        data = (truncated / "fc_frames.bin").read_bytes()
        (truncated / "fc_frames.bin").write_bytes(data[:-1]); reseal(truncated)
        rejects(lambda: M.validate_binary_result(truncated, specs, samples))
        attacks.append("frame_truncation")
        bad_header = base / "header"; shutil.copytree(result, bad_header)
        data = bytearray((bad_header / "fc_frames.bin").read_bytes())
        data[0] ^= 1; (bad_header / "fc_frames.bin").write_bytes(bytes(data))
        reseal(bad_header)
        rejects(lambda: M.validate_binary_result(bad_header, specs, samples))
        attacks.append("frame_header")
        bad_size = base / "size"; shutil.copytree(result, bad_size)
        data = bytearray((bad_size / "fc_frames.bin").read_bytes())
        fields = list(M.FRAME_HEADER.unpack(bytes(data[:M.FRAME_HEADER.size])))
        fields[12] = 0xffffffff
        data[:M.FRAME_HEADER.size] = M.FRAME_HEADER.pack(*fields)
        (bad_size / "fc_frames.bin").write_bytes(bytes(data)); reseal(bad_size)
        rejects(lambda: M.validate_binary_result(bad_size, specs, samples))
        attacks.append("frame_size_bound")
        bad_layers = base / "layers"; shutil.copytree(result, bad_layers)
        layer_data = json.loads((bad_layers / "layers.json").read_text())
        layer_data["layers"][1]["channel_axis"] = 0
        (bad_layers / "layers.json").write_text(
            json.dumps(layer_data, indent=2, sort_keys=True) + "\n")
        reseal(bad_layers)
        rejects(lambda: M.validate_binary_result(bad_layers, specs, samples))
        attacks.append("static_axis_inventory")

        budget = M.RuntimeBudget(10); budget.charge(5, 5)
        rejects(lambda: budget.charge(5, 0))
        attacks.append("runtime_hard_cap")

    rejects(M.production_release); attacks.append("production_release")
    assert len(attacks) == 33
    print("PASS M1582 minted-registry successor attacks=33 frames=6 "
          "fc_tokens=18 patch_rows=3 production_real_disk=1 "
          "real_free={} receipt_free={} ".format(
              int(actual_queries[0].free), actual_receipt["free_bytes_before"]) +
          "object_new_rejected=2 duplicate_consume_rejected=2 "
          "distinct_permit_registries=1 no_gpu=1 no_capture=1")


if __name__ == "__main__":
    main()

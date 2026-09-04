#!/usr/bin/env python3
"""Independent source hammer for the exact M1552 compact producer bytes.

Synthetic files only.  No checkpoint load, CUDA/GPU, SSH, capture release,
production attempt, or remote action is performed.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1552_motion_ep34_s2_tsbg_incremental_source_r1.py")
TEST = HW / "tests/test_m1552_motion_ep34_s2_tsbg_incremental_source.py"
CONTRACT = HW / "contracts/m1552_motion_ep34_s2_tsbg_incremental_producer_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1552_motion_ep34_s2_tsbg_incremental_producer_source_author_receipt_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "245d65c98f893811a48c36050d764f5269b45a0657f6118eb33a1e16f2192c94",
    TEST: "b37f438d02c9d5b86fded2176161a127e9743d47b2c3b46dc020e798e9d8c26f",
    CONTRACT: "32453aeaec33e89d369f144de6e780a5068d37f1e2ccf9e4c7f5993d5be8866c",
    AUTHOR / "review.json": "21a01215d4223e8a33735757a2d8fc75e81c5e86329efdd677196f32961ad06a",
    AUTHOR / "SHA256SUMS": "3d0ad0b545322f034559feea10049573d712e82bebb09896657370342e53d354",
    AUTHOR / "SHA256SUMS.seal.sha256": "e2ddeb34b84a156ae617de1f8a8b8275880982c9e7c2aaf6e72089d2d4323e9a",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject(name, function, attacks):
    try:
        function()
    except Exception:
        attacks.append(name)
        return
    raise AssertionError("attack accepted: " + name)


def verify_author_seal():
    assert (AUTHOR / "SHA256SUMS.seal.sha256").read_text().split() == [
        EXPECTED[AUTHOR / "SHA256SUMS"], "SHA256SUMS"]
    rows = {}
    for line in (AUTHOR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in rows and "/" not in name and ".." not in name
        assert sha256(AUTHOR / name) == digest
        rows[name] = digest
    actual = set(item.name for item in AUTHOR.iterdir()
                 if item.is_file() and item.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    assert actual == set(rows) and len(rows) == 4


for path, digest in EXPECTED.items():
    assert path.is_file() and sha256(path) == digest, "identity drift: " + str(path)
verify_author_seal()

spec = importlib.util.spec_from_file_location("m1555_bound_m1552", str(SOURCE))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
test_spec = importlib.util.spec_from_file_location("m1555_bound_m1552_test", str(TEST))
T = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(T)


class Scalar(object):
    def __init__(self, value):
        self.value = float(value)

    def item(self):
        return self.value


class Weight(object):
    """Tiny output-major fake implementing only the beta audit operations."""
    def __init__(self, values):
        self.values = list(float(value) for value in values)

    def detach(self):
        return self

    def __getitem__(self, key):
        return Weight(self.values[key])

    def abs(self):
        return Weight([abs(value) for value in self.values])

    def max(self):
        if not self.values:
            raise RuntimeError("empty output slice")
        return Scalar(max(self.values))


class WeightedModule(object):
    def __init__(self, kind, inputs, outputs, values):
        if kind == "Linear":
            self.in_features = inputs; self.out_features = outputs
        else:
            self.in_channels = inputs; self.out_channels = outputs
        self.weight = Weight(values)


def mutate_coordinate_and_reseal(root):
    token_path = root / "token_source_groups.jsonl.zlib"
    rows = zlib.decompress(token_path.read_bytes()).splitlines()
    first = json.loads(rows[0].decode("utf-8"))
    first["spatial_x"] = 999999999
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token_path.write_bytes(zlib.compress(b"\n".join(rows) + b"\n", 9))
    sums = root / "SHA256SUMS"
    replaced = []
    for line in sums.read_text().splitlines():
        _digest, name = line.split("  ", 1)
        replaced.append("{}  {}".format(sha256(root / name), name))
    sums.write_text("\n".join(replaced) + "\n")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums)))


def main():
    attacks = []
    source = M.source_self_check()
    assert source == {"status": "PASS_M1552_SOURCE_SELF_CHECK__NO_GPU_NO_CAPTURE",
                      "layers": 32, "samples": 40,
                      "hardware_quantization_authority": False}
    specs = M.frozen_layer_specs()
    assert len(specs) == 32
    assert [row["layer_id"] for row in specs] == list(range(32))
    assert [row["operator_order"] for row in specs] == sorted(
        row["operator_order"] for row in specs)
    assert {key: sum(row["target"] == key for row in specs)
            for key in M.TARGET_COUNTS} == {"FC1": 12, "FC2": 12, "PATCH": 8}
    assert len(set(row["module_name"] for row in specs)) == 32
    assert all(row["input_channels"] == row["input_shape"][row["channel_axis"]]
               for row in specs)
    assert all(row["output_channels"] == row["output_shape"][row["channel_axis"]]
               for row in specs)

    formal_tokens = 0
    target_tokens = dict((key, 0) for key in M.TARGET_COUNTS)
    for row in specs:
        tokens = 1
        for axis, value in enumerate(row["input_shape"]):
            if axis != row["channel_axis"]:
                tokens *= int(value)
        formal_tokens += tokens * 40
        target_tokens[row["target"]] += tokens * 40
    assert formal_tokens == 474720000
    assert target_tokens == {"PATCH": 430080000, "FC1": 22320000,
                             "FC2": 22320000}

    source_text = SOURCE.read_text()
    assert "import torch" not in source_text and "import numpy" not in source_text
    assert "chunk = value[begin:begin + int(wanted)].to(device=\"cpu\")" in source_text
    assert "value = tensor.detach()" in source_text
    assert "value.to(device=\"cpu\")" not in source_text
    assert "yield codes.tolist()" in source_text
    assert "self.tokens.write({" in source_text
    assert M.TOKEN_CHUNK == 4096

    assert M.token_coordinates({"input_shape": (2, 1, 4, 3, 5),
                                "channel_axis": 2}, 29) == (1, 2, 4)
    assert M.token_coordinates({"input_shape": (2, 1, 3, 5, 4),
                                "channel_axis": 4}, 29) == (1, 2, 4)
    assert M.token_coordinates({"input_shape": (2, 4, 3, 5),
                                "channel_axis": 1}, 29) == (1, 2, 4)
    assert M.encode_group([0] * 16) is None
    assert M.encode_group([1, -2, 0, 3]) == {
        "valid_channels": 4, "support_hex": "0b", "sign_hex": "02",
        "nonunit_hex": "0a", "nonzero_codes_le_hex": "01fe03"}
    reject("int8_high", lambda: M.encode_group([128]), attacks)
    reject("int8_low", lambda: M.encode_group([-129]), attacks)

    linear = WeightedModule("Linear", 384, 192,
                            [0.25] * 96 + [2.01] * 96)
    conv = WeightedModule("Conv2d", 48, 192,
                          [0.10] * 96 + [1.01] * 96)
    assert M.module_dimensions(linear) == (384, 192)
    assert M.module_dimensions(conv) == (48, 192)
    assert M.weight_beta_by_tile(linear, 192) == [1, 3]
    assert M.weight_beta_by_tile(conv, 192) == [1, 2]

    with tempfile.TemporaryDirectory(prefix="m1555_m1552.", dir=str(HERE)) as directory:
        root = Path(directory)
        future = root / "future"
        ok = M.preflight_before_checkpoint_load(
            future, M.MAX_ESTIMATED_BYTES,
            free_bytes=M.MAX_ESTIMATED_BYTES + M.MIN_FREE_AFTER_BYTES)
        assert ok["checkpoint_loaded"] is False
        reject("estimate_over_12gib", lambda: M.preflight_before_checkpoint_load(
            future, M.MAX_ESTIMATED_BYTES + 1,
            free_bytes=M.MAX_ESTIMATED_BYTES + M.MIN_FREE_AFTER_BYTES + 1), attacks)
        reject("free_below_16gib", lambda: M.preflight_before_checkpoint_load(
            future, 1, free_bytes=M.MIN_FREE_AFTER_BYTES), attacks)
        reject("negative_estimate", lambda: M.preflight_before_checkpoint_load(
            future, -1, free_bytes=M.MIN_FREE_AFTER_BYTES + 1), attacks)
        future.mkdir()
        reject("nonfresh_namespace", lambda: M.preflight_before_checkpoint_load(
            future, 1, free_bytes=M.MIN_FREE_AFTER_BYTES + 1), attacks)

        samples = M.verify_bindings()
        model = T.small_model()
        producer = M.SparseCaptureProducer(
            model, M.SyntheticTokenAdapter(), root / "order_attack",
            T.small_specs(), samples)
        reject("hook_before_begin", lambda: model.modules["fixture.patch"].fire(
            T.FakeTensor((1, 1, 4, 1, 2), [[0] * 4, [0] * 4])), attacks)
        wrong = dict(samples["samples"][0]); wrong["sha256"] = "0" * 64
        reject("sample_identity", lambda: producer.begin_sample(wrong), attacks)
        producer.begin_sample(samples["samples"][0])
        reject("hook_order", lambda: model.modules["fixture.mlp.fc1"].fire(
            T.FakeTensor((1, 1, 1, 2, 4), [[0] * 4, [0] * 4])), attacks)

        valid = root / "valid"
        validated, capture = T.run_valid(valid)
        assert validated["samples"] == 40 and validated["token_records"] == 240
        assert validated["cycles_admitted"] is False
        manifest = json.loads((capture / "capture_manifest.json").read_text())
        assert manifest["claim_boundary"]["hardware_quantization_authority"] is False
        assert manifest["claim_boundary"]["tsbg_exact_scope"] == (
            "captured_codeword_and_contributor_only")
        assert not any("tensor" in item.name for item in capture.iterdir())

        # The original M1544 validator is intentionally rerun.  It accepts a
        # structurally resealed coordinate mutation because layers.json omits
        # input shape/axis and the validator checks only non-negativity.  This
        # blocks capture/release, but not authoring the next integration source.
        mutate_coordinate_and_reseal(capture)
        coordinate_mutation_accepted = False
        try:
            M.load_validator().validate_capture(capture)
            coordinate_mutation_accepted = True
        except Exception:
            pass
        assert coordinate_mutation_accepted

        # The producer class itself has no preflight capability token.  A tiny
        # synthetic instance can be built without calling preflight.  Therefore
        # only a reviewed wrapper may enforce pre-checkpoint ordering.
        bypass_root = root / "preflight_helper_bypass"
        bypass = M.SparseCaptureProducer(
            T.small_model(), M.SyntheticTokenAdapter(), bypass_root,
            T.small_specs(), samples)
        preflight_is_helper_not_capability = bypass.root == bypass_root
        while bypass.handles:
            bypass.handles.pop().remove()
        bypass.tokens.close(); bypass.s1.close()
        assert preflight_is_helper_not_capability

    reject("production_release", M.production_release, attacks)
    for option in ("--capture", "--production", "--release"):
        reject("cli_" + option[2:], lambda option=option:
               subprocess.check_output([sys.executable, str(SOURCE), option],
                                       stderr=subprocess.STDOUT), attacks)
    author_test = subprocess.check_output([sys.executable, str(TEST)],
                                          stderr=subprocess.STDOUT).decode("utf-8")
    assert "PASS M1552 source tests attacks=10 synthetic_samples=40 tokens=240" in author_test

    assert len(attacks) == 13
    result = {
        "schema": "m1555_m1552_ep34_sparse_compact_producer_source_independent_hammer_output_r1_v1",
        "status": "NO_GO_AS_IS_REMOTE_PRODUCER_INTEGRATION__SUCCESSOR_REDESIGN_AUTHORING_ONLY__CAPTURE_FORBIDDEN",
        "python": sys.version.split()[0],
        "bindings": {
            "source_sha256": EXPECTED[SOURCE],
            "test_sha256": EXPECTED[TEST],
            "contract_sha256": EXPECTED[CONTRACT],
            "author_review_sha256": EXPECTED[AUTHOR / "review.json"],
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "verified": {
            "frozen_hook_count": len(specs),
            "target_counts": M.TARGET_COUNTS,
            "hook_inventory_sha256": M.canonical_sha(specs),
            "formal_token_records": formal_tokens,
            "formal_token_records_by_target": target_tokens,
            "full_cpu_tensor_materialized_by_adapter": False,
            "token_chunk": M.TOKEN_CHUNK,
            "token_coordinates_and_channel_axis": True,
            "zero_token_and_group_bitsets_and_signed_codes": True,
            "linear_and_conv_output_major_beta_direction": True,
            "s1_diagnostic_only": True,
            "hardware_quantization_authority": False,
            "original_m1544_validator_valid_synthetic": True,
            "ordinary_attacks_rejected": attacks,
        },
        "capture_blockers": {
            "python_jsonl_execution_feasibility_p0": "474.72M Python JSON objects and per-token json.dumps/zlib writes are not made feasible by the 12 GiB byte cap; 430.08M rows (90.60%) are PATCH",
            "tolist_execution_feasibility_p1": "chunk-local .tolist avoids a full CPU tensor but still creates Python integers/lists for every code and does not establish wall-time feasibility",
            "required_population_redesign": "For this deadline, PATCH should emit only streaming S1 histogram/debt and no per-token groups; retain FC1/FC2 for TSBG, and defer S2 PATCH token payload to paired-AEE integration. Even FC1+FC2 has 44.64M rows and should use strict binary/RLE rather than one JSON object per token.",
            "preflight_is_helper_not_capability": preflight_is_helper_not_capability,
            "coordinate_mutation_accepted_by_original_validator": coordinate_mutation_accepted,
            "validator_does_not_bind_exact_32_layer_inventory_or_input_shapes": True,
            "single_monolithic_zlib_validator_materializes_decompressed_stream": True,
            "required_integration_fix": "gate before importing/building model or CUDA, bind a signed estimate receipt, hard-bind exact 32-layer shape/axis inventory, and provide streaming/sharded validation before any capture release",
        },
        "authorization": {
            "exact_m1552_remote_producer_integration_source_authoring": False,
            "successor_reduced_population_or_binary_rle_integration_source_authoring": True,
            "checkpoint_load": False,
            "gpu": False,
            "ssh": False,
            "capture": False,
            "release": False,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "source_hammer": True, "opportunity": False, "cycles": False,
            "speedup": False, "traffic": False, "energy": False,
            "aee": False, "rtl": False, "paper_headline": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

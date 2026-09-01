#!/usr/bin/env python3
"""Local-only independent review of the M1558 reduced-binary source.

No checkpoint, GPU, SSH, capture, release, RTL, or EDA path is called.
Compatible with CPython 3.6.
"""
from __future__ import print_function

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
TEST = HW / "tests/test_m1558_motion_ep34_s2_tsbg_reduced_binary_source.py"
CONTRACT = HW / (
    "contracts/m1558_motion_ep34_s2_tsbg_reduced_binary_producer_source_"
    "contract_r1_20260901.json")
AUTHOR = HW / (
    "reviews/m1558_motion_ep34_s2_tsbg_reduced_binary_producer_source_"
    "author_receipt_r1_20260901")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINNED = {
    "source": "917cbcb3c4b7198678954ff8ecdc74e67c71dcb6cad2b6311bda32ee2f06b2ce",
    "test": "4baca3604f25a8a071bb0c4793b70a892eb6afabd7bc668f58bea43f646fc518",
    "contract": "cb422b6be28b097aee689af471c17c2923beb0dda6fe0d97482a6d09a6a89d5b",
    "author_review": "93d3143981b685ee7ae21cbf6f957d03435a1c41ee47a6c262fe2b44583f9f38",
    "author_manifest": "8af98a4200bbde26278b08ed5e528cef6df6bfeb284322ffac721aabdee31cd1",
    "author_outer": "2879c85603664ca65e80f65951bdc31282322954c991f808e0a6ed5b0dc5de8a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_author_seal():
    manifest = AUTHOR / "SHA256SUMS"
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    require(sha256(manifest) == PINNED["author_manifest"] and
            sha256(outer) == PINNED["author_outer"] and
            outer.read_text(encoding="ascii").split() ==
            [PINNED["author_manifest"], "SHA256SUMS"],
            "author outer seal drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in expected,
                "author manifest malformed")
        expected[fields[1]] = fields[0]
    actual = set(path.name for path in AUTHOR.iterdir()
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(expected), "author seal coverage drift")
    for name, digest in expected.items():
        require("/" not in name and ".." not in name and
                sha256(AUTHOR / name) == digest,
                "author sealed member drift: " + name)
    return len(expected)


def run_json(arguments):
    output = subprocess.check_output(
        [sys.executable] + [str(value) for value in arguments],
        stderr=subprocess.STDOUT).decode("utf-8")
    return json.loads(output)


def rejected(function):
    try:
        function()
    except Exception:
        return True
    return False


def main():
    inputs = {
        "source": sha256(SOURCE), "test": sha256(TEST),
        "contract": sha256(CONTRACT),
        "author_review": sha256(AUTHOR / "review.json"),
        "author_manifest": sha256(AUTHOR / "SHA256SUMS"),
        "author_outer": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
        "docs359": sha256(DOC359),
    }
    require(inputs == PINNED, "pinned M1558 input drift")
    author_sealed_members = verify_author_seal()
    module = load(SOURCE, "m1560_bound_m1558")
    test_module = load(TEST, "m1560_bound_m1558_test")

    self_check = module.source_self_check()
    description = module.describe()
    specs = module.frozen_layer_specs()
    estimate = module.estimate_from_specs(specs)
    target_counts = dict((target, sum(row["target"] == target for row in specs))
                         for target in ("FC1", "FC2", "PATCH"))
    require(len(specs) == 32 and target_counts ==
            {"FC1": 12, "FC2": 12, "PATCH": 8} and
            len(estimate["layers"]) == 24 and
            module.canonical_sha(specs) ==
            "726a6a8fe25aa1c33f95eeb91eec8d9fb1ce4cd61376c47d438e6e2711fc9979" and
            [int(row["layer_id"]) for row in specs] == list(range(32)) and
            [int(row["operator_order"]) for row in specs] == sorted(
                int(row["operator_order"]) for row in specs),
            "exact 24 FC + 8 PATCH inventory drift")
    require(estimate["fc_tokens"] == 44640000 and
            estimate["patch_tokens_histogram_only"] == 430080000 and
            estimate["raw_fc_payload_upper_bytes"] == 7528535874 and
            estimate["zlib_payload_upper_bytes"] == 7531010264 and
            estimate["frame_header_upper_bytes"] == 618240 and
            estimate["result_upper_bytes"] == 7598737368 and
            estimate["binary_frame_upper_count"] == 11040 and
            estimate["result_upper_bytes"] < module.MAX_RUNTIME_BYTES,
            "first-principles population/estimate drift")
    require(self_check["status"] ==
            "PASS_M1558_SOURCE_SELF_CHECK__NO_GPU_NO_CAPTURE" and
            self_check["hardware_quantization_authority"] is False and
            description["population"] == {
                "FC1_FC2_tokens": 44640000,
                "PATCH_tokens_histogram_only": 430080000} and
            description["preload"]["raw_upper_bytes"] == 7528535874 and
            description["preload"]["strict_max_bytes"] ==
            12 * 1024 * 1024 * 1024 and
            description["quantization"]["hardware_authority"] is False and
            all(description["execution"][key] is False for key in
                ("gpu", "ssh", "capture", "release", "automatic_retry")),
            "source description/claim boundary drift")

    # Independent mixed-code frame roundtrip: zero token, little-endian tail,
    # sign, unit/nonunit, uint16 nnz, zlib extent and CRC.
    import numpy as np
    codes = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 2, -3, 0, 127, -128, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    raw, nnz, bitrow = module.encode_frame_payload(codes, 11)
    decoded = module.decode_frame_payload(
        raw, 3, 11, bitrow, nnz, return_codes=True)
    require(np.array_equal(decoded["codes"], codes) and
            decoded["zero_tokens"] == 2 and
            decoded["nonzero_codes"] == 6 and
            decoded["nonunit_codes"] == 4 and bitrow == 2,
            "mixed-code binary roundtrip drift")

    with tempfile.TemporaryDirectory(prefix="m1560_m1558_local.") as directory:
        base = Path(directory)
        budget = module.RuntimeBudget(1024 * 1024)
        writer = module.BinaryFrameWriter(base / "one_frame.bin", budget)
        writer.write(7, 3, 0, 0, codes)
        writer.close()
        frame = (base / "one_frame.bin").read_bytes()
        header = module.FRAME_HEADER.unpack(frame[:module.FRAME_HEADER.size])
        require(header[0] == module.FRAME_MAGIC and
                header[1] == module.FRAME_VERSION and
                header[2] == module.FRAME_HEADER.size and
                header[3:7] == (7, 3, 0, 0) and header[7] == 3 and
                header[8] == 11 and header[9] == 2 and header[10] == nnz and
                header[11] == len(raw) and
                len(frame) == module.FRAME_HEADER.size + header[12],
                "binary fixed header drift")
        decoder = zlib.decompressobj()
        expanded = decoder.decompress(frame[module.FRAME_HEADER.size:]) + decoder.flush()
        require(decoder.eof and not decoder.unused_data and
                not decoder.unconsumed_tail and expanded == raw and
                (zlib.crc32(expanded) & 0xffffffff) == header[13],
                "independent zlib/CRC/extent drift")

        # Full local synthetic producer/parser path from the pinned fixture.
        validated, result_root, fake_specs, sample_order = test_module.run_valid(
            base / "valid")
        require(validated == {
            "status": "PASS_M1558_INCREMENTAL_BINARY_VALIDATION",
            "frames": 6, "fc_tokens": 18, "zero_fc_tokens": 6,
            "nonzero_codes": 36, "patch_histogram_rows": 3,
            "hardware_quantization_authority": False},
            "full synthetic parser/canonical-order drift")
        manifest = json.loads(
            (result_root / "capture_manifest.json").read_text(encoding="utf-8"))
        patch_raw = zlib.decompress(
            (result_root / "patch_s1_histogram_debt.jsonl.zlib").read_bytes())
        patch_rows = [json.loads(line.decode("utf-8"))
                      for line in patch_raw.splitlines() if line]
        require(len(patch_rows) == 3 and
                all(row["per_token_payload_emitted"] is False
                    for row in patch_rows) and
                manifest["encoding"]["patch_per_token_payload"] is False and
                manifest["encoding"]["zero_fc_tokens_retained"] is True and
                manifest["encoding"]["canonical_token_order"] is True and
                manifest["claim_boundary"]["hardware_quantization_authority"]
                    is False,
                "PATCH hist-only/manifest boundary drift")

        # The exact-type check exists, but its private mint closure is exported
        # as a module-global callable. This bypasses the disk-reserve issuer.
        forged_output = base / "forged_permit_output"
        forged_inventory = "independent-forgery-proof"
        forged = module._mint_permit(
            forged_output, forged_inventory,
            {"result_upper_bytes": 1, "sample_count": 1}, 0)
        require(type(forged) is module._PreloadPermit,
                "direct mint did not create exact producer permit type")
        forged_receipt = forged.consume(forged_output, forged_inventory)
        require(forged_receipt["consumed"] is True and
                forged_receipt["free_bytes_before"] == 0 and
                forged_receipt["free_bytes_after_upper"] == -1,
                "direct mint bypass was not demonstrated")

    # Runtime raw and disk counters are strict: equality with the cap rejects.
    cap = module.RuntimeBudget(10)
    cap.charge(9, 9)
    require(rejected(lambda: cap.charge(1, 0)) and
            rejected(lambda: cap.charge(0, 1)),
            "runtime hard cap is not strict")
    require("permit" in inspect.signature(
        module.ReducedBinaryProducer).parameters and
        inspect.signature(module.ReducedBinaryProducer).parameters[
            "permit"].default is inspect.Parameter.empty,
            "producer permit parameter is not mandatory")

    author_output = subprocess.check_output(
        [sys.executable, str(TEST)], stderr=subprocess.STDOUT).decode("utf-8")
    require("PASS M1558 reduced-binary source attacks=21 frames=6 "
            "fc_tokens=18 patch_rows=3 no_gpu=1 no_capture=1" in author_output,
            "author synthetic test did not pass")
    cli_self_check = run_json([SOURCE, "--source-self-check"])
    cli_describe = run_json([SOURCE, "--describe"])
    require(cli_self_check == self_check and cli_describe == description,
            "CLI/module consistency drift")

    result = {
        "schema": "m1560_m1558_reduced_binary_source_independent_review_r1_v1",
        "status": "NO_GO_M1560_REMOTE_WRAPPER_AUTHORING__PRELOAD_PERMIT_MINT_EXPOSED__LOCAL_FORMAT_AND_DUAL_RUNTIME_PASS",
        "runtime": {"executable": sys.executable,
                    "version": sys.version.split()[0]},
        "pinned_inputs": inputs,
        "author_sealed_members": author_sealed_members,
        "inventory": {"layers": 32, "FC1": 12, "FC2": 12, "PATCH": 8,
                      "inventory_sha256": module.canonical_sha(specs)},
        "population": {"fc_tokens": 44640000,
                       "patch_tokens_histogram_only": 430080000},
        "estimate": {"raw_fc_payload_upper_bytes": 7528535874,
                     "result_upper_bytes": 7598737368,
                     "runtime_hard_cap_bytes": module.MAX_RUNTIME_BYTES,
                     "strict_free_after_bytes": module.MIN_FREE_AFTER_BYTES},
        "local_consistency": {
            "source_self_check": True, "author_synthetic_test": True,
            "binary_fixed_header": True, "independent_zlib_frames": True,
            "crc32_and_exact_extent": True, "canonical_order_parser": True,
            "little_endian_tail_zero": True, "uint16_nnz": True,
            "sign_subset_and_code_match": True,
            "nonunit_subset_and_code_match": True,
            "zero_token_retained": True, "patch_histogram_only": True,
            "hardware_quantization_authority": False,
            "runtime_raw_and_disk_hard_caps": True},
        "p0_finding": {
            "producer_requires_exact_permit_type": True,
            "permit_one_shot_path_inventory_binding": True,
            "module_global_mint_callable": True,
            "direct_exact_type_mint_bypasses_free_space_issuer": True,
            "demonstrated_free_bytes_before": 0,
            "demonstrated_free_bytes_after_upper": -1,
            "permit_gate_truly_enforced": False,
            "required_fix": (
                "remove the module-global raw mint callable and close permit "
                "construction behind an issuer that always enforces fresh-path, "
                "estimate, 12-GiB and strict post-result 16-GiB checks")},
        "authorization": {
            "successor_permit_gate_fix_authoring": True,
            "remote_integration_wrapper_authoring": False,
            "checkpoint_load": False, "gpu": False, "ssh": False,
            "capture": False, "release": False, "automatic_retry": False,
            "rtl": False},
        "release_ladder": {
            "independent_rehammer_after_fix_required": True,
            "remote_wrapper_may_be_authored_after_fix_pass": True,
            "actual_capture_still_requires_separate_one_shot_release": True,
            "production_result_hammer_required": True},
        "claim_boundary": {
            "local_source_and_synthetic_only": True,
            "checkpoint_loaded": False, "gpu": False, "ssh": False,
            "capture_executed": False, "release_executed": False,
            "opportunity": False, "aee": False, "cycles": False,
            "traffic": False, "energy": False, "speedup": False,
            "rtl": False, "eda": False, "paper_headline": False}
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

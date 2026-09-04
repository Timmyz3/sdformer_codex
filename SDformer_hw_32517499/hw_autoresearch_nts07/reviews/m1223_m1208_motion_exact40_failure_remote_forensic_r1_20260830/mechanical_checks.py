#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent

EXPECTED = {
    "m1208": (ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1208_motion_ep29_unified_hardware_symlink_root_successor_r1.py", "41b5276c39b613b6568ad7c7486abf150c3d0db86c3a905d6a30cdbbb543a049"),
    "m1180": (ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py", "f88426c789c99a0d56c34ffaa742b052c73fcbad600c4ecd5797a62e2cf26479"),
    "m1177": (ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1177_motion_checkpoint_parametric_unified_hardware_r2.py", "b2b578f7f38643c7e05bd3413101bd5a6eb4f5224e0c0468ff088997f7602184"),
    "m1174": (ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1174_motion_checkpoint_parametric_unified_hardware.py", "b476fad6885be23aa63a6b5d8e690fb3e213421074270cbb25e8ec00c202080a"),
    "attention_writer": (ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_bit_trace.py", "75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e"),
    "docs359": (HW / "docs/359_DATE终局冻结_20260813.md", "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}

M1224 = HW / "reviews/m1224_m1208_capture_contract_first_principles_audit_r1_20260830"
M1224_EXPECTED = {
    "review.json": "56372da531b3c56b375d45372ff2aea9be1754df2ba4a3a8c0e50a62936505dc",
    "SHA256SUMS": "677bb08190b1f345db8f4e5535c73d22952cbb44324bfb4b465a728032705135",
    "SHA256SUMS.seal.sha256": "9a858b44edb2bf36c4bad251eaa4501860b92c5d7ff83bcc6ba60a318a165b96",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for label, (path, digest) in EXPECTED.items():
    assert path.is_file(), (label, path)
    assert sha(path) == digest, (label, sha(path))

for name, digest in M1224_EXPECTED.items():
    path = M1224 / name
    assert path.is_file(), path
    assert sha(path) == digest, (name, sha(path))
assert (M1224 / "SHA256SUMS.seal.sha256").read_text().strip() == (
    M1224_EXPECTED["SHA256SUMS"] + "  SHA256SUMS"
)

obs = json.loads((HERE / "remote_read_only_observation.json").read_text())
assert obs["access"]["remote_write"] is False
assert obs["access"]["capture_rerun"] is False
assert obs["access"]["gpu_compute"] is False
assert obs["staging"]["files"] == 1122
assert obs["staging"]["root_inventory_sha256_pass1"] == obs["staging"]["root_inventory_sha256_pass2"]
assert obs["failure"]["reason"] == "R2Error: per-module runtime call coverage is not exactly 40"
assert obs["attention"]["records"] == obs["attention"]["npz_files"] == 480
assert obs["attention"]["sample_block_cartesian_exact"] is True
assert obs["attention"]["validation_errors"] == 0
assert obs["retained_payloads"]["files"] == 640
assert obs["retained_payloads"]["record_pairs"] == 320
assert obs["retained_payloads"]["complete_pairs"] == 320
assert set(obs["retained_payloads"]["per_module_filename_calls"].values()) == {40}
assert obs["expected_runtime_inventory"]["modules_total"] == 259
assert sum(obs["expected_runtime_inventory"]["modules_by_category"].values()) == 259
assert obs["runtime_distribution_recovery"]["strictwriter_actual_nonpayload_distribution_reconstructable"] is False
assert obs["runtime_distribution_recovery"]["in_memory_records_after_process_exit"] == "UNRECOVERABLE"
correction = obs["runtime_distribution_recovery"]["m1224_correction"]
assert correction["review_json_sha256"] == M1224_EXPECTED["review.json"]
assert correction["manifest_sha256"] == M1224_EXPECTED["SHA256SUMS"]
assert correction["outer_seal_sha256"] == M1224_EXPECTED["SHA256SUMS.seal.sha256"]
assert correction["dead_modules"] == 12 and correction["category"] == "atlif"
assert correction["name_suffix"] == ".sn_v"
assert correction["actual_calls_total"] == 9880
assert correction["expected_calls_total"] == 10360
assert correction["deficit"] == 480

r2 = EXPECTED["m1177"][0].read_text()
r1 = EXPECTED["m1174"][0].read_text()
assert 'require(observed == expected_calls, "per-module runtime call coverage is not exactly 40")' in r2
assert r1.index("writer.close()") < r1.index('staging / "unified_ordered_records.jsonl"')
assert 'if category not in {"c1_conv3x3", "decoder_convtranspose"}' in r1

print("M1223_MECHANICAL_CHECKS_PASS read_only=true remote_write=false rerun=false gpu=false attention480=true payload640=true actual_nonpayload_unrecoverable=true")

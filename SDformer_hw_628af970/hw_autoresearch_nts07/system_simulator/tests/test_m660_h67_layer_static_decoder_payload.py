import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m660_h67_layer_static_decoder_payload.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m660_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json")
RUNNER = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "run_m660_h67_layer_static_decoder_payload_one_shot.sh")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m660_test_target", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M660 = load_module()


def test_contract_inputs_and_critical_frozen_roots():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(contract["inputs"]) == set(contract["required_input_names"])
    for name, entry in contract["inputs"].items():
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), name
        assert digest(path) == entry["sha256"], name
    assert contract["inputs"]["launcher"]["sha256"] == digest(LAUNCHER)
    assert contract["inputs"]["runner"]["sha256"] == digest(RUNNER)
    assert contract["inputs"]["m658_manifest"]["sha256"].startswith(
        "aed109ab")
    assert contract["inputs"]["m658_outer_seal"]["sha256"].startswith(
        "5d235106")
    assert contract["inputs"]["m659_manifest"]["sha256"].startswith(
        "032be831")


def test_predecessor_semantics_and_double_seals_recompute():
    contract = M660.strict_json(CONTRACT)
    evidence = M660.verify_predecessor_evidence(contract)
    assert len(evidence["expected_records"]) == 40
    assert evidence["expected_records"][(0, 0)]["one_count"] == 839586
    assert evidence["expected_records"][(0, 1)][
        "nonbinary_finite_count"] == 1716275
    assert evidence["m658_review_sha256"] == M660.M658_REVIEW_SHA256
    assert evidence["m662_review_sha256"] == M660.M662_REVIEW_SHA256


def test_binary_stream_bitorder_shape_popcount_and_raw_hash(tmp_path):
    values = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1,
                           1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    path = tmp_path / "x.bitpack"
    stats = M660.stream_binary_input(values, 8, path)
    assert path.read_bytes() == bytes([0b10010110, 0b00000001])
    assert stats["elements"] == 16
    assert stats["one_count"] == 5
    assert stats["zero_count"] == 11
    assert stats["packed_bytes"] == 2
    assert stats["raw_content_sha256"] == hashlib.sha256(
        values.numpy().tobytes(order="C")).hexdigest()


def test_binary_stream_rejects_nonbinary_and_leaks_no_file(tmp_path):
    values = torch.tensor([0, 1, 0, 1, 0, 0.5, 0, 1],
                          dtype=torch.float32)
    path = tmp_path / "bad.bitpack"
    with pytest.raises(RuntimeError, match="not exact"):
        M660.stream_binary_input(values, 8, path)
    assert not path.exists()
    assert not (tmp_path / "bad.bitpack.partial").exists()


def test_d1_fallback_has_no_path_parameter_and_saves_no_payload(tmp_path):
    assert "path" not in inspect.signature(M660.summarize_d1_fallback).parameters
    values = torch.tensor([0.0, 0.25, float("inf"), 1.0],
                          dtype=torch.float32)
    before = list(tmp_path.iterdir())
    result = M660.summarize_d1_fallback(values, 8)
    assert result["zero_count"] == 1
    assert result["one_count"] == 1
    assert result["nonbinary_finite_count"] == 1
    assert result["nonfinite_count"] == 1
    assert result["raw_payload_saved"] is False
    assert list(tmp_path.iterdir()) == before


def test_d1_exact_theta_candidate_pack_and_ieee_equality(tmp_path):
    theta = torch.tensor(np.float32(0.9999954104423523))
    values = torch.tensor([0, theta.item(), 0, theta.item(),
                           theta.item(), 0, 0, 0], dtype=torch.float32)
    path = tmp_path / "theta.bitpack"
    result = M660.stream_theta_binary_candidate(values, theta, 8, path)
    assert result["theta_gate_pass"] is True
    assert result["zero_count"] == 5
    assert result["theta_count"] == 3
    assert result["other_finite_count"] == 0
    assert path.read_bytes() == bytes([0b00011010])
    assert result["raw_payload_saved"] is False
    assert result["thresholded"] is False
    assert result["rounded"] is False


def test_d1_theta_candidate_fail_does_not_publish_payload(tmp_path):
    theta = torch.tensor(np.float32(0.9999954104423523))
    values = torch.tensor([0, theta.item(), 0, 0.5, 0, 0, 0, 0],
                          dtype=torch.float32)
    path = tmp_path / "theta_bad.bitpack"
    result = M660.stream_theta_binary_candidate(values, theta, 8, path)
    assert result["theta_gate_pass"] is False
    assert result["other_finite_count"] == 1
    assert result["packed_bytes"] == 0
    assert not path.exists()
    assert not (tmp_path / "theta_bad.bitpack.partial").exists()


def test_decoder_threshold_identity_requires_finite_positive_scalar():
    class Neuron(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.thresh = torch.nn.Parameter(torch.tensor(value,
                                                           dtype=torch.float32))
            self.threshold_mode = "official_atlif"
            self.output_mode = "binary"

    class Owner(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.sn = Neuron(value)
            self.deconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(
                1, 1, 3, bias=False))

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.owner = Owner(value)

    expected = {"name": "owner.deconv.0"}
    _tensor, identity = M660.decoder_threshold_identity(Model(0.75), expected)
    assert identity["value"] == 0.75
    assert identity["parameter_device"] == "cpu"
    assert identity["content_bytes"] == 4
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(RuntimeError, match="finite and positive"):
            M660.decoder_threshold_identity(Model(bad), expected)


def test_streamed_folded_miter_exact_and_nonexact():
    left = torch.arange(16, dtype=torch.float32)
    exact = M660.compare_tensors_streaming(left, left.clone(), 8)
    assert exact["bit_exact"] is True
    assert exact["bit_exact_mismatch_count"] == 0
    assert exact["max_abs_error"] == 0.0
    changed = left.clone()
    changed[7] += 0.25
    failed = M660.compare_tensors_streaming(left, changed, 8)
    assert failed["bit_exact"] is False
    assert failed["bit_exact_mismatch_count"] == 1
    assert failed["max_abs_error"] == 0.25


def test_nested_seals_bind_nested_seal_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "value.bin").write_bytes(b"value")
    M660.write_double_seal(nested)
    M660.verify_double_seal(nested)
    (tmp_path / "top.txt").write_text("top\n", encoding="utf-8")
    M660.write_double_seal(tmp_path)
    M660.verify_double_seal(tmp_path)
    top = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "nested/SHA256SUMS" in top
    assert "nested/SHA256SUMS.seal.sha256" in top


def test_dual_population_and_unique_lattices_are_exact():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fallback = contract["expected_population"]["fp32_fallback"]
    theta = contract["expected_population"]["theta_binary_go"]
    assert fallback["binary_payload_records"] == 30
    assert fallback["binary_packed_bytes_total"] == 75480000
    assert theta["binary_payload_records"] == 40
    assert theta["binary_packed_bytes_by_module"]["1"] == 11550000
    assert theta["binary_packed_bytes_total"] == 87030000
    lattice30 = [(sample, module) for sample in range(10)
                 for module in (0, 2, 3)]
    assert len(lattice30) == len(set(lattice30)) == 30
    assert lattice30[0] == (0, 0) and lattice30[-1] == (9, 3)


def test_runtime_receipt_closes_m658_p2_and_env_is_exact():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime = contract["runtime_provenance"]
    assert runtime["expected_environment"] == {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "M660_ATTEMPT_DIRECTORY": str(ROOT / (
            "hw_autoresearch_nts07/results/"
            ".m660_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed")),
        "M660_EXPECTED_CONTRACT_SHA256":
            "DERIVED_EQUAL_TO_RUNNING_CONTRACT_SHA256",
        "M660_EXPECTED_RUNNER_SHA256": digest(RUNNER),
        "M660_RUNNER_PATH": str(RUNNER),
        "PATH": "/usr/bin:/bin",
        "SDFORMER_USE_MLFLOW": "0",
    }
    required = " ".join(runtime["receipt_required"])
    for token in ("hostname", "Python executable", "torch/numpy/spikingjelly",
                  "CUDA", "driver", "GPU UUID", "exact argv", "environment"):
        assert token in required
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"runtime_receipt"' in source
    assert "write_double_seal(directory)" in source


def test_runner_is_sanitized_one_shot_and_static_gpu_false():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.startswith("#!/bin/bash -p\n")
    assert "/usr/bin/env -i" in source
    assert "mkdir \"${m660_attempt}\"" in source
    assert "GPU is not idle; one-shot remains unconsumed" in source
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["resource_policy"]["gpu_execution_currently_authorized"] is False
    assert contract["claim_boundary"]["gpu_run"] is False
    assert not (ROOT / contract["output"]["canonical_directory"]).exists()
    assert not (ROOT / contract["one_shot"]["attempt_directory"]).exists()


def test_d1_no_silent_lossy_conversion_and_dual_statuses_present():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "chunk == theta" in source
    assert "torch.round" not in source
    assert "torch.sign" not in source
    assert "thresholded\": False" in source
    assert "D1_FOLDED_WEIGHT_MITER_BIT_EXACT" in source
    assert "D1_FOLDED_WEIGHT_MITER_NONEXACT" in source
    assert "D1_COMMON_FP32_FALLBACK" in source
    assert '"EXACT_SCALED_BINARY_BITPACK"' in source
    assert '"folded_weight_deployment_admitted"' in source
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["layer_static_dual_result"][
        "threshold_round_or_binary_coercion_allowed"] is False
    nonexact = contract["folded_weight_miter"]["nonexact_policy"]
    assert "admit only the exact scaled-binary representation" in nonexact
    assert "diagnostic candidate marked deployment-not-admitted" in nonexact


def test_historical_evidence_and_docs359_are_unchanged():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name in ("m511_attempt_outer_seal", "m511_failed_receipt",
                 "m511_failed_d0_payload", "m649_result", "m658_review",
                 "m659_plan", "docs359"):
        entry = contract["inputs"][name]
        assert digest(ROOT / entry["path"]) == entry["sha256"]
    assert contract["inputs"]["docs359"]["sha256"] == (
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


def test_claim_boundary_has_no_performance_or_eda_admission():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    boundary = contract["claim_boundary"]
    for key in ("gpu_run", "payload_captured", "cycles", "speedup", "rtl",
                "vcs", "dc", "formality", "ptpx", "energy", "ppa",
                "system_speedup", "date_headline"):
        assert boundary[key] is False, key

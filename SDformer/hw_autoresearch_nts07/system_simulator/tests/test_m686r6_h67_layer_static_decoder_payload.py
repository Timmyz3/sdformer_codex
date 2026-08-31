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
    "capture_m686r6_h67_layer_static_decoder_payload.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m686r6_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json")
RUNNER = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "run_m686r6_h67_layer_static_decoder_payload_one_shot.sh")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m686r6_test_target", LAUNCHER)
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


def test_real_wrapper_identity_immutable_clone_and_transient_drift_detection():
    class Neuron(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.thresh = torch.nn.Parameter(torch.tensor(value,
                                                           dtype=torch.float32))
            self.threshold_mode = "official_atlif"
            self.output_mode = "binary"

    class Wrapper(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.spiking_neuron = Neuron(value)

    class Owner(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.sn = Wrapper(value)
            self.deconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(
                1, 1, 3, bias=False))

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.owner = Owner(value)

    expected = {"name": "owner.deconv.0"}
    model = Model(0.75)
    frozen, identity = M660.decoder_threshold_identity(model, expected)
    assert identity["value"] == 0.75
    assert identity["wrapper_class"] == "Wrapper"
    assert identity["leaf_name"] == "owner.sn.spiking_neuron"
    assert identity["parameter_name"] == "owner.sn.spiking_neuron.thresh"
    assert identity["parameter_device"] == "cpu"
    assert identity["content_bytes"] == 4
    with torch.no_grad():
        model.owner.sn.spiking_neuron.thresh.fill_(0.5)
    assert frozen.item() == 0.75  # immutable clone, not the live storage
    with pytest.raises(RuntimeError, match="threshold drift"):
        M660.threshold_identity_matches(model, expected, identity, "unit")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(RuntimeError, match="finite and positive"):
            M660.decoder_threshold_identity(Model(bad), expected)

    class BadOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sn = Neuron(0.75)  # r1's unrealistic direct-leaf topology
            self.deconv = torch.nn.Sequential(torch.nn.ConvTranspose2d(
                1, 1, 3, bias=False))

    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = BadOwner()

    with pytest.raises(RuntimeError, match="owner.sn must be"):
        M660.decoder_threshold_identity(BadModel(), expected)


def test_streamed_folded_miter_exact_and_nonexact():
    left = torch.arange(16, dtype=torch.float32)
    exact = M660.compare_tensors_streaming(left, left.clone(), 8)
    assert exact["bit_exact"] is True
    assert exact["bit_exact_mismatch_count"] == 0
    assert exact["max_ulp_error"] == 0
    assert exact["hashes_equal"] is True
    assert exact["max_abs_error"] == 0.0
    changed = left.clone()
    changed[7] += 0.25
    failed = M660.compare_tensors_streaming(left, changed, 8)
    assert failed["bit_exact"] is False
    assert failed["bit_exact_mismatch_count"] == 1
    assert failed["max_abs_error"] == 0.25


def test_signed_zero_and_one_ulp_cannot_enter_bit_exact_deployment():
    signed_zero = M660.compare_tensors_streaming(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([-0.0], dtype=torch.float32), 1)
    assert signed_zero["bit_exact"] is False
    assert signed_zero["bit_exact_mismatch_count"] == 1
    assert signed_zero["signed_zero_bit_mismatch_count"] == 1
    assert signed_zero["max_ulp_error"] == 1
    assert signed_zero["hashes_equal"] is False
    assert M660.folded_miter_admitted(
        [{"folded_weight_miter": signed_zero}], True) is False

    left = torch.tensor([1.0], dtype=torch.float32)
    right = torch.nextafter(left, torch.tensor([2.0], dtype=torch.float32))
    one_ulp = M660.compare_tensors_streaming(left, right, 1)
    assert one_ulp["max_ulp_error"] == 1
    assert one_ulp["bit_exact"] is False


@pytest.mark.parametrize("phase", ["early", "middle", "late"])
def test_failure_scrub_removes_every_d1_candidate_at_all_phases(
        tmp_path, phase):
    staging = tmp_path / phase
    calls = staging / "calls"
    weights = staging / "weights"
    candidate = staging / "d1_candidate"
    calls.mkdir(parents=True)
    weights.mkdir()
    candidate.mkdir()
    (calls / "s00_d0.activation.le.bitpack").write_bytes(b"keep")
    (weights / "d1.weight.f32le").write_bytes(b"keep-original")
    populations = {"early": 1, "middle": 5, "late": 10}
    for sample in range(populations[phase]):
        (candidate / ("s{:02d}_d1.activation.theta.le.bitpack".format(
            sample))).write_bytes(b"candidate")
    if phase != "early":
        (weights / "d1.weight.folded_theta.f32le").write_bytes(b"folded")
        (weights / "d1.original_weight_output_scale.sidecar.json").write_text(
            "{}\n", encoding="utf-8")
    if phase == "late":
        (calls / "s09_d1.activation.theta.le.bitpack").write_bytes(b"promoted")
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text("stale\n", encoding="utf-8")
    M660.write_double_seal(weights)
    M660.write_double_seal(staging)
    removed = M660.scrub_d1_candidates(staging)
    assert removed
    assert not candidate.exists()
    assert not list(calls.glob("s??_d1.activation.theta.le.bitpack"))
    assert not (weights / "d1.weight.folded_theta.f32le").exists()
    assert not (weights /
                "d1.original_weight_output_scale.sidecar.json").exists()
    assert (calls / "s00_d0.activation.le.bitpack").read_bytes() == b"keep"
    assert (weights / "d1.weight.f32le").read_bytes() == b"keep-original"
    assert not (staging / "SHA256SUMS").exists()
    assert not (weights / "SHA256SUMS").exists()


def test_determinism_and_theta_check_lattices_are_executable(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    observed = M660.configure_deterministic_execution()
    M660.require_deterministic_execution(observed)
    assert observed == {
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
    }
    source = LAUNCHER.read_text(encoding="utf-8")
    for token in ("register_forward_pre_hook(d1_leaf_pre_hook)",
                  "register_forward_hook(d1_leaf_post_hook)",
                  "d1_deconv_pre_hook", "d1_deconv_post_hook",
                  "sample_pre_forward", "sample_post_forward",
                  '"leaf_pre_forward": 10', '"leaf_post_forward": 10',
                  '"d1_deconv_pre_hook": 10',
                  '"d1_deconv_post_hook": 10'):
        assert token in source


def test_folded_and_sidecar_serialization_is_after_global_s10_gate_only():
    source = LAUNCHER.read_text(encoding="utf-8")
    build = source.index("d1_folded_weight_device = build_folded_weight_device")
    capture = source.index("for chunk, mask, label in take_exact")
    global_gate = source.index("if d1_theta_gate_pass:", capture)
    save = source.index("save_folded_weight_payload(", global_gate)
    assert build < capture < global_gate < save
    assert "save_folded_weight_payload(" not in source[capture:global_gate]
    exception = source.index("except BaseException as error:")
    scrub = source.index("scrubbed = scrub_d1_candidates(staging)", exception)
    failure = source.index("failure = failure_root / failure_name", exception)
    assert exception < scrub < failure


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


def test_runtime_receipt_keeps_m658_p2_pending_and_env_is_exact():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime = contract["runtime_provenance"]
    assert runtime["expected_environment"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "M660R2_ATTEMPT_DIRECTORY": str(ROOT / (
            "hw_autoresearch_nts07/results/"
            ".m686r6_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed")),
        "M660R2_EXPECTED_CONTRACT_SHA256":
            "DERIVED_EQUAL_TO_RUNNING_CONTRACT_SHA256",
        "M660R2_EXPECTED_RUNNER_SHA256": digest(RUNNER),
        "M660R2_RUNNER_PATH": str(RUNNER),
        "PATH": "/usr/bin:/bin",
        "SDFORMER_USE_MLFLOW": "0",
    }
    required = " ".join(runtime["receipt_required"])
    for token in ("hostname", "Python executable", "torch/numpy/spikingjelly",
                  "CUDA", "driver", "GPU UUID", "exact argv", "environment",
                  "deterministic", "TF32", "CUBLAS"):
        assert token in required


def test_execution_controls_are_rechecked_after_config_model_and_each_forward():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "def observe_execution_controls():" in source
    assert source.count(
        "require_deterministic_execution(observe_execution_controls())") >= 5
    assert "torch.backends.cudnn.allow_tf32 = True" in source
    assert "cuDNN-TF32 evaluation semantics" in source
    assert '"cupy_installed_version": "14.2.0"' in source
    assert '"set_backend_target_module_count": 0' in source
    assert '"effective_cupy_assignment_count": 0' in source
    assert '"Dropout:torch": 49, "IFNode:torch": 4' in source
    assert '"module_count": 105' in source
    assert '"remaining_spikingjelly_ifnode_count": 4' in source
    assert '"forward_primitive": "torch.addmm"' in source
    assert "resolver_label_is_not_claimed_as_actual_cupy_execution" in source
    assert "M686-r6 S00/D0 frozen bit-exact sentinel drift" in source
    assert 'stats["packed_sha256"] ==' in source
    contract_packages = contract["runtime_provenance"]["package_versions"]
    assert contract_packages["cupy-cuda12x"] == "14.2.0"
    assert contract["predecessors"]["m658_p2_runtime_provenance_closure"] == (
        "PENDING_POST_RESULT_INDEPENDENT_HAMMER")
    assert contract["predecessors"][
        "m658_p2_closed_by_static_contract_or_author"] is False
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"runtime_receipt"' in source
    assert "write_double_seal(directory)" in source


def test_runner_is_sanitized_one_shot_and_static_gpu_false():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.startswith("#!/bin/bash -p\n")
    assert "/usr/bin/env -i" in source
    assert "mkdir \"${m660r2_attempt}\"" in source
    assert "GPU is not idle; one-shot remains unconsumed" in source
    assert source.index("--cpu-preflight-only") < source.index(
        'mkdir "${m660r2_attempt}"')
    assert source.count(
        'sha256sum -c SHA256SUMS.seal.sha256 >/dev/null') >= 2
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
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
                 "m659_plan", "m662_review", "m666_review",
                 "m666_outer_seal", "docs359"):
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

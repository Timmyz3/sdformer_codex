import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m699_h67_ep35_multisequence_decoder_payload.py")
RUNNER = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "run_m699_h67_ep35_multisequence_decoder_payload_one_shot.sh")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_target():
    spec = importlib.util.spec_from_file_location("m699_author_test", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_shell_and_python_are_static_valid():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == \
        "m699_h67_ep35_multisequence_decoder_payload_contract_v1"
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run([
        "/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "py_compile",
        str(PRODUCER)], check=True)


def test_contract_pins_running_author_files_and_frozen_roots():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for entry in contract["inputs"].values():
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"]
        assert sha(path) == entry["sha256"]
    assert contract["inputs"]["docs359"]["sha256"] == \
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def test_selection_is_three_ordered_endpoint_covering_s10_cohorts():
    target = load_target()
    contract = target.strict_json(CONTRACT)
    sources = target.verify_selected_sources(contract)
    assert len(sources) == 30
    expected = {
        "interlaken_01_a": [0, 12, 24, 36, 48, 59, 71, 83, 95, 107],
        "thun_01_b": [0, 8, 16, 25, 33, 41, 49, 58, 66, 74],
        "zurich_city_12_a": [0, 8, 16, 25, 33, 41, 49, 58, 66, 74],
    }
    for sequence, indices in expected.items():
        cohort = [row for row in sources if row["sequence"] == sequence]
        assert [row["source_index"] for row in cohort] == indices
        assert cohort[0]["source_index"] == 0
        assert cohort[-1]["source_index"] == cohort[-1]["source_population"] - 1


def test_selected_source_sha_substitution_fails_closed():
    target = load_target()
    contract = target.strict_json(CONTRACT)
    mutant = copy.deepcopy(contract)
    mutant["selected_sources"][11]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="selected NPY identity drift"):
        target.verify_selected_sources(mutant)


def test_selected_source_index_substitution_fails_closed():
    target = load_target()
    contract = target.strict_json(CONTRACT)
    mutant = copy.deepcopy(contract)
    mutant["selected_sources"][1]["source_index"] = 13
    with pytest.raises(RuntimeError, match="evenly-spaced selection drift"):
        target.verify_selected_sources(mutant)


def test_helper_routes_preserve_raw_bits_and_never_threshold():
    target = load_target()
    m686_path = ROOT / json.loads(CONTRACT.read_text())[
        "inputs"]["m686_helper"]["path"]
    m686 = target.load_module("m699_test_m686", m686_path,
                              target.M686_SHA256)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        binary = torch.tensor([0.0, 1.0] * 8, dtype=torch.float32)
        binary_stats = m686.stream_binary_input(
            binary, 8, directory / "binary.bitpack")
        assert binary_stats["one_count"] == 8
        assert binary_stats["raw_content_sha256"] == hashlib.sha256(
            binary.numpy().tobytes(order="C")).hexdigest()

        theta = torch.tensor(0.9999954104423523, dtype=torch.float32)
        scaled = torch.stack([torch.tensor(0.0), theta] * 8)
        scaled_stats = m686.stream_theta_binary_candidate(
            scaled, theta, 8, directory / "scaled.bitpack")
        assert scaled_stats["theta_gate_pass"]
        assert scaled_stats["theta_count"] == 8
        assert not scaled_stats["thresholded"] and not scaled_stats["rounded"]
        assert scaled_stats["raw_content_sha256"] == hashlib.sha256(
            scaled.numpy().tobytes(order="C")).hexdigest()

        opaque = scaled.clone()
        opaque[3] = torch.nextafter(theta, torch.tensor(0.0))
        opaque_stats = m686.stream_theta_binary_candidate(
            opaque, theta, 8, directory / "opaque.bitpack")
        assert not opaque_stats["theta_gate_pass"]
        assert opaque_stats["other_finite_count"] == 1
        assert not (directory / "opaque.bitpack").exists()


def test_claim_boundary_has_no_performance_or_accuracy_admission():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    boundary = contract["claim_boundary"]
    assert boundary["payload"] and boundary["density"]
    for key in ("accuracy", "cycles", "speedup", "system_speedup", "rtl",
                "vcs", "eda", "dc", "formality", "ptpx", "energy",
                "ppa", "date_headline"):
        assert boundary[key] is False


def test_runner_requires_fresh_review_and_consumes_before_python():
    text = RUNNER.read_text(encoding="utf-8")
    assert "m700_m699_multisequence_decoder_capture_fresh_static_hammer" in text
    assert "GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0" in text
    assert text.index('mkdir "${m699_attempt}"') < text.index(
        '"${m699_python}" "${m699_producer}"')
    assert "/usr/bin/env -i" in text
    assert "requires 20 GiB free GPU memory" in text


def test_canonical_output_and_attempt_are_unconsumed_at_author_handoff():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert not (ROOT / contract["output"]["canonical_directory"]).exists()
    assert not (ROOT / contract["one_shot"]["attempt_directory"]).exists()


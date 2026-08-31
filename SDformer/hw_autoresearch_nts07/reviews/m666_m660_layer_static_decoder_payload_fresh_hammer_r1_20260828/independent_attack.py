#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent CPU-only adversarial checks for the frozen M660 candidate.

This checker deliberately does not invoke the author test module, the runner,
CUDA, the one-shot token, a cycle simulator, RTL, or EDA.  A zero exit status
means that the attacks themselves completed and that the documented defects
were reproduced; it is not an M660 GO verdict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[3]
PRODUCER = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m660_h67_layer_static_decoder_payload.py")
RUNNER = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "run_m660_h67_layer_static_decoder_payload_one_shot.sh")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m660_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json")
AUTHOR_TESTS = ROOT / (
    "hw_autoresearch_nts07/system_simulator/tests/"
    "test_m660_h67_layer_static_decoder_payload.py")
AUTHOR_HANDOFF = ROOT / (
    "hw_autoresearch_nts07/reviews/"
    "m660_h67_layer_static_decoder_payload_author_handoff_r1_20260828")

FROZEN = {
    PRODUCER: "2e1ea26b5293ba1063e7be0056cebd2b25e09903bb528c31427c032df8b73acc",
    RUNNER: "ae9902b42331f3e88e94b11d9c5a5f6f3bdfc3e2b473939a7569af38f2396281",
    CONTRACT: "38200ef4db5795d8be70e6e776aabf09dad10818344b972add535900a95f2cb4",
    AUTHOR_TESTS: "0dc63c88349dec0ecc77d2fb4aa51f0df82316d1c435a73f1d760ae50fb54cc0",
    AUTHOR_HANDOFF / "SHA256SUMS.seal.sha256":
        "341db83d1c084b3ea6e41b155d4a24039b858fafa9a23ca45e7a3319f105f414",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_seal(directory: Path) -> dict:
    """Independent double-seal verifier, not the author's helper."""
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    expected_outer, outer_name = outer.read_text(
        encoding="utf-8").strip().split("  ", 1)
    assert outer_name == "SHA256SUMS"
    assert digest(seal) == expected_outer
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        assert name not in sealed
        assert not relative.is_absolute() and ".." not in relative.parts
        member = directory / relative
        assert member.is_file() and not member.is_symlink()
        assert digest(member) == expected
        sealed[name] = expected
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {
            "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert actual == set(sealed)
    return {"members": len(sealed), "outer_sha256": digest(outer)}


def load_target():
    spec = importlib.util.spec_from_file_location("m666_target", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expect_runtime_error(callable_object, token: str) -> str:
    try:
        callable_object()
    except RuntimeError as error:
        message = str(error)
        assert token in message, message
        return message
    raise AssertionError("expected RuntimeError containing " + token)


def main() -> int:
    observed = {str(path.relative_to(ROOT)): digest(path)
                for path in FROZEN}
    for path, expected in FROZEN.items():
        assert digest(path) == expected, path

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, entry in contract["inputs"].items():
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), name
        assert digest(path) == entry["sha256"], name

    seal_results = {"author_handoff": verify_seal(AUTHOR_HANDOFF)}
    for key in ("m658_outer_seal", "m659_outer_seal", "m662_outer_seal"):
        if key in contract["inputs"]:
            seal_results[key] = verify_seal(
                (ROOT / contract["inputs"][key]["path"]).parent)

    target = load_target()

    # Path alias attacks are read-only except for a private temporary tree.
    with tempfile.TemporaryDirectory(prefix="m666_path_attack_") as raw:
        temporary = Path(raw)
        real = temporary / "real"
        real.mkdir()
        leaf = real / "leaf"
        leaf.write_text("x", encoding="utf-8")
        alias = temporary / "alias"
        alias.symlink_to(real, target_is_directory=True)
        traversal = expect_runtime_error(
            lambda: target.checked_path(real / ".." / "real" / "leaf"),
            "parent traversal")
        symlink = expect_runtime_error(
            lambda: target.checked_path(alias / "leaf"),
            "symlink path component")

    # The exact iterator must not probe item eleven.
    class CountingIterator:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 11:
                raise AssertionError("item eleven was requested")
            return self.calls

    iterator = CountingIterator()
    assert list(target.take_exact(iterator, 10)) == list(range(1, 11))
    assert iterator.calls == 10

    # Independently exercise first/middle/last invalid D1 locations.  No final
    # file or .partial may survive any negative chunk route.
    theta = torch.tensor(0.75, dtype=torch.float32)
    d1_negative = []
    with tempfile.TemporaryDirectory(prefix="m666_d1_attack_") as raw:
        temporary = Path(raw)
        for position in (0, 7, 8, 15):
            values = torch.zeros(16, dtype=torch.float32)
            values[1 if position == 0 else 0] = theta
            values[position] = 0.5
            path = temporary / ("invalid_{}.bitpack".format(position))
            result = target.stream_theta_binary_candidate(
                values, theta, 8, path)
            assert result["theta_gate_pass"] is False
            assert result["other_finite_count"] == 1
            assert result["packed_bytes"] == 0
            assert not path.exists() and not Path(str(path) + ".partial").exists()
            d1_negative.append(position)

        valid = torch.tensor(
            [0, .75, .75, 0, 0, .75, 0, 0,
             .75, 0, 0, 0, .75, 0, 0, 0], dtype=torch.float32)
        valid_path = temporary / "valid.bitpack"
        valid_result = target.stream_theta_binary_candidate(
            valid, theta, 8, valid_path)
        assert valid_path.read_bytes() == bytes([0b00100110, 0b00010001])
        assert valid_result["theta_count"] == 5
        assert sum(byte.bit_count() for byte in valid_path.read_bytes()) == 5

    # Faithfully reproduce the H67 owner boundary: Owner.sn is a wrapper and
    # the actual ATLIF object is Owner.sn.spiking_neuron.  The frozen helper
    # dereferences the wrapper itself and therefore rejects the real topology.
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.thresh = torch.nn.Parameter(torch.tensor(
                0.9999954104423523, dtype=torch.float32))
            self.threshold_mode = "official_atlif"
            self.output_mode = "binary"

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spiking_neuron = Inner()

    class Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sn = Wrapper()
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(1, 1, 3, bias=False))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = Owner()

    wrapper_error = expect_runtime_error(
        lambda: target.decoder_threshold_identity(
            Model(), {"name": "owner.deconv.0"}),
        "not the frozen scalar official-ATLIF binary neuron")

    # Numeric comparison is not byte comparison.  IEEE +0 and -0 compare
    # equal numerically, so the frozen helper emits bit_exact=true even though
    # the independently hashed byte streams are unequal.
    plus_zero = torch.tensor([0.0], dtype=torch.float32)
    minus_zero = torch.tensor([-0.0], dtype=torch.float32)
    signed_zero = target.compare_tensors_streaming(plus_zero, minus_zero, 1)
    plus_hash = hashlib.sha256(plus_zero.numpy().tobytes()).hexdigest()
    minus_hash = hashlib.sha256(minus_zero.numpy().tobytes()).hexdigest()
    assert signed_zero["bit_exact"] is True
    assert signed_zero["bit_exact_mismatch_count"] == 0
    assert plus_hash != minus_hash
    assert signed_zero["original_output_sha256"] != signed_zero[
        "folded_reference_output_sha256"]

    # Endpoint equality cannot prove a static threshold.  The returned tensor
    # is an alias, not a clone, and a mutate/restore interval is unobservable.
    direct = Model()
    inner = direct.owner.sn.spiking_neuron
    start_alias = inner.thresh.detach()
    start_bytes = inner.thresh.detach().cpu().numpy().tobytes()
    with torch.no_grad():
        inner.thresh.fill_(0.5)
    assert float(start_alias.item()) == 0.5  # snapshot mutated through alias
    transient_bytes = inner.thresh.detach().cpu().numpy().tobytes()
    with torch.no_grad():
        inner.thresh.fill_(0.9999954104423523)
    end_bytes = inner.thresh.detach().cpu().numpy().tobytes()
    assert start_bytes == end_bytes and transient_bytes != start_bytes

    source = PRODUCER.read_text(encoding="utf-8")
    # A candidate folded weight and sidecar are serialized before capture.  The
    # exception handler records failure but never scrubs those files.
    save_offset = source.index("d1_folded_weight_device = save_folded_weight_payload")
    capture_offset = source.index("for chunk, mask, label in take_exact")
    exception_offset = source.index("except BaseException as error:")
    exception_tail = source[exception_offset:source.index("return 0", exception_offset)]
    assert save_offset < capture_offset
    assert "d1_candidate" not in exception_tail
    assert "d1.weight.folded_theta.f32le" not in exception_tail
    assert "d1.original_weight_output_scale.sidecar.json" not in exception_tail

    deterministic_tokens = {
        "torch.use_deterministic_algorithms":
            "torch.use_deterministic_algorithms" in source,
        "cudnn.deterministic": "cudnn.deterministic" in source,
        "cudnn.benchmark": "cudnn.benchmark" in source,
        "cuda.matmul.allow_tf32": "cuda.matmul.allow_tf32" in source,
        "cudnn.allow_tf32": "cudnn.allow_tf32" in source,
        "max_ulp": "max_ulp" in source.lower(),
    }
    assert not any(deterministic_tokens.values())

    sidecar_function = source[
        source.index("def save_folded_weight_payload"):
        source.index("def nvidia_smi_identity")]
    assert '"admitted": False' in sidecar_function

    report = {
        "schema": "m666_m660_independent_attack_result_v1",
        "status": "ATTACKS_COMPLETE__DEFECTS_REPRODUCED__NOT_A_GO",
        "frozen_target_hashes": observed,
        "independent_double_seals": seal_results,
        "attacks": {
            "parent_traversal_rejected": traversal,
            "symlink_component_rejected": symlink,
            "take_exact_next_calls": iterator.calls,
            "d1_negative_positions_clean": d1_negative,
            "bitpack_little_first_bytes_hex": "2611",
            "wrapper_path_defect_reproduced": wrapper_error,
            "signed_zero_false_bit_exact": {
                "reported_bit_exact": signed_zero["bit_exact"],
                "reported_mismatch_count": signed_zero[
                    "bit_exact_mismatch_count"],
                "plus_zero_sha256": plus_hash,
                "minus_zero_sha256": minus_hash,
                "hashes_unequal": plus_hash != minus_hash,
            },
            "theta_snapshot_aliases_parameter": True,
            "transient_theta_endpoint_attack_invisible": True,
            "candidate_written_before_capture_and_not_scrubbed_on_exception":
                True,
            "deterministic_and_ulp_controls_present": deterministic_tokens,
            "output_scale_sidecar_statically_unadmitted": True,
        },
        "claim_boundary": {
            "gpu": False, "one_shot": False, "cycle_simulator": False,
            "rtl": False, "eda": False, "performance": False,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

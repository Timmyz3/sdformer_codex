#!/usr/bin/env python3
"""CPU micro and adversarial state-machine tests for M460R3."""

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R3_PATH = (ROOT / "hw_autoresearch_nts07/system_handoff/scripts/"
           "capture_m460r3_h67_g8_ffn_token_residual_s10.py")
OLD_TEST_PATH = (ROOT / "hw_autoresearch_nts07/tests/"
                 "test_m460_g8_ffn_residual_stream_capture.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R3 = load_module(R3_PATH, "m460r3_capture_tested")
FAKES = load_module(OLD_TEST_PATH, "m460r3_frozen_fake_modules")


def values(stage, block, poison_nan=False):
    channels = 3 + stage
    token_shape = (2, 1, 2, 1)
    count = int(np.prod(token_shape) * channels)
    x = ((np.arange(count, dtype=np.float32).reshape(
        token_shape + (channels,)) % 11) - 5.0) / 8.0
    x = x + np.float32(block * 0.03125)
    sn1 = np.where(np.abs(x) >= 0.25, x, 0.0).astype(np.float32)
    sn2 = np.repeat(sn1, 4, axis=-1)
    pre = (x * np.float32(0.75) +
           np.float32(0.125 + 0.01 * block)).astype(np.float32)
    residual = (pre * np.float32(1.5) - np.float32(0.0625)).astype(
        np.float32)
    residual[0, 0, 0, 0, :] = 0.0
    if poison_nan:
        residual[-1, 0, -1, 0, 0] = np.nan
    return x, sn1, sn2, pre, residual


def fire_normal(model, name, payload):
    x, sn1, sn2, pre, residual = payload
    model.named[name].fire_pre((x,))
    model.named[name + ".sn1"].fire_forward((x,), sn1)
    model.named[name + ".sn2"].fire_forward((pre,), sn2)
    model.named[name + ".fc2"].fire_forward((sn2,), pre)
    model.named[name].fire_forward((x,), residual)


def new_capture(directory, enforce=False, mutate_model=None):
    model = FAKES.FakeModel(R3.BASE)
    if mutate_model is not None:
        mutate_model(model)
    capture = R3.StrictFFNResidualStreamCapture(
        R3.BASE.NumpyTokenOps(), Path(directory),
        enforce_h67_geometry=enforce)
    capture.attach(model)
    capture.begin_sample(0, "synthetic_0001.npy", "synthetic")
    return model, capture


def expect_reject(name, body):
    try:
        body()
    except (RuntimeError, FileExistsError):
        return {"attack": name, "expected": "reject", "observed": "reject",
                "passes": True}
    return {"attack": name, "expected": "reject", "observed": "accept",
            "passes": False}


def run_normal_micro(directory):
    model, capture = new_capture(directory, enforce=False)
    first = None
    for stage, block, name in R3.BASE.all_targets():
        payload = values(stage, block, poison_nan=(stage == 3 and block == 1))
        fire_normal(model, name, payload)
        if first is None:
            first = payload
    capture.end_sample()
    if len(capture.records) != 12:
        raise AssertionError("M460R3 normal call population drift")

    x, sn1, sn2, pre, residual = first
    x_ref = FAKES.loop_reference(x)
    pre_ref = FAKES.loop_reference(pre)
    f_ref = FAKES.loop_reference(residual)
    expected = {
        "x_l1": x_ref["l1"],
        "x_l2_sq": x_ref["l2_sq"],
        "x_linf": x_ref["linf"],
        "sn1_nnz": np.count_nonzero(sn1, axis=-1).astype(np.int32),
        "sn2_nnz": np.count_nonzero(sn2, axis=-1).astype(np.int32),
        "pre_bn2_l1": pre_ref["l1"],
        "f_exact_zero": f_ref["exact_zero"],
        "f_l1": f_ref["l1"],
        "f_l2_sq": f_ref["l2_sq"],
        "f_linf": f_ref["linf"],
        "finite": x_ref["finite"] & pre_ref["finite"] & f_ref["finite"],
        "rho": f_ref["l1"] / np.maximum(
            x_ref["l1"], R3.BASE.DENOMINATOR_FLOOR),
    }
    mismatches = 0
    payload_path = Path(directory) / "s00_stage0_block0_ffn_metrics.npz"
    with np.load(payload_path, allow_pickle=False) as sealed:
        if set(sealed.files) != set(expected):
            raise AssertionError("literal NPZ member set drift")
        for key, reference in expected.items():
            if not np.array_equal(sealed[key], reference):
                mismatches += int(np.count_nonzero(sealed[key] != reference))
    capture.detach()
    return mismatches


def attack_matrix(root):
    target = R3.BASE.target_name(0, 0)
    matrix = []

    def run_attack(label, sequence, enforce=False, mutate=None, same_dir=None):
        directory = Path(same_dir) if same_dir is not None else Path(root) / label
        directory.mkdir(parents=True, exist_ok=True)

        def body():
            model, capture = new_capture(directory, enforce=enforce,
                                         mutate_model=mutate)
            payload = values(0, 0)
            sequence(model, capture, target, payload)
        matrix.append(expect_reject(label, body))

    run_attack("duplicate_pre_hook",
               lambda m, c, n, p: (m.named[n].fire_pre((p[0],)),
                                    m.named[n].fire_pre((p[0],))))
    run_attack("sn2_fc2_sn1_order",
               lambda m, c, n, p: (m.named[n].fire_pre((p[0],)),
                                    m.named[n + ".sn2"].fire_forward((p[3],), p[2]),
                                    m.named[n + ".fc2"].fire_forward((p[2],), p[3]),
                                    m.named[n + ".sn1"].fire_forward((p[0],), p[1])))
    run_attack("fc2_before_sn1",
               lambda m, c, n, p: (m.named[n].fire_pre((p[0],)),
                                    m.named[n + ".fc2"].fire_forward((p[2],), p[3])))
    run_attack("duplicate_sn1",
               lambda m, c, n, p: (m.named[n].fire_pre((p[0],)),
                                    m.named[n + ".sn1"].fire_forward((p[0],), p[1]),
                                    m.named[n + ".sn1"].fire_forward((p[0],), p[1])))
    run_attack("full_output_before_fc2",
               lambda m, c, n, p: (m.named[n].fire_pre((p[0],)),
                                    m.named[n + ".sn1"].fire_forward((p[0],), p[1]),
                                    m.named[n + ".sn2"].fire_forward((p[3],), p[2]),
                                    m.named[n].fire_forward((p[0],), p[4])))
    run_attack("wrong_h67_geometry",
               lambda m, c, n, p: m.named[n].fire_pre((p[0],)), enforce=True)

    def bad_channel(m, c, n, p):
        m.named[n].fire_pre((p[0],))
        m.named[n + ".sn1"].fire_forward((p[0],), p[1][..., :-1])
        m.named[n + ".sn2"].fire_forward((p[3],), p[2])
        m.named[n + ".fc2"].fire_forward((p[2],), p[3])
        m.named[n].fire_forward((p[0],), p[4])
    run_attack("sn1_channel_mismatch", bad_channel)

    def remove_target(model):
        del model.named[R3.BASE.target_name(3, 1)]
    run_attack("missing_target_module", lambda m, c, n, p: None,
               mutate=remove_target)

    overwrite_dir = Path(root) / "npz_overwrite"
    overwrite_dir.mkdir(parents=True, exist_ok=True)
    model, capture = new_capture(overwrite_dir, enforce=False)
    fire_normal(model, target, values(0, 0))
    capture.detach()
    run_attack("npz_overwrite", lambda m, c, n, p: fire_normal(m, n, p),
               same_dir=overwrite_dir)

    def missing_sn2(m, c, n, p):
        m.named[n].fire_pre((p[0],))
        m.named[n + ".sn1"].fire_forward((p[0],), p[1])
        c.end_sample()
    run_attack("missing_sn2_at_sample_end", missing_sn2)
    return matrix


def run_all():
    with tempfile.TemporaryDirectory(prefix="m460r3_cpu_") as directory:
        root = Path(directory)
        normal_dir = root / "normal"
        normal_dir.mkdir()
        mismatches = run_normal_micro(normal_dir)
        attacks = attack_matrix(root / "attacks")
    failed = [row["attack"] for row in attacks if not row["passes"]]
    result = {
        "status": "PASS_M460R3_CPU_MICRO_AND_STRICT_ORDER_ATTACKS",
        "normal_ffn_calls": 12,
        "normal_hooks": 60,
        "independent_reference_mismatches": mismatches,
        "attacks": attacks,
        "attack_total": len(attacks),
        "attack_passes": len(attacks) - len(failed),
        "failed_attacks": failed,
        "sn2_fc2_sn1_rejected": "sn2_fc2_sn1_order" not in failed,
        "gpu_touched": False,
        "remote_contacted": False,
        "training": False,
    }
    if mismatches or failed or len(attacks) < 7:
        raise AssertionError(json.dumps(result, sort_keys=True))
    return result


def test_m460r3_micro_and_attacks():
    result = run_all()
    assert result["independent_reference_mismatches"] == 0
    assert result["attack_total"] >= 7
    assert result["attack_total"] == result["attack_passes"]


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, sort_keys=True))

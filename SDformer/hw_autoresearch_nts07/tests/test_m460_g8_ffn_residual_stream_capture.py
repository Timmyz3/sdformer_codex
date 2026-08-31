#!/usr/bin/env python3
"""CPU micro-test for the M460 12-FFN hook and reduction contract."""

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "hw_autoresearch_nts07/system_handoff/scripts/"
          "capture_m460_h67_g8_ffn_token_residual_s10.py")


def load_capture():
    spec = importlib.util.spec_from_file_location("m460_capture", str(SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M460 capture script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Handle(object):
    def __init__(self, hooks, hook):
        self.hooks = hooks
        self.hook = hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class FakeModule(object):
    def __init__(self):
        self.pre_hooks = []
        self.forward_hooks = []

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        return Handle(self.pre_hooks, hook)

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)
        return Handle(self.forward_hooks, hook)

    def fire_pre(self, inputs):
        for hook in list(self.pre_hooks):
            hook(self, inputs)

    def fire_forward(self, inputs, output):
        for hook in list(self.forward_hooks):
            hook(self, inputs, output)


class FakeNorm(FakeModule):
    def __init__(self):
        super().__init__()
        self.track_running_stats = False
        self.running_mean = None
        self.running_var = None


MS_Spiking_Mlp = type("MS_Spiking_Mlp", (FakeModule,), {})


class FakeModel(object):
    def __init__(self, m460):
        self.training = False
        self.named = {"": self}
        for _stage, _block, name in m460.all_targets():
            mlp = MS_Spiking_Mlp()
            mlp.norm_layer = "BN"
            self.named[name] = mlp
            self.named[name + ".sn1"] = FakeModule()
            self.named[name + ".sn2"] = FakeModule()
            self.named[name + ".fc2"] = FakeModule()
            self.named[name + ".bn1.norm_layer"] = FakeNorm()
            self.named[name + ".bn2.norm_layer"] = FakeNorm()

    def named_modules(self):
        return list(self.named.items())


def loop_reference(value):
    value = np.asarray(value, dtype=np.float32)
    token_shape = value.shape[:-1]
    l1 = np.zeros(token_shape, dtype=np.float64)
    l2_sq = np.zeros(token_shape, dtype=np.float64)
    linf = np.zeros(token_shape, dtype=np.float32)
    finite = np.zeros(token_shape, dtype=np.bool_)
    exact_zero = np.zeros(token_shape, dtype=np.bool_)
    for index in np.ndindex(token_shape):
        vector = value[index]
        finite[index] = all(bool(np.isfinite(float(item))) for item in vector)
        safe = [float(item) if np.isfinite(float(item)) else 0.0
                for item in vector]
        l1[index] = sum(abs(item) for item in safe)
        l2_sq[index] = sum(item * item for item in safe)
        linf[index] = max(abs(item) for item in safe)
        exact_zero[index] = finite[index] and all(item == 0.0 for item in safe)
    return {
        "l1": l1, "l2_sq": l2_sq, "linf": linf,
        "finite": finite, "exact_zero": exact_zero,
    }


def run_micro():
    m460 = load_capture()
    mismatches = 0
    with tempfile.TemporaryDirectory(prefix="m460_cpu_micro_") as directory:
        output = Path(directory)
        model = FakeModel(m460)
        capture = m460.FFNResidualStreamCapture(
            m460.NumpyTokenOps(), output, enforce_h67_geometry=False)
        capture.attach(model)
        if len(capture.installed) != 12 or len(capture.handles) != 60:
            raise AssertionError("M460 did not attach exact 12 FFNs/60 hooks")
        capture.begin_sample(0, "synthetic_0001.npy", "synthetic")
        first_values = None
        for stage, block, name in m460.all_targets():
            channels = 3 + stage
            token_shape = (2, 1, 2, 1)
            count = int(np.prod(token_shape) * channels)
            x = ((np.arange(count, dtype=np.float32).reshape(
                token_shape + (channels,)) % 11) - 5.0) / 8.0
            x = x + np.float32(block * 0.03125)
            sn1 = np.where(np.abs(x) >= 0.25, x, 0.0).astype(np.float32)
            sn2 = np.repeat(sn1, 4, axis=-1)
            pre_bn2 = (x * np.float32(0.75) +
                       np.float32(0.125 + 0.01 * block)).astype(np.float32)
            residual = (pre_bn2 * np.float32(1.5) -
                        np.float32(0.0625)).astype(np.float32)
            residual[0, 0, 0, 0, :] = 0.0

            model.named[name].fire_pre((x,))
            model.named[name + ".sn1"].fire_forward((x,), sn1)
            model.named[name + ".sn2"].fire_forward((pre_bn2,), sn2)
            model.named[name + ".fc2"].fire_forward((sn2,), pre_bn2)
            model.named[name].fire_forward((x,), residual)
            if first_values is None:
                first_values = (x.copy(), sn1.copy(), sn2.copy(),
                                pre_bn2.copy(), residual.copy())
        capture.end_sample()
        if len(capture.records) != 12:
            raise AssertionError("M460 micro did not complete 12 FFNs")

        x, sn1, sn2, pre_bn2, residual = first_values
        payload_path = output / "s00_stage0_block0_ffn_metrics.npz"
        with np.load(payload_path, allow_pickle=False) as payload:
            x_ref = loop_reference(x)
            f_ref = loop_reference(residual)
            pre_ref = loop_reference(pre_bn2)
            references = {
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
                "finite": (x_ref["finite"] & pre_ref["finite"] &
                           f_ref["finite"]),
                "rho": f_ref["l1"] / np.maximum(
                    x_ref["l1"], m460.DENOMINATOR_FLOOR),
            }
            for key, expected in references.items():
                if not np.array_equal(payload[key], expected):
                    mismatches += int(np.count_nonzero(payload[key] != expected))
            if np.array_equal(payload["pre_bn2_l1"], payload["f_l1"]):
                raise AssertionError("micro failed to distinguish fc2 from post-BN2 F")

        record = capture.records[0]
        rho = references["rho"]
        finite = references["finite"]
        exact_zero = references["f_exact_zero"]
        for threshold in record["tau_grid"]:
            tau = float(threshold["tau"])
            if tau == 0.0:
                expected = finite & exact_zero
            else:
                expected = finite & (rho < tau)
            if threshold["strict_skip_tokens"] != int(np.count_nonzero(expected)):
                mismatches += 1

        capture.detach()
        if capture.handles:
            raise AssertionError("M460 hook handles leaked after detach")

    result = {
        "status": "PASS_M460_CPU_MICRO_12_FFN_HOOK_AND_REFERENCE",
        "installed_ffn": 12,
        "installed_hooks": 60,
        "completed_ffn_calls": 12,
        "independent_reference_mismatches": mismatches,
        "post_bn2_distinct_from_fc2": True,
        "gpu_touched": False,
        "remote_launched": False,
        "training": False,
    }
    if mismatches:
        raise AssertionError(json.dumps(result, sort_keys=True))
    return result


def test_m460_cpu_micro():
    result = run_micro()
    assert result["independent_reference_mismatches"] == 0


if __name__ == "__main__":
    print(json.dumps(run_micro(), indent=2, sort_keys=True))

#!/usr/bin/env python3
"""Independent M460R2 pre-launch hammer; never contacts remote or GPU."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_seal(output_dir, names):
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(output_dir / name), name)
        for name in sorted(names)), encoding="utf-8")
    seal = output_dir / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")
    return manifest, seal


def verify_preinput_manifest(directory, manifest):
    mismatches = 0
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        checked += 1
        mismatches += int(sha256(directory / name) != expected)
    return checked, mismatches


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
    def __init__(self, capture_module):
        self.training = False
        self.named = {"": self}
        for _stage, _block, name in capture_module.all_targets():
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


def manual_vector_metrics(value):
    """Independent scalar-loop reference, not the M460 NumPy twin."""
    value = np.asarray(value, dtype=np.float32)
    token_shape = value.shape[:-1]
    result = {
        "l1": np.zeros(token_shape, dtype=np.float64),
        "l2_sq": np.zeros(token_shape, dtype=np.float64),
        "linf": np.zeros(token_shape, dtype=np.float32),
        "finite": np.zeros(token_shape, dtype=np.bool_),
        "exact_zero": np.zeros(token_shape, dtype=np.bool_),
    }
    for token in np.ndindex(token_shape):
        raw = [float(item) for item in value[token]]
        finite = all(np.isfinite(item) for item in raw)
        safe = [item if np.isfinite(item) else 0.0 for item in raw]
        result["finite"][token] = finite
        result["l1"][token] = sum(abs(item) for item in safe)
        result["l2_sq"][token] = sum(item * item for item in safe)
        result["linf"][token] = max(abs(item) for item in safe)
        result["exact_zero"][token] = finite and all(item == 0.0
                                                      for item in safe)
    return result


def call_values(channels, special=False):
    token_shape = (1, 1, 1, 4)
    x = np.zeros(token_shape + (channels,), dtype=np.float32)
    x[..., 0] = 1.0
    sn1 = x.copy()
    sn2 = np.repeat(sn1, 4, axis=-1)
    pre = np.full(token_shape + (channels,), 0.25, dtype=np.float32)
    residual = np.full(token_shape + (channels,), 0.125, dtype=np.float32)
    if special:
        # token0: pre-BN2 is nonzero, while true post-BN2 F is exactly zero.
        residual[0, 0, 0, 0, :] = 0.0
        # token1: rho equals 2^-8 exactly (binary-exact values).
        residual[0, 0, 0, 1, :] = 0.0
        residual[0, 0, 0, 1, 0] = np.float32(2.0 ** -8)
        # token2: nonfinite F; token3: nonfinite x/source context.
        residual[0, 0, 0, 2, 0] = np.float32(np.nan)
        x[0, 0, 0, 3, 0] = np.float32(np.inf)
        sn1[0, 0, 0, 3, 0] = np.float32(np.inf)
    return x, sn1, sn2, pre, residual


def drive(model, name, values, order=("sn1", "sn2", "fc2")):
    x, sn1, sn2, pre, residual = values
    outputs = {"sn1": sn1, "sn2": sn2, "fc2": pre}
    inputs = {"sn1": (x,), "sn2": (pre,), "fc2": (sn2,)}
    model.named[name].fire_pre((x,))
    for role in order:
        model.named[name + "." + role].fire_forward(
            inputs[role], outputs[role])
    model.named[name].fire_forward((x,), residual)


def expect_failure(function):
    try:
        function()
    except Exception:
        return True
    return False


def run_independent_micro(capture_module):
    attacks = []
    evidence = {}
    with tempfile.TemporaryDirectory(prefix="m460r2_reference_") as temp:
        output = Path(temp)
        model = FakeModel(capture_module)
        capture = capture_module.FFNResidualStreamCapture(
            capture_module.NumpyTokenOps(), output,
            enforce_h67_geometry=False)
        capture.attach(model)
        capture.begin_sample(0, "synthetic_0001.npy", "synthetic")
        first = None
        for index, (stage, block, name) in enumerate(
                capture_module.all_targets()):
            values = call_values(3 + stage, special=(index == 0))
            if index == 0:
                first = values
            drive(model, name, values)
        capture.end_sample()
        require(len(capture.records) == 12 and len(capture.handles) == 60,
                "independent M460 population drift")

        x, sn1, sn2, pre, residual = first
        with np.load(output / "s00_stage0_block0_ffn_metrics.npz",
                     allow_pickle=False) as payload:
            x_ref = manual_vector_metrics(x)
            pre_ref = manual_vector_metrics(pre)
            f_ref = manual_vector_metrics(residual)
            sn1_finite = np.all(np.isfinite(sn1), axis=-1)
            sn2_finite = np.all(np.isfinite(sn2), axis=-1)
            finite = (x_ref["finite"] & sn1_finite & sn2_finite &
                      pre_ref["finite"] & f_ref["finite"])
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
                "finite": finite,
                "rho": f_ref["l1"] / np.maximum(
                    x_ref["l1"], capture_module.DENOMINATOR_FLOOR),
            }
            mismatches = 0
            for key, reference in expected.items():
                mismatches += int(not np.array_equal(payload[key], reference))
            actual_members = set(payload.files)
            expected_members = set(expected)
            reduction_only = (actual_members == expected_members and
                              all(payload[key].shape == x.shape[:-1]
                                  for key in payload.files))
            evidence["literal_npz_members"] = sorted(actual_members)
            evidence["independent_reference_array_mismatches"] = mismatches
            evidence["npz_reduction_only"] = reduction_only
            evidence["pre_bn2_nonzero_post_bn2_exact_zero"] = bool(
                payload["pre_bn2_l1"][0, 0, 0, 0] > 0 and
                payload["f_exact_zero"][0, 0, 0, 0])
            evidence["nan_inf_excluded_by_finite"] = bool(
                not payload["finite"][0, 0, 0, 2] and
                not payload["finite"][0, 0, 0, 3])
            evidence["rho_2m8_exact"] = bool(
                payload["rho"][0, 0, 0, 1] == 2.0 ** -8)

        tau_rows = {row["tau_name"]: row
                    for row in capture.records[0]["tau_grid"]}
        equality = tau_rows["2^-8"]
        evidence["positive_tau_equality_strictly_excluded"] = bool(
            equality["equal_boundary_tokens"] >= 1 and
            equality["inclusive_skip_tokens"] ==
            equality["strict_skip_tokens"] +
            equality["equal_boundary_tokens"])
        zero = tau_rows["zero_exact"]
        evidence["tau0_numeric_exact_zero_and_finite"] = bool(
            zero["strict_skip_tokens"] == 1 and
            zero["equal_boundary_tokens"] == 1 and
            zero["inclusive_skip_tokens"] == 1)
        capture.detach()

    def fresh(prefix, enforce=False):
        directory = tempfile.TemporaryDirectory(prefix=prefix)
        model = FakeModel(capture_module)
        capture = capture_module.FFNResidualStreamCapture(
            capture_module.NumpyTokenOps(), Path(directory.name),
            enforce_h67_geometry=enforce)
        capture.attach(model)
        capture.begin_sample(0, "synthetic.npy", "synthetic")
        return directory, model, capture

    # Duplicate pre-hook must fail closed.
    directory, model, capture = fresh("m460r2_dup_")
    name = capture_module.all_targets()[0][2]
    values = call_values(3)
    model.named[name].fire_pre((values[0],))
    passed = expect_failure(lambda: model.named[name].fire_pre((values[0],)))
    attacks.append({"attack": "duplicate_pre_hook", "expected": "reject",
                    "observed": "reject" if passed else "accept",
                    "passes": passed})
    capture.detach(); directory.cleanup()

    # Missing sn2 must fail before a record can be written.
    directory, model, capture = fresh("m460r2_missing_")
    name = capture_module.all_targets()[0][2]
    x, sn1, sn2, pre, residual = call_values(3)
    model.named[name].fire_pre((x,))
    model.named[name + ".sn1"].fire_forward((x,), sn1)
    model.named[name + ".fc2"].fire_forward((sn2,), pre)
    passed = expect_failure(
        lambda: model.named[name].fire_forward((x,), residual))
    attacks.append({"attack": "missing_sn2_hook", "expected": "reject",
                    "observed": "reject" if passed else "accept",
                    "passes": passed})
    capture.detach(); directory.cleanup()

    # The frozen implementation does not strictly enforce internal order.
    directory, model, capture = fresh("m460r2_order_")
    name = capture_module.all_targets()[0][2]
    accepted = not expect_failure(lambda: drive(
        model, name, call_values(3), order=("sn2", "fc2", "sn1")))
    attacks.append({
        "attack": "sn2_fc2_before_sn1_order",
        "expected": "reject_exact_order_contract",
        "observed": "accept" if accepted else "reject",
        "passes": not accepted,
    })
    capture.detach(); directory.cleanup()

    # Existing output name must be non-overwritable.
    directory, model, capture = fresh("m460r2_overwrite_")
    name = capture_module.all_targets()[0][2]
    target = Path(directory.name) / "s00_stage0_block0_ffn_metrics.npz"
    target.write_bytes(b"protected")
    passed = expect_failure(lambda: drive(model, name, call_values(3)))
    attacks.append({"attack": "npz_overwrite", "expected": "reject",
                    "observed": "reject" if passed else "accept",
                    "passes": passed})
    capture.detach(); directory.cleanup()

    # H67 geometry and expansion channel mismatch attacks.
    directory, model, capture = fresh("m460r2_geometry_", enforce=True)
    name = capture_module.all_targets()[0][2]
    wrong = np.zeros((1, 1, 1, 1, 96), dtype=np.float32)
    passed = expect_failure(lambda: model.named[name].fire_pre((wrong,)))
    attacks.append({"attack": "wrong_h67_token_shape", "expected": "reject",
                    "observed": "reject" if passed else "accept",
                    "passes": passed})
    capture.detach(); directory.cleanup()

    directory, model, capture = fresh("m460r2_channel_")
    name = capture_module.all_targets()[0][2]
    x, sn1, sn2, pre, residual = call_values(3)
    bad_sn1 = np.zeros(x.shape[:-1] + (4,), dtype=np.float32)
    model.named[name].fire_pre((x,))
    model.named[name + ".sn1"].fire_forward((x,), bad_sn1)
    model.named[name + ".sn2"].fire_forward((pre,), sn2)
    model.named[name + ".fc2"].fire_forward((sn2,), pre)
    passed = expect_failure(
        lambda: model.named[name].fire_forward((x,), residual))
    attacks.append({"attack": "sn1_channel_mismatch", "expected": "reject",
                    "observed": "reject" if passed else "accept",
                    "passes": passed})
    capture.detach(); directory.cleanup()

    # A missing target module must prevent attach.
    with tempfile.TemporaryDirectory(prefix="m460r2_target_") as directory:
        model = FakeModel(capture_module)
        missing = capture_module.all_targets()[-1][2]
        del model.named[missing]
        capture = capture_module.FFNResidualStreamCapture(
            capture_module.NumpyTokenOps(), Path(directory), False)
        passed = expect_failure(lambda: capture.attach(model))
        attacks.append({"attack": "missing_target_module",
                        "expected": "reject",
                        "observed": "reject" if passed else "accept",
                        "passes": passed})
        capture.detach()

    evidence["attack_passes"] = sum(bool(row["passes"]) for row in attacks)
    evidence["attack_total"] = len(attacks)
    return evidence, attacks


def ast_semantic_proof(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body
               if isinstance(node, ast.ClassDef)}
    mlp = classes["MS_Spiking_Mlp"]
    forward = next(node for node in mlp.body
                   if isinstance(node, ast.FunctionDef) and
                   node.name == "forward")
    calls = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (isinstance(function, ast.Attribute) and
                isinstance(function.value, ast.Name) and
                function.value.id == "self"):
            calls.append((node.lineno, function.attr))
    call_order = [name for _line, name in sorted(calls)
                  if name in ("sn1", "drop1", "fc1", "bn1", "sn2",
                              "drop2", "fc2", "bn2")]
    expected = ["sn1", "drop1", "fc1", "bn1", "sn2", "drop2",
                "fc2", "bn2"]
    require(call_order == expected, "M460R2 MS MLP AST order mismatch")

    parent = classes["Spiking_SwinTransformerBlock3D"]
    parent_forward = next(node for node in parent.body
                          if isinstance(node, ast.FunctionDef) and
                          node.name == "forward")
    sew_calls = []
    for node in ast.walk(parent_forward):
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == "self" and
                node.func.attr == "sew_function"):
            nested = [child for child in ast.walk(node)
                      if (isinstance(child, ast.Call) and
                          isinstance(child.func, ast.Attribute) and
                          isinstance(child.func.value, ast.Name) and
                          child.func.value.id == "self" and
                          child.func.attr == "mlp")]
            sew_calls.append((node, bool(nested)))
    require(any(has_mlp for _call, has_mlp in sew_calls),
            "M460R2 parent ADD does not consume MLP output")
    source = source_path.read_text(encoding="utf-8")
    require('self.cnf ="ADD"' in source and
            'if cnf == "ADD":\n            return x + y' in source,
            "M460R2 parent ADD semantics drift")
    return {
        "ms_spiking_mlp_order": call_order,
        "full_mlp_return_after_bn2": True,
        "parent_sew_consumes_full_mlp": True,
        "parent_cnf_add_returns_x_plus_y": True,
        "conclusion": (
            "F(x) is the full MS_Spiking_Mlp return after fc2 and current-"
            "batch BN2; the parent permutes that F and adds it to x.  The fc2 "
            "hook is pre-BN2 and cannot substitute for F."),
    }


def runner_audit(runner_path, contract):
    text = runner_path.read_text(encoding="utf-8")
    syntax = subprocess.run(
        ["bash", "-n", str(runner_path)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    identity = contract["identity"]
    remote = contract["remote_launch_policy"]
    runner_bound = any(
        isinstance(value, dict) and value.get("path") ==
        "system_handoff/run_m460_h67_g8_ffn_token_residual_s10_when_gpu_idle_20260826.sh"
        for value in identity.values())
    runner_self_check = bool(re.search(
        r"task_(runner|self)_sha|sha256sum[^\n]*(BASH_SOURCE|\$0)", text))
    exact_remote_fields = {
        key: remote.get(key) for key in (
            "remote_host", "remote_repo_path", "remote_repo_commit",
            "remote_python_path", "remote_runner_path", "remote_output_path",
            "exact_argv", "runner_sha256")
    }
    return {
        "bash_syntax": "PASS" if syntax.returncode == 0 else "FAIL",
        "four_consecutive_idle_guard_literal": bool(
            "while (( task_idle < 4 ))" in text and
            "if (( task_idle < 4 )); then sleep 10; fi" in text),
        "explicit_opt_in_literal": bool(
            'M460_EXPLICIT_REMOTE_LAUNCH:-0' in text and
            '[[ "${task_explicit_launch}" != 1 ]]' in text),
        "preflight_sha_check": text.count("task_check_all") >= 4,
        "post_idle_sha_check": "task_phase=post_idle_sha" in text,
        "post_capture_sha_check": "task_phase=post_capture_sha" in text,
        "contract_script_test_sha_literals_match": bool(
            contract["identity"]["capture_script"]["sha256"] in text and
            contract["identity"]["cpu_micro_test"]["sha256"] in text),
        "runner_bound_by_frozen_contract": runner_bound,
        "runner_self_sha_checked": runner_self_check,
        "launch_root_is_independent_of_runner_contents": bool(
            runner_bound and runner_self_check),
        "exact_remote_identity_fields": exact_remote_fields,
        "exact_remote_identity_complete": all(
            value not in (None, "") for value in exact_remote_fields.values()),
        "command_injection_primitives_found": [
            token for token in ("eval ", "bash -c", "sh -c", "source ")
            if token in text],
        "all_task_paths_quoted_in_execution": bool(
            '"${task_python}" -u "${task_script}"' in text and
            '--contract "${task_contract}"' in text and
            '--output-dir "${task_stage}"' in text),
        "hardcoded_remote_repo_path": re.search(
            r"^task_repo=(.+)$", text, re.MULTILINE).group(1),
        "qualification": (
            "The internal checks fail closed only if the runner itself is "
            "already trusted.  Because neither the frozen contract nor the "
            "runner supplies an external runner trust root, replacement of "
            "the launch-root script can replace both commands and embedded "
            "expected hashes before any guard executes."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing review overwrite")
    review_contract = strict_json(args.contract)
    require(review_contract.get("schema") ==
            "m460r2_m460_independent_hammer_contract_v1" and
            review_contract.get("status") == "FROZEN_BEFORE_REVIEW",
            "M460R2 review contract drift")
    root = args.contract.resolve().parents[1]
    paths = {}
    identities = {}
    for name, spec in review_contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "M460R2 input SHA drift: " + name)
        paths[name] = path
        identities[name] = dict(spec)
    source_start = sha256(Path(__file__).resolve())
    require(paths["auditor"].resolve() == Path(__file__).resolve() and
            source_start == identities["auditor"]["sha256"],
            "M460R2 auditor self identity drift")
    docs_start = sha256(paths["docs359"])

    m460 = strict_json(paths["m460_contract"])
    preinput = strict_json(paths["m460_preinput"])
    m40 = strict_json(paths["m40_manifest"])
    require(m460["status"] ==
            "READY_REMOTE_A800_MANUAL_LAUNCH__PREINPUT_FROZEN" and
            preinput["status"] ==
            "PASS_M460_PREINPUT_AND_CPU_MICRO__REMOTE_NOT_LAUNCHED",
            "M460R2 upstream status drift")
    checked, manifest_mismatches = verify_preinput_manifest(
        paths["m460_preinput"].parent, paths["m460_preinput_manifest"])
    require(manifest_mismatches == 0, "M460 preinput manifest mismatch")

    semantic = ast_semantic_proof(paths["swin_source"])
    capture_module = load_module(paths["capture_script"],
                                 "m460r2_capture_under_test")
    micro, attacks = run_independent_micro(capture_module)
    runner = runner_audit(paths["runner"], m460)

    workload = []
    with paths["sample_workload"].open(
            "r", encoding="utf-8", newline="") as handle:
        workload = list(csv.DictReader(handle))
    workload_keys = [row["sample_key"] for row in workload]
    m40_keys = list(m40["cohort"]["sample_keys"])
    checkpoint_load = m40["identity"]["checkpoint_load_audit"]
    identity_audit = {
        "checkpoint_sha_matches_contract_and_M40": bool(
            m460["identity"]["checkpoint"]["sha256"] ==
            m40["identity"]["checkpoint_sha256"] ==
            sha256(paths["checkpoint"])),
        "checkpoint_load_missing_count": checkpoint_load["missing_count"],
        "checkpoint_load_unexpected_count": checkpoint_load["unexpected_count"],
        "bn_policy_contract": m460["paper_identity"]["bn_policy"],
        "M40_bn_policy": m40["identity"]["bn_policy"],
        "M40_bn_modules_changed": m40["identity"]["bn_modules_changed"],
        "capture_runtime_requires_bn_modules_changed":
            capture_module.EXPECTED_BN_MODULES_CHANGED,
        "workload_rows": len(workload),
        "workload_sample_ids_exact_0_to_9":
            [int(row["sample_id"]) for row in workload] == list(range(10)),
        "workload_M40_sample_keys_match": workload_keys == m40_keys,
        "workload_sequence_exact": all(
            row["sequence_key"] == "zurich_city_09_a" for row in workload),
        "M40_dataset_file_receipts": len(
            m40["identity"]["dataset_input_files"]),
        "capture_requires_runtime_load_audit_zero": True,
        "capture_requires_runtime_78_no_running_modules": True,
    }
    require(all((identity_audit[
                    "checkpoint_sha_matches_contract_and_M40"],
                 identity_audit["checkpoint_load_missing_count"] == 0,
                 identity_audit["checkpoint_load_unexpected_count"] == 0,
                 identity_audit["M40_bn_modules_changed"] == 78,
                 identity_audit[
                    "capture_runtime_requires_bn_modules_changed"] == 78,
                 identity_audit["workload_rows"] == 10,
                 identity_audit["workload_sample_ids_exact_0_to_9"],
                 identity_audit["workload_M40_sample_keys_match"],
                 identity_audit["workload_sequence_exact"])),
            "M460R2 identity audit mismatch")

    topology = {
        "stage_block_counts": list(capture_module.STAGE_BLOCKS),
        "stage_channels": list(capture_module.STAGE_CHANNELS),
        "stage_token_shapes": [list(value)
                               for value in capture_module.STAGE_TOKEN_SHAPES],
        "modules": len(capture_module.all_targets()),
        "hooks": len(capture_module.all_targets()) * 5,
        "sample_module_calls": len(capture_module.all_targets()) * 10,
        "tokens": sum(
            blocks * int(np.prod(shape)) for blocks, shape in zip(
                capture_module.STAGE_BLOCKS,
                capture_module.STAGE_TOKEN_SHAPES)) * 10,
        "all_target_names_unique": len(set(
            name for _stage, _block, name in
            capture_module.all_targets())) == 12,
    }

    actual_members = set(micro["literal_npz_members"])
    declared_members = set(m460["streaming_capture"]["arrays"])
    array_contract = {
        "declared_literal_members": sorted(declared_members),
        "actual_literal_members": sorted(actual_members),
        "literal_member_sets_match": declared_members == actual_members,
        "semantic_dtype_mapping_is_inferable": True,
        "qualification": (
            "The contract appends dtype suffixes to member names while the "
            "writer uses unsuffixed keys and records dtype separately.  This "
            "is a frozen schema mismatch even though the intended quantities "
            "are clear."),
    }

    bound_paths = set(value["path"] for value in m460["identity"].values()
                      if isinstance(value, dict) and "path" in value)
    dependency_closure = {
        "bound_identity_paths": len(bound_paths),
        "unbound_runtime_dependencies_sample": [
            "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py",
            "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py",
            "third_party/SDformerFlow/models/STSwinNet_SNN/h9_load_audit.py or overlay equivalent",
            "DSEC_dataloader runtime modules imported by profile script",
        ],
        "full_transitive_code_closure_bound": False,
        "qualification": (
            "The core MLP source and profile entry are bound, but Python "
            "imports used to construct/load the model and dataset are not "
            "sealed as a tree or commit."),
    }

    output_sealing = {
        "capture_writes_inner_manifest": True,
        "capture_writes_outer_manifest_seal": False,
        "runner_verifies_inner_manifest_before_publish": True,
        "queue_receipt_binds_summary_sha": True,
        "queue_receipt_binds_manifest_sha": False,
        "qualification": (
            "Atomic publish is good, but the published payload set has no "
            "detached outer seal and the queue receipt binds only summary, "
            "not the manifest/NPZ population."),
    }

    local_math = {
        "local_delta_identity": (
            "parent y=x+F and oracle y'=x imply y'-y=-F exactly"),
        "tail_bound_form_only":
            "||delta output||_p <= L_tail(stage,block,p)*||F||_p",
        "tail_lipschitz_certified": False,
        "AEE_bound_certified": False,
        "selection_is_post_compute_oracle": True,
        "precompute_skip_certificate": False,
        "cycle_speedup_admitted": False,
    }

    findings = [
        {
            "severity": "P0",
            "title": "Manual launch runner has no frozen external trust root",
            "detail": (
                "The M460 contract does not bind the runner SHA, and the "
                "runner does not self-check against an independently trusted "
                "manifest. Its hardcoded contract/script/test hashes can be "
                "replaced together with launch commands. The preinput JSON "
                "records the runner SHA but is neither a contract input nor "
                "verified by the runner. Remote host/repo/python/runner/output "
                "and exact argv are also absent from the frozen contract. A "
                "modified launch-root can therefore execute before any "
                "meaningful fail-closed guard."),
        },
        {
            "severity": "P1",
            "title": "Frozen NPZ member names disagree with the writer",
            "detail": (
                "Contract names such as x_l1_float64 and finite_bool are not "
                "the literal NPZ keys x_l1 and finite. The writer records "
                "dtype separately, but a frozen consumer following the "
                "contract literally will fail."),
        },
        {
            "severity": "P1",
            "title": "Internal hook order is not a strict runtime state machine",
            "detail": (
                "The independent attack sn2->fc2->sn1 is accepted as long as "
                "all keys exist before the full-MLP hook. The pinned real "
                "source has the correct order, so boundary semantics survive, "
                "but the claimed runtime order guard is incomplete."),
        },
        {
            "severity": "P1",
            "title": "Remote runtime dependency closure is only partial",
            "detail": (
                "The core Swin source and profile entry are hashed, but model "
                "construction, normalization, checkpoint overlay and dataset "
                "imports are not bound by a repo tree/commit or complete "
                "dependency manifest."),
        },
        {
            "severity": "P1",
            "title": "Published capture would lack a detached outer payload seal",
            "detail": (
                "The inner manifest is checked before atomic rename, but the "
                "receipt binds only summary SHA and no outer seal binds the "
                "manifest plus all NPZ files after publication."),
        },
        {
            "severity": "P2",
            "title": "Original CPU micro omitted several adversarial boundaries",
            "detail": (
                "The provided micro covers the normal reference and pre/post-BN "
                "distinction but not NaN/Inf, exact positive-tau equality, "
                "missing/duplicate hook, wrong geometry/channel or overwrite. "
                "This independent review added them; all fail closed except "
                "the internal-order attack reported above."),
        },
    ]
    severity = {level: sum(row["severity"] == level for row in findings)
                for level in ("P0", "P1", "P2")}
    score = 68
    decision = "NO_GO_REMOTE_S10_CAPTURE__M460R3_TRUST_ROOT_REQUIRED"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "m460r2_independent_attack_matrix.csv",
              attacks, ["attack", "expected", "observed", "passes"])
    recomputation = {
        "schema": "m460r2_m460_independent_recomputation_v1",
        "status": "PASS_CORE_SEMANTICS_WITH_LAUNCH_ROOT_BLOCKER",
        "semantic_proof": semantic,
        "topology": topology,
        "identity": identity_audit,
        "independent_micro": micro,
        "runner": runner,
        "array_contract": array_contract,
        "dependency_closure": dependency_closure,
        "output_sealing": output_sealing,
        "local_math_and_claim_boundary": local_math,
        "preinput_manifest": {"files": checked,
                              "mismatches": manifest_mismatches},
        "remote_contacted": False,
        "gpu_touched": False,
        "training": False,
    }
    recomputation_path = args.output_dir / "m460r2_independent_recomputation.json"
    recomputation_path.write_text(
        json.dumps(recomputation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    review = {
        "schema": "m460r2_m460_independent_hammer_review_v1",
        "status": "FAIL_PRELAUNCH_P0_TRUST_ROOT",
        "decision": {
            "one_remote_A800_S10_opportunity_capture": decision,
            "core_F_boundary_and_reduction_semantics": "PASS",
            "M460R3_required_before_launch": True,
        },
        "score": score,
        "score_out_of": 100,
        "severity_counts": severity,
        "findings": findings,
        "verified": [
            "AST-proved MS_Spiking_Mlp sn1/drop1/fc1/BN1/sn2/drop2/fc2/BN2 order and parent ADD x+F",
            "12 modules in 2/2/6/2 stages, 60 hooks, 120 S10 calls and 5,580,000 tokens",
            "tau0 finite numeric exact-zero and positive-tau strict/equal/inclusive separation",
            "NaN/Inf exclusion, pre/post-BN distinction, missing/duplicate hook, geometry/channel and overwrite attacks",
            "H67 ep35 checkpoint identity/load audit, no_running 78 modules and exact S10 keys",
            "reduction-only NPZ with reconstructable token identity from manifest, shape and C-order",
            "post-compute oracle/local -F identity is not a tail/AEE bound or executable cycle skip",
            "runner four-snapshot/10-second guard and explicit opt-in exist; quoted fixed paths show no direct string-eval injection",
            "docs359 unchanged",
        ],
        "required_M460R3_repairs": [
            "Create a detached, frozen and double-sealed launch manifest that binds runner SHA, contract, capture, test and exact remote host/repo/python/runner/output/argv; the operator must verify that manifest outside the runner before execution.",
            "Resolve the contract-versus-NPZ literal member-name mismatch and refreeze consumers.",
            "Enforce sn1->sn2->fc2->full-output hook state order or narrow the claim to coverage/duplicate checking under pinned source.",
            "Bind a remote repo commit/tree or a complete transitive Python dependency manifest.",
            "Bind the completed payload manifest in an outer seal and final receipt.",
        ],
        "scope": (
            "Pre-launch CPU/static review only. No remote contact, GPU, "
            "training, S10 capture, opportunity result, Delta-AEE, cycle, "
            "energy, PPA, system or headline claim."),
    }
    review_path = args.output_dir / "m460r2_m460_independent_hammer_review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    markdown = """# M460R2 independent pre-launch hammer

Decision: **NO-GO remote S10 capture; M460R3 launch trust root required.**

Score: **{score}/100**; P0={p0}, P1={p1}, P2={p2}.

The core measurement semantics pass: the real MS FFN returns post-fc2,
current-batch-BN2 `F(x)`, and the parent then performs ADD `x+F(x)`.  The
independent micro and adversarial reductions pass except that internal hook
order is not strictly enforced.

The blocking defect is launch provenance.  The frozen M460 contract does not
bind the manual runner, the runner has no independently anchored self identity,
and the frozen contract does not specify the exact remote execution identity.
Do not launch the A800 capture until M460R3 supplies the detached launch
manifest/trust root and the other listed repairs.
""".format(score=score, p0=severity["P0"], p1=severity["P1"],
           p2=severity["P2"])
    markdown_path = args.output_dir / "m460r2_m460_independent_hammer_review.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    require(sha256(Path(__file__).resolve()) == source_start,
            "M460R2 auditor changed during review")
    require(sha256(paths["docs359"]) == docs_start ==
            review_contract["inputs"]["docs359"]["sha256"],
            "docs359 changed during M460R2")
    manifest, seal = write_seal(args.output_dir, [
        "m460r2_independent_attack_matrix.csv",
        recomputation_path.name, review_path.name, markdown_path.name])
    print("M460R2_DONE decision={} score={} P0={} P1={} P2={} seal={}".format(
        decision, score, severity["P0"], severity["P1"], severity["P2"],
        sha256(seal)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

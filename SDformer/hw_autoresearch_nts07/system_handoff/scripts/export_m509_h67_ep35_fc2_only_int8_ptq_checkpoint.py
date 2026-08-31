#!/usr/bin/env python3
"""Export a frozen H67 ep35 FC2-only row-wise INT8 PTQ checkpoint.

This exporter changes only the twelve ``mlp.fc2.weight`` tensors.  Each output
row uses a symmetric scale ``max(abs(w))/127`` and an INT8 code in [-127,127].
The checkpoint stores the dequantized weights so the unmodified PyTorch model
can be evaluated.  A separate NPZ stores the exact INT8 codes and scales used
by hardware.  No accuracy, cycle, PPA, or system claim is made here.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
FROZEN = {
    "checkpoint": (
        HW / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
        "h67_ep35_system_trace_handoff_20260821/checkpoint/"
        "checkpoint_epoch35.pth",
        "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    ),
    "m51_manifest": (
        HW / "system_handoff/incoming/m51_capture_bundle_r2_20260823/"
        "manifest.json",
        "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    ),
    "m160_parameters": (
        HW / "results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/"
        "per_ffn_bn_atlif_fusion.csv",
        "309a5d802c7e49d432285f09ff43b9d1ec797db815b949cd34798c0a94f4f464",
    ),
    "docs359": (
        HW / "docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    ),
}


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
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            RuntimeError("non-standard JSON token: " + value)),
    )


def signed_bits_for_magnitude(magnitude):
    magnitude = int(magnitude)
    require(magnitude >= 0, "negative magnitude")
    bits = 1
    while magnitude > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def install_pickle_import_paths():
    """Install the exact compatibility namespaces needed by old H67 pickles.

    The frozen ep35 checkpoint is a full-model pickle, not a plain state-dict
    wrapper.  This mirrors the already-used M32 extractor before torch.load.
    """
    baseline_root = ROOT / "third_party" / "SDformerFlow"
    overlay_root = (
        ROOT / "neuron_experiments" / "H9_bipolar_self_attention" / "overlay"
    )
    for path in (ROOT, baseline_root, overlay_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(overlay_root / "models")
    overlay_stsnn = str(overlay_root / "models" / "STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    import models.STSwinNet_SNN.atlif_ternary_psn  # noqa: F401
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat

    register_shiftmax_pickle_compat()


def find_state_dict(payload):
    candidates = []
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        value = payload.state_dict()
        candidates.append(("<full_model>.state_dict()", value))
    elif isinstance(payload, dict):
        for label in ("state_dict", "model_state_dict", "model"):
            value = payload.get(label)
            if hasattr(value, "state_dict") and not isinstance(value, dict):
                value = value.state_dict()
            if isinstance(value, dict) and value and all(
                    isinstance(key, str) for key in value):
                candidates.append((label, value))
        if payload and all(isinstance(key, str) for key in payload) and any(
                torch.is_tensor(value) for value in payload.values()):
            candidates.append(("<root>", payload))
    qualified = []
    for label, state in candidates:
        targets = sorted(
            key for key, value in state.items()
            if key.endswith(".mlp.fc2.weight") and torch.is_tensor(value))
        if len(targets) == 12:
            qualified.append((label, state, targets))
    require(len(qualified) == 1,
            "expected one checkpoint state dictionary with twelve FC2 weights")
    return qualified[0]


def canonical_state_copy(state):
    """Detach the deployment state from the unpickled model container."""
    copied = OrderedDict()
    for key, value in state.items():
        require(isinstance(key, str), "non-string checkpoint state key")
        if torch.is_tensor(value):
            copied[key] = value.detach().cpu().clone()
        else:
            copied[key] = copy.deepcopy(value)
    if hasattr(state, "_metadata"):
        copied._metadata = copy.deepcopy(state._metadata)
    return copied


def safe_npz_key(name, suffix):
    return name.replace(".", "__") + "__" + suffix


def write_seal(directory, members):
    manifest = directory / "SHA256SUMS"
    manifest.write_text("\n".join(
        "{}  {}".format(sha256(directory / name), name)
        for name in sorted(members)) + "\n", encoding="utf-8")
    seal = directory / "SHA256SUMS.seal.sha256"
    seal.write_text("{}  SHA256SUMS\n".format(sha256(manifest)),
                    encoding="utf-8")


def verify_seal(directory, expected_members):
    seal_path = directory / "SHA256SUMS"
    lines = seal_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == expected_members, "M509 seal member-count drift")
    names = []
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "M509 malformed SHA256SUMS line")
        expected, name = parts
        require(name not in names, "M509 duplicate sealed member: " + name)
        require("/" not in name and name not in {".", ".."},
                "M509 unsafe sealed member name")
        require(sha256(directory / name) == expected,
                "M509 sealed member rehash failed: " + name)
        names.append(name)
    seal_line = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(len(seal_line) == 2 and seal_line[1] == "SHA256SUMS" and
            sha256(seal_path) == seal_line[0], "M509 outer seal rehash failed")
    return names


def load_and_verify_contract(path, script_start, output):
    contract_path = path.resolve()
    require(contract_path.is_file(), "missing M509 reviewed contract")
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m509_h67_ep35_fc2_only_int8_ptq_export_contract_v2",
            "M509 contract schema drift")
    require(contract.get("status") ==
            "LOCKED_R2_STATIC_PREFLIGHT_REQUIRED_BEFORE_ONE_SHOT_EXPORT",
            "M509 contract status drift")
    require(contract["inputs"]["exporter"]["sha256"] == script_start,
            "M509 contract does not pin running exporter")
    expected_output = (HW / contract["output"]["canonical_directory"]).resolve()
    require(expected_output == output, "M509 non-canonical output directory")
    return contract_path, sha256(contract_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing M509 overwrite")
    require(output.parent.is_dir(), "M509 output parent must already exist")

    script_start = sha256(Path(__file__).resolve())
    contract_path, contract_sha = load_and_verify_contract(
        args.contract, script_start, output)
    disk_free = shutil.disk_usage(output.parent).free
    require(disk_free >= 2 * (1 << 30), "M509 requires at least 2 GiB free disk")
    observed = {}
    for name, (path, expected) in FROZEN.items():
        require(path.is_file(), "missing frozen M509 input: " + name)
        actual = sha256(path)
        require(actual == expected, "M509 input SHA drift: " + name)
        observed[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual}

    m51 = strict_json(FROZEN["m51_manifest"][0])
    require(m51.get("schema") == "m51_h67_ep35_binary_input_trace_manifest_v1"
            and m51.get("status") ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "M509 M51 identity drift")
    fc2_records = [record for record in m51["records"]
                   if record.get("operator") == "Linear"
                   and str(record.get("name", "")).endswith(".mlp.fc2")]
    fc2_names = sorted(set(record["name"] for record in fc2_records))
    require(len(fc2_records) == 120 and len(fc2_names) == 12,
            "M509 requires all twelve FC2 binary-input traces over S10")
    require(set(int(record["sample_id"]) for record in fc2_records) ==
            set(range(10)), "M509 requires exact sample_id population 0..9")
    record_pairs = [(record["name"], int(record["sample_id"]))
                    for record in fc2_records]
    require(len(set(record_pairs)) == 120,
            "M509 requires one FC2 record per module/sample pair")
    require(all(int(record["active_elements"]) <= int(record["input_elements"])
                for record in fc2_records), "M509 invalid M51 activity count")

    # The frozen source is an old full-model pickle.  Install the proven import
    # compatibility paths first, then normalize the result to a portable
    # model_state_dict-only deployment checkpoint.  This deliberately avoids
    # re-pickling executable model code into the candidate artifact.
    checkpoint_path = FROZEN["checkpoint"][0]
    install_pickle_import_paths()
    payload = torch.load(checkpoint_path, map_location="cpu",
                         weights_only=False)
    source_payload_type = type(payload).__module__ + "." + type(payload).__name__
    state_label, source_state, target_keys = find_state_dict(payload)
    state = canonical_state_copy(source_state)
    require(len(target_keys) == len(fc2_names), "M509 FC2 target-count drift")
    key_to_trace_name = {}
    for key in target_keys:
        matches = [name for name in fc2_names
                   if key.endswith(name + ".weight")]
        require(len(matches) == 1,
                "M509 checkpoint/M51 FC2 name mismatch: " + key)
        key_to_trace_name[key] = matches[0]
    require(set(key_to_trace_name.values()) == set(fc2_names),
            "M509 checkpoint/M51 FC2 population mismatch")

    module_identities = m51.get("module_identities")
    require(isinstance(module_identities, dict),
            "M509 M51 module identities must be an object")
    m160_rows = list(csv.DictReader(
        FROZEN["m160_parameters"][0].open("r", encoding="utf-8", newline="")))
    require(len(m160_rows) == 12, "M509 M160 FFN row population drift")
    m160_fc2_sumabs = {
        row["module"] + ".fc2": int(row["fc2_int8_sumabs_max"])
        for row in m160_rows
    }
    require(set(m160_fc2_sumabs) == set(fc2_names),
            "M509 M160/M51 FC2 module population mismatch")

    arrays = {}
    rows = []
    widths = {}
    for key in target_keys:
        source_weight = source_state[key]
        weight = state[key]
        require(weight.ndim == 2 and weight.is_floating_point(),
                "M509 FC2 weight must be floating 2-D: " + key)
        require(torch.equal(weight, source_weight.detach().cpu()),
                "M509 canonical state clone drift before quantization: " + key)
        module = key_to_trace_name[key]
        identity = module_identities.get(module)
        require(isinstance(identity, dict) and identity.get("operator") == "Linear"
                and identity.get("bias") is None,
                "M509 FC2 module identity/bias drift: " + module)
        weight_identity = identity.get("weight", {})
        source_numpy = source_weight.detach().cpu().contiguous().numpy()
        require(sys.byteorder == "little" and
                weight_identity.get("byte_order") == "little" and
                weight_identity.get("layout") == "C_ORDER_CONTIGUOUS" and
                weight_identity.get("dtype") == str(source_weight.dtype) and
                weight_identity.get("shape") == list(source_weight.shape) and
                int(weight_identity.get("content_bytes", -1)) ==
                int(source_numpy.nbytes) and
                hashlib.sha256(source_numpy.tobytes(order="C")).hexdigest() ==
                weight_identity.get("content_sha256"),
                "M509 source FC2 weight identity drift: " + module)
        module_records = [record for record in fc2_records
                          if record["name"] == module]
        require(len(module_records) == 10 and all(
            int(record["input_shape"][-1]) == int(source_weight.shape[1]) and
            int(record["output_shape"][-1]) == int(source_weight.shape[0])
            for record in module_records),
            "M509 FC2 trace shape differs from source weight: " + module)
        value = weight.detach().cpu().to(torch.float64)
        require(bool(torch.isfinite(value).all()),
                "M509 non-finite FC2 weight: " + key)
        row_max = value.abs().amax(dim=1)
        scale = torch.where(row_max == 0, torch.ones_like(row_max),
                            row_max / 127.0)
        qweight = torch.clamp(torch.round(value / scale[:, None]),
                              -127, 127).to(torch.int8)
        require(not bool((qweight == -128).any()),
                "M509 emitted reserved -128: " + key)
        dequant = qweight.to(torch.float64) * scale[:, None]
        maximum_error = float((value - dequant).abs().max().item())
        require(math.isfinite(maximum_error), "M509 non-finite dequant error")
        sumabs = qweight.to(torch.int64).abs().sum(dim=1)
        module_bound = int(sumabs.max().item())
        require(module_bound == m160_fc2_sumabs[module],
                "M509 per-module M160 sumabs drift: " + module)
        width = signed_bits_for_magnitude(module_bound)
        widths[width] = widths.get(width, 0) + 1

        # Preserve tensor dtype in a canonical CPU state dictionary.  The
        # output is the exact dequantized deployment candidate.
        state[key] = dequant.to(dtype=weight.dtype)
        arrays[safe_npz_key(module, "qweight_int8")] = qweight.numpy()
        arrays[safe_npz_key(module, "scale_float64")] = scale.numpy()
        arrays[safe_npz_key(module, "sumabs_int64")] = sumabs.numpy()
        rows.append({
            "state_key": key,
            "module": module,
            "out_features": int(value.shape[0]),
            "in_features": int(value.shape[1]),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "qweight_min": int(qweight.min().item()),
            "qweight_max": int(qweight.max().item()),
            "sumabs_max": module_bound,
            "binary_input_raw_signed_bits": width,
            "dequant_max_abs_weight_error": maximum_error,
        })
    require(widths == {15: 2, 16: 2, 17: 6, 18: 2},
            "M509 FC2 width census differs from frozen M160")

    target_set = set(target_keys)
    require(set(state) == set(source_state),
            "M509 canonical state key population drift")
    unchanged_tensors = 0
    for key in state:
        if key in target_set:
            continue
        before = source_state[key]
        after = state[key]
        if torch.is_tensor(before):
            require(torch.equal(after, before.detach().cpu()),
                    "M509 modified non-FC2 tensor: " + key)
            unchanged_tensors += 1
        else:
            require(after == before, "M509 modified non-tensor state: " + key)

    # Drop the executable full-model pickle before saving.  Only passive model
    # state is exported, which is accepted by the frozen H9 evaluator loader.
    output_payload = {"model_state_dict": state}
    del payload

    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.",
                                    dir=str(output.parent)))
    try:
        checkpoint_out = staging / "checkpoint_epoch35_fc2_only_int8_ptq.pth"
        torch.save(output_payload, checkpoint_out)
        npz_out = staging / "fc2_int8_codes_scales_sumabs.npz"
        np.savez_compressed(npz_out, **arrays)
        manifest = {
            "schema": "m509_h67_ep35_fc2_only_int8_ptq_export_v1",
            "status": "PASS_EXACT_EXPORT__ACCURACY_AND_HARDWARE_NOT_ADMITTED",
            "identity": {
                "exporter_start_end_sha256": script_start,
                "reviewed_contract": {
                    "path": str(contract_path.relative_to(HW)),
                    "sha256": contract_sha,
                    "schema": "m509_h67_ep35_fc2_only_int8_ptq_export_contract_v2",
                },
                "inputs": observed,
                "source_checkpoint_payload_type": source_payload_type,
                "checkpoint_state_container": state_label,
                "output_checkpoint_container": "model_state_dict",
            },
            "scope": {
                "checkpoint": "H67 ep35",
                "quantized_modules": 12,
                "quantized_parameter": "mlp.fc2.weight only",
                "input_domain": "binary, evidenced by all 120 M51 FC2 records",
                "quantizer": "per-output-row symmetric INT8 [-127,127]",
                "rounding": "torch.round ties-to-even",
                "checkpoint_payload": (
                    "canonical model_state_dict with twelve dequantized FP tensors; "
                    "no executable full-model pickle"
                ),
                "hardware_payload": "exact INT8 codes plus float64 row scales",
            },
            "binary_trace": {
                "records": len(fc2_records),
                "modules": len(fc2_names),
                "samples": len(set(int(record["sample_id"])
                                   for record in fc2_records)),
                "names": fc2_names,
            },
            "state_audit": {
                "source_state_entries": len(source_state),
                "target_weight_tensors": len(target_keys),
                "unchanged_tensor_entries_bit_exact": unchanged_tensors,
                "unchanged_non_tensor_entries": sum(
                    1 for key, value in source_state.items()
                    if key not in target_set and not torch.is_tensor(value)),
            },
            "width_distribution": {str(key): value
                                   for key, value in sorted(widths.items())},
            "modules": rows,
            "outputs": {
                "checkpoint_file": checkpoint_out.name,
                "checkpoint_sha256": sha256(checkpoint_out),
                "checkpoint_bytes": checkpoint_out.stat().st_size,
                "hardware_npz_file": npz_out.name,
                "hardware_npz_sha256": sha256(npz_out),
                "hardware_npz_bytes": npz_out.stat().st_size,
            },
            "required_next_gate": (
                "Run the frozen valid825 no-running protocol against the original "
                "and FC2-only PTQ checkpoint. Admit hardware deployment only if "
                "identity, per-frame aggregation and Delta-AEE gates pass."
            ),
            "claim_boundary": {
                "fc2_binary_input_identity": True,
                "fc2_int8_codes_and_scales": True,
                "analytic_raw_width_bound": True,
                "valid825_accuracy": False,
                "bn2_integer_bridge": False,
                "rtl": False,
                "cycles": False,
                "energy": False,
                "ppa": False,
                "system_speedup": False,
                "date_headline": False,
            },
        }
        manifest_path = staging / "m509_fc2_only_int8_ptq_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)
                                 + "\n", encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M509_H67_EP35_FC2_ONLY_INT8_PTQ_EXPORT\n",
            encoding="utf-8")
        write_seal(staging, [checkpoint_out.name, npz_out.name,
                             manifest_path.name, "RUN_COMPLETE.txt"])
        verify_seal(staging, 4)
        require(sha256(Path(__file__).resolve()) == script_start,
                "M509 exporter mutated during execution")
        require(not output.exists(), "M509 final output appeared during staging")
        os.replace(staging, output)
        final_names = verify_seal(output, 4)
        require(set(final_names) == {checkpoint_out.name, npz_out.name,
                                    manifest_path.name, "RUN_COMPLETE.txt"},
                "M509 final sealed population drift")
    except BaseException:
        # Preserve the unique staging directory for diagnosis; it is never an
        # admitted output because RUN_COMPLETE/seal/final rename are required.
        raise

    print(json.dumps({
        "status": "PASS_M509_H67_EP35_FC2_ONLY_INT8_PTQ_EXPORT",
        "output_dir": str(output),
        "modules": len(rows),
        "width_distribution": widths,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

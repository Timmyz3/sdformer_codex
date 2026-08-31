#!/usr/bin/env python3
"""Independently reopen and bit-audit the sealed M509 export.

This verifier does not trust the exporter's in-memory checks.  It reloads the
old frozen full-model checkpoint and the passive candidate checkpoint, proves
that every non-FC2 state entry is unchanged, and reconstructs every candidate
FC2 tensor from the sealed INT8/scale NPZ.  It emits no accuracy or hardware
claim.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = (
    HW / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
    "h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
)
SOURCE_SHA = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


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

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-standard JSON token: " + token)))


def install_pickle_import_paths():
    baseline = ROOT / "third_party" / "SDformerFlow"
    overlay = ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay"
    for path in (ROOT, baseline, overlay):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(overlay / "models")
    overlay_stsnn = str(overlay / "models/STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    import models.STSwinNet_SNN.atlif_ternary_psn  # noqa: F401
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
    register_shiftmax_pickle_compat()


def extract_source_state(payload):
    if hasattr(payload, "state_dict") and not isinstance(payload, dict):
        return payload.state_dict()
    require(isinstance(payload, dict), "unsupported M509 source payload")
    candidates = []
    for label in ("model_state_dict", "state_dict", "model"):
        value = payload.get(label)
        if hasattr(value, "state_dict") and not isinstance(value, dict):
            value = value.state_dict()
        if isinstance(value, dict):
            candidates.append(value)
    if any(torch.is_tensor(value) for value in payload.values()):
        candidates.append(payload)
    qualified = [state for state in candidates if
                 sum(key.endswith(".mlp.fc2.weight") for key in state) == 12]
    require(len(qualified) == 1, "ambiguous M509 source state dictionary")
    return qualified[0]


def verify_export_seal(directory):
    lines = (directory / "SHA256SUMS").read_text(
        encoding="utf-8").splitlines()
    require(len(lines) == 4, "M509 export seal population drift")
    members = {}
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "malformed M509 export SHA line")
        expected, name = parts
        require(name not in members and "/" not in name,
                "unsafe or duplicate M509 export member")
        require(sha256(directory / name) == expected,
                "M509 export member rehash failed: " + name)
        members[name] = expected
    outer = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(len(outer) == 2 and outer[1] == "SHA256SUMS" and
            sha256(directory / outer[1]) == outer[0],
            "M509 export outer seal failed")
    return members


def equal_non_tensor(left, right):
    try:
        return bool(left == right)
    except Exception:
        return False


def write_sealed_receipt(output, receipt):
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.",
                                    dir=str(output.parent)))
    try:
        report = staging / "m509_postexport_independent_verify.json"
        report.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M509_POSTEXPORT_INDEPENDENT_BIT_AUDIT\n", encoding="utf-8")
        members = [report.name, "RUN_COMPLETE.txt"]
        (staging / "SHA256SUMS").write_text("".join(
            "{}  {}\n".format(sha256(staging / name), name)
            for name in sorted(members)), encoding="utf-8")
        (staging / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(sha256(staging / "SHA256SUMS")),
            encoding="utf-8")
        require(not output.exists(), "M509 verify output appeared during staging")
        os.replace(staging, output)
        for line in (output / "SHA256SUMS").read_text(
                encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            require(sha256(output / name) == expected,
                    "M509 verify final member rehash failed")
        expected, name = (output / "SHA256SUMS.seal.sha256").read_text(
            encoding="utf-8").strip().split("  ", 1)
        require(name == "SHA256SUMS" and sha256(output / name) == expected,
                "M509 verify final outer seal failed")
    except BaseException:
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    export_dir = args.export_dir.resolve()
    contract_path = args.contract.resolve()
    output = args.output_dir.resolve()
    require(export_dir.is_dir() and contract_path.is_file(),
            "missing sealed M509 export or contract")
    require(output.parent.is_dir() and not output.exists(),
            "invalid M509 verifier output")
    script_start = sha256(Path(__file__).resolve())
    require(sha256(SOURCE) == SOURCE_SHA and sha256(DOCS359) == DOCS359_SHA,
            "M509 verifier frozen input drift")
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m509_h67_ep35_fc2_only_int8_ptq_export_contract_v2" and
            contract.get("status") ==
            "LOCKED_R2_STATIC_PREFLIGHT_REQUIRED_BEFORE_ONE_SHOT_EXPORT",
            "M509 verifier contract schema drift")
    require((HW / contract["output"]["canonical_directory"]).resolve() ==
            export_dir, "M509 verifier export path differs from contract")

    sealed = verify_export_seal(export_dir)
    checkpoint_name = "checkpoint_epoch35_fc2_only_int8_ptq.pth"
    npz_name = "fc2_int8_codes_scales_sumabs.npz"
    manifest_name = "m509_fc2_only_int8_ptq_manifest.json"
    require(set(sealed) == {checkpoint_name, npz_name, manifest_name,
                            "RUN_COMPLETE.txt"},
            "M509 export sealed names drift")
    manifest = strict_json(export_dir / manifest_name)
    require(manifest.get("schema") ==
            "m509_h67_ep35_fc2_only_int8_ptq_export_v1" and
            manifest.get("status") ==
            "PASS_EXACT_EXPORT__ACCURACY_AND_HARDWARE_NOT_ADMITTED",
            "M509 export manifest identity drift")
    require(manifest["identity"]["reviewed_contract"]["sha256"] ==
            sha256(contract_path), "M509 export contract SHA binding drift")
    require(manifest["outputs"]["checkpoint_sha256"] ==
            sha256(export_dir / checkpoint_name) and
            manifest["outputs"]["hardware_npz_sha256"] ==
            sha256(export_dir / npz_name), "M509 manifest output SHA drift")

    candidate_payload = torch.load(export_dir / checkpoint_name,
                                   map_location="cpu", weights_only=True)
    require(isinstance(candidate_payload, dict) and
            set(candidate_payload) == {"model_state_dict"} and
            isinstance(candidate_payload["model_state_dict"], dict),
            "M509 candidate is not passive model_state_dict-only")
    candidate = candidate_payload["model_state_dict"]
    install_pickle_import_paths()
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source = extract_source_state(source_payload)
    require(set(candidate) == set(source), "M509 candidate state keys drift")

    module_rows = manifest.get("modules")
    require(isinstance(module_rows, list) and len(module_rows) == 12,
            "M509 manifest target population drift")
    key_to_module = {row["state_key"]: row["module"] for row in module_rows}
    row_by_key = {row["state_key"]: row for row in module_rows}
    require(len(key_to_module) == 12 and all(
        key.endswith(".mlp.fc2.weight") for key in key_to_module),
        "M509 manifest target keys drift")
    arrays = np.load(export_dir / npz_name, allow_pickle=False)
    require(len(arrays.files) == 36, "M509 NPZ array population drift")

    target_equal_npz = 0
    target_equal_source_recomputed_quantizer = 0
    non_target_tensor_equal = 0
    non_target_other_equal = 0
    for key in candidate:
        if key in key_to_module:
            module = key_to_module[key]
            prefix = module.replace(".", "__")
            q_name = prefix + "__qweight_int8"
            scale_name = prefix + "__scale_float64"
            sumabs_name = prefix + "__sumabs_int64"
            require({q_name, scale_name, sumabs_name}.issubset(arrays.files),
                    "M509 missing NPZ arrays: " + module)
            qweight = arrays[q_name]
            scale = arrays[scale_name]
            sumabs = arrays[sumabs_name]
            require(qweight.dtype == np.int8 and scale.dtype == np.float64 and
                    sumabs.dtype == np.int64 and qweight.ndim == 2 and
                    scale.shape == (qweight.shape[0],) and
                    sumabs.shape == (qweight.shape[0],),
                    "M509 NPZ dtype/shape drift: " + module)
            require(np.array_equal(sumabs,
                                   np.abs(qweight.astype(np.int64)).sum(axis=1)),
                    "M509 NPZ sumabs drift: " + module)
            source_value = source[key].detach().cpu().to(torch.float64)
            require(source_value.ndim == 2 and
                    bool(torch.isfinite(source_value).all()),
                    "M509 source target is not finite 2-D: " + module)
            expected_row_max = source_value.abs().amax(dim=1)
            expected_scale_t = torch.where(
                expected_row_max == 0, torch.ones_like(expected_row_max),
                expected_row_max / 127.0)
            expected_qweight_t = torch.clamp(torch.round(
                source_value / expected_scale_t[:, None]), -127, 127).to(
                    torch.int8)
            expected_sumabs_t = expected_qweight_t.to(
                torch.int64).abs().sum(dim=1)
            expected_scale = expected_scale_t.numpy()
            expected_qweight = expected_qweight_t.numpy()
            expected_sumabs = expected_sumabs_t.numpy()
            require(np.array_equal(scale, expected_scale),
                    "M509 NPZ scale differs from frozen-source recompute: " +
                    module)
            require(np.array_equal(qweight, expected_qweight) and
                    not bool(np.any(qweight == -128)),
                    "M509 NPZ qweight differs from frozen-source recompute: " +
                    module)
            require(np.array_equal(sumabs, expected_sumabs),
                    "M509 NPZ sumabs differs from frozen-source recompute: " +
                    module)
            expected_dequant = (expected_qweight_t.to(torch.float64) *
                                expected_scale_t[:, None])
            reconstructed = expected_dequant.to(dtype=source[key].dtype)
            require(torch.equal(candidate[key], reconstructed),
                    "M509 candidate differs from frozen-source recompute: " +
                    module)
            expected_error = float(
                (source_value - expected_dequant).abs().max().item())
            row = row_by_key[key]
            require(row["module"] == module and
                    int(row["out_features"]) == int(source_value.shape[0]) and
                    int(row["in_features"]) == int(source_value.shape[1]) and
                    float(row["scale_min"]) == float(expected_scale_t.min()) and
                    float(row["scale_max"]) == float(expected_scale_t.max()) and
                    int(row["qweight_min"]) == int(expected_qweight_t.min()) and
                    int(row["qweight_max"]) == int(expected_qweight_t.max()) and
                    int(row["sumabs_max"]) == int(expected_sumabs_t.max()) and
                    float(row["dequant_max_abs_weight_error"]) == expected_error,
                    "M509 manifest module row differs from source recompute: " +
                    module)
            target_equal_npz += 1
            target_equal_source_recomputed_quantizer += 1
        elif torch.is_tensor(source[key]):
            require(torch.is_tensor(candidate[key]) and
                    torch.equal(candidate[key], source[key].detach().cpu()),
                    "M509 non-target tensor changed: " + key)
            non_target_tensor_equal += 1
        else:
            require(equal_non_tensor(candidate[key], source[key]),
                    "M509 non-target non-tensor changed: " + key)
            non_target_other_equal += 1
    require(target_equal_npz == 12, "M509 target audit count drift")
    require(target_equal_source_recomputed_quantizer == 12,
            "M509 source-recomputed quantizer audit count drift")
    require(set(arrays.files) == {
        module.replace(".", "__") + "__" + suffix
        for module in key_to_module.values()
        for suffix in ("qweight_int8", "scale_float64", "sumabs_int64")
    }, "M509 NPZ contains undeclared arrays")
    require(sha256(Path(__file__).resolve()) == script_start and
            sha256(SOURCE) == SOURCE_SHA and
            sha256(DOCS359) == DOCS359_SHA and
            sha256(contract_path) ==
            manifest["identity"]["reviewed_contract"]["sha256"],
            "M509 verifier/input mutated during execution")

    receipt = {
        "schema": "m509_h67_ep35_fc2_only_int8_ptq_postexport_verify_v1",
        "status": "PASS_M509_POSTEXPORT_INDEPENDENT_BIT_AUDIT__ACCURACY_AND_HARDWARE_NOT_ADMITTED",
        "identity": {
            "verifier_start_end_sha256": script_start,
            "contract_sha256": sha256(contract_path),
            "source_checkpoint_sha256": SOURCE_SHA,
            "export_seal_sha256": sha256(export_dir / "SHA256SUMS.seal.sha256"),
            "candidate_checkpoint_sha256": sha256(export_dir / checkpoint_name),
            "hardware_npz_sha256": sha256(export_dir / npz_name),
            "docs359_sha256": DOCS359_SHA,
        },
        "audit": {
            "state_entries": len(candidate),
            "fc2_targets_equal_npz": target_equal_npz,
            "fc2_targets_equal_frozen_source_recomputed_quantizer":
                target_equal_source_recomputed_quantizer,
            "non_target_tensor_entries_bit_exact": non_target_tensor_equal,
            "non_target_non_tensor_entries_equal": non_target_other_equal,
            "npz_arrays": len(arrays.files),
        },
        "claim_boundary": {
            "export_integrity": True,
            "valid825_accuracy": False,
            "bn2_integer_bridge": False,
            "rtl": False,
            "cycles": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "date_headline": False
        }
    }
    write_sealed_receipt(output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

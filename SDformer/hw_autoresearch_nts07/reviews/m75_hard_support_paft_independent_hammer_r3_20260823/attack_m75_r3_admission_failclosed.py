#!/usr/bin/env python3
"""Independent isolated admission/state negative tests for exact M75 r6."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

from torch import nn


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
OVERLAY = REPO / "neuron_experiments/H9_bipolar_self_attention/overlay"
PAFT = OVERLAY / "models/STSwinNet_SNN/pattern_paft.py"
REVOKED = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
TARGET_PAFT_SHA = (
    "d3eac645e5b4b2e1d9d2d5dcf9e535f936adb3be15abd86d86a4d6836120a066")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(PAFT) != TARGET_PAFT_SHA:
    raise AssertionError("r6 PAFT source drift")
sys.path.insert(0, str(OVERLAY))
from models.STSwinNet_SNN.pattern_paft import (  # noqa: E402
    _EXPECTED_CHECKPOINT_SHA256,
    _EXPECTED_OPERATORS,
    _EXPECTED_TRAIN_LIST_SHA256,
    _EXPECTED_VALID_LIST_SHA256,
    _REVOKED_CATALOG_SHA256,
    _STATE_ATTR,
    _load_catalog,
    install_pattern_paft,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Identity())
        self.conv2 = nn.Sequential(nn.Identity())


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = nn.Module()
        self.sttmultires_unet.resblocks = nn.ModuleList([Block(), Block()])


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8")


def reject(name, function, expected):
    try:
        function()
    except RuntimeError as error:
        message = str(error)
        if expected not in message:
            raise AssertionError(name + " wrong rejection: " + message)
        return {"name": name, "rejected": True, "message": message}
    raise AssertionError(name + " accepted")


def make_control(directory):
    catalog = json.loads(REVOKED.read_text(encoding="utf-8"))
    catalog["schema"] = (
        "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1")
    catalog["status"] = "SYNTHETIC_UNIT_CONTROL_NOT_TRAINING_EVIDENCE"
    catalog["split"].update({
        "role": "DSEC_TRAIN_ONLY_PAFT_CALIBRATION",
        "train_catalog_eligible": True,
        "test_or_validation_data_used": False,
    })
    trace_sha = "1" * 64
    catalog["identity"].update({
        "train_sequence_list_sha256": _EXPECTED_TRAIN_LIST_SHA256,
        "valid825_sequence_list_sha256": _EXPECTED_VALID_LIST_SHA256,
        "checkpoint_sha256": _EXPECTED_CHECKPOINT_SHA256,
        "train_trace_manifest_sha256": trace_sha,
    })
    catalog_path = directory / "synthetic_m77_catalog.json"
    write_json(catalog_path, catalog)
    contract = {
        "schema": "m77_pattern_paft_catalog_admission_contract_v1",
        "unit_test_only": False,
        "train_only_admitted": True,
        "catalog_sha256": sha256(catalog_path),
        "train_sequence_list_sha256": _EXPECTED_TRAIN_LIST_SHA256,
        "valid825_sequence_list_sha256": _EXPECTED_VALID_LIST_SHA256,
        "train_valid825_key_overlap": 0,
        "checkpoint_sha256": _EXPECTED_CHECKPOINT_SHA256,
        "operator_names": list(_EXPECTED_OPERATORS),
        "revoked_catalog_sha256": sorted(_REVOKED_CATALOG_SHA256),
        "train_trace_manifest_sha256": trace_sha,
    }
    contract_path = directory / "synthetic_m77_contract.json"
    write_json(contract_path, contract)
    cfg = {
        "catalog_sha256": sha256(catalog_path),
        "catalog_admission_contract": str(contract_path),
        "catalog_admission_contract_sha256": sha256(contract_path),
    }
    return catalog_path, contract_path, cfg


def contract_case(directory, catalog, contract, base, name, key, value,
                  expected):
    payload = json.loads(contract.read_text(encoding="utf-8"))
    if value is None:
        payload.pop(key, None)
    else:
        payload[key] = value
    path = directory / (name + ".json")
    write_json(path, payload)
    cfg = dict(base)
    cfg["catalog_admission_contract"] = str(path)
    cfg["catalog_admission_contract_sha256"] = sha256(path)
    return reject(name, lambda: _load_catalog(catalog, cfg), expected)


def catalog_split_case(directory, original, contract, base, name, key,
                       value, expected):
    payload = json.loads(json.dumps(original))
    payload["split"][key] = value
    path = directory / (name + ".json")
    write_json(path, payload)
    cfg = dict(base)
    cfg["catalog_sha256"] = sha256(path)
    # Rebind the contract catalog SHA so the requested semantic gate is tested.
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_payload["catalog_sha256"] = sha256(path)
    contract_path = directory / (name + "_contract.json")
    write_json(contract_path, contract_payload)
    cfg["catalog_admission_contract"] = str(contract_path)
    cfg["catalog_admission_contract_sha256"] = sha256(contract_path)
    return reject(name, lambda: _load_catalog(path, cfg), expected)


def main():
    results = [reject(
        "revoked_sha_with_legacy_config_override",
        lambda: _load_catalog(REVOKED, {
            "catalog_sha256": sha256(REVOKED),
            "unit_test_allow_unpinned_revoked_catalog": True,
        }), "permanently revoked")]
    with tempfile.TemporaryDirectory(prefix="m75_r3_attack_") as temporary:
        directory = Path(temporary)
        catalog, contract, base = make_control(directory)
        loaded = _load_catalog(catalog, base)
        if loaded.get("schema") != (
                "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1"):
            raise AssertionError("synthetic loader control failed")
        results.append(reject(
            "self_labeled_successor_without_contract",
            lambda: _load_catalog(catalog, {
                "catalog_sha256": sha256(catalog)}),
            "without external admission contract"))
        wrong_sha = dict(base)
        wrong_sha["catalog_admission_contract_sha256"] = "0" * 64
        results.append(reject(
            "wrong_contract_sha", lambda: _load_catalog(catalog, wrong_sha),
            "SHA absent or mismatched"))
        for args in (
                ("contract_catalog_binding_mismatch", "catalog_sha256",
                 "2" * 64, "contract/catalog SHA mismatch"),
                ("unit_test_only_true", "unit_test_only", True,
                 "unit-test-only admission contract"),
                ("unit_test_only_missing", "unit_test_only", None,
                 "unit-test-only admission contract"),
                ("train_only_not_admitted", "train_only_admitted", False,
                 "does not admit train-only use"),
                ("wrong_train_identity", "train_sequence_list_sha256",
                 "3" * 64, "train-list identity mismatch"),
                ("wrong_valid_identity", "valid825_sequence_list_sha256",
                 "4" * 64, "valid825 identity mismatch"),
                ("nonzero_overlap", "train_valid825_key_overlap", 1,
                 "does not prove zero overlap"),
                ("wrong_checkpoint_identity", "checkpoint_sha256", "5" * 64,
                 "checkpoint mismatch"),
                ("wrong_operators", "operator_names",
                 list(_EXPECTED_OPERATORS[:-1]), "operator mismatch"),
                ("revocation_omitted", "revoked_catalog_sha256", [],
                 "omits revoked catalog SHA"),
                ("trace_sha_missing", "train_trace_manifest_sha256", "short",
                 "lacks train-trace SHA")):
            results.append(contract_case(
                directory, catalog, contract, base, *args))
        original = json.loads(catalog.read_text(encoding="utf-8"))
        for args in (
                ("wrong_catalog_role", "role", "DSEC_VALID825",
                 "role receipt"),
                ("catalog_not_eligible", "train_catalog_eligible", False,
                 "not explicitly train eligible"),
                ("catalog_leakage_flag", "test_or_validation_data_used", True,
                 "validation/test leakage")):
            results.append(catalog_split_case(
                directory, original, contract, base, *args))

        model = Model()
        sentinel = {"forged_preinstalled_state": True}
        setattr(model, _STATE_ATTR, sentinel)
        hooks_before = len(model._forward_pre_hooks)
        results.append(reject(
            "preexisting_state",
            lambda: install_pattern_paft(model, {"enabled": True}, None),
            "preexisting or stale PAFT state"))
        if getattr(model, _STATE_ATTR) is not sentinel or len(
                model._forward_pre_hooks) != hooks_before:
            raise AssertionError("preexisting-state rejection mutated model")

        install_model = Model()
        install_cfg = dict(base)
        install_cfg.update({"enabled": True, "catalog": str(catalog)})
        results.append(reject(
            "missing_runtime_artifacts",
            lambda: install_pattern_paft(install_model, install_cfg, None),
            "runtime dataset/trace paths are incomplete"))
        if hasattr(install_model, _STATE_ATTR) or len(
                install_model._forward_pre_hooks) != 0:
            raise AssertionError("failed install mutated model")

    payload = {
        "schema": "m75_independent_hammer_r3_admission_attack_v1",
        "status": "PASS_R6_ADMISSION_AND_STATE_FAIL_CLOSED",
        "target_pattern_paft_sha256": TARGET_PAFT_SHA,
        "negative_attack_count": len(results),
        "all_rejected": all(row["rejected"] for row in results),
        "negative_attacks": results,
        "preexisting_state_preserved": True,
        "failed_install_left_model_unmodified": True,
        "synthetic_loader_control": {
            "accepted": True,
            "formal_training_evidence": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

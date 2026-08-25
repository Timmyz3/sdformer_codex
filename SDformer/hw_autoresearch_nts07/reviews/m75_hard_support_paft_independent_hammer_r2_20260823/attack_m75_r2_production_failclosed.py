#!/usr/bin/env python3
"""Isolated local negative tests against the exact reviewed PAFT source."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
OVERLAY = REPO / "neuron_experiments/H9_bipolar_self_attention/overlay"
PAFT = OVERLAY / "models/STSwinNet_SNN/pattern_paft.py"
REVOKED_CATALOG = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
TARGET_PAFT_SHA = (
    "bf15a2ea328a16d1d8c676de11a041fc3768c6717bcf73a21c1c6e3c2378087f")
sys.path.insert(0, str(OVERLAY))

from models.STSwinNet_SNN.pattern_paft import (  # noqa: E402
    _EXPECTED_CHECKPOINT_SHA256,
    _EXPECTED_OPERATORS,
    _EXPECTED_TRAIN_LIST_SHA256,
    _EXPECTED_VALID_LIST_SHA256,
    _REVOKED_CATALOG_SHA256,
    _STATE_ATTR,
    _cost_proxy,
    _hard_support_ste,
    _load_catalog,
    install_pattern_paft,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Identity())
        self.conv2 = nn.Sequential(nn.Identity())


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([Block(), Block()])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = Unet()


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8")


def expect_reject(name, function, expected_message=None):
    try:
        function()
    except RuntimeError as error:
        message = str(error)
        if expected_message is not None and expected_message not in message:
            raise AssertionError(name + " rejected for wrong reason: " + message)
        return {"name": name, "rejected": True, "message": message}
    raise AssertionError(name + " was accepted")


def make_successor(directory):
    catalog = json.loads(REVOKED_CATALOG.read_text(encoding="utf-8"))
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
    config = {
        "catalog_sha256": sha256(catalog_path),
        "catalog_admission_contract": str(contract_path),
        "catalog_admission_contract_sha256": sha256(contract_path),
    }
    return catalog_path, contract_path, config


def mutate_contract(directory, original, name, key, value):
    payload = json.loads(original.read_text(encoding="utf-8"))
    if value is None:
        payload.pop(key, None)
    else:
        payload[key] = value
    path = directory / (name + ".json")
    write_json(path, payload)
    return path


def with_contract(base, path):
    result = dict(base)
    result["catalog_admission_contract"] = str(path)
    result["catalog_admission_contract_sha256"] = sha256(path)
    return result


def production_oracle_equivalence():
    dtype_sweep = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        amplitude = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.7, 0.9, 1.0e-3, -0.2],
            dtype=dtype, requires_grad=True)
        observed = _hard_support_ste(amplitude)
        expected = amplitude.detach().ne(0).to(dtype=dtype)
        observed.sum().backward()
        dtype_sweep.append({
            "dtype": str(dtype),
            "exact_binary": bool(torch.equal(observed.detach(), expected)),
            "maximum_forward_error": float(
                (observed.detach() - expected).abs().max()),
            "identity_gradient": bool(torch.equal(
                amplitude.grad, torch.ones_like(amplitude.grad))),
        })

    patterns = torch.zeros((432, 16, 16), dtype=torch.float64)
    patterns[:, :, :4] = 1.0
    values = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        values[0, :4] = 0.2
        values[1, :3] = 0.9

    hard = _hard_support_ste(values)
    target_hard = values.detach().ne(0).to(values.dtype)
    binary_forward = bool(torch.equal(hard.detach(), target_hard))
    maximum_forward_error = float(
        (hard.detach() - target_hard).abs().max())
    hard.sum().backward(retain_graph=True)
    if not torch.equal(values.grad, torch.ones_like(values.grad)):
        raise AssertionError("production hard-support STE is not identity")
    values.grad.zero_()

    production_c, production_b, production_n = _cost_proxy(values, patterns, 31)
    # Force subtraction of identical tensors before adding the detached hard
    # value, avoiding the production expression's cancellation order.
    support = (values.detach().ne(0).to(values.dtype) +
               (values - values.detach()))
    grouped = support.reshape(2, 432, 16)
    distance = torch.abs(
        grouped.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    oracle_b = grouped.sum(dim=-1)
    oracle_c = torch.minimum(
        oracle_b, 1.0 + distance.min(dim=2).values)
    candidate_exact = bool(torch.equal(
        production_c.detach(), oracle_c.sum().detach()))
    baseline_exact = bool(torch.equal(
        production_b.detach(), oracle_b.sum().detach()))
    if production_n != 864:
        raise AssertionError("production proxy element count mismatch")
    production_c.backward()
    production_gradient = values.grad.detach().clone()

    oracle_values = values.detach().clone().requires_grad_(True)
    oracle_support = (
        oracle_values.detach().ne(0).to(oracle_values.dtype) +
        (oracle_values - oracle_values.detach()))
    oracle_grouped = oracle_support.reshape(2, 432, 16)
    oracle_distance = torch.abs(
        oracle_grouped.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    oracle_baseline = oracle_grouped.sum(dim=-1)
    oracle_candidate = torch.minimum(
        oracle_baseline, 1.0 + oracle_distance.min(dim=2).values).sum()
    oracle_candidate.backward()
    gradient_exact = bool(torch.equal(production_gradient, oracle_values.grad))
    if not bool(torch.isfinite(production_gradient).all().item()):
        raise AssertionError("production candidate gradient is nonfinite")
    if float(production_gradient.abs().sum()) <= 0.0:
        raise AssertionError("production candidate gradient is zero")
    return {
        "binary_forward": binary_forward,
        "maximum_forward_error_float64_case": maximum_forward_error,
        "dtype_sweep": dtype_sweep,
        "hard_support_gradient_is_identity": True,
        "candidate_vector_ops": float(production_c.detach()),
        "baseline_vector_ops": float(production_b.detach()),
        "partition_vectors": production_n,
        "candidate_gradient_l1": float(production_gradient.abs().sum()),
        "production_oracle_candidate_equal": candidate_exact,
        "production_oracle_baseline_equal": baseline_exact,
        "production_oracle_gradient_equal": gradient_exact,
        "maximum_candidate_delta": float(
            (production_c.detach() - oracle_c.sum().detach()).abs()),
        "maximum_baseline_delta": float(
            (production_b.detach() - oracle_b.sum().detach()).abs()),
        "maximum_gradient_delta": float(
            (production_gradient - oracle_values.grad).abs().max()),
    }


def main():
    live_sha = sha256(PAFT)
    if live_sha != TARGET_PAFT_SHA:
        raise AssertionError("target PAFT source drifted: " + live_sha)
    results = []
    revoked_sha = sha256(REVOKED_CATALOG)
    results.append(expect_reject(
        "revoked_sha_with_legacy_config_override",
        lambda: _load_catalog(REVOKED_CATALOG, {
            "catalog_sha256": revoked_sha,
            "unit_test_allow_unpinned_revoked_catalog": True,
        }), "permanently revoked"))

    with tempfile.TemporaryDirectory(prefix="m75_r2_attack_") as temporary:
        directory = Path(temporary)
        catalog, contract, base = make_successor(directory)
        loaded = _load_catalog(catalog, base)
        if loaded.get("schema") != (
                "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1"):
            raise AssertionError("synthetic loader control failed")

        no_contract = {"catalog_sha256": sha256(catalog)}
        results.append(expect_reject(
            "self_labeled_successor_without_contract",
            lambda: _load_catalog(catalog, no_contract),
            "without external admission contract"))
        wrong_sha = dict(base)
        wrong_sha["catalog_admission_contract_sha256"] = "0" * 64
        results.append(expect_reject(
            "wrong_contract_sha", lambda: _load_catalog(catalog, wrong_sha),
            "SHA absent or mismatched"))

        attacks = [
            ("contract_catalog_binding_mismatch", "catalog_sha256", "2" * 64,
             "contract/catalog SHA mismatch"),
            ("unit_test_only_true", "unit_test_only", True,
             "unit-test-only admission contract"),
            ("unit_test_only_missing", "unit_test_only", None,
             "unit-test-only admission contract"),
            ("train_only_not_admitted", "train_only_admitted", False,
             "does not admit train-only use"),
            ("wrong_train_identity", "train_sequence_list_sha256", "3" * 64,
             "train-list identity mismatch"),
            ("wrong_valid_identity", "valid825_sequence_list_sha256", "4" * 64,
             "valid825 identity mismatch"),
            ("nonzero_overlap", "train_valid825_key_overlap", 1,
             "does not prove zero overlap"),
            ("wrong_checkpoint_identity", "checkpoint_sha256", "5" * 64,
             "checkpoint mismatch"),
            ("wrong_operators", "operator_names", list(_EXPECTED_OPERATORS[:-1]),
             "operator mismatch"),
            ("revocation_omitted", "revoked_catalog_sha256", [],
             "omits revoked catalog SHA"),
            ("trace_sha_missing", "train_trace_manifest_sha256", "short",
             "lacks train-trace SHA"),
        ]
        for name, key, value, message in attacks:
            bad = mutate_contract(directory, contract, name, key, value)
            cfg = with_contract(base, bad)
            results.append(expect_reject(
                name, lambda p=bad, c=cfg: _load_catalog(catalog, c), message))

        # Catalog-side role and eligibility cannot be self-certified away.
        original_payload = json.loads(catalog.read_text(encoding="utf-8"))
        for name, key, value, message in (
                ("wrong_catalog_role", "role", "DSEC_VALID825", "role receipt"),
                ("catalog_not_eligible", "train_catalog_eligible", False,
                 "not explicitly train eligible"),
                ("catalog_leakage_flag", "test_or_validation_data_used", True,
                 "validation/test leakage")):
            changed = json.loads(json.dumps(original_payload))
            changed["split"][key] = value
            path = directory / (name + ".json")
            write_json(path, changed)
            cfg = dict(base)
            cfg["catalog_sha256"] = sha256(path)
            results.append(expect_reject(
                name, lambda p=path, c=cfg: _load_catalog(p, c), message))

        # Preexisting state is rejected before catalog/path processing, with no
        # replacement or hook mutation.
        model = Model()
        sentinel = {"forged_preinstalled_state": True}
        setattr(model, _STATE_ATTR, sentinel)
        hooks_before = len(model._forward_pre_hooks)
        results.append(expect_reject(
            "preexisting_state",
            lambda: install_pattern_paft(model, {"enabled": True}, None),
            "preexisting or stale PAFT state"))
        if getattr(model, _STATE_ATTR) is not sentinel:
            raise AssertionError("preexisting state was replaced")
        if len(model._forward_pre_hooks) != hooks_before:
            raise AssertionError("preexisting attack installed a model hook")

        # A loader-positive synthetic control still cannot reach model mutation
        # without independently pinned runtime data/trace/checkpoint artifacts.
        install_model = Model()
        install_cfg = dict(base)
        install_cfg.update({"enabled": True, "catalog": str(catalog)})
        results.append(expect_reject(
            "missing_runtime_artifacts",
            lambda: install_pattern_paft(install_model, install_cfg, None),
            "runtime dataset/trace paths are incomplete"))
        if hasattr(install_model, _STATE_ATTR):
            raise AssertionError("failed install left PAFT state")
        if len(install_model._forward_pre_hooks) != 0:
            raise AssertionError("failed install left a model hook")

    equivalence = production_oracle_equivalence()
    exact_support_pass = (
        equivalence["binary_forward"] and
        all(row["exact_binary"] for row in equivalence["dtype_sweep"]) and
        equivalence["production_oracle_candidate_equal"] and
        equivalence["production_oracle_baseline_equal"] and
        equivalence["production_oracle_gradient_equal"])
    payload = {
        "schema": "m75_independent_hammer_r2_attack_v1",
        "status": ("PASS_M75_R2_REQUIRED_NEGATIVE_ATTACKS_BUT_"
                   "FAIL_EXACT_HARD_SUPPORT_FORWARD"),
        "target_pattern_paft_sha256": TARGET_PAFT_SHA,
        "revoked_catalog_sha256": revoked_sha,
        "synthetic_loader_control": {
            "accepted": True,
            "unit_only_not_formal_training_evidence": True,
        },
        "production_oracle_equivalence": equivalence,
        "negative_attacks": results,
        "negative_attack_count": len(results),
        "all_rejected": all(row["rejected"] for row in results),
        "failed_install_left_model_unmodified": True,
        "exact_hard_support_contract_pass": exact_support_pass,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not exact_support_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

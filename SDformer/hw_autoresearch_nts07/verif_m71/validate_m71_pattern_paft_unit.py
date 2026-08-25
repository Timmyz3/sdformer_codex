#!/usr/bin/env python3
"""Directed CPU validation for the M71 PAFT hook and hardware cost proxy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay"
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
    _sample_conv3x3_vectors,
    install_pattern_paft,
    pattern_paft_summary,
    regularize_pattern_paft,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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

    def forward(self, value):
        return self.conv2(self.conv1(value))


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([Block(), Block()])

    def forward(self, value):
        for block in self.resblocks:
            value = block(value)
        return value


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sttmultires_unet = Unet()

    def forward(self, value):
        return self.sttmultires_unet(value)


def directed_cost_checks():
    patterns = torch.ones((432, 16, 16), dtype=torch.float32)
    patterns[:, :, :] = 1.0
    patterns[:, 0, :] = 0.0
    patterns[:, 0, :4] = 1.0
    vectors = torch.zeros((4, 6912), dtype=torch.float32)
    vectors[1, :4] = 1.0              # exact pattern: cost one
    vectors[2, :3] = 1.0              # one signed correction: cost two
    vectors[3, 8] = 1.0               # zero fallback: cost one
    candidate, baseline, elements = _cost_proxy(vectors, patterns, 37)
    # Remaining 431 partitions are zero and contribute no work.
    require(float(baseline.item()) == 8.0, "M71 baseline popcount mismatch")
    require(float(candidate.item()) == 4.0, "M71 candidate cost mismatch")
    require(elements == 4 * 432, "M71 proxy population mismatch")
    return {
        "baseline_vector_ops": float(baseline.item()),
        "candidate_vector_ops": float(candidate.item()),
        "directed_speedup": float(baseline.item() / candidate.item()),
        "partition_vectors": elements,
    }


def support_amplitude_invariance_check():
    patterns = torch.zeros((432, 16, 16), dtype=torch.float32)
    patterns[:, :, :4] = 1.0
    low = torch.zeros((2, 6912), dtype=torch.float32, requires_grad=True)
    high = torch.zeros((2, 6912), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        low[:, :4] = 0.2
        high[:, :4] = 0.9
    low_candidate, low_baseline, low_elements = _cost_proxy(low, patterns, 31)
    high_candidate, high_baseline, high_elements = _cost_proxy(high, patterns, 31)
    require(float(low_candidate.detach().item()) ==
            float(high_candidate.detach().item()),
            "M71 support proxy changed with nonzero amplitude")
    require(float(low_baseline.detach().item()) ==
            float(high_baseline.detach().item()) == 8.0,
            "M71 support popcount changed with nonzero amplitude")
    require(low_elements == high_elements == 2 * 432,
            "M71 amplitude-invariance population mismatch")
    (low_candidate + low_baseline).backward()
    require(low.grad is not None and torch.isfinite(low.grad).all(),
            "M71 hard-support STE gradient absent/nonfinite")
    hard_low = _hard_support_ste(low.detach())
    require(set(hard_low.unique().tolist()) <= {0.0, 1.0},
            "M71 hard-support forward is not binary")
    return {
        "low_nonzero_amplitude": 0.2,
        "high_nonzero_amplitude": 0.9,
        "candidate_vector_ops_both": float(low_candidate.detach().item()),
        "baseline_vector_ops_both": float(low_baseline.detach().item()),
        "low_gradient_l1": float(low.grad.abs().sum().item()),
        "forward_support_binary": True,
    }


def support_multidtype_exactness_check():
    rows = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        source = torch.tensor(
            [-0.9, -0.3, -0.1, 0.0, 0.1, 0.3, 0.9],
            dtype=dtype, requires_grad=True)
        support = _hard_support_ste(source)
        expected = (source.detach() != 0).to(dtype=dtype)
        require(torch.equal(support.detach(), expected),
                "M75 hard-support forward is not exact binary for {}".format(dtype))
        support.sum().backward()
        require(source.grad is not None and
                torch.equal(source.grad, torch.ones_like(source)),
                "M75 hard-support STE gradient is not exact one for {}".format(dtype))
        rows.append({
            "dtype": str(dtype),
            "forward_bit_exact_binary": True,
            "gradient_bit_exact_one": True,
            "amplitudes": [float(value) for value in source.detach().to(torch.float64)],
        })
    return rows


def build_unit_successor_catalog(revoked_catalog, directory):
    catalog = json.loads(revoked_catalog.read_text(encoding="utf-8"))
    catalog["schema"] = "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1"
    catalog["status"] = "UNIT_ONLY_SYNTHETIC_SUCCESSOR_NOT_TRAINING_EVIDENCE"
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
    catalog_path = directory / "unit_m77_catalog.json"
    catalog_path.write_text(json.dumps(catalog, sort_keys=True) + "\n",
                            encoding="utf-8")
    contract = {
        "schema": "m77_pattern_paft_catalog_admission_contract_v1",
        "train_only_admitted": True,
        "catalog_sha256": sha256(catalog_path),
        "train_sequence_list_sha256": _EXPECTED_TRAIN_LIST_SHA256,
        "valid825_sequence_list_sha256": _EXPECTED_VALID_LIST_SHA256,
        "train_valid825_key_overlap": 0,
        "checkpoint_sha256": _EXPECTED_CHECKPOINT_SHA256,
        "operator_names": list(_EXPECTED_OPERATORS),
        "revoked_catalog_sha256": sorted(_REVOKED_CATALOG_SHA256),
        "train_trace_manifest_sha256": trace_sha,
        "unit_test_only": False,
    }
    contract_path = directory / "unit_m77_admission.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n",
                             encoding="utf-8")
    return catalog_path, contract_path


def catalog_pin_attack_checks(revoked_catalog, unit_catalog, unit_contract,
                              directory):
    attacks = {}
    bad_contract = json.loads(unit_contract.read_text(encoding="utf-8"))
    bad_contract["catalog_sha256"] = "2" * 64
    bad_contract_path = directory / "bad_catalog_binding_contract.json"
    bad_contract_path.write_text(json.dumps(bad_contract, sort_keys=True) + "\n",
                                 encoding="utf-8")
    unit_only_contract = json.loads(unit_contract.read_text(encoding="utf-8"))
    unit_only_contract["unit_test_only"] = True
    unit_only_contract_path = directory / "unit_test_only_contract.json"
    unit_only_contract_path.write_text(
        json.dumps(unit_only_contract, sort_keys=True) + "\n",
        encoding="utf-8")
    cases = [
        ("revoked_sha_even_with_old_config_override", revoked_catalog, {
            "catalog_sha256": sha256(revoked_catalog),
            "unit_test_allow_unpinned_revoked_catalog": True,
        }),
        ("self_labeled_successor_without_contract", unit_catalog, {
            "catalog_sha256": sha256(unit_catalog),
        }),
        ("successor_wrong_contract_sha", unit_catalog, {
            "catalog_sha256": sha256(unit_catalog),
            "catalog_admission_contract": str(unit_contract),
            "catalog_admission_contract_sha256": "0" * 64,
        }),
        ("contract_catalog_binding_mismatch", unit_catalog, {
            "catalog_sha256": sha256(unit_catalog),
            "catalog_admission_contract": str(bad_contract_path),
            "catalog_admission_contract_sha256": sha256(bad_contract_path),
        }),
        ("unit_test_only_contract_rejected_by_production_loader", unit_catalog, {
            "catalog_sha256": sha256(unit_catalog),
            "catalog_admission_contract": str(unit_only_contract_path),
            "catalog_admission_contract_sha256": sha256(unit_only_contract_path),
        }),
    ]
    for name, catalog, cfg in cases:
        try:
            _load_catalog(catalog.resolve(), cfg)
        except RuntimeError as error:
            attacks[name] = {
                "rejected": True,
                "message": str(error),
            }
        else:
            attacks[name] = {"rejected": False, "message": None}
    require(all(item["rejected"] for item in attacks.values()),
            "M75 one or more catalog pin attacks were accepted")
    return attacks


def preexisting_state_attack_check():
    model = Model()
    setattr(model, _STATE_ATTR, {
        "operator_names": list(_EXPECTED_OPERATORS),
        "forged_preinstalled_state": True,
    })
    try:
        install_pattern_paft(model, {"enabled": True}, None)
    except RuntimeError as error:
        return {"rejected": True, "message": str(error)}
    raise RuntimeError("M75 forged preexisting PAFT state bypass was accepted")


def layout_check():
    value = torch.zeros((1, 1, 768, 1, 1), dtype=torch.float32)
    value[0, 0, 1, 0, 0] = 1.0
    vector = _sample_conv3x3_vectors(value, 1)
    nonzero = torch.nonzero(vector[0], as_tuple=False).reshape(-1).tolist()
    require(nonzero == [1 * 9 + 4], "M71 I_KY_KX layout mismatch")
    return nonzero[0]


def cost_proxy_gradient_check():
    torch.manual_seed(71)
    patterns = torch.zeros((432, 16, 16), dtype=torch.float32)
    patterns[:, :, :4] = 1.0
    logits = torch.randn((4, 6912), dtype=torch.float32, requires_grad=True)
    activation = torch.sigmoid(logits)
    candidate, baseline, elements = _cost_proxy(activation, patterns, 27)
    penalty = candidate / float(elements)
    penalty.backward()
    require(logits.grad is not None and torch.isfinite(logits.grad).all(),
            "M71 PAFT gradient is absent or nonfinite")
    require(float(logits.grad.abs().sum().item()) > 0.0,
            "M71 PAFT gradient is identically zero")
    return {
        "penalty": float(penalty.detach().item()),
        "gradient_l1": float(logits.grad.abs().sum().item()),
        "candidate_vector_ops": float(candidate.detach().item()),
        "baseline_vector_ops": float(baseline.detach().item()),
        "partition_vectors": elements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.catalog.is_file(), "M71 catalog missing")
    require(not args.output.exists(), "refusing M71 unit receipt overwrite")
    with tempfile.TemporaryDirectory(prefix="m75_unit_") as temporary:
        directory = Path(temporary)
        unit_catalog, unit_contract = build_unit_successor_catalog(
            args.catalog.resolve(), directory)
        success_cfg = {
            "catalog_sha256": sha256(unit_catalog),
            "catalog_admission_contract": str(unit_contract),
            "catalog_admission_contract_sha256": sha256(unit_contract),
        }
        loaded = _load_catalog(unit_catalog, success_cfg)
        payload = {
            "schema": "m75_pattern_paft_hard_support_ste_directed_unit_receipt_v3",
            "status": "PASS_M75_MULTI_DTYPE_EXACT_HARD_SUPPORT_AND_EXTERNAL_ADMISSION_FAIL_CLOSED",
            "identity": {
                "validator_sha256": sha256(Path(__file__).resolve()),
                "pattern_paft_sha256": sha256(
                    OVERLAY / "models/STSwinNet_SNN/pattern_paft.py"),
                "revoked_catalog_sha256": sha256(args.catalog),
            },
            "directed_cost": directed_cost_checks(),
            "support_amplitude_invariance": support_amplitude_invariance_check(),
            "support_multidtype_exactness": support_multidtype_exactness_check(),
            "cost_proxy_gradient": cost_proxy_gradient_check(),
            "successor_loader_control": {
                "accepted": True,
                "schema": loaded["schema"],
                "synthetic_unit_only": True,
            },
            "catalog_pin_attacks": catalog_pin_attack_checks(
                args.catalog, unit_catalog, unit_contract, directory),
            "preexisting_state_attack": preexisting_state_attack_check(),
            "i_ky_kx_onehot_index": layout_check(),
            "claim_boundary": {
                "accepted": (
                    "directed hard-support proxy arithmetic/gradient, layout, "
                    "revoked-SHA denylist and external admission-contract loader"),
                "hook_plumbing_retained_from_m71_r1_only": True,
                "formal_training_launch_admitted": False,
                "accuracy_admitted": False,
                "heldout_speedup_admitted": False,
                "cycle_speedup_admitted": False,
                "rtl_or_ppa_admitted": False,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M75 hard-support external-contract attacks={} gradient_l1={:.9g}".format(
        len(payload["catalog_pin_attacks"]) + 1,
        payload["cost_proxy_gradient"]["gradient_l1"]))


if __name__ == "__main__":
    main()

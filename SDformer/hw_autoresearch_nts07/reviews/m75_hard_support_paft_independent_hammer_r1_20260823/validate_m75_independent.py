#!/usr/bin/env python3
"""Independent M75 arithmetic, receipt, and disabled-config validation.

This script deliberately does not import pattern_paft.py or the production M71
validator.  The support/STE and cost equations are independently expressed from
the hardware contract.
"""

from __future__ import print_function

import hashlib
import json
from pathlib import Path

import torch
import yaml


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
PAFT_SOURCE = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/pattern_paft.py")
PRODUCTION_VALIDATOR = HW / "verif_m71/validate_m71_pattern_paft_unit.py"
CATALOG = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
RECEIPT = HW / (
    "results/m75_pattern_paft_hard_support_ste_unit_dev_r3_20260823/"
    "m75_pattern_paft_hard_support_ste_unit_receipt.json")
REVOCATION = HW / "contracts/m71_valid825_catalog_revocation_r1_20260823.json"
CONFIG_DIR = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated")
TARGET_PAFT_SHA256 = (
    "22292b265292b4d3c00cdeb1addd3020c7b2a417adc855aa043d1394735d3bf1")
TARGET_VALIDATOR_SHA256 = (
    "cb3eac62663fc3618e5b4019686fa0cf121bfad942992cd84bc12ffa7e79c4ba")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def hard_support_ste_independent(values):
    require(bool(torch.isfinite(values).all().item()), "nonfinite input")
    hard = values.detach().ne(0).to(dtype=values.dtype)
    # Exact support in forward; identity derivative in backward.
    return values + (hard - values).detach()


def cost_independent(values, patterns):
    support = hard_support_ste_independent(values).reshape(-1, 432, 16)
    distance = torch.abs(
        support.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    candidate = torch.minimum(
        support.sum(dim=-1), 1.0 + distance.min(dim=2).values)
    return candidate.sum(), support.sum(), int(candidate.numel()), support


def arithmetic_and_gradient():
    patterns = torch.zeros((432, 16, 16), dtype=torch.float64)
    patterns[:, :, :4] = 1.0
    low = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    high = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        low[:, :4] = 0.2
        high[:, :4] = 0.9
    low_candidate, low_baseline, low_elements, low_support = cost_independent(
        low, patterns)
    high_candidate, high_baseline, high_elements, high_support = cost_independent(
        high, patterns)
    require(torch.equal(low_support, high_support), "support differs by amplitude")
    require(set(low_support.unique().tolist()) == {0.0, 1.0},
            "forward support is not exactly binary")
    require(float(low_candidate.detach()) == float(high_candidate.detach()) == 2.0,
            "exact-pattern candidate cost changed by amplitude")
    require(float(low_baseline.detach()) == float(high_baseline.detach()) == 8.0,
            "baseline support cost changed by amplitude")
    require(low_elements == high_elements == 864, "population mismatch")
    (low_candidate + low_baseline).backward()
    (high_candidate + high_baseline).backward()
    for name, gradient in (("low", low.grad), ("high", high.grad)):
        require(gradient is not None, name + " gradient absent")
        require(bool(torch.isfinite(gradient).all().item()),
                name + " gradient nonfinite")
        require(float(gradient.abs().sum().item()) > 0.0,
                name + " gradient identically zero")
    require(torch.equal(low.grad, high.grad),
            "STE gradient changed with nonzero amplitude")

    # Candidate-only gradient: use a near pattern so Hamming has a live
    # correction derivative rather than relying on the baseline term.
    near_low = torch.zeros((1, 6912), dtype=torch.float64, requires_grad=True)
    near_high = torch.zeros((1, 6912), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        near_low[:, :3] = 0.2
        near_high[:, :3] = 0.9
    near_low_candidate, _, _, _ = cost_independent(near_low, patterns)
    near_high_candidate, _, _, _ = cost_independent(near_high, patterns)
    require(float(near_low_candidate.detach()) ==
            float(near_high_candidate.detach()) == 2.0,
            "near-pattern candidate cost changed by amplitude")
    near_low_candidate.backward()
    near_high_candidate.backward()
    for name, gradient in (("near_low", near_low.grad),
                           ("near_high", near_high.grad)):
        require(gradient is not None and
                bool(torch.isfinite(gradient).all().item()),
                name + " candidate-only gradient absent/nonfinite")
        require(float(gradient.abs().sum().item()) > 0.0,
                name + " candidate-only gradient zero")
    require(torch.equal(near_low.grad, near_high.grad),
            "candidate-only gradient changed with amplitude")
    return {
        "amplitudes": [0.2, 0.9],
        "exact_candidate_both": 2.0,
        "exact_baseline_both": 8.0,
        "support_equal": True,
        "support_binary": True,
        "low_total_gradient_l1": float(low.grad.abs().sum().item()),
        "high_total_gradient_l1": float(high.grad.abs().sum().item()),
        "near_candidate_both": 2.0,
        "near_low_candidate_gradient_l1": float(
            near_low.grad.abs().sum().item()),
        "near_high_candidate_gradient_l1": float(
            near_high.grad.abs().sum().item()),
        "all_gradients_finite_nonzero": True,
    }


def receipt_audit():
    payload = strict_json(RECEIPT)
    require(payload["schema"] ==
            "m75_pattern_paft_hard_support_ste_directed_unit_receipt_v1",
            "M75 receipt schema mismatch")
    require(payload["identity"]["validator_sha256"] ==
            TARGET_VALIDATOR_SHA256, "receipt target-validator SHA mismatch")
    require(payload["identity"]["catalog_sha256"] == sha256(CATALOG),
            "receipt catalog SHA drift")
    invariance = payload["support_amplitude_invariance"]
    require(float(invariance["low_nonzero_amplitude"]) == 0.2 and
            float(invariance["high_nonzero_amplitude"]) == 0.9,
            "receipt amplitude identities mismatch")
    require(float(invariance["candidate_vector_ops_both"]) == 2.0 and
            float(invariance["baseline_vector_ops_both"]) == 8.0,
            "receipt amplitude-invariance cost mismatch")
    require(float(invariance["low_gradient_l1"]) > 0.0,
            "receipt direct gradient is zero")
    hook = payload["hook_gradient"]
    require(float(hook["gradient_l1"]) > 0.0 and
            float(hook["penalty"]) > 0.0,
            "receipt hook gradient/penalty is zero")
    attacks = payload["catalog_pin_attacks"]
    require(set(attacks) == {"missing_catalog_sha", "wrong_catalog_sha",
                             "revoked_catalog_role"},
            "receipt attack extent mismatch")
    require(all(row["rejected"] is True for row in attacks.values()),
            "one of the required receipt attacks was accepted")
    boundary = payload["claim_boundary"]
    for key in ("accuracy_admitted", "heldout_speedup_admitted",
                "cycle_speedup_admitted", "rtl_or_ppa_admitted"):
        require(boundary[key] is False, "receipt boundary overclaims " + key)
    return {
        "receipt_sha256": sha256(RECEIPT),
        "receipt_validator_sha256": TARGET_VALIDATOR_SHA256,
        "live_validator_sha256": sha256(PRODUCTION_VALIDATOR),
        "target_validator_still_live": (
            sha256(PRODUCTION_VALIDATOR) == TARGET_VALIDATOR_SHA256),
        "catalog_sha256": sha256(CATALOG),
        "required_attacks_rejected": True,
        "hook_gradient_l1": float(hook["gradient_l1"]),
        "implementation_sha_present_in_receipt": (
            "pattern_paft_sha256" in payload["identity"]),
    }


def config_audit():
    configs = []
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "pattern_paft:" not in text:
            continue
        payload = yaml.safe_load(text)
        cfg = payload.get("pattern_paft", {})
        configs.append({
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "enabled": cfg.get("enabled"),
            "blocked_reason": cfg.get("blocked_reason"),
            "catalog": cfg.get("catalog"),
            "runtime_catalog_split": payload.get("runtime", {}).get(
                "paft_catalog_split"),
            "runtime_heldout_split": payload.get("runtime", {}).get(
                "paft_heldout_split"),
        })
    require(len(configs) == 2, "generated PAFT config population drift")
    for row in configs:
        require(row["enabled"] is False, "a generated PAFT config is enabled")
        require(row["blocked_reason"] ==
                "M71_VALID825_CATALOG_REVOKED_USE_TRAIN_ONLY_SUCCESSOR",
                "generated PAFT config lacks revocation block")
        require(row["runtime_catalog_split"] ==
                "REVOKED_M71_VALID825_INTERNAL_SAMPLES_0_TO_4",
                "generated config catalog split is not revoked")
        require(row["runtime_heldout_split"] ==
                "REVOKED_NOT_AN_INDEPENDENT_HELDOUT",
                "generated config heldout split is not revoked")
    revocation = strict_json(REVOCATION)
    require(revocation["status"] ==
            "REVOKED_FOR_PAFT_TRAINING_VALID825_DATA_LEAKAGE",
            "M71 revocation status drift")
    require(revocation["forbidden_uses"]["paft_training"] is True,
            "M71 PAFT training is no longer forbidden")
    require(revocation["admission"]["m71_catalog_train_eligible"] is False,
            "M71 catalog regained training eligibility")
    return {
        "configs": configs,
        "all_generated_paft_configs_disabled": True,
        "revocation_sha256": sha256(REVOCATION),
        "m71_train_eligible": False,
    }


def static_source_audit():
    source = PAFT_SOURCE.read_text(encoding="utf-8")
    live_sha = sha256(PAFT_SOURCE)
    return {
        "target_pattern_paft_sha256": TARGET_PAFT_SHA256,
        "live_pattern_paft_sha256": live_sha,
        "target_pattern_paft_still_live": live_sha == TARGET_PAFT_SHA256,
        "config_controlled_revoked_override_present": (
            "unit_test_allow_unpinned_revoked_catalog" in source),
        "known_revoked_catalog_sha_denylist_present": (
            sha256(CATALOG) in source),
    }


def main():
    arithmetic = arithmetic_and_gradient()
    receipt = receipt_audit()
    configs = config_audit()
    source = static_source_audit()
    print(json.dumps({
        "arithmetic": arithmetic,
        "receipt": receipt,
        "configs": configs,
        "source": source,
    }, indent=2, sort_keys=True))
    print("PASS_M75_INDEPENDENT_ARITHMETIC_GRADIENT_REQUIRED_ATTACKS_AND_DISABLED_CONFIG")


if __name__ == "__main__":
    main()

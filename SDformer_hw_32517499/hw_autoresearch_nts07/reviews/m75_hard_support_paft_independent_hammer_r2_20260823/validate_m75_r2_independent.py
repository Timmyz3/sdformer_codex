#!/usr/bin/env python3
"""Independent arithmetic, identity, receipt, and disabled-config checks.

This oracle deliberately does not import the PAFT implementation or its
production validator.  The hard-support STE and hardware-cost equations below
are expressed independently from the reviewed contract.
"""

import hashlib
import json
from pathlib import Path

import torch
import yaml


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
PAFT = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/pattern_paft.py")
VALIDATOR = HW / "verif_m71/validate_m71_pattern_paft_unit.py"
RECEIPT = HW / (
    "results/m75_pattern_paft_hard_support_ste_unit_dev_r5_20260823/"
    "m75_pattern_paft_hard_support_ste_unit_receipt.json")
REVOKED_CATALOG = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
REVOCATION = HW / "contracts/m71_valid825_catalog_revocation_r1_20260823.json"
CONFIG_DIR = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated")

TARGET_PAFT_SHA = (
    "bf15a2ea328a16d1d8c676de11a041fc3768c6717bcf73a21c1c6e3c2378087f")
TARGET_VALIDATOR_SHA = (
    "d882af175785cdcfb3a6ec5478039969a465bf156abae2e201d040b2208d59cd")
TARGET_RECEIPT_SHA = (
    "9832aa7c96a8a8699cde2bd29e249c124d29684ed42d7e2669e8b8c164fd7aae")
TARGET_REVOKED_CATALOG_SHA = (
    "142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd")


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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def support_ste_oracle(values):
    require(bool(torch.isfinite(values).all().item()), "nonfinite source")
    support = values.detach().ne(0).to(dtype=values.dtype)
    # Exact binary forward value with identity derivative.
    return values + (support - values).detach()


def cost_oracle(values, patterns):
    support = support_ste_oracle(values).reshape(-1, 432, 16)
    distance = torch.abs(
        support.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    baseline = support.sum(dim=-1)
    candidate = torch.minimum(
        baseline, 1.0 + distance.min(dim=2).values)
    return candidate.sum(), baseline.sum(), int(candidate.numel()), support


def arithmetic_and_gradient():
    patterns = torch.zeros((432, 16, 16), dtype=torch.float64)
    patterns[:, :, :4] = 1.0

    # Recompute the receipt's directed 8 -> 4 case.
    directed_patterns = torch.ones((432, 16, 16), dtype=torch.float64)
    directed_patterns[:, 0, :] = 0.0
    directed_patterns[:, 0, :4] = 1.0
    directed = torch.zeros((4, 6912), dtype=torch.float64)
    directed[1, :4] = 1.0
    directed[2, :3] = 1.0
    directed[3, 8] = 1.0
    directed_candidate, directed_baseline, directed_n, _ = cost_oracle(
        directed, directed_patterns)
    require(float(directed_candidate) == 4.0, "directed candidate mismatch")
    require(float(directed_baseline) == 8.0, "directed baseline mismatch")
    require(directed_n == 1728, "directed population mismatch")

    low = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    high = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        low[:, :4] = 0.2
        high[:, :4] = 0.9
    low_c, low_b, low_n, low_s = cost_oracle(low, patterns)
    high_c, high_b, high_n, high_s = cost_oracle(high, patterns)
    require(torch.equal(low_s, high_s), "support changed with amplitude")
    require(set(low_s.unique().tolist()) == {0.0, 1.0},
            "forward support is not binary")
    require(float(low_c.detach()) == float(high_c.detach()) == 2.0,
            "exact-pattern candidate mismatch")
    require(float(low_b.detach()) == float(high_b.detach()) == 8.0,
            "support baseline mismatch")
    require(low_n == high_n == 864, "amplitude population mismatch")
    (low_c + low_b).backward()
    (high_c + high_b).backward()
    for name, gradient in (("low", low.grad), ("high", high.grad)):
        require(gradient is not None, name + " gradient absent")
        require(bool(torch.isfinite(gradient).all().item()),
                name + " gradient nonfinite")
        require(float(gradient.abs().sum()) > 0.0,
                name + " gradient identically zero")
    require(torch.equal(low.grad, high.grad),
            "identity-STE total gradient changed with amplitude")

    # Candidate-only gradient, avoiding reliance on the baseline term.
    near_low = torch.zeros((1, 6912), dtype=torch.float64,
                           requires_grad=True)
    near_high = torch.zeros((1, 6912), dtype=torch.float64,
                            requires_grad=True)
    with torch.no_grad():
        near_low[:, :3] = 0.2
        near_high[:, :3] = 0.9
    near_low_c, _, _, _ = cost_oracle(near_low, patterns)
    near_high_c, _, _, _ = cost_oracle(near_high, patterns)
    require(float(near_low_c.detach()) ==
            float(near_high_c.detach()) == 2.0,
            "near-pattern candidate mismatch")
    near_low_c.backward()
    near_high_c.backward()
    for name, gradient in (("near_low", near_low.grad),
                           ("near_high", near_high.grad)):
        require(gradient is not None and
                bool(torch.isfinite(gradient).all().item()),
                name + " candidate-only gradient absent/nonfinite")
        require(float(gradient.abs().sum()) > 0.0,
                name + " candidate-only gradient zero")
    require(torch.equal(near_low.grad, near_high.grad),
            "identity-STE candidate gradient changed with amplitude")

    return {
        "directed_baseline": float(directed_baseline),
        "directed_candidate": float(directed_candidate),
        "directed_speedup": float(directed_baseline / directed_candidate),
        "amplitudes": [0.2, 0.9],
        "exact_baseline_both": float(low_b.detach()),
        "exact_candidate_both": float(low_c.detach()),
        "support_equal_and_binary": True,
        "total_gradient_l1_both": float(low.grad.abs().sum()),
        "candidate_only_gradient_l1_both": float(
            near_low.grad.abs().sum()),
        "all_gradients_finite_nonzero": True,
    }


def receipt_audit():
    require(sha256(PAFT) == TARGET_PAFT_SHA, "PAFT source drift")
    require(sha256(VALIDATOR) == TARGET_VALIDATOR_SHA, "validator drift")
    require(sha256(RECEIPT) == TARGET_RECEIPT_SHA, "r5 receipt drift")
    require(sha256(REVOKED_CATALOG) == TARGET_REVOKED_CATALOG_SHA,
            "revoked catalog drift")
    payload = strict_json(RECEIPT)
    require(payload["schema"] ==
            "m75_pattern_paft_hard_support_ste_directed_unit_receipt_v2",
            "receipt schema mismatch")
    require(payload["status"] ==
            "PASS_M75_HARD_SUPPORT_AND_EXTERNAL_ADMISSION_FAIL_CLOSED",
            "receipt status mismatch")
    identity = payload["identity"]
    require(identity["pattern_paft_sha256"] == TARGET_PAFT_SHA,
            "receipt PAFT identity mismatch")
    require(identity["validator_sha256"] == TARGET_VALIDATOR_SHA,
            "receipt validator identity mismatch")
    require(identity["revoked_catalog_sha256"] ==
            TARGET_REVOKED_CATALOG_SHA,
            "receipt revoked catalog identity mismatch")
    require(float(payload["directed_cost"]["baseline_vector_ops"]) == 8.0 and
            float(payload["directed_cost"]["candidate_vector_ops"]) == 4.0,
            "receipt directed arithmetic mismatch")
    invariance = payload["support_amplitude_invariance"]
    require(float(invariance["candidate_vector_ops_both"]) == 2.0 and
            float(invariance["baseline_vector_ops_both"]) == 8.0 and
            invariance["forward_support_binary"] is True and
            float(invariance["low_gradient_l1"]) > 0.0,
            "receipt amplitude/gradient evidence mismatch")
    attacks = payload["catalog_pin_attacks"]
    expected_attacks = {
        "contract_catalog_binding_mismatch",
        "revoked_sha_even_with_old_config_override",
        "self_labeled_successor_without_contract",
        "successor_wrong_contract_sha",
        "unit_test_only_contract_rejected_by_production_loader",
    }
    require(set(attacks) == expected_attacks, "receipt attacks drift")
    require(all(row["rejected"] is True for row in attacks.values()),
            "receipt admits a catalog attack")
    require(payload["preexisting_state_attack"]["rejected"] is True,
            "receipt admits preexisting state")
    boundary = payload["claim_boundary"]
    for key in ("formal_training_launch_admitted", "accuracy_admitted",
                "heldout_speedup_admitted", "cycle_speedup_admitted",
                "rtl_or_ppa_admitted"):
        require(boundary[key] is False, "receipt overclaims " + key)
    return {
        "receipt_sha256": TARGET_RECEIPT_SHA,
        "identities_bound": True,
        "catalog_attacks_rejected": len(attacks),
        "preexisting_state_rejected": True,
        "claim_boundary_fail_closed": True,
    }


def config_and_revocation_audit():
    configs = []
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "pattern_paft:" not in text:
            continue
        payload = yaml.safe_load(text)
        cfg = payload.get("pattern_paft", {})
        row = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "enabled": cfg.get("enabled"),
            "blocked_reason": cfg.get("blocked_reason"),
            "runtime_catalog_split": payload.get("runtime", {}).get(
                "paft_catalog_split"),
            "runtime_heldout_split": payload.get("runtime", {}).get(
                "paft_heldout_split"),
        }
        configs.append(row)
    require(len(configs) == 2, "generated PAFT config population drift")
    for row in configs:
        require(row["enabled"] is False, "generated PAFT config enabled")
        require(row["blocked_reason"] ==
                "M71_VALID825_CATALOG_REVOKED_USE_TRAIN_ONLY_SUCCESSOR",
                "generated config lost revocation block")
        require(row["runtime_catalog_split"] ==
                "REVOKED_M71_VALID825_INTERNAL_SAMPLES_0_TO_4",
                "generated config catalog split drift")
        require(row["runtime_heldout_split"] ==
                "REVOKED_NOT_AN_INDEPENDENT_HELDOUT",
                "generated config heldout split drift")
    revocation = strict_json(REVOCATION)
    require(revocation["status"] ==
            "REVOKED_FOR_PAFT_TRAINING_VALID825_DATA_LEAKAGE",
            "revocation status drift")
    require(revocation["forbidden_uses"]["paft_training"] is True,
            "revoked catalog regained training use")
    require(revocation["admission"]["m71_catalog_train_eligible"] is False,
            "revoked catalog regained eligibility")
    return {
        "configs": configs,
        "all_configs_disabled": True,
        "revocation_sha256": sha256(REVOCATION),
        "revoked_catalog_train_eligible": False,
    }


def source_static_audit():
    source = PAFT.read_text(encoding="utf-8")
    require("unit_test_allow_unpinned_revoked_catalog" not in source,
            "config-controlled revoked override remains")
    require(TARGET_REVOKED_CATALOG_SHA in source,
            "revoked SHA denylist absent")
    require("contract.get(\"unit_test_only\") is not False" in source,
            "explicit unit-test contract refusal absent")
    preexisting = source.index("if existing is not None:")
    catalog_load = source.index("catalog = _load_catalog")
    state_write = source.index("setattr(model, _STATE_ATTR, state)")
    require(preexisting < catalog_load < state_write,
            "preexisting-state gate is not before loading/mutation")
    return {
        "revoked_override_absent": True,
        "unit_test_only_explicitly_refused": True,
        "preexisting_gate_precedes_catalog_and_state_mutation": True,
    }


def main():
    payload = {
        "schema": "m75_independent_hammer_r2_oracle_v1",
        "status": "PASS_M75_R2_INDEPENDENT_ORACLE",
        "arithmetic": arithmetic_and_gradient(),
        "receipt": receipt_audit(),
        "configs": config_and_revocation_audit(),
        "source": source_static_audit(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

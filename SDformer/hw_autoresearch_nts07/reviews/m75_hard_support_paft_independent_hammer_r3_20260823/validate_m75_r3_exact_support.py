#!/usr/bin/env python3
"""Independent r6 identity, exact-support, gradient, and receipt validation."""

import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
REPO = HW.parent
OVERLAY = REPO / "neuron_experiments/H9_bipolar_self_attention/overlay"
PAFT = OVERLAY / "models/STSwinNet_SNN/pattern_paft.py"
VALIDATOR = HW / "verif_m71/validate_m71_pattern_paft_unit.py"
RECEIPT = HW / (
    "results/m75_pattern_paft_hard_support_ste_unit_dev_r6_20260823/"
    "m75_pattern_paft_hard_support_ste_unit_receipt.json")
REVOKED = HW / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json")
REVOCATION = HW / "contracts/m71_valid825_catalog_revocation_r1_20260823.json"
CONFIG_DIR = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated")
TARGET_PAFT_SHA = (
    "d3eac645e5b4b2e1d9d2d5dcf9e535f936adb3be15abd86d86a4d6836120a066")
TARGET_VALIDATOR_SHA = (
    "449aa672cb65bc645d343bfafe6fa276191e4a88cdb3cfb8699ba770b5ad1133")
TARGET_RECEIPT_SHA = (
    "1a84e07b296c652f1701cc25b4b27ce69ffa71ee9d75e2a59c17c4d1e40d53e2")
TARGET_REVOKED_SHA = (
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


def oracle_support(values):
    hard = values.detach().ne(0).to(values.dtype)
    return hard + (values - values.detach())


def oracle_cost(values, patterns):
    grouped = oracle_support(values).reshape(-1, 432, 16)
    distance = torch.abs(
        grouped.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    baseline = grouped.sum(dim=-1)
    candidate = torch.minimum(
        baseline, 1.0 + distance.min(dim=2).values)
    return candidate.sum(), baseline.sum(), int(candidate.numel())


def load_exact_target():
    require(sha256(PAFT) == TARGET_PAFT_SHA, "r6 PAFT source drift")
    require(sha256(VALIDATOR) == TARGET_VALIDATOR_SHA, "r6 validator drift")
    require(sha256(RECEIPT) == TARGET_RECEIPT_SHA, "r6 receipt drift")
    require(sha256(REVOKED) == TARGET_REVOKED_SHA, "revoked catalog drift")
    sys.path.insert(0, str(OVERLAY))
    from models.STSwinNet_SNN.pattern_paft import (  # noqa: E402
        _cost_proxy,
        _hard_support_ste,
    )
    return _hard_support_ste, _cost_proxy


def multidtype_exactness(production_support):
    rows = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        values = torch.tensor(
            [-0.9, -0.3, -0.1, -1.0e-3, 0.0,
             1.0e-3, 0.1, 0.2, 0.3, 0.7, 0.9],
            dtype=dtype, requires_grad=True)
        observed = production_support(values)
        expected = values.detach().ne(0).to(dtype)
        independent = oracle_support(values)
        require(torch.equal(observed.detach(), expected),
                "production forward not exact for " + str(dtype))
        require(torch.equal(observed.detach(), independent.detach()),
                "production/oracle support mismatch for " + str(dtype))
        observed.sum().backward()
        require(torch.equal(values.grad, torch.ones_like(values.grad)),
                "production gradient not exact one for " + str(dtype))
        rows.append({
            "dtype": str(dtype),
            "forward_torch_equal_binary": True,
            "gradient_torch_equal_one": True,
            "maximum_forward_error": 0.0,
        })
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            production_support(torch.tensor([value], dtype=torch.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("production accepted nonfinite source")
    return {
        "rows": rows,
        "all_four_dtypes_exact": True,
        "nonfinite_values_rejected": 3,
    }


def cost_and_gradient_equivalence(production_cost):
    patterns = torch.zeros((432, 16, 16), dtype=torch.float64)
    patterns[:, :, :4] = 1.0
    values = torch.zeros((2, 6912), dtype=torch.float64,
                         requires_grad=True)
    with torch.no_grad():
        values[0, :4] = 0.2
        values[1, :3] = 0.9
    prod_c, prod_b, prod_n = production_cost(values, patterns, 31)
    oracle_values = values.detach().clone().requires_grad_(True)
    oracle_c, oracle_b, oracle_n = oracle_cost(oracle_values, patterns)
    require(torch.equal(prod_c.detach(), oracle_c.detach()),
            "candidate forward differs from oracle")
    require(torch.equal(prod_b.detach(), oracle_b.detach()),
            "baseline forward differs from oracle")
    require(prod_n == oracle_n == 864, "proxy element count mismatch")
    prod_c.backward()
    oracle_c.backward()
    require(torch.equal(values.grad, oracle_values.grad),
            "candidate gradient differs from oracle")
    require(bool(torch.isfinite(values.grad).all().item()) and
            float(values.grad.abs().sum()) > 0.0,
            "candidate gradient absent/nonfinite/zero")

    directed_patterns = torch.ones((432, 16, 16), dtype=torch.float64)
    directed_patterns[:, 0, :] = 0.0
    directed_patterns[:, 0, :4] = 1.0
    directed = torch.zeros((4, 6912), dtype=torch.float64)
    directed[1, :4] = 1.0
    directed[2, :3] = 1.0
    directed[3, 8] = 1.0
    directed_c, directed_b, directed_n = production_cost(
        directed, directed_patterns, 37)
    require(float(directed_c) == 4.0 and float(directed_b) == 8.0,
            "directed 8-to-4 mismatch")
    require(directed_n == 1728, "directed element count mismatch")
    return {
        "candidate": float(prod_c.detach()),
        "baseline": float(prod_b.detach()),
        "candidate_gradient_l1": float(values.grad.abs().sum()),
        "forward_equal_to_independent_oracle": True,
        "gradient_equal_to_independent_oracle": True,
        "directed_baseline": float(directed_b),
        "directed_candidate": float(directed_c),
        "directed_speedup": float(directed_b / directed_c),
    }


def receipt_audit():
    payload = strict_json(RECEIPT)
    require(payload["schema"] ==
            "m75_pattern_paft_hard_support_ste_directed_unit_receipt_v3",
            "r6 schema mismatch")
    require(payload["status"] ==
            "PASS_M75_MULTI_DTYPE_EXACT_HARD_SUPPORT_AND_EXTERNAL_ADMISSION_FAIL_CLOSED",
            "r6 status mismatch")
    identity = payload["identity"]
    require(identity["pattern_paft_sha256"] == TARGET_PAFT_SHA,
            "receipt PAFT identity mismatch")
    require(identity["validator_sha256"] == TARGET_VALIDATOR_SHA,
            "receipt validator identity mismatch")
    require(identity["revoked_catalog_sha256"] == TARGET_REVOKED_SHA,
            "receipt revoked identity mismatch")
    rows = payload["support_multidtype_exactness"]
    require([row["dtype"] for row in rows] == [
        "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"],
        "receipt dtype extent/order mismatch")
    require(all(row["forward_bit_exact_binary"] is True and
                row["gradient_bit_exact_one"] is True for row in rows),
            "receipt records a dtype failure")
    require(all(row["rejected"] is True for row in
                payload["catalog_pin_attacks"].values()),
            "receipt records an accepted attack")
    require(payload["preexisting_state_attack"]["rejected"] is True,
            "receipt records preexisting-state acceptance")
    boundary = payload["claim_boundary"]
    for key in ("formal_training_launch_admitted", "accuracy_admitted",
                "heldout_speedup_admitted", "cycle_speedup_admitted",
                "rtl_or_ppa_admitted"):
        require(boundary[key] is False, "receipt overclaims " + key)
    return {
        "receipt_sha256": TARGET_RECEIPT_SHA,
        "identities_bound": True,
        "four_dtype_claims_present": True,
        "claim_boundary_fail_closed": True,
    }


def disabled_config_audit():
    rows = []
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "pattern_paft:" not in text:
            continue
        payload = yaml.safe_load(text)
        cfg = payload["pattern_paft"]
        require(cfg.get("enabled") is False,
                "generated PAFT config enabled")
        require(cfg.get("blocked_reason") ==
                "M71_VALID825_CATALOG_REVOKED_USE_TRAIN_ONLY_SUCCESSOR",
                "generated PAFT config block drift")
        rows.append({"path": str(path.relative_to(REPO)),
                     "sha256": sha256(path), "enabled": False})
    require(len(rows) == 2, "generated PAFT config population drift")
    revocation = strict_json(REVOCATION)
    require(revocation["forbidden_uses"]["paft_training"] is True and
            revocation["admission"]["m71_catalog_train_eligible"] is False,
            "revocation contract drift")
    return {"configs": rows, "all_disabled": True,
            "revocation_sha256": sha256(REVOCATION)}


def main():
    production_support, production_cost = load_exact_target()
    payload = {
        "schema": "m75_independent_hammer_r3_exact_support_v1",
        "status": "PASS_R6_EXACT_SUPPORT_INDEPENDENT_ORACLE",
        "identity": {
            "pattern_paft_sha256": TARGET_PAFT_SHA,
            "validator_sha256": TARGET_VALIDATOR_SHA,
            "receipt_sha256": TARGET_RECEIPT_SHA,
        },
        "multidtype": multidtype_exactness(production_support),
        "cost_and_gradient": cost_and_gradient_equivalence(production_cost),
        "receipt": receipt_audit(),
        "configs": disabled_config_audit(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

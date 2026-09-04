#!/usr/bin/env python3
"""Reproduce the exact-binary counterexample for immutable M75 r5.

The r5 receipt pins the reviewed implementation SHA.  Its reviewed expression
was ``hard + vectors - vectors.detach()``.  This isolated reproducer retains
that expression after the live source is superseded, so the historical failure
cannot be erased by a later fix.
"""

import hashlib
import json
from pathlib import Path

import torch


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
R5_RECEIPT = HW / (
    "results/m75_pattern_paft_hard_support_ste_unit_dev_r5_20260823/"
    "m75_pattern_paft_hard_support_ste_unit_receipt.json")
TARGET_R5_RECEIPT_SHA = (
    "9832aa7c96a8a8699cde2bd29e249c124d29684ed42d7e2669e8b8c164fd7aae")
TARGET_R5_PAFT_SHA = (
    "bf15a2ea328a16d1d8c676de11a041fc3768c6717bcf73a21c1c6e3c2378087f")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r5_expression(vectors):
    hard = (vectors != 0).to(dtype=vectors.dtype)
    return hard + vectors - vectors.detach()


def exact_expression(vectors):
    hard = (vectors != 0).to(dtype=vectors.dtype)
    return hard + (vectors - vectors.detach())


def cost(values, patterns, expression):
    support = expression(values).reshape(-1, 432, 16)
    distance = torch.abs(
        support.unsqueeze(2) - patterns.unsqueeze(0)).sum(dim=-1)
    baseline = support.sum(dim=-1)
    candidate = torch.minimum(
        baseline, 1.0 + distance.min(dim=2).values)
    return candidate.sum(), baseline.sum()


def main():
    if sha256(R5_RECEIPT) != TARGET_R5_RECEIPT_SHA:
        raise AssertionError("r5 receipt drift")
    receipt = json.loads(R5_RECEIPT.read_text(encoding="utf-8"))
    if receipt["identity"]["pattern_paft_sha256"] != TARGET_R5_PAFT_SHA:
        raise AssertionError("r5 implementation identity drift")

    sweep = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        values = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.7, 0.9, 1.0e-3, -0.2],
            dtype=dtype, requires_grad=True)
        observed = r5_expression(values)
        expected = exact_expression(values)
        observed.sum().backward()
        sweep.append({
            "dtype": str(dtype),
            "observed": [float(item) for item in observed.detach()],
            "maximum_forward_error": float(
                (observed.detach() - expected.detach()).abs().max()),
            "exact_binary": bool(torch.equal(
                observed.detach(), expected.detach())),
            "identity_gradient": bool(torch.equal(
                values.grad, torch.ones_like(values.grad))),
        })

    patterns = torch.zeros((432, 16, 16), dtype=torch.float64)
    patterns[:, :, :4] = 1.0
    values = torch.zeros((2, 6912), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        values[0, :4] = 0.2
        values[1, :3] = 0.9
    r5_candidate, r5_baseline = cost(values, patterns, r5_expression)
    r5_candidate.backward()
    r5_gradient = values.grad.detach().clone()
    oracle_values = values.detach().clone().requires_grad_(True)
    oracle_candidate, oracle_baseline = cost(
        oracle_values, patterns, exact_expression)
    oracle_candidate.backward()
    oracle_gradient = oracle_values.grad.detach()

    payload = {
        "schema": "m75_r5_exact_support_counterexample_v1",
        "status": "FAIL_R5_FORWARD_NOT_EXACT_BINARY",
        "identity": {
            "r5_receipt_sha256": TARGET_R5_RECEIPT_SHA,
            "r5_pattern_paft_sha256": TARGET_R5_PAFT_SHA,
            "reviewed_r5_expression": "hard + vectors - vectors.detach()",
        },
        "dtype_sweep": sweep,
        "all_dtypes_exact_binary": all(row["exact_binary"] for row in sweep),
        "all_dtypes_identity_gradient": all(
            row["identity_gradient"] for row in sweep),
        "cost_and_gradient_counterexample": {
            "r5_candidate": float(r5_candidate.detach()),
            "oracle_candidate": float(oracle_candidate.detach()),
            "candidate_delta": float(
                (r5_candidate - oracle_candidate).detach()),
            "r5_baseline": float(r5_baseline.detach()),
            "oracle_baseline": float(oracle_baseline.detach()),
            "baseline_delta": float((r5_baseline - oracle_baseline).detach()),
            "r5_gradient_l1": float(r5_gradient.abs().sum()),
            "oracle_gradient_l1": float(oracle_gradient.abs().sum()),
            "maximum_gradient_delta": float(
                (r5_gradient - oracle_gradient).abs().max()),
            "gradient_exact": bool(torch.equal(
                r5_gradient, oracle_gradient)),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["all_dtypes_exact_binary"]:
        raise AssertionError("counterexample unexpectedly disappeared")


if __name__ == "__main__":
    main()

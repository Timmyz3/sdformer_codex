#!/usr/bin/env python3
"""CPU-only arithmetic preflight; does not import or execute the old model.

The BN example is a counterexample about dependencies, not a measured model
output or AEE result. Compute quantities are the existing dense static proxy,
not simulator clocks, RTL measurements, or GPU operations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ledger = repo / "research/flow_story_20261010/FLOW_V2_STATIC_COST_SPLIT_20261011.json"
    model_dir = repo / "SDformer/third_party/SDformerFlow"
    sources = [
        ledger,
        model_dir / "models/STSwinNet_SNN/Spiking_modules.py",
        model_dir / "models/STSwinNet_SNN/Spiking_swin_transformer3D.py",
        model_dir / "eval_DSEC_flow_SNN.py",
    ]
    data = json.loads(ledger.read_text())
    detail = data["compute"]["detail"]
    total = data["compute"]["dense_MAC_PE_cycles_UPPER"]
    assert sum(detail.values()) == total
    attn = sum(v for k, v in detail.items() if k.endswith("_attn"))
    mlp = sum(v for k, v in detail.items() if k.endswith("_mlp"))
    merge = sum(v for k, v in detail.items() if k.endswith("_merge"))

    # Small deterministic per-channel BN example. Spatial locations 2 and 3
    # are designated "bypassed", while locations 0 and 1 must remain live.
    eps = 1e-5
    full = np.array([0.0, 1.0, 100.0, 101.0], dtype=np.float64)
    keep = np.array([True, True, False, False])
    def current_bn(x):
        return (x - x.mean()) / np.sqrt(x.var() + eps)
    full_result = current_bn(full)[keep]
    packed_result = current_bn(full[keep])
    zeroed = np.where(keep, full, 0.0)
    zero_result = current_bn(zeroed)[keep]
    mean, var = full.mean(), full.var()
    def frozen_bn(x):
        return (x - mean) / np.sqrt(var + eps)
    frozen_full = frozen_bn(full)[keep]
    frozen_packed = frozen_bn(full[keep])
    assert not np.allclose(full_result, packed_result)
    assert not np.allclose(full_result, zero_result)
    np.testing.assert_array_equal(frozen_full, frozen_packed)

    candidates = {}
    for name, removable in [("attention_and_mlp", attn + mlp), ("mlp_only", mlp)]:
        share = removable / total
        candidates[name] = {
            "candidate_dense_work": removable,
            "share_of_dense_total": share,
            "all_cancelled_zero_overhead_proxy_speedup": 1 / (1 - share),
            "required_effective_cancel_fraction_for_10pct_net_work_reduction": {
                str(overhead): (0.10 + overhead) / share
                for overhead in [0.0, 0.01, 0.02, 0.05]
            },
            "overhead_fractions_are_hypothetical_not_measured": True,
        }
    result = {
        "evidence_level": "CPU arithmetic counterexample and static dense-work recomputation",
        "gpu_used": False,
        "old_model_executed": False,
        "source_sha256": {str(p.relative_to(repo)): digest(p) for p in sources},
        "bn_counterexample": {
            "epsilon": eps,
            "input": full.tolist(),
            "kept_positions": np.flatnonzero(keep).tolist(),
            "full_current_statistics_kept_output": full_result.tolist(),
            "packed_kept_only_statistics_output": packed_result.tolist(),
            "zeroed_skipped_statistics_kept_output": zero_result.tolist(),
            "frozen_stats_full_vs_packed_max_abs_diff": float(np.max(np.abs(frozen_full - frozen_packed))),
            "scope": "Standard per-channel BN arithmetic only. This does not measure the network, its AEE, or a deployment conversion.",
        },
        "static_scope": {
            "total": total, "attention": attn, "mlp": mlp, "merge_not_cancelled": merge,
            "input": data["resolution"],
            "candidates": candidates,
            "not_a_cycle_simulator": True,
        },
    }
    out = here / "bn_and_compute_scope_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "bn_dependency_counterexample": "PASS", "dense_work_total_check": "PASS", "candidate_shares": {k: v["share_of_dense_total"] for k, v in candidates.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

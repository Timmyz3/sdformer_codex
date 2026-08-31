#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""M1537: checkpoint-bound static N:M weight-pruning opportunity audit.

This is deliberately not an accuracy, cycle, traffic, energy, or RTL result.
It measures how much FP32 weight mass an oracle magnitude mask would remove in
the ep34 checkpoint before authorizing any retraining or sparse hardware work.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M1531 = HW / "results/m1531_ep34_sparse_energy_first_principles_fastkill_r1_20260831/m1531_ep34_sparse_energy_first_principles_fastkill_r1.json"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HW / "results/m1537_ep34_nm_weight_pruning_static_fastkill_r1_20260831"

EXPECTED = {
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    M1531: "f880d75b2fd885a584d69724357add53b9abab0a9ca5df08281fb2d76dfdd5f8",
    M1512: "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

PATTERNS = ((2, 4), (4, 8), (8, 16), (16, 32), (2, 8), (4, 16))
SELECTED_CATEGORIES = (
    "patch_embed",
    "fc1",
    "fc2",
    "bottleneck_conv",
    "decoder",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def category(name):
    if not name.endswith(".weight"):
        return None
    if ".mlp.fc1" in name:
        return "fc1"
    if ".mlp.fc2" in name:
        return "fc2"
    if ".patch_embed." in name:
        return "patch_embed"
    if name.startswith("sttmultires_unet.resblocks.") and ".conv" in name:
        return "bottleneck_conv"
    if ".decoders." in name and ".deconv.0" in name:
        return "decoder"
    return None


def binomial_metadata_bits(n, m):
    # Information-theoretic lower bound only; real selector/layout costs more.
    combinations = math.factorial(m) // (math.factorial(n) * math.factorial(m - n))
    return int(math.ceil(math.log(float(combinations), 2.0)))


def audit_pattern(tensors, n, m):
    total_weights = 0
    grouped_weights = 0
    total_l1 = 0.0
    removed_l1 = 0.0
    total_l2_sq = 0.0
    removed_l2_sq = 0.0
    groups = 0
    for tensor in tensors:
        values = tensor.detach().float().abs().reshape(-1)
        usable = (values.numel() // m) * m
        total_weights += int(values.numel())
        if usable == 0:
            continue
        block = values[:usable].reshape(-1, m)
        kept = block.topk(n, dim=1, largest=True, sorted=False).values
        grouped_weights += usable
        groups += int(block.shape[0])
        total_l1 += float(block.sum())
        removed_l1 += float(block.sum() - kept.sum())
        total_l2_sq += float(block.square().sum())
        removed_l2_sq += float(block.square().sum() - kept.square().sum())
    require(groups > 0 and total_l1 > 0.0 and total_l2_sq > 0.0,
            "empty N:M audit population")
    return {
        "pattern": "{}:{}".format(n, m),
        "kept_fraction_by_count": float(n) / float(m),
        "pruned_fraction_by_count": 1.0 - float(n) / float(m),
        "tensors": len(tensors),
        "total_weights": total_weights,
        "grouped_weights": grouped_weights,
        "tail_weights_not_pruned": total_weights - grouped_weights,
        "groups": groups,
        "oracle_magnitude_removed_l1_fraction": removed_l1 / total_l1,
        "oracle_magnitude_removed_l2_squared_fraction": removed_l2_sq / total_l2_sq,
        "selector_metadata_information_lower_bound_bits_per_group": binomial_metadata_bits(n, m),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        require(path.is_file(), "missing input: {}".format(path))
        require(sha256(path) == expected, "SHA mismatch: {}".format(path))
    m1531 = json.loads(M1531.read_text())
    require(m1531["status"] ==
            "PASS_OPPORTUNITY_AUDIT__NO_SPEEDUP_ENERGY_OR_RTL_ADMISSION",
            "M1531 authority drift")
    require(m1531["measured_facts"]["all_selected_weight_exact_zero_fraction"] == 0.0,
            "M1531 exact-zero fact drift")
    m1512 = json.loads(M1512.read_text())
    require(m1512["status"] ==
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT",
            "M1512 authority drift")

    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    require(set(wrapper) == {"model_state_dict"}, "checkpoint wrapper drift")
    grouped = {key: [] for key in SELECTED_CATEGORIES}
    names = {key: [] for key in SELECTED_CATEGORIES}
    for name, tensor in wrapper["model_state_dict"].items():
        group = category(name)
        if group is None:
            continue
        require(torch.is_tensor(tensor), "non-tensor weight: {}".format(name))
        if tensor.ndim < 2:
            continue
        grouped[group].append(tensor)
        names[group].append(name)
    for group in SELECTED_CATEGORIES:
        require(grouped[group], "missing category: {}".format(group))

    categories = {}
    for group in SELECTED_CATEGORIES:
        categories[group] = {
            "tensor_names": names[group],
            "patterns": [audit_pattern(grouped[group], n, m) for n, m in PATTERNS],
        }

    half_4x8 = [categories[group]["patterns"][1]
                ["oracle_magnitude_removed_l1_fraction"] for group in SELECTED_CATEGORIES]
    half_8x16 = [categories[group]["patterns"][2]
                 ["oracle_magnitude_removed_l1_fraction"] for group in SELECTED_CATEGORIES]
    result = {
        "schema": "m1537_ep34_nm_weight_pruning_static_fastkill_v1",
        "status": "PASS_STATIC_OPPORTUNITY__RETRAIN_AND_AEE_REQUIRED__NO_HARDWARE_ADMISSION",
        "identity": {
            "checkpoint_sha256": EXPECTED[CHECKPOINT],
            "m1531_sha256": EXPECTED[M1531],
            "m1512_sha256": EXPECTED[M1512],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "mapping": {
            "grouping": "each checkpoint tensor independently, contiguous checkpoint storage order",
            "mask": "oracle magnitude top-N within each M group",
            "numeric_domain": "FP32 checkpoint weights; no INT8/PTQ identity is assumed",
            "tail_policy": "tensor tail shorter than M is retained and excluded from pruning mass",
        },
        "categories": categories,
        "aggregate_observations": {
            "four_of_eight_l1_removed_range": [min(half_4x8), max(half_4x8)],
            "eight_of_sixteen_l1_removed_range": [min(half_8x16), max(half_8x16)],
            "exact_zero_pruning_available": False,
            "fifty_percent_nm_is_lossy": True,
        },
        "decision": {
            "lossless_nm": "KILL",
            "direct_post_training_magnitude_mask": "NO_GO_WITHOUT_PAIRED_AEE",
            "hardware_aware_nm_retraining": "CANDIDATE_POOL_ONLY",
            "preferred_first_training_grid": ["4:8", "8:16"],
            "reason": "50% oracle masks remove material L1/L2 weight mass; only paired hardware-aware retraining can establish an accuracy Pareto",
            "promotion_gate": {
                "accuracy": "overall Delta-AEE <= 0.02 and every sequence Delta-AEE <= 0.03",
                "hardware": "same-resource local cycles >=1.15x, or cycles regress <=5% with weight bytes >=30% and measured memory energy >=20% lower",
                "identity": "new checkpoint/config/mask SHA and full ep34-equivalent capture/replay are mandatory",
                "implementation": "fixed selector metadata, weight fetch, decompressor, tails, bank conflicts and psum updates are all charged",
            },
        },
        "claim_boundary": {
            "static_weight_mass": True,
            "structured_mask_opportunity": True,
            "accuracy": False,
            "aee": False,
            "cycles": False,
            "speedup": False,
            "traffic": False,
            "energy": False,
            "rtl": False,
            "paper_headline": False,
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m1537_ep34_nm_weight_pruning_static_fastkill_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path = args.output / "m1537_REPORT.md"
    report_path.write_text(
        "# M1537 ep34 N:M static weight-pruning fast-kill\n\n"
        "Status: **{}**.\n\n"
        "The released ep34 checkpoint has no lossless N:M path. Across patch embed, FC1, FC2, "
        "bottleneck Conv and decoder, an oracle 4:8 magnitude mask removes {:.2f}%--{:.2f}% "
        "of L1 weight mass; 8:16 removes {:.2f}%--{:.2f}%. These are static FP32 "
        "opportunity measurements, not accuracy, cycle, traffic, energy or RTL results. "
        "Only a new hardware-aware retrained checkpoint with paired AEE and same-resource "
        "replay may promote this candidate.\n".format(
            result["status"], 100.0 * min(half_4x8), 100.0 * max(half_4x8),
            100.0 * min(half_8x16), 100.0 * max(half_8x16)))
    sums = []
    for path in (result_path, report_path):
        sums.append("{}  {}".format(sha256(path), path.name))
    sums_path = args.output / "SHA256SUMS"
    sums_path.write_text("\n".join(sums) + "\n")
    (args.output / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums_path)))
    print("PASS M1537 categories={} patterns={} output={}".format(
        len(categories), len(PATTERNS), args.output))


if __name__ == "__main__":
    main()

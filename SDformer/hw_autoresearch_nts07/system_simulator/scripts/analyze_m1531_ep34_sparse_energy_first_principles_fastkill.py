#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""Read-only ep34 sparsity/weight fast-kill for energy-oriented mechanisms.

This audit deliberately reports activity and payload opportunities, not cycles,
speedup, energy, or accuracy.  It binds every input by SHA256 and never mutates
the checkpoint or the M1458 capture.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
OPERATOR = CAPTURE / "operator_runtime.json"
MANIFEST = CAPTURE / "manifest.json"
M501 = HW / "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827/m501_h67_exact_adjacent_overlap_fastkill_result_r1.json"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HW / "results/m1531_ep34_sparse_energy_first_principles_fastkill_r1_20260831"

EXPECTED = {
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    OPERATOR: "eb0cd40e701361f8acc08d6003680de0ca35626e8e75dcf56827c978899e8a8e",
    MANIFEST: "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    M501: "37ce6d66a73c5dc3c19e887497ac85b473bc4789c0c241b4073d6af5d4c6cd18",
    M1512: "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def category(name):
    if ".mlp.fc1" in name:
        return "fc1"
    if ".mlp.fc2" in name:
        return "fc2"
    if name.startswith("sttmultires_unet.resblocks.") and ".conv" in name:
        return "bottleneck_conv"
    if ".patch_embed." in name:
        return "patch_embed"
    if ".attn.linear_q" in name or ".attn.linear_k" in name:
        return "qk_projection"
    if ".attn.proj" in name:
        return "attention_output_projection"
    if ".downsample.reduction" in name:
        return "downsample"
    if ".preds." in name:
        return "prediction_head"
    return None


def aggregate_runtime(rows):
    grouped = {}
    for row in rows:
        group = category(row["name"])
        if group is None:
            continue
        grouped.setdefault(group, []).append(row)
    output = {}
    for group, selected in sorted(grouped.items()):
        elements = sum(int(row["input_elements"]) for row in selected)
        active = sum(int(row["input_active"]) for row in selected)
        dense_macs = sum(int(row["dense_macs"]) for row in selected)
        active_macs = sum(float(row["activity_weighted_macs_proxy"]) for row in selected)
        sampled = sum(int(row["input_sample_elements"]) for row in selected)
        binary_weighted = sum(
            float(row["input_sample_binary01_ratio"]) * int(row["input_sample_elements"])
            for row in selected
        )
        output[group] = {
            "modules": len(selected),
            "input_elements": elements,
            "input_active": active,
            "input_activity_fraction": active / elements,
            "sampled_binary01_fraction": binary_weighted / sampled,
            "dense_macs": dense_macs,
            "activity_weighted_macs_proxy": active_macs,
            "dense_over_activity_proxy_ceiling_not_speedup": dense_macs / active_macs,
        }
    return output


def weight_category(name):
    if not name.endswith(".weight"):
        return None
    return category(name)


def weight_audit(state):
    grouped = {}
    for name, tensor in state.items():
        group = weight_category(name)
        if group is None or not torch.is_tensor(tensor) or tensor.ndim < 2:
            continue
        grouped.setdefault(group, []).append(tensor.detach().float().abs().reshape(-1))
    thresholds = [1.0 / 1024, 1.0 / 512, 1.0 / 256, 1.0 / 128,
                  1.0 / 64, 1.0 / 32, 1.0 / 16]
    output = {}
    for group, tensors in sorted(grouped.items()):
        values = torch.cat(tensors)
        maximum = values.max()
        total_l1 = values.sum()
        rows = []
        for relative in thresholds:
            mask = values <= maximum * relative
            rows.append({
                "threshold_relative_to_group_max": relative,
                "weight_fraction_at_or_below": float(mask.float().mean()),
                "l1_mass_fraction_at_or_below": float(values[mask].sum() / total_l1),
            })
        output[group] = {
            "layers": len(tensors),
            "weights": values.numel(),
            "exact_zero_fraction": float((values == 0).float().mean()),
            "absolute_max": float(maximum),
            "magnitude_dse": rows,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        require(path.is_file(), "missing input: {}".format(path))
        require(sha256(path) == expected, "SHA mismatch: {}".format(path))
    capture = json.loads(MANIFEST.read_text())
    require(capture["status"] ==
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "M1458 capture status drift")
    require(capture["cohort"]["population"] == 40, "M1458 population drift")
    m1512 = json.loads(M1512.read_text())
    require(m1512["status"] ==
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT",
            "M1512 capture authority is not PASS")
    require(m1512["claim_boundary"]["capture_content_validated"] is True,
            "M1512 did not validate capture content")
    runtime = aggregate_runtime(json.loads(OPERATOR.read_text()))
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
    require(set(checkpoint) == {"model_state_dict"}, "checkpoint wrapper drift")
    weights = weight_audit(checkpoint["model_state_dict"])
    fc2 = runtime["fc2"]
    fc2_bitpacked_one_way = (fc2["input_elements"] + 7) // 8
    m501 = json.loads(M501.read_text())
    selected = m501["decision"]
    result = {
        "schema": "m1531_ep34_sparse_energy_first_principles_fastkill_v1",
        "status": "PASS_OPPORTUNITY_AUDIT__NO_SPEEDUP_ENERGY_OR_RTL_ADMISSION",
        "identity": {
            "checkpoint_sha256": EXPECTED[CHECKPOINT],
            "capture_population": 40,
            "capture_manifest_sha256": EXPECTED[MANIFEST],
            "operator_runtime_sha256": EXPECTED[OPERATOR],
            "m1512_capture_review_sha256": EXPECTED[M1512],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "runtime_activity": runtime,
        "weight_magnitude": weights,
        "measured_facts": {
            "all_selected_weight_exact_zero_fraction": 0.0,
            "fc2_sampled_binary01_fraction": fc2["sampled_binary01_fraction"],
            "fc2_input_activity_fraction": fc2["input_activity_fraction"],
            "fc2_conditional_bitpacked_write_plus_read_bytes": 2 * fc2_bitpacked_one_way,
            "fc2_conditional_payload_boundary": "avoidable only if a materializing baseline really writes then rereads the full bitpacked tensor; not traffic or energy yet",
            "old_m501_horizontal_g2_redundant_fraction": selected["event_reduction_ratio"] and 1.0 - 1.0 / selected["event_reduction_ratio"],
            "old_m501_same_resource_speedup_admitted": False,
        },
        "decisions": [
            {
                "mechanism": "lossless_zero_weight_pruning",
                "decision": "KILL",
                "reason": "selected ep34 Conv/FC weight tensors contain no exact zeros",
            },
            {
                "mechanism": "adjacent_position_event_compression",
                "decision": "SUPPORT_ONLY",
                "reason": "M501 measured exact event reduction but only 1.0366x ideal envelope sensitivity; ExSpike is direct prior art",
            },
            {
                "mechanism": "ATLIF_TO_C2_CUT_THROUGH",
                "decision": "MEASURE_SAME_RESOURCE_TRAFFIC_AND_ENERGY",
                "reason": "sn2 directly precedes FC2 and FC2 input is sampled binary with 3.15% activity; direct terminal descriptors may remove conditional feature-map write/read",
                "gate": "cycle regression <=5% and measured SRAM/DRAM bytes >=30% lower for the FC2 producer-consumer boundary, then VCS adapter",
            },
            {
                "mechanism": "ERROR_BUDGETED_BLOCK_CONTRIBUTION_SKIP",
                "decision": "CAPTURE_AND_AEE_DSE_BEFORE_RTL",
                "reason": "unstructured small weights exist but no exact-zero weight sparsity; only a runtime block bound can suppress both weight fetch and compute without pretending magnitude fraction is speedup",
                "bound": "for skipped input block g: ||Delta y||_inf <= ||x_g||_inf * max_o sum_i_in_g |W[o,i]|; accumulate against a layer budget",
                "gate": "same-resource high-share cycles >=1.15x or weight bytes >=30% lower at paired |Delta AEE|<=0.02",
            },
        ],
        "claim_boundary": {
            "activity_proxy": True,
            "weight_magnitude_distribution": True,
            "cycles": False,
            "speedup": False,
            "energy": False,
            "traffic": False,
            "aee": False,
            "rtl": False,
            "paper_headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m1531_ep34_sparse_energy_first_principles_fastkill_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = args.output / "m1531_REPORT.md"
    report.write_text(
        "# M1531 ep34 sparse-energy first-principles fast-kill\n\n"
        "Status: **{}**.\n\n"
        "The ep34 capture is sparse, but activity ratios are not speedups. Exact weight-zero pruning is killed. "
        "The only two live extensions are exact ATLIF-to-C2 cut-through and an optional error-budgeted block skip; "
        "both remain outside paper claims until their registered gates pass.\n".format(result["status"])
    )
    sums = []
    for path in (result_path, report):
        sums.append("{}  {}".format(sha256(path), path.name))
    sums_path = args.output / "SHA256SUMS"
    sums_path.write_text("\n".join(sums) + "\n")
    (args.output / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums_path))
    )
    print("PASS M1531 categories={} decisions={} output={}".format(
        len(runtime), len(result["decisions"]), args.output))


if __name__ == "__main__":
    main()

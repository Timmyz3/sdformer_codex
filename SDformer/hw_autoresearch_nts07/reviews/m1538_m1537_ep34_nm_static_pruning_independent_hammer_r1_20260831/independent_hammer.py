#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""Independent mechanical audit of M1537's static N:M opportunity result."""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/analyze_m1537_ep34_nm_weight_pruning_static_fastkill.py"
RESULT_DIR = HW / "results/m1537_ep34_nm_weight_pruning_static_fastkill_r1_20260831"
RESULT = RESULT_DIR / "m1537_ep34_nm_weight_pruning_static_fastkill_r1.json"
REPORT = RESULT_DIR / "m1537_REPORT.md"
MANIFEST = RESULT_DIR / "SHA256SUMS"
OUTER = RESULT_DIR / "SHA256SUMS.seal.sha256"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M1531 = HW / "results/m1531_ep34_sparse_energy_first_principles_fastkill_r1_20260831/m1531_ep34_sparse_energy_first_principles_fastkill_r1.json"
M1512 = HW / "reviews/m1512_m1501_m1458_ep34_capture_source_result_independent_hammer_r1_20260831/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    SOURCE: "f9b96a2bc2506c857a7d0668ff354aa7453219bea09ddaa9dface9566481a5e2",
    RESULT: "1dbd8e42fde5dfe76ab1e79b110cac9040cdff14bfd19308ddc1d77e1e0977fe",
    REPORT: "2743348af3eb36618c5586dc40c690ba7214358f42e0bdbb1f6c0e3bd1ca63e8",
    MANIFEST: "09d48eb4f1c6707ab7f45ff3ff27f96db67abce775ea11bd719ce27ce0a3f3a8",
    OUTER: "db0b8c19ebd4e5eba53401608d47a48ed8ced7c33da998d8a53637d08726151e",
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    M1531: "f880d75b2fd885a584d69724357add53b9abab0a9ca5df08281fb2d76dfdd5f8",
    M1512: "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError("nonfinite JSON token " + token)))


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


def main():
    checks = []
    for path, expected in PINS.items():
        require(path.is_file() and not path.is_symlink(), "missing/unsafe " + str(path))
        require(sha256(path) == expected, "SHA drift " + str(path))
        checks.append("pin:" + path.name)

    expected_manifest = {
        "{}  m1537_ep34_nm_weight_pruning_static_fastkill_r1.json".format(PINS[RESULT]),
        "{}  m1537_REPORT.md".format(PINS[REPORT]),
    }
    require(set(MANIFEST.read_text(encoding="utf-8").splitlines()) == expected_manifest,
            "inner manifest membership drift")
    require(OUTER.read_text(encoding="utf-8").split() == [PINS[MANIFEST], "SHA256SUMS"],
            "outer seal content drift")
    checks.extend(["inner_manifest", "outer_seal"])

    result = load(RESULT)
    m1531 = load(M1531)
    m1512 = load(M1512)
    require(result["status"] ==
            "PASS_STATIC_OPPORTUNITY__RETRAIN_AND_AEE_REQUIRED__NO_HARDWARE_ADMISSION",
            "result status drift")
    require(m1531["status"] ==
            "PASS_OPPORTUNITY_AUDIT__NO_SPEEDUP_ENERGY_OR_RTL_ADMISSION",
            "M1531 status drift")
    require(m1512["status"] ==
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT",
            "M1512 status drift")
    require(result["identity"] == {
        "checkpoint_sha256": PINS[CHECKPOINT],
        "docs359_sha256": PINS[DOCS359],
        "m1512_sha256": PINS[M1512],
        "m1531_sha256": PINS[M1531],
    }, "embedded identity drift")
    require(m1531["measured_facts"]["all_selected_weight_exact_zero_fraction"] == 0.0,
            "exact-zero upstream fact drift")
    checks.extend(["status_chain", "embedded_identity", "zero_fact"])

    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    require(set(wrapper) == {"model_state_dict"}, "checkpoint wrapper drift")
    tensors = {name: [] for name in result["categories"]}
    for name, tensor in wrapper["model_state_dict"].items():
        group = category(name)
        if group is not None and tensor.ndim >= 2:
            tensors[group].append((name, tensor))

    max_l1_delta = 0.0
    max_l2_delta = 0.0
    for group, payload in result["categories"].items():
        require([name for name, _ in tensors[group]] == payload["tensor_names"],
                "tensor membership/order drift for " + group)
        for record in payload["patterns"]:
            n_value, m_value = [int(item) for item in record["pattern"].split(":")]
            total_weights = 0
            grouped_weights = 0
            groups = 0
            total_l1 = 0.0
            removed_l1 = 0.0
            total_l2 = 0.0
            removed_l2 = 0.0
            for _, tensor in tensors[group]:
                values = tensor.detach().float().abs().reshape(-1)
                usable = values.numel() // m_value * m_value
                total_weights += int(values.numel())
                grouped_weights += int(usable)
                if usable == 0:
                    continue
                block = values[:usable].reshape(-1, m_value)
                ordered = torch.sort(block, dim=1).values
                removed = ordered[:, :m_value - n_value]
                groups += int(block.shape[0])
                total_l1 += float(block.sum())
                removed_l1 += float(removed.sum())
                total_l2 += float(block.square().sum())
                removed_l2 += float(removed.square().sum())
            l1_fraction = removed_l1 / total_l1
            l2_fraction = removed_l2 / total_l2
            max_l1_delta = max(max_l1_delta, abs(
                l1_fraction - record["oracle_magnitude_removed_l1_fraction"]))
            max_l2_delta = max(max_l2_delta, abs(
                l2_fraction - record["oracle_magnitude_removed_l2_squared_fraction"]))
            require(total_weights == record["total_weights"] and
                    grouped_weights == record["grouped_weights"] and
                    groups == record["groups"], "population drift " + group + " " + record["pattern"])
            require(abs(l1_fraction - record["oracle_magnitude_removed_l1_fraction"]) < 1e-6,
                    "L1 recompute drift " + group + " " + record["pattern"])
            require(abs(l2_fraction - record["oracle_magnitude_removed_l2_squared_fraction"]) < 1e-6,
                    "L2 recompute drift " + group + " " + record["pattern"])
            require(record["selector_metadata_information_lower_bound_bits_per_group"] ==
                    int(math.ceil(math.log(float(math.comb(m_value, n_value)), 2.0))),
                    "metadata lower bound drift")
    checks.extend(["tensor_membership", "independent_sort_l1", "independent_sort_l2",
                   "population_counts", "metadata_lower_bounds"])

    four_by_eight = [next(row for row in result["categories"][group]["patterns"]
                          if row["pattern"] == "4:8")["oracle_magnitude_removed_l1_fraction"]
                     for group in result["categories"]]
    eight_by_sixteen = [next(row for row in result["categories"][group]["patterns"]
                             if row["pattern"] == "8:16")["oracle_magnitude_removed_l1_fraction"]
                        for group in result["categories"]]
    require(result["aggregate_observations"]["four_of_eight_l1_removed_range"] ==
            [min(four_by_eight), max(four_by_eight)] and
            result["aggregate_observations"]["eight_of_sixteen_l1_removed_range"] ==
            [min(eight_by_sixteen), max(eight_by_sixteen)], "aggregate range drift")
    checks.append("aggregate_ranges")

    boundary = result["claim_boundary"]
    require(boundary == {
        "accuracy": False, "aee": False, "cycles": False, "energy": False,
        "paper_headline": False, "rtl": False, "speedup": False,
        "static_weight_mass": True, "structured_mask_opportunity": True,
        "traffic": False,
    }, "claim boundary drift")
    require(result["decision"]["lossless_nm"] == "KILL" and
            result["decision"]["direct_post_training_magnitude_mask"] ==
            "NO_GO_WITHOUT_PAIRED_AEE" and
            result["decision"]["hardware_aware_nm_retraining"] == "CANDIDATE_POOL_ONLY",
            "decision boundary drift")
    report_text = REPORT.read_text(encoding="utf-8")
    require("not accuracy, cycle, traffic, energy or RTL results" in report_text and
            "Only a new hardware-aware retrained checkpoint" in report_text,
            "human report drops claim boundary")
    checks.extend(["claim_boundary", "training_gate", "human_report_boundary"])

    patch_tensors = tensors["patch_embed"]
    neuron_weights = sum(int(tensor.numel()) for name, tensor in patch_tensors
                         if ".spiking_neuron.weight" in name)
    patch_weights = sum(int(tensor.numel()) for _, tensor in patch_tensors)
    crossing = {}
    for m_value in (4, 8, 16, 32):
        crossings = 0
        total_groups = 0
        for _, tensor in patch_tensors:
            row_weights = int(tensor[0].numel())
            tensor_weights = int(tensor.numel())
            total_groups += tensor_weights // m_value
            crossings += sum(1 for boundary in range(row_weights, tensor_weights, row_weights)
                             if boundary % m_value != 0)
        crossing[str(m_value)] = {
            "groups_crossing_storage_row": crossings,
            "total_flat_groups": total_groups,
            "fraction": float(crossings) / float(total_groups),
        }
    require(neuron_weights == 600 and patch_weights == 466872,
            "patch-category contamination diagnostic drift")
    require(crossing["8"]["groups_crossing_storage_row"] == 78 and
            crossing["16"]["groups_crossing_storage_row"] == 90,
            "row-crossing diagnostic drift")
    checks.extend(["patch_neuron_contamination_measured", "row_crossing_measured"])

    output = {
        "status": "PASS_M1538_STATIC_STORAGE_ORDER_OPPORTUNITY_WITH_SCOPE__NO_CYCLE_OR_HARDWARE_ADMISSION",
        "checks_passed": len(checks),
        "checks": checks,
        "max_independent_l1_fraction_delta": max_l1_delta,
        "max_independent_l2_squared_fraction_delta": max_l2_delta,
        "patch_category_diagnostics": {
            "total_weights": patch_weights,
            "spiking_neuron_temporal_matrix_weights": neuron_weights,
            "contamination_fraction": float(neuron_weights) / float(patch_weights),
            "flat_groups_crossing_storage_rows": crossing,
        },
        "claim_boundary": {
            "static_storage_order_weight_mass": True,
            "standard_executable_nm_layout": False,
            "accuracy": False,
            "aee": False,
            "cycles": False,
            "speedup": False,
            "traffic": False,
            "energy": False,
            "rtl": False,
            "training_authorized_by_this_review": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the complete H67 FFN subgraph scope and its accounted work.

The existing FFN ledger intentionally counts only the two Linear operators.
This audit joins the production MS-spiking source topology to the ten-sample
ordered trace and adds the two FFN-local ATLIF services without double
counting the global ATLIF bucket.  BN/residual are enumerated but remain
outside the current compute-cycle denominator.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PATHS = {
    "swin_source": ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py",
    "model_source": ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py",
    "config": ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
    "execution_trace": HW / "results/h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv",
    "atlif_activity": HW / "results/h67_ep35_full_network_ordered_trace_s10_20260821/atlif_activity.csv",
    "ffn_review": HW / "results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/motion_ffn_resident_fusion_opportunity_review.json",
    "m156": HW / "results/m156_h67_ep35_nonconv_group_sparsity_census_r1_20260824/m156_h67_ep35_nonconv_group_sparsity_census.json",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "swin_source": "8d551eac17e2272813dd0238ea4aee2e84cfe0b7e9435a1407d0787fb5407768",
    "model_source": "b8d969f9b91c292197dbe47c7b9a11803f10b7c604daaf911ed4bb5d00999b71",
    "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    "execution_trace": "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd",
    "atlif_activity": "c40c568635b759e433b816f74c472a79c6080250540f65495e8bb57468e2e1ad",
    "ffn_review": "5be91af5155162f1e12a9192cccd9d2e94a2ae18d9b3370ce625b867f5706811",
    "m156": "97d5ca1469c21e35b4019f24c5fc9dff8da1f1aae8a0c838defe4688903ae736",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
LANES = 96
SAMPLES = 10
TOTAL_ENVELOPE_CYCLES = 620302905


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             RuntimeError("non-finite JSON: " + value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite M159 output")
    script_start = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M159 frozen input identity drift")

    swin_text = PATHS["swin_source"].read_text(encoding="utf-8")
    model_text = PATHS["model_source"].read_text(encoding="utf-8")
    required_source_fragments = [
        "class MS_Spiking_Mlp(Spiking_Mlp):",
        "mlp_module = MS_Spiking_Mlp",
        "x = self.sn1(x)",
        "x = self.fc1(x)",
        "x= self.bn1(",
        "x = self.sn2(x)",
        "x = self.fc2(x)",
        "x = self.bn2(",
        "x = self.sew_function(self.mlp(",
    ]
    for fragment in required_source_fragments:
        require(fragment in swin_text, "missing topology fragment: " + fragment)
    require(
        'self.spikformer_norm = stt_kwargs["norm"] if "norm" in stt_kwargs else unet_kwargs["spiking_neuron"]["spike_norm"]'
        in model_text, "spikformer norm resolution drift")
    require("drop_rate=0." in model_text, "encoder dropout policy drift")

    config = yaml.safe_load(PATHS["config"].read_text(encoding="utf-8"))
    require(config["model"]["name"] == "MS_SpikingformerFlowNet_en4",
            "M159 model class drift")
    require(config["spiking_neuron"]["spike_norm"] == "BN",
            "M159 spike norm drift")
    require("norm" not in config["swin_transformer"],
            "M159 explicit Swin norm override drift")
    require(float(config["swin_transformer"]["mlp_ratio"]) == 4.0,
            "M159 MLP expansion drift")

    rows_by_sample = defaultdict(list)
    with PATHS["execution_trace"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_sample[int(row["sample_id"])].append(row)
    require(sorted(rows_by_sample) == list(range(SAMPLES)),
            "M159 sample extent drift")

    counts = Counter()
    sums = Counter()
    stage_sums = defaultdict(Counter)
    prefixes = set()
    suffixes = (
        ("sn1", ".sn1.spiking_neuron", "atlif"),
        ("fc1", ".fc1", "operator"),
        ("sn2", ".sn2.spiking_neuron", "atlif"),
        ("fc2", ".fc2", "operator"),
    )
    for sample_id, rows in sorted(rows_by_sample.items()):
        ffn_rows = [row for row in rows if ".mlp." in row["name"]]
        require(len(ffn_rows) == 48,
                "M159 expected 12 four-record FFN trace groups")
        for index in range(0, len(ffn_rows), 4):
            group = ffn_rows[index:index + 4]
            prefix = group[0]["name"].split(".sn1.spiking_neuron")[0]
            require(prefix.endswith(".mlp"), "M159 FFN prefix drift")
            stage = int(prefix.split(".layers.")[1].split(".")[0])
            prefixes.add(prefix)
            for row, (label, suffix, kind) in zip(group, suffixes):
                require(row["name"] == prefix + suffix,
                        "M159 trace topology/order drift")
                require(row["kind"] == kind,
                        "M159 trace kind drift")
                counts[label] += 1
                sums[label + "_input_elements"] += int(row["input_elements"])
                sums[label + "_output_elements"] += int(row["output_elements"])
                if label.startswith("sn"):
                    sums[label + "_dense_macs"] += int(row["dense_macs"])
                    stage_sums[stage][label + "_dense_macs"] += int(
                        row["dense_macs"])
            require(group[0]["input_shape"] == group[1]["input_shape"],
                    "sn1/fc1 shape join drift")
            require(group[1]["output_shape"] == group[2]["input_shape"],
                    "fc1/sn2 shape join drift")
            require(group[2]["output_shape"] == group[3]["input_shape"],
                    "sn2/fc2 shape join drift")
            require(group[3]["output_shape"] == group[0]["input_shape"],
                    "fc2/residual shape join drift")

    require(len(prefixes) == 12, "M159 unique FFN pair drift")
    require(all(counts[label] == 120 for label, _, _ in suffixes),
            "M159 per-node dynamic count drift")
    ffn_review = strict_json(PATHS["ffn_review"])
    m156 = strict_json(PATHS["m156"])
    linear_cycles = int(ffn_review["ffn_totals"]["cycles_model"])
    sn1_cycles = sums["sn1_dense_macs"] // SAMPLES // LANES
    sn2_cycles = sums["sn2_dense_macs"] // SAMPLES // LANES
    require(sn1_cycles == 9120000 and sn2_cycles == 36480000,
            "M159 FFN-local ATLIF issue-cycle drift")
    accounted_cycles = linear_cycles + sn1_cycles + sn2_cycles
    bn1_elements = sums["fc1_output_elements"] // SAMPLES
    bn2_elements = sums["fc2_output_elements"] // SAMPLES
    residual_elements = bn2_elements
    require((bn1_elements, bn2_elements, residual_elements)
            == (350208000, 87552000, 87552000),
            "M159 BN/residual element extent drift")
    require(m156["ffn"]["exact_summary"]["float_exact_zero_groups"] == 0
            and m156["ffn"]["exact_summary"]["int8_exact_zero_groups"] == 0,
            "M159 inherited structured-zero result drift")

    stage_rows = []
    for stage in range(4):
        ledger = ffn_review["stage_ledger"][str(stage)]
        stage_sn1 = stage_sums[stage]["sn1_dense_macs"] // SAMPLES // LANES
        stage_sn2 = stage_sums[stage]["sn2_dense_macs"] // SAMPLES // LANES
        stage_rows.append({
            "stage": stage,
            "pairs": int(ledger["pairs"]),
            "channels": int(ledger["input_channels"]),
            "expanded_channels": int(ledger["expanded_channels"]),
            "linear_cycles_model": int(ledger["pair_cycles_model"]),
            "sn1_atlif_issue_cycles": stage_sn1,
            "sn2_atlif_issue_cycles": stage_sn2,
            "accounted_full_subgraph_cycles": int(ledger["pair_cycles_model"])
            + stage_sn1 + stage_sn2,
        })
    require(sum(row["accounted_full_subgraph_cycles"] for row in stage_rows)
            == accounted_cycles, "M159 stage cycle sum drift")

    payload = {
        "schema": "m159_h67_full_ffn_subgraph_scope_v1",
        "status": "PASS_FULL_FFN_SUBGRAPH_SCOPE_AND_ACCOUNTED_COMPUTE_PARTITION",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "inputs_sha256": observed,
        },
        "resolved_topology": {
            "model": "MS_SpikingformerFlowNet_en4",
            "pairs": 12,
            "dynamic_groups_s10": 120,
            "trace_visible_order": ["sn1_atlif", "fc1_linear", "sn2_atlif", "fc2_linear"],
            "source_complete_order": [
                "sn1_atlif", "dropout1_p0", "fc1_linear", "bn1",
                "sn2_atlif", "dropout2_p0", "fc2_linear", "bn2",
                "drop_path_eval_off", "residual_add"
            ],
            "spike_norm": "BN_with_running_stats_by_source_default",
            "mlp_expansion": 4,
            "dropout_probability": 0.0,
            "qualification": "BN/dropout/residual are source-topology evidence; the ordered trace hooks ATLIF and Linear only."
        },
        "accounted_compute_cycles_per_frame": {
            "fc1_plus_fc2": linear_cycles,
            "sn1_atlif": sn1_cycles,
            "sn2_atlif": sn2_cycles,
            "full_ffn_subgraph_excluding_bn_residual": accounted_cycles,
            "share_of_current_compute_envelope": accounted_cycles / TOTAL_ENVELOPE_CYCLES,
            "perfect_removal_amdahl_ceiling_not_design_speedup":
                TOTAL_ENVELOPE_CYCLES / (TOTAL_ENVELOPE_CYCLES - accounted_cycles),
            "global_envelope_cycles": TOTAL_ENVELOPE_CYCLES,
            "partition_rule": "The 45.6M FFN-local ATLIF cycles are moved conceptually from the global ATLIF bucket; they are not added to 620.3M."
        },
        "unmodeled_elementwise_extent_per_frame": {
            "bn1_elements": bn1_elements,
            "bn2_elements": bn2_elements,
            "residual_add_elements": residual_elements,
            "bn_plus_residual_96lane_row_lower_bound":
                (bn1_elements + bn2_elements + residual_elements) // LANES,
            "cycles_admitted": False,
            "reason": "No BN affine/read/write/overlap or residual-port recurrence exists in the current envelope."
        },
        "stage_rows": stage_rows,
        "hardware_algorithm_feedback": {
            "correct_paired_prune_unit": [
                "fc1 output rows", "bn1 expanded channels",
                "sn2 temporal parameters/state channels", "fc2 matching input columns"
            ],
            "bn2_rule": "BN2 remains on unpruned output channels and may be folded only after checkpoint/eval numeric proof.",
            "tile_skip_rule": "A zero sn1 group alone cannot skip the whole branch until BN1 zero-preservation and sn2 state/output-zero behavior are proved.",
            "shared_hardware": "sn1/sn2 use the ATLIF engine; fc1/fc2 use one time-shared Linear engine; no dedicated duplicate arithmetic pool.",
            "frozen_checkpoint_exact_zero_groups_16_32": 0,
            "training_required": True,
        },
        "admission": {
            "complete_ffn_topology_scope": True,
            "trace_visible_order_exact_s10": True,
            "linear_plus_atlif_compute_partition": True,
            "bn_residual_cycles": False,
            "paired_prune_hardware_speedup": False,
            "full_ffn_rtl": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "paper_safe_statement":
            "H67 contains 12 MS-spiking FFN blocks with source topology sn1-Linear-BN1-sn2-Linear-BN2-residual. The two Linear layers account for 159.784M modeled cycles and the two FFN-local ATLIF services for another 45.600M within the existing global envelope, so the accounted subgraph is 33.1103% before unmodeled BN/residual service. No pruning or speedup is admitted."
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "M159 analyzer changed during execution")
    output.mkdir(parents=True, exist_ok=False)
    (output / "m159_h67_full_ffn_subgraph_scope.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print("PASS M159 pairs=12 dynamic_groups=120 linear_cycles={} "
          "ffn_atlif_cycles={} full_accounted={} share={:.9f} "
          "bn_residual_cycles=false speedup=false headline=false".format(
              linear_cycles, sn1_cycles + sn2_cycles, accounted_cycles,
              accounted_cycles / TOTAL_ENVELOPE_CYCLES))


if __name__ == "__main__":
    main()

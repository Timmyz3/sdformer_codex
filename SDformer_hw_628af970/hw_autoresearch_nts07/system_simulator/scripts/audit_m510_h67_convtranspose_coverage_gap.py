#!/usr/bin/env python3
"""Recompute the H67 ConvTranspose2d coverage gap from frozen S100 evidence."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml


HW = Path(__file__).resolve().parents[2]
ROOT = HW.parent
PROFILE = (
    HW / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
    "h67_ep35_system_trace_handoff_20260821/profile100"
)
FROZEN = {
    "profiler": (
        ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
        "profile_nts11_hardware_p0.py",
        "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    ),
    "config": (
        ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
        "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
        "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    ),
    "snn_models": (
        ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py",
        "8056a7dbdf34653c5a35401feb6815a89fe528bbed7e18f1d93f8eaa9360853e",
    ),
    "spiking_stswinnet": (
        ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py",
        "b8d969f9b91c292197dbe47c7b9a11803f10b7c604daaf911ed4bb5d00999b71",
    ),
    "spiking_modules": (
        ROOT / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py",
        "130e32bba2c0bc23a4da091087d09b71cb50e9381ed2d1c25ff6b447bce52948",
    ),
    "operator_runtime": (
        PROFILE / "operator_runtime.csv",
        "9cb5ccfc15b83c680ca8c96a816df1cdd4b5c4d956bd5c2462175b175b1b6c85",
    ),
    "atlif_activity": (
        PROFILE / "atlif_activity.csv",
        "ba9053080c964d17645d0d21d5cb47bfc85c9e962050895ba05c7bf0ddee344b",
    ),
    "activation_records": (
        PROFILE / "activation_records.csv",
        "ce079fb40737bdf33f7328e919351e7cdb0f8358eef097dc8c4dbb66665063ee",
    ),
    "old_envelope": (
        HW / "results/m221_motion_layer_islands_unified_coexistence_r1_20260825/"
        "m221_motion_layer_islands_unified_coexistence_r1.json",
        "32f570354a8d9bf1a6755368c7935f04f8cb7adf0bc6f7a3833e6da119bf4565",
    ),
    "docs359": (
        HW / "docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    ),
}


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
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-standard JSON token: " + token)))


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tight_active_product_bounds(active, height, width, channels_in,
                                channels_out, temporal_steps, samples):
    # ConvTranspose2d maps oy = 2*y - 1 + ky for K3/S2/P1/output_padding1.
    # Therefore only y=0 (and x=0) loses a tap; bottom/right are not clipped.
    spatial_categories = (
        (4, 1),
        (6, (height - 1) + (width - 1)),
        (9, (height - 1) * (width - 1)),
    )
    capacity = [(taps, sites * temporal_steps * channels_in * samples)
                for taps, sites in spatial_categories]

    def fill(order):
        remaining = active
        products = 0
        for index in order:
            taps, entries = capacity[index]
            used = min(remaining, entries)
            products += used * taps * channels_out
            remaining -= used
        require(remaining == 0, "active count exceeds tensor capacity")
        return products

    return fill((0, 1, 2)), fill((2, 1, 0))


def write_seal(directory, members):
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / name), name)
        for name in sorted(members)), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(seal)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    contract_path = args.contract.resolve()
    require(output.parent.is_dir() and not output.exists(),
            "M510 output must be a new child of an existing directory")
    script_start = sha256(Path(__file__).resolve())
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m510_h67_convtranspose_coverage_gap_contract_v2" and
            contract.get("status") ==
            "LOCKED_R2_MS_TOPOLOGY_STATIC_PREFLIGHT_REQUIRED_BEFORE_ONE_SHOT_AUDIT",
            "M510 contract identity drift")
    require(contract["inputs"]["analyzer"]["sha256"] == script_start,
            "M510 contract does not pin running analyzer")
    require(set(contract["inputs"]) == set(FROZEN) | {"analyzer"},
            "M510 contract input population drift")
    analyzer_contract_path = (ROOT / contract["inputs"]["analyzer"]["path"]).resolve()
    require(analyzer_contract_path == Path(__file__).resolve(),
            "M510 contract analyzer path drift")
    require((HW / contract["output"]["canonical_directory"]).resolve() == output,
            "M510 non-canonical output directory")

    observed = {}
    for name, (path, expected) in FROZEN.items():
        entry = contract["inputs"][name]
        require((ROOT / entry["path"]).resolve() == path.resolve() and
                entry["sha256"] == expected,
                "M510 contract frozen input drift: " + name)
        require(path.is_file() and sha256(path) == expected,
                "M510 frozen input drift: " + name)
        observed[name] = {
            "path": str(path),
            "sha256": expected,
        }

    profiler_text = FROZEN["profiler"][0].read_text(encoding="utf-8")
    hook_token = (
        "isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, "
        "torch.nn.Conv3d))"
    )
    require(profiler_text.count(hook_token) == 1 and
            "torch.nn.ConvTranspose2d" not in profiler_text,
            "M510 profiler hook coverage premise drift")
    config = yaml.safe_load(FROZEN["config"][0].read_text(encoding="utf-8"))
    require(config["model"]["use_upsample_conv"] is False and
            config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
            int(config["model"]["kernel_size"]) == 3 and
            int(config["model"]["base_num_channels"]) == 96,
            "M510 H67 decoder config drift")
    snn_text = FROZEN["snn_models"][0].read_text(encoding="utf-8")
    model_text = FROZEN["spiking_stswinnet"][0].read_text(encoding="utf-8")
    module_text = FROZEN["spiking_modules"][0].read_text(encoding="utf-8")
    require(all(token in snn_text for token in (
        "if use_upsample_conv:",
        "self.UpsampleLayer = self.upsample_type",
        "self.UpsampleLayer = self.transpose_type",
    )), "M510 decoder selection source drift")
    require(all(token in model_text for token in (
        "class MS_Spikingformer_MultiResUNet",
        "transpose_type = MS_SpikingTransposeDecoderLayer",
        "class MS_SpikingformerFlowNet_en4",
        "unet_type = MS_Spikingformer_MultiResUNet",
        "num_en = 4",
    )), "M510 H67 MS model topology source drift")
    require(all(token in module_text for token in (
        "class SpikingTransposeDecoderLayer", "layer.ConvTranspose2d(",
        "stride=2", "padding=padding", "output_padding=1",
    )), "M510 ConvTranspose2d construction source drift")
    ms_start = module_text.index(
        "class MS_SpikingTransposeDecoderLayer(SpikingTransposeDecoderLayer):")
    ms_end = module_text.index("class MS_SpikingSepTransposeDecoderLayer", ms_start)
    ms_forward = module_text[ms_start:ms_end]
    require(ms_forward.index("x = self.sn(x)") <
            ms_forward.index("x = self.deconv(x)"),
            "M510 MS decoder is no longer ATLIF-before-ConvTranspose2d")

    operators = read_csv(FROZEN["operator_runtime"][0])
    operator_counts = Counter(row["operator"] for row in operators)
    require(len(operators) == 79 and operator_counts ==
            {"Linear": 63, "Conv2d": 16} and
            not any("deconv" in row["name"].lower() for row in operators),
            "M510 frozen operator population drift")

    atlif_rows = [row for row in read_csv(FROZEN["atlif_activity"][0])
                  if ".decoders." in row["name"]]
    require(len(atlif_rows) == 4, "M510 decoder ATLIF population drift")
    atlif_by_index = {}
    for row in atlif_rows:
        index = int(row["name"].split(".decoders.", 1)[1].split(".", 1)[0])
        require(index not in atlif_by_index and int(row["calls"]) == 100 and
                row["output_mode"] == "binary" and
                row["deployment_dead_result"] == "False",
                "M510 decoder ATLIF identity drift")
        atlif_by_index[index] = row
    require(set(atlif_by_index) == set(range(4)),
            "M510 decoder ATLIF index drift")

    decoder_shapes = defaultdict(set)
    decoder_shape_counts = Counter()
    for row in read_csv(FROZEN["activation_records"][0]):
        if row["kind"] != "decoder":
            continue
        index = int(row["name"].removeprefix("decoder"))
        decoder_shapes[index].add(tuple(json.loads(row["shape"])))
        decoder_shape_counts[index] += 1
    require(set(decoder_shapes) == set(range(4)) and
            all(len(decoder_shapes[index]) == 1 and
                decoder_shape_counts[index] == 100 for index in range(4)),
            "M510 decoder output shape population drift")

    old = strict_json(FROZEN["old_envelope"][0])
    old_envelope = int(old["frozen_h67_compute_envelope"]["cycles_per_frame"])
    qualification = old["frozen_h67_compute_envelope"]["qualification"]
    lanes = 96
    require(old_envelope == 620302905 and qualification.startswith("96-lane"),
            "M510 old envelope identity drift")

    layers = []
    total_dense_products = 0
    total_lower_products = 0
    total_upper_products = 0
    for index in range(4):
        row = atlif_by_index[index]
        input_shape = json.loads(row["input_first_shape"])
        output_shape = next(iter(decoder_shapes[index]))
        require(len(input_shape) == 5 and len(output_shape) == 5 and
                input_shape[0:2] == list(output_shape[0:2]) and
                output_shape[3] == 2 * input_shape[3] and
                output_shape[4] == 2 * input_shape[4],
                "M510 decoder input/output shape drift")
        temporal, batch, channels_in, height, width = map(int, input_shape)
        channels_out = int(output_shape[2])
        require(temporal == 10 and batch == 1,
                "M510 temporal/batch shape drift")
        active = int(row["active"])
        elements = int(row["elements"])
        require(elements == temporal * batch * channels_in * height * width * 100,
                "M510 decoder ATLIF element population drift")
        lower_s100, upper_s100 = tight_active_product_bounds(
            active, height, width, channels_in, channels_out, temporal, 100)
        dense = ((3 * height - 1) * (3 * width - 1) * channels_in *
                 channels_out * temporal)
        total_dense_products += dense
        total_lower_products += lower_s100
        total_upper_products += upper_s100
        layers.append({
            "decoder": index,
            "input_shape": input_shape,
            "output_shape": list(output_shape),
            "channels_in": channels_in,
            "channels_out": channels_out,
            "activity": active / elements,
            "active_s100": active,
            "dense_products_per_frame_exact": dense,
            "active_products_s100_tight_lower": lower_s100,
            "active_products_s100_tight_upper": upper_s100,
            "active_products_per_frame_mean_lower": lower_s100 / 100,
            "active_products_per_frame_mean_upper": upper_s100 / 100,
            "ideal_96lane_cycles_per_frame_mean_lower":
                lower_s100 / (100 * lanes),
            "ideal_96lane_cycles_per_frame_mean_upper":
                upper_s100 / (100 * lanes),
        })

    lower_cycles = total_lower_products / (100 * lanes)
    upper_cycles = total_upper_products / (100 * lanes)
    corrected_lower = old_envelope + lower_cycles
    corrected_upper = old_envelope + upper_cycles
    require((total_dense_products, total_lower_products, total_upper_products,
             ) ==
            (78848509440, 1637926293504, 1761318549504),
            "M510 independently recomputed totals drift")
    require(sha256(Path(__file__).resolve()) == script_start,
            "M510 analyzer mutated during execution")

    result = {
        "schema": "m510_h67_convtranspose_coverage_gap_audit_v1",
        "status": "PASS_CONFIRMED_OMITTED_CONVTRANSPOSE__TRACE_REQUIRED_BEFORE_RTL",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "contract_sha256": sha256(contract_path),
            "inputs": observed,
        },
        "coverage": {
            "operator_rows": len(operators),
            "operator_type_counts": dict(operator_counts),
            "convtranspose_operator_rows": 0,
            "decoder_atlif_rows": len(atlif_rows),
            "decoder_activation_records": sum(decoder_shape_counts.values()),
            "old_envelope_cycles": old_envelope,
            "old_envelope_safe_label": "included_scope_96lane_activity_weighted",
            "full_network_label_admitted": False,
        },
        "layers": layers,
        "analytic_bounds": {
            "dense_products_per_frame_exact": total_dense_products,
            "active_products_s100_tight_lower": total_lower_products,
            "active_products_s100_tight_upper": total_upper_products,
            "active_products_per_frame_mean_lower": total_lower_products / 100,
            "active_products_per_frame_mean_upper": total_upper_products / 100,
            "ideal_96lane_cycles_per_frame_mean_lower": lower_cycles,
            "ideal_96lane_cycles_per_frame_mean_upper": upper_cycles,
            "corrected_envelope_lower": corrected_lower,
            "corrected_envelope_upper": corrected_upper,
            "decoder_share_lower": lower_cycles / corrected_lower,
            "decoder_share_upper": upper_cycles / corrected_upper,
            "decoder_free_ceiling_lower": corrected_lower / old_envelope,
            "decoder_free_ceiling_upper": corrected_upper / old_envelope,
            "dense_over_sparse_opportunity_lower":
                (total_dense_products / lanes) / upper_cycles,
            "dense_over_sparse_opportunity_upper":
                (total_dense_products / lanes) / lower_cycles,
        },
        "decision": {
            "verdict": "CONDITIONAL_GO__EXACT_TRACE_AND_A1_FASTKILL_BEFORE_RTL",
            "next": "Capture exact S10 binary ConvTranspose2d input bitmaps and run A0/A1/EPD same-resource cycle screening.",
            "rtl_authorized": False,
        },
        "claim_boundary": {
            "coverage_gap_confirmed": True,
            "aggregate_count_tight_bounds": True,
            "per_sample_coordinate_bound": False,
            "exact_coordinate_product_count": False,
            "cycle_simulator": False,
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.",
                                    dir=str(output.parent)))
    report = staging / "m510_h67_convtranspose_coverage_gap_audit.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    (staging / "RUN_COMPLETE.txt").write_text(
        "PASS_M510_CONFIRMED_CONVTRANSPOSE_COVERAGE_GAP\n", encoding="utf-8")
    write_seal(staging, [report.name, "RUN_COMPLETE.txt"])
    require(not output.exists(), "M510 output appeared during staging")
    os.replace(staging, output)
    for line in (output / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(sha256(output / name) == expected,
                "M510 final member rehash failed")
    expected, name = (output / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(output / name) == expected,
            "M510 final outer seal failed")
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "corrected_envelope": [corrected_lower, corrected_upper],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact same-BN PAFT/control replay through the frozen M251 cycle model."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def popcount(value):
    return bin(int(value)).count("1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = load_json(args.contract)
    require(contract.get("schema") ==
            "m254_paft_vs_control_running_bn_pwp_cycle_contract_v1",
            "contract schema drift")
    root = args.contract.resolve().parents[1]
    source_start = sha256(Path(__file__).resolve())
    identities = {}
    paths = {}
    for name, spec in contract["inputs"].items():
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"],
                "SHA drift for {}: {}".format(name, observed))
        paths[name] = path
        identities[name] = {"path": spec["path"], "sha256": observed}

    m251 = load_module("m254_frozen_m251", paths["m251_analyzer"])
    m43 = m251.load_module(paths["m43_support_unpacker"])
    paft_result = load_json(paths["m251_paft_cycle_result"])
    correction = load_json(paths["m251r2_range_correction"])
    catalog = load_json(paths["m77_train_only_catalog"])
    paft_trace = load_json(paths["m248_paft_running_trace"])
    control_trace = load_json(paths["m252_control_running_trace"])
    accuracy = load_json(paths["m247_paired_accuracy"])

    require(paft_result["status"] ==
            "PASS_PAFT_RUNNING_BN_TRAIN_CATALOG_ISOLATED_CONV_CYCLE_MODEL" and
            correction["status"] ==
            "PASS_M251_FIXED12_RANGE_CORRECTED_CYCLES_UNCHANGED" and
            correction["corrected_full_signed_int8_pwp_range"]["sum_range"] ==
                [-2048, 2032],
            "M251/M251r2 admission drift")
    require(control_trace["status"] ==
            "PASS_CONTROL_EP4_RUNNING_BN_S10_FOUR_BOTTLENECK_EXACT_SOURCE_TRACE" and
            control_trace["identity"]["capture_bn_policy"] == "running" and
            control_trace["identity"]["paired_arm"] == "NO_PAFT_CONTROL",
            "M252 control trace admission drift")
    require(paft_trace["identity"]["capture_bn_policy"] == "running" and
            paft_trace["cohort"] == control_trace["cohort"],
            "paired cohort/operator geometry drift")
    require(paft_trace["identity"]["dataset_input_files"] ==
            control_trace["identity"]["dataset_input_files"],
            "paired dataset input identity drift")
    require(accuracy["paired_training_audit"]
            ["non_paft_config_fields_exactly_equal"] is True and
            accuracy["hardware_decision"]["primary_policy"] == "running",
            "M247 paired config/BN audit drift")

    geometry = {
        "samples": 10,
        "operators": 4,
        "rows_per_operator": 3000,
        "partition_bits": 16,
        "partitions_per_operator": 432,
        "patterns_per_partition": 16,
        "output_blocks": 8
    }
    cycle_model = {
        "weight_vector_bytes": 96,
        "fixed_pwp_vector_bytes": 144,
        "dram_bytes_per_cycle": 32,
        "matcher_pipeline_cycles": 16,
        "packer_lanes": 8,
        "packer_pipeline_cycles": 4,
        "compute_tail_cycles_per_partition": 2
    }
    ports = [
        {"name": "WIDE144_PWP_96_WEIGHT",
         "weight_vector_service_cycles": 1,
         "pwp_vector_service_cycles": 1},
        {"name": "SHARED96",
         "weight_vector_service_cycles": 1,
         "pwp_vector_service_cycles": 2}
    ]
    operator_names = control_trace["cohort"]["operators"]
    require([row["operator"] for row in catalog["operators"]] == operator_names,
            "catalog/control operator order drift")

    trace_dir = paths["m252_control_running_trace"].parent
    op_index = {name: index for index, name in enumerate(operator_names)}
    histograms = defaultdict(Counter)
    record_audit = []
    for record_index, record in enumerate(control_trace["records"]):
        packed = trace_dir / record["packed_file"]
        require(packed.is_file() and sha256(packed) ==
                record["packed_file_sha256"],
                "control packed payload drift")
        require(record["negative_count"] == 0,
                "control trace is not nonnegative")
        masks = m43.unpack_record_masks(trace_dir, record)
        expanded = 0
        for row in range(m43.ROWS):
            base = row * m43.TILES
            for tile in range(m43.TILES):
                value256 = masks[base + tile]
                partition_base = tile * (m43.TILE_BITS // 16)
                for subtile in range(m43.TILE_BITS // 16):
                    value = (value256 >> (16 * subtile)) & 0xffff
                    histograms[(record["sample_id"],
                                op_index[record["operator"]],
                                partition_base + subtile)][value] += 1
                    expanded += popcount(value)
        record_audit.append({
            "sample_id": record["sample_id"],
            "operator_index": record["operator_index"],
            "input_nonzero_count": record["nonzero_count"],
            "expanded_conv3x3_source_events": expanded
        })
        print("[M254 HIST] {}/40 sample={} op={} expanded={}".format(
            record_index + 1, record["sample_id"], record["operator_index"],
            expanded), flush=True)

    phases = defaultdict(list)
    aggregate = Counter()
    per_operator = [Counter() for _ in operator_names]
    for sample in range(10):
        for op in range(4):
            for partition, catalog_row in enumerate(
                    catalog["operators"][op]["partitions"]):
                require(catalog_row["partition"] == partition,
                        "catalog partition order drift")
                centers = [int(item["value_hex"], 16)
                           for item in catalog_row["patterns"]]
                phase = m251.phase_metrics(
                    histograms[(sample, op, partition)], centers)
                require(phase["partition_vectors"] == 3000,
                        "control phase row population drift")
                phases[sample].append(phase)
                clean = dict((key, value) for key, value in phase.items()
                             if not key.startswith("used_center_"))
                aggregate.update(clean)
                per_operator[op].update(clean)

    expected_vectors = 10 * 4 * 3000 * 432
    require(aggregate["partition_vectors"] == expected_vectors,
            "control partition-vector population drift")
    control_cycle_rows = []
    for port in ports:
        total = Counter()
        per_sample = []
        for sample in range(10):
            replay = m251.replay_sample(phases[sample], port, cycle_model,
                                        geometry)
            replay["sample_id"] = sample
            per_sample.append(replay)
            for key in ("dense_cycles", "bit_sparse_cycles", "candidate_cycles"):
                total[key] += replay[key]
            total.update(replay["binding_phases"])
        control_cycle_rows.append({
            "port": port["name"],
            "dense_cycles": total["dense_cycles"],
            "bit_sparse_cycles": total["bit_sparse_cycles"],
            "candidate_cycles": total["candidate_cycles"],
            "speedup_vs_dense": total["dense_cycles"] /
                                float(total["candidate_cycles"]),
            "speedup_vs_bit_sparse": total["bit_sparse_cycles"] /
                                     float(total["candidate_cycles"]),
            "binding_phases": {name: total[name]
                               for name in ("compute", "matcher", "packer", "dma")},
            "per_sample": per_sample
        })

    paft_natural = paft_result["exact_natural_work"]
    control_natural = {
        "bit_sparse_vector_ops_per_block":
            aggregate["bit_sparse_vector_ops_per_block"],
        "candidate_vector_ops_per_block":
            aggregate["candidate_vector_ops_per_block"],
        "pwp_ops_per_block": aggregate["pwp_ops_per_block"],
        "correction_ops_per_block": aggregate["correction_ops_per_block"],
        "natural_vector_op_speedup_vs_bit_sparse":
            aggregate["bit_sparse_vector_ops_per_block"] /
            float(aggregate["candidate_vector_ops_per_block"])
    }
    control_density = (control_natural["candidate_vector_ops_per_block"] /
                       float(control_natural["bit_sparse_vector_ops_per_block"]))
    activity_only_expected_paft_candidate = (
        paft_natural["bit_sparse_vector_ops_per_block"] * control_density)
    natural_comparison = {
        "control": control_natural,
        "paft": {
            key: paft_natural[key] for key in (
                "bit_sparse_vector_ops_per_block",
                "candidate_vector_ops_per_block",
                "pwp_ops_per_block",
                "correction_ops_per_block",
                "natural_vector_op_speedup_vs_bit_sparse")
        },
        "paft_bit_sparse_work_reduction_percent":
            100.0 * (1.0 - paft_natural["bit_sparse_vector_ops_per_block"] /
                     float(control_natural["bit_sparse_vector_ops_per_block"])),
        "paft_candidate_work_reduction_percent":
            100.0 * (1.0 - paft_natural["candidate_vector_ops_per_block"] /
                     float(control_natural["candidate_vector_ops_per_block"])),
        "paft_pattern_efficiency_improvement_percent":
            100.0 * (paft_natural["natural_vector_op_speedup_vs_bit_sparse"] /
                     control_natural["natural_vector_op_speedup_vs_bit_sparse"] - 1.0),
        "activity_only_expected_paft_candidate_vector_ops_per_block":
            activity_only_expected_paft_candidate,
        "additional_candidate_reduction_beyond_control_pattern_efficiency_percent":
            100.0 * (1.0 -
                     paft_natural["candidate_vector_ops_per_block"] /
                     activity_only_expected_paft_candidate)
    }

    paft_cycles = {row["port"]: row
                   for row in paft_result["same_resource_cycle_simulations"]}
    cycle_comparison = []
    for control in control_cycle_rows:
        paft = paft_cycles[control["port"]]
        require(control["dense_cycles"] == paft["dense_cycles"],
                "paired dense cycle baseline drift")
        per_sample = []
        for control_sample, paft_sample in zip(control["per_sample"],
                                               paft["per_sample"]):
            require(control_sample["sample_id"] == paft_sample["sample_id"],
                    "paired sample order drift")
            per_sample.append({
                "sample_id": control_sample["sample_id"],
                "control_candidate_cycles": control_sample["candidate_cycles"],
                "paft_candidate_cycles": paft_sample["candidate_cycles"],
                "control_over_paft_candidate_cycle_ratio":
                    control_sample["candidate_cycles"] /
                    float(paft_sample["candidate_cycles"])
            })
        cycle_comparison.append({
            "port": control["port"],
            "control": {key: control[key] for key in (
                "dense_cycles", "bit_sparse_cycles", "candidate_cycles",
                "speedup_vs_dense", "speedup_vs_bit_sparse", "binding_phases")},
            "paft": {key: paft[key] for key in (
                "dense_cycles", "bit_sparse_cycles", "candidate_cycles",
                "speedup_vs_dense", "speedup_vs_bit_sparse", "binding_phases")},
            "paft_candidate_cycle_reduction_percent":
                100.0 * (1.0 - paft["candidate_cycles"] /
                         float(control["candidate_cycles"])),
            "paft_throughput_gain_same_hardware":
                control["candidate_cycles"] / float(paft["candidate_cycles"]),
            "paft_bit_sparse_cycle_reduction_percent":
                100.0 * (1.0 - paft["bit_sparse_cycles"] /
                         float(control["bit_sparse_cycles"])),
            "paired_samples_paft_faster":
                sum(row["paft_candidate_cycles"] < row["control_candidate_cycles"]
                    for row in per_sample),
            "per_sample": per_sample
        })

    payload = {
        "schema": "m254_paft_vs_control_running_bn_pwp_cycle_v1",
        "status": "PASS_PAIRED_RUNNING_BN_PAFT_HARDWARE_GAIN_ISOLATED_CONV",
        "identity": identities,
        "paired_scope": {
            "bn_policy": "running",
            "samples": 10,
            "operators": operator_names,
            "catalog": "M77 DSEC-train-only k16/q16",
            "same_dataset_inputs": True,
            "same_cycle_model": True,
            "paired_valid825_aee_improvement_percent":
                accuracy["hardware_decision"][
                    "paft_running_aee_improvement_percent"]
        },
        "exact_natural_work_comparison": natural_comparison,
        "same_hardware_cycle_comparison": cycle_comparison,
        "control_record_expansion_audit": record_audit,
        "decision": {
            "paft_hardware_direction": "KEEP",
            "paper_role": "algorithm-hardware co-design ablation for the four bottleneck Conv module",
            "next_numeric_gate": "export PAFT ep4 INT8 weights, prove Acc19 or retain overflow guard, and build PWP payload identities",
            "next_rtl_gate": "close M241r2 variable-latency macro response protocol and replay a full96/full-trace VCS subset"
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": "Paired single-seed same-running-BN, same-cohort, same-catalog and same-resource isolated four-Conv cycle comparison. It isolates the PAFT checkpoint's trace-level hardware gain, but is not multi-seed significance, checkpoint INT8/Acc19 admission, RTL-integrated cycle equality, energy, system speedup, paper PPA or headline."
    }
    require(sha256(Path(__file__).resolve()) == source_start,
            "M254 analyzer changed during execution")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "m254_paft_vs_control_running_bn_pwp_cycle_r1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M254_PASS natural_control={:.6f} natural_paft={:.6f} wide_paft_gain={:.6f} shared96_paft_gain={:.6f}".format(
        control_natural["natural_vector_op_speedup_vs_bit_sparse"],
        paft_natural["natural_vector_op_speedup_vs_bit_sparse"],
        cycle_comparison[0]["paft_throughput_gain_same_hardware"],
        cycle_comparison[1]["paft_throughput_gain_same_hardware"]), flush=True)


if __name__ == "__main__":
    main()

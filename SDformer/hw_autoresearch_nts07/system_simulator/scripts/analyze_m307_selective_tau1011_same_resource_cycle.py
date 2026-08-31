#!/usr/bin/env python3
"""Replay the M306 [tau1,tau0,tau1,tau1] policy on M280 resources."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


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
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             RuntimeError("non-finite JSON: " + token)))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import pinned helper " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_work(target, work):
    for key, value in work.items():
        target[key] += int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m307_selective_tau1011_same_resource_cycle_contract_v1",
            "M307 contract schema drift")
    require(source_start == contract["analyzer"]["sha256"],
            "M307 analyzer SHA drift")
    hw = contract_path.parents[1]
    repo = hw.parent
    paths = {}
    identity = {
        "contract": {"path": str(contract_path.relative_to(hw)),
                     "sha256": sha256(contract_path)},
        "analyzer": {"path": str(source_path.relative_to(repo)),
                     "sha256": source_start},
    }
    for name, spec in contract["inputs"].items():
        base = repo if spec.get("relative_to") == "repo" else hw
        path = (base / spec["path"]).resolve()
        require(path.is_file(), "M307 missing input " + str(path))
        observed = sha256(path)
        require(observed == spec["sha256"], "M307 SHA drift: " + name)
        paths[name] = path
        identity[name] = {"path": spec["path"], "sha256": observed}

    helper = load_module(paths["m283_auditor"], "m307_pinned_m283")
    m251_contract = strict_json(paths["m251_contract"])
    trace = strict_json(paths["m248_trace"])
    catalog = strict_json(paths["m77_catalog"])
    m280 = strict_json(paths["m280_result"])
    receipt = strict_json(paths["m306_receipt"])
    require(trace["cohort"]["records"] == 40 and len(trace["records"]) == 40,
            "M307 trace record extent drift")
    require(catalog["split"]["train_catalog_eligible"] is True and
            catalog["split"]["test_or_validation_data_used"] is False and
            catalog["split"]["train_valid825_key_overlap"] == 0,
            "M307 catalog split drift")
    require(receipt["runtime"]["enabled_operator_indices"] == [0, 2, 3] and
            receipt["runtime"]["disabled_operator_indices"] == [1] and
            receipt["paired_baseline"]["accuracy_gate_pass"] is True and
            abs(float(receipt["paired_baseline"]
                             ["candidate_minus_baseline_aee"]) -
                0.014362) <= 1.0e-12,
            "M307 M306 accuracy/selection identity drift")

    policies = {
        "tau0000_exact_anchor": [0, 0, 0, 0],
        "tau1111_all_layer_attack": [1, 1, 1, 1],
        "tau1011_selected": [1, 0, 1, 1],
    }
    samples = 10
    operators = 4
    partitions = 432
    phases = dict((name, [[None] * (operators * partitions)
                         for _ in range(samples)]) for name in policies)
    aggregate = dict((name, Counter()) for name in policies)
    per_operator = dict((name, [Counter() for _ in range(operators)])
                        for name in policies)
    popcount = np.array([bin(value).count("1") for value in range(65536)],
                        dtype=np.uint8)
    record_by_key = dict(
        ((int(row["sample_id"]), int(row["operator_index"])), row)
        for row in trace["records"])
    require(len(record_by_key) == 40, "M307 duplicate trace identity")
    trace_dir = paths["m248_trace"].parent
    raw_payload_hashes_checked = 0
    expansion_audit = []

    for op in range(operators):
        packed_samples = []
        for sample in range(samples):
            record = record_by_key[(sample, op)]
            values, expanded = helper.unpack_conv_partitions(
                record, trace_dir, popcount)
            packed_samples.append(values)
            raw_payload_hashes_checked += 1
            expansion_audit.append({
                "sample_id": sample,
                "operator_index": op,
                "expanded_conv3x3_source_events": int(expanded),
            })
        operator_catalog = catalog["operators"][op]
        require(operator_catalog["operator"] ==
                trace["cohort"]["operators"][op],
                "M307 operator identity drift")
        for partition in range(partitions):
            row = operator_catalog["partitions"][partition]
            centers = np.sort(np.array(
                [int(item["value_hex"], 16) for item in row["patterns"]],
                dtype=np.uint16))
            require(int(row["partition"]) == partition and
                    np.unique(centers).size == 16 and
                    not np.any(centers == 0),
                    "M307 catalog partition drift")
            all_values = np.unique(np.concatenate(
                [packed_samples[sample][:, partition]
                 for sample in range(samples)]))
            populations = popcount[all_values].astype(np.int16)
            distances = popcount[np.bitwise_xor(
                all_values[:, None], centers[None, :])].min(axis=1).astype(np.int16)
            for sample in range(samples):
                values, counts = np.unique(
                    packed_samples[sample][:, partition], return_counts=True)
                counts = counts.astype(np.int64)
                indices = np.searchsorted(all_values, values)
                require(np.array_equal(all_values[indices], values),
                        "M307 sample/aggregate value lookup drift")
                for name, vector in policies.items():
                    work = helper.update_work(
                        values, counts, populations[indices], distances[indices],
                        int(vector[op]))
                    phase_index = op * partitions + partition
                    phases[name][sample][phase_index] = work
                    add_work(aggregate[name], work)
                    add_work(per_operator[name][op], work)

    model = m251_contract["same_resource_cycle_model"]
    cycle_model = {
        "weight_vector_bytes": int(model["weight_vector_bytes"]),
        "fixed_pwp_vector_bytes": int(model["fixed_pwp_vector_bytes"]),
        "patterns_per_partition": 16,
        "dram_bytes_per_cycle": int(model["dram_bytes_per_cycle"]),
        "matcher_pipeline_cycles": int(model["matcher_pipeline_cycles"]),
        "packer_lanes": int(model["packer_lanes"]),
        "packer_pipeline_cycles": int(model["packer_pipeline_cycles"]),
        "compute_tail_cycles_per_partition": int(
            model["compute_tail_cycles_per_partition"]),
    }
    ports = [(row["name"], int(row["pwp_vector_service_cycles"]))
             for row in model["ports"]]
    require(ports == [("WIDE144_PWP_96_WEIGHT", 1), ("SHARED96", 2)],
            "M307 port model drift")
    rows = {}
    for name, vector in policies.items():
        require(all(phase is not None for sample in phases[name]
                    for phase in sample), "M307 incomplete phase population")
        cycle_rows = []
        for port_name, pwp_service in ports:
            dense = sparse = candidate = 0
            bindings = Counter()
            for sample in range(samples):
                replay = helper.replay_sample(
                    phases[name][sample], pwp_service, cycle_model)
                dense += int(replay[0])
                sparse += int(replay[1])
                candidate += int(replay[2])
                bindings.update(replay[3])
            cycle_rows.append({
                "port": port_name,
                "pwp_vector_service_cycles": pwp_service,
                "dense_cycles": dense,
                "bit_sparse_cycles": sparse,
                "candidate_cycles": candidate,
                "speedup_vs_bit_sparse": sparse / float(candidate),
                "binding_phases": dict((key, int(bindings[key]))
                                       for key in
                                       ("compute", "matcher", "packer", "dma")),
            })
        work = dict((key, int(value)) for key, value in aggregate[name].items())
        rows[name] = {
            "per_operator_tau": vector,
            "work": work,
            "per_operator_work": [
                {"operator_index": op,
                 "operator": trace["cohort"]["operators"][op],
                 **dict((key, int(value))
                        for key, value in per_operator[name][op].items())}
                for op in range(operators)
            ],
            "same_resource_cycle_simulations": cycle_rows,
        }

    author_by_tau = dict((int(row["distance_threshold"]), row)
                         for row in m280["thresholds"])
    anchor_mismatches = []
    for name, tau in (("tau0000_exact_anchor", 0),
                      ("tau1111_all_layer_attack", 1)):
        author = author_by_tau[tau]
        for key in ("dense_vector_ops_per_block",
                    "bit_sparse_vector_ops_per_block",
                    "candidate_vector_ops_per_block", "pwp_ops_per_block",
                    "correction_ops_per_block",
                    "elided_correction_ops_per_block",
                    "approximated_partition_vectors"):
            expected = int(author["exact_work"].get(
                key, author["int8_accumulator_error"].get(key, 0)))
            observed = int(rows[name]["work"].get(key, 0))
            if observed != expected:
                anchor_mismatches.append({"anchor": name, "field": key,
                                          "observed": observed,
                                          "expected": expected})
        for observed, expected in zip(
                rows[name]["same_resource_cycle_simulations"],
                author["same_resource_cycle_simulations"]):
            for key in ("dense_cycles", "bit_sparse_cycles",
                        "candidate_cycles", "binding_phases"):
                if observed[key] != expected[key]:
                    anchor_mismatches.append({
                        "anchor": name, "field": observed["port"] + "." + key,
                        "observed": observed[key], "expected": expected[key]})
    require(not anchor_mismatches,
            "M307 all-zero/all-one anchors did not reproduce M280")

    selected = rows["tau1011_selected"]
    exact = rows["tau0000_exact_anchor"]
    for row in selected["same_resource_cycle_simulations"]:
        exact_row = next(value for value in
                         exact["same_resource_cycle_simulations"]
                         if value["port"] == row["port"])
        row["incremental_speedup_vs_tau0_exact"] = (
            exact_row["candidate_cycles"] / float(row["candidate_cycles"]))
    output = {
        "schema": "m307_selective_tau1011_same_resource_cycle_v1",
        "status": "PASS_EXACT_ANCHORS_AND_SELECTED_TRACE_CYCLE_OPPORTUNITY_ONLY",
        "identity": identity,
        "scope": {
            "checkpoint": "M87_PAFT_EP4",
            "bn_policy": "running",
            "samples": 10,
            "sequence": "zurich_city_09_a",
            "operators": trace["cohort"]["operators"],
            "selected_per_operator_tau": [1, 0, 1, 1],
            "raw_payload_hashes_checked": raw_payload_hashes_checked,
        },
        "anchor_mismatches": anchor_mismatches,
        "policies": rows,
        "selected_accuracy": {
            "baseline_aee": float(receipt["paired_baseline"]["aee"]),
            "candidate_aee": float(receipt["paired_baseline"]["candidate_aee"]),
            "candidate_minus_baseline_aee": float(
                receipt["paired_baseline"]["candidate_minus_baseline_aee"]),
            "s10_gate": float(
                receipt["paired_baseline"]["absolute_aee_increase_gate"]),
            "s10_gate_pass": bool(
                receipt["paired_baseline"]["accuracy_gate_pass"]),
            "valid825": False,
        },
        "count_semantics": {
            "approximated_partition_vectors":
                "distance>0 partitions whose residual is elided; this is the lossy population",
            "m306_runtime_total_snapped":
                "includes distance-zero exact pattern hits and must not be called the lossy population",
            "m306_runtime_total_snapped_observed": int(
                receipt["runtime"]["aggregate_snapped_partition_vectors"]),
        },
        "record_expansion_audit": expansion_audit,
        "admission": {
            "paired_s10_accuracy_candidate": True,
            "valid825_accuracy": False,
            "trace_cycle_opportunity": True,
            "executable_cycle_hardware": False,
            "rtl": False,
            "synopsys_dc": False,
            "paper_ppa_ready": False,
            "system_speedup": False,
            "headline": False,
        },
        "claim_boundary": "Exact frozen-trace same-resource replay for four isolated PAFT bottleneck Conv layers plus a paired running-BN S10 modified-forward screen. No valid825, executable SRAM/control RTL, Synopsys PPA, full-network or system speedup is admitted."
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "m307_selective_tau1011_same_resource_cycle_r1.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(source_path) == source_start,
            "M307 analyzer changed during execution")
    selected_ports = selected["same_resource_cycle_simulations"]
    print("M307_PASS wide={:.9f} shared={:.9f} delta_aee={:.9f}".format(
        selected_ports[0]["speedup_vs_bit_sparse"],
        selected_ports[1]["speedup_vs_bit_sparse"],
        output["selected_accuracy"]["candidate_minus_baseline_aee"]))


if __name__ == "__main__":
    main()

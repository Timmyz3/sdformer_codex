#!/usr/bin/env python3
"""Audit whether frozen assets can support token-level whole-FFN skip G8."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile


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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_inputs(contract_path, contract):
    root = contract_path.resolve().parents[1]
    paths = {}
    identities = {}
    for name, identity in contract["inputs"].items():
        if identity["path"].startswith("third_party/") or identity[
                "path"].startswith("neuron_experiments/"):
            path = root.parent / identity["path"]
        else:
            path = root / identity["path"]
        require(path.is_file(), "missing input {}: {}".format(name, path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift: " + name)
        paths[name] = path
        identities[name] = {"path": identity["path"], "sha256": observed}
    return root, paths, identities


def ordered_trace_audit(profile_path, execution_path, activation_path,
                        workload_path):
    profile = strict_json(profile_path)
    require(profile["ordered_trace"] is True and profile["samples"] == 10,
            "ordered S10 profile identity drift")
    require(profile.get("bit_trace_records") == 0 and
            profile.get("bit_trace_manifest") is None,
            "unexpected ordered bit payload")
    require(profile["artifact_identity"]["checkpoint_sha256"] ==
            "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "ordered checkpoint drift")
    require(profile["eval_protocol"]["bn_policy"] == "no_running",
            "ordered BN policy drift")

    with workload_path.open("r", encoding="utf-8", newline="") as handle:
        workload = list(csv.DictReader(handle))
    require(len(workload) == 10 and
            [int(row["sample_id"]) for row in workload] == list(range(10)),
            "ordered workload identity drift")
    sample_keys = {int(row["sample_id"]): row["sample_key"]
                   for row in workload}
    sequence_keys = {int(row["sample_id"]): row["sequence_key"]
                     for row in workload}

    with execution_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        if ".mlp." not in row["name"]:
            continue
        sample = int(row["sample_id"])
        require(row["sample_key"] == sample_keys[sample] and
                row["sequence_key"] == sequence_keys[sample],
                "execution/sample identity mismatch")
        prefix = row["name"].split(".mlp.")[0] + ".mlp"
        groups[(sample, prefix)].append(row)
    require(len(groups) == 120, "expected 12 FFNs x S10")
    suffixes = (".sn1.spiking_neuron", ".fc1",
                ".sn2.spiking_neuron", ".fc2")
    stage = defaultdict(Counter)
    unique_modules = set()
    total_dense_macs = 0
    total_tokens = 0
    for (sample, prefix), group in sorted(groups.items()):
        require(len(group) == 4, "FFN visible group width drift")
        group.sort(key=lambda row: int(row["call_index"]))
        require([row["name"] for row in group] ==
                [prefix + suffix for suffix in suffixes],
                "FFN visible topology/order drift")
        shapes = [json.loads(row["input_shape"]) for row in group]
        output_shapes = [json.loads(row["output_shape"]) for row in group]
        require(shapes[0] == shapes[1] and output_shapes[1] == shapes[2]
                and output_shapes[2] == shapes[3]
                and output_shapes[3] == shapes[0],
                "FFN shape join drift")
        stage_id = int(prefix.split(".layers.")[1].split(".")[0])
        channels = shapes[0][-1]
        tokens = 1
        for extent in shapes[0][:-1]:
            tokens *= int(extent)
        require(int(group[3]["output_elements"]) == tokens * channels,
                "FFN residual token extent drift")
        dense_macs = int(group[1]["dense_macs"]) + int(
            group[3]["dense_macs"])
        require(dense_macs == tokens * 8 * channels * channels,
                "FFN dense MAC geometry drift")
        stage[stage_id]["dynamic_groups"] += 1
        stage[stage_id]["tokens"] += tokens
        stage[stage_id]["dense_macs"] += dense_macs
        stage[stage_id]["channels"] = channels
        total_dense_macs += dense_macs
        total_tokens += tokens
        unique_modules.add(prefix)
    require(len(unique_modules) == 12, "unique FFN module count drift")

    with activation_path.open("r", encoding="utf-8", newline="") as handle:
        activation_rows = list(csv.DictReader(handle))
    mlp_activation_rows = [row for row in activation_rows
                           if ".mlp" in row.get("name", "")]
    require(not mlp_activation_rows,
            "ordered activation records unexpectedly contain MLP values")
    stage_rows = []
    for stage_id in range(4):
        row = stage[stage_id]
        require(row["dynamic_groups"] in (20, 60),
                "stage dynamic group count drift")
        # Divide S10 aggregation to expose the per-sample oracle population.
        stage_rows.append({
            "stage": stage_id,
            "channels": row["channels"],
            "blocks": row["dynamic_groups"] // 10,
            "tokens_per_sample_all_blocks": row["tokens"] // 10,
            "dense_macs_per_sample_all_blocks": row["dense_macs"] // 10,
        })
    return {
        "samples": 10,
        "sequence_keys": sorted(set(sequence_keys.values())),
        "unique_ffn_modules": len(unique_modules),
        "visible_ffn_groups": len(groups),
        "visible_order": ["sn1", "fc1", "sn2", "fc2"],
        "stage_rows": stage_rows,
        "tokens_per_sample_all_ffns": total_tokens // 10,
        "tokens_s10_all_ffns": total_tokens,
        "dense_macs_per_sample_all_ffns": total_dense_macs // 10,
        "dense_macs_s10_all_ffns": total_dense_macs,
        "bit_trace_records": 0,
        "mlp_activation_value_records": 0,
        "full_ffn_output_values_present": False,
        "token_level_ffn_residual_present": False,
        "qualification": (
            "sample/sequence and FFN geometry are frozen, but execution rows "
            "contain counts/shapes only and activation_records contain no MLP"
        ),
    }


def m32_audit(manifest_path, rows_path):
    manifest = strict_json(manifest_path)
    require(manifest["status"] ==
            "PASS_EXACT_PRODUCER_CONSUMER_TENSOR_IDENTITY",
            "M32 status drift")
    records = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    require(len(records) == manifest["records"] == 100,
            "M32 row population drift")
    ffn = [row for row in records if ".mlp." in row["producer"]]
    require(all("payload_file" not in row and "values" not in row
                for row in records), "M32 unexpectedly contains value payload")
    return {
        "records": len(records),
        "ffn_records": len(ffn),
        "ffn_producers": sorted(set(row["producer"] for row in ffn)),
        "scope": manifest["semantic_scope"],
        "raw_value_sha_only": True,
        "payload_present": False,
        "token_identity_present": False,
        "full_ffn_output_present": False,
    }


def source_manifest_audit(path):
    manifest = strict_json(path)
    operators = manifest["cohort"]["operators"]
    records = manifest["records"]
    require(records and all(row["operator"] in operators for row in records),
            "source manifest operator conservation drift")
    return {
        "schema": manifest["schema"],
        "status": manifest["status"],
        "samples": manifest["cohort"]["samples"],
        "records": len(records),
        "operators": operators,
        "all_operators_bottleneck_conv": all(
            ".resblocks." in name and ".conv" in name for name in operators),
        "ffn_operator_present": any(".mlp." in name for name in operators),
        "full_ffn_output_present": False,
    }


def m233_audit(summary_path, npz_path, records_path):
    summary = strict_json(summary_path)
    require(summary["capture"]["sample_count"] == 10 and
            summary["capture"]["target_module_count"] == 24 and
            summary["capture"]["records"] == 240,
            "M233 population drift")
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    require(len(records) == 240, "M233 CSV population drift")
    with zipfile.ZipFile(npz_path, "r") as archive:
        members = archive.namelist()
    allowed = ("__gamma.npy", "__beta.npy", "__mean.npy",
               "__variance.npy", "__invstd.npy", "__alpha.npy",
               "__offset.npy", "__input_min.npy", "__input_max.npy",
               "__output_min.npy", "__output_max.npy")
    require(members and all(name.endswith(allowed) for name in members),
            "M233 NPZ contains an unclassified tensor")
    require(not any("residual" in name.lower() or "token" in name.lower()
                    for name in members),
            "M233 unexpectedly contains token residual payload")
    return {
        "samples": 10,
        "modules": 24,
        "records": len(records),
        "npz_members": len(members),
        "member_classes": list(allowed),
        "reduction_dimensions": summary["capture"]["reduction_dimensions"],
        "sample_module_identity_present": True,
        "per_channel_statistics_present": True,
        "token_identity_present": False,
        "joint_token_vector_present": False,
        "exact_zero_or_residual_mass_recoverable": False,
        "full_ffn_output_present": False,
    }


def m366_audit(contract_path, script_path, preflight_path):
    contract = strict_json(contract_path)
    preflight = strict_json(preflight_path)
    script = script_path.read_text(encoding="utf-8")
    require(contract["admission"]["gpu_capture_complete"] is False and
            preflight["decision"]["capture_output_created"] is False,
            "M366 capture status drift")
    require('module.__class__.__name__ != "ATLIFTernaryPSN"' in script,
            "M366 ATLIF-only hook guard drift")
    return {
        "capture_complete": False,
        "installed_hook_scope": "ATLIFTernaryPSN input/output only",
        "ffn_module_output_hook": False,
        "reusable_infrastructure": [
            "exact-SHA validation", "frozen S10 sample loader",
            "checkpoint load audit", "no_running BN setup",
            "reset_net per sample", "streaming hook/counter pattern",
            "four-idle-check remote launch guard"
        ],
        "not_reusable_unchanged": [
            "M366 schema/contract", "ATLIF RemainingBudgetCapture",
            "ATLIF static-site tables", "M366 result format"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M383 overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m383_g8_ffn_token_residual_readiness_fastkill_contract_v1",
            "M383 schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M383_CPU_AUDIT",
            "M383 contract not frozen")
    _, paths, identities = load_inputs(args.contract, contract)

    ordered = ordered_trace_audit(
        paths["ordered_profile"], paths["ordered_execution_trace"],
        paths["ordered_activation_records"], paths["ordered_workload"])
    m32 = m32_audit(paths["m32_manifest"], paths["m32_rows"])
    source_manifests = {
        name: source_manifest_audit(paths[name]) for name in
        ("m248_paft_bottleneck", "m252_control_bottleneck",
         "m73_train_bottleneck")
    }
    m233 = m233_audit(paths["m233_summary"], paths["m233_npz"],
                      paths["m233_records"])
    m366 = m366_audit(paths["m366_contract"], paths["m366_script"],
                      paths["m366_preflight"])

    m159 = strict_json(paths["m159_result"])
    require(m159["resolved_topology"]["source_complete_order"] == [
        "sn1_atlif", "dropout1_p0", "fc1_linear", "bn1",
        "sn2_atlif", "dropout2_p0", "fc2_linear", "bn2",
        "drop_path_eval_off", "residual_add"],
        "M159 complete FFN topology drift")
    source = paths["swin_source"].read_text(encoding="utf-8")
    for fragment in ("class MS_Spiking_Mlp(Spiking_Mlp):",
                     "x = self.sn1(x)", "x = self.fc1(x)",
                     "x= self.bn1(", "x = self.sn2(x)",
                     "x = self.fc2(x)", "x = self.bn2("):
        require(fragment in source, "FFN source fragment missing: " + fragment)

    complete_sources = [
        ordered["token_level_ffn_residual_present"],
        m32["full_ffn_output_present"],
        m233["full_ffn_output_present"],
        m366["ffn_module_output_hook"],
    ] + [row["full_ffn_output_present"]
         for row in source_manifests.values()]
    require(not any(complete_sources),
            "a complete residual source exists; M383 fast-kill is stale")

    asset_rows = [
        {"asset": "ordered_trace_s10", "sample_sequence_identity": True,
         "ffn_geometry": True, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "shape/count/MAC only; zero bit traces and no MLP activation rows"},
        {"asset": "m32_dataflow_identity", "sample_sequence_identity": True,
         "ffn_geometry": False, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "producer-consumer SHA/object identity; no values"},
        {"asset": "m233_dynamic_bn_ranges", "sample_sequence_identity": True,
         "ffn_geometry": True, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "per-channel reduced stats only; joint tokens destroyed"},
        {"asset": "m248_paft_ep4", "sample_sequence_identity": True,
         "ffn_geometry": False, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "running-BN PAFT bottleneck Conv inputs only"},
        {"asset": "m252_control_ep4", "sample_sequence_identity": True,
         "ffn_geometry": False, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "running-BN control bottleneck Conv inputs only"},
        {"asset": "m73_h67_train_s32", "sample_sequence_identity": True,
         "ffn_geometry": False, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "train-only bottleneck Conv inputs only"},
        {"asset": "m366_hook", "sample_sequence_identity": True,
         "ffn_geometry": False, "token_identity": False,
         "full_residual_vector": False,
         "qualification": "capture absent and hook is ATLIF-specific"},
    ]

    result = {
        "schema": "m383_g8_ffn_token_residual_readiness_fastkill_v1",
        "status": "PASS_M383_CPU_READINESS_FASTKILL__DATA_INSUFFICIENT",
        "identity": identities,
        "existing_asset_audit": {
            "ordered_trace": ordered,
            "m32_dataflow_identity": m32,
            "source_payload_manifests": source_manifests,
            "m233_dynamic_bn_ranges": m233,
            "m366_reuse": m366,
            "complete_token_ffn_residual_source_count": sum(complete_sources),
            "token_sequence_identity_joinable_if_new_hook_runs": True,
        },
        "cpu_fast_kill": {
            "decision": "NO_GO_EXISTING_DATA_FOR_G8_RESIDUAL_MASS_SWEEP",
            "reason": (
                "No asset contains the post-BN2 full FFN residual vector "
                "F(x) at token granularity with token identity"
            ),
            "tau0_skip_rate": None,
            "strict_threshold_skip_rates": None,
            "source_work_saved_proxy": None,
            "mac_saved_envelope": None,
            "provable_delta_y_norm_budget": None,
            "delta_aee": None,
            "delta_aee_claim_forbidden": True,
            "aggregate_density_substitution_forbidden": True,
            "bn_channel_minmax_substitution_forbidden": True,
            "ideal_oracle_is_executable_hardware": False,
        },
        "known_geometry_for_future_sweep": {
            "stage_rows": ordered["stage_rows"],
            "tokens_per_sample_all_ffns": ordered[
                "tokens_per_sample_all_ffns"],
            "tokens_s10_all_ffns": ordered["tokens_s10_all_ffns"],
            "dense_macs_per_sample_all_ffns": ordered[
                "dense_macs_per_sample_all_ffns"],
            "ffn_accounted_cycles_per_frame_excluding_bn_residual":
                m159["accounted_compute_cycles_per_frame"]
                ["full_ffn_subgraph_excluding_bn_residual"],
            "qualification": (
                "geometry is ready for weighting a future token mask; it "
                "does not supply the mask or a speedup"
            ),
        },
        "minimum_a800_hook": contract["minimum_a800_hook"],
        "threshold_and_norm_contract": contract[
            "threshold_and_norm_contract"],
        "m366_runner_reuse_decision": contract[
            "m366_runner_reuse_decision"],
        "findings": contract["findings"],
        "scorecard": contract["scorecard"],
        "decision": contract["decision"],
        "admission": contract["admission"],
        "claim_boundary": contract["claim_boundary"],
        "asset_rows": asset_rows,
        "output_files": {"asset_readiness": "asset_readiness.csv"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "asset_readiness.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ["asset", "sample_sequence_identity", "ffn_geometry",
                  "token_identity", "full_residual_vector", "qualification"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asset_rows)
    output = args.output_dir / (
        "m383_g8_ffn_token_residual_readiness_fastkill_r1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M383_PASS residual_sources=0 tokens_s10={} macs_per_sample={} "
          "decision={}".format(
              ordered["tokens_s10_all_ffns"],
              ordered["dense_macs_per_sample_all_ffns"],
              result["decision"]), flush=True)


if __name__ == "__main__":
    main()

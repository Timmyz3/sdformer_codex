#!/usr/bin/env python3
"""Audit the exact unit-event identity and M231 bounded stream bridge scope."""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "m162": "6633627495c10b1b56f1ddaf9ff1762eb45aea87e33b00e853fdfb59fc8274f7",
    "m193": "5a6d6de14fe41b7fafab3f8ec1bb0daa2681aca5d41ee4731f549cfb4f3712e1",
    "m218": "f4e1c72a6d6030fd83543d262fd5262a55ac09f0ba95b00b9be8f6023135a9ea",
    "extractor": "f3e213a814d5b9eb3af725009222624aaa8d1c8f4c5eb9fc2a539226e3d6dd69",
    "m167_rtl": "9cb7bbeb4ef720c6d0ec09bb67df2a7ebd3438cde055fd7f6412fb55d1a9705c",
    "m231_rtl": "2df1e2deaf2ea397b60fa1632d571349155b0537fbdfe259b9049d4f722135bb",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
STAGE_WIDTHS = {0: 384, 1: 768, 2: 1536, 3: 3072}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(numerator, denominator):
    require(denominator, "zero denominator")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "float": float(numerator) / float(denominator),
    }


def load_module(path, expected, name):
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_thresholds(repo_root, checkpoint, extractor_path):
    import torch

    extractor = load_module(
        extractor_path, EXPECTED["extractor"], "m231_threshold_extractor")
    extractor.install_pickle_import_paths(repo_root)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if hasattr(loaded, "state_dict"):
        state = loaded.state_dict()
    elif isinstance(loaded, dict):
        state = loaded.get("state_dict", loaded.get("model", loaded))
    else:
        raise RuntimeError("unsupported checkpoint payload")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    keys = sorted(key for key in state
                  if key.endswith(".mlp.sn2.spiking_neuron.thresh"))
    require(len(keys) == 12, "expected 12 FFN sn2 thresholds")
    rows = []
    for key in keys:
        value = state[key]
        require(tuple(value.shape) == () and str(value.dtype) == "torch.float32",
                "sn2 threshold is not scalar float32: " + key)
        raw = value.detach().cpu().contiguous().numpy().tobytes().hex()
        require(float(value.item()) == 1.0 and raw == "0000803f",
                "sn2 threshold is not exact float32 one: " + key)
        rows.append({
            "state_dict_key": key,
            "value_float32": float(value.item()),
            "value_raw_le_hex": raw,
        })
    return type(loaded).__name__, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m162", required=True, type=Path)
    parser.add_argument("--m193", required=True, type=Path)
    parser.add_argument("--m218", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--m167-rtl", required=True, type=Path)
    parser.add_argument("--m231-rtl", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    for name in ("checkpoint", "manifest", "m162", "m193", "m218",
                 "extractor", "m167_rtl", "m231_rtl", "docs359"):
        path = getattr(args, name.replace("_rtl", "_rtl").replace("docs359", "docs359"))
        require(sha256(path) == EXPECTED[name], name + " identity drift")
    contract = json.loads(args.contract.read_text())
    require(contract["schema"]
            == "m231_checkpoint_bound_atlif_fc2_stream_bridge_contract_v1",
            "contract schema drift")
    require(contract["docs359_sha256_unchanged"] == EXPECTED["docs359"],
            "contract docs359 identity drift")

    payload_type, threshold_rows = checkpoint_thresholds(
        args.repo_root.resolve(), args.checkpoint.resolve(), args.extractor)
    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"]
               if record["operator"] == "Linear" and ".mlp.fc2" in record["name"]]
    require(len(records) == 120, "FC2 record population drift")
    stage = defaultdict(lambda: {"records": 0, "tokens": 0,
                                 "input_bits": 0, "packed_bytes": 0,
                                 "events": 0})
    for record in records:
        stage_id = int(record["name"].split(".layers.")[1].split(".")[0])
        width = STAGE_WIDTHS[stage_id]
        shape = [int(value) for value in record["input_shape"]]
        out_shape = [int(value) for value in record["output_shape"]]
        require(shape[-1] == width and out_shape[-1] * 4 == width,
                "FC2 geometry drift")
        tokens = 1
        for value in shape[:-1]:
            tokens *= value
        require(tokens * width == int(record["input_elements"]),
                "FC2 element extent drift")
        require(int(record["packed_bytes"]) * 8 == int(record["input_elements"]),
                "FC2 bitpack extent drift")
        row = stage[stage_id]
        row["records"] += 1
        row["tokens"] += tokens
        row["input_bits"] += int(record["input_elements"])
        row["packed_bytes"] += int(record["packed_bytes"])
        row["events"] += int(record["active_elements"])

    aggregate = {key: sum(row[key] for row in stage.values())
                 for key in ("records", "tokens", "input_bits",
                             "packed_bytes", "events")}
    require(aggregate == {
        "records": 120, "tokens": 5580000,
        "input_bits": 3502080000, "packed_bytes": 437760000,
        "events": 143894510}, "aggregate FC2 identity drift")

    m162 = json.loads(args.m162.read_text())
    m193 = json.loads(args.m193.read_text())
    require(m162["hardware_feedback"]["sn2_threshold_census"]
            == "12/12 FFN sn2 thresholds remain exactly 1.0 in the PAFT checkpoint",
            "PAFT unit threshold receipt drift")
    require(not m162["claim_boundary"]["hardware_accuracy_promotion"],
            "unexpected PAFT promotion")
    require(not m193["claim_boundary"]["bncalib_checkpoint_selected"],
            "unexpected BN calibration promotion")
    m218 = json.loads(args.m218.read_text())
    require(m218["status"] == "PASS_FROZEN_H67_TAGGED_SLICE_SERVICE_PREMODEL_GO",
            "M218 not admitted")

    stage_report = {}
    for stage_id in sorted(stage):
        row = stage[stage_id]
        width = STAGE_WIDTHS[stage_id]
        bridge_bits = 4 * width
        service = m218["per_stage"][str(stage_id)]["primary_service"]["k8_cycles"]
        service_tokens = m218["per_stage"][str(stage_id)]["tokens"]
        producer_cycles_per_token = width // 32
        stage_report[str(stage_id)] = {
            **row,
            "input_width": width,
            "bridge_storage_bits": bridge_bits,
            "bridge_storage_bytes": bridge_bits // 8,
            "mean_packed_feature_map_bytes_per_record": row["packed_bytes"] // row["records"],
            "feature_map_to_bridge_storage_ratio": fraction(
                row["packed_bytes"] // row["records"], bridge_bits // 8),
            "event_producer_issue_cycles_per_token_mean": producer_cycles_per_token,
            "m218_primary_k8_service_cycles_per_token": fraction(
                service, service_tokens),
            "mean_service_over_event_producer_ratio": fraction(
                service, service_tokens * producer_cycles_per_token),
        }

    result = {
        "schema": "m231_checkpoint_bound_atlif_fc2_stream_bridge_screen_v1",
        "status": "PASS_CHECKPOINT_UNIT_EVENT_AND_BOUNDED_STREAM_BRIDGE_SCREEN",
        "identity": {
            "analyzer_start_sha256": sha256(Path(__file__).resolve()),
            **{name + "_sha256": EXPECTED[name] for name in EXPECTED},
            "contract_sha256": sha256(args.contract),
        },
        "h67_ep35_unit_event_proof": {
            "checkpoint_payload_type": payload_type,
            "ffn_sn2_threshold_count": len(threshold_rows),
            "all_scalar_float32_exact_one": True,
            "raw_little_endian_hex": "0000803f",
            "thresholds": threshold_rows,
            "consequence": "M167 BACK event bits are exact FC2 binary values; no runtime amplitude multiplier or weight rescale is required at this boundary",
        },
        "paft_boundary": {
            "checkpoint_sha256": m162["identity"]["checkpoint_sha256"],
            "unit_threshold_receipt": True,
            "hardware_accuracy_promoted": False,
            "reason": "best PAFT accuracy remains dependent on sample-statistic dynamic BN; M193 recalibration was rejected",
        },
        "frozen_fc2_trace": {
            **aggregate,
            "samples": 10,
            "packed_bytes_per_sample": aggregate["packed_bytes"] // 10,
            "separate_write_plus_read_bytes": 2 * aggregate["packed_bytes"],
            "separate_write_plus_read_bytes_per_sample": 2 * aggregate["packed_bytes"] // 10,
            "runtime_amplitude_multiply_terms_elided": aggregate["events"],
        },
        "per_stage": stage_report,
        "bridge": {
            "supported_input_widths": list(STAGE_WIDTHS.values()),
            "two_pair_slots": True,
            "storage_bits": "4*INPUT_WIDTH",
            "maximum_storage_bits": 4 * max(STAGE_WIDTHS.values()),
            "maximum_storage_bytes": 4 * max(STAGE_WIDTHS.values()) // 8,
            "producer_word": "2 time rows x 16 channels",
            "consumer_packet": "4 raw lanes x 96 channels",
        },
        "go_gates": {
            "h67_all_12_sn2_thresholds_exact_one": True,
            "paft_receipt_all_12_sn2_thresholds_exact_one": True,
            "all_120_fc2_trace_records_shape_extent_checked": True,
            "bounded_maximum_bridge_storage_le_1536_bytes": True,
            "mean_m218_service_slower_than_event_producer_all_stages": all(
                item["mean_service_over_event_producer_ratio"]["float"] > 1.0
                for item in stage_report.values()),
        },
        "claim_boundary": {
            "traffic_elision_is_onchip_packed_activation_write_plus_read": True,
            "mean_rate_screen_only_not_finite_buffer_cycle_proof": True,
            "m167_rank3_accuracy_admitted": False,
            "paft_hardware_accuracy_promoted": False,
            "dynamic_bn_barrier_removed": False,
            "vcs": False,
            "physical_sram": False,
            "cycle_speedup": False,
            "energy": False,
            "complete_ffn": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    require(all(result["go_gates"].values()), "M231 go gate failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "traffic_bytes": result["frozen_fc2_trace"]["separate_write_plus_read_bytes"],
        "max_bridge_bytes": result["bridge"]["maximum_storage_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a full-network transaction ledger from the frozen H67 ep35 profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PROFILE = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "h67_fullres_ep35_t450_profile100_20260818"
)
SAMPLES = 100

LOCKED_INPUT_SHA256 = {
    "operator_runtime.csv": (
        "9cb5ccfc15b83c680ca8c96a816df1cdd4b5c4d956bd5c2462175b175b1b6c85"
    ),
    "activation_records.csv": (
        "ce079fb40737bdf33f7328e919351e7cdb0f8358eef097dc8c4dbb66665063ee"
    ),
    "atlif_activity.csv": (
        "ba9053080c964d17645d0d21d5cb47bfc85c9e962050895ba05c7bf0ddee344b"
    ),
    "sample_workload.csv": (
        "68da0e8e1e46e6196ecec2bc2467a664d4dad8b6894e3e4f4e95dfe737178cf2"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_inputs(profile: Path) -> dict[Path, str]:
    return {profile / name: digest for name, digest in LOCKED_INPUT_SHA256.items()}


def audit_inputs(profile: Path) -> None:
    for path, expected in locked_inputs(profile).items():
        if not path.is_file():
            raise RuntimeError(f"missing profile input: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"profile identity drift: {path}\nexpected={expected}\nactual={actual}"
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def integer(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(number(row, key, default)))


def category(name: str) -> str:
    if "patch_embed" in name:
        return "patch_embed"
    if ".attn.linear_q" in name:
        return "attention_q_projection"
    if ".attn.linear_k" in name:
        return "attention_k_projection"
    if ".attn.proj" in name:
        return "attention_rtl_replaced_projection"
    if ".mlp.fc1" in name:
        return "ffn_expand"
    if ".mlp.fc2" in name:
        return "ffn_contract"
    if ".downsample." in name:
        return "downsample"
    if ".decoders." in name:
        return "decoder"
    if ".resblocks." in name:
        return "bottleneck"
    if ".preds." in name:
        return "prediction"
    return "other"


def per_frame(value: float) -> float:
    return value / SAMPLES


def ceil_div(value: float, divisor: int) -> int:
    return int(math.ceil(value / divisor)) if value > 0 else 0


def build_operator_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = int(config["mac_lanes"])
    activation_bits = int(config["activation_bits"])
    weight_bits = int(config["weight_bits"])
    result = []
    for source in rows:
        calls = integer(source, "calls")
        if calls % SAMPLES != 0:
            raise RuntimeError(f"operator calls not divisible by {SAMPLES}: {source['name']}={calls}")
        input_elements = per_frame(integer(source, "input_elements"))
        output_elements = per_frame(integer(source, "output_elements"))
        dense_macs = per_frame(integer(source, "dense_macs"))
        active_macs = per_frame(number(source, "activity_weighted_macs_proxy"))
        input_active = per_frame(integer(source, "input_active"))
        op_category = category(source["name"])
        replaced = op_category == "attention_rtl_replaced_projection"
        binary_ratio = number(source, "input_sample_binary01_ratio", -1.0)
        result.append({
            "name": source["name"],
            "operator": source["operator"],
            "scope": source["scope"],
            "category": op_category,
            "calls_per_frame": calls // SAMPLES,
            "input_elements_per_frame": int(round(input_elements)),
            "input_nonzero_per_frame": int(round(input_active)),
            "input_activity": input_active / input_elements if input_elements else 0.0,
            "output_elements_per_frame": int(round(output_elements)),
            "dense_macs_per_frame": int(round(dense_macs)),
            "activity_weighted_macs_per_frame": int(round(active_macs)),
            "weight_elements": integer(source, "weight_elements"),
            "weight_bytes_int8": ceil_div(integer(source, "weight_elements") * weight_bits, 8),
            "input_bytes_int8_per_frame": ceil_div(input_elements * activation_bits, 8),
            "output_bytes_int8_per_frame": ceil_div(output_elements * activation_bits, 8),
            "input_binary_packed_eligible": binary_ratio >= 0.999,
            "input_bytes_binary_packed_per_frame": (
                ceil_div(input_elements, 8) if binary_ratio >= 0.999 else None
            ),
            "dense_cycles_at_config_lanes": 0 if replaced else ceil_div(dense_macs, lanes),
            "activity_cycles_at_config_lanes": 0 if replaced else ceil_div(active_macs, lanes),
            "replaced_by_attention_rtl_anchor": replaced,
            "input_shape_first": source.get("input_shape_first", ""),
            "output_shape_first": source.get("output_shape_first", ""),
        })
    return result


def build_atlif_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = int(config["atlif_lanes"])
    accumulator_bits = int(config["atlif_accumulator_bits"])
    result = []
    for source in rows:
        calls = integer(source, "calls")
        if calls % SAMPLES != 0:
            raise RuntimeError(f"ATLIF calls not divisible by {SAMPLES}: {source['name']}={calls}")
        elements = per_frame(integer(source, "elements"))
        temporal = max(1, integer(source, "temporal_steps", 1))
        dense_macs = elements * temporal
        first_call_elements = integer(source, "input_first_elements")
        if first_call_elements % temporal != 0:
            raise RuntimeError(
                f"ATLIF first-call elements not divisible by T: "
                f"{source['name']}={first_call_elements}/{temporal}"
            )
        output_row_elements = first_call_elements // temporal
        result.append({
            "name": source["name"],
            "calls_per_frame": calls // SAMPLES,
            "elements_per_frame": int(round(elements)),
            "active_per_frame": int(round(per_frame(integer(source, "active")))),
            "activity": number(source, "activity"),
            "temporal_steps": temporal,
            "dense_macs_per_frame": int(round(dense_macs)),
            "cycles_at_config_lanes": ceil_div(dense_macs, lanes),
            "full_temporal_output_buffer_bytes_per_frame": ceil_div(
                elements * accumulator_bits, 8
            ),
            "minimum_streaming_accumulator_bytes_per_call": ceil_div(
                output_row_elements * accumulator_bits, 8
            ),
            "parameter_entries": integer(source, "parameter_entries"),
            "deployment_dead_result": source.get("deployment_dead_result") == "True",
        })
    return result


def build_activation_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    activation_bits = int(config["activation_bits"])
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["name"], row["kind"])].append(row)
    result = []
    for (name, kind), group in sorted(grouped.items()):
        calls = len(group)
        if calls != SAMPLES:
            raise RuntimeError(f"activation records expected {SAMPLES}: {name}/{kind}={calls}")
        elements = sum(integer(row, "elements") for row in group) / calls
        active = sum(integer(row, "active") for row in group) / calls
        result.append({
            "name": name,
            "kind": kind,
            "calls_per_frame": 1,
            "elements_per_frame": int(round(elements)),
            "active_per_frame": int(round(active)),
            "bytes_int8": ceil_div(elements * activation_bits, 8),
            "bytes_fp16": ceil_div(elements * 16, 8),
            "shape": group[0].get("shape", ""),
        })
    return result


def attention_cycles_from_multisample_receipt(
    receipt: dict[str, Any], windows_per_frame: dict[str, int]
) -> dict[str, Any]:
    if (
        receipt.get("schema") != "h67_attention_multisample_vcs_anchor_v1"
        or receipt.get("status") != "PASS_FRESH_VCS_RTL"
        or receipt.get("identity") != "H67 ep35"
        or receipt.get("sample_count", 0) < 2
        or receipt.get("rows")
        != receipt.get("sample_count") * receipt.get("rows_per_sample", 0)
        or receipt.get("tokens_per_row") != 450
        or any(
            receipt.get(key) != 0
            for key in (
                "fixed_rqtb_equal_mismatches",
                "fixed_rqtb_emitted_mismatches",
                "rtl_index_emitted_mismatches",
            )
        )
    ):
        raise RuntimeError("multisample VCS attention receipt contract mismatch")
    sample_count = int(receipt["sample_count"])
    stage_rows = []
    fixed = 0
    rqtb = 0
    fixed_sum = 0
    rqtb_sum = 0
    if [row.get("stage") for row in receipt.get("stages", [])] != [0, 1, 2, 3]:
        raise RuntimeError("multisample VCS attention stage order mismatch")
    for row in receipt["stages"]:
        stage = int(row["stage"])
        windows = int(windows_per_frame[str(stage)])
        selected_fixed_sum = int(row["fixed_cycles_sum"])
        selected_rqtb_sum = int(row["rqtb_cycles_sum"])
        fixed_numerator = selected_fixed_sum * windows
        rqtb_numerator = selected_rqtb_sum * windows
        if fixed_numerator % sample_count or rqtb_numerator % sample_count:
            raise RuntimeError(
                f"stage {stage} multisample mean does not scale to integral frame cycles"
            )
        frame_fixed = fixed_numerator // sample_count
        frame_rqtb = rqtb_numerator // sample_count
        fixed += frame_fixed
        rqtb += frame_rqtb
        fixed_sum += selected_fixed_sum
        rqtb_sum += selected_rqtb_sum
        stage_rows.append({
            "stage": stage,
            "sample_count": sample_count,
            "selected_fixed_cycles_sum": selected_fixed_sum,
            "selected_rqtb_cycles_sum": selected_rqtb_sum,
            "selected_fixed_cycles_mean": selected_fixed_sum / sample_count,
            "selected_rqtb_cycles_mean": selected_rqtb_sum / sample_count,
            "windows_per_frame": windows,
            "frame_fixed_cycles": frame_fixed,
            "frame_rqtb_cycles": frame_rqtb,
        })
    if (
        fixed_sum != int(receipt["fixed_cycles_total"])
        or rqtb_sum != int(receipt["rqtb_cycles_total"])
    ):
        raise RuntimeError("multisample VCS attention receipt total mismatch")
    return {
        "evidence": "fresh_vcs_multisample10_selected_window_mean_by_stage",
        "claim_boundary": (
            "ten-sample mean of one selected T450 window per block, expanded by stage; "
            "not every spatial window or full-frame RTL"
        ),
        "sample_count": sample_count,
        "fixed_cycles_per_frame": fixed,
        "rqtb_cycles_per_frame": rqtb,
        "speedup": fixed / rqtb,
        "stages": stage_rows,
    }


def attention_cycles(config: dict[str, Any]) -> dict[str, Any]:
    source = config["attention_anchor"]
    if "receipt" in source:
        receipt_path = REPO / source["receipt"]
        if sha256(receipt_path) != source["receipt_sha256"]:
            raise RuntimeError("multisample VCS attention receipt drift")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return attention_cycles_from_multisample_receipt(
            receipt, source["windows_per_frame"]
        )
    source_doc = REPO / source["source_doc"]
    fair_log = REPO / source["fair_log"]
    if sha256(source_doc) != source["source_doc_sha256"]:
        raise RuntimeError("attention anchor source document drift")
    if sha256(fair_log) != source["fair_log_sha256"]:
        raise RuntimeError("attention anchor fair log drift")
    fixed = 0
    rqtb = 0
    stage_rows = []
    for row in source["stages"]:
        stage_fixed = int(row["selected_fixed_cycles"]) * int(row["windows_per_frame"])
        stage_rqtb = int(row["selected_rqtb_cycles"]) * int(row["windows_per_frame"])
        fixed += stage_fixed
        rqtb += stage_rqtb
        stage_rows.append({**row, "frame_fixed_cycles": stage_fixed, "frame_rqtb_cycles": stage_rqtb})
    if fixed != 4_137_640 or rqtb != 3_448_960:
        raise RuntimeError(f"attention anchor drift: {fixed}/{rqtb}")
    return {
        "evidence": source["evidence"],
        "claim_boundary": "sample0 selected window cloned by stage; not full-frame RTL",
        "fixed_cycles_per_frame": fixed,
        "rqtb_cycles_per_frame": rqtb,
        "speedup": fixed / rqtb,
        "stages": stage_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROFILE,
        help="Profile100 directory; hashes remain locked to the frozen ep35 inputs.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile.resolve()
    audit_inputs(profile)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    operators = build_operator_rows(read_csv(profile / "operator_runtime.csv"), config)
    atlif = build_atlif_rows(read_csv(profile / "atlif_activity.csv"), config)
    activations = build_activation_rows(read_csv(profile / "activation_records.csv"), config)
    workloads = read_csv(profile / "sample_workload.csv")
    if len(workloads) != SAMPLES:
        raise RuntimeError(f"expected {SAMPLES} sample workload rows, got {len(workloads)}")
    attention = attention_cycles(config)

    operator_activity_cycles = sum(row["activity_cycles_at_config_lanes"] for row in operators)
    operator_dense_cycles = sum(row["dense_cycles_at_config_lanes"] for row in operators)
    atlif_cycles = sum(row["cycles_at_config_lanes"] for row in atlif if not row["deployment_dead_result"])
    fixed_total = operator_activity_cycles + atlif_cycles + attention["fixed_cycles_per_frame"]
    rqtb_total = operator_activity_cycles + atlif_cycles + attention["rqtb_cycles_per_frame"]
    materialize_bytes = sum(
        row["input_bytes_int8_per_frame"] + row["output_bytes_int8_per_frame"]
        for row in operators
    )
    unique_weight_bytes = sum(row["weight_bytes_int8"] for row in operators)
    atlif_output_payload_bytes = sum(
        row["full_temporal_output_buffer_bytes_per_frame"]
        for row in atlif if not row["deployment_dead_result"]
    )
    largest_atlif_streaming_accumulator = max(
        row["minimum_streaming_accumulator_bytes_per_call"]
        for row in atlif if not row["deployment_dead_result"]
    )
    category_cycles: Counter[str] = Counter()
    for row in operators:
        category_cycles[row["category"]] += row["activity_cycles_at_config_lanes"]
    category_cycles["atlif"] += atlif_cycles
    category_cycles["attention_fixed_anchor"] += attention["fixed_cycles_per_frame"]

    summary = {
        "schema": "h67_ep35_full_network_ledger_v0",
        "status": "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE",
        "claim_boundary": [
            "Real ep35 profile transactions cover all Linear/Conv/ATLIF/stage/decoder/prediction modules.",
            "The attention anchor is RTL-calibrated but expands one selected sample0 window by stage.",
            "Generic operator cycles use an activity-weighted lane model, not RTL calibration.",
            "Residual scheduling, SRAM conflicts, DMA overlap, CACTI, and DRAMsim3 are pending.",
        ],
        "samples": SAMPLES,
        "config": config,
        "source_sha256": {
            "profile100/{}".format(path.name): digest
            for path, digest in locked_inputs(profile).items()
        },
        "counts": {
            "operators": len(operators),
            "atlif_modules": len(atlif),
            "activation_objects": len(activations),
        },
        "attention": attention,
        "cycles_per_frame_model": {
            "operator_activity_weighted": operator_activity_cycles,
            "operator_dense": operator_dense_cycles,
            "atlif_non_dead": atlif_cycles,
            "fixed_total": fixed_total,
            "rqtb_total": rqtb_total,
            "speedup": fixed_total / rqtb_total,
            "fixed_attention_share": attention["fixed_cycles_per_frame"] / fixed_total,
            "category_activity_cycles": dict(category_cycles),
        },
        "traffic_per_frame_proxy": {
            "contract": "materialize every profiled operator input/output; upper proxy, not residency",
            "operator_activation_bytes_int8": materialize_bytes,
            "unique_weight_bytes_int8": unique_weight_bytes,
            "atlif_full_temporal_output_payload_bytes": atlif_output_payload_bytes,
            "atlif_payload_contract": (
                "output payload only; no SRAM read/write traffic is implied"
            ),
            "largest_atlif_streaming_accumulator_bytes": largest_atlif_streaming_accumulator,
            "peak_profiled_activation_object_bytes_int8": max(row["bytes_int8"] for row in activations),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "operator_transactions.csv", operators)
    write_csv(args.output / "atlif_transactions.csv", atlif)
    write_csv(args.output / "activation_objects.csv", activations)
    (args.output / "system_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# H67 ep35 full-network transaction ledger v0",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"- operators: {len(operators)}; ATLIF modules: {len(atlif)}; activation objects: {len(activations)}",
        f"- activity-model cycles/frame: Fixed `{fixed_total}`; RQTB `{rqtb_total}`; speedup `{fixed_total / rqtb_total:.6f}x`",
        f"- modeled Fixed attention share: `{attention['fixed_cycles_per_frame'] / fixed_total:.6%}`",
        f"- materialize-all activation traffic: `{materialize_bytes}` bytes/frame",
        f"- unique INT8 weights: `{unique_weight_bytes}` bytes",
        f"- ATLIF full temporal-output payload: `{atlif_output_payload_bytes}` bytes/frame at {config['atlif_accumulator_bits']} bits",
        f"- largest one-row ATLIF streaming accumulator: `{largest_atlif_streaming_accumulator}` bytes",
        "",
        "This is the first full-network operator ledger. It is not yet the paper cycle/energy table.",
        "The ATLIF payload is not a membrane-state or SRAM-traffic measurement. CACTI, DRAMsim3, residency, bank conflicts, DMA overlap, and calibration of non-attention operators remain pending.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "fixed_cycles": fixed_total,
        "rqtb_cycles": rqtb_total,
        "speedup": fixed_total / rqtb_total,
        "attention_share": attention["fixed_cycles_per_frame"] / fixed_total,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

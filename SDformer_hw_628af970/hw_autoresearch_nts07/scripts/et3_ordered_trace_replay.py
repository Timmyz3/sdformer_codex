#!/usr/bin/env python3
"""回放冻结的 ordered term trace，公平比较 native-m queue 与 ET3。

该脚本只做 CPU 周期/流量模型，不运行网络，不输出目标工艺 PPA。输入 manifest
必须明确 evidence_level；synthetic trace 只能用于协议验证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # synthetic/pre_g0 replay does not need YAML
    yaml = None


SCHEMA = "et3_ordered_term_trace_v1"
SUPPORTED_SCHEMAS = {
    SCHEMA,
    "et3_ordered_term_trace_v2",
}
REQUIRED_ARRAYS = (
    "group_offsets",
    "group_tags",
    "item_mode_multiset",
    "item_gate_code",
    "item_lane_id",
    "item_multiplicity",
    "item_destination",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_list_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def int_list_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in values) + "\n").encode("utf-8")
    ).hexdigest()


def validate_ordered_trace_cohort(cohort: dict[str, Any]) -> None:
    schema = cohort.get("schema")
    if schema not in {"ordered_trace_cohort_v1", "ordered_trace_cohort_v2"}:
        raise ValueError("unsupported ordered trace cohort schema")

    sample_keys = [str(value) for value in cohort.get("sample_keys", [])]
    sequence_keys = [str(value) for value in cohort.get("sequence_keys", [])]
    if (
        int(cohort.get("count", -1)) != 100
        or len(sample_keys) != 100
        or len(sequence_keys) != 100
        or len(set(sample_keys)) != 100
        or any(not value for value in sample_keys)
        or any(not value for value in sequence_keys)
    ):
        raise ValueError("invalid ordered trace cohort cardinality")

    sample_hash = string_list_sha256(sample_keys)
    sequence_hash = string_list_sha256(sequence_keys)
    if cohort.get("sample_key_sha256") != sample_hash:
        raise ValueError("cohort artifact sample-key SHA256 mismatch")
    if cohort.get("sequence_key_sha256") != sequence_hash:
        raise ValueError("cohort artifact sequence-key SHA256 mismatch")

    if schema == "ordered_trace_cohort_v2":
        dataset_indices = [
            int(value) for value in cohort.get("dataset_indices", [])
        ]
        dataset_size = int(cohort.get("dataset_size", 0))
        sequence_counts = cohort.get("sequence_counts", {})
        if (
            cohort.get("dataset_sampling_id")
            != "sequence_proportional_temporal_midpoint_v1"
            or len(dataset_indices) != 100
            or len(set(dataset_indices)) != 100
            or dataset_indices != sorted(dataset_indices)
            or dataset_size < 100
            or any(index < 0 or index >= dataset_size for index in dataset_indices)
            or cohort.get("dataset_indices_sha256")
            != int_list_sha256(dataset_indices)
            or not isinstance(sequence_counts, dict)
            or not sequence_counts
            or sum(int(value) for value in sequence_counts.values()) != 100
            or any(int(value) <= 0 for value in sequence_counts.values())
        ):
            raise ValueError("invalid ordered trace cohort v2 sampling contract")


def resolve_artifact_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"bound provenance artifact is missing: {path}")
    return path


def deployment_contract_from_config(config: dict[str, Any]) -> dict[str, Any]:
    attention = config.get("bsa_attention", {})
    loader = config.get("loader", {})
    test = config.get("test", {})
    swin = config.get("swin_transformer", {})
    return {
        "attention_mode": str(attention.get("mode", "")),
        "hardware_quant_enabled": bool(
            attention.get("hardware_quant_enabled", False)
        ),
        "hardware_rtl_shiftmax_enabled": bool(
            attention.get("hardware_rtl_shiftmax_enabled", False)
        ),
        "hardware_mask_invalid_candidates": bool(
            attention.get("hardware_mask_invalid_candidates", False)
        ),
        "hardware_score_step": float(
            attention.get("hardware_score_step", 0.0)
        ),
        "hardware_gate_step": float(
            attention.get("hardware_gate_step", 0.0)
        ),
        "crop": loader.get("crop"),
        "resolution": loader.get("resolution"),
        "scale_factor": float(test.get("scale_factor", 1.0)),
        "bn_policy": str(test.get("bn_policy", "")),
        "window_size": swin.get("window_size"),
    }


def canonical_item_hash(arrays: dict[str, np.ndarray], start: int, end: int) -> str:
    digest = hashlib.sha256()
    for index in range(start, end):
        row = (
            int(arrays["item_mode_multiset"][index]),
            int(arrays["item_gate_code"][index]),
            int(arrays["item_lane_id"][index]),
            int(arrays["item_multiplicity"][index]),
            int(arrays["item_destination"][index]),
        )
        digest.update((",".join(map(str, row)) + "\n").encode("ascii"))
    return digest.hexdigest()


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return int(ordered[index])


@dataclass(frozen=True)
class ReplayConfig:
    key_cap: int
    segment_depth: int
    fallback_depth: int
    weight_read_latency: int = 1
    destination_issue_interval: int = 1
    partial_drain_transition_cycles: int = 1
    final_commit_cycles: int = 1

    def validate(self) -> None:
        for name in ("key_cap", "segment_depth", "fallback_depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_read_latency <= 0:
            raise ValueError("weight_read_latency must be positive")
        if self.destination_issue_interval <= 0:
            raise ValueError("destination_issue_interval must be positive")


@dataclass
class GroupStats:
    tag: int
    items: int = 0
    ideal_terms: int = 0
    online_terms: int = 0
    fallback_items: int = 0
    partial_drains: int = 0
    native_queue_cycles: int = 0
    et3_single_context_cycles: int = 0
    et3_dual_context_causal_cycles: int = 0
    native_product_computes: int = 0
    et3_product_computes: int = 0
    native_weight_reads: int = 0
    et3_weight_reads: int = 0
    destination_writes: int = 0
    peak_directory_segments: int = 0
    peak_directory_destinations: int = 0
    peak_fallback_items: int = 0
    empty: bool = False


def load_trace(manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported trace schema: {manifest.get('schema')}")
    evidence = manifest.get("evidence_level")
    if evidence not in {"synthetic", "pre_g0", "post_g0"}:
        raise ValueError("evidence_level must be synthetic/pre_g0/post_g0")
    if evidence == "post_g0":
        validate_post_g0_provenance(manifest_path, manifest)
    payload_path = manifest_path.parent / manifest["payload_file"]
    if manifest.get("payload_sha256") != file_sha256(payload_path):
        raise ValueError("payload SHA256 mismatch")
    with np.load(payload_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"missing trace arrays: {missing}")
    validate_trace(manifest, arrays)
    return manifest, arrays


def validate_post_g0_provenance(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    bound_artifacts = (
        ("config", "config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("cohort_file", "cohort_file_sha256"),
    )
    resolved: dict[str, Path] = {}
    for path_field, hash_field in bound_artifacts:
        expected_hash = str(manifest.get(hash_field, ""))
        if not sha256_pattern.fullmatch(expected_hash):
            raise ValueError(f"invalid post_g0 provenance field: {hash_field}")
        resolved[path_field] = resolve_artifact_path(
            manifest_path,
            str(manifest.get(path_field, "")),
        )
        if file_sha256(resolved[path_field]) != expected_hash:
            raise ValueError(f"bound artifact SHA256 mismatch: {path_field}")

    cohort = json.loads(resolved["cohort_file"].read_text(encoding="utf-8"))
    validate_ordered_trace_cohort(cohort)
    sample_keys = [str(value) for value in cohort.get("sample_keys", [])]
    sequence_keys = [str(value) for value in cohort.get("sequence_keys", [])]
    cohort_hash = string_list_sha256(sample_keys)
    if manifest.get("cohort_sha256") != cohort_hash:
        raise ValueError("cohort sample-key SHA256 mismatch")
    group_samples = {
        int(group.get("sample", -1))
        for group in manifest.get("groups", [])
    }
    if group_samples != set(range(100)):
        raise ValueError("post_g0 group sample IDs与100-sample cohort不一致")

    if yaml is None:
        raise ValueError(
            "post_g0 provenance validation requires PyYAML"
        )
    config = yaml.safe_load(resolved["config"].read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("deployment config must decode to a mapping")
    derived_contract = deployment_contract_from_config(config)
    if manifest.get("software_contract") != derived_contract:
        raise ValueError("software contract does not match bound config")

    if manifest.get("resolution", {}).get("full_resolution") is not True:
        raise ValueError("post_g0 evidence requires full_resolution=true")
    if derived_contract["crop"] is not None:
        raise ValueError("post_g0 evidence requires crop=null")
    if derived_contract["resolution"] != [480, 640]:
        raise ValueError("post_g0 evidence requires 480x640 resolution")
    if derived_contract["scale_factor"] != 1.0:
        raise ValueError("post_g0 evidence requires scale_factor=1")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("groups_per_block_sample", 0)) <= 0:
        raise ValueError("post_g0 evidence requires explicit sampling")
    contract = derived_contract
    if "local5" not in str(contract.get("attention_mode", "")):
        raise ValueError("post_g0 evidence requires Local5 attention")
    if contract.get("hardware_quant_enabled") is not True:
        raise ValueError("post_g0 evidence requires hardware quantization")
    if contract.get("hardware_rtl_shiftmax_enabled") is not True:
        raise ValueError("post_g0 evidence requires RTL Shiftmax")
    if contract.get("hardware_mask_invalid_candidates") is not True:
        raise ValueError("post_g0 evidence requires true invalid-candidate mask")
    if not math.isclose(
        float(contract.get("hardware_score_step", 0.0)),
        1.0 / 128.0,
    ):
        raise ValueError("post_g0 evidence requires Q7 score step")
    if not math.isclose(
        float(contract.get("hardware_gate_step", 0.0)),
        1.0 / 128.0,
    ):
        raise ValueError("post_g0 evidence requires Q1.7 gate step")

    producer = manifest.get("producer_order_contract", {})
    if producer.get("id") != "local5_mfep_lane_major_first_gate_v1":
        raise ValueError("unsupported producer-order contract")
    if producer.get("term_order") != [
        "destination_ascending",
        "lane_ascending",
        "gate_first_valid_candidate",
    ]:
        raise ValueError("producer-order term schedule mismatch")
    rtl_hash = str(producer.get("rtl_source_sha256", ""))
    if not sha256_pattern.fullmatch(rtl_hash):
        raise ValueError("invalid producer RTL SHA256")
    rtl_path = resolve_artifact_path(
        manifest_path,
        str(producer.get("rtl_source", "")),
    )
    if file_sha256(rtl_path) != rtl_hash:
        raise ValueError("producer RTL SHA256 mismatch")
    manifest["_post_g0_provenance_verified"] = True


def validate_trace(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    offsets = arrays["group_offsets"]
    tags = arrays["group_tags"]
    if offsets.ndim != 1 or tags.ndim != 1:
        raise ValueError("group offsets/tags must be one-dimensional")
    if len(offsets) != len(tags) + 1 or int(offsets[0]) != 0:
        raise ValueError("invalid group offset cardinality")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("group offsets are not monotonic")
    item_count = int(offsets[-1])
    for name in REQUIRED_ARRAYS[2:]:
        if arrays[name].ndim != 1 or len(arrays[name]) != item_count:
            raise ValueError(f"invalid item array length: {name}")

    groups = manifest.get("groups", [])
    if len(groups) != len(tags):
        raise ValueError("manifest group metadata count mismatch")
    for group_index, group in enumerate(groups):
        start = int(offsets[group_index])
        end = int(offsets[group_index + 1])
        if int(group["tag"]) != int(tags[group_index]):
            raise ValueError("group tag mismatch")
        if bool(group.get("empty", False)) != (start == end):
            raise ValueError("empty-group marker mismatch")
        expected_hash = group.get("ordered_item_sha256")
        if expected_hash != canonical_item_hash(arrays, start, end):
            raise ValueError("ordered item hash mismatch")

    for index in range(item_count):
        mode = int(arrays["item_mode_multiset"][index])
        gate = int(arrays["item_gate_code"][index])
        multiplicity = int(arrays["item_multiplicity"][index])
        if mode not in (0, 1):
            raise ValueError("mode must be binary")
        if gate <= 0:
            raise ValueError("gate code must be positive")
        if multiplicity < 1 or multiplicity > 5:
            raise ValueError("multiplicity must be in [1,5]")
        if not mode and multiplicity != 1:
            raise ValueError("Motion SET item must have multiplicity=1")


def replay_group(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    tag: int,
    config: ReplayConfig,
) -> GroupStats:
    stats = GroupStats(tag=tag, items=end - start, empty=start == end)
    if start == end:
        stats.native_queue_cycles = config.final_commit_cycles
        stats.et3_single_context_cycles = config.final_commit_cycles
        stats.et3_dual_context_causal_cycles = (
            config.final_commit_cycles
        )
        return stats

    items: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    ideal_keys: set[tuple[int, int, int, int]] = set()
    for index in range(start, end):
        item = (
            int(arrays["item_mode_multiset"][index]),
            int(arrays["item_gate_code"][index]),
            int(arrays["item_lane_id"][index]),
            int(arrays["item_multiplicity"][index]),
            int(arrays["item_destination"][index]),
        )
        if item in seen:
            raise ValueError(
                f"duplicate upstream item in group {tag}: {item}"
            )
        seen.add(item)
        items.append(item)
        ideal_keys.add(item[:4])

    stats.ideal_terms = len(ideal_keys)
    stats.native_product_computes = len(items)
    stats.native_weight_reads = len(items)
    stats.destination_writes = len(items)
    stats.native_queue_cycles = (
        len(items) * config.destination_issue_interval
        + (config.weight_read_latency - 1)
        + config.final_commit_cycles
    )

    directory: list[tuple[tuple[int, int, int, int], list[int]]] = []
    fallback: list[tuple[int, int, int, int, int]] = []
    collect_cycles_current_chunk = 0
    chunk_phases: list[tuple[int, int, int]] = []

    def drain(*, partial: bool) -> None:
        nonlocal directory, fallback
        nonlocal collect_cycles_current_chunk
        if not directory and not fallback:
            return
        terms = len(directory) + len(fallback)
        beats = sum(len(destinations) for _, destinations in directory)
        beats += len(fallback)
        emit_cycles = (
            beats * config.destination_issue_interval
            + (config.weight_read_latency - 1)
        )
        stats.online_terms += terms
        stats.et3_product_computes += terms
        stats.et3_weight_reads += terms
        stats.fallback_items += len(fallback)
        stats.peak_directory_segments = max(
            stats.peak_directory_segments, len(directory)
        )
        stats.peak_directory_destinations = max(
            stats.peak_directory_destinations,
            sum(len(destinations) for _, destinations in directory),
        )
        stats.peak_fallback_items = max(
            stats.peak_fallback_items, len(fallback)
        )
        transition_cycles = 0
        if partial:
            stats.partial_drains += 1
            transition_cycles = config.partial_drain_transition_cycles
        chunk_phases.append(
            (
                collect_cycles_current_chunk,
                emit_cycles,
                transition_cycles,
            )
        )
        directory = []
        fallback = []
        collect_cycles_current_chunk = 0

    for mode, gate, lane, multiplicity, destination in items:
        key = (mode, gate, lane, multiplicity)
        while True:
            match = next(
                (
                    entry
                    for entry in directory
                    if entry[0] == key
                    and len(entry[1]) < config.segment_depth
                ),
                None,
            )
            if match is not None:
                match[1].append(destination)
                collect_cycles_current_chunk += 1
                break
            if len(directory) < config.key_cap:
                directory.append((key, [destination]))
                collect_cycles_current_chunk += 1
                break
            if len(fallback) < config.fallback_depth:
                fallback.append(
                    (mode, gate, lane, multiplicity, destination)
                )
                collect_cycles_current_chunk += 1
                break
            drain(partial=True)

    drain(partial=False)
    stats.et3_single_context_cycles = sum(
        collect + transition + emit
        for collect, emit, transition in chunk_phases
    ) + config.final_commit_cycles
    stats.et3_dual_context_causal_cycles = (
        causal_dual_context_cycles(chunk_phases)
        + config.final_commit_cycles
    )
    return stats


def causal_dual_context_cycles(
    chunk_phases: list[tuple[int, int, int]],
) -> int:
    """Schedule ordered chunks on one collector, one emitter, two contexts."""
    context_free = [0, 0]
    collect_engine_free = 0
    emit_engine_free = 0
    for index, (collect, emit, transition) in enumerate(chunk_phases):
        context = index & 1
        collect_start = max(
            collect_engine_free,
            context_free[context],
        )
        collect_end = collect_start + collect + transition
        collect_engine_free = collect_end
        emit_start = max(collect_end, emit_engine_free)
        emit_end = emit_start + emit
        emit_engine_free = emit_end
        context_free[context] = emit_end
    return emit_engine_free


def aggregate(
    manifest: dict[str, Any],
    groups: list[GroupStats],
    config: ReplayConfig,
) -> dict[str, Any]:
    totals = {
        field: sum(getattr(group, field) for group in groups)
        for field in (
            "items",
            "ideal_terms",
            "online_terms",
            "fallback_items",
            "partial_drains",
            "native_queue_cycles",
            "et3_single_context_cycles",
            "et3_dual_context_causal_cycles",
            "native_product_computes",
            "et3_product_computes",
            "native_weight_reads",
            "et3_weight_reads",
            "destination_writes",
        )
    }
    ideal_saved = totals["native_product_computes"] - totals["ideal_terms"]
    online_saved = (
        totals["native_product_computes"] - totals["et3_product_computes"]
    )
    retention = online_saved / ideal_saved if ideal_saved > 0 else 1.0
    cycle_fields = (
        "native_queue_cycles",
        "et3_single_context_cycles",
        "et3_dual_context_causal_cycles",
    )
    distributions = {}
    for field in cycle_fields:
        values = [getattr(group, field) for group in groups]
        distributions[field] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0),
        }
    return {
        "schema": "et3_ordered_trace_replay_v1",
        "evidence_level": manifest["evidence_level"],
        "performance_claim_allowed": (
            manifest["evidence_level"] == "post_g0"
            and manifest.get("_post_g0_provenance_verified") is True
        ),
        "trace": {
            "config_sha256": manifest.get("config_sha256", ""),
            "checkpoint_sha256": manifest.get("checkpoint_sha256", ""),
            "cohort_sha256": manifest.get("cohort_sha256", ""),
            "resolution": manifest.get("resolution", {}),
        },
        "config": asdict(config),
        "totals": totals,
        "online_product_reuse_retention": retention,
        "cycle_distributions": distributions,
        "groups": [asdict(group) for group in groups],
        "evidence_boundary": (
            "CPU 参数化因果调度模型；不是 RTL 实测 cycle、DC/STA/SAIF "
            "或端到端 FPS。"
            "synthetic/pre_g0 输入不得形成部署性能结论。"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    dist = report["cycle_distributions"]
    lines = [
        "# ET3 Ordered Trace CPU Replay",
        "",
        f"- 证据等级：`{report['evidence_level']}`",
        f"- 允许性能主张：`{report['performance_claim_allowed']}`",
        f"- 证据边界：{report['evidence_boundary']}",
        "",
        "## 总工作量",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in totals.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            (
                "| online_product_reuse_retention | "
                f"{report['online_product_reuse_retention']:.6%} |"
            ),
            "",
            "## Group 周期分布",
            "",
            "| 模型 | mean | p95 | p99 | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in dist.items():
        lines.append(
            f"| {name} | {row['mean']:.2f} | {row['p95']} | "
            f"{row['p99']} | {row['max']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-cap", type=int, default=128)
    parser.add_argument("--segment-depth", type=int, default=16)
    parser.add_argument("--fallback-depth", type=int, default=16)
    parser.add_argument("--weight-read-latency", type=int, default=2)
    parser.add_argument(
        "--destination-issue-interval",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    config = ReplayConfig(
        key_cap=args.key_cap,
        segment_depth=args.segment_depth,
        fallback_depth=args.fallback_depth,
        weight_read_latency=args.weight_read_latency,
        destination_issue_interval=args.destination_issue_interval,
    )
    config.validate()
    manifest, arrays = load_trace(args.manifest)
    offsets = arrays["group_offsets"]
    tags = arrays["group_tags"]
    groups = [
        replay_group(
            arrays,
            int(offsets[index]),
            int(offsets[index + 1]),
            int(tags[index]),
            config,
        )
        for index in range(len(tags))
    ]
    report = aggregate(manifest, groups, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

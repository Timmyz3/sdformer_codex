#!/usr/bin/env python3
"""Fail-closed Local5 EREP formal topology preflight; never emits G0 admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "local5_erep_formal_preflight_v4"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = (
    ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
)
PROFILE_DIR = Path(
    os.environ.get("LOCAL5_EREP_PROFILE_DIR", str(DEFAULT_PROFILE_DIR))
).resolve()
SELECTION_PLAN = PROFILE_DIR / "joint_window_selection_plan.json"
FORMAL_MANIFEST = PROFILE_DIR / "ordered_term_manifest.json"
PROJECTION_CONTRACT = PROFILE_DIR / "checkpoint_projection_contract.json"
STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
STAGE_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
STAGE_WINDOWS = {0: 440, 1: 120, 2: 30, 3: 10}
BLOCK_ORDER = tuple(
    (stage, block)
    for stage in range(4)
    for block in range(STAGE_BLOCKS[stage])
)
SELECTION_PLAN_SHA256 = "4e8732210a64cfcb553e7f4eee3657be70cc975a38839527e4792668d6deaf6b"
PROJECTION_CONTRACT_SHA256 = "c2bf6f406345d1bcc0f8a883318f59dc63116a96c96cd4138af83ce495ce9669"
PROJECTION_PAYLOAD_SHA256 = "81edeefa16d2177c8739579f42485a58f2a70581a078e9ea367d7422f73446f4"
EXPECTED_HXH_TASK_SHA256 = "5e894781aaca24b307fc0c33ddb116b28082694f484e3bb15784b8da7a6b07c6"
SELECTION_PLAN_FIELDS = {
    "schema", "sampling_id", "seed", "cohort_sha256",
    "source_cohort_manifest", "source_cohort_manifest_sha256",
    "probability_contract", "analysis_contract", "records",
}
FORMAL_MANIFEST_FIELDS = {
    "schema", "evidence_level", "payload_file", "payload_sha256",
    "config", "config_sha256", "checkpoint", "checkpoint_sha256",
    "cohort_file", "cohort_file_sha256", "cohort_sha256",
    "run_identity_file", "run_identity_file_sha256", "qualification",
    "resolution", "software_contract", "threshold_training_semantics",
    "producer_order_contract", "source_frontier_contract",
    "source_descriptor_contract", "attention_score_trace_contract",
    "sampling", "groups", "projection_contract_file",
    "projection_contract_file_sha256", "projection_contract_payload",
    "projection_contract_payload_sha256",
}
PROJECTION_CONTRACT_FIELDS = {
    "schema", "status", "checkpoint", "checkpoint_sha256", "payload_file",
    "payload_sha256", "topology_contract", "blocks", "value_contract",
    "quantization_order", "quantization", "numeric_scope",
    "runtime_datapath", "raw_vs_folded", "bn_policy", "bn_folding",
    "source_sha256",
}
MANIFEST_GROUP_FIELDS = {
    "tag", "empty", "sample", "stage", "block", "window", "head",
    "flat_group", "batch_windows", "heads", "lanes", "tokens",
    "time_planes", "plane_tokens", "spatial_side", "plane_execution",
    "module", "selection", "ordered_item_sha256",
}
PROJECTION_BLOCK_FIELDS = {
    "stage", "block", "module", "prefix", "weight_name", "theta_name",
    "bias_name", "theta", "weight_shape", "heads", "head_dim",
    "bias_present", "weight_scale_exp2_min", "weight_scale_exp2_max",
    "raw_vs_folded_weight_int8_mismatch", "raw_vs_folded_scale_exp2_mismatch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    return value


def strict_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def safe_profile_artifact(profile_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a profile-local basename")
    profile = profile_dir.resolve()
    artifact = (profile / value).resolve()
    if artifact.parent != profile:
        raise ValueError(f"{name} escapes the frozen profile directory")
    return artifact


def root_relative(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside the repository root: {path}") from error


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required formal artifact is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def validate_selection_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(plan, Mapping)
        or set(plan) != SELECTION_PLAN_FIELDS
        or plan.get("schema") != "local5_uniform_joint_window_plan_v1"
        or plan.get("sampling_id") != "uniform_plan_window_all_heads_v1"
    ):
        raise ValueError("selection plan header violates the frozen contract")
    if strict_int(plan.get("seed"), "selection seed") != 20260809:
        raise ValueError("selection plan seed violates the frozen contract")
    strict_sha256(plan.get("cohort_sha256"), "selection cohort SHA")
    strict_sha256(
        plan.get("source_cohort_manifest_sha256"),
        "selection source cohort manifest SHA",
    )
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != 1200:
        raise ValueError("selection plan must contain exactly 1200 records")

    normalized: list[dict[str, Any]] = []
    expected_fields = {
        "sample", "stage", "block", "heads", "batch_windows", "window",
        "inclusion_probability", "analysis_weight",
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError(f"selection record {index} has a non-frozen field set")
        sample = strict_int(record["sample"], f"record {index} sample")
        stage = strict_int(record["stage"], f"record {index} stage")
        block = strict_int(record["block"], f"record {index} block")
        window = strict_int(record["window"], f"record {index} window")
        expected_sample = index // len(BLOCK_ORDER)
        expected_stage, expected_block = BLOCK_ORDER[index % len(BLOCK_ORDER)]
        if (sample, stage, block) != (expected_sample, expected_stage, expected_block):
            raise ValueError(f"selection record {index} is not canonical sample/block order")
        if stage not in STAGE_HEADS or block >= STAGE_BLOCKS[stage]:
            raise ValueError(f"selection record {index} has invalid topology")
        windows = STAGE_WINDOWS[stage]
        heads = strict_int(record["heads"], f"record {index} heads", minimum=1)
        batch_windows = strict_int(
            record["batch_windows"], f"record {index} batch_windows", minimum=1
        )
        if type(record["inclusion_probability"]) is not float or type(
            record["analysis_weight"]
        ) is not float:
            raise ValueError(f"selection record {index} probability/weight types are not frozen")
        if (
            heads != STAGE_HEADS[stage]
            or batch_windows != windows
            or not 0 <= window < windows
            or record["inclusion_probability"] != 1.0 / windows
            or record["analysis_weight"] != float(windows)
        ):
            raise ValueError(f"selection record {index} violates stage/window contract")
        normalized.append(
            {"sample": sample, "stage": stage, "block": block, "window": window}
        )
    if len({tuple(record.values()) for record in normalized}) != 1200:
        raise ValueError("selection plan contains duplicate canonical windows")
    return tuple(normalized)


def expected_head_keys(
    windows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int, int, int, int], ...]:
    keys = []
    for index, row in enumerate(windows):
        stage = strict_int(row["stage"], f"window {index} stage")
        keys.extend(
            (
                strict_int(row["sample"], f"window {index} sample"),
                stage,
                strict_int(row["block"], f"window {index} block"),
                strict_int(row["window"], f"window {index} window"),
                head,
            )
            for head in range(STAGE_HEADS[stage])
        )
    if len(keys) != 13_800 or len(set(keys)) != 13_800:
        raise ValueError("canonical plan does not expand to exactly 13800 unique head keys")
    return tuple(keys)


def validate_manifest_groups(
    manifest: Mapping[str, Any], windows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != FORMAL_MANIFEST_FIELDS
        or manifest.get("schema") != "et3_ordered_term_trace_v2"
        or manifest.get("evidence_level") != "post_g0"
        or (manifest.get("qualification") or {}).get("qualified") is not True
    ):
        raise ValueError("formal manifest header/qualification is not admissible")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != 13_800:
        raise ValueError("formal manifest must contain exactly 13800 head groups")
    observed = []
    ordered_identity = []
    tags = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != MANIFEST_GROUP_FIELDS:
            raise ValueError(f"manifest group {index} has a non-frozen field set")
        stage = strict_int(group.get("stage"), f"group {index} stage")
        if stage not in STAGE_HEADS:
            raise ValueError(f"manifest group {index} has invalid stage")
        key = (
            strict_int(group.get("sample"), f"group {index} sample"),
            stage,
            strict_int(group.get("block"), f"group {index} block"),
            strict_int(group.get("window"), f"group {index} window"),
            strict_int(group.get("head"), f"group {index} head"),
        )
        tag = strict_int(group.get("tag"), f"group {index} tag")
        tags.append(tag)
        heads = strict_int(group.get("heads"), f"group {index} heads", minimum=1)
        lanes = strict_int(group.get("lanes"), f"group {index} lanes", minimum=1)
        tokens = strict_int(group.get("tokens"), f"group {index} tokens", minimum=1)
        time_planes = strict_int(
            group.get("time_planes"), f"group {index} time_planes", minimum=1
        )
        plane_tokens = strict_int(
            group.get("plane_tokens"), f"group {index} plane_tokens", minimum=1
        )
        spatial_side = strict_int(
            group.get("spatial_side"), f"group {index} spatial_side", minimum=1
        )
        flat_group = strict_int(group.get("flat_group"), f"group {index} flat_group")
        batch_windows = strict_int(
            group.get("batch_windows"), f"group {index} batch_windows", minimum=1
        )
        expected_module = (
            "sttmultires_unet.encoders.swin3d.layers."
            f"{stage}.swin_blocks.{key[2]}.attn"
        )
        if (
            type(group.get("empty")) is not bool
            or heads != STAGE_HEADS[stage]
            or lanes != 32
            or tokens != 450
            or time_planes != 2
            or plane_tokens != 225
            or spatial_side != 15
            or batch_windows != STAGE_WINDOWS[stage]
            or flat_group != key[3] * heads + key[4]
            or group.get("plane_execution") != "plane_serial_drain"
            or group.get("selection") != "uniform_plan_window_all_heads_v1"
            or group.get("module") != expected_module
            or not isinstance(group.get("ordered_item_sha256"), str)
            or len(group["ordered_item_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in group["ordered_item_sha256"])
            or key[2] >= STAGE_BLOCKS[stage]
            or key[4] >= STAGE_HEADS[stage]
        ):
            raise ValueError(f"manifest group {index} violates full-resolution topology")
        observed.append(key)
        ordered_identity.append(
            (tag, *key, strict_sha256(group["ordered_item_sha256"], f"group {index} item SHA"))
        )
    expected = expected_head_keys(windows)
    if Counter(observed) != Counter(expected):
        raise ValueError("manifest groups do not exactly cover the 1200-window all-head key set")
    if len(set(observed)) != len(observed):
        raise ValueError("manifest contains duplicate head keys")
    if sorted(tags) != list(range(13_800)):
        raise ValueError("manifest tags are not exactly 0..13799")
    return {
        "head_group_count": len(observed),
        "group_order_key_sha256": canonical_sha(observed),
        "group_order_identity_sha256": canonical_sha(ordered_identity),
        "canonical_sorted_key_sha256": canonical_sha(sorted(observed)),
        "canonical_key_coverage_exact": True,
    }


def validate_projection_contract(
    contract: Mapping[str, Any], payload: Mapping[str, np.ndarray]
) -> dict[tuple[int, int], dict[str, Any]]:
    if (
        not isinstance(contract, Mapping)
        or set(contract) != PROJECTION_CONTRACT_FIELDS
        or contract.get("schema") != "local5_checkpoint_projection_contract_v2"
        or contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
    ):
        raise ValueError("projection contract header is invalid")
    blocks = contract.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(BLOCK_ORDER):
        raise ValueError("projection contract must contain exactly 12 blocks")
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    expected_arrays: set[str] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or set(block) != PROJECTION_BLOCK_FIELDS:
            raise ValueError(f"projection block {index} has a non-frozen field set")
        stage = strict_int(block.get("stage"), f"projection block {index} stage")
        block_id = strict_int(block.get("block"), f"projection block {index} block")
        if (stage, block_id) != BLOCK_ORDER[index]:
            raise ValueError("projection blocks are not in frozen topology order")
        heads = STAGE_HEADS[stage]
        channels = heads * 32
        prefix = f"s{stage}_b{block_id}"
        observed_heads = strict_int(block.get("heads"), f"projection block {index} heads", minimum=1)
        head_dim = strict_int(block.get("head_dim"), f"projection block {index} head_dim", minimum=1)
        shape = block.get("weight_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int for value in shape)
        ):
            raise ValueError(f"projection block {stage}/{block_id} weight shape type is invalid")
        if (
            observed_heads != heads
            or head_dim != 32
            or shape != [channels, channels]
            or block.get("prefix") != prefix
            or type(block.get("bias_present")) is not bool
            or type(block.get("theta")) is not float
            or strict_int(
                block.get("raw_vs_folded_weight_int8_mismatch"),
                f"projection block {index} weight mismatch",
            ) != 0
            or strict_int(
                block.get("raw_vs_folded_scale_exp2_mismatch"),
                f"projection block {index} scale mismatch",
            ) != 0
        ):
            raise ValueError(f"projection block {stage}/{block_id} is not H*32 square")
        specs = {
            f"{prefix}_theta_float32": ((1,), np.dtype("float32")),
            f"{prefix}_weight_float32": ((channels, channels), np.dtype("float32")),
            f"{prefix}_effective_weight_float32": ((channels, channels), np.dtype("float32")),
            f"{prefix}_weight_int8": ((channels, channels), np.dtype("int8")),
            f"{prefix}_weight_scale_exp2": ((channels,), np.dtype("int16")),
            f"{prefix}_bias_float32": ((channels,), np.dtype("float32")),
        }
        for name, (shape, dtype) in specs.items():
            if name not in payload:
                raise ValueError(f"projection payload is missing {name}")
            array = payload[name]
            if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
                raise ValueError(f"projection payload {name} has invalid shape/dtype")
        expected_arrays.update(specs)
        by_key[(stage, block_id)] = {
            "heads": heads,
            "input_head_count": heads,
            "output_tile_count": heads,
            "task_count_per_window": heads * heads,
        }
    if set(payload) != expected_arrays:
        raise ValueError("projection payload array set is not exact")
    return by_key


def enumerate_hxh_tasks(
    windows: Sequence[Mapping[str, Any]],
    blocks: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    if len(windows) != 1200:
        raise ValueError("H×H task enumeration requires exactly 1200 canonical windows")
    tasks = []
    for index, row in enumerate(windows):
        sample = strict_int(row.get("sample"), f"window {index} sample")
        stage = strict_int(row.get("stage"), f"window {index} stage")
        block_id = strict_int(row.get("block"), f"window {index} block")
        window = strict_int(row.get("window"), f"window {index} window")
        key = (stage, block_id)
        expected_sample = index // len(BLOCK_ORDER)
        expected_stage, expected_block = BLOCK_ORDER[index % len(BLOCK_ORDER)]
        if (sample, *key) != (
            expected_sample, expected_stage, expected_block
        ):
            raise ValueError("H×H task enumeration input is not canonical window order")
        block = blocks.get(key)
        if block is None:
            raise ValueError(f"projection contract lacks block {key}")
        heads = strict_int(block["heads"], f"projection block {key} heads", minimum=1)
        input_heads = strict_int(
            block.get("input_head_count"),
            f"projection block {key} input head count",
            minimum=1,
        )
        output_tiles = strict_int(
            block.get("output_tile_count"),
            f"projection block {key} output tile count",
            minimum=1,
        )
        if input_heads != heads or output_tiles != heads:
            raise ValueError(f"projection block {key} is not H input heads by H output tiles")
        tasks.extend(
            (
                sample, key[0], key[1], window, input_head, output_tile
            )
            for input_head in range(heads)
            for output_tile in range(heads)
        )
    if len(tasks) != 210_600 or len(set(tasks)) != 210_600:
        raise ValueError("formal projection expansion is not exactly 210600 unique HxH tasks")
    if canonical_sha(tasks) != EXPECTED_HXH_TASK_SHA256:
        raise ValueError("formal H×H task order differs from the preregistered digest")
    return tuple(tasks)


def validate_manifest_artifact_bindings(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    profile_dir: Path = PROFILE_DIR,
    formal_manifest: Path = FORMAL_MANIFEST,
    projection_contract: Path = PROJECTION_CONTRACT,
    root: Path = ROOT,
) -> dict[str, dict[str, str]]:
    payload = safe_profile_artifact(
        profile_dir, manifest.get("payload_file"), "formal payload file"
    )
    cohort = safe_profile_artifact(
        profile_dir, manifest.get("cohort_file"), "formal cohort file"
    )
    projection_json = safe_profile_artifact(
        profile_dir,
        manifest.get("projection_contract_file"),
        "formal projection contract file",
    )
    projection_npz = safe_profile_artifact(
        profile_dir,
        manifest.get("projection_contract_payload"),
        "formal projection payload file",
    )
    expected_projection_payload = safe_profile_artifact(
        profile_dir, contract.get("payload_file"), "projection contract payload file"
    )
    if projection_json.resolve() != projection_contract.resolve():
        raise ValueError("formal manifest is not bound to the frozen projection contract")
    if projection_npz.resolve() != expected_projection_payload.resolve():
        raise ValueError("formal manifest projection payload disagrees with its contract")

    expected = {
        "formal_manifest": (formal_manifest, sha256_file(formal_manifest)),
        "ordered_payload": (
            payload,
            strict_sha256(manifest.get("payload_sha256"), "formal payload SHA"),
        ),
        "cohort": (
            cohort,
            strict_sha256(manifest.get("cohort_file_sha256"), "formal cohort file SHA"),
        ),
        "projection_contract": (
            projection_json,
            strict_sha256(
                manifest.get("projection_contract_file_sha256"),
                "formal projection contract SHA",
            ),
        ),
        "projection_payload": (
            projection_npz,
            strict_sha256(
                manifest.get("projection_contract_payload_sha256"),
                "formal projection payload SHA",
            ),
        ),
    }
    bindings: dict[str, dict[str, str]] = {}
    for name, (path, claimed_sha) in expected.items():
        if not path.is_file() or sha256_file(path) != claimed_sha:
            raise ValueError(f"formal artifact binding failed: {name}")
        bindings[name] = {"path": root_relative(path, root), "sha256": claimed_sha}

    contract_payload_sha = strict_sha256(
        contract.get("payload_sha256"), "projection contract payload SHA"
    )
    if (
        bindings["projection_contract"]["sha256"] != sha256_file(projection_contract)
        or bindings["projection_payload"]["sha256"] != contract_payload_sha
    ):
        raise ValueError("formal projection artifact hashes disagree with fixed inputs")
    return bindings


def evaluate_fixed_preflight() -> dict[str, Any]:
    plan = load_json(SELECTION_PLAN)
    if sha256_file(SELECTION_PLAN) != SELECTION_PLAN_SHA256:
        raise ValueError("selection plan SHA differs from preregistered value")
    windows = validate_selection_plan(plan)
    if sha256_file(PROJECTION_CONTRACT) != PROJECTION_CONTRACT_SHA256:
        raise ValueError("projection contract SHA differs from preregistered value")
    contract = load_json(PROJECTION_CONTRACT)
    payload_path = PROFILE_DIR / str(contract.get("payload_file", ""))
    if (
        not payload_path.is_file()
        or sha256_file(payload_path) != PROJECTION_PAYLOAD_SHA256
        or contract.get("payload_sha256") != PROJECTION_PAYLOAD_SHA256
    ):
        raise ValueError("projection payload SHA binding failed")
    with np.load(payload_path, allow_pickle=False) as payload_file:
        payload = {name: payload_file[name] for name in payload_file.files}
    blocks = validate_projection_contract(contract, payload)
    tasks = enumerate_hxh_tasks(windows, blocks)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence": "[契约审计]",
        "admission_generated": False,
        "selection_plan": {
            "file": root_relative(SELECTION_PLAN),
            "window_count": len(windows),
            "sha256": sha256_file(SELECTION_PLAN),
        },
        "projection_contract": {
            "contract_file": root_relative(PROJECTION_CONTRACT),
            "contract_file_sha256": sha256_file(PROJECTION_CONTRACT),
            "payload_file": root_relative(payload_path),
            "block_count": len(blocks),
            "hxh_shape_exact": True,
            "payload_sha256": sha256_file(payload_path),
        },
        "expected_head_groups": len(expected_head_keys(windows)),
        "hxh_projection_tasks": len(tasks),
        "hxh_task_sha256": canonical_sha(tasks),
    }
    if not FORMAL_MANIFEST.is_file():
        report.update(
            status="DENY_FORMAL_MANIFEST_ABSENT",
            formal_manifest_present=False,
            formal_manifest_sha256=None,
            formal_group_contract=None,
            formal_artifact_bindings=None,
        )
        return report
    manifest = load_json(FORMAL_MANIFEST)
    sampling = manifest.get("sampling")
    if (
        not isinstance(sampling, Mapping)
        or sampling.get("method") != "uniform_plan_window_all_heads_v1"
        or strict_int(sampling.get("seed"), "formal sampling seed") != 20260809
        or sampling.get("selection_plan_sha256") != SELECTION_PLAN_SHA256
        or manifest.get("cohort_sha256") != plan.get("cohort_sha256")
    ):
        raise ValueError("formal manifest is not bound to the frozen selection/cohort")
    group_contract = validate_manifest_groups(manifest, windows)
    artifact_bindings = validate_manifest_artifact_bindings(manifest, contract)
    report.update(
        status="PREFLIGHT_PASS_NOT_G0",
        formal_manifest_present=True,
        formal_manifest_sha256=sha256_file(FORMAL_MANIFEST),
        formal_group_contract=group_contract,
        formal_artifact_bindings=artifact_bindings,
    )
    return report


def validate_report_for_packaging(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise ValueError("preflight report must be a mapping")
    expected = evaluate_fixed_preflight()
    if dict(report) != expected:
        raise ValueError("preflight report does not match independently replayed fixed inputs")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_fixed_preflight()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{report['status']} windows={report['selection_plan']['window_count']} "
        f"head_groups={report['expected_head_groups']} tasks={report['hxh_projection_tasks']}"
    )


if __name__ == "__main__":
    main()

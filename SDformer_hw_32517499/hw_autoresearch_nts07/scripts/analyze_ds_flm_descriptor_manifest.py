#!/usr/bin/env python3
"""分析post-G0 source-major descriptor trace中的DS-FLM结构特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from et3_ordered_trace_replay import file_sha256, load_trace


HW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW_ROOT.parent
EXP_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention"
POST_G0_BLOCK_PAIRS = {
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
}
ROLE_ORDER = ["self", "up", "down", "left", "right"]
SOURCE_CONSUMER_RELATION = [
    "self_destination",
    "down_destination_uses_source_as_up_candidate",
    "up_destination_uses_source_as_down_candidate",
    "right_destination_uses_source_as_left_candidate",
    "left_destination_uses_source_as_right_candidate",
]
RELATION_BINDING_NAMES = (
    "relation_transpose_rtl",
    "retirement_scheduler_rtl",
    "relation_sync_bank_rtl",
    "relation_assertions",
    "relation_sync_bank_assertions",
    "relation_vector_generator",
    "relation_miter_tb",
    "relation_miter_script",
)


def expected_source_binding_paths() -> dict[str, Path]:
    baseline = REPO_ROOT / "third_party/SDformerFlow"
    overlay = EXP_ROOT / "overlay/models/STSwinNet_SNN"
    return {
        "watcher": (
            HW_ROOT / "scripts/run_local5_qfsa_profile_after_fullres.py"
        ),
        "profiler": HW_ROOT / "scripts/profile_local5_hardware_features.py",
        "base_profiler": (
            EXP_ROOT / "entrypoints/profile_nts11_hardware_p0.py"
        ),
        "attention_impl": overlay / "bsa_attention.py",
        "checkpoint_loader": overlay / "h9_load_audit.py",
        "model_impl": (
            baseline
            / "models/STSwinNet_SNN/Spiking_STSwinNet.py"
        ),
        "dataset_impl": (
            baseline / "DSEC_dataloader/DSEC_dataset_lite.py"
        ),
        "trace_loader": HW_ROOT / "scripts/et3_ordered_trace_replay.py",
        "replay": HW_ROOT / "scripts/replay_local5_frontier_trace.py",
        "descriptor_analyzer": Path(__file__).resolve(),
        "acceptance": (
            HW_ROOT / "scripts/validate_local5_postg0_acceptance.py"
        ),
        "relation_transpose_rtl": (
            HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
        ),
        "retirement_scheduler_rtl": (
            HW_ROOT / "rtl_qfit/qfit_retirement_scheduler.sv"
        ),
        "relation_sync_bank_rtl": (
            HW_ROOT / "rtl_qfit/qfit_sync_1r1w_bank.sv"
        ),
        "relation_assertions": (
            HW_ROOT / "verif_qfit/qfit_relation_transpose_assertions.sv"
        ),
        "relation_sync_bank_assertions": (
            HW_ROOT / "verif_qfit/qfit_sync_bank_assertions.sv"
        ),
        "relation_vector_generator": (
            HW_ROOT / "scripts/generate_local5_relation_transpose_vectors.py"
        ),
        "relation_miter_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_relation_transpose_python_miter.sv"
        ),
        "relation_miter_script": (
            HW_ROOT / "sim_qfit/run_qfit_relation_transpose_python_miter.sh"
        ),
        "score_trace_generator": (
            HW_ROOT / "scripts/generate_local5_checkpoint_score_vectors.py"
        ),
        "score_trace_reporter": (
            HW_ROOT / "scripts/report_local5_checkpoint_score_rtl.py"
        ),
        "score_trace_tb": (
            HW_ROOT / "tb_local5/tb_local5_score_shiftmax_vectors.sv"
        ),
        "score_trace_script": (
            HW_ROOT / "sim_local5/run_local5_checkpoint_score_trace_checks.sh"
        ),
        # Keep aligned with production post-G0 writer bindings used by profile/acceptance.
        "projection_quantizer": (
            EXP_ROOT / "entrypoints/h67_bit_trace.py"
        ),
        "projection_contract_verifier": (
            HW_ROOT / "scripts/verify_local5_theta_folded_projection_contract.py"
        ),
        "projection_trace_generator": (
            HW_ROOT / "scripts/generate_local5_active_projection_postg0_vectors.py"
        ),
        "projection_trace_reporter": (
            HW_ROOT / "scripts/summarize_local5_gasr2c_fivebank_rtl.py"
        ),
        "projection_trace_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
        ),
        "projection_trace_script": (
            HW_ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"
        ),
    }


def rotating_flat_indices(
    *,
    total_groups: int,
    selected_groups: int,
    sample_id: int,
    stage: int,
    block: int,
) -> list[int]:
    selected = min(total_groups, selected_groups)
    step = max(1, total_groups // 2)
    while math.gcd(step, total_groups) != 1:
        step += 1
    offset = (stage * 131 + block * 17) % total_groups
    return [
        (offset + (sample_id * selected + slot) * step) % total_groups
        for slot in range(selected)
    ]


def validate_release_receipt(identity: dict[str, Any]) -> None:
    receipt_path = Path(str(identity.get("release_receipt", ""))).resolve()
    if (
        not receipt_path.is_file()
        or file_sha256(receipt_path)
        != identity.get("release_receipt_sha256")
    ):
        raise ValueError("run identity release receipt绑定失效")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    status_path = Path(str(receipt.get("status_path", ""))).resolve()
    status = status_path.read_bytes()
    prefix_bytes = int(receipt.get("status_prefix_bytes", -1))
    marker_start = int(receipt.get("marker_start_offset", -1))
    marker_end = int(receipt.get("marker_end_offset", -1))
    marker_line = str(receipt.get("marker_line", ""))
    if (
        receipt.get("schema") != "local5_release_receipt_v2"
        or receipt.get("release_marker")
        != "ALL COMPLETE fullres deploy followup"
        or not 0 <= prefix_bytes <= marker_start < marker_end <= len(status)
        or hashlib.sha256(status[:prefix_bytes]).hexdigest()
        != receipt.get("status_prefix_sha256")
        or status[marker_start:marker_end].decode(
            "utf-8", errors="strict"
        ).rstrip("\n") != marker_line
        or "ALL COMPLETE fullres deploy followup" not in marker_line
        or "H67" not in marker_line
        or "H66d" not in marker_line
        or identity.get("watcher_session_uuid")
        != receipt.get("watcher_session_uuid")
        or receipt.get("ranking_path") != identity.get("ranking")
        or receipt.get("ranking_sha256") != identity.get("ranking_sha256")
        or receipt.get("checkpoint_path") != identity.get("checkpoint")
        or receipt.get("checkpoint_sha256")
        != identity.get("checkpoint_sha256")
        or receipt.get("config_path") != identity.get("config")
        or receipt.get("config_sha256") != identity.get("config_sha256")
        or receipt.get("best_epoch") != identity.get("best_epoch")
    ):
        raise ValueError("run identity release receipt内容无效")


def validate_source_bindings(identity: dict[str, Any]) -> None:
    bindings = identity.get("source_bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("run identity生产软件绑定集合不完整")
    expected = expected_source_binding_paths()
    # Required software bindings must be present. Run-scoped extras such as
    # training_config_identity are allowed when their path+sha are valid.
    missing = set(expected) - set(bindings)
    if missing:
        raise ValueError(
            "run identity生产软件绑定集合不完整: missing="
            + ",".join(sorted(missing))
        )
    for name, expected_path in expected.items():
        binding = bindings[name]
        if not isinstance(binding, dict):
            raise ValueError(f"run identity生产软件绑定失效: {name}")
        path = Path(str(binding.get("path", ""))).resolve()
        if (
            path != expected_path.resolve()
            or not path.is_file()
            or file_sha256(path) != binding.get("sha256")
        ):
            raise ValueError(f"run identity生产软件绑定失效: {name}")
    for name, binding in bindings.items():
        if name in expected:
            continue
        if not isinstance(binding, dict):
            raise ValueError(f"run identity生产软件绑定失效: {name}")
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"run identity生产软件绑定失效: {name}")


def percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return int(ordered[index])


def int_list_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in values) + "\n").encode("utf-8")
    ).hexdigest()


def metric_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def descriptor_dictionary(
    gates: np.ndarray, valid_mask: int
) -> list[tuple[int, int]]:
    values: OrderedDict[int, int] = OrderedDict()
    for role in range(5):
        gate = int(gates[role])
        if ((valid_mask >> role) & 1) == 0 or gate == 0:
            continue
        values[gate] = values.get(gate, 0) | (1 << role)
    return list(values.items())


def validate_descriptor_contract(contract: dict[str, Any]) -> None:
    if contract.get("id") != "qfit_relation_transpose_source_descriptor_v3":
        raise ValueError("DS-FLM部署分析要求v3 source descriptor合同")
    if contract.get("candidate_role_order") != ROLE_ORDER:
        raise ValueError("source descriptor角色顺序必须为self/up/down/left/right")
    if contract.get("source_consumer_relation") != SOURCE_CONSUMER_RELATION:
        raise ValueError("source descriptor角色到consumer的反向关系错误")
    bindings = contract.get("rtl_dependency_bindings", {})
    expected = expected_source_binding_paths()
    if set(bindings) != set(RELATION_BINDING_NAMES):
        raise ValueError("relation-transpose传递依赖绑定集合不完整")
    for name in RELATION_BINDING_NAMES:
        path = Path(str(bindings[name].get("path", ""))).resolve()
        if (
            path != expected[name].resolve()
            or not path.is_file()
            or file_sha256(path) != bindings[name].get("sha256")
        ):
            raise ValueError(f"relation-transpose传递依赖绑定失效: {name}")


def source_destination_id(
    source_id: int,
    role: int,
    *,
    side: int,
) -> int:
    if role == 0:
        return source_id
    if role == 1:
        return source_id + side
    if role == 2:
        return source_id - side
    if role == 3:
        return source_id + 1
    if role == 4:
        return source_id - 1
    raise ValueError(f"非法Local5 role: {role}")


def normalized_destination_updates(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
) -> Counter[tuple[int, int, int]]:
    updates: Counter[tuple[int, int, int]] = Counter()
    for index in range(start, end):
        mode = int(arrays["item_mode_multiset"][index])
        multiplicity = int(arrays["item_multiplicity"][index])
        if mode != 1 or multiplicity <= 0 or multiplicity > 5:
            raise ValueError("destination-major item的mode或multiplicity非法")
        key = (
            int(arrays["item_destination"][index]),
            int(arrays["item_lane_id"][index]),
            int(arrays["item_gate_code"][index]),
        )
        if key[2] <= 0 or key[2] > 511:
            raise ValueError("destination-major gate超出非零9-bit合同")
        updates[key] += multiplicity
    return updates


def normalized_source_updates(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    *,
    tokens: int,
    lanes_width: int,
    side: int,
) -> Counter[tuple[int, int, int]]:
    updates: Counter[tuple[int, int, int]] = Counter()
    for index in range(start, end):
        source_id = int(arrays["descriptor_source_id"][index])
        k_bitmap = int(arrays["descriptor_k_bitmap"][index])
        valid_mask = int(arrays["descriptor_valid_mask"][index])
        gates = arrays["descriptor_incoming_gates"][index]
        for role in range(5):
            if ((valid_mask >> role) & 1) == 0:
                continue
            gate = int(gates[role])
            if gate == 0:
                continue
            destination = source_destination_id(source_id, role, side=side)
            if destination < 0 or destination >= tokens:
                raise ValueError("source-major consumer destination越界")
            for lane in range(lanes_width):
                if (k_bitmap >> lane) & 1:
                    updates[(destination, lane, gate)] += 1
    return updates


def hamming(value: int) -> int:
    return int(value).bit_count()


def sequence_toggles(
    sequence: list[tuple[int, int, int]],
    previous: tuple[int, int, int],
) -> tuple[dict[str, int], tuple[int, int, int]]:
    totals = {"lane": 0, "gate": 0, "mask": 0}
    before = previous
    for after in sequence:
        totals["lane"] += hamming(before[0] ^ after[0])
        totals["gate"] += hamming(before[1] ^ after[1])
        totals["mask"] += hamming(before[2] ^ after[2])
        before = after
    return totals, before


def lane_run_count(sequence: list[tuple[int, int, int]]) -> int:
    runs = 0
    previous: int | None = None
    for lane, _, _ in sequence:
        if previous is None or lane != previous:
            runs += 1
        previous = lane
    return runs


def validate_formal_identity_and_coverage(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    identity_path = Path(
        str(manifest.get("run_identity_file", ""))
    ).resolve()
    identity_hash = str(manifest.get("run_identity_file_sha256", ""))
    if (
        not identity_path.is_file()
        or file_sha256(identity_path) != identity_hash
    ):
        raise ValueError("post_g0 run identity SHA256绑定失效")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != "local5_post_g0_run_identity_v3":
        raise ValueError("post_g0 run identity schema错误")
    if (
        identity.get("config_sha256") != manifest.get("config_sha256")
        or identity.get("checkpoint_sha256")
        != manifest.get("checkpoint_sha256")
    ):
        raise ValueError("run identity与manifest模型绑定不一致")
    contract = manifest["source_descriptor_contract"]
    if (
        identity.get("relation_rtl_sha256")
        != contract.get("rtl_reference_sha256")
    ):
        raise ValueError("run identity与relation RTL绑定不一致")
    validate_release_receipt(identity)
    validate_source_bindings(identity)
    ranking_path = Path(str(identity.get("ranking", "")))
    if (
        not ranking_path.is_file()
        or file_sha256(ranking_path)
        != identity.get("ranking_sha256")
    ):
        raise ValueError("run identity ranking绑定失效")
    samples = int(identity.get("samples", 0))
    groups_per_pair = int(
        identity.get("groups_per_block_sample", 0)
    )
    if (
        samples != 100
        or groups_per_pair < 4
        or identity.get("sampling_id")
        != "coprime_rotating_flat_window_head_v1"
        or identity.get("dataset_sampling_id")
        != "sequence_proportional_temporal_midpoint_v1"
    ):
        raise ValueError("正式run identity采样参数不合格")
    cohort_path = (
        manifest_path.parent / str(manifest.get("cohort_file", ""))
    ).resolve()
    if (
        not cohort_path.is_file()
        or file_sha256(cohort_path) != manifest.get("cohort_file_sha256")
    ):
        raise ValueError("正式cohort文件绑定失效")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    dataset_indices = [int(value) for value in cohort.get("dataset_indices", [])]
    dataset_size = int(cohort.get("dataset_size", 0))
    sequence_counts = cohort.get("sequence_counts", {})
    if (
        cohort.get("schema") != "ordered_trace_cohort_v2"
        or cohort.get("dataset_sampling_id")
        != "sequence_proportional_temporal_midpoint_v1"
        or int(cohort.get("count", 0)) != samples
        or len(dataset_indices) != samples
        or len(set(dataset_indices)) != samples
        or dataset_indices != sorted(dataset_indices)
        or dataset_size < samples
        or any(index < 0 or index >= dataset_size for index in dataset_indices)
        or cohort.get("dataset_indices_sha256")
        != int_list_sha256(dataset_indices)
        or not sequence_counts
        or sum(int(value) for value in sequence_counts.values()) != samples
        or any(int(value) <= 0 for value in sequence_counts.values())
    ):
        raise ValueError("正式cohort不是有效的跨sequence分层抽样")

    groups = manifest["groups"]
    modules = sorted({str(group["module"]) for group in groups})
    block_pairs = {
        (int(group["stage"]), int(group["block"])) for group in groups
    }
    pair_counts = Counter(
        (str(group["module"]), int(group["sample"]))
        for group in groups
    )
    expected_pairs = {
        (module, sample)
        for module in modules
        for sample in range(samples)
    }
    if (
        len(modules) != 12
        or block_pairs != POST_G0_BLOCK_PAIRS
        or set(pair_counts) != expected_pairs
        or any(count != groups_per_pair for count in pair_counts.values())
    ):
        raise ValueError("正式trace未覆盖100 sample×12 block")
    module_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if (
            group.get("selection")
            != "coprime_rotating_flat_window_head_v1"
        ):
            raise ValueError("正式trace混入非轮转采样group")
        module_rows[str(group["module"])].append(group)
    coverage: dict[str, Any] = {}
    for module, rows in sorted(module_rows.items()):
        head_counts = {int(row["heads"]) for row in rows}
        total_counts = {
            int(row["heads"]) * int(row["batch_windows"])
            for row in rows
        }
        if len(head_counts) != 1 or len(total_counts) != 1:
            raise ValueError("同一module的head/window形状不稳定")
        heads = next(iter(head_counts))
        total_groups = next(iter(total_counts))
        observed_heads = {int(row["head"]) for row in rows}
        observed_flat = {int(row["flat_group"]) for row in rows}
        expected_flat = min(
            total_groups,
            samples * min(groups_per_pair, total_groups),
        )
        if observed_heads != set(range(heads)):
            raise ValueError(f"{module}未覆盖全部head")
        if len(observed_flat) != expected_flat:
            raise ValueError(f"{module}轮转window-head覆盖不完整")
        stage = int(rows[0]["stage"])
        block = int(rows[0]["block"])
        if (
            (stage, block) not in POST_G0_BLOCK_PAIRS
            or any(
                int(row["stage"]) != stage
                or int(row["block"]) != block
                for row in rows
            )
        ):
            raise ValueError(f"{module} stage/block身份不稳定")
        rows_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            flat = int(row["flat_group"])
            row_heads = int(row["heads"])
            row_total = int(row["batch_windows"]) * row_heads
            if (
                flat
                != int(row["window"]) * row_heads + int(row["head"])
                or not 0 <= flat < row_total
            ):
                raise ValueError(f"{module} flat/window/head关系错误")
            rows_by_sample[int(row["sample"])].append(row)
        for sample in range(samples):
            sample_rows = rows_by_sample.get(sample, [])
            sample_totals = {
                int(row["batch_windows"]) * int(row["heads"])
                for row in sample_rows
            }
            if len(sample_totals) != 1:
                raise ValueError(f"{module} sample{sample}形状不稳定")
            sample_total = next(iter(sample_totals))
            expected_indices = rotating_flat_indices(
                total_groups=sample_total,
                selected_groups=groups_per_pair,
                sample_id=sample,
                stage=stage,
                block=block,
            )
            if [
                int(row["flat_group"]) for row in sample_rows
            ] != expected_indices:
                raise ValueError(f"{module} sample{sample}采样索引不匹配")
        coverage[module] = {
            "heads": heads,
            "observed_unique_flat_groups": len(observed_flat),
            "total_flat_groups": total_groups,
        }
    return {
        "run_identity": str(identity_path),
        "run_identity_sha256": identity_hash,
        "samples": samples,
        "blocks": len(modules),
        "groups_per_block_sample": groups_per_pair,
        "groups": len(groups),
        "module_coverage": coverage,
        "manifest": str(manifest_path.resolve()),
        "dataset_sampling_id": cohort["dataset_sampling_id"],
        "dataset_size": dataset_size,
        "dataset_indices_sha256": cohort["dataset_indices_sha256"],
        "sequence_counts": sequence_counts,
    }


def analyze(
    manifest_path: Path,
    *,
    require_formal: bool = True,
) -> dict[str, Any]:
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = raw_manifest.get("source_descriptor_contract", {})
    validate_descriptor_contract(contract)
    rtl_path = Path(str(contract.get("rtl_reference", ""))).resolve()
    rtl_hash = str(contract.get("rtl_reference_sha256", ""))
    if not rtl_path.is_file() or file_sha256(rtl_path) != rtl_hash:
        raise ValueError("relation-transpose RTL SHA256绑定失效")
    manifest, arrays = load_trace(manifest_path)
    required = {
        "group_offsets",
        "item_mode_multiset",
        "item_gate_code",
        "item_lane_id",
        "item_multiplicity",
        "item_destination",
        "descriptor_group_offsets",
        "descriptor_source_id",
        "descriptor_source_plane",
        "descriptor_source_y",
        "descriptor_source_x",
        "descriptor_k_bitmap",
        "descriptor_incoming_gates",
        "descriptor_valid_mask",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"descriptor trace缺少数组: {missing}")
    qualification = manifest.get("qualification", {})
    formal_coverage: dict[str, Any] | None = None
    if require_formal:
        if manifest.get("evidence_level") != "post_g0":
            raise ValueError("DS-FLM正式部署分析只接受post_g0 trace")
        if (
            qualification.get("schema")
            != "local5_post_g0_qualification_v1"
            or qualification.get("qualified") is not True
            or not all(qualification.get("checks", {}).values())
        ):
            raise ValueError("DS-FLM正式部署分析要求qualification全通过")
        formal_coverage = validate_formal_identity_and_coverage(
            manifest_path,
            manifest,
        )
    offsets = arrays["descriptor_group_offsets"]
    item_offsets = arrays["group_offsets"]
    if offsets.ndim != 1 or len(offsets) != len(manifest["groups"]) + 1:
        raise ValueError("descriptor group offset数量必须为groups+1")
    if len(offsets) == 0 or int(offsets[0]) != 0:
        raise ValueError("descriptor group offset必须从0开始")
    if bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("descriptor group offset必须单调不减")
    if item_offsets.ndim != 1 or len(item_offsets) != len(manifest["groups"]) + 1:
        raise ValueError("destination group offset数量必须为groups+1")
    if int(item_offsets[0]) != 0 or bool(np.any(item_offsets[1:] < item_offsets[:-1])):
        raise ValueError("destination group offset必须从0开始且单调不减")
    count = len(arrays["descriptor_source_id"])
    if int(offsets[-1]) != count:
        raise ValueError("descriptor group offset与payload数量不一致")
    descriptor_arrays = {
        "descriptor_source_id",
        "descriptor_source_plane",
        "descriptor_source_y",
        "descriptor_source_x",
        "descriptor_k_bitmap",
        "descriptor_incoming_gates",
        "descriptor_valid_mask",
    }
    for name in descriptor_arrays:
        if len(arrays[name]) != count:
            raise ValueError(f"{name}长度与descriptor数量不一致")
    item_count = len(arrays["item_gate_code"])
    if int(item_offsets[-1]) != item_count:
        raise ValueError("destination group offset与item数量不一致")
    for name in (
        "item_mode_multiset",
        "item_lane_id",
        "item_multiplicity",
        "item_destination",
    ):
        if arrays[name].ndim != 1 or len(arrays[name]) != item_count:
            raise ValueError(f"{name}长度与destination item数量不一致")
    if arrays["descriptor_incoming_gates"].shape != (count, 5):
        raise ValueError("descriptor incoming gate必须为[N,5]")
    for name in (
        "descriptor_source_id",
        "descriptor_source_plane",
        "descriptor_source_y",
        "descriptor_source_x",
        "descriptor_k_bitmap",
        "descriptor_valid_mask",
    ):
        if arrays[name].ndim != 1:
            raise ValueError(f"{name}必须是一维数组")

    lane_counts: list[int] = []
    gate_counts: list[int] = []
    term_counts: list[int] = []
    lane_major_lane_runs: list[int] = []
    gate_major_lane_runs: list[int] = []
    nonempty = 0
    invariant_descriptors = 0
    equivalent_groups = 0
    equivalent_updates = 0
    group_rows = []
    stage_rows: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for group_index, group in enumerate(manifest["groups"]):
        start = int(offsets[group_index])
        end = int(offsets[group_index + 1])
        item_start = int(item_offsets[group_index])
        item_end = int(item_offsets[group_index + 1])
        tokens = int(group["tokens"])
        lanes_width = int(group["lanes"])
        if end - start != tokens:
            raise ValueError("每组descriptor数量必须等于group tokens")
        source_ids = arrays["descriptor_source_id"][start:end]
        if source_ids.tolist() != list(range(tokens)):
            raise ValueError("descriptor source_id必须完整、升序且无重复")
        if require_formal and (
            tokens != 450
            or lanes_width != 32
            or int(group["time_planes"]) != 2
            or int(group["plane_tokens"]) != 225
            or int(group["spatial_side"]) != 15
        ):
            raise ValueError("正式descriptor group不是T450×32/W15")
        destination_updates = normalized_destination_updates(
            arrays, item_start, item_end
        )
        source_updates = normalized_source_updates(
            arrays,
            start,
            end,
            tokens=tokens,
            lanes_width=lanes_width,
            side=max(1, int(group["spatial_side"])),
        )
        if source_updates != destination_updates:
            missing_updates = destination_updates - source_updates
            extra_updates = source_updates - destination_updates
            raise ValueError(
                "source-major与destination-major更新多重集不等价: "
                f"group={group_index}, missing={missing_updates.most_common(3)}, "
                f"extra={extra_updates.most_common(3)}"
            )
        equivalent_groups += 1
        equivalent_updates += sum(destination_updates.values())
        previous = (0, 0, 0)
        lane_hamming = {"lane": 0, "gate": 0, "mask": 0}
        gate_hamming = {"lane": 0, "gate": 0, "mask": 0}
        group_terms = 0
        group_nonempty = 0
        for index in range(start, end):
            local_source = index - start
            plane_tokens = int(group["plane_tokens"])
            side = int(group["spatial_side"])
            if side > 0 and side * side == plane_tokens:
                expected_plane = local_source // plane_tokens
                spatial = local_source % plane_tokens
                expected_y = spatial // side
                expected_x = spatial % side
            else:
                expected_plane = 0
                expected_y = 0
                expected_x = local_source
            if (
                int(arrays["descriptor_source_plane"][index])
                != expected_plane
                or int(arrays["descriptor_source_y"][index])
                != expected_y
                or int(arrays["descriptor_source_x"][index])
                != expected_x
            ):
                raise ValueError("descriptor plane/y/x与source_id不一致")
            k_bitmap = int(arrays["descriptor_k_bitmap"][index])
            if k_bitmap >> lanes_width:
                raise ValueError("descriptor K bitmap超出lane宽度")
            valid_mask = int(arrays["descriptor_valid_mask"][index])
            if valid_mask >> 5:
                raise ValueError("descriptor valid mask超出五角色")
            if require_formal:
                expected_valid_mask = (
                    1
                    | ((expected_y < side - 1) << 1)
                    | ((expected_y > 0) << 2)
                    | ((expected_x < side - 1) << 3)
                    | ((expected_x > 0) << 4)
                )
                if valid_mask != expected_valid_mask:
                    raise ValueError(
                        "source-major valid mask不符合W15 "
                        "N/S/E/W consumer几何"
                    )
            gates = arrays["descriptor_incoming_gates"][index]
            if bool(np.any(gates > 511)):
                raise ValueError("descriptor gate超出9-bit")
            for role in range(5):
                if ((valid_mask >> role) & 1) == 0 and int(gates[role]):
                    raise ValueError("无效descriptor角色携带非零gate")
            lanes = [
                lane
                for lane in range(lanes_width)
                if (k_bitmap >> lane) & 1
            ]
            dictionary = descriptor_dictionary(
                gates,
                valid_mask,
            )
            lane_count = len(lanes)
            gate_count = len(dictionary)
            term_count = lane_count * gate_count
            lane_counts.append(lane_count)
            gate_counts.append(gate_count)
            term_counts.append(term_count)
            stage = int(group["stage"])
            stage_rows[stage]["lane"].append(lane_count)
            stage_rows[stage]["gate"].append(gate_count)
            stage_rows[stage]["term"].append(term_count)
            if term_count:
                nonempty += 1
                group_nonempty += 1
            group_terms += term_count

            lane_major = [
                (lane, gate, mask)
                for lane in lanes
                for gate, mask in dictionary
            ]
            gate_major = [
                (lane, gate, mask)
                for gate, mask in dictionary
                for lane in lanes
            ]
            if set(lane_major) != set(gate_major):
                raise ValueError("两模式term集合不一致")
            if lane_major:
                if (
                    lane_major[0] != gate_major[0]
                    or lane_major[-1] != gate_major[-1]
                ):
                    raise ValueError("两模式首尾状态不一致")
                invariant_descriptors += 1
            lane_major_lane_runs.append(lane_run_count(lane_major))
            gate_major_lane_runs.append(lane_run_count(gate_major))
            lane_delta, lane_last = sequence_toggles(
                lane_major, previous
            )
            gate_delta, gate_last = sequence_toggles(
                gate_major, previous
            )
            if lane_last != gate_last:
                raise ValueError("两模式末状态不一致")
            for field in lane_hamming:
                lane_hamming[field] += lane_delta[field]
                gate_hamming[field] += gate_delta[field]
            previous = lane_last
        group_rows.append(
            {
                "sample": int(group["sample"]),
                "stage": int(group["stage"]),
                "block": int(group["block"]),
                "window": int(group["window"]),
                "head": int(group["head"]),
                "descriptors": end - start,
                "nonempty_descriptors": group_nonempty,
                "terms": group_terms,
                "lane_major_hamming": lane_hamming,
                "gate_major_hamming": gate_hamming,
            }
        )

    return {
        "schema": "ds_flm_post_g0_descriptor_analysis_v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "evidence_level": manifest["evidence_level"],
        "formal_qualification": bool(
            require_formal and qualification.get("qualified")
        ),
        "run_identity_file_sha256": manifest.get(
            "run_identity_file_sha256"
        ),
        "formal_coverage": formal_coverage,
        "groups": len(group_rows),
        "descriptors": count,
        "nonempty_descriptors": nonempty,
        "nonempty_ratio": nonempty / count if count else 0.0,
        "state_invariant_nonempty_descriptors": invariant_descriptors,
        "source_destination_equivalence": {
            "passed": True,
            "groups": equivalent_groups,
            "expanded_updates": equivalent_updates,
        },
        "active_lanes": metric_summary(lane_counts),
        "unique_gates": metric_summary(gate_counts),
        "terms": metric_summary(term_counts),
        "lane_major_hamming": {
            field: sum(
                row["lane_major_hamming"][field] for row in group_rows
            )
            for field in ("lane", "gate", "mask")
        },
        "gate_major_hamming": {
            field: sum(
                row["gate_major_hamming"][field] for row in group_rows
            )
            for field in ("lane", "gate", "mask")
        },
        "lane_major_within_descriptor_lane_runs": sum(
            lane_major_lane_runs
        ),
        "gate_major_within_descriptor_lane_runs": sum(
            gate_major_lane_runs
        ),
        "stage_summary": {
            str(stage): {
                name: metric_summary(values)
                for name, values in metrics.items()
            }
            for stage, metrics in sorted(stage_rows.items())
        },
        "group_rows": group_rows,
        "limitations": [
            "ordered groups是均匀抽样，不是完整workload总量",
            "本报告是descriptor结构与控制活动，不是功耗",
            "未标定SRAM、priority encoder、cache和selector能量",
            "不从非连续抽样group推断跨window LRU命中率",
            "within-descriptor lane run未合并descriptor边界的同lane状态",
        ],
    }


def write_report(value: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DS-FLM Fullres Post-G0 Descriptor统计",
        "",
        "## 1. 证据边界",
        "",
        f"- evidence level：`{value['evidence_level']}`；",
        f"- sampled groups：{value['groups']}；",
        f"- descriptors：{value['descriptors']}；",
        "- ordered groups为均匀抽样，不是完整workload总量；",
        "- 本报告只统计结构与切换，不把Hamming解释成功耗。",
        "",
        "## 2. Descriptor分布",
        "",
        f"- 非空比例：{value['nonempty_ratio']:.4%}；",
        "",
        "| 指标 | mean | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("active lane", "active_lanes"),
        ("unique gate", "unique_gates"),
        ("term", "terms"),
    ):
        row = value[key]
        lines.append(
            f"| {label} | {row['mean']:.4f} | {row['p50']} | "
            f"{row['p95']} | {row['p99']} | {row['max']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 双顺序结构活动",
            "",
            "| 顺序 | lane Hamming | gate Hamming | mask Hamming | "
            "descriptor内lane runs |",
            "|---|---:|---:|---:|---:|",
            (
                "| lane-major | "
                f"{value['lane_major_hamming']['lane']} | "
                f"{value['lane_major_hamming']['gate']} | "
                f"{value['lane_major_hamming']['mask']} | "
                f"{value['lane_major_within_descriptor_lane_runs']} |"
            ),
            (
                "| gate-major | "
                f"{value['gate_major_hamming']['lane']} | "
                f"{value['gate_major_hamming']['gate']} | "
                f"{value['gate_major_hamming']['mask']} | "
                f"{value['gate_major_within_descriptor_lane_runs']} |"
            ),
            f"- 状态不变量验证的非空descriptor："
            f"{value['state_invariant_nonempty_descriptors']}；",
            "",
            "Hamming按每个抽样group从全零状态开始，并保留descriptor之间的"
            "连续状态；lane run是descriptor内理论调度段数。没有集成SAIF与"
            "真实存储系数前，不得写成能耗收益。",
            "",
            "## 4. 分stage统计",
            "",
            "| stage | active lane mean/p95 | unique gate mean/p95 | "
            "term mean/p95 |",
            "|---:|---:|---:|---:|",
            "",
        ]
    )
    for stage, summary in value["stage_summary"].items():
        lines.append(
            f"| {stage} | {summary['lane']['mean']:.4f}/"
            f"{summary['lane']['p95']} | "
            f"{summary['gate']['mean']:.4f}/"
            f"{summary['gate']['p95']} | "
            f"{summary['term']['mean']:.4f}/"
            f"{summary['term']['p95']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 结论边界",
            "",
            "- 若两种顺序只是在不同字段上互换Hamming，不得直接晋级双模式硬件；",
            "- 只有代入综合后电容、真实SRAM访问和selector开销后仍改善EDP，"
            "双模式才可作为论文贡献；",
            "- 本报告不推断跨非连续group的cache/LRU命中率。",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.manifest)
    write_report(value, args.output_dir)
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

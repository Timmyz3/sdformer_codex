#!/usr/bin/env python3
"""在不改动既有 post-G0 profiler 的前提下导出同窗全 head trace。"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import profile_local5_hardware_features as base


JOINT_SAMPLING_ID = "uniform_plan_window_all_heads_v1"
JOINT_WINDOWS_PER_BLOCK_SAMPLE = 1
MAX_STAGE_HEADS = 24
JOINT_SAMPLING_SEED = 20260809
EXPECTED_STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
EXPECTED_STAGE_WINDOWS = {0: 440, 1: 120, 2: 30, 3: 10}
ACTIVE_SELECTION_PLAN: dict[tuple[int, int, int], int] | None = None
ACTIVE_SELECTION_PLAN_SHA256: str | None = None
ACTIVE_SELECTION_PLAN_PATH: Path | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_joint_window(
    *,
    batch_windows: int,
    sample_id: int,
    stage: int,
    block: int,
    seed: int = JOINT_SAMPLING_SEED,
) -> int:
    """用与数据无关的固定 PRF seed 均匀抽取一个 window。"""

    if batch_windows <= 0:
        raise ValueError("batch_windows必须为正数")
    material = f"{seed}:{sample_id}:{stage}:{block}".encode("ascii")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(local_seed).randrange(batch_windows)


def expected_joint_source_paths() -> dict[str, Path]:
    overlay = base.EXP_ROOT / "overlay/models/STSwinNet_SNN"
    baseline = base.REPO_ROOT / "third_party/SDformerFlow"
    return {
        "runner": base.HW_ROOT / "scripts/run_local5_joint_head_profile.py",
        "joint_profiler": Path(__file__).resolve(),
        "base_profiler": base.HW_ROOT / "scripts/profile_local5_hardware_features.py",
        "network_profiler": base.EXP_ROOT / "entrypoints/profile_nts11_hardware_p0.py",
        "attention_impl": overlay / "bsa_attention.py",
        "checkpoint_loader": overlay / "h9_load_audit.py",
        "model_impl": baseline / "models/STSwinNet_SNN/Spiking_STSwinNet.py",
        "dataset_impl": baseline / "DSEC_dataloader/DSEC_dataset_lite.py",
        "trace_contract": base.HW_ROOT / "scripts/et3_ordered_trace_replay.py",
        "projection_quantizer": base.EXP_ROOT / "entrypoints/h67_bit_trace.py",
    }


def validate_joint_plan_freeze_receipt(
    identity: dict[str, Any],
    *,
    selection_path: Path,
    selection_sha: str,
    plan: dict[str, Any],
    runner_binding: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(
        str(identity.get("selection_plan_freeze_receipt", ""))
    ).resolve()
    receipt_sha = str(identity.get("selection_plan_freeze_receipt_sha256", ""))
    if not receipt_path.is_file() or file_sha256(receipt_path) != receipt_sha:
        raise ValueError("joint selection-plan freeze receipt绑定失效")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    blob_oid = str(receipt.get("selection_plan_git_blob", ""))
    blob = subprocess.run(
        ["git", "cat-file", "blob", blob_oid],
        cwd=base.HW_ROOT,
        capture_output=True,
        check=False,
    )
    if (
        receipt.get("schema") != "local5_joint_trace_plan_freeze_receipt_v1"
        or receipt.get("status") != "LOCAL_BYTE_ANCHOR_NOT_EXTERNAL_TIMESTAMP"
        or Path(str(receipt.get("selection_plan", ""))).resolve()
        != selection_path.resolve()
        or receipt.get("selection_plan_sha256") != selection_sha
        or Path(str(receipt.get("generator", ""))).resolve()
        != Path(str(runner_binding.get("path", ""))).resolve()
        or receipt.get("generator_sha256") != runner_binding.get("sha256")
        or receipt.get("sampling_id") != JOINT_SAMPLING_ID
        or receipt.get("sampling_seed") != JOINT_SAMPLING_SEED
        or receipt.get("cohort_sha256") != plan.get("cohort_sha256")
        or receipt.get("records") != len(plan.get("records", []))
        or not re.fullmatch(r"[0-9a-f]{40}", blob_oid)
        or blob.returncode != 0
        or blob.stdout != selection_path.read_bytes()
    ):
        raise ValueError("joint selection-plan freeze receipt合同失效")
    return receipt


def active_joint_window(
    *, batch_windows: int, sample_id: int, stage: int, block: int
) -> int:
    if ACTIVE_SELECTION_PLAN is None:
        return uniform_joint_window(
            batch_windows=batch_windows,
            sample_id=sample_id,
            stage=stage,
            block=block,
        )
    key = (sample_id, stage, block)
    if key not in ACTIVE_SELECTION_PLAN:
        raise ValueError(f"selection plan缺少记录: {key}")
    window = int(ACTIVE_SELECTION_PLAN[key])
    if not 0 <= window < batch_windows:
        raise ValueError(f"selection plan window越界: {key} -> {window}")
    return window


class JointHeadOrderedTermTraceSink(base.OrderedTermTraceSink):
    """复用原 trace 编码，只把采样单位改成同一 window 的全部 head。"""

    def capture(
        self,
        *,
        name: str,
        stage: int,
        block: int,
        sample_id: int,
        k_candidates: torch.Tensor,
        valid: torch.Tensor,
        gate_code: torch.Tensor,
        neighbor_index: torch.Tensor,
        q_event: torch.Tensor | None = None,
    ) -> None:
        batch_windows, heads, _, _, _ = k_candidates.shape
        if heads > MAX_STAGE_HEADS:
            raise ValueError(f"head数{heads}超过正式合同上限{MAX_STAGE_HEADS}")
        window = active_joint_window(
            batch_windows=batch_windows,
            sample_id=sample_id,
            stage=stage,
            block=block,
        )
        before = len(self.groups)
        saved_groups = self.groups_per_block_sample
        saved_selector = base.rotating_flat_indices
        self.groups_per_block_sample = heads
        base.rotating_flat_indices = lambda **kwargs: list(
            range(int(kwargs["total_groups"]))
        )
        try:
            super().capture(
                name=name,
                stage=stage,
                block=block,
                sample_id=sample_id,
                k_candidates=k_candidates[window : window + 1],
                valid=valid,
                gate_code=gate_code[window : window + 1],
                neighbor_index=neighbor_index,
                q_event=(
                    q_event[window : window + 1]
                    if q_event is not None
                    else None
                ),
            )
        finally:
            self.groups_per_block_sample = saved_groups
            base.rotating_flat_indices = saved_selector
        rows = self.groups[before:]
        if len(rows) != heads or [int(row["head"]) for row in rows] != list(
            range(heads)
        ):
            raise RuntimeError("同窗全head采样不完整")
        for row in rows:
            head = int(row["head"])
            row["window"] = window
            row["flat_group"] = window * heads + head
            row["batch_windows"] = batch_windows
            row["selection"] = JOINT_SAMPLING_ID

    def write(self, **kwargs: Any) -> tuple[Path, Path]:
        manifest_path, payload_path = super().write(**kwargs)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sampling"] = {
            "method": JOINT_SAMPLING_ID,
            "joint_windows_per_block_sample": (
                JOINT_WINDOWS_PER_BLOCK_SAMPLE
            ),
            "groups_rule": "all heads of one precommitted uniform window",
            "selection_plan": str(ACTIVE_SELECTION_PLAN_PATH),
            "selection_plan_sha256": str(ACTIVE_SELECTION_PLAN_SHA256),
            "seed": JOINT_SAMPLING_SEED,
            "window_inclusion_probability": "1 / batch_windows",
            "horvitz_thompson_weight": "batch_windows",
            "performance_scope": (
                "same-window all-head joint groups; sampled windows, "
                "not full workload totals"
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, payload_path


def joint_post_g0_qualification(
    groups: list[dict[str, Any]],
    *,
    processed_samples: int,
    attached_blocks: int,
    groups_per_block_sample: int,
    run_identity_bound: bool,
) -> dict[str, Any]:
    """Fail-closed 验证每个 block/sample 恰好只有一个窗且覆盖全部 head。"""

    modules = sorted({str(group["module"]) for group in groups})
    block_pairs = {
        (int(group["stage"]), int(group["block"])) for group in groups
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped[(str(group["module"]), int(group["sample"]))].append(group)

    expected_keys = {
        (module, sample)
        for module in modules
        for sample in range(processed_samples)
    }
    complete_pairs = set(grouped) == expected_keys
    exact_joint_windows = complete_pairs
    expected_groups = 0
    module_coverage: dict[str, Any] = {}
    for module in modules:
        module_rows = [row for row in groups if str(row["module"]) == module]
        if not module_rows:
            exact_joint_windows = False
            continue
        heads_values = {int(row["heads"]) for row in module_rows}
        windows_values = {int(row["batch_windows"]) for row in module_rows}
        stage_values = {int(row["stage"]) for row in module_rows}
        block_values = {int(row["block"]) for row in module_rows}
        if not (
            len(heads_values)
            == len(windows_values)
            == len(stage_values)
            == len(block_values)
            == 1
        ):
            exact_joint_windows = False
            continue
        heads = next(iter(heads_values))
        batch_windows = next(iter(windows_values))
        stage = next(iter(stage_values))
        block = next(iter(block_values))
        expected_groups += processed_samples * heads
        module_ok = True
        for sample in range(processed_samples):
            rows = grouped.get((module, sample), [])
            expected_window = active_joint_window(
                batch_windows=batch_windows,
                sample_id=sample,
                stage=stage,
                block=block,
            )
            module_ok &= (
                heads == EXPECTED_STAGE_HEADS.get(stage)
                and batch_windows == EXPECTED_STAGE_WINDOWS.get(stage)
                and len(rows) == heads
                and [int(row["head"]) for row in rows] == list(range(heads))
                and {int(row["window"]) for row in rows}
                == {expected_window}
                and all(
                    int(row["flat_group"])
                    == expected_window * heads + int(row["head"])
                    for row in rows
                )
            )
        exact_joint_windows &= module_ok
        module_coverage[module] = {
            "heads": heads,
            "batch_windows": batch_windows,
            "samples": processed_samples,
            "groups": len(module_rows),
            "same_window_all_heads_ok": bool(module_ok),
        }

    checks = {
        "run_identity_bound": run_identity_bound,
        "processed_samples_100": processed_samples == base.POST_G0_SAMPLES,
        "attached_blocks_12": attached_blocks == base.POST_G0_BLOCKS,
        "captured_modules_12": len(modules) == base.POST_G0_BLOCKS,
        "exact_target_block_set": block_pairs == set(base.POST_G0_BLOCK_PAIRS),
        "exact_stage_head_mapping": all(
            int(group["heads"]) == EXPECTED_STAGE_HEADS.get(int(group["stage"]))
            for group in groups
        ),
        "exact_stage_window_mapping": all(
            int(group["batch_windows"])
            == EXPECTED_STAGE_WINDOWS.get(int(group["stage"]))
            for group in groups
        ),
        "implementation_bound_24": groups_per_block_sample == MAX_STAGE_HEADS,
        "module_sample_pair_coverage": complete_pairs,
        "exact_same_window_all_heads": exact_joint_windows,
        "expected_group_count": len(groups) == expected_groups,
        "shape_t450_l32": bool(groups)
        and all(
            int(group["tokens"]) == base.POST_G0_TOKENS
            and int(group["lanes"]) == base.POST_G0_LANES
            for group in groups
        ),
        "sampling_contract": all(
            group.get("selection") == JOINT_SAMPLING_ID for group in groups
        ),
    }
    return {
        "schema": "local5_post_g0_joint_head_qualification_v1",
        "qualified": all(checks.values()),
        "checks": checks,
        "processed_samples": processed_samples,
        "attached_blocks": attached_blocks,
        "captured_modules": len(modules),
        "captured_groups": len(groups),
        "expected_groups": expected_groups,
        "module_coverage": module_coverage,
    }


def load_joint_run_identity(
    path: Path,
    *,
    config: Path,
    checkpoint: Path,
    samples: int,
    groups_per_block_sample: int,
) -> dict[str, Any]:
    global ACTIVE_SELECTION_PLAN
    global ACTIVE_SELECTION_PLAN_PATH
    global ACTIVE_SELECTION_PLAN_SHA256
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "local5_joint_head_run_identity_v1",
        "config_sha256": file_sha256(config),
        "checkpoint_sha256": file_sha256(checkpoint),
        "samples": samples,
        "groups_per_block_sample": groups_per_block_sample,
        "joint_windows_per_block_sample": JOINT_WINDOWS_PER_BLOCK_SAMPLE,
        "sampling_id": JOINT_SAMPLING_ID,
        "sampling_seed": JOINT_SAMPLING_SEED,
        "dataset_sampling_id": base.POST_G0_DATASET_SAMPLING_ID,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"joint run identity字段不匹配: {name}")
    if Path(str(value.get("config", ""))).resolve() != config.resolve():
        raise ValueError("joint run identity config路径不匹配")
    if Path(str(value.get("checkpoint", ""))).resolve() != checkpoint.resolve():
        raise ValueError("joint run identity checkpoint路径不匹配")
    bindings = value.get("source_bindings")
    expected_paths = expected_joint_source_paths()
    if not isinstance(bindings, dict) or set(bindings) != set(expected_paths):
        raise ValueError("joint run identity缺少source bindings")
    for name, source in expected_paths.items():
        binding = bindings[name]
        if not isinstance(binding, dict):
            raise ValueError(f"joint source binding格式错误: {name}")
        if (
            Path(str(binding.get("path", ""))).resolve() != source.resolve()
            or not source.is_file()
            or binding.get("sha256") != file_sha256(source)
        ):
            raise ValueError(f"joint source binding失效: {name}")
    ranking = Path(str(value.get("ranking", "")))
    receipt = Path(str(value.get("release_receipt", "")))
    if (
        not ranking.is_file()
        or value.get("ranking_sha256") != file_sha256(ranking)
        or not receipt.is_file()
        or value.get("release_receipt_sha256") != file_sha256(receipt)
    ):
        raise ValueError("joint ranking/release receipt绑定失效")
    receipt_value = base.validate_release_receipt(
        receipt, str(value["release_receipt_sha256"])
    )
    receipt_expected = {
        "ranking_path": str(ranking.resolve()),
        "ranking_sha256": value["ranking_sha256"],
        "best_epoch": value["best_epoch"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": value["checkpoint_sha256"],
        "config_path": str(config.resolve()),
        "config_sha256": value["config_sha256"],
    }
    if any(receipt_value.get(name) != expected for name, expected in receipt_expected.items()):
        raise ValueError("joint identity与release receipt交叉绑定失效")
    selection_path = Path(str(value.get("selection_plan", ""))).resolve()
    selection_sha = str(value.get("selection_plan_sha256", ""))
    if not selection_path.is_file() or file_sha256(selection_path) != selection_sha:
        raise ValueError("joint selection plan绑定失效")
    plan = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != "local5_uniform_joint_window_plan_v1"
        or plan.get("seed") != JOINT_SAMPLING_SEED
        or plan.get("cohort_sha256") != value.get("cohort_sha256")
        or len(plan.get("records") or []) != base.POST_G0_SAMPLES * base.POST_G0_BLOCKS
    ):
        raise ValueError("joint selection plan合同失效")
    validate_joint_plan_freeze_receipt(
        value,
        selection_path=selection_path,
        selection_sha=selection_sha,
        plan=plan,
        runner_binding=bindings["runner"],
    )
    parsed: dict[tuple[int, int, int], int] = {}
    for row in plan["records"]:
        key = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        stage = key[1]
        batch_windows = EXPECTED_STAGE_WINDOWS.get(stage)
        if (
            key in parsed
            or int(row["heads"]) != EXPECTED_STAGE_HEADS.get(stage)
            or int(row["batch_windows"]) != batch_windows
            or not 0 <= int(row["window"]) < batch_windows
            or float(row["inclusion_probability"]) != 1.0 / batch_windows
            or float(row["analysis_weight"]) != float(batch_windows)
        ):
            raise ValueError(f"joint selection plan记录失效: {key}")
        parsed[key] = int(row["window"])
    expected_keys = {
        (sample, stage, block)
        for sample in range(base.POST_G0_SAMPLES)
        for stage, block in base.POST_G0_BLOCK_PAIRS
    }
    if set(parsed) != expected_keys:
        raise ValueError("joint selection plan覆盖不完整")
    ACTIVE_SELECTION_PLAN = parsed
    ACTIVE_SELECTION_PLAN_PATH = selection_path
    ACTIVE_SELECTION_PLAN_SHA256 = selection_sha
    return value


def main() -> int:
    base.POST_G0_SAMPLING_ID = JOINT_SAMPLING_ID
    base.OrderedTermTraceSink = JointHeadOrderedTermTraceSink
    base.post_g0_qualification = joint_post_g0_qualification
    base.load_post_g0_run_identity = load_joint_run_identity
    result = base.main()
    if ACTIVE_SELECTION_PLAN is None or ACTIVE_SELECTION_PLAN_SHA256 is None:
        raise RuntimeError("正式joint profile未加载selection plan")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

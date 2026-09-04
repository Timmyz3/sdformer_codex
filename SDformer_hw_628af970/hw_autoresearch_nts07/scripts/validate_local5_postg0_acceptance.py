#!/usr/bin/env python3
"""Fail-closed验收Local5 post-G0 profile、replay与descriptor报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_ds_flm_descriptor_manifest import analyze
from et3_ordered_trace_replay import file_sha256, load_trace
from replay_local5_frontier_trace import replay as replay_trace
from verify_local5_theta_folded_projection_contract import (
    TOPOLOGY_CONTRACT,
    verify_contract,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点不是object: {path}")
    return value


def validate_threshold_semantics(manifest: dict[str, Any]) -> dict[str, Any]:
    semantics = manifest.get("threshold_training_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("ordered manifest缺少ATLIF threshold训练/部署语义")
    expected = {
        "threshold_modes": ["official_atlif"],
        "homeostatic_freeze_after_step": 1224,
        "homeostatic_update_frozen_after_boundary": True,
        "optimizer_gradient_freeze_enabled": False,
        "official_atlif_runtime_clamp_applied": False,
        "inference_threshold_source": "checkpoint_static_parameter",
    }
    mismatches = {
        key: {"expected": value, "observed": semantics.get(key)}
        for key, value in expected.items()
        if semantics.get(key) != value
    }
    threshold_lr = semantics.get("optimizer_threshold_lr")
    if threshold_lr is None or abs(float(threshold_lr) - 5.0e-6) > 1.0e-12:
        mismatches["optimizer_threshold_lr"] = {
            "expected": 5.0e-6,
            "observed": threshold_lr,
        }
    if mismatches:
        raise ValueError(f"ATLIF threshold训练/部署语义漂移: {mismatches}")
    return semantics


def validate(
    *,
    manifest_path: Path,
    replay_report_path: Path,
    descriptor_report_path: Path,
    run_identity_path: Path,
) -> dict[str, Any]:
    manifest, arrays = load_trace(manifest_path)
    analysis = analyze(manifest_path)
    replay_recomputed = replay_trace(manifest, arrays)
    replay_recomputed["manifest"] = str(manifest_path.resolve())
    replay_recomputed["manifest_sha256"] = file_sha256(manifest_path)
    replay_recomputed["run_identity_file_sha256"] = manifest.get(
        "run_identity_file_sha256"
    )
    replay_report = load_json(replay_report_path)
    descriptor = load_json(descriptor_report_path)
    identity = load_json(run_identity_path)
    threshold_semantics = validate_threshold_semantics(manifest)
    manifest_hash = file_sha256(manifest_path)
    identity_hash = file_sha256(run_identity_path)
    projection_manifest_path = (
        manifest_path.parent / str(manifest.get("projection_contract_file", ""))
    ).resolve()
    projection_payload_path = (
        manifest_path.parent
        / str(manifest.get("projection_contract_payload", ""))
    ).resolve()
    if not projection_manifest_path.is_file() or not projection_payload_path.is_file():
        raise ValueError("checkpoint projection contract产物缺失")
    if (
        file_sha256(projection_manifest_path)
        != manifest.get("projection_contract_file_sha256")
        or file_sha256(projection_payload_path)
        != manifest.get("projection_contract_payload_sha256")
    ):
        raise ValueError("checkpoint projection contract SHA256绑定失效")
    projection_contract = load_json(projection_manifest_path)
    if (
        projection_contract.get("schema")
        != "local5_checkpoint_projection_contract_v2"
        or projection_contract.get("status") != "THETA_FOLDED_WEIGHT_CONTRACT"
        or projection_contract.get("topology_contract") != TOPOLOGY_CONTRACT
        or projection_contract.get("checkpoint") != manifest.get("checkpoint")
        or projection_contract.get("checkpoint_sha256")
        != manifest.get("checkpoint_sha256")
        or projection_contract.get("payload_file")
        != projection_payload_path.name
        or projection_contract.get("payload_sha256")
        != file_sha256(projection_payload_path)
        or len(projection_contract.get("blocks", [])) != 12
        or any(
            not np.isfinite(float(row.get("theta", float("nan"))))
            or float(row.get("theta", 0.0)) <= 0.0
            for row in projection_contract.get("blocks", [])
        )
        or not str(projection_contract.get("value_contract", "")).startswith(
            "V=K_binary_event*theta_K(block)"
        )
        or projection_contract.get("quantization_order")
        != "W_eff=theta_K*W_float; quantize_dyadic_int8(W_eff)"
        or projection_contract.get("runtime_datapath")
        != "K remains a 1-bit event; no runtime theta multiplier or event-width increase"
    ):
        raise ValueError("checkpoint projection contract内容或覆盖不完整")
    projection_numeric_recompute = verify_contract(
        projection_manifest_path,
        projection_payload_path,
    )
    if (
        projection_numeric_recompute.get("status") != "PASS"
        or projection_numeric_recompute.get("checkpoint_sha256")
        != manifest.get("checkpoint_sha256")
        or int(projection_numeric_recompute.get("blocks", 0)) != 12
    ):
        raise ValueError("checkpoint projection payload独立重算未通过")

    if manifest.get("qualification", {}).get("qualified") is not True:
        raise ValueError("manifest qualification未通过")
    if identity.get("schema") != "local5_post_g0_run_identity_v3":
        raise ValueError("run identity schema错误")
    if manifest.get("run_identity_file_sha256") != identity_hash:
        raise ValueError("manifest未绑定当前run identity")
    if analysis.get("manifest_sha256") != manifest_hash:
        raise ValueError("重算descriptor分析未绑定当前manifest")
    if descriptor != analysis:
        raise ValueError("落盘descriptor报告与当前重算结果逐字段不一致")
    if replay_report != replay_recomputed:
        raise ValueError("落盘replay报告与当前重算结果逐字段不一致")
    if int(replay_report.get("groups", -1)) != len(manifest["groups"]):
        raise ValueError("replay group数量与manifest不一致")
    return {
        "schema": "local5_post_g0_acceptance_v1",
        "accepted": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "run_identity": str(run_identity_path.resolve()),
        "run_identity_sha256": identity_hash,
        "bound_artifacts": {
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_hash,
            },
            "payload": {
                "path": str(
                    (
                        manifest_path.parent
                        / str(manifest["payload_file"])
                    ).resolve()
                ),
                "sha256": str(manifest["payload_sha256"]),
            },
            "cohort": {
                "path": str(
                    (
                        manifest_path.parent
                        / str(manifest["cohort_file"])
                    ).resolve()
                ),
                "sha256": str(manifest["cohort_file_sha256"]),
            },
            "run_identity": {
                "path": str(run_identity_path.resolve()),
                "sha256": identity_hash,
            },
            "release_receipt": {
                "path": str(identity["release_receipt"]),
                "sha256": str(identity["release_receipt_sha256"]),
            },
            "replay_report": {
                "path": str(replay_report_path.resolve()),
                "sha256": file_sha256(replay_report_path),
            },
            "descriptor_report": {
                "path": str(descriptor_report_path.resolve()),
                "sha256": file_sha256(descriptor_report_path),
            },
            "projection_contract_manifest": {
                "path": str(projection_manifest_path),
                "sha256": file_sha256(projection_manifest_path),
            },
            "projection_contract_payload": {
                "path": str(projection_payload_path),
                "sha256": file_sha256(projection_payload_path),
            },
        },
        "samples": analysis["formal_coverage"]["samples"],
        "blocks": analysis["formal_coverage"]["blocks"],
        "groups": analysis["groups"],
        "descriptors": analysis["descriptors"],
        "threshold_training_semantics": threshold_semantics,
        "checkpoint_projection_numeric_recompute": projection_numeric_recompute,
        "checks": {
            "loader_provenance": True,
            "formal_qualification": True,
            "relation_rtl_binding": True,
            "descriptor_geometry": True,
            "replay_binding": True,
            "descriptor_report_binding": True,
            "reports_recomputed_equal": True,
            "source_software_binding": True,
            "release_receipt_binding": True,
            "checkpoint_projection_weight_binding": True,
            "checkpoint_projection_payload_recomputed": True,
            "checkpoint_projection_topology_abi": True,
            "threshold_training_deployment_semantics": True,
        },
    }


def write_report(value: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Local5 Fullres Post-G0 Fail-Closed验收",
        "",
        f"- accepted：`{value['accepted']}`；",
        f"- samples：{value['samples']}；",
        f"- blocks：{value['blocks']}；",
        f"- sampled groups：{value['groups']}；",
        f"- descriptors：{value['descriptors']}；",
        f"- manifest SHA256：`{value['manifest_sha256']}`；",
        f"- run identity SHA256：`{value['run_identity_sha256']}`；",
        "",
        "只有本文件为 `accepted=true` 时，profile/replay/descriptor结果才可"
        "晋级为正式 post-G0 证据。",
        "",
    ]
    (output_dir / "acceptance.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--descriptor-report", type=Path, required=True)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = validate(
        manifest_path=args.manifest,
        replay_report_path=args.replay_report,
        descriptor_report_path=args.descriptor_report,
        run_identity_path=args.run_identity,
    )
    write_report(value, args.output_dir)
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

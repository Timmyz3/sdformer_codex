#!/usr/bin/env python3
"""Fail-closed final audit for the ranked Local5 ep44 hardware rebind."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from local5_release_receipt import file_sha256, validate_release_receipt


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RUN = (
    REPO
    / "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812"
)
PROFILE = ROOT / "results/local5_ep44_hardware_rebind_20260815_profile100"
REPLAY = ROOT / "results/local5_ep44_hardware_rebind_20260815_replay/report.json"
DESCRIPTOR = (
    ROOT / "results/local5_ep44_hardware_rebind_20260815_descriptor_analysis/report.json"
)
ACCEPTANCE = (
    ROOT / "results/local5_ep44_hardware_rebind_20260815_acceptance/acceptance.json"
)
POSTSCORE = (
    ROOT / "results/local5_ep44_hardware_rebind_20260815_postscore_rtl/report.json"
)
INTEGRATED_DIR = (
    ROOT / "results/local5_ep44_hardware_rebind_20260815_score_projection_rtl"
)
INTEGRATED = INTEGRATED_DIR / "report_ranked.json"
COMPLETE = INTEGRATED_DIR / "complete_ranked.json"
VECTOR_MANIFEST = (
    ROOT
    / "tb_qfit/vectors/"
    "local5_ep44_hardware_rebind_20260815_score_projection100/manifest.json"
)
OUTPUT = ROOT / "results/local5_ep44_hardware_rebind_20260815_final_audit.json"
CHECKPOINT_SHA = "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57"
FROZEN_359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def metric(path: Path, key: str) -> float:
    return float((load(path).get("metrics") or {})[key])


def sealed_files_match(complete: dict) -> bool:
    files = complete.get("files") or {}
    return bool(files) and all(
        (INTEGRATED_DIR / name).is_file()
        and file_sha256(INTEGRATED_DIR / name) == expected_sha
        for name, expected_sha in files.items()
    )


def vector_artifacts_match(manifest: dict) -> bool:
    artifacts = manifest.get("artifacts") or {}
    return bool(artifacts) and all(
        (VECTOR_MANIFEST.parent / row["file"]).is_file()
        and file_sha256(VECTOR_MANIFEST.parent / row["file"]) == row["sha256"]
        for row in artifacts.values()
    )


def main() -> int:
    receipt_path = PROFILE / "ranked_checkpoint_release_receipt.json"
    receipt = validate_release_receipt(receipt_path, file_sha256(receipt_path))
    manifest_path = PROFILE / "ordered_term_manifest.json"
    manifest = load(manifest_path)
    replay = load(REPLAY)
    descriptor = load(DESCRIPTOR)
    acceptance = load(ACCEPTANCE)
    postscore = load(POSTSCORE)
    integrated = load(INTEGRATED)
    complete = load(COMPLETE)
    vector_manifest = load(VECTOR_MANIFEST)

    checks = {
        "receipt_pass_ep44": (
            receipt.get("status") == "PASS" and receipt.get("best_epoch") == 44
        ),
        "manifest_qualified_100x12": (
            manifest.get("evidence_level") == "post_g0"
            and manifest.get("checkpoint_sha256") == CHECKPOINT_SHA
            and manifest.get("qualification", {}).get("qualified") is True
            and manifest.get("qualification", {}).get("processed_samples") == 100
            and manifest.get("qualification", {}).get("attached_blocks") == 12
            and len(manifest.get("groups", [])) == 4800
        ),
        "replay_bound_4800": (
            replay.get("groups") == 4800
            and replay.get("manifest_sha256") == file_sha256(manifest_path)
        ),
        "descriptor_equivalent_4800": (
            descriptor.get("formal_qualification") is True
            and descriptor.get("groups") == 4800
            and descriptor.get("source_destination_equivalence", {}).get("passed")
            is True
        ),
        "acceptance_pass": (
            acceptance.get("accepted") is True
            and acceptance.get("samples") == 100
            and acceptance.get("blocks") == 12
            and acceptance.get("groups") == 4800
            and all((acceptance.get("checks") or {}).values())
        ),
        "postscore_real_weight_acc32": (
            postscore.get("schema") == "local5_gasr2c_fivebank_rtl_summary_v1"
            and postscore.get("weight_mode")
            == "checkpoint_theta_folded_dyadic_int8_head_slice"
            and postscore.get("correctness", {}).get("acc32")
            == "100/100组PASS，逐元素零失配"
            and postscore.get("verification", {}).get("random_sva") == "PASS"
        ),
        "integrated_ep44_pass": (
            integrated.get("status") == "PASS"
            and integrated.get("checkpoint_sha256") == CHECKPOINT_SHA
            and integrated.get("groups") == 100
            and integrated.get("acc32_checks") == 360000
            and all(
                row.get("zero_mismatch") is True
                for row in (integrated.get("actual_acc32") or {}).values()
            )
            and all(
                row.get("groups") == 8
                for row in (integrated.get("random_stress") or {}).values()
            )
        ),
        "integrated_sealed": (
            complete.get("status") == "SEALED"
            and complete.get("report_sha256") == file_sha256(INTEGRATED)
        ),
        "integrated_sealed_files_live": sealed_files_match(complete),
        "integrated_vector_manifest_live": (
            complete.get("vector_manifest_sha256") == file_sha256(VECTOR_MANIFEST)
            and vector_artifacts_match(vector_manifest)
        ),
        "integrated_source_bindings_live": all(
            Path(row["path"]).is_file()
            and file_sha256(Path(row["path"])) == row["sha256"]
            for row in integrated.get("source_bindings", [])
        ),
        "frozen_359_unchanged": (
            file_sha256(ROOT / "docs/359_DATE终局冻结_20260813.md")
            == FROZEN_359_SHA
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Local5 ep44 final audit failed: " + ",".join(failed))

    float_profile = RUN / "standard_valid825/epoch44/spike_profile.json"
    dyadic_profile = RUN / "deploy_valid825/dyadic_q7q17/epoch44/spike_profile.json"
    hardware_profile = (
        RUN / "deploy_valid825/hardware_order_q7q17/epoch44/spike_profile.json"
    )
    result = {
        "schema": "local5_ep44_hardware_rebind_final_audit_v1",
        "status": "PASS",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checks": checks,
        "valid825_attention_core_numeric": {
            "float_aee": metric(float_profile, "AEE"),
            "dyadic_q7q17_aee": metric(dyadic_profile, "AEE"),
            "hardware_order_q7q17_aee": metric(hardware_profile, "AEE"),
            "full_network_fixed_point": False,
        },
        "profile": {
            "samples": 100,
            "blocks": 12,
            "groups": 4800,
            "descriptors": acceptance["descriptors"],
            "expanded_updates": descriptor["source_destination_equivalence"][
                "expanded_updates"
            ],
        },
        "rtl": {
            "scope": integrated["scope"],
            "groups": integrated["groups"],
            "out_dim": 2,
            "acc32_checks": integrated["acc32_checks"],
            "l1_speedup": integrated["speedups"]["l1"]["ratio_of_totals"],
            "l2_speedup": integrated["speedups"]["l2"]["ratio_of_totals"],
        },
        "claim_boundary": [
            "component-level RTL, not full encoder",
            "pre-bias/pre-BN/pre-requant/pre-residual and not cross-head",
            "OUT_DIM=2 real checkpoint projection slice",
            "no foundry DC/STA/SAIF/PTPX result",
            "no innovation-score increase and no frozen-table replacement",
        ],
        "artifacts": {
            str(path.relative_to(ROOT)): {
                "sha256": file_sha256(path),
            }
            for path in (
                receipt_path,
                manifest_path,
                REPLAY,
                DESCRIPTOR,
                ACCEPTANCE,
                POSTSCORE,
                INTEGRATED,
                COMPLETE,
            )
        },
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the frozen H67 DATE mainline after all queued controls finish."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
MVSEC = RESULTS / "mvsec_cicc_nb0_h67_local5_audit_20260812.json"
NOMOTION = REPO / "neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.json"
LOCAL50 = RESULTS / (
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/"
    "convergence_summary.json"
)
DSEC_CLOSURE = REPO / "neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json"
H67_HW = REPO / "hw_autoresearch_nts07/results/h67_postconvergence_rank1_hardware_evidence_20260805.json"
LOCAL_HW = REPO / (
    "hw_autoresearch_nts07/results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/"
    "checkpoint_bound_scope.json"
)
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
OUTPUT = REPO / "neuron_autoresearch/DATE_FINAL_MAINLINE_DECISION_20260812.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
REQUIRED = (MVSEC, NOMOTION, LOCAL50, DSEC_CLOSURE, H67_HW, LOCAL_HW)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def source_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def pct(reference: float, candidate: float) -> float:
    return 100.0 * (candidate - reference) / reference


def run() -> dict[str, object]:
    mvsec = load(MVSEC)
    nomotion = load(NOMOTION)
    local50 = load(LOCAL50)
    closure = load(DSEC_CLOSURE)
    h67_hw = load(H67_HW)
    local_hw = load(LOCAL_HW)
    statuses = {
        "MVSEC": mvsec.get("status") == "PASS",
        "no_motion": nomotion.get("status") == "PASS_PROTOCOL_AND_IDENTITY",
        "Local5_50": local50.get("status") == "PASS",
        "DSEC_closure": closure.get("status") == "PASS",
        "H67_hardware": h67_hw.get("status") == "PASS",
        "Local5_hardware": local_hw.get("status") == "PASS",
    }
    if not all(statuses.values()):
        raise RuntimeError(f"upstream evidence not qualified: {statuses}")

    dsec = closure["algorithm_targets"]
    h67_dsec = dsec["candidates"]["H67"]
    nb0_dsec = dsec["baseline"]
    local_points = local50.get("points") or []
    if {int(point["checkpoint_label"]) for point in local_points} != {39, 44, 49}:
        raise RuntimeError("Local5 40->50 point set mismatch")
    local_rank1 = min(local_points, key=lambda point: float(point["AEE"]))
    if int(local50["rank1_checkpoint_label"]) != int(local_rank1["checkpoint_label"]):
        raise RuntimeError("Local5 40->50 rank1 drift")
    h67_aee = float(h67_dsec["AEE"])
    h67_spikes = float(h67_dsec["total_spikes_g"])
    local_aee = float(local_rank1["AEE"])
    local_spikes = float(local_rank1["total_spikes_g"])
    nb0_aee = float(nb0_dsec["AEE"])
    nb0_spikes = float(nb0_dsec["total_spikes_g"])

    mvsec_h67 = mvsec["routes"]["h67"]["full_sequence"]
    mvsec_local = mvsec["routes"]["local5"]["full_sequence"]
    mvsec_nb0 = mvsec["routes"]["nb0"]["full_sequence"]
    gates = {
        "H67_DSEC_AEE_within_NB0_plus5": h67_aee <= nb0_aee * 1.05,
        "H67_DSEC_spikes_reduce20": h67_spikes <= nb0_spikes * 0.80,
        "Local5_DSEC_AEE_within_NB0_plus5": local_aee <= nb0_aee * 1.05,
        "Local5_DSEC_spikes_reduce20": local_spikes <= nb0_spikes * 0.80,
        "H67_MVSEC_qualifies": bool(mvsec["DATE_gates"]["h67"]["qualifies"]),
        "Local5_MVSEC_qualifies": bool(mvsec["DATE_gates"]["local5"]["qualifies"]),
        "Motion_control_protocol_pass": nomotion.get("status") == "PASS_PROTOCOL_AND_IDENTITY",
        "H67_same_checkpoint_component_RTL": int(h67_hw.get("rank1_epoch", -1)) == 35,
        "Local5_algorithm_rank1_has_same_checkpoint_RTL": (
            int(local_rank1["checkpoint_label"])
            == int(local_hw.get("checkpoint_identity", {}).get("best_epoch", -1))
        ),
    }
    if not all(
        gates[key]
        for key in (
            "H67_DSEC_AEE_within_NB0_plus5",
            "H67_DSEC_spikes_reduce20",
            "H67_MVSEC_qualifies",
            "Motion_control_protocol_pass",
            "H67_same_checkpoint_component_RTL",
        )
    ):
        raise RuntimeError(f"H67 final mandatory gates failed: {gates}")

    local_accuracy_gain_pct = -pct(h67_aee, local_aee)
    h67_spike_gain_pct = -pct(local_spikes, h67_spikes)
    h67_mvsec_gain_pct = -pct(
        float(mvsec_local["mean_aee"]), float(mvsec_h67["mean_aee"])
    )
    motion_aee_gain_pct = -pct(
        float(nomotion["h81_no_motion"]["AEE"]),
        float(nomotion["h67_motion"]["AEE"]),
    )

    # H67 is frozen as the DATE paper identity. Local5 remains an independently
    # measured accuracy/topology extension even if a later checkpoint has lower
    # DSEC AEE; this audit must not silently reopen a mixed or alternate mainline.
    selected = "H67_Motion_TTX"
    decision = "H67_MAINLINE_FROZEN_LOCAL5_EXTENSION_REPORTED"

    result = {
        "schema": "date_final_mainline_decision_v1",
        "status": "PASS_EVIDENCE_AUDIT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "selected_mainline": selected,
        "policy": {
            "algorithm_targets": "AEE <= NB0+5% and spikes <= NB0-20% on DSEC",
            "paper_identity": (
                "H67 Motion-TTX ep35 is frozen after passing DSEC accuracy/spike, "
                "MVSEC generalization, motion-control, and same-checkpoint evidence gates"
            ),
            "Local5_scope": "accuracy/topology extension; cannot replace or mix into the DATE mainline",
            "hardware_completeness_not_used_as_requested": True,
        },
        "upstream_status_checks": statuses,
        "gates": gates,
        "DSEC": {
            "NB0": {"AEE": nb0_aee, "total_spikes_g": nb0_spikes},
            "H67": {"AEE": h67_aee, "total_spikes_g": h67_spikes},
            "Local5_rank1": local_rank1,
            "Local5_AEE_gain_pct_vs_H67": local_accuracy_gain_pct,
            "H67_spikes_gain_pct_vs_Local5": h67_spike_gain_pct,
        },
        "MVSEC_full_sequence": {
            "NB0_AEE": float(mvsec_nb0["mean_aee"]),
            "H67_AEE": float(mvsec_h67["mean_aee"]),
            "Local5_AEE": float(mvsec_local["mean_aee"]),
            "H67_AEE_gain_pct_vs_Local5": h67_mvsec_gain_pct,
            "algorithm_only_selected": mvsec["algorithm_only_selection"]["selected"],
        },
        "motion_control": {
            "H67_AEE": float(nomotion["h67_motion"]["AEE"]),
            "H81_AEE": float(nomotion["h81_no_motion"]["AEE"]),
            "H67_AEE_gain_pct_vs_H81": motion_aee_gain_pct,
            "control_scope": nomotion["fairness_scope"],
            "H81_convergence": nomotion["h81_convergence"],
        },
        "hardware": {
            "H67": {
                "checkpoint_epoch": 35,
                "same_checkpoint_component_RTL": gates["H67_same_checkpoint_component_RTL"],
                "claim_boundary": "checkpoint_bound_component_RTL_exact_not_full_network",
            },
            "Local5": {
                "algorithm_rank1_epoch": int(local_rank1["checkpoint_label"]),
                "hardware_anchor_epoch": int(local_hw["checkpoint_identity"]["best_epoch"]),
                "same_checkpoint_component_RTL": gates["Local5_algorithm_rank1_has_same_checkpoint_RTL"],
                "claim_boundary": "ep29_component_RTL_only_unless_rank1_is_rebound",
            },
        },
        "inputs": {source_name(path): binding(path) for path in REQUIRED},
    }
    return result


def write_docs(result: dict[str, object]) -> None:
    dsec = result["DSEC"]
    mvsec = result["MVSEC_full_sequence"]
    motion = result["motion_control"]
    lines = [
        "# DATE final mainline decision",
        "",
        f"Status: `{result['status']}`; decision: `{result['decision']}`; selected: `{result['selected_mainline']}`.",
        "",
        "| evidence | H67 | Local5/control | comparison |",
        "|---|---:|---:|---:|",
        f"| DSEC AEE | {dsec['H67']['AEE']:.6f} | {dsec['Local5_rank1']['AEE']:.6f} | Local5 gain {dsec['Local5_AEE_gain_pct_vs_H67']:.3f}% |",
        f"| DSEC spikes(G) | {dsec['H67']['total_spikes_g']:.4f} | {dsec['Local5_rank1']['total_spikes_g']:.4f} | H67 gain {dsec['H67_spikes_gain_pct_vs_Local5']:.3f}% |",
        f"| MVSEC full AEE | {mvsec['H67_AEE']:.6f} | {mvsec['Local5_AEE']:.6f} | H67 gain {mvsec['H67_AEE_gain_pct_vs_Local5']:.3f}% |",
        f"| Motion control AEE | {motion['H67_AEE']:.6f} | H81 {motion['H81_AEE']:.6f} | H67 gain {motion['H67_AEE_gain_pct_vs_H81']:.3f}% |",
        "",
        "Hardware comparison uses innovation/co-design and same-checkpoint provenance; engineering completeness is deliberately not used as requested.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = "DATE_FINAL_MAINLINE_DECISION_20260812"
    for path in (REDESIGN,):
        text = path.read_text(encoding="utf-8")
        if marker in text:
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n"
                f"<!-- {marker} -->\n\n"
                "### DATE 最终跨证据主线裁决\n\n"
                f"- 状态=`{result['status']}`，决策=`{result['decision']}`，"
                f"主线=`{result['selected_mainline']}`。\n"
                f"- DSEC: H67 AEE/spikes=`{dsec['H67']['AEE']:.6f}/{dsec['H67']['total_spikes_g']:.4f}G`，"
                f"Local5 rank-1 ep{dsec['Local5_rank1']['checkpoint_label']}="
                f"`{dsec['Local5_rank1']['AEE']:.6f}/{dsec['Local5_rank1']['total_spikes_g']:.4f}G`。\n"
                f"- MVSEC full AEE: H67=`{mvsec['H67_AEE']:.6f}`，Local5=`{mvsec['Local5_AEE']:.6f}`；"
                f"Motion control H67/H81=`{motion['H67_AEE']:.6f}/{motion['H81_AEE']:.6f}`。\n"
                "- 硬件证据仅只读消费；该算法审计不修改硬件代码或硬件文档。\n"
                f"- 机器审计：`{OUTPUT.relative_to(REPO)}`。\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    while not all(path.is_file() for path in REQUIRED):
        missing = [str(path) for path in REQUIRED if not path.is_file()]
        if not args.wait:
            raise FileNotFoundError(missing)
        print(f"WAIT final DATE evidence: {missing}", flush=True)
        time.sleep(args.poll_seconds)
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_docs(result)
    print(f"PASS final DATE mainline audit: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

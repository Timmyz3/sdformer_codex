#!/usr/bin/env python3
"""Bind the completed H81 MVSEC rescue result to the hardware G1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_sequences(
    h81_rows: list[dict[str, object]],
    nb0_rows: list[dict[str, object]],
) -> dict[str, object]:
    h81 = {str(row["sequence"]): row for row in h81_rows}
    nb0 = {str(row["sequence"]): row for row in nb0_rows}
    if set(h81) != set(nb0) or len(h81) != 4:
        raise ValueError("H81 and NB0 do not contain the same four MVSEC sequences")
    per_sequence = []
    failing = []
    for sequence in ("outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"):
        h81_row = h81[sequence]
        nb0_row = nb0[sequence]
        if int(h81_row["samples"]) != int(nb0_row["samples"]):
            raise ValueError(f"sample-count mismatch for {sequence}")
        h81_aee = float(h81_row["AEE"])
        nb0_aee = float(nb0_row["AEE"])
        passed = h81_aee < nb0_aee
        if not passed:
            failing.append(sequence)
        per_sequence.append(
            {
                "sequence": sequence,
                "samples": int(h81_row["samples"]),
                "h81_AEE": h81_aee,
                "nb0_AEE": nb0_aee,
                "delta_AEE": h81_aee - nb0_aee,
                "better_than_NB0": passed,
            }
        )
    return {
        "per_sequence": per_sequence,
        "failing_sequences": failing,
        "all_sequence_better_than_NB0": not failing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h81-summary",
        type=Path,
        default=EXP
        / "results/mvsec_cicc_h81_nomotion_w8_seed0_full_20260816/mvsec_summary.json",
    )
    parser.add_argument(
        "--baseline-audit",
        type=Path,
        default=EXP / "results/mvsec_cicc_nb0_h67_local5_audit_20260812.json",
    )
    parser.add_argument(
        "--g0-report",
        type=Path,
        default=HW_ROOT / "results/h81_rqtb_g0_20260816/g0_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HW_ROOT / "results/h81_rqtb_g0_20260816/h81_mvsec_gate_receipt.json",
    )
    args = parser.parse_args()

    h81 = json.loads(args.h81_summary.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    g0 = json.loads(args.g0_report.read_text(encoding="utf-8"))
    if h81.get("protocol") != "full_sequence" or h81.get("skipped"):
        raise ValueError("H81 MVSEC summary is not a complete full-sequence result")

    comparison = compare_sequences(
        h81["sequences"],
        baseline["routes"]["nb0"]["full_sequence"]["per_sequence"],
    )
    g0_checkpoint = str(g0["checkpoint_sha256"])
    mvsec_checkpoint = str(h81["checkpoint_sha256"])
    if g0_checkpoint == mvsec_checkpoint:
        raise ValueError("expected MVSEC rescue and DSEC G0 to have distinct identities")

    status = (
        "PASS_H81_MVSEC_ALL_SEQUENCE_GATE"
        if comparison["all_sequence_better_than_NB0"]
        else "FAIL_H81_MVSEC_ALL_SEQUENCE_GATE"
    )
    receipt = {
        "schema": "h81_mvsec_hardware_gate_receipt_v1",
        "status": status,
        "evidence": "[model] same-protocol MVSEC full-sequence rescue evaluation",
        "claim_boundary": (
            "The MVSEC checkpoint is NB0-initialized CICC day2 and differs from "
            "the DSEC H81 G0 checkpoint. This gate cannot transfer DSEC workload, "
            "accuracy, RTL, cycle, activity, or PPA evidence."
        ),
        "h81_summary": str(args.h81_summary.resolve()),
        "h81_summary_sha256": sha256(args.h81_summary),
        "baseline_audit": str(args.baseline_audit.resolve()),
        "baseline_audit_sha256": sha256(args.baseline_audit),
        "mvsec_checkpoint_sha256": mvsec_checkpoint,
        "mvsec_config_sha256": h81["source_config_sha256"],
        "dsec_g0_checkpoint_sha256": g0_checkpoint,
        "identity_same_as_dsec_g0": False,
        "mean_AEE": float(h81["mean_aee"]),
        **comparison,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    g0["h81_mvsec_gate"] = receipt
    g0["blocking_gates"] = [
        item for item in g0["blocking_gates"] if item != "H81 MVSEC is missing"
    ]
    if status.startswith("FAIL_"):
        failure = ", ".join(comparison["failing_sequences"])
        gate_text = f"H81 MVSEC all-sequence gate failed: {failure}"
        if gate_text not in g0["blocking_gates"]:
            g0["blocking_gates"].append(gate_text)
        g0["status"] = "G0_PASS_G1_BLOCKED_BY_SELECTOR_AND_MVSEC_FAIL"
    args.g0_report.write_text(json.dumps(g0, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

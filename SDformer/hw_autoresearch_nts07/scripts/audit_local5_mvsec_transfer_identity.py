#!/usr/bin/env python3
"""Fail-closed identity audit for the Local5 DSEC-to-MVSEC rescue result."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


LOCAL5_SUMMARY = (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "mvsec_cicc_local5_dsec_ep44_ft15_full_20260816/mvsec_summary.json"
)
H81_SUMMARY = (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "mvsec_cicc_h81_nomotion_w8_seed0_full_20260816/mvsec_summary.json"
)
BASELINE_AUDIT = (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "mvsec_cicc_nb0_h67_local5_audit_20260812.json"
)
TRAIN_LOG = (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "mvsec_cicc_local5_dsec_ep44_ft15_20260816/train.log"
)
HARDWARE_MANIFEST = (
    "hw_autoresearch_nts07/results/"
    "local5_ep44_hardware_rebind_20260815_profile100/ordered_term_manifest.json"
)
RESCUE_RECEIPT = "neuron_autoresearch/MVSEC_H81_LOCAL5_RESCUE_20260816.json"

EXPECTED_SAMPLES = {
    "outdoor_day1": 2755,
    "indoor_flying1": 1883,
    "indoor_flying2": 1885,
    "indoor_flying3": 1885,
}
EXPECTED_LOCAL5_FT_CHECKPOINT = (
    "fe774db3463eb3b107737171df66885556f48d018b41eaf2e058e8c9087f496e"
)
EXPECTED_HARDWARE_CHECKPOINT = (
    "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57"
)
EXPECTED_INPUT_SHA256 = {
    "local5_full_summary": "10e39a6d21f04988e059e9fdcc6ceb5c90fe2a80512d204e56798c4d11160544",
    "h81_full_summary": "45bd4971ad6edf51977391b92332d4dda074ccc69d8019f7f43ac8b6204fa5a3",
    "baseline_audit": "d420ce0293c43a9353496abd856402154562638f218cb06988c7bcd76158590a",
    "local5_train_log": "267f133cdf25e8b20a63e229eeb98940d215d7100551ae355de02f9c1848d52c",
    "local5_hardware_manifest": "fdde0939b9f7c0a09a48986511b3f80d329b4674ac4a901b4dfef818a209690d",
    "rescue_receipt": "6ae9ab0c4f364ac7ee5c082fe888b98074a84b943fa67fac05a2904c6eb8a5c7",
}
DROP_RE = re.compile(r"dropped (\d+) shape-mismatched keys: (\[[^\n]+\])")
MISSING_RE = re.compile(r"missing keys sample: (\[[^\n]+\])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sequence_map(summary: dict, field: str = "sequences") -> dict[str, dict]:
    rows = summary.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"missing sequence list: {field}")
    mapped = {row.get("sequence"): row for row in rows}
    if set(mapped) != set(EXPECTED_SAMPLES) or len(mapped) != len(rows):
        raise ValueError(f"unexpected or duplicate sequences: {sorted(mapped)}")
    for name, expected in EXPECTED_SAMPLES.items():
        if mapped[name].get("samples") != expected:
            raise ValueError(
                f"sample count mismatch for {name}: "
                f"{mapped[name].get('samples')} != {expected}"
            )
    return mapped


def parse_reinitialized_pe_keys(text: str) -> list[str]:
    match = DROP_RE.search(text)
    if not match:
        raise ValueError("missing shape-mismatched key audit")
    declared = int(match.group(1))
    printed_sample = ast.literal_eval(match.group(2))
    missing_match = MISSING_RE.search(text)
    keys = ast.literal_eval(missing_match.group(1)) if missing_match else printed_sample
    if declared != 12 or len(keys) != declared or len(set(keys)) != declared:
        raise ValueError(
            f"unexpected dropped-key count: declared={declared} actual={len(keys)}"
        )
    if not set(printed_sample).issubset(keys):
        raise ValueError("dropped-key sample disagrees with complete missing-key list")
    if not all(key.endswith("attn.positional_encoding") for key in keys):
        raise ValueError("shape-mismatched keys are not exclusively positional encodings")
    if "checkpoint_overlay_keys=210, missing=12, unexpected=0" not in text:
        raise ValueError("missing checkpoint overlay load audit")
    return keys


def compare_lower_is_better(
    candidate: dict[str, dict], references: dict[str, dict]
) -> dict[str, dict]:
    result = {}
    for reference_name, reference_rows in references.items():
        deltas = {
            sequence: candidate[sequence]["AEE"] - reference_rows[sequence]["AEE"]
            for sequence in EXPECTED_SAMPLES
        }
        result[reference_name] = {
            "all_four_lower": all(delta < 0 for delta in deltas.values()),
            "aee_delta_candidate_minus_reference": deltas,
        }
    return result


def render_markdown(receipt: dict) -> str:
    result = receipt["algorithm_result"]
    lines = [
        "# Local5 MVSEC transfer identity receipt",
        "",
        f"Status: `{receipt['status']}`",
        "",
        "## Result",
        "",
        "| Sequence | Local5-FT | NB0 | H67 | old Local5 | H81 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sequence in EXPECTED_SAMPLES:
        row = result["per_sequence"][sequence]
        lines.append(
            f"| {sequence} | {row['local5_ft']:.6f} | {row['nb0']:.6f} | "
            f"{row['h67']:.6f} | {row['old_local5']:.6f} | {row['h81']:.6f} |"
        )
    means = result["macro_mean_aee"]
    lines.extend(
        [
            f"| macro mean | {means['local5_ft']:.6f} | {means['nb0']:.6f} | "
            f"{means['h67']:.6f} | {means['old_local5']:.6f} | {means['h81']:.6f} |",
            "",
            "The Local5 transfer run is lower in AEE on all four full sequences "
            "than each named reference under the recorded protocol.",
            "",
            "## Identity boundary",
            "",
            f"- DSEC ep44 hardware checkpoint: `{receipt['identity']['hardware_checkpoint_sha256']}`",
            f"- MVSEC transfer checkpoint: `{receipt['identity']['mvsec_transfer_checkpoint_sha256']}`",
            f"- Reinitialized positional encodings: `{receipt['identity']['reinitialized_positional_encoding_count']}`",
            "- The checkpoints are not identical. The MVSEC run used a DSEC overlay and "
            "reinitialized 12 shape-mismatched positional-encoding tensors.",
            "- This result is an algorithm rescue table, not an ep44 hardware rebind, "
            "not RTL/PPA evidence, and not a new DATE architecture contribution.",
            "",
            "## Hardware consequence",
            "",
            "Current Local5 RTL and its frozen cycle/PPA contracts do not inherit these "
            "MVSEC numbers. If this transfer identity is selected for a paper model, it "
            "requires a new hardware-order profile, trace export, score-to-Acc32 replay, "
            "and activity/PPA identity chain while reusing the existing Local5 DUT.",
            "",
            "H81 remains a separate failed all-sequence gate and no H81 RTL is admitted "
            "by this receipt.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    paths = {
        "local5_full_summary": root / LOCAL5_SUMMARY,
        "h81_full_summary": root / H81_SUMMARY,
        "baseline_audit": root / BASELINE_AUDIT,
        "local5_train_log": root / TRAIN_LOG,
        "local5_hardware_manifest": root / HARDWARE_MANIFEST,
        "rescue_receipt": root / RESCUE_RECEIPT,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        actual_sha = sha256(path)
        if actual_sha != EXPECTED_INPUT_SHA256[name]:
            raise ValueError(
                f"input identity changed for {name}: "
                f"{actual_sha} != {EXPECTED_INPUT_SHA256[name]}"
            )
    local5 = load_json(paths["local5_full_summary"])
    h81 = load_json(paths["h81_full_summary"])
    baseline = load_json(paths["baseline_audit"])
    hardware = load_json(paths["local5_hardware_manifest"])
    rescue = load_json(paths["rescue_receipt"])
    pe_keys = parse_reinitialized_pe_keys(
        paths["local5_train_log"].read_text(encoding="utf-8", errors="replace")
    )

    if local5.get("protocol") != "full_sequence" or local5.get("skipped") != []:
        raise ValueError("Local5 summary is not a complete full-sequence evaluation")
    if h81.get("protocol") != "full_sequence" or h81.get("skipped") != []:
        raise ValueError("H81 summary is not a complete full-sequence evaluation")
    if local5.get("checkpoint_sha256") != EXPECTED_LOCAL5_FT_CHECKPOINT:
        raise ValueError("Local5 transfer checkpoint identity changed")
    if hardware.get("checkpoint_sha256") != EXPECTED_HARDWARE_CHECKPOINT:
        raise ValueError("Local5 hardware checkpoint identity changed")
    if local5["checkpoint_sha256"] == hardware["checkpoint_sha256"]:
        raise ValueError("transfer and hardware identities unexpectedly match")
    if rescue.get("h81", {}).get("all_sequence_better_than_NB0") is not False:
        raise ValueError("H81 all-sequence gate is not failed")
    if rescue.get("local5_dsec_ft", {}).get("all_sequence_better_than_NB0") is not True:
        raise ValueError("Local5 transfer all-sequence gate is not passed")
    if rescue.get("local5_dsec_ft", {}).get("protocol_family") != "dsec_pretrain_day2_ft":
        raise ValueError("Local5 transfer protocol family changed")

    local5_rows = sequence_map(local5)
    h81_rows = sequence_map(h81)
    reference_rows = {}
    for route in ("nb0", "h67", "local5"):
        reference_rows[route] = sequence_map(
            baseline["routes"][route]["full_sequence"], field="per_sequence"
        )
    comparisons = compare_lower_is_better(
        local5_rows,
        {**reference_rows, "h81": h81_rows},
    )
    if not all(item["all_four_lower"] for item in comparisons.values()):
        raise ValueError("Local5 transfer does not beat every named reference on all sequences")

    def mean(rows: dict[str, dict]) -> float:
        return sum(row["AEE"] for row in rows.values()) / len(rows)

    receipt = {
        "schema": "local5_mvsec_transfer_identity_receipt_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_RESCUE_TABLE_IDENTITY_SPLIT_NOT_HARDWARE_REBIND",
        "identity": {
            "hardware_protocol_family": "DSEC_ep44_component_rebind",
            "mvsec_protocol_family": "dsec_pretrain_day2_ft",
            "hardware_checkpoint_sha256": hardware["checkpoint_sha256"],
            "mvsec_transfer_checkpoint_sha256": local5["checkpoint_sha256"],
            "mvsec_transfer_config_sha256": local5["source_config_sha256"],
            "same_checkpoint_identity": False,
            "checkpoint_overlay_keys": 210,
            "reinitialized_positional_encoding_count": len(pe_keys),
            "reinitialized_positional_encoding_keys": pe_keys,
        },
        "algorithm_result": {
            "protocol": "full_sequence",
            "sample_counts": EXPECTED_SAMPLES,
            "total_valid_pixels": local5["total_valid_pixels"],
            "valid_pixel_weighted_aee": local5["valid_pixel_weighted_aee"],
            "macro_mean_aee": {
                "local5_ft": mean(local5_rows),
                "nb0": mean(reference_rows["nb0"]),
                "h67": mean(reference_rows["h67"]),
                "old_local5": mean(reference_rows["local5"]),
                "h81": mean(h81_rows),
            },
            "per_sequence": {
                sequence: {
                    "local5_ft": local5_rows[sequence]["AEE"],
                    "nb0": reference_rows["nb0"][sequence]["AEE"],
                    "h67": reference_rows["h67"][sequence]["AEE"],
                    "old_local5": reference_rows["local5"][sequence]["AEE"],
                    "h81": h81_rows[sequence]["AEE"],
                }
                for sequence in EXPECTED_SAMPLES
            },
            "comparisons": comparisons,
        },
        "claim_boundary": {
            "algorithm_rescue_table": True,
            "hardware_identity_rebound": False,
            "rtl_cycle_or_acc32_evidence_for_mvsec_checkpoint": False,
            "ppa_or_saif_evidence_for_mvsec_checkpoint": False,
            "new_date_architecture_contribution": False,
            "selector_changed": False,
            "h81_rtl_admitted": False,
            "required_if_selected": [
                "hardware-order profile for the MVSEC transfer checkpoint",
                "trace export and score-to-Acc32 RTL replay",
                "activity and target-library PPA identity chain",
            ],
        },
        "inputs": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.markdown:
        markdown = args.markdown if args.markdown.is_absolute() else root / args.markdown
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build scoped full-frame attention and Amdahl models for Motion and Local5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STAGE_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
STAGE_WINDOWS = {0: 440, 1: 120, 2: 30, 3: 10}
MOTION_ROW_RANGES = {0: (0, 6), 1: (6, 18), 2: (18, 90), 3: (90, 138)}
FROZEN_MOTION = {
    "rows": 138,
    "fixed": 112_589,
    "rqtb": 94_891,
    "fpairs": 31_050,
    "fslots": 62_100,
    "fequal": 28_001,
    "rpairs": 31_050,
    "rslots": 34_099,
    "requal": 28_001,
}

MOTION_ROW_RE = re.compile(
    r"^FAIR_ROW row=(?P<row>\d+) active=(?P<active>\d+) skip=(?P<skip>[01]) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) "
    r"fslots=(?P<fslots>\d+) rslots=(?P<rslots>\d+) equal=(?P<equal>\d+)$"
)
MOTION_SUM_RE = re.compile(
    r"^FAIR_SUM rows=(?P<rows>\d+) skip=(?P<skip>\d+) fixed=(?P<fixed>\d+) "
    r"rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) fpairs=(?P<fpairs>\d+) "
    r"fslots=(?P<fslots>\d+) fequal=(?P<fequal>\d+) rpairs=(?P<rpairs>\d+) "
    r"rslots=(?P<rslots>\d+) requal=(?P<requal>\d+)$"
)
LOCAL_GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+).* terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
LOCAL_PASS_RE = re.compile(
    r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(?P<cycles>\d+)$"
)
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b|\bERROR:")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amdahl(component_speedup: float, fraction: float) -> float:
    if component_speedup <= 0.0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid Amdahl input")
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def parse_motion(path: Path) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in Motion log: {path}")
    rows: dict[int, dict[str, int]] = {}
    summaries: list[dict[str, int]] = []
    for line in text.splitlines():
        if match := MOTION_ROW_RE.fullmatch(line):
            item = {key: int(value) for key, value in match.groupdict().items()}
            index = item.pop("row")
            if index in rows:
                raise ValueError(f"duplicate Motion row {index}")
            rows[index] = item
        if match := MOTION_SUM_RE.fullmatch(line):
            summaries.append(
                {key: int(value) for key, value in match.groupdict().items()}
            )
    if sorted(rows) != list(range(FROZEN_MOTION["rows"])) or len(summaries) != 1:
        raise ValueError("Motion fair population is incomplete")
    summary = summaries[0]
    for key, expected in FROZEN_MOTION.items():
        if summary.get(key) != expected:
            raise ValueError(f"Motion frozen receipt differs for {key}")
    if sum(item["fixed"] for item in rows.values()) != summary["fixed"]:
        raise ValueError("Motion fixed row sum differs")
    if sum(item["rqtb"] for item in rows.values()) != summary["rqtb"]:
        raise ValueError("Motion RQTB row sum differs")
    if sum(item["fslots"] for item in rows.values()) != summary["fslots"]:
        raise ValueError("Motion fixed slot row sum differs")
    if sum(item["rslots"] for item in rows.values()) != summary["rslots"]:
        raise ValueError("Motion RQTB slot row sum differs")
    if sum(item["equal"] for item in rows.values()) != summary["fequal"]:
        raise ValueError("Motion equal row sum differs")
    return rows, summary


def parse_local(path: Path) -> dict[int, dict[str, int]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in Local5 log: {path}")
    rows: dict[int, dict[str, int]] = {}
    passes: list[int] = []
    for line in text.splitlines():
        if match := LOCAL_GROUP_RE.fullmatch(line):
            item = {key: int(value) for key, value in match.groupdict().items()}
            index = item.pop("group")
            if index in rows:
                raise ValueError(f"duplicate Local5 group {index}")
            rows[index] = item
        if match := LOCAL_PASS_RE.fullmatch(line):
            passes.append(int(match.group("cycles")))
    if sorted(rows) != list(range(100)):
        raise ValueError("Local5 population is incomplete")
    total = sum(item["cycles"] for item in rows.values())
    if passes != [total]:
        raise ValueError("Local5 PASS receipt differs from group ledger")
    return rows


def load_local_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shape = manifest.get("shape", {})
    selection = manifest.get("selection", {})
    rows = selection.get("rows", [])
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or shape.get("sources") != 450
        or shape.get("head_dim") != 32
        or shape.get("out_dim") != 32
        or selection.get("groups") != 100
        or len(rows) != 100
        or manifest.get("weight_mode")
        != "checkpoint_theta_folded_dyadic_int8_head_slice"
    ):
        raise ValueError("Local5 OUT32 manifest contract differs")
    stage_counts = {
        stage: sum(int(row["stage"]) == stage for row in rows)
        for stage in STAGE_BLOCKS
    }
    declared = {int(key): int(value) for key, value in selection["stage_counts"].items()}
    if stage_counts != declared or any(value == 0 for value in stage_counts.values()):
        raise ValueError("Local5 stage population differs")
    return manifest


def build_motion(rows: dict[int, dict[str, int]]) -> dict[str, Any]:
    per_stage: dict[str, Any] = {}
    fixed_total = 0
    rqtb_total = 0
    for stage, (first, stop) in MOTION_ROW_RANGES.items():
        expected_rows = STAGE_BLOCKS[stage] * STAGE_HEADS[stage]
        if stop - first != expected_rows:
            raise ValueError("Motion row-to-stage map differs")
        fixed_window = sum(rows[index]["fixed"] for index in range(first, stop))
        rqtb_window = sum(rows[index]["rqtb"] for index in range(first, stop))
        fixed_frame = fixed_window * STAGE_WINDOWS[stage]
        rqtb_frame = rqtb_window * STAGE_WINDOWS[stage]
        fixed_total += fixed_frame
        rqtb_total += rqtb_frame
        per_stage[str(stage)] = {
            "rows_in_selected_window": expected_rows,
            "windows_per_frame": STAGE_WINDOWS[stage],
            "selected_window_cycles": {"fixed2s": fixed_window, "rqtb2s": rqtb_window},
            "frame_scaled_cycles": {"fixed2s": fixed_frame, "rqtb2s": rqtb_frame},
            "component_speedup": fixed_window / rqtb_window,
        }
    return {
        "evidence": "[rtl-calibrated-model]",
        "calibration_scope": (
            "sample0 selected real T450 window; all 12 blocks and 138 head rows; "
            "stage row sums scaled by full-resolution windows per frame"
        ),
        "per_stage": per_stage,
        "frame_attention_row_model": {
            "fixed2s_cycles": fixed_total,
            "rqtb2s_cycles": rqtb_total,
            "component_speedup": fixed_total / rqtb_total,
        },
    }


def build_local(
    baseline: dict[int, dict[str, int]],
    candidate: dict[int, dict[str, int]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = manifest["selection"]["rows"]
    per_stage: dict[str, Any] = {}
    one_tile_baseline = 0.0
    one_tile_candidate = 0.0
    replay_baseline = 0.0
    replay_candidate = 0.0
    for stage in STAGE_BLOCKS:
        indexes = [index for index, row in enumerate(rows) if int(row["stage"]) == stage]
        mean_baseline = sum(baseline[index]["cycles"] for index in indexes) / len(indexes)
        mean_candidate = sum(candidate[index]["cycles"] for index in indexes) / len(indexes)
        input_head_units = (
            STAGE_BLOCKS[stage] * STAGE_HEADS[stage] * STAGE_WINDOWS[stage]
        )
        baseline_stage = mean_baseline * input_head_units
        candidate_stage = mean_candidate * input_head_units
        one_tile_baseline += baseline_stage
        one_tile_candidate += candidate_stage
        replay_baseline += baseline_stage * STAGE_HEADS[stage]
        replay_candidate += candidate_stage * STAGE_HEADS[stage]
        per_stage[str(stage)] = {
            "population_groups": len(indexes),
            "blocks": STAGE_BLOCKS[stage],
            "input_heads": STAGE_HEADS[stage],
            "windows_per_frame": STAGE_WINDOWS[stage],
            "mean_cycles_per_input_head_window_one_output_tile": {
                "t450": mean_baseline,
                "rolling": mean_candidate,
            },
            "frame_scaled_one_output_tile_cycles": {
                "t450": baseline_stage,
                "rolling": candidate_stage,
            },
            "component_speedup": mean_baseline / mean_candidate,
        }
    return {
        "evidence": "[rtl-population-to-model]",
        "calibration_scope": (
            "100 sample-disjoint stage-weighted OUT32 groups with real checkpoint INT8 "
            "weights; stage means scaled by blocks, input heads, and windows"
        ),
        "per_stage": per_stage,
        "frame_attention_one_output_tile_model": {
            "t450_cycles": one_tile_baseline,
            "rolling_cycles": one_tile_candidate,
            "component_speedup": one_tile_baseline / one_tile_candidate,
        },
        "packed_pipeline_tile_replay_model": {
            "meaning": (
                "naive schedule that replays the measured packed score-to-Acc pipeline "
                "once per stage output tile; excludes cross-head accumulation and final "
                "readout and is not a shared-front implementation"
            ),
            "t450_cycles": replay_baseline,
            "rolling_cycles": replay_candidate,
            "component_speedup": replay_baseline / replay_candidate,
        },
    }


def build_report(
    motion_log: Path,
    local_manifest_path: Path,
    local_t450_log: Path,
    local_rolling_log: Path,
) -> dict[str, Any]:
    motion_rows, motion_summary = parse_motion(motion_log)
    manifest = load_local_manifest(local_manifest_path)
    local_t450 = parse_local(local_t450_log)
    local_rolling = parse_local(local_rolling_log)
    for index in range(100):
        for field in ("score_rows", "score_service", "score_direct", "qsilent", "identk", "overlap", "active", "terms", "updates"):
            if local_t450[index][field] != local_rolling[index][field]:
                raise ValueError(f"Local5 work ledger differs at group {index} field {field}")

    motion = build_motion(motion_rows)
    local = build_local(local_t450, local_rolling, manifest)
    motion_speedup = motion["frame_attention_row_model"]["component_speedup"]
    local_one_speedup = local["frame_attention_one_output_tile_model"]["component_speedup"]
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    sensitivity = [
        {
            "optimized_fraction": fraction,
            "motion_system_speedup": amdahl(motion_speedup, fraction),
            "local5_one_output_tile_system_speedup": amdahl(local_one_speedup, fraction),
        }
        for fraction in fractions
    ]
    return {
        "schema": "dual_line_attention_frame_amdahl_v1",
        "status": "PASS",
        "motion": motion,
        "local5": local,
        "amdahl_sensitivity": sensitivity,
        "claim_boundary": [
            "these are scoped attention-cycle models, not full-frame or full-encoder RTL",
            "the models omit ATLIF/neuron update, residual, FFN, decoder, DMA, and external-memory effects",
            "Motion clones one sample0 selected window across each stage window count; activity bias in the selected window may make the model optimistic",
            "Local5 uses one aligned 32-channel output tile; the packed replay model excludes cross-head accumulation/final readout and is not measured all-output execution",
            "the Local5 packed replay diagnostic is excluded from Amdahl sensitivity and is not a paper performance estimate",
            "Amdahl fractions are sensitivity variables because a measured full-encoder operator breakdown is not yet available",
            "no DC, STA, SAIF, PTPX, signoff PPA, or energy claim is made",
            "docs/359 frozen columns are unchanged",
        ],
        "provenance": {
            "motion_log": str(motion_log.resolve()),
            "motion_log_sha256": sha256(motion_log),
            "motion_frozen_receipt": motion_summary,
            "local_manifest": str(local_manifest_path.resolve()),
            "local_manifest_sha256": sha256(local_manifest_path),
            "local_t450_log": str(local_t450_log.resolve()),
            "local_t450_log_sha256": sha256(local_t450_log),
            "local_rolling_log": str(local_rolling_log.resolve()),
            "local_rolling_log_sha256": sha256(local_rolling_log),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    motion = report["motion"]
    local = report["local5"]
    motion_frame = motion["frame_attention_row_model"]
    local_one = local["frame_attention_one_output_tile_model"]
    local_replay = local["packed_pipeline_tile_replay_model"]
    lines = [
        "# Dual-Line Scoped Attention and Amdahl Model",
        "",
        "## Component Models",
        "",
        f"- Motion `[rtl-calibrated-model]`: `{motion_frame['fixed2s_cycles']:,.0f} -> {motion_frame['rqtb2s_cycles']:,.0f} = {motion_frame['component_speedup']:.4f}x`.",
        f"- Local5 one OUT32 tile `[rtl-population-to-model]`: `{local_one['t450_cycles']:,.0f} -> {local_one['rolling_cycles']:,.0f} = {local_one['component_speedup']:.4f}x`.",
        f"- Local5 diagnostic-only packed-pipeline replay: `{local_replay['t450_cycles']:,.0f} -> {local_replay['rolling_cycles']:,.0f} = {local_replay['component_speedup']:.4f}x`; excluded from Amdahl and paper estimates.",
        "",
        "## Amdahl Sensitivity",
        "",
        "| Optimized fraction | Motion | Local5 one tile |",
        "|---:|---:|---:|",
    ]
    for row in report["amdahl_sensitivity"]:
        lines.append(
            f"| {row['optimized_fraction']:.0%} | {row['motion_system_speedup']:.4f}x | "
            f"{row['local5_one_output_tile_system_speedup']:.4f}x |"
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-log", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--local-t450-log", type=Path, required=True)
    parser.add_argument("--local-rolling-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.motion_log,
        args.local_manifest,
        args.local_t450_log,
        args.local_rolling_log,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_markdown(args.output_dir / "report.md", report)
    print(
        "PASS dual-line attention/Amdahl model "
        f"motion={report['motion']['frame_attention_row_model']['component_speedup']:.4f}x "
        f"local5={report['local5']['frame_attention_one_output_tile_model']['component_speedup']:.4f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

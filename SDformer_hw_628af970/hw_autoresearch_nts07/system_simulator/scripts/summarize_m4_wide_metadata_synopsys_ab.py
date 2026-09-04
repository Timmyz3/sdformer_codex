#!/usr/bin/env python3
"""Fail-closed Synopsys DC/Formality A/B for M4 wide metadata sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(pattern: str, text: str, *, integer: bool = False) -> int | float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing Synopsys report field: {pattern}")
    return int(match.group(1)) if integer else float(match.group(1))


def minimum_slack(path: Path) -> float:
    values = [
        float(value)
        for value in re.findall(
            r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9.]+)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
    ]
    if not values:
        raise ValueError(f"no timing slack in {path}")
    return min(values)


def parse_parameters(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"malformed elaboration parameter: {item!r}")
        name, setting = (part.strip() for part in item.split("=", 1))
        if not name or not setting or name in result:
            raise ValueError(f"invalid elaboration parameter: {item!r}")
        result[name] = setting
    return result


def parse_shell_report(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    instances = [line for line in lines if "=" not in line and line.strip()]
    if fields.get("scope") != "PREMACRO_LOGIC_ONLY":
        raise ValueError("logical SRAM shell scope is not PREMACRO_LOGIC_ONLY")
    if fields.get("paper_ppa_ready") != "false":
        raise ValueError("logical SRAM shell report overclaims paper readiness")
    expected = int(fields.get("expected_count", "-1"))
    observed = int(fields.get("observed_count", "-1"))
    if expected != 6 or observed != expected or len(instances) != expected:
        raise ValueError("logical SRAM shell population is not exactly six")
    return {
        "scope": fields["scope"],
        "paper_ppa_ready": False,
        "logical_shell_count": observed,
        "instances": instances,
    }


def load_point(run_dir: Path, expected_shared: int) -> dict[str, Any]:
    required = (
        "dc_run_manifest.json",
        "formality_run_manifest.json",
        "reports/area.rpt",
        "reports/qor.rpt",
        "reports/timing_setup.rpt",
        "reports/timing_hold.rpt",
        "reports/constraint_violators.rpt",
        "reports/premacro_logical_shells.rpt",
        "reports/formality_status.txt",
        "reports/formality_unmatched.rpt",
        "reports/formality_verify.rpt",
        "netlist/qfit_dual_line_descriptor_stateful_engine_mapped.v",
    )
    paths = {name: run_dir / name for name in required}
    missing = [name for name, path in paths.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing Synopsys A/B evidence in {run_dir}: {missing}")
    dc_manifest = json.loads(paths["dc_run_manifest.json"].read_text())
    fm_manifest = json.loads(paths["formality_run_manifest.json"].read_text())
    parameters = parse_parameters(dc_manifest.get("elab_parameters", ""))
    expected_parameters = {
        "STATE_QUEUE_DEPTH": "1",
        "USE_SHARED_WIDE_METADATA": str(expected_shared),
    }
    if parameters != expected_parameters:
        raise ValueError(f"unexpected elaboration parameters: {parameters}")
    if fm_manifest.get("elab_parameters") != dc_manifest.get("elab_parameters"):
        raise ValueError("DC/Formality elaboration parameters differ")
    if dc_manifest.get("ppa_admission") != "0" or dc_manifest.get("macro_dbs"):
        raise ValueError("pre-macro A/B unexpectedly claims macro/PPA admission")
    if paths["reports/formality_status.txt"].read_text().strip() != "PASS":
        raise ValueError("Formality status is not PASS")
    area_text = paths["reports/area.rpt"].read_text(encoding="utf-8", errors="replace")
    metrics = {
        "total_cell_area_um2": number(r"Total cell area:\s*([0-9.]+)", area_text),
        "sequential_cells": number(
            r"Number of sequential cells:\s*([0-9]+)", area_text, integer=True
        ),
        "macro_or_black_box_count": number(
            r"Number of macros/black boxes:\s*([0-9]+)", area_text, integer=True
        ),
        "dc_setup_slack_ns": minimum_slack(paths["reports/timing_setup.rpt"]),
        "dc_hold_slack_ns": minimum_slack(paths["reports/timing_hold.rpt"]),
    }
    if metrics["dc_setup_slack_ns"] < 0 or metrics["dc_hold_slack_ns"] < 0:
        raise ValueError("DC A/B point has negative timing slack")
    if re.search(r"VIOLATED", paths["reports/constraint_violators.rpt"].read_text()):
        raise ValueError("DC A/B point has a reported constraint violation")
    evidence = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "elab_parameters": parameters,
        "identity": {
            "design_name": dc_manifest.get("design_name"),
            "operating_condition": dc_manifest.get("operating_condition"),
            "clock_period_ns": dc_manifest.get("clock_period_ns"),
            "dc_hold_uncertainty_ns": dc_manifest.get("dc_hold_uncertainty_ns"),
            "dc_hold_report_uncertainty_ns": dc_manifest.get(
                "dc_hold_report_uncertainty_ns"
            ),
            "paths": dc_manifest.get("paths"),
        },
        "logical_sram_shells": parse_shell_report(
            paths["reports/premacro_logical_shells.rpt"]
        ),
        "metrics": metrics,
        "evidence": evidence,
    }


def summarize(shared: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    if shared["identity"] != legacy["identity"]:
        raise ValueError("shared/legacy Synopsys identities or constraints differ")
    shared_metrics = shared["metrics"]
    legacy_metrics = legacy["metrics"]
    legacy_area = float(legacy_metrics["total_cell_area_um2"])
    legacy_seq = int(legacy_metrics["sequential_cells"])
    area_saving = legacy_area - float(shared_metrics["total_cell_area_um2"])
    seq_saving = legacy_seq - int(shared_metrics["sequential_cells"])
    if area_saving <= 0 or seq_saving <= 0:
        raise ValueError("shared metadata did not reduce pre-macro logic")
    return {
        "status": "PASS_PREMACRO_LOGIC_ONLY_DC_FORMALITY_AB",
        "paper_ppa_admitted": False,
        "shared": shared,
        "legacy": legacy,
        "comparison": {
            "cell_area_reduction_um2": area_saving,
            "cell_area_reduction_fraction": area_saving / legacy_area,
            "sequential_cell_reduction": seq_saving,
            "sequential_cell_reduction_fraction": seq_saving / legacy_seq,
        },
        "claim_boundary": (
            "Same-top, same-constraint Synopsys pre-macro logic A/B with six "
            "zero-area/timing logical SRAM shells and separate RTL-to-gate "
            "Formality proofs. This is not SRAM-macro, post-route, power, or "
            "paper-PPA evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "m4_wide_metadata_synopsys_ab_v1",
        **summarize(load_point(args.shared, 1), load_point(args.legacy, 0)),
        "summarizer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS M4 wide metadata Synopsys A/B -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

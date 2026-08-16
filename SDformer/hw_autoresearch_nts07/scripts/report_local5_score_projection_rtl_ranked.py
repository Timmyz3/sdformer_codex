#!/usr/bin/env python3
"""Seal ep44 Local5 raw-Q/K-to-Acc32 RTL without legacy post-score coupling."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(
    r"GROUP backend=(?P<backend>\d+) latency=(?P<latency>\d+) "
    r"group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct_rows>\d+) "
    r"(?:qsilent_rows=\d+ identk_rows=\d+ overlap=\d+ )?"
    r"active=(?P<active>\d+) memory_wait=(?P<memory_wait>\d+) "
    r"terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_rows(path: Path) -> list[dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if "PASS Local5 score-to-projection" not in text:
        raise ValueError(f"missing PASS: {path}")
    rows = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in GROUP_RE.finditer(text)
    ]
    if not rows:
        raise ValueError(f"missing GROUP records: {path}")
    return rows


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "total": float(array.sum()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def read_lines(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines()]


def tool_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    manifest_path = args.vector_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "local5_score_projection_vectors_v1":
        raise ValueError("vector manifest schema mismatch")
    if manifest.get("weight_mode") != (
        "checkpoint_theta_folded_dyadic_int8_head_slice"
    ):
        raise ValueError("ranked report requires real checkpoint INT8 weights")
    groups = int((manifest.get("selection") or {}).get("groups", 0))
    if groups != 100:
        raise ValueError("ranked report requires 100 groups")
    source_manifest = Path(str(manifest.get("source_manifest", ""))).resolve()
    if (
        not source_manifest.is_file()
        or sha256(source_manifest) != manifest.get("source_manifest_sha256")
    ):
        raise ValueError("source manifest binding failed")
    source = load_json(source_manifest)
    if (
        source.get("qualification", {}).get("qualified") is not True
        or source.get("checkpoint_sha256")
        != "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57"
    ):
        raise ValueError("ep44 qualified source contract failed")

    rows_by_key: dict[str, list[dict[str, int]]] = {}
    configurations: dict[str, dict[str, object]] = {}
    for backend, backend_id in (("tcfm5", 0), ("linear5", 1)):
        for latency in (1, 2):
            key = f"{backend}_l{latency}"
            rows = parse_rows(args.result_dir / f"{key}_verilator.log")
            if len(rows) != groups or any(
                row["backend"] != backend_id
                or row["latency"] != latency
                or row["group"] != index
                or row["score_rows"] != 450
                for index, row in enumerate(rows)
            ):
                raise ValueError(f"{key} identity or group coverage mismatch")
            rows_by_key[key] = rows
            configurations[key] = {
                field: stats([row[field] for row in rows])
                for field in (
                    "cycles",
                    "score_service",
                    "score_direct_rows",
                    "active",
                    "memory_wait",
                    "terms",
                    "updates",
                )
            }

    speedups: dict[str, object] = {}
    stage_results: dict[str, object] = {}
    selection = manifest["selection"]["rows"]
    workload_fields = (
        "score_rows",
        "score_service",
        "score_direct_rows",
        "active",
        "terms",
        "updates",
    )
    for latency in (1, 2):
        t_rows = rows_by_key[f"tcfm5_l{latency}"]
        l_rows = rows_by_key[f"linear5_l{latency}"]
        if any(
            left[field] != right[field]
            for left, right in zip(t_rows, l_rows, strict=True)
            for field in workload_fields
        ):
            raise ValueError(f"L{latency} frontend/workload fairness mismatch")
        t_total = sum(row["cycles"] for row in t_rows)
        l_total = sum(row["cycles"] for row in l_rows)
        per_group = [
            left["cycles"] / right["cycles"]
            for right, left in zip(t_rows, l_rows, strict=True)
        ]
        speedups[f"l{latency}"] = {
            "ratio_of_totals": l_total / t_total,
            "per_group": stats(per_group),
        }
        stage_results[f"l{latency}"] = {
            str(stage): {
                "groups": len(indices),
                "tcfm5_cycles": sum(t_rows[index]["cycles"] for index in indices),
                "linear5_cycles": sum(l_rows[index]["cycles"] for index in indices),
                "ratio_of_totals": (
                    sum(l_rows[index]["cycles"] for index in indices)
                    / sum(t_rows[index]["cycles"] for index in indices)
                ),
            }
            for stage in range(4)
            for indices in [[
                index
                for index, metadata in enumerate(selection)
                if int(metadata["stage"]) == stage
            ]]
        }

    expected_acc = read_lines(
        args.vector_dir / manifest["artifacts"]["expected_acc"]["file"]
    )
    actual_acc32: dict[str, object] = {}
    random_stress: dict[str, object] = {}
    for key in rows_by_key:
        actual_path = args.result_dir / f"{key}_actual_acc32.memh"
        actual = read_lines(actual_path)
        if actual != expected_acc:
            raise ValueError(f"{key} Acc32 mismatch")
        actual_acc32[key] = {
            "entries": len(actual),
            "sha256": sha256(actual_path),
            "zero_mismatch": True,
        }
        stress_path = args.result_dir / f"{key}_random_stress_verilator.log"
        stress_rows = parse_rows(stress_path)
        if len(stress_rows) != 8:
            raise ValueError(f"{key} random stress group mismatch")
        random_stress[key] = {
            "groups": 8,
            "random_input_gaps": True,
            "random_read_gaps": True,
            "log_sha256": sha256(stress_path),
        }

    source_bindings = []
    source_dir = args.result_dir / "source_ranked"
    source_dir.mkdir(exist_ok=True)
    for source_path in args.source:
        target = source_dir / source_path.name
        shutil.copy2(source_path, target)
        source_bindings.append(
            {"path": str(source_path.resolve()), "sha256": sha256(source_path)}
        )

    report = {
        "schema": "local5_ranked_score_projection_rtl_report_v1",
        "status": "PASS",
        "evidence": "[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]",
        "scope": (
            "ep44 raw Q/K through alpha-XNOR Q7, masked Shiftmax5 Q1.7, "
            "inverse-stencil relation build, source-major terms and Acc32"
        ),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "groups": groups,
        "score_gate_row_checks": groups * 450 * 4,
        "score_gate_scalar_checks": groups * 450 * 5 * 2 * 4,
        "acc32_checks": len(expected_acc) * 4,
        "configurations": configurations,
        "speedups": speedups,
        "stage_results": stage_results,
        "actual_acc32": actual_acc32,
        "random_stress": random_stress,
        "vector_manifest": str(manifest_path.resolve()),
        "vector_manifest_sha256": sha256(manifest_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_bindings": source_bindings,
        "fairness": [
            "same ep44 raw Q/K, score leaf, Shiftmax5, relation frontier, term builder, INT8 weights and five Acc banks",
            "only destination-to-bank mapping and exact conflict replay differ",
        ],
        "limits": [
            "100 outcome-independent qualified T450 groups; not every deployment window",
            "OUT_DIM=2 real output channels per group",
            "pre-bias/pre-BN/pre-requant/pre-residual and not cross-head/full-encoder output",
            "no foundry PPA or SRAM macro signoff",
        ],
        "execution_receipt": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "iverilog": tool_version(["iverilog", "-V"]),
            "verilator": tool_version(["verilator", "--version"]),
            "yosys": tool_version(["yosys", "-V"]),
        },
    }
    report_path = args.result_dir / "report_ranked.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    complete = {
        "schema": "local5_ranked_score_projection_rtl_complete_v1",
        "status": "SEALED",
        "report": report_path.name,
        "report_sha256": sha256(report_path),
        "vector_manifest_sha256": sha256(manifest_path),
        "files": {
            path.name: sha256(path)
            for path in sorted(args.result_dir.iterdir())
            if path.is_file()
        },
    }
    (args.result_dir / "complete_ranked.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "SEALED",
                "l1_speedup": speedups["l1"]["ratio_of_totals"],
                "l2_speedup": speedups["l2"]["ratio_of_totals"],
                "acc32_checks": report["acc32_checks"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
